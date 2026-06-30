#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from gym import spaces

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface
from habitat.tasks.nav.nav import PointGoalSensor, Success
from hydra.core.config_store import ConfigStore
import habitat_sim
from habitat.tasks.rearrange.rearrange_sensors import NumStepsMeasure
from dataclasses import dataclass
from habitat.config.default_structured_configs import MeasurementConfig

from habitat.tasks.rearrange.utils import rearrange_collision
from habitat.core.embodied_task import Measure
from habitat.tasks.rearrange.social_nav.utils import (
    robot_human_vec_dot_product,
)
from habitat.tasks.nav.nav import DistanceToGoalReward, DistanceToGoal
from habitat.tasks.rearrange.utils import coll_name_matches
try:
    import magnum as mn
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class DidMultiAgentsCollide(Measure):
    """
    Detects if the multi-agent ( more than 1 humanoids agents) in the scene 
    are colliding with each other at the current step. 
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "did_multi_agents_collide"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(
            *args,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        sim = task._sim
        human_num = task._human_num
        sim.perform_discrete_collision_detection()
        contact_points = sim.get_physics_contact_points()
        found_contact = False

        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in sim.agents_mgr.articulated_agents_iter
        ]
        main_agent_id = agent_ids[0]
        other_agent_ids = set(agent_ids[1:human_num+1])  
        for cp in contact_points:
            if coll_name_matches(cp, main_agent_id):
                if any(coll_name_matches(cp, agent_id) for agent_id in other_agent_ids):
                    found_contact = True
                    break  

        self._metric = found_contact

@registry.register_measure
class HumanCollision(Measure):

    cls_uuid: str = "human_collision"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._ever_collide = False
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DidMultiAgentsCollide._get_uuid()]
        )
        self._metric = 0.0
        self._ever_collide = False

    def update_metric(self, *args, episode, task, observations, **kwargs):
        collid = task.measurements.measures[DidMultiAgentsCollide._get_uuid()].get_metric()
        if collid or self._ever_collide:
            self._metric = 1.0
            self._ever_collide = True
            task.should_end = True
        else:
            self._metric = 0.0

@registry.register_measure
class STL(Measure):
    r"""Success weighted by Completion Time
    """
    cls_uuid: str = "stl"
    
    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid, NumStepsMeasure.cls_uuid]
        )

        self._num_steps_taken = 0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, observations=observations, *args, **kwargs)

    def update_metric(self, *args, episode, task, observations, **kwargs):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric() 
        self._num_steps_taken = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()

        oracle_time = (
            self._start_end_episode_distance / (0.25 / 10)
        )
        oracle_time = max(oracle_time, 1e-6)
        agent_time = max(self._num_steps_taken, 1e-6)
        self._metric = ep_success * (oracle_time / max(oracle_time, agent_time))

@registry.register_measure
class PersonalSpaceCompliance(Measure):

    cls_uuid: str = "psc"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._use_geo_distance = config.use_geo_distance
        super().__init__()
        
    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [NumStepsMeasure.cls_uuid]
        )
        self._compliant_steps = 0
        self._num_steps = 0

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        if self._human_nums == 0:
            self._metric = 1.0
        else:
            robot_pos = self._sim.get_agent_state(0).position
            self._num_steps = task.measurements.measures[NumStepsMeasure.cls_uuid].get_metric()
            compliance = True
            for i in range(self._human_nums):
                human_position = self._sim.get_agent_state(i+1).position

                if self._use_geo_distance:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = robot_pos
                    path.requested_end = human_position
                    found_path = self._sim.pathfinder.find_path(path)

                    if found_path:
                        distance = self._sim.geodesic_distance(robot_pos, human_position)
                    else:
                        distance = np.linalg.norm(human_position - robot_pos, ord=2)
                else:
                    distance = np.linalg.norm(human_position - robot_pos, ord=2)

                if distance < 1.0:
                    compliance = False
                    break                    

            if compliance:
                self._compliant_steps += 1
            self._metric = (self._compliant_steps / self._num_steps)

@registry.register_measure
class MultiAgentNavReward(Measure):
    """
    Reward that gives a continuous reward for the social navigation task.

    奖励由两部分组成，通过 use_social_eq_reward 完全解耦：

    [原有组件] ─────────────────────────────── 始终生效
      R1.  1.5 × distance_to_goal_reward
      R2.  close_to_human_penalty         （行人<2m 且离目标>2m 时生效）
      R3.  collide_human_penalty
      R4.  collide_scene_penalty
      R5.  trajectory_cover_penalty        （行人未来轨迹重叠，离目标>2m 时生效）

    Social EQ 组件参数（默认值，可通过 yaml 覆盖）：
      S1. disperse_reward_coef = 0.1
      S2. pause_reward = 0.001
      S3. backward_reward = 0.001
      S4. social_efficiency_bonus = 0.02
      S5. max_consecutive_social_actions = 5, over_wait_penalty = -0.01
      S6. action_smoothing_penalty_coef = -0.005
      S7. high_freq_switch_threshold = 3, high_freq_switch_penalty_coef = -0.005
      S8. detour_zone_inner = 1.5, detour_zone_outer = 4.0, detour_reward_coef = 0.05
    """

    cls_uuid: str = "multi_agent_nav_reward"

    # 6 动作空间常量
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PAUSE = 4
    BACKWARD = 5

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metric = 0.0
        config = kwargs["config"]
        self._config = config
        self._sim = kwargs["sim"]

        # ── 原有组件参数 ──
        self._use_geo_distance = config.use_geo_distance
        self._allow_distance = config.allow_distance
        self._collide_scene_penalty = config.collide_scene_penalty
        self._collide_human_penalty = config.collide_human_penalty
        self._trajectory_cover_penalty = config.trajectory_cover_penalty
        self._threshold_squared = config.cover_future_dis_thre ** 2
        self._robot_idx = config.robot_idx
        self._close_to_human_penalty = config.close_to_human_penalty
        self._facing_human_dis = config.facing_human_dis

        # ── Social EQ 参数 ──
        self._use_social_eq = getattr(config, "use_social_eq_reward", False)

        # S1: 动态避让奖励（robot 远离行人→正，逼近→负）
        self._disperse_reward_coef = getattr(config, "disperse_reward_coef", 0.1)
        # S2: 礼让动作奖励（极小）
        self._pause_reward_val = getattr(config, "pause_reward", 0.001)
        self._backward_reward_val = getattr(config, "backward_reward", 0.001)
        # S3: 社交效率奖励（让步有效后恢复前进）
        self._social_efficiency_bonus = getattr(config, "social_efficiency_bonus", 0.02)
        # S4: 过度等待惩罚（连续礼让超过阈值）
        self._max_consecutive_social = getattr(config, "max_consecutive_social_actions", 5)
        self._over_wait_penalty = getattr(config, "over_wait_penalty", -0.01)
        # S5: 动作平滑惩罚
        self._smoothing_penalty_coef = getattr(config, "action_smoothing_penalty_coef", -0.005)
        # S6: 高频切换惩罚
        self._hfs_threshold = getattr(config, "high_freq_switch_threshold", 3)
        self._hfs_penalty_coef = getattr(config, "high_freq_switch_penalty_coef", -0.005)
        # S7: 提前避让奖励（新增）—— 行人在中距离时主动绕行/减速
        self._detour_zone_inner = getattr(config, "detour_zone_inner", 1.5)   # 内圈：必须让行
        self._detour_zone_outer = getattr(config, "detour_zone_outer", 4.0)   # 外圈：开始绕行
        self._detour_reward_coef = getattr(config, "detour_reward_coef", 0.05)

        self._human_nums = 0

        # ── Social EQ Episode 状态 ──
        self._state = {}

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        if "human_num" in episode.info:
            self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        else:
            self._human_nums = 0
        self._metric = 0.0

        # 重置 social_eq 状态
        self._state = {
            "prev_ped_distance": None,
            "is_yielding": False,
            "pedestrian_passed_during_yield": False,
            "consecutive_social_count": 0,
            "prev_moving_action": None,
            "high_freq_switch_count": 0,
            "prev_action": None,
            # 提前避让状态
            "in_detour_zone": False,
            "ped_in_range_before": False,
        }

    def _check_human_facing_robot(self, human_pos, robot_pos, human_idx):
        base_T = self._sim.get_agent_data(
            human_idx
        ).articulated_agent.sim_obj.transformation
        facing = (
            robot_human_vec_dot_product(human_pos, robot_pos, base_T)
            > self._config.human_face_robot_threshold
        )
        return facing

    # ── Social EQ 辅助方法 ──
    def _is_turning(self, action):
        return action in (self.TURN_LEFT, self.TURN_RIGHT)

    def _is_forward(self, action):
        return action == self.FORWARD

    def _is_backward(self, action):
        return action == self.BACKWARD

    def _is_pause(self, action):
        return action == self.PAUSE

    def _is_social(self, action):
        return action in (self.PAUSE, self.BACKWARD)

    def _is_moving(self, action):
        return action in (self.FORWARD, self.BACKWARD)

    def _get_nearest_ped_distance(self, robot_pos, observations):
        """获取最近的行人欧几里得距离（无 geodesic 开销）"""
        min_dist = None
        for i in range(1, 7):
            key = f"agent_{i}_localization_sensor"
            if key in observations:
                human_pos = observations[key]
                if human_pos is not None and len(human_pos) >= 3:
                    d = np.linalg.norm(np.array(human_pos[:3]) - robot_pos)
                    if min_dist is None or d < min_dist:
                        min_dist = d
        return min_dist

    def _compute_social_eq_reward(self, action, robot_pos, observations):
        """
        计算 Social EQ 奖励组件（完全独立于原有组件）
        仅在 use_social_eq_reward: true 时调用。

        设计原则：
          - 礼让奖励（pause/backward）鼓励让行人先行
          - 效率奖励（social_efficiency_bonus）要求行人实际通过
          - 过度等待惩罚防止无限等待
          - 动作平滑惩罚减少无意义抖动
          - 高频切换惩罚防止 FORWARD↔BACKWARD 乒乓
          - 动态避让奖励机器人主动拉开与行人的距离
          - 【移除】角速度惩罚：改由动作平滑惩罚间接约束
        """
        reward = 0.0

        # ── 0. 行人距离 ──
        curr_ped_dist = self._get_nearest_ped_distance(robot_pos, observations)
        prev_ped_dist = self._state.get("prev_ped_distance")

        # ── 1. 动态避让奖励（替代原来的被动 safe_distance_reward）──
        # 奖励机器人主动远离行人，惩罚机器人逼近行人
        if curr_ped_dist is not None and prev_ped_dist is not None:
            dist_change = curr_ped_dist - prev_ped_dist  # 正=远离, 负=逼近
            reward += self._disperse_reward_coef * dist_change

        if curr_ped_dist is not None:
            self._state["prev_ped_distance"] = curr_ped_dist

        # ── 2. 社交礼让奖励 ──
        if self._is_pause(action):
            reward += self._pause_reward_val
        elif self._is_backward(action):
            reward += self._backward_reward_val

        # ── 3. 行人通过检测 ──
        if curr_ped_dist is not None and prev_ped_dist is not None:
            # 行人距离增加超过 0.3m → 行人在移动中远离
            if curr_ped_dist > prev_ped_dist + 0.3:
                self._state["pedestrian_passed_during_yield"] = True

        # ── 4. 社交效率奖励 ──
        if self._is_forward(action):
            if (self._state.get("is_yielding") and
                    self._state.get("pedestrian_passed_during_yield")):
                reward += self._social_efficiency_bonus
                self._state["is_yielding"] = False
                self._state["pedestrian_passed_during_yield"] = False
        elif self._is_social(action):
            self._state["is_yielding"] = True
            self._state["pedestrian_passed_during_yield"] = False

        # ── 5. 过度等待惩罚 ──
        if self._is_social(action):
            self._state["consecutive_social_count"] += 1
        else:
            self._state["consecutive_social_count"] = 0

        if self._state["consecutive_social_count"] > self._max_consecutive_social:
            reward += self._over_wait_penalty

        # ── 6. 动作平滑惩罚 ──
        # 有效让步恢复（行人通过后恢复前进）不惩罚
        if self._is_forward(action) and self._state.get("is_yielding") and self._state.get("pedestrian_passed_during_yield"):
            pass  # skip smoothing penalty
        elif self._is_moving(action):
            prev_mv = self._state.get("prev_moving_action")
            if prev_mv is not None and prev_mv != action:
                reward += self._smoothing_penalty_coef
            self._state["prev_moving_action"] = action

        # ── 7. 高频切换惩罚 ──
        # 有效让步恢复不计入切换
        if self._is_forward(action) and self._state.get("is_yielding") and self._state.get("pedestrian_passed_during_yield"):
            self._state["high_freq_switch_count"] = 0
        else:
            prev_a = self._state.get("prev_action")
            is_switch = False
            if prev_a is not None:
                if self._is_forward(action) and self._is_backward(prev_a):
                    is_switch = True
                elif self._is_backward(action) and self._is_forward(prev_a):
                    is_switch = True

            if is_switch:
                self._state["high_freq_switch_count"] += 1
                if self._state["high_freq_switch_count"] >= self._hfs_threshold:
                    excess = self._state["high_freq_switch_count"] - self._hfs_threshold + 1
                    reward += self._hfs_penalty_coef * excess
            elif not self._is_moving(action):
                self._state["high_freq_switch_count"] = 0

        # ── 8. 提前避让奖励（S7）────────────────────────────────────
        # 行人进入外圈（4m）但不在内圈（1.5m）时：
        #   - 机器人在转向 → 主动绕行，奖励
        #   - 机器人在直行 → 冲向行人，惩罚
        if curr_ped_dist is not None:
            ped_in_range = (self._detour_zone_inner < curr_ped_dist < self._detour_zone_outer)
            was_in_range = self._state.get("ped_in_in_range", False)

            if ped_in_range:
                if self._is_forward(action) and self._is_turning(action):
                    # 转向避让：正奖励
                    reward += self._detour_reward_coef
                    self._state["in_detour_zone"] = True
                elif self._is_forward(action) and not self._is_turning(action):
                    # 直行冲向：负惩罚
                    reward -= self._detour_reward_coef
                    self._state["in_detour_zone"] = True
                # pause/backward 在此区域已有 S2 奖励，无需额外处理
            elif not ped_in_range and was_in_range:
                # 行人离开绕行区域，重置
                self._state["in_detour_zone"] = False

            self._state["ped_in_in_range"] = ped_in_range

        self._state["prev_action"] = action

        return reward

    def update_metric(self, *args, episode, task, observations, **kwargs):

        # 获取当前动作（task._action 存储的是上一步执行的动作）
        action = getattr(task, "_action", None)
        if action is None:
            action = observations.get("prev_action", self.FORWARD)
        if hasattr(action, "item"):
            action = action.item()
        action = int(action) if action is not None else self.FORWARD

        # Start social nav reward
        social_nav_reward = 0.0

        # Component 1: Goal distance reward (strengthened by multiplying by 1.5)
        distance_to_goal_reward = task.measurements.measures[
            DistanceToGoalReward.cls_uuid
        ].get_metric()
        social_nav_reward += 1.5 * distance_to_goal_reward

        # Component 2: Penalize being too close to humans
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        use_k_robot = f"agent_{self._robot_idx}_localization_sensor"
        robot_pos = np.array(observations[use_k_robot][:3])

        if distance_to_target > self._allow_distance:
            for i in range(self._human_nums):
                use_k_human = f"agent_{i+1}_localization_sensor"
                human_position = observations[use_k_human][:3]

                if self._use_geo_distance:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = robot_pos
                    path.requested_end = human_position
                    found_path = self._sim.pathfinder.find_path(path)
                    if found_path:
                        distance = self._sim.geodesic_distance(robot_pos, human_position)
                    else:
                        distance = np.linalg.norm(human_position - robot_pos, ord=2)
                else:
                    distance = np.linalg.norm(human_position - robot_pos, ord=2)

                if distance < self._facing_human_dis:
                    penalty = self._close_to_human_penalty * np.exp(-distance / self._facing_human_dis)
                    social_nav_reward += penalty

        # Component 3: Collision detection for two agents
        did_agents_collide = task.measurements.measures[
            DidMultiAgentsCollide._get_uuid()
        ].get_metric()
        if did_agents_collide:
            task.should_end = True
            social_nav_reward += self._collide_human_penalty

        # Component 4: Collision detection for the main agent and the scene
        did_rearrange_collide, collision_detail = rearrange_collision(
            self._sim, True, ignore_base=False, agent_idx=self._robot_idx
        )
        if did_rearrange_collide:
            social_nav_reward += self._collide_scene_penalty

        # Component 5: Trajectory overlap penalty with time-based weighting
        if distance_to_target > self._allow_distance and "human_future_trajectory" in task.measurements.measures:
            human_future_trajectory_temp = task.measurements.measures['human_future_trajectory']._metric
            for trajectory in human_future_trajectory_temp.values():
                for t, point in enumerate(trajectory):
                    time_weight = 1.0 / (1 + t)
                    if np.sum((robot_pos - point) ** 2) < self._threshold_squared:
                        social_nav_reward += self._trajectory_cover_penalty * time_weight
                        break

        # ── Component 6-13: Social EQ（仅在 use_social_eq_reward: true 时叠加）──
        if self._use_social_eq:
            social_eq_reward = self._compute_social_eq_reward(action, robot_pos, observations)
            social_nav_reward += social_eq_reward

        self._metric = social_nav_reward

@registry.register_measure
class HumanVelocityMeasure(UsesArticulatedAgentInterface, Measure):
    """
    The measure for ORCA
    """

    cls_uuid: str = "human_velocity_measure"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.human_num = kwargs['task']._human_num
        self.velo_coff = np.array([[0, 1]] * 6)
        self.velo_base = np.array([[0.25, np.deg2rad(10)]] * 6)
        
        super().__init__(*args, sim=sim, **kwargs)
        self._metric = self.velo_base * self.velo_coff 

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanVelocityMeasure.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.human_num = task._human_num
        self.velo_coff = np.array([[0.0, 0.0]] * 6)
        self.velo_base = np.array([[0.25, np.deg2rad(10)]] * 6)
        self._metric = self.velo_base * self.velo_coff 

    def update_metric(self, *args, episode, task, observations, **kwargs):
        self._metric = self.velo_base * self.velo_coff 

def merge_paths(paths):
    merged_path = []
    for i, path in enumerate(paths):
        if i > 0:
            path = path[1:]
        merged_path.extend(path)
    return merged_path


@registry.register_measure
class HumanFutureTrajectory(UsesArticulatedAgentInterface, Measure):
    """
    The measure for future prediction of social crowd navigation
    """

    cls_uuid: str = "human_future_trajectory"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.num_agents = sim.num_articulated_agents
        self.target_dict = [[[0, 0, 0]] for _ in range(self.num_agents-1)]
        self.path_dict = {}
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanFutureTrajectory.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def _path_to_point(self, point_a,point_b):

        path = habitat_sim.ShortestPath()
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        if not found_path:
            return [point_a, point_b]
        return path.points

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent_idx, target in enumerate(self.target_dict):
            path = []
            
            agent_pos = self._sim.get_agent_data(agent_idx+1).articulated_agent.base_pos
            for i in range(-1,len(target)):
                if i == -1:
                    path_point = np.array(agent_pos)
                else:
                    path_point = target[i]

                if i >= 0:
                    temp_path = self._path_to_point(prev_point, path_point)
                    path.append(temp_path)
                
                prev_point = path_point

            if path == []:
                self.path_dict[agent_idx + 1] = []
            else:
                temp_merged_path = merge_paths(path)
                output_length = min(5, len(temp_merged_path))
                self.path_dict[agent_idx + 1] = temp_merged_path[:output_length]

        self._metric = self.path_dict

@registry.register_measure
class HumanFutureTrajectory(UsesArticulatedAgentInterface, Measure):
    """
    The measure for future prediction of social crowd navigation.
    """

    cls_uuid: str = "human_future_trajectory"

    def __init__(self, *args, sim, **kwargs):
        self._sim = sim
        self.human_num = kwargs['task']._human_num
        self.output_length = 5
        self.target_dict = self._initialize_target_dict(self.human_num)
        self.path_dict = {}
        super().__init__(*args, sim=sim, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return HumanFutureTrajectory.cls_uuid

    def _initialize_target_dict(self, human_num):
        """Initialize the target dictionary with default values."""
        return np.full((human_num, 2, 3), -100, dtype=np.float32).tolist()

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.human_num = task._human_num
        self.target_dict = self._initialize_target_dict(self.human_num)
        self.path_dict = {}
        self._metric = {}

    def _path_to_point(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points if found_path else [point_a, point_b]

    def _process_path(self, path):
        """Process the path by merging and padding/truncating to the desired length."""
        temp_merged_path = merge_paths(path)
        
        if len(temp_merged_path) < self.output_length:
            padding = np.full((self.output_length - len(temp_merged_path), 3), temp_merged_path[-1], dtype=np.float32)
            temp_merged_path = np.concatenate([temp_merged_path, padding], axis=0)
        else:
            temp_merged_path = np.array(temp_merged_path[:self.output_length], dtype=np.float32)
        
        return temp_merged_path.tolist()

    def update_metric(self, *args, episode, task, observations, **kwargs):
        for agent_idx, target in enumerate(self.target_dict):
            path = []
            agent_pos = np.array(self._sim.get_agent_data(agent_idx + 1).articulated_agent.base_pos)

            prev_point = agent_pos
            for i in range(len(target)):
                path_point = np.array(target[i])
                temp_path = self._path_to_point(prev_point, path_point)
                path.append(temp_path)
                prev_point = path_point

            self.path_dict[agent_idx + 1] = self._process_path(path)
            
        self._metric = self.path_dict

@dataclass
class MultiAgentNavReward(MeasurementConfig):
    r"""
    The reward for the multi agent navigation tasks.

    [Social EQ 组件] ─── use_social_eq_reward: true 时叠加（默认关闭）
      S1. disperse_reward_coef          （主动避让奖励）
      S2. pause_reward                  （执行 PAUSE 动作）
      S3. backward_reward               （执行 BACKWARD 动作）
      S4. social_efficiency_bonus        （让步有效后恢复前进）
      S5. max_consecutive_social_actions, over_wait_penalty  （过度等待惩罚）
      S6. action_smoothing_penalty_coef  （F↔B 无效切换惩罚）
      S7. high_freq_switch_threshold, high_freq_switch_penalty_coef
    """
    type: str = "MultiAgentNavReward"

    # If we want to use geo distance to measure the distance
    # between the robot and the human
    use_geo_distance: bool = True
    # discomfort for multi agents
    allow_distance: float = 0.5
    collide_scene_penalty: float = -0.25
    collide_human_penalty: float = -0.5
    facing_human_dis: float = 1.0
    human_face_robot_threshold: float = 0.5
    close_to_human_penalty: float = -0.025
    trajectory_cover_penalty: float = -0.025
    cover_future_dis_thre: float = -0.05
    # Set the id of the agent
    robot_idx: int = 0

    # ── Social EQ 组件（默认关闭）─────────────────────────────
    use_social_eq_reward: bool = False
    # S1: 主动避让奖励（robot 远离行人→正，逼近行人→负）
    # 尺度参考：base 的 close_to_human_penalty 为 -0.003/步
    disperse_reward_coef: float = 0.1
    # S2: 礼让动作奖励（极小，base 无对应奖励，只是行为提示）
    pause_reward: float = 0.001
    backward_reward: float = 0.001
    # S3: 社交效率奖励（让步有效后恢复前进，补偿等待损失）
    social_efficiency_bonus: float = 0.02
    # S4: 过度等待惩罚（连续礼让超过阈值后触发，与 base 碰撞惩罚量级对齐）
    max_consecutive_social_actions: int = 5
    over_wait_penalty: float = -0.01
    # S5: 动作平滑惩罚（轻微，阻止无意义抖动）
    action_smoothing_penalty_coef: float = -0.005
    # S6: 高频切换惩罚（避免 FORWARD↔BACKWARD 乒乓）
    high_freq_switch_threshold: int = 3
    high_freq_switch_penalty_coef: float = -0.005
    # S7: 绕行区域奖励（行人行进方向外侧绕行→正奖励）
    detour_zone_inner: float = 1.5
    detour_zone_outer: float = 4.0
    detour_reward_coef: float = 0.05

@dataclass
class DidMultiAgentsCollideConfig(MeasurementConfig):
    type: str = "DidMultiAgentsCollide"
    
@dataclass
class STLMeasurementConfig(MeasurementConfig):
    type: str = "STL"

@dataclass
class PersonalSpaceComplianceMeasurementConfig(MeasurementConfig):
    type: str = "PersonalSpaceCompliance"
    use_geo_distance: bool = True
    
@dataclass
class HumanCollisionMeasurementConfig(MeasurementConfig):
    type: str = "HumanCollision"

@dataclass
class HumanVelocityMeasurementConfig(MeasurementConfig):
    type: str = "HumanVelocityMeasure"

@dataclass
class HumanFutureTrajectoryMeasurementConfig(MeasurementConfig):
    type: str = "HumanFutureTrajectory"


cs = ConfigStore.instance()

cs.store(
    package="habitat.task.measurements.multi_agent_nav_reward",
    group="habitat/task/measurements",
    name="multi_agent_nav_reward",
    node=MultiAgentNavReward,
)
cs.store(
    package="habitat.task.measurements.stl",
    group="habitat/task/measurements",
    name="stl",
    node=STLMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.psc",
    group="habitat/task/measurements",
    name="psc",
    node=PersonalSpaceComplianceMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.human_collision",
    group="habitat/task/measurements",
    name="human_collision",
    node=HumanCollisionMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.did_multi_agents_collide",
    group="habitat/task/measurements",
    name="did_multi_agents_collide",
    node=DidMultiAgentsCollideConfig,
)
cs.store(
    package="habitat.task.measurements.human_velocity_measure",
    group="habitat/task/measurements",
    name="human_velocity_measure",
    node=HumanVelocityMeasurementConfig,
)
cs.store(
    package="habitat.task.measurements.human_future_trajectory",
    group="habitat/task/measurements",
    name="human_future_trajectory",
    node=HumanFutureTrajectoryMeasurementConfig,
)