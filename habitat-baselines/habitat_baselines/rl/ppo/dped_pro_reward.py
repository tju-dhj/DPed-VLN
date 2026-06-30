#!/usr/bin/env python3

# -*- coding: utf-8 -*-



"""

DPed_pro 社交情商奖励函数模块



本模块实现了 DPed_pro 项目的奖励函数系统，支持 6 动作空间的社交情商奖励优化。



动作空间定义：

    0: STOP      — 停止

    1: FORWARD   — 前进

    2: TURN_LEFT — 左转

    3: TURN_RIGHT — 右转

    4: PAUSE     — 原地暂停（礼让行人）

    5: BACKWARD  — 后退（紧急避让）



奖励组件（可通过配置开关切换）：

    1. 基础距离奖励 (BaseDistanceReward)    — 原始 DPed-VLN 奖励，保证基本避障和到达

    2. 角速度惩罚 (AngularVelocityPenalty) — 抑制"陀螺式旋转"行为

    3. 社交礼让奖励 (SocialYieldingBonus)  — 鼓励 Pause/Backward 动作

    4. 动作平滑惩罚 (ActionSmoothingPenalty) — 减少 Forward/Backward 抖动

    5. 到达成功奖励 (SuccessReward)        — 到达目标的奖励

    6. 碰撞惩罚 (CollisionPenalty)         — 与行人碰撞的惩罚



使用方式：

    # 在配置中启用社交奖励

    habitat_baselines.rl.ppo:

        use_social_eq_reward: True

        angular_velocity_penalty_coef: -0.05

        pause_reward: 0.1

        social_efficiency_bonus: 0.5



设计理念：

    - 解耦设计：每个奖励组件独立计算，可自由组合

    - 配置驱动：通过配置文件控制启用/禁用和参数调优

    - 向后兼容：use_social_eq_reward=False 时等价于原始奖励



Author: DPed_pro Team

"""



from dataclasses import dataclass, field

from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np





# ═══════════════════════════════════════════════════════════════════════════

#  动作常量定义

# ═══════════════════════════════════════════════════════════════════════════



class ActionSpace:

    """6 动作空间常量定义"""

    STOP = 0

    FORWARD = 1

    TURN_LEFT = 2

    TURN_RIGHT = 3

    PAUSE = 4

    BACKWARD = 5

   

    TURNING_ACTIONS = {TURN_LEFT, TURN_RIGHT}

    MOVING_ACTIONS = {FORWARD, BACKWARD}

    SOCIAL_ACTIONS = {PAUSE, BACKWARD}

   

    @staticmethod

    def is_turning(action: int) -> bool:

        return action in ActionSpace.TURNING_ACTIONS

   

    @staticmethod

    def is_moving(action: int) -> bool:

        return action in ActionSpace.MOVING_ACTIONS

   

    @staticmethod

    def is_social(action: int) -> bool:

        return action in ActionSpace.SOCIAL_ACTIONS

   

    @staticmethod

    def is_forward(action: int) -> bool:

        return action == ActionSpace.FORWARD

   

    @staticmethod

    def is_backward(action: int) -> bool:

        return action == ActionSpace.BACKWARD

   

    @staticmethod

    def is_pause(action: int) -> bool:

        return action == ActionSpace.PAUSE

   

    @staticmethod

    def is_stop(action: int) -> bool:

        return action == ActionSpace.STOP





# ═══════════════════════════════════════════════════════════════════════════

#  主奖励计算器类

# ═══════════════════════════════════════════════════════════════════════════



class DPedProRewardCalculator:

    """

    DPed_pro 奖励计算器

   

    整合所有奖励组件，支持通过配置开关切换不同的奖励策略。

   

    接口兼容 dynamic_vln_trainer.py 中定义的 compute_reward 方法签名：

        compute_reward(observations, action, info, is_episode_done, is_success,

                      prev_distance_to_goal, current_distance_to_goal)

   

    使用方式：

        # 创建计算器

        config = {"use_social_eq_reward": True, ...}

        calculator = DPedProRewardCalculator(config)

       

        # 每步调用

        reward, extra_info = calculator.compute_reward(

            observations=obs,

            action=0-5,

            info=env_info,

            is_episode_done=False,

            is_success=False,

            prev_distance_to_goal=5.0,

            current_distance_to_goal=4.8,

        )

       

        # Episode 开始时重置

        calculator.reset_episode()

    """

   

    def __init__(self, config: Union[Dict, Any]):

        """

        初始化奖励计算器

       

        Args:

            config: 奖励配置（支持 Dict 或 omegaconf.DictConfig）

        """

        self.config = self._to_dict(config)

        self.state: Dict[str, Any] = {}

        self.use_social_eq = self.config.get("use_social_eq_reward", False)

        # ── 距离奖励开关 ──
        # 【关键修复】默认关闭，避免与 MultiAgentNavReward 中的 1.5*distance_to_goal_reward 叠加
        # 开启方式：在 yaml 中设置 use_distance_reward: true
        self.use_distance_reward = self.config.get("use_distance_reward", False)

        # ── 碰撞惩罚缩放系数 ──
        # 用于在 RL 层面放大/缩小碰撞惩罚，叠加在环境的 collide_human_penalty: -0.015 上
        # 默认 1.0 表示保持环境原值；设为 10.0 表示强化碰撞信号
        self.collision_penalty_scale = self.config.get("collision_penalty_scale", 1.0)

        # ── 安全距离奖励 ──
        # 【新增】当与行人保持安全距离时给予正奖励
        # 这鼓励 policy 主动与行人保持距离，而非仅仅不碰撞
        # 安全距离阈值（米）
        self.safe_distance_threshold = self.config.get("safe_distance_threshold", 1.5)
        # 安全距离奖励强度（每步）
        self.safe_distance_reward = self.config.get("safe_distance_reward", 0.01)

        # ── 读取各组件参数 ──

        # 角速度惩罚参数

        self.angular_penalty_coef = self.config.get("angular_velocity_penalty_coef", -0.05)

        self.max_acceptable_turn_rate = self.config.get("max_acceptable_turn_rate", 2)

        # ── 社交礼让奖励参数

        self.pause_reward_val = self.config.get("pause_reward", 0.1)

        self.backward_reward_val = self.config.get("backward_reward", 0.1)

        self.social_efficiency_bonus = self.config.get("social_efficiency_bonus", 0.5)

        self.max_consecutive_social = self.config.get("max_consecutive_social_actions", 3)

        self.over_wait_penalty = self.config.get("over_wait_penalty", -0.1)

       

        # 动作平滑惩罚参数

        self.smoothing_penalty_coef = self.config.get("action_smoothing_penalty_coef", -0.02)

       

        # 高频切换惩罚参数（新增）

        # 检测连续在 FORWARD ↔ BACKWARD 之间切换的次数

        self.high_freq_switch_threshold = self.config.get("high_freq_switch_threshold", 3)

        # 连续切换达到阈值后，每多一次切换施加的惩罚

        self.high_freq_switch_penalty_coef = self.config.get("high_freq_switch_penalty_coef", -0.05)

       

        # 基础奖励参数

        self.success_reward_val = self.config.get("success_reward", 10.0)

        self.collision_penalty_val = self.config.get("collision_penalty", -2.0)

       

        # ── 统计信息 ──

        self.stats = {

            "total_reward": 0.0,

            "distance_reward": 0.0,
            "safe_distance_reward": 0.0,

            "angular_penalty": 0.0,

            "social_bonus": 0.0,

            "smoothing_penalty": 0.0,

            "success_reward": 0.0,

            "collision_penalty": 0.0,

            "episode_count": 0,

        }

   

    def _to_dict(self, config: Union[Dict, Any]) -> Dict:

        """将配置对象转换为字典"""

        if isinstance(config, dict):

            return config

        if hasattr(config, "__dict__"):

            return vars(config)

        if hasattr(config, "_content"):

            return config._content

        try:

            from omegaconf import DictConfig

            if isinstance(config, DictConfig):

                return OmegaConf.to_container(config, resolve=True)

        except ImportError:

            pass

        return dict(config)

   

    def reset_episode(self):

        """重置 episode 状态（每个 episode 开始时调用）"""

        self.state = {}

        self.stats["episode_count"] += 1

   

    def compute_reward(

        self,

        observations: Dict,

        action: int,

        info: Dict,

        is_episode_done: bool = False,

        is_success: bool = False,

        prev_distance_to_goal: Optional[float] = None,

        current_distance_to_goal: Optional[float] = None,

    ) -> Tuple[float, Dict]:

        """

        计算奖励（兼容 trainer 接口）

       

        Args:

            observations: 观察字典

            action: 当前动作 (0-5)

            info: 环境返回的 info 字典

            is_episode_done: episode 是否结束

            is_success: 是否成功

            prev_distance_to_goal: 上一步到目标的距离

            current_distance_to_goal: 当前到目标的距离

           

        Returns:

            (total_reward, extra_info)

        """

        total_reward = 0.0

        extra_info = {}

       

        # ── 1. 基础距离奖励 ──

        if (self.use_distance_reward and 
            prev_distance_to_goal is not None and 
            current_distance_to_goal is not None):

            distance_reward = prev_distance_to_goal - current_distance_to_goal

            total_reward += distance_reward

            self.stats["distance_reward"] += distance_reward

            extra_info["distance_reward"] = distance_reward

       

        # ── 2. 角速度惩罚 ──

        if self.use_social_eq:

            angular_reward = self._compute_angular_penalty(action)

            total_reward += angular_reward

            self.stats["angular_penalty"] += angular_reward

            extra_info["angular_penalty"] = angular_reward

           

            # ── 3. 社交礼让奖励 ──

            social_reward, social_info = self._compute_social_reward(action, observations, info)

            total_reward += social_reward

            self.stats["social_bonus"] += social_reward

            extra_info["social_bonus"] = social_reward

            extra_info.update(social_info)

           

            # ── 4. 动作平滑惩罚 ──

            smoothing_reward = self._compute_smoothing_penalty(action)

            total_reward += smoothing_reward

            self.stats["smoothing_penalty"] += smoothing_reward

            extra_info["smoothing_penalty"] = smoothing_reward

           

            # ── 5. 高频切换惩罚（新增）─

            # 检测连续在 FORWARD ↔ BACKWARD 之间切换的行为

            hfs_reward = self._compute_high_freq_switch_penalty(action)

            total_reward += hfs_reward

            self.stats["high_freq_switch_penalty"] = self.stats.get("high_freq_switch_penalty", 0) + hfs_reward

            extra_info["high_freq_switch_penalty"] = hfs_reward

       

        # ── 6. 成功奖励 ──

        if is_success:

            total_reward += self.success_reward_val

            self.stats["success_reward"] += self.success_reward_val

            extra_info["success_reward"] = self.success_reward_val

       

        # ── 6. 碰撞惩罚（缩放版）──
        # 【关键修复】将环境惩罚乘以 collision_penalty_scale，增强碰撞信号
        collision_count = info.get("human_collision", 0) if isinstance(info, dict) else 0

        if collision_count > 0:
            scaled_collision_penalty = self.collision_penalty_val * collision_count * self.collision_penalty_scale
            total_reward += scaled_collision_penalty
            self.stats["collision_penalty"] += scaled_collision_penalty
            extra_info["collision_penalty"] = scaled_collision_penalty

        # ── 7. 安全距离奖励（新增）──
        # 【关键修复】当与行人保持安全距离时给予正奖励
        # 这鼓励 policy 主动远离行人，而非仅在即将碰撞时才躲避
        safe_reward = self._compute_safe_distance_reward(observations)
        if safe_reward > 0:
            total_reward += safe_reward
            self.stats["safe_distance_reward"] += safe_reward
            extra_info["safe_distance_reward"] = safe_reward

       

        self.stats["total_reward"] += total_reward

        return total_reward, extra_info

   

    def _compute_angular_penalty(self, action: int) -> float:

        """

        计算角速度惩罚

       

        抑制"陀螺式旋转"行为：记录连续转向次数，超过阈值后惩罚

        """

        penalty = 0.0

       

        # 更新连续转向计数

        turn_count_key = "consecutive_turn_count"

        current_count = self.state.get(turn_count_key, 0)

       

        if ActionSpace.is_turning(action):

            current_count += 1

        else:

            current_count = 0

       

        self.state[turn_count_key] = current_count

       

        # 超出阈值后施加惩罚

        if current_count > self.max_acceptable_turn_rate:

            excess = current_count - self.max_acceptable_turn_rate

            penalty = self.angular_penalty_coef * excess

       

        return penalty

   

    def _compute_social_reward(

        self,

        action: int,

        observations: Dict,

        info: Dict,

    ) -> Tuple[float, Dict]:

        """

        计算社交礼让奖励

       

        鼓励 Pause/Backward 动作，避免过度等待

       

        核心设计原则：

        - 礼让奖励：执行 PAUSE/BACKWARD 时给予正奖励

        - 效率奖励：仅在行人实际通过后（让步有效）恢复前进时给予正奖励

        - 避免冲突：效率奖励必须基于"真实让步"，而非任意 BACKWARD

        """

        reward = 0.0

        info_update = {}

       

        # ── 1. 基础礼让奖励 ──

        if ActionSpace.is_pause(action):

            reward += self.pause_reward_val

            info_update["is_pause"] = True

        elif ActionSpace.is_backward(action):

            reward += self.backward_reward_val

            info_update["is_backward"] = True

       

        # ── 2. 社交效率奖励（关键修复）───────────────────────────────

        #

        # 原问题：任何 BACKWARD 后的 FORWARD 都奖励，导致：

        #   BACKWARD → FORWARD: +0.5 (效率) + (-0.02) (平滑) = +0.48 净奖励

        #   这鼓励了无意义的 BACKWARD→FORWARD 切换！

        #

        # 新方案：效率奖励必须满足以下条件：

        #   (a) 必须先有 PAUSE/BACKWARD 让步动作

        #   (b) 行人在让步期间实际通过（基于距离变化检测）

        #   (c) 行人通过后才恢复 FORWARD

        #

        yielding_key = "is_yielding"

        ped_passed_key = "pedestrian_passed_during_yield"

        prev_distance_key = "prev_ped_distance"

       

        # 检测行人距离（如果有观测数据）

        current_ped_distance = self._get_ped_distance(observations)

       

        # 行人通过检测：当前行人距离增加（或变成很大），说明行人已通过

        prev_ped_distance = self.state.get(prev_distance_key, None)

        pedestrian_passed = False

       

        if current_ped_distance is not None and prev_ped_distance is not None:

            # 行人距离显著增加，或行人在让步期间远离到安全距离

            if current_ped_distance > prev_ped_distance + 0.3:  # 行人远离了30cm以上

                pedestrian_passed = True

       

        # 更新行人距离状态

        if current_ped_distance is not None:

            self.state[prev_distance_key] = current_ped_distance

       

        # 行人通过时设置标志

        if pedestrian_passed:

            self.state[ped_passed_key] = True

       

        # 仅当行人在让步期间通过后，才奖励恢复前进

        if ActionSpace.is_forward(action):

            has_yielded = self.state.get(yielding_key, False)

            did_ped_pass = self.state.get(ped_passed_key, False)

           

            if has_yielded and did_ped_pass:

                reward += self.social_efficiency_bonus

                info_update["social_efficiency_bonus"] = True

                info_update["yielding_was_effective"] = True

                # 重置状态，避免重复奖励

                self.state[yielding_key] = False

                self.state[ped_passed_key] = False

            elif has_yielded:

                # 让步了但行人还没通过就恢复前进 - 不惩罚但也不奖励

                info_update["yielding_early_resume"] = True

        elif ActionSpace.is_social(action):

            # 记录让步开始

            self.state[yielding_key] = True

            self.state[ped_passed_key] = False  # 重置行人通过标志

       

        # ── 3. 过度等待惩罚 ──

        social_count_key = "consecutive_social_count"

        social_count = self.state.get(social_count_key, 0)

       

        if ActionSpace.is_social(action):

            social_count += 1

        else:

            social_count = 0

       

        self.state[social_count_key] = social_count

       

        if social_count > self.max_consecutive_social:

            reward += self.over_wait_penalty

            info_update["over_wait_penalty"] = True

       

        return reward, info_update

   

    def _get_ped_distance(self, observations: Dict) -> Optional[float]:

        """

        从观测中获取行人距离

       

        优先使用传感器检测的行人位置，如果没有则返回 None

        """

        # 尝试从 observations 中提取行人距离

        # 这需要根据实际环境配置调整

        if "human_detections" in observations:

            dets = observations["human_detections"]

            if dets is not None and len(dets) > 0:

                # 返回最近的行人距离

                return float(dets[0].get("distance", float("inf")))

       

        if "ped_distance" in observations:

            return float(observations["ped_distance"])

       

        # 如果没有行人距离信息，返回 None（使用备选逻辑）

        return None

   

    def _compute_safe_distance_reward(self, observations: Dict) -> float:
        """
        计算安全距离奖励
        
        当 agent 与所有行人的距离都超过 safe_distance_threshold 时给予正奖励。
        这鼓励 policy 主动与行人保持安全距离，而非仅仅避免碰撞。
        
        使用 geoesic distance（如果可用）或欧几里得距离来检测行人距离。
        """
        reward = 0.0
        
        if not isinstance(observations, dict):
            return reward
        
        use_k_robot = "agent_0_localization_sensor"
        robot_pos = observations.get(use_k_robot, None)
        if robot_pos is None:
            return reward
        
        if len(robot_pos) < 3:
            return reward
        
        robot_pos = np.array(robot_pos[:3])
        
        # 遍历所有可能的行人传感器
        for i in range(1, 7):
            use_k_human = f"agent_{i}_localization_sensor"
            human_pos = observations.get(use_k_human, None)
            if human_pos is None:
                continue
            if len(human_pos) < 3:
                continue
            
            human_pos = np.array(human_pos[:3])
            distance = np.linalg.norm(human_pos - robot_pos, ord=2)
            
            if distance > self.safe_distance_threshold:
                reward += self.safe_distance_reward
        
        return reward

    def _compute_smoothing_penalty(self, action: int) -> float:

        """

        计算动作平滑惩罚

       

        减少 Forward/Backward 之间的频繁切换

       

        与效率奖励的协调设计：

        - 如果是有效的让步恢复（行人通过后恢复前进），则不施加平滑惩罚

        - 只有无意义的来回切换才惩罚

        """

        penalty = 0.0

       

        # 获取当前状态

        yielding_key = "is_yielding"

        ped_passed_key = "pedestrian_passed_during_yield"

       

        has_yielded = self.state.get(yielding_key, False)

        did_ped_pass = self.state.get(ped_passed_key, False)

       

        # 如果是有效的让步恢复，不惩罚（因为效率奖励会奖励这个行为）

        if ActionSpace.is_forward(action) and has_yielded and did_ped_pass:

            return penalty

       

        # 只有无意义的切换才惩罚

        smoothing_key = "prev_moving_action"

        prev_moving = self.state.get(smoothing_key, None)

       

        if ActionSpace.is_moving(action):

            self.state[smoothing_key] = action

           

            if prev_moving is not None and prev_moving != action:

                penalty = self.smoothing_penalty_coef

       

        return penalty

   

    def _compute_high_freq_switch_penalty(self, action: int) -> float:

        """

        计算高频切换惩罚

       

        检测连续在 FORWARD ↔ BACKWARD 之间切换的行为模式，

        当切换次数超过阈值时施加惩罚。

       

        例如：FORWARD → BACKWARD → FORWARD → BACKWARD → FORWARD

        这是 4 次切换（F→B, B→F, F→B, B→F），会触发惩罚。

       

        有效让步后的切换不惩罚（已被效率奖励覆盖）

        """

        penalty = 0.0

       

        # 获取当前状态

        yielding_key = "is_yielding"

        ped_passed_key = "pedestrian_passed_during_yield"

       

        has_yielded = self.state.get(yielding_key, False)

        did_ped_pass = self.state.get(ped_passed_key, False)

       

        # 如果是有效的让步恢复，不惩罚

        if ActionSpace.is_forward(action) and has_yielded and did_ped_pass:

            # 重置高频切换计数器（有效切换不计入）

            self.state["high_freq_switch_count"] = 0

            return penalty

       

        # 获取切换计数

        switch_count_key = "high_freq_switch_count"

        switch_count = self.state.get(switch_count_key, 0)

       

        prev_action_key = "prev_action"

        prev_action = self.state.get(prev_action_key, None)

       

        # 检测是否发生了 FORWARD ↔ BACKWARD 切换

        is_switch = False

        if (prev_action is not None and

            ActionSpace.is_forward(action) and ActionSpace.is_backward(prev_action)):

            is_switch = True

        elif (prev_action is not None and

              ActionSpace.is_backward(action) and ActionSpace.is_forward(prev_action)):

            is_switch = True

       

        # 更新状态

        self.state[prev_action_key] = action

       

        if is_switch:

            switch_count += 1

            self.state[switch_count_key] = switch_count

           

            # 超过阈值后开始惩罚

            if switch_count >= self.high_freq_switch_threshold:

                # 超出阈值的次数越多，惩罚越大

                excess = switch_count - self.high_freq_switch_threshold + 1

                penalty = self.high_freq_switch_penalty_coef * excess

        else:

            # 非切换动作，重置计数器

            # 但如果当前是 FORWARD/BACKWARD，保持计数器（用于检测下一次切换）

            if not ActionSpace.is_moving(action):

                self.state[switch_count_key] = 0

       

        return penalty

   

    def get_stats(self) -> Dict:

        """获取奖励统计信息"""

        return self.stats.copy()

   

    def reset_stats(self):

        """重置统计信息"""

        self.stats = {

            "total_reward": 0.0,

            "distance_reward": 0.0,
            "safe_distance_reward": 0.0,

            "angular_penalty": 0.0,

            "social_bonus": 0.0,

            "smoothing_penalty": 0.0,

            "high_freq_switch_penalty": 0.0,

            "success_reward": 0.0,

            "collision_penalty": 0.0,

            "episode_count": self.stats["episode_count"],

        }





# ═══════════════════════════════════════════════════════════════════════════

#  兼容层：从旧接口迁移

# ═══════════════════════════════════════════════════════════════════════════



class LegacyRewardWrapper:

    """

    旧接口兼容包装器

   

    如果 trainer 中使用的是旧版 reward calculator，

    可以使用这个包装器来兼容新的奖励组件。

    """

   

    def __init__(self, base_reward_func, social_eq_config: Optional[Dict] = None):

        """

        Args:

            base_reward_func: 基础奖励函数 (callable)

            social_eq_config: 社交奖励配置，如果为 None 则使用默认配置

        """

        self.base_reward_func = base_reward_func

        self.social_config = social_eq_config or get_default_social_eq_config()

       

        # 创建社交奖励组件

        self.social_calculator = DPedProRewardCalculator(self.social_config)

   

    def compute_reward(self, *args, **kwargs):

        """兼容旧接口"""

        base_reward = self.base_reward_func(*args, **kwargs)

       

        # 如果启用社交奖励，添加额外奖励

        if self.social_config.get("use_social_eq_reward", False):

            action = kwargs.get("action", 0)

            obs = kwargs.get("observations", {})

            info = kwargs.get("info", {})

           

            extra_reward, _ = self.social_calculator.compute_reward(

                observations=obs,

                action=action,

                info=info,

            )

            return base_reward + extra_reward

       

        return base_reward





# ═══════════════════════════════════════════════════════════════════════════

#  配置工厂函数

# ═══════════════════════════════════════════════════════════════════════════



def create_reward_calculator(config: Union[Dict, Any]) -> DPedProRewardCalculator:

    """

    从配置创建奖励计算器

   

    Args:

        config: PPO 配置对象（支持 Dict、omegaconf.DictConfig 或 dataclass）

       

    Returns:

        DPedProRewardCalculator 实例

    """

    return DPedProRewardCalculator(config)





def get_default_social_eq_config() -> Dict:

    """

    获取默认的社交情商奖励配置

   

    Returns:

        配置字典

    """

    return {

        # 开关

        "use_social_eq_reward": True,

       

        # 角速度惩罚

        "angular_velocity_penalty_coef": -0.05,

        "max_acceptable_turn_rate": 2,

       

        # 社交礼让奖励

        "pause_reward": 0.1,

        "backward_reward": 0.1,

        "social_efficiency_bonus": 0.5,

        "max_consecutive_social_actions": 3,

        "over_wait_penalty": -0.1,

       

        # 动作平滑惩罚

        "action_smoothing_penalty_coef": -0.02,

       

        # 高频切换惩罚（FORWARD ↔ BACKWARD 连续切换）

        "high_freq_switch_threshold": 3,      # 连续切换超过此次数后开始惩罚

        "high_freq_switch_penalty_coef": -0.05,  # 每次额外切换的惩罚幅度

       

        # 基础奖励

        "success_reward": 10.0,

        "collision_penalty": -2.0,

    }





def log_reward_breakdown(calculator: DPedProRewardCalculator) -> str:

    """

    生成奖励分解日志字符串

   

    Args:

        calculator: 奖励计算器实例

       

    Returns:

        格式化的日志字符串

    """

    stats = calculator.get_stats()

    lines = ["[Reward Breakdown]"]

   

    lines.append(f"  Distance Reward: {stats['distance_reward']:.4f}")

    lines.append(f"  Angular Penalty:  {stats['angular_penalty']:.4f}")

    lines.append(f"  Social Bonus:     {stats['social_bonus']:.4f}")

    lines.append(f"  Smoothing Penalty:{stats['smoothing_penalty']:.4f}")

    lines.append(f"  High-Freq Switch: {stats.get('high_freq_switch_penalty', 0):.4f}")

    lines.append(f"  Success Reward:   {stats['success_reward']:.4f}")

    lines.append(f"  Collision Penalty:{stats['collision_penalty']:.4f}")

    lines.append(f"  ─────────────────────────────")

    lines.append(f"  TOTAL:            {stats['total_reward']:.4f}")

    lines.append(f"  Episodes:         {stats['episode_count']}")

   

    return "\n".join(lines)