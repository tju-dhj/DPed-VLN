#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
专家数据采集器

这个模块实现了简化的专家数据采集功能，不依赖复杂的PPO训练机制。
主要用于收集专家动作执行的数据，用于后续的强化学习训练。

主要特性：
- 简化的数据采集流程
- 支持Oracle专家动作
- 不依赖rollout缓冲区
- 支持多环境并行数据采集
- 可配置的episode和步数限制

作者: 基于Falcon项目修改
"""

# 标准库导入
import os
import time
import json
import pathlib
import shutil
from typing import Dict, Any, Optional, List
import contextlib
from PIL import Image
import math
# 第三方库导入
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from habitat.tasks.rearrange.utils import get_angle_to_pos
   

# Habitat相关导入
import habitat_baselines.rl.multi_agent  # noqa: F401.  # 多智能体支持
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.utils import profiling_wrapper

# Habitat-baselines核心组件
from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)

# 分布式PPO相关导入
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

# 工具函数导入
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
    generate_video,
)
from habitat_baselines.utils.timing import g_timer
from habitat.utils.visualizations.utils import observations_to_image
from habitat.utils.visualizations import maps

# 导入自定义传感器
# import falcon.additional_sensor  # noqa: F401

def create_agent0_video_frame(observation: Dict, info: Dict = None) -> np.ndarray:
    """
    创建只包含agent0第一视角、第三视角和俯视图的视频帧
    
    Args:
        observation: 环境观测字典
        info: 信息字典（可选，包含top_down_map）
    
    Returns:
        拼接后的图像（第一视角、第三视角、俯视图水平排列）
    """
    if info is None:
        info = {}
    
    render_obs_images = []
    
    # 只选择agent0的第一视角和第三视角
    # 优先选择overhead_front_rgb，如果没有则选择jaw_rgb
    first_person_keys = [
        "agent_0_overhead_front_rgb",
        "agent_0_articulated_agent_jaw_rgb",
    ]
    
    third_person_key = "agent_0_third_rgb"
    
    # 查找第一视角
    for key in first_person_keys:
        if key in observation:
            img = observation[key]
            if not isinstance(img, np.ndarray):
                img = img.cpu().numpy()
            if img.dtype != np.uint8:
                img = (img * 255.0).astype(np.uint8)
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = np.concatenate([img for _ in range(3)], axis=2)
            render_obs_images.append(img)
            break
    
    # 查找第三视角
    if third_person_key in observation:
        img = observation[third_person_key]
        if not isinstance(img, np.ndarray):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = (img * 255.0).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.concatenate([img for _ in range(3)], axis=2)
        render_obs_images.append(img)
    
    if len(render_obs_images) == 0:
        # 如果都没有找到，使用原始的observations_to_image作为fallback
        return observations_to_image(observation, info)
    
    # 水平拼接第一视角和第三视角
    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        from habitat.utils.visualizations.utils import tile_images
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)
    
    # 添加碰撞标记
    collisions_key = "collisions"
    if collisions_key in info and info[collisions_key].get("is_collision", False):
        from habitat.utils.visualizations.utils import draw_collision
        render_frame = draw_collision(render_frame)
    
    # 添加俯视图（top_down_map）
    top_down_map_key = "top_down_map"
    if top_down_map_key in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info[top_down_map_key], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)
    
    return render_frame

def quat_to_pitch_asin(w, x, y, z):
    sinp = 2.0 * (w * y - z * x)
    # clamp 数值到 [-1, 1] 避免浮点误差导致 asin nan
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)  # 返回弧度

def save_to_disk(
    rgb,
    depth,
    third_rgb,
    human_num,
    action,
    distance_to_goal,
    ep_id,
    scene_id,
    pedestrian_in_view=None,
    trajectories=None,
    split="train",
    data_folder="data/collect_data",
    merge_ep=True,
):
    """
    保存数据到磁盘
    
    Args:
        rgb: RGB图像数据
        depth: 深度图像数据
        human_num: 人员数量数据
        action: 动作数据
        distance_to_goal: 到目标的距离数据
        ep_id: episode ID
        scene_id: 场景ID，用于创建场景文件夹
        pedestrian_in_view: 每一步行人是否在视野内的记录
        trajectories: 所有agents的轨迹数据
        split: 数据集分割
        data_folder: 数据保存根目录
        merge_ep: 是否合并episode
    """
    # 从scene_id中提取场景名称（去掉路径和扩展名）
    scene_name = pathlib.Path(scene_id).stem  # 获取文件名（不含扩展名）
    
    # 创建路径：data_folder/split/scene_name/episode_id/
    DATA_ROOT = pathlib.Path(data_folder) / split / scene_name
    os.makedirs(DATA_ROOT / ep_id / "rgb", exist_ok=True)
    os.makedirs(DATA_ROOT / ep_id / "depth", exist_ok=True)
    os.makedirs(DATA_ROOT / ep_id / "third_rgb", exist_ok=True)
    os.makedirs(DATA_ROOT / ep_id / "human_num", exist_ok=True)
    os.makedirs(DATA_ROOT / ep_id / "action", exist_ok=True)
    os.makedirs(DATA_ROOT / ep_id / "distance_to_goal", exist_ok=True)
    
    n = rgb.shape[0]
    
    # 保存RGB图像
    for i in range(n):
        img = Image.fromarray(rgb[i])
        img.save(DATA_ROOT / ep_id / "rgb" / f"{i}_0.jpg")
    
    # 保存深度图像
    for i in range(n):
        img = depth[i].squeeze()
        img = (img * 1000).astype(np.uint16)
        img = Image.fromarray(img)
        img.save(DATA_ROOT / ep_id / "depth" / f"{i}_0.png")
    
    # 保存第三视角RGB图像
    for i in range(n):
        img = Image.fromarray(third_rgb[i])
        img.save(DATA_ROOT / ep_id / "third_rgb" / f"{i}_0.jpg")
    
    # 保存人员数量数据
    with open(DATA_ROOT / ep_id / "human_num" / "0.json", "w") as f:
        json.dump(human_num.tolist(), f)
    
    # 保存动作数据
    with open(DATA_ROOT / ep_id / "action" / "0.json", "w") as f:
        json.dump(action, f)
    
    # 保存distance_to_goal数据
    with open(DATA_ROOT / ep_id / "distance_to_goal" / "0.json", "w") as f:
        json.dump(distance_to_goal, f)
    
    # 保存pedestrian_in_view数据
    if pedestrian_in_view is not None:
        os.makedirs(DATA_ROOT / ep_id / "pedestrian_in_view", exist_ok=True)
        with open(DATA_ROOT / ep_id / "pedestrian_in_view" / "0.json", "w") as f:
            json.dump(pedestrian_in_view, f)
    
    # 保存轨迹数据
    if trajectories is not None:
        os.makedirs(DATA_ROOT / ep_id / "trajectories", exist_ok=True)
        with open(DATA_ROOT / ep_id / "trajectories" / "0.json", "w") as f:
            json.dump(trajectories, f, indent=2)


@baseline_registry.register_trainer(name="expert_data_collector_multi_envs")
class ExpertDataCollectorV3(BaseRLTrainer):
    """
    专家数据采集器
    
    简化的数据采集器，专门用于收集专家动作执行的数据。
    不依赖复杂的PPO训练机制，直接执行专家动作并收集数据。
    
    主要特性：
    - 简化的数据采集流程
    - 支持Oracle专家动作
    - 不依赖rollout缓冲区
    - 支持多环境并行数据采集
    - 可配置的episode和步数限制
    """
    
    # 支持的任务类型
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        """
        初始化专家数据采集器
        
        Args:
            config: 采集配置，包含环境、数据保存等设置
        """
        # 调用父类初始化
        super().__init__(config)

        # 初始化核心组件
        self.envs = None               # 向量化环境
        self.obs_transforms = []       # 观察变换器列表
        self._env_spec = None          # 环境规范
        self.prev_actions = []         # 每个环境的前一步动作，用于滞回机制
        
        # 防卡机制相关变量
        self.repeated_turn_count = []  # 每个环境的重复转向计数器
        self.angle_diff = []           # 每个环境的角度差
        self.max_repeated_turns = 3    # 最大重复转向次数
        self.angle_threshold_for_forced_forward = np.deg2rad(5.0)  # 强制前进的角度阈值

        # 检查是否为分布式训练
        self._is_distributed = get_distrib_size()[2] > 1
        
        # 缓存速度配置（用于ORCA计算）
        self._robot_lin_speed = None
        self._pedestrian_lin_speed = None

    def _get_speed_config(self):
        """
        从配置中读取机器人和行人的速度配置
        
        注意速度比例：
        - 控制频率相同的情况下，速度值决定每步移动距离的比例
        - 配置中机器人:行人 = 30.0:12.0 = 2.5:1
        - 这意味着机器人每步移动距离是行人的2.5倍
        
        Returns:
            tuple: (robot_lin_speed, pedestrian_lin_speed)
        """
        if self._robot_lin_speed is None or self._pedestrian_lin_speed is None:
            try:
                # 尝试从配置中读取机器人速度
                robot_lin_speed = self.config.habitat.task.actions.agent_0_discrete_move_forward.lin_speed
            except (AttributeError, KeyError):
                robot_lin_speed = 30.0  # 默认值，对应配置中的30.0
                
            try:
                # 尝试从配置中读取行人速度
                pedestrian_lin_speed = self.config.habitat.task.actions.agent_1_oracle_nav_randcoord_action_obstacle.lin_speed
            except (AttributeError, KeyError):
                pedestrian_lin_speed = 12.0  # 默认值，对应配置中的12.0
            
            self._robot_lin_speed = robot_lin_speed
            self._pedestrian_lin_speed = pedestrian_lin_speed
        
        return self._robot_lin_speed, self._pedestrian_lin_speed

    def _init_envs(self, config=None, is_eval: bool = False):
        """
        初始化向量化环境
        
        Args:
            config: 环境配置，如果为None则使用self.config
            is_eval: 是否为评估模式，影响环境的创建方式
        """
        if config is None:
            config = self.config
            
        # 使用Hydra实例化环境工厂
        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )
        
        # 构造向量化环境
        self.envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )

        # 创建环境规范
        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )
        
        # 初始化每个环境的前一步动作状态（用于滞回机制）
        self.prev_actions = [0] * self.envs.num_envs
        
        # 初始化防卡机制变量
        self.repeated_turn_count = [0] * self.envs.num_envs
        self.angle_diff = [0.0] * self.envs.num_envs

    def _create_obs_transforms(self):
        """
        创建观察变换器
        """
        # 获取激活的观察变换器
        self.obs_transforms = get_active_obs_transforms(self.config)
        # 应用变换器到观察空间
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _recreate_envs(self) -> bool:
        """
        自愈：当子进程崩溃或vector env通信异常时，重建向量环境
        返回是否重建成功
        """
        try:
            with contextlib.suppress(Exception):
                if self.envs is not None:
                    self.envs.close()
            self._init_envs(self.config, is_eval=False)
            # 重新创建obs transforms绑定的obs space
            self._create_obs_transforms()
            # 重置并做一次post_step，确保可用
            observations = self.envs.reset()
            observations = self.envs.post_step(observations)
            # 重置本地状态
            self.prev_actions = [0] * self.envs.num_envs
            self.repeated_turn_count = [0] * self.envs.num_envs
            self.angle_diff = [0.0] * self.envs.num_envs
            return True
        except Exception as e:
            if rank0_only():
                logger.error(f"Failed to recreate vector envs: {e}")
            return False

    def _try_get_current_episode(self, env_idx: int):
        """
        安全获取当前episode，失败返回None
        """
        try:
            return self.envs.call_at(env_idx, "current_episode", {"all_info": True})
        except Exception as e:
            if rank0_only():
                logger.warning(f"current_episode failed on env {env_idx}: {e}")
            return None

    def _init_train(self, resume_state=None):
        """
        初始化数据采集过程
        
        Args:
            resume_state: 恢复状态，用于从检查点恢复采集
        """
        # 1. 处理恢复状态
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )
            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        # 2. 强制分布式训练设置
        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        # 3. 分布式训练设置
        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

        # 4. 记录配置信息
        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        # 5. 初始化环境
        self._init_envs()

        # 6. 获取计算设备
        self.device = get_device(self.config)

        # 7. 创建观察变换器
        self._create_obs_transforms()

        # 8. 获取初始观察并处理
        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # 9. 记录采集开始时间
        self.t_start = time.time()

    def _get_expert_action_for_agent_0_0(self, env_idx: int, observations: Dict[str, Any]) -> Optional[int]:
        """
        为agent_0获取专家动作
        
        使用Habitat内置的Oracle导航功能
        
        Args:
            env_idx: 环境索引
            observations: 当前观察
            
        Returns:
            专家动作，如果无法获取则返回None
        """
        oracle_path_key = "agent_0_main_oracle_shortest_path_sensor"
        # oracle_path = observations[oracle_path_key]
        # return self._compute_action_from_oracle_path(oracle_path, env_idx)
        expert_action = observations[env_idx][oracle_path_key]
        return expert_action
    
    def _get_expert_action_for_agent_0(self, env_idx: int, observations: Dict[str, Any], global_distance_to_goal: List[float]) -> Optional[int]:
        """
        为agent_0获取专家动作
        
        使用Habitat内置的Oracle导航功能
        
        Args:
            env_idx: 环境索引
            observations: 当前观察
            
        Returns:
            专家动作，如果无法获取则返回None
        """
        # oracle_path_key = "agent_0_oracle_shortest_path_sensor"
        oracle_path_key = "agent_0_main_oracle_shortest_path_sensor"
        oracle_path = observations[oracle_path_key]
        # 使用改进的避障算法
        # 选项1: 原始方法
        # return self._compute_action_from_oracle_path(oracle_path, env_idx, observations, global_distance_to_goal)
        # 选项2: 改进方法（更保守，会停止）
        # return self._compute_action_from_oracle_path_improved(oracle_path, env_idx, observations, global_distance_to_goal)
        # 选项3: ORCA改进方法（8秒预测，旋转避障，不会因行人停止）- 推荐使用
        return self._compute_action_from_oracle_path_orca_improved(oracle_path, env_idx, observations, global_distance_to_goal)
   
            
    def _check_pedestrians_in_camera_view(self, agent_position: np.ndarray, agent_rotation, 
                                          pedestrian_positions: list, 
                                          max_distance: float = 5.0, 
                                          fov_horizontal: float = 90.0,
                                          fov_vertical: float = 90.0) -> tuple:
        """
        检查行人是否在机器人相机视野内
        
        使用相机的FOV和3D几何关系来判断行人是否真正在相机视野内。
        考虑水平和垂直视场角，以及相机的实际朝向。
        
        Args:
            agent_position: 相机/机器人位置 (x, y, z)
            agent_rotation: 相机/机器人旋转（四元数）
            pedestrian_positions: 行人位置列表
            max_distance: 最大检测距离（米）
            fov_horizontal: 水平视场角（度）
            fov_vertical: 垂直视场角（度）
            
        Returns:
            tuple: (pedestrians_in_view, closest_distance)
                - pedestrians_in_view: 在视野内的行人数量
                - closest_distance: 最近行人的距离
        """
        if len(pedestrian_positions) == 0:
            return 0, float('inf')
        
        # 从四元数获取相机朝向
        if hasattr(agent_rotation, 'w'):
            w, x, y, z = agent_rotation.w, agent_rotation.x, agent_rotation.y, agent_rotation.z
        elif hasattr(agent_rotation, 'components'):
            w, x, y, z = agent_rotation.components
        else:
            w, x, y, z = agent_rotation[0], agent_rotation[1], agent_rotation[2], agent_rotation[3]
        
        # 构建旋转矩阵 - 将世界坐标转换到相机坐标
        # 四元数转旋转矩阵
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        # Habitat相机坐标系：Z轴向前，Y轴向下，X轴向右
        # 需要额外的坐标系转换：将相机Z轴对齐到前方
        # Habitat的相机通常是：Z向前（相机看向+Z），Y向下，X向右
        
        # 计算视场角范围（以弧度为单位）
        fov_h_rad = np.deg2rad(fov_horizontal)
        fov_v_rad = np.deg2rad(fov_vertical)
        half_fov_h = fov_h_rad / 2.0
        half_fov_v = fov_v_rad / 2.0
        
        pedestrians_in_view = 0
        closest_distance = float('inf')
        
        for ped_pos in pedestrian_positions:
            # 计算行人相对于相机的位置（世界坐标）
            relative_pos_world = ped_pos - agent_position
            
            # 转换到相机坐标系
            relative_pos_camera = R.T @ relative_pos_world
            
            # 在Habitat的相机坐标系中：
            # X: 右（正X）
            # Y: 下（正Y）  
            # Z: 前（正Z，相机看向这个方向）
            
            cam_x = relative_pos_camera[0]
            cam_y = relative_pos_camera[1]
            cam_z = relative_pos_camera[2]
            
            # 行人必须在相机前方（Z > 0）
            if cam_z <= 0.01:
                continue
            
            # 计算距离
            distance = np.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
            
            # 检查距离是否在范围内
            if distance > max_distance:
                continue
            
            # 计算水平和垂直角度
            # 水平角：相对于Z轴在XZ平面的角度
            horizontal_angle = np.arctan2(cam_x, cam_z)
            
            # 垂直角：相对于Z轴在YZ平面的角度  
            vertical_angle = np.arctan2(-cam_y, cam_z)  # 负号因为Y向下
            
            # 检查是否在视场角内
            if abs(horizontal_angle) <= half_fov_h and abs(vertical_angle) <= half_fov_v:
                pedestrians_in_view += 1
                closest_distance = min(closest_distance, distance)
        
        return pedestrians_in_view, closest_distance
    
    def _get_other_agents_info_from_observations(self, observations: dict, env_idx: int) -> tuple:
        """
        从observations中获取其他智能体（动态行人）的信息
        参考ORCA策略的实现方式，使用human_velocity_sensor
        
        Args:
            observations: 观察数据
            env_idx: 环境索引
            
        Returns:
            tuple: (positions, rotations, velocities) 其他智能体的位置、旋转、速度信息
        """
        try:
            positions = []
            rotations = []
            velocities = []
            
            # 检查是否有human_velocity_sensor数据
            if 'agent_0_human_velocity_sensor' in observations:
                human_velocity_data = observations['agent_0_human_velocity_sensor']
                
                # 处理CUDA张量转换
                if hasattr(human_velocity_data, 'cpu'):
                    human_velocity_data = human_velocity_data.cpu().numpy()
                elif hasattr(human_velocity_data, 'detach'):
                    human_velocity_data = human_velocity_data.detach().cpu().numpy()
                
                # 获取当前环境的数据
                if len(human_velocity_data.shape) > 1:
                    env_data = human_velocity_data[env_idx]  # 形状为 (6, 6)
                else:
                    env_data = human_velocity_data  # 形状为 (6,)
                
                # 遍历所有可能的智能体（最多6个）
                for j in range(6):
                    if len(env_data.shape) > 1:
                        agent_data = env_data[j]
                    else:
                        agent_data = env_data
                    
                    # 检查智能体是否存在（位置x < -90表示不存在）
                    if agent_data[0] < -90:
                        break
                    else:
                        # 提取位置 (x, y, z)
                        pos = agent_data[:3]
                        positions.append(pos)
                        
                        # 提取旋转角度
                        rot = agent_data[3]
                        rotations.append(rot)
                        
                        # 提取速度 (vx, vz)
                        vel = agent_data[-2:]
                        velocities.append(vel)
            
            return positions, rotations, velocities
            
        except Exception as e:
            logger.warning(f"Failed to get other agents info from observations for env {env_idx}: {e}")
            return [], [], []

    def _compute_orca_velocity(self, current_position: np.ndarray, current_velocity: np.ndarray, 
                             other_agents_pos: list, other_agents_rot: list, other_agents_vel: list,
                             max_speed: float = 0.25, time_horizon: float = 4.0) -> np.ndarray:
        """
        使用ORCA算法计算避障速度
        
        Args:
            current_position: 当前智能体位置 (x, y, z)
            current_velocity: 当前智能体速度 (vx, vz)
            other_agents_pos: 其他智能体位置列表
            other_agents_rot: 其他智能体旋转列表
            other_agents_vel: 其他智能体速度列表
            max_speed: 最大速度
            time_horizon: 预测时间范围
            
        Returns:
            np.ndarray: 调整后的速度向量
        """
        if len(other_agents_pos) == 0:
            return current_velocity
            
        new_velocity = current_velocity.copy()
        combined_avoidance_velocity = np.zeros(2)  # 只考虑x, z平面
        agent_radius = 0.3  # 单个智能体半径
        combined_radius = 2 * agent_radius  # 两个智能体的组合半径
        
        for i in range(len(other_agents_pos)):
            # 计算其他智能体的方向向量
            rotation_radians = other_agents_rot[i]
            direction_vector = np.array([np.sin(rotation_radians), np.cos(rotation_radians)])
            
            # 计算其他智能体的相对速度
            relative_velocity_other = other_agents_vel[i][0] * direction_vector
            
            # 计算相对位置（只考虑x, z平面）
            relative_position = (other_agents_pos[i] - current_position)[[0, 2]]
            relative_velocity = current_velocity - relative_velocity_other
            
            # 计算距离
            distance = np.linalg.norm(relative_position)
            
            # 如果距离太近，使用更强的避障力
            if distance < agent_radius:
                # 计算远离其他智能体的方向
                relative_position_normalized = relative_position / distance
                away_direction = -relative_position_normalized
                # 使用更强的避障力
                avoidance_velocity = away_direction * 0.5  # 强避障力
                combined_avoidance_velocity += avoidance_velocity
                continue
            
            if distance > 0.01:  # 避免除零
                relative_position_normalized = relative_position / distance
                
                # 关键修复：当距离小于组合半径时才需要避障
                if distance < combined_radius:
                    # ORCA避障计算
                    avoidance_velocity = (relative_velocity + 
                                         relative_position_normalized * 
                                         (combined_radius - distance) / time_horizon)
                    combined_avoidance_velocity += avoidance_velocity
        
        # 应用避障速度
        if len(other_agents_pos) > 0:
            adjusted_velocity = new_velocity + combined_avoidance_velocity / len(other_agents_pos)
        else:
            adjusted_velocity = new_velocity
            
        # 限制最大速度
        if np.linalg.norm(adjusted_velocity) > max_speed:
            adjusted_velocity = adjusted_velocity / np.linalg.norm(adjusted_velocity) * max_speed
            
        return adjusted_velocity

    def _compute_orca_velocity_improved(self, current_position: np.ndarray, current_velocity: np.ndarray, 
                                       other_agents_pos: list, other_agents_rot: list, other_agents_vel: list,
                                       max_speed: float = 0.25, time_horizon: float = 8.0,
                                       robot_lin_speed: float = 30.0, pedestrian_lin_speed: float = 12.0) -> np.ndarray:
        """
        改进的ORCA算法：提高预测时间，增强避障能力，考虑速度比例
        
        主要改进：
        1. 预测时间从4秒增加到8秒（更早预测行人行为）
        2. 考虑行人未来轨迹进行避障规划
        3. 更平滑的避障速度计算
        4. 考虑机器人与行人的速度比例（配置中机器人:行人 = 30:12 = 2.5:1）
        
        注意：控制频率相同的情况下，速度比例决定了每步移动距离的比例。
        配置中：
        - 机器人agent0: lin_speed=30.0, ang_speed=31.42 rad/s
        - 行人agent1-6: lin_speed=12.0, ang_speed=30.0 rad/s
        - 速度比例：线速度 2.5:1，角速度 1.05:1
        
        Args:
            current_position: 当前智能体位置 (x, y, z)
            current_velocity: 当前智能体速度 (vx, vz)，单位需要与实际配置一致
            other_agents_pos: 其他智能体位置列表
            other_agents_rot: 其他智能体旋转列表
            other_agents_vel: 其他智能体速度列表，单位需要与实际配置一致
            max_speed: 机器人最大速度（需要与配置中的lin_speed比例一致）
            time_horizon: 预测时间范围（默认8.0秒，比原来提高2倍）
            robot_lin_speed: 机器人线速度配置值（默认30.0，用于计算比例）
            pedestrian_lin_speed: 行人线速度配置值（默认12.0，用于计算比例）
            
        Returns:
            np.ndarray: 调整后的速度向量
        """
        if len(other_agents_pos) == 0:
            return current_velocity
            
        new_velocity = current_velocity.copy()
        combined_avoidance_velocity = np.zeros(2)  # 只考虑x, z平面
        agent_radius = 0.3  # 单个智能体半径
        combined_radius = 2 * agent_radius  # 两个智能体的组合半径
        
        # 计算速度比例（机器人速度 / 行人速度）
        # 配置中：机器人30.0 / 行人12.0 = 2.5:1
        # 控制频率相同，因此机器人每步移动距离是行人的2.5倍
        speed_ratio = robot_lin_speed / pedestrian_lin_speed if pedestrian_lin_speed > 0 else 2.5
        
        # 扩展的预测范围（考虑行人可能的移动范围）
        # 由于机器人速度更快（2.5倍），需要更大的预测缓冲
        extended_radius = combined_radius + 0.5 * (1 + speed_ratio * 0.2)  # 根据速度比例调整缓冲
        
        for i in range(len(other_agents_pos)):
            # 计算其他智能体的方向向量
            rotation_radians = other_agents_rot[i]
            direction_vector = np.array([np.sin(rotation_radians), np.cos(rotation_radians)])
            
            # 计算其他智能体的相对速度
            # 注意：other_agents_vel[i]可能已经是实际速度，或需要转换
            relative_velocity_other = other_agents_vel[i][0] * direction_vector if len(other_agents_vel[i]) > 0 else np.array([0.0, 0.0])
            
            # 预测行人未来位置（基于当前速度和方向）
            # 考虑速度单位：如果速度值是配置值，可能需要缩放
            pedestrian_speed = np.linalg.norm(other_agents_vel[i]) if len(other_agents_vel[i]) > 0 else 0.0
            
            # 注意：如果other_agents_vel已经是标准化速度，需要转换为实际速度
            # 假设传感器返回的速度值与配置值一致（lin_speed=12.0），
            # 但代码中使用的是标准化的m/s单位（约0.12 m/s对应配置的12.0）
            # 这里假设速度值已经是正确的单位，保持一致性
            
            if pedestrian_speed > 0.01:
                # 预测未来时间的位置
                # time_horizon * 0.5 表示保守预测（预测一半的时间窗口）
                prediction_time = time_horizon * 0.5
                
                # 预测未来位置（基于行人的实际速度）
                future_position_2d = other_agents_pos[i][[0, 2]] + direction_vector * pedestrian_speed * prediction_time
                
                # 使用当前位置和未来位置的加权平均（更保守）
                # 由于机器人速度是行人的2.5倍，需要更早预测，所以增加未来位置的权重
                future_weight = 0.3 * min(1.2, 1.0 + (speed_ratio - 1.0) * 0.1)  # 速度比例越大，未来权重越高
                current_weight = 1.0 - future_weight
                predicted_position = current_weight * other_agents_pos[i][[0, 2]] + future_weight * future_position_2d
                predicted_position_3d = np.array([predicted_position[0], other_agents_pos[i][1], predicted_position[1]])
            else:
                # 行人静止，使用当前位置
                predicted_position_3d = other_agents_pos[i]
            
            # 计算相对位置（只考虑x, z平面，使用预测位置）
            relative_position = (predicted_position_3d - current_position)[[0, 2]]
            relative_velocity = current_velocity - relative_velocity_other
            
            # 计算距离
            distance = np.linalg.norm(relative_position)
            
            # 如果距离太近，使用更强的避障力
            if distance < agent_radius:
                # 计算远离其他智能体的方向
                relative_position_normalized = relative_position / (distance + 0.01)  # 避免除零
                away_direction = -relative_position_normalized
                # 使用更强的避障力
                avoidance_velocity = away_direction * 0.6  # 更强的避障力
                combined_avoidance_velocity += avoidance_velocity
                continue
            
            if distance > 0.01:  # 避免除零
                relative_position_normalized = relative_position / distance
                
                # 使用扩展的预测半径，提前避障
                if distance < extended_radius:
                    # ORCA避障计算（使用更长的预测时间）
                    avoidance_velocity = (relative_velocity + 
                                         relative_position_normalized * 
                                         (combined_radius - distance) / time_horizon)
                    # 根据距离调整避障强度（距离越近，避障越强）
                    avoidance_strength = max(0.5, 1.0 - (distance / extended_radius))
                    combined_avoidance_velocity += avoidance_velocity * avoidance_strength
        
        # 应用避障速度
        if len(other_agents_pos) > 0:
            adjusted_velocity = new_velocity + combined_avoidance_velocity / len(other_agents_pos)
        else:
            adjusted_velocity = new_velocity
            
        # 限制最大速度
        velocity_norm = np.linalg.norm(adjusted_velocity)
        if velocity_norm > max_speed:
            adjusted_velocity = adjusted_velocity / velocity_norm * max_speed
        elif velocity_norm < 0.05:  # 如果速度太小，保持最小速度避免卡住
            # 保持方向，但确保有最小速度
            if velocity_norm > 0.01:
                adjusted_velocity = adjusted_velocity / velocity_norm * 0.1
            else:
                # 如果没有速度，使用最后一个非零速度方向或目标方向
                adjusted_velocity = new_velocity if np.linalg.norm(new_velocity) > 0.01 else np.array([0.0, 0.0])
            
        return adjusted_velocity

    def _compute_action_from_oracle_path(self, oracle_path: np.ndarray, env_idx: int, observations: dict = None, global_distance_to_goal: List[float] = None) -> int:
        """
        从Oracle路径计算动作，结合ORCA策略的防卡机制
        
        Args:
            oracle_path: Oracle路径，形状为 (batch_size, 2, 3)
            env_idx: 环境索引
            observations: 观察数据，用于获取其他智能体信息
            global_distance_to_goal: 到目标的距离列表
            
        Returns:
            动作
        """
        try:
            # 处理CUDA张量转换
            if hasattr(oracle_path, 'cpu'):
                oracle_path = oracle_path.cpu().numpy()
            elif hasattr(oracle_path, 'detach'):
                oracle_path = oracle_path.detach().cpu().numpy()
                        
            # 获取当前环境对应的路径点
            current_env_path = oracle_path[env_idx]  # 形状为 (2, 3)
            if len(current_env_path) < 2:
                return 0  # 停止                
                
            # 获取当前智能体状态
            try:
                agent_state = self.envs.call_at(env_idx, "get_agent_state")
                current_position = np.array(agent_state.position)
                current_rotation = agent_state.rotation
            except Exception as e:
                logger.warning(f"Failed to get agent state for env {env_idx}: {e}")
                return 0  # 停止动作
            
            # 从observations中获取其他智能体信息
            other_positions, other_rotations, other_velocities = self._get_other_agents_info_from_observations(observations, env_idx)
            
            # 获取路径中的下一个点
            next_point = current_env_path[1]  # 第二个点，形状为 (3,)
            direction_to_next = next_point - current_position
            distance_to_next = np.linalg.norm(direction_to_next[[0,2]])
            
            # 如果距离很近，尝试使用路径中的第三个点（如果存在）
            if distance_to_next < 0.3 and len(current_env_path) > 2:
                next_point = current_env_path[2]
                direction_to_next = next_point - current_position
            
            # 计算目标角度（参考ORCA策略的实现）
            delta_x = next_point[0] - current_position[0]
            delta_z = next_point[2] - current_position[2]
            target_angle = np.arctan2(-delta_z, delta_x)  # 注意这里是-delta_z
            
            # 获取当前角度
            current_angle = observations['agent_0_localization_sensor'].cpu().numpy()[env_idx, -1]
            
            # 检查是否有其他智能体需要避障
            min_distance = float('inf')
            human_num = 0
            if len(other_positions) > 0:
                for pos in other_positions:
                    distance = np.linalg.norm((current_position - pos)[[0, 2]])
                    min_distance = min(min_distance, distance)
                    human_num += 1
            
            # 如果检测到其他智能体，使用改进的避障策略
            safe_distance_threshold = 2.0  # 合理的预判距离
            if human_num > 0 and min_distance < safe_distance_threshold:
                # 预判行人未来位置（基于当前速度）
                future_positions = []
                for i, (pos, vel) in enumerate(zip(other_positions, other_velocities)):
                    # 预测未来2秒的位置，考虑行人可能的路径变化
                    future_pos = pos + np.array([vel[0], 0, vel[1]]) * 2.0
                    future_positions.append(future_pos)
                    
                    # 计算到未来位置的距离
                    future_distance = np.linalg.norm((current_position - future_pos)[[0, 2]])
                    min_distance = min(min_distance, future_distance)
                
                # 改进的当前速度计算
                if self.prev_actions[env_idx] == 1:  # 如果前一步是前进
                    # 考虑实际的速度和方向
                    current_vel = np.array([np.sin(current_angle), np.cos(current_angle)]) * 0.25
                elif self.prev_actions[env_idx] == 2:  # 左转
                    current_vel = np.array([np.sin(current_angle + np.pi/2), np.cos(current_angle + np.pi/2)]) * 0.1
                elif self.prev_actions[env_idx] == 3:  # 右转
                    current_vel = np.array([np.sin(current_angle - np.pi/2), np.cos(current_angle - np.pi/2)]) * 0.1
                else:
                    current_vel = np.array([0.0, 0.0])
                
                # 使用改进的ORCA算法计算避障速度
                orca_velocity = self._compute_orca_velocity(
                    current_position, current_vel, 
                    other_positions, other_rotations, other_velocities
                )
                
                # 计算ORCA角度
                if np.linalg.norm(orca_velocity) > 0.01:
                    orca_angle = np.arctan2(-orca_velocity[1], orca_velocity[0])
                else:
                    orca_angle = target_angle  # 如果ORCA速度太小，使用原始目标角度
                
                # 改进的权重计算：考虑行人速度和相对方向
                pedestrian_speed = np.linalg.norm(other_velocities[0]) if len(other_velocities) > 0 else 0.0
                relative_direction = np.dot([np.cos(current_angle), np.sin(current_angle)], 
                                          [np.cos(orca_angle), np.sin(orca_angle)])
                
                # 动态权重计算
                if min_distance < 0.3:
                    weight = 0.95  # 非常接近时，几乎完全使用ORCA角度
                elif min_distance < 0.8:
                    weight = 0.8 + 0.1 * pedestrian_speed  # 考虑行人速度
                elif min_distance < 1.5:
                    weight = 0.6 + 0.2 * pedestrian_speed
                else:
                    weight = 0.3 + 0.1 * pedestrian_speed
                
                # 限制权重范围
                weight = max(0.1, min(0.95, weight))
                
                # 计算最终目标角度
                target_angle = orca_angle * weight + (1 - weight) * target_angle
                
                if rank0_only() and env_idx == 0 and self.config.habitat_baselines.verbose:
                    logger.debug(f"Improved avoidance: min_distance={min_distance:.3f}, "
                              f"pedestrian_speed={pedestrian_speed:.3f}, "
                              f"orca_angle={orca_angle:.3f}, target_angle={target_angle:.3f}, "
                              f"weight={weight:.2f}")
            
            # 计算角度差
            self.angle_diff[env_idx] = target_angle - current_angle
            
            # 检查是否到达目标
            if global_distance_to_goal[env_idx] < 1:
                action = 0
            else:
                # 使用ORCA策略的角度判断逻辑
                turn_threshold = np.deg2rad(7.5)  # 与ORCA策略保持一致
                
                # 判断是否需要转向
                if (abs(self.angle_diff[env_idx]) < turn_threshold or 
                    abs(self.angle_diff[env_idx]) > 2 * np.pi - turn_threshold):
                    # 角度差很小，前进
                    action = 1
                    self.repeated_turn_count[env_idx] = 0
                elif ((self.angle_diff[env_idx] > -2 * np.pi + turn_threshold and 
                       self.angle_diff[env_idx] < -np.pi) or 
                      (self.angle_diff[env_idx] > turn_threshold and 
                       self.angle_diff[env_idx] < np.pi)):
                    # 需要左转
                    if self.prev_actions[env_idx] == 3:  # 如果前一步是右转
                        self.repeated_turn_count[env_idx] += 1
                    else:
                        self.repeated_turn_count[env_idx] = 0
                    action = 2  # 左转
                else:
                    # 需要右转
                    if self.prev_actions[env_idx] == 2:  # 如果前一步是左转
                        self.repeated_turn_count[env_idx] += 1
                    else:
                        self.repeated_turn_count[env_idx] = 0
                    action = 3  # 右转
                
                # 改进的防卡机制：考虑安全性的智能防卡
                if self.repeated_turn_count[env_idx] >= self.max_repeated_turns:
                    # 检查前方是否安全
                    if min_distance > 1.0:  # 如果前方安全距离足够
                        action = 1  # 强制前进
                        if rank0_only() and env_idx == 0 and self.config.habitat_baselines.verbose:
                            logger.debug(f"Anti-stuck triggered for env {env_idx}: safe forward after {self.max_repeated_turns} repeated turns")
                    else:  # 如果前方不安全，尝试侧向移动
                        # 选择与行人相反的方向
                        if len(other_positions) > 0:
                            # 计算到最近行人的方向
                            closest_pedestrian = other_positions[0]
                            pedestrian_direction = (closest_pedestrian - current_position)[[0, 2]]
                            pedestrian_angle = np.arctan2(-pedestrian_direction[1], pedestrian_direction[0])
                            
                            # 选择远离行人的转向
                            if abs(pedestrian_angle - current_angle) < np.pi/2:
                                action = 2  # 左转远离行人
                            else:
                                action = 3  # 右转远离行人
                        else:
                            action = 1  # 默认前进
                        
                        if rank0_only() and env_idx == 0 and self.config.habitat_baselines.verbose:
                            logger.debug(f"Anti-stuck triggered for env {env_idx}: lateral movement due to nearby pedestrian")
                    
                    self.repeated_turn_count[env_idx] = 0

            # 最终安全检查：如果动作会导致碰撞，强制停止
            if action == 1:  # 前进动作
                # 检查前方是否有行人
                if len(other_positions) > 0:
                    # 预测前进后的位置
                    forward_position = current_position + np.array([np.sin(current_angle), 0, np.cos(current_angle)]) * 0.25
                    
                    # 检查是否会与行人碰撞
                    for pos in other_positions:
                        collision_distance = np.linalg.norm((forward_position - pos)[[0, 2]])
                        if rank0_only() and env_idx == 0 and self.config.habitat_baselines.verbose:
                            logger.debug(f"Safety check: current_distance={min_distance:.3f}, "
                                      f"forward_distance={collision_distance:.3f}, "
                                      f"threshold=0.8")
                        
                        if collision_distance < 0.8:  # 调整安全距离阈值，避免过于保守
                            action = 0  # 强制停止
                            if rank0_only() and env_idx == 0:
                                logger.debug(f"Safety override: stopped to avoid collision with pedestrian at distance {collision_distance:.3f}")
                            break
            
            # 更新当前环境的历史动作
            self.prev_actions[env_idx] = action
            
            # 调试信息
            if rank0_only() and env_idx == 0 and self.config.habitat_baselines.verbose:
                logger.debug(f"Env {env_idx}: angle_diff={self.angle_diff[env_idx]:.4f}, "
                          f"repeated_turns={self.repeated_turn_count[env_idx]}, "
                          f"prev_action={self.prev_actions[env_idx]}, new_action={action}, "
                          f"min_distance={min_distance:.3f}, distance_to_goal={global_distance_to_goal[env_idx]:.3f}")
            
            return action
                
        except Exception as e:
            logger.error(f"Error computing action from oracle path: {e}")
            return 0  # 停止

    def _compute_action_from_oracle_path_improved(self, oracle_path: np.ndarray, env_idx: int, observations: dict = None, global_distance_to_goal: List[float] = None) -> int:
        """
        改进的避障算法：更保守、更安全，专门针对行人避障优化
        
        主要改进：
        1. 更大的安全距离（1.5-2.0米）
        2. 更长的预测时间（3-4秒）
        3. 智能等待策略（如果行人正在接近，优先等待）
        4. 更保守的前进检查
        5. 考虑行人的移动方向和速度
        6. 到达目标时返回停止动作（action=0），其他情况不返回停止避免episode提前结束
        
        Args:
            oracle_path: Oracle路径，形状为 (batch_size, 2, 3)
            env_idx: 环境索引
            observations: 观察数据，用于获取其他智能体信息
            global_distance_to_goal: 到目标的距离列表
            
        Returns:
            动作（0=停止当到达目标, 1=前进, 2=左转, 3=右转）
        """
        try:
            # 处理CUDA张量转换
            if hasattr(oracle_path, 'cpu'):
                oracle_path = oracle_path.cpu().numpy()
            elif hasattr(oracle_path, 'detach'):
                oracle_path = oracle_path.detach().cpu().numpy()
                        
            # 获取当前环境对应的路径点
            current_env_path = oracle_path[env_idx]  # 形状为 (2, 3)
            if len(current_env_path) < 2:
                return 1  # 如果没有路径，返回前进而不是停止
                
            # 获取当前智能体状态
            try:
                agent_state = self.envs.call_at(env_idx, "get_agent_state")
                current_position = np.array(agent_state.position)
                current_rotation = agent_state.rotation
            except Exception as e:
                logger.warning(f"Failed to get agent state for env {env_idx}: {e}")
                return 1  # 返回前进而不是停止
            
            # 从observations中获取其他智能体信息
            other_positions, other_rotations, other_velocities = self._get_other_agents_info_from_observations(observations, env_idx)
            
            # 获取当前角度
            current_angle = observations['agent_0_localization_sensor'].cpu().numpy()[env_idx, -1]
            
            # 检查是否到达目标（如果到达目标点附近，返回0停止）
            if global_distance_to_goal[env_idx] < 0.5:
                return 0  # 到达目标，停止
            
            # ========== 改进的避障逻辑 ==========
            # 1. 计算到所有行人的距离和相对位置
            pedestrian_info = []
            min_distance = float('inf')
            closest_pedestrian_idx = -1
            
            for i, (pos, rot, vel) in enumerate(zip(other_positions, other_rotations, other_velocities)):
                # 计算距离（只考虑x, z平面）
                distance = np.linalg.norm((current_position - pos)[[0, 2]])
                min_distance = min(min_distance, distance)
                
                # 计算相对位置和方向
                relative_pos = (pos - current_position)[[0, 2]]
                relative_angle = np.arctan2(-relative_pos[1], relative_pos[0])
                
                # 计算行人速度大小
                pedestrian_speed = np.linalg.norm(vel) if len(vel) > 0 else 0.0
                
                # 计算行人移动方向（相对于机器人）
                if pedestrian_speed > 0.01:
                    # 行人的移动方向
                    ped_direction = np.array([np.sin(rot), np.cos(rot)])
                    ped_velocity_vec = ped_direction * pedestrian_speed
                    # 计算行人是否在接近机器人
                    relative_velocity = ped_velocity_vec - np.array([np.sin(current_angle), np.cos(current_angle)]) * 0.25
                    is_approaching = np.dot(relative_velocity, relative_pos / (distance + 0.01)) < 0
                else:
                    is_approaching = False
                
                pedestrian_info.append({
                    'position': pos,
                    'distance': distance,
                    'relative_angle': relative_angle,
                    'speed': pedestrian_speed,
                    'is_approaching': is_approaching,
                    'velocity': vel
                })
                
                if distance == min_distance:
                    closest_pedestrian_idx = i
            
            # 2. 定义安全距离阈值（更保守）
            CRITICAL_DISTANCE = 1.2  # 危险距离：必须转向或等待
            WARNING_DISTANCE = 2.0   # 警告距离：需要谨慎
            SAFE_DISTANCE = 3.0      # 安全距离：可以正常前进
            
            # 3. 如果有行人在危险距离内，转向绕行而不是停止
            if min_distance < CRITICAL_DISTANCE and len(pedestrian_info) > 0:
                closest_ped = pedestrian_info[closest_pedestrian_idx]
                
                # 计算到最近行人的方向
                ped_to_robot = (current_position - closest_ped['position'])[[0, 2]]
                ped_to_robot_norm = ped_to_robot / (np.linalg.norm(ped_to_robot) + 0.01)
                
                # 计算绕行角度（垂直于行人到机器人的方向）
                perpendicular_angle = np.arctan2(-ped_to_robot_norm[1], ped_to_robot_norm[0]) + np.pi/2
                
                # 获取目标方向
                next_point = current_env_path[1]
                target_direction = np.arctan2(
                    next_point[0] - current_position[0],
                    next_point[2] - current_position[2]
                )
                
                # 选择与目标方向更接近的绕行方向
                angle_diff_left = abs(perpendicular_angle - target_direction)
                angle_diff_right = abs(perpendicular_angle + np.pi - target_direction)
                
                if angle_diff_left < angle_diff_right:
                    detour_angle = perpendicular_angle
                else:
                    detour_angle = perpendicular_angle + np.pi
                
                # 计算需要转向的角度
                angle_to_detour = detour_angle - current_angle
                # 归一化到 [-pi, pi]
                while angle_to_detour > np.pi:
                    angle_to_detour -= 2 * np.pi
                while angle_to_detour < -np.pi:
                    angle_to_detour += 2 * np.pi
                
                # 转向绕行
                turn_threshold = np.deg2rad(7.5)
                if abs(angle_to_detour) < turn_threshold:
                    # 角度已经对齐，但距离太近，继续转向
                    if angle_to_detour > 0:
                        self.prev_actions[env_idx] = 2
                        return 2  # 左转
                    else:
                        self.prev_actions[env_idx] = 3
                        return 3  # 右转
                elif angle_to_detour > 0:
                    self.prev_actions[env_idx] = 2
                    return 2  # 左转
                else:
                    self.prev_actions[env_idx] = 3
                    return 3  # 右转
            
            # 4. 如果有行人在警告距离内，使用更保守的策略
            if min_distance < WARNING_DISTANCE and len(pedestrian_info) > 0:
                closest_ped = pedestrian_info[closest_pedestrian_idx]
                
                # 4.1 如果行人正在接近，优先转向而不是前进
                if closest_ped['is_approaching'] and closest_ped['distance'] < 1.8:
                    # 计算绕行方向
                    ped_to_robot = (current_position - closest_ped['position'])[[0, 2]]
                    ped_to_robot_norm = ped_to_robot / (np.linalg.norm(ped_to_robot) + 0.01)
                    perpendicular_angle = np.arctan2(-ped_to_robot_norm[1], ped_to_robot_norm[0]) + np.pi/2
                    
                    next_point = current_env_path[1]
                    target_direction = np.arctan2(
                        next_point[0] - current_position[0],
                        next_point[2] - current_position[2]
                    )
                    
                    angle_diff_left = abs(perpendicular_angle - target_direction)
                    angle_diff_right = abs(perpendicular_angle + np.pi - target_direction)
                    
                    if angle_diff_left < angle_diff_right:
                        detour_angle = perpendicular_angle
                    else:
                        detour_angle = perpendicular_angle + np.pi
                    
                    angle_to_detour = detour_angle - current_angle
                    while angle_to_detour > np.pi:
                        angle_to_detour -= 2 * np.pi
                    while angle_to_detour < -np.pi:
                        angle_to_detour += 2 * np.pi
                    
                    turn_threshold = np.deg2rad(7.5)
                    if abs(angle_to_detour) > turn_threshold:
                        if angle_to_detour > 0:
                            self.prev_actions[env_idx] = 2
                            return 2  # 左转
                        else:
                            self.prev_actions[env_idx] = 3
                            return 3  # 右转
                
                # 4.2 预测行人未来3秒的位置
                prediction_time = 3.0
                future_positions = []
                for idx, ped_info in enumerate(pedestrian_info):
                    if ped_info['speed'] > 0.01:
                        ped_rot = other_rotations[idx]
                        ped_dir = np.array([np.sin(ped_rot), np.cos(ped_rot)])
                        future_pos = ped_info['position'] + np.array([ped_dir[0] * ped_info['speed'] * prediction_time, 
                                                                     0, 
                                                                     ped_dir[1] * ped_info['speed'] * prediction_time])
                        future_positions.append(future_pos)
                    else:
                        future_positions.append(ped_info['position'])
                
                # 4.3 检查前进路径是否会与行人未来位置冲突
                forward_step = 0.25  # 前进一步的距离
                forward_position = current_position + np.array([np.sin(current_angle), 0, np.cos(current_angle)]) * forward_step
                
                # 检查到所有未来位置的最小距离
                min_future_distance = float('inf')
                for future_pos in future_positions:
                    future_dist = np.linalg.norm((forward_position - future_pos)[[0, 2]])
                    min_future_distance = min(min_future_distance, future_dist)
                
                # 如果前进会导致与行人未来位置太近，转向绕行
                if min_future_distance < 1.5:
                    # 计算绕行角度
                    closest_ped_pos = closest_ped['position']
                    ped_to_robot = (current_position - closest_ped_pos)[[0, 2]]
                    ped_to_robot_norm = ped_to_robot / (np.linalg.norm(ped_to_robot) + 0.01)
                    
                    perpendicular_angle = np.arctan2(-ped_to_robot_norm[1], ped_to_robot_norm[0]) + np.pi/2
                    
                    next_point = current_env_path[1]
                    target_direction = np.arctan2(
                        next_point[0] - current_position[0],
                        next_point[2] - current_position[2]
                    )
                    
                    angle_diff_left = abs(perpendicular_angle - target_direction)
                    angle_diff_right = abs(perpendicular_angle + np.pi - target_direction)
                    
                    if angle_diff_left < angle_diff_right:
                        detour_angle = perpendicular_angle
                    else:
                        detour_angle = perpendicular_angle + np.pi
                    
                    angle_to_detour = detour_angle - current_angle
                    while angle_to_detour > np.pi:
                        angle_to_detour -= 2 * np.pi
                    while angle_to_detour < -np.pi:
                        angle_to_detour += 2 * np.pi
                    
                    turn_threshold = np.deg2rad(7.5)
                    if abs(angle_to_detour) < turn_threshold:
                        # 角度已经对齐，但距离太近，继续转向
                        if angle_to_detour > 0:
                            self.prev_actions[env_idx] = 2
                            return 2  # 左转
                        else:
                            self.prev_actions[env_idx] = 3
                            return 3  # 右转
                    elif angle_to_detour > 0:
                        self.prev_actions[env_idx] = 2
                        return 2  # 左转
                    else:
                        self.prev_actions[env_idx] = 3
                        return 3  # 右转
            
            # 5. 正常导航逻辑（没有行人在警告距离内，或距离足够安全）
            # 获取路径中的下一个点
            next_point = current_env_path[1]
            direction_to_next = next_point - current_position
            distance_to_next = np.linalg.norm(direction_to_next[[0,2]])
            
            # 如果距离很近，尝试使用路径中的第三个点（如果存在）
            if distance_to_next < 0.3 and len(current_env_path) > 2:
                next_point = current_env_path[2]
                direction_to_next = next_point - current_position
            
            # 计算目标角度
            delta_x = next_point[0] - current_position[0]
            delta_z = next_point[2] - current_position[2]
            target_angle = np.arctan2(-delta_z, delta_x)
            
            # 计算角度差
            self.angle_diff[env_idx] = target_angle - current_angle
            # 归一化到 [-pi, pi]
            while self.angle_diff[env_idx] > np.pi:
                self.angle_diff[env_idx] -= 2 * np.pi
            while self.angle_diff[env_idx] < -np.pi:
                self.angle_diff[env_idx] += 2 * np.pi
            
            # 6. 最终安全检查：即使距离较远，也要检查前进是否安全
            if min_distance < SAFE_DISTANCE and len(pedestrian_info) > 0:
                forward_position = current_position + np.array([np.sin(current_angle), 0, np.cos(current_angle)]) * 0.25
                for ped_info in pedestrian_info:
                    collision_distance = np.linalg.norm((forward_position - ped_info['position'])[[0, 2]])
                    if collision_distance < 1.2:  # 更保守的安全距离
                        # 如果前进不安全，转向绕行
                        ped_to_robot = (current_position - ped_info['position'])[[0, 2]]
                        ped_to_robot_norm = ped_to_robot / (np.linalg.norm(ped_to_robot) + 0.01)
                        perpendicular_angle = np.arctan2(-ped_to_robot_norm[1], ped_to_robot_norm[0]) + np.pi/2
                        
                        angle_to_detour = perpendicular_angle - current_angle
                        while angle_to_detour > np.pi:
                            angle_to_detour -= 2 * np.pi
                        while angle_to_detour < -np.pi:
                            angle_to_detour += 2 * np.pi
                        
                        turn_threshold = np.deg2rad(7.5)
                        if abs(angle_to_detour) > turn_threshold:
                            if angle_to_detour > 0:
                                self.prev_actions[env_idx] = 2
                                return 2  # 左转
                            else:
                                self.prev_actions[env_idx] = 3
                                return 3  # 右转
            
            # 7. 根据角度差决定动作
            turn_threshold = np.deg2rad(7.5)
            
            if abs(self.angle_diff[env_idx]) < turn_threshold or abs(self.angle_diff[env_idx]) > 2 * np.pi - turn_threshold:
                # 角度差很小，前进
                action = 1
                self.repeated_turn_count[env_idx] = 0
            elif ((self.angle_diff[env_idx] > -2 * np.pi + turn_threshold and 
                   self.angle_diff[env_idx] < -np.pi) or 
                  (self.angle_diff[env_idx] > turn_threshold and 
                   self.angle_diff[env_idx] < np.pi)):
                # 需要左转
                if self.prev_actions[env_idx] == 3:
                    self.repeated_turn_count[env_idx] += 1
                else:
                    self.repeated_turn_count[env_idx] = 0
                action = 2
            else:
                # 需要右转
                if self.prev_actions[env_idx] == 2:
                    self.repeated_turn_count[env_idx] += 1
                else:
                    self.repeated_turn_count[env_idx] = 0
                action = 3
            
            # 8. 防卡机制
            if self.repeated_turn_count[env_idx] >= self.max_repeated_turns:
                if min_distance > 1.5:  # 更保守的安全距离
                    action = 1
                    self.repeated_turn_count[env_idx] = 0
                else:
                    # 前方不安全，继续转向
                    self.repeated_turn_count[env_idx] = 0
            
            # 更新历史动作
            self.prev_actions[env_idx] = action
            
            # 记录专家动作决策到文件（仅在启用失败分析时）
            if rank0_only() and self.config.expert_data_collection.get('enable_failure_analysis', True):
                action_names = ['STOP', 'FORWARD', 'LEFT', 'RIGHT']
                action_name = action_names[action] if 0 <= action < 4 else f'UNKNOWN({action})'
                # 将动作信息存储到环境的临时缓存中，稍后写入文件
                if not hasattr(self, '_episode_action_logs'):
                    self._episode_action_logs = {}
                if env_idx not in self._episode_action_logs:
                    self._episode_action_logs[env_idx] = []
                self._episode_action_logs[env_idx].append(
                    f"  Step {len(self._episode_action_logs[env_idx])+1}: "
                    f"action={action_name}({action}) "
                    f"angle_diff={np.rad2deg(self.angle_diff[env_idx]):.1f}deg "
                    f"dist={global_distance_to_goal[env_idx]:.3f}m "
                    f"min_ped_dist={min_distance:.2f}m"
                )
            
            return action
                
        except Exception as e:
            logger.error(f"Error in improved action computation: {e}")
            import traceback
            traceback.print_exc()
            return 1  # 返回前进而不是停止

    def _compute_action_from_oracle_path_orca_improved(self, oracle_path: np.ndarray, env_idx: int, observations: dict = None, global_distance_to_goal: List[float] = None) -> int:
        """
        基于改进ORCA算法的动作计算：提高预测时间，旋转避障替代停止
        
        主要改进：
        1. 预测时间从3-4秒提高到8秒（更早预测行人行为）
        2. 遇到行人时，使用旋转绕行而不是停止（action=0）
        3. 只有在到达目标时才返回停止动作
        4. 更平滑的避障轨迹
        
        Args:
            oracle_path: Oracle路径，形状为 (batch_size, 2, 3)
            env_idx: 环境索引
            observations: 观察数据，用于获取其他智能体信息
            global_distance_to_goal: 到目标的距离列表
            
        Returns:
            动作（0=停止当到达目标, 1=前进, 2=左转, 3=右转）- 不会因为行人而返回0
        """
        try:
            # 处理CUDA张量转换
            if hasattr(oracle_path, 'cpu'):
                oracle_path = oracle_path.cpu().numpy()
            elif hasattr(oracle_path, 'detach'):
                oracle_path = oracle_path.detach().cpu().numpy()
                        
            # 获取当前环境对应的路径点
            current_env_path = oracle_path[env_idx]  # 形状为 (2, 3)
            if len(current_env_path) < 2:
                return 1  # 如果没有路径，返回前进而不是停止
                
            # 获取当前智能体状态
            try:
                agent_state = self.envs.call_at(env_idx, "get_agent_state")
                current_position = np.array(agent_state.position)
                current_rotation = agent_state.rotation
            except Exception as e:
                logger.warning(f"Failed to get agent state for env {env_idx}: {e}")
                return 1  # 返回前进而不是停止
            
            # 从observations中获取其他智能体信息
            other_positions, other_rotations, other_velocities = self._get_other_agents_info_from_observations(observations, env_idx)
            
            # 获取当前角度
            current_angle = observations['agent_0_localization_sensor'].cpu().numpy()[env_idx, -1]
            
            # 检查是否到达目标（如果到达目标点附近，返回0停止）
            # 使用1.0米作为阈值，与保存时的success_threshold保持一致
            if global_distance_to_goal[env_idx] < self.config.expert_data_collection.get('goal_radius', 3.0):
                if rank0_only() and self.config.expert_data_collection.get('enable_failure_analysis', True):
                    logger.info(f"[EXPERT] env={env_idx} action=STOP(0) reason=REACHED_GOAL dist={global_distance_to_goal[env_idx]:.3f}m")
                return 0  # 到达目标，停止
            
            # 获取路径中的下一个点
            next_point = current_env_path[1]
            delta_x = next_point[0] - current_position[0]
            delta_z = next_point[2] - current_position[2]
            target_angle = np.arctan2(-delta_z, delta_x)
            
            # 计算角度差
            angle_diff_to_target = target_angle - current_angle
            # 归一化到 [-pi, pi]
            while angle_diff_to_target > np.pi:
                angle_diff_to_target -= 2 * np.pi
            while angle_diff_to_target < -np.pi:
                angle_diff_to_target += 2 * np.pi
            
            # ========== 改进的ORCA避障逻辑（8秒预测） ==========
            min_distance = float('inf')
            if len(other_positions) > 0:
                for pos in other_positions:
                    distance = np.linalg.norm((current_position - pos)[[0, 2]])
                    min_distance = min(min_distance, distance)
            
            # 定义距离阈值
            CRITICAL_DISTANCE = 1.0   # 危险距离：必须立即转向
            WARNING_DISTANCE = 2.5    # 警告距离：需要谨慎（比原来更大，因为预测时间更长）
            SAFE_DISTANCE = 4.0       # 安全距离
            
            # 如果有行人在危险距离内，使用改进的ORCA计算避障角度
            if min_distance < WARNING_DISTANCE and len(other_positions) > 0:
                # 计算当前速度（基于前一步动作）
                if self.prev_actions[env_idx] == 1:  # 前进
                    current_vel = np.array([np.sin(current_angle), np.cos(current_angle)]) * 0.25
                elif self.prev_actions[env_idx] == 2:  # 左转
                    current_vel = np.array([np.sin(current_angle + np.pi/2), np.cos(current_angle + np.pi/2)]) * 0.1
                elif self.prev_actions[env_idx] == 3:  # 右转
                    current_vel = np.array([np.sin(current_angle - np.pi/2), np.cos(current_angle - np.pi/2)]) * 0.1
                else:
                    current_vel = np.array([0.0, 0.0])
                
                # 使用改进的ORCA算法（8秒预测）
                # 从配置中读取速度值（考虑速度比例）
                robot_lin_speed, pedestrian_lin_speed = self._get_speed_config()
                # 速度比例：机器人:行人 = 30.0:12.0 = 2.5:1
                # 控制频率相同，机器人每步移动距离是行人的2.5倍
                
                orca_velocity = self._compute_orca_velocity_improved(
                    current_position, current_vel, 
                    other_positions, other_rotations, other_velocities,
                    max_speed=0.25,  # 标准化速度值，实际比例由robot_lin_speed和pedestrian_lin_speed参数考虑
                    time_horizon=8.0,  # 提高到8秒
                    robot_lin_speed=robot_lin_speed,
                    pedestrian_lin_speed=pedestrian_lin_speed
                )
                
                # 计算ORCA角度（不会因为速度太小而停止，而是选择旋转）
                velocity_norm = np.linalg.norm(orca_velocity)
                if velocity_norm > 0.05:
                    # 有有效速度，使用ORCA角度
                    orca_angle = np.arctan2(-orca_velocity[1], orca_velocity[0])
                else:
                    # 速度太小（可能是卡住），计算绕行角度而不是停止
                    # 找到最近的行人
                    closest_idx = 0
                    closest_dist = float('inf')
                    for i, pos in enumerate(other_positions):
                        dist = np.linalg.norm((current_position - pos)[[0, 2]])
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_idx = i
                    
                    # 计算到最近行人的方向
                    closest_ped = other_positions[closest_idx]
                    ped_to_robot = (current_position - closest_ped)[[0, 2]]
                    ped_to_robot_norm = ped_to_robot / (np.linalg.norm(ped_to_robot) + 0.01)
                    
                    # 计算垂直于行人-机器人方向的绕行角度
                    perpendicular_angle = np.arctan2(-ped_to_robot_norm[1], ped_to_robot_norm[0]) + np.pi/2
                    
                    # 选择与目标方向更接近的绕行方向
                    angle_diff_left = abs(perpendicular_angle - target_angle)
                    angle_diff_right = abs(perpendicular_angle + np.pi - target_angle)
                    
                    if angle_diff_left < angle_diff_right:
                        orca_angle = perpendicular_angle
                    else:
                        orca_angle = perpendicular_angle + np.pi
                    
                    # 归一化
                    while orca_angle > np.pi:
                        orca_angle -= 2 * np.pi
                    while orca_angle < -np.pi:
                        orca_angle += 2 * np.pi
                
                # 根据距离动态调整ORCA权重
                if min_distance < CRITICAL_DISTANCE:
                    weight = 0.95  # 非常接近时，几乎完全使用ORCA角度
                elif min_distance < 1.5:
                    weight = 0.85  # 接近时，主要使用ORCA角度
                elif min_distance < WARNING_DISTANCE:
                    weight = 0.7   # 警告距离内，较多使用ORCA角度
                else:
                    weight = 0.5   # 较远时，平衡ORCA和目标方向
                
                # 计算最终目标角度
                final_target_angle = orca_angle * weight + (1 - weight) * target_angle
                # 归一化
                while final_target_angle > np.pi:
                    final_target_angle -= 2 * np.pi
                while final_target_angle < -np.pi:
                    final_target_angle += 2 * np.pi
                
                self.angle_diff[env_idx] = final_target_angle - current_angle
                # 归一化
                while self.angle_diff[env_idx] > np.pi:
                    self.angle_diff[env_idx] -= 2 * np.pi
                while self.angle_diff[env_idx] < -np.pi:
                    self.angle_diff[env_idx] += 2 * np.pi
            else:
                # 没有行人或距离安全，使用原始目标角度
                self.angle_diff[env_idx] = angle_diff_to_target
            
            # ========== 根据角度差决定动作（不使用停止） ==========
            turn_threshold = np.deg2rad(7.5)
            
            if abs(self.angle_diff[env_idx]) < turn_threshold or abs(self.angle_diff[env_idx]) > 2 * np.pi - turn_threshold:
                # 角度差很小，前进
                action = 1
                self.repeated_turn_count[env_idx] = 0
            elif ((self.angle_diff[env_idx] > -2 * np.pi + turn_threshold and 
                   self.angle_diff[env_idx] < -np.pi) or 
                  (self.angle_diff[env_idx] > turn_threshold and 
                   self.angle_diff[env_idx] < np.pi)):
                # 需要左转
                if self.prev_actions[env_idx] == 3:
                    self.repeated_turn_count[env_idx] += 1
                else:
                    self.repeated_turn_count[env_idx] = 0
                action = 2  # 左转
            else:
                # 需要右转
                if self.prev_actions[env_idx] == 2:
                    self.repeated_turn_count[env_idx] += 1
                else:
                    self.repeated_turn_count[env_idx] = 0
                action = 3  # 右转
            
            # ========== 防卡机制：如果卡住，选择旋转而不是停止 ==========
            if self.repeated_turn_count[env_idx] >= self.max_repeated_turns:
                if min_distance > 1.5:
                    # 前方安全，强制前进
                    action = 1
                    self.repeated_turn_count[env_idx] = 0
                else:
                    # 前方不安全，选择远离行人的旋转方向
                    if len(other_positions) > 0:
                        closest_ped = other_positions[0]
                        ped_to_robot = (current_position - closest_ped)[[0, 2]]
                        ped_to_robot_norm = ped_to_robot / (np.linalg.norm(ped_to_robot) + 0.01)
                        perpendicular_angle = np.arctan2(-ped_to_robot_norm[1], ped_to_robot_norm[0]) + np.pi/2
                        
                        angle_to_detour = perpendicular_angle - current_angle
                        while angle_to_detour > np.pi:
                            angle_to_detour -= 2 * np.pi
                        while angle_to_detour < -np.pi:
                            angle_to_detour += 2 * np.pi
                        
                        # 选择旋转方向（不停止）
                        if abs(angle_to_detour) > turn_threshold:
                            action = 2 if angle_to_detour > 0 else 3  # 左转或右转
                    # 如果无法确定，保持当前动作（不改为停止）
                    self.repeated_turn_count[env_idx] = 0
            
            # 更新历史动作
            self.prev_actions[env_idx] = action
            
            # 调试信息
            if rank0_only() and env_idx == 0 and self.config.habitat_baselines.verbose:
                logger.debug(f"ORCA-Improved: min_dist={min_distance:.3f}, time_horizon=8.0s, "
                           f"action={action}, angle_diff={np.rad2deg(self.angle_diff[env_idx]):.1f}deg")
            
            return action
                
        except Exception as e:
            logger.error(f"Error in ORCA-improved action computation: {e}")
            import traceback
            traceback.print_exc()
            return 1  # 返回前进而不是停止

    def _scan_existing_episodes(self, data_folder: str, split: str) -> set:
        """
        扫描已存在的episode，返回(scene_name, episode_id)元组集合
        
        Args:
            data_folder: 数据保存根目录
            split: 数据集分割名称
            
        Returns:
            已存在的(scene_name, episode_id)元组集合
        """
        existing_episodes = set()
        split_path = pathlib.Path(data_folder) / split
        
        if split_path.exists():
            # 遍历场景文件夹
            for scene_dir in split_path.iterdir():
                if scene_dir.is_dir():
                    scene_name = scene_dir.name
                    # 遍历episode文件夹
                    for episode_dir in scene_dir.iterdir():
                        if episode_dir.is_dir():
                            # 检查是否包含必要的文件
                            rgb_dir = episode_dir / "rgb"
                            depth_dir = episode_dir / "depth"
                            action_file = episode_dir / "action" / "0.json"
                            human_num_file = episode_dir / "human_num" / "0.json"
                            distance_to_goal_file = episode_dir / "distance_to_goal" / "0.json"
                            
                            if (rgb_dir.exists() and depth_dir.exists() and 
                                action_file.exists() and human_num_file.exists() and
                                distance_to_goal_file.exists()):
                                # 使用(scene_name, episode_id)作为唯一标识符
                                existing_episodes.add((scene_name, episode_dir.name))
        
        return existing_episodes

    def collect_expert_data(self) -> None:
        """
        简化的专家数据采集方法
        
        不依赖rollout缓冲区，直接执行专家动作并收集数据：
        1. 初始化环境
        2. 遍历episode，执行专家动作
        3. 收集数据并保存
        4. 清理资源
        """
        # 1. 初始化环境
        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        # 2. 获取数据采集配置
        max_episodes = self.config.expert_data_collection.max_episodes
        max_steps_per_episode = self.config.expert_data_collection.max_steps_per_episode
        data_folder = self.config.expert_data_collection.data_folder
        
        # 3. 创建数据保存目录和日志文件
        self.analysis_log_file = None
        if rank0_only():
            os.makedirs(data_folder, exist_ok=True)
            logger.info(f"Data collection started. Max episodes: {max_episodes}, Max steps per episode: {max_steps_per_episode}")
            logger.info(f"Data will be saved to: {data_folder}")
            
            # 创建分析日志文件（如果启用失败分析）
            enable_failure_analysis = self.config.expert_data_collection.get('enable_failure_analysis', True)
            if enable_failure_analysis:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                split_name = self.config.expert_data_collection.split
                self.analysis_log_file = os.path.join(data_folder, f"collection_analysis_{split_name}_{timestamp}.txt")
                with open(self.analysis_log_file, 'w') as f:
                    f.write(f"Data Collection Analysis Log\n")
                    f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Split: {split_name}\n")
                    f.write(f"Max episodes: {max_episodes}\n")
                    f.write(f"Max steps per episode: {max_steps_per_episode}\n")
                    f.write(f"Goal radius: {self.config.expert_data_collection.get('goal_radius', 1.0)}\n")
                    f.write(f"=" * 80 + "\n\n")
                logger.info(f"Analysis log will be saved to: {self.analysis_log_file}")
            
            # 扫描已存在的episode，避免重复保存
            existing_episodes = self._scan_existing_episodes(data_folder, self.config.expert_data_collection.split)
            logger.info(f"Found {len(existing_episodes)} existing episodes, will skip duplicates")

        # 4. 主数据采集循环
        collected_episodes = 0
        total_steps_collected = 0
        skipped_episodes = 0  # 跳过的episode计数
        successful_episodes = 0  # 成功到达目标的episode计数
        failed_episodes = 0  # 未成功到达目标的episode计数
        
        # 按场景统计成功和失败次数
        scene_stats = {}  # {scene_name: {'success': count, 'failed': count}}
        
        # 为每个环境初始化数据存储列表
        episodes_data = [[] for _ in range(self.envs.num_envs)]
        global_distance_to_goal = [np.inf for _ in range(self.envs.num_envs)]
        
        # 用于跟踪已保存的episode ID，避免重复保存
        saved_episode_ids = set()
        
        # 如果存在已扫描的episode，添加到已保存集合中
        if rank0_only() and 'existing_episodes' in locals():
            saved_episode_ids.update(existing_episodes)
        
        # 进度条（仅rank0显示）
        pbar = tqdm(total=max_episodes, desc="Episodes saved", dynamic_ncols=True) if rank0_only() else None
        
        # 检查是否需要保存视频
        save_video_enabled = (
            len(self.config.habitat_baselines.eval.video_option) > 0 
            if hasattr(self.config.habitat_baselines.eval, 'video_option') 
            else False
        )
        if save_video_enabled and rank0_only():
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)
            logger.info(f"Video recording enabled. Videos will be saved to: {self.config.habitat_baselines.video_dir}")
        
        # 为每个环境初始化视频帧列表
        rgb_frames = [[]] * self.envs.num_envs if save_video_enabled else None

        while collected_episodes < max_episodes:
            # 起始提示改为进度条显示，不再额外输出起始info日志
            
            # 4.1 重置环境，开始新的episode（带重试机制）
            max_retries = 3
            retry_count = 0
            observations = None
            
            while retry_count < max_retries:
                try:
                    observations = self.envs.reset()
                    observations = self.envs.post_step(observations)
                    break  # 成功重置，跳出重试循环
                except (ZeroDivisionError, RuntimeError, Exception) as e:
                    retry_count += 1
                    if rank0_only():
                        logger.warning(f"Environment reset failed (attempt {retry_count}/{max_retries}): {e}")
                    
                    if retry_count >= max_retries:
                        if rank0_only():
                            logger.error(f"Environment reset failed after {max_retries} attempts, skipping this episode")
                        # 连续失败后尝试重建向量环境
                        if self._recreate_envs():
                            if rank0_only():
                                logger.info("Vector envs recreated after reset failures. Continuing...")
                            # 重建成功，重新开始外层循环的本轮
                            observations = None
                            break
                        else:
                            if rank0_only():
                                logger.error("Unable to recover from reset failures via env recreation. Skipping episode.")
                            continue
                    else:
                        if rank0_only():
                            logger.info(f"Retrying environment reset...")
                        time.sleep(0.1)  # 短暂等待后重试
                    # 如果是典型的进程通信未读错误，直接尝试重建
                    if isinstance(e, RuntimeError) and "last write has not been read" in str(e):
                        if rank0_only():
                            logger.warning("Detected VectorEnv pipe desync. Attempting to recreate envs immediately.")
                        if self._recreate_envs():
                            if rank0_only():
                                logger.info("Vector envs recreated after pipe desync. Continuing...")
                            observations = None
                            break
                        else:
                            if rank0_only():
                                logger.error("Env recreation after pipe desync failed. Will continue retries.")
            
            if observations is None:
                if rank0_only():
                    logger.error("Failed to reset environment after all retries, skipping this episode")
                continue
            
            # 重置每个环境的前一步动作状态和防卡变量
            self.prev_actions = [0] * self.envs.num_envs
            self.repeated_turn_count = [0] * self.envs.num_envs
            self.angle_diff = [0.0] * self.envs.num_envs
            
            # 4.1.1 检查episode的geodesic_distance，筛选距离大于3米的episode
            valid_episodes = []
            need_recreate = False
            for env_idx in range(self.envs.num_envs):
                try:
                    # 获取完整的episode信息
                    current_episode = self._try_get_current_episode(env_idx)
                    if current_episode is None:
                        # 通信异常时标记重建
                        need_recreate = True
                        raise RuntimeError("current_episode is None")
                    geodesic_distance = current_episode.info.get("geodesic_distance", 0.0) if hasattr(current_episode, 'info') and current_episode.info else 0.0
                    
                    # 检查距离是否有效（避免除零错误）
                    if (geodesic_distance > 5.0 and 
                        not np.isnan(geodesic_distance) and 
                        not np.isinf(geodesic_distance) and
                        geodesic_distance > 0.0 and
                        geodesic_distance < 1000.0):  # 添加上限检查
                        valid_episodes.append(env_idx)
                        if rank0_only() and self.config.habitat_baselines.verbose:
                            logger.debug(f"Env {env_idx}: Episode {current_episode.episode_id} selected, geodesic_distance: {geodesic_distance:.2f}m")
                    else:
                        skipped_episodes += 1
                        if rank0_only() and self.config.habitat_baselines.verbose:
                            if geodesic_distance <= 6.0:
                                logger.debug(f"Env {env_idx}: Episode {current_episode.episode_id} skipped, geodesic_distance: {geodesic_distance:.2f}m <= 3.0m")
                            elif np.isnan(geodesic_distance) or np.isinf(geodesic_distance):
                                logger.debug(f"Env {env_idx}: Episode {current_episode.episode_id} skipped, invalid geodesic_distance: {geodesic_distance}")
                            elif geodesic_distance >= 1000.0:
                                logger.debug(f"Env {env_idx}: Episode {current_episode.episode_id} skipped, geodesic_distance: {geodesic_distance:.2f}m >= 1000.0m")
                            else:
                                logger.debug(f"Env {env_idx}: Episode {current_episode.episode_id} skipped, geodesic_distance: {geodesic_distance:.2f}m")
                except Exception as e:
                    if rank0_only():
                        logger.warning(f"Failed to get episode info for env {env_idx}: {e}")
                    skipped_episodes += 1
                    continue

            if need_recreate:
                # 发生子进程/通信问题，尝试重建环境后继续下一轮
                if self._recreate_envs():
                    if rank0_only():
                        logger.info("Vector envs recreated after failure. Continuing...")
                    continue
                else:
                    if rank0_only():
                        logger.error("Unable to recover from vector env failure. Aborting collection loop.")
                    break
            
            # 如果没有有效的episode，继续下一个循环
            if not valid_episodes:
                if rank0_only() and self.config.habitat_baselines.verbose:
                    logger.debug(f"No valid episodes found, skipping this batch. Skipped episodes: {skipped_episodes}")
                continue
            
            # 重置每个环境的数据存储
            for env_idx in range(self.envs.num_envs):
                episodes_data[env_idx] = []
            
            # 重置视频帧列表（如果启用）
            if save_video_enabled and rgb_frames is not None:
                for env_idx in valid_episodes:
                    rgb_frames[env_idx] = []
                    # 添加初始帧（在reset和post_step之后）
                    try:
                        env_obs = observations[env_idx]
                        # 初始化时，info_dict可能为空，但create_agent0_video_frame会处理这种情况
                        # top_down_map应该通过measurements自动添加到info中
                        # 但由于reset后还没有step，可能没有measurements，所以使用空字典
                        info_dict = {}
                        
                        # 使用自定义函数只显示agent0的第一视角、第三视角和俯视图
                        frame = create_agent0_video_frame(env_obs, info_dict)
                        rgb_frames[env_idx].append(frame)
                    except Exception as e:
                        if rank0_only():
                            logger.warning(f"Failed to add initial frame for env {env_idx}: {e}")
            
            # 初始化每个有效环境的当前distance（obs_0）
            for env_idx in valid_episodes:
                current_episode = self._try_get_current_episode(env_idx)
                if current_episode is None:
                    # 环境不可用，跳过该env的数据
                    if rank0_only():
                        logger.warning(f"Skipping env {env_idx} due to unavailable current_episode during init distance")
                    continue
                geodesic_distance = current_episode.info.get("geodesic_distance", 0.0) if hasattr(current_episode, 'info') and current_episode.info else 0.0
                global_distance_to_goal[env_idx] = geodesic_distance
            
            episode_steps = 0
            episode_done = False
            
            # 初始化活跃环境与完成标记
            dones = [False] * self.envs.num_envs
            active_envs = {env_idx for env_idx in valid_episodes}
            
            # 4.2 Episode内的步骤循环
            while not episode_done and episode_steps < max_steps_per_episode:
            # while not episode_done:
                try:
                    # 4.2.1 获取当前观察（用于计算动作）
                    current_obs = batch_obs(observations, device=self.device)
                    current_obs = apply_obs_transforms_batch(current_obs, self.obs_transforms)
                
                    # 4.2.2 为每个活跃环境计算专家动作，并在step之前保存obs与动作（动作相对obs滞后一帧）
                    actions = [0] * self.envs.num_envs
                    for env_idx in active_envs:
                        expert_action = self._get_expert_action_for_agent_0(env_idx, current_obs, global_distance_to_goal)
                        actions[env_idx] = int(expert_action)
                        # 在step前保存当前观测与即将执行的动作，distance使用当前缓存（对应当前obs）
                        env_obs = observations[env_idx]
                        rgb_data = env_obs.get("agent_0_overhead_front_rgb", None)
                        depth_data = env_obs.get("agent_0_overhead_front_depth", None)
                        third_rgb_data = env_obs.get("agent_0_third_rgb", None)
                        human_num_data = env_obs.get("agent_0_human_num_sensor", None)
                        # 优先从pointgoal传感器读取当前距离，保证与当前obs严格对齐
                        pg = env_obs.get("agent_0_pointgoal_with_gps_compass", None)
                        if pg is not None:
                            try:
                                # 常见格式：tensor/ndarray [dist, angle]
                                if hasattr(pg, "detach"):
                                    pg_np = pg.detach().cpu().numpy()
                                elif hasattr(pg, "cpu"):
                                    pg_np = pg.cpu().numpy()
                                else:
                                    pg_np = np.array(pg)
                                # 单环境: (2,), 多环境: (N,2)
                                if pg_np.ndim == 1:
                                    current_dist = float(pg_np[0])
                                elif pg_np.ndim == 2:
                                    current_dist = float(pg_np[env_idx, 0])
                                else:
                                    current_dist = float(global_distance_to_goal[env_idx])
                            except Exception:
                                current_dist = float(global_distance_to_goal[env_idx])
                        else:
                            current_dist = float(global_distance_to_goal[env_idx])
                        # 同步缓存，供下一处使用
                        global_distance_to_goal[env_idx] = current_dist
                        
                        # 每步动作与距离输出（rank0）
                        if rank0_only():
                            try:
                                msg = f"[STEP] env={env_idx} step={episode_steps} action={actions[env_idx]} dist={current_dist:.3f}"
                                # logger.info(msg)
                                # print(msg)
                            except Exception:
                                pass
                        if rgb_data is not None and depth_data is not None and human_num_data is not None:
                            # 检查行人是否在相机视野内并记录轨迹
                            pedestrians_in_view = 0
                            trajectory_data = {}
                            try:
                                # 获取机器人状态
                                agent_state = self.envs.call_at(env_idx, "get_agent_state")
                                current_position = np.array(agent_state.position)
                                current_rotation = agent_state.rotation
                                
                                # 记录机器人轨迹
                                trajectory_data['robot'] = {
                                    'position': current_position.tolist(),
                                    'rotation': [current_rotation.w, current_rotation.x, 
                                               current_rotation.y, current_rotation.z] if hasattr(current_rotation, 'w') else list(current_rotation)
                                }
                                
                                # 获取实际相机的位置和朝向（用于视野检测）
                                # overhead_front_rgb相机相对于机器人底盘的偏移
                                # cam_offset_pos = (0.166, 0.83, 0.0)
                                # cam_orientation = (0, -1.571, 0.0) - 俯仰角约-90度
                                
                                # 计算相机在世界坐标系中的位置
                                # 需要将相机偏移量从机器人坐标系转换到世界坐标系
                                if hasattr(current_rotation, 'w'):
                                    w, x, y, z = current_rotation.w, current_rotation.x, current_rotation.y, current_rotation.z
                                elif hasattr(current_rotation, 'components'):
                                    w, x, y, z = current_rotation.components
                                else:
                                    w, x, y, z = current_rotation[0], current_rotation[1], current_rotation[2], current_rotation[3]
                                
                                # 构建机器人的旋转矩阵
                                R_robot = np.array([
                                    [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                                    [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                                    [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
                                ])
                                
                                # 相机在机器人坐标系中的偏移
                                cam_offset_local = np.array([0.166, 0.83, 0.0])
                                
                                # 将偏移转换到世界坐标系
                                cam_offset_world = R_robot @ cam_offset_local
                                
                                # 计算相机在世界坐标系中的位置
                                camera_position = current_position + cam_offset_world
                                
                                # 相机朝向：需要叠加机器人朝向和相机相对朝向
                                # cam_orientation = (0, -1.571, 0.0) 表示相机相对于机器人有俯仰角
                                # 这里简化处理，使用机器人的旋转作为相机旋转
                                # 因为我们主要关心水平方向的视野检测
                                camera_rotation = current_rotation
                                
                                # 获取行人位置和信息
                                pedestrian_positions, pedestrian_rotations, pedestrian_velocities = \
                                    self._get_other_agents_info_from_observations(current_obs, env_idx)
                                
                                # 记录所有行人轨迹
                                trajectory_data['pedestrians'] = []
                                for i, (pos, rot, vel) in enumerate(zip(pedestrian_positions, pedestrian_rotations, pedestrian_velocities)):
                                    trajectory_data['pedestrians'].append({
                                        'id': i,
                                        'position': pos.tolist() if isinstance(pos, np.ndarray) else list(pos),
                                        'rotation': float(rot),
                                        'velocity': vel.tolist() if isinstance(vel, np.ndarray) else list(vel)
                                    })
                                
                                # 检查行人是否在相机视野内（使用相机的实际位置和朝向）
                                if len(pedestrian_positions) > 0:
                                    pedestrians_in_view, _ = self._check_pedestrians_in_camera_view(
                                        camera_position, camera_rotation, pedestrian_positions,
                                        max_distance=5.0, fov_horizontal=90.0
                                    )
                            except Exception as e:
                                if rank0_only() and self.config.habitat_baselines.verbose:
                                    logger.warning(f"Failed to check pedestrians in view for env {env_idx}: {e}")
                                pedestrians_in_view = 0
                                trajectory_data = {}
                            
                            step_data = {
                                'rgb': rgb_data,
                                'depth': depth_data,
                                'third_rgb': third_rgb_data,
                                'human_num': human_num_data,
                                'action': actions[env_idx],
                                'distance_to_goal': current_dist,
                                'step': episode_steps,
                                'pedestrian_in_view': pedestrians_in_view,  # 记录视野内的行人数量
                                'trajectory': trajectory_data,  # 记录轨迹数据
                            }
                            # 碰撞信息将在step()之后回填
                            episodes_data[env_idx].append(step_data)
                    
                    # 4.2.3 仅对活跃环境执行动作
                    for env_idx in active_envs:
                        self.envs.async_step_at(env_idx, np.array([actions[env_idx]]))
                    
                    # 4.2.4 收集结果
                    outputs = {}
                    for env_idx in active_envs:
                        outputs[env_idx] = self.envs.wait_step_at(env_idx)
                    
                    # 4.2.5 解包结果
                    # 临时存储每个环境的obs和info，用于post_step后生成视频帧
                    env_obs_for_video = {}
                    env_info_for_video = {}
                    # 存储碰撞信息（如果启用失败分析）
                    enable_failure_analysis = self.config.expert_data_collection.get('enable_failure_analysis', True)
                    
                    for env_idx, output in outputs.items():
                        obs_i, rew_i, done_i, info_i = output
                        # 写回对应索引
                        observations[env_idx] = obs_i
                        dones[env_idx] = done_i
                        
                        # 检查碰撞信息（从step返回的info中获取）并回填到已记录的数据中
                        if enable_failure_analysis and isinstance(info_i, dict) and len(episodes_data[env_idx]) > 0:
                            collisions_key = "collisions"
                            collision_detected = False
                            if collisions_key in info_i:
                                collision_info = info_i[collisions_key]
                                if isinstance(collision_info, dict):
                                    collision_detected = collision_info.get("is_collision", False)
                            # 将碰撞信息回填到最后一步的数据中
                            episodes_data[env_idx][-1]['collision'] = collision_detected
                        
                        # 在step后，尝试从info更新下一帧距离；若无则保留前值
                        if isinstance(info_i, dict):
                            info_dist = info_i.get("distance_to_goal", None)
                            if info_dist is not None:
                                try:
                                    global_distance_to_goal[env_idx] = float(info_dist)
                                except Exception:
                                    pass
                        
                        # 保存obs和info，供post_step后生成视频帧使用
                        if save_video_enabled and rgb_frames is not None and env_idx in active_envs:
                            env_obs_for_video[env_idx] = obs_i
                            env_info_for_video[env_idx] = info_i
                        # 若环境终止，记录终止原因到缓存
                        if done_i and rank0_only():
                            try:
                                reason_parts = []
                                termination_reason = "UNKNOWN"
                                
                                if isinstance(info_i, dict):
                                    # 判断具体的终止原因
                                    if info_i.get("success", False) or info_i.get("episode_success", False):
                                        termination_reason = "SUCCESS"
                                    elif info_i.get("collided", False) or (isinstance(info_i.get("collision"), dict) and info_i["collision"].get("is_collision", False)):
                                        termination_reason = "COLLISION"
                                    elif info_i.get("timeout", False) or info_i.get("truncated", False):
                                        termination_reason = "TIMEOUT"
                                    elif "distance_to_goal" in info_i:
                                        dist = info_i.get("distance_to_goal")
                                        if dist is not None and dist <= 1.0:
                                            termination_reason = "REACHED_GOAL"
                                        else:
                                            termination_reason = "STOPPED"
                                    
                                    # 收集关键信息
                                    candidate_keys = [
                                        "success", "episode_success", "collided", "collision",
                                        "timeout", "truncated", "spl", "num_steps",
                                        "distance_to_goal", "success_reward", "episode_len",
                                    ]
                                    for k in candidate_keys:
                                        if k in info_i:
                                            v = info_i.get(k)
                                            # 简洁打印布尔或数值
                                            if isinstance(v, (bool, int, float, str)):
                                                reason_parts.append(f"{k}={v}")
                                            elif k == "collision" and isinstance(v, dict):
                                                reason_parts.append(f"collision={v.get('is_collision', False)}")
                                
                                dist_str = None
                                try:
                                    dist_str = f"dist={float(global_distance_to_goal[env_idx]):.3f}"
                                except Exception:
                                    pass
                                
                                # 获取最后一个动作
                                last_action = actions[env_idx] if env_idx < len(actions) else -1
                                
                                # 存储终止信息到缓存
                                if not hasattr(self, '_episode_termination_info'):
                                    self._episode_termination_info = {}
                                termination_msg = f"Termination: reason={termination_reason} last_action={last_action} steps={episode_steps}"
                                if reason_parts:
                                    termination_msg += " | " + " ".join(reason_parts)
                                if dist_str is not None:
                                    termination_msg += f" {dist_str}"
                                self._episode_termination_info[env_idx] = termination_msg
                            except Exception as e:
                                logger.warning(f"Failed to log termination reason: {e}")
                        infos_placeholder = info_i  # 保留变量名占位，避免未使用告警
                    
                    # 4.2.6 后处理观察
                    observations = self.envs.post_step(observations)
                    
                    # 4.2.6.1 在post_step之后生成视频帧（确保top_down_map已更新）
                    if save_video_enabled and rgb_frames is not None and len(env_obs_for_video) > 0:
                        for env_idx in env_obs_for_video.keys():
                            if env_idx not in active_envs:
                                continue  # 跳过已完成的环境
                            try:
                                # 使用post_step后的observations（确保是最新的）
                                obs_i = observations[env_idx] if env_idx < len(observations) else env_obs_for_video[env_idx]
                                
                                # 直接使用wait_step_at返回的info_i，它应该已经包含了measurements（包括top_down_map）
                                # measurements会在step过程中自动计算并添加到info中
                                info_i = env_info_for_video.get(env_idx, {})
                                info_dict = info_i if isinstance(info_i, dict) else {}
                                
                                # 调试：检查info_dict中是否有top_down_map
                                if rank0_only() and self.config.habitat_baselines.verbose and episode_steps % 50 == 0:
                                    if "top_down_map" in info_dict:
                                        tdm = info_dict["top_down_map"]
                                        logger.debug(f"env {env_idx} step {episode_steps}: top_down_map found, shape={getattr(tdm, 'shape', type(tdm))}")
                                    else:
                                        logger.debug(f"env {env_idx} step {episode_steps}: top_down_map NOT in info_dict, keys={list(info_dict.keys())[:10]}")
                                
                                # 如果info_dict中没有top_down_map，尝试从observations中获取
                                # 有些measurements可能保存在observations中而不是info中
                                if "top_down_map" not in info_dict:
                                    # 尝试在observations中查找top_down_map相关的key
                                    obs_keys = [k for k in obs_i.keys() if "top_down" in k.lower() or "map" in k.lower()]
                                    if obs_keys and rank0_only() and self.config.habitat_baselines.verbose:
                                        logger.debug(f"top_down_map not in info_dict for env {env_idx}, found observation keys: {obs_keys}")
                                
                                # 使用自定义函数只显示agent0的第一视角、第三视角和俯视图
                                frame = create_agent0_video_frame(obs_i, info_dict)
                                rgb_frames[env_idx].append(frame)
                            except Exception as e:
                                if rank0_only() and self.config.habitat_baselines.verbose:
                                    logger.debug(f"Failed to add frame for env {env_idx} after post_step at step {episode_steps}: {e}")
                    
                    # 清空临时存储
                    env_obs_for_video = {}
                    env_info_for_video = {}
                    
                    # 4.2.7 处理完成环境：从活跃集合移除（不强制覆盖最后一步动作）
                    for env_idx in list(active_envs):
                        if dones[env_idx]:
                            active_envs.remove(env_idx)
                    
                    # 4.2.8 仅当所有有效环境都完成时才结束
                    episode_done = len(active_envs) == 0
                    episode_steps += 1
                    total_steps_collected += 1
                    
                    # 4.2.9 记录进度
                    if rank0_only() and episode_steps % 50 == 0:
                        logger.info(f"Episode {collected_episodes + 1}, Step {episode_steps}/{max_steps_per_episode}")
                        
                except Exception as e:
                    if rank0_only():
                        logger.error(f"Error during episode execution: {e}")
                        logger.info("Skipping this episode due to error")
                    episode_done = True  # 强制结束当前episode
            
            # 4.3 保存完成的episode数据（检查所有轨迹终止的情况）
            for env_idx in valid_episodes:
                # 检查轨迹是否终止（episode结束或有数据需要保存）
                if (dones[env_idx] or episode_done or episode_steps >= max_steps_per_episode) and len(episodes_data[env_idx]) > 0:
                    if rank0_only() and self.config.habitat_baselines.verbose:
                        msg = f"[SAVE-CHECK] env={env_idx} done={dones[env_idx]} episode_done={episode_done} steps={episode_steps} records={len(episodes_data[env_idx])}"
                        logger.info(msg)
                        print(msg)
                    # 获取episode信息
                    current_episode = self._try_get_current_episode(env_idx)
                    if current_episode is None:
                        if rank0_only():
                            logger.warning(f"Missing episode info at save time for env {env_idx}, skipping save for this env")
                        continue
                    ep_id = str(current_episode.episode_id)
                    scene_id = current_episode.scene_id
                    scene_name = pathlib.Path(scene_id).stem
                    
                    # 使用(scene_name, episode_id)作为唯一标识符
                    episode_key = (scene_name, ep_id)
                    
                    # 检查是否已经保存过这个episode
                    if episode_key in saved_episode_ids:
                        if rank0_only():
                            logger.info(f"Skipping duplicate episode {ep_id} (scene: {scene_name}) for env {env_idx}")
                        continue
                    
                    # 检查磁盘上是否已经存在这个episode的数据
                    data_root = pathlib.Path(data_folder) / self.config.expert_data_collection.split / scene_name / ep_id
                    if data_root.exists():
                        if rank0_only():
                            logger.info(f"Episode {ep_id} (scene: {scene_name}) already exists on disk, skipping save for env {env_idx}")
                        saved_episode_ids.add(episode_key)
                        continue
                    
                    # 准备保存的数据
                    episode_rgb = np.array([step['rgb'] for step in episodes_data[env_idx]])
                    episode_depth = np.array([step['depth'] for step in episodes_data[env_idx]])
                    third_list = [step['third_rgb'] for step in episodes_data[env_idx]]
                    episode_third_rgb = np.array(third_list) if all(x is not None for x in third_list) else None
                    episode_human_num = np.array([step['human_num'] for step in episodes_data[env_idx]])
                    episode_actions = [step['action'] for step in episodes_data[env_idx]]
                    episode_distance_to_goal = [step['distance_to_goal'] for step in episodes_data[env_idx]]
                    episode_pedestrian_in_view = [step.get('pedestrian_in_view', 0) for step in episodes_data[env_idx]]
                    episode_trajectories = [step.get('trajectory', {}) for step in episodes_data[env_idx]]
                    global_distance_to_goal[env_idx] = np.inf
                    
                    # 数据一致性检查
                    data_lengths = [
                        len(episode_rgb),
                        len(episode_depth),
                        len(episode_human_num),
                        len(episode_actions),
                        len(episode_distance_to_goal),
                        len(episode_pedestrian_in_view),
                    ]
                    if episode_third_rgb is not None:
                        data_lengths.append(len(episode_third_rgb))
                    
                    if len(set(data_lengths)) > 1:
                        if rank0_only():
                            logger.warning(f"Data length mismatch for env {env_idx}: {data_lengths}")
                            logger.warning("Skipping this episode due to data inconsistency")
                        continue
                    
                    if len(episode_actions) == 0:
                        if rank0_only():
                            logger.warning(f"No actions recorded for env {env_idx}, skipping episode")
                        continue
                    
                    # 检查episode是否成功到达目标
                    final_distance = episode_distance_to_goal[-1] if len(episode_distance_to_goal) > 0 else float('inf')
                    initial_distance = episode_distance_to_goal[0] if len(episode_distance_to_goal) > 0 else float('inf')
                    last_action = episode_actions[-1] if len(episode_actions) > 0 else None
                    success_threshold = self.config.expert_data_collection.get('goal_radius', 1)
                    
                    # 如果最后动作是STOP(0)，说明expert判断到达目标，直接保存，不检查距离阈值
                    if last_action == 0:
                        # Expert返回STOP(0)，信任expert的判断，直接保存
                        pass  # 跳过距离检查，继续执行保存逻辑
                    elif final_distance > success_threshold:
                        failed_episodes += 1
                        # 更新场景统计
                        if scene_name not in scene_stats:
                            scene_stats[scene_name] = {'success': 0, 'failed': 0}
                        scene_stats[scene_name]['failed'] += 1
                        
                        # 详细的失败原因分析（根据配置决定是否启用）
                        enable_failure_analysis = self.config.expert_data_collection.get('enable_failure_analysis', True)
                        if rank0_only():
                            if enable_failure_analysis:
                                # 计算进度
                                distance_progress = initial_distance - final_distance
                                progress_rate = (distance_progress / initial_distance * 100) if initial_distance > 0 else 0
                                
                                # 分析动作分布
                                action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                                for act in episode_actions:
                                    if act in action_counts:
                                        action_counts[act] += 1
                                
                                # 统计碰撞次数
                                collision_count = sum(1 for step in episodes_data[env_idx] if step.get('collision', False))
                                
                                # 分析是否卡住（重复动作）
                                if len(episode_actions) > 10:
                                    last_10_actions = episode_actions[-10:]
                                    unique_last_10 = len(set(last_10_actions))
                                    is_stuck = unique_last_10 <= 2  # 最后10步只有1-2种动作
                                else:
                                    is_stuck = False
                                
                                # 判断失败原因
                                failure_reasons = []
                                if episode_steps >= max_steps_per_episode:
                                    failure_reasons.append("TIMEOUT(达到最大步数)")
                                if collision_count > 0:
                                    failure_reasons.append(f"COLLISION({collision_count}次碰撞)")
                                if progress_rate < 20:
                                    failure_reasons.append(f"LOW_PROGRESS(进度{progress_rate:.1f}%)")
                                if is_stuck:
                                    failure_reasons.append("STUCK(动作重复)")
                                if action_counts[0] > len(episode_actions) * 0.3:
                                    failure_reasons.append(f"TOO_MANY_STOPS({action_counts[0]}次)")
                                if final_distance > initial_distance:
                                    failure_reasons.append("MOVING_AWAY(远离目标)")
                                if not failure_reasons:
                                    failure_reasons.append("UNKNOWN")
                                
                                # 写入分析日志文件
                                if self.analysis_log_file is not None:
                                    try:
                                        with open(self.analysis_log_file, 'a') as f:
                                            f.write(f"\n{'='*80}\n")
                                            f.write(f"FAILED Episode: {ep_id} | Scene: {scene_name}\n")
                                            f.write(f"{'='*80}\n")
                                            f.write(f"Distance: {initial_distance:.2f}m → {final_distance:.2f}m (progress: {progress_rate:.1f}%)\n")
                                            f.write(f"Steps: {len(episode_actions)}/{max_steps_per_episode}\n")
                                            f.write(f"Actions: STOP={action_counts[0]} FWD={action_counts[1]} LEFT={action_counts[2]} RIGHT={action_counts[3]}\n")
                                            f.write(f"Collisions: {collision_count} times\n")
                                            f.write(f"Reasons: {', '.join(failure_reasons)}\n")
                                            f.write(f"Threshold: {success_threshold}m\n\n")
                                            
                                            # 写入专家动作序列
                                            if hasattr(self, '_episode_action_logs') and env_idx in self._episode_action_logs:
                                                f.write(f"Expert Actions:\n")
                                                for action_log in self._episode_action_logs[env_idx]:
                                                    f.write(f"{action_log}\n")
                                                f.write("\n")
                                            
                                            # 写入终止信息
                                            if hasattr(self, '_episode_termination_info') and env_idx in self._episode_termination_info:
                                                f.write(f"{self._episode_termination_info[env_idx]}\n")
                                    except Exception as e:
                                        logger.warning(f"Failed to write to analysis log: {e}")
                                
                                # 清理缓存
                                if hasattr(self, '_episode_action_logs') and env_idx in self._episode_action_logs:
                                    del self._episode_action_logs[env_idx]
                                if hasattr(self, '_episode_termination_info') and env_idx in self._episode_termination_info:
                                    del self._episode_termination_info[env_idx]
                            else:
                                # 简化输出到终端
                                msg = f"[SAVE-SKIP] env={env_idx} ep={ep_id} scene={scene_name} final_dist={final_distance:.3f} threshold={success_threshold}"
                                logger.info(msg)
                        continue
                    # 若成功但最后动作不是0，则追加一个终止步（复用最后观测，动作=0）以保证语义一致
                    if last_action != 0:
                        last_step = episodes_data[env_idx][-1]
                        episodes_data[env_idx].append({
                            'rgb': last_step['rgb'],
                            'depth': last_step['depth'],
                            'third_rgb': last_step.get('third_rgb', None),
                            'human_num': last_step['human_num'],
                            'action': 0,
                            'distance_to_goal': final_distance,
                            'step': last_step.get('step', 0) + 1,
                            'pedestrian_in_view': last_step.get('pedestrian_in_view', 0),
                            'trajectory': last_step.get('trajectory', {}),
                        })
                        # 同步内存数组视图
                        episode_rgb = np.array([step['rgb'] for step in episodes_data[env_idx]])
                        episode_depth = np.array([step['depth'] for step in episodes_data[env_idx]])
                        third_list = [step['third_rgb'] for step in episodes_data[env_idx]]
                        episode_third_rgb = np.array(third_list) if all(x is not None for x in third_list) else None
                        episode_human_num = np.array([step['human_num'] for step in episodes_data[env_idx]])
                        episode_actions = [step['action'] for step in episodes_data[env_idx]]
                        episode_distance_to_goal = [step['distance_to_goal'] for step in episodes_data[env_idx]]
                        episode_pedestrian_in_view = [step.get('pedestrian_in_view', 0) for step in episodes_data[env_idx]]
                        episode_trajectories = [step.get('trajectory', {}) for step in episodes_data[env_idx]]
                    
                    # Episode成功到达目标，准备保存数据
                    if rank0_only():
                        # 计算行人出现统计
                        total_steps = len(episode_pedestrian_in_view)
                        steps_with_pedestrians = sum(1 for x in episode_pedestrian_in_view if x > 0)
                        appearance_frequency = (steps_with_pedestrians / total_steps * 100) if total_steps > 0 else 0
                        max_pedestrians = max(episode_pedestrian_in_view) if episode_pedestrian_in_view else 0
                        
                        save_root = pathlib.Path(data_folder) / self.config.expert_data_collection.split / scene_name / ep_id
                        msg = f"[SAVE] env={env_idx} ep={ep_id} scene={scene_name} status=success final_dist={final_distance:.3f} path={save_root}"
                        logger.info(msg)
                        
                        # 写入成功的episode到分析日志文件
                        enable_failure_analysis = self.config.expert_data_collection.get('enable_failure_analysis', True)
                        if enable_failure_analysis and self.analysis_log_file is not None:
                            try:
                                action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                                for act in episode_actions:
                                    if act in action_counts:
                                        action_counts[act] += 1
                                
                                with open(self.analysis_log_file, 'a') as f:
                                    f.write(f"\n{'='*80}\n")
                                    f.write(f"SUCCESS Episode: {ep_id} | Scene: {scene_name}\n")
                                    f.write(f"{'='*80}\n")
                                    f.write(f"Distance: {initial_distance:.2f}m → {final_distance:.2f}m\n")
                                    f.write(f"Steps: {len(episode_actions)}\n")
                                    f.write(f"Actions: STOP={action_counts[0]} FWD={action_counts[1]} LEFT={action_counts[2]} RIGHT={action_counts[3]}\n")
                                    f.write(f"Pedestrian Stats: {steps_with_pedestrians}/{total_steps} steps ({appearance_frequency:.1f}%), max={max_pedestrians}\n\n")
                                    
                                    # 写入终止信息
                                    if hasattr(self, '_episode_termination_info') and env_idx in self._episode_termination_info:
                                        f.write(f"{self._episode_termination_info[env_idx]}\n")
                                
                                # 清理缓存
                                if hasattr(self, '_episode_action_logs') and env_idx in self._episode_action_logs:
                                    del self._episode_action_logs[env_idx]
                                if hasattr(self, '_episode_termination_info') and env_idx in self._episode_termination_info:
                                    del self._episode_termination_info[env_idx]
                            except Exception as e:
                                logger.warning(f"Failed to write success log: {e}")

                    # 保存到磁盘
                    if rank0_only():
                        save_to_disk(
                            episode_rgb,
                            episode_depth,
                            episode_third_rgb,
                            episode_human_num,
                            episode_actions,
                            episode_distance_to_goal,
                            ep_id,
                            scene_id,
                            pedestrian_in_view=episode_pedestrian_in_view,
                            trajectories=episode_trajectories,
                            split=self.config.expert_data_collection.split,
                            data_folder=data_folder,
                            merge_ep=True
                        )
                        # 简洁成功日志：场景、episode、步数、保存路径
                        try:
                            msg = f"[SAVED] scene={scene_name} ep={ep_id} steps={len(episode_actions)}"
                            logger.info(msg)
                            print(msg)
                        except Exception:
                            pass
                        
                        # 生成并保存视频（如果启用）
                        if save_video_enabled and rgb_frames is not None and len(rgb_frames[env_idx]) > 0:
                            try:
                                # 将视频保存到对应的episode目录
                                generate_video(
                                    video_option=self.config.habitat_baselines.eval.video_option,
                                    video_dir=str(data_root),  # 保存到episode目录
                                    images=rgb_frames[env_idx],
                                    scene_id=scene_name,
                                    episode_id=ep_id,
                                    checkpoint_idx=0,
                                    metrics={},
                                    fps=10,  # 降低帧率，让视频播放更慢
                                    tb_writer=None,
                                    verbose=False,
                                )
                                logger.info(f"[VIDEO] Video saved to episode directory {ep_id}, frames={len(rgb_frames[env_idx])}")
                            except Exception as e:
                                logger.warning(f"[VIDEO] Failed to generate video for episode {ep_id}: {e}")
                    
                    # 记录已保存的episode (使用scene_name和ep_id作为唯一标识符)
                    saved_episode_ids.add(episode_key)
                    collected_episodes += 1
                    successful_episodes += 1
                    # 更新场景统计
                    if scene_name not in scene_stats:
                        scene_stats[scene_name] = {'success': 0, 'failed': 0}
                    scene_stats[scene_name]['success'] += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"saved={successful_episodes} failed={failed_episodes} skipped={skipped_episodes} steps={total_steps_collected}"
                        )
                    
                    if rank0_only():
                        logger.info(f"Completed episode {collected_episodes}/{max_episodes}, "
                                f"steps: {episode_steps}, total steps: {total_steps_collected}")
                        logger.info(f"Saved episode data for env {env_idx}, episode_id: {ep_id}, scene: {scene_name}, "
                                f"final_distance: {final_distance:.3f}")

        # 5. 数据采集完成
        if rank0_only():
            logger.info(f"Data collection completed! Total episodes: {collected_episodes}, Total steps: {total_steps_collected}")
            logger.info(f"Successful episodes (reached goal): {successful_episodes}")
            logger.info(f"Failed episodes (did not reach goal): {failed_episodes}")
            logger.info(f"Skipped episodes (geodesic_distance <= 6.0m): {skipped_episodes}")
            
            # 计算总体成功率
            total_attempted_episodes = successful_episodes + failed_episodes
            if total_attempted_episodes > 0:
                success_rate = (successful_episodes / total_attempted_episodes) * 100
                logger.info(f"Overall success rate: {success_rate:.1f}% ({successful_episodes}/{total_attempted_episodes})")
            else:
                logger.info("No episodes attempted")
            
            # 输出每个场景的成功率统计，并保存到场景目录
            if len(scene_stats) > 0:
                logger.info("=" * 80)
                logger.info("Scene-wise Success Rate Statistics:")
                logger.info("=" * 80)
                # 按场景名称排序
                sorted_scenes = sorted(scene_stats.items())
                for scene_name, stats in sorted_scenes:
                    total_attempted = stats['success'] + stats['failed']
                    if total_attempted > 0:
                        scene_success_rate = (stats['success'] / total_attempted) * 100
                        logger.info(f"Scene: {scene_name:40s} | Success: {stats['success']:4d} | Failed: {stats['failed']:4d} | "
                                  f"Total: {total_attempted:4d} | Success Rate: {scene_success_rate:6.2f}%")
                        
                        # 保存统计信息到场景目录
                        scene_dir = pathlib.Path(data_folder) / self.config.expert_data_collection.split / scene_name
                        if scene_dir.exists():
                            stats_file = scene_dir / "success_rate_stats.json"
                            stats_data = {
                                'scene_name': scene_name,
                                'success_count': stats['success'],
                                'failed_count': stats['failed'],
                                'total_attempted': total_attempted,
                                'success_rate': float(scene_success_rate),
                                'success_rate_percent': f"{scene_success_rate:.2f}%"
                            }
                            try:
                                with open(stats_file, 'w') as f:
                                    json.dump(stats_data, f, indent=2)
                                logger.info(f"  -> Statistics saved to: {stats_file}")
                            except Exception as e:
                                logger.warning(f"  -> Failed to save statistics for scene {scene_name}: {e}")
                    else:
                        logger.info(f"Scene: {scene_name:40s} | No episodes attempted")
                logger.info("=" * 80)
            else:
                logger.info("No scene statistics available")
            
            logger.info(f"Data saved to: {data_folder}")

        # 6. 清理资源
        if pbar is not None:
            pbar.close()
        self.envs.close()

    @profiling_wrapper.RangeContext("collect_data")
    def train(self) -> None:
        """
        专家数据采集的主方法
        
        调用简化的数据采集方法
        """
        self.collect_expert_data()

    def save_checkpoint(self, file_name: str, extra_state: Optional[Dict] = None) -> None:
        """
        保存检查点（数据采集器不需要，但保留接口兼容性）
        """
        pass

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """
        加载检查点（数据采集器不需要，但保留接口兼容性）
        """
        return {}


def get_device(config) -> torch.device:
    """
    获取计算设备
    
    根据配置和CUDA可用性返回适当的计算设备。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")
