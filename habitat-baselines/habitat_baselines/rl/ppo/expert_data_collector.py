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
from typing import Dict, Any, Optional, List
import contextlib

# 第三方库导入
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

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
)
from habitat_baselines.utils.timing import g_timer

# 导入自定义传感器
# import falcon.additional_sensor  # noqa: F401


@baseline_registry.register_trainer(name="expert_data_collector")
class ExpertDataCollector(BaseRLTrainer):
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

        # 检查是否为分布式训练
        self._is_distributed = get_distrib_size()[2] > 1

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

    def _get_expert_action_for_agent_0(self, env_idx: int, observations: Dict[str, Any]) -> Optional[int]:
        """
        为agent_0获取专家动作
        
        使用Habitat内置的Oracle导航功能
        
        Args:
            env_idx: 环境索引
            observations: 当前观察
            
        Returns:
            专家动作，如果无法获取则返回None
        """
        oracle_path_key = "agent_0_oracle_shortest_path_sensor"
        oracle_path = observations[oracle_path_key]
        return self._compute_action_from_oracle_path(oracle_path, env_idx)
            
    def _compute_action_from_oracle_path(self, oracle_path: np.ndarray, env_idx: int) -> int:
        """
        从Oracle路径计算动作
        
        Args:
            oracle_path: Oracle路径，形状为 (batch_size, 2, 3)
            env_idx: 环境索引
            
        Returns:
            动作
        """
        try:
            # 处理CUDA张量转换
            if hasattr(oracle_path, 'cpu'):
                oracle_path = oracle_path.cpu().numpy()
            elif hasattr(oracle_path, 'detach'):
                oracle_path = oracle_path.detach().cpu().numpy()
                        # oracle_path 形状为 (batch_size, 2, 3)，需要根据 env_idx 选择对应的环境
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
            
            # 获取路径中的下一个点
            next_point = current_env_path[1]  # 第二个点，形状为 (3,)
            direction_to_next = next_point - current_position
            distance_to_next = np.linalg.norm(direction_to_next[:2])
            
            # 如果距离很近，尝试使用路径中的第三个点（如果存在）
            if distance_to_next < 0.3 and len(current_env_path) > 2:
                next_point = current_env_path[2]
                direction_to_next = next_point - current_position
            
            # 计算角度差
            # 从四元数中提取yaw角度
            # 四元数格式: quaternion(w, x, y, z)
            w, x, y, z = current_rotation.w, current_rotation.x, current_rotation.y, current_rotation.z
            # 使用四元数到欧拉角转换公式计算yaw
            current_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            
            # direction_to_next 已经是方向向量：next_point - current_position
            # 在Habitat坐标系中，Y轴是高度维度，使用X和Z分量计算yaw角度
            x_component = direction_to_next[0]  # X分量
            z_component = direction_to_next[2]  # Z分量（忽略Y轴高度）
            
            # 计算目标角度
            target_yaw = np.arctan2(z_component, x_component)
            angle_diff = target_yaw - current_yaw
            
            # 标准化角度
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 根据角度差决定动作
            if abs(angle_diff) > 0.15:  # 需要转向
                if angle_diff > 0:
                    return 2  # 左转
                else:
                    return 3  # 右转
            else:  # 方向正确，前进
                return 1  # 前进
                
        except Exception as e:
            logger.error(f"Error computing action from oracle path: {e}")
            return 0  # 停止

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
        
        # 3. 创建数据保存目录
        if rank0_only():
            os.makedirs(data_folder, exist_ok=True)
            logger.info(f"Data collection started. Max episodes: {max_episodes}, Max steps per episode: {max_steps_per_episode}")
            logger.info(f"Data will be saved to: {data_folder}")

        # 4. 主数据采集循环
        collected_episodes = 0
        total_steps_collected = 0
        
        while collected_episodes < max_episodes:
            if rank0_only():
                logger.info(f"Starting episode {collected_episodes + 1}/{max_episodes}")
            
            # 4.1 重置环境，开始新的episode
            observations = self.envs.reset()
            observations = self.envs.post_step(observations)
            
            episode_steps = 0
            episode_done = False
            
            # 4.2 Episode内的步骤循环
            while episode_steps < max_steps_per_episode and not episode_done:
                # 4.2.1 获取当前观察
                current_obs = batch_obs(observations, device=self.device)
                current_obs = apply_obs_transforms_batch(current_obs, self.obs_transforms)
                
                # 4.2.2 为每个环境计算并执行专家动作
                actions = []
                for env_idx in range(self.envs.num_envs):
                    expert_action = self._get_expert_action_for_agent_0(env_idx, current_obs)
                    actions.append(expert_action)
                
                # 4.2.3 执行动作
                for env_idx, action in enumerate(actions):
                    self.envs.async_step_at(env_idx, np.array([action]))
                
                # 4.2.4 收集结果
                outputs = []
                for env_idx in range(self.envs.num_envs):
                    output = self.envs.wait_step_at(env_idx)
                    outputs.append(output)
                
                # 4.2.5 解包结果
                observations, rewards, dones, infos = zip(*outputs)
                observations = list(observations)
                rewards = list(rewards)
                dones = list(dones)
                infos = list(infos)
                
                # 4.2.6 后处理观察
                observations = self.envs.post_step(observations)
                
                # 4.2.7 检查episode是否结束
                episode_done = any(dones)
                episode_steps += 1
                total_steps_collected += 1
                
                # 4.2.8 记录数据（这里可以添加数据保存逻辑）
                if rank0_only() and episode_steps % 50 == 0:
                    logger.info(f"Episode {collected_episodes + 1}, Step {episode_steps}/{max_steps_per_episode}")
            
            # 4.3 记录episode完成
            collected_episodes += 1
            if rank0_only():
                logger.info(f"Completed episode {collected_episodes}/{max_episodes}, "
                          f"steps: {episode_steps}, total steps: {total_steps_collected}")

        # 5. 数据采集完成
        if rank0_only():
            logger.info(f"Data collection completed! Total episodes: {collected_episodes}, Total steps: {total_steps_collected}")
            logger.info(f"Data saved to: {data_folder}")

        # 6. 清理资源
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
