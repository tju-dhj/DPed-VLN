#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Falcon训练器实现

这个模块实现了Falcon算法的训练器，继承自BaseRLTrainer。
Falcon是一个用于多智能体社交导航的强化学习算法，特别针对
人类-机器人交互场景进行了优化。

主要特性：
- 支持多智能体训练（但实际只训练第一个智能体）
- 集成预训练的视觉编码器
- 支持分布式训练
- 包含辅助损失函数（人员计数、位置预测、轨迹预测）
- 兼容Habitat-Lab环境

作者: Meta Platforms
"""

# 标准库导入
import math
import contextlib
import os
import random
import time
from collections import defaultdict, deque
# from typing import TYPE_CHECKING, Dict, List, Optional, Set
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple, Set
# 第三方库导入
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

# Habitat相关导入
import habitat_baselines.rl.multi_agent  # noqa: F401.  # 多智能体支持
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
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
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
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

# 类型检查导入
if TYPE_CHECKING:
    from omegaconf import DictConfig

# PPO和策略相关导入
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)

# 工具函数导入
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    NON_SCALAR_METRICS,
    extract_scalars_from_infos,
)
from habitat_baselines.utils.timing import g_timer

# 导入自定义传感器
import falcon.additional_sensor  # noqa: F401

def contains_inf_or_nan(observations):
    """
    检查观察数据中是否包含无穷大或NaN值
    
    这个函数用于调试目的，帮助检测训练过程中可能出现的数据异常。
    在强化学习训练中，NaN或无穷大值可能导致梯度爆炸或训练不稳定。
    
    Args:
        observations (dict): 包含观察数据的字典
        
    Returns:
        bool: 如果发现NaN或无穷大值返回True，否则返回False
    """
    for key, value in observations.items():
        if isinstance(value, (float, int)):
            # 如果是标量，检查是否为 NaN 或 inf
            if math.isinf(value) or math.isnan(value):
                print(f"Key {key} contains inf or nan: {value}")
                return True
        elif isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            # 如果是列表、数组或张量，检查每个元素是否为 NaN 或 inf
            if isinstance(value, torch.Tensor):
                if torch.isinf(value).any() or torch.isnan(value).any():
                    print(f"Key {key} contains inf or nan in tensor")
                    return True
            elif isinstance(value, np.ndarray):
                if np.isinf(value).any() or np.isnan(value).any():
                    print(f"Key {key} contains inf or nan in numpy array")
                    return True
            else:
                for element in value:
                    if isinstance(element, (float, int)) and (math.isinf(element) or math.isnan(element)):
                        print(f"Key {key} contains inf or nan in list/tuple: {element}")
                        return True
    return False

@baseline_registry.register_trainer(name="collect_datar")
class Collect_data(BaseRLTrainer):
    """
    Falcon算法的训练器类
    
    Falcon是一个用于多智能体社交导航的强化学习算法，特别针对
    人类-机器人交互场景进行了优化。这个训练器继承自BaseRLTrainer，
    实现了PPO算法的训练逻辑，并集成了多智能体支持和预训练视觉编码器。
    
    主要特性：
    - 支持多智能体环境（但只训练第一个智能体）
    - 集成预训练的视觉编码器用于特征提取
    - 支持分布式训练
    - 包含辅助损失函数（人员计数、位置预测、轨迹预测）
    - 兼容Habitat-Lab环境
    
    关键修改：
    - 在多智能体设置中，使用第一个智能体的视觉编码器
    - 简化多智能体训练，只训练第一个智能体
    - 支持静态编码器模式，避免重复计算视觉特征
    """
    
    # 支持的任务类型
    supported_tasks = ["Nav-v0"]

    # 短rollout阈值，用于分布式训练中的早期终止
    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    
    # 类型注解
    _is_distributed: bool
    envs: VectorEnv
    _env_spec: Optional[EnvironmentSpec]

    def __init__(self, config=None):
        """
        初始化Falcon训练器
        
        Args:
            config: 训练配置，包含环境、模型、训练参数等设置
        """
        # 调用父类初始化
        super().__init__(config)

        # 初始化核心组件
        self._agent = None              # 智能体访问管理器
        self.envs = None               # 向量化环境
        self.obs_transforms = []       # 观察变换器列表
        self._is_static_encoder = False # 是否为静态编码器模式
        self._encoder = None           # 视觉编码器
        self._env_spec = None          # 环境规范

        # 检查是否为分布式训练
        # 如果分布式世界大小大于1，则启用分布式模式
        self._is_distributed = get_distrib_size()[2] > 1

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """
        分布式训练中的All-Reduce操作辅助方法
        
        在分布式训练中，需要将各个进程的梯度或统计信息进行聚合。
        这个方法将张量移动到正确的设备上，执行all-reduce操作，
        然后移回原始设备。
        
        Args:
            t (torch.Tensor): 需要执行all-reduce操作的张量
            
        Returns:
            torch.Tensor: 经过all-reduce操作的张量
        """
        # 如果不是分布式训练，直接返回原张量
        if not self._is_distributed:
            return t

        # 保存原始设备
        orig_device = t.device
        # 移动到训练设备
        t = t.to(device=self.device)
        # 执行all-reduce操作
        torch.distributed.all_reduce(t)
        # 移回原始设备
        return t.to(device=orig_device)

    def _create_obs_transforms(self):
        """
        创建观察变换器
        
        根据配置创建观察变换器列表，并更新环境规范中的观察空间。
        观察变换器用于预处理观察数据，例如归一化、裁剪等。
        """
        # 获取激活的观察变换器
        self.obs_transforms = get_active_obs_transforms(self.config)
        # 应用变换器到观察空间
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """
        创建智能体访问管理器
        
        设置AgentAccessMgr，这是智能体与训练器之间的接口。
        注意：调用此方法后，还需要调用agent.post_init()来完成初始化。
        此方法只构造对象，不进行完整的初始化。
        
        Args:
            resume_state: 恢复状态，用于从检查点恢复训练
            **kwargs: 额外的关键字参数
            
        Returns:
            AgentAccessMgr: 智能体访问管理器实例
        """
        # 创建观察变换器
        self._create_obs_transforms()
        
        # 从注册表获取智能体访问管理器类型并实例化
        return baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type  # 通常是"MultiAgentAccessMgr"
        )(
            config=self.config,
            env_spec=self._env_spec,
            is_distrib=self._is_distributed,
            device=self.device,
            resume_state=resume_state,
            num_envs=self.envs.num_envs,
            percent_done_fn=self.percent_done,
            **kwargs,
        )

    def _init_envs(self, config=None, is_eval: bool = False):
        """
        初始化向量化环境
        
        创建多个并行环境实例，用于高效的数据收集。
        在分布式训练中，每个进程负责一部分环境。
        
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
            workers_ignore_signals=is_slurm_batch_job(),  # 在SLURM批处理作业中忽略信号
            enforce_scenes_greater_eq_environments=is_eval,  # 评估时确保场景数>=环境数
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),  # 是否为第一个进程
        )

        # 创建环境规范，描述观察空间和动作空间
        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )

        # 设置只在rank0上记录的度量键
        # 这些度量将排除在所有其他工作进程之外，只从单个工作进程报告
        self._rank0_keys: Set[str] = set(
            list(self.config.habitat.task.rank0_env0_measure_names)
            + list(self.config.habitat.task.rank0_measure_names)
        )

        # 在`self._rank0_keys`中声明的度量信息
        # 这些信息与`self.window_episode_stats`分开记录
        self._single_proc_infos: Dict[str, List[float]] = {}

    def _init_train(self, resume_state=None):
        """
        初始化训练过程
        
        这是训练器初始化的核心方法，负责：
        1. 处理恢复状态和配置
        2. 设置分布式训练
        3. 初始化环境和智能体
        4. 设置观察编码器
        5. 初始化统计信息
        
        Args:
            resume_state: 恢复状态，用于从检查点恢复训练
        """
        # 1. 处理恢复状态
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            # 检查是否允许加载恢复状态配置
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            # 使用恢复状态中的配置
            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        # 2. 强制分布式训练设置
        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        # 3. 添加抢占信号处理器（用于SLURM作业管理）
        self._add_preemption_signal_handlers()

        # 4. 分布式训练设置
        if self._is_distributed:
            # 初始化分布式SLURM环境
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            
            # 在rank0上记录分布式初始化信息
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            # 配置分布式训练参数
            with read_write(self.config):
                # 设置GPU设备ID
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
                
                # 为每个进程设置唯一的随机种子
                # 乘以环境数量确保每个环境也有唯一的种子
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            # 设置所有随机数生成器的种子
            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            
            # 创建分布式存储用于跟踪rollout完成情况
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        # 5. 记录配置信息（仅在rank0和详细模式下）
        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        # 6. 配置性能分析器
        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # 7. 移除非标量度量（这些度量只能在评估时使用）
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(
                        non_scalar_metric_root
                    )
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                    )

        # 8. 初始化环境
        self._init_envs()

        # 9. 获取计算设备
        self.device = get_device(self.config)

        # 10. 创建检查点目录（仅在rank0上）
        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        # 11. 添加日志文件处理器
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        # 12. 创建智能体访问管理器
        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.init_distributed(find_unused_params=False)  # type: ignore
        self._agent.post_init()

        # 13. 设置编码器模式
        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        # 14. 获取初始观察并处理
        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        # 15. 关键修改：处理视觉编码器
        # 这是Falcon训练器与原始PPO训练器的主要区别
        if self._is_static_encoder:
            # 尝试获取智能体的视觉编码器
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                # 如果智能体没有直接的视觉编码器，使用第一个智能体的编码器
                # 这是多智能体设置中的关键修改
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode():
                    # 移除'agent_0_'前缀，因为编码器期望的输入格式不同
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                # 如果智能体有直接的视觉编码器，直接使用
                with inference_mode():
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)
        
        # 16. 插入初始观察到rollout存储
        self._agent.rollouts.insert_first_observations(batch)

        # 17. 初始化统计信息
        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )

        # 18. 记录训练开始时间
        self.t_start = time.time()

    
    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        """
        保存检查点
        
        将当前训练状态保存到文件，包括：
        - 智能体状态（模型权重、优化器状态等）
        - 训练配置
        - 额外的状态信息（如步数、时间等）
        
        同时保存指定名称的检查点和"latest.pth"文件。
        
        Args:
            file_name: 检查点文件名
            extra_state: 额外的状态信息，如训练步数、时间等
        """
        # 构建检查点字典
        checkpoint = {
            **self._agent.get_save_state(),  # 智能体状态
            "config": self.config,           # 训练配置
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state  # type: ignore

        # 保存指定名称的检查点
        save_file_path = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, file_name
        )
        torch.save(checkpoint, save_file_path)
        
        # 同时保存为"latest.pth"（用于恢复训练）
        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, "latest.pth"
            ),
        )
        
        # 调用保存回调函数（如果配置了）
        if self.config.habitat_baselines.on_save_ckpt_callback is not None:
            hydra.utils.call(
                self.config.habitat_baselines.on_save_ckpt_callback,
                save_file_path=save_file_path,
            )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """
        加载检查点
        
        从指定路径加载检查点文件，返回包含检查点信息的字典。
        
        Args:
            checkpoint_path: 检查点文件路径
            *args: 额外的位置参数，传递给torch.load
            **kwargs: 额外的关键字参数，传递给torch.load

        Returns:
            dict: 包含检查点信息的字典
        """
        return torch.load(checkpoint_path, *args, **kwargs)
    
    def _quaternion_to_yaw(self, quaternion) -> float:
        """
        从四元数中提取yaw角度
        
        Args:
            quaternion: 四元数对象，格式为 quaternion(w, x, y, z)
            
        Returns:
            float: yaw角度（弧度）
        """
        try:
            # 如果quaternion有属性，直接访问
            if hasattr(quaternion, 'w') and hasattr(quaternion, 'x') and hasattr(quaternion, 'y') and hasattr(quaternion, 'z'):
                w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
            # 如果是元组或列表
            elif isinstance(quaternion, (tuple, list)) and len(quaternion) == 4:
                w, x, y, z = quaternion
            # 如果是numpy数组
            elif hasattr(quaternion, '__getitem__') and len(quaternion) == 4:
                w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            else:
                # 尝试直接访问索引
                w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            
            # 计算yaw角度
            # 使用标准四元数到欧拉角转换公式
            # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            
            return yaw
            
        except Exception as e:
            logger.warning(f"Error converting quaternion to yaw: {e}, quaternion: {quaternion}")
            return 0.0  # 返回默认值

    ###new new new
    def _get_expert_action_for_agent_0(self, env_idx: int, observations: Dict[str, Any]) -> Optional[int]:
        """
        为agent_0获取专家动作
        
        使用Habitat内置的Oracle导航功能，考虑动态行人避障
        
        Args:
            env_idx: 环境索引
            observations: 当前观察
            
        Returns:
            专家动作，如果无法获取则返回None
        """ #使用Oracle最短路径传感器
        oracle_path_key = "agent_0_oracle_shortest_path_sensor"
        oracle_path = observations[oracle_path_key]
        return self._compute_action_from_oracle_path(oracle_path, env_idx)
            
    ###new new new
    def _compute_action_from_oracle_path(self, oracle_path: np.ndarray, env_idx: int) -> int:
        """
        从Oracle路径计算动作
        
        Args:
            oracle_path: Oracle路径，形状为 (2, 3)
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
            
            if len(oracle_path) < 2:
                return 0  # 停止
            
            # 获取当前智能体状态
            try:
                agent_state = self.envs.call_at(env_idx, "get_agent_state")
                current_position = np.array(agent_state.position)
                current_rotation = agent_state.rotation
            except Exception as e:
                logger.warning(f"Failed to get agent state for env {env_idx}: {e}")
                return 0  # 停止动作
            
            # oracle_path 形状为 (batch_size, 2, 3)，需要根据 env_idx 选择对应的环境
            # 获取当前环境对应的路径点
            current_env_path = oracle_path[env_idx]  # 形状为 (2, 3)
            
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
            # 使用四元数到欧拉角转换公式计算yaw# yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
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
    
    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        """
        计算动作并执行环境步进
        
        这是训练循环中的核心方法，负责：
        1. 从rollout存储中获取当前观察
        2. 使用策略网络计算动作
        3. 将动作应用到环境中
        4. 将动作数据存储到rollout中
        
        Args:
            buffer_index: 缓冲区索引，用于多缓冲区设置
        """
        # 计算环境切片（在多缓冲区设置中分配环境）
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        # 1. 动作采样阶段
        with g_timer.avg_time("trainer.sample_action"), inference_mode():
            # 从rollout存储中获取当前步骤的数据
            step_batch = self._agent.rollouts.get_current_step(
                env_slice, buffer_index
            )

            profiling_wrapper.range_push("compute actions")

            # 提取长度信息（用于处理变长序列）
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }
# 调试：检查数据中是否包含NaN或无穷大值（已注释）
            # contains_inf_or_nan(step_batch["observations"])
            # contains_inf_or_nan(step_batch["recurrent_hidden_states"])
            # contains_inf_or_nan(step_batch["prev_actions"])
            # contains_inf_or_nan(step_batch["masks"])
            
            # 使用策略网络计算动作
            action_data = self._agent.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        profiling_wrapper.range_pop()  # compute actions
            
        # 2. 环境步进阶段
        with g_timer.avg_time("trainer.obs_insert"):
            for index_env, act in zip(
                range(env_slice.start, env_slice.stop),
                action_data.env_actions.cpu().unbind(0),
            ):
                # 关键修改：根据智能体类型和动作空间类型处理动作
                if hasattr(self._agent, '_agents') and self._agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                    # 多智能体设置中的分类动作
                    act = act.numpy()
                elif is_continuous_action_space(self._env_spec.action_space):
                    # 连续动作空间：裁剪到指定范围
                    act = np.clip(
                        act.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                else:
                    # 离散动作空间：转换为标量
                    act = act.item()
                 # 为agent_0获取专家动作
                current_obs = step_batch["observations"]
                # current_obs = self.envs.call_at(index_env, "get_observations")
                expert_action = self._get_expert_action_for_agent_0(index_env, current_obs)
                expert_action = np.array([expert_action])
                # expert_action = np.array([1])
                
                # 异步执行环境步进
                self.envs.async_step_at(index_env, expert_action)
                observations = self.envs.wait_step_at(index_env)
                # print("**************************************************************")
                # self.envs.step_at(index_env, expert_action)

    def _collect_environment_result(self, buffer_index: int = 0):
        """
        收集环境执行结果
        
        这是训练循环中的另一个核心方法，负责：
        1. 等待环境执行完成并收集结果
        2. 处理观察、奖励、完成状态等信息
        3. 更新统计信息
        4. 处理视觉编码器（关键修改）
        5. 将结果存储到rollout中
        
        Args:
            buffer_index: 缓冲区索引，用于多缓冲区设置
            
        Returns:
            int: 处理的环境步数
        """
        # 计算环境切片
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        # 1. 等待环境执行完成
        with g_timer.avg_time("trainer.step_env"):
            # 等待所有环境完成步进
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            # 解包结果
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

        # 2. 处理环境结果
        with g_timer.avg_time("trainer.update_stats"):
            # 后处理观察 观察一下区别
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            # 处理奖励
            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            # 处理完成掩码
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            # 更新episode奖励和统计信息
            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

            # 提取单进程信息（只在rank0上记录）
            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            
            # 提取其他度量信息
            extracted_infos = extract_scalars_from_infos(
                infos, ignore_keys=self._rank0_keys
            )
            for k, v_k in extracted_infos.items():
                v = torch.tensor(
                    v_k,
                    dtype=torch.float,
                    device=self.current_episode_reward.device,
                ).unsqueeze(1)
                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["count"]
                    )
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

            # 重置已完成的episode的奖励
            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

        # 3. 关键修改：处理视觉编码器
        # 这是Falcon训练器与原始PPO训练器的主要区别
        if self._is_static_encoder:
            # 尝试获取智能体的视觉编码器
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                # 如果智能体没有直接的视觉编码器，使用第一个智能体的编码器
                # 这是多智能体设置中的关键修改
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    # 移除'agent_0_'前缀，因为编码器期望的输入格式不同
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp)
            else:
                # 如果智能体有直接的视觉编码器，直接使用
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch)
        
        # 4. 将结果存储到rollout中
        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        # 5. 推进rollout到下一步
        self._agent.rollouts.advance_rollout(buffer_index)

        # 6. 返回处理的环境步数
        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        """
        收集一个rollout步骤
        
        这是经验收集的完整步骤，包括动作计算和环境步进。
        
        Returns:
            int: 处理的环境步数
        """
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        """
        更新智能体
        
        这是PPO算法的核心更新步骤，包括：
        1. 计算下一状态的价值估计
        2. 计算GAE回报
        3. 执行PPO更新
        4. 后处理更新
        
        Returns:
            dict: 包含各种损失的字典
        """
        # 1. 计算下一状态的价值估计
        with inference_mode():
            step_batch = self._agent.rollouts.get_last_step()
            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }

            next_value = self._agent.actor_critic.get_value(
                step_batch["observations"],
                step_batch.get("recurrent_hidden_states", None),
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        # 2. 计算GAE回报
        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )

        # 3. 设置为训练模式并执行PPO更新
        self._agent.train()
        losses = self._agent.updater.update(self._agent.rollouts)

        # 4. 后处理更新
        self._agent.rollouts.after_update()
        self._agent.after_update()

        return losses

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        """
        合并后处理步骤
        
        在训练更新后处理统计信息和损失，包括：
        1. 聚合所有进程的统计信息
        2. 更新窗口统计信息
        3. 在分布式训练中同步损失和步数
        4. 更新总步数
        
        Args:
            losses: 当前更新的损失字典
            count_steps_delta: 本次更新的步数增量
            
        Returns:
            Dict[str, float]: 处理后的损失字典
        """
        # 1. 聚合统计信息
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        # 在分布式训练中执行all-reduce操作
        stats = self._all_reduce(stats)

        # 2. 更新窗口统计信息
        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        # 3. 分布式训练中的损失同步
        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            # 同步损失和步数
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            # 平均化损失（除以进程数）
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        # 4. 重置分布式rollout计数器
        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        # 5. 更新总步数
        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        """
        记录训练日志
        
        在rank0上记录训练过程中的各种指标和统计信息，包括：
        1. 计算窗口统计信息的变化量
        2. 记录奖励、损失、性能指标到TensorBoard
        3. 定期输出详细的训练统计信息到日志
        
        Args:
            writer: TensorBoard写入器
            losses: 当前更新的损失字典
            prev_time: 之前的时间（用于恢复训练时的时间计算）
        """
        # 1. 计算窗口统计信息的变化量
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()  # 计算窗口内第一个和最后一个值的差
                if len(v) > 1
                else v[0].sum().item()       # 如果只有一个值，直接使用
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)  # 确保count至少为1，避免除零

        # 2. 记录奖励到TensorBoard
        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # 3. 记录其他度量指标
        # 检查是否还有其他未记录的度量
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        # 记录度量指标到TensorBoard
        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        
        # 记录损失到TensorBoard
        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        # 记录单进程信息（只在rank0上记录）
        for k, v in self._single_proc_infos.items():
            writer.add_scalar(k, np.mean(v), self.num_steps_done)

        # 4. 计算并记录性能指标
        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # 记录FPS到TensorBoard
        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        # 记录各种计时器的性能统计
        for timer_name, timer_val in g_timer.items():
            writer.add_scalar(
                f"perf/{timer_name}",
                timer_val.mean,
                self.num_steps_done,
            )

        # 5. 定期输出详细统计信息到日志
        if (
            self.num_updates_done % self.config.habitat_baselines.log_interval
            == 0
        ):
            # 输出基本训练信息
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                f"Num updates: {self.num_updates_done}\tNum frames {self.num_steps_done}"
            )

            # 输出窗口统计信息
            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )
            
            # 输出性能统计信息
            perf_stats_str = " ".join(
                [f"{k}: {v.mean:.3f}" for k, v in g_timer.items()]
            )
            logger.info(f"\tPerf Stats: {perf_stats_str}")
            
            # 输出单进程信息（如果配置了）
            if self.config.habitat_baselines.should_log_single_proc_infos:
                for k, v in self._single_proc_infos.items():
                    logger.info(f" - {k}: {np.mean(v):.3f}")

    def should_end_early(self, rollout_step) -> bool:
        """
        判断是否应该提前结束rollout
        
        在分布式训练中，为了避免某些进程成为"拖后腿"的进程，
        当满足条件时会提前结束rollout以保持同步。
        
        Args:
            rollout_step: 当前rollout步骤数
            
        Returns:
            bool: 如果应该提前结束返回True，否则返回False
        """
        # 非分布式训练不需要提前结束
        if not self._is_distributed:
            return False
            
        # 这是工作进程抢占发生的地方。
        # 如果工作进程检测到自己将成为"拖后腿"的进程，它会抢占自己！
        return (
            # 检查是否已经执行了足够的步骤（达到阈值的25%）
            rollout_step
            >= self.config.habitat_baselines.rl.ppo.num_steps
            * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            # 检查是否已经有足够多的进程完成了rollout
            self.config.habitat_baselines.rl.ddppo.sync_frac
            * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        """
        Falcon训练的主方法
        
        这是Falcon训练器的核心训练循环，实现了完整的PPO训练流程：
        1. 初始化训练环境和智能体
        2. 处理恢复状态（从检查点恢复训练）
        3. 主训练循环：
           - 预滚动准备
           - 经验收集（rollout）
           - 智能体更新（PPO更新）
           - 日志记录和检查点保存
        4. 清理资源
        
        训练循环支持：
        - 分布式训练
        - 检查点保存和恢复
        - 性能监控和日志记录
        - 优雅的中断处理
        """
        # 1. 初始化训练
        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        # 2. 初始化训练状态变量
        count_checkpoints = 0  # 检查点计数器
        prev_time = 0         # 之前的时间（用于恢复训练）

        # 3. 分布式训练同步
        if self._is_distributed:
            torch.distributed.barrier()

        # 4. 处理恢复状态
        resume_run_id = None
        if resume_state is not None:
            # 加载智能体状态
            self._agent.load_state_dict(resume_state)

            # 恢复训练统计信息
            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            # 恢复episode统计信息
            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )
            resume_run_id = requeue_stats.get("run_id", None)

        # 5. 设置TensorBoard写入器（仅在rank0上）
        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            # 6. 主训练循环
            while not self.is_done():
                # 开始性能分析
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                # 6.1 预滚动准备
                self._agent.pre_rollout()

                # 6.2 保存恢复状态（用于SLURM作业管理）
                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )

                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                # 6.3 检查退出信号（用于SLURM作业管理）
                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update
                    self.envs.close()
                    requeue_job()
                    return

                # 6.4 设置为评估模式并开始经验收集
                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                # 6.5 经验收集阶段
                profiling_wrapper.range_push("_collect_rollout_step")
                with g_timer.avg_time("trainer.rollout_collect"):
                    # 计算初始动作（为所有缓冲区）
                    for buffer_index in range(self._agent.nbuffers):
                        self._compute_actions_and_step_envs(buffer_index)

                    # 执行rollout步骤循环
                    for step in range(self._ppo_cfg.num_steps):
                        # 判断是否为最后一步
                        is_last_step = (
                            self.should_end_early(step + 1)  # 分布式训练中的提前结束
                            or (step + 1) == self._ppo_cfg.num_steps  # 正常结束
                        )

                        # 收集环境结果（为所有缓冲区）
                        for buffer_index in range(self._agent.nbuffers):
                            count_steps_delta += (
                                self._collect_environment_result(buffer_index)
                            )

                            # 性能分析管理
                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_pop()  # _collect_rollout_step

                            # 如果不是最后一步，计算下一步的动作
                            if not is_last_step:
                                if (buffer_index + 1) == self._agent.nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(
                                    buffer_index
                                )

                        # 如果是最后一步，跳出循环
                        if is_last_step:
                            break

                profiling_wrapper.range_pop()  # rollouts loop

                # 6.6 分布式训练同步
                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                # 6.7 智能体更新（PPO核心）
                losses = self._update_agent()

                # 6.8 后处理更新
                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                # 6.9 记录训练日志
                self._training_log(writer, losses, prev_time)

                # 6.10 保存检查点
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    print(f'PPO save to ckpt.{count_checkpoints}.pth ')
                    count_checkpoints += 1

                # 结束性能分析
                profiling_wrapper.range_pop()  # train update

            # 7. 清理资源
            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """
        评估单个检查点
        
        加载指定的检查点并在评估环境中运行智能体，记录评估结果。
        支持视频录制和详细的评估指标记录。
        
        Args:
            checkpoint_path: 检查点文件路径
            writer: TensorBoard写入器，用于记录评估结果
            checkpoint_index: 检查点索引，用于日志记录
        """
        # 1. 检查分布式模式
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # 2. 加载检查点
        # 某些配置可能不需要加载检查点，例如使用分层策略时
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # 使用CPU加载检查点通常比直接映射到CUDA设备更好
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"Loaded checkpoint trained for {step_id} steps")
        else:
            ckpt_dict = {"config": None}

        # 3. 处理配置
        if "config" not in ckpt_dict:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        # 4. 配置视频录制（如果需要）
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            n_agents = len(config.habitat.simulator.agents)
            for agent_i in range(n_agents):
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agent_config = get_agent_config(
                    config.habitat.simulator, agent_i
                )

                # 添加额外的传感器用于视频录制
                agent_sensors = agent_config.sim_sensors
                extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
                with read_write(agent_sensors):
                    agent_sensors.update(extra_sensors)
                with read_write(config):
                    if config.habitat.gym.obs_keys is not None:
                        for render_view in extra_sensors.values():
                            if (
                                render_view.uuid
                                not in config.habitat.gym.obs_keys
                            ):
                                if n_agents > 1:
                                    config.habitat.gym.obs_keys.append(
                                        f"{agent_name}_{render_view.uuid}"
                                    )
                                else:
                                    config.habitat.gym.obs_keys.append(
                                        render_view.uuid
                                    )

        # 5. 记录配置信息
        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        # 6. 初始化评估环境
        self._init_envs(config, is_eval=True)

        # 7. 创建智能体并加载状态
        self._agent = self._create_agent(None)
        if (
            self._agent.actor_critic.should_load_agent_state
            and self.config.habitat_baselines.eval.should_load_ckpt
        ):
            self._agent.load_state_dict(ckpt_dict)

        # 8. 确定步骤ID
        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        # 9. 创建评估器并执行评估
        evaluator = hydra.utils.instantiate(config.habitat_baselines.evaluator)
        assert isinstance(evaluator, Evaluator)
        evaluator.evaluate_agent(
            self._agent,
            self.envs,
            self.config,
            checkpoint_index,
            step_id,
            writer,
            self.device,
            self.obs_transforms,
            self._env_spec,
            self._rank0_keys,
        )

        # 10. 清理资源
        self.envs.close()


def get_device(config: "DictConfig") -> torch.device:
    """
    获取计算设备
    
    根据配置和CUDA可用性返回适当的计算设备。
    如果CUDA可用，使用指定的GPU；否则使用CPU。
    
    Args:
        config: 配置对象，包含GPU ID设置
        
    Returns:
        torch.device: 计算设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")
