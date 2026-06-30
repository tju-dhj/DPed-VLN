# -*- coding: utf-8 -*-
"""
InstructionBrainPPOTrainer - 指令优化Brain的VLN训练器
====================================================

本训练器继承自BaseRLTrainer，集成了指令优化Brain模块：

核心逻辑：
1. 帧级记录：每帧记录图像、指令、动作、行人信息
2. 条件触发：有行人时才调用brain，无行人跳过
3. 指令优化：brain生成优化后的指令（不是动作）
4. 指令注入：优化后的指令作为神经网络的输入
5. 打印变更：当指令与原始指令不同时打印

训练策略：
- 冻结：行人检测器、CLIP视觉编码器、InstructionBrain模型
- 训练：主VLN策略网络

使用方式：
```python
trainer = InstructionBrainPPOTrainer(config)
trainer.train()
```
"""

import math
import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

import habitat_baselines.rl.multi_agent
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.utils import profiling_wrapper

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

from habitat_baselines.rl.ddppo.algo import DDPPO
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

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.evaluator import Evaluator
from habitat_baselines.rl.ppo.single_agent_access_mgr import SingleAgentAccessMgr

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

# 尝试导入Brain模块
try:
    from .brain import (
        PedestrianDetectionManager,
        InstructionBrain,
        InstructionOptimizationResult,
        FrameRecord,
    )
    from .brain.utils import BrainStats
    BRAIN_MODULE_AVAILABLE = True
except ImportError as e:
    BRAIN_MODULE_AVAILABLE = False
    logger.warn(f"[InstructionBrainPPOTrainer] Brain模块导入失败: {e}")

# 导入分布式工具
try:
    from habitat_baselines.rl.ddppo.ddp_utils import rank0_only
except ImportError:
    # 如果导入失败，定义一个本地版本
    def rank0_only():
        import torch.distributed
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0


def contains_inf_or_nan(observations):
    """
    检查观察数据中是否包含无穷大或NaN值
    """
    for key, value in observations.items():
        if isinstance(value, (float, int)):
            if math.isinf(value) or math.isnan(value):
                print(f"Key {key} contains inf or nan: {value}")
                return True
        elif isinstance(value, (list, tuple, np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                if torch.isinf(value).any() or torch.isnan(value).any():
                    print(f"Key {key} contains inf or nan in tensor")
                    return True
            elif isinstance(value, np.ndarray):
                if np.isinf(value).any() or np.isnan(value).any():
                    print(f"Key {key} contains inf or nan in numpy array")
                    return True
    return False


@baseline_registry.register_trainer(name="instruction_brain_ppo_trainer")
class InstructionBrainPPOTrainer(BaseRLTrainer):
    """
    指令优化Brain的VLN PPO训练器
    =================================

    该训练器集成了指令优化Brain模块，用于在训练过程中整合行人检测和指令优化。

    主要扩展：
    1. 帧级记录：记录每帧的完整信息
    2. 条件触发：只在有行人时调用brain
    3. 指令优化：生成优化后的导航指令
    4. 指令注入：优化后的指令作为神经网络的输入
    5. 变更打印：当指令改变时打印

    Attributes:
        instruction_brain: InstructionBrain实例
        pedestrian_manager: 行人检测管理器
        brain_stats: Brain统计收集器
        current_instruction: 当前使用的指令
        original_instruction: 原始指令
    """

    supported_tasks = ["Nav-v0"]
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    _is_distributed: bool
    envs: VectorEnv
    _env_spec: Optional[EnvironmentSpec]

    def __init__(self, config=None):
        """初始化训练器"""
        super().__init__(config)

        # Brain模块配置
        self.brain_config = getattr(config.habitat_baselines, "brain", None)

        # 模块实例
        self._agent = None
        self.envs = None
        self.obs_transforms = []
        self._is_static_encoder = False
        self._encoder = None
        self._env_spec = None

        # Brain模块
        self.instruction_brain: Optional["InstructionBrain"] = None
        self.pedestrian_manager: Optional["PedestrianDetectionManager"] = None
        self.brain_stats = BrainStats() if BRAIN_MODULE_AVAILABLE else None
        self._brain_initialized = False

        # 指令状态
        self.current_instruction: str = ""
        self.original_instruction: str = ""
        self._frame_counter: int = 0
        self._episode_id: str = ""
        
        # 当前批次的action_data（用于在_compute_actions_and_step_envs和_collect_environment_result之间共享）
        self._current_action_data: Any = None

        # ================================================================
        # 环境级别的状态存储（解决多环境覆盖问题）
        # ================================================================
        self._env_instructions: Dict[int, str] = {}  # 每个环境的当前指令
        self._env_episode_states: Dict[int, Dict[str, Any]] = {}  # 每个环境的episode状态
        self._env_last_instr: Dict[int, str] = {}  # 每个环境的上一个指令（用于检测指令变化）

        # 配置解析
        self.brain_enabled = self._get_brain_config_value("enabled", False)
        self.pedestrian_enabled = self._get_brain_config_value("pedestrian_enabled", True)
        self.freeze_brain = self._get_brain_config_value("freeze_brain", True)
        self.brain_device = self._get_brain_config_value("device", "cuda")
        self.instruction_mode = self._get_brain_config_value("instruction_mode", True)
        self.max_history_frames = self._get_brain_config_value("max_history_frames", 5)
        self.save_frame_records = self._get_brain_config_value("save_frame_records", True)
        self.output_dir = self._get_brain_config_value("output_dir", "./brain_records")
        self.log_prompt = self._get_brain_config_value("log_prompt", True)
        self.save_prompt_to_file = self._get_brain_config_value("save_prompt_to_file", True)
        self.save_frame_images = self._get_brain_config_value("save_frame_images", True)
        self.frame_images_root = self._get_brain_config_value("frame_images_root", "./brain_records/frame_images")

        # ================================================================
        # 异步检测配置（加速多卡训练）
        # ================================================================
        self._async_detection_enabled = self._get_brain_config_value("async_detection_enabled", False)
        self._detection_interval = self._get_brain_config_value("detection_interval", 1)
        self._cached_pedestrian_info: Dict[int, Dict[str, Any]] = {}  # env_idx -> 缓存的检测结果

        # 分布式训练
        self._is_distributed = get_distrib_size()[2] > 1

    def _get_brain_config_value(self, key: str, default: Any) -> Any:
        """从配置中获取Brain相关参数"""
        if self.brain_config is None:
            return default
        if isinstance(self.brain_config, dict):
            return self.brain_config.get(key, default)
        return getattr(self.brain_config, key, default)

    def _init_brain_modules(self) -> None:
        """初始化Brain模块（分布式训练适配：每个rank独立初始化自己的Brain）"""
        if not BRAIN_MODULE_AVAILABLE:
            logger.warn("[InstructionBrainPPOTrainer] Brain模块不可用")
            return

        if self._brain_initialized:
            return

        current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        
        # 每个rank都初始化自己的Brain模块（用于独立记录和保存）
        # 输出目录使用rank-specific子目录
        if self._is_distributed:
            rank_suffix = f"_rank{current_rank}"
        else:
            rank_suffix = ""
        
        # 创建rank-specific的输出目录
        if self.output_dir:
            import os
            self._rank_output_dir = os.path.join(self.output_dir, f"rank{current_rank}")
            os.makedirs(self._rank_output_dir, exist_ok=True)
        else:
            self._rank_output_dir = self.output_dir

        logger.info("=" * 60)
        logger.info(f"[InstructionBrainPPOTrainer] Rank {current_rank} 初始化指令优化Brain模块...")
        logger.info("=" * 60)

        # 初始化行人检测器（所有rank都需要）
        if self.pedestrian_enabled and self.pedestrian_manager is None:
            self.pedestrian_manager = PedestrianDetectionManager(
                enabled=True,
                detector_type=self._get_brain_config_value("pedestrian_detector", "yolov8n"),
                device=self.brain_device,
                confidence=self._get_brain_config_value("pedestrian_confidence", 0.25),
                async_enabled=self._async_detection_enabled,
            )
            logger.info(f"[Rank {current_rank}] 行人检测器已初始化")
            if self._async_detection_enabled:
                logger.info(f"[Rank {current_rank}] 异步检测模式已启用")

        # 初始化InstructionBrain（所有rank都需要，用于独立记录episode）
        if self.brain_enabled and self.instruction_brain is None:
            model_type = self._get_brain_config_value("model_type", "qwen3_vl")
            
            # Rank-specific的InstructionBrain实例
            self.instruction_brain = InstructionBrain(
                model_type=model_type,
                device=self.brain_device,
                model_id=self._get_brain_config_value("model_id", None),
                model_path=self._get_brain_config_value("model_path", None),
                max_history_frames=self.max_history_frames,
                save_frames=self.save_frame_records,
                output_dir=self._rank_output_dir,  # 使用rank-specific目录
                log_prompt=self.log_prompt,
                save_prompt_to_file=self.save_prompt_to_file,
                save_frame_images=self.save_frame_images,
                frame_images_root=self.frame_images_root,
                # API调用相关参数
                call_mode=self._get_brain_config_value("call_mode", "local_hf"),
                api_base_url=self._get_brain_config_value("api_base_url", None),
                api_model_name=self._get_brain_config_value("api_model_name", None),
                api_api_key=self._get_brain_config_value("api_api_key", None),
                # Brain调用置信度阈值
                brain_call_confidence=self._get_brain_config_value("brain_call_confidence", 0.7),
                # Brain调用节流：连续帧最小调用间隔
                min_brain_call_interval=self._get_brain_config_value("min_brain_call_interval", 3),
                # 动作空间大小
                num_actions=self._get_brain_config_value("num_actions", 6),
                # 定期调用Brain（即使无行人检测），适用于无行人训练环境
                call_brain_periodically=self._get_brain_config_value("call_brain_periodically", False),
                call_brain_interval=self._get_brain_config_value("call_brain_interval", 10),
            )
            logger.info(f"[Rank {current_rank}] InstructionBrain已初始化，输出目录: {self._rank_output_dir}")

        logger.info(f"  - Brain启用: {self.brain_enabled}")
        logger.info(f"  - 行人检测启用: {self.pedestrian_enabled}")
        logger.info(f"  - 模型类型: {self._get_brain_config_value('model_type', 'qwen3_vl')}")
        logger.info(f"  - 指令优化模式: {self.instruction_mode}")
        logger.info(f"  - 历史帧数: {self.max_history_frames}")
        logger.info(f"  - 冻结Brain: {self.freeze_brain}")
        logger.info(f"  - 打印Prompt: {self.log_prompt}")
        logger.info(f"  - 保存Prompt: {self.save_prompt_to_file}")
        logger.info(f"  - 保存图像: {self.save_frame_images}")
        logger.info(f"  - 图像目录: {self.frame_images_root}")
        logger.info("=" * 60)

        self._brain_initialized = True

    def _cleanup_brain_modules(self) -> None:
        """清理Brain模块资源"""
        if self.instruction_brain is not None:
            self.instruction_brain.cleanup()
            self.instruction_brain = None

        if self.pedestrian_manager is not None:
            self.pedestrian_manager.shutdown()
            self.pedestrian_manager = None

        self._brain_initialized = False

    def _extract_rgb_observation(self, obs: Dict[str, Any]) -> Optional[np.ndarray]:
        """从observation中提取RGB图像"""
        # 扩展RGB键列表以支持更多传感器配置
        rgb_keys = [
            # 优先使用高分辨率俯视图像（与行人检测配置匹配）
            "agent_0_overhead_front_rgb", "overhead_front_rgb",
            # 标准RGB键
            "rgb", "RGB", "color", "image",
            # 第三人称视角
            "third_rgb", "agent_0_third_rgb",
            # 关节视角
            "articulated_agent_jaw_rgb", "agent_0_articulated_agent_jaw_rgb",
            # 其他可能的变体
            "front_rgb", "first_rgb", "first_person_rgb",
        ]

        for key in rgb_keys:
            if key in obs and obs[key] is not None:
                rgb = obs[key]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                    rgb = np.transpose(rgb, (1, 2, 0))
                elif len(rgb.shape) == 3 and rgb.shape[2] != 3:
                    rgb = np.transpose(rgb, (1, 2, 0))
                return rgb

        return None

    def _get_instruction_from_observation(self, obs: Dict[str, Any]) -> str:
        """从observation中获取指令（优先使用原始指令）
        
        注意：agent_0_falcon_instruction 存储的是字节数组(uint8)，需要解码为字符串。
        """
        import numpy as np
        import torch
        
        # 优先从 agent_0_falcon_instruction 获取原始指令
        instruction_keys = ["agent_0_falcon_instruction", "instruction", "Instruction", "text", "goal"]
        
        for key in instruction_keys:
            if key in obs and obs[key] is not None:
                instr = obs[key]
                
                # 如果是numpy数组（uint8字节数组或int64），需要解码
                if isinstance(instr, np.ndarray):
                    if instr.dtype == np.uint8:
                        non_zero_mask = instr != 0
                        if non_zero_mask.sum() > 0:
                            decoded = bytes(instr[non_zero_mask]).decode('utf-8', errors='ignore')
                            return decoded.strip()
                        return ""
                    elif instr.dtype == np.int64:
                        # FalconInstructionSensor 返回 int64，需要限制在 0-255 范围内解码
                        valid_mask = (instr != 0) & (instr >= 0) & (instr <= 255)
                        if valid_mask.sum() > 0:
                            chars = instr[valid_mask].astype(np.uint8)
                            decoded = bytes(chars).decode('utf-8', errors='ignore')
                            return decoded.strip()
                        return ""
                    return str(instr) if len(str(instr)) < 500 else ""
                
                # 如果是tensor
                elif isinstance(instr, torch.Tensor):
                    instr_np = instr.cpu().numpy()
                    if instr_np.dtype == np.uint8:
                        non_zero_mask = instr_np != 0
                        if non_zero_mask.sum() > 0:
                            decoded = bytes(instr_np[non_zero_mask]).decode('utf-8', errors='ignore')
                            return decoded.strip()
                        return ""
                    elif instr_np.dtype == np.int64:
                        # FalconInstructionSensor 返回 int64，需要限制在 0-255 范围内解码
                        valid_mask = (instr_np != 0) & (instr_np >= 0) & (instr_np <= 255)
                        if valid_mask.sum() > 0:
                            chars = instr_np[valid_mask].astype(np.uint8)
                            decoded = bytes(chars).decode('utf-8', errors='ignore')
                            return decoded.strip()
                        return ""
                    return str(instr) if len(str(instr)) < 500 else ""
                
                # 如果是字符串，直接返回
                return str(instr)
        
        return ""

    def _detect_pedestrian(self, rgb_image: np.ndarray) -> Dict[str, Any]:
        """检测行人"""
        if self.pedestrian_manager is None:
            return {"pedestrian_detected": False, "pedestrian_count": 0}

        return self.pedestrian_manager.detect_frame(rgb_image, self._frame_counter)

    def _optimize_instruction(
        self,
        original_instruction: str,
        current_frame: np.ndarray,
        pedestrian_info: Dict[str, Any],
    ) -> Tuple[str, Optional["InstructionOptimizationResult"]]:
        """优化指令"""
        if not self.brain_enabled or self.instruction_brain is None:
            return original_instruction, None

        result = self.instruction_brain.optimize_instruction(
            original_instruction=original_instruction,
            current_frame=current_frame,
            history_frames=(self.instruction_brain.frame_history[-self.max_history_frames:]
                          if self.instruction_brain.frame_history else None),
            pedestrian_info=pedestrian_info,
            frame_id=self._frame_counter,  # 传递全局帧计数器用于节流
        )

        if (result.should_modify and
                self.instruction_brain.should_update_instruction(original_instruction, result.optimized_instruction)):
            return result.optimized_instruction, result
        else:
            return original_instruction, result

    def _process_step(
        self,
        obs: Dict[str, Any],
        action: int,
        step_number: int,
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        处理单步决策

        Returns:
            (增强后的observation, 当前指令, 行人检测信息)
        """
        self._frame_counter = step_number

        # 提取RGB图像
        rgb_image = self._extract_rgb_observation(obs)

        # 获取原始指令
        original_instruction = self._get_instruction_from_observation(obs)
        if not self.original_instruction:
            self.original_instruction = original_instruction
            self.current_instruction = original_instruction

            # 开始episode记录
            if self.brain_enabled and self.instruction_brain is not None:
                self.instruction_brain.start_episode(
                    episode_id=self._episode_id,
                    original_instruction=original_instruction,
                )

        # 行人检测
        pedestrian_info = {"pedestrian_detected": False, "pedestrian_count": 0}
        if self.pedestrian_enabled and rgb_image is not None:
            pedestrian_info = self._detect_pedestrian(rgb_image)

        # 指令优化（只在有行人时调用brain）
        optimized_instruction = self.current_instruction
        if self.brain_enabled and pedestrian_info.get("pedestrian_detected", False):
            optimized_instruction, optimization_result = self._optimize_instruction(
                original_instruction=self.original_instruction,
                current_frame=rgb_image,
                pedestrian_info=pedestrian_info,
            )

            # 如果指令改变，打印变更（已注释，减少日志输出）
            # if optimized_instruction != self.current_instruction:
            #     logger.info(f"\n[指令变更] Step {step_number}")
            #     logger.info(f"  原指令: {self.current_instruction[:80]}...")
            #     logger.info(f"  新指令: {optimized_instruction[:80]}...")
            #     if optimization_result:
            #         logger.info(f"  原因: {optimization_result.reasoning}")
            #         logger.info(f"  安全等级: {optimization_result.safety_level}")
            #     logger.info("")

            self.current_instruction = optimized_instruction

        # 记录帧数据
        if self.brain_enabled and self.instruction_brain is not None:
            self.instruction_brain.record_frame(
                frame_id=step_number,
                image=rgb_image,
                action=self._action_to_string(action),
                action_id=action,
                instruction=self.current_instruction,
                pedestrian_info=pedestrian_info,
            )

        # 构建增强的observation
        enhanced_obs = obs.copy() if isinstance(obs, dict) else {}
        enhanced_obs["instruction"] = self.current_instruction
        enhanced_obs["original_instruction"] = self.original_instruction
        enhanced_obs["pedestrian_info"] = pedestrian_info
        enhanced_obs["pedestrian_detected"] = pedestrian_info.get("pedestrian_detected", False)
        enhanced_obs["pedestrian_count"] = pedestrian_info.get("pedestrian_count", 0)
        enhanced_obs["instruction_modified"] = self.current_instruction != self.original_instruction

        return enhanced_obs, self.current_instruction, pedestrian_info

    def _action_to_string(self, action: int) -> str:
        """动作ID转字符串"""
        action_names = {
            0: "STOP",
            1: "FORWARD",
            2: "TURN_LEFT",
            3: "TURN_RIGHT",
            4: "WAIT",
            5: "BACKWARD"
        }
        return action_names.get(action, f"UNKNOWN_{action}")

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """分布式训练中的All-Reduce操作"""
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)
        return t.to(device=orig_device)

    def _create_obs_transforms(self):
        """创建观察变换器"""
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
        """创建智能体访问管理器"""
        self._create_obs_transforms()

        return baseline_registry.get_agent_access_mgr(
            self.config.habitat_baselines.rl.agent.type
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
        """初始化向量化环境"""
        if config is None:
            config = self.config

        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )

        self.envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )

        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )

        self._rank0_keys: Set[str] = set(
            list(self.config.habitat.task.rank0_env0_measure_names)
            + list(self.config.habitat.task.rank0_measure_names)
        )

        self._single_proc_infos: Dict[str, List[float]] = {}

    def _init_train(self, resume_state=None):
        """初始化训练过程"""
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

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        self._add_preemption_signal_handlers()

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

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)

            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

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

        self._init_envs()

        self.device = get_device(self.config)

        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        logger.add_filehandler(self.config.habitat_baselines.log_file)

        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.init_distributed(find_unused_params=False)
        self._agent.post_init()

        # ========== 从预训练checkpoint加载权重 ==========
        # 只有在没有resume_state时才从指定checkpoint加载
        if (
            resume_state is None
            and getattr(self.config.habitat_baselines, "load_from_checkpoint", False)
        ):
            checkpoint_path = getattr(
                self.config.habitat_baselines, "checkpoint_path", None
            )
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"从预训练checkpoint加载权重: {checkpoint_path}")
                try:
                    ckpt = self.load_checkpoint(
                        checkpoint_path, map_location=self.device, weights_only=False
                    )
                    # 打印checkpoint结构
                    logger.info(f"Checkpoint keys: {list(ckpt.keys())[:10]}...")
                    
                    # 直接将checkpoint传给agent加载
                    # MultiAgentAccessMgr.load_state_dict() 会自动处理格式
                    self._agent.load_state_dict(ckpt)
                    logger.info(f"Checkpoint加载完成")
                except Exception as e:
                    logger.error(f"加载checkpoint失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode():
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp).detach().clone()
            else:
                with inference_mode():
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch).detach().clone()

        self._agent.rollouts.insert_first_observations(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )

        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        """保存检查点"""
        checkpoint = {
            **self._agent.get_save_state(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        save_file_path = os.path.join(
            self.config.habitat_baselines.checkpoint_folder, file_name
        )
        torch.save(checkpoint, save_file_path)

        torch.save(
            checkpoint,
            os.path.join(
                self.config.habitat_baselines.checkpoint_folder, "latest.pth"
            ),
        )

        if self.config.habitat_baselines.on_save_ckpt_callback is not None:
            hydra.utils.call(
                self.config.habitat_baselines.on_save_ckpt_callback,
                save_file_path=save_file_path,
            )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """加载检查点"""
        # 添加 weights_only=False 以兼容 PyTorch 2.6+ 的安全特性
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return torch.load(checkpoint_path, *args, **kwargs)

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        """计算动作并执行环境步进"""
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.sample_action"), inference_mode():
            step_batch = self._agent.rollouts.get_current_step(
                env_slice, buffer_index
            )

            profiling_wrapper.range_push("compute actions")

            step_batch_lens = {
                k: v
                for k, v in step_batch.items()
                if k.startswith("index_len")
            }

            action_data = self._agent.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        # 保存action_data供_collect_environment_result使用（包含真实动作值）
        self._current_action_data = action_data
        
        profiling_wrapper.range_pop()

        with g_timer.avg_time("trainer.obs_insert"):
            for index_env, act in zip(
                range(env_slice.start, env_slice.stop),
                action_data.env_actions.cpu().unbind(0),
            ):
                if hasattr(self._agent, '_agents') and self._agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                    act = act.numpy()
                elif is_continuous_action_space(self._env_spec.action_space):
                    act = np.clip(
                        act.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                else:
                    act = act.item()
                self.envs.async_step_at(index_env, act)

        with g_timer.avg_time("trainer.obs_insert"):
            self._agent.rollouts.insert(
                next_recurrent_hidden_states=action_data.rnn_hidden_states,
                actions=action_data.actions,
                action_log_probs=action_data.action_log_probs,
                value_preds=action_data.values,
                buffer_index=buffer_index,
                should_inserts=action_data.should_inserts,
                action_data=action_data,
            )

    def _collect_environment_result(self, buffer_index: int = 0):
        """收集环境执行结果"""
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))
            self.running_episode_stats["count"][env_slice] += done_masks.float()

            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )

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
                self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))

            self.current_episode_reward[env_slice].masked_fill_(
                done_masks, 0.0
            )

            # 处理已完成episode的环境：保存记录并清理状态
            for i, done in enumerate(dones):
                env_idx = env_slice.start + i
                if done:
                    # 关键修复：使用trainer的episode_state来保存记录，而不是brain的current_episode_id
                    # 因为brain的current_episode_id可能在多环境时被其他环境覆盖
                    episode_state = self._env_episode_states.get(env_idx, {})
                    trainer_episode_id = episode_state.get('episode_id')
                    
                    if self.brain_enabled and self.instruction_brain is not None and trainer_episode_id is not None:
                        # 检查brain中是否有这个episode的记录
                        brain_ep = self.instruction_brain.episode_records.get(trainer_episode_id)
                        
                        if brain_ep is not None:
                            # Episode记录存在，手动保存
                            # logger.info(f"[Train-EndEpisode] Env {env_idx}, saving episode {trainer_episode_id} "
                            #           f"with {brain_ep.total_frames} frames")
                            filepath = self.instruction_brain.save_episode_record(trainer_episode_id)
                            # logger.info(f"[Train-EndEpisode] Saved to: {filepath}")
                            # 重要：保存后从brain记录中删除，避免重复保存
                            if trainer_episode_id in self.instruction_brain.episode_records:
                                del self.instruction_brain.episode_records[trainer_episode_id]
                            if trainer_episode_id in self.instruction_brain._frame_histories:
                                del self.instruction_brain._frame_histories[trainer_episode_id]
                            if trainer_episode_id in self.instruction_brain._pedestrian_trajectory_histories:
                                del self.instruction_brain._pedestrian_trajectory_histories[trainer_episode_id]
                        else:
                            # Episode记录不存在（可能brain尚未启动），调用end_episode但不保存（外部未保存）
                            logger.warning(f"[Train-EndEpisode] Env {env_idx}: episode {trainer_episode_id} "
                                         f"not found in brain.records, calling end_episode(save=False)")
                            self.instruction_brain.end_episode(trainer_episode_id, save=False, env_idx=env_idx)

                    # 清理brain的_env_current_episode_ids映射
                    if self.brain_enabled and self.instruction_brain is not None:
                        env_ep_id = self.instruction_brain.get_env_episode_id(env_idx)
                        if env_ep_id == trainer_episode_id:
                            # 清理该env的episode_id映射
                            if env_idx in self.instruction_brain._env_current_episode_ids:
                                del self.instruction_brain._env_current_episode_ids[env_idx]
                                logger.info(f"[Train-EndEpisode] Env {env_idx}: cleaned up _env_current_episode_ids")

                    # 重置该环境的指令状态
                    if env_idx in self._env_instructions:
                        del self._env_instructions[env_idx]
                    if env_idx in self._env_last_instr:
                        del self._env_last_instr[env_idx]
                    if env_idx in self._env_episode_states:
                        del self._env_episode_states[env_idx]
                    
                    # 修复：清理行人检测缓存，确保新episode使用正确的检测结果
                    if env_idx in self._cached_pedestrian_info:
                        del self._cached_pedestrian_info[env_idx]

        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            if self._encoder is None:
                self._encoder = self._agent._agents[0].actor_critic.visual_encoder
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch_temp = {key.replace('agent_0_', ''): value for key, value in batch.items()}
                    batch[
                        'agent_0_' + PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch_temp).detach().clone()
            else:
                with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                    batch[
                        PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                    ] = self._encoder(batch).detach().clone()

        # ================================================================
        # Brain处理：行人检测和指令优化（只在有行人时调用大模型）
        # 分布式训练适配：只在rank 0处理Brain，减少通信开销
        # 修复：每个环境维护独立的指令状态，避免跨环境覆盖问题
        # 修复：只在episode开始和结束时调用start_episode/end_episode
        # ================================================================
        if self.brain_enabled and self.instruction_brain is not None:
            # 分布式训练时，只有rank 0处理Brain
            is_rank0 = not self._is_distributed or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)
            current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

            for i, obs in enumerate(observations):
                # 计算环境的全局索引
                env_idx = env_slice.start + i
                is_done = dones[i] if i < len(dones) else False

                # 获取原始指令
                original_instr = self._get_instruction_from_observation(obs)

                # 初始化环境的指令状态（如果尚未初始化）
                if env_idx not in self._env_instructions:
                    self._env_instructions[env_idx] = original_instr
                    self._env_last_instr[env_idx] = original_instr
                    self._env_episode_states[env_idx] = {
                        'episode_id': None,  # 初始为None，等待开始
                        'brain_call_count': 0,
                        'step_count': 0,
                    }

                # 获取环境的episode状态
                episode_state = self._env_episode_states.get(env_idx, {})
                episode_state['step_count'] = episode_state.get('step_count', 0) + 1

                # 全局帧计数器（用于异步检测追踪）
                self._frame_counter += 1

                # 提取RGB图像
                rgb_image = self._extract_rgb_observation(obs)

                # 获取done状态
                is_done = dones[i] if i < len(dones) else False

                # ================================================================
                # 行人检测（优化版本：帧间隔检测 + 异步CUDA加速）
                # ================================================================
                pedestrian_info = {"pedestrian_detected": False, "pedestrian_count": 0}
                if self.pedestrian_enabled and rgb_image is not None:
                    # 获取该环境的step计数
                    env_step_count = episode_state.get('step_count', 0)
                    
                    # 修复：episode刚开始（step_count <= detection_interval）时强制检测，避免空缓存问题
                    # 或者当前缓存为空时也强制检测
                    cached_info = self._cached_pedestrian_info.get(env_idx)
                    should_use_cache = (
                        env_step_count > self._detection_interval and
                        cached_info is not None and
                        env_step_count % self._detection_interval != 0
                    )
                    
                    if should_use_cache:
                        # 使用缓存的检测结果
                        pedestrian_info = cached_info
                    else:
                        # 执行实际检测（第一帧或缓存为空时强制检测）
                        if self.pedestrian_manager is not None:
                            # 获取全局帧ID用于异步追踪
                            global_frame_id = self._frame_counter

                            # 根据配置选择同步或异步检测
                            if self._async_detection_enabled and hasattr(self.pedestrian_manager, 'detect_async_submit'):
                                # 异步模式：提交检测任务到CUDA流
                                self.pedestrian_manager.detect_async_submit(rgb_image, global_frame_id)
                                # 等待并获取结果（异步流同步）
                                pedestrian_info = self.pedestrian_manager.detect_async_wait(global_frame_id)
                            else:
                                # 同步模式：直接执行检测
                                pedestrian_info = self.pedestrian_manager.detect_frame(rgb_image, global_frame_id)

                            # 缓存结果
                            self._cached_pedestrian_info[env_idx] = pedestrian_info.copy()

                            # 记录行人检测统计（已注释，减少日志输出）
                            # if is_rank0 and pedestrian_info.get("pedestrian_detected", False):
                            #     if not hasattr(self, '_train_pedestrian_count'):
                            #         self._train_pedestrian_count = 0
                            #     self._train_pedestrian_count += 1
                            #     if self._train_pedestrian_count <= 5:
                            #         logger.info(f"[Train-Pedestrian] Rank{current_rank} Env {env_idx}: Detected {pedestrian_info.get('pedestrian_count', 0)} pedestrian(s)")

                # Brain调用和指令优化（所有rank都执行，每个rank有独立的Brain实例）
                if True:  # 移除is_rank0限制，让所有rank都能执行Brain
                    # 检查是否需要开始新的episode（只在首次或episode完成重新开始时）
                    if episode_state['episode_id'] is None:
                        # 开始新的episode，使用全局计数器确保唯一性
                        if not hasattr(self, '_rank_env_episode_counters'):
                            self._rank_env_episode_counters = {}
                        counter_key = f"{current_rank}_{env_idx}"
                        if counter_key not in self._rank_env_episode_counters:
                            self._rank_env_episode_counters[counter_key] = 0
                        episode_num = self._rank_env_episode_counters[counter_key]
                        new_episode_id = f"rank{current_rank}_env{env_idx}_ep{episode_num}"
                        episode_state['episode_id'] = new_episode_id
                        episode_state['brain_call_count'] = 0
                        episode_state['episode_number'] = episode_num
                        
                        # 更新该环境的全局episode计数器（持久化）
                        self._rank_env_episode_counters[counter_key] += 1

                        self._env_episode_states[env_idx] = episode_state

                        # 调试日志：打印即将传递给start_episode的参数（已注释）
                        # instr_preview = original_instr[:100] if original_instr else "EMPTY"
                        # logger.info(f"[Train-StartEpisode] Env {env_idx}: calling start_episode("
                        #           f"episode_id='{new_episode_id}', "
                        #           f"original_instruction='{instr_preview}...')")

                        # 开始新的episode记录
                        self.instruction_brain.start_episode(
                            episode_id=new_episode_id,
                            original_instruction=original_instr if original_instr else "No instruction",
                            env_idx=env_idx,
                        )
                        
                        # 验证start_episode是否成功
                        # 使用get_env_episode_id验证，而不是current_episode_id
                        if self.instruction_brain.get_env_episode_id(env_idx) != new_episode_id:
                            logger.error(f"[Train-StartEpisode] FAILED! brain.get_env_episode_id({env_idx})={self.instruction_brain.get_env_episode_id(env_idx)}, expected={new_episode_id}")

                        if not hasattr(self, '_train_last_instruction') or self._train_last_instruction != original_instr:
                            self._train_last_instruction = original_instr

                    # 指令优化（只在有行人时调用Brain，支持节流机制）
                    target_episode_id = episode_state.get('episode_id')
                    current_step = episode_state.get('step_count', 0)
                    should_call_brain = self.instruction_brain.should_call_brain(
                        pedestrian_info, episode_id=target_episode_id, frame_id=current_step
                    )

                    # 初始化为原始指令（每帧独立决策）
                    env_current_instruction = original_instr if original_instr else ""

                    if should_call_brain:
                        current_call_count = episode_state.get('brain_call_count', 0)

                        # 使用正确的episode_id获取历史帧
                        history_frames = self.instruction_brain.get_frame_history(target_episode_id)[-self.max_history_frames:] if target_episode_id else None

                        # 调用Brain进行指令优化（传入frame_id以启用节流机制）
                        # 包含异常保护：单次VLM失败不会导致整个训练崩溃
                        try:
                            result = self.instruction_brain.optimize_instruction(
                                original_instruction=original_instr if original_instr else "",
                                current_frame=rgb_image,
                                history_frames=history_frames,
                                pedestrian_info=pedestrian_info,
                                episode_id=target_episode_id,
                                env_idx=env_idx,
                                frame_id=current_step,
                            )
                        except Exception as e:
                            logger.error(
                                f"[Train-VLM] Rank{current_rank} Env{env_idx}: "
                                f"optimize_instruction failed: {e}. "
                                f"Using original instruction for this step."
                            )
                            result = InstructionOptimizationResult(
                                original_instruction=original_instr or "",
                                optimized_instruction=original_instr or "",
                                modifier_type=InstructionModifier.ORIGINAL,
                                confidence=0.0,
                                reasoning=f"VLM inference exception: {e}",
                                should_modify=False,
                                safety_level="unknown",
                                pedestrian_warning=False,
                                warning_message="",
                                raw_response="",
                                inference_time_ms=0.0,
                            )

                        episode_state['brain_call_count'] = current_call_count + 1

                        # 更新全局统计
                        if not hasattr(self, '_train_brain_call_count'):
                            self._train_brain_call_count = 0
                        self._train_brain_call_count += 1

                        # 如果指令被修改，使用优化后的指令（仅当前帧生效）
                        if result.should_modify and self.instruction_brain.should_update_instruction(
                            env_current_instruction, result.optimized_instruction
                        ):
                            env_current_instruction = result.optimized_instruction

                            # 打印指令变更日志（已注释）
                            # logger.info(f"\n[Train-指令变更] Rank{current_rank} Env {env_idx}, Call #{self._train_brain_call_count}")
                            # logger.info(f"  原指令: {original_instr[:80] if original_instr else 'N/A'}...")
                            # logger.info(f"  新指令: {result.optimized_instruction[:80] if result.optimized_instruction else 'N/A'}...")
                            # logger.info(f"  原因: {result.reasoning[:100] if result.reasoning else 'N/A'}...")
                            # logger.info(f"  安全等级: {result.safety_level}, 置信度: {result.confidence:.2f}")
                            # logger.info("")

                        # 打印Brain响应统计（已注释）
                        # if self._train_brain_call_count <= 10:
                        #     logger.info(f"[Train-Brain-Response] Call #{self._train_brain_call_count}: "
                        #               f"should_modify={result.should_modify}, "
                        #               f"safety_level={result.safety_level}, "
                        #               f"confidence={result.confidence:.2f}")

                    # 更新环境指令（每帧独立，不跨帧保留优化指令）
                    self._env_instructions[env_idx] = env_current_instruction
                    self._env_last_instr[env_idx] = original_instr

                # 获取该环境的当前指令（来自Brain优化或原始指令）
                env_instruction = self._env_instructions.get(env_idx, original_instr if original_instr else "")

                # 从action_data获取真实动作值并记录帧数据
                action_id = 1
                action_str = "FORWARD"
                if self._current_action_data is not None:
                    env_actions = self._current_action_data.env_actions
                    if env_idx < env_actions.shape[0]:
                        action_tensor = env_actions[env_idx].cpu()
                        if hasattr(self._agent, '_agents') and self._agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                            action_id = int(action_tensor.item())
                        elif is_continuous_action_space(self._env_spec.action_space):
                            action_id = 1
                        else:
                            action_id = int(action_tensor.item()) if action_tensor.numel() == 1 else 1
                        action_str = {
                            0: "STOP",
                            1: "FORWARD",
                            2: "TURN_LEFT",
                            3: "TURN_RIGHT",
                            4: "WAIT",
                            5: "BACKWARD"
                        }.get(action_id, f"UNKNOWN_{action_id}")

                # 记录帧数据（已在上面初始化episode，只需记录帧即可）
                # 调试日志（已注释）
                # _is_rank0 = is_rank0
                # _brain_not_none = self.instruction_brain is not None
                # _brain_enabled = self.brain_enabled
                # print(f"[DEBUG-RecordFrame] is_rank0={_is_rank0}, brain={_brain_not_none}, "
                #       f"brain_enabled={_brain_enabled}, env_idx={env_idx}, episode_id={episode_state.get('episode_id')}, "
                #       f"_is_distributed={self._is_distributed}, "
                #       f"distrib_init={torch.distributed.is_initialized() if hasattr(torch.distributed, 'is_initialized') else 'N/A'}")

                if self.instruction_brain is not None:
                    # 调试日志（已注释）
                    # if not hasattr(self, '_record_frame_debug_counter'):
                    #     self._record_frame_debug_counter = 0
                    # self._record_frame_debug_counter += 1
                    # if self._record_frame_debug_counter <= 20:
                    #     logger.info(f"[Train-RecordFrame-ENTER] Env {env_idx}: ENTERED record_frame branch, "
                    #               f"rgb_image={'None' if rgb_image is None else f'shape={rgb_image.shape}'}, "
                    #               f"episode_id={episode_state.get('episode_id')}")
                    
                    # 确保frame_id在所有情况下都被定义
                    if episode_state.get('episode_id') is None:
                        current_frame_id = episode_state.get('step_count', 0)
                    else:
                        current_frame_id = episode_state.get('step_count', 0)
                    
                    # if rgb_image is None:
                        # 调试：记录rgb_image为None的情况（已注释）
                        # if self._record_frame_debug_counter <= 10:
                        #     logger.warning(f"[Train-RecordFrame-Debug] Env {env_idx} Frame {current_frame_id}: rgb_image is None! "
                        #                  f"Available keys in obs: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
                    
                    # 检查episode是否已启动，如果未启动则先启动
                    if episode_state.get('episode_id') is None:
                        # 开始新的episode，使用全局计数器确保唯一性
                        current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                        if not hasattr(self, '_rank_env_episode_counters'):
                            self._rank_env_episode_counters = {}
                        counter_key = f"{current_rank}_{env_idx}"
                        if counter_key not in self._rank_env_episode_counters:
                            self._rank_env_episode_counters[counter_key] = 0
                        episode_num = self._rank_env_episode_counters[counter_key]
                        new_episode_id = f"rank{current_rank}_env{env_idx}_ep{episode_num}"
                        episode_state['episode_id'] = new_episode_id
                        episode_state['brain_call_count'] = 0
                        episode_state['episode_number'] = episode_num
                        
                        # 更新该环境的全局episode计数器（持久化）
                        self._rank_env_episode_counters[counter_key] += 1
                        self._env_episode_states[env_idx] = episode_state

                        # 调试日志（已注释）
                        # instr_preview = original_instr[:100] if original_instr else "EMPTY"
                        # logger.info(f"[Train-RecordFrame-AutoStart] Env {env_idx}: auto starting episode "
                        #           f"episode_id='{new_episode_id}', instruction='{instr_preview}...'")

                        # 开始新的episode记录
                        self.instruction_brain.start_episode(
                            episode_id=new_episode_id,
                            original_instruction=original_instr if original_instr else "No instruction",
                            env_idx=env_idx,
                        )
                        
                        # 验证start_episode是否成功
                        # 使用get_env_episode_id验证，而不是current_episode_id
                        if self.instruction_brain.get_env_episode_id(env_idx) != new_episode_id:
                            logger.error(f"[Train-RecordFrame-AutoStart] FAILED! brain.get_env_episode_id({env_idx})={self.instruction_brain.get_env_episode_id(env_idx)}, expected={new_episode_id}")
                    
                    frame_id = episode_state['step_count']
                    
                    # 验证brain的该env的episode_id与trainer的一致
                    # 使用env_idx对应的episode_id来验证，而不是依赖全局current_episode_id
                    env_episode_id = self.instruction_brain.get_env_episode_id(env_idx)
                    if env_episode_id != episode_state.get('episode_id'):
                        logger.warning(f"[Train-RecordFrame] Env {env_idx}: brain_env_episode_id mismatch! "
                                     f"brain_env={env_episode_id}, trainer={episode_state.get('episode_id')}")
                        # 重新调用start_episode同步状态
                        self.instruction_brain.start_episode(
                            episode_id=episode_state.get('episode_id'),
                            original_instruction=original_instr if original_instr else "No instruction",
                            env_idx=env_idx,
                        )
                    
                    # 记录帧数据
                    # 关键修复：传入正确的episode_id以支持多环境并行
                    target_episode_id = episode_state.get('episode_id')

                    # 调试日志（已注释）
                    # if not hasattr(self, '_record_frame_call_logged'):
                    #     self._record_frame_call_logged = set()
                    # log_key = f"{env_idx}_{target_episode_id}"
                    # if log_key not in self._record_frame_call_logged or frame_id <= 5:
                    #     logger.info(f"[Train-RecordFrame-CALL] Env {env_idx} Frame {frame_id}: "
                    #               f"calling record_frame with episode_id={target_episode_id}, "
                    #               f"rgb_image={'None' if rgb_image is None else f'shape={rgb_image.shape}'}")
                    #     self._record_frame_call_logged.add(log_key)

                    self.instruction_brain.record_frame(
                        frame_id=frame_id,
                        image=rgb_image,
                        action=action_str,
                        action_id=action_id,
                        instruction=env_instruction,
                        pedestrian_info=pedestrian_info,
                        episode_id=target_episode_id,
                        env_idx=env_idx,
                    )

                    # 调试日志：每10帧打印一次记录状态（已注释）
                    # if frame_id % 10 == 0 or frame_id <= 3:
                    #     brain_ep = self.instruction_brain.episode_records.get(episode_state.get('episode_id'))
                    #     if brain_ep:
                    #         logger.info(f"[Train-RecordFrame] Env {env_idx} Frame {frame_id}: "
                    #                   f"total_frames={brain_ep.total_frames}, "
                    #                   f"pedestrian_frames={brain_ep.frames_with_pedestrian}, "
                    #                   f"brain_calls={brain_ep.brain_calls}")

                # 将该环境的优化后指令注入到observations中
                if "instruction" in observations[i]:
                    observations[i]["instruction"] = env_instruction
                if "agent_0_falcon_instruction" in observations[i]:
                    INSTR_MAX_LEN = 512
                    instr_bytes = env_instruction.encode('utf-8')
                    if len(instr_bytes) > INSTR_MAX_LEN:
                        instr_bytes = instr_bytes[:INSTR_MAX_LEN]
                    instr_array = np.zeros(INSTR_MAX_LEN, dtype=np.uint8)
                    instr_array[:len(instr_bytes)] = np.frombuffer(instr_bytes, dtype=np.uint8)
                    observations[i]["agent_0_falcon_instruction"] = instr_array

        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        """收集一个rollout步骤"""
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        """更新智能体（含CUDA健康检查和rank同步，防止NCCL超时）"""
        # Step 1: 分布式 barrier — 确保所有 rank 在进入 DDP forward 前已同步
        # Habitat 模拟中不同 env 的 episode 长度可能差异很大（1帧 vs 10帧），
        # 导致各 rank 到达 _update_agent 的时间不一致。显式 barrier 确保所有
        # rank 就绪后再进行 NCCL 集体通信。
        if self._is_distributed:
            try:
                torch.distributed.barrier()
            except RuntimeError as e:
                logger.warning(
                    f"[Barrier-Error] Rank {torch.distributed.get_rank()}: "
                    f"barrier failed: {e}. Proceeding anyway..."
                )

        # Step 2: CUDA健康检查 — 在进入NCCL同步前检测残存的GPU错误
        # VLM Brain 的 generate() 可能留下孤儿 CUDA kernel，提前捕获避免 NCCL 级联超时
        if self.device.type == "cuda":
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError as e:
                logger.error(
                    f"[CUDA-Error] Rank {torch.distributed.get_rank()}: "
                    f"CUDA synchronize failed before _update_agent: {e}. "
                    f"Clearing CUDA cache and skipping this update to avoid NCCL cascade."
                )
                torch.cuda.empty_cache()
                return {}

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

        self._agent.rollouts.compute_returns(
            next_value,
            self._ppo_cfg.use_gae,
            self._ppo_cfg.gamma,
            self._ppo_cfg.tau,
        )

        self._agent.train()

        # Step 3: DDP forward with NCCL error handling
        # 如果 NCCL 通信仍然超时，捕获异常并记录诊断信息
        try:
            losses = self._agent.updater.update(self._agent.rollouts)
        except RuntimeError as e:
            if "NCCL" in str(e) or "timeout" in str(e).lower():
                logger.error(
                    f"[NCCL-Timeout] Rank {torch.distributed.get_rank()}: "
                    f"NCCL collective failed during updater.update(): {e}. "
                    f"This may be caused by rank imbalance in rollout collection. "
                    f"Check if some environments produce significantly fewer steps "
                    f"than others (e.g. 1-frame episodes vs 10-frame episodes)."
                )
                torch.cuda.empty_cache()
                return {}
            raise

        self._agent.rollouts.after_update()
        self._agent.after_update()

        return losses

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        """合并后处理步骤"""
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        """记录训练日志"""
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)

        for k, v in losses.items():
            writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

        for k, v in self._single_proc_infos.items():
            writer.add_scalar(k, np.mean(v), self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        writer.add_scalar("perf/fps", fps, self.num_steps_done)

        for timer_name, timer_val in g_timer.items():
            writer.add_scalar(
                f"perf/{timer_name}",
                timer_val.mean,
                self.num_steps_done,
            )

        if (
            self.num_updates_done % self.config.habitat_baselines.log_interval
            == 0
        ):
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                f"Num updates: {self.num_updates_done}\tNum frames {self.num_steps_done}"
            )

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

            perf_stats_str = " ".join(
                [f"{k}: {v.mean:.3f}" for k, v in g_timer.items()]
            )
            logger.info(f"\tPerf Stats: {perf_stats_str}")

            if self.config.habitat_baselines.should_log_single_proc_infos:
                for k, v in self._single_proc_infos.items():
                    logger.info(f" - {k}: {np.mean(v):.3f}")

            # ========== Brain 和行人检测统计 ==========
            if self.pedestrian_enabled and self.pedestrian_manager is not None:
                ped_stats = self.pedestrian_manager.get_stats()
                logger.info(
                    f"[Pedestrian Detection] Total: {ped_stats['detection_count']} detections, "
                    f"Avg: {ped_stats['avg_time_ms']:.2f} ms, "
                    f"Cache hit rate: {ped_stats['cache_hit_rate']:.1f}%"
                )

            if self.brain_enabled and self.instruction_brain is not None:
                brain_stats = self.instruction_brain.get_stats()
                logger.info(
                    f"[Brain Inference] Total: {brain_stats['inference_count']} calls, "
                    f"Avg: {brain_stats['avg_inference_time_ms']:.2f} ms, "
                    f"Total time: {brain_stats['total_inference_time_s']:.2f} s"
                )

    def should_end_early(self, rollout_step) -> bool:
        """判断是否应该提前结束rollout"""
        if not self._is_distributed:
            return False

        return (
            rollout_step
            >= self.config.habitat_baselines.rl.ppo.num_steps
            * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.habitat_baselines.rl.ddppo.sync_frac
            * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        """主训练方法"""
        # 初始化Brain模块
        self._init_brain_modules()

        try:
            self._do_train()
        finally:
            self._cleanup_brain_modules()

    def _do_train(self) -> None:
        """实际训练逻辑"""
        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        count_checkpoints = 0
        prev_time = 0

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        if resume_state is not None:
            self._agent.load_state_dict(resume_state)

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )
            resume_run_id = requeue_stats.get("run_id", None)

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
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                self._agent.pre_rollout()

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

                if EXIT.is_set():
                    profiling_wrapper.range_pop()
                    self.envs.close()
                    requeue_job()
                    return

                self._agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                with g_timer.avg_time("trainer.rollout_collect"):
                    for buffer_index in range(self._agent.nbuffers):
                        self._compute_actions_and_step_envs(buffer_index)

                    for step in range(self._ppo_cfg.num_steps):
                        is_last_step = (
                            self.should_end_early(step + 1)
                            or (step + 1) == self._ppo_cfg.num_steps
                        )

                        for buffer_index in range(self._agent.nbuffers):
                            count_steps_delta += (
                                self._collect_environment_result(buffer_index)
                            )

                            if (buffer_index + 1) == self._agent.nbuffers:
                                profiling_wrapper.range_pop()

                            if not is_last_step:
                                if (buffer_index + 1) == self._agent.nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(
                                    buffer_index
                                )

                        if is_last_step:
                            break

                profiling_wrapper.range_pop()

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                losses = self._update_agent()

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    logger.info(f'PPO save to ckpt.{count_checkpoints}.pth')
                    count_checkpoints += 1

                profiling_wrapper.range_pop()

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """评估单个检查点"""
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        if self.config.habitat_baselines.eval.should_load_ckpt:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            logger.info(f"Loaded checkpoint trained for {step_id} steps")
        else:
            ckpt_dict = {"config": None}

        if "config" not in ckpt_dict:
            ckpt_dict["config"] = None

        config = self._get_resume_state_config_or_new_config(
            ckpt_dict["config"]
        )
        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            n_agents = len(config.habitat.simulator.agents)
            for agent_i in range(n_agents):
                agent_name = config.habitat.simulator.agents_order[agent_i]
                agent_config = get_agent_config(
                    config.habitat.simulator, agent_i
                )

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

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        if (
            self._agent.actor_critic.should_load_agent_state
            and self.config.habitat_baselines.eval.should_load_ckpt
        ):
            self._agent.load_state_dict(ckpt_dict)

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        # 直接创建评估器实例（绕过Hydra的配置验证）
        from habitat_baselines.rl.ppo.instruction_brain_ppo_evaluator import (
            InstructionBrainPPOEvaluator,
        )
        # Brain配置 - 使用OmegaConf转换为字典，确保兼容
        brain_config = None
        try:
            if hasattr(config.habitat_baselines, 'brain'):
                brain_config = OmegaConf.to_container(config.habitat_baselines.brain, resolve=True)
        except Exception:
            pass
        evaluator = InstructionBrainPPOEvaluator(
            config=config,
            brain_config=brain_config
        )
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

        self.envs.close()

    @property
    def is_brain_enabled(self) -> bool:
        """检查Brain是否启用"""
        return self.brain_enabled and self._brain_initialized

    @property
    def is_pedestrian_enabled(self) -> bool:
        """检查行人检测是否启用"""
        return self.pedestrian_enabled

    def get_brain_statistics(self) -> Dict[str, Any]:
        """获取Brain统计"""
        if self.brain_stats is None:
            return {}
        return self.brain_stats.get_summary()

    def print_brain_statistics(self) -> None:
        """打印Brain统计"""
        if self.brain_stats is not None:
            self.brain_stats.print_summary()


def get_device(config: "DictConfig") -> torch.device:
    """获取计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")
