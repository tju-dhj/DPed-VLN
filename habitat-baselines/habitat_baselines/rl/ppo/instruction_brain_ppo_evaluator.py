# -*- coding: utf-8 -*-
"""
InstructionBrainPPOEvaluator - 指令优化Brain的VLN评估器
========================================================

继承自Evaluator基类，用于评估集成了指令优化Brain的VLN策略。
该评估器记录详细的帧级数据和评估进度。

主要功能：
1. 帧级记录：每帧记录完整数据
2. 指令变更追踪（需要时）
3. 行人检测统计（需要时）
4. 评估进度保存与恢复
5. 评估报告生成
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import tqdm
import gc

from habitat import logger
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

import json


def calculate_and_log_average_stats(stats_episodes, completed_episodes, logger):
    """
    计算并输出从开始到当前的平均评估结果

    Args:
        stats_episodes: 存储所有episode统计信息的字典
        completed_episodes: 已完成的episodes数量
        logger: logger对象用于输出日志
    """
    if len(stats_episodes) == 0:
        return

    aggregated_stats = {}
    all_ks = set()
    for ep in stats_episodes.values():
        all_ks.update(ep.keys())
    for stat_key in all_ks:
        aggregated_stats[stat_key] = np.mean(
            [v[stat_key] for v in stats_episodes.values() if stat_key in v]
        )

    logger.info(f"========== 已完成 {completed_episodes} 个episodes的平均评估结果 ==========")
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.4f}")
    logger.info("=" * 60)


class InstructionBrainPPOEvaluator(Evaluator):
    """
    指令优化Brain的VLN评估器
    =================================

    该评估器继承自Evaluator基类，集成了可选的指令优化Brain模块。
    主要功能：
    1. 帧级记录：每帧记录完整数据
    2. 指令变更追踪
    3. 行人检测统计
    4. 评估报告生成
    5. 评估进度保存与恢复

    Attributes:
        config: 配置对象
        brain_config: Brain模块配置
        instruction_brain: InstructionBrain实例
        pedestrian_manager: 行人检测管理器
        brain_enabled: Brain是否启用
        pedestrian_enabled: 行人检测是否启用
    """

    def __init__(self, config=None, brain_config=None):
        """
        初始化评估器

        Args:
            config: 配置对象，可以是包含brain配置的字典或对象
            brain_config: Brain配置对象，如果为None则从config中尝试提取
        """
        self.config = config

        # 从配置中获取brain_config
        # 优先使用传入的brain_config参数
        self.brain_config = brain_config
        if self.brain_config is None and config is not None:
            # 尝试作为有brain属性的对象访问
            if hasattr(config, 'brain'):
                self.brain_config = config.brain
            # 尝试作为字典访问
            elif isinstance(config, dict):
                self.brain_config = config.get('brain')
            # 如果没有brain，可能是直接传递的brain配置
            if self.brain_config is None:
                # 检查是否有其他brain相关属性
                if hasattr(config, 'enabled'):
                    self.brain_config = config

        # Brain模块
        self.instruction_brain: Optional[Any] = None
        self.pedestrian_manager: Optional[Any] = None
        self._brain_initialized = False

        # 指令状态
        self.current_instruction: str = ""
        self.original_instruction: str = ""
        self._frame_counter: int = 0
        self._episode_id: str = ""

        # ================================================================
        # 多环境隔离：每个环境独立的指令状态（修复跨环境覆盖bug）
        # ================================================================
        self._env_episode_ids: Dict[int, str] = {}  # env_idx -> current episode id
        self._env_brain_episode_ids: Dict[int, str] = {}  # env_idx -> unique episode id for InstructionBrain
        self._env_instructions: Dict[int, str] = {}  # env_idx -> current instruction
        self._env_original_instructions: Dict[int, str] = {}  # env_idx -> original instruction
        self._env_frame_counters: Dict[int, int] = {}  # env_idx -> frame counter
        self._env_episode_records: Dict[int, Dict[str, Any]] = {}  # env_idx -> episode record
        self._brain_episode_uid_counter: int = 0

        # ================================================================
        # 场景状态缓存：用于优化同一场景内 episode 之间的切换
        # 避免不必要的场景重新加载和初始化
        # ================================================================
        self._current_scene_id: str = ""  # 当前场景ID
        self._scene_initialized: bool = False  # 场景是否已初始化
        self._scene_init_episode_id: str = ""  # 初始化当前场景时的 episode_id
        self._env_scene_ids: Dict[int, str] = {}  # env_idx -> current scene id
        self._env_scene_initialized: Dict[int, bool] = {}  # env_idx -> scene initialized flag
        self._env_scene_init_episode_ids: Dict[int, str] = {}  # env_idx -> scene init episode id

        # 评估记录
        self.evaluation_records: List[Dict] = []
        self.current_episode_record: Dict = {}

        # 配置解析
        self.brain_enabled = self._get_brain_config_value("enabled", False)
        self.pedestrian_enabled = self._get_brain_config_value("pedestrian_enabled", True)
        self.brain_device = self._get_brain_config_value("device", "cuda")
        self.output_dir = self._get_brain_config_value("output_dir", "./brain_eval_records")

        # 尝试导入Brain模块
        self._init_brain_modules()

    def _get_brain_config_value(self, key: str, default: Any) -> Any:
        """从配置中获取Brain相关参数"""
        if self.brain_config is None:
            return default
        # 如果brain_config本身就是顶级brain配置（包含enabled等键）
        if hasattr(self.brain_config, 'enabled') or (isinstance(self.brain_config, dict) and 'enabled' in self.brain_config):
            # 直接从brain_config获取
            if isinstance(self.brain_config, dict):
                return self.brain_config.get(key, default)
            return getattr(self.brain_config, key, default)
        # 否则尝试嵌套访问
        if isinstance(self.brain_config, dict):
            return self.brain_config.get(key, default)
        return getattr(self.brain_config, key, default)

    def _init_brain_modules(self) -> None:
        """初始化Brain模块"""
        try:
            from .brain import (
                PedestrianDetectionManager,
                InstructionBrain,
            )
            from .brain.utils import BrainStats
        except ImportError as e:
            logger.warn(f"[InstructionBrainPPOEvaluator] Brain module import failed: {e}")
            return

        if self._brain_initialized:
            return

        logger.info("=" * 60)
        logger.info("[InstructionBrainPPOEvaluator] Initializing Instruction Brain module...")
        logger.info("=" * 60)

        # 初始化行人检测器
        if self.pedestrian_enabled:
            self.pedestrian_manager = PedestrianDetectionManager(
                enabled=True,
                detector_type=self._get_brain_config_value("pedestrian_detector", "yolov8n"),
                device=self.brain_device,
                confidence=self._get_brain_config_value("pedestrian_confidence", 0.25),
                checkpoint_path=self._get_brain_config_value("pedestrian_ckpt_path", None),
            )
            logger.info(f"[PedestrianDetector] Initialized YOLOv8n pedestrian detector")
            logger.info(f"  - Device: {self.brain_device}")
            logger.info(f"  - Confidence threshold: {self._get_brain_config_value('pedestrian_confidence', 0.25)}")
            logger.info(f"  - Checkpoint path: {self._get_brain_config_value('pedestrian_ckpt_path', 'default/auto-download')}")

        # 初始化InstructionBrain
        if self.brain_enabled:
            self.instruction_brain = InstructionBrain(
                model_type=self._get_brain_config_value("model_type", "qwen3_vl"),
                device=self.brain_device,
                model_id=self._get_brain_config_value("model_id", None),
                model_path=self._get_brain_config_value("model_path", None),
                max_history_frames=self._get_brain_config_value("max_history_frames", 5),
                save_frames=self._get_brain_config_value("save_frames", True),
                output_dir=self.output_dir,
                log_prompt=self._get_brain_config_value("log_prompt", True),
                save_prompt_to_file=self._get_brain_config_value("save_prompt_to_file", True),
                save_frame_images=self._get_brain_config_value("save_frame_images", True),
                frame_images_root=self._get_brain_config_value("frame_images_root",
                    "./brain_eval_records/frame_images"),
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
            )
            logger.info(f"[InstructionBrain] Initialized instruction optimization model")
            logger.info(f"  - Call mode: {self._get_brain_config_value('call_mode', 'local_hf')}")
            logger.info(f"  - Model type: {self._get_brain_config_value('model_type', 'qwen3_vl')}")
            logger.info(f"  - Model path: {self._get_brain_config_value('model_path', 'N/A')}")
            logger.info(f"  - API base URL: {self._get_brain_config_value('api_base_url', 'N/A')}")
            logger.info(f"  - History frames: {self._get_brain_config_value('max_history_frames', 5)}")
            logger.info(f"  - Save frame images: {self._get_brain_config_value('save_frame_images', True)}")

        logger.info(f"  - Brain enabled: {self.brain_enabled}")
        logger.info(f"  - Pedestrian detection enabled: {self.pedestrian_enabled}")
        logger.info(f"  - Output directory: {self.output_dir}")
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
        """从observation中提取RGB图像，优先使用overhead视角"""
        # 优先检查overhead视角图像（鸟瞰图视角，用于行人检测）
        overhead_keys = ["agent_0_overhead_front_rgb", "overhead_rgb", "agent_0_overhead_rgb"]
        for key in overhead_keys:
            if key in obs and obs[key] is not None:
                rgb = obs[key]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                # 处理通道在前的格式 (C, H, W) -> (H, W, C)
                if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                    rgb = np.transpose(rgb, (1, 2, 0))
                elif len(rgb.shape) == 3 and rgb.shape[2] == 3:
                    pass  # (H, W, C) 格式，直接使用
                elif len(rgb.shape) == 2:
                    rgb = np.stack([rgb] * 3, axis=-1)  # 灰度图转RGB
                return rgb

        # 其次检查third视角
        third_keys = ["agent_0_third_rgb", "third_rgb", "RGB"]
        for key in third_keys:
            if key in obs and obs[key] is not None:
                rgb = obs[key]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                    rgb = np.transpose(rgb, (1, 2, 0))
                return rgb

        # 通用fallback
        rgb_keys = ["rgb", "color", "image"]
        for key in rgb_keys:
            if key in obs and obs[key] is not None:
                rgb = obs[key]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                    rgb = np.transpose(rgb, (1, 2, 0))
                return rgb

        # 打印可用keys用于调试
        available_keys = list(obs.keys()) if isinstance(obs, dict) else []
        if available_keys:
            logger.debug(f"[Evaluator] 可用的observation keys: {available_keys}")
        return None

    def _get_instruction_from_observation(self, obs: Dict[str, Any]) -> str:
        """从observation中获取指令（优先使用原始指令）"""
        # 重要：原始指令存储在agent_0_falcon_instruction中，instruction键已被优化后的指令覆盖
        instruction_keys = ["agent_0_falcon_instruction", "instruction", "Instruction", "text", "goal"]
        for key in instruction_keys:
            if key in obs and obs[key] is not None:
                instr = obs[key]
                # 如果是numpy数组（uint8字节数组），需要解码
                if isinstance(instr, np.ndarray):
                    # 找到非零部分并解码
                    if instr.dtype == np.uint8:
                        non_zero_mask = instr != 0
                        if non_zero_mask.sum() > 0:
                            decoded = bytes(instr[non_zero_mask]).decode('utf-8', errors='ignore')
                            return decoded.strip()
                    # 如果是其他数值类型，尝试作为ASCII解码
                    if instr.dtype in (np.int32, np.int64, np.float32, np.float64):
                        non_zero_mask = instr != 0
                        if non_zero_mask.sum() > 0:
                            try:
                                decoded = bytes(instr[non_zero_mask].astype(np.uint8)).decode('utf-8', errors='ignore')
                                return decoded.strip()
                            except:
                                pass
                    return str(instr) if len(str(instr)) < 100 else ""
                # 如果是tensor
                elif isinstance(instr, torch.Tensor):
                    instr_np = instr.cpu().numpy()
                    if instr_np.dtype == np.uint8:
                        non_zero_mask = instr_np != 0
                        if non_zero_mask.sum() > 0:
                            decoded = bytes(instr_np[non_zero_mask]).decode('utf-8', errors='ignore')
                            return decoded.strip()
                    # 如果是其他数值类型，尝试作为ASCII解码
                    elif instr_np.dtype in (np.int32, np.int64, np.float32, np.float64):
                        non_zero_mask = instr_np != 0
                        if non_zero_mask.sum() > 0:
                            try:
                                decoded = bytes(instr_np[non_zero_mask].astype(np.uint8)).decode('utf-8', errors='ignore')
                                return decoded.strip()
                            except:
                                pass
                    return str(instr) if len(str(instr)) < 100 else ""
                # 如果是字符串，直接返回
                return str(instr)
        return ""

    def _detect_pedestrian(self, rgb_image: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """检测行人"""
        if self.pedestrian_manager is None:
            return {"pedestrian_detected": False, "pedestrian_count": 0}
        return self.pedestrian_manager.detect_frame(rgb_image, frame_id)

    def _action_to_string(self, action: Any) -> str:
        """动作转字符串"""
        action_names = {
            0: "STOP",
            1: "FORWARD",
            2: "TURN_LEFT",
            3: "TURN_RIGHT",
            4: "WAIT",
            5: "BACKWARD"
        }
        
        # 处理numpy数组
        if isinstance(action, np.ndarray):
            if action.size > 0:
                action = int(action.item())
            else:
                return "UNKNOWN_EMPTY"
        
        # 处理标量类型
        if isinstance(action, (int, np.integer)):
            return action_names.get(int(action), f"UNKNOWN_{action}")
        
        # 处理其他有item()方法的对象（如tensor）
        if hasattr(action, 'item'):
            try:
                return action_names.get(int(action.item()), f"UNKNOWN_{action.item()}")
            except:
                pass
        
        return str(action)

    def _load_eval_checkpoint(self, checkpoint_path: str):
        """
        加载已完成的评估checkpoint

        Returns:
            tuple: (stats_episodes, ep_eval_count, actions_record, completed_episodes_ids)
        """
        if not os.path.exists(checkpoint_path):
            logger.info(f"No evaluation checkpoint found at {checkpoint_path}, starting fresh evaluation.")
            return {}, defaultdict(lambda: 0), defaultdict(list), set()

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            stats_episodes = {}
            for key_str, stats in checkpoint_data.get('stats_episodes', {}).items():
                parts = key_str.split('|')
                if len(parts) == 3:
                    scene_id, episode_id, eval_count = parts[0], parts[1], int(parts[2])
                    stats_episodes[((scene_id, episode_id), eval_count)] = stats

            ep_eval_count = defaultdict(lambda: 0)
            for key_str, count in checkpoint_data.get('ep_eval_count', {}).items():
                parts = key_str.split('|')
                if len(parts) == 2:
                    scene_id, episode_id = parts[0], parts[1]
                    ep_eval_count[(scene_id, episode_id)] = count

            actions_record = defaultdict(list)
            for key_str, actions in checkpoint_data.get('actions_record', {}).items():
                parts = key_str.split('|')
                if len(parts) == 3:
                    scene_id, episode_id, eval_count = parts[0], parts[1], int(parts[2])
                    actions_record[(scene_id, episode_id, eval_count)] = actions

            completed_episodes_ids = set()
            for key_str in checkpoint_data.get('stats_episodes', {}).keys():
                parts = key_str.split('|')
                if len(parts) == 3:
                    scene_id, episode_id = parts[0], parts[1]
                    completed_episodes_ids.add((scene_id, episode_id))

            logger.info(f"Loaded evaluation checkpoint: {len(stats_episodes)} completed episodes, "
                       f"{len(completed_episodes_ids)} unique episodes evaluated.")

            return stats_episodes, ep_eval_count, actions_record, completed_episodes_ids

        except Exception as e:
            logger.error(f"Error loading evaluation checkpoint: {e}")
            logger.info("Starting fresh evaluation.")
            return {}, defaultdict(lambda: 0), defaultdict(list), set()

    def _save_eval_checkpoint(self, checkpoint_path: str, stats_episodes: Dict,
                              ep_eval_count: defaultdict, actions_record: defaultdict) -> None:
        """保存当前的评估checkpoint"""
        try:
            checkpoint_data = {
                'stats_episodes': {},
                'ep_eval_count': {},
                'actions_record': {}
            }

            for ((scene_id, episode_id), eval_count), stats in stats_episodes.items():
                key_str = f"{scene_id}|{episode_id}|{eval_count}"
                checkpoint_data['stats_episodes'][key_str] = stats

            for (scene_id, episode_id), count in ep_eval_count.items():
                key_str = f"{scene_id}|{episode_id}"
                checkpoint_data['ep_eval_count'][key_str] = count

            for (scene_id, episode_id, eval_count), actions in actions_record.items():
                key_str = f"{scene_id}|{episode_id}|{eval_count}"
                checkpoint_data['actions_record'][key_str] = actions

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.info(f"Saved evaluation checkpoint: {len(stats_episodes)} completed episodes.")

        except Exception as e:
            logger.error(f"Error saving evaluation checkpoint: {e}")

    def _make_episode_record(self, episode_id: str, instruction: str) -> Dict[str, Any]:
        return {
            "episode_id": episode_id,
            "instruction": instruction,
            "frames": [],
            "instruction_modifications": [],
            "total_frames": 0,
            "frames_with_pedestrian": 0,
            "brain_calls": 0,
            "instruction_changes": 0,
        }

    def _make_brain_episode_id(self, env_idx: int, episode_id: str, scene_id: Optional[str]) -> str:
        self._brain_episode_uid_counter += 1
        scene_name = os.path.basename(scene_id) if scene_id else "unknown_scene"
        scene_name = (
            scene_name
            .replace(".basis.glb", "")
            .replace(".glb", "")
            .replace(".basis", "")
            .replace(".", "_")
        )
        safe_episode_id = str(episode_id).replace("/", "_").replace("|", "_")
        return f"env{env_idx}_{scene_name}_ep{safe_episode_id}_run{self._brain_episode_uid_counter}"

    def _reset_env_state(self, env_idx: int, clear_scene_cache: bool = False) -> None:
        self._env_episode_ids.pop(env_idx, None)
        self._env_brain_episode_ids.pop(env_idx, None)
        self._env_instructions.pop(env_idx, None)
        self._env_original_instructions.pop(env_idx, None)
        self._env_frame_counters.pop(env_idx, None)
        self._env_episode_records.pop(env_idx, None)

        if clear_scene_cache:
            self._env_scene_ids.pop(env_idx, None)
            self._env_scene_initialized.pop(env_idx, None)
            self._env_scene_init_episode_ids.pop(env_idx, None)

    def _remap_env_state_after_pause(self, active_env_indices: List[int]) -> None:
        def _remap(mapping: Dict[int, Any]) -> Dict[int, Any]:
            return {
                new_idx: mapping[old_idx]
                for new_idx, old_idx in enumerate(active_env_indices)
                if old_idx in mapping
            }

        self._env_episode_ids = _remap(self._env_episode_ids)
        self._env_brain_episode_ids = _remap(self._env_brain_episode_ids)
        self._env_instructions = _remap(self._env_instructions)
        self._env_original_instructions = _remap(self._env_original_instructions)
        self._env_frame_counters = _remap(self._env_frame_counters)
        self._env_episode_records = _remap(self._env_episode_records)
        self._env_scene_ids = _remap(self._env_scene_ids)
        self._env_scene_initialized = _remap(self._env_scene_initialized)
        self._env_scene_init_episode_ids = _remap(self._env_scene_init_episode_ids)

    def _start_episode(self, env_idx: int, episode_id: str, instruction: str, force_reset: bool = False,
                       scene_id: str = None) -> None:
        """开始新的episode

        Args:
            env_idx: 环境索引
            episode_id: episode ID
            instruction: episode 指令
            force_reset: 是否强制重置状态（即使 episode_id 相同）
            scene_id: 场景ID，用于场景状态缓存
        """
        episode_id_str = str(episode_id)
        current_episode_id = self._env_episode_ids.get(env_idx, "")
        current_brain_episode_id = self._env_brain_episode_ids.get(env_idx, "")
        current_original_instruction = self._env_original_instructions.get(env_idx, "")
        current_scene_id = self._env_scene_ids.get(env_idx, "")
        scene_initialized = self._env_scene_initialized.get(env_idx, False)
        
        # ================================================================
        # 场景状态缓存优化：
        # 当同一场景内的 episode 切换时，不需要重新初始化场景
        # 只在场景变化时才需要完整的场景初始化
        # ================================================================
        is_same_scene = (
            scene_id is not None and 
            current_scene_id == scene_id and 
            scene_initialized
        )
        
        # 检测是否需要完全重置（场景变化或强制重置）
        scene_changed = (scene_id is not None and current_scene_id != scene_id)
        
        # 场景内部 episode 切换时的快速路径：
        # 如果是同一场景内的 episode 切换，且只有 episode_id 变化，不需要重置 Brain 状态
        is_same_scene_episode_switch = (
            is_same_scene and 
            current_episode_id != episode_id_str
        )
        
        # 需要重置的情况：
        # 1. force_reset=True
        # 2. 场景变化了
        # 3. episode_id 变化且指令也变化了
        # 4. 指令变化了
        # 5. 当前指令为空
        should_reset = (
            force_reset or
            scene_changed or
            (current_episode_id != episode_id_str and current_original_instruction != instruction) or
            current_original_instruction != instruction or
            not current_original_instruction
        )
        
        # 同一场景内 episode 切换的优化处理
        if is_same_scene_episode_switch and not force_reset:
            # 场景未变化，只更新 episode 状态
            logger.info(f"[Evaluator] _start_episode: Fast scene-switch for Env {env_idx} Episode {episode_id} (scene: {scene_id})")
            logger.info(f"  Same scene, reusing cached scene state")
            logger.info(f"  prev_ep_id={current_episode_id} -> new_ep_id={episode_id_str}")
            
            brain_episode_id = self._make_brain_episode_id(env_idx, episode_id_str, scene_id)
            self._env_episode_ids[env_idx] = episode_id_str
            self._env_brain_episode_ids[env_idx] = brain_episode_id
            self._env_original_instructions[env_idx] = instruction
            self._env_instructions[env_idx] = instruction
            self._env_frame_counters[env_idx] = 0
            
            # 更新 episode 记录
            self._env_episode_records[env_idx] = self._make_episode_record(
                episode_id=episode_id_str,
                instruction=instruction,
            )
            
            # 即使在同一场景内，也要调用 Brain 的 start_episode
            # 因为 end_episode 已经在 episode 结束时清理了 Brain 状态
            # start_episode 会重置 frame_history 等状态，为新 episode 做好准备
            if self.brain_enabled and self.instruction_brain is not None:
                self.instruction_brain.start_episode(brain_episode_id, instruction, env_idx=env_idx)
            
            logger.info(
                f"[Evaluator] Env {env_idx} Episode {episode_id} state UPDATED (fast path, brain_episode_id={brain_episode_id})"
            )
            return
        
        if not should_reset:
            logger.info(f"[Evaluator] _start_episode: Skipping reset for Env {env_idx} Episode {episode_id}")
            logger.info(f"  current_episode_id={current_episode_id}, episode_id={episode_id_str}")
            logger.info(f"  current_original_instruction={current_original_instruction[:50] if current_original_instruction else 'EMPTY'}")
            logger.info(f"  instruction={instruction[:50]}")
            return

        # 完整的 episode 重置逻辑
        logger.info(f"[Evaluator] _start_episode called for Env {env_idx} Episode {episode_id}")
        if current_episode_id and current_episode_id != episode_id_str:
            logger.info(f"  Previous Episode ID: {current_episode_id}")
            logger.info(f"  Previous Instruction: {current_original_instruction[:80] if current_original_instruction else 'EMPTY'}...")
        logger.info(f"  New Instruction: {instruction[:80]}...")
        
        # 更新场景状态缓存
        if scene_id is not None:
            if scene_changed:
                logger.info(f"  Scene changed: {current_scene_id} -> {scene_id}")
            self._env_scene_ids[env_idx] = scene_id
            self._env_scene_initialized[env_idx] = True
            self._env_scene_init_episode_ids[env_idx] = episode_id_str

        brain_episode_id = self._make_brain_episode_id(env_idx, episode_id_str, scene_id)
        self._env_episode_ids[env_idx] = episode_id_str
        self._env_brain_episode_ids[env_idx] = brain_episode_id
        self._env_original_instructions[env_idx] = instruction
        self._env_instructions[env_idx] = instruction
        self._env_frame_counters[env_idx] = 0
        self._env_episode_records[env_idx] = self._make_episode_record(
            episode_id=episode_id_str,
            instruction=instruction,
        )

        if self.brain_enabled and self.instruction_brain is not None:
            self.instruction_brain.start_episode(brain_episode_id, instruction, env_idx=env_idx)

        logger.info(
            f"[Evaluator] Env {env_idx} Episode {episode_id} state RESET COMPLETE "
            f"(brain_episode_id={brain_episode_id}, prev_brain_episode_id={current_brain_episode_id or 'EMPTY'})"
        )

    def _process_frame(self, step_number: int, obs: Dict[str, Any], action: Any,
                       pedestrian_info: Dict[str, Any]) -> None:
        """处理单帧数据"""
        self._frame_counter = step_number

        # 记录帧
        frame_record = {
            "step": step_number,
            "action": self._action_to_string(action),
            "pedestrian_detected": pedestrian_info.get("pedestrian_detected", False),
            "pedestrian_count": pedestrian_info.get("pedestrian_count", 0),
        }
        self.current_episode_record["frames"].append(frame_record)
        self.current_episode_record["total_frames"] += 1

        if pedestrian_info.get("pedestrian_detected"):
            self.current_episode_record["frames_with_pedestrian"] += 1

        # 指令优化（只在有行人时）
        if (self.brain_enabled and pedestrian_info.get("pedestrian_detected") and
                self.instruction_brain is not None):
            rgb_image = self._extract_rgb_observation(obs)
            if rgb_image is not None:
                result = self.instruction_brain.optimize_instruction(
                    original_instruction=self.original_instruction,
                    current_frame=rgb_image,
                    history_frames=(self.instruction_brain.frame_history[-5:]
                                   if self.instruction_brain.frame_history else None),
                    pedestrian_info=pedestrian_info,
                    frame_id=step_number,  # 传递帧ID用于节流
                )

                if result.should_modify:
                    old_instruction = self.current_instruction
                    self.current_instruction = result.optimized_instruction

                    mod_record = {
                        "step": step_number,
                        "original": old_instruction,
                        "optimized": result.optimized_instruction,
                        "reasoning": result.reasoning,
                        "safety_level": result.safety_level,
                    }
                    self.current_episode_record["instruction_modifications"].append(mod_record)
                    self.current_episode_record["instruction_changes"] += 1

                    logger.info(f"\n[Eval-Instruction Change] Episode {self._episode_id}, Step {step_number}")
                    logger.info(f"  Original: {old_instruction[:60]}...")
                    logger.info(f"  Optimized: {result.optimized_instruction[:60]}...")

                self.current_episode_record["brain_calls"] += 1

    def _end_episode(self, env_idx: int, episode_id: str, success: bool, metrics: Dict[str, float]) -> None:
        """结束当前episode"""
        brain_episode_id = self._env_brain_episode_ids.get(env_idx, str(episode_id))
        episode_record = self._env_episode_records.get(env_idx)
        if episode_record is None:
            episode_record = self._make_episode_record(
                episode_id=str(episode_id),
                instruction=self._env_original_instructions.get(env_idx, ""),
            )
            self._env_episode_records[env_idx] = episode_record

        episode_record["success"] = success
        episode_record["metrics"] = metrics
        self.evaluation_records.append(episode_record)

        if self.brain_enabled and self.instruction_brain is not None:
            # 使用传入的episode_id，而不是self._episode_id，确保正确清理
            self.instruction_brain.end_episode(brain_episode_id, env_idx=env_idx)
            logger.debug(
                f"[Evaluator] Ended env {env_idx} episode {episode_id} "
                f"(brain_episode_id={brain_episode_id}) for Brain module"
            )

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        """
        评估智能体

        Args:
            agent: 加载的策略
            envs: 向量化环境
            config: 配置对象
            checkpoint_index: 检查点索引
            step_id: 训练步数
            writer: 日志记录器
            device: PyTorch设备
            obs_transforms: 观察变换器列表
            env_spec: 环境规范
            rank0_keys: 只在rank0记录的键集合
        """
        # 设置checkpoint路径
        config_name = getattr(config.habitat_baselines, 'eval_config_name', 'default')
        dataset_name = config.habitat.dataset.data_path.split('/')[-1].replace('.json.gz', '').replace('.json', '')

        checkpoint_dir = os.path.join(
            config.habitat_baselines.checkpoint_folder,
            "eval_checkpoints",
            f"{config_name}_{dataset_name}"
        )
        checkpoint_path = os.path.join(checkpoint_dir, f"eval_progress_ckpt_{checkpoint_index}.json")

        logger.info(f"Evaluation checkpoint will be saved to: {checkpoint_path}")

        # 加载已完成的评估checkpoint
        stats_episodes, ep_eval_count, actions_record, completed_episodes_ids = self._load_eval_checkpoint(checkpoint_path)

        success_cal = 0
        for stats in stats_episodes.values():
            if 'success' in stats:
                success_cal += stats['success']

        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )

        if len(config.habitat_baselines.eval.video_option) > 0:
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items()}, {}
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
        else:
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep, initial=len(stats_episodes))
        agent.eval()

        # 内存清理
        # 设置为极大值以禁用定期清理，避免eval速度变慢
        memory_cleanup_interval = 99999999
        episodes_since_cleanup = 0

        # 统计报告间隔
        completed_episodes_count = len(stats_episodes)
        stats_report_interval = 50

        # checkpoint保存间隔
        checkpoint_save_interval = 10
        episodes_since_last_save = 0

        # 跳过已完成episode的env
        envs_skipping = set()

        # ================================================================
        # 关键修复：不要在这里预先获取指令
        # 让评估循环在每个 episode 正确获取指令
        # 初始状态应该为空，这样第一个 episode 就会被正确初始化
        # ================================================================
        # self.original_instruction = ""
        # self.current_instruction = ""
        # logger.info(f"[Evaluator] Waiting for first episode instruction...")

        self._env_episode_ids.clear()
        self._env_brain_episode_ids.clear()
        self._env_instructions.clear()
        self._env_original_instructions.clear()
        self._env_frame_counters.clear()
        self._env_episode_records.clear()
        self._env_scene_ids.clear()
        self._env_scene_initialized.clear()
        self._env_scene_init_episode_ids.clear()
        self._brain_episode_uid_counter = 0
        self._tracked_episodes: Set = set()  # 初始化追踪已处理的 episode

        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            # ================================================================
            # Episode 状态管理：检测并处理 episode 切换
            # ================================================================
            # 使用 episode_id 变化检测新 episode，而不是依赖复杂的跟踪逻辑
            current_episodes_info = envs.current_episodes()

            for i in range(envs.num_envs):
                episode_id = current_episodes_info[i].episode_id
                ep_obj = current_episodes_info[i]

                # ============================================================
                # 关键修复：必须从 Episode 对象获取指令，而不是依赖缓存
                # ============================================================
                # 从 Episode 对象获取指令（最高优先级）
                ep_instruction = None
                if hasattr(ep_obj, 'instruction') and ep_obj.instruction:
                    instr_data = ep_obj.instruction
                    if hasattr(instr_data, 'instruction_text'):
                        ep_instruction = instr_data.instruction_text
                    elif isinstance(instr_data, str):
                        ep_instruction = instr_data
                
                # 如果从 Episode 对象获取失败，尝试从 observation 获取
                if not ep_instruction:
                    obs_i = observations[i] if i < len(observations) else {}
                    if "agent_0_falcon_instruction" in obs_i:
                        instr = obs_i["agent_0_falcon_instruction"]
                        if isinstance(instr, np.ndarray) and instr.dtype == np.uint8:
                            non_zero_mask = instr != 0
                            if non_zero_mask.sum() > 0:
                                decoded = bytes(instr[non_zero_mask]).decode('utf-8', errors='ignore')
                                ep_instruction = decoded.strip()
                    if not ep_instruction:
                        ep_instruction = self._get_instruction_from_observation(obs_i)

                if not ep_instruction:
                    logger.warning(f"[Evaluator] Cannot get instruction for Episode {episode_id}, skipping...")
                    continue

                # ============================================================
                # 关键修复：无论 episode_id 是否变化，都检查指令是否正确
                # ============================================================
                # 检测是否需要重置状态
                # 注意：这里使用 episode_id 的字符串比较，以及指令的比较
                episode_id_str = str(episode_id)
                current_env_episode_id = self._env_episode_ids.get(i, "")
                current_env_instruction = self._env_original_instructions.get(i, "")
                is_new_episode = (current_env_episode_id != episode_id_str)
                is_instruction_changed = (current_env_instruction != ep_instruction)
                
                # 需要重置的情况：
                # 1. episode_id 变化了
                # 2. 或者指令变化了（即使 episode_id 相同，可能是同一个 episode 的不同评估轮次）
                # 3. 或者 self.original_instruction 为空（初始状态）
                need_reset = is_new_episode or is_instruction_changed or not current_env_instruction

                # 检查是否应该跳过这个 env（已完成所有评估）
                episode_key = (ep_obj.scene_id, episode_id)
                current_eval_count = ep_eval_count.get(episode_key, 0)
                should_skip = current_eval_count >= evals_per_ep

                if should_skip:
                    # 跳过已完成的 env，不重置状态
                    continue

                if need_reset:
                    # 新的 episode 开始，需要重置所有状态
                    logger.info(f"\n[Episode Start] Scene: {ep_obj.scene_id.split('/')[-1]}, "
                               f"Episode: {episode_id}")
                    logger.info(f"[Episode Instruction] {ep_instruction[:512]}...")

                    # 检查是否可以使用快速路径（同一场景内 episode 切换）
                    scene_id = ep_obj.scene_id
                    can_use_fast_path = (
                        self._env_scene_ids.get(i) == scene_id and
                        self._env_scene_initialized.get(i, False) and
                        is_new_episode
                    )
                    
                    if can_use_fast_path:
                        # 同一场景内 episode 切换，使用快速路径
                        logger.info(f"[Episode Reset] Using fast path (same scene: {scene_id.split('/')[-1]})")
                        self._start_episode(i, episode_id, ep_instruction, force_reset=False, scene_id=scene_id)
                    else:
                        # 场景变化或首次加载，需要完整重置
                        logger.info(f"[Episode Reset] force_reset=True, calling _start_episode")
                        self._start_episode(i, episode_id, ep_instruction, force_reset=True, scene_id=scene_id)
                    
                    self._tracked_episodes.add(episode_key)

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    **space_lengths,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # 处理动作
            if hasattr(agent, '_agents') and agent._agents[0]._actor_critic.action_distribution_type == 'categorical':
                step_data = [a.numpy() for a in action_data.env_actions.cpu()]
            elif is_continuous_action_space(env_spec.action_space):
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            # 对需要跳过的env发送STOP动作
            for i in envs_skipping:
                if i < len(step_data):
                    if isinstance(step_data[i], np.ndarray):
                        step_data[i] = np.zeros_like(step_data[i])
                    else:
                        step_data[i] = 0

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            for i in range(envs.num_envs):
                episode_key = (
                    current_episodes_info[i].scene_id,
                    current_episodes_info[i].episode_id,
                    ep_eval_count[
                        (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    ]
                )

                action_value = step_data[i]
                if isinstance(action_value, np.ndarray):
                    stored_action = {
                        "type": "array",
                        "value": action_value.tolist()
                    }
                else:
                    stored_action = {
                        "type": "array",
                        "value": np.array(action_value).tolist()
                    }

                actions_record[episode_key].append(stored_action)

            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)

            # ========================================================
            # Brain处理：行人检测和指令优化（影响下一步决策）
            # 多环境隔离：每个env维护独立的指令状态
            # ========================================================

            if self.brain_enabled and self.instruction_brain is not None:
                for i in range(envs.num_envs):
                    obs_i = observations[i]
                    rgb_image = self._extract_rgb_observation(obs_i)

                    # 与 trainer 对齐：每一帧都从原始指令开始，
                    # Brain 的改写只对当前帧/当前决策生效，不跨帧累计。
                    env_original_instr = self._env_original_instructions.get(i, "")
                    env_current_instr = env_original_instr
                    env_episode_id = self._env_brain_episode_ids.get(
                        i, self._env_episode_ids.get(i, str(current_episodes_info[i].episode_id))
                    )
                    env_frame_id = self._env_frame_counters.get(i, 0)
                    env_episode_record = self._env_episode_records.get(i)

                    if rgb_image is not None:
                        # 检测行人
                        pedestrian_info = self._detect_pedestrian(rgb_image, env_frame_id)

                        # 记录行人检测统计
                        if pedestrian_info.get("pedestrian_detected", False):
                            if not hasattr(self, '_pedestrian_stats_logged'):
                                self._pedestrian_stats_logged = 0
                            self._pedestrian_stats_logged += 1
                            if self._pedestrian_stats_logged <= 3:
                                logger.info(f"[Pedestrian Detection] Env{i} Frame {env_frame_id}: "
                                           f"Detected {pedestrian_info.get('pedestrian_count', 0)} pedestrian(s)")

                        # 检查是否需要调用brain
                        need_call_brain = self.instruction_brain.should_call_brain(
                            pedestrian_info, episode_id=env_episode_id, frame_id=env_frame_id)

                        if need_call_brain:
                            logger.info(f"\n[Eval-Step] Env{i} Frame {env_frame_id}: Calling Brain")

                            result = self.instruction_brain.optimize_instruction(
                                original_instruction=env_original_instr,
                                current_frame=rgb_image,
                                history_frames=(self.instruction_brain.get_frame_history(env_episode_id)[-5:]
                                               if env_episode_id else None),
                                pedestrian_info=pedestrian_info,
                                episode_id=env_episode_id,
                                env_idx=i,
                                frame_id=env_frame_id,
                            )
                            if env_episode_record is not None:
                                env_episode_record.setdefault("brain_calls", 0)
                                env_episode_record["brain_calls"] += 1

                            # 记录优化后的指令（仅该env生效）
                            if result.should_modify and self.instruction_brain.should_update_instruction(
                                env_current_instr, result.optimized_instruction
                            ):
                                env_current_instr = result.optimized_instruction
                                if env_episode_record is not None:
                                    env_episode_record.setdefault("instruction_changes", 0)
                                    env_episode_record["instruction_changes"] += 1
                                logger.info(f"  Env{i} instruction optimized: {result.optimized_instruction[:80]}...")

                        # 记录帧数据（传入env_idx和episode_id支持多环境隔离）
                        action_id = 1
                        action_val = step_data[i]
                        if isinstance(action_val, np.ndarray):
                            action_id = int(action_val.item()) if action_val.size > 0 else 1
                        elif isinstance(action_val, (int, np.integer)):
                            action_id = int(action_val)
                        elif hasattr(action_val, 'item'):
                            action_id = int(action_val.item())

                        self.instruction_brain.record_frame(
                            frame_id=env_frame_id,
                            image=rgb_image,
                            action=self._action_to_string(action_id),
                            action_id=action_id,
                            instruction=env_current_instr,
                            pedestrian_info=pedestrian_info,
                            episode_id=env_episode_id,
                            env_idx=i,
                        )

                        # 递增该环境的帧计数器
                        self._env_frame_counters[i] = env_frame_id + 1
                    else:
                        if not hasattr(self, '_obs_keys_logged'):
                            self._obs_keys_logged = True
                            logger.warning(f"[Evaluator] Cannot extract RGB image")

                    # 保存该帧使用的指令，供后续 observation 注入；
                    # 下一帧会重新从原始指令开始计算。
                    self._env_instructions[i] = env_current_instr

                # 将各环境的优化后指令注入到observations中（每个env独立）
                for i in range(envs.num_envs):
                    env_instr = self._env_instructions.get(i, self._env_original_instructions.get(i, ""))
                    obs = observations[i]
                    if "instruction" in obs:
                        observations[i]["instruction"] = env_instr
                    if "agent_0_falcon_instruction" in obs:
                        INSTR_MAX_LEN = 512
                        instr_bytes = env_instr.encode('utf-8')
                        if len(instr_bytes) > INSTR_MAX_LEN:
                            instr_bytes = instr_bytes[:INSTR_MAX_LEN]
                        instr_array = np.zeros(INSTR_MAX_LEN, dtype=np.uint8)
                        instr_array[:len(instr_bytes)] = np.frombuffer(instr_bytes, dtype=np.uint8)
                        observations[i]["agent_0_falcon_instruction"] = instr_array

            batch = batch_obs(
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()

            envs_skipping_this_step = set(envs_skipping)

            for i in range(envs.num_envs):
                next_ep_key = (
                    next_episodes_info[i].scene_id,
                    next_episodes_info[i].episode_id,
                )
                if ep_eval_count[next_ep_key] >= evals_per_ep:
                    envs_skipping.add(i)
                else:
                    envs_skipping.discard(i)

            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                    and i not in envs_skipping
                ):
                    envs_to_pause.append(i)

                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].any().item():
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].any().item():
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )

                    # 如果正在跳过，不记录统计，但仍需结束episode清理状态
                    if i in envs_skipping_this_step:
                        # 跳过统计，但仍需重置所有状态以保持一致性
                        if self.brain_enabled and self.instruction_brain is not None:
                            self._end_episode(i, current_episodes_info[i].episode_id, success=False, metrics={})

                        # 重置所有状态以保持一致性，防止残留状态影响下一个episode
                        self._reset_env_state(i)
                        
                        current_episode_reward[i] = 0
                        continue

                    # 正常结束episode，先清理Brain状态
                    if self.brain_enabled and self.instruction_brain is not None:
                        self._end_episode(i, current_episodes_info[i].episode_id,
                                         success=disp_info.get("success", False),
                                         metrics=extract_scalars_from_info(infos[i]))

                    # ============================================================
                    # 重要：episode 结束后必须重置所有状态
                    # 防止下一个 episode 残留上一个 episode 的状态
                    # 
                    # 注意：保留场景状态缓存 (_current_scene_id, _scene_initialized)
                    # 以便同一场景内的下一个 episode 使用快速路径
                    # ============================================================
                    self._reset_env_state(i)
                    # 注意：这里保留 _current_scene_id 和 _scene_initialized
                    # 以便同一场景内的 episode 切换使用快速路径
                    # 场景状态只有在场景真正变化时才会更新

                    # 如果这个 episode_key 完成了所有评估轮次，从追踪集合中移除
                    episode_key = (current_episodes_info[i].scene_id, current_episodes_info[i].episode_id)
                    if ep_eval_count.get(episode_key, 0) >= evals_per_ep:
                        self._tracked_episodes.discard(episode_key)

                    pbar.update()
                    episodes_since_cleanup += 1
                    completed_episodes_count += 1
                    episodes_since_last_save += 1

                    if "success" in disp_info:
                        success_cal += disp_info['success']
                        logger.info(f"Till now Success Rate: {success_cal/completed_episodes_count}")

                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    ep_eval_count[k] += 1
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # 定期输出平均结果
                    if completed_episodes_count % stats_report_interval == 0:
                        calculate_and_log_average_stats(stats_episodes, completed_episodes_count, logger)

                    # 定期保存checkpoint
                    if episodes_since_last_save >= checkpoint_save_interval:
                        self._save_eval_checkpoint(checkpoint_path, stats_episodes, ep_eval_count, actions_record)
                        episodes_since_last_save = 0

                    # 定期内存清理
                    if episodes_since_cleanup >= memory_cleanup_interval:
                        logger.info(f"[Memory Cleanup] Processed {episodes_since_cleanup} episodes, cleaning up memory...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        try:
                            import ctypes
                            libc = ctypes.CDLL("libc.so.6")
                            libc.malloc_trim(0)
                        except Exception:
                            pass
                        episodes_since_cleanup = 0

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        scene_id = current_episodes_info[i].scene_id.split('/')[-1].split('.')[0]
                        logger.info(f"This is Scene ID: {scene_id}, Episode ID: {current_episodes_info[i].episode_id}.")

                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            images=rgb_frames[i][:-1],
                            scene_id=f"{current_episodes_info[i].scene_id}".split('/')[-1].split('.')[0],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = rgb_frames[i][-1:]

            not_done_masks = not_done_masks.to(device=device)
            if len(envs_to_pause) > 0:
                active_env_indices = [
                    idx for idx in range(envs.num_envs) if idx not in envs_to_pause
                ]
                self._remap_env_state_after_pause(active_env_indices)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        # 最终内存清理
        logger.info("[Memory Cleanup] Evaluation loop completed, performing final cleanup...")
        torch.cuda.empty_cache()
        gc.collect()

        # 保存最终的checkpoint
        self._save_eval_checkpoint(checkpoint_path, stats_episodes, ep_eval_count, actions_record)
        logger.info("Final evaluation checkpoint saved.")

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        # 保存 result.json
        result_path = os.path.join("output/", "result.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        evalai_result = {
            "SR": round(aggregated_stats.get("success", 0), 4),
            "SPL": round(aggregated_stats.get("spl", 0), 4),
            "PSC": round(aggregated_stats.get("psc", 0), 4),
            "H-Coll": round(aggregated_stats.get("human_collision", 0), 4),
            "Total": round(
                0.4 * aggregated_stats.get("success", 0)
                + 0.3 * aggregated_stats.get("spl", 0)
                + 0.3 * aggregated_stats.get("psc", 0),
                4,
            ),
        }

        with open(result_path, "w") as f:
            json.dump(evalai_result, f, indent=2)

        # 保存 actions.json
        actions_output_path = os.path.join("output/", "actions.json")
        os.makedirs(os.path.dirname(actions_output_path), exist_ok=True)
        serializable_actions = {
            f"{scene_id}|{episode_id}|{eval_count}": actions
            for (scene_id, episode_id, eval_count), actions in actions_record.items()
        }
        with open(actions_output_path, "w") as f:
            json.dump(serializable_actions, f, indent=2)

        # 清理Brain模块
        self._cleanup_brain_modules()
