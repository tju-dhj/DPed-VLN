# -*- coding: utf-8 -*-
"""
DPedBrainTrainerServer - 带InstructionBrain的机器人部署Flask HTTP服务器
======================================================================

继承自 DPedTrainerServer，增加了：
1. YOLOv8 行人检测（服务端实时检测）
2. InstructionBrain（Qwen3-VL等VLM）指令优化
3. 优化后的指令注入到策略网络进行动作预测

HTTP Endpoints:
    POST /reset_hiddens  - 初始化/重置 RNN hidden states + Brain episode
    POST /predict_action  - 接收观测（RGB图像、指令、GPS等），
                            运行行人检测 → Brain指令优化 → 策略推理，
                            返回预测动作

动作空间（由配置中的 num_actions 决定）:
    4动作: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
    6动作: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT, 4=PAUSE, 5=BACKWARD
"""

import io
import os
import pickle
import time
from typing import Any, Dict, Optional

import cv2
import hydra
import numpy as np
import torch
from PIL import Image
from flask import Flask, request
from gym import spaces

import habitat_baselines.rl.multi_agent  # noqa: F401
from habitat import logger
from habitat.config import read_write
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
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.ddp_utils import (
    get_distrib_size,
    is_slurm_batch_job,
    rank0_only,
)
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.utils.common import (
    batch_obs,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)

# 导入 Brain 模块
try:
    from .brain import (
        PedestrianDetectionManager,
        InstructionBrain,
        InstructionOptimizationResult,
    )
    BRAIN_MODULE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[DPedBrainTrainerServer] Brain module import failed: {e}")
    BRAIN_MODULE_AVAILABLE = False

app = Flask("dped_brain_trainer_server")


def center_crop(im, target_size=224):
    """Center crop image to target size (square)."""
    width, height = im.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    im = im.crop((left, top, right, bottom))
    if target_size is not None:
        im = im.resize((target_size, target_size), Image.BILINEAR)
    return im


@baseline_registry.register_trainer(name="dped_brain_trainer_server")
class DPedBrainTrainerServer(BaseRLTrainer):
    """
    DPed_pro Brain Flask HTTP Server Trainer.

    在 DPedTrainerServer 基础上集成 InstructionBrain：
    - 实时行人检测（YOLOv8）
    - VLM 指令优化（Qwen3-VL 等）
    - 优化后指令注入策略网络

    HTTP Endpoints:
        POST /reset_hiddens - 初始化RNN hidden states + Brain episode
        POST /predict_action - 接收观测，返回预测动作

    动作空间:
        4动作: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
        6动作: 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT, 4=PAUSE, 5=BACKWARD
    """

    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self._agent = None
        self.envs = None
        self.obs_transforms = []
        self._env_spec = None
        self._is_distributed = get_distrib_size()[2] > 1

        # Flask server state
        self.num_envs = 1
        self.real_rnn_states = None
        self.real_prev_actions = None
        self.real_not_done_masks = None

        # Device
        self.device = None

        # Brain 模块
        self.brain_enabled = False
        self.pedestrian_enabled = False
        self.instruction_brain: Optional[InstructionBrain] = None
        self.pedestrian_manager: Optional[PedestrianDetectionManager] = None
        self.brain_config = None

        # 当前 episode 的指令缓存（用于 brain 优化后的指令持久化）
        self._current_instruction: str = ""
        self._current_episode_id: str = ""

        # 帧计数器（用于节流和定期调用）
        self._frame_counter: int = 0

    def _init_brain(self) -> None:
        """初始化 InstructionBrain 和 PedestrianDetectionManager"""
        cfg = self.config.habitat_baselines
        if not hasattr(cfg, "brain"):
            logger.info("[DPedBrainTrainerServer] No brain config found, brain disabled")
            self.brain_enabled = False
            return

        brain_cfg = cfg.brain
        self.brain_config = brain_cfg

        if not brain_cfg.get("enabled", False):
            logger.info("[DPedBrainTrainerServer] Brain disabled in config")
            self.brain_enabled = False
            return

        if not BRAIN_MODULE_AVAILABLE:
            logger.warning("[DPedBrainTrainerServer] Brain module not available, brain disabled")
            self.brain_enabled = False
            return

        self.brain_enabled = True

        # 初始化行人检测管理器
        if brain_cfg.get("pedestrian_enabled", True):
            self.pedestrian_enabled = True
            try:
                self.pedestrian_manager = PedestrianDetectionManager(
                    detector_type=brain_cfg.get("pedestrian_detector", "yolov8n"),
                    confidence_threshold=brain_cfg.get("pedestrian_confidence", 0.25),
                    ckpt_path=brain_cfg.get("pedestrian_ckpt_path", ""),
                    device=str(self.device),
                )
                logger.info("[DPedBrainTrainerServer] PedestrianDetectionManager initialized")
            except Exception as e:
                logger.warning(f"[DPedBrainTrainerServer] Pedestrian detection init failed: {e}")
                self.pedestrian_enabled = False

        # 初始化 InstructionBrain
        try:
            model_type = brain_cfg.get("model_type", "qwen3_vl_8b")
            model_id = brain_cfg.get("model_id", None)
            model_path = brain_cfg.get("model_path", None)

            self.instruction_brain = InstructionBrain(
                model_type=model_type,
                device=str(self.device),
                model_id=model_id,
                model_path=model_path,
                max_history_frames=brain_cfg.get("max_history_frames", 5),
                max_new_tokens=brain_cfg.get("max_new_tokens", 256),
                temperature=brain_cfg.get("temperature", 0.7),
                top_p=brain_cfg.get("top_p", 0.8),
                brain_call_confidence=brain_cfg.get("brain_call_confidence", 0.7),
                num_actions=brain_cfg.get("num_actions", 4),
                min_brain_call_interval=brain_cfg.get("min_brain_call_interval", 3),
                call_brain_periodically=brain_cfg.get("call_brain_periodically", False),
                call_brain_interval=brain_cfg.get("call_brain_interval", 10),
                save_frames=False,
                output_dir=brain_cfg.get("output_dir", "./brain_records_server"),
                log_prompt=brain_cfg.get("log_prompt", True),
                save_prompt_to_file=brain_cfg.get("save_prompt_to_file", False),
                save_frame_images=brain_cfg.get("save_frame_images", False),
                frame_images_root=brain_cfg.get("frame_images_root", "./brain_records_server/frame_images"),
            )
            logger.info(f"[DPedBrainTrainerServer] InstructionBrain initialized: model={model_type}")
        except Exception as e:
            logger.warning(f"[DPedBrainTrainerServer] InstructionBrain init failed: {e}")
            self.instruction_brain = None
            self.brain_enabled = False

    def _create_obs_transforms(self):
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _create_agent(self, resume_state, **kwargs) -> AgentAccessMgr:
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

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if not self._is_distributed:
            return t
        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)
        return t.to(device=orig_device)

    def percent_done(self) -> float:
        return 0.0

    def eval(self) -> None:
        cfg = self.config.habitat_baselines
        self.device = (
            torch.device("cuda", cfg.torch_gpu_id)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if "tensorboard" in cfg.eval.video_option:
            assert len(cfg.tensorboard_dir) > 0
            os.makedirs(cfg.tensorboard_dir, exist_ok=True)
        if "disk" in cfg.eval.video_option:
            assert len(cfg.video_dir) > 0

        with TensorboardWriter(
            cfg.tensorboard_dir, flush_secs=self.flush_secs
        ) as writer:
            ckpt_path = cfg.eval_ckpt_path_dir
            self._eval_checkpoint(ckpt_path, writer, checkpoint_index=0)

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Load checkpoint, initialize Brain, and start Flask server."""
        logger.info(f"DPedBrainTrainerServer checkpoint_path: {checkpoint_path}")

        cfg_hb = self.config.habitat_baselines
        config = self.config.copy()

        eval_split = cfg_hb.eval.split if hasattr(cfg_hb.eval, "split") else self.config.habitat.dataset.split
        with read_write(config):
            config.habitat.dataset.split = eval_split

        # 1. Initialize env to get observation/action spaces
        if os.path.exists("observation_space_brain.pkl"):
            with open("observation_space_brain.pkl", "rb") as f:
                observation_space = pickle.load(f)
            with open("action_space_brain.pkl", "rb") as f:
                action_space = pickle.load(f)
        else:
            self._init_envs(config, is_eval=True)
            observation_space = self._env_spec.observation_space
            action_space = self._env_spec.action_space
            self.envs.close()
            with open("observation_space_brain.pkl", "wb") as f:
                pickle.dump(observation_space, f)
            with open("action_space_brain.pkl", "wb") as f:
                pickle.dump(action_space, f)

        logger.info(f"Observation space keys: {list(observation_space.spaces.keys())}")
        logger.info(f"Action space: {action_space}")

        # 2. Create agent and load checkpoint
        self._init_envs(config, is_eval=True)
        self._agent = self._create_agent(None)

        if (
            self._agent.actor_critic.should_load_agent_state
            and cfg_hb.eval.should_load_ckpt
        ):
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            self._agent.load_state_dict(ckpt_dict)

        self._agent.eval()

        # 3. Extract actor_critic
        self.actor_critic = self._agent.actor_critic

        action_shape, discrete_actions = get_action_space_info(
            self.actor_critic.policy_action_space
        )
        logger.info(f"Action shape: {action_shape}, discrete: {discrete_actions}")

        # 4. Initialize Brain (must be after device is set)
        self._init_brain()

        # 5. Register Flask routes
        app.add_url_rule(
            "/reset_hiddens",
            "reset_hiddens",
            self.reset_hiddens,
            methods=["POST"],
        )
        app.add_url_rule(
            "/predict_action",
            "predict_action",
            self.predict_action,
            methods=["POST"],
        )

        # 6. Start Flask server
        port = getattr(self.config, "FLASK_SERVER_PORT", 32146)
        logger.info(f"[DPedBrainTrainerServer] Brain enabled: {self.brain_enabled}")
        logger.info(f"[DPedBrainTrainerServer] Starting Flask server on 0.0.0.0:{port}")
        self.envs.close()
        app.run("0.0.0.0", port, threaded=True)

    def reset_hiddens(self):
        """Initialize RNN hidden states + Brain episode for a new episode."""
        with torch.no_grad():
            num_envs = self.num_envs

            if hasattr(self.actor_critic, "_active_policies") and len(
                self.actor_critic._active_policies
            ) > 0:
                underlying_policy = self.actor_critic._active_policies[0]
            else:
                underlying_policy = self.actor_critic

            hidden_size = underlying_policy.recurrent_hidden_size
            num_layers = underlying_policy.num_recurrent_layers

            gru_state = torch.zeros(
                num_envs,
                num_layers,
                hidden_size,
                device=self.device,
            )

            self.real_rnn_states = gru_state

            action_shape, _ = get_action_space_info(
                self.actor_critic.policy_action_space
            )
            self.real_prev_actions = torch.zeros(
                num_envs, *action_shape, device=self.device, dtype=torch.long
            )
            self.real_not_done_masks = torch.zeros(
                num_envs, 1, dtype=torch.uint8, device=self.device
            )

        # 重置帧计数器
        self._frame_counter = 0

        # 重置 Brain episode（如果启用）
        # 机器人端可能发送空 POST 或带 form 数据的 POST，两种都要兼容
        try:
            ep_id = request.form.get("ep_id", "unknown") if request.method == "POST" else "unknown"
        except Exception:
            ep_id = "unknown"
        try:
            inst_text = request.form.get("inst", "") if request.method == "POST" else ""
        except Exception:
            inst_text = ""

        self._current_episode_id = ep_id
        self._current_instruction = inst_text

        if self.brain_enabled and self.instruction_brain is not None:
            try:
                self.instruction_brain.start_episode(
                    episode_id=ep_id,
                    original_instruction=inst_text,
                )
                logger.info(f"[DPedBrainTrainerServer] Brain episode started: {ep_id}")
            except Exception as e:
                logger.warning(f"[DPedBrainTrainerServer] Brain start_episode failed: {e}")

        return {"status": "success", "brain_enabled": self.brain_enabled}

    # =========================================================================
    # 图像预处理（与 DPedTrainerServer 保持一致）
    # =========================================================================

    def _preprocess_rgb(self, rgb_bytes: bytes, sensor_key: str = "overhead_front_rgb") -> torch.Tensor:
        """
        Preprocess RGB — 只做 center_crop，返回 HWC uint8。
        Normalize / ToTensor 由 apply_obs_transforms_batch 统一处理。
        """
        rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        rgb_pil = center_crop(rgb_pil, target_size=224)
        rgb = np.array(rgb_pil)  # (H, W, 3) uint8
        return torch.from_numpy(rgb).unsqueeze(0)  # (1, H, W, 3) uint8

    def _preprocess_depth(self, depth_bytes: bytes, sensor_key: str = "overhead_front_depth") -> torch.Tensor:
        """
        Preprocess depth — center_crop，返回 (H, W, 1) float32 meters。
        匹配 Habitat env 格式，让模型内部 [..., 0] 正确工作。
        """
        depth_pil = Image.open(io.BytesIO(depth_bytes))
        depth_np = np.array(depth_pil)

        if depth_np.dtype == np.uint16:
            depth_np = depth_np.astype(np.float32) / 1000.0  # mm → m

        depth_pil = Image.fromarray((np.clip(depth_np, 0, 10) / 10 * 255).astype(np.uint8))
        depth_pil = center_crop(depth_pil, target_size=224)
        depth_np = np.array(depth_pil).astype(np.float32) / 255.0 * 10.0

        depth_np = depth_np[..., np.newaxis]  # (224, 224, 1)
        return torch.from_numpy(depth_np).unsqueeze(0)  # (1, H, W, 1) float32

    def _preprocess_instruction(self, instruction_text: str, max_len: int = 512) -> torch.Tensor:
        """Preprocess instruction text (character-level encoding)."""
        obs = torch.zeros(max_len, dtype=torch.long)
        for i, char in enumerate(instruction_text[:max_len]):
            obs[i] = ord(char) % 1000
        return obs

    def _preprocess_gps_compass(self, gps_compass_data: Dict[str, float]) -> torch.Tensor:
        """StartingPointGPSCompassSensor: [goal_x, goal_y, heading_sin, heading_cos]."""
        import math
        goal_x = gps_compass_data.get("goal_x", 0.0)
        goal_y = gps_compass_data.get("goal_y", 0.0)
        compass = gps_compass_data.get("compass", 0.0)
        return torch.tensor([goal_x, goal_y, math.sin(compass), math.cos(compass)], dtype=torch.float32)

    # =========================================================================
    # 行人检测（服务端 YOLOv8）
    # =========================================================================

    def _detect_pedestrians(self, rgb_bytes: bytes) -> Dict[str, Any]:
        """
        对 RGB 图像运行行人检测。

        Returns:
            pedestrian_info dict，格式与 PedestrianDetectionManager 一致:
                - pedestrian_detected: bool
                - pedestrian_count: int
                - raw_detections: list of dicts (bbox, confidence, class)
                - warning_level: str
        """
        if not self.pedestrian_enabled or self.pedestrian_manager is None:
            return {
                "pedestrian_detected": False,
                "pedestrian_count": 0,
                "raw_detections": [],
                "warning_level": "unknown",
            }

        try:
            # 将 bytes 解码为 numpy 数组
            rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
            rgb_np = np.array(rgb_pil)

            # 调用行人检测
            result = self.pedestrian_manager.detect(rgb_np)
            return result
        except Exception as e:
            logger.warning(f"[DPedBrainTrainerServer] Pedestrian detection failed: {e}")
            return {
                "pedestrian_detected": False,
                "pedestrian_count": 0,
                "raw_detections": [],
                "warning_level": "unknown",
            }

    # =========================================================================
    # 主推理端点
    # =========================================================================

    def predict_action(self):
        """
        接收机器人观测，运行 行人检测 → Brain指令优化 → 策略推理，返回动作。

        POST form fields:
            - ep_id: episode identifier (string)
            - rgb: RGB image file (JPEG bytes)  **必需**
            - depth: Depth image file (PNG bytes, 可选)
            - inst: Instruction text (string)
            - gps_x, gps_y: robot GPS coordinates (float, 可选)
            - compass: robot compass heading in radians (float, 可选)
            - goal_x, goal_y: goal relative to robot (float, 可选)

        Returns:
            JSON dict:
                - status: "success" or "error"
                - action: predicted action index (int)
                - instruction_modified: bool (brain是否修改了指令)
                - optimized_instruction: str (优化后的指令，仅当修改时)
                - pedestrian_detected: bool
                - pedestrian_count: int
                - time_info: timing breakdown
        """
        tic_total = time.time()

        try:
            # --- 1. Parse request ---
            tic_net = time.time()
            ep_id = request.form.get("ep_id", "unknown")

            rgb_file = request.files.get("rgb")
            if rgb_file is None:
                return {"status": "error", "message": "rgb file is required"}, 400
            rgb_bytes = rgb_file.read()

            depth_file = request.files.get("depth")
            depth_bytes = None
            if depth_file is not None:
                depth_bytes = depth_file.read()

            inst_text = request.form.get("inst", "")
            time_info_net = time.time() - tic_net

            # --- 2. Brain: 行人检测 + 指令优化 ---
            tic_brain = time.time()
            instruction_modified = False
            optimized_instruction = inst_text
            pedestrian_detected = False
            pedestrian_count = 0
            brain_inference_ms = 0.0

            # 更新 episode 追踪
            if ep_id != self._current_episode_id:
                self._current_episode_id = ep_id
                self._current_instruction = inst_text
                self._frame_counter = 0
                if self.brain_enabled and self.instruction_brain is not None:
                    try:
                        self.instruction_brain.start_episode(
                            episode_id=ep_id,
                            original_instruction=inst_text,
                        )
                    except Exception:
                        pass

            self._frame_counter += 1
            frame_id = self._frame_counter

            if self.brain_enabled and self.instruction_brain is not None:
                try:
                    # Step 1: 行人检测
                    ped_info = self._detect_pedestrians(rgb_bytes)
                    pedestrian_detected = ped_info.get("pedestrian_detected", False)
                    pedestrian_count = ped_info.get("pedestrian_count", 0)

                    # Step 2: 判断是否需要调用 Brain
                    if self.instruction_brain.should_call_brain(
                        ped_info, episode_id=ep_id, frame_id=frame_id
                    ):
                        # 解码 RGB 为 numpy 用于 Brain
                        rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
                        rgb_np = np.array(rgb_pil)

                        # Step 3: Brain 指令优化
                        result = self.instruction_brain.optimize_instruction(
                            original_instruction=inst_text,
                            current_frame=rgb_np,
                            history_frames=None,  # Brain 内部维护 frame history
                            pedestrian_info=ped_info,
                            episode_id=ep_id,
                            frame_id=frame_id,
                        )

                        brain_inference_ms = result.inference_time_ms

                        if result.should_modify:
                            optimized_instruction = result.optimized_instruction
                            instruction_modified = True
                            self._current_instruction = optimized_instruction
                            logger.info(
                                f"[Brain] Instruction modified (frame {frame_id}): "
                                f"'{inst_text[:50]}...' -> '{optimized_instruction[:50]}...' "
                                f"confidence={result.confidence:.2f} modifier={result.modifier_type.value}"
                            )

                    # Step 4: 记录帧（用于 Brain 历史）
                    action_label = "unknown"  # 将在推理后更新
                    rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
                    rgb_np = np.array(rgb_pil)
                    self.instruction_brain.record_frame(
                        frame_id=frame_id,
                        image=rgb_np,
                        action=action_label,
                        instruction=optimized_instruction,
                        pedestrian_info=ped_info,
                        action_id=1,  # 默认 FORWARD，实际值在推理后才知道
                        episode_id=ep_id,
                    )
                except Exception as e:
                    logger.warning(f"[DPedBrainTrainerServer] Brain optimization failed: {e}")
                    # 回退：使用原始指令
                    optimized_instruction = inst_text

            time_info_brain = time.time() - tic_brain

            # --- 3. Preprocess observations ---
            tic_prepare = time.time()
            rgb_tensor = self._preprocess_rgb(rgb_bytes)

            depth_tensor = None
            if depth_bytes is not None:
                depth_tensor = self._preprocess_depth(depth_bytes)

            # 使用优化后的指令（如果有）
            inst_tensor = self._preprocess_instruction(optimized_instruction)

            gps_compass_data = {
                "goal_x": float(request.form.get("goal_x", 0.0)),
                "goal_y": float(request.form.get("goal_y", 0.0)),
                "compass": float(request.form.get("compass", 0.0)),
            }
            gps_compass_tensor = self._preprocess_gps_compass(gps_compass_data)

            time_info_prepare = time.time() - tic_prepare

            # --- 4. Build observation dict ---
            obs = {}
            obs["agent_0_overhead_front_rgb"] = rgb_tensor.squeeze(0)
            if depth_tensor is not None:
                obs["agent_0_overhead_front_depth"] = depth_tensor.squeeze(0)
            obs["agent_0_falcon_instruction"] = inst_tensor
            obs["agent_0_starting_point_gps_compass"] = gps_compass_tensor

            # 补全观测空间中其他传感器，跳过 pointgoal_with_gps_compass
            # (checkpoint 训练时无 tgt_embeding 层 → 不触发 GPS block)
            _skip_keys = {"pointgoal_with_gps_compass", "agent_0_pointgoal_with_gps_compass"}
            if self._env_spec is not None:
                obs_space = self._env_spec.observation_space
                for key in obs_space.spaces:
                    if key not in obs and key not in _skip_keys:
                        shape = obs_space.spaces[key].shape
                        dtype = obs_space.spaces[key].dtype
                        if dtype in (np.int64, np.int32):
                            obs[key] = torch.zeros(shape, dtype=torch.long)
                        else:
                            obs[key] = torch.zeros(shape, dtype=torch.float32)

            # --- 5. Run model inference ---
            tic_model = time.time()
            with torch.no_grad(), inference_mode():
                obs_batch = [obs]
                batch = batch_obs(obs_batch, device=self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                action_data = self.actor_critic.act(
                    batch,
                    self.real_rnn_states,
                    self.real_prev_actions,
                    self.real_not_done_masks,
                    deterministic=True,
                )

                # Update states
                self.real_rnn_states = action_data.rnn_hidden_states

                if action_data.should_inserts is None:
                    self.real_prev_actions.copy_(action_data.actions)
                else:
                    self.actor_critic.update_hidden_state(
                        self.real_rnn_states,
                        self.real_prev_actions,
                        action_data,
                    )

                self.real_not_done_masks = torch.ones(
                    self.num_envs, 1, dtype=torch.uint8, device=self.device
                )

                if hasattr(self._agent, "_agents") and self._agent._agents[
                    0
                ]._actor_critic.action_distribution_type == "categorical":
                    action_out = action_data.actions.cpu().item()
                elif is_continuous_action_space(self._env_spec.action_space):
                    action_out = action_data.actions.cpu().numpy()[0].tolist()
                else:
                    action_out = action_data.actions.cpu().item()

            time_info_model = time.time() - tic_model
            time_info_total = time.time() - tic_total

            # --- 6. 构建响应 ---
            response = {
                "status": "success",
                "action": action_out,
                "instruction_modified": instruction_modified,
                "pedestrian_detected": pedestrian_detected,
                "pedestrian_count": pedestrian_count,
                "time_info": {
                    "net": time_info_net,
                    "brain": time_info_brain,
                    "brain_inference_ms": brain_inference_ms,
                    "prepare": time_info_prepare,
                    "model": time_info_model,
                    "total": time_info_total,
                },
            }

            if instruction_modified:
                response["optimized_instruction"] = optimized_instruction

            return response

        except Exception as e:
            import traceback

            logger.error(f"Error in predict_action: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
            }, 500

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        kwargs.setdefault("weights_only", False)
        return torch.load(checkpoint_path, *args, **kwargs)
