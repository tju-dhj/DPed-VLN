import io
import os
import pickle
import time
from typing import Any, Dict

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

app = Flask("dped_trainer_server")


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


@baseline_registry.register_trainer(name="dped_trainer_server")
class DPedTrainerServer(BaseRLTrainer):
    """
    DPed_pro Flask HTTP Server Trainer.

    Embeds a Flask HTTP server into the DynamicVLNTrainer evaluation flow,
    enabling real-time action prediction for physical robot experiments.

    HTTP Endpoints:
        POST /reset_hiddens - Initialize/reset RNN hidden states
        POST /predict_action - Receive observation (image, instruction, etc.),
                                run model inference, return predicted action

    The server expects observations from a real robot (via STAMP-ROS),
    compatible with the 6-action DPed_pro action space:
        0: stop
        1: move_forward
        2: turn_left
        3: turn_right
        4: pause
        5: move_backward
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
        """Load checkpoint and start Flask server for real-time inference."""
        logger.info(f"DPedTrainerServer checkpoint_path: {checkpoint_path}")

        # Clone config and set split for evaluation
        cfg_hb = self.config.habitat_baselines
        config = self.config.copy()

        # Use eval.split from config, falling back to dataset.split
        eval_split = cfg_hb.eval.split if hasattr(cfg_hb.eval, "split") else self.config.habitat.dataset.split
        with read_write(config):
            config.habitat.dataset.split = eval_split

        # 2. Initialize env to get observation/action space
        pickle_loaded = False
        if os.path.exists("observation_space.pkl") and os.path.exists("orig_action_space.pkl"):
            with open("observation_space.pkl", "rb") as f:
                observation_space = pickle.load(f)
            with open("action_space.pkl", "rb") as f:
                action_space = pickle.load(f)
            with open("orig_action_space.pkl", "rb") as f:
                orig_action_space = pickle.load(f)
            pickle_loaded = True
            logger.info("Loaded observation/action/orig_action space from pickle files")
        else:
            self._init_envs(config, is_eval=True)
            observation_space = self._env_spec.observation_space
            action_space = self._env_spec.action_space
            orig_action_space = self._env_spec.orig_action_space
            self.envs.close()
            with open("observation_space.pkl", "wb") as f:
                pickle.dump(observation_space, f)
            with open("action_space.pkl", "wb") as f:
                pickle.dump(action_space, f)
            with open("orig_action_space.pkl", "wb") as f:
                pickle.dump(orig_action_space, f)
            logger.info("Saved observation/action/orig_action space to pickle files")

        logger.info(f"Observation space keys: {list(observation_space.spaces.keys())}")
        logger.info(f"Action space: {action_space}")
        logger.info(f"Orig action space keys: {list(orig_action_space.spaces.keys())}")

        # 3. Create agent and load checkpoint
        if pickle_loaded:
            # Build _env_spec from pickle data, skip env init for speed
            self._env_spec = EnvironmentSpec(
                observation_space=observation_space,
                action_space=action_space,
                orig_action_space=orig_action_space,
            )
            class _DummyEnvs:
                num_envs = 1
                def close(self):
                    pass
            self.envs = _DummyEnvs()
        else:
            self._init_envs(config, is_eval=True)
        self._agent = self._create_agent(None)

        if (
            self._agent.actor_critic.should_load_agent_state
            and cfg_hb.eval.should_load_ckpt
        ):
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            self._agent.load_state_dict(ckpt_dict)

        self._agent.eval()

        # 4. Extract actor_critic for inference
        # In MultiAgentAccessMgr, the actor_critic is a MultiPolicy wrapping agent_0's actor_critic
        self.actor_critic = self._agent.actor_critic

        # For discrete action spaces, get the action space info
        action_shape, discrete_actions = get_action_space_info(
            self.actor_critic.policy_action_space
        )
        logger.info(f"Action shape: {action_shape}, discrete: {discrete_actions}")

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
        port = getattr(self.config, "FLASK_SERVER_PORT", 32145)
        logger.info(f"Starting Flask server on 0.0.0.0:{port}")
        self.envs.close()
        app.run("0.0.0.0", port, threaded=True)

    def reset_hiddens(self):
        """Initialize RNN hidden states for a new episode."""
        with torch.no_grad():
            num_envs = self.num_envs

            # Get hidden state dimensions from agent_0's actor_critic
            # MultiPolicy stores the underlying policy in _active_policies
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

        return {"status": "success"}

    def _preprocess_rgb(self, rgb_bytes: bytes, sensor_key: str = "overhead_front_rgb") -> torch.Tensor:
        """
        Preprocess RGB image from robot sensor.

        只做 center_crop，返回 HWC uint8 格式。
        Normalize / ToTensor / CHW 转换由 apply_obs_transforms_batch 统一处理，
        与标准 Habitat eval 管线完全一致。

        Args:
            rgb_bytes: Raw JPEG image bytes
            sensor_key: Key for the sensor in observation space

        Returns:
            RGB tensor with shape (1, H, W, C) uint8
        """
        rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        rgb_pil = center_crop(rgb_pil, target_size=224)
        rgb = np.array(rgb_pil)  # (H, W, 3) uint8
        return torch.from_numpy(rgb).unsqueeze(0)  # (1, H, W, 3) uint8

    def _preprocess_depth(self, depth_bytes: bytes, sensor_key: str = "overhead_front_depth") -> torch.Tensor:
        """
        Preprocess depth — center_crop，返回 (H, W, 1) float32 meters。
        匹配 Habitat env 格式，让 obs_transforms + 模型内部 [..., 0] 正确工作。

        Returns:
            Depth tensor with shape (1, H, W, 1) float32 meters
        """
        depth_pil = Image.open(io.BytesIO(depth_bytes))
        depth_np = np.array(depth_pil)  # (H, W) uint16 (mm) or float32

        if depth_np.dtype == np.uint16:
            depth_np = depth_np.astype(np.float32) / 1000.0  # mm → m

        # center crop via PIL roundtrip
        depth_pil = Image.fromarray((np.clip(depth_np, 0, 10) / 10 * 255).astype(np.uint8))
        depth_pil = center_crop(depth_pil, target_size=224)
        depth_np = np.array(depth_pil).astype(np.float32) / 255.0 * 10.0

        # (H, W) → (H, W, 1) 匹配 Habitat env 格式
        depth_np = depth_np[..., np.newaxis]  # (224, 224, 1)
        return torch.from_numpy(depth_np).unsqueeze(0)  # (1, H, W, 1) float32

    def _preprocess_instruction(self, instruction_text: str, max_len: int = 512) -> torch.Tensor:
        """
        Preprocess instruction text (character-level encoding, matching falcon/vln_sensors.py).

        Args:
            instruction_text: Raw instruction string from robot
            max_len: Maximum instruction length

        Returns:
            Instruction tensor with shape (max_len,)
        """
        obs = torch.zeros(max_len, dtype=torch.long)
        for i, char in enumerate(instruction_text[:max_len]):
            obs[i] = ord(char) % 1000  # Map to vocabulary range [0, 999]
        return obs

    def _preprocess_gps_compass(self, gps_compass_data: Dict[str, float]) -> torch.Tensor:
        """
        StartingPointGPSCompassSensor format: [goal_x, goal_y, heading_sin, heading_cos].

        机器人端不传 goal_x/goal_y/compass 时默认全 0。
        """
        import math

        goal_x = gps_compass_data.get("goal_x", 0.0)
        goal_y = gps_compass_data.get("goal_y", 0.0)
        compass = gps_compass_data.get("compass", 0.0)

        return torch.tensor(
            [goal_x, goal_y, math.sin(compass), math.cos(compass)],
            dtype=torch.float32,
        )

    def predict_action(self):
        """
        Receive observation from robot, run model inference, return predicted action.

        Expected POST form fields:
            - ep_id: episode identifier (string)
            - rgb: RGB image file (JPEG bytes)
            - depth: Depth image file (PNG bytes, optional if not needed)
            - inst: Instruction text (string)
            - gps_x: robot GPS x coordinate (float, optional)
            - gps_y: robot GPS y coordinate (float, optional)
            - compass: robot compass heading in radians (float, optional)
            - goal_x: goal x relative to robot (float, optional)
            - goal_y: goal y relative to robot (float, optional)

        Returns:
            JSON dict with keys:
                - status: "success" or "error"
                - action: predicted action index (int, 0-5)
                - time_info: timing breakdown (optional)
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

            # --- 2. Preprocess observations ---
            tic_prepare = time.time()
            rgb_tensor = self._preprocess_rgb(rgb_bytes)

            # DEBUG: Print RGB image stats to verify image varies between steps
            if not hasattr(self, "_debug_step_count"):
                self._debug_step_count = 0
            self._debug_step_count += 1
            if self._debug_step_count <= 15:
                rgb_np = rgb_tensor.squeeze(0).numpy()  # (H, W, 3) uint8
                logger.info(
                    f"[SERVER DEBUG step={self._debug_step_count}] RGB stats: "
                    f"shape={rgb_np.shape} dtype={rgb_np.dtype} "
                    f"R(min={rgb_np[...,0].min():.0f} max={rgb_np[...,0].max():.0f} mean={rgb_np[...,0].mean():.1f}) "
                    f"G(min={rgb_np[...,1].min():.0f} max={rgb_np[...,1].max():.0f} mean={rgb_np[...,1].mean():.1f}) "
                    f"B(min={rgb_np[...,2].min():.0f} max={rgb_np[...,2].max():.0f} mean={rgb_np[...,2].mean():.1f})"
                )
                # Print image hash-like fingerprint (sum of pixels) to quickly compare between steps
                rgb_fingerprint = rgb_np.sum(axis=(0,1))
                logger.info(
                    f"[SERVER DEBUG step={self._debug_step_count}] RGB fingerprint (R,G,B sum): "
                    f"R={rgb_fingerprint[0]:.0f} G={rgb_fingerprint[1]:.0f} B={rgb_fingerprint[2]:.0f}"
                )

            depth_tensor = None
            if depth_bytes is not None:
                depth_tensor = self._preprocess_depth(depth_bytes)
                if self._debug_step_count <= 15:
                    depth_np = depth_tensor.squeeze(0).numpy()
                    logger.info(
                        f"[SERVER DEBUG step={self._debug_step_count}] Depth stats: "
                        f"shape={depth_np.shape} dtype={depth_np.dtype} "
                        f"min={depth_np.min():.4f} max={depth_np.max():.4f} mean={depth_np.mean():.4f}"
                    )

            inst_tensor = self._preprocess_instruction(inst_text)
            if self._debug_step_count <= 15:
                logger.info(
                    f"[SERVER DEBUG step={self._debug_step_count}] Instruction: [{inst_text}] "
                    f"(len={len(inst_text)}, tensor_shape={inst_tensor.shape})"
                )

            # GPS/compass from form fields
            gps_compass_data = {
                "goal_x": float(request.form.get("goal_x", 0.0)),
                "goal_y": float(request.form.get("goal_y", 0.0)),
                "compass": float(request.form.get("compass", 0.0)),
            }
            gps_compass_tensor = self._preprocess_gps_compass(gps_compass_data)

            time_info_prepare = time.time() - tic_prepare

            # --- 3. Build observation dict with agent_0 prefix (multi-agent format) ---
            tic_preprocess = time.time()

            obs = {}
            obs["agent_0_overhead_front_rgb"] = rgb_tensor.squeeze(0)  # (H, W, 3) uint8
            if depth_tensor is not None:
                obs["agent_0_overhead_front_depth"] = depth_tensor.squeeze(0)
            obs["agent_0_falcon_instruction"] = inst_tensor
            # 使用 pointgoal_with_gps_compass (IntegratedPointGoalGPSAndCompassSensor.cls_uuid)
            # 而不是 starting_point_gps_compass，因为模型 forward pass 检查前者
            obs["pointgoal_with_gps_compass"] = gps_compass_tensor[:2]  # [goal_x, goal_y] polar 2D

            # --- 4. Run model inference ---
            tic_model = time.time()
            with torch.no_grad(), inference_mode():
                # Wrap in list for batch_obs
                obs_batch = [obs]
                batch = batch_obs(obs_batch, device=self.device)
                batch = apply_obs_transforms_batch(batch, self.obs_transforms)

                # GPS/compass 数据已在上面格式化为 pointgoal_with_gps_compass (shape=2)
                # obs_transforms 不应对其进行额外重命名

                if not hasattr(self, '_batch_debug_printed'):
                    logger.info("[DEBUG] Batch keys AFTER obs_transforms:")
                    for k in sorted(batch.keys()):
                        v = batch[k]
                        if hasattr(v, 'shape'):
                            logger.info(f"  {k}: shape={v.shape}")
                    self._batch_debug_printed = True

                # Get action from MultiPolicy
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

                # Extract action value
                if hasattr(self._agent, "_agents") and self._agent._agents[
                    0
                ]._actor_critic.action_distribution_type == "categorical":
                    action_out = action_data.actions.cpu().item()
                elif is_continuous_action_space(self._env_spec.action_space):
                    action_out = action_data.actions.cpu().numpy()[0].tolist()
                else:
                    action_out = action_data.actions.cpu().item()

                # DEBUG: Print action prediction
                if hasattr(self, "_debug_step_count") and self._debug_step_count <= 15:
                    logits_info = ""
                    if hasattr(action_data, "action_log_probs") and action_data.action_log_probs is not None:
                        log_probs = action_data.action_log_probs.cpu().numpy().flatten()
                        logits_info = f" log_probs={log_probs.tolist()}"
                    rnn_info = ""
                    if hasattr(action_data, "rnn_hidden_states") and action_data.rnn_hidden_states is not None:
                        rnn = action_data.rnn_hidden_states
                        rnn_info = f" rnn_hidden(mean={rnn.mean().item():.6f} std={rnn.std().item():.6f})"
                    logger.info(
                        f"[SERVER DEBUG step={self._debug_step_count}] Action={action_out}{logits_info}{rnn_info}"
                    )

            time_info_model = time.time() - tic_model
            time_info_preprocess = time.time() - tic_preprocess
            time_info_total = time.time() - tic_total

            return {
                "status": "success",
                "action": action_out,
                "time_info": {
                    "net": time_info_net,
                    "prepare": time_info_prepare,
                    "preprocess": time_info_preprocess,
                    "model": time_info_model,
                    "total": time_info_total,
                },
            }

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
