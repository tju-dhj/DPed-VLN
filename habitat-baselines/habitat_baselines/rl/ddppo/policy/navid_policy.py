#!/usr/bin/env python3
"""
NaVid Policy for DPed-VLN Falcon Framework

仿照 NaVILAPolicy 实现，将 NaVid (Vicuna-7B + EVA-CLIP ViT-G) 模型接入 Falcon 框架。
"""

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gym import spaces

from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.rl.ddppo.policy.navid.action_parser import (
    NaVidActionParser,
    ACTION_ID_TO_NAME,
    ACTION_NAME_TO_ID,
)
from habitat_baselines.rl.ddppo.policy.navid.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
    NAVIGATION_SPECIAL_TOKEN,
    IAMGE_SEPARATOR,
    NAVIGATION_IDENTIFIER,
    IGNORE_INDEX,
)
from habitat_baselines.rl.ddppo.policy.navid.conversation import (
    conv_templates,
    SeparatorStyle,
)
from habitat_baselines.rl.ddppo.policy.navid.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from habitat_baselines.common.baseline_registry import baseline_registry


# ============================================================
# Action helpers
# ============================================================

ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move forward",
    2: "turn left",
    3: "turn right",
}


def sample_and_pad_images(image_list: List, num_frames: int) -> List:
    """采样并填充图像列表到固定帧数"""
    if len(image_list) == 0:
        return image_list

    if len(image_list) >= num_frames:
        indices = np.linspace(0, len(image_list) - 1, num_frames, dtype=int)
        return [image_list[i] for i in indices]
    else:
        last_img = image_list[-1]
        padded = list(image_list)
        while len(padded) < num_frames:
            padded.append(last_img)
        return padded


def extract_navid_instruction(observations, instruction_sensor_uuid=None, episode_instruction=None):
    """从 observations 中提取指令文本"""
    if instruction_sensor_uuid is not None:
        sensor_value = observations.get(instruction_sensor_uuid)
        if sensor_value is not None:
            if isinstance(sensor_value, torch.Tensor):
                sensor_value = sensor_value.item() if sensor_value.numel() == 1 else str(sensor_value)
            if isinstance(sensor_value, str) and len(sensor_value.strip()) > 0:
                return sensor_value.strip()

    for key in ["agent_0_falcon_instruction", "falcon_instruction", "instruction"]:
        value = observations.get(key)
        if value is not None:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    # Multi-element tensor = encoded text, not a readable string.
                    # Skip so we fall through to episode_instruction.
                    continue
            if isinstance(value, str) and len(value.strip()) > 0:
                return value.strip()

    if episode_instruction is not None and len(str(episode_instruction).strip()) > 0:
        return str(episode_instruction).strip()

    return "Navigate to the target location."


# ============================================================
# NaVidNet
# ============================================================

class NaVidNet(Net):
    """NaVid 网络模型"""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        model_path: Optional[str],
        num_video_frames: int = 4,
        forward_step: int = 25,
        turn_step: int = 15,
        instruction_sensor_uuid: str = "instruction",
        rgb_sensor_keys: Optional[List[str]] = None,
        action_sequence_mode: bool = False,
        action_sequence_length: int = 4,
    ):
        super().__init__()

        self.num_video_frames = num_video_frames
        self.instruction_sensor_uuid = instruction_sensor_uuid
        self.action_sequence_mode = action_sequence_mode
        self.action_sequence_length = action_sequence_length

        # RGB sensor keys
        self.available_rgb_keys = [
            key for key in observation_space.spaces.keys()
            if isinstance(key, str) and "rgb" in key.lower()
        ]
        self._preferred_rgb_keys = rgb_sensor_keys or self.available_rgb_keys

        self._step_counter = 0
        self._last_rgb_key = None

        # Action parser
        self.action_parser = NaVidActionParser(
            forward_step=forward_step,
            turn_step=turn_step,
        )

        # Load NaVid model
        if model_path is None:
            raise ValueError("NaVid model path is not provided")

        # Resolve model path
        if not os.path.isabs(model_path):
            resolved = None
            current = Path(__file__).resolve().parent
            for _ in range(10):
                if (current / "pretrained_model").exists():
                    candidate = current / model_path
                    if candidate.exists():
                        resolved = str(candidate.resolve())
                        break
                candidate = current / model_path
                if candidate.exists():
                    resolved = str(candidate.resolve())
                    break
                if current == current.parent:
                    break
                current = current.parent
            if resolved is not None:
                model_path = resolved

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NaVid model path does not exist: {model_path}")

        print(f"[NaVidNet] Loading model from: {model_path}")

        # Import and load
        from habitat_baselines.rl.ddppo.policy.navid.model.builder import load_pretrained_model

        model_name = os.path.basename(os.path.normpath(model_path))
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name
        )

        print(f"[NaVidNet] Model loaded: {type(self.model).__name__}")
        print(f"[NaVidNet] Vision tower: {type(self.model.get_vision_tower()).__name__}")
        print(f"[NaVidNet] Context length: {self.context_len}")

        # State
        self.past_rgbs: List = []
        self.action_queue: List[Dict[str, Any]] = []
        self.history_rgb_tensor = None

        self._hidden_size = 4  # 4 actions (one-hot)

        # When True, forward() uses action_head instead of text generation (for SFT eval)
        self.use_action_head = False

        # Training mode: learnable action head projecting VLM hidden states → 4 action logits
        self.action_head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4),
        )

        # Enable gradient checkpointing for VLM to save memory on L40 (44GB)
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False  # required for gradient_checkpointing

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 0  # No RNN

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    def _find_rgb(self, observations: Dict) -> Optional[Any]:
        """查找 RGB 图像"""
        for key in self._preferred_rgb_keys:
            if key in observations:
                rgb = observations[key]
                if isinstance(rgb, torch.Tensor):
                    if rgb.numel() == 0:
                        continue
                    rgb = rgb.cpu().numpy()
                if isinstance(rgb, np.ndarray):
                    if rgb.size == 0:
                        continue
                    if rgb.ndim == 3 and rgb.shape[0] == 3:
                        rgb = rgb.transpose(1, 2, 0)
                    rgb = rgb.astype(np.uint8)
                return rgb
        return None

    def _collect_rgb_from_obs(self, observations: Dict) -> Optional[Any]:
        """从观察中提取 RGB"""
        rgb = self._find_rgb(observations)
        if rgb is not None:
            return rgb

        # 尝试通用的 rgb key
        for key in observations:
            if "rgb" in str(key).lower():
                val = observations[key]
                if isinstance(val, torch.Tensor) and val.numel() > 0:
                    val = val.cpu().numpy()
                if isinstance(val, np.ndarray) and val.size > 0:
                    if val.ndim == 3 and val.shape[0] == 3:
                        val = val.transpose(1, 2, 0)
                    return val.astype(np.uint8)
        return None

    def _generate_navid_response(self, instruction: str) -> str:
        """调用 NaVid 模型生成导航动作文本"""
        from PIL import Image

        # Build video frames
        past_and_current = self.past_rgbs + [self.past_rgbs[-1]] if self.past_rgbs else []
        if not past_and_current:
            return "stop"

        sampled = sample_and_pad_images(past_and_current, max(1, self.num_video_frames))

        # Process images
        video_list = []
        for img in sampled:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            video_list.append(img)

        # Process through image processor
        if len(video_list) == 1:
            images_tensor = self.image_processor.preprocess(video_list[0], return_tensors='pt')['pixel_values']
        else:
            batch_np = np.stack([np.array(img) for img in video_list])
            images_tensor = self.image_processor.preprocess(batch_np, return_tensors='pt')['pixel_values']

        images_tensor = images_tensor.half().cuda()

        # Build prompt
        interleaved = "<image>\n" * (len(sampled) - 1) if len(sampled) > 1 else ""
        nav_prompt = (
            f"Imagine you are a robot programmed for navigation tasks. "
            f"You have been given a video of historical observations {interleaved}"
            f"and an image of the current observation <image>. "
            f'Your assigned task is: "{instruction}". '
            f"Analyze this series of images to decide your next move, which could involve "
            f"turning left or right by a specific degree or moving forward a certain distance."
        )

        # Conversation
        conv_mode = "vicuna_v1"
        conv = conv_templates[conv_mode].copy()
        qs = DEFAULT_IMAGE_TOKEN + '\n' + nav_prompt.replace('<image>', '')
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize with special tokens
        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

        # Insert NaVid special tokens
        image_start_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_sep = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_token)
            new_list.append(image_sep)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_token)
            new_list.append(image_start_token)
            new_list.append(image_end_token)
            new_list.append(navigation_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        # Stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        # Build image list
        imgs = [images_tensor]

        # Generate
        question_text = nav_prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        with torch.inference_mode():
            self.model.update_prompt([[question_text]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        return outputs

    def _forward_train(self, observations, rnn_hidden_states, prev_actions, masks):
        """
        Differentiable training forward pass (optimized batch version).

        Batches all samples' RGB images + instructions through the VLM
        in a single forward pass where possible, then pools hidden states
        through a learnable action head.
        """
        try:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        except Exception:
            pass
        from PIL import Image

        batch_size = 1
        for key, val in observations.items():
            if isinstance(val, torch.Tensor) and val.ndim >= 1:
                batch_size = val.shape[0]
                break

        device = next(self.model.parameters()).device

        # Collect all RGB images and instructions first
        rgbs, instructions, skip_mask = [], [], []
        for i in range(batch_size):
            obs_i = {}
            for key, val in observations.items():
                if isinstance(val, torch.Tensor):
                    obs_i[key] = val[i] if val.shape[0] > i else val[0]
                elif isinstance(val, list):
                    obs_i[key] = val[i] if len(val) > i else val[0]
                else:
                    obs_i[key] = val

            rgb = self._collect_rgb_from_obs(obs_i)
            if rgb is None:
                skip_mask.append(True)
                rgbs.append(None)
                instructions.append("")
                continue
            skip_mask.append(False)
            rgbs.append(rgb)
            instructions.append(extract_navid_instruction(obs_i, episode_instruction=None))

        if all(skip_mask):
            return torch.zeros(batch_size, self._hidden_size, device=device), rnn_hidden_states, {}

        # Process and batch images
        image_tensors = []
        input_ids_list = []
        prompts_list = []
        for i in range(batch_size):
            if skip_mask[i]:
                continue
            rgb = rgbs[i]
            if isinstance(rgb, np.ndarray):
                rgb = Image.fromarray(rgb)
            if rgb.mode != "RGB":
                rgb = rgb.convert("RGB")
            img_tensor = self.image_processor.preprocess(rgb, return_tensors='pt')['pixel_values']
            img_tensor = img_tensor.half().to(device)
            image_tensors.append(img_tensor)

            instruction = instructions[i]
            qs = DEFAULT_IMAGE_TOKEN + '\n' + instruction.replace('<image>', '')
            conv_mode = "vicuna_v1"
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            from habitat_baselines.rl.ddppo.policy.navid.mm_utils import tokenizer_image_token
            ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            ids = ids.unsqueeze(0).to(device)
            input_ids_list.append(ids)

            question_text = instruction.replace('<image>', '').replace('\n', ' ').strip()
            prompts_list.append([question_text])

        features_t = torch.zeros(batch_size, self._hidden_size, device=device)
        valid_idx = 0
        for i in range(batch_size):
            if skip_mask[i]:
                continue
            outputs = self.model(
                input_ids=input_ids_list[valid_idx],
                images=[image_tensors[valid_idx]],
                prompts=[prompts_list[valid_idx]],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1).to(dtype=torch.float32)
            features_t[i] = self.action_head(pooled)
            valid_idx += 1

        return features_t, rnn_hidden_states, {}

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        """
        Forward pass with training/inference branching.
        - Training: differentiable VLM forward → action_head (gradients flow to LoRA)
        - Inference (use_action_head=True): VLM forward → action_head (no text generation)
        - Inference (use_action_head=False): generate() text → parse action → one-hot features
        """
        use_action_head = getattr(self, 'use_action_head', False)
        if self.training or use_action_head:
            return self._forward_train(observations, rnn_hidden_states, prev_actions, masks)

        # === Inference path (original generate-based logic) ===
        batch_size = observations.get("rgb", observations.get(list(observations.keys())[0])).shape[0] \
            if isinstance(observations, dict) and len(observations) > 0 else 1

        features = torch.zeros(batch_size, self._hidden_size, device="cuda")

        for i in range(batch_size):
            # Check action queue
            if len(self.action_queue) > 0:
                queued = self.action_queue.pop(0)
                action = queued.get("action", 0) if isinstance(queued, dict) else int(queued)
            else:
                # 单步推理
                obs_i = {}
                for key, val in observations.items():
                    if isinstance(val, torch.Tensor):
                        obs_i[key] = val[i] if val.shape[0] > i else val[0]
                    elif isinstance(val, list):
                        obs_i[key] = val[i] if len(val) > i else val[0]
                    else:
                        obs_i[key] = val

                rgb = self._collect_rgb_from_obs(obs_i)
                if rgb is not None:
                    self.past_rgbs.append(rgb)
                    if len(self.past_rgbs) > 50:
                        self.past_rgbs = self.past_rgbs[-50:]

                instruction = extract_navid_instruction(obs_i, episode_instruction=None)
                output_text = self._generate_navid_response(instruction)
                action, num_repeats = self.action_parser.parse_action(output_text)

                # Queue repeats
                if num_repeats > 1:
                    for _ in range(num_repeats - 1):
                        self.action_queue.append({"action": action})

            features[i, action] = 1.0

        return features, rnn_hidden_states, {}

    def generate_action_sequence(self, rgb, depth, instruction, env_idx=0, run_model=True):
        """生成动作序列 (为 evaluator 提供统一接口)"""
        if not run_model:
            # 仅更新历史
            if rgb is not None:
                rgb_np = rgb
                if isinstance(rgb, torch.Tensor):
                    rgb_np = rgb.cpu().numpy()
                if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
                    rgb_np = rgb_np.transpose(1, 2, 0)
                self.past_rgbs.append(rgb_np.astype(np.uint8))
                if len(self.past_rgbs) > 50:
                    self.past_rgbs = self.past_rgbs[-50:]
            return []

        # Run model
        if rgb is not None:
            rgb_np = rgb
            if isinstance(rgb, torch.Tensor):
                rgb_np = rgb.cpu().numpy()
            if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
                rgb_np = rgb_np.transpose(1, 2, 0)
            self.past_rgbs.append(rgb_np.astype(np.uint8))
            if len(self.past_rgbs) > 50:
                self.past_rgbs = self.past_rgbs[-50:]

        if not self.past_rgbs:
            return []

        output_text = self._generate_navid_response(instruction)
        action, num_repeats = self.action_parser.parse_action(output_text)

        # 构建动作序列
        action_seq = [action]
        for _ in range(num_repeats - 1):
            action_seq.append(action)

        return action_seq

    def reset_history(self):
        """重置历史状态"""
        self.past_rgbs = []
        self.action_queue = []
        self.history_rgb_tensor = None


# ============================================================
# NaVidPolicy
# ============================================================

@baseline_registry.register_policy
class NaVidPolicy(NetPolicy):
    """NaVid Falcon Policy"""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        model_path: str = None,
        num_video_frames: int = 4,
        forward_step: int = 25,
        turn_step: int = 15,
        instruction_sensor_uuid: str = "instruction",
        rgb_sensor_keys: Optional[List[str]] = None,
        action_sequence_mode: bool = False,
        action_sequence_length: int = 4,
        policy_config=None,
    ):
        # Build net
        net = NaVidNet(
            observation_space=observation_space,
            action_space=action_space,
            model_path=model_path,
            num_video_frames=num_video_frames,
            forward_step=forward_step,
            turn_step=turn_step,
            instruction_sensor_uuid=instruction_sensor_uuid,
            rgb_sensor_keys=rgb_sensor_keys,
            action_sequence_mode=action_sequence_mode,
            action_sequence_length=action_sequence_length,
        )

        super().__init__(
            net,
            action_space=action_space,
            policy_config=policy_config,
        )

        self._policy_config = policy_config

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        """从配置创建 policy"""
        if hasattr(config, 'habitat_baselines'):
            # eval 模式
            policy_cfg = config.habitat_baselines.rl.policy.agent_0
        elif hasattr(config, 'il'):
            # 训练模式
            policy_cfg = config.habitat_baselines.il.policy.agent_0
        else:
            raise ValueError("Cannot find policy config")

        model_path = getattr(policy_cfg, 'model_path', kwargs.get('model_path'))
        num_video_frames = getattr(policy_cfg, 'num_video_frames', kwargs.get('num_video_frames', 4))
        forward_step = getattr(policy_cfg, 'forward_step', kwargs.get('forward_step', 25))
        turn_step = getattr(policy_cfg, 'turn_step', kwargs.get('turn_step', 15))
        instruction_sensor_uuid = getattr(policy_cfg, 'instruction_sensor_uuid',
                                          kwargs.get('instruction_sensor_uuid', 'instruction'))

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_path=model_path,
            num_video_frames=num_video_frames,
            forward_step=forward_step,
            turn_step=turn_step,
            instruction_sensor_uuid=instruction_sensor_uuid,
            policy_config=policy_cfg,
        )


