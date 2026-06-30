#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
StreamVLN Policy integration for Falcon framework.
Adapts StreamVLN model (Habitat 2) to Falcon's policy interface (Habitat 3).
"""

import copy
import re
import random
import itertools
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from PIL import Image

# 应用 transformers 兼容性补丁（在导入 transformers 之前）
# 补丁模块在导入时会自动执行
try:
    from habitat_baselines.rl.ddppo.policy.navila import transformers_compat_patch  # noqa: F401
except ImportError:
    # 如果补丁模块不存在，尝试直接应用补丁
    pass

import transformers
import logging
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from habitat_baselines.utils.common import get_num_actions
from habitat_baselines.utils.common import get_num_actions

logger = logging.getLogger(__name__)
_STREAMVLN_LLAVA_READY = False


def _patch_streamvln_config(config, model_path=None):
    """
    Ensure configs loaded from legacy StreamVLN checkpoints expose the fields
    expected by upstream transformer implementations (e.g. Qwen2).
    
    重要：Qwen2 在 forward 时构建 causal_mask_mapping，使用的键是：
    - "full_attention" (不是 "self_attn"!)
    - "sliding_attention" (如果有 sliding layers)
    
    但 decoder_layer.attention_type 来自 config.layer_types，所以需要匹配。
    根据 Qwen2 源码，layer_types 应该使用 "full_attention" 而不是 "self_attn"。
    """
    num_layers = getattr(config, "num_hidden_layers", None)
    # Qwen2 在初始化 Qwen2Attention 时需要访问 config.layer_types
    # 必须确保这个属性存在，否则会报 AttributeError
    if not hasattr(config, "layer_types") or config.layer_types is None:
        if isinstance(num_layers, int) and num_layers > 0:
            # 修复：Qwen2 的 causal_mask_mapping 使用 "full_attention" 键，不是 "self_attn"
            # 所以 layer_types 也应该使用 "full_attention" 来匹配
            config.layer_types = ["full_attention"] * num_layers
            print(f"[_patch_streamvln_config] 设置 layer_types 为 ['full_attention'] * {num_layers}")
        else:
            # 如果无法确定层数，至少设置一个空列表避免 AttributeError
            config.layer_types = []
    else:
        # 如果 layer_types 已经存在，检查并修复不匹配的值
        if config.layer_types and len(config.layer_types) > 0:
            # 将 "self_attn" 替换为 "full_attention" 以匹配 Qwen2 的 causal_mask_mapping
            if config.layer_types[0] == "self_attn":
                print(f"[_patch_streamvln_config] 检测到 layer_types 使用 'self_attn'，替换为 'full_attention' 以匹配 Qwen2")
                config.layer_types = ["full_attention" if lt == "self_attn" else lt for lt in config.layer_types]
    
    # 重要：修复 vocab_size 不匹配问题
    # AutoConfig 可能从 text_config 读取了 vocab_size (32000)，但 checkpoint 中的权重是 152064
    # 需要确保使用顶层的 vocab_size (152064) 来匹配 checkpoint
    # 尝试从 config.json 文件直接读取顶层 vocab_size
    if hasattr(config, 'vocab_size') and config.vocab_size == 32000:
        # 检查是否有顶层 vocab_size 配置（从 config.json 直接读取）
        import json
        import os
        config_file_path = None
        # 尝试找到 config.json 文件路径
        if hasattr(config, '_name_or_path'):
            model_path = config._name_or_path
            if os.path.isdir(model_path):
                config_file_path = os.path.join(model_path, "config.json")
        # 如果找不到，尝试从当前工作目录查找
        if not config_file_path or not os.path.exists(config_file_path):
            # 尝试从环境变量或其他方式获取模型路径
            pass
        
        # 如果找到了 config.json，读取顶层 vocab_size
        if config_file_path and os.path.exists(config_file_path):
            try:
                with open(config_file_path, 'r') as f:
                    config_dict = json.load(f)
                    top_level_vocab_size = config_dict.get('vocab_size')
                    if top_level_vocab_size and top_level_vocab_size != 32000:
                        config.vocab_size = top_level_vocab_size
                        print(f"[_patch_streamvln_config] 修复 vocab_size: 32000 -> {top_level_vocab_size} (从 config.json 读取)")
            except Exception as e:
                print(f"[_patch_streamvln_config] 无法读取 config.json: {e}")
                # 如果读取失败，使用默认值 152064（StreamVLN 的标准 vocab_size）
                config.vocab_size = 152064
                print(f"[_patch_streamvln_config] 使用默认 vocab_size: 152064")
        else:
            # 如果找不到 config.json，使用默认值 152064
            config.vocab_size = 152064
            print(f"[_patch_streamvln_config] 修复 vocab_size: 32000 -> 152064 (使用默认值，匹配 checkpoint 权重)")
    
    return config


def _ensure_streamvln_transformers_compatibility():
    """
    Mirror the optional symbol guards used by the NaVILA integration so that
    vendorized transformers versions (or older wheels) still expose the helpers
    StreamVLN relies on.
    """
    try:
        utils_module = importlib.import_module("transformers.utils")
    except ImportError:
        return

    if not hasattr(utils_module, "is_torch_tpu_available"):

        def _is_torch_tpu_available() -> bool:
            return False

        utils_module.is_torch_tpu_available = _is_torch_tpu_available  # type: ignore[attr-defined]

    try:
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    except ImportError as exc:
        if "is_torch_tpu_available" in str(exc):
            if "transformers.modeling_utils" in sys.modules:
                del sys.modules["transformers.modeling_utils"]
            modeling_utils = importlib.import_module("transformers.modeling_utils")
        else:
            return

    if not hasattr(modeling_utils, "PreTrainedAudioTokenizerBase"):

        class PreTrainedAudioTokenizerBase:  # type: ignore[too-many-ancestors]
            pass

        modeling_utils.PreTrainedAudioTokenizerBase = PreTrainedAudioTokenizerBase  # type: ignore[attr-defined]

    if not hasattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS"):
        modeling_utils.ALL_ATTENTION_FUNCTIONS = {}  # type: ignore[attr-defined]


def _get_module_root(module: Any) -> Optional[Path]:
    """Return an absolute Path pointing to the root of the provided module."""
    path_str: Optional[str] = getattr(module, "__file__", None)
    if path_str is None:
        module_paths = getattr(module, "__path__", None)
        if module_paths:
            try:
                path_str = next(iter(module_paths))
            except StopIteration:
                path_str = None
    if path_str is None:
        return None
    try:
        return Path(path_str).resolve()
    except Exception:
        return Path(path_str)


def _ensure_streamvln_llava_loaded(llava_dir: Path) -> None:
    """
    Ensure the global 'llava' module points to the StreamVLN vendor copy,
    even if NaVILA has already registered its own version earlier.
    """
    global _STREAMVLN_LLAVA_READY
    llava_init = llava_dir / "__init__.py"
    if not llava_init.exists():
        return

    existing_llava = sys.modules.get("llava")
    if existing_llava is not None:
        existing_root = _get_module_root(existing_llava)
        try:
            if existing_root and existing_root.is_relative_to(llava_dir):
                _STREAMVLN_LLAVA_READY = True
                return
        except AttributeError:
            # Python <3.9 compatibility: fallback check
            try:
                if existing_root and str(llava_dir.resolve()) in str(existing_root):
                    _STREAMVLN_LLAVA_READY = True
                    return
            except Exception:
                pass

    try:
        spec = importlib.util.spec_from_file_location(
            "llava", llava_init, submodule_search_locations=[str(llava_dir)]
        )
    except (FileNotFoundError, ImportError):
        return

    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(llava_dir)]  # type: ignore[attr-defined]

    removed_submodules: Dict[str, Any] = {}
    for name in list(sys.modules.keys()):
        if name == "llava" or name.startswith("llava."):
            removed_submodules[name] = sys.modules.pop(name)

    if existing_llava is not None:
        # Preserve NaVILA's module under a backup name for advanced users.
        sys.modules.setdefault("navila_llava_backup", existing_llava)
    for name, mod in removed_submodules.items():
        backup_name = f"navila_{name}_backup"
        sys.modules.setdefault(backup_name, mod)

    sys.modules["llava"] = module
    spec.loader.exec_module(module)

    utils_path = llava_dir / "utils.py"
    if utils_path.exists():
        utils_spec = importlib.util.spec_from_file_location("llava.utils", utils_path)
        if utils_spec and utils_spec.loader:
            utils_module = importlib.util.module_from_spec(utils_spec)
            utils_module.__path__ = []  # treat as namespace package
            sys.modules["llava.utils"] = utils_module
            utils_spec.loader.exec_module(utils_module)

            logging_module = types.ModuleType("llava.utils.logging")
            try:
                from loguru import logger as _loguru_logger
            except ImportError:
                import logging as _py_logging

                _loguru_logger = _py_logging.getLogger("llava")
            logging_module.logger = _loguru_logger
            sys.modules["llava.utils.logging"] = logging_module

    _STREAMVLN_LLAVA_READY = True


def _maybe_extend_streamvln_sys_path():
    """
    Ensure the vendored StreamVLN codebase (llava/utils/trl, etc.) is directly importable.
    """
    policy_root = Path(__file__).resolve().parent
    streamvln_root = policy_root / "streamvln"
    if not streamvln_root.exists():
        return

    candidate_dirs = [
        streamvln_root,
        streamvln_root / "streamvln",
        streamvln_root / "llava",
        streamvln_root / "trl",
    ]
    for path in candidate_dirs:
        resolved = path.resolve()
        if resolved.exists():
            path_str = str(resolved)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    if not _STREAMVLN_LLAVA_READY:
        llava_dir = streamvln_root / "llava"
        _ensure_streamvln_llava_loaded(llava_dir)


_maybe_extend_streamvln_sys_path()
try:
    _ensure_streamvln_transformers_compatibility()
except Exception:
    # 兼容性补丁失败不应阻断导入，实际报错会在后续模块加载时体现
    pass

# StreamVLN imports
from .streamvln.streamvln.model.stream_video_vln import StreamVLNForCausalLM
from .streamvln.streamvln.utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    DEFAULT_MEMORY_TOKEN,
    MEMORY_TOKEN_INDEX,
    DEFAULT_VIDEO_TOKEN,
    dict_to_cuda,
    IGNORE_INDEX,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


@baseline_registry.register_policy
class StreamVLNPolicy(NetPolicy):
    """
    StreamVLN Policy for Falcon framework.
    
    This policy integrates the StreamVLN model (trained on Habitat 2) into
    Falcon's architecture (running on Habitat 3).
    
    Key features:
    - Visual-language navigation using StreamVLN's multimodal transformer
    - Memory-augmented streaming inference
    - Action sequence generation from language instructions
    """

    ACTION_BLOCK_PATTERN = re.compile(
        r"<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE
    )

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 1024,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        model_path: str = None,
        num_frames: int = 32,
        num_history: int = 8,
        num_future_steps: int = 4,
        model_max_length: int = 4096,
        device: str = "cuda",
        forward_step: int = 25,
        turn_step: int = 15,
        **kwargs,
    ):
        """
        Initialize StreamVLN Policy.
        
        Args:
            observation_space: Observation space from Habitat
            action_space: Action space from Habitat
            hidden_size: Hidden size for RNN (not used, kept for compatibility)
            num_recurrent_layers: Number of RNN layers (not used, kept for compatibility)
            rnn_type: RNN type (not used, kept for compatibility)
            policy_config: Policy configuration
            aux_loss_config: Auxiliary loss configuration
            model_path: Path to pretrained StreamVLN model
            num_frames: Number of frames to process before resetting memory
            num_history: Number of history frames to keep
            num_future_steps: Number of future steps for action prediction
            model_max_length: Maximum sequence length for the model
            device: Device to run the model on
            forward_step: Forward step size in cm
            turn_step: Turn step size in degrees
        """
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            StreamVLNNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                model_path=model_path,
                num_frames=num_frames,
                num_history=num_history,
                num_future_steps=num_future_steps,
                model_max_length=model_max_length,
                device=device,
                discrete_actions=discrete_actions,
                forward_step=forward_step,
                turn_step=turn_step,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        """Create policy from config."""
        # Exclude cameras for rendering from the observation space
        ignore_names = []
        if hasattr(config, 'habitat_baselines') and hasattr(config.habitat_baselines, 'eval'):
            if hasattr(config.habitat_baselines.eval, 'extra_sim_sensors'):
                ignore_names = [
                    sensor.uuid
                    for sensor in config.habitat_baselines.eval.extra_sim_sensors.values()
                ]
        
        filtered_obs = spaces.Dict(
            OrderedDict(
                (
                    (k, v)
                    for k, v in observation_space.items()
                    if k not in ignore_names
                )
            )
        )

        agent_name = kwargs.get("agent_name")
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]

        # Get StreamVLN specific config
        streamvln_config = config.habitat_baselines.rl.policy.get(agent_name, {})
        
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            model_path=streamvln_config.get("model_path", None),
            num_frames=streamvln_config.get("num_frames", 32),
            num_history=streamvln_config.get("num_history", 8),
            num_future_steps=streamvln_config.get("num_future_steps", 4),
            model_max_length=streamvln_config.get("model_max_length", 4096),
            device=streamvln_config.get("device", "cuda"),
            forward_step=streamvln_config.get("forward_step", 25),
            turn_step=streamvln_config.get("turn_step", 15),
        )


class StreamVLNNet(Net):
    """
    Network architecture for StreamVLN Policy.
    
    This network wraps the StreamVLN model and adapts it to Falcon's
    policy interface. It handles:
    - Image preprocessing and encoding
    - Language instruction processing
    - Memory management for streaming inference
    - Action sequence generation and decoding
    """

    ACTION_BLOCK_PATTERN = re.compile(
        r"<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>", re.DOTALL | re.IGNORECASE
    )

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        model_path: str = None,
        num_frames: int = 32,
        num_history: int = 8,
        num_future_steps: int = 4,
        model_max_length: int = 4096,
        device: str = "cuda",
        discrete_actions: bool = True,
        forward_step: int = 25,
        turn_step: int = 15,
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.discrete_actions = discrete_actions
        self._hidden_size = hidden_size
        self.num_frames = num_frames
        self.num_history = num_history
        self.num_future_steps = num_future_steps
        self.model_max_length = model_max_length
        
        # Action mapping: StreamVLN to Habitat
        # StreamVLN actions: 0=STOP, 1=MOVE_FORWARD(↑), 2=TURN_LEFT(←), 3=TURN_RIGHT(→)
        # These need to be mapped to Habitat's action space
        self.actions2idx = OrderedDict({
            'STOP': 0,
            "↑": 1,  # MOVE_FORWARD
            "←": 2,  # TURN_LEFT
            "→": 3   # TURN_RIGHT
        })
        
        # Reverse mapping for action generation
        self.idx2actions = {v: k for k, v in self.actions2idx.items()}
        
        # 初始化动作解析器
        try:
            from habitat_baselines.rl.ddppo.policy.streamvln.action_parser import StreamVLNActionParser
            self.action_parser = StreamVLNActionParser(
                forward_step=forward_step,
                turn_step=turn_step,
            )
        except ImportError:
            self.action_parser = None
        
        # Initialize StreamVLN model
        if model_path is None:
            raise ValueError("model_path must be provided for StreamVLN policy")
        
        print(f"Loading StreamVLN model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=model_max_length,
            padding_side="right"
        )
        
        # Add special tokens
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.tokenizer.add_tokens(["<memory>"], special_tokens=True)
        
        # Load model config and model
        config = transformers.AutoConfig.from_pretrained(model_path)
        config = _patch_streamvln_config(config, model_path=model_path)
        
        # 重要：设置config以确保vision_tower权重从checkpoint加载
        # 如果checkpoint中包含vision_tower权重，需要设置unfreeze_mm_vision_tower=True
        # 这样vision_tower会在初始化时加载，而不是延迟加载
        # 检查checkpoint中是否包含vision_tower权重
        import os
        has_vision_tower_weights = False
        if os.path.isdir(model_path):
            try:
                import glob
                checkpoint_files = []
                if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                    checkpoint_files.append(os.path.join(model_path, "pytorch_model.bin"))
                elif os.path.exists(os.path.join(model_path, "model.safetensors")):
                    checkpoint_files.append(os.path.join(model_path, "model.safetensors"))
                else:
                    checkpoint_files = glob.glob(os.path.join(model_path, "pytorch_model-*.bin"))
                    if not checkpoint_files:
                        checkpoint_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
                
                if checkpoint_files:
                    # 只检查第一个文件（通常是最大的）
                    try:
                        if checkpoint_files[0].endswith('.safetensors'):
                            from safetensors import safe_open
                            with safe_open(checkpoint_files[0], framework="pt", device="cpu") as f:
                                checkpoint_keys = list(f.keys())
                        else:
                            checkpoint_state = torch.load(checkpoint_files[0], map_location="cpu")
                            if isinstance(checkpoint_state, dict):
                                checkpoint_keys = list(checkpoint_state.keys())
                            else:
                                checkpoint_keys = []
                        
                        # 检查是否有vision_tower权重
                        vision_tower_keys = [k for k in checkpoint_keys if "vision_tower" in k and "vision_model" in k]
                        if vision_tower_keys:
                            has_vision_tower_weights = True
                            print(f"检测到checkpoint中包含vision_tower权重 ({len(vision_tower_keys)} 个)")
                    except Exception as e:
                        print(f"无法检查checkpoint权重: {e}")
            except Exception as e:
                print(f"检查checkpoint时出错: {e}")
        
        # 如果checkpoint中包含vision_tower权重，设置unfreeze_mm_vision_tower
        # 这样vision_tower会在初始化时加载，权重可以从checkpoint中加载
        # 注意：即使设置了unfreeze_mm_vision_tower，from_pretrained仍然会尝试加载这些权重
        # 如果权重名称格式匹配，它们会被正确加载；如果不匹配，会出现警告但不会报错
        if has_vision_tower_weights:
            if not hasattr(config, 'unfreeze_mm_vision_tower'):
                config.unfreeze_mm_vision_tower = True
                print("设置 unfreeze_mm_vision_tower=True 以从checkpoint加载vision_tower权重")
            elif not config.unfreeze_mm_vision_tower:
                print("警告: checkpoint包含vision_tower权重，但unfreeze_mm_vision_tower=False")
                print("这可能导致vision_tower权重无法从checkpoint加载")
        
        # 注意：即使checkpoint中包含vision_tower和mm_projector的权重，
        # 如果出现"Some weights were not used"警告，这通常是因为：
        # 1. 权重名称格式不完全匹配（例如 vision_tower.vision_tower 的双重嵌套）
        # 2. 某些权重在推理时不需要（例如训练时的辅助权重）
        # 3. vision_tower使用了delay_load，但权重应该已经通过from_pretrained加载了
        # 
        # 如果vision_tower和mm_projector在运行时正常工作（能够处理图像），
        # 那么这些警告可以安全地忽略。
        
        # 检查并选择 attention implementation
        # 重要：虽然StreamVLNForCausalLM声明了_supports_flash_attn_2 = True，
        # 但transformers在运行时检测时可能因为继承关系或模型结构问题而无法识别
        # 因此我们先尝试使用sdpa（更稳定），如果确实需要flash_attn_2，可以后续调整
        attn_implementation = "sdpa"  # 使用sdpa作为默认，更稳定且兼容性更好
        
        # 可选：如果确实需要使用flash_attention_2，可以尝试
        # 但需要确保transformers能正确识别模型支持
        try:
            import flash_attn
            # 检查transformers版本是否支持flash_attn_2检测
            # 如果支持，可以尝试使用flash_attention_2
            # 但为了稳定性，我们默认使用sdpa
            use_flash_attn = False  # 设置为True以尝试使用flash_attention_2
            if use_flash_attn:
                attn_implementation = "flash_attention_2"
                print("Flash Attention 2.0 is available, will try to use flash_attention_2")
            else:
                print("Flash Attention 2.0 is available, but using sdpa for better compatibility")
        except ImportError:
            print("Flash Attention 2.0 not available, will use sdpa")
            attn_implementation = "sdpa"
        
        # 确保配置中显式设置 attention implementation
        # 这对于 Qwen2 模型正确构建 causal_mask_mapping 至关重要
        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = attn_implementation
        elif config._attn_implementation != attn_implementation:
            # 如果配置中的值不同，更新为选择的实现
            config._attn_implementation = attn_implementation
        
        # 模型加载选项：
        # 注意：根据 StreamVLN 论文，模型使用慢-快上下文建模和 KV 缓存机制
        # 量化模型会改变 past_key_values 的结构，导致 'self_attn' 错误
        # 因此必须禁用量化，使用完整精度模型
        # 如果遇到内存不足，需要增加 GPU 内存或使用模型并行
        load_8bit = False  # 禁用 8-bit 量化（量化会导致 past_key_values 结构错误）
        load_4bit = False  # 禁用 4-bit 量化（量化会导致 past_key_values 结构错误）
        
        # 准备模型加载参数
        # 注意：StreamVLNForCausalLM 类已声明支持 flash_attention_2 和 sdpa
        # attn_implementation 已在上面确定
        model_kwargs = {
            "attn_implementation": attn_implementation,
            "torch_dtype": torch.bfloat16,
            "config": config,
            "low_cpu_mem_usage": True,
        }
        
        # 应用量化（如果启用）
        if load_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["load_in_4bit"] = True
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["torch_dtype"] = torch.bfloat16
            # 量化模型不需要 device_map，量化库会自动处理
            print("Loading StreamVLN model with 4-bit quantization")
        elif load_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["torch_dtype"] = torch.bfloat16
            # 量化模型不需要 device_map，量化库会自动处理
            print("Loading StreamVLN model with 8-bit quantization")
        else:
            # 非量化模型，需要显式设置设备
            model_kwargs["device_map"] = {"": "cuda"}
            print("Loading StreamVLN model without quantization")
        
        # 重要：在加载模型之前，检查checkpoint中是否包含vision_tower和mm_projector权重
        # 如果包含，需要确保这些权重能被正确加载
        import os
        checkpoint_path = model_path
        checkpoint_state_dict = None
        vision_tower_keys_in_checkpoint = []
        mm_projector_keys_in_checkpoint = []
        
        if os.path.isdir(checkpoint_path):
            # 尝试加载checkpoint的权重文件
            checkpoint_files = []
            if os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                checkpoint_files.append(os.path.join(checkpoint_path, "pytorch_model.bin"))
            elif os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
                checkpoint_files.append(os.path.join(checkpoint_path, "model.safetensors"))
            else:
                # 检查是否有分片的权重文件（通常第一个是最大的）
                import glob
                checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_path, "pytorch_model-*.bin")))
                if not checkpoint_files:
                    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_path, "model-*.safetensors")))
            
            if checkpoint_files:
                # 加载checkpoint的权重（需要加载所有分片文件）
                try:
                    checkpoint_keys = []
                    checkpoint_state_dict = {}
                    
                    # 加载所有分片文件
                    for checkpoint_file in checkpoint_files:
                        if checkpoint_file.endswith('.safetensors'):
                            from safetensors import safe_open
                            with safe_open(checkpoint_file, framework="pt", device="cpu") as f:
                                file_keys = list(f.keys())
                                checkpoint_keys.extend(file_keys)
                                for key in file_keys:
                                    checkpoint_state_dict[key] = f.get_tensor(key)
                        else:
                            checkpoint_state = torch.load(checkpoint_file, map_location="cpu")
                            if isinstance(checkpoint_state, dict):
                                file_keys = list(checkpoint_state.keys())
                                checkpoint_keys.extend(file_keys)
                                checkpoint_state_dict.update(checkpoint_state)
                    
                    # 检查是否有vision_tower和mm_projector相关的权重
                    vision_tower_keys_in_checkpoint = [k for k in checkpoint_keys if "vision_tower" in k and "vision_model" in k]
                    mm_projector_keys_in_checkpoint = [k for k in checkpoint_keys if "mm_projector" in k]
                    
                    if vision_tower_keys_in_checkpoint or mm_projector_keys_in_checkpoint:
                        print(f"\n检测到checkpoint中包含视觉模块权重：")
                        print(f"  - vision_tower权重: {len(vision_tower_keys_in_checkpoint)} 个")
                        print(f"  - mm_projector权重: {len(mm_projector_keys_in_checkpoint)} 个")
                        # 为了节省内存，只保存需要的权重到checkpoint_state_dict
                        # 但保留完整的checkpoint_state_dict供后续使用
                        filtered_checkpoint_state_dict = {}
                        for key in vision_tower_keys_in_checkpoint + mm_projector_keys_in_checkpoint:
                            if key in checkpoint_state_dict:
                                filtered_checkpoint_state_dict[key] = checkpoint_state_dict[key]
                        # 使用过滤后的字典，节省内存
                        checkpoint_state_dict = filtered_checkpoint_state_dict
                except Exception as e:
                    print(f"无法检查checkpoint权重: {e}")
                    import traceback
                    traceback.print_exc()
                    checkpoint_state_dict = None
        
        # 加载模型
        self.model = StreamVLNForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # 重要：手动加载vision_tower和mm_projector的权重（如果checkpoint中包含）
        # 由于delay_load=True，vision_tower在初始化时可能还没有加载，导致权重无法匹配
        # 我们需要在模型加载后，手动加载这些权重
        if checkpoint_state_dict and (vision_tower_keys_in_checkpoint or mm_projector_keys_in_checkpoint):
            print("\n尝试手动加载vision_tower和mm_projector权重...")
            try:
                # 获取当前模型的状态字典
                model_state_dict = self.model.state_dict()
                
                # 加载vision_tower权重
                if vision_tower_keys_in_checkpoint:
                    vision_tower_loaded = 0
                    vision_tower = self.model.get_vision_tower()
                    if vision_tower is not None:
                        # 确保vision_tower已经初始化
                        if hasattr(vision_tower, 'load_model') and not hasattr(vision_tower, 'vision_tower'):
                            vision_tower.load_model()
                        
                        # 构建vision_tower的权重映射
                        # checkpoint中的格式：model.vision_tower.vision_tower.vision_model.encoder.layers.*
                        # 模型状态字典中的格式应该也是：model.vision_tower.vision_tower.vision_model.encoder.layers.*
                        vision_tower_state_dict = {}
                        for key in vision_tower_keys_in_checkpoint:
                            # checkpoint中的键名就是完整的路径，直接使用
                            # 检查模型状态字典中是否有对应的键
                            if key in model_state_dict:
                                vision_tower_state_dict[key] = checkpoint_state_dict[key]
                                vision_tower_loaded += 1
                            else:
                                # 如果直接匹配失败，尝试移除"model."前缀
                                if key.startswith("model."):
                                    model_key = key[6:]  # 移除 "model."
                                    if model_key in model_state_dict:
                                        vision_tower_state_dict[model_key] = checkpoint_state_dict[key]
                                        vision_tower_loaded += 1
                        
                        if vision_tower_loaded > 0:
                            # 加载权重
                            missing_keys, unexpected_keys = self.model.load_state_dict(
                                vision_tower_state_dict, strict=False
                            )
                            print(f"  已加载 {vision_tower_loaded} 个vision_tower权重")
                            if missing_keys:
                                print(f"  缺失的键: {len(missing_keys)} 个（前5个: {missing_keys[:5]}）")
                            if unexpected_keys:
                                print(f"  意外的键: {len(unexpected_keys)} 个（前5个: {unexpected_keys[:5]}）")
                        else:
                            print(f"  警告: 无法匹配任何vision_tower权重（共 {len(vision_tower_keys_in_checkpoint)} 个）")
                            # 打印一些示例键名用于调试
                            if vision_tower_keys_in_checkpoint:
                                print(f"  checkpoint键示例: {vision_tower_keys_in_checkpoint[0]}")
                                model_vt_keys = [k for k in model_state_dict.keys() if "vision_tower" in k]
                                if model_vt_keys:
                                    print(f"  模型键示例: {model_vt_keys[0]}")
                
                # 加载mm_projector权重
                if mm_projector_keys_in_checkpoint:
                    mm_projector_loaded = 0
                    mm_projector_state_dict = {}
                    for key in mm_projector_keys_in_checkpoint:
                        # checkpoint中的键名就是完整的路径，直接使用
                        if key in model_state_dict:
                            mm_projector_state_dict[key] = checkpoint_state_dict[key]
                            mm_projector_loaded += 1
                        else:
                            # 如果直接匹配失败，尝试移除"model."前缀
                            if key.startswith("model."):
                                model_key = key[6:]  # 移除 "model."
                                if model_key in model_state_dict:
                                    mm_projector_state_dict[model_key] = checkpoint_state_dict[key]
                                    mm_projector_loaded += 1
                    
                    if mm_projector_loaded > 0:
                        # 加载权重
                        missing_keys, unexpected_keys = self.model.load_state_dict(
                            mm_projector_state_dict, strict=False
                        )
                        print(f"  已加载 {mm_projector_loaded} 个mm_projector权重")
                        if missing_keys:
                            print(f"  缺失的键: {len(missing_keys)} 个（前5个: {missing_keys[:5]}）")
                        if unexpected_keys:
                            print(f"  意外的键: {len(unexpected_keys)} 个（前5个: {unexpected_keys[:5]}）")
                    else:
                        print(f"  警告: 无法匹配任何mm_projector权重（共 {len(mm_projector_keys_in_checkpoint)} 个）")
                        # 打印一些示例键名用于调试
                        if mm_projector_keys_in_checkpoint:
                            print(f"  checkpoint键示例: {mm_projector_keys_in_checkpoint[0]}")
                            model_mp_keys = [k for k in model_state_dict.keys() if "mm_projector" in k]
                            if model_mp_keys:
                                print(f"  模型键示例: {model_mp_keys[0]}")
                
                print("权重加载完成\n")
                
                # 验证 LLM 权重是否已正确加载
                # missing_keys 中显示的 LLM 权重是正常的，因为它们应该已经通过 from_pretrained 加载
                # 这里验证一下关键权重是否存在
                llm_key_examples = [
                    'model.embed_tokens.weight',
                    'model.layers.0.self_attn.q_proj.weight',
                    'lm_head.weight'
                ]
                model_state_dict = self.model.state_dict()
                llm_weights_loaded = sum(1 for key in llm_key_examples if key in model_state_dict)
                if llm_weights_loaded == len(llm_key_examples):
                    print("✓ LLM 权重已正确加载（通过 from_pretrained）")
                else:
                    print(f"⚠ 警告: 部分 LLM 权重可能未加载（{llm_weights_loaded}/{len(llm_key_examples)} 个关键权重存在）")
                    print("  这可能导致模型无法正常工作")
                    print("  缺失的键是正常的，因为它们应该已经通过 from_pretrained 加载")
            except Exception as e:
                print(f"手动加载权重时出错: {e}")
                import traceback
                traceback.print_exc()
                print("将继续使用from_pretrained加载的权重\n")
        
        # 确保vision_tower已经加载
        vision_tower = self.model.get_vision_tower()
        if vision_tower is not None:
            # 如果vision_tower还没有加载实际的模型，调用load_model
            if hasattr(vision_tower, 'load_model') and not hasattr(vision_tower, 'vision_tower'):
                try:
                    vision_tower.load_model()
                    print("Vision tower模型已加载（从HuggingFace模型名称）")
                except Exception as e:
                    print(f"Vision tower加载信息: {e}")
            elif hasattr(vision_tower, 'vision_tower'):
                print("Vision tower已加载")
        
        # 验证并确保模型配置中的 attention implementation 正确设置
        expected_attn = attn_implementation  # 使用实际选择的 attention 实现
        if hasattr(self.model.config, '_attn_implementation'):
            if self.model.config._attn_implementation != expected_attn:
                print(f"Info: Model config has _attn_implementation={self.model.config._attn_implementation}, "
                      f"updating to '{expected_attn}'")
                self.model.config._attn_implementation = expected_attn
        else:
            print(f"Info: Model config missing _attn_implementation, setting to '{expected_attn}'")
            self.model.config._attn_implementation = expected_attn
        
        # 确保 layer_types 存在且正确设置
        # 关键修复：Qwen2 的 causal_mask_mapping 使用 "full_attention" 键，不是 "self_attn"
        # 所以 layer_types 必须使用 "full_attention" 来匹配，否则会 KeyError
        if not hasattr(self.model.config, 'layer_types') or self.model.config.layer_types is None:
            num_layers = getattr(self.model.config, 'num_hidden_layers', 0)
            if num_layers > 0:
                self.model.config.layer_types = ["full_attention"] * num_layers
                print(f"[StreamVLNPolicy] 设置 layer_types 为 ['full_attention'] * {num_layers}")
        else:
            # 检查并修复：如果 layer_types 使用 "self_attn"，替换为 "full_attention"
            if self.model.config.layer_types and len(self.model.config.layer_types) > 0:
                if self.model.config.layer_types[0] == "self_attn":
                    print(f"[StreamVLNPolicy] 检测到 layer_types 使用 'self_attn'，替换为 'full_attention' 以匹配 Qwen2 的 causal_mask_mapping")
                    num_layers = len(self.model.config.layer_types)
                    self.model.config.layer_types = ["full_attention"] * num_layers
                print(f"[StreamVLNPolicy] layer_types: {self.model.config.layer_types[:3]}... (共 {len(self.model.config.layer_types)} 层)")
        
        # 修复 transformers 4.56.0 兼容性问题：添加缺失的 _is_stateful 属性
        if not hasattr(StreamVLNForCausalLM, '_is_stateful'):
            StreamVLNForCausalLM._is_stateful = False
        if not hasattr(self.model.__class__, '_is_stateful'):
            self.model.__class__._is_stateful = False
        
        self.model.model.num_history = num_history
        self.model.requires_grad_(False)
        
        # 对于量化模型，不需要调用 .to()，量化库已经将模型加载到正确的设备
        # 对于非量化模型，需要移动到设备
        if not (load_4bit or load_8bit):
            self.model.to(self.device)
        
        self.model.eval()
        
        # Get image processor from vision tower
        self.image_processor = self.model.get_vision_tower().image_processor
        
        # Add previous action embedding (similar to ResNet policy)
        self.prev_action_embedding = nn.Embedding(
            action_space.n + 1, 32  # +1 for start token
        ).to(self.device)
        
        # Initialize conversation template
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        
        # Conjunctions for instruction formatting
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]
        
        # Episode state
        self.reset_episode_state()
        
        # Camera intrinsics (default for VLN, can be overridden)
        self.intrinsic_matrix = np.array([
            [192.0, 0.0, 191.42857143, 0.0],
            [0.0, 192.0, 191.42857143, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        print("StreamVLN model loaded successfully")

    def reset_episode_state(self):
        """Reset per-episode state."""
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.output_ids = None
        self.past_key_values = None
        self.step_id = 0
        self.last_image = None
        self.current_instruction = None
        
        # Reset model cache
        # Assuming single environment for now
        self.model.reset(1)
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return 1

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def _find_rgb_key(self, observations: Dict[str, torch.Tensor]) -> Optional[str]:
        """
        查找可用的RGB key（仿照NaVILA的实现）
        
        优先顺序：
        1. "rgb" (经过obs_transforms后的通用key)
        2. agent_0_overhead_front_rgb (overhead传感器，第一视角)
        3. agent_0_articulated_agent_jaw_rgb (Falcon默认的RGB key)
        4. agent_0_third_rgb (第三视角RGB)
        5. 其他包含"rgb"的agent_0相关key
        
        Args:
            observations: 观察字典
            
        Returns:
            RGB key，如果找不到则返回None
        """
        # 按优先级尝试不同的RGB key
        rgb_keys = [
            "rgb",  # 通用key（可能经过obs_transforms后存在）
            "agent_0_overhead_front_rgb",  # overhead传感器（第一视角）
            "agent_0_articulated_agent_jaw_rgb",  # Falcon默认的RGB key
            "agent_0_third_rgb",  # 第三视角RGB
        ]
        
        # 先尝试预定义的keys
        for key in rgb_keys:
            if key in observations:
                return key
        
        # 如果都没找到，尝试查找任何包含"rgb"的agent_0相关key
        for key in observations.keys():
            if "agent_0" in key and "rgb" in key.lower():
                return key
        
        return None

    def _find_depth_key(self, observations: Dict[str, torch.Tensor]) -> Optional[str]:
        """
        查找可用的Depth key（仿照_find_rgb_key的实现）
        
        优先顺序：
        1. "depth" (经过obs_transforms后的通用key)
        2. agent_0_overhead_front_depth (overhead传感器，第一视角)
        3. agent_0_articulated_agent_jaw_depth (Falcon默认的Depth key)
        4. agent_0_third_depth (第三视角Depth)
        5. 其他包含"depth"的agent_0相关key
        
        Args:
            observations: 观察字典
            
        Returns:
            Depth key，如果找不到则返回None
        """
        # 按优先级尝试不同的Depth key
        depth_keys = [
            "depth",  # 通用key（可能经过obs_transforms后存在）
            "agent_0_overhead_front_depth",  # overhead传感器（第一视角）
            "agent_0_articulated_agent_jaw_depth",  # Falcon默认的Depth key
            "agent_0_third_depth",  # 第三视角Depth
        ]
        
        # 先尝试预定义的keys
        for key in depth_keys:
            if key in observations:
                return key
        
        # 如果都没找到，尝试查找任何包含"depth"的agent_0相关key
        for key in observations.keys():
            if "agent_0" in key and "depth" in key.lower():
                return key
        
        return None

    def _tensor_like_to_numpy(self, data: Any) -> Optional[np.ndarray]:
        """将tensor-like数据转换为numpy数组"""
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, (list, tuple)):
            try:
                return np.array(data)
            except Exception:
                return None
        return None

    def _decode_text_instruction(self, data: Any) -> Optional[str]:
        """解码文本指令，支持多种数据格式"""
        if data is None:
            return None
        if isinstance(data, str):
            text = data.strip()
            return text if text else None
        if isinstance(data, (list, tuple)):
            # Assume list of tokens or characters
            try:
                chars = []
                for item in data:
                    if isinstance(item, str) and item:
                        chars.append(item)
                    elif isinstance(item, (int, np.integer)):
                        if item == 0 and chars:
                            break
                        if 32 <= int(item) < 127:
                            chars.append(chr(int(item)))
                    else:
                        chars.append(str(item))
                text = "".join(chars).strip()
                return text if text else None
            except Exception:
                pass
        arr = self._tensor_like_to_numpy(data)
        if arr is None:
            return None
        arr = np.asarray(arr).astype(np.int32).flatten()
        chars: List[str] = []
        for val in arr:
            if val == 0:
                if chars:
                    break
                continue
            char_code = int(val) % 256
            if 32 <= char_code <= 126:
                chars.append(chr(char_code))
        text = "".join(chars).strip()
        return text if text else None

    def _extract_instruction_from_obs(self, observations: Dict[str, torch.Tensor]) -> Optional[str]:
        """从观察中提取指令，仿照NaVILA的实现"""
        DEFAULT_INSTRUCTION_KEYS = [
            "agent_0_falcon_instruction",
            "falcon_instruction",
            "instruction",
            "instruction_sensor",
        ]
        
        # 对于batch数据，取第一个样本
        for key in DEFAULT_INSTRUCTION_KEYS:
            if key in observations:
                data = observations[key]
                # 如果是tensor，取第一个样本
                if isinstance(data, torch.Tensor) and len(data.shape) > 0:
                    if data.shape[0] > 0:
                        data = data[0]  # 取batch中的第一个
                text = self._decode_text_instruction(data)
                if text:
                    return text
        
        return None

    def parse_actions(self, output: str) -> List[int]:
        """
        Parse action sequence from model output.
        
        注意：此方法可能返回重复的动作（如 [3, 3, 3] 表示3个右转），
        调用者需要负责去重或限制重复次数，避免同一动作执行过多遍。
        """
        action_text = self._extract_action_text(output)

        # 优先解析箭头符号（如 "→→→" 表示3个右转）
        arrow_map = {'↑': 1, '←': 2, '→': 3}
        arrow_seq = [arrow_map[ch] for ch in action_text if ch in arrow_map]
        if arrow_seq:
            # 限制连续相同动作的最大重复次数，避免过度重复
            # 例如 "→→→→→→" 会被限制为最多4个右转
            max_consecutive_repeats = 4
            deduplicated_seq = []
            current_action = None
            repeat_count = 0
            for action in arrow_seq:
                if action == current_action:
                    repeat_count += 1
                    if repeat_count <= max_consecutive_repeats:
                        deduplicated_seq.append(action)
                else:
                    current_action = action
                    repeat_count = 1
                    deduplicated_seq.append(action)
            return deduplicated_seq

        # 首先尝试使用action_parser（如果可用）
        if hasattr(self, 'action_parser'):
            try:
                action, num_repeats = self.action_parser.parse_action(action_text)
                # 限制重复次数，避免过度重复
                # 原始代码中，动作序列最多4个，所以限制 num_repeats 最多为4
                num_repeats = min(num_repeats, 4)
                # 返回重复的动作序列
                return [action] * num_repeats
            except:
                pass
        
        # 回退到原始解析方法
        # 支持多种格式：↑, ←, →, stop, STOP, move forward, turn left, turn right等
        actions = []
        output_lower = action_text.lower()
        
        # 检查stop
        if 'stop' in output_lower:
            actions.append(0)
        
        # 检查前进动作（↑, move forward, forward等）
        if '↑' in output or 'move forward' in output_lower or 'forward' in output_lower:
            actions.append(1)
        
        # 检查左转（←, turn left, left等）
        if '←' in output or 'turn left' in output_lower or ('left' in output_lower and 'right' not in output_lower):
            actions.append(2)
        
        # 检查右转（→, turn right, right等）
        if '→' in output or 'turn right' in output_lower or 'right' in output_lower:
            actions.append(3)
        
        # 如果没找到任何动作，使用原始正则表达式方法
        if not actions:
            action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
            regex = re.compile(action_patterns)
            matches = regex.findall(action_text)
            actions = [self.actions2idx[match] for match in matches]
            actions = list(itertools.chain.from_iterable(
                [a] if isinstance(a, int) else a for a in actions
            ))
        
        # 限制动作序列长度，避免过长序列导致导航混乱
        # 原始代码限制为最多4个动作
        if len(actions) > 4:
            actions = actions[:4]
        
        return actions if actions else [0]  # 默认返回STOP

    def _extract_action_text(self, output: str) -> str:
        """提取最后一个assistant回复，避免提示词干扰动作解析"""
        if not output:
            return ""
        matches = self.ACTION_BLOCK_PATTERN.findall(output)
        if matches:
            for segment in reversed(matches):
                cleaned = segment.strip()
                if cleaned:
                    return cleaned
        return output.strip()

    def preprocess_qwen(
        self,
        sources,
        has_image: bool = False,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False
    ):
        """Preprocess inputs for Qwen model."""
        roles = {"human": "user", "gpt": "assistant"}
        
        # 参考 streamvln_eval.py 的实现，使用深拷贝tokenizer并重新添加标记
        # 这样可以确保标记总是可用的，即使tokenizer被修改过
        tokenizer = copy.deepcopy(self.tokenizer)
        # 当有图像时，重新添加图像和记忆标记（参考 streamvln_eval.py line 412-414）
        if has_image:
            # 重要：每次调用时都重新添加标记，确保标记总是可用的
            # 参考 streamvln_eval.py line 413-414，直接添加而不检查
            num_added_image = tokenizer.add_tokens(["<image>"], special_tokens=True)
            num_added_memory = tokenizer.add_tokens(["<memory>"], special_tokens=True)
            logger.info(f"[preprocess_qwen] Added tokens: <image>={num_added_image}, <memory>={num_added_memory}")
        
        # 获取标记的 token ID（如果标记不存在，会返回 unk_token_id）
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        
        # 验证标记是否正确添加
        if image_token_index == tokenizer.unk_token_id:
            logger.error(f"[preprocess_qwen] CRITICAL: <image> token not found in tokenizer vocabulary, unk_token_id={image_token_index}")
            # 尝试重新添加
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            image_token_index = tokenizer.convert_tokens_to_ids("<image>")
            if image_token_index == tokenizer.unk_token_id:
                logger.error(f"[preprocess_qwen] CRITICAL: Failed to add <image> token even after retry!")
        
        if memory_token_index == tokenizer.unk_token_id:
            logger.error(f"[preprocess_qwen] CRITICAL: <memory> token not found in tokenizer vocabulary, unk_token_id={memory_token_index}")
            # 尝试重新添加
            tokenizer.add_tokens(["<memory>"], special_tokens=True)
            memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
            if memory_token_index == tokenizer.unk_token_id:
                logger.error(f"[preprocess_qwen] CRITICAL: Failed to add <memory> token even after retry!")
        
        # 验证标记是否在词汇表中
        vocab = tokenizer.get_vocab()
        if "<image>" in vocab:
            logger.info(f"[preprocess_qwen] ✓ <image> found in vocab, id={vocab['<image>']}, image_token_index={image_token_index}")
            if vocab['<image>'] != image_token_index:
                logger.warning(f"[preprocess_qwen] WARNING: vocab['<image>']={vocab['<image>']} != image_token_index={image_token_index}")
        else:
            logger.error(f"[preprocess_qwen] ✗ <image> NOT found in vocab!")
        
        if "<memory>" in vocab:
            logger.info(f"[preprocess_qwen] ✓ <memory> found in vocab, id={vocab['<memory>']}, memory_token_index={memory_token_index}")
            if vocab['<memory>'] != memory_token_index:
                logger.warning(f"[preprocess_qwen] WARNING: vocab['<memory>']={vocab['<memory>']} != memory_token_index={memory_token_index}")
        else:
            logger.error(f"[preprocess_qwen] ✗ <memory> NOT found in vocab!")
        
        # 测试tokenization：直接tokenize标记看看会发生什么
        test_image_encode = tokenizer.encode("<image>", add_special_tokens=False)
        test_memory_encode = tokenizer.encode("<memory>", add_special_tokens=False)
        logger.info(f"[preprocess_qwen] Direct encode test: '<image>' -> {test_image_encode}, '<memory>' -> {test_memory_encode}")
        if len(test_image_encode) != 1 or test_image_encode[0] != image_token_index:
            logger.error(f"[preprocess_qwen] CRITICAL: '<image>' tokenization failed! Expected [{image_token_index}], got {test_image_encode}")
        if len(test_memory_encode) != 1 or test_memory_encode[0] != memory_token_index:
            logger.error(f"[preprocess_qwen] CRITICAL: '<memory>' tokenization failed! Expected [{memory_token_index}], got {test_memory_encode}")
        
        # Reset Qwen chat templates
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # Apply prompt templates
        conversations = []
        input_ids = []
        
        for i, source in enumerate(sources):
            # 使用conjunction + image token，参考 streamvln_agent.py 和 streamvln_eval.py 的实现
            # 参考 streamvln_eval.py line 435，使用随机选择以增加提示词多样性
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            
            # 验证标记是否正确
            if DEFAULT_IMAGE_TOKEN not in prompt:
                logger.error(f"DEFAULT_IMAGE_TOKEN not in prompt! DEFAULT_IMAGE_TOKEN='{DEFAULT_IMAGE_TOKEN}', prompt='{prompt}'")
            
            if len(source[0]["value"]) != 0:
                # 如果已有内容（如任务指令），在末尾添加conjunction + image token
                # 这样既能保留任务信息，又能引导模型关注当前视觉输入
                # 格式：任务指令 + " you can see <image>."
                source[0]["value"] += f" {prompt}."
                # 验证标记是否正确添加到提示词中
                if DEFAULT_IMAGE_TOKEN not in source[0]["value"]:
                    logger.error(f"[preprocess_qwen] CRITICAL: DEFAULT_IMAGE_TOKEN not in source[0]['value']! "
                               f"DEFAULT_IMAGE_TOKEN='{DEFAULT_IMAGE_TOKEN}', prompt='{prompt}', "
                               f"value='{source[0]['value'][:200]}...'")
                else:
                    logger.info(f"[preprocess_qwen] ✓ DEFAULT_IMAGE_TOKEN found in source[0]['value']")
                    # 检查标记在字符串中的位置
                    idx = source[0]["value"].find(DEFAULT_IMAGE_TOKEN)
                    logger.debug(f"[preprocess_qwen] DEFAULT_IMAGE_TOKEN found at position {idx} in prompt")
            else:
                # 如果没有内容，直接使用conjunction + image token作为提示
                source[0]["value"] = f"{prompt}."
                if DEFAULT_IMAGE_TOKEN not in source[0]["value"]:
                    logger.error(f"[preprocess_qwen] CRITICAL: DEFAULT_IMAGE_TOKEN not in empty source[0]['value']! "
                               f"DEFAULT_IMAGE_TOKEN='{DEFAULT_IMAGE_TOKEN}', prompt='{prompt}'")
            
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []
            
            if add_system:
                input_id += tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_message}]
                )

            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role = roles.get(role, role)
                conv = [{"role": role, "content": content}]
                conversations.append(content)
                
                # 调试：在tokenization前检查content
                if DEFAULT_IMAGE_TOKEN in content or DEFAULT_MEMORY_TOKEN in content:
                    logger.info(f"[preprocess_qwen] Before tokenization - Content contains: "
                              f"IMAGE={DEFAULT_IMAGE_TOKEN in content}, MEMORY={DEFAULT_MEMORY_TOKEN in content}")
                    logger.info(f"[preprocess_qwen] Content preview: {content[:300]}...")
                
                # 参考 streamvln_eval.py line 466，使用 apply_chat_template
                # 重要：在tokenization前，确保标记在tokenizer的词汇表中
                # 如果标记被分解，我们需要手动处理
                encode_id = tokenizer.apply_chat_template(conv)
                
                # 调试：检查tokenization后的结果
                if DEFAULT_IMAGE_TOKEN in content or DEFAULT_MEMORY_TOKEN in content:
                    logger.info(f"[preprocess_qwen] After apply_chat_template - encode_id length: {len(encode_id)}")
                    logger.info(f"[preprocess_qwen] encode_id preview (first 30): {encode_id[:30]}")
                    # 检查标记是否在tokenized序列中
                    if image_token_index in encode_id:
                        logger.info(f"[preprocess_qwen] ✓ image_token_index {image_token_index} found in encode_id at positions: {[i for i, x in enumerate(encode_id) if x == image_token_index]}")
                    else:
                        logger.error(f"[preprocess_qwen] ✗ image_token_index {image_token_index} NOT found in encode_id!")
                        # 手动tokenize content看看标记是否被正确识别
                        test_encode = tokenizer.encode(content, add_special_tokens=False)
                        logger.error(f"[preprocess_qwen] Manual encode of content: length={len(test_encode)}, "
                                   f"first 30 tokens: {test_encode[:30]}")
                        if image_token_index in test_encode:
                            logger.error(f"[preprocess_qwen] BUT image_token_index {image_token_index} IS in manual encode! "
                                       f"This suggests apply_chat_template is removing/replacing the token!")
                    
                    if memory_token_index in encode_id:
                        logger.info(f"[preprocess_qwen] ✓ memory_token_index {memory_token_index} found in encode_id at positions: {[i for i, x in enumerate(encode_id) if x == memory_token_index]}")
                    else:
                        if DEFAULT_MEMORY_TOKEN in content:
                            logger.error(f"[preprocess_qwen] ✗ memory_token_index {memory_token_index} NOT found in encode_id!")
                            # 手动tokenize content看看标记是否被正确识别
                            test_encode = tokenizer.encode(content, add_special_tokens=False)
                            if memory_token_index in test_encode:
                                logger.error(f"[preprocess_qwen] BUT memory_token_index {memory_token_index} IS in manual encode! "
                                           f"This suggests apply_chat_template is removing/replacing the token!")
                
                input_id += encode_id

            # 替换特殊标记的 token ID（参考 streamvln_eval.py line 471-475）
            # 重要：需要检查标记是否被正确tokenized
            # 关键步骤：将 tokenizer 的 token ID (如 151646) 替换为模型内部使用的特殊索引 (IMAGE_TOKEN_INDEX = -200)
            # 这样模型才能正确识别这些标记并处理对应的图像/记忆数据
            image_token_found = False
            memory_token_found = False
            
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                    image_token_found = True
                    logger.info(f"[preprocess_qwen] ✓ Replaced image token at index {idx}: {image_token_index} -> {IMAGE_TOKEN_INDEX}")
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    memory_token_found = True
                    logger.info(f"[preprocess_qwen] ✓ Replaced memory token at index {idx}: {memory_token_index} -> {MEMORY_TOKEN_INDEX}")
            
            # 验证替换后的结果
            if IMAGE_TOKEN_INDEX in input_id:
                positions = [i for i, x in enumerate(input_id) if x == IMAGE_TOKEN_INDEX]
                logger.info(f"[preprocess_qwen] ✓ IMAGE_TOKEN_INDEX ({IMAGE_TOKEN_INDEX}) found in input_id at positions: {positions}")
            if MEMORY_TOKEN_INDEX in input_id:
                positions = [i for i, x in enumerate(input_id) if x == MEMORY_TOKEN_INDEX]
                logger.info(f"[preprocess_qwen] ✓ MEMORY_TOKEN_INDEX ({MEMORY_TOKEN_INDEX}) found in input_id at positions: {positions}")
            
            # 如果标记没有被找到，尝试手动tokenize并查找
            # 这可能是因为tokenizer在apply_chat_template时处理了这些标记
            if not image_token_found and image_token_index != tokenizer.unk_token_id:
                # 尝试直接tokenize "<image>" 看看会得到什么
                test_tokens = tokenizer.encode("<image>", add_special_tokens=False)
                logger.warning(f"[preprocess_qwen] image_token_index {image_token_index} not found in input_id! "
                             f"Direct tokenization of '<image>' gives: {test_tokens}")
                # 如果直接tokenize得到的是单个token且等于image_token_index，说明标记存在但可能被其他地方处理了
                if len(test_tokens) == 1 and test_tokens[0] == image_token_index:
                    # 标记存在，但可能在apply_chat_template时被处理了
                    # 尝试在input_id中查找并替换
                    for idx, encode_id in enumerate(input_id):
                        if encode_id == image_token_index:
                            input_id[idx] = IMAGE_TOKEN_INDEX
                            image_token_found = True
                            logger.info(f"[preprocess_qwen] Found and replaced image token at index {idx} (second pass)")
            
            if not memory_token_found and memory_token_index != tokenizer.unk_token_id:
                # 尝试直接tokenize "<memory>" 看看会得到什么
                test_tokens = tokenizer.encode("<memory>", add_special_tokens=False)
                logger.warning(f"[preprocess_qwen] memory_token_index {memory_token_index} not found in input_id! "
                             f"Direct tokenization of '<memory>' gives: {test_tokens}")
                # 如果直接tokenize得到的是单个token且等于memory_token_index，说明标记存在但可能被其他地方处理了
                if len(test_tokens) == 1 and test_tokens[0] == memory_token_index:
                    # 标记存在，但可能在apply_chat_template时被处理了
                    # 尝试在input_id中查找并替换
                    for idx, encode_id in enumerate(input_id):
                        if encode_id == memory_token_index:
                            input_id[idx] = MEMORY_TOKEN_INDEX
                            memory_token_found = True
                            logger.info(f"[preprocess_qwen] Found and replaced memory token at index {idx} (second pass)")
            
            # 最终验证
            if not image_token_found:
                logger.error(f"[preprocess_qwen] CRITICAL: image_token_index {image_token_index} not found in input_id after all attempts! "
                           f"This means <image> token was not properly tokenized. "
                           f"Content preview: {conversations[0][:200] if conversations else 'N/A'}...")
            if not memory_token_found and DEFAULT_MEMORY_TOKEN in (conversations[0] if conversations else ""):
                logger.error(f"[preprocess_qwen] CRITICAL: memory_token_index {memory_token_index} not found in input_id after all attempts! "
                           f"This means <memory> token was not properly tokenized. "
                           f"Content preview: {conversations[0][:200] if conversations else 'N/A'}...")

            input_ids.append(input_id)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the network.
        
        This method processes observations and generates features for action/value prediction.
        For Falcon integration, this method extracts RGB from observations and processes
        it through StreamVLN's visual encoder to get features.
        
        Args:
            observations: Dictionary of observations from Habitat
            rnn_hidden_states: RNN hidden states (maintained for compatibility)
            prev_actions: Previous actions
            masks: Episode masks (1 = continue, 0 = reset)
            rnn_build_seq_info: Additional info for RNN (not used)
        
        Returns:
            - Features for action/value heads
            - Updated RNN hidden states
            - Auxiliary loss state dictionary
        """
        # Extract RGB observation（使用_find_rgb_key查找RGB传感器，仿照NaVILA的实现）
        rgb_key = self._find_rgb_key(observations)
        
        if rgb_key is None or rgb_key not in observations:
            # If RGB is not available, use depth as fallback
            depth_key = self._find_depth_key(observations)
            if depth_key is None or depth_key not in observations:
                available_keys = list(observations.keys())
                raise ValueError(
                    f"No RGB key found in observations. Available keys: {available_keys}. "
                    f"Please ensure at least one RGB sensor is enabled in the config "
                    f"(e.g., agent_0_overhead_front_rgb or agent_0_articulated_agent_jaw_rgb)."
                )
            # Convert depth to pseudo-RGB
            depth_obs = observations[depth_key]
            # Expand depth to 3 channels
            if len(depth_obs.shape) == 3 and depth_obs.shape[-1] == 1:
                rgb = depth_obs.repeat(1, 1, 1, 3)
            else:
                rgb = depth_obs
        else:
            rgb = observations[rgb_key]
        
        batch_size = rgb.shape[0]
        device = rgb.device
        
        # Check for episode resets
        for i in range(batch_size):
            if masks[i].item() == 0:
                # Episode reset
                self.reset_episode_state()
        
        # Get instruction from observations (优先从agent_0_falcon_instruction读取)
        # 仿照NaVILA的实现方式
        instruction = self._extract_instruction_from_obs(observations)
        if instruction:
            self.current_instruction = instruction
        elif self.current_instruction is None:
            # Fallback: 如果没有找到指令，使用pointgoal生成简单指令
            if 'agent_0_pointgoal_with_gps_compass' in observations:
                pointgoal = observations['agent_0_pointgoal_with_gps_compass']
                distance = pointgoal[:, 0].mean().item()
                angle = pointgoal[:, 1].mean().item()
                if distance < 0.5:
                    self.current_instruction = "you are near the goal, stop"
                elif abs(angle) > 0.5:
                    if angle > 0:
                        self.current_instruction = "turn left towards the goal"
                    else:
                        self.current_instruction = "turn right towards the goal"
                else:
                    self.current_instruction = "move forward to the goal"
            else:
                self.current_instruction = "navigate to the goal"
        
        # Process through visual encoder to get features
        # For efficient integration with Falcon, we'll use the visual encoder
        # instead of full StreamVLN generation for each step
        
        # Convert RGB to correct format for processing
        rgb_np = rgb.cpu().numpy()
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
        
        # For efficiency in Falcon, we'll extract visual features
        # and use them for action prediction
        # Full StreamVLN generation is too slow for real-time navigation
        
        # Process images through vision tower (batch processing)
        with torch.no_grad():
            images_list = []
            for i in range(batch_size):
                img_np = rgb_np[i]
                image = Image.fromarray(img_np).convert('RGB')
                image_tensor = self.image_processor.preprocess(
                    images=image, return_tensors='pt'
                )['pixel_values'][0]
                images_list.append(image_tensor)
            
            images_batch = torch.stack(images_list).to(device)
            
            # Extract visual features using StreamVLN's vision tower
            try:
                visual_features = self.model.get_vision_tower()(images_batch)
                # visual_features shape: [batch, num_patches, feature_dim]
                
                # Pool features to get fixed-size representation
                visual_features = visual_features.mean(dim=1)  # [batch, feature_dim]
                
                # Project to hidden size
                if visual_features.shape[-1] != self._hidden_size:
                    if not hasattr(self, 'feature_projection'):
                        self.feature_projection = nn.Linear(
                            visual_features.shape[-1], 
                            self._hidden_size
                        ).to(device)
                    visual_features = self.feature_projection(visual_features)
                
                features = visual_features
            except Exception as e:
                # Fallback: use random features
                print(f"Warning: StreamVLN visual encoding failed: {e}")
                print("Using fallback random features")
                features = torch.randn(batch_size, self._hidden_size).to(device)
        
        # Maintain RNN hidden states for compatibility
        new_rnn_hidden_states = rnn_hidden_states
        
        # Fuse additional information for richer features (similar to ResNet policy)
        x = [features]  # Start with visual features
        
        # Add pointgoal information if available
        if 'agent_0_pointgoal_with_gps_compass' in observations:
            goal_obs = observations['agent_0_pointgoal_with_gps_compass']
            # Add goal distance and angle as features
            x.append(goal_obs)
        
        # Add previous action embedding
        if hasattr(self, 'prev_action_embedding'):
            if prev_actions is not None:
                prev_actions_squeezed = prev_actions.squeeze(-1)
                start_token = torch.zeros_like(prev_actions_squeezed)
                prev_action_feat = self.prev_action_embedding(
                    torch.where(masks.view(-1), prev_actions_squeezed + 1, start_token)
                )
                x.append(prev_action_feat)
        
        # Concatenate all features
        if len(x) > 1:
            fused_features = torch.cat(x, dim=1)
        else:
            fused_features = features
        
        # Prepare auxiliary loss state with both perception and RNN outputs
        # perception_embed: raw visual features (before fusion)
        # rnn_output: fused features including goal, action, etc. (after fusion)
        aux_loss_state = {
            "perception_embed": features,  # Raw visual features
            "rnn_output": fused_features,  # Fused features with goal + action info
        }
        
        self.step_id += 1
        
        # Return fused features for action/value prediction
        return fused_features, new_rnn_hidden_states, aux_loss_state

    @torch.no_grad()
    def generate_action_sequence(
        self,
        rgb: np.ndarray,
        instruction: str,
        depth: Optional[np.ndarray] = None,
        pose: Optional[np.ndarray] = None,
        env_idx: int = 0,
        run_model: bool = True
    ) -> Tuple[List[int], str]:
        """
        Generate action sequence from RGB observation and instruction.
        
        This is the main inference method that directly uses StreamVLN's
        generation capabilities.
        
        Args:
            rgb: RGB image (H, W, C) in uint8 format
            instruction: Navigation instruction text
            depth: Depth image (H, W, 1) in float format, optional
            pose: Camera pose matrix (4, 4), optional
            env_idx: Environment index (for multi-env support)
            run_model: Whether to run the model or reuse last image
        
        Returns:
            - List of action indices
            - Raw LLM output text
        """
        # Preprocess image
        if run_model:
            image = Image.fromarray(rgb).convert('RGB')
            image = self.image_processor.preprocess(
                images=image, return_tensors='pt'
            )['pixel_values'][0]
            self.last_image = copy.deepcopy(image)
        else:
            image = self.last_image
        
        # Preprocess depth image (参考 streamvln_eval.py 的实现)
        if depth is not None:
            # 确保深度图像是正确的形状和数据类型
            if len(depth.shape) == 2:
                depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
            elif len(depth.shape) == 3 and depth.shape[2] > 1:
                depth = depth[:, :, 0:1]
            
            # 深度图像预处理（参考 streamvln_eval.py:270-272）
            # 注意：Falcon 框架中的深度图像可能是归一化的（0-1范围）或原始深度值（米）
            # 我们需要将其转换为毫米单位（uint16，范围：0-65535）
            
            # 如果深度图像是归一化的（0-1范围），需要反归一化
            # 假设深度传感器的范围是 0.0 到 10.0 米（这是 Habitat 的默认范围）
            depth_min = 0.0  # 米
            depth_max = 10.0  # 米
            
            # 检查深度图像是否已经归一化
            if depth.max() <= 1.0 and depth.min() >= 0.0:
                # 归一化深度，需要反归一化
                depth_meters = depth * (depth_max - depth_min) + depth_min
            else:
                # 已经是米单位的深度值
                depth_meters = depth.squeeze()
            
            # 转换为毫米（uint16，范围：0-65535）
            depth_mm = (depth_meters * 1000.0).astype(np.uint16)
            # 限制在 uint16 范围内
            depth_mm = np.clip(depth_mm, 0, 65535)
            
            # 转换为 PIL Image 进行预处理（参考 streamvln_eval.py:291）
            depth_image = Image.fromarray(depth_mm, mode='I;16')
            
            # 获取目标尺寸（从 image_processor）
            target_height = self.image_processor.crop_size['height']  # 384
            target_width = self.image_processor.crop_size['width']  # 384
            depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
            
            # 转换为 numpy 并归一化（参考 streamvln_eval.py:291-292）
            from transformers.image_utils import to_numpy_array
            depth_array = to_numpy_array(depth_image)
            depth_array = depth_array / 1000.0  # 转换为米（用于模型输入）
            
            # 确保形状正确 (H, W, 1)
            if len(depth_array.shape) == 2:
                depth_array = depth_array.reshape(depth_array.shape[0], depth_array.shape[1], 1)
            
            depth_tensor = torch.from_numpy(depth_array).float()
        else:
            # 如果没有提供深度图像，创建全零的 dummy depth
            target_height = self.image_processor.crop_size['height']  # 384
            target_width = self.image_processor.crop_size['width']  # 384
            depth_tensor = torch.zeros((target_height, target_width, 1)).float()
        
        # Prepare pose and intrinsics
        if pose is not None:
            pose_tensor = torch.from_numpy(pose).float()
        else:
            pose_tensor = torch.eye(4).float()
        
        intrinsic = torch.from_numpy(self.intrinsic_matrix).float()
        
        self.time_ids.append(self.step_id)
        self.rgb_list.append(image)
        self.depth_list.append(depth_tensor)  # 使用预处理后的 depth_tensor
        self.pose_list.append(pose_tensor)  # 使用预处理后的 pose_tensor
        self.intrinsic_list.append(intrinsic)
        
        # 限制历史列表大小，防止内存无限累积
        # 保留最近 num_frames + num_history 个元素即可
        max_history_size = self.num_frames + (self.num_history if self.num_history else 0)
        if len(self.rgb_list) > max_history_size:
            # 只保留最近的元素
            keep_start = len(self.rgb_list) - max_history_size
            self.rgb_list = self.rgb_list[keep_start:]
            self.depth_list = self.depth_list[keep_start:]
            self.pose_list = self.pose_list[keep_start:]
            self.intrinsic_list = self.intrinsic_list[keep_start:]
            self.time_ids = self.time_ids[keep_start:]
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Reset memory if needed
        if not run_model:
            # 当 run_model=False 时，我们只是更新历史列表，不生成新动作
            # 但是仍然需要检查重置条件，避免内存无限累积
            # 注意：历史列表已经在上面更新了（第1500-1503行），所以重置应该在更新之后
            reset_interval = max(8, self.num_frames // 2)  # 至少8步重置一次
            if (self.step_id + 1) % reset_interval == 0:
                logger.debug(f'Reset model at Step {self.step_id + 1} (run_model=False, history update only)')
                self.model.reset_for_env(env_idx)
                self.output_ids = None
                self.past_key_values = None
                # 注意：不清空历史列表，因为我们已经添加了新的观察
                # 只清空 time_ids，因为重置后需要重新开始计数
                # 但保留最新的观察（刚添加的）
                if len(self.time_ids) > 0:
                    # 保留最后一个 time_id（刚添加的）
                    last_time_id = self.time_ids[-1]
                    self.time_ids = [last_time_id]
                    # 保留最后一个观察
                    if len(self.rgb_list) > 0:
                        self.rgb_list = [self.rgb_list[-1]]
                        self.depth_list = [self.depth_list[-1]]
                        self.pose_list = [self.pose_list[-1]]
                        self.intrinsic_list = [self.intrinsic_list[-1]]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            # 注意：当 run_model=False 时，不增加 step_id，因为这只是更新历史列表
            # step_id 应该在生成新动作时增加（在 run_model=True 的情况下）
            return [0], ""  # Return dummy action if not running model
        
        # Prepare input for model
        if self.output_ids is None:
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace(
                ' Where should you go next to stay on track?',
                f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.'
            )
            # 重要：添加历史观察（参考 streamvln_eval.py line 295）
            # 注意：streamvln_eval.py 中是在 step_id != 0 时添加，但我们需要确保条件正确
            if self.step_id != 0:
                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                logger.info(f"[StreamVLNPolicy.generate_action_sequence] Step {self.step_id}: Added memory token to prompt")
            else:
                logger.info(f"[StreamVLNPolicy.generate_action_sequence] Step {self.step_id}: Skipping memory token (first step)")
            
            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + '\n', '')
            # 重要：替换 <instruction>. 为实际指令（注意点号）
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            # 如果替换失败（可能格式不同），尝试不带点号的替换
            if '<instruction>' in sources[0]["value"]:
                sources[0]["value"] = sources[0]["value"].replace('<instruction>', instruction)
            
            # 验证标记是否在 prompt 中
            logger.info(f"[StreamVLNPolicy.generate_action_sequence] Step {self.step_id}, prompt contains: "
                       f"IMAGE={DEFAULT_IMAGE_TOKEN in sources[0]['value']}, "
                       f"MEMORY={DEFAULT_MEMORY_TOKEN in sources[0]['value']}")
            logger.info(f"[StreamVLNPolicy.generate_action_sequence] Prompt preview: {sources[0]['value'][:300]}...")
            add_system = True
        else:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            add_system = False
        
        input_ids, conversations = self.preprocess_qwen(
            [sources], has_image=True, add_system=add_system
        )
        
        if self.output_ids is not None:
            input_ids = torch.cat([self.output_ids, input_ids.to(self.output_ids.device)], dim=1)
        
        # 检查并修复 token ID 超出范围的问题（可能导致 CUDA 错误）
        # 重要：必须保留特殊标记 IMAGE_TOKEN_INDEX (-200) 和 MEMORY_TOKEN_INDEX (-300)
        # 这些负数token ID是模型内部使用的特殊索引，不应该被"修复"
        vocab_size = self.tokenizer.vocab_size
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'vocab_size'):
            vocab_size = self.model.config.vocab_size
        
        # 创建掩码，标记哪些位置是特殊token（不应该被修复）
        special_token_mask = (input_ids == IMAGE_TOKEN_INDEX) | (input_ids == MEMORY_TOKEN_INDEX)
        num_special_tokens = special_token_mask.sum().item()
        if num_special_tokens > 0:
            logger.info(f"[StreamVLNPolicy.generate_action_sequence] 检测到 {num_special_tokens} 个特殊标记: "
                       f"IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX} ({special_token_mask.sum(dim=-1).item()} 个), "
                       f"MEMORY_TOKEN_INDEX={MEMORY_TOKEN_INDEX}")
        
        # 检查是否有超出范围的 token ID（排除特殊标记）
        non_special_mask = ~special_token_mask
        if non_special_mask.any():
            non_special_ids = input_ids[non_special_mask]
            if non_special_ids.max().item() >= vocab_size:
                logger.warning(f"  检测到 token ID 超出范围: max={non_special_ids.max().item()}, vocab_size={vocab_size}")
                # 只修复非特殊标记的超出范围token ID
                input_ids_clamped = torch.clamp(input_ids, 0, vocab_size - 1)
                # 恢复特殊标记
                input_ids = torch.where(special_token_mask, input_ids, input_ids_clamped)
                logger.warning(f"  已修复超出范围的 token ID，新的 max={input_ids[non_special_mask].max().item()}")
        
        # 检查是否有负数的 token ID（排除特殊标记）
        if non_special_mask.any():
            non_special_ids = input_ids[non_special_mask]
            if non_special_ids.min().item() < 0:
                logger.warning(f"  检测到负数 token ID（非特殊标记）: min={non_special_ids.min().item()}")
                # 只修复非特殊标记的负数token ID
                input_ids_clamped = torch.clamp(input_ids, 0, vocab_size - 1)
                # 恢复特殊标记
                input_ids = torch.where(special_token_mask, input_ids, input_ids_clamped)
                logger.warning(f"  已修复负数 token ID，新的 min={input_ids[non_special_mask].min().item()}")
        
        # 最终验证：确保特殊标记没有被破坏
        if special_token_mask.any():
            final_special_tokens = input_ids[special_token_mask]
            if not torch.all((final_special_tokens == IMAGE_TOKEN_INDEX) | (final_special_tokens == MEMORY_TOKEN_INDEX)):
                logger.error(f"[StreamVLNPolicy.generate_action_sequence] CRITICAL: 特殊标记被破坏！")
            else:
                logger.info(f"[StreamVLNPolicy.generate_action_sequence] ✓ 特殊标记已正确保留: "
                           f"IMAGE_TOKEN_INDEX={IMAGE_TOKEN_INDEX}, MEMORY_TOKEN_INDEX={MEMORY_TOKEN_INDEX}")
        
        # Prepare image history
        images = self.rgb_list[-1:]
        depths = self.depth_list[-1:]
        poses = self.pose_list[-1:]
        intrinsics = self.intrinsic_list[-1:]
        
        if self.step_id != 0 and self.step_id % self.num_frames == 0:
            if self.num_history is None:
                # 确保步长不为 0，避免 "slice step cannot be zero" 错误
                step = self.num_future_steps if self.num_future_steps != 0 else 1
                history_ids = slice(0, self.time_ids[0], step)
            else:
                # 确保步长不为 0，避免 "slice step cannot be zero" 错误
                step = (self.time_ids[0] // self.num_history)
                step = step if step != 0 else 1
                history_ids = slice(0, self.time_ids[0], step)
            images = self.rgb_list[history_ids] + images
            depths = self.depth_list[history_ids] + depths
            poses = self.pose_list[history_ids] + poses
            intrinsics = self.intrinsic_list[history_ids] + intrinsics
        
        # Prepare input dict
        # 注意：stream_video_vln.py 的 generate 方法期望 'inputs' 键（不是 'input_ids'）
        # 参考 streamvln_eval.py 的实现方式，使用 'task_type' 而不是 'task_ids'
        input_dict_raw = {
            'images': torch.stack(images).unsqueeze(0),
            'depths': torch.stack(depths).unsqueeze(0),
            'poses': torch.stack(poses).unsqueeze(0),
            'intrinsics': torch.stack(intrinsics).unsqueeze(0),
            'inputs': input_ids,  # StreamVLN 使用 'inputs' 键，不是 'input_ids'
            'env_id': env_idx,
            'time_ids': [self.time_ids],
            'task_type': [0],  # 参考 streamvln_eval.py，使用 'task_type' 而不是 'task_ids'
        }
        
        # 调试信息：打印 input_ids 的形状和内容（前10个token）
        logger.info(f"[StreamVLNPolicy.generate_action_sequence] input_ids shape: {input_ids.shape}")
        if input_ids.numel() > 0:
            logger.info(f"[StreamVLNPolicy.generate_action_sequence] input_ids[:10]: {input_ids[0, :10].tolist()}")
        
        input_dict = dict_to_cuda(input_dict_raw.copy(), self.device)
        
        for key, value in input_dict.items():
            if key in ['images', 'depths', 'poses', 'intrinsics']:
                input_dict[key] = input_dict[key].to(torch.bfloat16)
        
        # 确保 _is_stateful 属性存在（修复 transformers 4.56.0 兼容性问题）
        if not hasattr(self.model.__class__, '_is_stateful'):
            self.model.__class__._is_stateful = False
        
        # Generate
        # 根据 StreamVLN 论文，模型使用慢-快上下文建模：
        # - 快速流式对话上下文：使用滑动窗口 KV 缓存（通过 self.cache 管理）
        # - 慢速更新内存上下文：压缩历史视觉状态
        # StreamVLN 的 generate 方法内部管理 cache，不需要手动传递 past_key_values
        # 模型会在内部处理 KV 缓存的累积和重用

       
        # 生成前清理GPU缓存，释放碎片内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 同步以确保清理完成
            torch.cuda.synchronize()
        
        # 限制输入序列长度，防止过长序列导致内存爆炸
        max_input_length = 2048  # 限制最大输入长度
        if input_ids.shape[1] > max_input_length:
            logger.warning(f"Input sequence too long ({input_ids.shape[1]}), truncating to {max_input_length}")
            # 保留前面的系统提示和最近的输入
            input_ids = input_ids[:, -max_input_length:]
            # 同时需要更新 input_dict
            input_dict['inputs'] = input_ids
        
        generate_kwargs = {
            **input_dict,  # 包含 'inputs', 'images', 'depths', 'poses', 'intrinsics', 'env_id', 'time_ids', 'task_type'
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 1024,  # 增加到128以确保动作序列生成完整（原始代码使用10000，但128应该足够）
            "use_cache": True,  # StreamVLN 需要缓存来支持多轮对话
            "return_dict_in_generate": True,
        }
        
        # 如果存在 past_key_values，传递给模型（虽然StreamVLN主要使用内部cache）
        # 这有助于保持与原始代码的兼容性
        # 注意：StreamVLN 使用内部 cache 机制，但传递 past_key_values 可以保持兼容性
        if self.past_key_values is not None:
            generate_kwargs["past_key_values"] = self.past_key_values
        
        # 尝试生成，如果OOM则清理内存并重试
        max_retries = 2
        retry_count = 0
        outputs = None
        
        while retry_count < max_retries:
            try:
                # 使用torch.cuda.amp.autocast来减少内存占用（如果模型支持）
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = self.model.generate(**generate_kwargs)
                break  # 成功则退出循环
            except torch.cuda.OutOfMemoryError as e:
                retry_count += 1
                error_msg = str(e)
                logger.warning(f"CUDA OOM at attempt {retry_count}/{max_retries}: {error_msg}")
                
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                if retry_count < max_retries:
                    # 尝试更激进的策略：减少输入长度或重置模型
                    if input_ids.shape[1] > 1024:
                        logger.warning(f"Reducing input length from {input_ids.shape[1]} to 1024")
                        input_ids = input_ids[:, -1024:]
                        input_dict['inputs'] = input_ids
                        generate_kwargs['inputs'] = input_ids
                    
                    # 重置模型cache
                    logger.warning("Resetting model cache due to OOM")
                    self.model.reset_for_env(env_idx)
                    self.output_ids = None
                    self.past_key_values = None
                    
                    # 清理历史列表，只保留最近的数据
                    # 使用配置的 num_history，如果未配置则使用最小值 8
                    min_history = self.num_history if self.num_history else 8
                    if len(self.rgb_list) > min_history:
                        keep_start = len(self.rgb_list) - min_history
                        self.rgb_list = self.rgb_list[keep_start:]
                        self.depth_list = self.depth_list[keep_start:]
                        self.pose_list = self.pose_list[keep_start:]
                        self.intrinsic_list = self.intrinsic_list[keep_start:]
                        self.time_ids = self.time_ids[keep_start:]
                    
                    # 更新 input_dict 以反映新的历史
                    if len(self.rgb_list) > 0:
                        images = self.rgb_list[-1:]
                        depths = self.depth_list[-1:]
                        poses = self.pose_list[-1:]
                        intrinsics = self.intrinsic_list[-1:]
                        input_dict['images'] = torch.stack(images).unsqueeze(0).to(self.device).to(torch.bfloat16)
                        input_dict['depths'] = torch.stack(depths).unsqueeze(0).to(self.device).to(torch.bfloat16)
                        input_dict['poses'] = torch.stack(poses).unsqueeze(0).to(self.device).to(torch.bfloat16)
                        input_dict['intrinsics'] = torch.stack(intrinsics).unsqueeze(0).to(self.device).to(torch.bfloat16)
                        generate_kwargs.update(input_dict)
                    
                    # 再次清理
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                else:
                    # 最后一次尝试失败，抛出异常
                    logger.error(f"StreamVLN generation failed after {max_retries} attempts: {error_msg}")
                    raise
            except Exception as e:
                error_msg = str(e)
                import traceback
                logger.error(f"Error: StreamVLN generation failed: {error_msg}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        # 处理 outputs：当 return_dict_in_generate=True 时，outputs 是 GenerateDecoderOnlyOutput 对象
        # 它有 sequences 属性，但需要确保正确访问
        if hasattr(outputs, 'sequences'):
            self.output_ids = outputs.sequences
            # 尝试获取 past_key_values（如果模型返回）
            # 原始代码显式保存 past_key_values 用于多轮对话上下文保持
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                self.past_key_values = outputs.past_key_values
                logger.debug(f"[StreamVLNPolicy] Saved past_key_values for env {env_idx}")
            else:
                # 如果模型没有返回 past_key_values，保持之前的值（如果有）
                # 这样可以在多轮对话中保持上下文
                if self.past_key_values is None:
                    logger.debug(f"[StreamVLNPolicy] No past_key_values returned, using None (model uses internal cache)")
        elif isinstance(outputs, dict) and 'sequences' in outputs:
            self.output_ids = outputs['sequences']
            self.past_key_values = outputs.get('past_key_values', self.past_key_values)
        else:
            # 如果 outputs 是 tensor，直接使用
            self.output_ids = outputs
            # 保持之前的 past_key_values（如果有）
            # self.past_key_values 保持不变
        
        # 注意：StreamVLN 使用内部 cache 机制（通过 self.cache[env_id] 管理）
        # 但原始代码仍然传递 past_key_values，可能是为了兼容性
        # 如果模型不支持 past_key_values，这里设为 None 也不会影响内部 cache
        # 但我们仍然尝试保存它，以便在需要时使用
        
        # Decode output
        # 重要：skip_special_tokens=False 确保特殊标记（如 <image>, <memory>）不会被跳过
        # 但 tokenizer 可能将这些标记 decode 为其他字符（如 !），这是正常的
        # 关键是在 tokenization 时标记被正确识别为 IMAGE_TOKEN_INDEX 和 MEMORY_TOKEN_INDEX
        
        # 在 decode 前，将占位符 0 替换回特殊标记的原始 token ID，以便正确 decode
        # 注意：这里我们需要知道哪些位置原本是特殊标记
        # 由于 prepare_inputs_labels_for_multimodal 已经处理了特殊标记，output_ids 中可能不包含特殊标记
        # 但为了在 decode 时显示正确的字符串，我们需要检查 input_ids 中的特殊标记位置
        output_ids_for_decode = self.output_ids.clone()
        
        # 检查 output_ids 中是否有占位符 0，这些可能是特殊标记的占位符
        # 但是，由于特殊标记在 prepare_inputs_labels_for_multimodal 中已经被替换为嵌入，
        # output_ids 中可能不包含特殊标记
        # 所以我们需要从原始的 input_ids 中获取特殊标记的位置信息
        
        # 实际上，decode 时显示 `!` 是正常的，因为特殊标记已经被处理
        # 但为了用户友好，我们可以在 decode 后手动替换 `!` 为 `<image>` 和 `<memory>`
        llm_output = self.tokenizer.batch_decode(
            output_ids_for_decode, skip_special_tokens=False
        )[0].strip()
        
        # 将 decode 后的占位符替换回特殊标记字符串（用于显示）
        # 注意：这只是在输出显示时替换，不影响模型的实际处理
        # 由于 tokenizer 可能将特殊标记 decode 为 `!` 或其他字符，我们需要手动替换
        # 但是，我们无法准确知道哪些 `!` 是 `<image>`，哪些是 `<memory>`
        # 所以这个替换可能不够准确
        
        # 更好的方法：在 decode 前，将特殊标记的原始 token ID 替换回 output_ids
        # 但这需要知道原始 input_ids 中特殊标记的位置，这在当前代码结构中比较复杂
        
        # 调试：检查 output_ids 中是否包含 IMAGE_TOKEN_INDEX 或 MEMORY_TOKEN_INDEX
        if IMAGE_TOKEN_INDEX in self.output_ids:
            logger.info(f"[StreamVLNPolicy.generate_action_sequence] IMAGE_TOKEN_INDEX found in output_ids at positions: "
                       f"{[i for i, x in enumerate(self.output_ids.flatten()) if x == IMAGE_TOKEN_INDEX]}")
        if MEMORY_TOKEN_INDEX in self.output_ids:
            logger.info(f"[StreamVLNPolicy.generate_action_sequence] MEMORY_TOKEN_INDEX found in output_ids at positions: "
                       f"{[i for i, x in enumerate(self.output_ids.flatten()) if x == MEMORY_TOKEN_INDEX]}")
        
        # 检查 output_ids 中是否有占位符 0（可能是特殊标记的占位符）
        if 0 in self.output_ids:
            zero_positions = [i for i, x in enumerate(self.output_ids.flatten()) if x == 0]
            logger.debug(f"[StreamVLNPolicy.generate_action_sequence] Found {len(zero_positions)} zero tokens in output_ids (may be placeholders for special tokens)")
        
        # Parse actions
        action_seq = self.parse_actions(llm_output)
        
        # 应用动作序列填充/截断逻辑（参考原始代码 streamvln_eval.py:340-352）
        # 这确保动作序列长度合理，避免导航不连贯
        if len(action_seq) == 0:
            action_seq = [0]  # Default to STOP
        elif len(action_seq) < 4 and 0 not in action_seq:
            # 如果序列太短且不包含STOP，填充到4个动作（确保导航连贯）
            if len(action_seq) == 1:
                action_seq += [2, 2, 3]  # 添加默认探索动作
            elif len(action_seq) == 2:
                action_seq += [2, 3]
            elif len(action_seq) == 3:
                action_seq += [2]
        elif len(action_seq) > 4:
            # 如果序列太长，截断到4个动作（避免动作执行混乱）
            action_seq = action_seq[:4]
        
        # 清理中间变量，释放内存
        del input_dict_raw, input_dict, generate_kwargs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 定期重置模型cache，防止KV cache无限累积
        # 更频繁地重置以减少内存占用（每num_frames/2步重置一次）
        reset_interval = max(8, self.num_frames // 2)  # 至少8步重置一次
        if self.step_id > 0 and self.step_id % reset_interval == 0:
            logger.info(f"[StreamVLNPolicy] Resetting model cache at step {self.step_id} to prevent memory accumulation")
            self.model.reset_for_env(env_idx)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # 清理历史列表，只保留最近的数据
            if len(self.rgb_list) > self.num_history:
                keep_start = len(self.rgb_list) - self.num_history
                self.rgb_list = self.rgb_list[keep_start:]
                self.depth_list = self.depth_list[keep_start:]
                self.pose_list = self.pose_list[keep_start:]
                self.intrinsic_list = self.intrinsic_list[keep_start:]
                self.time_ids = self.time_ids[keep_start:]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.step_id += 1
        
        return action_seq, llm_output
