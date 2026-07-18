#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
NaVILA Policy for Falcon Framework (Habitat3)
基于视觉-语言模型的导航策略
"""

import copy
import logging
import os
import sys
import textwrap
from collections import OrderedDict
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from PIL import Image
from torch import nn as nn

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Net, NetPolicy

# NaVILA相关导入
from habitat_baselines.rl.ddppo.policy.navila.action_parser import NaVILAActionParser

NAVILA_AVAILABLE = False
NAVILA_IMPORT_ERROR: Optional[ImportError] = None


def _register_llava_in_sys_modules():
    """手动将 navila/llava 注册到 sys.modules，使 'from llava.xxx' 的 import 能正常工作。"""
    navila_root = Path(__file__).resolve().parent
    llava_pkg_root = navila_root / "navila" / "llava"
    if not llava_pkg_root.exists():
        llava_pkg_root = navila_root / "llava"
    if not llava_pkg_root.exists():
        return

    if "llava" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "llava", str(llava_pkg_root / "__init__.py"),
            submodule_search_locations=[str(llava_pkg_root)]
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            mod.__path__ = [str(llava_pkg_root)]
            mod.__package__ = "llava"
            sys.modules["llava"] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass
    else:
        mod = sys.modules["llava"]
        if not getattr(mod, "__path__", None):
            mod.__path__ = [str(llava_pkg_root)]
        elif str(llava_pkg_root) not in mod.__path__:
            mod.__path__.insert(0, str(llava_pkg_root))


def _ensure_transformers_compatibility():
    import importlib
    import types

    optional_symbols = {}

    try:
        utils_module = importlib.import_module("transformers.utils")
    except ImportError:
        return

    if not hasattr(utils_module, "is_torch_tpu_available"):

        def _is_torch_tpu_available() -> bool:
            return False

        utils_module.is_torch_tpu_available = _is_torch_tpu_available  # type: ignore[attr-defined]
        optional_symbols["is_torch_tpu_available"] = _is_torch_tpu_available

    try:
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    except ImportError as exc:
        if "is_torch_tpu_available" in str(exc):
            import sys

            if "transformers.modeling_utils" in sys.modules:
                del sys.modules["transformers.modeling_utils"]
            modeling_utils = importlib.import_module("transformers.modeling_utils")
        else:
            return

    if not hasattr(modeling_utils, "PreTrainedAudioTokenizerBase"):

        class PreTrainedAudioTokenizerBase:  # type: ignore[too-many-ancestors]
            pass

        modeling_utils.PreTrainedAudioTokenizerBase = PreTrainedAudioTokenizerBase  # type: ignore[attr-defined]
        optional_symbols["PreTrainedAudioTokenizerBase"] = PreTrainedAudioTokenizerBase

    if not hasattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS"):
        modeling_utils.ALL_ATTENTION_FUNCTIONS = {}  # type: ignore[attr-defined]
        optional_symbols["ALL_ATTENTION_FUNCTIONS"] = {}

    return optional_symbols


def _maybe_extend_sys_path():
    navila_root = Path(__file__).resolve().parent
    candidates = [
        navila_root,
        navila_root / "..",
        navila_root / "../../../navila",
    ]
    for path in candidates:
        resolved = path.resolve()
        if resolved.exists() and str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))

    # llava 目录在 navila/llava/，而不是 policy/llava/
    llava_dir = navila_root / "navila" / "llava"
    # 如果上面的路径不存在，尝试其他可能的位置
    if not llava_dir.exists():
        llava_dir = navila_root / "llava"
    if llava_dir.exists() and str(llava_dir) not in sys.path:
        sys.path.insert(0, str(llava_dir))

    if (llava_dir / "__init__.py").exists():
        spec = importlib.util.spec_from_file_location(
            "llava", llava_dir / "__init__.py", submodule_search_locations=[str(llava_dir)]
        )
        if spec and spec.loader:
            # 如果 llava 已经存在，更新它的 __path__；否则创建新模块
            if "llava" in sys.modules:
                module = sys.modules["llava"]
                # 确保 __path__ 包含 llava 目录
                if not hasattr(module, '__path__') or not module.__path__:
                    module.__path__ = [str(llava_dir)]  # type: ignore[attr-defined]
                elif str(llava_dir) not in module.__path__:
                    module.__path__.insert(0, str(llava_dir))  # type: ignore[attr-defined]
            else:
                module = importlib.util.module_from_spec(spec)
                # 确保 __path__ 包含 llava 目录，这样 Python 才能找到子包
                if not hasattr(module, '__path__') or not module.__path__:
                    module.__path__ = [str(llava_dir)]  # type: ignore[attr-defined]
                elif str(llava_dir) not in module.__path__:
                    module.__path__.insert(0, str(llava_dir))  # type: ignore[attr-defined]
                sys.modules["llava"] = module
            # 执行模块初始化
            spec.loader.exec_module(module)
            
            # 确保 llava.utils 也被注册为子包
            llava_utils_dir = llava_dir / "utils"
            if llava_utils_dir.exists() and (llava_utils_dir / "__init__.py").exists():
                if "llava.utils" not in sys.modules:
                    utils_spec = importlib.util.spec_from_file_location(
                        "llava.utils", llava_utils_dir / "__init__.py",
                        submodule_search_locations=[str(llava_utils_dir)]
                    )
                    if utils_spec and utils_spec.loader:
                        utils_module = importlib.util.module_from_spec(utils_spec)
                        utils_module.__path__ = [str(llava_utils_dir)]  # type: ignore[attr-defined]
                        sys.modules["llava.utils"] = utils_module
                        # 先执行 __init__.py，确保 logging 被导入
                        try:
                            utils_spec.loader.exec_module(utils_module)
                        except Exception:
                            pass
                
                # 确保 logging 模块也被注册（无论 llava.utils 是否已存在）
                if "llava.utils.logging" not in sys.modules:
                    logging_file = llava_utils_dir / "logging.py"
                    if logging_file.exists():
                        logging_spec = importlib.util.spec_from_file_location(
                            "llava.utils.logging", logging_file
                        )
                        if logging_spec and logging_spec.loader:
                            logging_module = importlib.util.module_from_spec(logging_spec)
                            sys.modules["llava.utils.logging"] = logging_module
                            try:
                                logging_spec.loader.exec_module(logging_module)
                            except Exception:
                                pass


_maybe_extend_sys_path()

# 应用 transformers 兼容性补丁（在导入 transformers 之前）
# 补丁模块在导入时会自动执行
try:
    from habitat_baselines.rl.ddppo.policy.navila import transformers_compat_patch  # noqa: F401
except ImportError:
    # 如果补丁模块不存在，尝试直接应用补丁
    pass

# 确保 transformers 完全初始化，避免延迟导入问题
try:
    import transformers
    # 预加载 Mistral 和 Mixtral 相关类，确保它们可以被导入
    # 某些 transformers 版本中这些类在子模块中
    try:
        from transformers.models.mistral import MistralConfig, MistralModel, MistralForCausalLM
        # 手动添加到顶层（如果不存在）
        if not hasattr(transformers, 'MistralConfig'):
            transformers.MistralConfig = MistralConfig
        if not hasattr(transformers, 'MistralModel'):
            transformers.MistralModel = MistralModel
        if not hasattr(transformers, 'MistralForCausalLM'):
            transformers.MistralForCausalLM = MistralForCausalLM
    except ImportError:
        pass
    try:
        from transformers.models.mixtral import MixtralConfig, MixtralModel, MixtralForCausalLM
        # 手动添加到顶层（如果不存在）
        if not hasattr(transformers, 'MixtralConfig'):
            transformers.MixtralConfig = MixtralConfig
        if not hasattr(transformers, 'MixtralModel'):
            transformers.MixtralModel = MixtralModel
        if not hasattr(transformers, 'MixtralForCausalLM'):
            transformers.MixtralForCausalLM = MixtralForCausalLM
    except ImportError:
        pass
    # 同样处理 Llama 类
    try:
        from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
        from transformers.models.llama.configuration_llama import LlamaConfig
        if not hasattr(transformers, 'LlamaModel'):
            transformers.LlamaModel = LlamaModel
        if not hasattr(transformers, 'LlamaForCausalLM'):
            transformers.LlamaForCausalLM = LlamaForCausalLM
        if not hasattr(transformers, 'LlamaConfig'):
            transformers.LlamaConfig = LlamaConfig
    except ImportError:
        pass
except (ImportError, AttributeError):
    pass

try:
    _ensure_transformers_compatibility()
    # 在导入任何 llava 子模块之前，确保 llava.utils.logging 可以被导入
    # 这是因为 conversation.py 等模块在导入时会立即执行 from llava.utils.logging import logger
    navila_root = Path(__file__).resolve().parent
    # llava 目录在 navila/llava/，而不是 policy/llava/
    llava_dir = navila_root / "navila" / "llava"
    # 如果上面的路径不存在，尝试其他可能的位置
    if not llava_dir.exists():
        llava_dir = navila_root / "llava"
    llava_utils_dir = llava_dir / "utils"
    
    # 强制确保 llava 模块存在（即使 _maybe_extend_sys_path 已经执行过）
    if llava_dir.exists() and (llava_dir / "__init__.py").exists():
        if "llava" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "llava", llava_dir / "__init__.py", submodule_search_locations=[str(llava_dir)]
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                module.__path__ = [str(llava_dir)]  # type: ignore[attr-defined]
                sys.modules["llava"] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    pass
        else:
            # 如果 llava 已经存在，确保它的 __path__ 正确
            module = sys.modules["llava"]
            if not hasattr(module, '__path__') or not module.__path__:
                module.__path__ = [str(llava_dir)]  # type: ignore[attr-defined]
            elif str(llava_dir) not in module.__path__:
                module.__path__.insert(0, str(llava_dir))  # type: ignore[attr-defined]
    
    # 强制确保 llava.utils 被注册为包
    if llava_utils_dir.exists() and (llava_utils_dir / "__init__.py").exists():
        if "llava.utils" not in sys.modules:
            utils_spec = importlib.util.spec_from_file_location(
                "llava.utils", llava_utils_dir / "__init__.py",
                submodule_search_locations=[str(llava_utils_dir)]
            )
            if utils_spec and utils_spec.loader:
                utils_module = importlib.util.module_from_spec(utils_spec)
                utils_module.__path__ = [str(llava_utils_dir)]  # type: ignore[attr-defined]
                sys.modules["llava.utils"] = utils_module
                try:
                    utils_spec.loader.exec_module(utils_module)
                except Exception:
                    pass
    
    # 强制确保 llava.utils.logging 被注册
    if "llava.utils.logging" not in sys.modules:
        logging_file = llava_utils_dir / "logging.py"
        if logging_file.exists():
            logging_spec = importlib.util.spec_from_file_location(
                "llava.utils.logging", logging_file
            )
            if logging_spec and logging_spec.loader:
                logging_module = importlib.util.module_from_spec(logging_spec)
                sys.modules["llava.utils.logging"] = logging_module
                try:
                    logging_spec.loader.exec_module(logging_module)
                except Exception:
                    pass
    
    # 验证 llava.utils.logging 可以被导入（在导入 conversation 之前）
    try:
        from llava.utils.logging import logger as _test_logger  # type: ignore[import-untyped]
        _ = _test_logger  # 确保导入成功
    except (ImportError, ModuleNotFoundError) as e:
        # 如果还是失败，抛出更详细的错误
        raise ImportError(
            f"Failed to import llava.utils.logging before importing conversation. "
            f"llava in sys.modules: {'llava' in sys.modules}, "
            f"llava.utils in sys.modules: {'llava.utils' in sys.modules}, "
            f"llava.utils.logging in sys.modules: {'llava.utils.logging' in sys.modules}, "
            f"llava_dir exists: {llava_dir.exists()}, "
            f"llava_utils_dir exists: {llava_utils_dir.exists()}, "
            f"logging.py exists: {(llava_utils_dir / 'logging.py').exists() if llava_utils_dir.exists() else False}, "
            f"Error: {e}"
        ) from e
    
    # 现在可以安全地导入其他 llava 模块
    from habitat_baselines.rl.ddppo.policy.navila.llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.conversation import (
        SeparatorStyle,
        conv_templates,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from habitat_baselines.rl.ddppo.policy.navila.llava.model.builder import (
        load_pretrained_model,
    )
    NAVILA_AVAILABLE = True
except ImportError as primary_error:
    NAVILA_IMPORT_ERROR = primary_error
    try:
        # 再次确保 transformers 已初始化
        import transformers
        try:
            from transformers.models.mistral import MistralConfig, MistralModel, MistralForCausalLM
        except ImportError:
            pass
        try:
            from transformers.models.mixtral import MixtralConfig, MixtralModel, MixtralForCausalLM
        except ImportError:
            pass
        _ensure_transformers_compatibility()
        from navila.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from navila.llava.conversation import SeparatorStyle, conv_templates
        from navila.llava.mm_utils import (
            KeywordsStoppingCriteria,
            process_images,
            tokenizer_image_token,
        )
        from navila.llava.model.builder import load_pretrained_model
        NAVILA_AVAILABLE = True
    except ImportError as secondary_error:
        NAVILA_IMPORT_ERROR = secondary_error
        print(
            "Warning: NaVILA modules not available. 请确认按照 `navila/environment_setup.sh` 成功安装，"
            "或者将NaVILA代码目录加入 PYTHONPATH。详细错误："
            f" {primary_error}"
            f" {secondary_error}"
        )

if TYPE_CHECKING:
    from omegaconf import DictConfig


logger = logging.getLogger(__name__)
ACTION_ID_TO_NAME = {
    0: "stop",
    1: "move forward",
    2: "turn left",
    3: "turn right",
}
DEFAULT_INSTRUCTION_KEYS = [
    "agent_0_falcon_instruction",
    "falcon_instruction",
    "instruction",
    "instruction_sensor",
]
DEFAULT_GT_ACTION_KEYS = [
    "agent_0_falcon_gt_action",
    "falcon_gt_action",
    "gt_action",
    "expert_actions",
]
MAX_GT_ACTIONS_IN_PROMPT = 50


def _augment_rgb_sensor_keys(keys: Optional[List[str]]) -> Optional[List[str]]:
    if not keys:
        return None
    collected: List[str] = []
    for key in keys:
        if not isinstance(key, str):
            continue
        collected.append(key)
        if key.startswith("agent_0_"):
            collected.append(key[len("agent_0_") :])
    unique_keys: List[str] = []
    seen: set[str] = set()
    for key in collected:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys or None


def _tensor_like_to_numpy(data: Any) -> Optional[np.ndarray]:
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


def _decode_text_instruction(data: Any) -> Optional[str]:
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
    arr = _tensor_like_to_numpy(data)
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


def _format_gt_action_sequence(data: Any) -> Optional[str]:
    arr = _tensor_like_to_numpy(data)
    if arr is None:
        return None
    arr = np.asarray(arr).astype(np.int32).flatten().tolist()
    if not arr:
        return None

    actions: List[int] = []
    stop_seen = False
    for val in arr:
        if val < 0:
            break
        if val == 0:
            if stop_seen:
                break
            stop_seen = True
            actions.append(val)
        else:
            actions.append(val)
    if not actions:
        return None

    truncated = False
    if len(actions) > MAX_GT_ACTIONS_IN_PROMPT:
        actions = actions[:MAX_GT_ACTIONS_IN_PROMPT]
        truncated = True

    action_names = [ACTION_ID_TO_NAME.get(a, f"action_{a}") for a in actions]
    if not action_names:
        return None
    instruction = "Follow the ground-truth action sequence: " + ", ".join(action_names)
    if truncated:
        instruction += ", ..."
    return instruction


def extract_navila_instruction(
    observations: Dict[str, Any],
    instruction_sensor_uuid: Optional[str] = None,
    episode_instruction: Optional[str] = None,
) -> str:
    if episode_instruction and isinstance(episode_instruction, str) and episode_instruction.strip():
        return episode_instruction.strip()

    keys_to_try: List[str] = []
    if instruction_sensor_uuid:
        keys_to_try.append(instruction_sensor_uuid)
    keys_to_try.extend(DEFAULT_INSTRUCTION_KEYS)

    for key in keys_to_try:
        if key and key in observations:
            text = _decode_text_instruction(observations[key])
            # print("******************************")  # silenced: per-step verbose
            # print("DEBUG: text:", text)  # silenced: per-step verbose
            if text:
                return text

    # for key in DEFAULT_GT_ACTION_KEYS:
    #     if key in observations:
    #         text = _format_gt_action_sequence(observations[key])
    #         if text:
    #             return text

    return "Navigate to the goal location"


def sample_and_pad_images(images, num_frames=8, width=512, height=512):
    """
    采样和填充图像序列到固定帧数
    
    Args:
        images: 图像列表
        num_frames: 目标帧数
        width: 图像宽度
        height: 图像高度
        
    Returns:
        采样后的图像列表
    """
    frames = copy.deepcopy(images)
    
    # 如果帧数不足，用黑色图像填充
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
    
    # 采样：均匀采样历史帧 + 最新帧
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    
    return sampled_frames


@baseline_registry.register_policy
class NaVILAPolicy(NetPolicy):
    """
    基于视觉-语言模型(LLAVA)的导航策略
    
    该策略使用NaVILA模型生成语言指令，然后通过动作解析器转换为离散动作。
    适配Habitat3 Falcon框架，支持动态行人环境。
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        model_path: Optional[str] = None,
        num_video_frames: int = 8,
        forward_step: int = 25,
        turn_step: int = 15,
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        instruction_sensor_uuid: str = "instruction",
        rgb_sensor_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        初始化NaVILA策略
        
        Args:
            observation_space: 观察空间
            action_space: 动作空间
            hidden_size: 隐藏层大小（用于兼容性，实际不使用）
            model_path: NaVILA预训练模型路径
            num_video_frames: 输入视频帧数
            forward_step: 前进步长（cm）
            turn_step: 转向步长（度）
            policy_config: 策略配置
            aux_loss_config: 辅助损失配置
            instruction_sensor_uuid: 指令传感器UUID
        """
        if not NAVILA_AVAILABLE:
            detail = (
                f" 原始异常: {NAVILA_IMPORT_ERROR}"
                if NAVILA_IMPORT_ERROR is not None
                else ""
            )
            raise ImportError(
                "NaVILA modules are not available. 请确认已执行 `navila/environment_setup.sh` "
                "并且 Python 能够导入 `navila.llava`。"
                f"{detail}"
            ) from NAVILA_IMPORT_ERROR
        
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
        
        # 创建NaVILA网络
        net = NaVILANet(
            observation_space=observation_space,
            action_space=action_space,
            model_path=model_path,
            num_video_frames=num_video_frames,
            forward_step=forward_step,
            turn_step=turn_step,
            instruction_sensor_uuid=instruction_sensor_uuid,
            rgb_sensor_keys=rgb_sensor_keys,
        )

        # Multi-action sequence mode (from policy config)
        if policy_config is not None:
            net.action_sequence_mode = getattr(policy_config, 'action_sequence_mode', False)
            net.action_sequence_length = getattr(policy_config, 'action_sequence_length', 4)

        super().__init__(
            net,
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
        """从配置创建策略"""
        # 排除用于渲染的相机
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
        
        # 获取 agent_name：优先从 kwargs 获取，然后从配置获取，最后使用默认值
        agent_name = kwargs.get("agent_name", None)
        
        if agent_name is None:
            # 尝试从 config.habitat.simulator.agents_order 获取
            try:
                if hasattr(config.habitat, "simulator") and hasattr(config.habitat.simulator, "agents_order"):
                    agents_order = config.habitat.simulator.agents_order
                    if len(agents_order) > 1:
                        raise ValueError(
                            "If there is more than an agent, you need to specify the agent name"
                        )
                    elif len(agents_order) == 1:
                        agent_name = agents_order[0]
            except (AttributeError, KeyError):
                pass
        
        # 如果还是 None，尝试从 policy 配置中推断
        if agent_name is None:
            try:
                policy_config = config.habitat_baselines.rl.policy
                # 查找包含 NaVILAPolicy 的 agent
                for key in policy_config.keys():
                    if hasattr(policy_config[key], "name") and policy_config[key].name == "NaVILAPolicy":
                        agent_name = key
                        break
                # 如果没有找到，使用第一个 policy 配置
                if agent_name is None and len(policy_config) > 0:
                    agent_name = list(policy_config.keys())[0]
            except (AttributeError, KeyError):
                pass
        
        # 如果还是 None，使用默认值 "agent_0"
        if agent_name is None:
            agent_name = "agent_0"
            logger.warning(
                f"Could not determine agent_name from config, using default 'agent_0'. "
                f"Please ensure 'agent_name' is provided in kwargs or config.habitat.simulator.agents_order is set."
            )
        
        # 从配置中获取NaVILA特定参数
        try:
            navila_config = config.habitat_baselines.rl.policy[agent_name]
        except (KeyError, AttributeError) as e:
            raise ValueError(
                f"Could not find policy config for agent '{agent_name}'. "
                f"Available agents in policy config: {list(config.habitat_baselines.rl.policy.keys()) if hasattr(config.habitat_baselines.rl, 'policy') else 'N/A'}"
            ) from e
        
        # 获取 model_path：优先从 policy 配置中获取，如果没有则从 eval_ckpt_path_dir 获取
        model_path = navila_config.get("model_path", None)
        if model_path is None:
            # 尝试从 eval_ckpt_path_dir 获取
            try:
                eval_ckpt_path_dir = config.habitat_baselines.get("eval_ckpt_path_dir", None)
                if eval_ckpt_path_dir:
                    model_path = eval_ckpt_path_dir
                    logger.info(f"Using eval_ckpt_path_dir as model_path: {model_path}")
            except (AttributeError, KeyError):
                pass
        
        obs_rgb_candidates = [
            key
            for key in filtered_obs.spaces.keys()
            if isinstance(key, str) and "rgb" in key.lower()
        ]
        obs_rgb_candidates = _augment_rgb_sensor_keys(obs_rgb_candidates)

        rgb_sensor_keys = None
        try:
            gym_cfg = config.habitat.gym
            obs_keys_cfg = getattr(gym_cfg, "obs_keys", None)
            if obs_keys_cfg:
                extracted_keys = [
                    key for key in obs_keys_cfg if isinstance(key, str) and "rgb" in key.lower()
                ]
                rgb_sensor_keys = _augment_rgb_sensor_keys(extracted_keys)
                if rgb_sensor_keys:
                    print(
                        f"[NaVILA] 读取 habitat.gym.obs_keys 得到 RGB 传感器: {rgb_sensor_keys}"
                    )
        except AttributeError:
            pass

        if not rgb_sensor_keys:
            rgb_sensor_keys = obs_rgb_candidates
            if rgb_sensor_keys:
                print(
                    f"[NaVILA] 未指定 rgb_sensor_keys，使用 observation_space 中的: {rgb_sensor_keys}"
                )
            else:
                print("[NaVILA] 未在配置或 observation_space 中找到任何 RGB 传感器。")

        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            model_path=model_path,
            num_video_frames=navila_config.get("num_video_frames", 8),
            forward_step=navila_config.get("forward_step", 25),
            turn_step=navila_config.get("turn_step", 15),
            policy_config=config.habitat_baselines.rl.policy[agent_name],
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            instruction_sensor_uuid=config.habitat_baselines.rl.policy[agent_name].get(
                "instruction_sensor_uuid", "instruction"
            ),
            rgb_sensor_keys=_augment_rgb_sensor_keys(
                navila_config.get("rgb_sensor_keys", rgb_sensor_keys)
            ),
        )


class NaVILANet(Net):
    """
    NaVILA网络模型
    
    使用LLAVA视觉-语言模型处理RGB图像序列和指令文本，
    生成语言形式的导航动作。
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        model_path: Optional[str],
        num_video_frames: int = 8,
        forward_step: int = 25,
        turn_step: int = 15,
        instruction_sensor_uuid: str = "instruction",
        rgb_sensor_keys: Optional[List[str]] = None,
    ):
        super().__init__()

        self.num_video_frames = num_video_frames
        self.instruction_sensor_uuid = instruction_sensor_uuid
        self.action_sequence_mode = False
        self.action_sequence_length = 4
        self.available_rgb_keys = _augment_rgb_sensor_keys(
            [
                key
                for key in observation_space.spaces.keys()
                if isinstance(key, str) and "rgb" in key.lower()
            ]
        ) or []
        preferred_keys = _augment_rgb_sensor_keys(rgb_sensor_keys)
        if preferred_keys is None or len(preferred_keys) == 0:
            preferred_keys = self.available_rgb_keys or None
        self._preferred_rgb_keys = preferred_keys
        print(
            f"[NaVILA] observation_space 可用 RGB 传感器: {self.available_rgb_keys or 'None'}"
        )
        print(
            f"[NaVILA] NaVILA 实际使用的 RGB 传感器顺序: {self._preferred_rgb_keys or 'None'}"
        )
        self._debug_records: List[Dict[str, Any]] = []
        self._max_debug_records = 200
        self._step_counter = 0
        self._debug_log_interval = 50
        self._last_debug_record: Optional[Dict[str, Any]] = None
        self._last_rgb_key: Optional[str] = None
        
        # 初始化动作解析器
        self.action_parser = NaVILAActionParser(
            forward_step=forward_step,
            turn_step=turn_step,
        )
        
        # 加载LLAVA模型
        if model_path is None:
            raise ValueError("NaVILA model path is not provided")
        
        # 处理相对路径：如果路径是相对路径，尝试从项目根目录解析
        if not os.path.isabs(model_path):
            # 方法1: 尝试从当前文件位置向上查找项目根目录（包含pretrained_model的目录）
            resolved_path = None
            current = Path(__file__).resolve().parent
            max_depth = 10  # 限制向上查找的深度
            depth = 0
            
            while depth < max_depth and current != current.parent:
                # 检查当前目录或父目录是否有pretrained_model目录
                if (current / "pretrained_model").exists():
                    candidate = current / model_path
                    if candidate.exists():
                        resolved_path = candidate.resolve()
                        break
                
                # 检查当前目录下是否有该路径
                candidate = current / model_path
                if candidate.exists():
                    resolved_path = candidate.resolve()
                    break
                
                current = current.parent
                depth += 1
            
            # 方法2: 如果方法1失败，尝试相对于当前工作目录
            if resolved_path is None:
                candidate = Path(os.getcwd()) / model_path
                if candidate.exists():
                    resolved_path = candidate.resolve()
            
            # 方法3: 尝试从环境变量FALCON_ROOT获取项目根目录
            if resolved_path is None:
                falcon_root = os.environ.get("FALCON_ROOT", "")
                if falcon_root:
                    candidate = Path(falcon_root) / model_path
                    if candidate.exists():
                        resolved_path = candidate.resolve()
            
            if resolved_path is not None:
                model_path = str(resolved_path)
                logger.info(f"Resolved relative path to: {model_path}")
            else:
                # 如果所有尝试都失败，尝试使用绝对路径（相对于当前工作目录）
                # 这可能会失败，但至少会给出清晰的错误信息
                logger.warning(
                    f"Could not resolve relative path '{model_path}'. "
                    f"Current working directory: {os.getcwd()}. "
                    f"Trying to use it as-is."
                )
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            # 提供更详细的错误信息
            error_msg = (
                f"NaVILA model path does not exist: {model_path}\n"
                f"Current working directory: {os.getcwd()}\n"
            )
            if not os.path.isabs(model_path):
                error_msg += f"Resolved absolute path: {os.path.abspath(model_path)}\n"
            error_msg += (
                f"\nHint: If using a relative path, make sure it is relative to the project root directory\n"
                f"(the directory containing 'pretrained_model' folder).\n"
                f"Alternatively, use an absolute path or set the FALCON_ROOT environment variable."
            )
            raise ValueError(error_msg)
        
        # 使用get_model_name_from_path获取模型名（与navila项目一致）
        if NAVILA_AVAILABLE:
            try:
                model_name = get_model_name_from_path(model_path)
            except (NameError, AttributeError):
                model_name = os.path.basename(os.path.normpath(model_path))
        else:
            model_name = os.path.basename(os.path.normpath(model_path))
        
        # 模型加载选项：
        # - 不使用量化（默认）：load_8bit=False, load_4bit=False（内存占用最大，精度最高）
        # - 8-bit 量化：load_8bit=True（内存减少约 50%，精度略有下降）
        # - 4-bit 量化：load_4bit=True（内存减少约 75%，精度下降更多）
        # 
        # 注意：使用量化时，device_map 会被自动处理，量化库会将模型加载到 GPU
        # 如果遇到内存不足错误，可以启用量化：
        #   load_8bit=True  # 或 load_4bit=True
        # 
        # 当前使用 4-bit 量化以进一步减少内存占用（vision tower 仍需要大量内存）
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path, 
                model_name,
                device_map={"": "cuda"},  # 非量化时使用，量化时会被自动忽略
                device="cuda",
                load_4bit=False  # 8-bit 量化（内存减少约 50%）
                # load_4bit=True  # 4-bit 量化（内存减少约 75%，推荐用于内存严重不足的情况）
            )
        )
        
        # 将模型设置为评估模式
        self.model.eval()
        
        # 历史RGB帧缓存
        self.past_rgbs = []
        
        # 动作队列（用于处理需要重复的动作）
        self.action_queue = []
        
        # 输出大小（用于兼容性）
        self._output_size = 512
    
    @property
    def output_size(self):
        return self._output_size
    
    @property
    def is_blind(self):
        return False
    
    @property
    def num_recurrent_layers(self):
        return 1
    
    @property
    def recurrent_hidden_size(self):
        return self._output_size
    
    @property
    def perception_embedding_size(self):
        """返回感知嵌入的大小"""
        return self._output_size
    
    def _select_rgb_observation(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        根据优先级选择 RGB 观测；优先使用配置指定的 key，随后尝试常见命名，
        最后匹配任意包含 'rgb' 的键。
        """
        candidate_keys: List[str] = []
        if self._preferred_rgb_keys:
            candidate_keys.extend(self._preferred_rgb_keys)
        candidate_keys.extend(
            [
                "rgb",
                "agent_0_overhead_front_rgb",
                "agent_0_articulated_agent_jaw_rgb",
                "agent_0_third_rgb",
            ]
        )
        for key in candidate_keys:
            if key in observations:
                self._last_rgb_key = key
                return observations[key]
        for key in observations.keys():
            if "rgb" in key.lower():
                if key != self._last_rgb_key:
                    logger.warning(
                        "NaVILA policy falling back to RGB sensor '%s'. 可用传感器列表：%s",
                        key,
                        self._preferred_rgb_keys
                        or self.available_rgb_keys
                        or list(observations.keys()),
                    )
                self._last_rgb_key = key
                return observations[key]
        available = list(observations.keys())
        raise KeyError(
            "NaVILA policy 未能找到RGB输入。请在观测空间中启用RGB传感器，"
            "或通过 habitat_baselines.rl.policy.agent_0.rgb_sensor_keys 指定传感器键。"
            f" 当前可用观测：{available}"
        )
    
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            observations: 观察数据
            rnn_hidden_states: RNN隐藏状态（用于兼容性）
            prev_actions: 前一个动作
            masks: 掩码
            rnn_build_seq_info: RNN序列构建信息
            
        Returns:
            (features, rnn_hidden_states, aux_loss_state)
        """
        # 如果有动作队列，直接返回队列中的动作
        # 重要：只有当队列为空时，才会进行下一轮观测和模型推理
        # 这确保了基于一次观测生成的动作序列全部执行完后，才进行新的观测
        aux_loss_state: Dict[str, torch.Tensor] = {}
        if len(self.action_queue) > 0:
            queued_entry = self.action_queue.pop(0)
            if isinstance(queued_entry, dict):
                action = int(queued_entry.get("action", 0))
                debug_info = queued_entry.get("debug")
            else:
                action = int(queued_entry)
                debug_info = None
            # 返回one-hot编码的动作
            # 注意：此时不会更新历史帧，也不会进行模型推理
            rgb_obs = self._select_rgb_observation(observations)
            batch_size = rgb_obs.shape[0]
            features = torch.zeros(batch_size, self._output_size, device=rgb_obs.device)
            features[0, action] = 1.0  # 简单编码
            if debug_info:
                aux_loss_state["navila_debug"] = torch.tensor(0.0)
                self._store_debug_record(debug_info)
            return features, rnn_hidden_states, aux_loss_state
        
        # 队列为空，进行下一轮观测和模型推理
        # 获取当前RGB观察
        rgb_obs = self._select_rgb_observation(observations)  # [batch, H, W, C]
        batch_size = rgb_obs.shape[0]
        
        # 目前只支持batch_size=1
        if batch_size != 1:
            raise NotImplementedError("NaVILA policy currently only supports batch_size=1")
        
        # 转换为PIL图像
        curr_rgb = Image.fromarray(np.uint8(rgb_obs[0].cpu().numpy())).convert("RGB")
        
        # 添加到历史帧（只有在队列为空时才会更新历史帧）
        self.past_rgbs.append(curr_rgb)
        
        # 采样和填充图像到固定帧数
        sampled_frames = sample_and_pad_images(
            self.past_rgbs, 
            num_frames=self.num_video_frames
        )
        
        # 获取指令文本
        single_observations = {
            key: (
                value[0]
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size
                else value
            )
            for key, value in observations.items()
        }
        instruction = extract_navila_instruction(
            single_observations,
            instruction_sensor_uuid=self.instruction_sensor_uuid,
        )
        # 构建提示
        interleaved_images = "<image>\n" * (len(sampled_frames) - 1)
        if getattr(self, 'action_sequence_mode', False):
            K = getattr(self, 'action_sequence_length', 4)
            question = (
                f"<video>\n"
                f'Instruction: {instruction}\n'
                f"Predict the next {K} navigation actions. "
                f"Output each action on a separate line, or use semicolons between actions. "
                f"Actions: move forward 25 cm, turn left 15 degrees, turn right 15 degrees, stop."
            )
        else:
            question = (
                f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
                f'of historical observations {interleaved_images}, and current observation <image>\n. Your assigned task is: "{instruction}" '
                f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
                f"degree, moving forward a certain distance, or stop if the task is completed."
            )
        
        # 构建对话
        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # 处理图像
        images_tensor = process_images(
            sampled_frames, 
            self.image_processor, 
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        
        # Tokenize输入
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )
        
        # 停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # 生成输出
        # 注意：对于短序列生成（max_new_tokens=32），use_cache=False 性能影响不大
        # 且可以避免 transformers 4.56.0 中 past_key_values 为 None 的兼容性问题
        # 显式设置 past_key_values=None 以确保兼容性
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor.half().to(self.model.device),
                do_sample=False,
                temperature=0.0,
                max_new_tokens=32,
                use_cache=False,
                past_key_values=None,
                stopping_criteria=[stopping_criteria],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_text = output_text.strip()
        
        if output_text.endswith(stop_str):
            output_text = output_text[: -len(stop_str)]
        output_text = output_text.strip()
        
        # ── 解析动作（支持多动作序列模式）──
        if getattr(self, 'action_sequence_mode', False):
            action_sequence = self.action_parser.parse_action_sequence(output_text)
            if len(action_sequence) == 0:
                # Fallback to single-action parser
                action, num_repeats = self.action_parser.parse_action(output_text)
                action_sequence = [action]
                if num_repeats > 1:
                    for _ in range(1, num_repeats):
                        action_sequence.append(action)
            logger.info(
                "[NaVILA-seq] output=\"%s\" actions=%s queue_before=%d",
                output_text[:200],
                action_sequence,
                len(self.action_queue),
            )
            # Queue remaining actions (beyond the first)
            for act in action_sequence[1:]:
                self.action_queue.append({"action": int(act)})
            action = int(action_sequence[0])
            num_repeats = 1
        else:
            action, num_repeats = self.action_parser.parse_action(output_text)

        # 如果需要重复多次，将后续动作加入队列
        debug_record = self._build_debug_record(
            instruction=instruction,
            model_output=output_text,
            action=action,
            num_repeats=num_repeats,
            repeat_index=1,
            from_queue=False,
        )
        self._store_debug_record(debug_record)
        aux_loss_state["navila_debug"] = torch.tensor(0.0)
        if num_repeats > 1:
            for repeat_idx in range(2, num_repeats + 1):
                queued_record = dict(debug_record)
                queued_record["repeat_index"] = repeat_idx
                queued_record["from_queue"] = True
                self.action_queue.append({"action": action, "debug": queued_record})

        # 返回特征（简单编码）
        features = torch.zeros(batch_size, self._output_size, device=rgb_obs.device)
        features[0, action] = 1.0

        return features, rnn_hidden_states, aux_loss_state
    
    def reset_history(self):
        """重置历史帧和动作队列"""
        self.past_rgbs = []
        self.action_queue = []
    
    def _build_debug_record(
        self,
        instruction: str,
        model_output: str,
        action: int,
        num_repeats: int,
        repeat_index: int,
        from_queue: bool,
    ) -> Dict[str, Any]:
        return {
            "step": self._step_counter + 1,
            "instruction": instruction,
            "model_output": model_output,
            "action_id": int(action),
            "action_name": ACTION_ID_TO_NAME.get(action, f"action_{action}"),
            "repeats": int(max(1, num_repeats)),
            "repeat_index": int(max(1, repeat_index)),
            "from_queue": bool(from_queue),
        }
    
    def _store_debug_record(self, record: Dict[str, Any]) -> None:
        self._step_counter += 1
        self._last_debug_record = record
        self._debug_records.append(record)
        if len(self._debug_records) > self._max_debug_records:
            self._debug_records.pop(0)
        if self._step_counter % self._debug_log_interval == 0 or record.get("from_queue"):
            self._log_debug_record(record)
    
    def _log_debug_record(self, record: Dict[str, Any]) -> None:
        # logger.info(
        #     "[NaVILA][record] step=%d action=%s repeat=%d/%d queue=%s",
        #     record.get("step", -1),
        #     record.get("action_name", record.get("action_id")),
        #     record.get("repeat_index", 1),
        #     record.get("repeats", 1),
        #     record.get("from_queue", False),
        # )  # silenced: per-step verbose
        pass  # body silenced: per-step verbose
    
    def get_debug_records(self) -> List[Dict[str, Any]]:
        return list(self._debug_records)
    
    @property
    def last_debug_record(self) -> Optional[Dict[str, Any]]:
        return self._last_debug_record


def format_navila_debug_overlay(
    debug_info: Optional[Dict[str, Any]],
    width: int = 120,
) -> Optional[str]:
    if not debug_info:
        return None
    lines: List[str] = []
    instruction = debug_info.get("instruction")
    model_output = debug_info.get("model_output")
    action_name = debug_info.get("action_name")
    repeats = debug_info.get("repeats", 1)
    repeat_index = debug_info.get("repeat_index", 1)
    if instruction:
        lines.append(f"Instr: {textwrap.shorten(str(instruction), width=width, placeholder='…')}")
    if model_output:
        lines.append(f"LLM: {textwrap.shorten(str(model_output), width=width, placeholder='…')}")
    if action_name is not None:
        if repeats > 1:
            lines.append(f"Action: {action_name} ({repeat_index}/{repeats})")
        else:
            lines.append(f"Action: {action_name}")
    if not lines:
        return None
    return "\n".join(lines)
