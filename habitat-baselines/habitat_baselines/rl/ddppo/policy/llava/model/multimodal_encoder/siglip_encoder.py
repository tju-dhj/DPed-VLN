# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# 修复 transformers 兼容性问题：确保缺失的函数和类存在
# 必须在导入 transformers 模块之前修复，否则导入时会失败
import sys
import importlib
import types

# 修复 is_torch_tpu_available（这是最关键的修复）
try:
    utils_module = importlib.import_module("transformers.utils")
    if not hasattr(utils_module, "is_torch_tpu_available"):
        def _is_torch_tpu_available() -> bool:
            return False
        utils_module.is_torch_tpu_available = _is_torch_tpu_available
except Exception:
    pass

# 修复 PreTrainedAudioTokenizerBase：使用 import hook 在导入时修复
# 使用 meta_path 来拦截 transformers 模块的导入
class TransformersCompatibilityHook:
    """Import hook to fix transformers compatibility issues"""
    def find_spec(self, name, path, target=None):
        # 当导入 transformers.modeling_utils 时，在加载后立即修复
        if name == "transformers.modeling_utils":
            # 返回 None 以使用默认加载器，但在加载后修复
            return None
        return None
    
    def create_module(self, spec):
        # 不使用自定义创建，使用默认
        return None
    
    def exec_module(self, module):
        # 在模块加载后，立即修复 PreTrainedAudioTokenizerBase
        if module.__name__ == "transformers.modeling_utils":
            if not hasattr(module, "PreTrainedAudioTokenizerBase"):
                class PreTrainedAudioTokenizerBase:
                    """Placeholder for PreTrainedAudioTokenizerBase in transformers 4.56.0"""
                    pass
                module.PreTrainedAudioTokenizerBase = PreTrainedAudioTokenizerBase

# 注册 import hook（Python 3.4+）
try:
    sys.meta_path.insert(0, TransformersCompatibilityHook())
except Exception:
    # 如果注册失败，继续（我们会在导入后修复）
    pass

import torch

# 导入 transformers 模块
from transformers import PretrainedConfig, SiglipImageProcessor, AutoModel, AutoConfig

# 导入后立即尝试修复 PreTrainedAudioTokenizerBase（双重保险）
try:
    # 检查 modeling_utils 是否已加载，如果是，尝试修复
    if "transformers.modeling_utils" in sys.modules:
        modeling_utils_module = sys.modules["transformers.modeling_utils"]
        if not hasattr(modeling_utils_module, "PreTrainedAudioTokenizerBase"):
            class PreTrainedAudioTokenizerBase:
                """Placeholder for PreTrainedAudioTokenizerBase in transformers 4.56.0"""
                pass
            modeling_utils_module.PreTrainedAudioTokenizerBase = PreTrainedAudioTokenizerBase
except Exception:
    # 如果修复失败，继续（我们会在代码中使用 try-except 处理）
    pass

from .vision_encoder import VisionTower, VisionTowerS2


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        
        # 使用 AutoModel 代替 SiglipVisionModel，避免 transformers 兼容性问题
        # AutoModel 可以自动识别模型类型，更灵活
        # 注意：由于 transformers 4.56.0 的兼容性问题，需要特殊处理 dtype 参数
        
        # 获取目标数据类型
        try:
            target_dtype = eval(config.model_dtype)
        except Exception:
            target_dtype = None
        
        # 准备加载参数（不包含 torch_dtype，因为在 transformers 4.56.0 中可能不被支持）
        # 设置 attn_implementation="eager" 以避免 sdpa 兼容性问题
        load_kwargs = {
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # 使用 eager attention 实现，避免 sdpa 错误
        }
        
        # 尝试使用 AutoModel 加载（优先使用）
        try:
            # 先尝试不使用 dtype 参数加载
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path,
                **load_kwargs
            )
            # 如果指定了 dtype，在加载后设置
            if target_dtype is not None:
                try:
                    self.vision_tower = self.vision_tower.to(dtype=target_dtype)
                except Exception as e_dtype:
                    import warnings
                    warnings.warn(
                        f"Failed to set dtype to {target_dtype}: {e_dtype}. "
                        f"Continuing with default dtype."
                    )
            
            # 如果提供了 state_dict，加载它
            if state_dict is not None:
                try:
                    self.vision_tower.load_state_dict(state_dict, strict=False)
                except Exception as e_load:
                    # 如果加载 state_dict 失败，记录警告但继续
                    import warnings
                    warnings.warn(
                        f"Failed to load state_dict into vision tower: {e_load}. "
                        f"Continuing without state_dict."
                    )
        except ImportError as e_import:
            # 如果导入失败（可能是兼容性问题），提供更详细的错误信息
            raise RuntimeError(
                f"Failed to import vision tower model from {model_name_or_path}. "
                f"This may be due to transformers 4.56.0 compatibility issues. "
                f"Error: {e_import}\n"
                f"Please consider upgrading transformers or using a compatible version."
            ) from e_import
        except Exception as e:
            # 其他错误，尝试使用 SiglipVisionModel（可能需要兼容性补丁）
            try:
                # 尝试直接导入 SiglipVisionModel
                try:
                    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
                except ImportError:
                    from transformers import SiglipVisionModel
                
                # 先尝试不使用 dtype 参数加载
                # 设置 attn_implementation="eager" 以避免 sdpa 兼容性问题
                self.vision_tower = SiglipVisionModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",  # 使用 eager attention 实现，避免 sdpa 错误
                )
                
                # 如果指定了 dtype，在加载后设置
                if target_dtype is not None:
                    try:
                        self.vision_tower = self.vision_tower.to(dtype=target_dtype)
                    except Exception as e_dtype:
                        import warnings
                        warnings.warn(
                            f"Failed to set dtype to {target_dtype}: {e_dtype}. "
                            f"Continuing with default dtype."
                        )
                
                # 如果提供了 state_dict，加载它
                if state_dict is not None:
                    try:
                        self.vision_tower.load_state_dict(state_dict, strict=False)
                    except Exception as e_load:
                        import warnings
                        warnings.warn(
                            f"Failed to load state_dict into vision tower: {e_load}. "
                            f"Continuing without state_dict."
                        )
            except ImportError as e2_import:
                raise RuntimeError(
                    f"Failed to import SiglipVisionModel from transformers. "
                    f"This may be due to transformers 4.56.0 compatibility issues. "
                    f"AutoModel error: {e}, SiglipVisionModel import error: {e2_import}\n"
                    f"Please consider upgrading transformers or using a compatible version."
                ) from e2_import
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load vision tower from {model_name_or_path}. "
                    f"AutoModel error: {e}, SiglipVisionModel error: {e2}"
                ) from e2
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        
        # 使用 AutoModel 代替 SiglipVisionModel，避免 transformers 兼容性问题
        # 获取目标数据类型
        try:
            target_dtype = eval(config.model_dtype)
        except Exception:
            target_dtype = None
        
        try:
            # 先尝试不使用 dtype 参数加载
            # 设置 attn_implementation="eager" 以避免 sdpa 兼容性问题
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # 使用 eager attention 实现，避免 sdpa 错误
            )
            # 如果指定了 dtype，在加载后设置
            if target_dtype is not None:
                try:
                    self.vision_tower = self.vision_tower.to(dtype=target_dtype)
                except Exception as e_dtype:
                    import warnings
                    warnings.warn(
                        f"Failed to set dtype to {target_dtype}: {e_dtype}. "
                        f"Continuing with default dtype."
                    )
        except Exception as e:
            # 如果 AutoModel 失败，尝试使用 SiglipVisionModel
            try:
                # 尝试直接导入 SiglipVisionModel
                try:
                    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
                except ImportError:
                    from transformers import SiglipVisionModel
                
                # 先尝试不使用 dtype 参数加载
                # 设置 attn_implementation="eager" 以避免 sdpa 兼容性问题
                self.vision_tower = SiglipVisionModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager",  # 使用 eager attention 实现，避免 sdpa 错误
                )
                
                # 如果指定了 dtype，在加载后设置
                if target_dtype is not None:
                    try:
                        self.vision_tower = self.vision_tower.to(dtype=target_dtype)
                    except Exception as e_dtype:
                        import warnings
                        warnings.warn(
                            f"Failed to set dtype to {target_dtype}: {e_dtype}. "
                            f"Continuing with default dtype."
                        )
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load vision tower from {model_name_or_path}. "
                    f"AutoModel error: {e}, SiglipVisionModel error: {e2}"
                ) from e2

        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[-1]

        self.is_loaded = True
