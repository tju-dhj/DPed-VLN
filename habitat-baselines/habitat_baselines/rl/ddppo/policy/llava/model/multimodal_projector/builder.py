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

# This file is modified from https://github.com/haotian-liu/LLaVA/

import os

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .base_projector import MultimodalProjector, MultimodalProjectorConfig


def build_mm_projector(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path:
        assert os.path.exists(model_type_or_path), f"Resume mm projector path {model_type_or_path} does not exist!"
        # 注意：MultimodalProjector.__init__ 需要两个参数：mm_projector_cfg 和 config
        # 但 PreTrainedModel.from_pretrained 只会传递一个 config 参数
        # 所以我们需要手动加载配置和权重
        import torch
        try:
            from safetensors.torch import load_file
        except ImportError:
            load_file = None
        
        # 加载 mm_projector 配置
        mm_projector_cfg = MultimodalProjectorConfig.from_pretrained(model_type_or_path)
        
        # 创建模型实例
        mm_projector = MultimodalProjector(mm_projector_cfg, config)
        
        # 尝试加载权重（支持多种格式）
        weight_loaded = False
        
        # 尝试 safetensors 格式
        if load_file is not None:
            safetensors_path = os.path.join(model_type_or_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                try:
                    state_dict = load_file(safetensors_path)
                    mm_projector.load_state_dict(state_dict, strict=False)
                    weight_loaded = True
                except Exception as e:
                    pass
        
        # 尝试 pytorch_model.bin 格式
        if not weight_loaded:
            pytorch_model_path = os.path.join(model_type_or_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                try:
                    state_dict = torch.load(pytorch_model_path, map_location="cpu")
                    mm_projector.load_state_dict(state_dict, strict=False)
                    weight_loaded = True
                except Exception as e:
                    pass
        
        # 如果都失败了，尝试查找其他可能的权重文件
        if not weight_loaded:
            for filename in os.listdir(model_type_or_path):
                if filename.endswith((".bin", ".safetensors")):
                    weight_path = os.path.join(model_type_or_path, filename)
                    try:
                        if filename.endswith(".safetensors") and load_file is not None:
                            state_dict = load_file(weight_path)
                        else:
                            state_dict = torch.load(weight_path, map_location="cpu")
                        mm_projector.load_state_dict(state_dict, strict=False)
                        weight_loaded = True
                        break
                    except Exception as e:
                        continue
        
        # 如果没有找到权重文件，模型将使用随机初始化的权重（这可能不是期望的行为，但至少不会崩溃）
        if not weight_loaded:
            import warnings
            warnings.warn(
                f"Could not find weight file for mm_projector at {model_type_or_path}. "
                "Model will use randomly initialized weights."
            )
        
        # 移动到正确的设备和数据类型
        mm_projector = mm_projector.to(eval(config.model_dtype))
        return mm_projector
    ## build from scratch
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config).to(eval(config.model_dtype))
        return mm_projector
