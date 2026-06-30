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

import math
import os
import os.path as osp
import warnings
from typing import Tuple

import torch

# 修复 transformers 兼容性问题：必须在导入 transformers 之前执行
# 1. 确保 is_torch_tpu_available 存在
# 2. 手动注册 LlamaForCausalLM 到 transformers.models.llama 模块
try:
    import importlib
    
    # 修复 is_torch_tpu_available
    utils_module = importlib.import_module("transformers.utils")
    if not hasattr(utils_module, "is_torch_tpu_available"):
        def _is_torch_tpu_available() -> bool:
            return False
        utils_module.is_torch_tpu_available = _is_torch_tpu_available
    
    # 手动注册 LlamaForCausalLM
    # transformers 4.56.0 可能没有在 __init__.py 中正确导出 LlamaForCausalLM
    try:
        # 导入 modeling_llama 模块
        modeling_llama = importlib.import_module("transformers.models.llama.modeling_llama")
        if hasattr(modeling_llama, "LlamaForCausalLM"):
            # 检查是否已经在 transformers.models.llama 中导出
            llama_module = importlib.import_module("transformers.models.llama")
            if not hasattr(llama_module, "LlamaForCausalLM"):
                # 手动导出到 transformers.models.llama 模块
                setattr(llama_module, "LlamaForCausalLM", modeling_llama.LlamaForCausalLM)
                # 同时更新 __all__ 列表（如果存在）
                if hasattr(llama_module, "__all__"):
                    if "LlamaForCausalLM" not in llama_module.__all__:
                        llama_module.__all__.append("LlamaForCausalLM")
                
                # 确保 MODEL_FOR_CAUSAL_LM_MAPPING 中包含映射
                try:
                    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
                    from transformers import LlamaConfig
                    if LlamaConfig not in MODEL_FOR_CAUSAL_LM_MAPPING:
                        MODEL_FOR_CAUSAL_LM_MAPPING[LlamaConfig] = modeling_llama.LlamaForCausalLM
                except Exception:
                    pass
    except Exception:
        pass
except Exception:
    pass

import transformers
from huggingface_hub import file_exists, repo_exists
from huggingface_hub.utils import HFValidationError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ...utils.logging import logger


def has_tokenizer(repo_id_or_path: str) -> bool:
    # Check if the tokenizer is in a local directory
    if osp.exists(osp.join(repo_id_or_path, "tokenizer_config.json")):
        return True

    # Check if the tokenizer is in a Hugging Face Hub repo
    try:
        return repo_exists(repo_id_or_path) and file_exists(repo_id_or_path, "tokenizer_config.json")
    except HFValidationError:
        return False


def context_length_extension(config):
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    model_max_length = getattr(config, "model_max_length", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    return config


def build_llm_and_tokenizer(
    model_name_or_path: str,
    config: PretrainedConfig,
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    # 修复 transformers 兼容性问题：transformers 4.56.0 无法找到 LlamaForCausalLM
    # 即使手动注册了 LlamaForCausalLM，AutoModelForCausalLM.from_pretrained() 仍然会从配置文件读取 architectures 字段
    # 如果配置文件中包含 architectures 字段指向不存在的类，加载会失败
    # 解决方案：在加载之前临时移除配置文件中的 architectures 字段，让 AutoModelForCausalLM 根据 model_type 自动识别
    config_file = osp.join(model_name_or_path, "config.json")
    if not osp.exists(config_file):
        config_file = osp.join(model_name_or_path, "llm", "config.json")
    
    original_architectures_in_file = None
    config_modified = False
    
    # 加载配置（先读取，不修改文件）
    llm_cfg = AutoConfig.from_pretrained(model_name_or_path)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        context_length_extension(llm_cfg)
    
    # 检查配置文件是否有 architectures 字段指向 LlamaForCausalLM
    # 如果有，需要在加载模型之前临时移除它
    if osp.exists(config_file) and hasattr(llm_cfg, "architectures") and llm_cfg.architectures:
        arch_name = llm_cfg.architectures
        if isinstance(arch_name, list):
            arch_name = arch_name[0] if arch_name else ""
        else:
            arch_name = str(arch_name)
        
        # 如果 architectures 指向 LlamaForCausalLM，临时移除它
        if "LlamaForCausalLM" in arch_name:
            try:
                import json
                # 读取配置文件
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                # 保存原始 architectures
                if "architectures" in config_dict:
                    original_architectures_in_file = config_dict["architectures"]
                    # 移除 architectures 字段
                    del config_dict["architectures"]
                    # 写回配置文件
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_dict, f, indent=2, ensure_ascii=False)
                    config_modified = True
                    # 同时更新内存中的配置
                    llm_cfg.architectures = None
                    logger.info(
                        f"Temporarily removed 'architectures' field ({original_architectures_in_file}) from {config_file} "
                        f"to fix compatibility with transformers {transformers.__version__}. "
                        f"AutoModelForCausalLM will use model_type to identify the model."
                    )
            except Exception as e:
                logger.warning(f"Failed to modify config file {config_file}: {e}")
    
    # 准备加载参数
    # 对于量化模型，需要移除 device_map 相关参数，避免 accelerate 的 dispatch_model 尝试移动模型
    load_kwargs = {
        "torch_dtype": eval(config.model_dtype),
        "low_cpu_mem_usage": True,
    }
    
    # 检查是否使用量化
    is_quantized = kwargs.get("load_in_8bit", False) or kwargs.get("load_in_4bit", False)
    
    if is_quantized:
        # 量化模型：完全不传递 device_map 相关参数，避免 dispatch_model 调用
        # 量化模型已经自动加载到正确的设备，不需要额外的设备移动
        # 注意：不传递 device_map 参数（而不是传递 None），以避免 transformers 调用 dispatch_model
        for key, value in kwargs.items():
            if key not in ["device_map", "max_memory"]:
                load_kwargs[key] = value
        # 不设置 device_map，让量化库自动处理设备分配
    else:
        # 非量化模型：正常传递所有参数
        load_kwargs.update(kwargs)
    
    # 在加载模型之前，应用 transformers 量化兼容性补丁
    # 这必须在 transformers.modeling_utils 完全加载之后进行
    # 对于 4-bit 量化，必须确保补丁在模型加载之前应用
    try:
        import transformers.modeling_utils as modeling_utils
        if hasattr(modeling_utils, 'set_module_quantized_tensor_to_device'):
            original_func = modeling_utils.set_module_quantized_tensor_to_device
            
            # 检查函数是否已经被补丁（避免重复补丁）
            func_code = getattr(original_func, '__code__', None)
            is_already_patched = (
                hasattr(original_func, '__wrapped__') or
                (func_code and 'fp16_statistics' in str(func_code.co_names) and 'pop' in str(func_code.co_names))
            )
            
            if not is_already_patched:
                # 检查是否需要补丁（函数不接受 fp16_statistics）
                import inspect
                try:
                    sig = inspect.signature(original_func)
                    if 'fp16_statistics' not in sig.parameters:
                        # 应用补丁：创建一个包装函数，忽略 fp16_statistics 参数
                        def patched_func(*args, **kwargs):
                            kwargs.pop('fp16_statistics', None)
                            return original_func(*args, **kwargs)
                        # 保存原始函数的引用，以便调试
                        patched_func.__wrapped__ = original_func
                        modeling_utils.set_module_quantized_tensor_to_device = patched_func
                        logger.debug("Applied set_module_quantized_tensor_to_device patch for quantization")
                except Exception as e:
                    # 如果无法检查签名，直接应用补丁
                    def patched_func(*args, **kwargs):
                        kwargs.pop('fp16_statistics', None)
                        return original_func(*args, **kwargs)
                    patched_func.__wrapped__ = original_func
                    modeling_utils.set_module_quantized_tensor_to_device = patched_func
                    logger.debug(f"Applied set_module_quantized_tensor_to_device patch (no sig check): {e}")
            else:
                logger.debug("set_module_quantized_tensor_to_device already patched")
    except Exception as e:
        logger.warning(f"Failed to patch set_module_quantized_tensor_to_device: {e}")
        # 如果补丁失败，继续尝试加载
    
    # 也修补 integrations 模块（如果函数是从那里导入的）
    try:
        from transformers import integrations
        if hasattr(integrations, 'set_module_quantized_tensor_to_device'):
            original_func = integrations.set_module_quantized_tensor_to_device
            if not hasattr(original_func, '__wrapped__'):
                def patched_func(*args, **kwargs):
                    kwargs.pop('fp16_statistics', None)
                    return original_func(*args, **kwargs)
                patched_func.__wrapped__ = original_func
                integrations.set_module_quantized_tensor_to_device = patched_func
                logger.debug("Applied integrations.set_module_quantized_tensor_to_device patch")
    except Exception:
        pass
    
    # 对于量化模型，修补 PreTrainedModel.to 方法，使其对量化模型直接返回
    if is_quantized:
        try:
            import transformers.modeling_utils as modeling_utils
            if hasattr(modeling_utils, 'PreTrainedModel'):
                original_to = modeling_utils.PreTrainedModel.to
                if not hasattr(original_to, '__wrapped__'):
                    def patched_to(self, *args, **kwargs):
                        # 检查是否是量化模型
                        is_quantized_model = (
                            hasattr(self, 'hf_quantizer') or 
                            getattr(self, 'is_loaded_in_8bit', False) or 
                            getattr(self, 'is_loaded_in_4bit', False)
                        )
                        
                        if is_quantized_model:
                            # 量化模型不支持 .to()，直接返回自身
                            return self
                        
                        # 非量化模型，使用原始方法
                        return original_to(self, *args, **kwargs)
                    
                    patched_to.__wrapped__ = original_to
                    modeling_utils.PreTrainedModel.to = patched_to
                    logger.debug("Applied PreTrainedModel.to patch for quantization")
        except Exception as e:
            logger.warning(f"Failed to patch PreTrainedModel.to: {e}")
    
    # 对于量化模型，需要修补 dispatch_model 以避免调用 model.to()
    if is_quantized:
        # 再次检查并移除可能触发 dispatch_model 的参数
        load_kwargs.pop('device_map', None)
        load_kwargs.pop('max_memory', None)
        
        # 修补 accelerate 的 dispatch_model 函数，对于量化模型直接返回
        try:
            from accelerate import big_modeling
            original_dispatch_model = big_modeling.dispatch_model
            
            # 检查是否已经被补丁（避免重复补丁）
            is_already_patched = hasattr(original_dispatch_model, '__wrapped__')
            
            if not is_already_patched:
                def patched_dispatch_model(model, device_map=None, **kwargs):
                    # 首先检查是否是量化模型（无论 device_map 是什么）
                    # 方法1: 检查模型属性
                    is_quantized_model = (
                        hasattr(model, 'hf_quantizer') or 
                        getattr(model, 'is_loaded_in_8bit', False) or 
                        getattr(model, 'is_loaded_in_4bit', False)
                    )
                    
                    # 方法2: 检查模型参数（量化参数通常有 SCB 属性）
                    if not is_quantized_model:
                        try:
                            param_iter = iter(model.named_parameters())
                            for _ in range(10):  # 只检查前10个参数
                                name, param = next(param_iter)
                                if hasattr(param, 'SCB') or (hasattr(param, 'data') and hasattr(param.data, 'SCB')):
                                    is_quantized_model = True
                                    break
                        except (StopIteration, Exception):
                            pass
                    
                    # 方法3: 检查模型类名
                    if not is_quantized_model:
                        try:
                            model_class_name = model.__class__.__name__
                            if '8bit' in model_class_name.lower() or '4bit' in model_class_name.lower():
                                is_quantized_model = True
                        except Exception:
                            pass
                    
                    # 如果是量化模型，直接返回，避免调用 model.to()
                    if is_quantized_model:
                        return model
                    
                    # 非量化模型，使用原始函数
                    try:
                        return original_dispatch_model(model, device_map=device_map, **kwargs)
                    except ValueError as e:
                        # 如果错误是关于量化模型的 .to() 不支持，直接返回模型
                        error_str = str(e)
                        if "not supported for" in error_str and ("4-bit" in error_str or "8-bit" in error_str):
                            return model
                        raise
                
                patched_dispatch_model.__wrapped__ = original_dispatch_model
                big_modeling.dispatch_model = patched_dispatch_model
                logger.debug("Applied dispatch_model patch for quantization")
            else:
                logger.debug("dispatch_model already patched")
        except Exception as e:
            # 如果补丁失败，记录警告但继续
            import warnings
            warnings.warn(f"Failed to patch dispatch_model: {e}")
    
    # 尝试加载模型
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            config=llm_cfg, 
            trust_remote_code=False,
            *args, 
            **load_kwargs
        )
    except (ValueError, RuntimeError) as e:
        error_msg = str(e)
        # 如果仍然失败，尝试不传递 config 参数，让 AutoModelForCausalLM 自动从文件加载
        if "Could not find" in error_msg and ("LlamaForCausalLM" in error_msg or "transformers" in error_msg):
            logger.warning(
                f"Failed to load with config parameter, trying without config. Error: {error_msg}"
            )
            try:
                # 准备 fallback 加载参数
                # 对于量化模型，确保不传递 device_map 相关参数
                fallback_kwargs = {
                    "torch_dtype": eval(config.model_dtype),
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": False,
                }
                if is_quantized:
                    # 量化模型：只传递量化相关参数，不传递 device_map
                    for key, value in kwargs.items():
                        if key not in ["device_map", "max_memory"]:
                            fallback_kwargs[key] = value
                else:
                    # 非量化模型：传递所有参数
                    for key, value in kwargs.items():
                        if key != 'low_cpu_mem_usage':
                            fallback_kwargs[key] = value
                llm = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    *args,
                    **fallback_kwargs
                )
                # 更新配置
                llm_cfg = llm.config
                if attn_implementation is not None:
                    llm_cfg._attn_implementation = attn_implementation
                if model_max_length is not None:
                    llm_cfg.model_max_length = model_max_length
            except Exception as e2:
                logger.error(
                    f"All loading methods failed. Original error: {error_msg}, "
                    f"Fallback error: {str(e2)}"
                )
                raise ValueError(
                    f"Failed to load model. Transformers version {transformers.__version__} "
                    f"cannot find LlamaForCausalLM even after manual registration and config modification.\n"
                    f"Please consider upgrading transformers or using a compatible version.\n"
                    f"Original error: {error_msg}\n"
                    f"Fallback error: {str(e2)}"
                ) from e2
        else:
            raise e
    finally:
        # 确保配置文件已恢复（如果被修改了）
        if config_modified and original_architectures_in_file is not None and osp.exists(config_file):
            try:
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                # 只有在 architectures 字段不存在时才恢复
                if "architectures" not in config_dict:
                    config_dict["architectures"] = original_architectures_in_file
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_dict, f, indent=2, ensure_ascii=False)
                    logger.info(f"Restored 'architectures' field in {config_file}")
            except Exception as e:
                logger.warning(f"Failed to restore config file {config_file}: {e}")

    # Locate the tokenizer.
    llm_path = model_name_or_path
    if not has_tokenizer(llm_path):
        llm_path = osp.join(llm_path, "llm")
    if not has_tokenizer(llm_path):
        raise ValueError(f"Cannot find tokenizer in {llm_path}.")

    # TODO(ligeng): use LLM class to judge to better compability.
    # 尝试从 architectures 字段获取架构名称，如果失败则从 model_type 推断
    llm_arch = None
    try:
        architectures = getattr(llm_cfg, "architectures", None)
        if architectures and len(architectures) > 0:
            llm_arch = architectures[0].lower()
    except BaseException:
        pass
    
    # 如果无法从 architectures 获取，尝试从 model_type 推断
    if llm_arch is None:
        model_type = getattr(llm_cfg, "model_type", None)
        if model_type:
            llm_arch = model_type.lower()
            logger.debug(f"Cannot find architectures field, using model_type '{model_type}' instead.")
        else:
            warnings.warn(f'Cannot find LLM architecture or model_type, please check the "config.json" under "{llm_path}".')
            # 设置一个默认值，避免后续代码出错
            llm_arch = "llama"  # 默认使用 llama，因为这是最常见的架构
    
    if llm_arch and "mpt" in llm_arch:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
        )
    elif "yi" in llm_path or (
        getattr(llm_cfg, "num_hidden_layers", -1) == 60 and getattr(llm_cfg, "num_attention_heads", -1) == 56
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )

    # Load chat template if specified.
    if getattr(config, "chat_template", None) is not None:
        logger.info(f"Using chat template: {config.chat_template}")
        fpath = os.path.join(os.path.dirname(__file__), "chat_templates", f"{config.chat_template}.jinja")
        with open(fpath) as fd:
            chat_template = fd.read()
        tokenizer.chat_template = chat_template.replace("    ", "").replace("\n", "")

    # 修复 transformers 4.56.0 兼容性问题：添加缺失的 _is_stateful 属性
    # 这个属性在较新版本的 transformers 中存在，但在 4.56.0 中可能缺失
    # 需要在类级别设置，因为 transformers 会在类级别检查这个属性
    llm_class = llm.__class__
    if not hasattr(llm_class, '_is_stateful'):
        # 设置 _is_stateful 为 False（对于大多数模型都是 False）
        llm_class._is_stateful = False
        logger.debug(f"Added missing '_is_stateful' attribute to {llm_class.__name__}")
    
    # 同时检查基类，确保所有相关类都有这个属性
    for base_class in llm_class.__mro__:
        if hasattr(base_class, '__name__') and 'CausalLM' in base_class.__name__:
            if not hasattr(base_class, '_is_stateful'):
                base_class._is_stateful = False
                logger.debug(f"Added missing '_is_stateful' attribute to base class {base_class.__name__}")
    
    # 检查并修复设备分散问题
    # 如果模型使用了 device_map 或 accelerate，可能被分散到多个 GPU
    # 需要确保所有参数都在同一设备上
    try:
        param_devices = set()
        device_param_count = {}
        for name, param in llm.named_parameters():
            dev_str = str(param.device)
            param_devices.add(dev_str)
            device_param_count[dev_str] = device_param_count.get(dev_str, 0) + 1
        
        if len(param_devices) > 1:
            logger.warning(f"LLM parameters are distributed across multiple devices: {param_devices}")
            logger.warning(f"Device parameter count: {device_param_count}")
            logger.warning(f"This will cause device mismatch errors. Consolidating to single device...")
            
            # 获取第一个设备作为目标设备
            target_device = list(param_devices)[0]
            logger.info(f"Moving all LLM parameters to {target_device}")
            
            # 尝试禁用 device_map（如果存在）
            # 注意：不能设置为 None，因为 transformers 代码会访问 .values()
            # 应该设置为空字典 {}，表示没有设备映射
            if hasattr(llm, 'hf_device_map'):
                logger.info(f"Disabling hf_device_map: {llm.hf_device_map}")
                llm.hf_device_map = {}
            
            # 移动整个模型到目标设备
            llm = llm.to(target_device)
            
            # 验证移动是否成功
            new_param_devices = set()
            for name, param in llm.named_parameters():
                new_param_devices.add(str(param.device))
            
            if len(new_param_devices) > 1:
                logger.error(f"ERROR: Failed to consolidate LLM to single device. Still on: {new_param_devices}")
                # 尝试更激进的方法：逐个移动参数
                logger.warning("Attempting to move parameters individually...")
                for name, param in llm.named_parameters():
                    if param.device != target_device:
                        param.data = param.data.to(target_device)
                # 再次验证
                final_devices = set()
                for name, param in llm.named_parameters():
                    final_devices.add(str(param.device))
                logger.info(f"Final LLM parameter devices after individual move: {final_devices}")
            else:
                logger.info(f"Successfully consolidated LLM to single device: {new_param_devices}")
        else:
            logger.debug(f"LLM parameters are on single device: {param_devices}")
    except Exception as e:
        logger.warning(f"Could not check/consolidate LLM device distribution: {e}")
    
    # 修复 transformers 4.56.0 兼容性问题：修补 prepare_inputs_for_generation 方法
    # 1. 在访问 past_key_values 之前检查它是否为 None 或包含 None
    # 2. 支持 inputs_embeds 参数（transformers 4.56.0 的 LlamaForCausalLM 默认不支持）
    original_prepare_inputs = llm_class.prepare_inputs_for_generation
    def patched_prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 检查 past_key_values 是否为 None 或包含 None 值
        # 如果是，将其设置为 None，避免在原始方法中访问 None.shape 时出错
        if past_key_values is not None:
            try:
                # 尝试访问 past_key_values[0][0] 来检查是否为 None
                if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                    if past_key_values[0] is None:
                        past_key_values = None
                    elif isinstance(past_key_values[0], (list, tuple)) and len(past_key_values[0]) > 0:
                        if past_key_values[0][0] is None:
                            past_key_values = None
            except (IndexError, TypeError, AttributeError):
                # 如果访问失败，也将其设置为 None
                past_key_values = None
        
        # 支持 inputs_embeds：如果提供了 inputs_embeds，且 past_key_values 为 None（第一次生成），
        # 则返回包含 inputs_embeds 的 model_inputs
        if inputs_embeds is not None and past_key_values is None:
            # 第一次生成，使用 inputs_embeds
            model_inputs = {"inputs_embeds": inputs_embeds}
            if "attention_mask" in kwargs:
                model_inputs["attention_mask"] = kwargs["attention_mask"]
            return model_inputs
        
        # 调用原始方法
        try:
            return original_prepare_inputs(self, input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        except (AttributeError, TypeError, ValueError) as e:
            # 如果原始方法不支持 inputs_embeds，且这是第一次生成，手动处理
            if inputs_embeds is not None and past_key_values is None:
                if "inputs_embeds" in str(e) or "forwarding" in str(e).lower():
                    # 返回包含 inputs_embeds 的 model_inputs
                    model_inputs = {"inputs_embeds": inputs_embeds}
                    if "attention_mask" in kwargs:
                        model_inputs["attention_mask"] = kwargs["attention_mask"]
                    return model_inputs
            # 如果仍然出错，尝试强制设置 past_key_values 为 None
            if "NoneType" in str(e) or "shape" in str(e).lower():
                logger.warning(f"Error in prepare_inputs_for_generation: {e}. Forcing past_key_values=None.")
                return original_prepare_inputs(self, input_ids, past_key_values=None, inputs_embeds=inputs_embeds, **kwargs)
            raise
    
    # 替换方法
    llm_class.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
    logger.debug(f"Patched 'prepare_inputs_for_generation' method for {llm_class.__name__} to handle None past_key_values and support inputs_embeds")
    
    # TODO(ligeng): is this necessary for llava?
    config.hidden_size = llm.config.hidden_size
    return llm, tokenizer
