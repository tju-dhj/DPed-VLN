# This file is modified from https://github.com/haotian-liu/LLaVA/
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import shutil
import warnings

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PretrainedConfig

from ..constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from . import *
from .utils import is_mm_model


def _patch_transformers_quantization_compatibility():
    """
    修复 transformers 4.56.0 与量化模型的兼容性问题。
    1. set_module_quantized_tensor_to_device 函数在某些情况下不接受 fp16_statistics 参数
    2. dispatch_model 会尝试调用 model.to()，但量化模型不支持
    """
    try:
        patches_applied = []
        
        # 补丁1: 修补 set_module_quantized_tensor_to_device
        # 这个补丁对于 4-bit 和 8-bit 量化都是必需的
        try:
            import transformers.modeling_utils as modeling_utils
            if hasattr(modeling_utils, 'set_module_quantized_tensor_to_device'):
                original_func = modeling_utils.set_module_quantized_tensor_to_device
                
                # 检查是否已经被补丁（避免重复补丁）
                is_already_patched = hasattr(original_func, '__wrapped__')
                
                if not is_already_patched:
                    import inspect
                    try:
                        sig = inspect.signature(original_func)
                        if 'fp16_statistics' not in sig.parameters:
                            def patched_func(*args, **kwargs):
                                # 移除 fp16_statistics 参数（如果存在）
                                kwargs.pop('fp16_statistics', None)
                                return original_func(*args, **kwargs)
                            # 保存原始函数引用
                            patched_func.__wrapped__ = original_func
                            modeling_utils.set_module_quantized_tensor_to_device = patched_func
                            patches_applied.append('set_module_quantized_tensor_to_device')
                    except Exception:
                        # 如果无法检查签名，直接应用补丁
                        def patched_func(*args, **kwargs):
                            kwargs.pop('fp16_statistics', None)
                            return original_func(*args, **kwargs)
                        patched_func.__wrapped__ = original_func
                        modeling_utils.set_module_quantized_tensor_to_device = patched_func
                        patches_applied.append('set_module_quantized_tensor_to_device (no sig)')
                else:
                    patches_applied.append('set_module_quantized_tensor_to_device (already patched)')
        except Exception as e:
            warnings.warn(f"Failed to patch set_module_quantized_tensor_to_device: {e}")
        
        # 补丁1b: 也修补 integrations 模块（如果函数是从那里导入的）
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
                    patches_applied.append('integrations.set_module_quantized_tensor_to_device')
        except Exception:
            pass
        
        # 补丁2: 修补 accelerate 的 dispatch_model，避免对量化模型调用 model.to()
        try:
            from accelerate import big_modeling
            original_dispatch_model = big_modeling.dispatch_model
            
            def patched_dispatch_model(model, device_map=None, **kwargs):
                # 首先检查是否是量化模型（无论 device_map 是什么）
                # 方法1: 检查模型属性
                is_quantized = (
                    hasattr(model, 'hf_quantizer') or 
                    getattr(model, 'is_loaded_in_8bit', False) or 
                    getattr(model, 'is_loaded_in_4bit', False)
                )
                
                # 方法2: 检查模型参数（量化参数通常有 SCB 属性）
                if not is_quantized:
                    try:
                        # 只检查前几个参数，避免遍历所有参数（可能很慢）
                        param_iter = iter(model.named_parameters())
                        for _ in range(10):  # 只检查前10个参数
                            name, param = next(param_iter)
                            if hasattr(param, 'SCB') or (hasattr(param, 'data') and hasattr(param.data, 'SCB')):
                                is_quantized = True
                                break
                    except (StopIteration, Exception):
                        pass
                
                # 方法3: 检查模型类名或模块名（量化模型可能有特殊标识）
                if not is_quantized:
                    try:
                        model_class_name = model.__class__.__name__
                        if '8bit' in model_class_name.lower() or '4bit' in model_class_name.lower():
                            is_quantized = True
                    except Exception:
                        pass
                
                # 如果是量化模型，直接返回，避免调用 model.to()
                if is_quantized:
                    # 量化模型已经自动加载到正确的设备，不需要 dispatch
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
            
            big_modeling.dispatch_model = patched_dispatch_model
            patches_applied.append('dispatch_model')
        except Exception as e:
            warnings.warn(f"Failed to patch dispatch_model: {e}")
        
        # 补丁2b: 直接修补 PreTrainedModel.to 方法，对量化模型直接返回
        try:
            import transformers.modeling_utils as modeling_utils
            if hasattr(modeling_utils, 'PreTrainedModel'):
                original_to = modeling_utils.PreTrainedModel.to
                
                def patched_to(self, *args, **kwargs):
                    # 检查是否是量化模型
                    is_quantized = (
                        hasattr(self, 'hf_quantizer') or 
                        getattr(self, 'is_loaded_in_8bit', False) or 
                        getattr(self, 'is_loaded_in_4bit', False)
                    )
                    
                    if is_quantized:
                        # 量化模型不支持 .to()，直接返回自身
                        return self
                    
                    # 非量化模型，使用原始方法
                    return original_to(self, *args, **kwargs)
                
                modeling_utils.PreTrainedModel.to = patched_to
                patches_applied.append('PreTrainedModel.to')
        except Exception as e:
            warnings.warn(f"Failed to patch PreTrainedModel.to: {e}")
        
        # 补丁3: 修补 transformers.integrations.bitsandbytes.get_keys_to_not_convert，修复 IndexError
        try:
            from transformers.integrations import bitsandbytes
            if hasattr(bitsandbytes, 'get_keys_to_not_convert'):
                original_get_keys = bitsandbytes.get_keys_to_not_convert
                
                def patched_get_keys_to_not_convert(model):
                    try:
                        return original_get_keys(model)
                    except IndexError:
                        # 如果 list_modules 为空，返回空列表
                        return []
                
                bitsandbytes.get_keys_to_not_convert = patched_get_keys_to_not_convert
                patches_applied.append('bitsandbytes.get_keys_to_not_convert')
        except Exception as e:
            warnings.warn(f"Failed to patch get_keys_to_not_convert: {e}")
        
        # 补丁4: 修补 transformers.models.auto.auto_factory.AutoModelForCausalLM.from_pretrained
        # 只对 AutoModelForCausalLM 应用补丁，避免影响其他类（如 MultimodalProjector）
        try:
            from transformers.models.auto import auto_factory
            if hasattr(auto_factory, 'AutoModelForCausalLM'):
                original_auto_from_pretrained = auto_factory.AutoModelForCausalLM.from_pretrained
                
                @classmethod
                def patched_auto_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                    # 检查是否是量化加载
                    is_quantized = kwargs.get('load_in_8bit', False) or kwargs.get('load_in_4bit', False)
                    
                    if is_quantized:
                        # 量化模型：移除 device_map 相关参数，避免 dispatch_model 调用
                        # 但保留其他参数，让 bitsandbytes 正常工作
                        kwargs.pop('device_map', None)
                        kwargs.pop('max_memory', None)
                    
                    # 调用原始方法
                    return original_auto_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                
                auto_factory.AutoModelForCausalLM.from_pretrained = patched_auto_from_pretrained
                patches_applied.append('AutoModelForCausalLM.from_pretrained')
        except Exception as e:
            warnings.warn(f"Failed to patch AutoModelForCausalLM.from_pretrained: {e}")
        
        if patches_applied:
            warnings.warn(f"Applied quantization compatibility patches: {', '.join(patches_applied)}")
    except Exception as e:
        warnings.warn(f"Failed to patch transformers quantization compatibility: {e}")


# 在导入时应用补丁
_patch_transformers_quantization_compatibility()


def load_pretrained_model(
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    # 处理量化：当使用量化时，完全不传递 device_map 参数
    # 这样可以避免 transformers 4.56.0 的兼容性问题（set_module_quantized_tensor_to_device 不接受 fp16_statistics）
    # 以及避免 dispatch_model 尝试调用 model.to()（量化模型不支持）
    if load_8bit or load_4bit:
        # 量化时，完全不传递 device_map 参数，让量化库自动处理设备分配
        # 注意：不传递 device_map（而不是传递 None），以避免 transformers 调用 dispatch_model
        # 量化库会自动将模型加载到 CUDA 设备
        if "device_map" in kwargs:
            del kwargs["device_map"]
        if "max_memory" in kwargs:
            del kwargs["max_memory"]
    else:
        # 非量化时，正常处理 device_map
        if device_map is not None:
            kwargs = {"device_map": device_map, **kwargs}
        
        if device != "cuda":
            kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
        # 量化模型仍然需要 torch_dtype 用于某些计算
        kwargs["torch_dtype"] = torch.float16
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # 量化模型仍然需要 torch_dtype 用于某些计算
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16
        # kwargs["torch_dtype"] = torch.bfloat16

    if is_mm_model(model_path):
        # Load LLaVA model
        ## TODO @yunhao: mind fixing lora
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged."
            )
        if ("lora" in model_name.lower() or "dora" in model_name.lower()) and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            print(lora_cfg_pretrained)
            print("Loading LLaVA from base model...")
            config = AutoConfig.from_pretrained(model_base)
            prepare_config_for_eval(config, kwargs)
            model = LlavaLlamaModel.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
            tokenizer = model.tokenizer
            token_num, tokem_dim = model.llm.lm_head.out_features, model.llm.lm_head.in_features
            if model.llm.lm_head.weight.shape[0] != token_num:
                model.llm.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )
                model.llm.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        ## TODO @yunhao: mind fixing this
        elif model_base is not None:
            # this may be mm projector only
            print("Loading LLaVA from base model...")
            cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            mm_config_wrapper(config, kwargs)
            if "mpt" in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, "configuration_mpt.py")):
                    shutil.copyfile(
                        os.path.join(model_base, "configuration_mpt.py"),
                        os.path.join(model_path, "configuration_mpt.py"),
                    )
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, legacy=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs
                )
        else:
            config = AutoConfig.from_pretrained(model_path)
            config.resume_path = model_path
            prepare_config_for_eval(config, kwargs)
            if "mpt" in model_name.lower():
                model = LlavaMPTForCausalLM.from_pretrained(model_path, config=config, low_cpu_mem_usage=True, **kwargs)
            elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path, config=config, low_cpu_mem_usage=True, **kwargs
                )
            elif "gemma" in model_name.lower():
                model = LlavaGemmaForCausalLM.from_pretrained(
                    model_path, config=config, low_cpu_mem_usage=True, **kwargs
                )
            else:
                # kentang-mit@: llama-2 model
                # config._attn_implementation = "flash_attention_2"
                model = LlavaLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
            tokenizer = model.tokenizer
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    model.eval()
    
    # 对于量化模型，确保模型在正确的设备上
    # bitsandbytes 通常会自动处理，但为了确保，我们检查一下
    if (load_8bit or load_4bit) and device == "cuda":
        # 量化模型应该已经在 CUDA 上，但我们可以检查第一个参数的位置
        try:
            first_param = next(model.parameters()) if hasattr(model, 'parameters') else None
            if first_param is not None and first_param.device.type != "cuda":
                # 如果不在 CUDA 上，尝试移动到 CUDA（虽然量化模型通常不需要）
                warnings.warn(f"Quantized model parameter is on {first_param.device}, expected cuda")
        except Exception:
            pass  # 忽略检查错误，量化模型可能有特殊的参数结构
    
    image_processor = None
    if is_mm_model(model_path):
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=device, dtype=torch.float16)
        # vision_tower.to(device=device, dtype=torch.bfloat16)
        mm_projector = model.get_mm_projector()
        mm_projector.to(device=device, dtype=torch.float16)
        # mm_projector.to(device=device, dtype=torch.bfloat16)
        image_processor = vision_tower.image_processor

    if hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def parse_model_name_or_path(config: PretrainedConfig, model_name="llm", suffix="_cfg"):
    target_model = f"{model_name}{suffix}"
    target_cfg = getattr(config, target_model, None)

    if isinstance(target_cfg, str):
        return target_cfg
    elif isinstance(target_cfg, dict):
        return target_cfg["architectures"][0]
    else:
        raise ValueError(f"Invalid {target_model} configuration!")


def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")

    # 获取 torch_dtype，如果不存在则使用默认值 float16
    torch_dtype = kwargs.pop("torch_dtype", torch.float16)
    config.model_dtype = torch_dtype.__str__()
