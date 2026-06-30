#!/usr/bin/env python3
"""
Transformers 兼容性补丁模块

在 transformers 4.56.0 中，某些模型类（LlamaModel, MistralConfig 等）可能不在顶层导出。
此模块在导入时自动应用补丁，确保这些类可以从 transformers 顶层导入。
"""

import sys
import importlib


def _apply_transformers_compat_patch():
    """应用 transformers 兼容性补丁"""
    # 确保 transformers.utils 有必需的函数
    try:
        import transformers.utils as utils
        if not hasattr(utils, 'is_torch_tpu_available'):
            def _is_torch_tpu_available():
                return False
            utils.is_torch_tpu_available = _is_torch_tpu_available
    except ImportError:
        pass

    # 确保 transformers.modeling_utils 有必需的类
    try:
        import transformers.modeling_utils as modeling_utils
        if not hasattr(modeling_utils, 'PreTrainedAudioTokenizerBase'):
            class PreTrainedAudioTokenizerBase:
                pass
            modeling_utils.PreTrainedAudioTokenizerBase = PreTrainedAudioTokenizerBase

        if not hasattr(modeling_utils, 'ALL_ATTENTION_FUNCTIONS'):
            modeling_utils.ALL_ATTENTION_FUNCTIONS = {}
    except ImportError:
        pass

    # 导入 transformers 模块
    try:
        import transformers
    except ImportError:
        return

    # 动态添加缺失的模型类到 transformers 模块
    _add_model_classes_to_transformers(transformers)


def _add_model_classes_to_transformers(transformers_module):
    """将模型类添加到 transformers 模块的命名空间"""
    classes_to_add = {
        'LlamaModel': ('models.llama.modeling_llama', 'LlamaModel'),
        'LlamaForCausalLM': ('models.llama.modeling_llama', 'LlamaForCausalLM'),
        'LlamaConfig': ('models.llama.configuration_llama', 'LlamaConfig'),
        'MistralModel': ('models.mistral.modeling_mistral', 'MistralModel'),
        'MistralForCausalLM': ('models.mistral.modeling_mistral', 'MistralForCausalLM'),
        'MistralConfig': ('models.mistral.configuration_mistral', 'MistralConfig'),
        'MixtralModel': ('models.mixtral.modeling_mixtral', 'MixtralModel'),
        'MixtralForCausalLM': ('models.mixtral.modeling_mixtral', 'MixtralForCausalLM'),
        'MixtralConfig': ('models.mixtral.configuration_mixtral', 'MixtralConfig'),
    }

    for class_name, (module_path, attr_name) in classes_to_add.items():
        if not hasattr(transformers_module, class_name):
            try:
                full_module_path = f'transformers.{module_path}'
                module = importlib.import_module(full_module_path)
                class_obj = getattr(module, attr_name, None)
                if class_obj is not None:
                    setattr(transformers_module, class_name, class_obj)
            except (ImportError, AttributeError):
                # 静默失败，某些类可能不存在
                pass


# 自动应用补丁
_apply_transformers_compat_patch()

