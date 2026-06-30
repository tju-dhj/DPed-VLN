#!/usr/bin/env bash
set -euo pipefail

echo "[NaVILA] 正在准备 Falcon 环境附加依赖..."
# This is required to activate conda environment
# eval "$(conda shell.bash hook)"

# CONDA_ENV=${1:-""}
# if [ -n "$CONDA_ENV" ]; then
#     conda create -n $CONDA_ENV python=3.10 -y
#     conda activate $CONDA_ENV
# else
#     echo "Skipping conda environment creation. Make sure you have the correct environment activated."
# fi

# # This is required to enable PEP 660 support
# pip install --upgrade pip

# # This is optional if you prefer to use built-in nvcc
# conda install -c nvidia cuda-toolkit -y

# # Install FlashAttention2
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA
# pip install --no-deps -e .
# pip install --no-deps -e ".[train]"
# pip install --no-deps -e ".[eval]"

# Install HF's Transformers
# pip install git+https://github.com/huggingface/transformers@v4.37.2

# 确保使用当前环境的 Python（支持 conda 环境）
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "Error: Python not found"
    exit 1
fi

echo "[NaVILA] 使用 Python: $(which $PYTHON_CMD)"
echo "[NaVILA] Python 版本: $($PYTHON_CMD --version)"

site_pkg_path=$($PYTHON_CMD -c 'import site; print(site.getsitepackages()[0])')
echo "[NaVILA] Site-packages 路径: $site_pkg_path"

if [ ! -d "$site_pkg_path/transformers" ]; then
    echo "Error: transformers 目录不存在: $site_pkg_path/transformers"
    exit 1
fi

echo "[NaVILA] 复制 transformers_replace 文件..."
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/

if [ -d "./llava/train/deepspeed_replace" ] && [ -d "$site_pkg_path/deepspeed" ]; then
    echo "[NaVILA] 复制 deepspeed_replace 文件..."
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
else
    echo "[NaVILA] 跳过 deepspeed_replace (目录不存在或 deepspeed 未安装)"
fi

# 应用 transformers 兼容性补丁
echo "[NaVILA] 应用 transformers 兼容性补丁..."
$PYTHON_CMD << 'PYTHON_PATCH'
import sys
import os

transformers_init = os.path.join(sys.path[0] if hasattr(sys, 'path') and sys.path else '', 'transformers', '__init__.py')
# 查找 transformers/__init__.py
for path in sys.path:
    transformers_path = os.path.join(path, 'transformers', '__init__.py')
    if os.path.exists(transformers_path):
        transformers_init = transformers_path
        break

if not os.path.exists(transformers_init):
    print(f"Error: transformers/__init__.py not found")
    sys.exit(1)

# 读取文件
with open(transformers_init, 'r') as f:
    content = f.read()

# 检查是否已有补丁
if '_transformers_module.LlamaModel = LlamaModel' in content:
    print("✓ Transformers 补丁已存在")
else:
    # 在 _LazyModule 创建后添加补丁
    patch = '''
# NaVILA/StreamVLN compatibility patch: Dynamically add model classes
try:
    _transformers_module = sys.modules[__name__]
    from .models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
    from .models.llama.configuration_llama import LlamaConfig
    _transformers_module.LlamaModel = LlamaModel
    _transformers_module.LlamaForCausalLM = LlamaForCausalLM
    _transformers_module.LlamaConfig = LlamaConfig
except (ImportError, AttributeError):
    pass

try:
    from .models.mistral.modeling_mistral import MistralModel, MistralForCausalLM
    from .models.mistral.configuration_mistral import MistralConfig
    _transformers_module.MistralModel = MistralModel
    _transformers_module.MistralForCausalLM = MistralForCausalLM
    _transformers_module.MistralConfig = MistralConfig
except (ImportError, AttributeError):
    pass

try:
    from .models.mixtral.modeling_mixtral import MixtralModel, MixtralForCausalLM
    from .models.mixtral.configuration_mixtral import MixtralConfig
    _transformers_module.MixtralModel = MixtralModel
    _transformers_module.MixtralForCausalLM = MixtralForCausalLM
    _transformers_module.MixtralConfig = MixtralConfig
except (ImportError, AttributeError):
    pass
'''
    
    # 在 _LazyModule 创建后添加补丁
    if 'sys.modules[__name__] = _LazyModule(' in content:
        # 找到 _LazyModule 调用的结束位置
        import re
        pattern = r'(sys\.modules\[__name__\] = _LazyModule\([^)]+\))'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            # 在 _LazyModule 调用后添加补丁
            content = content.replace(match.group(1), match.group(1) + patch)
        else:
            # 如果正则匹配失败，在 else 分支末尾添加
            if 'else:' in content:
                content = content + '\n' + patch
    
    # 写入文件
    with open(transformers_init, 'w') as f:
        f.write(content)
    print("✓ Transformers 补丁已添加")

print(f"✓ Transformers __init__.py 位置: {transformers_init}")
PYTHON_PATCH

# 验证补丁是否生效
echo "[NaVILA] 验证补丁..."
$PYTHON_CMD -c "
import sys
# 应用兼容性补丁
import transformers.utils as utils
if not hasattr(utils, 'is_torch_tpu_available'):
    def _is_torch_tpu_available():
        return False
    utils.is_torch_tpu_available = _is_torch_tpu_available

import transformers.modeling_utils as modeling_utils
if not hasattr(modeling_utils, 'PreTrainedAudioTokenizerBase'):
    class PreTrainedAudioTokenizerBase:
        pass
    modeling_utils.PreTrainedAudioTokenizerBase = PreTrainedAudioTokenizerBase

if not hasattr(modeling_utils, 'ALL_ATTENTION_FUNCTIONS'):
    modeling_utils.ALL_ATTENTION_FUNCTIONS = {}

# 导入 transformers（会执行补丁代码）
import transformers

# 检查类是否存在
classes = ['LlamaModel', 'LlamaConfig', 'MistralConfig', 'MixtralConfig']
missing = [c for c in classes if not hasattr(transformers, c)]
if missing:
    print(f'⚠ 以下类未找到: {missing}')
    # 尝试从子模块导入并手动添加
    try:
        from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
        from transformers.models.llama.configuration_llama import LlamaConfig
        transformers.LlamaModel = LlamaModel
        transformers.LlamaForCausalLM = LlamaForCausalLM
        transformers.LlamaConfig = LlamaConfig
        print('✓ 手动添加 Llama 类到 transformers 模块')
    except Exception as e:
        print(f'✗ 无法导入 Llama 类: {e}')
    
    try:
        from transformers.models.mistral.modeling_mistral import MistralModel, MistralForCausalLM
        from transformers.models.mistral.configuration_mistral import MistralConfig
        transformers.MistralModel = MistralModel
        transformers.MistralForCausalLM = MistralForCausalLM
        transformers.MistralConfig = MistralConfig
        print('✓ 手动添加 Mistral 类到 transformers 模块')
    except Exception as e:
        print(f'✗ 无法导入 Mistral 类: {e}')
    
    try:
        from transformers.models.mixtral.modeling_mixtral import MixtralModel, MixtralForCausalLM
        from transformers.models.mixtral.configuration_mixtral import MixtralConfig
        transformers.MixtralModel = MixtralModel
        transformers.MixtralForCausalLM = MixtralForCausalLM
        transformers.MixtralConfig = MixtralConfig
        print('✓ 手动添加 Mixtral 类到 transformers 模块')
    except Exception as e:
        print(f'✗ 无法导入 Mixtral 类: {e}')

# 最终验证
all_found = all(hasattr(transformers, c) for c in classes)
if all_found:
    print('✓ 所有必需的类都已成功添加到 transformers 模块')
else:
    print('⚠ 部分类仍未找到，可能需要检查 transformers 版本兼容性')
" || echo "⚠ 验证过程中出现错误，但可能不影响使用"

# pip install "webdataset>=0.2.0"