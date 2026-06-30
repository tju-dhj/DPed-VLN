#!/bin/bash
# create_qwen_env.sh - 一键创建 Qwen3.6-27B 推理环境

set -e  # 遇到错误立即退出

# ========== 配置区域（可根据需要修改） ==========
ENV_NAME="vllm_qwen"          # 新环境名称（避免覆盖原有 vllm）
PYTHON_VERSION="3.10"         # Python 版本（必须 >= 3.10）
CUDA_VERSION="cu121"          # CUDA 版本：cu118 / cu121 / cu124
MODEL_PATH="/share/home/u19666033/dhj/models/Qwen3.6-27B"
# ==============================================

echo "🔧 Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
conda activate ${ENV_NAME}

echo "📦 Installing PyTorch with ${CUDA_VERSION}..."
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

echo "📦 Installing core dependencies..."
pip install transformers==4.57.1 \
            accelerate==0.34.0 \
            tokenizers==0.20.0 \
            safetensors==0.4.5 \
            huggingface_hub==0.26.0 \
            sentencepiece==0.2.0 \
            protobuf==4.25.0 \
            numpy==1.26.4 \
            psutil  # accelerate 依赖

echo "✅ Installing additional utilities..."
pip install tqdm rich  # 用于进度条和美化输出（可选）

echo "🎯 Environment setup complete. Verifying..."
python -c "
import sys, torch, transformers, accelerate
print(f'✅ Python: {sys.version.split()[0]}')
print(f'✅ PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ Accelerate: {accelerate.__version__}')
print(f'✅ GPU Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ bfloat16 Support: {torch.cuda.is_bf16_supported()}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "🎉 SUCCESS! Environment '${ENV_NAME}' is ready."
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: conda activate ${ENV_NAME}"
echo "2. (Optional) Run patch script if needed: python patch_qwen3_5.py"
echo "3. Clear old cache: find \$CONDA_PREFIX/lib/python*/site-packages/transformers/models/qwen* -name '*.pyc' -delete 2>/dev/null || true"
echo "4. Test model loading: python generate_l1_instructions.py --dry-run --max-files 1"
echo "5. Run full inference: python generate_l1_instructions.py --batch-size 4"
echo ""
echo "⚠️  If you encounter 'PreTrainedConfig' errors:"
echo "   - Ensure transformers==4.57.1 is installed"
echo "   - Run your patch script to fix configuration_qwen3_5.py"
echo "   - Clear __pycache__ directories"