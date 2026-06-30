#!/bin/bash
# ==============================================================================
# 文件: download_gemma4_e2b.sh
# 描述: 下载Gemma-4-E2B模型（最小显存）
# ==============================================================================
#
# 说明:
# - 下载 Gemma-4-E2B，约 2.3B 有效参数、磁盘约 5GB，建议 8GB+ 显存
# - 用法: bash download_gemma4_e2b.sh
# =============================================================================

SAVE_DIR="/share/home/u19666033/dhj/DPed_pro/pretrained_model"
mkdir -p "${SAVE_DIR}"

MODEL_ID="google/gemma-4-E2B"
MODEL_NAME="gemma-4-E2B"
MODEL_PATH="${SAVE_DIR}/${MODEL_NAME}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Gemma-4-E2B 下载${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

if [ -d "${MODEL_PATH}" ]; then
    echo -e "${GREEN}模型已存在，跳过下载${NC}"
    ls -la "${MODEL_PATH}" | head -5
    exit 0
fi

export HF_HOME="${SAVE_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

echo "开始下载..."
python << EOF
from huggingface_hub import snapshot_download

print("Downloading ${MODEL_ID}...")
save_path = snapshot_download(
    repo_id="${MODEL_ID}",
    cache_dir="${HF_HOME}",
    local_dir="${MODEL_PATH}",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("Downloaded to: " + save_path)
EOF

echo ""
echo -e "${GREEN}下载完成！${NC}"
ls -la "${MODEL_PATH}" | head -10
