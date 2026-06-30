#!/bin/bash
# ==============================================================================
# 文件: download_qwen2_5_vl_3b.sh
# 描述: 下载Qwen2.5-VL-3B模型（轻量级推荐）
# ==============================================================================
#
# 说明:
# - 下载 Qwen2.5-VL-3B-Instruct，约 3B 参数、磁盘约 6GB，建议 8GB+ 显存
# - 用法: bash download_qwen2_5_vl_3b.sh
# =============================================================================

# =============================================================================
# 配置
# =============================================================================

SAVE_DIR="/share/home/u19666033/dhj/DPed_pro/pretrained_model"
mkdir -p "${SAVE_DIR}"

MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen2.5-VL-3B-Instruct"
MODEL_PATH="${SAVE_DIR}/${MODEL_NAME}"

# =============================================================================
# 颜色
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# 下载
# =============================================================================

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Qwen2.5-VL-3B-Instruct 下载${NC}"
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
