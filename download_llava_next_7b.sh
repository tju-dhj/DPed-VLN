#!/bin/bash
# ==============================================================================
# 文件: download_llava_next_7b.sh
# 描述: 下载 LLaVA-NeXT-7B 模型（最新版，约 14GB）
# ==============================================================================

SAVE_DIR="/share/home/u19666033/dhj/DPed_pro/pretrained_model"
mkdir -p "${SAVE_DIR}"

MODEL_ID="llava-hf/llava-next-7b-hf"
MODEL_NAME="llava-next-7b-hf"
MODEL_PATH="${SAVE_DIR}/${MODEL_NAME}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  LLaVA-NeXT-7B 模型下载${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "模型ID: ${GREEN}${MODEL_ID}${NC}"
echo -e "保存路径: ${GREEN}${MODEL_PATH}${NC}"
echo ""

if [ -d "${MODEL_PATH}" ]; then
    echo -e "${YELLOW}模型已存在，跳过下载:${NC}"
    ls -la "${MODEL_PATH}" | head -10
    exit 0
fi

echo -e "${BLUE}检查磁盘空间...${NC}"
available_space=$(df -h "${SAVE_DIR}" | tail -1 | awk '{print $4}')
echo -e "可用空间: ${available_space}"
echo ""
echo -e "${YELLOW}开始下载（首次运行可能需要 20-50 分钟）...${NC}"
echo ""

export HF_HOME="${SAVE_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${SAVE_DIR}/hf_cache"
mkdir -p "${HF_HOME}"

python -u << EOF
import os
import sys
from huggingface_hub import snapshot_download

print("Downloading ${MODEL_ID}...", flush=True)

try:
    save_path = snapshot_download(
        repo_id="${MODEL_ID}",
        cache_dir="${HF_HOME}",
        local_dir="${MODEL_PATH}",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Successfully downloaded to: " + str(save_path), flush=True)
except Exception as e:
    print("Download failed: " + str(e), file=sys.stderr, flush=True)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  下载完成！${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "模型路径: ${MODEL_PATH}"
    echo ""
    ls -la "${MODEL_PATH}" | head -15
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  下载失败！${NC}"
    echo -e "${RED}============================================${NC}"
    exit 1
fi
