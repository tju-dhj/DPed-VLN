#!/bin/bash
# ==============================================================================
# 文件: download_qwen3_vl_8b.sh
# 描述: 下载Qwen3-VL-8B模型（推荐用于测试）
# ==============================================================================
#
# 中文说明（不要用 """ 包在 bash 里，会导致脚本解析失败、无输出或立即退出）：
# - 下载 Qwen3-VL-8B-Instruct，约 8B 参数、磁盘约十几 GB，建议 16GB+ 显存用于推理
# - 用法: bash download_qwen3_vl_8b.sh
#
# =============================================================================
# 配置
# =============================================================================

# 下载目录
SAVE_DIR="/share/home/u19666033/dhj/DPed_pro/pretrained_model"
mkdir -p "${SAVE_DIR}"

# 模型信息
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-8B-Instruct"
MODEL_PATH="${SAVE_DIR}/${MODEL_NAME}"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# 下载函数
# =============================================================================

download_model() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}  Qwen3-VL-8B-Instruct 模型下载${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "模型ID: ${GREEN}${MODEL_ID}${NC}"
    echo -e "保存路径: ${GREEN}${MODEL_PATH}${NC}"
    echo ""

    # 检查是否已存在
    if [ -d "${MODEL_PATH}" ]; then
        echo -e "${YELLOW}模型已存在，跳过下载:${NC}"
        ls -la "${MODEL_PATH}" | head -10
        return 0
    fi

    # 检查磁盘空间
    echo -e "${BLUE}检查磁盘空间...${NC}"
    available_space=$(df -h "${SAVE_DIR}" | tail -1 | awk '{print $4}')
    echo -e "可用空间: ${available_space}"

    echo ""
    echo -e "${YELLOW}开始下载（首次运行可能需要30-60分钟）...${NC}"
    echo ""

    # 设置环境变量
    export HF_HOME="${SAVE_DIR}/hf_cache"
    export TRANSFORMERS_CACHE="${SAVE_DIR}/hf_cache"
    mkdir -p "${HF_HOME}"

    # 下载模型
    python << EOF
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

print("Downloading ${MODEL_ID}...")
print("This may take 30-60 minutes on slow connections...")
print()

try:
    save_path = snapshot_download(
        repo_id="${MODEL_ID}",
        cache_dir="${HF_HOME}",
        local_dir="${MODEL_PATH}",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print()
    print(f"Successfully downloaded to: {save_path}")
except Exception as e:
    print(f"Download failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}============================================${NC}"
        echo -e "${GREEN}  下载完成！${NC}"
        echo -e "${GREEN}============================================${NC}"
        echo ""
        echo "模型路径: ${MODEL_PATH}"
        echo ""
        ls -la "${MODEL_PATH}" | head -15
        echo ""
    else
        echo ""
        echo -e "${RED}============================================${NC}"
        echo -e "${RED}  下载失败！${NC}"
        echo -e "${RED}============================================${NC}"
        exit 1
    fi
}

# =============================================================================
# 主函数
# =============================================================================

main() {
    download_model
}

main "$@"
