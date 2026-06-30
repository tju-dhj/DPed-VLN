#!/bin/bash
# ==============================================================================
# 文件: download_glm_v.sh
# 描述: 下载GLM-V系列视觉语言模型（本地部署版）
# ==============================================================================
#
# 支持的模型:
# - GLM-4.6V (106B) - 高性能版，需要多卡
# - GLM-4.6V-Flash (9B) - 轻量高速版，适合单卡
# - GLM-4.6V-FP8 - FP8量化版
# - GLM-4.5V - 平衡版
# - GLM-4.1V-9B-Thinking - 推理优化版（推荐）
#
# 模型下载来源:
# - HuggingFace: https://huggingface.co/zai-org
# - ModelScope: https://modelscope.cn/models/ZhipuAI
#
# 用法:
#   bash download_glm_v.sh <model_name>
#   例如: bash download_glm_v.sh glm-4.6v-flash
#
# =============================================================================

SAVE_DIR="/share/home/u19666033/dhj/DPed_pro/pretrained_model"
mkdir -p "${SAVE_DIR}"

# GLM-V模型映射表
declare -A GLM_MODELS
GLM_MODELS["glm-4.6v"]="zai-org/GLM-4.6V"
GLM_MODELS["glm-4.6v-fp8"]="zai-org/GLM-4.6V-FP8"
GLM_MODELS["glm-4.6v-flash"]="zai-org/GLM-4.6V-Flash"
GLM_MODELS["glm-4.5v"]="zai-org/GLM-4.5V"
GLM_MODELS["glm-4.5v-fp8"]="zai-org/GLM-4.5V-FP8"
GLM_MODELS["glm-4.1v-9b-thinking"]="zai-org/GLM-4.1V-9B-Thinking"
GLM_MODELS["glm-4.1v-9b-base"]="zai-org/GLM-4.1V-9B-Base"

# 模型大小描述（估算）
declare -A MODEL_SIZES
MODEL_SIZES["glm-4.6v"]="~60GB（需要4卡或多卡部署）"
MODEL_SIZES["glm-4.6v-fp8"]="~30GB（需要4卡部署）"
MODEL_SIZES["glm-4.6v-flash"]="~20GB（单卡可跑）"
MODEL_SIZES["glm-4.5v"]="~40GB（需要多卡部署）"
MODEL_SIZES["glm-4.5v-fp8"]="~20GB（单卡可跑）"
MODEL_SIZES["glm-4.1v-9b-thinking"]="~20GB（单卡可跑，推荐）"
MODEL_SIZES["glm-4.1v-9b-base"]="~20GB（单卡可跑）"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  GLM-V 系列模型下载脚本${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}用法: bash download_glm_v.sh <model_name>${NC}"
    echo ""
    echo -e "${CYAN}支持的模型:${NC}"
    echo "--------------------------------------------"
    for key in "${!GLM_MODELS[@]}"; do
        size=${MODEL_SIZES[$key]:-"未知"}
        printf "  %-30s %s\n" "$key" "$size"
    done
    echo "--------------------------------------------"
    echo ""
    echo -e "${GREEN}示例: bash download_glm_v.sh glm-4.6v-flash${NC}"
    echo -e "${GREEN}       bash download_glm_v.sh glm-4.1v-9b-thinking${NC}"
    exit 1
fi

MODEL_KEY=$1

# 检查模型是否支持
if [[ ! -v "GLM_MODELS[$MODEL_KEY]" ]]; then
    echo -e "${RED}错误: 不支持的模型 '$MODEL_KEY'${NC}"
    echo ""
    echo -e "${CYAN}支持的模型:${NC}"
    echo "--------------------------------------------"
    for key in "${!GLM_MODELS[@]}"; do
        size=${MODEL_SIZES[$key]:-"未知"}
        printf "  %-30s %s\n" "$key" "$size"
    done
    echo "--------------------------------------------"
    exit 1
fi

MODEL_ID=${GLM_MODELS[$MODEL_KEY]}
MODEL_PATH="${SAVE_DIR}/${MODEL_KEY}"
MODEL_SIZE=${MODEL_SIZES[$MODEL_KEY]:-"未知"}

echo -e "${CYAN}模型: ${MODEL_KEY}${NC}"
echo -e "${CYAN}HuggingFace ID: ${MODEL_ID}${NC}"
echo -e "${CYAN}保存路径: ${MODEL_PATH}${NC}"
echo -e "${CYAN}模型大小: ${MODEL_SIZE}${NC}"
echo ""

# 检查是否已存在
if [ -d "${MODEL_PATH}" ]; then
    echo -e "${GREEN}✓ 模型已存在，跳过下载${NC}"
    echo ""
    echo "目录内容:"
    ls -la "${MODEL_PATH}" | head -10
    echo ""
    echo -e "${GREEN}如需重新下载，请先删除目录: rm -rf ${MODEL_PATH}${NC}"
    exit 0
fi

# 设置缓存目录
export HF_HOME="${SAVE_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

echo -e "${YELLOW}开始下载...${NC}"
echo ""

# 下载模型
python << EOF
from huggingface_hub import snapshot_download
import sys

model_id = "${MODEL_ID}"
model_path = "${MODEL_PATH}"
cache_dir = "${HF_HOME}"

print(f"正在下载模型: {model_id}")
print(f"缓存目录: {cache_dir}")
print(f"保存目录: {model_path}")
print("-" * 50)

try:
    save_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=model_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("-" * 50)
    print(f"下载完成！")
    print(f"保存路径: {save_path}")
except Exception as e:
    print(f"下载失败: {e}")
    sys.exit(1)
EOF

DOWNLOAD_STATUS=$?

if [ $DOWNLOAD_STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  下载完成！${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "目录内容:"
    ls -la "${MODEL_PATH}" | head -15
    echo ""
    echo -e "${CYAN}接下来可以修改配置文件使用此模型:${NC}"
    echo "  brain.model_type: glm_4_6v_flash"
    echo "  brain.model_path: ${MODEL_PATH}"
else
    echo ""
    echo -e "${RED}下载失败，请检查网络连接和模型ID是否正确${NC}"
    exit 1
fi