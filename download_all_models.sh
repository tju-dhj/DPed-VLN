#!/bin/bash
# ==============================================================================
# 文件: download_all_models.sh
# 描述: 下载所有Brain模块所需的模型权重
# ==============================================================================
#
# 说明:
# - 下载所有支持的模型权重，包括 Qwen3-VL、Qwen2.5-VL、Gemma4、LLaVA、YOLO
# - 用法:
#   bash download_all_models.sh                # 下载所有模型
#   MODEL_TYPE=qwen3_vl_8b bash download_all_models.sh  # 只下载指定模型
# =============================================================================

# =============================================================================
# 配置
# =============================================================================

# 下载目录（默认：项目根目录下的pretrained_model文件夹）
SAVE_DIR="${SAVE_DIR:-/share/home/u19666033/dhj/DPed_pro/pretrained_model}"
mkdir -p "${SAVE_DIR}"

# 模型缓存目录
CACHE_DIR="${SAVE_DIR}/cache"
mkdir -p "${CACHE_DIR}"

# 要下载的模型类型（默认：全部）
MODEL_TYPE="${MODEL_TYPE:-all}"

# 设置HuggingFace镜像（加速下载）
# HF_ENDPOINT="https://hf-mirror.com"   # 使用镜像（国内推荐）
unset HF_ENDPOINT 2>/dev/null || true

# Python 无缓冲输出，避免长时间下载时终端看起来“卡住”
export PYTHONUNBUFFERED=1

# =============================================================================
# 颜色输出
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 立即打印一行，确认脚本已执行（早于 pip / python 下载）
script_banner_start() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  download_all_models.sh 已启动${NC}  $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}============================================================${NC}"
    log_info "保存目录 SAVE_DIR=${SAVE_DIR}"
    log_info "下载模式 MODEL_TYPE=${MODEL_TYPE}"
    if [ -n "${HF_ENDPOINT}" ]; then
        log_info "HuggingFace 端点 HF_ENDPOINT=${HF_ENDPOINT}"
    else
        log_info "HuggingFace 端点: 默认 (未设置 HF_ENDPOINT)"
    fi
    log_info "提示: 若 pip 出现 'Ignoring invalid distribution ~rpcio'，多为用户目录下损坏包，可忽略或删除对应 ~/.local/.../~rpcio* 目录后重试。"
    echo ""
}

# =============================================================================
# 检查环境
# =============================================================================

check_environment() {
    log_info "【环境】步骤 1/3: 检查 python 命令..."
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    log_info "【环境】当前 Python: $(command -v python)  |  $(python -V 2>&1)"

    log_info "【环境】步骤 2/3: 检查 huggingface_hub / transformers..."
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        log_warning "huggingface_hub 未安装，将执行 pip install（可能需要几分钟，请等待）..."
        pip install -U "huggingface_hub>=0.20" || {
            log_error "pip 安装 huggingface_hub 失败"
            exit 1
        }
    fi
    if ! python -c "import transformers" 2>/dev/null; then
        log_warning "transformers 未安装，将执行 pip install transformers torch（可能需要较长时间，请耐心等待）..."
        pip install transformers torch || {
            log_error "pip 安装 transformers/torch 失败"
            exit 1
        }
    fi

    log_info "【环境】步骤 3/3: 依赖已就绪。"
    log_success "环境检查完成，即将开始下载任务。"
    echo ""
}

# =============================================================================
# 下载函数
# =============================================================================

download_model() {
    local model_id=$1
    local save_path=$2
    local description=$3

    log_info "正在下载: ${description}"
    log_info "  模型ID: ${model_id}"
    log_info "  保存路径: ${save_path}"

    if [ -d "${save_path}" ]; then
        log_warning "模型已存在，跳过: ${save_path}"
        return 0
    fi

    # 创建临时目录
    local temp_dir="${CACHE_DIR}/temp_$(date +%s)"
    mkdir -p "${temp_dir}"
    export _HUB_REPO_ID="${model_id}"
    export _HUB_LOCAL_DIR="${save_path}"
    export _HUB_CACHE_DIR="${CACHE_DIR}"

    python -u << 'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

# 清理空或非法的 Hub URL 环境变量（避免 huggingface_hub 把 repo_id 当 URL 解析）
for _k in ("HF_ENDPOINT", "HUGGINGFACE_HUB_URL"):
    _v = os.environ.get(_k)
    if _v is not None:
        _v = _v.strip()
        if not _v or not _v.lower().startswith(("http://", "https://")):
            os.environ.pop(_k, None)

try:
    print("[HF] 开始 snapshot_download: " + os.environ["_HUB_REPO_ID"], flush=True)
    print("[HF] 目标目录: " + os.environ["_HUB_LOCAL_DIR"], flush=True)
    out = snapshot_download(
        repo_id=os.environ["_HUB_REPO_ID"],
        cache_dir=os.environ["_HUB_CACHE_DIR"],
        local_dir=os.environ["_HUB_LOCAL_DIR"],
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[HF] 完成: " + str(out), flush=True)
except Exception as e:
    print("[HF] 下载失败: " + str(e), file=sys.stderr, flush=True)
    sys.exit(1)
PYEOF

    unset _HUB_REPO_ID _HUB_LOCAL_DIR _HUB_CACHE_DIR 2>/dev/null || true

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        log_success "下载完成: ${description}"
    else
        log_error "下载失败: ${description}"
    fi

    return $exit_code
}

# =============================================================================
# Qwen3-VL 系列下载
# =============================================================================

download_qwen3_vl() {
    log_info "=========================================="
    log_info "开始下载 Qwen3-VL 系列模型"
    log_info "=========================================="

    # Qwen3-VL-2B
    download_model \
        "Qwen/Qwen3-VL-2B-Instruct" \
        "${SAVE_DIR}/Qwen3-VL-2B-Instruct" \
        "Qwen3-VL-2B (轻量级，~4GB)"

    # Qwen3-VL-4B
    download_model \
        "Qwen/Qwen3-VL-4B-Instruct" \
        "${SAVE_DIR}/Qwen3-VL-4B-Instruct" \
        "Qwen3-VL-4B (中量级，~8GB)"

    # Qwen3-VL-8B (推荐)
    download_model \
        "Qwen/Qwen3-VL-8B-Instruct" \
        "${SAVE_DIR}/Qwen3-VL-8B-Instruct" \
        "Qwen3-VL-8B (推荐，~16GB)"

    # Qwen3-VL-32B
    download_model \
        "Qwen/Qwen3-VL-32B-Instruct" \
        "${SAVE_DIR}/Qwen3-VL-32B-Instruct" \
        "Qwen3-VL-32B (大模型，~64GB)"
}

# =============================================================================
# Qwen2.5-VL 系列下载
# =============================================================================

download_qwen25_vl() {
    log_info "=========================================="
    log_info "开始下载 Qwen2.5-VL 系列模型"
    log_info "=========================================="

    # Qwen2.5-VL-3B
    download_model \
        "Qwen/Qwen2.5-VL-3B-Instruct" \
        "${SAVE_DIR}/Qwen2.5-VL-3B-Instruct" \
        "Qwen2.5-VL-3B (稳定版，~6GB)"

    # Qwen2.5-VL-7B
    download_model \
        "Qwen/Qwen2.5-VL-7B-Instruct" \
        "${SAVE_DIR}/Qwen2.5-VL-7B-Instruct" \
        "Qwen2.5-VL-7B (稳定版，~14GB)"

    # Qwen2.5-VL-72B
    download_model \
        "Qwen/Qwen2.5-VL-72B-Instruct" \
        "${SAVE_DIR}/Qwen2.5-VL-72B-Instruct" \
        "Qwen2.5-VL-72B (超大模型，~144GB)"
}

# =============================================================================
# Gemma 4 系列下载
# =============================================================================

download_gemma4() {
    log_info "=========================================="
    log_info "开始下载 Gemma 4 系列模型"
    log_info "=========================================="

    # Gemma-4-E2B
    download_model \
        "google/gemma-4-E2B" \
        "${SAVE_DIR}/gemma-4-E2B" \
        "Gemma-4-E2B (最小显存，~5GB)"

    # Gemma-4-E4B
    download_model \
        "google/gemma-4-E4B" \
        "${SAVE_DIR}/gemma-4-E4B" \
        "Gemma-4-E4B (轻量级，~8GB)"
}

# =============================================================================
# LLaVA 系列下载
# =============================================================================

download_llava() {
    log_info "=========================================="
    log_info "开始下载 LLaVA 系列模型"
    log_info "=========================================="

    # LLaVA-1.6-7B-Mistral
    download_model \
        "llava-hf/llava-v1.6-mistral-7b-hf" \
        "${SAVE_DIR}/llava-v1.6-mistral-7b-hf" \
        "LLaVA-1.6-7B-Mistral (~13GB)"

    # LLaVA-NeXT-7B
    download_model \
        "llava-hf/llava-next-7b-hf" \
        "${SAVE_DIR}/llava-next-7b-hf" \
        "LLaVA-NeXT-7B (最新版，~14GB)"

    # LLaVA-NeXT-34B
    download_model \
        "llava-hf/llava-next-34b-hf" \
        "${SAVE_DIR}/llava-next-34b-hf" \
        "LLaVA-NeXT-34B (大模型，~68GB)"
}

# =============================================================================
# YOLO 检测器下载
# =============================================================================

download_yolo() {
    log_info "=========================================="
    log_info "开始下载 YOLO 行人检测模型"
    log_info "=========================================="

    local yolo_dir="${SAVE_DIR}/yolo_models"
    mkdir -p "${yolo_dir}"

    python -u << EOF
import os
from ultralytics import YOLO

models = {
    "yolov8n-seg": "yolov8n-seg.pt",
    "yolov8s-seg": "yolov8s-seg.pt",
    "yolov8m-seg": "yolov8m-seg.pt",
}

for name, filename in models.items():
    save_path = os.path.join("${yolo_dir}", filename)
    if os.path.exists(save_path):
        print("Model exists, skipping: " + save_path)
    else:
        print("Downloading " + name + "...")
        model = YOLO(name + ".pt")
        # Move to save location
        import shutil
        src = name + ".pt"
        if os.path.exists(src):
            shutil.move(src, save_path)
            print("Saved to: " + save_path)

print("YOLO models download complete!")
EOF

    log_success "YOLO检测器下载完成"
}

# =============================================================================
# 主函数
# =============================================================================

main() {
    script_banner_start

    echo "=============================================="
    echo "       Brain 模块模型下载脚本"
    echo "=============================================="
    echo ""
    echo "保存目录: ${SAVE_DIR}"
    echo "模型类型: ${MODEL_TYPE}"
    echo ""

    # 检查环境
    check_environment

    log_info "【任务】开始执行下载（MODEL_TYPE=${MODEL_TYPE}）..."
    echo ""

    # 根据类型下载
    case "${MODEL_TYPE}" in
        all)
            download_qwen3_vl
            download_qwen25_vl
            download_gemma4
            download_llava
            download_yolo
            ;;
        qwen3_vl|qwen3)
            download_qwen3_vl
            ;;
        qwen25_vl|qwen2.5)
            download_qwen25_vl
            ;;
        gemma|gemma4)
            download_gemma4
            ;;
        llava)
            download_llava
            ;;
        yolo)
            download_yolo
            ;;
        qwen3_vl_8b|qwen3_8b)
            download_model "Qwen/Qwen3-VL-8B-Instruct" "${SAVE_DIR}/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct"
            ;;
        qwen3_vl_32b|qwen3_32b)
            download_model "Qwen/Qwen3-VL-32B-Instruct" "${SAVE_DIR}/Qwen3-VL-32B-Instruct" "Qwen3-VL-32B-Instruct"
            ;;
        qwen25_vl_3b|qwen2.5_3b)
            download_model "Qwen/Qwen2.5-VL-3B-Instruct" "${SAVE_DIR}/Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-3B-Instruct"
            ;;
        qwen25_vl_7b|qwen2.5_7b)
            download_model "Qwen/Qwen2.5-VL-7B-Instruct" "${SAVE_DIR}/Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-7B-Instruct"
            ;;
        qwen25_vl_72b|qwen2.5_72b)
            download_model "Qwen/Qwen2.5-VL-72B-Instruct" "${SAVE_DIR}/Qwen2.5-VL-72B-Instruct" "Qwen2.5-VL-72B-Instruct"
            ;;
        gemma4_e2b)
            download_model "google/gemma-4-E2B" "${SAVE_DIR}/gemma-4-E2B" "Gemma-4-E2B"
            ;;
        gemma4_e4b)
            download_model "google/gemma-4-E4B" "${SAVE_DIR}/gemma-4-E4B" "Gemma-4-E4B"
            ;;
        llava_v1_5_7b)
            download_model "llava-hf/llava-v1.5-7b-hf" "${SAVE_DIR}/llava-v1.5-7b-hf" "LLaVA-1.5-7B"
            ;;
        llava_v1_6_7b)
            download_model "llava-hf/llava-v1.6-mistral-7b-hf" "${SAVE_DIR}/llava-v1.6-mistral-7b-hf" "LLaVA-1.6-7B-Mistral"
            ;;
        llava_next_7b)
            download_model "llava-hf/llava-next-7b-hf" "${SAVE_DIR}/llava-next-7b-hf" "LLaVA-NeXT-7B"
            ;;
        llava_next_34b)
            download_model "llava-hf/llava-next-34b-hf" "${SAVE_DIR}/llava-next-34b-hf" "LLaVA-NeXT-34B"
            ;;
        *)
            log_error "未知模型类型: ${MODEL_TYPE}"
            echo "支持的类型: all, qwen3_vl, qwen25_vl, gemma4, llava, yolo"
            echo "单独指定: qwen3_vl_8b, qwen3_vl_4b, qwen3_vl_2b, qwen3_vl_32b"
            echo "         qwen25_vl_3b, qwen25_vl_7b, qwen25_vl_72b"
            echo "         gemma4_e2b, gemma4_e4b"
            echo "         llava_v1_5_7b, llava_v1_6_7b, llava_next_7b, llava_next_34b"
            exit 1
            ;;
    esac

    echo ""
    log_success "全部下载流程已结束（若某模型已存在则会显示跳过）。"
    echo ""
    echo "=============================================="
    echo "       下载完成！"
    echo "=============================================="
    echo ""
    echo "已下载的模型保存在: ${SAVE_DIR}"
    echo ""
    ls -la "${SAVE_DIR}"
    echo ""
}

# 运行主函数
main "$@"
