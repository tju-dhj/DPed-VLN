#!/bin/bash
# =============================================================================
# VLN-CE 机器人部署 - 服务端启动脚本 (start_server.sh)
# =============================================================================
#
# 用法:
#   # 4动作基础版 (无Brain)
#   bash start_server.sh 4a
#
#   # 4动作 + 8B Brain版
#   bash start_server.sh 4a_8b
#
#   # 6动作基础版 (无Brain)
#   bash start_server.sh 6a
#
#   # 6动作 + 8B Brain版
#   bash start_server.sh 6a_8b
#
#   # 同时启动 frp 内网穿透
#   bash start_server.sh 4a_8b --frp
#
#   # 自定义端口
#   bash start_server.sh 6a_8b --port 32148
#
# =============================================================================

set -euo pipefail

# =============================================================================
# 配置区 — 根据实际情况修改
# =============================================================================

# 项目路径
PROJECT_DIR="/share/home/u19666033/dhj/DPed_pro/habitat-baselines"
CONFIG_DIR="${PROJECT_DIR}/habitat_baselines/config/DPed_brain_new/robot_deploy"
CONDA_ENV="falcon"

# 日志目录
LOG_DIR="/share/home/u19666033/dhj/DPed_pro/robot_client/logs"
mkdir -p "${LOG_DIR}"

# Python 路径
PYTHON_BIN="${CONDA_PREFIX:-/share/home/u19666033/miniconda3/envs/falcon}/bin/python"

# =============================================================================
# 参数解析
# =============================================================================

MODE="${1:-}"
FRP_ENABLED=false
CUSTOM_PORT=""

shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --frp)
            FRP_ENABLED=true
            shift
            ;;
        --port)
            CUSTOM_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# 根据 MODE 选择配置
# =============================================================================

case "${MODE}" in
    4a)
        CONFIG_NAME="DPed_brain_new/robot_deploy/robot_deploy_4a"
        DEFAULT_PORT=32145
        TRAINER="dped_trainer_server"
        DESC="4动作 基础版 (无Brain)"
        ;;
    4a_8b)
        CONFIG_NAME="DPed_brain_new/robot_deploy/robot_deploy_4a_8b"
        DEFAULT_PORT=32146
        TRAINER="dped_brain_trainer_server"
        DESC="4动作 + Qwen3-VL-8B Brain"
        ;;
    6a)
        CONFIG_NAME="DPed_brain_new/robot_deploy/robot_deploy_6a"
        DEFAULT_PORT=32147
        TRAINER="dped_trainer_server"
        DESC="6动作 基础版 (无Brain)"
        ;;
    6a_8b)
        CONFIG_NAME="DPed_brain_new/robot_deploy/robot_deploy_6a_8b"
        DEFAULT_PORT=32148
        TRAINER="dped_brain_trainer_server"
        DESC="6动作 + Qwen3-VL-8B Brain"
        ;;
    *)
        echo ""
        echo "错误: 请指定启动模式"
        echo ""
        echo "用法: bash start_server.sh <mode> [--frp] [--port NNNNN]"
        echo ""
        echo "可用模式:"
        echo "  4a        4动作基础版 (无Brain)  → port 32145"
        echo "  4a_8b     4动作 + Qwen3-VL-8B    → port 32146"
        echo "  6a        6动作基础版 (无Brain)  → port 32147"
        echo "  6a_8b     6动作 + Qwen3-VL-8B    → port 32148"
        echo ""
        echo "选项:"
        echo "  --frp     启动后同时拉起 frp 内网穿透"
        echo "  --port N  自定义端口覆盖默认值"
        echo ""
        exit 1
        ;;
esac

PORT="${CUSTOM_PORT:-${DEFAULT_PORT}}"
LOG_FILE="${LOG_DIR}/server_${MODE}_$(date +%Y%m%d_%H%M%S).log"

# =============================================================================
# 预检查
# =============================================================================

echo ""
echo "=============================================================="
echo "  VLN-CE 机器人部署 - 服务端启动"
echo "=============================================================="
echo "  模式:         ${MODE} (${DESC})"
echo "  配置:         ${CONFIG_NAME}"
echo "  Trainer:      ${TRAINER}"
echo "  端口:         ${PORT}"
echo "  内网穿透:     ${FRP_ENABLED}"
echo "  日志:         ${LOG_FILE}"
echo "=============================================================="
echo ""

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo "警告: conda 未找到，将直接使用 python"
    PYTHON_BIN="python"
fi

# 检查配置文件
if [[ ! -d "${CONFIG_DIR}" ]]; then
    echo "错误: 配置目录不存在: ${CONFIG_DIR}"
    exit 1
fi

# 检查 checkpoint 路径 (在对应 YAML 中)
YAML_FILE="${CONFIG_DIR}/robot_deploy_${MODE}.yaml"
if [[ ! -f "${YAML_FILE}" ]]; then
    echo "错误: YAML 配置文件不存在: ${YAML_FILE}"
    exit 1
fi

# 提取 checkpoint 路径（可选，仅用于提示）
CKPT_PATH=$(grep "eval_ckpt_path_dir:" "${YAML_FILE}" | awk '{print $2}' | tr -d '"')
echo "  Checkpoint: ${CKPT_PATH}"
if [[ -n "${CKPT_PATH}" ]] && [[ ! -f "${CKPT_PATH}" ]]; then
    echo "  ！！！警告: checkpoint 文件不存在，请检查 YAML 中的 eval_ckpt_path_dir"
    echo "  服务器将尝试加载，如果失败会报错"
fi
echo ""

# =============================================================================
# 环境变量
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
export GLOG_minloglevel="2"
export OMP_NUM_THREADS="4"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FLASK_SERVER_PORT="${PORT}"

# 避免 NCCL 副作用
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-600}"

# =============================================================================
# 启动 FRP 内网穿透（如果启用）
# =============================================================================

if ${FRP_ENABLED}; then
    FRPC_CONFIG="/share/home/u19666033/dhj/DPed_pro/robot_client/frpc.toml"
    FRPC_LOG="/share/home/u19666033/dhj/DPed_pro/robot_client/frpc_${MODE}.log"

    if command -v frpc &> /dev/null; then
        echo "[FRP] 启动 frpc 内网穿透..."
        nohup frpc -c "${FRPC_CONFIG}" > "${FRPC_LOG}" 2>&1 &
        FRPC_PID=$!
        echo "[FRP] frpc 进程 PID: ${FRPC_PID}"
        echo "[FRP] 日志: ${FRPC_LOG}"
        echo ""
        sleep 2
    else
        echo "警告: frpc 未安装，跳过内网穿透"
        echo "  安装方法:"
        echo "    wget https://github.com/fatedier/frp/releases/download/v0.61.0/frp_0.61.0_linux_amd64.tar.gz"
        echo "    tar -xzf frp_0.61.0_linux_amd64.tar.gz -C /opt/"
        echo "    ln -s /opt/frp_0.61.0_linux_amd64/frpc /usr/local/bin/frpc"
    fi
fi

# =============================================================================
# 启动 VLN-CE Flask 服务
# =============================================================================

echo "[Server] 启动 VLN-CE Flask HTTP Server..."
echo "[Server] 配置: ${CONFIG_NAME}"
echo "[Server] 端口: ${PORT}"
echo "[Server] 日志: ${LOG_FILE}"
echo ""
echo "提示: 按 Ctrl+C 停止服务"
echo ""

cd "${PROJECT_DIR}"

# 前台运行（方便查看日志）
if command -v conda &> /dev/null; then
    conda run -n "${CONDA_ENV}" --no-capture-output \
        python -u -m habitat_baselines.run \
        --config-name="${CONFIG_NAME}" \
        2>&1 | tee "${LOG_FILE}"
else
    python -u -m habitat_baselines.run \
        --config-name="${CONFIG_NAME}" \
        2>&1 | tee "${LOG_FILE}"
fi

# =============================================================================
# 清理
# =============================================================================

if ${FRP_ENABLED} && [[ -n "${FRPC_PID:-}" ]]; then
    echo "[FRP] 停止 frpc (PID: ${FRPC_PID})"
    kill "${FRPC_PID}" 2>/dev/null || true
fi

echo "[Server] VLN-CE Flask 服务已停止"
