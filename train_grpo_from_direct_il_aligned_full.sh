#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
CONFIG_NAME="DPed_pro/new_data/v2/train_grpo_from_il_aligned_full.yaml"
LOG_FILE="${PROJECT_ROOT}/evaluation-vlnce-dpedpro/grpo/direct_il_aligned_full/hm3d/launcher.log"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate falcon
fi

mkdir -p "$(dirname "${LOG_FILE}")"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "========================================"
echo "Train GRPO from direct IL checkpoint"
echo "========================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "CONFIG_NAME  : ${CONFIG_NAME}"
echo "GPU          : ${CUDA_VISIBLE_DEVICES}"
echo "LOG_FILE     : ${LOG_FILE}"
echo "========================================"

python -u habitat-baselines/habitat_baselines/run.py \
    --config-name="${CONFIG_NAME}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
