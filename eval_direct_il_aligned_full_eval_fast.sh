#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
CONFIG_NAME="DPed_pro/new_data/eval/dped_eval_4a_direct_il_aligned_full_eval_fast.yaml"
CKPT_PATH="/share/home/u19666033/dhj/DPed_pro/evaluation-vlnce-dpedpro/dynamic_vlnce_clip_direct_il_aligned_full/hm3d/checkpoints/ckpt.epoch_6.step_170190.pth"
OUTPUT_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vlnce-dpedpro/dynamic_vlnce_clip_direct_il_aligned_full/hm3d/eval_fast/checkpoints"
LOG_DIR="/share/home/u19666033/dhj/DPed_pro/evaluation-vlnce-dpedpro/dynamic_vlnce_clip_direct_il_aligned_full/hm3d/eval_fast"
LOG_FILE="${LOG_DIR}/eval_ckpt.epoch_6.step_170190.log"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate falcon
fi

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

echo "========================================"
echo "Eval direct IL aligned_full checkpoint"
echo "========================================"
echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "CONFIG_NAME  : ${CONFIG_NAME}"
echo "CKPT_PATH    : ${CKPT_PATH}"
echo "OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "LOG_FILE     : ${LOG_FILE}"
echo "GPU          : ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

python -u habitat-baselines/habitat_baselines/run.py \
    --config-name="${CONFIG_NAME}" \
    habitat_baselines.eval_ckpt_path_dir="${CKPT_PATH}" \
    habitat_baselines.checkpoint_folder="${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"
