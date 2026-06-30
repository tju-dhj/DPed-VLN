#!/bin/bash
# ==============================================================================
# 文件: main_slurm_instruction_brain_qwen3_vl_8b_rl_val.bash
# 描述: Qwen3-VL-8B Brain评估SLURM启动脚本
# ==============================================================================

"""
InstructionBrain + Qwen3-VL-8B 评估脚本
=========================================

评估使用预训练checkpoint的模型效果。
"""

# =============================================================================
# SLURM配置
# =============================================================================

#SBATCH --job-name=inst_brain_qwen3_vl_8b_val
#SBATCH --output=slurm_logs/inst_brain_qwen3_vl_8b_val/%j_%x.out
#SBATCH --error=slurm_logs/inst_brain_qwen3_vl_8b_val/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a800:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G

# =============================================================================
# 环境配置
# =============================================================================

source /share/home/u14004/.bashrc
conda activate falcon

# =============================================================================
# 路径配置
# =============================================================================

PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/habitat-baselines:${PYTHONPATH}"

# 模型路径
PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_model"
export HF_HOME="${PRETRAINED_MODEL_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# checkpoint路径（需要指定）
CHECKPOINT_PATH="/share/home/u19666033/dhj/DPed_pro/checkpoints/instruction_brain_qwen3_vl_8b_train/ckpt.XX.pth"

# 输出路径
TENSORBOARD_DIR="${PROJECT_ROOT}/tb_logs/instruction_brain_qwen3_vl_8b_val"
VIDEO_DIR="${PROJECT_ROOT}/video_logs/instruction_brain_qwen3_vl_8b_val"

mkdir -p "${TENSORBOARD_DIR}" "${VIDEO_DIR}"

# =============================================================================
# 参数解析
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --episodes)
            EPISODE_COUNT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

EPISODE_COUNT="${EPISODE_COUNT:--1}"

# =============================================================================
# 打印配置
# =============================================================================

echo "=============================================="
echo "  InstructionBrain + Qwen3-VL-8B 评估"
echo "=============================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Episode数量: ${EPISODE_COUNT}"
echo "=============================================="

# =============================================================================
# 构建评估命令
# =============================================================================

EVAL_CMD="python -u -m habitat_baselines.habitat_baselines.run \
    --config-name=habitat_baselines/config/DPed_brain/instruction_brain_qwen3_vl_8b_rl_val.yaml \
    \
    habitat_baselines.evaluate=true \
    habitat_baselines.test_episode_count=${EPISODE_COUNT} \
    \
    habitat_baselines.eval_ckpt_path_dir=${CHECKPOINT_PATH} \
    habitat_baselines.checkpoint_folder=${PROJECT_ROOT}/checkpoints/instruction_brain_qwen3_vl_8b_val \
    \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.video_dir=${VIDEO_DIR} \
    \
    habitat_baselines.brain.enabled=true \
    habitat_baselines.brain.model_type=qwen3_vl_8b \
    habitat_baselines.brain.model_path=${PRETRAINED_MODEL_DIR}/Qwen3-VL-8B-Instruct \
    habitat_baselines.brain.pedestrian_enabled=true \
    habitat_baselines.brain.freeze_brain=true"

# =============================================================================
# 执行评估
# =============================================================================

echo ""
echo "开始评估..."
echo ""
echo "命令:"
echo "${EVAL_CMD}"
echo ""

eval ${EVAL_CMD}

echo ""
echo "评估完成!"
