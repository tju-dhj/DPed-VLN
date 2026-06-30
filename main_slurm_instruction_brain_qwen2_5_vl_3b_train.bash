#!/bin/bash
# ==============================================================================
# 文件: main_slurm_instruction_brain_qwen2_5_vl_3b_train.bash
# 描述: Qwen2.5-VL-3B Brain训练SLURM启动脚本
# ==============================================================================

#SBATCH --job-name=inst_brain_qwen25_3b
#SBATCH --output=slurm_logs/inst_brain_qwen25_3b/%j_%x.out
#SBATCH --error=slurm_logs/inst_brain_qwen25_3b/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a800:1
#SBATCH --time=120:00:00
#SBATCH --mem=64G

source /share/home/u14004/.bashrc
conda activate falcon

PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/habitat-baselines:${PYTHONPATH}"

PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_model"
export HF_HOME="${PRETRAINED_MODEL_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/instruction_brain_qwen2_5_vl_3b_train"
TENSORBOARD_DIR="${PROJECT_ROOT}/tb_logs/instruction_brain_qwen2_5_vl_3b_train"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/instruction_brain_qwen2_5_vl_3b_train"

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

echo "=============================================="
echo "  InstructionBrain + Qwen2.5-VL-3B 训练"
echo "=============================================="
echo "模型:        Qwen2.5-VL-3B-Instruct"
echo "=============================================="

TRAIN_CMD="python -u -m habitat_baselines.habitat_baselines.run \
    --config-name=habitat_baselines/config/DPed_brain/instruction_brain_qwen2_5_vl_3b_train.yaml \
    \
    habitat_baselines.trainer_name=instruction_brain_ppo_trainer \
    habitat_baselines.num_environments=4 \
    habitat_baselines.total_num_steps=50000000 \
    \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
    \
    habitat_baselines.brain.enabled=true \
    habitat_baselines.brain.model_type=qwen2_5_vl_3b \
    habitat_baselines.brain.model_id=Qwen/Qwen2.5-VL-3B-Instruct \
    habitat_baselines.brain.model_path=${PRETRAINED_MODEL_DIR}/Qwen2.5-VL-3B-Instruct \
    habitat_baselines.brain.freeze_brain=true"

eval ${TRAIN_CMD}

echo ""
echo "训练完成!"
