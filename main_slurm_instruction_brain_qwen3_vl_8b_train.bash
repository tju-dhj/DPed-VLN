#!/bin/bash
# ==============================================================================
# 文件: main_slurm_instruction_brain_qwen3_vl_8b_train.bash
# 描述: Qwen3-VL-8B Brain训练SLURM启动脚本（全量训练模式）
# ==============================================================================

# """
# InstructionBrain + Qwen3-VL-8B 全量训练脚本
# ============================================

# 训练模式：全量训练（Brain模型参数冻结）
# - freeze_brain: true  - 冻结Brain模型
# - freeze_pedestrian: true - 冻结行人检测器
# - train_encoder: false - 冻结CLIP编码器

# 硬件要求：
# - 显存: 16GB+ (用于Qwen3-VL-8B)
# - 内存: 32GB+
# - 建议使用A800或更高配置
# """

# =============================================================================
# SLURM配置
# =============================================================================

#SBATCH --job-name=inst_brain_qwen3_vl_8b
#SBATCH --output=slurm_logs/inst_brain_qwen3_vl_8b/%j_%x.out
#SBATCH --error=slurm_logs/inst_brain_qwen3_vl_8b/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a800:1
#SBATCH --time=120:00:00
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

# 输出路径
TENSORBOARD_DIR="${PROJECT_ROOT}/tb_logs/instruction_brain_qwen3_vl_8b_train"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/instruction_brain_qwen3_vl_8b_train"

mkdir -p "${TENSORBOARD_DIR}" "${CHECKPOINT_DIR}"

# =============================================================================
# 打印配置
# =============================================================================

echo "=============================================="
echo "  InstructionBrain + Qwen3-VL-8B 全量训练"
echo "=============================================="
echo "模型:        Qwen3-VL-8B-Instruct"
echo "训练模式:    全量训练（参数冻结）"
echo "freeze_brain: true"
echo "freeze_pedestrian: true"
echo "train_encoder: false"
echo "=============================================="

# =============================================================================
# 构建训练命令
# =============================================================================

TRAIN_CMD="python -u -m habitat_baselines.habitat_baselines.run \
    --config-name=habitat_baselines/config/DPed_brain/instruction_brain_qwen3_vl_8b_train.yaml \
    \
    habitat_baselines.trainer_name=instruction_brain_ppo_trainer \
    habitat_baselines.num_environments=4 \
    habitat_baselines.total_num_steps=50000000 \
    \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
    \
    habitat_baselines.brain.enabled=true \
    habitat_baselines.brain.instruction_mode=true \
    habitat_baselines.brain.max_history_frames=5 \
    \
    habitat_baselines.brain.model_type=qwen3_vl_8b \
    habitat_baselines.brain.model_id=Qwen/Qwen3-VL-8B-Instruct \
    habitat_baselines.brain.model_path=${PRETRAINED_MODEL_DIR}/Qwen3-VL-8B-Instruct \
    \
    habitat_baselines.brain.pedestrian_enabled=true \
    habitat_baselines.brain.pedestrian_detector=yolov8n \
    habitat_baselines.brain.pedestrian_ckpt_path=/share/home/u19666033/dhj/DPed_pro/pedestrian_benchmark/assets/checkpoints/yolov8n-seg.pt \
    \
    habitat_baselines.brain.freeze_brain=true \
    habitat_baselines.brain.freeze_pedestrian=true \
    habitat_baselines.brain.freeze_clip=true \
    \
    habitat_baselines.rl.ddppo.train_encoder=false"

# =============================================================================
# 执行训练
# =============================================================================

echo ""
echo "开始训练..."
echo ""
echo "命令:"
echo "${TRAIN_CMD}"
echo ""

eval ${TRAIN_CMD}

echo ""
echo "训练完成!"
