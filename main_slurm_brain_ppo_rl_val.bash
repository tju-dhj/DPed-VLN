#!/bin/bash
# ==============================================================================
# 文件: main_slurm_brain_ppo_rl_val.bash
# 描述: Brain增强PPO评估SLURM启动脚本
# ==============================================================================

"""
BrainPPOEvaluator SLURM启动脚本
================================

该脚本用于在SLURM集群上提交Brain增强的VLN PPO评估任务。

使用示例：
```bash
# 评估checkpoint
bash main_slurm_brain_ppo_rl_val.bash \
    --checkpoint /path/to/checkpoint.pth \
    --num_episodes 100

# 禁用Brain评估
bash main_slurm_brain_ppo_rl_val.bash \
    --checkpoint /path/to/checkpoint.pth \
    --brain_disabled
```
"""

# =============================================================================
# SLURM参数配置
# =============================================================================

#SBATCH --job-name=brain_ppo_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/brain_ppo_eval_%j.out
#SBATCH --error=logs/brain_ppo_eval_%j.err

# =============================================================================
# 环境配置
# =============================================================================

module purge
module load anaconda3/2023.09
module load cuda/11.8

# =============================================================================
# 路径配置
# =============================================================================

PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/habitat-baselines:${PYTHONPATH}"

DATA_ROOT="${PROJECT_ROOT}/data"
HF_HOME="${DATA_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/brain_ppo_eval"
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# 评估参数配置
# =============================================================================

# 检查点路径 (必需)
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}

# 评估配置
NUM_EPISODES=${NUM_EPISODES:-100}
SPLIT=${SPLIT:-"val_seen"}              # val_seen, val_unseen

# Brain模块配置
BRAIN_ENABLED=${BRAIN_ENABLED:-true}
GEMMA_MODEL=${GEMMA_MODEL:-"gemma4_e2b"}
PEDESTRIAN_DETECTOR=${PEDESTRIAN_DETECTOR:-"yolov8n"}

# =============================================================================
# 命令行参数解析
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --brain_disabled)
            BRAIN_ENABLED="false"
            shift
            ;;
        --gemma_model)
            GEMMA_MODEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH      模型检查点路径 (必需)"
            echo "  --num_episodes NUM     评估episode数 (默认: 100)"
            echo "  --split SPLIT          数据集分割 (默认: val_seen)"
            echo "  --brain_disabled       禁用Brain模块"
            echo "  --gemma_model MODEL    Gemma模型 (默认: gemma4_e2b)"
            echo "  --help, -h            显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "错误: 必须指定 --checkpoint 参数"
    exit 1
fi

# =============================================================================
# 打印配置
# =============================================================================

echo "=================================================="
echo "       BrainPPO 评估任务配置"
echo "=================================================="
echo "检查点:        ${CHECKPOINT_PATH}"
echo "Episode数:     ${NUM_EPISODES}"
echo "数据集分割:    ${SPLIT}"
echo "Brain模块:     ${BRAIN_ENABLED}"
echo "Gemma模型:    ${GEMMA_MODEL}"
echo "行人检测器:   ${PEDESTRIAN_DETECTOR}"
echo "=================================================="

# =============================================================================
# 构建评估命令
# =============================================================================

EVAL_CMD="python -m habitat_baselines.run \
    --config-name=habitat_baselines/config/DPed_brain/brain_ppo_rl_val.yaml \
    \
    habitat_baselines.checkpoint_folder=${CHECKPOINT_PATH} \
    habitat_baselines.num_evaluation_episodes=${NUM_EPISODES} \
    habitat.dataset.split=${SPLIT} \
    \
    habitat_baselines.brain.enabled=${BRAIN_ENABLED} \
    habitat_baselines.brain.model_type=${GEMMA_MODEL} \
    habitat_baselines.brain.pedestrian_detector=${PEDESTRIAN_DETECTOR} \
    habitat_baselines.brain.eval_with_brain=${BRAIN_ENABLED}"

# =============================================================================
# 执行评估
# =============================================================================

cd "${PROJECT_ROOT}"

echo ""
echo "开始评估..."
echo "命令: ${EVAL_CMD}"
echo ""

eval ${EVAL_CMD}

# =============================================================================
# 评估完成
# =============================================================================

echo ""
echo "=================================================="
echo "       评估任务完成"
echo "=================================================="
echo "结束时间: $(date)"
echo "输出目录: ${OUTPUT_DIR}"
echo "=================================================="
