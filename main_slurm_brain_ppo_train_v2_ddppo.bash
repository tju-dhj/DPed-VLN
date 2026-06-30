#!/bin/bash
# ==============================================================================
# 文件: main_slurm_brain_ppo_train_v2_ddppo.bash
# 描述: Brain增强PPO训练SLURM启动脚本
# ==============================================================================

"""
BrainPPOTrainer SLURM启动脚本
==============================

该脚本用于在SLURM集群上提交Brain增强的VLN PPO训练任务。

功能：
1. 分布式DDPPO训练支持
2. 可配置的GPU数量和节点数
3. Brain模块参数配置
4. 行人检测器配置
5. 自动下载Gemma模型

使用示例：
```bash
# 标准训练
bash main_slurm_brain_ppo_train_v2_ddppo.bash

# 自定义配置
bash main_slurm_brain_ppo_train_v2_ddppo.bash \
    --gpus 8 \
    --nodes 2 \
    --brain_enabled \
    --gemma_model gemma4_e2b
```

作者: DPed_pro Team
日期: 2026-04
"""

# =============================================================================
# SLURM参数配置
# =============================================================================

# 作业名称
#SBATCH --job-name=brain_ppo_train

# 资源请求
#SBATCH --nodes=1                      # 节点数量 (默认1)
#SBATCH --ntasks-per-node=8            # 每节点任务数
#SBATCH --gres=gpu:8                   # 每节点GPU数量 (默认8)
#SBATCH --cpus-per-task=10             # 每任务CPU数量

# 时间限制
#SBATCH --time=72:00:00               # 最大运行时间 (默认72小时)

# 内存配置
#SBATCH --mem=128G                     # 每节点内存 (默认128GB)

# 输出和错误日志
#SBATCH --output=logs/brain_ppo_train_%j.out
#SBATCH --error=logs/brain_ppo_train_%j.err

# 邮件通知 (可选)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@example.com

# 集群分区
#SBATCH --partition=compute           # 计算分区
#SBATCH --account=your_account         # 项目账户

# =============================================================================
# 环境加载
# =============================================================================

# 加载必要的模块
module purge                           # 清除已加载模块
module load anaconda3/2023.09          # Anaconda环境
module load cuda/11.8                  # CUDA工具包
module load cudnn/v8                   # cuDNN库

# 激活conda环境
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate habitat

# =============================================================================
# 路径配置
# =============================================================================

# 项目根目录
export PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"

# 数据集路径
export DATA_ROOT="${PROJECT_ROOT}/data"
export DATASET_PATH="${DATA_ROOT}/datasets/dynamic_vlnce_v1"

# 模型缓存路径
export HF_HOME="${DATA_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}"

# 输出路径
export OUTPUT_ROOT="${PROJECT_ROOT}/outputs/brain_ppo_train"
export TB_LOG_DIR="${PROJECT_ROOT}/tb_logs/brain_ppo_train"
export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/brain_ppo_train"

# 创建必要的目录
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${TB_LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${HF_HOME}"

# =============================================================================
# Python路径配置
# =============================================================================

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/habitat-baselines:${PYTHONPATH}"

# =============================================================================
# Brain模块配置
# =============================================================================

# Brain模块开关
BRAIN_ENABLED=${BRAIN_ENABLED:-true}

# Gemma模型配置
GEMMA_MODEL=${GEMMA_MODEL:-"gemma4_e2b"}     # 可选: gemma4_e2b, gemma4_e4b
GEMMA_MODEL_ID=${GEMMA_MODEL_ID:-""}          # 可选：自定义HuggingFace模型ID

# 行人检测器配置
PEDESTRIAN_DETECTOR=${PEDESTRIAN_DETECTOR:-"yolov8n"}  # 可选: yolov8n, yolov8s, yolov8m, rtdetr_r18, rtdetr_r50
PEDESTRIAN_CONFIDENCE=${PEDESTRIAN_CONFIDENCE:-0.25}
PEDESTRIAN_CKPT_PATH=${PEDESTRIAN_CKPT_PATH:-""}       # YOLO checkpoint路径

# Brain决策配置
BRAIN_OVERRIDE_THRESHOLD=${BRAIN_OVERRIDE_THRESHOLD:-0.8}

# 冻结配置
FREEZE_BRAIN=${FREEZE_BRAIN:-true}
FREEZE_PEDESTRIAN=${FREEZE_PEDESTRIAN:-true}

# =============================================================================
# 训练参数配置
# =============================================================================

# 分布式训练配置
DISTRIBUTED=${DISTRIBUTED:-true}

# 训练步数
TOTAL_NUM_STEPS=${TOTAL_NUM_STEPS:-50000000}

# 并行环境数
NUM_ENVIRONMENTS=${NUM_ENVIRONMENTS:-8}

# 学习率
LEARNING_RATE=${LEARNING_RATE:-0.0001}

# Epoch数量
NUM_UPDATES=${NUM_UPDATES:-100000}

# 检查点保存间隔
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100}

# =============================================================================
# SLURM环境变量 (分布式训练必需)
# =============================================================================

# 获取SLURM分配信息
export SLURM_JOBID=${SLURM_JOBID:-$$}
export WORLD_SIZE=${SLURM_NTASKS:-1}
export RANK=${SLURM_PROCID:-0}
export LOCAL_RANK=${SLURM_LOCALID:-0}
export MASTER_ADDR=${SLURM_JOB_NODELIST:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}

# =============================================================================
# CUDA配置
# =============================================================================

# 设置可见GPU
if [ -n "${SLURM_JOB_GPUS}" ]; then
    export CUDA_VISIBLE_DEVICES="${SLURM_GPUS}"
else
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
fi

# PyTorch CUDA配置
export TORCH_CUDA_ARCH_LIST="8.0"        # Ampere架构
export NCCL_DEBUG=INFO                   # NCCL调试信息
export NCCL_IB_DISABLE=0                # 启用InfiniBand

# =============================================================================
# 命令行参数解析
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            SBATCH_GPUS="$2"
            shift 2
            ;;
        --nodes)
            SBATCH_NODES="$2"
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
        --ped_detector)
            PEDESTRIAN_DETECTOR="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --total_steps)
            TOTAL_NUM_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus NUM              GPU数量 (默认: 8)"
            echo "  --nodes NUM              节点数量 (默认: 1)"
            echo "  --brain_disabled         禁用Brain模块"
            echo "  --gemma_model MODEL      Gemma模型 (默认: gemma4_e2b)"
            echo "  --ped_detector DETECTOR  行人检测器 (默认: yolov8n)"
            echo "  --lr RATE               学习率 (默认: 0.0001)"
            echo "  --total_steps STEPS      总训练步数 (默认: 50000000)"
            echo "  --help, -h              显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# 打印配置信息
# =============================================================================

echo "=================================================="
echo "       BrainPPO 训练任务配置"
echo "=================================================="
echo "作业ID:         ${SLURM_JOBID:-'本地运行'}"
echo "节点数:         ${SLURM_NTASKS:-1}"
echo "GPU数量:        ${SLURM_GPUS:-'8'}"
echo "------------------------------------------"
echo "Brain模块:      ${BRAIN_ENABLED}"
echo "Gemma模型:      ${GEMMA_MODEL}"
echo "行人检测器:     ${PEDESTRIAN_DETECTOR}"
echo "行人置信度:     ${PEDESTRIAN_CONFIDENCE}"
echo "------------------------------------------"
echo "总训练步数:     ${TOTAL_NUM_STEPS}"
echo "并行环境数:     ${NUM_ENVIRONMENTS}"
echo "学习率:         ${LEARNING_RATE}"
echo "------------------------------------------"
echo "输出目录:       ${OUTPUT_ROOT}"
echo "检查点目录:     ${CHECKPOINT_DIR}"
echo "=================================================="

# =============================================================================
# 构建训练命令
# =============================================================================

TRAIN_CMD="python -m habitat_baselines.run \
    --config-name=habitat_baselines/config/DPed_brain/brain_ppo_train_v2_ddppo.yaml \
    \
    habitat_baselines.trainer_name=brain_ppo_trainer \
    habitat_baselines.num_environments=${NUM_ENVIRONMENTS} \
    habitat_baselines.total_num_steps=${TOTAL_NUM_STEPS} \
    habitat_baselines.rl.ppo.lr=${LEARNING_RATE} \
    habitat_baselines.checkpoint.save_interval=${CHECKPOINT_INTERVAL} \
    \
    habitat_baselines.tensorboard_dir=${TB_LOG_DIR} \
    \
    habitat_baselines.brain.enabled=${BRAIN_ENABLED} \
    habitat_baselines.brain.pedestrian_enabled=true \
    habitat_baselines.brain.pedestrian_detector=${PEDESTRIAN_DETECTOR} \
    habitat_baselines.brain.pedestrian_confidence=${PEDESTRIAN_CONFIDENCE} \
    habitat_baselines.brain.model_type=${GEMMA_MODEL} \
    habitat_baselines.brain.brain_device=cuda \
    habitat_baselines.brain.override_threshold=${BRAIN_OVERRIDE_THRESHOLD} \
    habitat_baselines.brain.freeze_brain=${FREEZE_BRAIN} \
    habitat_baselines.brain.freeze_pedestrian=${FREEZE_PEDESTRIAN} \
    habitat_baselines.brain.freeze_clip=true"

# 添加可选参数
if [ -n "${GEMMA_MODEL_ID}" ]; then
    TRAIN_CMD="${TRAIN_CMD} habitat_baselines.brain.model_id=${GEMMA_MODEL_ID}"
fi

if [ -n "${PEDESTRIAN_CKPT_PATH}" ]; then
    TRAIN_CMD="${TRAIN_CMD} habitat_baselines.brain.pedestrian_ckpt_path=${PEDESTRIAN_CKPT_PATH}"
fi

# =============================================================================
# 执行训练
# =============================================================================

cd "${PROJECT_ROOT}"

echo ""
echo "开始训练..."
echo "命令: ${TRAIN_CMD}"
echo ""

# 执行训练
eval ${TRAIN_CMD}

# =============================================================================
# 训练完成
# =============================================================================

echo ""
echo "=================================================="
echo "       训练任务完成"
echo "=================================================="
echo "作业ID: ${SLURM_JOBID:-'N/A'}"
echo "结束时间: $(date)"
echo "=================================================="
