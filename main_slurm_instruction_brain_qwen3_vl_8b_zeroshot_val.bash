#!/bin/bash
# ==============================================================================
# 文件: main_slurm_instruction_brain_qwen3_vl_8b_zeroshot_val.bash
# 描述: Qwen3-VL-8B Zero-Shot评估SLURM启动脚本
# ==============================================================================

"""
Qwen3-VL-8B Zero-Shot评估脚本
===============================

使用预训练的Qwen3-VL-8B模型和之前RL训练出来的网络权重进行zero-shot评估。

配置说明：
- 预训练Qwen3-VL-8B模型: pretrained_model/Qwen3-VL-8B-Instruct
- YOLOv8n行人检测器: pretrained_model/yolov8n-seg.pt
- RL训练网络权重: evaluation-vln/dynamic_vlnce_clip_rl_v2_ddppo/hm3d/checkpoints_vln_new/ckpt.37.pth

核心功能：
- 每次调用大模型时打印prompt
- 保存历史帧图像到文件夹
- 保存prompt记录到文件
"""

# =============================================================================
# SLURM配置
# ==============================================================================

#SBATCH --job-name=inst_brain_qwen3_vl_8b_zeroshot_val
#SBATCH --output=slurm_logs/inst_brain_qwen3_vl_8b_zeroshot_val/%j_%x.out
#SBATCH --error=slurm_logs/inst_brain_qwen3_vl_8b_zeroshot_val/%j_%x.err
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
# ==============================================================================

source /share/home/u14004/.bashrc
conda activate falcon

# =============================================================================
# 路径配置
# ==============================================================================

PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/habitat-baselines:${PYTHONPATH}"

# 模型路径
PRETRAINED_MODEL_DIR="${PROJECT_ROOT}/pretrained_model"
export HF_HOME="${PRETRAINED_MODEL_DIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ============================================================
# 【关键配置】权重文件路径
# ============================================================

# 【1】RL训练网络权重路径（用于zero-shot评估的策略网络）
RL_CKPT_PATH="${PROJECT_ROOT}/evaluation-vln/dynamic_vlnce_clip_rl_v2_ddppo/hm3d/checkpoints_vln_new/ckpt.37.pth"

# 【2】预训练Qwen3-VL-8B模型路径
QWEN_MODEL_PATH="${PRETRAINED_MODEL_DIR}/Qwen3-VL-8B-Instruct"

# 【3】YOLOv8n行人检测权重路径
YOLO_CKPT_PATH="${PRETRAINED_MODEL_DIR}/yolov8n-seg.pt"

# 输出路径
TENSORBOARD_DIR="${PROJECT_ROOT}/tb_logs/instruction_brain_qwen3_vl_8b_zeroshot_val"
VIDEO_DIR="${PROJECT_ROOT}/video_logs/instruction_brain_qwen3_vl_8b_zeroshot_val"
RECORD_DIR="${PROJECT_ROOT}/brain_records_zeroshot_qwen3_vl_8b_rl37"
FRAME_IMAGES_DIR="${PROJECT_ROOT}/brain_records_zeroshot_qwen3_vl_8b_rl37/frame_images"

mkdir -p "${TENSORBOARD_DIR}" "${VIDEO_DIR}" "${RECORD_DIR}" "${FRAME_IMAGES_DIR}"

# =============================================================================
# 参数解析
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --rl_ckpt)
            RL_CKPT_PATH="$2"
            shift 2
            ;;
        --qwen_ckpt)
            QWEN_MODEL_PATH="$2"
            shift 2
            ;;
        --yolo_ckpt)
            YOLO_CKPT_PATH="$2"
            shift 2
            ;;
        --episodes)
            EPISODE_COUNT="$2"
            shift 2
            ;;
        --output_dir)
            RECORD_DIR="$2"
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
echo "  Qwen3-VL-8B Zero-Shot 评估"
echo "=============================================="
echo "【策略网络权重】: ${RL_CKPT_PATH}"
echo "【Qwen3-VL-8B模型】: ${QWEN_MODEL_PATH}"
echo "【YOLOv8n检测器】: ${YOLO_CKPT_PATH}"
echo "【Episode数量】: ${EPISODE_COUNT}"
echo "【记录目录】: ${RECORD_DIR}"
echo "【图像目录】: ${FRAME_IMAGES_DIR}"
echo "=============================================="

# =============================================================================
# 构建评估命令
# =============================================================================

EVAL_CMD="python -u -m habitat_baselines.habitat_baselines.run \
    --config-name=habitat_baselines/config/DPed_brain/instruction_brain_qwen3_vl_8b_zeroshot_val.yaml \
    \
    habitat_baselines.evaluate=true \
    habitat_baselines.test_episode_count=${EPISODE_COUNT} \
    \
    habitat_baselines.eval_ckpt_path_dir=${RL_CKPT_PATH} \
    habitat_baselines.checkpoint_folder=${PROJECT_ROOT}/checkpoints/instruction_brain_qwen3_vl_8b_zeroshot_val \
    \
    habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
    habitat_baselines.video_dir=${VIDEO_DIR} \
    \
    habitat_baselines.brain.enabled=true \
    habitat_baselines.brain.model_type=qwen3_vl_8b \
    habitat_baselines.brain.model_path=${QWEN_MODEL_PATH} \
    habitat_baselines.brain.pedestrian_enabled=true \
    habitat_baselines.brain.pedestrian_ckpt_path=${YOLO_CKPT_PATH} \
    habitat_baselines.brain.freeze_brain=true \
    \
    habitat_baselines.brain.save_frame_records=true \
    habitat_baselines.brain.output_dir=${RECORD_DIR} \
    habitat_baselines.brain.log_prompt=true \
    habitat_baselines.brain.save_prompt_to_file=true \
    habitat_baselines.brain.save_frame_images=true \
    habitat_baselines.brain.frame_images_root=${FRAME_IMAGES_DIR}"

# =============================================================================
# 执行评估
# =============================================================================

echo ""
echo "开始Zero-Shot评估..."
echo ""
echo "命令:"
echo "${EVAL_CMD}"
echo ""

eval ${EVAL_CMD}

echo ""
echo "评估完成!"
