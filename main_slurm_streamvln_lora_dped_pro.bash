#!/bin/bash
#SBATCH --job-name=streamvln_lora_dped
#SBATCH --output=slurm_logs/streamvln_lora/%j_%x.out
#SBATCH --error=slurm_logs/streamvln_lora/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#
# StreamVLN LoRA 微调脚本 - 在 DPed_pro_resplit 数据集上微调
# 使用 LoRA (Low-Rank Adaptation) 进行参数高效微调

set -euo pipefail

# ========== 基础设置 ==========
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/share/home/u19666033/.cache/huggingface

set +u
source /share/home/u19666033/.bashrc
set -u
conda activate falcon

# ========== 路径设置 ==========
PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
cd "$PROJECT_ROOT"

STREAMVLN_DIR="habitat-baselines/habitat_baselines/rl/ddppo/policy/streamvln"
TRAIN_SCRIPT="${STREAMVLN_DIR}/streamvln/streamvln_train.py"

# StreamVLN 预训练 checkpoint
BASE_MODEL="pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"

# LoRA 微调后的输出路径
OUTPUT_DIR="pretrained_model/streamvln_lora_dped_pro"

# 转换后的训练数据
TRAIN_DATA="${PROJECT_ROOT}/dped_pro_resplit/train_converted.json"

# ========== 数据准备 ==========
echo "=== 准备训练数据 ==="
if [ ! -f "$TRAIN_DATA" ]; then
    python scripts/prepare_dped_data.py \
        --input_dir "${PROJECT_ROOT}/dped_pro_resplit/train" \
        --output "$TRAIN_DATA" \
        --split train
fi

# ========== LoRA 微调 ==========
echo "=== 开始 StreamVLN LoRA 微调 ==="

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((RANDOM % 101 + 20001))

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
PROMPT_VERSION="qwen_1_5"

mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs/streamvln_lora

python -u "$TRAIN_SCRIPT" \
    --deepspeed "${STREAMVLN_DIR}/scripts/zero2.json" \
    --model_name_or_path "$BASE_MODEL" \
    --version "$PROMPT_VERSION" \
    --data_path "$TRAIN_DATA" \
    --group_by_task False \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --data_augmentation False \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --frames_upbound 32 \
    --force_sample True \
    --add_time_instruction True \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --bf16 True \
    --run_name "streamvln_lora_dped_pro" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --mm_vision_tower_lr 5e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --report_to none \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

echo "=== StreamVLN LoRA 微调完成，模型保存至: $OUTPUT_DIR ==="
