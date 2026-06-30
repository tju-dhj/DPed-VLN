#!/bin/bash
#SBATCH --job-name=navila_lora_dped
#SBATCH --output=slurm_logs/navilla_lora/%j_%x.out
#SBATCH --error=slurm_logs/navilla_lora/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#
# NaviLLa LoRA 微调脚本 - 在 DPed_pro_resplit 数据集上微调
# 使用 LoRA (Low-Rank Adaptation) 进行参数高效微调

set -euo pipefail

# ========== 基础设置 ==========
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export CUDA_VISIBLE_DEVICES=0,1

set +u
source /share/home/u19666033/.bashrc
set -u
conda activate falcon

# ========== 路径设置 ==========
PROJECT_ROOT="/share/home/u19666033/dhj/DPed_pro"
cd "$PROJECT_ROOT"

NAVILA_DIR="habitat-baselines/habitat_baselines/rl/ddppo/policy/navila"
TRAIN_SCRIPT="${NAVILA_DIR}/llava/train/train_mem.py"

# 预训练 checkpoint（NaviLLa 基础模型）
BASE_MODEL="pretrained_model/navila_checkpoint"

# LoRA 微调后的输出路径
OUTPUT_DIR="pretrained_model/navilla_lora_dped_pro"

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
echo "=== 开始 NaviLLa LoRA 微调 ==="

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((RANDOM % 101 + 20001))

mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs/navilla_lora

torchrun --nnodes=1 --nproc_per_node=2 \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --deepspeed "${NAVILA_DIR}/scripts/zero3.json" \
    --model_name_or_path "$BASE_MODEL" \
    --version llama_3 \
    --seed 42 \
    --data_path "$TRAIN_DATA" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 4 \
    --tune_vision_tower False \
    --tune_mm_projector False \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

echo "=== NaviLLa LoRA 微调完成，模型保存至: $OUTPUT_DIR ==="
