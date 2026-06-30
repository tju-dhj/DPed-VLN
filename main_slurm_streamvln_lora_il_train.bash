#!/bin/bash
#SBATCH --job-name=streamvln_lora_il
#SBATCH --output=slurm_logs/streamvln_lora_il/%j_%x.out
#SBATCH --error=slurm_logs/streamvln_lora_il/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7

#
# StreamVLN LoRA 微调 - DirectIL Trainer
# 使用 gt_action 作为监督信号，LoRA 参数高效微调
# 参考: dynamic_vlnce/streamvln_lora_direct_il_train.yaml

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

set +u
source /share/home/u19666033/.bashrc
set -u
conda activate falcon

cd /share/home/u19666033/dhj/DPed_pro

mkdir -p slurm_logs/streamvln_lora_il

echo "============================================"
echo "  StreamVLN LoRA Fine-tuning (DirectIL Trainer)"
echo "  Config: streamvln_lora_direct_il_train.yaml"
echo "============================================"

python -u -m habitat_baselines.run \
    --config-name=dynamic_vlnce/streamvln_lora_direct_il_train.yaml \
    habitat_baselines.evaluate=False

echo ""
echo "=== StreamVLN LoRA IL 微调完成 ==="
echo "Checkpoint saved to: evaluation-vln-dpedpro2/streamvln_lora_il_train/hm3d/checkpoints/"
