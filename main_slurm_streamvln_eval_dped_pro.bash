#!/bin/bash
#SBATCH --job-name=streamvln_eval_dped
#SBATCH --output=slurm_logs/streamvln_eval/%j_%x.out
#SBATCH --error=slurm_logs/streamvln_eval/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#
# StreamVLN LoRA 微调后评估 - DPed_pro_resplit val_seen / val_unseen / test_unseen

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=0

set +u
source /share/home/u19666033/.bashrc
set -u
conda activate falcon

cd /share/home/u19666033/dhj/DPed_pro

mkdir -p slurm_logs/streamvln_eval

# ========== 环境变量 ==========
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# ========== 评估三个 split ==========
for SPLIT in val_seen val_unseen test_unseen; do
    echo ""
    echo "============================================"
    echo "  StreamVLN LoRA Eval: ${SPLIT}"
    echo "============================================"

    python -u -m habitat_baselines.run \
        --config-name=DPed_pro/new_data/stream/dped_pro_streamvln_${SPLIT}.yaml \
        habitat_baselines.evaluate=True

    echo "Completed: ${SPLIT}"
done

echo ""
echo "=== StreamVLN LoRA 全部评估完成 ==="
