#!/bin/bash
#SBATCH --job-name=wangxiangyi_grpo_from_il
#SBATCH --output=slurm_logs/grpo_train/%j_%x.out
#SBATCH --error=slurm_logs/grpo_train/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:l40:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/home/u19666033/.bashrc
conda activate falcon

cd /share/home/u19666033/dhj/DPed_pro

unset PYTHONPATH
export PYTHONPATH=/share/home/u19666033/dhj/DPed_pro:/share/home/u19666033/dhj/DPed_pro/habitat-lab:/share/home/u19666033/dhj/DPed_pro/habitat-baselines

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072
export HYDRA_FULL_ERROR=1

mkdir -p slurm_logs/grpo_train

python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=DPed_pro/new_data/v2/train_grpo_from_il_aligned_full.yaml \
  "$@"
