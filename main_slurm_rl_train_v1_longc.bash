#!/bin/bash
#SBATCH --job-name=rl_v1
#SBATCH --output=slurm_logs/rl-v1/%j_%x.out
#SBATCH --error=slurm_logs/rl-v1/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -n 1
#SBATCH --ntasks-per-node=7
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
# 设置CUDA内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置环境变量以避免malloc错误
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072

source /share/home/u19666033/.bashrc
conda activate falcon

# 切换到工作目录
cd /share/home/u19666033/dhj/DPed_pro

# python -u -m habitat-baselines.habitat_baselines.run --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_direct_il_train_v2.yaml
python -u -m habitat-baselines.habitat_baselines.run --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train_v1_longc.yaml