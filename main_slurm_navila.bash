#!/bin/bash
#SBATCH --job-name=falcon-navila
#SBATCH --output=slurm_logs/navila/%j_%x.out
#SBATCH --error=slurm_logs/navila/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p L40
#SBATCH -n 7
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 确保单节点模式：显式设置SLURM环境变量（虽然-n 7，但评估模式需要避免分布式）
# 如果确实需要7个进程，可以根据实际情况调整这些值
export SLURM_NTASKS=1
export SLURM_PROCID=0
export SLURM_LOCALID=0
source /share/home/u14004/.bashrc
conda activate falcon

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=dynamic_vlnce/navila_falcon_hm3d.yaml