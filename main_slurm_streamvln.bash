#!/bin/bash
#SBATCH --job-name=falcon-streamvln
#SBATCH --output=slurm_logs/streamvln/%j_%x.out
#SBATCH --error=slurm_logs/streamvln/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p L40
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 确保单节点模式：显式设置SLURM环境变量
export SLURM_NTASKS=1
export SLURM_PROCID=0
export SLURM_LOCALID=0
source /share/home/u14004/.bashrc
conda activate falcon

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m habitat-baselines.habitat_baselines.run \
--config-name=dynamic_vlnce/streamvln_falcon_hm3d.yaml