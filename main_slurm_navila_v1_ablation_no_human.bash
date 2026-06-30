#!/bin/bash
#SBATCH --job-name=navila_v1_ablation_no_human
#SBATCH --output=slurm_logs/navila_v1_ablation_no_human/%j_navila_v1_ablation_no_human.out
#SBATCH --error=slurm_logs/navila_v1_ablation_no_human/%j_navila_v1_ablation_no_human.err
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --partition=gpu

# 创建日志目录
mkdir -p slurm_logs/navila_v1_ablation_no_human

source /share/home/u14004/.bashrc
conda activate falcon

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# NaVILA消融实验：移除动态行人
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=dynamic_vlnce/navila_falcon_hm3d_v1_ablation_no_human.yaml

