#!/bin/bash
#SBATCH --job-name=rl_v2
#SBATCH --output=slurm_logs/rl_v2/%j_%x.out
#SBATCH --error=slurm_logs/rl_v2/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p A800
#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请1个任务(进程)
#SBATCH --cpus-per-task=7        # 每个任务用7个cpu
#SBATCH --gres=gpu:a800:1

source /share/home/u14004/.bashrc
conda activate falcon
# python -u -m habitat-baselines.habitat_baselines.run --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_direct_il_train_v2.yaml
python -u -m habitat-baselines.habitat_baselines.run --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train_v2.yaml