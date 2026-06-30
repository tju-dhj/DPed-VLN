#!/bin/bash
#SBATCH --job-name=ab_static
#SBATCH --output=slurm_logs/ab_static/%j_%x.out
#SBATCH --error=slurm_logs/ab_static/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p A800
#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请1个任务(进程)
#SBATCH --cpus-per-task=7        # 每个任务用7个cpu
#SBATCH --gres=gpu:a800:1


source /share/home/u14004/.bashrc
conda activate falcon

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# NaVILA消融实验：完全静态场景（无动态行人）
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=dynamic_vlnce/navila_falcon_hm3d_v1_ablation_static_only.yaml

