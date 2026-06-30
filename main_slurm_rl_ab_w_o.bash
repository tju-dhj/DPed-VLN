#!/bin/bash
#SBATCH --job-name=rl_ab_w_o
#SBATCH --output=slurm_logs/rl_ab_w_o/%j_%x.out
#SBATCH --error=slurm_logs/rl_ab_w_o/%j_%x.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p L40
#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请1个任务(进程)
#SBATCH --cpus-per-task=7        # 每个任务用7个cpu
#SBATCH --gres=gpu:l40:1

source /share/home/u14004/.bashrc
conda activate falcon

# 切换到工作目录
cd /share/home/u14004/dhj/Falcon-main

# 设置环境变量以解决 EGL 初始化问题
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2

# 运行评估
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=dynamic_vlnce/ablation/dynamic_vlnce_rl_val_ab_w_o.yaml