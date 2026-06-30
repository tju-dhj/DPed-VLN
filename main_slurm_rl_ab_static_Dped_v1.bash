#!/bin/bash
#SBATCH --job-name=static_Dped_v1
#SBATCH --output=slurm_logs/ablation/static_Dped_v1/%j_%x.out
#SBATCH --error=slurm_logs/ablation/static_Dped_v1/%j_%x.err
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


# 运行评估
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=dynamic_vlnce/ablation/dynamic_vlnce_static_Dped_v1.yaml