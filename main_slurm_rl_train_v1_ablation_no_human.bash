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
# 切换到工作目录
cd /share/home/u14004/dhj/Falcon-main
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# 消融实验：移除动态行人奖励和辅助任务
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train_v1_ablation_human_static.yaml
