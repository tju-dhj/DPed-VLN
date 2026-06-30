#!/bin/bash
#SBATCH --job-name=rl_v2_ablation_no_human
#SBATCH --output=slurm_logs/rl_v2_ablation_no_human/%j_rl_v2_ablation_no_human.out
#SBATCH --error=slurm_logs/rl_v2_ablation_no_human/%j_rl_v2_ablation_no_human.err
#SBATCH --wckey=p14004
#SBATCH -A p_p14004
#SBATCH -p A800
#SBATCH -n 1
#SBATCH --ntasks-per-node=7
#SBATCH -N 1
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=7


source /share/home/u14004/.bashrc
conda activate falcon
# 切换到工作目录
cd /share/home/u14004/dhj/Falcon-main
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# 消融实验：移除动态行人奖励和辅助任务
python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train_v2_ablation_no_human.yaml
