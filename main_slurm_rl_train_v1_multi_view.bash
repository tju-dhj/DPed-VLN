#!/bin/bash
#SBATCH --job-name=rl_v1_multi_view
#SBATCH --output=slurm_logs/rl-v1-multi-view/%j_rl_v1_multi_view.out
#SBATCH --error=slurm_logs/rl-v1-multi-view/%j_rl_v1_multi_view.err
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=blgpu4
#SBATCH --mem=48G  # 增加内存限制（从默认的32G增加到48G）

# 多视角融合训练脚本
# 使用 overhead_front + third 两个视角的RGB和Depth

# 创建日志目录
mkdir -p slurm_logs/rl-v1-multi-view

# 激活conda环境
source ~/.bashrc
conda activate habitat

# 进入项目目录
cd /share/home/u14004/dhj/Falcon-main

# 打印环境信息
echo "=========================================="
echo "多视角融合训练 - 开始"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "配置文件: dynamic_vlnce_hm3d_train_v1_multi_view.yaml"
echo "视角: overhead_front + third (4个传感器)"
echo "融合模式: average"
echo "=========================================="

# 运行训练
python -u habitat-baselines/habitat_baselines/run.py \
    --exp-config habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train_v1_multi_view.yaml \
    --run-type train

# 打印结束信息
echo "=========================================="
echo "多视角融合训练 - 结束"
echo "结束时间: $(date)"
echo "=========================================="

