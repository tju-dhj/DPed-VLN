#!/bin/bash
#SBATCH --job-name=dhj_rl_v2_ddppo
#SBATCH --output=slurm_logs/rl-v2-ddppo/%j_%x.out
#SBATCH --error=slurm_logs/rl-v2-ddppo/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#
# DDPPO 分卡训练启动脚本（每 GPU 1 进程）

set -euo pipefail

# CUDA / NCCL 基本设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# 避免 malloc 相关问题（可选）
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072

set +u
export BASHRCSOURCED="${BASHRCSOURCED:-1}"
source /share/home/u14004/.bashrc
set -u
conda activate falcon

# 切换到代码根目录（确保包含 habitat-baselines 模块）
cd /share/home/u14004/dhj/Falcon-main


# 使用 srun 启动多进程；Habitat 的 init_distrib_slurm 会读取 SLURM 环境变量完成 DDP 初始化
srun --unbuffered \
  python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=dynamic_vlnce_ddppo/dynamic_vlnce_ddppo_hm3d_train_v2_ddppo.yaml

# python -u -m habitat-baselines.habitat_baselines.run \
#   --config-name=DPed_brain/instruction_brain_qwen3_vl_8b_rl_val.yaml

