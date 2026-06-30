#!/bin/bash
#SBATCH --job-name=wangxiangyi_2b_ib
#SBATCH --output=slurm_logs/instruction_brain_ddppo_2b/%j_%x.out
#SBATCH --error=slurm_logs/instruction_brain_ddppo_2b/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=7
#
# DDPPO 多卡训练启动脚本 - 使用 InstructionBrainPPOTrainer
# 每个GPU运行1个进程

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
source /share/home/u19666033/.bashrc
set -u
conda activate falcon

# 切换到代码根目录（确保包含 habitat-baselines 模块）
cd /share/home/u19666033/dhj/DPed_pro


# 使用 srun 启动多进程；Habitat 的 init_distrib_slurm 会读取 SLURM 环境变量完成 DDP 初始化
srun --unbuffered \
  python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=DPed_brain_new/instruction_4a_rlv1_8_brain_qwen3_vl_2b_ddppo_train.yaml
