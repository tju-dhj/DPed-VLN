#!/bin/bash
#SBATCH --job-name=dhj_instruction_brain_ddppo
#SBATCH --output=slurm_logs/instruction_brain_ddppo/%j_%x.out
#SBATCH --error=slurm_logs/instruction_brain_ddppo/%j_%x.err
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

# NCCL 健壮性设置 (防止 VLM 推理导致的 NCCL 超时)
# NCCL_TIMEOUT: 将默认 watchdog 超时从 600s 延长到 1800s，给慢 VLM 推理留足时间
# TORCH_NCCL_BLOCKING_WAIT: 启用阻塞等待检测模式，提供更清晰的错误信息
# TORCH_DISTRIBUTED_TIMEOUT: PyTorch 层级的分布式超时
# TORCH_NCCL_TRACE_BUFFER_SIZE: 记录 NCCL 操作调用栈，便于调试挂起问题
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
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
  --config-name=DPed_brain_new/instruction_4a_rlv1_8_brain_qwen3_vl_8b_ddppo_train.yaml
