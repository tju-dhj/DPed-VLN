#!/bin/bash
#SBATCH --job-name=rl_v32
#SBATCH --output=slurm_logs/rl_v32/%j_%x.out
#SBATCH --error=slurm_logs/rl_v32/%j_%x.err
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
# 强制使用 CPU 渲染以避免 EGL 多进程冲突（如果需要 GPU 渲染，改为 export EGL_DEVICE_ID=0）
# export MESA_GL_VERSION_OVERRIDE=3.3
# export MESA_GLSL_VERSION_OVERRIDE=330

# 运行评估
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_rl_val_v32.yaml