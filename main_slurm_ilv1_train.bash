#!/bin/bash
#SBATCH --job-name=il-v1-new-wxy
#SBATCH --output=slurm_logs/il_v1_wxy/%j_%x.out
#SBATCH --error=slurm_logs/il_v1_wxy/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请1个任务(进程)
#SBATCH --cpus-per-task=7        # 每个任务用7个cpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /share/home/u19666033/.bashrc
conda activate falcon

# 切换到工作目录
cd /share/home/u19666033/dhj/DPed_pro

# 清理PYTHONPATH，强制使用当前目录的代码

# 设置环境变量以避免malloc错误
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072

# 运行训练
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_direct_il_train_v1.yaml