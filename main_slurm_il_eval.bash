#!/bin/bash
#SBATCH --job-name=dhj_il_eval
#SBATCH --output=slurm_logs/il_eval/%j_%x.out
#SBATCH --error=slurm_logs/il_eval/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH --nodes=1                # 申请1个节点
#SBATCH --ntasks=1               # 申请1个任务(进程)
#SBATCH --cpus-per-task=7        # 每个任务用7个cpu
#SBATCH --gres=gpu:l40:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/home/u19666033/.bashrc
conda activate falcon

# 切换到工作目录
cd /share/home/u19666033/dhj/DPed_pro

# 强制使用当前工程代码
unset PYTHONPATH
export PYTHONPATH=/share/home/u19666033/dhj/DPed_pro:/share/home/u19666033/dhj/DPed_pro/habitat-lab:/share/home/u19666033/dhj/DPed_pro/habitat-baselines

# 设置环境变量以减少 Habitat 渲染日志和内存碎片问题
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072

# 确保 Slurm 日志目录存在
mkdir -p slurm_logs/il_train

# 运行 IL 训练
python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=DPed_pro/new_data/eval/dped_eval_4a_direct_il_aligned_full_eval_fast.yaml
