#!/bin/bash
#SBATCH --job-name=wangxiangyi_2b_14_fval_nobrain
#SBATCH --output=slurm_logs/instruction_brain_2b_14_fval_nobrain/%j_%x.out
#SBATCH --error=slurm_logs/instruction_brain_2b_14_fval_nobrain/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:l40:1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /share/home/u19666033/.bashrc
conda activate falcon

cd /share/home/u19666033/dhj/DPed_pro

unset PYTHONPATH
export PYTHONPATH=/share/home/u19666033/dhj/DPed_pro:/share/home/u19666033/dhj/DPed_pro/habitat-lab:/share/home/u19666033/dhj/DPed_pro/habitat-baselines

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072

mkdir -p slurm_logs/instruction_brain_2b_14_fval_nobrain

python -u -m habitat-baselines.habitat_baselines.run \
  --config-name=DPed_brain_new/fast_eval/instruction_4a_rlv1_8_brain_qwen3_vl_2b_14_fast_eval.yaml \
  'habitat_baselines.brain.enabled=False' \
  'habitat_baselines.brain.pedestrian_enabled=False' \
  'habitat_baselines.eval.split=val' \
  'habitat_baselines.checkpoint_folder=evaluation-vln/instruction_brain_qwen3_vl_2b_14-eval-nobrain/hm3d/checkpoints' \
  'habitat_baselines.tensorboard_dir=evaluation-vln/instruction_brain_qwen3_vl_2b_14-eval-nobrain/hm3d/tb' \
  'habitat_baselines.video_dir=evaluation-vln/instruction_brain_qwen3_vl_2b_14-eval-nobrain/hm3d/video'
