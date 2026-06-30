#!/usr/bin/env python3
import os

CFG_DIR = "/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines/config/DPed_vlm"
FALCON_CFG = "/share/home/u19666033/dhj/Falcon/habitat-baselines/habitat_baselines/config/DPed_vlm"
DATASET_BASE = "/share/home/u19666033/dhj/DPed_pro/dped_pro_resplit/dped-vln"
SCENES_DIR = "/share/home/u19666033/dhj/DPed_pro/data/scene_datasets/hm3d"
SBATCH_DIR = "/share/home/u19666033/dhj/DPed_pro/sbatch"

os.makedirs(CFG_DIR, exist_ok=True)
os.makedirs(FALCON_CFG, exist_ok=True)
os.makedirs(SBATCH_DIR, exist_ok=True)

# ── Templates ──
TRAIN_CFG = open("/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_direct_il_train_v1.yaml").read()

def make_train_cfg(model, level, level_dir, human_type, ckpt_path, desc, policy_block, model_block, instr_type):
    data_path = DATASET_BASE + "/" + level_dir + "/train/{scene}.json.gz"
    exp = model + "_lora_" + level + ("_static" if human_type == "static" else "")
    tb = "evaluation-vln-dpedpro2/" + exp + "/hm3d/tb"
    cp = "evaluation-vln-dpedpro2/" + exp + "/hm3d/checkpoints"
    lf = "evaluation-vln-dpedpro2/" + exp + "/hm3d/train.log"
    ld = "evaluation-vln-dpedpro2/" + exp + "/hm3d/logs"
    rd = "evaluation-vln-dpedpro2/" + exp + "/hm3d/results/{split}"

    # Replace dataset path
    cfg = TRAIN_CFG
    cfg = cfg.replace(
        'data_path: "/share/home/u19666033/dhj/DPed_pro/data/dynamic_dataset_final_v1/train/{scene}.json.gz"',
        'data_path: "' + data_path + '"')
    cfg = cfg.replace(
        'scenes_dir: "/share/home/u19666033/dhj/DPed_pro/data/scene_datasets/hm3d"',
        'scenes_dir: "' + SCENES_DIR + '"')
    # Replace eval dirs
    cfg = cfg.replace(
        'evaluation-vln/dynamic_vlnce_clip_direct_il_v1_new_wxy_63/hm3d/tb', tb)
    cfg = cfg.replace(
        'evaluation-vln/dynamic_vlnce_clip_direct_il_v1_new_wxy_63/hm3d/checkpoints_2', cp)
    cfg = cfg.replace(
        'evaluation-vln/dynamic_vlnce_clip_direct_il_v1_new_wxy_63/hm3d/train.log', lf)
    cfg = cfg.replace(
        'evaluation-vln/dynamic_vlnce_clip_direct_il_v1_new_wxy_63/hm3d/logs', ld)
    cfg = cfg.replace(
        'evaluation-vln/dynamic_vlnce_clip_direct_il_v1_new_wxy_63/hm3d/results/{split}', rd)

    # Replace policy
    old_policy = 'name: "PointNavResNetPolicy"  # 使用与ddppo相同的策略，支持CLIP架构\n        action_distribution_type: "categorical"  # 必须显式指定为离散动作分布'
    cfg = cfg.replace(old_policy, policy_block)

    # Replace model
    old_model_start = 'hidden_size: 512'
    old_model_end = 'train_encoder: True'
    si = cfg.index(old_model_start)
    ei = cfg.index(old_model_end) + len(old_model_end)
    cfg = cfg[:si] + model_block + cfg[ei:]

    # Replace instruction type
    cfg = cfg.replace('instruction_vl_level_1', instr_type)

    # Replace pretrained_weights (remove old path, VLM loads via model_path)
    cfg = cfg.replace(
        'pretrained_weights: /share/home/u19666033/dhj/DPed_pro/pretrained_model/falcon_pretrained_25.pth',
        '# VLM loads via model_path, no pretrained_weights needed')

    # Replace use_iw and epochs
    cfg = cfg.replace('use_iw: True', 'use_iw: True')
    cfg = cfg.replace('epochs: 20', 'epochs: 5')

    # Human static: replace human action speeds
    if human_type == "static":
        cfg = cfg.replace('lin_speed: 10.0\n        ang_speed: 10.0\n        allow_dyn_slide: True',
                          'lin_speed: 0.0\n        ang_speed: 0.0\n        allow_dyn_slide: True')

    fname = model + "_lora_" + level + ("_static" if human_type == "static" else "") + "_train.yaml"
    for d in [CFG_DIR, FALCON_CFG]:
        with open(d + "/" + fname, "w") as f:
            f.write(cfg)
    return fname


# ── Policy blocks ──
NAVILLA_POLICY = 'name: "NaVILAPolicy"\n        action_distribution_type: "categorical"\n        model_path: "pretrained_model/navila_checkpoint"\n        num_video_frames: 4\n        forward_step: 25\n        turn_step: 15'
NAVILLA_MODEL = '''hidden_size: 512
      backbone: navila
      rnn_type: LSTM
      num_recurrent_layers: 2
      use_text_instruction: True
      text_encoder_dim: 2048
      fusion_method: attention
      use_lora: True
      lora_r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      lora_target_modules:
        - "q_proj"
        - "k_proj"
        - "v_proj"
        - "o_proj"
        - "gate_proj"
        - "up_proj"
        - "down_proj"
      pretrained: False
      pretrained_encoder: False
      train_encoder: False'''

STREAMVLN_POLICY = 'name: "StreamVLNPolicy"\n        action_distribution_type: "categorical"\n        model_path: "pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"\n        num_frames: 32\n        num_history: 8\n        num_future_steps: 4\n        model_max_length: 4096\n        device: "cuda"\n        forward_step: 25\n        turn_step: 15'
STREAMVLN_MODEL = '''hidden_size: 1024
      backbone: streamvln
      rnn_type: GRU
      num_recurrent_layers: 1
      use_text_instruction: True
      text_encoder_dim: 2048
      fusion_method: attention
      use_lora: True
      lora_r: 16
      lora_alpha: 32
      lora_dropout: 0.05
      lora_target_modules:
        - "q_proj"
        - "k_proj"
        - "v_proj"
        - "o_proj"
        - "gate_proj"
        - "up_proj"
        - "down_proj"
      pretrained: False
      pretrained_encoder: False
      train_encoder: False'''

# ── BASH template ──
BASH_TMPL = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/dped_vlm/{job_name}/%j_%x.out
#SBATCH --error=slurm_logs/dped_vlm/{job_name}/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=72:00:00
#
# {desc}

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-7}}"
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072
set +u
source /share/home/u19666033/.bashrc
set -u
conda activate falcon
cd /share/home/u19666033/dhj/DPed_pro
mkdir -p slurm_logs/dped_vlm/{job_name}

python -u -m habitat_baselines.run --config-name=DPed_vlm/{cfg_name} habitat_baselines.evaluate=False
"""

# ── Generate ──
for model, policy_blk, model_blk, ckpt_path in [
    ("navilla", NAVILLA_POLICY, NAVILLA_MODEL, "pretrained_model/navila_checkpoint"),
    ("streamvln", STREAMVLN_POLICY, STREAMVLN_MODEL, "pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"),
]:
    for level, level_dir, instr_type in [
        ("l1", "v1", "instruction_vl_level_1"),
        ("l2", "v2", "instruction_vl_level_2"),
    ]:
        for human_type in ["normal", "static"]:
            desc = model + " LoRA " + level + " " + human_type + " human IL training"
            fname = make_train_cfg(model, level, level_dir, human_type, ckpt_path, desc, policy_blk, model_blk, instr_type)

            # Create sbatch script
            job = model[:3] + "_" + level + ("_" + human_type if human_type == "static" else "_train")
            bash = BASH_TMPL.format(job_name=job, desc=desc, cfg_name=fname)
            bpath = SBATCH_DIR + "/" + job + ".bash"
            with open(bpath, "w") as f:
                f.write(bash)
            os.chmod(bpath, 0o755)

print("Training configs:", len([f for f in os.listdir(CFG_DIR) if "train" in f]))
print("Total DPed_vlm configs:", len(os.listdir(CFG_DIR)))
print("Total sbatch scripts:", len(os.listdir(SBATCH_DIR)))
