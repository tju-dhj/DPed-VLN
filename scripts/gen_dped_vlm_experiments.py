#!/usr/bin/env python3
"""
Generate all DPed_vlm experiment YAML configs and sbatch scripts.
Usage: python scripts/gen_dped_vlm_experiments.py
"""
import os, shutil

BASE = "/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines/config"
CFG_DIR = f"{BASE}/DPed_vlm"
SBATCH_DIR = "/share/home/u19666033/dhj/DPed_pro/sbatch"
DATASET_BASE = "/share/home/u19666033/dhj/DPed_pro/dped_pro_resplit/dped-vln"
SCENES_DIR = "/share/home/u19666033/dhj/DPed_pro/data/scene_datasets/hm3d"
FALCON_CFG = f"/share/home/u19666033/dhj/Falcon/habitat-baselines/habitat_baselines/config/DPed_vlm"

os.makedirs(CFG_DIR, exist_ok=True)
os.makedirs(SBATCH_DIR, exist_ok=True)
os.makedirs(FALCON_CFG, exist_ok=True)

# ── Templates ──────────────────────────────────────────────
EVAL_HEADER = """# @package _global_
# {desc}

defaults:
  - /benchmark/nav/socialnav_v2: falcon_hm3d_task
  - /habitat_baselines: habitat_baselines_rl_config_base
  - /habitat/simulator/sim_sensors@habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor: third_rgb_sensor
  - /habitat_baselines/rl/policy@habitat_baselines.rl.policy.agent_1: single_fixed
  - /habitat_baselines/rl/policy@habitat_baselines.rl.policy.agent_2: single_fixed
  - /habitat_baselines/rl/policy@habitat_baselines.rl.policy.agent_3: single_fixed
  - /habitat_baselines/rl/policy@habitat_baselines.rl.policy.agent_4: single_fixed
  - /habitat_baselines/rl/policy@habitat_baselines.rl.policy.agent_5: single_fixed
  - /habitat_baselines/rl/policy@habitat_baselines.rl.policy.agent_6: single_fixed
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_stop: discrete_stop
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_move_forward: discrete_move_forward
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_turn_left: discrete_turn_left
  - /habitat/task/actions@habitat.task.actions.agent_0_discrete_turn_right: discrete_turn_right
  - /habitat/task/actions@habitat.task.actions.agent_1_oracle_nav_randcoord_action_obstacle: oracle_nav_action
  - /habitat/task/actions@habitat.task.actions.agent_2_oracle_nav_randcoord_action_obstacle: oracle_nav_action
  - /habitat/task/actions@habitat.task.actions.agent_3_oracle_nav_randcoord_action_obstacle: oracle_nav_action
  - /habitat/task/actions@habitat.task.actions.agent_4_oracle_nav_randcoord_action_obstacle: oracle_nav_action
  - /habitat/task/actions@habitat.task.actions.agent_5_oracle_nav_randcoord_action_obstacle: oracle_nav_action
  - /habitat/task/actions@habitat.task.actions.agent_6_oracle_nav_randcoord_action_obstacle: oracle_nav_action
  - /habitat/task/lab_sensors@habitat.task.lab_sensors.agent_0_pointgoal_with_gps_compass: pointgoal_with_gps_compass_sensor
  - _self_

habitat:
  dataset:
    type: "DynamicVLNCE-v1"
    split: {split}
    data_path: "{data_path}"
    scenes_dir: "{scenes_dir}"
    content_scenes: ["*"]
  environment:
    iterator_options:
      shuffle: False
  gym:
    obs_keys:
      - agent_0_overhead_front_rgb
      - agent_0_overhead_front_depth
      - agent_0_third_rgb
      - agent_0_third_depth
      - agent_0_pointgoal_with_gps_compass
      - agent_0_localization_sensor
      - agent_0_human_num_sensor
      - agent_0_oracle_humanoid_future_trajectory
      - agent_0_falcon_instruction
      - agent_0_falcon_gt_action
  task:
    measurements:
      success:
        type: Success
        success_distance: 3.0
    actions:
      agent_0_discrete_stop:
        lin_speed: 0.0
        ang_speed: 0.0
      agent_0_discrete_move_forward:
        lin_speed: 30.0
        ang_speed: 0.0
        allow_dyn_slide: True
      agent_0_discrete_turn_left:
        lin_speed: 0.0
        ang_speed: 31.415926535897
        allow_dyn_slide: True
      agent_0_discrete_turn_right:
        lin_speed: 0.0
        ang_speed: -31.415926535897
        allow_dyn_slide: True
      {human_actions}

habitat_baselines:
  evaluate: True
  verbose: True
  trainer_name: "ddppo"
  torch_gpu_id: 0
  tensorboard_dir: "{tb_dir}"
  test_episode_count: -1
  eval_ckpt_path_dir: "{ckpt_dir}"
  num_environments: 1
  checkpoint_folder: "{chkpt_folder}"
  num_updates: -1
  total_num_steps: 1e8
  log_interval: 10
  num_checkpoints: 200
  force_torch_single_threaded: True
  load_resume_state_config: False

  evaluator:
    _target_: {evaluator_target}

  eval:
    use_ckpt_config: False
    should_load_ckpt: False
    max_steps_per_episode: 500

  rl:
    agent:
      type: "MultiAgentAccessMgr"
      num_agent_types: 7
      num_active_agents_per_type: [1, 1, 1, 1, 1, 1, 1]
      num_pool_agents_per_type: [1, 1, 1, 1, 1, 1, 1]
      agent_sample_interval: 20
      force_partner_sample_idx: -1
    policy:
      agent_0:
{policy_block}
    ppo:
      clip_param: 0.2
      ppo_epoch: 2
      num_mini_batch: 2
      value_loss_coef: 0.5
      entropy_coef: 0.01
      lr: 2.5e-4
      eps: 1e-5
      max_grad_norm: 0.2
      num_steps: 128
      use_gae: True
      gamma: 0.99
      tau: 0.95
      use_linear_clip_decay: False
      use_linear_lr_decay: False
      reward_window_size: 50
      use_normalized_advantage: False
      hidden_size: {hidden_size}
      use_double_buffered_sampler: False

    ddppo:
      sync_frac: 0.6
      distrib_backend: NCCL
      force_distributed: False
      pretrained: False
      pretrained_encoder: False
      train_encoder: False
      reset_critic: True
      backbone: {backbone}
      rnn_type: {rnn_type}
      num_recurrent_layers: {num_recurrent_layers}
"""

HUMAN_NORMAL = """agent_1_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 10.0
        ang_speed: 10.0
        allow_dyn_slide: True
      agent_2_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 10.0
        ang_speed: 10.0
        allow_dyn_slide: True
      agent_3_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 10.0
        ang_speed: 10.0
        allow_dyn_slide: True
      agent_4_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 10.0
        ang_speed: 10.0
        allow_dyn_slide: True
      agent_5_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 10.0
        ang_speed: 10.0
        allow_dyn_slide: True
      agent_6_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 10.0
        ang_speed: 10.0
        allow_dyn_slide: True"""

HUMAN_STATIC = """agent_1_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 0.0
        ang_speed: 0.0
        allow_dyn_slide: True
      agent_2_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 0.0
        ang_speed: 0.0
        allow_dyn_slide: True
      agent_3_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 0.0
        ang_speed: 0.0
        allow_dyn_slide: True
      agent_4_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 0.0
        ang_speed: 0.0
        allow_dyn_slide: True
      agent_5_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 0.0
        ang_speed: 0.0
        allow_dyn_slide: True
      agent_6_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: 0.0
        ang_speed: 0.0
        allow_dyn_slide: True"""

# ── Model profiles ─────────────────────────────────────────
NAVILLA_POLICY = """       name: "NaVILAPolicy"
        model_path: "{model_path}"
        num_video_frames: 4
        forward_step: 25
        turn_step: 15
        action_distribution_type: "categorical\""""

STREAMVLN_POLICY = """       name: "StreamVLNPolicy"
        model_path: "{model_path}"
        num_frames: 32
        num_history: 8
        num_future_steps: 4
        model_max_length: 4096
        device: "cuda"
        forward_step: 25
        turn_step: 15
        action_distribution_type: "categorical\""""

NAVILLA_INFO = dict(policy=NAVILLA_POLICY, evaluator="habitat_baselines.rl.ppo.navila_evaluator.NaVILAEvaluator",
                    hidden_size=512, backbone="resnet50", rnn_type="LSTM", num_recurrent_layers=2,
                    model_path="pretrained_model/navila_checkpoint")

STREAMVLN_INFO = dict(policy=STREAMVLN_POLICY, evaluator="habitat_baselines.rl.ppo.streamvln_evaluator.StreamVLNEvaluator",
                      hidden_size=1024, backbone="streamvln", rnn_type="GRU", num_recurrent_layers=1,
                      model_path="pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln")

# ── Batch config ───────────────────────────────────────────
SPLITS = ["val_seen", "val_unseen", "test_unseen"]
LEVELS = {"l1": "v1", "l2": "v2"}
MODELS = {"navilla": NAVILLA_INFO, "streamvln": STREAMVLN_INFO}

def write_config(path, content):
    with open(path, "w") as f: f.write(content)

# ── 1. ZERO-SHOT EVAL CONFIGS ──────────────────────────────
print("Generating zero-shot eval configs...")
for model_name, mi in MODELS.items():
    for level_key, level_dir in LEVELS.items():
        for split in SPLITS:
            short = f"{model_name[0]}v"  # nv or sv
            fname = f"{model_name}_dped_{level_key}_{split}.yaml"
            exp_name = f"{model_name}_dped_{level_key}"
            desc = f"{model_name.upper()} zero-shot eval on DPed-vln {level_key.upper()} {split}"
            data_path = f"{DATASET_BASE}/{level_dir}/{split}/{{scene}}.json.gz"
            tb_dir = f"evaluation-vln-dpedpro2/{exp_name}_zeroshot/{level_key}/{split}/tb"
            chkpt_folder = f"evaluation-vln-dpedpro2/{exp_name}_zeroshot/{level_key}/{split}/checkpoints"

            cfg = EVAL_HEADER.format(
                desc=desc, split=split, data_path=data_path, scenes_dir=SCENES_DIR,
                human_actions=HUMAN_NORMAL,
                tb_dir=tb_dir, ckpt_dir=mi["model_path"], chkpt_folder=chkpt_folder,
                evaluator_target=mi["evaluator"],
                policy_block=mi["policy"].format(model_path=mi["model_path"]),
                hidden_size=mi["hidden_size"], backbone=mi["backbone"],
                rnn_type=mi["rnn_type"], num_recurrent_layers=mi["num_recurrent_layers"],
            )
            write_config(f"{CFG_DIR}/{fname}", cfg)
            write_config(f"{FALCON_CFG}/{fname}", cfg)
print(f"  Zero-shot: {len(MODELS)*len(LEVELS)*len(SPLITS)} files")

# ── 2. LORA EVAL CONFIGS ───────────────────────────────────
print("Generating LoRA eval configs...")
LORA_PATHS = {
    ("navilla","l1"): "pretrained_model/navilla_lora_dped_l1",
    ("navilla","l2"): "pretrained_model/navilla_lora_dped_l2",
    ("streamvln","l1"): "pretrained_model/streamvln_lora_dped_l1",
    ("streamvln","l2"): "pretrained_model/streamvln_lora_dped_l2",
}
for model_name, mi in MODELS.items():
    for level_key, level_dir in LEVELS.items():
        for split in SPLITS:
            exp_name = f"{model_name}_dped_{level_key}"
            fname = f"{model_name}_dped_{level_key}_{split}_lora.yaml"
            lora_path = LORA_PATHS[(model_name, level_key)]
            desc = f"{model_name.upper()} LoRA eval on DPed-vln {level_key.upper()} {split}"
            data_path = f"{DATASET_BASE}/{level_dir}/{split}/{{scene}}.json.gz"
            tb_dir = f"evaluation-vln-dpedpro2/{exp_name}_lora/{level_key}/{split}/tb"
            chkpt_folder = f"evaluation-vln-dpedpro2/{exp_name}_lora/{level_key}/{split}/checkpoints"

            cfg = EVAL_HEADER.format(
                desc=desc, split=split, data_path=data_path, scenes_dir=SCENES_DIR,
                human_actions=HUMAN_NORMAL,
                tb_dir=tb_dir, ckpt_dir=lora_path, chkpt_folder=chkpt_folder,
                evaluator_target=mi["evaluator"],
                policy_block=mi["policy"].format(model_path=lora_path),
                hidden_size=mi["hidden_size"], backbone=mi["backbone"],
                rnn_type=mi["rnn_type"], num_recurrent_layers=mi["num_recurrent_layers"],
            )
            write_config(f"{CFG_DIR}/{fname}", cfg)
            write_config(f"{FALCON_CFG}/{fname}", cfg)
print(f"  LoRA eval: {len(MODELS)*len(LEVELS)*len(SPLITS)} files")

# ── 3. STATIC HUMAN EVAL CONFIGS ───────────────────────────
print("Generating static human ablation eval configs...")
for model_name, mi in MODELS.items():
    for level_key, level_dir in LEVELS.items():
        for split in SPLITS:
            exp_name = f"{model_name}_dped_{level_key}"
            fname = f"{model_name}_dped_{level_key}_{split}_static.yaml"
            lora_path = LORA_PATHS[(model_name, level_key)]
            desc = f"{model_name.upper()} static-human eval on DPed-vln {level_key.upper()} {split}"
            data_path = f"{DATASET_BASE}/{level_dir}/{split}/{{scene}}.json.gz"
            tb_dir = f"evaluation-vln-dpedpro2/{exp_name}_static/{level_key}/{split}/tb"
            chkpt_folder = f"evaluation-vln-dpedpro2/{exp_name}_static/{level_key}/{split}/checkpoints"

            cfg = EVAL_HEADER.format(
                desc=desc, split=split, data_path=data_path, scenes_dir=SCENES_DIR,
                human_actions=HUMAN_STATIC,
                tb_dir=tb_dir, ckpt_dir=lora_path, chkpt_folder=chkpt_folder,
                evaluator_target=mi["evaluator"],
                policy_block=mi["policy"].format(model_path=lora_path),
                hidden_size=mi["hidden_size"], backbone=mi["backbone"],
                rnn_type=mi["rnn_type"], num_recurrent_layers=mi["num_recurrent_layers"],
            )
            write_config(f"{CFG_DIR}/{fname}", cfg)
            write_config(f"{FALCON_CFG}/{fname}", cfg)
print(f"  Static eval: {len(MODELS)*len(LEVELS)*len(SPLITS)} files")

# ── 4. BASH SCRIPTS ────────────────────────────────────────
BASH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/{log_dir}/%j_%x.out
#SBATCH --error=slurm_logs/{log_dir}/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time={time_limit}
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

mkdir -p slurm_logs/{log_dir}

{body}
"""

print("Generating sbatch scripts...")

# Helper to create both zero-shot eval and lora eval scripts
def make_eval_script(model_name, level_key, mode, desc_suffix, config_suffix, time_limit="24:00:00"):
    """mode: zeroshot, lora, static"""
    splits_cfg = []
    for split in SPLITS:
        cfg_name = f"DPed_vlm/{model_name}_dped_{level_key}_{split}{config_suffix}"
        splits_cfg.append(f'  python -u -m habitat_baselines.run --config-name={cfg_name} habitat_baselines.evaluate=True')
    body = "\n".join(splits_cfg)
    job = f"{model_name[:3]}_{level_key}_{mode}"
    content = BASH_TEMPLATE.format(
        job_name=job, log_dir=f"dped_vlm/{job}",
        time_limit=time_limit,
        desc=f"{model_name} {level_key} {desc_suffix}",
        body=body,
    )
    path = f"{SBATCH_DIR}/{job}.bash"
    write_config(path, content)
    os.chmod(path, 0o755)

for model_name in MODELS:
    for level_key in LEVELS:
        make_eval_script(model_name, level_key, "zeroshot", "zero-shot eval", "")
        make_eval_script(model_name, level_key, "lora", "LoRA fine-tuned eval", "_lora")
        make_eval_script(model_name, level_key, "static", "static human ablation eval", "_static")

print(f"\nDone! Generated:")
print(f"  Configs: {CFG_DIR}/")
print(f"  Configs (Falcon): {FALCON_CFG}/")
print(f"  Scripts: {SBATCH_DIR}/")
print(f"  Total configs: {len(list(os.listdir(CFG_DIR)))}")
print(f"  Total scripts: {len([f for f in os.listdir(SBATCH_DIR) if 'dped' in f.lower() or 'nav' in f.lower() or 'stream' in f.lower()])}")
