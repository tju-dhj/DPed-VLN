#!/usr/bin/env python3
"""Generate all DPed_vlm YAML configs and sbatch scripts."""

import os

PROJECT_ROOT = "/share/home/u19666033/dhj/DPed_pro"
CONFIG_ROOT = "/share/home/u19666033/dhj/dped-vln/habitat-baselines/habitat_baselines/config/DPed_vlm"
SBATCH_ROOT = os.path.join(PROJECT_ROOT, "sbatch/DPed_vlm")
SCENE_DIR = "/share/home/u19666033/dhj/DPed_pro/data/scene_datasets/hm3d"

NAVILLA_BASE = "pretrained_model/navila_checkpoint"
NAVILLA_LORA = "pretrained_model/navilla_lora_dped_{level}"
NAVILLA_STATIC = "pretrained_model/navilla_lora_dped_{level}_static"

STREAMVLN_BASE = "pretrained_model/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln"
STREAMVLN_LORA = "pretrained_model/streamvln_lora_dped_{level}"
STREAMVLN_STATIC = "pretrained_model/streamvln_lora_dped_{level}_static"

HUMAN_SPEED_NORMAL = "10.0"
HUMAN_SPEED_STATIC = "0.0"

# ── Templates (using {placeholders} for .format()) ──

EVAL_TEMPLATE = """# @package _global_
# {description}

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
    scenes_dir: "{SCENE_DIR}"
    content_scenes: ["*"]
  environment:
    iterator_options:
      shuffle: False
  gym:
    obs_keys:{obs_keys}
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
      agent_1_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_2_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_3_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_4_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_5_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_6_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True

habitat_baselines:
  evaluate: True
  verbose: True
  trainer_name: "ddppo"
  torch_gpu_id: 0
  tensorboard_dir: "{tb_dir}"
  test_episode_count: -1
  eval_ckpt_path_dir: "{model_path}"
  num_environments: 1
  checkpoint_folder: "{ckpt_dir}"
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
{policy_section}
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
      backbone: "{backbone}"
      rnn_type: "{rnn_type}"
      num_recurrent_layers: {num_recurrent}
"""

TRAIN_TEMPLATE = """# @package _global_
# {description}

defaults:
  - /benchmark/nav/socialnav_v2: falcon_hm3d_task_for_train_vln
  - /habitat_baselines: habitat_baselines_il_config_base
  - /habitat/simulator/sim_sensors@habitat_baselines.eval.extra_sim_sensors.third_rgb_sensor: third_rgb_sensor
  - /habitat_baselines/il/policy@habitat_baselines.il.policy.agent_1: single_fixed
  - /habitat_baselines/il/policy@habitat_baselines.il.policy.agent_2: single_fixed
  - /habitat_baselines/il/policy@habitat_baselines.il.policy.agent_3: single_fixed
  - /habitat_baselines/il/policy@habitat_baselines.il.policy.agent_4: single_fixed
  - /habitat_baselines/il/policy@habitat_baselines.il.policy.agent_5: single_fixed
  - /habitat_baselines/il/policy@habitat_baselines.il.policy.agent_6: single_fixed
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
    split: train
    type: DynamicVLNCE-v1
    data_path: "{data_path}"
    scenes_dir: "{SCENE_DIR}"
    content_scenes: ["*"]
  environment:
    iterator_options:
      shuffle: True
      cycle: True
  gym:
    obs_keys:
      - agent_0_overhead_front_rgb
      - agent_0_overhead_front_depth
      - agent_0_pointgoal_with_gps_compass
      - agent_0_localization_sensor
      - agent_0_human_num_sensor
      - agent_0_oracle_humanoid_future_trajectory
      - agent_0_falcon_instruction
      - agent_0_falcon_gt_action
  task:
    actions:
      agent_0_discrete_stop:
        lin_speed: 0.0
        ang_speed: 0.0
      agent_0_discrete_move_forward:
        lin_speed: 25.0
        ang_speed: 0.0
        allow_dyn_slide: True
      agent_0_discrete_turn_left:
        lin_speed: 0.0
        ang_speed: 10.0
        allow_dyn_slide: True
      agent_0_discrete_turn_right:
        lin_speed: 0.0
        ang_speed: -10.0
        allow_dyn_slide: True
      agent_1_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_2_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_3_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_4_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_5_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True
      agent_6_oracle_nav_randcoord_action_obstacle:
        type: OracleNavRandCoordAction_Obstacle
        motion_control: human_joints
        lin_speed: {human_lin_speed}
        ang_speed: {human_ang_speed}
        allow_dyn_slide: True

habitat_baselines:
  evaluate: False
  verbose: True
  trainer_name: "direct_il"
  torch_gpu_id: 0
  tensorboard_dir: "{tb_dir}"
  test_episode_count: -1
  checkpoint_folder: "{ckpt_dir}"
  num_updates: -1
  total_num_steps: 15000000
  log_interval: 50
  log_file: "{log_file}"
  num_checkpoints: 50
  checkpoint_interval: 3
  force_torch_single_threaded: False
  load_resume_state_config: False

  eval:
    use_ckpt_config: False
    should_load_ckpt: True

  il:
    epochs: 10
    batch_size: 1
    use_iw: True
    inflection_weight_coef: 3.0
    dataloader_num_workers: 0
    pin_memory: False

    distributed:
      enabled: False

    eval_save_results: False
    log_metrics: False
    output_log_dir: "{log_dir}"
    results_dir: "{log_dir}/results/{{split}}"
    save_resume_state_interval: 10
    save_state_batch_only: False

    agent:
      type: "MultiAgentAccessMgr"
      num_agent_types: 7
      num_active_agents_per_type: [1, 1, 1, 1, 1, 1, 1]
      num_pool_agents_per_type: [1, 1, 1, 1, 1, 1, 1]
      agent_sample_interval: 20
      force_partner_sample_idx: -1

    policy:
      agent_0:
{policy_section}

    model:
      hidden_size: {hidden_size}
      backbone: "{backbone}"
      rnn_type: "{rnn_type}"
      num_recurrent_layers: {num_recurrent}
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
      train_encoder: False

    optim:
      lr: 2e-4
      eps: 1e-5
      max_grad_norm: 0.5

    direct_il:
      dataset_type: "json"
      data_root: "/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/{level}/train"
      max_episodes: -1
      max_episode_length: 100
      instruction_priority:
        - instruction_vl_level_1
"""

EVAL_SBATCH = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/dped_vlm/{log_subdir}/%j_%x.out
#SBATCH --error=slurm_logs/dped_vlm/{log_subdir}/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=24:00:00

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-7}}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072

set +u
source /share/home/u19666033/.bashrc
set -u
conda activate falcon

unset PYTHONPATH
export PYTHONPATH=/share/home/u19666033/dhj/dped-vln:/share/home/u19666033/dhj/dped-vln/habitat-lab:/share/home/u19666033/dhj/dped-vln/habitat-baselines:/share/home/u19666033/dhj/DPed_pro:/share/home/u19666033/dhj/DPed_pro/habitat-lab:/share/home/u19666033/dhj/DPed_pro/habitat-baselines

cd /share/home/u19666033/dhj/dped-vln

mkdir -p slurm_logs/dped_vlm/{log_subdir}

echo "============================================"
echo "  {description}"
echo "============================================"

for SPLIT in val_seen val_unseen test_unseen; do
    echo "--- ${{SPLIT}} ---"
    python -u -m habitat_baselines.run \\
      --config-name={config_prefix}${{SPLIT}}.yaml \\
      habitat_baselines.evaluate=True
    echo "Done: ${{SPLIT}}"
done

echo "=== {description} Complete ==="
"""

TRAIN_SBATCH = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm_logs/dped_vlm/{log_subdir}/%j_%x.out
#SBATCH --error=slurm_logs/dped_vlm/{log_subdir}/%j_%x.err
#SBATCH --wckey=p19666033
#SBATCH -A p_p19666033
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --time=72:00:00

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-7}}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=131072

set +u
source /share/home/u19666033/.bashrc
set -u
conda activate falcon

unset PYTHONPATH
export PYTHONPATH=/share/home/u19666033/dhj/dped-vln:/share/home/u19666033/dhj/dped-vln/habitat-lab:/share/home/u19666033/dhj/dped-vln/habitat-baselines:/share/home/u19666033/dhj/DPed_pro:/share/home/u19666033/dhj/DPed_pro/habitat-lab:/share/home/u19666033/dhj/DPed_pro/habitat-baselines

cd /share/home/u19666033/dhj/dped-vln

mkdir -p slurm_logs/dped_vlm/{log_subdir}

echo "============================================"
echo "  {description}"
echo "============================================"

python -u -m habitat_baselines.run \\
  --config-name={config_name} \\
  habitat_baselines.evaluate=False

echo "=== {description} Complete ==="
"""


def navilla_policy_section(model_path):
    return f"""        name: "NaVILAPolicy"
        action_distribution_type: "categorical"
        model_path: "{model_path}"
        num_video_frames: 4
        forward_step: 25
        turn_step: 15"""

def streamvln_policy_section(model_path):
    return f"""        name: "StreamVLNPolicy"
        action_distribution_type: "categorical"
        model_path: "{model_path}"
        num_frames: 32
        num_history: 8
        num_future_steps: 4
        model_max_length: 4096
        device: "cuda"
        forward_step: 25
        turn_step: 15"""

NAVILLA_OBS_KEYS = """
      - agent_0_overhead_front_rgb
      - agent_0_overhead_front_depth
      - agent_0_third_rgb
      - agent_0_third_depth
      - agent_0_pointgoal_with_gps_compass
      - agent_0_localization_sensor
      - agent_0_human_num_sensor
      - agent_0_oracle_humanoid_future_trajectory
      - agent_0_falcon_instruction
      - agent_0_falcon_gt_action"""

STREAMVLN_OBS_KEYS = """
      - agent_0_overhead_front_rgb
      - agent_0_overhead_front_depth
      - agent_0_pointgoal_with_gps_compass
      - agent_0_localization_sensor
      - agent_0_human_num_sensor
      - agent_0_oracle_humanoid_future_trajectory
      - agent_0_falcon_instruction
      - agent_0_falcon_gt_action"""


def generate():
    os.makedirs(CONFIG_ROOT, exist_ok=True)
    os.makedirs(SBATCH_ROOT, exist_ok=True)

    models = {
        "navilla": {
            "base_path": NAVILLA_BASE,
            "lora_path": NAVILLA_LORA,
            "static_path": NAVILLA_STATIC,
            "policy_fn": navilla_policy_section,
            "hidden_size": "512",
            "backbone": "resnet50",
            "rnn_type": "LSTM",
            "num_recurrent": "2",
            "evaluator_target": "habitat_baselines.rl.ppo.navila_evaluator.NaVILAEvaluator",
            "obs_keys": NAVILLA_OBS_KEYS,
            "policy_name": "NaVILAPolicy",
        },
        "streamvln": {
            "base_path": STREAMVLN_BASE,
            "lora_path": STREAMVLN_LORA,
            "static_path": STREAMVLN_STATIC,
            "policy_fn": streamvln_policy_section,
            "hidden_size": "1024",
            "backbone": "streamvln",
            "rnn_type": "GRU",
            "num_recurrent": "1",
            "evaluator_target": "habitat_baselines.rl.ppo.streamvln_evaluator.StreamVLNEvaluator",
            "obs_keys": STREAMVLN_OBS_KEYS,
            "policy_name": "StreamVLNPolicy",
        },
    }

    levels = ["v1", "v2"]
    splits = ["val_seen", "val_unseen", "test_unseen"]

    yaml_count = 0
    sbatch_count = 0

    for model_name, m in models.items():
        for mode_dir in ["zero_shot", "zero_shot_static", "lora", "human_static"]:
            os.makedirs(os.path.join(CONFIG_ROOT, model_name, mode_dir), exist_ok=True)

        # ── 1. Zero-shot ──
        for level in levels:
            for split in splits:
                desc = f"{model_name.upper()} Zero-Shot DPed-{level.upper()} {split}"
                params = dict(
                    description=desc,
                    split=split,
                    data_path=f"/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/{level}/{split}/{{scene}}.json.gz",
                    obs_keys=m["obs_keys"],
                    human_lin_speed=HUMAN_SPEED_NORMAL,
                    human_ang_speed=HUMAN_SPEED_NORMAL,
                    tb_dir=f"evaluation-vln-dpedpro2/{model_name}_zero_shot_{level}_{split}/hm3d/tb",
                    ckpt_dir=f"evaluation-vln-dpedpro2/{model_name}_zero_shot_{level}_{split}/hm3d/checkpoints",
                    model_path=m["base_path"],
                    evaluator_target=m["evaluator_target"],
                    policy_section=m["policy_fn"](m["base_path"]),
                    hidden_size=m["hidden_size"],
                    backbone=m["backbone"],
                    rnn_type=m["rnn_type"],
                    num_recurrent=m["num_recurrent"],
                    SCENE_DIR=SCENE_DIR,
                level=level,
                )
                fpath = os.path.join(CONFIG_ROOT, model_name, "zero_shot", f"{level}_{split}.yaml")
                with open(fpath, 'w') as f:
                    f.write(EVAL_TEMPLATE.format(**params))
                yaml_count += 1

            # zero-shot eval sbatch
            log_subdir = f"{model_name}_zero_shot_{level}"
            sbatch = EVAL_SBATCH.format(
                job_name=f"{model_name[:4]}_zs_{level}",
                log_subdir=log_subdir,
                description=f"{model_name.upper()} Zero-Shot DPed-{level.upper()}",
                config_prefix=f"DPed_vlm/{model_name}/zero_shot/{level}_",
            )
            spath = os.path.join(SBATCH_ROOT, f"{model_name}_zero_shot_{level}.bash")
            with open(spath, 'w') as f:
                f.write(sbatch)
            os.chmod(spath, 0o755)
            sbatch_count += 1

            # ── 1b. Zero-shot human static (行人速度=0，不加微调) ──
            for split in splits:
                desc = f"{model_name.upper()} Zero-Shot-Static DPed-{level.upper()} {split}"
                params = dict(
                    description=desc,
                    split=split,
                    data_path=f"/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/{level}/{split}/{{scene}}.json.gz",
                    obs_keys=m["obs_keys"],
                    human_lin_speed=HUMAN_SPEED_STATIC,
                    human_ang_speed=HUMAN_SPEED_STATIC,
                    tb_dir=f"evaluation-vln-dpedpro2/{model_name}_zero_shot_static_{level}_{split}/hm3d/tb",
                    ckpt_dir=f"evaluation-vln-dpedpro2/{model_name}_zero_shot_static_{level}_{split}/hm3d/checkpoints",
                    model_path=m["base_path"],
                    evaluator_target=m["evaluator_target"],
                    policy_section=m["policy_fn"](m["base_path"]),
                    hidden_size=m["hidden_size"],
                    backbone=m["backbone"],
                    rnn_type=m["rnn_type"],
                    num_recurrent=m["num_recurrent"],
                    SCENE_DIR=SCENE_DIR,
                level=level,
                )
                fpath = os.path.join(CONFIG_ROOT, model_name, "zero_shot_static", f"{level}_{split}.yaml")
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, 'w') as f:
                    f.write(EVAL_TEMPLATE.format(**params))
                yaml_count += 1

            # zero-shot static eval sbatch
            log_subdir = f"{model_name}_zero_shot_static_{level}"
            sbatch = EVAL_SBATCH.format(
                job_name=f"{model_name[:4]}_zss_{level}",
                log_subdir=log_subdir,
                description=f"{model_name.upper()} Zero-Shot-Static DPed-{level.upper()}",
                config_prefix=f"DPed_vlm/{model_name}/zero_shot_static/{level}_",
            )
            spath = os.path.join(SBATCH_ROOT, f"{model_name}_zero_shot_static_{level}.bash")
            with open(spath, 'w') as f:
                f.write(sbatch)
            os.chmod(spath, 0o755)
            sbatch_count += 1

            # ── 2. LoRA ──
            lora_chkpt = m["lora_path"].format(level=level)
            base_suffix = f"lora"

            # Train config
            params_train = dict(
                description=f"{model_name.upper()} LoRA Train DPed-{level.upper()}",
                data_path=f"/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/{level}/train/{{scene}}.json.gz",
                human_lin_speed=HUMAN_SPEED_NORMAL,
                human_ang_speed=HUMAN_SPEED_NORMAL,
                tb_dir=f"evaluation-vln-dpedpro2/{model_name}_lora_train_{level}/hm3d/tb",
                ckpt_dir=f"evaluation-vln-dpedpro2/{model_name}_lora_train_{level}/hm3d/checkpoints",
                log_file=f"evaluation-vln-dpedpro2/{model_name}_lora_train_{level}/hm3d/train.log",
                log_dir=f"evaluation-vln-dpedpro2/{model_name}_lora_train_{level}/hm3d/logs",
                log_subdir=f"{model_name}_lora_{level}",
                policy_section=m["policy_fn"](m["base_path"]),
                hidden_size=m["hidden_size"],
                backbone=m["backbone"],
                rnn_type=m["rnn_type"],
                num_recurrent=m["num_recurrent"],
                SCENE_DIR=SCENE_DIR,
                level=level,
            )
            fpath = os.path.join(CONFIG_ROOT, model_name, "lora", f"{level}_train.yaml")
            with open(fpath, 'w') as f:
                f.write(TRAIN_TEMPLATE.format(**params_train))
            yaml_count += 1

            # LoRA eval configs
            for split in splits:
                desc = f"{model_name.upper()} LoRA Eval DPed-{level.upper()} {split}"
                params = dict(
                    description=desc,
                    split=split,
                    data_path=f"/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/{level}/{split}/{{scene}}.json.gz",
                    obs_keys=m["obs_keys"],
                    human_lin_speed=HUMAN_SPEED_NORMAL,
                    human_ang_speed=HUMAN_SPEED_NORMAL,
                    tb_dir=f"evaluation-vln-dpedpro2/{model_name}_lora_{level}_{split}/hm3d/tb",
                    ckpt_dir=f"evaluation-vln-dpedpro2/{model_name}_lora_{level}_{split}/hm3d/checkpoints",
                    model_path=lora_chkpt,
                    evaluator_target=m["evaluator_target"],
                    policy_section=m["policy_fn"](lora_chkpt),
                    hidden_size=m["hidden_size"],
                    backbone=m["backbone"],
                    rnn_type=m["rnn_type"],
                    num_recurrent=m["num_recurrent"],
                    SCENE_DIR=SCENE_DIR,
                level=level,
                )
                fpath = os.path.join(CONFIG_ROOT, model_name, "lora", f"{level}_{split}.yaml")
                with open(fpath, 'w') as f:
                    f.write(EVAL_TEMPLATE.format(**params))
                yaml_count += 1

            # LoRA train sbatch
            log_subdir = f"{model_name}_lora_{level}"
            sbatch = TRAIN_SBATCH.format(
                job_name=f"{model_name[:4]}_lora_{level}_t",
                log_subdir=f"{model_name}_lora_{level}",
                description=f"{model_name.upper()} LoRA Train DPed-{level.upper()}",
                config_name=f"DPed_vlm/{model_name}/lora/{level}_train.yaml",
            )
            spath = os.path.join(SBATCH_ROOT, f"{model_name}_lora_{level}_train.bash")
            with open(spath, 'w') as f:
                f.write(sbatch)
            os.chmod(spath, 0o755)
            sbatch_count += 1

            # LoRA eval sbatch
            log_subdir = f"{model_name}_lora_{level}"
            sbatch = EVAL_SBATCH.format(
                job_name=f"{model_name[:4]}_lora_{level}_e",
                log_subdir=f"{model_name}_lora_{level}",
                description=f"{model_name.upper()} LoRA Eval DPed-{level.upper()}",
                config_prefix=f"DPed_vlm/{model_name}/lora/{level}_",
            )
            spath = os.path.join(SBATCH_ROOT, f"{model_name}_lora_{level}_eval.bash")
            with open(spath, 'w') as f:
                f.write(sbatch)
            os.chmod(spath, 0o755)
            sbatch_count += 1

            # ── 3. Human static ablation ──
            static_chkpt = m["static_path"].format(level=level)

            # Train config (human velocity = 0)
            params_train_static = dict(
                description=f"{model_name.upper()} Human-Static LoRA Train DPed-{level.upper()}",
                data_path=f"/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/{level}/train/{{scene}}.json.gz",
                human_lin_speed=HUMAN_SPEED_STATIC,
                human_ang_speed=HUMAN_SPEED_STATIC,
                tb_dir=f"evaluation-vln-dpedpro2/{model_name}_human_static_train_{level}/hm3d/tb",
                ckpt_dir=f"evaluation-vln-dpedpro2/{model_name}_human_static_train_{level}/hm3d/checkpoints",
                log_file=f"evaluation-vln-dpedpro2/{model_name}_human_static_train_{level}/hm3d/train.log",
                log_dir=f"evaluation-vln-dpedpro2/{model_name}_human_static_train_{level}/hm3d/logs",
                log_subdir=f"{model_name}_human_static_{level}",
                policy_section=m["policy_fn"](m["base_path"]),
                hidden_size=m["hidden_size"],
                backbone=m["backbone"],
                rnn_type=m["rnn_type"],
                num_recurrent=m["num_recurrent"],
                SCENE_DIR=SCENE_DIR,
                level=level,
            )
            fpath = os.path.join(CONFIG_ROOT, model_name, "human_static", f"{level}_train.yaml")
            with open(fpath, 'w') as f:
                f.write(TRAIN_TEMPLATE.format(**params_train_static))
            yaml_count += 1

            # Human static eval configs
            for split in splits:
                desc = f"{model_name.upper()} Human-Static Eval DPed-{level.upper()} {split}"
                params = dict(
                    description=desc,
                    split=split,
                    data_path=f"/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets/{level}/{split}/{{scene}}.json.gz",
                    obs_keys=m["obs_keys"],
                    human_lin_speed=HUMAN_SPEED_STATIC,
                    human_ang_speed=HUMAN_SPEED_STATIC,
                    tb_dir=f"evaluation-vln-dpedpro2/{model_name}_human_static_{level}_{split}/hm3d/tb",
                    ckpt_dir=f"evaluation-vln-dpedpro2/{model_name}_human_static_{level}_{split}/hm3d/checkpoints",
                    model_path=static_chkpt,
                    evaluator_target=m["evaluator_target"],
                    policy_section=m["policy_fn"](static_chkpt),
                    hidden_size=m["hidden_size"],
                    backbone=m["backbone"],
                    rnn_type=m["rnn_type"],
                    num_recurrent=m["num_recurrent"],
                    SCENE_DIR=SCENE_DIR,
                level=level,
                )
                fpath = os.path.join(CONFIG_ROOT, model_name, "human_static", f"{level}_{split}.yaml")
                with open(fpath, 'w') as f:
                    f.write(EVAL_TEMPLATE.format(**params))
                yaml_count += 1

            # Human static train sbatch
            log_subdir = f"{model_name}_human_static_{level}"
            sbatch = TRAIN_SBATCH.format(
                job_name=f"{model_name[:4]}_stat_{level}_t",
                log_subdir=f"{model_name}_human_static_{level}",
                description=f"{model_name.upper()} Human-Static Train DPed-{level.upper()}",
                config_name=f"DPed_vlm/{model_name}/human_static/{level}_train.yaml",
            )
            spath = os.path.join(SBATCH_ROOT, f"{model_name}_human_static_{level}_train.bash")
            with open(spath, 'w') as f:
                f.write(sbatch)
            os.chmod(spath, 0o755)
            sbatch_count += 1

            # Human static eval sbatch
            log_subdir = f"{model_name}_human_static_{level}"
            sbatch = EVAL_SBATCH.format(
                job_name=f"{model_name[:4]}_stat_{level}_e",
                log_subdir=f"{model_name}_human_static_{level}",
                description=f"{model_name.upper()} Human-Static Eval DPed-{level.upper()}",
                config_prefix=f"DPed_vlm/{model_name}/human_static/{level}_",
            )
            spath = os.path.join(SBATCH_ROOT, f"{model_name}_human_static_{level}_eval.bash")
            with open(spath, 'w') as f:
                f.write(sbatch)
            os.chmod(spath, 0o755)
            sbatch_count += 1

    print(f"Generated {yaml_count} YAML configs")
    print(f"Generated {sbatch_count} sbatch scripts")
    print(f"YAML: {CONFIG_ROOT}")
    print(f"SBATCH: {SBATCH_ROOT}")

    # Print tree
    for root, dirs, files in os.walk(CONFIG_ROOT):
        lvl = root.replace(CONFIG_ROOT, '').count(os.sep)
        indent = '  ' * lvl
        print(f'{indent}{os.path.basename(root)}/')
        for f_name in sorted(files)[:6]:
            print(f'{indent}  {f_name}')
        if len(files) > 6:
            print(f'{indent}  ... ({len(files)} total)')

    print("\n=== Sbatch scripts ===")
    for f_name in sorted(os.listdir(SBATCH_ROOT)):
        print(f"  {f_name}")


if __name__ == "__main__":
    generate()
