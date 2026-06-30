#!/usr/bin/env python3
"""
引导脚本: 初始化环境并保存完整的 pickle 文件（包括 orig_action_space）
运行一次即可生成所有需要的 pickle 文件。
运行方式: cd /share/home/u19666033/dhj/DPed_pro/habitat-baselines && python bootstrap_pickle.py
"""

import os
import sys
import pickle

# Ensure we can import
sys.path.insert(0, '/share/home/u19666033/dhj/DPed_pro/habitat-baselines')

from habitat_baselines.rl.ppo.dped_trainer_server import DPedTrainerServer
from habitat.config import read_write
from omegaconf import OmegaConf
import hydra

# Load config
from habitat_baselines.run import main
# We'll create a minimal config and run init_envs

if __name__ == "__main__":
    import habitat_baselines.run
    from habitat_baselines.run import execute_exp

    # Manually build config
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()

    hydra.initialize_config_dir(
        config_dir="/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines/config",
        job_name="bootstrap_pickle"
    )

    cfg = hydra.compose(config_name="DPed_brain_new/robot_deploy/robot_deploy_4a")

    # Set up trainer
    trainer = DPedTrainerServer(config=cfg)
    trainer.device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0" else "cuda:0"

    config = cfg.copy()
    with read_write(config):
        config.habitat.dataset.split = "val"

    # Init envs
    print("Initializing environments (this may take a minute)...")
    trainer._init_envs(config, is_eval=True)

    # Save all three pickle files
    obs_space = trainer._env_spec.observation_space
    act_space = trainer._env_spec.action_space
    orig_act_space = trainer._env_spec.orig_action_space

    with open("observation_space.pkl", "wb") as f:
        pickle.dump(obs_space, f)
    with open("action_space.pkl", "wb") as f:
        pickle.dump(act_space, f)
    with open("orig_action_space.pkl", "wb") as f:
        pickle.dump(orig_act_space, f)

    print("Saved:")
    print(f"  observation_space: {type(obs_space)} keys={list(obs_space.spaces.keys())[:5]}...")
    print(f"  action_space: {type(act_space)}")
    print(f"  orig_action_space: {type(orig_act_space)} keys={list(orig_act_space.spaces.keys())[:5]}...")

    trainer.envs.close()
    print("Done!")
