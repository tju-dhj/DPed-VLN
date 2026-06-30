# DPed_STG Configuration Package
#
# This package contains configuration files for Spatio-Temporal Graph Neural Network
# enhanced Visual Language Navigation training.
#
# Configs:
#   - train/stg_ppo_train.yaml: PPO training config
#   - train/stg_ddppo_train.yaml: DDPPO distributed training config
#   - collect_data/stg_collect_train.yaml: Expert data collection config
#
# Usage:
#   PPO Training:
#     python -u -m habitat-baselines.habitat_baselines.run \
#         --config-name=DPed_STG/train/stg_ppo_train
#
#   DDPPO Training:
#     srun --unbuffered python -u -m habitat-baselines.habitat_baselines.run \
#         --config-name=DPed_STG/train/stg_ddppo_train
#
#   Data Collection:
#     python -u -m habitat-baselines.habitat_baselines.run \
#         --config-name=DPed_STG/collect_data/stg_collect_train
