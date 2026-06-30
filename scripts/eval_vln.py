#!/usr/bin/env python3

"""
VLN Evaluation Script
Evaluates trained VLN agents.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from habitat_baselines.run import execute_exp
from habitat_baselines.config.default import get_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLN agents")
    parser.add_argument(
        "--config-path",
        type=str,
        default="habitat-lab/habitat/config/benchmark/nav/socialnav_v2/vln_hm3d_train.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="vln_hm3d_eval",
        help="Experiment name"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/filtered_dataset",
        help="Path to filtered dataset"
    )
    
    args = parser.parse_args()
    
    # Get config
    config = get_config(
        args.config_path,
        [
            f"habitat_baselines.trainer_name=vln_il",
            f"habitat_baselines.num_processes={args.num_processes}",
            f"habitat_baselines.num_gpus={args.num_gpus}",
            f"habitat_baselines.exp_name={args.exp_name}",
            f"habitat_baselines.dataset_path={args.dataset_path}",
            f"habitat_baselines.eval.use_ckpt_path=true",
            f"habitat_baselines.eval.ckpt_path={args.checkpoint_path}",
        ]
    )
    
    # Execute evaluation
    execute_exp(config, "eval")


if __name__ == "__main__":
    main()



