#!/usr/bin/env python3
"""
Find the checkpoint with the highest reward from TensorBoard data.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Find best checkpoint by reward")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="evaluation-vln/dynamic_vlnce_clip_rl_v2",
        help="Base directory containing tensorboard logs and checkpoints",
    )
    parser.add_argument(
        "--tb_subdir",
        type=str,
        default="hm3d/tb",
        help="Subdirectory containing tensorboard event files (default: hm3d/tb)",
    )
    parser.add_argument(
        "--checkpoint_subdir",
        type=str,
        default="hm3d/checkpoints_vln_new",
        help="Subdirectory containing checkpoint files (default: hm3d/checkpoints_vln_new)",
    )
    parser.add_argument(
        "--num_environments",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=64,
        help="Steps per environment per update (default: 64)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Updates between checkpoints (default: 100)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Smoothing factor for curves (0-1, 0 for raw values)",
    )
    return parser.parse_args()


def smooth_curve(values: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """Exponential moving average smoothing."""
    if weight <= 0:
        return values
    
    smoothed = np.zeros_like(values)
    last = values[0]
    for i, val in enumerate(values):
        smoothed_val = last * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        last = smoothed_val
    return smoothed


def load_tensorboard_data(event_files: List[Path]) -> Dict[str, List[Tuple[int, float]]]:
    """Load and merge data from multiple TensorBoard event files."""
    print(f"\nLoading TensorBoard data from {len(event_files)} files...")
    
    merged_data = {}
    
    for event_file in event_files:
        print(f"  Loading: {event_file.name}")
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        
        # Get all scalar tags
        tags = ea.Tags()["scalars"]
        
        for tag in tags:
            events = ea.Scalars(tag)
            if tag not in merged_data:
                merged_data[tag] = []
            
            for event in events:
                merged_data[tag].append((event.step, event.value))
    
    # Sort each tag's data by step
    for tag in merged_data:
        merged_data[tag].sort(key=lambda x: x[0])
    
    print(f"\nLoaded {len(merged_data)} metrics")
    return merged_data


def find_all_event_files(tb_dir: Path) -> List[Path]:
    """Find all event files in the directory."""
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {tb_dir}")
    
    event_files.sort(key=lambda f: f.stat().st_mtime)
    print(f"\nFound {len(event_files)} event files")
    return event_files


def get_checkpoint_info(checkpoint_dir: Path, num_environments: int, num_steps: int, checkpoint_interval: int) -> List[Tuple[int, int, Path]]:
    """Get all checkpoint files with their epoch and step information."""
    steps_per_checkpoint = checkpoint_interval * num_environments * num_steps
    
    ckpt_files = []
    for f in sorted(checkpoint_dir.glob("ckpt.*.pth")):
        match = re.search(r'ckpt\.(\d+)\.pth', f.name)
        if match:
            epoch = int(match.group(1))
            step = epoch * steps_per_checkpoint
            ckpt_files.append((epoch, step, f))
    
    return sorted(ckpt_files, key=lambda x: x[0])


def get_reward_at_step(reward_data: List[Tuple[int, float]], target_step: int, window: int = 1000) -> Tuple[float, int]:
    """Get reward value at or near a target step."""
    steps, values = zip(*reward_data)
    steps = np.array(steps)
    values = np.array(values)
    
    # Find closest step within window
    mask = np.abs(steps - target_step) <= window
    if not mask.any():
        return None, None
    
    closest_idx = np.argmin(np.abs(steps[mask] - target_step))
    valid_steps = steps[mask]
    valid_values = values[mask]
    
    return valid_values[closest_idx], valid_steps[closest_idx]


def main():
    """Main function."""
    args = parse_args()
    
    # Setup paths
    base_dir = Path(args.base_dir)
    tb_dir = base_dir / args.tb_subdir
    checkpoint_dir = base_dir / args.checkpoint_subdir
    
    print("=" * 80)
    print("Find Best Checkpoint by Reward")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"TensorBoard directory: {tb_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"\nTraining Configuration:")
    print(f"  num_environments: {args.num_environments}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  checkpoint_interval: {args.checkpoint_interval} updates")
    steps_per_checkpoint = args.checkpoint_interval * args.num_environments * args.num_steps
    print(f"  steps_per_checkpoint: {steps_per_checkpoint}")
    
    # Validate directories
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Load TensorBoard data
    event_files = find_all_event_files(tb_dir)
    data = load_tensorboard_data(event_files)
    
    # Find reward metrics
    reward_tags = [tag for tag in data.keys() if "reward" in tag.lower()]
    if not reward_tags:
        print("Error: No reward metrics found in TensorBoard data")
        return
    
    main_reward_tag = reward_tags[0]
    print(f"\nUsing reward metric: {main_reward_tag}")
    print(f"Available reward tags: {reward_tags}")
    
    # Get reward data
    reward_data = data[main_reward_tag]
    steps, values = zip(*reward_data)
    steps = np.array(steps)
    values = np.array(values)
    
    # Apply smoothing if specified
    if args.smooth > 0:
        values = smooth_curve(values, args.smooth)
        print(f"Applied smoothing (weight={args.smooth})")
    
    # Get all checkpoints
    checkpoints = get_checkpoint_info(checkpoint_dir, args.num_environments, args.num_steps, args.checkpoint_interval)
    print(f"\nFound {len(checkpoints)} checkpoint files")
    
    # Evaluate each checkpoint
    print("\n" + "=" * 80)
    print("Checkpoint Evaluation")
    print("=" * 80)
    print(f"{'Epoch':<8} {'Step':<12} {'Reward':<12} {'Actual Step':<12} {'Filename'}")
    print("-" * 80)
    
    best_epoch = None
    best_reward = -float('inf')
    best_ckpt = None
    best_step = None
    
    for epoch, step, ckpt_path in checkpoints:
        reward, actual_step = get_reward_at_step(reward_data, step, window=steps_per_checkpoint)
        
        if reward is not None:
            print(f"{epoch:<8} {step:<12} {reward:<12.4f} {actual_step:<12} {ckpt_path.name}")
            
            if reward > best_reward:
                best_reward = reward
                best_epoch = epoch
                best_ckpt = ckpt_path
                best_step = actual_step
        else:
            print(f"{epoch:<8} {step:<12} {'N/A':<12} {'N/A':<12} {ckpt_path.name}")
    
    # Print best checkpoint
    print("\n" + "=" * 80)
    print("Best Checkpoint")
    print("=" * 80)
    if best_ckpt:
        print(f"Epoch: {best_epoch}")
        print(f"Expected Step: {best_epoch * steps_per_checkpoint}")
        print(f"Actual Step: {best_step}")
        print(f"Reward: {best_reward:.4f}")
        print(f"Filename: {best_ckpt.name}")
        print(f"Full Path: {best_ckpt}")
        
        # Show all reward metrics at this step
        print(f"\nAll reward metrics at step {best_step}:")
        for tag in reward_tags:
            reward, actual = get_reward_at_step(data[tag], best_step, window=steps_per_checkpoint)
            if reward is not None:
                print(f"  {tag}: {reward:.4f}")
    else:
        print("No valid checkpoint found with reward data")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
