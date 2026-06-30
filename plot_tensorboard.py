#!/usr/bin/env python3
"""
TensorBoard Data Plotter
Plot training curves (reward/loss) from TensorBoard event files.

Usage:
    python plot_tensorboard.py --base_dir /path/to/tb/dir
    python plot_tensorboard.py --base_dir /path/to/tb/dir --mode rl
    python plot_tensorboard.py --base_dir /path/to/tb/dir --mode il
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot training curves from TensorBoard event files"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        nargs="+",
        default=["evaluation-vln/dped_pro_clip_rl_v2_6actions"],
        help="Base directory containing tensorboard logs (default: dped_pro_clip_rl_v2_6actions). "
             "Pass two directories to compare them on the same plot.",
    )
    parser.add_argument(
        "--tb_subdir",
        type=str,
        default="hm3d/tb",
        help="Subdirectory containing tensorboard event files (default: hm3d/tb)",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="hm3d/plot",
        help="Subdirectory to save plots (default: hm3d/plot)",
    )
    parser.add_argument(
        "--checkpoint_subdir",
        type=str,
        default="hm3d/checkpoints",
        help="Subdirectory containing checkpoint files (default: hm3d/checkpoints)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "rl", "il"],
        default="auto",
        help="Training mode: auto (detect), rl (reinforcement learning), or il (imitation learning)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.6,
        help="Smoothing factor for curves (0-1, default: 0.6)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)",
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
    return parser.parse_args()


def find_all_event_files(tb_dir: Path) -> List[Path]:
    """Find all event files in the directory, sorted by modification time."""
    if isinstance(tb_dir, str):
        tb_dir = Path(tb_dir)
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {tb_dir}")
    
    # Sort by modification time (oldest to newest)
    event_files.sort(key=lambda f: f.stat().st_mtime)
    
    print(f"\nFound {len(event_files)} event files:")
    for i, f in enumerate(event_files):
        print(f"  [{i+1}] {f.name}")
        print(f"      Size: {f.stat().st_size / 1024 / 1024:.2f} MB")
    
    return event_files


def load_tensorboard_data(event_files: List[Path]) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load and merge data from multiple TensorBoard event files.
    
    Args:
        event_files: List of event file paths (sorted by time)
    
    Returns:
        Dict mapping tag name to list of (step, value) tuples
    """
    print(f"\n{'='*60}")
    print(f"Loading data from {len(event_files)} event files...")
    print(f"{'='*60}")
    
    all_data = {}
    
    for i, event_file in enumerate(event_files):
        print(f"\n[{i+1}/{len(event_files)}] Loading: {event_file.name}")
        
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            # Get all scalar tags
            tags = ea.Tags().get("scalars", [])
            print(f"  Found {len(tags)} scalar tags")
            
            # Load data for each tag
            for tag in tags:
                events = ea.Scalars(tag)
                new_data = [(e.step, e.value) for e in events]
                
                if tag not in all_data:
                    all_data[tag] = []
                
                all_data[tag].extend(new_data)
                print(f"    {tag}: +{len(new_data)} data points")
        
        except Exception as e:
            print(f"  ⚠️  Error loading {event_file.name}: {e}")
            continue
    
    # Sort and deduplicate data for each tag
    print(f"\n{'='*60}")
    print("Merging and deduplicating data...")
    print(f"{'='*60}")
    
    for tag in all_data:
        # Sort by step
        all_data[tag].sort(key=lambda x: x[0])
        
        # Remove duplicates (keep last value for each step)
        seen_steps = {}
        for step, value in all_data[tag]:
            seen_steps[step] = value
        
        all_data[tag] = [(step, value) for step, value in sorted(seen_steps.items())]
        
        print(f"  {tag}: {len(all_data[tag])} data points (after dedup)")
        if all_data[tag]:
            min_step = all_data[tag][0][0]
            max_step = all_data[tag][-1][0]
            print(f"    Step range: {min_step} -> {max_step}")
    
    return all_data


def detect_training_mode(data: Dict[str, List[Tuple[int, float]]]) -> str:
    """
    Detect training mode (RL or IL) based on available metrics.
    
    Returns:
        'rl' or 'il'
    """
    tags = list(data.keys())
    tags_lower = [tag.lower() for tag in tags]
    
    # Check for RL-specific metrics
    rl_keywords = ["reward", "return", "value", "advantage", "policy"]
    il_keywords = ["loss", "accuracy", "cross_entropy"]
    
    rl_score = sum(1 for kw in rl_keywords if any(kw in tag for tag in tags_lower))
    il_score = sum(1 for kw in il_keywords if any(kw in tag for tag in tags_lower))
    
    mode = "rl" if rl_score > il_score else "il"
    print(f"\nDetected training mode: {mode.upper()}")
    print(f"  RL score: {rl_score}, IL score: {il_score}")
    
    return mode


def smooth_curve(values: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """
    Smooth curve using exponential moving average.
    
    Args:
        values: Array of values to smooth
        weight: Smoothing factor (0-1), higher = more smoothing
    
    Returns:
        Smoothed array
    """
    if weight == 0:
        return values
    
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = smoothed[i - 1] * weight + values[i] * (1 - weight)
    return smoothed


def plot_metric(
    data: List[Tuple[int, float]],
    title: str,
    ylabel: str,
    output_path: Path,
    smooth_weight: float = 0.6,
    dpi: int = 300,
):
    """Plot a single metric."""
    if not data:
        print(f"Warning: No data for {title}")
        return
    
    steps, values = zip(*data)
    steps = np.array(steps)
    values = np.array(values)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot raw data (semi-transparent)
    plt.plot(steps, values, alpha=0.3, linewidth=0.5, color="blue", label="Raw")
    
    # Plot smoothed data
    if smooth_weight > 0:
        smoothed = smooth_curve(values, smooth_weight)
        plt.plot(steps, smoothed, linewidth=2, color="red", label="Smoothed")
    
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path.name}")
    plt.close()


def plot_combined_metrics(
    data_dict: Dict[str, List[Tuple[int, float]]],
    title: str,
    output_path: Path,
    smooth_weight: float = 0.6,
    dpi: int = 300,
):
    """Plot multiple metrics on the same figure with dual y-axes."""
    if not data_dict:
        print(f"Warning: No data for {title}")
        return
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    colors = ["blue", "red", "green", "orange", "purple"]
    axes = [ax1, ax2]
    
    for idx, (label, data) in enumerate(data_dict.items()):
        if not data:
            continue
        
        steps, values = zip(*data)
        steps = np.array(steps)
        values = np.array(values)
        
        # Choose axis (alternate between left and right)
        ax = axes[idx % 2]
        color = colors[idx % len(colors)]
        
        # Plot smoothed data
        if smooth_weight > 0:
            smoothed = smooth_curve(values, smooth_weight)
            ax.plot(steps, smoothed, linewidth=2, color=color, label=label)
        else:
            ax.plot(steps, values, linewidth=2, color=color, label=label)
        
        ax.tick_params(axis="y", labelcolor=color)
    
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Metrics", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path.name}")
    plt.close()


def plot_metric_comparison(
    runs: Dict[str, Dict[str, List[Tuple[int, float]]]],
    metric_keyword: str,
    title: str,
    ylabel: str,
    output_path: Path,
    smooth_weight: float = 0.6,
    dpi: int = 300,
):
    """Plot the same metric from multiple runs on a single figure for comparison."""
    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10.colors
    matched = 0

    for idx, (run_name, data) in enumerate(runs.items()):
        target_tag = None
        for tag in data.keys():
            if metric_keyword.lower() in tag.lower():
                target_tag = tag
                break
        if target_tag is None:
            continue
        metric_data = data[target_tag]
        if not metric_data:
            continue
        steps, values = zip(*metric_data)
        steps = np.array(steps)
        values = np.array(values)
        color = colors[idx % len(colors)]
        plt.plot(steps, values, alpha=0.25, linewidth=0.8, color=color, linestyle="--")
        if smooth_weight > 0:
            smoothed = smooth_curve(values, smooth_weight)
            plt.plot(steps, smoothed, linewidth=2.5, color=color, label=run_name)
        else:
            plt.plot(steps, values, linewidth=2.5, color=color, label=run_name)
        matched += 1

    if matched == 0:
        plt.close()
        return

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path.name}")
    plt.close()


def plot_all_comparisons(
    runs: Dict[str, Dict[str, List[Tuple[int, float]]]],
    output_dir: Path,
    smooth_weight: float = 0.6,
    dpi: int = 300,
):
    """Plot comparison charts for all common metrics across runs."""
    print("\n=== Plotting Comparison Curves ===")

    common_tags = None
    for data in runs.values():
        tags = set(data.keys())
        if common_tags is None:
            common_tags = tags
        else:
            common_tags = common_tags & tags

    if not common_tags:
        print("Warning: No common tags across runs, using all available tags")
        common_tags = set()
        for data in runs.values():
            common_tags.update(data.keys())

    reward_candidates = [t for t in common_tags if "reward" in t.lower() or "return" in t.lower()]
    loss_candidates = [t for t in common_tags if "loss" in t.lower()]
    success_candidates = [t for t in common_tags if "success" in t.lower()]
    spl_candidates = [t for t in common_tags if "spl" in t.lower()]

    for tag in reward_candidates:
        keyword = tag.split("/")[-1].lower()
        output_path = output_dir / f"compare_{tag.replace('/', '_')}.png"
        plot_metric_comparison(runs, keyword, title=f"Comparison - {tag}", ylabel="Reward",
                               output_path=output_path, smooth_weight=smooth_weight, dpi=dpi)

    for tag in loss_candidates:
        keyword = tag.split("/")[-1].lower()
        output_path = output_dir / f"compare_{tag.replace('/', '_')}.png"
        plot_metric_comparison(runs, keyword, title=f"Comparison - {tag}", ylabel="Loss",
                               output_path=output_path, smooth_weight=smooth_weight, dpi=dpi)

    for tag in success_candidates:
        keyword = tag.split("/")[-1].lower()
        output_path = output_dir / f"compare_{tag.replace('/', '_')}.png"
        plot_metric_comparison(runs, keyword, title=f"Comparison - {tag}", ylabel="Success Rate",
                               output_path=output_path, smooth_weight=smooth_weight, dpi=dpi)

    for tag in spl_candidates:
        keyword = tag.split("/")[-1].lower()
        output_path = output_dir / f"compare_{tag.replace('/', '_')}.png"
        plot_metric_comparison(runs, keyword, title=f"Comparison - {tag}", ylabel="SPL",
                               output_path=output_path, smooth_weight=smooth_weight, dpi=dpi)

    if not (reward_candidates or loss_candidates or success_candidates or spl_candidates):
        for tag in common_tags:
            keyword = tag.split("/")[-1].lower()
            output_path = output_dir / f"compare_{tag.replace('/', '_')}.png"
            plot_metric_comparison(runs, keyword, title=f"Comparison - {tag}", ylabel="Value",
                                   output_path=output_path, smooth_weight=smooth_weight, dpi=dpi)


def calculate_epoch_from_step(target_step: int, num_environments: int = 4, num_steps: int = 64, checkpoint_interval: int = 100) -> Tuple[int, int]:
    """
    Calculate epoch number from training step.
    
    Args:
        target_step: Target training step
        num_environments: Number of parallel environments
        num_steps: Steps per environment per update
        checkpoint_interval: Updates between checkpoints
    
    Returns:
        Tuple of (epoch, checkpoint_step)
    """
    steps_per_update = num_environments * num_steps
    steps_per_checkpoint = checkpoint_interval * steps_per_update
    
    epoch = target_step // steps_per_checkpoint
    checkpoint_step = epoch * steps_per_checkpoint
    
    return epoch, checkpoint_step


def find_checkpoint_for_step(checkpoint_dir: Path, target_step: int, num_environments: int = 4, num_steps: int = 64, checkpoint_interval: int = 100) -> Tuple[Path, int, int]:
    """
    Find the checkpoint file that corresponds to a given training step.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        target_step: Target training step
        num_environments: Number of parallel environments
        num_steps: Steps per environment per update
        checkpoint_interval: Updates between checkpoints (unused, kept for API compat)
    
    Returns:
        Tuple of (checkpoint_path, ckpt_count, checkpoint_step)
    """
    if not checkpoint_dir.exists():
        return None, None, None
    
    # Find all checkpoint files
    ckpt_files = []
    for f in checkpoint_dir.glob("ckpt.*.pth"):
        match = re.search(r'ckpt\.(\d+)\.pth', f.name)
        if match:
            ckpt_count = int(match.group(1))
            # Read step directly from extra_state["step"] stored in the pth file
            # This matches how dynamic_vln_trainer.py saves: step=self.num_steps_done
            try:
                ckpt_dict = torch.load(f, map_location="cpu", weights_only=False)
                step = ckpt_dict.get("extra_state", {}).get("step", None)
                if step is None:
                    step = ckpt_count * checkpoint_interval * num_environments * num_steps
                ckpt_files.append((step, f, ckpt_count))
            except Exception:
                step = ckpt_count * checkpoint_interval * num_environments * num_steps
                ckpt_files.append((step, f, ckpt_count))
    
    if not ckpt_files:
        return None, None, None
    
    # Sort by step number
    ckpt_files.sort(key=lambda x: x[0])
    
    # Find the checkpoint closest to target_step (but not exceeding it)
    best_ckpt = None
    best_step = 0
    best_count = None
    for step, path, count in ckpt_files:
        if step <= target_step:
            best_ckpt = path
            best_step = step
            best_count = count
        else:
            break
    
    return best_ckpt, best_count, best_step


def find_max_reward_checkpoint(
    data: Dict[str, List[Tuple[int, float]]],
    checkpoint_dir: Path,
    smooth_weight: float = 0.6,
    num_environments: int = 4,
    num_steps: int = 64,
    checkpoint_interval: int = 100,
) -> Dict[str, any]:
    """
    Find the checkpoint with maximum reward.
    
    Args:
        data: TensorBoard data
        checkpoint_dir: Directory containing checkpoint files
        smooth_weight: Smoothing factor for reward curve
        num_environments: Number of parallel environments
        num_steps: Steps per environment per update
        checkpoint_interval: Updates between checkpoints
    
    Returns:
        Dict containing max reward info
    """
    print("\n" + "=" * 80)
    print("=== Finding Best Checkpoint (Max Reward) ===")
    print("=" * 80)
    
    # Find main reward metric
    reward_tags = [tag for tag in data.keys() if "reward" in tag.lower()]
    
    if not reward_tags:
        print("No reward metrics found in TensorBoard data")
        return None
    
    # Use the first reward tag (usually the main reward)
    main_reward_tag = reward_tags[0]
    print(f"\nUsing reward metric: {main_reward_tag}")
    
    # Get reward data
    reward_data = data[main_reward_tag]
    steps, values = zip(*reward_data)
    steps = np.array(steps)
    values = np.array(values)
    
    # Apply smoothing if specified
    if smooth_weight > 0:
        smoothed_values = smooth_curve(values, smooth_weight)
        print(f"Using smoothed values (weight={smooth_weight})")
    else:
        smoothed_values = values
        print("Using raw values (no smoothing)")
    
    # Find maximum reward
    max_idx = np.argmax(smoothed_values)
    max_reward = smoothed_values[max_idx]
    max_step = steps[max_idx]
    
    print(f"\nMax reward: {max_reward:.4f}")
    print(f"At step: {max_step}")
    
    # Calculate expected epoch
    expected_epoch, expected_step = calculate_epoch_from_step(max_step, num_environments, num_steps, checkpoint_interval)
    print(f"Expected epoch: {expected_epoch}")
    print(f"Expected checkpoint step: {expected_step}")
    
    # Find corresponding checkpoint
    best_ckpt, ckpt_count, ckpt_step = find_checkpoint_for_step(checkpoint_dir, max_step, num_environments, num_steps, checkpoint_interval)
    
    result = {
        "reward_tag": main_reward_tag,
        "max_reward": max_reward,
        "max_step": max_step,
        "expected_epoch": expected_epoch,
        "expected_step": expected_step,
        "checkpoint_path": best_ckpt,
        "checkpoint_count": ckpt_count,
        "checkpoint_step": ckpt_step,
    }
    
    if best_ckpt:
        print(f"\nClosest available checkpoint: {best_ckpt.name}")
        print(f"Checkpoint count: {ckpt_count}")
        print(f"Checkpoint step: {ckpt_step}")
        
        if ckpt_step != max_step:
            step_diff = max_step - ckpt_step
            print(f"\nNote: Checkpoint is {step_diff} steps before max reward step")
            if ckpt_count is not None and ckpt_count < expected_epoch:
                print(f"      Training may still be in progress (expected epoch {expected_epoch}, found count {ckpt_count})")
    else:
        print(f"\nWarning: No checkpoint found for step {max_step}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Expected checkpoint: ckpt.{expected_epoch}.pth")
        
        # List available checkpoints
        if checkpoint_dir.exists():
            ckpt_files = sorted(checkpoint_dir.glob("ckpt.*.pth"))
            if ckpt_files:
                print(f"\nAvailable checkpoints: {len(ckpt_files)} files")
                print(f"First: {ckpt_files[0].name}")
                print(f"Last: {ckpt_files[-1].name}")
    
    # Print all reward metrics at max step
    print("\n" + "-" * 80)
    print("All reward metrics at best step:")
    print("-" * 80)
    for tag in reward_tags:
        tag_data = data[tag]
        # Find value closest to max_step
        tag_steps, tag_values = zip(*tag_data)
        tag_steps = np.array(tag_steps)
        tag_values = np.array(tag_values)
        
        # Find closest step
        closest_idx = np.argmin(np.abs(tag_steps - max_step))
        closest_step = tag_steps[closest_idx]
        closest_value = tag_values[closest_idx]
        
        print(f"  {tag}: {closest_value:.4f} (at step {closest_step})")
    
    return result


def plot_rl_curves(
    data: Dict[str, List[Tuple[int, float]]],
    output_dir: Path,
    checkpoint_dir: Path,
    smooth_weight: float = 0.6,
    dpi: int = 300,
    num_environments: int = 4,
    num_steps: int = 64,
    checkpoint_interval: int = 100,
):
    """Plot RL training curves (reward and loss)."""
    print("\n=== Plotting RL Curves ===")
    
    # Find reward-related metrics
    reward_tags = [tag for tag in data.keys() if any(kw in tag.lower() for kw in ["reward", "return"])]
    loss_tags = [tag for tag in data.keys() if "loss" in tag.lower()]
    
    print(f"Reward tags: {reward_tags}")
    print(f"Loss tags: {loss_tags}")
    
    # Find best checkpoint
    best_info = find_max_reward_checkpoint(data, checkpoint_dir, smooth_weight, num_environments, num_steps, checkpoint_interval)
    
    # Save best checkpoint info to file
    if best_info:
        info_file = output_dir / "best_checkpoint_info.txt"
        with open(info_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Best Checkpoint Information (Max Reward)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training Configuration:\n")
            f.write(f"  num_environments: {num_environments}\n")
            f.write(f"  num_steps: {num_steps}\n")
            f.write(f"  checkpoint_interval: {checkpoint_interval} updates\n")
            f.write(f"  steps_per_checkpoint: {num_environments * num_steps * checkpoint_interval}\n\n")
            f.write(f"Reward Metric: {best_info['reward_tag']}\n")
            f.write(f"Max Reward: {best_info['max_reward']:.4f}\n")
            f.write(f"At Step: {best_info['max_step']}\n")
            f.write(f"Expected Epoch: {best_info['expected_epoch']}\n")
            f.write(f"Expected Checkpoint Step: {best_info['expected_step']}\n\n")
            
            if best_info['checkpoint_path']:
                f.write(f"Closest Available Checkpoint:\n")
                f.write(f"  File: {best_info['checkpoint_path'].name}\n")
                f.write(f"  Full Path: {best_info['checkpoint_path']}\n")
                if best_info['checkpoint_count'] is not None:
                    f.write(f"  Count: {best_info['checkpoint_count']}\n")
                f.write(f"  Step: {best_info['checkpoint_step']}\n")
                
                if best_info['checkpoint_step'] != best_info['max_step']:
                    step_diff = best_info['max_step'] - best_info['checkpoint_step']
                    f.write(f"\n  Note: Checkpoint is {step_diff} steps before max reward step\n")
                    if best_info['checkpoint_count'] is not None and best_info['checkpoint_count'] < best_info['expected_epoch']:
                        f.write(f"        Training may still be in progress\n")
            else:
                f.write(f"Warning: No checkpoint found\n")
                f.write(f"Expected checkpoint: ckpt.{best_info['expected_epoch']}.pth\n")
            
            f.write("\n" + "=" * 80 + "\n")
        print(f"\nSaved best checkpoint info to: {info_file}")
    
    # Plot individual reward curves
    for tag in reward_tags:
        output_path = output_dir / f"reward_{tag.replace('/', '_')}.png"
        plot_metric(
            data[tag],
            title=f"RL Training - {tag}",
            ylabel="Reward",
            output_path=output_path,
            smooth_weight=smooth_weight,
            dpi=dpi,
        )
    
    # Plot individual loss curves
    for tag in loss_tags:
        output_path = output_dir / f"loss_{tag.replace('/', '_')}.png"
        plot_metric(
            data[tag],
            title=f"RL Training - {tag}",
            ylabel="Loss",
            output_path=output_path,
            smooth_weight=smooth_weight,
            dpi=dpi,
        )
    
    # Plot combined reward and loss
    if reward_tags and loss_tags:
        combined_data = {}
        if reward_tags:
            combined_data[reward_tags[0]] = data[reward_tags[0]]
        if loss_tags:
            combined_data[loss_tags[0]] = data[loss_tags[0]]
        
        output_path = output_dir / "combined_reward_loss.png"
        plot_combined_metrics(
            combined_data,
            title="RL Training - Reward and Loss",
            output_path=output_path,
            smooth_weight=smooth_weight,
            dpi=dpi,
        )


def plot_il_curves(
    data: Dict[str, List[Tuple[int, float]]],
    output_dir: Path,
    smooth_weight: float = 0.6,
    dpi: int = 300,
):
    """Plot IL training curves (loss)."""
    print("\n=== Plotting IL Curves ===")
    
    # Find loss-related metrics
    loss_tags = [tag for tag in data.keys() if "loss" in tag.lower()]
    
    print(f"Loss tags: {loss_tags}")
    
    # Plot individual loss curves
    for tag in loss_tags:
        output_path = output_dir / f"loss_{tag.replace('/', '_')}.png"
        plot_metric(
            data[tag],
            title=f"IL Training - {tag}",
            ylabel="Loss",
            output_path=output_path,
            smooth_weight=smooth_weight,
            dpi=dpi,
        )


def main():
    """Main function."""
    args = parse_args()
    
    base_dirs = [Path(d) for d in args.base_dir]
    compare_mode = len(base_dirs) >= 2
    
    print("=" * 80)
    print("TensorBoard Data Plotter")
    print("=" * 80)
    
    if compare_mode:
        print(f"COMPARE mode: {len(base_dirs)} directories")
        for d in base_dirs:
            print(f"  - {d}")
    else:
        print(f"Base directory: {base_dirs[0]}")
    
    print(f"Mode: {args.mode}")
    print(f"Smoothing: {args.smooth}")
    print(f"DPI: {args.dpi}")
    
    if not compare_mode:
        # Original single-directory behavior
        base_dir = base_dirs[0]
        tb_dir = base_dir / args.tb_subdir
        output_dir = base_dir / args.output_subdir
        checkpoint_dir = base_dir / args.checkpoint_subdir
        
        print(f"TensorBoard directory: {tb_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        
        if not tb_dir.exists():
            raise FileNotFoundError(f"TensorBoard directory not found: {tb_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        event_files = find_all_event_files(tb_dir)
        data = load_tensorboard_data(event_files)
        
        if not data:
            print("Error: No data loaded from TensorBoard file")
            return
        
        mode = args.mode if args.mode != "auto" else detect_training_mode(data)
        
        if mode == "rl":
            plot_rl_curves(data, output_dir, checkpoint_dir, args.smooth, args.dpi,
                           args.num_environments, args.num_steps, args.checkpoint_interval)
        elif mode == "il":
            plot_il_curves(data, output_dir, args.smooth, args.dpi)
        else:
            print(f"Unknown mode: {mode}")
            return
    else:
        # Compare mode: load all runs
        runs = {}  # {run_name: data}
        tb_dirs = []
        
        for i, base_dir in enumerate(base_dirs):
            tb_dir = base_dir / args.tb_subdir
            if not tb_dir.exists():
                print(f"Warning: TensorBoard directory not found: {tb_dir}, skipping")
                continue
            
            # Determine run name from directory
            run_name = base_dir.name if base_dirs[i] != Path(".") else f"run{i+1}"
            print(f"\nLoading run '{run_name}': {tb_dir}")
            
            try:
                event_files = find_all_event_files(tb_dir)
                data = load_tensorboard_data(event_files)
                if data:
                    runs[run_name] = data
                    tb_dirs.append(tb_dir)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        if len(runs) < 2:
            print("Error: Need at least 2 valid runs to compare")
            return
        
        # Output to first base_dir / compare_subdir
        compare_output_dir = base_dirs[0] / "compare"
        compare_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nComparison output directory: {compare_output_dir}")
        
        plot_all_comparisons(runs, compare_output_dir, args.smooth, args.dpi)
    
    print("\n" + "=" * 80)
    print("Done! All plots saved to:", output_dir if not compare_mode else compare_output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
