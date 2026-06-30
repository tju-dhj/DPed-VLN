#!/usr/bin/env python3
"""
Distribution comparison charts for dataset splits.
Generates action length and instruction length histograms for all 5 splits.
"""

import gzip
import json
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# Configuration
SPLITS_DIR = "/share/home/u19666033/dhj/DPed_pro/dataset_splits"
SPLITS = ["seen_val", "seen_test", "unseen_val", "unseen_test", "train"]
OUTPUT_PATH = "/share/home/u19666033/dhj/DPed_pro/dataset_splits/distribution_comparison.png"

# Professional color palette (pastel, distinct per split)
SPLIT_COLORS = {
    "seen_val": "#4C72B0",      # Blue
    "seen_test": "#8172B3",    # Purple-blue
    "unseen_val": "#DD8452",   # Orange
    "unseen_test": "#937860",  # Brown-orange
    "train": "#55A868",        # Green
}

# Light fill colors for KDE (pastel versions)
SPLIT_COLORS_LIGHT = {
    "seen_val": "#4C72B0",      # Blue
    "seen_test": "#8172B3",    # Purple-blue
    "unseen_val": "#DD8452",   # Orange
    "unseen_test": "#937860",  # Brown-orange
    "train": "#55A868",        # Green
}


def load_split_data(split_name: str) -> Tuple[List[int], List[int]]:
    """
    Load all episodes from a split and compute action_len and inst_len.
    
    Returns:
        action_lens: List of action lengths
        inst_lens: List of instruction lengths
    """
    split_dir = os.path.join(SPLITS_DIR, split_name)
    action_lens = []
    inst_lens = []
    
    # Find all .json.gz files
    pattern = os.path.join(split_dir, "*.json.gz")
    files = glob(pattern)
    
    for filepath in tqdm(files, desc=f"Loading {split_name}", leave=False):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            data = json.load(f)
        
        episodes = data.get("episodes", [])
        for ep in episodes:
            gt_action = ep.get("gt_action", [])
            instruction = ep.get("instruction", [])
            
            action_len = max(0, len(gt_action) - 1)
            inst_len = len(instruction)
            
            action_lens.append(action_len)
            inst_lens.append(inst_len)
    
    return action_lens, inst_lens


def compute_statistics(data: List[int]) -> Dict[str, float]:
    """Compute summary statistics for a dataset."""
    arr = np.array(data)
    return {
        "min": np.min(arr),
        "max": np.max(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
        "p25": np.percentile(arr, 25),
        "p50": np.percentile(arr, 50),
        "p75": np.percentile(arr, 75),
    }


def plot_kde_with_histogram(
    ax, 
    data_dict: Dict[str, List[int]], 
    metric: str,
    x_limit: int = None
) -> None:
    """
    Plot overlaid KDE histograms for multiple splits.
    
    Args:
        ax: Matplotlib axis
        data_dict: Dict mapping split name to list of values
        metric: Name of the metric (for labels)
        x_limit: Maximum x-value to show (for instruction length)
    """
    handles = []
    labels = []
    
    for split in SPLITS:
        data = np.array(data_dict[split])
        color = SPLIT_COLORS[split]
        count = len(data)
        
        # Plot KDE using scipy for cleaner control
        kde = stats.gaussian_kde(data, bw_method='scott')
        x_range = np.linspace(0, x_limit if x_limit else data.max() * 1.1, 500)
        y_kde = kde(x_range)
        
        # Plot filled KDE curve
        ax.fill_between(x_range, y_kde, alpha=0.3, color=color)
        ax.plot(x_range, y_kde, color=color, linewidth=2.5)
        
        # Create custom legend entry with count
        handle, = ax.plot([], [], color=color, linewidth=3, label=f"{split} (n={count:,})")
        handles.append(handle)
        labels.append(f"{split} (n={count:,})")
    
    ax.set_xlabel(metric, fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    
    if x_limit is not None:
        ax.set_xlim(0, x_limit)
    
    ax.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    print("=" * 60)
    print("Dataset Split Distribution Analysis")
    print("=" * 60)
    
    # Load all splits
    all_action_lens = {}
    all_inst_lens = {}
    
    for split in SPLITS:
        action_lens, inst_lens = load_split_data(split)
        all_action_lens[split] = action_lens
        all_inst_lens[split] = inst_lens
        print(f"  {split}: {len(action_lens):,} episodes")
    
    print()
    
    # Print summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for split in SPLITS:
        print(f"\n{split.upper()}")
        print("-" * 40)
        
        action_stats = compute_statistics(all_action_lens[split])
        inst_stats = compute_statistics(all_inst_lens[split])
        
        print(f"  Action Length:")
        print(f"    min={action_stats['min']:.0f}, max={action_stats['max']:.0f}, "
              f"mean={action_stats['mean']:.1f}, std={action_stats['std']:.1f}")
        print(f"    p25={action_stats['p25']:.1f}, p50={action_stats['p50']:.1f}, p75={action_stats['p75']:.1f}")
        
        print(f"  Instruction Length:")
        print(f"    min={inst_stats['min']:.0f}, max={inst_stats['max']:.0f}, "
              f"mean={inst_stats['mean']:.1f}, std={inst_stats['std']:.1f}")
        print(f"    p25={inst_stats['p25']:.1f}, p50={inst_stats['p50']:.1f}, p75={inst_stats['p75']:.1f}")
    
    # Create figure with 2 subplots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    
    # Left plot: Action length histogram
    plot_kde_with_histogram(
        axes[0],
        all_action_lens,
        "Action Length (max(0, len(gt_action) - 1))"
    )
    axes[0].set_title("Action Length Distribution by Split", fontsize=14, fontweight="bold", pad=10)
    
    # Right plot: Instruction length histogram (capped at 1000)
    plot_kde_with_histogram(
        axes[1],
        all_inst_lens,
        "Instruction Length (character count)",
        x_limit=1000
    )
    axes[1].set_title("Instruction Length Distribution by Split (x ≤ 1000)", fontsize=14, fontweight="bold", pad=10)
    
    # Main title
    fig.suptitle(
        "Dataset Split Distribution Comparison\n"
        "(Blue/Purple = Seen, Orange/Brown = Unseen, Green = Train)",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none"
    )
    
    print(f"\nFigure saved to: {OUTPUT_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
