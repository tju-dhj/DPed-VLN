#!/usr/bin/env python3
"""
Generate Trajectory Length Distribution Visualization
Standalone script for creating publication-quality trajectory length histogram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set professional academic style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts for PDF
plt.rcParams['ps.fonttype'] = 42

sns.set_style("whitegrid")


def plot_trajectory_length_distribution(csv_path, output_path, max_steps=500):
    """
    Plot trajectory length distribution with filtering
    
    Args:
        csv_path: Path to CSV file with episode data
        output_path: Path to save output PDF
        max_steps: Maximum steps to include (default: 500)
    """
    print("="*80)
    print("Generating Trajectory Length Distribution")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total episodes loaded: {len(df)}")
    
    # Check if num_steps column exists
    if 'num_steps' not in df.columns:
        print(f"Error: 'num_steps' column not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Use only training data
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
    else:
        # If no split column, use all data as train
        train_df = df
    
    print(f"  Training episodes: {len(train_df)}")
    
    # Filter out episodes with steps > max_steps
    train_steps = train_df['num_steps'].values
    train_steps_filtered = train_steps[(train_steps > 0) & (train_steps <= max_steps)]
    
    print(f"\nAfter filtering (≤{max_steps} steps):")
    print(f"  Training episodes: {len(train_steps_filtered)} ({len(train_steps_filtered)/len(train_steps)*100:.1f}%)")
    
    # Calculate statistics
    train_stats = {
        'mean': np.mean(train_steps_filtered),
        'median': np.median(train_steps_filtered),
        'std': np.std(train_steps_filtered),
        'min': np.min(train_steps_filtered),
        'max': np.max(train_steps_filtered),
        'count': len(train_steps_filtered)
    }
    
    print(f"\nTraining Set Statistics (≤{max_steps} steps):")
    print(f"  Episodes: {train_stats['count']}")
    print(f"  Mean: {train_stats['mean']:.2f} steps")
    print(f"  Median: {train_stats['median']:.2f} steps")
    print(f"  Std: {train_stats['std']:.2f} steps")
    print(f"  Range: [{train_stats['min']:.0f}, {train_stats['max']:.0f}] steps")
    
    # Create single figure for training data only
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    datasets = [
        ('Training Set', train_steps_filtered, train_stats, ax)
    ]
    
    # Plot each dataset
    for title, steps, stats, ax in datasets:
        # Create histogram
        n, bins, patches = ax.hist(
            steps, 
            bins=30, 
            alpha=0.75, 
            color='steelblue',
            edgecolor='black', 
            linewidth=0.5
        )
        
        # Get max histogram height for positioning
        max_hist_height = np.max(n)
        
        # Add mean line
        mean_val = stats['mean']
        ax.axvline(
            mean_val, 
            color='red', 
            linestyle='--', 
            linewidth=2.5, 
            label=f'Mean: {mean_val:.1f}',
            zorder=10
        )
        
        # Add median line
        median_val = stats['median']
        ax.axvline(
            median_val, 
            color='orange', 
            linestyle='-.', 
            linewidth=2.5, 
            label=f'Median: {median_val:.1f}',
            zorder=10
        )
        
        # Add statistics text box
        # Position in upper right corner, well above the histogram bars
        stats_text = (
            f"Episodes: {stats['count']:,}\n"
            f"Std: {stats['std']:.1f}\n"
            f"Min: {stats['min']:.0f}\n"
            f"Max: {stats['max']:.0f}"
        )
        
        # Position text box in upper right corner with more padding
        ax.text(
            0.98, 0.98, 
            stats_text, 
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(
                boxstyle='round,pad=0.7', 
                facecolor='wheat', 
                alpha=0.85,
                edgecolor='black',
                linewidth=1
            ),
            fontsize=10,
            family='monospace',
            zorder=100  # Ensure text is on top
        )
        
        # Set labels and title
        ax.set_xlabel('Trajectory Length (Steps)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Episodes', fontsize=13, fontweight='bold')
        ax.set_title(
            f'{title} - Trajectory Length Distribution\n(Episodes with ≤{max_steps} steps)', 
            fontsize=14, 
            fontweight='bold',
            pad=15
        )
        
        # Add legend in upper left corner (opposite to stats box)
        ax.legend(
            loc='upper left', 
            fontsize=11,
            framealpha=0.92,
            edgecolor='black',
            fancybox=True,
            shadow=True
        )
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        
        # Set axis limits with proper padding
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, max_hist_height * 1.15)  # Add 15% padding on top for text box
    
    # No overall title needed for single plot
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    plt.close()
    
    print("\n" + "="*80)
    print("Trajectory Length Distribution Generation Complete!")
    print("="*80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate trajectory length distribution visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using default paths
  python plot_trajectory_distribution.py
  
  # Specifying custom paths and max steps
  python plot_trajectory_distribution.py \\
    --csv all_episodes.csv \\
    --output trajectory_distribution.pdf \\
    --max_steps 400
        """
    )
    
    parser.add_argument(
        '--csv', 
        type=str,
        default='dataset_analysis_academic/all_episodes.csv',
        help='Path to CSV file with episode data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='trajectory_length_distribution.pdf',
        help='Output PDF file path'
    )
    
    parser.add_argument(
        '--max_steps',
        type=int,
        default=500,
        help='Maximum steps to include in visualization (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        exit(1)
    
    # Generate plot
    plot_trajectory_length_distribution(
        csv_path=args.csv,
        output_path=args.output,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()

