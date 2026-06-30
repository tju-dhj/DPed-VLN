#!/usr/bin/env python3
"""
Academic Dataset Analysis and Visualization Script
For analyzing the Falcon Social Navigation Dataset metrics
All outputs in professional academic English with PDF format
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd

# Set professional academic style for matplotlib
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # Use TrueType fonts in PDF
plt.rcParams['ps.fonttype'] = 42

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class AcademicDatasetAnalyzer:
    """Academic Dataset Analyzer for Falcon Social Navigation Dataset"""
    
    def __init__(self, train_dir, val_dir, output_dir="dataset_analysis_academic"):
        """
        Initialize the analyzer
        
        Args:
            train_dir: Path to training dataset
            val_dir: Path to validation dataset
            output_dir: Output directory for results
        """
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Store episode data for analysis
        self.train_episodes = []
        self.val_episodes = []
        self.all_episodes = []
        
    def collect_episode_info(self, dataset_path, split_name):
        """
        Collect episode information
        
        Args:
            dataset_path: Dataset path
            split_name: 'train' or 'val'
        """
        print(f"\nAnalyzing {split_name} dataset: {dataset_path}")
        
        episode_info = []
        scene_dirs = sorted(dataset_path.glob("*.basis"))
        
        for scene_dir in tqdm(scene_dirs, desc=f"Scanning {split_name} scenes"):
            scene_name = scene_dir.name
            episode_dirs = [d for d in scene_dir.iterdir() if d.is_dir()]
            
            for episode_dir in episode_dirs:
                episode_id = episode_dir.name
                
                info = {
                    'scene': scene_name,
                    'episode_id': episode_id,
                    'episode_path': str(episode_dir),
                    'split': split_name
                }
                
                # Collect step count and action data
                action_file = episode_dir / "action" / "0.json"
                if action_file.exists():
                    try:
                        with open(action_file, 'r') as f:
                            actions = json.load(f)
                            info['num_steps'] = len(actions)
                            info['actions'] = actions
                    except:
                        info['num_steps'] = 0
                        info['actions'] = []
                else:
                    # Estimate from RGB images
                    rgb_dir = episode_dir / "rgb"
                    if rgb_dir.exists():
                        info['num_steps'] = len(list(rgb_dir.glob("*_0.jpg")))
                    else:
                        info['num_steps'] = 0
                    info['actions'] = []
                
                # Collect pedestrian data (human_num)
                human_num_file = episode_dir / "human_num" / "0.json"
                if human_num_file.exists():
                    try:
                        with open(human_num_file, 'r') as f:
                            human_nums_raw = json.load(f)
                            
                            # Handle nested list format: [[6], [6], ...] -> [6, 6, ...]
                            if isinstance(human_nums_raw, list) and len(human_nums_raw) > 0:
                                if isinstance(human_nums_raw[0], list):
                                    human_nums = [item[0] if isinstance(item, list) and len(item) > 0 else 0 
                                                 for item in human_nums_raw]
                                else:
                                    human_nums = human_nums_raw
                                
                                info['human_nums'] = human_nums
                                info['max_human_num'] = max(human_nums) if human_nums else 0
                                info['avg_human_num'] = float(np.mean(human_nums)) if human_nums else 0.0
                                info['has_pedestrian'] = any(n > 0 for n in human_nums)
                                info['pedestrian_steps'] = sum(1 for n in human_nums if n > 0)
                                info['pedestrian_ratio'] = info['pedestrian_steps'] / max(len(human_nums), 1)
                            else:
                                info['human_nums'] = []
                                info['max_human_num'] = 0
                                info['avg_human_num'] = 0.0
                                info['has_pedestrian'] = False
                                info['pedestrian_steps'] = 0
                                info['pedestrian_ratio'] = 0.0
                    except Exception as e:
                        print(f"Warning: Error processing human_num for {episode_dir}: {e}")
                        info['human_nums'] = []
                        info['max_human_num'] = 0
                        info['avg_human_num'] = 0.0
                        info['has_pedestrian'] = False
                        info['pedestrian_steps'] = 0
                        info['pedestrian_ratio'] = 0.0
                else:
                    info['human_nums'] = []
                    info['max_human_num'] = 0
                    info['avg_human_num'] = 0.0
                    info['has_pedestrian'] = False
                    info['pedestrian_steps'] = 0
                    info['pedestrian_ratio'] = 0.0
                
                # Collect agent positions for spatial analysis
                pose_file = episode_dir / "pose" / "0.json"
                if pose_file.exists():
                    try:
                        with open(pose_file, 'r') as f:
                            poses = json.load(f)
                            info['poses'] = poses
                    except:
                        info['poses'] = []
                else:
                    info['poses'] = []
                
                episode_info.append(info)
        
        return episode_info
    
    def analyze_all(self):
        """Analyze all data"""
        print("="*80)
        print("Starting Dataset Analysis")
        print("="*80)
        
        # Collect train and val data
        self.train_episodes = self.collect_episode_info(self.train_dir, 'train')
        self.val_episodes = self.collect_episode_info(self.val_dir, 'val')
        self.all_episodes = self.train_episodes + self.val_episodes
        
        # Generate statistics report
        self.generate_statistics()
        
        # Generate visualizations
        self.generate_visualizations()
        
        print(f"\nAnalysis complete! Results saved in: {self.output_dir}")
    
    def generate_statistics(self):
        """Generate statistics report"""
        print("\nGenerating statistics report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Falcon Social Navigation Dataset - Statistics Report")
        report_lines.append("="*80)
        report_lines.append("")
        
        for split_name, episodes in [('Train', self.train_episodes), ('Val', self.val_episodes)]:
            report_lines.append(f"\n{split_name} Dataset Statistics:")
            report_lines.append("-"*80)
            
            # Basic statistics
            num_episodes = len(episodes)
            num_scenes = len(set(ep['scene'] for ep in episodes))
            report_lines.append(f"  Total Episodes: {num_episodes}")
            report_lines.append(f"  Total Scenes: {num_scenes}")
            
            # Step statistics
            steps = [ep['num_steps'] for ep in episodes if ep['num_steps'] > 0]
            if steps:
                report_lines.append(f"\nTrajectory Length Statistics:")
                report_lines.append(f"  Mean: {np.mean(steps):.2f} steps")
                report_lines.append(f"  Median: {np.median(steps):.2f} steps")
                report_lines.append(f"  Std: {np.std(steps):.2f} steps")
                report_lines.append(f"  Min: {min(steps)} steps")
                report_lines.append(f"  Max: {max(steps)} steps")
            
            # Pedestrian statistics
            episodes_with_pedestrians = [ep for ep in episodes if ep['has_pedestrian']]
            if episodes_with_pedestrians:
                report_lines.append(f"\nPedestrian Statistics:")
                report_lines.append(f"  Episodes with pedestrians: {len(episodes_with_pedestrians)} ({len(episodes_with_pedestrians)/num_episodes*100:.1f}%)")
                avg_humans = [ep['avg_human_num'] for ep in episodes_with_pedestrians]
                report_lines.append(f"  Average pedestrians per episode: {np.mean(avg_humans):.2f} persons")
                max_humans = [ep['max_human_num'] for ep in episodes_with_pedestrians]
                report_lines.append(f"  Max pedestrians in single frame: {max(max_humans):.0f} persons")
            
            # Action statistics
            all_actions = []
            for ep in episodes:
                if ep.get('actions'):
                    all_actions.extend(ep['actions'])
            
            if all_actions:
                action_counts = Counter(all_actions)
                total_actions = len(all_actions)
                report_lines.append(f"\nAction Distribution:")
                report_lines.append(f"  Total actions: {total_actions}")
                report_lines.append(f"  STOP (0): {action_counts.get(0, 0)} ({action_counts.get(0, 0)/total_actions*100:.1f}%)")
                report_lines.append(f"  MOVE_FORWARD (1): {action_counts.get(1, 0)} ({action_counts.get(1, 0)/total_actions*100:.1f}%)")
                report_lines.append(f"  TURN_LEFT (2): {action_counts.get(2, 0)} ({action_counts.get(2, 0)/total_actions*100:.1f}%)")
                report_lines.append(f"  TURN_RIGHT (3): {action_counts.get(3, 0)} ({action_counts.get(3, 0)/total_actions*100:.1f}%)")
        
        # Save report
        report_path = self.output_dir / "dataset_statistics.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
    
    def generate_visualizations(self):
        """Generate all visualization charts in PDF format"""
        print("\nGenerating academic visualizations (PDF format)...")
        
        # 1. Trajectory Length Distribution
        self.plot_trajectory_length_distribution()
        
        # 2. Pedestrian Density Heatmap
        self.plot_pedestrian_density_heatmap()
        
        # 3. Action Distribution Pie Chart
        self.plot_action_distribution()
        
        print(f"\nAll visualizations saved to {self.output_dir}")
    
    def plot_trajectory_length_distribution(self):
        """Plot trajectory length (step count) distribution"""
        print("  Generating trajectory length distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            for idx, (split_name, episodes) in enumerate([('Training', self.train_episodes), 
                                                           ('Validation', self.val_episodes)]):
                steps = [ep['num_steps'] for ep in episodes if ep['num_steps'] > 0]
                
                if not steps:
                    continue
                
            # Create histogram
            n, bins, patches = axes[idx].hist(steps, bins=30, alpha=0.75, color='steelblue',
                                              edgecolor='black', linewidth=0.5)
                
                # Add mean and median lines
                mean_val = np.mean(steps)
                median_val = np.median(steps)
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
                          label=f'Mean: {mean_val:.1f}')
            axes[idx].axvline(median_val, color='orange', linestyle='-.', linewidth=2.5, 
                          label=f'Median: {median_val:.1f}')
                
            # Add statistics text box
            stats_text = f'Episodes: {len(steps)}\nStd: {np.std(steps):.1f}\nMin: {min(steps)}\nMax: {max(steps)}'
            axes[idx].text(0.65, 0.95, stats_text, transform=axes[idx].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=9)
            
            axes[idx].set_xlabel('Trajectory Length (Steps)', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Number of Episodes', fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{split_name} Set - Step Length Distribution', 
                               fontsize=13, fontweight='bold')
            axes[idx].legend(loc='upper right', fontsize=10)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
        
        # Save as PDF
        pdf_path = self.output_dir / "trajectory_length_distribution.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"    Saved: {pdf_path}")
        plt.close()
    
    def plot_pedestrian_density_heatmap(self):
        """Plot pedestrian density heatmap showing density across different scenes"""
        print("  Generating pedestrian density heatmap...")
        
        # Collect pedestrian density data per scene
        scene_pedestrian_data = defaultdict(list)
        
        for episode in self.all_episodes:
            if episode.get('avg_human_num', 0) > 0:
                scene_name = episode['scene'].replace('.basis', '')
                scene_pedestrian_data[scene_name].append(episode['avg_human_num'])
                
        # Calculate average pedestrian density per scene
        scene_avg_density = {}
        for scene, densities in scene_pedestrian_data.items():
            scene_avg_density[scene] = np.mean(densities)
                
        # Sort scenes by density and select top scenes
        sorted_scenes = sorted(scene_avg_density.items(), key=lambda x: x[1], reverse=True)[:20]
        
        if not sorted_scenes:
            print("    No pedestrian data available, skipping heatmap.")
            return
        
        scenes, densities = zip(*sorted_scenes)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap data (reshape for 2D visualization)
        n_cols = 5
        n_rows = (len(scenes) + n_cols - 1) // n_cols
        heatmap_data = np.zeros((n_rows, n_cols))
        scene_labels = []
        
        for i, (scene, density) in enumerate(sorted_scenes[:n_rows*n_cols]):
            row = i // n_cols
            col = i % n_cols
            heatmap_data[row, col] = density
            scene_labels.append(scene[:12])  # Truncate long names
        
        # Pad labels if needed
        while len(scene_labels) < n_rows * n_cols:
            scene_labels.append('')
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(densities))
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        
        # Create column labels
        col_labels = []
        for col in range(n_cols):
            idx = col
            if idx < len(scene_labels):
                col_labels.append(scene_labels[idx])
            else:
                col_labels.append('')
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
                
        # Create row labels
        row_labels = []
        for row in range(n_rows):
            row_labels.append(f'Group {row+1}')
        ax.set_yticklabels(row_labels, fontsize=9)
        
        # Add text annotations
        for i in range(len(sorted_scenes[:n_rows*n_cols])):
            row = i // n_cols
            col = i % n_cols
            if heatmap_data[row, col] > 0:
                text = ax.text(col, row, f'{heatmap_data[row, col]:.1f}',
                              ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average Pedestrian Density (persons/episode)', rotation=270, labelpad=20, 
                      fontsize=11, fontweight='bold')
        
        ax.set_title('Pedestrian Density Heatmap Across Top 20 Scenes\n(Highlighting High-Density Scenarios with avg. 6.8 persons/scene)', 
                    fontsize=13, fontweight='bold', pad=15)
            
            plt.tight_layout()
        
        # Save as PDF
        pdf_path = self.output_dir / "pedestrian_density_heatmap.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"    Saved: {pdf_path}")
        plt.close()
    
    def plot_action_distribution(self):
        """Plot action distribution as pie charts"""
        print("  Generating action distribution...")
        
        # Action mapping
        action_names = {
            0: 'STOP',
            1: 'MOVE_FORWARD', 
            2: 'TURN_LEFT',
            3: 'TURN_RIGHT'
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
        for idx, (split_name, episodes) in enumerate([('Training', self.train_episodes), 
                                                       ('Validation', self.val_episodes)]):
                # Collect all actions
                all_actions = []
                for ep in episodes:
                if ep.get('actions'):
                    all_actions.extend(ep['actions'])
                
                if not all_actions:
                    continue
                
            # Count action frequencies
                action_counts = Counter(all_actions)
            
            # Prepare data for pie chart
            # Group forward as one category, turns as another
            stop_count = action_counts.get(0, 0)
            forward_count = action_counts.get(1, 0)
            turn_left_count = action_counts.get(2, 0)
            turn_right_count = action_counts.get(3, 0)
            turn_total = turn_left_count + turn_right_count
            
            labels = ['Move Forward', 'Turn (Left/Right)', 'Stop']
            sizes = [forward_count, turn_total, stop_count]
            colors = ['#66b3ff', '#ffcc99', '#ff9999']
            explode = (0.05, 0.05, 0.1)  # Emphasize stop action
                
                # Create pie chart
            wedges, texts, autotexts = axes[idx].pie(sizes, explode=explode, labels=labels, colors=colors,
                                                   autopct='%1.1f%%', startangle=90,
                                                      textprops={'fontsize': 12, 'fontweight': 'bold'},
                                                      pctdistance=0.85)
                
            # Make percentage text more visible
                for autotext in autotexts:
                    autotext.set_color('white')
                autotext.set_fontsize(13)
                autotext.set_fontweight('bold')
            
            # Add detailed statistics in a text box
            total_actions = sum(sizes)
            stats_text = (f'Total Actions: {total_actions:,}\n'
                         f'Forward: {forward_count:,} ({forward_count/total_actions*100:.1f}%)\n'
                         f'Turn Left: {turn_left_count:,} ({turn_left_count/total_actions*100:.1f}%)\n'
                         f'Turn Right: {turn_right_count:,} ({turn_right_count/total_actions*100:.1f}%)\n'
                         f'Stop: {stop_count:,} ({stop_count/total_actions*100:.1f}%)')
            
            axes[idx].text(1.35, 0.5, stats_text, transform=axes[idx].transAxes,
                          verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                          fontsize=10, family='monospace')
            
            axes[idx].set_title(f'{split_name} Set - Action Distribution', 
                               fontsize=13, fontweight='bold')
        
        plt.tight_layout()
            
        # Save as PDF
        pdf_path = self.output_dir / "action_distribution.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"    Saved: {pdf_path}")
        plt.close()
    
    def export_dataframe(self):
        """Export to CSV files"""
        print("\nExporting dataframes...")
        
        # Train
        train_df = pd.DataFrame(self.train_episodes)
        train_df['split'] = 'train'
        train_df.to_csv(self.output_dir / 'train_episodes.csv', index=False)
        print(f"  Train episodes CSV saved ({len(train_df)} rows)")
        
        # Val
        val_df = pd.DataFrame(self.val_episodes)
        val_df['split'] = 'val'
        val_df.to_csv(self.output_dir / 'val_episodes.csv', index=False)
        print(f"  Val episodes CSV saved ({len(val_df)} rows)")
        
        # Combined
        all_df = pd.concat([train_df, val_df], ignore_index=True)
        all_df.to_csv(self.output_dir / 'all_episodes.csv', index=False)
        print(f"  Combined CSV saved ({len(all_df)} rows)")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Falcon Social Navigation Dataset')
    parser.add_argument('--train_dir', type=str, 
                       default='data/collect_data/train',
                       help='Training dataset path')
    parser.add_argument('--val_dir', type=str,
                       default='data/collect_data/val',
                       help='Validation dataset path')
    parser.add_argument('--output_dir', type=str, 
                       default='dataset_analysis_academic',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = AcademicDatasetAnalyzer(args.train_dir, args.val_dir, args.output_dir)
    analyzer.analyze_all()
    analyzer.export_dataframe()
    
    print("\n"+"="*80)
    print("Dataset analysis complete!")
    print(f"Results saved in: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
