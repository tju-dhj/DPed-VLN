#!/usr/bin/env python3
"""
Dataset Analysis and Visualization Script
For analyzing the Falcon Social Navigation Dataset metrics
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
from matplotlib.backends.backend_pdf import PdfPages

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


class DatasetAnalyzer:
    """Dataset Analyzer for Falcon Social Navigation Dataset"""
    
    def __init__(self, train_dir, val_dir, output_dir="dataset_analysis"):
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
        
        self.stats = {
            'train': defaultdict(list),
            'val': defaultdict(list)
        }
        
        # Store episode data for analysis
        self.train_episodes = []
        self.val_episodes = []
        self.all_episodes = []
        
    def collect_episode_info(self, dataset_path, split_name):
        """
        收集 episode 信息
        
        Args:
            dataset_path: 数据集路径
            split_name: 'train' or 'val'
        """
        print(f"\n分析 {split_name} 数据集: {dataset_path}")
        
        episode_info = []
        scene_dirs = sorted(dataset_path.glob("*.basis"))
        
        for scene_dir in tqdm(scene_dirs, desc=f"扫描 {split_name} 场景"):
            scene_name = scene_dir.name
            episode_dirs = [d for d in scene_dir.iterdir() if d.is_dir()]
            
            for episode_dir in episode_dirs:
                episode_id = episode_dir.name
                
                info = {
                    'scene': scene_name,
                    'episode_id': episode_id,
                    'episode_path': str(episode_dir)
                }
                
                # Collect step count and action data
                action_file = episode_dir / "action" / "0.json"
                if action_file.exists():
                    try:
                        with open(action_file, 'r') as f:
                            actions = json.load(f)
                            info['num_steps'] = len(actions)
                            info['actions'] = actions  # Store actions for distribution analysis
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
                
                # 注意：已在下面的 human_num 部分统一处理
                
                # 检查行人数量（使用 human_num 数据）
                human_num_file = episode_dir / "human_num" / "0.json"
                if human_num_file.exists():
                    try:
                        with open(human_num_file, 'r') as f:
                            human_nums_raw = json.load(f)
                            
                            # 处理嵌套列表格式：[[6], [6], ...] -> [6, 6, ...]
                            if isinstance(human_nums_raw, list) and len(human_nums_raw) > 0:
                                # 检查是否是嵌套列表
                                if isinstance(human_nums_raw[0], list):
                                    # 展平嵌套列表，取每个子列表的第一个元素
                                    human_nums = [item[0] if isinstance(item, list) and len(item) > 0 else 0 
                                                 for item in human_nums_raw]
                                else:
                                    human_nums = human_nums_raw
                                
                                # 统计行人信息
                                info['max_human_num'] = max(human_nums) if human_nums else 0
                                info['avg_human_num'] = float(np.mean(human_nums)) if human_nums else 0.0
                                
                                # 判断是否有行人（任何 step 中行人数 > 0）
                                info['has_pedestrian'] = any(n > 0 for n in human_nums)
                                # 有行人的 step 数量
                                info['pedestrian_steps'] = sum(1 for n in human_nums if n > 0)
                                # 有行人的 step 比例
                                info['pedestrian_ratio'] = info['pedestrian_steps'] / max(len(human_nums), 1)
                            else:
                                info['max_human_num'] = 0
                                info['avg_human_num'] = 0.0
                                info['has_pedestrian'] = False
                                info['pedestrian_steps'] = 0
                                info['pedestrian_ratio'] = 0.0
                    except Exception as e:
                        print(f"Warning: Error processing human_num for {episode_dir}: {e}")
                        info['max_human_num'] = 0
                        info['avg_human_num'] = 0.0
                        info['has_pedestrian'] = False
                        info['pedestrian_steps'] = 0
                        info['pedestrian_ratio'] = 0.0
                else:
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
                            info['poses'] = poses  # Store for spatial distribution
                    except:
                        info['poses'] = []
                else:
                    info['poses'] = []
                
                # Check instruction data
                info['has_instruction_l1'] = (episode_dir / "instruction_vl_level_1").exists()
                info['has_instruction_l2'] = (episode_dir / "instruction_vl_level_2").exists()
                
                # 读取指令示例
                if info['has_instruction_l1']:
                    inst_l1_file = episode_dir / "instruction_vl_level_1" / "0.txt"
                    if inst_l1_file.exists():
                        try:
                            with open(inst_l1_file, 'r', encoding='utf-8') as f:
                                inst = f.read().strip()
                                info['instruction_l1_length'] = len(inst)
                                info['instruction_l1_words'] = len(inst.split())
                        except:
                            info['instruction_l1_length'] = 0
                            info['instruction_l1_words'] = 0
                    else:
                        info['instruction_l1_length'] = 0
                        info['instruction_l1_words'] = 0
                else:
                    info['instruction_l1_length'] = 0
                    info['instruction_l1_words'] = 0
                
                if info['has_instruction_l2']:
                    inst_l2_file = episode_dir / "instruction_vl_level_2" / "0.txt"
                    if inst_l2_file.exists():
                        try:
                            with open(inst_l2_file, 'r', encoding='utf-8') as f:
                                inst = f.read().strip()
                                info['instruction_l2_length'] = len(inst)
                                info['instruction_l2_words'] = len(inst.split())
                        except:
                            info['instruction_l2_length'] = 0
                            info['instruction_l2_words'] = 0
                    else:
                        info['instruction_l2_length'] = 0
                        info['instruction_l2_words'] = 0
                else:
                    info['instruction_l2_length'] = 0
                    info['instruction_l2_words'] = 0
                
                episode_info.append(info)
        
        return episode_info
    
    def analyze_all(self):
        """分析所有数据"""
        print("="*60)
        print("开始数据集分析")
        print("="*60)
        
        # 收集 train 和 val 数据
        self.train_episodes = self.collect_episode_info(self.train_dir, 'train')
        self.val_episodes = self.collect_episode_info(self.val_dir, 'val')
        
        # 生成统计报告
        self.generate_statistics()
        
        # 生成可视化
        self.generate_visualizations()
        
        print(f"\n分析完成！结果保存在: {self.output_dir}")
    
    def generate_statistics(self):
        """生成统计报告"""
        print("\n生成统计报告...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("Falcon Social Navigation Dataset - 统计报告")
        report_lines.append("="*80)
        report_lines.append("")
        
        for split_name, episodes in [('Train', self.train_episodes), ('Val', self.val_episodes)]:
            report_lines.append(f"\n{split_name} 数据集统计:")
            report_lines.append("-"*80)
            
            # 基本统计
            num_episodes = len(episodes)
            num_scenes = len(set(ep['scene'] for ep in episodes))
            
            report_lines.append(f"总 Episodes 数: {num_episodes}")
            report_lines.append(f"总场景数: {num_scenes}")
            report_lines.append(f"平均每个场景的 Episodes: {num_episodes/num_scenes:.2f}")
            report_lines.append("")
            
            # Steps 统计
            steps = [ep['num_steps'] for ep in episodes if ep['num_steps'] > 0]
            if steps:
                report_lines.append(f"Steps 统计:")
                report_lines.append(f"  - 总 Steps: {sum(steps):,}")
                report_lines.append(f"  - 平均 Steps/Episode: {np.mean(steps):.2f}")
                report_lines.append(f"  - 中位数 Steps: {np.median(steps):.1f}")
                report_lines.append(f"  - 最小 Steps: {min(steps)}")
                report_lines.append(f"  - 最大 Steps: {max(steps)}")
                report_lines.append(f"  - 标准差: {np.std(steps):.2f}")
                report_lines.append("")
            
            # 行人统计
            with_pedestrian = sum(1 for ep in episodes if ep['has_pedestrian'])
            pedestrian_ratio = with_pedestrian / num_episodes * 100 if num_episodes > 0 else 0
            
            report_lines.append(f"动态行人统计:")
            report_lines.append(f"  - 有行人的 Episodes: {with_pedestrian} ({pedestrian_ratio:.1f}%)")
            report_lines.append(f"  - 无行人的 Episodes: {num_episodes - with_pedestrian} ({100-pedestrian_ratio:.1f}%)")
            
            pedestrian_steps = [ep['pedestrian_steps'] for ep in episodes if ep['has_pedestrian']]
            if pedestrian_steps:
                report_lines.append(f"  - 有行人的 Episodes 中，平均行人出现步数: {np.mean(pedestrian_steps):.2f}")
            
            max_humans = [ep['max_human_num'] for ep in episodes if isinstance(ep.get('max_human_num'), (int, float)) and ep['max_human_num'] > 0]
            if max_humans:
                report_lines.append(f"  - 最大同时出现行人数: {max(max_humans)}")
                report_lines.append(f"  - 平均最大行人数: {np.mean(max_humans):.2f}")
            report_lines.append("")
            
            # 指令统计
            with_inst_l1 = sum(1 for ep in episodes if ep['has_instruction_l1'])
            with_inst_l2 = sum(1 for ep in episodes if ep['has_instruction_l2'])
            
            report_lines.append(f"指令生成统计:")
            report_lines.append(f"  - Level 1 指令覆盖: {with_inst_l1} Episodes ({with_inst_l1/num_episodes*100:.1f}%)")
            report_lines.append(f"  - Level 2 指令覆盖: {with_inst_l2} Episodes ({with_inst_l2/num_episodes*100:.1f}%)")
            
            inst_l1_lengths = [ep['instruction_l1_length'] for ep in episodes if ep['instruction_l1_length'] > 0]
            if inst_l1_lengths:
                report_lines.append(f"  - Level 1 平均字符数: {np.mean(inst_l1_lengths):.1f}")
                inst_l1_words = [ep['instruction_l1_words'] for ep in episodes if ep['instruction_l1_words'] > 0]
                report_lines.append(f"  - Level 1 平均单词数: {np.mean(inst_l1_words):.1f}")
            
            inst_l2_lengths = [ep['instruction_l2_length'] for ep in episodes if ep['instruction_l2_length'] > 0]
            if inst_l2_lengths:
                report_lines.append(f"  - Level 2 平均字符数: {np.mean(inst_l2_lengths):.1f}")
                inst_l2_words = [ep['instruction_l2_words'] for ep in episodes if ep['instruction_l2_words'] > 0]
                report_lines.append(f"  - Level 2 平均单词数: {np.mean(inst_l2_words):.1f}")
        
        # 保存报告
        report_path = self.output_dir / "dataset_statistics.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 打印到控制台
        print('\n'.join(report_lines))
    
    def generate_visualizations(self):
        """Generate professional academic visualizations"""
        print("\nGenerating visualizations...")
        
        # Combine train and val for comprehensive analysis
        self.all_episodes = self.train_episodes + self.val_episodes
        
        # 1. Trajectory Length Distribution (Step Length Distribution)
        self.plot_trajectory_length_distribution()
        
        # 2. Pedestrian Density Heatmap (Spatial Distribution by Scene)
        self.plot_pedestrian_density_heatmap()
        
        # 3. Action Distribution Pie Chart
        self.plot_action_distribution()
        
        # 4. Additional comprehensive visualizations
        self.plot_pedestrian_spatial_heatmap()
        
        print(f"All visualizations saved to {self.output_dir}")
    
    def plot_trajectory_length_distribution(self):
        """Plot trajectory length distribution (step count distribution)"""
        print("  Generating trajectory length distribution...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (split_name, episodes) in enumerate([('Train', self.train_episodes), ('Val', self.val_episodes)]):
            steps = [ep['num_steps'] for ep in episodes if ep['num_steps'] > 0]
            
            if not steps:
                continue
            
            # Plot histogram with better styling
            n, bins, patches = axes[idx].hist(steps, bins=50, alpha=0.75, color='steelblue', 
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
            
            axes[idx].set_xlabel('Trajectory Length (Steps)', fontsize=13, fontweight='bold')
            axes[idx].set_ylabel('Number of Episodes', fontsize=13, fontweight='bold')
            axes[idx].set_title(f'{split_name} Set - Step Length Distribution', 
                               fontsize=14, fontweight='bold')
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
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap data (reshape for 2D visualization)
        n_cols = 4
        n_rows = (len(scenes) + n_cols - 1) // n_cols
        heatmap_data = np.zeros((n_rows, n_cols))
        scene_labels = []
        
        for i, (scene, density) in enumerate(sorted_scenes[:n_rows*n_cols]):
            row = i // n_cols
            col = i % n_cols
            heatmap_data[row, col] = density
            scene_labels.append(scene[:15])  # Truncate long names
        
        # Pad labels if needed
        while len(scene_labels) < n_rows * n_cols:
            scene_labels.append('')
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(densities))
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        
        # Create tick labels
        x_labels = [scene_labels[i] for i in range(0, min(len(scene_labels), n_cols))]
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        
        y_labels = []
        for row in range(n_rows):
            idx = row * n_cols
            if idx < len(scene_labels):
                y_labels.append(f'Row {row+1}')
        ax.set_yticklabels(y_labels, fontsize=9)
        
        # Add text annotations
        for i in range(len(sorted_scenes[:n_rows*n_cols])):
            row = i // n_cols
            col = i % n_cols
            text = ax.text(col, row, f'{heatmap_data[row, col]:.1f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Average Pedestrian Density (persons/episode)', rotation=270, labelpad=20, 
                      fontsize=11, fontweight='bold')
        
        ax.set_title('Pedestrian Density Heatmap Across Top 20 Scenes\n(Highlighting High-Density Scenarios)', 
                    fontsize=14, fontweight='bold', pad=15)
        
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
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for idx, (split_name, episodes) in enumerate([('Train', self.train_episodes), ('Val', self.val_episodes)]):
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
                                                      textprops={'fontsize': 11, 'fontweight': 'bold'},
                                                      pctdistance=0.85)
            
            # Make percentage text more visible
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_fontweight('bold')
            
            # Add detailed statistics in a text box
            total_actions = sum(sizes)
            stats_text = (f'Total Actions: {total_actions:,}\n'
                         f'Forward: {forward_count:,} ({forward_count/total_actions*100:.1f}%)\n'
                         f'Turn L: {turn_left_count:,} ({turn_left_count/total_actions*100:.1f}%)\n'
                         f'Turn R: {turn_right_count:,} ({turn_right_count/total_actions*100:.1f}%)\n'
                         f'Stop: {stop_count:,} ({stop_count/total_actions*100:.1f}%)')
            
            axes[idx].text(1.3, 0.5, stats_text, transform=axes[idx].transAxes,
                          verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                          fontsize=9, family='monospace')
            
            axes[idx].set_title(f'{split_name} Set - Action Distribution', 
                               fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save as PDF
        pdf_path = self.output_dir / "action_distribution.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"    Saved: {pdf_path}")
        plt.close()
    
    def plot_pedestrian_spatial_heatmap(self):
        """Plot spatial distribution of pedestrian encounters"""
        print("  Generating pedestrian spatial distribution heatmap...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, (split_name, episodes) in enumerate([('Train', self.train_episodes), ('Val', self.val_episodes)]):
            # Collect spatial data (x, z positions) from episodes with pedestrians
            positions_x = []
            positions_z = []
            pedestrian_counts = []
            
            for ep in episodes:
                if ep.get('has_pedestrian') and ep.get('poses'):
                    poses = ep['poses']
                    for pose in poses:
                        if isinstance(pose, list) and len(pose) >= 3:
                            # Extract x and z coordinates (y is vertical, ignore)
                            x, y, z = pose[0], pose[1], pose[2]
                            positions_x.append(x)
                            positions_z.append(z)
                            # Weight by average pedestrian count in this episode
                            pedestrian_counts.append(ep.get('avg_human_num', 0))
            
            if not positions_x:
                axes[idx].text(0.5, 0.5, 'No spatial data available', 
                              ha='center', va='center', fontsize=12)
                axes[idx].set_title(f'{split_name} Set - Spatial Distribution', 
                                   fontsize=14, fontweight='bold')
                continue
            
            # Create 2D histogram (heatmap)
            h, xedges, yedges, im = axes[idx].hist2d(positions_x, positions_z, 
                                                      bins=50, cmap='hot', 
                                                      weights=pedestrian_counts,
                                                      cmin=0.1)
            
            axes[idx].set_xlabel('X Position (meters)', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Z Position (meters)', fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{split_name} Set - Spatial Distribution of Pedestrian Encounters', 
                               fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Pedestrian Encounter Density', rotation=270, labelpad=20, 
                          fontsize=10, fontweight='bold')
            
            axes[idx].grid(True, alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        
        # Save as PDF
        pdf_path = self.output_dir / "pedestrian_spatial_distribution.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"    Saved: {pdf_path}")
        plt.close()
    
    def plot_pedestrian_ratio(self):
        """绘制行人出现比例"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for idx, (split_name, episodes) in enumerate([('Train', self.train_episodes), ('Val', self.val_episodes)]):
            with_ped = sum(1 for ep in episodes if ep['has_pedestrian'])
            without_ped = len(episodes) - with_ped
            
            # 饼图
            sizes = [with_ped, without_ped]
            labels = [f'With Pedestrians\n({with_ped}, {with_ped/len(episodes)*100:.1f}%)',
                     f'Without Pedestrians\n({without_ped}, {without_ped/len(episodes)*100:.1f}%)']
            colors = ['#ff9999', '#66b3ff']
            explode = (0.1, 0)
            
            axes[idx].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                         shadow=True, startangle=90, textprops={'fontsize': 10})
            axes[idx].set_title(f'{split_name} - Pedestrian Presence', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pedestrian_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 行人出现比例图已保存")
    
    def plot_scene_distribution(self):
        """绘制场景 Episodes 分布"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        for idx, (split_name, episodes) in enumerate([('Train', self.train_episodes), ('Val', self.val_episodes)]):
            scene_counts = Counter(ep['scene'] for ep in episodes)
            scenes = list(scene_counts.keys())
            counts = list(scene_counts.values())
            
            # 排序
            sorted_pairs = sorted(zip(scenes, counts), key=lambda x: x[1], reverse=True)
            scenes, counts = zip(*sorted_pairs[:30])  # 只显示前30个场景
            
            axes[idx].barh(range(len(scenes)), counts, alpha=0.7, edgecolor='black')
            axes[idx].set_yticks(range(len(scenes)))
            axes[idx].set_yticklabels([s.replace('.basis', '') for s in scenes], fontsize=8)
            axes[idx].set_xlabel('Number of Episodes', fontsize=12)
            axes[idx].set_title(f'{split_name} - Top 30 Scenes by Episode Count', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='x')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scene_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 场景分布图已保存")
    
    def plot_human_num_distribution(self):
        """绘制行人数量分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for idx, (split_name, episodes) in enumerate([('Train', self.train_episodes), ('Val', self.val_episodes)]):
            # 确保只使用数值类型
            max_humans = [ep['max_human_num'] for ep in episodes if isinstance(ep.get('max_human_num'), (int, float))]
            
            # 统计每个数量的频率
            human_counts = Counter(max_humans)
            nums = sorted(human_counts.keys())
            freqs = [human_counts[n] for n in nums]
            
            axes[idx].bar(nums, freqs, alpha=0.7, edgecolor='black', width=0.8)
            axes[idx].set_xlabel('Maximum Number of Pedestrians in Episode', fontsize=12)
            axes[idx].set_ylabel('Frequency', fontsize=12)
            axes[idx].set_title(f'{split_name} - Pedestrian Count Distribution', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_xticks(nums)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'human_num_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 行人数量分布图已保存")
    
    def plot_instruction_length(self):
        """绘制指令长度分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for idx, (split_name, episodes) in enumerate([('Train', self.train_episodes), ('Val', self.val_episodes)]):
            # Level 1 字符数
            l1_lengths = [ep['instruction_l1_length'] for ep in episodes if ep['instruction_l1_length'] > 0]
            if l1_lengths:
                axes[0, idx].hist(l1_lengths, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
                axes[0, idx].axvline(np.mean(l1_lengths), color='red', linestyle='--', linewidth=2, 
                                    label=f'Mean: {np.mean(l1_lengths):.1f}')
                axes[0, idx].set_xlabel('Instruction Length (characters)', fontsize=11)
                axes[0, idx].set_ylabel('Frequency', fontsize=11)
                axes[0, idx].set_title(f'{split_name} - Level 1 Instruction Length', fontsize=12, fontweight='bold')
                axes[0, idx].legend()
                axes[0, idx].grid(True, alpha=0.3)
            
            # Level 2 单词数
            l2_words = [ep['instruction_l2_words'] for ep in episodes if ep['instruction_l2_words'] > 0]
            if l2_words:
                axes[1, idx].hist(l2_words, bins=30, alpha=0.7, edgecolor='black', color='lightcoral')
                axes[1, idx].axvline(np.mean(l2_words), color='red', linestyle='--', linewidth=2, 
                                    label=f'Mean: {np.mean(l2_words):.1f}')
                axes[1, idx].set_xlabel('Instruction Length (words)', fontsize=11)
                axes[1, idx].set_ylabel('Frequency', fontsize=11)
                axes[1, idx].set_title(f'{split_name} - Level 2 Instruction Length', fontsize=12, fontweight='bold')
                axes[1, idx].legend()
                axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'instruction_length.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 指令长度分布图已保存")
    
    def plot_dashboard(self):
        """生成综合仪表板"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 汇总统计
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        summary_text = []
        summary_text.append("Falcon Social Navigation Dataset - Overview Dashboard")
        summary_text.append("="*100)
        
        for split_name, episodes in [('Train', self.train_episodes), ('Val', self.val_episodes)]:
            num_eps = len(episodes)
            num_scenes = len(set(ep['scene'] for ep in episodes))
            total_steps = sum(ep['num_steps'] for ep in episodes)
            with_ped = sum(1 for ep in episodes if ep['has_pedestrian'])
            
            summary_text.append(f"{split_name:8s} | Episodes: {num_eps:6,d} | Scenes: {num_scenes:3d} | "
                              f"Total Steps: {total_steps:8,d} | With Pedestrians: {with_ped:5,d} ({with_ped/num_eps*100:5.1f}%)")
        
        ax1.text(0.5, 0.5, '\n'.join(summary_text), fontsize=11, family='monospace',
                ha='center', va='center', transform=ax1.transAxes)
        
        # 其他子图
        # Steps 对比
        ax2 = fig.add_subplot(gs[1, 0])
        train_steps = [ep['num_steps'] for ep in self.train_episodes if ep['num_steps'] > 0]
        val_steps = [ep['num_steps'] for ep in self.val_episodes if ep['num_steps'] > 0]
        ax2.boxplot([train_steps, val_steps], labels=['Train', 'Val'])
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps Distribution Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 行人比例对比
        ax3 = fig.add_subplot(gs[1, 1])
        train_ped_ratio = sum(1 for ep in self.train_episodes if ep['has_pedestrian']) / len(self.train_episodes) * 100
        val_ped_ratio = sum(1 for ep in self.val_episodes if ep['has_pedestrian']) / len(self.val_episodes) * 100
        ax3.bar(['Train', 'Val'], [train_ped_ratio, val_ped_ratio], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Episodes with Pedestrians')
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 指令覆盖率
        ax4 = fig.add_subplot(gs[1, 2])
        train_inst_l1 = sum(1 for ep in self.train_episodes if ep['has_instruction_l1']) / len(self.train_episodes) * 100
        train_inst_l2 = sum(1 for ep in self.train_episodes if ep['has_instruction_l2']) / len(self.train_episodes) * 100
        val_inst_l1 = sum(1 for ep in self.val_episodes if ep['has_instruction_l1']) / len(self.val_episodes) * 100
        val_inst_l2 = sum(1 for ep in self.val_episodes if ep['has_instruction_l2']) / len(self.val_episodes) * 100
        
        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, [train_inst_l1, train_inst_l2], width, label='Train', alpha=0.7, edgecolor='black')
        ax4.bar(x + width/2, [val_inst_l1, val_inst_l2], width, label='Val', alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Coverage (%)')
        ax4.set_title('Instruction Generation Coverage')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Level 1', 'Level 2'])
        ax4.legend()
        ax4.set_ylim([0, 105])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Steps vs Pedestrians 散点图 (Train)
        ax5 = fig.add_subplot(gs[2, 0])
        train_x = [ep['num_steps'] for ep in self.train_episodes if ep['num_steps'] > 0]
        train_y = [ep['pedestrian_steps'] for ep in self.train_episodes if ep['num_steps'] > 0]
        ax5.scatter(train_x, train_y, alpha=0.3, s=10)
        ax5.set_xlabel('Total Steps')
        ax5.set_ylabel('Steps with Pedestrians')
        ax5.set_title('Train: Steps vs Pedestrian Presence')
        ax5.grid(True, alpha=0.3)
        
        # Steps vs Pedestrians 散点图 (Val)
        ax6 = fig.add_subplot(gs[2, 1])
        val_x = [ep['num_steps'] for ep in self.val_episodes if ep['num_steps'] > 0]
        val_y = [ep['pedestrian_steps'] for ep in self.val_episodes if ep['num_steps'] > 0]
        ax6.scatter(val_x, val_y, alpha=0.3, s=10, color='orange')
        ax6.set_xlabel('Total Steps')
        ax6.set_ylabel('Steps with Pedestrians')
        ax6.set_title('Val: Steps vs Pedestrian Presence')
        ax6.grid(True, alpha=0.3)
        
        # 数据完整性
        ax7 = fig.add_subplot(gs[2, 2])
        completeness_data = []
        for split_name, episodes in [('Train', self.train_episodes), ('Val', self.val_episodes)]:
            has_action = sum(1 for ep in episodes if ep['num_steps'] > 0) / len(episodes) * 100
            has_ped_info = sum(1 for ep in episodes if 'has_pedestrian' in ep) / len(episodes) * 100
            has_inst = sum(1 for ep in episodes if ep['has_instruction_l1'] or ep['has_instruction_l2']) / len(episodes) * 100
            completeness_data.append([has_action, has_ped_info, has_inst])
        
        x = np.arange(3)
        width = 0.35
        ax7.bar(x - width/2, completeness_data[0], width, label='Train', alpha=0.7, edgecolor='black')
        ax7.bar(x + width/2, completeness_data[1], width, label='Val', alpha=0.7, edgecolor='black')
        ax7.set_ylabel('Completeness (%)')
        ax7.set_title('Data Completeness')
        ax7.set_xticks(x)
        ax7.set_xticklabels(['Action', 'Pedestrian\nInfo', 'Instruction'], fontsize=9)
        ax7.legend()
        ax7.set_ylim([0, 105])
        ax7.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(self.output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 综合仪表板已保存")
    
    def export_dataframe(self):
        """导出为 CSV 文件"""
        print("\n导出数据表...")
        
        # Train
        train_df = pd.DataFrame(self.train_episodes)
        train_df['split'] = 'train'
        train_df.to_csv(self.output_dir / 'train_episodes.csv', index=False)
        print(f"  ✓ Train episodes CSV 已保存 ({len(train_df)} rows)")
        
        # Val
        val_df = pd.DataFrame(self.val_episodes)
        val_df['split'] = 'val'
        val_df.to_csv(self.output_dir / 'val_episodes.csv', index=False)
        print(f"  ✓ Val episodes CSV 已保存 ({len(val_df)} rows)")
        
        # 合并
        all_df = pd.concat([train_df, val_df], ignore_index=True)
        all_df.to_csv(self.output_dir / 'all_episodes.csv', index=False)
        print(f"  ✓ 合并 CSV 已保存 ({len(all_df)} rows)")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 Falcon 社交导航数据集')
    parser.add_argument('--train_dir', type=str, 
                       default='data/collect_data/train',
                       help='训练集路径')
    parser.add_argument('--val_dir', type=str,
                       default='data/collect_data/val',
                       help='验证集路径')
    parser.add_argument('--output_dir', type=str,
                       default='dataset_analysis',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = DatasetAnalyzer(args.train_dir, args.val_dir, args.output_dir)
    analyzer.analyze_all()
    analyzer.export_dataframe()
    
    print("\n"+"="*60)
    print("数据集分析完成！")
    print(f"结果保存在: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

