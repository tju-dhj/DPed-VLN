#!/usr/bin/env python3
"""
分析训练数据的诊断脚本
- 统计每个episode的步数
- 绘制步数分布直方图
- 检查动作分布
- 检查指令质量
- 检查数据异常
"""

import os
import sys
import json
import glob
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

def analyze_episode(episode_path):
    """分析单个episode"""
    results = {
        'path': episode_path,
        'num_steps': 0,
        'actions': [],
        'has_instruction': False,
        'instruction_text': '',
        'has_rgb': False,
        'has_depth': False,
        'num_rgb_files': 0,
        'num_depth_files': 0,
        'valid': False,
        'error': None
    }
    
    try:
        # 检查action文件
        action_file = os.path.join(episode_path, "action", "0.json")
        if os.path.exists(action_file):
            with open(action_file, 'r') as f:
                actions = json.load(f)
            results['num_steps'] = len(actions)
            results['actions'] = actions
        else:
            results['error'] = 'No action file'
            return results
        
        # 检查指令文件
        instruction_dirs = ['instruction_vl_level_2', 'instruction_level_2', 'inst_navcomposer_v2']
        for inst_dir in instruction_dirs:
            inst_file = os.path.join(episode_path, inst_dir, "0.txt")
            if os.path.exists(inst_file):
                try:
                    with open(inst_file, 'r', encoding='utf-8') as f:
                        instruction_text = f.read().strip()
                    if instruction_text:
                        results['has_instruction'] = True
                        results['instruction_text'] = instruction_text
                        break
                except Exception as e:
                    pass
        
        # 检查RGB文件
        rgb_dir = os.path.join(episode_path, "rgb")
        if os.path.exists(rgb_dir):
            rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
            results['has_rgb'] = len(rgb_files) > 0
            results['num_rgb_files'] = len(rgb_files)
        
        # 检查depth文件
        depth_dir = os.path.join(episode_path, "depth")
        if os.path.exists(depth_dir):
            depth_files = glob.glob(os.path.join(depth_dir, "*.png"))
            results['has_depth'] = len(depth_files) > 0
            results['num_depth_files'] = len(depth_files)
        
        # 判断是否有效
        results['valid'] = (
            results['num_steps'] > 0 and
            results['has_instruction'] and
            results['has_rgb']
        )
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def main():
    data_root = "data/collect_data/train"
    output_dir = "data_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("训练数据分析")
    print("="*80)
    print(f"数据根目录: {data_root}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 扫描所有episode
    print("扫描episode...")
    episode_paths = []
    scene_dirs = glob.glob(os.path.join(data_root, "*"))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d) and not d.endswith('outputs')]
    
    for scene_dir in tqdm.tqdm(scene_dirs, desc="扫描场景"):
        episode_dirs = glob.glob(os.path.join(scene_dir, "*"))
        episode_dirs = [d for d in episode_dirs if os.path.isdir(d)]
        episode_paths.extend(episode_dirs)
    
    print(f"找到 {len(episode_paths)} 个episode目录")
    print()
    
    # 分析每个episode
    print("分析episode...")
    results = []
    for ep_path in tqdm.tqdm(episode_paths, desc="分析"):
        result = analyze_episode(ep_path)
        results.append(result)
    
    # 统计
    valid_episodes = [r for r in results if r['valid']]
    invalid_episodes = [r for r in results if not r['valid']]
    
    print()
    print("="*80)
    print("统计结果")
    print("="*80)
    print(f"总episode数: {len(results)}")
    print(f"有效episode数: {len(valid_episodes)}")
    print(f"无效episode数: {len(invalid_episodes)}")
    print(f"有效率: {100.0 * len(valid_episodes) / len(results):.2f}%")
    print()
    
    # 步数统计
    step_counts = [r['num_steps'] for r in valid_episodes]
    if step_counts:
        print("步数统计:")
        print(f"  最小步数: {min(step_counts)}")
        print(f"  最大步数: {max(step_counts)}")
        print(f"  平均步数: {np.mean(step_counts):.2f}")
        print(f"  中位数步数: {np.median(step_counts):.2f}")
        print(f"  标准差: {np.std(step_counts):.2f}")
        print()
        
        # 绘制直方图
        plt.figure(figsize=(12, 6))
        plt.hist(step_counts, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Episode步数', fontsize=12)
        plt.ylabel('频率', fontsize=12)
        plt.title(f'训练数据Episode步数分布 (N={len(step_counts)})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'均值: {np.mean(step_counts):.1f}\n中位数: {np.median(step_counts):.1f}\n标准差: {np.std(step_counts):.1f}'
        plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        hist_path = os.path.join(output_dir, "episode_step_distribution.png")
        plt.savefig(hist_path, dpi=150)
        print(f"✓ 步数分布直方图已保存到: {hist_path}")
        plt.close()
    
    # 动作分布统计
    all_actions = []
    for r in valid_episodes:
        all_actions.extend(r['actions'])
    
    if all_actions:
        action_counter = Counter(all_actions)
        print("\n动作分布:")
        total_actions = len(all_actions)
        for action, count in sorted(action_counter.items()):
            percentage = 100.0 * count / total_actions
            print(f"  动作 {action}: {count:,} ({percentage:.2f}%)")
        print()
        
        # 绘制动作分布饼图
        plt.figure(figsize=(10, 6))
        actions_sorted = sorted(action_counter.keys())
        counts = [action_counter[a] for a in actions_sorted]
        labels = [f'动作 {a}' for a in actions_sorted]
        colors = plt.cm.Set3(range(len(actions_sorted)))
        
        plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'动作分布 (总计: {total_actions:,})', fontsize=14)
        plt.axis('equal')
        
        action_dist_path = os.path.join(output_dir, "action_distribution.png")
        plt.savefig(action_dist_path, dpi=150)
        print(f"✓ 动作分布饼图已保存到: {action_dist_path}")
        plt.close()
    
    # 指令质量检查
    instructions_with_content = [r for r in valid_episodes if r['has_instruction'] and len(r['instruction_text']) > 0]
    print(f"\n指令统计:")
    print(f"  有指令的episode: {len(instructions_with_content)} / {len(valid_episodes)}")
    
    if instructions_with_content:
        inst_lengths = [len(r['instruction_text']) for r in instructions_with_content]
        print(f"  指令长度 - 最小: {min(inst_lengths)}, 最大: {max(inst_lengths)}, 平均: {np.mean(inst_lengths):.1f}")
        
        # 检查默认指令
        default_instructions = [
            'navigate to the target location.',
            'navigate to target location',
            'go to target',
        ]
        default_count = sum(1 for r in instructions_with_content 
                          if r['instruction_text'].lower() in [d.lower() for d in default_instructions])
        print(f"  默认指令数量: {default_count} ({100.0 * default_count / len(instructions_with_content):.2f}%)")
        
        # 打印一些示例指令
        print("\n指令示例（前10个）:")
        for i, r in enumerate(instructions_with_content[:10]):
            print(f"  {i+1}. {r['instruction_text'][:100]}...")
    
    # RGB/Depth文件数量统计
    rgb_counts = [r['num_rgb_files'] for r in valid_episodes]
    depth_counts = [r['num_depth_files'] for r in valid_episodes]
    
    print("\nRGB文件统计:")
    if rgb_counts:
        print(f"  平均RGB文件数: {np.mean(rgb_counts):.2f}")
        print(f"  RGB文件数范围: [{min(rgb_counts)}, {max(rgb_counts)}]")
    
    print("\nDepth文件统计:")
    if depth_counts:
        print(f"  有depth的episode: {sum(1 for c in depth_counts if c > 0)} / {len(valid_episodes)}")
        if any(c > 0 for c in depth_counts):
            depth_counts_nonzero = [c for c in depth_counts if c > 0]
            print(f"  平均depth文件数: {np.mean(depth_counts_nonzero):.2f}")
            print(f"  Depth文件数范围: [{min(depth_counts_nonzero)}, {max(depth_counts_nonzero)}]")
    
    # 检查数据一致性
    print("\n数据一致性检查:")
    inconsistent_episodes = []
    for r in valid_episodes:
        if r['num_steps'] != r['num_rgb_files']:
            inconsistent_episodes.append(r)
    
    if inconsistent_episodes:
        print(f"  ⚠ 发现 {len(inconsistent_episodes)} 个episode的步数与RGB文件数不一致")
        print(f"  示例（前5个）:")
        for i, r in enumerate(inconsistent_episodes[:5]):
            print(f"    {i+1}. {r['path']}: steps={r['num_steps']}, rgb_files={r['num_rgb_files']}")
    else:
        print(f"  ✓ 所有有效episode的步数与RGB文件数一致")
    
    # 保存详细报告
    report_path = os.path.join(output_dir, "data_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("训练数据分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"数据根目录: {data_root}\n")
        f.write(f"总episode数: {len(results)}\n")
        f.write(f"有效episode数: {len(valid_episodes)}\n")
        f.write(f"无效episode数: {len(invalid_episodes)}\n\n")
        
        if step_counts:
            f.write("步数统计:\n")
            f.write(f"  最小: {min(step_counts)}\n")
            f.write(f"  最大: {max(step_counts)}\n")
            f.write(f"  平均: {np.mean(step_counts):.2f}\n")
            f.write(f"  中位数: {np.median(step_counts):.2f}\n")
            f.write(f"  标准差: {np.std(step_counts):.2f}\n\n")
        
        if all_actions:
            f.write("动作分布:\n")
            for action, count in sorted(action_counter.items()):
                percentage = 100.0 * count / total_actions
                f.write(f"  动作 {action}: {count:,} ({percentage:.2f}%)\n")
            f.write("\n")
        
        if invalid_episodes:
            f.write(f"无效episode列表 (前100个):\n")
            for i, r in enumerate(invalid_episodes[:100]):
                f.write(f"  {i+1}. {r['path']}: {r['error']}\n")
    
    print(f"\n✓ 详细报告已保存到: {report_path}")
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)


if __name__ == "__main__":
    main()

