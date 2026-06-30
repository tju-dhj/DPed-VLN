#!/usr/bin/env python3
"""
统计每个episode的步数并绘制频率直方图
专门针对 data/collect_data/train/*/*/action/0.json
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

def main():
    data_root = "data/collect_data/train"
    output_file = "episode_steps_histogram.png"
    
    print("="*80)
    print("统计Episode步数")
    print("="*80)
    print(f"数据目录: {data_root}")
    print()
    
    # 查找所有action文件
    pattern = os.path.join(data_root, "*", "*", "action", "0.json")
    action_files = glob.glob(pattern)
    
    print(f"找到 {len(action_files)} 个action文件")
    print()
    
    # 统计每个episode的步数
    step_counts = []
    failed_files = []
    
    for action_file in tqdm.tqdm(action_files, desc="读取action文件"):
        try:
            with open(action_file, 'r') as f:
                actions = json.load(f)
            step_counts.append(len(actions))
        except Exception as e:
            failed_files.append((action_file, str(e)))
    
    print(f"\n成功读取: {len(step_counts)} 个文件")
    if failed_files:
        print(f"失败: {len(failed_files)} 个文件")
        print("失败文件示例（前5个）:")
        for f, err in failed_files[:5]:
            print(f"  {f}: {err}")
    
    if not step_counts:
        print("\n错误: 没有成功读取任何数据!")
        return
    
    # 统计信息
    print("\n" + "="*80)
    print("步数统计")
    print("="*80)
    print(f"总episode数: {len(step_counts)}")
    print(f"最小步数: {min(step_counts)}")
    print(f"最大步数: {max(step_counts)}")
    print(f"平均步数: {np.mean(step_counts):.2f}")
    print(f"中位数步数: {np.median(step_counts):.2f}")
    print(f"标准差: {np.std(step_counts):.2f}")
    
    # 分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n分位数:")
    for p in percentiles:
        val = np.percentile(step_counts, p)
        print(f"  {p}%: {val:.0f}")
    
    # 步数范围分布
    print("\n步数范围分布:")
    ranges = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 100), (100, float('inf'))]
    for start, end in ranges:
        if end == float('inf'):
            count = sum(1 for x in step_counts if x >= start)
            print(f"  {start}+: {count} ({100.0*count/len(step_counts):.2f}%)")
        else:
            count = sum(1 for x in step_counts if start <= x < end)
            print(f"  [{start}, {end}): {count} ({100.0*count/len(step_counts):.2f}%)")
    
    # 绘制直方图
    print("\n绘制直方图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 完整分布
    ax1 = axes[0, 0]
    ax1.hist(step_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Episode步数', fontsize=12)
    ax1.set_ylabel('频率', fontsize=12)
    ax1.set_title(f'Episode步数分布 - 完整视图 (N={len(step_counts)})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    stats_text = f'均值: {np.mean(step_counts):.1f}\n中位数: {np.median(step_counts):.1f}\n标准差: {np.std(step_counts):.1f}'
    ax1.text(0.7, 0.95, stats_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 子图2: 对数y轴
    ax2 = axes[0, 1]
    ax2.hist(step_counts, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.set_xlabel('Episode步数', fontsize=12)
    ax2.set_ylabel('频率 (对数)', fontsize=12)
    ax2.set_title('Episode步数分布 - 对数y轴', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 箱线图
    ax3 = axes[1, 0]
    bp = ax3.boxplot(step_counts, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][0].set_alpha(0.7)
    ax3.set_ylabel('Episode步数', fontsize=12)
    ax3.set_title('Episode步数分布 - 箱线图', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 子图4: 累积分布
    ax4 = axes[1, 1]
    sorted_steps = np.sort(step_counts)
    cumulative = np.arange(1, len(sorted_steps) + 1) / len(sorted_steps) * 100
    ax4.plot(sorted_steps, cumulative, linewidth=2, color='purple')
    ax4.set_xlabel('Episode步数', fontsize=12)
    ax4.set_ylabel('累积百分比 (%)', fontsize=12)
    ax4.set_title('Episode步数 - 累积分布函数', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # 标注关键点
    for p in [25, 50, 75, 90]:
        val = np.percentile(step_counts, p)
        ax4.axhline(y=p, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax4.axvline(x=val, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax4.text(val, p + 2, f'{p}%: {val:.0f}', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 直方图已保存到: {os.path.abspath(output_file)}")
    plt.close()
    
    # 保存数据到CSV
    csv_file = "episode_steps_data.csv"
    with open(csv_file, 'w') as f:
        f.write("episode_index,num_steps\n")
        for i, count in enumerate(step_counts):
            f.write(f"{i},{count}\n")
    print(f"✓ 数据已保存到: {os.path.abspath(csv_file)}")
    
    print("\n" + "="*80)
    print("统计完成!")
    print("="*80)


if __name__ == "__main__":
    main()

