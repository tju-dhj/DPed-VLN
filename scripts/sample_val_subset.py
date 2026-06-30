#!/usr/bin/env python3
"""
从验证集 val_full 中分层抽样200个episodes，生成 val_evalfast 数据集。
抽样策略：按照指令长度分布保持一致。
"""

import gzip
import json
import glob
import os
import random
import numpy as np
from collections import defaultdict

# 设置随机种子以保证可重复性
random.seed(42)
np.random.seed(42)

# 配置
INPUT_DIR = "data/dynamic_dataset_final_v1/val_full"
OUTPUT_DIR = "data/DPed_pro/val_evalfast"
TARGET_SIZE = 200

def load_all_episodes(input_dir):
    """加载所有episodes"""
    all_episodes = []
    files = sorted(glob.glob(os.path.join(input_dir, "*.json.gz")))
    print(f"读取 {len(files)} 个文件...")
    
    for fpath in files:
        with gzip.open(fpath, 'rt') as f:
            data = json.load(f)
        all_episodes.extend(data['episodes'])
    
    return all_episodes

def analyze_distribution(episodes):
    """分析指令长度分布"""
    lengths = [len(ep['instruction']) for ep in episodes]
    
    # 按50字符分组（0-49, 50-99, 100-149, ...）
    bins = defaultdict(list)
    for i, ep in enumerate(episodes):
        length = len(ep['instruction'])
        bin_key = (length // 50) * 50
        bins[bin_key].append(i)
    
    print("\n原始数据集指令长度分布:")
    total = len(episodes)
    for bin_start in sorted(bins.keys()):
        count = len(bins[bin_start])
        print(f"  {bin_start:4d}-{bin_start+49:4d}字符: {count:4d} ({count/total*100:5.1f}%)")
    
    return bins, lengths

def stratified_sample(episodes, bins, target_size):
    """分层抽样，保持指令长度分布一致"""
    # 计算每个bin应该抽样的数量
    total = len(episodes)
    bin_counts = {bin_start: len(indices) for bin_start, indices in bins.items()}
    
    sampled_episodes = []
    sampled_bins = defaultdict(int)
    
    # 按比例分配抽样数量
    bin_allocations = {}
    for bin_start in sorted(bin_counts.keys()):
        orig_count = bin_counts[bin_start]
        # 计算该bin应抽样的数量（按比例）
        target_count = max(1, round(orig_count / total * target_size))
        # 实际能抽样的数量不能超过原始数量
        target_count = min(target_count, orig_count)
        bin_allocations[bin_start] = target_count
    
    # 如果总数超过目标，从最大的bin减少（从末尾减少到正好200）
    total_allocated = sum(bin_allocations.values())
    excess = total_allocated - target_size
    
    if excess > 0:
        # 从末尾的bin开始减少（跳过只有一个样本的bin）
        for bin_start in sorted(bin_allocations.keys(), reverse=True):
            if excess <= 0:
                break
            if bin_allocations[bin_start] > 1 and bin_start != 0:
                reduce = min(excess, bin_allocations[bin_start] - 1)
                bin_allocations[bin_start] -= reduce
                excess -= reduce
    
    for bin_start in sorted(bin_allocations.keys()):
        target_count = bin_allocations[bin_start]
        
        if target_count > 0:
            indices = bins[bin_start]
            sampled_indices = random.sample(indices, target_count)
            for idx in sampled_indices:
                sampled_episodes.append(episodes[idx])
                sampled_bins[bin_start] += 1
    
    return sampled_episodes, sampled_bins

def save_dataset(episodes, output_dir):
    """保存抽样后的数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 按场景分组保存（每个json.gz一个场景）
    scenes_data = defaultdict(list)
    for ep in episodes:
        # 从scene_id提取场景标识
        scene_id = ep['scene_id']
        # 提取场景名称（如 00898-8CRYizAb6yd）
        parts = scene_id.split('/')
        if len(parts) >= 3:
            scene_name = parts[2].replace('.basis.glb', '')
        else:
            scene_name = "unknown"
        
        scenes_data[scene_name].append(ep)
    
    # 保存每个场景一个文件
    for scene_name, eps in sorted(scenes_data.items()):
        output_file = os.path.join(output_dir, f"{scene_name}.json.gz")
        with gzip.open(output_file, 'wt') as f:
            json.dump({'episodes': eps}, f)
    
    print(f"\n保存到 {output_dir}:")
    print(f"  场景数: {len(scenes_data)}")
    print(f"  总episodes: {len(episodes)}")

def main():
    print("="*60)
    print("验证集分层抽样工具")
    print("="*60)
    
    # 1. 加载所有episodes
    episodes = load_all_episodes(INPUT_DIR)
    print(f"总episodes: {len(episodes)}")
    
    # 2. 分析分布
    bins, lengths = analyze_distribution(episodes)
    
    # 3. 分层抽样
    print(f"\n目标抽样数量: {TARGET_SIZE}")
    sampled_episodes, sampled_bins = stratified_sample(episodes, bins, TARGET_SIZE)
    
    # 4. 显示抽样结果分布
    print("\n抽样后指令长度分布:")
    total_sampled = len(sampled_episodes)
    for bin_start in sorted(sampled_bins.keys()):
        orig_count = len(bins[bin_start])
        sample_count = sampled_bins[bin_start]
        if orig_count > 0:
            ratio = sample_count / orig_count * 100
        else:
            ratio = 0
        print(f"  {bin_start:4d}-{bin_start+49:4d}字符: {sample_count:4d} ({ratio:5.1f}% of original)")
    
    print(f"\n实际抽样数量: {total_sampled}")
    
    # 5. 保存
    save_dataset(sampled_episodes, OUTPUT_DIR)
    
    # 6. 验证
    print("\n" + "="*60)
    print("验证输出数据集")
    print("="*60)
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "*.json.gz"))
    print(f"输出文件数: {len(output_files)}")
    
    verify_episodes = []
    for fpath in output_files:
        with gzip.open(fpath, 'rt') as f:
            data = json.load(f)
        verify_episodes.extend(data['episodes'])
    
    print(f"输出总episodes: {len(verify_episodes)}")
    
    verify_lengths = [len(ep['instruction']) for ep in verify_episodes]
    print(f"\n输出数据集指令长度统计:")
    print(f"  最小: {min(verify_lengths)}")
    print(f"  最大: {max(verify_lengths)}")
    print(f"  平均: {np.mean(verify_lengths):.1f}")
    print(f"  中位数: {np.median(verify_lengths):.1f}")
    
    print("\n完成!")

if __name__ == "__main__":
    main()
