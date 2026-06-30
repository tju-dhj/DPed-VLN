#!/usr/bin/env python3
"""
调整DPed_pro数据集的分布比例
使Val和Test数据集在动作序列长度和指令长度上分布一致

策略：
1. 计算两个数据集在各维度上的分布
2. 使用分层采样使Test集分布与Val集匹配
3. 输出调整后的数据集
"""

import json
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from collections import defaultdict
import random

random.seed(42)
np.random.seed(42)

def load_episodes_from_dir(directory):
    """从目录中加载所有episode数据"""
    episodes = []
    json_gz_files = glob.glob(os.path.join(directory, "*.json.gz"))

    for filepath in json_gz_files:
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                if 'episodes' in data:
                    episodes.extend(data['episodes'])
                elif isinstance(data, list):
                    episodes.extend(data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return episodes

def save_episodes_to_dir(episodes, directory, base_filenames):
    """将episodes保存回原始文件结构"""
    os.makedirs(directory, exist_ok=True)
    
    # 按scene_id分组
    scenes = defaultdict(list)
    for ep in episodes:
        scene_id = ep.get('scene_id', 'unknown')
        # 从scene_id提取场景标识
        if 'hm3d/' in scene_id:
            scene_name = scene_id.split('/')[-2]
            scenes[scene_name].append(ep)
    
    # 保存到对应文件
    for scene_name, scene_eps in scenes.items():
        filepath = os.path.join(directory, f"{scene_name}.json.gz")
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump({'episodes': scene_eps}, f)

def extract_metrics(episodes):
    """从episodes中提取各种指标"""
    metrics = {
        'instruction_length': [],
        'gt_action_length': [],
    }
    
    for ep in episodes:
        # gt_action长度
        if 'gt_action' in ep and ep['gt_action']:
            metrics['gt_action_length'].append(len(ep['gt_action']))
        else:
            metrics['gt_action_length'].append(0)
            
        # 指令词长度
        if 'instruction' in ep and ep['instruction']:
            metrics['instruction_length'].append(len(ep['instruction'].split()))
        else:
            metrics['instruction_length'].append(0)
    
    return metrics

def create_bins(data, n_bins=10):
    """创建分箱边界"""
    min_val, max_val = np.min(data), np.max(data)
    return np.linspace(min_val, max_val, n_bins + 1)

def assign_to_bin(value, bins):
    """将值分配到对应的bin"""
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i+1]:
            return i
    return len(bins) - 2  # 最后一个bin包含最大值

def stratified_resample_to_match(source_data, target_data, n_bins=10):
    """
    对source数据进行分层重采样，使其分布匹配target
    
    source_data: 要重采样的数据
    target_data: 目标分布
    """
    # 创建target的分箱
    bins = create_bins(target_data, n_bins)
    
    # 统计target每个bin的数量
    target_counts = defaultdict(int)
    for val in target_data:
        bin_idx = assign_to_bin(val, bins)
        target_counts[bin_idx] += 1
    
    # 按bin分组source数据
    source_by_bin = defaultdict(list)
    for i, val in enumerate(source_data):
        bin_idx = assign_to_bin(val, bins)
        source_by_bin[bin_idx].append(i)
    
    # 计算需要采样的索引
    selected_indices = []
    total_source = len(source_data)
    total_target = len(target_data)
    
    for bin_idx in range(len(bins) - 1):
        target_count = target_counts[bin_idx]
        source_in_bin = source_by_bin[bin_idx]
        
        if len(source_in_bin) == 0:
            # 如果该bin没有源数据，从相邻bin借用
            continue
            
        # 按比例采样
        if total_source > 0:
            ratio = target_count / total_target
            sampled_count = max(1, int(len(source_in_bin) * ratio * (total_target / total_source)))
            sampled_count = min(sampled_count, len(source_in_bin), target_count + 5)
        else:
            sampled_count = target_count
        
        # 随机采样
        sampled = random.choices(source_in_bin, k=sampled_count)
        selected_indices.extend(sampled)
    
    return selected_indices

def resample_test_to_match_val(test_episodes, val_episodes, n_bins_action=15, n_bins_instr=10):
    """
    重采样Test集使其分布匹配Val集
    """
    # 提取指标
    test_action = np.array([len(ep.get('gt_action', [])) for ep in test_episodes])
    test_instr = np.array([len(ep.get('instruction', '').split()) for ep in test_episodes])
    val_action = np.array([len(ep.get('gt_action', [])) for ep in val_episodes])
    val_instr = np.array([len(ep.get('instruction', '').split()) for ep in val_episodes])
    
    # 方法：使用联合分布采样
    # 创建联合分布的分箱
    action_bins = create_bins(np.concatenate([test_action, val_action]), n_bins_action)
    instr_bins = create_bins(np.concatenate([test_instr, val_instr]), n_bins_instr)
    
    # 统计Val集的联合分布
    val_joint_dist = defaultdict(int)
    for i in range(len(val_episodes)):
        action_bin = assign_to_bin(test_action[i] if i < len(test_episodes) else val_action[0], action_bins)
        instr_bin = assign_to_bin(test_instr[i] if i < len(test_episodes) else val_instr[0], instr_bins)
        key = (action_bin, instr_bin)
        val_joint_dist[key] += 1
    
    # 重新统计val的联合分布
    val_joint_dist = defaultdict(int)
    for i in range(len(val_episodes)):
        action_bin = assign_to_bin(val_action[i], action_bins)
        instr_bin = assign_to_bin(val_instr[i], instr_bins)
        key = (action_bin, instr_bin)
        val_joint_dist[key] += 1
    
    # 按bin分组test数据
    test_by_bin = defaultdict(list)
    for i, ep in enumerate(test_episodes):
        action_bin = assign_to_bin(test_action[i], action_bins)
        instr_bin = assign_to_bin(test_instr[i], instr_bins)
        key = (action_bin, instr_bin)
        test_by_bin[key].append(i)
    
    # 按比例重采样
    selected_indices = []
    val_total = len(val_episodes)
    
    for key, val_count in val_joint_dist.items():
        test_in_key = test_by_bin.get(key, [])
        if len(test_in_key) == 0:
            # 尝试找最近的bin
            for d_action in [-1, 1]:
                for d_instr in [-1, 1]:
                    alt_key = (key[0] + d_action, key[1] + d_instr)
                    alt_data = test_by_bin.get(alt_key, [])
                    if alt_data:
                        test_in_key = alt_data
                        break
                if test_in_key:
                    break
        
        if test_in_key:
            # 按比例采样
            ratio = val_count / val_total
            n_samples = max(1, int(len(test_in_key) * ratio * 1.5))
            n_samples = min(n_samples, len(test_in_key), val_count + 2)
            sampled = random.sample(test_in_key, min(n_samples, len(test_in_key)))
            selected_indices.extend(sampled)
    
    return selected_indices

def iterative_balance_distribution(test_episodes, val_episodes, n_bins=12, max_iterations=5):
    """
    迭代平衡分布，使Test集的分布逐步逼近Val集
    """
    print("\n🔄 Starting iterative balancing...")
    
    # 提取指标
    val_action = np.array([len(ep.get('gt_action', [])) for ep in val_episodes])
    val_instr = np.array([len(ep.get('instruction', '').split()) for ep in val_episodes])
    test_action = np.array([len(ep.get('gt_action', [])) for ep in test_episodes])
    test_instr = np.array([len(ep.get('instruction', '').split()) for ep in test_episodes])
    
    current_indices = list(range(len(test_episodes)))
    
    for iteration in range(max_iterations):
        print(f"\n   Iteration {iteration + 1}/{max_iterations}")
        
        # 计算当前分布与目标分布的差异
        current_action = test_action[current_indices]
        current_instr = test_instr[current_indices]
        
        # 使用KS检验评估分布差异
        from scipy import stats
        
        ks_action = stats.ks_2samp(current_action, val_action)
        ks_instr = stats.ks_2samp(current_instr, val_instr)
        
        print(f"      KS test (action): statistic={ks_action.statistic:.4f}, p-value={ks_action.pvalue:.4f}")
        print(f"      KS test (instruction): statistic={ks_instr.statistic:.4f}, p-value={ks_instr.pvalue:.4f}")
        
        if ks_action.pvalue > 0.3 and ks_instr.pvalue > 0.3:
            print(f"      ✅ Distribution is sufficiently close (p > 0.3)")
            break
        
        # 分层重采样
        bins_action = create_bins(val_action, n_bins)
        bins_instr = create_bins(val_instr, n_bins)
        
        # 统计val的分布
        val_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(val_episodes)):
            a_bin = assign_to_bin(val_action[i], bins_action)
            i_bin = assign_to_bin(val_instr[i], bins_instr)
            val_counts[(a_bin, i_bin)] += 1
        
        # 分组当前test数据
        current_action_now = test_action[current_indices]
        current_instr_now = test_instr[current_indices]
        
        test_groups = defaultdict(list)
        for idx, global_idx in enumerate(current_indices):
            a_bin = assign_to_bin(test_action[global_idx], bins_action)
            i_bin = assign_to_bin(test_instr[global_idx], bins_instr)
            test_groups[(a_bin, i_bin)].append(global_idx)
        
        # 按目标分布比例重采样
        new_indices = []
        total_val = len(val_episodes)
        
        for (a_bin, i_bin), target_count in val_counts.items():
            candidates = test_groups.get((a_bin, i_bin), [])
            
            # 如果没有精确匹配，找邻近bin
            if not candidates:
                for da in [-1, 0, 1]:
                    for di in [-1, 0, 1]:
                        if da == 0 and di == 0:
                            continue
                        alt_key = (a_bin + da, i_bin + di)
                        candidates = test_groups.get(alt_key, [])
                        if candidates:
                            break
                    if candidates:
                        break
            
            if candidates:
                # 按比例采样
                ratio = target_count / total_val
                n_target = len(current_indices) * ratio
                n_samples = int(n_target * 1.0)  # 稍微过采样以确保足够
                n_samples = min(n_samples, len(candidates), max(1, target_count))
                sampled = random.sample(candidates, n_samples)
                new_indices.extend(sampled)
        
        # 如果采样数量不够，用随机采样补充
        if len(new_indices) < len(val_episodes):
            remaining_needed = len(val_episodes) - len(new_indices)
            all_indices = set(range(len(test_episodes)))
            used_indices = set(new_indices)
            available = list(all_indices - used_indices)
            if available:
                additional = random.sample(available, min(remaining_needed, len(available)))
                new_indices.extend(additional)
        
        # 限制数量与val集相近
        if len(new_indices) > len(val_episodes) * 1.2:
            new_indices = random.sample(new_indices, int(len(val_episodes) * 1.2))
        
        current_indices = new_indices
    
    return current_indices

def compute_distribution_stats(data):
    """计算分布统计信息"""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q50': np.percentile(data, 50),
        'q75': np.percentile(data, 75),
    }

def plot_before_after(val_episodes, test_episodes_original, test_episodes_balanced, output_path):
    """绘制调整前后的分布对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 提取指标
    val_action = [len(ep.get('gt_action', [])) for ep in val_episodes]
    val_instr = [len(ep.get('instruction', '').split()) for ep in val_episodes]
    
    test_orig_action = [len(ep.get('gt_action', [])) for ep in test_episodes_original]
    test_orig_instr = [len(ep.get('instruction', '').split()) for ep in test_episodes_original]
    
    test_bal_action = [len(ep.get('gt_action', [])) for ep in test_episodes_balanced]
    test_bal_instr = [len(ep.get('instruction', '').split()) for ep in test_episodes_balanced]
    
    colors = {'val': '#2E86AB', 'test_orig': '#E94F37', 'test_bal': '#28A745'}
    
    # === 第一行：调整前 ===
    
    # 动作序列长度 - 调整前
    ax = axes[0, 0]
    bins = np.linspace(min(val_action + test_orig_action), 
                       max(val_action + test_orig_action), 20)
    ax.hist(val_action, bins=bins, alpha=0.6, label='Val', color=colors['val'], edgecolor='white')
    ax.hist(test_orig_action, bins=bins, alpha=0.6, label='Test (Original)', color=colors['test_orig'], edgecolor='white')
    ax.set_xlabel('Action Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Before: Action Sequence Length', fontweight='bold')
    ax.legend()
    
    # 指令长度 - 调整前
    ax = axes[0, 1]
    bins = np.linspace(min(val_instr + test_orig_instr), 
                       max(val_instr + test_orig_instr), 15)
    ax.hist(val_instr, bins=bins, alpha=0.6, label='Val', color=colors['val'], edgecolor='white')
    ax.hist(test_orig_instr, bins=bins, alpha=0.6, label='Test (Original)', color=colors['test_orig'], edgecolor='white')
    ax.set_xlabel('Instruction Length (words)')
    ax.set_ylabel('Frequency')
    ax.set_title('Before: Instruction Length', fontweight='bold')
    ax.legend()
    
    # 样本数量对比 - 调整前
    ax = axes[0, 2]
    counts = [len(val_episodes), len(test_episodes_original)]
    bars = ax.bar(['Val', 'Test (Original)'], counts, color=[colors['val'], colors['test_orig']], edgecolor='white')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Count')
    ax.set_title('Before: Dataset Size', fontweight='bold')
    
    # === 第二行：调整后 ===
    
    # 动作序列长度 - 调整后
    ax = axes[1, 0]
    bins = np.linspace(min(val_action + test_bal_action), 
                       max(val_action + test_bal_action), 20)
    ax.hist(val_action, bins=bins, alpha=0.6, label='Val', color=colors['val'], edgecolor='white')
    ax.hist(test_bal_action, bins=bins, alpha=0.6, label='Test (Balanced)', color=colors['test_bal'], edgecolor='white')
    ax.set_xlabel('Action Sequence Length')
    ax.set_ylabel('Frequency')
    ax.set_title('After: Action Sequence Length', fontweight='bold')
    ax.legend()
    
    # 指令长度 - 调整后
    ax = axes[1, 1]
    bins = np.linspace(min(val_instr + test_bal_instr), 
                       max(val_instr + test_bal_instr), 15)
    ax.hist(val_instr, bins=bins, alpha=0.6, label='Val', color=colors['val'], edgecolor='white')
    ax.hist(test_bal_instr, bins=bins, alpha=0.6, label='Test (Balanced)', color=colors['test_bal'], edgecolor='white')
    ax.set_xlabel('Instruction Length (words)')
    ax.set_ylabel('Frequency')
    ax.set_title('After: Instruction Length', fontweight='bold')
    ax.legend()
    
    # 样本数量对比 - 调整后
    ax = axes[1, 2]
    counts = [len(val_episodes), len(test_episodes_balanced)]
    bars = ax.bar(['Val', 'Test (Balanced)'], counts, color=[colors['val'], colors['test_bal']], edgecolor='white')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Episode Count')
    ax.set_title('After: Dataset Size', fontweight='bold')
    
    # KS检验结果
    from scipy import stats
    
    # 调整前
    ks_action_before = stats.ks_2samp(val_action, test_orig_action)
    ks_instr_before = stats.ks_2samp(val_instr, test_orig_instr)
    
    # 调整后
    ks_action_after = stats.ks_2samp(val_action, test_bal_action)
    ks_instr_after = stats.ks_2samp(val_instr, test_bal_instr)
    
    # 添加统计信息文本
    fig.text(0.5, 0.02, 
             f"KS Test Results:\n"
             f"Before: Action p={ks_action_before.pvalue:.3f}, Instruction p={ks_instr_before.pvalue:.3f}\n"
             f"After:  Action p={ks_action_after.pvalue:.3f}, Instruction p={ks_instr_after.pvalue:.3f}",
             ha='center', va='bottom', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('DPed_pro Dataset Distribution Balancing Results', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Comparison plot saved to: {output_path}")
    
    return {
        'before': {
            'ks_action': ks_action_before,
            'ks_instr': ks_instr_before
        },
        'after': {
            'ks_action': ks_action_after,
            'ks_instr': ks_instr_after
        }
    }

def main():
    # 数据集路径
    val_dir = "/share/home/u19666033/dhj/DPed_pro/data/DPed_pro/val"
    test_dir = "/share/home/u19666033/dhj/DPed_pro/data/DPed_pro/test"
    output_dir = "/share/home/u19666033/dhj/DPed_pro/data/DPed_pro/val_balanced"
    output_plot = "/share/home/u19666033/dhj/DPed_pro/distribution_balancing_comparison.png"
    
    print("=" * 70)
    print("  DPed_pro Dataset Distribution Balancing")
    print("=" * 70)
    
    # 加载数据
    print("\n📂 Loading datasets...")
    val_episodes = load_episodes_from_dir(val_dir)
    test_episodes = load_episodes_from_dir(test_dir)
    
    print(f"   Val:   {len(val_episodes)} episodes")
    print(f"   Test:  {len(test_episodes)} episodes")
    
    # 提取原始指标
    print("\n📊 Original Distribution Stats:")
    
    val_action = np.array([len(ep.get('gt_action', [])) for ep in val_episodes])
    val_instr = np.array([len(ep.get('instruction', '').split()) for ep in val_episodes])
    test_action = np.array([len(ep.get('gt_action', [])) for ep in test_episodes])
    test_instr = np.array([len(ep.get('instruction', '').split()) for ep in test_episodes])
    
    print(f"\n   Val Action Length:   mean={np.mean(val_action):.1f}, std={np.std(val_action):.1f}")
    print(f"   Test Action Length:  mean={np.mean(test_action):.1f}, std={np.std(test_action):.1f}")
    print(f"\n   Val Instruction:     mean={np.mean(val_instr):.1f}, std={np.std(val_instr):.1f}")
    print(f"   Test Instruction:    mean={np.mean(test_instr):.1f}, std={np.std(test_instr):.1f}")
    
    # KS检验
    from scipy import stats
    print("\n🔬 Kolmogorov-Smirnov Test (before balancing):")
    ks_action_before = stats.ks_2samp(val_action, test_action)
    ks_instr_before = stats.ks_2samp(val_instr, test_instr)
    print(f"   Action Length:     statistic={ks_action_before.statistic:.4f}, p-value={ks_action_before.pvalue:.4f}")
    print(f"   Instruction Length: statistic={ks_instr_before.statistic:.4f}, p-value={ks_instr_before.pvalue:.4f}")
    
    # 执行平衡
    print("\n" + "=" * 70)
    print("  Balancing Process")
    print("=" * 70)
    
    balanced_indices = iterative_balance_distribution(
        test_episodes, val_episodes, n_bins=12, max_iterations=5
    )
    
    # 创建平衡后的数据集
    test_episodes_balanced = [test_episodes[i] for i in balanced_indices]
    
    # KS检验 - 平衡后
    balanced_action = np.array([len(ep.get('gt_action', [])) for ep in test_episodes_balanced])
    balanced_instr = np.array([len(ep.get('instruction', '').split()) for ep in test_episodes_balanced])
    
    print("\n🔬 Kolmogorov-Smirnov Test (after balancing):")
    ks_action_after = stats.ks_2samp(val_action, balanced_action)
    ks_instr_after = stats.ks_2samp(val_instr, balanced_instr)
    print(f"   Action Length:     statistic={ks_action_after.statistic:.4f}, p-value={ks_action_after.pvalue:.4f}")
    print(f"   Instruction Length: statistic={ks_instr_after.statistic:.4f}, p-value={ks_instr_after.pvalue:.4f}")
    
    # 统计信息对比
    print("\n📊 Balanced Distribution Stats:")
    print(f"   Balanced Test Action Length:  mean={np.mean(balanced_action):.1f}, std={np.std(balanced_action):.1f}")
    print(f"   Balanced Test Instruction:   mean={np.mean(balanced_instr):.1f}, std={np.std(balanced_instr):.1f}")
    print(f"   Balanced Test Size: {len(test_episodes_balanced)} episodes")
    
    # 绘制对比图
    print("\n📈 Generating comparison plots...")
    ks_results = plot_before_after(
        val_episodes, test_episodes, test_episodes_balanced, output_plot
    )
    
    # 保存平衡后的数据集
    print("\n💾 Saving balanced dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 按scene分组保存
    scenes = defaultdict(list)
    for ep in test_episodes_balanced:
        scene_id = ep.get('scene_id', 'unknown')
        if 'hm3d/' in scene_id:
            scene_name = scene_id.split('/')[-2]
        else:
            scene_name = 'unknown'
        scenes[scene_name].append(ep)
    
    for scene_name, scene_eps in scenes.items():
        filepath = os.path.join(output_dir, f"{scene_name}.json.gz")
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump({'episodes': scene_eps}, f)
    
    print(f"   Saved to: {output_dir}")
    
    # 总结
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"""
   Original Test Dataset:  {len(test_episodes)} episodes
   Balanced Test Dataset:  {len(test_episodes_balanced)} episodes
   
   Distribution Matching:
   ┌─────────────────┬────────────────────┬────────────────────┐
   │ Metric          │ Before (p-value)   │ After (p-value)    │
   ├─────────────────┼────────────────────┼────────────────────┤
   │ Action Length   │ {ks_action_before.pvalue:.4f}            │ {ks_action_after.pvalue:.4f}            │
   │ Instruction     │ {ks_instr_before.pvalue:.4f}            │ {ks_instr_after.pvalue:.4f}            │
   └─────────────────┴────────────────────┴────────────────────┘
   
   p-value > 0.05: Cannot reject null hypothesis (distributions are similar)
   
   Output files:
   - Balanced dataset: {output_dir}
   - Comparison plot: {output_plot}
""")
    
    print("=" * 70)
    print("  ✅ Balancing complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
