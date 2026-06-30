#!/usr/bin/env python3
"""
基于unseen数据集分布，重新划分seen数据集
目标：100 val + 700 test，总共800个样本
同时进行数据清洗
"""

import os
import gzip
import json
import glob
import numpy as np
from scipy import stats
from collections import defaultdict
import random
import shutil

random.seed(42)
np.random.seed(42)

def load_all_episodes(directory):
    """加载目录中所有json.gz文件的所有episodes"""
    episodes_data = []
    json_files = glob.glob(os.path.join(directory, "**", "*.json.gz"), recursive=True)
    
    for filepath in json_files:
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            basename = os.path.basename(filepath)
            
            for episode in data.get('episodes', []):
                gt_action = episode.get('gt_action', [])
                instruction = episode.get('instruction', '')
                
                if gt_action and instruction:
                    episodes_data.append({
                        'filepath': filepath,
                        'basename': basename,
                        'action_length': len(gt_action),
                        'instruction_length': len(instruction.strip()),
                        'instruction': instruction,
                        'episode_id': episode.get('episode_id', ''),
                        'scene_id': episode.get('scene_id', ''),
                        'episode': episode,
                        'start_position': episode.get('start_position', []),
                        'start_rotation': episode.get('start_rotation', []),
                    })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return episodes_data

def is_valid_instruction(instr, action_len):
    """检查指令是否有效"""
    instr = instr.strip()
    instr_len = len(instr)
    
    # 太短
    if instr_len < 20:
        return False, "too_short"
    
    # 去掉异常长的指令（包含大量重复换行符等）
    if instr_len > 3000:
        return False, "too_long"
    
    # 检查是否是格式错误的指令
    if ' ' not in instr and len(instr) < 50:
        return False, "no_spaces"
    
    # 检查是否几乎全是数字
    cleaned = instr.replace(' ', '').replace('.', '').replace(',', '')
    if cleaned.isdigit() or (cleaned.replace('-', '').replace('.', '').isdigit()):
        return False, "only_numbers"
    
    # 检查重复度（过于重复的指令）
    if len(instr) > 100:
        words = instr.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False, "too_repetitive"
    
    # 检查是否包含异常字符（大量换行符连续）
    if instr.count('\n\n') > 5 or instr.count('\n') > 20:
        return False, "too_many_newlines"
    
    return True, "valid"

def compute_distribution_distance(selected, target_action, target_instr):
    """
    计算选中样本与目标分布的差异
    使用 Kolmogorov-Smirnov 检验的统计量
    """
    if len(selected) == 0:
        return float('inf')
    
    selected_actions = [ep['action_length'] for ep in selected]
    selected_instrs = [ep['instruction_length'] for ep in selected]
    
    # KS 统计量
    action_ks, _ = stats.ks_2samp(selected_actions, target_action)
    instr_ks, _ = stats.ks_2samp(selected_instrs, target_instr)
    
    # 综合距离
    return action_ks + instr_ks

def select_best_matches(episodes, target_action, target_instr, n_samples, step=50):
    """
    贪心选择最匹配目标分布的样本
    """
    # 先过滤掉无效指令
    valid_episodes = []
    for ep in episodes:
        is_valid, reason = is_valid_instruction(ep['instruction'], ep['action_length'])
        if is_valid:
            valid_episodes.append(ep)
    
    print(f"过滤前: {len(episodes)}, 过滤后: {len(valid_episodes)}")
    
    # 按action长度分层，确保覆盖目标分布
    target_action_sorted = sorted(target_action)
    
    selected = []
    remaining = list(valid_episodes)
    
    # 目标分位数
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    target_quantile_values = np.percentile(target_action, [q * 100 for q in quantiles])
    
    print(f"目标action长度分位数: {target_quantile_values}")
    
    # 贪心选择
    best_distance = float('inf')
    best_selection = None
    
    # 分批尝试不同的选择
    for iteration in range(5):  # 多次尝试
        current_selection = []
        current_remaining = list(valid_episodes)
        random.shuffle(current_remaining)
        
        # 按action长度分层选择
        for q_idx, q_val in enumerate(target_quantile_values):
            # 从该长度区间选择样本
            n_per_quantile = n_samples // len(quantiles)
            
            # 找到该分位数附近的样本
            candidates = []
            for ep in current_remaining:
                action_diff = abs(ep['action_length'] - q_val)
                instr_target = np.percentile(target_instr, [(q_idx/len(quantiles))*100, ((q_idx+1)/len(quantiles))*100])
                instr_diff = abs(ep['instruction_length'] - np.mean(instr_target))
                
                # 综合评分（越小越好）
                score = action_diff * 0.6 + instr_diff * 0.4
                candidates.append((score, ep))
            
            # 按score排序
            candidates.sort(key=lambda x: x[0])
            
            for score, ep in candidates[:n_per_quantile]:
                if ep in current_remaining:
                    current_selection.append(ep)
                    current_remaining.remove(ep)
        
        # 如果选择的数量不够，从剩余样本中随机补充
        while len(current_selection) < n_samples and current_remaining:
            ep = random.choice(current_remaining)
            current_selection.append(ep)
            current_remaining.remove(ep)
        
        # 如果选择的数量超过，移除最远的
        while len(current_selection) > n_samples:
            # 找到对分布贡献最小的移除
            min_contribution = float('inf')
            worst_ep = None
            for ep in current_selection:
                contrib = abs(ep['action_length'] - np.mean([e['action_length'] for e in current_selection]))
                if contrib < min_contribution:
                    min_contribution = contrib
                    worst_ep = ep
            if worst_ep:
                current_selection.remove(worst_ep)
        
        # 计算当前选择的分布距离
        current_actions = [e['action_length'] for e in current_selection]
        current_instrs = [e['instruction_length'] for e in current_selection]
        
        action_ks, _ = stats.ks_2samp(current_actions, target_action)
        instr_ks, _ = stats.ks_2samp(current_instrs, target_instr)
        current_distance = action_ks + instr_ks
        
        if current_distance < best_distance:
            best_distance = current_distance
            best_selection = current_selection
            print(f"Iteration {iteration}: KS distance = {current_distance:.4f} (action={action_ks:.4f}, instr={instr_ks:.4f})")
    
    return best_selection, valid_episodes

def main():
    print("=" * 70)
    print("重新划分数据集")
    print("=" * 70)
    
    # 目标分布（unseen）
    unseen_dirs = [
        "/share/home/u19666033/dhj/DPed_pro/dataset_splits/unseen/unseen_test",
        "/share/home/u19666033/dhj/DPed_pro/dataset_splits/unseen/unseen_val"
    ]
    
    unseen_action = []
    unseen_instr = []
    
    for dir_path in unseen_dirs:
        if os.path.exists(dir_path):
            episodes = load_all_episodes(dir_path)
            unseen_action.extend([e['action_length'] for e in episodes])
            unseen_instr.extend([e['instruction_length'] for e in episodes])
    
    target_action = np.array(unseen_action)
    target_instr = np.array(unseen_instr)
    
    print(f"目标分布 (Unseen):")
    print(f"  Action: mean={np.mean(target_action):.2f}, std={np.std(target_action):.2f}, n={len(target_action)}")
    print(f"  Instruction: mean={np.mean(target_instr):.2f}, std={np.std(target_instr):.2f}, n={len(target_instr)}")
    
    # 加载原始train数据
    train_dir = "/share/home/u19666033/dhj/DPed_pro/data/dynamic_dataset_final_v1/train"
    train_episodes = load_all_episodes(train_dir)
    print(f"\n原始Train数据: {len(train_episodes)} episodes")
    
    # 统计过滤情况
    valid_count = 0
    invalid_stats = defaultdict(int)
    for ep in train_episodes:
        is_valid, reason = is_valid_instruction(ep['instruction'], ep['action_length'])
        if is_valid:
            valid_count += 1
        else:
            invalid_stats[reason] += 1
    
    print(f"有效episodes: {valid_count}")
    print(f"无效原因分布: {dict(invalid_stats)}")
    
    # 选择800个样本用于seen
    n_seen = 800
    n_val = 100
    n_test = 700
    
    print(f"\n选择 {n_seen} 个样本用于 seen 数据集 ({n_val} val + {n_test} test)...")
    
    selected, remaining = select_best_matches(
        train_episodes, 
        target_action.tolist(), 
        target_instr.tolist(),
        n_seen
    )
    
    if selected is None:
        print("选择失败!")
        return
    
    print(f"\n选中 {len(selected)} 个样本")
    
    # 分析选中样本的分布
    selected_actions = [e['action_length'] for e in selected]
    selected_instrs = [e['instruction_length'] for e in selected]
    
    print(f"选中样本分布:")
    print(f"  Action: mean={np.mean(selected_actions):.2f}, std={np.std(selected_actions):.2f}")
    print(f"  Instruction: mean={np.mean(selected_instrs):.2f}, std={np.std(selected_instrs):.2f}")
    
    # KS检验
    action_ks, action_p = stats.ks_2samp(selected_actions, target_action.tolist())
    instr_ks, instr_p = stats.ks_2samp(selected_instrs, target_instr.tolist())
    print(f"  KS检验: action KS={action_ks:.4f} (p={action_p:.4f}), instr KS={instr_ks:.4f} (p={instr_p:.4f})")
    
    # 打乱并分割为val和test
    random.shuffle(selected)
    seen_val = selected[:n_val]
    seen_test = selected[n_val:]
    
    print(f"\nSeen Val: {len(seen_val)} 个")
    print(f"Seen Test: {len(seen_test)} 个")
    
    # 剩余的作为新的train
    # 需要从remaining中排除已选择的
    remaining_files = set(e['basename'] for e in remaining)
    selected_files = set(e['basename'] for e in selected)
    
    # 实际剩余的episodes
    actual_remaining = [e for e in remaining if e['basename'] not in selected_files]
    print(f"\n剩余train episodes: {len(actual_remaining)}")
    print(f"涉及文件数: {len(set(e['basename'] for e in actual_remaining))}")
    
    # 保存结果
    output_base = "/share/home/u19666033/dhj/DPed_pro/dataset_splits_v2"
    os.makedirs(f"{output_base}/seen/seen_val", exist_ok=True)
    os.makedirs(f"{output_base}/seen/seen_test", exist_ok=True)
    os.makedirs(f"{output_base}/train", exist_ok=True)
    os.makedirs(f"{output_base}/unseen/unseen_val", exist_ok=True)
    os.makedirs(f"{output_base}/unseen/unseen_test", exist_ok=True)
    
    # 按文件聚合episodes并保存
    def save_episodes_to_file(episodes, output_path, subdir_name):
        """将episodes按文件聚合保存"""
        by_file = defaultdict(list)
        for ep in episodes:
            by_file[ep['basename']].append(ep)
        
        for basename, eps in by_file.items():
            output_file = os.path.join(output_path, basename)
            # 读取原始文件获取完整结构
            orig_path = None
            for search_path in [
                "/share/home/u19666033/dhj/DPed_pro/data/dynamic_dataset_final_v1/train",
                "/share/home/u19666033/dhj/DPed_pro/dataset_splits/unseen/unseen_test",
                "/share/home/u19666033/dhj/DPed_pro/dataset_splits/unseen/unseen_val"
            ]:
                candidate = os.path.join(search_path, basename)
                if os.path.exists(candidate):
                    orig_path = candidate
                    break
            
            if orig_path:
                with gzip.open(orig_path, 'rt') as f:
                    full_data = json.load(f)
                
                # 过滤episodes
                selected_ep_ids = set(e['episode_id'] for e in eps)
                full_data['episodes'] = [e for e in full_data['episodes'] if str(e.get('episode_id', '')) in selected_ep_ids]
                
                # 保存
                with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                    json.dump(full_data, f)
    
    # 保存seen val
    print("\n保存 seen val...")
    save_episodes_to_file(seen_val, f"{output_base}/seen/seen_val", "seen_val")
    
    # 保存seen test
    print("保存 seen test...")
    save_episodes_to_file(seen_test, f"{output_base}/seen/seen_test", "seen_test")
    
    # 保存unseen（直接复制）
    print("复制 unseen...")
    for subdir in ["unseen_test", "unseen_val"]:
        src = f"/share/home/u19666033/dhj/DPed_pro/dataset_splits/unseen/{subdir}"
        dst = f"{output_base}/unseen/{subdir}"
        if os.path.exists(src):
            for f in os.listdir(src):
                shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    
    # 保存train
    print("保存 train...")
    save_episodes_to_file(actual_remaining, f"{output_base}/train", "train")
    
    print(f"\n完成! 数据保存在: {output_base}")
    
    # 统计保存的文件数
    print("\n文件统计:")
    for subdir in ["seen/seen_val", "seen/seen_test", "unseen/seen_val", "unseen/seen_test", "train"]:
        path = f"{output_base}/{subdir}"
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"  {subdir}: {count} files")
    
    return {
        'seen_val': seen_val,
        'seen_test': seen_test,
        'remaining': actual_remaining,
        'invalid_stats': dict(invalid_stats)
    }

if __name__ == "__main__":
    results = main()
