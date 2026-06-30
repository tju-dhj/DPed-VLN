#!/usr/bin/env python3
"""
创建v1v2混合数据集脚本
从dynamic_dataset_final_v1和dynamic_dataset_final_v2创建混合数据集
每个episode出现两次：一次使用v1指令，一次使用v2指令
数据量翻倍，适合训练使用v1和v2两种指令的模型
"""

import os
import json
import gzip
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def load_json_gz(filepath):
    """加载gzip压缩的JSON文件"""
    if not os.path.exists(filepath):
        return None
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)

def save_json_gz(data, filepath):
    """保存为gzip压缩的JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_mixed_episodes(v1_episode, v2_episode, episode_id_base):
    """
    为每个episode创建v1和v2两个副本
    
    Args:
        v1_episode: v1数据集的episode（包含v1指令）
        v2_episode: v2数据集的episode（包含v2指令，可能为None）
        episode_id_base: 原始episode_id
    
    Returns:
        list: 包含v1和v2两个episode的列表
    """
    mixed_episodes = []
    
    # 创建v1指令的episode副本
    if v1_episode is not None:
        v1_copy = v1_episode.copy()
        # 修改episode_id以区分v1和v2版本
        v1_copy['episode_id'] = f"{episode_id_base}_v1"
        v1_copy['instruction_source'] = 'v1'
        # 确保使用v1指令（如果instruction字段存在）
        if 'instruction' in v1_copy:
            v1_copy['instruction'] = v1_copy.get('instruction', '')
        mixed_episodes.append(v1_copy)
    
    # 创建v2指令的episode副本
    if v2_episode is not None:
        v2_copy = v2_episode.copy()
        # 修改episode_id以区分v1和v2版本
        v2_copy['episode_id'] = f"{episode_id_base}_v2"
        v2_copy['instruction_source'] = 'v2'
        # 确保使用v2指令（如果instruction字段存在）
        if 'instruction' in v2_copy:
            v2_copy['instruction'] = v2_copy.get('instruction', '')
        mixed_episodes.append(v2_copy)
    elif v1_episode is not None:
        # 如果只有v1数据，也创建v2副本（使用v1指令，但标记为v2）
        # 这样确保每个episode都有两个副本
        v2_copy = v1_episode.copy()
        v2_copy['episode_id'] = f"{episode_id_base}_v2"
        v2_copy['instruction_source'] = 'v2'
        mixed_episodes.append(v2_copy)
    
    return mixed_episodes

def merge_datasets_with_duplication(v1_dir, v2_dir, output_dir, split='train'):
    """
    合并v1和v2数据集，每个episode出现两次（v1和v2各一次）
    
    Args:
        v1_dir: v1数据集目录
        v2_dir: v2数据集目录
        output_dir: 输出目录
        split: 'train' 或 'val_full'
    """
    v1_path = Path(v1_dir) / split
    v2_path = Path(v2_dir) / split
    output_path = Path(output_dir) / split
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有scene文件
    v1_scenes = set(f.name for f in v1_path.glob('*.json.gz')) if v1_path.exists() else set()
    v2_scenes = set(f.name for f in v2_path.glob('*.json.gz')) if v2_path.exists() else set()
    all_scenes = v1_scenes | v2_scenes
    
    print(f"\n合并{split}数据集（每个episode出现两次）:")
    print(f"  V1 scenes: {len(v1_scenes)}")
    print(f"  V2 scenes: {len(v2_scenes)}")
    print(f"  总scenes: {len(all_scenes)}")
    
    total_v1_episodes = 0
    total_v2_episodes = 0
    total_mixed_episodes = 0
    
    for scene_file in tqdm(sorted(all_scenes), desc=f"处理{split}场景"):
        v1_file = v1_path / scene_file
        v2_file = v2_path / scene_file
        output_file = output_path / scene_file
        
        # 加载v1和v2数据
        v1_data = load_json_gz(v1_file) if v1_file.exists() else None
        v2_data = load_json_gz(v2_file) if v2_file.exists() else None
        
        # 创建episode映射（按原始episode_id）
        v1_episodes_dict = {}
        v2_episodes_dict = {}
        
        if v1_data:
            for ep in v1_data.get('episodes', []):
                # 提取原始episode_id（去除可能的_v1或_v2后缀）
                ep_id = ep.get('episode_id', '')
                base_id = ep_id.rsplit('_v1', 1)[0].rsplit('_v2', 1)[0]
                v1_episodes_dict[base_id] = ep
                total_v1_episodes += 1
        
        if v2_data:
            for ep in v2_data.get('episodes', []):
                ep_id = ep.get('episode_id', '')
                base_id = ep_id.rsplit('_v1', 1)[0].rsplit('_v2', 1)[0]
                v2_episodes_dict[base_id] = ep
                total_v2_episodes += 1
        
        # 获取所有唯一的episode base_id
        all_episode_ids = set(v1_episodes_dict.keys()) | set(v2_episodes_dict.keys())
        
        # 为每个episode创建v1和v2两个副本
        mixed_episodes = []
        for base_id in sorted(all_episode_ids):
            v1_ep = v1_episodes_dict.get(base_id)
            v2_ep = v2_episodes_dict.get(base_id)
            
            episodes = create_mixed_episodes(v1_ep, v2_ep, base_id)
            mixed_episodes.extend(episodes)
            total_mixed_episodes += len(episodes)
        
        # 保存合并后的数据
        merged_data = {'episodes': mixed_episodes}
        save_json_gz(merged_data, str(output_file))
    
    print(f"\n合并统计:")
    print(f"  V1原始episodes: {total_v1_episodes}")
    print(f"  V2原始episodes: {total_v2_episodes}")
    print(f"  混合后总episodes: {total_mixed_episodes} (每个episode出现2次)")
    print(f"  数据量增加: {total_mixed_episodes / max(total_v1_episodes + total_v2_episodes, 1):.2f}x")

def main():
    base_dir = Path("data")
    v1_dir = base_dir / "dynamic_dataset_final_v1"
    v2_dir = base_dir / "dynamic_dataset_final_v2"
    output_dir = base_dir / "dynamic_dataset_final_v1v2_mixed"
    
    print("=" * 60)
    print("创建v1v2混合数据集（每个episode出现两次）")
    print("=" * 60)
    print(f"V1数据集: {v1_dir}")
    print(f"V2数据集: {v2_dir}")
    print(f"输出目录: {output_dir}")
    
    # 合并训练集
    print("\n" + "=" * 60)
    merge_datasets_with_duplication(v1_dir, v2_dir, output_dir, split='train')
    
    # 合并验证集
    print("\n" + "=" * 60)
    merge_datasets_with_duplication(v1_dir, v2_dir, output_dir, split='val_full')
    
    print("\n" + "=" * 60)
    print("数据集创建完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()



