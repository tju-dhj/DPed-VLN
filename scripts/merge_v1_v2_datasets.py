#!/usr/bin/env python3
"""
合并v1和v2数据集脚本
从dynamic_dataset_final_v1和dynamic_dataset_final_v2合并训练集和验证集
生成新的混合数据集dynamic_dataset_final_v1v2
"""

import os
import json
import gzip
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def load_json_gz(filepath):
    """加载gzip压缩的JSON文件"""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)

def save_json_gz(data, filepath):
    """保存为gzip压缩的JSON文件"""
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_datasets(v1_dir, v2_dir, output_dir, split='train'):
    """
    合并v1和v2数据集
    
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
    v1_scenes = set(f.name for f in v1_path.glob('*.json.gz'))
    v2_scenes = set(f.name for f in v2_path.glob('*.json.gz'))
    all_scenes = v1_scenes | v2_scenes
    
    print(f"合并{split}数据集:")
    print(f"  V1 scenes: {len(v1_scenes)}")
    print(f"  V2 scenes: {len(v2_scenes)}")
    print(f"  总scenes: {len(all_scenes)}")
    
    merged_count = 0
    v1_only_count = 0
    v2_only_count = 0
    
    for scene_file in tqdm(sorted(all_scenes), desc=f"处理{split}场景"):
        v1_file = v1_path / scene_file
        v2_file = v2_path / scene_file
        output_file = output_path / scene_file
        
        merged_episodes = []
        episode_ids = set()  # 用于去重
        
        # 加载v1数据
        if v1_file.exists():
            v1_data = load_json_gz(v1_file)
            for ep in v1_data.get('episodes', []):
                ep_id = ep.get('episode_id', '')
                # 添加标识，表明这是v1指令的episode
                ep['instruction_source'] = 'v1'
                merged_episodes.append(ep)
                episode_ids.add(ep_id)
            v1_only_count += 1
        
        # 加载v2数据
        if v2_file.exists():
            v2_data = load_json_gz(v2_file)
            for ep in v2_data.get('episodes', []):
                ep_id = ep.get('episode_id', '')
                # 如果episode_id相同，跳过（避免重复）
                if ep_id not in episode_ids:
                    # 添加标识，表明这是v2指令的episode
                    ep['instruction_source'] = 'v2'
                    merged_episodes.append(ep)
                    episode_ids.add(ep_id)
                else:
                    # 如果episode_id相同，保留v1版本（v1优先）
                    pass
            if not v1_file.exists():
                v2_only_count += 1
        
        # 如果两个文件都存在，标记为合并
        if v1_file.exists() and v2_file.exists():
            merged_count += 1
        
        # 保存合并后的数据
        merged_data = {'episodes': merged_episodes}
        save_json_gz(merged_data, output_file)
    
    print(f"\n合并完成:")
    print(f"  同时包含v1和v2的scenes: {merged_count}")
    print(f"  仅v1的scenes: {v1_only_count}")
    print(f"  仅v2的scenes: {v2_only_count}")
    print(f"  总episodes: {sum(len(load_json_gz(output_path / f)['episodes']) for f in all_scenes)}")

def main():
    base_dir = Path("data")
    v1_dir = base_dir / "dynamic_dataset_final_v1"
    v2_dir = base_dir / "dynamic_dataset_final_v2"
    output_dir = base_dir / "dynamic_dataset_final_v1v2"
    
    print("=" * 60)
    print("合并v1和v2数据集")
    print("=" * 60)
    
    # 合并训练集
    print("\n" + "=" * 60)
    merge_datasets(v1_dir, v2_dir, output_dir, split='train')
    
    # 合并验证集
    print("\n" + "=" * 60)
    merge_datasets(v1_dir, v2_dir, output_dir, split='val_full')
    
    print("\n" + "=" * 60)
    print("数据集合并完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

