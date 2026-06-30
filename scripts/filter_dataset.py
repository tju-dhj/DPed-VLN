#!/usr/bin/env python3
"""
筛选HM3D数据集，只保留在收集数据中存在的episodes，并添加对应的instructions
"""

import os
import json
import gzip
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse

def get_collected_episodes(collect_data_dir: str) -> Dict[str, Set[str]]:
    """
    获取收集数据中存在的episodes编号
    返回格式: {scene_name: {episode_ids}}
    """
    collected_episodes = {}
    
    # 遍历收集数据目录
    for scene_dir in os.listdir(collect_data_dir):
        scene_path = os.path.join(collect_data_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue
            
        # 提取场景名称（去掉.basis后缀）
        scene_name = scene_dir.replace('.basis', '')
        episode_ids = set()
        
        # 遍历该场景下的所有episode目录
        for episode_dir in os.listdir(scene_path):
            episode_path = os.path.join(scene_path, episode_dir)
            if os.path.isdir(episode_path):
                episode_ids.add(episode_dir)
        
        if episode_ids:
            collected_episodes[scene_name] = episode_ids
            print(f"场景 {scene_name}: 找到 {len(episode_ids)} 个episodes")
    
    return collected_episodes

def load_instruction(collect_data_dir: str, scene_name: str, episode_id: str) -> str:
    """
    加载指定episode的instruction
    """
    instruction_path = os.path.join(
        collect_data_dir, 
        f"{scene_name}.basis", 
        episode_id, 
        "inst_navcomposer_v2_ped", 
        "0.txt"
    )
    
    if os.path.exists(instruction_path):
        with open(instruction_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        print(f"警告: 找不到instruction文件: {instruction_path}")
        return ""

def load_gt_action(collect_data_dir: str, scene_name: str, episode_id: str) -> List[int]:
    """
    加载指定episode的gt_action
    """
    action_path = os.path.join(
        collect_data_dir, 
        f"{scene_name}.basis", 
        episode_id, 
        "action", 
        "0.json"
    )
    
    if os.path.exists(action_path):
        with open(action_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"警告: 找不到action文件: {action_path}")
        return []

def filter_and_enhance_dataset(
    original_data_dir: str,
    collect_data_dir: str,
    output_dir: str,
    collected_episodes: Dict[str, Set[str]]
):
    """
    筛选并增强数据集
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有原始数据文件
    original_files = glob.glob(os.path.join(original_data_dir, "*.json.gz"))
    
    processed_count = 0
    total_episodes = 0
    filtered_episodes = 0
    
    for original_file in original_files:
        # 提取场景名称
        scene_name = os.path.basename(original_file).replace('.json.gz', '')
        
        if scene_name not in collected_episodes:
            print(f"跳过场景 {scene_name}: 没有收集数据")
            continue
            
        print(f"处理场景 {scene_name}...")
        
        # 读取原始数据
        with gzip.open(original_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        # 筛选episodes
        filtered_episodes_list = []
        available_episodes = collected_episodes[scene_name]
        
        for episode in data['episodes']:
            episode_id = episode['episode_id']
            total_episodes += 1
            
            if episode_id in available_episodes:
                # 添加instruction
                instruction = load_instruction(collect_data_dir, scene_name, episode_id)
                episode['instruction'] = instruction
                
                # 添加gt_action
                gt_action = load_gt_action(collect_data_dir, scene_name, episode_id)
                episode['gt_action'] = gt_action
                
                filtered_episodes_list.append(episode)
                filtered_episodes += 1
        
        if filtered_episodes_list:
            # 创建新的数据集
            new_data = {
                'episodes': filtered_episodes_list
            }
            
            # 保存为json.gz格式
            output_file = os.path.join(output_dir, f"{scene_name}.json.gz")
            with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                json.dump(new_data, f, indent=2, ensure_ascii=False)
            
            print(f"  保存 {len(filtered_episodes_list)} 个episodes到 {output_file}")
            processed_count += 1
        else:
            print(f"  场景 {scene_name} 没有匹配的episodes")
    
    print(f"\n处理完成:")
    print(f"  处理了 {processed_count} 个场景")
    print(f"  总episodes: {total_episodes}")
    print(f"  筛选后episodes: {filtered_episodes}")
    print(f"  输出目录: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='筛选HM3D数据集')
    parser.add_argument('--original_data_dir', 
                       default='/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/datasets/pointnav/social-hm3d/train/content',
                       help='原始数据目录')
    parser.add_argument('--collect_data_dir',
                       default='/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train',
                       help='收集数据目录')
    parser.add_argument('--output_dir',
                       default='/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/filtered_dataset',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("开始筛选数据集...")
    print(f"原始数据目录: {args.original_data_dir}")
    print(f"收集数据目录: {args.collect_data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 获取收集的episodes
    print("\n扫描收集数据...")
    collected_episodes = get_collected_episodes(args.collect_data_dir)
    
    # 筛选并增强数据集
    print("\n开始筛选和增强数据集...")
    filter_and_enhance_dataset(
        args.original_data_dir,
        args.collect_data_dir,
        args.output_dir,
        collected_episodes
    )

if __name__ == "__main__":
    main()
