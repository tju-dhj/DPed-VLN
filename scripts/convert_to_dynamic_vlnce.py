#!/usr/bin/env python3
"""
将筛选后的数据集转换为DynamicVLNCE格式
"""

import os
import json
import gzip
import argparse
from pathlib import Path
from typing import Dict, List, Any


def convert_filtered_dataset_to_dynamic_vlnce(
    filtered_dataset_dir: str,
    output_dir: str,
    dataset_name: str = "dynamic_vlnce"
):
    """
    将筛选后的数据集转换为DynamicVLNCE格式
    
    Args:
        filtered_dataset_dir: 筛选后数据集目录
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有场景文件
    scene_files = [f for f in os.listdir(filtered_dataset_dir) if f.endswith('.json.gz')]
    
    print(f"找到 {len(scene_files)} 个场景文件")
    
    # 构建指令词汇表
    instruction_vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
    vocab_idx = 4
    
    all_episodes = []
    total_episodes = 0
    
    for scene_file in scene_files:
        scene_name = scene_file.replace('.json.gz', '')
        print(f"处理场景: {scene_name}")
        
        # 读取场景数据
        scene_path = os.path.join(filtered_dataset_dir, scene_file)
        with gzip.open(scene_path, 'rt', encoding='utf-8') as f:
            scene_data = json.load(f)
        
        # 处理每个episode
        for episode in scene_data.get('episodes', []):
            # 构建DynamicVLNCE episode
            dynamic_episode = {
                "episode_id": episode.get('episode_id', str(total_episodes)),
                "scene_id": episode.get('scene_id', scene_name),
                "start_position": episode.get('start_position', [0.0, 0.0, 0.0]),
                "start_rotation": episode.get('start_rotation', [0.0, 0.0, 0.0, 1.0]),
                "goals": episode.get('goals', []),
                "instruction": episode.get('instruction', ''),
                "instruction_tokens": [],
                "instruction_id": episode.get('instruction_id', 0),
                "gt_action": episode.get('gt_action', []),
                "trajectory_id": episode.get('trajectory_id', f"{scene_name}_{episode.get('episode_id', total_episodes)}"),
                "path": episode.get('path', []),
                "distance_to_goal": episode.get('distance_to_goal', 0.0),
                "reference_path": episode.get('reference_path', []),
                "info": episode.get('info', {})
            }
            
            # 处理指令tokenization
            instruction = dynamic_episode['instruction']
            if instruction:
                # 简单的tokenization
                tokens = instruction.split()
                dynamic_episode['instruction_tokens'] = tokens
                
                # 更新词汇表
                for token in tokens:
                    if token not in instruction_vocab:
                        instruction_vocab[token] = vocab_idx
                        vocab_idx += 1
            else:
                dynamic_episode['instruction_tokens'] = []
            
            # 确保gt_action是整数列表
            if isinstance(dynamic_episode['gt_action'], str):
                try:
                    dynamic_episode['gt_action'] = json.loads(dynamic_episode['gt_action'])
                except:
                    dynamic_episode['gt_action'] = []
            elif not isinstance(dynamic_episode['gt_action'], list):
                dynamic_episode['gt_action'] = []
            
            # 确保gt_action中的元素是整数
            dynamic_episode['gt_action'] = [int(x) for x in dynamic_episode['gt_action']]
            
            all_episodes.append(dynamic_episode)
            total_episodes += 1
    
    print(f"总共处理了 {total_episodes} 个episodes")
    print(f"指令词汇表大小: {len(instruction_vocab)}")
    
    # 创建完整的数据集
    dataset = {
        "episodes": all_episodes,
        "instruction_vocab": {
            "word_list": list(instruction_vocab.keys()),
            "word_to_idx": instruction_vocab
        }
    }
    
    # 保存完整数据集
    output_file = os.path.join(output_dir, f"{dataset_name}.json.gz")
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"保存完整数据集到: {output_file}")
    
    # 按场景分割保存
    content_dir = os.path.join(output_dir, "content")
    os.makedirs(content_dir, exist_ok=True)
    
    # 按场景分组episodes
    scenes_episodes = {}
    for episode in all_episodes:
        scene_id = episode['scene_id']
        if scene_id not in scenes_episodes:
            scenes_episodes[scene_id] = []
        scenes_episodes[scene_id].append(episode)
    
    # 保存每个场景的数据
    for scene_id, episodes in scenes_episodes.items():
        scene_data = {
            "episodes": episodes,
            "instruction_vocab": {
                "word_list": list(instruction_vocab.keys()),
                "word_to_idx": instruction_vocab
            }
        }
        
        scene_file = os.path.join(content_dir, f"{scene_id}.json.gz")
        with gzip.open(scene_file, 'wt', encoding='utf-8') as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
        
        print(f"保存场景 {scene_id}: {len(episodes)} 个episodes")
    
    # 生成统计信息
    stats = {
        "total_episodes": total_episodes,
        "total_scenes": len(scenes_episodes),
        "instruction_vocab_size": len(instruction_vocab),
        "episodes_with_gt_actions": sum(1 for ep in all_episodes if len(ep['gt_action']) > 0),
        "episodes_with_instructions": sum(1 for ep in all_episodes if ep['instruction']),
        "average_instruction_length": sum(len(ep['instruction_tokens']) for ep in all_episodes) / total_episodes if total_episodes > 0 else 0,
        "average_action_length": sum(len(ep['gt_action']) for ep in all_episodes) / total_episodes if total_episodes > 0 else 0,
    }
    
    stats_file = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"保存统计信息到: {stats_file}")
    print("\n数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='转换筛选数据集为DynamicVLNCE格式')
    parser.add_argument('--filtered_dataset_dir', 
                       default='/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/filtered_dataset',
                       help='筛选后数据集目录')
    parser.add_argument('--output_dir',
                       default='/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/dynamic_vlnce_dataset',
                       help='输出目录')
    parser.add_argument('--dataset_name',
                       default='dynamic_vlnce',
                       help='数据集名称')
    
    args = parser.parse_args()
    
    print("开始转换数据集...")
    print(f"输入目录: {args.filtered_dataset_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据集名称: {args.dataset_name}")
    
    convert_filtered_dataset_to_dynamic_vlnce(
        args.filtered_dataset_dir,
        args.output_dir,
        args.dataset_name
    )
    
    print("转换完成!")


if __name__ == "__main__":
    main()



