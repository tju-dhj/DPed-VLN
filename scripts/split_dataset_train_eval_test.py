#!/usr/bin/env python3
"""
将 dynamic_dataset_final 目录下的数据集按照 8:1:1 的比例拆分为 train/eval/test
参考 dynamic_dataset 的目录结构
"""

import os
import json
import gzip
import glob
import random
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import argparse


def load_all_episodes(input_dir: str) -> Dict[str, List[Dict]]:
    """
    加载所有场景的episodes
    返回格式: {scene_name: [episodes]}
    """
    all_episodes = defaultdict(list)
    
    # 获取所有.json.gz文件
    json_files = glob.glob(os.path.join(input_dir, "*.json.gz"))
    
    print(f"找到 {len(json_files)} 个数据文件")
    
    for json_file in json_files:
        scene_name = os.path.basename(json_file).replace('.json.gz', '')
        print(f"  加载场景 {scene_name}...")
        
        try:
            with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'episodes' in data:
                all_episodes[scene_name].extend(data['episodes'])
                print(f"    场景 {scene_name}: {len(data['episodes'])} 个episodes")
            else:
                print(f"    警告: 场景 {scene_name} 没有episodes字段")
        except Exception as e:
            print(f"    错误: 加载场景 {scene_name} 失败: {e}")
    
    return all_episodes


def split_episodes_by_scene(
    all_episodes: Dict[str, List[Dict]], 
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    按场景拆分episodes为train/eval/test
    返回格式: {scene_name: {'train': [...], 'eval': [...], 'test': [...]}}
    """
    # 验证比例
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, \
        f"比例之和必须为1.0，当前为 {train_ratio + eval_ratio + test_ratio}"
    
    # 设置随机种子以确保可重复性
    random.seed(random_seed)
    
    split_data = {}
    
    for scene_name, episodes in all_episodes.items():
        # 打乱episodes
        shuffled_episodes = episodes.copy()
        random.shuffle(shuffled_episodes)
        
        total = len(shuffled_episodes)
        train_end = int(total * train_ratio)
        eval_end = train_end + int(total * eval_ratio)
        
        split_data[scene_name] = {
            'train': shuffled_episodes[:train_end],
            'eval': shuffled_episodes[train_end:eval_end],
            'test': shuffled_episodes[eval_end:]
        }
        
        print(f"场景 {scene_name}:")
        print(f"  总episodes: {total}")
        print(f"  train: {len(split_data[scene_name]['train'])} ({len(split_data[scene_name]['train'])/total*100:.1f}%)")
        print(f"  eval: {len(split_data[scene_name]['eval'])} ({len(split_data[scene_name]['eval'])/total*100:.1f}%)")
        print(f"  test: {len(split_data[scene_name]['test'])} ({len(split_data[scene_name]['test'])/total*100:.1f}%)")
    
    return split_data


def save_split_dataset(
    split_data: Dict[str, Dict[str, List[Dict]]],
    output_dir: str
):
    """
    保存拆分后的数据集到train/eval/test目录
    """
    output_path = Path(output_dir)
    
    # 创建输出目录
    train_dir = output_path / 'train'
    eval_dir = output_path / 'eval'
    test_dir = output_path / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    total_eval = 0
    total_test = 0
    
    for scene_name, splits in split_data.items():
        # 保存train数据
        if splits['train']:
            train_file = train_dir / f"{scene_name}.json.gz"
            train_data = {'episodes': splits['train']}
            with gzip.open(train_file, 'wt', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            total_train += len(splits['train'])
        
        # 保存eval数据
        if splits['eval']:
            eval_file = eval_dir / f"{scene_name}.json.gz"
            eval_data = {'episodes': splits['eval']}
            with gzip.open(eval_file, 'wt', encoding='utf-8') as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
            total_eval += len(splits['eval'])
        
        # 保存test数据
        if splits['test']:
            test_file = test_dir / f"{scene_name}.json.gz"
            test_data = {'episodes': splits['test']}
            with gzip.open(test_file, 'wt', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            total_test += len(splits['test'])
    
    print(f"\n保存完成:")
    print(f"  train目录: {len(list(train_dir.glob('*.json.gz')))} 个文件, {total_train} 个episodes")
    print(f"  eval目录: {len(list(eval_dir.glob('*.json.gz')))} 个文件, {total_eval} 个episodes")
    print(f"  test目录: {len(list(test_dir.glob('*.json.gz')))} 个文件, {total_test} 个episodes")
    print(f"  总计: {total_train + total_eval + total_test} 个episodes")
    print(f"\n输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='将数据集按照8:1:1拆分为train/eval/test')
    parser.add_argument('--input_dir',
                       default='data/dynamic_dataset_final',
                       help='输入数据目录（包含.json.gz文件）')
    parser.add_argument('--output_dir',
                       default='data/dynamic_dataset_final',
                       help='输出目录（将创建train/eval/test子目录）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例（默认0.8）')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                       help='验证集比例（默认0.1）')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例（默认0.1）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子（默认42）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据集拆分工具")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"拆分比例: train={args.train_ratio}, eval={args.eval_ratio}, test={args.test_ratio}")
    print(f"随机种子: {args.random_seed}")
    print("=" * 60)
    print()
    
    # 加载所有episodes
    print("步骤1: 加载所有episodes...")
    all_episodes = load_all_episodes(args.input_dir)
    
    if not all_episodes:
        print("错误: 没有找到任何episodes")
        return
    
    total_episodes = sum(len(eps) for eps in all_episodes.values())
    print(f"\n总共加载 {len(all_episodes)} 个场景, {total_episodes} 个episodes\n")
    
    # 拆分episodes
    print("步骤2: 按场景拆分episodes...")
    split_data = split_episodes_by_scene(
        all_episodes,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # 保存拆分后的数据集
    print("\n步骤3: 保存拆分后的数据集...")
    save_split_dataset(split_data, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 拆分完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
















