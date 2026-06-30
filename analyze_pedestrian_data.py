#!/usr/bin/env python3
"""
行人视野数据分析脚本

用于分析采集的episode数据中行人在机器人视野中的出现情况。
"""

import json
import pathlib
import argparse
from typing import List, Dict, Tuple
import numpy as np


def load_pedestrian_data(episode_path: pathlib.Path) -> List[int]:
    """
    加载单个episode的行人视野数据
    
    Args:
        episode_path: episode文件夹路径
        
    Returns:
        行人视野数据列表
    """
    pedestrian_file = episode_path / "pedestrian_in_view" / "0.json"
    if not pedestrian_file.exists():
        return []
    
    with open(pedestrian_file, "r") as f:
        return json.load(f)


def analyze_episode(episode_path: pathlib.Path) -> Dict:
    """
    分析单个episode的行人视野数据
    
    Args:
        episode_path: episode文件夹路径
        
    Returns:
        统计信息字典
    """
    pedestrian_data = load_pedestrian_data(episode_path)
    
    if not pedestrian_data:
        return {
            "episode_id": episode_path.name,
            "total_steps": 0,
            "steps_with_pedestrians": 0,
            "appearance_frequency": 0.0,
            "max_pedestrians": 0,
            "avg_pedestrians": 0.0,
            "pedestrian_data": []
        }
    
    total_steps = len(pedestrian_data)
    steps_with_pedestrians = sum(1 for x in pedestrian_data if x > 0)
    appearance_frequency = (steps_with_pedestrians / total_steps * 100) if total_steps > 0 else 0
    max_pedestrians = max(pedestrian_data)
    avg_pedestrians = sum(pedestrian_data) / total_steps if total_steps > 0 else 0
    
    return {
        "episode_id": episode_path.name,
        "total_steps": total_steps,
        "steps_with_pedestrians": steps_with_pedestrians,
        "appearance_frequency": appearance_frequency,
        "max_pedestrians": max_pedestrians,
        "avg_pedestrians": avg_pedestrians,
        "pedestrian_data": pedestrian_data
    }


def analyze_scene(scene_path: pathlib.Path) -> Tuple[List[Dict], Dict]:
    """
    分析整个场景的所有episode
    
    Args:
        scene_path: 场景文件夹路径
        
    Returns:
        (episode统计列表, 场景总体统计)
    """
    episode_stats = []
    
    for episode_dir in sorted(scene_path.iterdir()):
        if episode_dir.is_dir():
            stats = analyze_episode(episode_dir)
            if stats["total_steps"] > 0:
                episode_stats.append(stats)
    
    # 计算场景总体统计
    if episode_stats:
        scene_stats = {
            "scene_name": scene_path.name,
            "num_episodes": len(episode_stats),
            "total_steps": sum(s["total_steps"] for s in episode_stats),
            "total_steps_with_pedestrians": sum(s["steps_with_pedestrians"] for s in episode_stats),
            "avg_appearance_frequency": np.mean([s["appearance_frequency"] for s in episode_stats]),
            "max_pedestrians_overall": max(s["max_pedestrians"] for s in episode_stats),
            "avg_pedestrians_per_step": np.mean([s["avg_pedestrians"] for s in episode_stats]),
        }
    else:
        scene_stats = {
            "scene_name": scene_path.name,
            "num_episodes": 0,
            "total_steps": 0,
            "total_steps_with_pedestrians": 0,
            "avg_appearance_frequency": 0.0,
            "max_pedestrians_overall": 0,
            "avg_pedestrians_per_step": 0.0,
        }
    
    return episode_stats, scene_stats


def analyze_split(data_folder: pathlib.Path, split: str = "train") -> Tuple[List[Dict], List[Dict]]:
    """
    分析整个数据集split的所有场景
    
    Args:
        data_folder: 数据根目录
        split: 数据集分割名称
        
    Returns:
        (所有episode统计列表, 所有场景统计列表)
    """
    split_path = data_folder / split
    
    if not split_path.exists():
        print(f"错误: 路径不存在: {split_path}")
        return [], []
    
    all_episode_stats = []
    all_scene_stats = []
    
    for scene_dir in sorted(split_path.iterdir()):
        if scene_dir.is_dir():
            print(f"正在分析场景: {scene_dir.name}")
            episode_stats, scene_stats = analyze_scene(scene_dir)
            all_episode_stats.extend(episode_stats)
            all_scene_stats.append(scene_stats)
    
    return all_episode_stats, all_scene_stats


def print_statistics(episode_stats: List[Dict], scene_stats: List[Dict]):
    """
    打印统计信息
    
    Args:
        episode_stats: episode统计列表
        scene_stats: 场景统计列表
    """
    print("\n" + "="*80)
    print("行人视野数据分析报告")
    print("="*80)
    
    # 场景级别统计
    print("\n场景统计:")
    print("-"*80)
    for stats in scene_stats:
        print(f"场景: {stats['scene_name']}")
        print(f"  Episodes数量: {stats['num_episodes']}")
        print(f"  总步数: {stats['total_steps']}")
        print(f"  有行人的步数: {stats['total_steps_with_pedestrians']}")
        print(f"  平均出现频率: {stats['avg_appearance_frequency']:.1f}%")
        print(f"  最大同时行人数: {stats['max_pedestrians_overall']}")
        print(f"  平均每步行人数: {stats['avg_pedestrians_per_step']:.2f}")
        print()
    
    # 总体统计
    if episode_stats:
        print("\n总体统计:")
        print("-"*80)
        print(f"总Episodes数量: {len(episode_stats)}")
        print(f"总步数: {sum(s['total_steps'] for s in episode_stats)}")
        print(f"有行人的总步数: {sum(s['steps_with_pedestrians'] for s in episode_stats)}")
        
        # 计算全局出现频率
        total_steps = sum(s['total_steps'] for s in episode_stats)
        total_with_ped = sum(s['steps_with_pedestrians'] for s in episode_stats)
        global_freq = (total_with_ped / total_steps * 100) if total_steps > 0 else 0
        print(f"全局出现频率: {global_freq:.1f}%")
        
        print(f"平均每个episode出现频率: {np.mean([s['appearance_frequency'] for s in episode_stats]):.1f}%")
        print(f"最大同时行人数: {max(s['max_pedestrians'] for s in episode_stats)}")
        print(f"平均每步行人数: {np.mean([s['avg_pedestrians'] for s in episode_stats]):.2f}")
        
        # 频率分布
        print("\n出现频率分布:")
        print("-"*80)
        freq_ranges = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
        for low, high in freq_ranges:
            count = sum(1 for s in episode_stats if low <= s['appearance_frequency'] < high)
            pct = (count / len(episode_stats) * 100) if episode_stats else 0
            print(f"  {low:3d}% - {high:3d}%: {count:5d} episodes ({pct:5.1f}%)")
        
        # 100%的特殊处理
        count_100 = sum(1 for s in episode_stats if s['appearance_frequency'] == 100)
        pct_100 = (count_100 / len(episode_stats) * 100) if episode_stats else 0
        print(f"  100%         : {count_100:5d} episodes ({pct_100:5.1f}%)")
    
    print("\n" + "="*80)


def save_detailed_report(episode_stats: List[Dict], scene_stats: List[Dict], output_file: pathlib.Path):
    """
    保存详细报告到JSON文件
    
    Args:
        episode_stats: episode统计列表
        scene_stats: 场景统计列表
        output_file: 输出文件路径
    """
    # 移除原始数据以减小文件大小
    episode_stats_clean = []
    for stats in episode_stats:
        stats_clean = stats.copy()
        stats_clean.pop('pedestrian_data', None)
        episode_stats_clean.append(stats_clean)
    
    report = {
        "scene_statistics": scene_stats,
        "episode_statistics": episode_stats_clean,
    }
    
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n详细报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="分析行人视野数据")
    parser.add_argument("--data_folder", type=str, required=True,
                      help="数据根目录路径")
    parser.add_argument("--split", type=str, default="train",
                      help="数据集分割名称 (默认: train)")
    parser.add_argument("--output", type=str, default=None,
                      help="输出JSON报告文件路径 (可选)")
    
    args = parser.parse_args()
    
    data_folder = pathlib.Path(args.data_folder)
    
    if not data_folder.exists():
        print(f"错误: 数据文件夹不存在: {data_folder}")
        return
    
    print(f"开始分析数据...")
    print(f"数据文件夹: {data_folder}")
    print(f"数据分割: {args.split}")
    
    # 分析数据
    episode_stats, scene_stats = analyze_split(data_folder, args.split)
    
    # 打印统计信息
    print_statistics(episode_stats, scene_stats)
    
    # 保存详细报告
    if args.output:
        output_file = pathlib.Path(args.output)
        save_detailed_report(episode_stats, scene_stats, output_file)


if __name__ == "__main__":
    main()

