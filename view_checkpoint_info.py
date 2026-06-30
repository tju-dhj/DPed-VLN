#!/usr/bin/env python3
"""
查看评估checkpoint信息的工具脚本
"""

import json
import os
import sys
from pathlib import Path


def view_checkpoint_info(checkpoint_path):
    """查看checkpoint文件的详细信息"""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return
    
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        print("=" * 70)
        print(f"Checkpoint文件: {checkpoint_path}")
        print("=" * 70)
        
        # 基本统计
        stats_episodes = data.get('stats_episodes', {})
        ep_eval_count = data.get('ep_eval_count', {})
        actions_record = data.get('actions_record', {})
        
        print(f"\n📊 基本统计:")
        print(f"  - 已完成episodes总数: {len(stats_episodes)}")
        print(f"  - 唯一episodes数量: {len(ep_eval_count)}")
        print(f"  - 动作记录数量: {len(actions_record)}")
        
        # 文件大小
        file_size = os.path.getsize(checkpoint_path)
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.2f} MB"
        print(f"  - 文件大小: {size_str}")
        
        # 评估统计
        if stats_episodes:
            print(f"\n📈 评估统计:")
            
            # 收集所有统计指标
            all_metrics = {}
            for stats in stats_episodes.values():
                for key, value in stats.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
            
            # 计算平均值
            for metric, values in sorted(all_metrics.items()):
                avg_value = sum(values) / len(values)
                print(f"  - 平均 {metric}: {avg_value:.4f}")
        
        # Episodes详情
        if ep_eval_count:
            print(f"\n📝 Episodes详情:")
            print(f"  {'Scene ID':<40} {'Episode ID':<20} {'评估次数':<10}")
            print(f"  {'-' * 70}")
            
            # 只显示前10个
            count = 0
            for (scene_id, episode_id), eval_count in sorted(ep_eval_count.items()):
                # 截断过长的scene_id
                short_scene_id = scene_id.split('/')[-1] if '/' in scene_id else scene_id
                if len(short_scene_id) > 40:
                    short_scene_id = short_scene_id[:37] + "..."
                
                print(f"  {short_scene_id:<40} {episode_id:<20} {eval_count:<10}")
                count += 1
                if count >= 10:
                    remaining = len(ep_eval_count) - 10
                    if remaining > 0:
                        print(f"  ... 还有 {remaining} 个episodes")
                    break
        
        # 动作统计
        if actions_record:
            print(f"\n🎮 动作记录:")
            total_actions = sum(len(actions) for actions in actions_record.values())
            avg_actions = total_actions / len(actions_record) if actions_record else 0
            print(f"  - 总动作数: {total_actions}")
            print(f"  - 平均每个episode的动作数: {avg_actions:.1f}")
        
        print("\n" + "=" * 70)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")


def list_all_checkpoints(checkpoint_dir):
    """列出所有checkpoint文件"""
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint目录不存在: {checkpoint_dir}")
        return []
    
    checkpoint_files = sorted(Path(checkpoint_dir).glob("eval_progress_ckpt_*.json"))
    
    if not checkpoint_files:
        print(f"ℹ️  在 {checkpoint_dir} 中没有找到checkpoint文件")
        return []
    
    print("=" * 70)
    print(f"找到 {len(checkpoint_files)} 个checkpoint文件:")
    print("=" * 70)
    
    for i, ckpt_file in enumerate(checkpoint_files, 1):
        file_size = os.path.getsize(ckpt_file)
        if file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.2f} MB"
        
        print(f"{i}. {ckpt_file.name:<50} ({size_str})")
    
    print("=" * 70)
    return checkpoint_files


def main():
    """主函数"""
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  1. 查看特定checkpoint文件:")
        print("     python view_checkpoint_info.py <checkpoint_file_path>")
        print()
        print("  2. 列出所有checkpoint文件:")
        print("     python view_checkpoint_info.py <checkpoint_directory>")
        print()
        print("示例:")
        print("  python view_checkpoint_info.py data/checkpoints/eval_checkpoints/eval_progress_ckpt_100.json")
        print("  python view_checkpoint_info.py data/checkpoints/eval_checkpoints/")
        return
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # 查看单个文件
        view_checkpoint_info(path)
    elif os.path.isdir(path):
        # 列出目录中的所有checkpoint
        checkpoint_files = list_all_checkpoints(path)
        
        if checkpoint_files:
            print("\n选择要查看的checkpoint文件编号（或按Enter跳过）:")
            try:
                choice = input("> ").strip()
                if choice:
                    idx = int(choice) - 1
                    if 0 <= idx < len(checkpoint_files):
                        print()
                        view_checkpoint_info(str(checkpoint_files[idx]))
                    else:
                        print("❌ 无效的选择")
            except (ValueError, KeyboardInterrupt):
                print("\n已取消")
    else:
        print(f"❌ 路径不存在: {path}")


if __name__ == "__main__":
    main()

