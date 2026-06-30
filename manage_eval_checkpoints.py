#!/usr/bin/env python3
"""
管理评估checkpoint的工具脚本
支持查看、清理、备份多个配置的checkpoint
"""

import json
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime


def list_all_configs(checkpoint_base_dir):
    """列出所有配置的checkpoint"""
    eval_checkpoints_dir = os.path.join(checkpoint_base_dir, "eval_checkpoints")
    
    if not os.path.exists(eval_checkpoints_dir):
        print(f"❌ Checkpoint目录不存在: {eval_checkpoints_dir}")
        return []
    
    # 获取所有配置目录
    config_dirs = [d for d in Path(eval_checkpoints_dir).iterdir() if d.is_dir()]
    
    if not config_dirs:
        print(f"ℹ️  没有找到任何配置的checkpoint")
        return []
    
    print("=" * 80)
    print(f"找到 {len(config_dirs)} 个配置的checkpoint:")
    print("=" * 80)
    
    config_info = []
    for i, config_dir in enumerate(sorted(config_dirs), 1):
        config_name = config_dir.name
        
        # 统计checkpoint文件
        checkpoint_files = list(config_dir.glob("eval_progress_ckpt_*.json"))
        
        if checkpoint_files:
            # 计算总大小
            total_size = sum(f.stat().st_size for f in checkpoint_files)
            if total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.2f} KB"
            else:
                size_str = f"{total_size / (1024 * 1024):.2f} MB"
            
            # 获取最新的checkpoint信息
            latest_ckpt = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            try:
                with open(latest_ckpt, 'r') as f:
                    data = json.load(f)
                    completed_eps = len(data.get('stats_episodes', {}))
            except:
                completed_eps = "?"
            
            print(f"{i}. {config_name:<40}")
            print(f"   Checkpoints: {len(checkpoint_files)} 个, 总大小: {size_str}")
            print(f"   已完成episodes: {completed_eps}")
            print(f"   最新更新: {datetime.fromtimestamp(latest_ckpt.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            config_info.append({
                'name': config_name,
                'dir': config_dir,
                'files': checkpoint_files,
                'completed': completed_eps
            })
        else:
            print(f"{i}. {config_name:<40} (空目录)")
            print()
    
    print("=" * 80)
    return config_info


def view_config_detail(config_dir):
    """查看特定配置的详细信息"""
    checkpoint_files = list(Path(config_dir).glob("eval_progress_ckpt_*.json"))
    
    if not checkpoint_files:
        print(f"❌ 在 {config_dir} 中没有找到checkpoint文件")
        return
    
    print("=" * 80)
    print(f"配置: {Path(config_dir).name}")
    print("=" * 80)
    
    for ckpt_file in sorted(checkpoint_files):
        try:
            with open(ckpt_file, 'r') as f:
                data = json.load(f)
            
            stats_episodes = data.get('stats_episodes', {})
            ep_eval_count = data.get('ep_eval_count', {})
            
            print(f"\nCheckpoint: {ckpt_file.name}")
            print(f"  - 已完成episodes: {len(stats_episodes)}")
            print(f"  - 唯一episodes: {len(ep_eval_count)}")
            
            # 计算平均指标
            if stats_episodes:
                all_metrics = {}
                for stats in stats_episodes.values():
                    for key, value in stats.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)
                
                print(f"  - 平均指标:")
                for metric, values in sorted(all_metrics.items()):
                    avg_value = sum(values) / len(values)
                    print(f"    • {metric}: {avg_value:.4f}")
        
        except Exception as e:
            print(f"\n❌ 读取 {ckpt_file.name} 失败: {e}")
    
    print("=" * 80)


def clean_config(config_dir, confirm=True):
    """清理特定配置的checkpoint"""
    if confirm:
        response = input(f"⚠️  确定要删除 {Path(config_dir).name} 的所有checkpoint吗? (yes/no): ")
        if response.lower() != 'yes':
            print("已取消")
            return
    
    try:
        shutil.rmtree(config_dir)
        print(f"✅ 已删除: {config_dir}")
    except Exception as e:
        print(f"❌ 删除失败: {e}")


def backup_config(config_dir, backup_base_dir):
    """备份特定配置的checkpoint"""
    config_name = Path(config_dir).name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(backup_base_dir, f"{config_name}_backup_{timestamp}")
    
    try:
        shutil.copytree(config_dir, backup_dir)
        print(f"✅ 已备份到: {backup_dir}")
    except Exception as e:
        print(f"❌ 备份失败: {e}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("评估Checkpoint管理工具")
        print("\n使用方法:")
        print("  1. 列出所有配置:")
        print("     python manage_eval_checkpoints.py list [checkpoint_base_dir]")
        print()
        print("  2. 查看特定配置:")
        print("     python manage_eval_checkpoints.py view <config_name> [checkpoint_base_dir]")
        print()
        print("  3. 清理特定配置:")
        print("     python manage_eval_checkpoints.py clean <config_name> [checkpoint_base_dir]")
        print()
        print("  4. 备份特定配置:")
        print("     python manage_eval_checkpoints.py backup <config_name> [checkpoint_base_dir] [backup_dir]")
        print()
        print("  5. 清理所有空目录:")
        print("     python manage_eval_checkpoints.py clean-empty [checkpoint_base_dir]")
        print()
        print("示例:")
        print("  python manage_eval_checkpoints.py list data/checkpoints")
        print("  python manage_eval_checkpoints.py view il_v3_eval_v1_val_v1 data/checkpoints")
        print("  python manage_eval_checkpoints.py clean il_v3_eval_v1_val_v1 data/checkpoints")
        return
    
    command = sys.argv[1]
    checkpoint_base_dir = sys.argv[2] if len(sys.argv) > 2 else "data/checkpoints"
    
    if command == "list":
        list_all_configs(checkpoint_base_dir)
    
    elif command == "view":
        if len(sys.argv) < 3:
            print("❌ 请指定配置名称")
            return
        config_name = sys.argv[2]
        checkpoint_base_dir = sys.argv[3] if len(sys.argv) > 3 else "data/checkpoints"
        config_dir = os.path.join(checkpoint_base_dir, "eval_checkpoints", config_name)
        
        if not os.path.exists(config_dir):
            print(f"❌ 配置不存在: {config_name}")
            print("\n可用的配置:")
            list_all_configs(checkpoint_base_dir)
            return
        
        view_config_detail(config_dir)
    
    elif command == "clean":
        if len(sys.argv) < 3:
            print("❌ 请指定配置名称")
            return
        config_name = sys.argv[2]
        checkpoint_base_dir = sys.argv[3] if len(sys.argv) > 3 else "data/checkpoints"
        config_dir = os.path.join(checkpoint_base_dir, "eval_checkpoints", config_name)
        
        if not os.path.exists(config_dir):
            print(f"❌ 配置不存在: {config_name}")
            return
        
        clean_config(config_dir)
    
    elif command == "backup":
        if len(sys.argv) < 3:
            print("❌ 请指定配置名称")
            return
        config_name = sys.argv[2]
        checkpoint_base_dir = sys.argv[3] if len(sys.argv) > 3 else "data/checkpoints"
        backup_base_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(checkpoint_base_dir, "backups")
        config_dir = os.path.join(checkpoint_base_dir, "eval_checkpoints", config_name)
        
        if not os.path.exists(config_dir):
            print(f"❌ 配置不存在: {config_name}")
            return
        
        os.makedirs(backup_base_dir, exist_ok=True)
        backup_config(config_dir, backup_base_dir)
    
    elif command == "clean-empty":
        eval_checkpoints_dir = os.path.join(checkpoint_base_dir, "eval_checkpoints")
        if not os.path.exists(eval_checkpoints_dir):
            print(f"❌ Checkpoint目录不存在: {eval_checkpoints_dir}")
            return
        
        empty_dirs = []
        for config_dir in Path(eval_checkpoints_dir).iterdir():
            if config_dir.is_dir():
                checkpoint_files = list(config_dir.glob("eval_progress_ckpt_*.json"))
                if not checkpoint_files:
                    empty_dirs.append(config_dir)
        
        if not empty_dirs:
            print("✅ 没有找到空目录")
            return
        
        print(f"找到 {len(empty_dirs)} 个空目录:")
        for d in empty_dirs:
            print(f"  - {d.name}")
        
        response = input("\n确定要删除这些空目录吗? (yes/no): ")
        if response.lower() == 'yes':
            for d in empty_dirs:
                shutil.rmtree(d)
                print(f"✅ 已删除: {d.name}")
        else:
            print("已取消")
    
    else:
        print(f"❌ 未知命令: {command}")
        print("使用 'python manage_eval_checkpoints.py' 查看帮助")


if __name__ == "__main__":
    main()

