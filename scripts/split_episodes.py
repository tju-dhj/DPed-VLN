#!/usr/bin/env python3
"""
将content目录下的文件均分成多个子目录
用于多GPU并行数据采集，避免重复采集
"""

import shutil
from pathlib import Path
import argparse


def split_files(content_dir: Path, output_base_dir: Path, num_splits: int = 5):
    """
    将content目录中的所有文件均分到多个子目录
    
    Args:
        content_dir: 原始content目录路径
        output_base_dir: 输出基础目录
        num_splits: 划分的份数
    """
    print(f"======================================")
    print(f"文件均分工具")
    print(f"======================================")
    print(f"输入目录: {content_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"划分份数: {num_splits}")
    print(f"======================================\n")
    
    # 获取所有.json.gz文件
    all_files = sorted(content_dir.glob("*.json.gz"))
    
    if not all_files:
        print(f"❌ 错误: 在 {content_dir} 中没有找到.json.gz文件")
        return
    
    total_files = len(all_files)
    files_per_split = total_files // num_splits
    
    print(f"📁 找到 {total_files} 个文件")
    print(f"📊 每份约 {files_per_split} 个文件\n")
    
    # 划分并复制文件
    for split_id in range(num_splits):
        print(f"--- 创建 Split {split_id} ---")
        
        # 创建输出目录
        split_dir = output_base_dir / f"content_split_{split_id}"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算这个split应该包含的文件范围
        start_idx = split_id * files_per_split
        if split_id == num_splits - 1:
            # 最后一份包含所有剩余的文件
            end_idx = total_files
        else:
            end_idx = (split_id + 1) * files_per_split
        
        split_files = all_files[start_idx:end_idx]
        
        print(f"  文件索引: {start_idx} 到 {end_idx-1}")
        print(f"  文件数量: {len(split_files)}")
        
        # 复制文件
        for file_path in split_files:
            dest_path = split_dir / file_path.name
            shutil.copy2(file_path, dest_path)
        
        print(f"  ✅ Split {split_id} 完成\n")
    
    print("======================================")
    print("✅ 划分完成!")
    print("======================================")
    
    # 输出统计信息
    print("\n📊 统计信息:")
    for split_id in range(num_splits):
        split_dir = output_base_dir / f"content_split_{split_id}"
        split_files = list(split_dir.glob("*.json.gz"))
        print(f"  Split {split_id}: {len(split_files)} 个文件")
    
    print(f"\n💾 输出目录:")
    for split_id in range(num_splits):
        print(f"  content_split_{split_id}/")
    
    print("\n✅ 可以使用以下命令提交任务:")
    print(f"   sbatch collect_data_split.bash")


def main():
    parser = argparse.ArgumentParser(description='将content目录的文件均分到多个子目录')
    parser.add_argument('--content-dir', type=str,
                       default='/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/datasets/pointnav/social-hm3d/train/content',
                       help='输入content目录路径')
    parser.add_argument('--output-dir', type=str,
                       default='/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/datasets/pointnav/social-hm3d/train',
                       help='输出基础目录')
    parser.add_argument('--num-splits', type=int, default=5,
                       help='划分的份数')
    
    args = parser.parse_args()
    
    content_dir = Path(args.content_dir)
    output_dir = Path(args.output_dir)
    
    if not content_dir.exists():
        print(f"❌ 错误: 输入目录不存在: {content_dir}")
        return
    
    split_files(content_dir, output_dir, args.num_splits)


if __name__ == "__main__":
    main()

