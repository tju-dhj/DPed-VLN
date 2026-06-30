import os
import shutil
from pathlib import Path

def filter_and_copy_files():
    # 定义路径
    content_ori_path = Path("/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/datasets/pointnav/social-hm3d/train/content_ori")
    train_part1_path = Path("/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/collect_data/train_part_1")
    output_path = Path("/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/data/datasets/pointnav/social-hm3d/train/content_filtered")
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取content_ori中的所有.json.gz文件
    content_ori_files = list(content_ori_path.glob("*.json.gz"))
    print(f"Found {len(content_ori_files)} .json.gz files in content_ori")
    
    # 获取train_part_1中的所有.basis文件夹名称（去掉.basis后缀）
    train_part1_names = [f.stem for f in train_part1_path.glob("*.basis")]
    print(f"Found {len(train_part1_names)} .basis folders in train_part_1")
    
    # 筛选需要保留的文件
    files_to_keep = []
    for file_path in content_ori_files:
        file_stem = file_path.stem  # 去掉.json.gz后缀的文件名
        if file_stem not in train_part1_names:
            files_to_keep.append(file_path)
    
    print(f"Found {len(files_to_keep)} files to keep (not in train_part_1)")
    
    # 复制文件到新文件夹
    for file_path in files_to_keep:
        destination = output_path / file_path.name
        shutil.copy2(file_path, destination)
        print(f"Copied: {file_path.name}")
    
    print(f"All files copied to: {output_path}")

if __name__ == "__main__":
    filter_and_copy_files()
