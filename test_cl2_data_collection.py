#!/usr/bin/env python3

"""
测试 cl_2.py 中的数据收集功能
"""

import numpy as np
import json
import pathlib
import os
from PIL import Image

def test_save_to_disk():
    """测试 save_to_disk 函数"""
    
    # 创建测试数据
    rgb_data = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)
    depth_data = np.random.rand(5, 224, 224, 1).astype(np.float32)
    human_num_data = np.array([2, 3, 1, 4, 2])
    action_data = [1, 2, 0, 1, 3]
    
    ep_id = "test_episode_001"
    split = "test"
    data_folder = "/tmp/test_data_collection"
    
    # 导入并测试保存函数
    import sys
    sys.path.append('/share/home/u19666033/dhj/falcon_collect_data/Falcon-main')
    
    from habitat_baselines.habitat_baselines.rl.ppo.cl_2 import save_to_disk
    
    # 执行保存
    save_to_disk(
        rgb_data,
        depth_data,
        human_num_data,
        action_data,
        ep_id,
        split=split,
        data_folder=data_folder,
        merge_ep=True
    )
    
    # 验证文件是否创建
    data_root = pathlib.Path(data_folder) / split / ep_id
    
    # 检查目录结构
    assert (data_root / "rgb").exists(), "RGB directory not created"
    assert (data_root / "depth").exists(), "Depth directory not created"
    assert (data_root / "human_num").exists(), "Human num directory not created"
    assert (data_root / "action").exists(), "Action directory not created"
    
    # 检查文件数量
    rgb_files = list((data_root / "rgb").glob("*.jpg"))
    depth_files = list((data_root / "depth").glob("*.png"))
    
    assert len(rgb_files) == 5, f"Expected 5 RGB files, got {len(rgb_files)}"
    assert len(depth_files) == 5, f"Expected 5 depth files, got {len(depth_files)}"
    
    # 检查JSON文件
    human_num_file = data_root / "human_num" / "0.json"
    action_file = data_root / "action" / "0.json"
    
    assert human_num_file.exists(), "Human num JSON file not created"
    assert action_file.exists(), "Action JSON file not created"
    
    # 验证JSON内容
    with open(human_num_file, 'r') as f:
        saved_human_num = json.load(f)
    assert saved_human_num == human_num_data.tolist(), "Human num data mismatch"
    
    with open(action_file, 'r') as f:
        saved_actions = json.load(f)
    assert saved_actions == action_data, "Action data mismatch"
    
    print("✅ All tests passed!")
    print(f"Data saved to: {data_root}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree(data_folder)
    print("🧹 Test files cleaned up")

if __name__ == "__main__":
    test_save_to_disk()
