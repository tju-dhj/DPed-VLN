#!/usr/bin/env python3

"""
测试多环境修复
"""

import sys
import os
import numpy as np
import torch

# 添加路径
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab")

def test_convert_to_scalar():
    """测试标量转换函数"""
    
    print("测试标量转换函数...")
    
    # 导入数据采集器
    from habitat_baselines.rl.ppo.expert_data_collector_v3 import ExpertDataCollectorV3
    
    # 创建实例
    collector = ExpertDataCollectorV3()
    
    # 测试各种输入类型
    test_cases = [
        # 单个值
        torch.tensor([2], device='cuda:0'),
        torch.tensor([2], device='cpu'),
        np.array([2]),
        np.array([2, 3, 4]),  # 数组
        torch.tensor([[2]], device='cuda:0'),  # 2D张量
        torch.tensor([[2, 3]], device='cuda:0'),  # 2D数组
        2,  # 标量
        [2],  # 列表
    ]
    
    for i, test_value in enumerate(test_cases):
        try:
            result = collector._convert_to_scalar(test_value)
            print(f"测试 {i+1}: {type(test_value)} -> {result} ✅")
        except Exception as e:
            print(f"测试 {i+1}: {type(test_value)} -> 错误: {e} ❌")

def test_array_handling():
    """测试数组处理"""
    
    print("\n测试数组处理...")
    
    # 模拟多环境情况
    test_arrays = [
        np.array([2]),           # 单元素数组
        np.array([2, 3, 4]),     # 多元素数组
        np.array([[2]]),         # 2D单元素数组
        np.array([[2, 3]]),      # 2D多元素数组
        torch.tensor([2]),       # 单元素张量
        torch.tensor([2, 3, 4]), # 多元素张量
        torch.tensor([[2]]),     # 2D单元素张量
        torch.tensor([[2, 3]]),  # 2D多元素张量
    ]
    
    for i, test_array in enumerate(test_arrays):
        try:
            if hasattr(test_array, 'cpu'):
                array_np = test_array.cpu().numpy()
            else:
                array_np = test_array
            
            if array_np.size == 1:
                result = int(array_np.item())
            elif len(array_np) > 0:
                result = int(array_np[0])
            else:
                result = 0
                
            print(f"数组 {i+1}: {test_array} -> {result} ✅")
        except Exception as e:
            print(f"数组 {i+1}: {test_array} -> 错误: {e} ❌")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_convert_to_scalar()
        test_array_handling()
    else:
        print("CUDA不可用，跳过CUDA测试")
        test_array_handling()
