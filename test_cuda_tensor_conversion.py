#!/usr/bin/env python3

"""
测试CUDA张量转换逻辑
"""

import torch
import numpy as np

def test_cuda_tensor_conversion():
    """测试CUDA张量转换逻辑"""
    
    print("测试CUDA张量转换逻辑...")
    
    # 模拟传感器返回的CUDA张量
    cuda_tensor = torch.tensor([[2]], device='cuda:0')
    print(f"原始CUDA张量: {cuda_tensor}")
    print(f"张量类型: {type(cuda_tensor)}")
    print(f"张量设备: {cuda_tensor.device}")
    
    # 应用转换逻辑
    action = cuda_tensor
    
    # 处理CUDA张量，转换为CPU上的numpy数组
    if hasattr(action, 'cpu'):
        action_cpu = action.cpu().numpy()
        print(f"转换后numpy数组: {action_cpu}")
        print(f"numpy数组类型: {type(action_cpu)}")
    elif hasattr(action, 'detach'):
        action_cpu = action.detach().cpu().numpy()
    else:
        action_cpu = action
    
    # 确保action是标量值
    if hasattr(action_cpu, 'item'):
        action_cpu = action_cpu.item()
        print(f"标量值: {action_cpu}")
        print(f"标量类型: {type(action_cpu)}")
    elif isinstance(action_cpu, np.ndarray):
        action_cpu = action_cpu.item()
    
    # 测试numpy数组创建
    try:
        np_array = np.array([action_cpu])
        print(f"最终numpy数组: {np_array}")
        print(f"最终数组类型: {type(np_array)}")
        print("✅ 转换成功！")
    except Exception as e:
        print(f"❌ 转换失败: {e}")
    
    # 测试数据存储
    test_data = {
        'action': action_cpu,
        'step': 0
    }
    print(f"存储的数据: {test_data}")
    
    # 验证JSON序列化
    import json
    try:
        json_str = json.dumps(test_data)
        print(f"JSON序列化成功: {json_str}")
        print("✅ JSON序列化成功！")
    except Exception as e:
        print(f"❌ JSON序列化失败: {e}")

def test_cpu_tensor_conversion():
    """测试CPU张量转换逻辑"""
    
    print("\n测试CPU张量转换逻辑...")
    
    # 模拟CPU张量
    cpu_tensor = torch.tensor([[3]], device='cpu')
    print(f"原始CPU张量: {cpu_tensor}")
    
    action = cpu_tensor
    
    # 应用转换逻辑
    if hasattr(action, 'cpu'):
        action_cpu = action.cpu().numpy()
    elif hasattr(action, 'detach'):
        action_cpu = action.detach().cpu().numpy()
    else:
        action_cpu = action
    
    # 确保action是标量值
    if hasattr(action_cpu, 'item'):
        action_cpu = action_cpu.item()
    elif isinstance(action_cpu, np.ndarray):
        action_cpu = action_cpu.item()
    
    print(f"转换后标量: {action_cpu}")
    print(f"标量类型: {type(action_cpu)}")
    
    # 测试numpy数组创建
    np_array = np.array([action_cpu])
    print(f"最终numpy数组: {np_array}")
    print("✅ CPU张量转换成功！")

if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        test_cuda_tensor_conversion()
    else:
        print("CUDA不可用，跳过CUDA张量测试")
    
    test_cpu_tensor_conversion()
