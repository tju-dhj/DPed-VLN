#!/usr/bin/env python3
"""
显示PyTorch模型文件的参数信息
"""

import torch
import os
from collections import OrderedDict

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def analyze_model(file_path):
    """分析模型文件并返回信息"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    print(f"\n{'='*80}")
    print(f"📁 文件路径: {file_path}")
    print(f"{'='*80}")
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    print(f"📊 文件大小: {format_size(file_size)}")
    
    try:
        # 加载模型
        print(f"\n🔄 正在加载模型...")
        # PyTorch 2.6+ 默认使用 weights_only=True，需要设置为 False 以加载包含 omegaconf 的模型
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # 检查checkpoint的类型
        if isinstance(checkpoint, dict):
            print(f"\n📦 Checkpoint类型: dict")
            print(f"🔑 Checkpoint键: {list(checkpoint.keys())}")
            
            # 尝试找到state_dict
            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\n✅ 找到 'state_dict' 键")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"\n✅ 找到 'model' 键")
            elif 'actor_critic' in checkpoint:
                state_dict = checkpoint['actor_critic']
                print(f"\n✅ 找到 'actor_critic' 键")
            else:
                # 检查是否直接是state_dict
                all_keys = list(checkpoint.keys())
                if all_keys and isinstance(checkpoint[all_keys[0]], torch.Tensor):
                    state_dict = checkpoint
                    print(f"\n✅ Checkpoint直接是state_dict")
                else:
                    print(f"\n⚠️  未找到标准的state_dict，显示所有键的内容:")
                    for key in checkpoint.keys():
                        if isinstance(checkpoint[key], dict):
                            print(f"  - {key}: dict with {len(checkpoint[key])} keys")
                        elif isinstance(checkpoint[key], torch.Tensor):
                            print(f"  - {key}: Tensor {checkpoint[key].shape}")
                        else:
                            print(f"  - {key}: {type(checkpoint[key])}")
            
            # 显示其他信息
            for key in ['config', 'optimizer', 'lr_scheduler', 'epoch', 'step', 'best_val']:
                if key in checkpoint:
                    value = checkpoint[key]
                    if isinstance(value, dict):
                        print(f"\n📋 {key}: dict with {len(value)} keys")
                    else:
                        print(f"\n📋 {key}: {value}")
            
            # 分析state_dict
            if state_dict is not None:
                print(f"\n{'='*80}")
                print(f"📊 State Dict 分析")
                print(f"{'='*80}")
                
                # 统计参数
                total_params = 0
                trainable_params = 0
                param_info = []
                
                for name, param in state_dict.items():
                    if isinstance(param, torch.Tensor):
                        num_params = param.numel()
                        total_params += num_params
                        if param.requires_grad if hasattr(param, 'requires_grad') else True:
                            trainable_params += num_params
                        
                        param_info.append({
                            'name': name,
                            'shape': tuple(param.shape),
                            'numel': num_params,
                            'dtype': str(param.dtype),
                            'requires_grad': param.requires_grad if hasattr(param, 'requires_grad') else None
                        })
                
                print(f"\n📈 参数统计:")
                print(f"  - 总参数数量: {len(param_info)}")
                print(f"  - 总参数量: {total_params:,} ({format_size(total_params * 4)})")
                print(f"  - 可训练参数量: {trainable_params:,} ({format_size(trainable_params * 4)})")
                
                # 按模块分组统计
                module_stats = OrderedDict()
                for info in param_info:
                    module_name = info['name'].split('.')[0]
                    if module_name not in module_stats:
                        module_stats[module_name] = {'count': 0, 'params': 0}
                    module_stats[module_name]['count'] += 1
                    module_stats[module_name]['params'] += info['numel']
                
                print(f"\n📦 模块统计 (前20个):")
                for i, (module, stats) in enumerate(list(module_stats.items())[:20]):
                    print(f"  {i+1:2d}. {module:30s} - {stats['count']:3d} 参数, {stats['params']:>12,} 元素")
                
                # 显示前30个参数详情
                print(f"\n🔍 参数详情 (前30个):")
                for i, info in enumerate(param_info[:30]):
                    grad_info = f", requires_grad={info['requires_grad']}" if info['requires_grad'] is not None else ""
                    print(f"  {i+1:2d}. {info['name']:60s}")
                    print(f"      Shape: {str(info['shape']):40s} | Elements: {info['numel']:>10,} | Dtype: {info['dtype']}{grad_info}")
                
                if len(param_info) > 30:
                    print(f"\n  ... 还有 {len(param_info) - 30} 个参数未显示")
                
                # 查找关键模块
                print(f"\n🔑 关键模块查找:")
                key_modules = ['visual_encoder', 'backbone', 'net', 'actor', 'critic', 'policy', 'rnn', 'lstm', 'gru']
                for key in key_modules:
                    matching = [info['name'] for info in param_info if key.lower() in info['name'].lower()]
                    if matching:
                        print(f"  - 包含 '{key}' 的参数: {len(matching)} 个")
                        for match in matching[:5]:
                            print(f"      {match}")
                        if len(matching) > 5:
                            print(f"      ... 还有 {len(matching) - 5} 个")
        else:
            print(f"\n⚠️  Checkpoint类型: {type(checkpoint)}")
            print(f"   内容: {checkpoint}")
            
    except Exception as e:
        print(f"\n❌ 加载模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    return checkpoint

def main():
    """主函数"""
    import sys
    
    # 模型文件路径
    base_dir = "."
    model1_path = "pretrained_model/falcon_noaux_25.pth"
    model2_path ="pretrained_model/pretrained_habitat3.pth"
    
    print("="*80)
    print("🔍 PyTorch 模型参数分析工具")
    print("="*80)
    
    # 分析第一个模型
    print(f"\n\n{'#'*80}")
    print(f"# 模型 1: falcon_noaux_25.pth")
    print(f"{'#'*80}")
    checkpoint1 = analyze_model(model1_path)
    
    # 分析第二个模型
    print(f"\n\n{'#'*80}")
    print(f"# 模型 2: pretrained_habitat3.pth")
    print(f"{'#'*80}")
    checkpoint2 = analyze_model(model2_path)
    
    # 对比两个模型
    if checkpoint1 is not None and checkpoint2 is not None:
        print(f"\n\n{'#'*80}")
        print(f"# 模型对比")
        print(f"{'#'*80}")
        
        # 获取state_dict
        def get_state_dict(checkpoint):
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    return checkpoint['state_dict']
                elif 'model' in checkpoint:
                    return checkpoint['model']
                elif 'actor_critic' in checkpoint:
                    return checkpoint['actor_critic']
                else:
                    # 检查是否直接是state_dict
                    all_keys = list(checkpoint.keys())
                    if all_keys and isinstance(checkpoint[all_keys[0]], torch.Tensor):
                        return checkpoint
            return None
        
        state_dict1 = get_state_dict(checkpoint1)
        state_dict2 = get_state_dict(checkpoint2)
        
        if state_dict1 and state_dict2:
            keys1 = set(state_dict1.keys())
            keys2 = set(state_dict2.keys())
            
            print(f"\n📊 键对比:")
            print(f"  - 模型1参数数量: {len(keys1)}")
            print(f"  - 模型2参数数量: {len(keys2)}")
            print(f"  - 共同参数: {len(keys1 & keys2)}")
            print(f"  - 仅在模型1: {len(keys1 - keys2)}")
            print(f"  - 仅在模型2: {len(keys2 - keys1)}")
            
            if keys1 - keys2:
                print(f"\n🔍 仅在模型1中的参数 (前10个):")
                for key in list(keys1 - keys2)[:10]:
                    print(f"    - {key}")
            
            if keys2 - keys1:
                print(f"\n🔍 仅在模型2中的参数 (前10个):")
                for key in list(keys2 - keys1)[:10]:
                    print(f"    - {key}")
            
            # 检查形状差异
            common_keys = keys1 & keys2
            shape_diff = []
            for key in common_keys:
                shape1 = tuple(state_dict1[key].shape) if isinstance(state_dict1[key], torch.Tensor) else None
                shape2 = tuple(state_dict2[key].shape) if isinstance(state_dict2[key], torch.Tensor) else None
                if shape1 != shape2:
                    shape_diff.append((key, shape1, shape2))
            
            if shape_diff:
                print(f"\n⚠️  形状不同的参数 (前10个):")
                for key, s1, s2 in shape_diff[:10]:
                    print(f"    - {key}")
                    print(f"        模型1: {s1}")
                    print(f"        模型2: {s2}")

if __name__ == "__main__":
    main()

