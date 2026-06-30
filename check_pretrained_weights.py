#!/usr/bin/env python3
"""
检查预训练权重文件并验证其内容
"""

import os
import sys
import torch
from pathlib import Path

def check_pretrained_weight(weight_path):
    """检查预训练权重文件"""
    print("="*80)
    print("检查预训练权重文件")
    print("="*80)
    
    # 检查文件是否存在
    print(f"\n路径: {weight_path}")
    
    if not os.path.exists(weight_path):
        print(f"❌ 文件不存在！")
        
        # 尝试查找类似的文件
        parent_dir = os.path.dirname(weight_path)
        if os.path.exists(parent_dir):
            print(f"\n在目录 {parent_dir} 中查找 .pth 文件...")
            pth_files = list(Path(parent_dir).glob("*.pth"))
            if pth_files:
                print(f"找到 {len(pth_files)} 个 .pth 文件:")
                for f in pth_files:
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  - {f.name} ({size_mb:.1f} MB)")
            else:
                print("  未找到任何 .pth 文件")
        else:
            print(f"❌ 父目录也不存在: {parent_dir}")
        
        # 尝试在其他可能的位置查找
        possible_dirs = [
            "pretrained_model",
            "data/pretrained",
            "pretrained_model",
        ]
        
        print("\n尝试在其他可能的位置查找...")
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                pth_files = list(Path(dir_path).glob("**/*.pth"))
                if pth_files:
                    print(f"\n在 {dir_path} 找到 .pth 文件:")
                    for f in pth_files[:10]:  # 只显示前10个
                        size_mb = f.stat().st_size / (1024*1024)
                        print(f"  - {f} ({size_mb:.1f} MB)")
        
        return False
    
    # 文件存在，检查内容
    file_size = os.path.getsize(weight_path) / (1024*1024)
    print(f"✓ 文件存在")
    print(f"  大小: {file_size:.2f} MB")
    
    # 尝试加载
    print("\n正在加载权重文件...")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        print("✓ 成功加载")
        
        # 检查内容
        if isinstance(checkpoint, dict):
            print(f"\n字典内容:")
            for key in checkpoint.keys():
                print(f"  - {key}")
                if key in ['state_dict', 'model_state_dict', 'actor_critic']:
                    state_dict = checkpoint[key]
                    if isinstance(state_dict, dict):
                        print(f"    包含 {len(state_dict)} 个键")
                        # 显示前10个键
                        for i, k in enumerate(list(state_dict.keys())[:10]):
                            v = state_dict[k]
                            if torch.is_tensor(v):
                                print(f"      [{i}] {k}: {v.shape}, {v.dtype}")
                        if len(state_dict) > 10:
                            print(f"      ... 还有 {len(state_dict)-10} 个参数")
        
        # 检查是否有critic相关参数
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'actor_critic' in checkpoint:
            state_dict = checkpoint['actor_critic']
        else:
            state_dict = checkpoint
        
        # 统计参数
        if isinstance(state_dict, dict):
            total_params = sum(p.numel() for p in state_dict.values() if torch.is_tensor(p))
            print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
            
            # 检查是否有actor和critic分离
            actor_keys = [k for k in state_dict.keys() if not k.startswith('critic')]
            critic_keys = [k for k in state_dict.keys() if k.startswith('critic')]
            
            print(f"\nActor相关参数: {len(actor_keys)}")
            print(f"Critic相关参数: {len(critic_keys)}")
            
            if critic_keys:
                print("\n⚠️  包含critic参数，训练时会被过滤掉")
                print(f"   Critic参数示例: {critic_keys[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 从配置文件中读取路径
    config_path = "habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_direct_il_train_v2.yaml"
    
    weight_path = "pretrained_model/falcon_pretrained_25.pth"
    
    print(f"配置文件: {config_path}")
    
    if os.path.exists(config_path):
        print("✓ 配置文件存在\n")
        # 可以解析yaml获取实际路径，这里先用硬编码
    else:
        print("❌ 配置文件不存在\n")
    
    success = check_pretrained_weight(weight_path)
    
    print("\n" + "="*80)
    if success:
        print("✓ 预训练权重文件检查通过")
        print("\n建议:")
        print("  1. 检查训练日志确认权重是否真的被加载")
        print("  2. 如果日志显示'No pretrained weights loaded'，检查代码中的加载逻辑")
    else:
        print("❌ 预训练权重文件检查失败")
        print("\n建议:")
        print("  1. 更新配置文件中的pretrained_weights路径")
        print("  2. 或者设置 pretrained: False 从头训练")
        print("  3. 或者从正确的位置复制权重文件到配置的路径")
    print("="*80)


if __name__ == "__main__":
    main()

