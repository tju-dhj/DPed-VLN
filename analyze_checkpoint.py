#!/usr/bin/env python3
"""分析checkpoint结构，找出参数被跳过的原因"""

import sys
sys.path.insert(0, "habitat-lab")
sys.path.insert(0, "habitat-baselines")

import torch

ckpt_path = "pretrained_model/falcon_pretrained_25.pth"

print("="*80)
print("Checkpoint Structure Analysis")
print("="*80)

checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("\n1. Top-level structure:")
print(f"   Type: {type(checkpoint)}")
print(f"   Keys: {list(checkpoint.keys())}")

print("\n2. Analyzing each top-level key:")
for k in checkpoint.keys():
    v = checkpoint[k]
    print(f"\n  Key [{k}] (type: {type(k).__name__}):")
    print(f"    Value type: {type(v).__name__}")
    
    if isinstance(v, dict):
        print(f"    Dict size: {len(v)} items")
        # 检查是否是state_dict（包含tensor的dict）
        has_tensors = any(torch.is_tensor(val) for val in list(v.values())[:10])
        print(f"    Contains tensors: {has_tensors}")
        
        if has_tensors:
            print(f"    First 10 keys:")
            for i, (key, val) in enumerate(list(v.items())[:10]):
                if torch.is_tensor(val):
                    print(f"      [{i}] {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"      [{i}] {key}: {type(val)}")
            
            # 检查是否有critic相关参数
            critic_keys = [k for k in v.keys() if isinstance(k, str) and 'critic' in k.lower()]
            print(f"    Critic keys: {len(critic_keys)}")
            
            # 检查key的命名模式
            sample_keys = list(v.keys())[:20]
            print(f"    Sample keys (first 20): {sample_keys}")

print("\n3. Recommended state_dict location:")
# 尝试找到正确的state_dict
if isinstance(checkpoint, dict):
    for key in [0, 1, 'state_dict', 'model_state_dict', 'actor_critic']:
        if key in checkpoint:
            candidate = checkpoint[key]
            if isinstance(candidate, dict):
                has_tensors = any(torch.is_tensor(v) for v in list(candidate.values())[:10])
                if has_tensors:
                    print(f"   ✓ Found state_dict at checkpoint[{key}]")
                    print(f"     Size: {len(candidate)} keys")
                    
                    # 检查参数名前缀
                    sample_keys = list(candidate.keys())[:5]
                    print(f"     Sample keys: {sample_keys}")
                    
                    # 检查是否需要添加"net."前缀
                    needs_prefix = not any(k.startswith('net.') for k in sample_keys if isinstance(k, str))
                    print(f"     Needs 'net.' prefix: {needs_prefix}")
                    break

print("\n" + "="*80)

