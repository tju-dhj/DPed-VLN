#!/usr/bin/env python3
"""
测试CLIP多视角配置是否正确

用法:
    python test_clip_multi_view_config.py --config path/to/your/config.yaml
"""

import argparse
from omegaconf import OmegaConf


def test_clip_config(config_path):
    """测试CLIP多视角配置"""
    print("=" * 80)
    print("🔍 CLIP多视角配置检查器")
    print("=" * 80)
    
    # 加载配置
    try:
        config = OmegaConf.load(config_path)
        print(f"✅ 成功加载配置文件: {config_path}\n")
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return False
    
    # 检查backbone
    print("📋 检查1: Backbone配置")
    print("-" * 80)
    try:
        backbone = config.habitat_baselines.rl.ddppo.backbone
        print(f"   Backbone: {backbone}")
        
        if "clip" in backbone.lower():
            print(f"   ✅ 使用CLIP backbone，支持多视角配置")
        else:
            print(f"   ⚠️  未使用CLIP backbone，多视角配置不会生效")
    except Exception as e:
        print(f"   ❌ 无法读取backbone配置: {e}")
        return False
    
    # 检查obs_keys
    print("\n📋 检查2: 观察空间传感器 (habitat.gym.obs_keys)")
    print("-" * 80)
    try:
        obs_keys = config.habitat.gym.obs_keys
        
        # 查找RGB和深度传感器
        rgb_sensors = [k for k in obs_keys if 'rgb' in k.lower() and 'agent_0' in k]
        depth_sensors = [k for k in obs_keys if 'depth' in k.lower() and 'agent_0' in k]
        
        print(f"   找到的RGB传感器:")
        for sensor in rgb_sensors:
            print(f"      - {sensor}")
        
        print(f"\n   找到的深度传感器:")
        for sensor in depth_sensors:
            print(f"      - {sensor}")
        
        if len(rgb_sensors) == 0 and len(depth_sensors) == 0:
            print(f"   ❌ 未找到任何RGB或深度传感器！")
            return False
        else:
            print(f"\n   ✅ 共找到 {len(rgb_sensors)} 个RGB传感器和 {len(depth_sensors)} 个深度传感器")
    except Exception as e:
        print(f"   ❌ 无法读取obs_keys: {e}")
        return False
    
    # 检查clip_visual_sensors配置
    print("\n📋 检查3: CLIP传感器配置 (habitat_baselines.rl.ddppo.clip_visual_sensors)")
    print("-" * 80)
    
    has_clip_config = False
    try:
        clip_config = config.habitat_baselines.rl.ddppo.clip_visual_sensors
        has_clip_config = True
        
        # 读取配置
        rgb_keys = clip_config.get('rgb_keys', [])
        depth_keys = clip_config.get('depth_keys', [])
        fusion_mode = clip_config.get('fusion_mode', 'average')
        normalize = clip_config.get('normalize_before_fusion', True)
        
        print(f"   配置的RGB传感器键名:")
        for key in rgb_keys:
            full_key = f"agent_0_{key}"
            if full_key in obs_keys or key in obs_keys:
                print(f"      ✅ {key} (在obs_keys中找到)")
            else:
                print(f"      ❌ {key} (未在obs_keys中找到！)")
        
        print(f"\n   配置的深度传感器键名:")
        for key in depth_keys:
            full_key = f"agent_0_{key}"
            if full_key in obs_keys or key in obs_keys:
                print(f"      ✅ {key} (在obs_keys中找到)")
            else:
                print(f"      ❌ {key} (未在obs_keys中找到！)")
        
        print(f"\n   融合模式: {fusion_mode}")
        if fusion_mode == "average":
            print(f"      ✅ 平均融合 - 内存友好，推荐")
        elif fusion_mode == "concat":
            print(f"      ⚠️  拼接融合 - 参数量增加")
        elif fusion_mode == "attention":
            print(f"      ⚠️  注意力融合 - 计算量大")
        else:
            print(f"      ❌ 未知的融合模式: {fusion_mode}")
        
        print(f"\n   融合前归一化: {normalize}")
        if normalize:
            print(f"      ✅ 已启用（推荐）")
        else:
            print(f"      ⚠️  未启用")
        
        # 检查是否为多视角
        total_sensors = len(rgb_keys) + len(depth_keys)
        if total_sensors > 1:
            print(f"\n   ✅ 多视角配置 - 共 {total_sensors} 个传感器将被融合")
        else:
            print(f"\n   ℹ️  单视角配置 - 融合模式不会生效")
            
    except AttributeError:
        print(f"   ⚠️  未找到 clip_visual_sensors 配置")
        print(f"   ℹ️  将使用默认行为（自动查找第一个可用的RGB和深度传感器）")
    except Exception as e:
        print(f"   ❌ 读取clip_visual_sensors配置时出错: {e}")
        return False
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 配置检查总结")
    print("=" * 80)
    
    if has_clip_config:
        print("✅ CLIP多视角配置完整")
        print("✅ 可以开始训练了！")
        print("\n💡 提示: 运行训练时，请检查日志中是否有以下信息:")
        print("   [ResNetCLIPTextEncoder] 配置的RGB传感器: [...]")
        print("   [ResNetCLIPTextEncoder] 配置的Depth传感器: [...]")
        print("   [ResNetCLIPTextEncoder] 多视角融合模式: ...")
    else:
        print("⚠️  未配置clip_visual_sensors，将使用默认行为")
        print("💡 如需启用多视角融合，请参考 CLIP_MULTI_VIEW_CONFIG_GUIDE.md")
    
    print("=" * 80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="测试CLIP多视角配置是否正确"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径（如 habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train_v1.yaml）"
    )
    
    args = parser.parse_args()
    
    success = test_clip_config(args.config)
    
    if success:
        print("\n✅ 配置检查完成")
        return 0
    else:
        print("\n❌ 配置检查失败，请修复上述错误")
        return 1


if __name__ == "__main__":
    exit(main())

