#!/usr/bin/env python3

"""
测试专家数据采集器配置

这个脚本用于测试专家数据采集器的配置是否正确。
"""

import os
import sys
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_loading():
    """测试配置文件加载"""
    print("测试配置文件加载...")
    
    try:
        from omegaconf import OmegaConf
        
        # 加载配置文件
        config_path = project_root / "habitat-baselines/habitat_baselines/config/social_nav_v2/expert_data_collection_v2.yaml"
        
        if not config_path.exists():
            print(f"✗ 配置文件不存在: {config_path}")
            return False
        
        # 加载配置
        config = OmegaConf.load(config_path)
        
        # 检查必要的配置项
        required_keys = [
            "habitat_baselines.trainer_name",
            "habitat_baselines.expert_data_collection.data_folder",
            "habitat_baselines.expert_data_collection.split",
            "habitat_baselines.expert_data_collection.max_episodes",
            "habitat_baselines.expert_data_collection.max_steps_per_episode",
        ]
        
        for key in required_keys:
            if not OmegaConf.select(config, key):
                print(f"✗ 缺少配置项: {key}")
                return False
        
        print("✓ 配置文件加载成功")
        print(f"  - trainer_name: {config.habitat_baselines.trainer_name}")
        print(f"  - data_folder: {config.habitat_baselines.expert_data_collection.data_folder}")
        print(f"  - split: {config.habitat_baselines.expert_data_collection.split}")
        print(f"  - max_episodes: {config.habitat_baselines.expert_data_collection.max_episodes}")
        print(f"  - max_steps_per_episode: {config.habitat_baselines.expert_data_collection.max_steps_per_episode}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False

def test_trainer_registration():
    """测试trainer注册"""
    print("测试trainer注册...")
    
    try:
        from habitat_baselines.common.baseline_registry import baseline_registry
        
        # 检查trainer是否已注册
        if "expert_data_collector_v2" in baseline_registry.trainer_registry:
            print("✓ trainer已注册")
            return True
        else:
            print("✗ trainer未注册")
            return False
            
    except Exception as e:
        print(f"✗ trainer注册测试失败: {e}")
        return False

def test_import():
    """测试导入"""
    print("测试导入...")
    
    try:
        from habitat_baselines.rl.ppo.expert_data_collector_v2 import ExpertDataCollectorV2
        print("✓ 成功导入ExpertDataCollectorV2")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试专家数据采集器配置...")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_trainer_registration,
        test_import,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！配置正确。")
        return 0
    else:
        print("✗ 部分测试失败，请检查配置。")
        return 1

if __name__ == "__main__":
    exit(main())
