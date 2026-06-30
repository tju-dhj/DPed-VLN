#!/usr/bin/env python3

"""
测试修复是否有效

这个脚本用于测试配置和注册修复是否有效。
"""

import os
import sys
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_trainer_registration():
    """测试trainer注册"""
    print("测试trainer注册...")
    
    try:
        # 导入模块以确保注册
        from habitat_baselines.rl.ppo import ExpertDataCollectorV2
        
        from habitat_baselines.common.baseline_registry import baseline_registry
        
        # 检查trainer是否已注册
        if "expert_data_collector_v2" in baseline_registry.trainer_registry:
            print("✓ trainer已注册")
            return True
        else:
            print("✗ trainer未注册")
            print(f"可用的trainer: {list(baseline_registry.trainer_registry.keys())}")
            return False
            
    except Exception as e:
        print(f"✗ trainer注册测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

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
        
        # 检查expert_data_collection配置
        if hasattr(config, 'expert_data_collection'):
            expert_config = config.expert_data_collection
            print("✓ expert_data_collection配置存在")
            print(f"  - data_folder: {expert_config.get('data_folder', 'N/A')}")
            print(f"  - split: {expert_config.get('split', 'N/A')}")
            print(f"  - max_episodes: {expert_config.get('max_episodes', 'N/A')}")
            print(f"  - max_steps_per_episode: {expert_config.get('max_steps_per_episode', 'N/A')}")
            return True
        else:
            print("✗ expert_data_collection配置不存在")
            return False
        
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_creation():
    """测试trainer创建"""
    print("测试trainer创建...")
    
    try:
        from omegaconf import OmegaConf
        from habitat_baselines.common.baseline_registry import baseline_registry
        
        # 加载配置
        config_path = project_root / "habitat-baselines/habitat_baselines/config/social_nav_v2/expert_data_collection_v2.yaml"
        config = OmegaConf.load(config_path)
        
        # 获取trainer
        trainer_init = baseline_registry.get_trainer("expert_data_collector_v2")
        
        if trainer_init is not None:
            print("✓ trainer可以获取")
            return True
        else:
            print("✗ trainer无法获取")
            return False
            
    except Exception as e:
        print(f"✗ trainer创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试修复...")
    print("=" * 50)
    
    tests = [
        test_trainer_registration,
        test_config_loading,
        test_trainer_creation,
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
        print("✓ 所有测试通过！修复成功。")
        return 0
    else:
        print("✗ 部分测试失败，请检查修复。")
        return 1

if __name__ == "__main__":
    exit(main())
