#!/usr/bin/env python3

"""
简单测试运行
"""

import sys
import os

# 添加路径
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab")

def test_simple_config():
    """测试简单配置"""
    
    print("测试简单配置...")
    
    try:
        # 导入必要的模块
        import falcon.additional_sensor  # noqa: F401
        print("✅ 传感器模块导入成功")
        
        # 测试配置加载
        from habitat.config.default import get_config
        config = get_config("habitat-baselines/habitat_baselines/config/social_nav_v2/collect_data_multi.yaml")
        print("✅ 配置加载成功")
        
        # 检查传感器配置
        if hasattr(config.habitat.task, 'lab_sensors'):
            print("✅ lab_sensors 配置存在")
            print(f"   lab_sensors: {config.habitat.task.lab_sensors}")
        else:
            print("❌ lab_sensors 配置不存在")
            
        # 检查 gym 配置
        if hasattr(config.habitat, 'gym'):
            print("✅ gym 配置存在")
            if hasattr(config.habitat.gym, 'obs_keys'):
                print(f"   obs_keys: {config.habitat.gym.obs_keys}")
                if 'agent_0_main_oracle_shortest_path_sensor' in config.habitat.gym.obs_keys:
                    print("✅ 传感器在 obs_keys 中")
                else:
                    print("❌ 传感器不在 obs_keys 中")
            else:
                print("❌ obs_keys 不存在")
        else:
            print("❌ gym 配置不存在")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_config()
