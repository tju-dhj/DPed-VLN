#!/usr/bin/env python3

"""
测试新的传感器配置
"""

import sys
import os

# 添加路径
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab")

def test_new_sensor_config():
    """测试新的传感器配置"""
    
    print("测试新的传感器配置...")
    
    try:
        # 导入传感器模块
        import falcon.additional_sensor
        print("✅ 成功导入 falcon.additional_sensor")
        
        # 检查传感器类
        from falcon.additional_sensor import MainOracleShortestPathSensor
        print("✅ 成功导入 MainOracleShortestPathSensor")
        
        # 检查配置类
        from falcon.additional_sensor import MainOracleShortestPathSensorConfig
        print("✅ 成功导入 MainOracleShortestPathSensorConfig")
        
        # 检查传感器注册
        from habitat.core.registry import registry
        sensor_class = registry.get_sensor("MainOracleShortestPathSensor")
        if sensor_class:
            print("✅ 传感器已注册到 registry")
            print(f"   传感器类: {sensor_class}")
        else:
            print("❌ 传感器未注册到 registry")
            
        # 检查配置存储
        from hydra.core.config_store import ConfigStore
        cs = ConfigStore.instance()
        
        # 尝试加载新的配置
        try:
            config = cs.load("habitat.task.lab_sensors.oracle_action_sensor")
            print("✅ 新的传感器配置可用")
            print(f"   配置类型: {type(config)}")
        except Exception as e:
            print(f"❌ 新的传感器配置不可用: {e}")
            
        # 检查旧的配置是否还存在
        try:
            old_config = cs.load("habitat.task.lab_sensors.main_oracle_shortest_path_sensor")
            print("❌ 旧的传感器配置仍然存在")
        except Exception as e:
            print("✅ 旧的传感器配置已移除")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_config_file():
    """测试配置文件"""
    
    print("\n测试配置文件...")
    
    try:
        config_file = '/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines/habitat_baselines/config/social_nav_v2/collect_data_multi.yaml'
        
        if os.path.exists(config_file):
            print("✅ 配置文件存在")
            
            # 检查文件内容
            with open(config_file, 'r') as f:
                content = f.read()
                
            if 'agent_0_oracle_action_sensor' in content:
                print("✅ 新的传感器引用存在")
            else:
                print("❌ 新的传感器引用不存在")
                
            if 'agent_0_main_oracle_shortest_path_sensor' in content:
                print("❌ 旧的传感器引用仍然存在")
            else:
                print("✅ 旧的传感器引用已移除")
                
        else:
            print("❌ 配置文件不存在")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_new_sensor_config()
    test_config_file()
