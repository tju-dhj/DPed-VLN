#!/usr/bin/env python3

"""
测试传感器修复
"""

import sys
import os

# 添加路径
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab")

def test_sensor_import():
    """测试传感器导入"""
    
    print("测试传感器导入...")
    
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
        
        # 尝试加载配置
        try:
            config = cs.load("habitat.task.lab_sensors.main_oracle_shortest_path_sensor")
            print("✅ 传感器配置可用")
            print(f"   配置类型: {type(config)}")
        except Exception as e:
            print(f"❌ 传感器配置不可用: {e}")
            
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()

def test_habitat_import():
    """测试 Habitat 导入"""
    
    print("\n测试 Habitat 导入...")
    
    try:
        import habitat
        print("✅ 成功导入 habitat")
        
        from habitat.core.registry import registry
        print("✅ 成功导入 registry")
        
        # 检查所有注册的传感器
        sensors = registry._env_sensors
        print(f"✅ 注册的传感器数量: {len(sensors)}")
        
        if "MainOracleShortestPathSensor" in sensors:
            print("✅ MainOracleShortestPathSensor 已注册")
        else:
            print("❌ MainOracleShortestPathSensor 未注册")
            print(f"   已注册的传感器: {list(sensors.keys())}")
            
    except Exception as e:
        print(f"❌ Habitat 导入失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sensor_import()
    test_habitat_import()
