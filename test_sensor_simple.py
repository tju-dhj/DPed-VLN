#!/usr/bin/env python3

"""
简单测试传感器文件
"""

import sys
import os

# 添加路径
sys.path.append('/share/home/u19666033/dhj/falcon_collect_data/Falcon-main')

def test_sensor_file():
    """测试传感器文件"""
    
    print("测试传感器文件...")
    
    try:
        # 直接读取传感器文件
        sensor_file = '/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/falcon/additional_sensor.py'
        
        if os.path.exists(sensor_file):
            print("✅ 传感器文件存在")
            
            # 检查文件内容
            with open(sensor_file, 'r') as f:
                content = f.read()
                
            # 检查关键组件
            if 'class MainOracleShortestPathSensor' in content:
                print("✅ MainOracleShortestPathSensor 类存在")
            else:
                print("❌ MainOracleShortestPathSensor 类不存在")
                
            if 'class MainOracleShortestPathSensorConfig' in content:
                print("✅ MainOracleShortestPathSensorConfig 类存在")
            else:
                print("❌ MainOracleShortestPathSensorConfig 类不存在")
                
            if '@registry.register_sensor(name="MainOracleShortestPathSensor")' in content:
                print("✅ 传感器注册装饰器存在")
            else:
                print("❌ 传感器注册装饰器不存在")
                
            if 'cs.store(' in content and 'main_oracle_shortest_path_sensor' in content:
                print("✅ 配置存储存在")
            else:
                print("❌ 配置存储不存在")
                
        else:
            print("❌ 传感器文件不存在")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

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
                
            if 'agent_0_main_oracle_shortest_path_sensor' in content:
                print("✅ 传感器引用存在")
            else:
                print("❌ 传感器引用不存在")
                
        else:
            print("❌ 配置文件不存在")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_sensor_file()
    test_config_file()
