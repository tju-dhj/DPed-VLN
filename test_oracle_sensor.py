#!/usr/bin/env python3

import sys
import os
sys.path.append('/share/home/u19666033/dhj/falcon_collect_data/Falcon-main')

# 导入必要的模块
from falcon.additional_sensor import OracleShortestPathSensor
from habitat.core.registry import registry
import numpy as np

def test_oracle_sensor():
    """测试Oracle传感器是否能正常注册和初始化"""
    try:
        # 检查传感器是否已注册
        if "OracleShortestPathSensor" in registry._registry["sensor"]:
            print("✓ OracleShortestPathSensor 已成功注册")
        else:
            print("✗ OracleShortestPathSensor 未注册")
            return False
            
        # 检查传感器配置
        sensor_config = registry._registry["sensor"]["OracleShortestPathSensor"]
        print(f"✓ 传感器配置: {sensor_config}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("测试Oracle传感器注册...")
    success = test_oracle_sensor()
    if success:
        print("Oracle传感器测试通过！")
    else:
        print("Oracle传感器测试失败！")









