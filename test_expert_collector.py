#!/usr/bin/env python3

"""
专家数据采集器测试脚本

这个脚本用于测试专家数据采集器的基本功能。
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加项目路径到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_data_saving():
    """测试数据保存功能"""
    print("测试数据保存功能...")
    
    # 导入数据保存函数
    from habitat_baselines.rl.ppo.expert_data_collector import save_to_disk
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试数据
        import numpy as np
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640, 1).astype(np.float32)
        action = [1, 2, 3]  # 前进、左转、右转
        ep_id = "test_ep_0"
        step_idx = 0
        
        # 保存数据
        save_to_disk(
            rgb=rgb,
            depth=depth,
            action=action,
            ep_id=ep_id,
            step_idx=step_idx,
            split="test",
            data_folder=temp_dir,
        )
        
        # 检查文件是否创建
        data_root = Path(temp_dir) / "test" / ep_id
        assert (data_root / "rgb" / f"{step_idx}_0.jpg").exists(), "RGB文件未创建"
        assert (data_root / "depth" / f"{step_idx}_0.png").exists(), "深度文件未创建"
        assert (data_root / "action" / f"{step_idx}.json").exists(), "动作文件未创建"
        
        print("✓ 数据保存功能测试通过")

def test_advanced_data_saving():
    """测试高级数据保存功能"""
    print("测试高级数据保存功能...")
    
    # 导入高级数据保存函数
    from habitat_baselines.rl.ppo.advanced_expert_collector import save_expert_data
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试数据
        import numpy as np
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = np.random.rand(480, 640, 1).astype(np.float32)
        action = 1  # 前进
        ep_id = "test_ep_0"
        step_idx = 0
        additional_data = {
            "episode_id": "0",
            "step": 0,
            "agent_position": [1.0, 0.0, 2.0],
            "goal_position": [5.0, 0.0, 3.0],
            "pointgoal": [3.0, 0.5],
            "human_num": 2
        }
        
        # 保存数据
        save_expert_data(
            rgb=rgb,
            depth=depth,
            action=action,
            ep_id=ep_id,
            step_idx=step_idx,
            split="test",
            data_folder=temp_dir,
            additional_data=additional_data,
        )
        
        # 检查文件是否创建
        data_root = Path(temp_dir) / "test" / ep_id
        assert (data_root / "rgb" / f"{step_idx}_0.jpg").exists(), "RGB文件未创建"
        assert (data_root / "depth" / f"{step_idx}_0.png").exists(), "深度文件未创建"
        assert (data_root / "action" / f"{step_idx}.json").exists(), "动作文件未创建"
        
        # 检查动作文件内容
        import json
        with open(data_root / "action" / f"{step_idx}.json", "r") as f:
            action_data = json.load(f)
        
        assert action_data["action"] == action, "动作数据不正确"
        assert action_data["episode_id"] == "0", "episode_id不正确"
        assert action_data["human_num"] == 2, "人类数量不正确"
        
        print("✓ 高级数据保存功能测试通过")

def test_action_computation():
    """测试动作计算功能"""
    print("测试动作计算功能...")
    
    # 导入动作计算函数
    from habitat_baselines.rl.ppo.advanced_expert_collector import AdvancedExpertCollector
    
    # 创建测试实例
    class MockConfig:
        data_folder = "test_data"
        split = "test"
        max_episodes = 10
        max_steps_per_episode = 100
    
    collector = AdvancedExpertCollector(MockConfig())
    
    # 测试点目标动作计算
    pointgoal = np.array([3.0, 0.5])  # 距离3米，角度0.5弧度
    action = collector._compute_action_from_pointgoal(pointgoal)
    assert action in [0, 1, 2, 3], f"无效动作: {action}"
    
    # 测试Oracle路径动作计算
    oracle_path = np.array([
        [0.0, 0.0, 0.0],  # 起点
        [1.0, 0.0, 0.0],  # 下一个点
        [2.0, 0.0, 0.0]   # 再下一个点
    ])
    
    # 模拟环境调用
    class MockEnv:
        def call_at(self, env_idx, method):
            if method == "get_agent_state":
                class MockAgentState:
                    position = np.array([0.0, 0.0, 0.0])
                    rotation = type('obj', (object,), {'yaw': 0.0})()
                return MockAgentState()
            return None
    
    collector.envs = MockEnv()
    action = collector._compute_action_from_oracle_path(oracle_path, 0)
    assert action in [0, 1, 2, 3], f"无效动作: {action}"
    
    print("✓ 动作计算功能测试通过")

def test_config_loading():
    """测试配置文件加载"""
    print("测试配置文件加载...")
    
    # 检查配置文件是否存在
    config_files = [
        "habitat-baselines/habitat_baselines/config/social_nav_v2/expert_data_collection.yaml",
        "habitat-baselines/habitat_baselines/config/social_nav_v2/advanced_expert_data_collection.yaml"
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        assert config_path.exists(), f"配置文件不存在: {config_file}"
        
        # 尝试读取配置文件
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert "habitat_baselines" in config, "配置文件格式不正确"
        assert "trainer_name" in config["habitat_baselines"], "缺少trainer_name配置"
        
        print(f"✓ 配置文件 {config_file} 加载成功")

def main():
    """主测试函数"""
    print("开始测试专家数据采集器...")
    print("=" * 50)
    
    try:
        test_data_saving()
        test_advanced_data_saving()
        test_action_computation()
        test_config_loading()
        
        print("=" * 50)
        print("✓ 所有测试通过！")
        print("\n专家数据采集器已准备就绪，可以开始收集数据。")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

