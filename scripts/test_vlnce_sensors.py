#!/usr/bin/env python3
"""
测试基于VLN-CE模式的Dynamic VLN-CE传感器
"""

import torch
import numpy as np
from habitat.core.registry import registry

# 导入我们的传感器
from falcon.vln_sensors import (
    DynamicVLNCEInstructionSensor,
    DynamicVLNCEGTActionSensor,
)


def test_vlnce_style_sensors():
    """测试遵循VLN-CE模式的传感器"""
    
    print("=== VLN-CE 风格传感器测试 ===")
    
    # 创建传感器实例
    instruction_sensor = DynamicVLNCEInstructionSensor()
    gt_action_sensor = DynamicVLNCEGTActionSensor()
    
    print(f"指令传感器UUID: {instruction_sensor._get_uuid()}")
    print(f"GT动作传感器UUID: {gt_action_sensor._get_uuid()}")
    print()
    
    # 模拟episode数据
    mock_episode = type('MockEpisode', (), {
        'episode_id': 'test_episode_001',
        'scene_id': 'test_scene',
        'instruction': 'Move forward through the hallway past the wall clock on the left. Turn right into the kitchen.',
        'instruction_tokens': ['Move', 'forward', 'through', 'the', 'hallway', 'past', 'the', 'wall', 'clock', 'on', 'the', 'left.', 'Turn', 'right', 'into', 'the', 'kitchen.'],
        'gt_action': [1, 1, 1, 2, 1, 1, 0, 1, 2, 1, 1, 1, 0],  # 示例动作序列
        'goals': [],
        'start_position': [0.0, 0.0, 0.0],
        'start_rotation': [0.0, 0.0, 0.0, 1.0],
    })()
    
    print(f"Episode ID: {mock_episode.episode_id}")
    print(f"Scene ID: {mock_episode.scene_id}")
    print(f"Instruction: {mock_episode.instruction}")
    print(f"GT Actions: {mock_episode.gt_action}")
    print()
    
    # 测试指令传感器
    print("=== 指令传感器测试 ===")
    instruction_obs = instruction_sensor.get_observation(episode=mock_episode)
    print(f"指令观察: {instruction_obs}")
    print(f"观察类型: {type(instruction_obs)}")
    print(f"观察空间: {instruction_sensor._get_observation_space()}")
    print()
    
    # 测试GT动作传感器
    print("=== GT动作传感器测试 ===")
    gt_action_obs = gt_action_sensor.get_observation(episode=mock_episode)
    print(f"GT动作观察形状: {gt_action_obs.shape}")
    print(f"GT动作观察类型: {type(gt_action_obs)}")
    print(f"GT动作观察空间: {gt_action_sensor._get_observation_space()}")
    print(f"前10个动作: {gt_action_obs[:10].tolist()}")
    print()
    
    # 测试没有episode的情况
    print("=== 无Episode测试 ===")
    empty_instruction = instruction_sensor.get_observation()
    empty_gt_action = gt_action_sensor.get_observation()
    
    print(f"空指令观察: '{empty_instruction}'")
    print(f"空GT动作观察形状: {empty_gt_action.shape}")
    print(f"空GT动作观察: {empty_gt_action.tolist()}")
    print()


def test_sensor_registration():
    """测试传感器注册"""
    
    print("=== 传感器注册测试 ===")
    
    # 检查传感器是否已注册
    instruction_registered = "DynamicVLNCEInstructionSensor" in registry.mapping["sensor"]
    gt_action_registered = "DynamicVLNCEGTActionSensor" in registry.mapping["sensor"]
    
    print(f"指令传感器已注册: {instruction_registered}")
    print(f"GT动作传感器已注册: {gt_action_registered}")
    
    if instruction_registered and gt_action_registered:
        print("✅ 所有传感器都已正确注册")
    else:
        print("❌ 部分传感器未注册")
    print()


def test_observation_spaces():
    """测试观察空间"""
    
    print("=== 观察空间测试 ===")
    
    instruction_sensor = DynamicVLNCEInstructionSensor()
    gt_action_sensor = DynamicVLNCEGTActionSensor()
    
    # 测试观察空间
    instruction_space = instruction_sensor._get_observation_space()
    gt_action_space = gt_action_sensor._get_observation_space()
    
    print(f"指令观察空间: {instruction_space}")
    print(f"GT动作观察空间: {gt_action_space}")
    
    # 测试观察空间的有效性
    mock_instruction = "Test instruction"
    mock_gt_action = torch.zeros(100, dtype=torch.long)
    
    instruction_valid = instruction_space.contains(mock_instruction)
    gt_action_valid = gt_action_space.contains(mock_gt_action)
    
    print(f"指令观察空间有效性: {instruction_valid}")
    print(f"GT动作观察空间有效性: {gt_action_valid}")
    print()


def test_with_different_episodes():
    """测试不同episode的情况"""
    
    print("=== 不同Episode测试 ===")
    
    instruction_sensor = DynamicVLNCEInstructionSensor()
    gt_action_sensor = DynamicVLNCEGTActionSensor()
    
    # 测试用例1: 完整episode
    episode1 = type('Episode1', (), {
        'instruction': 'Go to the kitchen',
        'gt_action': [1, 1, 2, 1, 0],
    })()
    
    # 测试用例2: 只有指令
    episode2 = type('Episode2', (), {
        'instruction': 'Turn left and walk forward',
        'gt_action': None,
    })()
    
    # 测试用例3: 只有GT动作
    episode3 = type('Episode3', (), {
        'instruction': None,
        'gt_action': [2, 1, 1, 0],
    })()
    
    test_cases = [
        ("完整Episode", episode1),
        ("只有指令", episode2),
        ("只有GT动作", episode3),
    ]
    
    for name, episode in test_cases:
        print(f"--- {name} ---")
        
        instruction_obs = instruction_sensor.get_observation(episode=episode)
        gt_action_obs = gt_action_sensor.get_observation(episode=episode)
        
        print(f"指令: '{instruction_obs}'")
        print(f"GT动作形状: {gt_action_obs.shape}")
        print(f"GT动作前5个: {gt_action_obs[:5].tolist()}")
        print()


if __name__ == "__main__":
    print("VLN-CE 风格传感器测试开始\n")
    
    # 基础功能测试
    test_vlnce_style_sensors()
    
    # 传感器注册测试
    test_sensor_registration()
    
    # 观察空间测试
    test_observation_spaces()
    
    # 不同episode测试
    test_with_different_episodes()
    
    print("测试完成!")



