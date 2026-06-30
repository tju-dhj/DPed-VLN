#!/usr/bin/env python3
"""
测试评估checkpoint功能的脚本
用于验证checkpoint的保存和加载功能
"""

import json
import os
from collections import defaultdict

def test_checkpoint_save_load():
    """测试checkpoint的保存和加载功能"""
    
    # 模拟数据
    test_checkpoint_path = "/tmp/test_eval_checkpoint.json"
    
    # 创建测试数据
    stats_episodes = {
        (("scene1", "ep1"), 1): {"reward": 0.5, "success": 1.0, "spl": 0.8},
        (("scene1", "ep2"), 1): {"reward": 0.3, "success": 0.0, "spl": 0.0},
        (("scene2", "ep1"), 1): {"reward": 0.7, "success": 1.0, "spl": 0.9},
    }
    
    ep_eval_count = defaultdict(lambda: 0)
    ep_eval_count[("scene1", "ep1")] = 1
    ep_eval_count[("scene1", "ep2")] = 1
    ep_eval_count[("scene2", "ep1")] = 1
    
    actions_record = defaultdict(list)
    actions_record[("scene1", "ep1", 1)] = [
        {"type": "array", "value": [0.1, 0.2, 0.3]},
        {"type": "array", "value": [0.4, 0.5, 0.6]},
    ]
    
    # 保存checkpoint
    print("=" * 60)
    print("测试保存checkpoint...")
    checkpoint_data = {
        'stats_episodes': {},
        'ep_eval_count': {},
        'actions_record': {}
    }
    
    for ((scene_id, episode_id), eval_count), stats in stats_episodes.items():
        key_str = f"{scene_id}|{episode_id}|{eval_count}"
        checkpoint_data['stats_episodes'][key_str] = stats
    
    for (scene_id, episode_id), count in ep_eval_count.items():
        key_str = f"{scene_id}|{episode_id}"
        checkpoint_data['ep_eval_count'][key_str] = count
    
    for (scene_id, episode_id, eval_count), actions in actions_record.items():
        key_str = f"{scene_id}|{episode_id}|{eval_count}"
        checkpoint_data['actions_record'][key_str] = actions
    
    with open(test_checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"✓ Checkpoint已保存到: {test_checkpoint_path}")
    print(f"  - stats_episodes: {len(checkpoint_data['stats_episodes'])} 条")
    print(f"  - ep_eval_count: {len(checkpoint_data['ep_eval_count'])} 条")
    print(f"  - actions_record: {len(checkpoint_data['actions_record'])} 条")
    
    # 加载checkpoint
    print("\n" + "=" * 60)
    print("测试加载checkpoint...")
    with open(test_checkpoint_path, 'r') as f:
        loaded_data = json.load(f)
    
    # 恢复stats_episodes
    loaded_stats_episodes = {}
    for key_str, stats in loaded_data.get('stats_episodes', {}).items():
        parts = key_str.split('|')
        if len(parts) == 3:
            scene_id, episode_id, eval_count = parts[0], parts[1], int(parts[2])
            loaded_stats_episodes[((scene_id, episode_id), eval_count)] = stats
    
    # 恢复ep_eval_count
    loaded_ep_eval_count = defaultdict(lambda: 0)
    for key_str, count in loaded_data.get('ep_eval_count', {}).items():
        parts = key_str.split('|')
        if len(parts) == 2:
            scene_id, episode_id = parts[0], parts[1]
            loaded_ep_eval_count[(scene_id, episode_id)] = count
    
    # 恢复actions_record
    loaded_actions_record = defaultdict(list)
    for key_str, actions in loaded_data.get('actions_record', {}).items():
        parts = key_str.split('|')
        if len(parts) == 3:
            scene_id, episode_id, eval_count = parts[0], parts[1], int(parts[2])
            loaded_actions_record[(scene_id, episode_id, eval_count)] = actions
    
    print(f"✓ Checkpoint已加载")
    print(f"  - stats_episodes: {len(loaded_stats_episodes)} 条")
    print(f"  - ep_eval_count: {len(loaded_ep_eval_count)} 条")
    print(f"  - actions_record: {len(loaded_actions_record)} 条")
    
    # 验证数据一致性
    print("\n" + "=" * 60)
    print("验证数据一致性...")
    
    assert len(stats_episodes) == len(loaded_stats_episodes), "stats_episodes数量不匹配"
    assert len(ep_eval_count) == len(loaded_ep_eval_count), "ep_eval_count数量不匹配"
    assert len(actions_record) == len(loaded_actions_record), "actions_record数量不匹配"
    
    # 验证具体内容
    for key, value in stats_episodes.items():
        assert key in loaded_stats_episodes, f"缺失key: {key}"
        assert loaded_stats_episodes[key] == value, f"值不匹配: {key}"
    
    print("✓ 所有数据验证通过！")
    
    # 清理测试文件
    os.remove(test_checkpoint_path)
    print(f"\n✓ 测试文件已清理: {test_checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！checkpoint功能正常工作。")
    print("=" * 60)


def test_checkpoint_format():
    """测试checkpoint文件格式"""
    print("\n" + "=" * 60)
    print("Checkpoint文件格式示例:")
    print("=" * 60)
    
    example_checkpoint = {
        "stats_episodes": {
            "scene1|ep1|1": {
                "reward": 0.5,
                "success": 1.0,
                "spl": 0.8,
                "psc": 0.7
            },
            "scene1|ep2|1": {
                "reward": 0.3,
                "success": 0.0,
                "spl": 0.0,
                "psc": 0.0
            }
        },
        "ep_eval_count": {
            "scene1|ep1": 1,
            "scene1|ep2": 1
        },
        "actions_record": {
            "scene1|ep1|1": [
                {"type": "array", "value": [0.1, 0.2, 0.3]},
                {"type": "array", "value": [0.4, 0.5, 0.6]}
            ]
        }
    }
    
    print(json.dumps(example_checkpoint, indent=2))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("评估Checkpoint功能测试")
    print("=" * 60)
    
    # 运行测试
    test_checkpoint_save_load()
    test_checkpoint_format()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

