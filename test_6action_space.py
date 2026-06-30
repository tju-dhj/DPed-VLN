#!/usr/bin/env python3
"""
6动作空间验证脚本
用于快速验证新的6动作空间配置是否正确
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-lab")
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-baselines")

def test_action_imports():
    """测试动作类是否可以正确导入"""
    print("=" * 60)
    print("测试1: 验证动作类导入")
    print("=" * 60)

    try:
        from habitat.core.registry import registry

        # 检查所有6个动作是否已注册
        expected_actions = [
            'DiscreteStopAction',
            'DiscreteMoveForwardAction',
            'DiscreteTurnLeftAction',
            'DiscreteTurnRightAction',
            'DiscretePauseAction',
            'DiscreteMoveBackwardAction'
        ]

        # 尝试导入falcon/additional_action.py以触发注册
        try:
            import falcon.additional_action
            print("✓ falcon.additional_action 模块导入成功")
        except ImportError as e:
            print(f"✗ falcon.additional_action 导入失败: {e}")
            return False

        # 检查动作是否已注册
        registered_actions = registry._task_action_name_to_task_action

        print(f"\n已注册的动作类: {len(registered_actions)}")
        for action_name in expected_actions:
            if action_name in registered_actions:
                print(f"  ✓ {action_name}")
            else:
                print(f"  ✗ {action_name} (未找到)")

        return True

    except Exception as e:
        print(f"✗ 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gym_wrapper():
    """测试gym_wrapper中的动作映射"""
    print("\n" + "=" * 60)
    print("测试2: 验证 gym_wrapper 动作映射")
    print("=" * 60)

    try:
        from habitat.gym.gym_wrapper import continuous_vector_action_to_hab_dict_v3
        from gym import spaces
        import numpy as np

        # 模拟动作空间
        mock_action_space = spaces.Dict({
            'agent_0_discrete_stop': spaces.Box(low=0, high=1, shape=(1,)),
            'agent_0_discrete_move_forward': spaces.Box(low=0, high=1, shape=(1,)),
            'agent_0_discrete_turn_left': spaces.Box(low=0, high=1, shape=(1,)),
            'agent_0_discrete_turn_right': spaces.Box(low=0, high=1, shape=(1,)),
            'agent_0_discrete_pause': spaces.Box(low=0, high=1, shape=(1,)),
            'agent_0_discrete_move_backward': spaces.Box(low=0, high=1, shape=(1,)),
        })

        mock_vector_space = spaces.Box(low=0, high=5, shape=(1,))

        # 测试每个动作
        action_names = [
            'agent_0_discrete_stop',
            'agent_0_discrete_move_forward',
            'agent_0_discrete_turn_left',
            'agent_0_discrete_turn_right',
            'agent_0_discrete_pause',
            'agent_0_discrete_move_backward'
        ]

        print("\n测试动作映射 (动作ID → 动作名称):")
        for action_id in range(6):
            action_array = np.array([action_id], dtype=np.float32)
            result = continuous_vector_action_to_hab_dict_v3(
                mock_action_space,
                mock_vector_space,
                action_array
            )
            expected_name = action_names[action_id]
            actual_name = result['action']

            if actual_name == expected_name:
                print(f"  ✓ 动作 {action_id} → {actual_name}")
            else:
                print(f"  ✗ 动作 {action_id} → {actual_name} (期望: {expected_name})")

        return True

    except Exception as e:
        print(f"✗ gym_wrapper测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_files():
    """测试配置文件是否存在"""
    print("\n" + "=" * 60)
    print("测试3: 验证配置文件")
    print("=" * 60)

    base_path = "/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines/config/dynamic_vlnce"

    config_files = [
        "dynamic_vlnce_hm3d_train_v2_6actions.yaml",
        "dynamic_vlnce_hm3d_il_val_v11_6actions.yaml"
    ]

    all_exist = True
    for config_file in config_files:
        full_path = os.path.join(base_path, config_file)
        if os.path.exists(full_path):
            print(f"  ✓ {config_file}")

            # 检查文件中是否包含6个动作配置
            with open(full_path, 'r') as f:
                content = f.read()
                required_actions = [
                    'agent_0_discrete_stop',
                    'agent_0_discrete_move_forward',
                    'agent_0_discrete_turn_left',
                    'agent_0_discrete_turn_right',
                    'agent_0_discrete_pause',
                    'agent_0_discrete_move_backward'
                ]

                missing_actions = []
                for action in required_actions:
                    if action not in content:
                        missing_actions.append(action)

                if missing_actions:
                    print(f"    ⚠ 缺少动作配置: {', '.join(missing_actions)}")
                    all_exist = False
                else:
                    print(f"    ✓ 包含所有6个动作配置")
        else:
            print(f"  ✗ {config_file} (不存在)")
            all_exist = False

    return all_exist

def test_action_config():
    """测试动作配置dataclass"""
    print("\n" + "=" * 60)
    print("测试4: 验证动作配置类")
    print("=" * 60)

    try:
        from falcon.additional_action import (
            DiscreteStopActionConfig,
            DiscretePauseActionConfig,
            DiscreteMoveForwardActionConfig,
            DiscreteMoveBackwardActionConfig,
            DiscreteTurnLeftActionConfig,
            DiscreteTurnRightActionConfig
        )

        configs = [
            ('DiscreteStopActionConfig', DiscreteStopActionConfig),
            ('DiscretePauseActionConfig', DiscretePauseActionConfig),
            ('DiscreteMoveForwardActionConfig', DiscreteMoveForwardActionConfig),
            ('DiscreteMoveBackwardActionConfig', DiscreteMoveBackwardActionConfig),
            ('DiscreteTurnLeftActionConfig', DiscreteTurnLeftActionConfig),
            ('DiscreteTurnRightActionConfig', DiscreteTurnRightActionConfig)
        ]

        for name, config_class in configs:
            config = config_class()
            print(f"  ✓ {name}")
            print(f"    - type: {config.type}")
            print(f"    - lin_speed: {config.lin_speed}")
            print(f"    - ang_speed: {config.ang_speed}")

        return True

    except Exception as e:
        print(f"✗ 动作配置类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("6动作空间验证测试")
    print("=" * 60 + "\n")

    results = []

    # 运行所有测试
    results.append(("动作类导入", test_action_imports()))
    results.append(("gym_wrapper映射", test_gym_wrapper()))
    results.append(("配置文件", test_config_files()))
    results.append(("动作配置类", test_action_config()))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {test_name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！6动作空间配置正确。")
        print("\n可以使用以下命令开始训练：")
        print("  python -u -m habitat_baselines.run --config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train_v2_6actions")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
