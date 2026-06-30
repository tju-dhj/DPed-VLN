#!/usr/bin/env python3

"""
测试NaVILA集成是否正常工作
Test script for NaVILA integration
"""

import sys
import os

def test_imports():
    """测试所有必要的模块是否可以导入"""
    print("=" * 60)
    print("测试模块导入 / Testing module imports")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    # 测试动作解析器
    try:
        from habitat_baselines.rl.ddppo.policy.navila.action_parser import NaVILAActionParser
        print("✓ NaVILAActionParser 导入成功")
        success_count += 1
    except Exception as e:
        print(f"✗ NaVILAActionParser 导入失败: {e}")
        fail_count += 1
    
    # 测试policy注册
    try:
        from habitat_baselines.rl.ddppo.policy import navila_policy
        print("✓ navila_policy 模块导入成功")
        success_count += 1
    except Exception as e:
        print(f"✗ navila_policy 模块导入失败: {e}")
        fail_count += 1
    
    # 测试evaluator
    try:
        from habitat_baselines.rl.ppo.navila_evaluator import NaVILAEvaluator
        print("✓ NaVILAEvaluator 导入成功")
        success_count += 1
    except Exception as e:
        print(f"✗ NaVILAEvaluator 导入失败: {e}")
        fail_count += 1
    
    # 测试LLAVA相关模块（可选）
    try:
        from habitat_baselines.rl.ddppo.policy.navila.llava.constants import IMAGE_TOKEN_INDEX
        print("✓ LLAVA 模块导入成功")
        success_count += 1
    except Exception as e:
        print(f"⚠ LLAVA 模块导入失败（可能需要额外依赖）: {e}")
        # 不计入失败，因为这可能需要额外的依赖
    
    print("-" * 60)
    print(f"成功: {success_count}, 失败: {fail_count}")
    return fail_count == 0


def test_action_parser():
    """测试动作解析器功能"""
    print("\n" + "=" * 60)
    print("测试动作解析器 / Testing Action Parser")
    print("=" * 60)
    
    try:
        from habitat_baselines.rl.ddppo.policy.navila.action_parser import NaVILAActionParser
        
        parser = NaVILAActionParser(forward_step=25, turn_step=15)
        
        # 测试用例
        test_cases = [
            ("The next action is stop", 0, 1),
            ("The next action is move forward 25 cm", 1, 1),
            ("The next action is move forward 50 cm", 1, 2),
            ("The next action is turn left 15 degree", 2, 1),
            ("The next action is turn left 30 degree", 2, 2),
            ("The next action is turn right 15 degree", 3, 1),
            ("The next action is turn right 45 degree", 3, 3),
        ]
        
        all_passed = True
        for instruction, expected_action, expected_repeats in test_cases:
            action, repeats = parser.parse_action(instruction)
            if action == expected_action and repeats == expected_repeats:
                print(f"✓ '{instruction}' → 动作={action}, 重复={repeats}")
            else:
                print(f"✗ '{instruction}' → 期望: 动作={expected_action}, 重复={expected_repeats}, "
                      f"实际: 动作={action}, 重复={repeats}")
                all_passed = False
        
        if all_passed:
            print("-" * 60)
            print("✓ 所有测试用例通过")
            return True
        else:
            print("-" * 60)
            print("✗ 部分测试用例失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_files():
    """测试配置文件是否存在"""
    print("\n" + "=" * 60)
    print("测试配置文件 / Testing Config Files")
    print("=" * 60)
    
    config_files = [
        "habitat-baselines/habitat_baselines/config/social_nav_v2/navila_falcon_hm3d.yaml",
        "habitat-baselines/habitat_baselines/config/social_nav_v2/navila_falcon_hm3d_train.yaml",
    ]
    
    all_exist = True
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file} 存在")
        else:
            print(f"✗ {config_file} 不存在")
            all_exist = False
    
    return all_exist


def test_module_structure():
    """测试模块目录结构"""
    print("\n" + "=" * 60)
    print("测试模块结构 / Testing Module Structure")
    print("=" * 60)
    
    required_files = [
        "habitat-baselines/habitat_baselines/rl/ddppo/policy/navila/__init__.py",
        "habitat-baselines/habitat_baselines/rl/ddppo/policy/navila/action_parser.py",
        "habitat-baselines/habitat_baselines/rl/ddppo/policy/navila_policy.py",
        "habitat-baselines/habitat_baselines/rl/ppo/navila_evaluator.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} 存在")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False
    
    # 检查LLAVA目录
    llava_dir = "habitat-baselines/habitat_baselines/rl/ddppo/policy/navila/llava"
    if os.path.exists(llava_dir) and os.path.isdir(llava_dir):
        print(f"✓ LLAVA模块目录 {llava_dir} 存在")
    else:
        print(f"⚠ LLAVA模块目录 {llava_dir} 不存在（可能需要手动复制）")
    
    return all_exist


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("NaVILA集成测试开始")
    print("NaVILA Integration Test")
    print("=" * 60 + "\n")
    
    # 添加路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "habitat-baselines"))
    
    results = {}
    
    # 运行所有测试
    results["module_structure"] = test_module_structure()
    results["config_files"] = test_config_files()
    results["imports"] = test_imports()
    results["action_parser"] = test_action_parser()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结 / Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("✓✓✓ 所有测试通过！NaVILA集成成功！")
        print("✓✓✓ All tests passed! NaVILA integration successful!")
    else:
        print("⚠⚠⚠ 部分测试失败，请检查上述错误")
        print("⚠⚠⚠ Some tests failed, please check errors above")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
