#!/usr/bin/env python3

"""
测试trainer注册是否正常工作
"""

import sys
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab")

# 导入必要的模块
import habitat_baselines.rl.ppo  # noqa: F401
from habitat_baselines.common.baseline_registry import baseline_registry

def test_trainer_registration():
    """测试trainer注册"""
    print("Testing trainer registration...")
    
    # 检查expert_data_collector是否已注册
    trainer_init = baseline_registry.get_trainer("expert_data_collector")
    
    if trainer_init is not None:
        print("✅ expert_data_collector is successfully registered!")
        print(f"Trainer class: {trainer_init}")
        return True
    else:
        print("❌ expert_data_collector is NOT registered!")
        return False

def list_all_trainers():
    """列出所有已注册的trainer"""
    print("\nAll registered trainers:")
    for name in baseline_registry._trainer_registry.keys():
        print(f"  - {name}")

if __name__ == "__main__":
    success = test_trainer_registration()
    list_all_trainers()
    
    if success:
        print("\n🎉 Registration test passed!")
    else:
        print("\n💥 Registration test failed!")
        sys.exit(1)

