#!/usr/bin/env python3

import habitat
from habitat.config import read_write
from habitat_baselines.config.default import get_config

def test_original_config():
    """Test original config without VLN sensors"""
    print("=== Testing Original Config (without VLN sensors) ===")
    try:
        cfg = get_config('habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d_train.yaml')
        
        # Create environment
        env = habitat.Env(config=cfg.habitat)
        
        # Try to get agent state
        state = env.sim.get_agent_state(0)
        print(f"✅ Original config works: Agent position: {state.position}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Original config failed: {e}")
        return False

def test_vln_config():
    """Test config with VLN sensors"""
    print("\n=== Testing VLN Config (with VLN sensors) ===")
    try:
        cfg = get_config('habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train.yaml')
        
        # Create environment
        env = habitat.Env(config=cfg.habitat)
        
        # Try to get agent state
        state = env.sim.get_agent_state(0)
        print(f"✅ VLN config works: Agent position: {state.position}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ VLN config failed: {e}")
        return False

def test_vln_config_without_sensors():
    """Test VLN config but remove VLN sensors from obs_keys"""
    print("\n=== Testing VLN Config without VLN sensors in obs_keys ===")
    try:
        cfg = get_config('habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train.yaml')
        
        # Remove VLN sensors from obs_keys
        cfg.habitat.gym.obs_keys = [
            'agent_0_articulated_agent_jaw_rgb',
            'agent_0_articulated_agent_jaw_depth',
            'agent_0_overhead_front_rgb',
            'agent_0_overhead_front_depth',
            'agent_0_third_rgb',
            'agent_0_third_depth',
            'agent_0_pointgoal_with_gps_compass',
            'agent_0_localization_sensor',
            'agent_0_human_num_sensor',
            'agent_0_oracle_humanoid_future_trajectory',
            # Removed VLN sensors
        ]
        
        # Create environment
        env = habitat.Env(config=cfg.habitat)
        
        # Try to get agent state
        state = env.sim.get_agent_state(0)
        print(f"✅ VLN config without sensors in obs_keys works: Agent position: {state.position}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ VLN config without sensors failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing different configurations to identify the issue...")
    
    # Test 1: Original config
    test1 = test_original_config()
    
    # Test 2: VLN config with sensors
    test2 = test_vln_config()
    
    # Test 3: VLN config without sensors in obs_keys
    test3 = test_vln_config_without_sensors()
    
    print(f"\n=== Results ===")
    print(f"Original config: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"VLN config with sensors: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"VLN config without sensors in obs_keys: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test1 and not test2 and test3:
        print("\n🔍 Issue identified: VLN sensors in obs_keys are causing the problem!")
    elif test1 and not test2 and not test3:
        print("\n🔍 Issue identified: VLN dataset or sensor registration is causing the problem!")
    else:
        print("\n🔍 Issue is more complex, need further investigation.")
