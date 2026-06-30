#!/usr/bin/env python3

import sys
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab")

# Import sensors directly to register them
import falcon.vln_sensors

from habitat.core.registry import registry
import torch

def test_sensor_functionality():
    """Test if sensors work correctly."""
    print("Testing sensor functionality...")
    
    # Test GT_ActionSensor
    try:
        gt_sensor_class = registry.get_sensor("GT_ActionSensor")
        gt_sensor = gt_sensor_class()
        
        # Test observation space
        obs_space = gt_sensor._get_observation_space()
        print(f"✅ GT_ActionSensor observation space: {obs_space}")
        
        # Test with mock episode
        class MockEpisode:
            def __init__(self):
                self.gt_action = [1, 2, 1, 0, 3]
        
        mock_episode = MockEpisode()
        obs = gt_sensor.get_observation(episode=mock_episode)
        print(f"✅ GT_ActionSensor observation: {obs[:10]}...")  # Show first 10 elements
        
    except Exception as e:
        print(f"❌ GT_ActionSensor test failed: {e}")
    
    # Test InstructionSensor
    try:
        inst_sensor_class = registry.get_sensor("InstructionSensor")
        inst_sensor = inst_sensor_class()
        
        # Test observation space
        obs_space = inst_sensor._get_observation_space()
        print(f"✅ InstructionSensor observation space: {obs_space}")
        
        # Test with mock episode
        class MockEpisode:
            def __init__(self):
                self.instruction = "Go to the kitchen and turn left"
        
        mock_episode = MockEpisode()
        obs = inst_sensor.get_observation(episode=mock_episode)
        print(f"✅ InstructionSensor observation: {obs[:20]}...")  # Show first 20 elements
        
    except Exception as e:
        print(f"❌ InstructionSensor test failed: {e}")

if __name__ == "__main__":
    test_sensor_functionality()
