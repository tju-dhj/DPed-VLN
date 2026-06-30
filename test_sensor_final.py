#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'habitat-lab'))

from habitat.core.registry import registry
from falcon import vln_sensors  # Import to register sensors

def test_sensor_registration():
    """Test sensor registration without duplication."""
    print("Testing sensor registration...")
    
    # Check if our sensors are registered
    try:
        falcon_instruction = registry.get_sensor("FalconInstructionSensor")
        print("✅ FalconInstructionSensor registered")
    except KeyError:
        print("❌ FalconInstructionSensor NOT registered")
    
    try:
        falcon_gt_action = registry.get_sensor("FalconGTActionSensor")
        print("✅ FalconGTActionSensor registered")
    except KeyError:
        print("❌ FalconGTActionSensor NOT registered")
    
    # Check if original sensors are still registered
    try:
        original_instruction = registry.get_sensor("InstructionSensor")
        print("✅ Original InstructionSensor registered")
    except KeyError:
        print("❌ Original InstructionSensor NOT registered")
    
    # Test sensor creation
    try:
        from habitat.core.simulator import SensorSuite
        from habitat.core.simulator import Sensor
        
        # Create mock configs
        class MockConfig:
            def __init__(self, uuid):
                self.uuid = uuid
        
        # Test sensor creation
        instruction_sensor = falcon_instruction(config=MockConfig("falcon_instruction"))
        print("✅ FalconInstructionSensor created successfully")
        
        gt_action_sensor = falcon_gt_action(config=MockConfig("falcon_gt_action"))
        print("✅ FalconGTActionSensor created successfully")
        
        # Test sensor suite creation
        sensor_suite = SensorSuite([instruction_sensor, gt_action_sensor])
        print("✅ SensorSuite created successfully")
        
        print("🎉 All tests passed! No UUID duplication detected.")
        
    except Exception as e:
        print(f"❌ Error creating sensors: {e}")

if __name__ == "__main__":
    test_sensor_registration()
