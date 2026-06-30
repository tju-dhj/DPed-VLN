#!/usr/bin/env python3

import sys
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-lab")

# Import only the sensors module to register sensors
from falcon import vln_sensors

from habitat.core.registry import registry

def test_sensor_registration():
    """Test if sensors are properly registered."""
    print("Testing sensor registration...")
    
    # Check if FalconGTActionSensor is registered
    try:
        sensor_class = registry.get_sensor("FalconGTActionSensor")
        print("✅ FalconGTActionSensor is registered")
        print(f"   Sensor class: {sensor_class}")
    except KeyError:
        print("❌ FalconGTActionSensor is NOT registered")
    
    # Check if FalconInstructionSensor is registered
    try:
        sensor_class = registry.get_sensor("FalconInstructionSensor")
        print("✅ FalconInstructionSensor is registered")
        print(f"   Sensor class: {sensor_class}")
    except KeyError:
        print("❌ FalconInstructionSensor is NOT registered")
    
    # Check if original InstructionSensor is registered (should be from VLN task)
    try:
        sensor_class = registry.get_sensor("InstructionSensor")
        print("✅ Original InstructionSensor is registered")
        print(f"   Sensor class: {sensor_class}")
    except KeyError:
        print("❌ Original InstructionSensor is NOT registered")

if __name__ == "__main__":
    test_sensor_registration()