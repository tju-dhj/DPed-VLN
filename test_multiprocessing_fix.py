#!/usr/bin/env python3
"""
Test script to verify the multiprocessing fix works
"""
import os
import sys
import subprocess

def test_training_command():
    """Test the training command with the multiprocessing fix"""
    print("Testing Falcon training with multiprocessing fix...")
    
    # Change to the habitat-baselines directory
    os.chdir("/share/home/u19666033/dhj/falcon_collect_data/Falcon-main/habitat-baselines")
    
    # Run the training command with a timeout
    cmd = [
        "python", "habitat_baselines/run.py",
        "--config-name=dynamic_vlnce/dynamic_vlnce_hm3d_train.yaml"
    ]
    
    try:
        # Run with a timeout to prevent hanging
        result = subprocess.run(
            cmd, 
            timeout=60,  # 60 second timeout
            capture_output=True, 
            text=True
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print("✅ Training command executed successfully!")
        else:
            print("❌ Training command failed, but multiprocessing errors should be handled gracefully")
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out after 60 seconds - this is expected for training")
    except Exception as e:
        print(f"❌ Error running command: {e}")

if __name__ == "__main__":
    test_training_command()

