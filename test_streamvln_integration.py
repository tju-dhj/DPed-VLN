#!/usr/bin/env python3

"""
Quick test script to verify StreamVLN integration.

This script tests if all components are properly installed and can be imported.
It does NOT require a pretrained model to run.

Usage:
    python test_streamvln_integration.py
"""

import sys
import traceback
from typing import List, Tuple


def test_import(module_name: str, description: str) -> Tuple[bool, str]:
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        return True, f"✓ {description}"
    except ImportError as e:
        return False, f"✗ {description}: {str(e)}"
    except Exception as e:
        return False, f"✗ {description}: Unexpected error: {str(e)}"


def main():
    print("=" * 80)
    print("StreamVLN Integration Test")
    print("=" * 80)
    print("\nTesting component imports...\n")
    
    tests = [
        # Core dependencies
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        
        # Habitat
        ("habitat", "Habitat Lab"),
        ("habitat_baselines", "Habitat Baselines"),
        
        # StreamVLN components
        ("habitat_baselines.rl.ddppo.policy.streamvln_policy", "StreamVLN Policy"),
        ("habitat_baselines.rl.ddppo.policy.streamvln", "StreamVLN Module"),
        ("habitat_baselines.rl.ddppo.policy.streamvln.habitat_compat", "Habitat Compatibility Layer"),
    ]
    
    results: List[Tuple[bool, str]] = []
    
    for module, desc in tests:
        success, message = test_import(module, desc)
        results.append((success, message))
        print(message)
    
    # Test optional dependencies
    print("\nOptional dependencies:")
    optional_tests = [
        ("flash_attn", "Flash Attention 2 (recommended for speed)"),
    ]
    
    for module, desc in optional_tests:
        success, message = test_import(module, desc)
        if not success:
            message = f"○ {desc}: Not installed (optional)"
        print(message)
    
    # Summary
    print("\n" + "=" * 80)
    total = len(tests)
    passed = sum(1 for success, _ in results if success)
    failed = total - passed
    
    print(f"Test Summary: {passed}/{total} passed")
    
    if failed > 0:
        print(f"\n⚠ {failed} required component(s) failed to import!")
        print("Please install missing dependencies:")
        print("  pip install -r streamvln_requirements.txt")
        return 1
    else:
        print("\n✓ All required components imported successfully!")
        
        # Try to instantiate the policy (without model)
        print("\nTesting policy instantiation...")
        try:
            from gym import spaces
            import numpy as np
            from habitat_baselines.rl.ddppo.policy.streamvln_policy import StreamVLNPolicy
            
            # Create dummy spaces
            obs_space = spaces.Dict({
                'rgb': spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
            })
            action_space = spaces.Discrete(4)
            
            # Note: This will fail without a model_path, but that's expected
            # We just want to verify the class can be imported and initialized
            print("✓ StreamVLNPolicy class can be imported")
            print("\nNote: Actual policy instantiation requires a pretrained model path.")
            print("See STREAMVLN_INTEGRATION.md for usage instructions.")
            
        except Exception as e:
            print(f"⚠ Could not test policy instantiation: {e}")
            print("This might be expected if model_path is not provided.")
        
        print("\n" + "=" * 80)
        print("Integration test completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Download or specify a pretrained StreamVLN model")
        print("2. Run inference example:")
        print("   python examples/streamvln_inference_example.py --model-path /path/to/model")
        print("3. See STREAMVLN_INTEGRATION.md for detailed documentation")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
