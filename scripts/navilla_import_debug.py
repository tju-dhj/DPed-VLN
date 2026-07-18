import sys, os
policy_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, policy_dir)

# Print which llava is loaded FIRST, before any import
import importlib.util
spec = importlib.util.find_spec('llava')
print(f"llava will load from: {spec.origin}", flush=True)

print("=== Step 1: import llava ===", flush=True)
import llava
print(f"  OK: {llava.__file__}", flush=True)

print("=== Step 2: import llava.constants ===", flush=True)
import llava.constants
print("  OK", flush=True)

print("=== Step 3: import llava.train ===", flush=True)
import llava.train
print("  OK", flush=True)

print("=== Step 4: import llava.train.train ===", flush=True)
try:
    import llava.train.train
    print("  OK", flush=True)
except Exception as e:
    print(f"  FAIL: {e}", flush=True)
    import traceback; traceback.print_exc()
    sys.exit(1)

print("=== ALL IMPORTS PASSED ===", flush=True)
