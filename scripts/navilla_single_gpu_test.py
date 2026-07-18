import sys, os
policy_dir = "/share/home/u19666033/dhj/dped-vln/habitat-baselines/habitat_baselines/rl/ddppo/policy"
sys.path.insert(0, policy_dir)

print("Starting NaviLLa training test...", flush=True)
from llava.train.train_mem import train
print("Import OK. Calling train()...", flush=True)
try:
    train()
except SystemExit as e:
    print(f"train() exited with: {e}", flush=True)
except Exception as e:
    print(f"train() FAILED: {e}", flush=True)
    import traceback; traceback.print_exc()
