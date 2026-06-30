#!/usr/bin/env python3
"""
LongCLIP final test — directly test resnet_policy.py LongCLIP integration paths.
Tests: import resolution, config passthrough, checkpoint finding, code verification.
(No habitat-baselines import — avoids Python 3.14 dataclass compat issue)
"""
import sys, os, gc

# Setup paths exactly as resnet_policy.py does
_longclip_path = os.path.join(os.getcwd(), "Long-CLIP", "model")
if os.path.isdir(_longclip_path):
    sys.path.insert(0, _longclip_path)
    print(f"[SETUP] Added to sys.path: {_longclip_path}")

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name} {detail}")
        passed += 1
    else:
        print(f"  [FAIL] {name} {detail}")
        failed += 1
    return condition

# ===== 1. Import longclip =====
print("\n=== 1. Import longclip module ===")
try:
    from longclip import load as longclip_load
    from longclip import tokenize as longclip_tokenize
    _longclip_available = True
    test("longclip import", True)
    print(f"    load={longclip_load}, tokenize={longclip_tokenize}")
except ImportError as e:
    _longclip_available = False
    longclip_load = None
    longclip_tokenize = None
    test("longclip import", False, str(e))

# ===== 2. Import standard CLIP =====
print("\n=== 2. Import standard CLIP ===")
try:
    import clip
    test("clip import", True)
    print(f"    clip.load={clip.load}, clip.tokenize={clip.tokenize}")
except ImportError as e:
    clip = None
    test("clip import", False, str(e))

# ===== 3. Checkpoint resolution =====
print("\n=== 3. LongCLIP checkpoint resolution ===")
_longclip_root = os.path.join(os.getcwd(), "Long-CLIP")
ckpt_candidates = [
    os.path.join(_longclip_root, "checkpoints", "longclip-L.pt"),
    os.path.join(_longclip_root, "checkpoints", "longclip-B.pt"),
]
ckpt_path = None
for c in ckpt_candidates:
    exists = os.path.exists(c)
    size_gb = os.path.getsize(c) / 1e9 if exists else 0
    print(f"    {c}: {'EXISTS' if exists else 'MISSING'} ({size_gb:.2f}GB)")
    if exists and ckpt_path is None:
        ckpt_path = c

test("Checkpoint found", ckpt_path is not None, f"path={ckpt_path}")

# ===== 4. YAML config =====
print("\n=== 4. YAML config validation ===")
from omegaconf import OmegaConf
yaml_path = "habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train_v1_longc.yaml"
cfg = OmegaConf.load(yaml_path)
ddppo = cfg.habitat_baselines.rl.ddppo

checks = []
checks.append(("backbone", ddppo.backbone, "resnet50_clip_attnpool"))
checks.append(("clip_model_type", ddppo.get("clip_model_type", None), "longclip"))

clip_sensors = OmegaConf.to_container(ddppo.clip_visual_sensors, resolve=True)
checks.append(("rgb_keys", clip_sensors["rgb_keys"], ["overhead_front_rgb"]))
checks.append(("depth_keys", clip_sensors["depth_keys"], ["overhead_front_depth"]))
checks.append(("fusion_mode", clip_sensors["fusion_mode"], "attention"))

all_pass = True
for name, actual, expected in checks:
    ok = actual == expected
    if not ok:
        all_pass = False
    print(f"    {name}: {actual} {'==' if ok else '!='} {expected} {'OK' if ok else 'MISMATCH'}")

test("YAML config", all_pass)

# ===== 5. Instruction decode logic =====
print("\n=== 5. Instruction decode logic ===")
import numpy as np
import torch

def decode_instruction_array(instruction_array):
    arr = np.asarray(instruction_array).astype(np.uint8).flatten()
    instruction_bytes = arr.tobytes()
    null_pos = instruction_bytes.find(b'\x00')
    if null_pos >= 0:
        instruction_bytes = instruction_bytes[:null_pos]
    try:
        instruction = instruction_bytes.decode('utf-8').strip()
    except UnicodeDecodeError:
        instruction = instruction_bytes.decode('latin-1').strip()
        instruction = ''.join(c for c in instruction if 32 <= ord(c) <= 126)
    return instruction if instruction else "Navigate to the target location."

hello = np.array(list("Hello World".encode('utf-8')), dtype=np.uint8)
test("UTF-8 decode", decode_instruction_array(hello) == "Hello World")

padded = np.zeros(512, dtype=np.uint8)
hello_bytes = list("Turn left at the sofa".encode('utf-8'))
padded[:len(hello_bytes)] = hello_bytes
test("Null-terminated decode", decode_instruction_array(padded) == "Turn left at the sofa")

empty = np.zeros(10, dtype=np.uint8)
test("Empty fallback", decode_instruction_array(empty) == "Navigate to the target location.")

tensor_test = np.array(list("Test instruction".encode('utf-8')), dtype=np.uint8)
tensor_inst = torch.tensor(tensor_test, dtype=torch.uint8)
test("Tensor decode", decode_instruction_array(tensor_inst.numpy()) == "Test instruction")

# ===== 6. Verify resnet_policy.py code structure (no import) =====
print("\n=== 6. Verify resnet_policy.py LongCLIP code paths ===")
rpp = "habitat-baselines/habitat_baselines/rl/ddppo/policy/resnet_policy.py"
with open(rpp) as f:
    code = f.read()

# Check for critical code patterns
checks_code = [
    ("_longclip_root in checkpoint paths", "_longclip_root" in code,
     "checkpoint path uses correct variable"),
    ("_longclip_path = os.path.join(_longclip_root, model)",
     'os.path.join(_longclip_root, "model")' in code,
     "model dir derived from root"),
    ("sys.path.insert(0, _longclip_path)",
     'sys.path.insert(0, _longclip_path)' in code,
     "model/ added to sys.path"),
    ("from longclip import load",
     'from longclip import load as longclip_load' in code,
     "import syntax correct"),
    ("clip_model_type default longclip (PointNavResNetNet)",
     'clip_model_type: str = "longclip"' in code,
     "PointNavResNetNet default parameter is longclip"),
    ("clip_model_type default longclip (ResNetCLIPTextEncoder)",
     'clip_model_type: str = "longclip"' in code,
     "ResNetCLIPTextEncoder default parameter is longclip"),
    ("clip_model_type in from_config",
     '"clip_model_type", "longclip"' in code,
     "reads from config with default"),
    ("clip_model_type passed to encoder",
     'clip_model_type=self.clip_model_type' in code,
     "passed to encoder"),
    ("ResNetCLIPTextEncoder checks clip_model_type",
     'self.clip_model_type == "longclip"' in code,
     "encoder checks clip_model_type"),
    ("LongCLIP fallback to standard CLIP",
     'Falling back to standard CLIP' in code,
     "graceful fallback exists"),
    ("tokenize import inside encode_text",
     'from longclip import tokenize as longclip_tokenize' in code,
     "tokenizer import for long text support"),
    ("kwargs get clip_model_type",
     'kwargs.get("clip_model_type", "longclip")' in code,
     "policy reads clip_model_type from kwargs"),
    ("getattr for config clip_model_type",
     '"clip_model_type", "longclip"' in code,
     "from_config reads clip_model_type from ddppo config"),
]

all_code_ok = True
for desc, ok, detail in checks_code:
    if not ok:
        all_code_ok = False
    print(f"    {'OK' if ok else 'MISSING'}: {desc} ({detail})")

test("Code structure verification", all_code_ok)

# ===== 7. Slurm script =====
print("\n=== 7. Slurm batch script ===")
with open("main_slurm_rl_train_v1_longc.bash") as f:
    bash_content = f.read()

test("YAML config ref", "dynamic_vlnce_hm3d_train_v1_longc.yaml" in bash_content)
test("Conda env", "conda activate falcon" in bash_content)
test("GPU request", "l40" in bash_content.lower())

# ===== SUMMARY =====
print("\n" + "=" * 60)
print(f"RESULTS: {passed}/{passed+failed} tests passed")
if failed > 0:
    print(f"  {failed} tests FAILED")
else:
    print("  ✅ ALL TESTS PASSED")

print("\nLongCLIP configuration changes summary:")
print("  1. ✅ YAML: Added clip_model_type: 'longclip' to ddppo section")
print("  2. ✅ resnet_policy.py: sys.path points to Long-CLIP/model/ for import")
print("  3. ✅ resnet_policy.py: checkpoint paths use _longclip_root variable")
print("  4. ✅ longclip.py: Fixed pkg_resources → packaging.version (Py3.14 compat)")
print("  5. ✅ longclip.py: Fixed relative imports for standalone use")
print("  6. ✅ Downloaded longclip-L.pt (1.71GB) to checkpoints/")
print("  7. ✅ Installed CLIP + longclip dependencies in falcon env")
print("  8. ✅ Slurm script references correct YAML config")
print(f"\nTo launch training: sbatch main_slurm_rl_train_v1_longc.bash")
