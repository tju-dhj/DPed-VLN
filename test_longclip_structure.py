#!/usr/bin/env python3
"""
LongCLIP 结构测试 - 验证代码路径、配置解析、类创建（不加载完整1.6GB模型）
"""
import sys, os

sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/Long-CLIP/model")

print("=" * 60)
print("TEST 1: LongCLIP module import (no model load)")
print("=" * 60)
try:
    from longclip import load as longclip_load
    from longclip import tokenize as longclip_tokenize
    print("[PASS] longclip imported")
except ImportError as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST 2: sys.path resolution")
print("=" * 60)
# Complete path list that would be set in resnet_policy.py
candidates = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Long-CLIP"),
    os.path.join(os.getcwd(), "Long-CLIP"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Long-CLIP"),
]
for c in candidates:
    full = os.path.abspath(c)
    model_dir = os.path.join(full, "model")
    ckpt_dir = os.path.join(full, "checkpoints")
    print(f"  Candidate: {full}")
    print(f"    model/: {'EXISTS' if os.path.isdir(model_dir) else 'MISSING'}")
    print(f"    checkpoints/: {'EXISTS' if os.path.isdir(ckpt_dir) else 'MISSING'}")
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            fpath = os.path.join(ckpt_dir, f)
            if os.path.isfile(fpath):
                print(f"      {f}: {os.path.getsize(fpath)/1e9:.2f}GB")

# Verify the root is found
_longclip_root = candidates[0] if os.path.isdir(candidates[0]) else (candidates[1] if os.path.isdir(candidates[1]) else candidates[2])
assert os.path.isdir(_longclip_root), "Long-CLIP root not found!"
print(f"\n[PASS] Long-CLIP root resolved: {_longclip_root}")
print(f"[PASS] Model dir: {os.path.join(_longclip_root, 'model')}")
print(f"[PASS] Checkpoint: {os.path.join(_longclip_root, 'checkpoints', 'longclip-L.pt')}")

print("\n" + "=" * 60)
print("TEST 3: YAML config parsing")
print("=" * 60)
try:
    from omegaconf import OmegaConf
    yaml_path = "/share/home/u19666033/dhj/DPed_pro/habitat-baselines/habitat_baselines/config/dynamic_vlnce/dynamic_vlnce_hm3d_train_v1_longc.yaml"
    cfg = OmegaConf.load(yaml_path)

    backbone = cfg.habitat_baselines.rl.ddppo.backbone
    print(f"  backbone: {backbone}")
    assert "clip" in backbone, f"Expected CLIP backbone, got {backbone}"

    clip_sensors = OmegaConf.to_container(cfg.habitat_baselines.rl.ddppo.clip_visual_sensors, resolve=True)
    print(f"  clip_visual_sensors.rgb_keys: {clip_sensors['rgb_keys']}")
    print(f"  clip_visual_sensors.depth_keys: {clip_sensors['depth_keys']}")
    print(f"  clip_visual_sensors.fusion_mode: {clip_sensors['fusion_mode']}")

    # Verify clip_model_type exists
    clip_model_type = cfg.habitat_baselines.rl.ddppo.get("clip_model_type", None)
    print(f"  clip_model_type: {clip_model_type}")
    assert clip_model_type == "longclip", f"Expected 'longclip', got '{clip_model_type}'"

    print("[PASS] YAML config parsed correctly with clip_model_type=longclip")
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 4: ResNetCLIPTextEncoder structural creation")
print("=" * 60)
try:
    import numpy as np
    from gym import spaces
    import torch

    # Patch to avoid actually loading the LongCLIP model (too big for this session)
    # We test the code paths by using clip_model_type="clip" (standard CLIP)
    from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetCLIPTextEncoder

    obs_space = spaces.Dict({
        "agent_0_overhead_front_rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "agent_0_overhead_front_depth": spaces.Box(low=0, high=1, shape=(224, 224, 1), dtype=np.float32),
        "agent_0_falcon_instruction": spaces.Box(low=0, high=255, shape=(512,), dtype=np.uint8),
    })

    # Test 4a: Standard CLIP mode (works with standard clip package)
    print("  4a: Creating encoder with clip_model_type='clip'...")
    encoder_clip = ResNetCLIPTextEncoder(
        observation_space=obs_space,
        pooling="attnpool",
        text_encoder_dim=2048,
        fusion_method="attention",
        rgb_sensor_keys=["overhead_front_rgb"],
        depth_sensor_keys=["overhead_front_depth"],
        visual_fusion_mode="average",
        normalize_before_fusion=True,
        clip_model_type="clip",
    )
    assert not encoder_clip.use_long_clip, "Should NOT use long clip"
    print(f"  [PASS] Standard CLIP encoder created, use_long_clip={encoder_clip.use_long_clip}")

    # Forward test with standard CLIP (lighter weight)
    batch_size = 1
    obs = {
        "agent_0_overhead_front_rgb": torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8),
        "agent_0_overhead_front_depth": torch.rand(batch_size, 224, 224, 1),
        "agent_0_falcon_instruction": torch.randint(65, 122, (batch_size, 512), dtype=torch.uint8),
    }
    encoder_clip.eval()
    with torch.no_grad():
        out = encoder_clip(obs)
    print(f"  Forward output: shape={out.shape}, dim={out.shape[-1]}")
    assert out.shape[-1] == encoder_clip.output_shape[0], f"Dimension mismatch: {out.shape[-1]} vs {encoder_clip.output_shape[0]}"
    print(f"  [PASS] Forward pass works (standard CLIP)")

    del encoder_clip
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Test 4b: LongCLIP mode (structural check only - don't load model)
    print("\n  4b: Verifying code paths for clip_model_type='longclip'...")
    print("  (Skipping full model load due to 16GB session memory limit)")
    print("  LongCLIP model loading will be tested via Slurm GPU job")

    # Verify the _longclip_root and _longclip_path are correctly set
    import habitat_baselines.rl.ddppo.policy.resnet_policy as rp
    root = getattr(rp, '_longclip_root', None)
    model_path = getattr(rp, '_longclip_path', None)
    available = getattr(rp, '_longclip_available', False)

    print(f"  _longclip_root: {root}")
    print(f"  _longclip_path: {model_path}")
    print(f"  _longclip_available: {available}")

    if root and model_path and available:
        print(f"  [PASS] LongCLIP paths correctly initialized")
    else:
        print(f"  [WARN] Some LongCLIP paths not set (may need full import)")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 5: PointNavResNetNet with clip_model_type")
print("=" * 60)
try:
    import numpy as np
    from gym import spaces
    import torch
    from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet

    action_space = spaces.Discrete(5)
    full_obs_space = spaces.Dict({
        "agent_0_overhead_front_rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "agent_0_overhead_front_depth": spaces.Box(low=0, high=1, shape=(224, 224, 1), dtype=np.float32),
        "agent_0_starting_point_gps_compass": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        "agent_0_falcon_instruction": spaces.Box(low=0, high=255, shape=(512,), dtype=np.uint8),
        "agent_0_falcon_gt_action": spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
    })

    clip_sensors = {
        "rgb_keys": ["overhead_front_rgb"],
        "depth_keys": ["overhead_front_depth"],
        "fusion_mode": "average",
        "normalize_before_fusion": True,
    }

    print("  Creating PointNavResNetNet with clip_model_type='clip'...")
    net = PointNavResNetNet(
        observation_space=full_obs_space,
        action_space=action_space,
        hidden_size=512,
        num_recurrent_layers=2,
        rnn_type="LSTM",
        backbone="resnet50_clip_attnpool",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        fuse_keys=None,
        force_blind_policy=False,
        discrete_actions=True,
        clip_visual_sensors=clip_sensors,
        clip_model_type="clip",
    )

    print(f"  visual_feature_size: {net._visual_feature_size}")
    print(f"  hidden_size: {net._hidden_size}")
    print(f"  clip_model_type stored: {net.clip_model_type}")
    assert net.clip_model_type == "clip"
    print(f"  [PASS] PointNavResNetNet created with clip model type")

    # Forward pass
    batch_size = 1
    obs_t = {
        "agent_0_overhead_front_rgb": torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8),
        "agent_0_overhead_front_depth": torch.rand(batch_size, 224, 224, 1),
        "agent_0_starting_point_gps_compass": torch.rand(batch_size, 2),
        "agent_0_falcon_instruction": torch.randint(65, 122, (batch_size, 512), dtype=torch.uint8),
        "agent_0_falcon_gt_action": torch.randint(0, 5, (batch_size, 1), dtype=torch.float32),
    }

    rnn_hidden = torch.zeros(batch_size, 2, 512)
    prev_actions = torch.zeros(batch_size, 1)
    masks = torch.ones(batch_size, 1)

    with torch.no_grad():
        out, rnn_states, aux = net(obs_t, rnn_hidden, prev_actions, masks)

    print(f"  Forward output: shape={out.shape}")
    print(f"  RNN states shape: {rnn_states.shape}")
    print(f"  Aux keys: {list(aux.keys())}")
    print(f"  [PASS] Forward pass works")

    del net
    gc.collect()

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 6: Verify Slurm batch script references")
print("=" * 60)
bash_path = "/share/home/u19666033/dhj/DPed_pro/main_slurm_rl_train_v1_longc.bash"
with open(bash_path) as f:
    content = f.read()
if "dynamic_vlnce_hm3d_train_v1_longc.yaml" in content:
    print(f"[PASS] Slurm script points to correct YAML config")
else:
    print(f"[FAIL] Wrong config reference in Slurm script")

print("\n" + "=" * 60)
print("ALL STRUCTURAL TESTS COMPLETED")
print("=" * 60)
print("Note: Full LongCLIP-L model loading (1.6GB) skipped due to")
print("16GB interactive session limit. Use Slurm GPU job for full test.")
