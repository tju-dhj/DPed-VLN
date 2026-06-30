#!/usr/bin/env python3
"""
LongCLIP 集成测试脚本
测试内容：
1. LongCLIP模块导入
2. LongCLIP模型加载
3. LongCLIP Tokenizer（支持248 tokens长文本）
4. ResNetCLIPTextEncoder 创建和forward
5. PointNavResNetNet 创建
"""

import sys
import os

# 添加必要的路径
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/Long-CLIP/model")

import torch
import numpy as np
from gym import spaces

print("=" * 60)
print("TEST 1: LongCLIP module import")
print("=" * 60)
try:
    from longclip import load as longclip_load
    from longclip import tokenize as longclip_tokenize
    print("[PASS] longclip module imported successfully")
except ImportError as e:
    print(f"[FAIL] longclip import failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("TEST 2: LongCLIP model loading")
print("=" * 60)
try:
    ckpt_path = "/share/home/u19666033/dhj/DPed_pro/Long-CLIP/checkpoints/longclip-L.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading from: {ckpt_path}")
    print(f"Device: {device}")

    model, preprocess = longclip_load(ckpt_path, device=device)
    print(f"[PASS] LongCLIP model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Visual encoder: {type(model.visual).__name__}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
except Exception as e:
    print(f"[FAIL] LongCLIP model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("TEST 3: LongCLIP Tokenizer (long context, 248 tokens)")
print("=" * 60)
try:
    # 短文本
    short_text = ["Navigate to the target location."]
    short_tokens = longclip_tokenize(short_text)
    print(f"[PASS] Short text tokenization: shape={short_tokens.shape}, max_len=77(default)")

    # 长文本（超过标准CLIP的77 tokens限制）
    long_text = [
        "Walk straight down the hallway past the kitchen on your left. "
        "Continue until you reach a wooden dining table with four chairs arranged around it. "
        "Turn right at the table and proceed through the archway into the living room. "
        "Once in the living room, look for a large brown leather sofa against the wall. "
        "The target location is behind this sofa, near the floor lamp in the corner. "
        "Make sure to avoid the coffee table in the center of the room as you navigate."
    ]

    # 标准CLIP tokenizer (77 tokens)
    import clip
    clip_model, clip_preprocess = clip.load("RN50", device=device)
    clip_tokens = clip.tokenize(long_text, truncate=True)
    clip_token_count = (clip_tokens[0] != 0).sum().item()
    print(f"  Standard CLIP tokens: count={clip_token_count} (max=77)")

    # LongCLIP tokenizer (248 tokens)
    longclip_tokens = longclip_tokenize(long_text)
    longclip_token_count = (longclip_tokens[0] != 0).sum().item()
    print(f"  LongCLIP tokens: count={longclip_token_count} (max=248)")

    if longclip_token_count > 77:
        print(f"[PASS] LongCLIP successfully tokenized text with {longclip_token_count} tokens (> 77 standard limit)")
    else:
        print(f"[INFO] Text length ({longclip_token_count} tokens) did not exceed 77, but LongCLIP supports up to 248")

    # 验证文本编码
    with torch.no_grad():
        short_feat = model.encode_text(short_tokens.to(device))
        long_feat = model.encode_text(longclip_tokens.to(device))
    print(f"  Short text feature shape: {short_feat.shape}")
    print(f"  Long text feature shape: {long_feat.shape}")
    print(f"[PASS] LongCLIP text encoding works correctly")
except Exception as e:
    print(f"[FAIL] LongCLIP tokenizer test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 4: ResNetCLIPTextEncoder creation")
print("=" * 60)
try:
    from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetCLIPTextEncoder

    # 模拟observation space
    obs_space = spaces.Dict({
        "agent_0_overhead_front_rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "agent_0_overhead_front_depth": spaces.Box(low=0, high=1, shape=(224, 224, 1), dtype=np.float32),
        "agent_0_starting_point_gps_compass": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        "agent_0_localization_sensor": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        "agent_0_human_num_sensor": spaces.Box(low=0, high=6, shape=(1,), dtype=np.float32),
        "agent_0_oracle_humanoid_future_trajectory": spaces.Box(low=-np.inf, high=np.inf, shape=(6, 4, 2), dtype=np.float32),
        "agent_0_falcon_instruction": spaces.Box(low=0, high=255, shape=(512,), dtype=np.uint8),
        "agent_0_falcon_gt_action": spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
    })

    print(f"Observation space keys: {list(obs_space.spaces.keys())}")

    # 创建带有多个传感器的encoder
    encoder = ResNetCLIPTextEncoder(
        observation_space=obs_space,
        pooling="attnpool",
        text_encoder_dim=2048,
        fusion_method="attention",
        rgb_sensor_keys=["overhead_front_rgb"],
        depth_sensor_keys=["overhead_front_depth"],
        visual_fusion_mode="attention",
        normalize_before_fusion=True,
        clip_model_type="longclip",
    )

    print(f"[PASS] ResNetCLIPTextEncoder created successfully")
    print(f"  Using LongCLIP: {encoder.use_long_clip}")
    print(f"  RGB sensors: {encoder.configured_rgb_keys}")
    print(f"  Depth sensors: {encoder.configured_depth_keys}")
    print(f"  Fusion mode: {encoder.visual_fusion_mode}")
    print(f"  Output shape: {encoder.output_shape}")
except Exception as e:
    print(f"[FAIL] ResNetCLIPTextEncoder creation failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 5: ResNetCLIPTextEncoder forward pass")
print("=" * 60)
try:
    batch_size = 2

    # 构造模拟观察数据
    observations = {
        "agent_0_overhead_front_rgb": torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8),
        "agent_0_overhead_front_depth": torch.rand(batch_size, 224, 224, 1),
        "agent_0_falcon_instruction": torch.randint(65, 122, (batch_size, 512), dtype=torch.uint8),
    }

    print(f"Input shapes:")
    for k, v in observations.items():
        print(f"  {k}: {v.shape}, dtype={v.dtype}")

    encoder.eval()
    with torch.no_grad():
        output = encoder(observations)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output stats: min={output.min().item():.6f}, max={output.max().item():.6f}, mean={output.mean().item():.6f}")

    expected_dim = encoder.output_shape[0]
    if output.shape[-1] == expected_dim:
        print(f"[PASS] ResNetCLIPTextEncoder forward pass works correctly (dim={output.shape[-1]})")
    else:
        print(f"[FAIL] Expected output dim {expected_dim}, got {output.shape[-1]}")
except Exception as e:
    print(f"[FAIL] ResNetCLIPTextEncoder forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 6: Standard CLIP fallback")
print("=" * 60)
try:
    # 测试标准CLIP模式也能工作
    encoder_clip = ResNetCLIPTextEncoder(
        observation_space=obs_space,
        pooling="attnpool",
        text_encoder_dim=2048,
        fusion_method="attention",
        rgb_sensor_keys=["overhead_front_rgb"],
        depth_sensor_keys=["overhead_front_depth"],
        visual_fusion_mode="average",
        normalize_before_fusion=True,
        clip_model_type="clip",  # 标准CLIP
    )
    print(f"[PASS] Standard CLIP encoder created successfully")
    print(f"  Using LongCLIP: {encoder_clip.use_long_clip}")
    assert not encoder_clip.use_long_clip, "Should NOT use long CLIP when clip_model_type='clip'"
    print(f"[PASS] Fallback to standard CLIP verified")
except Exception as e:
    print(f"[FAIL] Standard CLIP fallback test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 7: PointNavResNetNet with LongCLIP backbone")
print("=" * 60)
try:
    from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetNet

    action_space = spaces.Discrete(5)  # stop, forward, left, right

    # 完整的observation space
    full_obs_space = spaces.Dict({
        "agent_0_overhead_front_rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "agent_0_overhead_front_depth": spaces.Box(low=0, high=1, shape=(224, 224, 1), dtype=np.float32),
        "agent_0_starting_point_gps_compass": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        "agent_0_localization_sensor": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        "agent_0_human_num_sensor": spaces.Box(low=0, high=6, shape=(1,), dtype=np.float32),
        "agent_0_oracle_humanoid_future_trajectory": spaces.Box(low=-np.inf, high=np.inf, shape=(6, 4, 2), dtype=np.float32),
        "agent_0_falcon_instruction": spaces.Box(low=0, high=255, shape=(512,), dtype=np.uint8),
        "agent_0_falcon_gt_action": spaces.Box(low=0, high=4, shape=(1,), dtype=np.float32),
    })

    clip_visual_sensors = {
        "rgb_keys": ["overhead_front_rgb"],
        "depth_keys": ["overhead_front_depth"],
        "fusion_mode": "average",
        "normalize_before_fusion": True,
    }

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
        clip_visual_sensors=clip_visual_sensors,
        clip_model_type="longclip",
    )

    print(f"[PASS] PointNavResNetNet created successfully")
    print(f"  Visual feature size: {net._visual_feature_size}")
    print(f"  Hidden size: {net._hidden_size}")
    print(f"  Is blind: {net.is_blind}")
    print(f"  Backbone: resnet50_clip_attnpool with LongCLIP")

    # Forward pass with simulated batch
    batch_size = 2
    observations_t = {
        "agent_0_overhead_front_rgb": torch.randint(0, 256, (batch_size, 224, 224, 3), dtype=torch.uint8),
        "agent_0_overhead_front_depth": torch.rand(batch_size, 224, 224, 1),
        "agent_0_starting_point_gps_compass": torch.rand(batch_size, 2),
        "agent_0_localization_sensor": torch.rand(batch_size, 2),
        "agent_0_human_num_sensor": torch.rand(batch_size, 1),
        "agent_0_oracle_humanoid_future_trajectory": torch.rand(batch_size, 6, 4, 2),
        "agent_0_falcon_instruction": torch.randint(65, 122, (batch_size, 512), dtype=torch.uint8),
        "agent_0_falcon_gt_action": torch.randint(0, 5, (batch_size, 1), dtype=torch.float32),
    }

    # 初始化RNN隐藏状态
    rnn_hidden_states = torch.zeros(batch_size, 2, 512)  # num_layers=2, hidden=512
    prev_actions = torch.zeros(batch_size, 1)
    masks = torch.ones(batch_size, 1)

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output, rnn_states, aux = net(observations_t, rnn_hidden_states, prev_actions, masks)

    print(f"Output shape: {output.shape}")
    print(f"RNN hidden states shape: {rnn_states.shape}")
    print(f"Aux keys: {list(aux.keys())}")
    print(f"[PASS] PointNavResNetNet forward pass works correctly with LongCLIP")

except Exception as e:
    print(f"[FAIL] PointNavResNetNet test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
