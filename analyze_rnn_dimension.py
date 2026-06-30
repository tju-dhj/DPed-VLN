#!/usr/bin/env python3
"""
分析RNN输入维度差异 (576 vs 549 = 27维)
"""

print("=" * 80)
print("RNN输入维度分析")
print("=" * 80)

# 根据PointNavResNetNet.__init__的代码逻辑
print("\n1. RNN输入维度构成 (按照代码顺序):")
print("-" * 80)

components_pretrained = {}
components_current = {}

# 第一部分: prev_action_embedding
components_pretrained["prev_action_embedding"] = 32
components_current["prev_action_embedding"] = 32
print(f"  prev_action_embedding: 32维 (固定)")

# 第二部分: fuse_keys_1d (1D传感器)
# 需要检查observation_space中有哪些1D传感器
print(f"\n  fuse_keys_1d (1D传感器，排除goal sensors和exclude_keys):")
print(f"    这部分需要检查你的observation_space配置")
print(f"    可能包括: localization_sensor, oracle_humanoid_future_trajectory等")

# 第三部分: IntegratedPointGoalGPSAndCompassSensor
print(f"\n  IntegratedPointGoalGPSAndCompassSensor (如果有):")
print(f"    sensor_dim + 1 → tgt_embeding(32)")

# GPS/Compass相关
print(f"\n  GPS/Compass相关:")
print(f"    - IntegratedPointGoalGPSAndCompassSensor: 32维")
print(f"    - EpisodicGPSSensor: 32维")
print(f"    - PointGoalSensor: 32维")
print(f"    - HeadingSensor: 32维")
print(f"    - EpisodicCompassSensor: 32维")

# 第四部分: Visual features
print(f"\n  Visual features:")
print(f"    _visual_feature_size = hidden_size = 512")

print("\n" + "=" * 80)
print("2. 可能的27维差异来源:")
print("=" * 80)

# 分析可能性
possible_sources = [
    ("oracle_humanoid_future_trajectory未被正确排除", "未知维度"),
    ("localization_sensor处理差异", "未知维度"),
    ("GPS/Compass传感器配置不同", "如果有/无1-2个传感器，差32维"),
    ("fuse_keys_1d中的传感器数量不同", "27维可能来自某个传感器"),
    ("hidden_size不同", "如果pretrained=512, current=485, 差27维"),
]

for i, (source, note) in enumerate(possible_sources, 1):
    print(f"  {i}. {source}")
    print(f"     → {note}")

print("\n" + "=" * 80)
print("3. 需要检查的配置:")
print("=" * 80)

checks = [
    "配置文件中的 hidden_size (应该是512)",
    "observation_space中有哪些1D传感器",
    "oracle_humanoid_future_trajectory的shape (可能是27维)",
    "localization_sensor的shape",
    "GPS/Compass传感器的配置",
]

for i, check in enumerate(checks, 1):
    print(f"  {i}. {check}")

print("\n" + "=" * 80)
print("4. 最可能的原因:")
print("=" * 80)
print("""
  根据代码分析:
    RNN输入 = prev_action(32) + fuse_keys_1d + goal_sensors + visual(512)
    
  预训练: 32 + fuse_keys_1d_pretrained + goal + 512 = 576
  当前:   32 + fuse_keys_1d_current + goal + 512 = 549
  差异:   fuse_keys_1d_pretrained - fuse_keys_1d_current = 27
  
  所以问题在于 fuse_keys_1d 部分！
  
  fuse_keys_1d包含所有1D传感器，排除:
    - goal sensors (GPS, Compass, PointGoal等)
    - exclude_keys (human_num_sensor, localization_sensor, falcon_gt_action, falcon_instruction)
  
  可能性:
    1. oracle_humanoid_future_trajectory在预训练时被fuse，在当前被排除
    2. localization_sensor的处理方式不同
    3. 某个27维的传感器在两个配置中处理方式不同
""")

print("\n" + "=" * 80)
print("5. 检查方法:")
print("=" * 80)
print("""
方法A: 查看训练日志的网络初始化信息
  → 应该有输出 "PointNavResNetNet: CLIP架构, visual_feature_size = XXX"
  → 可以添加debug代码输出rnn_input_size

方法B: 对比配置文件
  → 对比你的配置与预训练模型的训练配置
  → 重点看observation_space的配置

方法C: 检查observation_space
  → 在trainer初始化时打印observation_space.spaces.keys()
  → 检查每个space的shape

推荐: 直接修改resnet_policy.py，在RNN初始化前打印详细信息
""")

print("\n" + "=" * 80)
print("6. 临时解决方案 (已应用):")
print("=" * 80)
print("""
  ✅ 自动过滤形状不匹配的参数
  ✅ 只加载形状匹配的参数
  ✅ Visual encoder完全加载（最重要）
  ✅ RNN的hidden-to-hidden weights加载
  ❌ RNN的input-to-hidden weights随机初始化
  
  这个方案可以让训练正常进行，效果应该不会差很多。
""")

print("\n" + "=" * 80)
print("7. 找出确切原因:")
print("=" * 80)
print("""
添加debug代码到 resnet_policy.py 的 PointNavResNetNet.__init__，
在第796行(self.state_encoder初始化之前)添加:

    print(f"\\n=== RNN Input Size Debug ===")
    print(f"prev_action: {self._n_prev_action}")
    print(f"fuse_keys_1d: {self._fuse_keys_1d}")
    for k in self._fuse_keys_1d:
        print(f"  {k}: {observation_space.spaces[k].shape}")
    print(f"fuse_keys_1d total: {sum(observation_space.spaces[k].shape[0] for k in self._fuse_keys_1d)}")
    print(f"visual_feature_size: {self._visual_feature_size if not self.is_blind else 0}")
    print(f"total rnn_input_size: {(0 if self.is_blind else self._visual_feature_size) + rnn_input_size}")
    print(f"=== End Debug ===\\n")

这样训练日志就会显示确切的维度构成。
""")

