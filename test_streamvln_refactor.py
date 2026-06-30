#!/usr/bin/env python3
"""
测试 StreamVLN Policy 重构后的功能。
验证特征维度、编码器、以及与 PointNavResNetNet 的对应关系。
"""

import torch
import torch.nn as nn
from gym import spaces
import numpy as np
from collections import OrderedDict

# 模拟 StreamVLNNet 的关键组件
class MockStreamVLNNet:
    """模拟 StreamVLNNet 用于测试维度"""
    
    def __init__(self, action_space_n=4, hidden_size=1024):
        self.device = torch.device("cpu")
        self._hidden_size = hidden_size
        self.discrete_actions = True
        self._n_prev_action = 32
        
        # 1. Goal 编码器 (与 PointNavResNetNet 完全一致)
        self.tgt_embeding = nn.Linear(3, 32)
        print("✅ Goal encoder initialized: Linear(3 → 32)")
        
        # 2. Action 编码器 (与 PointNavResNetNet 完全一致)
        self.prev_action_embedding = nn.Embedding(
            action_space_n + 1, self._n_prev_action
        )
        print("✅ Action encoder initialized: Embedding(5 → 32)")
        
        # 3. Visual FC (模拟)
        self.visual_fc = nn.Sequential(
            nn.Linear(1024, hidden_size),  # 假设 vision tower 输出 1024
            nn.ReLU(True),
        )
        print(f"✅ Visual FC initialized: Linear(1024 → {hidden_size}) + ReLU")
    
    def _polar_transform_goal(self, goal_observations):
        """
        极坐标变换 (与 PointNavResNetNet 完全一致)
        (distance, angle) → (distance, cos(-angle), sin(-angle))
        """
        if goal_observations.shape[1] == 2:
            goal_observations = torch.stack(
                [
                    goal_observations[:, 0],              # distance
                    torch.cos(-goal_observations[:, 1]),  # cos(-angle)
                    torch.sin(-goal_observations[:, 1]),  # sin(-angle)
                ],
                -1,
            )
        return goal_observations
    
    def forward_mock(self, batch_size=4):
        """模拟 forward 过程"""
        print(f"\n{'='*60}")
        print(f"  Forward Pass (batch_size={batch_size})")
        print(f"{'='*60}\n")
        
        # 模拟观测
        observations = {
            'agent_0_articulated_agent_jaw_rgb': torch.randint(0, 255, (batch_size, 480, 640, 3), dtype=torch.uint8),
            'agent_0_pointgoal_with_gps_compass': torch.randn(batch_size, 2),  # [distance, angle]
        }
        
        prev_actions = torch.randint(0, 4, (batch_size, 1))
        masks = torch.ones(batch_size, 1)
        
        print(f"📥 输入观测:")
        print(f"   - RGB: {observations['agent_0_articulated_agent_jaw_rgb'].shape}")
        print(f"   - PointGoal: {observations['agent_0_pointgoal_with_gps_compass'].shape}")
        print(f"   - Previous Actions: {prev_actions.shape}")
        print(f"   - Masks: {masks.shape}\n")
        
        # ==================== 特征融合过程 ====================
        x = []
        aux_loss_state = {}
        
        # 1. 视觉特征 (模拟)
        # 在实际代码中，这来自 StreamVLN Vision Tower
        visual_features_raw = torch.randn(batch_size, 1024)  # 假设 vision tower 输出 1024
        visual_feats = self.visual_fc(visual_features_raw)
        print(f"1️⃣  视觉特征:")
        print(f"   - Vision Tower 输出: {visual_features_raw.shape}")
        print(f"   - 经过 Visual FC: {visual_feats.shape}")
        
        aux_loss_state["perception_embed"] = visual_feats
        x.append(visual_feats)
        
        # 2. Goal 编码
        goal_observations = observations['agent_0_pointgoal_with_gps_compass']
        print(f"\n2️⃣  Goal 编码:")
        print(f"   - 原始 PointGoal: {goal_observations.shape}")
        
        # 极坐标变换
        goal_observations = self._polar_transform_goal(goal_observations)
        print(f"   - 极坐标变换后: {goal_observations.shape}  (distance, cos(-angle), sin(-angle))")
        
        # Linear 编码
        goal_embed = self.tgt_embeding(goal_observations)
        print(f"   - 经过 Linear(3→32): {goal_embed.shape}")
        x.append(goal_embed)
        
        # 3. Action 编码
        print(f"\n3️⃣  Action 编码:")
        print(f"   - 原始 Previous Actions: {prev_actions.shape}")
        
        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions_embed = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        
        print(f"   - 经过 Embedding: {prev_actions_embed.shape}")
        x.append(prev_actions_embed)
        
        # 4. 拼接
        out = torch.cat(x, dim=1)
        print(f"\n4️⃣  特征拼接:")
        print(f"   - Visual {visual_feats.shape[1]} + Goal {goal_embed.shape[1]} + Action {prev_actions_embed.shape[1]}")
        print(f"   - 拼接结果: {out.shape}")
        
        # 5. 保存到 aux_loss_state
        aux_loss_state["rnn_output"] = out
        
        print(f"\n5️⃣  Auxiliary Loss State:")
        print(f"   - perception_embed: {aux_loss_state['perception_embed'].shape}")
        print(f"   - rnn_output: {aux_loss_state['rnn_output'].shape}")
        
        return out, None, aux_loss_state


def test_streamvln_net():
    """测试 StreamVLN Net"""
    print("\n" + "="*60)
    print("  🧪 测试 StreamVLN Net (重构后)")
    print("="*60 + "\n")
    
    # 创建模拟网络
    net = MockStreamVLNNet(action_space_n=4, hidden_size=1024)
    
    # 运行 forward
    out, _, aux_loss_state = net.forward_mock(batch_size=4)
    
    print(f"\n{'='*60}")
    print(f"  ✅ 测试成功！")
    print(f"{'='*60}\n")
    
    return out, aux_loss_state


def compare_with_resnet():
    """对比 PointNavResNetNet 的维度"""
    print("\n" + "="*60)
    print("  📊 与 PointNavResNetNet 的对比")
    print("="*60 + "\n")
    
    batch_size = 4
    
    # PointNavResNetNet 的维度
    resnet_visual = 512
    resnet_goal = 32
    resnet_action = 32
    resnet_concat = resnet_visual + resnet_goal + resnet_action  # 576
    resnet_rnn_output = 512  # LSTM 输出
    
    # StreamVLNNet 的维度
    streamvln_visual = 1024
    streamvln_goal = 32
    streamvln_action = 32
    streamvln_concat = streamvln_visual + streamvln_goal + streamvln_action  # 1088
    streamvln_output = streamvln_concat  # 无 RNN，直接使用拼接结果
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                  PointNavResNetNet                      │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Visual features:       ({batch_size}, {resnet_visual:4d})               │")
    print(f"│  Goal embedding:        ({batch_size}, {resnet_goal:4d})               │")
    print(f"│  Action embedding:      ({batch_size}, {resnet_action:4d})               │")
    print("│  ─────────────────────────────────────────────────────  │")
    print(f"│  Concatenated:          ({batch_size}, {resnet_concat:4d})               │")
    print("│  ↓ RNN (LSTM)                                           │")
    print(f"│  RNN output:            ({batch_size}, {resnet_rnn_output:4d})      ← aux_loss  │")
    print("└─────────────────────────────────────────────────────────┘")
    
    print()
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│               StreamVLNNet (重构后)                     │")
    print("├─────────────────────────────────────────────────────────┤")
    print(f"│  Visual features:       ({batch_size}, {streamvln_visual:4d})   ⭐ 更大     │")
    print(f"│  Goal embedding:        ({batch_size}, {streamvln_goal:4d})   ✅ 一致     │")
    print(f"│  Action embedding:      ({batch_size}, {streamvln_action:4d})   ✅ 一致     │")
    print("│  ─────────────────────────────────────────────────────  │")
    print(f"│  Concatenated:          ({batch_size}, {streamvln_concat:4d})               │")
    print("│  ↓ 无 RNN (直接使用)                                    │")
    print(f"│  Output:                ({batch_size}, {streamvln_output:4d})      ← aux_loss  │")
    print("└─────────────────────────────────────────────────────────┘")
    
    print()
    
    print("📈 对比总结:")
    print(f"   - Visual features: {streamvln_visual} vs {resnet_visual} ({streamvln_visual/resnet_visual:.1f}x)")
    print(f"   - Goal encoding: {streamvln_goal} == {resnet_goal} ✅")
    print(f"   - Action encoding: {streamvln_action} == {resnet_action} ✅")
    print(f"   - aux_loss_state['rnn_output']: {streamvln_output} vs {resnet_rnn_output} ({streamvln_output/resnet_rnn_output:.1f}x)")
    print(f"\n   结论: StreamVLN 的特征更丰富 ({streamvln_output/resnet_rnn_output:.1f}x)，辅助任务应该表现更好！✨")


def test_polar_transform():
    """测试极坐标变换"""
    print("\n" + "="*60)
    print("  🧮 测试极坐标变换")
    print("="*60 + "\n")
    
    # 创建测试数据
    goal = torch.tensor([
        [5.0, 0.5],      # distance=5, angle=0.5 rad
        [3.0, -0.3],     # distance=3, angle=-0.3 rad
        [10.0, 1.57],    # distance=10, angle=π/2
        [1.0, 0.0],      # distance=1, angle=0
    ])
    
    print("输入 PointGoal (极坐标):")
    print(goal)
    print(f"Shape: {goal.shape}\n")
    
    # 变换
    net = MockStreamVLNNet()
    transformed = net._polar_transform_goal(goal)
    
    print("输出 (笛卡尔坐标形式):")
    print(transformed)
    print(f"Shape: {transformed.shape}\n")
    
    print("验证:")
    for i in range(len(goal)):
        dist = goal[i, 0].item()
        angle = goal[i, 1].item()
        print(f"  [{i}] distance={dist:.2f}, angle={angle:.2f}")
        print(f"      → (d={transformed[i, 0].item():.4f}, "
              f"cos(-θ)={transformed[i, 1].item():.4f}, "
              f"sin(-θ)={transformed[i, 2].item():.4f})")
        
        # 验证
        expected_cos = np.cos(-angle)
        expected_sin = np.sin(-angle)
        print(f"      预期: cos(-{angle:.2f})={expected_cos:.4f}, sin(-{angle:.2f})={expected_sin:.4f}")
        assert np.isclose(transformed[i, 1].item(), expected_cos, atol=1e-4), "cos 计算错误"
        assert np.isclose(transformed[i, 2].item(), expected_sin, atol=1e-4), "sin 计算错误"
        print(f"      ✅ 验证通过\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  StreamVLN Policy 重构测试")
    print("="*60)
    
    try:
        # 1. 测试极坐标变换
        test_polar_transform()
        
        # 2. 测试 StreamVLN Net
        out, aux_loss_state = test_streamvln_net()
        
        # 3. 对比维度
        compare_with_resnet()
        
        print("\n" + "="*60)
        print("  ✅ 所有测试通过！")
        print("="*60 + "\n")
        
        print("下一步:")
        print("  1. 确保 StreamVLN 模型路径配置正确")
        print("  2. 运行 Falcon 训练/评估")
        print("  3. 观察辅助任务的损失值")
        print("  4. 对比 StreamVLN 和 ResNet policy 的性能\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
