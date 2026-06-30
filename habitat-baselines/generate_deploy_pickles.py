#!/usr/bin/env python3
"""
生成部署所需的 pickle 文件，完全不需要 Habitat 仿真器。

运行: cd /share/home/u19666033/dhj/DPed_pro/habitat-baselines && python generate_deploy_pickles.py
"""
import os
import sys
import pickle

os.chdir("/share/home/u19666033/dhj/DPed_pro/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-baselines")
sys.path.insert(0, "/share/home/u19666033/dhj/DPed_pro/habitat-lab")

from gym import spaces
import numpy as np

# ============================================================
# 1. observation_space: 复用已有的 pickle（或从零构造）
# ============================================================
obs_pkl = "observation_space.pkl"
act_pkl = "action_space.pkl"
orig_act_pkl = "orig_action_space.pkl"

if not os.path.exists(obs_pkl):
    # 构造最小 observation_space（匹配 robot_deploy_4a.yaml 中的 obs_keys）
    obs_space = spaces.Dict({
        "agent_0_overhead_front_rgb":     spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        "agent_0_overhead_front_depth":   spaces.Box(0, 10, (224, 224, 1), dtype=np.float32),
        "agent_0_falcon_instruction":     spaces.Box(0, 999, (512,), dtype=np.int64),
        "agent_0_starting_point_gps_compass": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        # 以下为多 agent 兼容字段（推理时不使用，但需存在）
        "agent_0_localization_sensor":    spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
        "agent_0_human_num_sensor":       spaces.Box(0, 100, (1,), dtype=np.int64),
        "agent_0_oracle_humanoid_future_trajectory": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        "agent_0_falcon_gt_action":       spaces.Box(-1, 1, (1,), dtype=np.float32),
        "agent_0_pointgoal_with_gps_compass": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
        "agent_0_third_rgb":              spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        "agent_0_third_depth":            spaces.Box(0, 10, (224, 224, 1), dtype=np.float32),
        "agent_0_articulated_agent_jaw_rgb": spaces.Box(0, 255, (224, 224, 3), dtype=np.uint8),
        "agent_0_articulated_agent_jaw_depth": spaces.Box(0, 10, (224, 224, 1), dtype=np.float32),
    })
    with open(obs_pkl, "wb") as f:
        pickle.dump(obs_space, f)
    print(f"Created {obs_pkl}")

# ============================================================
# 2. orig_action_space: 手动构造（完全不需要仿真器）
# ============================================================
# MultiAgentAccessMgr 需要:
#   - keys 以 "agent" 开头
#   - 每个 agent_N 的子 dict 通过 update_dict_with_agent_prefix 拆分
#   - agent_0 有 4 个离散动作: stop / move_forward / turn_left / turn_right
#   - agent_1~6 有 oracle_nav 动作
#
# 注意: 动作空间本身用什么类型不太重要，因为 create_action_space 会将
#        Dict 转为 Discrete(N) 或 Box，且推理时只用 policy_action_space。
#       关键是 Dict 的 key 数量和名称。

# 对于 agent_0: 4-action 设置
# oracle_nav_action 在 env 中返回 12 维 Box（连续控制），这里用 Box 模拟
oracle_nav_space = spaces.Box(-1.0, 1.0, (12,), dtype=np.float32)

# 构造 orig_action_space（key 格式与 config 中 habitat.task.actions 对应）
orig_action_dict = {
    # agent_0: 4 个离散动作 (stop, forward, left, right)
    "agent_0_discrete_stop": spaces.Discrete(2),
    "agent_0_discrete_move_forward": spaces.Discrete(2),
    "agent_0_discrete_turn_left": spaces.Discrete(2),
    "agent_0_discrete_turn_right": spaces.Discrete(2),
    # agent_1~6: oracle nav 动作
    "agent_1_oracle_nav_randcoord_action_obstacle": oracle_nav_space,
    "agent_2_oracle_nav_randcoord_action_obstacle": oracle_nav_space,
    "agent_3_oracle_nav_randcoord_action_obstacle": oracle_nav_space,
    "agent_4_oracle_nav_randcoord_action_obstacle": oracle_nav_space,
    "agent_5_oracle_nav_randcoord_action_obstacle": oracle_nav_space,
    "agent_6_oracle_nav_randcoord_action_obstacle": oracle_nav_space,
}
orig_action_space = spaces.Dict(orig_action_dict)

with open(orig_act_pkl, "wb") as f:
    pickle.dump(orig_action_space, f)
print(f"Created {orig_act_pkl} with {len(orig_action_dict)} action slots")

# ============================================================
# 3. action_space: transform 后的统一动作空间（如果不存在则构造）
# ============================================================
if not os.path.exists(act_pkl):
    # apply_obs_transforms_obs_space 把 multi-agent Dict 转成统一空间
    # 对于 4-action，最终是 Discrete(4)；对于 6-action，是 Discrete(6)
    # 但这里先设一个兼容值，实际上 action_space 在推理中不直接使用
    from habitat_baselines.utils.common import get_action_space_info
    action_space = spaces.Discrete(4)  # 4-action setup
    with open(act_pkl, "wb") as f:
        pickle.dump(action_space, f)
    print(f"Created {act_pkl}")

print("\nAll pickle files ready. Server can now start without simulator.")
print(f"  {obs_pkl}")
print(f"  {act_pkl}")
print(f"  {orig_act_pkl}")
