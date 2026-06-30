#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
DPed_pro_expert.py  ——  6动作空间专家动作规划器

动作空间定义：
    0 : STOP     — 停止并终止当前轨迹（到达目标时发出）
    1 : FORWARD  — 前进
    2 : TURN_LEFT  — 向左转
    3 : TURN_RIGHT — 向右转
    4 : PAUSE    — 原地暂停（不终止轨迹），用于礼让行人
    5 : BACKWARD — 后退，用于紧急危险躲避

规划器设计理念（SFM + ORCA 混合策略）：
    ─────────────────────────────────────────────────────────────
    【参考文献】
    · Helbing & Molnár (1995): Social Force Model for Pedestrian Dynamics.
      机器人被驱动力（目标吸引）和排斥力（行人排斥）的合力控制。
    · Van den Berg et al. (2008): Reciprocal Velocity Obstacles (RVO/ORCA).
      在 SFM 基础上加入速度-障碍锥约束，保证无碰速度。
    · DPed_VLN Benchmark 论文关键设计原则：
      - 机器人优先让行（礼让行人为主），因此引入 PAUSE(4) 动作。
      - 行人间距过近时可适度后退 BACKWARD(5) 以开辟安全缓冲区。
    ─────────────────────────────────────────────────────────────

    决策优先级（从高到低）：
        P0 到达目标  → STOP(0)
        P1 极度危险  < EMERGENCY_DIST: → BACKWARD(5) 或紧急转向
        P2 礼让等待  行人正面来袭 + 距离 < YIELD_DIST: → PAUSE(4)
        P3 SFM 避障  计算合力方向角，偏离目标方向时转向
        P4 正常导航  Oracle 最短路径方向 → FORWARD/TURN_LEFT/TURN_RIGHT

    SFM 合力公式（在 xz 平面）：
        F_drive  = (v_desired * e_goal - v_current) / τ
            τ = 0.5s（松弛时间）
            e_goal = 目标方向单位向量
            v_desired = robot_max_speed

        F_repulse_i = A * exp( (r_i - d_i) / B ) * n_i
            A = 2.0  （SFM 排斥力强度，单位 m/s²）
            B = 0.3  （SFM 特征距离，单位 m）
            r_i = combined_radius（机器人半径 + 行人半径 ≈ 0.6m）
            d_i = 当前到行人 i 的距离
            n_i = 从行人指向机器人的单位向量

        F_total = F_drive + Σ F_repulse_i

    ORCA 速度障碍锥约束（叠加在 SFM 之上）：
        当行人在 WARNING_DIST 内时，计算 RVO 速度锥，
        将合力速度投影到 ORCA 可行速度集合中，
        避免纯 SFM 因合力惯性导致的碰撞残差。

    动作选择映射：
        F_total 方向角 → angle_diff_to_current
        |angle_diff| < TURN_THRESH → FORWARD(1)
        angle_diff  > +TURN_THRESH → TURN_LEFT(2)
        angle_diff  < -TURN_THRESH → TURN_RIGHT(3)
        + PAUSE(4) / BACKWARD(5) 覆写条件（见上方优先级）

注册方式：
    @baseline_registry.register_trainer(name="expert_data_collector_6action")
    类名：ExpertDataCollector6Action

作者：基于 expert_data_collector_v3.py 升级，新增 SFM + PAUSE/BACKWARD 支持
"""

# ──────────────────────────── 标准库 ────────────────────────────
import os
import time
import json
import math
import pathlib
import contextlib
from typing import Dict, Any, Optional, List, Tuple

# ──────────────────────────── 第三方库 ──────────────────────────
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from PIL import Image

# ──────────────────────────── Habitat ───────────────────────────
import habitat_baselines.rl.multi_agent  # noqa: F401
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.utils import profiling_wrapper
from habitat.tasks.rearrange.utils import get_angle_to_pos

from habitat_baselines.common import VectorEnvFactory
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import get_writer
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.utils.common import (
    batch_obs,
    inference_mode,
    is_continuous_action_space,
    generate_video,
)
from habitat_baselines.utils.timing import g_timer
from habitat.utils.visualizations.utils import observations_to_image
from habitat.utils.visualizations import maps

# ────────────────────────── 导入 6 动作注册 ─────────────────────
import falcon.additional_sensor  # noqa: F401  ← 注册传感器
import falcon.additional_action  # noqa: F401  ← 注册 6 个离散动作
import falcon.additional_metric  # noqa: F401  ← 注册度量

# ═══════════════════════════════════════════════════════════════════
#  全局超参数（SFM + ORCA 混合策略）
# ═══════════════════════════════════════════════════════════════════

# ── 行人距离阈值（米）── [修复v3: 大幅收紧阈值，提前避障]
EMERGENCY_DIST   = 0.45   # 极度危险：触发 BACKWARD(5)
CRITICAL_DIST    = 1.00   # 危险：强制后退或转向（对齐 v3 的 1.0m）
YIELD_DIST       = 1.20   # 礼让区：行人迎面靠近 → PAUSE(4)（收紧，加快导航）
WARNING_DIST     = 2.50   # 警告区：开始 SFM+ORCA 避障（增大到2.5m）
SAFE_DIST        = 5.00   # 安全区：纯 Oracle 导航

# ── PAUSE 和防卡死参数 ──
MAX_CONSECUTIVE_PAUSE = 3   # 最大连续 PAUSE 次数，超过后强制转向
STUCK_THRESHOLD = 0.10      # 距离变化阈值（米），低于此值视为"卡住"（收紧）
STUCK_CHECK_WINDOW = 10     # 检查窗口（步数）（缩短，快速响应）
STUCK_FORCE_TURN = True     # 卡住时强制转向

# ── SFM 参数 ── [修复v3: 降低速度增大安全裕量]
SFM_A            = 2.0    # 排斥力强度 (m/s²)
SFM_B            = 0.25   # 特征距离 (m) - 减小影响范围
SFM_RELAX_TAU    = 0.60   # 松弛时间 (s) - 增大使速度变化更平缓
SFM_COMBINED_R   = 0.50   # 机器人+行人组合半径 (m)
ROBOT_MAX_SPEED  = 0.20   # 机器人期望速度 (m/s) - 降低更安全

# ── ORCA 参数 ── [修复v3: 增大安全裕量]
ORCA_TIME_HORIZON = 8.0   # ORCA 预测时间 (s) - 延长预测时间（对齐 v3）
ORCA_AGENT_RADIUS = 0.30  # 单个智能体半径 (m) - 增大安全区

# ── 动作决策阈值 ──
TURN_THRESHOLD   = np.deg2rad(7.5)  # 转向阈值（对齐 v3）
MAX_REPEATED_TURNS = 3               # 连续转向次数上限（对齐 v3）

# ── 目标距离阈值（到达即 STOP） ──
GOAL_RADIUS_DEFAULT = 1.5  # 到达目标距离阈值

# ── 行人迎面判断参数 ── [修复v3: 收紧判断条件]
HEAD_ON_DOT_THRESH   = 0.5   # 行人速度与机器人正向点积阈值
HEAD_ON_DIST_THRESH  = 0.90  # 迎面时的礼让生效距离（米）

# ── 预测性避障参数 ──
PREDICTION_HORIZON = 2.0    # 行人位置预测时间窗口 (s)
TTC_THRESHOLD = 2.5         # 碰撞时间阈值 (s)，低于此值强制避障
COLLISION_PROB_THRESHOLD = 0.3  # 碰撞概率阈值，超过则强制PAUSE/BACKWARD


# ═══════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════

def _to_numpy(tensor) -> np.ndarray:
    """安全地将 Tensor / ndarray 转换为 numpy。"""
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


def _normalize_angle(angle: float) -> float:
    """将角度归一化到 (-π, π]。"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def _xz(pos: np.ndarray) -> np.ndarray:
    """提取 3D 位置的 (x, z) 分量，返回形状 (2,) 数组。"""
    return np.array([pos[0], pos[2]], dtype=np.float64)


def compute_pedestrian_trajectory(
    ped_pos: np.ndarray,
    ped_vel_xz: np.ndarray,
    prediction_horizon: float = PREDICTION_HORIZON,
    dt: float = 0.25,
) -> List[np.ndarray]:
    """
    预测行人在未来一段时间内的轨迹位置。

    Args:
        ped_pos: 行人当前位置 (x, z)
        ped_vel_xz: 行人当前速度 (vx, vz)
        prediction_horizon: 预测时间窗口 (s)
        dt: 采样间隔 (s)

    Returns:
        预测位置列表，每个元素形状 (2,)
    """
    ped_speed = np.linalg.norm(ped_vel_xz)
    if ped_speed < 0.01:
        return [ped_pos.copy()]

    positions = []
    num_steps = int(prediction_horizon / dt)
    for t in range(num_steps + 1):
        time_offset = t * dt
        future_pos = ped_pos + ped_vel_xz * time_offset
        positions.append(future_pos)
    return positions


def compute_time_to_collision(
    robot_pos: np.ndarray,
    robot_vel_xz: np.ndarray,
    robot_radius: float,
    ped_pos: np.ndarray,
    ped_vel_xz: np.ndarray,
    ped_radius: float = 0.3,
) -> Tuple[float, float]:
    """
    计算机器人和行人之间的碰撞时间（TTC）。

    Args:
        robot_pos: 机器人当前位置 (x, z)
        robot_vel_xz: 机器人当前速度 (vx, vz)
        robot_radius: 机器人半径
        ped_pos: 行人当前位置 (x, z)
        ped_vel_xz: 行人速度 (vx, vz)
        ped_radius: 行人半径

    Returns:
        (ttc, min_dist): 碰撞时间和最近距离
                        ttc = inf 表示不会碰撞
    """
    rel_pos = robot_pos - ped_pos
    rel_vel = robot_vel_xz - ped_vel_xz
    combined_radius = robot_radius + ped_radius

    min_dist = np.linalg.norm(rel_pos) - combined_radius

    a = np.dot(rel_vel, rel_vel)
    if a < 1e-6:
        return float("inf"), min_dist

    b = 2.0 * np.dot(rel_pos, rel_vel)
    c = np.dot(rel_pos, rel_pos) - combined_radius * combined_radius

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return float("inf"), min_dist

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    ttc = float("inf")
    if t1 > 0:
        ttc = t1
    elif t2 > 0:
        ttc = t2

    return ttc, min_dist


def compute_collision_probability(
    robot_pos: np.ndarray,
    robot_fwd: np.ndarray,
    ped_pos: np.ndarray,
    ped_vel_xz: np.ndarray,
    combined_radius: float = SFM_COMBINED_R,
) -> float:
    """
    估算在下一时刻发生碰撞的概率（简化模型）。

    考虑因素：
    1. 行人相对机器人的方向（正前方更危险）
    2. 行人速度方向（朝向机器人更危险）
    3. 当前距离（越近概率越高）

    Args:
        robot_pos: 机器人位置 (x, z)
        robot_fwd: 机器人前进方向单位向量 (x, z)
        ped_pos: 行人位置 (x, z)
        ped_vel_xz: 行人速度 (vx, vz)
        combined_radius: 组合半径

    Returns:
        碰撞概率 [0, 1]
    """
    to_ped = ped_pos - robot_pos
    dist = np.linalg.norm(to_ped)

    if dist < 1e-4:
        return 1.0

    to_ped_norm = to_ped / dist

    ped_speed = np.linalg.norm(ped_vel_xz)
    if ped_speed < 0.01:
        return 0.0

    ped_dir = ped_vel_xz / ped_speed

    angle_to_robot = math.acos(np.clip(np.dot(to_ped_norm, ped_dir), -1.0, 1.0))

    dot = np.dot(robot_fwd, to_ped_norm)
    frontal_factor = max(0.0, dot)

    collision_dir_factor = 1.0 - (angle_to_robot / math.pi)

    speed_factor = min(1.0, ped_speed / 1.0)

    dist_factor = max(0.0, 1.0 - dist / (combined_radius * 5.0))

    prob = frontal_factor * collision_dir_factor * speed_factor * dist_factor

    return min(1.0, prob)


def create_agent0_video_frame(observation: Dict, info: Dict = None) -> np.ndarray:
    """创建仅包含 agent_0 第一视角、第三视角和俯视图的视频帧。"""
    if info is None:
        info = {}
    render_obs_images = []

    for key in ["agent_0_overhead_front_rgb", "agent_0_articulated_agent_jaw_rgb"]:
        if key in observation:
            img = _to_numpy(observation[key])
            if img.dtype != np.uint8:
                img = (img * 255.0).astype(np.uint8)
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=2)
            render_obs_images.append(img)
            break

    third_key = "agent_0_third_rgb"
    if third_key in observation:
        img = _to_numpy(observation[third_key])
        if img.dtype != np.uint8:
            img = (img * 255.0).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        render_obs_images.append(img)

    if len(render_obs_images) == 0:
        return observations_to_image(observation, info)

    shapes_equal = len(set(x.shape for x in render_obs_images)) == 1
    render_frame = (
        np.concatenate(render_obs_images, axis=1)
        if shapes_equal
        else _tile_images(render_obs_images)
    )

    if info.get("collisions", {}).get("is_collision", False):
        from habitat.utils.visualizations.utils import draw_collision
        render_frame = draw_collision(render_frame)

    if "top_down_map" in info:
        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], render_frame.shape[0]
        )
        render_frame = np.concatenate((render_frame, top_down_map), axis=1)

    return render_frame


def _tile_images(images):
    from habitat.utils.visualizations.utils import tile_images
    return tile_images(images)


def save_to_disk(
    rgb, depth, third_rgb, human_num, action,
    distance_to_goal, ep_id, scene_id,
    pedestrian_in_view=None, trajectories=None,
    split="train", data_folder="data/collect_data", merge_ep=True,
):
    """保存单条轨迹数据到磁盘（与 v3 版本格式兼容）。"""
    scene_name = pathlib.Path(scene_id).stem
    DATA_ROOT = pathlib.Path(data_folder) / split / scene_name

    for sub in ["rgb", "depth", "third_rgb", "human_num", "action", "distance_to_goal"]:
        os.makedirs(DATA_ROOT / ep_id / sub, exist_ok=True)

    n = rgb.shape[0]
    for i in range(n):
        Image.fromarray(rgb[i]).save(DATA_ROOT / ep_id / "rgb" / f"{i}_0.jpg")

    for i in range(n):
        img_u16 = (depth[i].squeeze() * 1000).astype(np.uint16)
        Image.fromarray(img_u16).save(DATA_ROOT / ep_id / "depth" / f"{i}_0.png")

    if third_rgb is not None:
        for i in range(n):
            Image.fromarray(third_rgb[i]).save(DATA_ROOT / ep_id / "third_rgb" / f"{i}_0.jpg")

    with open(DATA_ROOT / ep_id / "human_num" / "0.json", "w") as f:
        json.dump(human_num.tolist(), f)
    with open(DATA_ROOT / ep_id / "action" / "0.json", "w") as f:
        json.dump(action, f)
    with open(DATA_ROOT / ep_id / "distance_to_goal" / "0.json", "w") as f:
        json.dump(distance_to_goal, f)

    if pedestrian_in_view is not None:
        os.makedirs(DATA_ROOT / ep_id / "pedestrian_in_view", exist_ok=True)
        with open(DATA_ROOT / ep_id / "pedestrian_in_view" / "0.json", "w") as f:
            json.dump(pedestrian_in_view, f)

    if trajectories is not None:
        os.makedirs(DATA_ROOT / ep_id / "trajectories", exist_ok=True)
        with open(DATA_ROOT / ep_id / "trajectories" / "0.json", "w") as f:
            json.dump(trajectories, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  SFM 社会力计算核心
# ═══════════════════════════════════════════════════════════════════

def compute_sfm_force(
    robot_pos_xz: np.ndarray,
    robot_vel_xz: np.ndarray,
    goal_dir_xz: np.ndarray,
    ped_positions_xz: List[np.ndarray],
    ped_velocities_xz: List[np.ndarray],
    max_speed: float = ROBOT_MAX_SPEED,
    A: float = SFM_A,
    B: float = SFM_B,
    tau: float = SFM_RELAX_TAU,
    combined_radius: float = SFM_COMBINED_R,
) -> np.ndarray:
    """
    计算 SFM（社会力模型）合力向量（xz 平面）。

    参考 Helbing & Molnár (1995) 公式：
        F_total = F_drive + Σ F_repulse_i

    F_drive  = (max_speed * e_goal - v_current) / tau
    F_repulse_i = A * exp((combined_radius - dist_i) / B) * n_i
        n_i = 从行人 i 指向机器人的单位向量

    Args:
        robot_pos_xz:      机器人位置 (x, z)
        robot_vel_xz:      机器人当前速度 (vx, vz)
        goal_dir_xz:       归一化目标方向向量 (x, z)
        ped_positions_xz:  行人位置列表，每项 (x, z)
        ped_velocities_xz: 行人速度列表，每项 (vx, vz)
        max_speed:         机器人期望速度（目标引力速度参考）
        A, B:              SFM 排斥力强度与特征距离
        tau:               松弛时间 (s)
        combined_radius:   机器人+行人组合半径

    Returns:
        合力向量 (fx, fz)，形状 (2,)
    """
    # 驱动力：将当前速度驱动到目标方向最大速度
    v_desired = max_speed * goal_dir_xz
    f_drive = (v_desired - robot_vel_xz) / tau  # shape (2,)

    # 排斥力：对每个行人叠加 SFM 排斥力
    f_repulse = np.zeros(2, dtype=np.float64)
    for ped_xz, ped_vel_xz in zip(ped_positions_xz, ped_velocities_xz):
        diff = robot_pos_xz - ped_xz          # 从行人指向机器人
        dist = np.linalg.norm(diff)
        if dist < 1e-4:
            # 极端情况：完全重叠，选择随机方向推开
            diff = np.random.randn(2)
            dist = np.linalg.norm(diff) + 1e-6

        n_i = diff / dist                      # 单位方向向量
        # Helbing 排斥力：随距离指数衰减
        force_mag = A * math.exp((combined_radius - dist) / B)
        f_repulse += force_mag * n_i

    f_total = f_drive + f_repulse
    return f_total


# ═══════════════════════════════════════════════════════════════════
#  ORCA / RVO 速度障碍约束（叠加在 SFM 之上）
# ═══════════════════════════════════════════════════════════════════

def compute_orca_avoidance_velocity(
    robot_pos: np.ndarray,
    robot_vel_xz: np.ndarray,
    ped_positions: List[np.ndarray],
    ped_rotations: List[float],
    ped_velocities: List[np.ndarray],
    max_speed: float = ROBOT_MAX_SPEED,
    time_horizon: float = ORCA_TIME_HORIZON,
    combined_radius: float = SFM_COMBINED_R,
) -> np.ndarray:
    """
    基于 ORCA/RVO 计算无碰避障速度（xz 平面）。

    参考 Van den Berg et al. (2008): RVO / ORCA。
    对每个行人计算速度障碍锥，将期望速度投影到可行速度区域。

    Args:
        robot_pos:      机器人 3D 位置，仅取 xz
        robot_vel_xz:   当前机器人速度 (vx, vz)
        ped_positions:  行人 3D 位置列表
        ped_rotations:  行人朝向角（弧度）列表
        ped_velocities: 行人速度列表，每项 (vx, vz) or scalar
        max_speed:      最大速度
        time_horizon:   ORCA 预测时间窗口 (s)
        combined_radius: 两智能体组合半径

    Returns:
        调整后的速度向量 (vx, vz)，已裁剪到 max_speed
    """
    if not ped_positions:
        return robot_vel_xz.copy()

    robot_xz = _xz(robot_pos)
    adjusted_vel = robot_vel_xz.copy()
    combined_correction = np.zeros(2, dtype=np.float64)

    for ped_pos, ped_rot, ped_vel in zip(ped_positions, ped_rotations, ped_velocities):
        ped_xz = _xz(ped_pos)

        # 行人速度向量
        if np.ndim(ped_vel) == 0 or (hasattr(ped_vel, "__len__") and len(ped_vel) == 1):
            speed_scalar = float(ped_vel[0]) if hasattr(ped_vel, "__len__") else float(ped_vel)
            ped_vel_xz = np.array([
                speed_scalar * math.sin(ped_rot),
                speed_scalar * math.cos(ped_rot),
            ])
        else:
            ped_vel_arr = np.asarray(ped_vel, dtype=np.float64)
            if len(ped_vel_arr) >= 2:
                ped_vel_xz = ped_vel_arr[:2]
            else:
                speed_scalar = float(ped_vel_arr[0])
                ped_vel_xz = np.array([
                    speed_scalar * math.sin(ped_rot),
                    speed_scalar * math.cos(ped_rot),
                ])

        # 相对位置与速度
        rel_pos = robot_xz - ped_xz          # 机器人 - 行人
        rel_vel = adjusted_vel - ped_vel_xz  # 相对速度
        dist = np.linalg.norm(rel_pos)

        if dist < 1e-4:
            combined_correction += np.random.randn(2) * 0.3
            continue

        rel_pos_norm = rel_pos / dist

        # 极近距离：强力推开
        if dist < ORCA_AGENT_RADIUS:
            combined_correction += rel_pos_norm * 0.8  # 增强推开力度
            continue

        # ORCA：仅在组合半径 + 缓冲内触发
        extended_radius = combined_radius + 0.5  # 增大缓冲区域
        if dist < extended_radius:
            # 速度障碍修正量 - 增强修正力度
            correction = rel_vel + rel_pos_norm * (combined_radius - dist) / time_horizon
            # 增强修正强度：近距离时更强
            strength = max(0.5, 1.2 - dist / extended_radius)
            combined_correction += correction * strength

    if len(ped_positions) > 0:
        adjusted_vel = adjusted_vel + combined_correction / len(ped_positions)

    # 限制最大速度
    vel_norm = np.linalg.norm(adjusted_vel)
    if vel_norm > max_speed:
        adjusted_vel = adjusted_vel / vel_norm * max_speed

    return adjusted_vel


# ═══════════════════════════════════════════════════════════════════
#  专家动作规划器核心：SFM + ORCA + 6 动作映射
# ═══════════════════════════════════════════════════════════════════

class SFMExpertPlanner:
    """
    SFM + ORCA 混合专家动作规划器（支持 6 动作空间）。

    动作映射：
        0 STOP     到达目标
        1 FORWARD  正常前进
        2 TURN_LEFT  需要左转
        3 TURN_RIGHT 需要右转
        4 PAUSE    礼让行人（行人迎面 + 距离 < YIELD_DIST）
        5 BACKWARD 紧急后退（行人 < EMERGENCY_DIST）

    决策分层（优先级从高到低）：
        L0: 到达目标 → STOP(0)
        L1: 极度危险（< EMERGENCY_DIST）→ BACKWARD(5) 或紧急转向
        L2: 礼让（行人迎面 + dist < YIELD_DIST）→ PAUSE(4)
        L3: SFM+ORCA 避障（行人在 WARNING_DIST 内）→ 修正目标角度
        L4: 正常导航（Oracle 路径角度）→ FORWARD/LEFT/RIGHT
    """

    # ── 动作常量 ──
    STOP      = 0
    FORWARD   = 1
    TURN_LEFT  = 2
    TURN_RIGHT = 3
    PAUSE     = 4
    BACKWARD  = 5

    ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT", 4: "PAUSE", 5: "BACKWARD"}

    def __init__(self, num_envs: int, config=None):
        self.num_envs = num_envs
        self.config = config

        # 每个环境的状态
        self.prev_actions: List[int] = [self.FORWARD] * num_envs
        self.repeated_turn_count: List[int] = [0] * num_envs
        self.pause_countdown: List[int] = [0] * num_envs   # 剩余礼让步数
        self.consecutive_pause_count: List[int] = [0] * num_envs  # 连续暂停次数
        self.backward_countdown: List[int] = [0] * num_envs  # 剩余后退步数
        self._recent_actions: List[List[int]] = [[] for _ in range(num_envs)]  # 振荡检测
        self._post_collision_flag: List[bool] = [False] * num_envs  # 上步碰撞标记（跨帧触发BACKWARD）

        # 防卡死：距离历史记录
        self._dist_history: List[List[float]] = [[] for _ in range(num_envs)]

        # 速度缓存（用于 SFM 驱动力计算）
        self._robot_vel_xz: List[np.ndarray] = [
            np.zeros(2, dtype=np.float64) for _ in range(num_envs)
        ]

        # 物理碰撞检测：跟踪上一步位置，检测 FORWARD/BACKWARD 时是否实际位移
        self._prev_positions: List[Optional[np.ndarray]] = [None] * num_envs
        self._stationary_frames: List[int] = [0] * num_envs        # 连续静止帧数
        self._phys_collision_flag: List[bool] = [False] * num_envs  # 物理碰撞标记（触发BACKWARD）

        # 物理碰撞检测参数
        self.PHYS_COLLISION_THRESH = 0.005  # 移动阈值（米），低于此值视为"物理静止"
        self.PHYS_STATIONARY_LIMIT = 3       # 连续静止帧数上限，超过则判定为物理碰撞

        # ── Episode 级震荡逃逸追踪 ──
        # 跨 FORWARD 持续追踪 L↔R 震荡，防止 FORWARD 插入后震荡计数器重置
        self._ep_osc_count: List[int] = [0] * num_envs     # 累积震荡次数
        self._ep_last_action: List[int] = [self.FORWARD] * num_envs  # 上一步动作（跨 FORWARD 保留震荡上下文）

        # ── Episode 级进度监控 ──
        # 检测机器人持续远离目标时强制 BACKWARD（针对撞墙后持续后退的情况）
        self._ep_init_dist: List[float] = [float("inf")] * num_envs   # episode 初始距离
        self._ep_max_dist: List[float] = [0.0] * num_envs            # episode 期间最大距离
        self._ep_steps_without_progress: List[int] = [0] * num_envs   # 连续无进展步数

    def reset_env(self, env_idx: int):
        """重置单个环境的规划器状态。"""
        self.prev_actions[env_idx] = self.FORWARD
        self.repeated_turn_count[env_idx] = 0
        self.pause_countdown[env_idx] = 0
        self.consecutive_pause_count[env_idx] = 0
        self.backward_countdown[env_idx] = 0
        self._recent_actions[env_idx] = []
        self._dist_history[env_idx] = []
        self._robot_vel_xz[env_idx] = np.zeros(2, dtype=np.float64)
        self._post_collision_flag[env_idx] = False
        self._prev_positions[env_idx] = None
        self._stationary_frames[env_idx] = 0
        self._phys_collision_flag[env_idx] = False
        self._ep_osc_count[env_idx] = 0
        self._ep_last_action[env_idx] = self.FORWARD
        self._ep_init_dist[env_idx] = float("inf")
        self._ep_max_dist[env_idx] = 0.0
        self._ep_steps_without_progress[env_idx] = 0

    def reset_all(self):
        """重置所有环境的规划器状态。"""
        for i in range(self.num_envs):
            self.reset_env(i)

    # ──────────────────────────────────────────────────────────────
    #  内部工具：从 observations 提取行人信息
    # ──────────────────────────────────────────────────────────────

    def _get_pedestrian_info(
        self, obs: dict, env_idx: int
    ) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """
        从 human_velocity_sensor 提取行人位置、朝向角、速度。

        数据格式：agent_0_human_velocity_sensor[env_idx] 形状 (6, 6)
            每行：[x, y, z, rotation_rad, vel_x, vel_z]
            x < -90 表示该槽位无行人。

        Returns:
            positions  : List of np.ndarray 形状 (3,)
            rotations  : List of float（弧度）
            velocities : List of np.ndarray 形状 (2,) → (vx, vz)
        """
        key = "agent_0_human_velocity_sensor"
        positions, rotations, velocities = [], [], []

        if key not in obs:
            return positions, rotations, velocities

        raw = _to_numpy(obs[key])
        # 形状可能为 (6,6) 单环境 或 (N_env, 6, 6) 批量
        if raw.ndim == 3:
            env_data = raw[env_idx]   # (6, 6)
        else:
            env_data = raw            # (6, 6)

        for j in range(env_data.shape[0]):
            row = env_data[j]
            if row[0] < -90:
                break
            positions.append(row[:3].copy())
            rotations.append(float(row[3]))
            velocities.append(row[4:6].copy())

        return positions, rotations, velocities

    # ──────────────────────────────────────────────────────────────
    #  L2 辅助：判断行人是否迎面而来
    # ──────────────────────────────────────────────────────────────

    def _is_pedestrian_head_on(
        self,
        robot_pos: np.ndarray,
        robot_angle: float,
        ped_pos: np.ndarray,
        ped_vel_xz: np.ndarray,
    ) -> bool:
        """
        判断行人是否正面迎向机器人。

        条件：
          1. 行人速度方向与机器人前进方向相反（dot < 0）
          2. 行人运动趋向机器人（行人速度方向朝向机器人）
          3. 行人速度大于阈值（排除静止行人）

        Args:
            robot_pos:    机器人 3D 位置
            robot_angle:  机器人朝向角（弧度）
            ped_pos:      行人 3D 位置
            ped_vel_xz:   行人速度向量 (vx, vz)

        Returns:
            True 如果行人迎面靠近，否则 False
        """
        ped_speed = np.linalg.norm(ped_vel_xz)
        if ped_speed < 0.02:
            return False  # 行人静止，不触发礼让

        # 机器人前进方向（单位向量）
        robot_fwd = np.array([math.sin(robot_angle), math.cos(robot_angle)])

        # 行人速度方向（单位向量）
        ped_dir = ped_vel_xz / ped_speed

        # 条件1: 行人运动方向与机器人前进方向相反
        if np.dot(robot_fwd, ped_dir) > -HEAD_ON_DOT_THRESH:
            return False

        # 条件2: 行人运动朝向机器人（行人到机器人方向与行人速度方向点积 > 0）
        robot_xz = _xz(robot_pos)
        ped_xz = _xz(ped_pos)
        to_robot = robot_xz - ped_xz
        to_robot_dist = np.linalg.norm(to_robot)
        if to_robot_dist < 1e-4:
            return False
        to_robot_norm = to_robot / to_robot_dist

        if np.dot(ped_dir, to_robot_norm) < HEAD_ON_DOT_THRESH:
            return False

        return True

    # ──────────────────────────────────────────────────────────────
    #  L3 核心：SFM + ORCA 合力计算 → 目标角度修正
    # ──────────────────────────────────────────────────────────────

    def _compute_sfm_orca_target_angle(
        self,
        env_idx: int,
        robot_pos: np.ndarray,
        robot_angle: float,
        oracle_target_angle: float,
        ped_positions: List[np.ndarray],
        ped_rotations: List[float],
        ped_velocities: List[np.ndarray],
        min_ped_dist: float,
    ) -> float:
        """
        使用 SFM + ORCA 混合策略计算修正后的目标角度。

        步骤：
          1. 以 Oracle 角度为驱动方向，计算 SFM 合力
          2. 用 ORCA 对合力速度进行可行性投影
          3. 返回合力方向对应的目标角度

        Args:
            env_idx:           环境索引
            robot_pos:         机器人 3D 位置
            robot_angle:       机器人当前朝向角
            oracle_target_angle: Oracle 路径角度（无障碍时的目标方向）
            ped_positions:     行人 3D 位置列表
            ped_rotations:     行人朝向角列表
            ped_velocities:    行人速度列表 (vx, vz)
            min_ped_dist:      最近行人距离

        Returns:
            修正后的目标角度（弧度），在 (-π, π]
        """
        robot_xz = _xz(robot_pos)

        # Oracle 驱动方向（单位向量）
        goal_dir = np.array([
            math.sin(oracle_target_angle),
            math.cos(oracle_target_angle),
        ])

        # 当前机器人速度（基于上一步动作估算）
        prev = self.prev_actions[env_idx]
        if prev == self.FORWARD:
            robot_vel = goal_dir * ROBOT_MAX_SPEED * 0.8
        elif prev == self.BACKWARD:
            robot_vel = -goal_dir * ROBOT_MAX_SPEED * 0.5
        elif prev == self.PAUSE:
            robot_vel = np.zeros(2)
        elif prev == self.TURN_LEFT:
            robot_vel = np.array([
                math.sin(robot_angle + math.pi / 2),
                math.cos(robot_angle + math.pi / 2),
            ]) * 0.05
        elif prev == self.TURN_RIGHT:
            robot_vel = np.array([
                math.sin(robot_angle - math.pi / 2),
                math.cos(robot_angle - math.pi / 2),
            ]) * 0.05
        else:
            robot_vel = np.zeros(2)

        # 行人 xz 位置与速度列表
        ped_xz_list = [_xz(p) for p in ped_positions]
        ped_vel_xz_list = []
        for rot, vel in zip(ped_rotations, ped_velocities):
            v = np.asarray(vel, dtype=np.float64)
            if v.ndim == 0 or len(v) == 1:
                spd = float(v.flat[0])
                ped_vel_xz_list.append(np.array([spd * math.sin(rot), spd * math.cos(rot)]))
            elif len(v) >= 2:
                ped_vel_xz_list.append(v[:2].copy())
            else:
                ped_vel_xz_list.append(np.zeros(2))

        # ── Step 1: SFM 合力 ──
        sfm_force = compute_sfm_force(
            robot_xz, robot_vel, goal_dir,
            ped_xz_list, ped_vel_xz_list,
        )

        # 将合力归一化为速度方向
        sfm_speed = np.linalg.norm(sfm_force)
        if sfm_speed < 1e-4:
            sfm_vel = goal_dir * ROBOT_MAX_SPEED
        else:
            sfm_vel = sfm_force / sfm_speed * ROBOT_MAX_SPEED

        # ── Step 2: ORCA 可行性投影 ──
        orca_vel = compute_orca_avoidance_velocity(
            robot_pos, sfm_vel,
            ped_positions, ped_rotations, ped_velocities,
        )

        # ── Step 3: 融合 ORCA 修正速度与 Oracle 目标角度 ──
        # 距离越近，ORCA/SFM 权重越高（增强避障优先级）
        # [修复v3: 根据新阈值调整权重，增强避障]
        if min_ped_dist < EMERGENCY_DIST:
            sfm_weight = 0.95  # 极度危险：95% 避障
        elif min_ped_dist < CRITICAL_DIST:
            sfm_weight = 0.85  # 临界危险：85% 避障
        elif min_ped_dist < YIELD_DIST:
            sfm_weight = 0.85  # 礼让区：85% 避障（增大，更主动）
        elif min_ped_dist < WARNING_DIST:
            sfm_weight = 0.60  # 警告区：60% 避障
        else:
            sfm_weight = 0.05  # 安全区：仅 5% 避障，优先最短路径

        orca_norm = np.linalg.norm(orca_vel)
        if orca_norm < 1e-4:
            orca_dir = goal_dir
        else:
            orca_dir = orca_vel / orca_norm

        fused_dir = sfm_weight * orca_dir + (1.0 - sfm_weight) * goal_dir
        fused_norm = np.linalg.norm(fused_dir)
        if fused_norm < 1e-4:
            fused_dir = goal_dir
        else:
            fused_dir = fused_dir / fused_norm

        # 计算融合方向对应的角度 atan2(x, z) → 与 Oracle 角度同坐标系
        fused_angle = math.atan2(fused_dir[0], fused_dir[1])
        return _normalize_angle(fused_angle)

    # ──────────────────────────────────────────────────────────────
    #  对外接口：compute_action
    # ──────────────────────────────────────────────────────────────

    def compute_action(
        self,
        env_idx: int,
        oracle_path: np.ndarray,
        obs: dict,
        robot_pos: np.ndarray,
        robot_angle: float,
        dist_to_goal: float,
        goal_radius: float = GOAL_RADIUS_DEFAULT,
        prev_robot_pos: Optional[np.ndarray] = None,
    ) -> int:
        """
        为单个环境计算专家动作（6 动作空间）。

        Args:
            env_idx:      环境索引
            oracle_path:  Oracle 路径，已提取为当前环境的 (K, 3)
            obs:          当前观察字典（批量，含所有环境）
            robot_pos:    机器人 3D 位置 np.ndarray (3,)
            robot_angle:  机器人当前朝向角（弧度）
            dist_to_goal: 到目标的当前测地距离（米）
            goal_radius:  到达目标的距离阈值
            prev_robot_pos: 上一步机器人位置，用于物理碰撞检测

        Returns:
            整数动作 0-5
        """
        # ── Episode 级初始化（首次调用时记录初始距离） ──
        if not math.isfinite(self._ep_init_dist[env_idx]):
            self._ep_init_dist[env_idx] = dist_to_goal
            self._ep_max_dist[env_idx] = dist_to_goal
            self._ep_steps_without_progress[env_idx] = 0
            self._ep_osc_count[env_idx] = 0
            self._ep_last_action[env_idx] = self.prev_actions[env_idx]

        # 追踪最大距离
        if dist_to_goal > self._ep_max_dist[env_idx]:
            self._ep_max_dist[env_idx] = dist_to_goal
            self._ep_steps_without_progress[env_idx] = 0
        else:
            self._ep_steps_without_progress[env_idx] += 1

        # ── P-1: 严重偏离检测 —— 累积远离目标超过阈值 → BACKWARD ──
        # 当机器人持续远离目标（可能是撞墙后漂移），强制后退
        if self._ep_max_dist[env_idx] - dist_to_goal > 2.0:
            # 机器人曾经比现在近了 2m 以上，说明走错方向了
            self._ep_osc_count[env_idx] = 0
            self.backward_countdown[env_idx] = 4
            self.consecutive_pause_count[env_idx] = 0
            self.prev_actions[env_idx] = self.BACKWARD
            self._ep_last_action[env_idx] = self.BACKWARD
            return self.BACKWARD

        # ── P0: 物理碰撞检测 —— FORWARD/BACKWARD 时没有实际移动 → 撞墙了 ──
        if prev_robot_pos is not None:
            prev_action = self.prev_actions[env_idx]
            if prev_action in (self.FORWARD, self.BACKWARD):
                pos_diff = np.linalg.norm(_xz(robot_pos) - _xz(prev_robot_pos))
                if pos_diff < self.PHYS_COLLISION_THRESH:
                    self._stationary_frames[env_idx] += 1
                    if self._stationary_frames[env_idx] >= self.PHYS_STATIONARY_LIMIT:
                        # 连续 N 帧物理静止 → 撞墙，强制后退脱离
                        self._stationary_frames[env_idx] = 0
                        self._phys_collision_flag[env_idx] = False
                        self.backward_countdown[env_idx] = 4
                        self.consecutive_pause_count[env_idx] = 0
                        self.prev_actions[env_idx] = self.BACKWARD
                        self._ep_last_action[env_idx] = self.BACKWARD
                        return self.BACKWARD
                else:
                    self._stationary_frames[env_idx] = 0
            else:
                # 非移动动作时也重置（转向/暂停不计入）
                self._stationary_frames[env_idx] = 0

        # ── P1: 上步发生行人碰撞 → 立即强制 BACKWARD（绕过距离判断） ──
        if self._post_collision_flag[env_idx]:
            self._post_collision_flag[env_idx] = False
            self.backward_countdown[env_idx] = 4
            self.consecutive_pause_count[env_idx] = 0
            self.prev_actions[env_idx] = self.BACKWARD
            self._ep_last_action[env_idx] = self.BACKWARD
            return self.BACKWARD

        # ── L0: 到达目标 → STOP ──
        if dist_to_goal < goal_radius:
            self.prev_actions[env_idx] = self.STOP
            self._ep_last_action[env_idx] = self.STOP
            return self.STOP

        # ── 提取行人信息 ──
        ped_pos, ped_rot, ped_vel = self._get_pedestrian_info(obs, env_idx)
        robot_xz = _xz(robot_pos)

        # 计算各行人距离（xz 平面）
        ped_dists = []
        for pp in ped_pos:
            ped_dists.append(np.linalg.norm(_xz(pp) - robot_xz))
        min_ped_dist = min(ped_dists) if ped_dists else float("inf")

        # ── 处理 PAUSE/BACKWARD 倒计时（防止频繁切换） ──
        if self.backward_countdown[env_idx] > 0:
            self.backward_countdown[env_idx] -= 1
            self.prev_actions[env_idx] = self.BACKWARD
            self._ep_last_action[env_idx] = self.BACKWARD
            return self.BACKWARD

        if self.pause_countdown[env_idx] > 0:
            self.pause_countdown[env_idx] -= 1
            self.prev_actions[env_idx] = self.PAUSE
            self._ep_last_action[env_idx] = self.PAUSE
            return self.PAUSE

        # ── P-E1: Episode 级震荡逃逸 —— 跨 FORWARD 持续追踪 L↔R 震荡 ──
        # 如果累积震荡超过阈值，强制执行多步 BACKWARD 脱离
        if self._ep_osc_count[env_idx] >= 3:
            self._ep_osc_count[env_idx] = 0
            self.backward_countdown[env_idx] = 3
            self.consecutive_pause_count[env_idx] = 0
            self.prev_actions[env_idx] = self.BACKWARD
            self._ep_last_action[env_idx] = self.BACKWARD
            return self.BACKWARD

        # ── 诊断日志：记录 L/R 振荡阶段的决策关键变量 ──
        # 仅在连续转向超过阈值时打印
        if self.repeated_turn_count[env_idx] >= 2:
            logger.info(
                f"[DIAG-LR] env={env_idx} step_ct={self.repeated_turn_count[env_idx]} "
                f"min_ped={min_ped_dist:.3f} < EM={min_ped_dist < EMERGENCY_DIST} "
                f"< CR={min_ped_dist < CRITICAL_DIST} < YD={min_ped_dist < YIELD_DIST} "
                f"< WD={min_ped_dist < WARNING_DIST}"
            )

        # ── 预测性避障检查：TTC 和碰撞概率 ──
        robot_fwd = np.array([math.sin(robot_angle), math.cos(robot_angle)])

        # 估算机器人当前速度
        prev = self.prev_actions[env_idx]
        if prev == self.FORWARD:
            robot_vel = robot_fwd * ROBOT_MAX_SPEED * 0.8
        elif prev == self.BACKWARD:
            robot_vel = -robot_fwd * ROBOT_MAX_SPEED * 0.5
        elif prev in (self.TURN_LEFT, self.TURN_RIGHT):
            robot_vel = np.zeros(2)
        else:
            robot_vel = np.zeros(2)

        # 检查每个行人的 TTC 和碰撞概率
        dangerous_ped_indices = []
        for i, (pp, pv) in enumerate(zip(ped_pos, ped_vel)):
            pv_arr = np.asarray(pv, dtype=np.float64)
            if pv_arr.ndim == 1 and len(pv_arr) >= 2:
                pv_xz = pv_arr[:2]
            elif pv_arr.ndim == 0 or len(pv_arr) == 1:
                spd = float(pv_arr.flat[0]) if hasattr(pv_arr, "flat") else float(pv_arr)
                ped_rot_i = ped_rot[i] if i < len(ped_rot) else 0.0
                pv_xz = np.array([spd * math.sin(ped_rot_i), spd * math.cos(ped_rot_i)])
            else:
                pv_xz = np.zeros(2)

            # TTC 计算
            ttc, min_dist = compute_time_to_collision(
                robot_xz, robot_vel,
                SFM_COMBINED_R * 0.5,  # 机器人半径估计
                _xz(pp), pv_xz
            )

            # 碰撞概率计算
            collision_prob = compute_collision_probability(
                robot_xz, robot_fwd, _xz(pp), pv_xz
            )

            # 如果 TTC 过短或碰撞概率过高，标记为危险行人
            if ttc < TTC_THRESHOLD or collision_prob > COLLISION_PROB_THRESHOLD:
                dangerous_ped_indices.append(i)

        # ── L0.5: 预测性危险 → 强制避障（新增） ──
        if dangerous_ped_indices:
            # 找到最危险的行人
            most_dangerous_idx = dangerous_ped_indices[0]
            most_dangerous_dist = ped_dists[most_dangerous_idx]

            # 基于最危险行人的距离决定动作
            if most_dangerous_dist < EMERGENCY_DIST:
                # 极度危险：立即后退
                self.backward_countdown[env_idx] = 4
                self.consecutive_pause_count[env_idx] = 0
                self.prev_actions[env_idx] = self.BACKWARD
                self._ep_last_action[env_idx] = self.BACKWARD
                return self.BACKWARD
            elif most_dangerous_dist < CRITICAL_DIST:
                # 临界危险：后退或转向
                pp = ped_pos[most_dangerous_idx]
                to_ped = _xz(pp) - robot_xz
                to_ped_norm = np.linalg.norm(to_ped)
                if to_ped_norm > 1e-4:
                    to_ped_unit = to_ped / to_ped_norm
                    dot = float(np.dot(robot_fwd, to_ped_unit))
                    if dot > 0.0:
                        self.backward_countdown[env_idx] = 3
                        self.prev_actions[env_idx] = self.BACKWARD
                        self._ep_last_action[env_idx] = self.BACKWARD
                        return self.BACKWARD
                    else:
                        perp_angle = math.atan2(-to_ped_unit[1], to_ped_unit[0]) + math.pi / 2
                        diff = _normalize_angle(perp_angle - robot_angle)
                        act = self.TURN_LEFT if diff > 0 else self.TURN_RIGHT
                        self.prev_actions[env_idx] = act
                        self._ep_last_action[env_idx] = act
                        return act
            else:
                # 较远距离：PAUSE 等待行人通过
                self.pause_countdown[env_idx] = 1
                self.consecutive_pause_count[env_idx] += 1
                self.prev_actions[env_idx] = self.PAUSE
                self._ep_last_action[env_idx] = self.PAUSE
                return self.PAUSE

        # ── L1: 极度危险 → BACKWARD ──
        if min_ped_dist < EMERGENCY_DIST:
            # 极度危险：立即后退5步（增加安全缓冲）
            self.backward_countdown[env_idx] = 5
            self.consecutive_pause_count[env_idx] = 0  # 重置PAUSE计数
            self.prev_actions[env_idx] = self.BACKWARD
            self._ep_last_action[env_idx] = self.BACKWARD
            return self.BACKWARD

        # ── L1.5: 中度危险（EMERGENCY_DIST ~ CRITICAL_DIST）→ 后退或绕行 ──
        if min_ped_dist < CRITICAL_DIST:
            closest_idx = int(np.argmin(ped_dists))
            closest_ped_xz = _xz(ped_pos[closest_idx])
            to_ped = closest_ped_xz - robot_xz
            to_ped_norm = np.linalg.norm(to_ped)
            if to_ped_norm > 1e-4:
                to_ped_unit = to_ped / to_ped_norm
                robot_fwd = np.array([math.sin(robot_angle), math.cos(robot_angle)])
                dot = float(np.dot(robot_fwd, to_ped_unit))
                if dot > 0.0:
                    # 行人在前方任何角度 → 后退3步开辟安全距离（从2增加到3）
                    self.backward_countdown[env_idx] = 3
                    self.prev_actions[env_idx] = self.BACKWARD
                    self._ep_last_action[env_idx] = self.BACKWARD
                    return self.BACKWARD
                else:
                    # 行人在侧方或后方 → 转向远离
                    perp_angle = math.atan2(-to_ped_unit[1], to_ped_unit[0]) + math.pi / 2
                    diff = _normalize_angle(perp_angle - robot_angle)
                    act = self.TURN_LEFT if diff > 0 else self.TURN_RIGHT
                    self.prev_actions[env_idx] = act
                    self._ep_last_action[env_idx] = act
                    return act

        # ── 计算 Oracle 目标角度 ──
        oracle_target_angle = robot_angle  # 默认不变
        if oracle_path is not None and len(oracle_path) >= 2:
            # [修复] 对齐 v3：使用下一路径点（look_ahead=1），
            # 避免远处路径点导致的角度振荡
            target_idx = 1  # 下一路径点（对齐 v3）
            next_pt = oracle_path[target_idx]
            dx = float(next_pt[0]) - float(robot_pos[0])
            dz = float(next_pt[2]) - float(robot_pos[2])
            oracle_target_angle = math.atan2(dx, dz)

        # ── L2: 礼让等待 → PAUSE（仅在真正需要时设置倒计时） ──
        if min_ped_dist < YIELD_DIST and ped_pos:
            closest_idx = int(np.argmin(ped_dists))
            closest_ped_dist = ped_dists[closest_idx]

            # 超过限制，强制绕行：计算所有行人的 SFM 合力方向，选择最优绕行侧
            if self.consecutive_pause_count[env_idx] >= MAX_CONSECUTIVE_PAUSE:
                self.consecutive_pause_count[env_idx] = 0
                robot_xz = _xz(robot_pos)
                robot_fwd = np.array([math.sin(robot_angle), math.cos(robot_angle)])

                ped_xz_list = [_xz(p) for p in ped_pos]
                ped_vel_xz_list = []
                for rot_i, vel_i in zip(ped_rot, ped_vel):
                    v_arr = np.asarray(vel_i, dtype=np.float64)
                    if v_arr.ndim == 1 and len(v_arr) >= 2:
                        ped_vel_xz_list.append(v_arr[:2].copy())
                    elif v_arr.ndim == 0 or len(v_arr) == 1:
                        spd = float(v_arr.flat[0]) if hasattr(v_arr, "flat") else float(v_arr)
                        ped_vel_xz_list.append(np.array([spd * math.sin(rot_i), spd * math.cos(rot_i)]))
                    else:
                        ped_vel_xz_list.append(np.zeros(2))

                sfm_force = compute_sfm_force(
                    robot_xz, np.zeros(2),
                    np.array([math.sin(oracle_target_angle), math.cos(oracle_target_angle)]),
                    ped_xz_list, ped_vel_xz_list,
                )

                left_perp = np.array([-robot_fwd[1], robot_fwd[0]])
                right_perp = -left_perp
                dot_left = float(np.dot(sfm_force, left_perp))
                dot_right = float(np.dot(sfm_force, right_perp))

                act = self.TURN_LEFT if dot_left >= dot_right else self.TURN_RIGHT
                self.prev_actions[env_idx] = act
                self._ep_last_action[env_idx] = act
                return act

            # 只在倒计时为 0 时设置新倒计时，防止无限重置
            if self.pause_countdown[env_idx] == 0:
                self.pause_countdown[env_idx] = 2
            self.consecutive_pause_count[env_idx] += 1
            self.prev_actions[env_idx] = self.PAUSE
            self._ep_last_action[env_idx] = self.PAUSE
            return self.PAUSE

        # ── L3: SFM + ORCA 避障修正目标角度 ──
        if min_ped_dist < WARNING_DIST and ped_pos:
            final_target_angle = self._compute_sfm_orca_target_angle(
                env_idx,
                robot_pos, robot_angle,
                oracle_target_angle,
                ped_pos, ped_rot, ped_vel,
                min_ped_dist,
            )
        else:
            # ── L4: 无障碍，直接使用 Oracle 角度 ──
            final_target_angle = oracle_target_angle

        # ── 角度差 → 动作 ──
        angle_diff = _normalize_angle(final_target_angle - robot_angle)

        # ── 诊断日志：记录 L3/L4 决策和角度差 ──
        if self.repeated_turn_count[env_idx] >= 2:
            in_sfm = min_ped_dist < WARNING_DIST and ped_pos
            logger.info(
                f"[DIAG-ANGLE] env={env_idx} mode={'SFM(L3)' if in_sfm else 'Oracle(L4)'} "
                f"oracle_angle={math.degrees(oracle_target_angle):.1f}° "
                f"final_angle={math.degrees(final_target_angle):.1f}° "
                f"angle_diff={math.degrees(angle_diff):.1f}° → action={'L' if angle_diff > 0 else 'R' if angle_diff < 0 else 'FWD'}"
            )

        # ── 防卡死检测：如果连续前进但距离无进展，强制转向 ──
        # 注意：必须传入 _ep_last_action 以便在函数调用前检查 episode 级震荡
        action = self._angle_to_action_with_stuck_check(
            env_idx, angle_diff, min_ped_dist, dist_to_goal,
            ep_last=self._ep_last_action[env_idx],
        )

        self.prev_actions[env_idx] = action
        # 更新 episode 级动作追踪（用于跨 FORWARD 震荡检测）
        self._ep_last_action[env_idx] = action
        return action

    def _angle_to_action_with_stuck_check(
        self, env_idx: int, angle_diff: float, min_ped_dist: float,
        dist_to_goal: float, ep_last: int,
    ) -> int:
        """
        根据角度差和周边情况映射为最终动作（不含 PAUSE/BACKWARD/STOP）。

        增强的防卡机制：
        1. 超过 MAX_REPEATED_TURNS 次同向转向后强制前进
        2. 连续前进但距离无进展时强制 BACKWARD
        3. Episode 级 L↔R 震荡追踪（跨 FORWARD 持续计数）
           — FORWARD 后跳过振荡检测一个步，防止误判
           — 只有当 episode 震荡计数 >= 2 才触发 FORWARD
        """
        # 更新距离历史
        self._dist_history[env_idx].append(dist_to_goal)
        if len(self._dist_history[env_idx]) > STUCK_CHECK_WINDOW:
            self._dist_history[env_idx].pop(0)

        # 检查是否卡住（连续前进但距离无进展）
        stuck_detected = False
        if len(self._dist_history[env_idx]) >= STUCK_CHECK_WINDOW:
            dists = self._dist_history[env_idx]
            min_dist_in_window = min(dists)
            max_dist_in_window = max(dists)
            if (max_dist_in_window - min_dist_in_window) < STUCK_THRESHOLD:
                recent = self._recent_actions[env_idx][-STUCK_CHECK_WINDOW:]
                forward_ratio = recent.count(self.FORWARD) / max(1, len(recent))
                if forward_ratio > 0.7:
                    stuck_detected = True

        if abs(angle_diff) < TURN_THRESHOLD or abs(angle_diff) > 2 * math.pi - TURN_THRESHOLD:
            self.repeated_turn_count[env_idx] = 0
            self._recent_actions[env_idx] = []
            self._recent_actions[env_idx].append(self.FORWARD)
            return self.FORWARD

        # ── 振荡检测：基于 ep_last 检测跨步 L↔R 交替 ──
        osc_detected = (ep_last in (self.TURN_LEFT, self.TURN_RIGHT))

        # ── Episode 级震荡追踪（FORWARD 后历史清空，用 ep_last 判断） ──
        if ep_last == self.FORWARD:
            if self.repeated_turn_count[env_idx] > 0:
                self._ep_osc_count[env_idx] += self.repeated_turn_count[env_idx]
            self.repeated_turn_count[env_idx] = 0
            self._recent_actions[env_idx] = []

        # 根据角度差决定目标动作
        if angle_diff > 0:
            action = self.TURN_LEFT
        else:
            action = self.TURN_RIGHT

        # 振荡时计数器增加；非振荡时重置
        if osc_detected:
            self.repeated_turn_count[env_idx] += 1
        else:
            self.repeated_turn_count[env_idx] = 0

        # 超过阈值：强制 FORWARD 打破振荡
        if self.repeated_turn_count[env_idx] >= MAX_REPEATED_TURNS:
            self.repeated_turn_count[env_idx] = 0
            self._recent_actions[env_idx] = [self.FORWARD]
            return self.FORWARD

        # 物理卡住：强制 BACKWARD 脱离障碍
        if stuck_detected and STUCK_FORCE_TURN:
            self.repeated_turn_count[env_idx] = 0
            self._recent_actions[env_idx] = [self.BACKWARD]
            return self.BACKWARD

        self._recent_actions[env_idx].append(action)
        return action


# ═══════════════════════════════════════════════════════════════════
#  ExpertDataCollector6Action —— 训练器主类
# ═══════════════════════════════════════════════════════════════════

@baseline_registry.register_trainer(name="expert_data_collector_6action")
class ExpertDataCollector6Action(BaseRLTrainer):
    """
    6 动作空间专家数据采集器。

    在 expert_data_collector_v3.py（4 动作）的基础上，
    集成 SFMExpertPlanner（社会力 + ORCA 混合策略），
    充分利用新增的 PAUSE(4) 和 BACKWARD(5) 动作。

    注册名称：expert_data_collector_6action

    使用方式（config yaml）：
        habitat_baselines:
            trainer_name: expert_data_collector_6action
    """

    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.envs = None
        self.obs_transforms = []
        self._env_spec = None
        self._is_distributed = get_distrib_size()[2] > 1
        self._planner: Optional[SFMExpertPlanner] = None

    # ──────────────────────────────────────────────────────────────
    #  环境初始化
    # ──────────────────────────────────────────────────────────────

    def _init_envs(self, config=None, is_eval: bool = False):
        if config is None:
            config = self.config

        env_factory: VectorEnvFactory = hydra.utils.instantiate(
            config.habitat_baselines.vector_env_factory
        )
        self.envs = env_factory.construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
            enforce_scenes_greater_eq_environments=is_eval,
            is_first_rank=(
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ),
        )
        self._env_spec = EnvironmentSpec(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            orig_action_space=self.envs.orig_action_spaces[0],
        )

        # 初始化 SFM+ORCA 规划器
        self._planner = SFMExpertPlanner(self.envs.num_envs, config)

    def _create_obs_transforms(self):
        self.obs_transforms = get_active_obs_transforms(self.config)
        self._env_spec.observation_space = apply_obs_transforms_obs_space(
            self._env_spec.observation_space, self.obs_transforms
        )

    def _recreate_envs(self) -> bool:
        try:
            with contextlib.suppress(Exception):
                if self.envs is not None:
                    self.envs.close()
            self._init_envs(self.config, is_eval=False)
            self._create_obs_transforms()
            observations = self.envs.reset()
            self.envs.post_step(observations)
            if self._planner:
                self._planner.reset_all()
            return True
        except Exception as e:
            if rank0_only():
                logger.error(f"Failed to recreate envs: {e}")
            return False

    def _try_get_current_episode(self, env_idx: int):
        try:
            return self.envs.call_at(env_idx, "current_episode", {"all_info": True})
        except Exception as e:
            if rank0_only():
                logger.warning(f"current_episode failed on env {env_idx}: {e}")
            return None

    def _init_train(self, resume_state=None):
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    "Previous training state found. Delete checkpoint folder or "
                    "set load_resume_state_config=True."
                )
            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    f"Initialized DD-PPO with {torch.distributed.get_world_size()} workers"
                )
            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        self._init_envs()
        self.device = _get_device(self.config)
        self._create_obs_transforms()

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch_obs(observations, device=self.device)
        self.t_start = time.time()

    # ──────────────────────────────────────────────────────────────
    #  专家动作计算
    # ──────────────────────────────────────────────────────────────

    def _get_expert_action(
        self,
        env_idx: int,
        current_obs: dict,
        global_dist_to_goal: List[float],
        prev_robot_pos: Optional[np.ndarray] = None,
    ) -> int:
        """
        为单个环境计算 6 动作专家动作。

        Args:
            env_idx:            环境索引
            current_obs:        已经过 obs_transforms 的批量观察字典
            global_dist_to_goal: 各环境到目标距离缓存列表
            prev_robot_pos:     上一步机器人位置（用于物理碰撞检测）

        Returns:
            整数动作 0-5
        """
        # 1. 获取 Oracle 路径
        oracle_key = "agent_0_main_oracle_shortest_path_sensor"
        if oracle_key not in current_obs:
            return SFMExpertPlanner.FORWARD

        oracle_path_raw = _to_numpy(current_obs[oracle_key])
        # 形状可能 (N_env, K, 3) 或 (K, 3)
        if oracle_path_raw.ndim == 3:
            env_path = oracle_path_raw[env_idx]   # (K, 3)
        else:
            env_path = oracle_path_raw             # (K, 3)

        # 2. 获取机器人状态
        try:
            agent_state = self.envs.call_at(env_idx, "get_agent_state")
            robot_pos = np.array(agent_state.position, dtype=np.float64)
        except Exception as e:
            logger.warning(f"get_agent_state failed env {env_idx}: {e}")
            return SFMExpertPlanner.FORWARD

        # 3. 获取机器人朝向角
        loc_key = "agent_0_localization_sensor"
        if loc_key in current_obs:
            loc_data = _to_numpy(current_obs[loc_key])
            if loc_data.ndim == 2:
                robot_angle = float(loc_data[env_idx, -1])
            else:
                robot_angle = float(loc_data[-1])
        else:
            # fallback：从四元数提取
            rot = agent_state.rotation
            if hasattr(rot, "w"):
                robot_angle = 2.0 * math.asin(float(rot.y))
            else:
                robot_angle = 0.0

        # 4. 到目标距离（优先从 pointgoal 传感器读取最新值，不再依赖缓存）
        dist = global_dist_to_goal[env_idx]  # fallback 值
        pg_key = "agent_0_pointgoal_with_gps_compass"
        if pg_key in current_obs:
            try:
                pg = _to_numpy(current_obs[pg_key])
                dist = float(pg[env_idx, 0]) if pg.ndim == 2 else float(pg[0])
            except Exception:
                # 读取失败时保持上一次的缓存值，但不更新缓存
                pass
        # 始终更新缓存，保证下一次能获取到最新值
        global_dist_to_goal[env_idx] = dist

        # 5. 目标半径配置
        goal_radius = float(
            self.config.expert_data_collection.get("goal_radius", GOAL_RADIUS_DEFAULT)
        )

        # 6. 调用 SFM+ORCA 规划器（传入前一步位置用于物理碰撞检测）
        action = self._planner.compute_action(
            env_idx=env_idx,
            oracle_path=env_path,
            obs=current_obs,
            robot_pos=robot_pos,
            robot_angle=robot_angle,
            dist_to_goal=dist,
            goal_radius=goal_radius,
            prev_robot_pos=prev_robot_pos,
        )

        if rank0_only() and env_idx == 0 and self.config.habitat_baselines.verbose:
            logger.debug(
                f"[SFM-Expert] env={env_idx} action={SFMExpertPlanner.ACTION_NAMES[action]}({action}) "
                f"dist={dist:.3f}m"
            )

        return action

    # ──────────────────────────────────────────────────────────────
    #  扫描已有 episode（断点续传）
    # ──────────────────────────────────────────────────────────────

    def _scan_existing_episodes(self, data_folder: str, split: str) -> set:
        existing = set()
        split_path = pathlib.Path(data_folder) / split
        if not split_path.exists():
            return existing
        for scene_dir in split_path.iterdir():
            if not scene_dir.is_dir():
                continue
            for ep_dir in scene_dir.iterdir():
                if not ep_dir.is_dir():
                    continue
                if (
                    (ep_dir / "rgb").exists()
                    and (ep_dir / "depth").exists()
                    and (ep_dir / "action" / "0.json").exists()
                    and (ep_dir / "human_num" / "0.json").exists()
                    and (ep_dir / "distance_to_goal" / "0.json").exists()
                ):
                    existing.add((scene_dir.name, ep_dir.name))
        return existing

    # ──────────────────────────────────────────────────────────────
    #  行人视野检测（与 v3 版本一致）
    # ──────────────────────────────────────────────────────────────

    def _check_pedestrians_in_camera_view(
        self,
        agent_position: np.ndarray,
        agent_rotation,
        pedestrian_positions: list,
        max_distance: float = 5.0,
        fov_horizontal: float = 90.0,
    ) -> Tuple[int, float]:
        if not pedestrian_positions:
            return 0, float("inf")

        if hasattr(agent_rotation, "w"):
            w, x, y, z = (agent_rotation.w, agent_rotation.x,
                          agent_rotation.y, agent_rotation.z)
        elif hasattr(agent_rotation, "components"):
            w, x, y, z = agent_rotation.components
        else:
            w, x, y, z = agent_rotation

        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

        half_fov_h = math.radians(fov_horizontal) / 2.0
        count = 0
        closest = float("inf")

        for ped in pedestrian_positions:
            rel = np.asarray(ped) - np.asarray(agent_position)
            cam = R.T @ rel
            if cam[2] <= 0.01:
                continue
            dist = np.linalg.norm(cam)
            if dist > max_distance:
                continue
            if abs(math.atan2(cam[0], cam[2])) <= half_fov_h:
                count += 1
                closest = min(closest, dist)

        return count, closest

    # ──────────────────────────────────────────────────────────────
    #  主数据采集循环
    # ──────────────────────────────────────────────────────────────

    def collect_expert_data(self) -> None:
        """
        6 动作专家数据采集主方法。

        与 v3 版本结构一致，主要区别：
          · 使用 SFMExpertPlanner 计算动作（支持 6 个动作）
          · 日志中记录 PAUSE/BACKWARD 的发生比例
        """
        # 初始化
        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        max_episodes = self.config.expert_data_collection.max_episodes
        max_steps_per_episode = self.config.expert_data_collection.max_steps_per_episode
        data_folder = self.config.expert_data_collection.data_folder
        split_name = self.config.expert_data_collection.split
        goal_radius = float(self.config.expert_data_collection.get("goal_radius", GOAL_RADIUS_DEFAULT))

        if rank0_only():
            os.makedirs(data_folder, exist_ok=True)
            logger.info(
                f"[6ActionExpert] Data collection started. "
                f"max_episodes={max_episodes}, max_steps={max_steps_per_episode}, "
                f"goal_radius={goal_radius}m"
            )
            existing_eps = self._scan_existing_episodes(data_folder, split_name)
            logger.info(f"Found {len(existing_eps)} existing episodes.")
        else:
            existing_eps = set()

        # 统计计数器
        collected_eps = 0
        successful_eps = 0
        failed_eps = 0
        skipped_eps = 0
        total_steps = 0
        action_counter = {i: 0 for i in range(6)}  # 统计各动作使用次数

        saved_episode_ids: set = set(existing_eps)
        global_dist: List[float] = [float("inf")] * self.envs.num_envs
        episodes_data = [[] for _ in range(self.envs.num_envs)]
        # 每个 env 独立步数计数器，用于精确判断 timeout
        env_steps: List[int] = [0] * self.envs.num_envs
        # 每个 env 的终止原因
        env_termination_reason: List[str] = ["unknown"] * self.envs.num_envs
        # 每个 env 的前一步位置（用于物理碰撞检测）
        prev_robot_pos: List[Optional[np.ndarray]] = [None] * self.envs.num_envs

        pbar = tqdm(total=max_episodes, desc="Episodes", dynamic_ncols=True) if rank0_only() else None

        while collected_eps < max_episodes:
            # ── 重置环境 ──
            if rank0_only():
                logger.info(f"[LOOP] collected={collected_eps} failed={failed_eps} skipped={skipped_eps} — calling envs.reset()...")
            observations = None
            for _retry in range(3):
                try:
                    observations = self.envs.reset()
                    if rank0_only():
                        logger.info(f"[LOOP] envs.reset() returned, calling post_step()...")
                    observations = self.envs.post_step(observations)
                    if rank0_only():
                        logger.info(f"[LOOP] post_step() done. Checking valid envs...")
                    break
                except Exception as e:
                    logger.warning(f"Reset failed (attempt {_retry+1}/3): {e}")
                    time.sleep(0.1)
                    if _retry == 2 and self._recreate_envs():
                        observations = None
                        break

            if observations is None:
                continue

            # 重置规划器状态
            if self._planner:
                self._planner.reset_all()

            # ── 筛选有效 episode（测地距离 > 5m） ──
            valid_envs = []
            for env_idx in range(self.envs.num_envs):
                ep = self._try_get_current_episode(env_idx)
                if ep is None:
                    skipped_eps += 1
                    continue
                # 从多个来源尝试获取 geodesic_distance
                geo_dist = 0.0
                if hasattr(ep, "info") and ep.info:
                    geo_dist = float(ep.info.get("geodesic_distance", 0.0))
                # fallback：从 episode 本身的 geodesic_distance 字段读取
                if geo_dist == 0.0 and hasattr(ep, "geodesic_distance") and ep.geodesic_distance is not None:
                    geo_dist = float(ep.geodesic_distance)
                # fallback：从 info_dict（有些版本叫 additional_obj_config_file）
                if geo_dist == 0.0 and hasattr(ep, "info") and ep.info:
                    for key in ("geo_distance", "geo_dist", "start_position_residual"):
                        val = ep.info.get(key, None)
                        if val is not None:
                            try:
                                geo_dist = float(val)
                                break
                            except Exception:
                                pass

                # 打印前几次 reset 的 ep 信息，帮助诊断 geo_dist 读取问题
                if collected_eps == 0 and skipped_eps < 3 and env_idx == 0 and rank0_only():
                    ep_info_keys = list(ep.info.keys()) if (hasattr(ep, "info") and ep.info) else []
                    ep_attrs = [a for a in dir(ep) if not a.startswith("_")]
                    logger.info(
                        f"[DEBUG-EP] env={env_idx} geo_dist={geo_dist:.3f} "
                        f"ep.info keys={ep_info_keys} ep attrs={ep_attrs[:15]}"
                    )

                if 5.0 < geo_dist < 1000.0 and not (
                    math.isnan(geo_dist) or math.isinf(geo_dist)
                ):
                    valid_envs.append(env_idx)
                    global_dist[env_idx] = geo_dist
                else:
                    skipped_eps += 1

            if not valid_envs:
                if rank0_only():
                    logger.info(f"[LOOP] No valid envs (all geo_dist <= 5m or unreadable), skipping. skipped_total={skipped_eps}")
                continue

            if rank0_only():
                logger.info(f"[LOOP] valid_envs={valid_envs}, starting episode step loop...")

            for env_idx in range(self.envs.num_envs):
                episodes_data[env_idx] = []
                env_steps[env_idx] = 0
                env_termination_reason[env_idx] = "unknown"
                prev_robot_pos[env_idx] = None  # 重置位置追踪

            dones = [False] * self.envs.num_envs
            active_envs = set(valid_envs)
            episode_steps = 0
            episode_done = False

            # ── Episode 步骤循环 ──
            while not episode_done and episode_steps < max_steps_per_episode:
                try:
                    cur_obs = batch_obs(observations, device=self.device)
                    cur_obs = apply_obs_transforms_batch(cur_obs, self.obs_transforms)

                    # 预先获取所有活跃环境的当前机器人位置（用于物理碰撞检测）
                    curr_robot_pos_map: Dict[int, np.ndarray] = {}
                    for env_idx in active_envs:
                        try:
                            agent_state = self.envs.call_at(env_idx, "get_agent_state")
                            curr_robot_pos_map[env_idx] = np.array(agent_state.position, dtype=np.float64)
                        except Exception:
                            curr_robot_pos_map[env_idx] = None

                    actions = [0] * self.envs.num_envs
                    for env_idx in active_envs:
                        act = self._get_expert_action(
                            env_idx, cur_obs, global_dist,
                            prev_robot_pos=prev_robot_pos[env_idx],
                        )
                        actions[env_idx] = int(act)
                        action_counter[int(act)] = action_counter.get(int(act), 0) + 1

                        # 保存当前步数据
                        env_obs = observations[env_idx]
                        rgb_d = env_obs.get("agent_0_overhead_front_rgb", None)
                        dep_d = env_obs.get("agent_0_overhead_front_depth", None)
                        third_d = env_obs.get("agent_0_third_rgb", None)
                        hnum_d = env_obs.get("agent_0_human_num_sensor", None)

                        if rgb_d is not None and dep_d is not None and hnum_d is not None:
                            step_data = {
                                "rgb": rgb_d,
                                "depth": dep_d,
                                "third_rgb": third_d,
                                "human_num": hnum_d,
                                "action": int(act),
                                "distance_to_goal": float(global_dist[env_idx]),
                                "step": episode_steps,
                                "pedestrian_in_view": 0,
                                "trajectory": {},
                                "info": {},  # 添加 info 字段
                            }

                            # 记录轨迹和行人视野
                            try:
                                agent_state = self.envs.call_at(env_idx, "get_agent_state")
                                cur_pos = np.array(agent_state.position)
                                cur_rot = agent_state.rotation
                                step_data["trajectory"]["robot"] = {
                                    "position": cur_pos.tolist(),
                                    "rotation": (
                                        [cur_rot.w, cur_rot.x, cur_rot.y, cur_rot.z]
                                        if hasattr(cur_rot, "w")
                                        else list(cur_rot)
                                    ),
                                }
                                ped_pos_list, ped_rot_list, ped_vel_list = (
                                    self._planner._get_pedestrian_info(cur_obs, env_idx)
                                )
                                step_data["trajectory"]["pedestrians"] = [
                                    {
                                        "id": i,
                                        "position": pp.tolist(),
                                        "rotation": float(pr),
                                        "velocity": pv.tolist(),
                                    }
                                    for i, (pp, pr, pv) in enumerate(
                                        zip(ped_pos_list, ped_rot_list, ped_vel_list)
                                    )
                                ]
                                if ped_pos_list:
                                    cam_offset = np.array([0.166, 0.83, 0.0])
                                    cam_pos = cur_pos + cam_offset
                                    n_in_view, _ = self._check_pedestrians_in_camera_view(
                                        cam_pos, cur_rot, ped_pos_list
                                    )
                                    step_data["pedestrian_in_view"] = n_in_view
                            except Exception:
                                pass

                            episodes_data[env_idx].append(step_data)

                    # 执行动作
                    for env_idx in active_envs:
                        self.envs.async_step_at(env_idx, np.array([actions[env_idx]]))

                    outputs = {env_idx: self.envs.wait_step_at(env_idx) for env_idx in active_envs}

                    for env_idx, (obs_i, rew_i, done_i, info_i) in outputs.items():
                        observations[env_idx] = obs_i
                        dones[env_idx] = done_i
                        if isinstance(info_i, dict):
                            info_dist = info_i.get("distance_to_goal", None)
                            if info_dist is not None:
                                try:
                                    global_dist[env_idx] = float(info_dist)
                                except Exception:
                                    pass
                            # 更新最后一步的 info
                            if episodes_data[env_idx]:
                                episodes_data[env_idx][-1]["info"] = info_i
                            # ── P2: 检测行人碰撞，设置跨帧标记供下一帧 BACKWARD ──
                            human_col = info_i.get("human_collision", 0)
                            if human_col and float(human_col) > 0:
                                self._planner._post_collision_flag[env_idx] = True
                            # ── 诊断日志：每 50 步打印行人碰撞状态 ──
                            if env_steps[env_idx] % 50 == 0 and env_steps[env_idx] > 0:
                                ped_pos_diag, _, _ = self._planner._get_pedestrian_info(observations, env_idx)
                                ped_dists_diag = [
                                    float(np.linalg.norm(_xz(pp) - _xz(self.envs.call_at(env_idx, "get_agent_state").position)))
                                    for pp in ped_pos_diag
                                ] if ped_pos_diag else []
                                # 打印 info 字典的所有键和值
                                info_keys = list(info_i.keys()) if isinstance(info_i, dict) else []
                                logger.info(
                                    f"[DIAG step={env_steps[env_idx]}] env={env_idx} "
                                    f"human_col={human_col} collision_flag={self._planner._post_collision_flag[env_idx]} "
                                    f"ped_dists={ped_dists_diag} ped_count={len(ped_pos_diag)} "
                                    f"prev_action={self._planner.prev_actions[env_idx]} "
                                    f"backward_cd={self._planner.backward_countdown[env_idx]} "
                                    f"info_keys={info_keys}"
                                )

                        # 每步递增该 env 的步数
                        env_steps[env_idx] += 1

                        # env done：追加一步包含执行后真实距离的记录，并记录终止原因
                        if done_i and episodes_data[env_idx]:
                            last_step = episodes_data[env_idx][-1].copy()
                            last_step["distance_to_goal"] = float(global_dist[env_idx])
                            last_step["step"] = last_step["step"] + 1
                            last_step["info"] = info_i if isinstance(info_i, dict) else {}
                            episodes_data[env_idx].append(last_step)

                            # 判断 env 终止原因
                            i_info = info_i if isinstance(info_i, dict) else {}
                            human_col = i_info.get("human_collision", 0)
                            is_success = bool(i_info.get("success", False))
                            if is_success:
                                env_termination_reason[env_idx] = "env_success"
                            elif human_col and float(human_col) > 0:
                                env_termination_reason[env_idx] = "collision"
                            elif env_steps[env_idx] >= max_steps_per_episode:
                                env_termination_reason[env_idx] = "timeout"
                            else:
                                # 环境自身触发 done（max_episode_steps 或其他）
                                env_termination_reason[env_idx] = f"env_done(steps={env_steps[env_idx]},dist={global_dist[env_idx]:.2f})"

                    observations = self.envs.post_step(observations)

                    # 移除已完成环境
                    for env_idx in list(active_envs):
                        if dones[env_idx]:
                            active_envs.remove(env_idx)

                    # ── 更新前一步位置（用于下一帧的物理碰撞检测）──
                    # curr_robot_pos_map 记录的是执行动作前的位置，
                    # 执行后的位置在 observations 中，下次循环时重新获取
                    for env_idx in list(active_envs):
                        if curr_robot_pos_map.get(env_idx) is not None:
                            prev_robot_pos[env_idx] = curr_robot_pos_map[env_idx]

                    episode_done = len(active_envs) == 0
                    episode_steps += 1
                    total_steps += 1

                except Exception as e:
                    if rank0_only():
                        logger.error(f"Episode step error: {e}")
                    # 异常终止时标记所有还在活跃的 env
                    for env_idx in list(active_envs):
                        env_termination_reason[env_idx] = f"exception({type(e).__name__})"
                    episode_done = True

            # 超时退出循环时，为仍在 active_envs 中的 env 标记 timeout
            for env_idx in list(active_envs):
                if env_termination_reason[env_idx] == "unknown":
                    env_termination_reason[env_idx] = "timeout"

            # ── 保存 episode 数据 ──
            for env_idx in valid_envs:
                if (
                    (dones[env_idx] or episode_done or episode_steps >= max_steps_per_episode)
                    and len(episodes_data[env_idx]) > 0
                ):
                    ep = self._try_get_current_episode(env_idx)
                    if ep is None:
                        continue

                    ep_id = str(ep.episode_id)
                    scene_id = ep.scene_id
                    scene_name = pathlib.Path(scene_id).stem
                    ep_key = (scene_name, ep_id)

                    if ep_key in saved_episode_ids:
                        continue

                    data_root = pathlib.Path(data_folder) / split_name / scene_name / ep_id
                    if data_root.exists():
                        saved_episode_ids.add(ep_key)
                        continue

                    steps_d = episodes_data[env_idx]
                    ep_actions = [s["action"] for s in steps_d]
                    ep_dists = [s["distance_to_goal"] for s in steps_d]
                    final_dist = ep_dists[-1]
                    success_thresh = float(self.config.expert_data_collection.get("goal_radius", GOAL_RADIUS_DEFAULT))

                    # 判断是否成功：必须到达目标附近（距离 < success_thresh）
                    last_action = ep_actions[-1]
                    if final_dist > success_thresh:
                        failed_eps += 1
                        # 使用每个 env 独立追踪的精确终止原因
                        termination_reason = env_termination_reason[env_idx]
                        # 计算距离变化（初始距离 vs 最终距离）
                        init_dist = ep_dists[0] if ep_dists else float("inf")
                        dist_progress = init_dist - final_dist

                        if rank0_only():
                            logger.info(
                                f"[SKIP-NOT-REACHED] ep={ep_id} scene={scene_name} "
                                f"final_dist={final_dist:.3f}m > goal_radius={success_thresh:.3f}m "
                                f"reason={termination_reason} steps={len(steps_d)} "
                                f"init_dist={init_dist:.3f}m progress={dist_progress:+.3f}m "
                                f"actions=STOP:{ep_actions.count(0)},FWD:{ep_actions.count(1)},"
                                f"L:{ep_actions.count(2)},R:{ep_actions.count(3)},"
                                f"PAUSE:{ep_actions.count(4)},BACK:{ep_actions.count(5)}"
                            )

                        # ── 保存失败轨迹到 fail_analysis/ 目录供事后分析 ──
                        try:
                            fail_dir = pathlib.Path(data_folder) / "fail_analysis" / split_name / scene_name / ep_id
                            fail_dir.mkdir(parents=True, exist_ok=True)
                            import json as _json
                            fail_meta = {
                                "ep_id": ep_id,
                                "scene": scene_name,
                                "reason": termination_reason,
                                "steps": len(steps_d),
                                "init_dist": init_dist,
                                "final_dist": final_dist,
                                "progress": dist_progress,
                                "goal_radius": success_thresh,
                                "action_counts": {
                                    "STOP": ep_actions.count(0),
                                    "FWD": ep_actions.count(1),
                                    "LEFT": ep_actions.count(2),
                                    "RIGHT": ep_actions.count(3),
                                    "PAUSE": ep_actions.count(4),
                                    "BACK": ep_actions.count(5),
                                },
                                "actions": ep_actions,
                                "distances": ep_dists,
                            }
                            with open(fail_dir / "fail_meta.json", "w") as _f:
                                _json.dump(fail_meta, _f, indent=2)
                        except Exception as _e:
                            if rank0_only():
                                logger.warning(f"[WARN] Failed to save fail_analysis: {_e}")

                        continue

                    # 追加终止步（若最后动作不是 STOP）
                    if last_action != SFMExpertPlanner.STOP:
                        last_step = steps_d[-1]
                        steps_d.append({
                            **last_step,
                            "action": SFMExpertPlanner.STOP,
                            "step": last_step["step"] + 1,
                        })
                        ep_actions = [s["action"] for s in steps_d]

                    ep_rgb = np.array([s["rgb"] for s in steps_d])
                    ep_depth = np.array([s["depth"] for s in steps_d])
                    third_list = [s["third_rgb"] for s in steps_d]
                    ep_third = np.array(third_list) if all(x is not None for x in third_list) else None
                    ep_hnum = np.array([s["human_num"] for s in steps_d])
                    ep_dists2 = [s["distance_to_goal"] for s in steps_d]
                    ep_piv = [s.get("pedestrian_in_view", 0) for s in steps_d]
                    ep_traj = [s.get("trajectory", {}) for s in steps_d]

                    if rank0_only():
                        save_to_disk(
                            ep_rgb, ep_depth, ep_third,
                            ep_hnum, ep_actions, ep_dists2,
                            ep_id, scene_id,
                            pedestrian_in_view=ep_piv,
                            trajectories=ep_traj,
                            split=split_name,
                            data_folder=data_folder,
                        )
                        # 统计动作分布
                        act_dist = {
                            SFMExpertPlanner.ACTION_NAMES[a]: ep_actions.count(a)
                            for a in range(6)
                        }
                        logger.info(
                            f"[SAVED] scene={scene_name} ep={ep_id} "
                            f"steps={len(ep_actions)} "
                            f"actions={act_dist} "
                            f"final_dist={final_dist:.3f}"
                        )

                    saved_episode_ids.add(ep_key)
                    collected_eps += 1
                    successful_eps += 1
                    global_dist[env_idx] = float("inf")

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"saved={successful_eps} fail={failed_eps} skip={skipped_eps} steps={total_steps}"
                        )

        # ── 采集结束统计 ──
        if rank0_only():
            logger.info(
                f"\n{'='*60}\n"
                f"[6ActionExpert] Collection complete!\n"
                f"  Collected : {collected_eps}\n"
                f"  Successful: {successful_eps}\n"
                f"  Failed    : {failed_eps}\n"
                f"  Skipped   : {skipped_eps}\n"
                f"  TotalSteps: {total_steps}\n"
                f"  ActionDist: {action_counter}\n"
                f"{'='*60}"
            )
            logger.info(f"Data saved to: {data_folder}")

        if pbar is not None:
            pbar.close()
        self.envs.close()

    # ──────────────────────────────────────────────────────────────
    #  BaseRLTrainer 接口
    # ──────────────────────────────────────────────────────────────

    @profiling_wrapper.RangeContext("collect_data_6action")
    def train(self) -> None:
        """主入口：调用 collect_expert_data()。"""
        self.collect_expert_data()

    def save_checkpoint(self, file_name: str, extra_state=None) -> None:
        pass  # 数据采集器不需要检查点

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        return {}


# ═══════════════════════════════════════════════════════════════════
#  辅助函数（模块级）
# ═══════════════════════════════════════════════════════════════════

def _get_device(config) -> torch.device:
    """根据配置获取计算设备。"""
    if torch.cuda.is_available():
        device = torch.device("cuda", config.habitat_baselines.torch_gpu_id)
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")
