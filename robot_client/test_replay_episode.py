#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Episode 回放测试 — 对齐机器人端完整数据流

服务端保留的全部接口:
  POST /reset_hiddens            → 初始化 RNN hidden states
    可选 form: ep_id, inst        (机器人端可不传)

  POST /predict_action            → 返回预测动作
    必传 files: rgb (JPEG)
    可选 files: depth (PNG uint16 mm)
    可选 form:  ep_id, inst, goal_x, goal_y, compass
              (机器人端不传时默认 0.0)

机器人端实际数据流:
  1. RealSense BGR (1280×720) + Depth uint16 mm
  2. crop_and_resize → 224×224
  3. JPEG (rgb) + PNG (depth) 编码
  4. POST /reset_hiddens  (空 POST)
  5. 逐帧 POST /predict_action  {ep_id, inst} + files {rgb, depth}
  6. 通常不发送 goal_x / goal_y / compass

用法:
  # 模拟机器人端 (不发送 GPS/compass)
  python test_replay_episode.py \\
      --server-url http://127.0.0.1:32145 \\
      --ep-dir /path/to/episode

  # 完整接口测试 (发送 GPS/compass，使用轨迹数据)
  python test_replay_episode.py \\
      --server-url http://127.0.0.1:32145 \\
      --ep-dir /path/to/episode \\
      --with-gps
"""

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] cv2 not available, using PIL fallback for image resize")

ACTION_4_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
ACTION_6_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT", 4: "PAUSE", 5: "BACKWARD"}


# ═══════════════════════════════════════════════════════════════════════════
# 图像预处理 — 对齐机器人端 crop_and_resize
# ═══════════════════════════════════════════════════════════════════════════

def crop_and_resize(img, size=224):
    """中心裁剪 + 缩放 — 与机器人端 PathInference 完全一致"""
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    if HAS_CV2:
        return cv2.resize(cropped, (size, size))
    else:
        from PIL import Image
        return np.array(Image.fromarray(cropped).resize((size, size), Image.BILINEAR))


def rgb_to_jpeg_bytes(rgb_np):
    """RGB (H,W,3) uint8 → JPEG bytes"""
    if HAS_CV2:
        _, enc = cv2.imencode('.jpg', cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        return enc.tobytes()
    else:
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(rgb_np).save(buf, format='JPEG')
        return buf.getvalue()


def depth_to_png_bytes(depth_np):
    """Depth (H,W) uint16 → PNG bytes"""
    if HAS_CV2:
        _, enc = cv2.imencode('.png', depth_np)
        return enc.tobytes()
    else:
        buf = io.BytesIO()
        from PIL import Image
        Image.fromarray(depth_np).save(buf, format='PNG')
        return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# GPS / compass 计算 — 从轨迹数据提取
# ═══════════════════════════════════════════════════════════════════════════

def quaternion_to_yaw(q):
    """四元数 [x, y, z, w] → yaw (rad)"""
    x, y, z, w = q
    return math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))


def compute_gps_compass(robot_data, goal_pos):
    """
    从机器人位姿计算 goal_x, goal_y, compass。

    Args:
        robot_data: {"position": [x,y,z], "rotation": [x,y,z,w]}
        goal_pos:   [x, y, z]  目标绝对坐标（通常为轨迹最后一帧位置）

    Returns:
        (goal_x, goal_y, compass)  — compass 为世界坐标系 yaw (rad)
    """
    pos = robot_data.get("position", [0, 0, 0])
    rot = robot_data.get("rotation", [0, 0, 0, 1])

    yaw = quaternion_to_yaw(rot)

    if goal_pos is not None:
        dx = goal_pos[0] - pos[0]
        dz = goal_pos[2] - pos[2]
        # 世界坐标系 → 机器人坐标系
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        goal_x = dx * cos_y + dz * sin_y
        goal_y = -dx * sin_y + dz * cos_y
    else:
        goal_x, goal_y = 5.0, 0.0

    return goal_x, goal_y, yaw


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_episode_data(ep_dir: str, load_trajectories: bool = False):
    """加载 episode 全部数据"""
    ep_path = Path(ep_dir)

    # 1. RGB
    rgb_dir = ep_path / "rgb"
    if not rgb_dir.exists():
        rgb_dir = ep_path / "third_rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"No rgb/ or third_rgb/ found in {ep_dir}")
    rgb_files = sorted(rgb_dir.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[0]))
    if not rgb_files:
        rgb_files = sorted(rgb_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[0]))
    print(f"[Data] RGB: {len(rgb_files)} frames from {rgb_dir}")

    # 2. GT actions
    gt_actions = []
    action_dir = ep_path / "action"
    if action_dir.exists():
        af = sorted(action_dir.glob("*.json"))
        if af:
            with open(af[0]) as f:
                gt_actions = json.load(f)
            print(f"[Data] GT actions: {len(gt_actions)}")

    # 3. Instruction
    instruction = ""
    for d in ["instruction_level_2", "instruction_val_level_2",
              "instruction_vl_level_1", "instruction_vl_level_2",
              "inst_navcomposer_v2"]:
        inst_dir = ep_path / d
        if inst_dir.exists():
            tfs = list(inst_dir.glob("*.txt"))
            if tfs:
                with open(tfs[0]) as f:
                    instruction = f.read().strip()
                break
    if not instruction:
        instruction = "Go to the target location"
    print(f"[Data] Instruction: {instruction[:100]}...")

    # 4. Depth (optional)
    depth_files = []
    depth_dir = ep_path / "depth"
    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem.split("_")[0]))
        if depth_files:
            print(f"[Data] Depth: {len(depth_files)} frames from {depth_dir}")

    # 5. Trajectories (for GPS/compass)
    trajectories = None
    if load_trajectories:
        traj_dir = ep_path / "trajectories"
        if traj_dir.exists():
            tf = sorted(traj_dir.glob("*.json"))
            if tf:
                with open(tf[0]) as f:
                    trajectories = json.load(f)
                print(f"[Data] Trajectories: {len(trajectories)} entries (for GPS/compass)")

    # 6. distance_to_goal
    d2g_list = None
    d2g_dir = ep_path / "distance_to_goal"
    if d2g_dir.exists():
        df = sorted(d2g_dir.glob("*.json"))
        if df:
            with open(df[0]) as f:
                d2g_list = json.load(f)

    return {
        "rgb_files": rgb_files,
        "depth_files": depth_files,
        "gt_actions": gt_actions,
        "instruction": instruction,
        "trajectories": trajectories,
        "d2g_list": d2g_list,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 主测试
# ═══════════════════════════════════════════════════════════════════════════

def test_episode_replay(server_url: str, ep_dir: str,
                        use_6_actions: bool = False,
                        send_depth: bool = True,
                        send_gps: bool = False,
                        image_size: int = 224):
    """主测试 — 对齐机器人端完整数据流"""
    server_url = server_url.rstrip("/")

    data = load_episode_data(ep_dir, load_trajectories=send_gps)
    rgb_files = data["rgb_files"]
    depth_files = data["depth_files"]
    gt_actions = data["gt_actions"]
    instruction = data["instruction"]
    trajectories = data["trajectories"]
    d2g_list = data["d2g_list"]

    has_depth = len(depth_files) > 0 and send_depth
    total_frames = len(rgb_files)
    action_names = ACTION_6_NAMES if use_6_actions else ACTION_4_NAMES

    # 预计算 goal position
    goal_pos = None
    if send_gps and trajectories and len(trajectories) > 0:
        goal_pos = trajectories[-1].get("robot", {}).get("position", None)

    # 打印发送字段清单
    fields_sent = ["ep_id", "inst", "rgb (JPEG)"]
    if has_depth:
        fields_sent.append("depth (PNG uint16)")
    if send_gps:
        fields_sent.extend(["goal_x", "goal_y", "compass"])
    else:
        fields_sent.append("goal_x/y/compass → DEFAULT 0.0")

    print(f"\n{'='*70}")
    print(f"Episode Replay Test")
    print(f"{'='*70}")
    print(f"  Server:        {server_url}")
    print(f"  Episode:       {Path(ep_dir).name}")
    print(f"  Frames:        {total_frames}")
    print(f"  Action space:  {'6-action' if use_6_actions else '4-action'}")
    print(f"  Crop+resize:   → {image_size}×{image_size}")
    print(f"  Fields sent:   {', '.join(fields_sent)}")
    print(f"  Instruction:   {instruction[:80]}...")
    if gt_actions:
        print(f"  GT actions:    {len(gt_actions)}")
    print(f"{'='*70}\n")

    session = requests.Session()

    # ═════════════════════════════════════════════════════════════════
    # Step 1: Reset
    # ═════════════════════════════════════════════════════════════════
    print("[1/3] Resetting hidden states ...")
    t0 = time.time()
    resp = session.post(f"{server_url}/reset_hiddens", timeout=10)
    resp.raise_for_status()
    rr = resp.json()
    print(f"  Status: {rr.get('status')} | Brain: {rr.get('brain_enabled', False)} "
          f"| {((time.time()-t0)*1000):.0f}ms")

    # ═════════════════════════════════════════════════════════════════
    # Step 2: 逐帧推理
    # ═════════════════════════════════════════════════════════════════
    print(f"\n[2/3] Running {total_frames} frames ...\n")

    predictions = []
    total_model_ms = 0.0
    correct = 0
    compared = 0

    hdr = (f"{'Step':>4s} | {'Pred':>8s} | {'GT':>8s} | {'Match':>5s} | "
           f"{'Total':>7s} | {'Model':>7s} | {'D2G':>8s}")
    if send_gps:
        hdr += " | {'GX':>6s} | {'GY':>6s}"
    print(hdr)
    print("-" * (90 if send_gps else 78))

    for step_idx in range(total_frames):
        # 2a. RGB → crop_and_resize → JPEG
        from PIL import Image
        rgb_np = np.array(Image.open(rgb_files[step_idx]).convert("RGB"))
        rgb_bytes = rgb_to_jpeg_bytes(crop_and_resize(rgb_np, image_size))

        # 2b. Depth → crop_and_resize → PNG
        depth_bytes = None
        if has_depth and step_idx < len(depth_files):
            d_np = np.array(Image.open(depth_files[step_idx]))
            depth_bytes = depth_to_png_bytes(crop_and_resize(d_np, image_size))

        # 2c. GPS/compass
        goal_x, goal_y, compass = 0.0, 0.0, 0.0
        if send_gps and trajectories and step_idx < len(trajectories):
            rd = trajectories[step_idx].get("robot", {})
            goal_x, goal_y, compass = compute_gps_compass(rd, goal_pos)

        # 2d. 构建请求
        files = {"rgb": (f"{step_idx}.jpg", rgb_bytes, "image/jpeg")}
        if depth_bytes is not None:
            files["depth"] = (f"{step_idx}.png", depth_bytes, "image/png")

        form_data = {
            "ep_id": f"replay_{Path(ep_dir).name}",
            "inst": instruction,
        }
        if send_gps:
            form_data["goal_x"] = str(goal_x)
            form_data["goal_y"] = str(goal_y)
            form_data["compass"] = str(compass)

        t_start = time.time()
        resp = session.post(
            f"{server_url}/predict_action",
            data=form_data,
            files=files,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != "success":
            print(f"  [ERROR] Frame {step_idx}: {result.get('message')}")
            break

        pred_action = result["action"]
        predictions.append(pred_action)

        ti = result.get("time_info", {})
        t_total = (ti.get("total", 0) or 0)
        t_model = (ti.get("model", 0) or 0)
        if t_total < 1.0:  # seconds → ms
            t_total *= 1000
            t_model *= 1000
        total_model_ms += t_model

        gt_action = gt_actions[step_idx] if step_idx < len(gt_actions) else None
        is_correct = (gt_action is not None and pred_action == gt_action)
        if gt_action is not None:
            if is_correct:
                correct += 1
            compared += 1

        d2g = d2g_list[step_idx] if d2g_list and step_idx < len(d2g_list) else 0.0

        pn = action_names.get(pred_action, f"UNK_{pred_action}")
        gn = action_names.get(gt_action, "N/A") if gt_action is not None else "N/A"
        ms = "✓" if is_correct else ("✗" if gt_action is not None else "-")

        extras = []
        if result.get("pedestrian_detected"):
            extras.append(f"PEDx{result.get('pedestrian_count', 0)}")
        if result.get("instruction_modified"):
            extras.append("BRAIN")
        extra_str = f" | {' | '.join(extras)}" if extras else ""

        line = (f"{step_idx:4d} | {pn:>8s} | {gn:>8s} | {ms:>5s} | "
                f"{t_total:6.0f}ms | {t_model:6.0f}ms | {d2g:7.2f}m")
        if send_gps:
            line += f" | {goal_x:5.1f} | {goal_y:5.1f}"
        print(line + extra_str)

        if pred_action == 0:
            print(f"\n  >>> STOP predicted at step {step_idx + 1}")
            break

    # ═════════════════════════════════════════════════════════════════
    # Step 3: 统计
    # ═════════════════════════════════════════════════════════════════
    n = len(predictions)
    print(f"\n{'='*70}\n[3/3] Results\n{'='*70}")
    print(f"  Frames processed:     {n}")
    if compared > 0:
        print(f"  GT action accuracy:   {correct}/{compared} = {correct/compared*100:.1f}%")
    else:
        print(f"  GT action accuracy:   N/A")
    print(f"  Avg model latency:    {total_model_ms/n:.1f}ms" if n else "")
    print(f"  GPS/compass sent:     {send_gps}")
    print(f"  Depth sent:           {has_depth}")

    ac = {}
    for a in predictions:
        ac[a] = ac.get(a, 0) + 1
    print(f"  Predicted actions:")
    for aid in sorted(ac):
        print(f"    {action_names.get(aid, f'UNK_{aid}')}: {ac[aid]}")

    if gt_actions:
        gc = {}
        for a in gt_actions[:n]:
            gc[a] = gc.get(a, 0) + 1
        print(f"  GT actions:")
        for aid in sorted(gc):
            print(f"    {action_names.get(aid, f'UNK_{aid}')}: {gc[aid]}")

    session.close()
    return predictions


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Episode 回放测试 — 对齐机器人端完整数据流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 模拟机器人 (不发送 GPS/compass/depth)
  python test_replay_episode.py --server-url http://127.0.0.1:32145 \\
      --ep-dir /path/to/episode --no-depth

  # 完整接口测试 (发送 GPS/compass + depth)
  python test_replay_episode.py --server-url http://127.0.0.1:32145 \\
      --ep-dir /path/to/episode --with-gps
        """,
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:32145")
    parser.add_argument("--ep-dir",
                        default="/share/home/u19666033/dhj/DPed_pro/data/collect_data/train/1EiJpeRNEs1.basis/14")
    parser.add_argument("--num-actions", type=int, choices=[4, 6], default=4)
    parser.add_argument("--no-depth", action="store_true",
                        help="不发送深度图像")
    parser.add_argument("--with-gps", action="store_true",
                        help="从 trajectory 数据提取并发送 goal_x/goal_y/compass")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if not os.path.isdir(args.ep_dir):
        print(f"错误: episode 目录不存在: {args.ep_dir}")
        sys.exit(1)

    test_episode_replay(
        server_url=args.server_url,
        ep_dir=args.ep_dir,
        use_6_actions=(args.num_actions == 6),
        send_depth=not args.no_depth,
        send_gps=args.with_gps,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
