#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证机器人端到服务器端的图像/指令传输是否正常

用法:
    在机器人端终端运行:
        python test_network_transmission.py --url http://47.116.197.118:4173

    可选参数:
        --use-local-images /path/to/infer_rgb/  使用本地保存的图像进行测试
        --num-steps 5                             发送多少步的请求
"""

import argparse
import io
import json
import os
import sys
import time
import numpy as np
import requests
from PIL import Image


def create_test_image(step, size=224):
    """
    创建测试图像 - 每个step产生明显不同的图像
    step=0: 纯红色
    step=1: 纯绿色
    step=2: 纯蓝色
    step=3: 黑白渐变
    step=4: 随机噪声
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if step % 5 == 0:
        # 红色 + step变化
        img[:, :, 0] = 200 + step * 5
        img[:, :, 1] = 0
        img[:, :, 2] = 0
        label = f"Red (brightness={200+step*5})"
    elif step % 5 == 1:
        # 绿色
        img[:, :, 0] = 0
        img[:, :, 1] = 200 + step * 5
        img[:, :, 2] = 0
        label = f"Green (brightness={200+step*5})"
    elif step % 5 == 2:
        # 蓝色
        img[:, :, 0] = 0
        img[:, :, 1] = 0
        img[:, :, 2] = 200 + step * 5
        label = f"Blue (brightness={200+step*5})"
    elif step % 5 == 3:
        # 垂直条纹 (每step移动条纹位置)
        for x in range(size):
            intensity = 255 if ((x + step * 20) // 30) % 2 == 0 else 0
            img[:, x, :] = intensity
        label = f"Vertical stripes (offset={step*20})"
    else:
        # 随机噪声 (每step固定种子，可复现但不同)
        rng = np.random.RandomState(step)
        img = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
        label = f"Random noise (seed={step})"

    return img, label


def create_test_depth(step, size=224):
    """创建测试深度图（单通道，模拟mm单位的uint16）"""
    if step % 5 == 0:
        depth = np.full((size, size), 1500 + step * 100, dtype=np.uint16)  # 1.5m ~ 2.5m
    elif step % 5 == 1:
        depth = np.full((size, size), 3000 + step * 100, dtype=np.uint16)  # 3.0m ~ 4.0m
    elif step % 5 == 2:
        # 距离梯度
        depth = np.linspace(500, 5000, size, dtype=np.uint16)
        depth = np.tile(depth, (size, 1))
    elif step % 5 == 3:
        depth = np.full((size, size), 2000, dtype=np.uint16)
    else:
        rng = np.random.RandomState(step + 1000)
        depth = (rng.rand(size, size) * 4000 + 1000).astype(np.uint16)

    return depth


def post_json(base_url, endpoint, files=None, data=None, timeout=120):
    """发送 POST 请求，返回 JSON"""
    url = f"{base_url}{endpoint}"
    try:
        resp = requests.post(url, timeout=timeout, files=files, data=data)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 请求失败: {url} - {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  响应内容: {e.response.text[:500]}")
        raise
    except ValueError as e:
        print(f"[ERROR] 响应不是合法 JSON: {url} - {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="测试 VLN 服务端传输")
    parser.add_argument("--url", default="http://47.116.197.118:4173",
                        help="服务器 URL (默认: http://47.116.197.118:4173)")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="测试步数 (默认: 5)")
    parser.add_argument("--use-local-images", default=None,
                        help="使用本地图像目录 (如: /path/to/infer_rgb/) 代替生成图像")
    parser.add_argument("--instruction", default="Go straight and turn left at the hallway.",
                        help="测试指令")
    parser.add_argument("--no-reset", action="store_true",
                        help="跳过 reset_hiddens 调用")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"=" * 70)
    print(f"网络传输测试")
    print(f"  服务器: {base_url}")
    print(f"  步数: {args.num_steps}")
    print(f"  指令: {args.instruction}")
    print(f"=" * 70)

    # Step 1: Reset hidden states
    if not args.no_reset:
        print("\n[1] 重置 RNN 隐藏状态...")
        try:
            resp = post_json(base_url, "/reset_hiddens")
            print(f"  响应: {resp}")
        except Exception as e:
            print(f"  [FAIL] reset_hiddens 失败: {e}")
            print("  请检查服务器是否在运行，以及 FRP 隧道是否正常。")
            sys.exit(1)

    # Step 2: 准备测试图像
    if args.use_local_images and os.path.isdir(args.use_local_images):
        print(f"\n[2] 使用本地图像: {args.use_local_images}")
        image_files = sorted([
            f for f in os.listdir(args.use_local_images)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])[:args.num_steps]
        if len(image_files) < args.num_steps:
            print(f"  [WARN] 只有 {len(image_files)} 张图像，将重复使用")
            while len(image_files) < args.num_steps:
                image_files.append(image_files[-1])
        use_synthetic = False
    else:
        print(f"\n[2] 使用合成测试图像 ({args.num_steps} 张明显不同的图像)")
        use_synthetic = True

    # Step 3: 发送请求并比较
    print(f"\n[3] 发送 {args.num_steps} 个 /predict_action 请求...\n")

    actions = []
    timings = []

    for step in range(args.num_steps):
        print(f"--- Step {step+1}/{args.num_steps} ---")

        # 准备图像
        if use_synthetic:
            rgb_img, img_label = create_test_image(step)
            depth_img = create_test_depth(step)
            print(f"  图像: {img_label}")

            # 编码为 JPEG/PNG bytes
            rgb_pil = Image.fromarray(rgb_img)
            rgb_buf = io.BytesIO()
            rgb_pil.save(rgb_buf, format="JPEG", quality=95)
            rgb_bytes = rgb_buf.getvalue()

            depth_pil = Image.fromarray(depth_img)
            depth_buf = io.BytesIO()
            depth_pil.save(depth_buf, format="PNG")
            depth_bytes = depth_buf.getvalue()

            print(f"  RGB大小: {len(rgb_bytes)} bytes, Depth大小: {len(depth_bytes)} bytes")
        else:
            img_path = os.path.join(args.use_local_images, image_files[step])
            with open(img_path, 'rb') as f:
                rgb_bytes = f.read()
            print(f"  图像文件: {image_files[step]} ({len(rgb_bytes)} bytes)")
            # 合成深度图
            depth_img = create_test_depth(step)
            depth_pil = Image.fromarray(depth_img)
            depth_buf = io.BytesIO()
            depth_pil.save(depth_buf, format="PNG")
            depth_bytes = depth_buf.getvalue()

        # 显示 RGB 图像基本信息
        verify_img = Image.open(io.BytesIO(rgb_bytes))
        verify_np = np.array(verify_img)
        print(f"  图像尺寸: {verify_img.size}, 模式: {verify_img.mode}")
        if verify_np.ndim == 3:
            print(f"  RGB统计: R({verify_np[...,0].min()}-{verify_np[...,0].max()}) "
                  f"G({verify_np[...,1].min()}-{verify_np[...,1].max()}) "
                  f"B({verify_np[...,2].min()}-{verify_np[...,2].max()})")

        # 发送请求
        files = {
            "rgb": (f"{step}.jpg", rgb_bytes, "image/jpeg"),
            "depth": (f"{step}.png", depth_bytes, "image/png"),
        }
        data = {
            "ep_id": f"test_{step}",
            "inst": args.instruction,
        }

        try:
            tic = time.time()
            resp = post_json(base_url, "/predict_action", files=files, data=data)
            elapsed = time.time() - tic

            action = resp.get("action", "N/A")
            status = resp.get("status", "N/A")
            time_info = resp.get("time_info", {})

            actions.append(action)
            timings.append(elapsed)

            print(f"  状态: {status}")
            print(f"  动作: {action} ({['STOP','FORWARD','LEFT','RIGHT'][action] if isinstance(action, int) and 0 <= action <= 3 else 'UNKNOWN'})")
            print(f"  耗时: {elapsed:.3f}s (model: {time_info.get('model', 'N/A')})")

        except Exception as e:
            print(f"  [FAIL] 请求失败: {e}")
            actions.append(None)
            timings.append(None)

    # Step 4: 汇总
    print(f"\n" + "=" * 70)
    print(f"测试汇总")
    print(f"=" * 70)
    print(f"  所有动作: {actions}")
    print(f"  唯一动作数: {len(set(a for a in actions if a is not None))}")
    print(f"  耗时: {' → '.join(f'{t:.1f}s' for t in timings if t is not None)}")

    if len(set(a for a in actions if a is not None)) <= 1:
        print(f"\n  ⚠️  警告: 所有图像产生了相同的动作!")
        print(f"  可能原因:")
        print(f"    1. 模型没有接收到不同的图像（检查 FRP/网络传输是否丢包）")
        print(f"    2. 模型输入预处理有问题（图像被归一化为相同值）")
        print(f"    3. RNN 隐藏状态没有正确更新")
        print(f"    4. 模型权重问题")
        print(f"  请查看服务器端打印的 DEBUG 输出来确定具体原因。")
    else:
        print(f"\n  ✅ 不同的图像产生了不同的动作，图像传输正常工作。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
