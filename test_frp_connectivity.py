#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRP 网络连通性测试 - 不依赖模型，仅测试 HTTP 传输是否正常

用法:
    # 在服务器端 (GPU 机器) 运行:
    python test_frp_connectivity.py --mode server --port 32145

    # 在机器人端 (通过 FRP) 运行:
    python test_frp_connectivity.py --mode client --url http://47.116.197.118:4173
"""

import argparse
import io
import json
import sys
import time
import numpy as np
import requests
from PIL import Image


def run_server(port):
    """启动简单的 Flask 测试服务器"""
    from flask import Flask, request
    app = Flask("frp_test")

    @app.route("/ping", methods=["GET", "POST"])
    def ping():
        return {"status": "ok", "message": "Server is reachable"}

    @app.route("/echo", methods=["POST"])
    def echo():
        """回显接收到的图像统计信息和指令"""
        result = {"status": "ok"}

        # 检查图像文件
        rgb_file = request.files.get("rgb")
        if rgb_file:
            rgb_bytes = rgb_file.read()
            img = Image.open(io.BytesIO(rgb_bytes))
            img_np = np.array(img)
            result["rgb"] = {
                "size_bytes": len(rgb_bytes),
                "image_shape": list(img_np.shape),
                "dtype": str(img_np.dtype),
            }
            if img_np.ndim == 3 and img_np.shape[-1] >= 3:
                for i, ch in enumerate(["R", "G", "B"]):
                    result["rgb"][f"{ch}_min"] = float(img_np[..., i].min())
                    result["rgb"][f"{ch}_max"] = float(img_np[..., i].max())
                    result["rgb"][f"{ch}_mean"] = float(img_np[..., i].mean())
            # 计算图像"指纹"用于验证不同图像
            result["rgb"]["fingerprint"] = float(img_np.sum())

        depth_file = request.files.get("depth")
        if depth_file:
            depth_bytes = depth_file.read()
            depth_img = Image.open(io.BytesIO(depth_bytes))
            depth_np = np.array(depth_img)
            result["depth"] = {
                "size_bytes": len(depth_bytes),
                "image_shape": list(depth_np.shape),
                "fingerprint": float(depth_np.sum()),
            }

        # 检查表单字段
        inst = request.form.get("inst", "")
        result["instruction"] = inst
        result["instruction_len"] = len(inst)

        ep_id = request.form.get("ep_id", "")
        result["ep_id"] = ep_id

        return result

    print(f"[SERVER] 启动测试服务器 0.0.0.0:{port}")
    app.run("0.0.0.0", port, threaded=True)


def run_client(base_url, num_steps=5):
    """通过 FRP 向服务器发送测试请求"""
    base_url = base_url.rstrip("/")
    print(f"[CLIENT] 测试服务器: {base_url}")
    print(f"[CLIENT] 步数: {num_steps}")

    # 1. Ping 测试
    print("\n[1] Ping 测试...")
    try:
        resp = requests.get(f"{base_url}/ping", timeout=10)
        print(f"  响应: {resp.json()}")
        print("  ✅ Ping 成功，网络连通")
    except Exception as e:
        print(f"  ❌ Ping 失败: {e}")
        print("  请确认 FRP 隧道已建立: frpc -c frpc2.ini")
        return 1

    # 2. Echo 测试 - 发送不同图像
    print(f"\n[2] Echo 测试 ({num_steps} 步)...")

    fingerprints = []
    for step in range(num_steps):
        print(f"\n--- Step {step+1}/{num_steps} ---")

        # 生成明显不同的测试图像
        size = 224
        img = np.zeros((size, size, 3), dtype=np.uint8)
        if step % 3 == 0:
            # 红色渐变
            img[:, :, 0] = np.linspace(0, 255, size).astype(np.uint8).reshape(1, -1)
            label = "Red gradient"
        elif step % 3 == 1:
            # 绿色渐变
            img[:, :, 1] = np.linspace(0, 255, size).astype(np.uint8).reshape(1, -1)
            label = "Green gradient"
        else:
            # 蓝色渐变
            img[:, :, 2] = np.linspace(0, 255, size).astype(np.uint8).reshape(1, -1)
            label = "Blue gradient"

        # 编码
        rgb_pil = Image.fromarray(img)
        rgb_buf = io.BytesIO()
        rgb_pil.save(rgb_buf, format="JPEG", quality=95)
        rgb_bytes = rgb_buf.getvalue()

        depth_img = np.full((size, size), 1000 + step * 500, dtype=np.uint16)
        depth_pil = Image.fromarray(depth_img)
        depth_buf = io.BytesIO()
        depth_pil.save(depth_buf, format="PNG")
        depth_bytes = depth_buf.getvalue()

        print(f"  发送图像: {label}")
        print(f"  RGB size: {len(rgb_bytes)} bytes, Depth size: {len(depth_bytes)} bytes")
        print(f"  发送 fingerprint (客户端): {img.sum():.0f}")

        files = {
            "rgb": (f"{step}.jpg", rgb_bytes, "image/jpeg"),
            "depth": (f"{step}.png", depth_bytes, "image/png"),
        }
        data = {
            "ep_id": f"test_{step}",
            "inst": f"Go straight and turn left at step {step}.",
        }

        try:
            resp = requests.post(f"{base_url}/echo", files=files, data=data, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            if result.get("rgb"):
                server_fp = result["rgb"]["fingerprint"]
                print(f"  接收 fingerprint (服务器): {server_fp:.0f}")
                fingerprints.append(server_fp)
                if step > 0 and server_fp == fingerprints[step-1]:
                    print(f"  ⚠️  图像指纹与上一步相同！传输可能有问题。")
                else:
                    print(f"  ✅ 图像正确传输（指纹与前一步不同）")

            inst_back = result.get("instruction", "")
            print(f"  回显指令: [{inst_back}] (len={result.get('instruction_len', 0)})")

        except Exception as e:
            print(f"  ❌ 请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"     响应: {e.response.text[:300]}")

    # 汇总
    print(f"\n[3] 汇总")
    print(f"  所有指纹: {fingerprints}")
    unique_fps = len(set(fingerprints))
    if unique_fps == len(fingerprints) and unique_fps > 1:
        print(f"  ✅ 所有 {len(fingerprints)} 张图像均正确传输且互不相同")
    elif unique_fps == 1:
        print(f"  ❌ 所有图像指纹相同 - 服务器收到了相同的图像数据")
    else:
        print(f"  ⚠️  有 {len(fingerprints)} 张图像，但只有 {unique_fps} 个独特指纹")

    return 0


def main():
    parser = argparse.ArgumentParser(description="FRP 网络连通性测试")
    parser.add_argument("--mode", choices=["server", "client"], default="client",
                        help="运行模式: server (服务器端) 或 client (机器人端)")
    parser.add_argument("--port", type=int, default=32145,
                        help="服务器端口 (server 模式)")
    parser.add_argument("--url", default="http://47.116.197.118:4173",
                        help="服务器 URL (client 模式)")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="测试步数 (client 模式)")
    args = parser.parse_args()

    if args.mode == "server":
        return run_server(args.port)
    else:
        return run_client(args.url, args.num_steps)


if __name__ == "__main__":
    sys.exit(main())
