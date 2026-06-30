#!/usr/bin/env python3
"""
将 DPed_pro_resplit 数据集转换为 StreamVLN/NaviLLa 训练格式。

DPed_pro_resplit 的 episode 格式是 DynamicVLNCEEpisode (Habitat)，
需转换为 LLaVA-style 的 conversation JSON 格式以支持 SFT/LoRA 微调。

输出格式 (每行一个 JSON):
{
    "id": "train_scene_ep001",
    "video": "path/to/frames/",        # 帧图像目录
    "conversations": [
        {"from": "human", "value": "<video>\n<instruction>"},
        {"from": "gpt", "value": "<action_sequence>"}
    ]
}
"""

import argparse
import gzip
import json
import os
import sys
from pathlib import Path


def convert_episode(ep, split: str, scene_name: str) -> dict:
    """将单个 episode 转换为训练格式"""
    instruction = ep.get("instruction", "")
    gt_action = ep.get("gt_action", [])
    episode_id = ep.get("episode_id", "")
    scene_id = ep.get("scene_id", "")

    # 构建唯一 ID
    uid = f"{split}_{scene_name}_ep{episode_id}"

    # 动作序列转为文本（用于 SFT）
    # 0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT
    action_map = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
    action_tokens = [action_map.get(a, f"ACTION_{a}") for a in gt_action]
    action_text = ", ".join(action_tokens)

    # LLaVA conversation 格式
    conversations = [
        {"from": "human", "value": f"<video>\n{instruction}\nWhat are the next navigation actions?"},
        {"from": "gpt", "value": action_text},
    ]

    return {
        "id": uid,
        "video": "",  # 由 Habitat 运行时提供帧
        "conversations": conversations,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "gt_action": gt_action,
        "instruction": instruction,
        "start_position": ep.get("start_position", []),
        "goals": ep.get("goals", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert DPed_pro dataset to training format")
    parser.add_argument("--input_dir", type=str,
                        default="/share/home/u19666033/dhj/DPed_pro/dped_pro_resplit/train",
                        help="Input directory containing scene .json.gz files")
    parser.add_argument("--output", type=str,
                        default="/share/home/u19666033/dhj/DPed_pro/dped_pro_resplit/train_converted.json",
                        help="Output JSON file path")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split name")
    args = parser.parse_args()

    all_converted = []
    total_eps = 0
    skipped = 0

    for fn in sorted(os.listdir(args.input_dir)):
        if not fn.endswith(".json.gz"):
            continue
        fpath = os.path.join(args.input_dir, fn)
        scene_name = fn.replace(".json.gz", "")

        with gzip.open(fpath, "rt") as f:
            data = json.load(f)

        episodes = data.get("episodes", [])
        for ep in episodes:
            total_eps += 1
            converted = convert_episode(ep, args.split, scene_name)
            all_converted.append(converted)

    # 写为 JSONL 格式（每行一个 JSON 对象）
    with open(args.output, "w") as f:
        json.dump(all_converted, f, ensure_ascii=False)

    print(f"Converted {len(all_converted)}/{total_eps} episodes to {args.output}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
