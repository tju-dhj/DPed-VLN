#!/usr/bin/env python3
"""
将 DPed-VLN 数据集转换为 StreamVLN train 格式。

DPed-VLN: dped-vln/DPed_VLN/data_sets/{v1|v2}/train/{scene}.json.gz
Collect:  DPed_pro/data/collect_data/train/{scene}.basis/{episode_id}/rgb/

输出格式 (供 streamvln_train.py 使用):
  {output_dir}/
    annotations.json    # [{"id": 0, "video": "v1/train/scene_ep", "instructions": [...], "actions": [...]}, ...]
    v1/train/
      {scene}_{episode_id}/
        rgb/  -> symlink to collect_data frames
"""

import argparse
import gzip
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dped_data", type=str,
                        default="/share/home/u19666033/dhj/dped-vln/DPed_VLN/data_sets")
    parser.add_argument("--collect_data", type=str,
                        default="/share/home/u19666033/dhj/DPed_pro/data/collect_data/train")
    parser.add_argument("--level", type=str, default="v1",
                        help="v1 or v2")
    parser.add_argument("--output_dir", type=str,
                        default="/share/home/u19666033/dhj/dped-vln/DPed_VLN/streamvln_training_data")
    parser.add_argument("--max_episodes", type=int, default=-1,
                        help="Max episodes per scene (-1 = all)")
    parser.add_argument("--min_actions", type=int, default=4,
                        help="Minimum actions required (StreamVLN needs >= 4)")
    args = parser.parse_args()

    train_dir = os.path.join(args.dped_data, args.level, "train")
    video_root = os.path.join(args.output_dir, args.level, "train")
    os.makedirs(video_root, exist_ok=True)

    all_annotations = []
    global_id = 0
    total_eps = 0
    skipped_short = 0
    skipped_no_frames = 0
    total_used = 0

    for fn in sorted(os.listdir(train_dir)):
        if not fn.endswith(".json.gz"):
            continue

        scene_name = fn.replace(".json.gz", "")
        # scene_id format: hm3d/train/XXXX/scene.basis.glb
        # collect_data uses: scene.basis/episode_id/
        collect_scene = f"{scene_name}.basis"
        collect_scene_dir = os.path.join(args.collect_data, collect_scene)

        scene_path = os.path.join(train_dir, fn)
        with gzip.open(scene_path, "rt") as f:
            data = json.load(f)

        episodes = data.get("episodes", [])
        used_this_scene = 0

        for ep in episodes:
            total_eps += 1
            episode_id = str(ep.get("episode_id", ""))
            gt_action = ep.get("gt_action", [])
            instruction = ep.get("instruction", "")

            # Filter: need at least min_actions
            if len(gt_action) < args.min_actions:
                skipped_short += 1
                continue

            # Check collect_data has frames for this episode
            collect_ep_dir = os.path.join(collect_scene_dir, episode_id)
            collect_rgb_dir = os.path.join(collect_ep_dir, "rgb")
            if not os.path.isdir(collect_rgb_dir):
                skipped_no_frames += 1
                continue

            # Create output video directory with symlinks to frames
            video_name = f"{scene_name}_{episode_id}"
            video_dir = os.path.join(video_root, video_name)
            rgb_link_dir = os.path.join(video_dir, "rgb")

            if not os.path.isdir(rgb_link_dir):
                os.makedirs(video_dir, exist_ok=True)
                try:
                    os.symlink(collect_rgb_dir, rgb_link_dir)
                except FileExistsError:
                    pass
                except OSError:
                    # symlink failed, try copying
                    shutil.copytree(collect_rgb_dir, rgb_link_dir)

            # Create annotation entry
            # StreamVLN expects actions list; -1 = initial step, then 0=STOP,1=FWD,2=LEFT,3=RIGHT
            # StreamVLN internally adds -1 prefix if missing based on generate_annotations.py
            all_annotations.append({
                "id": global_id,
                "video": os.path.join(args.level, "train", video_name),
                "instructions": [instruction],
                "actions": gt_action,
            })
            global_id += 1
            used_this_scene += 1

            if args.max_episodes > 0 and used_this_scene >= args.max_episodes:
                break

        total_used += used_this_scene

    # Write annotations.json
    anno_path = os.path.join(args.output_dir, "annotations.json")
    with open(anno_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)

    print(f"DPed-{args.level.upper()} train:")
    print(f"  Total episodes processed: {total_eps}")
    print(f"  Used: {total_used}")
    print(f"  Skipped (actions < {args.min_actions}): {skipped_short}")
    print(f"  Skipped (no RGB frames): {skipped_no_frames}")
    print(f"  Output: {args.output_dir}")
    print(f"  annotations.json entries: {len(all_annotations)}")
    print(f"  Estimated batch time with 4xA800: ~{(total_used * 0.5 / 3600):.1f}h (streamvln_train.py)")


if __name__ == "__main__":
    main()
