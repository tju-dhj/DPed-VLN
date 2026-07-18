#!/usr/bin/env python3
"""
Convert DPed VLN trajectory data to NaVid/Uni-NaVid multi-action SFT format.

Changes from NaVILA version:
  - NaVid action text: "forward"/"left"/"right"/"stop" (no distance/angle numbers)
  - NaVid prompt: uses NAVIGATION_IDENTIFIER for special token injection
  - NaVid multi-action format: space-separated words, not semicolons
  - Sample IDs prefixed with "NAV_" to match NaVid's navigation detection

Input:
  --data_root:       path to DPed VLN JSON.GZ episode files
  --rgb_roots:       list of RGB data root directories
  --output_dir:      where to write output JSON and media
  --action_sequence_length K  (default 4)
  --action_stride    (default 4)
  --num_video_frames (default 4)
  --max_samples      (optional, for debug)
  --video_format     (default: True, save as GIF/PNG)

Output format:
  [
    {"id": "NAV_ID_dped_v1_train_ep000001_t0008",
     "video": "videos/ep000001_t0004_t0008.gif",
     "conversations": [
       {"from": "human", "value": "<image>\\nImagine you are a robot... Your assigned task is: 'instruction'..."},
       {"from": "gpt", "value": "forward forward left stop"}
     ]}
  ]
"""

import argparse
import gzip
import glob
import json
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# NaVid action space
ACTION_MAP = {
    0: "stop",
    1: "forward",
    2: "left",
    3: "right",
}

# NaVid navigation prompt (includes NAVIGATION_IDENTIFIER for special token injection)
NAVID_PROMPT_TEMPLATE = (
    "<image>\n"
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation <image>. "
    'Your assigned task is: "{instruction}". '
    "Analyze this series of images to determine your next {k} actions. "
    "The predicted action should be one of the following: forward, left, right, or stop."
)


def find_rgb_path(episode_dir_name, rgb_roots):
    for root in rgb_roots:
        for suffix in ["", ".basis"]:
            candidate = os.path.join(root, episode_dir_name + suffix)
            if os.path.isdir(candidate):
                return candidate
    for root in rgb_roots:
        if os.path.isdir(root):
            for d in os.listdir(root):
                if d.startswith(episode_dir_name) or episode_dir_name.startswith(d.replace(".basis", "")):
                    candidate = os.path.join(root, d)
                    if os.path.isdir(candidate):
                        return candidate
    return None


def load_rgb_frames(episode_rgb_dir, step_indices):
    frames = []
    rgb_dir = os.path.join(episode_rgb_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        return frames
    for si in step_indices:
        candidates = [
            os.path.join(rgb_dir, f"{int(si)}_0.jpg"),
            os.path.join(rgb_dir, f"{int(si)}_0.png"),
            os.path.join(rgb_dir, f"{int(si):06d}_0.jpg"),
            os.path.join(rgb_dir, f"{int(si)}.jpg"),
            os.path.join(rgb_dir, f"{int(si)}.png"),
        ]
        found = None
        for c in candidates:
            if os.path.isfile(c):
                found = c
                break
        if found:
            try:
                img = Image.open(found).convert("RGB")
                frames.append((si, img))
            except Exception as e:
                print(f"  WARNING: failed to load {found}: {e}", file=sys.stderr)
    return frames


def action_to_text(action_id):
    aid = int(action_id)
    if aid in ACTION_MAP:
        return ACTION_MAP[aid]
    return "stop"


def extract_episode_name(episode_path):
    basename = os.path.basename(episode_path)
    for ext in [".json.gz", ".json", ".gz"]:
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
    return basename


def sample_video_frames(rgb_frames, num_video_frames):
    if len(rgb_frames) == 0:
        return []
    if len(rgb_frames) <= num_video_frames:
        return rgb_frames
    indices = np.linspace(0, len(rgb_frames) - 1, num_video_frames, dtype=int)
    return [rgb_frames[i] for i in indices]


def convert_dataset(
    data_root,
    rgb_roots,
    output_dir,
    max_samples,
    max_eps,
    action_sequence_length,
    action_stride,
    num_video_frames,
    include_stop,
    video_format,
    dataset_tag="v1",
):
    os.makedirs(output_dir, exist_ok=True)

    json_files = []
    for ext in ["*.json.gz", "*.json"]:
        json_files.extend(glob.glob(os.path.join(data_root, ext)))
    json_files = sorted(set(json_files))
    print(f"[convert] Found {len(json_files)} JSON files in {data_root}")

    if max_eps > 0:
        json_files = json_files[:max_eps]

    if video_format:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    else:
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

    all_samples = []
    action_counter = Counter()
    sequence_lengths = []
    total_original_episodes = 0
    skipped_empty = 0
    skipped_no_rgb = 0
    samples_with_stop = 0
    sample_idx = 0

    for jf in json_files:
        if max_samples > 0 and len(all_samples) >= max_samples:
            break

        ep_name = extract_episode_name(jf)
        print(f"[convert] Processing {ep_name} ... ", end="", flush=True)

        try:
            if jf.endswith(".gz"):
                with gzip.open(jf, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except Exception as e:
            print(f"SKIP (load error: {e})")
            continue

        episodes = data.get("episodes", [])
        print(f"{len(episodes)} episodes", end="")

        scene_rgb_dir = find_rgb_path(ep_name, rgb_roots)
        if scene_rgb_dir is None:
            print(f" SKIP (no RGB scene dir {ep_name})")
            skipped_no_rgb += len(episodes)
            continue
        print(f" [rgb: {os.path.basename(scene_rgb_dir)}]", end=" ")

        ep_count = 0
        for ep in episodes:
            if max_samples > 0 and len(all_samples) >= max_samples:
                break

            gt_actions = ep.get("gt_action", [])
            instruction = ep.get("instruction", "")
            instruction = instruction.strip().strip('"')
            episode_id = str(ep.get("episode_id", ""))

            if not isinstance(gt_actions, list) or len(gt_actions) == 0:
                skipped_empty += 1
                continue
            if not instruction:
                skipped_empty += 1
                continue

            total_original_episodes += 1
            ep_count += 1
            T = len(gt_actions)

            episode_rgb_dir = os.path.join(scene_rgb_dir, episode_id)
            if not os.path.isdir(episode_rgb_dir):
                skipped_no_rgb += 1
                continue

            action_texts = [action_to_text(a) for a in gt_actions]

            t = 0
            while t < T:
                if max_samples > 0 and len(all_samples) >= max_samples:
                    break

                end_t = min(t + action_sequence_length, T)
                seq_actions = action_texts[t:end_t]

                if not include_stop and len(seq_actions) > 0 and seq_actions[-1] == "stop":
                    seq_actions = seq_actions[:-1]
                    if len(seq_actions) == 0:
                        t += action_stride
                        continue

                for a in seq_actions:
                    action_counter[a] += 1
                if "stop" in seq_actions:
                    samples_with_stop += 1

                sequence_lengths.append(len(seq_actions))

                # Sample frames from observation window
                frame_indices = list(range(t, min(end_t + num_video_frames, T)))
                frame_indices = frame_indices[:num_video_frames]

                sample_id = f"NAV_ID_dped_{dataset_tag}_{ep_name}_ep{episode_id}_t{t:04d}_k{action_sequence_length}"

                if not video_format:
                    # Save single frame as image
                    frames = load_rgb_frames(episode_rgb_dir, [t])
                    if len(frames) == 0:
                        t += action_stride
                        continue
                    img_path = os.path.join(video_dir, f"{sample_id}.png")
                    frames[0][1].save(img_path)
                    media_ref = f"videos/{sample_id}.png"
                else:
                    # Save selected frames as GIF
                    obs_frames = load_rgb_frames(episode_rgb_dir, frame_indices)
                    if len(obs_frames) == 0:
                        t += action_stride
                        continue
                    pil_frames = [f[1] for f in obs_frames]
                    if len(pil_frames) > 1:
                        gif_path = os.path.join(video_dir, f"{sample_id}.gif")
                        pil_frames[0].save(
                            gif_path, save_all=True,
                            append_images=pil_frames[1:], duration=500, loop=0
                        )
                        media_ref = f"videos/{sample_id}.gif"
                    else:
                        img_path = os.path.join(video_dir, f"{sample_id}.png")
                        pil_frames[0].save(img_path)
                        media_ref = f"videos/{sample_id}.png"

                # Build NaVid conversation
                k = action_sequence_length
                prompt = NAVID_PROMPT_TEMPLATE.format(instruction=instruction, k=k)
                # NaVid multi-action format: space-separated words
                action_seq_text = " ".join(seq_actions)

                sample = {
                    "id": sample_id,
                    "video" if video_format else "image": media_ref,
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": action_seq_text},
                    ],
                }
                all_samples.append(sample)
                sample_idx += 1

                t += action_stride

        total_original_episodes += ep_count
        print(f"-> {ep_count} new eps, {len(all_samples)} total samples")

    output_json = os.path.join(output_dir, "navid_sft_annotations.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\n[convert] Saved {len(all_samples)} samples to {output_json}")

    # Statistics
    print(f"\n{'='*70}")
    print(f" NaVid SFT Dataset Statistics")
    print(f"{'='*70}")
    print(f" Original episodes:                {total_original_episodes}")
    print(f" Generated SFT samples:            {len(all_samples)}")
    print(f" Action sequence length (K):       {action_sequence_length}")
    print(f" Action stride:                    {action_stride}")
    print(f" Video frames per sample:          {num_video_frames}")
    print(f"")

    if sequence_lengths:
        arr_sl = np.array(sequence_lengths)
        print(f" Average actions per sample:       {arr_sl.mean():.1f}")
        print(f" Sequence length distribution:")
        for k in range(1, action_sequence_length + 2):
            count_k = int((arr_sl == k).sum())
            pct_k = count_k / len(arr_sl) * 100
            label = f"{k} actions" if k <= action_sequence_length else f"{k}+ actions"
            print(f"   {label:<15s}: {count_k:>8d} ({pct_k:5.1f}%)")
        print(f"")

    print(f" Action distribution (in sequences):")
    total_actions = sum(action_counter.values())
    for action in ["forward", "left", "right", "stop"]:
        count = action_counter.get(action, 0)
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"   {action:<20s}: {count:>8d} ({pct:5.1f}%)")
    print(f"")

    if samples_with_stop > 0 and len(all_samples) > 0:
        print(f" Samples containing STOP:           {samples_with_stop} ({samples_with_stop / len(all_samples) * 100:.1f}%)")
        print(f"")

    if all_samples:
        response_lengths = [len(s["conversations"][1]["value"].split()) for s in all_samples]
        arr_rl = np.array(response_lengths)
        print(f" Response word count:     mean={arr_rl.mean():.1f}")
        print(f"")

    print(f" Skipped (empty actions/instruction): {skipped_empty}")
    print(f" Skipped (no RGB found):              {skipped_no_rgb}")
    print(f"{'='*70}")

    return output_json


def main():
    parser = argparse.ArgumentParser(
        description="Convert DPed VLN trajectories to NaVid multi-action SFT format"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing JSON.GZ episode files")
    parser.add_argument("--rgb_roots", type=str, nargs="+", required=True,
                        help="RGB data root directories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for SFT JSON and images")
    parser.add_argument("--action_sequence_length", type=int, default=4,
                        help="Number of actions in each sequence (default: 4)")
    parser.add_argument("--action_stride", type=int, default=4,
                        help="Stride between consecutive sequences (default: 4)")
    parser.add_argument("--num_video_frames", type=int, default=4,
                        help="Number of video frames to sample per sample (default: 4)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max total samples to generate (0 = unlimited)")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="Max episodes to process from each scene (0 = unlimited)")
    parser.add_argument("--include_stop", action="store_true", default=True,
                        help="Include STOP as the final action (default: True)")
    parser.add_argument("--no-include_stop", action="store_false", dest="include_stop")
    parser.add_argument("--video_format", action="store_true", default=True,
                        help="Save as GIF video (True) or single image (False)")
    parser.add_argument("--no-video_format", action="store_false", dest="video_format")
    parser.add_argument("--dataset_tag", type=str, default="v1",
                        help="Dataset tag in sample IDs (default: v1)")

    args = parser.parse_args()

    convert_dataset(
        data_root=args.data_root,
        rgb_roots=args.rgb_roots,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_eps=args.max_episodes,
        action_sequence_length=args.action_sequence_length,
        action_stride=args.action_stride,
        num_video_frames=args.num_video_frames,
        include_stop=args.include_stop,
        video_format=args.video_format,
        dataset_tag=args.dataset_tag,
    )


if __name__ == "__main__":
    main()
