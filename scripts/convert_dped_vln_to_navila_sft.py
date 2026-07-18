#!/usr/bin/env python3
"""
Convert DPed VLN trajectory data to NaVILA/LLaVA multi-action sequence SFT format.

Changes from one-step SFT:
  - Generates multi-action sequences: obs_t -> a_t; a_{t+1}; ...; a_{t+K-1}
  - Supports --action_sequence_length, --action_stride, --num_video_frames
  - Output samples with multiple actions joined by "; "

Input:
  --data_root:  path to DPed VLN JSON.GZ episode files
  --rgb_roots:  list of RGB data root directories
  --output_dir: where to write output JSON
  --action_sequence_length K (default 4)
  --action_stride (default 4)
  --max_samples (optional)
  --num_video_frames (default 4)
  --include_stop (default true)

Output format:
  [
    {
      "id": "dped_train_ep000001_t0008_k4",
      "video": "videos/ep000001_t0004_t0008.mp4",
      "conversations": [
        {"from": "human", "value": "<video>\nInstruction: xxx\nPredict the next several navigation actions."},
        {"from": "gpt", "value": "move forward 25 cm; turn left 15 degrees; stop"}
      ]
    }
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

ACTION_MAP = {
    0: "stop",
    1: "move forward 25 cm",
    2: "turn left 15 degrees",
    3: "turn right 15 degrees",
}

ACTION_FROM_TEXT = {
    "stop": 0,
    "move forward 25 cm": 1,
    "turn left 15 degrees": 2,
    "turn right 15 degrees": 3,
    "move backward 25 cm": 4,
    "wait": 5,
}

SEQUENCE_PROMPT_TEMPLATE = """<video>
Instruction: {instruction}
Predict the next several navigation actions."""


def find_rgb_path(episode_dir_name, rgb_roots):
    """Find the RGB data directory for an episode given the scene/directory name."""
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
    """Load specific RGB frames from an episode's rgb directory."""
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
    """Convert discrete action ID to language action text."""
    aid = int(action_id)
    if aid in ACTION_MAP:
        return ACTION_MAP[aid]
    return "stop"


def extract_episode_name(episode_path):
    """Extract scene name from episode JSON path."""
    basename = os.path.basename(episode_path)
    for ext in [".json.gz", ".json", ".gz"]:
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
    return basename


def sample_video_frames(rgb_frames, num_video_frames):
    """Sample N frames evenly from the available frames list (list of (step_idx, PIL.Image))."""
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
):
    """Convert the entire dataset to multi-action sequence SFT format."""
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
    instruction_lengths = []
    frame_counts_list = []
    total_original_steps = 0
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
            total_original_steps += T

            episode_rgb_dir = os.path.join(scene_rgb_dir, episode_id)
            if not os.path.isdir(episode_rgb_dir):
                skipped_no_rgb += 1
                continue

            action_texts = [action_to_text(a) for a in gt_actions]

            # Build multi-action sequence samples
            t = 0
            while t < T:
                if max_samples > 0 and len(all_samples) >= max_samples:
                    break

                # Determine the action sequence window
                end_t = min(t + action_sequence_length, T)
                seq_actions = action_texts[t:end_t]

                # If include_stop is False, trim trailing stop
                if not include_stop and len(seq_actions) > 0 and seq_actions[-1] == "stop":
                    seq_actions = seq_actions[:-1]
                    if len(seq_actions) == 0:
                        t += action_stride
                        continue

                # Count actions in this sequence
                for a in seq_actions:
                    action_counter[a] += 1
                if "stop" in seq_actions:
                    samples_with_stop += 1

                sequence_lengths.append(len(seq_actions))
                instruction_lengths.append(len(instruction.split()))

                # Sample video frames from observation window [t, end_t)
                frame_indices = list(range(t, min(end_t + num_video_frames, T)))
                frame_indices = frame_indices[:num_video_frames]
                frame_counts_list.append(len(frame_indices))

                sample_id = f"dped_{ep_name}_ep{episode_id}_t{t:04d}_k{action_sequence_length}"

                if not video_format:
                    # Save the last observation frame as image
                    frames = load_rgb_frames(episode_rgb_dir, [t])
                    if len(frames) == 0:
                        t += action_stride
                        continue
                    img_path = os.path.join(image_dir, f"{sample_id}.png")
                    frames[0][1].save(img_path)
                    media_ref = f"images/{sample_id}.png"
                else:
                    # For video: sample frames across the observation window
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
                        img_path = os.path.join(image_dir, f"{sample_id}.png")
                        pil_frames[0].save(img_path)
                        media_ref = f"images/{sample_id}.png"

                # Build conversation with multi-action response
                prompt = SEQUENCE_PROMPT_TEMPLATE.format(instruction=instruction)
                action_seq_text = "; ".join(seq_actions)

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
        print(f"-> {ep_count} new eps, {len(all_samples)} total sequence samples")

    # Save JSON
    output_json = os.path.join(output_dir, "navilla_sft_annotations.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"\n[convert] Saved {len(all_samples)} samples to {output_json}")

    # ===================== Statistics =====================
    print(f"\n{'='*70}")
    print(f" Multi-Action Sequence SFT Dataset Statistics")
    print(f"{'='*70}")
    print(f" Original episodes:                {total_original_episodes}")
    print(f" Original steps/actions:           {total_original_steps}")
    print(f" Generated SFT sequence samples:   {len(all_samples)}")
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
    for action in ["move forward 25 cm", "turn left 15 degrees", "turn right 15 degrees", "stop", "wait", "move backward 25 cm"]:
        count = action_counter.get(action, 0)
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"   {action:<30s}: {count:>8d} ({pct:5.1f}%)")
    print(f"")

    if samples_with_stop > 0 and len(all_samples) > 0:
        print(f" Samples containing STOP:           {samples_with_stop} ({samples_with_stop / len(all_samples) * 100:.1f}%)")
        print(f"")

    if instruction_lengths:
        arr_il = np.array(instruction_lengths)
        print(f" Instruction word count:  mean={arr_il.mean():.1f}, median={np.median(arr_il):.0f}, p90={np.percentile(arr_il, 90):.0f}, p95={np.percentile(arr_il, 95):.0f}")
        print(f"")

    # Token length estimate (rough: ~2 tokens per word for LLaMA tokenizer)
    if all_samples:
        response_lengths = [len(s["conversations"][1]["value"].split()) for s in all_samples]
        arr_rl = np.array(response_lengths)
        print(f" Response word count:     mean={arr_rl.mean():.1f}, p90={np.percentile(arr_rl, 90):.0f}, p95={np.percentile(arr_rl, 95):.0f}")
        print(f" (Estimated tokens ~2x:  mean={arr_rl.mean()*2:.0f}, p90={np.percentile(arr_rl, 90)*2:.0f}, p95={np.percentile(arr_rl, 95)*2:.0f})")
        print(f"")

    if frame_counts_list:
        arr_fc = np.array(frame_counts_list)
        print(f" Frame count per sample:  mean={arr_fc.mean():.1f}, p90={np.percentile(arr_fc, 90):.0f}, p95={np.percentile(arr_fc, 95):.0f}")
        print(f"")

    print(f" Skipped (empty actions/instruction): {skipped_empty}")
    print(f" Skipped (no RGB found):              {skipped_no_rgb}")
    print(f"{'='*70}")

    return output_json


def main():
    parser = argparse.ArgumentParser(
        description="Convert DPed VLN trajectories to NaVILA multi-action sequence SFT format"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing JSON.GZ episode files")
    parser.add_argument("--rgb_roots", type=str, nargs="+", required=True,
                        help="RGB data root directories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for SFT JSON and images")
    parser.add_argument("--action_sequence_length", type=int, default=4,
                        help="Number of action steps to predict per sequence (K)")
    parser.add_argument("--action_stride", type=int, default=4,
                        help="Stride between sequence start positions")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Maximum SFT samples to generate (-1 = all)")
    parser.add_argument("--max_eps", type=int, default=-1,
                        help="Maximum episode files to process (-1 = all)")
    parser.add_argument("--num_video_frames", type=int, default=4,
                        help="Number of video frames per observation window")
    parser.add_argument("--include_stop", action="store_true", default=True,
                        help="Include STOP actions in sequences")
    parser.add_argument("--no_include_stop", dest="include_stop", action="store_false",
                        help="Exclude trailing STOP from sequences")
    parser.add_argument("--video_format", action="store_true",
                        help="Output video clips instead of single images")
    parser.add_argument("--level", type=str, default="v1",
                        help="Dataset level (v1 or v2)")

    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"ERROR: data_root not found: {args.data_root}")
        sys.exit(1)

    convert_dataset(
        data_root=args.data_root,
        rgb_roots=args.rgb_roots,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_eps=args.max_eps,
        action_sequence_length=args.action_sequence_length,
        action_stride=args.action_stride,
        num_video_frames=args.num_video_frames,
        include_stop=args.include_stop,
        video_format=args.video_format,
    )


if __name__ == "__main__":
    main()
