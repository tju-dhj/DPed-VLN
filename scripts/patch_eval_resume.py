#!/usr/bin/env python3
"""
patch_eval_resume.py - Mark episodes from crashed scenes as evaluated.

When the Habitat simulator crashes on a specific scene (e.g., C++ malloc error
during scene loading), this script patches the eval_resume JSON file to mark all
episodes using that scene as "already evaluated" (with success=0).

This allows the auto-restart wrapper to skip the problematic scene entirely
during fast-forward, preventing infinite crash loops.

Usage:
    python3 scripts/patch_eval_resume.py \
        --eval-resume /path/to/eval_resume_val_unseen.json \
        --dataset-dir /path/to/dataset/val_unseen/ \
        --scene-names 1zDbEdygBeW.basis ...

    python3 scripts/patch_eval_resume.py \
        --project-dir /path/to/project \
        --config-name DPed_vlm/navilla/zero_shot_static/v2_val_unseen.yaml
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def find_resume_files(project_dir: str) -> list:
    """Find all eval_resume JSON files in the project."""
    results = []
    for root, dirs, files in os.walk(project_dir):
        for f in files:
            if f.startswith("eval_resume_") and f.endswith(".json"):
                results.append(os.path.join(root, f))
    return results


def load_resume(resume_path: str) -> dict:
    """Load eval_resume JSON."""
    with open(resume_path, "r") as f:
        return json.load(f)


def save_resume(resume_path: str, data: dict) -> None:
    """Save eval_resume JSON (with backup)."""
    backup_path = resume_path + ".backup"
    if os.path.exists(resume_path) and not os.path.exists(backup_path):
        import shutil
        shutil.copy2(resume_path, backup_path)
        print(f"[patch] Backup saved to {backup_path}")

    with open(resume_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[patch] Saved {data.get('total_completed', 0)} episodes to {resume_path}")


def load_dataset_episodes(dataset_subdir: str, scene_names: list) -> dict:
    """
    Load episode info from dataset files.

    Returns: dict mapping (scene_id, episode_id) -> episode_info
    """
    episodes = {}
    scene_base_names = [s.replace(".basis.glb", "").replace(".basis.scn", "").replace(".basis", "")
                       for s in scene_names]

    for fname in os.listdir(dataset_subdir):
        if not fname.endswith(".json.gz"):
            continue
        scene_name = fname.replace(".json.gz", "")
        if scene_name in scene_base_names:
            fpath = os.path.join(dataset_subdir, fname)
            with gzip.open(fpath, "rt") as f:
                data = json.load(f)
            for ep in data.get("episodes", []):
                episodes[(ep["scene_id"], ep["episode_id"])] = ep
            print(f"[patch] Found {len(data.get('episodes', []))} episodes for scene {scene_name}")

    return episodes


def build_scene_path_patterns(scene_names: list, resume_data: dict) -> list:
    """
    Build patterns to match scene names against resume file keys.
    The resume file stores full paths like:
    /share/.../data/scene_datasets/hm3d/train/00573-1zDbEdygBeW/1zDbEdygBeW.basis.glb

    scene_names might be like '1zDbEdygBeW.basis' or '1zDbEdygBeW'
    """
    patterns = []
    for name in scene_names:
        # Extract just the scene hash/ID
        match = re.search(r'([A-Za-z0-9]{11})', name)
        if match:
            patterns.append(match.group(1))
        else:
            patterns.append(name.replace(".basis", "").replace(".glb", "").replace(".scn", ""))
    return patterns


def patch_resume_for_scenes(resume_path: str, scene_patterns: list) -> int:
    """
    Patch eval_resume to mark episodes from specified scenes as evaluated.

    Returns number of episodes patched.
    """
    data = load_resume(resume_path)
    stats_episodes = data.get("stats_episodes", {})
    ep_eval_count = data.get("ep_eval_count", {})

    # First, scan ep_eval_count to find matching scene/episode IDs
    # Keys in ep_eval_count: "scene_full_path|||episode_id"
    # Keys in stats_episodes: "scene_full_path|||episode_id|||eval_count"

    patched_count = 0

    for ep_key in list(ep_eval_count.keys()):
        for pattern in scene_patterns:
            if pattern in ep_key:
                parts = ep_key.split("|||")
                if len(parts) >= 2:
                    scene_id = parts[0]
                    episode_id = int(parts[1])

                    # Check if already in stats_episodes
                    existing = False
                    for stats_key in list(stats_episodes.keys()):
                        if stats_key.startswith(ep_key):
                            existing = True
                            break

                    if not existing:
                        # Add as failed episode
                        stats_key = f"{ep_key}|||1"
                        stats_episodes[stats_key] = {
                            "reward": 0.0,
                            "success": 0.0,
                            "spl": 0.0,
                            "psc": 0.0,
                            "distance_to_goal": 0.0,
                            "multi_agent_nav_reward": 0.0,
                            "human_collision": 0.0,
                            "distance_to_goal_reward": 0.0,
                            "stl": 0.0,
                            "did_multi_agents_collide": 0.0,
                            "num_steps": 0,
                        }
                        patched_count += 1
                        print(f"[patch] Marked as failed: scene={scene_id}, episode={episode_id}")

    if patched_count > 0:
        data["stats_episodes"] = stats_episodes
        data["total_completed"] = len(stats_episodes)
        save_resume(resume_path, data)
    else:
        print("[patch] No new episodes to patch (all may already be marked)")

    return patched_count


def patch_resume_from_dataset(resume_path: str, dataset_subdir: str, scene_names: list) -> int:
    """
    More robust approach: load episodes from dataset and match against resume.
    """
    data = load_resume(resume_path)

    scene_base_names = set()
    for name in scene_names:
        match = re.search(r'([A-Za-z0-9]{11})', name)
        if match:
            scene_base_names.add(match.group(1))
        else:
            clean = name.replace(".basis", "").replace(".glb", "").replace(".scn", "")
            scene_base_names.add(clean)

    patched_count = 0

    for fname in os.listdir(dataset_subdir):
        if not fname.endswith(".json.gz"):
            continue
        fscene = fname.replace(".json.gz", "")
        if fscene not in scene_base_names:
            continue

        fpath = os.path.join(dataset_subdir, fname)
        with gzip.open(fpath, "rt") as f:
            dataset = json.load(f)

        for ep in dataset.get("episodes", []):
            scene_id = ep["scene_id"]
            episode_id = ep["episode_id"]

            # Build the key as it would appear in ep_eval_count
            count_key = f"{scene_id}|||{episode_id}"

            if count_key in data.get("ep_eval_count", {}):
                continue  # Already tracked

            # Add to ep_eval_count
            if "ep_eval_count" not in data:
                data["ep_eval_count"] = {}
            data["ep_eval_count"][count_key] = 1

            # Add to stats_episodes as failed
            stats_key = f"{count_key}|||1"
            if "stats_episodes" not in data:
                data["stats_episodes"] = {}
            data["stats_episodes"][stats_key] = {
                "reward": 0.0,
                "success": 0.0,
                "spl": 0.0,
                "psc": 0.0,
                "distance_to_goal": 0.0,
                "multi_agent_nav_reward": 0.0,
                "human_collision": 0.0,
                "distance_to_goal_reward": 0.0,
                "stl": 0.0,
                "did_multi_agents_collide": 0.0,
                "num_steps": 0,
                "_skipped": True,  # Marker indicating this was auto-skipped
            }
            patched_count += 1

        print(f"[patch] Processed scene {fscene}: {len(dataset.get('episodes', []))} episodes")

    if patched_count > 0:
        data["total_completed"] = len(data["stats_episodes"])
        save_resume(resume_path, data)

    return patched_count


def main():
    parser = argparse.ArgumentParser(
        description="Patch eval_resume to skip episodes from crashing scenes"
    )
    parser.add_argument(
        "--eval-resume",
        type=str,
        help="Path to eval_resume JSON file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to dataset directory containing scene JSON.gz files",
    )
    parser.add_argument(
        "--scene-names",
        type=str,
        nargs="+",
        help="Scene names/patterns to skip (e.g., '1zDbEdygBeW.basis')",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=os.getcwd(),
        help="Project root directory (for auto-detecting resume files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be patched without actually modifying files",
    )

    args = parser.parse_args()

    if not args.scene_names:
        print("[patch] ERROR: --scene-names is required")
        sys.exit(1)

    # Find resume files
    if args.eval_resume:
        resume_files = [args.eval_resume]
    else:
        resume_files = find_resume_files(args.project_dir)

    if not resume_files:
        print("[patch] ERROR: No eval_resume files found")
        sys.exit(1)

    print(f"[patch] Found {len(resume_files)} resume file(s)")
    for rf in resume_files:
        print(f"  - {rf}")

    total_patched = 0
    for resume_path in resume_files:
        print(f"\n[patch] Processing: {resume_path}")

        if args.dataset_dir and os.path.isdir(args.dataset_dir):
            patched = patch_resume_from_dataset(
                resume_path, args.dataset_dir, args.scene_names
            )
        else:
            scene_patterns = build_scene_path_patterns(args.scene_names, {})
            patched = patch_resume_for_scenes(resume_path, scene_patterns)

        total_patched += patched

    print(f"\n[patch] Total episodes patched: {total_patched}")

    if total_patched > 0:
        print("[patch] These episodes will be skipped on next eval run (fast-forward).")
        print("[patch] They are marked with success=0 and _skipped=True.")


if __name__ == "__main__":
    main()
