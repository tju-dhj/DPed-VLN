#!/usr/bin/env python3
"""DPed_pro resplit with matched action-length distributions.

This script merges ALL existing dataset splits under `SRC` and re-splits into:
  - train           (all remaining episodes)
  - val_seen        (1000 episodes; scenes DO appear in train)
  - val_unseen      (1000 episodes; scenes DO NOT appear in train)
  - test_unseen     (1000 episodes; scenes DO NOT appear in train; disjoint from val_unseen)

Key constraint:
  - "unseen" is defined ONLY relative to NEW train: unseen scenes must not appear in train.

Goal:
  - Make action-length distributions (len(gt_action)) of val_seen/val_unseen/test_unseen as similar as possible
    via stratified sampling over action-length bins.

Outputs:
  OUT/
    train/data/<scene>.json
    val_seen/data/<scene>.json
    val_unseen/data/<scene>.json
    test_unseen/data/<scene>.json

Notes:
  - Episodes are identified by (scene, episode_id).
  - We try to choose an unseen-scene set whose total episode count is close to 2000
    (plus a small buffer) to avoid wasting too much data.

"""

from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
SRC = "/share/home/u19666033/dhj/DPed_pro/dped_pro"
OUT = "/share/home/u19666033/dhj/DPed_pro/dped_pro_resplit"

N_VAL_SEEN = 1000
N_VAL_UNSEEN = 1000
N_TEST_UNSEEN = 1000

# Action-length bins (same bins used for all splits)
BINS = [0, 20, 40, 60, 80, 100, 150, 202]  # upper exclusive; 201 fits
BIN_LABELS = [
    "[0-20)",
    "[20-40)",
    "[40-60)",
    "[60-80)",
    "[80-100)",
    "[100-150)",
    "[150-202)",
]

# When selecting unseen scenes, ensure enough per-bin capacity for TWO 1000-episode splits.
UNSEEN_TARGET = N_VAL_UNSEEN + N_TEST_UNSEEN
UNSEEN_BUFFER = 250  # extra slack to make bin coverage easier


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpKey:
    scene: str
    episode_id: str


def action_len(ep: dict) -> int:
    return len(ep.get("gt_action", []))


def bin_index(alen: int) -> int:
    # np.digitize returns 1..len(BINS)-1; we want 0..len-2
    idx = int(np.digitize([alen], BINS, right=False)[0] - 1)
    return max(0, min(idx, len(BINS) - 2))


def iter_json_files() -> Iterable[Tuple[str, str]]:
    """Yield (split_name, path) for all .json files under SRC."""
    patterns = [
        ("train", os.path.join(SRC, "train", "*.json")),
        ("val_seen", os.path.join(SRC, "val", "val_seen", "*.json")),
        ("val_unseen", os.path.join(SRC, "val", "val_unseen", "*.json")),
        ("test_seen", os.path.join(SRC, "test", "test_seen", "*.json")),
        ("test_unseen", os.path.join(SRC, "test", "test_unseen", "*.json")),
    ]
    for split, pat in patterns:
        for p in sorted(glob.glob(pat)):
            yield split, p


def load_all_episodes() -> Dict[str, List[dict]]:
    """Return dict scene -> list of episode dicts (merged across all original splits)."""
    by_scene: Dict[str, List[dict]] = defaultdict(list)

    # Deduplicate by (scene, episode_id)
    seen_keys: set[EpKey] = set()

    n_files = 0
    n_eps_raw = 0
    n_eps_kept = 0

    for _split, path in iter_json_files():
        n_files += 1
        scene = os.path.basename(path).replace(".json", "")
        with open(path) as fp:
            d = json.load(fp)
        episodes = d["episodes"] if isinstance(d, dict) else d
        n_eps_raw += len(episodes)

        for ep in episodes:
            key = EpKey(scene=scene, episode_id=str(ep["episode_id"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            by_scene[scene].append(ep)
            n_eps_kept += 1

    print(f"Loaded {n_eps_kept} unique episodes from {n_files} files (raw episodes={n_eps_raw}).")
    return dict(by_scene)


def compute_target_bin_counts(lens: List[int], n_target: int) -> List[int]:
    """Compute per-bin target counts (sum == n_target) using global proportions."""
    bins = [bin_index(x) for x in lens]
    counts = np.bincount(bins, minlength=len(BIN_LABELS)).astype(float)
    props = counts / counts.sum()

    raw = props * n_target
    tgt = np.floor(raw).astype(int)
    remainder = n_target - int(tgt.sum())
    if remainder > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for i in order[:remainder]:
            tgt[i] += 1

    # Ensure exact sum
    assert int(tgt.sum()) == n_target
    return tgt.tolist()


def sample_by_bins(
    pool: List[Tuple[EpKey, dict, int]],
    target_bin_counts: List[int],
    rng: random.Random,
) -> Tuple[List[Tuple[EpKey, dict, int]], List[int]]:
    """Sample from pool to match target_bin_counts as closely as possible.

    pool items: (key, ep, bin_idx)

    Returns: (selected_items, achieved_bin_counts)

    If a bin has insufficient examples, it will take all available and the deficit
    is reallocated to other bins with remaining capacity.
    """
    by_bin: Dict[int, List[Tuple[EpKey, dict, int]]] = defaultdict(list)
    for item in pool:
        by_bin[item[2]].append(item)

    for b in by_bin:
        rng.shuffle(by_bin[b])

    selected: List[Tuple[EpKey, dict, int]] = []
    achieved = [0] * len(BIN_LABELS)

    remaining_need = target_bin_counts[:]

    # First pass: take min(need, available)
    leftovers: List[Tuple[EpKey, dict, int]] = []
    for b in range(len(BIN_LABELS)):
        avail = by_bin.get(b, [])
        take = min(remaining_need[b], len(avail))
        if take > 0:
            selected.extend(avail[:take])
            achieved[b] += take
        leftovers.extend(avail[take:])
        remaining_need[b] -= take

    deficit = sum(remaining_need)
    if deficit == 0:
        return selected, achieved

    # Second pass: fill deficit from any leftovers (any bins), keeping randomness
    rng.shuffle(leftovers)
    fill = min(deficit, len(leftovers))
    selected.extend(leftovers[:fill])
    for _, _, b in leftovers[:fill]:
        achieved[b] += 1

    return selected, achieved


def write_split(ep_items: List[Tuple[EpKey, dict, int]], split_name: str) -> None:
    out_dir = os.path.join(OUT, split_name, "data")
    os.makedirs(out_dir, exist_ok=True)

    by_scene: Dict[str, List[dict]] = defaultdict(list)
    for key, ep, _b in ep_items:
        by_scene[key.scene].append(ep)

    for scene, episodes in by_scene.items():
        out_path = os.path.join(out_dir, f"{scene}.json")
        with open(out_path, "w") as fp:
            json.dump({"episodes": episodes}, fp)

    print(f"Wrote {split_name:11s}: {sum(len(v) for v in by_scene.values())} episodes in {len(by_scene)} scenes -> {out_dir}")


def split_stats(name: str, ep_items: List[Tuple[EpKey, dict, int]]) -> None:
    lens = [action_len(ep) for _k, ep, _b in ep_items]
    mean = float(np.mean(lens)) if lens else 0.0
    median = float(np.median(lens)) if lens else 0.0
    p75 = float(np.percentile(lens, 75)) if lens else 0.0
    mn = int(min(lens)) if lens else 0
    mx = int(max(lens)) if lens else 0

    bin_counts = np.bincount([bin_index(x) for x in lens], minlength=len(BIN_LABELS)).tolist() if lens else [0] * len(BIN_LABELS)
    parts = " | ".join(f"{BIN_LABELS[i]}:{bin_counts[i]}" for i in range(len(BIN_LABELS)))

    print(f"{name:11s}: n={len(lens):5d} mean={mean:5.1f} med={median:4.0f} p75={p75:4.0f} min={mn:3d} max={mx:3d}")
    print(f"  bins: {parts}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)
    os.makedirs(OUT, exist_ok=True)

    episodes_by_scene = load_all_episodes()
    all_scenes = sorted(episodes_by_scene.keys())

    # Build scene -> items list
    scene_items: Dict[str, List[Tuple[EpKey, dict, int]]] = {}
    all_items: List[Tuple[EpKey, dict, int]] = []
    for scene, eps in episodes_by_scene.items():
        items = []
        for ep in eps:
            key = EpKey(scene=scene, episode_id=str(ep["episode_id"]))
            b = bin_index(action_len(ep))
            items.append((key, ep, b))
        scene_items[scene] = items
        all_items.extend(items)

    all_lens = [action_len(ep) for _k, ep, _b in all_items]
    print(f"Total scenes: {len(all_scenes)}, total episodes: {len(all_items)}")
    print(f"Global action_len: mean={np.mean(all_lens):.1f} med={np.median(all_lens):.0f} p90={np.percentile(all_lens,90):.0f}")

    # Decide unseen scenes.
    # We need to sample TWO disjoint 1000-episode sets (val_unseen, test_unseen)
    # with the same target bin histogram. So we pick unseen scenes until, for every bin,
    # unseen_pool has at least 2 * target_bins_1000[bin] examples (plus some slack).
    shuffled_scenes = all_scenes[:]
    rng.shuffle(shuffled_scenes)

    # Target bin distribution: use GLOBAL distribution
    target_bins_1000 = compute_target_bin_counts(all_lens, 1000)

    unseen_scenes: set[str] = set()
    unseen_bin_counts = [0] * len(BIN_LABELS)

    def unseen_sufficient() -> bool:
        # Require per-bin capacity for 2 splits, but be tolerant for rare tail bin
        need = [2 * c for c in target_bins_1000]
        return all(unseen_bin_counts[i] >= need[i] for i in range(len(need)))

    for scene in shuffled_scenes:
        if unseen_sufficient() and sum(unseen_bin_counts) >= UNSEEN_TARGET + UNSEEN_BUFFER:
            break
        unseen_scenes.add(scene)
        for _k, _ep, b in scene_items[scene]:
            unseen_bin_counts[b] += 1

    # Final sanity check
    if sum(unseen_bin_counts) < UNSEEN_TARGET:
        raise RuntimeError(
            f"Not enough episodes for unseen splits: have {sum(unseen_bin_counts)}, need {UNSEEN_TARGET}"
        )

    train_scenes = set(all_scenes) - unseen_scenes
    unseen_pool = [item for s in unseen_scenes for item in scene_items[s]]
    seen_pool = [item for s in train_scenes for item in scene_items[s]]

    print(f"\nChosen unseen scenes: {len(unseen_scenes)} scenes, {len(unseen_pool)} episodes")
    print(f"Train scenes:         {len(train_scenes)} scenes, {len(seen_pool)} episodes")

    print("\nTarget bin counts for 1000 episodes:")
    for i, c in enumerate(target_bins_1000):
        print(f"  {BIN_LABELS[i]}: {c}")
    print("\nUnseen pool bin counts:")
    for i, c in enumerate(unseen_bin_counts):
        print(f"  {BIN_LABELS[i]}: {c}")

    # Sample val_unseen and test_unseen from unseen_pool (disjoint)
    rng.shuffle(unseen_pool)
    val_unseen, _achieved_val_unseen = sample_by_bins(unseen_pool, target_bins_1000, rng)
    val_unseen_keys = {k for k, _ep, _b in val_unseen}

    remaining_unseen = [it for it in unseen_pool if it[0] not in val_unseen_keys]
    test_unseen, _achieved_test_unseen = sample_by_bins(remaining_unseen, target_bins_1000, rng)

    if len(val_unseen) != N_VAL_UNSEEN or len(test_unseen) != N_TEST_UNSEEN:
        raise RuntimeError(
            f"Unseen sampling failed: val_unseen={len(val_unseen)} test_unseen={len(test_unseen)}"
        )

    # Sample val_seen from seen_pool
    rng.shuffle(seen_pool)
    val_seen, _achieved_val_seen = sample_by_bins(seen_pool, target_bins_1000, rng)
    val_seen_keys = {k for k, _ep, _b in val_seen}

    if len(val_seen) != N_VAL_SEEN:
        raise RuntimeError(f"Seen sampling failed: val_seen={len(val_seen)}")

    # Train is: all episodes from train_scenes (seen_pool) minus val_seen episodes.
    train = [it for it in seen_pool if it[0] not in val_seen_keys]

    # Stats
    print("\n=== Split stats ===")
    split_stats("val_seen", val_seen)
    split_stats("val_unseen", val_unseen)
    split_stats("test_unseen", test_unseen)
    split_stats("train", train)

    # Write
    write_split(train, "train")
    write_split(val_seen, "val_seen")
    write_split(val_unseen, "val_unseen")
    write_split(test_unseen, "test_unseen")

    # Save manifest
    manifest = {
        "seed": SEED,
        "src": SRC,
        "out": OUT,
        "sizes": {
            "val_seen": len(val_seen),
            "val_unseen": len(val_unseen),
            "test_unseen": len(test_unseen),
            "train": len(train),
        },
        "unseen_scenes": sorted(unseen_scenes),
        "bins": {"edges": BINS, "labels": BIN_LABELS, "target_1000": target_bins_1000},
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\nDone. Manifest written to {os.path.join(OUT, 'manifest.json')}")


if __name__ == "__main__":
    main()
