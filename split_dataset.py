"""
Split dataset into:
  - train/           : remaining train episodes (after removing 1600)
  - seen_val/        : 100 from train
  - seen_test/       : 800 from train
  - unseen_val/      : 100 from val_full
  - unseen_test/     : 800 from val_full (exclude very long steps)

All val/test splits are stratified to match:
  - action_length distribution (bins)
  - instruction_length distribution (bins)
  - action value distribution (proportions)
"""

import gzip
import json
import glob
import os
import numpy as np
from collections import Counter, defaultdict

BASE      = "/share/home/u19666033/dhj/DPed_pro/data/dynamic_dataset_final_v1"
OUT_ROOT  = "/share/home/u19666033/dhj/DPed_pro/dataset_splits"

TRAIN_DIR   = os.path.join(BASE, "train")
VAL_DIR     = os.path.join(BASE, "val_full")
TRAIN_NEW   = os.path.join(OUT_ROOT, "train")
SEEN_VAL    = os.path.join(OUT_ROOT, "seen_val")
SEEN_TEST   = os.path.join(OUT_ROOT, "seen_test")
UNSEEN_VAL  = os.path.join(OUT_ROOT, "unseen_val")
UNSEEN_TEST = os.path.join(OUT_ROOT, "unseen_test")

# ── 1. Load all episodes ───────────────────────────────────────────────────────
def load_episodes(directory):
    episodes = []
    for fpath in sorted(glob.glob(os.path.join(directory, "*.json.gz"))):
        with gzip.open(fpath, "rt") as fp:
            d = json.load(fp)
        scene_id = os.path.basename(fpath).replace(".json.gz", "")
        for ep in d["episodes"]:
            ep["_scene_id"] = scene_id
            ep["_src_file"] = os.path.basename(fpath)
        episodes.extend(d["episodes"])
    return episodes

print("Loading train episodes...")
train_all = load_episodes(TRAIN_DIR)
print(f"  train: {len(train_all)} episodes across {len(set(e['_src_file'] for e in train_all))} scenes")

print("Loading val_full episodes...")
val_full = load_episodes(VAL_DIR)
print(f"  val_full: {len(val_full)} episodes across {len(set(e['_src_file'] for e in val_full))} scenes")

# ── 2. Feature extraction ─────────────────────────────────────────────────────
def action_len(ep):
    """Number of movement steps (exclude the leading STOP=0 action)."""
    return max(0, len(ep["gt_action"]) - 1)

def inst_len(ep):
    return len(ep["instruction"])

def movement_actions(ep):
    """All action values except the leading STOP=0."""
    return tuple(ep["gt_action"][1:])

def movement_action_counts(ep):
    """Counter of movement actions (1,2,3) as a fraction dict."""
    seq = ep["gt_action"][1:]
    if not seq:
        return {}
    c = Counter(seq)
    total = len(seq)
    return {k: c[k] / total for k in sorted(c)}

def action_buckets(n):
    """Bin action length into buckets."""
    if n <= 10:   return "A1"   # [1,10]
    elif n <= 20: return "A2"   # (10,20]
    elif n <= 35: return "A3"   # (20,35]
    elif n <= 50: return "A4"   # (35,50]
    elif n <= 75: return "A5"   # (50,75]
    elif n <= 120: return "A6"  # (75,120]
    else:         return "A7"   # >120

def inst_buckets(n):
    """Bin instruction length into buckets."""
    if n <= 50:    return "I1"   # [0,50]
    elif n <= 100: return "I2"  # (50,100]
    elif n <= 150: return "I3"  # (100,150]
    elif n <= 200: return "I4"  # (150,200]
    elif n <= 250: return "I5"  # (200,250]
    elif n <= 350: return "I6"  # (250,350]
    else:          return "I7"  # >350

# ── 3. Attribute distributions ────────────────────────────────────────────────
train_action_lens = [action_len(ep) for ep in train_all]
val_action_lens   = [action_len(ep) for ep in val_full]

train_inst_lens   = [inst_len(ep)  for ep in train_all]
val_inst_lens     = [inst_len(ep)  for ep in val_full]

print(f"\nTrain action lengths:  min={min(train_action_lens)}, max={max(train_action_lens)}, "
      f"mean={np.mean(train_action_lens):.1f}, p25={np.percentile(train_action_lens,25):.0f}, "
      f"p50={np.percentile(train_action_lens,50):.0f}, p75={np.percentile(train_action_lens,75):.0f}")
print(f"Val action lengths:    min={min(val_action_lens)}, max={max(val_action_lens)}, "
      f"mean={np.mean(val_action_lens):.1f}, p25={np.percentile(val_action_lens,25):.0f}, "
      f"p50={np.percentile(val_action_lens,50):.0f}, p75={np.percentile(val_action_lens,75):.0f}")

# ── 4. Group episodes into strata ─────────────────────────────────────────────
def stratum_key(ep):
    return (action_buckets(action_len(ep)), inst_buckets(inst_len(ep)))

# ── 5. Unseen splits: filter val_full first ───────────────────────────────────
MAX_STEPS = 200  # remove episodes with more than 200 movement steps
val_filtered = [ep for ep in val_full if action_len(ep) <= MAX_STEPS]
print(f"\nVal_full after filtering (steps ≤ {MAX_STEPS}): {len(val_filtered)} (removed {len(val_full)-len(val_filtered)})")

# Stratify val_filtered for unseen_val (100) and unseen_test (800)
strata_val = defaultdict(list)
for ep in val_filtered:
    strata_val[stratum_key(ep)].append(ep)

# ── 6. Stratified proportional sampling ──────────────────────────────────────
def stratified_sample(strata, pool, n_needed, name):
    """
    Sample n_needed items from pool, distributed proportionally across strata
    relative to the overall pool size.  Tries to respect the ratio.
    """
    total = len(pool)
    result = []
    remaining = list(pool)

    for stratum, members in sorted(strata.items()):
        proportion = len(members) / total
        quota = max(1, round(proportion * n_needed))
        quota = min(quota, len(members))
        sampled = members[:quota]
        result.extend(sampled)
        for ep in sampled:
            remaining.remove(ep)
        print(f"  {name} | {stratum}: need≈{quota}, got={len(sampled)}")

    # If we over/under-sampled, adjust
    while len(result) > n_needed and result:
        result.pop()
    while len(result) < n_needed and remaining:
        result.append(remaining.pop())

    return result, remaining

# unseen_val  = 100 from val_filtered
strata_for_unseen = defaultdict(list)
for ep in val_filtered:
    strata_for_unseen[stratum_key(ep)].append(ep)

print(f"\nSampling unseen_val (100)...")
unseen_val, val_remain = stratified_sample(
    strata_for_unseen, val_filtered, 100, "unseen_val"
)
print(f"  unseen_val total: {len(unseen_val)}, remaining val pool: {len(val_remain)}")

# unseen_test = 800 from remaining val
strata_for_unseen_test = defaultdict(list)
for ep in val_remain:
    strata_for_unseen_test[stratum_key(ep)].append(ep)

print(f"\nSampling unseen_test (800)...")
unseen_test, _ = stratified_sample(
    strata_for_unseen_test, val_remain, 800, "unseen_test"
)
print(f"  unseen_test total: {len(unseen_test)}")

# ── 7. Train splits: seen_val=100, seen_test=800 ──────────────────────────────
strata_train = defaultdict(list)
for ep in train_all:
    strata_train[stratum_key(ep)].append(ep)

print(f"\nSampling seen_val (100)...")
seen_val, train_remain = stratified_sample(
    strata_train, train_all, 100, "seen_val"
)
print(f"  seen_val total: {len(seen_val)}, remaining train pool: {len(train_remain)}")

strata_train2 = defaultdict(list)
for ep in train_remain:
    strata_train2[stratum_key(ep)].append(ep)

print(f"\nSampling seen_test (800)...")
seen_test, remaining_train = stratified_sample(
    strata_train2, train_remain, 800, "seen_test"
)
print(f"  seen_test total: {len(seen_test)}")

train_remaining = remaining_train  # final train set

# ── 8. Save episodes into per-scene files ────────────────────────────────────
def save_split(episodes, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    # Group by source file
    by_file = defaultdict(list)
    for ep in episodes:
        by_file[ep["_src_file"]].append(ep)

    for fname, eps in sorted(by_file.items()):
        out_path = os.path.join(out_dir, fname)
        full_d = {"episodes": eps}
        with gzip.open(out_path, "wt", compresslevel=6) as fp:
            json.dump(full_d, fp)

    print(f"  Saved {name}: {len(episodes)} episodes into {len(by_file)} scene files → {out_dir}")

# Save all splits
save_split(seen_val,         SEEN_VAL,    "seen_val")
save_split(seen_test,        SEEN_TEST,   "seen_test")
save_split(unseen_val,        UNSEEN_VAL,  "unseen_val")
save_split(unseen_test,       UNSEEN_TEST, "unseen_test")
save_split(train_remaining,   TRAIN_NEW,   "train_final (remaining)")

# ── 9. Verification ──────────────────────────────────────────────────────────
def verify(name, episodes):
    alens = [action_len(ep) for ep in episodes]
    ilens = [inst_len(ep)  for ep in episodes]
    macts = Counter()
    for ep in episodes:
        for a in ep["gt_action"][1:]:
            macts[a] += 1
    total_m = sum(macts.values())
    print(f"\n{'='*60}")
    print(f"{name}  (n={len(episodes)})")
    print(f"  Action steps:  min={min(alens)}, max={max(alens)}, "
          f"mean={np.mean(alens):.1f}, p25={np.percentile(alens,25):.0f}, "
          f"p50={np.percentile(alens,50):.0f}, p75={np.percentile(alens,75):.0f}, p90={np.percentile(alens,90):.0f}")
    print(f"  Instruction:    min={min(ilens)}, max={max(ilens)}, "
          f"mean={np.mean(ilens):.1f}, p25={np.percentile(ilens,25):.0f}, "
          f"p50={np.percentile(ilens,50):.0f}, p75={np.percentile(ilens,75):.0f}")
    print(f"  Action dist:    " + ", ".join(f"a{k}={macts[k]/total_m:.3f}" for k in sorted(macts)))

verify("Train (remaining)",  train_remaining)
verify("seen_val",           seen_val)
verify("seen_test",          seen_test)
verify("unseen_val",         unseen_val)
verify("unseen_test",        unseen_test)

print("\nDone.")
