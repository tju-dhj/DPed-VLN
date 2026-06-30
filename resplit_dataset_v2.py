"""
Rebuild all splits with correct counts:
  - seen_val  = 100  (from train, steps ≤ 200)
  - seen_test = 800  (from train, steps ≤ 200)
  - unseen_val  = 100  (from val_full, steps ≤ 200)
  - unseen_test = 800  (from val_full, steps ≤ 200)
  - train      = remaining train episodes (no overlap with seen/seen splits)

All val/test share the same (coarser) stratification buckets so distributions match.
"""

import gzip, json, glob, os, numpy as np
from collections import Counter, defaultdict

BASE    = "/share/home/u19666033/dhj/DPed_pro/data/dynamic_dataset_final_v1"
OUT     = "/share/home/u19666033/dhj/DPed_pro/dataset_splits"
SRC_TRAIN = os.path.join(BASE, "train")
SRC_VAL    = os.path.join(BASE, "val_full")

# ── Feature helpers ────────────────────────────────────────────────────────────
def action_len(ep):
    return max(0, len(ep["gt_action"]) - 1)

def inst_len(ep):
    return len(ep["instruction"])

MAX_STEPS = 200

# Coarser buckets so strata are less likely to go empty
def action_bucket(n):
    if n <= 5:   return "A1"
    elif n <= 15: return "A2"
    elif n <= 30: return "A3"
    elif n <= 50: return "A4"
    elif n <= 80: return "A5"
    elif n <= 120: return "A6"
    else:          return "A7"

def inst_bucket(n):
    if n <= 80:   return "I1"
    elif n <= 150: return "I2"
    elif n <= 220: return "I3"
    elif n <= 300: return "I4"
    else:          return "I5"

def stratum_key(ep):
    return (action_bucket(action_len(ep)), inst_bucket(inst_len(ep)))

# ── Load episodes ─────────────────────────────────────────────────────────────
def load_episodes(directory):
    episodes = []
    for fpath in sorted(glob.glob(os.path.join(directory, "*.json.gz"))):
        with gzip.open(fpath, "rt") as fp:
            d = json.load(fp)
        scene = os.path.basename(fpath).replace(".json.gz", "")
        for ep in d["episodes"]:
            ep["_scene"] = scene
            ep["_file"]  = os.path.basename(fpath)
        episodes.extend(d["episodes"])
    return episodes

print("Loading data...")
train_all = load_episodes(SRC_TRAIN)
val_all   = load_episodes(SRC_VAL)
print(f"  train:   {len(train_all)} eps")
print(f"  val_full:{len(val_all)} eps")

# ── Apply step filter ─────────────────────────────────────────────────────────
def apply_filter(episodes):
    before = len(episodes)
    eps = [ep for ep in episodes if action_len(ep) <= MAX_STEPS]
    return eps, before - len(eps)

train_filt, train_removed = apply_filter(train_all)
val_filt,   val_removed   = apply_filter(val_all)
print(f"\nAfter filtering (steps ≤ {MAX_STEPS}):")
print(f"  train:   {len(train_filt)} eps (removed {train_removed})")
print(f"  val_full:{len(val_filt)} eps (removed {val_removed})")

# ── Stratified sampler ────────────────────────────────────────────────────────
def build_strata(episodes):
    s = defaultdict(list)
    for ep in episodes:
        s[stratum_key(ep)].append(ep)
    return s

def stratified_sample(episodes, strata, n_needed, name):
    """
    Proportional quota per stratum, taking from the front of each stratum list.
    After proportional fill, distributes any remainder round-robin.
    Returns (selected, remaining).
    """
    total = len(episodes)
    selected  = []
    remaining = {k: list(v) for k, v in strata.items()}  # mutable copy

    # Phase 1: proportional quotas
    for key in sorted(remaining.keys()):
        quota = round(len(strata[key]) / total * n_needed)
        quota = max(1, quota)  # at least 1 if stratum non-empty
        taken = remaining[key][:quota]
        selected.extend(taken)
        remaining[key] = remaining[key][quota:]

    # Phase 2: adjust if over/under
    while len(selected) > n_needed:
        selected.pop()
    while len(selected) < n_needed:
        for key in sorted(remaining.keys()):
            if remaining[key]:
                selected.append(remaining[key].pop(0))
                if len(selected) == n_needed:
                    break

    # Build remaining list
    rest = []
    for key in remaining:
        rest.extend(remaining[key])

    return selected, rest

# ── Sample unseen splits from val_filt ───────────────────────────────────────
print(f"\nSampling unseen_val (100) from val_filt ({len(val_filt)} available)...")
strata_val = build_strata(val_filt)
unseen_val, val_remain = stratified_sample(val_filt, strata_val, 100, "unseen_val")
print(f"  unseen_val: {len(unseen_val)}, remaining pool: {len(val_remain)}")

print(f"\nSampling unseen_test (800) from remaining val ({len(val_remain)} available)...")
strata_val2 = build_strata(val_remain)
unseen_test, _ = stratified_sample(val_remain, strata_val2, 800, "unseen_test")
print(f"  unseen_test: {len(unseen_test)}")

# ── Sample seen splits from train_filt ───────────────────────────────────────
print(f"\nSampling seen_val (100) from train_filt ({len(train_filt)} available)...")
strata_train = build_strata(train_filt)
seen_val, train_remain = stratified_sample(train_filt, strata_train, 100, "seen_val")
print(f"  seen_val: {len(seen_val)}, remaining pool: {len(train_remain)}")

print(f"\nSampling seen_test (800) from remaining train ({len(train_remain)} available)...")
strata_train2 = build_strata(train_remain)
seen_test, train_final = stratified_sample(train_remain, strata_train2, 800, "seen_test")
print(f"  seen_test: {len(seen_test)}")

# ── Save all splits ───────────────────────────────────────────────────────────
def save_split(episodes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*.json.gz")):
        os.remove(f)
    by_file = defaultdict(list)
    for ep in episodes:
        by_file[ep["_file"]].append(ep)
    for fname, eps in sorted(by_file.items()):
        with gzip.open(os.path.join(out_dir, fname), "wt", compresslevel=6) as fp:
            json.dump({"episodes": eps}, fp)
    return len(by_file)

print("\nSaving splits...")
sf = save_split(seen_val,    os.path.join(OUT, "seen_val"))
st = save_split(seen_test,   os.path.join(OUT, "seen_test"))
uf = save_split(unseen_val,  os.path.join(OUT, "unseen_val"))
ut = save_split(unseen_test, os.path.join(OUT, "unseen_test"))
tr = save_split(train_final, os.path.join(OUT, "train"))
print(f"  seen_val:    {len(seen_val)} eps ({sf} files)")
print(f"  seen_test:   {len(seen_test)} eps ({st} files)")
print(f"  unseen_val:  {len(unseen_val)} eps ({uf} files)")
print(f"  unseen_test: {len(unseen_test)} eps ({ut} files)")
print(f"  train:       {len(train_final)} eps ({tr} files)")

# ── Verification ─────────────────────────────────────────────────────────────
def verify(name, episodes, max_alen=None):
    alens = [action_len(ep) for ep in episodes]
    ilens = [inst_len(ep)  for ep in episodes]
    macts = Counter()
    for ep in episodes:
        for a in ep["gt_action"][1:]:
            macts[a] += 1
    total_m = sum(macts.values()) or 1
    if max_alen:
        alens = [x for x in alens if x <= max_alen]
    print(f"\n{'='*60}")
    print(f"{name}  (n={len(episodes)})")
    print(f"  Action steps:  min={min(alens)}, max={max(alens)}, "
          f"mean={np.mean(alens):.1f}, "
          f"p25={np.percentile(alens,25):.0f}, p50={np.percentile(alens,50):.0f}, p75={np.percentile(alens,75):.0f}, p90={np.percentile(alens,90):.0f}")
    print(f"  Instruction:   min={min(ilens)}, max={max(ilens)}, "
          f"mean={np.mean(ilens):.1f}, "
          f"p50={np.percentile(ilens,50):.0f}, p75={np.percentile(ilens,75):.0f}")
    print(f"  Action dist:   " + ", ".join(f"a{k}={macts[k]/total_m:.3f}" for k in sorted(macts)))

verify("seen_val",    seen_val,    MAX_STEPS)
verify("seen_test",   seen_test,   MAX_STEPS)
verify("unseen_val",  unseen_val,  MAX_STEPS)
verify("unseen_test", unseen_test, MAX_STEPS)
verify("train",       train_final, MAX_STEPS)

# ── Check train has no seen episodes ────────────────────────────────────────
seen_ids = {(ep["_scene"], ep["episode_id"]) for ep in seen_val + seen_test}
overlap = [(ep["_scene"], ep["episode_id"])
           for ep in train_final
           if (ep["_scene"], ep["episode_id"]) in seen_ids]
print(f"\n{'='*60}")
if overlap:
    print(f"ERROR: train overlaps with seen splits: {len(overlap)} episodes")
else:
    print("OK: train has ZERO overlap with seen_val/seen_test")
print("Done.")
