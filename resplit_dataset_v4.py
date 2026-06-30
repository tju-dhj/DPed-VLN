"""
Rebuild all dataset splits with the new counts:
  seen_val    = 200  (from train, steps ≤ 200)
  seen_test   = 600  (from train, steps ≤ 200)
  unseen_val  = 200  (from val_full, steps ≤ 200)
  unseen_test = 600  (from val_full, steps ≤ 200)
  train       = remaining train episodes (steps ≤ 200, NO overlap with seen splits)

All splits use the SAME bucket boundaries computed from training data,
ensuring consistent action-length and instruction-length distributions.
Output goes to /share/home/u19666033/dhj/DPed_pro/dped_pro.
"""

import gzip, json, glob, os, numpy as np
from collections import Counter, defaultdict

BASE       = "/share/home/u19666033/dhj/DPed_pro/data/dynamic_dataset_final_v1"
OUT        = "/share/home/u19666033/dhj/DPed_pro/dped_pro"
SRC_TRAIN  = os.path.join(BASE, "train")
SRC_VAL    = os.path.join(BASE, "val_full")

SEEN_VAL_N    = 200
SEEN_TEST_N   = 600
UNSEEN_VAL_N  = 200
UNSEEN_TEST_N = 600
MAX_STEPS     = 200

# ── Feature helpers ────────────────────────────────────────────────────────────
def action_len(ep):
    return max(0, len(ep["gt_action"]) - 1)

def inst_len(ep):
    return len(ep["instruction"])

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

# ── Apply step filter (≤ 200) ─────────────────────────────────────────────────
def apply_filter(episodes):
    before = len(episodes)
    eps = [ep for ep in episodes if action_len(ep) <= MAX_STEPS]
    return eps, before - len(eps)

train_filt, tr_removed = apply_filter(train_all)
val_filt,   vl_removed = apply_filter(val_all)
print(f"\nAfter filtering (steps ≤ {MAX_STEPS}):")
print(f"  train:   {len(train_filt)} eps (removed {tr_removed})")
print(f"  val_full:{len(val_filt)} eps (removed {vl_removed})")

# ── Compute unified percentile-based bucket boundaries ─────────────────────────
all_action_lens = [action_len(ep) for ep in train_filt]
all_inst_lens   = [inst_len(ep)  for ep in train_filt]

ACTION_PCTS    = [10, 25, 40, 55, 70, 85]
ACTION_BOUNDS  = [0] + [int(np.percentile(all_action_lens, p)) for p in ACTION_PCTS] + [MAX_STEPS + 1]

INST_PCTS      = [20, 40, 60, 80]
INST_BOUNDS    = [0] + [int(np.percentile(all_inst_lens, p)) for p in INST_PCTS] + [100000]

def action_bucket(n):
    for i in range(len(ACTION_BOUNDS) - 1):
        if ACTION_BOUNDS[i] <= n < ACTION_BOUNDS[i + 1]:
            return f"A{i+1}"
    return f"A{len(ACTION_BOUNDS)-1}"

def inst_bucket(n):
    for i in range(len(INST_BOUNDS) - 1):
        if INST_BOUNDS[i] <= n < INST_BOUNDS[i + 1]:
            return f"I{i+1}"
    return f"I{len(INST_BOUNDS)-1}"

def stratum_key(ep):
    return (action_bucket(action_len(ep)), inst_bucket(inst_len(ep)))

print(f"\nUnified bucket boundaries (from train_filt):")
print(f"  Action bounds: {ACTION_BOUNDS}")
print(f"  Inst bounds:   {INST_BOUNDS}")

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
    remaining = {k: list(v) for k, v in strata.items()}

    for key in sorted(remaining.keys()):
        quota = round(len(strata[key]) / total * n_needed)
        quota = max(1, quota) if strata[key] else 0
        taken = remaining[key][:quota]
        selected.extend(taken)
        remaining[key] = remaining[key][quota:]

    while len(selected) > n_needed:
        selected.pop()
    while len(selected) < n_needed:
        for key in sorted(remaining.keys()):
            if remaining[key]:
                selected.append(remaining[key].pop(0))
                if len(selected) == n_needed:
                    break

    rest = []
    for key in remaining:
        rest.extend(remaining[key])
    return selected, rest

# ── Build unseen splits from val_filt ─────────────────────────────────────────
# Step 1: sample unseen_test=600 first, leave room for unseen_val=200
# Step 2: sample unseen_val=200 from remaining val pool
print(f"\nSampling unseen_test ({UNSEEN_TEST_N}) from val_filt ({len(val_filt)})...")
strata_val0 = build_strata(val_filt)
unseen_test, val_after_test = stratified_sample(val_filt, strata_val0, UNSEEN_TEST_N, "unseen_test")
print(f"  unseen_test: {len(unseen_test)}, remaining pool: {len(val_after_test)}")

print(f"\nSampling unseen_val ({UNSEEN_VAL_N}) from remaining val ({len(val_after_test)})...")
strata_val1 = build_strata(val_after_test)
unseen_val, _ = stratified_sample(val_after_test, strata_val1, UNSEEN_VAL_N, "unseen_val")
print(f"  unseen_val:  {len(unseen_val)}")

# ── Build seen splits from train_filt ─────────────────────────────────────────
# Step 1: sample seen_test=600 first
# Step 2: sample seen_val=200 from remaining train pool
print(f"\nSampling seen_test ({SEEN_TEST_N}) from train_filt ({len(train_filt)})...")
strata_train0 = build_strata(train_filt)
seen_test, train_after_test = stratified_sample(train_filt, strata_train0, SEEN_TEST_N, "seen_test")
print(f"  seen_test:   {len(seen_test)}, remaining pool: {len(train_after_test)}")

print(f"\nSampling seen_val ({SEEN_VAL_N}) from remaining train ({len(train_after_test)})...")
strata_train1 = build_strata(train_after_test)
seen_val, train_final = stratified_sample(train_after_test, strata_train1, SEEN_VAL_N, "seen_val")
print(f"  seen_val:    {len(seen_val)}")

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
# Mirror the existing dped_pro directory structure:
#   seen/seen_val, seen/seen_test, unseen/unseen_val, unseen/unseen_test, train
sv_f = save_split(seen_val,    os.path.join(OUT, "seen", "seen_val"))
st_f = save_split(seen_test,   os.path.join(OUT, "seen", "seen_test"))
uv_f = save_split(unseen_val,  os.path.join(OUT, "unseen", "unseen_val"))
ut_f = save_split(unseen_test, os.path.join(OUT, "unseen", "unseen_test"))
tr_f = save_split(train_final, os.path.join(OUT, "train"))
print(f"  seen/seen_val:    {len(seen_val)} eps ({sv_f} files)")
print(f"  seen/seen_test:   {len(seen_test)} eps ({st_f} files)")
print(f"  unseen/unseen_val:  {len(unseen_val)} eps ({uv_f} files)")
print(f"  unseen/unseen_test: {len(unseen_test)} eps ({ut_f} files)")
print(f"  train:             {len(train_final)} eps ({tr_f} files)")

# ── Verification ──────────────────────────────────────────────────────────────
def verify(name, episodes):
    alens = [action_len(ep) for ep in episodes]
    ilens = [inst_len(ep)  for ep in episodes]
    macts = Counter()
    for ep in episodes:
        for a in ep["gt_action"][1:]:
            macts[a] += 1
    total_m = sum(macts.values()) or 1
    print(f"\n{'='*60}")
    print(f"{name}  (n={len(episodes)})")
    print(f"  Action steps:  min={min(alens)}, max={max(alens)}, "
          f"mean={np.mean(alens):.1f}, "
          f"p25={np.percentile(alens,25):.0f}, p50={np.percentile(alens,50):.0f}, "
          f"p75={np.percentile(alens,75):.0f}, p90={np.percentile(alens,90):.0f}")
    print(f"  Instruction:   min={min(ilens)}, max={max(ilens)}, "
          f"mean={np.mean(ilens):.1f}, "
          f"p50={np.percentile(ilens,50):.0f}, p75={np.percentile(ilens,75):.0f}")
    print(f"  Action dist:   " + ", ".join(f"a{k}={macts[k]/total_m:.3f}" for k in sorted(macts)))

verify("seen_val",    seen_val)
verify("seen_test",   seen_test)
verify("unseen_val",  unseen_val)
verify("unseen_test", unseen_test)
verify("train",       train_final)

# ── Bucket distribution comparison ─────────────────────────────────────────────
def bucket_pct(name, episodes):
    ab = Counter(action_bucket(action_len(ep)) for ep in episodes)
    total = sum(ab.values())
    return {k: round(100 * ab[k] / total, 1) for k in sorted(ab)}

print(f"\n{'='*60}")
print("Action bucket % across all splits:")
header = f"{'Bucket':>8}"
for s in ["seen_val", "seen_test", "unseen_val", "unseen_test"]:
    header += f" {s:>14}"
print(header)
print("-" * 70)
all_buckets = sorted(set().union(*[
    set(bucket_pct(n, e).keys())
    for n, e in [("sv", seen_val), ("st", seen_test), ("uv", unseen_val), ("ut", unseen_test)]
]))
for b in all_buckets:
    row = f"{b:>8}"
    for n, eps in [("seen_val", seen_val), ("seen_test", seen_test),
                   ("unseen_val", unseen_val), ("unseen_test", unseen_test)]:
        row += f" {bucket_pct(n, eps).get(b, 0.0):>13.1f}%"
    print(row)

print(f"\n{'='*60}")
print("Instruction bucket % across all splits:")
def ibucket_pct(name, episodes):
    ib = Counter(inst_bucket(inst_len(ep)) for ep in episodes)
    total = sum(ib.values())
    return {k: round(100 * ib[k] / total, 1) for k in sorted(ib)}

header2 = f"{'Bucket':>8}"
for s in ["seen_val", "seen_test", "unseen_val", "unseen_test"]:
    header2 += f" {s:>14}"
print(header2)
print("-" * 70)
all_ibuckets = sorted(set().union(*[
    set(ibucket_pct(n, e).keys())
    for n, e in [("sv", seen_val), ("st", seen_test), ("uv", unseen_val), ("ut", unseen_test)]
]))
for b in all_ibuckets:
    row = f"{b:>8}"
    for n, eps in [("seen_val", seen_val), ("seen_test", seen_test),
                   ("unseen_val", unseen_val), ("unseen_test", unseen_test)]:
        row += f" {ibucket_pct(n, eps).get(b, 0.0):>13.1f}%"
    print(row)

# ── Check train has no seen episodes ─────────────────────────────────────────
seen_ids = {(ep["_scene"], ep["episode_id"]) for ep in seen_val + seen_test}
overlap = [ep for ep in train_final
           if (ep["_scene"], ep["episode_id"]) in seen_ids]
print(f"\n{'='*60}")
if overlap:
    print(f"ERROR: train overlaps with seen splits: {len(overlap)} episodes")
else:
    print("OK: train has ZERO overlap with seen_val/seen_test")
print("Done.")
