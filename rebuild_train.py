"""
Rebuild train/ by excluding all episodes in seen_val and seen_test.
Original train = 32,172 episodes
seen_val       =   100 episodes
seen_test      =   800 episodes
Expected train = 31,272 episodes
"""

import gzip, json, glob, os
from collections import defaultdict

SRC_TRAIN   = "/share/home/u19666033/dhj/DPed_pro/data/dynamic_dataset_final_v1/train"
DST_TRAIN   = "/share/home/u19666033/dhj/DPed_pro/dataset_splits/train"
SEEN_VAL    = "/share/home/u19666033/dhj/DPed_pro/dataset_splits/seen_val"
SEEN_TEST   = "/share/home/u19666033/dhj/DPed_pro/dataset_splits/seen_test"

# ── 1. Collect IDs to exclude ─────────────────────────────────────────────────
def load_ep_id_set(directory):
    s = set()
    for fpath in sorted(glob.glob(os.path.join(directory, "*.json.gz"))):
        with gzip.open(fpath, "rt") as fp:
            d = json.load(fp)
        scene = os.path.basename(fpath).replace(".json.gz", "")
        for ep in d["episodes"]:
            s.add((scene, ep["episode_id"]))
    return s

exclude = load_ep_id_set(SEEN_VAL) | load_ep_id_set(SEEN_TEST)
print(f"Excluding {len(exclude)} episodes (100 seen_val + 800 seen_test)")

# ── 2. Rebuild train ──────────────────────────────────────────────────────────
by_file = defaultdict(list)
kept = 0
excluded = 0

for fpath in sorted(glob.glob(os.path.join(SRC_TRAIN, "*.json.gz"))):
    with gzip.open(fpath, "rt") as fp:
        d = json.load(fp)

    scene = os.path.basename(fpath).replace(".json.gz", "")
    kept_eps = []
    for ep in d["episodes"]:
        key = (scene, ep["episode_id"])
        if key in exclude:
            excluded += 1
        else:
            kept_eps.append(ep)
            kept += 1

    if kept_eps:
        by_file[os.path.basename(fpath)] = kept_eps

print(f"Kept {kept} episodes, excluded {excluded} from {len(by_file)} scene files")

# ── 3. Save ───────────────────────────────────────────────────────────────────
os.makedirs(DST_TRAIN, exist_ok=True)
# Clear existing files first
for f in glob.glob(os.path.join(DST_TRAIN, "*.json.gz")):
    os.remove(f)

for fname, eps in sorted(by_file.items()):
    out_path = os.path.join(DST_TRAIN, fname)
    with gzip.open(out_path, "wt", compresslevel=6) as fp:
        json.dump({"episodes": eps}, fp)

print(f"Saved train: {kept} episodes into {len(by_file)} scene files → {DST_TRAIN}")

# ── 4. Quick verify ───────────────────────────────────────────────────────────
print("\nVerification:")
total = sum(len(json.load(gzip.open(f))["episodes"])
            for f in glob.glob(os.path.join(DST_TRAIN, "*.json.gz")))
print(f"  Actual episodes written: {total}")
print(f"  Scene files: {len(glob.glob(os.path.join(DST_TRAIN, '*.json.gz')))}")
