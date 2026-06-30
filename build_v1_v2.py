"""Build v1 / v2 datasets from dped_pro_resplit by replacing each episode's
``instruction`` field with the content of ``instruction_vl_level_1/0.txt``
(v1) or ``instruction_vl_level_2/0.txt`` (v2).

Conventions:
- Each ``.json.gz`` file corresponds to one scene whose name is the file's
  stem (e.g. ``1EiJpeRNEs1.json.gz`` ↔ scene ``1EiJpeRNEs1``). Episodes whose
  own ``scene_id`` does not match the file's scene are dropped (mismatched).
- Episodes whose collect_data folder is missing or whose level_1 / level_2
  instruction file is missing are dropped.
- The cleaned episode set is identical across v1 and v2 (only the
  ``instruction`` field differs).

After cleaning:
- val_seen, val_unseen, test_unseen must each contain exactly 1000 episodes.
- If val_unseen has fewer than 1000 after dropping bad episodes, we top it
  up from clean train episodes (deterministic, by sorted file/episode order).
"""

import gzip
import json
import os
import shutil
from pathlib import Path

REPO = Path("/share/home/u19666033/dhj/DPed_pro")
SRC = REPO / "dped_pro_resplit"
COLLECT_TRAIN = REPO / "data/collect_data/train"
COLLECT_VAL = REPO / "data/collect_data/val"
SUBSETS = ["train", "val_seen", "val_unseen", "test_unseen"]

INSTR_FILENAME = "0.txt"
INSTR_SUBDIRS = {
    "v1": "instruction_vl_level_1",
    "v2": "instruction_vl_level_2",
}


def build_scene_root_map() -> dict[str, Path]:
    """Map ``scene_name.basis`` -> collect_data root containing it."""
    scene_root: dict[str, Path] = {}
    for s in os.listdir(COLLECT_TRAIN):
        scene_root[s] = COLLECT_TRAIN
    for s in os.listdir(COLLECT_VAL):
        scene_root[s] = COLLECT_VAL
    return scene_root


def episode_is_clean(scene_root_map: dict[str, Path], fname: str, ep: dict) -> bool:
    """Strict rule: filename stem must match episode scene name; both
    level_1 and level_2 instruction files must exist."""
    scene_name_from_file = fname.replace(".json.gz", "")
    scene_id = ep["scene_id"]
    ep_scene_name = scene_id.split("/")[-1].replace(".basis.glb", "")
    if ep_scene_name != scene_name_from_file:
        return False
    scene_folder = f"{scene_name_from_file}.basis"
    root = scene_root_map.get(scene_folder)
    if root is None:
        return False
    ep_dir = root / scene_folder / ep["episode_id"]
    if not ep_dir.is_dir():
        return False
    l1 = ep_dir / INSTR_SUBDIRS["v1"] / INSTR_FILENAME
    l2 = ep_dir / INSTR_SUBDIRS["v2"] / INSTR_FILENAME
    return l1.is_file() and l2.is_file()


def read_instruction(scene_root_map: dict[str, Path], fname: str, ep: dict, level: str) -> str:
    scene_name_from_file = fname.replace(".json.gz", "")
    scene_folder = f"{scene_name_from_file}.basis"
    root = scene_root_map[scene_folder]
    instr_path = root / scene_folder / ep["episode_id"] / INSTR_SUBDIRS[level] / INSTR_FILENAME
    return instr_path.read_text(encoding="utf-8").rstrip("\n")


def clean_subset(subset: str, scene_root_map: dict[str, Path], log: list[str]):
    """Return list of (fname, kept_episodes, dropped_details). Modifies nothing
    on disk."""
    src_subset = SRC / subset
    results = []
    for fname in sorted(os.listdir(src_subset)):
        if not fname.endswith(".json.gz"):
            continue
        with gzip.open(src_subset / fname, "rt") as f:
            data = json.load(f)
        kept = []
        dropped = []
        for ep in data["episodes"]:
            if episode_is_clean(scene_root_map, fname, ep):
                kept.append(ep)
            else:
                dropped.append(ep["episode_id"])
        if dropped:
            log.append(f"[{subset}/{fname}] dropped {len(dropped)} episode(s): {dropped}")
        results.append((fname, data, kept))
    return results


def total_episodes(results) -> int:
    return sum(len(kept) for _, _, kept in results)


def pick_topup_episodes(train_results, deficit: int):
    """Pick ``deficit`` episodes deterministically from train results in sorted
    file/episode order. Returns list of (fname, ep) tuples."""
    picked = []
    for fname, _, kept in train_results:
        for ep in kept:
            picked.append((fname, ep))
            if len(picked) == deficit:
                return picked
            # pick at most 1 episode per file
            break
        if len(picked) == deficit:
            break
    return picked


def assign_topup(topup_eps, target_results):
    """Group top-up episodes by source filename. For each group, either append
    to the existing file with the same name in target_results, or create a
    new entry that uses the source filename. Returns a new list of results."""
    by_fname: dict[str, list] = {}
    for fname, ep in topup_eps:
        by_fname.setdefault(fname, []).append(ep)
    new_results = list(target_results)
    for fname, eps in by_fname.items():
        existing = next((i for i, r in enumerate(new_results) if r[0] == fname), None)
        if existing is not None:
            f, data, kept = new_results[existing]
            new_kept = list(kept) + eps
            new_results[existing] = (f, data, new_kept)
        else:
            # Build a fresh file entry mirroring the structure of the source.
            src_path = SRC / "train" / fname
            with gzip.open(src_path, "rt") as f:
                src_data = json.load(f)
            new_data = dict(src_data)
            new_data["episodes"] = eps
            new_results.append((fname, new_data, eps))
    return new_results


def write_subset(level: str, subset: str, results, scene_root_map: dict[str, Path]):
    out_subset = SRC / level / subset
    out_subset.mkdir(parents=True, exist_ok=True)
    for fname, data, kept in results:
        # Replace instruction field for each kept episode.
        new_eps = []
        for ep in kept:
            new_ep = dict(ep)
            new_ep["instruction"] = read_instruction(scene_root_map, fname, ep, level)
            new_eps.append(new_ep)
        new_data = dict(data)
        new_data["episodes"] = new_eps
        with gzip.open(out_subset / fname, "wt") as f:
            json.dump(new_data, f)


def main():
    scene_root_map = build_scene_root_map()
    log: list[str] = []

    # 1) Clean every subset
    cleaned = {sub: clean_subset(sub, scene_root_map, log) for sub in SUBSETS}
    counts_clean = {sub: total_episodes(cleaned[sub]) for sub in SUBSETS}
    print(f"After cleaning (before top-up): {counts_clean}")

    # 2) Top-up val_seen, val_unseen, test_unseen from train if needed.
    TARGET = 1000
    final = {sub: cleaned[sub] for sub in SUBSETS}
    for sub in ("val_seen", "val_unseen", "test_unseen"):
        deficit = TARGET - total_episodes(final[sub])
        if deficit <= 0:
            continue
        topup = pick_topup_episodes(cleaned["train"], deficit)
        log.append(f"[{sub}] top-up {len(topup)} episode(s) from train: {[(f, e['episode_id']) for f, e in topup]}")
        # Remove the top-upped episodes from train
        topup_ids = {(f, e["episode_id"]) for f, e in topup}
        new_train = []
        for fname, data, kept in cleaned["train"]:
            new_kept = [ep for ep in kept if (fname, ep["episode_id"]) not in topup_ids]
            new_train.append((fname, data, new_kept))
        cleaned["train"] = new_train
        final["train"] = new_train
        final[sub] = assign_topup(topup, final[sub])

    # 3) Sanity check
    for sub in SUBSETS:
        n = total_episodes(final[sub])
        log.append(f"[{sub}] final episode count = {n}")
        if sub in ("val_seen", "val_unseen", "test_unseen"):
            assert n == TARGET, f"{sub} has {n} episodes, expected {TARGET}"
    print(f"Final counts: {{sub: total_episodes(final[sub]) for sub in SUBSETS}}")
    final_counts = {sub: total_episodes(final[sub]) for sub in SUBSETS}
    print(f"Final counts: {final_counts}")

    # 4) Clear old v1/v2 outputs and write new
    for level in ("v1", "v2"):
        for sub in SUBSETS:
            out_subset = SRC / level / sub
            if out_subset.exists():
                shutil.rmtree(out_subset)
            write_subset(level, sub, final[sub], scene_root_map)
        print(f"Wrote {level}/{{train,val_seen,val_unseen,test_unseen}}")

    # 5) Save log
    log_path = REPO / f"build_v1_v2_log_{os.getpid()}.txt"
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Log written to: {log_path}")
    print(f"Total log entries: {len(log)}")


if __name__ == "__main__":
    main()