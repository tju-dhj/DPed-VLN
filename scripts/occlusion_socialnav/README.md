# Occlusion-SocialNav Episode Tools

This folder contains the first offline implementation for building an
occlusion-focused SocialNav subset. The goal is to isolate episodes where a
pedestrian appears after being hidden, especially useful for doorway, corner,
T-junction, and blind-spot evaluation.

## What The Filter Detects

`filter_occlusion_episodes.py` scans episode JSON/JSON.GZ files and, when
available, matches them with `data/collect_data/<split>/<scene>.basis/<episode>/`
rollout logs.

The strongest evidence comes from:

- `pedestrian_in_view/0.json`: detects hidden-to-visible emergence events.
- `trajectories/0.json`: measures first-visible robot-human distance, near-miss,
  and time-to-collision after emergence.
- `action/0.json`: stores a short pre/post action window for later diagnosis.

Each kept episode is annotated under:

```json
{
  "info": {
    "occlusion_socialnav": {
      "is_candidate": true,
      "score": 3.1,
      "reason": "candidate",
      "event_step": 7,
      "hidden_steps_before": 7,
      "first_visible_distance": 1.8,
      "min_distance_after_emergence": 0.9,
      "ttc_seconds": 1.25,
      "risk_level": "medium"
    }
  }
}
```

## Quick Start

From `DPed_pro`:

```bash
python scripts/occlusion_socialnav/filter_occlusion_episodes.py \
  --input_dir data/datasets/pointnav/social-hm3d/val/content \
  --collect_data_dir data/collect_data/val \
  --output_dir data/occlusion_socialnav/val
```

For a small smoke test:

```bash
python scripts/occlusion_socialnav/filter_occlusion_episodes.py \
  --input_dir data/datasets/pointnav/social-hm3d/val/content \
  --collect_data_dir data/collect_data/val \
  --output_dir /tmp/occlusion_socialnav_smoke \
  --limit_files 1 \
  --limit_episodes 20
```

If collect-data logs are not available, use the weak waypoint fallback only to
produce candidates for later Habitat verification:

```bash
python scripts/occlusion_socialnav/filter_occlusion_episodes.py \
  --input_dir data/datasets/pointnav/social-hm3d/val/content \
  --output_dir data/occlusion_socialnav/val_unverified \
  --fallback_waypoints
```

## Outputs

The output directory contains:

- Filtered `*.json.gz` episode files, preserving the original file names.
- `occlusion_summary.json`: aggregate counts, thresholds, and per-file stats.
- `occlusion_candidates.jsonl`: one analysis record per scanned episode.

By default only candidate episodes are written. Add `--copy_all` to copy every
episode while annotating both positive and negative labels.

## Important Thresholds

- `--min_hidden_steps`: minimum consecutive invisible steps before emergence.
- `--min_visible_steps`: minimum visible steps after emergence.
- `--max_first_visible_distance`: first-visible distance threshold in meters.
- `--near_miss_distance`: near-miss threshold in meters.
- `--max_ttc_seconds`: risk threshold after first visibility.
- `--step_seconds`: simulator step duration used to convert TTC steps.

Recommended defaults are intentionally conservative for a first pass. Tighten
`--max_first_visible_distance` to `1.5` for hard blind-spot subsets.

## Suggested Benchmark Splits

Build three progressively stronger subsets:

- `Natural-Occlusion`: run this filter on existing collected rollouts.
- `Hard-Occlusion`: same filter with `--max_first_visible_distance 1.5`.
- `Unverified-Waypoint`: fallback candidates that should be verified with
  Habitat ray casting or visual inspection before being used in final results.

## Next Extension Point

The filter does not yet synthesize new doorway/corner/T-junction episodes. The
intended next step is a Habitat-backed constructor that:

1. extracts doorway/corner/junction anchors from the navmesh or occupancy grid;
2. samples robot start/goal and hidden human start/goal around each anchor;
3. verifies initial line-of-sight occlusion with ray casting;
4. writes normal SocialNav JSON.GZ files that this filter can re-score.

This keeps the current tool useful without requiring Habitat imports, while
leaving a clean interface for later controlled episode generation.
