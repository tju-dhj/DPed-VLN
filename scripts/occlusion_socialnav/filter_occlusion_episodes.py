#!/usr/bin/env python3
"""Filter SocialNav episodes into an occlusion-focused benchmark subset.

This offline pass uses collected rollout logs when available:
  - pedestrian_in_view/0.json: hidden-to-visible emergence events
  - trajectories/0.json: first-visible distance, near-miss, time-to-collision

If collect-data logs are missing, the script can optionally fall back to a weak
waypoint-intersection heuristic and mark those episodes as unverified.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from occlusion_utils import (
    OcclusionAnalysis,
    OcclusionFilterConfig,
    analyze_collect_episode,
    analyze_waypoint_episode,
    as_jsonl,
    find_collect_episode_dir,
    iter_dataset_files,
    normalize_episode_id,
    read_json,
    scene_key_for_episode,
    strip_json_suffix,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an occlusion-focused SocialNav episode subset."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Episode JSON/JSON.GZ file or directory containing episode files.",
    )
    parser.add_argument(
        "--collect_data_dir",
        type=Path,
        default=None,
        help="Optional collect_data split directory, e.g. data/collect_data/val.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for filtered dataset files and reports.",
    )
    parser.add_argument(
        "--input_pattern",
        default="*.json.gz",
        help="Recursive glob pattern used when --input_dir is a directory.",
    )
    parser.add_argument(
        "--fallback_waypoints",
        action="store_true",
        help="Use a weak waypoint-intersection heuristic when collect-data logs are absent.",
    )
    parser.add_argument(
        "--copy_all",
        action="store_true",
        help="Copy all episodes and add occlusion labels; otherwise keep candidates only.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Analyze and write reports without writing filtered dataset files.",
    )
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--limit_episodes", type=int, default=None)

    parser.add_argument("--min_hidden_steps", type=int, default=5)
    parser.add_argument("--min_visible_steps", type=int, default=2)
    parser.add_argument("--lookback_steps", type=int, default=20)
    parser.add_argument("--post_emergence_steps", type=int, default=30)
    parser.add_argument("--max_first_visible_distance", type=float, default=2.5)
    parser.add_argument("--near_miss_distance", type=float, default=1.0)
    parser.add_argument("--max_ttc_seconds", type=float, default=2.0)
    parser.add_argument("--step_seconds", type=float, default=0.25)
    parser.add_argument("--min_score", type=float, default=1.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OcclusionFilterConfig:
    return OcclusionFilterConfig(
        min_hidden_steps=args.min_hidden_steps,
        min_visible_steps=args.min_visible_steps,
        lookback_steps=args.lookback_steps,
        post_emergence_steps=args.post_emergence_steps,
        max_first_visible_distance=args.max_first_visible_distance,
        near_miss_distance=args.near_miss_distance,
        max_ttc_seconds=args.max_ttc_seconds,
        step_seconds=args.step_seconds,
        min_score=args.min_score,
    )


def labels_for_episode(
    dataset_file: Path,
    episode: Dict[str, Any],
    collect_data_dir: Optional[Path],
    config: OcclusionFilterConfig,
    fallback_waypoints: bool,
) -> tuple[OcclusionAnalysis, Optional[Path], str, str]:
    scene_key = scene_key_for_episode(dataset_file, episode)
    episode_id = normalize_episode_id(episode.get("episode_id"))
    collect_ep_dir = find_collect_episode_dir(collect_data_dir, scene_key, episode_id)

    if collect_ep_dir is not None:
        analysis = analyze_collect_episode(collect_ep_dir, config)
    elif fallback_waypoints:
        analysis = analyze_waypoint_episode(episode, config)
    else:
        analysis = OcclusionAnalysis(False, 0.0, "missing_collect_data")

    return analysis, collect_ep_dir, scene_key, episode_id


def annotate_episode(
    episode: Dict[str, Any],
    analysis: OcclusionAnalysis,
    *,
    scene_key: str,
    collect_ep_dir: Optional[Path],
) -> Dict[str, Any]:
    annotated = copy.deepcopy(episode)
    info = annotated.setdefault("info", {})
    if not isinstance(info, dict):
        info = {}
        annotated["info"] = info

    info["occlusion_socialnav"] = {
        "is_candidate": analysis.is_candidate,
        "score": analysis.score,
        "reason": analysis.reason,
        "scene_key": scene_key,
        "collect_episode_dir": None if collect_ep_dir is None else str(collect_ep_dir),
        **analysis.labels,
    }
    return annotated


def process_dataset_file(
    dataset_file: Path,
    output_dir: Path,
    collect_data_dir: Optional[Path],
    config: OcclusionFilterConfig,
    *,
    fallback_waypoints: bool,
    copy_all: bool,
    dry_run: bool,
    limit_episodes: Optional[int],
) -> Dict[str, Any]:
    data = read_json(dataset_file)
    episodes = data.get("episodes", []) if isinstance(data, dict) else []
    if not isinstance(episodes, list):
        episodes = []

    kept: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    risk_levels: Counter[str] = Counter()

    for index, episode in enumerate(episodes):
        if limit_episodes is not None and index >= limit_episodes:
            break
        if not isinstance(episode, dict):
            continue

        analysis, collect_ep_dir, scene_key, episode_id = labels_for_episode(
            dataset_file,
            episode,
            collect_data_dir,
            config,
            fallback_waypoints,
        )
        reasons[analysis.reason] += 1
        if "risk_level" in analysis.labels:
            risk_levels[str(analysis.labels["risk_level"])] += 1

        record = {
            "dataset_file": str(dataset_file),
            "scene_key": scene_key,
            "episode_id": episode_id,
            "is_candidate": analysis.is_candidate,
            "score": analysis.score,
            "reason": analysis.reason,
            "labels": analysis.labels,
            "collect_episode_dir": None if collect_ep_dir is None else str(collect_ep_dir),
        }
        records.append(record)

        if analysis.is_candidate or copy_all:
            kept.append(
                annotate_episode(
                    episode,
                    analysis,
                    scene_key=scene_key,
                    collect_ep_dir=collect_ep_dir,
                )
            )

    output_file = output_dir / dataset_file.name
    if kept and not dry_run:
        new_data = copy.deepcopy(data)
        new_data["episodes"] = kept
        write_json(output_file, new_data)

    return {
        "dataset_file": str(dataset_file),
        "output_file": str(output_file) if kept and not dry_run else None,
        "total_episodes": len(episodes) if limit_episodes is None else min(len(episodes), limit_episodes),
        "kept_episodes": len(kept),
        "candidate_episodes": sum(1 for r in records if r["is_candidate"]),
        "reasons": dict(reasons),
        "risk_levels": dict(risk_levels),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    config = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = iter_dataset_files(args.input_dir, args.input_pattern)
    if args.limit_files is not None:
        dataset_files = dataset_files[: args.limit_files]

    all_records: List[Dict[str, Any]] = []
    file_summaries: List[Dict[str, Any]] = []
    aggregate_reasons: Counter[str] = Counter()
    aggregate_risk_levels: Counter[str] = Counter()
    total_episodes = 0
    total_kept = 0
    total_candidates = 0

    print(f"Input: {args.input_dir}")
    print(f"Collect data: {args.collect_data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Dataset files: {len(dataset_files)}")

    for dataset_file in dataset_files:
        summary = process_dataset_file(
            dataset_file,
            args.output_dir,
            args.collect_data_dir,
            config,
            fallback_waypoints=args.fallback_waypoints,
            copy_all=args.copy_all,
            dry_run=args.dry_run,
            limit_episodes=args.limit_episodes,
        )
        records = summary.pop("records")
        all_records.extend(records)
        file_summaries.append(summary)
        aggregate_reasons.update(summary["reasons"])
        aggregate_risk_levels.update(summary["risk_levels"])
        total_episodes += summary["total_episodes"]
        total_kept += summary["kept_episodes"]
        total_candidates += summary["candidate_episodes"]

        scene_name = strip_json_suffix(dataset_file)
        print(
            f"{scene_name}: candidates={summary['candidate_episodes']} "
            f"kept={summary['kept_episodes']} total={summary['total_episodes']}"
        )

    summary = {
        "input_dir": str(args.input_dir),
        "collect_data_dir": None if args.collect_data_dir is None else str(args.collect_data_dir),
        "output_dir": str(args.output_dir),
        "dry_run": args.dry_run,
        "copy_all": args.copy_all,
        "fallback_waypoints": args.fallback_waypoints,
        "config": config.__dict__,
        "num_dataset_files": len(dataset_files),
        "total_episodes": total_episodes,
        "candidate_episodes": total_candidates,
        "kept_episodes": total_kept,
        "reasons": dict(aggregate_reasons),
        "risk_levels": dict(aggregate_risk_levels),
        "files": file_summaries,
    }

    summary_path = args.output_dir / "occlusion_summary.json"
    records_path = args.output_dir / "occlusion_candidates.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(as_jsonl(record) + "\n")

    print("\nDone")
    print(f"Summary: {summary_path}")
    print(f"Candidates: {records_path}")
    print(f"Candidate episodes: {total_candidates}/{total_episodes}")


if __name__ == "__main__":
    main()
