#!/usr/bin/env python3
"""Utilities for building occlusion-focused SocialNav subsets.

The functions in this file intentionally avoid Habitat imports so the first
filtering pass can run on login nodes or local machines with only Python.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Vec2 = Tuple[float, float]


def read_json(path: Path) -> Any:
    """Read a JSON or JSON.GZ file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, *, indent: Optional[int] = 2) -> None:
    """Write a JSON or JSON.GZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def strip_json_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def scene_key_from_scene_id(scene_id: str) -> str:
    """Extract the HM3D/MP3D scene key from a Habitat scene id."""
    scene_path = Path(scene_id)
    name = scene_path.name
    if name.endswith(".basis.glb"):
        return name[: -len(".basis.glb")]
    if name.endswith(".glb"):
        return name[: -len(".glb")]
    if name.endswith(".basis"):
        return name[: -len(".basis")]
    return scene_path.stem or scene_id


def scene_key_for_episode(dataset_file: Path, episode: Dict[str, Any]) -> str:
    scene_id = episode.get("scene_id")
    if isinstance(scene_id, str) and scene_id:
        return scene_key_from_scene_id(scene_id)
    return strip_json_suffix(dataset_file)


def normalize_episode_id(episode_id: Any) -> str:
    return str(episode_id)


def candidate_collect_episode_dirs(
    collect_data_dir: Path, scene_key: str, episode_id: str
) -> Iterable[Path]:
    """Yield likely collect-data episode directories for a scene/episode pair."""
    scene_variants = [
        scene_key,
        f"{scene_key}.basis",
        scene_key.replace(".basis", ""),
    ]
    seen: set[Path] = set()
    for scene_name in scene_variants:
        scene_dir = collect_data_dir / scene_name
        ep_dir = scene_dir / episode_id
        if ep_dir not in seen:
            seen.add(ep_dir)
            yield ep_dir


def find_collect_episode_dir(
    collect_data_dir: Optional[Path], scene_key: str, episode_id: str
) -> Optional[Path]:
    if collect_data_dir is None:
        return None
    for ep_dir in candidate_collect_episode_dirs(collect_data_dir, scene_key, episode_id):
        if ep_dir.exists():
            return ep_dir
    return None


def load_optional_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return default


def flatten_human_counts(raw_counts: Sequence[Any]) -> List[int]:
    counts: List[int] = []
    for item in raw_counts:
        if isinstance(item, list):
            counts.append(int(item[0]) if item else 0)
        elif item is None:
            counts.append(0)
        else:
            counts.append(int(item))
    return counts


def load_visibility_counts(ep_dir: Path) -> List[int]:
    raw = load_optional_json(ep_dir / "pedestrian_in_view" / "0.json", [])
    if not isinstance(raw, list):
        return []
    return flatten_human_counts(raw)


def load_actions(ep_dir: Path) -> List[int]:
    raw = load_optional_json(ep_dir / "action" / "0.json", [])
    return [int(x) for x in raw] if isinstance(raw, list) else []


def load_trajectories(ep_dir: Path) -> List[Dict[str, Any]]:
    raw = load_optional_json(ep_dir / "trajectories" / "0.json", [])
    return raw if isinstance(raw, list) else []


def xz(position: Sequence[float]) -> Vec2:
    if len(position) >= 3:
        return float(position[0]), float(position[2])
    if len(position) >= 2:
        return float(position[0]), float(position[1])
    return 0.0, 0.0


def euclidean(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_pedestrian_distance(step: Dict[str, Any]) -> Optional[float]:
    robot = step.get("robot") or {}
    robot_position = robot.get("position")
    pedestrians = step.get("pedestrians") or []
    if robot_position is None or not pedestrians:
        return None
    robot_xz = xz(robot_position)
    distances: List[float] = []
    for ped in pedestrians:
        pos = ped.get("position") if isinstance(ped, dict) else None
        if pos is not None:
            distances.append(euclidean(robot_xz, xz(pos)))
    return min(distances) if distances else None


def min_nearest_distance(
    trajectories: Sequence[Dict[str, Any]], start: int, end: int
) -> Optional[float]:
    distances: List[float] = []
    for step in trajectories[max(start, 0) : max(end, 0)]:
        distance = nearest_pedestrian_distance(step)
        if distance is not None:
            distances.append(distance)
    return min(distances) if distances else None


def estimate_ttc_steps(
    trajectories: Sequence[Dict[str, Any]],
    start: int,
    near_miss_distance: float,
    horizon_steps: int,
) -> Optional[int]:
    """Return steps until the nearest pedestrian distance crosses near_miss_distance."""
    for offset, step in enumerate(trajectories[start : start + horizon_steps + 1]):
        distance = nearest_pedestrian_distance(step)
        if distance is not None and distance <= near_miss_distance:
            return offset
    return None


def find_emergence_events(
    visibility_counts: Sequence[int],
    *,
    min_hidden_steps: int,
    min_visible_steps: int,
    lookback_steps: int,
) -> List[int]:
    """Find timesteps where pedestrians appear after a hidden interval."""
    if not visibility_counts:
        return []

    events: List[int] = []
    n = len(visibility_counts)
    for t, count in enumerate(visibility_counts):
        if count <= 0:
            continue
        prev_start = max(0, t - lookback_steps)
        hidden_window = visibility_counts[prev_start:t]
        if len(hidden_window) < min_hidden_steps:
            continue
        hidden_run = 0
        for prev_count in reversed(hidden_window):
            if prev_count == 0:
                hidden_run += 1
            else:
                break
        if hidden_run < min_hidden_steps:
            continue

        visible_window = visibility_counts[t : min(n, t + min_visible_steps)]
        if len(visible_window) < min_visible_steps:
            continue
        if all(v > 0 for v in visible_window):
            events.append(t)
    return events


def risk_level(
    *,
    first_visible_distance: Optional[float],
    ttc_steps: Optional[int],
    step_seconds: float,
) -> str:
    if first_visible_distance is not None and first_visible_distance <= 1.5:
        return "hard"
    if ttc_steps is not None and ttc_steps * step_seconds <= 1.5:
        return "hard"
    if first_visible_distance is not None and first_visible_distance <= 2.5:
        return "medium"
    return "easy"


@dataclass
class OcclusionFilterConfig:
    min_hidden_steps: int = 5
    min_visible_steps: int = 2
    lookback_steps: int = 20
    post_emergence_steps: int = 30
    max_first_visible_distance: float = 2.5
    near_miss_distance: float = 1.0
    max_ttc_seconds: float = 2.0
    step_seconds: float = 0.25
    min_score: float = 1.0


@dataclass
class OcclusionAnalysis:
    is_candidate: bool
    score: float
    reason: str
    labels: Dict[str, Any] = field(default_factory=dict)


def analyze_collect_episode(
    ep_dir: Path, config: OcclusionFilterConfig
) -> OcclusionAnalysis:
    visibility_counts = load_visibility_counts(ep_dir)
    trajectories = load_trajectories(ep_dir)
    actions = load_actions(ep_dir)

    if not visibility_counts:
        return OcclusionAnalysis(False, 0.0, "missing_pedestrian_in_view")

    events = find_emergence_events(
        visibility_counts,
        min_hidden_steps=config.min_hidden_steps,
        min_visible_steps=config.min_visible_steps,
        lookback_steps=config.lookback_steps,
    )
    if not events:
        return OcclusionAnalysis(False, 0.0, "no_hidden_to_visible_emergence")

    best: Optional[Dict[str, Any]] = None
    for t in events:
        first_distance = None
        min_distance_after = None
        ttc_steps = None
        if trajectories:
            first_distance = nearest_pedestrian_distance(trajectories[t])
            min_distance_after = min_nearest_distance(
                trajectories, t, t + config.post_emergence_steps + 1
            )
            ttc_steps = estimate_ttc_steps(
                trajectories, t, config.near_miss_distance, config.post_emergence_steps
            )

        hidden_steps = 0
        for prev_count in reversed(visibility_counts[:t]):
            if prev_count == 0:
                hidden_steps += 1
            else:
                break

        score = 1.0
        if first_distance is not None:
            score += max(0.0, config.max_first_visible_distance - first_distance)
        if min_distance_after is not None:
            score += max(0.0, config.near_miss_distance - min_distance_after) * 2.0
        if ttc_steps is not None:
            ttc_seconds = ttc_steps * config.step_seconds
            score += max(0.0, config.max_ttc_seconds - ttc_seconds)
        score += min(hidden_steps / max(config.lookback_steps, 1), 1.0)

        labels = {
            "event_step": t,
            "hidden_steps_before": hidden_steps,
            "first_visible_distance": first_distance,
            "min_distance_after_emergence": min_distance_after,
            "ttc_steps": ttc_steps,
            "ttc_seconds": None if ttc_steps is None else ttc_steps * config.step_seconds,
            "risk_level": risk_level(
                first_visible_distance=first_distance,
                ttc_steps=ttc_steps,
                step_seconds=config.step_seconds,
            ),
            "visibility_count_at_event": visibility_counts[t],
            "episode_steps": max(len(visibility_counts), len(actions), len(trajectories)),
            "pre_event_actions": actions[max(0, t - 5) : t] if actions else [],
            "post_event_actions": actions[t : t + 5] if actions else [],
        }

        if best is None or score > best["score"]:
            best = {"score": score, "labels": labels}

    assert best is not None
    labels = best["labels"]
    first_distance = labels.get("first_visible_distance")
    min_distance_after = labels.get("min_distance_after_emergence")
    ttc_seconds = labels.get("ttc_seconds")

    passes_distance = (
        first_distance is not None and first_distance <= config.max_first_visible_distance
    )
    passes_near_miss = (
        min_distance_after is not None and min_distance_after <= config.near_miss_distance
    )
    passes_ttc = ttc_seconds is not None and ttc_seconds <= config.max_ttc_seconds
    passes_score = best["score"] >= config.min_score

    is_candidate = passes_score and (passes_distance or passes_near_miss or passes_ttc)
    reason = "candidate" if is_candidate else "emergence_not_risky_enough"
    return OcclusionAnalysis(is_candidate, float(best["score"]), reason, labels)


def analyze_waypoint_episode(
    episode: Dict[str, Any], config: OcclusionFilterConfig
) -> OcclusionAnalysis:
    """Fallback heuristic when no collect-data logs are available.

    It labels episodes whose robot start-goal segment passes close to any human
    waypoint. This is not an occlusion proof; it only keeps potentially useful
    episodes for later Habitat visibility verification.
    """
    start = episode.get("start_position")
    goals = episode.get("goals") or []
    goal_position = goals[0].get("position") if goals and isinstance(goals[0], dict) else None
    info = episode.get("info") or {}
    if start is None or goal_position is None or not isinstance(info, dict):
        return OcclusionAnalysis(False, 0.0, "missing_waypoint_fields")

    start_xz = xz(start)
    goal_xz = xz(goal_position)
    human_points: List[Vec2] = []
    for key, value in info.items():
        if (
            isinstance(key, str)
            and key.startswith("human_")
            and "_waypoint_" in key
            and key.endswith("_position")
            and isinstance(value, list)
        ):
            human_points.append(xz(value))

    if not human_points:
        return OcclusionAnalysis(False, 0.0, "missing_human_waypoints")

    closest = min(point_to_segment_distance(p, start_xz, goal_xz) for p in human_points)
    score = max(0.0, config.max_first_visible_distance - closest)
    is_candidate = score >= config.min_score
    labels = {
        "event_step": None,
        "hidden_steps_before": None,
        "first_visible_distance": None,
        "min_distance_after_emergence": closest,
        "ttc_steps": None,
        "ttc_seconds": None,
        "risk_level": "unverified",
        "fallback": "waypoint_path_intersection",
        "closest_human_waypoint_to_robot_path": closest,
    }
    return OcclusionAnalysis(
        is_candidate,
        float(score),
        "waypoint_candidate" if is_candidate else "waypoints_not_close_to_robot_path",
        labels,
    )


def point_to_segment_distance(point: Vec2, start: Vec2, end: Vec2) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom == 0:
        return euclidean(point, start)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    projection = (sx + t * dx, sy + t * dy)
    return euclidean(point, projection)


def iter_dataset_files(input_dir: Path, pattern: str) -> List[Path]:
    if input_dir.is_file():
        return [input_dir]
    return sorted(input_dir.rglob(pattern))


def as_jsonl(record: Dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)
