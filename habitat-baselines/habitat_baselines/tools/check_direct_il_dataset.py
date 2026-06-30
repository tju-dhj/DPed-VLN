#!/usr/bin/env python3
"""Direct IL dataset health checker.

This script validates the directory layout used by DirectFileDataset and
reports structural, decoding, and shape issues before training.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


DEFAULT_INSTRUCTION_PRIORITY = [
    "instruction_vl_level_1",
    "instruction_vl_level_2",
    "instruction_level_1",
    "instruction_level_2",
    "inst_navcomposer_v1",
    "inst_navcomposer_v2",
]


@dataclass
class EpisodeIssue:
    episode_path: str
    level: str
    message: str


@dataclass
class EpisodeReport:
    episode_path: str
    scene_name: str
    episode_id: str
    action_count: int
    rgb_count: int
    depth_count: int
    instruction_dir: str
    instruction_preview: str
    rgb_size: str
    depth_size: str
    ok: bool
    issues: List[EpisodeIssue]


@dataclass
class ScanSummary:
    root: str
    total_episodes: int
    ok_episodes: int
    bad_episodes: int
    min_steps: Optional[int]
    max_steps: Optional[int]
    avg_steps: Optional[float]
    rgb_size_set: List[str]
    depth_size_set: List[str]
    missing_instruction_episodes: int
    missing_depth_episodes: int


def iter_episode_dirs(data_root: Path) -> Iterable[Path]:
    for scene_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for episode_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            yield episode_dir


def read_json_list(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def safe_read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding).strip()
        except Exception:
            continue
    raise UnicodeDecodeError("text", b"", 0, 1, f"Unable to decode {path}")


def infer_instruction_dir(episode_dir: Path, priority: Sequence[str]) -> Optional[Path]:
    for name in priority:
        candidate = episode_dir / name / "0.txt"
        if candidate.exists():
            return candidate.parent
    return None


def inspect_image(path: Path) -> Tuple[Tuple[int, int], str]:
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        return img.size, img.mode


def inspect_episode(
    episode_dir: Path,
    instruction_priority: Sequence[str],
    expected_max_episode_length: Optional[int],
) -> EpisodeReport:
    issues: List[EpisodeIssue] = []
    action_file = episode_dir / "action" / "0.json"
    rgb_dir = episode_dir / "rgb"
    depth_dir = episode_dir / "depth"

    scene_name = episode_dir.parent.name
    episode_id = episode_dir.name

    action_count = rgb_count = depth_count = 0
    instruction_dir = ""
    instruction_preview = ""
    rgb_size = ""
    depth_size = ""

    if not action_file.exists():
        issues.append(EpisodeIssue(str(episode_dir), "error", "missing action/0.json"))
    else:
        try:
            actions = read_json_list(action_file)
            action_count = len(actions)
            if action_count == 0:
                issues.append(EpisodeIssue(str(episode_dir), "error", "action list is empty"))
        except Exception as e:
            issues.append(EpisodeIssue(str(episode_dir), "error", f"failed to read actions: {e}"))

    if not rgb_dir.exists():
        issues.append(EpisodeIssue(str(episode_dir), "error", "missing rgb directory"))
    else:
        rgb_files = sorted(rgb_dir.glob("*.jpg"))
        rgb_count = len(rgb_files)
        if rgb_count == 0:
            issues.append(EpisodeIssue(str(episode_dir), "error", "no jpg files in rgb/"))
        else:
            try:
                size, mode = inspect_image(rgb_files[0])
                rgb_size = f"{size[0]}x{size[1]} {mode}"
                if size != (256, 256):
                    issues.append(EpisodeIssue(str(episode_dir), "warn", f"rgb size is {size[0]}x{size[1]}, expected 256x256"))
            except Exception as e:
                issues.append(EpisodeIssue(str(episode_dir), "error", f"failed to inspect first rgb image: {e}"))

    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("*.png"))
        depth_count = len(depth_files)
        if depth_count == 0:
            issues.append(EpisodeIssue(str(episode_dir), "warn", "depth directory exists but contains no png files"))
        else:
            try:
                size, mode = inspect_image(depth_files[0])
                depth_size = f"{size[0]}x{size[1]} {mode}"
                if size != (256, 256):
                    issues.append(EpisodeIssue(str(episode_dir), "warn", f"depth size is {size[0]}x{size[1]}, expected 256x256"))
            except Exception as e:
                issues.append(EpisodeIssue(str(episode_dir), "error", f"failed to inspect first depth image: {e}"))
    else:
        issues.append(EpisodeIssue(str(episode_dir), "warn", "missing depth directory"))

    instruction_path = infer_instruction_dir(episode_dir, instruction_priority)
    if instruction_path is None:
        issues.append(EpisodeIssue(str(episode_dir), "warn", f"no instruction file found with priority: {list(instruction_priority)}"))
    else:
        instruction_dir = instruction_path.name
        inst_file = instruction_path / "0.txt"
        try:
            instruction_preview = safe_read_text(inst_file)[:160]
            if not instruction_preview:
                issues.append(EpisodeIssue(str(episode_dir), "warn", f"instruction file is empty: {inst_file}"))
        except Exception as e:
            issues.append(EpisodeIssue(str(episode_dir), "error", f"failed to read instruction: {e}"))

    if action_count and rgb_count and action_count != rgb_count:
        issues.append(EpisodeIssue(str(episode_dir), "warn", f"action_count({action_count}) != rgb_count({rgb_count})"))
    if action_count and depth_count and action_count != depth_count:
        issues.append(EpisodeIssue(str(episode_dir), "warn", f"action_count({action_count}) != depth_count({depth_count})"))

    if expected_max_episode_length is not None and action_count > expected_max_episode_length:
        issues.append(EpisodeIssue(str(episode_dir), "warn", f"episode length {action_count} exceeds max_episode_length {expected_max_episode_length}"))

    ok = not any(item.level == "error" for item in issues)
    return EpisodeReport(
        episode_path=str(episode_dir),
        scene_name=scene_name,
        episode_id=episode_id,
        action_count=action_count,
        rgb_count=rgb_count,
        depth_count=depth_count,
        instruction_dir=instruction_dir,
        instruction_preview=instruction_preview,
        rgb_size=rgb_size,
        depth_size=depth_size,
        ok=ok,
        issues=issues,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check DirectFileDataset data quality")
    parser.add_argument("--data-root", required=True, help="dataset root: scene_name/episode_id")
    parser.add_argument("--instruction-priority", nargs="*", default=DEFAULT_INSTRUCTION_PRIORITY)
    parser.add_argument("--max-episodes", type=int, default=-1, help="limit episodes to scan")
    parser.add_argument("--max-episode-length", type=int, default=400, help="expected truncation limit")
    parser.add_argument("--json", action="store_true", help="print JSON report")
    parser.add_argument("--show-ok", action="store_true", help="print OK episodes too")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data root does not exist: {data_root}")

    reports: List[EpisodeReport] = []
    for idx, episode_dir in enumerate(iter_episode_dirs(data_root)):
        if args.max_episodes > 0 and idx >= args.max_episodes:
            break
        reports.append(inspect_episode(episode_dir, args.instruction_priority, args.max_episode_length))

    total = len(reports)
    ok_reports = [r for r in reports if r.ok]
    bad_reports = [r for r in reports if not r.ok]

    step_counts = [r.action_count for r in reports if r.action_count > 0]
    rgb_sizes = sorted({r.rgb_size for r in reports if r.rgb_size})
    depth_sizes = sorted({r.depth_size for r in reports if r.depth_size})
    missing_instruction = sum(1 for r in reports if not r.instruction_dir)
    missing_depth = sum(1 for r in reports if not r.depth_count)

    summary = ScanSummary(
        root=str(data_root),
        total_episodes=total,
        ok_episodes=len(ok_reports),
        bad_episodes=len(bad_reports),
        min_steps=min(step_counts) if step_counts else None,
        max_steps=max(step_counts) if step_counts else None,
        avg_steps=(sum(step_counts) / len(step_counts)) if step_counts else None,
        rgb_size_set=rgb_sizes,
        depth_size_set=depth_sizes,
        missing_instruction_episodes=missing_instruction,
        missing_depth_episodes=missing_depth,
    )

    if args.json:
        print(json.dumps({
            "summary": asdict(summary),
            "reports": [asdict(r) for r in reports],
        }, ensure_ascii=False, indent=2))
        return 0 if not bad_reports else 2

    print("=== Direct IL Dataset Health Check ===")
    print(f"Root: {summary.root}")
    print(f"Episodes: {summary.total_episodes} | OK: {summary.ok_episodes} | Bad: {summary.bad_episodes}")
    print(f"Steps: min={summary.min_steps}, max={summary.max_steps}, avg={summary.avg_steps}")
    print(f"RGB sizes: {summary.rgb_size_set or ['<none>']}")
    print(f"Depth sizes: {summary.depth_size_set or ['<none>']}")
    print(f"Missing instruction episodes: {summary.missing_instruction_episodes}")
    print(f"Missing depth episodes: {summary.missing_depth_episodes}")
    print()

    if bad_reports:
        print("=== Problems ===")
        for report in bad_reports[:200]:
            print(f"- {report.episode_path}")
            for issue in report.issues:
                print(f"  [{issue.level}] {issue.message}")
        print()

    if args.show_ok:
        print("=== OK Samples ===")
        for report in ok_reports[:20]:
            print(f"- {report.episode_path} | steps={report.action_count} | rgb={report.rgb_size} | depth={report.depth_size} | inst={report.instruction_dir}")
            if report.instruction_preview:
                print(f"  preview: {report.instruction_preview[:100]}")

    return 0 if not bad_reports else 2


if __name__ == "__main__":
    raise SystemExit(main())
