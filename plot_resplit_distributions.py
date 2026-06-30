#!/usr/bin/env python3
"""Plot action-length and instruction-length distributions for DPed_pro resplit.

Generates a single, publication-style figure that overlays all 4 splits in the same
figure for both Histogram and KDE views.

Outputs are written under:
  <DATA_ROOT>/distribution_plots/

Instruction length is measured as whitespace-token count.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = "/share/home/u19666033/dhj/DPed_pro/dped_pro_resplit"
OUT_DIR = os.path.join(DATA_ROOT, "distribution_plots")
SPLITS = ["train", "val_seen", "val_unseen", "test_unseen"]

COLORS = {
    "train": "#4C78A8",
    "val_seen": "#F58518",
    "val_unseen": "#54A24B",
    "test_unseen": "#E45756",
}

LABELS = {
    "train": "train",
    "val_seen": "val_seen",
    "val_unseen": "val_unseen",
    "test_unseen": "test_unseen",
}


def _instruction_text(ep: dict) -> str:
    inst = ep.get("instruction", "")
    if isinstance(inst, dict):
        for key in ("instruction_text", "text", "instruction"):
            if key in inst:
                return str(inst[key])
        return json.dumps(inst, ensure_ascii=False)
    return str(inst)


def _instruction_len(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


def load_split_arrays(split: str) -> Dict[str, np.ndarray]:
    split_dir = os.path.join(DATA_ROOT, split, "data")
    files = sorted(glob.glob(os.path.join(split_dir, "*.json")))

    action_lens: List[int] = []
    instr_lens: List[int] = []

    for path in files:
        with open(path) as fp:
            data = json.load(fp)
        episodes = data.get("episodes", []) if isinstance(data, dict) else data
        for ep in episodes:
            action_lens.append(len(ep.get("gt_action", [])))
            instr_lens.append(_instruction_len(_instruction_text(ep)))

    return {
        "action": np.asarray(action_lens, dtype=np.int32),
        "instruction": np.asarray(instr_lens, dtype=np.int32),
    }


def kde_gaussian_1d(values: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    """Simple Gaussian KDE (no scipy dependency)."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.zeros_like(grid, dtype=np.float64)

    if bandwidth is None:
        std = float(np.std(values))
        if std <= 1e-12:
            std = 1.0
        bandwidth = 1.06 * std * (n ** (-1.0 / 5.0))
        bandwidth = float(max(bandwidth, 1.0))

    # Evaluate KDE on grid in chunks to control memory
    inv = 1.0 / (bandwidth * np.sqrt(2.0 * np.pi))
    out = np.zeros_like(grid, dtype=np.float64)

    chunk = 4096
    for i in range(0, n, chunk):
        v = values[i : i + chunk][:, None]  # (m,1)
        z = (grid[None, :] - v) / bandwidth
        out += np.sum(np.exp(-0.5 * z * z), axis=0)

    out *= inv / n
    return out


def set_pretty_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.size": 11,
        }
    )


@dataclass
class MetricSpec:
    key: str
    title: str
    xlabel: str
    bins: np.ndarray
    xlim: tuple[float, float]
    kde_grid: np.ndarray
    kde_downsample: int


def make_metric_specs(all_data: Dict[str, Dict[str, np.ndarray]]) -> List[MetricSpec]:
    # Action
    action_bins = np.arange(0, 203, 4)
    action_grid = np.linspace(0, 201, 600)

    # Instruction
    all_instr = np.concatenate([all_data[s]["instruction"] for s in SPLITS])
    hi = float(np.percentile(all_instr, 99.5))
    hi = max(60.0, min(250.0, hi))
    instr_bins = np.arange(0, hi + 5, 5)
    instr_grid = np.linspace(0, hi, 600)

    return [
        MetricSpec(
            key="action",
            title="Action length distribution",
            xlabel="Action length  (len(gt_action))",
            bins=action_bins,
            xlim=(0.0, 201.0),
            kde_grid=action_grid,
            kde_downsample=6000,
        ),
        MetricSpec(
            key="instruction",
            title="Instruction length distribution",
            xlabel="Instruction length  (whitespace token count)",
            bins=instr_bins,
            xlim=(0.0, hi),
            kde_grid=instr_grid,
            kde_downsample=12000,
        ),
    ]


def draw_hist_kde_panel(ax_hist, ax_kde, all_data, spec: MetricSpec):
    # Histogram overlay
    for split in SPLITS:
        vals = all_data[split][spec.key]
        ax_hist.hist(
            vals,
            bins=spec.bins,
            density=True,
            alpha=0.22,
            color=COLORS[split],
            edgecolor="none",
            label=f"{LABELS[split]} (n={len(vals)})",
        )

    ax_hist.set_title("Histogram")
    ax_hist.set_xlabel(spec.xlabel)
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlim(*spec.xlim)

    # KDE overlay
    for split in SPLITS:
        vals = all_data[split][spec.key]
        if vals.size > spec.kde_downsample:
            # deterministic downsample for speed (preserve distribution)
            idx = np.linspace(0, vals.size - 1, spec.kde_downsample).astype(int)
            v = np.sort(vals)[idx]
        else:
            v = vals

        dens = kde_gaussian_1d(v, spec.kde_grid)
        ax_kde.plot(
            spec.kde_grid,
            dens,
            color=COLORS[split],
            linewidth=2.4,
            label=LABELS[split],
        )

    ax_kde.set_title("KDE")
    ax_kde.set_xlabel(spec.xlabel)
    ax_kde.set_ylabel("Density")
    ax_kde.set_xlim(*spec.xlim)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    set_pretty_style()

    all_data = {split: load_split_arrays(split) for split in SPLITS}
    specs = make_metric_specs(all_data)

    # One figure for both metrics: 2 rows (action/instruction) × 2 cols (hist/kde)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14.5, 9.5),
        constrained_layout=True,
    )

    for r, spec in enumerate(specs):
        axes[r, 0].set_title(f"{spec.title} — Histogram")
        axes[r, 1].set_title(f"{spec.title} — KDE")
        draw_hist_kde_panel(axes[r, 0], axes[r, 1], all_data, spec)

    # Shared legend (top center)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle("DPed_pro Resplit Fairness Check: Action & Instruction Distributions", fontsize=16, fontweight="bold")

    out_path = os.path.join(OUT_DIR, "fairness_overview_histogram_kde.png")
    fig.savefig(out_path, dpi=260)
    plt.close(fig)

    # Also write per-metric figures (still each contains all 4 splits)
    for spec in specs:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14.5, 4.9), constrained_layout=True)
        draw_hist_kde_panel(axes[0], axes[1], all_data, spec)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.10))
        fig.suptitle(f"{spec.title} across splits", fontsize=15, fontweight="bold")
        fig.savefig(os.path.join(OUT_DIR, f"{spec.key}_histogram_kde.png"), dpi=260)
        plt.close(fig)

    print(f"Saved pretty plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
