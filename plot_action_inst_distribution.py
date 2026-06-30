"""
Plot action length vs instruction length distributions across dataset splits.
"""
import gzip
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 120

BASE = "/share/home/u19666033/dhj/DPed_pro/dataset_splits"

SPLITS = {
    "train":      os.path.join(BASE, "train"),
    "seen_val":   os.path.join(BASE, "seen_val"),
    "seen_test":  os.path.join(BASE, "seen_test"),
    "unseen_val": os.path.join(BASE, "unseen_val"),
    "unseen_test":os.path.join(BASE, "unseen_test"),
}

def load_episodes(directory):
    episodes = []
    for fpath in sorted(glob.glob(os.path.join(directory, "*.json.gz"))):
        with gzip.open(fpath, "rt") as fp:
            d = json.load(fp)
        episodes.extend(d.get("episodes", []))
    return episodes

def action_len(ep):
    return max(0, len(ep["gt_action"]) - 1)

def inst_len(ep):
    return len(ep["instruction"])

print("Loading episodes...")
data = {}
for name, directory in SPLITS.items():
    eps = load_episodes(directory)
    al = [action_len(ep) for ep in eps]
    il = [inst_len(ep) for ep in eps]
    data[name] = {"action_lens": al, "inst_lens": il, "n": len(eps)}
    print(f"  {name}: {len(eps)} episodes")

# ── Figure 1: Histograms (side-by-side per split) ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

split_names = list(SPLITS.keys())
colors_a = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
colors_i = ["#64B5F6", "#FFB74D", "#81C784", "#E57373", "#BA68C8"]

for idx, name in enumerate(split_names):
    ax = axes[idx]
    al = data[name]["action_lens"]
    il = data[name]["inst_lens"]

    ax.hist(al, bins=40, alpha=0.6, label=f"Action length (μ={np.mean(al):.1f})", color=colors_a[idx], density=True)
    ax.hist(il, bins=40, alpha=0.6, label=f"Instruction length (μ={np.mean(il):.1f})", color=colors_i[idx], density=True)
    ax.set_title(f"{name} (n={data[name]['n']})")
    ax.set_xlabel("Length (chars / steps)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].axis("off")
fig.suptitle("Action Length vs Instruction Length per Split", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("/share/home/u19666033/dhj/DPed_pro/action_inst_distribution_split.png", bbox_inches="tight")
print("\nSaved: action_inst_distribution_split.png")

# ── Figure 2: Scatter + marginal histograms ───────────────────────────────────
fig, axes = plt.subplots(5, 5, figsize=(16, 16),
                         gridspec_kw={"width_ratios": [4, 1, 4, 1, 1],
                                       "height_ratios": [1, 4, 1, 4, 1]})

for row, name in enumerate(split_names):
    al = np.array(data[name]["action_lens"])
    il = np.array(data[name]["inst_lens"])

    # scatter
    ax_sc = axes[row, 0] if False else fig.add_subplot(5, 5, row * 5 + 1)
    ax_sc.scatter(il, al, alpha=0.3, s=5, c=colors_a[row])
    ax_sc.set_ylabel(f"{name}\nAction len")
    ax_sc.set_xlabel("Instruction length")
    ax_sc.grid(alpha=0.3)

    # marginal X (instruction)
    ax_ix = fig.add_subplot(5, 5, row * 5 + 2)
    ax_ix.hist(il, bins=30, orientation="horizontal", alpha=0.6, color=colors_i[row], density=True)
    ax_ix.set_xticks([])
    ax_ix.set_yticks([])
    ax_ix.spines["top"].set_visible(False)
    ax_ix.spines["right"].set_visible(False)
    ax_ix.spines["bottom"].set_visible(False)

    # marginal Y (action)
    ax_ia = fig.add_subplot(5, 5, row * 5 + 4)
    ax_ia.hist(al, bins=30, alpha=0.6, color=colors_a[row], density=True)
    ax_ia.set_xticks([])
    ax_ia.set_yticks([])
    ax_ia.spines["top"].set_visible(False)
    ax_ia.spines["left"].set_visible(False)
    ax_ia.spines["right"].set_visible(False)

    # stats
    ax_stat = fig.add_subplot(5, 5, row * 5 + 5)
    ax_stat.axis("off")
    stats = f"n={len(al)}\n"
    stats += f"action μ={np.mean(al):.1f}\n"
    stats += f"action σ={np.std(al):.1f}\n"
    stats += f"inst μ={np.mean(il):.1f}\n"
    stats += f"inst σ={np.std(il):.1f}\n"
    stats += f"corr={np.corrcoef(il, al)[0,1]:.3f}"
    ax_stat.text(0.1, 0.5, stats, transform=ax_stat.transAxes,
                 fontsize=8, verticalalignment="center", family="monospace")

# hide empty subplots
for i in range(5):
    axes[i, 1].axis("off")
    axes[i, 3].axis("off")
    axes[i, 4].axis("off")

fig.suptitle("Action Length vs Instruction Length — Scatter + Marginals", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("/share/home/u19666033/dhj/DPed_pro/action_inst_scatter_marginal.png", bbox_inches="tight")
print("Saved: action_inst_scatter_marginal.png")

# ── Figure 3: Box plots across splits ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

split_labels = list(SPLITS.keys())
action_data  = [data[k]["action_lens"] for k in split_labels]
inst_data    = [data[k]["inst_lens"]  for k in split_labels]

bp1 = axes[0].boxplot(action_data, labels=split_labels, patch_artist=True, notch=True)
for patch, color in zip(bp1["boxes"], colors_a):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_title("Action Length Distribution by Split")
axes[0].set_ylabel("Action Steps")
axes[0].grid(axis="y", alpha=0.3)

bp2 = axes[1].boxplot(inst_data, labels=split_labels, patch_artist=True, notch=True)
for patch, color in zip(bp2["boxes"], colors_i):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_title("Instruction Length Distribution by Split")
axes[1].set_ylabel("Character Count")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Box Plot Comparison Across Splits", fontsize=14)
plt.tight_layout()
plt.savefig("/share/home/u19666033/dhj/DPed_pro/action_inst_boxplot.png", bbox_inches="tight")
print("Saved: action_inst_boxplot.png")

# ── Figure 4: Violin plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

parts1 = axes[0].violinplot(action_data, positions=range(len(split_labels)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts1["bodies"]):
    pc.set_facecolor(colors_a[i])
    pc.set_alpha(0.7)
axes[0].set_xticks(range(len(split_labels)))
axes[0].set_xticklabels(split_labels, rotation=15)
axes[0].set_title("Action Length Violin")
axes[0].set_ylabel("Action Steps")
axes[0].grid(axis="y", alpha=0.3)

parts2 = axes[1].violinplot(inst_data, positions=range(len(split_labels)), showmeans=True, showmedians=True)
for i, pc in enumerate(parts2["bodies"]):
    pc.set_facecolor(colors_i[i])
    pc.set_alpha(0.7)
axes[1].set_xticks(range(len(split_labels)))
axes[1].set_xticklabels(split_labels, rotation=15)
axes[1].set_title("Instruction Length Violin")
axes[1].set_ylabel("Character Count")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Violin Plot Comparison Across Splits", fontsize=14)
plt.tight_layout()
plt.savefig("/share/home/u19666033/dhj/DPed_pro/action_inst_violin.png", bbox_inches="tight")
print("Saved: action_inst_violin.png")

print("\nAll plots saved to /share/home/u19666033/dhj/DPed_pro/")
