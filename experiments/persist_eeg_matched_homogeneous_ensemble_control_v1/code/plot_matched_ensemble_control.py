"""Compact pre-specified plot for the matched ensemble control."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["EEGNet_homogeneous", "LiteBN_homogeneous", "Heterogeneous_orientation_mean"]
LABELS = ["EEGNet+EEGNet", "LiteBN+LiteBN", "EEGNet+LiteBN"]
BLOCKS = ["OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"]

def plot(path: Path, out: Path) -> None:
    data = pd.read_csv(path)
    data = data[(data.fusion_rule == "LOGIT50") & data.family.isin(ORDER)]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.2), sharey=True)
    colors = ["#5271a3", "#8b6b9c", "#c45d39"]
    for ax, block in zip(axes, BLOCKS):
        subset = data[data.block == block]
        for pos, (family, color) in enumerate(zip(ORDER, colors)):
            values = subset[subset.family == family].G_best_pp.to_numpy()
            jitter = np.linspace(-0.11, 0.11, len(values))
            ax.scatter(np.full(len(values), pos) + jitter, values, s=18, color=color, alpha=.72, edgecolor="none")
            mean = values.mean(); se = values.std(ddof=1) / max(len(values), 1) ** .5
            ax.errorbar(pos, mean, yerr=1.96*se, fmt="o", color="black", capsize=3, zorder=4)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set_title(block.replace("_", " "), fontsize=10)
        ax.set_xticks(range(3), LABELS, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Gain over best constituent (pp)")
    fig.suptitle("Matched two-model ensembles (all five folds × three seed pairs)", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot(Path(sys.argv[1]), Path(sys.argv[2]))
