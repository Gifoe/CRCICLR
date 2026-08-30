"""Render the compact R1/R2/R3 source-gate summary figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "repair_round_summary.png"


def main() -> None:
    rounds = np.arange(3)
    labels = ["R1", "R2", "R3"]
    class_fidelity = np.array([[0.3129895833, 0.3263749983], [0.1960920139, 0.2244585904], [0.1969835069, 0.2328908431]])
    coverage = np.array([[0.4036666667, 0.4095274475], [0.3369861111, 0.3599998638], [0.3403819444, 0.3675384374]])
    survival = np.array([0.4161611671, 0.3486281890, 0.3541724052])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), dpi=180)
    width = 0.34
    axes[0].bar(rounds - width / 2, class_fidelity[:, 0], width, label="OpenBMI")
    axes[0].bar(rounds + width / 2, class_fidelity[:, 1], width, label="WBCIC")
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1, label="gate: 0.90")
    axes[0].set_title("Class fidelity")
    axes[0].set_ylabel("mean fidelity")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_xticks(rounds, labels)
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].bar(rounds - width / 2, coverage[:, 0], width, label="OpenBMI")
    axes[1].bar(rounds + width / 2, coverage[:, 1], width, label="WBCIC")
    axes[1].plot(rounds, survival, "o-", color="tab:green", label="candidate survival")
    axes[1].axhline(0.50, color="black", linestyle="--", linewidth=1, label="coverage gate: 0.50")
    axes[1].set_title("Coverage and survival")
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xticks(rounds, labels)
    axes[1].legend(frameon=False, fontsize=7)

    fig.suptitle("PERSIST-EEG autonomous repair: source-only gates", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
