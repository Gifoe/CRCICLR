from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v3_common import FIGURES, RESULTS


DISPLAY_NAMES = {
    "M0_B6_KEEP_ENSEMBLE": "M0 B6",
    "M1_ENSEMBLE_CONFIDENCE_RULE": "M1 confidence",
    "M2_ENSEMBLE_DISAGREEMENT_RULE": "M2 disagreement",
    "M3_ACTION_MOVEMENT_LOGISTIC": "M3 movement logistic",
    "M4_FULL_LEGAL_LOGISTIC": "M4 full legal logistic",
    "M5_HIST_GRADIENT_BOOSTING": "M5 HGB",
    "I006_CONDITIONAL_ACTION_LOGISTIC": "I006 conditional logistic",
    "I007_CONDITIONAL_ACTION_HGB": "I007 conditional HGB",
}


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_figures() -> None:
    results = pd.read_csv(RESULTS / "RESIDUAL_POLICY_RESULTS.csv")
    results = results[results.model_id.ne("M0_B6_KEEP_ENSEMBLE")].copy()
    results["display"] = results.model_id.map(DISPLAY_NAMES)
    results = results.sort_values("mean_subject_delta_BA_vs_B6")
    mean = 100 * results.mean_subject_delta_BA_vs_B6.to_numpy(dtype=float)
    low = 100 * results.bootstrap_CI95_L.to_numpy(dtype=float)
    high = 100 * results.bootstrap_CI95_U.to_numpy(dtype=float)
    y = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - low, high - mean]),
        fmt="o",
        color="#1f4e79",
        ecolor="#6c8ead",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.axvline(0.5, color="#b2182b", linewidth=1, linestyle=":", label="+0.5 pp criterion")
    ax.set_yticks(y, results.display)
    ax.set_xlabel("Mean subject-balanced Delta BA vs B6 (percentage points)")
    ax.set_title("No prospective residual policy improves on B6")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "POLICY_DELTA_BA_CI")

    folds = pd.read_csv(RESULTS / "FOLD_RESULTS.csv")
    folds = folds[folds.model_id.ne("M0_B6_KEEP_ENSEMBLE")].copy()
    matrix = folds.pivot(index="model_id", columns="outer_fold", values="mean_delta_BA_vs_B6")
    matrix = matrix.reindex(results.model_id)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    image = ax.imshow(100 * matrix.to_numpy(dtype=float), cmap="RdBu", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_yticks(np.arange(len(matrix)), [DISPLAY_NAMES[value] for value in matrix.index])
    ax.set_xticks(np.arange(len(matrix.columns)), [f"Fold {value}" for value in matrix.columns])
    ax.set_title("Grouped OOF fold Delta BA vs B6 (pp)")
    for row in range(len(matrix)):
        for column in range(len(matrix.columns)):
            value = 100 * float(matrix.iloc[row, column])
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Delta BA (pp)", shrink=0.8)
    _save(fig, "FOLD_DELTA_BA_HEATMAP")

    learn = pd.read_csv(RESULTS / "RESIDUAL_LEARNABILITY_CONDITIONAL.csv")
    rescue = learn[learn.target.eq("rescue")].copy()
    rescue["display"] = rescue.model_id.map(DISPLAY_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.5), sharey=True)
    for axis, action in zip(axes, ("AMPLIFY", "GEOMETRY", "ERASE")):
        subset = rescue[rescue.action_family.eq(action)].sort_values("AUROC")
        colors = ["#b2182b" if value < 0.5 else "#2166ac" for value in subset.AUROC]
        axis.barh(subset.display, subset.AUROC, color=colors)
        axis.axvline(0.5, color="black", linewidth=1, linestyle="--")
        axis.set_xlim(0.35, 0.76)
        axis.set_title(action)
        axis.set_xlabel("Conditional rescue AUROC")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_ylabel("Grouped OOF model")
    fig.suptitle("Only ERASE has moderate conditional ranking signal")
    _save(fig, "CONDITIONAL_RESCUE_LEARNABILITY")


if __name__ == "__main__":
    write_figures()
