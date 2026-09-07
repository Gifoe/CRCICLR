from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
WIDTH_MM = 183
COLORS = {"EEGNet": "#2671b8", "EEGConformer": "#d95f02", "Pooled": "#4d4d4d"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def export(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(RESULTS / "PER_SUBJECT_UTILITY.csv", dtype={"subject_id": str})
    policy = pd.read_csv(RESULTS / "POLICY_COMPARISON.csv")
    harm = pd.read_csv(RESULTS / "HARM_AND_COVERAGE.csv")
    corr = pd.read_csv(RESULTS / "UTILITY_TRANSFER_CORRELATION.csv")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, 83 / 25.4), constrained_layout=True)
    for ax, scope in zip(axes, ("EEGNet", "EEGConformer")):
        frame = subjects[subjects.scope == scope]
        x, y = 100 * frame.Delta_S2_BA, 100 * frame.Delta_S3_BA
        limit = max(1.0, float(np.max(np.abs(np.r_[x, y]))) * 1.12)
        rho = corr[(corr.scope == scope) & (corr.method == "spearman")].iloc[0]
        ax.axhline(0, color="0.68", lw=0.65)
        ax.axvline(0, color="0.68", lw=0.65)
        ax.plot([-limit, limit], [-limit, limit], ls="--", lw=0.65, color="0.55")
        ax.scatter(x, y, s=18, alpha=0.85, color=COLORS[scope], edgecolor="white", linewidth=0.25)
        ax.text(0.03, 0.97, f"Spearman = {rho.estimate:.2f}\n95% CI [{rho.CI95_low:.2f}, {rho.CI95_high:.2f}]",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.5)
        ax.set(title=scope, xlabel=r"S2 utility, $\Delta$ BA (pp)", ylabel=r"S3 utility, $\Delta$ BA (pp)",
               xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    export(fig, "utility_transfer_scatter")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, 82 / 25.4), constrained_layout=True)
    labels = ["Anchor", "Always Adapt", "S2-Gated Adapt"]
    x = np.arange(3)
    offsets = [-0.13, 0, 0.13]
    for offset, scope in zip(offsets, ("EEGNet", "EEGConformer", "Pooled")):
        row = policy[policy.scope == scope].iloc[0]
        means = 100 * np.array([row.anchor_policy_BA, row.always_adapt_policy_BA, row.S2_gated_policy_BA])
        lows = 100 * np.array([row.anchor_policy_BA_CI95_low, row.always_adapt_policy_BA_CI95_low, row.S2_gated_policy_BA_CI95_low])
        highs = 100 * np.array([row.anchor_policy_BA_CI95_high, row.always_adapt_policy_BA_CI95_high, row.S2_gated_policy_BA_CI95_high])
        ax.errorbar(x + offset, means, yerr=np.vstack([means - lows, highs - means]), fmt="o", ms=4,
                    capsize=2, lw=0.8, color=COLORS[scope], label=scope)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean S3 balanced accuracy (%)")
    ax.set_title("Prospective S3 policy performance (95% subject-bootstrap CI)")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    export(fig, "policy_comparison")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, 78 / 25.4), constrained_layout=True)
    scopes = ["EEGNet", "EEGConformer", "Pooled"]
    table = harm.set_index("scope").loc[scopes]
    xx = np.arange(3)
    axes[0].bar(xx - 0.18, 100 * table.harm_always, 0.36, label="Always Adapt", color="#8c8c8c")
    axes[0].bar(xx + 0.18, 100 * table.harm_certified, 0.36, label="S2-positive", color="#2ca25f", hatch="//")
    axes[0].set(xticks=xx, xticklabels=scopes, ylabel="S3 negative-transfer rate (%)", title="Future harm", ylim=(0, 65))
    axes[0].legend(frameon=False)
    axes[1].bar(xx, 100 * table.coverage, 0.55, color=[COLORS[s] for s in scopes])
    axes[1].axhline(25, ls="--", lw=0.7, color="0.5", label="25% gate")
    axes[1].set(xticks=xx, xticklabels=scopes, ylabel="Certificate coverage (%)", title="S2-positive coverage", ylim=(0, 100))
    axes[1].legend(frameon=False)
    export(fig, "harm_coverage")
    plt.close(fig)

    pooled = subjects[subjects.scope == "Pooled"].copy()
    pooled["_sort"] = pooled.subject_id.astype(int)
    pooled = pooled.sort_values("_sort")
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, 92 / 25.4), constrained_layout=True)
    x = np.arange(len(pooled))
    d2, d3 = 100 * pooled.Delta_S2_BA.to_numpy(), 100 * pooled.Delta_S3_BA.to_numpy()
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [d2[i], d3[i]], color="0.75", lw=0.6, zorder=1)
    ax.scatter(x, d2, s=12, label=r"S2 $\Delta$ BA", color="#2671b8", zorder=2)
    ax.scatter(x, d3, s=12, label=r"S3 $\Delta$ BA", color="#d95f02", marker="s", zorder=2)
    ax.axhline(0, color="0.35", lw=0.7)
    ax.set(xticks=x, xticklabels=pooled.subject_id.astype(str), xlabel="Development subject", ylabel="Utility (pp)")
    ax.tick_params(axis="x", labelrotation=90)
    ax.legend(frameon=False, ncol=2)
    export(fig, "per_subject_transfer")
    plt.close(fig)
    print("SCAA_STAGE0_PUBLICATION_FIGURES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
