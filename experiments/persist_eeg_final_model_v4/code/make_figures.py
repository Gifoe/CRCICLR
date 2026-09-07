from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DIAGNOSTICS, FIGURES, LEADERBOARD, OUTPUTS, ensure_directories, write_csv


PALETTE = {
    "neutral": "#6B7280",
    "keep": "#3B82A0",
    "action": "#D28B43",
    "persist": "#8A6FA8",
    "negative": "#B85C5C",
    "grid": "#D8DCE2",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "legend.fontsize": 6.5,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(
        FIGURES / f"{stem}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def _delta_panel(ax: plt.Axes, frame: pd.DataFrame, labels: dict[str, str], colors: dict[str, str], title: str) -> None:
    values = frame[frame.method_id.isin(labels)].copy()
    values["order"] = values.method_id.map({method: index for index, method in enumerate(labels)})
    values = values.sort_values("order", ascending=False)
    y = np.arange(len(values))
    delta = 100 * values.Delta_BA_vs_B_STRONG.to_numpy(dtype=float)
    lower = delta - 100 * values.CI95_L.to_numpy(dtype=float)
    upper = 100 * values.CI95_U.to_numpy(dtype=float) - delta
    for index, row in enumerate(values.itertuples(index=False)):
        method = str(row.method_id)
        ax.errorbar(
            delta[index],
            y[index],
            xerr=np.asarray([[lower[index]], [upper[index]]]),
            fmt="o",
            color=colors[method],
            ecolor=colors[method],
            elinewidth=1.1,
            capsize=2,
            markersize=4,
            zorder=3,
        )
    ax.axvline(0, color="#222222", lw=0.8, ls="--")
    ax.set_yticks(y, [labels[str(value)] for value in values.method_id])
    ax.set_xlabel(r"$\Delta$ balanced accuracy vs. strongest static ensemble (pp)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.5, zorder=0)


def make() -> None:
    ensure_directories()
    _style()
    source = FIGURES / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    ablation = pd.read_csv(OUTPUTS / "ablations" / "FINAL_MODEL_ABLATIONS.csv")
    incremental = pd.read_csv(OUTPUTS / "ablations" / "PERSIST_INCREMENTAL_VALUE.csv")
    open_leader = pd.read_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv")
    wbcic_search = pd.read_csv(LEADERBOARD / "WBCIC_DEV_KEEP_SEARCH.csv")
    wbcic_transfer = pd.read_csv(LEADERBOARD / "WBCIC_DEV_MODEL_LEADERBOARD.csv")
    subject = pd.read_csv(DIAGNOSTICS / "FINAL_ABLATION_SUBJECT_RESULTS.csv")
    baseline = json.loads((OUTPUTS / "protocol" / "BASELINE_RECONSTRUCTION.json").read_text(encoding="utf-8"))

    # Figure 1: one hero comparison plus validation, subject stability, and oracle headroom.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    open_labels = {
        "A1_DYNAMIC_KEEP_FINAL": "Dynamic KEEP",
        "A2_KEEP_ACTION_NO_PERSIST": "+ ACTION",
        "A3_KEEP_ACTION_PERSIST": "+ ACTION + PERSIST",
    }
    open_colors = {
        "A1_DYNAMIC_KEEP_FINAL": PALETTE["keep"],
        "A2_KEEP_ACTION_NO_PERSIST": PALETTE["action"],
        "A3_KEEP_ACTION_PERSIST": PALETTE["persist"],
    }
    _delta_panel(axes[0, 0], ablation, open_labels, open_colors, "OpenBMI discovery (n=52 subjects)")
    _panel_label(axes[0, 0], "a")

    direct_transfer = wbcic_transfer[
        wbcic_transfer.method_id.eq("W1_MASKED_POOL_SHRUNK_THR")
    ].copy()
    direct_transfer["Delta_BA_vs_B_STRONG"] = direct_transfer[
        "Delta_BA_vs_WBCIC_B_STRONG"
    ]
    direct_transfer["CI95_L"] = direct_transfer["CI95_L_vs_WBCIC_B_STRONG"]
    direct_transfer["CI95_U"] = direct_transfer["CI95_U_vs_WBCIC_B_STRONG"]
    wbcic_combined = pd.concat(
        [
            wbcic_search[
                wbcic_search.method_id.isin(["W1_RAW_LINEAR", "W1_DEEPSETS_KEEP"])
            ],
            direct_transfer,
        ],
        ignore_index=True,
    )
    wbcic_labels = {
        "W1_RAW_LINEAR": "Generic linear stack",
        "W1_DEEPSETS_KEEP": "DeepSets",
        "W1_MASKED_POOL_SHRUNK_THR": "Direct architecture transfer",
    }
    wbcic_colors = {
        "W1_RAW_LINEAR": PALETTE["keep"],
        "W1_DEEPSETS_KEEP": "#6B9C78",
        "W1_MASKED_POOL_SHRUNK_THR": PALETTE["negative"],
    }
    _delta_panel(axes[0, 1], wbcic_combined, wbcic_labels, wbcic_colors, "WBCIC development (n=41 subjects)")
    _panel_label(axes[0, 1], "b")

    methods = ["A1_DYNAMIC_KEEP_FINAL", "A2_KEEP_ACTION_NO_PERSIST", "A3_KEEP_ACTION_PERSIST"]
    for index, method in enumerate(methods):
        values = 100 * subject[subject.method_id.eq(method)].delta_BA_vs_B_STRONG.to_numpy(dtype=float)
        # Deterministic display-only offset; every subject remains present.
        jitter = 0.045 * np.sin(np.arange(len(values), dtype=float) * 2.399963229728653)
        axes[1, 0].scatter(
            np.full(len(values), index) + jitter,
            values,
            s=9,
            alpha=0.55,
            color=open_colors[method],
            edgecolors="none",
            rasterized=True,
        )
        axes[1, 0].plot([index - 0.18, index + 0.18], [np.mean(values), np.mean(values)], color="#111111", lw=1.5)
    axes[1, 0].axhline(0, color="#222222", lw=0.8, ls="--")
    axes[1, 0].set_xticks(range(3), ["Dynamic\nKEEP", "+ ACTION", "+ ACTION\n+ PERSIST"])
    axes[1, 0].set_ylabel("Subject-level $\Delta$ BA (pp)")
    axes[1, 0].set_title("OpenBMI subject effects (all 52 shown)", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="y", color=PALETTE["grid"], lw=0.5)
    _panel_label(axes[1, 0], "c")

    oracle_labels = ["Single model", "Static B_STRONG", "Dynamic KEEP", "KEEP oracle", "KEEP+ACTION oracle"]
    oracle_values = [
        0.8233173076923076,
        float(baseline["B_STRONG_mean_subject_BA"]),
        float(ablation[ablation.method_id.eq("A1_DYNAMIC_KEEP_FINAL")].iloc[0].mean_subject_BA),
        float(baseline["KEEP_only_oracle_BA"]),
        float(baseline["complete_KEEP_ACTION_oracle_BA"]),
    ]
    bars = axes[1, 1].barh(
        np.arange(len(oracle_labels)),
        100 * np.asarray(oracle_values),
        color=[PALETTE["neutral"], "#9AA2AD", PALETTE["keep"], "#9DBFD0", "#D7B384"],
        height=0.65,
    )
    axes[1, 1].set_yticks(np.arange(len(oracle_labels)), oracle_labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(80, 97)
    axes[1, 1].set_xlabel("Mean subject balanced accuracy (%)")
    axes[1, 1].set_title("Headroom remains selection-limited", loc="left", fontweight="bold")
    axes[1, 1].grid(axis="x", color=PALETTE["grid"], lw=0.5)
    for bar, value in zip(bars, oracle_values):
        axes[1, 1].text(100 * value + 0.18, bar.get_y() + bar.get_height() / 2, f"{100*value:.2f}", va="center", fontsize=6.3)
    _panel_label(axes[1, 1], "d")
    _save(fig, "FIG1_FINAL_MODEL_EVIDENCE")

    write_csv(source / "FIG1_OPENBMI_METHODS.csv", ablation[ablation.method_id.isin(open_labels)])
    write_csv(source / "FIG1_WBCIC_METHODS.csv", wbcic_combined[wbcic_combined.method_id.isin(wbcic_labels)])
    write_csv(source / "FIG1_SUBJECT_EFFECTS.csv", subject[subject.method_id.isin(methods)])
    write_csv(source / "FIG1_ORACLE_LADDER.csv", pd.DataFrame({"method": oracle_labels, "mean_subject_BA": oracle_values}))

    # Figure 2: component increments with paired subject-bootstrap confidence intervals.
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    plot = incremental.copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot))
    delta = 100 * plot.Delta_BA.to_numpy(dtype=float)
    lower = delta - 100 * plot.CI95_L.to_numpy(dtype=float)
    upper = 100 * plot.CI95_U.to_numpy(dtype=float) - delta
    color = [PALETTE["keep"] if value > 0 else PALETTE["negative"] for value in delta]
    for index in range(len(plot)):
        ax.errorbar(delta[index], y[index], xerr=[[lower[index]], [upper[index]]], fmt="o", color=color[index], capsize=2, lw=1.1, ms=4)
    pretty = {
        "dynamic_KEEP_value": "Dynamic KEEP vs static",
        "ACTION_increment": "ACTION beyond dynamic KEEP",
        "PERSIST_increment": "PERSIST beyond KEEP+ACTION",
        "protected_increment": "Protected features",
        "decision_dependence_increment": "Decision-dependence features",
        "persistence_increment": "Persistence features",
        "action_movement_increment": "Action-movement features",
        "disagreement_increment": "Ensemble-disagreement features",
        "PERSIST_vs_capacity_control": "PERSIST vs capacity control",
    }
    ax.set_yticks(y, [pretty[value] for value in plot.comparison])
    ax.axvline(0, color="#222222", lw=0.8, ls="--")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.5)
    ax.set_xlabel(r"Paired subject-level $\Delta$ balanced accuracy (pp; 95% bootstrap CI)")
    ax.set_title("Only dynamic KEEP has a confidence interval above zero", loc="left", fontweight="bold")
    _save(fig, "FIG2_COMPONENT_ABLATIONS")
    write_csv(source / "FIG2_COMPONENT_ABLATIONS.csv", incremental)

    # Figure 3: cross-benchmark decision surface, showing robust vs exploratory gains.
    fig, ax = plt.subplots(figsize=(5.0, 3.8), constrained_layout=True)
    cross = pd.DataFrame(
        [
            {
                "candidate": "Direct masked-pool transfer",
                "OpenBMI_delta_pp": 100 * float(open_leader[open_leader.method_id.eq("M1_MASKED_POOL_SHRUNK_THR")].iloc[0].Delta_BA_vs_B_STRONG),
                "WBCIC_delta_pp": 100 * float(wbcic_transfer[wbcic_transfer.method_id.eq("W1_MASKED_POOL_SHRUNK_THR")].iloc[0].Delta_BA_vs_WBCIC_B_STRONG),
                "color": PALETTE["negative"],
            },
            {
                "candidate": "Benchmark-adapted linear stack",
                "OpenBMI_delta_pp": 100 * float(open_leader[open_leader.method_id.eq("M1_DYNAMIC_KEEP_LINEAR")].iloc[0].Delta_BA_vs_B_STRONG),
                "WBCIC_delta_pp": 100 * float(wbcic_search[wbcic_search.method_id.eq("W1_RAW_LINEAR")].iloc[0].Delta_BA_vs_B_STRONG),
                "color": PALETTE["keep"],
            },
        ]
    )
    ax.axhline(0, color="#222222", lw=0.7, ls="--")
    ax.axvline(0, color="#222222", lw=0.7, ls="--")
    for row in cross.itertuples(index=False):
        ax.scatter(row.OpenBMI_delta_pp, row.WBCIC_delta_pp, s=42, color=row.color, zorder=3)
        ax.annotate(row.candidate, (row.OpenBMI_delta_pp, row.WBCIC_delta_pp), xytext=(5, 5), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("OpenBMI $\Delta$ BA (pp)")
    ax.set_ylabel("WBCIC-dev $\Delta$ BA (pp)")
    ax.set_title("No architecture is robustly positive on both benchmarks", loc="left", fontweight="bold")
    ax.grid(color=PALETTE["grid"], lw=0.5)
    _save(fig, "FIG3_CROSS_BENCHMARK_TRANSFER")
    write_csv(source / "FIG3_CROSS_BENCHMARK_TRANSFER.csv", cross.drop(columns="color"))


if __name__ == "__main__":
    make()
