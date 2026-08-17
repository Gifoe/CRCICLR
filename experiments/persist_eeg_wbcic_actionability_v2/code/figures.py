"""Generate compact negative-result figures from frozen WBCIC result tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2"
OUT = EXP_ROOT / "outputs"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
BLOCK_ORDER = ["P01_04", "P05_08", "P09_16", "P17_32"]
COLORS = {
    "PROTECTED": "#2f6f9f",
    "DECISION-NULL / WEAKLY ACTIVE": "#888888",
    "DECISION-ACTIVE NON-ACTIONABLE": "#b07d2b",
    "UNCERTAIN": "#bdbdbd",
    "ACTIONABLE-HARMFUL": "#b23a48",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def persistence_spectrum() -> None:
    frame = pd.read_csv(RESULTS / "PERSISTENCE_RESULTS.csv").set_index("block").loc[BLOCK_ORDER]
    x = np.arange(len(frame))
    mean = frame["mean_specific_advantage"].to_numpy()
    low = frame["CI95_L"].to_numpy()
    high = frame["CI95_U"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.axhline(0, color="black", lw=0.8)
    ax.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), fmt="o", color="#315f8c", capsize=3)
    ax.set_xticks(x, BLOCK_ORDER)
    ax.set_ylabel("Same-subject persistence advantage\n(candidate − matched random)")
    ax.set_title("WBCIC cross-session persistence spectrum")
    for index, row in enumerate(frame.itertuples()):
        ax.text(index, high[index] + 0.04, f"Holm p={row.p_holm:.3g}", ha="center", va="bottom", fontsize=7)
    fig.savefig(FIGURES / "figure2_wbcic_persistence_spectrum.png")
    plt.close(fig)


def geometry_map() -> None:
    utility = pd.read_csv(RESULTS / "SIGNED_UTILITY_RESULTS.csv")
    decision = pd.read_csv(RESULTS / "DECISION_DEPENDENCE_RESULTS.csv")
    action = pd.read_csv(RESULTS / "ACTIONABILITY_RESULTS.csv")
    assignment = pd.read_csv(RESULTS / "BLOCK_ASSIGNMENTS.csv")
    frame = utility.merge(decision, on=["block", "rank"]).merge(action, on=["block", "rank"])
    frame = frame.merge(assignment, on=["block", "rank"]).set_index("block").loc[BLOCK_ORDER]
    sizes = 80 + 900 * np.minimum(np.abs(frame["delta_BA_specific_mean"].to_numpy()), 0.08) / 0.08
    colors = [COLORS.get(value, "#888888") for value in frame["assignment"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.scatter(frame["u_spec_mean"], frame["finite_ratio_mean"], s=sizes, c=colors, edgecolor="white", lw=0.8)
    for block, row in frame.iterrows():
        ax.annotate(block, (row["u_spec_mean"], row["finite_ratio_mean"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(r"Signed utility $u_{spec}$ (positive = preserve)")
    ax.set_ylabel("Finite decision-dependence ratio")
    ax.set_title("Utility, decision dependence, and actionability")
    ax.text(0.02, 0.02, "Marker area ∝ |specific BA effect|", transform=ax.transAxes, fontsize=8)
    fig.savefig(FIGURES / "figure3_utility_decision_actionability.png")
    plt.close(fig)


def subject_pairs() -> None:
    frame = pd.read_csv(RESULTS / "WBCIC_AUDIT_SUBJECT_RESULTS.csv")
    frame = frame[frame["block"] == "P01_04"].sort_values("base_BA")
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.3, 3.5))
    for index, row in enumerate(frame.itertuples()):
        color = "#b23a48" if row.candidate_BA < row.base_BA else "#477f54"
        ax.plot([index, index], [row.base_BA, row.candidate_BA], color=color, alpha=0.55, lw=0.9)
    ax.scatter(x, frame["base_BA"], s=12, color="#333333", label="Baseline")
    ax.scatter(x, frame["candidate_BA"], s=12, color="#2f6f9f", label="Erase P01_04")
    ax.set_ylim(0.25, 1.02)
    ax.set_xlabel("Held-out outcome subject (ordered by baseline BA)")
    ax.set_ylabel("S3 balanced accuracy")
    ax.set_title("Protected P01_04 erasure harms future-session performance")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(FIGURES / "figure5_subject_level_p01_erasure.png")
    plt.close(fig)


def project_concept() -> None:
    labels = ["Persistence", "Signed utility", "Decision\ndependence", "Actionability", "Action"]
    fig, ax = plt.subplots(figsize=(8.0, 1.8))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(x, 0.55, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.35", "fc": "#eef3f7", "ec": "#315f8c"})
        if index < len(labels) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.075, 0.55), xytext=(x + 0.075, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.5, 0.08, "A gate failure maps to NO-OP; persistence alone never authorizes suppression.", ha="center", color="#555555")
    fig.savefig(FIGURES / "figure1_project_concept.png")
    plt.close(fig)


def final_action() -> None:
    final = json.loads((OUT / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.axis("off")
    ax.text(0.25, 0.68, "OpenBMI", ha="center", fontsize=12, weight="bold")
    ax.text(0.25, 0.38, "Existing evidence:\nProtected / preserve", ha="center", bbox={"boxstyle": "round,pad=0.45", "fc": "#e8f1f8", "ec": "#2f6f9f"})
    ax.text(0.75, 0.68, "WBCIC", ha="center", fontsize=12, weight="bold")
    ax.text(0.75, 0.38, "No H1–H5 target:\nNO-OP", ha="center", bbox={"boxstyle": "round,pad=0.45", "fc": "#f1f1f1", "ec": "#777777"})
    ax.text(0.5, 0.05, final["terminal_state"], ha="center", fontsize=8, color="#555555")
    fig.savefig(FIGURES / "figure6_final_cross_dataset_action.png")
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    project_concept()
    persistence_spectrum()
    geometry_map()
    subject_pairs()
    final_action()
    repair = {
        "status": "POST_OUTCOME_REPORTING_REPAIR_ONLY",
        "outcome_metrics_changed": False,
        "protocol_or_gate_changed": False,
        "changes": [
            "Corrected REPRODUCIBILITY commands from the prospectively suggested workers=4 to the actually completed workers=0 runtime after a Windows torch worker-spawn deadlock.",
            "Added deterministic figures generated only from already frozen compact result tables.",
        ],
    }
    (OUT / "POST_OUTCOME_REPORTING_REPAIR.json").write_text(json.dumps(repair, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FIGURES_COMPLETE", "figure_count": 5}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
