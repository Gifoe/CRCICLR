from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as c


COLORS = {"EEGNet": "#2878b5", "EEGConformer": "#d65f2e"}


def export(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(c.FIGURES / f"{name}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    c.verify_feature_lock(require_committed=True)
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(c.RESULTS / "PER_SUBJECT_FEATURES.csv", dtype={"subject_id": str})
    outcomes = pd.read_csv(c.RESULTS / "PER_SUBJECT_RELIABILITY_OUTCOMES.csv", dtype={"subject_id": str})
    frame = features.merge(outcomes, on=["backbone", "fold", "subject_id"], validate="one_to_one")
    cv = pd.read_csv(c.RESULTS / "CROSS_VALIDATED_RELIABILITY.csv")
    policy_results = pd.read_csv(c.RESULTS / "RELIABILITY_POLICY_RESULTS.csv")
    policy = pd.read_csv(c.RESULTS / "PER_SUBJECT_POLICY.csv", dtype={"subject_id": str})
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    # Figure 1: frozen architecture contrast, without treating rows as independent humans.
    summary = []
    for backbone in c.BACKBONES:
        current = frame[frame.backbone == backbone]
        accepted = current[current.Delta2 > 0]
        summary.append(
            {
                "backbone": backbone,
                "sign": current.R_sign.mean(),
                "harm": accepted.H.mean() if len(accepted) else np.nan,
            }
        )
    summary = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, 100 * summary.sign, width, label="Sign persistence", color="#6a9fb5")
    ax.bar(x + width / 2, 100 * summary.harm, width, label="Harm among S2-positive", color="#d95f5f")
    ax.set(xticks=x, xticklabels=summary.backbone, ylabel="Rate (%)", ylim=(0, 100), title="Certificate reliability by backbone")
    ax.legend(frameon=False)
    export(fig, "figure1_backbone_certificate_reliability")

    # Figure 2: the two highest-priority stability families against signed persistence.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    for ax, score, title in zip(
        axes,
        ("adaptation_effect_stability", "decision_stability"),
        ("Adaptation-effect stability", "Decision stability"),
    ):
        for backbone in c.BACKBONES:
            current = frame[frame.backbone == backbone]
            ax.scatter(
                current[score],
                1e4 * current.signed_persistence,
                s=23,
                alpha=0.78,
                color=COLORS[backbone],
                label=backbone,
                edgecolor="white",
                linewidth=0.25,
            )
        rho = pd.Series(frame[score]).corr(pd.Series(frame.signed_persistence), method="spearman")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set(xlabel="Frozen stability score (higher = more stable)", ylabel=r"$10^4\,\Delta_2\Delta_3$", title=f"{title} (rho={rho:.2f})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center")
    export(fig, "figure2_stability_vs_signed_persistence")

    # Figure 3: out-of-subject prediction; M4 is visibly unavailable.
    order = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    labels = ["Backbone", "Raw Delta2", "SNR", "Identity", "Rep. stability", "Decision stability", "Effect stability", "Combined"]
    current = cv[(cv.outcome == "R_sign")].set_index("model").reindex(order)
    values = 100 * current.AUROC.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.8, 3.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(order)), np.nan_to_num(values, nan=0.0), color=["#777777", "#999999", "#66a61e", "#dddddd", "#1b9e77", "#7570b3", "#e7298a", "#d95f02"])
    ax.axhline(50, color="0.35", ls="--", lw=0.9)
    ax.set(xticks=np.arange(len(order)), xticklabels=labels, ylabel="OOF AUROC (%)", title="Predicting sign persistence", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=25)
    identity_index = order.index("M4")
    ax.text(identity_index, 4, "N/A", ha="center", va="bottom", fontsize=8)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}", ha="center", fontsize=7)
    export(fig, "figure3_identity_vs_mechanism_predictors")

    # Figure 4: risk-coverage policy plane.
    fig, ax = plt.subplots(figsize=(5.8, 3.9), constrained_layout=True)
    plotted = policy_results[policy_results.policy != "Anchor"]
    for _, row in plotted.iterrows():
        if np.isfinite(row.future_harm_given_adaptation):
            ax.scatter(100 * row.coverage, 100 * row.future_harm_given_adaptation, s=75)
            ax.annotate(row.policy, (100 * row.coverage, 100 * row.future_harm_given_adaptation), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Adaptation coverage (%)", ylabel="Future harm among adapted (%)", title="Development risk-coverage", xlim=(-2, 103), ylim=(-2, 103))
    export(fig, "figure4_policy_risk_coverage")

    # Figure 5: every paired subject/backbone outcome and accepted policy decisions.
    ordered_subjects = c.subject_sort(policy.subject_id.unique())
    position = {subject: index for index, subject in enumerate(ordered_subjects)}
    fig, ax = plt.subplots(figsize=(10.2, 4.3), constrained_layout=True)
    offsets = {"EEGNet": -0.13, "EEGConformer": 0.13}
    for backbone in c.BACKBONES:
        current = policy[policy.backbone == backbone]
        x = np.array([position[subject] for subject in current.subject_id]) + offsets[backbone]
        face = np.where(current.reliability_gate.astype(bool), COLORS[backbone], "white")
        ax.scatter(x, 100 * current.Delta3, s=28, facecolors=face, edgecolors=COLORS[backbone], linewidth=0.9, label=backbone)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(xticks=np.arange(len(ordered_subjects)), xticklabels=ordered_subjects, xlabel="Development subject", ylabel="Future utility Delta3 (pp)", title="Reliability-gated decisions (filled = adapted)")
    ax.legend(frameon=False, ncol=2)
    export(fig, "figure5_per_subject_policy_outcome")

    c.write_text(
        c.EXP / "FIGURE_QA.md",
        """# Figure QA

All five figures are generated from compact Stage-0.5 result tables. Axes, units, legends, and the unavailable Identity control are explicit. PNG and vector PDF versions were exported. No outer or OpenBMI data were used.
""",
    )
    print("SCAA_RELIABILITY_STAGE05_FIGURES_COMPLETE")


if __name__ == "__main__":
    main()

