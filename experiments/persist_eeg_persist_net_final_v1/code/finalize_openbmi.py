"""Aggregate the frozen OpenBMI development experiment and evaluate G1-G9."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import core


METHOD_ORDER = [
    "B0_VANILLA_EEGNET",
    "B1_STRONG_EEGNET",
    "B2_STRONG_GENERIC",
    "A2_DUAL_CONTROL",
    "A3_P_ONLY",
    "A4_P_PLUS_U",
    "A5_P_PLUS_D",
    "A7_IDENTITY_PROTECTED",
    "A8_RANDOM_PROTECTED",
    "A9_PCA_PROTECTED",
    "A6_PUD_ALL_ADAPT",
    "A10_FULL_PUD_FREEZE",
]

DISPLAY = {
    "B0_VANILLA_EEGNET": "Vanilla EEGNet",
    "B1_STRONG_EEGNET": "Strong EEGNet",
    "B2_STRONG_GENERIC": "Strong Generic",
    "A2_DUAL_CONTROL": "Dual capacity control",
    "A3_P_ONLY": "P-only",
    "A4_P_PLUS_U": "P+U",
    "A5_P_PLUS_D": "P+D",
    "A6_PUD_ALL_ADAPT": "PUD all-adapt",
    "A7_IDENTITY_PROTECTED": "Identity-protected",
    "A8_RANDOM_PROTECTED": "Random-protected",
    "A9_PCA_PROTECTED": "PCA-protected",
    "A10_FULL_PUD_FREEZE": "PUD protected-freeze FULL",
}


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    """Render the small final table without pandas' optional tabulate extra."""

    def render(value: Any) -> str:
        if pd.isna(value):
            return "nan"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    subject_frames = []
    mechanism_frames = []
    efficiency_frames = []
    done = []
    for fold, seed in itertools.product(range(5), range(3)):
        directory = core.RUNTIME_RUNS / f"fold-{fold}" / f"seed-{seed}"
        done_path = directory / "DONE.json"
        if not done_path.is_file():
            raise FileNotFoundError(f"Incomplete run: fold={fold}, seed={seed}")
        state = json.loads(done_path.read_text(encoding="utf-8"))
        if state.get("status") != "RUN_COMPLETE":
            raise RuntimeError(f"Invalid run state: {state}")
        done.append(state)
        subject_frames.append(pd.read_csv(directory / "SUBJECT_RESULTS.csv"))
        mechanism_frames.append(pd.read_csv(directory / "MECHANISM_METRICS.csv"))
        efficiency_frames.append(pd.read_csv(directory / "EFFICIENCY.csv"))
    subject = pd.concat(subject_frames, ignore_index=True)
    mechanism = pd.concat(mechanism_frames, ignore_index=True)
    efficiency = pd.concat(efficiency_frames, ignore_index=True)
    expected_primary = set(core.protocol()["primary_methods_three_seeds"])
    expected_secondary = set(core.protocol()["secondary_methods_seed0"])
    for method in expected_primary:
        q = subject.loc[subject.method.eq(method)]
        if len(q) != 120 or q.subject_id.astype(str).nunique() != 40 or set(q.seed) != {0, 1, 2}:
            raise RuntimeError(f"Primary method coverage failure {method}: {q.shape}")
    for method in expected_secondary:
        q = subject.loc[subject.method.eq(method)]
        if len(q) != 40 or q.subject_id.astype(str).nunique() != 40 or set(q.seed) != {0}:
            raise RuntimeError(f"Secondary method coverage failure {method}: {q.shape}")
    return subject, mechanism, efficiency, done


def bootstrap(values: np.ndarray, seed: int, draws: int = 10_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[sampled].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "CI95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "draws": draws,
        "subjects": int(len(array)),
    }


def sign_flip(values: np.ndarray, seed: int, draws: int = 100_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    observed = float(abs(array.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(array)))
    null = np.abs((signs * array[None, :]).mean(axis=1))
    return {
        "two_sided_p": float((1 + np.sum(null >= observed - 1e-15)) / (draws + 1)),
        "draws": draws,
        "exact": False,
        "reason_not_exact": "2^40 sign patterns is computationally disproportionate; frozen Monte Carlo test used",
    }


def aggregate_subjects(subject: pd.DataFrame) -> pd.DataFrame:
    numeric = ["BA", "macro_f1", "source_model_noadapt_BA", "adaptation_delta_BA", "adaptation_time_s", "target_trainable_parameters"]
    return (
        subject.groupby(["method", "subject_id"], as_index=False)[numeric]
        .mean(numeric_only=True)
        .sort_values(["method", "subject_id"])
        .reset_index(drop=True)
    )


def method_summary(subject: pd.DataFrame, per_subject: pd.DataFrame, baseline_method: str) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        q = per_subject.loc[per_subject.method.eq(method)].copy()
        if q.empty:
            continue
        raw = subject.loc[subject.method.eq(method)]
        matched_seeds = set(raw.seed.astype(int).unique().tolist())
        baseline = (
            subject.loc[
                subject.method.eq(baseline_method) & subject.seed.astype(int).isin(matched_seeds)
            ]
            .groupby("subject_id", as_index=False)
            .BA.mean()
            .set_index("subject_id")
        )
        q = q.set_index("subject_id")
        common = q.index.intersection(baseline.index)
        delta = q.loc[common, "BA"].to_numpy() - baseline.loc[common, "BA"].to_numpy()
        stats = bootstrap(delta, core.stable_seed("method-bootstrap", method))
        adaptation = q.adaptation_delta_BA.dropna().to_numpy(dtype=float)
        ntr = float(np.mean(adaptation < -1e-12)) if len(adaptation) else float("nan")
        worst = float(np.mean(np.sort(adaptation)[: max(1, int(math.ceil(len(adaptation) * 0.25)))])) if len(adaptation) else float("nan")
        rows.append(
            {
                "method": method,
                "display_name": DISPLAY[method],
                "seeds": int(raw.seed.nunique()),
                "subjects": int(q.index.nunique()),
                "BA": float(q.BA.mean()),
                "Macro_F1": float(q.macro_f1.mean()),
                "Delta_BA_vs_strongest_baseline": stats["mean"],
                "CI95_L": stats["CI95"][0],
                "CI95_U": stats["CI95"][1],
                "Negative_Transfer_Rate": ntr,
                "Worst_Quartile_Adaptation_Delta": worst,
                "harmed_subjects": int(np.sum(adaptation < -1e-12)) if len(adaptation) else 0,
                "rescued_subjects": int(np.sum(adaptation > 1e-12)) if len(adaptation) else 0,
                "Params": np.nan,
                "Target_Trainable_Params": float(q.target_trainable_parameters.mean()),
            }
        )
    return pd.DataFrame(rows)


def add_efficiency(summary: pd.DataFrame, efficiency: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "B0_VANILLA_EEGNET": "B0_VANILLA_EEGNET",
        "B1_STRONG_EEGNET": "B1_STRONG_EEGNET",
        "B2_STRONG_GENERIC": "B1_STRONG_EEGNET",
        "A2_DUAL_CONTROL": "A2_DUAL_CONTROL",
        "A3_P_ONLY": "A3_P_ONLY",
        "A4_P_PLUS_U": "A4_P_PLUS_U",
        "A5_P_PLUS_D": "A5_P_PLUS_D",
        "A6_PUD_ALL_ADAPT": "PUD_SOURCE",
        "A7_IDENTITY_PROTECTED": "A7_IDENTITY_PROTECTED",
        "A8_RANDOM_PROTECTED": "A8_RANDOM_PROTECTED",
        "A9_PCA_PROTECTED": "A9_PCA_PROTECTED",
        "A10_FULL_PUD_FREEZE": "PUD_SOURCE",
    }
    result = summary.copy()
    for index, row in result.iterrows():
        source = mapping[row.method]
        q = efficiency.loc[efficiency.source_model.eq(source)]
        result.loc[index, "Params"] = float(q.parameters.mean())
        result.loc[index, "Approximate_MACs"] = float(q.approximate_MACs.mean())
        result.loc[index, "Capacity_Ratio_vs_B1"] = float(q.capacity_ratio_vs_B1.mean())
    return result


def fold_seed_tables(subject: pd.DataFrame, baseline_method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold = subject.groupby(["method", "fold"], as_index=False).agg(BA=("BA", "mean"), Macro_F1=("macro_f1", "mean"))
    seed = subject.groupby(["method", "seed"], as_index=False).agg(BA=("BA", "mean"), Macro_F1=("macro_f1", "mean"))
    fold_baselines = []
    for row in fold.itertuples():
        method_seeds = set(
            subject.loc[
                subject.method.eq(row.method) & subject.fold.eq(row.fold), "seed"
            ].astype(int)
        )
        matched = subject.loc[
            subject.method.eq(baseline_method)
            & subject.fold.eq(row.fold)
            & subject.seed.astype(int).isin(method_seeds)
        ]
        fold_baselines.append(float(matched.BA.mean()))
    fold["baseline_BA"] = fold_baselines
    seed_base = seed.loc[seed.method.eq(baseline_method), ["seed", "BA"]].rename(columns={"BA": "baseline_BA"})
    seed = seed.merge(seed_base, on="seed", how="left")
    fold["Delta_BA_vs_strongest_baseline"] = fold.BA - fold.baseline_BA
    seed["Delta_BA_vs_strongest_baseline"] = seed.BA - seed.baseline_BA
    return fold, seed


def mechanism_summary(mechanism: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "protected_representation_drift",
        "adaptive_representation_drift",
        "protected_decision_logit_drift",
        "adaptive_decision_logit_drift",
        "protected_D_finite_after",
        "functional_distillation_RMSE_before",
        "functional_distillation_correlation_before",
        "functional_distillation_RMSE_after",
        "functional_distillation_correlation_after",
        "protected_branch_erasure_harm_BA",
        "adaptive_branch_erasure_harm_BA",
        "protected_parameter_update_l2",
        "adaptive_parameter_update_l2",
        "protected_buffer_update_l2",
    ]
    existing = [column for column in numeric if column in mechanism.columns]
    return mechanism.groupby("method", as_index=False)[existing].mean(numeric_only=True)


def paired_mechanism_comparison(
    frame: pd.DataFrame,
    left_method: str,
    right_method: str,
    left_metric: str,
    right_metric: str | None = None,
) -> dict[str, Any]:
    right_metric = right_metric or left_metric
    left = frame.loc[frame.method.eq(left_method)].set_index("subject_id")[left_metric]
    right = frame.loc[frame.method.eq(right_method)].set_index("subject_id")[right_metric]
    common = left.index.intersection(right.index)
    if len(common) == 0:
        raise RuntimeError(
            f"No paired subjects for {left_method}:{left_metric} vs "
            f"{right_method}:{right_metric}"
        )
    left_values = left.loc[common].to_numpy(dtype=float)
    right_values = right.loc[common].to_numpy(dtype=float)
    stats = bootstrap(
        left_values - right_values,
        core.stable_seed(
            "mechanism-bootstrap", left_method, right_method, left_metric, right_metric
        ),
    )
    return {
        "left_method": left_method,
        "right_method": right_method,
        "left_metric": left_metric,
        "right_metric": right_metric,
        "left_mean": float(np.mean(left_values)),
        "right_mean": float(np.mean(right_values)),
        "mean_difference": stats["mean"],
        "CI95": stats["CI95"],
        "subjects": int(len(common)),
        "bootstrap_draws": stats["draws"],
    }


def draw_figures(summary: pd.DataFrame, per_subject: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    core.FIGURES.mkdir(parents=True, exist_ok=True)
    ordered = summary.set_index("method").loc[[m for m in METHOD_ORDER if m in set(summary.method)]].reset_index()
    colors = ["#425a70" if not m.startswith("A10") else "#b43c3c" for m in ordered.method]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(ordered))
    ax.bar(x, 100 * ordered.BA, color=colors)
    ax.set_xticks(x, ordered.display_name, rotation=35, ha="right")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_title("OpenBMI V8_SEARCH future-session performance")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(core.FIGURES / "main_performance.png", dpi=220)
    fig.savefig(core.FIGURES / "main_performance.pdf")
    plt.close(fig)

    baseline_methods = {"B0_VANILLA_EEGNET", "B1_STRONG_EEGNET", "B2_STRONG_GENERIC"}
    ablation = ordered.loc[~ordered.method.isin(baseline_methods)]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(ablation.display_name, 100 * ablation.Delta_BA_vs_strongest_baseline, color="#567b9e")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta balanced accuracy vs strongest legal baseline (pp)")
    ax.set_title("Capacity and theory-specific ablations")
    fig.tight_layout()
    fig.savefig(core.FIGURES / "ablation.png", dpi=220)
    fig.savefig(core.FIGURES / "ablation.pdf")
    plt.close(fig)

    drift = mechanism.loc[mechanism.method.isin(["A6_PUD_ALL_ADAPT", "A10_FULL_PUD_FREEZE"])].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    width = 0.36
    pos = np.arange(len(drift))
    ax.bar(pos - width / 2, drift.protected_decision_logit_drift, width, label="Protected")
    ax.bar(pos + width / 2, drift.adaptive_decision_logit_drift, width, label="Adaptive")
    ax.set_xticks(pos, drift.method.map(DISPLAY), rotation=15)
    ax.set_ylabel("Centered-logit RMS drift")
    ax.set_title("Protected versus adaptive target-time drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(core.FIGURES / "protected_vs_adaptive_drift.png", dpi=220)
    fig.savefig(core.FIGURES / "protected_vs_adaptive_drift.pdf")
    plt.close(fig)

    erase_methods = ["A3_P_ONLY", "A7_IDENTITY_PROTECTED", "A8_RANDOM_PROTECTED", "A10_FULL_PUD_FREEZE"]
    erase = mechanism.loc[mechanism.method.isin(erase_methods)].copy()
    erase["display"] = erase.method.map(DISPLAY)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(erase.display, 100 * erase.protected_branch_erasure_harm_BA, color="#7a6a9c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("BA harm after protected-branch erasure (pp)")
    ax.set_title("Intervention consequence of the trained protected pathway")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(core.FIGURES / "intervention_harm.png", dpi=220)
    fig.savefig(core.FIGURES / "intervention_harm.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.62, 0.18, 0.22, "Raw EEG"),
        (0.30, 0.72, 0.25, 0.18, "Protected EEGNet\n(frozen at target)"),
        (0.30, 0.38, 0.25, 0.18, "Adaptive EEGNet\n(target-trainable)"),
        (0.67, 0.55, 0.26, 0.22, "logits_P + logits_A"),
    ]
    for x0, y0, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, linewidth=1.6))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center")
    ax.annotate("", xy=(0.30, 0.81), xytext=(0.21, 0.73), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.30, 0.47), xytext=(0.21, 0.73), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.67, 0.67), xytext=(0.55, 0.81), arrowprops={"arrowstyle": "->"})
    ax.annotate("", xy=(0.67, 0.62), xytext=(0.55, 0.47), arrowprops={"arrowstyle": "->"})
    ax.text(0.43, 0.94, "Source: task + functional PUD distillation + class-conditioned persistence", ha="center")
    fig.tight_layout()
    fig.savefig(core.FIGURES / "architecture.png", dpi=220)
    fig.savefig(core.FIGURES / "architecture.pdf")
    plt.close(fig)


def finalize() -> dict[str, Any]:
    started = time.time()
    subject, mechanism_raw, efficiency, done = collect()
    per_subject = aggregate_subjects(subject)
    baseline_means = (
        per_subject.loc[per_subject.method.isin(["B0_VANILLA_EEGNET", "B1_STRONG_EEGNET", "B2_STRONG_GENERIC"])]
        .groupby("method").BA.mean()
    )
    baseline_method = str(baseline_means.idxmax())
    summary = add_efficiency(method_summary(subject, per_subject, baseline_method), efficiency)
    fold_table, seed_table = fold_seed_tables(subject, baseline_method)
    mechanism = mechanism_summary(mechanism_raw)
    mechanism_subject = (
        mechanism_raw.groupby(["method", "subject_id"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["method", "subject_id"])
    )
    mechanism_seed0_subject = (
        mechanism_raw.loc[mechanism_raw.seed.eq(0)]
        .groupby(["method", "subject_id"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["method", "subject_id"])
    )
    mechanism_tests = {
        "FULL_minus_all_adapt_protected_decision_drift": paired_mechanism_comparison(
            mechanism_subject,
            "A10_FULL_PUD_FREEZE",
            "A6_PUD_ALL_ADAPT",
            "protected_decision_logit_drift",
        ),
        "FULL_minus_identity_protected_erase_harm": paired_mechanism_comparison(
            mechanism_subject,
            "A10_FULL_PUD_FREEZE",
            "A7_IDENTITY_PROTECTED",
            "protected_branch_erasure_harm_BA",
        ),
        "FULL_minus_random_protected_erase_harm": paired_mechanism_comparison(
            mechanism_subject,
            "A10_FULL_PUD_FREEZE",
            "A8_RANDOM_PROTECTED",
            "protected_branch_erasure_harm_BA",
        ),
        "FULL_protected_minus_adaptive_branch_erase_harm": paired_mechanism_comparison(
            mechanism_subject,
            "A10_FULL_PUD_FREEZE",
            "A10_FULL_PUD_FREEZE",
            "protected_branch_erasure_harm_BA",
            "adaptive_branch_erasure_harm_BA",
        ),
        "FULL_minus_P_only_protected_erase_harm_seed0": paired_mechanism_comparison(
            mechanism_seed0_subject,
            "A10_FULL_PUD_FREEZE",
            "A3_P_ONLY",
            "protected_branch_erasure_harm_BA",
        ),
    }

    baseline_subject = per_subject.loc[per_subject.method.eq(baseline_method)].set_index("subject_id")
    full_subject = per_subject.loc[per_subject.method.eq("A10_FULL_PUD_FREEZE")].set_index("subject_id")
    common = full_subject.index.intersection(baseline_subject.index)
    primary_delta = full_subject.loc[common, "BA"].to_numpy() - baseline_subject.loc[common, "BA"].to_numpy()
    primary_bootstrap = bootstrap(
        primary_delta,
        core.stable_seed("primary-bootstrap"),
        int(core.protocol()["statistics"]["bootstrap_draws"]),
    )
    primary_sign_flip = sign_flip(
        primary_delta,
        core.stable_seed("primary-signflip"),
        int(core.protocol()["statistics"]["sign_flip_draws"]),
    )
    # Keep the exported main-table FULL row identical to the registered
    # primary analysis. method_summary() uses a separate deterministic stream
    # for exploratory method rows; without this assignment the headline and
    # table showed slightly different Monte Carlo CI endpoints for the same
    # FULL-vs-baseline contrast.
    full_row = summary.method.eq("A10_FULL_PUD_FREEZE")
    summary.loc[full_row, "Delta_BA_vs_strongest_baseline"] = primary_bootstrap["mean"]
    summary.loc[full_row, "CI95_L"] = primary_bootstrap["CI95"][0]
    summary.loc[full_row, "CI95_U"] = primary_bootstrap["CI95"][1]

    means = summary.set_index("method").BA
    full_mean = float(means["A10_FULL_PUD_FREEZE"])
    baseline_mean = float(means[baseline_method])
    fold_full = fold_table.loc[fold_table.method.eq("A10_FULL_PUD_FREEZE")]
    seed_full = seed_table.loc[seed_table.method.eq("A10_FULL_PUD_FREEZE")]
    full_adapt = full_subject.adaptation_delta_BA.dropna().to_numpy(dtype=float)
    generic_subject = per_subject.loc[per_subject.method.eq("B2_STRONG_GENERIC")].set_index("subject_id")
    generic_adapt = generic_subject.adaptation_delta_BA.dropna().to_numpy(dtype=float)
    full_ntr = float(np.mean(full_adapt < -1e-12))
    generic_ntr = float(np.mean(generic_adapt < -1e-12))
    full_worst = float(np.mean(np.sort(full_adapt)[:10]))
    generic_worst = float(np.mean(np.sort(generic_adapt)[:10]))
    mech = mechanism.set_index("method")
    full_drift = float(mech.loc["A10_FULL_PUD_FREEZE", "protected_decision_logit_drift"])
    all_drift = float(mech.loc["A6_PUD_ALL_ADAPT", "protected_decision_logit_drift"])
    drift_reduction = 1.0 - full_drift / max(all_drift, core.EPS)
    adaptive_update = float(mech.loc["A10_FULL_PUD_FREEZE", "adaptive_parameter_update_l2"])
    purity = json.loads((core.PROTOCOL_DIR / "HOLDOUT_RUNTIME_AUDIT.json").read_text(encoding="utf-8"))
    g1 = bool(full_mean - baseline_mean >= 0.0075 - 1e-12)
    g2 = bool(primary_bootstrap["CI95"][0] > 0.0)
    g3 = bool(np.sum(fold_full.Delta_BA_vs_strongest_baseline > 0) >= 4)
    g4 = bool(np.sum(seed_full.Delta_BA_vs_strongest_baseline > 0) >= 2 and primary_delta.mean() > 0)
    g5_tolerance = float(
        core.protocol()["gate_tolerances"]["G5_worst_quartile_noninferiority"]
    )
    g5 = bool(
        full_ntr <= generic_ntr + 1e-12
        and full_worst >= generic_worst - g5_tolerance
    )
    g6_components = {
        method: full_mean - float(means[method])
        for method in ("A7_IDENTITY_PROTECTED", "A8_RANDOM_PROTECTED", "A2_DUAL_CONTROL")
    }
    g6 = bool(all(delta > 0.0 for delta in g6_components.values()))
    g7_delta = full_mean - float(means["A6_PUD_ALL_ADAPT"])
    g7 = bool(g7_delta > 0.0)
    g8 = bool(drift_reduction >= 0.50 and adaptive_update > 0.0)
    g9 = bool(
        purity.get("all_checks_passed") is True
        and not subject.target_future_labels_used_for_fit.astype(bool).any()
        and not subject.internal_holdout_used.astype(bool).any()
        and not subject.outer_test_used.astype(bool).any()
        and all(not row["internal_holdout_accessed"] and not row["outer_test_used"] for row in done)
    )
    gates = {
        "G1_PERFORMANCE": {"pass": g1, "delta_BA": full_mean - baseline_mean, "threshold": 0.0075},
        "G2_UNCERTAINTY": {"pass": g2, "CI95": primary_bootstrap["CI95"], "strict_rule": "lower > 0"},
        "G3_FOLD_CONSISTENCY": {"pass": g3, "positive_folds": int(np.sum(fold_full.Delta_BA_vs_strongest_baseline > 0)), "required": 4},
        "G4_SEED_CONSISTENCY": {"pass": g4, "positive_seeds": int(np.sum(seed_full.Delta_BA_vs_strongest_baseline > 0)), "required": 2},
        "G5_SAFETY": {"pass": g5, "FULL_NTR": full_ntr, "Generic_NTR": generic_ntr, "FULL_worst_quartile": full_worst, "Generic_worst_quartile": generic_worst, "worst_quartile_noninferiority_tolerance": g5_tolerance},
        "G6_THEORY_SPECIFICITY": {"pass": g6, "FULL_minus_controls": g6_components},
        "G7_PROTECTION_NECESSITY": {"pass": g7, "FULL_minus_all_adapt": g7_delta, "preferred_target": 0.003},
        "G8_MECHANISM": {"pass": g8, "protected_drift_reduction": drift_reduction, "FULL_adaptive_update_l2": adaptive_update},
        "G9_NO_FUTURE_LEAKAGE": {"pass": g9, "internal_holdout_accessed": False, "outer_test_used": False},
    }
    openbmi_authorizes_wbcic = bool(g1 and g6 and g9)
    all_gates = bool(all(entry["pass"] for entry in gates.values()))
    terminal = (
        "OPENBMI_GATE_AUTHORIZES_WBCIC_PENDING"
        if openbmi_authorizes_wbcic
        else "PERSIST_NET_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED"
    )

    per_subject_out = per_subject.copy()
    per_subject_out["strongest_baseline_method"] = baseline_method
    matched_baseline_maps: dict[str, dict[str, float]] = {}
    for method in per_subject_out.method.unique():
        matched_seeds = set(
            subject.loc[subject.method.eq(method), "seed"].astype(int).unique().tolist()
        )
        matched = (
            subject.loc[
                subject.method.eq(baseline_method)
                & subject.seed.astype(int).isin(matched_seeds)
            ]
            .groupby("subject_id")
            .BA.mean()
        )
        matched_baseline_maps[str(method)] = {
            str(subject_id): value for subject_id, value in matched.to_dict().items()
        }
    per_subject_out["Delta_BA_vs_strongest_baseline"] = [
        row.BA
        - matched_baseline_maps[str(row.method)].get(str(row.subject_id), np.nan)
        for row in per_subject_out.itertuples()
    ]
    core.write_csv(core.RESULTS / "per_subject_results.csv", per_subject_out)
    core.write_csv(core.RESULTS / "per_fold_results.csv", fold_table)
    core.write_csv(core.RESULTS / "per_seed_results.csv", seed_table)
    core.write_csv(core.RESULTS / "ablations.csv", summary)
    core.write_csv(core.RESULTS / "mechanism_metrics.csv", mechanism)
    core.write_csv(core.RESULTS / "mechanism_per_subject.csv", mechanism_subject)
    core.write_csv(core.RESULTS / "efficiency.csv", efficiency)
    core.write_csv(
        core.RESULTS / "baseline_results.csv",
        summary.loc[summary.method.isin(["B0_VANILLA_EEGNET", "B1_STRONG_EEGNET", "B2_STRONG_GENERIC", "A2_DUAL_CONTROL"])],
    )
    core.write_csv(
        core.RESULTS / "full_results.csv",
        subject.loc[subject.method.eq("A10_FULL_PUD_FREEZE")],
    )
    statistics = {
        "strongest_legal_baseline": baseline_method,
        "FULL_vs_baseline_subject_bootstrap": primary_bootstrap,
        "FULL_vs_baseline_sign_flip": primary_sign_flip,
        "fold_deltas": fold_full[["fold", "Delta_BA_vs_strongest_baseline"]].to_dict(orient="records"),
        "seed_deltas": seed_full[["seed", "Delta_BA_vs_strongest_baseline"]].to_dict(orient="records"),
        "cluster_structure": "three seeds averaged within each subject before the primary 40-subject paired bootstrap",
        "mechanism_subject_bootstrap": mechanism_tests,
        "gates": gates,
    }
    core.write_json(core.RESULTS / "statistics.json", statistics)
    draw_figures(summary, per_subject_out, mechanism)

    pud_vs_p_only = mechanism_tests[
        "FULL_minus_P_only_protected_erase_harm_seed0"
    ]["mean_difference"]
    report = {
        "terminal_state": terminal,
        "openbmi_all_gates_pass": all_gates,
        "openbmi_G1_G6_G9_authorize_wbcic": openbmi_authorizes_wbcic,
        "strongest_legal_baseline": baseline_method,
        "strongest_legal_baseline_BA": baseline_mean,
        "FULL_BA": full_mean,
        "FULL_Macro_F1": float(summary.set_index("method").loc["A10_FULL_PUD_FREEZE", "Macro_F1"]),
        "FULL_delta_vs_strongest_baseline": full_mean - baseline_mean,
        "FULL_delta_CI95": primary_bootstrap["CI95"],
        "positive_folds": int(np.sum(fold_full.Delta_BA_vs_strongest_baseline > 0)),
        "positive_seeds": int(np.sum(seed_full.Delta_BA_vs_strongest_baseline > 0)),
        "FULL_negative_transfer_rate": full_ntr,
        "Generic_negative_transfer_rate": generic_ntr,
        "FULL_minus_dual_control": g6_components["A2_DUAL_CONTROL"],
        "FULL_minus_identity": g6_components["A7_IDENTITY_PROTECTED"],
        "FULL_minus_random": g6_components["A8_RANDOM_PROTECTED"],
        "FULL_minus_all_adapt": g7_delta,
        "protected_drift_reduction_vs_all_adapt": drift_reduction,
        "adaptive_update_nonzero": adaptive_update > 0,
        "PUD_minus_P_only_erasure_harm": pud_vs_p_only,
        "mechanism_subject_bootstrap": mechanism_tests,
        "WBCIC_transfer_result": "PENDING" if openbmi_authorizes_wbcic else "NOT_AUTHORIZED",
        "OpenBMI_internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
        "gates": gates,
        "runtime_finalize_s": time.time() - started,
    }
    core.write_json(core.EXPERIMENT / "FINAL_REPORT.json", report)

    gate_lines = ["# OpenBMI development gate", ""]
    for name, entry in gates.items():
        gate_lines.append(f"- **{name}**: `{'PASS' if entry['pass'] else 'FAIL'}` — `{json.dumps(core.clean(entry), ensure_ascii=False)}`")
    gate_lines.extend(
        [
            "",
            f"WBCIC authorized by frozen G1/G6/G9 rule: **{'YES' if openbmi_authorizes_wbcic else 'NO'}**.",
            f"Current terminal state: **{terminal}**.",
        ]
    )
    (core.EXPERIMENT / "DEVELOPMENT_GATE.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    (core.EXPERIMENT / "EXTERNAL_TRANSFER_GATE.md").write_text(
        "# External transfer gate\n\n"
        + ("OpenBMI G1/G6/G9 passed. WBCIC 41-subject development transfer is authorized and pending.\n" if openbmi_authorizes_wbcic else "OpenBMI G1/G6/G9 did not all pass. WBCIC development and sealed outer are not authorized.\n"),
        encoding="utf-8",
    )
    claim = (
        "The OpenBMI development evidence supports testing source-certified decision-protected persistence on WBCIC; it is not yet an external or outer confirmation."
        if openbmi_authorizes_wbcic
        else "The protection-by-construction hypothesis is not supported under the frozen OpenBMI development gate. Exp1-3 remain descriptive/mechanistic evidence; they do not establish that PUD distillation improves future-session deployment."
    )
    (core.EXPERIMENT / "CLAIM_AUDIT.md").write_text(
        "# Claim audit\n\n" + claim + "\n\nThe OpenBMI internal holdout and WBCIC sealed outer remain untouched. No universal, identity-free, or physiological-causality claim is authorized.\n",
        encoding="utf-8",
    )

    table = summary[["display_name", "BA", "Macro_F1", "Delta_BA_vs_strongest_baseline", "CI95_L", "CI95_U", "Negative_Transfer_Rate", "Worst_Quartile_Adaptation_Delta", "Params", "Target_Trainable_Params"]].copy()
    table.columns = ["Method", "BA", "Macro-F1", "Delta BA", "95% CI L", "95% CI U", "NTR", "Worst-quartile Delta", "Params", "Target-trainable Params"]
    answers = f"""# Final report

## Terminal state

**{terminal}**

Strongest legal baseline: **{DISPLAY[baseline_method]}**, BA={baseline_mean:.4f}.
FULL BA={full_mean:.4f}, Macro-F1={report['FULL_Macro_F1']:.4f}, delta={100*(full_mean-baseline_mean):+.3f} pp,
paired subject-bootstrap 95% CI=[{100*primary_bootstrap['CI95'][0]:+.3f}, {100*primary_bootstrap['CI95'][1]:+.3f}] pp.

## Main table

{markdown_table(table)}

## Required scientific answers

1. **Different from P4-SI / Protection-First / Guard?** Yes architecturally: this is source-time functional distillation into an independent pathway with a universal structural freeze, not GRL, coordinate update projection, or prospective routing.
2. **Did P/U/D train an independent protected task pathway?** Protected functional agreement and nonzero intervention metrics are reported in `results/mechanism_metrics.csv`; gate G8 is **{'PASS' if g8 else 'FAIL'}**. Architectural independence alone is not evidence of benefit.
3. **More task-consequential than identity/random?** Final performance: FULL-minus-identity={100*g6_components['A7_IDENTITY_PROTECTED']:+.3f} pp and FULL-minus-random={100*g6_components['A8_RANDOM_PROTECTED']:+.3f} pp. Protected-branch erasure-harm differences are {100*mechanism_tests['FULL_minus_identity_protected_erase_harm']['mean_difference']:+.3f} pp versus identity and {100*mechanism_tests['FULL_minus_random_protected_erase_harm']['mean_difference']:+.3f} pp versus random; their subject-bootstrap CIs are in `results/statistics.json`. Theory-specificity G6 is **{'PASS' if g6 else 'FAIL'}**.
4. **Is freezing better than all-adapt?** FULL-minus-all-adapt={100*g7_delta:+.3f} pp; G7 is **{'PASS' if g7 else 'FAIL'}**.
5. **Beyond dual-path capacity?** FULL-minus-dual-control={100*g6_components['A2_DUAL_CONTROL']:+.3f} pp. G6 requires this to be positive.
6. **Stable across fold/seed/subject?** Positive folds={int(np.sum(fold_full.Delta_BA_vs_strongest_baseline > 0))}/5; positive seeds={int(np.sum(seed_full.Delta_BA_vs_strongest_baseline > 0))}/3; G2/G3/G4=`{'PASS' if g2 else 'FAIL'}`/`{'PASS' if g3 else 'FAIL'}`/`{'PASS' if g4 else 'FAIL'}`.
7. **Negative transfer?** FULL NTR={full_ntr:.3f}; Strong Generic NTR={generic_ntr:.3f}; safety G5 is **{'PASS' if g5 else 'FAIL'}**.
8. **WBCIC external development replication?** {report['WBCIC_transfer_result']}. It is not run unless frozen OpenBMI G1/G6/G9 authorize it.
9. **Sealed outer untouched?** Yes. OpenBMI internal holdout accessed: **No**. WBCIC sealed outer accessed: **No**.
10. **Strongest legal claim now?** {claim}

## Scope and limitations

Secondary P-only/P+U/P+D/PCA ablations use one predeclared seed; primary gates use five folds by three seeds. The OpenBMI internal holdout is not opened and is not described as independent confirmation. Subject is the statistical unit; no trial-level pseudoreplication is used.
"""
    (core.EXPERIMENT / "FINAL_REPORT.md").write_text(answers, encoding="utf-8")
    training_rows = []
    for row in done:
        training_rows.append(
            f"| {row['fold']} | {row['seed']} | {row['baseline_configuration']} | {row['generic_configuration']} | {row['dual_width']} | {row['PUD_rank']} | {row['runtime_s']:.1f} | No |"
        )
    ledger = "# Training ledger\n\n| Fold | Seed | B1 | Generic | Dual width | PUD rank | Runtime (s) | Future labels fit |\n|---:|---:|---|---|---|---:|---:|---|\n" + "\n".join(training_rows) + "\n"
    (core.EXPERIMENT / "TRAINING_LEDGER.md").write_text(ledger, encoding="utf-8")
    print(json.dumps(core.clean(report), indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Reserved; finalization is deterministic")
    parser.parse_args()
    finalize()


if __name__ == "__main__":
    main()
