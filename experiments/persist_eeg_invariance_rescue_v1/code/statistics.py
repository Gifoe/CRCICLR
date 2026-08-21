from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, OUTPUTS, load_config, stable_seed, write_csv, write_json
from models import primary_pairs, roster
from rescue import determine_eligibility


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def hierarchical_bootstrap(frame: pd.DataFrame, value: str, seed: int, draws: int) -> dict[str, Any]:
    data = frame[["fold", "seed", "subject_id", value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data[np.isfinite(data[value])]
    if not len(data):
        return {"mean": None, "median": None, "ci95": [None, None], "draws": draws, "n": 0}
    folds = sorted(data.fold.unique())
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        fold_sample = rng.choice(folds, size=len(folds), replace=True)
        values = []
        for fold in fold_sample:
            fold_frame = data[data.fold == fold]
            seeds = sorted(fold_frame.seed.unique())
            seed_sample = rng.choice(seeds, size=len(seeds), replace=True)
            for run_seed in seed_sample:
                run = fold_frame[fold_frame.seed == run_seed]
                indices = rng.integers(0, len(run), size=len(run))
                values.append(float(run.iloc[indices][value].mean()))
        sampled_means[draw] = float(np.mean(values))
    raw = data[value].to_numpy(dtype=np.float64)
    return {
        "mean": float(raw.mean()),
        "median": float(np.median(raw)),
        "ci95": [float(np.quantile(sampled_means, 0.025)), float(np.quantile(sampled_means, 0.975))],
        "sign_probability": float(np.mean(sampled_means > 0)),
        "draws": int(draws),
        "n": int(len(raw)),
        "n_folds": int(len(folds)),
        "n_fold_seed_runs": int(data.groupby(["fold", "seed"]).ngroups),
        "hierarchy": "fold -> seed -> subject",
    }


def pooled_family_bootstrap(frame: pd.DataFrame, value: str, seed: int, draws: int) -> dict[str, Any]:
    data = frame[["family", "fold", "seed", "subject_id", value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data[np.isfinite(data[value])]
    if not len(data):
        return {"mean": None, "ci95": [None, None], "draws": draws, "n": 0}
    families = sorted(data.family.unique())
    rng = np.random.default_rng(seed)
    sampled = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        family_sample = rng.choice(families, size=len(families), replace=True)
        family_values = []
        for family in family_sample:
            family_frame = data[data.family == family]
            folds = sorted(family_frame.fold.unique())
            fold_values = []
            for fold in rng.choice(folds, size=len(folds), replace=True):
                fold_frame = family_frame[family_frame.fold == fold]
                seeds = sorted(fold_frame.seed.unique())
                seed_values = []
                for run_seed in rng.choice(seeds, size=len(seeds), replace=True):
                    run = fold_frame[fold_frame.seed == run_seed]
                    picks = rng.integers(0, len(run), size=len(run))
                    seed_values.append(float(run.iloc[picks][value].mean()))
                fold_values.append(float(np.mean(seed_values)))
            family_values.append(float(np.mean(fold_values)))
        sampled[index] = float(np.mean(family_values))
    raw = data[value].to_numpy(dtype=np.float64)
    return {
        "mean": float(raw.mean()),
        "median": float(np.median(raw)),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "sign_probability": float(np.mean(sampled > 0)),
        "draws": int(draws),
        "n": int(len(raw)),
        "n_families": len(families),
        "hierarchy": "family -> fold -> seed -> subject",
    }


def sign_flip_p(values: Sequence[float], alternative: str, seed: int, draws: int = 100_000) -> float | None:
    array = _finite(values)
    if not len(array):
        return None
    if alternative == "less":
        array = -array
    elif alternative != "greater":
        raise ValueError(alternative)
    observed = float(array.mean())
    if len(array) <= 20:
        signs = ((np.arange(2 ** len(array), dtype=np.uint64)[:, None] >> np.arange(len(array), dtype=np.uint64)) & 1)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null = (signs * array[None, :]).mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        counts = 0
        completed = 0
        batch = 10_000
        while completed < int(draws):
            current = min(batch, int(draws) - completed)
            signs = rng.integers(0, 2, size=(current, len(array)), dtype=np.int8) * 2 - 1
            counts += int(np.sum((signs * array[None, :]).mean(axis=1) >= observed - 1e-15))
            completed += current
        return float((counts + 1) / (int(draws) + 1))
    return float((np.sum(null >= observed - 1e-15) + 1) / (len(null) + 1))


def holm_adjust(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((key, value) for key, value in p_values.items() if value is not None), key=lambda item: item[1])
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    total = len(valid)
    for index, (key, value) in enumerate(valid):
        running = max(running, min(1.0, float(value) * (total - index)))
        adjusted[key] = running
    return adjusted


def summarize_delta(frame: pd.DataFrame, value: str, seed: int, draws: int) -> dict[str, Any]:
    boot = hierarchical_bootstrap(frame, value, seed, draws)
    data = frame.copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data[np.isfinite(data[value])]
    unique_subject = data.groupby("subject_id", as_index=False)[value].mean() if len(data) else pd.DataFrame(columns=[value])
    fold_means = data.groupby("fold")[value].mean() if len(data) else pd.Series(dtype=float)
    run_means = data.groupby(["fold", "seed"])[value].mean() if len(data) else pd.Series(dtype=float)
    values = unique_subject[value].to_numpy(dtype=np.float64) if len(unique_subject) else np.asarray([])
    return {
        **boot,
        "positive_subject_fraction": float(np.mean(values > 0)) if len(values) else None,
        "nonnegative_subject_fraction": float(np.mean(values >= 0)) if len(values) else None,
        "worst_subject_delta": float(values.min()) if len(values) else None,
        "n_unique_subjects": int(len(values)),
        "fold_positivity": int(np.sum(fold_means > 0)),
        "n_folds_observed": int(len(fold_means)),
        "seed_run_positivity": int(np.sum(run_means > 0)),
        "n_seed_runs": int(len(run_means)),
    }


def _primary_audit_name(family: str, config: Mapping[str, Any]) -> str:
    if family == "A_SUBJECT_GRL_EEGNET":
        value = int(round(float(config["primary_grl_lambda"]) * 1000))
        return f"A_SUBJECT_GRL_EEGNET_L{value:04d}"
    return family


def audit_statistics(
    audit: pd.DataFrame,
    audit_subjects: pd.DataFrame,
    eligibility: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    draws = int(config["bootstrap_draws"])
    rows, hypothesis_rows = [], []
    raw_p: dict[str, dict[str, float | None]] = {metric: {} for metric in ("delta_ID", "delta_PRS", "delta_BA_INV")}
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for family in primary_pairs(config):
        name = _primary_audit_name(family, config)
        subject_frame = audit_subjects[audit_subjects.family == name]
        for metric in raw_p:
            summary = summarize_delta(subject_frame, metric, stable_seed("audit-stat", family, metric), draws)
            summaries[(family, metric)] = summary
            per_subject = subject_frame.groupby("subject_id")[metric].mean()
            raw_p[metric][family] = sign_flip_p(
                per_subject.values, "less", stable_seed("audit-p", family, metric)
            )
    adjusted = {metric: holm_adjust(values) for metric, values in raw_p.items()}
    for family in primary_pairs(config):
        name = _primary_audit_name(family, config)
        run_frame = audit[audit.family == name]
        entry = eligibility["families"][family]
        row = {
            "family": family,
            "task_only_BA": float(run_frame.task_only_BA.mean()),
            "invariant_BA": float(run_frame.invariant_BA.mean()),
            "delta_BA_INV": float(run_frame.delta_BA_INV.mean()),
            "task_only_subject_probe": float(run_frame.task_only_subject_probe.mean()),
            "invariant_subject_probe": float(run_frame.invariant_subject_probe.mean()),
            "delta_ID": float(run_frame.delta_ID.mean()),
            "task_only_protected_retention": float(run_frame.task_only_PRS.mean()) if run_frame.task_only_PRS.notna().any() else None,
            "invariant_protected_retention": float(run_frame.invariant_PRS.mean()) if run_frame.invariant_PRS.notna().any() else None,
            "delta_PRS": float(run_frame.delta_PRS.mean()) if run_frame.delta_PRS.notna().any() else None,
            "I1": entry["I1"],
            "I2": entry["I2"],
            "I3": entry["I3"],
            "eligibility_status": entry["status"],
            "protected_assignment_runs": int(run_frame.protected_assignment_exists.sum()),
            "runs": len(run_frame),
            "outer_test_used": False,
        }
        for metric in raw_p:
            summary = summaries[(family, metric)]
            row[f"{metric}_CI95_low"] = summary["ci95"][0]
            row[f"{metric}_CI95_high"] = summary["ci95"][1]
            row[f"{metric}_raw_p_less"] = raw_p[metric][family]
            row[f"{metric}_holm_p_less"] = adjusted[metric][family]
            hypothesis_rows.append(
                {
                    "family": family,
                    "hypothesis": metric,
                    "alternative": "less_than_zero",
                    **summary,
                    "raw_p": raw_p[metric][family],
                    "holm_p": adjusted[metric][family],
                    "outer_test_used": False,
                }
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    hypotheses = pd.DataFrame(hypothesis_rows)
    write_csv(OUTPUTS / "TABLE1_INVARIANCE_AUDIT.csv", table)
    write_csv(OUTPUTS / "INVARIANCE_STATISTICS.csv", hypotheses)
    return table, hypotheses


def _collapse_random_run(frame: pd.DataFrame) -> pd.DataFrame:
    random = frame[frame.rescue_method == "R1_RANDOM_RESIDUAL"]
    fixed = frame[frame.rescue_method != "R1_RANDOM_RESIDUAL"]
    if len(random):
        numeric = [column for column in ("balanced_accuracy", "accuracy", "macro_f1", "subject_probe_accuracy") if column in random]
        collapsed = random.groupby(["family", "fold", "seed"], as_index=False)[numeric].mean()
        collapsed["rescue_method"] = "R1_RANDOM_RESIDUAL_MEAN"
        collapsed["random_draw"] = "mean_of_all_draws"
        collapsed["outer_test_used"] = False
        return pd.concat([fixed, collapsed], ignore_index=True, sort=False)
    return fixed


def _collapse_random_subject(frame: pd.DataFrame) -> pd.DataFrame:
    random = frame[frame.rescue_method == "R1_RANDOM_RESIDUAL"]
    fixed = frame[frame.rescue_method != "R1_RANDOM_RESIDUAL"]
    if len(random):
        numeric = [column for column in ("balanced_accuracy", "accuracy", "macro_f1", "subject_probe_accuracy") if column in random]
        collapsed = random.groupby(["family", "fold", "seed", "subject_id"], as_index=False)[numeric].mean()
        collapsed["rescue_method"] = "R1_RANDOM_RESIDUAL_MEAN"
        collapsed["random_draw"] = "mean_of_all_draws"
        collapsed["outer_test_used"] = False
        return pd.concat([fixed, collapsed], ignore_index=True, sort=False)
    return fixed


def rescue_statistics(
    rescue: pd.DataFrame,
    rescue_subjects: pd.DataFrame,
    eligibility: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = load_config()
    draws = int(config["bootstrap_draws"])
    if not len(rescue):
        empty_summary = pd.DataFrame(columns=["family", "rescue_method", "balanced_accuracy", "macro_f1"])
        empty_attr = pd.DataFrame(columns=["family", "comparison", "mean", "ci95_low", "ci95_high", "holm_p"])
        write_csv(OUTPUTS / "TABLE2_RESCUE.csv", empty_summary)
        write_csv(OUTPUTS / "TABLE3_ATTRIBUTION.csv", empty_attr)
        return empty_summary, empty_attr, {}
    run = _collapse_random_run(rescue)
    subject = _collapse_random_subject(rescue_subjects)
    summary = run.groupby(["family", "rescue_method"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"),
        accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        subject_probe_accuracy=("subject_probe_accuracy", "mean"),
        runs=("balanced_accuracy", "size"),
    )
    summary["outer_test_used"] = False

    comparators = {
        "PERSIST_MINUS_INVARIANT": "R0_INVARIANT_ONLY",
        "PERSIST_MINUS_GENERIC": "R2_GENERIC_PERSISTENT_RESIDUAL",
        "PERSIST_MINUS_RANDOM_MEAN": "R1_RANDOM_RESIDUAL_MEAN",
        "PERSIST_MINUS_PCA": "R3_PCA_RESIDUAL",
    }
    attribution_frames: dict[tuple[str, str], pd.DataFrame] = {}
    raw_p: dict[str, dict[str, float | None]] = {comparison: {} for comparison in comparators}
    for family in sorted(subject.family.unique()):
        persist = subject[subject.rescue_method == "R4_PERSIST_PROTECTED_RESIDUAL"]
        persist = persist[persist.family == family]
        for comparison, comparator in comparators.items():
            control = subject[(subject.family == family) & (subject.rescue_method == comparator)]
            merged = persist.merge(control, on=["family", "fold", "seed", "subject_id"], suffixes=("_persist", "_control"))
            merged["delta"] = merged.balanced_accuracy_persist - merged.balanced_accuracy_control
            attribution_frames[(family, comparison)] = merged
            unique = merged.groupby("subject_id").delta.mean()
            raw_p[comparison][family] = sign_flip_p(
                unique.values, "greater", stable_seed("rescue-p", family, comparison)
            )
    adjusted = {comparison: holm_adjust(values) for comparison, values in raw_p.items()}
    attribution_rows = []
    for (family, comparison), frame in attribution_frames.items():
        stats = summarize_delta(frame, "delta", stable_seed("rescue-stat", family, comparison), draws)
        attribution_rows.append(
            {
                "family": family,
                "comparison": comparison,
                **stats,
                "ci95_low": stats["ci95"][0],
                "ci95_high": stats["ci95"][1],
                "raw_p": raw_p[comparison][family],
                "holm_p": adjusted[comparison][family],
                "outer_test_used": False,
            }
        )
    attribution = pd.DataFrame(attribution_rows)

    family_decisions: dict[str, Any] = {}
    for family in sorted(subject.family.unique()):
        family_summary = summary[summary.family == family].set_index("rescue_method")
        task = float(family_summary.loc["R5_TASK_ONLY_UPPER_REFERENCE", "balanced_accuracy"])
        invariant = float(family_summary.loc["R0_INVARIANT_ONLY", "balanced_accuracy"])
        lost = task - invariant
        ratios = {}
        for label, method in {
            "PERSIST": "R4_PERSIST_PROTECTED_RESIDUAL",
            "GENERIC": "R2_GENERIC_PERSISTENT_RESIDUAL",
            "RANDOM": "R1_RANDOM_RESIDUAL_MEAN",
            "PCA": "R3_PCA_RESIDUAL",
        }.items():
            score = float(family_summary.loc[method, "balanced_accuracy"])
            ratios[label] = (score - invariant) / lost if lost > 0 else None
        persist_inv = attribution[(attribution.family == family) & (attribution.comparison == "PERSIST_MINUS_INVARIANT")].iloc[0]
        persist_generic = attribution[(attribution.family == family) & (attribution.comparison == "PERSIST_MINUS_GENERIC")].iloc[0]
        minimum = bool(persist_inv["mean"] > 0)
        certified = bool(persist_inv.ci95_low > 0 and persist_generic.ci95_low > 0)
        family_decisions[family] = {
            "minimum_support": minimum,
            "certified_support": certified,
            "strong_rescue": bool(certified and ratios["PERSIST"] is not None and ratios["PERSIST"] >= 0.5),
            "lost_performance": lost,
            "recovery_ratios": ratios,
            "persist_minus_invariant_CI95": [float(persist_inv.ci95_low), float(persist_inv.ci95_high)],
            "persist_minus_generic_CI95": [float(persist_generic.ci95_low), float(persist_generic.ci95_high)],
        }
    write_csv(OUTPUTS / "TABLE2_RESCUE.csv", summary)
    write_csv(OUTPUTS / "TABLE3_ATTRIBUTION.csv", attribution)
    return summary, attribution, family_decisions


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown(frame: pd.DataFrame) -> str:
    if not len(frame):
        return "(empty)"
    columns = list(frame.columns)
    def value(item: Any) -> str:
        if isinstance(item, (float, np.floating)):
            return "NA" if not np.isfinite(item) else f"{float(item):.4f}"
        return str(item).replace("|", "\\|")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(value(item) for item in record) + " |" for record in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _make_figures(
    audit: pd.DataFrame,
    audit_subjects: pd.DataFrame,
    rescue: pd.DataFrame,
    rescue_subjects: pd.DataFrame,
    decisions: Mapping[str, Any],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    config = load_config()
    a = audit[audit.family.str.startswith("A_SUBJECT_GRL_EEGNET_L")].copy()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    if len(a):
        grouped = a.groupby("family", as_index=False).agg(
            identity=("invariant_subject_probe", "mean"), ba=("invariant_BA", "mean"), prs=("invariant_PRS", "mean")
        )
        baseline = a.groupby(["fold", "seed"], as_index=False).first()
        ax.scatter(baseline.task_only_subject_probe.mean(), baseline.task_only_BA.mean(), marker="s", s=70, label="task-only")
        for _, row in grouped.sort_values("family").iterrows():
            label = row.family.rsplit("L", 1)[-1]
            ax.scatter(row.identity, row.ba, s=55)
            ax.annotate(f"lambda={int(label)/1000:g}\nPRS={row.prs:.2f}" if np.isfinite(row.prs) else f"lambda={int(label)/1000:g}", (row.identity, row.ba), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Cross-session subject probe balanced accuracy")
    ax.set_ylabel("Development outcome task BA")
    ax.set_title("A. Controlled GRL invariance ladder")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "FIGURE_A_GRL_LADDER.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    primary_names = {_primary_audit_name(family, config): family for family in primary_pairs(config)}
    shown = audit[audit.family.isin(primary_names)]
    for name, group in shown.groupby("family"):
        ax.scatter(group.delta_PRS, group.delta_BA_INV, label=primary_names.get(name, name), alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta protected retention (invariant - task-only)")
    ax.set_ylabel("Delta task BA (invariant - task-only)")
    ax.set_title("B. Protected retention versus task outcome")
    ax.grid(alpha=0.25)
    if len(shown):
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "FIGURE_B_RETENTION_VS_BA.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if decisions:
        labels, values = [], []
        for family, decision in decisions.items():
            for method in ("RANDOM", "GENERIC", "PCA", "PERSIST"):
                labels.append(f"{family.split('_')[0]}\n{method}")
                value = decision["recovery_ratios"].get(method)
                values.append(np.nan if value is None else value)
        ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(values)), labels, fontsize=7)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="strong descriptor")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No eligible protected-loss family", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylabel("Recovery ratio")
    ax.set_title("C. Rank-matched recovery ratios")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "FIGURE_C_RECOVERY_RATIO.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if len(rescue_subjects):
        collapsed = _collapse_random_subject(rescue_subjects)
        persist = collapsed[collapsed.rescue_method == "R4_PERSIST_PROTECTED_RESIDUAL"]
        generic = collapsed[collapsed.rescue_method == "R2_GENERIC_PERSISTENT_RESIDUAL"]
        merged = persist.merge(generic, on=["family", "fold", "seed", "subject_id"], suffixes=("_p", "_g"))
        merged["delta"] = merged.balanced_accuracy_p - merged.balanced_accuracy_g
        values = merged.groupby(["family", "subject_id"], as_index=False).delta.mean().sort_values(["family", "delta"])
        colors = {family: color for family, color in zip(sorted(values.family.unique()), plt.cm.tab10.colors)}
        ax.bar(np.arange(len(values)), values.delta, color=[colors[family] for family in values.family])
        ax.set_xticks(np.arange(len(values)), [f"{row.family.split('_')[0]}:{row.subject_id}" for _, row in values.iterrows()], rotation=90, fontsize=6)
        ax.axhline(0, color="black", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "No eligible rescue comparison", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    ax.set_ylabel("PERSIST - generic BA")
    ax.set_title("D. Per-subject selective-rescue attribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "FIGURE_D_SUBJECT_PERSIST_VS_GENERIC.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    if len(rescue):
        collapsed = _collapse_random_run(rescue)
        methods = [
            "R0_INVARIANT_ONLY", "R1_RANDOM_RESIDUAL_MEAN", "R2_GENERIC_PERSISTENT_RESIDUAL",
            "R4_PERSIST_PROTECTED_RESIDUAL", "R5_TASK_ONLY_UPPER_REFERENCE"
        ]
        for method in methods:
            frame = collapsed[collapsed.rescue_method == method]
            if len(frame):
                ax.scatter(frame.subject_probe_accuracy.mean(), frame.balanced_accuracy.mean(), label=method.replace("_RESIDUAL", ""), s=55)
    else:
        ax.text(0.5, 0.5, "No eligible rescue comparison", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Cross-session identity probe accuracy")
    ax.set_ylabel("Task BA")
    ax.set_title("E. Identity recovery versus task recovery")
    ax.grid(alpha=0.25)
    if len(rescue):
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "FIGURE_E_SELECTIVITY.png", dpi=200)
    plt.close(fig)


def finalize() -> dict[str, Any]:
    config = load_config()
    audit = pd.read_csv(OUTPUTS / "INVARIANCE_AUDIT.csv")
    audit_subjects = pd.read_csv(OUTPUTS / "SUBJECT_LEVEL_AUDIT.csv")
    rescue = pd.read_csv(OUTPUTS / "RESCUE_RESULTS.csv")
    rescue_subjects = pd.read_csv(OUTPUTS / "RESCUE_SUBJECT_RESULTS.csv")
    eligibility = determine_eligibility(audit)
    table1, audit_hypotheses = audit_statistics(audit, audit_subjects, eligibility)
    table2, table3, family_decisions = rescue_statistics(rescue, rescue_subjects, eligibility)

    certified = [family for family, value in family_decisions.items() if value["certified_support"]]
    eligible = [family for family, value in eligibility["families"].items() if value["eligible"]]
    pooled = None
    if len(rescue_subjects) and eligible:
        subject = _collapse_random_subject(rescue_subjects)
        persist = subject[subject.rescue_method == "R4_PERSIST_PROTECTED_RESIDUAL"]
        generic = subject[subject.rescue_method == "R2_GENERIC_PERSISTENT_RESIDUAL"]
        merged = persist.merge(generic, on=["family", "fold", "seed", "subject_id"], suffixes=("_p", "_g"))
        merged["delta"] = merged.balanced_accuracy_p - merged.balanced_accuracy_g
        pooled = pooled_family_bootstrap(
            merged, "delta", stable_seed("pooled-persist-generic"), int(config["bootstrap_draws"])
        )
    if not eligible:
        terminal = "NO_ELIGIBLE_PROTECTED_LOSS_OBSERVED"
    elif len(certified) >= 2 and pooled is not None and pooled["ci95"][0] > 0:
        terminal = "PERSIST_RESCUE_CROSS_FAMILY_SUPPORTED"
    elif len(certified) == 1:
        terminal = "PERSIST_RESCUE_SINGLE_FAMILY_ONLY"
    else:
        terminal = "PERSIST_RESCUE_NOT_SUPPORTED"

    decision = {
        "terminal_scientific_state": terminal,
        "development_exploratory": True,
        "families": eligibility["families"],
        "eligible_families": eligible,
        "certified_rescue_families": certified,
        "family_rescue_decisions": family_decisions,
        "pooled_persist_minus_generic": pooled,
        "cross_family_supported": terminal == "PERSIST_RESCUE_CROSS_FAMILY_SUPPORTED",
        "outer_test_used": False,
        "outer_split_field_read": False,
        "outcome_used_for_protected_assignment": False,
        "outcome_used_for_rescue_selection": False,
        "scientific_gates_changed_after_freeze": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", decision)

    audit_copy = audit_subjects.copy()
    audit_copy.insert(0, "result_type", "invariance_audit")
    rescue_copy = rescue_subjects.copy()
    rescue_copy.insert(0, "result_type", "rescue")
    write_csv(OUTPUTS / "SUBJECT_LEVEL_RESULTS.csv", pd.concat([audit_copy, rescue_copy], ignore_index=True, sort=False))

    result_ledger = table1[["family", "eligibility_status", "delta_ID", "delta_PRS", "delta_BA_INV"]].copy()
    result_ledger["terminal_scientific_state"] = terminal
    result_ledger["outer_test_used"] = False
    write_csv(OUTPUTS / "RESULT_LEDGER.csv", result_ledger)

    method_rows = []
    for method in roster(config):
        family = method[0]
        method_rows.append(
            {
                "method_id": method,
                "family": family,
                "role": "task_only" if method.startswith(("A0", "B0", "C0")) else "invariant_or_ladder",
                "implementation": "local" if family == "A" else "clean_room_method_level",
                "upstream_source_copied": False,
                "exclusion": False,
                "outer_test_used": False,
            }
        )
    write_csv(OUTPUTS / "METHOD_LEDGER.csv", pd.DataFrame(method_rows))
    _make_figures(audit, audit_subjects, rescue, rescue_subjects, family_decisions)

    table1_md = _markdown(table1)
    table2_md = _markdown(table2) if len(table2) else "No eligible family; rescue table is empty by protocol."
    table3_md = _markdown(table3[["family", "comparison", "mean", "ci95_low", "ci95_high", "holm_p"]]) if len(table3) else "No eligible family; attribution table is empty by protocol."
    q1 = [family for family, value in eligibility["families"].items() if value["I1"]]
    q2 = [family for family, value in eligibility["families"].items() if value["I1"] and value["I2"]]
    q3 = eligible
    strongest_counterexample = max(
        eligibility["families"].items(),
        key=lambda item: (item[1]["mean_delta_BA_INV"] if item[1]["mean_delta_BA_INV"] is not None else -math.inf),
    )[0]
    report = f"""# Experiment 1 scientific report

Status: `{terminal}`. All results are **DEVELOPMENT / EXPLORATORY**.

## Direct answers

1. Methods with measurable mean subject-information reduction: `{q1}`.
2. Methods also showing mean protected-retention loss: `{q2}`.
3. Families where protected loss accompanied future-session task harm: `{q3}`.
4. PERSIST rescue certified for: `{certified}`.
5. Capacity-matched generic persistence was beaten with certified paired CIs for: `{certified}`; see Table 3 for the actual intervals.
6. Selectivity is reported, not assumed: Figure E and Table 2 compare identity recovery against BA recovery. Identity returning below the task-only level is descriptive, not a hard gate.
7. Evidence spans `{len(config['development_folds'])}` folds, `{len(config['seeds'])}` seeds, and the subject counts in the statistics tables. Cross-family support: `{decision['cross_family_supported']}`.
8. Strongest counterexample to a blanket invariance-harm claim: `{strongest_counterexample}`. Family-specific statuses are binding.
9. Outer test used: `false`.
10. Terminal state: `{terminal}`.

Protected-retention task-only R2 is expected to approach one because its target is a frozen linear coordinate of the same task-only representation. The evidential quantity is the cross-subject, cross-session recoverability change in the independently trained invariant representation, not task-only self-reconstruction in isolation.

## Table 1 — Invariance audit

{table1_md}

## Table 2 — Rescue

{table2_md}

## Table 3 — Attribution

{table3_md}

## Interpretation boundary

This experiment does not establish that subject invariance is generally wrong, does not test absolute SOTA, and does not authorize an outer-test claim. It only tests whether the preregistered independently trained invariance objectives exhibit the full identity-loss/protected-loss/task-harm chain and, if so, whether intervention-defined protected restoration beats matched controls.
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")

    claim_audit = f"""# Claim audit

- Terminal claim: `{terminal}`.
- `outer_test_used=false`.
- No outer split field was accessed and no outer subject, label, signal, feature, or score was enumerated by this experiment.
- Protected assignment used model-fit subjects only.
- Rescue hyperparameters used calibration subjects Session 2 only.
- Development outcome labels were used only for final task scoring; outcome Session 1/2 identity labels were used only for the preregistered cross-session probe.
- B/C are clean-room method-level reproductions, not exact official-source reproductions; conclusions are limited to these audited instantiations.
- Task-only PRS is a self-coordinate recoverability reference and is not independent evidence.
- No V6/V7/V8 generic adaptation, router, Conformer blend, action bank, or meta-selector was used.
"""
    (OUTPUTS.parent / "CLAIM_AUDIT.md").write_text(claim_audit, encoding="utf-8")
    return decision
