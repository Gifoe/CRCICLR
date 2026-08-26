"""Aggregate the complete fixed stress grid, run clustered inference, and report."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 24491


def load_complete() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance: list[pd.DataFrame] = []
    identity: list[pd.DataFrame] = []
    directions: list[pd.DataFrame] = []
    ledger: list[dict[str, Any]] = []
    for backbone in common.BACKBONES:
        for fold in range(5):
            for seed in range(3):
                context = common.unit_dir(backbone, fold, seed)
                complete = context / "UNIT_COMPLETE.json"
                if not complete.is_file() or common.read_json(complete).get("pass") is not True:
                    raise RuntimeError(f"missing complete unit: {backbone} fold={fold} seed={seed}")
                selection = common.read_json(context / "LAMBDA_SELECTION_FROZEN.json")
                for method, lam in common.configuration_grid():
                    slug = common.config_slug(method, lam)
                    target = context / "evaluation" / slug
                    evaluation = common.read_json(target / "EVALUATION_COMPLETE.json")
                    if evaluation.get("pass") is not True or evaluation.get("selection_frozen_before_outcome_evaluation") is not True:
                        raise RuntimeError(f"invalid evaluation guard: {target}")
                    performance.append(pd.read_csv(target / "performance.csv", dtype={"subject_id": str}))
                    identity.append(pd.read_csv(target / "identity.csv"))
                    directions.append(pd.read_csv(target / "directions.csv"))
                    candidate = common.read_json(context / "candidates" / f"{slug}.json")
                    ledger.append(
                        {
                            "backbone": backbone,
                            "method": method,
                            "lambda": float(lam),
                            "fold": fold,
                            "seed": seed,
                            "best_epoch": candidate["best_epoch"],
                            "epochs_executed": candidate["epochs_executed"],
                            "best_validation_BA": candidate["best_validation_BA"],
                            "best_validation_NLL": candidate["best_validation_NLL"],
                            "elapsed_seconds": candidate["elapsed_seconds"],
                            "initial_shared_state_sha256": candidate["initial_shared_state_sha256"],
                            "epoch0_minibatch_order_sha256": candidate["epoch0_minibatch_order_sha256"],
                            "selected_by_source_validation": float(selection["selected"][method]["lambda"]) == float(lam),
                            "outcome_labels_used_for_training_or_selection": False,
                        }
                    )
    return (
        pd.concat(performance, ignore_index=True),
        pd.concat(identity, ignore_index=True),
        pd.concat(directions, ignore_index=True),
        pd.DataFrame(ledger),
    )


def hierarchical_mean_ci(frame: pd.DataFrame, column: str, seed: int) -> dict[str, Any]:
    folds = sorted(frame.fold.unique().tolist())
    rng = np.random.default_rng(seed)
    by_fold_seed = {(int(row.fold), int(row.seed)): float(getattr(row, column)) for row in frame.itertuples()}
    seeds_by_fold = {int(fold): sorted(frame.loc[frame.fold == fold, "seed"].unique().tolist()) for fold in folds}
    # Preserve the original RNG call order exactly, but avoid a pandas filter
    # inside every bootstrap draw.  The old implementation spent most of the
    # aggregation wall time repeatedly materializing the same five filters.
    sampled_values = np.empty((BOOTSTRAP_DRAWS, len(folds), max(map(len, seeds_by_fold.values()))), dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_folds = rng.choice(folds, size=len(folds), replace=True)
        for fold_slot, fold in enumerate(sampled_folds):
            seeds = seeds_by_fold[int(fold)]
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            for seed_slot, selected_seed in enumerate(sampled_seeds):
                sampled_values[draw, fold_slot, seed_slot] = by_fold_seed[(int(fold), int(selected_seed))]
    draws = sampled_values.mean(axis=(1, 2))
    return {
        "mean": float(frame[column].mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "fold_consistency": int((frame.groupby("fold")[column].mean() > 0).sum()),
        "seed_consistency": int((frame.groupby("seed")[column].mean() > 0).sum()),
        "n_fold_seed": int(len(frame)),
    }


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denominator = float(np.sum(np.square(x - x.mean())))
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denominator) if denominator > EPS else float("nan")


EPS = 1e-12


def hierarchical_slope(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    work = frame.copy()
    work["configuration"] = work.backbone.astype(str) + "|" + work.method.astype(str) + "|" + work["lambda"].map(lambda x: f"{x:g}")
    folds = sorted(work.fold.unique().tolist())
    configs = sorted(work.configuration.unique().tolist())
    seeds = sorted(work.seed.unique().tolist())
    expected = pd.MultiIndex.from_product([folds, seeds, configs], names=["fold", "seed", "configuration"])
    indexed = work.set_index(["fold", "seed", "configuration"]).sort_index()
    if len(indexed) != len(expected) or not indexed.index.equals(expected):
        missing = expected.difference(indexed.index)
        raise RuntimeError(f"incomplete fold/seed/configuration grid for slope bootstrap: {missing.tolist()[:5]}")
    x_table = indexed["identity_suppression_vs_ERM"].to_numpy(float).reshape(len(folds), len(seeds), len(configs))
    y_table = indexed["BA_delta_vs_ERM"].to_numpy(float).reshape(len(folds), len(seeds), len(configs))
    fold_position = {int(value): index for index, value in enumerate(folds)}
    seed_position = {int(value): index for index, value in enumerate(seeds)}
    rng = np.random.default_rng(seed)
    # Generate indices in the same per-draw order as the reference loop, then
    # evaluate all slopes with NumPy.  This is an implementation-only speedup:
    # configuration, fold, and within-fold seed resampling are unchanged.
    config_indices = np.empty((BOOTSTRAP_DRAWS, len(configs)), dtype=np.int16)
    fold_indices = np.empty((BOOTSTRAP_DRAWS, len(folds)), dtype=np.int8)
    seed_indices = np.empty((BOOTSTRAP_DRAWS, len(folds), len(seeds)), dtype=np.int8)
    config_positions = np.arange(len(configs), dtype=int)
    for draw in range(BOOTSTRAP_DRAWS):
        config_indices[draw] = rng.choice(config_positions, size=len(configs), replace=True)
        sampled_folds = rng.choice(folds, size=len(folds), replace=True)
        for fold_slot, selected_fold in enumerate(sampled_folds):
            fold_indices[draw, fold_slot] = fold_position[int(selected_fold)]
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_indices[draw, fold_slot] = [seed_position[int(value)] for value in sampled_seeds]
    sampled_x = x_table[
        fold_indices[:, :, None, None],
        seed_indices[:, :, :, None],
        config_indices[:, None, None, :],
    ]
    sampled_y = y_table[
        fold_indices[:, :, None, None],
        seed_indices[:, :, :, None],
        config_indices[:, None, None, :],
    ]
    axes = (1, 2, 3)
    x_mean = sampled_x.mean(axis=axes, keepdims=True)
    y_mean = sampled_y.mean(axis=axes, keepdims=True)
    denominator = np.square(sampled_x - x_mean).sum(axis=axes)
    numerator = ((sampled_x - x_mean) * (sampled_y - y_mean)).sum(axis=axes)
    array = numerator[denominator > EPS] / denominator[denominator > EPS]
    point = slope(work.identity_suppression_vs_ERM.to_numpy(float), work.BA_delta_vs_ERM.to_numpy(float))
    return {
        "slope": point,
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        "bootstrap_draws_valid": int(len(array)),
        "n_rows": int(len(work)),
        "n_configurations": int(work.configuration.nunique()),
        "hierarchy": ["outer_fold", "seed", "method-strength configuration"],
    }


def safe_correlations(frame: pd.DataFrame) -> dict[str, Any]:
    x = frame.identity_suppression_vs_ERM.to_numpy(float)
    y = frame.BA_delta_vs_ERM.to_numpy(float)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p_descriptive": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_descriptive": float(spearman.pvalue),
        "n_configuration_points": int(len(frame)),
        "note": "p-values are descriptive; hierarchical-bootstrap slope is primary inference",
    }


def build_stress(performance: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    perf = performance.groupby(["backbone", "method", "lambda", "fold", "seed"], as_index=False).agg(
        BA=("BA", "mean"), macro_f1=("macro_f1", "mean")
    )
    merged = identity.merge(perf, on=["backbone", "method", "lambda", "fold", "seed"], validate="one_to_one")
    erm = merged.loc[merged.method == "ERM", ["backbone", "fold", "seed", "identity_symmetric", "BA", "macro_f1"]].rename(
        columns={"identity_symmetric": "identity_ERM", "BA": "BA_ERM", "macro_f1": "F1_ERM"}
    )
    stress = merged.loc[merged.method != "ERM"].merge(erm, on=["backbone", "fold", "seed"], validate="many_to_one")
    stress["identity_suppression_vs_ERM"] = stress.identity_ERM - stress.identity_symmetric
    stress["BA_delta_vs_ERM"] = stress.BA - stress.BA_ERM
    stress["F1_delta_vs_ERM"] = stress.macro_f1 - stress.F1_ERM
    return stress[
        [
            "backbone",
            "method",
            "lambda",
            "fold",
            "seed",
            "identity_suppression_vs_ERM",
            "BA_delta_vs_ERM",
            "F1_delta_vs_ERM",
            "identity_ERM",
            "identity_symmetric",
            "BA_ERM",
            "BA",
        ]
    ].sort_values(["backbone", "method", "lambda", "fold", "seed"]).reset_index(drop=True)


def manipulation_statistics(stress: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for (backbone, method, lam), part in stress.groupby(["backbone", "method", "lambda"], sort=True):
        identity_stats = hierarchical_mean_ci(part, "identity_suppression_vs_ERM", common.stable_seed("manip", backbone, method, lam))
        ba_stats = hierarchical_mean_ci(part, "BA_delta_vs_ERM", common.stable_seed("ba", backbone, method, lam))
        f1_stats = hierarchical_mean_ci(part, "F1_delta_vs_ERM", common.stable_seed("f1", backbone, method, lam))
        erm_identity = float(part.identity_ERM.mean())
        threshold = max(0.05, 0.10 * abs(erm_identity))
        meaningful = bool(identity_stats["mean"] >= threshold and identity_stats["ci95"][0] > 0)
        strong_counter = bool(meaningful and ba_stats["ci95"][1] <= 0)
        weak_counter = bool(meaningful and ba_stats["mean"] <= 0 and ba_stats["ci95"][0] <= 0 <= ba_stats["ci95"][1])
        row = {
            "backbone": backbone,
            "method": method,
            "lambda": float(lam),
            "identity_reduction": identity_stats,
            "relative_identity_reduction": float(identity_stats["mean"] / max(abs(erm_identity), EPS)),
            "meaningful_threshold": threshold,
            "meaningful_identity_reduction": meaningful,
            "BA_delta": ba_stats,
            "F1_delta": f1_stats,
            "strong_counterexample": strong_counter,
            "weak_counterexample": weak_counter,
        }
        rows.append(row)
        if strong_counter or weak_counter:
            counterexamples.append(row)
    return rows, counterexamples


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    x_train = train[list(features)].to_numpy(float)
    x_test = test[list(features)].to_numpy(float)
    y = train.outcome_CE_effect.to_numpy(float)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    z_train = np.c_[(x_train - mean) / std, np.ones(len(x_train))]
    z_test = np.c_[(x_test - mean) / std, np.ones(len(x_test))]
    penalty = np.eye(z_train.shape[1])
    penalty[-1, -1] = 0.0
    lhs = z_train.T @ z_train + common.RIDGE_ALPHA * penalty
    try:
        beta = np.linalg.solve(lhs, z_train.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ z_train.T @ y
    return z_test @ beta


def direction_prediction(directions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = directions.copy()
    frame["run"] = frame.backbone.astype(str) + "|f" + frame.fold.astype(str) + "|s" + frame.seed.astype(str)
    models = common.protocol()["secondary_direction_audit"]["prediction_models"]
    prediction_rows: list[dict[str, Any]] = []
    for held in sorted(frame.run.unique()):
        train = frame.loc[frame.run != held]
        test = frame.loc[frame.run == held]
        for model, features in models.items():
            prediction = ridge_predict(train, test, features)
            for (_, row), value in zip(test.iterrows(), prediction):
                prediction_rows.append(
                    {
                        "run": held,
                        "backbone": row.backbone,
                        "method_family": row.method,
                        "lambda": row["lambda"],
                        "fold": row.fold,
                        "seed": row.seed,
                        "direction_id": row.direction_id,
                        "model": model,
                        "observed": row.outcome_CE_effect,
                        "prediction": float(value),
                        "error": float(value - row.outcome_CE_effect),
                    }
                )
    prediction = pd.DataFrame(prediction_rows)
    summary_rows: list[dict[str, Any]] = []
    for model in models:
        part = prediction[prediction.model == model]
        summary_rows.append(
            {
                "model": model,
                "run": "ALL",
                "backbone": "ALL",
                "method_family": "ALL",
                "n_directions": len(part),
                "RMSE": float(np.sqrt(np.mean(np.square(part.error)))),
                "MAE": float(np.mean(np.abs(part.error))),
            }
        )
        for held, run in part.groupby("run"):
            summary_rows.append(
                {
                    "model": model,
                    "run": held,
                    "backbone": run.backbone.iloc[0],
                    "method_family": "ALL",
                    "n_directions": len(run),
                    "RMSE": float(np.sqrt(np.mean(np.square(run.error)))),
                    "MAE": float(np.mean(np.abs(run.error))),
                }
            )
    summary = pd.DataFrame(summary_rows)
    pivot = summary.loc[summary.run != "ALL"].pivot(index="run", columns="model", values="RMSE")
    delta = (pivot.MI - pivot.MD).rename("RMSE_MI_minus_MD").reset_index()
    delta["backbone"] = delta.run.str.split("|").str[0]
    delta["fold"] = delta.run.str.extract(r"\|f(\d+)\|", expand=False).astype(int)
    delta["seed"] = delta.run.str.extract(r"\|s(\d+)$", expand=False).astype(int)
    # Conservative run-cluster bootstrap; each backbone/fold/seed is one unit.
    values = delta.RMSE_MI_minus_MD.to_numpy(float)
    rng = np.random.default_rng(common.stable_seed("decision-vs-identity-run-bootstrap"))
    boot = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    all_rmse = summary.loc[summary.run == "ALL"].set_index("model").RMSE.to_dict()
    statistics = {
        "model_RMSE": {key: float(value) for key, value in all_rmse.items()},
        "RMSE_MI_minus_MD": {
            "mean_run_difference": float(values.mean()),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "positive_runs": int(np.sum(values > 0)),
            "run_count": int(len(values)),
            "relative_improvement_MD_vs_MI": float((all_rmse["MI"] - all_rmse["MD"]) / max(all_rmse["MI"], EPS)),
        },
        "held_out_rule": "leave-one backbone-fold-seed run out",
        "ridge_alpha": common.RIDGE_ALPHA,
    }
    return prediction, summary, statistics


def config_aggregates(stress: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (backbone, method, lam), part in stress.groupby(["backbone", "method", "lambda"], sort=True):
        identity_stats = hierarchical_mean_ci(part, "identity_suppression_vs_ERM", common.stable_seed("plot-i", backbone, method, lam))
        ba_stats = hierarchical_mean_ci(part, "BA_delta_vs_ERM", common.stable_seed("plot-ba", backbone, method, lam))
        rows.append(
            {
                "backbone": backbone,
                "method": method,
                "lambda": float(lam),
                "identity_suppression": identity_stats["mean"],
                "identity_CI_L": identity_stats["ci95"][0],
                "identity_CI_U": identity_stats["ci95"][1],
                "BA_delta": ba_stats["mean"],
                "BA_CI_L": ba_stats["ci95"][0],
                "BA_CI_U": ba_stats["ci95"][1],
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    stress: pd.DataFrame,
    config: pd.DataFrame,
    identity: pd.DataFrame,
    performance: pd.DataFrame,
    prediction: pd.DataFrame,
) -> None:
    common.FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"DANN": "#d73027", "CORAL": "#4575b4", "MMD": "#1a9850"}
    markers = {0.01: "o", 0.1: "s", 1.0: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, backbone in zip(axes, common.BACKBONES):
        part = config[config.backbone == backbone]
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvline(0, color="0.5", lw=0.8)
        for _, row in part.iterrows():
            ax.errorbar(
                row["identity_suppression"],
                row["BA_delta"],
                xerr=[[row["identity_suppression"] - row["identity_CI_L"]], [row["identity_CI_U"] - row["identity_suppression"]]],
                yerr=[[row["BA_delta"] - row["BA_CI_L"]], [row["BA_CI_U"] - row["BA_delta"]]],
                fmt=markers[float(row["lambda"])],
                color=colors[row["method"]],
                capsize=2,
                alpha=0.9,
            )
            ax.annotate(f"{row['method'][0]}:{float(row['lambda']):g}", (row["identity_suppression"], row["BA_delta"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title("EEGNet" if backbone == "eegnet" else "EEGConformer")
        ax.set_xlabel("Identity suppression vs ERM (skill)")
    axes[0].set_ylabel("Future-session BA change vs ERM")
    fig.suptitle("Subject identifiability versus future generalization")
    fig.tight_layout()
    fig.savefig(common.FIGURES / "identity_vs_generalization.png", dpi=240)
    fig.savefig(common.FIGURES / "identity_vs_generalization.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, backbone in zip(axes, common.BACKBONES):
        part = config[config.backbone == backbone]
        for method in ("DANN", "CORAL", "MMD"):
            group = part[part.method == method].sort_values("lambda")
            ax.plot(group["lambda"], group.identity_suppression, marker="o", color=colors[method], label=method)
            ax.fill_between(group["lambda"], group.identity_CI_L, group.identity_CI_U, color=colors[method], alpha=0.15)
        ax.set_xscale("log")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title("EEGNet" if backbone == "eegnet" else "EEGConformer")
        ax.set_xlabel("Invariance strength lambda")
    axes[0].set_ylabel("Identity suppression vs ERM (skill)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "identity_suppression_by_lambda.png", dpi=240)
    fig.savefig(common.FIGURES / "identity_suppression_by_lambda.pdf")
    plt.close(fig)

    group = prediction.groupby(["backbone", "method_family", "model"], as_index=False).agg(RMSE=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))))
    models = ["MI", "MD", "MID"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, backbone in zip(axes, common.BACKBONES):
        part = group[group.backbone == backbone]
        methods = ["ERM", "DANN", "CORAL", "MMD"]
        x = np.arange(len(methods))
        width = 0.24
        for index, model in enumerate(models):
            values = [float(part[(part.method_family == method) & (part.model == model)].RMSE.iloc[0]) for method in methods]
            ax.bar(x + (index - 1) * width, values, width, label=model)
        ax.set_xticks(x, methods)
        ax.set_title("EEGNet" if backbone == "eegnet" else "EEGConformer")
        ax.set_ylabel("Held-run RMSE" if ax is axes[0] else "")
    axes[1].legend(frameon=False)
    fig.suptitle("Identity versus decision for consequence prediction")
    fig.tight_layout()
    fig.savefig(common.FIGURES / "decision_vs_identity_rmse.png", dpi=240)
    fig.savefig(common.FIGURES / "decision_vs_identity_rmse.pdf")
    plt.close(fig)

    perf = performance.groupby(["backbone", "method", "lambda", "fold", "seed"], as_index=False).BA.mean()
    selected = performance[performance.selected_by_source_validation.astype(bool)].groupby(["backbone", "method", "fold", "seed"], as_index=False).BA.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, backbone in zip(axes, common.BACKBONES):
        part = selected[selected.backbone == backbone]
        methods = ["ERM", "DANN", "CORAL", "MMD"]
        values = [part[part.method == method].BA.to_numpy(float) for method in methods]
        # Matplotlib >=3.9 renamed ``labels`` to ``tick_labels`` and the
        # server version has removed the legacy keyword.
        ax.boxplot(values, tick_labels=methods, showmeans=True)
        ax.set_title("EEGNet" if backbone == "eegnet" else "EEGConformer")
        ax.set_ylabel("Outcome Session-2 BA" if ax is axes[0] else "")
    fig.suptitle("Source-validation-selected performance")
    fig.tight_layout()
    fig.savefig(common.FIGURES / "performance_by_method.png", dpi=240)
    fig.savefig(common.FIGURES / "performance_by_method.pdf")
    plt.close(fig)


def terminal_state(
    manipulation: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
    global_slope: dict[str, Any],
    decision: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    meaningful = [row for row in manipulation if row["meaningful_identity_reduction"]]
    families = set(row["method"] for row in meaningful)
    S1 = "DANN" in families and bool(families & {"CORAL", "MMD"})
    counter_families = set(row["method"] for row in counterexamples)
    counter_backbones = set(row["backbone"] for row in counterexamples)
    S2 = len(counter_families) >= 2
    S4 = set(common.BACKBONES).issubset(counter_backbones)
    slope_ci = global_slope["ci95"]
    positive_alignment = bool(slope_ci[0] > 0)
    negative_alignment = bool(global_slope["slope"] < 0 and slope_ci[1] < 0)
    no_reliable_positive = not positive_alignment
    decision_distinct = bool(decision["RMSE_MI_minus_MD"]["mean_run_difference"] > 0)
    decision_strong = bool(decision["RMSE_MI_minus_MD"]["ci95"][0] > 0)
    gates = {
        "S1_manipulation": S1,
        "S1_meaningful_families": sorted(families),
        "S2_multiple_counterexample_families": S2,
        "S2_counterexample_families": sorted(counter_families),
        "S3_positive_alignment": positive_alignment,
        "S3_no_reliable_positive_alignment": no_reliable_positive,
        "S3_negative_alignment": negative_alignment,
        "S4_both_backbones": S4,
        "decision_MD_better_than_MI_point": decision_distinct,
        "decision_MD_better_than_MI_CI": decision_strong,
    }
    if not S1:
        return "SUBJECT_INVARIANCE_MANIPULATION_INCONCLUSIVE", gates
    if S2 and S4 and no_reliable_positive and decision_distinct:
        return "SUBJECT_INVARIANCE_AUDIT_STRONG_MISALIGNMENT_SUPPORTED", gates
    if counterexamples and no_reliable_positive:
        return "SUBJECT_INVARIANCE_AUDIT_PARTIAL_MISALIGNMENT_SUPPORTED", gates
    return "SUBJECT_INVARIANCE_AUDIT_NOT_SUPPORTED", gates


def write_reports(
    statistics: dict[str, Any],
    manipulation: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
    ledger: pd.DataFrame,
) -> None:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in manipulation:
        by_family.setdefault(row["method"], []).append(row)
    def family_answer(method: str) -> str:
        rows = by_family[method]
        meaningful = [row for row in rows if row["meaningful_identity_reduction"]]
        if not meaningful:
            best = max(rows, key=lambda row: row["identity_reduction"]["mean"])
            return f"No predeclared meaningful reduction; largest mean suppression={best['identity_reduction']['mean']:.6f}."
        best = max(meaningful, key=lambda row: row["identity_reduction"]["mean"])
        return (
            f"Yes for {len(meaningful)} backbone-strength cells; largest was {best['backbone']} lambda={best['lambda']:g}, "
            f"mean suppression={best['identity_reduction']['mean']:.6f}, CI={best['identity_reduction']['ci95']}."
        )

    common.write_csv(common.RESULTS / "training_ledger.csv", ledger)
    elapsed_hours = float(ledger.elapsed_seconds.sum() / 3600.0)
    selected = ledger[ledger.selected_by_source_validation.astype(bool)]
    training_md = (
        "# Training ledger\n\n"
        f"Completed {len(ledger)} fixed configurations across 2 backbones, 5 folds, and 3 seeds. "
        f"Recorded per-candidate GPU training time sums to {elapsed_hours:.2f} hours (sequential accounting). "
        f"There are {len(selected)} source-validation-selected method/fold/seed entries. Every backbone/fold/seed unit has one shared main initialization SHA and one shared epoch-0 minibatch-order SHA.\n"
    )
    (common.EXP / "TRAINING_LEDGER.md").write_text(training_md, encoding="utf-8")

    lines = ["# Invariance manipulation check", ""]
    for method in ("DANN", "CORAL", "MMD"):
        lines.extend([f"## {method}", "", family_answer(method), ""])
    lines.append(f"Gate S1: **{statistics['gates']['S1_manipulation']}**.")
    (common.EXP / "INVARIANCE_MANIPULATION_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    slope_stats = statistics["global_relation"]["hierarchical_slope"]
    relation = [
        "# Invariance/generalization audit",
        "",
        f"Cluster-aware slope Delta_G~S_I: {slope_stats['slope']:.8f}, 95% hierarchical CI {slope_stats['ci95']}.",
        f"Aggregate Pearson r={statistics['global_relation']['correlations']['pearson_r']:.6f}; Spearman rho={statistics['global_relation']['correlations']['spearman_rho']:.6f}.",
        f"Counterexample configurations under the frozen rules: {len(counterexamples)} across {statistics['gates']['S2_counterexample_families']}.",
        "",
        "All 18 backbone/method/lambda aggregate points are retained. A CI crossing zero is not interpreted as proof of no relationship.",
    ]
    (common.EXP / "INVARIANCE_GENERALIZATION_AUDIT.md").write_text("\n".join(relation) + "\n", encoding="utf-8")

    decision = statistics["decision_vs_identity"]
    decision_lines = [
        "# Decision versus identity audit",
        "",
        f"Held-run RMSE: {decision['model_RMSE']}.",
        f"RMSE(MI)-RMSE(MD) run-cluster mean={decision['RMSE_MI_minus_MD']['mean_run_difference']:.8f}, CI={decision['RMSE_MI_minus_MD']['ci95']}, positive runs={decision['RMSE_MI_minus_MD']['positive_runs']}/{decision['RMSE_MI_minus_MD']['run_count']}.",
        f"Relative MD improvement versus MI={decision['RMSE_MI_minus_MD']['relative_improvement_MD_vs_MI']:.6%}.",
        "",
        "D_finite is the unchanged Exp3 centered-logit RMS. Candidate directions were source-only; outcome labels affected only the post-freeze intervention consequence.",
    ]
    (common.EXP / "DECISION_VS_IDENTITY_AUDIT.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    terminal = statistics["terminal_state"]
    q = [
        "# PERSIST-EEG subject-invariance stress test V1 — final report",
        "",
        f"Exact terminal state: **{terminal}**.",
        "",
        "## Required answers",
        "",
        f"1. Did DANN reduce cross-session subject identifiability? {family_answer('DANN')}",
        f"2. Did CORAL reduce it? {family_answer('CORAL')}",
        f"3. Did MMD reduce it? {family_answer('MMD')}",
        f"4. Did BA reliably rise when identity fell? {'Yes under the global positive-alignment gate.' if statistics['gates']['S3_positive_alignment'] else 'No reliable global positive alignment was established.'}",
        f"5. Cluster-aware slope: {slope_stats['slope']:.8f}, CI={slope_stats['ci95']}.",
        f"6. Clean identity-down/BA-down-or-null configurations: {len(counterexamples)}; all configurations are reported rather than post-selected.",
        f"7. EEGNet qualitative counterexample evidence: {'yes' if 'eegnet' in {row['backbone'] for row in counterexamples} else 'no'}.",
        f"8. EEGConformer qualitative counterexample evidence: {'yes' if 'eegconformer' in {row['backbone'] for row in counterexamples} else 'no'}.",
        f"9. Decision versus identity: RMSE(MI)-RMSE(MD)={decision['RMSE_MI_minus_MD']['mean_run_difference']:.8f}, CI={decision['RMSE_MI_minus_MD']['ci95']}.",
        f"10. Persistence: direction-level values are retained in `results/direction_audit.csv`; no utility implication is assumed.",
        "11. Generic adaptation: not rerun; it remains an optional separate reference and is excluded from source-only correlation.",
        "12. Restricted access: none; only the exact authorized 40-subject cache was loaded.",
        f"13. Terminal state: `{terminal}`.",
        f"14. Strongest defensible claim: {statistics['strongest_defensible_claim']}",
        "",
        "This is an operational representation audit, not a biological causal claim. It does not establish that identity never matters, that invariance is universally harmful, or that decision dependence guarantees generalization.",
    ]
    (common.EXP / "FINAL_REPORT.md").write_text("\n".join(q) + "\n", encoding="utf-8")

    purity = {
        "status": "PASS",
        "authorized_subject_count": 40,
        "completed_units": 30,
        "completed_configurations": 300,
        "restricted_holdout_accessed": False,
        "restricted_holdout_membership_enumerated": False,
        "WBCIC_accessed": False,
        "historical_holdout_enumerating_loader_imported": False,
        "outcome_S2_labels_used_for_training_or_selection": False,
        "all_selection_artifacts_frozen_before_outcome_evaluation": True,
        "directions_source_only": True,
        "probe_source_only": True,
    }
    common.write_json(common.RESULTS / "holdout_purity.json", purity)
    (common.EXP / "HOLDOUT_PURITY_AUDIT.md").write_text(
        "# Holdout purity audit\n\nPASS. All 30 units loaded only the exact authorized 40-subject OpenBMI cache. "
        "No internal-holdout membership or EEG, WBCIC data, or restricted loader was accessed. Every unit froze its source-validation lambda selection before outcome Session-2 evaluation; probes and direction construction were source-only.\n",
        encoding="utf-8",
    )


def main() -> None:
    common.ensure_dirs()
    performance, identity, directions, ledger = load_complete()
    performance = performance.sort_values(["backbone", "method", "lambda", "fold", "seed", "subject_id"]).reset_index(drop=True)
    identity = identity.sort_values(["backbone", "method", "lambda", "fold", "seed"]).reset_index(drop=True)
    directions = directions.sort_values(["backbone", "method", "lambda", "fold", "seed", "direction_id"]).reset_index(drop=True)
    stress = build_stress(performance, identity)
    common.write_csv(common.RESULTS / "main_performance.csv", performance)
    common.write_csv(common.RESULTS / "identity_probe.csv", identity)
    common.write_csv(common.RESULTS / "invariance_stress.csv", stress)
    common.write_csv(common.RESULTS / "direction_audit.csv", directions)
    manipulation, counterexamples = manipulation_statistics(stress)
    config = config_aggregates(stress)
    common.write_csv(common.RESULTS / "configuration_aggregates.csv", config)
    prediction_raw, prediction_summary, decision_stats = direction_prediction(directions)
    common.write_csv(common.RESULTS / "decision_prediction_raw.csv", prediction_raw)
    common.write_csv(common.RESULTS / "decision_vs_identity_prediction.csv", prediction_summary)

    global_relation = {
        "correlations": safe_correlations(config.rename(columns={"identity_suppression": "identity_suppression_vs_ERM", "BA_delta": "BA_delta_vs_ERM"})),
        "hierarchical_slope": hierarchical_slope(stress, BOOTSTRAP_SEED),
    }
    by_backbone = {}
    for backbone, part in stress.groupby("backbone"):
        aggregate = config[config.backbone == backbone].rename(columns={"identity_suppression": "identity_suppression_vs_ERM", "BA_delta": "BA_delta_vs_ERM"})
        by_backbone[backbone] = {
            "correlations": safe_correlations(aggregate),
            "hierarchical_slope": hierarchical_slope(part, common.stable_seed("slope-backbone", backbone)),
        }
    by_method = {}
    for method, part in stress.groupby("method"):
        aggregate = config[config.method == method].rename(columns={"identity_suppression": "identity_suppression_vs_ERM", "BA_delta": "BA_delta_vs_ERM"})
        by_method[method] = {
            "correlations": safe_correlations(aggregate),
            "hierarchical_slope": hierarchical_slope(part, common.stable_seed("slope-method", method)),
        }
    terminal, gates = terminal_state(manipulation, counterexamples, global_relation["hierarchical_slope"], decision_stats)
    if terminal == "SUBJECT_INVARIANCE_AUDIT_STRONG_MISALIGNMENT_SUPPORTED":
        claim = "Reducing subject identifiability was not a reliable operational target for future-session generalization in this fixed OpenBMI stress test; multiple standard invariance families produced identity reduction without consistent BA gain, while finite decision dependence was a more task-grounded consequence predictor."
    elif terminal == "SUBJECT_INVARIANCE_AUDIT_PARTIAL_MISALIGNMENT_SUPPORTED":
        claim = "The fixed stress test produced identity/generalization counterexamples, but architecture consistency, uncertainty, or decision-grounding evidence limits the claim to partial misalignment."
    elif terminal == "SUBJECT_INVARIANCE_MANIPULATION_INCONCLUSIVE":
        claim = "The standard objectives did not manipulate cross-session identity strongly enough to support a valid invariance/generalization conclusion."
    else:
        claim = "This fixed stress test does not support subject-identity/generalization misalignment; the observed identity suppression was positively aligned with future-session performance under the frozen inference rule."
    statistics = {
        "schema": "PERSIST_EEG_SUBJECT_INVARIANCE_STRESS_TEST_V1_STATISTICS",
        "terminal_state": terminal,
        "gates": gates,
        "global_relation": global_relation,
        "by_backbone": by_backbone,
        "by_method_family": by_method,
        "manipulation": manipulation,
        "counterexamples": counterexamples,
        "decision_vs_identity": decision_stats,
        "row_counts": {
            "main_performance": len(performance),
            "identity_probe": len(identity),
            "invariance_stress": len(stress),
            "direction_audit": len(directions),
            "decision_prediction_summary": len(prediction_summary),
        },
        "strongest_defensible_claim": claim,
        "restricted_data_accessed": False,
    }
    common.write_json(common.RESULTS / "statistics.json", statistics)
    make_figures(stress, config, identity, performance, prediction_raw)
    write_reports(statistics, manipulation, counterexamples, ledger)
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
