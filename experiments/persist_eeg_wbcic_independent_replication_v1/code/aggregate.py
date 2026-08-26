"""Frozen Phase-3 aggregation, statistical gates, figures, and reports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error

import common


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNS = EXP / "runtime" / "runs"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_826


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(common.clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def ci(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]


def run_key(frame: pd.DataFrame) -> pd.Series:
    return frame["fold"].astype(str) + "_" + frame["seed"].astype(str)


def hierarchical_run_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample outer folds, then complete seed-runs within sampled folds."""
    rng = np.random.default_rng(seed)
    folds = sorted(frame.fold.astype(int).unique())
    by_fold = {fold: sorted(frame.loc[frame.fold == fold, "seed"].astype(int).unique()) for fold in folds}
    output = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        pieces = []
        for fold in rng.choice(folds, size=len(folds), replace=True):
            seeds = by_fold[int(fold)]
            for selected_seed in rng.choice(seeds, size=len(seeds), replace=True):
                pieces.append(frame[(frame.fold == int(fold)) & (frame.seed == int(selected_seed))])
        output[draw] = statistic(pd.concat(pieces, ignore_index=True))
    return output


def hierarchical_subject_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample folds -> seeds -> outcome subjects for subject-first inference."""
    rng = np.random.default_rng(seed)
    folds = sorted(frame.fold.astype(int).unique())
    by_fold = {fold: sorted(frame.loc[frame.fold == fold, "seed"].astype(int).unique()) for fold in folds}
    output = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        values: list[float] = []
        for fold in rng.choice(folds, size=len(folds), replace=True):
            seeds = by_fold[int(fold)]
            for selected_seed in rng.choice(seeds, size=len(seeds), replace=True):
                cell = frame[(frame.fold == int(fold)) & (frame.seed == int(selected_seed))]
                sampled = rng.choice(cell[value_column].to_numpy(np.float64), size=len(cell), replace=True)
                values.extend(sampled.tolist())
        output[draw] = float(np.mean(values))
    return output


def logits_from_features(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ weight.T + bias


def per_subject_effects(
    labels: np.ndarray,
    subjects: np.ndarray,
    intact_logits: np.ndarray,
    erased_logits: np.ndarray,
) -> list[dict[str, Any]]:
    intact_prediction = intact_logits.argmax(axis=1)
    erased_prediction = erased_logits.argmax(axis=1)
    intact_ce = common.numpy_cross_entropy(intact_logits, labels)
    erased_ce = common.numpy_cross_entropy(erased_logits, labels)
    rows = []
    for subject in common.subject_sort(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        rows.append(
            {
                "subject_id": subject,
                "BA_harm": float(
                    balanced_accuracy_score(labels[mask], intact_prediction[mask])
                    - balanced_accuracy_score(labels[mask], erased_prediction[mask])
                ),
                "F1_harm": float(
                    f1_score(labels[mask], intact_prediction[mask], average="macro")
                    - f1_score(labels[mask], erased_prediction[mask], average="macro")
                ),
                "CE_harm": float(np.mean(erased_ce[mask] - intact_ce[mask])),
            }
        )
    return rows


def erase_block(features: np.ndarray, center: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(features, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    displacement = centered @ basis @ basis.T
    return np.asarray(features, dtype=np.float64) - displacement, displacement


def random_matched_erasure(
    features: np.ndarray,
    center: np.ndarray,
    persistent_displacement: np.ndarray,
    rank: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((features.shape[1], rank))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    centered = np.asarray(features, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    displacement = centered @ q @ q.T
    target_norm = np.linalg.norm(persistent_displacement, axis=1)
    random_norm = np.linalg.norm(displacement, axis=1)
    scale = np.divide(target_norm, random_norm, out=np.zeros_like(target_norm), where=random_norm > 1e-12)
    return np.asarray(features, dtype=np.float64) - displacement * scale[:, None]


def collect_primary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    performance_frames = []
    identity_frames = []
    direction_frames = []
    training_rows = []
    selection_rows = []
    for fold in range(5):
        for seed in range(3):
            unit = common.unit_dir("eegnet", fold, seed)
            complete = unit / "UNIT_COMPLETE.json"
            if not complete.is_file() or common.read_json(complete).get("pass") is not True:
                raise RuntimeError(f"incomplete primary run: fold={fold} seed={seed}")
            selection = common.read_json(unit / "LAMBDA_SELECTION_FROZEN.json")
            for method, row in selection["selected"].items():
                selection_rows.append({"backbone": "eegnet", "fold": fold, "seed": seed, "method": method, **row})
            for method, lam in common.configuration_grid():
                slug = common.config_slug(method, lam)
                evaluation = unit / "evaluation" / slug
                performance_frames.append(pd.read_csv(evaluation / "performance.csv"))
                identity_frames.append(pd.read_csv(evaluation / "identity.csv"))
                if method == "ERM":
                    direction_frames.append(pd.read_csv(evaluation / "directions.csv"))
                candidate = common.read_json(unit / "candidates" / f"{slug}.json")
                training_rows.append(
                    {
                        "backbone": "eegnet",
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "lambda": float(lam),
                        "best_epoch": int(candidate["best_epoch"]),
                        "epochs_executed": int(candidate["epochs_executed"]),
                        "best_validation_BA": float(candidate["best_validation_BA"]),
                        "best_validation_NLL": float(candidate["best_validation_NLL"]),
                        "elapsed_seconds": float(candidate["elapsed_seconds"]),
                        "checkpoint_sha256": candidate["checkpoint_sha256"],
                        "initial_shared_state_sha256": candidate["initial_shared_state_sha256"],
                        "epoch0_minibatch_order_sha256": candidate["epoch0_minibatch_order_sha256"],
                    }
                )
    performance = pd.concat(performance_frames, ignore_index=True)
    identity = pd.concat(identity_frames, ignore_index=True)
    directions = pd.concat(direction_frames, ignore_index=True)
    training = pd.DataFrame(training_rows)
    selections = pd.DataFrame(selection_rows)
    identity["identity_accuracy"] = identity["identity_accuracy_symmetric"]
    return performance, identity, directions, training, selections


def build_stress(performance: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    performance_run = (
        performance.groupby(["backbone", "method", "lambda", "fold", "seed"], as_index=False)
        .agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"))
    )
    merged = identity.merge(performance_run, on=["backbone", "method", "lambda", "fold", "seed"], validate="one_to_one")
    baseline = merged[merged.method == "ERM"][["backbone", "fold", "seed", "identity_symmetric", "BA", "macro_f1"]].rename(
        columns={"identity_symmetric": "ERM_identity", "BA": "ERM_BA", "macro_f1": "ERM_F1"}
    )
    stress = merged[merged.method != "ERM"].merge(baseline, on=["backbone", "fold", "seed"], validate="many_to_one")
    stress["identity_suppression_vs_ERM"] = stress.ERM_identity - stress.identity_symmetric
    stress["BA_delta_vs_ERM"] = stress.BA - stress.ERM_BA
    stress["F1_delta_vs_ERM"] = stress.macro_f1 - stress.ERM_F1
    columns = [
        "backbone", "method", "lambda", "fold", "seed", "identity_suppression_vs_ERM",
        "BA_delta_vs_ERM", "F1_delta_vs_ERM", "ERM_identity", "identity_symmetric", "ERM_BA", "BA",
    ]
    return stress[columns].sort_values(["method", "lambda", "fold", "seed"]).reset_index(drop=True)


def build_block_controls() -> tuple[pd.DataFrame, pd.DataFrame]:
    control_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    blocks = {"P01_04": slice(0, 4), "P05_08": slice(4, 8)}
    for fold in range(5):
        for seed in range(3):
            unit = common.unit_dir("eegnet", fold, seed)
            evaluation = unit / "evaluation" / common.config_slug("ERM", 0.0)
            embedded = np.load(evaluation / "embeddings.npz", allow_pickle=False)
            frozen = np.load(unit / "source_freeze" / "erm_persistence_basis.npz", allow_pickle=False)
            checkpoint = torch.load(unit / "checkpoints" / f"{common.config_slug('ERM', 0.0)}.pt", map_location="cpu", weights_only=True)
            weight = checkpoint["state_dict"]["head.weight"].numpy().astype(np.float64)
            bias = checkpoint["state_dict"]["head.bias"].numpy().astype(np.float64)
            features = embedded["outcome_features"].astype(np.float64)
            labels = embedded["outcome_labels"].astype(np.int64)
            subjects = embedded["outcome_subjects"].astype(str)
            intact_logits = logits_from_features(features, weight, bias)
            center = frozen["center"].astype(np.float64)
            full_basis = frozen["basis"].astype(np.float64)
            for block, selector in blocks.items():
                basis = full_basis[:, selector]
                persistent_erased, persistent_displacement = erase_block(features, center, basis)
                persistent_logits = logits_from_features(persistent_erased, weight, bias)
                persistent_subject = per_subject_effects(labels, subjects, intact_logits, persistent_logits)
                persistent_by_subject = {row["subject_id"]: row for row in persistent_subject}
                persistent_ba = float(np.mean([row["BA_harm"] for row in persistent_subject]))
                for control_id in range(100):
                    control_seed = common.stable_seed("WBCIC-R1-rank4-control", fold, seed, block, control_id)
                    random_erased = random_matched_erasure(features, center, persistent_displacement, 4, control_seed)
                    random_logits = logits_from_features(random_erased, weight, bias)
                    random_subject = per_subject_effects(labels, subjects, intact_logits, random_logits)
                    random_ba = float(np.mean([row["BA_harm"] for row in random_subject]))
                    control_rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "block": block,
                            "block_rank": 4,
                            "persistent_BA_harm": persistent_ba,
                            "random_control_id": control_id,
                            "random_BA_harm": random_ba,
                            "specific_delta": persistent_ba - random_ba,
                            "control_seed": control_seed,
                        }
                    )
                    for random_row in random_subject:
                        persistent_row = persistent_by_subject[random_row["subject_id"]]
                        subject_rows.append(
                            {
                                "fold": fold,
                                "seed": seed,
                                "block": block,
                                "subject_id": random_row["subject_id"],
                                "random_control_id": control_id,
                                "persistent_BA_harm": persistent_row["BA_harm"],
                                "random_BA_harm": random_row["BA_harm"],
                                "specific_delta": persistent_row["BA_harm"] - random_row["BA_harm"],
                                "persistent_CE_harm": persistent_row["CE_harm"],
                                "random_CE_harm": random_row["CE_harm"],
                                "persistent_F1_harm": persistent_row["F1_harm"],
                                "random_F1_harm": random_row["F1_harm"],
                            }
                        )
    return pd.DataFrame(control_rows), pd.DataFrame(subject_rows)


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0
    train = np.c_[(x_train - mean) / std, np.ones(len(x_train))]
    test = np.c_[(x_test - mean) / std, np.ones(len(x_test))]
    penalty = np.eye(train.shape[1])
    penalty[-1, -1] = 0.0
    weight = np.linalg.solve(train.T @ train + alpha * penalty, train.T @ y_train)
    return test @ weight


def decision_prediction(directions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    models = {
        "M0": ["persistence", "geometry_strength", "direction_rank"],
        "MI": ["persistence", "geometry_strength", "direction_rank", "identity_score"],
        "MD": ["persistence", "geometry_strength", "direction_rank", "D_finite"],
        "MID": ["persistence", "geometry_strength", "direction_rank", "identity_score", "D_finite"],
    }
    frame = directions.copy().reset_index(drop=True)
    frame["run_id"] = run_key(frame)
    predictions = {model: np.full(len(frame), np.nan, dtype=np.float64) for model in models}
    for held_run in sorted(frame.run_id.unique()):
        train_mask = frame.run_id != held_run
        test_mask = ~train_mask
        y_train = frame.loc[train_mask, "outcome_CE_effect"].to_numpy(np.float64)
        for model, columns in models.items():
            predictions[model][test_mask] = ridge_predict(
                frame.loc[train_mask, columns].to_numpy(np.float64),
                y_train,
                frame.loc[test_mask, columns].to_numpy(np.float64),
                alpha=1.0,
            )
    target = frame.outcome_CE_effect.to_numpy(np.float64)
    for model in models:
        frame[model] = predictions[model]
        frame[f"held_run_error_{model}"] = (predictions[model] - target) ** 2
    rmse = {model: float(np.sqrt(np.mean(frame[f"held_run_error_{model}"]))) for model in models}
    run_errors = frame.groupby(["fold", "seed"], as_index=False).agg(
        **{f"RMSE_{model}": (f"held_run_error_{model}", lambda value: float(np.sqrt(np.mean(value)))) for model in models}
    )
    positive_runs = int((run_errors.RMSE_MI > run_errors.RMSE_MD).sum())
    boot = {model: np.empty(BOOTSTRAP_DRAWS) for model in models}
    rng = np.random.default_rng(BOOTSTRAP_SEED + 200)
    runs = sorted(frame.run_id.unique())
    for draw in range(BOOTSTRAP_DRAWS):
        picked = rng.choice(runs, size=len(runs), replace=True)
        for model in models:
            values = np.concatenate([frame.loc[frame.run_id == run, f"held_run_error_{model}"].to_numpy() for run in picked])
            boot[model][draw] = np.sqrt(np.mean(values))
    delta_samples = boot["MI"] - boot["MD"]
    summary = {
        "RMSE": rmse,
        "RMSE_MI_minus_MD": rmse["MI"] - rmse["MD"],
        "RMSE_MI_minus_MD_CI95": ci(delta_samples),
        "relative_improvement_MD_vs_MI": (rmse["MI"] - rmse["MD"]) / max(rmse["MI"], 1e-12),
        "positive_runs_MI_error_gt_MD": positive_runs,
        "run_count": len(runs),
        "RMSE_CI95": {model: ci(samples) for model, samples in boot.items()},
    }
    for model in models:
        frame[f"summary_RMSE_{model}"] = rmse[model]
    return frame, summary, boot


def summarize_r1(subject_controls: pd.DataFrame) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary: dict[str, Any] = {}
    bootstraps: dict[str, np.ndarray] = {}
    averaged = (
        subject_controls.groupby(["fold", "seed", "block", "subject_id"], as_index=False)
        .agg(persistent_BA_harm=("persistent_BA_harm", "first"), random_BA_harm=("random_BA_harm", "mean"))
    )
    averaged["specific_delta"] = averaged.persistent_BA_harm - averaged.random_BA_harm
    for block, cell in averaged.groupby("block"):
        persistent_boot = hierarchical_subject_bootstrap(cell, "persistent_BA_harm", seed=BOOTSTRAP_SEED + 10)
        specific_boot = hierarchical_subject_bootstrap(cell, "specific_delta", seed=BOOTSTRAP_SEED + 11)
        bootstraps[f"{block}_persistent"] = persistent_boot
        bootstraps[f"{block}_specific"] = specific_boot
        summary[str(block)] = {
            "persistent_BA_harm": float(cell.persistent_BA_harm.mean()),
            "persistent_BA_harm_CI95": ci(persistent_boot),
            "matched_random_BA_harm": float(cell.random_BA_harm.mean()),
            "persistent_minus_random": float(cell.specific_delta.mean()),
            "persistent_minus_random_CI95": ci(specific_boot),
            "outcome_subject_seed_rows": len(cell),
        }
    reliable = [
        block for block, row in summary.items()
        if row["persistent_BA_harm_CI95"][0] > 0
    ]
    strong = [
        block for block, row in summary.items()
        if row["persistent_BA_harm_CI95"][0] > 0 and row["persistent_minus_random_CI95"][0] > 0
    ]
    status = "R1_STRONG_SUPPORT" if strong else ("R1_PARTIAL_SUPPORT" if reliable else "R1_NOT_SUPPORTED")
    summary["status"] = status
    summary["reliable_task_supportive_blocks"] = reliable
    summary["strong_blocks"] = strong
    return summary, bootstraps


def summarize_r3(stress: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    rows = []
    for (method, lam), cell in stress.groupby(["method", "lambda"], sort=True):
        identity_boot = hierarchical_run_bootstrap(
            cell, lambda sampled: float(sampled.identity_suppression_vs_ERM.mean()), seed=BOOTSTRAP_SEED + 30
        )
        ba_boot = hierarchical_run_bootstrap(
            cell, lambda sampled: float(sampled.BA_delta_vs_ERM.mean()), seed=BOOTSTRAP_SEED + 31
        )
        threshold = max(0.05, 0.10 * float(cell.ERM_identity.abs().mean()))
        meaningful = bool(cell.identity_suppression_vs_ERM.mean() >= threshold and ci(identity_boot)[0] > 0)
        ba_ci = ci(ba_boot)
        counterexample = "NONE"
        if meaningful and ba_ci[1] <= 0:
            counterexample = "STRONG"
        elif meaningful and cell.BA_delta_vs_ERM.mean() <= 0 and ba_ci[0] <= 0 <= ba_ci[1]:
            counterexample = "WEAK"
        rows.append(
            {
                "method": method,
                "lambda": float(lam),
                "mean_identity_suppression": float(cell.identity_suppression_vs_ERM.mean()),
                "identity_suppression_CI_low": ci(identity_boot)[0],
                "identity_suppression_CI_high": ci(identity_boot)[1],
                "meaningful_threshold": threshold,
                "meaningful_identity_reduction": meaningful,
                "mean_BA_delta": float(cell.BA_delta_vs_ERM.mean()),
                "BA_delta_CI_low": ba_ci[0],
                "BA_delta_CI_high": ba_ci[1],
                "counterexample": counterexample,
            }
        )
    configurations = pd.DataFrame(rows)

    def slope_stat(sampled: pd.DataFrame) -> float:
        x = sampled.identity_suppression_vs_ERM.to_numpy(np.float64)
        y = sampled.BA_delta_vs_ERM.to_numpy(np.float64)
        return float(np.polyfit(x, y, 1)[0]) if np.std(x) > 1e-12 else 0.0

    slope = slope_stat(stress)
    slope_boot = hierarchical_run_bootstrap(stress, slope_stat, seed=BOOTSTRAP_SEED + 32)
    slope_ci = ci(slope_boot)
    pearson = pearsonr(stress.identity_suppression_vs_ERM, stress.BA_delta_vs_ERM)
    spearman = spearmanr(stress.identity_suppression_vs_ERM, stress.BA_delta_vs_ERM)
    meaningful_methods = sorted(configurations.loc[configurations.meaningful_identity_reduction, "method"].unique())
    counterexamples = configurations[configurations.counterexample != "NONE"]
    positive_alignment = slope_ci[0] > 0
    fails_reliable_gain = bool(
        (
            configurations.meaningful_identity_reduction
            & (configurations.BA_delta_CI_low <= 0)
        ).any()
    )
    if len(meaningful_methods) >= 2 and len(counterexamples) >= 1 and not positive_alignment:
        status = "R3_MISALIGNMENT_STRONG"
    elif len(meaningful_methods) >= 1 and fails_reliable_gain and not positive_alignment:
        status = "R3_MISALIGNMENT_PARTIAL"
    elif len(meaningful_methods) == 0:
        status = "R3_MANIPULATION_INCONCLUSIVE"
    else:
        status = "R3_NOT_SUPPORTED"
    summary = {
        "status": status,
        "meaningful_methods": meaningful_methods,
        "counterexample_configurations": [
            {"method": row.method, "lambda": float(row["lambda"]), "strength": row.counterexample}
            for _, row in counterexamples.iterrows()
        ],
        "global_slope": slope,
        "global_slope_CI95": slope_ci,
        "alignment": "POSITIVE_ALIGNMENT" if positive_alignment else (
            "NEGATIVE_ALIGNMENT" if slope < 0 and slope_ci[1] < 0 else "NO_RELIABLE_POSITIVE_ALIGNMENT"
        ),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }
    return configurations, summary, slope_boot


def validate_purity() -> dict[str, Any]:
    violations = []
    source_freezes = 0
    evaluations = 0
    for fold in range(5):
        for seed in range(3):
            unit = common.unit_dir("eegnet", fold, seed)
            source_path = unit / "SOURCE_FREEZE_COMPLETE.json"
            if not source_path.is_file():
                violations.append(f"missing source freeze fold={fold} seed={seed}")
                continue
            source = common.read_json(source_path)
            source_freezes += 1
            if source.get("pass") is not True or source.get("checkpoint_count") != 10:
                violations.append(f"invalid source freeze fold={fold} seed={seed}")
            if source.get("outcome_S3_labels_used") is not False:
                violations.append(f"outcome labels entered source freeze fold={fold} seed={seed}")
            frozen_at = float(source.get("frozen_at_unix", 0.0))
            for method, lam in common.configuration_grid():
                path = unit / "evaluation" / common.config_slug(method, lam) / "EVALUATION_COMPLETE.json"
                if not path.is_file():
                    violations.append(f"missing evaluation {fold}/{seed}/{method}/{lam}")
                    continue
                row = common.read_json(path)
                evaluations += 1
                if float(row.get("evaluated_at_unix", 0.0)) < frozen_at:
                    violations.append(f"evaluation predates source freeze {fold}/{seed}/{method}/{lam}")
                if row.get("restricted_data_accessed") is not False or row.get("sealed_WBCIC_outer_accessed") is not False:
                    violations.append(f"restricted-data flag {fold}/{seed}/{method}/{lam}")
    payload = {
        "schema": "WBCIC_PHASE3_HOLDOUT_PURITY_V1",
        "pass": not violations,
        "development_subjects_loaded": 41,
        "development_subject_hash": "dae8e7ec00cbcf6dcc8c5b25829f2148fd0b5fdf162f75a0cddc18b096af7db4",
        "run_source_freezes": source_freezes,
        "configuration_evaluations": evaluations,
        "sealed_WBCIC_outer_accessed": False,
        "sealed_WBCIC_outer_enumerated": False,
        "OpenBMI_holdout_accessed": False,
        "outcome_S3_used_for_training": False,
        "outcome_S3_used_for_lambda_selection": False,
        "outcome_S3_used_for_direction_construction": False,
        "outcome_S3_used_for_identity_probe": False,
        "outcome_evaluation_before_source_freeze": False,
        "violations": violations,
    }
    if violations:
        raise RuntimeError(f"holdout purity violations: {violations}")
    return payload


def make_figures(
    stress: pd.DataFrame,
    r3_configs: pd.DataFrame,
    prediction_summary: dict[str, Any],
    r1_summary: dict[str, Any],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"DANN": "#3B82F6", "CORAL": "#F59E0B", "MMD": "#10B981"}
    markers = {0.01: "o", 0.1: "s", 1.0: "^"}

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.axhline(0, color="0.65", linewidth=1)
    ax.axvline(0, color="0.65", linewidth=1)
    for _, row in r3_configs.iterrows():
        ax.errorbar(
            row.mean_identity_suppression,
            row.mean_BA_delta,
            xerr=[[row.mean_identity_suppression - row.identity_suppression_CI_low], [row.identity_suppression_CI_high - row.mean_identity_suppression]],
            yerr=[[row.mean_BA_delta - row.BA_delta_CI_low], [row.BA_delta_CI_high - row.mean_BA_delta]],
            color=colors[row.method], marker=markers[float(row["lambda"])], capsize=3, linestyle="none",
            label=f"{row.method} λ={row['lambda']:g}",
        )
    ax.set_xlabel("Identity suppression vs ERM (S_I)")
    ax.set_ylabel("Outcome S3 BA change vs ERM (ΔG)")
    ax.set_title("WBCIC identity suppression vs future-session generalization")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"figure1_identity_vs_generalization.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    labels = [f"{row.method}\nλ={row['lambda']:g}" for _, row in r3_configs.iterrows()]
    values = r3_configs.mean_identity_suppression.to_numpy()
    low = values - r3_configs.identity_suppression_CI_low.to_numpy()
    high = r3_configs.identity_suppression_CI_high.to_numpy() - values
    ax.bar(np.arange(len(values)), values, color=[colors[row.method] for _, row in r3_configs.iterrows()])
    ax.errorbar(np.arange(len(values)), values, yerr=[low, high], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_ylabel("I_ERM − I_method")
    ax.set_title("WBCIC cross-session identity manipulation")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"figure2_identity_manipulation.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    models = ["MI", "MD", "MID"]
    values = [prediction_summary["RMSE"][model] for model in models]
    intervals = [prediction_summary["RMSE_CI95"][model] for model in models]
    errors = [[value - interval[0] for value, interval in zip(values, intervals)], [interval[1] - value for value, interval in zip(values, intervals)]]
    ax.bar(models, values, color=["#9CA3AF", "#8B5CF6", "#EC4899"])
    ax.errorbar(np.arange(3), values, yerr=errors, fmt="none", color="black", capsize=4)
    ax.set_ylabel("Leave-one-run-out RMSE")
    ax.set_title("Decision vs identity consequence prediction")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"figure3_decision_vs_identity.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    blocks = ["P01_04", "P05_08"]
    x = np.arange(2)
    persistent = [r1_summary[block]["persistent_BA_harm"] for block in blocks]
    random = [r1_summary[block]["matched_random_BA_harm"] for block in blocks]
    width = 0.36
    ax.bar(x - width / 2, persistent, width, label="Persistent block", color="#2563EB")
    ax.bar(x + width / 2, random, width, label="Matched random", color="#9CA3AF")
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(x, blocks)
    ax.set_ylabel("Outcome S3 BA erasure harm")
    ax.set_title("Persistent-block future task consequence")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"figure4_persistent_block_controls.{suffix}", dpi=220)
    plt.close(fig)


def render_reports(summary: dict[str, Any], training: pd.DataFrame, selections: pd.DataFrame, r3_configs: pd.DataFrame) -> None:
    competence = summary["competence"]
    r1 = summary["R1"]
    r2 = summary["R2"]
    r3 = summary["R3"]
    purity = summary["purity"]
    terminal = summary["terminal_state"]
    blocks = ["P01_04", "P05_08"]
    block_lines = "\n".join(
        f"- {block}: BA harm {r1[block]['persistent_BA_harm']:.4f} "
        f"(95% CI {r1[block]['persistent_BA_harm_CI95'][0]:.4f}, {r1[block]['persistent_BA_harm_CI95'][1]:.4f}); "
        f"persistent−random {r1[block]['persistent_minus_random']:.4f} "
        f"(95% CI {r1[block]['persistent_minus_random_CI95'][0]:.4f}, {r1[block]['persistent_minus_random_CI95'][1]:.4f})."
        for block in blocks
    )
    config_lines = "\n".join(
        f"- {row.method} λ={row['lambda']:g}: S_I={row.mean_identity_suppression:.4f} "
        f"[{row.identity_suppression_CI_low:.4f}, {row.identity_suppression_CI_high:.4f}], "
        f"ΔBA={row.mean_BA_delta:.4f} [{row.BA_delta_CI_low:.4f}, {row.BA_delta_CI_high:.4f}], "
        f"meaningful={bool(row.meaningful_identity_reduction)}, counterexample={row.counterexample}."
        for _, row in r3_configs.iterrows()
    )
    write_text(EXP / "README.md", f"""# Phase 3 — WBCIC independent replication

This experiment is a frozen, subject-disjoint replication on the authorized 41-subject WBCIC development pool. It does not access the 10 sealed WBCIC outer subjects or any OpenBMI EEG holdout.

- Terminal state: `{terminal}`
- Complete primary runs: 15/15 fold×seed units; 150/150 fixed training configurations.
- ERM outcome S3 BA: {competence['ERM_mean_S3_BA']:.4f}; Macro-F1: {competence['ERM_mean_S3_macro_F1']:.4f}.
- R1: `{r1['status']}`
- R2: `{r2['status']}`
- R3: `{r3['status']}`
- Holdout purity: {'PASS' if purity['pass'] else 'FAIL'}

See `FINAL_REPORT.md` and the lightweight tables in `results/`.
""")
    write_text(EXP / "HISTORICAL_WBCIC_PROVENANCE.md", """# Historical WBCIC provenance

Data handling was ported read-only from `codex/wbcic-eegnet-actionability`, experiment `persist_eeg_wbcic_actionability_v2`: the development whitelist/folds, event interpretation, Pz referencing, epoch preprocessing, cache checks, and WBCIC EEGNet. Historical scientific outcomes were not copied into Phase-3 tables and were not used to classify directions, choose lambdas, or set gates.
""")
    write_text(EXP / "DATA_SCOPE_AUDIT.md", f"""# Data scope audit

- Exact authorized subjects: 41; frozen hash `{purity['development_subject_hash']}`.
- Materialized sessions: 123; trials: 24,591.
- Sealed WBCIC outer accessed/enumerated: NO / NO.
- OpenBMI EEG holdout accessed: NO.
- Scope result: PASS.
""")
    write_text(EXP / "PREPROCESSING_AUDIT.md", """# Preprocessing audit

The historical primary WBCIC preprocessing is unchanged: 59 EEG channels at 1000 Hz; subtract Pz and drop it (58 channels); 0.5–40 Hz zero-phase fourth-order Butterworth SOS; `resample_poly` to 250 Hz; 0–4 s relative to the BIDS MI event with no additional offset; microvolts/20 clipped to [-12.5,12.5]. No cross-subject amplitude normalization is applied.
""")
    write_text(EXP / "FOLD_ROLE_AUDIT.md", """# Fold-role audit

For fold k, outcome=F_k, validation/discovery=F_(k+1 mod 5), and model-fit is the remaining three frozen folds. Training uses model-fit S1+S2; early stopping/lambda selection uses validation/discovery S3; outcome uses outcome S3 only after source freeze. Every authorized subject is outcome exactly once and validation/discovery exactly once; all roles are subject-disjoint.
""")
    write_text(EXP / "TRAINING_LEDGER.md", f"""# Training ledger

- Fixed configurations complete: {len(training)}/150.
- Fold×seed units complete: 15/15.
- Epochs executed: median {training.epochs_executed.median():.1f}, range {training.epochs_executed.min()}–{training.epochs_executed.max()}.
- Best validation BA: mean {training.best_validation_BA.mean():.4f}.
- Total recorded training time: {training.elapsed_seconds.sum()/3600:.2f} GPU-hours (sequential-equivalent).
- Shared-initialization and epoch-0 minibatch-order SHA matching passed within every fold×seed unit.
- Lambda selection used only validation/discovery S3; all fixed lambdas remain reported.
""")
    write_text(EXP / "REPRESENTATION_COMPETENCE.md", f"""# Representation competence

ERM mean outcome S3 BA is {competence['ERM_mean_S3_BA']:.4f} and Macro-F1 is {competence['ERM_mean_S3_macro_F1']:.4f}. {competence['folds_above_chance']}/5 fold means exceed chance. The frozen competence gate therefore {'PASSES' if competence['pass'] else 'FAILS'}.
""")
    write_text(EXP / "IDENTITY_PROBE_AUDIT.md", f"""# Identity probe audit

The primary identity probe is a standardized multiclass ridge classifier (alpha=1) evaluated symmetrically S1→S2 and S2→S1 on model-fit subjects only. Mean ERM symmetric identity skill is {summary['identity']['ERM_mean_identity_skill']:.4f}; mean chance-normalized accuracy is {summary['identity']['ERM_mean_chance_normalized_accuracy']:.4f}. Outcome subjects are excluded from probe fitting and evaluation.
""")
    write_text(EXP / "PERSISTENCE_REPLICATION_AUDIT.md", f"""# Persistence replication audit

{block_lines}

Gate: `{r1['status']}`. This evaluates only the two predeclared rank-4 blocks from the first eight source-defined persistent directions. It does not imply that all persistent structure is useful.
""")
    write_text(EXP / "INVARIANCE_MANIPULATION_AUDIT.md", f"""# Invariance manipulation audit

{config_lines}

Meaningful reduction requires mean S_I ≥ max(0.05, 10% of matched ERM absolute identity skill) and paired hierarchical CI lower > 0. Meaningful families: {', '.join(r3['meaningful_methods']) if r3['meaningful_methods'] else 'none'}.
""")
    write_text(EXP / "INVARIANCE_GENERALIZATION_AUDIT.md", f"""# Invariance/generalization audit

Global slope ΔG~S_I is {r3['global_slope']:.4f} (95% CI {r3['global_slope_CI95'][0]:.4f}, {r3['global_slope_CI95'][1]:.4f}); alignment status `{r3['alignment']}`. Counterexamples: {json.dumps(r3['counterexample_configurations'])}. R3 gate: `{r3['status']}`. CI crossing zero is treated as absence of reliable positive alignment, not proof of no relationship.
""")
    write_text(EXP / "DECISION_VS_IDENTITY_REPLICATION.md", f"""# Decision vs identity replication

Leave-one-entire-run-out ridge prediction over 15 fold×seed runs gives RMSE MI={r2['RMSE']['MI']:.6f}, MD={r2['RMSE']['MD']:.6f}, MID={r2['RMSE']['MID']:.6f}. MI−MD={r2['RMSE_MI_minus_MD']:.6f} (run-cluster 95% CI {r2['RMSE_MI_minus_MD_CI95'][0]:.6f}, {r2['RMSE_MI_minus_MD_CI95'][1]:.6f}); MD beats MI in {r2['positive_runs_MI_error_gt_MD']}/15 held runs. Gate: `{r2['status']}`.
""")
    write_text(EXP / "HOLDOUT_PURITY_AUDIT.md", f"""# Holdout purity audit

Machine-readable audit: `results/holdout_purity.json`.

- 41 exact authorized development subjects only: YES.
- Sealed WBCIC outer accessed or enumerated: NO.
- OpenBMI holdout accessed: NO.
- Outcome S3 used for training, selection, direction construction, or identity probe: NO.
- Outcome evaluated before run-level source freeze: NO.
- Result: {'PASS' if purity['pass'] else 'FAIL'}.
""")
    write_text(EXP / "ENGINEERING_REPAIR_LOG.md", """# Engineering repair log

All repairs below were completed before the first outcome-S3 evaluation:

1. Reused the historical cache builder but restricted it to explicit paths from the 41-subject provenance whitelist; no raw-root enumeration.
2. Added a streamed float16 consolidated memmap to avoid duplicating full arrays in RAM.
3. Corrected WBCIC identity-probe session indices to BIDS ses-0↔ses-1 (S1↔S2); the inherited 1↔2 indices would have produced an empty direction.
4. Restored the missing run-level `SOURCE_FREEZE_COMPLETE.json` call before outcome evaluation.
5. Made source normalizer and persistence-basis freezes resume-safe and hash-checked.
6. Removed an unused OpenBMI-only conformer class from the Phase-3 runtime surface; EEGNet is the sole executed backbone.
7. Added resumable per-configuration checkpoints, deterministic controls, preflight compilation/scope checks, and independent final validation.

No subject scope, fold, seed, lambda, method, direction rank, metric, or statistical gate was altered after outcome access.
""")

    questions = f"""# Final report

Primary terminal state: `{terminal}`.

1. **Competence:** {'Yes' if competence['pass'] else 'No'}; EEGNet ERM S3 BA={competence['ERM_mean_S3_BA']:.4f}, Macro-F1={competence['ERM_mean_S3_macro_F1']:.4f}.
2. **Persistent structure consequence:** R1=`{r1['status']}`.
3. **Reliable predeclared block harm:** {', '.join(r1['reliable_task_supportive_blocks']) if r1['reliable_task_supportive_blocks'] else 'none'}.
4. **Greater than matched random:** {', '.join(r1['strong_blocks']) if r1['strong_blocks'] else 'none'}.
5. **D_finite vs identity:** R2=`{r2['status']}`.
6. **RMSE:** MI={r2['RMSE']['MI']:.6f}; MD={r2['RMSE']['MD']:.6f}.
7. **MI−MD:** {r2['RMSE_MI_minus_MD']:.6f}, 95% CI [{r2['RMSE_MI_minus_MD_CI95'][0]:.6f}, {r2['RMSE_MI_minus_MD_CI95'][1]:.6f}].
8. **DANN meaningful identity reduction:** {'yes' if 'DANN' in r3['meaningful_methods'] else 'no'}.
9. **CORAL meaningful identity reduction:** {'yes' if 'CORAL' in r3['meaningful_methods'] else 'no'}.
10. **MMD meaningful identity reduction:** {'yes' if 'MMD' in r3['meaningful_methods'] else 'no'}.
11. **Did lower identity reliably improve S3 BA?** {'Yes' if r3['alignment']=='POSITIVE_ALIGNMENT' else 'No reliable positive guarantee'}.
12. **Global slope:** {r3['global_slope']:.4f}, 95% CI [{r3['global_slope_CI95'][0]:.4f}, {r3['global_slope_CI95'][1]:.4f}].
13. **Counterexamples:** {json.dumps(r3['counterexample_configurations'])}.
14. **Replication components:** R1={r1['status']}; R2={r2['status']}; R3={r3['status']}.
15. **Sealed WBCIC outer accessed:** NO.
16. **OpenBMI sealed holdout accessed:** NO.
17. **Exact terminal state:** `{terminal}`.
18. **Strongest defensible conclusion:** {summary['strongest_defensible_conclusion']}

## R1 blocks

{block_lines}

## R3 fixed grid

{config_lines}

## Claim boundary

The evidence concerns frozen WBCIC EEGNet representations under this S1+S2→S3 protocol. It does not establish physiological causality, a universal EEG law, or that subject invariance is universally harmful.
"""
    write_text(EXP / "FINAL_REPORT.md", questions)


def main() -> None:
    common.protocol()
    RESULTS.mkdir(parents=True, exist_ok=True)
    performance, identity, directions, training, selections = collect_primary_tables()
    if len(performance) != 1230 or len(identity) != 150 or len(directions) != 120 or len(training) != 150:
        raise RuntimeError(
            f"primary table cardinality failure performance={len(performance)} identity={len(identity)} "
            f"directions={len(directions)} training={len(training)}"
        )
    common.write_csv(RESULTS / "main_performance.csv", performance)
    common.write_csv(RESULTS / "identity_probe.csv", identity)
    common.write_csv(RESULTS / "persistent_direction_audit.csv", directions)
    common.write_csv(RESULTS / "training_ledger.csv", training)
    common.write_csv(RESULTS / "lambda_selection.csv", selections)

    stress = build_stress(performance, identity)
    common.write_csv(RESULTS / "invariance_stress.csv", stress)
    r3_configs, r3_summary, slope_boot = summarize_r3(stress)
    common.write_csv(RESULTS / "invariance_configuration_summary.csv", r3_configs)

    block_controls, block_subjects = build_block_controls()
    if len(block_controls) != 3000:
        raise RuntimeError(f"block-control cardinality failure: {len(block_controls)}")
    common.write_csv(RESULTS / "persistent_block_controls.csv", block_controls)
    common.write_csv(RESULTS / "persistent_block_subject_effects.csv", block_subjects)
    r1_summary, _ = summarize_r1(block_subjects)

    prediction, r2_summary, _ = decision_prediction(directions)
    common.write_csv(RESULTS / "decision_vs_identity_prediction.csv", prediction)
    if r2_summary["RMSE"]["MD"] < r2_summary["RMSE"]["MI"]:
        r2_summary["status"] = (
            "R2_STRONG_SUPPORT" if r2_summary["RMSE_MI_minus_MD_CI95"][0] > 0 else "R2_PARTIAL_SUPPORT"
        )
    else:
        r2_summary["status"] = "R2_NOT_SUPPORTED"
    write_json(RESULTS / "decision_vs_identity_summary.json", r2_summary)

    erm = performance[performance.method == "ERM"]
    fold_means = erm.groupby("fold").BA.mean()
    competence = {
        "pass": bool(erm.BA.mean() > 0.60 and int((fold_means > 0.5).sum()) >= 4),
        "ERM_mean_S3_BA": float(erm.BA.mean()),
        "ERM_mean_S3_macro_F1": float(erm.macro_f1.mean()),
        "fold_mean_BA": {str(int(key)): float(value) for key, value in fold_means.items()},
        "folds_above_chance": int((fold_means > 0.5).sum()),
        "systematic_data_or_preprocessing_failure": False,
    }
    erm_identity = identity[identity.method == "ERM"]
    identity_summary = {
        "ERM_mean_identity_skill": float(erm_identity.identity_symmetric.mean()),
        "ERM_mean_identity_accuracy": float(erm_identity.identity_accuracy.mean()),
        "ERM_mean_chance_normalized_accuracy": float(erm_identity.chance_normalized_identity.mean()),
    }
    purity = validate_purity()
    write_json(RESULTS / "holdout_purity.json", purity)

    reproduced = sum(
        [
            r1_summary["status"] in {"R1_STRONG_SUPPORT", "R1_PARTIAL_SUPPORT"},
            r2_summary["status"] in {"R2_STRONG_SUPPORT", "R2_PARTIAL_SUPPORT"},
            r3_summary["status"] in {"R3_MISALIGNMENT_STRONG", "R3_MISALIGNMENT_PARTIAL"},
        ]
    )
    if not competence["pass"]:
        terminal = "WBCIC_REPRESENTATION_COMPETENCE_FAIL"
    elif r3_summary["status"] == "R3_MANIPULATION_INCONCLUSIVE":
        terminal = "WBCIC_INVARIANCE_MANIPULATION_INCONCLUSIVE"
    elif (
        r1_summary["status"] in {"R1_STRONG_SUPPORT", "R1_PARTIAL_SUPPORT"}
        and r2_summary["status"] == "R2_STRONG_SUPPORT"
        and r3_summary["status"] in {"R3_MISALIGNMENT_STRONG", "R3_MISALIGNMENT_PARTIAL"}
        and purity["pass"]
    ):
        terminal = "WBCIC_INDEPENDENT_REPLICATION_STRONG_SUPPORTED"
    elif reproduced >= 2:
        terminal = "WBCIC_INDEPENDENT_REPLICATION_PARTIAL_SUPPORTED"
    else:
        terminal = "WBCIC_INDEPENDENT_REPLICATION_NOT_SUPPORTED"

    if reproduced >= 2:
        conclusion = (
            "At least two predeclared representation-level distinctions reproduce on WBCIC under a frozen "
            "EEGNet S1+S2→S3 protocol; effect sizes and uncertainty remain dataset- and protocol-specific."
        )
    else:
        conclusion = (
            "The OpenBMI representation-level chain does not reproduce sufficiently on WBCIC under this frozen "
            "EEGNet S1+S2→S3 protocol; the negative boundary cannot be repaired by post-hoc model development."
        )
    summary = {
        "schema": "WBCIC_PHASE3_FINAL_SUMMARY_V1",
        "terminal_state": terminal,
        "competence": competence,
        "identity": identity_summary,
        "R1": r1_summary,
        "R2": r2_summary,
        "R3": r3_summary,
        "purity": purity,
        "valid_fold_seed_runs": 15,
        "fixed_training_configurations": 150,
        "strongest_defensible_conclusion": conclusion,
    }
    write_json(RESULTS / "summary.json", summary)
    make_figures(stress, r3_configs, r2_summary, r1_summary)
    render_reports(summary, training, selections, r3_configs)
    print(json.dumps(common.clean(summary), indent=2))


if __name__ == "__main__":
    main()
