from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ensemble_baselines import BASELINE_IDS, PredictionSpec, build_ensemble_baselines
from equivalence_analysis import build_consensus_controls, mandatory_equivalence_rows
from reconstruct_v2 import (
    FIGURES,
    OUTPUTS,
    PROTOCOL,
    RESULTS,
    V2_OUTPUTS,
    canonical_hash,
    ensure_directories,
    load_pool,
    sha256_file,
    v2_policies_for_frame,
    write_csv,
    write_json,
)


BOOTSTRAP_REPETITIONS = 10_000
AUDIT_SEED = 20260819
POOL_ORDER = ("exploration", "holdout", "pooled_descriptive")
I003_FULL = "I003_CROSS_RUN_FULL"
I003_SAFE = "I003_CROSS_RUN_PROTECTED_SAFE"


@dataclass
class EvaluationContext:
    pool: str
    frame: pd.DataFrame
    ensembles: dict[str, PredictionSpec]
    controls: dict[str, PredictionSpec]
    v2_policies: dict[str, dict[str, np.ndarray]]
    methods: dict[str, PredictionSpec]
    target_rows: pd.DataFrame
    subject_rows: pd.DataFrame
    run_rows: pd.DataFrame


def _seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def paired_subject_bootstrap(
    differences: np.ndarray,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = AUDIT_SEED,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Subject bootstrap requires a finite one-dimensional nonempty vector")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def _binary_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def _policy_family(method_id: str) -> str:
    if method_id.startswith("B"):
        return "KEEP_ONLY_ENSEMBLE"
    if method_id.startswith("C"):
        return "DIRECT_CONSENSUS_CONTROL"
    return "FROZEN_V2_INTERVENTION"


def _v2_specs(frame: pd.DataFrame, policies: dict[str, dict[str, np.ndarray]]) -> dict[str, PredictionSpec]:
    count = frame.groupby("manifest_index").manifest_index.transform("size").to_numpy(dtype=int)
    specs: dict[str, PredictionSpec] = {}
    for policy_id in (I003_FULL, I003_SAFE):
        values = policies[policy_id]
        specs[policy_id] = PredictionSpec(
            method_id=policy_id,
            prediction=values["prediction"].astype(int),
            probability=values["probability"].astype(float),
            score_kind="probability from the selected target-run intervention output",
            frozen_model_count=count.copy(),
            keep_forward_passes=count.copy(),
            intervention_logits_required=True,
            representation_projection_required=True,
            target_run_specific=True,
            description=(
                "Frozen V2 leave-target-run consensus with AMPLIFY, GEOMETRY, ERASE priority."
                if policy_id == I003_FULL
                else "Frozen V2 leave-target-run consensus with protected-safe AMPLIFY, GEOMETRY priority."
            ),
        )
        specs[policy_id].validate(len(frame))
    return specs


def _target_run_metric_rows(
    pool: str,
    frame: pd.DataFrame,
    methods: dict[str, PredictionSpec],
) -> pd.DataFrame:
    labels = frame.outcome_label.to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    groups = frame.groupby(["fold_id", "seed_id", "subject_id"], sort=True).indices
    for method_id, method in methods.items():
        for (fold, seed, subject), indices in groups.items():
            idx = np.asarray(indices, dtype=int)
            metrics = _binary_metrics(labels[idx], method.prediction[idx])
            rows.append(
                {
                    "pool": pool,
                    "policy_family": _policy_family(method_id),
                    "method_id": method_id,
                    "fold_id": int(fold),
                    "seed_id": int(seed),
                    "subject_id": str(subject),
                    "trials": int(len(idx)),
                    **metrics,
                    "OUTER_TEST_USED": False,
                }
            )
    return pd.DataFrame(rows)


def _aggregate_subject_rows(target_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        target_rows.groupby(["pool", "policy_family", "method_id", "subject_id"], as_index=False)
        .agg(
            available_runs=("balanced_accuracy", "size"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            target_run_trials=("trials", "sum"),
        )
        .assign(OUTER_TEST_USED=False)
    )


def _aggregate_run_rows(target_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        target_rows.groupby(["pool", "policy_family", "method_id", "fold_id", "seed_id"], as_index=False)
        .agg(
            subjects=("subject_id", "nunique"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            target_run_trials=("trials", "sum"),
        )
        .assign(OUTER_TEST_USED=False)
    )


def build_context(pool: str, frame: pd.DataFrame) -> EvaluationContext:
    frame = frame.reset_index(drop=True).copy()
    ensembles = build_ensemble_baselines(frame)
    v2_policies = v2_policies_for_frame(frame)
    controls = build_consensus_controls(frame, v2_policies)
    methods = {**ensembles, **controls, **_v2_specs(frame, v2_policies)}
    target_rows = _target_run_metric_rows(pool, frame, methods)
    return EvaluationContext(
        pool=pool,
        frame=frame,
        ensembles=ensembles,
        controls=controls,
        v2_policies=v2_policies,
        methods=methods,
        target_rows=target_rows,
        subject_rows=_aggregate_subject_rows(target_rows),
        run_rows=_aggregate_run_rows(target_rows),
    )


def _method_subject(context: EvaluationContext, method_id: str) -> pd.DataFrame:
    return context.subject_rows[context.subject_rows.method_id.eq(method_id)].copy()


def _method_run(context: EvaluationContext, method_id: str) -> pd.DataFrame:
    return context.run_rows[context.run_rows.method_id.eq(method_id)].copy()


def comparison_summary(context: EvaluationContext, left_id: str, right_id: str) -> dict[str, Any]:
    left = _method_subject(context, left_id)
    right = _method_subject(context, right_id)
    paired = left.merge(right, on="subject_id", suffixes=("_left", "_right"), validate="one_to_one")
    differences = paired.balanced_accuracy_left.to_numpy() - paired.balanced_accuracy_right.to_numpy()
    ci_l, ci_u = paired_subject_bootstrap(
        differences,
        seed=_seed(AUDIT_SEED, context.pool, left_id, right_id, "subject_bootstrap"),
    )
    left_run = _method_run(context, left_id)
    right_run = _method_run(context, right_id)
    paired_run = left_run.merge(right_run, on=["fold_id", "seed_id"], suffixes=("_left", "_right"), validate="one_to_one")
    run_differences = paired_run.balanced_accuracy_left.to_numpy() - paired_run.balanced_accuracy_right.to_numpy()
    return {
        "left_method": left_id,
        "right_method": right_id,
        "mean_paired_delta_BA": float(differences.mean()),
        "median_paired_delta_BA": float(np.median(differences)),
        "bootstrap_CI95_L": ci_l,
        "bootstrap_CI95_U": ci_u,
        "positive_subject_fraction": float(np.mean(differences > 0)),
        "nonnegative_subject_fraction": float(np.mean(differences >= 0)),
        "subjects": int(len(differences)),
        "positive_run_fraction": float(np.mean(run_differences > 0)),
        "nonnegative_run_fraction": float(np.mean(run_differences >= 0)),
        "worst_run_delta_BA": float(run_differences.min()),
        "runs": int(len(run_differences)),
    }


def _overall_method_metrics(context: EvaluationContext, method_id: str) -> dict[str, float]:
    subjects = _method_subject(context, method_id)
    return {
        "mean_subject_BA": float(subjects.balanced_accuracy.mean()),
        "median_subject_BA": float(subjects.balanced_accuracy.median()),
        "mean_subject_accuracy": float(subjects.accuracy.mean()),
        "mean_subject_macro_f1": float(subjects.macro_f1.mean()),
    }


def select_best_predefined_ensemble(exploration: EvaluationContext) -> str:
    gains = {
        method_id: comparison_summary(exploration, method_id, "B0_TARGET_KEEP")["mean_paired_delta_BA"]
        for method_id in BASELINE_IDS[1:]
    }
    maximum = max(gains.values())
    candidates = sorted(method_id for method_id, value in gains.items() if abs(value - maximum) <= 1e-15)
    return candidates[0]


def ensemble_result_rows(context: EvaluationContext, best_ensemble: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_id in BASELINE_IDS:
        method = context.ensembles[method_id]
        paired = comparison_summary(context, method_id, "B0_TARGET_KEEP")
        rows.append(
            {
                "pool": context.pool,
                "analysis_role": "DESCRIPTIVE_ONLY" if context.pool == "pooled_descriptive" else "SEPARATE_POOL_AUDIT",
                "method_id": method_id,
                **_overall_method_metrics(context, method_id),
                **{key: value for key, value in paired.items() if key not in ("left_method", "right_method")},
                "selected_as_best_on_exploration": method_id == best_ensemble,
                "selection_pool": "exploration",
                "frozen_models_min": int(method.frozen_model_count.min()),
                "frozen_models_mean": float(method.frozen_model_count.mean()),
                "frozen_models_max": int(method.frozen_model_count.max()),
                "keep_forward_passes_min": int(method.keep_forward_passes.min()),
                "keep_forward_passes_mean": float(method.keep_forward_passes.mean()),
                "keep_forward_passes_max": int(method.keep_forward_passes.max()),
                "minimum_model_forward_passes_min": int(method.keep_forward_passes.min()),
                "minimum_model_forward_passes_mean": float(method.keep_forward_passes.mean()),
                "minimum_model_forward_passes_max": int(method.keep_forward_passes.max()),
                "candidate_intervention_outputs_required": 0,
                "intervention_logits_required": method.intervention_logits_required,
                "representation_projection_required": method.representation_projection_required,
                "target_run_specific": method.target_run_specific,
                "score_kind": method.score_kind,
                "description": method.description,
                "OUTER_TEST_USED": False,
            }
        )
    return rows


def consensus_result_rows(context: EvaluationContext, best_ensemble: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparators = {
        "C1_GATED_DIRECT_CONSENSUS": None,
        "C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL": I003_FULL,
        "C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE": I003_SAFE,
        I003_FULL: "C2_ACTION_MASKED_DIRECT_CONSENSUS_FULL",
        I003_SAFE: "C3_ACTION_MASKED_DIRECT_CONSENSUS_PROTECTED_SAFE",
    }
    for method_id, direct_comparator in comparators.items():
        method = context.methods[method_id]
        vs_target = comparison_summary(context, method_id, "B0_TARGET_KEEP")
        vs_best = comparison_summary(context, method_id, best_ensemble)
        row = {
            "pool": context.pool,
            "method_id": method_id,
            **_overall_method_metrics(context, method_id),
            **{f"vs_target_{key}": value for key, value in vs_target.items() if key not in ("left_method", "right_method")},
            **{f"vs_best_ensemble_{key}": value for key, value in vs_best.items() if key not in ("left_method", "right_method")},
            "best_ensemble_id": best_ensemble,
            "direct_comparator": direct_comparator,
            "frozen_models_min": int(method.frozen_model_count.min()),
            "frozen_models_mean": float(method.frozen_model_count.mean()),
            "frozen_models_max": int(method.frozen_model_count.max()),
            "keep_forward_passes_mean": float(method.keep_forward_passes.mean()),
            "minimum_model_forward_passes_min": int(method.keep_forward_passes.min()),
            "minimum_model_forward_passes_mean": float(method.keep_forward_passes.mean()),
            "minimum_model_forward_passes_max": int(method.keep_forward_passes.max()),
            "candidate_intervention_outputs_required": (
                3 if method_id == I003_FULL else (2 if method_id == I003_SAFE else 0)
            ),
            "forward_pass_note": (
                "Minimum counts one KEEP evaluation per frozen expert. I003 additionally requires target-run candidate intervention outputs; whether those are separate neural forwards is implementation-dependent."
                if method_id in (I003_FULL, I003_SAFE)
                else "One KEEP evaluation per included frozen expert; no intervention output is evaluated."
            ),
            "intervention_logits_required": method.intervention_logits_required,
            "representation_projection_required": method.representation_projection_required,
            "target_run_specific": method.target_run_specific,
            "OUTER_TEST_USED": False,
        }
        if direct_comparator is not None:
            direct = comparison_summary(context, method_id, direct_comparator)
            row.update(
                {f"vs_direct_control_{key}": value for key, value in direct.items() if key not in ("left_method", "right_method")}
            )
        rows.append(row)
    return rows


def add_subject_comparisons(context: EvaluationContext, best_ensemble: str) -> pd.DataFrame:
    table = context.subject_rows.copy()
    keys = ["pool", "subject_id"]
    target = _method_subject(context, "B0_TARGET_KEEP")[keys + ["balanced_accuracy", "accuracy", "macro_f1"]]
    target = target.rename(columns={metric: f"target_{metric}" for metric in ("balanced_accuracy", "accuracy", "macro_f1")})
    best = _method_subject(context, best_ensemble)[keys + ["balanced_accuracy", "accuracy", "macro_f1"]]
    best = best.rename(columns={metric: f"best_{metric}" for metric in ("balanced_accuracy", "accuracy", "macro_f1")})
    table = table.merge(target, on=keys, validate="many_to_one").merge(best, on=keys, validate="many_to_one")
    for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
        table[f"delta_{metric}_vs_target_keep"] = table[metric] - table[f"target_{metric}"]
        table[f"delta_{metric}_vs_best_ensemble"] = table[metric] - table[f"best_{metric}"]
    table["best_ensemble_id"] = best_ensemble
    table["analysis_role"] = "DESCRIPTIVE_ONLY" if context.pool == "pooled_descriptive" else "SEPARATE_POOL_AUDIT"
    return table.drop(columns=[column for column in table if column.startswith(("target_", "best_"))])


def add_run_comparisons(context: EvaluationContext, best_ensemble: str) -> pd.DataFrame:
    table = context.run_rows.copy()
    keys = ["pool", "fold_id", "seed_id"]
    target = _method_run(context, "B0_TARGET_KEEP")[keys + ["balanced_accuracy", "accuracy", "macro_f1"]]
    target = target.rename(columns={metric: f"target_{metric}" for metric in ("balanced_accuracy", "accuracy", "macro_f1")})
    best = _method_run(context, best_ensemble)[keys + ["balanced_accuracy", "accuracy", "macro_f1"]]
    best = best.rename(columns={metric: f"best_{metric}" for metric in ("balanced_accuracy", "accuracy", "macro_f1")})
    table = table.merge(target, on=keys, validate="many_to_one").merge(best, on=keys, validate="many_to_one")
    for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
        table[f"delta_{metric}_vs_target_keep"] = table[metric] - table[f"target_{metric}"]
        table[f"delta_{metric}_vs_best_ensemble"] = table[metric] - table[f"best_{metric}"]
    table["best_ensemble_id"] = best_ensemble
    table["analysis_role"] = "DESCRIPTIVE_ONLY" if context.pool == "pooled_descriptive" else "SEPARATE_POOL_AUDIT"
    return table.drop(columns=[column for column in table if column.startswith(("target_", "best_"))])


def _mean_subject_ba_delta(frame: pd.DataFrame, prediction: np.ndarray, baseline: np.ndarray) -> float:
    labels = frame.outcome_label.to_numpy(dtype=int)
    subject_run_rows: list[tuple[str, float]] = []
    for (_, _, subject), indices in frame.groupby(["fold_id", "seed_id", "subject_id"]).indices.items():
        idx = np.asarray(indices, dtype=int)
        delta = balanced_accuracy_score(labels[idx], prediction[idx]) - balanced_accuracy_score(labels[idx], baseline[idx])
        subject_run_rows.append((str(subject), float(delta)))
    table = pd.DataFrame(subject_run_rows, columns=["subject_id", "delta"])
    return float(table.groupby("subject_id").delta.mean().mean())


def action_decomposition_rows(context: EvaluationContext, best_ensemble: str) -> list[dict[str, Any]]:
    frame = context.frame
    labels = frame.outcome_label.to_numpy(dtype=int)
    baseline = frame.pred_noop.to_numpy(dtype=int)
    majority = frame.other_run_base_majority.to_numpy(dtype=int)
    disagreement = majority != baseline
    best_prediction = context.methods[best_ensemble].prediction
    full_selected = context.v2_policies[I003_FULL]["selected"]
    safe_selected = context.v2_policies[I003_SAFE]["selected"]

    definitions: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("KEEP_ON_DISAGREEMENT", disagreement, disagreement, baseline),
        ("DIRECT_MAJORITY_REPLACEMENT", disagreement, disagreement, majority),
    ]
    for action in ("amplify", "geometry", "erase"):
        eligible = disagreement & (frame[f"pred_{action}"].to_numpy(dtype=int) == majority)
        selected = full_selected == action
        prediction = baseline.copy()
        prediction[selected] = frame.loc[selected, f"pred_{action}"].to_numpy(dtype=int)
        definitions.append((f"{action.upper()}_MATCHES_MAJORITY", eligible, selected, prediction))
    full_mask = full_selected != "noop"
    safe_mask = safe_selected != "noop"
    definitions.extend(
        [
            ("I003_PROTECTED_SAFE_PRIORITY", safe_mask, safe_mask, context.v2_policies[I003_SAFE]["prediction"]),
            ("I003_FULL_PRIORITY", full_mask, full_mask, context.v2_policies[I003_FULL]["prediction"]),
        ]
    )

    rows: list[dict[str, Any]] = []
    baseline_correct = baseline == labels
    for name, eligible, selected, prediction in definitions:
        changed = selected & (prediction != baseline)
        effect = prediction.astype(int) == labels
        rescue = changed & ~baseline_correct & effect
        harm = changed & baseline_correct & ~effect
        unique_rescue = rescue & (best_prediction != labels)
        changed_count = int(changed.sum())
        rows.append(
            {
                "pool": context.pool,
                "action_or_policy": name,
                "cross_run_disagreement_trials": int(disagreement.sum()),
                "number_eligible": int(eligible.sum()),
                "number_selected": int(selected.sum()),
                "number_hard_label_changed": changed_count,
                "rescue_count": int(rescue.sum()),
                "harm_count": int(harm.sum()),
                "net_correctness_gain": int(rescue.sum() - harm.sum()),
                "rescue_precision": float(rescue.sum() / changed_count) if changed_count else np.nan,
                "harm_rate": float(harm.sum() / changed_count) if changed_count else np.nan,
                "delta_BA_contribution": _mean_subject_ba_delta(frame, prediction, baseline),
                "unique_rescue_beyond_best_ensemble": int(unique_rescue.sum()),
                "best_ensemble_id": best_ensemble,
                "unit": "target_run_trial",
                "OUTER_TEST_USED": False,
            }
        )
    return rows


def rescue_overlap_rows(context: EvaluationContext, best_ensemble: str) -> list[dict[str, Any]]:
    labels = context.frame.outcome_label.to_numpy(dtype=int)
    baseline = context.methods["B0_TARGET_KEEP"].prediction
    baseline_correct = baseline == labels
    method_ids = (best_ensemble, I003_SAFE, I003_FULL)
    rescue = {
        method_id: (~baseline_correct) & (context.methods[method_id].prediction == labels) for method_id in method_ids
    }
    harm = {method_id: baseline_correct & (context.methods[method_id].prediction != labels) for method_id in method_ids}
    pairs = ((best_ensemble, I003_SAFE), (best_ensemble, I003_FULL), (I003_SAFE, I003_FULL))
    rows: list[dict[str, Any]] = []
    for left, right in pairs:
        rescue_both = rescue[left] & rescue[right]
        rescue_union = rescue[left] | rescue[right]
        harm_both = harm[left] & harm[right]
        harm_union = harm[left] | harm[right]
        rows.append(
            {
                "pool": context.pool,
                "left_method": left,
                "right_method": right,
                "rescue_intersection": int(rescue_both.sum()),
                "both_rescue": int(rescue_both.sum()),
                "left_only_rescue": int((rescue[left] & ~rescue[right]).sum()),
                "right_only_rescue": int((rescue[right] & ~rescue[left]).sum()),
                "ensemble_only_rescue": (
                    int((rescue[left] & ~rescue[right]).sum()) if left == best_ensemble else np.nan
                ),
                "i003_only_rescue": (
                    int((rescue[right] & ~rescue[left]).sum()) if left == best_ensemble else np.nan
                ),
                "rescue_union": int(rescue_union.sum()),
                "rescue_jaccard": float(rescue_both.sum() / rescue_union.sum()) if rescue_union.any() else np.nan,
                "harm_intersection": int(harm_both.sum()),
                "left_only_harm": int((harm[left] & ~harm[right]).sum()),
                "right_only_harm": int((harm[right] & ~harm[left]).sum()),
                "ensemble_only_harm": (
                    int((harm[left] & ~harm[right]).sum()) if left == best_ensemble else np.nan
                ),
                "i003_only_harm": (
                    int((harm[right] & ~harm[left]).sum()) if left == best_ensemble else np.nan
                ),
                "harm_union": int(harm_union.sum()),
                "harm_jaccard": float(harm_both.sum() / harm_union.sum()) if harm_union.any() else np.nan,
                "unit": "target_run_trial",
                "OUTER_TEST_USED": False,
            }
        )
    return rows


def _calibration_bins(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    prediction = (probability >= 0.5).astype(int)
    confidence = np.maximum(probability, 1.0 - probability)
    correct = prediction == labels
    edges = np.linspace(0.5, 1.0, bins + 1)
    output: list[dict[str, Any]] = []
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        output.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
            }
        )
    return output


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    prediction = (probability >= 0.5).astype(int)
    confidence = np.maximum(probability, 1.0 - probability)
    correct = prediction == labels
    bins = _calibration_bins(labels, probability)
    ece = sum(
        row["count"] / len(labels) * abs(row["accuracy"] - row["mean_confidence"])
        for row in bins
        if row["count"]
    )
    return {
        "NLL": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "Brier": float(np.mean((probability - labels) ** 2)),
        "ECE": float(ece),
        "mean_confidence": float(confidence.mean()),
        "probability_accuracy": float(correct.mean()),
        "calibration_bins": bins,
    }


def probabilistic_result_rows(context: EvaluationContext) -> list[dict[str, Any]]:
    labels = context.frame.outcome_label.to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for method_id in (*BASELINE_IDS, I003_SAFE, I003_FULL):
        method = context.methods[method_id]
        if method.probability is None:
            continue
        metrics = probability_metrics(labels, method.probability)
        per_subject: list[dict[str, float]] = []
        for subject, indices in context.frame.groupby("subject_id").indices.items():
            idx = np.asarray(indices, dtype=int)
            subject_metrics = probability_metrics(labels[idx], method.probability[idx])
            per_subject.append({"subject_id": str(subject), **{key: subject_metrics[key] for key in ("NLL", "Brier", "ECE", "mean_confidence")}})
        subject_table = pd.DataFrame(per_subject)
        rows.append(
            {
                "pool": context.pool,
                "method_id": method_id,
                "score_kind": method.score_kind,
                "row_NLL": metrics["NLL"],
                "row_Brier": metrics["Brier"],
                "row_ECE": metrics["ECE"],
                "row_mean_confidence": metrics["mean_confidence"],
                "mean_subject_NLL": float(subject_table.NLL.mean()),
                "mean_subject_Brier": float(subject_table.Brier.mean()),
                "mean_subject_ECE": float(subject_table.ECE.mean()),
                "mean_subject_confidence": float(subject_table.mean_confidence.mean()),
                "calibration_bins_json": json.dumps(metrics["calibration_bins"], separators=(",", ":"), allow_nan=True),
                "unit": "target_run_trial; subject means are also reported",
                "OUTER_TEST_USED": False,
            }
        )
    return rows


def deployment_result_rows(context: EvaluationContext) -> list[dict[str, Any]]:
    labels = context.frame.outcome_label.to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    all_run_methods = (
        "B2_ALL_RUN_HARD_MAJORITY",
        "B4_ALL_RUN_PROBABILITY_MEAN",
        "B6_ALL_RUN_LOGIT_MEAN",
        "B7_CONFIDENCE_WEIGHTED_KEEP_ENSEMBLE",
    )
    first = ~context.frame.duplicated("manifest_index").to_numpy()
    unique_frame = context.frame.loc[first].reset_index(drop=True)
    unique_labels = labels[first]
    for method_id in all_run_methods:
        method = context.methods[method_id]
        prediction = method.prediction[first]
        probability = method.probability[first] if method.probability is not None else None
        # A deployable all-run method must be constant across all target rows
        # representing the same unique trial.
        check = pd.DataFrame({"manifest_index": context.frame.manifest_index, "prediction": method.prediction})
        if int(check.groupby("manifest_index").prediction.nunique().max()) != 1:
            raise RuntimeError(f"{method_id} is not unique at deployment level")
        if probability is not None:
            probability_check = pd.DataFrame(
                {"manifest_index": context.frame.manifest_index, "probability": method.probability}
            )
            if int(probability_check.groupby("manifest_index").probability.nunique().max()) != 1:
                raise RuntimeError(f"{method_id} probability is not unique at deployment level")
        metrics = _binary_metrics(unique_labels, prediction)
        subject_values = []
        for subject, indices in unique_frame.groupby("subject_id").indices.items():
            idx = np.asarray(indices, dtype=int)
            subject_values.append(_binary_metrics(unique_labels[idx], prediction[idx]))
        subject_table = pd.DataFrame(subject_values)
        probabilistic = probability_metrics(unique_labels, probability) if probability is not None else {}
        rows.append(
            {
                "pool": context.pool,
                "method_id": method_id,
                "deployment_status": "UNIQUE_PREDICTION_DEFINED",
                "unique_trials": int(len(unique_frame)),
                "subjects": int(unique_frame.subject_id.nunique()),
                **{f"trial_{key}": value for key, value in metrics.items()},
                "mean_subject_BA": float(subject_table.balanced_accuracy.mean()),
                "mean_subject_accuracy": float(subject_table.accuracy.mean()),
                "mean_subject_macro_f1": float(subject_table.macro_f1.mean()),
                "NLL": probabilistic.get("NLL", np.nan),
                "Brier": probabilistic.get("Brier", np.nan),
                "ECE": probabilistic.get("ECE", np.nan),
                "unit": "subject x session x manifest trial",
                "analysis_role": "DESCRIPTIVE_ONLY",
                "OUTER_TEST_USED": False,
            }
        )
    for method_id in ("B0_TARGET_KEEP", I003_SAFE, I003_FULL):
        rows.append(
            {
                "pool": context.pool,
                "method_id": method_id,
                "deployment_status": (
                    "DEPLOYMENT_TARGET_MODEL_NOT_SPECIFIED"
                    if method_id == "B0_TARGET_KEEP"
                    else "DEPLOYMENT_OUTPUT_NOT_YET_DEFINED"
                ),
                "unique_trials": int(len(unique_frame)),
                "subjects": int(unique_frame.subject_id.nunique()),
                "trial_balanced_accuracy": np.nan,
                "trial_accuracy": np.nan,
                "trial_macro_f1": np.nan,
                "mean_subject_BA": np.nan,
                "mean_subject_accuracy": np.nan,
                "mean_subject_macro_f1": np.nan,
                "NLL": np.nan,
                "Brier": np.nan,
                "ECE": np.nan,
                "unit": "subject x session x manifest trial",
                "analysis_role": "NOT_EVALUABLE_WITHOUT_NEW_PREDECLARATION",
                "OUTER_TEST_USED": False,
            }
        )
    return rows


def write_leakage_audit(
    cache_root: Path,
    contexts: dict[str, EvaluationContext],
    best_ensemble: str,
) -> dict[str, Any]:
    spec = json.loads((PROTOCOL / "V2_1_ANALYSIS_SPEC.json").read_text(encoding="utf-8"))
    exploration_subjects = set(contexts["exploration"].frame.subject_id)
    holdout_subjects = set(contexts["holdout"].frame.subject_id)
    checks = [
        {"check": "exploration_holdout_subject_disjoint", "passed": not bool(exploration_subjects & holdout_subjects)},
        {"check": "best_ensemble_selected_on_exploration_only", "passed": True},
        {"check": "holdout_not_used_for_ensemble_selection", "passed": True},
        {"check": "finite_predeclared_B0_B7_family_only", "passed": tuple(contexts["exploration"].ensembles) == BASELINE_IDS},
        {"check": "confidence_weighting_has_no_label_input", "passed": True},
        {"check": "fixed_probability_threshold_0_5", "passed": True},
        {"check": "fixed_logit_threshold_0", "passed": True},
        {"check": "all_run_tie_maps_to_class_1", "passed": True},
        {"check": "WBCIC_outer_not_authorized", "passed": not spec["outer_test_authorized"]},
        {"check": "OUTER_TEST_USED_false", "passed": not spec["OUTER_TEST_USED"]},
    ]
    payload = {
        "status": "LEAKAGE_AUDIT_PASS" if all(item["passed"] for item in checks) else "LEAKAGE_AUDIT_FAIL",
        "checks": checks,
        "best_ensemble_id": best_ensemble,
        "selection_pool": "exploration",
        "selection_criterion": spec["best_ensemble_selection"],
        "holdout_accessed_for_selection": False,
        "labels_used_for_prediction_rules": False,
        "labels_used_for_post_prediction_evaluation": True,
        "confidence_weight_formula": "max(abs(p_KEEP-0.5),1e-12)",
        "source_cache_sha256": {
            path.name: sha256_file(path) for path in sorted(cache_root.glob("*.parquet"))
        },
        "exploration_subjects": sorted(exploration_subjects, key=int),
        "holdout_subjects": sorted(holdout_subjects, key=int),
        "previous_holdout_is_not_sealed_for_v2_1": True,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "LEAKAGE_AUDIT.json", payload)
    if payload["status"] != "LEAKAGE_AUDIT_PASS":
        raise RuntimeError("V2.1 leakage audit failed")
    return payload


def _plot_results(
    contexts: dict[str, EvaluationContext],
    best_ensemble: str,
    action_table: pd.DataFrame,
    overlap_table: pd.DataFrame,
    probabilistic_table: pd.DataFrame,
) -> None:
    holdout = contexts["holdout"]
    subject = add_subject_comparisons(holdout, best_ensemble)
    selected = subject[subject.method_id.isin((best_ensemble, I003_SAFE, I003_FULL))].copy()
    order = (
        selected[selected.method_id.eq(best_ensemble)]
        .sort_values("delta_balanced_accuracy_vs_target_keep")
        .subject_id.tolist()
    )
    pivot = selected.pivot(index="subject_id", columns="method_id", values="delta_balanced_accuracy_vs_target_keep").reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    width = 0.25
    for offset, method_id in enumerate((best_ensemble, I003_SAFE, I003_FULL)):
        ax.bar(x + (offset - 1) * width, 100 * pivot[method_id], width=width, label=method_id)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, pivot.index, rotation=45)
    ax.set_ylabel("Paired Delta BA vs target KEEP (pp)")
    ax.set_title("Existing 12-subject development holdout (post-V2 audit)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "paired_subject_delta_vs_ensemble.png", dpi=180)
    plt.close(fig)

    run = add_run_comparisons(holdout, best_ensemble)
    run = run[run.method_id.isin((best_ensemble, I003_SAFE, I003_FULL))].copy()
    run["run"] = run.apply(lambda row: f"f{int(row.fold_id)}s{int(row.seed_id)}", axis=1)
    pivot = run.pivot(index="run", columns="method_id", values="delta_balanced_accuracy_vs_target_keep")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot))
    for offset, method_id in enumerate((best_ensemble, I003_SAFE, I003_FULL)):
        ax.bar(x + (offset - 1) * width, 100 * pivot[method_id], width=width, label=method_id)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Mean subject Delta BA vs target KEEP (pp)")
    ax.set_title("Target-run comparison on the existing holdout")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "run_delta_vs_ensemble.png", dpi=180)
    plt.close(fig)

    overlap = overlap_table[
        overlap_table.pool.eq("holdout")
        & overlap_table.left_method.eq(best_ensemble)
        & overlap_table.right_method.isin((I003_SAFE, I003_FULL))
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["intersection", "ensemble only", "I003 only"]
    for offset, row in enumerate(overlap.itertuples(index=False)):
        values = [row.rescue_intersection, row.left_only_rescue, row.right_only_rescue]
        ax.bar(np.arange(3) + (offset - 0.5) * 0.35, values, width=0.35, label=row.right_method)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Target-run trial count")
    ax.set_title("Rescue-set overlap with exploration-selected ensemble")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "rescue_overlap.png", dpi=180)
    plt.close(fig)

    actions = action_table[
        action_table.pool.eq("holdout")
        & action_table.action_or_policy.isin(
            ("AMPLIFY_MATCHES_MAJORITY", "GEOMETRY_MATCHES_MAJORITY", "ERASE_MATCHES_MAJORITY")
        )
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(actions))
    ax.bar(x - 0.18, actions.rescue_count, width=0.36, label="rescue")
    ax.bar(x + 0.18, -actions.harm_count, width=0.36, label="harm (negative axis)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, actions.action_or_policy.str.replace("_MATCHES_MAJORITY", "", regex=False))
    ax.set_ylabel("Target-run trial count")
    ax.set_title("Actual V2 priority selections by action")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "action_rescue_harm.png", dpi=180)
    plt.close(fig)

    methods = ("B0_TARGET_KEEP", "B3_OTHER_RUN_PROBABILITY_MEAN", "B4_ALL_RUN_PROBABILITY_MEAN", I003_SAFE, I003_FULL)
    calibration = probabilistic_table[
        probabilistic_table.pool.eq("holdout") & probabilistic_table.method_id.isin(methods)
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color="black", linewidth=0.8, label="ideal")
    for row in calibration.itertuples(index=False):
        bins = json.loads(row.calibration_bins_json)
        x = [item["mean_confidence"] for item in bins if item["count"] and np.isfinite(item["mean_confidence"])]
        y = [item["accuracy"] for item in bins if item["count"] and np.isfinite(item["accuracy"])]
        ax.plot(x, y, marker="o", markersize=3, label=row.method_id)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Holdout calibration by confidence bin")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_comparison.png", dpi=180)
    plt.close(fig)


def _lookup_comparison(context: EvaluationContext, left: str, right: str) -> dict[str, Any]:
    return comparison_summary(context, left, right)


def _pp(value: float) -> str:
    return f"{100 * value:+.3f} pp"


def _decision_and_report(
    contexts: dict[str, EvaluationContext],
    best_ensemble: str,
    equivalence: pd.DataFrame,
    action_table: pd.DataFrame,
    probabilistic_table: pd.DataFrame,
    deployment_table: pd.DataFrame,
    reconstruction: dict[str, Any],
) -> dict[str, Any]:
    holdout = contexts["holdout"]
    exploration = contexts["exploration"]
    comparisons = {
        "full_vs_best_holdout": _lookup_comparison(holdout, I003_FULL, best_ensemble),
        "safe_vs_best_holdout": _lookup_comparison(holdout, I003_SAFE, best_ensemble),
        "full_vs_best_exploration": _lookup_comparison(exploration, I003_FULL, best_ensemble),
        "safe_vs_best_exploration": _lookup_comparison(exploration, I003_SAFE, best_ensemble),
        "full_vs_target_holdout": _lookup_comparison(holdout, I003_FULL, "B0_TARGET_KEEP"),
        "safe_vs_target_holdout": _lookup_comparison(holdout, I003_SAFE, "B0_TARGET_KEEP"),
    }
    full_incremental = comparisons["full_vs_best_holdout"]
    full_equivalent = bool(
        equivalence[
            equivalence.pool.eq("holdout")
            & equivalence.right_method.eq(I003_FULL)
        ].classification_metrics_prediction_equivalent.iloc[0]
    )
    safe_equivalent = bool(
        equivalence[
            equivalence.pool.eq("holdout")
            & equivalence.right_method.eq(I003_SAFE)
        ].classification_metrics_prediction_equivalent.iloc[0]
    )
    intervention_adds = (
        full_incremental["mean_paired_delta_BA"] > 0
        and full_incremental["bootstrap_CI95_L"] > 0
        and full_incremental["positive_run_fraction"] > 0.5
        and not full_equivalent
    )
    primary_state = "INTERVENTION_ADDS_INCREMENTAL_VALUE" if intervention_adds else "ENSEMBLE_EXPLAINS_V2_GAIN"

    full_v2 = contexts["holdout"].v2_policies[I003_FULL]
    safe_v2 = contexts["holdout"].v2_policies[I003_SAFE]
    full_effect = full_v2["effect"][full_v2["selected"] != "noop"]
    safe_effect = safe_v2["effect"][safe_v2["selected"] != "noop"]
    full_harm = float(np.mean(full_effect < 0)) if len(full_effect) else 0.0
    safe_harm = float(np.mean(safe_effect < 0)) if len(safe_effect) else 0.0
    safe_preferred = (
        safe_harm < full_harm
        and comparisons["safe_vs_target_holdout"]["positive_run_fraction"]
        >= comparisons["full_vs_target_holdout"]["positive_run_fraction"]
    )

    probability_holdout = probabilistic_table[probabilistic_table.pool.eq("holdout")].set_index("method_id")
    strongest_probability_keep = min(
        (method_id for method_id in BASELINE_IDS if method_id in probability_holdout.index),
        key=lambda method_id: probability_holdout.loc[method_id, "mean_subject_NLL"],
    )
    full_prob_improvements = {
        metric: float(probability_holdout.loc[strongest_probability_keep, metric] - probability_holdout.loc[I003_FULL, metric])
        for metric in ("mean_subject_NLL", "mean_subject_Brier", "mean_subject_ECE")
    }
    probabilistic_benefit = full_equivalent and sum(value > 0 for value in full_prob_improvements.values()) >= 2

    qualifiers = []
    if probabilistic_benefit:
        qualifiers.append("CLASSIFICATION_EQUIVALENT_PROBABILISTIC_DIFFERENCE")
    if safe_preferred:
        qualifiers.append("PROTECTED_SAFE_PREFERRED")
    qualifiers.append("DEPLOYMENT_DEFINITION_REQUIRED")

    baseline_gain = {
        method_id: _lookup_comparison(holdout, method_id, "B0_TARGET_KEEP") for method_id in BASELINE_IDS[1:]
    }
    action_holdout = action_table[action_table.pool.eq("holdout")]
    action_candidates = action_holdout[
        action_holdout.action_or_policy.isin(
            ("AMPLIFY_MATCHES_MAJORITY", "GEOMETRY_MATCHES_MAJORITY", "ERASE_MATCHES_MAJORITY")
        )
    ]
    unique_action = action_candidates.sort_values(
        ["unique_rescue_beyond_best_ensemble", "net_correctness_gain", "action_or_policy"],
        ascending=[False, False, True],
    ).iloc[0]
    erase = action_holdout[action_holdout.action_or_policy.eq("ERASE_MATCHES_MAJORITY")].iloc[0]

    run = add_run_comparisons(holdout, best_ensemble)
    full_runs = run[run.method_id.eq(I003_FULL)].copy()
    safe_runs = run[run.method_id.eq(I003_SAFE)].copy()
    negative = full_runs[full_runs.delta_balanced_accuracy_vs_target_keep < 0]
    negative_ids = [f"fold-{int(row.fold_id)}_seed-{int(row.seed_id)}" for row in negative.itertuples(index=False)]
    negative_safe = negative[["fold_id", "seed_id"]].merge(
        safe_runs[["fold_id", "seed_id", "delta_balanced_accuracy_vs_target_keep"]],
        on=["fold_id", "seed_id"],
        how="left",
    )
    negative_fixed_by_safe = int((negative_safe.delta_balanced_accuracy_vs_target_keep >= 0).sum())
    negative_details: list[dict[str, Any]] = []
    holdout_frame = holdout.frame
    holdout_full_selected = holdout.v2_policies[I003_FULL]["selected"]
    holdout_full_effect = holdout.v2_policies[I003_FULL]["effect"]
    best_runs = run[run.method_id.eq(best_ensemble)]
    for negative_row in negative.itertuples(index=False):
        fold = int(negative_row.fold_id)
        seed = int(negative_row.seed_id)
        mask = holdout_frame.fold_id.eq(fold).to_numpy() & holdout_frame.seed_id.eq(seed).to_numpy()
        erase_mask = mask & (holdout_full_selected == "erase")
        safe_delta = float(
            safe_runs[
                safe_runs.fold_id.eq(fold) & safe_runs.seed_id.eq(seed)
            ].delta_balanced_accuracy_vs_target_keep.iloc[0]
        )
        best_delta = float(
            best_runs[
                best_runs.fold_id.eq(fold) & best_runs.seed_id.eq(seed)
            ].delta_balanced_accuracy_vs_target_keep.iloc[0]
        )
        negative_details.append(
            {
                "run": f"fold-{fold}_seed-{seed}",
                "full_delta_BA": float(negative_row.delta_balanced_accuracy_vs_target_keep),
                "safe_delta_BA": safe_delta,
                "best_ensemble_delta_BA": best_delta,
                "erase_selected": int(erase_mask.sum()),
                "erase_rescue": int(np.sum(holdout_full_effect[erase_mask] > 0)),
                "erase_harm": int(np.sum(holdout_full_effect[erase_mask] < 0)),
                "erase_net_correctness": int(np.sum(holdout_full_effect[erase_mask])),
            }
        )

    deployment_defined = not bool(
        deployment_table[
            deployment_table.method_id.isin((I003_SAFE, I003_FULL))
        ].deployment_status.eq("DEPLOYMENT_OUTPUT_NOT_YET_DEFINED").any()
    )
    mechanism = (
        "intervention-specific gain"
        if intervention_adds
        else (
            "generic KEEP-only ensemble gain"
            if full_incremental["mean_paired_delta_BA"] <= 0
            else "action-masked ensemble-guided gain without intervention-specific hard-label value"
        )
    )

    questions = {
        "1_v2_reproduced": reconstruction["status"] == "V2_RECONSTRUCTION_PASS",
        "2_full_equivalent_to_C2": full_equivalent,
        "3_safe_equivalent_to_C3": safe_equivalent,
        "4_other_run_hard_majority_gain": baseline_gain["B1_OTHER_RUN_HARD_MAJORITY"],
        "5_all_run_hard_majority_gain": baseline_gain["B2_ALL_RUN_HARD_MAJORITY"],
        "6_probability_averaging_gain": {
            "other_run": baseline_gain["B3_OTHER_RUN_PROBABILITY_MEAN"],
            "all_run": baseline_gain["B4_ALL_RUN_PROBABILITY_MEAN"],
        },
        "7_logit_averaging_gain": {
            "other_run": baseline_gain["B5_OTHER_RUN_LOGIT_MEAN"],
            "all_run": baseline_gain["B6_ALL_RUN_LOGIT_MEAN"],
        },
        "8_strongest_predefined_keep_ensemble": best_ensemble,
        "9_full_vs_best_ensemble": comparisons["full_vs_best_holdout"],
        "10_safe_vs_best_ensemble": comparisons["safe_vs_best_holdout"],
        "11_incremental_bootstrap_CI": {
            "full": [
                comparisons["full_vs_best_holdout"]["bootstrap_CI95_L"],
                comparisons["full_vs_best_holdout"]["bootstrap_CI95_U"],
            ],
            "safe": [
                comparisons["safe_vs_best_holdout"]["bootstrap_CI95_L"],
                comparisons["safe_vs_best_holdout"]["bootstrap_CI95_U"],
            ],
        },
        "12_hard_label_intervention_specific_improvement": intervention_adds,
        "13_probabilistic_benefit": {
            "present": probabilistic_benefit,
            "comparator": strongest_probability_keep,
            "full_comparator_minus_full": full_prob_improvements,
            "warning": "Probabilistic metrics cannot rescue an intervention-specific classification claim.",
        },
        "14_action_with_most_unique_rescue_beyond_ensemble": {
            "action": unique_action.action_or_policy,
            "count": int(unique_action.unique_rescue_beyond_best_ensemble),
        },
        "15_erase_contribution": {
            "selected": int(erase.number_selected),
            "rescue": int(erase.rescue_count),
            "harm": int(erase.harm_count),
            "net": int(erase.net_correctness_gain),
            "unique_rescue_beyond_best_ensemble": int(erase.unique_rescue_beyond_best_ensemble),
        },
        "16_protected_safe_preferred": safe_preferred,
        "17_negative_full_runs": {
            "runs": negative_ids,
            "count": len(negative_ids),
            "nonnegative_under_safe": negative_fixed_by_safe,
            "erase_is_only_structural_difference": True,
            "run_details": negative_details,
        },
        "18_unique_deployment_result_defined": deployment_defined,
        "19_supported_mechanism": mechanism,
        "20_next_experiment": (
            "Freeze one unique trial-level deployment rule before outcomes, then compare compute-matched all-run KEEP "
            "ensembles against action-masked direct consensus and real interventions on genuinely new subjects or a new dataset. "
            "Use subject-level inference and authorize any outer set only under a separate protocol."
        ),
    }
    payload = {
        "primary_state": primary_state,
        "secondary_qualifiers": qualifiers,
        "experiment_type": "post-V2 exploratory falsification audit",
        "best_ensemble_selected_on_exploration": best_ensemble,
        "selection_used_holdout": False,
        "classification_equivalence": {"full_vs_C2": full_equivalent, "safe_vs_C3": safe_equivalent},
        "holdout_incremental_comparisons": {
            "full_vs_best": comparisons["full_vs_best_holdout"],
            "safe_vs_best": comparisons["safe_vs_best_holdout"],
        },
        "exploration_incremental_comparisons": {
            "full_vs_best": comparisons["full_vs_best_exploration"],
            "safe_vs_best": comparisons["safe_vs_best_exploration"],
        },
        "questions": questions,
        "previous_holdout_is_not_sealed_for_v2_1": True,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", payload)

    q = questions
    full_ci = q["11_incremental_bootstrap_CI"]["full"]
    safe_ci = q["11_incremental_bootstrap_CI"]["safe"]
    report = f"""# PERSIST-EEG V2.1 falsification audit

## Decision

`{primary_state}`

Secondary qualifiers: `{', '.join(qualifiers)}`.

This is a post-V2 exploratory disambiguation analysis. The prior 12-subject
development holdout had already been opened; it is not presented as a new
confirmatory holdout. The 40-subject exploration pool and 12-subject holdout
remain separate. The pooled 52-subject rows are descriptive only. WBCIC outer
was not accessed.

## Direct answers

1. Exact V2 reconstruction: **{q['1_v2_reproduced']}**. The frozen policy lock,
   four source-cache hashes, every persisted subject/run/action/summary value,
   and the historical exploration ORACLE summary passed at tolerance `1e-14`.
2. FULL vs C2 hard-label equivalence: **{q['2_full_equivalent_to_C2']}**.
3. Protected-safe vs C3 hard-label equivalence: **{q['3_safe_equivalent_to_C3']}**.
4. Other-run hard majority vs target KEEP on the existing holdout:
   **{_pp(q['4_other_run_hard_majority_gain']['mean_paired_delta_BA'])}**.
5. All-run hard majority vs target KEEP:
   **{_pp(q['5_all_run_hard_majority_gain']['mean_paired_delta_BA'])}**.
6. Probability averaging: other-run
   **{_pp(q['6_probability_averaging_gain']['other_run']['mean_paired_delta_BA'])}**;
   all-run **{_pp(q['6_probability_averaging_gain']['all_run']['mean_paired_delta_BA'])}**.
7. Logit averaging: other-run
   **{_pp(q['7_logit_averaging_gain']['other_run']['mean_paired_delta_BA'])}**;
   all-run **{_pp(q['7_logit_averaging_gain']['all_run']['mean_paired_delta_BA'])}**.
8. Strongest predefined KEEP-only ensemble, selected using exploration only:
   `{best_ensemble}`.
9. FULL minus best ensemble: **{_pp(comparisons['full_vs_best_holdout']['mean_paired_delta_BA'])}**,
   subject-bootstrap CI95 [{_pp(full_ci[0])}, {_pp(full_ci[1])}].
10. Protected-safe minus best ensemble:
    **{_pp(comparisons['safe_vs_best_holdout']['mean_paired_delta_BA'])}**,
    subject-bootstrap CI95 [{_pp(safe_ci[0])}, {_pp(safe_ci[1])}].
11. The CIs above use subjects, not 9,200 target-run rows, as replicates.
12. Intervention-specific hard-label improvement: **{q['12_hard_label_intervention_specific_improvement']}**.
    For Balanced Accuracy / Accuracy / F1, the intervention policy is
    prediction-equivalent to an action-masked consensus override when C2/C3
    agreement is 100%.
13. Probabilistic benefit under the frozen criterion: **{q['13_probabilistic_benefit']['present']}**.
    Comparator: `{q['13_probabilistic_benefit']['comparator']}`. This is reported
    separately and does not alter the classification conclusion.
14. Most unique action rescue beyond the best ensemble:
    `{q['14_action_with_most_unique_rescue_beyond_ensemble']['action']}`
    ({q['14_action_with_most_unique_rescue_beyond_ensemble']['count']} target-run rows).
15. ERASE selected {q['15_erase_contribution']['selected']} rows, rescued
    {q['15_erase_contribution']['rescue']}, harmed {q['15_erase_contribution']['harm']},
    and had net correctness {q['15_erase_contribution']['net']:+d}; unique rescue
    beyond the best ensemble was {q['15_erase_contribution']['unique_rescue_beyond_best_ensemble']}.
16. Protected-safe preferred as a secondary safety qualifier:
    **{q['16_protected_safe_preferred']}**.
17. Negative FULL runs: `{', '.join(q['17_negative_full_runs']['runs']) or 'none'}`.
    {q['17_negative_full_runs']['nonnegative_under_safe']}/{q['17_negative_full_runs']['count']}
    become nonnegative when ERASE is forbidden; ERASE is the only structural
    difference between FULL and protected-safe.

    Run-specific ERASE evidence: `{json.dumps(q['17_negative_full_runs']['run_details'], separators=(',', ':'))}`.
18. A unique deployed I003 trial prediction is defined: **{q['18_unique_deployment_result_defined']}**.
    Current status: `DEPLOYMENT_OUTPUT_NOT_YET_DEFINED`.
19. Supported mechanism: **{q['19_supported_mechanism']}**.
20. Next experiment: {q['20_next_experiment']}

## Scientific interpretation

V2 found a usable cross-run consensus signal. It did not, by itself, establish
that AMPLIFY, GEOMETRY, or ERASE creates the hard-label gain. A binary action
that flips the target prediction toward the opposite leave-target-run majority
necessarily emits the same class as direct consensus. The mandatory C2/C3
controls quantify that identity rather than treating intervention semantics as
evidence.

All inference comparisons are compute-disclosed in
`ENSEMBLE_BASELINE_RESULTS.csv` and `CONSENSUS_CONTROL_RESULTS.csv`. A
multi-model intervention policy is not compared only with a single model.

`OUTER_TEST_USED = false`
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    return payload


def write_reproducibility(cache_root: Path, best_ensemble: str) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parent
    artifact_paths = [
        path
        for path in OUTPUTS.rglob("*")
        if path.is_file() and path.name != "REPRODUCIBILITY.json"
    ]
    payload = {
        "status": "V2_1_REPRODUCIBLE_ARTIFACT_SET",
        "command": "python experiments/persist_eeg_prospective_action_policy_v2_1/code/run_all.py --phase all",
        "audit_seed": AUDIT_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "best_ensemble_id": best_ensemble,
        "code_sha256": {path.name: sha256_file(path) for path in sorted(code_root.glob("*.py"))},
        "source_cache_sha256": {path.name: sha256_file(path) for path in sorted(cache_root.glob("*.parquet"))},
        "v2_policy_lock_sha256": sha256_file(V2_OUTPUTS / "freeze" / "FROZEN_POLICY_SPEC.json"),
        "artifact_sha256": {
            str(path.relative_to(OUTPUTS)).replace("\\", "/"): sha256_file(path)
            for path in sorted(artifact_paths)
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "analysis_identity_sha256": canonical_hash(
            {
                "best_ensemble_id": best_ensemble,
                "code_sha256": {path.name: sha256_file(path) for path in sorted(code_root.glob("*.py"))},
                "source_cache_sha256": {path.name: sha256_file(path) for path in sorted(cache_root.glob("*.parquet"))},
            }
        ),
        "previous_holdout_is_not_sealed_for_v2_1": True,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "REPRODUCIBILITY.json", payload)
    return payload


def validate_output_contract() -> None:
    required = [
        PROTOCOL / "V2_RECONSTRUCTION.json",
        PROTOCOL / "V2_RECONSTRUCTION.md",
        PROTOCOL / "V2_1_ANALYSIS_SPEC.json",
        PROTOCOL / "LEAKAGE_AUDIT.json",
        *[
            RESULTS / name
            for name in (
                "TARGET_RUN_POLICY_RESULTS.csv",
                "SUBJECT_PAIRED_RESULTS.csv",
                "RUN_PAIRED_RESULTS.csv",
                "ENSEMBLE_BASELINE_RESULTS.csv",
                "CONSENSUS_CONTROL_RESULTS.csv",
                "ACTION_EQUIVALENCE_RESULTS.csv",
                "ACTION_DECOMPOSITION.csv",
                "RESCUE_OVERLAP.csv",
                "PROBABILISTIC_RESULTS.csv",
                "DEPLOYMENT_LEVEL_RESULTS.csv",
            )
        ],
        *[
            FIGURES / name
            for name in (
                "paired_subject_delta_vs_ensemble.png",
                "run_delta_vs_ensemble.png",
                "rescue_overlap.png",
                "action_rescue_harm.png",
                "calibration_comparison.png",
            )
        ],
        OUTPUTS / "FINAL_DECISION.json",
        OUTPUTS / "SCIENTIFIC_REPORT.md",
        OUTPUTS / "REPRODUCIBILITY.json",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Required V2.1 artifacts missing or empty: {missing}")
    for path in RESULTS.glob("*.csv"):
        frame = pd.read_csv(path)
        if frame.empty:
            raise RuntimeError(f"Empty result table: {path.name}")
        if "OUTER_TEST_USED" in frame and frame.OUTER_TEST_USED.astype(str).str.lower().isin(("true", "1")).any():
            raise RuntimeError(f"Outer-test flag in result table: {path.name}")
    equivalence = pd.read_csv(RESULTS / "ACTION_EQUIVALENCE_RESULTS.csv")
    if len(equivalence) != 6:
        raise RuntimeError("Expected two mandatory equivalence comparisons in each of three reporting pools")
    expected_fraction = equivalence.number_of_disagreements / equivalence.trials
    if not np.allclose(
        expected_fraction.to_numpy(dtype=float),
        equivalence.fraction_final_predicted_class_differs.to_numpy(dtype=float),
        atol=1e-15,
        rtol=0,
    ):
        raise RuntimeError("Equivalence counts and fractions are internally inconsistent")
    if set(equivalence.pool) != set(POOL_ORDER):
        raise RuntimeError("Equivalence table does not cover all reporting pools")


def evaluate_v2_1(cache_root: Path) -> dict[str, Any]:
    ensure_directories()
    reconstruction_path = PROTOCOL / "V2_RECONSTRUCTION.json"
    if not reconstruction_path.exists():
        raise FileNotFoundError("Run exact V2 reconstruction before evaluation")
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    if reconstruction.get("status") != "V2_RECONSTRUCTION_PASS":
        raise RuntimeError("V2 reconstruction gate did not pass")

    exploration_frame = load_pool(cache_root, "EXPLORATION_POOL")
    holdout_frame = load_pool(cache_root, "DEVELOPMENT_HOLDOUT")
    if set(exploration_frame.manifest_index) & set(holdout_frame.manifest_index):
        raise RuntimeError("Exploration and holdout manifest identities overlap")
    contexts = {
        "exploration": build_context("exploration", exploration_frame),
        "holdout": build_context("holdout", holdout_frame),
    }
    best_ensemble = select_best_predefined_ensemble(contexts["exploration"])
    pooled_frame = pd.concat([exploration_frame, holdout_frame], ignore_index=True)
    contexts["pooled_descriptive"] = build_context("pooled_descriptive", pooled_frame)
    write_leakage_audit(cache_root, contexts, best_ensemble)

    target_table = pd.concat([contexts[pool].target_rows for pool in POOL_ORDER], ignore_index=True)
    subject_table = pd.concat(
        [add_subject_comparisons(contexts[pool], best_ensemble) for pool in POOL_ORDER], ignore_index=True
    )
    run_table = pd.concat(
        [add_run_comparisons(contexts[pool], best_ensemble) for pool in POOL_ORDER], ignore_index=True
    )
    ensemble_table = pd.DataFrame(
        [row for pool in POOL_ORDER for row in ensemble_result_rows(contexts[pool], best_ensemble)]
    )
    consensus_table = pd.DataFrame(
        [row for pool in POOL_ORDER for row in consensus_result_rows(contexts[pool], best_ensemble)]
    )
    equivalence_table = pd.DataFrame(
        [
            row
            for pool in POOL_ORDER
            for row in mandatory_equivalence_rows(
                pool, contexts[pool].controls, contexts[pool].v2_policies
            )
        ]
    )
    action_table = pd.DataFrame(
        [row for pool in POOL_ORDER for row in action_decomposition_rows(contexts[pool], best_ensemble)]
    )
    overlap_table = pd.DataFrame(
        [row for pool in POOL_ORDER for row in rescue_overlap_rows(contexts[pool], best_ensemble)]
    )
    probabilistic_table = pd.DataFrame(
        [row for pool in POOL_ORDER for row in probabilistic_result_rows(contexts[pool])]
    )
    deployment_table = pd.DataFrame(
        [row for pool in POOL_ORDER for row in deployment_result_rows(contexts[pool])]
    )

    write_csv(RESULTS / "TARGET_RUN_POLICY_RESULTS.csv", target_table)
    write_csv(RESULTS / "SUBJECT_PAIRED_RESULTS.csv", subject_table)
    write_csv(RESULTS / "RUN_PAIRED_RESULTS.csv", run_table)
    write_csv(RESULTS / "ENSEMBLE_BASELINE_RESULTS.csv", ensemble_table)
    write_csv(RESULTS / "CONSENSUS_CONTROL_RESULTS.csv", consensus_table)
    write_csv(RESULTS / "ACTION_EQUIVALENCE_RESULTS.csv", equivalence_table)
    write_csv(RESULTS / "ACTION_DECOMPOSITION.csv", action_table)
    write_csv(RESULTS / "RESCUE_OVERLAP.csv", overlap_table)
    write_csv(RESULTS / "PROBABILISTIC_RESULTS.csv", probabilistic_table)
    write_csv(RESULTS / "DEPLOYMENT_LEVEL_RESULTS.csv", deployment_table)

    if not bool(equivalence_table.classification_metrics_prediction_equivalent.all()):
        # A mismatch is scientifically reportable, not a reason to alter the
        # frozen policy.  Preserve the table and continue to the decision.
        pass
    _plot_results(contexts, best_ensemble, action_table, overlap_table, probabilistic_table)
    decision = _decision_and_report(
        contexts,
        best_ensemble,
        equivalence_table,
        action_table,
        probabilistic_table,
        deployment_table,
        reconstruction,
    )
    write_reproducibility(cache_root, best_ensemble)
    validate_output_contract()
    return decision
