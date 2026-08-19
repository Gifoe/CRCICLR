from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from build_unique_trials import load_all_target_rows
from v3_common import sigmoid


ACTION_PRIORITY = {"AMPLIFY": 0, "GEOMETRY": 1, "ERASE": 2}


def _entropy(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    return -(values * np.log(values) + (1 - values) * np.log(1 - values))


def _vote_entropy(prediction: np.ndarray) -> float:
    fraction = float(np.mean(prediction))
    return float(_entropy(np.asarray([fraction]))[0])


def assert_no_outcome_features(columns: list[str]) -> None:
    forbidden = ("label", "outcome", "correct", "rescue", "harm", "effect", "target_baseline_error")
    offenders = [column for column in columns if any(token in column.lower() for token in forbidden)]
    if offenders:
        raise RuntimeError(f"Outcome-dependent model features are forbidden: {offenders}")


def _aggregate_persist_features(target_rows: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    original = sorted(column for column in target_rows.columns if column.startswith("original_"))
    if not original:
        raise RuntimeError("No cross-fitted historical router diagnostics are available")
    assert_no_outcome_features(original)
    grouped = target_rows.groupby("manifest_index", sort=True)[original]
    mean = grouped.mean().add_prefix("persist_mean_")
    std = grouped.std(ddof=0).fillna(0.0).add_prefix("persist_std_")
    aggregate = pd.concat([mean, std], axis=1).reset_index()
    aggregate = trials[["manifest_index"]].merge(aggregate, on="manifest_index", validate="one_to_one")
    return aggregate.drop(columns="manifest_index")


def build_legal_residual_features(
    trials: pd.DataFrame,
    action_candidates: pd.DataFrame,
    cache_root,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, list[str]]]:
    """Build one legal row per unique trial and global action candidate."""
    target_rows = load_all_target_rows(cache_root)
    persist = _aggregate_persist_features(target_rows, trials)
    candidates = action_candidates[action_candidates.action_scope.eq("GLOBAL")].copy()
    candidates["action_priority"] = candidates.action_family.map(ACTION_PRIORITY)
    candidates = candidates.sort_values(["trial_index", "action_priority"]).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        trial = trials.iloc[int(candidate.trial_index)]
        keep_margin = np.asarray(trial.keep_margins, dtype=float)
        keep_probability = sigmoid(keep_margin)
        keep_prediction = (keep_margin >= 0).astype(int)
        action = str(candidate.action_family).lower()
        action_margin = np.asarray(trial[f"{action}_margins"], dtype=float)
        action_probability = sigmoid(action_margin)
        action_prediction = (action_margin >= 0).astype(int)
        b6_margin = float(trial.z_keep_ens)
        b6_probability = float(trial.p_keep_ens)
        record = {
            "trial_index": int(candidate.trial_index),
            "trial_uid": trial.trial_uid,
            "subject_id": str(trial.subject_id),
            "session_id": str(trial.session_id),
            "source_pool": str(trial.source_pool),
            "action_family": str(candidate.action_family),
            "action_priority": int(ACTION_PRIORITY[str(candidate.action_family)]),
            "action_prediction": int(candidate.prediction),
            "action_probability": float(candidate.probability),
            "action_margin": float(candidate.margin),
            "b6_prediction": int(trial.y_keep_ens),
            "b6_probability": b6_probability,
            "b6_margin": b6_margin,
            "outcome_label": int(trial.outcome_label),
            "target_rescue": int(
                trial.y_keep_ens != trial.outcome_label and int(candidate.prediction) == trial.outcome_label
            ),
            "target_harm": int(
                trial.y_keep_ens == trial.outcome_label and int(candidate.prediction) != trial.outcome_label
            ),
            "base_abs_margin": abs(b6_margin),
            "base_entropy": float(_entropy(np.asarray([b6_probability]))[0]),
            "base_run_margin_std": float(keep_margin.std()),
            "base_run_probability_std": float(keep_probability.std()),
            "base_disagreeing_run_count": int(np.sum(keep_prediction != trial.y_keep_ens)),
            "base_vote_fraction_class1": float(keep_prediction.mean()),
            "base_vote_entropy": _vote_entropy(keep_prediction),
            "base_majority_strength": float(abs(keep_prediction.mean() - 0.5) * 2),
            "base_min_margin": float(keep_margin.min()),
            "base_max_margin": float(keep_margin.max()),
            "base_margin_range": float(np.ptp(keep_margin)),
            "base_min_probability": float(keep_probability.min()),
            "base_max_probability": float(keep_probability.max()),
            "n_runs": int(trial.n_runs),
            "action_abs_margin": abs(float(candidate.margin)),
            "action_entropy": float(_entropy(np.asarray([candidate.probability]))[0]),
            "action_margin_movement": float(candidate.margin - b6_margin),
            "action_probability_movement": float(candidate.probability - b6_probability),
            "action_boundary_cross": int(candidate.prediction != trial.y_keep_ens),
            "action_vs_b6_disagreement": int(candidate.prediction != trial.y_keep_ens),
            "action_run_margin_std": float(action_margin.std()),
            "action_run_probability_std": float(action_probability.std()),
            "action_vote_fraction_class1": float(action_prediction.mean()),
            "action_vote_entropy": _vote_entropy(action_prediction),
            "action_majority_strength": float(abs(action_prediction.mean() - 0.5) * 2),
            "action_min_margin": float(action_margin.min()),
            "action_max_margin": float(action_margin.max()),
            "action_margin_range": float(np.ptp(action_margin)),
            "movement_run_std": float((action_margin - keep_margin).std()),
            "movement_run_min": float((action_margin - keep_margin).min()),
            "movement_run_max": float((action_margin - keep_margin).max()),
            "movement_sign_agreement": float(np.mean(np.sign(action_margin - keep_margin) == np.sign(candidate.margin - b6_margin))),
            "is_amplify": int(candidate.action_family == "AMPLIFY"),
            "is_geometry": int(candidate.action_family == "GEOMETRY"),
            "is_erase": int(candidate.action_family == "ERASE"),
            "protected_safe": int(candidate.action_family != "ERASE"),
            "OUTER_TEST_USED": False,
        }
        rows.append(record)
    features = pd.DataFrame(rows)
    persist_repeated = persist.iloc[features.trial_index.to_numpy(dtype=int)].reset_index(drop=True)
    features = pd.concat([features, persist_repeated], axis=1)
    if len(features) != 3 * len(trials):
        raise RuntimeError("Expected exactly three global action rows per unique trial")
    if features.duplicated(["trial_index", "action_family"]).any():
        raise RuntimeError("Duplicate global action feature row")

    movement_features = [
        column
        for column in features.columns
        if column.startswith(("base_", "action_", "movement_", "is_"))
        and pd.api.types.is_numeric_dtype(features[column])
        and column
        not in (
            "action_prediction",
            "action_probability",
            "action_margin",
            "action_priority",
        )
    ] + ["b6_probability", "b6_margin", "n_runs", "protected_safe"]
    movement_features = sorted(set(movement_features))
    persist_features = sorted(column for column in features.columns if column.startswith("persist_"))
    full_features = sorted(set(movement_features + persist_features))
    assert_no_outcome_features(movement_features)
    assert_no_outcome_features(full_features)
    non_numeric = [
        column for column in full_features if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric:
        raise RuntimeError(f"Non-numeric legal model features: {non_numeric}")
    features[full_features] = features[full_features].replace([np.inf, -np.inf], np.nan)
    categories = {
        "movement": movement_features,
        "persist_all": persist_features,
        "persist_protected": [column for column in persist_features if "protected_" in column],
        "persist_decision_dependence": [
            column
            for column in persist_features
            if any(token in column for token in ("agreement", "margin_change", "entropy_change", "kl_", "js_", "probability_shift"))
        ],
        "persist_trust_geometry": [
            column
            for column in persist_features
            if any(token in column for token in ("distance", "projection", "geometry", "contribution_norm"))
        ],
    }
    return features, movement_features, full_features, categories
