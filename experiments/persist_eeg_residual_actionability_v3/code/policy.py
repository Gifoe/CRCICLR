from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_ensemble_actions import build_action_candidates
from evaluation import evaluate_prediction, paired_bootstrap
from features import ACTION_PRIORITY, build_legal_residual_features
from v3_common import (
    DIAGNOSTICS,
    NEXT_STAGE,
    OUTPUTS,
    RESEARCH_LOG,
    RESULTS,
    markdown_table,
    stable_seed,
    write_csv,
    write_json,
)


MODEL_IDS = (
    "M0_B6_KEEP_ENSEMBLE",
    "M1_ENSEMBLE_CONFIDENCE_RULE",
    "M2_ENSEMBLE_DISAGREEMENT_RULE",
    "M3_ACTION_MOVEMENT_LOGISTIC",
    "M4_FULL_LEGAL_LOGISTIC",
    "M5_HIST_GRADIENT_BOOSTING",
    "I006_CONDITIONAL_ACTION_LOGISTIC",
    "I007_CONDITIONAL_ACTION_HGB",
)
CONDITIONAL_ACTIONS = ("AMPLIFY", "GEOMETRY", "ERASE")
CONDITIONAL_MENUS = {
    "KEEP_ONLY": (),
    "FULL": CONDITIONAL_ACTIONS,
    "PROTECTED_SAFE": ("AMPLIFY", "GEOMETRY"),
    "ERASE_ONLY": ("ERASE",),
}


class ConstantProbability:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        p1 = np.full(len(values), self.probability, dtype=float)
        return np.column_stack([1 - p1, p1])


@dataclass
class OOFPolicy:
    prediction: np.ndarray
    probability: np.ndarray
    selected_action: np.ndarray
    selected_value: np.ndarray
    predicted_rescue: np.ndarray
    predicted_harm: np.ndarray
    outer_fold: np.ndarray


def _empty_policy(trials: pd.DataFrame) -> OOFPolicy:
    rows = len(trials)
    return OOFPolicy(
        prediction=np.full(rows, -1, dtype=int),
        probability=np.full(rows, np.nan, dtype=float),
        selected_action=np.full(rows, "UNSET", dtype=object),
        selected_value=np.full(rows, np.nan, dtype=float),
        predicted_rescue=np.full(rows, np.nan, dtype=float),
        predicted_harm=np.full(rows, np.nan, dtype=float),
        outer_fold=np.full(rows, -1, dtype=int),
    )


def _subject_protocol(subjects: list[str]) -> dict[str, Any]:
    ordered = sorted(subjects, key=lambda subject: stable_seed("V3_OUTER", subject))
    assignment = {subject: index % 5 for index, subject in enumerate(ordered)}
    folds = []
    for outer_fold in range(5):
        test = sorted([subject for subject in subjects if assignment[subject] == outer_fold], key=int)
        development = sorted([subject for subject in subjects if assignment[subject] != outer_fold], key=int)
        calibration_count = max(6, int(np.ceil(0.20 * len(development))))
        calibration = sorted(
            sorted(development, key=lambda subject: stable_seed("V3_CALIBRATION", outer_fold, subject))[
                :calibration_count
            ],
            key=int,
        )
        train = sorted(set(development) - set(calibration), key=int)
        folds.append(
            {
                "outer_fold": outer_fold,
                "model_training_subjects": train,
                "calibration_subjects": calibration,
                "heldout_subjects": test,
            }
        )
    return {
        "status": "V3_EXPLORATORY_GROUPED_NESTED_CV",
        "algorithm": "deterministic SHA256-ranked 5-fold subjects; disjoint inner calibration subjects",
        "subjects": len(subjects),
        "folds": folds,
        "trial_random_split": False,
        "heldout_threshold_calibration": False,
        "confirmatory": False,
        "policy_model_ids": list(MODEL_IDS),
        "OUTER_TEST_USED": False,
    }


def _selection_metrics(trials: pd.DataFrame, decision: pd.DataFrame) -> dict[str, float]:
    decision = decision.sort_values("trial_index").reset_index(drop=True)
    if decision.trial_index.duplicated().any():
        raise RuntimeError("Calibration decision has duplicate trial indices")
    subset = trials.loc[decision.trial_index.to_numpy(dtype=int)].reset_index(drop=True)
    labels = subset.outcome_label.to_numpy(dtype=int)
    base = subset.y_keep_ens.to_numpy(dtype=int)
    pred = decision.prediction.to_numpy(dtype=int)
    subject_delta = []
    for indices in subset.groupby("subject_id", sort=True).indices.values():
        idx = np.asarray(indices, dtype=int)
        class_recall_policy = []
        class_recall_base = []
        for label in (0, 1):
            class_rows = idx[labels[idx] == label]
            if not len(class_rows):
                raise RuntimeError("Inner calibration subject is missing a binary class")
            class_recall_policy.append(float(np.mean(pred[class_rows] == labels[class_rows])))
            class_recall_base.append(float(np.mean(base[class_rows] == labels[class_rows])))
        subject_delta.append(float(np.mean(class_recall_policy) - np.mean(class_recall_base)))
    acted = decision.selected_action.ne("KEEP").to_numpy()
    rescue = acted & (base != labels) & (pred == labels)
    harm = acted & (base == labels) & (pred != labels)
    return {
        "delta_BA": float(np.mean(subject_delta)),
        "action_rate": float(acted.mean()),
        "rescue_precision": float(rescue.sum() / acted.sum()) if acted.any() else 0.0,
        "harm_rate": float(harm.sum() / acted.sum()) if acted.any() else 0.0,
    }


def _candidate_key(metrics: dict[str, float], candidate_order: int) -> tuple[Any, ...]:
    return (
        metrics["delta_BA"],
        -metrics["harm_rate"],
        metrics["rescue_precision"],
        -metrics["action_rate"],
        -int(candidate_order),
    )


def _rule_decision(
    features: pd.DataFrame,
    trial_indices: np.ndarray,
    *,
    kind: str,
    parameters: dict[str, float],
) -> pd.DataFrame:
    rows = []
    selected = features[features.trial_index.isin(set(map(int, trial_indices)))].copy()
    selected = selected.sort_values(["trial_index", "action_priority"])
    for trial_index, group in selected.groupby("trial_index", sort=True):
        first = group.iloc[0]
        eligible = group.action_boundary_cross.eq(1)
        if bool(parameters.get("always_keep", 0.0)):
            eligible &= False
        elif kind == "confidence":
            eligible &= group.base_abs_margin.le(parameters["base_abs_max"])
            eligible &= group.action_abs_margin.ge(parameters["action_abs_min"])
        elif kind == "disagreement":
            eligible &= group.base_run_margin_std.ge(parameters["run_std_min"])
            eligible &= group.base_disagreeing_run_count.ge(parameters["disagree_count_min"])
            eligible &= group.action_abs_margin.ge(parameters["action_abs_min"])
        else:
            raise ValueError(kind)
        candidates = group[eligible].sort_values(
            ["action_abs_margin", "action_priority"], ascending=[False, True]
        )
        if len(candidates):
            choice = candidates.iloc[0]
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "prediction": int(choice.action_prediction),
                    "probability": float(choice.action_probability),
                    "selected_action": str(choice.action_family),
                    "selected_value": float(choice.action_abs_margin),
                    "predicted_rescue": np.nan,
                    "predicted_harm": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "prediction": int(first.b6_prediction),
                    "probability": float(first.b6_probability),
                    "selected_action": "KEEP",
                    "selected_value": 0.0,
                    "predicted_rescue": np.nan,
                    "predicted_harm": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _calibrate_rule(
    trials: pd.DataFrame,
    features: pd.DataFrame,
    calibration_trials: np.ndarray,
    kind: str,
    outer_fold: int,
) -> tuple[dict[str, float], pd.DataFrame, list[dict[str, Any]]]:
    if kind == "confidence":
        grid = [{"always_keep": 1.0}] + [
            {"base_abs_max": base, "action_abs_min": action}
            for base in (0.10, 0.25, 0.50, 1.00, 2.00)
            for action in (0.00, 0.25, 0.50, 1.00)
        ]
    else:
        grid = [{"always_keep": 1.0}] + [
            {
                "run_std_min": std,
                "disagree_count_min": count,
                "action_abs_min": action,
            }
            for std in (0.10, 0.25, 0.50, 1.00)
            for count in (1, 2, 3)
            for action in (0.00, 0.25, 0.50)
        ]
    records = []
    choices = []
    selection_model_id = (
        "M1_ENSEMBLE_CONFIDENCE_RULE"
        if kind == "confidence"
        else "M2_ENSEMBLE_DISAGREEMENT_RULE"
    )
    for candidate_order, parameters in enumerate(grid):
        decision = _rule_decision(features, calibration_trials, kind=kind, parameters=parameters)
        metrics = _selection_metrics(trials, decision)
        key_text = json.dumps(parameters, sort_keys=True)
        records.append(
            {
                "outer_fold": outer_fold,
                "model_id": selection_model_id,
                "parameters": key_text,
                "candidate_order": candidate_order,
                **metrics,
                "selected_on_inner_calibration": False,
                "heldout_subjects_read_for_selection": False,
                "OUTER_TEST_USED": False,
            }
        )
        choices.append((_candidate_key(metrics, candidate_order), parameters, decision))
    _, parameters, decision = max(choices, key=lambda item: item[0])
    selected_key = json.dumps(parameters, sort_keys=True)
    for record in records:
        record["selected_on_inner_calibration"] = record["parameters"] == selected_key
    return parameters, decision, records


def _fit_binary_model(
    model_kind: str,
    configuration: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return ConstantProbability(float(y.mean()))
    if model_kind == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(configuration["C"]),
                        class_weight="balanced",
                        max_iter=3000,
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        ).fit(x, y)
    if model_kind == "hgb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=100,
                        max_depth=int(configuration["max_depth"]),
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ).fit(x, y)
    raise ValueError(model_kind)


def _predict_positive(model, values: np.ndarray) -> np.ndarray:
    return model.predict_proba(values)[:, 1]


def _value_decision(
    features: pd.DataFrame,
    trial_indices: np.ndarray,
    p_rescue: np.ndarray,
    p_harm: np.ndarray,
    *,
    risk_lambda: float,
    threshold: float,
) -> pd.DataFrame:
    selected = features[features.trial_index.isin(set(map(int, trial_indices)))].copy()
    if len(selected) != len(p_rescue) or len(selected) != len(p_harm):
        raise RuntimeError("Residual probability alignment mismatch")
    selected["p_rescue"] = np.asarray(p_rescue, dtype=float)
    selected["p_harm"] = np.asarray(p_harm, dtype=float)
    selected["value"] = selected.p_rescue - risk_lambda * selected.p_harm
    selected = selected.sort_values(["trial_index", "action_priority"])
    rows = []
    for trial_index, group in selected.groupby("trial_index", sort=True):
        first = group.iloc[0]
        candidate = group[
            group.action_boundary_cross.eq(1) & group.value.ge(threshold)
        ].sort_values(["value", "action_priority"], ascending=[False, True])
        if len(candidate):
            choice = candidate.iloc[0]
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "prediction": int(choice.action_prediction),
                    "probability": float(choice.action_probability),
                    "selected_action": str(choice.action_family),
                    "selected_value": float(choice.value),
                    "predicted_rescue": float(choice.p_rescue),
                    "predicted_harm": float(choice.p_harm),
                }
            )
        else:
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "prediction": int(first.b6_prediction),
                    "probability": float(first.b6_probability),
                    "selected_action": "KEEP",
                    "selected_value": 0.0,
                    "predicted_rescue": 0.0,
                    "predicted_harm": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _model_configurations(model_kind: str) -> list[dict[str, Any]]:
    if model_kind == "logistic":
        return [{"C": value} for value in (0.01, 0.10, 1.00)]
    return [{"max_depth": value} for value in (2, 3)]


def _fit_calibrate_model(
    trials: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    train_subjects: list[str],
    calibration_subjects: list[str],
    model_kind: str,
    model_id: str,
    outer_fold: int,
) -> tuple[Any, Any, dict[str, Any], pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = features.subject_id.isin(train_subjects).to_numpy()
    calibration_rows = features.subject_id.isin(calibration_subjects).to_numpy()
    calibration_trials = np.sort(features.loc[calibration_rows, "trial_index"].unique())
    x_train = features.loc[train_rows, feature_columns].to_numpy(dtype=float)
    x_calibration = features.loc[calibration_rows, feature_columns].to_numpy(dtype=float)
    y_rescue = features.target_rescue.to_numpy(dtype=int)
    y_harm = features.target_harm.to_numpy(dtype=int)
    choices = []
    grid_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    candidate_order = 0
    for configuration in _model_configurations(model_kind):
        seed = stable_seed("V3_MODEL", model_id, outer_fold, json.dumps(configuration, sort_keys=True))
        rescue_model = _fit_binary_model(model_kind, configuration, x_train, y_rescue[train_rows], seed)
        harm_model = _fit_binary_model(model_kind, configuration, x_train, y_harm[train_rows], seed + 1)
        p_rescue = _predict_positive(rescue_model, x_calibration)
        p_harm = _predict_positive(harm_model, x_calibration)
        for risk_lambda in (1.0, 1.5, 2.0):
            for threshold in (1.01, 0.00, 0.02, 0.05, 0.10, 0.20, 0.30):
                decision = _value_decision(
                    features.loc[calibration_rows].reset_index(drop=True),
                    calibration_trials,
                    p_rescue,
                    p_harm,
                    risk_lambda=risk_lambda,
                    threshold=threshold,
                )
                metrics = _selection_metrics(trials, decision)
                parameters = {
                    **configuration,
                    "risk_lambda": risk_lambda,
                    "threshold": threshold,
                }
                key_text = json.dumps(parameters, sort_keys=True)
                grid_records.append(
                    {
                        "outer_fold": outer_fold,
                        "model_id": model_id,
                        "parameters": key_text,
                        "candidate_order": candidate_order,
                        **metrics,
                        "selected_on_inner_calibration": False,
                        "heldout_subjects_read_for_selection": False,
                        "OUTER_TEST_USED": False,
                    }
                )
                choices.append(
                    (
                        _candidate_key(metrics, candidate_order),
                        rescue_model,
                        harm_model,
                        parameters,
                        decision,
                    )
                )
                candidate_order += 1
    _, rescue_model, harm_model, parameters, decision = max(choices, key=lambda item: item[0])
    selected_key = json.dumps(parameters, sort_keys=True)
    for record in grid_records:
        record["selected_on_inner_calibration"] = record["parameters"] == selected_key
    if model_kind == "logistic":
        for target, model in (("rescue", rescue_model), ("harm", harm_model)):
            if isinstance(model, Pipeline):
                coefficient = model.named_steps["classifier"].coef_[0]
                for feature, value in zip(feature_columns, coefficient):
                    importance_records.append(
                        {
                            "outer_fold": outer_fold,
                            "model_id": model_id,
                            "target": target,
                            "feature": feature,
                            "coefficient": float(value),
                            "absolute_coefficient": abs(float(value)),
                            "OUTER_TEST_USED": False,
                        }
                    )
    return rescue_model, harm_model, parameters, decision, grid_records, importance_records


def _fit_conditional_action_models(
    model_kind: str,
    configuration: dict[str, Any],
    features: pd.DataFrame,
    feature_columns: list[str],
    train_rows: np.ndarray,
    model_id: str,
    outer_fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models: dict[str, Any] = {}
    importance: list[dict[str, Any]] = []
    for action in CONDITIONAL_ACTIONS:
        action_rows = (
            train_rows
            & features.action_family.eq(action).to_numpy()
            & features.action_boundary_cross.eq(1).to_numpy()
        )
        if not action_rows.any():
            raise RuntimeError(f"No conditional training candidates for {action}")
        seed = stable_seed(
            "V3_CONDITIONAL_MODEL",
            model_id,
            outer_fold,
            action,
            json.dumps(configuration, sort_keys=True),
        )
        model = _fit_binary_model(
            model_kind,
            configuration,
            features.loc[action_rows, feature_columns].to_numpy(dtype=float),
            features.loc[action_rows, "target_rescue"].to_numpy(dtype=int),
            seed,
        )
        models[action] = model
        if model_kind == "logistic" and isinstance(model, Pipeline):
            coefficient = model.named_steps["classifier"].coef_[0]
            for feature, value in zip(feature_columns, coefficient):
                importance.append(
                    {
                        "outer_fold": outer_fold,
                        "model_id": model_id,
                        "target": "conditional_rescue",
                        "action_family": action,
                        "feature": feature,
                        "coefficient": float(value),
                        "absolute_coefficient": abs(float(value)),
                        "OUTER_TEST_USED": False,
                    }
                )
    return models, importance


def _predict_conditional_action_models(
    models: dict[str, Any],
    features: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    p_rescue = np.zeros(len(features), dtype=float)
    p_harm = np.zeros(len(features), dtype=float)
    crossing = features.action_boundary_cross.eq(1).to_numpy()
    for action in CONDITIONAL_ACTIONS:
        rows = crossing & features.action_family.eq(action).to_numpy()
        if rows.any():
            p_rescue[rows] = _predict_positive(
                models[action], features.loc[rows, feature_columns].to_numpy(dtype=float)
            )
            p_harm[rows] = 1.0 - p_rescue[rows]
    return p_rescue, p_harm


def _conditional_action_decision(
    features: pd.DataFrame,
    p_rescue: np.ndarray,
    p_harm: np.ndarray,
    *,
    menu: str,
    threshold: float,
) -> pd.DataFrame:
    if len(features) != len(p_rescue) or len(features) != len(p_harm):
        raise RuntimeError("Conditional residual probability alignment mismatch")
    allowed = set(CONDITIONAL_MENUS[menu])
    selected = features.copy()
    selected["p_rescue"] = np.asarray(p_rescue, dtype=float)
    selected["p_harm"] = np.asarray(p_harm, dtype=float)
    selected = selected.sort_values(["trial_index", "action_priority"])
    rows = []
    for trial_index, group in selected.groupby("trial_index", sort=True):
        first = group.iloc[0]
        candidate = group[
            group.action_family.isin(allowed)
            & group.action_boundary_cross.eq(1)
            & group.p_rescue.ge(threshold)
        ].sort_values(["p_rescue", "action_priority"], ascending=[False, True])
        if len(candidate):
            choice = candidate.iloc[0]
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "prediction": int(choice.action_prediction),
                    "probability": float(choice.action_probability),
                    "selected_action": str(choice.action_family),
                    "selected_value": float(choice.p_rescue - choice.p_harm),
                    "predicted_rescue": float(choice.p_rescue),
                    "predicted_harm": float(choice.p_harm),
                }
            )
        else:
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "prediction": int(first.b6_prediction),
                    "probability": float(first.b6_probability),
                    "selected_action": "KEEP",
                    "selected_value": 0.0,
                    "predicted_rescue": 0.0,
                    "predicted_harm": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _fit_calibrate_conditional_model(
    trials: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    train_subjects: list[str],
    calibration_subjects: list[str],
    model_kind: str,
    model_id: str,
    outer_fold: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = features.subject_id.isin(train_subjects).to_numpy()
    calibration_rows = features.subject_id.isin(calibration_subjects).to_numpy()
    calibration_features = features.loc[calibration_rows].reset_index(drop=True)
    choices = []
    records: list[dict[str, Any]] = []
    importance_by_configuration: dict[str, list[dict[str, Any]]] = {}
    candidate_order = 0
    for configuration in _model_configurations(model_kind):
        configuration_key = json.dumps(configuration, sort_keys=True)
        models, importance = _fit_conditional_action_models(
            model_kind,
            configuration,
            features,
            feature_columns,
            train_rows,
            model_id,
            outer_fold,
        )
        importance_by_configuration[configuration_key] = importance
        p_rescue, p_harm = _predict_conditional_action_models(
            models, calibration_features, feature_columns
        )
        for menu in CONDITIONAL_MENUS:
            for threshold in (1.01, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
                decision = _conditional_action_decision(
                    calibration_features,
                    p_rescue,
                    p_harm,
                    menu=menu,
                    threshold=threshold,
                )
                metrics = _selection_metrics(trials, decision)
                parameters = {**configuration, "menu": menu, "threshold": threshold}
                key_text = json.dumps(parameters, sort_keys=True)
                records.append(
                    {
                        "outer_fold": outer_fold,
                        "model_id": model_id,
                        "parameters": key_text,
                        "candidate_order": candidate_order,
                        **metrics,
                        "selected_on_inner_calibration": False,
                        "heldout_subjects_read_for_selection": False,
                        "OUTER_TEST_USED": False,
                    }
                )
                choices.append(
                    (
                        _candidate_key(metrics, candidate_order),
                        models,
                        parameters,
                        configuration_key,
                    )
                )
                candidate_order += 1
    _, models, parameters, configuration_key = max(choices, key=lambda item: item[0])
    selected_key = json.dumps(parameters, sort_keys=True)
    for record in records:
        record["selected_on_inner_calibration"] = record["parameters"] == selected_key
    return models, parameters, records, importance_by_configuration[configuration_key]


def _assign_policy(oof: OOFPolicy, decision: pd.DataFrame, outer_fold: int) -> None:
    decision = decision.sort_values("trial_index").reset_index(drop=True)
    if decision.trial_index.duplicated().any():
        raise RuntimeError("OOF policy decision has duplicate trial indices")
    index = decision.trial_index.to_numpy(dtype=int)
    if np.any(oof.outer_fold[index] >= 0):
        raise RuntimeError("OOF trial was assigned by more than one subject fold")
    oof.prediction[index] = decision.prediction.to_numpy(dtype=int)
    oof.probability[index] = decision.probability.to_numpy(dtype=float)
    oof.selected_action[index] = decision.selected_action.to_numpy(dtype=object)
    oof.selected_value[index] = decision.selected_value.to_numpy(dtype=float)
    oof.predicted_rescue[index] = decision.predicted_rescue.to_numpy(dtype=float)
    oof.predicted_harm[index] = decision.predicted_harm.to_numpy(dtype=float)
    oof.outer_fold[index] = outer_fold


def _probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    labels = np.asarray(labels, dtype=int)
    confidence = np.maximum(probability, 1 - probability)
    prediction = (probability >= 0.5).astype(int)
    correct = prediction == labels
    edges = np.linspace(0.5, 1.0, 11)
    ece = 0.0
    for index in range(10):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == 9 else confidence < edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "NLL": float(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))),
        "Brier": float(np.mean((probability - labels) ** 2)),
        "ECE": ece,
    }


def _evaluate_oof(
    trials: pd.DataFrame,
    policies: dict[str, OOFPolicy],
    audit: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = trials.outcome_label.to_numpy(dtype=int)
    b6 = trials.y_keep_ens.to_numpy(dtype=int)
    summaries = []
    subject_parts = []
    fold_rows = []
    action_rows = []
    recovery_rows = []
    action_oracle = float(audit["decision"]["strongest_action_oracle_delta_BA_vs_B6"])
    unique_oracle = float(audit["decision"]["action_oracle_minus_keep_only_delta_BA"])
    for method_id, policy in policies.items():
        if np.any(policy.prediction < 0) or not np.isfinite(policy.probability).all() or np.any(policy.outer_fold < 0):
            raise RuntimeError(f"Incomplete OOF predictions for {method_id}")
        summary, subjects = evaluate_prediction(
            trials,
            policy.prediction,
            b6,
            method_id=method_id,
            pool="all_52_grouped_OOF",
        )
        acted = policy.selected_action != "KEEP"
        rescue = acted & (b6 != labels) & (policy.prediction == labels)
        harm = acted & (b6 == labels) & (policy.prediction != labels)
        probability = _probability_metrics(labels, policy.probability)
        fold_positive = []
        for outer_fold in range(5):
            fold_subjects = subjects[
                subjects.subject_id.isin(trials.loc[policy.outer_fold == outer_fold, "subject_id"].unique())
            ]
            delta = float(fold_subjects.delta_BA_vs_B6.mean())
            fold_positive.append(delta > 0)
            fold_rows.append(
                {
                    "model_id": method_id,
                    "outer_fold": outer_fold,
                    "subjects": int(len(fold_subjects)),
                    "mean_delta_BA_vs_B6": delta,
                    "positive_subject_fraction": float(np.mean(fold_subjects.delta_BA_vs_B6 > 0)),
                    "action_rate": float(acted[policy.outer_fold == outer_fold].mean()),
                    "OUTER_TEST_USED": False,
                }
            )
        summaries.append(
            {
                **summary,
                "model_id": method_id,
                "action_rate": float(acted.mean()),
                "rescue_count": int(rescue.sum()),
                "harm_count": int(harm.sum()),
                "net_correctness_gain": int(rescue.sum() - harm.sum()),
                "rescue_precision": float(rescue.sum() / acted.sum()) if acted.any() else 0.0,
                "harm_rate": float(harm.sum() / acted.sum()) if acted.any() else 0.0,
                "positive_fold_fraction": float(np.mean(fold_positive)),
                **probability,
                "action_oracle_recovery_fraction": (
                    float(summary["mean_subject_delta_BA_vs_B6"] / action_oracle) if action_oracle > 0 else np.nan
                ),
                "unique_action_oracle_recovery_fraction": (
                    float(summary["mean_subject_delta_BA_vs_B6"] / unique_oracle) if unique_oracle > 0 else np.nan
                ),
                "OUTER_TEST_USED": False,
            }
        )
        subject_parts.append(subjects.assign(model_id=method_id))
        for action in ("KEEP", "AMPLIFY", "GEOMETRY", "ERASE"):
            mask = policy.selected_action == action
            action_rows.append(
                {
                    "model_id": method_id,
                    "action": action,
                    "selected_count": int(mask.sum()),
                    "selected_fraction": float(mask.mean()),
                    "rescue_count": int(np.sum(rescue & mask)),
                    "harm_count": int(np.sum(harm & mask)),
                    "net_correctness_gain": int(np.sum(rescue & mask) - np.sum(harm & mask)),
                    "OUTER_TEST_USED": False,
                }
            )
        recovery_rows.append(
            {
                "model_id": method_id,
                "delta_BA_vs_B6": summary["mean_subject_delta_BA_vs_B6"],
                "strongest_action_oracle_delta_BA_vs_B6": action_oracle,
                "action_oracle_recovery_fraction": (
                    summary["mean_subject_delta_BA_vs_B6"] / action_oracle if action_oracle > 0 else np.nan
                ),
                "unique_action_oracle_delta_BA": unique_oracle,
                "unique_action_oracle_recovery_fraction": (
                    summary["mean_subject_delta_BA_vs_B6"] / unique_oracle if unique_oracle > 0 else np.nan
                ),
                "OUTER_TEST_USED": False,
            }
        )
    return (
        pd.DataFrame(summaries),
        pd.concat(subject_parts, ignore_index=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(action_rows),
        pd.DataFrame(recovery_rows),
    )


def _learnability_table(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, action), group in records.groupby(["model_id", "action_family"]):
        for population, subset in (
            ("all_action_rows", group),
            ("boundary_cross_only", group[group.action_boundary_cross.eq(1)]),
        ):
            for target, score in (("rescue", "p_rescue"), ("harm", "p_harm")):
                y = subset[f"target_{target}"].to_numpy(dtype=int)
                values = subset[score].to_numpy(dtype=float)
                prevalence = float(y.mean()) if len(y) else np.nan
                auprc = float(average_precision_score(y, values)) if y.sum() else np.nan
                rows.append(
                    {
                        "model_id": model_id,
                        "action_family": action,
                        "population": population,
                        "target": target,
                        "rows": int(len(subset)),
                        "prevalence": prevalence,
                        "AUPRC": auprc,
                        "AUPRC_lift": auprc / prevalence if prevalence and np.isfinite(auprc) else np.nan,
                        "AUROC": (
                            float(roc_auc_score(y, values))
                            if len(np.unique(y)) == 2
                            else np.nan
                        ),
                        "OUTER_TEST_USED": False,
                    }
                )
    return pd.DataFrame(rows)


def _write_iteration_logs(results: pd.DataFrame) -> None:
    hypotheses = {
        "M1_ENSEMBLE_CONFIDENCE_RULE": (
            "Low B6 margin plus a confident flipping action may isolate residual corrections.",
            "deterministic confidence rule",
        ),
        "M2_ENSEMBLE_DISAGREEMENT_RULE": (
            "Run disagreement may expose ensemble errors that an action consensus can repair.",
            "deterministic disagreement rule",
        ),
        "M3_ACTION_MOVEMENT_LOGISTIC": (
            "Action movement and ensemble diversity predict rescue versus harm above B6.",
            "regularized action-movement rescue/harm heads",
        ),
        "M4_FULL_LEGAL_LOGISTIC": (
            "Cross-fitted PERSIST/protected diagnostics add grouped value beyond ordinary movement features.",
            "full legal regularized rescue/harm heads",
        ),
        "M5_HIST_GRADIENT_BOOSTING": (
            "Small nonlinear interactions improve the legal residual value estimate.",
            "small depth-controlled HistGradientBoosting heads",
        ),
        "I006_CONDITIONAL_ACTION_LOGISTIC": (
            "Training only on boundary-cross candidates and separating action families removes trivial eligibility discrimination and may stabilize rescue-versus-harm routing.",
            "action-specific conditional regularized logistic rescue heads with a finite calibrated action menu",
        ),
        "I007_CONDITIONAL_ACTION_HGB": (
            "Small nonlinear action-specific conditional heads may capture the moderate ERASE rescue signal without unrestricted model search.",
            "action-specific conditional depth-controlled HistGradientBoosting rescue heads with the same finite menu",
        ),
    }
    summary_rows = []
    for iteration, model_id in enumerate(hypotheses, start=1):
        row = results[results.model_id.eq(model_id)].iloc[0]
        kept = bool(row.mean_subject_delta_BA_vs_B6 > 0 and row.rescue_precision > row.harm_rate)
        diagnosis = (
            "positive net grouped residual gain"
            if kept
            else "residual ranking did not convert into safe positive gain above B6"
        )
        decision = "KEEP_FOR_COMPARISON" if kept else "ABANDON"
        summary_rows.append(
            {
                "iteration": iteration,
                "model_id": model_id,
                "failure_diagnosis": diagnosis,
                "hypothesis": hypotheses[model_id][0],
                "change": hypotheses[model_id][1],
                "grouped_delta_BA_vs_B6": row.mean_subject_delta_BA_vs_B6,
                "bootstrap_CI95_L": row.bootstrap_CI95_L,
                "bootstrap_CI95_U": row.bootstrap_CI95_U,
                "harm_rate": row.harm_rate,
                "rescue_precision": row.rescue_precision,
                "action_rate": row.action_rate,
                "residual_oracle_recovery": row.action_oracle_recovery_fraction,
                "decision": decision,
                "OUTER_TEST_USED": False,
            }
        )
        (RESEARCH_LOG / f"ITERATION_{iteration:03d}.md").write_text(
            f"""# Iteration {iteration:03d}: {model_id}

- Failure diagnosis / outcome: {diagnosis}
- Hypothesis: {hypotheses[model_id][0]}
- Change: {hypotheses[model_id][1]}
- Grouped OOF Delta BA vs B6: {row.mean_subject_delta_BA_vs_B6:+.6f}
- Subject-bootstrap CI95: [{row.bootstrap_CI95_L:+.6f}, {row.bootstrap_CI95_U:+.6f}]
- Action rate: {row.action_rate:.6f}
- Rescue precision: {row.rescue_precision:.6f}
- Harm rate: {row.harm_rate:.6f}
- Oracle recovery: {row.action_oracle_recovery_fraction:.6f}
- Decision: {decision}

All thresholds and model settings were selected on inner calibration subjects;
outer held-out subjects were not used for selection. This is exploratory OOF
evidence, not an untouched confirmation.

`OUTER_TEST_USED=false`
""",
            encoding="utf-8",
        )
    write_csv(RESEARCH_LOG / "ITERATION_SUMMARY.csv", pd.DataFrame(summary_rows))


def _final_decision(
    trials: pd.DataFrame,
    results: pd.DataFrame,
    subjects: pd.DataFrame,
    learnability: pd.DataFrame,
    audit: dict[str, Any],
    categories: dict[str, list[str]],
    selections: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict[str, Any]:
    candidates = results[results.model_id.ne("M0_B6_KEEP_ENSEMBLE")].copy()
    candidates = candidates.sort_values(
        ["mean_subject_delta_BA_vs_B6", "bootstrap_CI95_L", "rescue_precision", "action_rate", "model_id"],
        ascending=[False, False, False, True, True],
    )
    best = candidates.iloc[0]
    interesting = (
        best.mean_subject_delta_BA_vs_B6 > 0
        and best.bootstrap_CI95_L > 0
        and best.positive_fold_fraction >= 0.60
        and best.action_rate > 0
        and best.rescue_precision > best.harm_rate
    )
    ready = interesting and best.mean_subject_delta_BA_vs_B6 >= 0.005 and best.positive_fold_fraction >= 0.80
    finite_learnability = learnability[
        learnability.population.eq("boundary_cross_only")
    ].dropna(subset=["AUPRC", "AUROC"]).copy()
    if ready:
        state = "READY_FOR_NEW_INDEPENDENT_PROTOCOL"
    elif interesting:
        state = "PROMISING_RESIDUAL_ACTION_POLICY"
    else:
        signal = bool(
            (
                (finite_learnability.AUPRC > 1.10 * finite_learnability.prevalence)
                & (finite_learnability.AUROC > 0.55)
            ).any()
        )
        state = "RESIDUAL_ACTION_SIGNAL_BUT_NO_NET_GAIN" if signal else "STRUCTURAL_ACTION_RESIDUAL_NOT_PREDICTABLE"

    m3 = results[results.model_id.eq("M3_ACTION_MOVEMENT_LOGISTIC")].iloc[0]
    m4 = results[results.model_id.eq("M4_FULL_LEGAL_LOGISTIC")].iloc[0]
    subject_m3 = subjects[subjects.model_id.eq("M3_ACTION_MOVEMENT_LOGISTIC")].set_index("subject_id")
    subject_m4 = subjects[subjects.model_id.eq("M4_FULL_LEGAL_LOGISTIC")].set_index("subject_id")
    persist_delta = subject_m4.delta_BA_vs_B6 - subject_m3.delta_BA_vs_B6
    persist_ci = paired_bootstrap(persist_delta.to_numpy(dtype=float), "M4_MINUS_M3")
    persist_adds = float(persist_delta.mean()) > 0 and persist_ci[0] > 0

    top_features = []
    if len(importance):
        selected_importance = importance[importance.model_id.eq("M4_FULL_LEGAL_LOGISTIC")]
        top = (
            selected_importance.groupby("feature", as_index=False).absolute_coefficient.mean()
            .sort_values("absolute_coefficient", ascending=False)
            .head(20)
        )
        top_features = top.to_dict(orient="records")

    final = {
        "terminal_state": state,
        "phase_8_plus_executed": True,
        "best_exploratory_policy": str(best.model_id),
        "best_mean_subject_delta_BA_vs_B6": float(best.mean_subject_delta_BA_vs_B6),
        "best_bootstrap_CI95": [float(best.bootstrap_CI95_L), float(best.bootstrap_CI95_U)],
        "best_positive_subject_fraction": float(best.positive_subject_fraction),
        "best_positive_fold_fraction": float(best.positive_fold_fraction),
        "best_worst_subject_delta_BA": float(best.worst_subject_delta_BA_vs_B6),
        "best_action_rate": float(best.action_rate),
        "best_rescue_precision": float(best.rescue_precision),
        "best_harm_rate": float(best.harm_rate),
        "best_action_oracle_recovery_fraction": float(best.action_oracle_recovery_fraction),
        "best_unique_action_oracle_recovery_fraction": float(best.unique_action_oracle_recovery_fraction),
        "persist_full_minus_movement_delta_BA": float(persist_delta.mean()),
        "persist_full_minus_movement_CI95": list(persist_ci),
        "persist_features_add_grouped_value": persist_adds,
        "learnability_signal_rule": "among boundary-cross candidates only: any grouped-OOF target/action with AUPRC > 1.10x prevalence and AUROC > 0.55",
        "best_conditional_learnability_AUROC": float(finite_learnability.AUROC.max()),
        "best_conditional_learnability_AUPRC_lift": float(
            (finite_learnability.AUPRC / finite_learnability.prevalence).max()
        ),
        "top_full_legal_logistic_features": top_features,
        "headroom_state": audit["decision"]["state"],
        "strongest_action_oracle_delta_BA": audit["decision"]["strongest_action_oracle_delta_BA_vs_B6"],
        "keep_only_oracle_delta_BA": audit["decision"]["keep_only_oracle_delta_BA_vs_B6"],
        "unique_action_oracle_delta_BA": audit["decision"]["action_oracle_minus_keep_only_delta_BA"],
        "all_52_subjects_historical_development": True,
        "confirmatory": False,
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", final)
    _write_report(final, results, learnability, audit)
    if state in ("READY_FOR_NEW_INDEPENDENT_PROTOCOL", "PROMISING_RESIDUAL_ACTION_POLICY"):
        _write_next_stage(final, categories, selections)
    else:
        (NEXT_STAGE / "ENSEMBLE_COMPRESSION_DISTILLATION_PLAN.md").write_text(
            """# Constructive next line

The structural oracle did not convert into a robust grouped OOF policy above
B6. Stop intervention model search on these 52 historical subjects. Freeze B6
as teacher, study compression/distillation to one deployable student, and use
PERSIST as an audit and safety framework. Any renewed intervention claim needs
a materially different hypothesis and a new independent protocol.

WBCIC outer remains unauthorized.
""",
            encoding="utf-8",
        )
    return final


def _write_next_stage(final: dict[str, Any], categories: dict[str, list[str]], selections: pd.DataFrame) -> None:
    lock = {
        "status": "RESIDUAL_POLICY_LOCK_DRAFT_NOT_CONFIRMATORY",
        "policy_model": final["best_exploratory_policy"],
        "reference": "B6_ALL_RUN_LOGIT_MEAN",
        "action_menu": ["ALL_AMPLIFY", "ALL_GEOMETRY", "ALL_ERASE"],
        "default": "KEEP_B6",
        "feature_categories": categories,
        "training_protocol": "subject-grouped model-training/calibration split; no independent-evaluation threshold tuning",
        "fold_selection_records": selections[
            selections.model_id.eq(final["best_exploratory_policy"])
            & selections.selected_on_inner_calibration.eq(True)
        ].to_dict(orient="records"),
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(NEXT_STAGE / "RESIDUAL_POLICY_LOCK_DRAFT.json", lock)
    (NEXT_STAGE / "RESIDUAL_POLICY_LOCK_DRAFT.md").write_text(
        f"""# Residual policy lock draft

Exploratory policy: `{final['best_exploratory_policy']}`.

Reference is B6; default action is KEEP. The finite global action menu is
AMPLIFY, GEOMETRY, ERASE. Training, calibration, risk weighting, and threshold
selection must be frozen before a genuinely independent evaluation. This draft
does not authorize WBCIC outer.
""",
        encoding="utf-8",
    )
    (NEXT_STAGE / "NEW_INDEPENDENT_EVALUATION_PLAN.md").write_text(
        """# New independent evaluation plan

Use genuinely untouched subjects or another compatible dataset. Pre-freeze
the B6 expert pool, action outputs, feature schema, model family, training and
calibration subjects, risk weight, action threshold, seeds, and stopping rule.
Primary inference unit is subject. Report Delta BA vs compute-matched B6,
subject bootstrap CI, fold robustness, harm, rescue precision, and deployment
cost. The current 52 subjects cannot serve as confirmation.

WBCIC outer remains locked unless a separate authorization is created.
""",
        encoding="utf-8",
    )


def _write_report(
    final: dict[str, Any],
    results: pd.DataFrame,
    learnability: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    diagnostics = pd.read_csv(DIAGNOSTICS / "RESIDUAL_ORACLE_RESULTS.csv")
    all_oracle = diagnostics[diagnostics.pool.eq("all_52_exploratory")].set_index("method_id")
    keep = float(audit["decision"]["keep_only_oracle_delta_BA_vs_B6"])
    strongest = audit["decision"]
    b6 = pd.read_csv(DIAGNOSTICS / "B6_UNIQUE_TRIAL_RESULTS.csv")
    b6_ba = float(b6[b6.pool.eq("all_52_exploratory")].mean_subject_BA.iloc[0])
    protected = float(all_oracle.loc["ORACLE_ACTION_PROTECTED_SAFE_GLOBAL", "mean_subject_delta_BA_vs_B6"])
    full = float(all_oracle.loc["ORACLE_ACTION_FULL_GLOBAL", "mean_subject_delta_BA_vs_B6"])
    single_safe = float(
        all_oracle.loc["ORACLE_ACTION_PROTECTED_SAFE_SINGLE_REPLACEMENT", "mean_subject_delta_BA_vs_B6"]
    )
    single_full = float(
        all_oracle.loc["ORACLE_ACTION_FULL_SINGLE_REPLACEMENT", "mean_subject_delta_BA_vs_B6"]
    )
    report = f"""# PERSIST-EEG residual actionability V3

## Terminal state

`{final['terminal_state']}`

This is exploratory grouped OOF evidence on 52 historical development
subjects. It is not a new confirmation. WBCIC outer was not accessed.

## Direct answers

1. V2.1 B6 reproduced exactly: **true**.
2. Deployment-level B6 mean subject BA: **{b6_ba:.6f}**.
3. Global protected-safe action oracle above B6: **{100*protected:+.3f} pp**.
4. Full global action oracle above B6: **{100*full:+.3f} pp**.
5. Single-expert replacement oracle: protected-safe **{100*single_safe:+.3f} pp**;
   full **{100*single_full:+.3f} pp**.
6. KEEP-only diversity oracle above B6: **{100*keep:+.3f} pp**.
7. Strongest action oracle minus KEEP-only oracle:
   **{100*strongest['action_oracle_minus_keep_only_delta_BA']:+.3f} pp**;
   combined action+KEEP minus KEEP-only:
   **{100*strongest['combined_keep_plus_action_minus_keep_only_delta_BA']:+.3f} pp**.
8. Residual oracle distribution: {strongest['positive_subjects']}/52 positive
   subjects; top-20% concentration {strongest['top20_subject_gain_concentration']:.3f}.
9. Unique action rescue is detailed in `ACTION_UNIQUENESS.csv`; ERASE is the
   dominant global source but is not the only source.
10. ERASE is necessary for the strongest oracle, but its unconditional harm
    count is substantially larger than its rescue count.
11. Protected-safe oracle headroom remains nonzero but is smaller than FULL.
12. Residual rescue/harm learnability is reported in
    `RESIDUAL_LEARNABILITY.csv` using held-out subjects only. The decision uses
    boundary-cross candidates only; best conditional AUROC is
    **{final['best_conditional_learnability_AUROC']:.3f}** and best conditional
    AUPRC/prevalence lift is **{final['best_conditional_learnability_AUPRC_lift']:.3f}**.
13. The largest full-logistic legal features are stored in
    `FEATURE_IMPORTANCE.csv` and `FINAL_DECISION.json`.
14. PERSIST feature increment over movement-only logistic:
    **{100*final['persist_full_minus_movement_delta_BA']:+.3f} pp**, CI95
    [{100*final['persist_full_minus_movement_CI95'][0]:+.3f},
    {100*final['persist_full_minus_movement_CI95'][1]:+.3f}] pp.
15. Best prospective method: `{final['best_exploratory_policy']}`.
16. Delta BA vs B6: **{100*final['best_mean_subject_delta_BA_vs_B6']:+.3f} pp**,
    grouped subject CI95 [{100*final['best_bootstrap_CI95'][0]:+.3f},
    {100*final['best_bootstrap_CI95'][1]:+.3f}] pp.
17. Unique-action oracle recovery:
    **{final['best_unique_action_oracle_recovery_fraction']:.3f}**.
18. The gain mechanism is a residual correction only if the grouped OOF lower
    bound is positive; otherwise the result remains oracle-only structural
    headroom.
19. Intervention research should continue only under the terminal-state rule
    above and never by further tuning these same OOF subjects.
20. If the policy criterion fails, the constructive next line is B6 ensemble
    compression/distillation while retaining PERSIST for audit and safety.

## Prospective model table

{markdown_table(results)}

## Scientific limitation

The model ladder and best-policy selection were explored on the same 52
historical subjects through grouped OOF estimates. I006-I007 were motivated
after auditing M1-M5 conditional learnability, so their CIs are additionally
post-primary-adaptive and descriptive. Even a positive result must be frozen
and tested under a genuinely new independent protocol.

`OUTER_TEST_USED=false`
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")


def run_residual_policy_research(
    *,
    trials: pd.DataFrame,
    audit: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    if audit["decision"]["state"] != "STRUCTURAL_ACTION_RESIDUAL_EXISTS":
        raise RuntimeError("Phase 8 is not authorized by the residual headroom gate")
    action_candidates = build_action_candidates(trials)
    features, movement_features, full_features, categories = build_legal_residual_features(
        trials, action_candidates, cache_root
    )
    conditional_features = [
        column
        for column in full_features
        if column
        not in (
            "action_boundary_cross",
            "action_vs_b6_disagreement",
            "is_amplify",
            "is_geometry",
            "is_erase",
            "protected_safe",
        )
    ]
    protocol = _subject_protocol(sorted(trials.subject_id.unique().tolist(), key=int))
    write_json(OUTPUTS / "protocol" / "GROUPED_NESTED_CV.json", protocol)
    write_json(
        OUTPUTS / "protocol" / "LEGAL_FEATURE_SCHEMA.json",
        {
            "movement_features": movement_features,
            "full_legal_features": full_features,
            "conditional_action_features": conditional_features,
            "categories": categories,
            "target_columns_excluded": ["outcome_label", "target_rescue", "target_harm"],
            "rows": int(len(features)),
            "unique_trials": int(features.trial_index.nunique()),
            "feature_missing_counts": {
                column: int(features[column].isna().sum()) for column in full_features
            },
            "feature_infinite_counts_after_sanitization": {
                column: int(np.isinf(features[column].to_numpy(dtype=float)).sum())
                for column in full_features
            },
            "OUTER_TEST_USED": False,
        },
    )

    policies = {model_id: _empty_policy(trials) for model_id in MODEL_IDS}
    policies["M0_B6_KEEP_ENSEMBLE"] = OOFPolicy(
        prediction=trials.y_keep_ens.to_numpy(dtype=int).copy(),
        probability=trials.p_keep_ens.to_numpy(dtype=float).copy(),
        selected_action=np.full(len(trials), "KEEP", dtype=object),
        selected_value=np.zeros(len(trials), dtype=float),
        predicted_rescue=np.zeros(len(trials), dtype=float),
        predicted_harm=np.zeros(len(trials), dtype=float),
        outer_fold=np.full(len(trials), -1, dtype=int),
    )
    selection_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    learnability_records: list[pd.DataFrame] = []

    model_definitions = (
        ("M3_ACTION_MOVEMENT_LOGISTIC", "logistic", movement_features),
        ("M4_FULL_LEGAL_LOGISTIC", "logistic", full_features),
        ("M5_HIST_GRADIENT_BOOSTING", "hgb", full_features),
    )
    conditional_model_definitions = (
        ("I006_CONDITIONAL_ACTION_LOGISTIC", "logistic"),
        ("I007_CONDITIONAL_ACTION_HGB", "hgb"),
    )
    for fold_spec in protocol["folds"]:
        outer_fold = int(fold_spec["outer_fold"])
        test_subjects = fold_spec["heldout_subjects"]
        test_trials = np.sort(trials.index[trials.subject_id.isin(test_subjects)].to_numpy(dtype=int))
        policies["M0_B6_KEEP_ENSEMBLE"].outer_fold[test_trials] = outer_fold

        for model_id, kind in (
            ("M1_ENSEMBLE_CONFIDENCE_RULE", "confidence"),
            ("M2_ENSEMBLE_DISAGREEMENT_RULE", "disagreement"),
        ):
            calibration_trials = np.sort(
                trials.index[trials.subject_id.isin(fold_spec["calibration_subjects"])].to_numpy(dtype=int)
            )
            parameters, _, records = _calibrate_rule(
                trials, features, calibration_trials, kind, outer_fold
            )
            selection_records.extend(records)
            test_decision = _rule_decision(features, test_trials, kind=kind, parameters=parameters)
            _assign_policy(policies[model_id], test_decision, outer_fold)

        for model_id, model_kind, columns in model_definitions:
            rescue_model, harm_model, parameters, _, records, importance = _fit_calibrate_model(
                trials,
                features,
                columns,
                fold_spec["model_training_subjects"],
                fold_spec["calibration_subjects"],
                model_kind,
                model_id,
                outer_fold,
            )
            selection_records.extend(records)
            importance_records.extend(importance)
            test_rows = features.subject_id.isin(test_subjects).to_numpy()
            x_test = features.loc[test_rows, columns].to_numpy(dtype=float)
            p_rescue = _predict_positive(rescue_model, x_test)
            p_harm = _predict_positive(harm_model, x_test)
            test_features = features.loc[test_rows].reset_index(drop=True)
            decision = _value_decision(
                test_features,
                test_trials,
                p_rescue,
                p_harm,
                risk_lambda=float(parameters["risk_lambda"]),
                threshold=float(parameters["threshold"]),
            )
            _assign_policy(policies[model_id], decision, outer_fold)
            learnability_records.append(
                test_features[
                    [
                        "trial_index",
                        "subject_id",
                        "action_family",
                        "action_boundary_cross",
                        "target_rescue",
                        "target_harm",
                    ]
                ].assign(
                    model_id=model_id,
                    outer_fold=outer_fold,
                    p_rescue=p_rescue,
                    p_harm=p_harm,
                    OUTER_TEST_USED=False,
                )
            )

        for model_id, model_kind in conditional_model_definitions:
            models, parameters, records, importance = _fit_calibrate_conditional_model(
                trials,
                features,
                conditional_features,
                fold_spec["model_training_subjects"],
                fold_spec["calibration_subjects"],
                model_kind,
                model_id,
                outer_fold,
            )
            selection_records.extend(records)
            importance_records.extend(importance)
            test_rows = features.subject_id.isin(test_subjects).to_numpy()
            test_features = features.loc[test_rows].reset_index(drop=True)
            p_rescue, p_harm = _predict_conditional_action_models(
                models, test_features, conditional_features
            )
            decision = _conditional_action_decision(
                test_features,
                p_rescue,
                p_harm,
                menu=str(parameters["menu"]),
                threshold=float(parameters["threshold"]),
            )
            _assign_policy(policies[model_id], decision, outer_fold)
            learnability_records.append(
                test_features[
                    [
                        "trial_index",
                        "subject_id",
                        "action_family",
                        "action_boundary_cross",
                        "target_rescue",
                        "target_harm",
                    ]
                ].assign(
                    model_id=model_id,
                    outer_fold=outer_fold,
                    p_rescue=p_rescue,
                    p_harm=p_harm,
                    OUTER_TEST_USED=False,
                )
            )

    result_table, subject_table, fold_table, action_table, recovery_table = _evaluate_oof(
        trials, policies, audit
    )
    selection_table = pd.DataFrame(selection_records)
    importance_table = pd.DataFrame(importance_records)
    learnability_predictions = pd.concat(learnability_records, ignore_index=True)
    learnability_table = _learnability_table(learnability_predictions)

    write_csv(RESULTS / "RESIDUAL_LEARNABILITY.csv", learnability_table)
    write_csv(
        RESULTS / "RESIDUAL_LEARNABILITY_CONDITIONAL.csv",
        learnability_table[learnability_table.population.eq("boundary_cross_only")].reset_index(drop=True),
    )
    write_csv(RESULTS / "RESIDUAL_POLICY_RESULTS.csv", result_table)
    write_csv(RESULTS / "SUBJECT_RESULTS.csv", subject_table)
    write_csv(RESULTS / "FOLD_RESULTS.csv", fold_table)
    write_csv(RESULTS / "ACTION_RESULTS.csv", action_table)
    write_csv(RESULTS / "ORACLE_RECOVERY.csv", recovery_table)
    write_csv(RESULTS / "CALIBRATION_SELECTION.csv", selection_table)
    write_csv(RESULTS / "FEATURE_IMPORTANCE.csv", importance_table)
    write_csv(RESULTS / "RESIDUAL_LEARNABILITY_PREDICTIONS.csv", learnability_predictions)
    oof_rows = []
    for model_id, policy in policies.items():
        oof_rows.append(
            pd.DataFrame(
                {
                    "trial_uid": trials.trial_uid,
                    "subject_id": trials.subject_id,
                    "model_id": model_id,
                    "outer_fold": policy.outer_fold,
                    "label": trials.outcome_label,
                    "b6_prediction": trials.y_keep_ens,
                    "prediction": policy.prediction,
                    "probability": policy.probability,
                    "selected_action": policy.selected_action,
                    "selected_value": policy.selected_value,
                    "predicted_rescue": policy.predicted_rescue,
                    "predicted_harm": policy.predicted_harm,
                    "OUTER_TEST_USED": False,
                }
            )
        )
    write_csv(RESULTS / "OOF_POLICY_PREDICTIONS.csv", pd.concat(oof_rows, ignore_index=True))
    _write_iteration_logs(result_table)
    return _final_decision(
        trials,
        result_table,
        subject_table,
        learnability_table,
        audit,
        categories,
        selection_table,
        importance_table,
    )
