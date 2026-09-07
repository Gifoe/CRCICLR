from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common import stable_seed
from datasets import ExpertDataset


THRESHOLDS = tuple(np.round(np.arange(0.35, 0.651, 0.025), 3))


@dataclass
class OOFResult:
    method_id: str
    prediction: np.ndarray
    probability: np.ndarray
    outer_fold: np.ndarray
    selections: pd.DataFrame


def _subject_delta(
    data: ExpertDataset,
    indices: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float, float, float]:
    labels = data.labels[indices]
    base = data.base_prediction[indices]
    subjects = data.subjects[indices]
    deltas = []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        value = balanced_accuracy_score(labels[mask], prediction[mask])
        reference = balanced_accuracy_score(labels[mask], base[mask])
        deltas.append(float(value - reference))
    changed = prediction != base
    rescue = changed & (base != labels) & (prediction == labels)
    harm = changed & (base == labels) & (prediction != labels)
    return (
        float(np.mean(deltas)),
        float(np.min(deltas)),
        float(changed.mean()),
        float(rescue.sum() / changed.sum()) if changed.any() else 0.0,
        float(harm.sum() / changed.sum()) if changed.any() else 0.0,
    )


def _selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(record["calibration_delta_BA"]),
        float(record["calibration_worst_subject_delta"]),
        -float(record["calibration_harm_rate"]),
        float(record["calibration_rescue_precision"]),
        -float(record["calibration_switch_rate"]),
        -float(record["candidate_order"]),
    )


def _fit_model(
    family: str,
    builder: Callable[[dict[str, Any], int], Any],
    configuration: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    base_train: np.ndarray,
    seed: int,
) -> Any:
    model = builder(configuration, seed)
    if family == "residual":
        return model.fit(x_train, y_train, base_train)
    return model.fit(x_train, y_train)


def _predict_probability(family: str, model: Any, values: np.ndarray, base: np.ndarray) -> np.ndarray:
    if family == "residual":
        return np.asarray(model.predict_proba(values, base)[:, 1], dtype=float)
    return np.asarray(model.predict_proba(values)[:, 1], dtype=float)


def run_nested_oof(
    data: ExpertDataset,
    method_id: str,
    x: np.ndarray,
    family: str,
    configurations: list[dict[str, Any]],
    builder: Callable[[dict[str, Any], int], Any],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> OOFResult:
    n = len(data.labels)
    probability = np.full(n, np.nan, dtype=float)
    prediction = np.full(n, -1, dtype=int)
    outer_fold = np.full(n, -1, dtype=int)
    selections: list[dict[str, Any]] = []
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        if np.any(train & calibration) or np.any(train & test) or np.any(calibration & test):
            raise RuntimeError(f"Subject leakage in outer fold {fold_id}")
        if not train.any() or not calibration.any() or not test.any():
            raise RuntimeError(f"Empty nested split in outer fold {fold_id}")

        candidates: list[tuple[dict[str, Any], Any]] = []
        candidate_order = 0
        for configuration in configurations:
            seed = stable_seed("V4_MODEL", method_id, fold_id, json.dumps(configuration, sort_keys=True))
            model = _fit_model(
                family,
                builder,
                configuration,
                x[train],
                data.labels[train],
                data.base_logits[train],
                seed,
            )
            calibration_probability = _predict_probability(
                family, model, x[calibration], data.base_logits[calibration]
            )
            calibration_indices = np.flatnonzero(calibration)
            for threshold in thresholds:
                calibration_prediction = (calibration_probability >= threshold).astype(int)
                delta, worst, switch, precision, harm = _subject_delta(
                    data, calibration_indices, calibration_prediction
                )
                record = {
                    "dataset": data.dataset_id,
                    "method_id": method_id,
                    "outer_fold": fold_id,
                    "model_family": family,
                    "configuration": json.dumps(configuration, sort_keys=True),
                    "threshold": float(threshold),
                    "candidate_order": candidate_order,
                    "calibration_delta_BA": delta,
                    "calibration_worst_subject_delta": worst,
                    "calibration_switch_rate": switch,
                    "calibration_rescue_precision": precision,
                    "calibration_harm_rate": harm,
                    "selected": False,
                    "heldout_subjects_read_for_selection": False,
                    "OUTER_TEST_USED": False,
                }
                candidates.append((record, model))
                candidate_order += 1
        selected_record, selected_model = max(candidates, key=lambda item: _selection_key(item[0]))
        selected_record["selected"] = True
        for record, _ in candidates:
            selections.append(record)
        test_probability = _predict_probability(
            family, selected_model, x[test], data.base_logits[test]
        )
        test_indices = np.flatnonzero(test)
        probability[test_indices] = test_probability
        prediction[test_indices] = (test_probability >= float(selected_record["threshold"])).astype(int)
        outer_fold[test_indices] = fold_id
    if np.isnan(probability).any() or np.any(prediction < 0) or np.any(outer_fold < 0):
        raise RuntimeError(f"Incomplete OOF coverage for {method_id}")
    if len(np.unique(data.trial_uid)) != n:
        raise RuntimeError("Trial identity is not unique")
    return OOFResult(
        method_id=method_id,
        prediction=prediction,
        probability=probability,
        outer_fold=outer_fold,
        selections=pd.DataFrame(selections),
    )


def run_nested_model_average_oof(
    data: ExpertDataset,
    method_id: str,
    x: np.ndarray,
    family: str,
    configurations: list[dict[str, Any]],
    builder: Callable[[dict[str, Any], int], Any],
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> OOFResult:
    n = len(data.labels)
    probability = np.full(n, np.nan, dtype=float)
    prediction = np.full(n, -1, dtype=int)
    outer_fold = np.full(n, -1, dtype=int)
    selections: list[dict[str, Any]] = []
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        calibration_probabilities, test_probabilities = [], []
        for configuration in configurations:
            seed = stable_seed("V4_MODEL_AVERAGE", method_id, fold_id, json.dumps(configuration, sort_keys=True))
            model = _fit_model(
                family,
                builder,
                configuration,
                x[train],
                data.labels[train],
                data.base_logits[train],
                seed,
            )
            calibration_probabilities.append(
                _predict_probability(family, model, x[calibration], data.base_logits[calibration])
            )
            test_probabilities.append(_predict_probability(family, model, x[test], data.base_logits[test]))
        calibration_probability = np.mean(np.stack(calibration_probabilities), axis=0)
        test_probability = np.mean(np.stack(test_probabilities), axis=0)
        choices = []
        calibration_indices = np.flatnonzero(calibration)
        for candidate_order, threshold in enumerate(thresholds):
            calibration_prediction = (calibration_probability >= threshold).astype(int)
            delta, worst, switch, precision, harm = _subject_delta(
                data, calibration_indices, calibration_prediction
            )
            choices.append(
                {
                    "dataset": data.dataset_id,
                    "method_id": method_id,
                    "outer_fold": fold_id,
                    "model_family": f"{family}_configuration_average",
                    "configuration": json.dumps(configurations, sort_keys=True),
                    "threshold": float(threshold),
                    "candidate_order": candidate_order,
                    "calibration_delta_BA": delta,
                    "calibration_worst_subject_delta": worst,
                    "calibration_switch_rate": switch,
                    "calibration_rescue_precision": precision,
                    "calibration_harm_rate": harm,
                    "selected": False,
                    "heldout_subjects_read_for_selection": False,
                    "OUTER_TEST_USED": False,
                }
            )
        selected = max(choices, key=_selection_key)
        selected["selected"] = True
        selections.extend(choices)
        test_indices = np.flatnonzero(test)
        probability[test_indices] = test_probability
        prediction[test_indices] = (test_probability >= float(selected["threshold"])).astype(int)
        outer_fold[test_indices] = fold_id
    if np.isnan(probability).any() or np.any(prediction < 0) or np.any(outer_fold < 0):
        raise RuntimeError(f"Incomplete model-average OOF coverage for {method_id}")
    return OOFResult(method_id, prediction, probability, outer_fold, pd.DataFrame(selections))


def baseline_result(data: ExpertDataset) -> OOFResult:
    fold_assignment = np.full(len(data.labels), -1, dtype=int)
    for fold in data.folds:
        mask = np.isin(data.subjects, fold["test_subjects"])
        fold_assignment[mask] = int(fold["outer_fold"])
    if np.any(fold_assignment < 0):
        raise RuntimeError("Baseline fold assignment is incomplete")
    return OOFResult(
        method_id="M0_B_STRONG_B6",
        prediction=data.base_prediction,
        probability=data.base_probability,
        outer_fold=fold_assignment,
        selections=pd.DataFrame(),
    )


def run_threshold_only_oof(
    data: ExpertDataset,
    method_id: str,
    source_probability: np.ndarray,
) -> OOFResult:
    source_probability = np.asarray(source_probability, dtype=float)
    probability = np.full(len(data.labels), np.nan, dtype=float)
    prediction = np.full(len(data.labels), -1, dtype=int)
    outer_fold = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        calibration_indices = np.flatnonzero(calibration)
        choices = []
        for candidate_order, threshold in enumerate(THRESHOLDS):
            calibration_prediction = (source_probability[calibration] >= threshold).astype(int)
            delta, worst, switch, precision, harm = _subject_delta(
                data, calibration_indices, calibration_prediction
            )
            record = {
                "dataset": data.dataset_id,
                "method_id": method_id,
                "outer_fold": fold_id,
                "model_family": "threshold_only",
                "configuration": "{}",
                "threshold": float(threshold),
                "candidate_order": candidate_order,
                "calibration_delta_BA": delta,
                "calibration_worst_subject_delta": worst,
                "calibration_switch_rate": switch,
                "calibration_rescue_precision": precision,
                "calibration_harm_rate": harm,
                "selected": False,
                "heldout_subjects_read_for_selection": False,
                "OUTER_TEST_USED": False,
            }
            choices.append(record)
        selected = max(choices, key=_selection_key)
        selected["selected"] = True
        selections.extend(choices)
        test_indices = np.flatnonzero(test)
        probability[test_indices] = source_probability[test]
        prediction[test_indices] = (
            source_probability[test] >= float(selected["threshold"])
        ).astype(int)
        outer_fold[test_indices] = fold_id
    if np.isnan(probability).any() or np.any(prediction < 0) or np.any(outer_fold < 0):
        raise RuntimeError(f"Incomplete threshold-only OOF coverage for {method_id}")
    return OOFResult(method_id, prediction, probability, outer_fold, pd.DataFrame(selections))
