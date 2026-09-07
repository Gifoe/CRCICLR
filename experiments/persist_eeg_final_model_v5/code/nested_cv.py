from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from datasets import V5Dataset


def fold_assignment(data: V5Dataset) -> np.ndarray:
    result = np.full(len(data.labels), -1, dtype=int)
    for fold in data.folds:
        result[np.isin(data.subjects, fold["test_subjects"])] = int(fold["outer_fold"])
    if np.any(result < 0):
        raise RuntimeError("Incomplete subject-disjoint fold assignment")
    return result


def calibration_record(
    data: V5Dataset,
    indices: np.ndarray,
    prediction: np.ndarray,
    candidate_order: int,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    indices = np.asarray(indices, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    labels = data.labels[indices]
    baseline = data.current_prediction[indices]
    subjects = data.subjects[indices]
    deltas = []
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        candidate_ba = balanced_accuracy_score(labels[mask], prediction[mask])
        baseline_ba = balanced_accuracy_score(labels[mask], baseline[mask])
        deltas.append(float(candidate_ba - baseline_ba))
    changed = prediction != baseline
    rescue = changed & (baseline != labels) & (prediction == labels)
    harm = changed & (baseline == labels) & (prediction != labels)
    return {
        "configuration": configuration,
        "candidate_order": int(candidate_order),
        "calibration_Delta_BA": float(np.mean(deltas)),
        "calibration_worst_subject_delta": float(np.min(deltas)),
        "calibration_positive_subject_fraction": float(np.mean(np.asarray(deltas) > 0)),
        "calibration_switch_rate": float(changed.mean()),
        "calibration_rescue_count": int(rescue.sum()),
        "calibration_harm_count": int(harm.sum()),
        "calibration_rescue_precision": float(rescue.sum() / changed.sum()) if changed.any() else 0.0,
        "heldout_S3_labels_read_for_selection": False,
        "OUTER_TEST_USED": False,
    }


def selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(record["calibration_Delta_BA"]),
        float(record["calibration_worst_subject_delta"]),
        float(record["calibration_positive_subject_fraction"]),
        -float(record["calibration_harm_count"]),
        float(record["calibration_rescue_precision"]),
        -float(record["calibration_switch_rate"]),
        -float(record["candidate_order"]),
    )
