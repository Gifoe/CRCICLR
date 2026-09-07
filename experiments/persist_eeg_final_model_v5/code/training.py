from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aggregation import anchored_postprocess
from common import stable_seed
from datasets import V5Dataset
from models import output_competence
from nested_cv import calibration_record, selection_key


@dataclass
class OOFResult:
    method_id: str
    probability: np.ndarray
    prediction: np.ndarray
    outer_fold: np.ndarray
    selections: pd.DataFrame


def _sample_weight(data: V5Dataset, mask: np.ndarray) -> np.ndarray:
    subjects = data.subjects[mask]
    labels = data.labels[mask]
    subject_count = pd.Series(subjects).value_counts().to_dict()
    class_count = pd.Series(labels).value_counts().to_dict()
    weight = np.asarray(
        [1.0 / subject_count[str(subject)] / class_count[int(label)] for subject, label in zip(subjects, labels)],
        dtype=float,
    )
    return weight / weight.mean()


def run_nested_direct(
    data: V5Dataset,
    method_id: str,
    x: np.ndarray,
    configurations: list[dict[str, Any]],
    *,
    anchored: bool,
) -> OOFResult:
    x = np.asarray(x, dtype=np.float32)
    probability = np.full(len(data.labels), np.nan, dtype=float)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    if anchored:
        alphas = (0.25, 0.5, 0.75, 1.0)
        gates = ("all", "not_unanimous", "max_disagreement", "current_uncertain_010", "current_uncertain_020")
        thresholds = (0.475, 0.5, 0.525)
    else:
        alphas, gates, thresholds = (1.0,), ("all",), (0.475, 0.5, 0.525)
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        if np.any(train & calibration) or np.any(train & test) or np.any(calibration & test):
            raise RuntimeError("Subject leakage in nested direct model")
        calibration_indices = np.flatnonzero(calibration)
        candidates: list[tuple[dict[str, Any], Any, float, str, float]] = []
        order = 0
        for configuration in configurations:
            seed = stable_seed("V5_DIRECT", method_id, fold_id, json.dumps(configuration, sort_keys=True))
            model = output_competence.build(configuration, seed)
            output_competence.fit(model, x[train], data.labels[train], _sample_weight(data, train))
            p_cal_raw = model.predict_proba(x[calibration])[:, 1]
            for alpha in alphas:
                for gate in gates:
                    for threshold in thresholds:
                        p_cal, y_cal = anchored_postprocess(
                            data.current_probability[calibration],
                            data.current_prediction[calibration],
                            p_cal_raw,
                            data.expert_logits[calibration],
                            alpha=alpha,
                            gate=gate,
                            threshold=threshold,
                        )
                        payload = {
                            "model": configuration,
                            "alpha": alpha,
                            "gate": gate,
                            "threshold": threshold,
                            "anchored": anchored,
                        }
                        record = calibration_record(data, calibration_indices, y_cal, order, payload)
                        record.update(
                            {
                                "method_id": method_id,
                                "outer_fold": fold_id,
                                "selected": False,
                                "OUTER_TEST_USED": False,
                            }
                        )
                        candidates.append((record, model, alpha, gate, threshold))
                        order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        p_raw = selected[1].predict_proba(x[test])[:, 1]
        p_test, y_test = anchored_postprocess(
            data.current_probability[test],
            data.current_prediction[test],
            p_raw,
            data.expert_logits[test],
            alpha=selected[2],
            gate=selected[3],
            threshold=selected[4],
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        print(
            f"[{method_id}] fold={fold_id} selected={json.dumps(selected[0]['configuration'], sort_keys=True)} "
            f"cal_delta={selected[0]['calibration_Delta_BA']:.6f}",
            flush=True,
        )
    if np.isnan(probability).any() or np.any(prediction < 0) or np.any(assignment < 0):
        raise RuntimeError(f"Incomplete OOF result: {method_id}")
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))


def run_nested_foldwise_direct(
    data: V5Dataset,
    method_id: str,
    feature_getter,
    configurations: list[dict[str, Any]],
    *,
    anchored: bool,
) -> OOFResult:
    """Nested evaluation for representations whose coordinates are fold specific.

    ``feature_getter(fold_id)`` must return one row per trial.  Within an
    outer fold the model-fit, calibration, and held-out rows are therefore in
    exactly the same representation space; representations are never mixed
    across checkpoints.
    """
    probability = np.full(len(data.labels), np.nan, dtype=float)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    if anchored:
        alphas = (0.25, 0.5, 0.75, 1.0)
        gates = ("all", "not_unanimous", "max_disagreement", "current_uncertain_010", "current_uncertain_020")
        thresholds = (0.475, 0.5, 0.525)
    else:
        alphas, gates, thresholds = (1.0,), ("all",), (0.475, 0.5, 0.525)
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        x = np.asarray(feature_getter(fold_id), dtype=np.float32)
        if x.ndim != 2 or len(x) != len(data.labels) or not np.isfinite(x).all():
            raise RuntimeError(f"Invalid foldwise features for fold {fold_id}: {x.shape}")
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        if np.any(train & calibration) or np.any(train & test) or np.any(calibration & test):
            raise RuntimeError("Subject leakage in nested foldwise direct model")
        calibration_indices = np.flatnonzero(calibration)
        candidates: list[tuple[dict[str, Any], Any, float, str, float]] = []
        order = 0
        for configuration in configurations:
            seed = stable_seed("V5_FOLDWISE_DIRECT", method_id, fold_id, json.dumps(configuration, sort_keys=True))
            model = output_competence.build(configuration, seed)
            output_competence.fit(model, x[train], data.labels[train], _sample_weight(data, train))
            p_cal_raw = model.predict_proba(x[calibration])[:, 1]
            for alpha in alphas:
                for gate in gates:
                    for threshold in thresholds:
                        p_cal, y_cal = anchored_postprocess(
                            data.current_probability[calibration],
                            data.current_prediction[calibration],
                            p_cal_raw,
                            data.expert_logits[calibration],
                            alpha=alpha,
                            gate=gate,
                            threshold=threshold,
                        )
                        payload = {
                            "model": configuration,
                            "alpha": alpha,
                            "gate": gate,
                            "threshold": threshold,
                            "anchored": anchored,
                            "fold_specific_representation": True,
                        }
                        record = calibration_record(data, calibration_indices, y_cal, order, payload)
                        record.update(
                            {
                                "method_id": method_id,
                                "outer_fold": fold_id,
                                "selected": False,
                                "OUTER_TEST_USED": False,
                            }
                        )
                        candidates.append((record, model, alpha, gate, threshold))
                        order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        p_raw = selected[1].predict_proba(x[test])[:, 1]
        p_test, y_test = anchored_postprocess(
            data.current_probability[test],
            data.current_prediction[test],
            p_raw,
            data.expert_logits[test],
            alpha=selected[2],
            gate=selected[3],
            threshold=selected[4],
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        print(
            f"[{method_id}] fold={fold_id} selected={json.dumps(selected[0]['configuration'], sort_keys=True)} "
            f"cal_delta={selected[0]['calibration_Delta_BA']:.6f}",
            flush=True,
        )
    if np.isnan(probability).any() or np.any(prediction < 0) or np.any(assignment < 0):
        raise RuntimeError(f"Incomplete foldwise OOF result: {method_id}")
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))
