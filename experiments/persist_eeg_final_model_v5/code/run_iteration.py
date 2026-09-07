from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import CACHE, DIAGNOSTICS, LEADERBOARD, RESEARCH_LOG, ensure_directories, write_csv, write_json
from datasets import WBCIC_EXPERTS, load_wbcic
from evaluation import summarize
from models.hierarchical_reliability import predict_subjects, starting_configurations
from nested_cv import calibration_record, fold_assignment, selection_key


def _postprocess(
    base_probability: np.ndarray,
    base_prediction: np.ndarray,
    local_probability: np.ndarray,
    expert_logits: np.ndarray,
    alpha: float,
    gate: str,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    probability = base_probability + float(alpha) * (local_probability - base_probability)
    votes = (expert_logits >= 0).astype(int)
    majority = np.maximum(votes.sum(axis=1), votes.shape[1] - votes.sum(axis=1))
    if gate == "all":
        eligible = np.ones(len(probability), dtype=bool)
    elif gate == "not_unanimous":
        eligible = majority < votes.shape[1]
    elif gate == "max_disagreement":
        eligible = majority == (votes.shape[1] // 2 + 1)
    elif gate == "current_uncertain_010":
        eligible = np.abs(base_probability - 0.5) <= 0.10
    elif gate == "current_uncertain_020":
        eligible = np.abs(base_probability - 0.5) <= 0.20
    else:
        raise ValueError(gate)
    prediction = base_prediction.copy()
    prediction[eligible] = (probability[eligible] >= float(threshold)).astype(int)
    probability = np.where(eligible, probability, base_probability)
    return probability, prediction


def run_family(method_id: str, families: set[str]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_wbcic()
    all_sessions = pd.read_parquet(CACHE / "WBCIC_DEV_ALL_SESSION_EXPERTS.parquet")
    all_sessions.attrs["expert_names"] = list(WBCIC_EXPERTS)
    target = all_sessions.loc[all_sessions.session_id.astype(int).eq(2)].copy()
    target = target.set_index("trial_uid").loc[data.trial_uid].reset_index()
    if not np.array_equal(target.label.to_numpy(int), data.labels):
        raise RuntimeError("All-session/S3 alignment failed")
    configs = starting_configurations(families)
    alphas = (0.25, 0.5, 0.75, 1.0)
    gates = ("all", "not_unanimous", "max_disagreement", "current_uncertain_010", "current_uncertain_020")
    thresholds = (0.475, 0.5, 0.525)
    probability = np.full(len(data.labels), np.nan, dtype=float)
    prediction = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        train_subjects = list(map(str, fold["train_subjects"]))
        calibration_subjects = list(map(str, fold["calibration_subjects"]))
        test_subjects = list(map(str, fold["test_subjects"]))
        calibration_mask = np.isin(data.subjects, calibration_subjects)
        test_mask = np.isin(data.subjects, test_subjects)
        calibration_indices = np.flatnonzero(calibration_mask)
        candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray, Any]] = []
        order = 0
        for config in configs:
            local_calibration = predict_subjects(
                all_sessions,
                target.loc[calibration_mask].reset_index(drop=True),
                calibration_subjects,
                config,
                train_subjects,
            )
            for alpha in alphas:
                for gate in gates:
                    for threshold in thresholds:
                        p_cal, y_cal = _postprocess(
                            data.current_probability[calibration_mask],
                            data.current_prediction[calibration_mask],
                            local_calibration,
                            data.expert_logits[calibration_mask],
                            alpha,
                            gate,
                            threshold,
                        )
                        full_config = {
                            **config.as_dict(),
                            "alpha": alpha,
                            "gate": gate,
                            "threshold": threshold,
                        }
                        record = calibration_record(data, calibration_indices, y_cal, order, full_config)
                        record.update(
                            {
                                "method_id": method_id,
                                "outer_fold": fold_id,
                                "selected": False,
                                "target_prior_sessions_used": True,
                                "target_S3_labels_used_for_fit": False,
                            }
                        )
                        candidates.append((record, p_cal, y_cal, config))
                        order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        chosen = selected[0]["configuration"]
        chosen_config = selected[3]
        local_test = predict_subjects(
            all_sessions,
            target.loc[test_mask].reset_index(drop=True),
            test_subjects,
            chosen_config,
            train_subjects,
        )
        p_test, y_test = _postprocess(
            data.current_probability[test_mask],
            data.current_prediction[test_mask],
            local_test,
            data.expert_logits[test_mask],
            float(chosen["alpha"]),
            str(chosen["gate"]),
            float(chosen["threshold"]),
        )
        probability[test_mask] = p_test
        prediction[test_mask] = y_test
        print(
            f"[{method_id}] fold={fold_id} selected={json.dumps(chosen, sort_keys=True)} "
            f"cal_delta={selected[0]['calibration_Delta_BA']:.6f}",
            flush=True,
        )
    if np.isnan(probability).any() or np.any(prediction < 0):
        raise RuntimeError(f"Incomplete OOF prediction for {method_id}")
    folds = fold_assignment(data)
    row, subjects, fold_table = summarize(
        data, method_id, prediction, probability, folds, baseline="current"
    )
    pred_frame = pd.DataFrame(
        {
            "dataset": data.dataset_id,
            "trial_uid": data.trial_uid,
            "subject_id": data.subjects,
            "outer_fold": folds,
            "label": data.labels,
            "B_STRONG_CURRENT_prediction": data.current_prediction,
            "prediction": prediction,
            "probability": probability,
            "target_prior_sessions_used": True,
            "target_S3_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        }
    )
    return row, subjects, fold_table, pd.DataFrame(selections), pred_frame


def run() -> None:
    ensure_directories()
    jobs = [
        ("M3_SOFT_RELIABILITY", {"soft_ba", "soft_nll", "hard_best"}),
        ("M3_LOCAL_SESSION_LOGISTIC", {"local_logistic"}),
        ("M3_HIERARCHICAL_RELIABILITY_SEARCH", {"soft_ba", "soft_nll", "hard_best", "local_logistic"}),
    ]
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for method_id, families in jobs:
        row, subject, fold, selection, prediction = run_family(method_id, families)
        rows.append(row)
        subjects.append(subject)
        folds.append(fold)
        selections.append(selection)
        prediction["method_id"] = method_id
        predictions.append(prediction)
        print(json.dumps(row, indent=2), flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(["Delta_BA", "NLL"], ascending=[False, True])
    write_csv(LEADERBOARD / "WBCIC_CROSS_SESSION_RELIABILITY.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_CROSS_SESSION_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_CROSS_SESSION_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_CROSS_SESSION_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_CROSS_SESSION_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_001.json",
        {
            "previous_failure": "V4 output-only stacking was below target and fold-unstable.",
            "hypothesis": "Target-subject S1/S2 expert reliability predicts S3 local competence.",
            "what_changed": "Added legal prior-session reliability, compact-pool pruning, conservative current-baseline anchoring, and subject-local logistic aggregation.",
            "grouped_result": best,
            "conclusion": "KEEP" if best["Delta_BA"] > 0 else "MODIFY",
            "target_prior_sessions_used": True,
            "target_S3_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
