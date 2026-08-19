"""Reliability-interacted and variance-reduced subject-adaptation stacks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss

from aggregation import anchored_postprocess
from common import CACHE, DIAGNOSTICS, LEADERBOARD, RESEARCH_LOG, logit, sigmoid, stable_seed, ensure_directories, write_csv, write_json
from datasets import WBCIC_EXPERTS, load_wbcic
from evaluation import summarize
from nested_cv import fold_assignment
from run_search import _configs, _output_features
from run_structural_search import run_error_rescue
from training import OOFResult, _sample_weight, run_nested_direct
from models import output_competence


def _aligned_local_probabilities(data) -> np.ndarray:
    frame = pd.read_csv(DIAGNOSTICS / "WBCIC_SUBJECT_ADAPTATION_OOF_PREDICTIONS.csv")
    values = []
    for name in WBCIC_EXPERTS:
        method = f"M10_LOCAL_HEAD_{name}"
        part = frame.loc[frame.method_id.eq(method)].set_index("trial_uid")
        if len(part) != len(data.labels):
            raise RuntimeError(f"Missing local head predictions for {name}")
        values.append(part.loc[data.trial_uid].probability.to_numpy(float))
    return np.column_stack(values)


def _history_features(data) -> tuple[np.ndarray, list[str]]:
    all_sessions = pd.read_parquet(CACHE / "WBCIC_DEV_ALL_SESSION_EXPERTS.parquet")
    selected = pd.read_csv(DIAGNOSTICS / "WBCIC_SUBJECT_ADAPTATION_HISTORY_SELECTIONS.csv")
    selected_flag = selected.selected.astype(str).str.lower().eq("true")
    selected = selected.loc[selected_flag].copy()
    subject_rows, names = {}, []
    metric_names = (
        "head_cv_ba",
        "head_cv_worst_ba",
        "head_cv_nll",
        "raw_s1_ba",
        "raw_s2_ba",
        "raw_history_min_ba",
        "raw_history_ba_drift",
        "raw_history_nll",
    )
    for expert in WBCIC_EXPERTS:
        names.extend([f"{expert}_{metric}" for metric in metric_names])
    for subject in sorted(np.unique(data.subjects).tolist()):
        values = []
        history = all_sessions.loc[
            all_sessions.subject_id.astype(str).eq(subject)
            & all_sessions.session_id.astype(int).isin([0, 1])
        ]
        for expert in WBCIC_EXPERTS:
            head = selected.loc[
                selected.subject_id.astype(str).eq(subject) & selected.expert.eq(expert)
            ]
            if len(head) != 1:
                raise RuntimeError(f"History selection mismatch for {subject}/{expert}")
            probability = sigmoid(history[f"margin_{expert}"].to_numpy(float))
            label = history.label.to_numpy(int)
            session_ba = []
            for session in (0, 1):
                mask = history.session_id.astype(int).eq(session).to_numpy()
                session_ba.append(
                    float(balanced_accuracy_score(label[mask], (probability[mask] >= 0.5).astype(int)))
                )
            values.extend(
                [
                    float(head.history_cv_BA.iloc[0]),
                    float(head.history_cv_worst_session_BA.iloc[0]),
                    float(head.history_cv_NLL.iloc[0]),
                    session_ba[0],
                    session_ba[1],
                    min(session_ba),
                    abs(session_ba[0] - session_ba[1]),
                    float(log_loss(label, probability, labels=[0, 1])),
                ]
            )
        subject_rows[subject] = np.asarray(values, dtype=np.float32)
    matrix = np.stack([subject_rows[str(subject)] for subject in data.subjects])
    return matrix, names


def _build_features(data):
    output = _output_features(data)
    local_probability = _aligned_local_probabilities(data)
    local_logits = logit(local_probability)
    history, history_names = _history_features(data)
    raw_probability = sigmoid(data.expert_logits)
    raw_logits = data.expert_logits
    head_ba = history[:, np.arange(0, history.shape[1], 8)]
    head_worst = history[:, np.arange(1, history.shape[1], 8)]
    raw_min_ba = history[:, np.arange(5, history.shape[1], 8)]
    reliability_interactions = np.column_stack(
        [
            local_logits - raw_logits,
            local_probability - raw_probability,
            local_logits * (head_ba - 0.5),
            local_logits * (head_worst - 0.5),
            raw_logits * (raw_min_ba - 0.5),
            (local_logits - raw_logits) * (head_ba - 0.5),
            local_probability.mean(axis=1),
            local_probability.std(axis=1),
            (local_probability >= 0.5).mean(axis=1),
            np.abs(local_probability.mean(axis=1) - 0.5),
        ]
    )
    simple = np.column_stack([output, local_logits, local_probability]).astype(np.float32)
    reliability = np.column_stack([simple, history, reliability_interactions]).astype(np.float32)
    if not np.isfinite(reliability).all():
        raise RuntimeError("Non-finite reliability stack features")
    return simple, reliability, history_names


def _fixed_stack(data, method_id, x, c_values, gate):
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    records = []
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        train = np.isin(data.subjects, fold["train_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        raw_probabilities = []
        for c_value in c_values:
            config = {"family": "logistic", "C": float(c_value), "pca_components": None}
            model = output_competence.build(config, stable_seed("V5_FIXED_STACK", method_id, fold_id, c_value))
            output_competence.fit(model, x[train], data.labels[train], _sample_weight(data, train))
            raw_probabilities.append(model.predict_proba(x[test])[:, 1])
        raw_probability = np.mean(np.stack(raw_probabilities), axis=0)
        p_test, y_test = anchored_postprocess(
            data.current_probability[test], data.current_prediction[test], raw_probability, data.expert_logits[test],
            alpha=1.0, gate=gate, threshold=0.5,
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        records.append(
            {
                "method_id": method_id,
                "outer_fold": fold_id,
                "C_values": list(map(float, c_values)),
                "gate": gate,
                "threshold": 0.5,
                "inner_calibration_used": False,
                "target_prior_sessions_used": True,
                "target_S3_labels_used_for_fit": False,
                "OUTER_TEST_USED": False,
            }
        )
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(records))


class _StaticFeatureProvider:
    def __init__(self, matrix):
        self.matrix = matrix

    def get(self, name, fold_id):
        del name, fold_id
        return self.matrix


def run() -> None:
    ensure_directories()
    data = load_wbcic()
    simple, reliability, history_names = _build_features(data)
    jobs = [
        ("M11_FIXED_SIMPLE_STACK_C1", "variance_reduced_stack", _fixed_stack(data, "M11_FIXED_SIMPLE_STACK_C1", simple, (1.0,), "not_unanimous")),
        ("M11_FIXED_SIMPLE_STACK_AVG", "variance_reduced_stack", _fixed_stack(data, "M11_FIXED_SIMPLE_STACK_AVG", simple, (0.01, 0.1, 1.0), "all")),
        ("M11_FIXED_RELIABILITY_STACK_AVG", "reliability_interacted_stack", _fixed_stack(data, "M11_FIXED_RELIABILITY_STACK_AVG", reliability, (0.01, 0.1, 1.0), "all")),
        (
            "M11_RELIABILITY_STACK_LOGISTIC",
            "reliability_interacted_stack",
            run_nested_direct(data, "M11_RELIABILITY_STACK_LOGISTIC", reliability, _configs("logistic", reliability.shape[1]), anchored=True),
        ),
        (
            "M11_RELIABILITY_STACK_HISTGB",
            "reliability_interacted_stack",
            run_nested_direct(data, "M11_RELIABILITY_STACK_HISTGB", reliability, _configs("histgb", reliability.shape[1]), anchored=True),
        ),
        (
            "M11_RELIABILITY_ERROR_THREE_TWO",
            "history_conditioned_rescue",
            run_error_rescue(data, _StaticFeatureProvider(reliability), "M11_RELIABILITY_ERROR_THREE_TWO", "RELIABILITY", "extra_trees", "three_two"),
        ),
    ]
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for method_id, family, result in jobs:
        row, subject, fold = summarize(data, method_id, result.prediction, result.probability, result.outer_fold, baseline="current")
        row.update({"architecture_family": family, "target_prior_sessions_used": True})
        rows.append(row); subjects.append(subject); folds.append(fold); selections.append(result.selections)
        predictions.append(
            pd.DataFrame(
                {
                    "dataset": data.dataset_id,
                    "trial_uid": data.trial_uid,
                    "subject_id": data.subjects,
                    "method_id": method_id,
                    "outer_fold": result.outer_fold,
                    "label": data.labels,
                    "B_STRONG_CURRENT_prediction": data.current_prediction,
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "target_prior_sessions_used": True,
                    "target_S3_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
        )
        print(json.dumps(row, indent=2), flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(["Delta_BA", "NLL"], ascending=[False, True])
    write_csv(LEADERBOARD / "WBCIC_RELIABILITY_STACK.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_RELIABILITY_STACK_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_RELIABILITY_STACK_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_RELIABILITY_STACK_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_RELIABILITY_STACK_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_006.json",
        {
            "previous_failure": "Subject-local logits added +0.562 pp but retained one negative fold and a negative CI lower bound.",
            "hypothesis": "History-only reliability and explicit local-vs-global interactions can shrink unstable subject adaptation while fixed model averaging reduces inner-selection variance.",
            "what_changed": "Added S1/S2 head CV, raw-expert session stability, reliability interactions, fixed configuration averaging, and a history-conditioned rescue control.",
            "history_feature_count": len(history_names),
            "grouped_result": best,
            "conclusion": "KEEP" if best["Delta_BA"] >= 0.007 else "MODIFY",
            "target_prior_sessions_used": True,
            "target_S3_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard.to_string(index=False), flush=True)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
