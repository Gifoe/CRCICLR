"""Evaluate legal S1/S2-only subject adaptation on future-session S3."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aggregation import anchored_postprocess
from common import CACHE, DIAGNOSTICS, LEADERBOARD, RESEARCH_LOG, logit, sigmoid, stable_seed, ensure_directories, write_csv, write_json
from datasets import WBCIC_EXPERTS, load_wbcic
from evaluation import summarize
from models import subject_adaptation
from nested_cv import calibration_record, fold_assignment, selection_key
from run_search import _configs, _output_features
from training import OOFResult, run_nested_direct


def _subject_local_probabilities(data):
    metadata = pd.read_parquet(CACHE / "WBCIC_DEV_ALL_SESSION_EXPERTS.parquet")
    if metadata.OUTER_TEST_USED.astype(bool).any() or metadata.subject_id.astype(str).nunique() != 41:
        raise RuntimeError("All-session cache scope violation")
    target = metadata.loc[metadata.session_id.astype(int).eq(2)].copy()
    positions = pd.Series(np.arange(len(target)), index=target.trial_uid.astype(str))
    if positions.index.duplicated().any():
        raise RuntimeError("Duplicate S3 target identities")
    target_positions = positions.loc[list(map(str, data.trial_uid))].to_numpy(int)
    if not np.array_equal(target.iloc[target_positions].label.to_numpy(int), data.labels):
        raise RuntimeError("S3 target alignment failure")
    all_results: dict[str, np.ndarray] = {}
    selection_rows: list[dict[str, object]] = []
    for expert_name in WBCIC_EXPERTS:
        path = CACHE / f"WBCIC_DEV_ALL_SESSION_{expert_name}_EMBEDDINGS.npy"
        embeddings = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(embeddings) != len(metadata) or not np.isfinite(embeddings).all():
            raise RuntimeError(f"Malformed all-session embedding cache: {expert_name}")
        target_probability = np.full(len(target), np.nan, dtype=float)
        for subject in sorted(metadata.subject_id.astype(str).unique().tolist()):
            subject_mask = metadata.subject_id.astype(str).eq(subject).to_numpy()
            history_mask = subject_mask & metadata.session_id.astype(int).isin([0, 1]).to_numpy()
            subject_target_mask = subject_mask & metadata.session_id.astype(int).eq(2).to_numpy()
            history_x = np.asarray(embeddings[history_mask], dtype=np.float32)
            history_y = metadata.loc[history_mask, "label"].to_numpy(int)
            history_session = metadata.loc[history_mask, "session_id"].to_numpy(int)
            target_x = np.asarray(embeddings[subject_target_mask], dtype=np.float32)
            if set(history_session.tolist()) != {0, 1} or len(target_x) < 190:
                raise RuntimeError(f"Incomplete subject history for {expert_name}/{subject}")
            seed = stable_seed("V5_SUBJECT_HEAD", expert_name, subject)
            selected, records = subject_adaptation.select_from_history(
                history_x,
                history_y,
                history_session,
                subject_adaptation.configurations(),
                seed,
            )
            model = subject_adaptation.build(selected, seed)
            model.fit(history_x, history_y)
            target_probability[subject_target_mask[metadata.session_id.astype(int).eq(2).to_numpy()]] = model.predict_proba(target_x)[:, 1]
            for record in records:
                selection_rows.append(
                    {
                        "expert": expert_name,
                        "subject_id": subject,
                        **record,
                        "selected": record["configuration"] == selected,
                        "target_sessions_used": "S1,S2",
                        "target_S3_labels_used_for_fit": False,
                        "OUTER_TEST_USED": False,
                    }
                )
        if np.isnan(target_probability).any():
            raise RuntimeError(f"Incomplete local-head predictions: {expert_name}")
        all_results[expert_name] = target_probability[target_positions]
        print(f"[subject adaptation] completed {expert_name}", flush=True)
    return all_results, pd.DataFrame(selection_rows)


def _fixed_result(data, method_id, probability):
    probability = np.asarray(probability, dtype=float)
    return OOFResult(method_id, probability, (probability >= 0.5).astype(int), fold_assignment(data), pd.DataFrame())


def _blend_result(data, method_id, source_probability):
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections = []
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        calibration_indices = np.flatnonzero(calibration)
        candidates = []
        order = 0
        for alpha in (0.25, 0.5, 0.75, 1.0):
            for gate in ("all", "not_unanimous", "max_disagreement", "current_uncertain_010", "current_uncertain_020"):
                for threshold in (0.475, 0.5, 0.525):
                    p_cal, y_cal = anchored_postprocess(
                        data.current_probability[calibration], data.current_prediction[calibration], source_probability[calibration],
                        data.expert_logits[calibration], alpha=alpha, gate=gate, threshold=threshold,
                    )
                    configuration = {"alpha": alpha, "gate": gate, "threshold": threshold}
                    record = calibration_record(data, calibration_indices, y_cal, order, configuration)
                    record.update({"method_id": method_id, "outer_fold": fold_id, "selected": False, "target_prior_sessions_used": True, "target_S3_labels_used_for_fit": False, "OUTER_TEST_USED": False})
                    candidates.append(record)
                    order += 1
        selected = max(candidates, key=selection_key)
        selected["selected"] = True
        selections.extend(candidates)
        chosen = selected["configuration"]
        p_test, y_test = anchored_postprocess(
            data.current_probability[test], data.current_prediction[test], source_probability[test], data.expert_logits[test],
            alpha=float(chosen["alpha"]), gate=str(chosen["gate"]), threshold=float(chosen["threshold"]),
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        print(f"[{method_id}] fold={fold_id} cal_delta={selected['calibration_Delta_BA']:.6f}", flush=True)
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))


def run() -> None:
    ensure_directories()
    data = load_wbcic()
    local, history_selections = _subject_local_probabilities(data)
    local_matrix = np.column_stack([logit(local[name]) for name in WBCIC_EXPERTS]).astype(np.float32)
    local_probability_matrix = np.column_stack([local[name] for name in WBCIC_EXPERTS]).astype(np.float32)
    local_competent_mean = local_probability_matrix[:, :3].mean(axis=1)
    local_all_mean = local_probability_matrix.mean(axis=1)
    output = _output_features(data)
    stack_features = np.column_stack([output, local_matrix, local_probability_matrix]).astype(np.float32)
    jobs: list[tuple[str, str, OOFResult]] = []
    for name in WBCIC_EXPERTS:
        method_id = f"M10_LOCAL_HEAD_{name}"
        jobs.append((method_id, "subject_local_head", _fixed_result(data, method_id, local[name])))
    jobs.extend(
        [
            ("M10_LOCAL_COMPETENT3_MEAN", "subject_local_pool", _fixed_result(data, "M10_LOCAL_COMPETENT3_MEAN", local_competent_mean)),
            ("M10_LOCAL_ALL5_MEAN", "subject_local_pool", _fixed_result(data, "M10_LOCAL_ALL5_MEAN", local_all_mean)),
            ("M10_CURRENT_BLEND_LOCAL_STABLE", "subject_local_blend", _blend_result(data, "M10_CURRENT_BLEND_LOCAL_STABLE", local["EEGNet_STABLE"])),
            ("M10_CURRENT_BLEND_LOCAL_COMPETENT3", "subject_local_blend", _blend_result(data, "M10_CURRENT_BLEND_LOCAL_COMPETENT3", local_competent_mean)),
            (
                "M10_LOCAL_STACK_LOGISTIC",
                "subject_local_stack",
                run_nested_direct(data, "M10_LOCAL_STACK_LOGISTIC", stack_features, _configs("logistic", stack_features.shape[1]), anchored=True),
            ),
            (
                "M10_LOCAL_STACK_HISTGB",
                "subject_local_stack",
                run_nested_direct(data, "M10_LOCAL_STACK_HISTGB", stack_features, _configs("histgb", stack_features.shape[1]), anchored=True),
            ),
        ]
    )
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for method_id, family, result in jobs:
        row, subject, fold = summarize(data, method_id, result.prediction, result.probability, result.outer_fold, baseline="current")
        row["architecture_family"] = family
        row["target_prior_sessions_used"] = True
        rows.append(row)
        subjects.append(subject)
        folds.append(fold)
        if not result.selections.empty:
            selections.append(result.selections)
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
    write_csv(LEADERBOARD / "WBCIC_SUBJECT_ADAPTATION.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_SUBJECT_ADAPTATION_HISTORY_SELECTIONS.csv", history_selections)
    write_csv(DIAGNOSTICS / "WBCIC_SUBJECT_ADAPTATION_SELECTIONS.csv", pd.concat(selections, ignore_index=True) if selections else pd.DataFrame())
    write_csv(DIAGNOSTICS / "WBCIC_SUBJECT_ADAPTATION_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_SUBJECT_ADAPTATION_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_SUBJECT_ADAPTATION_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_005.json",
        {
            "previous_failure": "Zero-shot local competence was not stable under subject shift despite moderate error AUROC.",
            "hypothesis": "Legal subject S1/S2 labels can adapt a frozen expert representation to subject-specific task geometry that transfers to S3.",
            "what_changed": "Each target subject receives a frozen-representation linear head selected only by S1<->S2 validation; local heads are then conservatively blended or stacked without using target S3 labels.",
            "grouped_result": best,
            "conclusion": "KEEP" if best["Delta_BA"] >= 0.003 else "MODIFY",
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
