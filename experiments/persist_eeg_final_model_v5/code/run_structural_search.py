"""Structurally distinct local-competence searches for WBCIC development.

This file intentionally keeps all outer-fold outcomes write-only until a
complete family has produced OOF predictions.  Hyperparameters, gates, pools,
and thresholds are selected only on the designated calibration subjects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aggregation import anchored_postprocess
from common import DIAGNOSTICS, LEADERBOARD, RESEARCH_LOG, ensure_directories, sigmoid, stable_seed, write_csv, write_json
from datasets import V5Dataset, load_wbcic
from evaluation import summarize
from fold_features import WBCICFoldFeatures
from models import correlation_aware, local_knn, minority_rescue, output_competence, ranking
from nested_cv import calibration_record, selection_key
from training import OOFResult


POOLS: dict[str, tuple[int, ...]] = {
    "stable_deep": (0, 2),
    "competent3": (0, 1, 2),
    "no_conformer4": (0, 1, 2, 4),
    "all5": (0, 1, 2, 3, 4),
}


def _target_weight(subjects: np.ndarray, target: np.ndarray) -> np.ndarray:
    subjects = np.asarray(subjects).astype(str)
    target = np.asarray(target, dtype=int)
    subject_count = pd.Series(subjects).value_counts().to_dict()
    class_count = pd.Series(target).value_counts().to_dict()
    weight = np.asarray(
        [1.0 / subject_count[subject] / class_count[int(label)] for subject, label in zip(subjects, target)],
        dtype=float,
    )
    return weight / weight.mean()


def _binary_configs(family: str) -> list[dict[str, Any]]:
    if family == "logistic":
        return [
            {"family": "logistic", "C": value, "pca_components": components}
            for components in (None, 32)
            for value in (0.01, 0.1)
        ]
    if family == "histgb":
        return [
            {
                "family": "histgb",
                "learning_rate": 0.05,
                "max_leaf_nodes": leaves,
                "max_iter": 200,
                "l2": l2,
                "min_samples_leaf": 40,
            }
            for leaves in (7, 15)
            for l2 in (1.0, 10.0)
        ]
    if family == "extra_trees":
        return [
            {
                "family": "extra_trees",
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": leaf,
                "max_features": "sqrt",
            }
            for leaf in (20, 50)
        ]
    raise ValueError(family)


def _fit_binary(configuration, seed, x, target, subjects):
    model = output_competence.build(configuration, seed)
    output_competence.fit(model, x, target, _target_weight(subjects, target))
    return model


def _apply_error_subset(data, indices, risk, scope, threshold):
    full_risk = np.zeros(len(data.labels), dtype=float)
    full_risk[np.asarray(indices, dtype=int)] = np.asarray(risk, dtype=float)
    probability, prediction, switched = minority_rescue.switched_prediction(
        data,
        full_risk,
        scope=scope,
        threshold=threshold,
    )
    indices = np.asarray(indices, dtype=int)
    return probability[indices], prediction[indices], switched[indices]


def run_error_rescue(
    data: V5Dataset,
    features: WBCICFoldFeatures,
    method_id: str,
    feature_name: str,
    family: str,
    scope: str,
) -> OOFResult:
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    eligible_all = minority_rescue.eligibility(data, scope)
    error_target = (data.current_prediction != data.labels).astype(int)
    thresholds = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        x = features.get(feature_name, fold_id)
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        fit_mask = train & eligible_all
        calibration_indices = np.flatnonzero(calibration)
        candidates = []
        order = 0
        for configuration in _binary_configs(family):
            seed = stable_seed("V5_ERROR_RESCUE", method_id, fold_id, json.dumps(configuration, sort_keys=True))
            model = _fit_binary(
                configuration,
                seed,
                x[fit_mask],
                error_target[fit_mask],
                data.subjects[fit_mask],
            )
            risk_cal = model.predict_proba(x[calibration])[:, 1]
            eligible_cal = eligible_all[calibration]
            if eligible_cal.any() and np.unique(error_target[calibration][eligible_cal]).size == 2:
                auc = float(roc_auc_score(error_target[calibration][eligible_cal], risk_cal[eligible_cal]))
            else:
                auc = float("nan")
            for threshold in thresholds:
                p_cal, y_cal, switched = _apply_error_subset(
                    data, calibration_indices, risk_cal, scope, threshold
                )
                payload = {
                    "model": configuration,
                    "feature_set": feature_name,
                    "scope": scope,
                    "error_threshold": threshold,
                }
                record = calibration_record(data, calibration_indices, y_cal, order, payload)
                record.update(
                    {
                        "method_id": method_id,
                        "outer_fold": fold_id,
                        "calibration_error_AUROC": auc,
                        "calibration_eligible_rate": float(eligible_cal.mean()),
                        "calibration_model_switch_rate": float(switched.mean()),
                        "selected": False,
                        "OUTER_TEST_USED": False,
                    }
                )
                candidates.append((record, model, threshold))
                order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        test_indices = np.flatnonzero(test)
        risk_test = selected[1].predict_proba(x[test])[:, 1]
        p_test, y_test, _ = _apply_error_subset(data, test_indices, risk_test, scope, selected[2])
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        print(
            f"[{method_id}] fold={fold_id} cal_delta={selected[0]['calibration_Delta_BA']:.6f} "
            f"auc={selected[0]['calibration_error_AUROC']:.4f} config={json.dumps(selected[0]['configuration'], sort_keys=True)}",
            flush=True,
        )
    if np.isnan(probability).any() or np.any(prediction < 0):
        raise RuntimeError(f"Incomplete rescue OOF result: {method_id}")
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))


def run_knn_rescue(
    data: V5Dataset,
    features: WBCICFoldFeatures,
    method_id: str,
    feature_name: str,
    scope: str,
) -> OOFResult:
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    eligible_all = minority_rescue.eligibility(data, scope)
    error_target = (data.current_prediction != data.labels).astype(int)
    configs = [
        {"n_neighbors": neighbors, "weights": "distance", "pca_components": components}
        for components in (16, 32)
        for neighbors in (25, 75, 200)
    ]
    thresholds = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        x = features.get(feature_name, fold_id)
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        fit_mask = train & eligible_all
        calibration_indices = np.flatnonzero(calibration)
        candidates = []
        order = 0
        for configuration in configs:
            model = local_knn.build(configuration)
            model.fit(x[fit_mask], error_target[fit_mask])
            risk_cal = model.predict_proba(x[calibration])[:, 1]
            for threshold in thresholds:
                p_cal, y_cal, _ = _apply_error_subset(data, calibration_indices, risk_cal, scope, threshold)
                payload = {"model": configuration, "feature_set": feature_name, "scope": scope, "error_threshold": threshold}
                record = calibration_record(data, calibration_indices, y_cal, order, payload)
                record.update({"method_id": method_id, "outer_fold": fold_id, "selected": False, "OUTER_TEST_USED": False})
                candidates.append((record, model, threshold))
                order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        test_indices = np.flatnonzero(test)
        risk_test = selected[1].predict_proba(x[test])[:, 1]
        p_test, y_test, _ = _apply_error_subset(data, test_indices, risk_test, scope, selected[2])
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        print(f"[{method_id}] fold={fold_id} cal_delta={selected[0]['calibration_Delta_BA']:.6f}", flush=True)
    if np.isnan(probability).any() or np.any(prediction < 0):
        raise RuntimeError(f"Incomplete kNN OOF result: {method_id}")
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))


def _competence_pool_probability(logits, score, pool, mode, temperature):
    pool = np.asarray(pool, dtype=int)
    probability = sigmoid(np.asarray(logits)[:, pool])
    score = np.asarray(score, dtype=float)[:, pool]
    if mode == "top1":
        chosen = np.argmax(score, axis=1)
        return probability[np.arange(len(probability)), chosen]
    if mode == "top2":
        count = min(2, len(pool))
        chosen = np.argpartition(score, -count, axis=1)[:, -count:]
        return np.take_along_axis(probability, chosen, axis=1).mean(axis=1)
    centered = score - np.max(score, axis=1, keepdims=True)
    weight = np.exp(float(temperature) * centered)
    weight /= weight.sum(axis=1, keepdims=True)
    return np.sum(weight * probability, axis=1)


def _fit_competence_models(data, x, fit_mask, configuration, seed_prefix):
    models = []
    correctness = (data.expert_logits >= 0).astype(int) == data.labels[:, None]
    for expert in range(data.expert_logits.shape[1]):
        seed = stable_seed(seed_prefix, expert, json.dumps(configuration, sort_keys=True))
        model = _fit_binary(
            configuration,
            seed,
            x[fit_mask],
            correctness[fit_mask, expert].astype(int),
            data.subjects[fit_mask],
        )
        models.append(model)
    return models


def _predict_competence(models, x):
    return np.column_stack([model.predict_proba(x)[:, 1] for model in models])


def run_multilabel_competence(
    data: V5Dataset,
    features: WBCICFoldFeatures,
    method_id: str,
    feature_name: str,
    family: str,
) -> OOFResult:
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    votes = (data.expert_logits >= 0).astype(int)
    nonunanimous = (votes.min(axis=1) != votes.max(axis=1))
    model_configs = _binary_configs(family)
    aggregations = [
        (pool_name, mode, temperature, alpha, gate)
        for pool_name in ("stable_deep", "competent3", "all5")
        for mode, temperature in (("soft", 2.0), ("soft", 6.0), ("top1", 1.0), ("top2", 1.0))
        for alpha in (0.5, 1.0)
        for gate in ("not_unanimous", "max_disagreement")
    ]
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        x = features.get(feature_name, fold_id)
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        calibration_indices = np.flatnonzero(calibration)
        candidates = []
        order = 0
        for train_scope in ("all", "disagreement"):
            fit_mask = train if train_scope == "all" else train & nonunanimous
            for configuration in model_configs:
                models = _fit_competence_models(
                    data,
                    x,
                    fit_mask,
                    configuration,
                    ("V5_MULTILABEL", method_id, fold_id, train_scope),
                )
                score_cal = _predict_competence(models, x[calibration])
                for pool_name, mode, temperature, alpha, gate in aggregations:
                    local = _competence_pool_probability(
                        data.expert_logits[calibration], score_cal, POOLS[pool_name], mode, temperature
                    )
                    p_cal, y_cal = anchored_postprocess(
                        data.current_probability[calibration],
                        data.current_prediction[calibration],
                        local,
                        data.expert_logits[calibration],
                        alpha=alpha,
                        gate=gate,
                        threshold=0.5,
                    )
                    payload = {
                        "model": configuration,
                        "feature_set": feature_name,
                        "train_scope": train_scope,
                        "pool": pool_name,
                        "aggregation": mode,
                        "temperature": temperature,
                        "alpha": alpha,
                        "gate": gate,
                    }
                    record = calibration_record(data, calibration_indices, y_cal, order, payload)
                    record.update({"method_id": method_id, "outer_fold": fold_id, "selected": False, "OUTER_TEST_USED": False})
                    candidates.append((record, models))
                    order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        chosen = selected[0]["configuration"]
        score_test = _predict_competence(selected[1], x[test])
        local_test = _competence_pool_probability(
            data.expert_logits[test],
            score_test,
            POOLS[str(chosen["pool"])],
            str(chosen["aggregation"]),
            float(chosen["temperature"]),
        )
        p_test, y_test = anchored_postprocess(
            data.current_probability[test],
            data.current_prediction[test],
            local_test,
            data.expert_logits[test],
            alpha=float(chosen["alpha"]),
            gate=str(chosen["gate"]),
            threshold=0.5,
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        print(f"[{method_id}] fold={fold_id} cal_delta={selected[0]['calibration_Delta_BA']:.6f}", flush=True)
    if np.isnan(probability).any() or np.any(prediction < 0):
        raise RuntimeError(f"Incomplete multi-label competence result: {method_id}")
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))


def run_pairwise_ranking(
    data: V5Dataset,
    features: WBCICFoldFeatures,
    method_id: str,
    feature_name: str,
    family: str,
) -> OOFResult:
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    aggregations = [
        (pool_name, mode, temperature, alpha, gate)
        for pool_name in ("stable_deep", "competent3", "all5")
        for mode, temperature in (("soft", 2.0), ("soft", 6.0), ("top1", 1.0), ("top2", 1.0))
        for alpha in (0.5, 1.0)
        for gate in ("not_unanimous", "max_disagreement")
    ]
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        x = features.get(feature_name, fold_id)
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        calibration_indices = np.flatnonzero(calibration)
        x_train_pair, y_train_pair, train_trial, _ = ranking.pair_examples(
            x[train], data.expert_logits[train], data.labels[train]
        )
        train_subjects = data.subjects[train][train_trial]
        x_cal_pair, _, cal_trial, cal_pairs = ranking.pair_examples(x[calibration], data.expert_logits[calibration])
        candidates = []
        order = 0
        for configuration in _binary_configs(family):
            seed = stable_seed("V5_PAIRWISE", method_id, fold_id, json.dumps(configuration, sort_keys=True))
            model = _fit_binary(configuration, seed, x_train_pair, y_train_pair, train_subjects)
            p_left = model.predict_proba(x_cal_pair)[:, 1]
            score_cal = ranking.scores_from_pair_probability(
                int(calibration.sum()), data.expert_logits.shape[1], cal_trial, cal_pairs, p_left
            )
            for pool_name, mode, temperature, alpha, gate in aggregations:
                local = _competence_pool_probability(
                    data.expert_logits[calibration], score_cal, POOLS[pool_name], mode, temperature
                )
                p_cal, y_cal = anchored_postprocess(
                    data.current_probability[calibration],
                    data.current_prediction[calibration],
                    local,
                    data.expert_logits[calibration],
                    alpha=alpha,
                    gate=gate,
                    threshold=0.5,
                )
                payload = {
                    "model": configuration,
                    "feature_set": feature_name,
                    "pairwise_target": "correct_expert_over_incorrect_expert",
                    "pool": pool_name,
                    "aggregation": mode,
                    "temperature": temperature,
                    "alpha": alpha,
                    "gate": gate,
                }
                record = calibration_record(data, calibration_indices, y_cal, order, payload)
                record.update({"method_id": method_id, "outer_fold": fold_id, "selected": False, "OUTER_TEST_USED": False})
                candidates.append((record, model))
                order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        chosen = selected[0]["configuration"]
        x_test_pair, _, test_trial, test_pairs = ranking.pair_examples(x[test], data.expert_logits[test])
        p_left_test = selected[1].predict_proba(x_test_pair)[:, 1]
        score_test = ranking.scores_from_pair_probability(
            int(test.sum()), data.expert_logits.shape[1], test_trial, test_pairs, p_left_test
        )
        local_test = _competence_pool_probability(
            data.expert_logits[test], score_test, POOLS[str(chosen["pool"])], str(chosen["aggregation"]), float(chosen["temperature"])
        )
        p_test, y_test = anchored_postprocess(
            data.current_probability[test], data.current_prediction[test], local_test, data.expert_logits[test],
            alpha=float(chosen["alpha"]), gate=str(chosen["gate"]), threshold=0.5,
        )
        probability[test] = p_test
        prediction[test] = y_test
        assignment[test] = fold_id
        print(f"[{method_id}] fold={fold_id} cal_delta={selected[0]['calibration_Delta_BA']:.6f}", flush=True)
    if np.isnan(probability).any() or np.any(prediction < 0):
        raise RuntimeError(f"Incomplete pairwise result: {method_id}")
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))


def run_correlation_aware(data: V5Dataset, method_id: str) -> OOFResult:
    probability = np.full(len(data.labels), np.nan)
    prediction = np.full(len(data.labels), -1, dtype=int)
    assignment = np.full(len(data.labels), -1, dtype=int)
    selections: list[dict[str, Any]] = []
    all_probability = sigmoid(data.expert_logits)
    for fold in data.folds:
        fold_id = int(fold["outer_fold"])
        train = np.isin(data.subjects, fold["train_subjects"])
        calibration = np.isin(data.subjects, fold["calibration_subjects"])
        test = np.isin(data.subjects, fold["test_subjects"])
        calibration_indices = np.flatnonzero(calibration)
        candidates = []
        order = 0
        for pool_name, pool_tuple in POOLS.items():
            pool = np.asarray(pool_tuple, dtype=int)
            for shrinkage in (0.0, 0.25, 0.5, 0.75, 1.0):
                weight = correlation_aware.weights(all_probability[train][:, pool], data.labels[train], shrinkage)
                local_cal = correlation_aware.aggregate(all_probability[calibration][:, pool], weight)
                for alpha in (0.5, 1.0):
                    blended = data.current_probability[calibration] + alpha * (local_cal - data.current_probability[calibration])
                    for threshold in (0.475, 0.5, 0.525):
                        y_cal = (blended >= threshold).astype(int)
                        payload = {
                            "pool": pool_name,
                            "shrinkage": shrinkage,
                            "weights": weight.tolist(),
                            "alpha": alpha,
                            "threshold": threshold,
                        }
                        record = calibration_record(data, calibration_indices, y_cal, order, payload)
                        record.update({"method_id": method_id, "outer_fold": fold_id, "selected": False, "OUTER_TEST_USED": False})
                        candidates.append((record, pool, weight, alpha, threshold))
                        order += 1
        selected = max(candidates, key=lambda item: selection_key(item[0]))
        selected[0]["selected"] = True
        selections.extend(item[0] for item in candidates)
        local_test = correlation_aware.aggregate(all_probability[test][:, selected[1]], selected[2])
        p_test = data.current_probability[test] + selected[3] * (local_test - data.current_probability[test])
        probability[test] = np.clip(p_test, 1e-7, 1 - 1e-7)
        prediction[test] = (p_test >= selected[4]).astype(int)
        assignment[test] = fold_id
        print(f"[{method_id}] fold={fold_id} cal_delta={selected[0]['calibration_Delta_BA']:.6f}", flush=True)
    return OOFResult(method_id, probability, prediction, assignment, pd.DataFrame(selections))


@dataclass
class Job:
    method_id: str
    family_label: str
    run: Callable[[], OOFResult]


def run() -> None:
    ensure_directories()
    data = load_wbcic()
    features = WBCICFoldFeatures.load(data)
    jobs = [
        Job("M8_OUTPUT_ERROR_HISTGB", "minority_rescue", lambda: run_error_rescue(data, features, "M8_OUTPUT_ERROR_HISTGB", "OUTPUT", "histgb", "opposed")),
        Job("M8_SHARED_ERROR_LOGISTIC", "minority_rescue", lambda: run_error_rescue(data, features, "M8_SHARED_ERROR_LOGISTIC", "OUTPUT_SHARED", "logistic", "opposed")),
        Job("M8_THREE_TWO_ALL_EXTRATREES", "minority_rescue", lambda: run_error_rescue(data, features, "M8_THREE_TWO_ALL_EXTRATREES", "OUTPUT_ALL", "extra_trees", "three_two")),
        Job("M5_LOCAL_KNN_ERROR", "local_knn", lambda: run_knn_rescue(data, features, "M5_LOCAL_KNN_ERROR", "OUTPUT_SHARED", "opposed")),
        Job("M2_MULTILABEL_SHARED_LOGISTIC", "multi_label_correctness", lambda: run_multilabel_competence(data, features, "M2_MULTILABEL_SHARED_LOGISTIC", "OUTPUT_SHARED", "logistic")),
        Job("M2_MULTILABEL_ALL_EXTRATREES", "multi_label_correctness", lambda: run_multilabel_competence(data, features, "M2_MULTILABEL_ALL_EXTRATREES", "OUTPUT_ALL", "extra_trees")),
        Job("M7_PAIRWISE_SHARED_HISTGB", "pairwise_ranking", lambda: run_pairwise_ranking(data, features, "M7_PAIRWISE_SHARED_HISTGB", "OUTPUT_SHARED", "histgb")),
        Job("M9_CORRELATION_AWARE", "correlation_aware", lambda: run_correlation_aware(data, "M9_CORRELATION_AWARE")),
    ]
    rows, subjects, folds, selections, predictions = [], [], [], [], []
    for job in jobs:
        print(f"[structural search] start {job.method_id}", flush=True)
        result = job.run()
        row, subject, fold = summarize(data, job.method_id, result.prediction, result.probability, result.outer_fold, baseline="current")
        row["architecture_family"] = job.family_label
        rows.append(row)
        subjects.append(subject)
        folds.append(fold)
        selections.append(result.selections)
        predictions.append(
            pd.DataFrame(
                {
                    "dataset": data.dataset_id,
                    "trial_uid": data.trial_uid,
                    "subject_id": data.subjects,
                    "method_id": job.method_id,
                    "outer_fold": result.outer_fold,
                    "label": data.labels,
                    "B_STRONG_CURRENT_prediction": data.current_prediction,
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "target_prior_sessions_used": False,
                    "target_S3_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
        )
        print(json.dumps(row, indent=2), flush=True)
    leaderboard = pd.DataFrame(rows).sort_values(["Delta_BA", "NLL"], ascending=[False, True])
    write_csv(LEADERBOARD / "WBCIC_STRUCTURAL_SEARCH.csv", leaderboard)
    write_csv(DIAGNOSTICS / "WBCIC_STRUCTURAL_SEARCH_SUBJECT_RESULTS.csv", pd.concat(subjects, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_STRUCTURAL_SEARCH_FOLD_RESULTS.csv", pd.concat(folds, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_STRUCTURAL_SEARCH_SELECTIONS.csv", pd.concat(selections, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_STRUCTURAL_SEARCH_OOF_PREDICTIONS.csv", pd.concat(predictions, ignore_index=True))
    best = leaderboard.iloc[0].to_dict()
    write_json(
        RESEARCH_LOG / "ITERATION_004.json",
        {
            "previous_failure": "Direct class prediction and session-level expert reliability did not identify local rescues stably.",
            "hypothesis": "Targets aligned to baseline error, multi-label expert correctness, pairwise rank, local neighborhoods, or error covariance may expose learnable complementarity.",
            "what_changed": "Evaluated five structurally different competence/aggregation formulations with fold-compatible shared context and compact expert pools.",
            "grouped_result": best,
            "conclusion": "KEEP" if best["Delta_BA"] >= 0.003 else "MODIFY",
            "OUTER_TEST_USED": False,
        },
    )
    print(leaderboard.to_string(index=False), flush=True)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
