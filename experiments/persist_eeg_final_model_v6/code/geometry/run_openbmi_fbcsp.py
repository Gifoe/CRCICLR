"""Evaluate subject-history FBCSP and deep/geometry fusion on OpenBMI S1->S2."""

from __future__ import annotations

import ast
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from baselines.methods import fit_population, logit, target_probability
from common import ABLATIONS, BASELINES, CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, V6_SEED, stable_seed, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import load_openbmi_fold


BANDSETS = {
    "mu": (1, 2),
    "beta": (3, 4, 5, 6),
    "mu_beta_8_28": (1, 2, 3, 4, 5),
    "mu_beta_8_32": (1, 2, 3, 4, 5, 6),
    "full_4_40": tuple(range(9)),
}


def _features(cov_history: np.ndarray, cov_future: np.ndarray, labels: np.ndarray, bands: tuple[int, ...], pairs: int, regularization: float) -> tuple[np.ndarray, np.ndarray]:
    history_parts, future_parts = [], []
    dimension = cov_history.shape[-1]
    identity = np.eye(dimension, dtype=np.float64) / dimension
    for band in bands:
        c0 = np.asarray(cov_history[labels == 0, band].mean(axis=0), dtype=np.float64)
        c1 = np.asarray(cov_history[labels == 1, band].mean(axis=0), dtype=np.float64)
        c0 = (1 - regularization) * c0 + regularization * identity
        c1 = (1 - regularization) * c1 + regularization * identity
        _, vectors = eigh(c1, c0 + c1 + 1e-8 * identity, check_finite=False)
        selected = np.r_[np.arange(pairs), np.arange(dimension - pairs, dimension)]
        filters = vectors[:, selected]
        train_variance = np.einsum("ik,nij,jk->nk", filters, cov_history[:, band], filters, optimize=True)
        future_variance = np.einsum("ik,nij,jk->nk", filters, cov_future[:, band], filters, optimize=True)
        train_variance /= np.maximum(train_variance.sum(axis=1, keepdims=True), 1e-12)
        future_variance /= np.maximum(future_variance.sum(axis=1, keepdims=True), 1e-12)
        history_parts.append(np.log(np.maximum(train_variance, 1e-12)))
        future_parts.append(np.log(np.maximum(future_variance, 1e-12)))
    return np.column_stack(history_parts), np.column_stack(future_parts)


def _configs() -> list[dict[str, Any]]:
    classifiers = (
        ("lda", "auto"),
        ("lda", 0.5),
        ("lda", 0.9),
        ("logistic", 0.03),
        ("logistic", 0.1),
        ("logistic", 0.3),
        ("logistic", 1.0),
    )
    return [
        {
            "bandset": bandset,
            "pairs": pairs,
            "csp_regularization": regularization,
            "classifier": classifier,
            "classifier_parameter": parameter,
        }
        for bandset, pairs, regularization, (classifier, parameter) in itertools.product(
            BANDSETS, (1, 2, 3), (0.0, 0.1), classifiers
        )
    ]


def _fit_probability(x_history: np.ndarray, labels: np.ndarray, x_future: np.ndarray, configuration: dict[str, Any], seed: int) -> np.ndarray:
    if configuration["classifier"] == "lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=configuration["classifier_parameter"]).fit(x_history, labels)
        return model.predict_proba(x_future)[:, 1]
    scaler = StandardScaler().fit(x_history)
    model = LogisticRegression(
        C=float(configuration["classifier_parameter"]),
        class_weight="balanced",
        solver="liblinear",
        max_iter=4_000,
        random_state=seed,
    ).fit(scaler.transform(x_history), labels)
    return model.predict_proba(scaler.transform(x_future))[:, 1]


def _score(subject_predictions: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    ba, nll = [], []
    for labels, probability in subject_predictions:
        ba.append(float(balanced_accuracy_score(labels, probability >= 0.5)))
        nll.append(float(log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])))
    return float(np.mean(ba)), float(np.mean(nll))


def _deep_predictions(subjects_by_fold: dict[int, set[str]]) -> dict[tuple[int, str], np.ndarray]:
    selections = pd.read_csv(DIAGNOSTICS / "OPENBMI_BASELINE_SELECTIONS.csv")
    result: dict[tuple[int, str], np.ndarray] = {}
    for fold in range(5):
        data = load_openbmi_fold(fold)
        fit_mask = data.mask(list(data.model_fit_subjects), list(data.history_sessions) + [data.future_session])
        pop_row = selections.loc[(selections.outer_fold.eq(fold)) & selections.method_id.eq("B_POPULATION_LINEAR")].iloc[0]
        pop_config = ast.literal_eval(str(pop_row.configuration))
        context = fit_population(
            data.embeddings[fit_mask],
            data.metadata.loc[fit_mask, "label"].to_numpy(int),
            float(pop_config["C"]),
            stable_seed(V6_SEED, "geometry-deep", fold),
        )
        fusion_row = selections.loc[(selections.outer_fold.eq(fold)) & selections.method_id.eq("B_HISTORY_FUSION_LDA")].iloc[0]
        fusion_config = ast.literal_eval(str(fusion_row.configuration))
        for subject in subjects_by_fold[fold]:
            history = data.mask([subject], list(data.history_sessions))
            future = data.mask([subject], [data.future_session])
            result[(fold, subject)] = target_probability(
                "fusion_lda",
                fusion_config,
                context,
                data.embeddings[history],
                data.metadata.loc[history, "label"].to_numpy(int),
                data.embeddings[future],
                stable_seed(V6_SEED, "geometry-deep", fold, subject),
            )
    return result


def run() -> None:
    covariance = np.load(CACHE / "OPENBMI_MI_MOTOR20_FB_COVARIANCE.npy", mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(CACHE / "OPENBMI_MI_MOTOR20_FB_METADATA.parquet")
    if covariance.shape != (10_800, 9, 20, 20) or len(metadata) != 10_800:
        raise RuntimeError("OpenBMI geometry cache is incomplete")
    configs = _configs()
    subjects = sorted(metadata.subject_id.astype(str).unique(), key=int)
    subject_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}
    structural_features: dict[tuple[str, str, int, float], tuple[np.ndarray, np.ndarray]] = {}
    for subject in subjects:
        history = metadata.subject_id.astype(str).eq(subject) & metadata.session_id.astype(int).eq(1)
        future = metadata.subject_id.astype(str).eq(subject) & metadata.session_id.astype(int).eq(2)
        cov_history = np.asarray(covariance[history.to_numpy()], dtype=np.float64)
        cov_future = np.asarray(covariance[future.to_numpy()], dtype=np.float64)
        labels_history = metadata.loc[history, "label"].to_numpy(int)
        labels_future = metadata.loc[future, "label"].to_numpy(int)
        future_meta = metadata.loc[future].reset_index(drop=True)
        subject_data[subject] = (cov_history, labels_history, cov_future, labels_future, future_meta)
        for bandset, pairs, regularization in itertools.product(BANDSETS, (1, 2, 3), (0.0, 0.1)):
            structural_features[(subject, bandset, pairs, regularization)] = _features(
                cov_history,
                cov_future,
                labels_history,
                BANDSETS[bandset],
                pairs,
                regularization,
            )
        print(f"[FBCSP] prepared subject={subject}", flush=True)
    probabilities: dict[tuple[int, str], np.ndarray] = {}
    for order, configuration in enumerate(configs):
        for subject in subjects:
            x_history, x_future = structural_features[
                (subject, configuration["bandset"], configuration["pairs"], configuration["csp_regularization"])
            ]
            labels_history = subject_data[subject][1]
            probabilities[(order, subject)] = _fit_probability(
                x_history,
                labels_history,
                x_future,
                configuration,
                stable_seed(V6_SEED, "FBCSP", order, subject),
            )
    folds = {fold: load_openbmi_fold(fold) for fold in range(5)}
    subjects_by_fold = {
        fold: set(folds[fold].discovery_subjects) | set(folds[fold].outcome_subjects)
        for fold in range(5)
    }
    deep = _deep_predictions(subjects_by_fold)
    prediction_parts: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    for fold, fold_data in folds.items():
        geometry_records = []
        fusion_records = []
        for order, configuration in enumerate(configs):
            geometry_subjects = [
                (subject_data[subject][3], probabilities[(order, subject)])
                for subject in fold_data.discovery_subjects
            ]
            ba, nll = _score(geometry_subjects)
            geometry_records.append({"order": order, "configuration": configuration, "weight": 1.0, "discovery_BA": ba, "discovery_NLL": nll})
            for weight in (0.25, 0.5, 0.75):
                fused = [
                    (
                        subject_data[subject][3],
                        1.0 / (1.0 + np.exp(-np.clip((1 - weight) * logit(deep[(fold, subject)]) + weight * logit(probabilities[(order, subject)]), -50, 50))),
                    )
                    for subject in fold_data.discovery_subjects
                ]
                fused_ba, fused_nll = _score(fused)
                fusion_records.append({"order": order, "configuration": configuration, "weight": weight, "discovery_BA": fused_ba, "discovery_NLL": fused_nll})
        selected_geometry = max(geometry_records, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["order"]))
        selected_fusion = max(fusion_records, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["order"], -row["weight"]))
        for method, selected in (("B_FBCSP", selected_geometry), ("B_DEEP_FBCSP_FUSION", selected_fusion)):
            selection_rows.append(
                {
                    "benchmark": fold_data.benchmark,
                    "outer_fold": fold,
                    "method_id": method,
                    **selected,
                    "candidate_count": len(geometry_records) if method == "B_FBCSP" else len(fusion_records),
                    "target_future_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
            for subject in fold_data.outcome_subjects:
                geometry_probability = probabilities[(int(selected["order"]), subject)]
                weight = float(selected["weight"])
                probability = geometry_probability if method == "B_FBCSP" else 1.0 / (
                    1.0 + np.exp(-np.clip((1 - weight) * logit(deep[(fold, subject)]) + weight * logit(geometry_probability), -50, 50))
                )
                future_meta = subject_data[subject][4]
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "benchmark": fold_data.benchmark,
                            "method_id": method,
                            "trial_uid": future_meta.trial_uid.astype(str),
                            "subject_id": subject,
                            "outer_fold": fold,
                            "label": future_meta.label.to_numpy(int),
                            "probability": probability,
                            "prediction": (probability >= 0.5).astype(int),
                            "selected_configuration": json.dumps(selected["configuration"], sort_keys=True),
                            "geometry_weight": weight,
                            "target_history_labels_used": True,
                            "target_future_labels_used_for_fit": False,
                            "exploratory": True,
                            "OUTER_TEST_USED": False,
                        }
                    )
                )
        print(
            f"[FBCSP] fold={fold} geometry={selected_geometry['discovery_BA']:.4f} fusion={selected_fusion['discovery_BA']:.4f}",
            flush=True,
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    baseline_predictions = pd.read_csv(DIAGNOSTICS / "OPENBMI_BASELINE_PREDICTIONS.csv")
    reference = baseline_predictions.loc[baseline_predictions.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    rows, subject_parts, fold_parts = [], [], []
    for method in sorted(predictions.method_id.unique()):
        frame = predictions.loc[predictions.method_id.eq(method)].copy()
        row, subject_result, fold_result = summarize(frame, reference=reference)
        rows.append(row)
        subject_parts.append(subject_result)
        fold_parts.append(fold_result)
    result = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(BASELINES / "OPENBMI_GEOMETRY_BASELINES.csv", result)
    write_csv(DIAGNOSTICS / "OPENBMI_GEOMETRY_SELECTIONS.csv", pd.DataFrame(selection_rows))
    write_csv(DIAGNOSTICS / "OPENBMI_GEOMETRY_PREDICTIONS.csv", predictions)
    write_csv(DIAGNOSTICS / "OPENBMI_GEOMETRY_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "OPENBMI_GEOMETRY_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(ABLATIONS / "GEOMETRY_ABLATION.csv", result)
    write_json(
        PROTOCOL / "OPENBMI_GEOMETRY_LEGALITY_AUDIT.json",
        {
            "history": "Session 1 labeled trials only",
            "future": "Session 2 evaluation only",
            "CSP_filters_fit_on_target_future": False,
            "classifier_fit_on_target_future": False,
            "configuration_selected_on": "discovery-subject S1-to-S2 episodes",
            "outcome_future_labels_used_for_fit": False,
            "exploratory": True,
            "OUTER_TEST_USED": False,
        },
    )
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
