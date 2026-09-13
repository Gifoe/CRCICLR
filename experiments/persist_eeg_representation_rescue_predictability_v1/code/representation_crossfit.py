#!/usr/bin/env python3
"""Strict subject-disjoint cross-fitting for F0/F1/F2 representation diagnostics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("CLEAN_RESCUE_VS_HARM", "SAFE_SWITCH")
FAMILIES = ("F0", "F1", "F2")
OOF_KEY = ["comparison", "task", "seed", "fold", "subject_id", "session", "trial_id", "outcome"]


def atomic_csv(path: Path, frame: pd.DataFrame, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False, **kwargs)
    os.replace(temp, path)


def target_rows(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    if target == "CLEAN_RESCUE_VS_HARM":
        frame = frame[frame.outcome.isin(["RESCUE", "HARM"])].copy()
    return frame, frame.outcome.eq("RESCUE").to_numpy(int)


def subject_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    subjects = frame.subject_id.astype(str)
    counts = subjects.value_counts()
    weights = subjects.map(lambda subject: 1.0 / counts[subject]).to_numpy(float)
    return weights * (len(weights) / weights.sum())


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
                                         class_weight="balanced", random_state=0)),
    ])


def fit_model(frame: pd.DataFrame, columns: list[str]) -> tuple[Pipeline, np.ndarray]:
    y = frame.outcome.eq("RESCUE").to_numpy(int)
    if len(np.unique(y)) != 2:
        raise RuntimeError("diagnostic training target has one class")
    weights = subject_equal_weights(frame)
    model = build_model()
    model.fit(frame[columns].to_numpy(float), y, logistic__sample_weight=weights)
    return model, weights


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    pred = score >= .5
    return {
        "n_rows": int(len(y)), "n_rescue": int(y.sum()),
        "ROC_AUC": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "PR_AUC": float(average_precision_score(y, score)) if y.sum() else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
        "precision_RESCUE": float(precision_score(y, pred, zero_division=0)),
        "recall_RESCUE": float(recall_score(y, pred, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    repo, runtime = args.repo.resolve(), args.runtime.resolve()
    exp = repo / "experiments/persist_eeg_representation_rescue_predictability_v1"
    outputs = exp / "outputs"
    schema = json.loads((outputs / "REPRESENTATION_FEATURE_SCHEMA.json").read_text(encoding="utf-8"))
    data = pd.read_csv(runtime / "REPRESENTATION_FEATURES.csv.gz", dtype={"subject_id": str})
    result_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    loso_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for comparison in ("X", "XS"):
        comp = data[data.comparison == comparison]
        for task in sorted(comp.task.unique()):
            base = comp[comp.task == task]
            for target in TARGETS:
                scoped, _ = target_rows(base, target)
                for family in FAMILIES:
                    columns = schema["tasks"][task][family]
                    for held_fold in range(5):
                        train, _ = target_rows(scoped[scoped.fold != held_fold], target)
                        test, y_test = target_rows(scoped[scoped.fold == held_fold], target)
                        model, weights = fit_model(train, columns)
                        audit_frame = train[["subject_id"]].copy()
                        audit_frame["fit_weight"] = weights
                        subject_totals = audit_frame.groupby("subject_id", as_index=False).fit_weight.sum()
                        expected = float(subject_totals.fit_weight.mean())
                        for row in subject_totals.itertuples(index=False):
                            weight_rows.append({"comparison": comparison, "task": task, "target": target,
                                                "feature_family": family, "heldout_fold": held_fold,
                                                "subject_id": str(row.subject_id), "total_fit_weight": float(row.fit_weight),
                                                "expected_equal_weight": expected,
                                                "absolute_difference": abs(float(row.fit_weight) - expected),
                                                "equal_weight_pass": abs(float(row.fit_weight) - expected) <= 1e-10})
                        score = model.predict_proba(test[columns].to_numpy(float))[:, 1]
                        groups: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] = [
                            ("pooled" if comparison == "XS" else "seed0", test, y_test, score)
                        ]
                        if comparison == "XS":
                            seed_values = test.seed.to_numpy(int)
                            groups += [(f"seed{seed}", test[seed_values == seed], y_test[seed_values == seed], score[seed_values == seed]) for seed in (0, 1, 2)]
                        for scope, group, labels, scores in groups:
                            result_rows.append({"comparison": comparison, "task": task, "target": target,
                                                "feature_family": family, "heldout_fold": held_fold,
                                                "evaluation_scope": scope, **metrics(labels, scores)})
                        oof = test[OOF_KEY].copy()
                        oof["target"] = target
                        oof["feature_family"] = family
                        oof["score_RESCUE"] = score
                        oof_rows.append(oof)
                        logistic = model.named_steps["logistic"]
                        for name, value in zip(columns, logistic.coef_[0]):
                            coefficient_rows.append({"comparison": comparison, "task": task, "target": target,
                                                     "feature_family": family, "heldout_fold": held_fold,
                                                     "training_scope": "pooled_seeds" if comparison == "XS" else "seed0",
                                                     "feature": name, "standardized_coefficient": float(value)})
                    if comparison == "XS":
                        for held_fold in range(5):
                            for held_seed in (0, 1, 2):
                                train, _ = target_rows(scoped[(scoped.fold != held_fold) & (scoped.seed != held_seed)], target)
                                test, y_test = target_rows(scoped[(scoped.fold == held_fold) & (scoped.seed == held_seed)], target)
                                model, _ = fit_model(train, columns)
                                score = model.predict_proba(test[columns].to_numpy(float))[:, 1]
                                loso_rows.append({"task": task, "target": target, "feature_family": family,
                                                  "heldout_fold": held_fold, "heldout_seed": held_seed,
                                                  **metrics(y_test, score)})
                    print(f"CROSSFIT {comparison} {task} {target} {family} PASS", flush=True)

    coefficients = pd.DataFrame(coefficient_rows)
    summary = coefficients.groupby(["comparison", "task", "target", "feature_family", "feature"], as_index=False).standardized_coefficient.agg(
        mean_coefficient="mean", mean_absolute_coefficient=lambda x: float(np.abs(x).mean()),
        sign_consistency=lambda x: float(max((x > 0).mean(), (x < 0).mean())))
    coefficients = coefficients.merge(summary, on=["comparison", "task", "target", "feature_family", "feature"], how="left")
    weights = pd.DataFrame(weight_rows)
    if not weights.equal_weight_pass.all():
        raise RuntimeError("subject-equal fitting-weight audit failed")
    atomic_csv(outputs / "REPRESENTATION_CROSSFOLD_RESULTS.csv", pd.DataFrame(result_rows))
    atomic_csv(outputs / "XS_LEAVE_ONE_SEED_OUT_REPRESENTATION_RESULTS.csv", pd.DataFrame(loso_rows))
    atomic_csv(outputs / "REPRESENTATION_LOGISTIC_COEFFICIENTS.csv", coefficients)
    atomic_csv(outputs / "SUBJECT_WEIGHT_AUDIT.csv", weights)
    atomic_csv(runtime / "REPRESENTATION_OOF_SCORES.csv.gz", pd.concat(oof_rows, ignore_index=True), compression="gzip")
    print(f"REPRESENTATION_CROSSFIT_COMPLETE results={len(result_rows)} loso={len(loso_rows)}", flush=True)


if __name__ == "__main__":
    main()
