#!/usr/bin/env python3
"""Strict five-outer-fold cross-fitting for label-free rescue prediction."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CONF_FEATURES = [
    "b0_top1_probability", "b0_margin", "b0_entropy", "b0_logit_l2_norm",
    "candidate_top1_probability", "candidate_margin", "candidate_entropy", "candidate_logit_l2_norm",
    "top1_probability_difference", "margin_difference", "entropy_difference",
    "probability_L1_distance", "JS_divergence",
]
MECH_COMMON = CONF_FEATURES + [
    "candidate_embedding_l2_norm", "candidate_prehead_feature_l2_norm",
    "mixer_block1_relative_update_norm", "mixer_block2_relative_update_norm", "mixer_block3_relative_update_norm",
    "scale_alpha_max", "scale_alpha_min", "scale_alpha_entropy", "scale_alpha_range",
    "b0_embedding_l2_norm", "b0_final_temporal_feature_l2_norm",
]
XS_EXTRA = ["channel_mod_mean_abs_delta", "channel_mod_std_delta", "channel_mod_max_abs_delta"]
TARGETS = ("CLEAN_RESCUE_VS_HARM", "SAFE_SWITCH")


def atomic_csv(path: Path, frame: pd.DataFrame, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False, **kwargs); os.replace(temp, path)


def features(comparison: str, feature_set: str) -> list[str]:
    if feature_set == "CONF_ONLY": return CONF_FEATURES
    return MECH_COMMON + (XS_EXTRA if comparison == "XS" else [])


def target_rows(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    if target == "CLEAN_RESCUE_VS_HARM":
        frame = frame[frame.outcome.isin(["RESCUE", "HARM"])].copy()
    return frame, frame.outcome.eq("RESCUE").to_numpy(int)


def subject_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = list(zip(frame.task.astype(str), frame.subject_id.astype(str)))
    counts = pd.Series(keys).value_counts()
    weights = np.asarray([1.0 / counts[key] for key in keys], dtype=float)
    return weights * (len(weights) / weights.sum())


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
                                         class_weight="balanced", random_state=0)),
    ])


def fit_model(frame: pd.DataFrame, y: np.ndarray, columns: list[str]) -> Pipeline:
    if len(np.unique(y)) != 2: raise RuntimeError("diagnostic training target has one class")
    model = build_model(); model.fit(frame[columns].to_numpy(float), y, logistic__sample_weight=subject_equal_weights(frame))
    return model


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
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(); repo, runtime = args.repo.resolve(), args.runtime.resolve()
    outputs = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1/outputs"
    data = pd.read_csv(outputs / "PREDICTABILITY_FEATURES.csv", dtype={"subject_id": str})
    result_rows, coefficient_rows, oof_rows, loso_rows = [], [], [], []
    for comparison in ("X", "XS"):
        comp = data[data.comparison == comparison]
        for task in sorted(comp.task.unique()):
            base = comp[comp.task == task]
            for target in TARGETS:
                scoped, _ = target_rows(base, target)
                for feature_set in ("CONF_ONLY", "MECH"):
                    cols = features(comparison, feature_set)
                    for held_fold in range(5):
                        train, _ = target_rows(scoped[scoped.fold != held_fold], target)
                        test, y_test = target_rows(scoped[scoped.fold == held_fold], target)
                        y_train = train.outcome.eq("RESCUE").to_numpy(int)
                        model = fit_model(train, y_train, cols)
                        score = model.predict_proba(test[cols].to_numpy(float))[:, 1]
                        eval_groups: list[tuple[str, pd.DataFrame, np.ndarray, np.ndarray]] = [("pooled" if comparison == "XS" else "seed0", test, y_test, score)]
                        if comparison == "XS":
                            eval_groups += [(f"seed{seed}", test[test.seed == seed], y_test[test.seed.to_numpy(int) == seed], score[test.seed.to_numpy(int) == seed]) for seed in (0, 1, 2)]
                        for scope, group, labels, scores in eval_groups:
                            result_rows.append({"comparison": comparison, "task": task, "target": target, "feature_set": feature_set,
                                                "heldout_fold": held_fold, "evaluation_scope": scope, **metrics(labels, scores)})
                        oof = test[["comparison", "task", "seed", "fold", "subject_id", "session", "trial_id", "outcome"]].copy()
                        oof["target"] = target; oof["feature_set"] = feature_set; oof["score_RESCUE"] = score
                        oof_rows.append(oof)
                        logistic = model.named_steps["logistic"]
                        for name, value in zip(cols, logistic.coef_[0]):
                            coefficient_rows.append({"comparison": comparison, "task": task, "target": target,
                                                     "feature_set": feature_set, "heldout_fold": held_fold,
                                                     "training_scope": "pooled_seeds" if comparison == "XS" else "seed0",
                                                     "feature": name, "standardized_coefficient": float(value)})
                    if comparison == "XS":
                        for held_fold in range(5):
                            for held_seed in (0, 1, 2):
                                train, _ = target_rows(scoped[(scoped.fold != held_fold) & (scoped.seed != held_seed)], target)
                                test, y_test = target_rows(scoped[(scoped.fold == held_fold) & (scoped.seed == held_seed)], target)
                                model = fit_model(train, train.outcome.eq("RESCUE").to_numpy(int), cols)
                                score = model.predict_proba(test[cols].to_numpy(float))[:, 1]
                                loso_rows.append({"task": task, "target": target, "feature_set": feature_set,
                                                  "heldout_fold": held_fold, "heldout_seed": held_seed, **metrics(y_test, score)})
    result = pd.DataFrame(result_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    summary = coefficients.groupby(["comparison", "task", "target", "feature_set", "feature"], as_index=False).standardized_coefficient.agg(
        mean_coefficient="mean", mean_absolute_coefficient=lambda x: float(np.abs(x).mean()),
        sign_consistency=lambda x: float(max((x > 0).mean(), (x < 0).mean())))
    coefficients = coefficients.merge(summary, on=["comparison", "task", "target", "feature_set", "feature"], how="left")
    atomic_csv(outputs / "PREDICTABILITY_CROSSFOLD_RESULTS.csv", result)
    atomic_csv(outputs / "XS_LEAVE_ONE_SEED_OUT_RESULTS.csv", pd.DataFrame(loso_rows))
    atomic_csv(outputs / "LOGISTIC_COEFFICIENTS.csv", coefficients)
    atomic_csv(runtime / "PREDICTABILITY_OOF_SCORES.csv.gz", pd.concat(oof_rows, ignore_index=True), compression="gzip")
    print(f"PREDICTABILITY_CROSSFIT_COMPLETE results={len(result)} loso={len(loso_rows)}", flush=True)


if __name__ == "__main__":
    main()
