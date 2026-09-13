#!/usr/bin/env python3
"""Train-fold threshold selection and held-fold selective-policy evaluation."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


KEY = ["task", "seed", "fold", "subject_id", "session", "trial_id"]
THRESHOLDS = np.round(np.arange(.05, .951, .01), 2)


def load_py(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module); return module


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False); os.replace(temp, path)


def apply_policy(trials: pd.DataFrame, scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    scores = scored[KEY + ["score_RESCUE"]]
    frame = trials.merge(scores, on=KEY, how="left", validate="one_to_one")
    disagreement = frame.B0_prediction.ne(frame.candidate_prediction)
    if frame.loc[disagreement, "score_RESCUE"].isna().any(): raise RuntimeError("missing disagreement score")
    frame["switched"] = disagreement & frame.score_RESCUE.ge(threshold)
    frame["policy_prediction"] = np.where(frame.switched, frame.candidate_prediction, frame.B0_prediction).astype(int)
    frame["policy_correct"] = frame.policy_prediction.eq(frame.true_label)
    frame["switch_outcome"] = np.where(frame.switched, frame.error_state, "NOT_SWITCHED")
    return frame


def subject_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, seed, fold, subject), group in frame.groupby(["task", "seed", "fold", "subject_id"], sort=True):
        labels = group.true_label.to_numpy(int)
        b0 = balanced_accuracy_score(labels, group.B0_prediction.to_numpy(int))
        candidate = balanced_accuracy_score(labels, group.candidate_prediction.to_numpy(int))
        policy = balanced_accuracy_score(labels, group.policy_prediction.to_numpy(int))
        oracle_pred = np.where(group.candidate_correct & ~group.B0_correct, group.candidate_prediction, group.B0_prediction)
        oracle = balanced_accuracy_score(labels, oracle_pred)
        switched = group[group.switched]
        rescue = int(switched.error_state.eq("RESCUE").sum()); harm = int(switched.error_state.eq("HARM").sum())
        rows.append({"task": task, "seed": int(seed), "fold": int(fold), "subject_id": str(subject),
                     "B0_BA": float(b0), "candidate_BA": float(candidate), "policy_BA": float(policy), "oracle_BA": float(oracle),
                     "policy_delta_vs_B0_pp": float(100 * (policy - b0)), "policy_delta_vs_candidate_pp": float(100 * (policy - candidate)),
                     "oracle_headroom_pp": float(100 * (oracle - b0)), "trials": len(group),
                     "disagreement_trials": int(group.B0_prediction.ne(group.candidate_prediction).sum()),
                     "switched_trials": int(group.switched.sum()), "realized_rescue_count": rescue, "realized_harm_count": harm,
                     "switched_both_wrong_count": int(switched.error_state.eq("WW").sum()),
                     "intervention_precision": float(rescue / (rescue + harm)) if rescue + harm else np.nan})
    return pd.DataFrame(rows)


def subject_equal_policy_ba(frame: pd.DataFrame, comparison: str) -> float:
    values = subject_metrics(frame).groupby("seed").policy_BA.mean()
    return float(values.mean()) if comparison == "XS" else float(values.iloc[0])


def select_threshold(train_trials: pd.DataFrame, train_scores: pd.DataFrame, comparison: str) -> tuple[float, float]:
    frame = train_trials.merge(train_scores[KEY + ["score_RESCUE"]], on=KEY, how="left", validate="one_to_one")
    disagreement = frame.B0_prediction.ne(frame.candidate_prediction).to_numpy(bool)
    if frame.loc[disagreement, "score_RESCUE"].isna().any(): raise RuntimeError("missing training disagreement score")
    grouped = []
    for (seed, subject), group in frame.groupby(["seed", "subject_id"], sort=False):
        grouped.append((int(seed), group.true_label.to_numpy(int), group.B0_prediction.to_numpy(int),
                        group.candidate_prediction.to_numpy(int), group.score_RESCUE.fillna(-1).to_numpy(float),
                        group.B0_prediction.ne(group.candidate_prediction).to_numpy(bool)))
    scored = []
    for threshold in THRESHOLDS:
        seed_values: dict[int, list[float]] = {}
        for seed, labels, b0, candidate, score, differs in grouped:
            pred = np.where(differs & (score >= threshold), candidate, b0)
            classes = np.unique(labels)
            ba = float(np.mean([(pred[labels == cls] == cls).mean() for cls in classes]))
            seed_values.setdefault(seed, []).append(ba)
        value = float(np.mean([np.mean(values) for values in seed_values.values()]))
        scored.append((value, float(threshold)))
    best_value = max(x[0] for x in scored)
    best_threshold = max(x[1] for x in scored if abs(x[0] - best_value) <= 1e-12)
    return best_threshold, best_value


def bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(0)
    draws = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(); repo, runtime = args.repo.resolve(), args.runtime.resolve()
    exp = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1"; outputs = exp / "outputs"
    crossfit = load_py("crossfit_helpers", exp / "code/predictability_crossfit.py")
    features = pd.read_csv(outputs / "PREDICTABILITY_FEATURES.csv", dtype={"subject_id": str})
    fold_rows, subject_rows = [], []
    for comparison, trial_path in (("X", runtime / "X_ROWS.csv.gz"), ("XS", runtime / "FIXED_B0_XS_ROWS.csv.gz")):
        trials = pd.read_csv(trial_path, dtype={"subject_id": str})
        feature_data = features[features.comparison == comparison]
        for task in sorted(trials.task.unique()):
            task_trials, task_features = trials[trials.task == task], feature_data[feature_data.task == task]
            for feature_set in ("CONF_ONLY", "MECH"):
                columns = crossfit.features(comparison, feature_set)
                for held_fold in range(5):
                    train_features = task_features[task_features.fold != held_fold]
                    model = crossfit.fit_model(train_features, train_features.outcome.eq("RESCUE").to_numpy(int), columns)
                    train_scored = train_features[KEY].copy(); train_scored["score_RESCUE"] = model.predict_proba(train_features[columns].to_numpy(float))[:, 1]
                    test_features = task_features[task_features.fold == held_fold]
                    test_scored = test_features[KEY].copy(); test_scored["score_RESCUE"] = model.predict_proba(test_features[columns].to_numpy(float))[:, 1]
                    train_trials = task_trials[task_trials.fold != held_fold]
                    threshold, train_ba = select_threshold(train_trials, train_scored, comparison)
                    evaluated = apply_policy(task_trials[task_trials.fold == held_fold], test_scored, threshold)
                    subjects = subject_metrics(evaluated); subjects["comparison"] = comparison; subjects["feature_set"] = feature_set
                    subjects["selected_threshold"] = threshold; subject_rows.append(subjects)
                    for seed, group in subjects.groupby("seed"):
                        switched = evaluated[(evaluated.seed == seed) & evaluated.switched]
                        rescue = int(switched.error_state.eq("RESCUE").sum()); harm = int(switched.error_state.eq("HARM").sum())
                        seed_trials = evaluated[evaluated.seed == seed]
                        fold_rows.append({"comparison": comparison, "task": task, "seed": int(seed), "feature_set": feature_set,
                                          "heldout_fold": held_fold, "selected_threshold": threshold,
                                          "training_subject_equal_policy_BA": train_ba,
                                          "B0_BA": float(group.B0_BA.mean()), "candidate_BA": float(group.candidate_BA.mean()),
                                          "selective_policy_BA": float(group.policy_BA.mean()),
                                          "policy_delta_vs_B0_pp": float(group.policy_delta_vs_B0_pp.mean()),
                                          "policy_delta_vs_candidate_pp": float(group.policy_delta_vs_candidate_pp.mean()),
                                          "fraction_disagreement_trials_switched": float(switched.shape[0] / max(1, seed_trials.B0_prediction.ne(seed_trials.candidate_prediction).sum())),
                                          "fraction_all_trials_switched": float(switched.shape[0] / len(seed_trials)),
                                          "realized_rescue_count": rescue, "realized_harm_count": harm,
                                          "switched_both_wrong_count": int(switched.error_state.eq("WW").sum()),
                                          "intervention_precision": float(rescue / (rescue + harm)) if rescue + harm else np.nan})
                    print(f"POLICY {comparison} {task} {feature_set} fold={held_fold} threshold={threshold:.2f}", flush=True)
    folds = pd.DataFrame(fold_rows); subjects = pd.concat(subject_rows, ignore_index=True)
    atomic_csv(outputs / "SELECTIVE_POLICY_FOLD_RESULTS.csv", folds)
    atomic_csv(outputs / "SELECTIVE_POLICY_SUBJECT_RESULTS.csv", subjects)

    task_rows, boot_rows = [], []
    for (comparison, task, feature_set, seed), group in subjects.groupby(["comparison", "task", "feature_set", "seed"], sort=True):
        point, low, high = bootstrap(group.policy_delta_vs_B0_pp.to_numpy(float))
        fold_group = folds[(folds.comparison == comparison) & (folds.task == task) & (folds.feature_set == feature_set) & (folds.seed == seed)]
        rescue, harm = int(group.realized_rescue_count.sum()), int(group.realized_harm_count.sum())
        task_rows.append({"comparison": comparison, "task": task, "candidate": comparison, "seed_scope": f"seed{seed}", "feature_set": feature_set,
                          "B0_BA": float(group.B0_BA.mean()), "candidate_BA": float(group.candidate_BA.mean()), "selective_policy_BA": float(group.policy_BA.mean()),
                          "policy_delta_vs_B0_pp": point, "policy_delta_vs_candidate_pp": float(group.policy_delta_vs_candidate_pp.mean()),
                          "ci_low_pp": low, "ci_high_pp": high, "nonnegative_folds": int(fold_group.policy_delta_vs_B0_pp.ge(0).sum()),
                          "positive_folds": int(fold_group.policy_delta_vs_B0_pp.gt(0).sum()), "worst_fold_pp": float(fold_group.policy_delta_vs_B0_pp.min()),
                          "switch_rate_disagreements": float(group.switched_trials.sum() / max(1, group.disagreement_trials.sum())),
                          "switch_rate_all_trials": float(group.switched_trials.sum() / group.trials.sum()),
                          "intervention_precision": float(rescue / (rescue + harm)) if rescue + harm else np.nan,
                          "realized_rescue_count": rescue, "realized_harm_count": harm,
                          "switched_both_wrong_count": int(group.switched_both_wrong_count.sum()),
                          "oracle_headroom_pp": float(group.oracle_headroom_pp.mean()),
                          "recovered_oracle_fraction": float(point / group.oracle_headroom_pp.mean()) if group.oracle_headroom_pp.mean() > 0 else np.nan})
        boot_rows.append({"comparison": comparison, "task": task, "seed_scope": f"seed{seed}", "feature_set": feature_set,
                          "bootstrap_unit": "real_subject", "bootstrap_resamples": 10000,
                          "point_delta_pp": point, "ci_low_pp": low, "ci_high_pp": high})
    if not subjects[subjects.comparison == "XS"].empty:
        for (task, feature_set), group in subjects[subjects.comparison == "XS"].groupby(["task", "feature_set"], sort=True):
            by_subject = group.groupby("subject_id").agg(policy_delta_vs_B0_pp=("policy_delta_vs_B0_pp", "mean"),
                                                            oracle_headroom_pp=("oracle_headroom_pp", "mean"))
            point, low, high = bootstrap(by_subject.policy_delta_vs_B0_pp.to_numpy(float))
            seed_rows = [r for r in task_rows if r["comparison"] == "XS" and r["task"] == task and r["feature_set"] == feature_set]
            task_rows.append({"comparison": "XS", "task": task, "candidate": "XS", "seed_scope": "equal_seed_mean", "feature_set": feature_set,
                              "B0_BA": float(np.mean([r["B0_BA"] for r in seed_rows])), "candidate_BA": float(np.mean([r["candidate_BA"] for r in seed_rows])),
                              "selective_policy_BA": float(np.mean([r["selective_policy_BA"] for r in seed_rows])),
                              "policy_delta_vs_B0_pp": point, "policy_delta_vs_candidate_pp": float(np.mean([r["policy_delta_vs_candidate_pp"] for r in seed_rows])),
                              "ci_low_pp": low, "ci_high_pp": high,
                              "nonnegative_folds": int(sum(np.mean([folds[(folds.comparison == "XS") & (folds.task == task) & (folds.feature_set == feature_set) & (folds.heldout_fold == f)].policy_delta_vs_B0_pp]) >= 0 for f in range(5))),
                              "positive_folds": int(sum(np.mean([folds[(folds.comparison == "XS") & (folds.task == task) & (folds.feature_set == feature_set) & (folds.heldout_fold == f)].policy_delta_vs_B0_pp]) > 0 for f in range(5))),
                              "worst_fold_pp": float(min(np.mean(folds[(folds.comparison == "XS") & (folds.task == task) & (folds.feature_set == feature_set) & (folds.heldout_fold == f)].policy_delta_vs_B0_pp) for f in range(5))),
                              "switch_rate_disagreements": float(np.mean([r["switch_rate_disagreements"] for r in seed_rows])),
                              "switch_rate_all_trials": float(np.mean([r["switch_rate_all_trials"] for r in seed_rows])),
                              "intervention_precision": float(np.nanmean([r["intervention_precision"] for r in seed_rows])),
                              "realized_rescue_count": int(sum(r["realized_rescue_count"] for r in seed_rows)), "realized_harm_count": int(sum(r["realized_harm_count"] for r in seed_rows)),
                              "switched_both_wrong_count": int(sum(r["switched_both_wrong_count"] for r in seed_rows)),
                              "oracle_headroom_pp": float(by_subject.oracle_headroom_pp.mean()),
                              "recovered_oracle_fraction": float(point / by_subject.oracle_headroom_pp.mean()) if by_subject.oracle_headroom_pp.mean() > 0 else np.nan})
            boot_rows.append({"comparison": "XS", "task": task, "seed_scope": "equal_seed_mean", "feature_set": feature_set,
                              "bootstrap_unit": "real_subject_with_seed_replicas_together", "bootstrap_resamples": 10000,
                              "point_delta_pp": point, "ci_low_pp": low, "ci_high_pp": high})
    atomic_csv(outputs / "SELECTIVE_POLICY_TASK_SUMMARY.csv", pd.DataFrame(task_rows))
    atomic_csv(outputs / "PREDICTABILITY_SUBJECT_BOOTSTRAP.csv", pd.DataFrame(boot_rows))
    print(f"SELECTIVE_POLICY_COMPLETE fold_rows={len(folds)} subject_rows={len(subjects)}", flush=True)


if __name__ == "__main__":
    main()
