from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import stable_seed
from metrics import policy_metrics, score_diagnostics, select_actions


FULL_MENU = ("amplify", "geometry", "erase")
PROTECTED_SAFE_MENU = ("amplify", "geometry")
AMPLIFY_ONLY_MENU = ("amplify",)


@dataclass
class Evaluation:
    policy_id: str
    selected: np.ndarray
    score: np.ndarray
    metrics: dict[str, Any]
    diagnostics: dict[str, float]
    thresholds: list[float]


def _candidate_thresholds(kind: str) -> np.ndarray:
    if kind == "confidence":
        return np.r_[np.linspace(0.05, 0.49, 23), 1.1]
    return np.r_[np.linspace(0.10, 0.90, 33), 1.1]


def _choose_calibration_threshold(
    frame: pd.DataFrame,
    score: np.ndarray,
    kind: str,
    oracle_gain: float,
) -> float:
    rows: list[tuple[float, dict[str, Any]]] = []
    for threshold in _candidate_thresholds(kind):
        selected = select_actions(frame, score >= threshold, FULL_MENU)
        metrics = policy_metrics(frame, selected, oracle_gain=oracle_gain, bootstrap_repetitions=0)
        rows.append((float(threshold), metrics))
    safe = [
        item
        for item in rows
        if item[1]["action_rate"] > 0
        and item[1]["rescue_precision"] > item[1]["unsafe_intervention_rate"]
        and item[1]["mean_subject_delta_BA"] > 0
    ]
    if not safe:
        return 1.1
    safe.sort(
        key=lambda item: (
            item[1]["mean_subject_delta_BA"],
            -item[1]["unsafe_intervention_rate"],
            -item[1]["action_rate"],
        ),
        reverse=True,
    )
    return safe[0][0]


def _logistic_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.03, max_iter=1500, random_state=seed)),
        ]
    )


def _fit_logistic(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> None:
    model.fit(
        frame[features],
        frame.target_baseline_error.to_numpy(dtype=int),
        model__sample_weight=frame.unit_weight.to_numpy(dtype=float),
    )


def nested_single_run_evaluation(
    frame: pd.DataFrame,
    features: list[str],
    kind: str,
    oracle_gain: float,
) -> Evaluation:
    if kind not in ("confidence", "logistic"):
        raise ValueError(kind)
    oof_score = np.full(len(frame), np.nan, dtype=float)
    oof_selected = np.full(len(frame), "noop", dtype=object)
    thresholds: list[float] = []
    for outer_fold in range(5):
        validation = frame.exploration_cv_fold.eq(outer_fold).to_numpy()
        calibration = frame.exploration_cv_fold.eq((outer_fold + 1) % 5).to_numpy()
        training = ~(validation | calibration)
        if kind == "confidence":
            calibration_score = 1.0 - frame.loc[calibration, "confidence_noop"].to_numpy(dtype=float)
            validation_score = 1.0 - frame.loc[validation, "confidence_noop"].to_numpy(dtype=float)
        else:
            model = _logistic_model(stable_seed("single-run-logistic", outer_fold))
            _fit_logistic(model, frame.loc[training], features)
            calibration_score = model.predict_proba(frame.loc[calibration, features])[:, 1]
            validation_score = model.predict_proba(frame.loc[validation, features])[:, 1]
        threshold = _choose_calibration_threshold(
            frame.loc[calibration].reset_index(drop=True), calibration_score, kind, oracle_gain
        )
        thresholds.append(threshold)
        oof_score[validation] = validation_score
        oof_selected[validation] = select_actions(
            frame.loc[validation].reset_index(drop=True), validation_score >= threshold, FULL_MENU
        )
    if np.isnan(oof_score).any():
        raise RuntimeError("Nested exploration did not produce every out-of-fold score")
    policy_id = "I001_SINGLE_RUN_CONFIDENCE" if kind == "confidence" else "I002_SINGLE_RUN_LOGISTIC_ERROR"
    return Evaluation(
        policy_id=policy_id,
        selected=oof_selected,
        score=oof_score,
        metrics=policy_metrics(frame, oof_selected, oracle_gain=oracle_gain, seed_offset=1 if kind == "confidence" else 2),
        diagnostics=score_diagnostics(frame, oof_score),
        thresholds=thresholds,
    )


def consensus_evaluation(
    frame: pd.DataFrame,
    policy_id: str,
    menu: tuple[str, ...],
    oracle_gain: float,
    seed_offset: int,
) -> Evaluation:
    score = frame.other_run_base_disagrees.to_numpy(dtype=float)
    selected = select_actions(frame, score > 0.5, menu)
    return Evaluation(
        policy_id=policy_id,
        selected=selected,
        score=score,
        metrics=policy_metrics(frame, selected, oracle_gain=oracle_gain, seed_offset=seed_offset),
        diagnostics=score_diagnostics(frame, score),
        thresholds=[0.5],
    )


def meets_strong_candidate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["mean_subject_delta_BA"] >= 0.01
        and metrics["bootstrap_CI95_L"] > 0
        and metrics["nonnegative_run_fraction"] >= 5 / 6
        and metrics["positive_run_fraction"] >= 4 / 6
        and metrics["action_rate"] > 0
        and metrics["rescue_precision"] > metrics["unsafe_intervention_rate"]
        and metrics["recovered_oracle_headroom"] >= 0.05
    )


def pareto_frontier(results: pd.DataFrame) -> pd.DataFrame:
    candidates = results[results.policy_id.ne("M0_KEEP")].copy().reset_index(drop=True)
    keep: list[bool] = []
    for index, row in candidates.iterrows():
        dominated = False
        for other_index, other in candidates.iterrows():
            if index == other_index:
                continue
            no_worse = (
                other.mean_subject_delta_BA >= row.mean_subject_delta_BA
                and other.unsafe_intervention_rate <= row.unsafe_intervention_rate
                and other.action_rate <= row.action_rate
                and other.recovered_oracle_headroom >= row.recovered_oracle_headroom
                and other.positive_run_fraction >= row.positive_run_fraction
            )
            strictly_better = (
                other.mean_subject_delta_BA > row.mean_subject_delta_BA
                or other.unsafe_intervention_rate < row.unsafe_intervention_rate
                or other.action_rate < row.action_rate
                or other.positive_run_fraction > row.positive_run_fraction
            )
            if no_worse and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    return candidates.loc[keep].sort_values("mean_subject_delta_BA", ascending=False).reset_index(drop=True)

