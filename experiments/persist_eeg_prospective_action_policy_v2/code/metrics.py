from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from common import SEED


def select_actions(frame: pd.DataFrame, intervene: Iterable[bool], menu: tuple[str, ...]) -> np.ndarray:
    intervene_array = np.asarray(intervene, dtype=bool)
    selected = np.full(len(frame), "noop", dtype=object)
    for action in menu:
        available = frame[f"pred_{action}"].to_numpy() != frame.pred_noop.to_numpy()
        take = intervene_array & (selected == "noop") & available
        selected[take] = action
    return selected


def oracle_actions(frame: pd.DataFrame, menu: tuple[str, ...]) -> np.ndarray:
    selected = np.full(len(frame), "noop", dtype=object)
    for action in menu:
        take = (selected == "noop") & (frame[f"effect_{action}"].to_numpy() > 0)
        selected[take] = action
    return selected


def _realized(frame: pd.DataFrame, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = frame.pred_noop.to_numpy(dtype=int).copy()
    effect = np.zeros(len(frame), dtype=float)
    for action in ("erase", "amplify", "geometry"):
        mask = selected == action
        prediction[mask] = frame.loc[mask, f"pred_{action}"].to_numpy(dtype=int)
        effect[mask] = frame.loc[mask, f"effect_{action}"].to_numpy(dtype=float)
    return prediction, effect


def policy_tables(frame: pd.DataFrame, selected: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction, effect = _realized(frame, selected)
    labels = frame.outcome_label.to_numpy(dtype=int)
    baseline = frame.pred_noop.to_numpy(dtype=int)
    subject_run_rows: list[dict[str, Any]] = []
    for (fold, seed, subject), indices in frame.groupby(["fold_id", "seed_id", "subject_id"]).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        delta = balanced_accuracy_score(labels[idx], prediction[idx]) - balanced_accuracy_score(labels[idx], baseline[idx])
        subject_run_rows.append(
            {"fold_id": int(fold), "seed_id": int(seed), "subject_id": str(subject), "delta_BA": float(delta)}
        )
    subject_run = pd.DataFrame(subject_run_rows)
    subject = subject_run.groupby("subject_id", as_index=False).agg(
        delta_BA=("delta_BA", "mean"), available_runs=("delta_BA", "size")
    )
    run = subject_run.groupby(["fold_id", "seed_id"], as_index=False).agg(
        delta_BA=("delta_BA", "mean"), subjects=("subject_id", "nunique")
    )
    action_rows: list[dict[str, Any]] = []
    for action in ("noop", "amplify", "geometry", "erase"):
        mask = selected == action
        action_rows.append(
            {
                "action": action.upper(),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "mean_effect": float(effect[mask].mean()) if mask.any() else np.nan,
                "rescue_count": int(np.sum(effect[mask] > 0)),
                "harm_count": int(np.sum(effect[mask] < 0)),
            }
        )
    return subject, run, pd.DataFrame(action_rows)


def policy_metrics(
    frame: pd.DataFrame,
    selected: np.ndarray,
    *,
    oracle_gain: float | None = None,
    bootstrap_repetitions: int = 3000,
    seed_offset: int = 0,
) -> dict[str, Any]:
    _, effect = _realized(frame, selected)
    subject, run, _ = policy_tables(frame, selected)
    subject_values = subject.delta_BA.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED + seed_offset)
    if bootstrap_repetitions:
        draws = rng.choice(subject_values, size=(bootstrap_repetitions, len(subject_values)), replace=True).mean(axis=1)
        ci_l, ci_u = np.quantile(draws, (0.025, 0.975))
    else:
        ci_l = ci_u = np.nan
    intervened = selected != "noop"
    mean_gain = float(subject_values.mean())
    return {
        "mean_subject_delta_BA": mean_gain,
        "bootstrap_CI95_L": float(ci_l),
        "bootstrap_CI95_U": float(ci_u),
        "median_subject_delta_BA": float(np.median(subject_values)),
        "worst_subject_delta_BA": float(subject_values.min()),
        "positive_subject_fraction": float(np.mean(subject_values > 0)),
        "nonnegative_subject_fraction": float(np.mean(subject_values >= 0)),
        "mean_run_subject_delta_BA": float(
            np.average(subject.delta_BA.to_numpy(dtype=float), weights=subject.available_runs.to_numpy(dtype=float))
        ),
        "positive_run_fraction": float(np.mean(run.delta_BA > 0)),
        "nonnegative_run_fraction": float(np.mean(run.delta_BA >= 0)),
        "worst_run_delta_BA": float(run.delta_BA.min()),
        "action_rate": float(np.mean(intervened)),
        "unsafe_intervention_rate": float(np.mean(effect[intervened] < 0)) if intervened.any() else 0.0,
        "rescue_precision": float(np.mean(effect[intervened] > 0)) if intervened.any() else 0.0,
        "mean_effect_when_intervene": float(effect[intervened].mean()) if intervened.any() else 0.0,
        "recovered_oracle_headroom": mean_gain / oracle_gain if oracle_gain and oracle_gain > 0 else np.nan,
        "subjects": int(len(subject)),
        "runs": int(len(run)),
        "OUTER_TEST_USED": False,
    }


def score_diagnostics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    flippable = frame.flip_count.to_numpy() > 0
    target = frame.target_baseline_error.to_numpy(dtype=int)[flippable]
    values = np.asarray(score, dtype=float)[flippable]
    if len(np.unique(target)) < 2:
        return {"AUPRC_rescue": np.nan, "AUROC_rescue": np.nan, "AUPRC_harm": np.nan}
    return {
        "AUPRC_rescue": float(average_precision_score(target, values)),
        "AUROC_rescue": float(roc_auc_score(target, values)),
        "AUPRC_harm": float(average_precision_score(1 - target, 1 - values)),
    }
