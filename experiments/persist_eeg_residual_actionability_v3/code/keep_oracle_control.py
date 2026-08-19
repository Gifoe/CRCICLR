from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from v3_common import logit, sigmoid


def build_keep_candidates(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial_index, trial in trials.iterrows():
        margins = np.asarray(trial.keep_margins, dtype=float)
        probabilities = sigmoid(margins)
        predictions = (margins >= 0).astype(int)
        n_runs = len(margins)
        definitions = [
            ("B6_ALL_RUN_LOGIT_MEAN", "ALL_RUN", float(margins.mean()), int(margins.mean() >= 0), None),
            (
                "B4_ALL_RUN_PROBABILITY_MEAN",
                "ALL_RUN",
                float(logit(np.asarray([probabilities.mean()]))[0]),
                int(probabilities.mean() >= 0.5),
                None,
            ),
            (
                "B2_ALL_RUN_HARD_MAJORITY",
                "ALL_RUN",
                float(predictions.mean() - 0.5),
                int(predictions.mean() >= 0.5),
                None,
            ),
        ]
        for position, run_id in enumerate(trial.run_ids):
            other_margin = float((margins.sum() - margins[position]) / (n_runs - 1))
            definitions.extend(
                [
                    (
                        f"LEAVE_{run_id}_OUT_KEEP_LOGIT_MEAN",
                        "LEAVE_ONE_RUN_OUT",
                        other_margin,
                        int(other_margin >= 0),
                        run_id,
                    ),
                    (
                        f"INDIVIDUAL_{run_id}_KEEP",
                        "INDIVIDUAL_RUN",
                        float(margins[position]),
                        int(margins[position] >= 0),
                        run_id,
                    ),
                ]
            )
        for priority, (candidate_id, family, margin, prediction, run_id) in enumerate(definitions):
            rows.append(
                {
                    "trial_index": int(trial_index),
                    "trial_uid": trial.trial_uid,
                    "candidate_id": candidate_id,
                    "keep_family": family,
                    "run_id": run_id,
                    "priority": priority,
                    "margin": margin,
                    "probability": float(sigmoid(np.asarray([margin]))[0]),
                    "prediction": prediction,
                    "n_runs": n_runs,
                    "OUTER_TEST_USED": False,
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.duplicated(["trial_index", "candidate_id"]).any():
        raise RuntimeError("Duplicate KEEP-only candidate within a trial")
    return candidates


def keep_only_rescue_mask(trials: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    labels = trials.outcome_label.to_numpy(dtype=int)
    b6 = trials.y_keep_ens.to_numpy(dtype=int)
    candidate_correct = candidates.prediction.to_numpy(dtype=int) == labels[candidates.trial_index.to_numpy(dtype=int)]
    available = np.zeros(len(trials), dtype=bool)
    np.logical_or.at(available, candidates.trial_index.to_numpy(dtype=int), candidate_correct)
    return (b6 != labels) & available
