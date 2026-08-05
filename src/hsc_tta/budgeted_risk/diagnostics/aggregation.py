from __future__ import annotations

import numpy as np
import pandas as pd

from hsc_tta.contextual_risk.statistics import clopper_pearson_upper, paired_bootstrap_ci


def subject_efficiency(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Aggregate repeats, then seeds, leaving exactly one row per subject."""
    keys = ["dataset", "subject_id", "requested_budget", "strategy", "calibration_scheme"]
    numeric = [c for c in columns if c in frame]
    repeat_mean = frame.groupby(keys + ["seed"], as_index=False)[numeric].mean()
    return repeat_mean.groupby(keys, as_index=False)[numeric].mean()


def subject_paired_ci(values: pd.Series | np.ndarray, reps: int = 5000, seed: int = 20260805) -> tuple[float, float]:
    return paired_bootstrap_ci(np.asarray(values, float), reps=reps, seed=seed)


def seedwise_validity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, current in frame.groupby("seed"):
        by_subject = current.groupby("subject_id", as_index=False).violation.mean()
        bernoulli = (by_subject.violation > 0).astype(int)
        violations = int(bernoulli.sum()); n = len(bernoulli)
        rows.append({"seed": int(seed), "n_subjects": n, "violations": violations,
                     "violation": violations / n, "cp_upper": clopper_pearson_upper(violations, n, .95)})
    return pd.DataFrame(rows)

