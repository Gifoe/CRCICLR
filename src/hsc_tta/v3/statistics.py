from __future__ import annotations

import numpy as np
import pandas as pd


def average_seeds_by_subject(frame: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    required = {"dataset", "subject_id", "seed", *value_columns}
    if missing := required - set(frame):
        raise ValueError(f"missing aggregation columns: {sorted(missing)}")
    return frame.groupby(["dataset", "subject_id"], as_index=False)[value_columns].mean()


def subject_cluster_bootstrap(frame: pd.DataFrame, value: str, *, repetitions: int = 2000,
                              seed: int = 4301) -> dict[str, float]:
    if frame.subject_id.duplicated().any():
        raise ValueError("average repeated seeds before subject bootstrap")
    values = frame[value].to_numpy(float)
    if len(values) == 0:
        raise ValueError("no subjects")
    rng = np.random.default_rng(seed); draw = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[draw].mean(1)
    return {"mean": float(values.mean()), "ci_lower": float(np.quantile(means, .025)),
            "ci_upper": float(np.quantile(means, .975)), "n_subjects": int(len(values)),
            "bootstrap_repetitions": int(repetitions)}


def paired_subject_bootstrap(frame: pd.DataFrame, proposed: str, baseline: str, *, repetitions: int = 2000,
                             seed: int = 4302) -> dict[str, float]:
    pivot = frame.pivot(index="subject_id", columns="policy", values="average_set_size")
    if proposed not in pivot or baseline not in pivot or pivot[[proposed, baseline]].isna().any().any():
        raise ValueError("paired policies must cover identical subjects")
    difference = (pivot[baseline] - pivot[proposed]).to_numpy(float)
    rng = np.random.default_rng(seed); draw = rng.integers(0, len(difference), size=(repetitions, len(difference)))
    means = difference[draw].mean(1)
    return {"mean_set_size_reduction": float(difference.mean()), "ci_lower": float(np.quantile(means, .025)),
            "ci_upper": float(np.quantile(means, .975)), "n_subjects": int(len(difference))}
