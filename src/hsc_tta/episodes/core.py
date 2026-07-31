from __future__ import annotations

import numpy as np


def validate_episode(context_indices: list[int], future_indices: list[int]) -> None:
    if set(context_indices) & set(future_indices):
        raise ValueError("U_s/V_s leakage: overlapping indices")


def build_sleep_episode(window_start_seconds: np.ndarray, valid_scored: np.ndarray, context_minutes: int = 90, minimum_future_epochs: int = 240) -> dict[str, object]:
    starts = np.asarray(window_start_seconds, dtype=float)
    valid = np.asarray(valid_scored, dtype=bool)
    if starts.shape != valid.shape or starts.ndim != 1 or not np.any(valid):
        raise ValueError("aligned one-dimensional starts and at least one valid epoch required")
    first = float(starts[np.flatnonzero(valid)[0]])
    boundary = first + context_minutes * 60
    context = np.flatnonzero(valid & (starts >= first) & (starts < boundary)).tolist()
    future = np.flatnonzero(valid & (starts >= boundary)).tolist()
    validate_episode(context, future)
    exclusion = None if len(future) >= minimum_future_epochs else "insufficient_future_epochs"
    return {"protocol": "first_valid_clock_context", "context_indices": context, "future_indices": future, "n_context": len(context), "n_future": len(future), "exclusion_reason": exclusion}


def build_mi_episode(run_ids: np.ndarray, context_runs: tuple[int, ...] = (4, 6), future_runs: tuple[int, ...] = (8, 10, 12, 14)) -> dict[str, object]:
    runs = np.asarray(run_ids, dtype=int)
    missing = sorted((set(context_runs) | set(future_runs)) - set(runs.tolist()))
    context = np.flatnonzero(np.isin(runs, context_runs)).tolist()
    future = np.flatnonzero(np.isin(runs, future_runs)).tolist()
    validate_episode(context, future)
    return {"protocol": "run_disjoint", "context_indices": context, "future_indices": future, "context_runs": list(context_runs), "future_runs": list(future_runs), "n_context": len(context), "n_future": len(future), "exclusion_reason": f"missing_runs:{missing}" if missing else None}

