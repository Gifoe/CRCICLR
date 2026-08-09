from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyCalibration:
    lambda_index: int
    order_index: int
    calibration_size: int
    insufficient: bool
    sentinel_index: int


def joint_critical_index(risks: np.ndarray, degradation: float, *, alpha: float, epsilon: float,
                         sentinel_index: int) -> int:
    values = np.asarray(risks, dtype=float)
    if values.ndim != 1 or len(values) != sentinel_index + 1 or np.any(~np.isfinite(values)):
        raise ValueError("risk curve must include all nontrivial indices and sentinel")
    if degradation > epsilon:
        return sentinel_index
    eligible = np.flatnonzero(values[:sentinel_index] <= alpha)
    return int(eligible[0]) if len(eligible) else sentinel_index


def calibrate_policy_index(indices: np.ndarray, *, delta: float, sentinel_index: int) -> PolicyCalibration:
    values = np.asarray(indices, dtype=int)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values > sentinel_index)):
        raise ValueError("invalid calibration joint indices")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    m = len(values); k = math.ceil((m + 1) * (1 - delta))
    if k > m:
        return PolicyCalibration(sentinel_index, k, m, True, sentinel_index)
    selected = int(np.sort(values, kind="stable")[k - 1])
    return PolicyCalibration(selected, k, m, False, sentinel_index)
