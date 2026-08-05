from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def higher_quantile(values: Iterable[float], probability: float) -> float:
    values = np.sort(np.asarray(list(values), dtype=float))
    if len(values) == 0:
        raise ValueError("quantile requires at least one value")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0,1]")
    k = max(1, math.ceil(probability * len(values)))
    return float(values[k - 1])


def split_conformal_upper(
    values: Iterable[float], delta: float, *, insufficient: float = math.inf
) -> float:
    values = np.sort(np.asarray(list(values), dtype=float))
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    m = len(values)
    k = math.ceil((m + 1) * (1 - delta))
    return float(insufficient if k > m else values[k - 1])
