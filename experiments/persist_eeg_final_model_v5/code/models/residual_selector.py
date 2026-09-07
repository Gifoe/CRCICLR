"""B-strong-anchored residual probability correction."""

from __future__ import annotations

import numpy as np


def apply(base_probability: np.ndarray, residual: np.ndarray, alpha: float) -> np.ndarray:
    value = np.asarray(base_probability, dtype=float) + float(alpha) * np.asarray(residual, dtype=float)
    return np.clip(value, 1e-7, 1 - 1e-7)
