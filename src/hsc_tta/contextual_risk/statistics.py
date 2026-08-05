from __future__ import annotations

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(violations: int, n: int, confidence: float = .95) -> float:
    if not 0 <= violations <= n or n <= 0:
        raise ValueError("invalid binomial counts")
    return 1.0 if violations == n else float(beta.ppf(confidence, violations + 1, n - violations))


def paired_bootstrap_ci(values: np.ndarray, *, reps: int = 5000, seed: int = 20260805) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("bootstrap input must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(1)
    return tuple(map(float, np.quantile(means, [.025, .975])))
