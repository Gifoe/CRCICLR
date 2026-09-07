"""Conservative PERSIST prior/safety utilities.

V5 did not obtain predictive BA value from PERSIST.  These functions encode
the only retained use: a train-only prior bias for KEEP experts and a fail-
closed veto for actions whose protected-risk estimate exceeds its frozen
limit.
"""

from __future__ import annotations

import numpy as np


def reliability_bias(score: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    prior = np.clip(np.asarray(prior, dtype=float), 1e-6, 1 - 1e-6)
    return score + float(strength) * (np.log(prior) - np.log1p(-prior))


def safe_action_mask(predicted_utility: np.ndarray, protected_risk: np.ndarray, risk_limit: float) -> np.ndarray:
    return (np.asarray(predicted_utility, dtype=float) > 0) & (
        np.asarray(protected_risk, dtype=float) <= float(risk_limit)
    )
