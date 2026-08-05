"""Closed V5.1 diagnostics for few-label risk prediction and calibration."""

from .calibration_schemes import (
    SCHEMES,
    conformal_q,
    fold_split,
    is_outlier_driven,
    sentinel_transition,
)

__all__ = [
    "SCHEMES", "conformal_q", "fold_split", "is_outlier_driven",
    "sentinel_transition",
]

