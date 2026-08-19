"""Calibration helpers.

Threshold calibration is implemented centrally in ``training.py`` so no
model family can accidentally inspect an outer evaluation fold.
"""

from training import THRESHOLDS

__all__ = ["THRESHOLDS"]
