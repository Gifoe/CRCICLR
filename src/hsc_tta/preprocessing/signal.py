from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from fractions import Fraction


def preprocess_signal(signal: np.ndarray, source_rate: float, target_rate: float, bandpass: tuple[float, float]) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2 or min(source_rate, target_rate) <= 0:
        raise ValueError("signal must be channels x time with positive sampling rates")
    low, high = bandpass
    if not 0 < low < high < source_rate / 2:
        raise ValueError("invalid bandpass for source sampling rate")
    sos = butter(4, [low, high], btype="bandpass", fs=source_rate, output="sos")
    filtered = sosfiltfilt(sos, x, axis=-1)
    ratio = Fraction(target_rate / source_rate).limit_denominator(1000)
    return resample_poly(filtered, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)


def quality_flags(signal: np.ndarray) -> dict[str, float | bool]:
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x)
    return {"nonfinite_rate": float(1 - finite.mean()), "flat_channel": bool(np.any(np.nanstd(x, axis=-1) < 1e-10)), "peak_abs": float(np.nanmax(np.abs(x))) if np.any(finite) else float("nan")}

