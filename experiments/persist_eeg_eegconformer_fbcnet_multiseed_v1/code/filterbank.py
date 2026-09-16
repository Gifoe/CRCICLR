"""FBCNet's fixed, training-independent 4-Hz Chebyshev-II filter bank.

The official FBCNet release applies nine 4-Hz bands from 4 to 40 Hz,
Chebyshev-II stop attenuation 30 dB, pass ripple 3 dB, 2-Hz allowance,
and causal scipy.signal.lfilter. This is a fixed model adapter, not a
dataset-level fitted preprocessing step. No labels or target statistics enter.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import signal


BANDS = tuple((low, low + 4) for low in range(4, 40, 4))
FS = 250


def coefficients() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    values = []
    for low, high in BANDS:
        passband = [low / (FS / 2), high / (FS / 2)]
        stopband = [(low - 2) / (FS / 2), (high + 2) / (FS / 2)]
        order, _ = signal.cheb2ord(passband, stopband, 3, 30)
        # The official FBCNet implementation deliberately passes the fixed
        # stop-band edges, not cheb2ord's returned natural frequency, to
        # cheby2. Preserve that exact published adapter contract.
        values.append(signal.cheby2(order, 30, stopband, btype="bandpass"))
    return tuple(values)


COEFFICIENTS = coefficients()
FILTER_SHA256 = hashlib.sha256(b"".join(
    array.tobytes() for pair in COEFFICIENTS for array in pair)).hexdigest()


def transform(batch: np.ndarray) -> np.ndarray:
    """Return [trials, bands, channels, samples] float32 for [N,C,T]."""
    if batch.ndim != 3:
        raise ValueError("filter-bank input must be [N,C,T]")
    result = np.empty((len(batch), len(BANDS), batch.shape[1], batch.shape[2]), dtype=np.float32)
    for band, (numerator, denominator) in enumerate(COEFFICIENTS):
        filtered = signal.lfilter(numerator, denominator, batch.astype(np.float64, copy=False), axis=-1)
        if not np.isfinite(filtered).all():
            raise RuntimeError(f"non-finite FBCNet filter output in band {BANDS[band]}")
        result[:, band] = filtered.astype(np.float32)
    return result


def make_memmap(raw: np.ndarray, destination: Path, identity: dict, chunk: int = 96) -> np.memmap:
    """Build a resumable fold-local cache; all values are deterministic."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta_path = destination.with_suffix(".json")
    metadata = {**identity, "shape": [len(raw), len(BANDS), raw.shape[1], raw.shape[2]],
                "dtype": "float32", "filter_sha256": FILTER_SHA256}
    if destination.exists() and meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError(f"filter-bank cache identity mismatch: {destination}")
        existing = np.load(destination, mmap_mode="r")
        if list(existing.shape) != metadata["shape"]:
            raise RuntimeError(f"filter-bank cache shape mismatch: {destination}")
        return existing
    temporary = destination.with_suffix(".npy.part")
    if temporary.exists():
        temporary.unlink()  # incomplete scratch owned by this exact cell
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                        shape=tuple(metadata["shape"]))
    for start in range(0, len(raw), chunk):
        output[start:start + chunk] = transform(raw[start:start + chunk])
    output.flush()
    del output
    os.replace(temporary, destination)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return np.load(destination, mmap_mode="r")
