from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from scipy.signal import resample_poly


SLEEP_CHANNEL = {"hmc": "EEG C4-M1", "cap": "C4-A1"}


@dataclass(frozen=True)
class AdapterBatch:
    tensor: torch.Tensor
    channel_mask: np.ndarray
    input_valid_mask: np.ndarray
    metadata: dict[str, object]


class CBraModInputAdapter:
    """Deterministic bridge from frozen subject HDF5 arrays to official CBraMod input."""

    patch_size = 200
    target_rate = 200.0
    embedding_dim = 200

    def __init__(self, normalization_scale: float = 1.0e4):
        self.normalization_scale = float(normalization_scale)

    @property
    def config(self) -> dict[str, object]:
        return {
            "version": "cbramod-adapter-v1",
            "target_rate_hz": self.target_rate,
            "patch_size": self.patch_size,
            "normalization": "cached volts * 1e4 (official downstream uV/100)",
            "sleep_channels": SLEEP_CHANNEL,
            "sleep_pooling": "mean over channel and 30 patch representations",
            "mi_reference": "average reference across official 64 channels",
            "mi_pooling": "mean over 64 channels and 4 patch representations",
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def adapt(self, dataset: str, signal: np.ndarray, channel_names: list[str], sampling_rate: float,
              *, device: str | torch.device = "cpu") -> AdapterBatch:
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim != 3 or not np.isfinite(x).all():
            raise ValueError("signal must be finite [windows, channels, samples]")
        if dataset in SLEEP_CHANNEL:
            expected = SLEEP_CHANNEL[dataset].lower()
            normalized = [name.strip().lower() for name in channel_names]
            if expected not in normalized:
                raise ValueError(f"required formal channel missing: {SLEEP_CHANNEL[dataset]}")
            index = normalized.index(expected)
            x = x[:, index : index + 1, :]
            if not np.isclose(sampling_rate, self.target_rate) or x.shape[-1] != 6000:
                raise ValueError("sleep cache must be exactly 30 seconds at 200 Hz")
            x = x.reshape(x.shape[0], 1, 30, self.patch_size)
            reference = "C4-M1" if dataset == "hmc" else "C4-A1"
        elif dataset == "eegmmidb":
            if len(channel_names) != 64 or x.shape[1] != 64:
                raise ValueError("EEGMMIDB requires the complete official 64-channel order")
            if not np.isclose(sampling_rate, 160.0) or x.shape[-1] != 640:
                raise ValueError("EEGMMIDB cache must be exactly 4 seconds at 160 Hz")
            x = x - x.mean(axis=1, keepdims=True)
            x = resample_poly(x, 5, 4, axis=-1).astype(np.float32, copy=False)
            if x.shape[-1] != 800:
                raise RuntimeError("MI resampling did not produce 800 samples")
            x = x.reshape(x.shape[0], 64, 4, self.patch_size)
            reference = "average"
        else:
            raise ValueError(f"unknown dataset: {dataset}")
        x = np.ascontiguousarray(x * self.normalization_scale, dtype=np.float32)
        valid = np.isfinite(x).all(axis=-1)
        if not valid.all():
            raise ValueError("adapter produced invalid patches")
        return AdapterBatch(torch.from_numpy(x).to(device), np.ones(x.shape[1], bool), valid,
                            {"reference": reference, "input_shape": list(x.shape[1:])})
