from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGWindowDataset(Dataset):
    """In-memory subject cache; all baselines consume the exact same arrays."""

    def __init__(self, paths: Iterable[str | Path]):
        signals, labels, subjects, recordings = [], [], [], []
        self.channel_names: list[str] | None = None
        self.sampling_rate: float | None = None
        for path_like in sorted(map(Path, paths)):
            with h5py.File(path_like) as handle:
                names = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["channel_names"][:]]
                if self.channel_names is None:
                    self.channel_names = names
                    self.sampling_rate = float(handle["sampling_rate"][()])
                elif names != self.channel_names:
                    raise ValueError("channel order differs across cached subjects")
                current = np.asarray(handle["signal"][:], dtype=np.float32)
                current_labels = np.asarray(handle["label"][:], dtype=np.int64)
                signals.append(current); labels.append(current_labels)
                subjects.extend([path_like.stem] * len(current_labels))
                recordings.extend([value.decode() if isinstance(value, bytes) else str(value) for value in handle["recording_id"][:]])
        if not signals:
            raise ValueError("empty EEG dataset")
        self.signal = np.concatenate(signals)
        self.label = np.concatenate(labels)
        self.subjects = np.asarray(subjects)
        self.recordings = np.asarray(recordings)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return torch.from_numpy(self.signal[index]), torch.tensor(self.label[index], dtype=torch.long), int(index)


def fixed_subject_split(paths: Iterable[str | Path], seed: int = 2027, train_fraction: float = 0.65, validation_fraction: float = 0.15) -> dict[str, list[str]]:
    values = sorted(str(Path(path)) for path in paths)
    rng = np.random.default_rng(seed)
    order = np.asarray(values, dtype=object); rng.shuffle(order)
    train_end = int(round(len(order) * train_fraction))
    validation_end = train_end + int(round(len(order) * validation_fraction))
    return {
        "train": order[:train_end].tolist(),
        "validation": order[train_end:validation_end].tolist(),
        "test": order[validation_end:].tolist(),
    }
