from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass
class QueryOracle:
    dataset: str
    subject_id: str
    seed: int
    sample_indices: np.ndarray
    _labels: np.ndarray = field(repr=False)
    budget: int = 0
    strategy: str = "none"
    _transcript: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _queried: set[int] = field(default_factory=set, init=False, repr=False)
    _frozen: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.sample_indices = np.asarray(self.sample_indices, dtype=int).copy()
        self._labels = np.asarray(self._labels, dtype=int).copy()
        if len(self.sample_indices) != len(self._labels):
            raise ValueError("context index/label length mismatch")
        self.budget = min(max(int(self.budget), 0), len(self.sample_indices))

    def query(self, sample_index: int, **metadata: Any) -> int:
        if self._frozen:
            raise RuntimeError("QUERY_FROZEN")
        sample_index = int(sample_index)
        positions = np.flatnonzero(self.sample_indices == sample_index)
        if not len(positions):
            raise PermissionError("sample is outside context")
        if sample_index in self._queried:
            return int(self._labels[positions[0]])
        if len(self._queried) >= self.budget:
            raise RuntimeError("query budget exhausted")
        position = int(positions[0]); label = int(self._labels[position])
        if "kappa_by_label" in metadata:
            kappa_by_label = np.asarray(metadata["kappa_by_label"], dtype=int)
            observed_kappa = int(kappa_by_label[label])
        else:
            observed_kappa = int(metadata.get("observed_kappa", -1))
        row = {
            "dataset": self.dataset, "subject_id": self.subject_id, "seed": self.seed,
            "budget": self.budget, "strategy": self.strategy, "step": len(self._transcript),
            "sample_index": sample_index, "time_position": position,
            "predicted_class": int(metadata.get("predicted_class", -1)),
            "risk_index_entropy": float(metadata.get("risk_index_entropy", np.nan)),
            "selected_score": float(metadata.get("selected_score", np.nan)),
            "observed_label": label, "observed_kappa": observed_kappa,
        }
        row["transcript_hash"] = _hash({"previous": self._transcript[-1]["transcript_hash"] if self._transcript else None, **row})
        self._transcript.append(row); self._queried.add(sample_index)
        return label

    @property
    def transcript(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._transcript)

    @property
    def queried_count(self) -> int:
        return len(self._queried)

    def freeze(self) -> str:
        self._frozen = True
        return _hash(list(self._transcript))

    def verify(self, transcript: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> bool:
        previous = None
        for row in transcript:
            current = dict(row); given = current.pop("transcript_hash", None)
            if given != _hash({"previous": previous, **current}):
                return False
            previous = given
        return _hash(list(transcript)) == _hash(list(self._transcript))
