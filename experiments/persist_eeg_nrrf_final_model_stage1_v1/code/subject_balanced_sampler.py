"""Deterministic K-subject/M-trial sampler for NRRF-v1.

Every emitted batch has a stable within-subject ordering, so the same manifest
is used byte-for-byte by JOINT-CE and NRRF-v1 for a dataset/fold/seed cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


K_SUBJECTS = 8
M_TRIALS = 16


@dataclass(frozen=True)
class SubjectBatch:
    subject_ids: list[str]
    indices: list[int]
    fallbacks: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SubjectBalancedSampler:
    def __init__(self, subject_to_indices: dict[str, np.ndarray], labels: np.ndarray, seed: int) -> None:
        self.subject_to_indices = {str(k): np.asarray(v, dtype=np.int64) for k, v in subject_to_indices.items()}
        self.labels = np.asarray(labels, dtype=np.int64)
        self.seed = int(seed)
        if not self.subject_to_indices:
            raise RuntimeError("no source-training subjects")

    def _take(self, rng: np.random.Generator, indices: np.ndarray, count: int) -> tuple[list[int], bool]:
        replace = len(indices) < count
        return rng.choice(indices, size=count, replace=replace).astype(np.int64).tolist(), replace

    def one(self, epoch: int, step: int) -> SubjectBatch:
        # Independent generator per batch makes resume independent of process history.
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + step * 9_973)
        all_subjects = np.asarray(sorted(self.subject_to_indices), dtype=object)
        k = min(K_SUBJECTS, len(all_subjects))
        chosen = rng.choice(all_subjects, size=k, replace=False).astype(str).tolist()
        all_indices: list[int] = []
        fallbacks: list[dict[str, Any]] = []
        for subject in chosen:
            pool = self.subject_to_indices[subject]
            zero, one = pool[self.labels[pool] == 0], pool[self.labels[pool] == 1]
            if len(zero) >= 8 and len(one) >= 8:
                a, ra = self._take(rng, zero, 8); b, rb = self._take(rng, one, 8)
                vals = a + b
                rng.shuffle(vals)
                all_indices.extend(vals)
                if ra or rb:
                    fallbacks.append({"subject_id": subject, "kind": "replacement_for_balanced_class"})
            else:
                vals, replaced = self._take(rng, pool, M_TRIALS)
                all_indices.extend(vals)
                fallbacks.append({"subject_id": subject, "kind": "class_balance_unavailable", "replacement": replaced,
                                  "n_class0": int(len(zero)), "n_class1": int(len(one))})
        if len(all_indices) != k * M_TRIALS:
            raise RuntimeError("invalid subject-balanced batch size")
        return SubjectBatch(chosen, all_indices, fallbacks)

    def manifest(self, epochs: int, steps_per_epoch: int) -> list[list[SubjectBatch]]:
        return [[self.one(epoch, step) for step in range(steps_per_epoch)] for epoch in range(1, epochs + 1)]
