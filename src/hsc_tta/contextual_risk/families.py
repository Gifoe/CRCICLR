from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = np.linspace(0.50, 0.99, 20)


def _validate(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all():
        raise ValueError("probabilities must be a finite n-by-C matrix")
    if np.any(p < 0) or not np.allclose(p.sum(1), 1.0, atol=1e-6):
        raise ValueError("invalid probabilities")
    return p


def deterministic_order(probabilities: np.ndarray) -> np.ndarray:
    p = _validate(probabilities)
    classes = np.arange(p.shape[1])
    return np.vstack([np.lexsort((classes, -row)) for row in p])


def monotone_union(sets: np.ndarray) -> tuple[np.ndarray, int]:
    repaired = np.asarray(sets, dtype=bool).copy()
    count = 0
    for j in range(1, repaired.shape[1]):
        added = repaired[:, j - 1] & ~repaired[:, j]
        count += int(added.sum())
        repaired[:, j] |= repaired[:, j - 1]
    return repaired, count


class PredictionSetFamily:
    name = "abstract"

    def build_sets(self, probabilities: np.ndarray) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def context_sizes(self, probabilities: np.ndarray) -> np.ndarray:
        sets, _ = self.build_sets(probabilities)
        return sets.sum(2).mean(0)

    def future_curve(self, probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        sets, repairs = self.build_sets(probabilities)
        y = np.asarray(labels, dtype=int)
        risk = 1.0 - sets[np.arange(len(y)), :, y].mean(0)
        sizes = sets.sum(2).mean(0)
        return risk, sizes, repairs


class TPSFamily(PredictionSetFamily):
    name = "TPS"

    def build_sets(self, probabilities: np.ndarray) -> tuple[np.ndarray, int]:
        p = _validate(probabilities)
        raw = np.stack([p >= (1.0 - value) for value in GRID], axis=1)
        argmax = deterministic_order(p)[:, 0]
        raw[np.arange(len(p))[:, None], np.arange(len(GRID))[None, :], argmax[:, None]] = True
        raw = np.concatenate([raw, np.ones((len(p), 1, p.shape[1]), bool)], axis=1)
        return monotone_union(raw)


class APSFamily(PredictionSetFamily):
    name = "APS"

    def build_sets(self, probabilities: np.ndarray) -> tuple[np.ndarray, int]:
        p = _validate(probabilities)
        order = deterministic_order(p)
        raw = np.zeros((len(p), len(GRID) + 1, p.shape[1]), bool)
        for i, row_order in enumerate(order):
            cumulative = np.cumsum(p[i, row_order])
            for j, q in enumerate(GRID):
                rank = min(int(np.searchsorted(cumulative, q, side="left")), p.shape[1] - 1)
                raw[i, j, row_order[: rank + 1]] = True
        raw[:, -1, :] = True
        return monotone_union(raw)


@dataclass
class RAPSFamily(PredictionSetFamily):
    k_reg: int = 1
    lambda_reg: float = 0.01
    name: str = "RAPS"

    def __post_init__(self) -> None:
        if self.k_reg not in {1, 2, 3} or self.lambda_reg not in {0.01, 0.05, 0.10}:
            raise ValueError("RAPS parameters are outside the frozen grid")

    def build_sets(self, probabilities: np.ndarray) -> tuple[np.ndarray, int]:
        p = _validate(probabilities)
        order = deterministic_order(p)
        raw = np.zeros((len(p), len(GRID) + 1, p.shape[1]), bool)
        for i, row_order in enumerate(order):
            ranks = np.arange(1, p.shape[1] + 1)
            penalized = np.cumsum(p[i, row_order]) + self.lambda_reg * np.maximum(ranks - self.k_reg, 0)
            for j, q in enumerate(GRID):
                rank = min(int(np.searchsorted(penalized, q, side="left")), p.shape[1] - 1)
                raw[i, j, row_order[: rank + 1]] = True
        raw[:, -1, :] = True
        return monotone_union(raw)


def critical_index(risk: np.ndarray, alpha: float) -> int:
    legal = np.flatnonzero(np.asarray(risk) <= alpha)
    return int(legal[0]) if len(legal) else 20
