"""Frozen, legal multi-electrode operator library for Stage-0 v2.

The library is deliberately deterministic: all pools, graph neighbours, and
weights are defined from channel names/coordinates before any model is run.
Every view is composed with the same CAR64 source map, so ``B=A@C@Psi`` is
auditable from the catalog without inspecting performance.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from .operators import OperatorView


def _selector(indices: list[int], size: int) -> np.ndarray:
    out = np.zeros((len(indices), size), float)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def _bipolar(pairs: list[tuple[int, int]], size: int) -> np.ndarray:
    out = np.zeros((len(pairs), size), float)
    for row, (a, b) in enumerate(pairs):
        out[row, a], out[row, b] = 1.0, -1.0
    return out


def _subset_car(indices: list[int], pool: list[int], size: int) -> np.ndarray:
    out = np.zeros((len(indices), size), float)
    reference = np.zeros(size, float)
    reference[pool] = 1.0 / len(pool)
    for row, idx in enumerate(indices):
        out[row, idx] = 1.0
        out[row] -= reference
    return out


def _graph_operator(indices: list[int], coordinates: np.ndarray, size: int, weighted: bool = False) -> np.ndarray:
    out = np.zeros((len(indices), size), float)
    for row, idx in enumerate(indices):
        distances = np.linalg.norm(coordinates - coordinates[idx], axis=1)
        order = [j for j in np.argsort(distances) if j != idx][:4]
        if weighted:
            weights = np.asarray([0.5, 0.3, 0.15, 0.05][:len(order)], float)
            weights /= weights.sum()
        else:
            weights = np.full(len(order), 1.0 / len(order))
        out[row, idx] = 1.0
        out[row, order] -= weights
    return out


def _view(identifier: str, family: str, split: str, a: np.ndarray, source_b: np.ndarray,
          source_coefficients: np.ndarray, definitions: Iterable[str], reference: str) -> OperatorView:
    a = np.asarray(a, float)
    return OperatorView(identifier, family, "eegmmidb_car64", a, a @ source_b,
                        a @ source_coefficients, tuple(definitions), reference, split)


def generate_eegmmidb_v2_operators(
    channel_names: list[str], source_b: np.ndarray, coordinates: np.ndarray,
    source_coefficients: np.ndarray,
) -> list[OperatorView]:
    """Return the fixed v2 train/validation/test operator protocol.

    ``polarity`` views are diagnostic-only and are never included in Gate A/B
    primary aggregates. All other held-out views contain genuine multi-electrode
    references or topologies, not mere sign flips.
    """
    n = len(channel_names)
    if source_b.shape[0] != n or source_coefficients.shape != (n, n):
        raise ValueError("source B/reference/channel mismatch")
    lookup = {name: i for i, name in enumerate(channel_names)}

    def ids(names: Iterable[str]) -> list[int]:
        names = list(names)
        missing = sorted(set(names) - set(lookup))
        if missing:
            raise ValueError(f"operator electrodes missing: {missing}")
        return [lookup[name] for name in names]

    # Every list is frozen by name, and the train/validation/test sets are
    # topologically disjoint. Pools intentionally contain multiple electrodes.
    anterior = ids(["FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8"])
    central = ids(["FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6"])
    posterior = ids(["CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"])
    left = ids(["F7", "F5", "F3", "F1", "FT7", "T7", "TP7", "P7", "P5", "P3"])
    right = ids(["F2", "F4", "F6", "F8", "FT8", "T8", "TP8", "P8", "P6", "P4"])
    sparse = ids(["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"])
    train_active = ids(["F3", "C3", "P3", "O1", "F4", "C4", "P4", "O2", "F7", "T7", "P7", "F8", "T8", "P8"])
    validation_active = ids(["FC3", "CP3", "PO3", "FC4", "CP4", "PO4", "C5", "P5", "C6", "P6", "AF3", "AF4"])
    test_active = ids(["F1", "F2", "C1", "C2", "P1", "P2", "PO7", "PO8", "T9", "T10", "FC1", "FC2"])

    views: list[OperatorView] = []
    def add(identifier: str, family: str, split: str, a: np.ndarray, definitions: Iterable[str], reference: str):
        views.append(_view(identifier, family, split, a, source_b, source_coefficients, definitions, reference))

    # Matched source is CAR64 itself, frozen before training.
    views.append(OperatorView("source_car64", "source", "raw_eegmmidb", np.eye(n), source_b,
                              source_coefficients, tuple(channel_names), "CAR64 source observation", "matched"))
    for split, active, pool, tag in (("train", train_active, anterior, "anterior"), ("validation", validation_active, central, "central"), ("test", test_active, posterior, "posterior")):
        add(f"subset_car_{split}_{tag}", "subset_car", split,
            _subset_car(active, pool, n), [f"{channel_names[i]}-mean({tag}_pool)" for i in active], f"fixed {tag} multi-electrode reference pool")
        add(f"local_average_{split}", "local_average", split,
            _graph_operator(active, coordinates, n), [f"{channel_names[i]}-mean(nearest4)" for i in active], "fixed four-neighbour geometric average")
        add(f"laplacian_{split}", "laplacian", split,
            _graph_operator(active, coordinates, n, weighted=True), [f"{channel_names[i]}-weighted(nearest4)" for i in active], "fixed weighted graph Laplacian")
        add(f"weighted_reference_{split}", "weighted_reference", split,
            _subset_car(active, pool[: min(6, len(pool))], n), [f"{channel_names[i]}-weighted_pool" for i in active], "fixed six-electrode weighted reference")

    add("sparse_subset_train", "sparse_subset", "train", _selector(train_active[:8], n), [channel_names[i] for i in train_active[:8]], "fixed sparse electrode subset")
    add("sparse_subset_validation", "sparse_subset", "validation", _selector(validation_active[:8], n), [channel_names[i] for i in validation_active[:8]], "fixed sparse electrode subset")
    add("sparse_subset_test", "sparse_subset", "test", _selector(sparse, n), [channel_names[i] for i in sparse], "held-out sparse electrode subset")

    train_pairs = [("F3", "C3"), ("C3", "P3"), ("F4", "C4"), ("C4", "P4")]
    validation_pairs = [("FC3", "CP3"), ("CP3", "PO3"), ("FC4", "CP4"), ("CP4", "PO4")]
    test_pairs = [("F1", "P1"), ("C1", "O1"), ("F2", "P2"), ("C2", "O2"), ("T9", "P7"), ("T10", "P8")]
    for identifier, split, pairs in (("bipolar_train", "train", train_pairs), ("bipolar_validation", "validation", validation_pairs), ("bipolar_test", "test", test_pairs)):
        numeric = [(lookup[a], lookup[b]) for a, b in pairs]
        add(identifier, "bipolar", split, _bipolar(numeric, n), [f"{a}-{b}" for a, b in pairs], "explicit two-electrode derivation")
    # Polarity is retained as a separately labeled diagnostic only.
    test_bipolar = _bipolar([(lookup[a], lookup[b]) for a, b in test_pairs], n)
    add("polarity_diagnostic", "polarity", "test", -test_bipolar, [f"reverse({a}-{b})" for a, b in test_pairs], "sign reversal diagnostic; excluded from Gate A/B")
    return views
