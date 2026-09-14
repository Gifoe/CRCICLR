"""Deterministic, disjoint source-session blocks for optimizer-aligned harm audit."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


BLOCK_NAMES = ("B1", "B2", "B3", "B4", "B_future")


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "little")


@dataclass(frozen=True)
class SubjectBlocks:
    subject: str
    m_per_class: int
    blocks: dict[str, np.ndarray]


def make_subject_blocks(subject: str, class_pools: dict[int, np.ndarray], *, dataset: str,
                        fold: int, seed: int = 0) -> SubjectBlocks:
    """Split each class pool into five class-balanced, non-overlapping blocks."""
    minimum = min(len(values) for values in class_pools.values())
    m = min(8, minimum // 5)
    if m < 4:
        raise RuntimeError(
            f"insufficient legal source trials for five blocks: {dataset}/f{fold}/{subject}; "
            f"min_class={minimum}, m_per_class={m}"
        )
    buckets: dict[str, list[int]] = {name: [] for name in BLOCK_NAMES}
    for cls, values in sorted(class_pools.items()):
        rng = np.random.default_rng(stable_seed("optimizer-aligned-guard-blocks", dataset, fold, seed, subject, cls))
        chosen = values[rng.permutation(len(values))][:5 * m]
        for number, name in enumerate(BLOCK_NAMES):
            buckets[name].extend(chosen[number * m:(number + 1) * m].tolist())
    blocks = {name: np.asarray(indices, dtype=np.int64) for name, indices in buckets.items()}
    union = np.concatenate(list(blocks.values()))
    if len(np.unique(union)) != len(union):
        raise RuntimeError(f"overlapping source trials in guard blocks: {dataset}/f{fold}/{subject}")
    expected = len(class_pools) * m
    if any(len(value) != expected for value in blocks.values()):
        raise RuntimeError(f"unbalanced guard blocks: {dataset}/f{fold}/{subject}")
    return SubjectBlocks(str(subject), m, blocks)
