from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hsc_tta.prediction_sets import prediction_sets


LOG2 = float(np.log(2.0))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    left, right = np.asarray(p, float), np.asarray(q, float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("probability matrices must have equal [sample,class] shape")
    if np.any(left < 0) or np.any(right < 0) or not np.allclose(left.sum(1), 1, atol=1e-6) or not np.allclose(right.sum(1), 1, atol=1e-6):
        raise ValueError("invalid probabilities")
    middle = (left + right) / 2
    kl_left = np.sum(left * (np.log(np.maximum(left, 1e-12)) - np.log(np.maximum(middle, 1e-12))), axis=1)
    kl_right = np.sum(right * (np.log(np.maximum(right, 1e-12)) - np.log(np.maximum(middle, 1e-12))), axis=1)
    return np.clip((kl_left + kl_right) / 2, 0, LOG2)


def normalized_set_efficiency(probabilities: np.ndarray, lambdas: np.ndarray) -> float:
    p = np.asarray(probabilities, float); grid = np.asarray(lambdas, float)
    if len(grid) < 2 or not np.isclose(grid[-1], 1.0):
        raise ValueError("lambda grid must end in a sentinel")
    sizes = prediction_sets(p, grid[:-1]).sum(2) / p.shape[1]
    return float(sizes.mean())


def effective_class_number(probabilities: np.ndarray) -> float:
    mass = np.asarray(probabilities, float).mean(0)
    entropy = -float(np.sum(mass * np.log(np.maximum(mass, 1e-12))))
    return float(np.exp(entropy))


def temporal_blocks(n_samples: int, requested: int = 3, minimum: int = 5) -> list[np.ndarray]:
    if n_samples < 2 * minimum:
        raise ValueError("Probe is too short for temporal stability")
    blocks = min(requested, n_samples // minimum)
    if blocks < 2:
        raise ValueError("at least two Probe blocks are required")
    return [x for x in np.array_split(np.arange(n_samples), blocks) if len(x)]


@dataclass(frozen=True)
class ProbeDiagnostics:
    g_set: float
    g_set_relative: float
    g_aug: float
    s_time: float
    positive_probe_block_fraction: float
    worst_block_gain: float
    temporal_mad: float
    d_src: float
    action_available: bool
    r_class: float
    normalized_update_magnitude: float


def compute_probe_diagnostics(source: np.ndarray, adapted: np.ndarray, augmented_source: list[np.ndarray],
                              augmented_adapted: list[np.ndarray], lambdas: np.ndarray, *,
                              action_available: bool, normalized_update_magnitude: float) -> ProbeDiagnostics:
    source = np.asarray(source, float); adapted = np.asarray(adapted, float)
    if source.shape != adapted.shape or len(augmented_source) != len(augmented_adapted) or not augmented_source:
        raise ValueError("unaligned Probe predictions")
    source_eff = normalized_set_efficiency(source, lambdas); adapted_eff = normalized_set_efficiency(adapted, lambdas)
    g_set = source_eff - adapted_eff
    source_consistency = 1 - np.mean([jensen_shannon(source, x).mean() / LOG2 for x in augmented_source])
    adapted_consistency = 1 - np.mean([jensen_shannon(adapted, x).mean() / LOG2 for x in augmented_adapted])
    blocks = temporal_blocks(len(source)); block_gains = np.asarray([
        normalized_set_efficiency(source[index], lambdas) - normalized_set_efficiency(adapted[index], lambdas)
        for index in blocks], float)
    median = float(np.median(block_gains)); mad = float(np.median(np.abs(block_gains - median)))
    source_neff = effective_class_number(source); adapted_neff = effective_class_number(adapted)
    return ProbeDiagnostics(g_set=g_set, g_set_relative=g_set / max(source_eff, 1e-12),
        g_aug=float(adapted_consistency - source_consistency), s_time=1 / (1 + mad),
        positive_probe_block_fraction=float(np.mean(block_gains > 0)), worst_block_gain=float(block_gains.min()),
        temporal_mad=mad, d_src=float(jensen_shannon(adapted, source).mean() / LOG2),
        action_available=bool(action_available), r_class=float(adapted_neff / max(source_neff, 1e-12)),
        normalized_update_magnitude=float(normalized_update_magnitude))
