from __future__ import annotations

import numpy as np

from .families import TPSFamily


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.maximum(p, 1e-12))).sum(1)


def _bernstein_signature(curve: np.ndarray) -> np.ndarray:
    y = np.asarray(curve, dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    basis = np.column_stack(((1-x)**3, 3*x*(1-x)**2, 3*x*x*(1-x), x**3))
    return np.linalg.lstsq(basis, y, rcond=None)[0]


def context_features(probabilities: np.ndarray) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=float)
    if len(p) == 0:
        raise ValueError("empty context")
    entropy = _entropy(p)
    confidence = p.max(1)
    order = np.sort(p, axis=1)
    margin = order[:, -1] - order[:, -2]
    pred = p.argmax(1)
    result: dict[str, float] = {"context_sample_count": float(len(p))}
    for name, values in (("entropy", entropy), ("maxprob", confidence), ("margin", margin)):
        for label, q in (("q10", .10), ("q50", .50), ("q90", .90)):
            result[f"{name}_{label}"] = float(np.quantile(values, q))
    for cls, value in enumerate(np.bincount(pred, minlength=p.shape[1]) / len(p)):
        result[f"predicted_class_proportion_{cls}"] = float(value)
    result["switch_rate"] = float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0
    half = max(1, len(p) // 2)
    result["confidence_drift"] = float(confidence[half:].mean() - confidence[:half].mean())
    result["entropy_drift"] = float(entropy[half:].mean() - entropy[:half].mean())

    family = TPSFamily()
    full_curve = family.context_sizes(p)[:-1]
    full_coef = _bernstein_signature(full_curve)
    prefix_coef = []
    for fraction in (.25, .50, .75, 1.0):
        n = max(1, int(np.ceil(len(p) * fraction)))
        prefix_coef.append(_bernstein_signature(family.context_sizes(p[:n])[:-1]))
    blocks = [_bernstein_signature(family.context_sizes(block)[:-1]) for block in np.array_split(p, 3) if len(block)]
    prefix_mad = np.median(np.abs(np.vstack(prefix_coef) - np.median(prefix_coef, axis=0)), axis=0)
    block_mad = np.median(np.abs(np.vstack(blocks) - np.median(blocks, axis=0)), axis=0)
    for i in range(4):
        result[f"signature_level_b{i}"] = float(full_coef[i])
        result[f"signature_prefix_mad_b{i}"] = float(prefix_mad[i])
        result[f"signature_block_mad_b{i}"] = float(block_mad[i])
    return result


SIGNATURE_COLUMNS = [
    *(f"signature_level_b{i}" for i in range(4)),
    *(f"signature_prefix_mad_b{i}" for i in range(4)),
    *(f"signature_block_mad_b{i}" for i in range(4)),
]
