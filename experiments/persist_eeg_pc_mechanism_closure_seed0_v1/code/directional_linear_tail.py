"""Exact mixed-difference shortcut for a frozen affine classifier tail.

Only the explicitly audited classifier-input stages (and EEGNet's identical
embedding_64d alias) are eligible. Centered logits and a fixed top-two margin
are affine there, hence their mixed finite difference is mathematically zero.
The true-class margin may still be nonlinear for multiclass tasks because the
largest rival class can change; evaluate that margin for all four variants.
"""
from __future__ import annotations

import numpy as np
import torch


def eligible(runner, stage: str) -> bool:
    return isinstance(runner.head, torch.nn.Linear) and (
        stage == "classifier_input" or
        (runner.name == "EEGNet" and stage == "embedding_64d")
    )


def affine_tail_mixed_difference(runner, stage: str, shape: tuple[int, ...],
                                 a: np.ndarray, labels: np.ndarray,
                                 native_logits: np.ndarray, directions,
                                 pair: tuple[int, int], *, epsilon: float = 0.5):
    if not eligible(runner, stage):
        raise ValueError("stage has no proven affine classifier tail")
    if runner.m.training or runner.head.training:
        raise RuntimeError("frozen network must be in eval mode")
    if epsilon not in (0.25, 0.5, 1.0):
        raise ValueError("epsilon outside locked grid")
    x = np.asarray(a, np.float64)
    y = np.asarray(labels, np.int64)
    native = np.asarray(native_logits, np.float64)
    if (x.ndim != 2 or x.shape[1] != int(np.prod(shape)) or
            native.shape != (len(x), runner.head.out_features) or
            len(y) != len(x) or not len(x) or
            not np.isfinite(x).all() or not np.isfinite(native).all() or
            np.any(y < 0) or np.any(y >= native.shape[1])):
        raise ValueError("invalid frozen classifier-input data")
    p, c = pair
    if not (0 <= p < directions.p_rank and 0 <= c < directions.c_rank):
        raise ValueError("direction pair outside frozen axes")
    w = runner.head.weight.detach().cpu().numpy().astype(np.float64)
    b = (runner.head.bias.detach().cpu().numpy().astype(np.float64)
         if runner.head.bias is not None else np.zeros(w.shape[0], np.float64))
    baseline = x @ w.T + b
    if np.max(np.abs(baseline - native)) >= 1e-5:
        raise RuntimeError("affine shortcut does not reproduce native logits")
    dp = epsilon * directions.p_sd[p] * (directions.p_axes[:, p] @ w.T)
    dc = epsilon * directions.c_sd[c] * (directions.c_axes[:, c] @ w.T)
    z = np.stack((baseline + dp + dc, baseline + dp - dc,
                  baseline - dp + dc, baseline - dp - dc), axis=0)
    own = z[:, np.arange(len(x)), y]
    rivals = z.copy()
    rivals[:, np.arange(len(x)), y] = -np.inf
    margins = own - rivals.max(axis=2)
    true_margin = 0.25 * (margins[0] - margins[1] - margins[2] + margins[3])
    return {
        "centered_logit": np.zeros_like(native, dtype=np.float32),
        "norm": np.zeros(len(x), dtype=np.float32),
        "true_margin": true_margin.astype(np.float32),
        "top1_margin": np.zeros(len(x), dtype=np.float32),
    }
