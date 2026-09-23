"""TRAIN-fitted directional axes and frozen-network mixed differences.

No labels or outer data enter axis fitting or pair ranking. A classifier-input
axis can use the frozen canonical oblique P projector; all earlier P axes are
the frozen TRAIN mapping's ordered orthonormal columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class DirectionSet:
    p_axes: np.ndarray
    c_axes: np.ndarray
    p_sd: np.ndarray
    c_sd: np.ndarray
    p_rank: int
    c_rank: int
    projector_mode: str


def _batches(length: int, size: int) -> Iterator[slice]:
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def fit_directions(train_a: np.ndarray, q: np.ndarray, mean: np.ndarray, *,
                   c_limit: int = 16,
                   final_projector: tuple[np.ndarray, np.ndarray] | None = None,
                   device: torch.device | str = "cpu") -> DirectionSet:
    """Fit label-free C PCA and TRAIN coefficient scales.

    ``final_projector=(left,right)`` uses the exact canonical oblique P. At
    final, right rows are physical P axes; their coefficient SD is adjusted
    by each row norm after axis normalization.
    """
    a = np.asarray(train_a, np.float32)
    q = np.asarray(q, np.float32)
    mu = np.asarray(mean, np.float32)
    if a.ndim != 2 or len(a) < 2 or q.ndim != 2 or q.shape[0] != a.shape[1] or mu.shape != (a.shape[1],):
        raise ValueError("invalid TRAIN activation/projector dimensions")
    if c_limit not in (16, 32) or not np.isfinite(a).all() or not np.isfinite(q).all() or not np.isfinite(mu).all():
        raise ValueError("invalid fixed PCA limit or nonfinite TRAIN inputs")
    centered = a - mu
    if final_projector is None:
        if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-4):
            raise ValueError("intermediate P columns are not orthonormal")
        p_axes = q
        coefficients = centered @ q
        p = coefficients @ q.T
        mode = "TRAIN_ONLY_ORTHOGONAL_PATHWAY"
    else:
        left, right = (np.asarray(item, np.float32) for item in final_projector)
        if left.shape != (a.shape[1], q.shape[1]) or right.shape != (q.shape[1], a.shape[1]):
            raise ValueError("invalid frozen canonical final projector")
        row_norm = np.linalg.norm(right.astype(np.float64), axis=1)
        if np.any(row_norm < 1e-8):
            raise ValueError("degenerate frozen final P raw axis")
        p_axes = (right / row_norm[:, None]).T.astype(np.float32)
        coefficients = (centered @ left) * row_norm[None, :]
        p = (centered @ left) @ right
        mode = "FROZEN_CANONICAL_OBLIQUE"
    p_sd = coefficients.std(axis=0, ddof=0).astype(np.float32)
    if np.any(p_sd < 1e-8):
        raise ValueError("degenerate TRAIN P coefficient SD")
    c = np.ascontiguousarray(centered - p, dtype=np.float32)
    # Feature space may have tens of thousands of coordinates. The TRAIN
    # sample Gram matrix preserves exact nonzero singular directions while
    # avoiding a dense feature-by-feature covariance matrix.
    target = torch.device(device)
    # The locked 1e-6 singular-value rank threshold is below the spurious
    # singular values that float32 Gram eigendecomposition can introduce.
    # Compute this fit in float64 so rank is not inflated by roundoff.
    c_tensor = torch.as_tensor(c, device=target, dtype=torch.float64)
    c_zero = c_tensor - c_tensor.mean(0, keepdim=True)
    gram = c_zero @ c_zero.T
    eigval, eigvec = torch.linalg.eigh(gram)
    eigval = eigval.flip(0).clamp_min(0)
    eigvec = eigvec.flip(1)
    singular = torch.sqrt(eigval)
    if not len(singular) or singular[0].item() <= 0:
        raise ValueError("zero-rank TRAIN complement")
    threshold = max(1e-6 * float(singular[0].item()), 1e-8)
    rank = int((singular > threshold).sum().item())
    count = min(c_limit, rank)
    if count < 1:
        raise ValueError("TRAIN complement has no usable direction")
    components = (c_zero.T @ eigvec[:, :count]) / singular[:count][None, :]
    # SVD sign is arbitrary. Canonicalize each component by making its
    # largest-magnitude coordinate positive, so frozen ranks are portable.
    dominant = torch.argmax(components.abs(), dim=0)
    signs = torch.sign(components[dominant, torch.arange(count, device=target)])
    components = components * signs[None, :]
    c_axes = components.cpu().numpy().astype(np.float32)
    c_coefficients = c_zero @ components
    c_sd = c_coefficients.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    if np.any(c_sd < 1e-8):
        raise ValueError("degenerate TRAIN C coefficient SD")
    return DirectionSet(np.ascontiguousarray(p_axes), np.ascontiguousarray(c_axes),
                        p_sd, c_sd, int(p_axes.shape[1]), int(count), mode)


def frozen_top_pairs(train_means: np.ndarray, count: int = 10) -> tuple[tuple[int, int], ...]:
    values = np.asarray(train_means, np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("finite nonempty TRAIN interaction matrix required")
    order = sorted(np.ndindex(values.shape), key=lambda ij: (-values[ij], ij[0], ij[1]))
    return tuple((int(i), int(j)) for i, j in order[: min(count, len(order))])


def mixed_difference(runner: Any, stage: str, shape: tuple[int, ...],
                     a: np.ndarray, labels: np.ndarray, native_logits: np.ndarray,
                     directions: DirectionSet, pair: tuple[int, int], *, epsilon: float = 0.5,
                     batch_size: int = 64) -> dict[str, np.ndarray]:
    """Evaluate symmetric 4-point P×C intervention on frozen native logits."""
    if epsilon not in (0.25, 0.5, 1.0):
        raise ValueError("epsilon outside frozen primary/sensitivity grid")
    x = np.asarray(a, np.float32)
    y = np.asarray(labels, np.int64)
    native = np.asarray(native_logits, np.float32)
    if x.ndim != 2 or x.shape[1] != int(np.prod(shape)) or len(x) != len(y) or native.shape[0] != len(x):
        raise ValueError("unaligned activation, shape, labels or native logits")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(native).all():
        raise ValueError("empty or nonfinite interaction inputs")
    p_index, c_index = pair
    if not (0 <= p_index < directions.p_rank and 0 <= c_index < directions.c_rank):
        raise ValueError("direction pair index outside frozen axes")
    if np.any(y < 0) or np.any(y >= native.shape[1]) or native.shape[1] < 2:
        raise ValueError("invalid class label for fixed margins")
    if runner.m.training or runner.head.training:
        raise RuntimeError("native model/head must remain in evaluation mode")
    dp = epsilon * directions.p_sd[p_index] * directions.p_axes[:, p_index]
    dc = epsilon * directions.c_sd[c_index] * directions.c_axes[:, c_index]
    top = native.argmax(1)
    rival = native.copy()
    rival[np.arange(len(rival)), top] = -np.inf
    second = rival.argmax(1)
    output: dict[str, list[np.ndarray]] = {key: [] for key in
                                           ("centered_logit", "norm", "true_margin", "top1_margin")}
    for part in _batches(len(x), batch_size):
        xx = x[part]
        variants = np.concatenate((xx + dp + dc, xx + dp - dc,
                                   xx - dp + dc, xx - dp - dc)).astype(np.float32)
        with torch.inference_mode():
            _, logits = runner.from_stage(stage, runner.tensor(variants.reshape((len(variants), *shape))))
        z = logits.float().cpu().numpy().reshape((4, len(xx), -1))
        raw = (z[0] - z[1] - z[2] + z[3]) * 0.25
        centered = raw - raw.mean(axis=1, keepdims=True)
        yy = y[part]
        margins = z[:, np.arange(len(xx)), yy] - np.max(
            np.where(np.eye(z.shape[2], dtype=bool)[yy][None, :, :], -np.inf, z), axis=2)
        true_margin = (margins[0] - margins[1] - margins[2] + margins[3]) * 0.25
        top_margin = z[:, np.arange(len(xx)), top[part]] - z[:, np.arange(len(xx)), second[part]]
        top_margin = (top_margin[0] - top_margin[1] - top_margin[2] + top_margin[3]) * 0.25
        output["centered_logit"].append(centered)
        output["norm"].append(np.linalg.norm(centered, axis=1))
        output["true_margin"].append(true_margin)
        output["top1_margin"].append(top_margin)
    return {key: np.concatenate(values).astype(np.float32) for key, values in output.items()}


def mixed_differences_grouped(runner: Any, stage: str, shape: tuple[int, ...],
                              a: np.ndarray, labels: np.ndarray, native_logits: np.ndarray,
                              directions: DirectionSet, pairs: tuple[tuple[int, int], ...], *,
                              epsilon: float = 0.5, trial_batch: int = 16) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Evaluate several exact four-point pairs in one native forward call.

    Each pair retains the same float32 arithmetic order as ``mixed_difference``.
    This changes scheduling only; axes, trials, perturbations, and metrics do not
    change. The caller must still benchmark the actual model before adoption.
    """
    if epsilon not in (0.25, 0.5, 1.0) or not pairs or trial_batch < 1:
        raise ValueError("invalid fixed grid, empty pairs, or trial batch")
    x = np.asarray(a, np.float32)
    y = np.asarray(labels, np.int64)
    native = np.asarray(native_logits, np.float32)
    if x.ndim != 2 or x.shape[1] != int(np.prod(shape)) or len(x) != len(y) or native.shape[0] != len(x):
        raise ValueError("unaligned activation, shape, labels or native logits")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(native).all():
        raise ValueError("empty or nonfinite interaction inputs")
    if np.any(y < 0) or np.any(y >= native.shape[1]) or native.shape[1] < 2:
        raise ValueError("invalid class labels")
    if runner.m.training or runner.head.training:
        raise RuntimeError("native model/head must remain in evaluation mode")
    for p, c in pairs:
        if not (0 <= p < directions.p_rank and 0 <= c < directions.c_rank):
            raise ValueError("direction pair index outside frozen axes")
    top = native.argmax(1)
    rival = native.copy()
    rival[np.arange(len(rival)), top] = -np.inf
    second = rival.argmax(1)
    names = ("centered_logit", "norm", "true_margin", "top1_margin")
    collected = {pair: {name: [] for name in names} for pair in pairs}
    for part in _batches(len(x), trial_batch):
        xx = x[part]
        blocks = []
        for p, c in pairs:
            dp = epsilon * directions.p_sd[p] * directions.p_axes[:, p]
            dc = epsilon * directions.c_sd[c] * directions.c_axes[:, c]
            blocks.extend((xx + dp + dc, xx + dp - dc, xx - dp + dc, xx - dp - dc))
        variants = np.concatenate(blocks).astype(np.float32)
        with torch.inference_mode():
            _, logits = runner.from_stage(stage, runner.tensor(variants.reshape((len(variants), *shape))))
        zall = logits.float().cpu().numpy().reshape((len(pairs), 4, len(xx), -1))
        for pair, z in zip(pairs, zall):
            raw = (z[0] - z[1] - z[2] + z[3]) * 0.25
            centered = raw - raw.mean(axis=1, keepdims=True)
            yy = y[part]
            margins = z[:, np.arange(len(xx)), yy] - np.max(
                np.where(np.eye(z.shape[2], dtype=bool)[yy][None, :, :], -np.inf, z), axis=2)
            true_margin = (margins[0] - margins[1] - margins[2] + margins[3]) * 0.25
            top_margin = z[:, np.arange(len(xx)), top[part]] - z[:, np.arange(len(xx)), second[part]]
            top_margin = (top_margin[0] - top_margin[1] - top_margin[2] + top_margin[3]) * 0.25
            collected[pair]["centered_logit"].append(centered)
            collected[pair]["norm"].append(np.linalg.norm(centered, axis=1))
            collected[pair]["true_margin"].append(true_margin)
            collected[pair]["top1_margin"].append(top_margin)
    return {pair: {name: np.concatenate(values).astype(np.float32) for name, values in record.items()}
            for pair, record in collected.items()}
