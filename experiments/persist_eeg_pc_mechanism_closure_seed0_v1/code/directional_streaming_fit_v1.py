"""Memory-bounded exact-form TRAIN PCA and directional matrix reductions."""
from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch


def fit_directions_streamed(train_a: np.ndarray, q: np.ndarray, mean: np.ndarray, *,
                            c_limit: int = 16, final_projector=None,
                            device: torch.device | str = "cpu", chunk_features: int = 8192,
                            direction_set_type=None):
    if chunk_features < 1 or direction_set_type is None:
        raise ValueError("invalid streamed PCA configuration")
    a = np.asarray(train_a, np.float32)
    q = np.asarray(q, np.float32)
    mu = np.asarray(mean, np.float32)
    if (a.ndim != 2 or len(a) < 2 or q.ndim != 2 or q.shape[0] != a.shape[1]
            or mu.shape != (a.shape[1],) or c_limit not in (16, 32)
            or not np.isfinite(a).all() or not np.isfinite(q).all() or not np.isfinite(mu).all()):
        raise ValueError("invalid TRAIN activation/projector dimensions or nonfinite values")
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
    n, d = c.shape
    target = torch.device(device)
    gram = torch.zeros((n, n), dtype=torch.float64, device=target)
    for start in range(0, d, chunk_features):
        stop = min(start + chunk_features, d)
        block = torch.as_tensor(c[:, start:stop], device=target, dtype=torch.float64)
        block -= block.mean(0, keepdim=True)
        gram.addmm_(block, block.T)
        del block
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
    eigvec_keep, singular_keep = eigvec[:, :count], singular[:count]
    best_abs = torch.full((count,), -1.0, dtype=torch.float64, device=target)
    best_index = torch.zeros((count,), dtype=torch.int64, device=target)
    for start in range(0, d, chunk_features):
        stop = min(start + chunk_features, d)
        block = torch.as_tensor(c[:, start:stop], device=target, dtype=torch.float64)
        block -= block.mean(0, keepdim=True)
        comp = (block.T @ eigvec_keep) / singular_keep[None, :]
        maxima, indices = comp.abs().max(dim=0)
        replace = maxima > best_abs
        best_abs[replace] = maxima[replace]
        best_index[replace] = indices[replace] + start
        del block, comp, maxima, indices, replace
    signs = torch.ones_like(best_abs)
    for start in range(0, d, chunk_features):
        stop = min(start + chunk_features, d)
        block = torch.as_tensor(c[:, start:stop], device=target, dtype=torch.float64)
        block -= block.mean(0, keepdim=True)
        comp = (block.T @ eigvec_keep) / singular_keep[None, :]
        in_chunk = (best_index >= start) & (best_index < stop)
        local = best_index[in_chunk] - start
        cols = torch.arange(count, device=target)[in_chunk]
        signs[cols] = torch.sign(comp[local, cols])
        del block, comp, in_chunk, local, cols
    components = np.empty((d, count), np.float32)
    c_coefficients = torch.zeros((n, count), dtype=torch.float64, device=target)
    for start in range(0, d, chunk_features):
        stop = min(start + chunk_features, d)
        block = torch.as_tensor(c[:, start:stop], device=target, dtype=torch.float64)
        block -= block.mean(0, keepdim=True)
        comp = ((block.T @ eigvec_keep) / singular_keep[None, :]) * signs[None, :]
        components[start:stop] = comp.cpu().numpy().astype(np.float32)
        c_coefficients.addmm_(block, comp)
        del block, comp
    c_sd = c_coefficients.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
    if np.any(c_sd < 1e-8):
        raise ValueError("degenerate TRAIN C coefficient SD")
    c_axes = np.ascontiguousarray(components)
    del gram, eigval, eigvec, eigvec_keep, singular, singular_keep, c_coefficients
    gc.collect()
    return direction_set_type(np.ascontiguousarray(p_axes), c_axes, p_sd, c_sd,
                              int(p_axes.shape[1]), int(count), mode)


class DeferredEffects(dict):
    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def __missing__(self, key):
        value = self._factory(key)
        self[key] = value
        return value


def streamed_matrix_and_pairs(base: Any, runner, stage, shape, selection, view, axes,
                              *, pair_subset=None, epsilon=0.5, spatial_base=None):
    rows = base.unique_recipient_rows(selection)
    activations = selection.activations[rows]
    labels, subjects, sessions, native = base.selected_identity(selection, view, rows)
    pairs = tuple(np.ndindex((axes.p_rank, axes.c_rank))) if pair_subset is None else tuple(pair_subset)
    means = np.full((axes.p_rank, axes.c_rank), np.nan, np.float64)

    def effect_for(pair):
        if spatial_base is None:
            return base.mixed_difference(runner, stage, shape, activations, labels,
                                         native, axes, pair, epsilon=epsilon)
        return base.factorized_mixed_difference(runner, stage, shape, activations, labels,
                                                native, axes, pair, epsilon=epsilon,
                                                spatial_base=spatial_base)

    for pair in pairs:
        effect = effect_for(pair)
        means[pair] = base.subject_equal_mean(effect["norm"], subjects)
        del effect
    gc.collect()
    return means, DeferredEffects(effect_for), (labels, subjects, sessions, native)
