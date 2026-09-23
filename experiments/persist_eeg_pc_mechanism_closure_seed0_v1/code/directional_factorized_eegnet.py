"""EEGNet temporal_bn four-point inference with an exactly affine conv prefix.

Only the computational schedule changes: for each fixed P/C perturbation we
apply the linear spatial convolution to base and axes separately, then run the
same frozen nonlinear suffix and four-point statistic. Real-model equivalence
must be established before scientific use.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from directional_core_v3 import DirectionSet, _batches


def _spatial_linear(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model.spatial(x)
    if model.spatial.bias is not None:
        out = out - model.spatial.bias.reshape(1, -1, 1, 1)
    return out


def precompute_spatial_base(runner: Any, shape: tuple[int, ...], a: np.ndarray,
                            *, batch_size: int = 64) -> torch.Tensor:
    """Cache frozen spatial responses for one recipient set, reusable by all P/C pairs."""
    if runner.name != "EEGNet" or runner.m.training or runner.head.training or batch_size < 1:
        raise ValueError("only frozen EEGNet evaluation is eligible")
    x = np.asarray(a, np.float32)
    if x.ndim != 2 or x.shape[1] != int(np.prod(shape)) or not len(x) or not np.isfinite(x).all():
        raise ValueError("invalid fixed recipient activations")
    values = []
    with torch.inference_mode():
        for part in _batches(len(x), batch_size):
            xx = x[part]
            values.append(_spatial_linear(runner.m, runner.tensor(xx.reshape((len(xx), *shape)))))
        return torch.cat(values, dim=0)


def factorized_mixed_difference(runner: Any, stage: str, shape: tuple[int, ...],
                                a: np.ndarray, labels: np.ndarray, native_logits: np.ndarray,
                                directions: DirectionSet, pair: tuple[int, int], *,
                                epsilon: float = 0.5, batch_size: int = 64,
                                spatial_base: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    if runner.name != "EEGNet" or stage != "temporal_bn":
        raise ValueError("affine-prefix method applies only to EEGNet temporal_bn")
    if epsilon not in (0.25, 0.5, 1.0) or batch_size < 1:
        raise ValueError("outside frozen epsilon grid or invalid batch")
    x = np.asarray(a, np.float32)
    y = np.asarray(labels, np.int64)
    native = np.asarray(native_logits, np.float32)
    if x.ndim != 2 or x.shape[1] != int(np.prod(shape)) or len(x) != len(y) or native.shape[0] != len(x):
        raise ValueError("unaligned activation, shape, labels or native logits")
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(native).all():
        raise ValueError("empty or nonfinite interaction inputs")
    if np.any(y < 0) or np.any(y >= native.shape[1]) or native.shape[1] < 2:
        raise ValueError("invalid labels")
    p, c = pair
    if not (0 <= p < directions.p_rank and 0 <= c < directions.c_rank):
        raise ValueError("direction pair outside frozen axes")
    if runner.m.training or runner.head.training:
        raise RuntimeError("native model/head must remain in evaluation mode")
    if spatial_base is not None and (len(spatial_base) != len(x)
                                     or spatial_base.device.type != runner.device.type):
        raise ValueError("cached spatial responses do not match recipient/device")
    dp = (epsilon * directions.p_sd[p] * directions.p_axes[:, p]).astype(np.float32)
    dc = (epsilon * directions.c_sd[c] * directions.c_axes[:, c]).astype(np.float32)
    m = runner.m
    with torch.inference_mode():
        pd = _spatial_linear(m, runner.tensor(dp.reshape((1, *shape))))
        cd = _spatial_linear(m, runner.tensor(dc.reshape((1, *shape))))
    bias = m.spatial.bias.reshape(1, -1, 1, 1) if m.spatial.bias is not None else 0
    top = native.argmax(1)
    rival = native.copy()
    rival[np.arange(len(rival)), top] = -np.inf
    second = rival.argmax(1)
    output: dict[str, list[np.ndarray]] = {key: [] for key in
                                           ("centered_logit", "norm", "true_margin", "top1_margin")}
    for part in _batches(len(x), batch_size):
        xx = x[part]
        with torch.inference_mode():
            base = (_spatial_linear(m, runner.tensor(xx.reshape((len(xx), *shape))))
                    if spatial_base is None else spatial_base[part])
            spatial = torch.cat((base + pd + cd, base + pd - cd,
                                 base - pd + cd, base - pd - cd), dim=0) + bias
            b = m.drop1(m.pool1(F.elu(m.bn2(spatial))))
            v = m.drop2(m.pool2(F.elu(m.bn3(m.point(m.depth(b))))))
            h = m.embedding(v.flatten(1))
            logits = runner.native_head(h)
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
