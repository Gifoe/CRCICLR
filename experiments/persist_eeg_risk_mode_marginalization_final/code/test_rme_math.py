"""Deterministic pre-outcome checks for PERSIST-RME."""
import numpy as np


def projected_update(g_uniform, g_risk, beta=0.5):
    gu = np.asarray(g_uniform, dtype=float)
    r = np.asarray(g_risk, dtype=float) - gu
    dot = float(r @ gu)
    if dot < 0:
        r = r - dot / (float(gu @ gu) + 1e-12) * gu
    return gu + beta * r


def test_projection_sign():
    g = projected_update([1.0, 0.0], [-1.0, 1.0])
    assert float(g @ np.asarray([1.0, 0.0])) > 0


def test_projection_nonnegative():
    for gu, gr in [([1.0, 2.0], [3.0, 1.0]), ([1.0, 0.0], [-1.0, 1.0]), ([0.0, 0.0], [0.0, 0.0])]:
        out = projected_update(gu, gr)
        assert float(out @ np.asarray(gu)) >= -1e-8


def test_ratio_bounds_and_sum():
    raw = np.array([0.05, 0.4, 1.0, 2.2, 8.0])
    ratios = np.clip(raw, 0.25, 4.0)
    for _ in range(128):
        diff = len(ratios) - ratios.sum()
        free = (ratios > 0.25 + 1e-12) & (ratios < 4.0 - 1e-12)
        if abs(float(diff)) < 1e-12 or not free.any():
            break
        ratios[free] += diff / free.sum()
        ratios = np.clip(ratios, 0.25, 4.0)
    assert abs(float(ratios.sum()) - len(ratios)) < 1e-8
    assert ratios.min() >= 0.25 - 1e-8 and ratios.max() <= 4.0 + 1e-8


def test_rme_aggregation():
    anchor = np.array([[0.2, 0.8]])
    experts = np.array([[[0.3, 0.7]], [[0.1, 0.9]], [[0.2, 0.8]], [[0.4, 0.6]]])
    risk = experts.mean(axis=0)
    out = 0.5 * anchor + 0.5 * risk
    np.testing.assert_allclose(out, np.array([[0.225, 0.775]]))
