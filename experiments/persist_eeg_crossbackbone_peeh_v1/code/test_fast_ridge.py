import importlib.util
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).with_name("run_crossbackbone_peeh.py")
SPEC = importlib.util.spec_from_file_location("peeh_runner_test", RUNNER)
assert SPEC is not None and SPEC.loader is not None
peeh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(peeh)


def synthetic_spec(rng: np.random.Generator, d: int, rank: int) -> dict:
    basis, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    directions, _ = np.linalg.qr(rng.normal(size=(rank, rank)))
    return {
        "mean": rng.normal(size=d).astype(np.float32),
        "basis": basis.astype(np.float32),
        "scale": rng.uniform(0.4, 2.0, size=rank).astype(np.float32),
        "directions": directions.astype(np.float32),
        "rank": rank,
    }


def explicit_affine_scores(hfit, yfit, heval, classes, spec, dims):
    mu = hfit.mean(0, dtype=np.float64)
    sd = hfit.std(0, dtype=np.float64)
    sd[sd < 1e-6] = 1.0
    X = ((hfit - mu) / sd).astype(np.float32)
    Z = ((heval - mu) / sd).astype(np.float32)
    A = (peeh._erasure_base(spec) / sd[None, :]).astype(np.float32)
    Q = peeh.canonical(hfit, spec).astype(np.float32)
    R = peeh.canonical(heval, spec).astype(np.float32)
    ix = np.asarray(dims, dtype=int)
    if len(ix):
        X = X - Q[:, ix] @ A[ix]
        Z = Z - R[:, ix] @ A[ix]
    Y = np.eye(classes, dtype=np.float64)[yfit]
    ym = Y.mean(0)
    coef = np.linalg.solve(
        (X @ X.T).astype(np.float64) + peeh.RIDGE_ALPHA * np.eye(len(X)),
        Y - ym,
    )
    return (Z @ X.T) @ coef + ym


def test_woodbury_matches_explicit_affine_kernel():
    rng = np.random.default_rng(20260915)
    hfit = rng.normal(size=(29, 47)).astype(np.float32)
    heval = rng.normal(size=(17, 47)).astype(np.float32)
    yfit = np.arange(len(hfit)) % 3
    spec = synthetic_spec(rng, 47, 7)
    engine = peeh.FixedStandardizerKernelRidge(hfit, yfit, heval, 3, spec)
    for dims in ((), (0,), (1, 4), (0, 2, 3, 6)):
        got = engine.scores(dims)
        expected = explicit_affine_scores(hfit, yfit, heval, 3, spec, dims)
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-5)
        np.testing.assert_array_equal(got.argmax(1), expected.argmax(1))


def test_fast_path_matches_locked_raw_erasure_reference():
    rng = np.random.default_rng(991)
    hfit = rng.normal(size=(31, 53)).astype(np.float32)
    heval = rng.normal(size=(19, 53)).astype(np.float32)
    yfit = np.arange(len(hfit)) % 4
    spec = synthetic_spec(rng, 53, 8)
    engine = peeh.FixedStandardizerKernelRidge(hfit, yfit, heval, 4, spec)
    for dims in ((), (2,), (1, 5), (0, 3, 4, 7)):
        pred, got = engine.predict(dims)
        ref_pred, expected = peeh.ridge_fixed_direct(
            hfit, yfit, heval, 4, spec, dims, (engine.mu, engine.sd)
        )
        np.testing.assert_allclose(got, expected, rtol=3e-4, atol=3e-5)
        np.testing.assert_array_equal(pred, ref_pred)


def test_primal_path_matches_locked_reference():
    rng = np.random.default_rng(17)
    hfit = rng.normal(size=(61, 13)).astype(np.float32)
    heval = rng.normal(size=(23, 13)).astype(np.float32)
    yfit = np.arange(len(hfit)) % 3
    spec = synthetic_spec(rng, 13, 6)
    engine = peeh.FixedStandardizerKernelRidge(hfit, yfit, heval, 3, spec)
    assert engine.mode == "primal"
    for dims in ((), (0,), (2, 5), (0, 1, 3, 4)):
        pred, got = engine.predict(dims)
        ref_pred, expected = peeh.ridge_fixed_direct(
            hfit, yfit, heval, 3, spec, dims, (engine.mu, engine.sd)
        )
        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(pred, ref_pred)
