"""Fast shape/integrity tests; no EEG file access."""
import numpy as np

import run as c
import outer as o


def test_rank_one_and_eight():
    rng = np.random.default_rng(3)
    source = [str(i) for i in range(12)]
    target = "20"
    desc = {s: {f: rng.normal(size=20) for f in c.FAMILIES} for s in source + [target]}
    oracle = {s: rng.normal(size=65) for s in source}
    ep = {"source": source, "target": target, "pop": rng.normal(size=65), "oracle": oracle,
          "desc": desc, "z2": rng.normal(size=(20, 64)), "y2": np.tile([0, 1], 10)}
    assert len(list(c.predict_grid_episode(ep, c.FAMILIES))) == 64
    for rank in (1, 8):
        theta, delta, _, _ = c.predict(ep, "CTX_PRED", rank, 1.0)
        assert theta.shape == delta.shape == (65,)
        assert np.isfinite(theta).all()
        mat = rng.normal(size=(12, 20)); vec = rng.normal(size=20)
        pred, corr, _, _ = o.one_predict(mat, vec, rng.normal(size=(12, 65)), ep["pop"], rank, 1.0)
        assert pred.shape == corr.shape == (65,)


def test_descriptor_label_free_shapes():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(8, 64)); theta = rng.normal(size=65)
    q, _ = np.linalg.qr(rng.normal(size=(64, 4)))
    axes = q[:, :4]
    for family in c.FAMILIES:
        values = c.descriptor(z, theta, axes, family)
        assert np.isfinite(values).all()
    values = c.descriptor(z, theta, axes, "CTX_COMBINED_PLUS_PC", q)
    assert np.isfinite(values).all()


if __name__ == "__main__":
    test_rank_one_and_eight()
    test_descriptor_label_free_shapes()
    print("2 core tests passed")
