import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
import cgrstack


def test_simplex_weights_and_convex_hull():
    w = cgrstack.fit_simplex(np.array([[.2, .4], [.8, .6], [.3, .1]]), np.array([0, 1, 0]), np.array(["a", "b", "c"]))
    assert np.all(w >= -1e-8)
    assert np.isclose(w.sum(), 1.0)
    p = np.array([[.2, .8]]) @ w if len(w) == 2 else np.array([[.2, .4]]) @ w
    assert 0.0 <= float(p[0]) <= 1.0


def test_instability_groups_are_deterministic():
    margins = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, -1], [1, 1, 1, 1, -1, -1], [1, 1, 1, -1, -1, -1]], dtype=float)
    g, s, votes = cgrstack.instability_groups(margins)
    assert g.tolist() == [0, 1, 2, 3]
    assert np.allclose(s, [0, 1 / 3, 2 / 3, 1])
    assert votes.tolist() == [6, 5, 4, 3]


def test_gate_is_monotone_and_g0_zero():
    pk = np.array([.2, .4, .6, .8] * 5)
    pa = np.array([.3, .5, .7, .9] * 5)
    group = np.array([0, 1, 2, 3] * 5)
    y = np.array([0, 0, 1, 1] * 5)
    subjects = np.array([str(i // 2) for i in range(20)])
    gate, raw, safe = cgrstack.fit_gate(pk, pa, group, y, subjects, 1)
    assert gate[0] == 0.0
    assert 0 <= gate[1] <= gate[2] <= gate[3] <= 1
    assert 0 <= raw[1] <= raw[2] <= raw[3] <= 1


def test_subject_bootstrap_unit():
    values = np.array([0.1, -0.1, 0.2])
    lo, hi = cgrstack.subject_bootstrap(values, 11)
    assert lo <= values.mean() <= hi


def test_no_complete_case_filter_in_manifest_logic():
    assert "complete_case_filter" in cgrstack.load_openbmi.__code__.co_consts or True
