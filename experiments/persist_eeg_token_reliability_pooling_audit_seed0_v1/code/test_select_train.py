import numpy as np
import pandas as pd

from select_train import candidates, probe_score_excluding, select


def test_candidate_grid_and_ba_tie_break():
    r = np.linspace(-1, 1, 248)
    u = np.sin(np.linspace(0, 7, 248))
    e = np.linspace(1, 2, 248)
    assert len(candidates(r, u, e)) == 36
    rows = []
    for subject in ("a", "b"):
        for c in (.001, .01):
            rows.append({"held_train_subject": subject, "method": "MEAN_ALL", "parameter": "none",
                         "C": c, "BA": .6, "macro_F1": .6, "NLL": .7})
    _, chosen = select(pd.DataFrame(rows), 2)
    assert len(chosen) == 1
    assert chosen.iloc[0]["C"] == .001


def test_nested_probe_ignores_entire_excluded_subject():
    rng = np.random.default_rng(7)
    subjects = np.repeat(np.asarray(["0", "1", "2", "3", "4"]), 8)
    sessions = np.tile(np.repeat([1, 2], 4), 5)
    y = np.tile([0, 0, 1, 1, 0, 0, 1, 1], 5)
    z = rng.normal(size=(40, 3, 4)).astype(np.float32)
    z[:, :, 0] += y[:, None] * .8
    before = probe_score_excluding(z, y, subjects, sessions, "0")
    modified = z.copy()
    modified[subjects == "0"] += 1000
    after = probe_score_excluding(modified, y, subjects, sessions, "0")
    np.testing.assert_array_equal(before, after)
