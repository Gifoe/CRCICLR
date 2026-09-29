import numpy as np

from token_scores import cosine_reliability, decomposition, directions, label_null, reproducibility, spread


def test_matched_direction_and_token_order():
    # Eight subjects, two sessions, balanced binary trials, four tiny tokens.
    z, y, owner, session = [], [], [], []
    for subject in range(8):
        for visit in (1, 2):
            for label in (0, 1):
                for _ in range(3):
                    block = np.zeros((4, 3), np.float32)
                    block[:, 0] = label * np.array([1., .7, -.2, .1])
                    z.append(block); y.append(label); owner.append(f"sub-{subject:02d}"); session.append(visit)
    d, counts = directions(np.stack(z), np.array(y), np.array(owner), np.array(session),
                           [f"sub-{i:02d}" for i in range(8)])
    assert d.shape == (8, 2, 4, 3)
    assert np.all(counts == 3)
    assert np.allclose(cosine_reliability(d), 1)
    assert spread(cosine_reliability(d))["top_bottom_quartile_gap"] == 0
    varied = d.copy()
    varied[:, 1, :, 1] = np.linspace(.1, .8, 8)[:, None] * np.arange(1, 5)[None, :]
    assert len(reproducibility(varied, 200, fold=0)) == 200
    assert len(label_null(np.stack(z), np.array(y), np.array(owner), np.array(session),
                          [f"sub-{i:02d}" for i in range(8)], 500, fold=0)) == 500
    assert abs(sum(decomposition(np.array([1., 2., 3., 4.]), 2, 2).values()) - 1) < 1e-10
