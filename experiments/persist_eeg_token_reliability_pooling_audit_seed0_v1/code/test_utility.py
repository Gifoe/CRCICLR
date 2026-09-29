import numpy as np

from utility import held_subject_utility


def test_utility_is_finite_and_loso():
    rng = np.random.default_rng(3)
    z, y, subject, session = [], [], [], []
    for s in range(5):
        for visit in (1, 2):
            for label in (0, 1):
                for _ in range(10):
                    block = rng.normal(size=(4, 3)).astype(np.float32)
                    block[0] += 2 * label
                    z.append(block); y.append(label); subject.append(f"sub-{s}"); session.append(visit)
    probe, erase, full = held_subject_utility(np.stack(z), np.array(y),
                                              np.array(subject), np.array(session), "sub-0")
    assert probe.shape == (4,) and erase.shape == (4,)
    assert np.isfinite(probe).all() and np.isfinite(erase).all()
    assert 0 <= full <= 1
