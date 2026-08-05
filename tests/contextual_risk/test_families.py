import numpy as np

from hsc_tta.contextual_risk.families import APSFamily, RAPSFamily, TPSFamily, deterministic_order


P = np.array([[.5, .5, 0.0], [.1, .7, .2]])


def test_deterministic_class_id_tie_break():
    assert deterministic_order(P)[0].tolist() == [0, 1, 2]


def test_all_families_have_argmax_full_sentinel_and_monotonicity():
    for family in (TPSFamily(), APSFamily(), RAPSFamily(2, .05)):
        sets, repairs = family.build_sets(P)
        assert sets.shape == (2, 21, 3)
        assert all(sets[i, :, P[i].argmax()].all() for i in range(len(P)))
        assert sets[:, -1].all()
        assert np.all(sets[:, 1:] | ~sets[:, :-1])
        assert repairs >= 0
