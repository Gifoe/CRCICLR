import numpy as np
from hsc_tta.actions import T3A


def test_t3a_reset_discards_target_supports():
    model = T3A(np.eye(3), filter_k=-1).adapt(np.eye(3), np.eye(3) * 4)
    assert len(model.supports) == 6
    model.reset()
    assert len(model.supports) == 3
