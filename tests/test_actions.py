import numpy as np
from hsc_tta.actions import NoTTA,T3A,EntropyAdapterMock


def test_action_interfaces_and_subject_reset():
    z=np.eye(3); logits=np.eye(3)*5
    assert NoTTA().predict_proba(z,logits).shape==(3,3)
    model=T3A(.5).adapt(z,logits)
    assert np.array_equal(model.predict_proba(z).argmax(1),np.arange(3))
    assert EntropyAdapterMock().adapt(z,logits).predict_proba(z,logits).shape==(3,3)

