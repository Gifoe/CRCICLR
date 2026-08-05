import numpy as np

from hsc_tta.budgeted_risk.diagnostics.calibration_schemes import sentinel_transition


def test_raw_certified_and_correction_cost_arithmetic():
    raw=np.ceil([4.1,8.0]).astype(int);cert=np.ceil(np.array([4.1,8.0])+2.2).astype(int)
    size=np.arange(21,dtype=float)**2
    method=size[cert]-size[raw]
    global_cost=size[[9,9]]-size[[7,7]]
    excess=method-global_cost
    assert raw.tolist()==[5,8] and cert.tolist()==[7,11]
    assert np.allclose(excess,method-global_cost)
    assert sentinel_transition(raw,cert).sum()==0

