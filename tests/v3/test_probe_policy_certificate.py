import numpy as np
import pandas as pd
import pytest

from hsc_tta.v3.policy_certificate import calibrate_policy_index, joint_critical_index
from hsc_tta.v3.probe_metrics import (LOG2, compute_probe_diagnostics, effective_class_number,
    jensen_shannon, normalized_set_efficiency, temporal_blocks)
from hsc_tta.v3.probe_policy import ProbePolicy, ProbeThresholds


def _prob(seed=1):
    x=np.random.default_rng(seed).uniform(size=(30,4)); return x/x.sum(1,keepdims=True)


def test_probe_metrics_bounds_grid_and_identity():
    p=_prob(); q=_prob(2); js=jensen_shannon(p,q)
    assert np.all((js>=0)&(js<=LOG2+1e-12)) and np.allclose(jensen_shannon(p,p),0)
    grid=np.r_[np.linspace(.5,.99,20),1.]
    assert normalized_set_efficiency(p,grid)<=1 and 1<=effective_class_number(p)<=4
    diagnostics=compute_probe_diagnostics(p,p,[p,p],[p,p],grid,action_available=True,normalized_update_magnitude=0)
    assert abs(diagnostics.g_set)<1e-12 and abs(diagnostics.d_src)<1e-12 and diagnostics.r_class==pytest.approx(1)
    blocks=temporal_blocks(30); assert len(blocks)==3 and not(set(blocks[0])&set(blocks[1]))


def test_policy_gate_and_tie_break_are_deterministic():
    thresholds=ProbeThresholds(.7,.1,0,.01,2/3,.1,.2); policy=ProbePolicy(thresholds)
    frame=pd.DataFrame([
      {"action":"t3a","action_available":True,"r_class":1.,"normalized_update_magnitude":.01,"g_set":.02,"g_aug":0.,"positive_probe_block_fraction":1.,"temporal_mad":.01,"d_src":.03,"action_cost":1},
      {"action":"adapter","action_available":True,"r_class":1.,"normalized_update_magnitude":.01,"g_set":.02,"g_aug":0.,"positive_probe_block_fraction":1.,"temporal_mad":.01,"d_src":.04,"action_cost":2}])
    assert policy.decide(frame)["selected_action"]=="t3a"
    frame["action_available"]=False
    assert policy.decide(frame)["selected_action"]=="no_tta"


def test_policy_certificate_exact_and_sentinel_rules():
    risks=np.linspace(.5,0,21)
    assert joint_critical_index(risks,0,alpha=.2,epsilon=.01,sentinel_index=20)==12
    assert joint_critical_index(risks,.02,alpha=.2,epsilon=.01,sentinel_index=20)==20
    insufficient=calibrate_policy_index(np.array([1,2,3,4]),delta=.1,sentinel_index=20)
    assert insufficient.insufficient and insufficient.lambda_index==20
    exact=calibrate_policy_index(np.arange(20),delta=.1,sentinel_index=20)
    assert exact.order_index==19 and exact.lambda_index==18
    assert joint_critical_index(np.ones(21),0,alpha=.2,epsilon=.01,sentinel_index=20)==20
    with pytest.raises(ValueError,match="risk curve"): joint_critical_index(np.ones(2),0,alpha=.2,epsilon=.01,sentinel_index=20)
    with pytest.raises(ValueError,match="invalid calibration"): calibrate_policy_index(np.array([]),delta=.1,sentinel_index=20)
    with pytest.raises(ValueError,match="delta"): calibrate_policy_index(np.arange(5),delta=1.,sentinel_index=20)
