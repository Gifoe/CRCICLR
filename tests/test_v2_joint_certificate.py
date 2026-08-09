import numpy as np
import pandas as pd
import pytest

from hsc_tta.v2.joint_certificate import (JointScales, estimate_scales, finite_sample_quantile,
                                           joint_bounds, subject_joint_scores)
from hsc_tta.v2.selector_v2 import select_joint_action


def test_benefit_lower_and_risk_upper_directions():
    upper,lower=joint_bounds(np.array([2.2]),np.array([.1]),2,JointScales(1.5,.02),20)
    assert upper[0]==6 and lower[0]==pytest.approx(.06)


def test_scale_meta_oof_only_and_joint_subject_max():
    frame=pd.DataFrame({"subject_id":["a","a"],"action":["no_tta","t3a"],"true_critical_index":[3,5],
        "predicted_critical_index":[2,2],"true_benefit":[0,.02],"predicted_benefit":[0,.08],"residual_source":["meta_oof"]*2})
    scales=estimate_scales(frame); scores=subject_joint_scores(frame,scales)
    assert len(scores)==1 and scores.joint_score.iloc[0]>=0
    frame.residual_source="calibration"
    with pytest.raises(ValueError): estimate_scales(frame)


def test_finite_sample_quantile_exactness():
    q,k=finite_sample_quantile(np.arange(20),.1)
    assert k==19 and q==18


def _candidates(safe=True,beneficial=True,available=True,no_index=2):
    return pd.DataFrame([
        {"action":"no_tta","available":True,"certified_critical_index":no_index,"benefit_lower":0,
         "context_average_set_size":2,"adaptation_cost":0},
        {"action":"t3a","available":available,"certified_critical_index":3 if safe else 20,
         "benefit_lower":.02 if beneficial else -.01,"context_average_set_size":1.5,"adaptation_cost":1}])


def test_joint_selector_accepts_only_safe_and_beneficial():
    assert select_joint_action(_candidates(),sentinel_index=20)["selected_action"]=="t3a"
    for frame in (_candidates(safe=False),_candidates(beneficial=False),_candidates(available=False)):
        assert select_joint_action(frame,sentinel_index=20)["selected_action"]=="no_tta"


def test_all_harmful_and_full_fallback():
    result=select_joint_action(_candidates(beneficial=False,no_index=20),sentinel_index=20)
    assert result["selected_action"]=="no_tta" and result["full_set_fallback"]
