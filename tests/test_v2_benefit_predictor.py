import numpy as np
import pandas as pd
import pytest

from hsc_tta.v2.benefit_predictor import assert_u_only_features, benefit_target, fit_benefit_predictor


def test_benefit_target_direction():
    assert benefit_target(np.array([.4]),np.array([.3]))[0]==pytest.approx(.1)


def test_benefit_features_reject_future():
    with pytest.raises(ValueError): assert_u_only_features(["entropy_q50","future_logits"])


def test_benefit_predictor_group_oof():
    rows=[]
    for i in range(15):
        for action,offset in (("t3a",.02),("robust",-.01)):
            rows.append({"subject_id":f"s{i}","action":action,"entropy_q50":i/15,
                         "prediction_agreement":1-i/30,"benefit_target":offset+i/1000})
    result=fit_benefit_predictor(pd.DataFrame(rows),["action","entropy_q50","prediction_agreement"])
    assert result.oof.groupby("model").size().min()==30
    assert {"constant_zero","global_mean"}.issubset(set(result.results.model))
