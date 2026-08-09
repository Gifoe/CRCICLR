import pandas as pd
import pytest

from hsc_tta.v2.risk_predictor import assert_risk_u_only_features,fit_risk_predictor


def test_risk_features_are_u_only():
    with pytest.raises(ValueError): assert_risk_u_only_features(["entropy_q50","future_risk"])


def test_risk_predictor_baselines_and_oof():
    rows=[]
    for i in range(15):
        for action in ("no_tta","t3a"):
            rows.append({"subject_id":f"s{i}","action":action,"alpha":.1,"entropy_q50":i/15,
                         "true_critical_index":min(20,i+(action=="t3a"))})
    result=fit_risk_predictor(pd.DataFrame(rows),["action","entropy_q50"])
    assert {"constant_critical_index","action_mean"}.issubset(set(result.results.model))
    assert result.oof.residual_source.eq("meta_oof").all()
