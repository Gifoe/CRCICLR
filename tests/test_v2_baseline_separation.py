import pandas as pd

from hsc_tta.v2.baselines import _global_index,_policy_actions


def test_baseline_calibration_is_quantile_only():
    assert _global_index([1]*18+[20]*2,.1)==20


def test_no_tta_policy_is_fixed():
    frame=pd.DataFrame([{"subject_id":"s","action":"no_tta","action_available":True,"entropy_q50":1,
                         "prediction_agreement":1,"action_cost":0},
                        {"subject_id":"s","action":"official_t3a","action_available":True,"entropy_q50":1,
                         "prediction_agreement":1,"action_cost":1}])
    assert _policy_actions("no_tta_global_crc",frame).loc["s"]=="no_tta"
