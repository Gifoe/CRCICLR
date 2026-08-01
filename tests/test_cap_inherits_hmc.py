from hsc_tta.splits import make_internal_subject_split


def test_cap_has_no_task_head_or_meta_predictor_training_split():
    roles = {"target_site_calibration": [f"c{i}" for i in range(25)], "external_final_test": [f"t{i}" for i in range(78)]}
    internal = make_internal_subject_split(roles, "cap", 0)
    assert internal["task_head"] == "inherit_hmc"
    assert internal["meta_predictor"] == "inherit_hmc"
    assert internal["task_head_fit"] == [] and internal["meta_risk_folds"] == {}
