from hsc_tta.splits import make_internal_subject_split


def test_task_head_fit_val_and_meta_folds_are_frozen_and_disjoint():
    roles = {
        "task_head_train": [f"t{i}" for i in range(70)],
        "meta_risk_train": [f"m{i}" for i in range(35)],
        "conformal_calibration": [f"c{i}" for i in range(20)],
        "final_test": [f"f{i}" for i in range(26)],
    }
    first = make_internal_subject_split(roles, "hmc", 3)
    second = make_internal_subject_split(roles, "hmc", 3)
    assert first == second
    assert len(first["task_head_fit"]) == 60 and len(first["task_head_val"]) == 10
    assert not set(first["task_head_fit"]) & set(first["task_head_val"])
    assert set(first["meta_risk_folds"].values()) == set(range(5))
