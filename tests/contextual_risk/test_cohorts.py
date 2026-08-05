from hsc_tta.contextual_risk.cohorts import attach_screening_folds, build_master_cohorts


def test_master_cohort_is_seed_independent_and_disjoint():
    subjects=[f"hmc:{i:03d}" for i in range(100)]
    a=attach_screening_folds(build_master_cohorts("hmc",subjects))
    b=attach_screening_folds(build_master_cohorts("hmc",reversed(subjects)))
    assert a.sort_values("subject_id").master_cohort.tolist()==b.sort_values("subject_id").master_cohort.tolist()
    assert a.subject_id.nunique()==len(a)
    assert a.master_cohort.value_counts().to_dict()=={"method_development":60,"formal_calibration":20,"internal_final_evaluation":20}
    assert set(a[a.master_cohort!="method_development"].screening_fold)=={-1}
