import numpy as np
import pandas as pd
import json
from pathlib import Path

from hsc_tta.budgeted_risk.diagnostics.calibration_schemes import S2,S3,conformal_q,fold_split
from hsc_tta.budgeted_risk.diagnostics.mini import _fit_scale,_mini_scale_design,_pass,_predict_scale


def test_s2_s3_exact_split_no_overlap():
    for e in range(5):
        for scheme in (S2,S3):
            s=fold_split(scheme,e);assert s.evaluation==(e,);assert s.calibration==((e+1)%5,(e+2)%5);assert s.training==((e+3)%5,(e+4)%5)
            assert not (set(s.evaluation)&set(s.calibration) or set(s.evaluation)&set(s.training) or set(s.calibration)&set(s.training))


def test_finite_rank():
    assert conformal_q(np.arange(26),.1)==(24.,25);assert conformal_q(np.arange(36),.1)==(33.,34)


def test_scale_positive_and_nonnegative():
    f=pd.DataFrame({"local_kappa_std":[1,2,3],"local_kappa_q90":[4,5,6],"local_kappa_q50":[1,2,2],"effective_budget":[20]*3,"prefix_instability":[.1,.2,.3]})
    c=_fit_scale(_mini_scale_design(f),np.asarray([1,2,3]));assert len(c)==3 and np.all(c>=0) and np.all(_predict_scale(f,c)>0)


def test_scale_api_has_no_calibration_argument():
    assert _fit_scale.__code__.co_varnames[:2]==("x","y")


def test_gate_requires_all_components():
    class R:pass
    r=R();r.mean_seed_violation=.1;r.worst_seed_violation=.2;r.max_seed_cp_upper=.2;r.relative_gain_vs_global=.05;r.paired_gain_ci_low=.001;r.oracle_gain_recovered=.2;r.sentinel_delta=.05;r.sentinel_transition_rate=.1;r.q_driver_unstable_fold_rate=.2;r.evaluation_loo_sign_stable=True
    assert _pass(r);r.paired_gain_ci_low=0;assert not _pass(r)


REPO=Path(__file__).resolve().parents[2]


def test_complete_jobs_share_corresponding_global_and_base_predictor():
    out=REPO/"outputs/budgeted_risk_v51_mini"
    if not out.exists():return
    assert len(list((out/"base_predictions").glob("BASE_PREDICTIONS_*.parquet")))==50
    assert len(list((out/"base_predictions").glob("BASE_MODEL_*.json")))==50
    assert len(list((out/"job_results").glob("JOB_RESULTS_*.parquet")))==50
    for p in (out/"job_results").glob("JOB_RESULTS_*.parquet"):
        d=pd.read_parquet(p);wide=d.pivot(index="subject_id",columns="scheme",values=["raw_prediction","global_base_index","global_q","global_certified_index"])
        for metric in ("raw_prediction","global_base_index","global_q","global_certified_index"):
            assert np.allclose(wide[metric].iloc[:,0],wide[metric].iloc[:,1])


def test_scale_and_q_driver_subjects_are_training_or_calibration_only():
    out=REPO/"outputs/budgeted_risk_v51_mini"
    if not out.exists():return
    q=pd.read_parquet(out/"Q_DRIVER_RESULTS.parquet")
    for p in (out/"base_predictions").glob("BASE_MODEL_*.json"):
        m=json.loads(p.read_text());assert set(m["scale_training_subjects"])==set(m["scale_training_subjects"])-set(m["calibration_subjects"])-set(m["evaluation_subjects"])
        current=q[(q.dataset==m["dataset"])&(q.seed==m["seed"])&(q.outer_fold==m["outer_fold"])]
        assert set(current.q_driver_subject)<=set(m["calibration_subjects"])


def test_protected_cohort_not_in_cache_manifest():
    cohort_path=REPO/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet";manifest_path=REPO/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet"
    if not cohort_path.exists():return
    cohort=pd.read_parquet(cohort_path);manifest=pd.read_parquet(manifest_path)
    allowed=set(zip(cohort[cohort.master_cohort=="method_development"].dataset,cohort[cohort.master_cohort=="method_development"].subject_id))
    assert set(zip(manifest.dataset,manifest.subject_id))<=allowed
