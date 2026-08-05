import pandas as pd

from hsc_tta.budgeted_risk.diagnostics.aggregation import seedwise_validity, subject_efficiency


def test_random_repeats_then_seeds_leave_subject_unit():
    rows=[]
    for subject in ("a","b"):
        for seed in (0,1):
            for repeat in (0,1,2):
                rows.append({"dataset":"hmc","subject_id":subject,"requested_budget":5,"strategy":"random",
                             "calibration_scheme":"S","seed":seed,"repeat":repeat,"gain":repeat+seed})
    out=subject_efficiency(pd.DataFrame(rows),["gain"])
    assert len(out)==2 and set(out.subject_id)=={"a","b"} and (out.gain==1.5).all()


def test_cp_uses_each_seed_subject_count_not_seed_times_subjects():
    rows=[]
    for seed in range(5):
        for subject in range(7):rows.append({"seed":seed,"subject_id":subject,"violation":subject==0})
    out=seedwise_validity(pd.DataFrame(rows))
    assert (out.n_subjects==7).all() and (out.violations==1).all()


def test_random_repeat_violation_is_collapsed_within_subject():
    rows=[]
    for subject in ("a","b"):
        for repeat in range(20):rows.append({"seed":0,"subject_id":subject,"violation":repeat==0})
    out=seedwise_validity(pd.DataFrame(rows))
    assert out.iloc[0].n_subjects==2

