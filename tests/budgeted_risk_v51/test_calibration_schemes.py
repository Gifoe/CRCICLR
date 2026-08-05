import numpy as np
import pandas as pd

from hsc_tta.budgeted_risk.diagnostics.calibration_schemes import (
    S1, S2, S3, S4, conformal_q, fit_nonnegative_scale, fold_split,
    is_outlier_driven, predict_scale, scale_design, sentinel_transition,
)


def test_s1_fold_assignment():
    split = fold_split(S1, 4)
    assert split.evaluation == (4,) and split.calibration == (0,) and split.training == (1, 2, 3)


def test_s2_s3_fold_assignment_and_no_overlap():
    for scheme in (S2, S3):
        split = fold_split(scheme, 1)
        assert split.evaluation == (1,) and split.calibration == (2, 3) and split.training == (4, 0)
        assert not (set(split.evaluation) & set(split.calibration) | set(split.evaluation) & set(split.training) | set(split.calibration) & set(split.training))


def test_s4_never_uses_eval_in_pool():
    split = fold_split(S4, 3)
    assert 3 not in split.calibration and split.evaluation == (3,) and len(split.calibration) == 4


def test_finite_sample_rank_and_sentinel():
    q, k = conformal_q(np.arange(13), .1)
    assert k == 13 and q == 12
    q, k = conformal_q(np.arange(8), .1)
    assert k == 9 and q == 20


def test_scale_positive_nonnegative_and_design_fields():
    frame = pd.DataFrame({"local_kappa_std":[1.,2.,3.],"local_kappa_q90":[4.,5.,6.],"local_kappa_q50":[1.,2.,2.],
                          "effective_budget":[5,10,20],"prefix_instability":[.2,.4,.1]})
    design = scale_design(frame); coef = fit_nonnegative_scale(design, np.array([1.,2.,3.]))
    assert design.shape == (3,4) and np.all(coef >= 0) and np.all(predict_scale(design,coef) > 0)


def test_scale_training_api_has_no_calibration_argument():
    assert fit_nonnegative_scale.__code__.co_varnames[:2] == ("design", "residual")


def test_sentinel_transition_and_outlier_rule():
    assert sentinel_transition([19,20,18],[20,20,19]).tolist() == [True,False,False]
    assert is_outlier_driven(2,0,0) and is_outlier_driven(0,.1,0) and is_outlier_driven(0,0,.05)
    assert not is_outlier_driven(1.9,.09,.049)

