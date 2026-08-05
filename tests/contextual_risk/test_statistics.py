import numpy as np

from hsc_tta.contextual_risk.statistics import clopper_pearson_upper, paired_bootstrap_ci


def test_exact_binomial_and_bootstrap_are_deterministic():
    assert 0 < clopper_pearson_upper(0,20) < .2
    a=paired_bootstrap_ci(np.array([1.,2.,3.]))
    b=paired_bootstrap_ci(np.array([1.,2.,3.]))
    assert a==b and a[0]<=2<=a[1]
