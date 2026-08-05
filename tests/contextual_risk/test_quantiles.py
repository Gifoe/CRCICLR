import math

from hsc_tta.contextual_risk.quantiles import higher_quantile, split_conformal_upper


def test_higher_quantile_uses_ceil_order_statistic():
    assert higher_quantile([4, 1, 3, 2], .50) == 2
    assert higher_quantile([4, 1, 3, 2], .90) == 4


def test_split_conformal_insufficient_is_sentinel():
    assert math.isinf(split_conformal_upper(range(8), .10))
    assert split_conformal_upper(range(9), .10) == 8
