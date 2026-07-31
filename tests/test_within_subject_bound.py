import math
import numpy as np
from hsc_tta.certification import empirical_bernstein_bound


def test_bound_formula_and_edges():
    x=np.array([.1,.2,.3,.2])
    out=empirical_bernstein_bound(x,.05)
    expected=math.sqrt(2*x.var(ddof=1)*math.log(60)/4)+3*math.log(60)/4
    assert np.isclose(out["margin"],expected)
    assert 0 <= out["upper_risk"] <= 1
    assert empirical_bernstein_bound(np.array([0.,0.]))["status"] == "insufficient_future_blocks"

