import numpy as np
import pandas as pd
from hsc_tta.certification import fit_simultaneous_quantile, apply_certificate


def test_higher_quantile_is_subject_level_and_order_invariant():
    f=pd.DataFrame({"subject_id":sum(([f"s{i}"]*2 for i in range(20)),[]),"upper_risk":np.repeat(np.linspace(.1,.3,20),2),"predicted_risk":0.05})
    a=fit_simultaneous_quantile(f,.1); b=fit_simultaneous_quantile(f.sample(frac=1,random_state=7),.1)
    assert a.q == b.q and a.order_k == 19
    assert np.all(apply_certificate(np.array([.9]),a)<=1)


def test_conservative_fallback_when_k_exceeds_m():
    f=pd.DataFrame({"subject_id":["a","b"],"upper_risk":[.2,.3],"predicted_risk":[.1,.1]})
    assert fit_simultaneous_quantile(f,.1).q == 1

