import numpy as np
import pandas as pd
from hsc_tta.metrics import subject_metrics,subject_bootstrap_ci


def test_subject_level_metrics_and_bootstrap():
    d=pd.DataFrame({"subject_id":["a","b","c"],"status":["certified","certified","uncertified"],"true_future_risk":[.1,.3,1.],"certified_upper_bound":[.2,.2,1.],"average_set_size":[2.,3.,5.],"singleton_rate":[.5,.2,0.],"selected_action":["no_tta","t3a",None],"selected_error":[.1,.4,1.],"no_tta_error":[.1,.2,1.]})
    m=subject_metrics(d,.2,5)
    assert np.isclose(m["csr"],2/3) and np.isclose(m["negative_adaptation_rate"],1)
    lo,hi=subject_bootstrap_ci(d,lambda x:(x.status=="certified").mean(),100,0); assert 0<=lo<=hi<=1

