import pandas as pd
import pytest

from hsc_tta.v2.development_surfaces import _features
from hsc_tta.v2.benefit_predictor import assert_u_only_features


def test_development_feature_schema_is_u_only():
    import numpy as np
    p=np.full((5,3),1/3); h=np.zeros((5,4))
    row=_features("s","no_tta",p,p,h,{"status":"ok"})
    assert_u_only_features(row.keys())
    assert not any(name.startswith("future_") for name in row)
