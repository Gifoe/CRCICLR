import numpy as np
import pandas as pd
from hsc_tta.risk_prediction import MetaRiskPredictor,enforce_lambda_monotonicity


def test_grouped_fit_monotonicity_and_reload(tmp_path):
    rows=[]
    for s in range(8):
      for l in [.5,.7,.9]: rows.append({"subject_id":f"s{s}","action":"no_tta","lambda":l,"x":s/8,"upper_risk":.5-.2*l+.01*s})
    f=pd.DataFrame(rows); m=MetaRiskPredictor(["x","lambda"]).fit(f)
    assert len(m.grouped_cv_score(f,folds=4))==4
    f["predicted_risk"]=m.predict(f); fixed=enforce_lambda_monotonicity(f)
    assert np.all(fixed.groupby(["subject_id","action"]).predicted_risk.apply(lambda x: np.all(np.diff(x)<=1e-12)))
    p=tmp_path/"m.joblib"; m.save(p); assert np.allclose(m.predict(f),MetaRiskPredictor.load(p).predict(f))

