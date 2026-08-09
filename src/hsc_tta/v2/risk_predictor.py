from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FORBIDDEN=frozenset({"true_critical_index","critical_index","future_risk","true_future_risk","future_logits",
                     "future_labels","macro_f1","argmax_error","benefit_target"})


def assert_risk_u_only_features(columns:Iterable[str])->None:
    bad=sorted(c for c in columns if c in FORBIDDEN or c.startswith("future_"))
    if bad: raise ValueError(f"risk features contain V-derived fields: {bad}")


@dataclass
class RiskFitResult:
    model_name:str
    oof:pd.DataFrame
    results:pd.DataFrame
    model:object


def _prep(frame:pd.DataFrame,features:list[str])->ColumnTransformer:
    cat=[c for c in features if frame[c].dtype==object]; num=[c for c in features if c not in cat]
    return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer()),("scale",StandardScaler())]),num),
                              ("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat)])


def fit_risk_predictor(frame:pd.DataFrame,features:list[str],*,target:str="true_critical_index",
                       group_column:str="subject_id",sentinel_index:int=20)->RiskFitResult:
    assert_risk_u_only_features(features); data=frame.reset_index(drop=True)
    groups=data[group_column]; split=GroupKFold(min(5,groups.nunique()))
    models={"elastic_net":ElasticNet(alpha=.01,l1_ratio=.1,max_iter=10000),
            "hist_gradient_boosting":HistGradientBoostingRegressor(max_depth=3,max_iter=100,l2_regularization=1)}
    rows=[]; results=[]; y=data[target].to_numpy(float)
    baselines={"constant_critical_index":np.full(len(data),np.median(y)),
               "action_mean":data.action.map(data.groupby("action")[target].mean()).to_numpy(float)}
    for name,regressor in models.items():
        prediction=np.full(len(data),np.nan)
        for train,test in split.split(data,groups=groups):
            model=Pipeline([("prep",_prep(data.iloc[train],features)),("model",regressor)])
            model.fit(data.iloc[train][features],y[train]); prediction[test]=np.clip(model.predict(data.iloc[test][features]),0,sentinel_index)
        results.append({"model":name,"mae":mean_absolute_error(y,prediction),"underestimation_rate":float(np.mean(prediction<y))})
        rows.extend({"model":name,"subject_id":data.loc[i,group_column],"action":data.loc[i,"action"],
                     "alpha":data.loc[i,"alpha"],"true_critical_index":y[i],"predicted_critical_index":prediction[i],
                     "residual_source":"meta_oof"} for i in range(len(data)))
    for name,prediction in baselines.items():
        results.append({"model":name,"mae":mean_absolute_error(y,prediction),"underestimation_rate":float(np.mean(prediction<y))})
    result_frame=pd.DataFrame(results); chosen=result_frame[result_frame.model.isin(models)].sort_values(["mae","model"]).iloc[0].model
    final=Pipeline([("prep",_prep(data,features)),("model",models[chosen])]).fit(data[features],y)
    return RiskFitResult(str(chosen),pd.DataFrame(rows),result_frame,final)
