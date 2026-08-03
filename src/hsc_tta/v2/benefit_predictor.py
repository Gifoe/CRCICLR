from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FORBIDDEN = frozenset({"future_labels", "future_logits", "future_risk", "future_f1", "true_future_risk",
                       "macro_f1", "balanced_accuracy", "cohen_kappa", "argmax_error", "benefit_target"})


def assert_u_only_features(columns: Iterable[str]) -> None:
    bad = sorted(c for c in columns if c in FORBIDDEN or c.startswith("future_"))
    if bad:
        raise ValueError(f"benefit features contain V-derived fields: {bad}")


def benefit_target(no_tta_error: np.ndarray, action_error: np.ndarray) -> np.ndarray:
    no_tta = np.asarray(no_tta_error, float); action = np.asarray(action_error, float)
    if no_tta.shape != action.shape: raise ValueError("benefit target shape mismatch")
    return no_tta - action


@dataclass
class BenefitFitResult:
    model_name: str
    oof: pd.DataFrame
    results: pd.DataFrame
    model: object


def _preprocessor(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    categorical = [c for c in features if frame[c].dtype == object or str(frame[c].dtype).startswith("string")]
    numeric = [c for c in features if c not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def fit_benefit_predictor(frame: pd.DataFrame, features: list[str], *, group_column: str="subject_id",
                          target_column: str="benefit_target", folds: int=5) -> BenefitFitResult:
    assert_u_only_features(features)
    data=frame[frame.action!="no_tta"].reset_index(drop=True).copy()
    if data.empty: raise ValueError("no available TTA rows")
    models={
        "elastic_net": ElasticNet(alpha=.01,l1_ratio=.2,max_iter=10000),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=100,max_depth=3,l2_regularization=1),
        "pairwise_sign_gain": Ridge(alpha=1.0),
    }
    splitter=GroupKFold(n_splits=min(folds,data[group_column].nunique()))
    rows=[]; scores=[]
    for name,regressor in models.items():
        prediction=np.full(len(data),np.nan)
        for train,test in splitter.split(data,groups=data[group_column]):
            model=Pipeline([("prep",_preprocessor(data.iloc[train],features)),("model",regressor)])
            model.fit(data.iloc[train][features],data.iloc[train][target_column]); prediction[test]=model.predict(data.iloc[test][features])
        truth=data[target_column].to_numpy(float)
        sign_truth=truth>0; sign_prediction=prediction>0
        balanced=balanced_accuracy_score(sign_truth,sign_prediction) if len(np.unique(sign_truth))>1 else np.nan
        correlation=spearmanr(truth,prediction).statistic
        scores.append({"model":name,"gain_mae":mean_absolute_error(truth,prediction),
                       "sign_balanced_accuracy":balanced,"spearman":correlation})
        rows.extend({"model":name,"subject_id":data.loc[i,group_column],"action":data.loc[i,"action"],
                     "true_gain":truth[i],"predicted_gain":prediction[i]} for i in range(len(data)))
    truth=data[target_column].to_numpy(float)
    simple={"constant_zero":np.zeros(len(data)),"global_mean":np.full(len(data),truth.mean())}
    for feature,name in (("entropy_q50","entropy_only"),("prediction_agreement","agreement_only")):
        if feature in data:
            surrogate=Ridge(alpha=1).fit(data[[feature]].fillna(0),truth)
            simple[name]=surrogate.predict(data[[feature]].fillna(0))
    for name,prediction in simple.items():
        scores.append({"model":name,"gain_mae":mean_absolute_error(truth,prediction),
                       "sign_balanced_accuracy":balanced_accuracy_score(truth>0,prediction>0) if len(np.unique(truth>0))>1 else np.nan,
                       "spearman":spearmanr(truth,prediction).statistic})
    results=pd.DataFrame(scores)
    candidates=results[results.model.isin(models)]
    chosen=candidates.sort_values(["gain_mae","sign_balanced_accuracy","spearman","model"],ascending=[True,False,False,True]).iloc[0].model
    final=Pipeline([("prep",_preprocessor(data,features)),("model",models[chosen])]).fit(data[features],truth)
    return BenefitFitResult(str(chosen),pd.DataFrame(rows),results,final)


def save_benefit_result(result: BenefitFitResult, directory: str|Path) -> None:
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    joblib.dump(result.model,directory/"benefit_model.joblib")
    result.oof.to_parquet(directory/"BENEFIT_PREDICTOR_OOF.parquet",index=False)
    result.results.to_csv(directory/"BENEFIT_PREDICTOR_RESULTS.csv",index=False)
