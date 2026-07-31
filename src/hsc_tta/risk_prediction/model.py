from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score


def enforce_lambda_monotonicity(frame: pd.DataFrame, value: str = "predicted_risk") -> pd.DataFrame:
    required = {"subject_id", "action", "lambda", value}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    out = frame.sort_values(["subject_id", "action", "lambda"]).copy()
    out[value] = out.groupby(["subject_id", "action"], sort=False)[value].transform(lambda x: np.minimum.accumulate(x.to_numpy(float)))
    return out


class MetaRiskPredictor:
    def __init__(self, feature_columns: list[str], random_state: int = 0):
        self.feature_columns = list(feature_columns)
        self.model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, learning_rate=0.05, l2_regularization=1.0, random_state=random_state)

    def fit(self, frame: pd.DataFrame, target: str = "upper_risk") -> "MetaRiskPredictor":
        if frame.subject_id.nunique() < 2:
            raise ValueError("at least two independent subjects are required")
        self.model.fit(frame[self.feature_columns], frame[target])
        return self

    def grouped_cv_score(self, frame: pd.DataFrame, target: str = "upper_risk", folds: int = 5) -> np.ndarray:
        n = frame.subject_id.nunique()
        if n < 2:
            raise ValueError("at least two independent subjects are required")
        cv = GroupKFold(n_splits=min(folds, n))
        return cross_val_score(self.model, frame[self.feature_columns], frame[target], groups=frame.subject_id, cv=cv, scoring="neg_mean_absolute_error")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(frame[self.feature_columns]), 0, 1)

    def save(self, path: str | Path) -> None:
        joblib.dump({"feature_columns": self.feature_columns, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MetaRiskPredictor":
        payload = joblib.load(path)
        obj = cls(payload["feature_columns"])
        obj.model = payload["model"]
        return obj

