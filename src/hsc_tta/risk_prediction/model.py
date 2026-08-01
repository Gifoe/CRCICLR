from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


FULL_CURVE_KEY = ("dataset", "seed", "episode_id", "subject_id", "action", "alpha")
PARAMETER_GRID = {
    "max_leaf_nodes": (3, 7),
    "learning_rate": (0.03, 0.05),
    "l2_regularization": (1.0, 5.0),
    "max_iter": (100,),
}


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")


def subject_group_ids(frame: pd.DataFrame) -> np.ndarray:
    """Build collision-free subject groups across datasets, seeds, and episodes."""
    columns = ["dataset", "seed", "episode_id", "subject_id"]
    _require_columns(frame, set(columns))
    return frame[columns].astype(str).agg("\x1f".join, axis=1).to_numpy()


def enforce_lambda_monotonicity(frame: pd.DataFrame, value: str = "future_risk") -> pd.DataFrame:
    """Legacy curve utility using the complete isolation key."""
    required = {*FULL_CURVE_KEY, "lambda", value}
    _require_columns(frame, required)
    order = [*FULL_CURVE_KEY, "lambda"]
    out = frame.sort_values(order, kind="mergesort").copy()
    out[value] = out.groupby(list(FULL_CURVE_KEY), sort=False)[value].transform(
        lambda x: np.minimum.accumulate(x.to_numpy(float))
    )
    return out


class CriticalIndexPredictor:
    """Low-capacity alpha-specific critical-index predictor with grouped CV."""

    def __init__(
        self,
        feature_columns: list[str],
        *,
        alpha: float,
        n_nontrivial_lambdas: int,
        random_state: int = 0,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if n_nontrivial_lambdas < 1:
            raise ValueError("n_nontrivial_lambdas must be positive")
        self.feature_columns = list(feature_columns)
        self.alpha = float(alpha)
        self.n_nontrivial_lambdas = int(n_nontrivial_lambdas)
        self.random_state = int(random_state)
        self.model: HistGradientBoostingRegressor | None = None
        self.best_params: dict[str, object] | None = None
        self.cv_results: list[dict[str, object]] = []

    def _base_model(self, params: dict[str, object]) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(random_state=self.random_state, **params)

    def fit(
        self,
        frame: pd.DataFrame,
        target: str = "critical_index",
        folds: int = 5,
        fold_column: str | None = None,
    ) -> "CriticalIndexPredictor":
        required = {*self.feature_columns, target, "alpha", "dataset", "seed", "episode_id", "subject_id"}
        _require_columns(frame, required)
        if frame["alpha"].nunique() != 1 or not np.isclose(float(frame["alpha"].iloc[0]), self.alpha):
            raise ValueError("predictor must be fitted on exactly its configured alpha")
        y = frame[target].to_numpy(float)
        if np.any(~np.isfinite(y)) or np.any((y < 0) | (y > self.n_nontrivial_lambdas)):
            raise ValueError("critical-index target is outside [0, L]")
        groups = subject_group_ids(frame)
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError("at least two independent subjects are required")
        if fold_column is not None:
            _require_columns(frame, {fold_column})
            fixed_folds = frame[fold_column].to_numpy(int)
            unique_folds = np.unique(fixed_folds)
            if not np.array_equal(unique_folds, np.arange(len(unique_folds))):
                raise ValueError("fixed folds must be consecutive integers from zero")
            splits = [(np.flatnonzero(fixed_folds != fold), np.flatnonzero(fixed_folds == fold))
                      for fold in unique_folds]
        else:
            cv = GroupKFold(n_splits=min(folds, n_groups))
            splits = list(cv.split(frame[self.feature_columns], y, groups))
        x = frame[self.feature_columns]
        candidates = [
            dict(zip(PARAMETER_GRID, values))
            for values in product(*(PARAMETER_GRID[name] for name in PARAMETER_GRID))
        ]
        scored: list[tuple[float, str, dict[str, object]]] = []
        self.cv_results = []
        for params in candidates:
            fold_scores: list[float] = []
            base = self._base_model(params)
            for train, valid in splits:
                model = clone(base).fit(x.iloc[train], y[train])
                prediction = np.clip(model.predict(x.iloc[valid]), 0, self.n_nontrivial_lambdas)
                fold_scores.append(float(mean_absolute_error(y[valid], prediction)))
            mean_mae = float(np.mean(fold_scores))
            signature = json.dumps(params, sort_keys=True)
            self.cv_results.append({"params": params, "fold_mae": fold_scores, "mean_mae": mean_mae})
            scored.append((mean_mae, signature, params))
        # Protocol tie-break: smaller tree, larger L2, smaller learning rate.
        _, _, self.best_params = min(
            scored,
            key=lambda item: (
                item[0],
                int(item[2]["max_leaf_nodes"]),
                -float(item[2]["l2_regularization"]),
                float(item[2]["learning_rate"]),
            ),
        )
        self.model = self._base_model(self.best_params).fit(x, y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("predictor is not fitted")
        _require_columns(frame, set(self.feature_columns))
        return np.clip(self.model.predict(frame[self.feature_columns]), 0, self.n_nontrivial_lambdas)

    @property
    def model_id(self) -> str:
        if self.model is None or self.best_params is None:
            raise RuntimeError("predictor is not fitted")
        payload = {
            "alpha": self.alpha,
            "L": self.n_nontrivial_lambdas,
            "features": self.feature_columns,
            "params": self.best_params,
            "random_state": self.random_state,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("predictor is not fitted")
        joblib.dump(
            {
                "feature_columns": self.feature_columns,
                "alpha": self.alpha,
                "n_nontrivial_lambdas": self.n_nontrivial_lambdas,
                "random_state": self.random_state,
                "best_params": self.best_params,
                "cv_results": self.cv_results,
                "model": self.model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CriticalIndexPredictor":
        payload = joblib.load(path)
        obj = cls(
            payload["feature_columns"],
            alpha=payload["alpha"],
            n_nontrivial_lambdas=payload["n_nontrivial_lambdas"],
            random_state=payload["random_state"],
        )
        obj.best_params = payload["best_params"]
        obj.cv_results = payload["cv_results"]
        obj.model = payload["model"]
        return obj


# Old name retained solely to make stale imports fail with an actionable message.
class MetaRiskPredictor:
    def __init__(self, *args: object, **kwargs: object):
        raise RuntimeError(
            "MetaRiskPredictor/upper_risk is retired from the formal method; use CriticalIndexPredictor"
        )
