from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GENERIC_COLUMNS = [
    "history_base_ce",
    "history_base_ba",
    "history_margin_mean",
    "history_margin_std",
    "history_entropy_mean",
    "history_prototype_separation",
    "history_gradient_norm",
    "history_session_drift",
    "history_samples",
    "update_rms",
    "update_mean_abs",
    "history_component_ce_gain",
    "history_component_ba_gain",
    "split_update_disagreement",
]
PERSIST_COLUMNS = [
    "P_persistence",
    "U_signed_utility_prior",
    "D_decision_dependence",
    "G_task_overlap",
    "R_history_transfer",
]


@dataclass(frozen=True)
class ControllerConfig:
    controller_id: str
    family: str
    parameter: float


CONTROLLERS = (
    ControllerConfig("RIDGE_A1", "ridge", 1.0),
    ControllerConfig("RIDGE_A10", "ridge", 10.0),
    ControllerConfig("EXTRA_TREES_D4", "extra_trees", 4.0),
)


def _utility_prior(train: pd.DataFrame, target: pd.DataFrame, leave_subject_out: bool) -> np.ndarray:
    means = train.groupby("component_id").future_ce_gain.mean()
    if not leave_subject_out:
        return target.component_id.map(means).fillna(float(train.future_ce_gain.mean())).to_numpy(float)
    totals = train.groupby("component_id").future_ce_gain.transform("sum").to_numpy(float)
    counts = train.groupby("component_id").future_ce_gain.transform("count").to_numpy(float)
    values = np.divide(
        totals - train.future_ce_gain.to_numpy(float),
        np.maximum(counts - 1.0, 1.0),
    )
    return values


def _matrix(
    frame: pd.DataFrame,
    persist: bool,
    component_levels: list[str],
    utility_prior: np.ndarray,
) -> np.ndarray:
    numeric = frame[GENERIC_COLUMNS].to_numpy(float)
    persist_values = frame[PERSIST_COLUMNS].to_numpy(float).copy()
    persist_values[:, 1] = np.asarray(utility_prior, dtype=float)
    if not persist:
        persist_values[:] = 0.0
    component = np.zeros((len(frame), len(component_levels)), dtype=float)
    mapping = {value: index for index, value in enumerate(component_levels)}
    for row, value in enumerate(frame.component_id.astype(str)):
        component[row, mapping[value]] = 1.0
    result = np.concatenate([numeric, persist_values, component], axis=1)
    result[~np.isfinite(result)] = 0.0
    return result


def _model(configuration: ControllerConfig, seed: int):
    if configuration.family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(configuration.parameter)))
    if configuration.family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=256,
            max_depth=int(configuration.parameter),
            min_samples_leaf=8,
            max_features=0.8,
            random_state=int(seed),
            n_jobs=-1,
        )
    raise ValueError(configuration)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return _correlation(pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy())


def utility_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame.future_ce_gain.to_numpy(float)
    prediction = frame.predicted_utility.to_numpy(float)
    return {
        "utility_R2": float(r2_score(target, prediction)),
        "utility_pearson": _correlation(target, prediction),
        "utility_spearman": _rank_correlation(target, prediction),
        "utility_sign_accuracy": float(np.mean((target > 0.0) == (prediction > 0.0))),
        "positive_target_fraction": float(np.mean(target > 0.0)),
        "rows": int(len(frame)),
        "subjects": int(frame.subject_id.nunique()),
    }


def crossfit_predict(
    rows: pd.DataFrame,
    persist: bool,
    configuration: ControllerConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = rows.copy().reset_index(drop=True)
    levels = sorted(frame.component_id.astype(str).unique())
    groups = frame.subject_id.astype(str).to_numpy()
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    prediction = np.empty(len(frame), dtype=float)
    uncertainty = np.empty(len(frame), dtype=float)
    for split_index, (fit_index, validation_index) in enumerate(splitter.split(frame, groups=groups)):
        fit = frame.iloc[fit_index].copy()
        validation = frame.iloc[validation_index].copy()
        train_prior = _utility_prior(fit, fit, leave_subject_out=True)
        validation_prior = _utility_prior(fit, validation, leave_subject_out=False)
        x_fit = _matrix(fit, persist, levels, train_prior)
        x_validation = _matrix(validation, persist, levels, validation_prior)
        model = _model(configuration, seed + split_index)
        target = fit.future_ce_gain.to_numpy(float)
        model.fit(x_fit, target)
        prediction[validation_index] = model.predict(x_validation)
        component_std = fit.groupby("component_id").future_ce_gain.std(ddof=1)
        fallback = float(np.std(target, ddof=1))
        uncertainty[validation_index] = validation.component_id.map(component_std).fillna(fallback).fillna(0.0).to_numpy(float)
    frame["predicted_utility"] = prediction
    frame["predicted_sigma"] = uncertainty
    metrics = utility_metrics(frame)
    metrics.update({
        "controller_id": configuration.controller_id,
        "mode": "PERSIST_META" if persist else "META_GENERIC",
        "cross_fitted": True,
    })
    return frame, metrics


def fit_predict(
    train_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    persist: bool,
    configuration: ControllerConfig,
    seed: int,
) -> pd.DataFrame:
    train = train_rows.copy().reset_index(drop=True)
    target = target_rows.copy().reset_index(drop=True)
    levels = sorted(train.component_id.astype(str).unique())
    if set(target.component_id.astype(str)) != set(levels):
        raise RuntimeError("Controller component mismatch")
    train_prior = _utility_prior(train, train, leave_subject_out=True)
    target_prior = _utility_prior(train, target, leave_subject_out=False)
    x_train = _matrix(train, persist, levels, train_prior)
    x_target = _matrix(target, persist, levels, target_prior)
    model = _model(configuration, seed)
    values = train.future_ce_gain.to_numpy(float)
    model.fit(x_train, values)
    target["predicted_utility"] = model.predict(x_target)
    component_std = train.groupby("component_id").future_ce_gain.std(ddof=1)
    fallback = float(np.std(values, ddof=1))
    target["predicted_sigma"] = target.component_id.map(component_std).fillna(fallback).fillna(0.0).to_numpy(float)
    return target
