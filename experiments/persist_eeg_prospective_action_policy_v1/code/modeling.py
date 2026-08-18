from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import DIAGNOSTICS, FIGURES, REPO_ROOT, RESULTS, stable_seed, write_csv, write_json


ROUTER_ACTIONS = ("erase", "amplify", "geometry")
MODEL_NAMES = ("RidgeEffect", "ElasticNetEffect", "SmallHGBEffect")
BOOTSTRAP_MODELS = 12
RISK_LAMBDA = 1.645
RISK_TAU = 0.005


def _tokens(columns: Iterable[str], tokens: Iterable[str]) -> list[str]:
    return [column for column in columns if any(token in column for token in tokens)]


def feature_sets(frame: pd.DataFrame, family: str, scheme: str) -> dict[str, list[str] | None]:
    columns = [column for column in frame.columns if column.startswith("f_") and frame[column].notna().any()]
    if family == "openbmi_sample_router":
        persistence = _tokens(columns, ["protected_slot0", "protected_slot1"])
        decision = _tokens(
            columns,
            ["delta_p", "full_erase", "probability_shift", "agreement_full_erase", "contribution_norm"],
        )
        geometry = _tokens(
            columns,
            ["margin_geo", "entropy_geo", "base_geo", "projection", "midpoint", "prototype", "geometry_margin"],
        )
        competence = _tokens(columns, ["p_full_max", "entropy_full", "top2_margin", "nll_proxy", "logit_norm"])
        return {
            "P": persistence,
            "U": None,
            "D": decision,
            "P+U": None,
            "P+U+D": None,
            "geometry": geometry,
            "stability": None,
            "competence": competence,
            "P+U+D+geometry": None,
            "all_legal": columns,
        }
    persistence = _tokens(columns, ["persistence", "f_rank"])
    if family == "openbmi_dda_block":
        if scheme == "leave_one_outer_fold_out":
            utility = _tokens(columns, ["u_crossouterfold"])
            forbidden_u = "u_crossrun"
        else:
            utility = _tokens(columns, ["u_crossrun"])
            forbidden_u = "u_crossouterfold"
    elif scheme == "leave_one_backbone_out":
        utility = _tokens(columns, ["u_crossbackbone"])
        forbidden_u = "u_crossfold"
    else:
        utility = _tokens(columns, ["u_crossfold"])
        forbidden_u = "u_crossbackbone"
    decision = _tokens(columns, ["decision", "jacobian"])
    geometry = _tokens(
        columns,
        ["geometry", "eigenvalue", "positive_rank", "covariance_condition", "centroid", "principal_angle"],
    )
    stability = [column for column in columns if column.endswith("_sd") and forbidden_u not in column]
    competence = _tokens(columns, ["task_BA", "baseline_BA"])
    all_legal = [column for column in columns if forbidden_u not in column]
    return {
        "P": persistence,
        "U": utility,
        "D": decision,
        "P+U": sorted(set(persistence + utility)),
        "P+U+D": sorted(set(persistence + utility + decision)),
        "geometry": geometry,
        "stability": stability or None,
        "competence": competence or None,
        "P+U+D+geometry": sorted(set(persistence + utility + decision + geometry)),
        "all_legal": all_legal,
    }


def _estimator(name: str, seed: int, n_rows: int) -> Pipeline:
    if name == "RidgeEffect":
        model: Any = Ridge(alpha=10.0)
    elif name == "ElasticNetEffect":
        model = ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5000, random_state=seed)
    elif name == "SmallHGBEffect":
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=50 if n_rows > 1000 else 5,
            l2_regularization=1.0,
            random_state=seed,
        )
    else:
        raise KeyError(name)
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if name != "SmallHGBEffect":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _fit(model: Pipeline, x: pd.DataFrame, y: np.ndarray, weight: np.ndarray | None = None) -> Pipeline:
    kwargs: dict[str, Any] = {}
    if weight is not None and model.steps[-1][0] == "model":
        kwargs["model__sample_weight"] = weight
    try:
        model.fit(x, y, **kwargs)
    except TypeError:
        model.fit(x, y)
    return model


def _splits(frame: pd.DataFrame, family: str, scheme: str):
    if family == "openbmi_sample_router":
        groups = frame.subject_id.astype(str).to_numpy()
        splitter = GroupKFold(n_splits=5)
    elif family == "openbmi_dda_block":
        if scheme == "leave_one_outer_fold_out":
            groups = frame.fold_id.astype(str).to_numpy()
        else:
            groups = (frame.fold_id.astype(str) + "_" + frame.seed_id.astype(str)).to_numpy()
        splitter = LeaveOneGroupOut()
    elif family == "wbcic_development_block":
        groups = (
            frame.backbone_id.astype(str).to_numpy()
            if scheme == "leave_one_backbone_out"
            else frame.fold_id.astype(str).to_numpy()
        )
        splitter = LeaveOneGroupOut()
    else:
        raise KeyError(family)
    return groups, list(splitter.split(frame, groups=groups))


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    result = {
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "Spearman": float(spearmanr(y, pred).statistic) if len(np.unique(y)) > 1 else float("nan"),
        "sign_accuracy": float(np.mean((y > 0) == (pred > 0))),
    }
    positive = y >= 0.005
    if positive.any() and (~positive).any():
        result["AUROC_practical"] = float(roc_auc_score(positive, pred))
        result["AUPRC_practical"] = float(average_precision_score(positive, pred))
    else:
        result["AUROC_practical"] = float("nan")
        result["AUPRC_practical"] = float("nan")
    return result


def crossfit_predictions(
    frame: pd.DataFrame,
    family: str,
    scheme: str,
    features: list[str],
    model_name: str = "RidgeEffect",
) -> pd.DataFrame:
    groups, splits = _splits(frame, family, scheme)
    rows: list[pd.DataFrame] = []
    actions = ROUTER_ACTIONS if family == "openbmi_sample_router" else ("suppress",)
    for split_index, (train, test) in enumerate(splits):
        for action in actions:
            target = f"effect_{action}"
            y_train = frame.iloc[train][target].to_numpy(dtype=float)
            model = _estimator(model_name, stable_seed("learnability", family, scheme, model_name, split_index, action), len(train))
            weight = frame.iloc[train].unit_weight.fillna(1.0).to_numpy(dtype=float)
            _fit(model, frame.iloc[train][features], y_train, weight)
            pred = model.predict(frame.iloc[test][features])
            rows.append(
                pd.DataFrame(
                    {
                        "row_index": frame.index.to_numpy()[test],
                        "validation_group": groups[test],
                        "action": action,
                        "target": frame.iloc[test][target].to_numpy(dtype=float),
                        "prediction": pred,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def learnability_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    schemes = {
        "openbmi_sample_router": ["leave_one_subject_group_out"],
        "openbmi_dda_block": ["leave_one_run_out", "leave_one_outer_fold_out"],
        "wbcic_development_block": ["leave_one_fold_out", "leave_one_backbone_out"],
    }
    for family, family_schemes in schemes.items():
        frame = data[data.family_id == family].copy().reset_index(drop=True)
        for scheme in family_schemes:
            for feature_family, features in feature_sets(frame, family, scheme).items():
                base = {"family_id": family, "validation_scheme": scheme, "feature_family": feature_family}
                if not features:
                    rows.append({**base, "status": "NOT_AVAILABLE_WITHOUT_OUTCOME_LEAKAGE"})
                    continue
                predictions = crossfit_predictions(frame, family, scheme, features, "RidgeEffect")
                rows.append(
                    {
                        **base,
                        "status": "EVALUATED",
                        "n_units": len(predictions),
                        "n_features": len(features),
                        **_metrics(predictions.target.to_numpy(), predictions.prediction.to_numpy()),
                    }
                )
    result = pd.DataFrame(rows)
    write_csv(DIAGNOSTICS / "ACTIONABILITY_LEARNABILITY.csv", result)
    return result


def _association_frame(frame: pd.DataFrame, family: str, scheme: str) -> pd.DataFrame:
    features = feature_sets(frame, family, scheme)["all_legal"] or []
    if family == "openbmi_sample_router":
        parts: list[pd.DataFrame] = []
        for action in ROUTER_ACTIONS:
            agg = frame.groupby("subject_id", as_index=False)[features + [f"effect_{action}"]].mean(numeric_only=True)
            agg = agg.rename(columns={f"effect_{action}": "target"})
            agg["action"] = action
            agg["group"] = agg.subject_id.astype(str)
            parts.append(agg)
        return pd.concat(parts, ignore_index=True)
    group_columns = ["fold_id", "seed_id", "block_id"]
    if family == "wbcic_development_block":
        group_columns = ["backbone_id", "fold_id", "block_id"]
    agg = frame.groupby(group_columns, as_index=False)[features + ["effect_suppress"]].mean(numeric_only=True)
    agg = agg.rename(columns={"effect_suppress": "target"})
    agg["action"] = "suppress"
    agg["group"] = (
        agg.backbone_id.astype(str)
        if family == "wbcic_development_block" and scheme == "leave_one_backbone_out"
        else agg.fold_id.astype(str) + "_" + agg.seed_id.astype(str)
        if "seed_id" in agg
        else agg.fold_id.astype(str)
    )
    return agg


def _bootstrap_corr(frame: pd.DataFrame, feature: str, draws: int, seed: int) -> tuple[float, float]:
    groups = sorted(frame.group.astype(str).unique())
    by_group = {group: frame[frame.group.astype(str) == group] for group in groups}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([by_group[group] for group in sampled], ignore_index=True)
        if boot[feature].nunique() > 1 and boot.target.nunique() > 1:
            values.append(float(spearmanr(boot[feature], boot.target).statistic))
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (float("nan"), float("nan"))


def _permutation_p(frame: pd.DataFrame, feature: str, draws: int, seed: int) -> float:
    if frame[feature].nunique() < 2 or frame.target.nunique() < 2:
        return float("nan")
    observed = abs(float(spearmanr(frame[feature], frame.target).statistic))
    rng = np.random.default_rng(seed)
    count = 0
    target = frame.target.to_numpy(dtype=float)
    for _ in range(draws):
        permuted = rng.permutation(target)
        value = abs(float(spearmanr(frame[feature], permuted).statistic))
        count += value >= observed
    return float((count + 1) / (draws + 1))


def feature_associations(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    schemes = {
        "openbmi_sample_router": "leave_one_subject_group_out",
        "openbmi_dda_block": "leave_one_run_out",
        "wbcic_development_block": "leave_one_fold_out",
    }
    for family, scheme in schemes.items():
        source = data[data.family_id == family].copy().reset_index(drop=True)
        frame = _association_frame(source, family, scheme)
        features = feature_sets(source, family, scheme)["all_legal"] or []
        for action, action_frame in frame.groupby("action"):
            for feature in features:
                valid = action_frame[[feature, "target", "group"]].dropna()
                if len(valid) < 8 or valid[feature].nunique() < 2 or valid.target.nunique() < 2:
                    continue
                pearson = pearsonr(valid[feature], valid.target)
                spearman = spearmanr(valid[feature], valid.target)
                ci_l, ci_u = _bootstrap_corr(valid, feature, 300, stable_seed("assoc-boot", family, action, feature))
                rows.append(
                    {
                        "family_id": family,
                        "action": action,
                        "feature": feature,
                        "n_grouped_units": len(valid),
                        "pearson_r": float(pearson.statistic),
                        "pearson_p": float(pearson.pvalue),
                        "spearman_r": float(spearman.statistic),
                        "spearman_p": float(spearman.pvalue),
                        "spearman_group_bootstrap_CI_L": ci_l,
                        "spearman_group_bootstrap_CI_U": ci_u,
                        "grouped_permutation_p": _permutation_p(
                            valid, feature, 300, stable_seed("assoc-perm", family, action, feature)
                        ),
                    }
                )
    result = pd.DataFrame(rows).sort_values(["family_id", "grouped_permutation_p", "feature"])
    write_csv(DIAGNOSTICS / "FEATURE_ASSOCIATION.csv", result)
    return result


def _bootstrap_predictions(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    features: list[str],
    target: str,
    group_column: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted(train_frame[group_column].astype(str).unique())
    rng = np.random.default_rng(seed)
    predictions: list[np.ndarray] = []
    for draw in range(BOOTSTRAP_MODELS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        weight = train_frame[group_column].astype(str).map(counts).fillna(0).to_numpy(dtype=float)
        selected = weight > 0
        model = _estimator("RidgeEffect", stable_seed(seed, draw), int(selected.sum()))
        _fit(
            model,
            train_frame.loc[selected, features],
            train_frame.loc[selected, target].to_numpy(dtype=float),
            train_frame.loc[selected].unit_weight.fillna(1).to_numpy(dtype=float) * weight[selected],
        )
        predictions.append(model.predict(test_frame[features]))
    values = np.stack(predictions)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def _router_policy_metrics(test: pd.DataFrame, selected: np.ndarray) -> dict[str, float]:
    selected = np.asarray(selected, dtype=object)
    y = test.outcome_label.to_numpy(dtype=int)
    noop = test.pred_noop.to_numpy(dtype=int)
    pred = noop.copy()
    effects = np.zeros(len(test), dtype=float)
    for action in ROUTER_ACTIONS:
        mask = selected == action
        pred[mask] = test.loc[mask, f"pred_{action}"].to_numpy(dtype=int)
        effects[mask] = test.loc[mask, f"effect_{action}"].to_numpy(dtype=float)
    subject_deltas: list[float] = []
    for _, positions in test.groupby(["fold_id", "seed_id", "subject_id"]).indices.items():
        idx = np.asarray(positions, dtype=int)
        if len(np.unique(y[idx])) < 2:
            continue
        subject_deltas.append(
            float(balanced_accuracy_score(y[idx], pred[idx]) - balanced_accuracy_score(y[idx], noop[idx]))
        )
    intervened = selected != "noop"
    return {
        "mean_delta_BA": float(np.mean(subject_deltas)) if subject_deltas else float("nan"),
        "median_subject_delta_BA": float(np.median(subject_deltas)) if subject_deltas else float("nan"),
        "worst_subject_delta_BA": float(np.min(subject_deltas)) if subject_deltas else float("nan"),
        "positive_subject_fraction": float(np.mean(np.asarray(subject_deltas) > 0)) if subject_deltas else float("nan"),
        "action_rate": float(np.mean(intervened)),
        "unsafe_intervention_rate": float(np.mean(effects[intervened] < 0)) if intervened.any() else 0.0,
        "positive_intervention_precision": float(np.mean(effects[intervened] > 0)) if intervened.any() else 0.0,
        "mean_effect_when_intervene": float(np.mean(effects[intervened])) if intervened.any() else 0.0,
        "missed_opportunity_rate": float(np.mean((~intervened) & (test[[f"effect_{a}" for a in ROUTER_ACTIONS]].max(axis=1) > 0))),
    }


def _router_policy_cv(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = feature_sets(frame, "openbmi_sample_router", "leave_one_subject_group_out")["all_legal"] or []
    groups, splits = _splits(frame, "openbmi_sample_router", "leave_one_subject_group_out")
    group_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for split_index, (train_idx, test_idx) in enumerate(splits):
        train = frame.iloc[train_idx].reset_index(drop=True)
        test = frame.iloc[test_idx].reset_index(drop=True)
        validation_group = f"subject_fold_{split_index}"
        fixed_gain: dict[str, float] = {}
        for action in ROUTER_ACTIONS:
            metrics = _router_policy_metrics(train, np.full(len(train), action, dtype=object))
            fixed_gain[action] = metrics["mean_delta_BA"]
        best_fixed = max(fixed_gain, key=fixed_gain.get)
        selections: dict[str, np.ndarray] = {
            "M0_NO_OP": np.full(len(test), "noop", dtype=object),
            "BestFixedTrain": np.full(len(test), best_fixed if fixed_gain[best_fixed] > 0 else "noop", dtype=object),
            "AlwaysSuppress": np.full(len(test), "erase", dtype=object),
            "P_only_gate": np.full(len(test), "noop", dtype=object),
            "P_plus_U_gate": np.full(len(test), "noop", dtype=object),
            "P_plus_U_plus_D_gate": np.full(len(test), "noop", dtype=object),
        }
        predicted: dict[str, np.ndarray] = {}
        risk_mean: list[np.ndarray] = []
        risk_sd: list[np.ndarray] = []
        for model_name in MODEL_NAMES:
            action_prediction: list[np.ndarray] = []
            for action in ROUTER_ACTIONS:
                model = _estimator(model_name, stable_seed("router-policy", split_index, model_name, action), len(train))
                _fit(
                    model,
                    train[features],
                    train[f"effect_{action}"].to_numpy(dtype=float),
                    train.unit_weight.to_numpy(dtype=float),
                )
                action_prediction.append(model.predict(test[features]))
            values = np.stack(action_prediction, axis=1)
            predicted[model_name] = values
            choice = values.argmax(axis=1)
            best = values[np.arange(len(test)), choice]
            selections[model_name] = np.where(best > 0, np.asarray(ROUTER_ACTIONS, dtype=object)[choice], "noop")
        for action in ROUTER_ACTIONS:
            mean, sd = _bootstrap_predictions(
                train,
                test,
                features,
                f"effect_{action}",
                "subject_id",
                stable_seed("router-risk", split_index, action),
            )
            risk_mean.append(mean)
            risk_sd.append(sd)
        risk_mean_arr = np.stack(risk_mean, axis=1)
        risk_sd_arr = np.stack(risk_sd, axis=1)
        lcb = risk_mean_arr - RISK_LAMBDA * risk_sd_arr
        # ERASE suppresses a Certified Protected span and is therefore barred.
        lcb[:, 0] = -np.inf
        choice = lcb.argmax(axis=1)
        best = lcb[np.arange(len(test)), choice]
        selections["RiskAwarePERSIST"] = np.where(
            best > RISK_TAU, np.asarray(ROUTER_ACTIONS, dtype=object)[choice], "noop"
        )
        oracle_effects = test[[f"effect_{action}" for action in ROUTER_ACTIONS]].to_numpy(dtype=float)
        oracle_choice = oracle_effects.argmax(axis=1)
        oracle_best = oracle_effects[np.arange(len(test)), oracle_choice]
        selections["Oracle"] = np.where(
            oracle_best > 0, np.asarray(ROUTER_ACTIONS, dtype=object)[oracle_choice], "noop"
        )
        oracle_gain = _router_policy_metrics(test, selections["Oracle"])["mean_delta_BA"]
        for method, selected in selections.items():
            metrics = _router_policy_metrics(test, selected)
            group_rows.append(
                {
                    "family_id": "openbmi_sample_router",
                    "validation_scheme": "leave_one_subject_group_out",
                    "validation_group": validation_group,
                    "method": method,
                    **metrics,
                    "oracle_gain": oracle_gain,
                    "recovered_headroom": metrics["mean_delta_BA"] / oracle_gain if oracle_gain > 0 else np.nan,
                    "outer_test_used": False,
                }
            )
            effects = np.zeros(len(test), dtype=float)
            for action in ("noop", *ROUTER_ACTIONS):
                mask = selected == action
                if action != "noop":
                    effects[mask] = test.loc[mask, f"effect_{action}"].to_numpy(dtype=float)
                action_rows.append(
                    {
                        "family_id": "openbmi_sample_router",
                        "validation_scheme": "leave_one_subject_group_out",
                        "validation_group": validation_group,
                        "method": method,
                        "action": action.upper(),
                        "count": int(mask.sum()),
                        "mean_realized_effect": float(effects[mask].mean()) if mask.any() else np.nan,
                        "harm_count": int(np.sum(effects[mask] < 0)) if mask.any() else 0,
                        "rescue_count": int(np.sum(effects[mask] > 0)) if mask.any() else 0,
                    }
                )
    return group_rows, action_rows


def _block_metrics(test: pd.DataFrame, suppress: np.ndarray) -> dict[str, float]:
    suppress = np.asarray(suppress, dtype=bool)
    effect = test.effect_suppress.to_numpy(dtype=float)
    realized = np.where(suppress, effect, 0.0)
    return {
        "mean_delta_BA": float(realized.mean()),
        "median_subject_delta_BA": float(np.median(realized)),
        "worst_subject_delta_BA": float(realized.min()),
        "positive_subject_fraction": float(np.mean(realized > 0)),
        "action_rate": float(np.mean(suppress)),
        "unsafe_intervention_rate": float(np.mean(effect[suppress] < 0)) if suppress.any() else 0.0,
        "positive_intervention_precision": float(np.mean(effect[suppress] > 0)) if suppress.any() else 0.0,
        "mean_effect_when_intervene": float(effect[suppress].mean()) if suppress.any() else 0.0,
        "missed_opportunity_rate": float(np.mean((~suppress) & (effect > 0))),
    }


def _block_policy_cv(frame: pd.DataFrame, family: str, scheme: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sets = feature_sets(frame, family, scheme)
    features = sets["all_legal"] or []
    if family == "openbmi_dda_block":
        u_column = "f_u_crossouterfold" if scheme == "leave_one_outer_fold_out" else "f_u_crossrun"
    else:
        u_column = "f_u_crossbackbone" if scheme == "leave_one_backbone_out" else "f_u_crossfold"
    groups, splits = _splits(frame, family, scheme)
    group_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    group_column = "backbone_id" if scheme == "leave_one_backbone_out" else (
        "fold_id" if scheme == "leave_one_fold_out" or scheme == "leave_one_outer_fold_out" else "run_group"
    )
    working = frame.copy()
    if group_column == "run_group":
        working["run_group"] = working.fold_id.astype(str) + "_" + working.seed_id.astype(str)
    for split_index, (train_idx, test_idx) in enumerate(splits):
        train = working.iloc[train_idx].reset_index(drop=True)
        test = working.iloc[test_idx].reset_index(drop=True)
        validation_group = str(groups[test_idx][0])
        train_mean = float(train.effect_suppress.mean())
        persistence = test.f_persistence_strength.fillna(0).to_numpy(dtype=float) > 0
        harmful_u = test[u_column].fillna(0).to_numpy(dtype=float) < 0
        decision = test.f_decision_logit_ratio.fillna(0).to_numpy(dtype=float) > 1
        selections: dict[str, np.ndarray] = {
            "M0_NO_OP": np.zeros(len(test), dtype=bool),
            "BestFixedTrain": np.full(len(test), train_mean > 0, dtype=bool),
            "AlwaysSuppress": np.ones(len(test), dtype=bool),
            "P_only_gate": persistence,
            "P_plus_U_gate": persistence & harmful_u,
            "P_plus_U_plus_D_gate": persistence & harmful_u & decision,
        }
        for model_name in MODEL_NAMES:
            model = _estimator(model_name, stable_seed("block-policy", family, scheme, split_index, model_name), len(train))
            _fit(model, train[features], train.effect_suppress.to_numpy(dtype=float), train.unit_weight.to_numpy(dtype=float))
            pred = model.predict(test[features])
            selections[model_name] = pred > 0
        mean, sd = _bootstrap_predictions(
            train,
            test,
            features,
            "effect_suppress",
            group_column,
            stable_seed("block-risk", family, scheme, split_index),
        )
        protected = test[u_column].fillna(0).to_numpy(dtype=float) > 0
        selections["RiskAwarePERSIST"] = (mean - RISK_LAMBDA * sd > RISK_TAU) & (~protected)
        selections["Oracle"] = test.effect_suppress.to_numpy(dtype=float) > 0
        oracle_gain = _block_metrics(test, selections["Oracle"])["mean_delta_BA"]
        for method, selected in selections.items():
            metrics = _block_metrics(test, selected)
            group_rows.append(
                {
                    "family_id": family,
                    "validation_scheme": scheme,
                    "validation_group": validation_group,
                    "method": method,
                    **metrics,
                    "oracle_gain": oracle_gain,
                    "recovered_headroom": metrics["mean_delta_BA"] / oracle_gain if oracle_gain > 0 else np.nan,
                    "outer_test_used": False,
                }
            )
            effect = test.effect_suppress.to_numpy(dtype=float)
            for action, mask in (("SUPPRESS_BLOCK", selected), ("NO_OP", ~selected)):
                action_rows.append(
                    {
                        "family_id": family,
                        "validation_scheme": scheme,
                        "validation_group": validation_group,
                        "method": method,
                        "action": action,
                        "count": int(mask.sum()),
                        "mean_realized_effect": float(effect[mask].mean()) if mask.any() and action != "NO_OP" else 0.0,
                        "harm_count": int(np.sum(effect[mask] < 0)) if mask.any() and action != "NO_OP" else 0,
                        "rescue_count": int(np.sum(effect[mask] > 0)) if mask.any() and action != "NO_OP" else 0,
                    }
                )
    return group_rows, action_rows


def _bootstrap_group_mean(values: np.ndarray, seed: int, draws: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def _historical_router_rows() -> list[dict[str, Any]]:
    path = REPO_ROOT / "experiments" / "persist_eeg_router" / "outputs" / "R0" / "RUN_RESULTS" / "R0_RUN_RESULTS.csv"
    frame = pd.read_csv(path)
    selected = frame[(frame.variant == "full_protected") & (frame.config == "r0__lc0.1__w8__lr0.0003")]
    rows: list[dict[str, Any]] = []
    for _, item in selected.iterrows():
        rows.append(
            {
                "family_id": "openbmi_sample_router",
                "validation_scheme": "historical_subject_crossfit",
                "validation_group": f"fold-{int(item['fold'])}_seed-{int(item['seed'])}",
                "method": "OldLegalRouter_R0",
                "mean_delta_BA": float(item["mean_subject_delta_BA"]),
                "median_subject_delta_BA": np.nan,
                "worst_subject_delta_BA": np.nan,
                "positive_subject_fraction": float(item["positive_subjects"] / item["n_subjects"]),
                "action_rate": float(item["action_rate_abs_a_minus_1_gt_0.05"]),
                "unsafe_intervention_rate": float(
                    item["harm_count"] / max(item["harm_count"] + item["rescue_count"], 1)
                ),
                "positive_intervention_precision": float(
                    item["rescue_count"] / max(item["harm_count"] + item["rescue_count"], 1)
                ),
                "mean_effect_when_intervene": np.nan,
                "missed_opportunity_rate": np.nan,
                "oracle_gain": np.nan,
                "recovered_headroom": np.nan,
                "outer_test_used": False,
            }
        )
    return rows


def policy_model_ladder(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    group_rows, action_rows = _router_policy_cv(
        data[data.family_id == "openbmi_sample_router"].copy().reset_index(drop=True)
    )
    group_rows.extend(_historical_router_rows())
    for family, schemes in (
        ("openbmi_dda_block", ["leave_one_run_out", "leave_one_outer_fold_out"]),
        ("wbcic_development_block", ["leave_one_fold_out", "leave_one_backbone_out"]),
    ):
        frame = data[data.family_id == family].copy().reset_index(drop=True)
        for scheme in schemes:
            groups, actions = _block_policy_cv(frame, family, scheme)
            group_rows.extend(groups)
            action_rows.extend(actions)
    groups_frame = pd.DataFrame(group_rows)
    actions_frame = pd.DataFrame(action_rows)
    write_csv(RESULTS / "POLICY_GROUP_RESULTS.csv", groups_frame)
    write_csv(RESULTS / "POLICY_ACTION_RESULTS.csv", actions_frame)
    summary_rows: list[dict[str, Any]] = []
    for keys, group in groups_frame.groupby(["family_id", "validation_scheme", "method"], sort=True):
        family, scheme, method = keys
        ci_l, ci_u = _bootstrap_group_mean(
            group.mean_delta_BA.to_numpy(dtype=float), stable_seed("model-summary", family, scheme, method)
        )
        positive_contribution = group.mean_delta_BA.clip(lower=0)
        largest_share = (
            float(positive_contribution.max() / positive_contribution.sum()) if positive_contribution.sum() > 0 else np.nan
        )
        summary_rows.append(
            {
                "family_id": family,
                "validation_scheme": scheme,
                "method": method,
                "evaluation_groups": len(group),
                "mean_delta_BA": float(group.mean_delta_BA.mean()),
                "delta_BA_group_bootstrap_CI_L": ci_l,
                "delta_BA_group_bootstrap_CI_U": ci_u,
                "positive_group_fraction": float(np.mean(group.mean_delta_BA > 0)),
                "largest_positive_group_share": largest_share,
                "mean_action_rate": float(group.action_rate.mean()),
                "unsafe_intervention_rate": float(group.unsafe_intervention_rate.mean()),
                "positive_intervention_precision": float(group.positive_intervention_precision.mean()),
                "mean_recovered_headroom": float(group.recovered_headroom.mean()),
                "worst_group_delta_BA": float(group.mean_delta_BA.min()),
                "outer_test_used": False,
            }
        )
    summary = pd.DataFrame(summary_rows)
    write_csv(RESULTS / "MODEL_LADDER_RESULTS.csv", summary)
    risk = summary[summary.method == "RiskAwarePERSIST"].copy()
    risk["lambda"] = RISK_LAMBDA
    risk["tau"] = RISK_TAU
    risk["bootstrap_models"] = BOOTSTRAP_MODELS
    write_csv(RESULTS / "RISK_POLICY_RESULTS.csv", risk)
    recovery = summary[
        summary.method.isin(["BestFixedTrain", "OldLegalRouter_R0", *MODEL_NAMES, "RiskAwarePERSIST", "Oracle"])
    ].copy()
    write_csv(RESULTS / "ORACLE_RECOVERY_RESULTS.csv", recovery)
    eligible = risk.sort_values("mean_delta_BA", ascending=False)
    best = eligible.iloc[0].to_dict()
    same = summary[
        (summary.family_id == best["family_id"])
        & (summary.validation_scheme == best["validation_scheme"])
    ]
    fixed_row = same[same.method == "BestFixedTrain"]
    ridge_row = same[same.method == "RidgeEffect"]
    old_rows = summary[
        (summary.family_id == best["family_id"]) & (summary.method == "OldLegalRouter_R0")
    ]
    fixed_gain = float(fixed_row.mean_delta_BA.iloc[0]) if len(fixed_row) else 0.0
    old_gain = float(old_rows.mean_delta_BA.mean()) if len(old_rows) else float("nan")
    ridge_unsafe = float(ridge_row.unsafe_intervention_rate.iloc[0]) if len(ridge_row) else float("nan")
    beats_fixed = bool(best["mean_delta_BA"] > fixed_gain)
    beats_old = bool(pd.isna(old_gain) or best["mean_delta_BA"] > old_gain)
    safer_than_nonconservative = bool(pd.isna(ridge_unsafe) or best["unsafe_intervention_rate"] < ridge_unsafe)
    ready = bool(
        best["mean_delta_BA"] >= 0.005
        and best["delta_BA_group_bootstrap_CI_L"] > 0
        and best["positive_group_fraction"] >= 0.60
        and (pd.isna(best["largest_positive_group_share"]) or best["largest_positive_group_share"] <= 0.50)
        and beats_fixed
        and beats_old
        and safer_than_nonconservative
        and best["mean_recovered_headroom"] > 0.10
    )
    promising = bool(
        best["mean_delta_BA"] > 0
        and best["positive_group_fraction"] >= 0.50
        and best["unsafe_intervention_rate"] < 0.50
    )
    candidate = {
        "status": "READY_FOR_PROSPECTIVE_POLICY_FREEZE" if ready else "PROMISING_PROSPECTIVE_POLICY" if promising else "NO_POLICY_CANDIDATE",
        "best_risk_policy": best,
        "comparison_gates": {
            "best_fixed_gain": fixed_gain,
            "historical_router_gain": old_gain,
            "ridge_unsafe_intervention_rate": ridge_unsafe,
            "beats_best_fixed": beats_fixed,
            "beats_historical_router": beats_old,
            "safer_than_nonconservative_ridge": safer_than_nonconservative,
        },
        "fixed_policy": {"lambda": RISK_LAMBDA, "tau": RISK_TAU, "bootstrap_models": BOOTSTRAP_MODELS},
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(RESULTS / "FINAL_POLICY_CANDIDATE.json", candidate)
    _policy_figures(summary)
    return summary, groups_frame, actions_frame, candidate


def _policy_figures(summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected = summary[
        (summary.validation_scheme.isin(["leave_one_subject_group_out", "leave_one_run_out", "leave_one_fold_out"]))
        & summary.method.isin(["M0_NO_OP", "BestFixedTrain", "P_only_gate", "P_plus_U_gate", "P_plus_U_plus_D_gate", "RidgeEffect", "ElasticNetEffect", "SmallHGBEffect", "RiskAwarePERSIST", "Oracle"])
    ].copy()
    methods = list(dict.fromkeys(selected.method.tolist()))
    families = list(dict.fromkeys(selected.family_id.tolist()))
    fig, axes = plt.subplots(len(families), 1, figsize=(11, 3.2 * len(families)), squeeze=False)
    for ax, family in zip(axes[:, 0], families):
        block = selected[selected.family_id == family].set_index("method")
        values = [float(block.loc[method, "mean_delta_BA"]) if method in block.index else np.nan for method in methods]
        ax.bar(np.arange(len(methods)), values, color=["#4f7fa3" if value >= 0 else "#b45f5f" for value in values])
        ax.axhline(0, color="#555", linewidth=1)
        ax.set_title(family)
        ax.set_ylabel("Grouped ΔBA")
        ax.set_xticks(np.arange(len(methods)), methods, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure6_policy_comparison.png", dpi=240)
    fig.savefig(FIGURES / "figure6_policy_comparison.pdf")
    plt.close(fig)

    risk = summary[summary.method == "RiskAwarePERSIST"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for family, group in risk.groupby("family_id"):
        ax.scatter(group.unsafe_intervention_rate, group.mean_delta_BA, s=70, label=family)
    ax.axhline(0, color="#555", linewidth=1)
    ax.set_xlabel("Unsafe intervention rate")
    ax.set_ylabel("Grouped mean ΔBA")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure7_risk_vs_gain.png", dpi=240)
    fig.savefig(FIGURES / "figure7_risk_vs_gain.pdf")
    plt.close(fig)


def write_learnability_report(learnability: pd.DataFrame, associations: pd.DataFrame) -> None:
    evaluated = learnability[learnability.status == "EVALUATED"].copy()
    best = evaluated.sort_values(["family_id", "validation_scheme", "R2"], ascending=[True, True, False]).groupby(
        ["family_id", "validation_scheme"], as_index=False
    ).first()
    lines = ["# Actionability learnability", "", "All metrics are grouped out-of-sample; no random row split is used.", ""]
    for row in best.itertuples(index=False):
        lines.append(
            f"- `{row.family_id}` / `{row.validation_scheme}`: best `{row.feature_family}`, "
            f"R2={row.R2:.4f}, RMSE={row.RMSE:.4f}, Spearman={row.Spearman:.4f}."
        )
    lines.extend(
        [
            "",
            "`U` is marked unavailable for the historical sample router because no utility estimate independent of the target trial outcome exists. Filling it with realised rescue/harm would be leakage.",
            "",
            "Same-cell DDA/WBCIC signed utility is excluded. Only explicitly cross-fitted utility priors enter the corresponding legal feature set.",
            "",
            "`OUTER_TEST_USED = false`.",
        ]
    )
    (DIAGNOSTICS / "LEARNABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_modeling(data: pd.DataFrame) -> dict[str, Any]:
    learnability = learnability_diagnostics(data)
    associations = feature_associations(data)
    write_learnability_report(learnability, associations)
    summary, groups, actions, candidate = policy_model_ladder(data)
    return {
        "learnability_rows": len(learnability),
        "association_rows": len(associations),
        "model_rows": len(summary),
        "candidate": candidate,
    }


if __name__ == "__main__":
    from common import DATA

    frame = pd.read_csv(DATA / "ACTION_OUTCOME_DATASET.csv", low_memory=False)
    print(json.dumps(run_modeling(frame), indent=2, default=str))
