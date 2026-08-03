from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hsc_tta.v2.joint_certificate import finite_sample_quantile


ABLATIONS = (
    "A1_risk_only",
    "A2_utility_only_uncertified",
    "A3_joint_without_benefit_certificate",
    "A4_separate_risk_gain_calibration",
    "A5_pointwise_per_action_calibration",
    "A6_no_action_specific_diagnostics",
    "A7_context_set_size_selector",
    "A8_global_critical_index",
    "A9_no_positive_gain_gate",
    "A10_policy_level_calibration",
)

IDENTIFIERS = {"dataset", "seed", "subject_id", "alpha", "action_available"}


def _q(values: pd.Series) -> float:
    return float(finite_sample_quantile(values.to_numpy(float), 0.1)[0])


def _point_action(group: pd.DataFrame) -> str:
    candidates = group[(group.available) & (group.action != "no_tta") &
                       (group.predicted_critical_index < 20) & (group.predicted_benefit > 0)]
    if candidates.empty:
        return "no_tta"
    return str(candidates.sort_values(["predicted_benefit", "predicted_critical_index", "action"],
                                      ascending=[False, True, True]).iloc[0].action)


def _select(group: pd.DataFrame, ablation: str) -> tuple[str, int, float]:
    available = group[group.available].copy()
    no_tta = available[available.action == "no_tta"].iloc[0]
    tta = available[available.action != "no_tta"].copy()
    if ablation == "A2_utility_only_uncertified":
        tta = tta[tta.predicted_benefit > 0]
        if tta.empty:
            return "no_tta", int(np.clip(np.ceil(no_tta.predicted_critical_index), 0, 20)), 0.0
        row = tta.sort_values(["predicted_benefit", "predicted_critical_index", "action"],
                              ascending=[False, True, True]).iloc[0]
        return str(row.action), int(np.clip(np.ceil(row.predicted_critical_index), 0, 20)), float(row.predicted_benefit)
    critical_column = "separate_certified_critical_index" if ablation == "A4_separate_risk_gain_calibration" else "ablation_index"
    benefit_column = "separate_benefit_lower" if ablation == "A4_separate_risk_gain_calibration" else "ablation_lower"
    safe = tta[tta[critical_column] < 20].copy()
    if ablation == "A1_risk_only":
        if safe.empty:
            return "no_tta", int(no_tta[critical_column]), 0.0
        row = safe.sort_values([critical_column, "action"]).iloc[0]
    elif ablation == "A3_joint_without_benefit_certificate":
        safe = safe[safe.predicted_benefit > 0]
        if safe.empty:
            return "no_tta", int(no_tta[critical_column]), 0.0
        row = safe.sort_values(["predicted_benefit", critical_column, "action"], ascending=[False, True, True]).iloc[0]
    elif ablation == "A7_context_set_size_selector":
        safe = safe[safe[benefit_column] > 0]
        if safe.empty:
            return "no_tta", int(no_tta[critical_column]), 0.0
        safe["context_size_at_index"] = [r[f"context_set_size_j{int(r[critical_column])}"] for _, r in safe.iterrows()]
        row = safe.sort_values(["context_size_at_index", benefit_column, "action"], ascending=[True, False, True]).iloc[0]
    elif ablation == "A9_no_positive_gain_gate":
        if safe.empty:
            return "no_tta", int(no_tta[critical_column]), 0.0
        row = safe.sort_values([benefit_column, critical_column, "action"], ascending=[False, True, True]).iloc[0]
    else:
        safe = safe[safe[benefit_column] > 0]
        if safe.empty:
            return "no_tta", int(no_tta[critical_column]), 0.0
        row = safe.sort_values([benefit_column, critical_column, "action"], ascending=[False, True, True]).iloc[0]
    return str(row.action), int(row[critical_column]), float(row[benefit_column])


def _evaluate(selected: pd.DataFrame, alpha: float) -> dict[str, float]:
    tta = selected.action != "no_tta"
    certified = selected.selected_index < 20
    risks = np.asarray([r[f"risk_j{int(r.selected_index)}"] for _, r in selected.iterrows()])
    sizes = np.asarray([r[f"set_size_j{int(r.selected_index)}"] for _, r in selected.iterrows()])
    singleton = np.asarray([r[f"singleton_j{int(r.selected_index)}"] for _, r in selected.iterrows()])
    return {
        "marginal_violation": float(np.mean(risks > alpha)),
        "certified_only_violation": float(np.mean(risks[certified] > alpha)) if certified.any() else np.nan,
        "nonharm_violation": float(np.mean(tta & (selected.true_benefit < 0))),
        "joint_validity": float(np.mean((risks <= alpha) & (~tta | (selected.true_benefit >= 0)))),
        "csr": float(certified.mean()),
        "full_set_fallback": float((~certified).mean()),
        "average_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "argmax_error": float(selected.argmax_error.mean()),
        "macro_f1": float(selected.macro_f1.mean()),
        "balanced_accuracy": float(selected.balanced_accuracy.mean()),
        "cohen_kappa": float(selected.cohen_kappa.mean()),
        "selected_vs_no_tta_gain": float(selected.true_benefit.mean()),
        "tta_selection_rate": float(tta.mean()),
        "safe_beneficial_selection_precision": float(np.mean((risks[tta] <= alpha) & (selected.loc[tta, "true_benefit"] > 0))) if tta.any() else np.nan,
        "nar": float(np.mean(selected.loc[tta, "true_benefit"] < 0)) if tta.any() else np.nan,
    }


def run_ablations(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    base = root / "outputs/v2_joint_certified"
    features = pd.read_parquet(base / "actions/DEVELOPMENT_CONTEXT_FEATURES.parquet")
    outcomes = pd.read_parquet(base / "actions/DEVELOPMENT_ACTION_SURFACE.parquet")
    bounds = pd.read_parquet(base / "nested_dev/ALL_DEV_JOINT_BOUNDS.parquet")
    feature_columns = [c for c in features.columns if c not in IDENTIFIERS]
    rows: list[dict[str, object]] = []
    for keys, ev_bounds in bounds.groupby(["dataset", "seed", "outer_fold", "alpha"]):
        dataset, seed, fold, alpha = keys
        split = json.loads((root / "data/splits_v2_dev" / dataset / f"seed_{seed}" / f"outer_fold_{fold}.json").read_text())
        meta_ids, cal_ids = set(split["meta_fit_subjects"]), set(split["calibration_subjects"])
        sf = features[(features.dataset == dataset) & (features.seed == seed)]
        so = outcomes[(outcomes.dataset == dataset) & (outcomes.seed == seed) & np.isclose(outcomes.alpha, alpha)]
        cal = sf[sf.subject_id.isin(cal_ids)].merge(
            so[so.subject_id.isin(cal_ids)][["subject_id", "action", "true_critical_index", "true_benefit"]],
            on=["subject_id", "action"], validate="one_to_one")
        model_dir = base / "nested_dev/models" / dataset / f"seed_{seed}" / f"fold_{fold}"
        risk_model = joblib.load(model_dir / f"risk_alpha_{alpha:.2f}.joblib")
        benefit_model = joblib.load(model_dir / "benefit.joblib")
        cal["predicted_critical_index"] = risk_model.predict(cal[feature_columns])
        cal["predicted_benefit"] = benefit_model.predict(cal[feature_columns])
        cal.loc[cal.action == "no_tta", ["true_benefit", "predicted_benefit"]] = 0.0
        c_j, c_delta = float(ev_bounds.c_j.iloc[0]), float(ev_bounds.c_delta.iloc[0])
        cal["risk_score"] = (cal.true_critical_index - cal.predicted_critical_index) / c_j
        cal["benefit_score"] = np.where(cal.action == "no_tta", -np.inf,
                                         (cal.predicted_benefit - cal.true_benefit) / c_delta)
        point_q = cal.groupby("action").apply(
            lambda g: _q(pd.concat([g.risk_score, g.benefit_score.replace(-np.inf, np.nan).dropna()])),
            include_groups=False).to_dict()
        ev = ev_bounds.merge(sf, on=["dataset", "seed", "subject_id", "action"], validate="one_to_one",
                             suffixes=("", "_feature"))
        ev["ablation_index"] = ev.certified_critical_index
        ev["ablation_lower"] = ev.benefit_lower

        frames: dict[str, pd.DataFrame] = {}
        for name in ("A1_risk_only", "A2_utility_only_uncertified", "A3_joint_without_benefit_certificate",
                     "A4_separate_risk_gain_calibration", "A7_context_set_size_selector", "A9_no_positive_gain_gate"):
            chosen = []
            for subject, group in ev.groupby("subject_id"):
                action, index, lower = _select(group, name)
                chosen.append({"subject_id": subject, "action": action, "selected_index": index, "selected_lower": lower})
            frames[name] = pd.DataFrame(chosen)

        point = ev.copy()
        point["ablation_index"] = [int(np.clip(np.ceil(r.predicted_critical_index + point_q[r.action] * c_j), 0, 20)) for r in point.itertuples()]
        point["ablation_lower"] = [float(r.predicted_benefit - point_q[r.action] * c_delta) for r in point.itertuples()]
        frames["A5_pointwise_per_action_calibration"] = pd.DataFrame(
            [{"subject_id": s, "action": a, "selected_index": j, "selected_lower": l}
             for s, g in point.groupby("subject_id") for a, j, l in [_select(g, "A5_pointwise_per_action_calibration")]])

        meta = so[so.subject_id.isin(meta_ids)]
        mean_j = meta.groupby("action").true_critical_index.mean().to_dict()
        mean_b = meta.groupby("action").true_benefit.mean().to_dict()
        cal6 = cal.copy()
        cal6["predicted_critical_index"] = cal6.action.map(mean_j)
        cal6["predicted_benefit"] = cal6.action.map(mean_b)
        scale_j = max(float(np.median(np.abs(cal6.true_critical_index - cal6.predicted_critical_index))), 1.0)
        scale_b = max(float(np.median(np.abs(cal6.true_benefit - cal6.predicted_benefit))), 1e-3)
        cal6["score"] = np.maximum((cal6.true_critical_index - cal6.predicted_critical_index) / scale_j,
                                    np.where(cal6.action == "no_tta", -np.inf,
                                             (cal6.predicted_benefit - cal6.true_benefit) / scale_b))
        q6 = _q(cal6.groupby("subject_id").score.max())
        no_diag = ev.copy()
        no_diag["predicted_critical_index"] = no_diag.action.map(mean_j)
        no_diag["predicted_benefit"] = no_diag.action.map(mean_b)
        no_diag["ablation_index"] = np.clip(np.ceil(no_diag.predicted_critical_index + q6 * scale_j), 0, 20).astype(int)
        no_diag["ablation_lower"] = no_diag.predicted_benefit - q6 * scale_b
        frames["A6_no_action_specific_diagnostics"] = pd.DataFrame(
            [{"subject_id": s, "action": a, "selected_index": j, "selected_lower": l}
             for s, g in no_diag.groupby("subject_id") for a, j, l in [_select(g, "A6_no_action_specific_diagnostics")]])

        global_index = int(min(20, math.ceil(np.quantile(cal.true_critical_index, 1.0))))
        global_frame = frames["A3_joint_without_benefit_certificate"].copy()
        global_frame["selected_index"] = global_index
        frames["A8_global_critical_index"] = global_frame

        cal_actions = {s: _point_action(g.assign(available=g.action_available)) for s, g in cal.groupby("subject_id")}
        cal_policy = cal[cal.apply(lambda r: cal_actions[r.subject_id] == r.action, axis=1)]
        q10 = _q(pd.Series(np.maximum(cal_policy.risk_score, cal_policy.benefit_score)))
        policy = ev.copy()
        policy_actions = {subject: _point_action(group) for subject, group in policy.groupby("subject_id")}
        policy["preselected"] = policy.subject_id.map(policy_actions)
        policy = policy[policy.action == policy.preselected].copy()
        policy["selected_index"] = np.clip(np.ceil(policy.predicted_critical_index + q10 * c_j), 0, 20).astype(int)
        policy["selected_lower"] = policy.predicted_benefit - q10 * c_delta
        frames["A10_policy_level_calibration"] = policy[["subject_id", "action", "selected_index", "selected_lower"]]

        for name, chosen in frames.items():
            selected = chosen.merge(so, on=["subject_id", "action"], validate="one_to_one")
            for metric, value in _evaluate(selected, float(alpha)).items():
                rows.append({"dataset": dataset, "seed": seed, "outer_fold": fold, "alpha": alpha,
                             "ablation": name, "metric": metric, "value": value})
    result = pd.DataFrame(rows)
    out = base / "ablations"
    out.mkdir(exist_ok=True)
    result.to_csv(out / "ABLATION_RESULTS.csv", index=False)
    summary = result.groupby(["dataset", "alpha", "ablation", "metric"]).value.mean().reset_index()
    lines = ["# HSC-TTA v2 ablation audit", "",
             "All rows use the same nested subject splits. A2 is explicitly uncertified; A5 and A10 alter the calibration unit and therefore do not inherit the proposed simultaneous post-selection theorem.", ""]
    for dataset in ("hmc", "eegmmidb"):
        lines.extend([f"## {dataset.upper()}", "", "| Ablation | alpha | Joint validity | CSR | TTA rate | Gain |", "|---|---:|---:|---:|---:|---:|"])
        pivot = summary[summary.dataset == dataset].pivot_table(index=["ablation", "alpha"], columns="metric", values="value")
        for (name, alpha_value), row in pivot.iterrows():
            lines.append(f"| {name} | {alpha_value:.2f} | {row.get('joint_validity', np.nan):.3f} | {row.get('csr', np.nan):.3f} | {row.get('tta_selection_rate', np.nan):.3f} | {row.get('selected_vs_no_tta_gain', np.nan):.4f} |")
        lines.append("")
    (out / "ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result
