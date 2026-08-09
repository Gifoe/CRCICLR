from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .policy_certificate import calibrate_policy_index, joint_critical_index
from .policy_search import select_thresholds
from .probe_policy import ProbePolicy


def _outcomes(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows=[]
    for _,row in frame.iterrows():
        action_j=next(j for j in range(21) if row[f"risk_j{j}"]<=alpha); source_j=next(j for j in range(21) if row[f"source_risk_j{j}"]<=alpha)
        rows.append({"dataset":row.dataset,"seed":row.seed,"subject_id":row.subject_id,"action":row.action,"alpha":alpha,
            "action_available":row.action_available,"source_safe_size":row[f"source_size_j{source_j}"],"safe_size":row[f"size_j{action_j}"],
            "oracle_gain":row[f"source_size_j{source_j}"]-row[f"size_j{action_j}"],"source_argmax_error":row.source_argmax_error,
            "argmax_error":row.argmax_error,"classification_degradation":row.degradation})
    return pd.DataFrame(rows)


def _policy_action(frame: pd.DataFrame, thresholds, variant: str) -> str:
    if variant=="g_set_only":
        accepted=frame[frame.g_set>=thresholds.tau_set]; return "no_tta" if accepted.empty else str(accepted.sort_values(["g_set","action"],ascending=[False,True]).iloc[0].action)
    return ProbePolicy(thresholds).decide(frame)["selected_action"]


def evaluate_ablation(surfaces: pd.DataFrame, policy_grid: dict[str,object], alpha: float, epsilon: float, delta: float,
                      variant: str, calibration_limit: int | None = None, actions: list[str] | None = None) -> pd.DataFrame:
    rows=[]
    for (dataset,seed,fold),current in surfaces.groupby(["dataset","seed","outer_fold"]):
        if actions is not None: current=current[current.action.isin(actions)]
        meta=current[current.role=="meta_fit_subjects"]; cal=current[current.role=="calibration_subjects"]; outer=current[current.role=="outer_evaluation_subjects"]
        thresholds,_=select_thresholds(meta,_outcomes(meta,alpha),policy_grid,epsilon)
        if variant=="without_augmentation_consistency": thresholds=replace(thresholds,tau_aug_margin=float("inf"))
        elif variant=="without_temporal_stability": thresholds=replace(thresholds,tau_positive_blocks=0.0,tau_time_mad=float("inf"))
        elif variant=="without_source_drift": thresholds=replace(thresholds,tau_drift=float("inf"))
        elif variant=="without_collapse_gate": thresholds=replace(thresholds,tau_class=0.0)
        elif variant=="without_update_gate": thresholds=replace(thresholds,tau_update=float("inf"))
        selected_cal=sorted(cal.subject_id.unique())[:calibration_limit] if calibration_limit else sorted(cal.subject_id.unique())
        indices=[]
        for subject in selected_cal:
            frame=cal[cal.subject_id==subject]; action=_policy_action(frame,thresholds,variant); row=frame.iloc[0] if action=="no_tta" else frame[frame.action==action].iloc[0]
            risks=np.asarray([row[f"source_risk_j{j}"] if action=="no_tta" else row[f"risk_j{j}"] for j in range(21)])
            degradation=0.0 if action=="no_tta" or variant=="risk_only_without_noninferiority" else float(row.degradation)
            indices.append(joint_critical_index(risks,degradation,alpha=alpha,epsilon=epsilon,sentinel_index=20))
        calibrated=calibrate_policy_index(np.asarray(indices),delta=delta,sentinel_index=20)
        for subject,frame in outer.groupby("subject_id"):
            action=_policy_action(frame,thresholds,variant); row=frame.iloc[0] if action=="no_tta" else frame[frame.action==action].iloc[0]
            prefix="source_" if action=="no_tta" else "";j=calibrated.lambda_index;risk=float(row[f"{prefix}risk_j{j}"]);size=float(row[f"{prefix}size_j{j}"]);degradation=0.0 if action=="no_tta" else float(row.degradation)
            rows.append({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"subject_id":subject,"ablation":variant,
                "average_set_size":size,"future_risk":risk,"degradation":degradation,"joint_violation":risk>alpha or degradation>epsilon,
                "intervention":action!="no_tta","sentinel":j==20,"calibration_size":len(selected_cal),"number_actions":len(frame.action.unique())})
    return pd.DataFrame(rows)

