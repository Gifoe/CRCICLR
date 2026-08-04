from __future__ import annotations

import numpy as np
import pandas as pd

from .policy_certificate import calibrate_policy_index, joint_critical_index


def _point(row: pd.Series, action: str, alpha: float) -> tuple[int,float,float,float]:
    prefix="source_" if action=="no_tta" else ""
    for j in range(21):
        if float(row[f"{prefix}risk_j{j}"])<=alpha:
            size=float(row[f"{prefix}size_j{j}"]); risk=float(row[f"{prefix}risk_j{j}"])
            return j,size,risk,float(row.source_argmax_error if action=="no_tta" else row.argmax_error)
    raise AssertionError("sentinel must be valid")


def _decide(frame: pd.DataFrame, policy: str, parameter: object) -> str:
    if policy=="best_fixed_action_crc": return str(parameter)
    if policy=="entropy_gate_crc":
        eligible=frame[frame.entropy_change<=float(parameter)]; return "no_tta" if eligible.empty else str(eligible.sort_values(["entropy_change","action"]).iloc[0].action)
    if policy=="agreement_gate_crc":
        eligible=frame[frame.prediction_agreement>=float(parameter)]; return "no_tta" if eligible.empty else str(eligible.sort_values(["prediction_agreement","action"],ascending=[False,True]).iloc[0].action)
    if policy in ("probe_set_gain_only_crc","v2_actionwise_joint"):
        eligible=frame[frame.g_set>=float(parameter)]; return "no_tta" if eligible.empty else str(eligible.sort_values(["g_set","d_src","action"],ascending=[False,True,True]).iloc[0].action)
    raise ValueError(policy)


def _fit_rule(meta: pd.DataFrame, policy: str, alpha: float, epsilon: float) -> object:
    if policy=="best_fixed_action_crc":
        candidates=sorted(meta.action.unique())
    elif policy=="entropy_gate_crc": candidates=sorted(set(float(meta.entropy_change.quantile(q)) for q in [0,.25,.5,.75,.9,1]))
    elif policy=="agreement_gate_crc": candidates=sorted(set(float(meta.prediction_agreement.quantile(q)) for q in [0,.25,.5,.75,.9,1]))
    else: candidates=[0.0,.01,.02]
    scores=[]
    for parameter in candidates:
        rows=[]
        for subject,frame in meta.groupby("subject_id"):
            action=_decide(frame,policy,parameter); reference=frame.iloc[0]; selected=reference if action=="no_tta" else frame[frame.action==action].iloc[0]
            _,size,_,error=_point(selected,action,alpha); degradation=error-float(reference.source_argmax_error)
            rows.append((size,degradation,action!="no_tta"))
        values=np.asarray(rows,dtype=object); intervention=values[:,2].astype(bool); degradation=values[:,1].astype(float)
        harm=float((degradation[intervention]>epsilon).mean()) if intervention.any() else 0.0
        feasible=intervention.mean()>.0 and degradation.mean()<=epsilon and harm<=.2
        scores.append((not feasible,float(values[:,0].astype(float).mean()),harm,-float(intervention.mean()),str(parameter),parameter))
    return sorted(scores)[0][-1]


def evaluate_baselines(surfaces: pd.DataFrame, alpha: float, epsilon: float, delta: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    subject_rows=[]; settings=[]
    for (dataset,seed,fold),current in surfaces.groupby(["dataset","seed","outer_fold"]):
        meta=current[current.role=="meta_fit_subjects"]; cal=current[current.role=="calibration_subjects"]; outer=current[current.role=="outer_evaluation_subjects"]
        for policy in ["best_fixed_action_crc","entropy_gate_crc","agreement_gate_crc","probe_set_gain_only_crc"]:
            parameter=_fit_rule(meta,policy,alpha,epsilon); settings.append({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"policy":policy,"parameter":parameter})
            indices=[]
            for subject,frame in cal.groupby("subject_id"):
                action=_decide(frame,policy,parameter); row=frame.iloc[0] if action=="no_tta" else frame[frame.action==action].iloc[0]
                risks=np.asarray([row[f"source_risk_j{j}"] if action=="no_tta" else row[f"risk_j{j}"] for j in range(21)])
                degradation=0.0 if action=="no_tta" else float(row.degradation)
                indices.append(joint_critical_index(risks,degradation,alpha=alpha,epsilon=epsilon,sentinel_index=20))
            calibrated=calibrate_policy_index(np.asarray(indices),delta=delta,sentinel_index=20)
            for subject,frame in outer.groupby("subject_id"):
                action=_decide(frame,policy,parameter); row=frame.iloc[0] if action=="no_tta" else frame[frame.action==action].iloc[0]
                prefix="source_" if action=="no_tta" else ""; j=calibrated.lambda_index; risk=float(row[f"{prefix}risk_j{j}"]); size=float(row[f"{prefix}size_j{j}"])
                degradation=0.0 if action=="no_tta" else float(row.degradation)
                subject_rows.append({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"subject_id":subject,"policy":policy,
                    "selected_action":action,"lambda_index":j,"future_risk":risk,"average_set_size":size,"degradation":degradation,
                    "joint_violation":risk>alpha or degradation>epsilon,"intervention":action!="no_tta","sentinel":j==20})
        # Conservative actionwise simultaneous calibration, using the g_set-only decision.
        parameter=_fit_rule(meta,"v2_actionwise_joint",alpha,epsilon); action_indices=[]
        for action,group in cal.groupby("action"):
            values=[]
            for _,row in group.iterrows():
                risks=np.asarray([row[f"risk_j{j}"] for j in range(21)])
                values.append(joint_critical_index(risks,float(row.degradation),alpha=alpha,epsilon=epsilon,sentinel_index=20))
            action_indices.append(calibrate_policy_index(np.asarray(values),delta=delta/2,sentinel_index=20).lambda_index)
        simultaneous=max(action_indices)
        for subject,frame in outer.groupby("subject_id"):
            action=_decide(frame,"v2_actionwise_joint",parameter); row=frame.iloc[0] if action=="no_tta" else frame[frame.action==action].iloc[0]
            prefix="source_" if action=="no_tta" else ""; risk=float(row[f"{prefix}risk_j{simultaneous}"]);size=float(row[f"{prefix}size_j{simultaneous}"])
            degradation=0.0 if action=="no_tta" else float(row.degradation)
            subject_rows.append({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"subject_id":subject,"policy":"v2_actionwise_joint",
                "selected_action":action,"lambda_index":simultaneous,"future_risk":risk,"average_set_size":size,"degradation":degradation,
                "joint_violation":risk>alpha or degradation>epsilon,"intervention":action!="no_tta","sentinel":simultaneous==20})
        # Label-informed upper bound: not deployable and not calibrated.
        for subject,frame in outer.groupby("subject_id"):
            source=frame.iloc[0]; _,source_size,source_risk,_=_point(source,"no_tta",alpha); best=(0.0,"no_tta",source_size,source_risk,0.0)
            for _,row in frame.iterrows():
                _,size,risk,_=_point(row,row.action,alpha); gain=source_size-size
                if row.degradation<=epsilon and gain>best[0]: best=(gain,row.action,size,risk,float(row.degradation))
            _,action,size,risk,degradation=best
            subject_rows.append({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"subject_id":subject,"policy":"oracle_policy",
                "selected_action":action,"lambda_index":-1,"future_risk":risk,"average_set_size":size,"degradation":degradation,
                "joint_violation":risk>alpha or degradation>epsilon,"intervention":action!="no_tta","sentinel":False})
    return pd.DataFrame(subject_rows),pd.DataFrame(settings)

