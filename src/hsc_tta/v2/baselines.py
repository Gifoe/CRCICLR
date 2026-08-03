from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


POLICIES=("no_tta_global_crc","best_fixed_policy_crc","entropy_gate_policy_crc","agreement_gate_policy_crc")


def _global_index(values:np.ndarray,delta:float=.1)->int:
    x=np.sort(np.asarray(values,int)); k=math.ceil((len(x)+1)*(1-delta)); return int(x[min(k,len(x))-1])


def _policy_actions(policy:str,features:pd.DataFrame,*,fixed_action:str="no_tta",threshold:float=0)->pd.Series:
    subjects=features.subject_id.unique(); result={}
    for subject in subjects:
        g=features[features.subject_id==subject].set_index("action")
        if policy=="no_tta_global_crc": action="no_tta"
        elif policy=="best_fixed_policy_crc": action=fixed_action
        elif policy=="entropy_gate_policy_crc":
            source=float(g.loc["no_tta","entropy_q50"]); action=fixed_action if source>=threshold else "no_tta"
        else:
            tta=g.loc[[x for x in g.index if x!="no_tta"]].sort_values(["prediction_agreement","action_cost"],ascending=[False,True])
            action=str(tta.index[0]) if float(tta.iloc[0].prediction_agreement)>=threshold else "no_tta"
        if not bool(g.loc[action,"action_available"]): action="no_tta"
        result[subject]=action
    return pd.Series(result,name="action")


def _choose_meta(policy:str,features:pd.DataFrame,outcomes:pd.DataFrame)->tuple[str,float]:
    action_error=outcomes.groupby("action").argmax_error.mean(); fixed=str(action_error.idxmin())
    if policy=="no_tta_global_crc": return "no_tta",0
    if policy=="best_fixed_policy_crc": return fixed,0
    grid=(.25,.5,.75) if policy=="entropy_gate_policy_crc" else (.8,.9,.95)
    candidates=[]
    for threshold in grid:
        actions=_policy_actions(policy,features,fixed_action=fixed,threshold=threshold)
        selected=outcomes.merge(actions.rename("chosen"),left_on="subject_id",right_index=True)
        selected=selected[selected.action==selected.chosen]
        candidates.append((selected.argmax_error.mean(),threshold))
    return fixed,min(candidates)[1]


def run_external_baselines(root:str|Path)->tuple[pd.DataFrame,pd.DataFrame]:
    root=Path(root); base=root/"outputs/v2_joint_certified"; features=pd.read_parquet(base/"actions/DEVELOPMENT_CONTEXT_FEATURES.parquet")
    outcomes=pd.read_parquet(base/"actions/DEVELOPMENT_ACTION_SURFACE.parquet"); rows=[]; fixed_rows=[]
    for dataset in ("hmc","eegmmidb"):
        for seed in range(5):
            sf=features[(features.dataset==dataset)&(features.seed==seed)]
            so=outcomes[(outcomes.dataset==dataset)&(outcomes.seed==seed)]
            for fold in range(5):
                split=json.loads((root/"data/splits_v2_dev"/dataset/f"seed_{seed}"/f"outer_fold_{fold}.json").read_text())
                meta=set(split["meta_fit_subjects"]); cal=set(split["calibration_subjects"]); ev=set(split["outer_evaluation_subjects"])
                for alpha in (.1,.2):
                    current=so[np.isclose(so.alpha,alpha)]
                    for policy in POLICIES:
                        fixed,threshold=_choose_meta(policy,sf[sf.subject_id.isin(meta)],current[current.subject_id.isin(meta)])
                        cal_actions=_policy_actions(policy,sf[sf.subject_id.isin(cal)],fixed_action=fixed,threshold=threshold)
                        cal_selected=current[current.subject_id.isin(cal)].merge(cal_actions.rename("chosen"),left_on="subject_id",right_index=True)
                        cal_selected=cal_selected[cal_selected.action==cal_selected.chosen]
                        index=_global_index(cal_selected.true_critical_index.to_numpy())
                        ev_actions=_policy_actions(policy,sf[sf.subject_id.isin(ev)],fixed_action=fixed,threshold=threshold)
                        selected=current[current.subject_id.isin(ev)].merge(ev_actions.rename("chosen"),left_on="subject_id",right_index=True)
                        selected=selected[selected.action==selected.chosen].copy()
                        tta=selected.action!="no_tta"; values={
                            "marginal_violation":np.mean(selected[f"risk_j{index}"]>alpha),"csr":float(index<20),
                            "certified_only_violation":np.mean(selected[f"risk_j{index}"]>alpha) if index<20 else np.nan,
                            "full_set_fallback":float(index==20),"average_set_size":selected[f"set_size_j{index}"].mean(),
                            "singleton_rate":selected[f"singleton_j{index}"].mean(),"argmax_error":selected.argmax_error.mean(),
                            "macro_f1":selected.macro_f1.mean(),"balanced_accuracy":selected.balanced_accuracy.mean(),
                            "cohen_kappa":selected.cohen_kappa.mean(),"selected_vs_no_tta_gain":selected.true_benefit.mean(),
                            "nar":np.mean(selected.loc[tta,"true_benefit"]<0) if tta.any() else np.nan,
                            "tta_selection_rate":tta.mean(),"safe_beneficial_selection_precision":np.mean((selected.loc[tta,f"risk_j{index}"]<=alpha)&(selected.loc[tta,"true_benefit"]>0)) if tta.any() else np.nan}
                        harmful=current[current.subject_id.isin(ev)&(current.action!="no_tta")&(current.true_benefit<0)].merge(
                            ev_actions.rename("chosen"),left_on="subject_id",right_index=True)
                        values["her"]=float(np.mean(harmful.action==harmful.chosen)) if len(harmful) else 0.0
                        rows.extend({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"policy":policy,"metric":name,"value":value,
                                     "global_index":index,"fixed_action":fixed,"threshold":threshold} for name,value in values.items())
                    for action,g in current[current.subject_id.isin(ev)].groupby("action"):
                        fixed_rows.append({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"action":action,
                            "argmax_error":g.argmax_error.mean(),"macro_f1":g.macro_f1.mean(),"balanced_accuracy":g.balanced_accuracy.mean(),
                            "cohen_kappa":g.cohen_kappa.mean(),"gain_vs_no_tta":g.true_benefit.mean(),"harm_rate":np.mean(g.true_benefit<0)})
    out=base/"baselines"; out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows); frame.to_csv(out/"EXTERNAL_BASELINE_RESULTS.csv",index=False)
    fixed=pd.DataFrame(fixed_rows); fixed.to_csv(out/"FIXED_ACTION_RESULTS.csv",index=False)
    return frame,fixed
