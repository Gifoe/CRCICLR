from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from hsc_tta.v2.benefit_predictor import fit_benefit_predictor
from hsc_tta.v2.joint_certificate import estimate_scales,finite_sample_quantile,joint_bounds,subject_joint_scores
from hsc_tta.v2.risk_predictor import fit_risk_predictor
from hsc_tta.v2.selector_v2 import select_joint_action


IDENTIFIERS={"dataset","seed","subject_id","alpha","action_available"}


def _atomic(frame:pd.DataFrame,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); part=path.with_suffix(path.suffix+".part")
    frame.to_parquet(part,index=False); os.replace(part,path)


def _hash(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(selected:pd.DataFrame,counter:pd.DataFrame,alpha:float)->dict[str,float]:
    tta=selected.selected_action!="no_tta"; certified=selected.certified_critical_index<20
    selected_risk=np.asarray([row[f"risk_j{int(row.certified_critical_index)}"] for _,row in selected.iterrows()],float)
    selected_set_size=np.asarray([row[f"set_size_j{int(row.certified_critical_index)}"] for _,row in selected.iterrows()],float)
    selected_singleton=np.asarray([row[f"singleton_j{int(row.certified_critical_index)}"] for _,row in selected.iterrows()],float)
    safe_beneficial=counter[(counter.action!="no_tta")&(counter.true_critical_index<20)&(counter.true_benefit>0)]
    safe_oracle=safe_beneficial.groupby("subject_id").true_benefit.max()
    selected_gain=selected.set_index("subject_id").true_benefit
    common=safe_oracle.index
    oracle_total=float(safe_oracle.clip(lower=0).sum())
    captured=float(selected_gain.reindex(common).clip(lower=0).sum()/oracle_total) if oracle_total>0 else np.nan
    return {"marginal_violation":float(np.mean(selected_risk>alpha)),
        "empirical_risk_violation":float(np.mean(selected_risk>alpha)),
        "nonharm_violation":float(np.mean(tta&(selected.true_benefit<0))),
        "joint_validity":float(np.mean((selected_risk<=alpha)&(~tta|(selected.true_benefit>=0)))),
        "csr":float(certified.mean()),"full_set_fallback":float((~certified).mean()),
        "safe_beneficial_subject_rate":float(safe_beneficial.subject_id.nunique()/selected.subject_id.nunique()),
        "certified_positive_adaptation_rate":float(tta.mean()),
        "selected_tta_ppv":float(np.mean(selected.loc[tta,"true_benefit"]>0)) if tta.any() else np.nan,
        "safe_oracle_gain_captured":captured,
        "policy_regret":float(np.mean([max(0,float(safe_oracle.get(s,0)))-float(g) for s,g in selected_gain.items()])),
        "average_set_size":float(selected_set_size.mean()),"singleton_rate":float(selected_singleton.mean()),
        "argmax_error":float(selected.argmax_error.mean()),"macro_f1":float(selected.macro_f1.mean()),
        "balanced_accuracy":float(selected.balanced_accuracy.mean()),"cohen_kappa":float(selected.cohen_kappa.mean()),
        "selected_vs_no_tta_gain":float(selected.true_benefit.mean())}


def _bootstrap_metrics(selected:pd.DataFrame,counter:pd.DataFrame,alpha:float,rng:np.random.Generator,
                       repetitions:int=1000)->dict[str,np.ndarray]:
    subjects=selected.subject_id.to_numpy(); n=len(subjects); draw=rng.integers(0,n,size=(repetitions,n))
    tta=(selected.selected_action!="no_tta").to_numpy(bool); certified=(selected.certified_critical_index<20).to_numpy(bool)
    risk=np.asarray([row[f"risk_j{int(row.certified_critical_index)}"] for _,row in selected.iterrows()],float)
    set_size=np.asarray([row[f"set_size_j{int(row.certified_critical_index)}"] for _,row in selected.iterrows()],float)
    singleton=np.asarray([row[f"singleton_j{int(row.certified_critical_index)}"] for _,row in selected.iterrows()],float)
    gain=selected.true_benefit.to_numpy(float)
    safe=counter[(counter.action!="no_tta")&(counter.true_critical_index<20)&(counter.true_benefit>0)]
    oracle=safe.groupby("subject_id").true_benefit.max()
    oracle_gain=np.asarray([float(oracle.get(s,0)) for s in subjects],float)
    captured_gain=np.where(oracle_gain>0,np.maximum(gain,0),0)
    simple={"marginal_violation":risk>alpha,"empirical_risk_violation":risk>alpha,
            "nonharm_violation":tta&(gain<0),"joint_validity":(risk<=alpha)&(~tta|(gain>=0)),
            "csr":certified,"full_set_fallback":~certified,"safe_beneficial_subject_rate":oracle_gain>0,
            "certified_positive_adaptation_rate":tta,"policy_regret":np.maximum(0,oracle_gain-gain),
            "average_set_size":set_size,"singleton_rate":singleton,"argmax_error":selected.argmax_error.to_numpy(float),
            "macro_f1":selected.macro_f1.to_numpy(float),"balanced_accuracy":selected.balanced_accuracy.to_numpy(float),
            "cohen_kappa":selected.cohen_kappa.to_numpy(float),"selected_vs_no_tta_gain":gain}
    result={name:np.asarray(values,float)[draw].mean(axis=1) for name,values in simple.items()}
    tta_draw=tta.astype(float)[draw]; positive_draw=(tta&(gain>0)).astype(float)[draw]
    denominator=tta_draw.sum(axis=1)
    result["selected_tta_ppv"]=np.divide(positive_draw.sum(axis=1),denominator,out=np.full(repetitions,np.nan),where=denominator>0)
    oracle_denominator=oracle_gain[draw].sum(axis=1)
    result["safe_oracle_gain_captured"]=np.divide(captured_gain[draw].sum(axis=1),oracle_denominator,
                                                  out=np.full(repetitions,np.nan),where=oracle_denominator>0)
    return result


def run_nested_development(root:str|Path)->tuple[pd.DataFrame,pd.DataFrame]:
    root=Path(root); base=root/"outputs/v2_joint_certified"; features=pd.read_parquet(base/"actions/DEVELOPMENT_CONTEXT_FEATURES.parquet")
    outcomes=pd.read_parquet(base/"actions/DEVELOPMENT_ACTION_SURFACE.parquet")
    feature_columns=[c for c in features.columns if c not in IDENTIFIERS]
    all_decisions=[]; all_counter=[]; all_scores=[]; all_bounds=[]; fold_metrics=[]; risk_oof=[]; benefit_oof=[]; predictor_results=[]
    for dataset in ("hmc","eegmmidb"):
        for seed in range(5):
            sf=features[(features.dataset==dataset)&(features.seed==seed)].copy()
            so=outcomes[(outcomes.dataset==dataset)&(outcomes.seed==seed)].copy()
            for fold in range(5):
                split=json.loads((root/"data/splits_v2_dev"/dataset/f"seed_{seed}"/f"outer_fold_{fold}.json").read_text())
                meta=set(split["meta_fit_subjects"]); calibration=set(split["calibration_subjects"]); evaluation=set(split["outer_evaluation_subjects"])
                meta_b=sf[sf.subject_id.isin(meta)].merge(so[np.isclose(so.alpha,.1)][["subject_id","action","true_benefit"]],on=["subject_id","action"])
                meta_b=meta_b.rename(columns={"true_benefit":"benefit_target"})
                benefit=fit_benefit_predictor(meta_b,feature_columns)
                chosen_b=benefit.oof[benefit.oof.model==benefit.model_name].rename(columns={"true_gain":"true_benefit","predicted_gain":"predicted_benefit"})
                chosen_b["dataset"]=dataset; chosen_b["seed"]=seed; chosen_b["outer_fold"]=fold; benefit_oof.append(chosen_b)
                fold_dir=base/"nested_dev/models"/dataset/f"seed_{seed}"/f"fold_{fold}"; fold_dir.mkdir(parents=True,exist_ok=True)
                joblib.dump(benefit.model,fold_dir/"benefit.joblib")
                for row in benefit.results.to_dict("records"): predictor_results.append({"dataset":dataset,"seed":seed,"fold":fold,"target":"benefit",**row})
                for alpha in (.1,.2):
                    meta_r=sf[sf.subject_id.isin(meta)].merge(so[np.isclose(so.alpha,alpha)][["subject_id","action","true_critical_index"]],on=["subject_id","action"])
                    meta_r["alpha"]=alpha; risk=fit_risk_predictor(meta_r,feature_columns)
                    chosen_r=risk.oof[risk.oof.model==risk.model_name].copy(); chosen_r["dataset"]=dataset; chosen_r["seed"]=seed; chosen_r["outer_fold"]=fold
                    risk_oof.append(chosen_r)
                    joblib.dump(risk.model,fold_dir/f"risk_alpha_{alpha:.2f}.joblib")
                    for row in risk.results.to_dict("records"): predictor_results.append({"dataset":dataset,"seed":seed,"fold":fold,"alpha":alpha,"target":"risk",**row})
                    scale_frame=chosen_r.merge(chosen_b[["subject_id","action","true_benefit","predicted_benefit"]],
                                               on=["subject_id","action"],how="left")
                    no_mask=scale_frame.action=="no_tta"
                    scale_frame.loc[no_mask,["true_benefit","predicted_benefit"]]=0
                    if scale_frame.loc[~no_mask,["true_benefit","predicted_benefit"]].isna().any().any():
                        raise RuntimeError("missing TTA benefit OOF residual")
                    scales=estimate_scales(scale_frame)
                    cal=sf[sf.subject_id.isin(calibration)].merge(so[np.isclose(so.alpha,alpha)][["subject_id","action","true_critical_index","true_benefit"]],on=["subject_id","action"])
                    cal["predicted_critical_index"]=risk.model.predict(cal[feature_columns])
                    cal["predicted_benefit"]=benefit.model.predict(cal[feature_columns]); cal.loc[cal.action=="no_tta",["true_benefit","predicted_benefit"]]=0
                    scores=subject_joint_scores(cal,scales); q,k=finite_sample_quantile(scores.joint_score,.1)
                    cal["risk_score"]=(cal.true_critical_index-cal.predicted_critical_index)/scales.c_j
                    cal["benefit_score"]=np.where(cal.action=="no_tta",-np.inf,(cal.predicted_benefit-cal.true_benefit)/scales.c_delta)
                    q_risk,k_risk=finite_sample_quantile(cal.groupby("subject_id").risk_score.max(),.1)
                    q_benefit,k_benefit=finite_sample_quantile(cal.groupby("subject_id").benefit_score.max(),.1)
                    scores[["dataset","seed","outer_fold","alpha","q","k","c_j","c_delta","q_risk_separate","q_benefit_separate"]]=[dataset,seed,fold,alpha,q,k,scales.c_j,scales.c_delta,q_risk,q_benefit]
                    all_scores.append(scores)
                    ev=sf[sf.subject_id.isin(evaluation)].copy(); ev["alpha"]=alpha
                    ev["predicted_critical_index"]=risk.model.predict(ev[feature_columns]); ev["predicted_benefit"]=benefit.model.predict(ev[feature_columns]); ev.loc[ev.action=="no_tta","predicted_benefit"]=0
                    upper,lower=joint_bounds(ev.predicted_critical_index,ev.predicted_benefit,q,scales,20)
                    ev["certified_critical_index"]=upper; ev["benefit_lower"]=lower; ev["available"]=ev.action_available
                    separate_upper=np.clip(np.ceil(ev.predicted_critical_index+q_risk*scales.c_j),0,20).astype(int)
                    separate_lower=ev.predicted_benefit-q_benefit*scales.c_delta
                    ev["separate_certified_critical_index"]=separate_upper; ev["separate_benefit_lower"]=separate_lower
                    ev["outer_fold"]=fold; ev["q"]=q; ev["c_j"]=scales.c_j; ev["c_delta"]=scales.c_delta
                    all_bounds.append(ev[["dataset","seed","outer_fold","subject_id","action","alpha","predicted_critical_index","predicted_benefit","certified_critical_index","benefit_lower","separate_certified_critical_index","separate_benefit_lower","available","q","c_j","c_delta"]])
                    decisions=[]
                    for subject,g in ev.groupby("subject_id"):
                        candidates=[]
                        for row in g.itertuples(index=False):
                            index=int(row.certified_critical_index)
                            candidates.append({"action":row.action,"available":row.available,"certified_critical_index":index,
                                "benefit_lower":row.benefit_lower,"context_average_set_size":getattr(row,f"context_set_size_j{index}"),
                                "adaptation_cost":row.action_cost})
                        selection=select_joint_action(pd.DataFrame(candidates),sentinel_index=20)
                        decisions.append({"dataset":dataset,"seed":seed,"outer_fold":fold,"subject_id":subject,"alpha":alpha,**selection})
                    decision_frame=pd.DataFrame(decisions); decision_path=base/"nested_dev/decisions"/dataset/f"seed_{seed}"/f"fold_{fold}_alpha_{alpha:.2f}.parquet"
                    _atomic(decision_frame,decision_path)
                    freeze_path=decision_path.with_suffix(".freeze.json"); freeze_path.write_text(json.dumps({"decision_sha256":_hash(decision_path),"V_opened":False},indent=2),encoding="utf-8")
                    # Only after decision hash exists do we join the outer-evaluation V outcomes.
                    counter=so[so.subject_id.isin(evaluation)&np.isclose(so.alpha,alpha)].copy(); counter["outer_fold"]=fold
                    selected=decision_frame.merge(counter,left_on=["dataset","seed","outer_fold","subject_id","alpha","selected_action"],
                        right_on=["dataset","seed","outer_fold","subject_id","alpha","action"],validate="one_to_one")
                    freeze_payload=json.loads(freeze_path.read_text()); assert freeze_payload["decision_sha256"]==_hash(decision_path)
                    freeze_payload["V_opened"]=True; freeze_path.write_text(json.dumps(freeze_payload,indent=2),encoding="utf-8")
                    all_decisions.append(selected); all_counter.append(counter)
                    values=_metrics(selected,counter,alpha)
                    fold_metrics.extend({"dataset":dataset,"seed":seed,"outer_fold":fold,"alpha":alpha,"policy":"joint_hsc_tta_v2","metric":name,"value":value}
                                        for name,value in values.items())
    decisions=pd.concat(all_decisions,ignore_index=True); counters=pd.concat(all_counter,ignore_index=True)
    nested=base/"nested_dev"; _atomic(decisions,nested/"ALL_DEV_DECISIONS.parquet"); _atomic(counters,nested/"ALL_DEV_COUNTERFACTUALS.parquet")
    _atomic(pd.concat(all_scores,ignore_index=True),nested/"ALL_DEV_CALIBRATION_SCORES.parquet"); _atomic(pd.concat(all_bounds,ignore_index=True),nested/"ALL_DEV_JOINT_BOUNDS.parquet")
    fold_frame=pd.DataFrame(fold_metrics); fold_frame.to_csv(nested/"DEV_RESULTS_BY_FOLD.csv",index=False)
    by_seed=fold_frame.groupby(["dataset","seed","alpha","policy","metric"]).value.mean().reset_index(); by_seed.to_csv(nested/"DEV_RESULTS_BY_SEED.csv",index=False)
    summary=by_seed.groupby(["dataset","alpha","policy","metric"]).value.agg(["mean","std"]).reset_index(); summary.to_csv(nested/"DEV_RESULTS_SUMMARY.csv",index=False)
    # Fold-local subject bootstrap; repeated subjects across seeds are never pooled inside a bootstrap.
    ci=[]
    for keys,g in decisions.groupby(["dataset","seed","outer_fold","alpha"]):
        rng=np.random.default_rng(81000+int(keys[1])*100+int(keys[2])*10+int(float(keys[3])*100)); subjects=g.subject_id.unique()
        counter=counters[(counters.dataset==keys[0])&(counters.seed==keys[1])&(counters.outer_fold==keys[2])&np.isclose(counters.alpha,keys[3])]
        samples=_bootstrap_metrics(g,counter,float(keys[3]),rng,1000)
        point=_metrics(g,counter,float(keys[3]))
        for metric,value in point.items():
            vals=samples[metric]
            ci.append({"dataset":keys[0],"seed":keys[1],"outer_fold":keys[2],"alpha":keys[3],"metric":metric,"point_estimate":value,
                       "ci_lower":np.nanquantile(vals,.025),"ci_upper":np.nanquantile(vals,.975),"n_subjects":len(subjects)})
    pd.DataFrame(ci).to_csv(nested/"DEV_RESULTS_WITH_CI.csv",index=False)
    predictors=base/"predictors"; predictors.mkdir(exist_ok=True)
    _atomic(pd.concat(benefit_oof,ignore_index=True),predictors/"BENEFIT_PREDICTOR_OOF.parquet")
    _atomic(pd.concat(risk_oof,ignore_index=True),predictors/"RISK_PREDICTOR_OOF.parquet")
    pd.DataFrame(predictor_results).to_csv(predictors/"PREDICTOR_RESULTS_ALL.csv",index=False)
    return decisions,fold_frame
