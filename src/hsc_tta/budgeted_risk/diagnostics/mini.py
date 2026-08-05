from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.stats import spearmanr

from hsc_tta.budgeted_risk.acquisition import temporal_stratified_order
from hsc_tta.budgeted_risk.budget import _load, _local_index
from hsc_tta.budgeted_risk.inclusion_index import critical_index_from_kappa, inclusion_index_table, inclusion_indices
from hsc_tta.budgeted_risk.stage0 import _fit, _predict, _select_model
from hsc_tta.contextual_risk.families import TPSFamily
from hsc_tta.contextual_risk.statistics import clopper_pearson_upper, paired_bootstrap_ci
from hsc_tta.contextual_risk.io import atomic_parquet

from .calibration_schemes import S2, S3, conformal_q, fold_split


CANDIDATES = ("direct", "isotonic", "ridge_10", "ordinal_1")
STATES = ("INITIALIZED", "INPUT_AUDIT_COMPLETE", "PROTOCOL_FROZEN", "SMOKE_TEST_COMPLETE",
          "S2_COMPLETE", "S3_COMPLETE", "STATISTICS_COMPLETE", "DECISION_COMPLETE",
          "DELIVERY_COMPLETE", "STOPPED")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"); tmp.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _state(path: Path, state: str, *, commit: str, hashes: dict[str,str], config_hash: str,
           completed: list[str], failed: list[str]) -> None:
    if state not in STATES: raise ValueError(state)
    if path.exists():
        old = json.loads(path.read_text()); previous = old["state"]; history = old["history"]
        if STATES.index(state) != STATES.index(previous) + 1: raise RuntimeError(f"bad state {previous}->{state}")
    else:
        if state != "INITIALIZED": raise RuntimeError("state must initialize first")
        previous = None; history = []
    row = {"state":state,"previous_state":previous,"timestamp":datetime.now(timezone.utc).isoformat(),
           "git_commit":commit,"input_hashes":hashes,"config_hash":config_hash,
           "completed_jobs":sorted(completed),"failed_jobs":sorted(failed),
           "formal_calibration_opened":False,"internal_final_opened":False,"cap_opened":False}
    history.append(row); _atomic_json({**row,"history":history},path)


def _prior(arrays: dict[str,Any], subjects: list[str]) -> np.ndarray:
    cdfs=[]
    for s in subjects:
        a=arrays[s]
        if a.inclusion_table is None: a.inclusion_table=inclusion_index_table(a.probabilities)
        k=a.inclusion_table[np.arange(len(a.labels)),a.labels]
        cdfs.append([float(np.mean(k<=j)) for j in range(21)])
    return np.mean(cdfs,axis=0)


def _base_values(a, budget: int) -> dict[str,Any]:
    if a.inclusion_table is None:a.inclusion_table=inclusion_index_table(a.probabilities)
    pos=temporal_stratified_order(len(a.indices))[:min(budget,len(a.indices))]
    k=a.inclusion_table[pos,a.labels[pos]].astype(int); effective=len(k)
    q=np.quantile(k,[.5,.8,.9])
    prefixes=[critical_index_from_kappa(k[:max(1,int(np.ceil(len(k)*f)))],.1) for f in (.25,.5,.75,1.)]
    if a.future_j is None:
        a.future_j=critical_index_from_kappa(inclusion_indices(a.future_probabilities,a.future_labels),.1)
        _,a.future_sizes,_=TPSFamily().future_curve(a.future_probabilities,a.future_labels)
    return {"effective_budget":effective,"local_kappa_q50":float(q[0]),"local_kappa_q80":float(q[1]),
            "local_kappa_q90":float(q[2]),"local_kappa_std":float(k.std()),
            "prefix_instability":float(np.mean(abs(np.asarray(prefixes)-prefixes[-1]))),
            "j_future":int(a.future_j),"_kappa":k}


def _feature_row(base: dict[str,Any], context: dict[str,float], prior: np.ndarray, tau: float) -> dict[str,float]:
    return {**context,"j_local":float(_local_index(prior,base["_kappa"],tau,.1)),"budget":float(base["effective_budget"]),
            **{k:base[k] for k in ("local_kappa_q50","local_kappa_q80","local_kappa_q90","local_kappa_std")}}


def _columns(frame: pd.DataFrame) -> list[str]:
    return sorted(set(frame.columns)-{"dataset","subject_id","screening_fold","role","j_future","effective_budget","prefix_instability"})


def _mini_scale_design(frame: pd.DataFrame) -> np.ndarray:
    spread=np.asarray(frame.local_kappa_std,float)+(np.asarray(frame.local_kappa_q90,float)-np.asarray(frame.local_kappa_q50,float))
    return np.column_stack([spread,1/np.sqrt(np.asarray(frame.effective_budget,float)+1),np.asarray(frame.prefix_instability,float)])


def _fit_scale(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float);y=np.maximum(np.asarray(y,float),0)
    def fun(a):
        e=y-(.25+x@a);return float(np.maximum(.75*e,-.25*e).mean()+1e-3*np.square(a).sum())
    def jac(a):
        e=y-(.25+x@a);w=np.where(e>0,-.75,np.where(e<0,.25,0));return (w[:,None]*x).mean(0)+2e-3*a
    r=minimize(fun,np.zeros(3),jac=jac,method="SLSQP",bounds=[(0,None)]*3,options={"ftol":1e-10,"maxiter":500})
    if not r.success: raise RuntimeError(f"scale fit failed: {r.message}")
    return np.maximum(r.x,0)


def _predict_scale(frame: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    out=.25+_mini_scale_design(frame)@coef
    if np.any(out<=0):raise RuntimeError("non-positive scale")
    return out


def _job_paths(out: Path,dataset:str,seed:int,fold:int):
    stem=f"{dataset}_seed{seed}_fold{fold}"
    return out/"base_predictions"/f"BASE_PREDICTIONS_{stem}.parquet",out/"base_predictions"/f"BASE_MODEL_{stem}.json",out/"job_results"/f"JOB_RESULTS_{stem}.parquet"


def run_job(project_root: str, dataset: str, seed: int, fold: int, smoke: bool=False) -> dict[str,Any]:
    started=time.time();root=Path(project_root);repo=root/"repo";out=repo/"outputs/budgeted_risk_v51_mini"
    bp,mp,rp=_job_paths(out,dataset,seed,fold)
    if bp.exists() and mp.exists() and rp.exists():return {"job":f"{dataset}:{seed}:{fold}","status":"skipped","seconds":0,"result":str(rp)}
    cohorts=pd.read_parquet(repo/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet")
    dev=cohorts[(cohorts.master_cohort=="method_development")&(cohorts.dataset==dataset)]
    fold_map=dev.set_index("subject_id").screening_fold.astype(int).to_dict();subjects=sorted(fold_map)
    split=fold_split(S2,fold);train=[s for s in subjects if fold_map[s] in split.training];cal=[s for s in subjects if fold_map[s] in split.calibration];ev=[s for s in subjects if fold_map[s]==fold]
    if set(train)&set(cal) or set(train)&set(ev) or set(cal)&set(ev):raise RuntimeError("split overlap")
    manifest=pd.read_parquet(repo/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet")
    index=manifest[(manifest.dataset==dataset)&(manifest.fold==fold)&(manifest.seed==seed)].set_index("subject_id")
    arrays={s:_load(Path(index.loc[s,"cache_path"])) for s in subjects}
    uf=pd.read_parquet(repo/"outputs/budgeted_risk/features/UNLABELED_CONTEXT_FEATURES.parquet")
    uf=uf[(uf.dataset==dataset)&(uf.seed==seed)&(uf.outer_fold==fold)].drop_duplicates("subject_id").set_index("subject_id")
    excluded={"schema_version","dataset","subject_id","seed","outer_fold","rotation_role","context_embedding_norm_mean"}
    contexts={s:{c:float(v) for c,v in uf.loc[s].items() if c not in excluded and pd.notna(v)} for s in subjects}
    base={s:_base_values(arrays[s],20) for s in subjects};prior=_prior(arrays,train)
    global_index=int(np.flatnonzero(prior>=.9)[0]) if np.any(prior>=.9) else 20
    choices=[]
    for tau in (2.,5.,10.,20.):
        rows=[]
        for s in train:rows.append({"dataset":dataset,"subject_id":s,"screening_fold":fold_map[s],"role":"train","j_future":base[s]["j_future"],"effective_budget":base[s]["effective_budget"],"prefix_instability":base[s]["prefix_instability"],**_feature_row(base[s],contexts[s],prior,tau)})
        frame=pd.DataFrame(rows);cols=_columns(frame);selected,scores=_select_model(frame,cols,CANDIDATES,index_column="j_local");score=scores[scores.candidate==selected].iloc[0]
        choices.append((float(score.mae),-float(score.spearman),float(score.underestimation_rate),tau,selected,cols,frame,scores))
    best=min(choices,key=lambda x:(x[0],x[1],x[2],x[3],x[4]));_,_,_,tau,selected,cols,train_frame,scores=best
    fitted=_fit(selected,train_frame[cols].to_numpy(),train_frame.j_future.to_numpy(),train_frame.j_local.to_numpy())
    all_rows=[]
    for s in subjects:
        role="train" if s in train else "calibration" if s in cal else "evaluation" if s in ev else "forbidden"
        row={"dataset":dataset,"subject_id":s,"seed":seed,"outer_fold":fold,"screening_fold":fold_map[s],"role":role,
             "effective_budget":base[s]["effective_budget"],"prefix_instability":base[s]["prefix_instability"],"j_future":base[s]["j_future"],
             **_feature_row(base[s],contexts[s],prior,tau)}
        row["raw_prediction"]=float(_predict(fitted,pd.DataFrame([row])[cols].to_numpy(),np.asarray([row["j_local"]]))[0])
        for j,v in enumerate(arrays[s].future_sizes):row[f"size_{j}"]=float(v)
        all_rows.append(row)
    all_frame=pd.DataFrame(all_rows);cal_frame=all_frame[all_frame.role=="calibration"].copy();eval_frame=all_frame[all_frame.role=="evaluation"].copy()
    residual=np.maximum(cal_frame.j_future-cal_frame.raw_prediction,0).to_numpy();q2,k2=conformal_q(residual,.1)
    global_q,_=conformal_q(np.maximum(cal_frame.j_future-global_index,0).to_numpy(),.1)
    # Strict two-way OOF scale fitting: each held training fold is excluded from predictor and prior fitting.
    oof=[]
    for held in split.training:
        inner_folds=tuple(f for f in split.training if f!=held);inner_subjects=[s for s in train if fold_map[s] in inner_folds];held_subjects=[s for s in train if fold_map[s]==held]
        inner_prior=_prior(arrays,inner_subjects);ir=[]
        for s in inner_subjects:ir.append({"subject_id":s,"screening_fold":fold_map[s],"j_future":base[s]["j_future"],"effective_budget":base[s]["effective_budget"],"prefix_instability":base[s]["prefix_instability"],**_feature_row(base[s],contexts[s],inner_prior,tau)})
        inf=pd.DataFrame(ir);inner_fit=_fit(selected,inf[cols].to_numpy(),inf.j_future.to_numpy(),inf.j_local.to_numpy())
        for s in held_subjects:
            r={"subject_id":s,"j_future":base[s]["j_future"],"effective_budget":base[s]["effective_budget"],"prefix_instability":base[s]["prefix_instability"],**_feature_row(base[s],contexts[s],inner_prior,tau)}
            r["prediction"]=float(_predict(inner_fit,pd.DataFrame([r])[cols].to_numpy(),np.asarray([r["j_local"]]))[0]);oof.append(r)
    oof_frame=pd.DataFrame(oof);oof_res=np.maximum(oof_frame.j_future-oof_frame.prediction,0).to_numpy();coef=_fit_scale(_mini_scale_design(oof_frame),oof_res)
    cal_sigma=_predict_scale(cal_frame,coef);q3,k3=conformal_q(residual/cal_sigma,.1);eval_sigma=_predict_scale(eval_frame,coef)
    job_rows=[]
    for scheme,q,k,scale in ((S2,q2,k2,np.ones(len(eval_frame))),(S3,q3,k3,eval_sigma)):
        for i,(_,r) in enumerate(eval_frame.iterrows()):
            raw=int(np.clip(np.ceil(r.raw_prediction),0,20));cert=int(np.clip(np.ceil(r.raw_prediction+q*scale[i]),0,20));gcert=int(np.clip(np.ceil(global_index+global_q),0,20));true=int(r.j_future)
            sizes=np.asarray([r[f"size_{j}"] for j in range(21)],float)
            job_rows.append({"dataset":dataset,"subject_id":r.subject_id,"seed":seed,"outer_fold":fold,"scheme":scheme,"budget":20,"strategy":"temporal",
                "j_future":true,"raw_prediction":r.raw_prediction,"raw_index":raw,"certified_index":cert,"global_base_index":global_index,"global_q":global_q,"global_certified_index":gcert,
                "q":q,"scale":scale[i],"calibration_m":len(cal_frame),"order_statistic_k":k,"selected_model":selected,"tau":tau,
                "raw_absolute_error":abs(r.raw_prediction-true),"global_absolute_error":abs(global_index-true),"raw_underestimation":r.raw_prediction<true,
                "raw_set_size":sizes[raw],"set_size":sizes[cert],"global_raw_set_size":sizes[global_index],"global_set_size":sizes[gcert],"oracle_set_size":sizes[true],
                "violation":cert<true,"sentinel":cert==20,"global_sentinel":gcert==20,"sentinel_transition":raw<20 and cert==20,
                "method_correction_cost":sizes[cert]-sizes[raw],"global_correction_cost":sizes[gcert]-sizes[global_index],
                "excess_correction_cost":sizes[cert]-sizes[raw]-(sizes[gcert]-sizes[global_index]),**{f"size_{j}":sizes[j] for j in range(21)}})
    model={"dataset":dataset,"seed":seed,"outer_fold":fold,"train_folds":split.training,"calibration_folds":split.calibration,"evaluation_fold":fold,
           "selected_model":selected,"tau":tau,"feature_columns":cols,"global_base_index":global_index,"global_q":global_q,
           "s2_q":q2,"s2_k":k2,"s3_q":q3,"s3_k":k3,"scale_coefficients":coef.tolist(),"scale_intercept":.25,
           "scale_design":"[std+(q90-q50),1/sqrt(effective_budget+1),prefix_instability]","scale_training_subjects":sorted(train),
           "calibration_subjects":sorted(cal),"evaluation_subjects":sorted(ev)}
    bp.parent.mkdir(parents=True,exist_ok=True);rp.parent.mkdir(parents=True,exist_ok=True)
    atomic_parquet(all_frame,bp);_atomic_json(model,mp);atomic_parquet(pd.DataFrame(job_rows),rp)
    return {"job":f"{dataset}:{seed}:{fold}","status":"complete","seconds":time.time()-started,"result":str(rp)}


def _aggregate(repo: Path, config: dict[str,Any]) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    files=sorted((repo/"outputs/budgeted_risk_v51_mini/job_results").glob("JOB_RESULTS_*.parquet"));results=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    s2=results[results.scheme==S2];s3=results[results.scheme==S3];atomic_parquet(s2,repo/"outputs/budgeted_risk_v51_mini/S2_RESULTS.parquet");atomic_parquet(s3,repo/"outputs/budgeted_risk_v51_mini/S3_RESULTS.parquet")
    subject=results.groupby(["dataset","subject_id","scheme"],as_index=False).mean(numeric_only=True);atomic_parquet(subject,repo/"outputs/budgeted_risk_v51_mini/RESULTS_BY_SUBJECT.parquet")
    summaries=[];seed_rows=[]
    for (dataset,scheme),cur in results.groupby(["dataset","scheme"]):
        sub=subject[(subject.dataset==dataset)&(subject.scheme==scheme)];gain=sub.global_set_size-sub.set_size;ci=paired_bootstrap_ci(gain.to_numpy(),reps=5000,seed=20260805)
        seedstats=[]
        for seed,g in cur.groupby("seed"):
            v=int(g.violation.sum());n=len(g);cp=clopper_pearson_upper(v,n,.95);seedstats.append((v/n,cp));seed_rows.append({"dataset":dataset,"scheme":scheme,"seed":seed,"n_subjects":n,"violations":v,"violation":v/n,"cp_upper":cp,"relative_gain":float((g.global_set_size-g.set_size).mean()/g.global_set_size.mean()),"sentinel_rate":g.sentinel.mean()})
        qcells=cur.groupby(["seed","outer_fold"]).first()
        rho=float(spearmanr(sub.raw_prediction,sub.j_future).statistic);mae=float(sub.raw_absolute_error.mean());gmae=float(sub.global_absolute_error.mean())
        recovery=float(gain.sum()/max((sub.global_set_size-sub.oracle_set_size).sum(),1e-12))
        summaries.append({"dataset":dataset,"scheme":scheme,"n_subjects":len(sub),"calibration_m_by_fold":"26" if dataset=="eegmmidb" else "36","order_statistic_k_by_fold":"25" if dataset=="eegmmidb" else "34",
            "raw_spearman":rho,"raw_mae":mae,"global_mae":gmae,"raw_mae_improvement":(gmae-mae)/gmae if gmae else 0,"raw_underestimation":sub.raw_underestimation.mean(),
            "q_mean":qcells.q.mean(),"q_median":qcells.q.median(),"q_max":qcells.q.max(),"mean_seed_violation":np.mean([x[0] for x in seedstats]),"worst_seed_violation":max(x[0] for x in seedstats),"max_seed_cp_upper":max(x[1] for x in seedstats),
            "average_set_size":sub.set_size.mean(),"global_average_set_size":sub.global_set_size.mean(),"relative_gain_vs_global":gain.mean()/sub.global_set_size.mean(),"paired_gain_ci_low":ci[0],"paired_gain_ci_high":ci[1],"oracle_gain_recovered":recovery,
            "sentinel_rate":sub.sentinel.mean(),"global_sentinel_rate":sub.global_sentinel.mean(),"sentinel_delta":sub.sentinel.mean()-sub.global_sentinel.mean(),"sentinel_transition_rate":sub.sentinel_transition.mean(),
            "method_correction_cost":sub.method_correction_cost.mean(),"global_correction_cost":sub.global_correction_cost.mean(),"excess_correction_cost":sub.excess_correction_cost.mean()})
    summary=pd.DataFrame(summaries);seed_frame=pd.DataFrame(seed_rows)
    # One cheap q-driver deletion per job/scheme.
    qrows=[]
    for mp in sorted((repo/"outputs/budgeted_risk_v51_mini/base_predictions").glob("BASE_MODEL_*.json")):
        model=json.loads(mp.read_text());dataset=model["dataset"];seed=model["seed"];fold=model["outer_fold"]
        bp=mp.with_name(mp.name.replace("BASE_MODEL_","BASE_PREDICTIONS_").replace(".json",".parquet"));base=pd.read_parquet(bp);cal=base[base.role=="calibration"]
        jr=results[(results.dataset==dataset)&(results.seed==seed)&(results.outer_fold==fold)]
        residual=np.maximum(cal.j_future-cal.raw_prediction,0).to_numpy()
        coef=np.asarray(model["scale_coefficients"]);sigma=_predict_scale(cal,coef)
        for scheme,scores in ((S2,residual),(S3,residual/sigma)):
            order=np.argsort(scores,kind="stable");k=conformal_q(scores,.1)[1];driver_pos=order[k-1];driver=cal.iloc[driver_pos].subject_id
            keep=np.arange(len(cal))!=driver_pos;qnew,_=conformal_q(scores[keep],.1);ev=jr[jr.scheme==scheme].copy();scale=np.ones(len(ev)) if scheme==S2 else ev.scale.to_numpy();newcert=np.clip(np.ceil(ev.raw_prediction+qnew*scale),0,20).astype(int)
            newsizes=np.asarray([r[f"size_{c}"] for (_,r),c in zip(ev.iterrows(),newcert,strict=True)]);basegain=float((ev.global_set_size-ev.set_size).mean()/ev.global_set_size.mean());newgain=float((ev.global_set_size.to_numpy()-newsizes).mean()/ev.global_set_size.mean());bs=ev.sentinel.mean();ns=float(np.mean(newcert==20));unstable=bool(np.sign(basegain)!=np.sign(newgain) or newgain-basegain>=.05 or bs-ns>=.10)
            qrows.append({"dataset":dataset,"scheme":scheme,"seed":seed,"outer_fold":fold,"q_driver_subject":driver,"q_original":ev.q.iloc[0],"q_without_driver":qnew,"q_drop":ev.q.iloc[0]-qnew,"gain_original":basegain,"gain_without_driver":newgain,"gain_change":newgain-basegain,"sentinel_original":bs,"sentinel_without_driver":ns,"sentinel_drop":bs-ns,"gain_sign_changed":np.sign(basegain)!=np.sign(newgain),"unstable":unstable})
    qframe=pd.DataFrame(qrows);atomic_parquet(qframe,repo/"outputs/budgeted_risk_v51_mini/Q_DRIVER_RESULTS.parquet")
    # Evaluation-subject sign stability.
    sign={}
    for (dataset,scheme),g in subject.groupby(["dataset","scheme"]):
        base=float((g.global_set_size-g.set_size).mean()/g.global_set_size.mean());vals=[]
        for s in g.subject_id:
            x=g[g.subject_id!=s];vals.append(float((x.global_set_size-x.set_size).mean()/x.global_set_size.mean()))
        sign[(dataset,scheme)]=not any(np.sign(v)!=np.sign(base) for v in vals)
    rates=qframe.groupby(["dataset","scheme"]).unstable.mean().to_dict()
    summary["q_driver_unstable_fold_rate"]=[rates[(r.dataset,r.scheme)] for r in summary.itertuples()]
    summary["evaluation_loo_sign_stable"]=[sign[(r.dataset,r.scheme)] for r in summary.itertuples()]
    summary.to_csv(repo/"outputs/budgeted_risk_v51_mini/MINI_SUMMARY.csv",index=False);seed_frame.to_csv(repo/"outputs/budgeted_risk_v51_mini/RESULTS_BY_SEED.csv",index=False)
    return results,subject,summary,qframe


def _pass(r) -> bool:
    return bool(r.mean_seed_violation<=.10 and r.worst_seed_violation<=.20 and r.max_seed_cp_upper<=.20 and r.relative_gain_vs_global>=.05 and r.paired_gain_ci_low>0 and r.oracle_gain_recovered>=.20 and r.sentinel_delta<=.05 and r.sentinel_transition_rate<=.10 and r.q_driver_unstable_fold_rate<=.20 and r.evaluation_loo_sign_stable)


def _table(df: pd.DataFrame) -> str:
    cols=list(df.columns);lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    for row in df.itertuples(index=False,name=None):lines.append("| "+" | ".join(f"{x:.4f}" if isinstance(x,float) else str(x) for x in row)+" |")
    return "\n".join(lines)


def run_mini(project_root: str, resume: bool=False) -> str:
    started=time.time();root=Path(project_root);repo=root/"repo";out=repo/"outputs/budgeted_risk_v51_mini";delivery=repo/"delivery/budgeted_risk_v51_mini";out.mkdir(parents=True,exist_ok=True);delivery.mkdir(parents=True,exist_ok=True)
    config_path=repo/"configs/budgeted_risk_v51_mini/mini.yaml";config=yaml.safe_load(config_path.read_text());commit=_git(repo,"rev-parse","HEAD")
    inputs=[repo/"delivery/budgeted_risk_v51/V51_PAUSED_BY_USER.md",repo/"delivery/budgeted_risk_v51/V51_PARTIAL_CHECKPOINT.json",repo/"delivery/budgeted_risk_v51/V51_DIAGNOSTIC_FREEZE.json",repo/"delivery/budgeted_risk_v51/V51_REPOSITORY_AND_RESULT_AUDIT.md",repo/"outputs/budgeted_risk_v51/results/S1_RESULTS.parquet",repo/"outputs/budgeted_risk_v51/results/RAW_PREDICTOR_RESULTS.parquet",repo/"outputs/budgeted_risk_v51/results/CALIBRATION_RESIDUALS_S1.parquet",repo/"outputs/budgeted_risk_v51/results/S1_REPRODUCTION.json",repo/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet",repo/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet"]
    hashes={str(p.relative_to(repo)):_sha(p) for p in inputs};repro=json.loads((repo/"outputs/budgeted_risk_v51/results/S1_REPRODUCTION.json").read_text());partial=json.loads((repo/"delivery/budgeted_risk_v51/V51_PARTIAL_CHECKPOINT.json").read_text());audit=json.loads((repo/"outputs/budgeted_risk_v51/audit/COMMIT_AUDIT.json").read_text())
    valid=bool(repro["passed"] and audit["ordinal_fix_is_run_ancestor"] and not partial["formal_calibration_opened"] and not partial["internal_final_opened"] and not partial["cap_opened"])
    if not valid:raise RuntimeError("MINI_TECHNICAL_BLOCK")
    state=out/"RUN_STATE.json";completed=[];failed=[]
    if not state.exists():
        _state(state,"INITIALIZED",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=completed,failed=failed)
        (delivery/"MINI_INPUT_AUDIT.md").write_text(f"# Mini input audit\n\nS1 reproduction: passed. Ordinal ancestry: passed. Protected flags: false. Input hashes:\n\n```json\n{json.dumps(hashes,indent=2)}\n```\n",encoding="utf-8")
        _state(state,"INPUT_AUDIT_COMPLETE",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=completed,failed=failed)
        freeze={**config,"input_hashes":hashes,"ordinal_fix_commit":audit["ordinal_fix_commit"],"stage0_run_commit":audit["run_commit"],"scale_spread_definition":"local_kappa_std + (local_kappa_q90-local_kappa_q50)","hard_stop":True}
        _atomic_json(freeze,delivery/"MINI_FREEZE.json");(delivery/"MINI_PROTOCOL.md").write_text("# V5.1-Mini frozen protocol\n\nOnly HMC/EEGMMIDB method-development, temporal b=20, five seeds/folds, S2/S3, frozen CBraMod cache. S2/S3 share one base predictor per job. S3 scale uses training-fold OOF residuals only. Subject is the independent unit; seed-wise CP uses subjects only. No S1/S4/random/GPU/source retraining/protected cohort.\n",encoding="utf-8")
        _state(state,"PROTOCOL_FROZEN",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=completed,failed=failed)
        smoke=run_job(project_root,"eegmmidb",0,0,True);completed.append(smoke["job"]);_state(state,"SMOKE_TEST_COMPLETE",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=completed,failed=failed)
    else:
        current=json.loads(state.read_text());completed=current["completed_jobs"];failed=current["failed_jobs"]
    jobs=[(d,s,f) for d in config["datasets"] for s in config["seeds"] for f in config["folds"]]
    workers=min(8,max(1,(os.cpu_count() or 2)-2));fail_rows=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(run_job,project_root,*j):j for j in jobs}
        for fut in as_completed(futures):
            j=futures[fut];name=f"{j[0]}:{j[1]}:{j[2]}"
            try:r=fut.result();completed.append(name);print(json.dumps({"heartbeat":name,"completed":len(set(completed)),"total":50,"latest":r["result"]}),flush=True)
            except Exception as e:failed.append(name);fail_rows.append({"job":name,"error":repr(e)});print(json.dumps({"failed":name,"error":repr(e)}),flush=True)
    pd.DataFrame(fail_rows,columns=["job","error"]).to_csv(out/"FAILURES.csv",index=False)
    if failed:raise RuntimeError(f"MINI_TECHNICAL_BLOCK failed jobs: {sorted(set(failed))}")
    _state(state,"S2_COMPLETE",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=list(set(completed)),failed=[])
    _state(state,"S3_COMPLETE",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=list(set(completed)),failed=[])
    results,subject,summary,qframe=_aggregate(repo,config);_state(state,"STATISTICS_COMPLETE",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=list(set(completed)),failed=[])
    summary["pass"]=[_pass(r) for r in summary.itertuples()];winners=[]
    summary.to_csv(out/"MINI_SUMMARY.csv",index=False)
    for scheme in (S2,S3):
        x=summary[summary.scheme==scheme];
        if len(x)==2 and x["pass"].all():winners.append(scheme)
    verdict="MINI_CONTINUE_TO_FULL_METHOD" if winners else "MINI_STOP_FEWSHOT_FUTURE_CRITICAL_INDEX"
    decision={"verdict":verdict,"passing_schemes":winners,"formal_calibration_opened":False,"internal_final_opened":False,"cap_opened":False,"s4_run":False,"active_acquisition_run":False,"full_method_entered":False,"runtime_seconds":time.time()-started}
    _atomic_json(decision,delivery/"MINI_DECISION.json")
    cols=["dataset","scheme","raw_spearman","raw_mae_improvement","q_mean","mean_seed_violation","worst_seed_violation","max_seed_cp_upper","relative_gain_vs_global","paired_gain_ci_low","oracle_gain_recovered","sentinel_delta","sentinel_transition_rate","q_driver_unstable_fold_rate","evaluation_loo_sign_stable","pass"]
    table=_table(summary[cols]);
    (delivery/"MINI_S2_REPORT.md").write_text("# Mini S2 report\n\n"+_table(summary[summary.scheme==S2][cols])+"\n",encoding="utf-8")
    (delivery/"MINI_S3_REPORT.md").write_text("# Mini S3 report\n\n"+_table(summary[summary.scheme==S3][cols])+"\n",encoding="utf-8")
    (delivery/"MINI_COMPARISON.md").write_text("# Mini S2/S3 comparison\n\n"+table+"\n",encoding="utf-8")
    (delivery/"MINI_Q_DRIVER_ANALYSIS.md").write_text("# Mini q-driver analysis\n\n"+_table(qframe.groupby(["dataset","scheme"],as_index=False).agg(q_drop=("q_drop","mean"),gain_change=("gain_change","mean"),sentinel_drop=("sentinel_drop","mean"),unstable_fold_rate=("unstable","mean")))+"\n",encoding="utf-8")
    (delivery/"MINI_DECISION.md").write_text(f"# Mini decision\n\n`{verdict}`\n\n{table}\n\nNo threshold was relaxed. Formal calibration, internal final, CAP, S4, active acquisition, and the full method were not opened.\n",encoding="utf-8")
    (delivery/"LIMITATIONS.md").write_text("# Limitations\n\nOne frozen backbone, two development datasets, temporal b=20 only. This closed diagnostic does not establish formal held-out generalization.\n",encoding="utf-8")
    (delivery/"REPRODUCE.md").write_text("# Reproduce\n\n```bash\ncd /root/autodl-tmp/hsc_tta_eeg\n/root/miniconda3/envs/hsc_gpu/bin/python repo/scripts/budgeted_risk_v51_mini/run_mini.py --project-root /root/autodl-tmp/hsc_tta_eeg --resume\n```\n",encoding="utf-8")
    _state(state,"DECISION_COMPLETE",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=list(set(completed)),failed=[])
    manifest=[]
    for rootdir in (delivery,out):
        for p in sorted(x for x in rootdir.rglob("*") if x.is_file() and x.name!="DELIVERY_MANIFEST.json"):manifest.append({"path":str(p.relative_to(repo)),"sha256":_sha(p),"bytes":p.stat().st_size})
    _atomic_json({"verdict":verdict,"files":manifest},delivery/"DELIVERY_MANIFEST.json")
    _state(state,"DELIVERY_COMPLETE",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=list(set(completed)),failed=[])
    _state(state,"STOPPED",commit=commit,hashes=hashes,config_hash=_sha(config_path),completed=list(set(completed)),failed=[])
    return verdict
