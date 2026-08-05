from __future__ import annotations

import json
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel,delayed,parallel_backend
from hsc_tta.contextual_risk.features import context_features
from hsc_tta.contextual_risk.families import TPSFamily
from hsc_tta.contextual_risk.io import atomic_parquet
from hsc_tta.contextual_risk.quantiles import split_conformal_upper
from hsc_tta.contextual_risk.statistics import clopper_pearson_upper,paired_bootstrap_ci

from .access import BudgetedAccessController
from .acquisition import acquisition_order
from .inclusion_index import critical_index_from_kappa,inclusion_index_table,inclusion_indices,risk_index_entropy
from .query_oracle import QueryOracle
from .stage0 import _fit,_predict,_select_model


TRANSFER_SPECS=("direct","isotonic","ridge_0.01","ridge_0.1","ridge_1","ridge_10","ridge_100","ordinal_0.01","ordinal_0.1","ordinal_1","ordinal_10")


@dataclass
class Arrays:
    indices:np.ndarray; probabilities:np.ndarray; embeddings:np.ndarray; labels:np.ndarray
    future_probabilities:np.ndarray; future_labels:np.ndarray; source_hash:str; episode_hash:str
    inclusion_table:np.ndarray|None=field(default=None,repr=False)
    risk_entropy:np.ndarray|None=field(default=None,repr=False)
    context_feature_cache:dict[str,float]|None=field(default=None,repr=False)
    future_j:int|None=field(default=None,repr=False)
    future_sizes:np.ndarray|None=field(default=None,repr=False)


@dataclass
class Observation:
    dataset:str;subject_id:str;seed:int;fold:int;role:str;budget:int;strategy:str;repeat:int
    arrays:Arrays;queried_kappa:np.ndarray;controller:BudgetedAccessController;query_hash:str;transcript:list[dict[str,Any]]


def _load(path:Path)->Arrays:
    with np.load(path,allow_pickle=False) as z:return Arrays(z["context_sample_indices"].astype(int),z["context_probabilities"].astype(float),z["context_embeddings"].astype(float),z["context_labels_guarded"].astype(int),z["future_probabilities"].astype(float),z["future_labels_guarded"].astype(int),str(z["source_model_hash"]),str(z["episode_hash"]))


def _prior(arrays:dict[str,Arrays],meta:list[str])->np.ndarray:
    cdfs=[]
    for subject in meta:
        current=arrays[subject];oracle=QueryOracle("meta",subject,0,current.indices,current.labels,budget=len(current.indices),strategy="meta_global_prior");table=inclusion_index_table(current.probabilities);values=[]
        for position,index in enumerate(current.indices):
            label=oracle.query(int(index),kappa_by_label=table[position]);values.append(table[position,label])
        oracle.freeze();values=np.asarray(values);cdfs.append([np.mean(values<=j) for j in range(21)])
    return np.mean(cdfs,axis=0)


def _observe(dataset:str,subject:str,seed:int,fold:int,role:str,arrays:Arrays,budget:int,strategy:str,repeat:int)->Observation:
    actual=min(budget,len(arrays.indices));order=acquisition_order(strategy,arrays.probabilities,arrays.embeddings,dataset=dataset,seed=seed,subject_id=subject,repeat=repeat)
    oracle=QueryOracle(dataset,subject,seed,arrays.indices,arrays.labels,budget=actual,strategy=strategy);controller=BudgetedAccessController(dataset,subject,seed,role);controller.begin_queries()
    if arrays.inclusion_table is None:arrays.inclusion_table=inclusion_index_table(arrays.probabilities)
    if arrays.risk_entropy is None:arrays.risk_entropy=risk_index_entropy(arrays.probabilities)
    table=arrays.inclusion_table;entropy=arrays.risk_entropy;values=[]
    for position in order[:actual]:
        label=oracle.query(int(arrays.indices[position]),predicted_class=int(arrays.probabilities[position].argmax()),risk_index_entropy=float(entropy[position]),kappa_by_label=table[position]);values.append(int(table[position,label]))
    query_hash=oracle.freeze();controller.freeze_queries(query_hash)
    return Observation(dataset,subject,seed,fold,role,actual,strategy,repeat,arrays,np.asarray(values,int),controller,query_hash,list(oracle.transcript))


def _local_index(prior:np.ndarray,kappa:np.ndarray,tau:float,alpha:float)->int:
    counts=np.asarray([np.sum(kappa<=j) for j in range(21)],float);cdf=(tau*prior+counts)/(tau+len(kappa));legal=np.flatnonzero(cdf>=1-alpha);return int(legal[0]) if len(legal) else 20


def _features(observation:Observation,prior:np.ndarray,tau:float,alpha:float)->dict[str,float]:
    k=observation.queried_kappa;j=_local_index(prior,k,tau,alpha)
    if observation.arrays.context_feature_cache is None:observation.arrays.context_feature_cache=context_features(observation.arrays.probabilities)
    base=observation.arrays.context_feature_cache
    if len(k):quantiles=np.quantile(k,[.5,.8,.9]);std=float(k.std())
    else:
        mass=np.diff(np.r_[0,prior]);expanded=np.repeat(np.arange(21),np.maximum(np.round(mass*1000).astype(int),0));expanded=expanded if len(expanded) else np.asarray([20]);quantiles=np.quantile(expanded,[.5,.8,.9]);std=float(expanded.std())
    return {**base,"j_local":float(j),"budget":float(observation.budget),"local_kappa_q50":float(quantiles[0]),"local_kappa_q80":float(quantiles[1]),"local_kappa_q90":float(quantiles[2]),"local_kappa_std":std}


def _open(observation:Observation,prediction:float,repo:Path,alpha:float,delta:float,tag:str)->int:
    target=repo/"outputs/budgeted_risk/risk_decisions"/tag/observation.dataset/f"fold_{observation.fold}"/f"seed_{observation.seed}"/f"b{observation.budget}_{observation.strategy}_r{observation.repeat}_{observation.subject_id.replace(':','_')}.json"
    payload={"dataset":observation.dataset,"subject_id":observation.subject_id,"seed":observation.seed,"role":observation.role,"budget":observation.budget,"strategy":observation.strategy,"alpha":alpha,"delta":delta,"query_hash":observation.query_hash,"source_model_hash":observation.arrays.source_hash,"episode_hash":observation.arrays.episode_hash,"certified_index":int(np.clip(np.ceil(prediction),0,20))}
    observation.controller.freeze_decision(payload,target);labels=observation.controller.open_future(observation.arrays.future_labels,target)
    if observation.arrays.future_j is None:
        observation.arrays.future_j=critical_index_from_kappa(inclusion_indices(observation.arrays.future_probabilities,labels),alpha)
        _,observation.arrays.future_sizes,_=TPSFamily().future_curve(observation.arrays.future_probabilities,labels)
    return int(observation.arrays.future_j)


def _evaluate(observation:Observation,features:dict[str,float],fitted:dict[str,Any],columns:list[str],correction:float,global_index:int,global_correction:float,true:int)->dict[str,Any]:
    x=pd.DataFrame([features])[columns].to_numpy();raw=float(_predict(fitted,x,np.asarray([features["j_local"]]))[0]);cert=int(np.clip(np.ceil(raw+correction),0,20));global_cert=int(np.clip(np.ceil(global_index+global_correction),0,20));sizes=observation.arrays.future_sizes
    if sizes is None:raise RuntimeError("Future sizes accessed before a frozen decision")
    denominator=max(float(sizes[global_cert]-sizes[true]),1e-12)
    return {"dataset":observation.dataset,"subject_id":observation.subject_id,"seed":observation.seed,"outer_fold":observation.fold,"budget":observation.budget,"strategy":observation.strategy,"repeat":observation.repeat,"j_future":true,"raw_prediction":raw,"certified_index":cert,"global_certified_index":global_cert,"violation":cert<true,"global_violation":global_cert<true,"set_size":float(sizes[cert]),"global_set_size":float(sizes[global_cert]),"oracle_set_size":float(sizes[true]),"relative_set_size_gain":float((sizes[global_cert]-sizes[cert])/max(sizes[global_cert],1e-12)),"oracle_gain_recovered":float((sizes[global_cert]-sizes[cert])/denominator),"sentinel":cert==20,"global_sentinel":global_cert==20,"queried_count":observation.budget,"gain_per_label":float((sizes[global_cert]-sizes[cert])/max(observation.budget,1)),"selected_model":fitted["name"]}


def run_budget_baselines(project_root:str|Path,config:dict[str,Any])->tuple[pd.DataFrame,pd.DataFrame]:
    root=Path(project_root);repo=root/"repo";cohorts=pd.read_parquet(repo/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet");dev=cohorts[cohorts.master_cohort=="method_development"];manifest=pd.read_parquet(repo/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet");alpha=float(config["alpha"]);delta=float(config["delta"]);rows=[];tuning=[];transcripts=[]
    for dataset in ("hmc","eegmmidb"):
        fold_map=dev[dev.dataset==dataset].set_index("subject_id").screening_fold.astype(int).to_dict();subjects=sorted(fold_map)
        for fold in range(5):
            roles={s:("evaluation" if fold_map[s]==fold else "calibration" if fold_map[s]==(fold+1)%5 else "meta") for s in subjects};meta_subjects=[s for s in subjects if roles[s]=="meta"]
            for seed in range(5):
                arrays={s:_load(Path(manifest[(manifest.dataset==dataset)&(manifest.fold==fold)&(manifest.seed==seed)&(manifest.subject_id==s)].iloc[0].cache_path)) for s in subjects};prior=_prior(arrays,meta_subjects);global_index=int(np.flatnonzero(prior>=1-alpha)[0]) if np.any(prior>=1-alpha) else 20
                for requested_budget in (0,1,2,5,10,20,50):
                    base={s:_observe(dataset,s,seed,fold,roles[s],arrays[s],min(requested_budget,len(arrays[s].indices)),"temporal",0) for s in subjects}
                    meta_outcome={s:_open(base[s],_local_index(prior,base[s].queried_kappa,5,alpha),repo,alpha,delta,"stage0_budget_meta") for s in meta_subjects}
                    tau_frames=[]
                    for tau in config["tau_candidates"]:
                        frame=pd.DataFrame([{**_features(base[s],prior,float(tau),alpha),"subject_id":s,"screening_fold":fold_map[s],"j_future":meta_outcome[s]} for s in meta_subjects]);ids={"subject_id","screening_fold","j_future"};columns=sorted(set(frame.columns)-ids);tau_frames.append((float(tau),frame,columns))
                    with parallel_backend("loky",inner_max_num_threads=1):
                        selected_scores=Parallel(n_jobs=len(tau_frames))(delayed(_select_model)(frame,columns,TRANSFER_SPECS,index_column="j_local") for _,frame,columns in tau_frames)
                    all_scores=[]
                    for (tau,frame,columns),(selected,scores) in zip(tau_frames,selected_scores,strict=True):
                        score=scores[scores.candidate==selected].iloc[0];all_scores.append((float(score.mae),-float(score.spearman),float(score.underestimation_rate),tau,selected,columns,frame,scores))
                    best=min(all_scores,key=lambda item:(item[0],item[1],item[2],item[3],item[4]));_,_,_,tau,selected,columns,meta_frame,scores=best;fitted=_fit(selected,meta_frame[columns].to_numpy(),meta_frame.j_future.to_numpy(),meta_frame.j_local.to_numpy())
                    tuning.append({"dataset":dataset,"outer_fold":fold,"seed":seed,"budget":requested_budget,"tau":tau,"selected_model":selected,"inner_mae":best[0]})
                    cal_subjects=[s for s in subjects if roles[s]=="calibration"];cal_pred=[];cal_true=[]
                    for s in cal_subjects:
                        features=_features(base[s],prior,tau,alpha);prediction=float(_predict(fitted,pd.DataFrame([features])[columns].to_numpy(),np.asarray([features["j_local"]]))[0]);true=_open(base[s],prediction,repo,alpha,delta,"stage0_budget_cal");cal_pred.append(prediction);cal_true.append(true)
                    correction=split_conformal_upper(np.maximum(np.asarray(cal_true)-np.asarray(cal_pred),0),delta,insufficient=20);global_correction=split_conformal_upper(np.maximum(np.asarray(cal_true)-global_index,0),delta,insufficient=20)
                    for s in [x for x in subjects if roles[x]=="evaluation"]:
                        features=_features(base[s],prior,tau,alpha);raw=float(_predict(fitted,pd.DataFrame([features])[columns].to_numpy(),np.asarray([features["j_local"]]))[0]);true=_open(base[s],raw+correction,repo,alpha,delta,"stage0_budget_eval");rows.append(_evaluate(base[s],features,fitted,columns,correction,global_index,global_correction,true));transcripts.extend(base[s].transcript)
                    # Random repeats use the exact same frozen estimator and
                    # one-sided calibration as temporal; only evaluation
                    # acquisition changes, preventing estimator/acquisition
                    # confounding.
                    for repeat in range(int(config["random_repeats"])):
                        evaluation_subjects=[x for x in subjects if roles[x]=="evaluation"]
                        random_obs={s:_observe(dataset,s,seed,fold,roles[s],arrays[s],min(requested_budget,len(arrays[s].indices)),"random",repeat) for s in evaluation_subjects}
                        for s in evaluation_subjects:
                            features=_features(random_obs[s],prior,tau,alpha);raw=float(_predict(fitted,pd.DataFrame([features])[columns].to_numpy(),np.asarray([features["j_local"]]))[0]);true=_open(random_obs[s],raw+correction,repo,alpha,delta,"stage0_random_eval");rows.append(_evaluate(random_obs[s],features,fitted,columns,correction,global_index,global_correction,true));transcripts.extend(random_obs[s].transcript)
    frame=pd.DataFrame(rows);tuning_frame=pd.DataFrame(tuning);output=repo/"outputs/budgeted_risk/stage0";atomic_parquet(frame,output/"BUDGET_RESULTS.parquet");atomic_parquet(tuning_frame,output/"BUDGET_TUNING.parquet");budget_transcripts=pd.DataFrame(transcripts);atomic_parquet(budget_transcripts,output/"BUDGET_QUERY_TRANSCRIPTS.parquet");return frame,tuning_frame


def budget_gate(results:pd.DataFrame,config:dict[str,Any])->tuple[pd.DataFrame,bool]:
    temporal=results[results.strategy=="temporal"];rows=[];gate=config["budget_gate"]
    for (dataset,budget),current in temporal[temporal.budget.isin(gate["candidate_budgets"])].groupby(["dataset","budget"]):
        by_subject=current.groupby("subject_id",as_index=False).agg(set_size=("set_size","mean"),global_set_size=("global_set_size","mean"),oracle_set_size=("oracle_set_size","mean"),sentinel=("sentinel","mean"),global_sentinel=("global_sentinel","mean"));n=len(current);violations=int(current.violation.sum());violation=violations/n;cp=clopper_pearson_upper(violations,n,.95);worst=float(current.groupby("seed").violation.mean().max());gain=by_subject.global_set_size-by_subject.set_size;ci=paired_bootstrap_ci(gain.to_numpy(),reps=int(config["bootstrap_repetitions"]),seed=int(config["bootstrap_seed"]));recovery=float(gain.sum()/max((by_subject.global_set_size-by_subject.oracle_set_size).sum(),1e-12));relative=float(gain.mean()/max(by_subject.global_set_size.mean(),1e-12));sentinel_delta=float(by_subject.sentinel.mean()-by_subject.global_sentinel.mean());passed=bool(violation<=gate["maximum_violation"] and cp<=gate["maximum_cp_upper"] and worst<=gate["maximum_worst_seed_violation"] and relative>gate["minimum_relative_gain"] and ci[0]>=0 and recovery>=gate["minimum_oracle_recovery"] and sentinel_delta<=gate["maximum_sentinel_increase"])
        rows.append({"dataset":dataset,"budget":budget,"violation":violation,"cp_upper":cp,"worst_seed_violation":worst,"relative_set_size_gain":relative,"paired_gain_ci_low":ci[0],"paired_gain_ci_high":ci[1],"oracle_gain_recovered":recovery,"sentinel_rate_delta":sentinel_delta,"budget_pass":passed})
    summary=pd.DataFrame(rows);return summary,bool(len(summary) and all(summary.groupby("dataset").budget_pass.any()))
