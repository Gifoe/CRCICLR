from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mord
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

from .families import APSFamily, GRID, RAPSFamily, TPSFamily, critical_index
from .features import SIGNATURE_COLUMNS, context_features
from .quantiles import higher_quantile, split_conformal_upper
from .statistics import clopper_pearson_upper, paired_bootstrap_ci


def build_shared_tables(project_root: str | Path, cohorts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(project_root)
    feature_rows: list[dict[str, Any]] = []
    surface_rows: list[dict[str, Any]] = []
    membership = cohorts.set_index(["dataset", "subject_id"])
    family_factories = [
        ("TPS", TPSFamily()), ("APS", APSFamily()),
        *[(f"RAPS-k{k}-l{lam:.2f}", RAPSFamily(k, lam)) for k in (1,2,3) for lam in (.01,.05,.10)],
    ]
    for dataset in ("hmc", "eegmmidb"):
        for seed in range(5):
            cache_dir = root / "repo/outputs/contextual_risk/source_cache" / dataset / f"seed_{seed}"
            for path in sorted(cache_dir.glob("*.npz")):
                with np.load(path, allow_pickle=False) as z:
                    subject = path.stem.replace(f"{dataset}_", f"{dataset}:", 1)
                    context_p = z["context_probabilities"].astype(float)
                    episode_hash, model_hash = str(z["episode_hash"]), str(z["source_model_hash"])
                meta = membership.loc[(dataset, subject)]
                features = context_features(context_p)
                context_curve = TPSFamily().context_sizes(context_p)[:-1]
                features.update({
                    "dataset": dataset, "seed": seed, "subject_id": subject,
                    "master_cohort": meta.master_cohort, "screening_fold": int(meta.screening_fold),
                    "episode_hash": episode_hash, "source_model_hash": model_hash,
                    "tps_context_area": float(context_curve.mean()),
                })
                feature_rows.append(features)
                # Formal-calibration and internal-final Future fields remain
                # closed until a full method is selected and frozen. Offline
                # source caches may contain them, but screening never loads or
                # derives outcome surfaces from those cache members.
                if meta.master_cohort != "method_development":
                    continue
                with np.load(path, allow_pickle=False) as z:
                    future_p = z["future_probabilities"].astype(float)
                    labels = z["future_labels"].astype(int)
                for family_name, family in family_factories:
                    risks, sizes, repairs = family.future_curve(future_p, labels)
                    half = len(labels) // 2
                    risk_1, _, _ = family.future_curve(future_p[:half], labels[:half])
                    risk_2, _, _ = family.future_curve(future_p[half:], labels[half:])
                    for alpha in (.10, .20):
                        row: dict[str, Any] = {
                            "dataset": dataset, "seed": seed, "subject_id": subject,
                            "master_cohort": meta.master_cohort, "screening_fold": int(meta.screening_fold),
                            "family": family_name, "alpha": alpha,
                            "critical_index": critical_index(risks, alpha),
                            "critical_index_first_half": critical_index(risk_1, alpha),
                            "critical_index_second_half": critical_index(risk_2, alpha),
                            "monotonic_repairs": repairs, "episode_hash": episode_hash,
                            "source_model_hash": model_hash,
                        }
                        for j in range(21):
                            row[f"risk_j{j}"] = float(risks[j])
                            row[f"size_j{j}"] = float(sizes[j])
                        surface_rows.append(row)
    return pd.DataFrame(feature_rows), pd.DataFrame(surface_rows)


def _feature_columns(frame: pd.DataFrame, view: str) -> list[str]:
    blocked = {"dataset","seed","subject_id","master_cohort","screening_fold","episode_hash","source_model_hash"}
    numeric = [c for c in frame.columns if c not in blocked and pd.api.types.is_numeric_dtype(frame[c])]
    signature = [c for c in SIGNATURE_COLUMNS if c in numeric]
    basic = [c for c in numeric if c not in signature]
    return basic if view == "basic" else signature if view == "signature" else basic + signature


def _scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(x, axis=0)
    iqr = np.quantile(x, .75, axis=0) - np.quantile(x, .25, axis=0)
    return median, np.where(iqr > 1e-8, iqr, 1.0)


def _reg_predict(kind: str, param: float, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if kind == "constant" or len(np.unique(y_train)) == 1:
        return np.full(len(x_test), np.median(y_train))
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(x_train[:, 0], y_train)
        return model.predict(x_test[:, 0])
    median, scale = _scale_fit(x_train); a=(x_train-median)/scale; b=(x_test-median)/scale
    if kind == "ridge":
        return Ridge(alpha=param).fit(a, y_train).predict(b)
    if kind == "ordinal":
        ordered_classes = np.sort(np.unique(y_train.astype(int))).astype(float)
        encoded = np.searchsorted(ordered_classes, y_train.astype(int))
        model = mord.LogisticAT(alpha=max(float(param), 1e-8), max_iter=2000).fit(a, encoded)
        probabilities = model.predict_proba(b)
        return probabilities @ ordered_classes
    if kind == "knn":
        k = min(int(param), len(y_train))
        return KNeighborsRegressor(k).fit(a, y_train).predict(b)
    raise ValueError(kind)


def _reg_candidates(n_train: int) -> list[tuple[str,str,float,int]]:
    candidates=[("constant","basic",0,0), ("isotonic","area",0,1)]
    for view in ("basic","signature","combined"):
        candidates += [("ridge",view,a,2) for a in (.01,.1,1,10,100)]
        candidates += [("ordinal",view,a,3) for a in (.001,.01,.1,1)]
        candidates += [("knn",view,k,4) for k in (3,5,7,11) if k < n_train]
    return candidates


def _matrix(features: pd.DataFrame, subjects: list[str], view: str) -> np.ndarray:
    rows = features.set_index("subject_id").loc[subjects]
    if view == "area":
        return rows[["tps_context_area"]].to_numpy(float)
    # Concatenating datasets introduces an absent-class column for the
    # four-class EEGMMIDB task. An absent predicted class has proportion zero;
    # no data-dependent imputation is involved.
    return rows[_feature_columns(features, view)].fillna(0.0).to_numpy(float)


def _select_regressor(features: pd.DataFrame, targets: pd.Series, train_subjects: list[str]) -> tuple[str,str,float,int,dict[str,float]]:
    folds = features.set_index("subject_id").screening_fold.to_dict()
    scores=[]
    for candidate in _reg_candidates(len(train_subjects)):
        kind,view,param,simplicity=candidate; truth=[]; predictions=[]
        for fold in sorted(set(folds[s] for s in train_subjects)):
            va=[s for s in train_subjects if folds[s]==fold]; tr=[s for s in train_subjects if folds[s]!=fold]
            if not va or len(tr)<3: continue
            pred=_reg_predict(kind,param,_matrix(features,tr,view),targets.loc[tr].to_numpy(float),_matrix(features,va,view))
            truth.extend(targets.loc[va].to_numpy(float)); predictions.extend(pred)
        if not truth: continue
        mae=float(np.mean(np.abs(np.asarray(truth)-predictions)))
        rho=float(spearmanr(truth,predictions).statistic) if np.std(predictions)>0 and np.std(truth)>0 else 0.0
        scores.append((mae,-rho,simplicity,kind,view,param,rho))
    scores.sort()
    best_mae=scores[0][0]
    near=[s for s in scores if s[0] <= best_mae + .05]
    chosen=min(near, key=lambda s:(s[1],s[2],s[3],s[4],s[5]))
    return chosen[3],chosen[4],chosen[5],chosen[2],{"inner_mae":chosen[0],"inner_spearman":chosen[6]}


def _classifier_predict(kind: str, param: float, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, balanced: bool=True) -> np.ndarray:
    majority=Counter(y_train).most_common(1)[0][0]
    if kind=="majority" or len(np.unique(y_train))==1: return np.full(len(x_test),majority,dtype=object)
    median,scale=_scale_fit(x_train); a=(x_train-median)/scale; b=(x_test-median)/scale
    if kind=="logistic":
        return LogisticRegression(C=param,class_weight="balanced" if balanced else None,max_iter=2000).fit(a,y_train).predict(b)
    if kind=="tree":
        return DecisionTreeClassifier(max_depth=2,min_samples_leaf=int(param),class_weight="balanced" if balanced else None,random_state=20260805).fit(a,y_train).predict(b)
    if kind=="knn":
        return KNeighborsClassifier(min(int(param),len(y_train)),weights="uniform").fit(a,y_train).predict(b)
    raise ValueError(kind)


def _class_candidates(n_train:int)->list[tuple[str,str,float,int]]:
    out=[("majority","basic",0,0)]
    for view in ("basic","signature","combined"):
        out += [("logistic",view,c,1) for c in (.01,.1,1,10)]
        out += [("tree",view,m,2) for m in (5,10,20) if m < n_train]
        out += [("knn",view,k,3) for k in (3,5,7,11) if k < n_train]
    return out


def _select_classifier(features:pd.DataFrame,labels:pd.Series,train_subjects:list[str])->tuple[str,str,float,int,dict[str,float]]:
    folds=features.set_index("subject_id").screening_fold.to_dict(); scores=[]
    for kind,view,param,simplicity in _class_candidates(len(train_subjects)):
        truth=[];pred=[]
        for fold in sorted(set(folds[s] for s in train_subjects)):
            va=[s for s in train_subjects if folds[s]==fold];tr=[s for s in train_subjects if folds[s]!=fold]
            if not va or len(tr)<3: continue
            current=_classifier_predict(kind,param,_matrix(features,tr,view),labels.loc[tr].to_numpy(object),_matrix(features,va,view))
            truth.extend(labels.loc[va]);pred.extend(current)
        if not truth: continue
        acc=accuracy_score(truth,pred); bal=balanced_accuracy_score(truth,pred)
        scores.append((-acc,-bal,simplicity,kind,view,param,acc,bal))
    scores.sort(); chosen=scores[0]
    return chosen[3],chosen[4],chosen[5],chosen[2],{"inner_accuracy":chosen[6],"inner_balanced_accuracy":chosen[7]}


def _aggregate_seed_rows(frame:pd.DataFrame,value_columns:list[str])->pd.DataFrame:
    keys=["dataset","subject_id","alpha"]
    agg={c:"mean" for c in value_columns}; agg.update({"screening_fold":"first"})
    return frame.groupby(keys,as_index=False).agg(agg)


def run_screening_a(features:pd.DataFrame,surfaces:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    rows=[]
    for dataset in ("hmc","eegmmidb"):
        fds=features[(features.dataset==dataset)&(features.master_cohort=="method_development")]
        for seed in range(5):
            fs=fds[fds.seed==seed].copy()
            for alpha in (.10,.20):
                surface=surfaces[(surfaces.dataset==dataset)&(surfaces.seed==seed)&(surfaces.alpha==alpha)&(surfaces.family=="TPS")&(surfaces.master_cohort=="method_development")].set_index("subject_id")
                targets=surface.critical_index
                for rotation in range(5):
                    eval_s=fs[fs.screening_fold==rotation].subject_id.tolist(); cal_s=fs[fs.screening_fold==(rotation+1)%5].subject_id.tolist(); meta_s=fs[~fs.screening_fold.isin([rotation,(rotation+1)%5])].subject_id.tolist()
                    chosen=_select_regressor(fs,targets,meta_s);kind,view,param,_,inner=chosen
                    raw=_reg_predict(kind,param,_matrix(fs,meta_s,view),targets.loc[meta_s].to_numpy(float),_matrix(fs,eval_s,view))
                    raw_cal=_reg_predict(kind,param,_matrix(fs,meta_s,view),targets.loc[meta_s].to_numpy(float),_matrix(fs,cal_s,view))
                    q=split_conformal_upper(np.maximum(targets.loc[cal_s].to_numpy(float)-raw_cal,0),.10,insufficient=20)
                    global_j=int(split_conformal_upper(targets.loc[cal_s],.10,insufficient=20))
                    for subject,pred in zip(eval_s,raw,strict=True):
                        true_j=int(targets.loc[subject]); cert=int(np.clip(math.ceil(pred+q),0,20)); sr=surface.loc[subject]
                        rows.append({"dataset":dataset,"seed":seed,"subject_id":subject,"screening_fold":rotation,"alpha":alpha,"model_kind":kind,"feature_view":view,"model_param":param,**inner,"raw_prediction":float(pred),"true_index":true_j,"certified_index":cert,"global_index":global_j,"oracle_size":float(sr[f"size_j{true_j}"]),"method_size":float(sr[f"size_j{cert}"]),"global_size":float(sr[f"size_j{global_j}"]),"method_risk":float(sr[f"risk_j{cert}"]),"global_risk":float(sr[f"risk_j{global_j}"]),"violation":int(sr[f"risk_j{cert}"]>alpha),"global_violation":int(sr[f"risk_j{global_j}"]>alpha),"method_sentinel":int(cert==20),"global_sentinel":int(global_j==20),"half_one_index":int(sr.critical_index_first_half),"half_two_index":int(sr.critical_index_second_half)})
    result=pd.DataFrame(rows)
    summary=evaluate_a_gate(result)
    return result,summary


def evaluate_a_gate(result:pd.DataFrame)->dict[str,Any]:
    primary=result[result.alpha==.10].copy(); per_dataset={}; all_go=True
    for dataset,group in primary.groupby("dataset"):
        g=_aggregate_seed_rows(group,["raw_prediction","true_index","oracle_size","method_size","global_size","method_risk","violation","global_violation","method_sentinel","global_sentinel","half_one_index","half_two_index"])
        rel_oracle=(g.global_size-g.oracle_size)/np.maximum(g.global_size,1e-12)
        realized=(g.global_size-g.method_size)/np.maximum(g.global_size,1e-12)
        rho=float(spearmanr(g.true_index,g.raw_prediction).statistic) if g.true_index.std()>0 and g.raw_prediction.std()>0 else 0.0
        reliability=float(spearmanr(g.half_one_index,g.half_two_index).statistic) if g.half_one_index.std()>0 and g.half_two_index.std()>0 else 0.0
        kappa=float(cohen_kappa_score(np.rint(g.half_one_index),np.rint(g.half_two_index),weights="linear"))
        constant_mae=float(np.mean(np.abs(g.true_index-np.median(g.true_index)))); mae=float(np.mean(np.abs(g.true_index-g.raw_prediction)))
        ci_oracle=paired_bootstrap_ci(g.global_size-g.oracle_size); ci_realized=paired_bootstrap_ci(g.global_size-g.method_size)
        violations=int((g.method_risk>.10).sum()); n=len(g); cp=clopper_pearson_upper(violations,n)
        worst=float(group.groupby("seed").violation.mean().max())
        sensitivity=result[(result.dataset==dataset)&(result.alpha==.20)]
        sensitivity_gain=float(((sensitivity.global_size-sensitivity.method_size)/np.maximum(sensitivity.global_size,1e-12)).mean())
        gates={"oracle_gain":float(rel_oracle.mean())>=.10,"oracle_ci":ci_oracle[0]>0,"oracle_positive":float((rel_oracle>0).mean())>.50,"reliability":reliability>=.45 or kappa>=.40,"predictor_rho":rho>=.30,"predictor_mae":constant_mae>0 and (constant_mae-mae)/constant_mae>=.10,"validity":violations/n<=.10 and cp<=.20 and worst<=.20,"realized_gain":float(realized.mean())>=.05,"realized_ci":ci_realized[0]>0,"sentinel":float(g.method_sentinel.mean())<=float(g.global_sentinel.mean())+.02,"alpha20_direction":sensitivity_gain>=0}
        go=all(gates.values());all_go &= go
        per_dataset[dataset]={"go":go,"n_subjects":n,"oracle_relative_gain":float(rel_oracle.mean()),"oracle_ci":ci_oracle,"positive_rate":float((rel_oracle>0).mean()),"target_reliability_spearman":reliability,"target_reliability_kappa":kappa,"predictor_spearman":rho,"predictor_mae":mae,"constant_mae":constant_mae,"violation_rate":violations/n,"clopper_pearson_upper":cp,"worst_seed_violation":worst,"realized_relative_gain":float(realized.mean()),"realized_ci":ci_realized,"alpha20_gain":sensitivity_gain,"gates":gates}
    return {"decision":"A_GO" if all_go else "A_NO_GO","datasets":per_dataset}


def _choose_raps(surface:pd.DataFrame,meta_s:list[str])->str:
    candidates=sorted([x for x in surface.family.unique() if str(x).startswith("RAPS")])
    scored=[]
    for family in candidates:
        g=surface[(surface.family==family)&(surface.index.isin(meta_s))]
        sizes=[float(r[f"size_j{int(r.critical_index)}"]) for _,r in g.iterrows()]
        scored.append((np.mean(sizes),family))
    return min(scored)[1]


def _policy_library(surface:pd.DataFrame,meta_s:list[str],raps_name:str)->dict[str,tuple[str,int]]:
    policies={}
    for family in ("TPS","APS",raps_name):
        values=surface[(surface.family==family)&(surface.index.isin(meta_s))].critical_index
        previous=0
        for tier,q in (("efficient",.35),("moderate",.65),("conservative",.90)):
            index=max(previous,int(higher_quantile(values,q))); previous=index
            policies[f"{family}|{tier}"]=(family,index)
    policies["FULL_SET_FALLBACK"]=("TPS",20)
    return policies


def _oracle_policy(subject:str,surface:pd.DataFrame,policies:dict[str,tuple[str,int]])->tuple[str,float]:
    legal=[]
    for policy,(family,index) in policies.items():
        row=surface[(surface.index==subject)&(surface.family==family)].iloc[0]
        if index>=int(row.critical_index): legal.append((float(row[f"size_j{index}"]),policy))
    return min(legal,key=lambda x:(x[0],x[1]))


def run_screening_b(features:pd.DataFrame,surfaces:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    rows=[]
    for dataset in ("hmc","eegmmidb"):
        fds=features[(features.dataset==dataset)&(features.master_cohort=="method_development")]
        for seed in range(5):
            fs=fds[fds.seed==seed].copy()
            for alpha in (.10,.20):
                surface=surfaces[(surfaces.dataset==dataset)&(surfaces.seed==seed)&(surfaces.alpha==alpha)&(surfaces.master_cohort=="method_development")].set_index("subject_id",drop=False)
                for rotation in range(5):
                    eval_s=fs[fs.screening_fold==rotation].subject_id.tolist();cal_s=fs[fs.screening_fold==(rotation+1)%5].subject_id.tolist();meta_s=fs[~fs.screening_fold.isin([rotation,(rotation+1)%5])].subject_id.tolist()
                    raps=_choose_raps(surface,meta_s);policies=_policy_library(surface,meta_s,raps)
                    oracle={s:_oracle_policy(s,surface,policies)[1] for s in meta_s+cal_s+eval_s}
                    labels=pd.Series(oracle)
                    kind,view,param,_,inner=_select_classifier(fs,labels,meta_s)
                    pred=_classifier_predict(kind,param,_matrix(fs,meta_s,view),labels.loc[meta_s].to_numpy(object),_matrix(fs,eval_s,view))
                    pred_cal=_classifier_predict(kind,param,_matrix(fs,meta_s,view),labels.loc[meta_s].to_numpy(object),_matrix(fs,cal_s,view))
                    deficits=[]
                    for subject,policy in zip(cal_s,pred_cal,strict=True):
                        family,index=policies[str(policy)]; true=int(surface[(surface.index==subject)&(surface.family==family)].iloc[0].critical_index);deficits.append(max(true-index,0))
                    q=int(split_conformal_upper(deficits,.10,insufficient=20))
                    fixed=Counter(labels.loc[meta_s]).most_common(1)[0][0]
                    fixed_def=[]
                    ff,fi=policies[fixed]
                    for subject in cal_s:
                        true=int(surface[(surface.index==subject)&(surface.family==ff)].iloc[0].critical_index);fixed_def.append(max(true-fi,0))
                    fixed_q=int(split_conformal_upper(fixed_def,.10,insufficient=20))
                    for subject,policy in zip(eval_s,pred,strict=True):
                        family,index=policies[str(policy)];cert=min(20,index+q);sr=surface[(surface.index==subject)&(surface.family==family)].iloc[0]
                        oracle_policy=oracle[subject]; of,oi=policies[oracle_policy]; osr=surface[(surface.index==subject)&(surface.family==of)].iloc[0]
                        fcert=min(20,fi+fixed_q);fsr=surface[(surface.index==subject)&(surface.family==ff)].iloc[0]
                        rows.append({"dataset":dataset,"seed":seed,"subject_id":subject,"screening_fold":rotation,"alpha":alpha,"model_kind":kind,"feature_view":view,"model_param":param,**inner,"predicted_policy":str(policy),"oracle_policy":oracle_policy,"fixed_policy":fixed,"raps_config":raps,"certified_index":cert,"fixed_index":fcert,"oracle_index":oi,"method_size":float(sr[f"size_j{cert}"]),"method_risk":float(sr[f"risk_j{cert}"]),"fixed_size":float(fsr[f"size_j{fcert}"]),"fixed_risk":float(fsr[f"risk_j{fcert}"]),"oracle_size":float(osr[f"size_j{oi}"]),"violation":int(sr[f"risk_j{cert}"]>alpha),"method_sentinel":int(cert==20),"fixed_sentinel":int(fcert==20),"selector_correct":int(str(policy)==oracle_policy)})
    result=pd.DataFrame(rows);return result,evaluate_b_gate(result)


def evaluate_b_gate(result:pd.DataFrame)->dict[str,Any]:
    primary=result[result.alpha==.10];per_dataset={};all_go=True
    for dataset,group in primary.groupby("dataset"):
        g=_aggregate_seed_rows(group,["method_size","fixed_size","oracle_size","method_risk","violation","method_sentinel","fixed_sentinel","selector_correct"])
        oracle=(g.fixed_size-g.oracle_size)/np.maximum(g.fixed_size,1e-12);realized=(g.fixed_size-g.method_size)/np.maximum(g.fixed_size,1e-12)
        ci_oracle=paired_bootstrap_ci(g.fixed_size-g.oracle_size);ci_realized=paired_bootstrap_ci(g.fixed_size-g.method_size)
        majority=float(group.groupby("seed").oracle_policy.apply(lambda x:x.value_counts(normalize=True).max()).mean())
        accuracy=float(g.selector_correct.mean());recovered=float(realized.mean()/max(oracle.mean(),1e-12))
        wins=group.oracle_policy.value_counts(normalize=True);nonfallback=wins[~wins.index.str.contains("FULL")]
        violations=int((g.method_risk>.10).sum());n=len(g);cp=clopper_pearson_upper(violations,n);worst=float(group.groupby("seed").violation.mean().max())
        sens=result[(result.dataset==dataset)&(result.alpha==.20)];sens_gain=float(((sens.fixed_size-sens.method_size)/np.maximum(sens.fixed_size,1e-12)).mean())
        gates={"oracle_gain":float(oracle.mean())>=.10,"oracle_ci":ci_oracle[0]>0,"oracle_positive":float((oracle>0).mean())>.50,"policy_diversity":(float(nonfallback.max()) if len(nonfallback) else 1)<.85 and int((nonfallback>=.10).sum())>=2,"selector":accuracy>=majority+.10 and recovered>=.40,"validity":violations/n<=.10 and cp<=.20 and worst<=.20,"realized_gain":float(realized.mean())>=.05,"realized_ci":ci_realized[0]>0,"sentinel":float(g.method_sentinel.mean())<=float(g.fixed_sentinel.mean())+.02,"alpha20_direction":sens_gain>=0}
        go=all(gates.values());all_go&=go
        per_dataset[dataset]={"go":go,"n_subjects":n,"oracle_relative_gain":float(oracle.mean()),"oracle_ci":ci_oracle,"positive_rate":float((oracle>0).mean()),"max_nonfallback_win_rate":float(nonfallback.max()) if len(nonfallback) else 0.0,"nonfallback_policies_ge_10pct":int((nonfallback>=.10).sum()),"selector_accuracy":accuracy,"majority_accuracy":majority,"recovered_oracle_fraction":recovered,"violation_rate":violations/n,"clopper_pearson_upper":cp,"worst_seed_violation":worst,"realized_relative_gain":float(realized.mean()),"realized_ci":ci_realized,"alpha20_gain":sens_gain,"gates":gates,"policy_wins":wins.to_dict()}
    return {"decision":"B_GO" if all_go else "B_NO_GO","datasets":per_dataset}
