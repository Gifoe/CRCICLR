from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr, binomtest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors

import common as c


EPS=1e-12
ALPHA_GRID=np.arange(17,dtype=np.float64)/64.0


def verify_lock() -> dict:
    path=c.PROTOCOL/"FM_RESCUE_STAGE0_PROTOCOL_LOCK.json"
    subprocess.run(["git","ls-files","--error-unmatch",str(path.relative_to(c.REPO))],cwd=c.REPO,check=True,capture_output=True)
    if subprocess.check_output(["git","status","--porcelain","--",str(path.relative_to(c.REPO))],cwd=c.REPO,text=True).strip(): raise RuntimeError("protocol lock dirty")
    lock=c.read_json(path)
    if not lock.get("frozen_before_primary_outcomes"): raise RuntimeError("not frozen")
    for rel,expected in lock["code_hashes"].items():
        if c.sha256(c.EXP/rel)!=expected: raise RuntimeError(f"post-freeze code change: {rel}")
    return lock


def rep_path(fm:str,dataset:str,fold:int,seed:int,role:str)->Path:
    return c.RUNTIME/"representations"/fm/dataset/f"fold-{fold}"/f"seed-{seed}"/f"{role}.npz"


def save_rep(path:Path,value:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(".npz.part")
    # Subject identifiers arrive from pandas as object arrays.  Persist them as
    # fixed-width Unicode so the cache remains loadable with allow_pickle=False.
    safe={}
    for key,item in value.items():
        array=np.asarray(item)
        safe[key]=array.astype(str) if array.dtype.kind=="O" else array
    with temp.open("wb") as f: np.savez_compressed(f,**safe)
    temp.replace(path)


def load_rep(fm:str,dataset:str,fold:int,seed:int,role:str)->dict:
    z=np.load(rep_path(fm,dataset,fold,seed,role),allow_pickle=False); return {k:z[k] for k in z.files}


def build_representations(lock:dict)->pd.DataFrame:
    device=torch.device("cuda"); rows=[]
    for fm in c.FMS:
        for dataset in c.DATASETS:
            data=c.load_data(dataset); source_sessions=c.SOURCE_SESSIONS[dataset]
            for fold in c.FOLDS:
                roles=c.fold_roles(dataset,fold)
                indices={"model_fit":c.row_indices(data.metadata,roles["model_fit"],source_sessions),"validation":c.row_indices(data.metadata,roles["validation"],source_sessions),"outcome":c.row_indices(data.metadata,roles["outcome"],source_sessions)}
                if dataset=="WBCIC": indices["outcome_all"]=c.row_indices(data.metadata,roles["outcome"],(0,1,2))
                for seed in c.SEEDS:
                    model=c.load_anchor(fm,dataset,fold,seed,device)
                    for role,idx in indices.items():
                        path=rep_path(fm,dataset,fold,seed,role)
                        if not path.is_file(): save_rep(path,c.infer(model,dataset,idx,device))
                    out=load_rep(fm,dataset,fold,seed,"outcome"); per=[]
                    for subject in c.subject_sort(np.unique(out["subjects"].astype(str))):
                        m=out["subjects"].astype(str)==subject; per.append(balanced_accuracy_score(out["labels"][m],out["logits"][m].argmax(1)))
                    rows.append({"dataset":dataset,"model":fm,"fold":fold,"seed":seed,"task_BA":float(np.mean(per)),"subjects":len(per),"competence_threshold":c.COMPETENCE_THRESHOLDS[dataset],"competent":float(np.mean(per))>=c.COMPETENCE_THRESHOLDS[dataset],"target_seen_by_anchor":False})
                    del model; torch.cuda.empty_cache(); print(f"[representations] {fm} {dataset} fold={fold} seed={seed}",flush=True)
    frame=pd.DataFrame(rows); c.write_csv(c.RESULTS/"FM_TASK_PERFORMANCE_PER_RUN.csv",frame)
    summary=frame.groupby(["dataset","model"],as_index=False).agg(task_BA=("task_BA","mean"),task_BA_sd=("task_BA","std"),runs=("task_BA","size"),competence_threshold=("competence_threshold","first")); summary["competent"]=summary.task_BA>=summary.competence_threshold
    c.write_csv(c.RESULTS/"FM_TASK_PERFORMANCE.csv",summary); return summary


def identity_skill(features:np.ndarray,subjects:np.ndarray,sessions:np.ndarray,pair:tuple[int,int])->float:
    ordered=c.subject_sort(np.unique(subjects.astype(str))); code={s:i for i,s in enumerate(ordered)}; values=[]
    for tr_session,ev_session in (pair,pair[::-1]):
        tr=np.flatnonzero(sessions.astype(int)==tr_session); ev=np.flatnonzero(sessions.astype(int)==ev_session)
        ytr=np.asarray([code[s] for s in subjects[tr].astype(str)]); yev=np.asarray([code[s] for s in subjects[ev].astype(str)])
        clf=LogisticRegression(C=1.0,max_iter=1000,solver="lbfgs").fit(features[tr],ytr); p=np.clip(clf.predict_proba(features[ev]),1e-12,1)
        ce=-np.log(p[np.arange(len(ev)),yev]); values.append(math.log(len(ordered))-float(ce.mean()))
    return float(np.mean(values))


def persistent_directions(dataset:str,features:np.ndarray,subjects:np.ndarray,sessions:np.ndarray,count:int=8):
    x=np.asarray(features,np.float64); center=x.mean(0); ordered=c.subject_sort(np.unique(subjects.astype(str))); a,b=c.SOURCE_SESSIONS[dataset]
    m1=np.stack([x[(subjects.astype(str)==s)&(sessions==a)].mean(0) for s in ordered]); m2=np.stack([x[(subjects.astype(str)==s)&(sessions==b)].mean(0) for s in ordered])
    if dataset=="OpenBMI":
        geom=np.concatenate((m1-center,m2-center)); _,_,vt=np.linalg.svd(geom,full_matrices=False); pool=vt[:min(24,len(vt))].T; candidates=[]
        for i in range(pool.shape[1]):
            v=pool[:,i]; p1=(m1-center)@v; p2=(m2-center)@v; rho=0. if min(np.std(p1),np.std(p2))<1e-12 else float(np.corrcoef(p1,p2)[0,1]); g=float(np.sqrt(np.mean(((x-center)@v)**2))); candidates.append((i,rho,g))
        order=sorted(range(len(candidates)),key=lambda i:(-candidates[i][1],-candidates[i][2],i))[:count]; basis=pool[:,order]; meta=[{"persistence":candidates[i][1],"geometry_strength":candidates[i][2],"rank":j+1} for j,i in enumerate(order)]
    else:
        c1=m1-m1.mean(0); c2=m2-m2.mean(0); cross=(c1.T@c2+c2.T@c1)/(2*max(len(ordered)-1,1)); vals,vecs=np.linalg.eigh((cross+cross.T)/2); order=np.argsort(vals)[::-1][:count]; basis=vecs[:,order]; meta=[]
        for j,i in enumerate(order):
            p1=c1@vecs[:,i];p2=c2@vecs[:,i];rho=0. if min(np.std(p1),np.std(p2))<1e-12 else float(np.corrcoef(p1,p2)[0,1]);meta.append({"persistence":rho,"geometry_strength":float(vals[i]),"rank":j+1})
    return center,basis,meta


def erase(x,center,v):
    v=np.asarray(v,np.float64);v/=max(np.linalg.norm(v),EPS);z=np.asarray(x,np.float64);return z-((z-center)@v)[:,None]*v[None,:]


def ce(logits,labels):
    z=np.asarray(logits,np.float64);z-=z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True));return -lp[np.arange(len(labels)),labels.astype(int)]


def run_d_vs_i()->tuple[pd.DataFrame,dict]:
    cells=[]
    for fm in c.FMS:
        for dataset in c.DATASETS:
            pair=c.SOURCE_SESSIONS[dataset]
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    src=load_rep(fm,dataset,fold,seed,"model_fit"); val=load_rep(fm,dataset,fold,seed,"validation"); out=load_rep(fm,dataset,fold,seed,"outcome"); center,basis,meta=persistent_directions(dataset,src["features"],src["subjects"],src["sessions"])
                    model=c.load_anchor(fm,dataset,fold,seed,torch.device("cuda"));w=model.head.weight.detach().cpu().numpy().astype(np.float64);b=model.head.bias.detach().cpu().numpy().astype(np.float64);del model;torch.cuda.empty_cache()
                    full_i=identity_skill(val["features"],val["subjects"],val["sessions"],pair);clean_src=src["features"].astype(np.float64)@w.T+b;clean_out=out["features"].astype(np.float64)@w.T+b;clean_ce=ce(clean_out,out["labels"])
                    for j,(v,m) in enumerate(zip(basis.T,meta)):
                        es=erase(src["features"],center,v);ev=erase(val["features"],center,v);eo=erase(out["features"],center,v);esl=es@w.T+b;eol=eo@w.T+b;delta=esl-clean_src;delta-=delta.mean(1,keepdims=True)
                        cells.append({"dataset":dataset,"model":fm,"fold":fold,"seed":seed,"run":f"f{fold}","direction":j,"persistence":m["persistence"],"geometry_strength":m["geometry_strength"],"rank":m["rank"],"identity":full_i-identity_skill(ev,val["subjects"],val["sessions"],pair),"decision":float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))),"consequence":float(np.mean(ce(eol,out["labels"])-clean_ce))})
                    print(f"[D>I] {fm} {dataset} fold={fold} seed={seed}",flush=True)
    frame=pd.DataFrame(cells);c.write_csv(c.RESULTS/"FM_D_VS_I_CELLS.csv",frame);pred=[];summary=[]
    specs={"M0":["persistence","geometry_strength","rank"],"MI":["persistence","geometry_strength","rank","identity"],"MD":["persistence","geometry_strength","rank","decision"],"MID":["persistence","geometry_strength","rank","identity","decision"]}
    for (dataset,fm),g in frame.groupby(["dataset","model"]):
        for held in sorted(g.run.unique()):
            tr=g[g.run!=held];te=g[g.run==held]
            for name,cols in specs.items():
                mu=tr[cols].mean().to_numpy();sd=tr[cols].std(ddof=0).replace(0,1).to_numpy();reg=Ridge(alpha=1.0).fit((tr[cols].to_numpy()-mu)/sd,tr.consequence);p=reg.predict((te[cols].to_numpy()-mu)/sd)
                for idx,y,yhat in zip(te.index,te.consequence,p):pred.append({"dataset":dataset,"model":fm,"run":held,"cell_index":idx,"regression":name,"truth":y,"prediction":yhat})
        q=pd.DataFrame([x for x in pred if x["dataset"]==dataset and x["model"]==fm])
        rms={name:float(np.sqrt(mean_squared_error(q[q.regression==name].truth,q[q.regression==name].prediction))) for name in specs}
        summary.append({"dataset":dataset,"model":fm,"D_RMSE":rms["MD"],"I_RMSE":rms["MI"],"M0_RMSE":rms["M0"],"MID_RMSE":rms["MID"],"RMSE_I_minus_D":rms["MI"]-rms["MD"],"D_better":rms["MD"]<rms["MI"]})
    predictions=pd.DataFrame(pred);c.write_csv(c.RESULTS/"FM_D_VS_I_PREDICTIONS.csv",predictions);result=pd.DataFrame(summary);c.write_csv(c.RESULTS/"FM_D_VS_I.csv",result)
    stats=summarize_d(predictions,result)
    return result,stats


def summarize_d(predictions:pd.DataFrame,result:pd.DataFrame)->dict:
    run_d=[]
    for (dataset,fm,run),g in predictions.groupby(["dataset","model","run"]):
        vals={name:float(np.sqrt(mean_squared_error(v.truth,v.prediction))) for name,v in g.groupby("regression")};run_d.append({"dataset":dataset,"model":fm,"run":run,"difference":vals["MI"]-vals["MD"]})
    rd=pd.DataFrame(run_d);rng=np.random.default_rng(c.stable_seed("d-i-bootstrap"));boot=[]
    # Pre-aggregate the two synchronized FM rows for every dataset/fold.  This
    # is algebraically identical to the former per-draw DataFrame filtering,
    # but avoids repeated native allocations on Windows during 10k draws.
    grouped=rd.groupby(["dataset","run"],as_index=False).difference.mean()
    by_dataset={dataset:g.sort_values("run").difference.to_numpy(dtype=np.float64) for dataset,g in grouped.groupby("dataset")}
    for _ in range(10000):
        values=[]
        for dataset in sorted(by_dataset):
            fold_values=by_dataset[dataset];values.extend(fold_values[rng.integers(0,len(fold_values),len(fold_values))])
        boot.append(float(np.mean(values)))
    obs=float(rd.difference.mean());stats={"settings_D_better":int(result.D_better.sum()),"settings":len(result),"pooled_fold_mean_RMSE_I_minus_D":obs,"pooled_run_mean_RMSE_I_minus_D":obs,"bootstrap_group":"fold within dataset; all seeds held together; fold ids synchronized across FMs of the same dataset","bootstrap_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"bootstrap_draws":10000,"terminal":"FM_D_GT_I_REPLICATED" if int(result.D_better.sum())>=3 and np.quantile(boot,.025)>0 else "FM_D_GT_I_NOT_REPLICATED"}
    c.write_json(c.RESULTS/"FM_D_VS_I_STATISTICS.json",stats);return stats


def resume_or_run_d_vs_i()->tuple[pd.DataFrame,dict]:
    result_path=c.RESULTS/"FM_D_VS_I.csv";pred_path=c.RESULTS/"FM_D_VS_I_PREDICTIONS.csv";cells_path=c.RESULTS/"FM_D_VS_I_CELLS.csv";stats_path=c.RESULTS/"FM_D_VS_I_STATISTICS.json"
    if stats_path.is_file(): return pd.read_csv(result_path),c.read_json(stats_path)
    if result_path.is_file() and pred_path.is_file() and cells_path.is_file():
        result=pd.read_csv(result_path);predictions=pd.read_csv(pred_path);cells=pd.read_csv(cells_path)
        expected_cells=len(c.FMS)*len(c.DATASETS)*len(c.FOLDS)*len(c.SEEDS)*8
        if len(result)!=len(c.FMS)*len(c.DATASETS) or len(cells)!=expected_cells or len(predictions)!=expected_cells*4: raise RuntimeError("incomplete D>I resume artifacts")
        if predictions[["dataset","model","run","cell_index","regression"]].duplicated().any(): raise RuntimeError("duplicate D>I resume keys")
        print("[D>I] resuming statistics from validated complete cell/prediction tables",flush=True)
        return result,summarize_d(predictions,result)
    return run_d_vs_i()


def bootstrap_corr(x,y,seed):
    rng=np.random.default_rng(seed);n=len(x);vals=[]
    for _ in range(10000):
        idx=rng.integers(0,n,n);v=spearmanr(x[idx],y[idx]).statistic
        if np.isfinite(v):vals.append(v)
    if not vals:
        return [float("nan"),float("nan")]
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]


def run_scaa(lock:dict)->tuple[pd.DataFrame,dict]:
    device=torch.device("cuda");rows=[];selected=lock["SCAA"]["selected_lr"]
    for fm in c.FMS:
        for fold in c.FOLDS:
            targets=c.fold_roles("WBCIC",fold)["outcome"]
            for seed in c.SEEDS:
                ex=load_rep(fm,"WBCIC",fold,seed,"outcome_all");model=c.load_anchor(fm,"WBCIC",fold,seed,device);w=model.head.weight.detach().cpu().numpy();b=model.head.bias.detach().cpu().numpy();del model;torch.cuda.empty_cache()
                for subject in targets:
                    m=ex["subjects"].astype(str)==subject;f=ex["features"][m];y=ex["labels"][m];ses=ex["sessions"][m];anchor=ex["logits"][m];s1=np.flatnonzero(ses==0);reltr,relva=c.chronological_class_split(y[s1]);tr=s1[reltr];va=s1[relva]
                    ad=c.adapt_linear_head(f,y,tr,va,w,b,float(selected[fm]),c.stable_seed("scaa-primary",fm,fold,seed,subject));rec={"model":fm,"fold":fold,"seed":seed,"subject_id":subject,"selected_lr":float(selected[fm])}
                    for session,label in ((1,"S2"),(2,"S3")):
                        idx=np.flatnonzero(ses==session);a=c.metrics(y[idx],anchor[idx]);d=c.metrics(y[idx],ad["logits"][idx]);rec.update({f"anchor_{label}_BA":a["BA"],f"adapted_{label}_BA":d["BA"],f"Delta_{label}":d["BA"]-a["BA"]})
                    rows.append(rec)
                print(f"[SCAA] {fm} fold={fold} seed={seed}",flush=True)
    frame=pd.DataFrame(rows);c.write_csv(c.RESULTS/"FM_SCAA_PER_SUBJECT_SEED.csv",frame);subject=frame.groupby(["model","subject_id"],as_index=False).agg(Delta_S2=("Delta_S2","mean"),Delta_S3=("Delta_S3","mean"),anchor_S3_BA=("anchor_S3_BA","mean"),adapted_S3_BA=("adapted_S3_BA","mean"));c.write_csv(c.RESULTS/"FM_SCAA_PER_SUBJECT.csv",subject)
    sums=[]
    for fm,g in subject.groupby("model"):
        pear=float(pearsonr(g.Delta_S2,g.Delta_S3).statistic);rho=float(spearmanr(g.Delta_S2,g.Delta_S3).statistic);ci=bootstrap_corr(g.Delta_S2.to_numpy(),g.Delta_S3.to_numpy(),c.stable_seed("scaa-boot",fm));sel=g.Delta_S2>0;always_harm=float(np.mean(g.Delta_S3<0));gate_harm=float(np.mean(g.loc[sel,"Delta_S3"]<0)) if sel.any() else None;coverage=float(sel.mean());gate_ba=float(np.mean(np.where(sel,g.adapted_S3_BA,g.anchor_S3_BA)))
        sums.append({"model":fm,"subjects":len(g),"Pearson":pear,"Spearman":rho,"Spearman_CI_low":ci[0],"Spearman_CI_high":ci[1],"sign_concordance":float(np.mean(np.sign(g.Delta_S2)==np.sign(g.Delta_S3))),"sign_binomial_p":float(binomtest(int(np.sum(np.sign(g.Delta_S2)==np.sign(g.Delta_S3))),len(g),.5,alternative="greater").pvalue),"always_adapt_harm":always_harm,"S2_gate_harm":gate_harm,"relative_harm_reduction":(always_harm-gate_harm)/always_harm if always_harm>0 and gate_harm is not None else None,"coverage":coverage,"anchor_S3_BA":float(g.anchor_S3_BA.mean()),"always_adapt_S3_BA":float(g.adapted_S3_BA.mean()),"S2_gated_S3_BA":gate_ba})
    summary=pd.DataFrame(sums);c.write_csv(c.RESULTS/"FM_SCAA_SUMMARY.csv",summary)
    wide={fm:g.set_index("subject_id") for fm,g in subject.groupby("model")};ids=sorted(set.intersection(*(set(x.index) for x in wide.values())),key=int);x=np.concatenate([wide[fm].loc[ids].Delta_S2 for fm in c.FMS]);y=np.concatenate([wide[fm].loc[ids].Delta_S3 for fm in c.FMS]);rho=float(spearmanr(x,y).statistic);rng=np.random.default_rng(c.stable_seed("scaa-pooled"));boots=[]
    for _ in range(10000):
        sample=rng.choice(ids,len(ids),replace=True);xb=np.concatenate([wide[fm].loc[sample].Delta_S2 for fm in c.FMS]);yb=np.concatenate([wide[fm].loc[sample].Delta_S3 for fm in c.FMS]);v=spearmanr(xb,yb).statistic
        if np.isfinite(v):boots.append(v)
    sign_by_subject=np.asarray([np.mean([np.sign(wide[fm].loc[s].Delta_S2)==np.sign(wide[fm].loc[s].Delta_S3) for fm in c.FMS]) for s in ids],np.float64);sign_boot=[]
    for _ in range(10000):sign_boot.append(float(np.mean(rng.choice(sign_by_subject,len(sign_by_subject),replace=True))))
    sign_ci=[float(np.quantile(sign_boot,.025)),float(np.quantile(sign_boot,.975))];allg=subject;sel=allg.Delta_S2>0;always=float(np.mean(allg.Delta_S3<0));gate=float(np.mean(allg.loc[sel,"Delta_S3"]<0)) if sel.any() else None;relative=(always-gate)/always if always>0 and gate is not None else None;pooled={"Spearman":rho,"CI95":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],"sign_concordance":float(sign_by_subject.mean()),"sign_concordance_CI95":sign_ci,"always_adapt_harm":always,"S2_gate_harm":gate,"relative_harm_reduction":relative,"coverage":float(sel.mean())}
    task=pd.read_csv(c.RESULTS/"FM_TASK_PERFORMANCE.csv");competent={fm:bool(task[(task.dataset=="WBCIC")&(task.model==fm)].competent.iloc[0]) for fm in c.FMS};summary["FM_task_competent"]=summary.model.map(competent)
    individual_positive=all(summary.Spearman>0);strong=all(competent.values()) and individual_positive and pooled["CI95"][0]>0 and pooled["sign_concordance"]>=.65 and sign_ci[0]>.5 and relative is not None and relative>=.25 and pooled["coverage"]>=.25 and all(summary.S2_gated_S3_BA>=summary.anchor_S3_BA-.01)
    individual_strong=(summary.FM_task_competent)&(summary.Spearman_CI_low>0)&(summary.sign_concordance>=.65)&(summary.relative_harm_reduction>=.25)&(summary.coverage>=.25)&(summary.S2_gated_S3_BA>=summary.anchor_S3_BA-.01)
    one_strong=int(individual_strong.sum())==1
    pooled["FM_task_competence"]=competent;pooled["terminal"]="FM_HISTORY_UTILITY_RESCUE_CANDIDATE" if strong else ("FM_HISTORY_UTILITY_ARCHITECTURE_DEPENDENT" if one_strong else "FM_HISTORY_UTILITY_RESCUE_NOT_SUPPORTED");c.write_csv(c.RESULTS/"FM_SCAA_SUMMARY.csv",summary)
    c.write_json(c.RESULTS/"FM_SCAA_STATISTICS.json",pooled);return summary,pooled


def support_distance(query,support):
    q=np.asarray(query,np.float64);s=np.asarray(support,np.float64);d=np.sum(q*q,1)[:,None]+np.sum(s*s,1)[None,:]-2*q@s.T;np.maximum(d,0,out=d);return np.sqrt(np.partition(d,2,axis=1)[:,:3]).mean(1)


def support_radius(support):
    s=np.asarray(support,np.float64);d=np.linalg.norm(s[:,None]-s[None,:],axis=2);np.fill_diagonal(d,np.inf);clean=np.partition(d,2,axis=1)[:,:3].mean(1);return float(np.quantile(clean,.95))


def solve_alpha(q,delta,support,radius):
    selected=np.zeros(len(q));
    for alpha in ALPHA_GRID[1:]:
        ok=support_distance(q+alpha*delta,support)<=radius;selected[ok]=alpha
    return selected


def centroids(rep):
    out={}
    for s in c.subject_sort(np.unique(rep["subjects"].astype(str))):
        for y in sorted(np.unique(rep["labels"])):
            for ses in sorted(np.unique(rep["sessions"])):
                m=(rep["subjects"].astype(str)==s)&(rep["labels"]==y)&(rep["sessions"]==ses)
                if m.any():out[(s,int(y),int(ses))]=rep["features"][m].mean(0).astype(np.float64)
    return out


def cosine(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),EPS))


def subject_ci(frame,value,seed):
    per=frame.groupby("source_subject",as_index=False)[value].mean();values=per[value].to_numpy(np.float64)
    if len(values)<4 or not np.isfinite(values).all():
        return float(np.nanmean(values)),float("nan"),float("nan")
    rng=np.random.default_rng(seed);draws=rng.integers(0,len(values),size=(10000,len(values)));dist=values[draws].mean(1)
    return float(values.mean()),float(np.quantile(dist,.025)),float(np.quantile(dist,.975))


def scst_unit(fm,dataset,fold,seed):
    src=load_rep(fm,dataset,fold,seed,"model_fit");val=load_rep(fm,dataset,fold,seed,"validation");a,b=c.SOURCE_SESSIONS[dataset]
    # Exact Repair-2-compatible source-bank feature scaling, defined without
    # validation or outcome statistics.
    bank=src["sessions"].astype(int)==a;center=src["features"][bank].mean(0);scale=src["features"][bank].std(0);scale[scale<1e-6]=1.0
    src={**src,"features":((src["features"]-center)/scale).astype(np.float32)};val={**val,"features":((val["features"]-center)/scale).astype(np.float32)}
    cs=centroids(src);cv=centroids(val);subjects=c.subject_sort(np.unique(src["subjects"].astype(str)));vsubjects=c.subject_sort(np.unique(val["subjects"].astype(str)));labels=sorted(np.unique(src["labels"]).astype(int));pop={(y,ses):np.mean([cs[(sub,y,ses)] for sub in subjects],axis=0) for y in labels for ses in (a,b)};res={(sub,y,ses):cs[(sub,y,ses)]-pop[(y,ses)] for sub in subjects for y in labels for ses in (a,b)}
    validation_bank=val["sessions"].astype(int)==a;validation_eval=val["sessions"].astype(int)==b
    probe=LogisticRegression(C=1.0,class_weight="balanced",solver="lbfgs",max_iter=2000,random_state=c.stable_seed("scst-class-probe",fm,dataset,fold,seed)).fit(val["features"][validation_bank],val["labels"][validation_bank])
    probe_ba=float(balanced_accuracy_score(val["labels"][validation_eval],probe.predict(val["features"][validation_eval])))
    source_metrics={sub:{"stability_effect":[],"affinity_improvement":[],"advantage_over_random":[],"class_accuracy_change":[],"class_true_log_probability_change":[]} for sub in subjects}
    for sub in subjects:
        for y in labels:
            base=res[(sub,y,a)];matched=cosine(base,res[(sub,y,b)]);mismatch=float(np.mean([cosine(base,res[(other,y,b)]) for other in subjects if other!=sub]));source_metrics[sub]["stability_effect"].append(matched-mismatch)
    rng=np.random.default_rng(c.stable_seed("scst",fm,dataset,fold,seed));knn_transport=[];knn_clean=[];off=[];offr=[]
    for y in labels:
        support=np.stack([cs[(sub,y,a)] for sub in subjects]);radius=support_radius(support);real=np.stack([cs[(sub,y,b)] for sub in subjects]+[cv[(sub,y,b)] for sub in vsubjects]);knn=NearestNeighbors(n_neighbors=3).fit(real);loo=NearestNeighbors(n_neighbors=4).fit(real);loo_dist=loo.kneighbors(real,return_distance=True)[0][:,1:].mean(1);threshold=float(np.quantile(loo_dist,.95))
        for source in subjects:
            targets=[t for t in subjects if t!=source];q=np.repeat(cs[(source,y,b)][None,:],len(targets),0);target=np.stack([cs[(t,y,b)] for t in targets]);delta=np.stack([res[(t,y,a)]-res[(source,y,a)] for t in targets]);alpha=solve_alpha(q,delta,support,radius);transport=q+alpha[:,None]*delta
            random=[]
            for d in delta:
                r=rng.normal(size=len(d));r*=np.linalg.norm(d)/max(np.linalg.norm(r),EPS);random.append(r)
            random=np.asarray(random);rand=q+alpha[:,None]*random;clean_dist=np.linalg.norm(q-target,axis=1);trans_dist=np.linalg.norm(transport-target,axis=1);rand_dist=np.linalg.norm(rand-target,axis=1);relative=(clean_dist-trans_dist)/np.maximum(clean_dist,EPS);random_relative=(clean_dist-rand_dist)/np.maximum(clean_dist,EPS);source_metrics[source]["affinity_improvement"].extend(relative.tolist());source_metrics[source]["advantage_over_random"].extend((relative-random_relative).tolist())
            dk=knn.kneighbors(transport,return_distance=True)[0].mean(1);ck=knn.kneighbors(q,return_distance=True)[0].mean(1);rk=knn.kneighbors(rand,return_distance=True)[0].mean(1);knn_transport.extend(dk.tolist());knn_clean.extend(ck.tolist());off.extend((dk>threshold).tolist());offr.extend((rk>threshold).tolist())
            trials=np.flatnonzero((src["subjects"].astype(str)==source)&(src["labels"]==y)&(src["sessions"]==b));clean=src["features"][trials].astype(np.float64);clean_pred=probe.predict(clean);clean_prob=probe.predict_proba(clean)[:,list(probe.classes_).index(y)];clean_acc=float(np.mean(clean_pred==y));clean_logp=np.log(np.clip(clean_prob,1e-12,1))
            for j,t in enumerate(targets):
                trial_delta=np.repeat(delta[j][None,:],len(clean),axis=0);trial_alpha=solve_alpha(clean,trial_delta,support,radius);moved=clean+trial_alpha[:,None]*trial_delta;pred=probe.predict(moved);prob=probe.predict_proba(moved)[:,list(probe.classes_).index(y)];source_metrics[source]["class_accuracy_change"].append(float(np.mean(pred==y)-clean_acc));source_metrics[source]["class_true_log_probability_change"].append(float(np.mean(np.log(np.clip(prob,1e-12,1))-clean_logp)))
    subject_rows=[]
    for source,values in source_metrics.items():
        subject_rows.append({"dataset":dataset,"model":fm,"fold":fold,"seed":seed,"source_subject":source,**{key:float(np.mean(value)) for key,value in values.items()}})
    sub=pd.DataFrame(subject_rows);st,stlo,sthi=subject_ci(sub,"stability_effect",c.stable_seed("scst-unit-stability",fm,dataset,fold,seed));af,aflo,afhi=subject_ci(sub,"affinity_improvement",c.stable_seed("scst-unit-affinity",fm,dataset,fold,seed));ra,ralo,rahi=subject_ci(sub,"advantage_over_random",c.stable_seed("scst-unit-random",fm,dataset,fold,seed))
    unit={"dataset":dataset,"model":fm,"fold":fold,"seed":seed,"independent_probe_BA":probe_ba,"residual_stability":st,"stability_CI_low":stlo,"stability_CI_high":sthi,"affinity_improvement":af,"affinity_CI_low":aflo,"affinity_CI_high":afhi,"advantage_over_random":ra,"advantage_over_random_CI_low":ralo,"advantage_over_random_CI_high":rahi,"class_accuracy_change":float(sub.class_accuracy_change.mean()),"class_accuracy_loss":float(-sub.class_accuracy_change.mean()),"class_true_log_probability_change":float(sub.class_true_log_probability_change.mean()),"manifold_transport_mean":float(np.mean(knn_transport)),"manifold_clean_mean":float(np.mean(knn_clean)),"independent_session_3NN_ratio":float(np.mean(knn_transport)/max(np.mean(knn_clean),EPS)),"off_manifold_rate":float(np.mean(off)),"random_off_manifold_rate":float(np.mean(offr)),"off_manifold_excess_vs_random":float(np.mean(off)-np.mean(offr)),"source_subjects":len(sub)}
    return unit,subject_rows


def run_scst()->tuple[pd.DataFrame,dict]:
    rows=[];subject_rows=[]
    for fm in c.FMS:
        for dataset in c.DATASETS:
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    unit,cells=scst_unit(fm,dataset,fold,seed);rows.append(unit);subject_rows.extend(cells);print(f"[SCST] {fm} {dataset} fold={fold} seed={seed}",flush=True)
    per=pd.DataFrame(rows);subjects=pd.DataFrame(subject_rows);c.write_csv(c.RESULTS/"FM_SCST_PER_FOLD.csv",per);c.write_csv(c.RESULTS/"FM_SCST_PER_SOURCE_SUBJECT.csv",subjects);task=pd.read_csv(c.RESULTS/"FM_TASK_PERFORMANCE.csv");summaries=[]
    for (dataset,fm),units in per.groupby(["dataset","model"]):
        cells=subjects[(subjects.dataset==dataset)&(subjects.model==fm)];st,stlo,sthi=subject_ci(cells,"stability_effect",c.stable_seed("scst-summary-stability",fm,dataset));af,aflo,afhi=subject_ci(cells,"affinity_improvement",c.stable_seed("scst-summary-affinity",fm,dataset));ra,ralo,rahi=subject_ci(cells,"advantage_over_random",c.stable_seed("scst-summary-random",fm,dataset));class_change=float(cells.groupby("source_subject").class_accuracy_change.mean().mean());logp=float(cells.groupby("source_subject").class_true_log_probability_change.mean().mean());ratio=float(units.manifold_transport_mean.mean()/max(units.manifold_clean_mean.mean(),EPS));off=float(units.off_manifold_rate.mean()-units.random_off_manifold_rate.mean());task_ok=bool(task[(task.dataset==dataset)&(task.model==fm)].competent.iloc[0]);probe_ok=bool(units.independent_probe_BA.mean()>=.55);gate_stability=bool(st>0 and stlo>0);gate_subject=bool(af>0 and aflo>0 and ra>0 and ralo>0);gate_class=bool(-class_change<=.02 and logp>=-.05);gate_manifold=bool(ratio<=1.25 and off<=.02)
        summaries.append({"dataset":dataset,"model":fm,"FM_task_competent":task_ok,"independent_probe_BA":float(units.independent_probe_BA.mean()),"residual_stability":st,"stability_CI_low":stlo,"stability_CI_high":sthi,"affinity_improvement":af,"affinity_CI_low":aflo,"affinity_CI_high":afhi,"advantage_over_random":ra,"advantage_over_random_CI_low":ralo,"advantage_over_random_CI_high":rahi,"class_accuracy_change":class_change,"class_accuracy_loss":-class_change,"class_true_log_probability_change":logp,"independent_session_3NN_ratio":ratio,"off_manifold_excess_vs_random":off,"gate_FM_task_competence":task_ok,"gate_independent_probe_competence":probe_ok,"gate_stability":gate_stability,"gate_subject_fidelity":gate_subject,"gate_class_fidelity":gate_class,"gate_manifold":gate_manifold,"valid":bool(task_ok and probe_ok and gate_stability and gate_subject and gate_class and gate_manifold)})
    summary=pd.DataFrame(summaries);c.write_csv(c.RESULTS/"FM_SCST_SUMMARY.csv",summary)
    w=summary[summary.dataset=="WBCIC"];op=summary[summary.dataset=="OpenBMI"];strong=bool(w.valid.all() and op.valid.all());one=bool(w.valid.sum()==1);terminal="FM_SCST_RESCUE_CANDIDATE" if strong else ("FM_SCST_ARCHITECTURE_DEPENDENT" if one else "FM_SCST_RESCUE_NOT_SUPPORTED");stats={"terminal":terminal,"WBCIC_both_pass":bool(w.valid.all()),"OpenBMI_both_pass":bool(op.valid.all())};c.write_json(c.RESULTS/"FM_SCST_STATISTICS.json",stats);return summary,stats


def main():
    lock=verify_lock();c.prepare_inputs("WBCIC",include_future=True);task=build_representations(lock);d,ds=resume_or_run_d_vs_i();scaa,ss=run_scaa(lock);scst,ts=run_scst();c.write_json(c.RUNTIME/"PRIMARY_COMPLETE.json",{"complete":True,"D":ds,"SCAA":ss,"SCST":ts});print("FM_RESCUE_PRIMARY_COMPLETE",flush=True)


if __name__=="__main__":main()
