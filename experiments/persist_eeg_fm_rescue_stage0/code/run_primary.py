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
    with temp.open("wb") as f: np.savez_compressed(f,**value)
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
        clf=LogisticRegression(C=1.0,max_iter=1000,solver="lbfgs",multi_class="auto").fit(features[tr],ytr); p=np.clip(clf.predict_proba(features[ev]),1e-12,1)
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
                        cells.append({"dataset":dataset,"model":fm,"fold":fold,"seed":seed,"run":f"f{fold}s{seed}","direction":j,"persistence":m["persistence"],"geometry_strength":m["geometry_strength"],"rank":m["rank"],"identity":full_i-identity_skill(ev,val["subjects"],val["sessions"],pair),"decision":float(np.sqrt(np.mean(np.sum(delta*delta,axis=1)))),"consequence":float(np.mean(ce(eol,out["labels"])-clean_ce))})
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
    run_d=[]
    for (dataset,fm,run),g in predictions.groupby(["dataset","model","run"]):
        vals={name:float(np.sqrt(mean_squared_error(v.truth,v.prediction))) for name,v in g.groupby("regression")};run_d.append({"dataset":dataset,"model":fm,"run":run,"difference":vals["MI"]-vals["MD"]})
    rd=pd.DataFrame(run_d);rng=np.random.default_rng(c.stable_seed("d-i-bootstrap")); groups=sorted(rd.run.unique());boot=[]
    for _ in range(10000):
        sample=rng.choice(groups,len(groups),replace=True);boot.append(float(np.mean([rd[rd.run==r].difference.mean() for r in sample])))
    obs=float(rd.difference.mean());stats={"settings_D_better":int(result.D_better.sum()),"settings":len(result),"pooled_run_mean_RMSE_I_minus_D":obs,"bootstrap_ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"bootstrap_draws":10000,"terminal":"FM_D_GT_I_REPLICATED" if int(result.D_better.sum())>=3 and np.quantile(boot,.025)>0 else "FM_D_GT_I_NOT_REPLICATED"}
    c.write_json(c.RESULTS/"FM_D_VS_I_STATISTICS.json",stats);return result,stats


def bootstrap_corr(x,y,seed):
    rng=np.random.default_rng(seed);n=len(x);vals=[]
    for _ in range(10000):
        idx=rng.integers(0,n,n);v=spearmanr(x[idx],y[idx]).statistic
        if np.isfinite(v):vals.append(v)
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
    allg=subject;sel=allg.Delta_S2>0;always=float(np.mean(allg.Delta_S3<0));gate=float(np.mean(allg.loc[sel,"Delta_S3"]<0));pooled={"Spearman":rho,"CI95":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],"sign_concordance":float(np.mean(np.sign(x)==np.sign(y))),"always_adapt_harm":always,"S2_gate_harm":gate,"relative_harm_reduction":(always-gate)/always if always else None,"coverage":float(sel.mean())}
    individual_positive=all(summary.Spearman>0);strong=individual_positive and pooled["CI95"][0]>0 and pooled["sign_concordance"]>=.65 and pooled["relative_harm_reduction"]>=.25 and pooled["coverage"]>=.25 and all(summary.S2_gated_S3_BA>=summary.anchor_S3_BA-.01)
    one_strong=sum((summary.Spearman_CI_low>0)&(summary.sign_concordance>=.65)&(summary.relative_harm_reduction>=.25)&(summary.coverage>=.25))==1
    pooled["terminal"]="FM_HISTORY_UTILITY_RESCUE_CANDIDATE" if strong else ("FM_HISTORY_UTILITY_ARCHITECTURE_DEPENDENT" if one_strong else "FM_HISTORY_UTILITY_RESCUE_NOT_SUPPORTED")
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


def scst_unit(fm,dataset,fold,seed):
    src=load_rep(fm,dataset,fold,seed,"model_fit");val=load_rep(fm,dataset,fold,seed,"validation");a,b=c.SOURCE_SESSIONS[dataset];cs=centroids(src);cv=centroids(val);subjects=c.subject_sort(np.unique(src["subjects"].astype(str)));vsubjects=c.subject_sort(np.unique(val["subjects"].astype(str)));labels=sorted(np.unique(src["labels"]).astype(int));pop={(y,ses):np.mean([cs[(sub,y,ses)] for sub in subjects],axis=0) for y in labels for ses in (a,b)};res={(sub,y,ses):cs[(sub,y,ses)]-pop[(y,ses)] for sub in subjects for y in labels for ses in (a,b)}
    stable=[]
    for sub in subjects:
        x=np.concatenate([res[(sub,y,a)] for y in labels]);z=np.concatenate([res[(sub,y,b)] for y in labels]);stable.append(float(np.dot(x,z)/(max(np.linalg.norm(x)*np.linalg.norm(z),EPS))))
    model=c.load_anchor(fm,dataset,fold,seed,torch.device("cuda"));w=model.head.weight.detach().cpu().numpy().astype(np.float64);bias=model.head.bias.detach().cpu().numpy().astype(np.float64);del model;torch.cuda.empty_cache();rng=np.random.default_rng(c.stable_seed("scst",fm,dataset,fold,seed));aff=[];aff_source=[];rand_adv=[];class_loss=[];tlp=[];knn_transport=[];knn_clean=[];off=[];offr=[]
    for y in labels:
        support=np.stack([cs[(sub,y,a)] for sub in subjects]);radius=support_radius(support);real=np.stack([cs[(sub,y,b)] for sub in subjects]+[cv[(sub,y,b)] for sub in vsubjects]);knn=NearestNeighbors(n_neighbors=3).fit(real)
        for source in subjects:
            targets=[t for t in subjects if t!=source];q=np.repeat(cs[(source,y,b)][None,:],len(targets),0);target=np.stack([cs[(t,y,b)] for t in targets]);delta=np.stack([res[(t,y,a)]-res[(source,y,a)] for t in targets]);alpha=solve_alpha(q,delta,support,radius);transport=q+alpha[:,None]*delta
            random=[]
            for d in delta:
                r=rng.normal(size=len(d));r-=r.dot(d)*d/max(d.dot(d),EPS);r*=np.linalg.norm(d)/max(np.linalg.norm(r),EPS);random.append(r)
            random=np.asarray(random);rand=q+alpha[:,None]*random;clean_dist=np.linalg.norm(q-target,axis=1);trans_dist=np.linalg.norm(transport-target,axis=1);rand_dist=np.linalg.norm(rand-target,axis=1);improvement=clean_dist-trans_dist;aff.extend(improvement.tolist());aff_source.append(float(np.mean(improvement)));rand_adv.extend((rand_dist-trans_dist).tolist())
            dk=knn.kneighbors(transport,return_distance=True)[0].mean(1);ck=knn.kneighbors(q,return_distance=True)[0].mean(1);rk=knn.kneighbors(rand,return_distance=True)[0].mean(1);knn_transport.extend(dk.tolist());knn_clean.extend(ck.tolist());thr=float(np.quantile(knn.kneighbors(real,return_distance=True)[0].mean(1),.95));off.extend((dk>thr).tolist());offr.extend((rk>thr).tolist())
            trials=np.flatnonzero((src["subjects"].astype(str)==source)&(src["labels"]==y)&(src["sessions"]==b));clean=src["features"][trials].astype(np.float64);cleanlog=clean@w.T+bias
            for j,t in enumerate(targets):
                moved=clean+alpha[j]*delta[j];logit=moved@w.T+bias;class_loss.append(float(np.mean(cleanlog.argmax(1)==y)-np.mean(logit.argmax(1)==y)));pc=np.exp(cleanlog-cleanlog.max(1,keepdims=True));pc/=pc.sum(1,keepdims=True);pt=np.exp(logit-logit.max(1,keepdims=True));pt/=pt.sum(1,keepdims=True);tlp.append(float(np.mean(np.log(np.clip(pt[:,y],1e-12,1))-np.log(np.clip(pc[:,y],1e-12,1)))))
    arr=np.asarray(aff_source);rngb=np.random.default_rng(c.stable_seed("scst-boot",fm,dataset,fold,seed));boot=[float(np.mean(rngb.choice(arr,len(arr),replace=True))) for _ in range(10000)]
    return {"dataset":dataset,"model":fm,"fold":fold,"seed":seed,"residual_stability":float(np.mean(stable)),"affinity_improvement":float(np.mean(aff)),"affinity_CI_low":float(np.quantile(boot,.025)),"advantage_over_random":float(np.mean(rand_adv)),"class_accuracy_loss":float(np.mean(class_loss)),"class_true_log_probability_change":float(np.mean(tlp)),"independent_session_3NN_ratio":float(np.mean(knn_transport)/max(np.mean(knn_clean),EPS)),"off_manifold_rate":float(np.mean(off)),"random_off_manifold_rate":float(np.mean(offr)),"off_manifold_excess_vs_random":float(np.mean(off)-np.mean(offr))}


def run_scst()->tuple[pd.DataFrame,dict]:
    rows=[]
    for fm in c.FMS:
        for dataset in c.DATASETS:
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    rows.append(scst_unit(fm,dataset,fold,seed));print(f"[SCST] {fm} {dataset} fold={fold} seed={seed}",flush=True)
    per=pd.DataFrame(rows);c.write_csv(c.RESULTS/"FM_SCST_PER_FOLD.csv",per);summary=per.groupby(["dataset","model"],as_index=False).agg(residual_stability=("residual_stability","mean"),affinity_improvement=("affinity_improvement","mean"),affinity_CI_low=("affinity_CI_low","mean"),advantage_over_random=("advantage_over_random","mean"),class_accuracy_loss=("class_accuracy_loss","mean"),class_true_log_probability_change=("class_true_log_probability_change","mean"),independent_session_3NN_ratio=("independent_session_3NN_ratio","mean"),off_manifold_excess_vs_random=("off_manifold_excess_vs_random","mean"));summary["valid"]=(summary.affinity_improvement>0)&(summary.affinity_CI_low>0)&(summary.advantage_over_random>0)&(summary.class_accuracy_loss<=.02)&(summary.class_true_log_probability_change>=-.05)&(summary.independent_session_3NN_ratio<=1.25)&(summary.off_manifold_excess_vs_random<=.02);c.write_csv(c.RESULTS/"FM_SCST_SUMMARY.csv",summary)
    w=summary[summary.dataset=="WBCIC"];op=summary[summary.dataset=="OpenBMI"];strong=bool(w.valid.all() and op.valid.all());one=bool(w.valid.sum()==1);terminal="FM_SCST_RESCUE_CANDIDATE" if strong else ("FM_SCST_ARCHITECTURE_DEPENDENT" if one else "FM_SCST_RESCUE_NOT_SUPPORTED");stats={"terminal":terminal,"WBCIC_both_pass":bool(w.valid.all()),"OpenBMI_both_pass":bool(op.valid.all())};c.write_json(c.RESULTS/"FM_SCST_STATISTICS.json",stats);return summary,stats


def main():
    lock=verify_lock();c.prepare_inputs("WBCIC",include_future=True);task=build_representations(lock);d,ds=run_d_vs_i();scaa,ss=run_scaa(lock);scst,ts=run_scst();c.write_json(c.RUNTIME/"PRIMARY_COMPLETE.json",{"complete":True,"D":ds,"SCAA":ss,"SCST":ts});print("FM_RESCUE_PRIMARY_COMPLETE",flush=True)


if __name__=="__main__":main()
