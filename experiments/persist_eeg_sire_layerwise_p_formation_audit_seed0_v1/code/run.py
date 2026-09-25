"""Frozen, layerwise native SIRE Protected-formation audit (OpenBMI MI, seed 0)."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP=Path(__file__).resolve().parents[1]
REPO=EXP.parents[1]
PRIOR=REPO/"experiments/persist_eeg_sire_local_p_construction_seed0_v1/code/run.py"
spec=importlib.util.spec_from_file_location("sire_layerwise_prior",PRIOR)
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
A=P.A
DEVICE=P.DEVICE
OUT=EXP/"outputs";PROTOCOL=EXP/"protocol"
RUNTIME=Path(os.environ.get("SIRE_LAYERWISE_RUNTIME",str(REPO.parent/"sire_layerwise_p_formation_seed0_runtime"))).resolve()
P.EXP,P.OUT,P.PROTOCOL,P.RUNTIME=EXP,OUT,PROTOCOL,RUNTIME
TASK="OpenBMI_MI";FOLDS=tuple(range(5));BATCH=100
STAGES=("BRANCH_15","BRANCH_63","BRANCH_127","H_CONCAT","H_SHARED1","H_SHARED2","EMBEDDING")
SERIAL=("H_CONCAT","H_SHARED1","H_SHARED2","EMBEDDING")
TRANSITIONS=tuple(zip(SERIAL[:-1],SERIAL[1:]))
RANDOM_DRAWS=4
BOOTSTRAPS=20_000
PERTURB_FRACTION=.02
TRANSFER_TRIALS=256
EPS=1e-8
torch.set_num_threads(min(int(os.environ.get("SIRE_LAYERWISE_CPU_THREADS","12")),os.cpu_count() or 1))
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False


def seed(*parts):return P.seed("layerwise",*parts)
def rcell(fold):return RUNTIME/"cells"/f"fold{fold}_seed0"


def preflight():
    A.verify_lock();cells=[]
    for fold in FOLDS:
        c=P.cell_source(fold)
        role,split,cache,source_sessions,future=P.B.role(TASK,fold)
        train=A.ordered_subjects(role["inner_train_subjects"])
        discovery=A.ordered_subjects(role["inner_val_subjects"])
        outer=A.ordered_subjects(role["outer_dev_subjects"])
        if train!=c["inner_train_subjects"] or discovery!=c["discovery_subjects"] or split!=c["split_sha256"] or set(train)&set(discovery) or set(train)&set(outer) or set(discovery)&set(outer):
            raise RuntimeError(f"subject split drift fold{fold}")
        ck=Path(c["frozen_checkpoint_path"]);norm=Path(c["normalizer_path"])
        if P.sha(ck)!=c["frozen_checkpoint_sha256"] or P.sha(norm)!=c["normalizer_sha256"]:raise RuntimeError("canonical source asset drift")
        cells.append({"fold":fold,"checkpoint_path":str(ck),"checkpoint_sha256":P.sha(ck),
            "normalizer_path":str(norm),"normalizer_sha256":P.sha(norm),"split_sha256":split,
            "protected_coordinates":c["protected_coordinates_from_frozen_sire_persist_record"],
            "train_subjects":train,"discovery_subjects":discovery,"outer_dev_subjects_excluded":outer,
            "source_sessions":list(map(int,source_sessions)),"future_session":int(future),"cache_name":cache})
    lock={"schema":"SIRE_LAYERWISE_P_FORMATION_AUDIT_SEED0_V1","task":TASK,"seed":0,
        "folds":list(FOLDS),"cells":cells,"stages":list(STAGES),
        "serial_transitions":[list(x) for x in TRANSITIONS],
        "canonical_actionability_code_sha256":P.sha(P.SOURCE/"code/run.py"),
        "canonical_actionability_protocol_sha256":P.sha(P.SOURCE_LOCK),
        "prior_local_p_code_sha256":P.sha(PRIOR),
        "pathfit":"exact source centroid cap32, alpha and raw_q; train-only subject-session-class centroids",
        "ridge_alpha":A.RIDGE_ALPHA,"final_P":"frozen source PERSIST/PEEH coordinate indices and whitening/eigenvectors",
        "random_erasure_draws":RANDOM_DRAWS,"random_subspace":"rank-matched orthonormal in orthogonal complement of Q",
        "transfer_trials_per_fold":TRANSFER_TRIALS,"perturb_fraction_of_source_RMS_norm":PERTURB_FRACTION,
        "transfer_control":"equal per-trial Euclidean energy; structured top16 source-C projection vs random C direction",
        "bootstrap_draws":BOOTSTRAPS,"no_parameter_updates":True,
        "roles_read":["inner_train","discovery"],"outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,
        "formation_rule":"three of four criteria; each fold-sign criterion at least 4/5",
        "historical_provenance_caveat":P.SOURCE_RECORD["historical_provenance_caveat"]}
    path=PROTOCOL/"PROTOCOL_LOCK.json"
    if path.exists() and json.loads(path.read_text())!=lock:raise RuntimeError("protocol lock changed")
    P.jwrite(path,lock)
    return lock


P.preflight=preflight


def stage_forward(model,x):
    xx=x.unsqueeze(1);branches=[]
    for t,tn,s,sn in zip(model.temporal,model.temporal_norm,model.spatial,model.spatial_norm):
        b=F.avg_pool2d(F.elu(sn(s(F.elu(tn(t(xx)))))),(1,4))
        branches.append(b)
    hc=torch.cat(branches,1)
    h1=F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(hc)))),(1,2))
    h2=F.avg_pool2d(F.elu(model.norm2(model.point2(model.depth2(h1)))),(1,2))
    emb=model.embedding(model.pool(h2).flatten(1));logits=model.head(emb)
    return {**{STAGES[i]:branches[i].flatten(1) for i in range(3)},
        "H_CONCAT":hc.flatten(1),"H_SHARED1":h1.flatten(1),
        "H_SHARED2":h2.flatten(1),"EMBEDDING":emb,"LOGITS":logits}


def next_stage(model,stage,h):
    if stage=="H_CONCAT":
        x=h.reshape(-1,48,1,h.shape[1]//48)
        return F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(x)))),(1,2)).flatten(1)
    if stage=="H_SHARED1":
        x=h.reshape(-1,64,1,h.shape[1]//64)
        return F.avg_pool2d(F.elu(model.norm2(model.point2(model.depth2(x)))),(1,2)).flatten(1)
    if stage=="H_SHARED2":
        x=h.reshape(-1,64,1,h.shape[1]//64)
        return model.embedding(model.pool(x).flatten(1))
    raise ValueError(stage)


def continuation(model,stage,h,other_branches=None):
    if stage.startswith("BRANCH_"):
        if other_branches is None:raise ValueError("branch continuation needs native others")
        parts=list(other_branches)
        parts[STAGES.index(stage)]=h
        h=torch.cat([p.reshape(-1,16,1,p.shape[1]//16) for p in parts],1).flatten(1)
        stage="H_CONCAT"
    if stage=="H_CONCAT":h=next_stage(model,stage,h);stage="H_SHARED1"
    if stage=="H_SHARED1":h=next_stage(model,stage,h);stage="H_SHARED2"
    if stage=="H_SHARED2":h=next_stage(model,stage,h);stage="EMBEDDING"
    if stage!="EMBEDDING":raise ValueError(stage)
    return h,model.head(h)


def extract(model,pieces,mean,std,fold):
    rows=[];state=A.model_state_sha(model)
    model.eval()
    with torch.inference_mode():
        for session,subject,raw,y,owners in pieces:
            acts={name:[] for name in STAGES};logits=[]
            for start in range(0,len(raw),BATCH):
                x=((raw[start:start+BATCH]-mean[None,:,None])/np.maximum(std[None,:,None],1e-6)).astype(np.float32)
                xx=torch.as_tensor(np.ascontiguousarray(x),device=DEVICE)
                got=stage_forward(model,xx)
                replay=model(xx)[0]
                if float(torch.max(torch.abs(got["LOGITS"]-replay)))>=1e-6:raise RuntimeError("native SIRE logits do not replay")
                for name in STAGES:acts[name].append(got[name].float().cpu().numpy())
                logits.append(got["LOGITS"].float().cpu().numpy())
            acts={name:np.concatenate(v) for name,v in acts.items()}
            z=np.concatenate(logits)
            rows.append({"task":TASK,"fold":fold,"session":session,"subject":subject,
                "raw_y":np.asarray(y,np.int64),"owners":owners,"acts":acts,
                "hs":acts["H_CONCAT"],"yd":acts["H_SHARED1"],
                "emb":acts["EMBEDDING"],"logits":z})
    if A.model_state_sha(model)!=state or model.training:raise RuntimeError("model/BN state changed in extraction")
    return rows


def centroids(rows,stage,fold):
    out=[];keys=[]
    for row in rows:
        for label in sorted(np.unique(row["raw_y"])):
            ix=np.flatnonzero(row["raw_y"]==label)
            rng=np.random.default_rng(A.stable_seed("sire_pathfit_cap32",TASK,fold,row["subject"],row["session"],int(label)))
            if len(ix)>32:ix=np.sort(rng.choice(ix,32,replace=False))
            out.append((row["subject"],int(row["session"]),int(label),row["acts"][stage][ix].mean(0)))
    out.sort(key=lambda r:(int(r[0].replace("sub-","")),r[1],r[2]))
    keys=[(r[0],r[1],r[2]) for r in out]
    return keys,np.stack([r[3] for r in out]).astype(np.float32)


def ridge_fit(x,y):
    """The source PathFit dual ridge, returning the full map plus its Q span."""
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    mu=x.mean(0);sd=np.maximum(x.std(0),1e-6);z=(x-mu)/sd
    xt=torch.as_tensor(np.ascontiguousarray(z,np.float32),device=DEVICE)
    yt=torch.as_tensor(np.ascontiguousarray(y,np.float32),device=DEVICE)
    with torch.inference_mode():
        kernel=xt@xt.T/max(xt.shape[1],1)
        chol=torch.linalg.cholesky(kernel+A.RIDGE_ALPHA*torch.eye(len(xt),device=DEVICE))
        alpha=torch.cholesky_solve(yt,chol)
        coef=((xt.T@alpha)/max(xt.shape[1],1)).cpu().numpy().astype(np.float64)
    u,s,_=np.linalg.svd(coef,full_matrices=False)
    rank=int(np.sum(s>max(float(s[0])*1e-6,1e-8))) if len(s) else 0
    if rank<1:raise RuntimeError("empty PathFit rank")
    q,_=np.linalg.qr(u[:,:rank]/sd[:,None],mode="reduced")
    return {"mu":mu.astype(np.float32),"sd":sd.astype(np.float32),
        "coef":coef.astype(np.float32),"q":np.ascontiguousarray(q.astype(np.float32)),
        "rank":rank,"training_centroids":len(x)}


def predict_path(x,fit):
    if fit.get("kind")=="CANONICAL_FINAL":
        return ((np.asarray(x,np.float64)-fit["mu"])@fit["coef"]).astype(np.float32)
    return (((np.asarray(x,np.float64)-fit["mu"])/fit["sd"])@fit["coef"]).astype(np.float32)


def final_fit(spectrum,dims):
    w=(np.asarray(spectrum["whitener"],np.float64)@
       np.asarray(spectrum["directions"],np.float64)[:,dims])
    q,_=np.linalg.qr(w,mode="reduced")
    return {"kind":"CANONICAL_FINAL","mu":spectrum["mean"],
        "coef":w.astype(np.float32),"q":q.astype(np.float32),
        "rank":len(dims),"training_centroids":None}


def stage_arrays(rows,stage):
    x=np.concatenate([r["acts"][stage] for r in rows])
    subject=np.concatenate([np.full(len(r["raw_y"]),r["subject"],dtype="U16") for r in rows])
    session=np.concatenate([np.full(len(r["raw_y"]),r["session"],dtype=np.int16) for r in rows])
    label=np.concatenate([r["raw_y"] for r in rows])
    return x,subject,session,label


def recoverability_rows(fold,stage,split,y,pred,subjects):
    y=np.asarray(y,np.float64);pred=np.asarray(pred,np.float64)
    if y.shape!=pred.shape:raise RuntimeError("P prediction shape mismatch")
    mean=y.mean(0);den=((y-mean)**2).sum(0)
    err=((y-pred)**2).sum(0)
    r2=1-err/np.maximum(den,EPS)
    corr=[]
    for i in range(y.shape[1]):
        corr.append(float(np.corrcoef(y[:,i],pred[:,i])[0,1]) if np.std(pred[:,i])>1e-12 and np.std(y[:,i])>1e-12 else None)
    yn=np.linalg.norm(y,axis=1);pn=np.linalg.norm(pred,axis=1)
    valid=(yn>1e-12)&(pn>1e-12)
    cosine=float(np.mean(np.sum(y[valid]*pred[valid],axis=1)/(yn[valid]*pn[valid]))) if valid.any() else None
    overall={"task":TASK,"fold":fold,"stage":stage,"split":split,"coordinate":"ALL",
        "trials_or_centroids":len(y),"biological_subjects":len(set(subjects)),
        "mean_R2":float(np.mean(r2)),"variance_weighted_R2":float(1-err.sum()/max(den.sum(),EPS)),
        "mean_Pearson":float(np.mean([v for v in corr if v is not None])) if any(v is not None for v in corr) else None,
        "cosine":cosine,"normalized_MSE":float(err.sum()/max(den.sum(),EPS))}
    rows=[overall]
    for i in range(y.shape[1]):
        rows.append({"task":TASK,"fold":fold,"stage":stage,"split":split,"coordinate":i,
            "trials_or_centroids":len(y),"biological_subjects":len(set(subjects)),
            "R2":float(r2[i]),"Pearson":corr[i]})
    if split=="DISCOVERY_SUBJECT_DISJOINT_TRIALS":
        subs=np.asarray(subjects)
        for subject in sorted(set(subs),key=lambda x:int(str(x).replace("sub-",""))):
            ix=subs==subject
            yy=y[ix];pp=pred[ix];center=yy-yy.mean(0)
            d=np.sum(center**2,axis=0);e=np.sum((yy-pp)**2,axis=0)
            rows.append({"task":TASK,"fold":fold,"stage":stage,"split":split,
                "coordinate":"SUBJECT","subject":subject,"trials_or_centroids":int(ix.sum()),
                "biological_subjects":1,"mean_R2":float(np.mean(1-e/np.maximum(d,EPS))),
                "variance_weighted_R2":float(1-e.sum()/max(d.sum(),EPS)),
                "normalized_MSE":float(e.sum()/max(d.sum(),EPS))})
    return rows


def grouped_oof(stage,fold,keys,x,y):
    subs=np.asarray([k[0] for k in keys]);out=np.empty_like(y)
    for subject in sorted(set(subs),key=lambda x:int(x.replace("sub-",""))):
        test=subs==subject;fit=ridge_fit(x[~test],y[~test])
        out[test]=predict_path(x[test],fit)
    return out


def persistence(rows,fit):
    pred=[];keys=[]
    for r in rows:
        for label in sorted(np.unique(r["raw_y"])):
            ix=r["raw_y"]==label
            keys.append((r["subject"],r["session"],int(label)))
            pred.append(predict_path(r["acts"][fit["stage"]][ix],fit).mean(0))
    keymap={key:p for key,p in zip(keys,pred)}
    sessions=sorted(set(k[1] for k in keys));pairs=[]
    for subject in sorted(set(k[0] for k in keys)):
        for label in sorted(set(k[2] for k in keys)):
            a=(subject,sessions[0],label);b=(subject,sessions[1],label)
            if a in keymap and b in keymap:pairs.append((subject,label,keymap[a],keymap[b]))
    if len(pairs)<3:raise RuntimeError("too few matched cross-session centroids")
    a=np.stack([p[2] for p in pairs]).astype(np.float64)
    b=np.stack([p[3] for p in pairs]).astype(np.float64)
    aa=a-a.mean(0);bb=b-b.mean(0)
    diag=[]
    for j in range(a.shape[1]):
        diag.append(float(np.corrcoef(a[:,j],b[:,j])[0,1]) if np.std(a[:,j])>1e-12 and np.std(b[:,j])>1e-12 else None)
    rho=float(np.mean([v for v in diag if v is not None])) if any(v is not None for v in diag) else None
    cov=float(np.sum(aa*bb)/np.sqrt(max(np.sum(aa**2)*np.sum(bb**2),EPS)))
    cos=float(np.mean(np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),EPS)))
    within=float(np.mean(np.linalg.norm(a-b,axis=1)))
    between=[]
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            if pairs[i][0]!=pairs[j][0] and pairs[i][1]==pairs[j][1]:between.append(np.linalg.norm(a[i]-b[j]))
    return {"matched_subject_class_pairs":len(pairs),"coordinate_Pearson":json.dumps(diag),
        "mean_diagonal_cross_session_Pearson":rho,
        "normalized_symmetric_cross_session_covariance_trace":cov,
        "matched_centroid_cosine":cos,"within_subject_cross_session_distance":within,
        "between_subject_distance_reference":float(np.mean(between)) if between else None}


def random_q(q,draw,fold,stage):
    rng=np.random.default_rng(seed("random_subspace",fold,stage,draw))
    z=rng.standard_normal(q.shape).astype(np.float64)
    qq=np.asarray(q,np.float64);z-=qq@(qq.T@z)
    basis,_=np.linalg.qr(z,mode="reduced")
    if np.max(np.abs(basis.T@basis-np.eye(q.shape[1])))>1e-6 or np.max(np.abs(qq.T@basis))>1e-5:
        raise RuntimeError("random control rank/orthogonality failure")
    return basis.astype(np.float32)


def erase(h,mu,q):
    centered=h-mu
    return h-(centered@q)@q.T


def counterfactual(model,rows,stage,fit,q,final,map_targets=False,keep_only=False):
    """Evaluate exact frozen continuation after a stage pathway erasure."""
    metrics=[]
    with torch.inference_mode():
        for r in rows:
            values=[];embeddings=[]
            n=len(r["raw_y"])
            for start in range(0,n,BATCH):
                stop=min(n,start+BATCH)
                h=torch.as_tensor(np.ascontiguousarray(r["acts"][stage][start:stop]),device=DEVICE)
                mu=torch.as_tensor(fit["mu"],device=DEVICE)
                qt=torch.as_tensor(q,device=DEVICE)
                modified=mu+((h-mu)@qt)@qt.T if keep_only else erase(h,mu,qt)
                others=None
                if stage.startswith("BRANCH_"):
                    others=[torch.as_tensor(np.ascontiguousarray(r["acts"][s][start:stop]),device=DEVICE) for s in STAGES[:3]]
                emb,z=continuation(model,stage,modified,others)
                values.append(z.float().cpu().numpy())
                if map_targets:embeddings.append(emb.float().cpu().numpy())
            z=np.concatenate(values)
            result=P.metric(r["raw_y"],z)
            if map_targets:
                baseline=final(r["emb"]);altered=final(np.concatenate(embeddings))
                dp=altered-baseline
                norm=np.linalg.norm(baseline,axis=1)
                cos=np.sum(baseline*altered,axis=1)/np.maximum(norm*np.linalg.norm(altered,axis=1),EPS)
                result.update({"final_P_change_L2_mean":float(np.linalg.norm(dp,axis=1).mean()),
                    "final_P_norm_change_mean":float((np.linalg.norm(altered,axis=1)-norm).mean()),
                    "final_P_cosine_mean":float(cos.mean())})
            metrics.append({"subject":r["subject"],"session":r["session"],**result})
    return metrics


def consequences(fold,model,discovery,fits,final):
    erasure=[];branch=[];utility=[]
    for stage in STAGES:
        fit=fits[stage];q=fit["q"]
        actual=counterfactual(model,discovery,stage,fit,q,final,stage.startswith("BRANCH_"))
        keep=counterfactual(model,discovery,stage,fit,q,final,False,True) if stage in SERIAL else None
        randoms=[counterfactual(model,discovery,stage,fit,random_q(q,j,fold,stage),final,stage.startswith("BRANCH_")) for j in range(RANDOM_DRAWS)]
        for i,r in enumerate(discovery):
            intact=P.metric(r["raw_y"],r["logits"])
            for kind,draw,row in [("PROTECTED",-1,actual[i])]+[("RANDOM",j,randoms[j][i]) for j in range(RANDOM_DRAWS)]:
                item={"task":TASK,"fold":fold,"stage":stage,"subject":r["subject"],"session":r["session"],
                    "control":kind,"draw":draw,"rank":fit["rank"],"intact_BA":intact["BA"],
                    "intact_macro_F1":intact["macro_F1"],"intact_NLL":intact["NLL"],
                    "erased_BA":row["BA"],"erased_macro_F1":row["macro_F1"],"erased_NLL":row["NLL"],
                    "BA_loss":intact["BA"]-row["BA"],"macro_F1_loss":intact["macro_F1"]-row["macro_F1"],
                    "NLL_change":row["NLL"]-intact["NLL"]}
                if stage.startswith("BRANCH_"):
                    item.update({k:row[k] for k in ("final_P_change_L2_mean","final_P_norm_change_mean","final_P_cosine_mean")})
                    branch.append(item)
                else:erasure.append(item)
            if stage in SERIAL:
                utility.append({"task":TASK,"fold":fold,"stage":stage,"subject":r["subject"],"session":r["session"],
                    "intact_BA":intact["BA"],"complement_retained_BA":actual[i]["BA"],
                    "protected_path_retained_BA":keep[i]["BA"],
                    "protected_path_retained_macro_F1":keep[i]["macro_F1"],
                    "protected_path_retained_NLL":keep[i]["NLL"],
                    "intervention_definition":"retain fitted Q component; replace complementary centered component by zero/train centroid",
                    "note":"diagnostic off-manifold retention, not additive causal attribution"})
        print("ERASURE_COMPLETE",fold,stage,flush=True)
    return erasure,branch,utility


def normalize_direction(v,fallback,energy):
    norm=np.linalg.norm(v,axis=1,keepdims=True)
    out=np.where(norm>1e-10,v/np.maximum(norm,1e-10),fallback[None,:])
    out=out/np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-10)*energy
    if np.max(np.abs(np.linalg.norm(out,axis=1)-energy))/energy>1e-5:
        raise RuntimeError("perturbation energy mismatch")
    return out.astype(np.float32)


def adjacent_transfer(fold,model,train,discovery,fits):
    outputs=[];interactions=[]
    for source,successor in TRANSITIONS:
        fit=fits[source];after=fits[successor]
        x_train=np.concatenate([r["acts"][source] for r in train])
        basis,binfo=A.complement_basis(x_train,fit["q"],fit["mu"])
        x=np.concatenate([r["acts"][source] for r in discovery])[:TRANSFER_TRIALS]
        if len(x)!=TRANSFER_TRIALS:raise RuntimeError("insufficient discovery trials for adjacent audit")
        q=np.asarray(fit["q"],np.float64);mu=np.asarray(fit["mu"],np.float64)
        center=x.astype(np.float64)-mu
        p=(center@q)@q.T
        c=center-p
        bb=np.asarray(basis,np.float64)
        c_struct=(c@bb)@bb.T
        energy=PERTURB_FRACTION*float(np.sqrt(np.mean(np.sum((x_train.astype(np.float64)-mu)**2,axis=1))))
        if energy<=0:raise RuntimeError("zero source perturbation scale")
        dp=normalize_direction(p,q[:,0],energy)
        dc=normalize_direction(c_struct,bb[:,0],energy)
        rng=np.random.default_rng(seed("random_C_transfer",fold,source))
        random_vec=rng.standard_normal(x.shape)
        random_vec-=(random_vec@q)@q.T
        dr=normalize_direction(random_vec,bb[:,0],energy)
        match_error=float(max(np.max(np.abs(np.linalg.norm(v,axis=1)-energy))/energy for v in (dp,dc,dr)))
        if match_error>1e-5:raise RuntimeError("actual perturbation energies differ")
        qnext=np.asarray(after["q"],np.float64)
        def forward(values):
            all_y=[]
            with torch.inference_mode():
                for start in range(0,len(values),64):
                    h=torch.as_tensor(np.ascontiguousarray(values[start:start+64]),device=DEVICE)
                    all_y.append(next_stage(model,source,h).float().cpu().numpy())
            return np.concatenate(all_y).astype(np.float64)
        base=forward(x)
        yp=forward(x+dp);yc=forward(x+dc);yr=forward(x+dr);ypc=forward(x+dp+dc)
        def effects(delta):
            v=delta@qnext
            pnorm=np.linalg.norm(v,axis=1)
            cnorm=np.linalg.norm(delta-v@qnext.T,axis=1)
            return pnorm,cnorm
        pp,pc=effects(yp-base)
        cp,cc=effects(yc-base)
        rp,rc=effects(yr-base)
        residue=ypc-yp-yc+base
        ip,ic=effects(residue)
        for kind,sp,sc in (("P",pp,pc),("STRUCTURED_C",cp,cc),("RANDOM_C",rp,rc)):
            outputs.append({"task":TASK,"fold":fold,"source_stage":source,"successor_stage":successor,
                "perturbation":kind,"trials":len(x),"source_basis_rank":fit["rank"],
                "successor_basis_rank":after["rank"],"structured_complement_rank":basis.shape[1],
                "source_energy":energy,"successor_P_movement_over_input_energy":float(np.mean(sp)/energy),
                "successor_C_movement_over_input_energy":float(np.mean(sc)/energy),
                "matched_energy_max_relative_error":match_error})
        interactions.append({"task":TASK,"fold":fold,"source_stage":source,"successor_stage":successor,
            "trials":len(x),"source_energy":energy,"successor_P_nonlinear_residual_over_input_energy":float(np.mean(ip)/energy),
            "successor_C_nonlinear_residual_over_input_energy":float(np.mean(ic)/energy),
            "successor_P_nonlinear_fraction":float(np.mean(ip)/max(np.mean(pp)+np.mean(cp),EPS)),
            "successor_C_nonlinear_fraction":float(np.mean(ic)/max(np.mean(pc)+np.mean(cc),EPS)),
            "interaction_definition":"F(h+dP+dC)-F(h+dP)-F(h+dC)+F(h)"})
        print("TRANSFER_COMPLETE",fold,source,successor,flush=True)
        del x_train,basis
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return outputs,interactions


def prepare_fold(fold,with_discovery=True):
    lock=preflight();c=lock["cells"][fold]
    sessions=sorted(set(c["source_sessions"]+[c["future_session"]]))
    pieces,mapping=A.fetch_role(TASK,c["train_subjects"],sessions,c["cache_name"],None)
    model=A.build_model(TASK,len(mapping),Path(c["checkpoint_path"]))
    state=A.model_state_sha(model)
    mean,std=A.load_normalizer(Path(c["normalizer_path"]))
    train=extract(model,pieces,mean,std,fold);del pieces
    original_qs,original_ms,original_qd,original_md,spectrum,proj=A.train_centroid_rows(train,c["protected_coordinates"])
    oldproj=[r for r in P.cread(P.SOURCE/"outputs/SIRE_PROJECTOR_AUDIT.csv") if r["task"]==TASK and int(r["fold"])==fold]
    expected={r["stage"]:r["projector_sha256"] for r in oldproj}
    if P.arr_sha(original_qs,original_ms)!=expected["H_concat"] or P.arr_sha(original_qd,original_md)!=expected["H_shared1"]:
        raise RuntimeError(f"historical P/C projector mismatch fold{fold}")
    keys0,emb_cent=centroids(train,"EMBEDDING",fold)
    target=A.canonical_targets(emb_cent,spectrum,c["protected_coordinates"])
    fits={};oof=[];recover=[];persistent=[];geoms=[]
    for stage in STAGES:
        keys,x=centroids(train,stage,fold)
        if keys!=keys0:raise RuntimeError("stage centroid key mismatch")
        if stage=="H_CONCAT" and not np.array_equal(x,proj["train_stage_source"]):raise RuntimeError("source centroid mismatch")
        if stage=="H_SHARED1" and not np.array_equal(x,proj["train_stage_successor"]):raise RuntimeError("successor centroid mismatch")
        fit=final_fit(spectrum,c["protected_coordinates"]) if stage=="EMBEDDING" else ridge_fit(x,target)
        if stage!="EMBEDDING":
            q,mu,info=A.raw_q(x,target)
            if fit["rank"]!=info["rank"]:raise RuntimeError("PathFit rank mismatch")
            fit["q"],fit["mu"]=q,mu
            if stage in ("H_CONCAT","H_SHARED1") and P.arr_sha(q,mu)!=expected["H_concat" if stage=="H_CONCAT" else "H_shared1"]:
                raise RuntimeError("historical stage geometry changed")
        fit["stage"]=stage
        if np.max(np.abs(fit["q"].T@fit["q"]-np.eye(fit["rank"])))>1e-5:
            raise RuntimeError("stage Q not orthonormal")
        fits[stage]=fit
        oof_pred=A.canonical_targets(x,spectrum,c["protected_coordinates"]) if stage=="EMBEDDING" else grouped_oof(stage,fold,keys,x,target)
        recover.extend(recoverability_rows(fold,stage,"INNER_TRAIN_SUBJECT_GROUPED_OOF_CENTROIDS",target,oof_pred,[k[0] for k in keys]))
        persistent.append({"task":TASK,"fold":fold,"stage":stage,"split":"INNER_TRAIN","coordinate_system":"CANONICAL_FINAL_P",**persistence(train,fit)})
        geoms.append({"stage":stage,"rank":fit["rank"],"q_mu_sha256":P.arr_sha(fit["q"],fit["mu"]),
            "centroid_count":len(x),"feature_dimension":x.shape[1],"fit_role":"inner_train_only",
            "canonical_exact":stage=="EMBEDDING"})
    if not np.array_equal(A.canonical_targets(emb_cent,spectrum,c["protected_coordinates"]),target):
        raise RuntimeError("final canonical P coordinate drift")
    discovery=[]
    if with_discovery:
        pieces,mapping2=A.fetch_role(TASK,c["discovery_subjects"],sessions,c["cache_name"],mapping)
        if mapping2!=mapping:raise RuntimeError("discovery label mapping changed")
        discovery=extract(model,pieces,mean,std,fold);del pieces
        for stage in STAGES:
            x,subs,_,_=stage_arrays(discovery,stage)
            y=A.canonical_targets(np.concatenate([r["emb"] for r in discovery]),spectrum,c["protected_coordinates"])
            pred=predict_path(x,fits[stage])
            recover.extend(recoverability_rows(fold,stage,"DISCOVERY_SUBJECT_DISJOINT_TRIALS",y,pred,subs))
            persistent.append({"task":TASK,"fold":fold,"stage":stage,"split":"DISCOVERY","coordinate_system":"CANONICAL_FINAL_P",**persistence(discovery,fits[stage])})
    if A.model_state_sha(model)!=state or model.training:raise RuntimeError("frozen model changed")
    return model,train,discovery,fits,spectrum,recover,persistent,geoms


def smoke():
    preflight()
    model,train,_,fits,spectrum,rec,persist,geometry=prepare_fold(0,False)
    r=train[0];state=A.model_state_sha(model)
    hs=torch.as_tensor(np.ascontiguousarray(r["acts"]["H_CONCAT"][:8]),device=DEVICE)
    with torch.inference_mode():
        h1=next_stage(model,"H_CONCAT",hs)
        h2=next_stage(model,"H_SHARED1",h1)
        emb=next_stage(model,"H_SHARED2",h2)
        _,logits=continuation(model,"H_CONCAT",hs)
    err=max(float(np.max(np.abs(h1.cpu().numpy()-r["acts"]["H_SHARED1"][:8]))),
            float(np.max(np.abs(h2.cpu().numpy()-r["acts"]["H_SHARED2"][:8]))),
            float(np.max(np.abs(emb.cpu().numpy()-r["emb"][:8]))),
            float(np.max(np.abs(logits.cpu().numpy()-r["logits"][:8]))))
    if err>=1e-6:raise RuntimeError(f"serial native reconstruction failed: {err}")
    for stage in STAGES:
        q=fits[stage]["q"]
        rq=random_q(q,0,0,stage)
        x=r["acts"][stage][:8]
        centered=x-fits[stage]["mu"]
        recon=(centered@q)@q.T+(centered-(centered@q)@q.T)
        if np.max(np.abs(recon-centered))>=1e-5:raise RuntimeError("P+C stage reconstruction failed")
        if rq.shape!=q.shape:raise RuntimeError("random control rank mismatch")
    if A.model_state_sha(model)!=state or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("frozen model or BN state changed")
    P.jwrite(PROTOCOL/"SMOKE_AUDIT.json",{"status":"PASS","fold":0,
        "source_checkpoint_sha256":P.cell_source(0)["frozen_checkpoint_sha256"],
        "native_stage_and_logits_max_abs":err,"historical_geometry_exact":True,
        "final_P_coordinates":P.cell_source(0)["protected_coordinates_from_frozen_sire_persist_record"],
        "train_only_centroids":True,"random_controls_rank_matched":True,
        "all_stage_P_C_reconstructions_passed":True,"no_neural_parameter_or_BN_state_change":True,
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,
        "stage_geometry":geometry})
    print("SMOKE_PASS",flush=True)


CELL_FILES=("LAYERWISE_P_RECOVERABILITY.csv","LAYERWISE_PERSISTENCE.csv",
    "LAYERWISE_ERASURE_CONSEQUENCE.csv","BRANCH_TO_FINAL_P_CONTRIBUTION.csv",
    "ADJACENT_TRANSFER.csv","ADJACENT_PC_INTERACTION.csv",
    "LAYERWISE_UTILITY_DECOMPOSITION.csv")


def run_cell(fold):
    preflight()
    if fold not in FOLDS:raise ValueError("unlocked fold")
    if not (PROTOCOL/"SMOKE_AUDIT.json").exists():raise RuntimeError("fold0 smoke must pass first")
    d=rcell(fold);d.mkdir(parents=True,exist_ok=True)
    done=d/"CELL_COMPLETE.json"
    if done.exists():
        rec=json.loads(done.read_text())
        if rec["protocol_sha256"]==P.sha(PROTOCOL/"PROTOCOL_LOCK.json") and all(P.sha(d/name)==digest for name,digest in rec["output_sha256"].items()):
            print("CELL_ALREADY_COMPLETE",fold,flush=True);return
        raise RuntimeError("completed cell drift")
    model,train,discovery,fits,spectrum,recover,persistent,geometry=prepare_fold(fold,True)
    source_state=A.model_state_sha(model)
    pathway_payload={f"{stage}__{key}":np.asarray(value) for stage,fit in fits.items()
        for key,value in fit.items() if key in ("mu","sd","coef","q")}
    pathway_payload.update({f"final_spectrum__{key}":np.asarray(spectrum[key]) for key in ("mean","whitener","directions","rho")})
    pathway_tmp=d/"FITTED_PATHWAYS.part.npz";pathway_file=d/"FITTED_PATHWAYS.npz"
    np.savez_compressed(pathway_tmp,**pathway_payload);os.replace(pathway_tmp,pathway_file)
    final=lambda emb:A.canonical_targets(emb,spectrum,P.cell_source(fold)["protected_coordinates_from_frozen_sire_persist_record"])
    erasure,branch,utility=consequences(fold,model,discovery,fits,final)
    transfer,interactions=adjacent_transfer(fold,model,train,discovery,fits)
    if A.model_state_sha(model)!=source_state or model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("neural parameter/BN state changed")
    data=dict(zip(CELL_FILES,(recover,persistent,erasure,branch,transfer,interactions,utility)))
    for name,rows in data.items():P.cwrite(d/name,rows)
    P.jwrite(d/"GEOMETRY_AUDIT.json",{"task":TASK,"fold":fold,"final_P_dimensions":P.cell_source(fold)["protected_coordinates_from_frozen_sire_persist_record"],
        "spectrum_sha256":P.arr_sha(spectrum["mean"],spectrum["whitener"],spectrum["directions"],spectrum["rho"]),
        "source_model_state_sha256":source_state,"final_model_state_sha256":A.model_state_sha(model),
        "fitted_pathway_runtime_path":str(pathway_file),"fitted_pathway_archive_sha256":P.sha(pathway_file),
        "historical_Q_source_successor_exact":True,"fitted_on_inner_train_only":True,
        "stages":geometry,"outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0})
    files=list(CELL_FILES)+["GEOMETRY_AUDIT.json"]
    P.jwrite(done,{"task":TASK,"fold":fold,"protocol_sha256":P.sha(PROTOCOL/"PROTOCOL_LOCK.json"),
        "checkpoint_sha256":P.cell_source(fold)["frozen_checkpoint_sha256"],
        "fitted_pathway_archive_sha256":P.sha(pathway_file),
        "output_sha256":{name:P.sha(d/name) for name in files},
        "neural_parameter_updates":0,"outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0})
    print("CELL_COMPLETE",fold,flush=True)


def groups(rows,*keys):
    out=defaultdict(list)
    for r in rows:out[tuple(r[k] for k in keys)].append(r)
    return out


def subject_stage_map(rows,stage,field):
    out=defaultdict(list)
    for r in rows:
        if r["stage"]==stage and r["split"]=="DISCOVERY_SUBJECT_DISJOINT_TRIALS" and r["coordinate"]=="SUBJECT":
            out[r["subject"]].append(float(r[field]))
    return {sub:float(np.mean(vals)) for sub,vals in out.items()}


def erase_excess(rows,stage,fold=None):
    chosen=[r for r in rows if r["stage"]==stage and (fold is None or int(r["fold"])==fold)]
    per_session=groups(chosen,"fold","subject","session")
    bysubject=defaultdict(list)
    for (_,subject,_),vals in per_session.items():
        protected=next(v for v in vals if v["control"]=="PROTECTED")
        randoms=[v for v in vals if v["control"]=="RANDOM"]
        if len(randoms)!=RANDOM_DRAWS:raise RuntimeError("random draw count mismatch")
        bysubject[subject].append(float(protected["BA_loss"])-np.mean([float(v["BA_loss"]) for v in randoms]))
    result={s:float(np.mean(v)) for s,v in bysubject.items()}
    return result,float(np.mean(list(result.values())))


def pool_transfer(rows,source,successor,kind,field,fold=None):
    values=[float(r[field]) for r in rows if r["source_stage"]==source and r["successor_stage"]==successor and r["perturbation"]==kind and (fold is None or int(r["fold"])==fold)]
    if not values:raise RuntimeError("missing adjacent transfer")
    return float(np.mean(values))


def build_formation(buckets):
    rec=buckets["LAYERWISE_P_RECOVERABILITY.csv"]
    persist=buckets["LAYERWISE_PERSISTENCE.csv"]
    erasure=buckets["LAYERWISE_ERASURE_CONSEQUENCE.csv"]
    transfer=buckets["ADJACENT_TRANSFER.csv"]
    interaction=buckets["ADJACENT_PC_INTERACTION.csv"]
    lookup={(int(r["fold"]),r["stage"],r["split"]):r for r in rec if r["coordinate"]=="ALL"}
    rho={(int(r["fold"]),r["stage"],r["split"]):r for r in persist}
    curve=[]
    for fold in FOLDS:
        for stage in STAGES:
            row=lookup[(fold,stage,"DISCOVERY_SUBJECT_DISJOINT_TRIALS")]
            prior=SERIAL[SERIAL.index(stage)-1] if stage in SERIAL and stage!="H_CONCAT" else None
            delta=float(row["mean_R2"])-float(lookup[(fold,prior,"DISCOVERY_SUBJECT_DISJOINT_TRIALS")]["mean_R2"]) if prior else None
            curve.append({"task":TASK,"fold":fold,"stage":stage,"stage_role":"SERIAL" if stage in SERIAL else "PARALLEL_BRANCH",
                "discovery_mean_R2":float(row["mean_R2"]),"discovery_variance_weighted_R2":float(row["variance_weighted_R2"]),
                "delta_mean_R2_from_serial_predecessor":delta,"serial_predecessor":prior,
                "discovery_subjects":int(row["biological_subjects"])})
    for stage in STAGES:
        values=subject_stage_map(rec,stage,"mean_R2")
        ci=P.paired(values,{s:0. for s in values},f"layerwise_stage_{stage}_subject_R2")
        foldrows=[r for r in curve if r["stage"]==stage]
        prior=SERIAL[SERIAL.index(stage)-1] if stage in SERIAL and stage!="H_CONCAT" else None
        dc=P.paired(values,subject_stage_map(rec,prior,"mean_R2"),f"layerwise_delta_{stage}") if prior else None
        curve.append({"task":TASK,"fold":"POOLED_BIOLOGICAL_SUBJECTS","stage":stage,
            "stage_role":"SERIAL" if stage in SERIAL else "PARALLEL_BRANCH",
            "discovery_mean_R2":ci["delta"],"discovery_mean_R2_ci95_lower":ci["ci95_lower"],
            "discovery_mean_R2_ci95_upper":ci["ci95_upper"],
            "discovery_variance_weighted_R2":float(np.mean([float(r["discovery_variance_weighted_R2"]) for r in foldrows])),
            "delta_mean_R2_from_serial_predecessor":dc["delta"] if dc else None,
            "delta_ci95_lower":dc["ci95_lower"] if dc else None,
            "delta_ci95_upper":dc["ci95_upper"] if dc else None,
            "serial_predecessor":prior,"biological_subjects":ci["biological_subjects"]})
    growth=[];formation=[]
    for source,successor in TRANSITIONS:
        foldrecs=[]
        for fold in FOLDS:
            r0=lookup[(fold,source,"DISCOVERY_SUBJECT_DISJOINT_TRIALS")]
            r1=lookup[(fold,successor,"DISCOVERY_SUBJECT_DISJOINT_TRIALS")]
            pr0=rho[(fold,source,"DISCOVERY")];pr1=rho[(fold,successor,"DISCOVERY")]
            v0=float(pr0["mean_diagonal_cross_session_Pearson"])
            v1=float(pr1["mean_diagonal_cross_session_Pearson"])
            growth.append({"task":TASK,"fold":fold,"split":"DISCOVERY","source_stage":source,
                "successor_stage":successor,"source_rho":v0,"successor_rho":v1,"delta_rho":v1-v0})
            excess=erase_excess(erasure,successor,fold)[1]
            cp=pool_transfer(transfer,source,successor,"STRUCTURED_C","successor_P_movement_over_input_energy",fold)
            cc=pool_transfer(transfer,source,successor,"STRUCTURED_C","successor_C_movement_over_input_energy",fold)
            rp=pool_transfer(transfer,source,successor,"RANDOM_C","successor_P_movement_over_input_energy",fold)
            inter=next(float(r["successor_P_nonlinear_fraction"]) for r in interaction if int(r["fold"])==fold and r["source_stage"]==source)
            foldrecs.append({"task":TASK,"fold":fold,"source_stage":source,"successor_stage":successor,
                "source_discovery_mean_R2":float(r0["mean_R2"]),"successor_discovery_mean_R2":float(r1["mean_R2"]),
                "delta_recoverability":float(r1["mean_R2"])-float(r0["mean_R2"]),
                "source_persistence_rho":v0,"successor_persistence_rho":v1,"delta_persistence":v1-v0,
                "successor_protected_erasure_excess_BA_harm":excess,
                "structured_C_to_P":cp,"structured_C_to_C":cc,"random_C_to_P":rp,
                "structured_minus_random_C_to_P":cp-rp,
                "PC_nonlinear_interaction_into_successor_P":inter})
        c1=sum(r["delta_recoverability"]>0 for r in foldrecs)
        c2=sum(r["delta_persistence"]>0 for r in foldrecs)
        c3=sum(r["structured_minus_random_C_to_P"]>0 for r in foldrecs)
        c4=sum(r["successor_protected_erasure_excess_BA_harm"]>0 for r in foldrecs)
        pooled={key:float(np.mean([r[key] for r in foldrecs])) for key in (
            "source_discovery_mean_R2","successor_discovery_mean_R2","delta_recoverability",
            "source_persistence_rho","successor_persistence_rho","delta_persistence",
            "successor_protected_erasure_excess_BA_harm","structured_C_to_P",
            "structured_C_to_C","random_C_to_P","structured_minus_random_C_to_P",
            "PC_nonlinear_interaction_into_successor_P")}
        source_curve=next(r for r in curve if r["fold"]=="POOLED_BIOLOGICAL_SUBJECTS" and r["stage"]==source)
        successor_curve=next(r for r in curve if r["fold"]=="POOLED_BIOLOGICAL_SUBJECTS" and r["stage"]==successor)
        pooled["source_discovery_mean_R2"]=float(source_curve["discovery_mean_R2"])
        pooled["successor_discovery_mean_R2"]=float(successor_curve["discovery_mean_R2"])
        pooled["delta_recoverability"]=float(successor_curve["delta_mean_R2_from_serial_predecessor"])
        pooled["successor_protected_erasure_excess_BA_harm"]=erase_excess(erasure,successor)[1]
        tests={"recoverability":c1>=4,"persistence":c2>=4,
            "structured_C_to_P":c3>=4 and pooled["structured_minus_random_C_to_P"]>0,
            "erasure_consequence":c4>=4 and pooled["successor_protected_erasure_excess_BA_harm"]>0}
        active=sum(tests.values())>=3
        for r in foldrecs:
            r.update({"recoverability_positive_folds":c1,"persistence_positive_folds":c2,
                "structured_C_to_P_positive_folds":c3,"erasure_excess_positive_folds":c4,
                "criteria_satisfied":sum(tests.values()),"P_FORMATION_ACTIVE_TRANSITION":active})
        formation.extend(foldrecs)
        formation.append({"task":TASK,"fold":"POOLED","source_stage":source,"successor_stage":successor,
            **pooled,"recoverability_positive_folds":c1,"persistence_positive_folds":c2,
            "structured_C_to_P_positive_folds":c3,"erasure_excess_positive_folds":c4,
            "criteria_satisfied":sum(tests.values()),"P_FORMATION_ACTIVE_TRANSITION":active})
        growth.append({"task":TASK,"fold":"POOLED_FOLDS","split":"DISCOVERY",
            "source_stage":source,"successor_stage":successor,
            "source_rho":pooled["source_persistence_rho"],"successor_rho":pooled["successor_persistence_rho"],
            "delta_rho":pooled["delta_persistence"],"positive_folds":c2})
        for split in ("INNER_TRAIN",):
            for fold in FOLDS:
                v0=float(rho[(fold,source,split)]["mean_diagonal_cross_session_Pearson"])
                v1=float(rho[(fold,successor,split)]["mean_diagonal_cross_session_Pearson"])
                growth.append({"task":TASK,"fold":fold,"split":split,"source_stage":source,
                    "successor_stage":successor,"source_rho":v0,"successor_rho":v1,"delta_rho":v1-v0})
    return curve,growth,formation


def scope_table():
    lock=preflight();c=lock["cells"][0]
    model=A.build_model(TASK,2,Path(c["checkpoint_path"]))
    names={name:param.numel() for name,param in model.named_parameters()}
    groups={
        "SCOPE_A_LOCAL":("depth1.","point1."),
        "SCOPE_B_SHARED":("depth1.","point1.","depth2.","point2."),
        "SCOPE_C_SPATIAL_SHARED":("spatial.","depth1.","point1.","depth2.","point2."),
        "SCOPE_D_FULL_FEATURE":("temporal.","spatial.","depth1.","point1.","depth2.","point2."),
        "SCOPE_E_FEATURE_PLUS_EMBED":("temporal.","spatial.","depth1.","point1.","depth2.","point2.","embedding.")}
    out=[]
    for scope,prefixes in groups.items():
        selected={name:n for name,n in names.items() if name.startswith(prefixes)}
        if not selected:raise RuntimeError("empty trainable scope")
        out.append({"task":TASK,"scope":scope,"parameter_count":sum(selected.values()),
            "parameter_names":json.dumps(sorted(selected)),"classifier_head_frozen":True,
            "used_for_training_in_this_audit":False})
    if not all(out[i]["parameter_count"]<out[i+1]["parameter_count"] for i in range(len(out)-1)):
        raise RuntimeError("scope parameter counts not nested/increasing")
    return out


def branch_evidence(rows):
    evidence=[]
    for stage in STAGES[:3]:
        byfold=[]
        for fold in FOLDS:
            selected=[r for r in rows if r["stage"]==stage and int(r["fold"])==fold]
            grouped=groups(selected,"subject","session")
            harms=[];changes=[]
            for vals in grouped.values():
                p=next(r for r in vals if r["control"]=="PROTECTED")
                rand=[r for r in vals if r["control"]=="RANDOM"]
                harms.append(float(p["BA_loss"])-np.mean([float(r["BA_loss"]) for r in rand]))
                changes.append(float(p["final_P_change_L2_mean"])-np.mean([float(r["final_P_change_L2_mean"]) for r in rand]))
            byfold.append({"fold":fold,"excess_BA_harm":float(np.mean(harms)),
                "excess_final_P_change":float(np.mean(changes))})
        evidence.append({"stage":stage,
            "pooled_excess_BA_harm":float(np.mean([r["excess_BA_harm"] for r in byfold])),
            "positive_harm_folds":sum(r["excess_BA_harm"]>0 for r in byfold),
            "pooled_excess_final_P_change":float(np.mean([r["excess_final_P_change"] for r in byfold])),
            "positive_final_P_change_folds":sum(r["excess_final_P_change"]>0 for r in byfold),
            "folds":byfold})
    return evidence


def recommend(formation,branch):
    pooled=[r for r in formation if r["fold"]=="POOLED"]
    active={r["successor_stage"] for r in pooled if r["P_FORMATION_ACTIVE_TRANSITION"]}
    distinctive=[r["stage"] for r in branch if r["pooled_excess_BA_harm"]>0 and r["positive_harm_folds"]>=4 and r["pooled_excess_final_P_change"]>0 and r["positive_final_P_change_folds"]>=4]
    if "EMBEDDING" in active:scope="SCOPE_E_FEATURE_PLUS_EMBED"
    elif "H_SHARED2" in active:scope="SCOPE_B_SHARED"
    elif "H_SHARED1" in active:scope="SCOPE_A_LOCAL"
    else:scope="NO_UNIQUE_TRAINABLE_SCOPE"
    if "H_SHARED2" in active or "EMBEDDING" in active:verdict="LOCAL_SCOPE_TOO_NARROW"
    elif "H_SHARED1" in active:verdict="LOCAL_SCOPE_SUPPORTED"
    else:verdict="LOCAL_SCOPE_UNRESOLVED"
    return {"active_serial_successor_stages":sorted(active),
        "active_serial_transitions":[f"{r['source_stage']} -> {r['successor_stage']}" for r in pooled if r["P_FORMATION_ACTIVE_TRANSITION"]],
        "distinctive_branch_outputs":distinctive,"local_scope_verdict":verdict,
        "recommended_scope":scope,
        "rule":"3/4 locked criteria for each serial transition; smallest nested scope covering active parameterized transitions; branch output effects are diagnostic but do not by themselves require temporal or spatial parameters to train",
        "limitation":"Branch-output erasure cannot separate the parameter-specific effects of temporal versus spatial filters; a full temporal scope is not identifiable from this audit.",
        "development_diagnostic_only":True,"no_scope_trained":True}


def aggregate():
    lock=preflight();buckets={name:[] for name in CELL_FILES};geometry=[]
    for fold in FOLDS:
        d=rcell(fold);done=json.loads((d/"CELL_COMPLETE.json").read_text())
        if done["protocol_sha256"]!=P.sha(PROTOCOL/"PROTOCOL_LOCK.json") or done["checkpoint_sha256"]!=lock["cells"][fold]["checkpoint_sha256"] or done["neural_parameter_updates"]!=0 or done["outer_dev_eeg_reads"] or done["final_heldout_eeg_reads"]:
            raise RuntimeError(f"cell provenance/leakage failure fold{fold}")
        if any(P.sha(d/name)!=digest for name,digest in done["output_sha256"].items()):
            raise RuntimeError(f"cell output drift fold{fold}")
        audit=json.loads((d/"GEOMETRY_AUDIT.json").read_text())
        if not audit["historical_Q_source_successor_exact"] or audit["source_model_state_sha256"]!=audit["final_model_state_sha256"]:
            raise RuntimeError(f"geometry/frozen model audit failure fold{fold}")
        if audit["fitted_pathway_archive_sha256"]!=P.sha(d/"FITTED_PATHWAYS.npz") or done["fitted_pathway_archive_sha256"]!=audit["fitted_pathway_archive_sha256"]:
            raise RuntimeError(f"fitted pathway archive drift fold{fold}")
        geometry.append(audit)
        for name in CELL_FILES:buckets[name].extend(P.cread(d/name))
    prior_metrics=P.cread(PRIOR.parents[1]/"outputs/DISCOVERY_METRICS.csv")
    baseline={(r["fold"],r["subject"],r["session"]):r for r in prior_metrics if r["arm"]=="BASELINE"}
    intact={(r["fold"],r["subject"],r["session"]):r for r in buckets["LAYERWISE_ERASURE_CONSEQUENCE.csv"]
        if r["stage"]=="H_CONCAT" and r["control"]=="PROTECTED"}
    if intact.keys()!=baseline.keys() or any(abs(float(intact[k][a])-float(baseline[k][b]))>1e-6
        for k in intact for a,b in (("intact_BA","BA"),("intact_macro_F1","macro_F1"),("intact_NLL","NLL"))):
        raise RuntimeError("native intact discovery inference differs from prior canonical baseline")
    P.jwrite(OUT/"LAYERWISE_GEOMETRY_AUDIT.json",{"task":TASK,"folds":list(FOLDS),
        "train_only_pathfit_and_final_P":True,"all_historical_source_successor_hashes_exact":True,
        "cells":geometry})
    for r in buckets["LAYERWISE_P_RECOVERABILITY.csv"]:
        geom=next(g for g in geometry if g["fold"]==int(r["fold"]))
        s=next(s for s in geom["stages"] if s["stage"]==r["stage"])
        r["pathway_rank"]=s["rank"]
        r["feature_dimension"]=s["feature_dimension"]
        r["Q_mu_sha256"]=s["q_mu_sha256"]
    erasure_groups=groups(buckets["LAYERWISE_ERASURE_CONSEQUENCE.csv"],"fold","stage","subject","control","draw")
    for vals in erasure_groups.values():
        if len(vals)!=2:raise RuntimeError("erasure session pairing failure")
        worst=min(float(r["intact_BA"]) for r in vals)-min(float(r["erased_BA"]) for r in vals)
        for r in vals:r["worst_session_BA_loss"]=worst
    protected_by_key={(r["fold"],r["stage"],r["subject"],r["session"]):r
        for r in buckets["LAYERWISE_ERASURE_CONSEQUENCE.csv"] if r["control"]=="PROTECTED"}
    for r in buckets["LAYERWISE_UTILITY_DECOMPOSITION.csv"]:
        e=protected_by_key[(r["fold"],r["stage"],r["subject"],r["session"])]
        r["intact_macro_F1"]=e["intact_macro_F1"]
        r["intact_NLL"]=e["intact_NLL"]
        r["complement_retained_macro_F1"]=e["erased_macro_F1"]
        r["complement_retained_NLL"]=e["erased_NLL"]
    for name,rows in buckets.items():P.cwrite(OUT/name,rows)
    curve,growth,formation=build_formation(buckets)
    erasure_summaries=[]
    for stage in SERIAL:
        vals=buckets["LAYERWISE_ERASURE_CONSEQUENCE.csv"]
        bysub,excess=erase_excess(vals,stage)
        ci=P.paired(bysub,{s:0. for s in bysub},f"layerwise_erasure_excess_{stage}")
        p=[r for r in vals if r["stage"]==stage and r["control"]=="PROTECTED"]
        rand=[r for r in vals if r["stage"]==stage and r["control"]=="RANDOM"]
        erasure_summaries.append({"task":TASK,"fold":"POOLED_BIOLOGICAL_SUBJECTS","stage":stage,
            "control":"PROTECTED_MINUS_RANDOM","rank":p[0]["rank"],
            "biological_subjects":ci["biological_subjects"],"protected_BA_loss_mean":float(np.mean([float(r["BA_loss"]) for r in p])),
            "random_BA_loss_mean":float(np.mean([float(r["BA_loss"]) for r in rand])),
            "protected_minus_random_BA_harm":excess,
            "paired_ci95_lower":ci["ci95_lower"],"paired_ci95_upper":ci["ci95_upper"],
            "bootstrap_replicates":BOOTSTRAPS,
            "protected_worst_session_BA_loss_mean":float(np.mean([float(r["worst_session_BA_loss"]) for r in p])),
            "protected_macro_F1_loss_mean":float(np.mean([float(r["macro_F1_loss"]) for r in p])),
            "protected_NLL_change_mean":float(np.mean([float(r["NLL_change"]) for r in p]))})
    P.cwrite(OUT/"LAYERWISE_ERASURE_CONSEQUENCE.csv",buckets["LAYERWISE_ERASURE_CONSEQUENCE.csv"]+erasure_summaries)
    transfer_summaries=[]
    for source,successor in TRANSITIONS:
        tr=buckets["ADJACENT_TRANSFER.csv"]
        cp=pool_transfer(tr,source,successor,"STRUCTURED_C","successor_P_movement_over_input_energy")
        cc=pool_transfer(tr,source,successor,"STRUCTURED_C","successor_C_movement_over_input_energy")
        rp=pool_transfer(tr,source,successor,"RANDOM_C","successor_P_movement_over_input_energy")
        transfer_summaries.append({"task":TASK,"fold":"POOLED_FOLDS","source_stage":source,
            "successor_stage":successor,"perturbation":"STRUCTURED_VS_RANDOM_C_SUMMARY",
            "structured_C_to_P":cp,"structured_C_to_C":cc,"random_C_to_P":rp,
            "structured_minus_random_C_to_P":cp-rp,
            "positive_folds":sum(pool_transfer(tr,source,successor,"STRUCTURED_C","successor_P_movement_over_input_energy",f)>
                pool_transfer(tr,source,successor,"RANDOM_C","successor_P_movement_over_input_energy",f) for f in FOLDS)})
    P.cwrite(OUT/"ADJACENT_TRANSFER.csv",buckets["ADJACENT_TRANSFER.csv"]+transfer_summaries)
    P.cwrite(OUT/"P_FORMATION_CURVE.csv",curve)
    P.cwrite(OUT/"PERSISTENCE_GROWTH.csv",growth)
    P.cwrite(OUT/"P_FORMATION_TRANSITION_SUMMARY.csv",formation)
    scopes=scope_table();P.cwrite(OUT/"TRAINABLE_SCOPE_TABLE.csv",scopes)
    branch=branch_evidence(buckets["BRANCH_TO_FINAL_P_CONTRIBUTION.csv"])
    decision=recommend(formation,branch)
    P.jwrite(OUT/"TRAINABLE_SCOPE_RECOMMENDATION.json",decision)
    P.jwrite(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json",{
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,"FINAL_HELDOUT_ACCESSED":False,
        "neural_parameter_updates":0,"train_only_geometry":True,
        "train_only_perturbation_direction_selection":True,
        "native_discovery_inference_matches_prior_baseline":True,
        "discovery_roles_only_after_geometry_lock":True,"completed_cells":5,
        "historical_checkpoint_final_heldout_diagnostic_exposure":True,
        "historical_provenance_caveat":lock["historical_provenance_caveat"],
        "geometry_audit_sha256":[P.sha(rcell(f)/"GEOMETRY_AUDIT.json") for f in FOLDS]})
    report(buckets,curve,growth,formation,branch,scopes,decision)
    print("AGGREGATE_COMPLETE",flush=True)


def report(buckets,curve,growth,formation,branch,scopes,decision):
    stage_lines=[];pers=buckets["LAYERWISE_PERSISTENCE.csv"]
    erase=buckets["LAYERWISE_ERASURE_CONSEQUENCE.csv"]
    transfer=buckets["ADJACENT_TRANSFER.csv"]
    interactions=buckets["ADJACENT_PC_INTERACTION.csv"]
    for stage in STAGES:
        pooled=next(r for r in curve if r["fold"]=="POOLED_BIOLOGICAL_SUBJECTS" and r["stage"]==stage)
        oof=[r for r in buckets["LAYERWISE_P_RECOVERABILITY.csv"] if r["stage"]==stage and r["coordinate"]=="ALL" and r["split"]=="INNER_TRAIN_SUBJECT_GROUPED_OOF_CENTROIDS"]
        dr=[r for r in pers if r["stage"]==stage and r["split"]=="DISCOVERY"]
        tr=[r for r in pers if r["stage"]==stage and r["split"]=="INNER_TRAIN"]
        stage_lines.append(f"| {stage} | {np.mean([float(r['mean_R2']) for r in oof]):+.3f} | {float(pooled['discovery_mean_R2']):+.3f} [{float(pooled['discovery_mean_R2_ci95_lower']):+.3f}, {float(pooled['discovery_mean_R2_ci95_upper']):+.3f}] | {np.mean([float(r['mean_diagonal_cross_session_Pearson']) for r in tr]):+.3f} | {np.mean([float(r['mean_diagonal_cross_session_Pearson']) for r in dr]):+.3f} |")
    growth_lines=[]
    for source,successor in TRANSITIONS:
        row=next(r for r in growth if r["fold"]=="POOLED_FOLDS" and r["source_stage"]==source)
        foldrows=[r for r in growth if r["source_stage"]==source and r["split"]=="DISCOVERY" and str(r["fold"]).isdigit()]
        growth_lines.append(f"| {source} → {successor} | {float(row['delta_rho']):+.3f} | {sum(float(r['delta_rho'])>0 for r in foldrows)}/5 |")
    erase_lines=[]
    for stage in SERIAL:
        bysub,excess=erase_excess(erase,stage)
        ci=P.paired(bysub,{s:0. for s in bysub},f"layerwise_erasure_excess_{stage}")
        p=[r for r in erase if r["stage"]==stage and r["control"]=="PROTECTED"]
        rand=[r for r in erase if r["stage"]==stage and r["control"]=="RANDOM"]
        erase_lines.append(f"| {stage} | {np.mean([float(r['BA_loss']) for r in p]):+.4f} | {np.mean([float(r['BA_loss']) for r in rand]):+.4f} | {excess:+.4f} [{ci['ci95_lower']:+.4f}, {ci['ci95_upper']:+.4f}] | {np.mean([float(r['worst_session_BA_loss']) for r in p]):+.4f} | {np.mean([float(r['NLL_change']) for r in p]):+.4f} |")
    transfer_lines=[];interaction_lines=[];formation_lines=[]
    for source,successor in TRANSITIONS:
        def val(kind,field):return pool_transfer(transfer,source,successor,kind,field)
        pp=val("P","successor_P_movement_over_input_energy")
        pc=val("P","successor_C_movement_over_input_energy")
        cp=val("STRUCTURED_C","successor_P_movement_over_input_energy")
        cc=val("STRUCTURED_C","successor_C_movement_over_input_energy")
        rp=val("RANDOM_C","successor_P_movement_over_input_energy")
        transfer_lines.append(f"| {source} → {successor} | {pp:.3f} | {pc:.3f} | {cp:.3f} | {cc:.3f} | {rp:.3f} | {cp-rp:+.3f} |")
        interaction_lines.append(f"| {source} → {successor} | {np.mean([float(r['successor_P_nonlinear_fraction']) for r in interactions if r['source_stage']==source]):.4f} | {np.mean([float(r['successor_C_nonlinear_fraction']) for r in interactions if r['source_stage']==source]):.4f} |")
        row=next(r for r in formation if r["fold"]=="POOLED" and r["source_stage"]==source)
        formation_lines.append(f"| {source} → {successor} | {float(row['delta_recoverability']):+.3f} ({row['recoverability_positive_folds']}/5) | {float(row['delta_persistence']):+.3f} ({row['persistence_positive_folds']}/5) | {float(row['structured_minus_random_C_to_P']):+.3f} ({row['structured_C_to_P_positive_folds']}/5) | {float(row['successor_protected_erasure_excess_BA_harm']):+.4f} ({row['erasure_excess_positive_folds']}/5) | {row['criteria_satisfied']}/4 | {row['P_FORMATION_ACTIVE_TRANSITION']} |")
    branch_lines=[]
    rawbranch=buckets["BRANCH_TO_FINAL_P_CONTRIBUTION.csv"]
    for r in branch:
        stage=r["stage"]
        p=[x for x in rawbranch if x["stage"]==stage and x["control"]=="PROTECTED"]
        rand=[x for x in rawbranch if x["stage"]==stage and x["control"]=="RANDOM"]
        branch_lines.append(f"| {stage} | {np.mean([float(x['final_P_change_L2_mean']) for x in p]):.4f} | {np.mean([float(x['final_P_change_L2_mean']) for x in rand]):.4f} | {r['pooled_excess_final_P_change']:+.4f} ({r['positive_final_P_change_folds']}/5) | {r['pooled_excess_BA_harm']:+.4f} ({r['positive_harm_folds']}/5) | {np.mean([float(x['final_P_cosine_mean']) for x in p]):.4f} |")
    scope_lines=[f"| {r['scope']} | {r['parameter_count']} |" for r in scopes]
    local=decision["local_scope_verdict"];recommended=decision["recommended_scope"]
    text=f"""# Frozen SIRE layerwise Protected-formation audit, seed 0

Canonical CompactLite SIRE-EEG, OpenBMI MI, folds 0–4. The canonical selected checkpoints, PERSIST final coordinates, normalizers, split roles and H_CONCAT/H_SHARED1 geometry were hash checked. All pathway fitting used only inner-train subject-session-class centroids. The network remained in eval mode, with no optimizer and zero parameter or BN updates. Discovery subjects were used only for evaluation. Outer-dev and final-heldout EEG reads were zero. The reused checkpoints retain the historical final-heldout diagnostic exposure disclosed in the source record.

**Interpretation:** recoverability, cross-session persistence, prediction consequences and adjacent C→P transfer are distinct measurements. No stage is declared a formation transition from linear readout alone. Pooled discovery recoverability below is biological-subject equal; its 95% CI uses 20,000 biological-subject bootstrap draws. Persistence is a matched subject×class centroid statistic averaged across five folds.

## Q1. Where does final P become recoverable?

The final embedding uses the exact canonical train-fitted PERSIST whitening/eigenvector map, making its R² of 1 definitional rather than new formation evidence. Earlier stages use the source PathFit dual ridge, with subject-grouped OOF on train centroids and subject-disjoint discovery evaluation. The OOF split refits PathFit; the canonical final-P frame remains fixed from all inner-train subjects as required by the source protocol. Branches are parallel, so their rows are not serial steps.

| Stage | Train subject-grouped OOF mean R² | Discovery subject-equal mean R² [95% CI] | Train persistence rho | Discovery persistence rho |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(stage_lines)}

Coordinate-wise R², Pearson, variance-weighted R², cosine and normalized MSE are in `LAYERWISE_P_RECOVERABILITY.csv`. Per-fold serial differences and subject-paired CIs are in `P_FORMATION_CURVE.csv`.

## Q2. Where does cross-session persistence grow?

The same final-P coordinate system is used at every layer. `rho` is the mean diagonal cross-session Pearson correlation of matched subject×class centroids; the CSV also reports normalized symmetric covariance trace, centroid cosine, within-subject distance and a between-subject reference. Discovery has few matched centroids per fold, so these are development estimates.

| Serial transition | Discovery Δrho | Positive folds |
| --- | ---: | ---: |
{chr(10).join(growth_lines)}

## Q3. Where does P acquire predictive consequence?

The stage-specific fitted P span was erased and the exact frozen suffix was run. Four deterministic, equal-rank orthonormal random subspaces in the same stage served as controls. Positive excess means P erasure harms BA more than random erasure. The CI is paired by biological subject.

| Stage | P erasure BA loss | Random BA loss | Excess [95% CI] | P worst-session BA loss | P NLL change |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(erase_lines)}

The random subspaces are drawn in the fitted P orthogonal complement and are not covariance matched to the Protected span. Excess harm therefore establishes sensitivity relative to this locked control, but may also reflect concentration of native activation energy.

`LAYERWISE_UTILITY_DECOMPOSITION.csv` additionally reports intact, complement-retained and P-path-retained native continuation. The P-path-retained probe replaces C by the train centroid and can leave the native representation manifold; its utilities are diagnostic and are not additive attribution.

## Q4. Where does C causally contribute to successor P?

Each source P, structured top-16 source C, and random source-C perturbation had identical per-trial Euclidean energy (2% of the train source RMS norm). The table reports mean successor movement divided by that input energy. Directions and scales were selected without discovery labels.

| Transition | P→P | P→C | Structured C→P | Structured C→C | Random C→P | Structured minus random C→P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(transfer_lines)}

These are finite perturbations of the frozen native block. They show sensitivity, not how much a future optimizer can change the block.

## Q5. Where is P/C interaction nonlinear?

The interaction residual is `F(h+dP+dC)-F(h+dP)-F(h+dC)+F(h)`, projected into successor P and C. Each fraction divides its mean norm by the sum of the corresponding separate P- and C-perturbation response norms.

| Transition | Nonlinear successor-P fraction | Nonlinear successor-C fraction |
| --- | ---: | ---: |
{chr(10).join(interaction_lines)}

## Q6. Do temporal branches contribute differently?

Only the fitted P component of one branch was erased at a time; the other two branches remained native. The final-P coordinate change and classifier effect are compared with equal-rank random erasure in that same branch.

| Branch | Final-P L2 change | Random change | Excess change (positive folds) | Excess BA harm (positive folds) | Final-P cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(branch_lines)}

These branch-output interventions cannot distinguish the separate causal contribution of temporal filters from the grouped spatial filters.

## Q7. Was `depth1 + point1` alone too narrow?

`{local}`. The locked formation rule requires at least three of four criteria: discovery recoverability grows in ≥4/5 folds; persistence grows in ≥4/5 folds; structured C→P exceeds random C→P at the pooled point estimate and in ≥4/5 folds; successor P erasure has excess BA harm at the pooled point estimate and in ≥4/5 folds.

| Transition | Δrecoverability (positive folds) | Δpersistence (positive folds) | Structured-random C→P (positive folds) | Successor P-erasure excess (positive folds) | Criteria | Active |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(formation_lines)}

`H_SHARED1 → H_SHARED2` meets 3/4 criteria even though its measured persistence does not grow; the verdict does not assume monotonic persistence. `H_SHARED2 → EMBEDDING` fails the locked rule despite the definitional final-stage recoverability increase.

## Q8. Smallest evidence-supported future trainable scope

`{recommended}`. Active transitions: {', '.join(decision['active_serial_transitions']) or 'none'}. Distinctive branch outputs: {', '.join(decision['distinctive_branch_outputs']) or 'none'}. This is a diagnostic scope recommendation, not a trained-model result. Branch-output measurements cannot uniquely identify temporal versus spatial filter trainability.

The minimum scope covering the active *parameterized serial transitions* is selected. Branch-output erasure shows that frozen branch signals matter, but does not establish that updating spatial or temporal filters is necessary; it therefore does not promote the recommendation from B to C or D.
The recommendation locates the observed frozen formation path. It does not prove that training the broader block will improve accuracy or Protected persistence.

| Candidate | Trainable parameter count |
| --- | ---: |
{chr(10).join(scope_lines)}

No classifier or method was trained. Large activations were held in runtime memory; only compact CSV/JSON evidence and provenance receipts are committed.
"""
    (OUT/"FINAL_REPORT.md").write_text(text,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","smoke","cell","aggregate"))
    parser.add_argument("--fold",type=int)
    args=parser.parse_args()
    if args.mode=="preflight":preflight()
    elif args.mode=="smoke":smoke()
    elif args.mode=="cell":
        if args.fold is None:raise ValueError("cell requires --fold")
        run_cell(args.fold)
    else:aggregate()


if __name__=="__main__":main()
