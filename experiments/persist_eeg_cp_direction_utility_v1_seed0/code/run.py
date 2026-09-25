"""Frozen, development-only EEGNet Complement-direction utility audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

EXP=Path(__file__).resolve().parents[1]
REPO=EXP.parents[1]
V2_EXP=REPO/"experiments"/"persist_eeg_native_cp_transfer_gate_v2_seed0"
V2_CODE=V2_EXP/"code"/"run.py"
V2_RUNTIME=Path(os.environ.get("NATIVE_GATE_V2_RUNTIME",str(REPO.parent/"native_gate_v2_strict_runtime"))).resolve()
os.environ["NATIVE_GATE_RUNTIME"]=str(V2_RUNTIME)
SPEC=importlib.util.spec_from_file_location("cp_utility_native_gate_v2",V2_CODE)
assert SPEC and SPEC.loader
V2=importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name]=V2; SPEC.loader.exec_module(V2)
B=V2.B
TASKS=B.TASKS
FOLDS=range(5)
RANDOM_DRAWS=20
MAX_C_DIRECTIONS=16
EPSILONS=(0.25,0.5,1.0)
DEVICE=B.DEVICE
ROOT=Path(os.environ.get("CP_DIRECTION_RUNTIME",str(REPO.parent/"cp_direction_utility_v1_runtime"))).resolve()
OUT=EXP/"outputs"
PROTOCOL=EXP/"protocol"
torch.set_num_threads(min(int(os.environ.get("CP_DIRECTION_CPU_THREADS","8")),os.cpu_count() or 1))


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()


def cell(task,fold,stage):
    return ROOT/stage/task.lower()/f"fold{fold}_seed0"


def manifest(task,fold):
    rec,split,cache,src,future=B.role(task,fold)
    return rec,split,cache,tuple(src),int(future)


def heldout_exclusion_audit():
    return {"FINAL_HELDOUT_ACCESSED":False,"heldout_eeg_array_reads":0,
            "OpenBMI_final_heldout_subjects_used":0,"WBCIC_true_outer_subjects_used":0,
            "loader_policy":"B.rows(..., final=False) only; no final-heldout loader called",
            "analysis_scope":"inner_train model-fit subjects and inner_val discovery subjects",
            "outer_dev_subjects_used":0,"final_heldout_ids_or_arrays_loaded":False}


def preflight():
    if (PROTOCOL/"PROTOCOL_LOCK.json").exists():raise RuntimeError("protocol already locked")
    v2_lock=json.loads((V2_EXP/"protocol"/"PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if v2_lock.get("code_sha256")!=sha(V2_CODE):raise RuntimeError("V2 code differs from its protocol lock")
    audit_path=V2_EXP/"outputs"/"PROJECTOR_AUDIT.csv"
    projector_audit={(r["task"],int(r["fold"]),r["stage"]):r for r in csv.DictReader(audit_path.open(newline="",encoding="utf-8"))}
    anchors=[]; projectors=[]
    for task in TASKS:
        for fold in FOLDS:
            role,split,cache,source_sessions,future=B.role(task,fold)
            record,ck=B.source_anchor(task,fold)
            if record["split_sha256"]!=split:raise RuntimeError(f"canonical anchor split mismatch: {task}/{fold}")
            d=V2.cell(task,fold,"discovery")
            c=json.loads((d/"COMPLETE.json").read_text(encoding="utf-8"))
            info=json.loads((d/"projector.json").read_text(encoding="utf-8"))
            npz=d/"projectors.npz"
            if c.get("status")!="COMPLETE" or info.get("status")!="COMPLETE" or sha(npz)!=c["projectors_sha256"]:
                raise RuntimeError(f"V2 discovery projector unavailable or changed: {task}/{fold}")
            arr=np.load(npz,allow_pickle=False)
            ph=B.arr_sha(arr["qs"],arr["ms"],arr["qd"],arr["md"])
            pa=projector_audit[task,fold,"discovery"]
            if ph!=info["projector_hash"] or ph!=pa["projector_hash"]:
                raise RuntimeError(f"V2 discovery projector provenance mismatch: {task}/{fold}")
            anchors.append({"task":task,"fold":fold,"checkpoint_sha256":sha(ck),
                            "anchor_selected_epoch":int(record["selected_epoch"]),
                            "split_sha256":split,"normalizer_sha256":record["normalizer"]["mean_std_sha256"],
                            "train_subjects_hash":hashlib.sha256(json.dumps(sorted(map(str,role["inner_train_subjects"]))).encode()).hexdigest(),
                            "discovery_subjects_hash":hashlib.sha256(json.dumps(sorted(map(str,role["inner_val_subjects"]))).encode()).hexdigest()})
            projectors.append({"task":task,"fold":fold,"status":"COMPLETE","source":"V2 discovery TRAIN-only",
                               "projector_file_sha256":sha(npz),"projector_hash":ph,
                               "protected_rank":info["protected_rank"],"spatial_rank":arr["qs"].shape[1],
                               "successor_rank":arr["qd"].shape[1],"train_data_hash":info["train_data_hash"],
                               "spatial_orth_error":float(np.max(np.abs(arr["qs"].T@arr["qs"]-np.eye(arr["qs"].shape[1])))),
                               "successor_orth_error":float(np.max(np.abs(arr["qd"].T@arr["qd"]-np.eye(arr["qd"].shape[1]))))})
            if projectors[-1]["spatial_orth_error"]>=1e-5 or projectors[-1]["successor_orth_error"]>=1e-5:
                raise RuntimeError("V2 projector orthogonality check failed")
    record={"schema":"PERSIST_EEG_CP_DIRECTION_UTILITY_V1_SEED0","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "git_source_commit":os.environ.get("CP_DIRECTION_SOURCE_COMMIT","UNCOMMITTED"),
            "code_sha256":sha(Path(__file__).resolve()),
            "tasks":TASKS,"folds":list(FOLDS),"seed":0,"backbone":"canonical frozen EEGNet",
            "train_role":"inner_train_subjects","discovery_role":"inner_val_subjects",
            "excluded_role":"outer_dev_subjects; all formal final heldout subjects and arrays",
            "FINAL_HELDOUT_ACCESSED":False,"final_heldout_exclusion_audit":heldout_exclusion_audit(),
            "construction_zone":"spatial_elu_pool1 -> depth_point_elu_pool2",
            "projector_source":"hash-verified V2 discovery-stage TRAIN-only orthogonal projectors",
            "C_basis":{"fit":"unsupervised PCA on TRAIN spatial complement only; labels never used to fit basis",
                       "max_rank":MAX_C_DIRECTIONS,"rank_tolerance":"singular value > max(s0*1e-6,1e-8)",
                       "PCA":"randomized solver, deterministic per-cell seed, signs canonicalized by largest-magnitude loading"},
            "estimands":{"primary":"native-coordinate erasure U_P=m(z0,y)-m(z_-j^P,y)",
                          "direct":"U_C=m(z0,y)-m(z_-j^C,y)","total":"U_total=m(z0,y)-m(z_-j,y)",
                          "interaction_residual":"U_total-U_P-U_C; descriptive nonadditive residual, not causal decomposition",
                          "subject_pooling":"average session means within biological subject, then subject-equal"},
            "finite_difference":{"epsilons":EPSILONS,"sigma":"TRAIN standard deviation of native coefficient a_j",
                                 "A_P":"a_j*(m(z_+^P)-m(z_-^P))/(2*epsilon); sign invariant; PCA basis sensitivity only"},
            "random_control":{"draws":RANDOM_DRAWS,"seed":"stable SHA256 per task/fold/draw","basis":"Gaussian QR after projection into spatial complement",
                              "rank":"exactly K_C","same_primary_audit":True,"finite_difference":"PCA basis only"},
            "bootstrap":{"unit":"biological subject","draws":20000,"seed":"stable SHA256 per task/fold/role/basis"},
            "direction_index_policy":"fold-local; never equate PCA indices across folds",
            "source_hashes":{"V2_code":sha(V2_CODE),"V2_protocol":sha(V2_EXP/"protocol"/"PROTOCOL_LOCK.json"),
                             "canonical_EEGNet":sha(B.SEVEN_CODE/"backbone_models.py"),
                             "cache_loader":sha(B.SEVEN_CODE/"tech_recipe_selection.py"),
                             "PERSIST_selection":sha(B.REPO/"experiments"/"persist_eeg_crossbackbone_peeh_v1"/"code"/"run_crossbackbone_peeh.py")},
            "baseline_checkpoints":anchors,"reused_projectors":projectors}
    B.json_write(PROTOCOL/"PROTOCOL_LOCK.json",record)
    (PROTOCOL/"PROTOCOL_LOCK.sha256").write_text(sha(PROTOCOL/"PROTOCOL_LOCK.json")+"\n",encoding="utf-8")
    B.json_write(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json",heldout_exclusion_audit())
    B.csv_write(OUT/"PROJECTOR_AUDIT.csv",projectors)
    print("PREFLIGHT_LOCKED",len(projectors),flush=True)


def metadata(task,fold):
    role,split,cache,src,future=manifest(task,fold)
    train=set(map(str,role["inner_train_subjects"]))
    val=set(map(str,role["inner_val_subjects"]))
    outer=set(map(str,role["outer_dev_subjects"]))
    if train&val or train&outer or val&outer:raise RuntimeError("development roles overlap")
    record,ck=B.source_anchor(task,fold)
    if record["split_sha256"]!=split:raise RuntimeError("anchor split hash mismatch")
    return {"role":role,"split":split,"cache_name":cache,"source_sessions":src,"future_session":future,
            "normalizer":record["normalizer"],"record":record,"checkpoint":ck}


def load_role_arrays(task,data,subjects, sessions):
    xs=[];ys=[];ss=[];ses=[];mapping=dict(data["mapping"])
    for session in sessions:
        x,y,s,mapping=B.rows(task,list(map(str,subjects)),(int(session),),data["cache_name"],mapping)
        x=((x-data["mu"][None,:,None])/np.maximum(data["sd"][None,:,None],1e-6)).astype(np.float32)
        xs.append(x);ys.append(y.astype(np.int64));ss.append(s.astype(str));ses.append(np.full(len(y),int(session),dtype=np.int64))
    if not xs:raise RuntimeError("empty role session list")
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ss),np.concatenate(ses)


def model_and_train_data(task,fold):
    data=metadata(task,fold)
    train_ids=list(map(str,data["role"]["inner_train_subjects"]))
    source,sy,ss,mapping=B.rows(task,train_ids,data["source_sessions"],data["cache_name"])
    future,fy,fs,_=B.rows(task,train_ids,(data["future_session"],),data["cache_name"],mapping)
    _,_,norm,mu,sd=B.normalize(source,future)
    if norm["mean_std_sha256"]!=data["normalizer"]["mean_std_sha256"]:raise RuntimeError("TRAIN normalizer differs from frozen anchor")
    data.update({"mu":mu,"sd":sd,"mapping":mapping,"classes":int(np.unique(sy).size),
                 "channels":int(source.shape[1]),"samples":int(source.shape[2])})
    model,record,ck=B.load_anchor(task,fold,data)
    return data,model,record,ck


def load_projectors(task,fold):
    path=V2.cell(task,fold,"discovery")/"projectors.npz"
    arr=np.load(path,allow_pickle=False)
    return {k:arr[k].astype(np.float32) for k in ("qs","qd","ms","md")}


def spatial(model,x,batch=128):
    parts=[]; model.eval()
    with torch.inference_mode():
        for i in range(0,len(x),batch):
            xb=torch.from_numpy(np.ascontiguousarray(x[i:i+batch])).to(DEVICE)
            s=model.drop1(model.pool1(F.elu(model.bn2(model.spatial(model.bn1(model.temporal(xb.unsqueeze(1))))))))
            parts.append(s.flatten(1).float().cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def f0(model,hflat,samples):
    s=hflat.reshape(-1,16,1,samples//4)
    return model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(s)))))).flatten(1)


def suffix(model,yflat):
    return model.head(model.embedding(yflat))


def margins(z,y):
    truth=z.gather(1,y.reshape(-1,1)).squeeze(1)
    masked=z.clone(); masked.scatter_(1,y.reshape(-1,1),-torch.inf)
    return truth-masked.max(1).values


def full_forward(model,hs,labels,qd,md,samples,batch=128):
    ys=[]; ps=[]; cs=[]; zs=[]; ms=[]
    q=torch.from_numpy(qd).to(DEVICE); mu=torch.from_numpy(md).to(DEVICE)
    yy=torch.as_tensor(labels,dtype=torch.long,device=DEVICE)
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(hs),batch):
            h=torch.from_numpy(np.ascontiguousarray(hs[i:i+batch])).to(DEVICE)
            y=f0(model,h,samples).float(); p=(y-mu)@q; c=(y-mu)-p@q.T
            z=suffix(model,y).float(); m=margins(z,yy[i:i+batch])
            ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy()); cs.append(c.cpu().numpy()); zs.append(z.cpu().numpy()); ms.append(m.cpu().numpy())
    return tuple(np.concatenate(x).astype(np.float32) for x in (ys,ps,cs,zs,ms))


def make_c_basis(hs,qs,mu,task,fold):
    centered=hs-mu[None,:]
    hc=centered-(centered@qs)@qs.T
    maxrank=min(MAX_C_DIRECTIONS,len(hc)-1,hc.shape[1]-qs.shape[1])
    if maxrank<1:raise RuntimeError("empty spatial complement")
    pca=PCA(n_components=maxrank,svd_solver="randomized",random_state=B.stable_seed("C-PCA-TRAIN",task,fold,0))
    pca.fit(hc)
    vals=pca.singular_values_
    rank=int((vals>max(float(vals[0])*1e-6,1e-8)).sum()) if len(vals) else 0
    if rank<1:return None,{"status":"UNDEFINED_C_RANK","singular_values":vals.tolist()}
    u=pca.components_[:rank].T.astype(np.float64)
    q=qs.astype(np.float64)
    u-=q@(q.T@u)
    u,_=np.linalg.qr(u,mode="reduced")
    for j in range(u.shape[1]):
        pivot=int(np.argmax(np.abs(u[:,j])))
        if u[pivot,j]<0:u[:,j]*=-1
    u=u.astype(np.float32)
    record={"status":"COMPLETE","K_C":int(u.shape[1]),"singular_values":vals[:rank].tolist(),
            "explained_variance_ratio":pca.explained_variance_ratio_[:rank].tolist(),
            "pca_seed":B.stable_seed("C-PCA-TRAIN",task,fold,0),"c_basis_hash":B.arr_sha(u),
            "orthogonality_error":float(np.max(np.abs(u.T@u-np.eye(u.shape[1])))),
            "protected_overlap_error":float(np.max(np.abs(qs.T@u))),
            "c_reconstruction_max_abs":float(np.max(np.abs(centered-(centered@qs)@qs.T-hc)))}
    return u,record


def random_complement_bases(qs,rank,task,fold):
    bases=[];seeds=[];hashes=[]
    q=qs.astype(np.float64)
    for r in range(RANDOM_DRAWS):
        seed=B.stable_seed("CP_DIRECTION_RANDOM_COMPLEMENT",task,fold,r,0)
        rng=np.random.default_rng(seed)
        z=rng.standard_normal((qs.shape[0],rank))
        z-=q@(q.T@z)
        u,_=np.linalg.qr(z,mode="reduced")
        if u.shape[1]!=rank:raise RuntimeError("random complement rank mismatch")
        for j in range(rank):
            pivot=int(np.argmax(np.abs(u[:,j])))
            if u[pivot,j]<0:u[:,j]*=-1
        u=u.astype(np.float32)
        if np.max(np.abs(qs.T@u))>=1e-5 or np.max(np.abs(u.T@u-np.eye(rank)))>=1e-5:
            raise RuntimeError("random basis is not orthogonal")
        bases.append(u);seeds.append(seed);hashes.append(B.arr_sha(u))
    return np.stack(bases),seeds,hashes


def train_arrays(task,data):
    subjects=list(map(str,data["role"]["inner_train_subjects"]))
    sessions=tuple(sorted(set(data["source_sessions"]+(data["future_session"],))))
    return load_role_arrays(task,data,subjects,sessions)


def discovery_arrays(task,data):
    ids=list(map(str,data["role"]["inner_val_subjects"]))
    sessions=tuple(sorted(set(data["source_sessions"]+(data["future_session"],))))
    return load_role_arrays(task,data,ids,sessions)


def require_locked_protocol():
    lock=PROTOCOL/"PROTOCOL_LOCK.json";digest=PROTOCOL/"PROTOCOL_LOCK.sha256"
    if not lock.is_file() or not digest.is_file() or sha(lock)!=digest.read_text(encoding="utf-8").strip():
        raise RuntimeError("protocol lock absent or hash mismatch")
    obj=json.loads(lock.read_text(encoding="utf-8"))
    if obj.get("FINAL_HELDOUT_ACCESSED") is not False or obj.get("tasks")!=list(TASKS):
        raise RuntimeError("protocol scope or heldout policy mismatch")
    return obj


def assert_frozen(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)


def project_complement(hs,qs,mu):
    centered=hs-mu[None,:]
    hc=centered-(centered@qs)@qs.T
    err=float(np.max(np.abs(centered-((centered@qs)@qs.T)-hc)))
    if err>=1e-5:raise RuntimeError(f"spatial P/C reconstruction failed: {err}")
    return hc,err


def eval_primary(model,hs,labels,subjects,sessions,qs,ms,qd,md,bases,samples,role_name,batch_samples=128,batch_dirs=8):
    """Evaluate native erasures for every basis direction, retaining no trial-level files."""
    n=len(hs);nd=bases.shape[2]
    y0,p0,c0,z0,m0=full_forward(model,hs,labels,qd,md,samples,batch=batch_samples)
    hc,_=project_complement(hs,qs,ms)
    coeff=np.stack([hc@bases[b] for b in range(bases.shape[0])],axis=0) if bases.ndim==3 else hc@bases
    # Normalize basis layout to [basis, feature, direction].
    if bases.ndim==2:
        bases=bases[None,:,:];coeff=coeff[None,:,:]
    n_bases=bases.shape[0];n_dir=bases.shape[2];n_all=n_bases*n_dir
    up=np.empty((n,n_all),np.float32);uc=np.empty_like(up);ut=np.empty_like(up);ui=np.empty_like(up);dp=np.empty_like(up)
    qd_t=torch.from_numpy(qd).to(DEVICE);md_t=torch.from_numpy(md).to(DEVICE)
    y0_t=torch.from_numpy(y0).to(DEVICE);p0_t=torch.from_numpy(p0).to(DEVICE);m0_t=torch.from_numpy(m0).to(DEVICE)
    yy=torch.as_tensor(labels,dtype=torch.long,device=DEVICE)
    model.eval()
    with torch.inference_mode():
        for i in range(0,n,batch_samples):
            j=min(n,i+batch_samples);xb=torch.from_numpy(np.ascontiguousarray(hs[i:j])).to(DEVICE)
            for b in range(n_bases):
                ub=bases[b]
                for d0 in range(0,n_dir,batch_dirs):
                    d1=min(n_dir,d0+batch_dirs);u=torch.from_numpy(np.ascontiguousarray(ub[:,d0:d1].T)).to(DEVICE)
                    a=torch.from_numpy(np.ascontiguousarray(coeff[b,i:j,d0:d1])).to(DEVICE)
                    hminus=xb[:,None,:]-a[:,:,None]*u[None,:,:]
                    ym=f0(model,hminus.reshape(-1,xb.shape[1]),samples).float().reshape(j-i,d1-d0,-1)
                    pm=(ym-md_t)@qd_t
                    delta=pm-p0_t[i:j,None,:]
                    ypatchp=y0_t[i:j,None,:]+delta@qd_t.T
                    ypatchc=ym-delta@qd_t.T
                    zminus=suffix(model,ym.reshape(-1,ym.shape[-1])).float().reshape(j-i,d1-d0,-1)
                    zp=suffix(model,ypatchp.reshape(-1,ypatchp.shape[-1])).float().reshape(j-i,d1-d0,-1)
                    zc=suffix(model,ypatchc.reshape(-1,ypatchc.shape[-1])).float().reshape(j-i,d1-d0,-1)
                    yyb=yy[i:j,None].expand(-1,d1-d0).reshape(-1)
                    mm=lambda z:margins(z.reshape(-1,z.shape[-1]),yyb).reshape(j-i,d1-d0)
                    base=m0_t[i:j,None]
                    ix=(b*n_dir+d0,b*n_dir+d1)
                    up[i:j,ix[0]:ix[1]]=(base-mm(zp)).cpu().numpy()
                    uc[i:j,ix[0]:ix[1]]=(base-mm(zc)).cpu().numpy()
                    ut[i:j,ix[0]:ix[1]]=(base-mm(zminus)).cpu().numpy()
                    ui[i:j,ix[0]:ix[1]]=ut[i:j,ix[0]:ix[1]]-up[i:j,ix[0]:ix[1]]-uc[i:j,ix[0]:ix[1]]
                    dp[i:j,ix[0]:ix[1]]=torch.linalg.vector_norm(delta,dim=-1).cpu().numpy()
    return {"U_P":up,"U_C":uc,"U_total":ut,"U_int":ui,"delta_p_norm":dp,"baseline_margin":m0,
            "subjects":np.asarray(subjects).astype(str),"sessions":np.asarray(sessions,dtype=np.int64),"labels":np.asarray(labels,dtype=np.int64),
            "n_directions":n_dir,"n_bases":n_bases,"role":role_name}


def eval_finite_difference(model,hs,labels,qs,ms,qd,md,basis,sigma,samples,epsilons=EPSILONS,batch_samples=128,batch_dirs=8):
    n=len(hs);k=basis.shape[1];y0,p0,_,_,m0=full_forward(model,hs,labels,qd,md,samples,batch=batch_samples)
    hc,_=project_complement(hs,qs,ms);a_all=hc@basis
    qd_t=torch.from_numpy(qd).to(DEVICE);md_t=torch.from_numpy(md).to(DEVICE)
    y0_t=torch.from_numpy(y0).to(DEVICE);p0_t=torch.from_numpy(p0).to(DEVICE);m0_t=torch.from_numpy(m0).to(DEVICE)
    yy=torch.as_tensor(labels,dtype=torch.long,device=DEVICE)
    result={eps:np.empty((n,k),np.float32) for eps in epsilons}
    model.eval()
    with torch.inference_mode():
        for eps in epsilons:
            for i in range(0,n,batch_samples):
                j=min(n,i+batch_samples);xb=torch.from_numpy(np.ascontiguousarray(hs[i:j])).to(DEVICE)
                for d0 in range(0,k,batch_dirs):
                    d1=min(k,d0+batch_dirs);u=torch.from_numpy(np.ascontiguousarray(basis[:,d0:d1].T)).to(DEVICE)
                    scale=torch.from_numpy(np.ascontiguousarray(sigma[d0:d1])).to(DEVICE)
                    delta=eps*scale[None,:,None]*u[None,:,:]
                    hp=xb[:,None,:]+delta;hm=xb[:,None,:]-delta
                    yp=f0(model,hp.reshape(-1,xb.shape[1]),samples).float().reshape(j-i,d1-d0,-1)
                    ym=f0(model,hm.reshape(-1,xb.shape[1]),samples).float().reshape(j-i,d1-d0,-1)
                    pp=(yp-md_t)@qd_t;pm=(ym-md_t)@qd_t
                    yp_patch=y0_t[i:j,None,:]+(pp-p0_t[i:j,None,:])@qd_t.T
                    ym_patch=y0_t[i:j,None,:]+(pm-p0_t[i:j,None,:])@qd_t.T
                    zp=suffix(model,yp_patch.reshape(-1,yp_patch.shape[-1])).float()
                    zm=suffix(model,ym_patch.reshape(-1,ym_patch.shape[-1])).float()
                    yyb=yy[i:j,None].expand(-1,d1-d0).reshape(-1)
                    mp=margins(zp,yyb).reshape(j-i,d1-d0);mn=margins(zm,yyb).reshape(j-i,d1-d0)
                    deriv=(mp-mn)/(2.0*eps)
                    coeff=torch.from_numpy(np.ascontiguousarray(a_all[i:j,d0:d1])).to(DEVICE)
                    result[eps][i:j,d0:d1]=(coeff*deriv).cpu().numpy()
    return result


def grouped_trial_rows(values,subjects,sessions,basis_names,direction_ids,fd=None):
    rows=[]
    subjects=np.asarray(subjects).astype(str);sessions=np.asarray(sessions,dtype=np.int64)
    keys=sorted(set(zip(subjects.tolist(),sessions.tolist())),key=lambda x:(x[0],x[1]))
    for sub,ses in keys:
        ix=np.flatnonzero((subjects==sub)&(sessions==ses))
        for d in range(values["U_P"].shape[1]):
            rec={"subject":sub,"session":int(ses),"basis_type":basis_names[d][0],"basis_draw":int(basis_names[d][1]),
                 "direction":int(direction_ids[d]),"n_trials":int(len(ix))}
            for name in ("U_P","U_C","U_total","U_int","delta_p_norm"):
                v=values[name][ix,d]
                rec[name+"_mean"]=float(np.mean(v));rec[name+"_median"]=float(np.median(v))
                if name in ("U_P","U_C","U_total"):
                    rec[name+"_positive_trial_fraction"]=float(np.mean(v>0));rec[name+"_negative_trial_fraction"]=float(np.mean(v<0))
            if fd:
                for eps,mat in fd.items():rec[f"A_P_eps_{eps:g}_mean"]=float(mat[ix,d].mean()) if d<mat.shape[1] else None
            rows.append(rec)
    return rows


def subject_equal_matrix(rows,field,ids,n_directions,basis_types,basis_draws):
    out=np.full((len(ids),n_directions*len(basis_types)*len(basis_draws)),np.nan,np.float64)
    idmap={s:i for i,s in enumerate(ids)}
    for r in rows:
        d=int(r["direction"]);bt=r["basis_type"];bd=int(r["basis_draw"])
        ix=((basis_types.index(bt)*len(basis_draws)+basis_draws.index(bd))*n_directions+d)
        out[idmap[str(r["subject"])],ix]+=float(r[field]) if np.isfinite(out[idmap[str(r["subject"])],ix]) else 0.0
        # Track number of sessions separately, then divide below.
    counts=np.zeros_like(out)
    for r in rows:
        ix=((basis_types.index(r["basis_type"])*len(basis_draws)+basis_draws.index(int(r["basis_draw"]))) * n_directions+int(r["direction"]))
        counts[idmap[str(r["subject"])],ix]+=1
    with np.errstate(invalid="ignore",divide="ignore"):
        out=np.where(counts>0,out/counts,np.nan)
    return out


def bootstrap_ci(subject_values,seed,draws=20000,chunk=256):
    x=np.asarray(subject_values,dtype=np.float64)
    if x.ndim==1:x=x[:,None]
    x=x[np.isfinite(x).all(axis=1)]
    if len(x)<2:return np.full(x.shape[1],np.nan),np.full(x.shape[1],np.nan)
    rng=np.random.default_rng(seed);boot=np.empty((draws,x.shape[1]),np.float32)
    for start in range(0,draws,chunk):
        stop=min(draws,start+chunk);idx=rng.integers(0,len(x),(stop-start,len(x)))
        boot[start:stop]=x[idx].mean(axis=1)
    return np.quantile(boot,0.025,axis=0),np.quantile(boot,0.975,axis=0)


def direction_summary(subject_rows,task,fold,stage,basis_names,direction_ids,fd_fields=()):
    ids=sorted({str(r["subject"]) for r in subject_rows})
    if not ids:raise RuntimeError("no biological subjects in utility table")
    summaries=[]
    keylist=sorted(set((r["basis_type"],int(r["basis_draw"]),int(r["direction"])) for r in subject_rows))
    bykey={k:[r for r in subject_rows if (r["basis_type"],int(r["basis_draw"]),int(r["direction"]))==k] for k in keylist}
    subject_matrices={}
    for metric in ("U_P_mean","U_C_mean","U_total_mean","U_int_mean","delta_p_norm_mean",*fd_fields):
        mat=np.full((len(ids),len(keylist)),np.nan,np.float64);si={s:i for i,s in enumerate(ids)}
        accum={}
        for r in subject_rows:
            k=(r["basis_type"],int(r["basis_draw"]),int(r["direction"]))
            value=r.get(metric)
            if value is None or value=="":continue
            accum.setdefault((str(r["subject"]),k),[]).append(float(value))
        ki={k:i for i,k in enumerate(keylist)}
        for (s,k),vals in accum.items():mat[si[s],ki[k]]=float(np.mean(vals))
        subject_matrices[metric]=mat
    ci_by_key={};basis_groups={}
    for i,k in enumerate(keylist):basis_groups.setdefault(k[:2],[]).append((i,k))
    for (bt,bd),items in basis_groups.items():
        cols=[i for i,_ in items];x=subject_matrices["U_P_mean"][:,cols]
        good=np.isfinite(x).all(axis=1);seed=B.stable_seed("CP_DIRECTION_BOOTSTRAP",task,fold,stage,bt,bd)
        lo,hi=bootstrap_ci(x[good],seed)
        for j,(_,k) in enumerate(items):ci_by_key[k]=(float(lo[j]),float(hi[j]),seed)
    for i,k in enumerate(keylist):
        bt,bd,d=k
        upvals=subject_matrices["U_P_mean"][:,i]
        good=np.isfinite(upvals);nsub=int(good.sum())
        lo_value,hi_value,seed=ci_by_key[k]
        rec={"task":task,"fold":fold,"stage":stage,"basis_type":bt,"basis_draw":bd,"direction":d,
             "n_subjects":nsub,"bootstrap_draws":20000,"bootstrap_seed":seed,
             "U_P_subject_equal_mean":float(np.mean(upvals[good])),"U_P_CI95_lower":lo_value,"U_P_CI95_upper":hi_value}
        for m in ("U_C_mean","U_total_mean","U_int_mean","delta_p_norm_mean",*fd_fields):
            v=subject_matrices[m][:,i];v=v[np.isfinite(v)];rec[m.replace("_mean","_subject_equal_mean")]=float(v.mean()) if len(v) else None
        if lo_value>0:rec["train_candidate"]="C_PLUS_CANDIDATE" if stage=="TRAIN" else ""
        elif hi_value<0:rec["train_candidate"]="C_MINUS_CANDIDATE" if stage=="TRAIN" else ""
        else:rec["train_candidate"]="C_UNSTABLE_OR_NEUTRAL" if stage=="TRAIN" else ""
        rec["candidate_ci_lower"]=lo_value;rec["candidate_ci_upper"]=hi_value
        summaries.append(rec)
    return summaries,subject_matrices,keylist


def session_summary(subject_rows,task,fold,stage):
    groups={}
    for r in subject_rows:
        key=(r["basis_type"],int(r["basis_draw"]),int(r["direction"]),int(r["session"]))
        groups.setdefault(key,[]).append(r)
    output=[]
    for (bt,bd,d,ses),rows in sorted(groups.items()):
        rec={"task":task,"fold":fold,"stage":stage,"basis_type":bt,"basis_draw":bd,"direction":d,
             "session":ses,"n_subjects":len(rows)}
        for field in ("U_P_mean","U_C_mean","U_total_mean","U_int_mean","delta_p_norm_mean"):
            vals=[float(r[field]) for r in rows];rec[field.replace("_mean","_subject_equal_mean")]=float(np.mean(vals))
        rec["U_P_positive_subject_fraction"]=float(np.mean([float(r["U_P_mean"])>0 for r in rows]))
        output.append(rec)
    return output


def finite_rows(fd,subjects,sessions,basis_names):
    rows=[];subjects=np.asarray(subjects).astype(str);sessions=np.asarray(sessions,dtype=np.int64)
    for sub,ses in sorted(set(zip(subjects.tolist(),sessions.tolist())),key=lambda x:(x[0],x[1])):
        ix=np.flatnonzero((subjects==sub)&(sessions==ses))
        for eps,mat in fd.items():
            for d in range(mat.shape[1]):
                bt,bd=basis_names[d]
                rows.append({"subject":sub,"session":int(ses),"basis_type":bt,"basis_draw":bd,"direction":d,
                             "epsilon":eps,"A_P_mean":float(mat[ix,d].mean()),"n_trials":int(len(ix))})
    return rows


def save_cell(stage,task,fold,subject_rows,summary_rows,fd_rows,basis_record,extra=None,additional_files=None):
    d=cell(task,fold,stage.lower());d.mkdir(parents=True,exist_ok=True)
    B.csv_write(d/f"{stage}_SUBJECT_DIRECTION_UTILITY.csv",subject_rows)
    B.csv_write(d/f"{stage}_DIRECTION_SUMMARY.csv",summary_rows)
    if fd_rows is not None:B.csv_write(d/f"{stage}_FD_SENSITIVITY.csv",fd_rows)
    if basis_record is not None:B.json_write(d/"C_BASIS_AUDIT.json",basis_record)
    for name,rows in (additional_files or {}).items():B.csv_write(d/name,rows)
    files=[p for p in d.iterdir() if p.is_file() and p.name!="COMPLETE.json"]
    complete={"status":"COMPLETE","task":task,"fold":fold,"stage":stage,
              "files":{p.name:sha(p) for p in sorted(files)},"extra":extra or {}}
    B.json_write(d/"COMPLETE.json",complete)
    return complete


def verify_cell(stage,task,fold):
    d=cell(task,fold,stage.lower());p=d/"COMPLETE.json"
    if not p.is_file():raise RuntimeError(f"missing {stage} completion: {task}/{fold}")
    c=json.loads(p.read_text(encoding="utf-8"))
    if c.get("status")!="COMPLETE":raise RuntimeError(f"incomplete {stage} cell: {task}/{fold}")
    for name,digest in c["files"].items():
        if not (d/name).is_file() or sha(d/name)!=digest:raise RuntimeError(f"{stage} artifact hash mismatch: {task}/{fold}/{name}")
    return d,c


def prepare(task,fold):
    plock=require_locked_protocol()
    if sha(Path(__file__).resolve())!=plock["code_sha256"]:raise RuntimeError("source code differs from locked protocol")
    out=cell(task,fold,"train")
    if (out/"COMPLETE.json").exists():
        verify_cell("TRAIN",task,fold);print("PREPARE_ALREADY_COMPLETE",task,fold,flush=True);return
    data,model,anchor,ck=model_and_train_data(task,fold);assert_frozen(model)
    qs_data=load_projectors(task,fold);qs,qd,ms,md=(qs_data[k] for k in ("qs","qd","ms","md"))
    x,y,subjects,sessions=train_arrays(task,data)
    h=spatial(model,x)
    hc,recon=project_complement(h,qs,ms)
    basis,basis_record=make_c_basis(h,qs,ms,task,fold)
    if basis is None:raise RuntimeError(f"undefined TRAIN complement PCA: {task}/{fold}")
    k=basis.shape[1]
    random_bases,random_seeds,random_hashes=random_complement_bases(qs,k,task,fold)
    sigma=np.maximum((hc@basis).std(axis=0,ddof=1),1e-8).astype(np.float32)
    basis_record.update({"task":task,"fold":fold,"K_C":k,"spatial_reconstruction_max_abs":recon,
                         "train_subject_count":len(set(subjects.tolist())),"train_trial_count":len(x),
                         "train_data_hash":B.arr_sha(x,y,subjects.astype("U"),sessions),
                         "random_basis_draws":RANDOM_DRAWS,"random_basis_hashes":random_hashes,"random_basis_seeds":random_seeds,
                         "train_sigma":sigma.tolist(),"checkpoint_sha256":sha(ck),
                         "projector_hash":B.arr_sha(qs,ms,qd,md)})
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"bases.npz",pca=basis,random=random_bases,sigma=sigma,
                        explained_variance_ratio=np.asarray(basis_record["explained_variance_ratio"],np.float32))
    all_bases=np.concatenate((basis[None,:,:],random_bases),axis=0)
    vals=eval_primary(model,h,y,subjects,sessions,qs,ms,qd,md,all_bases,data["samples"],"TRAIN")
    fd=eval_finite_difference(model,h,y,qs,ms,qd,md,basis,sigma,data["samples"])
    names=[("PCA",0)]+[("RANDOM",r) for r in range(RANDOM_DRAWS)]
    basis_names=[x for x in names for _ in range(k)]
    ids=list(range(k))*len(names)
    subj_rows=grouped_trial_rows(vals,subjects,sessions,basis_names,ids,fd)
    fd_rows=finite_rows(fd,subjects,sessions,[ ("PCA",0) for _ in range(k) ])
    fd_fields=tuple(f"A_P_eps_{eps:g}_mean" for eps in EPSILONS)
    summaries,_,_=direction_summary(subj_rows,task,fold,"TRAIN",basis_names,ids,fd_fields)
    session_rows=session_summary(subj_rows,task,fold,"TRAIN")
    for r in summaries:
        r["random_basis_hash"]=random_hashes[int(r["basis_draw"])] if r["basis_type"]=="RANDOM" else basis_record["c_basis_hash"]
    extra={"checkpoint_sha256":sha(ck),"projector_hash":basis_record["projector_hash"],
           "basis_file_sha256":sha(out/"bases.npz"),"basis_record_sha256":B.arr_sha(np.frombuffer(json.dumps(basis_record,sort_keys=True).encode(),dtype=np.uint8)),
           "train_role":"inner_train_subjects","discovery_loaded":False,"outer_dev_loaded":False,
           "final_heldout_loaded":False}
    complete=save_cell("TRAIN",task,fold,subj_rows,summaries,fd_rows,basis_record,extra,
                        {"TRAIN_SESSION_SUMMARY.csv":session_rows})
    print("TRAIN_COMPLETE",task,fold,"K",k,"rows",len(subj_rows),"draws",complete["files"].keys(),flush=True)


def lock_discovery():
    lock=require_locked_protocol()
    if sha(Path(__file__).resolve())!=lock["code_sha256"]:raise RuntimeError("source code differs from locked protocol")
    path=PROTOCOL/"DISCOVERY_DIRECTION_LOCK.json"
    if path.exists():raise RuntimeError("discovery direction lock already exists")
    cells=[]
    for task in TASKS:
        for fold in FOLDS:
            d,c=verify_cell("TRAIN",task,fold)
            basis_path=d/"bases.npz";basis=np.load(basis_path,allow_pickle=False)
            summaries=list(csv.DictReader((d/"TRAIN_DIRECTION_SUMMARY.csv").open(newline="",encoding="utf-8")))
            candidates=[{"basis_type":r["basis_type"],"basis_draw":int(r["basis_draw"]),"direction":int(r["direction"]),
                         "label":r["train_candidate"],"ci_lower":float(r["candidate_ci_lower"]),"ci_upper":float(r["candidate_ci_upper"])} for r in summaries]
            if not candidates:raise RuntimeError(f"no frozen TRAIN candidate labels: {task}/{fold}")
            cells.append({"task":task,"fold":fold,"train_complete_sha256":sha(d/"COMPLETE.json"),
                          "train_complete_files":c["files"],"bases_sha256":sha(basis_path),
                          "pca_hash":B.arr_sha(basis["pca"]),"random_hashes":[B.arr_sha(v) for v in basis["random"]],
                          "candidate_labels":candidates,
                          "candidate_labels_sha256":hashlib.sha256(json.dumps(candidates,sort_keys=True,separators=(",",":")).encode()).hexdigest()})
    record={"schema":"PERSIST_EEG_CP_DIRECTION_DISCOVERY_LOCK_V1","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "protocol_lock_sha256":sha(PROTOCOL/"PROTOCOL_LOCK.json"),"source_code_sha256":sha(Path(__file__).resolve()),
            "FINAL_HELDOUT_ACCESSED":False,"candidate_source":"TRAIN only; frozen before any inner_val EEG read",
            "discovery_role":"inner_val_subjects","excluded_role":"outer_dev_subjects and final heldout",
            "cells":cells}
    B.json_write(path,record);(PROTOCOL/"DISCOVERY_DIRECTION_LOCK.sha256").write_text(sha(path)+"\n",encoding="utf-8")
    print("DISCOVERY_LOCK_FROZEN",len(cells),flush=True)


def read_discovery_lock():
    p=PROTOCOL/"DISCOVERY_DIRECTION_LOCK.json";h=PROTOCOL/"DISCOVERY_DIRECTION_LOCK.sha256"
    if not p.is_file() or not h.is_file() or sha(p)!=h.read_text(encoding="utf-8").strip():raise RuntimeError("discovery lock absent or hash mismatch")
    obj=json.loads(p.read_text(encoding="utf-8"))
    if obj["protocol_lock_sha256"]!=sha(PROTOCOL/"PROTOCOL_LOCK.json") or obj["source_code_sha256"]!=sha(Path(__file__).resolve()):
        raise RuntimeError("locked discovery provenance changed")
    return obj


def discovery(task,fold):
    plock=require_locked_protocol();dlock=read_discovery_lock()
    if sha(Path(__file__).resolve())!=plock["code_sha256"]:raise RuntimeError("source code differs from locked protocol")
    locked=next((x for x in dlock["cells"] if x["task"]==task and int(x["fold"])==fold),None)
    if locked is None:raise RuntimeError(f"cell absent from discovery lock: {task}/{fold}")
    train_dir,tc=verify_cell("TRAIN",task,fold)
    if (cell(task,fold,"discovery")/"COMPLETE.json").is_file():
        verify_cell("DISCOVERY",task,fold);print("DISCOVERY_ALREADY_COMPLETE",task,fold,flush=True);return
    if sha(train_dir/"COMPLETE.json")!=locked["train_complete_sha256"] or sha(train_dir/"bases.npz")!=locked["bases_sha256"]:
        raise RuntimeError("TRAIN candidate source changed after discovery lock")
    for name,digest in locked["train_complete_files"].items():
        if sha(train_dir/name)!=digest:raise RuntimeError("TRAIN locked artifact modified")
    basisfile=np.load(train_dir/"bases.npz",allow_pickle=False)
    basis=basisfile["pca"].astype(np.float32);random_bases=basisfile["random"].astype(np.float32);sigma=basisfile["sigma"].astype(np.float32)
    if B.arr_sha(basis)!=locked["pca_hash"] or [B.arr_sha(v) for v in random_bases]!=locked["random_hashes"]:
        raise RuntimeError("locked basis hash mismatch")
    # Candidate labels and all direction bases are now hash-locked. Only this point may open inner_val EEG.
    data,model,anchor,ck=model_and_train_data(task,fold);assert_frozen(model)
    x,y,subjects,sessions=discovery_arrays(task,data)
    h=spatial(model,x);qs_data=load_projectors(task,fold);qs,qd,ms,md=(qs_data[k] for k in ("qs","qd","ms","md"))
    all_bases=np.concatenate((basis[None,:,:],random_bases),axis=0)
    vals=eval_primary(model,h,y,subjects,sessions,qs,ms,qd,md,all_bases,data["samples"],"DISCOVERY")
    fd=eval_finite_difference(model,h,y,qs,ms,qd,md,basis,sigma,data["samples"])
    k=basis.shape[1];names=[("PCA",0)]+[("RANDOM",r) for r in range(RANDOM_DRAWS)]
    basis_names=[x for x in names for _ in range(k)];ids=list(range(k))*len(names)
    subj_rows=grouped_trial_rows(vals,subjects,sessions,basis_names,ids,fd)
    fd_rows=finite_rows(fd,subjects,sessions,[("PCA",0) for _ in range(k)])
    fd_fields=tuple(f"A_P_eps_{eps:g}_mean" for eps in EPSILONS)
    summaries,_,_=direction_summary(subj_rows,task,fold,"DISCOVERY",basis_names,ids,fd_fields)
    candidate_map={(c["basis_type"],int(c["basis_draw"]),int(c["direction"])):c["label"] for c in locked["candidate_labels"]}
    validations=[];routing=[]
    for r in summaries:
        key=(r["basis_type"],int(r["basis_draw"]),int(r["direction"]))
        candidate=candidate_map[key];mean=float(r["U_P_subject_equal_mean"]);lo=float(r["U_P_CI95_lower"]);hi=float(r["U_P_CI95_upper"])
        if candidate=="C_PLUS_CANDIDATE" and mean>0 and lo>0:status="VALIDATED_C_PLUS"
        elif candidate=="C_MINUS_CANDIDATE" and mean<0 and hi<0:status="VALIDATED_C_MINUS"
        else:status="NOT_STABLY_SIGNED"
        r["train_candidate"]=candidate;r["validated_status"]=status
        route="NOT_VALIDATED"
        if status in ("VALIDATED_C_PLUS","VALIDATED_C_MINUS"):
            up=float(r["U_P_subject_equal_mean"]);uc=float(r["U_C_subject_equal_mean"]);ut=float(r["U_total_subject_equal_mean"])
            if up>0 and uc>=0:route="ROUTE_TO_P_AND_KEEP_C"
            elif up>0 and uc<0:route="ROUTE_TO_P"
            elif up<0 and uc>0:route="BLOCK_FROM_P_KEEP_C"
            elif up<0 and uc<0 and ut<0:route="HARMFUL_OVERALL"
            else:route="MIXED_OR_UNRESOLVED"
            route_row={"task":task,"fold":fold,"basis_type":r["basis_type"],"basis_draw":r["basis_draw"],
                       "direction":r["direction"],"train_candidate":candidate,"validated_status":status,"route_type":route,
                       "U_P":up,"U_C":uc,"U_total":ut,"U_int":float(r["U_int_subject_equal_mean"])}
            routing.append(route_row)
        r["route_type"]=route
        validations.append({"task":task,"fold":fold,"basis_type":r["basis_type"],"basis_draw":r["basis_draw"],
                            "direction":r["direction"],"train_candidate":candidate,"train_ci_lower":next(c["ci_lower"] for c in locked["candidate_labels"] if (c["basis_type"],int(c["basis_draw"]),int(c["direction"]))==key),
                            "train_ci_upper":next(c["ci_upper"] for c in locked["candidate_labels"] if (c["basis_type"],int(c["basis_draw"]),int(c["direction"]))==key),
                            "discovery_mean_U_P":mean,"discovery_ci_lower":lo,"discovery_ci_upper":hi,"validated_status":status,"route_type":route})
    basis_record=json.loads((train_dir/"C_BASIS_AUDIT.json").read_text(encoding="utf-8"))
    session_rows=session_summary(subj_rows,task,fold,"DISCOVERY")
    extra={"train_complete_sha256":sha(train_dir/"COMPLETE.json"),"discovery_lock_sha256":sha(PROTOCOL/"DISCOVERY_DIRECTION_LOCK.json"),
           "basis_file_sha256":sha(train_dir/"bases.npz"),"inner_val_subject_count":len(set(subjects.tolist())),
           "inner_val_trial_count":len(x),"outer_dev_loaded":False,"final_heldout_loaded":False}
    save_cell("DISCOVERY",task,fold,subj_rows,summaries,fd_rows,basis_record,extra,
              {"DISCOVERY_DIRECTION_VALIDATION.csv":validations,"VALIDATED_C_ROUTING.csv":routing,
               "DISCOVERY_SESSION_SUMMARY.csv":session_rows})
    print("DISCOVERY_COMPLETE",task,fold,"subjects",len(set(subjects.tolist())),"validated",sum(r["validated_status"]!="NOT_STABLY_SIGNED" for r in validations),flush=True)


def fnum(row,key,default=0.0):
    try:
        v=float(row.get(key,""));return v if np.isfinite(v) else default
    except (TypeError,ValueError):return default


def aggregate():
    require_locked_protocol();dlock=read_discovery_lock()
    basis_audit=[];subject_all=[];train_summaries=[];disc_summaries=[];validation_all=[];routing_all=[];fd_all=[];session_all=[]
    for task in TASKS:
        for fold in FOLDS:
            tr,tc=verify_cell("TRAIN",task,fold);di,dc=verify_cell("DISCOVERY",task,fold)
            basis_audit.append(json.loads((tr/"C_BASIS_AUDIT.json").read_text(encoding="utf-8")))
            for stage,dr in (("TRAIN",tr),("DISCOVERY",di)):
                for r in csv.DictReader((dr/f"{stage}_SUBJECT_DIRECTION_UTILITY.csv").open(newline="",encoding="utf-8")):
                    r["task"]=task;r["fold"]=fold;r["stage"]=stage;subject_all.append(r)
                for r in csv.DictReader((dr/f"{stage}_DIRECTION_SUMMARY.csv").open(newline="",encoding="utf-8")):
                    r["task"]=task;r["fold"]=fold; (train_summaries if stage=="TRAIN" else disc_summaries).append(r)
                for r in csv.DictReader((dr/f"{stage}_FD_SENSITIVITY.csv").open(newline="",encoding="utf-8")):
                    r["task"]=task;r["fold"]=fold;r["stage"]=stage;fd_all.append(r)
                for r in csv.DictReader((dr/f"{stage}_SESSION_SUMMARY.csv").open(newline="",encoding="utf-8")):
                    session_all.append(r)
            for r in csv.DictReader((di/"DISCOVERY_DIRECTION_VALIDATION.csv").open(newline="",encoding="utf-8")):
                validation_all.append(r)
            for r in csv.DictReader((di/"VALIDATED_C_ROUTING.csv").open(newline="",encoding="utf-8")):
                routing_all.append(r)
    basis_by={(r["task"],int(r["fold"])):r for r in basis_audit}
    trmap={(r["task"],int(r["fold"]),r["basis_type"],int(r["basis_draw"]),int(r["direction"])):r for r in train_summaries}
    dimap={(r["task"],int(r["fold"]),r["basis_type"],int(r["basis_draw"]),int(r["direction"])):r for r in disc_summaries}
    valmap={(r["task"],int(r["fold"]),r["basis_type"],int(r["basis_draw"]),int(r["direction"])):r for r in validation_all}
    merged=[]
    for k,t in trmap.items():
        d=dimap[k];v=valmap[k]
        row={"task":k[0],"fold":k[1],"basis_type":k[2],"basis_draw":k[3],"direction":k[4],
             "train_candidate":t["train_candidate"],"train_U_P":fnum(t,"U_P_subject_equal_mean"),
             "train_ci_lower":fnum(t,"U_P_CI95_lower"),"train_ci_upper":fnum(t,"U_P_CI95_upper"),
             "discovery_U_P":fnum(d,"U_P_subject_equal_mean"),"discovery_ci_lower":fnum(d,"U_P_CI95_lower"),
             "discovery_ci_upper":fnum(d,"U_P_CI95_upper"),"validated_status":v["validated_status"],"route_type":v["route_type"]}
        for metric in ("U_C_subject_equal_mean","U_total_subject_equal_mean","U_int_subject_equal_mean","delta_p_norm_subject_equal_mean"):
            row["train_"+metric]=fnum(t,metric);row["discovery_"+metric]=fnum(d,metric)
        merged.append(row)
    # Paired empirical control rows. PCA and each randomized basis are reported at fold level.
    random_control=[];fold_rows=[]
    for task in TASKS:
        for fold in FOLDS:
            allkeys=[r for r in merged if r["task"]==task and r["fold"]==fold]
            b=basis_by[task,fold];k=int(b["K_C"]);ev=np.asarray(b.get("explained_variance_ratio",[]),dtype=float)
            bybasis={}
            for typ,draw in [("PCA",0)]+[("RANDOM",r) for r in range(RANDOM_DRAWS)]:
                rows=[r for r in allkeys if r["basis_type"]==typ and int(r["basis_draw"])==draw]
                ups=np.asarray([r["discovery_U_P"] for r in rows],dtype=float)
                trainc=[r for r in rows if r["train_candidate"]=="C_PLUS_CANDIDATE"]
                trainm=[r for r in rows if r["train_candidate"]=="C_MINUS_CANDIDATE"]
                vp=[r for r in rows if r["validated_status"]=="VALIDATED_C_PLUS"]
                vm=[r for r in rows if r["validated_status"]=="VALIDATED_C_MINUS"]
                route_rows=[r for r in routing_all if r["task"]==task and int(r["fold"])==fold and r["basis_type"]==typ and int(r["basis_draw"])==draw]
                metric={"train_C_plus_candidates":len(trainc),"train_C_minus_candidates":len(trainm),
                        "validated_C_plus":len(vp),"validated_C_minus":len(vm),
                        "M_plus":float(np.maximum(ups,0).sum()),"M_minus":float(np.maximum(-ups,0).sum()),
                        "strongest_positive":float(ups.max()) if len(ups) else 0.,"strongest_negative":float(ups.min()) if len(ups) else 0.,
                        "block_from_P_keep_C":sum(r["route_type"]=="BLOCK_FROM_P_KEEP_C" for r in route_rows),
                        "C_plus_variance_fraction":float(sum(ev[int(r["direction"])] for r in vp if int(r["direction"])<len(ev))) if typ=="PCA" else None,
                        "C_minus_variance_fraction":float(sum(ev[int(r["direction"])] for r in vm if int(r["direction"])<len(ev))) if typ=="PCA" else None}
                bybasis[typ,draw]=metric
            rand=[bybasis["RANDOM",r] for r in range(RANDOM_DRAWS)];pca=bybasis["PCA",0]
            for metric_name in ("train_C_plus_candidates","train_C_minus_candidates","validated_C_plus","validated_C_minus","M_plus","M_minus","strongest_positive","strongest_negative","block_from_P_keep_C"):
                vals=np.asarray([r[metric_name] for r in rand],dtype=float);actual=float(pca[metric_name])
                pca[f"{metric_name}_random_mean"]=float(vals.mean());pca[f"{metric_name}_random_p95"]=float(np.quantile(vals,.95,method="higher"))
                pca[f"{metric_name}_random_percentile"]=float(np.mean(vals<=actual))
            for typ,draw in [("PCA",0)]+[("RANDOM",r) for r in range(RANDOM_DRAWS)]:
                random_control.append({"task":task,"fold":fold,"basis_type":typ,"basis_draw":draw,**bybasis[typ,draw]})
            pplus=sum(r["route_type"]=="ROUTE_TO_P_AND_KEEP_C" for r in routing_all if r["task"]==task and int(r["fold"])==fold and r["basis_type"]=="PCA")
            ponly=sum(r["route_type"]=="ROUTE_TO_P" for r in routing_all if r["task"]==task and int(r["fold"])==fold and r["basis_type"]=="PCA")
            block=pca["block_from_P_keep_C"]
            harmful=sum(r["route_type"]=="HARMFUL_OVERALL" for r in routing_all if r["task"]==task and int(r["fold"])==fold and r["basis_type"]=="PCA")
            pca_rows=[r for r in allkeys if r["basis_type"]=="PCA"]
            vp_utility=sum(r["discovery_U_P"] for r in pca_rows if r["validated_status"]=="VALIDATED_C_PLUS")
            vm_utility=sum(r["discovery_U_P"] for r in pca_rows if r["validated_status"]=="VALIDATED_C_MINUS")
            fold_rows.append({"scope":"TASK_FOLD","task":task,"fold":fold,"K_C":k,
                              "C_plus_candidates":pca["train_C_plus_candidates"],"validated_C_plus":pca["validated_C_plus"],
                              "C_minus_candidates":pca["train_C_minus_candidates"],"validated_C_minus":pca["validated_C_minus"],
                              "positive_utility_mass":pca["M_plus"],"negative_utility_mass":pca["M_minus"],
                              "validated_C_plus_P_utility":vp_utility,"validated_C_minus_P_utility":vm_utility,
                              "C_plus_variance_fraction":pca["C_plus_variance_fraction"],"C_minus_variance_fraction":pca["C_minus_variance_fraction"],
                              "route_to_P_and_keep_C":pplus,"route_to_P":ponly,"block_from_P_keep_C":block,"harmful_overall":harmful,
                              "PCA_M_plus_random_p95":pca["M_plus_random_p95"],"PCA_M_plus_random_percentile":pca["M_plus_random_percentile"],
                              "PCA_validated_C_plus_random_p95":pca["validated_C_plus_random_p95"],
                              "PCA_block_random_p95":pca["block_from_P_keep_C_random_p95"]})
    task_rows=[]
    for task in TASKS:
        cells=[r for r in fold_rows if r["task"]==task]
        task_rows.append({"scope":"TASK","task":task,"fold":"ALL","K_C":float(np.mean([r["K_C"] for r in cells])),
                          **{field:int(sum(r[field] for r in cells)) for field in ("C_plus_candidates","validated_C_plus","C_minus_candidates","validated_C_minus","route_to_P_and_keep_C","route_to_P","block_from_P_keep_C","harmful_overall")},
                          "positive_utility_mass":float(sum(r["positive_utility_mass"] for r in cells)),
                          "negative_utility_mass":float(sum(r["negative_utility_mass"] for r in cells)),
                          "validated_C_plus_P_utility":float(sum(r["validated_C_plus_P_utility"] for r in cells)),
                          "validated_C_minus_P_utility":float(sum(r["validated_C_minus_P_utility"] for r in cells)),
                          "C_plus_variance_fraction":float(np.mean([r["C_plus_variance_fraction"] for r in cells])),
                          "C_minus_variance_fraction":float(np.mean([r["C_minus_variance_fraction"] for r in cells])),
                          "PCA_M_plus_random_p95_exceed_folds":sum(r["positive_utility_mass"]>r["PCA_M_plus_random_p95"] for r in cells),
                          "PCA_validated_C_plus_random_p95_exceed_folds":sum(r["validated_C_plus"]>r["PCA_validated_C_plus_random_p95"] for r in cells)})
    B.csv_write(OUT/"C_BASIS_AUDIT.csv",basis_audit);B.csv_write(OUT/"SUBJECT_DIRECTION_UTILITY.csv",subject_all)
    B.csv_write(OUT/"DIRECTION_UTILITY_SUMMARY.csv",merged);B.csv_write(OUT/"DISCOVERY_DIRECTION_VALIDATION.csv",validation_all)
    B.csv_write(OUT/"VALIDATED_C_ROUTING.csv",routing_all);B.csv_write(OUT/"RANDOM_DIRECTION_CONTROL.csv",random_control)
    B.csv_write(OUT/"FINITE_DIFFERENCE_SENSITIVITY.csv",fd_all);B.csv_write(OUT/"TASK_FOLD_SUMMARY.csv",fold_rows+task_rows)
    B.csv_write(OUT/"SESSION_DIRECTION_SUMMARY.csv",session_all)
    exclusion=heldout_exclusion_audit();exclusion.update({"protocol_lock_sha256":sha(PROTOCOL/"PROTOCOL_LOCK.json"),
                                                         "discovery_lock_sha256":sha(PROTOCOL/"DISCOVERY_DIRECTION_LOCK.json"),
                                                         "train_cells_complete":20,"discovery_cells_complete":20,
                                                         "outer_dev_subjects_loaded":0,"final_heldout_subjects_loaded":0,
                                                         "final_heldout_array_reads":0})
    B.json_write(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json",exclusion)
    write_report(fold_rows,task_rows,routing_all,random_control)
    print("AGGREGATE_COMPLETE",len(subject_all),len(merged),"FINAL_HELDOUT_ACCESSED=FALSE",flush=True)


def write_report(folds,tasks,routes,random_rows):
    pca_control={(r["task"],int(r["fold"])):r for r in random_rows if r["basis_type"]=="PCA"}
    train_candidates=sum(r["C_plus_candidates"]+r["C_minus_candidates"] for r in folds)
    validated=sum(r["validated_C_plus"]+r["validated_C_minus"] for r in folds)
    cp_specific=sum(r["positive_utility_mass"]>r["PCA_M_plus_random_p95"] or r["validated_C_plus"]>r["PCA_validated_C_plus_random_p95"] for r in folds)
    block_specific=sum(r["block_from_P_keep_C"]>r["PCA_block_random_p95"] for r in folds)
    blocks=sum(r["route_type"]=="BLOCK_FROM_P_KEEP_C" for r in routes if r["basis_type"]=="PCA")
    if validated==0:states=["NO_DIRECTIONAL_CP_ACTIONABILITY"]
    elif cp_specific:states=["SELECTIVE_CP_ROUTING_SUPPORTED"]
    else:states=["NO_BASIS_SPECIFIC_DIRECTION_STRUCTURE"]
    if blocks and block_specific:states.append("SELECTIVE_BLOCK_FROM_P_SUPPORTED")
    if train_candidates and validated/max(train_candidates,1)<0.5:states.append("DIRECTION_UTILITY_TASK_OR_SUBJECT_DEPENDENT")
    lines=["# Frozen C-direction utility audit", "", "**Scope:** four EEGNet seed-zero tasks, folds 0–4. No backbone, adapter, or gate was trained. TRAIN candidates were frozen before discovery EEG was loaded. `outer_dev_subjects` and every formal final-heldout subject/array were excluded.", "", "**FINAL_HELDOUT_ACCESSED = FALSE**", "", "All direction indices are fold-local. Utilities are functional interventions on a frozen network and do not establish biological causality. `U_int = U_total - U_P - U_C` is reported only as a descriptive nonlinear residual. PCA finite differences are secondary sensitivity results; the primary conclusions use native-coordinate erasure.", "", "## Task and fold summary", "", "| Task | Fold | K(C) | C+ candidates | Validated C+ | C- candidates | Validated C- | Positive utility mass | Negative utility mass |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in folds:lines.append(f"| {r['task']} | {r['fold']} | {r['K_C']} | {r['C_plus_candidates']} | {r['validated_C_plus']} | {r['C_minus_candidates']} | {r['validated_C_minus']} | {r['positive_utility_mass']:.6g} | {r['negative_utility_mass']:.6g} |")
    lines += ["", "## Task totals", "", "| Task | Validated C+ count | Validated C- count | C+ P-mediated utility | C- P-mediated utility | Random specificity |", "|---|---:|---:|---:|---:|---|"]
    for t in tasks:
        q=[r for r in folds if r["task"]==t["task"]]
        lines.append(f"| {t['task']} | {t['validated_C_plus']} | {t['validated_C_minus']} | {t['validated_C_plus_P_utility']:.6g} | {t['validated_C_minus_P_utility']:.6g} | {t['PCA_M_plus_random_p95_exceed_folds']}/5 folds exceed random 95th percentile on M+; {t['PCA_validated_C_plus_random_p95_exceed_folds']}/5 on validated C+ count |")
    lines += ["", "## Validated routing breakdown", "", "| Route type | Count | Mean U_P | Mean U_C | Mean U_total |", "|---|---:|---:|---:|---:|"]
    for typ in sorted(set(r["route_type"] for r in routes if r["basis_type"]=="PCA")):
        rr=[r for r in routes if r["basis_type"]=="PCA" and r["route_type"]==typ]
        lines.append(f"| {typ} | {len(rr)} | {np.mean([fnum(r,'U_P') for r in rr]):.6g} | {np.mean([fnum(r,'U_C') for r in rr]):.6g} | {np.mean([fnum(r,'U_total') for r in rr]):.6g} |")
    lines += ["", "## Session summaries", "", "`SESSION_DIRECTION_SUMMARY.csv` reports subject-equal direction means separately for each physical session. Candidate bootstraps first average a subject's session means, then resample biological subjects. OpenBMI reports S1/S2; WBCIC reports S0/S1/S2.", "", "## Interpretation", "", "Observed state: **"+", ".join(states)+"**.", "", f"TRAIN identified {train_candidates} PCA C+/C- candidates; discovery validated {validated}. {blocks} PCA directions were validated as `U_P < 0, U_C > 0`. Random specificity compares each fold's PCA basis with 20 deterministic equal-rank bases in the same spatial complement; the 95th percentile uses the empirical higher quantile across those 20 draws.", "", "A state that says a pattern was supported means the prespecified bootstrap sign repeated in discovery and the PCA summary exceeded the corresponding random-control 95th percentile in at least one fold. This is development evidence for later architecture design, not final generalization evidence.", "", "## Audit artifacts", "", "The per-cell completion manifests hash every analysis file. `PROTOCOL_LOCK.json` fixes roles, checkpoint/projector provenance, estimands, seeds, and thresholds. `DISCOVERY_DIRECTION_LOCK.json` records every TRAIN candidate and basis hash before discovery evaluation. `FINAL_HELDOUT_EXCLUSION_AUDIT.json` records zero heldout EEG reads and zero outer-dev loads.", ""]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=("preflight","prepare","lock-discovery","discovery","aggregate"));ap.add_argument("--task",choices=TASKS);ap.add_argument("--fold",type=int)
    args=ap.parse_args()
    if args.stage=="preflight":preflight();return
    if args.stage=="lock-discovery":lock_discovery();return
    if args.stage=="aggregate":aggregate();return
    if args.task is None or args.fold not in FOLDS:ap.error("prepare/discovery require --task and --fold 0..4")
    if args.stage=="prepare":prepare(args.task,args.fold)
    else:discovery(args.task,args.fold)


if __name__=="__main__":main()

