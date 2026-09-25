"""Train-only local C-to-P targets for the native SIRE shared block."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
SOURCE = REPO / "experiments/persist_eeg_sire_cp_actionability_observability_seed0_v1"
SOURCE_LOCK = SOURCE / "protocol/PROTOCOL_LOCK.json"
SOURCE_RECORD = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
CANONICAL_REPO = Path(SOURCE_RECORD["source"]["carrier_source"]).parents[3]
os.environ["SIRE_CANONICAL_SOURCE_REPO"] = str(CANONICAL_REPO)
spec = importlib.util.spec_from_file_location("sire_local_p_frozen_source", SOURCE / "code/run.py")
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)
B = A.B
DEVICE = A.DEVICE
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("SIRE_LOCAL_P_RUNTIME", str(REPO.parent / "sire_local_p_construction_seed0_runtime"))).resolve()
TASK = "OpenBMI_MI"
FOLDS = tuple(range(5))
ARMS = ("CE_ONLY_CONTINUATION", "LOCAL_CONSTRAINED_P")
ALPHAS = np.asarray((0.5, 0.75, 1.25, 1.5), dtype=np.float32)
EPOCHS = 20
BATCH = 64
EPS = 1e-6
BOOTSTRAPS = 20_000
torch.set_num_threads(min(int(os.environ.get("SIRE_LOCAL_P_CPU_THREADS", "12")), os.cpu_count() or 1))
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()


def arr_sha(*xs):
    return A.arr_sha(*xs)


def seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:4],"big")


def jwrite(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".part")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def cwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    tmp=path.with_suffix(path.suffix+".part")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    os.replace(tmp,path)


def cread(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def cell_source(fold):
    return next(c for c in SOURCE_RECORD["cells"] if c["task"]==TASK and int(c["fold"])==fold)


def rcell(fold):
    return RUNTIME/"cells"/TASK.lower()/f"fold{fold}_seed0"


def preflight():
    A.verify_lock()
    cells=[]
    for fold in FOLDS:
        c=cell_source(fold)
        role,split,cache,source_sessions,future=B.role(TASK,fold)
        train=A.ordered_subjects(role["inner_train_subjects"])
        discovery=A.ordered_subjects(role["inner_val_subjects"])
        outer=A.ordered_subjects(role["outer_dev_subjects"])
        if (set(train)&set(discovery)) or (set(train)&set(outer)) or (set(discovery)&set(outer)):
            raise RuntimeError("role overlap")
        if train!=c["inner_train_subjects"] or discovery!=c["discovery_subjects"] or split!=c["split_sha256"]:
            raise RuntimeError(f"source role drift fold{fold}")
        ck=Path(c["frozen_checkpoint_path"]);normalizer=Path(c["normalizer_path"])
        if sha(ck)!=c["frozen_checkpoint_sha256"] or sha(normalizer)!=c["normalizer_sha256"]:
            raise RuntimeError(f"canonical checkpoint/normalizer changed fold{fold}")
        cells.append({"fold":fold,"task":TASK,"checkpoint_path":str(ck),"checkpoint_sha256":sha(ck),
            "normalizer_path":str(normalizer),"normalizer_sha256":sha(normalizer),"split_sha256":split,
            "protected_coordinates":c["protected_coordinates_from_frozen_sire_persist_record"],
            "train_subjects":train,"discovery_subjects":discovery,"outer_dev_subjects_excluded":outer,
            "source_sessions":list(map(int,source_sessions)),"future_session":int(future),"cache_name":cache,
            "historical_final_heldout_diagnostic_exposure":bool(c["historically_exposed_diagnostic"])})
    lock={"schema":"SIRE_LOCAL_P_CONSTRUCTION_SEED0_V1","task":TASK,"seed":0,"folds":list(FOLDS),
          "source_actionability_lock_sha256":sha(SOURCE_LOCK),
          "source_actionability_code_sha256":sha(SOURCE/"code/run.py"),
          "source_carrier_code_sha256":sha(A.CARRIER_SOURCE),
          "source_task_wrapper_sha256":sha(A.TASK_SOURCE),
          "source_projector_audit_sha256":sha(SOURCE/"outputs/SIRE_PROJECTOR_AUDIT.csv"),
          "source_c_basis_audit_sha256":sha(SOURCE/"outputs/SIRE_C_BASIS_AUDIT.csv"),
          "cells":cells,"source_stage":"H_concat","successor_stage":"H_shared1",
          "geometry":"frozen SIRE PERSIST/PEEH spectrum, PathFit projectors and train-only complement basis; exact source functions",
          "local_alphas":ALPHAS.tolist(),"one_direction_only":True,
          "teacher_acceptance":{"CE_improvement_strictly_gt":1e-4,"preserve_native_correct":True,
              "selection":"minimum standardized successor-P movement; flat direction-major alpha-order tie"},
          "teacher_weight":"clip((CE_native-CE_teacher)/(CE_native+1e-6),0,1)",
          "scale_floor":EPS,"training_arms":list(ARMS),"trainable_parameters":["depth1.weight","point1.weight"],
          "trainable_modules_eval_mode":True,"optimizer":"AdamW","lr":1e-4,"weight_decay":5e-4,
          "gradient_clip":5.0,"epochs":EPOCHS,"batch_size":BATCH,"lambda_P":1.0,"lambda_C":1.0,
          "checkpoint_endpoint":"epoch20_no_discovery_selection","bootstrap_subject_draws":BOOTSTRAPS,
          "primary_aggregate":"biological-subject-equal mean of both discovery sessions; duplicate fold appearances averaged within subject",
          "primary_comparison":"LOCAL_CONSTRAINED_P minus CE_ONLY_CONTINUATION",
          "forbidden_array_roles":["outer_dev","final_heldout"],
          "historical_provenance_caveat":SOURCE_RECORD["historical_provenance_caveat"]}
    path=PROTOCOL/"PROTOCOL_LOCK.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=lock:
        raise RuntimeError("protocol lock changed")
    jwrite(path,lock)
    jwrite(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json",{
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,"FINAL_HELDOUT_ACCESSED":False,
        "teacher_construction":"inner_train only","scale_estimation":"inner_train only",
        "student_training":"inner_train only","discovery_training_feedback":False,
        "diagnostic_discovery_teacher_after_checkpoints_frozen":True,
        "historical_checkpoint_final_heldout_diagnostic_exposure":True,
        "historical_provenance_caveat":lock["historical_provenance_caveat"]})
    return lock


def manifest(train_rows):
    labels=[];subject=[];session=[];trial=[]
    for r in train_rows:
        n=len(r["raw_y"])
        labels.append(np.asarray(r["raw_y"],np.int64))
        subject.extend([str(r["subject"])]*n)
        session.extend([int(r["session"])]*n)
        trial.extend(range(n))
    labels=np.concatenate(labels)
    subject=np.asarray(subject,dtype="U16")
    session=np.asarray(session,dtype=np.int16)
    trial=np.asarray(trial,dtype=np.int32)
    h=arr_sha(labels,subject,session,trial)
    return {"y":labels,"subject":subject,"session":session,"trial":trial,"sha256":h}


def train_data(fold):
    lock=preflight();c=lock["cells"][fold]
    ids=c["train_subjects"]
    sessions=sorted(set(c["source_sessions"]+[c["future_session"]]))
    pieces,mapping=A.fetch_role(TASK,ids,sessions,c["cache_name"],None)
    model=A.build_model(TASK,len(mapping),Path(c["checkpoint_path"]))
    mean,std=A.load_normalizer(Path(c["normalizer_path"]))
    rows=A.apply_stages(model,pieces,mean,std)
    del pieces
    for row in rows:row.update({"task":TASK,"fold":fold})
    m=manifest(rows)
    return model,rows,m


def decomposition(yd,qd,md):
    x=np.asarray(yd,np.float64)-np.asarray(md,np.float64)
    q=np.asarray(qd,np.float64)
    p=x@q
    c=x-p@q.T
    error=float(np.max(np.abs((np.asarray(md,np.float64)+p@q.T+c)-yd)))
    if error>=1e-6:raise RuntimeError(f"P+C reconstruction failed: {error}")
    return p.astype(np.float32),c.astype(np.float32),error


def geometry_for(fold,rows,m):
    d=rcell(fold);d.mkdir(parents=True,exist_ok=True)
    gpath=d/"FROZEN_GEOMETRY.npz";meta_path=d/"FROZEN_GEOMETRY.json"
    if gpath.exists()!=meta_path.exists():raise RuntimeError("incomplete geometry cache")
    if gpath.exists():
        meta=json.loads(meta_path.read_text())
        if meta["protocol_sha256"]!=sha(PROTOCOL/"PROTOCOL_LOCK.json") or meta["train_manifest_sha256"]!=m["sha256"] or meta["geometry_file_sha256"]!=sha(gpath):
            raise RuntimeError("frozen geometry cache mismatch")
        with np.load(gpath,allow_pickle=False) as z:g={key:z[key] for key in z.files}
    else:
        dims=cell_source(fold)["protected_coordinates_from_frozen_sire_persist_record"]
        qs,ms,qd,md,spectrum,proj=A.train_centroid_rows(rows,dims)
        hs=np.concatenate([r["hs"] for r in rows]);basis,binfo=A.complement_basis(hs,qs,ms)
        yd=np.concatenate([r["yd"] for r in rows]);p0,c0,recon=decomposition(yd,qd,md)
        sigmaP=np.maximum(p0.astype(np.float64).std(0),EPS).astype(np.float32)
        sigmaC=np.maximum(c0.astype(np.float64).std(0),EPS).astype(np.float32)
        g={"qs":qs,"ms":ms,"qd":qd,"md":md,"basis":basis,"sigmaP":sigmaP,"sigmaC":sigmaC}
        prior_proj=[r for r in cread(SOURCE/"outputs/SIRE_PROJECTOR_AUDIT.csv") if r["task"]==TASK and int(r["fold"])==fold]
        prior_basis=next(r for r in cread(SOURCE/"outputs/SIRE_C_BASIS_AUDIT.csv") if r["task"]==TASK and int(r["fold"])==fold)
        hashes={"source":arr_sha(qs,ms),"successor":arr_sha(qd,md),"basis":arr_sha(basis)}
        expected={"source":next(r["projector_sha256"] for r in prior_proj if r["stage"]=="H_concat"),
                  "successor":next(r["projector_sha256"] for r in prior_proj if r["stage"]=="H_shared1"),
                  "basis":prior_basis["basis_sha256"]}
        same={key:hashes[key]==expected[key] for key in hashes}
        # A different basis orientation changes the candidate set. Fail rather than silently redefine it.
        if not all(same.values()):raise RuntimeError(f"historical geometry hash mismatch fold{fold}: {same} {hashes} {expected}")
        tmp=d/"FROZEN_GEOMETRY.part.npz"
        np.savez_compressed(tmp,**g);os.replace(tmp,gpath)
        meta={"task":TASK,"fold":fold,"protocol_sha256":sha(PROTOCOL/"PROTOCOL_LOCK.json"),
              "train_manifest_sha256":m["sha256"],"geometry_file_sha256":sha(gpath),
              "source_geometry_hashes":hashes,"historical_geometry_hashes":expected,
              "historical_hashes_exact":True,"native_reconstruction_max_abs":recon,
              "basis_rank":int(basis.shape[1]),"P_rank":int(qd.shape[1]),
              "basis_orthogonality_max_abs":float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1])))),
              "source_projector_orthogonality_max_abs":float(np.max(np.abs(qs.T@qs-np.eye(qs.shape[1])))),
              "successor_projector_orthogonality_max_abs":float(np.max(np.abs(qd.T@qd-np.eye(qd.shape[1])))),
              "train_trial_count":len(m["y"]),"scale_fit_role":"inner_train_only"}
        jwrite(meta_path,meta)
    return g


def teacher_search(model,hs,yd,z0,y,g,batch=8):
    """Exact old source intervention/successor replacement, local alpha subset."""
    n=len(y);k=g["basis"].shape[1];na=len(ALPHAS)
    qs=torch.as_tensor(g["qs"],dtype=torch.float32,device=DEVICE)
    ms=torch.as_tensor(g["ms"],dtype=torch.float32,device=DEVICE)
    basis=torch.as_tensor(g["basis"],dtype=torch.float32,device=DEVICE)
    qd=torch.as_tensor(g["qd"],dtype=torch.float64,device=DEVICE)
    md=torch.as_tensor(g["md"],dtype=torch.float64,device=DEVICE)
    sigma=torch.as_tensor(g["sigmaP"],dtype=torch.float64,device=DEVICE)
    alphas=torch.as_tensor(ALPHAS,dtype=torch.float32,device=DEVICE)
    native_p=[];target_p=[];accepted=[];weights=[];dirs=[];chosen_alpha=[]
    ce0s=[];ces=[];improvements=[];moves=[];native_correct=[];teacher_correct=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,n,batch):
            if start%520==0:print("TEACHER_PROGRESS",start,n,flush=True)
            stop=min(n,start+batch);b=stop-start
            h=torch.as_tensor(np.ascontiguousarray(hs[start:stop]),device=DEVICE)
            d0=torch.as_tensor(np.ascontiguousarray(yd[start:stop]),device=DEVICE)
            z=torch.as_tensor(np.ascontiguousarray(z0[start:stop]),device=DEVICE)
            yy=torch.as_tensor(y[start:stop],dtype=torch.long,device=DEVICE)
            ce0=F.cross_entropy(z,yy,reduction="none")
            correct0=z.argmax(1)==yy
            centered=h-ms;coeff=(centered-(centered@qs)@qs.T)@basis
            scaled=h[:,None,None,:]+(alphas[None,None,:,None]-1)*coeff[:,:,None,None]*basis.T[None,:,None,:]
            flat=scaled.reshape(-1,h.shape[1])
            shaped=flat.reshape(-1,48,1,h.shape[1]//48)
            d=F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(shaped)))),(1,2)).flatten(1)
            p=(d.double()-md)@qd
            p0=(d0.double()-md)@qd
            c0=(d0.double()-md)-p0@qd.T
            pview=p.reshape(b,k,na,-1)
            routed=(md+pview@qd.T+c0[:,None,None,:]).float()
            logits=A.suffix(model,routed.reshape(-1,d.shape[1])).reshape(b,k,na,-1)
            rep=yy[:,None,None].expand(-1,k,na).reshape(-1)
            ce=F.cross_entropy(logits.reshape(-1,z.shape[1]),rep,reduction="none").reshape(b,k,na)
            pred=logits.argmax(-1)
            distance=torch.sqrt(torch.mean(((pview-p0[:,None,None,:])/(sigma+EPS))**2,dim=-1))
            valid=(ce<ce0[:,None,None]-1e-4)&((~correct0[:,None,None])|(pred==yy[:,None,None]))
            flatdist=distance.masked_fill(~valid,torch.inf).reshape(b,-1)
            ix=flatdist.argmin(1)
            found=valid.reshape(b,-1).any(1)
            rowix=torch.arange(b,device=DEVICE)
            pf=pview.reshape(b,k*na,-1)[rowix,ix]
            cf=ce.reshape(b,k*na)[rowix,ix]
            predsel=pred.reshape(b,k*na)[rowix,ix]
            movement=distance.reshape(b,k*na)[rowix,ix]
            pf=torch.where(found[:,None],pf,p0)
            cf=torch.where(found,cf,ce0)
            movement=torch.where(found,movement,torch.zeros_like(movement))
            gain=ce0-cf
            weight=torch.where(found,torch.clamp(gain/(ce0+EPS),0,1),torch.zeros_like(gain))
            selected_dir=torch.where(found,ix//na,torch.full_like(ix,-1))
            selected_alpha=torch.where(found,alphas[ix%na],torch.full_like(weight,-1.0))
            native_p.append(p0.float().cpu().numpy());target_p.append(pf.float().cpu().numpy())
            accepted.append(found.cpu().numpy());weights.append(weight.cpu().numpy())
            dirs.append(selected_dir.cpu().numpy());chosen_alpha.append(selected_alpha.cpu().numpy())
            ce0s.append(ce0.cpu().numpy());ces.append(cf.cpu().numpy());improvements.append(gain.cpu().numpy())
            moves.append(movement.float().cpu().numpy());native_correct.append(correct0.cpu().numpy())
            teacher_correct.append(torch.where(found,predsel==yy,correct0).cpu().numpy())
    out={"p0":np.concatenate(native_p),"target_p":np.concatenate(target_p),
         "accepted":np.concatenate(accepted),"weight":np.concatenate(weights),
         "direction":np.concatenate(dirs).astype(np.int16),"alpha":np.concatenate(chosen_alpha),
         "native_CE":np.concatenate(ce0s),"teacher_CE":np.concatenate(ces),
         "CE_improvement":np.concatenate(improvements),"standardized_P_movement":np.concatenate(moves),
         "native_correct":np.concatenate(native_correct),"teacher_correct":np.concatenate(teacher_correct)}
    if np.any(out["accepted"]&(out["CE_improvement"]<=1e-4-1e-6)):
        raise RuntimeError("teacher accepted non-improving candidate")
    if np.any(out["accepted"]&out["native_correct"]&~out["teacher_correct"]):
        raise RuntimeError("teacher damaged native-correct sample")
    return out


def train_arrays(rows,m,g):
    hs=np.concatenate([r["hs"] for r in rows]).astype(np.float32)
    yd=np.concatenate([r["yd"] for r in rows]).astype(np.float32)
    z0=np.concatenate([r["logits"] for r in rows]).astype(np.float32)
    p0,c0,error=decomposition(yd,g["qd"],g["md"])
    if len(hs)!=len(m["y"]):raise RuntimeError("training manifest feature mismatch")
    return {"hs":hs,"yd":yd,"z0":z0,"p0":p0,"c0":c0,"y":m["y"],"manifest":m}


def cache_teacher(fold,model,train,g):
    d=rcell(fold);path=d/"TRAIN_LOCAL_TEACHER_CACHE.npz";meta_path=d/"TRAIN_LOCAL_TEACHER_CACHE.json"
    if path.exists()!=meta_path.exists():raise RuntimeError("incomplete teacher cache")
    if path.exists():
        meta=json.loads(meta_path.read_text())
        if meta["cache_sha256"]!=sha(path) or meta["protocol_sha256"]!=sha(PROTOCOL/"PROTOCOL_LOCK.json") or meta["train_manifest_sha256"]!=train["manifest"]["sha256"]:
            raise RuntimeError("teacher cache hash/manifest mismatch")
        with np.load(path,allow_pickle=False) as z:cache={k:z[k] for k in z.files}
    else:
        values=teacher_search(model,train["hs"],train["yd"],train["z0"],train["y"],g)
        m=train["manifest"]
        cache={"task":np.asarray(TASK),"fold":np.asarray(fold),
            "subject":m["subject"],"session":m["session"],"trial":m["trial"],"label":m["y"],
            "geometry_sha256":np.asarray(sha(d/"FROZEN_GEOMETRY.npz")),
            "train_manifest_sha256":np.asarray(m["sha256"]),"protocol_sha256":np.asarray(sha(PROTOCOL/"PROTOCOL_LOCK.json")),
            **values}
        tmp=d/"TRAIN_LOCAL_TEACHER_CACHE.part.npz"
        np.savez_compressed(tmp,**cache);os.replace(tmp,path)
        meta={"task":TASK,"fold":fold,"cache_sha256":sha(path),
            "protocol_sha256":sha(PROTOCOL/"PROTOCOL_LOCK.json"),
            "train_manifest_sha256":m["sha256"],
            "geometry_sha256":sha(d/"FROZEN_GEOMETRY.npz"),
            "checkpoint_sha256":cell_source(fold)["frozen_checkpoint_sha256"],
            "trials":len(m["y"]),"role":"inner_train_only","teacher_frozen":True}
        jwrite(meta_path,meta)
    if str(cache["train_manifest_sha256"])!=train["manifest"]["sha256"] or str(cache["geometry_sha256"])!=sha(d/"FROZEN_GEOMETRY.npz"):
        raise RuntimeError("teacher cache contents not bound to source")
    for key,arr in (("subject",train["manifest"]["subject"]),("session",train["manifest"]["session"]),
                    ("trial",train["manifest"]["trial"]),("label",train["manifest"]["y"])):
        if not np.array_equal(cache[key],arr):raise RuntimeError(f"teacher trial indexing mismatch: {key}")
    if not np.allclose(cache["p0"],train["p0"],atol=5e-6):raise RuntimeError("teacher native P drift")
    return cache,sha(path)


def teacher_audit(fold,cache,cache_hash):
    accepted=cache["accepted"].astype(bool);correct=cache["native_correct"].astype(bool)
    def frac(mask):return float(accepted[mask].mean()) if np.any(mask) else None
    def meanmask(values,mask):return float(np.mean(values[mask])) if np.any(mask) else None
    def medmask(values,mask):return float(np.median(values[mask])) if np.any(mask) else None
    alpha=Counter(str(float(a)) for a in cache["alpha"][accepted])
    directions=Counter(str(int(d)) for d in cache["direction"][accepted])
    return {"task":TASK,"fold":fold,"total_trials":len(accepted),"accepted_trials":int(accepted.sum()),
        "acceptance_fraction":float(accepted.mean()),"native_correct_trials":int(correct.sum()),
        "acceptance_fraction_native_correct":frac(correct),"acceptance_fraction_native_wrong":frac(~correct),
        "mean_CE_improvement_all":float(np.mean(cache["CE_improvement"])),
        "median_CE_improvement_all":float(np.median(cache["CE_improvement"])),
        "mean_CE_improvement_accepted":meanmask(cache["CE_improvement"],accepted),
        "median_CE_improvement_accepted":medmask(cache["CE_improvement"],accepted),
        "mean_standardized_P_movement_accepted":meanmask(cache["standardized_P_movement"],accepted),
        "median_standardized_P_movement_accepted":medmask(cache["standardized_P_movement"],accepted),
        "mean_teacher_weight_accepted":meanmask(cache["weight"],accepted),
        "selected_alpha_distribution":json.dumps(dict(sorted(alpha.items()))),
        "selected_direction_distribution":json.dumps(dict(sorted(directions.items(),key=lambda x:int(x[0])))),
        "native_correct_damage_count":int(np.sum(accepted&correct&~cache["teacher_correct"])),
        "prediction_rescue_count":int(np.sum(accepted&~correct&cache["teacher_correct"])),
        "teacher_cache_sha256":cache_hash,"teacher_role":"inner_train_only"}


def orders_for(fold,n):
    orders=[np.random.default_rng(seed("batch_order",TASK,fold,epoch,0)).permutation(n).astype(np.int32)
            for epoch in range(1,EPOCHS+1)]
    return orders,arr_sha(*orders)


def block_forward(model,hs):
    shaped=hs.reshape(-1,48,1,hs.shape[1]//48)
    hd=F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(shaped)))),(1,2)).flatten(1)
    return hd,A.suffix(model,hd)


def state_bytes(model):
    return {k:v.detach().cpu().contiguous().numpy().tobytes() for k,v in model.state_dict().items()}


def student_model(fold,classes):
    c=cell_source(fold)
    model=A.build_model(TASK,classes,Path(c["frozen_checkpoint_path"]))
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    model.depth1.weight.requires_grad_(True)
    model.point1.weight.requires_grad_(True)
    allowed={"depth1.weight","point1.weight"}
    if {name for name,p in model.named_parameters() if p.requires_grad}!=allowed:
        raise RuntimeError("trainable parameter whitelist failed")
    return model


def smoke():
    preflight()
    model,rows,m=train_data(0)
    g=geometry_for(0,rows,m)
    train=train_arrays(rows,m,g)
    curves,_,identity=A.eval_curve_bank(model,train["hs"][:2],train["yd"][:2],
        train["z0"][:2],train["y"][:2],g["qs"],g["ms"],g["qd"],g["md"],g["basis"],batch_size=2)
    if identity["representation_max_abs"]>=1e-6 or identity["logits_max_abs"]>=1e-6 or not identity["prediction_exact"]:
        raise RuntimeError("source alpha=1 identity failed")
    targets=teacher_search(model,train["hs"][:16],train["yd"][:16],train["z0"][:16],train["y"][:16],g,batch=8)
    if len(targets["target_p"])!=16:raise RuntimeError("teacher cache indexing smoke failed")
    student=student_model(0,train["z0"].shape[1])
    init=state_bytes(student)
    qd=torch.as_tensor(g["qd"],device=DEVICE)
    md=torch.as_tensor(g["md"],device=DEVICE)
    sigmaP=torch.as_tensor(g["sigmaP"],device=DEVICE)
    sigmaC=torch.as_tensor(g["sigmaC"],device=DEVICE)
    h=torch.as_tensor(train["hs"][:16],device=DEVICE)
    y=torch.as_tensor(train["y"][:16],device=DEVICE)
    d,z=block_forward(student,h)
    p=(d-md)@qd;c=(d-md)-p@qd.T
    pt=torch.as_tensor(targets["target_p"],device=DEVICE)
    c0=torch.as_tensor(train["c0"][:16],device=DEVICE)
    w=torch.as_tensor(targets["weight"],device=DEVICE)
    LP=((((p-pt)/(sigmaP+EPS))**2).mean(1)*w).sum()/(w.sum()+EPS) if float(w.sum())>0 else z.new_zeros(())
    LC=(((c-c0)/(sigmaC+EPS))**2).mean()
    loss=F.cross_entropy(z,y)+LP+LC
    optimizer=torch.optim.AdamW([student.depth1.weight,student.point1.weight],lr=1e-4,weight_decay=5e-4)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    grads={name:float(p.grad.abs().sum()) for name,p in student.named_parameters() if p.grad is not None}
    if set(grads)!={"depth1.weight","point1.weight"} or not all(v>0 for v in grads.values()):
        raise RuntimeError(f"gradient whitelist smoke failed: {grads}")
    torch.nn.utils.clip_grad_norm_([student.depth1.weight,student.point1.weight],5.0)
    optimizer.step()
    after=state_bytes(student)
    changed={k for k in init if init[k]!=after[k]}
    if changed!={"depth1.weight","point1.weight"} or student.training:
        raise RuntimeError(f"state whitelist smoke failed: {changed}")
    order,manifest_sha=orders_for(0,len(train["y"]))
    if not all(np.array_equal(a,b) for a,b in zip(order,orders_for(0,len(train["y"]))[0])):
        raise RuntimeError("arm training orders differ")
    audit={"task":TASK,"fold":0,"status":"PASS","train_trials":len(train["y"]),
        "alpha_one_native_reconstruction":identity,"P_plus_C_reconstruction_max_abs":
            float(np.max(np.abs(train["yd"]-(g["md"]+(train["p0"]@g["qd"].T+train["c0"]))))),
        "teacher_sample_index_count":len(targets["target_p"]),
        "teacher_accepted_in_smoke":int(targets["accepted"].sum()),
        "gradient_parameter_names":sorted(grads),"changed_state_names":sorted(changed),
        "BN_running_state_byte_identical":all(init[k]==after[k] for k in init if "running_" in k or "num_batches_tracked" in k),
        "eval_mode_with_gradients":True,"identical_arm_order_sha256":manifest_sha,
        "identical_optimizer_step_count":len(order)*math.ceil(len(train["y"])/BATCH),
        "geometry_file_sha256":sha(rcell(0)/"FROZEN_GEOMETRY.npz")}
    jwrite(PROTOCOL/"SMOKE_AUDIT.json",audit)
    print("SMOKE_PASS",json.dumps(audit),flush=True)


def train_arm(fold,arm,train,cache,g,order,order_sha):
    d=rcell(fold);ck=d/f"{arm}.pt";audit_path=d/f"{arm}_STATE_AUDIT.json";history_path=d/f"{arm}_TRAINING_HISTORY.csv"
    if ck.exists()!=audit_path.exists():raise RuntimeError(f"partial completed arm refuses overwrite {fold}/{arm}")
    if ck.exists():
        audit=json.loads(audit_path.read_text())
        if audit["checkpoint_sha256"]!=sha(ck) or audit["protocol_sha256"]!=sha(PROTOCOL/"PROTOCOL_LOCK.json") or audit["order_sha256"]!=order_sha:
            raise RuntimeError("completed arm audit mismatch")
        if not history_path.exists() or len(cread(history_path))!=EPOCHS:raise RuntimeError("completed arm history missing")
        model=student_model(fold,train["z0"].shape[1])
        model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["state_dict"],strict=True)
        print("ARM_ALREADY_COMPLETE",fold,arm,flush=True)
        return model,cread(history_path),audit
    s=seed("student",TASK,fold,0)
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    model=student_model(fold,train["z0"].shape[1])
    init=state_bytes(model);init_sha=A.model_state_sha(model)
    opt=torch.optim.AdamW([model.depth1.weight,model.point1.weight],lr=1e-4,weight_decay=5e-4)
    qd=torch.as_tensor(g["qd"],device=DEVICE)
    md=torch.as_tensor(g["md"],device=DEVICE)
    sigmaP=torch.as_tensor(g["sigmaP"],device=DEVICE)
    sigmaC=torch.as_tensor(g["sigmaC"],device=DEVICE)
    n=len(train["y"]);history=[];steps=0
    for epoch,permutation in enumerate(order,1):
        total=np.zeros(8,dtype=np.float64)
        for start in range(0,n,BATCH):
            ix=permutation[start:start+BATCH];b=len(ix)
            h=torch.as_tensor(np.ascontiguousarray(train["hs"][ix]),device=DEVICE)
            y=torch.as_tensor(train["y"][ix],device=DEVICE,dtype=torch.long)
            dtheta,z=block_forward(model,h)
            ce=F.cross_entropy(z,y)
            centered=dtheta-md;p=centered@qd;c=centered-p@qd.T
            p0=torch.as_tensor(cache["p0"][ix],device=DEVICE)
            c0=torch.as_tensor(np.ascontiguousarray(train["c0"][ix]),device=DEVICE)
            w=torch.as_tensor(cache["weight"][ix],device=DEVICE)
            lp=z.new_zeros(());lc=z.new_zeros(())
            if arm=="LOCAL_CONSTRAINED_P":
                target=torch.as_tensor(cache["target_p"][ix],device=DEVICE)
                if float(w.sum())>0:
                    lp=((((p-target)/(sigmaP+EPS))**2).mean(1)*w).sum()/(w.sum()+EPS)
                lc=(((c-c0)/(sigmaC+EPS))**2).mean()
            loss=ce+lp+lc
            opt.zero_grad(set_to_none=True);loss.backward()
            if any(param.grad is None for param in (model.depth1.weight,model.point1.weight)):
                raise RuntimeError("missing shared-block gradient")
            torch.nn.utils.clip_grad_norm_([model.depth1.weight,model.point1.weight],5.0)
            opt.step();steps+=1
            pdrift=torch.sqrt(torch.mean(((p.detach()-p0)/(sigmaP+EPS))**2,dim=1)).mean()
            cdrift=torch.sqrt(torch.mean(((c.detach()-c0)/(sigmaC+EPS))**2,dim=1)).mean()
            total+=np.asarray([float(ce.detach()),float(lp.detach()),float(lc.detach()),float(loss.detach()),
                float(w.mean()),float(pdrift),float(cdrift),1.0])*b
        history.append({"task":TASK,"fold":fold,"arm":arm,"epoch":epoch,"trials":n,
            "CE":total[0]/n,"LP":total[1]/n,"LC":total[2]/n,"total_loss":total[3]/n,
            "mean_teacher_weight_in_batches":total[4]/n,"P_drift_standardized_rms":total[5]/n,
            "C_drift_standardized_rms":total[6]/n,"optimizer_steps_cumulative":steps,
            "batch_order_manifest_sha256":order_sha})
        cwrite(history_path,history)
        print("EPOCH",fold,arm,epoch,"CE",round(total[0]/n,5),"LP",round(total[1]/n,5),"LC",round(total[2]/n,5),flush=True)
    after=state_bytes(model)
    changed={name for name in init if init[name]!=after[name]}
    if changed!={"depth1.weight","point1.weight"}:
        raise RuntimeError(f"unallowed state change: {sorted(changed)}")
    if any(init[name]!=after[name] for name in init if "running_" in name or "num_batches_tracked" in name):
        raise RuntimeError("BatchNorm running-state drift")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("student left eval mode")
    temp=ck.with_suffix(".pt.part")
    torch.save({"state_dict":model.state_dict(),"task":TASK,"fold":fold,"arm":arm,
        "epoch":EPOCHS,"protocol_sha256":sha(PROTOCOL/"PROTOCOL_LOCK.json"),
        "teacher_cache_sha256":sha(d/"TRAIN_LOCAL_TEACHER_CACHE.npz")},temp)
    os.replace(temp,ck)
    audit={"task":TASK,"fold":fold,"arm":arm,"source_checkpoint_sha256":cell_source(fold)["frozen_checkpoint_sha256"],
        "initial_model_state_sha256":init_sha,"final_model_state_sha256":A.model_state_sha(model),
        "checkpoint_sha256":sha(ck),"checkpoint_runtime_path":str(ck),"protocol_sha256":sha(PROTOCOL/"PROTOCOL_LOCK.json"),
        "train_manifest_sha256":train["manifest"]["sha256"],"order_sha256":order_sha,
        "optimizer":"AdamW","lr":1e-4,"weight_decay":5e-4,"gradient_clip":5.0,
        "epochs":EPOCHS,"optimizer_steps":steps,"expected_steps":EPOCHS*math.ceil(n/BATCH),
        "changed_state_names":sorted(changed),"all_other_state_bit_identical":True,
        "BN_running_state_byte_identical":True,"dropout_disabled":True,"model_eval_mode":True,
        "trainable_parameter_names":["depth1.weight","point1.weight"],
        "trainable_parameter_count":sum(p.numel() for p in (model.depth1.weight,model.point1.weight))}
    if steps!=audit["expected_steps"]:raise RuntimeError("optimizer step count drift")
    jwrite(audit_path,audit)
    return model,history,audit


def metric(y,z):
    y=np.asarray(y,np.int64);z=np.asarray(z,np.float32)
    pred=z.argmax(1)
    nll=float(F.cross_entropy(torch.as_tensor(z),torch.as_tensor(y),reduction="mean"))
    return {"BA":float(balanced_accuracy_score(y,pred)),
            "macro_F1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "NLL":nll,"trials":len(y)}


def eval_native(model,raw,mean,std):
    model.eval();hs_all=[];hd_all=[];z_all=[]
    with torch.inference_mode():
        for start in range(0,len(raw),100):
            x=((raw[start:start+100]-mean[None,:,None])/np.maximum(std[None,:,None],1e-6)).astype(np.float32)
            xt=torch.as_tensor(np.ascontiguousarray(x),device=DEVICE)
            hs,hd,z,_=A.forward_stages(model,xt)
            # Audit one canonical forward, outside the measured inference outputs.
            if start==0:
                native=model(xt)[0]
                if float(torch.max(torch.abs(native-z)))>=1e-6:raise RuntimeError("student manual native forward mismatch")
            hs_all.append(hs.flatten(1).float().cpu().numpy())
            hd_all.append(hd.flatten(1).float().cpu().numpy())
            z_all.append(z.float().cpu().numpy())
    return {"hs":np.concatenate(hs_all),"hd":np.concatenate(hd_all),"z":np.concatenate(z_all)}


def suffix_logits(model,rep):
    out=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(rep),100):
            x=torch.as_tensor(np.ascontiguousarray(rep[start:start+100]),device=DEVICE)
            out.append(A.suffix(model,x).float().cpu().numpy())
    return np.concatenate(out)


def alignment_row(fold,subject,session,arm,p0,p,target,sigmaP,accepted):
    target_delta=(target-p0)/(sigmaP+EPS)
    student_delta=(p-p0)/(sigmaP+EPS)
    e0=np.mean(target_delta**2,axis=1)
    es=np.mean((student_delta-target_delta)**2,axis=1)
    target_norm=np.linalg.norm(target_delta,axis=1)
    student_norm=np.linalg.norm(student_delta,axis=1)
    mask=(target_norm>1e-8)&(student_norm>1e-8)&accepted
    cos=np.sum(target_delta[mask]*student_delta[mask],axis=1)/np.maximum(target_norm[mask]*student_norm[mask],1e-12)
    E0=float(e0.mean());ES=float(es.mean())
    return {"task":TASK,"fold":fold,"subject":subject,"session":session,"arm":arm,
        "label":"POST_HOC_DISCOVERY_DIAGNOSTIC","trials":len(p0),
        "teacher_accepted_trials":int(accepted.sum()),"teacher_acceptance_fraction":float(accepted.mean()),
        "E_zero":E0,"E_student":ES,
        "E_student_over_E_zero":ES/E0 if E0>1e-12 else "undefined_zero_teacher_update",
        "student_better_than_zero":ES<E0,"nonzero_cosine_trials":int(mask.sum()),
        "mean_nonzero_update_cosine":float(cos.mean()) if len(cos) else "undefined_no_nonzero_pair"}


def drift_row(fold,subject,session,arm,p0,c0,p,c,g):
    dp=p-p0;dc=c-c0
    pn=np.linalg.norm(p0,axis=1);cn=np.linalg.norm(c0,axis=1)
    return {"task":TASK,"fold":fold,"subject":subject,"session":session,"arm":arm,
        "trials":len(p),"P_raw_L2_movement_mean":float(np.linalg.norm(dp,axis=1).mean()),
        "P_standardized_RMS_movement_mean":float(np.sqrt(np.mean((dp/(g["sigmaP"]+EPS))**2,axis=1)).mean()),
        "P_movement_to_native_norm_ratio_mean":float(np.mean(np.linalg.norm(dp,axis=1)/np.maximum(pn,EPS))),
        "C_raw_L2_movement_mean":float(np.linalg.norm(dc,axis=1).mean()),
        "C_standardized_RMS_movement_mean":float(np.sqrt(np.mean((dc/(g["sigmaC"]+EPS))**2,axis=1)).mean()),
        "C_movement_to_native_norm_ratio_mean":float(np.mean(np.linalg.norm(dc,axis=1)/np.maximum(cn,EPS)))}


def evaluate_cell(fold,models,g):
    d=rcell(fold);c=cell_source(fold)
    ids=c["discovery_subjects"]
    sessions=sorted(set(map(int,c["source_sessions"]+[c["future_session"]])))
    pieces,mapping=A.fetch_role(TASK,ids,sessions,c["cache_name"],None)
    mean,std=A.load_normalizer(Path(c["normalizer_path"]))
    baseline=A.build_model(TASK,len(mapping),Path(c["frozen_checkpoint_path"]))
    metrics=[];drift=[];transplant=[];alignment=[]
    # Native evaluations finish and are persisted before any discovery oracle is built.
    native_records=[]
    for session,subject,raw,y,owners in pieces:
        values={"BASELINE":eval_native(baseline,raw,mean,std)}
        for arm in ARMS:values[arm]=eval_native(models[arm],raw,mean,std)
        for arm,v in values.items():
            metrics.append({"task":TASK,"fold":fold,"subject":subject,"session":session,
                "is_future_session":session==c["future_session"],"arm":arm,**metric(y,v["z"])})
        native_records.append((session,subject,np.asarray(y,np.int64),values))
    cwrite(d/"DISCOVERY_METRICS.csv",metrics)
    jwrite(d/"DISCOVERY_NATIVE_EVALUATION_LOCK.json",{
        "task":TASK,"fold":fold,"checkpoint_hashes":{arm:sha(d/f"{arm}.pt") for arm in ARMS},
        "metrics_sha256":sha(d/"DISCOVERY_METRICS.csv"),
        "before_discovery_label_conditional_teacher":True,"model_selection_from_discovery":False})
    # Diagnostic teacher may now read discovery labels. It cannot change trained weights.
    before={arm:A.model_state_sha(model) for arm,model in models.items()}
    for session,subject,y,values in native_records:
        base=values["BASELINE"]
        p0,c0,_=decomposition(base["hd"],g["qd"],g["md"])
        local=values["LOCAL_CONSTRAINED_P"]
        p_local,c_local,_=decomposition(local["hd"],g["qd"],g["md"])
        for arm in ARMS:
            p,c,_=decomposition(values[arm]["hd"],g["qd"],g["md"])
            drift.append(drift_row(fold,subject,session,arm,p0,c0,p,c,g))
        qd=np.asarray(g["qd"],np.float64);md=np.asarray(g["md"],np.float64)
        rep_base=(md+p0.astype(np.float64)@qd.T+c0.astype(np.float64)).astype(np.float32)
        rep_full=(md+p_local.astype(np.float64)@qd.T+c_local.astype(np.float64)).astype(np.float32)
        rep_p=(md+p_local.astype(np.float64)@qd.T+c0.astype(np.float64)).astype(np.float32)
        rep_c=(md+p0.astype(np.float64)@qd.T+c_local.astype(np.float64)).astype(np.float32)
        if max(float(np.max(np.abs(rep_base-base["hd"]))),float(np.max(np.abs(rep_full-local["hd"]))))>=1e-5:
            raise RuntimeError("transplant representation reconstruction failed")
        zs={"BASE":base["z"],"FULL_STUDENT":local["z"],
            "P_ONLY_CHANGE":suffix_logits(baseline,rep_p),"C_ONLY_CHANGE":suffix_logits(baseline,rep_c)}
        if not np.array_equal(suffix_logits(baseline,rep_base).argmax(1),zs["BASE"].argmax(1)):
            raise RuntimeError("BASE transplant prediction mismatch")
        if not np.array_equal(suffix_logits(baseline,rep_full).argmax(1),zs["FULL_STUDENT"].argmax(1)):
            raise RuntimeError("FULL transplant prediction mismatch")
        for variant,z in zs.items():
            transplant.append({"task":TASK,"fold":fold,"subject":subject,"session":session,
                "variant":variant,"suffix":"original_frozen_SIRE_suffix","label":"POST_HOC_TRANSPLANT_DIAGNOSTIC",
                **metric(y,z)})
        teacher=teacher_search(baseline,base["hs"],base["hd"],base["z"],y,g,batch=8)
        for arm in ARMS:
            p,_,_=decomposition(values[arm]["hd"],g["qd"],g["md"])
            alignment.append(alignment_row(fold,subject,session,arm,p0,p,teacher["target_p"],g["sigmaP"],teacher["accepted"]))
        print("DISCOVERY_DIAGNOSTIC",fold,subject,session,flush=True)
    if any(A.model_state_sha(models[arm])!=before[arm] for arm in ARMS):
        raise RuntimeError("discovery diagnostic changed student state")
    cwrite(d/"P_UPDATE_ALIGNMENT.csv",alignment)
    cwrite(d/"PC_DRIFT_AUDIT.csv",drift)
    cwrite(d/"TRANSPLANT_RESULTS.csv",transplant)
    return metrics,alignment,drift,transplant


def run_cell(fold):
    lock=preflight()
    if fold not in FOLDS:raise ValueError("unlocked fold")
    if not (PROTOCOL/"SMOKE_AUDIT.json").exists():raise RuntimeError("fold0 smoke audit required before five-fold run")
    d=rcell(fold);d.mkdir(parents=True,exist_ok=True)
    done=d/"CELL_COMPLETE.json"
    if done.exists():
        record=json.loads(done.read_text())
        if record["protocol_sha256"]==sha(PROTOCOL/"PROTOCOL_LOCK.json") and all(sha(d/name)==digest for name,digest in record["output_sha256"].items()):
            print("CELL_ALREADY_COMPLETE",fold,flush=True);return
        raise RuntimeError("completed cell changed; refusing overwrite")
    model,rows,m=train_data(fold)
    g=geometry_for(fold,rows,m)
    train=train_arrays(rows,m,g)
    source_state=A.model_state_sha(model)
    cache,cache_hash=cache_teacher(fold,model,train,g)
    cwrite(d/"TEACHER_TARGET_AUDIT.csv",[teacher_audit(fold,cache,cache_hash)])
    if A.model_state_sha(model)!=source_state:raise RuntimeError("teacher changed during cache generation")
    orders,order_sha=orders_for(fold,len(m["y"]))
    models={};histories=[];audits=[]
    for arm in ARMS:
        student,history,audit=train_arm(fold,arm,train,cache,g,orders,order_sha)
        models[arm]=student;histories.extend(history);audits.append(audit)
    if len({a["initial_model_state_sha256"] for a in audits})!=1 or len({a["order_sha256"] for a in audits})!=1 or len({a["optimizer_steps"] for a in audits})!=1:
        raise RuntimeError("arms differ in initial state, sample order or optimizer step count")
    cwrite(d/"TRAINING_HISTORY.csv",histories)
    jwrite(d/"TRAINABLE_STATE_AUDIT.json",{"task":TASK,"fold":fold,
        "identical_initialization":True,"identical_batch_order":True,"identical_optimizer_steps":True,
        "arms":audits,"source_teacher_state_sha256":source_state,
        "source_teacher_state_unchanged":A.model_state_sha(model)==source_state,
        "teacher_cache_sha256":cache_hash})
    del rows,train,cache,model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    evaluate_cell(fold,models,g)
    files=["TEACHER_TARGET_AUDIT.csv","TRAINING_HISTORY.csv","DISCOVERY_METRICS.csv","P_UPDATE_ALIGNMENT.csv",
           "PC_DRIFT_AUDIT.csv","TRANSPLANT_RESULTS.csv","TRAINABLE_STATE_AUDIT.json",
           "DISCOVERY_NATIVE_EVALUATION_LOCK.json"]
    jwrite(done,{"task":TASK,"fold":fold,"protocol_sha256":sha(PROTOCOL/"PROTOCOL_LOCK.json"),
        "geometry_sha256":sha(d/"FROZEN_GEOMETRY.npz"),"teacher_cache_sha256":cache_hash,
        "checkpoint_sha256":{arm:sha(d/f"{arm}.pt") for arm in ARMS},
        "output_sha256":{name:sha(d/name) for name in files},
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,
        "discovery_diagnostic_after_native_evaluation":True})
    print("CELL_COMPLETE",fold,flush=True)


def subject_values(rows,arm,field,fold=None):
    grouped=defaultdict(list)
    for r in rows:
        if r["arm"]!=arm or (fold is not None and int(r["fold"])!=fold):continue
        grouped[str(r["subject"])].append(float(r[field]))
    return {subject:float(np.mean(values)) for subject,values in grouped.items()}


def paired(a,b,estimand):
    subjects=sorted(set(a)&set(b),key=int)
    if not subjects:raise RuntimeError(f"no paired biological subjects: {estimand}")
    delta=np.asarray([a[s]-b[s] for s in subjects],np.float64)
    rng=np.random.default_rng(seed("paired_bootstrap",estimand))
    draw=rng.integers(0,len(delta),size=(BOOTSTRAPS,len(delta)))
    bootstrap=delta[draw].mean(1)
    return {"biological_subjects":len(subjects),"delta":float(delta.mean()),
        "ci95_lower":float(np.quantile(bootstrap,.025)),
        "ci95_upper":float(np.quantile(bootstrap,.975)),"bootstrap_replicates":BOOTSTRAPS}


def summarize_metric(rows,arm,field,fold=None):
    values=subject_values(rows,arm,field,fold)
    return float(np.mean(list(values.values())))


def group_subject_metrics(metrics):
    grouped=defaultdict(list)
    for r in metrics:grouped[(str(r["subject"]),r["arm"])].append(r)
    subject_rows=[]
    for (subject,arm),rows in grouped.items():
        by_fold=defaultdict(list)
        for r in rows:by_fold[int(r["fold"])].append(float(r["BA"]))
        future=[float(r["BA"]) for r in rows if str(r["is_future_session"]).lower()=="true"]
        subject_rows.append({"task":TASK,"subject":subject,"arm":arm,
            "fold_appearances":len(by_fold),"sessions_per_fold":2,
            "BA":float(np.mean([float(r["BA"]) for r in rows])),
            "macro_F1":float(np.mean([float(r["macro_F1"]) for r in rows])),
            "NLL":float(np.mean([float(r["NLL"]) for r in rows])),
            "future_session_BA":float(np.mean(future)),
            "worst_session_BA":float(np.mean([min(values) for values in by_fold.values()]))})
    return sorted(subject_rows,key=lambda r:(int(r["subject"]),r["arm"]))


def aggregate():
    lock=preflight()
    buckets={name:[] for name in ("TEACHER_TARGET_AUDIT.csv","TRAINING_HISTORY.csv","DISCOVERY_METRICS.csv",
        "P_UPDATE_ALIGNMENT.csv","PC_DRIFT_AUDIT.csv","TRANSPLANT_RESULTS.csv")}
    state=[];cell_audits=[]
    for fold in FOLDS:
        d=rcell(fold);done=json.loads((d/"CELL_COMPLETE.json").read_text())
        if done["protocol_sha256"]!=sha(PROTOCOL/"PROTOCOL_LOCK.json") or done["outer_dev_eeg_reads"] or done["final_heldout_eeg_reads"]:
            raise RuntimeError(f"cell lock/leakage audit failure {fold}")
        if done["teacher_cache_sha256"]!=sha(d/"TRAIN_LOCAL_TEACHER_CACHE.npz") or done["geometry_sha256"]!=sha(d/"FROZEN_GEOMETRY.npz"):
            raise RuntimeError(f"cell cache hash changed {fold}")
        if any(sha(d/name)!=digest for name,digest in done["output_sha256"].items()):
            raise RuntimeError(f"cell output changed {fold}")
        cell_audits.append(done)
        state_row=json.loads((d/"TRAINABLE_STATE_AUDIT.json").read_text())
        arms=state_row["arms"]
        if {a["arm"] for a in arms}!=set(ARMS) or len({a["initial_model_state_sha256"] for a in arms})!=1 or len({a["order_sha256"] for a in arms})!=1 or len({a["optimizer_steps"] for a in arms})!=1:
            raise RuntimeError(f"arms incomparable {fold}")
        for a in arms:
            if a["changed_state_names"]!=["depth1.weight","point1.weight"] or not a["all_other_state_bit_identical"] or not a["BN_running_state_byte_identical"] or a["optimizer_steps"]!=a["expected_steps"]:
                raise RuntimeError(f"trainable state whitelist failure {fold}/{a['arm']}")
            if sha(Path(a["checkpoint_runtime_path"]))!=a["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch {fold}/{a['arm']}")
        state.append(state_row)
        for name in buckets:buckets[name].extend(cread(d/name))
    for name,rows in buckets.items():cwrite(OUT/name,rows)
    jwrite(OUT/"TRAINABLE_STATE_AUDIT.json",{"task":TASK,"folds":list(FOLDS),"all_invariants_passed":True,
        "same_optimizer_manifest_and_steps_per_fold":True,"only_depth1_and_point1_changed":True,
        "BN_running_states_unchanged":True,"cells":state})
    metrics=buckets["DISCOVERY_METRICS.csv"]
    subject_rows=group_subject_metrics(metrics)
    cwrite(OUT/"SUBJECT_METRICS.csv",subject_rows)
    folds=[]
    for fold in FOLDS:
        for arm in ("BASELINE",)+ARMS:
            row={"task":TASK,"fold":fold,"arm":arm,"biological_subjects":len(subject_values(metrics,arm,"BA",fold))}
            for field in ("BA","macro_F1","NLL"):
                row[field]=summarize_metric(metrics,arm,field,fold)
            bysubject=[r for r in subject_rows if r["arm"]==arm]
            row["future_session_BA"]=float(np.mean([float(r["BA"]) for r in metrics if r["arm"]==arm and int(r["fold"])==fold and str(r["is_future_session"]).lower()=="true"]))
            worst=[]
            for subject in subject_values(metrics,arm,"BA",fold):
                worst.append(min(float(r["BA"]) for r in metrics if r["arm"]==arm and int(r["fold"])==fold and r["subject"]==subject))
            row["worst_session_BA"]=float(np.mean(worst))
            if arm=="LOCAL_CONSTRAINED_P":
                p=paired(subject_values(metrics,arm,"BA",fold),subject_values(metrics,"CE_ONLY_CONTINUATION","BA",fold),f"fold{fold}_local_vs_CE_BA")
                row.update({"delta_BA_vs_CE":p["delta"],"delta_BA_ci95_lower":p["ci95_lower"],"delta_BA_ci95_upper":p["ci95_upper"]})
            folds.append(row)
    cwrite(OUT/"FOLD_SUMMARY.csv",folds)
    contrasts=[]
    for arm,comparator in (("LOCAL_CONSTRAINED_P","CE_ONLY_CONTINUATION"),
                           ("LOCAL_CONSTRAINED_P","BASELINE"),("CE_ONLY_CONTINUATION","BASELINE")):
        for field in ("BA","macro_F1","NLL","worst_session_BA"):
            a=subject_values(subject_rows,arm,field);b=subject_values(subject_rows,comparator,field)
            contrasts.append({"task":TASK,"arm":arm,"comparator":comparator,"metric":field,
                **paired(a,b,f"all_{arm}_{comparator}_{field}")})
    cwrite(OUT/"PAIRED_CONTRASTS.csv",contrasts)
    jwrite(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json",{
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,"FINAL_HELDOUT_ACCESSED":False,
        "teacher_construction":"inner_train only","scale_estimation":"inner_train only",
        "student_training":"inner_train only","discovery_training_feedback":False,
        "diagnostic_discovery_teacher_after_checkpoints_frozen":True,
        "historical_checkpoint_final_heldout_diagnostic_exposure":True,
        "historical_provenance_caveat":lock["historical_provenance_caveat"],
        "completed_cells":len(cell_audits)})
    report(folds,subject_rows,contrasts,buckets)
    print("AGGREGATE_COMPLETE",flush=True)


def report(folds,subjects,contrasts,buckets):
    teacher=buckets["TEACHER_TARGET_AUDIT.csv"]
    align=buckets["P_UPDATE_ALIGNMENT.csv"]
    drift=buckets["PC_DRIFT_AUDIT.csv"]
    transplant=buckets["TRANSPLANT_RESULTS.csv"]
    metric=buckets["DISCOVERY_METRICS.csv"]
    get=lambda arm,field: summarize_metric(subjects,arm,field)
    primary=next(r for r in contrasts if r["arm"]=="LOCAL_CONSTRAINED_P" and r["comparator"]=="CE_ONLY_CONTINUATION" and r["metric"]=="BA")
    foldlines=[]
    for fold in FOLDS:
        rec={r["arm"]:r for r in folds if int(r["fold"])==fold}
        loc=rec["LOCAL_CONSTRAINED_P"]
        foldlines.append(f"| {fold} | {float(rec['BASELINE']['BA']):.4f} | {float(rec['CE_ONLY_CONTINUATION']['BA']):.4f} | {float(loc['BA']):.4f} | {float(loc['delta_BA_vs_CE']):+.4f} | [{float(loc['delta_BA_ci95_lower']):+.4f}, {float(loc['delta_BA_ci95_upper']):+.4f}] |")
    foldlines.append(f"| Pooled biological subjects | {get('BASELINE','BA'):.4f} | {get('CE_ONLY_CONTINUATION','BA'):.4f} | {get('LOCAL_CONSTRAINED_P','BA'):.4f} | {float(primary['delta']):+.4f} | [{float(primary['ci95_lower']):+.4f}, {float(primary['ci95_upper']):+.4f}] |")
    secondary=[]
    for arm in ("BASELINE",)+ARMS:
        secondary.append(f"| {arm} | {get(arm,'macro_F1'):.4f} | {get(arm,'NLL'):.4f} | {get(arm,'worst_session_BA'):.4f} | {get(arm,'future_session_BA'):.4f} |")
    teacherlines=[]
    for r in teacher:
        teacherlines.append(f"| {int(r['fold'])} | {float(r['acceptance_fraction']):.1%} | {float(r['mean_CE_improvement_accepted']):.4f} | {float(r['mean_standardized_P_movement_accepted']):.4f} | {r['selected_alpha_distribution']} | {r['selected_direction_distribution']} |")
    total_trials=sum(int(r["total_trials"]) for r in teacher)
    accepted=sum(int(r["accepted_trials"]) for r in teacher)
    accept_rate=accepted/total_trials
    # Equal-size subject/session groups allow direct mean of diagnostic rows within fold.
    alignlines=[];byfold={}
    for fold in FOLDS:
        a=[r for r in align if int(r["fold"])==fold and r["arm"]=="LOCAL_CONSTRAINED_P"]
        E0=float(np.mean([float(r["E_zero"]) for r in a]));ES=float(np.mean([float(r["E_student"]) for r in a]))
        cosrows=[r for r in a if r["mean_nonzero_update_cosine"]!="undefined_no_nonzero_pair"]
        cos=float(np.mean([float(r["mean_nonzero_update_cosine"]) for r in cosrows])) if cosrows else float("nan")
        byfold[fold]=(E0,ES,cos)
        alignlines.append(f"| {fold} | {ES/E0 if E0>1e-12 else float('nan'):.3f} | {cos:+.3f} | {ES<E0} |")
    allalign=[r for r in align if r["arm"]=="LOCAL_CONSTRAINED_P"]
    E0=float(np.mean([float(r["E_zero"]) for r in allalign]));ES=float(np.mean([float(r["E_student"]) for r in allalign]))
    ratio=ES/E0 if E0>1e-12 else float("nan")
    validcos=[float(r["mean_nonzero_update_cosine"]) for r in allalign if r["mean_nonzero_update_cosine"]!="undefined_no_nonzero_pair"]
    cosine=float(np.mean(validcos)) if validcos else float("nan")
    alignlines.append(f"| Pooled | {ratio:.3f} | {cosine:+.3f} | {ES<E0} |")
    variants=("BASE","FULL_STUDENT","P_ONLY_CHANGE","C_ONLY_CHANGE")
    translines=[]
    for fold in FOLDS:
        vals={}
        for v in variants:
            group=defaultdict(list)
            for r in transplant:
                if int(r["fold"])==fold and r["variant"]==v:group[r["subject"]].append(float(r["BA"]))
            vals[v]=float(np.mean([np.mean(x) for x in group.values()]))
        translines.append(f"| {fold} | {vals['BASE']:.4f} | {vals['FULL_STUDENT']:.4f} | {vals['P_ONLY_CHANGE']:.4f} | {vals['C_ONLY_CHANGE']:.4f} |")
    pooled={}
    for v in variants:
        group=defaultdict(list)
        for r in transplant:
            if r["variant"]==v:group[r["subject"]].append(float(r["BA"]))
        pooled[v]=float(np.mean([np.mean(x) for x in group.values()]))
    # The first two transplant rows use the same native logits as Q1. Reuse
    # that aggregate exactly to avoid a last-digit rounding discrepancy.
    pooled["BASE"]=get("BASELINE","BA")
    pooled["FULL_STUDENT"]=get("LOCAL_CONSTRAINED_P","BA")
    full_gain=pooled["FULL_STUDENT"]-pooled["BASE"]
    p_gain=pooled["P_ONLY_CHANGE"]-pooled["BASE"]
    p_ratio=p_gain/full_gain if full_gain>1e-12 else None
    translines.append(f"| Pooled | {pooled['BASE']:.4f} | {pooled['FULL_STUDENT']:.4f} | {pooled['P_ONLY_CHANGE']:.4f} | {pooled['C_ONLY_CHANGE']:.4f} |")
    driftlines=[]
    for arm in ARMS:
        a=[r for r in drift if r["arm"]==arm]
        pd=float(np.mean([float(r["P_standardized_RMS_movement_mean"]) for r in a]))
        cd=float(np.mean([float(r["C_standardized_RMS_movement_mean"]) for r in a]))
        pr=float(np.mean([float(r["P_movement_to_native_norm_ratio_mean"]) for r in a]))
        cr=float(np.mean([float(r["C_movement_to_native_norm_ratio_mean"]) for r in a]))
        driftlines.append(f"| {arm} | {pd:.4f} | {cd:.4f} | {pr:.4f} | {cr:.4f} |")
    positive_folds=sum(float(next(r for r in folds if int(r["fold"])==fold and r["arm"]=="LOCAL_CONSTRAINED_P")["delta_BA_vs_CE"])>0 for fold in FOLDS)
    delta=float(primary["delta"])
    if accept_rate<.05 or float(np.mean([float(r["mean_CE_improvement_all"]) for r in teacher]))<1e-4:
        label="NO_USEFUL_LOCAL_TEACHER_SIGNAL"
    elif delta>0 and positive_folds>=4 and ratio<1 and p_ratio is not None and p_ratio>=.5:
        label="PROMISING_P_CONSTRUCTION_SIGNAL"
    elif delta>0:
        label="PERFORMANCE_GAIN_NOT_P_MEDIATED"
    else:
        label="TEACHER_NOT_DISTILLABLE"
    p_ratio_text=f"{p_ratio:.3f}" if p_ratio is not None else "undefined because FULL_STUDENT does not beat BASE"
    text=f"""# Native SIRE local P construction: completed seed-0 pilot

Canonical SIRE-EEG / CompactLite, OpenBMI MI, folds 0–4. Only `depth1.weight` and `point1.weight` were trainable; all BN states, dropout behavior, source branches and suffix remained frozen. Epoch 20 is the fixed endpoint. The primary aggregate averages the two discovery sessions within each biological subject, then averages subjects, with repeated fold appearances deduplicated. Future-session BA is also shown. All confidence intervals use 20,000 paired biological-subject bootstrap draws.

The canonical checkpoints have historical final-heldout diagnostic exposure in the source record. This run read zero outer-dev or final-heldout EEG arrays and used no discovery label for training or selection.

## Q1. Performance against CE-only continuation

| Fold | Baseline BA | CE-only BA | Local-P BA | Local-P minus CE-only | 95% paired CI |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(foldlines)}

| Arm | Macro-F1 | NLL | Worst-session BA | Future-session BA |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(secondary)}

The comparison with the original checkpoint is secondary. The complete per-subject/session values and all paired metric contrasts are in `DISCOVERY_METRICS.csv`, `SUBJECT_METRICS.csv`, and `PAIRED_CONTRASTS.csv`.

## Q2. Usable local teacher supervision

The teacher searches one source-C direction at a time over alpha 0.5, 0.75, 1.25 and 1.5, accepts only a CE improvement greater than 1e-4 without damaging a native-correct prediction, and selects the smallest standardized P movement. It is generated on inner-train labels using the original frozen shared block. Accepted {accepted}/{total_trials} training trials ({accept_rate:.1%}).

| Fold | Accepted | Mean accepted CE improvement | Mean accepted P movement | Selected alpha counts | Selected direction counts |
| --- | ---: | ---: | ---: | --- | --- |
{chr(10).join(teacherlines)}

## Q3. Learned P update

`E_zero` is the squared standardized error of making no P update; `E_student` is the error of the native student P update. Values below 1 favor the student. Cosines use nonzero update pairs and are post-hoc diagnostics only.

| Fold | E_student / E_zero | Mean cosine | Better than zero update |
| --- | ---: | ---: | --- |
{chr(10).join(alignlines)}

## Q4. P/C transplant attribution

All four representations pass through the same frozen suffix. These evaluations are diagnostic and need not be additive.

| Fold | BASE BA | FULL_STUDENT BA | P_ONLY_CHANGE BA | C_ONLY_CHANGE BA |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(translines)}

P-only/full gain ratio: {p_ratio_text}. FULL minus BASE BA: {full_gain:+.4f}; P-only minus BASE BA: {p_gain:+.4f}. A positive classifier result is P mediated only when the P-only transplant retains a substantial part of the full gain.

## Q5. Native P/C drift

| Arm | Standardized P drift | Standardized C drift | P drift/native norm | C drift/native norm |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(driftlines)}

## Diagnostic classification

`{label}`. Local-P minus CE-only BA is {delta:+.4f}; {positive_folds}/5 folds have a positive difference; P-update error ratio is {ratio:.3f}. The label is a development diagnostic, not an independent validation claim.

The exact old SIRE geometry hashes, teacher cache hashes, trainable-state whitelist, BN byte identity and identical batch/optimizer-step audits are recorded in the protocol and compact outputs. Trial caches and checkpoints remain in server runtime storage.
"""
    (OUT/"FINAL_REPORT.md").write_text(text,encoding="utf-8")
    jwrite(OUT/"DECISION_SUMMARY.json",{"diagnostic_label":label,"primary_delta_BA":delta,
        "primary_ci95_lower":float(primary["ci95_lower"]),"primary_ci95_upper":float(primary["ci95_upper"]),
        "positive_folds":positive_folds,"teacher_acceptance_fraction":accept_rate,
        "E_student_over_E_zero":ratio,"transplant_P_only_over_full_gain":p_ratio,
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("preflight","smoke","cell","aggregate"))
    parser.add_argument("--fold",type=int)
    args=parser.parse_args()
    if args.mode=="preflight":preflight()
    elif args.mode=="smoke":smoke()
    elif args.mode=="cell":
        if args.fold is None:raise ValueError("cell needs --fold")
        run_cell(args.fold)
    else:aggregate()


if __name__=="__main__":main()
