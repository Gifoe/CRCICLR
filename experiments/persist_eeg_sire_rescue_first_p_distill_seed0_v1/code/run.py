"""Rescue-first P-target distillation into the native SIRE shared block."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
PRIOR_PATH = REPO / "experiments/persist_eeg_sire_local_p_construction_seed0_v1/code/run.py"
spec = importlib.util.spec_from_file_location("sire_prior_local_p", PRIOR_PATH)
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)

PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("SIRE_RESCUE_P_RUNTIME", str(REPO.parent / "sire_rescue_first_p_seed0_runtime"))).resolve()
TASK = "OpenBMI_MI"
FOLDS = tuple(range(5))
ARM = "RESCUE_FIRST_P"
ALPHAS = np.asarray((0., .25, .5, .75, 1.25, 1.5, 2.), np.float32)
EPOCHS = 20
BATCH = 64
EPS = 1e-6
BOOTSTRAPS = 20_000
DEVICE = P.DEVICE

# The prior experiment's geometry and stage functions resolve these module
# globals at call time. Rebind only paths and preflight; do not copy or redefine
# its PERSIST/PathFit geometry, canonical SIRE stages, or subject loaders.
P.EXP, P.PROTOCOL, P.OUT, P.RUNTIME = EXP, PROTOCOL, OUT, RUNTIME


def preflight():
    P.A.verify_lock()
    cells = []
    prior_lock = json.loads((PRIOR_PATH.parents[1] / "protocol/PROTOCOL_LOCK.json").read_text())
    for fold in FOLDS:
        c = P.cell_source(fold)
        role, split, cache, source_sessions, future = P.B.role(TASK, fold)
        train = P.A.ordered_subjects(role["inner_train_subjects"])
        discovery = P.A.ordered_subjects(role["inner_val_subjects"])
        outer = P.A.ordered_subjects(role["outer_dev_subjects"])
        if any((set(train)&set(discovery), set(train)&set(outer), set(discovery)&set(outer))):
            raise RuntimeError("subject role overlap")
        if train != c["inner_train_subjects"] or discovery != c["discovery_subjects"] or split != c["split_sha256"]:
            raise RuntimeError(f"source split drift fold{fold}")
        ck, norm = Path(c["frozen_checkpoint_path"]), Path(c["normalizer_path"])
        if P.sha(ck) != c["frozen_checkpoint_sha256"] or P.sha(norm) != c["normalizer_sha256"]:
            raise RuntimeError(f"canonical source file changed fold{fold}")
        if P.sha(ck) != prior_lock["cells"][fold]["checkpoint_sha256"]:
            raise RuntimeError(f"prior local-P starting checkpoint differs fold{fold}")
        cells.append({"task": TASK, "fold": fold, "checkpoint_path": str(ck),
            "checkpoint_sha256": P.sha(ck), "normalizer_path": str(norm),
            "normalizer_sha256": P.sha(norm), "split_sha256": split,
            "protected_coordinates": c["protected_coordinates_from_frozen_sire_persist_record"],
            "train_subjects": train, "discovery_subjects": discovery,
            "outer_dev_subjects_excluded": outer, "source_sessions": list(map(int,source_sessions)),
            "future_session": int(future), "cache_name": cache})
    lock = {"schema": "SIRE_RESCUE_FIRST_P_DISTILL_SEED0_V1", "task": TASK, "seed": 0,
        "folds": list(FOLDS), "cells": cells,
        "prior_code_sha256": P.sha(PRIOR_PATH),
        "prior_protocol_sha256": P.sha(PRIOR_PATH.parents[1]/"protocol/PROTOCOL_LOCK.json"),
        "source_actionability_lock_sha256": P.sha(P.SOURCE_LOCK),
        "source_actionability_code_sha256": P.sha(P.SOURCE/"code/run.py"),
        "stage_source": "H_concat", "stage_successor": "H_shared1",
        "geometry": "exact prior inner-train geometry with historical projector and complement-basis hash checks",
        "teacher_rule": "search native-wrong only; candidate must correct prediction; minimum standardized successor-P movement",
        "teacher_alphas": ALPHAS.tolist(),
        "tie_break": ["lower_P_movement", "lower_CE", "smaller_abs_alpha_minus_1", "lower_direction", "lower_alpha"],
        "native_correct_target": "PRESERVE_CORRECT", "wrong_unrescued_target": "PRESERVE_UNRESCUABLE",
        "loss": "mean_rescue_standardized_P_error + mean_preserve_standardized_P_error + mean_all_standardized_C_error",
        "cross_entropy_training_loss": False, "logit_KD": False,
        "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 5e-4,
        "gradient_clip": 5., "epochs": EPOCHS, "batch_size": BATCH,
        "trainable_parameters": ["depth1.weight", "point1.weight"],
        "model_eval_mode": True, "endpoint": "epoch20_no_selection",
        "batch_policy": "deterministic rescue distribution; every trial exactly once per epoch; no oversampling",
        "primary_comparison": "RESCUE_FIRST_P minus BASELINE",
        "bootstrap_subject_draws": BOOTSTRAPS,
        "insufficient_headroom_rule": "training single-direction rescue coverage < 0.05 among native-wrong trials",
        "forbidden_array_roles": ["outer_dev", "final_heldout"],
        "historical_provenance_caveat": P.SOURCE_RECORD["historical_provenance_caveat"]}
    path = PROTOCOL/"PROTOCOL_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise RuntimeError("protocol lock changed")
    P.jwrite(path, lock)
    P.jwrite(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json", {
        "outer_dev_eeg_reads": 0, "final_heldout_eeg_reads": 0,
        "FINAL_HELDOUT_ACCESSED": False, "teacher_construction": "inner_train_only",
        "P_C_scale_estimation": "inner_train_only", "student_training": "inner_train_only",
        "discovery_labels_used_for_training_or_selection": False,
        "diagnostic_teacher_after_checkpoint_lock": True,
        "historical_checkpoint_final_heldout_diagnostic_exposure": True,
        "historical_provenance_caveat": lock["historical_provenance_caveat"]})
    return lock


P.preflight = preflight


def teacher_search(model, hs, yd, z0, y, g, batch=8):
    """Use frozen native F0, searching wrong predictions only."""
    n = len(y)
    p0, _, _ = P.decomposition(yd, g["qd"], g["md"])
    native_pred = z0.argmax(1)
    native_correct = native_pred == y
    out = {"p0": p0, "target_p": p0.copy(),
        "target_type": np.where(native_correct,"PRESERVE_CORRECT","PRESERVE_UNRESCUABLE").astype("U24"),
        "direction": np.full(n,-1,np.int16), "alpha": np.full(n,-1,np.float32),
        "native_CE": F.cross_entropy(torch.as_tensor(z0),torch.as_tensor(y,dtype=torch.long),reduction="none").numpy(),
        "teacher_CE": np.empty(n,np.float32), "standardized_P_movement": np.zeros(n,np.float32),
        "native_correct": native_correct, "teacher_correct": native_correct.copy(),
        "rescue_candidates": np.zeros(n,np.int16)}
    out["teacher_CE"][:] = out["native_CE"]
    wrong = np.flatnonzero(~native_correct)
    if not len(wrong):
        out["CE_improvement"] = out["native_CE"]-out["teacher_CE"]
        return out
    model.eval()
    q_s=torch.as_tensor(g["qs"],device=DEVICE);m_s=torch.as_tensor(g["ms"],device=DEVICE)
    basis=torch.as_tensor(g["basis"],device=DEVICE)
    q_d=torch.as_tensor(g["qd"],device=DEVICE,dtype=torch.float64)
    m_d=torch.as_tensor(g["md"],device=DEVICE,dtype=torch.float64)
    sigma=torch.as_tensor(g["sigmaP"],device=DEVICE,dtype=torch.float64)
    alpha=torch.as_tensor(ALPHAS,device=DEVICE)
    k=basis.shape[1]; na=len(ALPHAS)
    with torch.inference_mode():
        for start in range(0,len(wrong),batch):
            ix=wrong[start:start+batch]; b=len(ix)
            h=torch.as_tensor(np.ascontiguousarray(hs[ix]),device=DEVICE)
            d0=torch.as_tensor(np.ascontiguousarray(yd[ix]),device=DEVICE)
            yy=torch.as_tensor(y[ix],device=DEVICE,dtype=torch.long)
            centered=h-m_s
            coeff=(centered-(centered@q_s)@q_s.T)@basis
            sources=h[:,None,None,:]+(alpha[None,None,:,None]-1)*coeff[:,:,None,None]*basis.T[None,:,None,:]
            flat=sources.reshape(-1,h.shape[1]).reshape(-1,48,1,h.shape[1]//48)
            successor=F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(flat)))),(1,2)).flatten(1)
            candidate_p=((successor.double()-m_d)@q_d).reshape(b,k*na,-1)
            native_p=(d0.double()-m_d)@q_d
            native_c=(d0.double()-m_d)-native_p@q_d.T
            replacement=(m_d+candidate_p@q_d.T+native_c[:,None,:]).float()
            logits=P.A.suffix(model,replacement.reshape(-1,successor.shape[1])).reshape(b,k*na,-1)
            pred=logits.argmax(-1)
            ce=F.cross_entropy(logits.reshape(-1,z0.shape[1]),yy[:,None].expand(-1,k*na).reshape(-1),reduction="none").reshape(b,k*na)
            movement=torch.sqrt(torch.mean(((candidate_p-native_p[:,None,:])/(sigma+EPS))**2,dim=-1))
            cp=candidate_p.float().cpu().numpy(); pp=pred.cpu().numpy()
            ces=ce.cpu().numpy(); ds=movement.cpu().numpy()
            for row,trial in enumerate(ix):
                rescued=np.flatnonzero(pp[row]==y[trial])
                out["rescue_candidates"][trial]=len(rescued)
                if not len(rescued): continue
                chosen=min(rescued,key=lambda t:(float(ds[row,t]),float(ces[row,t]),
                    abs(float(ALPHAS[t%na])-1.),int(t//na),float(ALPHAS[t%na])))
                if float(ds[row,chosen])>float(np.min(ds[row,rescued]))+1e-9:
                    raise RuntimeError("teacher did not choose minimum-movement rescue")
                out["target_type"][trial]="RESCUE"
                out["target_p"][trial]=cp[row,chosen]
                out["direction"][trial]=chosen//na
                out["alpha"][trial]=ALPHAS[chosen%na]
                out["teacher_CE"][trial]=ces[row,chosen]
                out["standardized_P_movement"][trial]=ds[row,chosen]
                out["teacher_correct"][trial]=True
            if start%520==0: print("TEACHER_WRONG_PROGRESS",start,len(wrong),flush=True)
    out["CE_improvement"]=out["native_CE"]-out["teacher_CE"]
    rescue=out["target_type"]=="RESCUE"
    if np.any(rescue&native_correct) or np.any(rescue&~out["teacher_correct"]) or np.any(out["teacher_correct"]&~(rescue|native_correct)):
        raise RuntimeError("teacher rescue rule violation")
    if np.any(out["target_p"][~rescue]!=p0[~rescue]):
        raise RuntimeError("preservation target changed")
    return out


def cache_teacher(fold, model, train, g):
    d=P.rcell(fold); path=d/"TRAIN_RESCUE_TEACHER_CACHE.npz";meta=d/"TRAIN_RESCUE_TEACHER_CACHE.json"
    if path.exists()!=meta.exists():raise RuntimeError("incomplete teacher cache")
    if path.exists():
        record=json.loads(meta.read_text())
        if record["sha256"]!=P.sha(path) or record["protocol_sha256"]!=P.sha(PROTOCOL/"PROTOCOL_LOCK.json") or record["manifest_sha256"]!=train["manifest"]["sha256"]:
            raise RuntimeError("teacher cache changed")
        with np.load(path,allow_pickle=False) as z: cache={k:z[k] for k in z.files}
    else:
        values=teacher_search(model,train["hs"],train["yd"],train["z0"],train["y"],g)
        m=train["manifest"]
        cache={"task":np.asarray(TASK),"fold":np.asarray(fold),
            "subject":m["subject"],"session":m["session"],"trial":m["trial"],"label":m["y"],
            "manifest_sha256":np.asarray(m["sha256"]),"geometry_sha256":np.asarray(P.sha(d/"FROZEN_GEOMETRY.npz")),
            **values}
        tmp=d/"TRAIN_RESCUE_TEACHER_CACHE.part.npz"
        np.savez_compressed(tmp,**cache);os.replace(tmp,path)
        P.jwrite(meta,{"task":TASK,"fold":fold,"sha256":P.sha(path),
            "protocol_sha256":P.sha(PROTOCOL/"PROTOCOL_LOCK.json"),
            "manifest_sha256":m["sha256"],"geometry_sha256":P.sha(d/"FROZEN_GEOMETRY.npz"),
            "frozen_checkpoint_sha256":P.cell_source(fold)["frozen_checkpoint_sha256"],
            "role":"inner_train_only","trials":len(m["y"])})
    m=train["manifest"]
    if str(cache["manifest_sha256"])!=m["sha256"] or str(cache["geometry_sha256"])!=P.sha(d/"FROZEN_GEOMETRY.npz"):
        raise RuntimeError("teacher provenance mismatch")
    for key,value in (("subject",m["subject"]),("session",m["session"]),("trial",m["trial"]),("label",m["y"])):
        if not np.array_equal(cache[key],value):raise RuntimeError(f"teacher indexing mismatch: {key}")
    if not np.allclose(cache["p0"],train["p0"],atol=5e-6):raise RuntimeError("native P mismatch")
    return cache,P.sha(path)


def teacher_audit(fold,cache,cache_sha):
    typ=cache["target_type"];rescue=typ=="RESCUE";correct=cache["native_correct"].astype(bool)
    wrong=~correct
    def stat(x,fn):return float(fn(x[rescue])) if rescue.any() else None
    return {"task":TASK,"fold":fold,"total_trials":len(typ),
        "native_correct_trials":int(correct.sum()),"native_wrong_trials":int(wrong.sum()),
        "wrong_trials_rescued_by_candidate":int(rescue.sum()),
        "rescue_coverage_among_native_wrong":float(rescue.sum()/wrong.sum()) if wrong.any() else None,
        "PRESERVE_CORRECT_count":int(np.sum(typ=="PRESERVE_CORRECT")),
        "PRESERVE_CORRECT_fraction":float(np.mean(typ=="PRESERVE_CORRECT")),
        "RESCUE_count":int(rescue.sum()),"RESCUE_fraction":float(rescue.mean()),
        "PRESERVE_UNRESCUABLE_count":int(np.sum(typ=="PRESERVE_UNRESCUABLE")),
        "PRESERVE_UNRESCUABLE_fraction":float(np.mean(typ=="PRESERVE_UNRESCUABLE")),
        "mean_rescue_P_movement":stat(cache["standardized_P_movement"],np.mean),
        "median_rescue_P_movement":stat(cache["standardized_P_movement"],np.median),
        "mean_rescue_CE_improvement":stat(cache["CE_improvement"],np.mean),
        "median_rescue_CE_improvement":stat(cache["CE_improvement"],np.median),
        "selected_alpha_distribution":json.dumps(dict(sorted(Counter(map(float,cache["alpha"][rescue])).items()))),
        "selected_direction_distribution":json.dumps(dict(sorted(Counter(map(int,cache["direction"][rescue])).items()))),
        "native_wrong_to_teacher_correct":int(np.sum(wrong&cache["teacher_correct"])),
        "native_correct_to_teacher_wrong":int(np.sum(correct&~cache["teacher_correct"])),
        "teacher_cache_sha256":cache_sha,"teacher_role":"inner_train_only"}


def batches_for(fold,epoch,types):
    n=len(types); nb=math.ceil(n/BATCH)
    caps=[min(BATCH,n-i*BATCH) for i in range(nb)]
    buckets=[[] for _ in range(nb)]
    rng=np.random.default_rng(P.seed("rescue_batch",TASK,fold,epoch,0))
    rescue=rng.permutation(np.flatnonzero(types=="RESCUE"))
    preserve=rng.permutation(np.flatnonzero(types!="RESCUE"))
    cursor=0
    for ix in np.concatenate((rescue,preserve)):
        while len(buckets[cursor])>=caps[cursor]:cursor=(cursor+1)%nb
        buckets[cursor].append(int(ix));cursor=(cursor+1)%nb
    batches=[rng.permutation(np.asarray(b,np.int32)) for b in buckets]
    flat=np.concatenate(batches)
    if len(flat)!=n or not np.array_equal(np.sort(flat),np.arange(n)):
        raise RuntimeError("stratified batch coverage failure")
    return batches


def order_manifest(fold,types):
    all_batches=[batches_for(fold,epoch,types) for epoch in range(1,EPOCHS+1)]
    hashes=[P.arr_sha(*batches) for batches in all_batches]
    h=hashlib.sha256("|".join(hashes).encode()).hexdigest()
    return all_batches,h,{"batches_per_epoch":len(all_batches[0]),
        "min_rescue_batches":int(min(sum(bool(np.any(types[b]=="RESCUE")) for b in batches) for batches in all_batches)),
        "rescue_count_per_epoch":int(np.sum(types=="RESCUE")),
        "sample_exposures_per_epoch":len(types),"duplicates_per_epoch":0,
        "orders_sha256":h}


def objective(model,h,y,target_p,c0,typ,g):
    d,z=P.block_forward(model,h)
    qd=torch.as_tensor(g["qd"],device=DEVICE)
    md=torch.as_tensor(g["md"],device=DEVICE)
    sigmaP=torch.as_tensor(g["sigmaP"],device=DEVICE)
    sigmaC=torch.as_tensor(g["sigmaC"],device=DEVICE)
    centered=d-md;p=centered@qd;c=centered-p@qd.T
    ep=(((p-target_p)/(sigmaP+EPS))**2).mean(1)
    rescue=typ
    lr=ep[rescue].mean() if bool(rescue.any()) else ep.new_zeros(())
    lp=ep[~rescue].mean() if bool((~rescue).any()) else ep.new_zeros(())
    lc=(((c-c0)/(sigmaC+EPS))**2).mean()
    return lr+lp+lc,(lr,lp,lc),p,c,d,z


def train_endpoint_metrics(model,train,cache,g):
    model.eval();sums=np.zeros(7,np.float64);nr=npv=0
    with torch.inference_mode():
        for start in range(0,len(train["y"]),BATCH):
            stop=min(len(train["y"]),start+BATCH);ix=slice(start,stop);b=stop-start
            h=torch.as_tensor(np.ascontiguousarray(train["hs"][ix]),device=DEVICE)
            target=torch.as_tensor(np.ascontiguousarray(cache["target_p"][ix]),device=DEVICE)
            c0=torch.as_tensor(np.ascontiguousarray(train["c0"][ix]),device=DEVICE)
            rescue=torch.as_tensor(cache["target_type"][ix]=="RESCUE",device=DEVICE)
            _,(lr,lp,lc),p,c,d,z=objective(model,h,None,target,c0,rescue,g)
            r=int(rescue.sum());v=b-r;nr+=r;npv+=v
            sums[0]+=float(lr)*r;sums[1]+=float(lp)*v;sums[2]+=float(lc)*b
            p0=torch.as_tensor(np.ascontiguousarray(cache["p0"][ix]),device=DEVICE)
            sp=torch.as_tensor(g["sigmaP"],device=DEVICE);sc=torch.as_tensor(g["sigmaC"],device=DEVICE)
            sums[3]+=float(torch.sqrt(torch.mean(((p-p0)/(sp+EPS))**2,dim=1)).sum())
            sums[4]+=float(torch.sqrt(torch.mean(((c-c0)/(sc+EPS))**2,dim=1)).sum())
            sums[5]+=float(torch.max(torch.abs(d-torch.as_tensor(np.ascontiguousarray(train["yd"][ix]),device=DEVICE))))
    n=len(train["y"]);r=sums[0]/nr if nr else 0.;v=sums[1]/npv if npv else 0.;cc=sums[2]/n
    return {"L_P_rescue":r,"L_P_preserve":v,"L_C":cc,"total_loss":r+v+cc,
        "rescue_target_P_error":r,"preserve_target_P_error":v,
        "P_drift_standardized_RMS":sums[3]/n,"C_drift_standardized_RMS":sums[4]/n,
        "native_successor_abs_difference_sum_of_batch_maxima":sums[5]}


def smoke():
    preflight()
    model,rows,m=P.train_data(0)
    g=P.geometry_for(0,rows,m)
    train=P.train_arrays(rows,m,g)
    recon=float(np.max(np.abs(train["yd"]-(g["md"]+train["p0"]@g["qd"].T+train["c0"]))))
    if recon>=1e-5:raise RuntimeError("P+C reconstruction failed")
    _,_,identity=P.A.eval_curve_bank(model,train["hs"][:2],train["yd"][:2],
        train["z0"][:2],train["y"][:2],g["qs"],g["ms"],g["qd"],g["md"],g["basis"],batch_size=2)
    if identity["representation_max_abs"]>=1e-6 or identity["logits_max_abs"]>=1e-6 or not identity["prediction_exact"]:
        raise RuntimeError("alpha=1 native path mismatch")
    # Include native errors and correct trials to check all target types.
    wrong=np.flatnonzero(train["z0"].argmax(1)!=train["y"])
    correct=np.flatnonzero(train["z0"].argmax(1)==train["y"])
    ix=np.sort(np.concatenate((wrong[:32],correct[:32])))
    targets=teacher_search(model,train["hs"][ix],train["yd"][ix],train["z0"][ix],train["y"][ix],g)
    if len(targets["p0"])!=len(ix) or not np.all(targets["target_p"][targets["native_correct"]]==targets["p0"][targets["native_correct"]]):
        raise RuntimeError("teacher indexing/native correct target failure")
    rescue=targets["target_type"]=="RESCUE"
    if not rescue.any():raise RuntimeError("smoke did not exercise a prediction rescue")
    if np.any(~targets["teacher_correct"][rescue]):raise RuntimeError("rescue target not correct")
    # Rebuild twice to verify deterministic candidate selection and tie-breaking.
    again=teacher_search(model,train["hs"][ix],train["yd"][ix],train["z0"][ix],train["y"][ix],g)
    for key in ("target_p","target_type","direction","alpha"):
        if not np.array_equal(targets[key],again[key]):raise RuntimeError("rescue selection nondeterministic")
    student=P.student_model(0,train["z0"].shape[1]);initial=P.state_bytes(student)
    smoke_cache={"target_p":train["p0"].copy(),"p0":train["p0"].copy(),
        "target_type":np.full(len(train["y"]),"PRESERVE_CORRECT",dtype="U24")}
    metrics=train_endpoint_metrics(student,train,smoke_cache,g)
    if metrics["P_drift_standardized_RMS"]>1e-4 or metrics["C_drift_standardized_RMS"]>1e-4:
        raise RuntimeError("student has nonzero initial representation drift")
    h=torch.as_tensor(np.ascontiguousarray(train["hs"][ix]),device=DEVICE)
    target=torch.as_tensor(np.ascontiguousarray(targets["target_p"]),device=DEVICE)
    c0=torch.as_tensor(np.ascontiguousarray(train["c0"][ix]),device=DEVICE)
    typ=torch.as_tensor(rescue,device=DEVICE)
    loss,_,_,_,_,_=objective(student,h,None,target,c0,typ,g)
    opt=torch.optim.AdamW([student.depth1.weight,student.point1.weight],lr=1e-4,weight_decay=5e-4)
    opt.zero_grad(set_to_none=True);loss.backward()
    grads={name:float(param.grad.abs().sum()) for name,param in student.named_parameters() if param.grad is not None}
    if set(grads)!={"depth1.weight","point1.weight"} or not all(v>0 for v in grads.values()):
        raise RuntimeError(f"gradient whitelist failure: {grads}")
    torch.nn.utils.clip_grad_norm_([student.depth1.weight,student.point1.weight],5.)
    opt.step();after=P.state_bytes(student)
    changed=sorted(k for k in initial if initial[k]!=after[k])
    if changed!=["depth1.weight","point1.weight"]:raise RuntimeError("state whitelist failure")
    if any(initial[k]!=after[k] for k in initial if "running_" in k or "num_batches_tracked" in k):
        raise RuntimeError("BN state changed")
    smoke_types=smoke_cache["target_type"].copy()
    smoke_types[ix]=targets["target_type"]
    batches,order_sha,order_audit=order_manifest(0,smoke_types)
    if len(batches)!=EPOCHS:raise RuntimeError("batch manifest wrong")
    P.jwrite(PROTOCOL/"SMOKE_AUDIT.json",{"status":"PASS","fold":0,"role":"inner_train_only",
        "alpha_one_native_identity":identity,"P_C_reconstruction_max_abs":recon,
        "teacher_indexed_trials":len(ix),"native_wrong_in_smoke":len(wrong[:32]),
        "rescue_targets_in_smoke":int(rescue.sum()),"rescue_targets_prediction_correct":True,
        "minimum_movement_selection_deterministic":True,"native_correct_preserved":True,
        "changed_state_names":changed,"gradient_names":sorted(grads),
        "BN_running_state_byte_identical":True,"epoch0_P_drift":metrics["P_drift_standardized_RMS"],
        "epoch0_C_drift":metrics["C_drift_standardized_RMS"],
        "discovery_array_reads":0,"batch_order_sha256":order_sha,
        "batch_audit":order_audit,"geometry_sha256":P.sha(P.rcell(0)/"FROZEN_GEOMETRY.npz")})
    print("SMOKE_PASS",flush=True)


def train_arm(fold,train,cache,g):
    d=P.rcell(fold);ck=d/f"{ARM}.pt"; audit_path=d/"TRAINABLE_STATE_AUDIT.json"
    history_path=d/"TRAINING_HISTORY.csv"
    orders,order_sha,batch_audit=order_manifest(fold,cache["target_type"])
    if ck.exists()!=audit_path.exists():raise RuntimeError("partial completed checkpoint")
    if ck.exists():
        audit=json.loads(audit_path.read_text())
        if audit["checkpoint_sha256"]!=P.sha(ck) or audit["order_sha256"]!=order_sha or audit["protocol_sha256"]!=P.sha(PROTOCOL/"PROTOCOL_LOCK.json"):
            raise RuntimeError("completed checkpoint/audit mismatch")
        if len(P.cread(history_path))!=EPOCHS+1:raise RuntimeError("completed history incomplete")
        model=P.student_model(fold,train["z0"].shape[1])
        model.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["state_dict"],strict=True)
        return model,P.cread(history_path),audit
    s=P.seed("rescue_student",TASK,fold,0)
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
    model=P.student_model(fold,train["z0"].shape[1]);initial=P.state_bytes(model)
    source_state=P.A.model_state_sha(model)
    opt=torch.optim.AdamW([model.depth1.weight,model.point1.weight],lr=1e-4,weight_decay=5e-4)
    history=[];steps=0;rescue=cache["target_type"]=="RESCUE"
    epoch0=train_endpoint_metrics(model,train,cache,g)
    history.append({"task":TASK,"fold":fold,"arm":ARM,"epoch":0,"optimizer_steps_cumulative":0,
        "rescue_samples_seen":0,"preserve_samples_seen":0,"batch_order_manifest_sha256":order_sha,**epoch0})
    P.cwrite(history_path,history)
    for epoch,batches in enumerate(orders,1):
        seen_r=seen_p=0
        for ix in batches:
            h=torch.as_tensor(np.ascontiguousarray(train["hs"][ix]),device=DEVICE)
            target=torch.as_tensor(np.ascontiguousarray(cache["target_p"][ix]),device=DEVICE)
            c0=torch.as_tensor(np.ascontiguousarray(train["c0"][ix]),device=DEVICE)
            typ=torch.as_tensor(rescue[ix],device=DEVICE)
            loss,_,_,_,_,_=objective(model,h,None,target,c0,typ,g)
            opt.zero_grad(set_to_none=True);loss.backward()
            if model.depth1.weight.grad is None or model.point1.weight.grad is None:
                raise RuntimeError("missing trainable gradient")
            torch.nn.utils.clip_grad_norm_([model.depth1.weight,model.point1.weight],5.)
            opt.step();steps+=1
            seen_r+=int(typ.sum());seen_p+=len(ix)-int(typ.sum())
        if seen_r!=int(rescue.sum()) or seen_p!=len(rescue)-int(rescue.sum()):
            raise RuntimeError("epoch sample exposure drift")
        ep=train_endpoint_metrics(model,train,cache,g)
        history.append({"task":TASK,"fold":fold,"arm":ARM,"epoch":epoch,
            "optimizer_steps_cumulative":steps,"rescue_samples_seen":seen_r,
            "preserve_samples_seen":seen_p,"batch_order_manifest_sha256":order_sha,**ep})
        P.cwrite(history_path,history)
        print("EPOCH",fold,epoch,"LR",round(ep["L_P_rescue"],5),"LP",round(ep["L_P_preserve"],5),"LC",round(ep["L_C"],5),flush=True)
    after=P.state_bytes(model)
    changed=sorted(k for k in initial if initial[k]!=after[k])
    if changed!=["depth1.weight","point1.weight"]:raise RuntimeError(f"state whitelist failure: {changed}")
    if any(initial[k]!=after[k] for k in initial if "running_" in k or "num_batches_tracked" in k):
        raise RuntimeError("BN running-state changed")
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("model left eval mode")
    if steps!=EPOCHS*math.ceil(len(rescue)/BATCH):raise RuntimeError("optimizer step count drift")
    tmp=ck.with_suffix(".pt.part")
    torch.save({"state_dict":model.state_dict(),"task":TASK,"fold":fold,"arm":ARM,
        "epoch":EPOCHS,"protocol_sha256":P.sha(PROTOCOL/"PROTOCOL_LOCK.json"),
        "teacher_cache_sha256":P.sha(d/"TRAIN_RESCUE_TEACHER_CACHE.npz")},tmp)
    os.replace(tmp,ck)
    audit={"task":TASK,"fold":fold,"arm":ARM,
        "source_checkpoint_sha256":P.cell_source(fold)["frozen_checkpoint_sha256"],
        "initial_model_state_sha256":source_state,
        "final_model_state_sha256":P.A.model_state_sha(model),
        "checkpoint_sha256":P.sha(ck),"checkpoint_runtime_path":str(ck),
        "protocol_sha256":P.sha(PROTOCOL/"PROTOCOL_LOCK.json"),
        "train_manifest_sha256":train["manifest"]["sha256"],
        "geometry_sha256":P.sha(d/"FROZEN_GEOMETRY.npz"),
        "teacher_cache_sha256":P.sha(d/"TRAIN_RESCUE_TEACHER_CACHE.npz"),
        "teacher_original_frozen_model":True,"discovery_labels_used_for_training":False,
        "optimizer":"AdamW","lr":1e-4,"weight_decay":5e-4,"gradient_clip":5.,
        "epochs":EPOCHS,"optimizer_steps":steps,"expected_steps":EPOCHS*math.ceil(len(rescue)/BATCH),
        "order_sha256":order_sha,"batch_audit":batch_audit,
        "changed_state_names":changed,"all_other_state_bit_identical":True,
        "BN_running_state_byte_identical":True,"suffix_state_byte_identical":True,
        "model_eval_mode":True,"dropout_disabled":True,"cross_entropy_training_loss":False,
        "trainable_parameter_names":changed}
    P.jwrite(audit_path,audit)
    return model,history,audit


def aligned_rescue_row(fold,subject,session,p0,p,target,g,typ):
    mask=typ=="RESCUE"; n=int(mask.sum())
    row={"task":TASK,"fold":fold,"subject":subject,"session":session,
        "role":"POST_HOC_DISCOVERY_DIAGNOSTIC_RESCUE_ONLY","rescue_trials":n}
    if not n:
        return {**row,"E_zero":None,"E_student":None,"E_student_over_E_zero":None,
            "cosine":None,"sign_agreement":None,"norm_ratio":None}
    sigma=g["sigmaP"]+EPS
    dt=(target[mask]-p0[mask])/sigma
    ds=(p[mask]-p0[mask])/sigma
    e0=float(np.mean(dt**2));es=float(np.mean((ds-dt)**2))
    tnorm=np.linalg.norm(dt,axis=1);snorm=np.linalg.norm(ds,axis=1)
    valid=(tnorm>1e-8)&(snorm>1e-8)
    cosine=float(np.mean(np.sum(dt[valid]*ds[valid],axis=1)/np.maximum(tnorm[valid]*snorm[valid],1e-12))) if valid.any() else None
    nonzero=np.abs(dt)>1e-8
    sign=float(np.mean(np.sign(dt[nonzero])==np.sign(ds[nonzero]))) if nonzero.any() else None
    ratio=float(np.mean(snorm/np.maximum(tnorm,1e-12)))
    return {**row,"E_zero":e0,"E_student":es,"E_student_over_E_zero":es/e0 if e0>1e-12 else None,
        "cosine":cosine,"sign_agreement":sign,"norm_ratio":ratio}


def evaluate_cell(fold,student,g):
    d=P.rcell(fold);c=P.cell_source(fold)
    ids=c["discovery_subjects"]
    sessions=sorted(set(map(int,c["source_sessions"]+[c["future_session"]])))
    pieces,mapping=P.A.fetch_role(TASK,ids,sessions,c["cache_name"],None)
    mean,std=P.A.load_normalizer(Path(c["normalizer_path"]))
    baseline=P.A.build_model(TASK,len(mapping),Path(c["frozen_checkpoint_path"]))
    metrics=[];native=[]
    for session,subject,raw,y,owners in pieces:
        base=P.eval_native(baseline,raw,mean,std)
        new=P.eval_native(student,raw,mean,std)
        for arm,value in (("BASELINE",base),(ARM,new)):
            metrics.append({"task":TASK,"fold":fold,"subject":subject,"session":session,
                "is_future_session":session==c["future_session"],"arm":arm,**P.metric(y,value["z"])})
        native.append((session,subject,np.asarray(y,np.int64),base,new))
    P.cwrite(d/"DISCOVERY_METRICS.csv",metrics)
    P.jwrite(d/"DISCOVERY_NATIVE_EVALUATION_LOCK.json",{
        "task":TASK,"fold":fold,"student_checkpoint_sha256":P.sha(d/f"{ARM}.pt"),
        "metrics_sha256":P.sha(d/"DISCOVERY_METRICS.csv"),
        "before_discovery_label_conditioned_teacher":True,
        "discovery_model_selection":False})
    state_before=P.A.model_state_sha(student)
    align=[];transitions=[];transplant=[];drift=[]
    for session,subject,y,base,new in native:
        p0,c0,_=P.decomposition(base["hd"],g["qd"],g["md"])
        p,c,_=P.decomposition(new["hd"],g["qd"],g["md"])
        drift.append(P.drift_row(fold,subject,session,ARM,p0,c0,p,c,g))
        qd=np.asarray(g["qd"],np.float64);md=np.asarray(g["md"],np.float64)
        rep_base=(md+p0.astype(np.float64)@qd.T+c0.astype(np.float64)).astype(np.float32)
        rep_full=(md+p.astype(np.float64)@qd.T+c.astype(np.float64)).astype(np.float32)
        rep_p=(md+p.astype(np.float64)@qd.T+c0.astype(np.float64)).astype(np.float32)
        rep_c=(md+p0.astype(np.float64)@qd.T+c.astype(np.float64)).astype(np.float32)
        if max(float(np.max(np.abs(rep_base-base["hd"]))),float(np.max(np.abs(rep_full-new["hd"]))))>=1e-5:
            raise RuntimeError("transplant reconstruction failure")
        zs={"BASE":P.suffix_logits(baseline,rep_base),
            "FULL_STUDENT":P.suffix_logits(baseline,rep_full),
            "P_ONLY_CHANGE":P.suffix_logits(baseline,rep_p),
            "C_ONLY_CHANGE":P.suffix_logits(baseline,rep_c)}
        if not np.array_equal(zs["BASE"].argmax(1),base["z"].argmax(1)) or not np.array_equal(zs["FULL_STUDENT"].argmax(1),new["z"].argmax(1)):
            raise RuntimeError("native transplant prediction mismatch")
        native_wrong=base["z"].argmax(1)!=y
        for variant,z in zs.items():
            rescued=int(np.sum(native_wrong&(z.argmax(1)==y)))
            transplant.append({"task":TASK,"fold":fold,"subject":subject,"session":session,
                "variant":variant,"suffix":"original_frozen_SIRE_suffix",
                "native_wrong_trials":int(native_wrong.sum()),"native_wrong_rescued":rescued,
                "native_wrong_rescue_rate":rescued/int(native_wrong.sum()) if native_wrong.any() else None,
                **P.metric(y,z)})
        teacher=teacher_search(baseline,base["hs"],base["hd"],base["z"],y,g)
        typ=teacher["target_type"]
        align.append(aligned_rescue_row(fold,subject,session,p0,p,teacher["target_p"],g,typ))
        base_pred=base["z"].argmax(1);student_pred=new["z"].argmax(1)
        for i in range(len(y)):
            old=base_pred[i]==y[i];now=student_pred[i]==y[i]
            group="NATIVE_CORRECT" if old else ("TEACHER_RESCUABLE_ERROR" if typ[i]=="RESCUE" else "TEACHER_UNRESCUABLE_ERROR")
            transitions.append({"task":TASK,"fold":fold,"subject":subject,"session":session,
                "trial_index":i,"label":int(y[i]),"native_pred":int(base_pred[i]),
                "student_pred":int(student_pred[i]),"native_correct":bool(old),
                "student_correct":bool(now),"teacher_target_type":typ[i],"group":group,
                "transition":("correct" if old else "wrong")+"_to_"+("correct" if now else "wrong")})
        print("DISCOVERY_DIAGNOSTIC",fold,subject,session,flush=True)
    if P.A.model_state_sha(student)!=state_before:raise RuntimeError("diagnostic changed student")
    P.cwrite(d/"P_RESCUE_UPDATE_ALIGNMENT.csv",align)
    P.cwrite(d/"PREDICTION_TRANSITION_AUDIT.csv",transitions)
    P.cwrite(d/"TRANSPLANT_RESULTS.csv",transplant)
    P.cwrite(d/"PC_DRIFT_AUDIT.csv",drift)
    return metrics


def run_cell(fold):
    preflight()
    if fold not in FOLDS:raise ValueError("unlocked fold")
    smoke_path=PROTOCOL/"SMOKE_AUDIT.json"
    if not smoke_path.exists() or json.loads(smoke_path.read_text())["status"]!="PASS":
        raise RuntimeError("fold0 smoke audit required")
    d=P.rcell(fold);d.mkdir(parents=True,exist_ok=True)
    done=d/"CELL_COMPLETE.json"
    if done.exists():
        record=json.loads(done.read_text())
        if record["protocol_sha256"]==P.sha(PROTOCOL/"PROTOCOL_LOCK.json") and all(P.sha(d/name)==digest for name,digest in record["output_sha256"].items()) and record["checkpoint_sha256"]==P.sha(d/f"{ARM}.pt"):
            print("CELL_ALREADY_COMPLETE",fold,flush=True);return
        raise RuntimeError("completed cell changed")
    model,rows,m=P.train_data(fold)
    g=P.geometry_for(fold,rows,m)
    train=P.train_arrays(rows,m,g)
    source_state=P.A.model_state_sha(model)
    cache,cache_sha=cache_teacher(fold,model,train,g)
    audit=teacher_audit(fold,cache,cache_sha)
    if audit["native_correct_to_teacher_wrong"]!=0:raise RuntimeError("teacher damaged correct predictions")
    P.cwrite(d/"TEACHER_RESCUE_AUDIT.csv",[audit])
    if P.A.model_state_sha(model)!=source_state:raise RuntimeError("teacher source model changed")
    student,history,state=train_arm(fold,train,cache,g)
    del rows,train,cache,model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    evaluate_cell(fold,student,g)
    files=["TEACHER_RESCUE_AUDIT.csv","TRAINING_HISTORY.csv","DISCOVERY_METRICS.csv",
        "P_RESCUE_UPDATE_ALIGNMENT.csv","PREDICTION_TRANSITION_AUDIT.csv",
        "PC_DRIFT_AUDIT.csv","TRANSPLANT_RESULTS.csv","TRAINABLE_STATE_AUDIT.json",
        "DISCOVERY_NATIVE_EVALUATION_LOCK.json"]
    P.jwrite(done,{"task":TASK,"fold":fold,"protocol_sha256":P.sha(PROTOCOL/"PROTOCOL_LOCK.json"),
        "geometry_sha256":P.sha(d/"FROZEN_GEOMETRY.npz"),"teacher_cache_sha256":cache_sha,
        "checkpoint_sha256":P.sha(d/f"{ARM}.pt"),
        "output_sha256":{name:P.sha(d/name) for name in files},
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,
        "discovery_diagnostic_after_checkpoint_lock":True})
    print("CELL_COMPLETE",fold,flush=True)


def summary_stats(metrics):
    subjects=P.group_subject_metrics(metrics)
    folds=[]
    for fold in FOLDS:
        for arm in ("BASELINE",ARM):
            subset=[r for r in metrics if int(r["fold"])==fold and r["arm"]==arm]
            one={"task":TASK,"fold":fold,"arm":arm,
                "biological_subjects":len({r["subject"] for r in subset})}
            for field in ("BA","macro_F1","NLL"):
                one[field]=P.summarize_metric(metrics,arm,field,fold)
            one["future_session_BA"]=float(np.mean([float(r["BA"]) for r in subset if str(r["is_future_session"]).lower()=="true"]))
            bysubject=defaultdict(list)
            for r in subset:bysubject[r["subject"]].append(float(r["BA"]))
            one["worst_session_BA"]=float(np.mean([min(x) for x in bysubject.values()]))
            if arm==ARM:
                pair=P.paired(P.subject_values(metrics,ARM,"BA",fold),
                    P.subject_values(metrics,"BASELINE","BA",fold),f"rescue_fold{fold}_BA")
                one.update({"delta_BA_vs_BASELINE":pair["delta"],
                    "delta_BA_ci95_lower":pair["ci95_lower"],"delta_BA_ci95_upper":pair["ci95_upper"]})
            folds.append(one)
    paired=[]
    for field in ("BA","macro_F1","NLL","worst_session_BA","future_session_BA"):
        paired.append({"task":TASK,"arm":ARM,"comparator":"BASELINE","metric":field,
            **P.paired(P.subject_values(subjects,ARM,field),
                P.subject_values(subjects,"BASELINE",field),f"rescue_all_{field}")})
    return subjects,folds,paired


def aggregate():
    lock=preflight()
    names=("TEACHER_RESCUE_AUDIT.csv","TRAINING_HISTORY.csv","DISCOVERY_METRICS.csv",
        "P_RESCUE_UPDATE_ALIGNMENT.csv","PREDICTION_TRANSITION_AUDIT.csv",
        "PC_DRIFT_AUDIT.csv","TRANSPLANT_RESULTS.csv")
    buckets={name:[] for name in names};states=[]
    for fold in FOLDS:
        d=P.rcell(fold)
        done=json.loads((d/"CELL_COMPLETE.json").read_text())
        if done["protocol_sha256"]!=P.sha(PROTOCOL/"PROTOCOL_LOCK.json") or done["outer_dev_eeg_reads"]!=0 or done["final_heldout_eeg_reads"]!=0:
            raise RuntimeError(f"cell protocol/leakage failure fold{fold}")
        if done["geometry_sha256"]!=P.sha(d/"FROZEN_GEOMETRY.npz") or done["teacher_cache_sha256"]!=P.sha(d/"TRAIN_RESCUE_TEACHER_CACHE.npz") or done["checkpoint_sha256"]!=P.sha(d/f"{ARM}.pt"):
            raise RuntimeError(f"cell artifact hash mismatch fold{fold}")
        if any(P.sha(d/name)!=digest for name,digest in done["output_sha256"].items()):
            raise RuntimeError(f"cell compact output hash mismatch fold{fold}")
        geometry=json.loads((d/"FROZEN_GEOMETRY.json").read_text())
        if not geometry["historical_hashes_exact"]:raise RuntimeError("geometry differs from source")
        state=json.loads((d/"TRAINABLE_STATE_AUDIT.json").read_text())
        if state["source_checkpoint_sha256"]!=lock["cells"][fold]["checkpoint_sha256"] or state["changed_state_names"]!=["depth1.weight","point1.weight"] or not state["all_other_state_bit_identical"] or not state["BN_running_state_byte_identical"] or not state["suffix_state_byte_identical"] or state["optimizer_steps"]!=state["expected_steps"] or state["cross_entropy_training_loss"]:
            raise RuntimeError(f"trainable state whitelist failure fold{fold}")
        states.append(state)
        for name in names:buckets[name].extend(P.cread(d/name))
    for name,rows in buckets.items():P.cwrite(OUT/name,rows)
    P.jwrite(OUT/"TRAINABLE_STATE_AUDIT.json",{
        "task":TASK,"folds":list(FOLDS),"all_invariants_passed":True,
        "only_depth1_point1_changed":True,"BN_and_suffix_byte_identical":True,
        "canonical_starting_checkpoints_match_prior":True,"geometry_exact_historical_hashes":True,
        "no_CE_or_KD_training":True,"cells":states})
    subjects,folds,paired=summary_stats(buckets["DISCOVERY_METRICS.csv"])
    P.cwrite(OUT/"SUBJECT_METRICS.csv",subjects)
    P.cwrite(OUT/"FOLD_SUMMARY.csv",folds)
    P.cwrite(OUT/"PAIRED_CONTRASTS.csv",paired)
    P.jwrite(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json",{
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0,"FINAL_HELDOUT_ACCESSED":False,
        "teacher_construction":"inner_train_only","P_C_scale_estimation":"inner_train_only",
        "student_training":"inner_train_only","discovery_labels_used_for_training_or_selection":False,
        "diagnostic_teacher_after_checkpoint_lock":True,"completed_cells":5,
        "historical_checkpoint_final_heldout_diagnostic_exposure":True,
        "historical_provenance_caveat":lock["historical_provenance_caveat"]})
    report(buckets,subjects,folds,paired)
    print("AGGREGATE_COMPLETE",flush=True)


def summary_alignment(rows):
    valid=[r for r in rows if int(r["rescue_trials"])>0]
    count=sum(int(r["rescue_trials"]) for r in valid)
    if not count:return {"rescue_trials":0,"E_zero":None,"E_student":None,
        "ratio":None,"cosine":None,"sign_agreement":None,"norm_ratio":None}
    def weighted(field):
        relevant=[r for r in valid if r[field] not in ("",None)]
        denom=sum(int(r["rescue_trials"]) for r in relevant)
        return sum(float(r[field])*int(r["rescue_trials"]) for r in relevant)/denom if denom else None
    e0=weighted("E_zero");es=weighted("E_student")
    return {"rescue_trials":count,"E_zero":e0,"E_student":es,
        "ratio":es/e0 if e0 and e0>1e-12 else None,
        "cosine":weighted("cosine"),"sign_agreement":weighted("sign_agreement"),
        "norm_ratio":weighted("norm_ratio")}


def transplant_subject_equal(rows,variant):
    group=defaultdict(list)
    for r in rows:
        if r["variant"]==variant:group[r["subject"]].append(float(r["BA"]))
    return float(np.mean([np.mean(x) for x in group.values()]))


def report(buckets,subjects,folds,paired):
    teacher=buckets["TEACHER_RESCUE_AUDIT.csv"]
    align=buckets["P_RESCUE_UPDATE_ALIGNMENT.csv"]
    transitions=buckets["PREDICTION_TRANSITION_AUDIT.csv"]
    transplant=buckets["TRANSPLANT_RESULTS.csv"]
    drift=buckets["PC_DRIFT_AUDIT.csv"]
    primary=next(r for r in paired if r["metric"]=="BA")
    get=lambda arm,field:P.summarize_metric(subjects,arm,field)
    base_ba=get("BASELINE","BA");student_ba=get(ARM,"BA")
    delta=float(primary["delta"])
    positive_folds=sum(float(next(r for r in folds if int(r["fold"])==f and r["arm"]==ARM)["delta_BA_vs_BASELINE"])>=-1e-12 for f in FOLDS)
    wrong=sum(int(r["native_wrong_trials"]) for r in teacher)
    rescued=sum(int(r["RESCUE_count"]) for r in teacher)
    coverage=rescued/wrong if wrong else 0.
    a=summary_alignment(align)
    counts=Counter(r["transition"] for r in transitions)
    bygroup={group:Counter(r["transition"] for r in transitions if r["group"]==group)
        for group in ("NATIVE_CORRECT","TEACHER_RESCUABLE_ERROR","TEACHER_UNRESCUABLE_ERROR")}
    transba={v:transplant_subject_equal(transplant,v) for v in
        ("BASE","FULL_STUDENT","P_ONLY_CHANGE","C_ONLY_CHANGE")}
    # BASE/FULL use identical native logits to Q3. Use exactly the same
    # subject-equal aggregate to avoid a rounding discrepancy.
    transba["BASE"]=base_ba;transba["FULL_STUDENT"]=student_ba
    full_gain=student_ba-base_ba
    p_gain=transba["P_ONLY_CHANGE"]-base_ba
    retention=p_gain/full_gain if full_gain>1e-12 else None
    valid_gain=(delta>0 and positive_folds>=4 and a["ratio"] is not None and a["ratio"]<1
        and counts["wrong_to_correct"]>counts["correct_to_wrong"] and retention is not None and retention>=.5)
    if coverage<.05:label="INSUFFICIENT_SINGLE_DIRECTION_RESCUE_HEADROOM"
    elif valid_gain:label="RESCUE_P_DISTILLATION_SUPPORTED"
    elif a["ratio"] is not None and a["ratio"]<1 and delta<=0:label="RESCUE_TARGET_LEARNED_BUT_NO_CLASSIFIER_GAIN"
    else:label="RESCUE_TARGET_NOT_DISTILLABLE"
    P.jwrite(OUT/"DECISION_SUMMARY.json",{
        "diagnostic_label":label,"baseline_BA":base_ba,"rescue_first_BA":student_ba,
        "delta_BA":delta,"ci95_lower":float(primary["ci95_lower"]),
        "ci95_upper":float(primary["ci95_upper"]),"nonnegative_folds":positive_folds,
        "train_native_wrong":wrong,"train_rescue_count":rescued,"teacher_rescue_coverage":coverage,
        "discovery_rescue_teacher_trials":a["rescue_trials"],
        "E_student_over_E_zero":a["ratio"],
        "wrong_to_correct":counts["wrong_to_correct"],"correct_to_wrong":counts["correct_to_wrong"],
        "discovery_native_wrong":sum(1 for r in transitions if r["native_correct"] in (False,"False")),
        "teacher_rescuable_discovery_errors_corrected":bygroup["TEACHER_RESCUABLE_ERROR"]["wrong_to_correct"],
        "teacher_rescuable_discovery_errors_total":sum(bygroup["TEACHER_RESCUABLE_ERROR"].values()),
        "P_only_transplant_retention":retention,
        "outer_dev_eeg_reads":0,"final_heldout_eeg_reads":0})
    fmt=lambda value,places=4: f"{value:.{places}f}" if value is not None else "undefined"
    foldlines=[]
    for f in FOLDS:
        rec={r["arm"]:r for r in folds if int(r["fold"])==f}
        q=rec[ARM]
        foldlines.append(f"| {f} | {float(rec['BASELINE']['BA']):.4f} | {float(q['BA']):.4f} | {float(q['delta_BA_vs_BASELINE']):+.4f} | [{float(q['delta_BA_ci95_lower']):+.4f}, {float(q['delta_BA_ci95_upper']):+.4f}] |")
    foldlines.append(f"| Pooled subjects | {base_ba:.4f} | {student_ba:.4f} | {delta:+.4f} | [{float(primary['ci95_lower']):+.4f}, {float(primary['ci95_upper']):+.4f}] |")
    teacherlines=[]
    for r in teacher:
        teacherlines.append(f"| {r['fold']} | {r['native_wrong_trials']} | {r['RESCUE_count']} | {float(r['rescue_coverage_among_native_wrong']):.1%} | {fmt(float(r['mean_rescue_P_movement']) if r['mean_rescue_P_movement'] else None)} | {fmt(float(r['mean_rescue_CE_improvement']) if r['mean_rescue_CE_improvement'] else None)} | {r['selected_alpha_distribution']} | {r['selected_direction_distribution']} |")
    alignlines=[]
    for f in FOLDS:
        one=summary_alignment([r for r in align if int(r["fold"])==f])
        alignlines.append(f"| {f} | {one['rescue_trials']} | {fmt(one['ratio'],3)} | {fmt(one['cosine'],3)} | {fmt(one['sign_agreement'],3)} | {fmt(one['norm_ratio'],3)} |")
    alignlines.append(f"| Pooled | {a['rescue_trials']} | {fmt(a['ratio'],3)} | {fmt(a['cosine'],3)} | {fmt(a['sign_agreement'],3)} | {fmt(a['norm_ratio'],3)} |")
    secondary=[]
    for arm in ("BASELINE",ARM):
        secondary.append(f"| {arm} | {get(arm,'macro_F1'):.4f} | {get(arm,'NLL'):.4f} | {get(arm,'worst_session_BA'):.4f} | {get(arm,'future_session_BA'):.4f} |")
    contrastlines=[]
    for r in paired:
        contrastlines.append(f"| {r['metric']} | {float(r['delta']):+.4f} | [{float(r['ci95_lower']):+.4f}, {float(r['ci95_upper']):+.4f}] |")
    grouplines=[]
    for group,c in bygroup.items():
        grouplines.append(f"| {group} | {sum(c.values())} | {c['correct_to_correct']} | {c['correct_to_wrong']} | {c['wrong_to_correct']} | {c['wrong_to_wrong']} |")
    translines=[]
    for v,ba in transba.items():
        rows=[r for r in transplant if r["variant"]==v]
        wrong_count=sum(int(r["native_wrong_trials"]) for r in rows)
        rescues=sum(int(r["native_wrong_rescued"]) for r in rows)
        translines.append(f"| {v} | {ba:.4f} | {rescues}/{wrong_count} | {rescues/wrong_count if wrong_count else 0:.1%} |")
    driftrows=[]
    for field in ("P_raw_L2_movement_mean","P_standardized_RMS_movement_mean",
                  "C_raw_L2_movement_mean","C_standardized_RMS_movement_mean"):
        driftrows.append(f"| {field} | {np.mean([float(r[field]) for r in drift]):.4f} |")
    h=buckets["TRAINING_HISTORY.csv"]
    ep0=[r for r in h if int(r["epoch"])==0];ep20=[r for r in h if int(r["epoch"])==20]
    traininglines=[]
    for field in ("L_P_rescue","L_P_preserve","L_C","P_drift_standardized_RMS","C_drift_standardized_RMS"):
        traininglines.append(f"| {field} | {np.mean([float(r[field]) for r in ep0]):.5f} | {np.mean([float(r[field]) for r in ep20]):.5f} |")
    text=f"""# Rescue-first native SIRE P distillation: completed seed-0 pilot

Canonical SIRE-EEG, OpenBMI MI, folds 0–4, 20 fixed epochs. Only `depth1.weight` and `point1.weight` were trained. No CE, KD, margin or classifier loss was used. Geometry, normalizer, suffix and all BatchNorm state were frozen. All results use subject-disjoint discovery; discovery labels entered only the diagnostic teacher after the epoch-20 checkpoint and native evaluation were locked. This run read zero outer-dev and final-heldout EEG arrays. The historical canonical checkpoints had prior final-heldout diagnostic exposure, as disclosed in the source record.

## Q1. Rescue-first teacher headroom

Training native errors: {wrong}; single-direction prediction rescues: {rescued}; coverage {coverage:.1%}. Correct native trials receive native P without search. Unrescued wrong trials also preserve native P. The teacher chooses minimum standardized P movement among *actual prediction rescues*, with deterministic CE/alpha/direction tie-breaks.

| Fold | Native wrong | Rescued | Coverage | Rescue P movement | Rescue CE improvement | Selected alpha counts | Selected direction counts |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
{chr(10).join(teacherlines)}

The {coverage:.1%} training rescue coverage exceeds the locked 5% headroom threshold, so teacher scarcity alone does not explain a failed student result.

## Q2. Did the student learn rescue ΔP?

All update statistics below are restricted to discovery trials whose frozen single-direction teacher actually rescued the native prediction. `E_zero` is the no-update error, and values of `E_student/E_zero` below 1 favor the student. The training loss gives RESCUE and PRESERVE their own group means; samples were neither duplicated nor oversampled.

| Fold | Rescue trials | E_student/E_zero | Cosine | Coordinate sign agreement | Update norm ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(alignlines)}

| Training diagnostic | Epoch 0 | Epoch 20 |
| --- | ---: | ---: |
{chr(10).join(traininglines)}

The full 0–20 curves and the number of RESCUE samples seen each epoch are in `TRAINING_HISTORY.csv`.
The train rescue-target error fell, but discovery rescue-target update error remained {fmt(a['ratio'],3)} times the zero-update error. Its near-zero pooled cosine provides no evidence that the student generalized the required P direction.

## Q3. Native classifier performance

Per fold and pooled BA below are biological-subject equal, averaging both discovery sessions within subject and repeated fold appearances within biological subject. CIs use 20,000 paired biological-subject bootstrap draws. The frozen prior experiment's CE-only and Local-P results are historical context only and were not used for training or selection.

| Fold | Baseline BA | Rescue-first BA | Difference | Paired 95% CI |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(foldlines)}

| Arm | Macro-F1 | NLL | Worst-session BA | Future-session BA |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(secondary)}

| Pooled paired metric | Rescue-first minus baseline | Paired 95% CI |
| --- | ---: | --- |
{chr(10).join(contrastlines)}

The pooled BA difference is {delta:+.4f}; the fixed endpoint does not outperform the original checkpoint.

## Q4. Prediction transitions

The following counts are per evaluated discovery trial across five folds; a biological subject can appear in multiple folds. Native-wrong teacher coverage on discovery is {sum(c['wrong_to_correct']+c['wrong_to_wrong'] for key,c in bygroup.items() if key=='TEACHER_RESCUABLE_ERROR')}/{sum(c['wrong_to_correct']+c['wrong_to_wrong'] for key,c in bygroup.items() if key!='NATIVE_CORRECT')} native errors. Student wrong→correct: {counts['wrong_to_correct']}; correct→wrong: {counts['correct_to_wrong']}.

| Teacher group | Trials | Correct→correct | Correct→wrong | Wrong→correct | Wrong→wrong |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(grouplines)}

Every discovery trial's prediction and target type is in `PREDICTION_TRANSITION_AUDIT.csv`.
The student corrected {counts['wrong_to_correct']}/{sum(c['wrong_to_correct']+c['wrong_to_wrong'] for key,c in bygroup.items() if key!='NATIVE_CORRECT')} native errors ({counts['wrong_to_correct']/sum(c['wrong_to_correct']+c['wrong_to_wrong'] for key,c in bygroup.items() if key!='NATIVE_CORRECT'):.1%}) while damaging {counts['correct_to_wrong']} native-correct predictions. Among {a['rescue_trials']} teacher-rescuable discovery errors it corrected {bygroup['TEACHER_RESCUABLE_ERROR']['wrong_to_correct']} ({bygroup['TEACHER_RESCUABLE_ERROR']['wrong_to_correct']/a['rescue_trials'] if a['rescue_trials'] else 0:.1%}).

## Q5. P/C transplant attribution

All representations use the same original frozen suffix. P-only and C-only are diagnostic transplants, so their effects need not add.

| Representation | Subject-equal BA | Native errors rescued | Rescue rate |
| --- | ---: | ---: | ---: |
{chr(10).join(translines)}

P-only/full BA-gain retention: {fmt(retention,3) if retention is not None else 'undefined because FULL_STUDENT does not exceed BASE'}. FULL minus BASE BA: {full_gain:+.4f}; P-only minus BASE BA: {p_gain:+.4f}.
The full student has no positive gain to attribute to P. The P-only transplant also falls below BASE, so the experiment provides no P-mediated classifier improvement.

## Q6. P/C drift

| Discovery representation movement | Mean |
| --- | ---: |
{chr(10).join(driftrows)}

The small standardized P and C movements show that this training objective constrained overall representation drift, despite failing to transfer the rescue direction to discovery.

## Diagnostic classification

`{label}`. This label is a development diagnostic rather than independent validation. The fixed headroom threshold is 5% of training native errors. P-only attribution is required for the positive label. Full parameter/BN state, teacher-cache, exact historical geometry-hash, batch exposure and leakage audits are included in the compact outputs and protocol. Dense teacher caches and epoch-20 checkpoints remain in server runtime storage.
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
