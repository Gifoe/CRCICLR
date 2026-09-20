"""Frozen native-head audit of current PERSIST Protected coordinates.

This program is deliberately fail-closed.  It never trains a backbone, refits
a probe, or opens the final/true-heldout cohort.  Canonical geometry and P are
reconstructed only from the PERSIST TRAIN data; outcomes are evaluated only
on each frozen fold's outer-development subjects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("NATIVE_PROTECTED_RUNTIME", str(ROOT.parent / "native_protected_utilization_runtime")))
SEVEN_REPO = Path(os.environ.get("SEVEN_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK"))
SEVEN_CODE = SEVEN_REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"
SEVEN_RUNTIME = Path(os.environ.get("SEVEN_RUNTIME", r"D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime"))
PEEH_EXTENSION = Path(os.environ.get("PEEH_EXTENSION_ROOT", str(ROOT / "experiments" / "persist_eeg_eegconformer_fbcnet_peeh_pswa_v1")))
PEEH_EEGNET = Path(os.environ.get("PEEH_EEGNET_ROOT", str(ROOT / "experiments" / "persist_eeg_crossbackbone_peeh_v1")))
PEEH_EXT_RUNTIME = Path(os.environ.get("PEEH_EXTENSION_RUNTIME", r"D:\nips-temp\TotalP\P1\eegconformer_fbcnet_peeh_runtime"))
PEEH_EEGNET_RUNTIME = Path(os.environ.get("PEEH_EEGNET_RUNTIME", r"D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime"))

MODELS = ("EEGNet", "EEGConformer", "FBCNet")
TASKS = ("OpenBMI_MI", "OpenBMI_SSVEP")
FOLDS = tuple(range(5))
SEED = 0
RANDOM_DRAWS = 100
CAP = 32
EPS = 1e-12


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows([{k: clean(row.get(k, "")) for k in fields} for row in rows])
    os.replace(tmp, path)


EXT = import_file("native_current_peeh_extension", PEEH_EXTENSION / "code" / "run_eegconformer_fbcnet_peeh.py")
EEN = import_file("native_current_peeh_eegnet", PEEH_EEGNET / "code" / "run_crossbackbone_peeh.py")
BENCH = import_file("native_outer_development_benchmark", SEVEN_CODE / "benchmark_data.py")
INNER = BENCH._load_module("native_outer_development_inner", SEVEN_CODE / "tech_recipe_selection.py")


def helper(model: str) -> Any:
    return EEN if model == "EEGNet" else EXT


def native_cell_path(model: str, task: str, fold: int) -> Path:
    root = PEEH_EEGNET_RUNTIME if model == "EEGNet" else PEEH_EXT_RUNTIME
    return root / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def target_path(model: str, task: str, fold: int) -> Path:
    return RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def natural(values: Iterable[object]) -> list[str]:
    return EXT.natural_subjects(values)


def outer_data(task: str, fold: int) -> dict[str, Any]:
    """Load only frozen TRAIN and outer-development rows; never final heldout."""
    modern, taskmod = BENCH._sources()
    cache = EXT.OPENBMI_CACHE
    if task == "OpenBMI_MI":
        folds, _, split_hash = modern.load_split()
        role = next(x for x in folds["OpenBMI"] if int(x["fold_id"]) == fold)
        source_session, future_session, cache_name = tuple(modern.SOURCE_SESSIONS["OpenBMI"]), int(modern.EVAL_SESSION), "mi"
    elif task == "OpenBMI_SSVEP":
        _, _, reference, split_hash = taskmod.split_reference()
        role = next(x for x in reference["folds"] if int(x["fold_id"]) == fold)
        spec = taskmod.TASKS["SSVEP"]
        source_session, future_session, cache_name = (int(spec["source_session"]),), int(spec["future_session"]), spec["cache_name"]
    else:
        raise KeyError(task)
    train = list(map(str, role["inner_train_subjects"])); outer = list(map(str, role["outer_dev_subjects"]))
    if set(train) & set(outer) or set(outer) & set(map(str, role.get("inner_val_subjects", []))):
        raise RuntimeError("outer-development subject split overlaps TRAIN or checkpoint-selection subjects")
    sx, sy, ss, mapping = INNER._openbmi_rows(cache, train, source_session, cache_name)
    fx, fy, fs, _ = INNER._openbmi_rows(cache, train, (future_session,), cache_name, mapping)
    ox1, oy1, os1, _ = INNER._openbmi_rows(cache, outer, source_session, cache_name, mapping)
    ox2, oy2, os2, _ = INNER._openbmi_rows(cache, outer, (future_session,), cache_name, mapping)
    sx, values, norm = BENCH._normalise(sx, fx, ox1, ox2)
    fx, ox1, ox2 = values
    if not len(outer) or not len(ox2): raise RuntimeError("outer-development evaluation is unavailable")
    return {"train_x": sx, "train_y": sy.astype(np.int64), "train_subjects": ss.astype(str),
            "future_train_x": fx, "future_train_y": fy.astype(np.int64), "future_train_subjects": fs.astype(str),
            "outer_source_x": ox1, "outer_source_y": oy1.astype(np.int64), "outer_source_subjects": os1.astype(str),
            "outer_future_x": ox2, "outer_future_y": oy2.astype(np.int64), "outer_future_subjects": os2.astype(str),
            "train_subject_ids": train, "outer_subject_ids": outer, "source_session": int(source_session[0]),
            "future_session": future_session, "split_sha256": split_hash, "normalizer": norm,
            "classes": int(max(np.max(sy), np.max(fy), np.max(oy2)) + 1)}


def capped_train(data: dict[str, Any], task: str, model: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = helper(model)
    a = h.capped_indices(data["train_subjects"], data["train_y"], data["source_session"], task, fold, "train-source")
    b = h.capped_indices(data["future_train_subjects"], data["future_train_y"], data["future_session"], task, fold, "train-future")
    x = np.concatenate([data["train_x"][a], data["future_train_x"][b]])
    y = np.concatenate([data["train_y"][a], data["future_train_y"][b]])
    owner = np.concatenate([data["train_subjects"][a], data["future_train_subjects"][b]]).astype(str)
    sessions = np.concatenate([np.full(len(a), data["source_session"]), np.full(len(b), data["future_session"])]).astype(np.int64)
    return x, y, owner, sessions


def output_logits(value: Any) -> torch.Tensor:
    if isinstance(value, (tuple, list)): value = value[0]
    if not isinstance(value, torch.Tensor): raise TypeError("model forward did not return logits")
    return value


def hook_representations(model: torch.nn.Module, head: torch.nn.Module, x: np.ndarray, model_name: str, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capture immediately before the frozen native head and replay it exactly."""
    parts: list[np.ndarray] = []; direct: list[np.ndarray] = []; native: list[np.ndarray] = []; captured: list[torch.Tensor] = []
    fb = import_file("native_fixed_filterbank", EXT.BASELINE_CODE / "filterbank.py") if model_name == "FBCNet" else None
    handle = head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    try:
        with torch.inference_mode():
            for start in range(0, len(x), 32):
                value = np.ascontiguousarray(x[start:start + 32], dtype=np.float32)
                if fb is not None: value = fb.transform(value)
                captured.clear(); z = output_logits(model(torch.from_numpy(value).to(device)))
                if len(captured) != 1: raise RuntimeError("classifier-input hook did not fire exactly once")
                raw = captured[0]; flat = raw.reshape(raw.shape[0], -1).float()
                zh = head(flat)
                if not torch.allclose(z, zh, rtol=1e-5, atol=1e-6):
                    raise RuntimeError("native-head replay logits mismatch")
                parts.append(flat.cpu().numpy()); direct.append(z.float().cpu().numpy()); native.append(zh.float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(parts).astype(np.float32), np.concatenate(direct), np.concatenate(native)


def head_logits(head: torch.nn.Module, h: np.ndarray, device: torch.device) -> np.ndarray:
    out=[]
    with torch.inference_mode():
        for i in range(0, len(h), 256): out.append(head(torch.from_numpy(np.ascontiguousarray(h[i:i+256])).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def centered(z: np.ndarray) -> np.ndarray: return z - z.mean(axis=1, keepdims=True)
def probs(z: np.ndarray) -> np.ndarray:
    a = z - z.max(axis=1, keepdims=True); e = np.exp(a); return e / np.maximum(e.sum(1, keepdims=True), EPS)
def ce(z: np.ndarray, y: np.ndarray) -> np.ndarray: return -np.log(np.clip(probs(z)[np.arange(len(y)), y], EPS, 1.0))
def margin(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    own=z[np.arange(len(y)),y]; tmp=z.copy(); tmp[np.arange(len(y)),y]=-np.inf; return own-tmp.max(1)
def js(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p,q=probs(a),probs(b); m=.5*(p+q); return .5*np.sum(p*np.log(np.clip(p/m,EPS,None)),1)+.5*np.sum(q*np.log(np.clip(q/m,EPS,None)),1)
def ba(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y==c] == c) for c in np.unique(y)])) if len(y) else float("nan")


def subject_finite(clean_z: np.ndarray, changed_z: np.ndarray, y: np.ndarray, owner: np.ndarray) -> dict[str, dict[str, float]]:
    delta=centered(changed_z-clean_z); p,q=probs(clean_z),probs(changed_z)
    tv=.5*np.abs(p-q).sum(1); flip=(clean_z.argmax(1)!=changed_z.argmax(1)).astype(float)
    out={}
    for s in natural(owner):
        ix=np.flatnonzero(owner.astype(str)==s)
        out[s]={"centered_logit_rms":float(np.sqrt(np.mean(delta[ix]**2))),"true_margin_change":float(np.mean(margin(clean_z[ix],y[ix])-margin(changed_z[ix],y[ix]))),"TV":float(np.mean(tv[ix])),"JS":float(np.mean(js(clean_z[ix],changed_z[ix]))),"flip_rate":float(np.mean(flip[ix])),"CE_change":float(np.mean(ce(changed_z[ix],y[ix])-ce(clean_z[ix],y[ix]))),"BA_change":float(ba(y[ix],clean_z[ix].argmax(1))-ba(y[ix],changed_z[ix].argmax(1)))}
    return out


def mean_dict(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for s in rows[0] if rows else []:
        out[s] = {k: float(np.mean([r[s][k] for r in rows])) for k in rows[0][s]}
    return out


def raw_base(spec: dict[str, Any]) -> np.ndarray:
    return ((spec["directions"].T * spec["scale"][None, :]) @ spec["basis"].T).astype(np.float32)


def residual(h: np.ndarray, q: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return h - spec["mean"] - q @ raw_base(spec)


def reconstruct(q: np.ndarray, r: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return (r + spec["mean"] + q @ raw_base(spec)).astype(np.float32)


def random_dims(rank: int, k: int, model: str, task: str, fold: int, purpose: str) -> list[np.ndarray]:
    return [np.sort(np.random.default_rng(stable_seed(purpose, model, task, fold, d)).choice(rank, k, replace=False)).astype(int) for d in range(RANDOM_DRAWS)]


def local_energy(head: torch.nn.Module, q: np.ndarray, r: np.ndarray, spec: dict[str, Any], dims: np.ndarray, owner: np.ndarray, device: torch.device) -> dict[str, float]:
    base=torch.from_numpy(raw_base(spec)).to(device); mu=torch.from_numpy(spec["mean"].astype(np.float32)).to(device)
    values=[]
    for start in range(0,len(q),64):
        qq=torch.from_numpy(q[start:start+64].astype(np.float32)).to(device).requires_grad_(True); rr=torch.from_numpy(r[start:start+64].astype(np.float32)).to(device)
        z=head(rr+mu+qq@base); z=z-z.mean(1,keepdim=True); cols=[]
        for c in range(z.shape[1]): cols.append(torch.autograd.grad(z[:,c].sum(),qq,retain_graph=c+1<z.shape[1])[0])
        j=torch.stack(cols,1).detach().cpu().numpy(); values.append(np.mean(j[:,:,dims]**2,axis=(1,2)))
    e=np.concatenate(values); return {s:float(np.mean(e[owner.astype(str)==s])) for s in natural(owner)}


def cross_label_pairs(q: np.ndarray, y: np.ndarray, owner: np.ndarray, dims: np.ndarray) -> list[tuple[int,int]]:
    n=np.setdiff1d(np.arange(q.shape[1]),dims); pairs=[]
    for s in natural(owner):
        ids=np.flatnonzero(owner.astype(str)==s)
        for i in ids:
            candidates=ids[y[ids]!=y[i]]
            for lab in np.unique(y[candidates]):
                z=candidates[y[candidates]==lab]
                if len(z): pairs.append((int(i),int(z[np.argmin(np.sum((q[z][:,n]-q[i,n])**2,axis=1))])))
    return pairs


def cross_session_pairs(q1: np.ndarray, y1: np.ndarray, s1: np.ndarray, q2: np.ndarray, y2: np.ndarray, s2: np.ndarray, dims: np.ndarray) -> list[tuple[int,int,int]]:
    n=np.setdiff1d(np.arange(q1.shape[1]),dims); pairs=[]
    for side,(qa,ya,sa,qb,yb,sb) in enumerate(((q1,y1,s1,q2,y2,s2),(q2,y2,s2,q1,y1,s1))):
        for i in range(len(qa)):
            z=np.flatnonzero((sb.astype(str)==str(sa[i]))&(yb==ya[i]))
            if len(z): pairs.append((side,i,int(z[np.argmin(np.sum((qb[z][:,n]-qa[i,n])**2,axis=1))])))
    return pairs


def swap_summary(head: torch.nn.Module, q: np.ndarray, r: np.ndarray, y: np.ndarray, owner: np.ndarray, pairs: list[tuple[int,int]], dims: np.ndarray, spec: dict[str, Any], device: torch.device) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not pairs: return {"pairs":0}, {}
    ii=np.asarray([a for a,_ in pairs]); jj=np.asarray([b for _,b in pairs]); qs=q[ii].copy(); qs[:,dims]=q[jj][:,dims]
    z0=head_logits(head,reconstruct(q[ii],r[ii],spec),device); z=head_logits(head,reconstruct(qs,r[ii],spec),device); donor=y[jj]
    rows={}
    for s in natural(owner[ii]):
        ix=np.flatnonzero(owner[ii].astype(str)==s); rows[s]={"donor_centered_shift":float(np.mean(centered(z-z0)[ix,donor[ix]])),"recipient_margin_change":float(np.mean(margin(z0[ix],y[ii][ix])-margin(z[ix],y[ii][ix]))),"transfer_rate":float(np.mean(z[ix].argmax(1)==donor[ix])),"flip_rate":float(np.mean(z[ix].argmax(1)!=z0[ix].argmax(1))),"JS":float(np.mean(js(z0[ix],z[ix]))),"logit_rms":float(np.sqrt(np.mean(centered(z-z0)[ix]**2)))}
    return {"pairs":len(pairs),**{k:float(np.mean([v[k] for v in rows.values()])) for k in next(iter(rows.values()))}},rows


def interaction(head: torch.nn.Module, q: np.ndarray, r: np.ndarray, owner: np.ndarray, pairs: list[tuple[int,int]], dims: np.ndarray, spec: dict[str,Any], device: torch.device) -> dict[str,float]:
    if not pairs:return {}
    ii=np.asarray([a for a,_ in pairs]); jj=np.asarray([b for _,b in pairs]); q1,q2=q[ii],q[jj]; n=np.setdiff1d(np.arange(q.shape[1]),dims)
    def mix(p,nv):
        z=q1.copy(); z[:,dims]=p[:,dims]; z[:,n]=nv[:,n]; return head_logits(head,reconstruct(z,r[ii],spec),device)
    z11,z21,z12,z22=mix(q1,q1),mix(q2,q1),mix(q1,q2),mix(q2,q2); inter=centered(z11-z21-z12+z22); out={}
    for s in natural(owner[ii]):out[s]=float(np.mean(np.linalg.norm(inter[owner[ii].astype(str)==s],axis=1)))
    return out


def provenance_row(model: str, task: str, fold: int) -> tuple[dict[str,Any],Path,dict[str,Any],dict[str,Any]]:
    h=helper(model); record, ckpt=h.cell(model,task,fold); stored_path=native_cell_path(model,task,fold)
    if not stored_path.is_file(): raise FileNotFoundError(f"current PERSIST assignment missing: {stored_path}")
    stored=json.loads(stored_path.read_text(encoding="utf-8"))
    if stored.get("checkpoint_sha256")!=sha(ckpt): raise RuntimeError("stored Protected assignment checkpoint mismatch")
    return record,ckpt,stored,{"checkpoint_sha256":sha(ckpt),"protected_assignment_sha256":hashlib.sha256(json.dumps(stored.get("protected_assignment",[]),sort_keys=True).encode()).hexdigest(),"stored_cell_sha256":sha(stored_path)}


def cell(model: str, task: str, fold: int) -> None:
    target=target_path(model,task,fold)
    if target.is_file(): print("CELL_CACHED",model,task,fold,flush=True); return
    base={"model":model,"task":task,"fold":fold,"seed":0,"training_performed":False,"final_heldout_accessed":False}
    try:
        record,ckpt,stored,prov=provenance_row(model,task,fold); data=outer_data(task,fold)
        if data["normalizer"].get("mean_std_sha256") != record["normalizer"]["mean_std_sha256"]: raise RuntimeError("TRAIN normalizer hash mismatch")
        x,y,owner,sessions=capped_train(data,task,model,fold); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net,head=helper(model).build_model({"Model":model,"Task":task,"fold":fold,"seed":0,"channels":int(record.get("channels") or 62),"samples":int(record.get("samples") or 1000),"classes":int(record["classes"]),"checkpoint_path":str(ckpt),"recipe_name":record.get("recipe",{}).get("name"),"trainable_parameters":int(record.get("trainable_parameters",record.get("parameters",0)))},device)
        ht,_,_=hook_representations(net,head,x,model,device); spec=helper(model).spectrum(ht,y,owner,sessions,task,model,fold); spec["classes"]=data["classes"]
        dims=np.asarray(stored.get("protected_blocks",[]),dtype=int)
        if int(stored.get("rank",-1))!=int(spec["rank"]) or np.any(dims<0) or np.any(dims>=spec["rank"]): raise RuntimeError("current Protected coordinate/basis provenance cannot be matched")
        prov.update({"normalizer_sha256":data["normalizer"]["mean_std_sha256"],"basis_sha256":array_sha(spec["mean"],spec["basis"],spec["scale"],spec["directions"]),"split_sha256":data["split_sha256"],"outer_subject_ids":data["outer_subject_ids"],"protected_dims":dims.tolist()})
        if not len(dims): write_json(target,{**base,"status":"EMPTY_PROTECTED",**prov}); return
        he,z_direct,z_clean=hook_representations(net,head,data["outer_future_x"],model,device)
        if not np.allclose(z_direct,z_clean,rtol=1e-5,atol=1e-6): raise RuntimeError("clean native logits do not replay")
        qe=helper(model).canonical(he,spec); re=residual(he,qe,spec); erased=helper(model).erase(he,spec,dims); zp=head_logits(head,erased,device)
        finite_p=subject_finite(z_clean,zp,data["outer_future_y"],data["outer_future_subjects"])
        rd=random_dims(spec["rank"],len(dims),model,task,fold,"native-equal-rank-random")
        ra=[]; rb=[]
        dp=np.linalg.norm(he-erased,axis=1,keepdims=True)
        for d in rd:
            hr=helper(model).erase(he,spec,d); zr=head_logits(head,hr,device); ra.append(subject_finite(z_clean,zr,data["outer_future_y"],data["outer_future_subjects"]))
            dr=he-hr; hm=he-dr*(dp/np.maximum(np.linalg.norm(dr,axis=1,keepdims=True),EPS)); rb.append(subject_finite(z_clean,head_logits(head,hm.astype(np.float32),device),data["outer_future_y"],data["outer_future_subjects"]))
        finite_a,finite_b=mean_dict(ra),mean_dict(rb)
        local_p=local_energy(head,qe,re,spec,dims,data["outer_future_subjects"],device); local_r=[local_energy(head,qe,re,spec,d,data["outer_future_subjects"],device) for d in rd]
        local_rm={s:float(np.mean([r[s] for r in local_r])) for s in local_p}
        pairs=cross_label_pairs(qe,data["outer_future_y"],data["outer_future_subjects"],dims); swap_p,swap_subject=swap_summary(head,qe,re,data["outer_future_y"],data["outer_future_subjects"],pairs,dims,spec,device)
        swaps_r=[swap_summary(head,qe,re,data["outer_future_y"],data["outer_future_subjects"],pairs,d,spec,device)[0] for d in rd]; swap_r={k:float(np.mean([v.get(k,np.nan) for v in swaps_r])) for k in swap_p if k!="pairs"}
        inter_p=interaction(head,qe,re,data["outer_future_subjects"],pairs,dims,spec,device); inter_r=[interaction(head,qe,re,data["outer_future_subjects"],pairs,d,spec,device) for d in rd]; inter_rm={s:float(np.mean([v.get(s,np.nan) for v in inter_r])) for s in inter_p}
        hs,zs0,_=hook_representations(net,head,data["outer_source_x"],model,device); qs=helper(model).canonical(hs,spec); rs=residual(hs,qs,spec); spairs=cross_session_pairs(qs,data["outer_source_y"],data["outer_source_subjects"],qe,data["outer_future_y"],data["outer_future_subjects"],dims)
        session_rows=[]
        for side,i,j in spairs:
            qa,ra_,za,ya,oa=(qs,rs,zs0,data["outer_source_y"],data["outer_source_subjects"]) if side==0 else (qe,re,z_clean,data["outer_future_y"],data["outer_future_subjects"])
            qb=qe if side==0 else qs; qx=qa[[i]].copy(); qx[:,dims]=qb[[j]][:,dims]; zz=head_logits(head,reconstruct(qx,ra_[[i]],spec),device)[0]; session_rows.append({"subject_id":str(oa[i]),"centered_logit_rms":float(np.sqrt(np.mean(centered((zz-za[i])[None,:])**2))),"true_margin_change":float(margin(za[[i]],np.asarray([ya[i]]))[0]-margin(zz[None,:],np.asarray([ya[i]]))[0]),"JS":float(js(za[[i]],zz[None,:])[0]),"prediction_consistent":float(zz.argmax()==za[i].argmax())})
        subj=[]
        for s in natural(data["outer_future_subjects"]):
            row={**base,"subject_id":s,"finite_P":finite_p[s],"finite_random":finite_a[s],"finite_normmatched_random":finite_b[s],"local_P":local_p[s],"local_random":local_rm[s],"local_ratio":local_p[s]/max(local_rm[s],EPS),"interaction_P":inter_p.get(s,float("nan")),"interaction_random":inter_rm.get(s,float("nan"))}; subj.append(row)
        same={k:float(np.mean([r[k] for r in session_rows])) for k in ("centered_logit_rms","true_margin_change","JS","prediction_consistent")} if session_rows else {"centered_logit_rms":float("nan"),"true_margin_change":float("nan"),"JS":float("nan"),"prediction_consistent":float("nan")}
        result={**base,"status":"COMPLETE","protected_rank":len(dims),"rank":spec["rank"],**prov,"subjects":subj,"swap":{"protected":swap_p,"random_mean":swap_r,"subject_rows":swap_subject},"same_label_cross_session":same,"same_label_pair_count":len(session_rows)}
        write_json(target,result); print("CELL_COMPLETE",model,task,fold,flush=True)
        del net,ht,he,hs; torch.cuda.empty_cache();
    except Exception as e:
        write_json(target,{**base,"status":"FAIL_CLOSED","reason":f"{type(e).__name__}: {e}"}); print("CELL_FAIL_CLOSED",model,task,fold,str(e),flush=True)


def aggregate() -> None:
    cells=[]
    for m in MODELS:
        for t in TASKS:
            for f in FOLDS:
                p=target_path(m,t,f)
                if not p.is_file(): raise RuntimeError(f"missing cell {m}/{t}/f{f}")
                cells.append(json.loads(p.read_text(encoding="utf-8")))
    summary=[];subject=[];swaps=[]
    for c in cells:
        base={k:c.get(k) for k in ("model","task","fold","seed","status","protected_rank","rank")}
        if c["status"]!="COMPLETE": summary.append(base); continue
        ss=c["subjects"]; f=lambda key:float(np.nanmean([x[key] if not isinstance(x[key],dict) else x[key]["centered_logit_rms"] for x in ss]))
        p=f("finite_P"); r=f("finite_random"); b=f("finite_normmatched_random")
        local=float(np.nanmean([x["local_P"] for x in ss])); lr=float(np.nanmean([x["local_random"] for x in ss])); inter=float(np.nanmean([x["interaction_P"] for x in ss])); ir=float(np.nanmean([x["interaction_random"] for x in ss]))
        summary.append({**base,"finite_P":p,"finite_random_mean":r,"finite_normmatched_random_mean":b,"finite_ratio":p/max(r,EPS),"local_P":local,"local_random_mean":lr,"local_ratio":local/max(lr,EPS),"cross_label_P_transfer":c["swap"]["protected"].get("transfer_rate"),"random_transfer":c["swap"]["random_mean"].get("transfer_rate"),"same_label_crosssession_P_change":c["same_label_cross_session"].get("centered_logit_rms"),"P_nonP_interaction":inter,"random_interaction":ir})
        subject.extend(ss); swaps.append({**base,**c["swap"]["protected"],**{"random_"+k:v for k,v in c["swap"]["random_mean"].items()},"same_label_pair_count":c["same_label_pair_count"],**{"same_"+k:v for k,v in c["same_label_cross_session"].items()}})
    write_csv(OUT/"CELL_SUMMARY.csv",summary); write_csv(OUT/"SUBJECT_LEVEL_RESULTS.csv",subject); write_csv(OUT/"SWAP_LEVEL_AUDIT.csv",swaps)
    report=["# Native Protected utilization audit", "", "All completed cells use frozen native heads and outer-development subjects only. No backbone, classifier, or probe was trained/refit.", "", "A mechanism label is deliberately withheld unless all finite, local, swap, and interaction evidence agree; consult `CELL_SUMMARY.csv` rather than treating any one metric as Protected usage percentage."]
    (OUT/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("AGGREGATE_COMPLETE",flush=True)


def lock() -> None:
    rows=[]
    for m in MODELS:
        for t in TASKS:
            for f in FOLDS:
                record,ckpt,stored,prov=provenance_row(m,t,f); data=outer_data(t,f)
                rows.append({"model":m,"task":t,"fold":f,"seed":0,"checkpoint_sha256":prov["checkpoint_sha256"],"normalizer_sha256":data["normalizer"]["mean_std_sha256"],"protected_assignment_sha256":prov["protected_assignment_sha256"],"stored_cell_sha256":prov["stored_cell_sha256"],"outer_subject_ids":data["outer_subject_ids"],"final_heldout_accessed":False})
    value={"schema":"PERSIST_EEG_NATIVE_PROTECTED_UTILIZATION_SEED0_V1","created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"models":MODELS,"tasks":TASKS,"folds":FOLDS,"seed":0,"random_draws":RANDOM_DRAWS,"evaluation":"outer-development only","final_heldout_accessed":False,"cells":rows}
    write_json(PROTOCOL/"PROVENANCE.json",value); (PROTOCOL/"PROVENANCE.sha256").write_text(sha(PROTOCOL/"PROVENANCE.json")+"\n")
    print("PROTOCOL_LOCKED",len(rows),flush=True)


def main() -> None:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True); sub.add_parser("lock"); one=sub.add_parser("cell"); one.add_argument("model",choices=MODELS);one.add_argument("task",choices=TASKS);one.add_argument("fold",type=int,choices=FOLDS); sub.add_parser("run-all");sub.add_parser("aggregate"); a=p.parse_args()
    if a.cmd=="lock":lock()
    elif a.cmd=="cell":cell(a.model,a.task,a.fold)
    elif a.cmd=="run-all":
        for m in MODELS:
            for t in TASKS:
                for f in FOLDS: cell(m,t,f)
    else:aggregate()

if __name__=="__main__":main()
