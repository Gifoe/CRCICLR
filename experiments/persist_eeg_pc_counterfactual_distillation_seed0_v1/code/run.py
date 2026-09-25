"""Frozen-oracle counterfactual distillation into native EEGNet, seed zero."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("PC_DISTILL_RUNTIME", str(REPO.parent / "pc_distill_v1_runtime")))
ACTION = REPO / "experiments" / "persist_eeg_cp_actionability_oracle_v1_seed0"
V3_EXP = REPO / "experiments" / "persist_eeg_selective_cp_routing_v3_seed0"
TASKS = ("OpenBMI_MI", "OpenBMI_SSVEP")
FOLDS = tuple(range(5))
ARMS = ("ORIGINAL", "CE_ONLY_CONTINUATION", "TOP3_DISTILL", "MARGIN_UPPER_DISTILL",
        "CE_UPPER_LOGIT_ONLY", "CE_UPPER_FULL")
TEACHERS = ("TOP3", "MARGIN_UPPER", "CE_UPPER")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(int(os.environ.get("PC_DISTILL_CPU_THREADS", "12")), os.cpu_count() or 1))
torch.backends.cudnn.benchmark = False


def load_source():
    import importlib.util
    import sys
    path = ACTION / "code" / "run.py"
    spec = importlib.util.spec_from_file_location("pc_distill_frozen_actionability", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


A = load_source()
V3 = A.V3
B = A.B


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "big")


def source_cell(task, fold):
    old = json.loads((ACTION / "protocol" / "PROTOCOL_LOCK.json").read_text())
    return next(c for c in old["reused_cells"] if c["task"] == task and int(c["fold"]) == fold)


def paths(task, fold):
    c = source_cell(task, fold)
    ck = REPO.parent / "pc_refine_v1_runtime" / "refit" / task.lower() / f"fold{fold}_seed0" / "BASELINE.pt"
    geom = V3_EXP / "outputs" / "cells" / task.lower() / f"fold{fold}_seed0" / "geometry.npz"
    return c, ck, geom


def preflight_lock():
    old_path = ACTION / "protocol" / "PROTOCOL_LOCK.json"
    old = json.loads(old_path.read_text())
    deps = {str(p.relative_to(REPO)): sha(p) for p in [old_path,
            ACTION / "code" / "run.py", V3_EXP / "code" / "run.py"]}
    cells = []
    for task in TASKS:
        for fold in FOLDS:
            c, ck, geom = paths(task, fold)
            if sha(ck) != c["checkpoint_sha256"] or sha(geom) != c["geometry_sha256"]:
                raise RuntimeError(f"frozen source hash mismatch {task} fold{fold}")
            role, split, _, _, _ = B.role(task, fold)
            if str(split) != str(c["split_sha256"]):
                raise RuntimeError(f"split hash mismatch {task} fold{fold}")
            train = set(map(str, role["inner_train_subjects"]))
            discovery = set(map(str, role["inner_val_subjects"]))
            outer = set(map(str, role["outer_dev_subjects"]))
            if train & discovery or train & outer or discovery & outer:
                raise RuntimeError("split role overlap")
            if train != set(map(str, c["train_subjects"])) or discovery != set(map(str, c["discovery_subjects"])):
                raise RuntimeError("source subject role drift")
            cells.append({"task": task, "fold": fold, "checkpoint_sha256": sha(ck),
                          "geometry_sha256": sha(geom), "split_sha256": str(split),
                          "top3": c["top3_directions_abs_utility_rank"],
                          "inner_train_subjects": sorted(train, key=int),
                          "discovery_subjects": sorted(discovery, key=int),
                          "outer_dev_subject_count_excluded": len(outer)})
    lock = {"experiment": EXP.name, "seed": 0, "tasks": list(TASKS), "folds": list(FOLDS),
            "source_sha256": deps, "cells": cells,
            "teacher_generation_role": "inner_train_only", "student_training_role": "inner_train_only",
            "evaluation_role": "inner_val_discovery", "primary_session": "future_session",
            "final_heldout_access_allowed": False, "outer_dev_array_access_allowed": False,
            "teacher_geometry": "frozen_actionability_v1_V3", "source_stage": "spatial_elu_pool1",
            "successor_stage": "depth_point_elu_pool2", "alpha_grid": A.ALPHAS.tolist(),
            "accept_delta_ce_gt": 1e-4, "weight": "clip(delta_CE/(CE_native+1e-6),0,1)",
            "temperature": 2.0, "lambda_z": 1.0, "lambda_P": 0.1,
            "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 5e-4,
            "epochs": 10, "batch_size": 128, "batch_seed": 0,
            "BN_running_state": "frozen", "student_initialization": "exact_canonical_checkpoint",
            "discovery_epoch_selection": False, "bootstrap_replicates": 20000,
            "oracle_status": "LABEL_PRIVILEGED_TRAINING_ORACLE",
            "provenance_limitation": "canonical V3 checkpoint and geometry were refit on all non-final source-session subjects, including discovery subjects"}
    out = OUT / "PROTOCOL_LOCK.json"
    if out.exists() and json.loads(out.read_text()) != lock:
        raise RuntimeError("existing protocol lock changed")
    write_json(out, lock)
    write_json(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json",
               {"FINAL_HELDOUT_ACCESSED": False, "final_heldout_array_reads": 0,
                "outer_dev_array_reads": 0, "permitted_loader_roles": ["inner_train", "inner_val_discovery"],
                "source_role_metadata_read_for_exclusion": True})
    write_json(OUT / "TEACHER_SOURCE_AUDIT.json",
               {"source_protocol_sha256": sha(old_path), "source_code_hashes": deps,
                "cells": cells, "frozen": True, "refit_provenance_limitation": lock["provenance_limitation"]})
    return lock


def model_from_checkpoint(raw_shape, classes, checkpoint):
    model = B.eegnet({"channels": raw_shape[1], "samples": raw_shape[2], "classes": classes})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(DEVICE)


def native_forward_with_p(model, x, qd, md):
    h = model.drop1(model.pool1(F.elu(model.bn2(model.spatial(model.bn1(model.temporal(x.unsqueeze(1))))))))
    hd = model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(h)))))).flatten(1)
    p = (hd - md) @ qd
    return model.head(model.embedding(hd)), p


def freeze_bn_running(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def bn_bytes(model):
    return {k: v.detach().cpu().numpy().tobytes() for k, v in model.state_dict().items()
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k}


def teacher_batch(model, hs, base, labels, geo, top3, samples):
    qs, qd, ms, md, basis = (torch.as_tensor(geo[k], device=DEVICE, dtype=torch.float32)
                              for k in ("qs", "qd", "ms", "md", "c_basis"))
    ht = torch.as_tensor(hs, device=DEVICE, dtype=torch.float32)
    y = torch.as_tensor(labels, device=DEVICE, dtype=torch.long)
    c0 = torch.as_tensor(base["complement"], device=DEVICE, dtype=torch.float32)
    centered = ht - ms
    coeff = (centered - ((centered @ qs) @ qs.T)) @ basis
    curves, identity, preserve, _ = A.evaluate_curve_bank(model, hs, labels, base,
        geo["qs"], geo["ms"], geo["qd"], geo["md"], geo["c_basis"], samples, 32)
    # The old Top3 selector is called verbatim; P is read from the same final routed h.
    with torch.inference_mode():
        ztop_old, _, top_preserve = A.sequential_top3(model, ht, coeff, basis, qd, md, c0, y, top3, samples)
        current = ht.clone(); chosen_top = []
        alpha_t = torch.as_tensor(A.ALPHAS, device=DEVICE)
        for direction in top3:
            variants = current[:, None, :] + (alpha_t[None, :, None] - 1) * coeff[:, direction, None, None] * basis[:, direction][None, None, :]
            z, _, _ = A.routed_logits_and_p(model, variants.reshape(-1, ht.shape[1]), qd, md,
                c0[:, None, :].expand(-1, len(A.ALPHAS), -1).reshape(-1, c0.shape[1]), samples)
            margin = A.torch_margins(z, y[:, None].expand(-1, len(A.ALPHAS)).reshape(-1)).reshape(len(y), -1)
            idx = A.pick_index(margin.cpu().numpy(), maximize=True)
            chosen = alpha_t[torch.as_tensor(idx, device=DEVICE, dtype=torch.long)]
            chosen_top.append(chosen.cpu().numpy())
            current = current + (chosen[:, None] - 1) * coeff[:, direction, None] * basis[:, direction][None, :]
        ztop, ptop, err = A.routed_logits_and_p(model, current, qd, md, c0, samples)
        if not torch.allclose(ztop, ztop_old, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Top3 source definition drift")
        out = {"TOP3": (ztop, ptop, np.stack(chosen_top, axis=1))}
        for name, key, maximize in (("MARGIN_UPPER", "margin", True), ("CE_UPPER", "CE", False)):
            ix = A.pick_index(curves[key].reshape(-1, len(A.ALPHAS)), maximize=maximize).reshape(len(y), -1)
            alphas = A.ALPHAS[ix]
            routed_h = ht + (coeff * torch.as_tensor(alphas - 1, device=DEVICE)) @ basis.T
            z, p, error = A.routed_logits_and_p(model, routed_h, qd, md, c0, samples)
            preserve = max(preserve, error)
            out[name] = (z, p, alphas)
        result = {}
        for name, (z, p, alpha) in out.items():
            result[name] = {"z": z.cpu().numpy(), "p": p.cpu().numpy(),
                            "alpha": alpha.astype(np.float32),
                            "ce": F.cross_entropy(z, y, reduction="none").cpu().numpy(),
                            "margin": A.torch_margins(z, y).cpu().numpy()}
    return result, max(float(preserve), float(top_preserve), float(err)), float(identity)


def collect_role(task, fold, role_name, model, geo, top3, checkpoint):
    role, _, cache_name, sources, future = B.role(task, fold)
    ids = sorted(map(str, role["inner_train_subjects" if role_name == "train" else "inner_val_subjects"]), key=int)
    mapping = None; x_all=[]; y_all=[]; subject_all=[]; session_all=[]; trial_all=[]
    z0_all=[]; p0_all=[]; ce0_all=[]; teacher = {k: defaultdict(list) for k in TEACHERS}
    max_preserve = max_identity = 0.0
    for subject in ids:
        for session in sorted(set(map(int, sources)) | {int(future)}):
            raw, labels, owners, next_mapping = B.rows(task, [subject], (session,), cache_name, mapping, final=False)
            if mapping is None: mapping = dict(next_mapping)
            elif mapping != next_mapping: raise RuntimeError("label mapping drift")
            if not len(raw) or not np.all(np.asarray(owners).astype(str) == subject):
                raise RuntimeError("subject slice drift")
            x = ((raw - geo["normalizer_mu"][None, :, None]) /
                 np.maximum(geo["normalizer_sd"][None, :, None], 1e-6)).astype(np.float32)
            if model is None:
                model = model_from_checkpoint(x.shape, len(mapping), checkpoint)
                model.eval()
                for p in model.parameters(): p.requires_grad_(False)
            if x.shape[1] != model.spatial.kernel_size[0]: raise RuntimeError("channel mismatch")
            hs = V3.spatial(model, x, batch=128)
            base = V3.full_forward(model, hs, geo["qd"], geo["md"], x.shape[2], batch=128)
            y = np.asarray(labels, dtype=np.int64)
            z0 = base["logits"].astype(np.float32)
            ce0 = F.cross_entropy(torch.from_numpy(z0), torch.from_numpy(y), reduction="none").numpy()
            if role_name == "train":
                batch_teacher, preserve, identity = teacher_batch(model, hs, base, y, geo, top3, x.shape[2])
                max_preserve=max(max_preserve,preserve); max_identity=max(max_identity,identity)
                for name in TEACHERS:
                    for key, value in batch_teacher[name].items(): teacher[name][key].append(value)
            x_all.append(x); y_all.append(y); subject_all.extend([subject]*len(y)); session_all.extend([session]*len(y))
            trial_all.extend(range(len(y))); z0_all.append(z0); p0_all.append(base["protected"]); ce0_all.append(ce0)
            print("COLLECT",task,fold,role_name,subject,session,len(y),flush=True)
    data = {"x": np.concatenate(x_all), "y": np.concatenate(y_all),
            "subject": np.asarray(subject_all), "session": np.asarray(session_all),
            "trial": np.asarray(trial_all), "z0": np.concatenate(z0_all),
            "p0": np.concatenate(p0_all), "ce0": np.concatenate(ce0_all),
            "future_session": int(future), "model": model}
    if role_name == "train":
        data["teacher"] = {name: {key: np.concatenate(parts) for key,parts in value.items()}
                           for name,value in teacher.items()}
        data["max_preserve"] = max_preserve; data["max_identity"] = max_identity
    return data


def teacher_audit(task, fold, train, cache_sha):
    y=train["y"]; z0=train["z0"]; pred0=z0.argmax(1); p0=train["p0"]; ce0=train["ce0"]
    native_margin = A.torch_margins(torch.from_numpy(z0), torch.from_numpy(y)).numpy()
    rows=[]; analysis=[]
    allz={"NATIVE": z0, **{k:v["z"] for k,v in train["teacher"].items()}}
    T=2.0; q0=F.softmax(torch.from_numpy(z0)/T,dim=1).numpy()
    for name,z in allz.items():
        q=F.softmax(torch.from_numpy(z)/T,dim=1).numpy().clip(1e-12,1)
        entropy=-(q*np.log(q)).sum(1)
        trueprob=q[np.arange(len(y)),y]
        klnative=(q*(np.log(q)-np.log(q0.clip(1e-12,1)))).sum(1)
        analysis.append({"task":task,"fold":fold,"teacher":name,"trials":len(y),
                         "temperature":T,"mean_entropy":float(entropy.mean()),
                         "mean_true_class_probability":float(trueprob.mean()),
                         "KL_qT_to_onehot": "inf" if np.any(q[np.arange(q.shape[1])[None,:]!=y[:,None]]>0) else 0,
                         "mean_KL_onehot_to_qT":float((-np.log(trueprob)).mean()),
                         "mean_KL_qT_to_q_native":float(klnative.mean())})
    for name,v in train["teacher"].items():
        delta=ce0-v["ce"]; w=np.where(delta>1e-4,np.clip(delta/(ce0+1e-6),0,1),0).astype(np.float32)
        v["w"]=w
        pred=v["z"].argmax(1)
        rows.append({"task":task,"fold":fold,"teacher":name,"trials":len(y),
                     "accepted_fraction":float(np.mean(w>0)),"mean_native_CE":float(ce0.mean()),
                     "mean_teacher_CE":float(v["ce"].mean()),"mean_delta_CE":float(delta.mean()),
                     "median_delta_CE":float(np.median(delta)),
                     "mean_margin_gain":float((v["margin"]-native_margin).mean()),
                     "prediction_rescue_count":int(np.sum((pred0!=y)&(pred==y))),
                     "correct_prediction_damage_count":int(np.sum((pred0==y)&(pred!=y))),
                     "mean_P_movement":float(np.linalg.norm(v["p"]-p0,axis=1).mean()),
                     "cache_sha256":cache_sha,"max_successor_C_error":train["max_preserve"],
                     "max_alpha1_identity_error":train["max_identity"]})
    return rows,analysis


def train_arm(task, fold, arm, train, checkpoint, geo, sigma):
    seed=stable_seed("student", task, fold, 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=model_from_checkpoint(train["x"].shape, train["z0"].shape[1], checkpoint)
    init_sha=A.model_state_sha(model)
    model.train(); freeze_bn_running(model)
    before=bn_bytes(model)
    qd=torch.as_tensor(geo["qd"],device=DEVICE,dtype=torch.float32)
    md=torch.as_tensor(geo["md"],device=DEVICE,dtype=torch.float32)
    sigma_t=torch.as_tensor(sigma,device=DEVICE,dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-4)
    teacher_name={"TOP3_DISTILL":"TOP3","MARGIN_UPPER_DISTILL":"MARGIN_UPPER",
                  "CE_UPPER_LOGIT_ONLY":"CE_UPPER","CE_UPPER_FULL":"CE_UPPER"}.get(arm)
    n=len(train["y"]); trajectory=[]; begin=time.perf_counter()
    for epoch in range(1,11):
        model.train(); freeze_bn_running(model)
        rng=np.random.default_rng(stable_seed("batch",task,fold,epoch,0))
        order=rng.permutation(n)
        totals=np.zeros(4,dtype=np.float64)
        for start in range(0,n,128):
            idx=order[start:start+128]; b=len(idx)
            x=torch.as_tensor(train["x"][idx],device=DEVICE)
            y=torch.as_tensor(train["y"][idx],device=DEVICE,dtype=torch.long)
            z,p=native_forward_with_p(model,x,qd,md)
            ce=F.cross_entropy(z,y)
            kd=torch.zeros((),device=DEVICE); pl=torch.zeros((),device=DEVICE)
            if teacher_name:
                target=train["teacher"][teacher_name]
                w=torch.as_tensor(target["w"][idx],device=DEVICE)
                zt=torch.as_tensor(target["z"][idx],device=DEVICE)
                qt=F.softmax(zt/2,dim=1)
                kd=((qt*(qt.clamp_min(1e-12).log()-F.log_softmax(z/2,dim=1))).sum(1)*w).mean()*4
                if arm!="CE_UPPER_LOGIT_ONLY":
                    pt=torch.as_tensor(target["p"][idx],device=DEVICE)
                    pl=((((p-pt)/(sigma_t+1e-6))**2).mean(1)*w).mean()
            loss=ce+kd+0.1*pl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            totals += np.array([float(ce.detach()),float(kd.detach()),float(pl.detach()),float(loss.detach())])*b
        trajectory.append({"task":task,"fold":fold,"arm":arm,"epoch":epoch,"trials":n,
                           "CE":totals[0]/n,"weighted_KD":totals[1]/n,
                           "weighted_P":totals[2]/n,"total_loss":totals[3]/n})
        print("EPOCH",task,fold,arm,epoch,"loss",round(totals[3]/n,5),flush=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    train_seconds=time.perf_counter()-begin
    after=bn_bytes(model)
    exact=all(before[k]==after[k] for k in before)
    if not exact: raise RuntimeError(f"BN running state drift: {task}/{fold}/{arm}")
    bn={"task":task,"fold":fold,"arm":arm,"running_state_byte_identical":exact,
        "buffer_count":len(before),"all_affine_trainable":all(
            m.weight.requires_grad and m.bias.requires_grad for m in model.modules()
            if isinstance(m,nn.modules.batchnorm._BatchNorm))}
    ckdir=RUNTIME/"checkpoints"/task.lower()/f"fold{fold}_seed0"
    ckdir.mkdir(parents=True,exist_ok=True)
    ckpath=ckdir/f"{arm}.pt"
    torch.save({"state_dict":model.state_dict(),"arm":arm,"task":task,"fold":fold,
                "epoch":10,"protocol_sha256":sha(OUT/"PROTOCOL_LOCK.json")},ckpath)
    check={"task":task,"fold":fold,"arm":arm,"epoch":10,
           "checkpoint_sha256":sha(ckpath),"initial_model_state_sha256":init_sha,
           "final_model_state_sha256":A.model_state_sha(model),
           "source_checkpoint_sha256":sha(checkpoint),"parameter_count":sum(p.numel() for p in model.parameters()),
           "checkpoint_runtime_path":str(ckpath),"bn_running_state_identical":exact}
    return model,trajectory,bn,check,train_seconds


def eval_native(model,x,batch=100):
    model.eval(); out=[]
    if torch.cuda.is_available(): torch.cuda.synchronize()
    begin=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(x),batch):
            xb=torch.as_tensor(np.ascontiguousarray(x[i:i+batch]),device=DEVICE)
            out.append(model(xb).float().cpu().numpy())
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return np.concatenate(out),time.perf_counter()-begin


def metric_rows(task,fold,arm,disc,z):
    rows=[]; y=disc["y"]
    for subject in sorted(set(disc["subject"]),key=int):
        for session in sorted(set(disc["session"])):
            ix=np.flatnonzero((disc["subject"]==subject)&(disc["session"]==session))
            if not len(ix):continue
            zz=torch.as_tensor(z[ix]); yy=torch.as_tensor(y[ix],dtype=torch.long)
            pred=z[ix].argmax(1)
            rows.append({"task":task,"fold":fold,"role":"discovery","subject":subject,
                "session":int(session),"is_future_session":int(session)==disc["future_session"],
                "arm":arm,"trials":len(ix),
                "BA":float(balanced_accuracy_score(y[ix],pred)),
                "macro_F1":float(f1_score(y[ix],pred,average="macro",zero_division=0)),
                "NLL":float(F.cross_entropy(zz,yy).item())})
    return rows


def posthoc_oracle(task,fold,disc,geo,top3,sigma,models):
    teacher=disc["model"]; oracle={k:{"z":[],"p":[]} for k in TEACHERS}
    native_p=[]; student_p={arm:[] for arm in models}
    qd=torch.as_tensor(geo["qd"],device=DEVICE,dtype=torch.float32)
    md=torch.as_tensor(geo["md"],device=DEVICE,dtype=torch.float32)
    # All students have already completed native discovery inference before this point.
    for start in range(0,len(disc["y"]),64):
        stop=min(len(disc["y"]),start+64)
        x=np.ascontiguousarray(disc["x"][start:stop]); y=disc["y"][start:stop]
        hs=V3.spatial(teacher,x,batch=64)
        base=V3.full_forward(teacher,hs,geo["qd"],geo["md"],x.shape[2],batch=64)
        targets,preserve,identity=teacher_batch(teacher,hs,base,y,geo,top3,x.shape[2])
        native_p.append(base["protected"])
        for name in TEACHERS:
            oracle[name]["z"].append(targets[name]["z"])
            oracle[name]["p"].append(targets[name]["p"])
        with torch.inference_mode():
            for arm,model in models.items():
                model.eval()
                _,p=native_forward_with_p(model,torch.as_tensor(x,device=DEVICE),qd,md)
                student_p[arm].append(p.cpu().numpy())
    oracle={k:{a:np.concatenate(v) for a,v in d.items()} for k,d in oracle.items()}
    p0=np.concatenate(native_p); align=[]
    target=oracle["CE_UPPER"]["p"]
    delta=target-p0; dn=np.linalg.norm(delta,axis=1)
    for arm,parts in student_p.items():
        p=np.concatenate(parts); move=p-p0
        sn=np.linalg.norm(move,axis=1)
        cos=np.sum(move*delta,axis=1)/np.maximum(sn*dn,1e-12)
        targetcos=np.sum(p*target,axis=1)/np.maximum(np.linalg.norm(p,axis=1)*np.linalg.norm(target,axis=1),1e-12)
        for subject in sorted(set(disc["subject"]),key=int):
            ix=np.flatnonzero((disc["subject"]==subject)&(disc["session"]==disc["future_session"]))
            if not len(ix):continue
            align.append({"task":task,"fold":fold,"subject":subject,"session":disc["future_session"],
                "arm":arm,"label":"POST_HOC_DISCOVERY_DIAGNOSTIC","trials":len(ix),
                "mean_P_distance":float(np.linalg.norm(p[ix]-target[ix],axis=1).mean()),
                "cosine_similarity":float(targetcos[ix].mean()),
                "standardized_P_MSE":float(np.mean(((p[ix]-target[ix])/(sigma+1e-6))**2)),
                "native_to_oracle_movement_recovery":float(np.sum(move[ix]*delta[ix])/max(np.sum(delta[ix]**2),1e-12)),
                "movement_direction_agreement":float(cos[ix].mean())})
    rows=[]
    for name in TEACHERS:
        rows.extend(metric_rows(task,fold,"ORACLE_"+name,disc,oracle[name]["z"]))
    return rows,align


def run_cell(task,fold):
    lock=preflight_lock()
    if task not in TASKS or fold not in FOLDS:raise ValueError("unlocked cell")
    cell_dir=RUNTIME/"cells"/task.lower()/f"fold{fold}_seed0"
    cell_dir.mkdir(parents=True,exist_ok=True)
    complete=cell_dir/"COMPLETE.json"
    if complete.exists():
        obj=json.loads(complete.read_text())
        if obj["protocol_sha256"]==sha(OUT/"PROTOCOL_LOCK.json"):
            print("CELL_ALREADY_COMPLETE",task,fold,flush=True);return
        raise RuntimeError("stale cell complete marker")
    c,checkpoint,geompath=paths(task,fold)
    with np.load(geompath,allow_pickle=False) as g:geo={key:g[key] for key in g.files}
    top3=list(map(int,c["top3_directions_abs_utility_rank"]))
    train=collect_role(task,fold,"train",None,geo,top3,checkpoint)
    source_state=A.model_state_sha(train["model"])
    sigma=train["p0"].std(axis=0).astype(np.float32)
    if not np.all(np.isfinite(sigma)):raise RuntimeError("bad P sigma")
    for value in train["teacher"].values():
        delta=train["ce0"]-value["ce"]
        value["w"]=np.where(delta>1e-4,np.clip(delta/(train["ce0"]+1e-6),0,1),0).astype(np.float32)
    cache=cell_dir/"TRAIN_TEACHER_CACHE.npz"
    save={"subject":train["subject"],"session":train["session"],"trial":train["trial"],
          "y":train["y"],"z0":train["z0"],"p0":train["p0"],"ce0":train["ce0"],"sigma_P":sigma}
    for name,v in train["teacher"].items():
        for key,value in v.items():save[f"{name}_{key}"]=value
    np.savez_compressed(cache,**save)
    cache_sha=sha(cache)
    audit,target_analysis=teacher_audit(task,fold,train,cache_sha)
    # Persist audits before student training, so a failure is diagnosable.
    write_csv(cell_dir/"TRAIN_TEACHER_CACHE_AUDIT.csv",audit)
    write_csv(cell_dir/"TEACHER_TARGET_ANALYSIS.csv",target_analysis)
    disc=collect_role(task,fold,"discovery",train["model"],geo,top3,checkpoint)
    if A.model_state_sha(train["model"])!=source_state:raise RuntimeError("frozen teacher drift")
    rows=metric_rows(task,fold,"ORIGINAL",disc,disc["z0"])
    trajectory=[]; bn_rows=[]; ck_rows=[]; efficiency=[]; models={}
    for arm in ARMS[1:]:
        model,traj,bn,ck,train_seconds=train_arm(task,fold,arm,train,checkpoint,geo,sigma)
        z,eval_seconds=eval_native(model,disc["x"])
        rows.extend(metric_rows(task,fold,arm,disc,z))
        trajectory.extend(traj);bn_rows.append(bn);ck_rows.append(ck)
        efficiency.append({"task":task,"fold":fold,"arm":arm,"train_seconds":train_seconds,
            "discovery_native_inference_seconds":eval_seconds,"discovery_trials":len(disc["y"]),
            "inference_parameter_count":ck["parameter_count"],"inference_forward_count_per_trial":1,
            "inference_PC_module":False,"inference_FLOPs_relation":"same_as_EEGNet"})
        models[arm]=model
        write_csv(cell_dir/"TRAINING_TRAJECTORY.csv",trajectory)
        write_csv(cell_dir/"BN_STATE_AUDIT.csv",bn_rows)
        write_csv(cell_dir/"CHECKPOINT_AUDIT.csv",ck_rows)
        write_csv(cell_dir/"DISCOVERY_SUBJECT_RESULTS.csv",rows)
    original_count=sum(p.numel() for p in train["model"].parameters())
    original_z,t=eval_native(train["model"],disc["x"])
    if np.max(np.abs(original_z-disc["z0"]))>1e-5:raise RuntimeError("original native path mismatch")
    efficiency.append({"task":task,"fold":fold,"arm":"ORIGINAL","train_seconds":0,
        "discovery_native_inference_seconds":t,"discovery_trials":len(disc["y"]),
        "inference_parameter_count":original_count,"inference_forward_count_per_trial":1,
        "inference_PC_module":False,"inference_FLOPs_relation":"EEGNet"})
    # Labels enter the oracle only after all student inference and metrics were saved.
    oracle_rows,align=posthoc_oracle(task,fold,disc,geo,top3,sigma,models)
    if A.model_state_sha(train["model"])!=source_state:raise RuntimeError("teacher drift after posthoc")
    write_csv(cell_dir/"DISCOVERY_ORACLE_POSTHOC.csv",oracle_rows)
    write_csv(cell_dir/"REPRESENTATION_ALIGNMENT.csv",align)
    write_csv(cell_dir/"EFFICIENCY.csv",efficiency)
    write_json(complete,{"task":task,"fold":fold,"protocol_sha256":sha(OUT/"PROTOCOL_LOCK.json"),
        "teacher_state_sha256":source_state,"teacher_state_unchanged":True,
        "teacher_cache_sha256":cache_sha,"students_evaluated_before_posthoc_oracle":True,
        "final_heldout_array_reads":0})
    print("CELL_COMPLETE",task,fold,flush=True)


def finalize_cell(task,fold):
    """Finish a cell whose five epoch-10 checkpoints already exist."""
    preflight_lock()
    c,checkpoint,geompath=paths(task,fold)
    d=RUNTIME/"cells"/task.lower()/f"fold{fold}_seed0"
    ckrows=read_csv(d/"CHECKPOINT_AUDIT.csv")
    if {r["arm"] for r in ckrows}!=set(ARMS[1:]):
        raise RuntimeError("cannot finalize without all five audited checkpoints")
    with np.load(geompath,allow_pickle=False) as g:geo={k:g[k] for k in g.files}
    cache=d/"TRAIN_TEACHER_CACHE.npz"
    with np.load(cache,allow_pickle=False) as z:cached={k:z[k] for k in z.files}
    if "TOP3_w" not in cached:
        for name in TEACHERS:
            delta=cached["ce0"]-cached[name+"_ce"]
            cached[name+"_w"]=np.where(delta>1e-4,np.clip(delta/(cached["ce0"]+1e-6),0,1),0).astype(np.float32)
        tmp=d/"TRAIN_TEACHER_CACHE.repaired.npz"
        np.savez_compressed(tmp,**cached);os.replace(tmp,cache)
        audits=read_csv(d/"TRAIN_TEACHER_CACHE_AUDIT.csv")
        for r in audits:r["cache_sha256"]=sha(cache)
        write_csv(d/"TRAIN_TEACHER_CACHE_AUDIT.csv",audits)
    sigma=cached["sigma_P"]
    disc=collect_role(task,fold,"discovery",None,geo,c["top3_directions_abs_utility_rank"],checkpoint)
    teacher=disc["model"];source_state=A.model_state_sha(teacher)
    z0,original_seconds=eval_native(teacher,disc["x"])
    maxdiff=float(np.max(np.abs(z0-disc["z0"])))
    if maxdiff>=1e-5 or not np.array_equal(z0.argmax(1),disc["z0"].argmax(1)):
        raise RuntimeError(f"original native path mismatch {maxdiff}")
    rows=metric_rows(task,fold,"ORIGINAL",disc,z0)
    models={};eff=[]
    for r in ckrows:
        arm=r["arm"];ckpath=Path(r["checkpoint_runtime_path"])
        if sha(ckpath)!=r["checkpoint_sha256"]:raise RuntimeError("student checkpoint hash mismatch")
        model=model_from_checkpoint(disc["x"].shape,disc["z0"].shape[1],checkpoint)
        model.load_state_dict(torch.load(ckpath,map_location="cpu",weights_only=False)["state_dict"],strict=True)
        z,seconds=eval_native(model,disc["x"])
        rows.extend(metric_rows(task,fold,arm,disc,z))
        eff.append({"task":task,"fold":fold,"arm":arm,
            "train_seconds":"see_original_training_log","discovery_native_inference_seconds":seconds,
            "discovery_trials":len(disc["y"]),"inference_parameter_count":r["parameter_count"],
            "inference_forward_count_per_trial":1,"inference_PC_module":False,
            "inference_FLOPs_relation":"same_as_EEGNet"})
        models[arm]=model
    eff.append({"task":task,"fold":fold,"arm":"ORIGINAL","train_seconds":0,
        "discovery_native_inference_seconds":original_seconds,"discovery_trials":len(disc["y"]),
        "inference_parameter_count":sum(p.numel() for p in teacher.parameters()),
        "inference_forward_count_per_trial":1,"inference_PC_module":False,
        "inference_FLOPs_relation":"EEGNet"})
    write_csv(d/"DISCOVERY_SUBJECT_RESULTS.csv",rows)
    oracle_rows,align=posthoc_oracle(task,fold,disc,geo,c["top3_directions_abs_utility_rank"],sigma,models)
    if A.model_state_sha(teacher)!=source_state:raise RuntimeError("teacher drift")
    write_csv(d/"DISCOVERY_ORACLE_POSTHOC.csv",oracle_rows)
    write_csv(d/"REPRESENTATION_ALIGNMENT.csv",align)
    write_csv(d/"EFFICIENCY.csv",eff)
    write_json(d/"COMPLETE.json",{"task":task,"fold":fold,
        "protocol_sha256":sha(OUT/"PROTOCOL_LOCK.json"),
        "teacher_state_sha256":source_state,"teacher_state_unchanged":True,
        "teacher_cache_sha256":sha(cache),"students_evaluated_before_posthoc_oracle":True,
        "final_heldout_array_reads":0,"native_eval_batch_size":100,
        "max_original_path_logit_difference":maxdiff})
    print("CELL_COMPLETE",task,fold,flush=True)


def biological_subject_values(rows,task,arm,metric,primary=True):
    selected=[r for r in rows if r["task"]==task and r["arm"]==arm]
    by_subject=defaultdict(list)
    if metric=="worst_session_BA":
        by_cell=defaultdict(list)
        for r in selected: by_cell[(r["subject"],r["fold"])].append(float(r["BA"]))
        for (subject,_),values in by_cell.items():by_subject[subject].append(min(values))
    else:
        for r in selected:
            if primary and str(r["is_future_session"]).lower() not in ("true","1"):continue
            by_subject[r["subject"]].append(float(r[metric]))
    return {s:float(np.mean(v)) for s,v in by_subject.items()}


def bootstrap_paired(a,b,seed):
    subjects=sorted(set(a)&set(b),key=int)
    x=np.asarray([a[s]-b[s] for s in subjects],dtype=np.float64)
    if not len(x):raise RuntimeError("empty paired subjects")
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(x),size=(20000,len(x)))
    boot=x[indices].mean(axis=1)
    return len(x),float(x.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def external_oracle_check(task,oracle_rows):
    source={"TOP3":ACTION/"outputs"/"ORACLE_TOP3_RESULTS.csv",
            "MARGIN_UPPER":ACTION/"outputs"/"ORACLE_UPPER_BOUND.csv"}
    checks=[]
    for name,path in source.items():
        old=[r for r in read_csv(path) if r["task"]==task and r["role"]=="discovery"]
        new=[r for r in oracle_rows if r["task"]==task and r["arm"]=="ORACLE_"+name]
        oldmap={(r["fold"],r["subject"],r["session"]):r for r in old}
        err=[]
        for r in new:
            key=(str(r["fold"]),r["subject"],str(r["session"]))
            if key not in oldmap:raise RuntimeError("missing old oracle row "+str(key))
            err.append(abs(float(r["BA"])-float(oldmap[key]["BA"])))
        if not err or max(err)>1e-8:raise RuntimeError(f"source oracle reproduction failure {task}/{name}: {max(err,default=-1)}")
        checks.append({"task":task,"teacher":name,"matched_subject_sessions":len(err),
                       "max_absolute_BA_error":max(err),"source_sha256":sha(path)})
    return checks


def aggregate():
    lock=preflight_lock()
    allrows=[]; oracle=[]; audit=[]; analysis=[]; bn=[];trajectory=[];checkpoint=[];align=[];eff=[];checks=[]
    completed_sources=[]
    for task in TASKS:
        for fold in FOLDS:
            d=RUNTIME/"cells"/task.lower()/f"fold{fold}_seed0"
            done=json.loads((d/"COMPLETE.json").read_text())
            if done["protocol_sha256"]!=sha(OUT/"PROTOCOL_LOCK.json"):
                raise RuntimeError("cell protocol mismatch")
            if not done["teacher_state_unchanged"] or not done["students_evaluated_before_posthoc_oracle"]:
                raise RuntimeError("teacher or posthoc sequencing audit failed")
            if sha(d/"TRAIN_TEACHER_CACHE.npz")!=done["teacher_cache_sha256"]:
                raise RuntimeError(f"teacher cache hash mismatch {task}/{fold}")
            completed_sources.append({"task":task,"fold":fold,"teacher_state_sha256":done["teacher_state_sha256"],
                "teacher_state_unchanged":True,"teacher_cache_sha256":done["teacher_cache_sha256"]})
            c,ck,geometry=paths(task,fold)
            with np.load(geometry,allow_pickle=False) as g:
                norm_hash=hashlib.sha256(g["normalizer_mu"].tobytes()+g["normalizer_sd"].tobytes()).hexdigest()
            if norm_hash!=c["normalizer_sha256"]:
                raise RuntimeError(f"normalizer hash mismatch {task}/{fold}")
            for key,target in (("DISCOVERY_SUBJECT_RESULTS.csv",allrows),
                               ("DISCOVERY_ORACLE_POSTHOC.csv",oracle),
                               ("TRAIN_TEACHER_CACHE_AUDIT.csv",audit),
                               ("TEACHER_TARGET_ANALYSIS.csv",analysis),
                               ("BN_STATE_AUDIT.csv",bn),
                               ("TRAINING_TRAJECTORY.csv",trajectory),
                               ("CHECKPOINT_AUDIT.csv",checkpoint),
                               ("REPRESENTATION_ALIGNMENT.csv",align),
                               ("EFFICIENCY.csv",eff)):
                target.extend(read_csv(d/key))
    if len(checkpoint)!=len(TASKS)*len(FOLDS)*(len(ARMS)-1):
        raise RuntimeError("missing epoch-10 checkpoints")
    for task in TASKS:
        for fold in FOLDS:
            keyrows=[r for r in checkpoint if r["task"]==task and int(r["fold"])==fold]
            if {r["arm"] for r in keyrows}!=set(ARMS[1:]) or len({r["initial_model_state_sha256"] for r in keyrows})!=1:
                raise RuntimeError(f"student initial states or arms differ {task}/{fold}")
            if any(int(r["epoch"])!=10 or str(r["bn_running_state_identical"]).lower()!="true" for r in keyrows):
                raise RuntimeError(f"checkpoint epoch or BN audit failed {task}/{fold}")
            if any(str(r["running_state_byte_identical"]).lower()!="true" for r in bn if r["task"]==task and int(r["fold"])==fold):
                raise RuntimeError(f"BN running state changed {task}/{fold}")
            for arm in ARMS[1:]:
                epochs=sorted(int(r["epoch"]) for r in trajectory if r["task"]==task and int(r["fold"])==fold and r["arm"]==arm)
                if epochs!=list(range(1,11)):raise RuntimeError(f"training trajectory incomplete {task}/{fold}/{arm}")
    for task in TASKS:checks.extend(external_oracle_check(task,oracle))
    source_audit=json.loads((OUT/"TEACHER_SOURCE_AUDIT.json").read_text())
    source_audit["completed_execution_cells"]=completed_sources
    write_json(OUT/"TEACHER_SOURCE_AUDIT.json",source_audit)
    write_csv(OUT/"DISCOVERY_SUBJECT_RESULTS.csv",allrows)
    for name,rows in (("TRAIN_TEACHER_CACHE_AUDIT.csv",audit),("TEACHER_TARGET_ANALYSIS.csv",analysis),
        ("BN_STATE_AUDIT.csv",bn),("TRAINING_TRAJECTORY.csv",trajectory),
        ("CHECKPOINT_AUDIT.csv",checkpoint),("REPRESENTATION_ALIGNMENT.csv",align),
        ("EFFICIENCY.csv",eff)):
        write_csv(OUT/name,rows)
    write_json(OUT/"SOURCE_ORACLE_REPRODUCTION_AUDIT.json",{"checks":checks})
    summary=[];contrast=[];headroom=[]
    comparisons=[("TOP3_DISTILL","CE_ONLY_CONTINUATION"),
        ("MARGIN_UPPER_DISTILL","CE_ONLY_CONTINUATION"),
        ("CE_UPPER_LOGIT_ONLY","CE_ONLY_CONTINUATION"),
        ("CE_UPPER_FULL","CE_ONLY_CONTINUATION"),
        ("CE_UPPER_FULL","CE_UPPER_LOGIT_ONLY"),
        ("CE_UPPER_FULL","MARGIN_UPPER_DISTILL")]
    for task in TASKS:
        for arm in ARMS:
            values={m:biological_subject_values(allrows,task,arm,m) for m in ("BA","macro_F1","NLL")}
            worst=biological_subject_values(allrows,task,arm,"worst_session_BA")
            sessions={}
            for session in sorted({r["session"] for r in allrows if r["task"]==task}):
                sub=defaultdict(list)
                for r in allrows:
                    if r["task"]==task and r["arm"]==arm and r["session"]==session:
                        sub[r["subject"]].append(float(r["BA"]))
                sessions[session]=float(np.mean([np.mean(v) for v in sub.values()]))
            summary.append({"task":task,"arm":arm,"biological_subjects":len(values["BA"]),
                "future_session_subject_equal_BA":float(np.mean(list(values["BA"].values()))),
                "future_session_subject_equal_macro_F1":float(np.mean(list(values["macro_F1"].values()))),
                "future_session_subject_equal_NLL":float(np.mean(list(values["NLL"].values()))),
                "worst_session_subject_equal_BA":float(np.mean(list(worst.values()))),
                "per_session_BA_json":json.dumps(sessions,sort_keys=True)})
        for arm,baseline in comparisons:
            for metric in ("BA","macro_F1","NLL","worst_session_BA"):
                a=biological_subject_values(allrows,task,arm,metric)
                b=biological_subject_values(allrows,task,baseline,metric)
                n,delta,low,high=bootstrap_paired(a,b,stable_seed("bootstrap",task,arm,baseline,metric))
                contrast.append({"task":task,"arm":arm,"comparator":baseline,"metric":metric,
                    "biological_subjects":n,"delta":delta,"ci95_lower":low,"ci95_upper":high,
                    "bootstrap_replicates":20000})
        orig=np.mean(list(biological_subject_values(allrows,task,"ORIGINAL","BA").values()))
        ceonly=np.mean(list(biological_subject_values(allrows,task,"CE_ONLY_CONTINUATION","BA").values()))
        for teacher_name,arm in (("TOP3","TOP3_DISTILL"),("MARGIN_UPPER","MARGIN_UPPER_DISTILL"),
                                 ("CE_UPPER","CE_UPPER_LOGIT_ONLY"),("CE_UPPER","CE_UPPER_FULL")):
            oracle_ba=float(np.mean(list(biological_subject_values(oracle,task,"ORACLE_"+teacher_name,"BA").values())))
            student=float(np.mean(list(biological_subject_values(allrows,task,arm,"BA").values())))
            den=oracle_ba-orig
            headroom.append({"task":task,"teacher":teacher_name,"arm":arm,"oracle_status":"LABEL_PRIVILEGED_TRAINING_ORACLE",
                "original_future_session_BA":orig,"CE_only_future_session_BA":ceonly,
                "oracle_future_session_BA":oracle_ba,"oracle_headroom_BA":den,
                "student_delta_vs_CE_only_BA":student-ceonly,
                "recovered_fraction":(student-ceonly)/den if abs(den)>1e-12 else "undefined_zero_denominator"})
    write_csv(OUT/"DISCOVERY_TASK_SUMMARY.csv",summary)
    write_csv(OUT/"PAIRED_DISCOVERY_CONTRASTS.csv",contrast)
    write_csv(OUT/"DISTILLABLE_HEADROOM.csv",headroom)
    report(lock,summary,contrast,headroom,audit,analysis,align,checks)
    print("AGGREGATE_COMPLETE",flush=True)


def report(lock,summary,contrast,headroom,audit,analysis,align,checks):
    lookup={(r["task"],r["arm"]):r for r in summary}
    ci={(r["task"],r["arm"],r["comparator"],r["metric"]):r for r in contrast}
    rows=[]
    for task in TASKS:
        vals=[float(lookup[(task,arm)]["future_session_subject_equal_BA"]) for arm in ARMS]
        rows.append("| "+task+" | "+" | ".join(f"{v:.4f}" for v in vals)+" |")
    table1="\n".join(rows)
    secondary_lines=[]
    for task in TASKS:
        for arm in ARMS:
            r=lookup[(task,arm)]
            secondary_lines.append(f"| {task} | {arm} | {float(r['future_session_subject_equal_macro_F1']):.4f} | {float(r['future_session_subject_equal_NLL']):.4f} | {float(r['worst_session_subject_equal_BA']):.4f} |")
    secondary_table="\n".join(secondary_lines)
    rows=[]
    for r in headroom:
        task,arm=r["task"],r["arm"]
        c=ci[(task,arm,"CE_ONLY_CONTINUATION","BA")]
        rows.append(f"| {task} | {arm} | {float(c['delta']):+.4f} | [{float(c['ci95_lower']):+.4f}, {float(c['ci95_upper']):+.4f}] | {float(r['oracle_headroom_BA']):+.4f} | {float(r['recovered_fraction']):+.3f} |")
    table2="\n".join(rows)
    rows=[]
    for task in TASKS:
        for teacher in TEACHERS:
            selected=[r for r in audit if r["task"]==task and r["teacher"]==teacher]
            w=np.asarray([float(r["trials"]) for r in selected]);w/=w.sum()
            weighted=lambda key:float(np.dot(w,[float(r[key]) for r in selected]))
            h=next(r for r in headroom if r["task"]==task and r["teacher"]==teacher)
            rows.append(f"| {task} | {teacher} | {weighted('accepted_fraction'):.1%} | {weighted('mean_delta_CE'):+.4f} | {weighted('mean_P_movement'):.4f} | {float(h['oracle_future_session_BA']):.4f} |")
    table3="\n".join(rows)
    target_lines=[]
    for task in TASKS:
        for name in ("NATIVE","CE_UPPER"):
            selected=[r for r in analysis if r["task"]==task and r["teacher"]==name]
            weights=np.asarray([float(r["trials"]) for r in selected]);weights/=weights.sum()
            prob=float(np.dot(weights,[float(r["mean_true_class_probability"]) for r in selected]))
            entropy=float(np.dot(weights,[float(r["mean_entropy"]) for r in selected]))
            target_lines.append(f"| {task} | {name} | {prob:.4f} | {entropy:.4f} |")
    target_table="\n".join(target_lines)
    alignment_lines=[]
    for task in TASKS:
        for arm in ("CE_ONLY_CONTINUATION","CE_UPPER_LOGIT_ONLY","CE_UPPER_FULL"):
            selected=[r for r in align if r["task"]==task and r["arm"]==arm]
            mse=float(np.mean([float(r["standardized_P_MSE"]) for r in selected]))
            direction=float(np.mean([float(r["movement_direction_agreement"]) for r in selected]))
            alignment_lines.append(f"| {task} | {arm} | {mse:.4f} | {direction:+.4f} |")
    alignment_table="\n".join(alignment_lines)
    states=[]
    for task in TASKS:
        full=ci[(task,"CE_UPPER_FULL","CE_ONLY_CONTINUATION","BA")]
        logit=ci[(task,"CE_UPPER_FULL","CE_UPPER_LOGIT_ONLY","BA")]
        full_delta=float(full["delta"]); full_low=float(full["ci95_lower"])
        if full_delta>0 and full_low>0: states.append(f"{task}: PC_ORACLE_DISTILLATION_SUPPORTED")
        else:states.append(f"{task}: PC_ORACLE_DISTILLATION_SUPPORTED not established")
        # Alignment is a separate diagnostic; a positive BA point estimate alone is insufficient.
        grouped=defaultdict(list)
        for r in align:
            if r["task"]==task:grouped[r["arm"]].append(float(r["standardized_P_MSE"]))
        if float(logit["ci95_lower"])>0 and np.mean(grouped["CE_UPPER_FULL"])<np.mean(grouped["CE_UPPER_LOGIT_ONLY"]):
            states.append(f"{task}: P_REPRESENTATION_TARGET_ADDS_VALUE")
        elif abs(float(logit["delta"]))<.005 and float(logit["ci95_lower"])<=0<=float(logit["ci95_upper"]):
            states.append(f"{task}: LOGIT_DISTILLATION_SUFFICIENT is compatible with the observed difference; equivalence is not proven")
        if all(float(ci[(task,arm,"CE_ONLY_CONTINUATION","BA")]["ci95_lower"])<=0 for arm in ARMS[2:]):
            states.append(f"{task}: ORACLE_HEADROOM_NOT_DISTILLABLE under this fixed protocol")
        top=float(ci[(task,"TOP3_DISTILL","CE_ONLY_CONTINUATION","BA")]["delta"])
        upper=max(float(ci[(task,a,"CE_ONLY_CONTINUATION","BA")]["delta"]) for a in ARMS[3:])
        states.append(f"{task}: "+("CONSERVATIVE_ORACLE_MORE_DISTILLABLE" if top>upper else "LARGE_ORACLE_MORE_DISTILLABLE")+" (point estimates only)")
    text=f"""# Counterfactual distillation, seed 0: completed development experiment

## Scope and provenance

Canonical EEGNet, OpenBMI MI and SSVEP, folds 0–4. All five continuations start from the exact frozen checkpoint, use the same 10 epochs, batch order, AdamW settings, and frozen BN running state. Student inference is one native EEGNet forward, with no P/C module. Final heldout and outer dev arrays were never read. Oracle labels are privileged and are not deployable.

The source checkpoint and V3 geometry were previously refit on non-final source-session subjects, including the discovery subjects. Future-session discovery is therefore a development cross-session check, not a fully subject-independent test. No new source checkpoint or geometry was fitted here.

## Primary future-session subject-equal BA

| Task | Original | CE-only continuation | Top3 distill | Margin-Upper distill | CE-Upper logit | CE-Upper full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table1}

### Secondary discovery metrics

| Task | Student | Future-session macro-F1 | Future-session NLL | Worst-session BA |
| --- | --- | ---: | ---: | ---: |
{secondary_table}

All four distillation arms have higher (worse) future-session NLL than CE-only on both tasks. The SSVEP BA gains therefore come with worse probabilistic predictions under this fixed protocol.

## Paired biological-subject BA contrasts and recovered headroom

| Task | Method | ΔBA vs CE-only | 95% paired bootstrap CI | Matched future-session oracle headroom | Recovered fraction |
| --- | --- | ---: | --- | ---: | ---: |
{table2}

The denominator uses matched discovery future-session oracle BA minus matched original BA. It is distinct from the earlier pooled source/future-session headroom. All other paired metrics, including F1, NLL, and worst-session BA, appear in `PAIRED_DISCOVERY_CONTRASTS.csv`.

## Teacher strength

| Task | Teacher | Train accepted % | Mean ΔCE | Mean P movement | Teacher oracle future-session BA |
| --- | --- | ---: | ---: | ---: | ---: |
{table3}

Teacher generation and student training used inner-train labels only. For each trial, KD and P supervision were disabled unless teacher CE improved native CE by more than 1e-4. Old Top3 and margin Upper discovery subject/session BA were reproduced exactly (maximum absolute error {max(c['max_absolute_BA_error'] for c in checks):g}). CE-Upper was evaluated only after every student had completed native discovery inference.

OpenBMI MI has two classes. For two-class softmax, cross-entropy is strictly decreasing in the true-class margin, so CE-Upper and margin Upper select the same direction-wise alpha except for numerical ties. This comparison does not provide an independent teacher objective on MI.

`KL(q_T || onehot(y))` is mathematically infinite for a softmax teacher with nonzero off-class mass. `TEACHER_TARGET_ANALYSIS.csv` records that infinity and reports finite reverse KL, entropy, true-class probability, and KL to native at T=2. A nearly one-hot CE-Upper teacher should be interpreted as extra hard-label pressure, not independent class structure.

At T=2, the observed target sharpness is:

| Task | Target | Mean true-class probability | Mean entropy (nats) |
| --- | --- | ---: | ---: |
{target_table}

## Post-hoc P alignment and incremental value

| Task | Student | Standardized P-MSE to CE-Upper oracle | Movement direction cosine |
| --- | --- | ---: | ---: |
{alignment_table}

This diagnostic does not enter student inference or model selection. On MI, full training slightly lowers P-MSE relative to logit-only but does not improve BA reliably. On SSVEP, full training raises P-MSE relative to logit-only; its small BA point gain therefore does not establish that matching the oracle P representation caused the gain. The full-minus-logit paired BA and NLL intervals are in `PAIRED_DISCOVERY_CONTRASTS.csv`.

## Interpretation

"""+"\n".join("- "+s for s in states)+"\n\nRepresentation distances and movement agreement are post-hoc discovery diagnostics only. The P target is not used during inference. Fixed epoch 10 was used for every arm without discovery selection. Checkpoint binaries and trial teacher caches remain in server runtime and are excluded from publication.\n"
    (OUT/"FINAL_REPORT.md").write_text(text,encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("preflight","cell","finalize","aggregate"))
    p.add_argument("--task",choices=TASKS)
    p.add_argument("--fold",type=int)
    args=p.parse_args()
    if args.mode=="preflight":preflight_lock()
    elif args.mode=="cell":
        if args.task is None or args.fold is None:raise ValueError("cell needs task/fold")
        run_cell(args.task,args.fold)
    elif args.mode=="finalize":
        if args.task is None or args.fold is None:raise ValueError("finalize needs task/fold")
        finalize_cell(args.task,args.fold)
    else:aggregate()


if __name__=="__main__":main()
