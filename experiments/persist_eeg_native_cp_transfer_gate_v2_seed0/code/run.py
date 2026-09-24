"""Frozen EEGNet native complement-to-Protected transfer gate, seed zero."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
V1 = REPO / "experiments" / "persist_eeg_pc_refine_v1_seed0" / "code" / "run.py"
os.environ.setdefault("PC_REFINE_ANCHORS",str(REPO/"anchors"))
SPEC = importlib.util.spec_from_file_location("native_transfer_v1", V1)
assert SPEC and SPEC.loader
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
TASKS = B.TASKS
FOLDS = B.FOLDS
VARIANTS = ("BASELINE", "P_ONLY_TRANSFER_GATE", "RANDOM_TRANSFER_GATE", "PROTECTED_NATIVE_TRANSFER_GATE")
GATES = VARIANTS[1:]
DEVICE = B.DEVICE
ROOT = Path(os.environ.get("NATIVE_GATE_RUNTIME", str(REPO.parent / "native_gate_v2_runtime")))
V1_RUNTIME = Path(os.environ.get("PC_REFINE_RUNTIME", str(REPO.parent / "pc_refine_v1_runtime")))
OUT = EXP / "outputs"
PROT = EXP / "protocol"
torch.set_num_threads(min(12, os.cpu_count() or 1))


def cell(task, fold, stage):
    return ROOT / stage / task.lower() / f"fold{fold}_seed0"


def frozen_bn(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}


def check_bn(model, before):
    if any(not torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items()):
        raise RuntimeError("frozen BN state changed")


def extract(model, x, batch=64):
    spatial, depth, embedding = [], [], []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i+batch])).to(DEVICE)
            s = model.drop1(model.pool1(F.elu(model.bn2(model.spatial(model.bn1(model.temporal(xb.unsqueeze(1))))))))
            d = model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(s))))))
            e = model.embedding(d.flatten(1))
            spatial.append(s.flatten(1).cpu().numpy())
            depth.append(d.flatten(1).cpu().numpy())
            embedding.append(e.cpu().numpy())
    return np.concatenate(spatial), np.concatenate(depth), np.concatenate(embedding)


def fit_bases(model, data, task, fold):
    cap = B.PEEH.capped_indices
    ia = cap(data["ss"], data["sy"], data["source_sessions"][0], task, fold, "train-source")
    ib = cap(data["fs"], data["fy"], data["future_session"], task, fold, "train-future")
    x = np.concatenate((data["source"][ia], data["future"][ib]))
    y = np.concatenate((data["sy"][ia], data["fy"][ib]))
    sub = np.concatenate((data["ss"][ia], data["fs"][ib])).astype(str)
    ses = np.concatenate((np.full(len(ia), data["source_sessions"][0]), np.full(len(ib), data["future_session"])))
    a, d, h = extract(model, x)
    spec = B.PEEH.spectrum(h, y, sub, ses, task, "EEGNet", fold)
    spec["classes"] = data["classes"]
    dims, assignment = B.PEEH.select_protected(h, y, sub, ses, spec, "EEGNet", task, fold)
    if not dims:
        return None, {"status": "UNDEFINED_PROTECTED", "task": task, "fold": fold, "assignment": assignment}
    keys = sorted(set(zip(sub, ses, y)), key=lambda t: (int(str(t[0]).replace("sub-", "")), int(t[1]), int(t[2])))
    mask = [(sub == s) & (ses == j) & (y == lab) for s,j,lab in keys]
    ca = np.stack([a[m].mean(0) for m in mask]).astype(np.float32)
    cd = np.stack([d[m].mean(0) for m in mask]).astype(np.float32)
    ch = np.stack([h[m].mean(0) for m in mask]).astype(np.float32)
    target = B.PEEH.canonical(ch, spec)[:,dims].astype(np.float32)
    qs, ms = B.projector(ca, target)
    qd, md = B.projector(cd, target)
    rng = np.random.default_rng(B.stable_seed("NATIVE_GATE_RANDOM", task, fold, "refit" if data.get("refit") else "discovery"))
    def random_q(n, rank):
        z = rng.standard_normal((n, rank))
        q, _ = np.linalg.qr(z, mode="reduced")
        return q.astype(np.float32)
    rs, rd = random_q(len(ms), qs.shape[1]), random_q(len(md), qd.shape[1])
    basis = {"qs":qs,"qd":qd,"ms":ms,"md":md,"rs":rs,"rd":rd}
    orth = {k:float(np.max(np.abs(basis[k].T @ basis[k] - np.eye(basis[k].shape[1])))) for k in ("qs","qd","rs","rd")}
    if max(orth.values()) >= 1e-5 or qs.shape[1] != rs.shape[1] or qd.shape[1] != rd.shape[1]:
        raise RuntimeError("projector orthogonality or rank failed")
    record = {"status":"COMPLETE","task":task,"fold":fold,"protected_rank":len(dims),
              "protected_coordinates":dims,"canonical_basis_hash":B.arr_sha(spec["mean"],spec["basis"],spec["scale"],spec["directions"]),
              "train_data_hash":B.arr_sha(x,y,sub.astype("U"),ses),"projector_hash":B.arr_sha(qs,ms,qd,md),
              "random_projector_hash":B.arr_sha(rs,ms,rd,md),"spatial_rank":qs.shape[1],"successor_rank":qd.shape[1],
              "orthogonality":orth,"pathway_ridge_alpha":1.0,"assignment":assignment}
    return basis, record


class Gate(nn.Module):
    def __init__(self, rs, rd, use_tau):
        super().__init__()
        self.use_tau = use_tau
        self.lnp = nn.LayerNorm(rs, elementwise_affine=False)
        self.lnt = nn.LayerNorm(rd, elementwise_affine=False)
        self.w1 = nn.Linear(rs + (rd if use_tau else 0), 8)
        self.w2 = nn.Linear(8, rd)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, p, tau):
        u = torch.cat((self.lnp(p), self.lnt(tau)), dim=1) if self.use_tau else self.lnp(p)
        return 1.0 + 0.5*torch.tanh(self.w2(F.gelu(self.w1(u))))


def stage2(model, s):
    return model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(s))))))


def features(model, x, basis, variant, batch=128):
    q_s = torch.from_numpy(basis["rs" if variant == "RANDOM_TRANSFER_GATE" else "qs"]).to(DEVICE)
    q_d = torch.from_numpy(basis["rd" if variant == "RANDOM_TRANSFER_GATE" else "qd"]).to(DEVICE)
    ms = torch.from_numpy(basis["ms"]).to(DEVICE)
    md = torch.from_numpy(basis["md"]).to(DEVICE)
    chunks = {k:[] for k in ("yfull","p","tau","pd","baseline","ponly_logits")}
    model.eval()
    with torch.inference_mode():
        for i in range(0,len(x),batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i+batch])).to(DEVICE)
            s = model.drop1(model.pool1(F.elu(model.bn2(model.spatial(model.bn1(model.temporal(xb.unsqueeze(1))))))))
            sv = s.flatten(1)
            p = (sv-ms) @ q_s
            sp = (ms+p @ q_s.T).reshape_as(s)
            full = stage2(model,s)
            only = stage2(model,sp)
            fv, ov = full.flatten(1), only.flatten(1)
            pd = (fv-md) @ q_d
            tau = (fv-ov) @ q_d
            z = model.head(model.embedding(fv))
            zp = model.head(model.embedding(ov))
            for k,v in (("yfull",fv),("p",p),("tau",tau),("pd",pd),("baseline",z),("ponly_logits",zp)):
                chunks[k].append(v)
    return {k:torch.cat(v,dim=0) for k,v in chunks.items()}


def gate_logits(model, gate, feat, basis, variant, idx=None):
    f = feat if idx is None else {k:v.index_select(0,idx) for k,v in feat.items()}
    g = gate(f["p"],f["tau"])
    qd = torch.from_numpy(basis["rd" if variant == "RANDOM_TRANSFER_GATE" else "qd"]).to(DEVICE)
    correction = (g-1.0)*f["tau"]
    revised = f["yfull"]+correction @ qd.T
    z = model.head(model.embedding(revised))
    return z,g,correction


def make_gate(basis,variant,task,fold):
    B.seed_all(B.stable_seed("NATIVE_GATE_INIT",task,fold))
    return Gate(basis["qs"].shape[1],basis["qd"].shape[1],variant != "P_ONLY_TRANSFER_GATE").to(DEVICE)


def identity(model,gate,feat,basis,variant):
    with torch.inference_mode():
        z,g,c = gate_logits(model,gate,{k:v[:min(128,len(v))] for k,v in feat.items()},basis,variant)
        ref = feat["baseline"][:len(z)]
        diff = float((z-ref).abs().max().cpu())
        pred = bool(torch.equal(z.argmax(1),ref.argmax(1)))
        representation = float(c.abs().max().cpu())
        gate_error = float((g-1).abs().max().cpu())
    if diff >= 1e-6 or representation >= 1e-6 or not pred or gate_error != 0:
        raise RuntimeError(f"zero-init identity failed {variant}: {diff}, {representation}")
    return {"variant":variant,"logits_max_abs_diff":diff,"representations_max_abs_diff":representation,
            "predictions_exact_match":pred,"gate_initial_max_abs_diff":gate_error,"status":"PASS"}


def evaluate(model,gate,feat,basis,variant,y,subjects):
    gate.eval()
    with torch.inference_mode():
        z,g,c = gate_logits(model,gate,feat,basis,variant)
    return B.score(z.cpu().numpy(),y,subjects),z.cpu().numpy(),g.cpu().numpy(),c.cpu().numpy()


def train(model,gate,feat,y,task,fold,basis,variant,epochs,phase,val=None):
    before = frozen_bn(model)
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()
    gate.train()
    opt = torch.optim.AdamW(gate.parameters(),lr=3e-4,weight_decay=5e-4)
    yy = torch.as_tensor(y,dtype=torch.long,device=DEVICE)
    weight = None
    if task == "OpenBMI_ERP":
        counts = np.bincount(y,minlength=int(yy.max().item())+1)
        weight = torch.as_tensor(len(y)/(len(counts)*counts),dtype=torch.float32,device=DEVICE)
    bestkey,beststate,bestepoch = None,None,0
    history=[]
    for epoch in range(1,epochs+1):
        gate.train()
        losses=[]
        for batch in B.batch_order(len(y),task,fold,phase,epoch):
            ix = torch.as_tensor(batch,dtype=torch.long,device=DEVICE)
            opt.zero_grad(set_to_none=True)
            z,g,_ = gate_logits(model,gate,feat,basis,variant,ix)
            loss = F.cross_entropy(z,yy.index_select(0,ix),weight=weight) + 1e-3*(g-1).square().sum(1).mean()
            if not torch.isfinite(loss): raise RuntimeError("nonfinite gate loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if val is not None:
            met,_,_,_ = evaluate(model,gate,val[0],basis,variant,val[1],val[2])
            key = (-met["BA"],met["NLL"],epoch)
            if bestkey is None or key < bestkey:
                bestkey,bestepoch,beststate = key,epoch,copy.deepcopy({k:v.detach().cpu() for k,v in gate.state_dict().items()})
        else:
            met={}
        row={"task":task,"fold":fold,"phase":phase,"variant":variant,"epoch":epoch,"loss":float(np.mean(losses)),**met}
        history.append(row)
        print("GATE_EPOCH",task,fold,phase,variant,epoch,row["loss"],met.get("BA"),flush=True)
    if val is not None: gate.load_state_dict(beststate)
    else: bestepoch=epochs
    check_bn(model,before)
    if any(p.requires_grad for p in model.parameters()): raise RuntimeError("EEGNet unexpectedly trainable")
    return gate,bestepoch,history


def save_gate(path,gate,epoch):
    B.torch_write(path,{"state_dict":{k:v.detach().cpu() for k,v in gate.state_dict().items()},"selected_epoch_budget":epoch})


def load_gate(path,basis,variant,task,fold):
    gate=make_gate(basis,variant,task,fold)
    gate.load_state_dict(torch.load(path,map_location="cpu",weights_only=False)["state_dict"],strict=True)
    gate.eval()
    return gate


def protocol_lock():
    path=PROT/"PROTOCOL_LOCK.json"
    if path.exists(): raise RuntimeError("protocol already locked")
    record={"schema":"NATIVE_CP_TRANSFER_GATE_V2_SEED0","tasks":TASKS,"folds":list(FOLDS),"seed":0,
            "variants":VARIANTS,"zone":"spatial_elu_pool1_after_drop1 -> depth_point_elu_pool2_after_drop2",
            "pathway_ridge_alpha":1.0,"random":"seeded Gaussian QR; exactly rank matched; one draw per cell/stage",
            "gate":{"hidden_dim":8,"activation":"GELU","range":[0.5,1.5],"output_zero_init":True,
                    "layer_norm_affine":False,"trainable":"W1 and W2 only"},
            "training":{"optimizer":"AdamW","lr":3e-4,"weight_decay":5e-4,"epochs_max":20,
                        "batch_size":128,"trust_lambda":1e-3,"selection":"subject equal BA, lower NLL, earlier epoch",
                        "batch_order":"same within task/fold/phase/epoch across all gate variants"},
            "baseline":"V1 verified canonical seed0 anchor for discovery; verified canonical V1 refit baseline for final",
            "final":"five-fold probability mean; biological-subject equal metrics; 20000 paired bootstrap draws",
            "heldout_accessed":False,"code_sha256":B.sha(Path(__file__)),"v1_utility_sha256":B.sha(V1),
            "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    B.json_write(path,record)
    (PROT/"PROTOCOL_LOCK.sha256").write_text(B.sha(path)+"\n",encoding="utf-8")


def preflight():
    audits=[]
    for task in TASKS:
        for fold in FOLDS:
            data=B.development(task,fold)
            _,rec,anchor=B.load_anchor(task,fold,data)
            ref=V1_RUNTIME/"refit"/task.lower()/f"fold{fold}_seed0"
            record=json.loads((ref/"COMPLETE.json").read_text())
            if record["status"]!="COMPLETE" or record["split_sha256"]!=data["split"]:
                raise RuntimeError("V1 refit record mismatch")
            if B.sha(ref/"BASELINE.pt")!=record["checkpoints"]["BASELINE"]:
                raise RuntimeError("V1 refit checkpoint hash mismatch")
            audits.append({"task":task,"fold":fold,"anchor_sha256":B.sha(anchor),
                           "v1_refit_baseline_sha256":B.sha(ref/"BASELINE.pt"),
                           "split_sha256":data["split"],"discovery_normalizer_sha256":data["normalizer"]["mean_std_sha256"],
                           "canonical_selected_epoch":rec["selected_epoch"]})
            print("PREFLIGHT",task,fold,flush=True)
    B.csv_write(OUT/"BASELINE_CHECKPOINT_AUDIT.csv",audits)
    protocol_lock()


def discover(task,fold):
    target=cell(task,fold,"discovery")
    if (target/"COMPLETE.json").exists(): return
    data=B.development(task,fold)
    model,rec,anchor=B.load_anchor(task,fold,data)
    basis,audit=fit_bases(model,data,task,fold)
    B.json_write(target/"projector.json",audit)
    if basis is None:
        B.json_write(target/"COMPLETE.json",{"status":"UNDEFINED_PROTECTED","task":task,"fold":fold})
        return
    np.savez_compressed(target/"projectors.npz",**basis)
    before=frozen_bn(model)
    baseval=B.score(B.logits(model,data["val"]),data["vy"],data["vs"])
    results=[{"task":task,"fold":fold,"variant":"BASELINE","selected_epoch":0,**baseval}]
    ids=[]; histories=[]
    trainfeat=features(model,data["source"],basis,"PROTECTED_NATIVE_TRANSFER_GATE")
    valfeat=features(model,data["val"],basis,"PROTECTED_NATIVE_TRANSFER_GATE")
    for variant in GATES:
        if variant=="RANDOM_TRANSFER_GATE":
            tr=features(model,data["source"],basis,variant)
            va=features(model,data["val"],basis,variant)
        else:
            tr,va=trainfeat,valfeat
        gate=make_gate(basis,variant,task,fold)
        ids.append({"task":task,"fold":fold,"stage":"discovery",**identity(model,gate,tr,basis,variant)})
        gate,epoch,history=train(model,gate,tr,data["sy"],task,fold,basis,variant,20,1,(va,data["vy"],data["vs"]))
        met,_,_,_=evaluate(model,gate,va,basis,variant,data["vy"],data["vs"])
        ck=target/f"{variant}.pt"
        save_gate(ck,gate,epoch)
        results.append({"task":task,"fold":fold,"variant":variant,"selected_epoch":epoch,"checkpoint_sha256":B.sha(ck),**met})
        histories.extend(history)
        del tr,va,gate
    check_bn(model,before)
    B.csv_write(target/"IDENTITY.csv",ids)
    B.csv_write(target/"RESULTS.csv",results)
    B.csv_write(target/"TRAINING.csv",histories)
    B.json_write(target/"COMPLETE.json",{"status":"COMPLETE","task":task,"fold":fold,
                                          "anchor_sha256":B.sha(anchor),"projectors_sha256":B.sha(target/"projectors.npz")})


def refit(task,fold):
    target=cell(task,fold,"refit")
    if (target/"COMPLETE.json").exists(): return
    source=cell(task,fold,"discovery")
    record=json.loads((source/"COMPLETE.json").read_text())
    if record["status"]!="COMPLETE":
        B.json_write(target/"COMPLETE.json",{"status":"UNDEFINED_PROTECTED","task":task,"fold":fold})
        return
    data=B.development(task,fold,refit=True)
    data["refit"]=True
    old=V1_RUNTIME/"refit"/task.lower()/f"fold{fold}_seed0"
    oldrecord=json.loads((old/"COMPLETE.json").read_text())
    if oldrecord["status"]!="COMPLETE" or oldrecord["normalizer_sha256"]!=data["normalizer"]["mean_std_sha256"]:
        raise RuntimeError("canonical V1 refit baseline normalizer mismatch")
    checkpoint=old/"BASELINE.pt"
    if B.sha(checkpoint)!=oldrecord["checkpoints"]["BASELINE"]:
        raise RuntimeError("canonical V1 refit baseline checkpoint mismatch")
    model=B.eegnet(data)
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["state_dict"],strict=True)
    model.eval()
    basis,audit=fit_bases(model,data,task,fold)
    B.json_write(target/"projector.json",audit)
    if basis is None:
        B.json_write(target/"COMPLETE.json",{"status":"UNDEFINED_PROTECTED_REFIT","task":task,"fold":fold})
        return
    np.savez_compressed(target/"projectors.npz",**basis)
    selected={r["variant"]:int(r["selected_epoch"]) for r in csv.DictReader((source/"RESULTS.csv").open(newline="",encoding="utf-8"))}
    before=frozen_bn(model)
    trp=features(model,data["source"],basis,"PROTECTED_NATIVE_TRANSFER_GATE")
    histories=[]; ids=[]; ckhash={"BASELINE":B.sha(checkpoint)}
    for variant in GATES:
        tr=features(model,data["source"],basis,variant) if variant=="RANDOM_TRANSFER_GATE" else trp
        gate=make_gate(basis,variant,task,fold)
        ids.append({"task":task,"fold":fold,"stage":"refit",**identity(model,gate,tr,basis,variant)})
        gate,epoch,history=train(model,gate,tr,data["sy"],task,fold,basis,variant,selected[variant],2)
        ck=target/f"{variant}.pt"
        save_gate(ck,gate,epoch)
        ckhash[variant]=B.sha(ck)
        histories.extend(history)
        del tr,gate
    check_bn(model,before)
    B.csv_write(target/"IDENTITY.csv",ids)
    B.csv_write(target/"TRAINING.csv",histories)
    B.json_write(target/"COMPLETE.json",{"status":"COMPLETE","task":task,"fold":fold,
                                          "normalizer_sha256":data["normalizer"]["mean_std_sha256"],
                                          "split_sha256":data["split"],"baseline_checkpoint":str(checkpoint),
                                          "selected_epochs":{v:selected[v] for v in GATES},"checkpoints":ckhash,
                                          "projectors_sha256":B.sha(target/"projectors.npz"),
                                          "projector_hash":audit["projector_hash"],"random_projector_hash":audit["random_projector_hash"]})


def consolidate():
    projectors=[]; random=[]; identities=[]; discovery=[]; training=[]; efficiency=[]
    anchors={(r["task"],int(r["fold"])):r for r in csv.DictReader((OUT/"BASELINE_CHECKPOINT_AUDIT.csv").open(newline="",encoding="utf-8"))}
    for task in TASKS:
        for fold in FOLDS:
            for stage in ("discovery","refit"):
                d=cell(task,fold,stage)
                complete=json.loads((d/"COMPLETE.json").read_text())
                if complete["status"]!="COMPLETE": raise RuntimeError(f"fail closed: {task}/{fold}/{stage}: {complete['status']}")
                audit=json.loads((d/"projector.json").read_text())
                arr=np.load(d/"projectors.npz",allow_pickle=False)
                if B.sha(d/"projectors.npz")!=complete["projectors_sha256"] or audit["projector_hash"]!=B.arr_sha(arr["qs"],arr["ms"],arr["qd"],arr["md"]):
                    raise RuntimeError("projector hash mismatch")
                if audit["random_projector_hash"]!=B.arr_sha(arr["rs"],arr["ms"],arr["rd"],arr["md"]):
                    raise RuntimeError("random projector hash mismatch")
                common={"task":task,"fold":fold,"stage":stage,"status":"COMPLETE","protected_rank":audit["protected_rank"],
                        "spatial_rank":audit["spatial_rank"],"successor_rank":audit["successor_rank"],
                        "canonical_basis_hash":audit["canonical_basis_hash"],"train_data_hash":audit["train_data_hash"]}
                projectors.append({**common,"projector_hash":audit["projector_hash"],"spatial_orth_error":audit["orthogonality"]["qs"],"successor_orth_error":audit["orthogonality"]["qd"]})
                random.append({**common,"random_projector_hash":audit["random_projector_hash"],"spatial_orth_error":audit["orthogonality"]["rs"],"successor_orth_error":audit["orthogonality"]["rd"]})
                identities += list(csv.DictReader((d/"IDENTITY.csv").open(newline="",encoding="utf-8")))
                training += list(csv.DictReader((d/"TRAINING.csv").open(newline="",encoding="utf-8")))
                if stage=="discovery": discovery += list(csv.DictReader((d/"RESULTS.csv").open(newline="",encoding="utf-8")))
            base=B.development(task,fold,refit=True)
            model=B.eegnet(base)
            nbase=sum(p.numel() for p in model.parameters())
            a=anchors[task,fold]
            if a["v1_refit_baseline_sha256"]!=complete["checkpoints"]["BASELINE"]:
                raise RuntimeError("baseline changed after preflight")
            rs=int(audit["spatial_rank"]); rd=int(audit["successor_rank"])
            spatial=16*(base["samples"]//4); successor=16*(base["samples"]//4//8)
            f0_macs=16*16*(base["samples"]//4)+16*16*(base["samples"]//4)
            baseline_macs=(8*64*base["channels"]*base["samples"] +
                           16*base["channels"]*base["samples"] + f0_macs +
                           successor*64 + 64*base["classes"])
            for variant in VARIANTS:
                gateparams=0 if variant=="BASELINE" else (rs+(0 if variant=="P_ONLY_TRANSFER_GATE" else rd))*8+8+8*rd+rd
                inputmac=0 if variant=="BASELINE" else spatial*rs+successor*rd+successor*rd+8*(rs+(0 if variant=="P_ONLY_TRANSFER_GATE" else rd))+8*rd+successor*rd
                efficiency.append({"task":task,"fold":fold,"variant":variant,"baseline_params":nbase,
                                   "trainable_params":gateparams,"total_params":nbase+gateparams,
                                   "baseline_MACs_estimate":baseline_macs,
                                   "extra_MACs_estimate":inputmac+(0 if variant=="BASELINE" else f0_macs),
                                   "total_MACs_estimate":baseline_macs+inputmac+(0 if variant=="BASELINE" else f0_macs),
                                   "extra_F0_forwards":0 if variant=="BASELINE" else 1,
                                   "full_F0_forwards":1,"MAC_definition":"conv/linear multiply accumulate only, estimated; excludes norm/activation"})
    if len(identities)!=len(TASKS)*len(FOLDS)*2*len(GATES): raise RuntimeError("identity audit incomplete")
    if any(float(r["logits_max_abs_diff"])>=1e-6 or float(r["representations_max_abs_diff"])>=1e-6 or r["predictions_exact_match"]!="True" for r in identities):
        raise RuntimeError("identity audit failed")
    for name,rows in (("PROJECTOR_AUDIT.csv",projectors),("RANDOM_PROJECTOR_AUDIT.csv",random),
                      ("ZERO_INIT_IDENTITY_AUDIT.csv",identities),("DISCOVERY_RESULTS.csv",discovery),
                      ("TRAINING_AUDIT.csv",training),("EFFICIENCY.csv",efficiency)):
        B.csv_write(OUT/name,rows)


def lock():
    if (PROT/"FINAL_EVAL_LOCK.json").exists(): raise RuntimeError("final lock already exists")
    consolidate()
    cells=[]
    for task in TASKS:
        for fold in FOLDS:
            d=cell(task,fold,"refit")
            c=json.loads((d/"COMPLETE.json").read_text())
            if c["status"]!="COMPLETE": raise RuntimeError("incomplete refit")
            for v in GATES:
                if B.sha(d/f"{v}.pt")!=c["checkpoints"][v]: raise RuntimeError("gate checkpoint changed")
            if B.sha(Path(c["baseline_checkpoint"]))!=c["checkpoints"]["BASELINE"]: raise RuntimeError("baseline checkpoint changed")
            cells.append(c)
    manifest=json.loads((REPO/"experiments"/"persist_eeg_final_heldout_confirmation_v1"/"protocol"/"FINAL_HOLDOUT_MANIFEST.json").read_text())
    ids=sorted(map(str,manifest["OpenBMI"]["subject_ids"]),key=int)
    if len(ids)!=14 or len(B.TRUE_WBCIC)!=10: raise RuntimeError("heldout metadata cohort mismatch")
    dependencies=[V1,B.SEVEN_CODE/"backbone_models.py",B.SEVEN_CODE/"tech_recipe_selection.py",
                  REPO/"experiments"/"persist_eeg_outcome_blind_modern_backbone_seed0_v1"/"code"/"modern_common.py",
                  REPO/"experiments"/"persist_eeg_openbmi_task_generality_v1"/"code"/"task_datasets.py",
                  REPO/"experiments"/"persist_eeg_crossbackbone_peeh_v1"/"code"/"run_crossbackbone_peeh.py",
                  REPO/"experiments"/"persist_eeg_baseline_metrics_closure_v1"/"code"/"run_frozen_sessions.py"]
    record={"schema":"NATIVE_CP_TRANSFER_GATE_V2_FINAL_EVAL_LOCK", "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "source_commit":os.environ.get("NATIVE_GATE_SOURCE_COMMIT","UNAVAILABLE"),
            "code_sha256":B.sha(Path(__file__)),"dependency_hashes":{str(p.relative_to(REPO)):B.sha(p) for p in dependencies},
            "protocol_sha256":B.sha(PROT/"PROTOCOL_LOCK.json"),"cells":cells,
            "hyperparameters":json.loads((PROT/"PROTOCOL_LOCK.json").read_text())["training"],
            "metric_definitions":{"primary":"future physical S2 subject equal BA after five-fold probability mean",
                                  "macro_F1":"subject equal future S2","worst_session_BA":"subject equal minimum per physical session",
                                  "NLL":"subject equal future S2","bootstrap":"20000 paired biological subject resamples"},
            "heldout_subjects":{"OpenBMI":ids,"WBCIC_true_outer":list(B.TRUE_WBCIC)}}
    path=PROT/"FINAL_EVAL_LOCK.json"
    B.json_write(path,record)
    (PROT/"FINAL_EVAL_LOCK.sha256").write_text(B.sha(path)+"\n",encoding="utf-8")
    print("FINAL_LOCKED",B.sha(path),flush=True)


def check_lock():
    path=PROT/"FINAL_EVAL_LOCK.json"
    if not path.exists() or B.sha(path)!=(PROT/"FINAL_EVAL_LOCK.sha256").read_text().strip(): raise RuntimeError("final lock missing/changed")
    rec=json.loads(path.read_text())
    if rec["code_sha256"]!=B.sha(Path(__file__)) or rec["protocol_sha256"]!=B.sha(PROT/"PROTOCOL_LOCK.json"):
        raise RuntimeError("source/protocol changed after final lock")
    for name,expected in rec["dependency_hashes"].items():
        if B.sha(REPO/name)!=expected: raise RuntimeError("dependency changed after final lock")
    for c in rec["cells"]:
        d=cell(c["task"],c["fold"],"refit")
        if B.sha(d/"projectors.npz")!=c["projectors_sha256"]: raise RuntimeError("projectors changed after final lock")
        for v in GATES:
            if B.sha(d/f"{v}.pt")!=c["checkpoints"][v]: raise RuntimeError("gate changed after final lock")
        if B.sha(Path(c["baseline_checkpoint"]))!=c["checkpoints"]["BASELINE"]: raise RuntimeError("baseline changed after final lock")
    return rec


def final_eval(task,fold):
    lockrec=check_lock() # This must precede any heldout EEG array read.
    c=next(x for x in lockrec["cells"] if x["task"]==task and x["fold"]==fold)
    target=cell(task,fold,"heldout")
    if (target/"COMPLETE.json").exists(): return
    data=B.development(task,fold,refit=True)
    if data["normalizer"]["mean_std_sha256"]!=c["normalizer_sha256"]: raise RuntimeError("normalizer changed")
    ids=lockrec["heldout_subjects"]["WBCIC_true_outer" if task=="WBCIC_MI" else "OpenBMI"]
    if set(ids)&set(data["subjects"]): raise RuntimeError("development/heldout overlap")
    model=B.eegnet(data)
    model.load_state_dict(torch.load(c["baseline_checkpoint"],map_location="cpu",weights_only=False)["state_dict"],strict=True)
    model.eval()
    arr=np.load(cell(task,fold,"refit")/"projectors.npz",allow_pickle=False)
    basis={k:arr[k] for k in arr.files}
    gates={v:load_gate(cell(task,fold,"refit")/f"{v}.pt",basis,v,task,fold) for v in GATES}
    payload={}; audit=[]
    for session in ((0,1,2) if task=="WBCIC_MI" else (1,2)):
        raw,y,sub,_=B.rows(task,ids,(session,),data["cache_name"],data["mapping"],final=(task=="WBCIC_MI"))
        x=((raw-data["mu"][None,:,None])/np.maximum(data["sd"][None,:,None],1e-6)).astype(np.float32)
        del raw
        pf=features(model,x,basis,"PROTECTED_NATIVE_TRANSFER_GATE")
        rf=features(model,x,basis,"RANDOM_TRANSFER_GATE")
        payload[f"S{session}_y"]=y.astype(np.int64)
        payload[f"S{session}_subjects"]=sub.astype("U")
        payload[f"BASELINE_S{session}_z"]=pf["baseline"].cpu().numpy()
        for v in GATES:
            ft=rf if v=="RANDOM_TRANSFER_GATE" else pf
            _,z,g,corr=evaluate(model,gates[v],ft,basis,v,y.astype(np.int64),sub.astype(str))
            tau=ft["tau"].cpu().numpy(); pd=ft["pd"].cpu().numpy()
            metrics={"tau_norm":np.linalg.norm(tau,axis=1),"intervention_norm":np.linalg.norm(corr,axis=1),
                     "relative_intervention":np.linalg.norm(corr,axis=1)/np.maximum(np.linalg.norm(pd,axis=1),1e-6),
                     "gate_mean":g.mean(axis=1),"gate_low_fraction":(g<0.9).mean(axis=1),
                     "gate_high_fraction":(g>1.1).mean(axis=1),
                     "direction_dot_pfull":np.sum(corr*pd,axis=1)/np.maximum(np.linalg.norm(corr,axis=1)*np.linalg.norm(pd,axis=1),1e-6),
                     "pc_disagreement":(ft["baseline"].argmax(1)!=ft["ponly_logits"].argmax(1)).cpu().numpy().astype(np.float32)}
            payload[f"{v}_S{session}_z"]=z
            for name,value in metrics.items(): payload[f"{v}_S{session}_{name}"]=value.astype(np.float32)
            for j in range(g.shape[1]):
                audit.append({"task":task,"fold":fold,"session":f"S{session}","variant":v,"dimension":j,
                              "mean_gate":float(g[:,j].mean()),"std_gate":float(g[:,j].std()),
                              "Pr_gate_lt_0p9":float((g[:,j]<0.9).mean()),"Pr_gate_gt_1p1":float((g[:,j]>1.1).mean()),
                              "label":"POST_HOC_HELDOUT_DIAGNOSTIC"})
            audit.append({"task":task,"fold":fold,"session":f"S{session}","variant":v,"dimension":"ALL",
                          "mean_gate":float(g.mean()),"std_gate":float(g.std()),
                          "Pr_gate_lt_0p9":float((g<0.9).mean()),"Pr_gate_gt_1p1":float((g>1.1).mean()),
                          "label":"POST_HOC_HELDOUT_DIAGNOSTIC"})
        print("HELDOUT_SESSION",task,fold,session,len(x),flush=True)
        del x,pf,rf
    target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/"predictions.npz",**payload)
    B.csv_write(target/"GATE_DIMENSIONS.csv",audit)
    B.json_write(target/"COMPLETE.json",{"status":"COMPLETE","task":task,"fold":fold,
                                           "predictions_sha256":B.sha(target/"predictions.npz"),
                                           "gate_dimensions_sha256":B.sha(target/"GATE_DIMENSIONS.csv"),
                                           "final_lock_sha256":B.sha(PROT/"FINAL_EVAL_LOCK.json")})


def aggregate():
    check_lock()
    subject_rows=[]; summary=[]; sessionsummary=[]; contrasts=[]; audit=[]; rescue=[]
    diagkeys=("tau_norm","intervention_norm","relative_intervention","gate_mean","gate_low_fraction",
              "gate_high_fraction","direction_dot_pfull","pc_disagreement")
    for task in TASKS:
        parts=[]
        for fold in FOLDS:
            target=cell(task,fold,"heldout")
            c=json.loads((target/"COMPLETE.json").read_text())
            if c["status"]!="COMPLETE" or B.sha(target/"predictions.npz")!=c["predictions_sha256"] or B.sha(target/"GATE_DIMENSIONS.csv")!=c["gate_dimensions_sha256"]:
                raise RuntimeError("heldout cell incomplete/changed")
            parts.append(np.load(target/"predictions.npz",allow_pickle=False))
            audit+=list(csv.DictReader((target/"GATE_DIMENSIONS.csv").open(newline="",encoding="utf-8")))
        by={}
        for session in ((0,1,2) if task=="WBCIC_MI" else (1,2)):
            y=parts[0][f"S{session}_y"]
            sub=parts[0][f"S{session}_subjects"].astype(str)
            for p in parts[1:]:
                if not np.array_equal(y,p[f"S{session}_y"]) or not np.array_equal(sub,p[f"S{session}_subjects"].astype(str)):
                    raise RuntimeError("fold heldout trial order mismatch")
            probs={}
            for variant in VARIANTS:
                z=np.stack([p[f"{variant}_S{session}_z"] for p in parts])
                z=z-z.max(axis=2,keepdims=True)
                pr=np.exp(z); pr/=pr.sum(axis=2,keepdims=True)
                probs[variant]=pr.mean(0)
            for s in sorted(set(sub),key=lambda t:int(t.replace("sub-",""))):
                ix=sub==s
                for v in VARIANTS:
                    prob=probs[v][ix]
                    pred=prob.argmax(1)
                    row={"task":task,"subject":s,"session":f"S{session}","variant":v,
                         "BA":float(balanced_accuracy_score(y[ix],pred)),
                         "macro_F1":float(f1_score(y[ix],pred,average="macro",zero_division=0)),
                         "NLL":float(-np.log(np.clip(prob[np.arange(len(prob)),y[ix]],1e-12,1)).mean()),
                         "trials":int(ix.sum())}
                    subject_rows.append(row); by[s,session,v]=row
                bp=probs["BASELINE"][ix].argmax(1)
                for v in GATES:
                    vp=probs[v][ix].argmax(1)
                    rescued=(bp!=y[ix])&(vp==y[ix]); damaged=(bp==y[ix])&(vp!=y[ix])
                    record={"task":task,"subject":s,"session":f"S{session}","variant":v,
                            "baseline_errors_rescued":int(rescued.sum()),"baseline_correct_damaged":int(damaged.sum()),
                            "trials":int(ix.sum()),"label":"POST_HOC_HELDOUT_DIAGNOSTIC"}
                    for key in diagkeys:
                        values=np.mean([p[f"{v}_S{session}_{key}"][ix] for p in parts],axis=0)
                        record[key+"_all_mean"]=float(values.mean())
                        record[key+"_rescued_mean"]=float(values[rescued].mean()) if rescued.any() else ""
                        record[key+"_damaged_mean"]=float(values[damaged].mean()) if damaged.any() else ""
                    rescue.append(record)
            for v in VARIANTS:
                rr=[r for r in subject_rows if r["task"]==task and r["session"]==f"S{session}" and r["variant"]==v]
                sessionsummary.append({"task":task,"session":f"S{session}","variant":v,"subjects":len(rr),
                                       **{m:float(np.mean([r[m] for r in rr])) for m in ("BA","macro_F1","NLL")}})
        subjects=sorted({s for s,_,_ in by},key=lambda t:int(t.replace("sub-","")))
        if len(subjects)!=(10 if task=="WBCIC_MI" else 14): raise RuntimeError("heldout biological subject count mismatch")
        primary=2
        vectors={}
        for v in VARIANTS:
            for m in ("BA","macro_F1","NLL"):
                vectors[v,m]=np.asarray([by[s,primary,v][m] for s in subjects])
            vectors[v,"worst_session_BA"]=np.asarray([min(by[s,session,v]["BA"] for session in ((0,1,2) if task=="WBCIC_MI" else (1,2))) for s in subjects])
            summary.append({"task":task,"variant":v,"subjects":len(subjects),"primary_session":"S2",
                            **{m:float(vectors[v,m].mean()) for m in ("BA","macro_F1","worst_session_BA","NLL")}})
        main="PROTECTED_NATIVE_TRANSFER_GATE"
        for comparator in VARIANTS[:-1]:
            for metric in ("BA","macro_F1","worst_session_BA"):
                delta,lo,hi=B.paired_ci(vectors[main,metric],vectors[comparator,metric],(task,comparator,metric,"NATIVE_GATE_V2"))
                contrasts.append({"task":task,"contrast":main+" - "+comparator,"metric":metric,
                                  "difference":delta,"CI95_low":lo,"CI95_high":hi,"bootstrap_draws":20000,
                                  "unit":"biological subject"})
        for p in parts:p.close()
    for name,rows in (("HELDOUT_SUBJECT_RESULTS.csv",subject_rows),("HELDOUT_MODEL_TASK_SUMMARY.csv",summary),
                      ("HELDOUT_SESSION_SUMMARY.csv",sessionsummary),("PAIRED_HELDOUT_CONTRASTS.csv",contrasts),
                      ("TRANSFER_GATE_AUDIT.csv",audit),("HELDOUT_RESCUE_HARM.csv",rescue)):
        B.csv_write(OUT/name,rows)
    table={(r["task"],r["variant"]):r for r in summary}
    lines=["# Native C-to-P transfer gate V2, seed 0", "", "Formal heldout; five-fold probability mean; subject-equal future physical S2 BA.",
           "", "| Task | Baseline | P-only Gate | Random Gate | Protected Native Gate | Delta vs Base | Delta vs Random |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for task in TASKS:
        b,p,r,g=[table[task,v]["BA"] for v in VARIANTS]
        lines.append(f"| {task} | {b:.4f} | {p:.4f} | {r:.4f} | {g:.4f} | {g-b:+.4f} | {g-r:+.4f} |")
    lines += ["", "## Paired biological-subject bootstrap", "", "20,000 draws; absolute metric differences.", "",
              "| Task | Contrast | Metric | Difference | 95% CI |", "| --- | --- | --- | ---: | ---: |"]
    for row in contrasts:
        lines.append(f"| {row['task']} | {row['contrast']} | {row['metric']} | {row['difference']:+.4f} | [{row['CI95_low']:+.4f}, {row['CI95_high']:+.4f}] |")
    lines += ["", "## Interpretation", ""]
    for task in TASKS:
        b,p,r,g=[table[task,v]["BA"] for v in VARIANTS]
        vsr=next(x for x in contrasts if x["task"]==task and x["contrast"].endswith("RANDOM_TRANSFER_GATE") and x["metric"]=="BA")
        vsb=next(x for x in contrasts if x["task"]==task and x["contrast"].endswith("BASELINE") and x["metric"]=="BA")
        if vsb["CI95_low"]>0 and vsr["CI95_low"]>0: state="PROTECTED_TRANSFER_SPECIFIC_GAIN"
        elif g>b and abs(g-r)<0.005: state="GENERIC_NATIVE_TRANSFER_CONTROL_GAIN"
        elif abs(g-p)<0.005: state="TRANSFER_STATE_NOT_CONTEXT_SPECIFIC"
        elif g<=b and r<=b and p<=b: state="NATIVE_TRANSFER_NOT_ACTIONABLE"
        else: state="TASK_DEPENDENT_NATIVE_TRANSFER_GAIN"
        lines.append(f"- {task}: {state}. Protected minus Random BA {g-r:+.4f}; paired 95% CI [{vsr['CI95_low']:+.4f}, {vsr['CI95_high']:+.4f}].")
    lines += ["", "These states describe seed-zero results only. The 0.005 equivalence heuristic is descriptive, not an equivalence test.",
              "All gate, transfer, and rescue/harm results are POST_HOC_HELDOUT_DIAGNOSTIC and were not used to select models.",
              "The gate costs one additional frozen F0 forward per trial; see EFFICIENCY.csv."]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("preflight","discover","refit","lock","final-eval","aggregate"))
    parser.add_argument("--task",choices=TASKS)
    parser.add_argument("--fold",type=int,choices=FOLDS)
    args=parser.parse_args()
    if args.stage in ("discover","refit","final-eval") and (args.task is None or args.fold is None):
        parser.error("task and fold required")
    {"preflight":preflight,"discover":lambda:discover(args.task,args.fold),
     "refit":lambda:refit(args.task,args.fold),"lock":lock,
     "final-eval":lambda:final_eval(args.task,args.fold),"aggregate":aggregate}[args.stage]()


if __name__=="__main__":main()
