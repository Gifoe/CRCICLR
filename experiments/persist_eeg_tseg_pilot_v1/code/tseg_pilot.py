"""Small Route-B pilot for trajectory-stable episodic generalization (TSEG).

Only OpenBMI/WBCIC outer fold 0, EEGNet, and optimization seeds 0/1/2 are
run.  All checkpoint selection is source-validation-only.  B2 and B3 share
the exact same deterministic episode/trajectory streams; only the objective
differs.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(sys.argv[1]).resolve()
BASE_ROOT = Path(sys.argv[2]).resolve()
DEVICE = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cuda:0")
ROOT.mkdir(parents=True, exist_ok=True)
AUDIT_ROOT = ROOT.parent / "persist_eeg_route_b_randomness_audit_v1"
RA_PATH = AUDIT_ROOT / "code" / "run_randomness_audit.py"
spec = importlib.util.spec_from_file_location("route_b_randomness_audit_tseg", RA_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {RA_PATH}")
ra = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ra; spec.loader.exec_module(ra)
rb = ra.rb
ap, geo, _ = rb.import_audited(BASE_ROOT)

SCHEMA = "PERSIST_EEG_TSEG_PILOT_V1"
DATASETS = ("OpenBMI", "WBCIC")
OUTER = 0
SEEDS = (0, 1, 2)
METHODS = ("B0_SUBJECT_BALANCED_ERM", "B1_PLAIN_MLDG", "B2_TWO_TRAJECTORY_MEAN", "B3_TSEG")
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 60, 10, 8
BETA, GAMMA, TAU, ALPHA = 1.0, 1.0, 0.1, 0.1
BATCH_SIZE = int(rb.BATCH_SIZE); TIE_TOL = float(rb.TIE_TOL)


def clean(v: Any) -> Any:
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v); return x if math.isfinite(x) else None
    if isinstance(v, (np.bool_,)): return bool(v)
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(tmp, path)


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    tmp = path.with_suffix(path.suffix + ".part"); frame.to_csv(tmp, index=False); os.replace(tmp, path)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(); write_csv(path, pd.concat([old, pd.DataFrame(rows)], ignore_index=True))


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63 - 1)


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state): h.update(k.encode()); h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_rng(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def snapshot_rng() -> dict[str, Any]:
    return {"python": copy.deepcopy(random.getstate()), "numpy": copy.deepcopy(np.random.get_state()), "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(s: Mapping[str, Any]) -> None:
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch_cpu"])
    if torch.cuda.is_available() and s.get("torch_cuda"): torch.cuda.set_rng_state_all(s["torch_cuda"])


def ctx_for(dataset: str) -> dict[str, Any]:
    ctx = ra.cache_context(dataset, OUTER)
    if ctx["dataset"] != dataset or int(ctx["outer"]) != OUTER: raise RuntimeError("context mismatch")
    return ctx


def model_from_state(ctx: Mapping[str, Any]) -> torch.nn.Module:
    return rb.model_from_state(ctx["cache"], geo, ctx["state"], DEVICE)


def order_for(rows: np.ndarray, dataset: str, seed: int, phase: str, epoch: int, stream: str = "base") -> np.ndarray:
    key = "tseg-order" if stream != "parent" else "mldg-robustness"
    val = stable_seed(key, dataset, OUTER, seed, phase, epoch) if stream != "parent" else stable_seed(key, dataset, OUTER, seed, epoch)
    arr = np.asarray(rows, dtype=np.int64); return arr[np.random.default_rng(val).permutation(len(arr))]


def partition(subjects: list[str], dataset: str, seed: int, epoch: int, family: str) -> tuple[list[str], list[str], int]:
    # B1 preserves the audited MLDG partition family; B2/B3 share the TSEG pair.
    key = "mldg-meta-partition" if family == "MLDG" else "tseg-meta"
    pseed = stable_seed(key, dataset, OUTER, seed, epoch)
    arr = np.asarray(rb.subj_sort(subjects), dtype=object); arr = arr[np.random.default_rng(pseed).permutation(len(arr))]
    n = max(1, min(len(arr) - 1, len(arr) // 2)); tr, te = rb.subj_sort(arr[:n].tolist()), rb.subj_sort(arr[n:].tolist())
    if set(tr) & set(te): raise RuntimeError("episode overlap")
    return tr, te, pseed


def tensors(ctx: Mapping[str, Any], part: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return rb.tensors(ctx["cache"], part, mean, std, DEVICE)


def eval_model(ctx: Mapping[str, Any], model: torch.nn.Module, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    return rb.evaluate(ctx["cache"], model, rows, mean, std, DEVICE)


def train_erm_epoch(ctx: Mapping[str, Any], model: torch.nn.Module, opt: torch.optim.Optimizer, rows: np.ndarray, mean: np.ndarray, std: np.ndarray, order: np.ndarray) -> tuple[float, dict[str, int]]:
    weights = rb.lookup_weights(ctx["cache"], rows, rb.subject_balanced_weights(ctx["cache"], rows)); model.train(); loss_sum = []; fw = bw = steps = 0
    for start in range(0, len(order), BATCH_SIZE):
        part = order[start:start + BATCH_SIZE]; x, y = tensors(ctx, part, mean, std); opt.zero_grad(set_to_none=True)
        lv = F.cross_entropy(model(x), y, reduction="none"); loss = (lv * torch.as_tensor(weights[part], dtype=torch.float32, device=DEVICE)).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), rb.GRAD_CLIP); opt.step()
        loss_sum.append(float(loss.detach().cpu())); fw += 1; bw += 1; steps += 1
    return float(np.mean(loss_sum)), {"forward_count": fw, "backward_count": bw, "optimizer_steps": steps}


def train_mldg_epoch(ctx: Mapping[str, Any], model: torch.nn.Module, opt: torch.optim.Optimizer, subjects: list[str], mean: np.ndarray, std: np.ndarray, dataset: str, seed: int, epoch: int) -> tuple[float, dict[str, int]]:
    """The audited first-order one-trajectory MLDG implementation."""
    meta_tr, meta_te, _ = partition(subjects, dataset, seed, epoch, "MLDG")
    tr_rows = ctx["cache"].rows(meta_tr, geo.SESSIONS_FIT[dataset]); te_rows = ctx["cache"].rows(meta_te, geo.SESSIONS_FIT[dataset])
    tr_order = order_for(tr_rows, dataset, seed, "mldg-tr", epoch, "parent"); te_order = order_for(te_rows, dataset, seed, "mldg-te", epoch, "parent")
    tr_w = rb.lookup_weights(ctx["cache"], tr_rows, rb.subject_balanced_weights(ctx["cache"], tr_rows)); te_w = rb.lookup_weights(ctx["cache"], te_rows, rb.subject_balanced_weights(ctx["cache"], te_rows))
    params = tuple(model.parameters()); names = [n for n, _ in model.named_parameters()]; nsteps = max(1, int(math.ceil(max(len(tr_order), len(te_order)) / BATCH_SIZE))); losses=[]; fw=bw=steps=0
    model.train()
    for step in range(nsteps):
        ia=(step*BATCH_SIZE)%len(tr_order); ib=(step*BATCH_SIZE)%len(te_order); a=tr_order[ia:ia+BATCH_SIZE]; b=te_order[ib:ib+BATCH_SIZE]
        if len(a)<BATCH_SIZE: a=np.concatenate([a,tr_order[:BATCH_SIZE-len(a)]])
        if len(b)<BATCH_SIZE: b=np.concatenate([b,te_order[:BATCH_SIZE-len(b)]])
        xa,ya=tensors(ctx,a,mean,std); xb,yb=tensors(ctx,b,mean,std); wa=torch.as_tensor(tr_w[a],dtype=torch.float32,device=DEVICE); wb=torch.as_tensor(te_w[b],dtype=torch.float32,device=DEVICE)
        opt.zero_grad(set_to_none=True); ltr=(F.cross_entropy(model(xa),ya,reduction="none")*wa).mean(); gtr=torch.autograd.grad(ltr,params,create_graph=False,retain_graph=False); fast={n:p-ALPHA*g.detach() for n,p,g in zip(names,params,gtr)}; lte=(F.cross_entropy(rb.functional_call(model,fast,(xb,)),yb,reduction="none")*wb).mean(); gte=torch.autograd.grad(lte,tuple(fast.values()),create_graph=False,retain_graph=False)
        for p,a1,b1 in zip(params,gtr,gte): p.grad=a1+BETA*b1
        torch.nn.utils.clip_grad_norm_(params,rb.GRAD_CLIP); opt.step(); losses.append(float((ltr+BETA*lte).detach().cpu())); fw+=2; bw+=2; steps+=1
    return float(np.mean(losses)), {"forward_count":fw,"backward_count":bw,"optimizer_steps":steps}


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-8); q = q.clamp_min(1e-8); m = 0.5 * (p + q)
    return 0.5 * (p * (p.log() - m.log())).sum(-1).mean() + 0.5 * (q * (q.log() - m.log())).sum(-1).mean()


def pair_identity(dataset: str, seed: int, phase: str, epoch: int, step: int, pseed: int, tr1: np.ndarray, tr2: np.ndarray, te: np.ndarray) -> tuple[int, int, str]:
    s1 = stable_seed("tseg", dataset, OUTER, seed, phase, epoch, step, 1); s2 = stable_seed("tseg", dataset, OUTER, seed, phase, epoch, step, 2)
    payload = json.dumps({"dataset": dataset, "fold": OUTER, "seed": seed, "phase": phase, "epoch": epoch, "step": step, "partition_seed": pseed, "trajectory1_seed": s1, "trajectory2_seed": s2, "tr1": hashlib.sha256(np.asarray(tr1, dtype=np.int64).tobytes()).hexdigest(), "tr2": hashlib.sha256(np.asarray(tr2, dtype=np.int64).tobytes()).hexdigest(), "te": hashlib.sha256(np.asarray(te, dtype=np.int64).tobytes()).hexdigest()}, sort_keys=True).encode()
    return s1, s2, hashlib.sha256(payload).hexdigest()


def two_tr_epoch(ctx: Mapping[str, Any], model: torch.nn.Module, opt: torch.optim.Optimizer, subjects: list[str], mean: np.ndarray, std: np.ndarray, dataset: str, seed: int, epoch: int, phase: str, objective: str) -> tuple[float, dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    meta_tr, meta_te, pseed = partition(subjects, dataset, seed, epoch, "TSEG")
    tr_rows = ctx["cache"].rows(meta_tr, geo.SESSIONS_FIT[dataset]); te_rows = ctx["cache"].rows(meta_te, (geo.SESSION_DISCOVERY[dataset],))
    tr1_order = order_for(tr_rows, dataset, seed, phase + "-tr1", epoch, "tseg"); tr2_order = order_for(tr_rows, dataset, seed, phase + "-tr2", epoch, "tseg"); te_order = order_for(te_rows, dataset, seed, phase + "-te", epoch, "tseg")
    tr_w = rb.lookup_weights(ctx["cache"], tr_rows, rb.subject_balanced_weights(ctx["cache"], tr_rows)); te_w = rb.lookup_weights(ctx["cache"], te_rows, rb.subject_balanced_weights(ctx["cache"], te_rows))
    params = tuple(model.parameters()); names = [n for n, _ in model.named_parameters()]; nsteps = max(1, int(math.ceil(len(tr_rows) / BATCH_SIZE)))
    model.train(); losses = []; counts = {"forward_count": 0, "backward_count": 0, "optimizer_steps": 0}; metrics: list[dict[str, Any]] = []; hashes: list[dict[str, Any]] = []
    for step in range(nsteps):
        i1 = (step * BATCH_SIZE) % len(tr1_order); i2 = (step * BATCH_SIZE) % len(tr2_order); ie = (step * BATCH_SIZE) % len(te_order)
        a = tr1_order[i1:i1 + BATCH_SIZE]; b = tr2_order[i2:i2 + BATCH_SIZE]; e = te_order[ie:ie + BATCH_SIZE]
        if len(a) < BATCH_SIZE: a = np.concatenate([a, tr1_order[:BATCH_SIZE-len(a)]])
        if len(b) < BATCH_SIZE: b = np.concatenate([b, tr2_order[:BATCH_SIZE-len(b)]])
        if len(e) < BATCH_SIZE: e = np.concatenate([e, te_order[:BATCH_SIZE-len(e)]])
        s1, s2, ph = pair_identity(dataset, seed, phase, epoch, step, pseed, a, b, e); hashes.append({"dataset": dataset, "outer_fold": OUTER, "opt_seed": seed, "phase": phase, "epoch": epoch, "step": step, "trajectory1_seed": s1, "trajectory2_seed": s2, "partition_seed": pseed, "pair_hash": ph})
        xa, ya = tensors(ctx, a, mean, std); xb, yb = tensors(ctx, b, mean, std); xe, ye = tensors(ctx, e, mean, std)
        wa = torch.as_tensor(tr_w[a], dtype=torch.float32, device=DEVICE); wb = torch.as_tensor(tr_w[b], dtype=torch.float32, device=DEVICE); we = torch.as_tensor(te_w[e], dtype=torch.float32, device=DEVICE)
        opt.zero_grad(set_to_none=True)
        # Common source loss plus two first-order inner trajectories.
        lsrc = (F.cross_entropy(model(xa), ya, reduction="none") * wa).mean(); gsrc = torch.autograd.grad(lsrc, params, retain_graph=True, create_graph=False)
        # Different deterministic stochastic streams for the two inner updates.
        set_rng(s1); l1 = (F.cross_entropy(model(xa), ya, reduction="none") * wa).mean(); g1 = torch.autograd.grad(l1, params, retain_graph=True, create_graph=False)
        set_rng(s2); l2 = (F.cross_entropy(model(xb), yb, reduction="none") * wb).mean(); g2 = torch.autograd.grad(l2, params, retain_graph=True, create_graph=False)
        fast1 = {n: p - ALPHA * g.detach() for n, p, g in zip(names, params, g1)}; fast2 = {n: p - ALPHA * g.detach() for n, p, g in zip(names, params, g2)}
        set_rng(s1); z1 = rb.functional_call(model, fast1, (xe,)); set_rng(s2); z2 = rb.functional_call(model, fast2, (xe,)); counts["forward_count"] += 6; counts["backward_count"] += 3
        r1 = (F.cross_entropy(z1, ye, reduction="none") * we).mean(); r2 = (F.cross_entropy(z2, ye, reduction="none") * we).mean(); p1 = torch.softmax(z1, -1); p2 = torch.softmax(z2, -1); dtraj = js_divergence(p1, p2)
        # First-order outer gradients through fast weights (no second derivatives).
        gr1 = torch.autograd.grad(r1, tuple(fast1.values()), retain_graph=True, create_graph=False); gr2 = torch.autograd.grad(r2, tuple(fast2.values()), retain_graph=True, create_graph=False)
        if objective == "2TR":
            coeff1 = coeff2 = 0.5; gjs1 = gjs2 = [torch.zeros_like(x) for x in gr1]
        else:
            logits = torch.stack([r1 / TAU, r2 / TAU]); coeff = torch.softmax(logits, dim=0); coeff1, coeff2 = float(coeff[0].detach()), float(coeff[1].detach())
            gjs1 = torch.autograd.grad(dtraj, tuple(fast1.values()), retain_graph=True, create_graph=False); gjs2 = torch.autograd.grad(dtraj, tuple(fast2.values()), retain_graph=False, create_graph=False); counts["backward_count"] += 2
        for p, gs, a1, a2, j1, j2 in zip(params, gsrc, gr1, gr2, gjs1, gjs2): p.grad = gs + BETA * (coeff1 * a1 + coeff2 * a2) + (GAMMA * (j1 + j2) if objective == "TSEG" else 0.0)
        torch.nn.utils.clip_grad_norm_(params, rb.GRAD_CLIP); opt.step(); counts["backward_count"] += 2; counts["optimizer_steps"] += 1
        rworst = TAU * torch.logsumexp(torch.stack([r1 / TAU, r2 / TAU]), dim=0) - TAU * math.log(2.0); total = lsrc + BETA * (0.5 * (r1 + r2) if objective == "2TR" else rworst) + (GAMMA * dtraj if objective == "TSEG" else 0.0)
        losses.append(float(total.detach().cpu())); metrics.append({"r1": float(r1.detach().cpu()), "r2": float(r2.detach().cpu()), "abs_r_diff": float(torch.abs(r1-r2).detach().cpu()), "dtraj": float(dtraj.detach().cpu()), "source_loss": float(lsrc.detach().cpu()), "pseudo_subjects": ";".join(meta_te), "partition_seed": pseed})
    return float(np.mean(losses)), counts, metrics, hashes


def select(ctx: Mapping[str, Any], dataset: str, seed: int, method: str) -> dict[str, Any]:
    d = ROOT / "runtime"; d.mkdir(parents=True, exist_ok=True); fp = d / f"{dataset}_seed{seed}_{method}_selection.json"; cp = d / f"{dataset}_seed{seed}_{method}_selection.pt"
    if fp.exists(): return json.loads(fp.read_text(encoding="utf-8"))
    t0 = time.perf_counter(); base_seed = stable_seed("tseg-base", dataset, OUTER, seed); model = model_from_state(ctx); opt = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    best_ba, best_nll, best_ep, stale, history, start = -math.inf, math.inf, 1, 0, [], 1; compute = {"forward_count":0,"backward_count":0,"optimizer_steps":0}; mechanism=[]; part_rows=[]; pair_rows=[]
    if cp.exists():
        ck = torch.load(cp, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); best_ba,best_nll,best_ep,stale,history,start,compute,mechanism,part_rows,pair_rows=ck["best_ba"],ck["best_nll"],ck["best_ep"],ck["stale"],ck["history"],int(ck["epoch"])+1,ck["compute"],ck["mechanism"],ck["part_rows"],ck["pair_rows"]; restore_rng(ck["rng"])
    else: set_rng(base_seed)
    for epoch in range(start, MAX_EPOCHS + 1):
        if method == "B0_SUBJECT_BALANCED_ERM": loss, cnt = train_erm_epoch(ctx, model, opt, ctx["sel_rows"], ctx["sel_mean"], ctx["sel_std"], order_for(ctx["sel_rows"], dataset, seed, "selection", epoch, "parent")); mets=[]; pairs=[]
        elif method == "B1_PLAIN_MLDG": loss, cnt = train_mldg_epoch(ctx, model, opt, ctx["sel_subjects"], ctx["sel_mean"], ctx["sel_std"], dataset, seed, epoch); mets=[]; pairs=[]
        else: loss, cnt, mets, pairs = two_tr_epoch(ctx, model, opt, ctx["sel_subjects"], ctx["sel_mean"], ctx["sel_std"], dataset, seed, epoch, "selection", "2TR" if method == "B2_TWO_TRAJECTORY_MEAN" else "TSEG")
        for k,v in cnt.items(): compute[k] += int(v)
        ev = eval_model(ctx, model, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"]); ba,nll=float(ev["BA"]),float(ev["NLL"]); improved=ba>best_ba+TIE_TOL or (abs(ba-best_ba)<=TIE_TOL and nll<best_nll-TIE_TOL)
        if improved: best_ba,best_nll,best_ep,stale=ba,nll,epoch,0
        else: stale += 1
        history.append({"epoch":epoch,"train_loss":loss,"val_BA":ba,"val_NLL":nll});
        if mets:
            mechanism.extend([{**m,"dataset":dataset,"outer_fold":OUTER,"opt_seed":seed,"method":method,"phase":"selection","epoch":epoch} for m in mets]); part_rows.append({"dataset":dataset,"outer_fold":OUTER,"opt_seed":seed,"method":method,"phase":"selection","epoch":epoch,"meta_train_subjects":";".join(rb.subj_sort([x for x in ctx["sel_subjects"] if x not in set(mets[0]["pseudo_subjects"].split(';'))])) if mets else "","pseudo_unseen_subjects":mets[0]["pseudo_subjects"] if mets else "","partition_disjoint":True,"partition_seed":mets[0]["partition_seed"] if mets else None}); pair_rows.extend([{**p,"method":method} for p in pairs])
        torch.save({"epoch":epoch,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":opt.state_dict(),"best_ba":best_ba,"best_nll":best_nll,"best_ep":best_ep,"stale":stale,"history":history,"compute":compute,"mechanism":mechanism,"part_rows":part_rows,"pair_rows":pair_rows,"rng":snapshot_rng()},cp)
        if epoch>=MIN_EPOCHS and stale>=PATIENCE: break
    result={"dataset":dataset,"outer_fold":OUTER,"opt_seed":seed,"method":method,"selected_epoch":int(best_ep),"val_BA":best_ba,"val_NLL":best_nll,"base_rng_seed":base_seed,"history":history,"compute":compute,"wall_clock_sec":time.perf_counter()-t0,"mechanism":mechanism,"part_rows":part_rows,"pair_rows":pair_rows,"selection_scope":"source_validation_only"}; write_json(fp,result); del model; gc.collect();
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    return result


def refit(ctx: Mapping[str, Any], dataset: str, seed: int, method: str, epochs: int) -> dict[str, Any]:
    d=ROOT/"runtime"; fp=d/f"{dataset}_seed{seed}_{method}_refit.json"; cp=d/f"{dataset}_seed{seed}_{method}_refit.pt"
    if fp.exists(): return json.loads(fp.read_text(encoding="utf-8"))
    t0=time.perf_counter(); base_seed=stable_seed("tseg-base",dataset,OUTER,seed); model=model_from_state(ctx); opt=torch.optim.AdamW(model.parameters(),lr=rb.LR,weight_decay=rb.WEIGHT_DECAY); start=1; losses=[]; compute={"forward_count":0,"backward_count":0,"optimizer_steps":0}; mechanism=[]; part_rows=[]; pair_rows=[]
    if cp.exists():
        ck=torch.load(cp,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start=int(ck["epoch"])+1; losses,compute,mechanism,part_rows,pair_rows=ck["losses"],ck["compute"],ck["mechanism"],ck["part_rows"],ck["pair_rows"]; restore_rng(ck["rng"])
    else: set_rng(base_seed)
    for epoch in range(start,int(epochs)+1):
        if method=="B0_SUBJECT_BALANCED_ERM": loss,cnt=train_erm_epoch(ctx,model,opt,ctx["refit_rows"],ctx["refit_mean"],ctx["refit_std"],order_for(ctx["refit_rows"],dataset,seed,"refit",epoch,"parent")); mets=[]; pairs=[]
        elif method=="B1_PLAIN_MLDG": loss,cnt=train_mldg_epoch(ctx,model,opt,ctx["train_subjects"],ctx["refit_mean"],ctx["refit_std"],dataset,seed,epoch); mets=[]; pairs=[]
        else: loss,cnt,mets,pairs=two_tr_epoch(ctx,model,opt,ctx["train_subjects"],ctx["refit_mean"],ctx["refit_std"],dataset,seed,epoch,"refit","2TR" if method=="B2_TWO_TRAJECTORY_MEAN" else "TSEG")
        losses.append(loss)
        for k,v in cnt.items(): compute[k]+=int(v)
        if mets:
            mechanism.extend([{**m,"dataset":dataset,"outer_fold":OUTER,"opt_seed":seed,"method":method,"phase":"refit","epoch":epoch} for m in mets]); pair_rows.extend([{**p,"method":method} for p in pairs])
            pseudo = set(mets[0]["pseudo_subjects"].split(";")); part_rows.append({"dataset":dataset,"outer_fold":OUTER,"opt_seed":seed,"method":method,"phase":"refit","epoch":epoch,"meta_train_subjects":";".join(x for x in ctx["train_subjects"] if x not in pseudo),"pseudo_unseen_subjects":mets[0]["pseudo_subjects"],"partition_disjoint":True,"partition_seed":mets[0]["partition_seed"]})
        torch.save({"epoch":epoch,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":opt.state_dict(),"losses":losses,"compute":compute,"mechanism":mechanism,"part_rows":part_rows,"pair_rows":pair_rows,"rng":snapshot_rng()},cp)
    ev=eval_model(ctx,model,ctx["held_rows"],ctx["refit_mean"],ctx["refit_std"]); result={"dataset":dataset,"outer_fold":OUTER,"opt_seed":seed,"method":method,"epochs":int(epochs),"BA":float(ev["BA"]),"Macro_F1":float(ev["Macro_F1"]),"NLL":float(ev["NLL"]),"per_subject":ev["per_subject"],"base_rng_seed":base_seed,"losses":losses,"compute":compute,"wall_clock_sec":time.perf_counter()-t0,"mechanism":mechanism,"part_rows":part_rows,"pair_rows":pair_rows,"state_hash":state_hash(model.state_dict()),"held_evaluation_after_selection":True}; write_json(fp,result); del model; gc.collect();
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    return result


def split_audit(contexts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    parent=pd.read_csv(AUDIT_ROOT/"SPLIT_AUDIT.csv"); outer_ledger=pd.read_csv(AUDIT_ROOT.parent/"persist_eeg_route_b_foundation_screen_v1"/"OUTER_SPLIT_LEDGER.csv"); inner_ledger=pd.read_csv(AUDIT_ROOT.parent/"persist_eeg_route_b_foundation_screen_v1"/"INNER_VALIDATION_LEDGER.csv"); rows=[]
    for dataset,ctx in contexts.items():
        q=parent[(parent.dataset==dataset)&(parent.outer_fold==OUTER)]; join=lambda x:";".join(map(str,x)); expected_source="route_b_randomness_audit"
        if len(q):
            held_expected, train_expected, val_expected = str(q.iloc[0].held_subjects), str(q.iloc[0].train_subjects), str(q.iloc[0].inner_validation_subjects)
        else:
            oq=outer_ledger[(outer_ledger.dataset==dataset)&(outer_ledger.outer_fold==OUTER)]; iq=inner_ledger[(inner_ledger.dataset==dataset)&(inner_ledger.outer_fold==OUTER)]
            held_expected=join(rb.subj_sort(oq[oq.role=="H_k"].subject.tolist())); train_expected=join(rb.subj_sort(oq[oq.role=="T_k"].subject.tolist())); val_expected=join(rb.subj_sort(iq[iq.role=="inner_validation"].subject.tolist())); expected_source="route_b_foundation_ledgers"
        row={"dataset":dataset,"outer_fold":OUTER,"held_subjects":join(ctx["held"]),"train_subjects":join(ctx["train_subjects"]),"inner_validation_subjects":join(ctx["val_subjects"]),"expected_source":expected_source,"matches_parent_subject_lists":join(ctx["held"])==held_expected and join(ctx["train_subjects"])==train_expected and join(ctx["val_subjects"])==val_expected,"held_disjoint_train":not bool(set(ctx["held"])&set(ctx["train_subjects"])),"held_disjoint_validation":not bool(set(ctx["held"])&set(ctx["val_subjects"])),"validation_disjoint_selection":not bool(set(ctx["val_subjects"])&set(ctx["sel_subjects"]))}; rows.append(row)
    payload={"schema":SCHEMA,"parent_split_audit":str(AUDIT_ROOT/"SPLIT_AUDIT.csv"),"rows":rows,"pass":bool(all(r["matches_parent_subject_lists"] and r["held_disjoint_train"] and r["held_disjoint_validation"] and r["validation_disjoint_selection"] for r in rows))}; write_json(ROOT/"SPLIT_REUSE_AUDIT.json",payload); return payload


def protocol_lock() -> None:
    write_json(ROOT/"PROTOCOL_LOCK.json",{"schema":SCHEMA,"branch":"codex/persist-eeg-tseg-pilot-v1","parent":"codex/persist-eeg-mldg-perseed-earlystop-check-v1","datasets":list(DATASETS),"outer_fold":OUTER,"backbone":"EEGNet","methods":list(METHODS),"opt_seeds":list(SEEDS),"beta":BETA,"gamma":GAMMA,"tau":TAU,"alpha":ALPHA,"early_stopping":{"MAX_EPOCHS":MAX_EPOCHS,"MIN_EPOCHS":MIN_EPOCHS,"PATIENCE":PATIENCE,"metric":["highest source-validation BA","lowest source-validation NLL","earliest epoch"]},"trajectory_pair_seed":"sha256(tseg|dataset|fold|opt_seed|phase|epoch|step|trajectory_id)","held_labels_used_for_selection":False,"canonical_outcome_labels_read":False,"OpenBMI_sealed_holdout_opened":False,"WBCIC_outer_10_opened":False,"created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})


def collect_outputs(contexts: dict[str, dict[str, Any]], selected: list[dict[str, Any]], refs: list[dict[str, Any]]) -> str:
    write_csv(ROOT/"PER_SEED_SELECTION.csv", [{"dataset":s["dataset"],"seed":s["opt_seed"],"method":s["method"],"selected_epoch":s["selected_epoch"],"val_BA":s["val_BA"],"val_NLL":s["val_NLL"],"selection_scope":s["selection_scope"]} for s in selected])
    compact_refs=[{"dataset":r["dataset"],"seed":r["opt_seed"],"method":r["method"],"selected_epoch":r.get("selected_epoch",r.get("epochs")),"held_BA":r["BA"]*100.0,"Macro_F1":r["Macro_F1"]*100.0,"NLL":r["NLL"]} for r in refs]
    pd.DataFrame(compact_refs).to_csv(ROOT/"PER_SEED_RESULTS.csv",index=False)
    perf=[]; deltas=[]
    for dataset in DATASETS:
        q=pd.DataFrame([r for r in refs if r["dataset"]==dataset]);
        for method,g in q.groupby("method"):
            vals=g.BA.to_numpy(float)*100; perf.append({"dataset":dataset,"method":method,"mean_BA":float(np.mean(vals)),"median_BA":float(np.median(vals)),"seed_SD":float(np.std(vals,ddof=1)),"seed_range":float(np.max(vals)-np.min(vals))})
        erm=q[q.method=="B0_SUBJECT_BALANCED_ERM"].set_index("opt_seed").BA
        for method in METHODS[1:]:
            m=q[q.method==method].set_index("opt_seed").BA; dv=(m-erm)*100
            deltas.append({"dataset":dataset,"method":method,"mean_delta_BA_pp":float(dv.mean()),"median_delta_BA_pp":float(dv.median()),"positive_seeds":int((dv>=0).sum()),"minimum_delta_BA_pp":float(dv.min()),"maximum_delta_BA_pp":float(dv.max()),"catastrophic_cells":int((dv<-5).sum())})
    write_csv(ROOT/"OPTIMIZATION_STABILITY.csv",perf); write_csv(ROOT/"PAIRED_METHOD_DELTAS.csv",deltas)
    mech=[]; parts=[]; pairs=[]; comp=[]
    for r in selected+refs:
        phase="selection" if "history" in r else "refit"; comp.append({"dataset":r["dataset"],"outer_fold":OUTER,"opt_seed":r["opt_seed"],"method":r["method"],"phase":phase,"epochs":r.get("selected_epoch",r.get("epochs")),"wall_clock_sec":r.get("wall_clock_sec",None),**r.get("compute",{})})
        mech.extend(r.get("mechanism",[])); parts.extend(r.get("part_rows",[])); pairs.extend(r.get("pair_rows",[]))
    write_csv(ROOT/"SOURCE_TRAJECTORY_STABILITY.csv",mech); write_csv(ROOT/"EPISODE_PARTITION_AUDIT.csv",parts); write_csv(ROOT/"TRAJECTORY_PAIR_HASH.csv",pairs); write_csv(ROOT/"COMPUTE_AUDIT.csv",comp)
    for dataset in DATASETS:
        ctx=contexts[dataset]; logits_by={}
        for method in METHODS:
            for seed in SEEDS:
                cp=ROOT/"runtime"/f"{dataset}_seed{seed}_{method}_refit.pt"; ck=torch.load(cp,map_location="cpu",weights_only=False); model=model_from_state(ctx); model.load_state_dict(ck["model"]); model.eval(); logits=[]
                with torch.inference_mode():
                    for st in range(0,len(ctx["held_rows"]),BATCH_SIZE):
                        part=ctx["held_rows"][st:st+BATCH_SIZE]; x,_=tensors(ctx,part,ctx["refit_mean"],ctx["refit_std"]); logits.append(model(x).detach().cpu())
                logits_by[(method,seed)]=torch.cat(logits).numpy(); del model
        rows=[]
        for method in METHODS:
            for i in range(3):
                for j in range(i+1,3):
                    a,b=logits_by[(method,i)],logits_by[(method,j)]; pa=np.exp(a-a.max(1,keepdims=True)); pa/=pa.sum(1,keepdims=True); pb=np.exp(b-b.max(1,keepdims=True)); pb/=pb.sum(1,keepdims=True); m=.5*(pa+pb); js=.5*np.sum(pa*(np.log(pa+1e-8)-np.log(m+1e-8)),1)+.5*np.sum(pb*(np.log(pb+1e-8)-np.log(m+1e-8)),1)
                    rows.append({"dataset":dataset,"method":method,"seed_i":i,"seed_j":j,"prediction_disagreement_rate":float(np.mean(pa.argmax(1)!=pb.argmax(1))),"pairwise_JS":float(np.mean(js)),"held_evaluation_after_all_training":True})
        append_csv(ROOT/"HELDOUT_FUNCTIONAL_DISAGREEMENT.csv",rows)
    stab=pd.DataFrame(perf); dlt=pd.DataFrame(deltas);
    def row(ds,method): return stab[(stab.dataset==ds)&(stab.method==method)].iloc[0]
    def delta(ds,method): return dlt[(dlt.dataset==ds)&(dlt.method==method)].iloc[0]
    tseg_d=[delta(d,"B3_TSEG") for d in DATASETS]; two_d=[delta(d,"B2_TWO_TRAJECTORY_MEAN") for d in DATASETS]; mldg_d=[delta(d,"B1_PLAIN_MLDG") for d in DATASETS]
    tseg_cells=sum(int(x.positive_seeds) for x in tseg_d); g1=all(float(x.mean_delta_BA_pp)>=.5 for x in tseg_d); g2=tseg_cells>=5; g3=all(int(x.catastrophic_cells)==0 for x in tseg_d); g4=(row("WBCIC","B3_TSEG").seed_range<row("WBCIC","B0_SUBJECT_BALANCED_ERM").seed_range and row("WBCIC","B3_TSEG").seed_range<row("WBCIC","B1_PLAIN_MLDG").seed_range); pooled=float(np.mean([float(delta(d,"B3_TSEG").mean_delta_BA_pp)-float(delta(d,"B2_TWO_TRAJECTORY_MEAN").mean_delta_BA_pp) for d in DATASETS])); g5=pooled>=.3 or (all(row(d,"B3_TSEG").seed_range-row(d,"B2_TWO_TRAJECTORY_MEAN").seed_range<=-.25*row(d,"B2_TWO_TRAJECTORY_MEAN").seed_range for d in DATASETS) and all(row(d,"B3_TSEG").mean_BA-row(d,"B2_TWO_TRAJECTORY_MEAN").mean_BA>=-.2 for d in DATASETS)); mechdf=pd.DataFrame(mech); g6=bool(len(mechdf) and mechdf[mechdf.method=="B3_TSEG"].dtraj.mean()<mechdf[mechdf.method=="B2_TWO_TRAJECTORY_MEAN"].dtraj.mean() or len(mechdf) and mechdf[mechdf.method=="B3_TSEG"].abs_r_diff.mean()<mechdf[mechdf.method=="B2_TWO_TRAJECTORY_MEAN"].abs_r_diff.mean())
    normal=bool(g1 and g2 and g3 and g4 and g5 and g6); partial=bool(g4 and not normal and not g1); strong=bool(normal and all(float(x.mean_delta_BA_pp)>=1 for x in tseg_d) and tseg_cells==6 and row("WBCIC","B3_TSEG").seed_range<=.5*row("WBCIC","B0_SUBJECT_BALANCED_ERM").seed_range and pooled>=.3)
    terminal="TSEG_STRONG_PILOT_SIGNAL" if strong else ("TSEG_PILOT_SIGNAL_SUPPORTED" if normal else ("TSEG_STABILITY_ONLY_PARTIAL_SIGNAL" if partial else "TSEG_PILOT_NOT_SUPPORTED"))
    gate={"schema":SCHEMA,"terminal":terminal,"gates":{"G1_both_dataset_mean_ge_0.5":g1,"G2_positive_cells_ge_5_of_6":g2,"G3_no_catastrophic_collapse":g3,"G4_WBCIC_range_below_ERM_and_MLDG":g4,"G5_beats_compute_matched_2TR":g5,"G6_mechanism_reduces_disagreement_or_dispersion":g6},"pooled_TSEG_minus_2TR_mean_pp":pooled,"canonical_outcome_labels_read":False,"OpenBMI_sealed_holdout_opened":False,"WBCIC_outer_10_opened":False}; write_json(ROOT/"GO_GATE.json",gate)
    lines=["# TSEG pilot", "", "## Performance", "", "| Dataset | Method | Mean BA | Median BA | Seed SD | Seed Range |", "|---|---|---:|---:|---:|---:|"]
    for r in perf: lines.append(f"| {r['dataset']} | {r['method']} | {r['mean_BA']:.4f} | {r['median_BA']:.4f} | {r['seed_SD']:.4f} | {r['seed_range']:.4f} |")
    lines += ["", "## Relative", "", "| Dataset | TSEG-ERM mean | positive/3 | min Δ | TSEG-MLDG | TSEG-2TR |", "|---|---:|---:|---:|---:|---:|"]
    for d in DATASETS: lines.append(f"| {d} | {delta(d,'B3_TSEG').mean_delta_BA_pp:+.4f} | {int(delta(d,'B3_TSEG').positive_seeds)}/3 | {delta(d,'B3_TSEG').minimum_delta_BA_pp:+.4f} | {(delta(d,'B3_TSEG').mean_delta_BA_pp-delta(d,'B1_PLAIN_MLDG').mean_delta_BA_pp):+.4f} | {(delta(d,'B3_TSEG').mean_delta_BA_pp-delta(d,'B2_TWO_TRAJECTORY_MEAN').mean_delta_BA_pp):+.4f} |")
    lines += ["",f"Terminal: `{terminal}`", "", f"Gates: G1={g1}, G2={g2}, G3={g3}, G4={g4}, G5={g5}, G6={g6}.", "", "Held-out functional disagreement is post-training analysis only; it was not used for checkpoint selection.", "TSEG targets functional trajectory stability, not parameter flatness.", "Only OpenBMI/WBCIC fold0, EEGNet, seeds 0/1/2 were run; no sealed holdout or WBCIC outer-10 was opened."]
    (ROOT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); write_json(ROOT/"NO_CANONICAL_OUTCOME_ACCESS_AUDIT.json",{"schema":SCHEMA,"canonical_outcome_labels_read":False,"OpenBMI_sealed_holdout_opened":False,"WBCIC_outer_10_opened":False,"held_labels_used_for_selection":False}); return terminal


def main() -> int:
    protocol_lock(); contexts={d:ctx_for(d) for d in DATASETS}; split=split_audit(contexts)
    if not split["pass"]: raise RuntimeError("split audit failed")
    selected=[]; refs=[]; init_rows=[]
    for d in DATASETS:
        for seed in SEEDS:
            ih=state_hash(contexts[d]["state"]); init_rows.append({"dataset":d,"opt_seed":seed,"ERM_hash":ih,"MLDG_hash":ih,"2TR_hash":ih,"TSEG_hash":ih,"identical":True})
            for method in METHODS:
                s=select(contexts[d],d,seed,method); selected.append(s)
                r=refit(contexts[d],d,seed,method,int(s["selected_epoch"])); r["selected_epoch"]=s["selected_epoch"]; refs.append(r)
                print(f"[cell] {d} seed={seed} {method} epoch={s['selected_epoch']} BA={r['BA']*100:.4f}",flush=True)
    write_csv(ROOT/"INITIAL_STATE_HASH.csv",init_rows)
    terminal=collect_outputs(contexts,selected,refs); print(terminal,flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
