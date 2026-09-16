#!/usr/bin/env python3
"""Seed-0 single-LiteBN BN-state 2x2 intervention.

The direct branch-only adaptation analogue is frozen before any heldout bundle is
opened: 20 source-only CE epochs, AdamW 1e-4, wd 5e-4, clip 5, no selection.
Unlike BNLOCK, BatchNorm is intentionally left in train mode so nu1 can differ
from nu0; this is the causal quantity tested by M00/M11/M10/M01.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get("BNSTATE_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_singlemodel_bn_state_seed0_v1"
OUT, PROTOCOL, RUN = EXP / "outputs", EXP / "protocol", EXP / "runtime"
BASE_RUN = REPO / "experiments/persist_eeg_litebn_ablation_v1/code/run_litebn_ablation.py"
CSGD_RUN = REPO / "experiments/persist_eeg_litebn_tfformer_csgd_v1/code/run_csgd.py"
TASKS = ("OpenBMI_MI", "WBCIC_MI")
FOLDS = range(5)
EPOCHS, LR, WD, CLIP, BOOT = 20, 1e-4, 5e-4, 5.0, 20_000


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"); os.replace(tmp, path)


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, path)


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text.rstrip() + "\n"); os.replace(tmp, path)


def seed(value=0):
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def tensor_hash(items):
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def bn_keys(model):
    keys = []
    state = model.state_dict()
    for name, item in model.named_modules():
        if isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f"{name}." if name else ""
            keys.extend(prefix + suffix for suffix in ("running_mean", "running_var", "num_batches_tracked"))
    if not keys or not set(keys) <= set(state): raise RuntimeError("BN key discovery failed")
    return sorted(keys)


def combine(primary, buffers, allowed):
    result = {k: v.detach().clone() for k, v in primary.items()}
    for key in allowed: result[key] = buffers[key].detach().clone()
    return result


def checkpoint(csgd, task, fold):
    return csgd.litebn_path(task, 0, fold)


def train_cell(base, csgd, task, fold, fold_spec, device):
    target = RUN / "checkpoints" / task / f"fold{fold}" / "adapted.pt"
    if target.is_file(): return target
    # Only canonical source/development data is materialized here.
    bundle = base.build_bundle(task, fold_spec["inner_train_subjects"] + fold_spec["inner_val_subjects"])
    mean, std, meta = base.load_tensor_pair(csgd.normalizer_path(task, fold))
    raw = base.RawGPUCache(bundle, device)
    cache = base_runner.NormalizedCache(raw, mean, std)
    model = base.build_model("LiteBN_BASELINE", task)
    source = checkpoint(csgd, task, fold)
    model.load_state_dict(torch.load(source, map_location="cpu", weights_only=False), strict=True)
    model = model.to(device)
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    allowed = bn_keys(model)
    seed(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    episodes = base.mi_manifest(bundle, fold_spec, task)[0]
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for indices in episodes[epoch - 1]:
            x, y = cache.batch(indices); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(x); loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "mean_CE": float(np.mean(losses))})
        print(f"BNSTATE_TRAIN {task} f{fold} e{epoch:02d} CE={history[-1]['mean_CE']:.6f}", flush=True)
    adapted = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"original_state": original, "adapted_state": adapted, "bn_buffer_keys": allowed,
                "source_checkpoint": str(source), "normalizer_metadata": meta,
                "history": history, "recipe": "single-LiteBN source-only CE direct analogue"}, target)
    del model, cache, raw, bundle
    gc.collect(); torch.cuda.empty_cache()
    return target


def eval_conditions(base, csgd, runtime, task, fold, path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    original, adapted, allowed = payload["original_state"], payload["adapted_state"], payload["bn_buffer_keys"]
    states = {"M00_ORIGINAL": original, "M11_ADAPTED": adapted,
              "M10_RESTORE_STATE": combine(adapted, original, allowed),
              "M01_TRANSPLANT_STATE": combine(original, adapted, allowed)}
    template = base.build_model("LiteBN_BASELINE", task)
    parameter_keys = {name for name, _ in template.named_parameters()}
    hashes = {}
    for condition, state in states.items():
        hashes[condition] = {"parameter_hash": tensor_hash((k, state[k]) for k in parameter_keys),
                             "bn_buffer_hash": tensor_hash((k, state[k]) for k in allowed)}
    if hashes["M10_RESTORE_STATE"]["parameter_hash"] != hashes["M11_ADAPTED"]["parameter_hash"]: raise RuntimeError("M10 parameter mismatch")
    if hashes["M01_TRANSPLANT_STATE"]["parameter_hash"] != hashes["M00_ORIGINAL"]["parameter_hash"]: raise RuntimeError("M01 parameter mismatch")
    subjects, sessions = csgd.subjects_and_sessions(task)
    bundle = csgd.build_wbcic_outer_bundle(runtime)[0] if task == "WBCIC_MI" else base.build_bundle(task, subjects)
    raw = base.RawGPUCache(bundle, device)
    mean, std, meta = base.load_tensor_pair(csgd.normalizer_path(task, fold))
    rows = []
    for condition, state in states.items():
        model = base.build_model("LiteBN_BASELINE", task); model.load_state_dict(state, strict=True); model = model.to(device).eval()
        current = csgd.evaluate_sessions(runtime, model, "LiteBN", bundle, raw, None, subjects, sessions, mean, std)
        for row in current: row.update({"task": task, "fold": fold, "seed": 0, "condition": condition, **hashes[condition]})
        rows.extend(current); del model; torch.cuda.empty_cache()
    del raw, bundle; gc.collect(); torch.cuda.empty_cache()
    return rows, hashes


def bootstrap(values, key):
    values = np.asarray(values, float); rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    means = np.empty(BOOT)
    for start in range(0, BOOT, 2000):
        stop = min(BOOT, start + 2000); idx = rng.integers(0, len(values), (stop-start, len(values))); means[start:stop] = values[idx].mean(1)
    lo, hi = np.quantile(means, [.025, .975]); return float(values.mean()), float(lo), float(hi)


def summarize(rows):
    frame = pd.DataFrame(rows)
    session = frame.groupby(["task", "condition", "subject_id", "session"], as_index=False).agg(BA=("BA","mean"), macro_F1=("macro_F1","mean"), folds=("fold","nunique"))
    if not (session.folds == 5).all(): raise RuntimeError("incomplete fold coverage")
    subject = []
    for (task, condition, sid), part in session.groupby(["task","condition","subject_id"]):
        cells = {r.session:r for r in part.itertuples(index=False)}; required = ("S0","S1","S2") if task == "WBCIC_MI" else ("S1","S2")
        subject.append({"task":task,"condition":condition,"subject_id":sid,"future_BA":cells["S2"].BA,"WS_BA":min(cells[s].BA for s in required)})
    sf = pd.DataFrame(subject); conditions = ("M00_ORIGINAL","M11_ADAPTED","M10_RESTORE_STATE","M01_TRANSPLANT_STATE")
    task_rows, effects = [], []
    contrasts = {"adaptation_loss":("M11_ADAPTED","M00_ORIGINAL"), "state_restoration":("M10_RESTORE_STATE","M11_ADAPTED"),
                 "state_transplant":("M01_TRANSPLANT_STATE","M00_ORIGINAL"), "parameter_only_retained":("M10_RESTORE_STATE","M00_ORIGINAL")}
    for task in TASKS:
        for condition in conditions:
            p=sf[(sf.task==task)&(sf.condition==condition)]; task_rows.append({"task":task,"condition":condition,"future_BA":p.future_BA.mean(),"WS_BA":p.WS_BA.mean(),"subjects":len(p)})
        for label,(a,b) in contrasts.items():
            x=sf[(sf.task==task)&(sf.condition==a)].set_index("subject_id"); y=sf[(sf.task==task)&(sf.condition==b)].set_index("subject_id"); ids=sorted(set(x.index)&set(y.index))
            row={"task":task,"contrast":label,"condition_A":a,"condition_B":b,"subjects":len(ids)}
            for metric in ("future_BA","WS_BA"):
                mean,lo,hi=bootstrap(100*(x.loc[ids,metric].to_numpy()-y.loc[ids,metric].to_numpy()),f"{task}/{label}/{metric}")
                row[f"delta_{metric}_pp"]=mean; row[f"delta_{metric}_CI95_low_pp"]=lo; row[f"delta_{metric}_CI95_high_pp"]=hi
            effects.append(row)
    return session.to_dict("records"), subject, task_rows, effects


def main():
    global base_runner
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    base_runner = module("bnstate_base_runner", BASE_RUN); base = base_runner.base
    csgd = module("bnstate_csgd", CSGD_RUN); runtime = csgd.load_runtime(); device=torch.device("cuda")
    _, fold_map, split_sha = base.load_folds()
    paths=[]
    # Freeze every adapted checkpoint before opening heldout evaluation data.
    for task in TASKS:
        dataset=base.TASKS[task]["dataset"]
        for fold_spec in fold_map[dataset]:
            f=int(fold_spec["fold_id"]); paths.append((task,f,train_cell(base,csgd,task,f,fold_spec,device)))
    print("BNSTATE_ALL_ADAPTATION_FROZEN", flush=True)
    rows=[]; hashes=[]
    for task,f,path in paths:
        current,audit=eval_conditions(base,csgd,runtime,task,f,path,device); rows.extend(current)
        hashes.append({"task":task,"fold":f,**{f"{c}_{k}":v for c,d in audit.items() for k,v in d.items()}})
        atomic_csv(RUN/"SESSION_RESULTS.csv",rows); print(f"BNSTATE_EVAL {task} f{f}",flush=True)
    session,subject,task_rows,effects=summarize(rows)
    atomic_csv(OUT/"SESSION_RESULTS.csv",session); atomic_csv(OUT/"SUBJECT_RESULTS.csv",subject); atomic_csv(OUT/"TASK_SUMMARY.csv",task_rows); atomic_csv(OUT/"PAIRED_EFFECTS.csv",effects)
    atomic_json(PROTOCOL/"STATE_HASH_VALIDATION.json",{"status":"PASS","cells":hashes,"M10_parameters_equal_theta1":True,"M01_parameters_equal_theta0":True,"only_BN_buffers_swapped":True})
    atomic_json(PROTOCOL/"PROTOCOL.json",{"seed":0,"tasks":TASKS,"folds":list(FOLDS),"adaptation_epochs":EPOCHS,"optimizer":"AdamW","lr":LR,"weight_decay":WD,"gradient_clip":CLIP,"heldout_used_for_adaptation_or_selection":False,"direct_analogue_deviation":"single LiteBN CE only; EEGNet anchor and fusion removed; BN deliberately train-mode to produce nu1"})
    lines=["# Single-LiteBN BN-state seed-0 intervention","","All adapted checkpoints were frozen on source-only data before heldout evaluation.","","| Task | Condition | Future BA | WS-BA |","|---|---|---:|---:|"]
    for r in task_rows: lines.append(f"| {r['task']} | {r['condition']} | {r['future_BA']:.4f} | {r['WS_BA']:.4f} |")
    lines += ["","| Task | Contrast | Delta future BA pp [95% CI] | Delta WS-BA pp [95% CI] |","|---|---|---:|---:|"]
    for r in effects: lines.append(f"| {r['task']} | {r['contrast']} | {r['delta_future_BA_pp']:+.3f} [{r['delta_future_BA_CI95_low_pp']:+.3f}, {r['delta_future_BA_CI95_high_pp']:+.3f}] | {r['delta_WS_BA_pp']:+.3f} [{r['delta_WS_BA_CI95_low_pp']:+.3f}, {r['delta_WS_BA_CI95_high_pp']:+.3f}] |")
    atomic_text(OUT/"FINAL_SINGLE_LITEBN_BN_STATE_REPORT.md","\n".join(lines)); atomic_json(OUT/"COMPLETION.json",{"status":"COMPLETE","cells":10,"session_rows":len(session),"subject_rows":len(subject)})
    print("BNSTATE_COMPLETE",flush=True)


if __name__ == "__main__": main()
