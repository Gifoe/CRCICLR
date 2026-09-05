"""WBCIC fold-0 per-seed early-stop ambiguity check for Route-B MLDG.

This driver reuses the frozen Route-B split/cache and changes only source-only
epoch selection: ERM and MLDG each select their own epoch on source
validation, then refit from the paired initial state on all outer-training
subjects.  No canonical outcome or sealed holdout is opened.
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
CODE_DIR = AUDIT_ROOT / "code"
RA_PATH = CODE_DIR / "run_randomness_audit.py"
spec = importlib.util.spec_from_file_location("route_b_randomness_audit_perseed", RA_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {RA_PATH}")
ra = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ra
spec.loader.exec_module(ra)
rb = ra.rb
ap, geo, _ = rb.import_audited(BASE_ROOT)

SCHEMA = "PERSIST_EEG_MLDG_PERSEED_EARLYSTOP_CHECK_V1"
DATASET = "WBCIC"
OUTER = 0
SEEDS = (0, 1, 2)
BETA = 1.0
MAX_EPOCHS = 60
MIN_EPOCHS = 10
PATIENCE = 8
BATCH_SIZE = int(rb.BATCH_SIZE)
TIE_TOL = float(rb.TIE_TOL)


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v); return x if math.isfinite(x) else None
    if isinstance(v, (np.bool_,)): return bool(v)
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, value: Any) -> None:
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_rng(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed) % (2**32 - 1)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def snapshot_rng() -> dict[str, Any]:
    return {"python": copy.deepcopy(random.getstate()), "numpy": copy.deepcopy(np.random.get_state()),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"): torch.cuda.set_rng_state_all(state["torch_cuda"])


def cache_context() -> dict[str, Any]:
    # This is the exact context constructor used by the parent Route-B audit.
    ctx = ra.cache_context(DATASET, OUTER)
    if ctx["dataset"] != DATASET or int(ctx["outer"]) != OUTER: raise RuntimeError("context mismatch")
    return ctx


def model_from_state(ctx: Mapping[str, Any]) -> torch.nn.Module:
    model = rb.model_from_state(ctx["cache"], geo, ctx["state"], DEVICE)
    return model


def order_for(rows: np.ndarray, opt_seed: int, phase: str, epoch: int) -> np.ndarray:
    # Preserve the parent robustness-confirm order schedule.  Per-seed
    # early stopping is the only training change; the only new hash is the
    # explicitly registered MLDG meta-subject partition below.
    seed = stable_seed("mldg-robustness", DATASET, OUTER, opt_seed, epoch)
    arr = np.asarray(rows, dtype=np.int64)
    return arr[np.random.default_rng(seed).permutation(len(arr))]


def mldg_partition(subjects: list[str], opt_seed: int, epoch: int) -> tuple[list[str], list[str], int]:
    # Exact pre-registered partition seed formula.
    seed = stable_seed("mldg-perseed-es-meta", DATASET, OUTER, opt_seed, epoch)
    arr = np.asarray(rb.subj_sort(subjects), dtype=object)
    arr = arr[np.random.default_rng(seed).permutation(len(arr))]
    n = max(1, min(len(arr) - 1, len(arr) // 2))
    tr, te = rb.subj_sort(arr[:n].tolist()), rb.subj_sort(arr[n:].tolist())
    if set(tr) & set(te): raise RuntimeError("MLDG partition overlap")
    return tr, te, seed


def train_erm_epoch(ctx: Mapping[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    rows: np.ndarray, mean: np.ndarray, std: np.ndarray, order: np.ndarray) -> float:
    weights = rb.lookup_weights(ctx["cache"], rows, rb.subject_balanced_weights(ctx["cache"], rows))
    model.train(); losses: list[float] = []
    for start in range(0, len(order), BATCH_SIZE):
        part = order[start:start + BATCH_SIZE]
        x, y = rb.tensors(ctx["cache"], part, mean, std, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss_vec = F.cross_entropy(model(x), y, reduction="none")
        loss = (loss_vec * torch.as_tensor(weights[part], dtype=torch.float32, device=DEVICE)).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), rb.GRAD_CLIP); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_mldg_epoch(ctx: Mapping[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     subjects: list[str], mean: np.ndarray, std: np.ndarray, opt_seed: int, epoch: int) -> tuple[float, int]:
    meta_train, meta_test, pseed = mldg_partition(subjects, opt_seed, epoch)
    tr_rows = ctx["cache"].rows(meta_train, geo.SESSIONS_FIT[DATASET]); te_rows = ctx["cache"].rows(meta_test, geo.SESSIONS_FIT[DATASET])
    tr_order = order_for(tr_rows, opt_seed, "mldg-tr", epoch); te_order = order_for(te_rows, opt_seed, "mldg-te", epoch)
    tr_w = rb.lookup_weights(ctx["cache"], tr_rows, rb.subject_balanced_weights(ctx["cache"], tr_rows))
    te_w = rb.lookup_weights(ctx["cache"], te_rows, rb.subject_balanced_weights(ctx["cache"], te_rows))
    params = tuple(model.parameters()); names = [n for n, _ in model.named_parameters()]
    nsteps = max(1, int(math.ceil(max(len(tr_order), len(te_order)) / BATCH_SIZE)))
    model.train(); losses: list[float] = []
    for step in range(nsteps):
        ia = (step * BATCH_SIZE) % len(tr_order); ib = (step * BATCH_SIZE) % len(te_order)
        tr_part = tr_order[ia:ia + BATCH_SIZE]; te_part = te_order[ib:ib + BATCH_SIZE]
        if len(tr_part) < BATCH_SIZE: tr_part = np.concatenate([tr_part, tr_order[:BATCH_SIZE - len(tr_part)]])
        if len(te_part) < BATCH_SIZE: te_part = np.concatenate([te_part, te_order[:BATCH_SIZE - len(te_part)]])
        xtr, ytr = rb.tensors(ctx["cache"], tr_part, mean, std, DEVICE); xte, yte = rb.tensors(ctx["cache"], te_part, mean, std, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        ltr = (F.cross_entropy(model(xtr), ytr, reduction="none") * torch.as_tensor(tr_w[tr_part], dtype=torch.float32, device=DEVICE)).mean()
        gtr = torch.autograd.grad(ltr, params, create_graph=False, retain_graph=False)
        fast = {n: p - rb.MLDG_ALPHA * g.detach() for n, p, g in zip(names, params, gtr)}
        lte = (F.cross_entropy(rb.functional_call(model, fast, (xte,)), yte, reduction="none") * torch.as_tensor(te_w[te_part], dtype=torch.float32, device=DEVICE)).mean()
        gte = torch.autograd.grad(lte, tuple(fast.values()), create_graph=False, retain_graph=False)
        for p, a, b in zip(params, gtr, gte): p.grad = a + BETA * b
        torch.nn.utils.clip_grad_norm_(params, rb.GRAD_CLIP); optimizer.step()
        losses.append(float((ltr + BETA * lte).detach().cpu()))
    return float(np.mean(losses)), pseed


def evaluate(ctx: Mapping[str, Any], model: torch.nn.Module, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    return rb.evaluate(ctx["cache"], model, rows, mean, std, DEVICE)


def selection(ctx: Mapping[str, Any], seed: int, method: str) -> dict[str, Any]:
    out = ROOT / "runtime"; out.mkdir(parents=True, exist_ok=True)
    final_path = out / f"seed{seed}_{method}_selection.json"; ckpt_path = out / f"seed{seed}_{method}_selection.pt"
    if final_path.exists(): return json.loads(final_path.read_text(encoding="utf-8"))
    base_seed = stable_seed("mldg-robustness", DATASET, OUTER, seed, "train")
    model = model_from_state(ctx); optimizer = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    best_ba, best_nll, best_ep, stale = -math.inf, math.inf, 1, 0; history: list[dict[str, Any]] = []; start_epoch = 1
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        best_ba, best_nll, best_ep, stale = ck["best_ba"], ck["best_nll"], ck["best_ep"], ck["stale"]; history, start_epoch = ck["history"], int(ck["epoch"]) + 1; restore_rng(ck["rng"])
    else: set_rng(base_seed)
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if method == "ERM":
            loss = train_erm_epoch(ctx, model, optimizer, ctx["sel_rows"], ctx["sel_mean"], ctx["sel_std"], order_for(ctx["sel_rows"], seed, "selection", epoch))
            pseed = None
        else:
            loss, pseed = train_mldg_epoch(ctx, model, optimizer, ctx["sel_subjects"], ctx["sel_mean"], ctx["sel_std"], seed, epoch)
        ev = evaluate(ctx, model, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"]); ba, nll = float(ev["BA"]), float(ev["NLL"])
        improved = ba > best_ba + TIE_TOL or (abs(ba - best_ba) <= TIE_TOL and nll < best_nll - TIE_TOL)
        if improved: best_ba, best_nll, best_ep, stale = ba, nll, epoch, 0
        else: stale += 1
        history.append({"epoch": epoch, "train_loss": loss, "val_BA": ba, "val_NLL": nll, "partition_seed": pseed})
        torch.save({"epoch": epoch, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": optimizer.state_dict(), "best_ba": best_ba, "best_nll": best_nll, "best_ep": best_ep, "stale": stale, "history": history, "rng": snapshot_rng()}, ckpt_path)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE: break
    result = {"dataset": DATASET, "outer_fold": OUTER, "opt_seed": seed, "method": method, "selected_epoch": int(best_ep), "val_BA": best_ba, "val_NLL": best_nll, "base_rng_seed": base_seed, "history": history, "selection_scope": "source_validation_only"}
    write_json(final_path, result); del model; gc.collect();
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return result


def refit(ctx: Mapping[str, Any], seed: int, method: str, epochs: int) -> dict[str, Any]:
    out = ROOT / "runtime"; out.mkdir(parents=True, exist_ok=True)
    final_path = out / f"seed{seed}_{method}_refit.json"; ckpt_path = out / f"seed{seed}_{method}_refit.pt"
    if final_path.exists(): return json.loads(final_path.read_text(encoding="utf-8"))
    base_seed = stable_seed("mldg-robustness", DATASET, OUTER, seed, "train")
    model = model_from_state(ctx); optimizer = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    losses: list[float] = []; start_epoch = 1
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); losses, start_epoch = ck["losses"], int(ck["epoch"]) + 1; restore_rng(ck["rng"])
    else: set_rng(base_seed)
    for epoch in range(start_epoch, int(epochs) + 1):
        if method == "ERM": loss = train_erm_epoch(ctx, model, optimizer, ctx["refit_rows"], ctx["refit_mean"], ctx["refit_std"], order_for(ctx["refit_rows"], seed, "refit", epoch)); pseed = None
        else: loss, pseed = train_mldg_epoch(ctx, model, optimizer, ctx["train_subjects"], ctx["refit_mean"], ctx["refit_std"], seed, epoch)
        losses.append(loss)
        torch.save({"epoch": epoch, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": optimizer.state_dict(), "losses": losses, "rng": snapshot_rng()}, ckpt_path)
    # Held labels are touched only after source-only selection and legal refit.
    ev = evaluate(ctx, model, ctx["held_rows"], ctx["refit_mean"], ctx["refit_std"])
    result = {"dataset": DATASET, "outer_fold": OUTER, "opt_seed": seed, "method": method, "epochs": int(epochs), "BA": float(ev["BA"]), "Macro_F1": float(ev["Macro_F1"]), "NLL": float(ev["NLL"]), "per_subject": ev["per_subject"], "base_rng_seed": base_seed, "losses": losses, "state_hash": state_hash(model.state_dict()), "held_evaluation_after_selection": True}
    write_json(final_path, result); del model; gc.collect();
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return result


def split_audit(ctx: Mapping[str, Any]) -> dict[str, Any]:
    parent = pd.read_csv(AUDIT_ROOT / "SPLIT_AUDIT.csv")
    def join(vals: Any) -> str: return ";".join(map(str, vals))
    current = {"dataset": DATASET, "outer_fold": OUTER, "held_subjects": join(ctx["held"]), "train_subjects": join(ctx["train_subjects"]), "inner_validation_subjects": join(ctx["val_subjects"]), "source_selection_subjects": join(ctx["sel_subjects"])}
    q = parent[(parent.dataset == DATASET) & (parent.outer_fold == OUTER)].iloc[0]
    expected = {"held_subjects": str(q.held_subjects), "train_subjects": str(q.train_subjects), "inner_validation_subjects": str(q.inner_validation_subjects)}
    current["matches_parent_subject_lists"] = all(current[k] == expected[k] for k in expected)
    current["held_disjoint_train"] = not bool(set(ctx["held"]) & set(ctx["train_subjects"]))
    current["held_disjoint_validation"] = not bool(set(ctx["held"]) & set(ctx["val_subjects"]))
    current["validation_disjoint_selection"] = not bool(set(ctx["val_subjects"]) & set(ctx["sel_subjects"]))
    payload = {"schema": SCHEMA, "parent_split_audit": str(AUDIT_ROOT / "SPLIT_AUDIT.csv"), "rows": [current], "pass": bool(current["matches_parent_subject_lists"] and current["held_disjoint_train"] and current["held_disjoint_validation"] and current["validation_disjoint_selection"])}
    write_json(ROOT / "SPLIT_REUSE_AUDIT.json", payload); return payload


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(); write_csv(path, pd.concat([old, pd.DataFrame(rows)], ignore_index=True))


def run_seed(ctx: Mapping[str, Any], seed: int) -> None:
    existing = pd.read_csv(ROOT / "PAIRED_RESULTS.csv") if (ROOT / "PAIRED_RESULTS.csv").exists() else pd.DataFrame()
    if not existing.empty and int(seed) in set(existing.opt_seed.astype(int)): return
    er_sel = selection(ctx, seed, "ERM"); ml_sel = selection(ctx, seed, "MLDG")
    er_ref = refit(ctx, seed, "ERM", int(er_sel["selected_epoch"])); ml_ref = refit(ctx, seed, "MLDG", int(ml_sel["selected_epoch"]))
    init = state_hash(ctx["state"])
    append_csv(ROOT / "INITIAL_STATE_HASH.csv", [{"opt_seed": seed, "ERM_hash": init, "MLDG_hash": init, "identical": True}])
    append_csv(ROOT / "PER_SEED_SELECTION.csv", [{"opt_seed": seed, "method": "ERM", "selected_epoch": er_sel["selected_epoch"], "val_BA": er_sel["val_BA"], "val_NLL": er_sel["val_NLL"], "selection_scope": er_sel["selection_scope"]}, {"opt_seed": seed, "method": "MLDG", "selected_epoch": ml_sel["selected_epoch"], "val_BA": ml_sel["val_BA"], "val_NLL": ml_sel["val_NLL"], "selection_scope": ml_sel["selection_scope"]}])
    delta = (ml_ref["BA"] - er_ref["BA"]) * 100.0
    append_csv(ROOT / "PAIRED_RESULTS.csv", [{"opt_seed": seed, "ERM_epoch": er_sel["selected_epoch"], "MLDG_epoch": ml_sel["selected_epoch"], "ERM_BA": er_ref["BA"] * 100.0, "MLDG_BA": ml_ref["BA"] * 100.0, "delta_BA_pp": delta, "ERM_Macro_F1": er_ref["Macro_F1"] * 100.0, "MLDG_Macro_F1": ml_ref["Macro_F1"] * 100.0, "ERM_NLL": er_ref["NLL"], "MLDG_NLL": ml_ref["NLL"], "ERM_base_rng_seed": er_sel["base_rng_seed"], "MLDG_base_rng_seed": ml_sel["base_rng_seed"], "initial_state_hash": init, "held_labels_used_for_selection": False}])
    er_sub = {str(x["subject"]): x for x in er_ref["per_subject"]}; ml_sub = {str(x["subject"]): x for x in ml_ref["per_subject"]}
    append_csv(ROOT / "PER_SUBJECT_PAIRED_DELTAS.csv", [{"opt_seed": seed, "subject": s, "ERM_BA": er_sub[s]["BA"] * 100.0, "MLDG_BA": ml_sub[s]["BA"] * 100.0, "delta_BA_pp": (ml_sub[s]["BA"] - er_sub[s]["BA"]) * 100.0} for s in sorted(set(er_sub) & set(ml_sub))])
    traj = []
    for row in ml_sel["history"]: traj.append({"opt_seed": seed, "epoch": row["epoch"], "val_BA": row["val_BA"] * 100.0, "val_NLL": row["val_NLL"], "train_loss": row["train_loss"], "partition_seed": row["partition_seed"], "after_epoch6": bool(int(row["epoch"]) > 6)})
    append_csv(ROOT / "MLDG_VALIDATION_TRAJECTORY.csv", traj)
    print(f"[seed {seed}] ERM epoch={er_sel['selected_epoch']} MLDG epoch={ml_sel['selected_epoch']} ERM BA={er_ref['BA']*100:.4f} MLDG BA={ml_ref['BA']*100:.4f} delta={delta:.4f}pp", flush=True)


def finalize(ctx: Mapping[str, Any]) -> str:
    p = pd.read_csv(ROOT / "PAIRED_RESULTS.csv"); vals = p.delta_BA_pp.to_numpy(float); mean, med = float(np.mean(vals)), float(np.median(vals)); positives = int(np.sum(vals >= 0.0)); collapse = bool(np.any(vals < -5.0))
    strong = bool(len(vals) == 3 and positives == 3 and mean >= 1.0 and med >= 0.5 and not collapse)
    success = bool(len(vals) == 3 and mean >= 0.5 and med >= 0.25 and positives >= 2 and not collapse)
    partial = bool(len(vals) == 3 and mean > 0 and positives >= 2 and not success)
    terminal = "MLDG_PER_SEED_EARLY_STOP_STRONG_RESCUE" if strong else ("PER_SEED_EARLY_STOP_RESCUES_MLDG" if success else ("MLDG_PER_SEED_EARLY_STOP_PARTIAL_ONLY" if partial else "MLDG_PER_SEED_EARLY_STOP_NOT_SUPPORTED"))
    trajectory = pd.read_csv(ROOT / "MLDG_VALIDATION_TRAJECTORY.csv"); traj_rows = []
    for seed, g in trajectory.groupby("opt_seed"):
        e6 = g[g.epoch == 6].val_BA.iloc[0] if len(g[g.epoch == 6]) else float("nan"); post = g[g.epoch > 6]; best_post = float(post.val_BA.max()) if len(post) else float("nan")
        traj_rows.append({"opt_seed": int(seed), "epoch6_val_BA": e6, "best_val_BA_after_epoch6": best_post, "improvement_after_epoch6_pp": best_post - e6 if math.isfinite(best_post) else float("nan"), "improved_after_epoch6": bool(math.isfinite(best_post) and best_post > e6 + 1e-12)})
    write_csv(ROOT / "MLDG_EPOCH6_DIAGNOSTIC.csv", traj_rows)
    summary = {"schema": SCHEMA, "dataset": DATASET, "outer_fold": OUTER, "beta": BETA, "opt_seeds": list(SEEDS), "mean_delta_BA_pp": mean, "median_delta_BA_pp": med, "positive_seeds": positives, "minimum_delta_BA_pp": float(np.min(vals)), "any_delta_below_minus5pp": collapse, "terminal": terminal, "gates": {"G1_mean_ge_0.5": bool(mean >= 0.5), "G2_median_ge_0.25": bool(med >= 0.25), "G3_positive_ge_2_of_3": bool(positives >= 2), "G4_no_delta_below_minus5": bool(not collapse)}, "canonical_outcome_labels_read": False, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False}
    write_json(ROOT / "PER_SEED_GATE.json", summary)
    lines = ["# Per-seed early-stop MLDG check", "", f"Terminal: `{terminal}`", "", "| Seed | ERM epoch | MLDG epoch | ERM BA | MLDG BA | ΔBA |", "|---:|---:|---:|---:|---:|---:|"]
    for _, r in p.sort_values("opt_seed").iterrows(): lines.append(f"| {int(r.opt_seed)} | {int(r.ERM_epoch)} | {int(r.MLDG_epoch)} | {r.ERM_BA:.4f} | {r.MLDG_BA:.4f} | {r.delta_BA_pp:+.4f} |")
    d = pd.DataFrame(traj_rows); lines += ["", f"Mean ΔBA: `{mean:+.4f} pp`; median: `{med:+.4f} pp`; positive seeds: `{positives}/3`; minimum: `{float(np.min(vals)):+.4f} pp`; any < -5 pp: `{collapse}`.", "", "## Answers", "", "1. Selected epochs differ across seeds: see PER_SEED_SELECTION.csv.", "2. Epoch-6 versus post-epoch-6 source-validation BA is in MLDG_EPOCH6_DIAGNOSTIC.csv; held-out BA was not used for selection.", f"3. Per-seed legal early stopping terminal is `{terminal}`.", "4. This single WBCIC fold cannot support a broader final-model claim.", "5. If the failure terminal is present, stop plain MLDG and require a new preregistered hypothesis.", "", "Only WBCIC fold0, EEGNet, ERM/MLDG, and optimization seeds 0/1/2 were run. Split/cache were reused from Route-B; no canonical outcome or WBCIC outer-10 was opened."]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ROOT / "NO_CANONICAL_OUTCOME_ACCESS_AUDIT.json", {"schema": SCHEMA, "canonical_outcome_labels_read": False, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False, "held_labels_used_for_selection": False})
    return terminal


def protocol_lock() -> None:
    write_json(ROOT / "PROTOCOL_LOCK.json", {"schema": SCHEMA, "branch": "codex/persist-eeg-mldg-perseed-earlystop-check-v1", "parent": "codex/persist-eeg-mldg-robustness-confirm-v1", "dataset": DATASET, "outer_fold": OUTER, "backbone": "EEGNet", "methods": ["B0_SUBJECT_BALANCED_ERM", "B2_SUBJECT_EPISODIC_MLDG"], "opt_seeds": list(SEEDS), "beta": BETA, "early_stopping": {"MAX_EPOCHS": MAX_EPOCHS, "MIN_EPOCHS": MIN_EPOCHS, "PATIENCE": PATIENCE, "metric": ["highest source-validation BA", "lowest source-validation NLL", "earliest epoch"]}, "split_source": str(AUDIT_ROOT / "SPLIT_AUDIT.csv"), "order_seed": "sha256(mldg-robustness|WBCIC|fold0|opt_seed|epoch)", "base_rng_seed": "sha256(mldg-robustness|WBCIC|fold0|opt_seed|train)", "mldg_meta_seed": "sha256(mldg-perseed-es-meta|WBCIC|fold0|opt_seed|epoch)", "protocol_revision": "order/base RNG restored to parent; only per-seed source early stopping changes", "canonical_outcome_labels_read": False, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def main() -> int:
    protocol_lock(); ctx = cache_context(); split = split_audit(ctx)
    if not split["pass"]: raise RuntimeError("split reuse audit failed")
    for seed in SEEDS: run_seed(ctx, seed)
    terminal = finalize(ctx); print(terminal, flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
