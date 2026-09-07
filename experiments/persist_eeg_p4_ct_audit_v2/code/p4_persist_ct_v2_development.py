from __future__ import annotations

import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import p4_persist_ct as base
import p4_persist_ct_v2 as v2

ROOT = base.ROOT
OUT = ROOT / "outputs" / "persist_eeg_p4_ct_v2"
TASKS = base.TASKS
CLASSES = base.CLASSES
FOLDS = base.FOLDS
SEEDS = base.SEEDS
EPOCHS = 18
PATIENCE = 5
K = 2
TAU = 0.5
LAMBDA_CT = 0.35
LAMBDA_CONS = 0.10


def coords(h: np.ndarray, spec: dict) -> np.ndarray:
    return (np.asarray(h, dtype=np.float64) - spec["mean"]) @ spec["whitener"] @ spec["directions"]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_positions(meta: pd.DataFrame, task: str) -> np.ndarray:
    return np.flatnonzero((meta.paradigm == task).to_numpy())


def build_shift_bank(meta: pd.DataFrame, h: np.ndarray, spec: dict, task: str, nuisance_blocks: list[int]) -> dict[tuple[str, str], np.ndarray]:
    q = coords(h, spec)
    dims = sorted(set(sum((spec["blocks"][b] for b in nuisance_blocks), [])))
    bank: dict[tuple[str, str], np.ndarray] = {}
    for (subject, paradigm, event), g in meta.groupby(["subject_id", "paradigm", "event_label"], sort=True):
        if str(paradigm) == task:
            bank[(str(subject), str(event))] = q[g.index.to_numpy(dtype=np.int64)].mean(axis=0)
    for event in sorted(set(k[1] for k in bank)):
        vals = [v for (s, e), v in bank.items() if e == event]
        mean = np.mean(vals, axis=0)
        for key in [k for k in bank if k[1] == event]:
            d = bank[key] - mean
            mask = np.zeros(q.shape[1], dtype=np.float32); mask[dims] = 1.0
            bank[key] = d * mask
    return bank


def transport(h: np.ndarray, meta: pd.DataFrame, spec: dict, bank: dict[tuple[str, str], np.ndarray], seed: int, amplitude: float = 1.0) -> np.ndarray:
    q = coords(h, spec)
    out = h.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    donors = {}
    for subject, event in bank:
        donors.setdefault(event, []).append(subject)
    for i, row in meta.reset_index(drop=True).iterrows():
        key = (str(row.subject_id), str(row.event_label))
        choices = [s for s in donors.get(key[1], []) if s != key[0]]
        if not choices or key not in bank:
            continue
        donor = choices[int(rng.integers(len(choices)))]
        dq = amplitude * (bank[(donor, key[1])] - bank[key])
        out[i] = out[i] + ((dq @ spec["directions"].T) @ spec["dewhitener"]).astype(np.float32)
    return out


def make_heads(model: nn.Module) -> nn.ModuleDict:
    heads = nn.ModuleDict({t: nn.Linear(128, CLASSES[t]) for t in TASKS})
    heads.load_state_dict(model.heads.state_dict())
    return heads


def evaluate(heads: nn.ModuleDict, hv: dict[str, np.ndarray], yv: dict[str, np.ndarray]) -> dict[str, float]:
    heads.eval(); out = {}
    with torch.no_grad():
        for task in TASKS:
            pred = heads[task](torch.as_tensor(hv[task], dtype=torch.float32)).argmax(1).numpy()
            out[task] = float(base.balanced_accuracy_score(yv[task], pred))
    return out


def train_variant(model: nn.Module, htr: dict[str, np.ndarray], ytr: dict[str, np.ndarray], cf: dict[str, list[np.ndarray]], hv: dict[str, np.ndarray], yv: dict[str, np.ndarray], seed: int, ct: bool, out: Path, amplitude: float = 1.0, lambda_ct: float = LAMBDA_CT, lambda_cons: float = LAMBDA_CONS) -> dict:
    torch.manual_seed(500_000 + seed)
    heads = make_heads(model)
    opt = torch.optim.Adam(heads.parameters(), lr=3e-3, weight_decay=1e-3)
    best = -float("inf"); best_state = None; best_epoch = 0; stale = 0; curve = []
    for epoch in range(EPOCHS):
        heads.train(); losses = []
        for task in TASKS:
            x = torch.as_tensor(htr[task], dtype=torch.float32)
            y = torch.as_tensor(ytr[task], dtype=torch.long)
            perm = torch.randperm(len(y))[: min(len(y), 2048)]
            clean_logits = heads[task](x[perm])
            clean_loss = F.cross_entropy(clean_logits, y[perm])
            loss = clean_loss
            if ct and cf[task]:
                cf_losses = []
                kl_losses = []
                p_clean = torch.softmax(clean_logits.detach(), dim=1)
                for cf_x_np in cf[task]:
                    cf_logits = heads[task](torch.as_tensor(cf_x_np[perm.numpy()], dtype=torch.float32))
                    cf_losses.append(F.cross_entropy(cf_logits, y[perm]))
                    kl_losses.append(F.kl_div(torch.log_softmax(cf_logits, dim=1), p_clean, reduction="batchmean"))
                cf_stack = torch.stack(cf_losses)
                l_ct = TAU * (torch.logsumexp(cf_stack / TAU, dim=0) - np.log(len(cf_losses)))
                l_cons = torch.stack(kl_losses).mean()
                loss = clean_loss + lambda_ct * l_ct + lambda_cons * l_cons
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(heads.parameters(), 5.0); opt.step(); losses.append(float(loss.detach()))
        metrics = evaluate(heads, hv, yv); macro = float(np.mean(list(metrics.values())))
        curve.append({"epoch": epoch, "macro_BA": macro, **{f"BA_{t}": metrics[t] for t in TASKS}, "loss": float(np.mean(losses))})
        if macro > best + 1e-8:
            best, best_epoch, stale = macro, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        heads.load_state_dict(best_state)
    final = evaluate(heads, hv, yv)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curve).to_csv(out / "curves.csv", index=False)
    torch.save({"heads": heads.state_dict(), "best_epoch": best_epoch, "metrics": final, "ct": ct, "amplitude": amplitude}, out / "best.pt")
    return {"task_BA": final, "macro_BA": float(np.mean(list(final.values()))), "best_epoch": best_epoch, "epochs_run": len(curve), "ct": ct, "amplitude": amplitude, "loss": {"lambda_CT": lambda_ct if ct else 0.0, "lambda_cons": lambda_cons if ct else 0.0, "tau": TAU, "K": len(cf[TASKS[0]]) if ct else 0}}


def run_one(fold: int, seed: int, device: torch.device, version: str = "CT_V0", amplitude: float = 1.0, k_value: int = K, lambda_ct: float = LAMBDA_CT, lambda_cons: float = LAMBDA_CONS) -> dict:
    m = base.load_manifest(); split = next(x for x in base.load_splits() if int(x["fold"]) == fold)
    ckpt, mean, std = base.historical(fold, seed)
    model = base.load_model(ckpt, m, device)
    tr_meta, tr_h, tr_y = base.extract(model, m, split["train_subjects"], mean, std, device, 290_000 + fold * 101 + seed, cap=3000)
    va_meta, va_h, va_y = base.extract(model, m, split["validation_subjects"], mean, std, device, 300_000 + fold * 101 + seed, cap=1500)
    spec = v2.build_spectrum_v2(tr_meta, tr_h, 40_000 + fold * 101 + seed)
    audit = json.loads((OUT / "audit" / f"fold-{fold}" / f"seed-{seed}" / "AUDIT_V2.json").read_text(encoding="utf-8"))
    htr, ytr, hv, yv, cf = {}, {}, {}, {}, {}
    for ti, task in enumerate(TASKS):
        ti_idx, vi_idx = task_positions(tr_meta, task), task_positions(va_meta, task)
        htr[task], ytr[task], hv[task], yv[task] = tr_h[ti_idx], tr_y[ti_idx], va_h[vi_idx], va_y[vi_idx]
        nuisance = audit["assignments"][task]["nuisance"]
        bank = build_shift_bank(tr_meta.iloc[ti_idx].reset_index(drop=True), htr[task], spec, task, nuisance)
        cf[task] = []
        for k in range(k_value):
            cf[task].append(transport(htr[task], tr_meta.iloc[ti_idx].reset_index(drop=True), spec, bank, 600_000 + fold * 100 + seed * 10 + ti + k, amplitude=amplitude))
    control = train_variant(model, htr, ytr, {t: [] for t in TASKS}, hv, yv, seed, False, OUT / "controls" / "CONTINUED_TRAINING" / f"fold-{fold}" / f"seed-{seed}")
    ct = train_variant(model, htr, ytr, cf, hv, yv, seed, True, OUT / "development" / version / f"fold-{fold}" / f"seed-{seed}", amplitude=amplitude, lambda_ct=lambda_ct, lambda_cons=lambda_cons)
    result = {"fold": fold, "seed": seed, "version": version, "control": control, "ct": ct, "delta_macro_BA": ct["macro_BA"] - control["macro_BA"], "train_subjects": split["train_subjects"], "validation_subjects": split["validation_subjects"], "outer_test_used": False, "historical_checkpoint": str(ckpt.relative_to(ROOT)).replace("\\", "/"), "method_config": {"frozen_encoder": True, "loss": ct["loss"], "transport_amplitude": amplitude}}
    run = OUT / "development" / version / f"fold-{fold}" / f"seed-{seed}"
    write_json(run / "DEVELOPMENT_RESULT.json", result)
    return result


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CT development requires GPU")
    results = []
    folds = [args.fold] if args.fold is not None else list(FOLDS)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for fold in folds:
        for seed in seeds:
            print(f"[P4-CT-V2-DEV] CT_V1 fold={fold} seed={seed}", flush=True)
            results.append(run_one(fold, seed, device, version="CT_V1", amplitude=1.5, k_value=4, lambda_ct=0.60, lambda_cons=0.05))
            gc.collect()
            torch.cuda.empty_cache()
    rows = [{"fold": r["fold"], "seed": r["seed"], "control_macro_BA": r["control"]["macro_BA"], "ct_macro_BA": r["ct"]["macro_BA"], "delta_macro_BA": r["delta_macro_BA"], "outer_test_used": False} for r in results]
    pd.DataFrame(rows).to_csv(OUT / "P4_CT_DEVELOPMENT_SUMMARY.csv", index=False)
    positive = sum(r["delta_macro_BA"] > 0 for r in results)
    mean_delta = float(np.mean([r["delta_macro_BA"] for r in results]))
    status = "P4_CT_VIABLE" if mean_delta >= 0.003 and positive >= 4 else "P4_CT_NOT_SUPPORTED"
    write_json(OUT / "P4_CT_DEVELOPMENT_REPORT_CT_V1.json", {"status": status, "version": "CT_V1", "mean_delta_macro_BA": mean_delta, "positive_runs": positive, "runs": rows, "outer_test_used": False})
    log_path = OUT / "P4_CT_ADAPTATION_LOG.json"
    previous = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    previous.setdefault("development_versions", [])
    previous["development_versions"].append({"version": "CT_V1", "failure": None if status == "P4_CT_VIABLE" else "CT-V1 development gate not met", "evidence": {"mean_delta_macro_BA": mean_delta, "positive_runs": positive}, "modification": "K=4, 1.5x empirical nuisance transport amplitude, lambda_CT=0.60, lambda_cons=0.05", "why_it_addresses_failure": "CT-V0 showed near-zero clean gain, indicating insufficient counterfactual pressure", "data_used": ["TRAIN", "VALIDATION"], "outer_test_used": False})
    write_json(log_path, previous)
    print(json.dumps(base.clean({"status": status, "mean_delta_macro_BA": mean_delta, "positive_runs": positive}), indent=2), flush=True)


if __name__ == "__main__":
    main()
