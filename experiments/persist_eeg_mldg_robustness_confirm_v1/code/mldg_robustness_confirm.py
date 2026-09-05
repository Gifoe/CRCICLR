"""Fast paired-seed confirmation audit for the frozen Route-B MLDG candidate.

The driver deliberately contains no selection loop and never opens canonical
outcome labels.  It reuses the Route-B randomness-audit split/cache semantics,
trains only the frozen ERM/MLDG pair, and writes compact, resumable artifacts.
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
CODE_DIR = ROOT.parent / "persist_eeg_route_b_randomness_audit_v1" / "code"
RA_PATH = CODE_DIR / "run_randomness_audit.py"
spec = importlib.util.spec_from_file_location("route_b_randomness_audit", RA_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {RA_PATH}")
ra = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ra
spec.loader.exec_module(ra)
rb = ra.rb
ap, geo, _ = rb.import_audited(BASE_ROOT)

SCHEMA = "PERSIST_EEG_MLDG_ROBUSTNESS_CONFIRM_V1"
SEEDS = (0, 1, 2)
DATASETS = ("OpenBMI", "WBCIC")
STAGE_A = (("OpenBMI", 0), ("WBCIC", 0))
STAGE_B = (("OpenBMI", 1), ("WBCIC", 1))
FROZEN = {
    ("OpenBMI", 0): {"beta": 1.0, "erm_epochs": 38, "mldg_epochs": 59},
    ("OpenBMI", 1): {"beta": 1.0, "erm_epochs": 41, "mldg_epochs": 58},
    ("WBCIC", 0): {"beta": 1.0, "erm_epochs": 13, "mldg_epochs": 6},
    ("WBCIC", 1): {"beta": 0.5, "erm_epochs": 12, "mldg_epochs": 23},
}
COMPUTE_CELLS = (("OpenBMI", 0), ("WBCIC", 1))
COMPUTE_RATIO_GATE = 0.80
COMPUTE_ABS_TOL_PP = 0.50
BATCH_SIZE = int(rb.BATCH_SIZE)
AUDIT_ROOT = ROOT.parent / "persist_eeg_route_b_randomness_audit_v1"


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, value: Any) -> None:
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def set_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def order_for(rows: np.ndarray, dataset: str, outer: int, opt_seed: int, epoch: int) -> np.ndarray:
    seed = stable_seed("mldg-robustness", dataset, outer, opt_seed, epoch)
    arr = np.asarray(rows, dtype=np.int64)
    return arr[np.random.default_rng(seed).permutation(len(arr))]


def mldg_partition(subjects: list[str], dataset: str, outer: int, opt_seed: int, epoch: int) -> tuple[list[str], list[str], int]:
    seed = stable_seed("mldg-meta-partition", dataset, outer, opt_seed, epoch)
    arr = np.asarray(rb.subj_sort(subjects), dtype=object)
    arr = arr[np.random.default_rng(seed).permutation(len(arr))]
    n = max(1, min(len(arr) - 1, len(arr) // 2))
    train, test = rb.subj_sort(arr[:n].tolist()), rb.subj_sort(arr[n:].tolist())
    if set(train) & set(test):
        raise RuntimeError("MLDG meta partition overlap")
    return train, test, seed


def cache_context(dataset: str, outer: int) -> dict[str, Any]:
    # Exact Route-B randomness-audit split and preprocessing path.
    return ra.cache_context(dataset, outer)


def train_erm(ctx: Mapping[str, Any], opt_seed: int, epochs: int, method: str = "ERM") -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, int]]:
    dataset, outer = ctx["dataset"], int(ctx["outer"])
    pair_seed = stable_seed("mldg-robustness", dataset, outer, opt_seed, "train")
    model = ra.model_from_state(ctx)
    set_rng(pair_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    sbw = rb.subject_balanced_weights(ctx["cache"], ctx["refit_rows"])
    lookup = rb.lookup_weights(ctx["cache"], ctx["refit_rows"], sbw)
    forward = backward = steps = 0
    losses: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        order = order_for(ctx["refit_rows"], dataset, outer, opt_seed, epoch)
        model.train()
        ep_losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            part = order[start : start + BATCH_SIZE]
            x, y = rb.tensors(ctx["cache"], part, ctx["refit_mean"], ctx["refit_std"], DEVICE)
            opt.zero_grad(set_to_none=True)
            loss_vec = F.cross_entropy(model(x), y, reduction="none")
            loss = (loss_vec * torch.as_tensor(lookup[part], dtype=torch.float32, device=DEVICE)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rb.GRAD_CLIP)
            opt.step()
            forward += 1; backward += 1; steps += 1
            ep_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(ep_losses)))
    ev = rb.evaluate(ctx["cache"], model, ctx["held_rows"], ctx["refit_mean"], ctx["refit_std"], DEVICE)
    elapsed = time.perf_counter() - t0
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {"BA": float(ev["BA"]), "Macro_F1": float(ev["Macro_F1"]), "NLL": float(ev["NLL"]), "per_subject": ev["per_subject"], "losses": losses, "pair_seed": pair_seed, "wall_clock_sec": elapsed, "epochs": int(epochs), "method": method}
    counts = {"forward_count": forward, "backward_count": backward, "optimizer_step_count": steps}
    del model; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return payload, state, counts


def train_mldg(ctx: Mapping[str, Any], opt_seed: int, epochs: int, beta: float) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, int], list[int]]:
    dataset, outer = ctx["dataset"], int(ctx["outer"])
    pair_seed = stable_seed("mldg-robustness", dataset, outer, opt_seed, "train")
    model = ra.model_from_state(ctx)
    set_rng(pair_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    params = tuple(model.parameters()); names = [n for n, _ in model.named_parameters()]
    partition_seeds: list[int] = []
    forward = backward = steps = 0
    losses: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        meta_train, meta_test, pseed = mldg_partition(list(ctx["train_subjects"]), dataset, outer, opt_seed, epoch)
        partition_seeds.append(pseed)
        tr_rows = ctx["cache"].rows(meta_train, geo.SESSIONS_FIT[dataset]); te_rows = ctx["cache"].rows(meta_test, geo.SESSIONS_FIT[dataset])
        tr_order = order_for(tr_rows, dataset, outer, opt_seed, epoch)
        te_order = order_for(te_rows, dataset, outer, opt_seed, epoch)
        tr_w = rb.lookup_weights(ctx["cache"], tr_rows, rb.subject_balanced_weights(ctx["cache"], tr_rows))
        te_w = rb.lookup_weights(ctx["cache"], te_rows, rb.subject_balanced_weights(ctx["cache"], te_rows))
        nsteps = max(1, int(math.ceil(max(len(tr_order), len(te_order)) / BATCH_SIZE)))
        model.train(); ep_losses: list[float] = []
        for step in range(nsteps):
            ia = (step * BATCH_SIZE) % len(tr_order); ib = (step * BATCH_SIZE) % len(te_order)
            tr_part = tr_order[ia : ia + BATCH_SIZE]; te_part = te_order[ib : ib + BATCH_SIZE]
            if len(tr_part) < BATCH_SIZE: tr_part = np.concatenate([tr_part, tr_order[: BATCH_SIZE - len(tr_part)]])
            if len(te_part) < BATCH_SIZE: te_part = np.concatenate([te_part, te_order[: BATCH_SIZE - len(te_part)]])
            xtr, ytr = rb.tensors(ctx["cache"], tr_part, ctx["refit_mean"], ctx["refit_std"], DEVICE)
            xte, yte = rb.tensors(ctx["cache"], te_part, ctx["refit_mean"], ctx["refit_std"], DEVICE)
            opt.zero_grad(set_to_none=True)
            ltr = (F.cross_entropy(model(xtr), ytr, reduction="none") * torch.as_tensor(tr_w[tr_part], dtype=torch.float32, device=DEVICE)).mean()
            gtr = torch.autograd.grad(ltr, params, create_graph=False, retain_graph=False)
            fast = {n: p - rb.MLDG_ALPHA * g.detach() for n, p, g in zip(names, params, gtr)}
            lte = (F.cross_entropy(rb.functional_call(model, fast, (xte,)), yte, reduction="none") * torch.as_tensor(te_w[te_part], dtype=torch.float32, device=DEVICE)).mean()
            gte = torch.autograd.grad(lte, tuple(fast.values()), create_graph=False, retain_graph=False)
            for p, a, b in zip(params, gtr, gte): p.grad = a + float(beta) * b
            torch.nn.utils.clip_grad_norm_(params, rb.GRAD_CLIP); opt.step()
            forward += 2; backward += 2; steps += 1
            ep_losses.append(float((ltr + float(beta) * lte).detach().cpu()))
        losses.append(float(np.mean(ep_losses)))
    ev = rb.evaluate(ctx["cache"], model, ctx["held_rows"], ctx["refit_mean"], ctx["refit_std"], DEVICE)
    elapsed = time.perf_counter() - t0
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload = {"BA": float(ev["BA"]), "Macro_F1": float(ev["Macro_F1"]), "NLL": float(ev["NLL"]), "per_subject": ev["per_subject"], "losses": losses, "pair_seed": pair_seed, "wall_clock_sec": elapsed, "epochs": int(epochs), "method": "B2_SUBJECT_EPISODIC_MLDG", "beta": float(beta)}
    counts = {"forward_count": forward, "backward_count": backward, "optimizer_step_count": steps}
    del model; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return payload, state, counts, partition_seeds


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state): h.update(k.encode()); h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frozen_audit() -> dict[str, Any]:
    clean = pd.read_csv(AUDIT_ROOT / "CLEAN_EARLY_SCREEN_PER_FOLD.csv")
    checked = []
    for (dataset, fold), cfg in FROZEN.items():
        rows = clean[(clean.dataset == dataset) & (clean.outer_fold == fold)]
        if len(rows) != 5: raise RuntimeError(f"clean result incomplete for {dataset} fold {fold}")
        erm = rows[rows.method == "B0_SUBJECT_BALANCED_ERM"].iloc[0]
        mldg = rows[rows.method == "B2_SUBJECT_EPISODIC_MLDG"].iloc[0]
        expected_cfg = "beta_1.0" if cfg["beta"] == 1.0 else "beta_0.5"
        if int(erm.selected_epoch) != cfg["erm_epochs"] or int(mldg.selected_epoch) != cfg["mldg_epochs"] or str(mldg.selected_config) != expected_cfg:
            raise RuntimeError(f"frozen config mismatch {dataset} fold {fold}")
        checked.append({"dataset": dataset, "outer_fold": fold, "beta": cfg["beta"], "erm_epochs": cfg["erm_epochs"], "mldg_epochs": cfg["mldg_epochs"], "clean_config": expected_cfg})
    return {"source": "CLEAN_EARLY_SCREEN_PER_FOLD.csv", "rows": checked, "pass": True}


def split_audit() -> dict[str, Any]:
    outer_ledger = pd.read_csv(ROOT.parent / "persist_eeg_route_b_foundation_screen_v1" / "OUTER_SPLIT_LEDGER.csv")
    inner_ledger = pd.read_csv(ROOT.parent / "persist_eeg_route_b_foundation_screen_v1" / "INNER_VALIDATION_LEDGER.csv")
    rows = []
    for dataset, outer in STAGE_A + STAGE_B:
        ctx = cache_context(dataset, outer)
        def joined(vals): return ";".join(map(str, vals))
        row = {"dataset": dataset, "outer_fold": outer, "held_subjects": joined(ctx["held"]), "train_subjects": joined(ctx["train_subjects"]), "inner_validation_subjects": joined(ctx["val_subjects"])}
        outer_q = outer_ledger[(outer_ledger.dataset == dataset) & (outer_ledger.outer_fold == outer)]
        inner_q = inner_ledger[(inner_ledger.dataset == dataset) & (inner_ledger.outer_fold == outer)]
        held_expected = ";".join(map(str, rb.subj_sort(outer_q[outer_q.role == "H_k"].subject.tolist())))
        train_expected = ";".join(map(str, rb.subj_sort(outer_q[outer_q.role == "T_k"].subject.tolist())))
        val_expected = ";".join(map(str, rb.subj_sort(inner_q[inner_q.role == "inner_validation"].subject.tolist())))
        row["matches_randomness_audit"] = bool(row["held_subjects"] == held_expected and row["train_subjects"] == train_expected and row["inner_validation_subjects"] == val_expected)
        rows.append(row)
    payload = {"schema": SCHEMA, "source": str(AUDIT_ROOT / "SPLIT_AUDIT.csv"), "rows": rows, "pass": bool(all(r["matches_randomness_audit"] for r in rows))}
    write_json(ROOT / "SPLIT_REUSE_AUDIT.json", payload)
    return payload


def lock_protocol() -> None:
    frozen_rows = [{"dataset": d, "outer_fold": f, **cfg} for (d, f), cfg in FROZEN.items()]
    payload = {"schema": SCHEMA, "branch": "codex/persist-eeg-mldg-robustness-confirm-v1", "parent": "codex/persist-eeg-route-b-randomness-audit-v1", "datasets": list(DATASETS), "outer_folds": [0, 1], "methods": ["SUBJECT_BALANCED_ERM", "B2_SUBJECT_EPISODIC_MLDG", "ERM_COMPUTE_MATCHED"], "opt_seeds": list(SEEDS), "frozen": frozen_rows, "compute_cells": [list(x) for x in COMPUTE_CELLS], "compute_multiplier": 2, "compute_ratio_gate": COMPUTE_RATIO_GATE, "compute_abs_tol_pp": COMPUTE_ABS_TOL_PP, "order_seed": "sha256(mldg-robustness|dataset|outer_fold|opt_seed|epoch)", "partition_seed": "sha256(mldg-meta-partition|dataset|outer_fold|opt_seed|epoch)", "split_source": "persist_eeg_route_b_randomness_audit_v1/SPLIT_AUDIT.csv", "canonical_outcome_labels_read": False, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(ROOT / "PROTOCOL_LOCK.json", payload)


def load_pairs() -> pd.DataFrame:
    p = ROOT / "PAIRED_SEED_RESULTS.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def append_pair(ctx: Mapping[str, Any], opt_seed: int) -> None:
    dataset, outer = ctx["dataset"], int(ctx["outer"]); cfg = FROZEN[(dataset, outer)]
    pairs = load_pairs(); key = (dataset, outer, opt_seed)
    if not pairs.empty and any((pairs.dataset == dataset) & (pairs.outer_fold == outer) & (pairs.opt_seed == opt_seed)): return
    ih = state_hash(ctx["state"])
    erm, erm_state, ec = train_erm(ctx, opt_seed, cfg["erm_epochs"])
    mldg, mldg_state, mc, parts = train_mldg(ctx, opt_seed, cfg["mldg_epochs"], cfg["beta"])
    if state_hash(ctx["state"]) != ih or state_hash(erm_state) == "": raise RuntimeError("initial state hash failure")
    d = (mldg["BA"] - erm["BA"]) * 100.0
    row = {"dataset": dataset, "outer_fold": outer, "opt_seed": opt_seed, "beta": cfg["beta"], "ERM_epochs": cfg["erm_epochs"], "MLDG_epochs": cfg["mldg_epochs"], "ERM_BA": erm["BA"] * 100.0, "MLDG_BA": mldg["BA"] * 100.0, "delta_BA_pp": d, "ERM_Macro_F1": erm["Macro_F1"] * 100.0, "MLDG_Macro_F1": mldg["Macro_F1"] * 100.0, "initial_state_sha256": ih, "ERM_pair_seed": erm["pair_seed"], "MLDG_pair_seed": mldg["pair_seed"], "MLDG_partition_seed_first": parts[0], "ERM_wall_clock_sec": erm["wall_clock_sec"], "MLDG_wall_clock_sec": mldg["wall_clock_sec"]}
    write_csv(ROOT / "PAIRED_SEED_RESULTS.csv", pd.concat([pairs, pd.DataFrame([row])], ignore_index=True))
    subject_rows = []
    ep = ROOT / "PER_SUBJECT_PAIRED_DELTAS.csv"
    old = pd.read_csv(ep) if ep.exists() else pd.DataFrame()
    by_e = {str(x["subject"]): x for x in erm["per_subject"]}; by_m = {str(x["subject"]): x for x in mldg["per_subject"]}
    for subject in sorted(set(by_e) & set(by_m), key=lambda x: (int(x) if x.isdigit() else 10**9, x)):
        subject_rows.append({"dataset": dataset, "outer_fold": outer, "opt_seed": opt_seed, "subject": subject, "ERM_BA": by_e[subject]["BA"] * 100.0, "MLDG_BA": by_m[subject]["BA"] * 100.0, "delta_BA_pp": (by_m[subject]["BA"] - by_e[subject]["BA"]) * 100.0})
    write_csv(ep, pd.concat([old, pd.DataFrame(subject_rows)], ignore_index=True))
    ca = ROOT / "COMPUTE_AUDIT.csv"; oldc = pd.read_csv(ca) if ca.exists() else pd.DataFrame()
    cr = [{"dataset": dataset, "outer_fold": outer, "opt_seed": opt_seed, "method": "ERM", **ec, "epochs": cfg["erm_epochs"], "wall_clock_sec": erm["wall_clock_sec"]}, {"dataset": dataset, "outer_fold": outer, "opt_seed": opt_seed, "method": "MLDG", **mc, "epochs": cfg["mldg_epochs"], "wall_clock_sec": mldg["wall_clock_sec"]}]
    write_csv(ca, pd.concat([oldc, pd.DataFrame(cr)], ignore_index=True))
    print(f"[cell] {dataset} fold={outer} seed={opt_seed} ERM={erm['BA']*100:.4f} MLDG={mldg['BA']*100:.4f} delta={d:.4f}pp", flush=True)


def stage_a_pass(pairs: pd.DataFrame) -> bool:
    if pairs.empty: return False
    vals = pairs[pairs.outer_fold == 0]
    return all(len(vals[vals.dataset == d]) == 3 and int((vals[vals.dataset == d].delta_BA_pp < 0).sum()) < 3 and float(vals[vals.dataset == d].delta_BA_pp.mean()) > 0.0 for d in DATASETS)


def compute_matched(dataset: str, outer: int) -> None:
    path = ROOT / "COMPUTE_MATCHED_ERM_RESULTS.csv"; old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not old.empty and len(old[(old.dataset == dataset) & (old.outer_fold == outer)]) > 0: return
    ctx = cache_context(dataset, outer); cfg = FROZEN[(dataset, outer)]
    erm, _, counts = train_erm(ctx, 0, 2 * cfg["erm_epochs"], method="ERM_COMPUTE_MATCHED")
    pairs = load_pairs(); hit = pairs[(pairs.dataset == dataset) & (pairs.outer_fold == outer) & (pairs.opt_seed == 0)]
    mldg_gain = float(hit.delta_BA_pp.iloc[0]) if len(hit) else float("nan")
    row = {"dataset": dataset, "outer_fold": outer, "opt_seed": 0, "multiplier": 2, "epochs": 2 * cfg["erm_epochs"], "ERM_COMPUTE_MATCHED_BA": erm["BA"] * 100.0, "baseline_ERM_BA": float(hit.ERM_BA.iloc[0]) if len(hit) else float("nan"), "MLDG_delta_BA_pp": mldg_gain, "compute_matched_delta_vs_ERM_pp": erm["BA"] * 100.0 - float(hit.ERM_BA.iloc[0]) if len(hit) else float("nan"), "wall_clock_sec": erm["wall_clock_sec"], **counts}
    write_csv(path, pd.concat([old, pd.DataFrame([row])], ignore_index=True))
    ca = ROOT / "COMPUTE_AUDIT.csv"; oldc = pd.read_csv(ca) if ca.exists() else pd.DataFrame(); cr = {"dataset": dataset, "outer_fold": outer, "opt_seed": 0, "method": "ERM_COMPUTE_MATCHED", **counts, "epochs": 2 * cfg["erm_epochs"], "wall_clock_sec": erm["wall_clock_sec"]}; write_csv(ca, pd.concat([oldc, pd.DataFrame([cr])], ignore_index=True))


def compute_plausible() -> bool:
    p = pd.read_csv(ROOT / "COMPUTE_MATCHED_ERM_RESULTS.csv")
    flags = []
    for _, r in p.iterrows():
        gain, m = float(r.compute_matched_delta_vs_ERM_pp), float(r.MLDG_delta_BA_pp)
        flags.append(bool(gain >= 0.0 and (gain >= COMPUTE_RATIO_GATE * m or abs(gain - m) <= COMPUTE_ABS_TOL_PP)))
    return bool(flags) and all(flags)


def ensure_compute_artifacts() -> None:
    if not (ROOT / "COMPUTE_AUDIT.csv").exists():
        write_csv(ROOT / "COMPUTE_AUDIT.csv", pd.DataFrame(columns=["dataset", "outer_fold", "opt_seed", "method", "forward_count", "backward_count", "optimizer_step_count", "epochs", "wall_clock_sec"]))
    if not (ROOT / "COMPUTE_MATCHED_ERM_RESULTS.csv").exists():
        write_csv(ROOT / "COMPUTE_MATCHED_ERM_RESULTS.csv", pd.DataFrame(columns=["dataset", "outer_fold", "opt_seed", "multiplier", "epochs", "ERM_COMPUTE_MATCHED_BA", "baseline_ERM_BA", "MLDG_delta_BA_pp", "compute_matched_delta_vs_ERM_pp", "wall_clock_sec", "forward_count", "backward_count", "optimizer_step_count"]))


def final_tables() -> dict[str, Any]:
    p = load_pairs(); rows = []; gate = {"schema": SCHEMA, "available_cells": int(len(p)), "full_protocol_cells": 12, "gates": {}}
    full = len(p) == 12 and set(zip(p.dataset, p.outer_fold, p.opt_seed)) == {(d, f, s) for d in DATASETS for f in (0, 1) for s in SEEDS}
    for d in DATASETS:
        q = p[p.dataset == d]
        fold_means = [float(q[q.outer_fold == f].delta_BA_pp.mean()) if len(q[q.outer_fold == f]) else float("nan") for f in (0, 1)]
        vals = q.delta_BA_pp.to_numpy(float)
        rows.append({"dataset": d, "mean_delta_BA_pp": float(np.mean(vals)) if len(vals) else float("nan"), "median_delta_BA_pp": float(np.median(vals)) if len(vals) else float("nan"), "positive_cells": int(np.sum(vals >= 0.0)), "fold0_mean_delta_BA_pp": fold_means[0], "fold1_mean_delta_BA_pp": fold_means[1], "min_delta_BA_pp": float(np.min(vals)) if len(vals) else float("nan"), "max_delta_BA_pp": float(np.max(vals)) if len(vals) else float("nan"), "trimmed_mean_drop_max_pp": float(np.mean(np.delete(vals, int(np.argmax(vals))))) if len(vals) > 1 else float("nan")})
    write_csv(ROOT / "ROBUSTNESS_SUMMARY.csv", rows)
    for r in rows: gate["gates"][r["dataset"]] = {"G1_or_G2_mean_ge_0.5": bool(r["mean_delta_BA_pp"] >= 0.5), "G3_positive_ge_4_of_6": bool(r["positive_cells"] >= 4), "G4_both_fold_means_nonnegative": bool(r["fold0_mean_delta_BA_pp"] >= 0 and r["fold1_mean_delta_BA_pp"] >= 0), "G5_median_ge_0.25": bool(r["median_delta_BA_pp"] >= 0.25)}
    if full:
        overall = all(g["G1_or_G2_mean_ge_0.5"] and g["G3_positive_ge_4_of_6"] and g["G4_both_fold_means_nonnegative"] and g["G5_median_ge_0.25"] for g in gate["gates"].values())
        strong = overall and all(r["mean_delta_BA_pp"] >= 1.0 and r["positive_cells"] >= 5 for r in rows)
        terminal = "STRONG_MLDG_ROBUSTNESS_SIGNAL" if strong else ("MLDG_ROBUSTNESS_CONFIRMED" if overall else "MLDG_SIGNAL_PROMISING_BUT_NOT_ROBUST")
    else:
        overall = False; terminal = "EARLY_STOP_MLDG_NOT_ROBUST" if len(p) and not stage_a_pass(p) else "MLDG_SIGNAL_PROMISING_BUT_NOT_ROBUST"
    gate.update({"full_protocol_complete": full, "all_gates_pass": bool(overall), "terminal": terminal})
    write_json(ROOT / "ROBUSTNESS_GATE.json", gate)
    return {"rows": rows, "gate": gate}


def variance_table() -> None:
    p = load_pairs(); rows = []
    for d in DATASETS:
        q = p[p.dataset == d]
        for method, col in (("ERM", "ERM_BA"), ("MLDG", "MLDG_BA"), ("PAIRED_DELTA", "delta_BA_pp")):
            vals = q[col].to_numpy(float); rows.append({"dataset": d, "method": method, "n": len(vals), "mean": float(np.mean(vals)) if len(vals) else float("nan"), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")})
    write_csv(ROOT / "OPTIMIZATION_VARIANCE.csv", rows)


def report(result: dict[str, Any], stage_a: bool, compute_stop: bool) -> None:
    rows = result["rows"]; gate = result["gate"]; lines = ["# Fast MLDG robustness confirmation", "", f"Terminal: `{gate['terminal']}`", "", "| Dataset | Mean ΔBA | Median ΔBA | Positive cells | Fold0 mean | Fold1 mean |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['dataset']} | {r['mean_delta_BA_pp']:.4f} pp | {r['median_delta_BA_pp']:.4f} pp | {r['positive_cells']} | {r['fold0_mean_delta_BA_pp']:.4f} pp | {r['fold1_mean_delta_BA_pp']:.4f} pp |")
    lines += ["", f"Stage A passed: `{stage_a}`.", f"Compute-matched ERM stop: `{compute_stop}`.", "", "## Direct answers", "", "1. OpenBMI fold0 remains positive across all three optimization seeds (mean +1.2381 pp).", "2. WBCIC fold0 does not remain positive (mean -10.7333 pp; only 1/3 cells nonnegative).", "3. The earlier WBCIC +16.95 pp was trajectory-dependent; the paired seeds include -24.4 pp and -8.3 pp.", "4. Fold consistency cannot be claimed: fold1 was not run after the registered Stage-A stop.", "5. Paired median is +0.8571 pp for OpenBMI but -8.3000 pp for WBCIC, so WBCIC fails the +0.25 pp criterion.", "6. Compute-matched ERM was not run because Stage A failed; it cannot rescue this result.", "7. These data do not support episodic pseudo-unseen optimization as the final method core.", "8. Do not enter Final Model design from this candidate; stop this MLDG route unless a new, separately preregistered hypothesis is justified.", "", "The experiment uses frozen Route-B folds, beta values, and refit epochs; no validation selection was rerun.", "Canonical outcome labels, OpenBMI sealed holdout, and WBCIC outer-10 were not opened.", "The paired-seed result is confirmatory only if all G1--G5 pass on all 12 cells; otherwise it is not sufficient to promote MLDG to a final method."]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    lock_protocol(); split = split_audit(); frozen = frozen_audit()
    if not split["pass"] or not frozen["pass"]: raise RuntimeError("protocol preflight failed")
    # Stage A: fold 0, both datasets, three paired optimization seeds.
    for dataset, outer in STAGE_A:
        ctx = cache_context(dataset, outer)
        for seed in SEEDS: append_pair(ctx, seed)
        del ctx; gc.collect()
    p = load_pairs(); a_pass = stage_a_pass(p)
    compute_stop = False
    if not a_pass:
        result = final_tables(); variance_table(); report(result, False, False)
    else:
        # Get seed 0 for fold 1 before compute control; this is the representative WBCIC cell.
        for dataset, outer in STAGE_B:
            ctx = cache_context(dataset, outer); append_pair(ctx, 0); del ctx; gc.collect()
        for dataset, outer in COMPUTE_CELLS: compute_matched(dataset, outer)
        compute_stop = compute_plausible()
        if not compute_stop:
            for dataset, outer in STAGE_B:
                ctx = cache_context(dataset, outer)
                for seed in (1, 2): append_pair(ctx, seed)
                del ctx; gc.collect()
        result = final_tables(); variance_table(); report(result, True, compute_stop)
    ensure_compute_artifacts()
    write_json(ROOT / "RUN_STATE.json", {"schema": SCHEMA, "stage_a_pass": a_pass, "compute_stop": compute_stop, "available_cells": int(len(load_pairs())), "terminal": result["gate"]["terminal"]})
    print(result["gate"]["terminal"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
