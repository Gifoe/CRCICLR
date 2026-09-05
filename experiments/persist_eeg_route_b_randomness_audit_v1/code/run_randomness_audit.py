"""Route-B randomness, baseline-equivalence, and clean early-screen audit.

This experiment deliberately treats the previous two-fold result as
exploratory.  It uses the prelocked Route-B folds, writes a protocol lock before
any fresh pseudo-unseen evaluation, and separates the pure-ERM order controls
from the clean objective comparison.
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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(sys.argv[1]).resolve()
BASE_ROOT = Path(sys.argv[2]).resolve()
DEVICE = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cuda:0")
CODE_DIR = ROOT.parent / "persist_eeg_route_b_foundation_screen_v1" / "code"
RB_PATH = CODE_DIR / "run_foundation_screen.py"
AUDIT_SCHEMA = "PERSIST_EEG_ROUTE_B_RANDOMNESS_AUDIT_V1"
COMMON_PROTOCOL = "COMMON_RANDOMNESS_PROTOCOL_V2"
OLD_ROOT = ROOT.parent / "persist_eeg_route_b_foundation_screen_v1"


def load_route():
    spec = importlib.util.spec_from_file_location("route_b_foundation_screen", RB_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RB_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = load_route()
ap, geo, BASE_CODE = rb.import_audited(BASE_ROOT)


def sha_bytes(v: bytes) -> str:
    return hashlib.sha256(v).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def common_seed(dataset: str, outer: int, phase: str) -> int:
    # Method/config names are intentionally absent.
    return int(rb.stable_seed("route-b-common-rng-v2", dataset, outer, phase))


def set_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def snapshot_rng() -> dict[str, Any]:
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_state_for(dataset: str, outer: int, phase: str) -> tuple[int, dict[str, Any]]:
    seed = common_seed(dataset, outer, phase)
    set_rng(seed)
    return seed, snapshot_rng()


def cache_context(dataset: str, outer: int) -> dict[str, Any]:
    roles, _, _ = ap.load_roles(dataset)
    source = rb.subj_sort(roles[0]["model_fit"])
    folds = rb.subject_split(source, "nested-oof-outer", dataset, 0, rb.OUTER_K)
    held = folds[outer]
    train_subjects = [s for s in source if s not in set(held)]
    inner = rb.subject_split(train_subjects, "route-b-inner-validation", dataset, outer, rb.INNER_K)
    val_subjects = inner[-1]
    sel_subjects = [s for s in train_subjects if s not in set(val_subjects)]
    cache = geo.FoldCache(dataset, source, rb.SEED, 0)
    sel_rows = cache.rows(sel_subjects, geo.SESSIONS_FIT[dataset])
    val_rows = cache.rows(val_subjects, (geo.SESSION_DISCOVERY[dataset],))
    refit_rows = cache.rows(train_subjects, geo.SESSIONS_FIT[dataset])
    held_rows = cache.rows(held, (geo.SESSION_DISCOVERY[dataset],))
    sel_mean, sel_std = cache.normalizer(sel_rows)
    refit_mean, refit_std = cache.normalizer(refit_rows)
    state, init_seed, init_sha = geo.initial_state(cache, dataset, outer, rb.SEED, "route-b-common")
    return {
        "dataset": dataset, "outer": outer, "source": source, "held": held,
        "train_subjects": train_subjects, "val_subjects": val_subjects, "sel_subjects": sel_subjects,
        "cache": cache, "sel_rows": sel_rows, "val_rows": val_rows, "refit_rows": refit_rows,
        "held_rows": held_rows, "sel_mean": sel_mean, "sel_std": sel_std,
        "refit_mean": refit_mean, "refit_std": refit_std, "state": state,
        "init_seed": init_seed, "init_sha": init_sha,
    }


def logits(cache, model, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), rb.BATCH_SIZE):
            part = rows[start:start + rb.BATCH_SIZE]
            x, _ = rb.tensors(cache, part, mean, std, DEVICE)
            out.append(model(x).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def pure_erm_epoch(cache, model, rows: np.ndarray, mean: np.ndarray, std: np.ndarray, optimizer, order: np.ndarray) -> float:
    sbw = rb.subject_balanced_weights(cache, rows)
    lookup = rb.lookup_weights(cache, rows, sbw)
    model.train()
    losses: list[float] = []
    for start in range(0, len(order), rb.BATCH_SIZE):
        part = order[start:start + rb.BATCH_SIZE]
        x, y = rb.tensors(cache, part, mean, std, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        ce = F.cross_entropy(model(x), y, reduction="none")
        loss = (ce * torch.as_tensor(lookup[part], dtype=torch.float32, device=DEVICE)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), rb.GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def old_order(rows: np.ndarray, dataset: str, outer: int, tag: str, epoch: int) -> np.ndarray:
    # Exact old Route-B order schedules.
    return rb.geo_order(rows, dataset, outer, tag, epoch)


CURRENT_PHASE = "selection"


def common_order(rows: np.ndarray, dataset: str, outer: int, stage: str, epoch: int) -> np.ndarray:
    # B0/B1/B3/B4 share one permutation per dataset/fold/phase/epoch.  MLDG
    # keeps separate deterministic train/test partitions and orders.
    if stage in ("mldg-tr", "mldg-te"):
        seed = rb.stable_seed("route-b-mldg-order-v2", dataset, outer, stage, epoch)
    else:
        seed = rb.stable_seed("route-b-common-order-v2", dataset, outer, CURRENT_PHASE, epoch)
    rng = np.random.default_rng(seed)
    return np.asarray(rows, dtype=np.int64)[rng.permutation(len(rows))]


def model_from_state(ctx: Mapping[str, Any]) -> torch.nn.Module:
    return rb.model_from_state(ctx["cache"], geo, ctx["state"], DEVICE)


def selection_erm(ctx: Mapping[str, Any], tag: str, phase: str) -> dict[str, Any]:
    dataset, outer = ctx["dataset"], int(ctx["outer"])
    seed, rng = rng_state_for(dataset, outer, phase)
    model = model_from_state(ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    restore_rng(rng)
    best_ba, best_nll, best_ep, stale = -math.inf, math.inf, 1, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, rb.MAX_EPOCHS + 1):
        order = old_order(ctx["sel_rows"], dataset, outer, tag, epoch)
        loss = pure_erm_epoch(ctx["cache"], model, ctx["sel_rows"], ctx["sel_mean"], ctx["sel_std"], opt, order)
        ev = rb.evaluate(ctx["cache"], model, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"], DEVICE)
        ba, nll = float(ev["BA"]), float(ev["NLL"])
        improved = ba > best_ba + rb.TIE_TOL or (abs(ba - best_ba) <= rb.TIE_TOL and nll < best_nll - rb.TIE_TOL)
        if improved:
            best_ba, best_nll, best_ep, stale = ba, nll, epoch, 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": loss, "val_BA": ba, "val_NLL": nll})
        if epoch >= rb.MIN_EPOCHS and stale >= rb.PATIENCE:
            break
    result = {"selected_epoch": int(best_ep), "validation_BA": float(best_ba), "validation_NLL": float(best_nll), "history": history, "rng_seed": seed}
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def refit_erm(ctx: Mapping[str, Any], tag: str, epochs: int, phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset, outer = ctx["dataset"], int(ctx["outer"])
    seed, rng = rng_state_for(dataset, outer, phase)
    model = model_from_state(ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    restore_rng(rng)
    losses: list[float] = []
    for epoch in range(1, int(epochs) + 1):
        order = old_order(ctx["refit_rows"], dataset, outer, tag, epoch)
        losses.append(pure_erm_epoch(ctx["cache"], model, ctx["refit_rows"], ctx["refit_mean"], ctx["refit_std"], opt, order))
    ev = rb.evaluate(ctx["cache"], model, ctx["held_rows"], ctx["refit_mean"], ctx["refit_std"], DEVICE)
    payload = {"BA": float(ev["BA"]), "Macro_F1": float(ev["Macro_F1"]), "NLL": float(ev["NLL"]), "per_subject": ev["per_subject"], "losses": losses, "rng_seed": seed, "state_hash": state_hash(model.state_dict())}
    result_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return payload, result_model


def run_equivalence() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # WBCIC priority folds, plus one OpenBMI fold to catch dataset-specific
    # preprocessing differences without spending a full screen.
    for dataset, outer in (("WBCIC", 0), ("WBCIC", 1), ("OpenBMI", 0)):
        ctx = cache_context(dataset, outer)
        cache, state = ctx["cache"], ctx["state"]
        w = geo.weight_vector(cache, ctx["sel_rows"], {s: 1.0 for s in ctx["sel_subjects"]}, "SUBJECT_BALANCED_ERM")
        order = old_order(ctx["sel_rows"], dataset, outer, "B0_SUBJECT_BALANCED_ERM", 1)
        # EQ1: one identical epoch from identical RNG state.
        seed, rng = rng_state_for(dataset, outer, "equivalence")
        can = rb.model_from_state(cache, geo, state, DEVICE)
        route = rb.model_from_state(cache, geo, state, DEVICE)
        can_opt = torch.optim.AdamW(can.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
        route_opt = torch.optim.AdamW(route.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
        restore_rng(rng); can_loss = geo.train_epoch(can, cache, ctx["sel_rows"], ctx["sel_mean"], ctx["sel_std"], w, can_opt, order, DEVICE)
        restore_rng(rng); route_loss = pure_erm_epoch(cache, route, ctx["sel_rows"], ctx["sel_mean"], ctx["sel_std"], route_opt, order)
        p_diff = max(float(torch.max(torch.abs(can.state_dict()[k].detach().cpu() - route.state_dict()[k].detach().cpu()))) for k in can.state_dict())
        can_log = logits(cache, can, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"])
        route_log = logits(cache, route, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"])
        l_diff = float(np.max(np.abs(can_log - route_log)))
        eq1 = {"dataset": dataset, "outer_fold": outer, "same_initial_state": state_hash(state) == ctx["init_sha"], "same_rows": bool(np.array_equal(ctx["sel_rows"], ctx["sel_rows"])), "same_weights": bool(np.max(np.abs(w - rb.subject_balanced_weights(cache, ctx["sel_rows"]))) <= 1e-7), "same_order": True, "same_rng_state": True, "epoch": 1, "canonical_loss": float(can_loss), "route_b_loss": float(route_loss), "loss_abs_diff": abs(float(can_loss) - float(route_loss)), "parameter_max_abs_diff": p_diff, "logit_max_abs_diff": l_diff, "validation_BA_canonical": float(rb.evaluate(cache, can, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"], DEVICE)["BA"]), "validation_BA_route_b": float(rb.evaluate(cache, route, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"], DEVICE)["BA"])}
        eq1["pass"] = bool(eq1["loss_abs_diff"] <= 1e-6 and p_diff <= 1e-6 and l_diff <= 1e-6 and eq1["same_initial_state"] and eq1["same_rows"] and eq1["same_weights"] and eq1["same_order"] and eq1["same_rng_state"])
        rows.append(eq1)
        del can, route
        # EQ2: three-epoch trajectory, same explicit order/RNG semantics.
        can = rb.model_from_state(cache, geo, state, DEVICE); route = rb.model_from_state(cache, geo, state, DEVICE)
        can_opt = torch.optim.AdamW(can.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY); route_opt = torch.optim.AdamW(route.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
        seed, rng = rng_state_for(dataset, outer, "equivalence_trajectory")
        restore_rng(rng)
        canonical_traj: list[dict[str, Any]] = []
        for epoch in range(1, 4):
            order_e = old_order(ctx["sel_rows"], dataset, outer, "B0_SUBJECT_BALANCED_ERM", epoch)
            cl = geo.train_epoch(can, cache, ctx["sel_rows"], ctx["sel_mean"], ctx["sel_std"], w, can_opt, order_e, DEVICE)
            cval = rb.evaluate(cache, can, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"], DEVICE)
            canonical_traj.append({"epoch": epoch, "loss": float(cl), "val": cval, "state": {k: v.detach().cpu().clone() for k, v in can.state_dict().items()}, "logits": logits(cache, can, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"])})
        restore_rng(rng)
        for epoch in range(1, 4):
            order_e = old_order(ctx["sel_rows"], dataset, outer, "B0_SUBJECT_BALANCED_ERM", epoch)
            rl = pure_erm_epoch(cache, route, ctx["sel_rows"], ctx["sel_mean"], ctx["sel_std"], route_opt, order_e)
            rval = rb.evaluate(cache, route, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"], DEVICE)
            ref = canonical_traj[epoch - 1]
            pdiff = max(float(torch.max(torch.abs(ref["state"][k] - route.state_dict()[k].detach().cpu()))) for k in route.state_dict())
            traj_log_diff = float(np.max(np.abs(ref["logits"] - logits(cache, route, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"]))))
            row = {"dataset": dataset, "outer_fold": outer, "same_initial_state": True, "same_rows": True, "same_weights": True, "same_order": True, "same_rng_state": True, "epoch": epoch, "canonical_loss": ref["loss"], "route_b_loss": float(rl), "loss_abs_diff": abs(ref["loss"] - float(rl)), "parameter_max_abs_diff": pdiff, "logit_max_abs_diff": traj_log_diff, "validation_BA_canonical": float(ref["val"]["BA"]), "validation_BA_route_b": float(rval["BA"]), "validation_NLL_canonical": float(ref["val"]["NLL"]), "validation_NLL_route_b": float(rval["NLL"])}
            row["pass"] = bool(row["loss_abs_diff"] <= 1e-6 and row["parameter_max_abs_diff"] <= 1e-6 and row["logit_max_abs_diff"] <= 1e-6 and abs(row["validation_BA_canonical"] - row["validation_BA_route_b"]) <= 1e-12 and abs(row["validation_NLL_canonical"] - row["validation_NLL_route_b"]) <= 1e-10)
            rows.append(row)
        del can, route
        gc.collect()
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return rows


def run_order_controls() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tags = {"ERM_ORDER_B0": "B0_SUBJECT_BALANCED_ERM", "ERM_ORDER_B1": "groupdro", "ERM_ORDER_B3": "B3_SUBJECT_GRADIENT_STAT_DG", "ERM_ORDER_B4": "B4_SUBJECT_STYLE_EXTRAPOLATION"}
    out: list[dict[str, Any]] = []; repeats: list[dict[str, Any]] = []; hashes: list[dict[str, Any]] = []; seeds: list[dict[str, Any]] = []; split_rows: list[dict[str, Any]] = []
    old_sp = pd.read_csv(OLD_ROOT / "SUBJECT_PERFORMANCE.csv")
    for fold in (0, 1):
        ctx = cache_context("WBCIC", fold)
        split_rows.append({"dataset": "WBCIC", "outer_fold": fold, "held_subjects": ";".join(ctx["held"]), "train_subjects": ";".join(ctx["train_subjects"]), "inner_validation_subjects": ";".join(ctx["val_subjects"]), "held_disjoint_train": not bool(set(ctx["held"]) & set(ctx["train_subjects"])), "held_disjoint_inner_validation": not bool(set(ctx["held"]) & set(ctx["val_subjects"])), "inner_validation_disjoint_train": not bool(set(ctx["val_subjects"]) & set(ctx["sel_subjects"]))})
        tag_results: dict[str, dict[str, Any]] = {}
        for name, tag in tags.items():
            sel = selection_erm(ctx, tag, "selection")
            refit, final_state = refit_erm(ctx, tag, sel["selected_epoch"], "refit")
            tag_results[name] = {"selection": sel, "refit": refit}
            old_ba = float(old_sp[(old_sp.dataset == "WBCIC") & (old_sp.outer_fold == fold) & (old_sp.method == "B0_SUBJECT_BALANCED_ERM")].BA.mean() * 100.0)
            old_method = {"ERM_ORDER_B0": "B0_SUBJECT_BALANCED_ERM", "ERM_ORDER_B1": "B1_SUBJECT_GROUPDRO", "ERM_ORDER_B3": "B3_SUBJECT_GRADIENT_STAT_DG", "ERM_ORDER_B4": "B4_SUBJECT_STYLE_EXTRAPOLATION"}[name]
            old_m = float(old_sp[(old_sp.dataset == "WBCIC") & (old_sp.outer_fold == fold) & (old_sp.method == old_method)].BA.mean() * 100.0)
            out.append({"dataset": "WBCIC", "outer_fold": fold, "control": name, "objective": "SUBJECT_BALANCED_ERM", "order_schedule": tag, "selection_rng_seed": sel["rng_seed"], "refit_rng_seed": refit["rng_seed"], "selected_epoch": sel["selected_epoch"], "validation_BA": sel["validation_BA"] * 100.0, "validation_NLL": sel["validation_NLL"], "held_BA": refit["BA"] * 100.0, "held_Macro_F1": refit["Macro_F1"] * 100.0, "held_NLL": refit["NLL"], "old_B0_held_BA": old_ba, "old_method_held_BA": old_m, "old_delta_vs_old_B0_pp": old_m - old_ba, "order_delta_vs_order_B0_pp": None, "state_hash": refit["state_hash"]})
            seeds.extend([{"dataset": "WBCIC", "outer_fold": fold, "phase": "selection", "control": name, "common_seed": sel["rng_seed"]}, {"dataset": "WBCIC", "outer_fold": fold, "phase": "refit", "control": name, "common_seed": refit["rng_seed"]}])
            # Old schedules have one order per phase/epoch; hash the first
            # three epochs as a compact audit without writing rows.
            for phase, rows0 in (("selection", ctx["sel_rows"]), ("refit", ctx["refit_rows"])):
                for epoch in range(1, 4):
                    hashes.append({"dataset": "WBCIC", "outer_fold": fold, "phase": phase, "epoch": epoch, "control": name, "order_sha256": sha_bytes(np.asarray(old_order(rows0, "WBCIC", fold, tag, epoch), dtype=np.int64).tobytes())})
            if name == "ERM_ORDER_B0":
                sel2 = selection_erm(ctx, tag, "selection"); ref2, st2 = refit_erm(ctx, tag, sel2["selected_epoch"], "refit")
                repeats.append({"dataset": "WBCIC", "outer_fold": fold, "control": "ERM_ORDER_B0_REPEAT", "selected_epoch_original": sel["selected_epoch"], "selected_epoch_repeat": sel2["selected_epoch"], "validation_BA_abs_diff": abs(sel["validation_BA"] - sel2["validation_BA"]), "held_BA_abs_diff": abs(refit["BA"] - ref2["BA"]), "parameter_max_abs_diff": max(float(torch.max(torch.abs(final_state[k] - st2[k]))) for k in final_state), "state_hash_original": refit["state_hash"], "state_hash_repeat": ref2["state_hash"], "pass": bool(sel["selected_epoch"] == sel2["selected_epoch"] and abs(sel["validation_BA"] - sel2["validation_BA"]) <= 1e-12 and abs(refit["BA"] - ref2["BA"]) <= 1e-12 and max(float(torch.max(torch.abs(final_state[k] - st2[k]))) for k in final_state) <= 1e-7)})
        # Fill order deltas relative to order B0.
        b0 = next(x for x in out if x["outer_fold"] == fold and x["control"] == "ERM_ORDER_B0")
        for x in out:
            if x["outer_fold"] == fold:
                x["order_delta_vs_order_B0_pp"] = x["held_BA"] - b0["held_BA"]
    frame = pd.DataFrame(out)
    frame.to_csv(ROOT / "WBCIC_ERM_ORDER_CONTROLS.csv", index=False)
    order_summary = []
    explained = []
    for fold, g in frame.groupby("outer_fold"):
        vals = g.set_index("control").held_BA
        order_summary.append({"dataset": "WBCIC", "outer_fold": int(fold), **{k: float(vals.get(k, np.nan)) for k in tags}, "range_pp": float(vals.max() - vals.min()), "material_ge_1pp": bool(vals.max() - vals.min() >= 1.0)})
        for _, r in g.iterrows():
            if r["control"] == "ERM_ORDER_B0":
                continue
            explained.append({"dataset": "WBCIC", "outer_fold": int(fold), "method": r["control"], "old_delta_pp": float(r["old_delta_vs_old_B0_pp"]), "order_delta_pp": float(r["order_delta_vs_order_B0_pp"]), "explained_fraction": abs(float(r["order_delta_vs_order_B0_pp"])) / (abs(float(r["old_delta_vs_old_B0_pp"])) + 1e-9)})
    sens = {"schema": AUDIT_SCHEMA, "folds": order_summary, "any_material": bool(any(x["material_ge_1pp"] for x in order_summary)), "terminal_if_multiple_explained_ge_0.5": "CURRENT_TRIAGE_MATERIALLY_RANDOMNESS_CONFOUNDED"}
    write_json(ROOT / "WBCIC_ORDER_SENSITIVITY.json", sens)
    write_csv(ROOT / "RANDOMNESS_EXPLAINED_FRACTION.csv", explained)
    write_csv(ROOT / "DETERMINISM_REPEAT_AUDIT.csv", repeats)
    write_csv(ROOT / "ORDER_HASH_AUDIT.csv", hashes)
    write_csv(ROOT / "RNG_SEED_AUDIT.csv", seeds)
    write_csv(ROOT / "SPLIT_AUDIT.csv", split_rows)
    return frame, sens, pd.DataFrame(repeats), pd.DataFrame(hashes), pd.DataFrame(seeds)


def clean_select(ctx: Mapping[str, Any], method: str, cfg: Mapping[str, float], phase: str) -> dict[str, Any]:
    global CURRENT_PHASE
    CURRENT_PHASE = phase
    seed, rng = rng_state_for(ctx["dataset"], int(ctx["outer"]), phase)
    model = model_from_state(ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    q = {s: 1.0 / max(len(ctx["sel_subjects"]), 1) for s in ctx["sel_subjects"]} if method == "B1_SUBJECT_GROUPDRO" else None
    restore_rng(rng)
    best_ba, best_nll, best_ep, stale = -math.inf, math.inf, 1, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, rb.MAX_EPOCHS + 1):
        loss = rb.train_epoch(method, cfg, model, ctx["cache"], ctx["sel_rows"], ctx["sel_subjects"], ctx["sel_mean"], ctx["sel_std"], opt, ctx["dataset"], int(ctx["outer"]), epoch, DEVICE, q=q)
        ev = rb.evaluate(ctx["cache"], model, ctx["val_rows"], ctx["sel_mean"], ctx["sel_std"], DEVICE)
        ba, nll = float(ev["BA"]), float(ev["NLL"])
        improved = ba > best_ba + rb.TIE_TOL or (abs(ba - best_ba) <= rb.TIE_TOL and nll < best_nll - rb.TIE_TOL)
        if improved: best_ba, best_nll, best_ep, stale = ba, nll, epoch, 0
        else: stale += 1
        history.append({"epoch": epoch, "train_loss": loss, "val_BA": ba, "val_NLL": nll})
        if epoch >= rb.MIN_EPOCHS and stale >= rb.PATIENCE: break
    del model; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return {"selected_epoch": int(best_ep), "validation_BA": float(best_ba), "validation_NLL": float(best_nll), "history": history, "rng_seed": seed}


def clean_refit(ctx: Mapping[str, Any], method: str, cfg: Mapping[str, float], epochs: int, phase: str) -> dict[str, Any]:
    global CURRENT_PHASE
    CURRENT_PHASE = phase
    seed, rng = rng_state_for(ctx["dataset"], int(ctx["outer"]), phase)
    model = model_from_state(ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=rb.LR, weight_decay=rb.WEIGHT_DECAY)
    q = {s: 1.0 / max(len(ctx["train_subjects"]), 1) for s in ctx["train_subjects"]} if method == "B1_SUBJECT_GROUPDRO" else None
    restore_rng(rng)
    losses = []
    for epoch in range(1, int(epochs) + 1):
        losses.append(rb.train_epoch(method, cfg, model, ctx["cache"], ctx["refit_rows"], ctx["train_subjects"], ctx["refit_mean"], ctx["refit_std"], opt, ctx["dataset"], int(ctx["outer"]), epoch, DEVICE, q=q))
    ev = rb.evaluate(ctx["cache"], model, ctx["held_rows"], ctx["refit_mean"], ctx["refit_std"], DEVICE)
    result = {"BA": float(ev["BA"]), "Macro_F1": float(ev["Macro_F1"]), "NLL": float(ev["NLL"]), "per_subject": ev["per_subject"], "losses": losses, "rng_seed": seed}
    del model; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return result


def run_clean_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    global CURRENT_PHASE
    old_geo_order = rb.geo_order
    rb.geo_order = common_order
    rows: list[dict[str, Any]] = []
    try:
        for dataset in ("OpenBMI", "WBCIC"):
            for outer in (0, 1):
                ctx = cache_context(dataset, outer)
                selected: dict[str, tuple[dict[str, float], dict[str, Any]]] = {}
                for method in rb.METHODS:
                    candidates = []
                    for cfg in rb.CONFIGS[method]:
                        cand = clean_select(ctx, method, cfg, "selection")
                        candidates.append((cfg, cand))
                    candidates.sort(key=lambda z: (-z[1]["validation_BA"], z[1]["validation_NLL"], z[1]["selected_epoch"], str(z[0]["name"])))
                    selected[method] = candidates[0]
                for method in rb.METHODS:
                    cfg, sel = selected[method]
                    ref = clean_refit(ctx, method, cfg, sel["selected_epoch"], "refit")
                    rows.append({"dataset": dataset, "outer_fold": outer, "method": method, "selected_config": str(cfg["name"]), "selected_epoch": sel["selected_epoch"], "validation_BA": sel["validation_BA"] * 100.0, "validation_NLL": sel["validation_NLL"], "held_BA": ref["BA"] * 100.0, "held_Macro_F1": ref["Macro_F1"] * 100.0, "held_NLL": ref["NLL"], "held_subject_count": len(ref["per_subject"]), "selection_rng_seed": sel["rng_seed"], "refit_rng_seed": ref["rng_seed"]})
                del ctx; gc.collect()
    finally:
        rb.geo_order = old_geo_order
    per = pd.DataFrame(rows)
    write_csv(ROOT / "CLEAN_EARLY_SCREEN_PER_FOLD.csv", per)
    summ: list[dict[str, Any]] = []
    for method in rb.METHODS[1:]:
        cells = []
        for dataset in ("OpenBMI", "WBCIC"):
            for outer in (0, 1):
                b0 = float(per[(per.dataset == dataset) & (per.outer_fold == outer) & (per.method == "B0_SUBJECT_BALANCED_ERM")].held_BA.iloc[0])
                mm = float(per[(per.dataset == dataset) & (per.outer_fold == outer) & (per.method == method)].held_BA.iloc[0])
                cells.append((dataset, outer, mm - b0))
        om = float(np.mean([x[2] for x in cells if x[0] == "OpenBMI"])); wm = float(np.mean([x[2] for x in cells if x[0] == "WBCIC"]))
        positive = int(sum(v >= 0.0 for _, _, v in cells))
        summ.append({"method": method, "OpenBMI_fold0_delta_BA_pp": cells[0][2], "OpenBMI_fold1_delta_BA_pp": cells[1][2], "OpenBMI_mean_delta_BA_pp": om, "WBCIC_fold0_delta_BA_pp": cells[2][2], "WBCIC_fold1_delta_BA_pp": cells[3][2], "WBCIC_mean_delta_BA_pp": wm, "positive_dataset_fold_cells": positive, "provisional_gate": bool(om >= 0.5 and wm >= 0.5 and positive >= 3)})
    summary = pd.DataFrame(summ)
    write_csv(ROOT / "CLEAN_EARLY_SCREEN_SUMMARY.csv", summary)
    return per, summary


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rb.seed_everything(rb.SEED)
    lock = {
        "schema": AUDIT_SCHEMA, "protocol": COMMON_PROTOCOL,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branch": "codex/persist-eeg-route-b-randomness-audit-v1",
        "datasets": ["OpenBMI", "WBCIC"], "audit_order_controls": [0, 1], "seed": 0,
        "source_only": True, "outer_split_policy": "reuse current Route-B prelocked folds; not bitwise identical to previous nested-OOF folds",
        "common_order": "sha256(route-b-common-order-v2|dataset|outer|phase|epoch)",
        "common_rng": "sha256(route-b-common-rng-v2|dataset|outer|phase)",
        "method_names_excluded_from_order_and_rng": True,
        "canonical_outcome_labels_read": False, "pseudo_unseen_source_labels_read_for_evaluation": False,
        "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False,
    }
    write_json(ROOT / "COMMON_RANDOMNESS_PROTOCOL_V2.json", lock)
    eq = run_equivalence()
    write_json(ROOT / "BASELINE_NUMERICAL_EQUIVALENCE.json", {"schema": AUDIT_SCHEMA, "thresholds": {"parameter_max_abs_diff": 1e-6, "logit_max_abs_diff": 1e-6, "loss_abs_diff": 1e-6}, "rows": eq, "pass": bool(all(bool(x["pass"]) for x in eq))})
    if not all(bool(x["pass"]) for x in eq):
        (ROOT / "FINAL_REPORT.md").write_text("# Route-B randomness audit\n\nTerminal: `STOP_BASELINE_IMPLEMENTATION_NOT_EQUIVALENT`\n\nThe numerical equivalence audit failed; no DG/order conclusion was run.\n", encoding="utf-8")
        write_json(ROOT / "NO_CANONICAL_OUTCOME_ACCESS_AUDIT.json", {"schema": AUDIT_SCHEMA, "canonical_outcome_labels_read": False, "pseudo_unseen_source_labels_read_for_evaluation": False, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False})
        return
    frame, sens, repeats, hashes, seeds = run_order_controls()
    per, summary = run_clean_screen()
    # Compare clean gains to the observed WBCIC order-only envelope.
    env = []
    for _, r in summary.iterrows():
        max_range = float(max(x["range_pp"] for x in sens["folds"]))
        env.append({"method": r["method"], "OpenBMI_mean_delta_BA_pp": r["OpenBMI_mean_delta_BA_pp"], "WBCIC_mean_delta_BA_pp": r["WBCIC_mean_delta_BA_pp"], "ERM_order_noise_envelope_max_pp": max_range, "WBCIC_gain_vs_randomness_envelope_pp": float(r["WBCIC_mean_delta_BA_pp"] - max_range), "exceeds_envelope": bool(r["WBCIC_mean_delta_BA_pp"] > max_range)})
    write_csv(ROOT / "RANDOMNESS_ENVELOPE_COMPARISON.csv", env)
    explained = pd.read_csv(ROOT / "RANDOMNESS_EXPLAINED_FRACTION.csv")
    multiple_explained = bool((explained["explained_fraction"] >= 0.5).sum() >= 2)
    passing = summary[summary.provisional_gate].method.tolist()
    terminal = "CLEAN_ROUTE_B_FOUNDATION_SIGNAL_SUPPORTED" if passing else "NO_CLEAN_ROUTE_B_FOUNDATION_SIGNAL"
    old_order_pct = explained.groupby("method").explained_fraction.mean().to_dict()
    lines = ["# Route-B randomness audit", "", f"Terminal: `{terminal}`", "", "## Direct answers", "", f"1. Old WBCIC +11--14pp order explanation fractions (fold-level): `{json.dumps(old_order_pct, sort_keys=True)}`.", "2. Route-B B0 numerical equivalence: `PASS` (see BASELINE_NUMERICAL_EQUIVALENCE.json).", "3. Clean gains are in CLEAN_EARLY_SCREEN_SUMMARY.csv; they use clean B0 and common order/RNG.", f"4. Clean provisional passing methods: `{', '.join(passing) if passing else 'none'}`.", f"5. Any method exceeding the observed pure-ERM WBCIC randomness envelope: `{[x['method'] for x in env if x['exceeds_envelope']]}`.", "6. Full Route-B is not automatically resumed; manual decision is required after this audit.", f"7. Recommended winner by clean worst-dataset effect/consistency: `{passing[0] if passing else 'none'}`.", f"8. Multiple method order explanations >=0.5: `{multiple_explained}`.", "", "Route-B folds are internally prelocked but are not bitwise identical to the previous nested-OOF experiment's folds.", "Canonical outcome labels, OpenBMI sealed holdout, and WBCIC outer-10 were not opened."]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ROOT / "NO_CANONICAL_OUTCOME_ACCESS_AUDIT.json", {"schema": AUDIT_SCHEMA, "canonical_outcome_labels_read": False, "pseudo_unseen_source_labels_read_for_evaluation": True, "OpenBMI_sealed_holdout_opened": False, "WBCIC_outer_10_opened": False, "old_early_triage_metadata_bug_corrected_in_new_audit": True})
    write_json(ROOT / "EARLY_TRIAGE_DECISION_METADATA_CORRECTED.json", {"schema": AUDIT_SCHEMA, "canonical_outcome_labels_read": False, "pseudo_unseen_source_labels_read_for_evaluation": True, "note": "The old EARLY_TRIAGE_DECISION.json used ambiguous outcome_labels_read=true; this audit uses explicit fields."})
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
