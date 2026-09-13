"""Run the fixed LiteBN-MS4 WBCIC-MI seed-0 five-fold experiment."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path(os.environ.get("MS4_REPO", "/root/rivermind-data/CRCICLR_SC_CONTROLLED_V2_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_ms4_linux_seed0_v1"
PARENT = REPO / "experiments" / "persist_eeg_litebn_sc_controlled_linux_seed0_v2"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("MS4_RUNTIME", "/root/rivermind-data/litebn_ms4_seed0_runtime")).resolve()
sys.path.insert(0, str(PARENT / "code"))
sys.path.insert(0, str(EXP / "code"))

import run_c1_wbcic as shared
from litebn_ms4 import controlled_multi_forward


base = shared.base
TASK, METHOD, SEED, K = "WBCIC_MI", "MS4_LiteBN_FourSampleCE", 0, 4
SECONDARY_SEED = shared.SECONDARY_SEED
GRADIENT_DIAGNOSTIC_EPOCHS = {1, 10, 20, 30, 40, 50, 60}
B0, C0 = "B0_Exact_LiteBN", "C0_LiteBN_DualCE"


def tensor_content_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    paths = [
        Path(__file__), EXP / "code" / "litebn_ms4.py", PARENT / "code" / "run_c0_wbcic.py",
        PARENT / "code" / "run_c1_wbcic.py", PARENT / "code" / "litebn_sc_controlled.py",
        shared.ORIGINAL_EXP / "code" / "litebn_x.py",
    ]
    return base.sha256_bytes(b"".join(path.read_bytes() for path in paths))


def checkpoint_path(fold: int, name: str) -> Path:
    return RUNTIME / "checkpoints" / "wbcic_mi" / f"fold{fold}_ms4" / name


def state_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> tuple[bool, float]:
    return shared.state_diff(left, right)


def torch_rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return shared.torch_rng_equal(left, right)


def _torch_only(state: dict[str, Any]) -> dict[str, Any]:
    return {"cpu": state["torch"], "cuda": state.get("cuda", [])}


def ms4_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    value: torch.Tensor, labels: torch.Tensor, secondary: dict[str, Any], amp: bool, k: int = K,
) -> tuple[float, dict[str, Any]]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=value.device.type, dtype=torch.float16, enabled=amp):
        logits, _, secondary, _ = controlled_multi_forward(model, value, secondary, k=k)
        loss = torch.stack([F.cross_entropy(current, labels) for current in logits]).mean()
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
    scaler.step(optimizer); scaler.update()
    return float(loss.detach()), secondary


def prefix_audit(model: nn.Module, value: torch.Tensor, fold: int, amp: bool) -> dict[str, Any]:
    left, right = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    base.set_seed(510_000 + fold); start = base.rng_state(); secondary = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        la, ea, lb, eb, secondary_c0 = shared.controlled_dual_forward(left, value, copy.deepcopy(secondary))
    c0_rng, c0_bn = base.rng_state(), shared.bn_snapshot(left)
    base.restore_rng(start)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits, embeddings, _, trace = controlled_multi_forward(right, value, copy.deepcopy(secondary), k=K)
    ms4_rng, ms4_bn = base.rng_state(), shared.bn_snapshot(right)
    bn_exact, bn_max = state_diff(c0_bn, ms4_bn)
    passed = all((
        torch.equal(la, logits[0]), torch.equal(lb, logits[1]), torch.equal(ea, embeddings[0]), torch.equal(eb, embeddings[1]),
        torch_rng_equal(secondary_c0, trace[0]), torch_rng_equal(_torch_only(c0_rng), _torch_only(ms4_rng)), bn_exact, bn_max == 0.0,
    ))
    return {
        "task": TASK, "fold": fold, "precision": "AMP_FP16" if amp else "FP32",
        "A_logits_exact": torch.equal(la, logits[0]), "B_logits_exact": torch.equal(lb, logits[1]),
        "A_embedding_exact": torch.equal(ea, embeddings[0]), "B_embedding_exact": torch.equal(eb, embeddings[1]),
        "secondary_after_B_exact": torch_rng_equal(secondary_c0, trace[0]),
        "main_rng_exact": torch_rng_equal(_torch_only(c0_rng), _torch_only(ms4_rng)),
        "live_bn_exact": bn_exact, "max_abs_live_bn_diff": bn_max, "status": "PASS" if passed else "FAIL",
    }


def bn_autograd_audit(model: nn.Module, value: torch.Tensor, labels: torch.Tensor, fold: int, amp: bool) -> dict[str, Any]:
    baseline, candidate = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    base.set_seed(520_000 + fold); start = base.rng_state(); secondary = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        baseline(value)
    baseline_bn, baseline_rng = shared.bn_snapshot(baseline), base.rng_state()
    base.restore_rng(start)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits, _, secondary_after, _ = controlled_multi_forward(candidate, value, secondary, k=K)
        losses = [F.cross_entropy(current, labels) for current in logits]
        total = torch.stack(losses).mean()
    post_forward_bn, post_forward_rng = shared.bn_snapshot(candidate), base.rng_state()
    parameters = [parameter for parameter in candidate.parameters() if parameter.requires_grad]
    extra_finite, extra_nonzero = [], []
    for index in range(1, K):
        gradients = torch.autograd.grad(losses[index], parameters, retain_graph=True, allow_unused=True)
        extra_finite.append(all(g is None or torch.isfinite(g).all() for g in gradients))
        extra_nonzero.append(any(g is not None and g.abs().sum() > 0 for g in gradients))
    total.backward()
    post_backward_bn = shared.bn_snapshot(candidate)
    bn_a_exact, bn_a_max = state_diff(baseline_bn, post_forward_bn)
    backward_exact, backward_max = state_diff(post_forward_bn, post_backward_bn)
    stochastic = any(not torch.equal(logits[0], current) for current in logits[1:])
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters)
    gradients_nonzero = any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in parameters)
    passed = all((
        bn_a_exact, bn_a_max == 0.0, backward_exact, backward_max == 0.0,
        torch_rng_equal(_torch_only(baseline_rng), _torch_only(post_forward_rng)),
        all(extra_finite), all(extra_nonzero), gradients_finite, gradients_nonzero, stochastic,
    ))
    return {
        "task": TASK, "fold": fold, "precision": "AMP_FP16" if amp else "FP32", "audit": "BN_AND_AUTOGRAD",
        "forward_A_live_bn_exact": bn_a_exact, "max_abs_forward_A_bn_diff": bn_a_max,
        "backward_preserves_bn_exact": backward_exact, "max_abs_backward_bn_diff": backward_max,
        "main_rng_matches_single_A": torch_rng_equal(_torch_only(baseline_rng), _torch_only(post_forward_rng)),
        "B_gradient_finite_nonzero": bool(extra_finite[0] and extra_nonzero[0]),
        "C_gradient_finite_nonzero": bool(extra_finite[1] and extra_nonzero[1]),
        "D_gradient_finite_nonzero": bool(extra_finite[2] and extra_nonzero[2]),
        "total_gradient_finite_nonzero": bool(gradients_finite and gradients_nonzero),
        "stochastic_logits_not_all_identical": stochastic, "uses_data_bypass": False,
        "status": "PASS" if passed else "FAIL",
    }


def k2_reduction_audit(model: nn.Module, value: torch.Tensor, labels: torch.Tensor, fold: int, amp: bool) -> dict[str, Any]:
    c0_model, k2_model = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    oc = torch.optim.AdamW(c0_model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    ok = torch.optim.AdamW(k2_model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    sc = torch.amp.GradScaler("cuda", enabled=amp); sk = torch.amp.GradScaler("cuda", enabled=amp)
    base.set_seed(530_000 + fold); start = base.rng_state(); secondary = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    loss_c0, sec_c0 = shared._one_step(c0_model, oc, sc, value, labels, copy.deepcopy(secondary), amp, 0.0)
    rng_c0 = base.rng_state(); grads_c0 = {name: parameter.grad.detach().clone() for name, parameter in c0_model.named_parameters() if parameter.grad is not None}
    base.restore_rng(start)
    loss_k2, sec_k2 = ms4_step(k2_model, ok, sk, value, labels, copy.deepcopy(secondary), amp, k=2)
    rng_k2 = base.rng_state(); grads_k2 = {name: parameter.grad.detach().clone() for name, parameter in k2_model.named_parameters() if parameter.grad is not None}
    state_exact, state_max = state_diff(c0_model.state_dict(), k2_model.state_dict())
    grad_exact, grad_max = state_diff(grads_c0, grads_k2)
    passed = all((
        loss_c0 == loss_k2, state_exact, state_max == 0.0, grad_exact, grad_max == 0.0,
        torch_rng_equal(sec_c0, sec_k2), torch_rng_equal(_torch_only(rng_c0), _torch_only(rng_k2)), sc.state_dict() == sk.state_dict(),
    ))
    return {
        "task": TASK, "fold": fold, "precision": "AMP_FP16" if amp else "FP32",
        "loss_exact": loss_c0 == loss_k2, "max_abs_loss_diff": abs(loss_c0 - loss_k2),
        "gradients_exact": grad_exact, "max_abs_gradient_diff": grad_max,
        "model_state_exact": state_exact, "max_abs_model_state_diff": state_max,
        "main_rng_exact": torch_rng_equal(_torch_only(rng_c0), _torch_only(rng_k2)),
        "secondary_after_B_exact": torch_rng_equal(sec_c0, sec_k2), "scaler_exact": sc.state_dict() == sk.state_dict(),
        "status": "PASS" if passed else "FAIL",
    }


def resume_audit(model: nn.Module, value: torch.Tensor, labels: torch.Tensor, fold: int) -> dict[str, Any]:
    continuous, interrupted = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    oc = torch.optim.AdamW(continuous.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    oi = torch.optim.AdamW(interrupted.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    sc = torch.amp.GradScaler("cuda", enabled=True); si = torch.amp.GradScaler("cuda", enabled=True)
    base.set_seed(540_000 + fold); start = base.rng_state(); sec_start = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    sec_c = copy.deepcopy(sec_start)
    for _ in range(3):
        _, sec_c = ms4_step(continuous, oc, sc, value, labels, sec_c, True)
    rng_c = base.rng_state()
    base.restore_rng(start); sec_i = copy.deepcopy(sec_start)
    _, sec_i = ms4_step(interrupted, oi, si, value, labels, sec_i, True)
    payload = {"model": copy.deepcopy(interrupted.state_dict()), "optimizer": copy.deepcopy(oi.state_dict()), "scaler": copy.deepcopy(si.state_dict()), "main_rng": base.rng_state(), "secondary_rng": copy.deepcopy(sec_i)}
    resumed = copy.deepcopy(model).train(); optimizer = torch.optim.AdamW(resumed.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY); scaler = torch.amp.GradScaler("cuda", enabled=True)
    resumed.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"]); scaler.load_state_dict(payload["scaler"])
    base.restore_rng(payload["main_rng"]); sec_r = payload["secondary_rng"]
    for _ in range(2):
        _, sec_r = ms4_step(resumed, optimizer, scaler, value, labels, sec_r, True)
    rng_r = base.rng_state(); exact, maximum = state_diff(continuous.state_dict(), resumed.state_dict())
    passed = exact and maximum == 0.0 and torch_rng_equal(sec_c, sec_r) and torch_rng_equal(_torch_only(rng_c), _torch_only(rng_r)) and sc.state_dict() == scaler.state_dict()
    return {
        "task": TASK, "fold": fold, "precision": "AMP_FP16", "audit": "THREE_STEP_RESUME",
        "model_state_exact": exact, "max_abs_model_state_diff": maximum,
        "main_rng_exact": torch_rng_equal(_torch_only(rng_c), _torch_only(rng_r)),
        "secondary_rng_exact": torch_rng_equal(sec_c, sec_r), "scaler_exact": sc.state_dict() == scaler.state_dict(),
        "status": "PASS" if passed else "FAIL",
    }


def gradient_geometry(losses: list[torch.Tensor], parameters: list[nn.Parameter]) -> dict[str, float]:
    objectives = [losses[0], torch.stack(losses[:2]).mean(), torch.stack(losses).mean()]
    grads = [torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True) for loss in objectives]
    def dot(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> torch.Tensor:
        return sum((a.float().mul(b.float()).sum() for a, b in zip(left, right) if a is not None and b is not None), torch.zeros((), device=losses[0].device))
    norms = [dot(row, row).sqrt() for row in grads]
    cos12 = dot(grads[0], grads[1]) / (norms[0] * norms[1]).clamp_min(1e-20)
    cos24 = dot(grads[1], grads[2]) / (norms[1] * norms[2]).clamp_min(1e-20)
    diff12 = sum((((b.float() - a.float()).square().sum()) for a, b in zip(grads[0], grads[1]) if a is not None and b is not None), torch.zeros((), device=losses[0].device)).sqrt()
    diff24 = sum((((b.float() - a.float()).square().sum()) for a, b in zip(grads[1], grads[2]) if a is not None and b is not None), torch.zeros((), device=losses[0].device)).sqrt()
    return {"g1_norm": float(norms[0]), "g2_norm": float(norms[1]), "g4_norm": float(norms[2]), "cosine_g1_g2": float(cos12), "cosine_g2_g4": float(cos24), "norm_g2_minus_g1": float(diff12), "norm_g4_minus_g2": float(diff24)}


def train_one(model: nn.Module, fold: dict[str, Any], bundle: base.SignalBundle, cache: base.RawGPUCache, mean: np.ndarray, std: np.ndarray, norm_meta: dict[str, Any], batch_info: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_id = int(fold["fold_id"]); latest, selected = checkpoint_path(fold_id, "checkpoint_latest.pt"), checkpoint_path(fold_id, "selected_best.pt")
    initial_hash = tensor_content_hash(model.state_dict())
    invariants = {"task": TASK, "fold": fold_id, "method": METHOD, "seed": SEED, "K": K, "secondary_seed": SECONDARY_SEED, "initial_tensor_content_sha256": initial_hash, "source_sha256": source_hash(), "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"], "persistent_bn_updates_per_batch": 1}
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY); scaler = torch.amp.GradScaler("cuda", enabled=True)
    secondary = shared.initialize_secondary_rng(SECONDARY_SEED); start, history, grad_rows, best, best_epoch, best_state, global_step = 1, [], [], -float("inf"), None, None, 0
    elapsed_before, peak_alloc_before, peak_reserved_before = 0.0, 0, 0
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants:
            raise RuntimeError(f"MS4 resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        base.restore_rng(saved["main_rng"]); secondary = saved["secondary_rng"]
        start, history, grad_rows = int(saved["epoch"]) + 1, list(saved["history"]), list(saved["gradient_diagnostics"])
        best, best_epoch, best_state, global_step = float(saved["best"]), saved["best_epoch"], saved["best_state"], int(saved["global_step"])
        elapsed_before = float(saved.get("elapsed_seconds_completed", 0.0)); peak_alloc_before = int(saved.get("peak_cuda_allocated_bytes", 0)); peak_reserved_before = int(saved.get("peak_cuda_reserved_bytes", 0))
    attempted_total = sum(int(row["attempted_optimizer_steps"]) for row in history); successful_total = sum(int(row["successful_optimizer_steps"]) for row in history)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, base.EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); values = {key: [] for key in ("cea", "ceb", "cec", "ced", "mean", "std", "disagree", "l1", "skl", "grad", "scale")}; skipped = 0
        for batch_index, indices in enumerate(batch_info["episodes"][epoch - 1]):
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _, secondary, _ = controlled_multi_forward(model, value, secondary, k=K)
                losses = [F.cross_entropy(current, labels) for current in logits]; loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite MS4 loss fold={fold_id} epoch={epoch}")
            if epoch in GRADIENT_DIAGNOSTIC_EPOCHS and batch_index == 0:
                grad_rows.append({"task": TASK, "method": METHOD, "fold": fold_id, "epoch": epoch, "global_step": global_step, **gradient_geometry(losses, parameters)})
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); grad_norm = torch.nn.utils.clip_grad_norm_(parameters, base.CLIP)
            scale_before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update(); skipped += int(float(scaler.get_scale()) < scale_before); global_step += 1
            with torch.no_grad():
                probabilities = [F.softmax(current.float(), dim=-1) for current in logits]
                pair_indices = list(combinations(range(K), 2)); pair_l1, pair_kl, pair_disagree = [], [], []
                for left, right in pair_indices:
                    pair_l1.append(float((probabilities[left] - probabilities[right]).abs().sum(-1).mean()))
                    pair_kl.append(float(shared.symmetric_kl(logits[left], logits[right])[0]))
                    pair_disagree.append(float((logits[left].argmax(-1) != logits[right].argmax(-1)).float().mean()))
                ce_values = torch.stack([row.detach().float() for row in losses])
                for key, current in zip(("cea", "ceb", "cec", "ced"), losses): values[key].append(float(current))
                values["mean"].append(float(loss)); values["std"].append(float(ce_values.std(unbiased=False)))
                values["disagree"].append(float(np.mean(pair_disagree))); values["l1"].append(float(np.mean(pair_l1))); values["skl"].append(float(np.mean(pair_kl))); values["grad"].append(float(grad_norm)); values["scale"].append(scale_before)
        validation = base.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        val_ba = float(np.mean([row["BA"] for row in validation.values()])); val_f1 = float(np.mean([row["macro_F1"] for row in validation.values()]))
        chose = epoch >= base.MIN_EPOCH and val_ba > best + base.TIE_TOL
        if chose: best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        row = {"task": TASK, "method": METHOD, "fold": fold_id, "epoch": epoch, "mean_CE_A": np.mean(values["cea"]), "mean_CE_B": np.mean(values["ceb"]), "mean_CE_C": np.mean(values["cec"]), "mean_CE_D": np.mean(values["ced"]), "mean_CE": np.mean(values["mean"]), "mean_within_batch_CE_std": np.mean(values["std"]), "mean_pairwise_prediction_disagreement": np.mean(values["disagree"]), "mean_pairwise_probability_L1": np.mean(values["l1"]), "mean_pairwise_symmetric_KL_diagnostic_only": np.mean(values["skl"]), "mean_preclip_gradient_norm": np.mean(values["grad"]), "mean_gradscaler_scale": np.mean(values["scale"]), "attempted_optimizer_steps": len(values["mean"]), "successful_optimizer_steps": len(values["mean"]) - skipped, "amp_skips": skipped, "inner_val_subject_BA": val_ba, "inner_val_subject_macro_F1": val_f1, "selected": bool(chose), "epoch_runtime_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        base.atomic_torch_save(latest, {"epoch": epoch, "global_step": global_step, "history": history, "gradient_diagnostics": grad_rows, "best": best, "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "main_rng": base.rng_state(), "secondary_rng": secondary, "invariants": invariants, "elapsed_seconds_completed": elapsed_before + time.perf_counter() - started, "peak_cuda_allocated_bytes": max(peak_alloc_before, int(torch.cuda.max_memory_allocated(device))), "peak_cuda_reserved_bytes": max(peak_reserved_before, int(torch.cuda.max_memory_reserved(device)))})
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[MS4 {TASK} f{fold_id}] epoch={epoch:02d} CE={row['mean_CE']:.4f} valBA={val_ba:.4f} step={global_step}", flush=True)
    if best_state is None: raise RuntimeError(f"no eligible MS4 checkpoint fold={fold_id}")
    model.load_state_dict(best_state, strict=True); base.atomic_torch_save(selected, model.state_dict())
    record = {"task": TASK, "dataset": "WBCIC", "fold": fold_id, "method": METHOD, "seed": SEED, "K": K, "parameter_count": base.parameter_count(model), "initial_tensor_content_sha256": initial_hash, "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best), "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)), "checkpoint_path": str(selected), "checkpoint_sha256": base.sha256_file(selected), "checkpoint_tensor_content_sha256": tensor_content_hash(model.state_dict()), "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"], "epochs_completed": len(history), "global_step": global_step, "attempted_optimizer_steps": attempted_total + sum(int(row["attempted_optimizer_steps"]) for row in history[start-1:]), "successful_optimizer_steps": successful_total + sum(int(row["successful_optimizer_steps"]) for row in history[start-1:]), "elapsed_seconds_total": elapsed_before + time.perf_counter() - started, "peak_cuda_allocated_bytes": max(peak_alloc_before, int(torch.cuda.max_memory_allocated(device))), "peak_cuda_reserved_bytes": max(peak_reserved_before, int(torch.cuda.max_memory_reserved(device))), "source_sha256": source_hash()}
    return record, history, grad_rows


def load_model(record: dict[str, Any], fold: int, device: torch.device) -> nn.Module:
    path = Path(record["checkpoint_path"])
    if not path.is_file() or base.sha256_file(path) != record["checkpoint_sha256"]: raise RuntimeError(f"invalid MS4 checkpoint fold={fold}")
    model, _ = shared.initialize_c1(TASK, fold, SEED); state = torch.load(path, map_location=device, weights_only=False)
    if tensor_content_hash(state) != record["checkpoint_tensor_content_sha256"]: raise RuntimeError(f"MS4 tensor hash mismatch fold={fold}")
    model.load_state_dict(state, strict=True); return model.to(device).eval()


def comparison(frame: pd.DataFrame, cohort: str, right: str) -> dict[str, Any]:
    pivot = frame.pivot(index="subject_id", columns="method", values="BA"); delta = (pivot[METHOD] - pivot[right]) * 100; boot = shared.paired_bootstrap(delta.to_numpy())
    return {"task": TASK, "cohort": cohort, "comparison": f"MS4-{right.split('_')[0]}", "left_method": METHOD, "right_method": right, "left_BA": float(pivot[METHOD].mean()), "right_BA": float(pivot[right].mean()), "delta_pp": float(delta.mean()), "positive_subjects": boot["positive_subjects"], "negative_subjects": boot["negative_subjects"], "tied_subjects": boot["tied_subjects"], "bootstrap_ci_low_pp": boot["ci_low_pp"], "bootstrap_ci_high_pp": boot["ci_high_pp"], "resamples": 10_000}


def evaluate_and_finalize(folds: dict[int, dict[str, Any]], records: list[dict[str, Any]], b0_records: list[dict[str, Any]], c0_records: list[dict[str, Any]], device: torch.device) -> str:
    parent_outer = pd.read_csv(PARENT / "outputs" / "OUTER_SUBJECT_RESULTS.csv"); parent_outer = parent_outer[parent_outer.method.isin([B0, C0])].copy()
    ms4_outer = []
    for fold_id in range(5):
        fold = folds[fold_id]; bundle = shared.build_future_bundle(fold["outer_dev_subjects"]); cache = base.RawGPUCache(bundle, device); mean, std, _ = base.load_tensor_pair(shared.ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz"); model = load_model(records[fold_id], fold_id, device)
        for subject, metrics in base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std).items(): ms4_outer.append({"task": TASK, "fold": fold_id, "subject_id": subject, "method": METHOD, **metrics})
        del model, bundle, cache; gc.collect(); torch.cuda.empty_cache()
    outer = pd.concat([parent_outer, pd.DataFrame(ms4_outer)], ignore_index=True).sort_values(["fold", "subject_id", "method"]); base.write_csv(OUT / "WBCIC_MS4_OUTER_SUBJECT_RESULTS.csv", outer)
    fold_rows, selected = [], {B0: b0_records, C0: c0_records, METHOD: records}
    for (fold_id, method), q in outer.groupby(["fold", "method"]):
        record = selected[method][int(fold_id)]; fold_rows.append({"task": TASK, "fold": int(fold_id), "method": method, "subjects": len(q), "BA": q.BA.mean(), "macro_F1": q.macro_F1.mean(), "accuracy": q.accuracy.mean(), "selected_epoch": record["selected_epoch"], "selected_inner_val_BA": record["best_inner_val_BA"]})
    fold_frame = pd.DataFrame(fold_rows); pivot_fold = fold_frame.pivot(index="fold", columns="method", values="BA")
    fold_frame = fold_frame.merge(((pivot_fold[METHOD] - pivot_fold[B0]) * 100).rename("MS4_minus_B0_BA_pp"), left_on="fold", right_index=True).merge(((pivot_fold[METHOD] - pivot_fold[C0]) * 100).rename("MS4_minus_C0_BA_pp"), left_on="fold", right_index=True)
    base.write_csv(OUT / "WBCIC_MS4_OUTER_FOLD_RESULTS.csv", fold_frame.sort_values(["fold", "method"]))
    outer_comparisons = [comparison(outer, "OUTER", B0), comparison(outer, "OUTER", C0)]

    parent_heldout = pd.read_csv(PARENT / "outputs" / "HELDOUT_SUBJECT_RESULTS.csv"); parent_heldout = parent_heldout[parent_heldout.method.isin([B0, C0])].copy(); subjects = base.subject_sort(parent_heldout.subject_id.unique(), "WBCIC")
    replicates = []
    for fold_id in range(5):
        bundle = shared.build_future_bundle(subjects); cache = base.RawGPUCache(bundle, device); mean, std, _ = base.load_tensor_pair(shared.ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz"); model = load_model(records[fold_id], fold_id, device)
        for subject, metrics in base.evaluate(model, bundle, cache, subjects, mean, std).items(): replicates.append({"subject_id": subject, "method": METHOD, **metrics})
        del model, bundle, cache; gc.collect(); torch.cuda.empty_cache()
    ms4_heldout = pd.DataFrame(replicates).groupby(["subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean(); heldout = pd.concat([parent_heldout, ms4_heldout], ignore_index=True).sort_values(["subject_id", "method"])
    base.write_csv(OUT / "WBCIC_MS4_INTERNAL_HELDOUT_SUBJECT_RESULTS.csv", heldout)
    heldout_comparisons = [comparison(heldout, "INTERNAL_HELDOUT", B0), comparison(heldout, "INTERNAL_HELDOUT", C0)]
    comparisons = pd.DataFrame([*outer_comparisons, *heldout_comparisons]); base.write_csv(OUT / "WBCIC_MS4_PAIRED_COMPARISONS.csv", comparisons); base.write_csv(OUT / "WBCIC_MS4_BOOTSTRAP.csv", comparisons[["task", "cohort", "comparison", "positive_subjects", "negative_subjects", "tied_subjects", "bootstrap_ci_low_pp", "bootstrap_ci_high_pp", "resamples"]])
    heldout_summary = comparisons[comparisons.cohort == "INTERNAL_HELDOUT"].copy(); heldout_summary.insert(0, "data_status", "DEVELOPMENT_MODEL_SELECTION_DATA"); base.write_csv(OUT / "WBCIC_MS4_INTERNAL_HELDOUT_SUMMARY.csv", heldout_summary)
    lookup = {(row.cohort, row.comparison): row for _, row in comparisons.iterrows()}; o_b0, o_c0 = lookup[("OUTER", "MS4-B0")], lookup[("OUTER", "MS4-C0")]; h_b0, h_c0 = lookup[("INTERNAL_HELDOUT", "MS4-B0")], lookup[("INTERNAL_HELDOUT", "MS4-C0")]
    fold_b0, fold_c0 = (pivot_fold[METHOD] - pivot_fold[B0]) * 100, (pivot_fold[METHOD] - pivot_fold[C0]) * 100
    audit_files = ["MS4_INITIALIZATION_AUDIT.csv", "MS4_C0_PREFIX_EQUIVALENCE.csv", "MS4_K2_REDUCTION_AUDIT.csv", "MS4_BN_AUTOGRAD_AUDIT.csv", "MANIFEST_AUDIT.csv"]
    audits_pass = all((pd.read_csv(OUT / name).status == "PASS").all() for name in audit_files)
    conditions = {"MS4_outer_BA_ge_0_7940": float(o_b0.left_BA) >= 0.7940, "MS4_outer_gt_C0": float(o_c0.delta_pp) > 0.0, "MS4_outer_gt_B0": float(o_b0.delta_pp) > 0.0, "positive_folds_vs_B0_ge_3": int((fold_b0 > base.TIE_TOL).sum()) >= 3, "worst_fold_vs_B0_gt_minus_1_pp": float(fold_b0.min()) > -1.0, "MS4_heldout_BA_ge_0_8000": float(h_b0.left_BA) >= 0.8000, "MS4_heldout_ge_C0": float(h_c0.delta_pp) >= 0.0, "all_audits_pass": audits_pass}
    if not audits_pass: terminal = "MS4_PROTOCOL_FAIL"
    elif all(conditions.values()): terminal = "MS4_WBCIC_STRONG_SIGNAL"
    elif float(o_c0.delta_pp) > 0 and float(h_c0.delta_pp) < 0: terminal = "MS4_OUTER_HELDOUT_TRADEOFF"
    elif float(o_b0.delta_pp) > 0 and float(h_b0.delta_pp) > 0: terminal = "MS4_WBCIC_WEAK_POSITIVE_SIGNAL"
    else: terminal = "MS4_NO_USEFUL_SIGNAL"
    decision = {"status": terminal, "model": "LiteBN-MS4", "task": TASK, "seed": SEED, "K": K, "conditions": conditions, "outer": {"MS4-B0": o_b0.to_dict(), "MS4-C0": o_c0.to_dict()}, "internal_heldout": {"MS4-B0": h_b0.to_dict(), "MS4-C0": h_c0.to_dict()}, "positive_folds_vs_B0": int((fold_b0 > base.TIE_TOL).sum()), "positive_folds_vs_C0": int((fold_c0 > base.TIE_TOL).sum()), "worst_fold_delta_vs_B0_pp": float(fold_b0.min()), "internal_heldout_status": "DEVELOPMENT_MODEL_SELECTION_DATA", "new_sealed_test_accessed": False, "other_tasks_run": False, "seed1_2_run": False}
    base.write_json(OUT / "FINAL_MS4_DECISION.json", decision)
    c0_b0_outer = (float(o_c0.right_BA) - float(o_b0.right_BA)) * 100; c0_b0_heldout = (float(h_c0.right_BA) - float(h_b0.right_BA)) * 100
    lines = ["# LiteBN-MS4 WBCIC seed-0 report", "", "MODEL = LiteBN-MS4", "BASE_ARCHITECTURE = EXACT_HISTORICAL_LITEBN", "SEED = 0", "TASK = WBCIC_MI", "ARCHITECTURE_CHANGE = NO", "PARAMETER_INCREASE = 0", "B0_STOCHASTIC_FORWARDS = 1", "C0_STOCHASTIC_FORWARDS = 2", "MS4_STOCHASTIC_FORWARDS = 4", "MS4_OBJECTIVE = MEAN_OF_FOUR_HISTORICAL_SUPERVISED_CE", "KL_LOSS = NO", "CONSISTENCY_LOSS = NO", "PERSISTENT_BN_UPDATES_PER_BATCH = 1", "OPTIMIZER_STEPS_PER_BATCH = 1", "INFERENCE_FORWARD_PASSES = 1", "NEW_SEALED_TEST_ACCESSED = NO", "INTERNAL_HELDOUT_STATUS = DEVELOPMENT_MODEL_SELECTION_DATA", "", f"B0 outer BA = {o_b0.right_BA:.12f}", f"C0 outer BA = {o_c0.right_BA:.12f}", f"MS4 outer BA = {o_b0.left_BA:.12f}", f"C0-B0 outer pp = {c0_b0_outer:+.6f}", f"MS4-B0 outer pp = {o_b0.delta_pp:+.6f}", f"MS4-C0 outer pp = {o_c0.delta_pp:+.6f}", f"B0 heldout BA = {h_b0.right_BA:.12f}", f"C0 heldout BA = {h_c0.right_BA:.12f}", f"MS4 heldout BA = {h_b0.left_BA:.12f}", f"C0-B0 heldout pp = {c0_b0_heldout:+.6f}", f"MS4-B0 heldout pp = {h_b0.delta_pp:+.6f}", f"MS4-C0 heldout pp = {h_c0.delta_pp:+.6f}", f"positive folds vs B0 = {int((fold_b0 > base.TIE_TOL).sum())}/5", f"positive folds vs C0 = {int((fold_c0 > base.TIE_TOL).sum())}/5", f"outer subjects MS4-B0 positive/negative/tied = {int(o_b0.positive_subjects)}/{int(o_b0.negative_subjects)}/{int(o_b0.tied_subjects)}", f"outer subjects MS4-C0 positive/negative/tied = {int(o_c0.positive_subjects)}/{int(o_c0.negative_subjects)}/{int(o_c0.tied_subjects)}", f"heldout subjects MS4-B0 positive/negative/tied = {int(h_b0.positive_subjects)}/{int(h_b0.negative_subjects)}/{int(h_b0.tied_subjects)}", f"heldout subjects MS4-C0 positive/negative/tied = {int(h_c0.positive_subjects)}/{int(h_c0.negative_subjects)}/{int(h_c0.tied_subjects)}", f"outer bootstrap MS4-B0 95% CI pp = [{o_b0.bootstrap_ci_low_pp:.6f}, {o_b0.bootstrap_ci_high_pp:.6f}]", f"outer bootstrap MS4-C0 95% CI pp = [{o_c0.bootstrap_ci_low_pp:.6f}, {o_c0.bootstrap_ci_high_pp:.6f}]", f"heldout bootstrap MS4-B0 95% CI pp = [{h_b0.bootstrap_ci_low_pp:.6f}, {h_b0.bootstrap_ci_high_pp:.6f}]", f"heldout bootstrap MS4-C0 95% CI pp = [{h_c0.bootstrap_ci_low_pp:.6f}, {h_c0.bootstrap_ci_high_pp:.6f}]", "", terminal]
    base.write_text(OUT / "FINAL_MS4_REPORT.md", "\n".join(lines)); print(terminal, flush=True); return terminal


def write_metadata(b0_records: list[dict[str, Any]], c0_records: list[dict[str, Any]]) -> None:
    try: driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception: driver = "unavailable"
    metadata = {"os": platform.platform(), "python": sys.version, "pytorch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0), "driver": driver, "amp": "FP16_autocast_with_GradScaler", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cudnn_benchmark": torch.backends.cudnn.benchmark, "matmul_precision": torch.get_float32_matmul_precision()}
    base.write_json(PROTOCOL / "LINUX_RUNTIME_METADATA.json", metadata)
    base.write_json(PROTOCOL / "REFERENCE_PROVENANCE.json", {"parent_experiment": str(PARENT.relative_to(REPO)), "B0_records": b0_records, "C0_records": c0_records, "reference_results_source": str((PARENT / 'outputs' / 'OUTER_SUBJECT_RESULTS.csv').relative_to(REPO)), "checkpoint_replay_is_not_training_reproduction": True})


def main() -> int:
    if not torch.cuda.is_available(): raise RuntimeError("MS4_PROTOCOL_FAIL CUDA unavailable")
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); device = torch.device("cuda")
    b0_records = sorted(json.loads((shared.REFERENCE_EXP / "protocol" / "LITEBN_CHECKPOINT_PROVENANCE.json").read_text())["records"], key=lambda row: int(row["fold"]))
    c0_records = sorted(json.loads((PARENT / "protocol" / "C0_CHECKPOINT_PROVENANCE.json").read_text())["records"], key=lambda row: int(row["fold"]))
    for rows, label in ((b0_records, B0), (c0_records, C0)):
        if len(rows) != 5: raise RuntimeError(f"MS4_PROTOCOL_FAIL {label} provenance count")
        for row in rows:
            path = Path(row["checkpoint_path"])
            if not path.is_file() or base.sha256_file(path) != row["checkpoint_sha256"]: raise RuntimeError(f"MS4_PROTOCOL_FAIL invalid {label} checkpoint")
    write_metadata(b0_records, c0_records); _, datasets, split_hash = base.load_folds(); folds = {int(row["fold_id"]): row for row in datasets["WBCIC"]}
    provenance_path = PROTOCOL / "MS4_CHECKPOINT_PROVENANCE.json"; existing = json.loads(provenance_path.read_text())["records"] if provenance_path.is_file() else []; records = {int(row["fold"]): row for row in existing}
    paths = {"init": OUT / "MS4_INITIALIZATION_AUDIT.csv", "prefix": OUT / "MS4_C0_PREFIX_EQUIVALENCE.csv", "k2": OUT / "MS4_K2_REDUCTION_AUDIT.csv", "bn": OUT / "MS4_BN_AUTOGRAD_AUDIT.csv", "manifest": OUT / "MANIFEST_AUDIT.csv", "trajectory": OUT / "MS4_TRAINING_TRAJECTORY.csv", "gradient": OUT / "MS4_GRADIENT_VARIANCE_DIAGNOSTICS.csv"}
    def prior(name: str) -> list[dict[str, Any]]: return pd.read_csv(paths[name]).to_dict("records") if paths[name].is_file() else []
    init_rows, prefix_rows, k2_rows, bn_rows, manifest_rows, trajectory_rows, gradient_rows = (prior(name) for name in ("init", "prefix", "k2", "bn", "manifest", "trajectory", "gradient"))
    for fold_id in range(5):
        fold = folds[fold_id]; allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]; bundle = base.build_bundle(TASK, allowed); cache = base.RawGPUCache(bundle, device)
        mean, std, norm_meta, batch_info, manifest = shared.load_inputs(fold); manifest["method"] = METHOD
        model, initialization = shared.initialize_c1(TASK, fold_id, SEED); initialization["method"] = METHOD; initialization["K"] = K; model.to(device)
        value, labels = cache.batch(np.asarray(batch_info["episodes"][0][0], dtype=np.int64), mean, std)
        current_prefix = [prefix_audit(model, value, fold_id, amp) for amp in (False, True)]; current_k2 = [k2_reduction_audit(model, value, labels, fold_id, amp) for amp in (False, True)]; current_bn = [bn_autograd_audit(model, value, labels, fold_id, amp) for amp in (False, True)]; current_bn.append(resume_audit(model, value, labels, fold_id))
        current = [*current_prefix, *current_k2, *current_bn, initialization, manifest]
        if not all(row["status"] == "PASS" for row in current):
            base.write_json(OUT / "FINAL_MS4_DECISION.json", {"status": "MS4_PROTOCOL_FAIL", "fold": fold_id, "failed_audits": [row for row in current if row["status"] != "PASS"]}); raise RuntimeError(f"MS4_PROTOCOL_FAIL fold={fold_id}")
        def replace(rows: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]: return [row for row in rows if int(row["fold"]) != fold_id] + additions
        init_rows = replace(init_rows, [initialization]); prefix_rows = replace(prefix_rows, current_prefix); k2_rows = replace(k2_rows, current_k2); bn_rows = replace(bn_rows, current_bn); manifest_rows = replace(manifest_rows, [manifest])
        for key, rows in (("init", init_rows), ("prefix", prefix_rows), ("k2", k2_rows), ("bn", bn_rows), ("manifest", manifest_rows)): base.write_csv(paths[key], pd.DataFrame(rows).sort_values(["fold"] + (["precision"] if "precision" in pd.DataFrame(rows).columns else [])))
        del model; model, _ = shared.initialize_c1(TASK, fold_id, SEED); model.to(device); base.set_seed(SEED + 100_000)
        record, history, gradient = train_one(model, fold, bundle, cache, mean, std, norm_meta, batch_info, device); records[fold_id] = record
        trajectory_rows = replace(trajectory_rows, history); gradient_rows = replace(gradient_rows, gradient); base.write_csv(paths["trajectory"], pd.DataFrame(trajectory_rows).sort_values(["fold", "epoch"])); base.write_csv(paths["gradient"], pd.DataFrame(gradient_rows).sort_values(["fold", "epoch"])); base.write_json(provenance_path, {"method": METHOD, "seed": SEED, "K": K, "split_sha256": split_hash, "records": [records[index] for index in sorted(records)]})
        del model, bundle, cache, value, labels; gc.collect(); torch.cuda.empty_cache()
    if len(records) != 5: raise RuntimeError("MS4_PROTOCOL_FAIL incomplete checkpoint grid")
    serial = [records[index] for index in range(5)]; base.write_json(PROTOCOL / "MS4_CHECKPOINT_LOCK.json", {"status": "PASS", "method": METHOD, "seed": SEED, "K": K, "five_checkpoints_frozen": True, "outer_accessed_at_lock": False, "internal_heldout_accessed_at_lock": False, "records": [{"fold": row["fold"], "checkpoint_path": row["checkpoint_path"], "checkpoint_sha256": row["checkpoint_sha256"], "checkpoint_tensor_content_sha256": row["checkpoint_tensor_content_sha256"]} for row in serial]})
    evaluate_and_finalize(folds, serial, b0_records, c0_records, device); return 0


if __name__ == "__main__": raise SystemExit(main())
