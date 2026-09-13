"""Run controlled LiteBN-SC (C1 only) on WBCIC-MI seed 0."""
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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path(os.environ.get("SC_V2_REPO", "/root/rivermind-data/CRCICLR_SC_CONTROLLED_V2_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_sc_controlled_linux_seed0_v2"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("SC_V2_RUNTIME", "/root/rivermind-data/litebn_sc_controlled_v2_runtime")).resolve()
REFERENCE_EXP = REPO / "experiments" / "persist_eeg_litebn_xng_linux_seed0_v1"
ORIGINAL_EXP = REPO / "experiments" / "persist_eeg_litebn_x_singlemodel_seed0_v1"
ORIGINAL_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
sys.path.insert(0, str(ORIGINAL_EXP / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("LITEBN_X_REPO", str(REPO))

import litebn_x as base
from litebn_sc_controlled import (
    BN_TYPES,
    bn_snapshot,
    controlled_dual_forward,
    inference_identity_audit,
    initialize_c1,
    initialize_secondary_rng,
    restore_torch_rng,
    symmetric_kl,
    torch_rng_state,
)


TASK, METHOD, SEED = "WBCIC_MI", "C1_LiteBN_SC", 0
LAMBDA_SC, SECONDARY_SEED = 0.5, 200_000
GRADIENT_DIAGNOSTIC_EPOCHS = {1, 10, 20, 30, 40, 50, 60}


def tensor_content_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def source_hash() -> str:
    paths = [
        Path(__file__), Path(__file__).parent / "litebn_sc_controlled.py",
        ORIGINAL_EXP / "code" / "litebn_x.py", base.CARRIER_CODE / "run_carrier_screen.py",
    ]
    return base.sha256_bytes(b"".join(path.read_bytes() for path in paths))


def checkpoint_path(fold: int, name: str) -> Path:
    return RUNTIME / "checkpoints" / "wbcic_mi" / f"fold{fold}_c1_litebn_sc" / name


def state_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> tuple[bool, float]:
    if tuple(left) != tuple(right):
        return False, float("inf")
    maximum = max(
        (float((left[key].detach().float() - right[key].detach().float()).abs().max()) for key in left),
        default=0.0,
    )
    return all(torch.equal(left[key], right[key]) for key in left), maximum


def torch_rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not torch.equal(left["cpu"], right["cpu"]):
        return False
    return len(left.get("cuda", [])) == len(right.get("cuda", [])) and all(
        torch.equal(a, b) for a, b in zip(left.get("cuda", []), right.get("cuda", []))
    )


def load_inputs(fold: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    fold_id = int(fold["fold_id"])
    normalizer_path = ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz"
    manifest_path = ORIGINAL_RUNTIME / "episode_manifests" / f"wbcic_mi_fold{fold_id}.json"
    expected = pd.read_csv(REFERENCE_EXP / "outputs" / "MANIFEST_AUDIT.csv").set_index("fold").loc[fold_id]
    mean, std, norm_meta = base.load_tensor_pair(normalizer_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = base.sha256_file(manifest_path)
    episodes = [[np.asarray(indices, dtype=np.int64) for indices in epoch] for epoch in manifest["epochs"]]
    exposed = np.concatenate([np.concatenate(epoch) for epoch in episodes])
    row = {
        "task": TASK, "fold": fold_id, "manifest_sha256": manifest_hash,
        "reference_manifest_sha256": str(expected.manifest_sha256),
        "manifest_hash_match": manifest_hash == str(expected.manifest_sha256),
        "normalizer_sha256": norm_meta["mean_std_sha256"],
        "reference_normalizer_sha256": str(expected.normalizer_sha256),
        "normalizer_hash_match": norm_meta["mean_std_sha256"] == str(expected.normalizer_sha256),
        "training_subjects_match": base.subject_sort(manifest["training_subjects"], "WBCIC") == base.subject_sort(fold["inner_train_subjects"], "WBCIC"),
        "outer_rows_absent": bool(manifest["outer_rows_absent"]), "epochs": len(episodes),
        "updates": int(sum(len(epoch) for epoch in episodes)), "trial_draws": int(len(exposed)),
        "unique_training_rows": int(len(np.unique(exposed))),
    }
    row["status"] = "PASS" if all((
        row["manifest_hash_match"], row["normalizer_hash_match"], row["training_subjects_match"],
        row["outer_rows_absent"], row["epochs"] == base.EPOCHS,
    )) else "FAIL"
    if row["status"] != "PASS":
        raise RuntimeError(f"SC_PROTOCOL_FAIL manifest: {row}")
    return mean, std, norm_meta, {"episodes": episodes, "manifest_sha256": manifest_hash}, row


def build_future_bundle(subjects: Iterable[str]) -> base.SignalBundle:
    canonical = base.subject_sort(subjects, "WBCIC")
    root = base.CACHE / "wbcic" / "wbcic_epochs"
    rows: list[base.Row] = []
    for subject in canonical:
        signal, labels = root / subject / "ses-2_epochs.npy", root / subject / "ses-2_labels.npy"
        if not signal.is_file() or not labels.is_file():
            raise FileNotFoundError(f"WBCIC future cache missing: {subject}")
        x = np.load(signal, mmap_mode="r", allow_pickle=False)
        y = np.load(labels, mmap_mode="r", allow_pickle=False)
        if x.ndim != 3 or tuple(x.shape[1:]) != (58, 1000) or x.dtype != np.float16:
            raise RuntimeError(f"WBCIC schema mismatch: {signal}")
        if y.shape != (x.shape[0],) or set(map(int, np.unique(y))) != {0, 1}:
            raise RuntimeError(f"WBCIC labels mismatch: {labels}")
        rows.extend(base.Row(str(subject), 2, str(signal), index, int(label)) for index, label in enumerate(y))
    return base.SignalBundle(TASK, canonical, rows)


def _one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    value: torch.Tensor,
    labels: torch.Tensor,
    secondary: dict[str, Any],
    amp: bool,
    coefficient: float,
    use_b_loss: bool = True,
) -> tuple[float, dict[str, Any]]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=value.device.type, dtype=torch.float16, enabled=amp):
        logits_a, _, logits_b, _, secondary = controlled_dual_forward(model, value, secondary)
        ce_a = F.cross_entropy(logits_a, labels)
        ce_b = F.cross_entropy(logits_b, labels)
        ce = 0.5 * (ce_a + ce_b) if use_b_loss else ce_a
    j, _ = symmetric_kl(logits_a, logits_b)
    loss = ce + float(coefficient) * j if use_b_loss else ce
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
    scaler.step(optimizer); scaler.update()
    return float(loss.detach()), secondary


def graph_and_bn_audit(model: nn.Module, value: torch.Tensor, labels: torch.Tensor, fold: int, amp: bool) -> dict[str, Any]:
    baseline, dual = copy.deepcopy(model), copy.deepcopy(model)
    baseline.train(); dual.train()
    base.set_seed(271_828 + fold)
    secondary = initialize_secondary_rng(SECONDARY_SEED + fold)
    a_start = torch_rng_state()
    with torch.no_grad(), torch.autocast(device_type=value.device.type, dtype=torch.float16, enabled=amp):
        baseline_logits, _ = baseline(value)
    baseline_post = bn_snapshot(baseline)
    baseline_rng_post = torch_rng_state()
    baseline_logits = baseline_logits.detach().clone()
    del baseline; gc.collect(); torch.cuda.empty_cache()
    restore_torch_rng(a_start)
    with torch.no_grad(), torch.autocast(device_type=value.device.type, dtype=torch.float16, enabled=amp):
        logits_a, _, logits_b, _, _ = controlled_dual_forward(dual, value, secondary)
    dual_post = bn_snapshot(dual)
    dual_rng_post = torch_rng_state()
    bn_exact, bn_max = state_diff(baseline_post, dual_post)
    first_logits_exact = torch.equal(baseline_logits, logits_a)
    logits_different = not torch.equal(logits_a, logits_b)
    del dual, logits_a, logits_b; gc.collect(); torch.cuda.empty_cache()

    b_graph = copy.deepcopy(model).train()
    base.set_seed(271_828 + fold)
    b_secondary = initialize_secondary_rng(SECONDARY_SEED + fold)
    with torch.autocast(device_type=value.device.type, dtype=torch.float16, enabled=amp):
        _, _, b_logits, _, _ = controlled_dual_forward(b_graph, value, b_secondary)
        b_ce = F.cross_entropy(b_logits, labels)
    b_ce.backward()
    b_grad_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in b_graph.parameters()) and any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in b_graph.parameters()
    )
    del b_graph, b_logits, b_ce; gc.collect(); torch.cuda.empty_cache()

    total_graph = copy.deepcopy(model).train()
    base.set_seed(271_828 + fold)
    total_secondary = initialize_secondary_rng(SECONDARY_SEED + fold)
    before_backward = None
    with torch.autocast(device_type=value.device.type, dtype=torch.float16, enabled=amp):
        ta, _, tb, _, _ = controlled_dual_forward(total_graph, value, total_secondary)
        tce = 0.5 * (F.cross_entropy(ta, labels) + F.cross_entropy(tb, labels))
    tj, _ = symmetric_kl(ta, tb)
    total = tce + LAMBDA_SC * tj
    before_backward = bn_snapshot(total_graph)
    total.backward()
    after_backward = bn_snapshot(total_graph)
    backward_bn_exact, backward_bn_max = state_diff(before_backward, after_backward)
    total_grad_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in total_graph.parameters())
    del total_graph, ta, tb, tce, tj, total; gc.collect(); torch.cuda.empty_cache()
    passed = all((
        bn_exact, bn_max == 0.0, first_logits_exact, logits_different,
        torch_rng_equal(baseline_rng_post, dual_rng_post), bool(b_grad_finite),
        bool(total_grad_finite), backward_bn_exact, backward_bn_max == 0.0,
    ))
    return {
        "task": TASK, "fold": fold, "precision": "AMP_FP16" if amp else "FP32",
        "bn_tensor_count": len(baseline_post), "bn_persistent_exact": bn_exact,
        "max_abs_bn_diff": bn_max, "forward_a_matches_b0": first_logits_exact,
        "different_dropout_logits": logits_different, "main_rng_matches_b0_after_a": torch_rng_equal(baseline_rng_post, dual_rng_post),
        "forward_b_ce_shared_parameter_gradients_finite": bool(b_grad_finite),
        "total_backward_gradients_finite": bool(total_grad_finite),
        "backward_does_not_change_bn": backward_bn_exact, "backward_bn_max_abs_diff": backward_bn_max,
        "uses_data_bypass": False, "status": "PASS" if passed else "FAIL",
    }


def equivalence_audits(model: nn.Module, value: torch.Tensor, labels: torch.Tensor, fold: int, amp: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    precision = "AMP_FP16" if amp else "FP32"
    # C0 and lambda-zero C1 must be the same numerical program except for +0*J.
    c0, l0 = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    o0 = torch.optim.AdamW(c0.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    o1 = torch.optim.AdamW(l0.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    s0 = torch.amp.GradScaler("cuda", enabled=amp); s1 = torch.amp.GradScaler("cuda", enabled=amp)
    base.set_seed(314_159 + fold); main = base.rng_state(); secondary = initialize_secondary_rng(SECONDARY_SEED + fold)
    loss0, secondary0 = _one_step(c0, o0, s0, value, labels, copy.deepcopy(secondary), amp, 0.0)
    end0 = base.rng_state()
    base.restore_rng(main)
    loss1, secondary1 = _one_step(l0, o1, s1, value, labels, copy.deepcopy(secondary), amp, 0.0)
    end1 = base.rng_state()
    exact, maximum = state_diff(c0.state_dict(), l0.state_dict())
    grad_exact = all(
        (a.grad is None and b.grad is None) or (a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad))
        for a, b in zip(c0.parameters(), l0.parameters())
    )
    main_exact = torch_rng_equal(
        {"cpu": end0["torch"], "cuda": end0.get("cuda", [])},
        {"cpu": end1["torch"], "cuda": end1.get("cuda", [])},
    )
    l0_pass = abs(loss0 - loss1) <= 1e-12 and grad_exact and exact and maximum == 0.0 and main_exact and torch_rng_equal(secondary0, secondary1)
    lambda_row = {
        "task": TASK, "fold": fold, "precision": precision, "c0_loss": loss0, "c1_lambda0_loss": loss1,
        "abs_loss_diff": abs(loss0 - loss1), "model_state_exact_after_step": exact,
        "max_abs_model_state_diff": maximum, "gradients_exact": grad_exact, "main_rng_exact": main_exact,
        "secondary_rng_exact": torch_rng_equal(secondary0, secondary1), "status": "PASS" if l0_pass else "FAIL",
    }

    # Null/symmetry checks.
    probe_a = torch.tensor([[1.0, -0.5], [0.2, 0.8]], device=value.device)
    probe_b = torch.tensor([[0.7, -0.1], [-0.2, 1.1]], device=value.device)
    null_j, _ = symmetric_kl(probe_a, probe_a)
    ab, _ = symmetric_kl(probe_a, probe_b); ba, _ = symmetric_kl(probe_b, probe_a)
    null_row = {
        "task": TASK, "fold": fold, "precision": precision, "audit": "NULL_AND_SYMMETRY", "null_J": float(null_j),
        "swap_abs_diff": float((ab - ba).abs()),
        "status": "PASS" if abs(float(null_j)) <= 1e-7 and float((ab - ba).abs()) <= 1e-7 else "FAIL",
    }

    # Extra B must be a no-op when only A CE contributes, including short replay.
    b0, noop = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    ob = torch.optim.AdamW(b0.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    on = torch.optim.AdamW(noop.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    sb = torch.amp.GradScaler("cuda", enabled=amp); sn = torch.amp.GradScaler("cuda", enabled=amp)
    base.set_seed(161_803 + fold); start = base.rng_state(); secondary_n = initialize_secondary_rng(SECONDARY_SEED + fold)
    b0_losses = []
    for _ in range(3):
        ob.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            zb, _ = b0(value); lb = F.cross_entropy(zb, labels)
        sb.scale(lb).backward(); sb.unscale_(ob); torch.nn.utils.clip_grad_norm_(b0.parameters(), base.CLIP); sb.step(ob); sb.update()
        b0_losses.append(float(lb.detach()))
    b0_rng = base.rng_state()
    base.restore_rng(start)
    noop_losses = []
    for _ in range(3):
        ln, secondary_n = _one_step(noop, on, sn, value, labels, secondary_n, amp, 0.0, use_b_loss=False)
        noop_losses.append(ln)
    noop_rng = base.rng_state()
    noop_exact, noop_max = state_diff(b0.state_dict(), noop.state_dict())
    noop_grad_exact = all(
        (a.grad is None and b.grad is None) or (a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad))
        for a, b in zip(b0.parameters(), noop.parameters())
    )
    loss_max_diff = max(abs(a - b) for a, b in zip(b0_losses, noop_losses))
    noop_pass = loss_max_diff <= 1e-12 and noop_grad_exact and noop_exact and torch_rng_equal(
        {"cpu": b0_rng["torch"], "cuda": b0_rng.get("cuda", [])}, {"cpu": noop_rng["torch"], "cuda": noop_rng.get("cuda", [])}
    )
    noop_row = {
        "task": TASK, "fold": fold, "precision": precision, "audit": "NOOP_EXTRA_B_THREE_STEP_REPLAY",
        "max_abs_loss_diff": loss_max_diff, "gradients_exact": noop_grad_exact, "model_state_exact": noop_exact,
        "max_abs_model_state_diff": noop_max, "main_rng_exact": torch_rng_equal(
            {"cpu": b0_rng["torch"], "cuda": b0_rng.get("cuda", [])}, {"cpu": noop_rng["torch"], "cuda": noop_rng.get("cuda", [])}
        ), "status": "PASS" if noop_pass else "FAIL",
    }

    extra_rows = [null_row, noop_row]
    if not amp:
        return lambda_row, extra_rows

    # Actual-AMP ten-step continuous versus 5+resume replay, including both RNG streams.
    continuous, interrupted = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    oc = torch.optim.AdamW(continuous.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    oi = torch.optim.AdamW(interrupted.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    sc = torch.amp.GradScaler("cuda", enabled=True); si = torch.amp.GradScaler("cuda", enabled=True)
    base.set_seed(141_421 + fold); replay_start = base.rng_state(); sec_start = initialize_secondary_rng(SECONDARY_SEED + fold)
    sec_c = copy.deepcopy(sec_start)
    for _ in range(10):
        _, sec_c = _one_step(continuous, oc, sc, value, labels, sec_c, amp, LAMBDA_SC)
    rng_c = base.rng_state()
    base.restore_rng(replay_start); sec_i = copy.deepcopy(sec_start)
    for _ in range(5):
        _, sec_i = _one_step(interrupted, oi, si, value, labels, sec_i, amp, LAMBDA_SC)
    payload = {
        "model": copy.deepcopy(interrupted.state_dict()), "optimizer": copy.deepcopy(oi.state_dict()),
        "scaler": copy.deepcopy(si.state_dict()), "main_rng": base.rng_state(), "secondary_rng": copy.deepcopy(sec_i),
    }
    resumed = copy.deepcopy(model).train(); or_ = torch.optim.AdamW(resumed.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    sr = torch.amp.GradScaler("cuda", enabled=True)
    resumed.load_state_dict(payload["model"], strict=True); or_.load_state_dict(payload["optimizer"]); sr.load_state_dict(payload["scaler"])
    base.restore_rng(payload["main_rng"]); sec_r = payload["secondary_rng"]
    for _ in range(5):
        _, sec_r = _one_step(resumed, or_, sr, value, labels, sec_r, amp, LAMBDA_SC)
    rng_r = base.rng_state()
    resume_exact, resume_max = state_diff(continuous.state_dict(), resumed.state_dict())
    resume_pass = resume_exact and resume_max == 0.0 and torch_rng_equal(sec_c, sec_r) and sc.state_dict() == sr.state_dict() and torch_rng_equal(
        {"cpu": rng_c["torch"], "cuda": rng_c.get("cuda", [])}, {"cpu": rng_r["torch"], "cuda": rng_r.get("cuda", [])}
    )
    resume_row = {
        "task": TASK, "fold": fold, "precision": precision, "audit": "TEN_STEP_RESUME", "model_state_exact": resume_exact,
        "max_abs_model_state_diff": resume_max, "main_rng_exact": torch_rng_equal(
            {"cpu": rng_c["torch"], "cuda": rng_c.get("cuda", [])}, {"cpu": rng_r["torch"], "cuda": rng_r.get("cuda", [])}
        ), "secondary_rng_exact": torch_rng_equal(sec_c, sec_r), "scaler_exact": sc.state_dict() == sr.state_dict(),
        "status": "PASS" if resume_pass else "FAIL",
    }
    return lambda_row, [*extra_rows, resume_row]


def gradient_relation(ce: torch.Tensor, half_j: torch.Tensor, parameters: list[nn.Parameter]) -> tuple[float, float, float, float]:
    grad_ce = torch.autograd.grad(ce, parameters, retain_graph=True, allow_unused=True)
    grad_j = torch.autograd.grad(half_j, parameters, retain_graph=True, allow_unused=True)
    ce_sq = sum((g.float().square().sum() for g in grad_ce if g is not None), torch.zeros((), device=ce.device))
    j_sq = sum((g.float().square().sum() for g in grad_j if g is not None), torch.zeros((), device=ce.device))
    dot = sum((a.float().mul(b.float()).sum() for a, b in zip(grad_ce, grad_j) if a is not None and b is not None), torch.zeros((), device=ce.device))
    ce_norm, j_norm = ce_sq.sqrt(), j_sq.sqrt()
    ratio = j_norm / ce_norm.clamp_min(1e-20)
    cosine = dot / (ce_norm * j_norm).clamp_min(1e-20)
    return float(ce_norm), float(j_norm), float(ratio), float(cosine)


def train_one(
    model: nn.Module, fold: dict[str, Any], bundle: base.SignalBundle, cache: base.RawGPUCache,
    mean: np.ndarray, std: np.ndarray, norm_meta: dict[str, Any], batch_info: dict[str, Any], device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_id = int(fold["fold_id"])
    latest, selected = checkpoint_path(fold_id, "checkpoint_latest.pt"), checkpoint_path(fold_id, "selected_best.pt")
    initial_content_hash = tensor_content_hash(model.state_dict())
    invariants = {
        "task": TASK, "fold": fold_id, "method": METHOD, "seed": SEED, "lambda_sc": LAMBDA_SC,
        "secondary_seed": SECONDARY_SEED, "initial_tensor_content_sha256": initial_content_hash,
        "source_sha256": source_hash(), "normalizer_sha256": norm_meta["mean_std_sha256"],
        "batch_manifest_sha256": batch_info["manifest_sha256"], "bn_persistent_updates_per_batch": 1,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    secondary = initialize_secondary_rng(SECONDARY_SEED)
    start, history, grad_rows, best, best_epoch, best_state = 1, [], [], -float("inf"), None, None
    elapsed_before, peak_before = 0.0, 0
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"]); base.restore_rng(saved["main_rng"]); secondary = saved["secondary_rng"]
        start, history, grad_rows = int(saved["epoch"]) + 1, list(saved["history"]), list(saved["gradient_diagnostics"])
        best, best_epoch, best_state = float(saved["best"]), saved["best_epoch"], saved["best_state"]
        elapsed_before = float(saved.get("elapsed_seconds_completed", 0.0)); peak_before = int(saved.get("peak_cuda_memory_bytes", 0))
    attempted_total = sum(int(row["attempted_optimizer_steps"]) for row in history)
    successful_total = sum(int(row["successful_optimizer_steps"]) for row in history)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(start, base.EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        values: dict[str, list[float]] = {key: [] for key in (
            "ce_a", "ce_b", "ce", "j", "total", "disagree", "prob_l1", "logit_l2", "entropy_a", "entropy_b", "grad", "scale",
        )}
        attempted, successful, skips = 0, 0, 0
        for batch_index, indices in enumerate(batch_info["episodes"][epoch - 1]):
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_a, _, logits_b, _, secondary = controlled_dual_forward(model, value, secondary)
                ce_a, ce_b = F.cross_entropy(logits_a, labels), F.cross_entropy(logits_b, labels)
                ce = 0.5 * (ce_a + ce_b)
            j, per_sample_j = symmetric_kl(logits_a, logits_b)
            loss = ce + LAMBDA_SC * j
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite C1 loss fold={fold_id} epoch={epoch}")
            if epoch in GRADIENT_DIAGNOSTIC_EPOCHS and batch_index == 0:
                ce_norm, j_norm, ratio, cosine = gradient_relation(ce, LAMBDA_SC * j, parameters)
                grad_rows.append({
                    "task": TASK, "method": METHOD, "fold": fold_id, "epoch": epoch, "batch_index": batch_index,
                    "ce_gradient_norm": ce_norm, "weighted_J_gradient_norm": j_norm,
                    "weighted_J_to_CE_gradient_norm_ratio": ratio, "gradient_cosine": cosine,
                })
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, base.CLIP)
            scale_before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update(); scale_after = float(scaler.get_scale())
            skipped = int(scale_after < scale_before); attempted += 1; successful += 1 - skipped; skips += skipped
            with torch.no_grad():
                pa, pb = F.softmax(logits_a.float(), dim=-1), F.softmax(logits_b.float(), dim=-1)
                values["ce_a"].append(float(ce_a)); values["ce_b"].append(float(ce_b)); values["ce"].append(float(ce))
                values["j"].append(float(j)); values["total"].append(float(loss))
                values["disagree"].append(float((logits_a.argmax(-1) != logits_b.argmax(-1)).float().mean()))
                values["prob_l1"].append(float((pa - pb).abs().sum(-1).mean()))
                values["logit_l2"].append(float(torch.linalg.vector_norm(logits_a.float() - logits_b.float(), dim=-1).mean()))
                values["entropy_a"].append(float(-(pa * pa.clamp_min(1e-12).log()).sum(-1).mean()))
                values["entropy_b"].append(float(-(pb * pb.clamp_min(1e-12).log()).sum(-1).mean()))
                values["grad"].append(float(grad_norm)); values["scale"].append(scale_before)
        attempted_total += attempted; successful_total += successful
        validation = base.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        val_ba = float(np.mean([row["BA"] for row in validation.values()]))
        val_f1 = float(np.mean([row["macro_F1"] for row in validation.values()]))
        chose = epoch >= base.MIN_EPOCH and val_ba > best + base.TIE_TOL
        if chose:
            best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        row = {
            "task": TASK, "method": METHOD, "fold": fold_id, "epoch": epoch,
            "mean_CE_A": np.mean(values["ce_a"]), "mean_CE_B": np.mean(values["ce_b"]),
            "mean_supervised_CE": np.mean(values["ce"]), "mean_J": np.mean(values["j"]),
            "median_J": np.median(values["j"]), "mean_total_loss": np.mean(values["total"]),
            "train_mode_prediction_disagreement": np.mean(values["disagree"]),
            "mean_probability_L1_disagreement": np.mean(values["prob_l1"]),
            "mean_logit_L2_distance": np.mean(values["logit_l2"]), "mean_entropy_A": np.mean(values["entropy_a"]),
            "mean_entropy_B": np.mean(values["entropy_b"]), "mean_preclip_gradient_norm": np.mean(values["grad"]),
            "mean_gradscaler_scale": np.mean(values["scale"]), "attempted_optimizer_steps": attempted,
            "successful_optimizer_steps": successful, "amp_skips": skips, "inner_val_subject_BA": val_ba,
            "inner_val_subject_macro_F1": val_f1, "selected": bool(chose), "epoch_runtime_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        base.atomic_torch_save(latest, {
            "epoch": epoch, "history": history, "gradient_diagnostics": grad_rows, "best": best,
            "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "main_rng": base.rng_state(),
            "secondary_rng": secondary, "invariants": invariants,
            "elapsed_seconds_completed": elapsed_before + time.perf_counter() - started,
            "peak_cuda_memory_bytes": max(peak_before, int(torch.cuda.max_memory_allocated(device))),
        })
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[C1 {TASK} f{fold_id}] epoch={epoch:02d} L={row['mean_total_loss']:.4f} CE={row['mean_supervised_CE']:.4f} J={row['mean_J']:.5f} valBA={val_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("SC_PROTOCOL_FAIL no eligible checkpoint")
    model.load_state_dict(best_state, strict=True); base.atomic_torch_save(selected, model.state_dict())
    return {
        "task": TASK, "dataset": "WBCIC", "fold": fold_id, "method": METHOD, "seed": SEED,
        "parameter_count": base.parameter_count(model), "initial_tensor_content_sha256": initial_content_hash,
        "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best),
        "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)),
        "checkpoint_path": str(selected), "checkpoint_sha256": base.sha256_file(selected),
        "checkpoint_tensor_content_sha256": tensor_content_hash(model.state_dict()),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
        "elapsed_seconds_total": elapsed_before + time.perf_counter() - started, "epochs_completed": len(history),
        "attempted_optimizer_steps": attempted_total, "successful_optimizer_steps": successful_total,
        "peak_cuda_memory_bytes": max(peak_before, int(torch.cuda.max_memory_allocated(device))), "source_sha256": source_hash(),
    }, history, grad_rows


def load_model(record: dict[str, Any], fold: int, device: torch.device) -> nn.Module:
    model, _ = initialize_c1(TASK, fold, SEED)
    checkpoint = Path(record["checkpoint_path"])
    if not checkpoint.is_file() or base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError("SC_PROTOCOL_FAIL C1 checkpoint provenance")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if tensor_content_hash(state) != record["checkpoint_tensor_content_sha256"]:
        raise RuntimeError("SC_PROTOCOL_FAIL C1 tensor-content hash")
    model.load_state_dict(state, strict=True); model.to(device).eval()
    return model


def load_b0(fold: int, device: torch.device, references: list[dict[str, Any]]) -> nn.Module:
    record = references[fold]; checkpoint = Path(record["checkpoint_path"])
    if not checkpoint.is_file() or base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError("SC_PROTOCOL_FAIL B0 checkpoint provenance")
    model, _ = initialize_c1(TASK, fold, SEED)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
    return model.to(device).eval()


def dropout_diagnostic(model: nn.Module, value: torch.Tensor, labels: torch.Tensor, method: str, fold: int) -> dict[str, Any]:
    saved_rng, modes = base.rng_state(), [(module, module.training) for module in model.modules()]
    model.eval()
    with torch.no_grad():
        eval_logits, _ = model(value)
        eval_prob = F.softmax(eval_logits.float(), dim=-1)
    model.train()
    for module in model.modules():
        if isinstance(module, BN_TYPES):
            module.eval()
    base.set_seed(700_000 + fold)
    stochastic: list[torch.Tensor] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(16):
            stochastic.append(model(value)[0].float())
    for module, mode in modes:
        module.training = mode
    base.restore_rng(saved_rng)
    probabilities = [F.softmax(logits, dim=-1) for logits in stochastic]
    disagreements, divergences = [], []
    for i in range(16):
        for j in range(i + 1, 16):
            disagreements.append(float((stochastic[i].argmax(-1) != stochastic[j].argmax(-1)).float().mean()))
            divergence, _ = symmetric_kl(stochastic[i], stochastic[j]); divergences.append(float(divergence))
    eval_metrics = base.classification_metrics(labels.detach().cpu().numpy(), eval_logits.detach().cpu().numpy())
    nll = np.mean([float(F.cross_entropy(logits, labels)) for logits in stochastic])
    entropy = np.mean([float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean()) for p in probabilities])
    return {
        "task": TASK, "fold": fold, "method": method, "status": "RUN",
        "samples": int(len(labels)), "stochastic_passes": 16, "bn_mode": "EVAL", "dropout_active": True,
        "mean_pairwise_prediction_disagreement": float(np.mean(disagreements)),
        "mean_pairwise_symmetric_KL": float(np.mean(divergences)),
        "mean_probability_L1_from_standard_eval": float(np.mean([float((p - eval_prob).abs().sum(-1).mean()) for p in probabilities])),
        "stochastic_mean_NLL": float(nll), "stochastic_mean_entropy": float(entropy),
        "standard_eval_BA": eval_metrics["BA"], "standard_eval_macro_F1": eval_metrics["macro_F1"],
        "standard_eval_accuracy": eval_metrics["accuracy"], "used_for_selection": False,
    }


def paired_bootstrap(delta: np.ndarray, seed: int = 0) -> dict[str, Any]:
    draws = np.random.default_rng(seed).choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
    return {
        "n_subjects": len(delta), "bootstrap_seed": seed, "resamples": 10_000,
        "mean_delta_pp": float(delta.mean()), "ci_low_pp": float(np.quantile(draws, 0.025)),
        "ci_high_pp": float(np.quantile(draws, 0.975)), "positive_subjects": int((delta > base.TIE_TOL).sum()),
        "negative_subjects": int((delta < -base.TIE_TOL).sum()), "tied_subjects": int((np.abs(delta) <= base.TIE_TOL).sum()),
    }


def evaluate_all(
    folds: dict[int, dict[str, Any]], records: list[dict[str, Any]], references: list[dict[str, Any]], device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference_outer_all = pd.read_csv(REFERENCE_EXP / "outputs" / "WBCIC_OUTER_SUBJECT_RESULTS.csv")
    reference_outer = reference_outer_all[reference_outer_all.method == "LiteBN_BASELINE"].copy()
    outer_rows, fold_rows, diagnostic_rows = [], [], []
    for fold_id in range(5):
        fold = folds[fold_id]; bundle = build_future_bundle(fold["outer_dev_subjects"]); cache = base.RawGPUCache(bundle, device)
        mean, std, _ = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        models = {"B0_Exact_LiteBN": load_b0(fold_id, device, references), METHOD: load_model(records[fold_id], fold_id, device)}
        for method, model in models.items():
            metrics_by_subject = base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std)
            for subject, metrics in metrics_by_subject.items():
                outer_rows.append({"task": TASK, "fold": fold_id, "subject_id": subject, "method": method, **metrics})
        validation_bundle = base.build_bundle(TASK, fold["inner_train_subjects"] + fold["inner_val_subjects"])
        validation_cache = base.RawGPUCache(validation_bundle, device)
        diagnostic_indices = validation_bundle.indices(fold["inner_val_subjects"], (int(base.TASKS[TASK]["future_session"]),))[:128]
        diagnostic_value, diagnostic_labels = validation_cache.batch(diagnostic_indices, mean, std)
        for method, model in models.items():
            diagnostic_rows.append(dropout_diagnostic(model, diagnostic_value, diagnostic_labels, method, fold_id))
        diagnostic_rows.append({"task": TASK, "fold": fold_id, "method": "C0_LiteBN_DualCE", "status": "NOT_RUN_CURRENT_SCOPE"})
        del models, cache, bundle, validation_cache, validation_bundle; gc.collect(); torch.cuda.empty_cache()
    outer = pd.DataFrame(outer_rows)
    # Replay must match the frozen machine-readable B0 result per true subject/fold key.
    replay = outer[outer.method == "B0_Exact_LiteBN"].sort_values(["fold", "subject_id"]).reset_index(drop=True)
    frozen = reference_outer.sort_values(["fold", "subject_id"]).reset_index(drop=True)
    if list(zip(replay.fold, replay.subject_id)) != list(zip(frozen.fold, frozen.subject_id)):
        raise RuntimeError("SC_PROTOCOL_FAIL B0 outer replay key mismatch")
    replay_max = float(np.max(np.abs(replay.BA.to_numpy() - frozen.BA.to_numpy())))
    if replay_max > 1e-12:
        raise RuntimeError(f"SC_PROTOCOL_FAIL B0 outer replay metric diff {replay_max}")
    base.write_csv(OUT / "OUTER_SUBJECT_RESULTS.csv", outer.sort_values(["fold", "subject_id", "method"]))
    for (fold_id, method), q in outer.groupby(["fold", "method"]):
        fold_rows.append({
            "task": TASK, "fold": fold_id, "method": method, "subjects": len(q), "BA": q.BA.mean(),
            "macro_F1": q.macro_F1.mean(), "accuracy": q.accuracy.mean(),
            "selected_epoch": references[int(fold_id)]["selected_epoch"] if method == "B0_Exact_LiteBN" else records[int(fold_id)]["selected_epoch"],
            "selected_inner_val_BA": references[int(fold_id)]["best_inner_val_BA"] if method == "B0_Exact_LiteBN" else records[int(fold_id)]["best_inner_val_BA"],
        })
    fold_frame = pd.DataFrame(fold_rows)
    pivot = fold_frame.pivot(index="fold", columns="method", values="BA")
    fold_delta = (pivot[METHOD] - pivot["B0_Exact_LiteBN"]) * 100
    fold_frame = fold_frame.merge(fold_delta.rename("C1_minus_B0_BA_pp"), left_on="fold", right_index=True)
    base.write_csv(OUT / "OUTER_FOLD_RESULTS.csv", fold_frame.sort_values(["fold", "method"]))
    outer_subject = outer.pivot(index="subject_id", columns="method", values=["BA", "macro_F1", "accuracy"])
    delta = (outer_subject[("BA", METHOD)] - outer_subject[("BA", "B0_Exact_LiteBN")]) * 100
    boot_outer = paired_bootstrap(delta.to_numpy())
    outer_summary = {
        "B0_BA": float(outer_subject[("BA", "B0_Exact_LiteBN")].mean()), "C1_BA": float(outer_subject[("BA", METHOD)].mean()),
        "C1_minus_B0_pp": float(delta.mean()), "positive_folds": int((fold_delta > base.TIE_TOL).sum()),
        "negative_folds": int((fold_delta < -base.TIE_TOL).sum()), "tied_folds": int((fold_delta.abs() <= base.TIE_TOL).sum()),
        "worst_fold_delta_pp": float(fold_delta.min()), "best_fold_delta_pp": float(fold_delta.max()),
        "positive_subjects": boot_outer["positive_subjects"], "negative_subjects": boot_outer["negative_subjects"],
        "tied_subjects": boot_outer["tied_subjects"], "bootstrap_ci_low_pp": boot_outer["ci_low_pp"],
        "bootstrap_ci_high_pp": boot_outer["ci_high_pp"], "B0_replay_max_abs_BA_diff": replay_max,
    }

    holdout_manifest = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
    subjects = base.subject_sort(json.loads(holdout_manifest.read_text(encoding="utf-8"))["WBCIC"]["subject_ids"], "WBCIC")
    heldout_rows = []
    for fold_id in range(5):
        bundle = build_future_bundle(subjects); cache = base.RawGPUCache(bundle, device)
        mean, std, _ = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        models = {"B0_Exact_LiteBN": load_b0(fold_id, device, references), METHOD: load_model(records[fold_id], fold_id, device)}
        for method, model in models.items():
            for subject, metrics in base.evaluate(model, bundle, cache, subjects, mean, std).items():
                heldout_rows.append({
                    "data_status": "DEVELOPMENT_MODEL_SELECTION_DATA", "task": TASK, "fold": fold_id,
                    "subject_id": subject, "method": method, **metrics,
                })
        del models, cache, bundle; gc.collect(); torch.cuda.empty_cache()
    heldout = pd.DataFrame(heldout_rows)
    frozen_heldout_all = pd.read_csv(REFERENCE_EXP / "outputs" / "WBCIC_INTERNAL_HELDOUT_REPLICATE_RESULTS.csv")
    frozen_h = frozen_heldout_all[frozen_heldout_all.method == "LiteBN_BASELINE"].sort_values(["fold", "subject_id"]).reset_index(drop=True)
    replay_h = heldout[heldout.method == "B0_Exact_LiteBN"].sort_values(["fold", "subject_id"]).reset_index(drop=True)
    heldout_replay_max = float(np.max(np.abs(replay_h.BA.to_numpy() - frozen_h.BA.to_numpy())))
    if list(zip(replay_h.fold, replay_h.subject_id)) != list(zip(frozen_h.fold, frozen_h.subject_id)) or heldout_replay_max > 1e-12:
        raise RuntimeError("SC_PROTOCOL_FAIL B0 heldout replay mismatch")
    heldout_subject = heldout.groupby(["subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    base.write_csv(OUT / "HELDOUT_SUBJECT_RESULTS.csv", heldout_subject.sort_values(["subject_id", "method"]))
    hp = heldout_subject.pivot(index="subject_id", columns="method", values=["BA", "macro_F1", "accuracy"])
    hdelta = (hp[("BA", METHOD)] - hp[("BA", "B0_Exact_LiteBN")]) * 100
    boot_holdout = paired_bootstrap(hdelta.to_numpy())
    heldout_summary = {
        "data_status": "DEVELOPMENT_MODEL_SELECTION_DATA", "B0_BA": float(hp[("BA", "B0_Exact_LiteBN")].mean()),
        "C1_BA": float(hp[("BA", METHOD)].mean()), "C1_minus_B0_pp": float(hdelta.mean()),
        "positive_subjects": boot_holdout["positive_subjects"], "negative_subjects": boot_holdout["negative_subjects"],
        "tied_subjects": boot_holdout["tied_subjects"], "bootstrap_ci_low_pp": boot_holdout["ci_low_pp"],
        "bootstrap_ci_high_pp": boot_holdout["ci_high_pp"], "B0_replay_max_abs_BA_diff": heldout_replay_max,
    }
    comparisons = [
        {"task": TASK, "cohort": "OUTER", "comparison": "C1-B0", **outer_summary},
        {"task": TASK, "cohort": "INTERNAL_HELDOUT", "comparison": "C1-B0", **heldout_summary},
        {"task": TASK, "cohort": "OUTER", "comparison": "C0-B0", "status": "NOT_RUN_CURRENT_SCOPE"},
        {"task": TASK, "cohort": "OUTER", "comparison": "C1-C0", "status": "NOT_RUN_CURRENT_SCOPE"},
        {"task": TASK, "cohort": "INTERNAL_HELDOUT", "comparison": "C0-B0", "status": "NOT_RUN_CURRENT_SCOPE"},
        {"task": TASK, "cohort": "INTERNAL_HELDOUT", "comparison": "C1-C0", "status": "NOT_RUN_CURRENT_SCOPE"},
    ]
    base.write_csv(OUT / "PAIRED_COMPARISONS.csv", pd.DataFrame(comparisons))
    base.write_csv(OUT / "DROPOUT_DIAGNOSTICS.csv", pd.DataFrame(diagnostic_rows))
    return outer_summary, heldout_summary


def write_metadata(identity: dict[str, Any], references: list[dict[str, Any]]) -> None:
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0].strip()
    except Exception:
        driver = None
    runtime = {
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0), "driver": driver, "amp": True, "gradscaler": True,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "matmul_precision": torch.get_float32_matmul_precision(), "source_commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "source_sha256": source_hash(),
    }
    base.write_json(PROTOCOL / "LINUX_RUNTIME_METADATA.json", runtime)
    identity.update({
        "architecture_source": str(base.CARRIER_CODE / "run_carrier_screen.py"),
        "architecture_source_sha256": base.sha256_file(base.CARRIER_CODE / "run_carrier_screen.py"),
        "training_manifests_source": str(ORIGINAL_RUNTIME / "episode_manifests"),
        "normalizers_source": str(ORIGINAL_RUNTIME / "normalizers"), "seed": SEED,
    })
    base.write_json(PROTOCOL / "MODEL_AND_DATA_IDENTITY.json", identity)
    provenance = {
        "reference_experiment": str(REFERENCE_EXP), "reference_branch": "codex/persist-eeg-litebn-xng-linux-seed0-v1",
        "reference_commit": "0576524b9fb03b77f81f0a642b4fc78d9d5b191e",
        "reference_outer_subject_results_sha256": base.sha256_file(REFERENCE_EXP / "outputs" / "WBCIC_OUTER_SUBJECT_RESULTS.csv"),
        "reference_heldout_replicates_sha256": base.sha256_file(REFERENCE_EXP / "outputs" / "WBCIC_INTERNAL_HELDOUT_REPLICATE_RESULTS.csv"),
        "checkpoint_records": references, "checkpoint_evaluation_replay": "PENDING_UNTIL_POST_TRAIN_EVALUATION",
        "full_training_trajectory_replayed": False,
    }
    base.write_json(PROTOCOL / "REFERENCE_PROVENANCE.json", provenance)


def finalize(records: list[dict[str, Any]], outer: dict[str, Any], heldout: dict[str, Any]) -> str:
    audit_paths = ["INITIALIZATION_AUDIT.csv", "MANIFEST_AUDIT.csv", "BN_AND_AUTOGRAD_AUDIT.csv", "LAMBDA0_EQUIVALENCE.csv", "NOOP_AND_RESUME_AUDIT.csv"]
    audits_pass = all((pd.read_csv(OUT / name).status == "PASS").all() for name in audit_paths)
    known_conditions = {
        "C1_minus_B0_outer_ge_0_30_pp": outer["C1_minus_B0_pp"] >= 0.30,
        "C1_minus_B0_heldout_ge_0_pp": heldout["C1_minus_B0_pp"] >= 0.0,
        "positive_outer_folds_ge_3": outer["positive_folds"] >= 3,
        "worst_outer_fold_gt_minus_1_pp": outer["worst_fold_delta_pp"] > -1.0,
        "all_protocol_and_graph_audits_pass": bool(audits_pass),
    }
    independent_c1_failure = not all(known_conditions.values())
    terminal = "SC_NO_CONTINUATION_SIGNAL" if independent_c1_failure else "C1_WBCIC_COMPLETE_C0_REQUIRED"
    decision = {
        "status": terminal, "scope_completed": "C1_WBCIC_MI_SEED0_FIVE_FOLDS_OUTER_AND_INTERNAL_HELDOUT",
        "known_conditions": known_conditions, "C1_gt_C0_outer": "NOT_EVALUABLE_C0_NOT_RUN_CURRENT_SCOPE",
        "C1_ge_C0_heldout": "NOT_EVALUABLE_C0_NOT_RUN_CURRENT_SCOPE", "phase2_executed": False,
        "reason_phase2_not_executed": "C0 comparison is a mandatory gate component and was excluded by the current user scope" if not independent_c1_failure else "At least one C1-versus-B0 gate condition failed",
        "outer": outer, "internal_heldout": heldout, "new_sealed_test_accessed": False, "seed1_2_run": False,
    }
    base.write_json(OUT / "CONTINUATION_DECISION.json", decision)
    trajectory = pd.read_csv(OUT / "TRAINING_TRAJECTORIES.csv")
    gradients = pd.read_csv(OUT / "GRADIENT_DIAGNOSTICS.csv")
    dropout = pd.read_csv(OUT / "DROPOUT_DIAGNOSTICS.csv")
    diag = dropout[dropout.status == "RUN"].groupby("method")[["mean_pairwise_prediction_disagreement", "mean_pairwise_symmetric_KL"]].mean()
    b0_reference = json.loads((PROTOCOL / "REFERENCE_PROVENANCE.json").read_text())["checkpoint_records"]
    c1_seconds = sum(float(row["elapsed_seconds_total"]) for row in records)
    b0_seconds = sum(float(row["elapsed_seconds_this_invocation"]) for row in b0_reference)
    lines = [
        "# Controlled LiteBN-SC v2: C1 WBCIC seed-0 report", "",
        "ARCHITECTURE_CHANGE = NO", "INFERENCE_PARAMETER_INCREASE = 0", "INFERENCE_FORWARD_PASSES = 1",
        "SC_REFERENCE = R_DROP_STYLE", "DUAL_CE_CONTROL = YES", "BN_PERSISTENT_UPDATES_PER_BATCH = 1",
        "NEW_SEALED_TEST_ACCESSED = NO", "SEED1_2_RUN = NO", "", "## Scope", "",
        "C1 WBCIC_MI seed0 folds 0-4 were trained. B0 was replayed from exact frozen Linux checkpoints. C0 was not run under the current user instruction.",
        "The internal-heldout cohort is DEVELOPMENT_MODEL_SELECTION_DATA, not a sealed final test.", "", "## Results", "",
        f"Outer B0 BA = {outer['B0_BA']:.8f}", f"Outer C1 BA = {outer['C1_BA']:.8f}",
        f"Outer C1-B0 = {outer['C1_minus_B0_pp']:+.4f} pp", f"Outer improved/harmed/tied subjects = {outer['positive_subjects']}/{outer['negative_subjects']}/{outer['tied_subjects']}",
        f"Outer improved folds = {outer['positive_folds']}/5; worst fold = {outer['worst_fold_delta_pp']:+.4f} pp",
        f"Heldout B0 BA = {heldout['B0_BA']:.8f}", f"Heldout C1 BA = {heldout['C1_BA']:.8f}",
        f"Heldout C1-B0 = {heldout['C1_minus_B0_pp']:+.4f} pp", f"Heldout improved/harmed/tied subjects = {heldout['positive_subjects']}/{heldout['negative_subjects']}/{heldout['tied_subjects']}",
        "", "## Required interpretation", "",
        f"1. C1 versus original LiteBN: {'higher' if outer['C1_minus_B0_pp'] > 0 else 'not higher'} on outer; {'higher-or-equal' if heldout['C1_minus_B0_pp'] >= 0 else 'lower'} on heldout.",
        "2. C1 versus DualCE: not answerable because C0 was not run in the current scope.",
        f"3. Outer and heldout direction: {'same' if np.sign(outer['C1_minus_B0_pp']) == np.sign(heldout['C1_minus_B0_pp']) else 'opposite'}.",
        f"4. Subject effects: outer {outer['positive_subjects']} improved and {outer['negative_subjects']} harmed; heldout {heldout['positive_subjects']} improved and {heldout['negative_subjects']} harmed.",
        f"5. Selected-checkpoint stochastic consistency B0/C1 pairwise KL = {diag.loc['B0_Exact_LiteBN','mean_pairwise_symmetric_KL']:.6f}/{diag.loc[METHOD,'mean_pairwise_symmetric_KL']:.6f}; this is observational.",
        f"6. Consistency improved while BA declined: outer={'YES' if diag.loc[METHOD,'mean_pairwise_symmetric_KL'] < diag.loc['B0_Exact_LiteBN','mean_pairwise_symmetric_KL'] and outer['C1_minus_B0_pp'] < 0 else 'NO'}; heldout={'YES' if diag.loc[METHOD,'mean_pairwise_symmetric_KL'] < diag.loc['B0_Exact_LiteBN','mean_pairwise_symmetric_KL'] and heldout['C1_minus_B0_pp'] < 0 else 'NO'}. The consistency diagnostic is observational.",
        f"7. Mean selected inner-val BA B0/C1 = {np.mean([r['best_inner_val_BA'] for r in b0_reference]):.6f}/{np.mean([r['best_inner_val_BA'] for r in records]):.6f}; no causal conclusion without C0.",
        f"8. Median ||grad(0.5J)||/||grad(CE)|| = {gradients.weighted_J_to_CE_gradient_norm_ratio.median():.6f}.",
        "9. Persistent BN state updated once per original batch; exact BN/autograd audits passed." if audits_pass else "9. BN/autograd audit did not fully pass.",
        f"10. Recorded C1 training wall time = {c1_seconds:.1f} s. Historical B0 artifacts record {b0_seconds:.1f} s, but this is not a controlled timing benchmark, so their ratio does not estimate compute change. C1 executes two training forwards versus B0's one. C1 peak allocated CUDA memory = {max(r['peak_cuda_memory_bytes'] for r in records)/2**30:.3f} GiB; B0 peak memory was not recorded, so the memory increase is unknown.",
        "11. Inference remains exact historical LiteBN with one deterministic forward.",
        f"12. Phase2 executed: NO. Reason: {decision['reason_phase2_not_executed']}.", "", terminal,
    ]
    base.write_text(OUT / "FINAL_REPORT.md", "\n".join(lines))
    ledger = pd.DataFrame([
        {"item": "B0 exact Linux checkpoints and metric replay", "status": "RUN", "evidence": "REFERENCE_PROVENANCE.json and replay diffs"},
        {"item": "C1 WBCIC seed0 five-fold training", "status": "RUN", "evidence": "checkpoint provenance and TRAINING_TRAJECTORIES.csv"},
        {"item": "C1 outer evaluation", "status": "RUN", "evidence": "OUTER_SUBJECT_RESULTS.csv"},
        {"item": "C1 internal heldout", "status": "RUN", "evidence": "HELDOUT_SUBJECT_RESULTS.csv; development data"},
        {"item": "C0 DualCE", "status": "PLANNED_NOT_RUN_CURRENT_SCOPE", "evidence": "required before C1-C0 claim"},
        {"item": "Phase2 OpenBMI tasks", "status": "PLANNED_GATE_NOT_EVALUABLE", "evidence": decision["reason_phase2_not_executed"]},
        {"item": "Old SC v1", "status": "PROTOCOL_MISMATCH_PAUSED", "evidence": "natural shared RNG stream; not renamed or reused"},
        {"item": "New sealed final test", "status": "NOT_ACCESSED", "evidence": "protocol lock"},
    ])
    base.write_csv(OUT / "EXPERIMENT_EVIDENCE_LEDGER.csv", ledger)
    provenance = json.loads((PROTOCOL / "REFERENCE_PROVENANCE.json").read_text())
    provenance["checkpoint_evaluation_replay"] = {
        "status": "PASS", "outer_max_abs_BA_diff": outer["B0_replay_max_abs_BA_diff"],
        "heldout_max_abs_BA_diff": heldout["B0_replay_max_abs_BA_diff"],
    }
    base.write_json(PROTOCOL / "REFERENCE_PROVENANCE.json", provenance)
    print(terminal, flush=True)
    return terminal


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("SC_PROTOCOL_FAIL CUDA unavailable")
    device = torch.device("cuda")
    identity = inference_identity_audit(TASK)
    reference_provenance = json.loads((REFERENCE_EXP / "protocol" / "LITEBN_CHECKPOINT_PROVENANCE.json").read_text())
    references = sorted(reference_provenance["records"], key=lambda row: int(row["fold"]))
    if len(references) != 5:
        raise RuntimeError("SC_PROTOCOL_FAIL expected five B0 checkpoints")
    write_metadata(identity, references)
    _, folds_by_dataset, split_hash = base.load_folds()
    folds = {int(row["fold_id"]): row for row in folds_by_dataset["WBCIC"]}
    provenance_path = PROTOCOL / "C1_CHECKPOINT_PROVENANCE.json"
    existing = json.loads(provenance_path.read_text())["records"] if provenance_path.is_file() else []
    by_fold = {int(row["fold"]): row for row in existing}
    init_rows, manifest_rows, bn_rows, lambda_rows, noop_rows = [], [], [], [], []
    all_history = pd.read_csv(OUT / "TRAINING_TRAJECTORIES.csv").to_dict("records") if (OUT / "TRAINING_TRAJECTORIES.csv").is_file() else []
    all_gradients = pd.read_csv(OUT / "GRADIENT_DIAGNOSTICS.csv").to_dict("records") if (OUT / "GRADIENT_DIAGNOSTICS.csv").is_file() else []
    for fold_id in range(5):
        fold = folds[fold_id]
        allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
        if set(allowed) & set(fold["outer_dev_subjects"]):
            raise RuntimeError("SC_PROTOCOL_FAIL outer subject in fitting bundle")
        bundle = base.build_bundle(TASK, allowed); cache = base.RawGPUCache(bundle, device)
        mean, std, norm_meta, batch_info, manifest_row = load_inputs(fold); manifest_rows.append(manifest_row)
        model, init_row = initialize_c1(TASK, fold_id, SEED); model.to(device); init_rows.append(init_row)
        audit_indices = np.asarray(batch_info["episodes"][0][0], dtype=np.int64)
        audit_value, audit_labels = cache.batch(audit_indices, mean, std)
        bn_rows.extend([
            graph_and_bn_audit(model, audit_value, audit_labels, fold_id, amp=False),
            graph_and_bn_audit(model, audit_value, audit_labels, fold_id, amp=True),
        ])
        current_lambda_rows, current_extra_rows = [], []
        for audit_amp in (False, True):
            lambda_row, extra_rows = equivalence_audits(model, audit_value, audit_labels, fold_id, audit_amp)
            current_lambda_rows.append(lambda_row); current_extra_rows.extend(extra_rows)
        lambda_rows.extend(current_lambda_rows); noop_rows.extend(current_extra_rows)
        base.write_csv(OUT / "INITIALIZATION_AUDIT.csv", pd.DataFrame(init_rows))
        base.write_csv(OUT / "MANIFEST_AUDIT.csv", pd.DataFrame(manifest_rows))
        base.write_csv(OUT / "BN_AND_AUTOGRAD_AUDIT.csv", pd.DataFrame(bn_rows))
        base.write_csv(OUT / "LAMBDA0_EQUIVALENCE.csv", pd.DataFrame(lambda_rows))
        base.write_csv(OUT / "NOOP_AND_RESUME_AUDIT.csv", pd.DataFrame(noop_rows))
        if not all(row["status"] == "PASS" for row in [*bn_rows[-2:], *current_lambda_rows, *current_extra_rows]):
            raise RuntimeError("SC_PROTOCOL_FAIL preflight audit")
        # Recreate initial state and formal RNG after all audits so preflight is a true no-op.
        del model; model, _ = initialize_c1(TASK, fold_id, SEED); model.to(device)
        base.set_seed(SEED + 100_000)
        record, history, gradients = train_one(model, fold, bundle, cache, mean, std, norm_meta, batch_info, device)
        by_fold[fold_id] = record
        all_history = [row for row in all_history if int(row["fold"]) != fold_id] + history
        all_gradients = [row for row in all_gradients if int(row["fold"]) != fold_id] + gradients
        base.write_csv(OUT / "TRAINING_TRAJECTORIES.csv", pd.DataFrame(all_history).sort_values(["fold", "epoch"]))
        base.write_csv(OUT / "GRADIENT_DIAGNOSTICS.csv", pd.DataFrame(all_gradients).sort_values(["fold", "epoch"]))
        serial = [by_fold[index] for index in sorted(by_fold)]
        base.write_json(provenance_path, {"method": METHOD, "seed": SEED, "split_sha256": split_hash, "records": serial})
        del model, cache, bundle, audit_value, audit_labels; gc.collect(); torch.cuda.empty_cache()
    records = [by_fold[index] for index in range(5)]
    base.write_json(PROTOCOL / "C1_WBCIC_CHECKPOINT_LOCK.json", {
        "status": "PASS", "method": METHOD, "seed": SEED, "five_checkpoints_frozen": True,
        "outer_accessed_at_lock": False, "internal_heldout_accessed_at_lock": False, "split_sha256": split_hash,
        "checkpoints": [{"fold": row["fold"], "path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"], "tensor_content_sha256": row["checkpoint_tensor_content_sha256"]} for row in records],
    })
    outer, heldout = evaluate_all(folds, records, references, device)
    finalize(records, outer, heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
