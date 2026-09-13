"""Run the controlled DualCE compute/sampling control (C0) on WBCIC-MI seed 0."""
from __future__ import annotations

import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path(os.environ.get("SC_V2_REPO", "/root/rivermind-data/CRCICLR_SC_CONTROLLED_V2_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_sc_controlled_linux_seed0_v2"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("SC_V2_RUNTIME", "/root/rivermind-data/litebn_sc_controlled_v2_runtime")).resolve()
sys.path.insert(0, str(EXP / "code"))

import run_c1_wbcic as shared


base = shared.base
TASK, METHOD, SEED = shared.TASK, "C0_LiteBN_DualCE", shared.SEED
SECONDARY_SEED = shared.SECONDARY_SEED


def source_hash() -> str:
    paths = [Path(__file__), EXP / "code" / "litebn_sc_controlled.py", shared.ORIGINAL_EXP / "code" / "litebn_x.py"]
    return base.sha256_bytes(b"".join(path.read_bytes() for path in paths))


def checkpoint_path(fold: int, name: str) -> Path:
    return RUNTIME / "checkpoints" / "wbcic_mi" / f"fold{fold}_c0_litebn_dualce" / name


def train_one(
    model: nn.Module, fold: dict[str, Any], bundle: base.SignalBundle, cache: base.RawGPUCache,
    mean: np.ndarray, std: np.ndarray, norm_meta: dict[str, Any], batch_info: dict[str, Any], device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_id = int(fold["fold_id"])
    latest, selected = checkpoint_path(fold_id, "checkpoint_latest.pt"), checkpoint_path(fold_id, "selected_best.pt")
    initial_hash = shared.tensor_content_hash(model.state_dict())
    invariants = {
        "task": TASK, "fold": fold_id, "method": METHOD, "seed": SEED, "lambda_sc": 0.0,
        "secondary_seed": SECONDARY_SEED, "initial_tensor_content_sha256": initial_hash,
        "source_sha256": source_hash(), "normalizer_sha256": norm_meta["mean_std_sha256"],
        "batch_manifest_sha256": batch_info["manifest_sha256"], "bn_persistent_updates_per_batch": 1,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    secondary = shared.initialize_secondary_rng(SECONDARY_SEED)
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
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, base.EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train()
        values: dict[str, list[float]] = {key: [] for key in (
            "ce_a", "ce_b", "ce", "j", "disagree", "prob_l1", "logit_l2", "entropy_a", "entropy_b", "grad", "scale",
        )}
        attempted, successful, skips = 0, 0, 0
        for batch_index, indices in enumerate(batch_info["episodes"][epoch - 1]):
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_a, _, logits_b, _, secondary = shared.controlled_dual_forward(model, value, secondary)
                ce_a, ce_b = F.cross_entropy(logits_a, labels), F.cross_entropy(logits_b, labels)
                ce = 0.5 * (ce_a + ce_b)
            j, _ = shared.symmetric_kl(logits_a, logits_b)
            loss = ce
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite C0 loss fold={fold_id} epoch={epoch}")
            if epoch in shared.GRADIENT_DIAGNOSTIC_EPOCHS and batch_index == 0:
                ce_norm, j_norm, ratio, cosine = shared.gradient_relation(ce, shared.LAMBDA_SC * j, parameters)
                grad_rows.append({
                    "task": TASK, "method": METHOD, "fold": fold_id, "epoch": epoch, "batch_index": batch_index,
                    "ce_gradient_norm": ce_norm, "weighted_J_gradient_norm": j_norm,
                    "weighted_J_to_CE_gradient_norm_ratio": ratio, "gradient_cosine": cosine,
                    "J_used_in_optimizer_loss": False,
                })
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, base.CLIP)
            scale_before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update(); scale_after = float(scaler.get_scale())
            skipped = int(scale_after < scale_before); attempted += 1; successful += 1 - skipped; skips += skipped
            with torch.no_grad():
                pa, pb = F.softmax(logits_a.float(), dim=-1), F.softmax(logits_b.float(), dim=-1)
                values["ce_a"].append(float(ce_a)); values["ce_b"].append(float(ce_b)); values["ce"].append(float(ce))
                values["j"].append(float(j)); values["disagree"].append(float((logits_a.argmax(-1) != logits_b.argmax(-1)).float().mean()))
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
            "median_J": np.median(values["j"]), "mean_total_loss": np.mean(values["ce"]),
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
            print(f"[C0 {TASK} f{fold_id}] epoch={epoch:02d} CE={row['mean_supervised_CE']:.4f} observedJ={row['mean_J']:.5f} valBA={val_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("SC_PROTOCOL_FAIL no eligible C0 checkpoint")
    model.load_state_dict(best_state, strict=True); base.atomic_torch_save(selected, model.state_dict())
    return {
        "task": TASK, "dataset": "WBCIC", "fold": fold_id, "method": METHOD, "seed": SEED,
        "parameter_count": base.parameter_count(model), "initial_tensor_content_sha256": initial_hash,
        "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best),
        "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)),
        "checkpoint_path": str(selected), "checkpoint_sha256": base.sha256_file(selected),
        "checkpoint_tensor_content_sha256": shared.tensor_content_hash(model.state_dict()),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
        "elapsed_seconds_total": elapsed_before + time.perf_counter() - started, "epochs_completed": len(history),
        "attempted_optimizer_steps": attempted_total, "successful_optimizer_steps": successful_total,
        "peak_cuda_memory_bytes": max(peak_before, int(torch.cuda.max_memory_allocated(device))), "source_sha256": source_hash(),
    }, history, grad_rows


def c0_resume_audit(model: nn.Module, value: torch.Tensor, labels: torch.Tensor, fold: int) -> dict[str, Any]:
    continuous, interrupted = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    oc = torch.optim.AdamW(continuous.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    oi = torch.optim.AdamW(interrupted.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    sc = torch.amp.GradScaler("cuda", enabled=True); si = torch.amp.GradScaler("cuda", enabled=True)
    base.set_seed(223_607 + fold); start = base.rng_state(); secondary_start = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    secondary_c = copy.deepcopy(secondary_start)
    for _ in range(10):
        _, secondary_c = shared._one_step(continuous, oc, sc, value, labels, secondary_c, True, 0.0)
    rng_c = base.rng_state()
    base.restore_rng(start); secondary_i = copy.deepcopy(secondary_start)
    for _ in range(5):
        _, secondary_i = shared._one_step(interrupted, oi, si, value, labels, secondary_i, True, 0.0)
    payload = {
        "model": copy.deepcopy(interrupted.state_dict()), "optimizer": copy.deepcopy(oi.state_dict()),
        "scaler": copy.deepcopy(si.state_dict()), "main_rng": base.rng_state(), "secondary_rng": copy.deepcopy(secondary_i),
    }
    resumed = copy.deepcopy(model).train(); optimizer_r = torch.optim.AdamW(resumed.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    scaler_r = torch.amp.GradScaler("cuda", enabled=True)
    resumed.load_state_dict(payload["model"], strict=True); optimizer_r.load_state_dict(payload["optimizer"]); scaler_r.load_state_dict(payload["scaler"])
    base.restore_rng(payload["main_rng"]); secondary_r = payload["secondary_rng"]
    for _ in range(5):
        _, secondary_r = shared._one_step(resumed, optimizer_r, scaler_r, value, labels, secondary_r, True, 0.0)
    rng_r = base.rng_state(); exact, maximum = shared.state_diff(continuous.state_dict(), resumed.state_dict())
    main_exact = shared.torch_rng_equal({"cpu": rng_c["torch"], "cuda": rng_c.get("cuda", [])}, {"cpu": rng_r["torch"], "cuda": rng_r.get("cuda", [])})
    passed = exact and maximum == 0.0 and main_exact and shared.torch_rng_equal(secondary_c, secondary_r) and sc.state_dict() == scaler_r.state_dict()
    return {
        "task": TASK, "method": METHOD, "fold": fold, "precision": "AMP_FP16", "audit": "C0_TEN_STEP_RESUME",
        "model_state_exact": exact, "max_abs_model_state_diff": maximum, "main_rng_exact": main_exact,
        "secondary_rng_exact": shared.torch_rng_equal(secondary_c, secondary_r), "scaler_exact": sc.state_dict() == scaler_r.state_dict(),
        "status": "PASS" if passed else "FAIL",
    }


def load_c0(record: dict[str, Any], fold: int, device: torch.device) -> nn.Module:
    model, _ = shared.initialize_c1(TASK, fold, SEED)
    checkpoint = Path(record["checkpoint_path"])
    if not checkpoint.is_file() or base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError("SC_PROTOCOL_FAIL C0 checkpoint provenance")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if shared.tensor_content_hash(state) != record["checkpoint_tensor_content_sha256"]:
        raise RuntimeError("SC_PROTOCOL_FAIL C0 tensor-content hash")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def comparison_row(frame: pd.DataFrame, cohort: str, left: str, right: str) -> dict[str, Any]:
    pivot = frame.pivot(index="subject_id", columns="method", values="BA")
    delta = (pivot[left] - pivot[right]) * 100
    boot = shared.paired_bootstrap(delta.to_numpy())
    return {
        "task": TASK, "cohort": cohort, "comparison": f"{left.split('_')[0]}-{right.split('_')[0]}",
        "left_method": left, "right_method": right, "left_BA": float(pivot[left].mean()),
        "right_BA": float(pivot[right].mean()), "delta_pp": float(delta.mean()),
        "positive_subjects": boot["positive_subjects"], "negative_subjects": boot["negative_subjects"],
        "tied_subjects": boot["tied_subjects"], "bootstrap_ci_low_pp": boot["ci_low_pp"],
        "bootstrap_ci_high_pp": boot["ci_high_pp"], "resamples": 10_000, "status": "RUN",
    }


def evaluate_and_finalize(
    folds: dict[int, dict[str, Any]], records: list[dict[str, Any]], c1_records: list[dict[str, Any]],
    references: list[dict[str, Any]], device: torch.device,
) -> str:
    outer = pd.read_csv(OUT / "OUTER_SUBJECT_RESULTS.csv")
    outer = outer[outer.method != METHOD]
    c0_rows, diagnostics = [], []
    for fold_id in range(5):
        fold = folds[fold_id]; bundle = shared.build_future_bundle(fold["outer_dev_subjects"]); cache = base.RawGPUCache(bundle, device)
        mean, std, _ = base.load_tensor_pair(shared.ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        model = load_c0(records[fold_id], fold_id, device)
        for subject, metrics in base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std).items():
            c0_rows.append({"task": TASK, "fold": fold_id, "subject_id": subject, "method": METHOD, **metrics})
        validation_bundle = base.build_bundle(TASK, fold["inner_train_subjects"] + fold["inner_val_subjects"])
        validation_cache = base.RawGPUCache(validation_bundle, device)
        indices = validation_bundle.indices(fold["inner_val_subjects"], (int(base.TASKS[TASK]["future_session"]),))[:128]
        value, labels = validation_cache.batch(indices, mean, std)
        diagnostics.append(shared.dropout_diagnostic(model, value, labels, METHOD, fold_id))
        del model, bundle, cache, validation_bundle, validation_cache, value, labels; gc.collect(); torch.cuda.empty_cache()
    outer = pd.concat([outer, pd.DataFrame(c0_rows)], ignore_index=True)
    base.write_csv(OUT / "OUTER_SUBJECT_RESULTS.csv", outer.sort_values(["fold", "subject_id", "method"]))
    fold_rows = []
    selected = {"B0_Exact_LiteBN": references, shared.METHOD: c1_records, METHOD: records}
    for (fold_id, method), q in outer.groupby(["fold", "method"]):
        record = selected[method][int(fold_id)]
        fold_rows.append({
            "task": TASK, "fold": int(fold_id), "method": method, "subjects": len(q), "BA": q.BA.mean(),
            "macro_F1": q.macro_F1.mean(), "accuracy": q.accuracy.mean(), "selected_epoch": record["selected_epoch"],
            "selected_inner_val_BA": record["best_inner_val_BA"],
        })
    fold_frame = pd.DataFrame(fold_rows)
    fp = fold_frame.pivot(index="fold", columns="method", values="BA")
    fold_frame = fold_frame.merge(((fp[shared.METHOD] - fp["B0_Exact_LiteBN"]) * 100).rename("C1_minus_B0_BA_pp"), left_on="fold", right_index=True)
    fold_frame = fold_frame.merge(((fp[METHOD] - fp["B0_Exact_LiteBN"]) * 100).rename("C0_minus_B0_BA_pp"), left_on="fold", right_index=True)
    fold_frame = fold_frame.merge(((fp[shared.METHOD] - fp[METHOD]) * 100).rename("C1_minus_C0_BA_pp"), left_on="fold", right_index=True)
    base.write_csv(OUT / "OUTER_FOLD_RESULTS.csv", fold_frame.sort_values(["fold", "method"]))

    heldout = pd.read_csv(OUT / "HELDOUT_SUBJECT_RESULTS.csv")
    heldout = heldout[heldout.method != METHOD]
    c0_heldout_replicates = []
    manifest = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
    subjects = base.subject_sort(json.loads(manifest.read_text())["WBCIC"]["subject_ids"], "WBCIC")
    for fold_id in range(5):
        bundle = shared.build_future_bundle(subjects); cache = base.RawGPUCache(bundle, device)
        mean, std, _ = base.load_tensor_pair(shared.ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        model = load_c0(records[fold_id], fold_id, device)
        for subject, metrics in base.evaluate(model, bundle, cache, subjects, mean, std).items():
            c0_heldout_replicates.append({"subject_id": subject, "method": METHOD, **metrics})
        del model, bundle, cache; gc.collect(); torch.cuda.empty_cache()
    c0_heldout = pd.DataFrame(c0_heldout_replicates).groupby(["subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    heldout = pd.concat([heldout, c0_heldout], ignore_index=True)
    base.write_csv(OUT / "HELDOUT_SUBJECT_RESULTS.csv", heldout.sort_values(["subject_id", "method"]))

    comparisons = []
    for cohort, frame in (("OUTER", outer), ("INTERNAL_HELDOUT", heldout)):
        comparisons.extend([
            comparison_row(frame, cohort, shared.METHOD, "B0_Exact_LiteBN"),
            comparison_row(frame, cohort, METHOD, "B0_Exact_LiteBN"),
            comparison_row(frame, cohort, shared.METHOD, METHOD),
        ])
    comparison_frame = pd.DataFrame(comparisons)
    base.write_csv(OUT / "PAIRED_COMPARISONS.csv", comparison_frame)
    diag = pd.read_csv(OUT / "DROPOUT_DIAGNOSTICS.csv")
    diag = diag[diag.method != METHOD]
    base.write_csv(OUT / "DROPOUT_DIAGNOSTICS.csv", pd.concat([diag, pd.DataFrame(diagnostics)], ignore_index=True).sort_values(["fold", "method"]))

    lookup = {(row.cohort, row.comparison): row for _, row in comparison_frame.iterrows()}
    o10, o00, o1c0 = lookup[("OUTER", "C1-B0")], lookup[("OUTER", "C0-B0")], lookup[("OUTER", "C1-C0")]
    h10, h00, h1c0 = lookup[("INTERNAL_HELDOUT", "C1-B0")], lookup[("INTERNAL_HELDOUT", "C0-B0")], lookup[("INTERNAL_HELDOUT", "C1-C0")]
    c1_fold_delta = (fp[shared.METHOD] - fp["B0_Exact_LiteBN"]) * 100
    audit_files = ["INITIALIZATION_AUDIT.csv", "MANIFEST_AUDIT.csv", "BN_AND_AUTOGRAD_AUDIT.csv", "LAMBDA0_EQUIVALENCE.csv", "NOOP_AND_RESUME_AUDIT.csv"]
    audits_pass = all((pd.read_csv(OUT / name).status == "PASS").all() for name in audit_files)
    conditions = {
        "C1_minus_B0_outer_ge_0_30_pp": float(o10.delta_pp) >= 0.30,
        "C1_minus_B0_heldout_ge_0_pp": float(h10.delta_pp) >= 0.0,
        "positive_outer_folds_ge_3": int((c1_fold_delta > base.TIE_TOL).sum()) >= 3,
        "worst_outer_fold_gt_minus_1_pp": float(c1_fold_delta.min()) > -1.0,
        "C1_outer_gt_C0_outer": float(o1c0.delta_pp) > 0.0,
        "C1_heldout_ge_C0_heldout": float(h1c0.delta_pp) >= 0.0,
        "all_protocol_and_graph_audits_pass": bool(audits_pass),
    }
    if all(conditions.values()):
        terminal = "SC_WBCIC_PROMISING"
    elif conditions["C1_minus_B0_outer_ge_0_30_pp"] and conditions["C1_minus_B0_heldout_ge_0_pp"] and (
        not conditions["C1_outer_gt_C0_outer"] or not conditions["C1_heldout_ge_C0_heldout"]
    ):
        terminal = "SC_GAIN_NOT_SEPARATED_FROM_DUAL_CE"
    else:
        terminal = "SC_NO_CONTINUATION_SIGNAL"
    decision = {
        "status": terminal, "scope_completed": "B0_C0_C1_WBCIC_MI_SEED0_FIVE_FOLDS_OUTER_AND_INTERNAL_HELDOUT",
        "conditions": conditions, "phase2_executed": False,
        "reason_phase2_not_executed": "WBCIC continuation gate failed" if terminal != "SC_WBCIC_PROMISING" else "Current user scope ended after WBCIC",
        "outer": {"C1-B0": o10.to_dict(), "C0-B0": o00.to_dict(), "C1-C0": o1c0.to_dict()},
        "internal_heldout": {"C1-B0": h10.to_dict(), "C0-B0": h00.to_dict(), "C1-C0": h1c0.to_dict()},
        "new_sealed_test_accessed": False, "seed1_2_run": False,
    }
    base.write_json(OUT / "CONTINUATION_DECISION.json", decision)

    diagnostics_all = pd.read_csv(OUT / "DROPOUT_DIAGNOSTICS.csv")
    diagnostics_mean = diagnostics_all[diagnostics_all.status == "RUN"].groupby("method")[["mean_pairwise_prediction_disagreement", "mean_pairwise_symmetric_KL"]].mean()
    trajectories = pd.read_csv(OUT / "TRAINING_TRAJECTORIES.csv")
    gradients = pd.read_csv(OUT / "GRADIENT_DIAGNOSTICS.csv")
    b0_time = sum(float(row["elapsed_seconds_this_invocation"]) for row in references)
    c0_time = sum(float(row["elapsed_seconds_total"]) for row in records)
    c1_time = sum(float(row["elapsed_seconds_total"]) for row in c1_records)
    lines = [
        "# Controlled LiteBN-SC v2: WBCIC seed-0 report", "",
        "ARCHITECTURE_CHANGE = NO", "INFERENCE_PARAMETER_INCREASE = 0", "INFERENCE_FORWARD_PASSES = 1",
        "SC_REFERENCE = R_DROP_STYLE", "DUAL_CE_CONTROL = YES", "BN_PERSISTENT_UPDATES_PER_BATCH = 1",
        "NEW_SEALED_TEST_ACCESSED = NO", "SEED1_2_RUN = NO", "", "## Primary results", "",
        f"Outer B0/C0/C1 BA = {o10.right_BA:.8f}/{o00.left_BA:.8f}/{o10.left_BA:.8f}",
        f"Outer C1-B0 = {o10.delta_pp:+.4f} pp; C0-B0 = {o00.delta_pp:+.4f} pp; C1-C0 = {o1c0.delta_pp:+.4f} pp",
        f"Heldout B0/C0/C1 BA = {h10.right_BA:.8f}/{h00.left_BA:.8f}/{h10.left_BA:.8f}",
        f"Heldout C1-B0 = {h10.delta_pp:+.4f} pp; C0-B0 = {h00.delta_pp:+.4f} pp; C1-C0 = {h1c0.delta_pp:+.4f} pp",
        "Internal-heldout is DEVELOPMENT_MODEL_SELECTION_DATA, not a sealed final test.", "", "## Required interpretation", "",
        f"1. C1 exceeds exact B0 on outer: {float(o10.delta_pp) > 0}; on heldout: {float(h10.delta_pp) >= 0}.",
        f"2. C1 exceeds matched DualCE C0 on outer: {float(o1c0.delta_pp) > 0}; on heldout: {float(h1c0.delta_pp) >= 0}.",
        f"3. C1-B0 outer/heldout directions are {'the same' if np.sign(o10.delta_pp) == np.sign(h10.delta_pp) else 'opposite'}.",
        f"4. C1-B0 subjects improved/harmed/tied: outer {int(o10.positive_subjects)}/{int(o10.negative_subjects)}/{int(o10.tied_subjects)}; heldout {int(h10.positive_subjects)}/{int(h10.negative_subjects)}/{int(h10.tied_subjects)}.",
        f"5. Mean selected-checkpoint stochastic pairwise KL B0/C0/C1 = {diagnostics_mean.loc['B0_Exact_LiteBN','mean_pairwise_symmetric_KL']:.6f}/{diagnostics_mean.loc[METHOD,'mean_pairwise_symmetric_KL']:.6f}/{diagnostics_mean.loc[shared.METHOD,'mean_pairwise_symmetric_KL']:.6f}.",
        f"6. C1 is more consistent than C0 while lower in BA: outer={bool(diagnostics_mean.loc[shared.METHOD,'mean_pairwise_symmetric_KL'] < diagnostics_mean.loc[METHOD,'mean_pairwise_symmetric_KL'] and o1c0.delta_pp < 0)}; heldout={bool(diagnostics_mean.loc[shared.METHOD,'mean_pairwise_symmetric_KL'] < diagnostics_mean.loc[METHOD,'mean_pairwise_symmetric_KL'] and h1c0.delta_pp < 0)}.",
        f"7. Mean selected inner-val BA B0/C0/C1 = {np.mean([r['best_inner_val_BA'] for r in references]):.6f}/{np.mean([r['best_inner_val_BA'] for r in records]):.6f}/{np.mean([r['best_inner_val_BA'] for r in c1_records]):.6f}.",
        f"8. Median observational ||grad(0.5J)||/||grad(CE)|| C0/C1 = {gradients[gradients.method == METHOD].weighted_J_to_CE_gradient_norm_ratio.median():.6f}/{gradients[gradients.method == shared.METHOD].weighted_J_to_CE_gradient_norm_ratio.median():.6f}.",
        f"9. Persistent BN state updated once per batch; all audits passed: {audits_pass}.",
        f"10. Recorded wall time B0/C0/C1 = {b0_time:.1f}/{c0_time:.1f}/{c1_time:.1f} s; these historical/sequential measurements are not a controlled timing benchmark. Peak allocated CUDA C0/C1 = {max(r['peak_cuda_memory_bytes'] for r in records)/2**30:.3f}/{max(r['peak_cuda_memory_bytes'] for r in c1_records)/2**30:.3f} GiB; B0 peak was not recorded.",
        "11. All methods use the exact historical LiteBN and one deterministic inference forward.",
        f"12. Phase2 executed: NO. {decision['reason_phase2_not_executed']}.", "", terminal,
    ]
    base.write_text(OUT / "FINAL_REPORT.md", "\n".join(lines))
    ledger = pd.DataFrame([
        {"item": "B0 exact Linux checkpoint replay", "status": "RUN", "evidence": "zero-tolerance replay recorded by C1 stage"},
        {"item": "C0 WBCIC seed0 five-fold training", "status": "RUN", "evidence": "C0_CHECKPOINT_PROVENANCE.json and trajectories"},
        {"item": "C1 WBCIC seed0 five-fold training", "status": "RUN", "evidence": "C1_CHECKPOINT_PROVENANCE.json and trajectories"},
        {"item": "B0/C0/C1 outer comparisons", "status": "RUN", "evidence": "OUTER_SUBJECT_RESULTS.csv and PAIRED_COMPARISONS.csv"},
        {"item": "B0/C0/C1 internal-heldout comparisons", "status": "RUN", "evidence": "DEVELOPMENT_MODEL_SELECTION_DATA"},
        {"item": "Phase2 OpenBMI tasks", "status": "NOT_RUN_GATE_FAIL" if terminal != "SC_WBCIC_PROMISING" else "NOT_RUN_CURRENT_SCOPE", "evidence": decision["reason_phase2_not_executed"]},
        {"item": "Old SC v1", "status": "PROTOCOL_MISMATCH_PAUSED", "evidence": "not reused or renamed"},
        {"item": "New sealed final test", "status": "NOT_ACCESSED", "evidence": "protocol lock"},
    ])
    base.write_csv(OUT / "EXPERIMENT_EVIDENCE_LEDGER.csv", ledger)
    print(terminal, flush=True)
    return terminal


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("SC_PROTOCOL_FAIL CUDA unavailable")
    device = torch.device("cuda")
    c1_provenance = json.loads((PROTOCOL / "C1_CHECKPOINT_PROVENANCE.json").read_text())
    c1_records = sorted(c1_provenance["records"], key=lambda row: int(row["fold"]))
    reference_provenance = json.loads((REFERENCE_EXP := shared.REFERENCE_EXP) .joinpath("protocol", "LITEBN_CHECKPOINT_PROVENANCE.json").read_text())
    references = sorted(reference_provenance["records"], key=lambda row: int(row["fold"]))
    _, by_dataset, split_hash = base.load_folds(); folds = {int(row["fold_id"]): row for row in by_dataset["WBCIC"]}
    provenance_path = PROTOCOL / "C0_CHECKPOINT_PROVENANCE.json"
    existing = json.loads(provenance_path.read_text())["records"] if provenance_path.is_file() else []
    by_fold = {int(row["fold"]): row for row in existing}
    init_existing = pd.read_csv(OUT / "INITIALIZATION_AUDIT.csv"); init_existing["method"] = init_existing.get("method", shared.METHOD)
    init_rows = init_existing[init_existing.method != METHOD].to_dict("records")
    manifest_existing = pd.read_csv(OUT / "MANIFEST_AUDIT.csv"); manifest_existing["method"] = manifest_existing.get("method", shared.METHOD)
    manifest_rows = manifest_existing[manifest_existing.method != METHOD].to_dict("records")
    bn_existing = pd.read_csv(OUT / "BN_AND_AUTOGRAD_AUDIT.csv"); bn_existing["method"] = bn_existing.get("method", shared.METHOD)
    bn_rows = bn_existing[bn_existing.method != METHOD].to_dict("records")
    resume_existing = pd.read_csv(OUT / "NOOP_AND_RESUME_AUDIT.csv"); resume_existing["method"] = resume_existing.get("method", shared.METHOD)
    resume_rows = resume_existing[resume_existing.method != METHOD].to_dict("records")
    trajectories = pd.read_csv(OUT / "TRAINING_TRAJECTORIES.csv"); trajectory_rows = trajectories[trajectories.method != METHOD].to_dict("records")
    gradients = pd.read_csv(OUT / "GRADIENT_DIAGNOSTICS.csv"); gradient_rows = gradients[gradients.method != METHOD].to_dict("records")
    for fold_id in range(5):
        fold = folds[fold_id]; allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
        bundle = base.build_bundle(TASK, allowed); cache = base.RawGPUCache(bundle, device)
        mean, std, norm_meta, batch_info, manifest = shared.load_inputs(fold); manifest["method"] = METHOD; manifest_rows.append(manifest)
        model, initialization = shared.initialize_c1(TASK, fold_id, SEED); initialization["method"] = METHOD; model.to(device); init_rows.append(initialization)
        value, labels = cache.batch(np.asarray(batch_info["episodes"][0][0], dtype=np.int64), mean, std)
        current_bn = [shared.graph_and_bn_audit(model, value, labels, fold_id, False), shared.graph_and_bn_audit(model, value, labels, fold_id, True)]
        for row in current_bn:
            row["method"] = METHOD
        resume = c0_resume_audit(model, value, labels, fold_id)
        bn_rows.extend(current_bn); resume_rows.append(resume)
        base.write_csv(OUT / "INITIALIZATION_AUDIT.csv", pd.DataFrame(init_rows))
        base.write_csv(OUT / "MANIFEST_AUDIT.csv", pd.DataFrame(manifest_rows))
        base.write_csv(OUT / "BN_AND_AUTOGRAD_AUDIT.csv", pd.DataFrame(bn_rows))
        base.write_csv(OUT / "NOOP_AND_RESUME_AUDIT.csv", pd.DataFrame(resume_rows))
        if not all(row["status"] == "PASS" for row in [*current_bn, resume]):
            raise RuntimeError("SC_PROTOCOL_FAIL C0 preflight")
        del model; model, _ = shared.initialize_c1(TASK, fold_id, SEED); model.to(device); base.set_seed(SEED + 100_000)
        record, history, grad = train_one(model, fold, bundle, cache, mean, std, norm_meta, batch_info, device)
        by_fold[fold_id] = record
        trajectory_rows = [row for row in trajectory_rows if not (row["method"] == METHOD and int(row["fold"]) == fold_id)] + history
        gradient_rows = [row for row in gradient_rows if not (row["method"] == METHOD and int(row["fold"]) == fold_id)] + grad
        base.write_csv(OUT / "TRAINING_TRAJECTORIES.csv", pd.DataFrame(trajectory_rows).sort_values(["method", "fold", "epoch"]))
        base.write_csv(OUT / "GRADIENT_DIAGNOSTICS.csv", pd.DataFrame(gradient_rows).sort_values(["method", "fold", "epoch"]))
        serial = [by_fold[index] for index in sorted(by_fold)]
        base.write_json(provenance_path, {"method": METHOD, "seed": SEED, "split_sha256": split_hash, "records": serial})
        del model, bundle, cache, value, labels; gc.collect(); torch.cuda.empty_cache()
    records = [by_fold[index] for index in range(5)]
    base.write_json(PROTOCOL / "C0_WBCIC_CHECKPOINT_LOCK.json", {
        "status": "PASS", "method": METHOD, "seed": SEED, "five_checkpoints_frozen": True,
        "outer_accessed_at_lock": False, "internal_heldout_accessed_at_lock": False, "split_sha256": split_hash,
        "checkpoints": [{"fold": row["fold"], "path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"], "tensor_content_sha256": row["checkpoint_tensor_content_sha256"]} for row in records],
    })
    evaluate_and_finalize(folds, records, c1_records, references, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
