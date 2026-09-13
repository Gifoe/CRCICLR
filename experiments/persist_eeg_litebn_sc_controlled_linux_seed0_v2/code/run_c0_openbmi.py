"""Post-gate exploratory C0 runs for the three remaining OpenBMI tasks."""
from __future__ import annotations

import copy
import gc
import json
import os
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
sys.path.insert(0, str(EXP / "code"))

import run_c1_wbcic as shared


base = shared.base
METHOD, SEED, SECONDARY_SEED = "C0_LiteBN_DualCE", 0, shared.SECONDARY_SEED
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP")
SCOPE = "POST_GATE_EXPLORATORY_C0"


def source_hash() -> str:
    paths = [
        Path(__file__),
        EXP / "code" / "litebn_sc_controlled.py",
        EXP / "code" / "run_c1_wbcic.py",
        shared.ORIGINAL_EXP / "code" / "litebn_x.py",
    ]
    return base.sha256_bytes(b"".join(path.read_bytes() for path in paths))


def checkpoint_path(task: str, fold: int, name: str) -> Path:
    return RUNTIME / "post_gate_c0_openbmi" / task.lower() / f"fold{fold}_c0_litebn_dualce" / name


def reference_records() -> dict[tuple[str, int], dict[str, Any]]:
    source = shared.ORIGINAL_EXP / "protocol" / "CHECKPOINT_PROVENANCE.json"
    rows = json.loads(source.read_text())["records"]
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("architecture") != "LiteBN_BASELINE" or row.get("task") not in TASKS:
            continue
        key = (str(row["task"]), int(row["fold"]))
        if key in result:
            old = result[key]
            if old["checkpoint_sha256"] != row["checkpoint_sha256"] or old["normalizer_sha256"] != row["normalizer_sha256"]:
                raise RuntimeError(f"conflicting B0 provenance: {key}")
        else:
            result[key] = row
    if len(result) != 15:
        raise RuntimeError(f"expected 15 unique OpenBMI B0 references, got {len(result)}")
    for key, row in result.items():
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file() or base.sha256_file(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"invalid B0 checkpoint: {key}")
    return result


def load_inputs(task: str, fold: dict[str, Any], bundle: base.SignalBundle, reference: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], torch.Tensor | None, dict[str, Any], dict[str, Any]]:
    fold_id = int(fold["fold_id"])
    normalizer_path = shared.ORIGINAL_RUNTIME / "normalizers" / f"{task.lower()}_fold{fold_id}.npz"
    mean, std, normalizer_meta = base.load_tensor_pair(normalizer_path)
    rebuilt_mean, rebuilt_std, rebuilt_normalizer_meta = base.normalizer(bundle, fold["inner_train_subjects"])
    canonical_normalizer_match = (
        np.array_equal(mean, rebuilt_mean)
        and np.array_equal(std, rebuilt_std)
        and normalizer_meta["mean_std_sha256"] == rebuilt_normalizer_meta["mean_std_sha256"]
    )
    reference_normalizer_sha = str(reference["normalizer_sha256"])
    reference_has_normalizer_hash = len(reference_normalizer_sha) == 64
    normalizer_match = canonical_normalizer_match and (
        not reference_has_normalizer_hash or normalizer_meta["mean_std_sha256"] == reference_normalizer_sha
    )
    if base.TASKS[task]["mi_protocol"]:
        manifest_path = shared.ORIGINAL_RUNTIME / "episode_manifests" / f"{task.lower()}_fold{fold_id}.json"
        manifest = json.loads(manifest_path.read_text())
        episodes = [[np.asarray(indices, dtype=np.int64) for indices in epoch] for epoch in manifest["epochs"]]
        manifest_hash = base.sha256_file(manifest_path)
        batch_info = {"episodes": episodes, "manifest_sha256": manifest_hash, "kind": "frozen_MI_episode_manifest"}
        prior_runtime = base.RUNTIME
        base.RUNTIME = RUNTIME / "audit_reconstructed_manifests"
        try:
            rebuilt_episodes, rebuilt_manifest_meta = base.mi_manifest(bundle, fold, task)
        finally:
            base.RUNTIME = prior_runtime
        canonical_manifest_match = (
            manifest_hash == rebuilt_manifest_meta["manifest_sha256"]
            and len(episodes) == len(rebuilt_episodes)
            and all(
                len(left) == len(right)
                and all(np.array_equal(a, b) for a, b in zip(left, right))
                for left, right in zip(episodes, rebuilt_episodes)
            )
        )
        reference_manifest_sha = reference.get("batch_manifest_sha256")
        manifest_match = canonical_manifest_match and (
            reference_manifest_sha is None or manifest_hash == str(reference_manifest_sha)
        )
        updates = int(sum(len(epoch) for epoch in episodes))
    else:
        manifest_hash = None
        canonical_manifest_match = True
        manifest_match = reference.get("batch_manifest_sha256") is None
        batch_info = {"episodes": None, "manifest_sha256": None, "kind": "historical_task_full_permutation_batch64"}
        train_indices = bundle.indices(fold["inner_train_subjects"], base.TASKS[task]["source_sessions"])
        updates = int(sum(len(base.task_epoch_batches(train_indices, task, fold_id, epoch)) for epoch in range(1, base.EPOCHS + 1)))
    weight, weight_meta = base.class_weights(bundle, fold["inner_train_subjects"])
    class_weight_match = bool(weight_meta["weighted_cross_entropy"]) == bool(reference["class_weighted_ce"])
    row = {
        "scope": SCOPE, "task": task, "fold": fold_id, "method": METHOD,
        "normalizer_sha256": normalizer_meta["mean_std_sha256"], "reference_normalizer_sha256": reference["normalizer_sha256"],
        "normalizer_match": normalizer_match,
        "normalizer_reference_mode": "HASHED_B0_PROVENANCE" if reference_has_normalizer_hash else "EXACT_CANONICAL_RECONSTRUCTION",
        "canonical_normalizer_reconstruction_match": canonical_normalizer_match,
        "batch_manifest_sha256": manifest_hash,
        "reference_batch_manifest_sha256": reference.get("batch_manifest_sha256"), "manifest_match": manifest_match,
        "batch_manifest_reference_mode": "HASHED_B0_PROVENANCE" if reference.get("batch_manifest_sha256") else "EXACT_CANONICAL_RECONSTRUCTION",
        "canonical_batch_manifest_reconstruction_match": canonical_manifest_match,
        "class_weighted_ce": bool(weight_meta["weighted_cross_entropy"]), "reference_class_weighted_ce": bool(reference["class_weighted_ce"]),
        "class_weight_match": class_weight_match, "epochs": base.EPOCHS, "optimizer_updates": updates,
        "outer_subjects_absent_from_fit": not bool(set(fold["outer_dev_subjects"]) & set(fold["inner_train_subjects"] + fold["inner_val_subjects"])),
    }
    row["status"] = "PASS" if all((normalizer_match, manifest_match, class_weight_match, row["outer_subjects_absent_from_fit"])) else "FAIL"
    if row["status"] != "PASS":
        raise RuntimeError(f"SC_PROTOCOL_FAIL OpenBMI C0 manifest: {row}")
    return mean, std, normalizer_meta, batch_info, weight, weight_meta, row


def task_step(
    model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
    value: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None,
    secondary: dict[str, Any], amp: bool, add_zero_j: bool = False,
) -> tuple[float, dict[str, Any]]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=value.device.type, dtype=torch.float16, enabled=amp):
        logits_a, _, logits_b, _, secondary = shared.controlled_dual_forward(model, value, secondary)
        ce = 0.5 * (F.cross_entropy(logits_a, labels, weight=weight) + F.cross_entropy(logits_b, labels, weight=weight))
    j, _ = shared.symmetric_kl(logits_a, logits_b)
    loss = ce + 0.0 * j if add_zero_j else ce
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
    scaler.step(optimizer); scaler.update()
    return float(loss.detach()), secondary


def preflight_audit(task: str, fold: int, model: nn.Module, value: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None, amp: bool) -> dict[str, Any]:
    precision = "AMP_FP16" if amp else "FP32"
    baseline, dual = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    base.set_seed(410_000 + fold); a_start = shared.torch_rng_state(); secondary = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits_b0, _ = baseline(value)
    b0_bn, b0_rng = shared.bn_snapshot(baseline), shared.torch_rng_state()
    logits_b0 = logits_b0.detach().clone(); del baseline; gc.collect(); torch.cuda.empty_cache()
    shared.restore_torch_rng(a_start)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        logits_a, _, logits_b, _, _ = shared.controlled_dual_forward(dual, value, secondary)
    dual_bn, dual_rng = shared.bn_snapshot(dual), shared.torch_rng_state()
    bn_exact, bn_max = shared.state_diff(b0_bn, dual_bn)
    forward_a_exact, dropout_diff = torch.equal(logits_b0, logits_a), not torch.equal(logits_a, logits_b)
    main_rng_exact = shared.torch_rng_equal(b0_rng, dual_rng)
    del dual, logits_a, logits_b, logits_b0; gc.collect(); torch.cuda.empty_cache()

    graph = copy.deepcopy(model).train(); optimizer = torch.optim.AdamW(graph.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=amp); base.set_seed(410_000 + fold); secondary_graph = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    before = shared.bn_snapshot(graph)
    _, secondary_graph = task_step(graph, optimizer, scaler, value, labels, weight, secondary_graph, amp)
    after = shared.bn_snapshot(graph)
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in graph.parameters())
    gradients_nonzero = any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in graph.parameters())
    # The one optimizer step must not alter running buffers after the forward-A update.
    backward_safe = all(torch.isfinite(value_).all() for value_ in after.values()) and any(not torch.equal(before[key], after[key]) for key in before)
    del graph; gc.collect(); torch.cuda.empty_cache()

    c0, lambda0 = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    optimizer0 = torch.optim.AdamW(c0.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    optimizer1 = torch.optim.AdamW(lambda0.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    scaler0 = torch.amp.GradScaler("cuda", enabled=amp); scaler1 = torch.amp.GradScaler("cuda", enabled=amp)
    base.set_seed(420_000 + fold); main = base.rng_state(); secondary0 = shared.initialize_secondary_rng(SECONDARY_SEED + fold)
    loss0, secondary_end0 = task_step(c0, optimizer0, scaler0, value, labels, weight, copy.deepcopy(secondary0), amp, False)
    end0 = base.rng_state(); base.restore_rng(main)
    loss1, secondary_end1 = task_step(lambda0, optimizer1, scaler1, value, labels, weight, copy.deepcopy(secondary0), amp, True)
    end1 = base.rng_state(); state_exact, state_max = shared.state_diff(c0.state_dict(), lambda0.state_dict())
    lambda0_exact = abs(loss0 - loss1) <= 1e-12 and state_exact and state_max == 0.0 and shared.torch_rng_equal(secondary_end0, secondary_end1) and shared.torch_rng_equal(
        {"cpu": end0["torch"], "cuda": end0.get("cuda", [])}, {"cpu": end1["torch"], "cuda": end1.get("cuda", [])}
    )
    passed = all((bn_exact, bn_max == 0.0, forward_a_exact, dropout_diff, main_rng_exact, bool(gradients_finite), bool(gradients_nonzero), backward_safe, lambda0_exact))
    return {
        "scope": SCOPE, "task": task, "fold": fold, "method": METHOD, "precision": precision,
        "bn_persistent_exact": bn_exact, "max_abs_bn_diff": bn_max, "forward_a_matches_b0": forward_a_exact,
        "dropout_logits_different": dropout_diff, "main_rng_matches_b0_after_a": main_rng_exact,
        "actual_task_ce_gradients_finite": bool(gradients_finite), "actual_task_ce_gradients_nonzero": bool(gradients_nonzero),
        "one_persistent_bn_update": backward_safe, "lambda0_equivalence_exact": lambda0_exact,
        "lambda0_max_abs_state_diff": state_max, "uses_live_buffer_copy_before_backward": False,
        "status": "PASS" if passed else "FAIL",
    }


def train_one(
    task: str, model: nn.Module, fold: dict[str, Any], bundle: base.SignalBundle, cache: base.RawGPUCache,
    mean: np.ndarray, std: np.ndarray, normalizer_meta: dict[str, Any], batch_info: dict[str, Any],
    weight: torch.Tensor | None, weight_meta: dict[str, Any], device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_id = int(fold["fold_id"]); latest = checkpoint_path(task, fold_id, "checkpoint_latest.pt"); selected = checkpoint_path(task, fold_id, "selected_best.pt")
    initial_hash = shared.tensor_content_hash(model.state_dict())
    invariants = {
        "scope": SCOPE, "task": task, "fold": fold_id, "method": METHOD, "seed": SEED,
        "secondary_seed": SECONDARY_SEED, "initial_tensor_content_sha256": initial_hash, "source_sha256": source_hash(),
        "normalizer_sha256": normalizer_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
        "class_weight_info": weight_meta, "bn_persistent_updates_per_batch": 1,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True); secondary = shared.initialize_secondary_rng(SECONDARY_SEED)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    elapsed_before, peak_before = 0.0, 0
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
        base.restore_rng(saved["main_rng"]); secondary = saved["secondary_rng"]
        start, history = int(saved["epoch"]) + 1, list(saved["history"])
        best, best_epoch, best_state = float(saved["best"]), saved["best_epoch"], saved["best_state"]
        elapsed_before = float(saved.get("elapsed_seconds_completed", 0.0)); peak_before = int(saved.get("peak_cuda_memory_bytes", 0))
    train_indices = bundle.indices(fold["inner_train_subjects"], base.TASKS[task]["source_sessions"])
    weight_device = None if weight is None else weight.to(device)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, base.EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); metrics = {key: [] for key in ("cea", "ceb", "ce", "j", "disagree", "grad", "scale")}
        skipped = 0
        batches: Iterable[np.ndarray] = batch_info["episodes"][epoch - 1] if base.TASKS[task]["mi_protocol"] else base.task_epoch_batches(train_indices, task, fold_id, epoch)
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_a, _, logits_b, _, secondary = shared.controlled_dual_forward(model, value, secondary)
                ce_a, ce_b = F.cross_entropy(logits_a, labels, weight=weight_device), F.cross_entropy(logits_b, labels, weight=weight_device)
                ce = 0.5 * (ce_a + ce_b)
            j, _ = shared.symmetric_kl(logits_a, logits_b)
            if not torch.isfinite(ce):
                raise RuntimeError(f"non-finite C0 loss {task}/f{fold_id}")
            scaler.scale(ce).backward(); scaler.unscale_(optimizer); grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
            scale_before = float(scaler.get_scale()); scaler.step(optimizer); scaler.update(); skipped += int(float(scaler.get_scale()) < scale_before)
            metrics["cea"].append(float(ce_a)); metrics["ceb"].append(float(ce_b)); metrics["ce"].append(float(ce)); metrics["j"].append(float(j))
            metrics["disagree"].append(float((logits_a.argmax(-1) != logits_b.argmax(-1)).float().mean())); metrics["grad"].append(float(grad_norm)); metrics["scale"].append(scale_before)
        validation = base.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        val_ba = float(np.mean([row["BA"] for row in validation.values()])); val_f1 = float(np.mean([row["macro_F1"] for row in validation.values()]))
        chose = epoch >= base.MIN_EPOCH and val_ba > best + base.TIE_TOL
        if chose:
            best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        row = {
            "scope": SCOPE, "task": task, "method": METHOD, "fold": fold_id, "epoch": epoch,
            "mean_CE_A": np.mean(metrics["cea"]), "mean_CE_B": np.mean(metrics["ceb"]), "mean_supervised_CE": np.mean(metrics["ce"]),
            "observed_mean_J_not_in_loss": np.mean(metrics["j"]), "mean_total_loss": np.mean(metrics["ce"]),
            "train_mode_prediction_disagreement": np.mean(metrics["disagree"]), "mean_preclip_gradient_norm": np.mean(metrics["grad"]),
            "mean_gradscaler_scale": np.mean(metrics["scale"]), "attempted_optimizer_steps": len(metrics["ce"]),
            "successful_optimizer_steps": len(metrics["ce"]) - skipped, "amp_skips": skipped,
            "inner_val_subject_BA": val_ba, "inner_val_subject_macro_F1": val_f1, "selected": bool(chose),
            "epoch_runtime_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        base.atomic_torch_save(latest, {
            "epoch": epoch, "history": history, "best": best, "best_epoch": best_epoch, "best_state": best_state,
            "current_state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "main_rng": base.rng_state(), "secondary_rng": secondary, "invariants": invariants,
            "elapsed_seconds_completed": elapsed_before + time.perf_counter() - started,
            "peak_cuda_memory_bytes": max(peak_before, int(torch.cuda.max_memory_allocated(device))),
        })
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[POST-GATE C0 {task} f{fold_id}] epoch={epoch:02d} CE={row['mean_supervised_CE']:.4f} observedJ={row['observed_mean_J_not_in_loss']:.5f} valBA={val_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"no selected C0 checkpoint {task}/f{fold_id}")
    model.load_state_dict(best_state, strict=True); base.atomic_torch_save(selected, model.state_dict())
    return {
        "scope": SCOPE, "task": task, "dataset": "OpenBMI", "fold": fold_id, "method": METHOD, "seed": SEED,
        "parameter_count": base.parameter_count(model), "initial_tensor_content_sha256": initial_hash,
        "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best),
        "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)),
        "checkpoint_path": str(selected), "checkpoint_sha256": base.sha256_file(selected),
        "checkpoint_tensor_content_sha256": shared.tensor_content_hash(model.state_dict()),
        "normalizer_sha256": normalizer_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
        "class_weighted_ce": bool(weight_meta["weighted_cross_entropy"]), "epochs_completed": len(history),
        "elapsed_seconds_total": elapsed_before + time.perf_counter() - started,
        "peak_cuda_memory_bytes": max(peak_before, int(torch.cuda.max_memory_allocated(device))), "source_sha256": source_hash(),
    }, history


def load_model(task: str, record: dict[str, Any], device: torch.device) -> nn.Module:
    checkpoint = Path(record["checkpoint_path"])
    if not checkpoint.is_file() or base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint provenance failure: {checkpoint}")
    model, _ = shared.initialize_c1(task, int(record["fold"]), SEED)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def paired_summary(frame: pd.DataFrame, task: str, cohort: str) -> dict[str, Any]:
    pivot = frame.pivot(index="subject_id", columns="method", values="BA")
    delta = (pivot[METHOD] - pivot["B0_Exact_LiteBN"]) * 100
    boot = shared.paired_bootstrap(delta.to_numpy())
    return {
        "scope": SCOPE, "task": task, "cohort": cohort, "B0_BA": float(pivot["B0_Exact_LiteBN"].mean()),
        "C0_BA": float(pivot[METHOD].mean()), "C0_minus_B0_pp": float(delta.mean()),
        "positive_subjects": boot["positive_subjects"], "negative_subjects": boot["negative_subjects"],
        "tied_subjects": boot["tied_subjects"], "bootstrap_ci_low_pp": boot["ci_low_pp"],
        "bootstrap_ci_high_pp": boot["ci_high_pp"], "resamples": 10_000,
    }


def evaluate_all(folds: dict[int, dict[str, Any]], records: dict[tuple[str, int], dict[str, Any]], references: dict[tuple[str, int], dict[str, Any]], device: torch.device) -> None:
    frozen = pd.read_csv(shared.ORIGINAL_EXP / "outputs" / "OUTER_SUBJECT_RESULTS.csv")
    frozen = frozen[(frozen.method == "LiteBN_BASELINE") & (frozen.task.isin(TASKS))]
    outer_rows, fold_rows, summaries, heldout_rows, heldout_summaries = [], [], [], [], []
    holdout_manifest = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
    heldout_subjects = base.subject_sort(json.loads(holdout_manifest.read_text())["OpenBMI"]["subject_ids"], "OpenBMI")
    for task in TASKS:
        for fold_id in range(5):
            fold = folds[fold_id]; mean, std, _ = base.load_tensor_pair(shared.ORIGINAL_RUNTIME / "normalizers" / f"{task.lower()}_fold{fold_id}.npz")
            bundle = base.build_bundle(task, fold["outer_dev_subjects"]); cache = base.RawGPUCache(bundle, device)
            models = {"B0_Exact_LiteBN": load_model(task, references[(task, fold_id)], device), METHOD: load_model(task, records[(task, fold_id)], device)}
            for method, model in models.items():
                for subject, metrics in base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std).items():
                    outer_rows.append({"scope": SCOPE, "task": task, "fold": fold_id, "subject_id": subject, "method": method, **metrics})
            del models, bundle, cache; gc.collect(); torch.cuda.empty_cache()
        task_outer = pd.DataFrame([row for row in outer_rows if row["task"] == task])
        replay = task_outer[task_outer.method == "B0_Exact_LiteBN"].sort_values(["fold", "subject_id"]).reset_index(drop=True)
        expected = frozen[frozen.task == task].sort_values(["fold", "subject_id"]).reset_index(drop=True)
        if list(zip(replay.fold, replay.subject_id)) != list(zip(expected.fold, expected.subject_id)):
            raise RuntimeError(f"B0 replay key mismatch: {task}")
        replay_diff = float(np.max(np.abs(replay.BA.to_numpy() - expected.BA.to_numpy())))
        if replay_diff > 1e-12:
            raise RuntimeError(f"B0 replay metric mismatch {task}: {replay_diff}")
        summary = paired_summary(task_outer, task, "OUTER"); summary["B0_replay_max_abs_BA_diff"] = replay_diff; summaries.append(summary)
        for (fold_id, method), q in task_outer.groupby(["fold", "method"]):
            fold_rows.append({"scope": SCOPE, "task": task, "fold": fold_id, "method": method, "subjects": len(q), "BA": q.BA.mean(), "macro_F1": q.macro_F1.mean(), "accuracy": q.accuracy.mean()})

        task_heldout_replicates = []
        for fold_id in range(5):
            mean, std, _ = base.load_tensor_pair(shared.ORIGINAL_RUNTIME / "normalizers" / f"{task.lower()}_fold{fold_id}.npz")
            bundle = base.build_bundle(task, heldout_subjects); cache = base.RawGPUCache(bundle, device)
            models = {"B0_Exact_LiteBN": load_model(task, references[(task, fold_id)], device), METHOD: load_model(task, records[(task, fold_id)], device)}
            for method, model in models.items():
                for subject, metrics in base.evaluate(model, bundle, cache, heldout_subjects, mean, std).items():
                    task_heldout_replicates.append({"scope": SCOPE, "data_status": "DEVELOPMENT_MODEL_SELECTION_DATA", "task": task, "fold": fold_id, "subject_id": subject, "method": method, **metrics})
            del models, bundle, cache; gc.collect(); torch.cuda.empty_cache()
        aggregated = pd.DataFrame(task_heldout_replicates).groupby(["scope", "data_status", "task", "subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
        heldout_rows.extend(aggregated.to_dict("records")); heldout_summaries.append(paired_summary(aggregated, task, "INTERNAL_HELDOUT"))
    base.write_csv(OUT / "POST_GATE_C0_OPENBMI_OUTER_SUBJECT_RESULTS.csv", pd.DataFrame(outer_rows).sort_values(["task", "fold", "subject_id", "method"]))
    base.write_csv(OUT / "POST_GATE_C0_OPENBMI_OUTER_FOLD_RESULTS.csv", pd.DataFrame(fold_rows).sort_values(["task", "fold", "method"]))
    base.write_csv(OUT / "POST_GATE_C0_OPENBMI_OUTER_TASK_SUMMARY.csv", pd.DataFrame(summaries))
    base.write_csv(OUT / "POST_GATE_C0_OPENBMI_INTERNAL_HELDOUT_SUBJECT_RESULTS.csv", pd.DataFrame(heldout_rows).sort_values(["task", "subject_id", "method"]))
    base.write_csv(OUT / "POST_GATE_C0_OPENBMI_INTERNAL_HELDOUT_SUMMARY.csv", pd.DataFrame(heldout_summaries))
    lines = [
        "# Post-gate exploratory C0 OpenBMI seed-0 results", "",
        "STATUS = POST_GATE_EXPLORATORY_C0", "WBCIC_GATE_WAS_PASSED = NO", "ARCHITECTURE_CHANGE = NO",
        "INFERENCE_PARAMETER_INCREASE = 0", "INFERENCE_FORWARD_PASSES = 1", "SEED = 0",
        "NEW_SEALED_TEST_ACCESSED = NO", "INTERNAL_HELDOUT_STATUS = DEVELOPMENT_MODEL_SELECTION_DATA", "",
        "These runs were explicitly requested after the locked WBCIC gate failed. They do not retroactively authorize Phase 2 or alter the WBCIC decision.", "",
        "| Task | Outer B0 | Outer C0 | Delta pp | Heldout B0 | Heldout C0 | Delta pp |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_outer = {row["task"]: row for row in summaries}; by_heldout = {row["task"]: row for row in heldout_summaries}
    for task in TASKS:
        outer, heldout = by_outer[task], by_heldout[task]
        lines.append(f"| {task} | {outer['B0_BA']:.6f} | {outer['C0_BA']:.6f} | {outer['C0_minus_B0_pp']:+.3f} | {heldout['B0_BA']:.6f} | {heldout['C0_BA']:.6f} | {heldout['C0_minus_B0_pp']:+.3f} |")
    base.write_text(OUT / "POST_GATE_C0_OPENBMI_REPORT.md", "\n".join(lines))


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("SC_PROTOCOL_FAIL CUDA unavailable")
    device = torch.device("cuda"); references = reference_records(); _, datasets, split_hash = base.load_folds()
    folds = {int(row["fold_id"]): row for row in datasets["OpenBMI"]}
    provenance_path = PROTOCOL / "POST_GATE_C0_OPENBMI_CHECKPOINT_PROVENANCE.json"
    existing = json.loads(provenance_path.read_text())["records"] if provenance_path.is_file() else []
    records = {(row["task"], int(row["fold"])): row for row in existing}
    audit_path, manifest_path, init_path, trajectory_path = (
        OUT / "POST_GATE_C0_OPENBMI_AUDIT.csv", OUT / "POST_GATE_C0_OPENBMI_MANIFEST_AUDIT.csv",
        OUT / "POST_GATE_C0_OPENBMI_INITIALIZATION_AUDIT.csv", OUT / "POST_GATE_C0_OPENBMI_TRAINING_TRAJECTORIES.csv",
    )
    audit_rows = pd.read_csv(audit_path).to_dict("records") if audit_path.is_file() else []
    manifest_rows = pd.read_csv(manifest_path).to_dict("records") if manifest_path.is_file() else []
    init_rows = pd.read_csv(init_path).to_dict("records") if init_path.is_file() else []
    trajectory_rows = pd.read_csv(trajectory_path).to_dict("records") if trajectory_path.is_file() else []
    for task in TASKS:
        for fold_id in range(5):
            fold = folds[fold_id]; allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            bundle = base.build_bundle(task, allowed); cache = base.RawGPUCache(bundle, device)
            mean, std, normalizer_meta, batch_info, weight, weight_meta, manifest_row = load_inputs(task, fold, bundle, references[(task, fold_id)])
            model, initialization = shared.initialize_c1(task, fold_id, SEED); initialization.update({"scope": SCOPE, "method": METHOD}); model.to(device)
            if base.TASKS[task]["mi_protocol"]:
                audit_indices = np.asarray(batch_info["episodes"][0][0], dtype=np.int64)
            else:
                train_indices = bundle.indices(fold["inner_train_subjects"], base.TASKS[task]["source_sessions"])
                audit_indices = np.asarray(base.task_epoch_batches(train_indices, task, fold_id, 1)[0], dtype=np.int64)
            value, labels = cache.batch(audit_indices, mean, std); weight_device = None if weight is None else weight.to(device)
            current_audits = [preflight_audit(task, fold_id, model, value, labels, weight_device, False), preflight_audit(task, fold_id, model, value, labels, weight_device, True)]
            if not all(row["status"] == "PASS" for row in current_audits):
                raise RuntimeError(f"SC_PROTOCOL_FAIL preflight {task}/f{fold_id}")
            audit_rows = [row for row in audit_rows if not (row["task"] == task and int(row["fold"]) == fold_id)] + current_audits
            manifest_rows = [row for row in manifest_rows if not (row["task"] == task and int(row["fold"]) == fold_id)] + [manifest_row]
            init_rows = [row for row in init_rows if not (row["task"] == task and int(row["fold"]) == fold_id)] + [initialization]
            base.write_csv(audit_path, pd.DataFrame(audit_rows).sort_values(["task", "fold", "precision"]))
            base.write_csv(manifest_path, pd.DataFrame(manifest_rows).sort_values(["task", "fold"]))
            base.write_csv(init_path, pd.DataFrame(init_rows).sort_values(["task", "fold"]))
            del model; model, _ = shared.initialize_c1(task, fold_id, SEED); model.to(device); base.set_seed(SEED + 100_000)
            record, history = train_one(task, model, fold, bundle, cache, mean, std, normalizer_meta, batch_info, weight, weight_meta, device)
            records[(task, fold_id)] = record
            trajectory_rows = [row for row in trajectory_rows if not (row["task"] == task and int(row["fold"]) == fold_id)] + history
            base.write_csv(trajectory_path, pd.DataFrame(trajectory_rows).sort_values(["task", "fold", "epoch"]))
            serial = [records[key] for key in sorted(records)]
            base.write_json(provenance_path, {"scope": SCOPE, "seed": SEED, "split_sha256": split_hash, "records": serial})
            del model, bundle, cache, value, labels; gc.collect(); torch.cuda.empty_cache()
    if len(records) != 15:
        raise RuntimeError(f"incomplete post-gate C0 grid: {len(records)}/15")
    base.write_json(PROTOCOL / "POST_GATE_C0_OPENBMI_CHECKPOINT_LOCK.json", {
        "status": "PASS", "scope": SCOPE, "seed": SEED, "checkpoints_frozen_before_evaluation": True,
        "WBCIC_gate_was_passed": False, "user_authorized_post_gate_run": True, "new_sealed_test_accessed": False,
        "records": [{"task": row["task"], "fold": row["fold"], "checkpoint_sha256": row["checkpoint_sha256"]} for row in records.values()],
    })
    evaluate_all(folds, records, references, device)
    print("POST_GATE_C0_OPENBMI_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
