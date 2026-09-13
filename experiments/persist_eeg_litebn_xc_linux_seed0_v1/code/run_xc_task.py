"""Run fixed LiteBN-XC on WBCIC-MI seed0, then freeze outer and heldout diagnostics."""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get("XC_REPO", "/root/rivermind-data/CRCICLR_XC_LINUX_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_xc_linux_seed0_v1"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("XC_RUNTIME", "/root/rivermind-data/litebn_xc_linux_seed0_runtime")).resolve()
XNG_EXP = REPO / "experiments" / "persist_eeg_litebn_xng_linux_seed0_v1"
XNG_RUNTIME = Path("/root/rivermind-data/litebn_xng_linux_seed0_runtime")
ORIGINAL_EXP = REPO / "experiments" / "persist_eeg_litebn_x_singlemodel_seed0_v1"
ORIGINAL_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
ORIGINAL_CODE = ORIGINAL_EXP / "code"
XNG_CODE = XNG_EXP / "code"
os.environ.setdefault("LITEBN_X_REPO", str(REPO))
sys.path.insert(0, str(ORIGINAL_CODE))
sys.path.insert(0, str(XNG_CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import litebn_x as base
import litebn_xng as xng_source
from litebn_xc import function_audit, initialize_xc, parameter_audit

TASK, ARCH, SEED = "WBCIC_MI", "LiteBN_XC", 0
DIAGNOSTIC_EPOCHS = {0, 1, 5, 10, 20, 40, 60}


def source_hash() -> str:
    files = [Path(__file__), Path(__file__).parent / "litebn_xc.py", XNG_CODE / "litebn_xng.py", ORIGINAL_CODE / "litebn_x.py"]
    return base.sha256_bytes(b"".join(path.read_bytes() for path in files))


def checkpoint_path(fold_id: int, which: str) -> Path:
    return RUNTIME / "checkpoints" / "wbcic_mi" / f"fold{fold_id}_litebn_xc" / which


def load_inputs(fold: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    fold_id = int(fold["fold_id"])
    normalizer_path = ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz"
    manifest_path = ORIGINAL_RUNTIME / "episode_manifests" / f"wbcic_mi_fold{fold_id}.json"
    xng_audits = pd.read_csv(XNG_EXP / "outputs" / "MANIFEST_AUDIT.csv").set_index("fold")
    expected = xng_audits.loc[fold_id]
    mean, std, norm_meta = base.load_tensor_pair(normalizer_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = base.sha256_file(manifest_path)
    row = {
        "task": TASK, "fold": fold_id,
        "manifest_sha256": manifest_hash, "xng_manifest_sha256": str(expected.manifest_sha256),
        "manifest_hash_matches_xng": manifest_hash == str(expected.manifest_sha256),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "xng_normalizer_sha256": str(expected.normalizer_sha256),
        "normalizer_hash_matches_xng": norm_meta["mean_std_sha256"] == str(expected.normalizer_sha256),
        "training_subjects_match": base.subject_sort(manifest["training_subjects"], "WBCIC") == base.subject_sort(fold["inner_train_subjects"], "WBCIC"),
        "outer_rows_absent": bool(manifest["outer_rows_absent"]), "epochs": len(manifest["epochs"]),
    }
    row["status"] = "PASS" if all((row["manifest_hash_matches_xng"], row["normalizer_hash_matches_xng"],
                                     row["training_subjects_match"], row["outer_rows_absent"], row["epochs"] == 60)) else "FAIL"
    if row["status"] != "PASS":
        raise RuntimeError(f"XC_PROTOCOL_FAIL: {row}")
    episodes = [[np.asarray(indices, dtype=np.int64) for indices in epoch] for epoch in manifest["epochs"]]
    return mean, std, norm_meta, {"episodes": episodes, "manifest_sha256": manifest_hash}, row


def build_wbcic_future_bundle(subjects: Iterable[str]) -> base.SignalBundle:
    """Load only WBCIC session 2 for frozen outer/heldout inference."""
    canonical = base.subject_sort(subjects, "WBCIC")
    root = base.CACHE / "wbcic" / "wbcic_epochs"
    rows: list[base.Row] = []
    for subject in canonical:
        signal = root / subject / "ses-2_epochs.npy"
        labels = root / subject / "ses-2_labels.npy"
        if not signal.is_file() or not labels.is_file():
            raise FileNotFoundError(f"WBCIC future cache missing: {subject}")
        x = np.load(signal, mmap_mode="r", allow_pickle=False)
        y = np.load(labels, mmap_mode="r", allow_pickle=False)
        if x.ndim != 3 or tuple(x.shape[1:]) != (58, 1000) or x.dtype != np.float16:
            raise RuntimeError(f"WBCIC future schema mismatch: {signal}")
        if y.shape != (x.shape[0],) or set(map(int, np.unique(y))) != {0, 1}:
            raise RuntimeError(f"WBCIC future label mismatch: {labels}")
        rows.extend(base.Row(str(subject), 2, str(signal), index, int(label)) for index, label in enumerate(y))
    return base.SignalBundle(TASK, canonical, rows)


def gate_diagnostic(model: torch.nn.Module, value: torch.Tensor, fold_id: int, epoch: int, label: str) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
        diff_rms = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + 1e-8)
        raw = model.channel_mlp(torch.stack((rms, diff_rms), dim=-1)).squeeze(-1)
        amplitude = torch.tanh(model.lambda_channel)
        residual = amplitude * torch.tanh(raw)
        scale = 1.0 + residual
        row = {
            "task": TASK, "fold": fold_id, "checkpoint": label, "epoch": int(epoch),
            "lambda_channel": float(model.lambda_channel.detach().float().cpu()),
            "tanh_lambda_channel": float(amplitude.detach().float().cpu()),
            "mean_abs_raw_gate_output": float(raw.abs().mean().float().cpu()),
            "mean_abs_effective_residual": float(residual.abs().mean().float().cpu()),
            "mean_scale": float(scale.mean().float().cpu()), "std_scale": float(scale.std(unbiased=False).float().cpu()),
            "min_scale": float(scale.min().float().cpu()), "max_scale": float(scale.max().float().cpu()),
            "mean_per_electrode_rms": float(rms.mean().float().cpu()),
            "mean_temporal_difference_rms": float(diff_rms.mean().float().cpu()),
        }
    model.train(was_training)
    return row


def train_one(model: torch.nn.Module, fold: dict[str, Any], bundle: base.SignalBundle, cache: base.RawGPUCache,
              mean: np.ndarray, std: np.ndarray, norm_meta: dict[str, Any], batch_info: dict[str, Any],
              diagnostic_value: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_id = int(fold["fold_id"])
    latest, selected = checkpoint_path(fold_id, "checkpoint_latest.pt"), checkpoint_path(fold_id, "selected_best.pt")
    initial_hash = base.state_hash(model)
    invariants = {
        "task": TASK, "fold": fold_id, "architecture": ARCH, "seed": SEED,
        "initial_sha256": initial_hash, "source_sha256": source_hash(),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, diagnostics, best, best_epoch, best_state = 1, [], [], -float("inf"), None, None
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"]); base.restore_rng(saved["rng"])
        start, history, diagnostics = int(saved["epoch"]) + 1, list(saved["history"]), list(saved["diagnostics"])
        best, best_epoch, best_state = float(saved["best"]), saved["best_epoch"], saved["best_state"]
    elif 0 in DIAGNOSTIC_EPOCHS:
        diagnostics.append(gate_diagnostic(model, diagnostic_value, fold_id, 0, "epoch0"))

    started = time.perf_counter()
    for epoch in range(start, base.EPOCHS + 1):
        model.train(); losses = []
        batches: Iterable[np.ndarray] = batch_info["episodes"][epoch - 1]
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(value)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite CE: {TASK}/fold{fold_id}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = base.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        val_ba = float(np.mean([row["BA"] for row in validation.values()]))
        val_f1 = float(np.mean([row["macro_F1"] for row in validation.values()]))
        chose = epoch >= base.MIN_EPOCH and val_ba > best + base.TIE_TOL
        if chose:
            best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "cross_entropy": float(np.mean(losses)), "inner_val_subject_BA": val_ba,
                        "inner_val_subject_macro_F1": val_f1, "selected": bool(chose), "batches": len(losses)})
        if epoch in DIAGNOSTIC_EPOCHS:
            diagnostics.append(gate_diagnostic(model, diagnostic_value, fold_id, epoch, f"epoch{epoch}"))
        base.atomic_torch_save(latest, {"epoch": epoch, "history": history, "diagnostics": diagnostics, "best": best,
                                        "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(),
                                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": base.rng_state(), "invariants": invariants})
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[XC {TASK} f{fold_id}] epoch={epoch:02d} CE={history[-1]['cross_entropy']:.4f} valBA={val_ba:.4f} lambda={float(model.lambda_channel.detach()):+.6f}", flush=True)
    if best_state is None:
        raise RuntimeError("no XC checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    diagnostics = [row for row in diagnostics if row["checkpoint"] != "selected_epoch"]
    diagnostics.append(gate_diagnostic(model, diagnostic_value, fold_id, int(best_epoch), "selected_epoch"))
    base.atomic_torch_save(selected, model.state_dict())
    record = {
        "task": TASK, "dataset": "WBCIC", "fold": fold_id, "architecture": ARCH, "seed": SEED,
        "parameter_count": base.parameter_count(model), "initial_sha256": initial_hash,
        "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best),
        "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)),
        "checkpoint_path": str(selected), "checkpoint_sha256": base.sha256_file(selected),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
        "elapsed_seconds_this_invocation": time.perf_counter() - started, "epochs_completed": len(history), "source_sha256": source_hash(),
    }
    return record, diagnostics


def load_frozen(method: str, checkpoint: Path, fold_id: int, device: torch.device) -> torch.nn.Module:
    if method == "LiteBN_XNG":
        model, _ = xng_source.initialize_xng(TASK, fold_id, SEED)
    elif method == ARCH:
        _, model, _ = initialize_xc(TASK, fold_id, SEED)
    else:
        raise ValueError(method)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def paired_bootstrap(delta: np.ndarray) -> dict[str, Any]:
    draws = np.random.default_rng(0).choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
    return {"n_subjects": len(delta), "bootstrap_seed": 0, "resamples": 10_000, "mean_delta_pp": float(delta.mean()),
            "ci_low_pp": float(np.quantile(draws, 0.025)), "ci_high_pp": float(np.quantile(draws, 0.975)),
            "positive_subjects": int((delta > base.TIE_TOL).sum()), "tied_subjects": int((np.abs(delta) <= base.TIE_TOL).sum()),
            "negative_subjects": int((delta < -base.TIE_TOL).sum())}


def outer_evaluation(folds: dict[int, dict[str, Any]], xc_records: list[dict[str, Any]], xng_records: list[dict[str, Any]], device: torch.device) -> tuple[dict[str, Any], pd.DataFrame]:
    xng_frame = pd.read_csv(XNG_EXP / "outputs" / "XNG_ONLY_WBCIC_OUTER_SUBJECT_RESULTS.csv")
    xc_rows = []
    for fold_id in range(5):
        fold, record = folds[fold_id], xc_records[fold_id]
        checkpoint = Path(record["checkpoint_path"])
        if base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError("XC_PROTOCOL_FAIL: XC checkpoint hash mismatch")
        bundle = build_wbcic_future_bundle(fold["outer_dev_subjects"]); cache = base.RawGPUCache(bundle, device)
        mean, std, norm_meta = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        model = load_frozen(ARCH, checkpoint, fold_id, device)
        for subject, metrics in base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std).items():
            xc_rows.append({"task": TASK, "fold": fold_id, "subject_id": subject, "method": ARCH, **metrics,
                            "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm_meta["mean_std_sha256"]})
        del model, cache, bundle; gc.collect(); torch.cuda.empty_cache()
    xc_frame = pd.DataFrame(xc_rows)
    if len(xc_frame) != 31 or xc_frame.duplicated(["subject_id"]).any():
        raise RuntimeError(f"XC_PROTOCOL_FAIL: XC outer cardinality {len(xc_frame)}/31")
    xng = xng_frame.set_index("subject_id").sort_index(); xc = xc_frame.set_index("subject_id").reindex(xng.index)
    if xc.BA.isna().any() or not np.array_equal(xng.fold.to_numpy(), xc.fold.to_numpy()):
        raise RuntimeError("XC_PROTOCOL_FAIL: outer subject/fold pairing mismatch")
    paired = pd.DataFrame({
        "task": TASK, "fold": xng.fold.astype(int).to_numpy(), "subject_id": xng.index,
        "XNG_BA": xng.BA.to_numpy(), "XC_BA": xc.BA.to_numpy(), "delta_BA_pp": (xc.BA - xng.BA).to_numpy() * 100,
        "XNG_macro_F1": xng.macro_F1.to_numpy(), "XC_macro_F1": xc.macro_F1.to_numpy(), "delta_macro_F1_pp": (xc.macro_F1 - xng.macro_F1).to_numpy() * 100,
        "XNG_accuracy": xng.accuracy.to_numpy(), "XC_accuracy": xc.accuracy.to_numpy(),
    })
    base.write_csv(OUT / "WBCIC_XC_OUTER_SUBJECT_RESULTS.csv", paired.sort_values(["fold", "subject_id"]))
    fold_rows = []
    for fold_id in range(5):
        q = paired[paired.fold == fold_id]
        record = xc_records[fold_id]
        fold_rows.append({
            "task": TASK, "fold": fold_id, "subjects": len(q), "XNG_BA": q.XNG_BA.mean(), "XC_BA": q.XC_BA.mean(),
            "delta_BA_pp": q.delta_BA_pp.mean(), "XNG_macro_F1": q.XNG_macro_F1.mean(), "XC_macro_F1": q.XC_macro_F1.mean(),
            "delta_macro_F1_pp": q.delta_macro_F1_pp.mean(), "XNG_accuracy": q.XNG_accuracy.mean(), "XC_accuracy": q.XC_accuracy.mean(),
            "XC_selected_epoch": record["selected_epoch"], "XC_selected_inner_val_BA": record["best_inner_val_BA"],
        })
    fold_frame = pd.DataFrame(fold_rows)
    base.write_csv(OUT / "WBCIC_XC_OUTER_FOLD_RESULTS.csv", fold_frame)
    boot = paired_bootstrap(paired.delta_BA_pp.to_numpy(float))
    base.write_csv(OUT / "WBCIC_XC_BOOTSTRAP.csv", pd.DataFrame([{"comparison": "LiteBN_XC-LiteBN_XNG", **boot}]))
    deltas = fold_frame.delta_BA_pp.to_numpy(float)
    summary = {
        "task": TASK, "n_outer_subjects": len(paired), "XNG_BA": float(paired.XNG_BA.mean()), "XC_BA": float(paired.XC_BA.mean()),
        "delta_pp": float(paired.delta_BA_pp.mean()), "positive_folds": int((deltas > 0).sum()), "negative_folds": int((deltas < 0).sum()),
        "median_fold_delta_pp": float(np.median(deltas)), "worst_fold_delta_pp": float(deltas.min()), "best_fold_delta_pp": float(deltas.max()),
        "fold_delta_sd_pp": float(deltas.std(ddof=1)), "XNG_macro_F1": float(paired.XNG_macro_F1.mean()), "XC_macro_F1": float(paired.XC_macro_F1.mean()),
        "XNG_accuracy": float(paired.XNG_accuracy.mean()), "XC_accuracy": float(paired.XC_accuracy.mean()),
        "subject_mean_delta_pp": float(paired.delta_BA_pp.mean()), "subject_median_delta_pp": float(paired.delta_BA_pp.median()),
        **{f"bootstrap_{key}": value for key, value in boot.items() if key in ("ci_low_pp", "ci_high_pp")},
        "positive_subjects": boot["positive_subjects"], "tied_subjects": boot["tied_subjects"], "negative_subjects": boot["negative_subjects"],
    }
    base.write_csv(OUT / "WBCIC_XC_OUTER_TASK_SUMMARY.csv", pd.DataFrame([summary]))
    return summary, paired


def heldout_evaluation(xc_records: list[dict[str, Any]], xng_records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    manifest_path = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = base.subject_sort(manifest["WBCIC"]["subject_ids"], "WBCIC")
    if len(subjects) != 10 or int(manifest["WBCIC"]["session_index"]) != 2:
        raise RuntimeError("XC_PROTOCOL_FAIL: internal heldout manifest mismatch")
    base.write_json(PROTOCOL / "WBCIC_INTERNAL_HELDOUT_PREFLIGHT.json", {
        "status": "PASS", "label": "INTERNAL_HELDOUT_DEVELOPMENT_DIAGNOSTIC", "development_data": True,
        "manifest_sha256": base.sha256_file(manifest_path), "subjects": subjects, "labels_accessed_at_preflight": False,
        "all_xc_checkpoints_frozen": True, "outer_results_frozen": True, "new_sealed_final_test_accessed": False,
    })
    rows = []
    for fold_id in range(5):
        bundle = build_wbcic_future_bundle(subjects); cache = base.RawGPUCache(bundle, device)
        mean, std, norm_meta = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        for method, record in (("LiteBN_XNG", xng_records[fold_id]), (ARCH, xc_records[fold_id])):
            checkpoint = Path(record["checkpoint_path"])
            if base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError("XC_PROTOCOL_FAIL: heldout checkpoint hash mismatch")
            model = load_frozen(method, checkpoint, fold_id, device)
            for subject, metrics in base.evaluate(model, bundle, cache, subjects, mean, std).items():
                rows.append({"diagnostic": "INTERNAL_HELDOUT_DEVELOPMENT_DIAGNOSTIC", "task": TASK, "fold": fold_id,
                             "subject_id": subject, "method": method, **metrics, "checkpoint_sha256": record["checkpoint_sha256"],
                             "normalizer_sha256": norm_meta["mean_std_sha256"]})
            del model; torch.cuda.empty_cache()
        del cache, bundle; gc.collect(); torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["fold", "subject_id", "method"])
    if len(frame) != 100 or frame.duplicated(["fold", "subject_id", "method"]).any():
        raise RuntimeError(f"XC_PROTOCOL_FAIL: heldout cardinality {len(frame)}/100")
    base.write_csv(OUT / "WBCIC_XNG_VS_XC_HELDOUT_RESULTS.csv", frame)
    subject = frame.groupby(["subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    xng = subject[subject.method == "LiteBN_XNG"].set_index("subject_id")
    xc = subject[subject.method == ARCH].set_index("subject_id").reindex(xng.index)
    delta = (xc.BA - xng.BA) * 100
    return {
        "diagnostic": "INTERNAL_HELDOUT_DEVELOPMENT_DIAGNOSTIC", "status": "DEVELOPMENT_DATA", "subjects": len(xng),
        "XNG_BA": float(xng.BA.mean()), "XC_BA": float(xc.BA.mean()), "delta_pp": float(delta.mean()),
        "XNG_macro_F1": float(xng.macro_F1.mean()), "XC_macro_F1": float(xc.macro_F1.mean()),
        "XNG_accuracy": float(xng.accuracy.mean()), "XC_accuracy": float(xc.accuracy.mean()),
        "positive_subjects": int((delta > base.TIE_TOL).sum()), "tied_subjects": int((delta.abs() <= base.TIE_TOL).sum()),
        "negative_subjects": int((delta < -base.TIE_TOL).sum()), "new_sealed_final_test_accessed": False,
    }


def finalize(outer: dict[str, Any], heldout: dict[str, Any]) -> str:
    audits_pass = all([
        (pd.read_csv(OUT / "XC_INITIALIZATION_AUDIT.csv").status == "PASS").all(),
        (pd.read_csv(OUT / "XC_PARAMETER_AUDIT.csv").status == "PASS").all(),
        (pd.read_csv(OUT / "MANIFEST_AUDIT.csv").status == "PASS").all(),
    ])
    conditions = {
        "A_XC_outer_BA_gt_0_7900": outer["XC_BA"] > 0.7900,
        "B_outer_gain_vs_XNG_ge_0_80_pp": outer["delta_pp"] >= 0.80,
        "C_positive_folds_ge_3": outer["positive_folds"] >= 3,
        "D_worst_fold_delta_gt_minus_1_00_pp": outer["worst_fold_delta_pp"] > -1.00,
        "E_XC_heldout_BA_ge_XNG": heldout["XC_BA"] >= heldout["XNG_BA"],
        "F_XC_heldout_BA_ge_0_7900": heldout["XC_BA"] >= 0.7900,
        "G_all_audits_pass": bool(audits_pass),
    }
    terminal = "WBCIC_XC_GATE_PASS" if all(conditions.values()) else "WBCIC_XC_GATE_FAIL"
    base.write_json(OUT / "WBCIC_XC_GATE.json", {"status": terminal, "seed": 0, "conditions": conditions,
                    "outer": outer, "internal_heldout": heldout, "phase2_executed": False})
    decision = {
        "model": "LiteBN-XC", "base_model": "LiteBN-XNG", "seed": 0, "terminal_label": terminal,
        "phase2_executed": False, "internal_heldout_status": "DEVELOPMENT_DATA", "new_sealed_final_test_accessed": False,
    }
    base.write_json(OUT / "FINAL_XC_DECISION.json", decision)
    lines = [
        "# Final LiteBN-XC seed-0 WBCIC decision", "", "MODEL = LiteBN-XC", "BASE_MODEL = LiteBN-XNG", "SEED = 0", "",
        "TEMPORAL_CHANNELS = 12", "SPATIAL_CHANNELS = 24", "BACKEND_FEATURES = 96", "EMBEDDING_DIM = 128", "",
        "TEMPORAL_MIXERS = YES", "MEAN_STD_4BIN_READOUT = YES", "", "CHANNEL_GATE = YES", "CHANNEL_GATE_MLP = 2->8->1",
        "CHANNEL_GATE_SHARED_ACROSS_ELECTRODES = YES", "LAMBDA_CHANNEL_INIT = 0", "", "TEMPORAL_SCALE_GATE = NO", "SCALE_MLP = NO",
        "LAMBDA_SCALE = NO", "", "XNG_PARAMETER_COUNT = 216705", "XC_EXPECTED_PARAMETER_COUNT = 216739", "ADDED_PARAMETERS = 34", "",
        "WBCIC_RUN_FIRST = YES", "WBCIC_TARGET_OUTER_BA = >0.7900", "INTERNAL_HELDOUT_USED_FOR_SELECTION = YES",
        "INTERNAL_HELDOUT_STATUS = DEVELOPMENT_DATA", "", "PHASE2_EXECUTED = NO", "", "NEW_SEALED_FINAL_TEST_ACCESSED = NO", "",
        "## WBCIC results", "", f"Outer XNG BA: {outer['XNG_BA']:.6f}", f"Outer XC BA: {outer['XC_BA']:.6f}",
        f"Outer delta: {outer['delta_pp']:+.3f} pp", f"Positive folds: {outer['positive_folds']}/5",
        f"Worst fold delta: {outer['worst_fold_delta_pp']:+.3f} pp", f"Internal-heldout XNG BA: {heldout['XNG_BA']:.6f}",
        f"Internal-heldout XC BA: {heldout['XC_BA']:.6f}", f"Internal-heldout delta: {heldout['delta_pp']:+.3f} pp", "", terminal,
    ]
    base.write_text(OUT / "FINAL_XC_DECISION.md", "\n".join(lines))
    print(terminal, flush=True)
    return terminal


def write_runtime_metadata() -> None:
    xng_runtime = json.loads((XNG_EXP / "protocol" / "LINUX_RUNTIME_METADATA.json").read_text(encoding="utf-8"))
    current = {
        "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "amp_enabled": torch.cuda.is_available(), "amp_dtype": "float16", "cudnn_benchmark_configured": False,
        "cudnn_deterministic_configured": True,
    }
    material = ["platform", "python", "torch", "cuda_runtime", "cudnn_version", "gpu", "gpu_capability", "amp_enabled", "amp_dtype"]
    differences = {key: {"xng": xng_runtime.get(key), "xc": current.get(key)} for key in material if xng_runtime.get(key) != current.get(key)}
    current["xng_reference_metadata"] = str(XNG_EXP / "protocol" / "LINUX_RUNTIME_METADATA.json")
    current["material_differences"] = differences
    current["status"] = "PASS" if not differences else "RUNTIME_PROVENANCE_DIFFERENCE"
    base.write_json(PROTOCOL / "LINUX_RUNTIME_METADATA.json", current)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("XC_PROTOCOL_FAIL: CUDA unavailable")
    write_runtime_metadata()
    param = parameter_audit(TASK)
    base.write_csv(OUT / "XC_PARAMETER_AUDIT.csv", pd.DataFrame([param]))
    search, folds_by_dataset, split_hash = base.load_folds()
    folds = {int(row["fold_id"]): row for row in folds_by_dataset["WBCIC"]}
    xng_provenance = json.loads((XNG_EXP / "protocol" / "XNG_CHECKPOINT_PROVENANCE.json").read_text(encoding="utf-8"))
    xng_by_fold = {int(row["fold"]): row for row in xng_provenance["records"]}
    if set(xng_by_fold) != set(range(5)):
        raise RuntimeError("XC_PROTOCOL_FAIL: incomplete XNG reference")
    base.write_json(PROTOCOL / "XNG_REFERENCE_PROVENANCE.json", {
        "source_experiment": str(XNG_EXP), "source_git_branch": "codex/persist-eeg-litebn-xng-linux-seed0-v1",
        "outer_fold_results": str(XNG_EXP / "outputs" / "XNG_ONLY_WBCIC_OUTER_FOLD_RESULTS.csv"),
        "outer_subject_results": str(XNG_EXP / "outputs" / "XNG_ONLY_WBCIC_OUTER_SUBJECT_RESULTS.csv"),
        "records": [xng_by_fold[index] for index in range(5)],
    })

    provenance_path = PROTOCOL / "XC_CHECKPOINT_PROVENANCE.json"
    existing = json.loads(provenance_path.read_text(encoding="utf-8"))["records"] if provenance_path.is_file() else []
    by_fold = {int(row["fold"]): row for row in existing}
    init_rows, manifest_rows, diagnostic_rows = [], [], []
    for fold_id in range(5):
        fold = folds[fold_id]
        allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
        if set(allowed) & set(fold["outer_dev_subjects"]):
            raise RuntimeError("XC_PROTOCOL_FAIL: outer subject in training bundle")
        bundle = base.build_bundle(TASK, allowed)
        mean, std, norm_meta, batch_info, manifest_audit = load_inputs(fold)
        manifest_rows.append(manifest_audit)
        cache = base.RawGPUCache(bundle, device)
        xng, model, init_audit = initialize_xc(TASK, fold_id, SEED)
        xng.to(device); model.to(device)
        diagnostic_indices = bundle.indices(fold["inner_train_subjects"], base.TASKS[TASK]["source_sessions"])[:16]
        real_value, _ = cache.batch(diagnostic_indices, mean, std)
        init_audit.update(function_audit(xng, model, real_value))
        init_rows.append(init_audit)
        base.write_csv(OUT / "XC_INITIALIZATION_AUDIT.csv", pd.DataFrame(init_rows).sort_values("fold"))
        base.write_csv(OUT / "MANIFEST_AUDIT.csv", pd.DataFrame(manifest_rows).sort_values("fold"))
        del xng; torch.cuda.empty_cache()
        base.set_seed(SEED + 100_000)
        record, fold_diagnostics = train_one(model, fold, bundle, cache, mean, std, norm_meta, batch_info, real_value, device)
        diagnostic_rows.extend(fold_diagnostics)
        by_fold[fold_id] = record
        base.write_csv(OUT / "XC_GATE_DIAGNOSTICS.csv", pd.DataFrame(diagnostic_rows).sort_values(["fold", "epoch", "checkpoint"]))
        serial = [{key: value for key, value in by_fold[index].items()} for index in sorted(by_fold)]
        base.write_json(provenance_path, {"stage": "WBCIC_XC", "seed": 0, "split_sha256": split_hash, "records": serial})
        del model, cache, bundle, real_value; gc.collect(); torch.cuda.empty_cache()
    xc_records = [by_fold[index] for index in range(5)]
    xng_records = [xng_by_fold[index] for index in range(5)]
    base.write_json(PROTOCOL / "XC_WBCIC_CHECKPOINT_LOCK.json", {
        "status": "PASS", "seed": 0, "task": TASK, "five_xc_checkpoints_frozen": True,
        "outer_accessed_at_lock": False, "internal_heldout_accessed_at_lock": False, "split_sha256": split_hash,
        "checkpoints": [{"fold": row["fold"], "path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"]} for row in xc_records],
    })
    outer, _ = outer_evaluation(folds, xc_records, xng_records, device)
    base.write_json(PROTOCOL / "WBCIC_XC_OUTER_FREEZE.json", {"status": "PASS", "outer_analysis_complete": True,
                    "internal_heldout_accessed": False, "summary": outer})
    heldout = heldout_evaluation(xc_records, xng_records, device)
    finalize(outer, heldout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
