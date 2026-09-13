"""Matched Linux LiteBN baseline, WBCIC comparison, heldout diagnostic, and gate."""
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
import torch.nn.functional as F

REPO = Path(os.environ.get("XNG_REPO", "/root/rivermind-data/CRCICLR_XNG_LINUX_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_xng_linux_seed0_v1"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("XNG_RUNTIME", "/root/rivermind-data/litebn_xng_linux_seed0_runtime")).resolve()
ORIGINAL_EXP = REPO / "experiments" / "persist_eeg_litebn_x_singlemodel_seed0_v1"
ORIGINAL_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
ORIGINAL_CODE = ORIGINAL_EXP / "code"
os.environ.setdefault("LITEBN_X_REPO", str(REPO))
sys.path.insert(0, str(ORIGINAL_CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import litebn_x as base
import run_xng_task as xng_run
from litebn_xng import initialize_xng

SEED = 0
BASELINE = "LiteBN_BASELINE"
CANDIDATE = "LiteBN_XNG"


def source_hash() -> str:
    return base.sha256_bytes(Path(__file__).read_bytes() + (ORIGINAL_CODE / "litebn_x.py").read_bytes())


def checkpoint_path(fold_id: int, which: str) -> Path:
    return RUNTIME / "checkpoints" / "wbcic_mi" / f"fold{fold_id}_litebn_baseline" / which


def train_one(model: torch.nn.Module, fold: dict[str, Any], bundle: base.SignalBundle, cache: base.RawGPUCache,
              mean: np.ndarray, std: np.ndarray, norm_meta: dict[str, Any], batch_info: dict[str, Any],
              device: torch.device) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    latest, selected = checkpoint_path(fold_id, "checkpoint_latest.pt"), checkpoint_path(fold_id, "selected_best.pt")
    initial_hash = base.state_hash(model)
    invariants = {
        "task": "WBCIC_MI", "fold": fold_id, "architecture": BASELINE, "seed": SEED,
        "initial_sha256": initial_hash, "source_sha256": source_hash(),
        "authoritative_litebn_source_sha256": base.sha256_file(base.CARRIER_CODE / "run_carrier_screen.py"),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        base.restore_rng(saved["rng"])
        start = int(saved["epoch"]) + 1
        history, best, best_epoch, best_state = list(saved["history"]), float(saved["best"]), saved["best_epoch"], saved["best_state"]
    started = time.perf_counter()
    for epoch in range(start, base.EPOCHS + 1):
        model.train()
        losses = []
        batches: Iterable[np.ndarray] = batch_info["episodes"][epoch - 1]
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(value)
                loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite CE: WBCIC_MI/fold{fold_id}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = base.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        val_ba = float(np.mean([row["BA"] for row in validation.values()]))
        val_f1 = float(np.mean([row["macro_F1"] for row in validation.values()]))
        chose = epoch >= base.MIN_EPOCH and val_ba > best + base.TIE_TOL
        if chose:
            best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "cross_entropy": float(np.mean(losses)), "inner_val_subject_BA": val_ba,
                        "inner_val_subject_macro_F1": val_f1, "selected": bool(chose), "batches": len(losses)})
        base.atomic_torch_save(latest, {"epoch": epoch, "history": history, "best": best, "best_epoch": best_epoch,
                                        "best_state": best_state, "current_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                                        "scaler": scaler.state_dict(), "rng": base.rng_state(), "invariants": invariants})
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[LiteBN WBCIC_MI f{fold_id}] epoch={epoch:02d} CE={history[-1]['cross_entropy']:.4f} valBA={val_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no baseline checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    base.atomic_torch_save(selected, model.state_dict())
    return {
        "task": "WBCIC_MI", "dataset": "WBCIC", "fold": fold_id, "architecture": BASELINE, "seed": SEED,
        "parameter_count": base.parameter_count(model), "initial_sha256": initial_hash,
        "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best),
        "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)),
        "checkpoint_path": str(selected), "checkpoint_sha256": base.sha256_file(selected),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
        "elapsed_seconds_this_invocation": time.perf_counter() - started, "epochs_completed": len(history),
        "history": history, "source_sha256": source_hash(),
    }


def load_model(method: str, checkpoint: Path, fold_id: int, device: torch.device) -> torch.nn.Module:
    if method == BASELINE:
        base.set_seed(SEED)
        model = base.LiteBN_BASELINE(58, 2)
    else:
        model, _ = initialize_xng("WBCIC_MI", fold_id, SEED)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
    model.to(device).eval()
    return model


def evaluate_outer(folds: dict[int, dict[str, Any]], baseline_records: list[dict[str, Any]], xng_records: list[dict[str, Any]], device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prior_xng = pd.read_csv(OUT / "XNG_ONLY_WBCIC_OUTER_SUBJECT_RESULTS.csv")
    baseline_rows = []
    for fold_id in range(5):
        fold, record = folds[fold_id], baseline_records[fold_id]
        checkpoint = Path(record["checkpoint_path"])
        if base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError("XNG_PROTOCOL_FAIL: baseline checkpoint hash mismatch")
        bundle = base.build_bundle("WBCIC_MI", fold["outer_dev_subjects"])
        cache = base.RawGPUCache(bundle, device)
        mean, std, norm_meta = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        model = load_model(BASELINE, checkpoint, fold_id, device)
        for subject, metrics in base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std).items():
            baseline_rows.append({"task": "WBCIC_MI", "fold": fold_id, "subject_id": subject, "method": BASELINE, **metrics,
                                  "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm_meta["mean_std_sha256"]})
        del model, cache, bundle
        gc.collect(); torch.cuda.empty_cache()
    baseline_frame = pd.DataFrame(baseline_rows)
    if len(baseline_frame) != 31 or baseline_frame.duplicated(["subject_id"]).any():
        raise RuntimeError(f"XNG_PROTOCOL_FAIL: baseline outer cardinality {len(baseline_frame)}/31")
    subject_frame = pd.concat([baseline_frame, prior_xng], ignore_index=True).sort_values(["fold", "subject_id", "method"])
    if len(subject_frame) != 62 or subject_frame.duplicated(["subject_id", "method"]).any():
        raise RuntimeError("XNG_PROTOCOL_FAIL: paired outer cardinality")
    base.write_csv(OUT / "WBCIC_OUTER_SUBJECT_RESULTS.csv", subject_frame)

    folds_rows = []
    for fold_id in range(5):
        subset = subject_frame[subject_frame.fold == fold_id]
        b = subset[subset.method == BASELINE]
        x = subset[subset.method == CANDIDATE]
        br, xr = baseline_records[fold_id], xng_records[fold_id]
        folds_rows.append({
            "task": "WBCIC_MI", "fold": fold_id, "subjects": len(b),
            "LiteBN_BA": b.BA.mean(), "XNG_BA": x.BA.mean(), "delta_BA_pp": (x.BA.mean() - b.BA.mean()) * 100,
            "LiteBN_macro_F1": b.macro_F1.mean(), "XNG_macro_F1": x.macro_F1.mean(), "delta_macro_F1_pp": (x.macro_F1.mean() - b.macro_F1.mean()) * 100,
            "LiteBN_accuracy": b.accuracy.mean(), "XNG_accuracy": x.accuracy.mean(),
            "LiteBN_selected_epoch": br["selected_epoch"], "XNG_selected_epoch": xr["selected_epoch"],
            "LiteBN_selected_inner_val_BA": br["best_inner_val_BA"], "XNG_selected_inner_val_BA": xr["best_inner_val_BA"],
        })
    fold_frame = pd.DataFrame(folds_rows)
    base.write_csv(OUT / "WBCIC_OUTER_FOLD_RESULTS.csv", fold_frame)

    b = subject_frame[subject_frame.method == BASELINE].set_index("subject_id").sort_index()
    x = subject_frame[subject_frame.method == CANDIDATE].set_index("subject_id").reindex(b.index)
    if x.BA.isna().any():
        raise RuntimeError("XNG_PROTOCOL_FAIL: subject pairing mismatch")
    paired = pd.DataFrame({
        "subject_id": b.index, "fold": b.fold.astype(int).to_numpy(),
        "LiteBN_BA": b.BA.to_numpy(), "XNG_BA": x.BA.to_numpy(), "delta_BA_pp": (x.BA - b.BA).to_numpy() * 100,
        "LiteBN_macro_F1": b.macro_F1.to_numpy(), "XNG_macro_F1": x.macro_F1.to_numpy(), "delta_macro_F1_pp": (x.macro_F1 - b.macro_F1).to_numpy() * 100,
    })
    base.write_csv(OUT / "WBCIC_OUTER_PAIRED_SUBJECT_DELTAS.csv", paired)
    boot = base.paired_bootstrap(paired.delta_BA_pp.to_numpy(float))
    base.write_csv(OUT / "WBCIC_OUTER_BOOTSTRAP.csv", pd.DataFrame([{"task": "WBCIC_MI", "comparison": "LiteBN_XNG-LiteBN_BASELINE", **boot}]))
    deltas = fold_frame.delta_BA_pp.to_numpy(float)
    summary = {
        "task": "WBCIC_MI", "n_outer_subjects": len(b),
        "LiteBN_BA": float(b.BA.mean()), "XNG_BA": float(x.BA.mean()), "mean_delta_pp": float(paired.delta_BA_pp.mean()),
        "median_fold_delta_pp": float(np.median(deltas)), "positive_folds": int((deltas > 0).sum()), "negative_folds": int((deltas < 0).sum()),
        "worst_fold_delta_pp": float(deltas.min()), "best_fold_delta_pp": float(deltas.max()), "fold_delta_sd_pp": float(deltas.std(ddof=1)),
        "LiteBN_macro_F1": float(b.macro_F1.mean()), "XNG_macro_F1": float(x.macro_F1.mean()),
        "LiteBN_accuracy": float(b.accuracy.mean()), "XNG_accuracy": float(x.accuracy.mean()),
        "subject_mean_delta_pp": float(paired.delta_BA_pp.mean()), "subject_median_delta_pp": float(paired.delta_BA_pp.median()),
        "positive_subjects": int((paired.delta_BA_pp > base.TIE_TOL).sum()), "tied_subjects": int((paired.delta_BA_pp.abs() <= base.TIE_TOL).sum()),
        "negative_subjects": int((paired.delta_BA_pp < -base.TIE_TOL).sum()),
        "bootstrap_ci_low_pp": boot["ci_low_pp"], "bootstrap_ci_high_pp": boot["ci_high_pp"],
    }
    base.write_csv(OUT / "WBCIC_OUTER_TASK_SUMMARY.csv", pd.DataFrame([summary]))
    return subject_frame, fold_frame, summary


def evaluate_internal_heldout(baseline_records: list[dict[str, Any]], xng_records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    manifest_path = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    subjects = base.subject_sort(manifest["WBCIC"]["subject_ids"], "WBCIC")
    if len(subjects) != 10 or int(manifest["WBCIC"]["session_index"]) != 2:
        raise RuntimeError("XNG_PROTOCOL_FAIL: heldout manifest mismatch")
    base.write_json(PROTOCOL / "WBCIC_INTERNAL_HELDOUT_PREFLIGHT.json", {
        "status": "PASS", "label": "INTERNAL_HELDOUT_DEVELOPMENT_DIAGNOSTIC",
        "development_model_selection_data": True, "manifest_path": str(manifest_path), "manifest_sha256": base.sha256_file(manifest_path),
        "subjects": subjects, "labels_accessed_at_preflight": False, "outer_analysis_frozen": True,
        "new_sealed_final_test_accessed": False,
    })
    rows = []
    for fold_id in range(5):
        mean, std, norm_meta = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
        bundle = base.build_bundle("WBCIC_MI", subjects)
        cache = base.RawGPUCache(bundle, device)
        for method, record in ((BASELINE, baseline_records[fold_id]), (CANDIDATE, xng_records[fold_id])):
            checkpoint = Path(record["checkpoint_path"])
            if base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError("XNG_PROTOCOL_FAIL: heldout checkpoint provenance mismatch")
            model = load_model(method, checkpoint, fold_id, device)
            for subject, metrics in base.evaluate(model, bundle, cache, subjects, mean, std).items():
                rows.append({"diagnostic": "INTERNAL_HELDOUT_DEVELOPMENT_DIAGNOSTIC", "task": "WBCIC_MI", "fold": fold_id,
                             "subject_id": subject, "method": method, **metrics, "checkpoint_sha256": record["checkpoint_sha256"],
                             "normalizer_sha256": norm_meta["mean_std_sha256"]})
            del model
            torch.cuda.empty_cache()
        del cache, bundle
        gc.collect(); torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["fold", "subject_id", "method"])
    if len(frame) != 100 or frame.duplicated(["fold", "subject_id", "method"]).any():
        raise RuntimeError(f"XNG_PROTOCOL_FAIL: heldout replicate cardinality {len(frame)}/100")
    base.write_csv(OUT / "WBCIC_INTERNAL_HELDOUT_REPLICATE_RESULTS.csv", frame)
    subject = frame.groupby(["subject_id", "method"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    b = subject[subject.method == BASELINE].set_index("subject_id")
    x = subject[subject.method == CANDIDATE].set_index("subject_id").reindex(b.index)
    delta = (x.BA - b.BA) * 100
    summary = {
        "diagnostic": "INTERNAL_HELDOUT_DEVELOPMENT_DIAGNOSTIC", "status": "DEVELOPMENT_MODEL_SELECTION_DATA",
        "task": "WBCIC_MI", "subjects": len(b), "replicates_per_method": 50,
        "LiteBN_BA": float(b.BA.mean()), "XNG_BA": float(x.BA.mean()), "delta_pp": float(delta.mean()),
        "LiteBN_macro_F1": float(b.macro_F1.mean()), "XNG_macro_F1": float(x.macro_F1.mean()),
        "LiteBN_accuracy": float(b.accuracy.mean()), "XNG_accuracy": float(x.accuracy.mean()),
        "positive_subjects": int((delta > base.TIE_TOL).sum()), "tied_subjects": int((delta.abs() <= base.TIE_TOL).sum()),
        "negative_subjects": int((delta < -base.TIE_TOL).sum()), "new_sealed_final_test_accessed": False,
    }
    base.write_csv(OUT / "WBCIC_INTERNAL_HELDOUT_SUMMARY.csv", pd.DataFrame([summary]))
    return summary


def final_report(outer: dict[str, Any], heldout: dict[str, Any]) -> None:
    audits = [
        (pd.read_csv(OUT / "XNG_INITIALIZATION_AUDIT.csv").status == "PASS").all(),
        (pd.read_csv(OUT / "XNG_PARAMETER_AUDIT.csv").status == "PASS").all(),
        (pd.read_csv(OUT / "MANIFEST_AUDIT.csv").status == "PASS").all(),
    ]
    conditions = {
        "outer_mean_delta_ge_0_20_pp": outer["mean_delta_pp"] >= 0.20,
        "positive_outer_folds_ge_3": outer["positive_folds"] >= 3,
        "worst_outer_fold_gt_minus_1_00_pp": outer["worst_fold_delta_pp"] > -1.00,
        "internal_heldout_delta_ge_0_00_pp": heldout["delta_pp"] >= 0.00,
        "all_audits_pass": bool(all(audits)),
    }
    passed = all(conditions.values())
    terminal = "WBCIC_GATE_PASS" if passed else "WBCIC_GATE_FAIL"
    gate = {"status": terminal, "seed": 0, "conditions": conditions, "outer": outer, "internal_heldout": heldout,
            "phase2_executed": False, "reason_phase2_not_executed": "awaiting explicit continuation" if passed else "predeclared WBCIC gate failed"}
    base.write_json(OUT / "WBCIC_GATE.json", gate)
    if passed:
        print("WBCIC_GATE_PASS", flush=True)
        return
    decision_terminal = "WBCIC_GATE_FAIL"
    decision = {
        "model": "LiteBN-XNG", "baseline": "EXACT_HISTORICAL_LITEBN_MATCHED_LINUX_RUNTIME", "seed": 0,
        "phase2_executed": False, "wbcic_gate": terminal, "terminal_label": decision_terminal,
        "internal_heldout_status": "DEVELOPMENT_MODEL_SELECTION_DATA", "new_sealed_final_test_accessed": False,
    }
    base.write_json(OUT / "FINAL_XNG_DECISION.json", decision)
    lines = [
        "# Final LiteBN-XNG seed-0 decision", "", "MODEL = LiteBN-XNG", "BASELINE = EXACT_HISTORICAL_LITEBN_MATCHED_LINUX_RUNTIME", "SEED = 0", "",
        "WIDE_TEMPORAL_CHANNELS = 12", "WIDE_SPATIAL_CHANNELS = 24", "BACKEND_FEATURES = 96", "EMBEDDING_DIM = 128", "",
        "RESIDUAL_TEMPORAL_MIXERS = YES", "READOUT_MEAN = YES", "READOUT_STD = YES", "READOUT_4_BINS = YES", "",
        "SCALE_GATE = NO", "SCALE_MLP = NO", "LAMBDA_SCALE = NO", "CHANNEL_GATE = NO", "",
        "AUXILIARY_LOSS = NO", "ENSEMBLE = NO", "DISTILLATION = NO", "TTA = NO", "",
        "WBCIC_RUN_FIRST = YES", "WBCIC_OUTER_FOLDS = 5", "WBCIC_INTERNAL_HELDOUT_USED_FOR_CONTINUATION = YES", "",
        "INTERNAL_HELDOUT_STATUS =", "DEVELOPMENT_MODEL_SELECTION_DATA", "", "NEW_SEALED_FINAL_TEST_ACCESSED = NO", "", "PHASE2_EXECUTED = NO", "",
        "## WBCIC result", "", f"Outer LiteBN BA: {outer['LiteBN_BA']:.6f}", f"Outer XNG BA: {outer['XNG_BA']:.6f}",
        f"Outer delta: {outer['mean_delta_pp']:+.3f} pp", f"Positive folds: {outer['positive_folds']}/5", f"Worst fold delta: {outer['worst_fold_delta_pp']:+.3f} pp",
        f"Internal-heldout LiteBN BA: {heldout['LiteBN_BA']:.6f}", f"Internal-heldout XNG BA: {heldout['XNG_BA']:.6f}", f"Internal-heldout delta: {heldout['delta_pp']:+.3f} pp", "",
        decision_terminal,
    ]
    base.write_text(OUT / "FINAL_XNG_DECISION.md", "\n".join(lines))
    print(decision_terminal, flush=True)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("XNG_PROTOCOL_FAIL: CUDA unavailable")
    search, folds_by_dataset, split_hash = base.load_folds()
    folds = {int(row["fold_id"]): row for row in folds_by_dataset["WBCIC"]}
    xng_lock = json.loads((PROTOCOL / "XNG_FIRST_LOCK.json").read_text(encoding="utf-8"))
    if not xng_lock.get("five_checkpoints_frozen") or xng_lock.get("internal_heldout_accessed"):
        raise RuntimeError("XNG_PROTOCOL_FAIL: invalid candidate freeze lock")
    xng_records_raw = json.loads((PROTOCOL / "XNG_CHECKPOINT_PROVENANCE.json").read_text(encoding="utf-8"))["records"]
    xng_by_fold = {int(row["fold"]): row for row in xng_records_raw}
    if set(xng_by_fold) != set(range(5)):
        raise RuntimeError("XNG_PROTOCOL_FAIL: missing candidate checkpoints")

    records = []
    provenance_file = PROTOCOL / "LITEBN_CHECKPOINT_PROVENANCE.json"
    if provenance_file.is_file():
        records = json.loads(provenance_file.read_text(encoding="utf-8")).get("records", [])
    by_fold = {int(row["fold"]): row for row in records}
    for fold_id in range(5):
        fold = folds[fold_id]
        allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
        bundle = base.build_bundle("WBCIC_MI", allowed)
        mean, std, norm_meta, batch_info, manifest_audit = xng_run.load_stored_inputs(fold, bundle)
        if manifest_audit["status"] != "PASS":
            raise RuntimeError("XNG_PROTOCOL_FAIL: baseline manifest audit")
        cache = base.RawGPUCache(bundle, device)
        base.set_seed(SEED)
        model = base.LiteBN_BASELINE(58, 2).to(device)
        base.set_seed(SEED + 100_000)
        record = train_one(model, fold, bundle, cache, mean, std, norm_meta, batch_info, device)
        by_fold[fold_id] = record
        serial = [{key: value for key, value in by_fold[index].items() if key != "history"} for index in sorted(by_fold)]
        base.write_json(provenance_file, {"stage": "MATCHED_LINUX_LITEBN", "seed": 0, "split_sha256": split_hash, "records": serial})
        del model, cache, bundle
        gc.collect(); torch.cuda.empty_cache()
    baseline_records = [by_fold[index] for index in range(5)]
    xng_records = [xng_by_fold[index] for index in range(5)]
    base.write_json(PROTOCOL / "MATCHED_WBCIC_CHECKPOINT_LOCK.json", {
        "status": "PASS", "seed": 0, "task": "WBCIC_MI", "candidate_checkpoints_frozen": True,
        "baseline_checkpoints_frozen": True, "outer_accessed_at_lock": False, "internal_heldout_accessed_at_lock": False,
        "split_sha256": split_hash,
        "baseline": [{"fold": r["fold"], "sha256": r["checkpoint_sha256"]} for r in baseline_records],
        "candidate": [{"fold": r["fold"], "sha256": r["checkpoint_sha256"]} for r in xng_records],
    })
    _, _, outer_summary = evaluate_outer(folds, baseline_records, xng_records, device)
    base.write_json(PROTOCOL / "WBCIC_OUTER_FREEZE.json", {"status": "PASS", "outer_analysis_complete": True,
                    "internal_heldout_accessed": False, "summary": outer_summary})
    heldout_summary = evaluate_internal_heldout(baseline_records, xng_records, device)
    final_report(outer_summary, heldout_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
