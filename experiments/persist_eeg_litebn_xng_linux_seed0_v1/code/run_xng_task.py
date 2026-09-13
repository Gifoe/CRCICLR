"""Run XNG first on WBCIC-MI seed0; matched LiteBN is intentionally deferred."""
from __future__ import annotations

import argparse
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
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("XNG_RUNTIME", "/root/rivermind-data/litebn_xng_linux_seed0_runtime")).resolve()
ORIGINAL_EXP = REPO / "experiments" / "persist_eeg_litebn_x_singlemodel_seed0_v1"
ORIGINAL_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
ORIGINAL_CODE = ORIGINAL_EXP / "code"
os.environ.setdefault("LITEBN_X_REPO", str(REPO))
sys.path.insert(0, str(ORIGINAL_CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import litebn_x as base
from litebn_xng import initialize_xng, parameter_audit

SEED = 0
ARCH = "LiteBN_XNG"


def combined_source_hash() -> str:
    content = Path(__file__).read_bytes() + (Path(__file__).parent / "litebn_xng.py").read_bytes() + (ORIGINAL_CODE / "litebn_x.py").read_bytes()
    return base.sha256_bytes(content)


def checkpoint_path(fold_id: int, which: str) -> Path:
    return RUNTIME / "checkpoints" / "wbcic_mi" / f"fold{fold_id}_litebn_xng" / which


def train_one(model: torch.nn.Module, fold: dict[str, Any], bundle: base.SignalBundle, cache: base.RawGPUCache,
              mean: np.ndarray, std: np.ndarray, norm_meta: dict[str, Any], batch_info: dict[str, Any],
              device: torch.device) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    latest, selected = checkpoint_path(fold_id, "checkpoint_latest.pt"), checkpoint_path(fold_id, "selected_best.pt")
    source_hash = combined_source_hash()
    initial_hash = base.state_hash(model)
    invariants = {
        "task": "WBCIC_MI", "fold": fold_id, "architecture": ARCH, "seed": SEED,
        "initial_sha256": initial_hash, "source_sha256": source_hash,
        "authoritative_x_source_sha256": base.sha256_file(ORIGINAL_CODE / "litebn_x.py"),
        "normalizer_sha256": norm_meta["mean_std_sha256"],
        "batch_manifest_sha256": batch_info["manifest_sha256"],
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
            print(f"[XNG WBCIC_MI f{fold_id}] epoch={epoch:02d} CE={history[-1]['cross_entropy']:.4f} valBA={val_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    base.atomic_torch_save(selected, model.state_dict())
    return {
        "task": "WBCIC_MI", "dataset": "WBCIC", "fold": fold_id, "architecture": ARCH, "seed": SEED,
        "parameter_count": base.parameter_count(model), "initial_sha256": initial_hash,
        "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best),
        "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)),
        "checkpoint_path": str(selected), "checkpoint_sha256": base.sha256_file(selected),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info["manifest_sha256"],
        "elapsed_seconds_this_invocation": time.perf_counter() - started, "epochs_completed": len(history),
        "history": history, "source_sha256": source_hash,
    }


def load_stored_inputs(fold: dict[str, Any], bundle: base.SignalBundle) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    fold_id = int(fold["fold_id"])
    normalizer_path = ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz"
    manifest_path = ORIGINAL_RUNTIME / "episode_manifests" / f"wbcic_mi_fold{fold_id}.json"
    provenance_path = ORIGINAL_EXP / "protocol" / "CHECKPOINT_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    reference = next(row for row in provenance["records"] if row["task"] == "WBCIC_MI" and int(row["fold"]) == fold_id and row["architecture"] == "LiteBN_X")
    mean, std, norm_meta = base.load_tensor_pair(normalizer_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = base.sha256_file(manifest_path)
    expected_subjects = base.subject_sort(fold["inner_train_subjects"], "WBCIC")
    checks = {
        "task": "WBCIC_MI", "fold": fold_id,
        "manifest_sha256": manifest_hash,
        "expected_manifest_sha256": reference["batch_manifest_sha256"],
        "manifest_hash_match": manifest_hash == reference["batch_manifest_sha256"],
        "normalizer_sha256": norm_meta["mean_std_sha256"],
        "expected_normalizer_sha256": reference["normalizer_sha256"],
        "normalizer_hash_match": norm_meta["mean_std_sha256"] == reference["normalizer_sha256"],
        "training_subjects_match": base.subject_sort(manifest["training_subjects"], "WBCIC") == expected_subjects,
        "outer_rows_absent": bool(manifest["outer_rows_absent"]),
        "epochs": len(manifest["epochs"]),
    }
    checks["status"] = "PASS" if all([checks["manifest_hash_match"], checks["normalizer_hash_match"], checks["training_subjects_match"], checks["outer_rows_absent"], checks["epochs"] == base.EPOCHS]) else "FAIL"
    if checks["status"] != "PASS":
        raise RuntimeError(f"XNG_PROTOCOL_FAIL manifest/normalizer: {checks}")
    episodes = [[np.asarray(indices, dtype=np.int64) for indices in epoch] for epoch in manifest["epochs"]]
    return mean, std, norm_meta, {"episodes": episodes, "manifest_sha256": manifest_hash}, checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    requested = list(dict.fromkeys(args.folds))
    if not requested or any(fold not in range(5) for fold in requested):
        raise SystemExit("--folds must be a non-empty subset of 0..4")
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("XNG_PROTOCOL_FAIL: CUDA unavailable")
    search, folds_by_dataset, split_hash = base.load_folds()
    folds = {int(row["fold_id"]): row for row in folds_by_dataset["WBCIC"]}
    parameter_row = parameter_audit("WBCIC_MI")
    parameter_row["authoritative_x_source_sha256"] = base.sha256_file(ORIGINAL_CODE / "litebn_x.py")
    base.write_csv(OUT / "XNG_PARAMETER_AUDIT.csv", pd.DataFrame([parameter_row]))

    audit_rows, manifest_rows, records = [], [], []
    provenance_file = PROTOCOL / "XNG_CHECKPOINT_PROVENANCE.json"
    if provenance_file.is_file():
        existing = json.loads(provenance_file.read_text(encoding="utf-8"))
        records = list(existing.get("records", []))
    by_fold = {int(row["fold"]): row for row in records}

    for fold_id in requested:
        fold = folds[fold_id]
        allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
        if set(allowed) & set(fold["outer_dev_subjects"]):
            raise RuntimeError("XNG_PROTOCOL_FAIL: outer subject in training bundle")
        bundle = base.build_bundle("WBCIC_MI", allowed)
        mean, std, norm_meta, batch_info, manifest_audit = load_stored_inputs(fold, bundle)
        manifest_rows.append(manifest_audit)
        cache = base.RawGPUCache(bundle, device)
        model, init_audit = initialize_xng("WBCIC_MI", fold_id, SEED)
        audit_rows.append(init_audit)
        base.write_csv(OUT / "XNG_INITIALIZATION_AUDIT.csv", pd.DataFrame(audit_rows).sort_values("fold"))
        base.write_csv(OUT / "MANIFEST_AUDIT.csv", pd.DataFrame(manifest_rows).sort_values("fold"))
        base.set_seed(SEED + 100_000)
        record = train_one(model.to(device), fold, bundle, cache, mean, std, norm_meta, batch_info, device)
        by_fold[fold_id] = record
        serial_records = [{key: value for key, value in by_fold[index].items() if key != "history"} for index in sorted(by_fold)]
        base.write_json(provenance_file, {"stage": "XNG_FIRST", "seed": SEED, "split_sha256": split_hash, "records": serial_records})
        del model, cache, bundle
        gc.collect()
        torch.cuda.empty_cache()

    if requested == [0, 1, 2, 3, 4] or set(by_fold) == set(range(5)):
        records = [by_fold[index] for index in range(5)]
        lock = {
            "stage": "XNG_FIRST", "task": "WBCIC_MI", "seed": SEED, "five_checkpoints_frozen": True,
            "litebn_baseline_trained": False, "internal_heldout_accessed": False, "split_sha256": split_hash,
            "source_sha256": combined_source_hash(),
            "checkpoints": [{"fold": row["fold"], "path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"]} for row in records],
        }
        base.write_json(PROTOCOL / "XNG_FIRST_LOCK.json", lock)

        outer_rows = []
        for fold_id in range(5):
            fold, record = folds[fold_id], records[fold_id]
            checkpoint = Path(record["checkpoint_path"])
            if base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError("XNG_PROTOCOL_FAIL: frozen checkpoint hash mismatch")
            bundle = base.build_bundle("WBCIC_MI", fold["outer_dev_subjects"])
            cache = base.RawGPUCache(bundle, device)
            mean, std, norm_meta = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"wbcic_mi_fold{fold_id}.npz")
            model, _ = initialize_xng("WBCIC_MI", fold_id, SEED)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
            model.to(device)
            for subject, metrics in base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std).items():
                outer_rows.append({"task": "WBCIC_MI", "fold": fold_id, "subject_id": subject, "method": ARCH, **metrics,
                                   "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm_meta["mean_std_sha256"]})
            del model, cache, bundle
            gc.collect(); torch.cuda.empty_cache()
        frame = pd.DataFrame(outer_rows).sort_values(["fold", "subject_id"])
        if len(frame) != 31 or frame.duplicated(["subject_id"]).any():
            raise RuntimeError(f"XNG_PROTOCOL_FAIL: outer cardinality {len(frame)}/31")
        base.write_csv(OUT / "XNG_ONLY_WBCIC_OUTER_SUBJECT_RESULTS.csv", frame)
        fold_frame = frame.groupby("fold", as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), accuracy=("accuracy", "mean"), subjects=("subject_id", "count"))
        fold_frame.insert(1, "method", ARCH)
        base.write_csv(OUT / "XNG_ONLY_WBCIC_OUTER_FOLD_RESULTS.csv", fold_frame)
        base.write_json(OUT / "XNG_FIRST_STATUS.json", {
            "status": "XNG_ONLY_WBCIC_COMPLETE", "seed": 0, "folds": 5,
            "outer_subjects": 31, "outer_subject_mean_BA": float(frame.BA.mean()),
            "outer_subject_mean_macro_F1": float(frame.macro_F1.mean()),
            "litebn_baseline_trained": False, "gate_evaluated": False,
            "internal_heldout_accessed": False,
        })
        print("XNG_ONLY_WBCIC_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
