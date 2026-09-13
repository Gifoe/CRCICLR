"""Train the exact historical LiteBN baseline on OpenBMI-MI seed0 first."""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("XNG_REPO", "/root/rivermind-data/CRCICLR_XNG_LINUX_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_xng_linux_seed0_v1"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("XNG_RUNTIME", "/root/rivermind-data/litebn_xng_linux_seed0_runtime")).resolve()
ORIGINAL_EXP = REPO / "experiments" / "persist_eeg_litebn_x_singlemodel_seed0_v1"
ORIGINAL_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
ORIGINAL_CODE = ORIGINAL_EXP / "code"
os.environ.setdefault("LITEBN_X_REPO", str(REPO))
sys.path.insert(0, str(ORIGINAL_CODE))

import litebn_x as base

TASK = "OpenBMI_MI"
ARCH = "LiteBN_BASELINE"
SEED = 0
base.RUNTIME = RUNTIME


def stored_inputs(fold: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    fold_id = int(fold["fold_id"])
    norm_path = ORIGINAL_RUNTIME / "normalizers" / f"openbmi_mi_fold{fold_id}.npz"
    manifest_path = ORIGINAL_RUNTIME / "episode_manifests" / f"openbmi_mi_fold{fold_id}.json"
    provenance = json.loads((ORIGINAL_EXP / "protocol" / "CHECKPOINT_PROVENANCE.json").read_text(encoding="utf-8"))
    reference = next(row for row in provenance["records"] if row["task"] == TASK and int(row["fold"]) == fold_id and row["architecture"] == "LiteBN_X")
    mean, std, norm_meta = base.load_tensor_pair(norm_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = base.sha256_file(manifest_path)
    recorded_manifest_hash = reference.get("batch_manifest_sha256")
    recorded_normalizer_hash = reference.get("normalizer_sha256")
    provenance_hashes_available = bool(recorded_manifest_hash and isinstance(recorded_normalizer_hash, str) and len(recorded_normalizer_hash) == 64)
    row = {
        "task": TASK, "fold": fold_id, "method": ARCH, "manifest_sha256": manifest_hash,
        "stored_reference_hashes_available": provenance_hashes_available,
        "recorded_manifest_sha256": recorded_manifest_hash,
        "manifest_hash_match": (manifest_hash == recorded_manifest_hash) if provenance_hashes_available else True,
        "normalizer_sha256": norm_meta["mean_std_sha256"], "recorded_normalizer_sha256": recorded_normalizer_hash,
        "normalizer_hash_match": (norm_meta["mean_std_sha256"] == recorded_normalizer_hash) if provenance_hashes_available else True,
        "training_subjects_match": base.subject_sort(manifest["training_subjects"], "OpenBMI") == base.subject_sort(fold["inner_train_subjects"], "OpenBMI"),
        "outer_rows_absent": bool(manifest["outer_rows_absent"]), "epochs": len(manifest["epochs"]),
    }
    row["status"] = "PASS" if all((row["manifest_hash_match"], row["normalizer_hash_match"], row["training_subjects_match"], row["outer_rows_absent"], row["epochs"] == 60)) else "FAIL"
    if row["status"] != "PASS":
        raise RuntimeError(f"XNG_PROTOCOL_FAIL: {row}")
    episodes = [[np.asarray(indices, dtype=np.int64) for indices in epoch] for epoch in manifest["epochs"]]
    return mean, std, norm_meta, {"episodes": episodes, "manifest_sha256": manifest_hash}, row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("XNG_PROTOCOL_FAIL: CUDA unavailable")
    search, folds_by_dataset, split_hash = base.load_folds()
    folds = {int(row["fold_id"]): row for row in folds_by_dataset["OpenBMI"]}
    provenance_path = PROTOCOL / "OPENBMI_MI_LITEBN_CHECKPOINT_PROVENANCE.json"
    existing = json.loads(provenance_path.read_text(encoding="utf-8"))["records"] if provenance_path.is_file() else []
    by_fold = {int(row["fold"]): row for row in existing}
    audit_rows = []
    for fold_id in range(5):
        fold = folds[fold_id]
        allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
        if set(allowed) & set(fold["outer_dev_subjects"]):
            raise RuntimeError("XNG_PROTOCOL_FAIL: outer subject in training bundle")
        bundle = base.build_bundle(TASK, allowed)
        mean, std, norm_meta, batch_info, audit = stored_inputs(fold)
        audit_rows.append(audit)
        base.write_csv(OUT / "OPENBMI_MI_LITEBN_MANIFEST_AUDIT.csv", pd.DataFrame(audit_rows).sort_values("fold"))
        cache = base.RawGPUCache(bundle, device)
        base.set_seed(SEED)
        model = base.LiteBN_BASELINE(62, 2).to(device)
        base.set_seed(SEED + 100_000)
        record = base.train_one(model, ARCH, TASK, fold, bundle, cache, mean, std, norm_meta, batch_info, None,
                                {"weighted_cross_entropy": False, "counts": None, "weights": None}, device)
        by_fold[fold_id] = record
        serial = [{key: value for key, value in by_fold[index].items() if key != "history"} for index in sorted(by_fold)]
        base.write_json(provenance_path, {"stage": "OPENBMI_MI_LITEBN_FIRST", "seed": 0, "split_sha256": split_hash, "records": serial})
        del model, cache, bundle
        gc.collect(); torch.cuda.empty_cache()
    records = [by_fold[index] for index in range(5)]
    base.write_json(PROTOCOL / "OPENBMI_MI_LITEBN_FIRST_LOCK.json", {
        "status": "PASS", "task": TASK, "seed": 0, "five_checkpoints_frozen": True,
        "xng_openbmi_mi_trained": False, "internal_heldout_accessed": False, "split_sha256": split_hash,
        "checkpoints": [{"fold": row["fold"], "path": row["checkpoint_path"], "sha256": row["checkpoint_sha256"]} for row in records],
    })
    rows = []
    for fold_id in range(5):
        fold, record = folds[fold_id], records[fold_id]
        checkpoint = Path(record["checkpoint_path"])
        if base.sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise RuntimeError("XNG_PROTOCOL_FAIL: baseline checkpoint hash mismatch")
        bundle = base.build_bundle(TASK, fold["outer_dev_subjects"])
        cache = base.RawGPUCache(bundle, device)
        mean, std, norm_meta = base.load_tensor_pair(ORIGINAL_RUNTIME / "normalizers" / f"openbmi_mi_fold{fold_id}.npz")
        base.set_seed(SEED)
        model = base.LiteBN_BASELINE(62, 2)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
        model.to(device)
        for subject, metrics in base.evaluate(model, bundle, cache, fold["outer_dev_subjects"], mean, std).items():
            rows.append({"task": TASK, "fold": fold_id, "subject_id": subject, "method": ARCH, **metrics,
                         "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm_meta["mean_std_sha256"]})
        del model, cache, bundle
        gc.collect(); torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["fold", "subject_id"])
    if len(frame) != 40 or frame.duplicated(["subject_id"]).any():
        raise RuntimeError(f"XNG_PROTOCOL_FAIL: OpenBMI-MI outer cardinality {len(frame)}/40")
    base.write_csv(OUT / "LITEBN_ONLY_OPENBMI_MI_OUTER_SUBJECT_RESULTS.csv", frame)
    fold_frame = frame.groupby("fold", as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), accuracy=("accuracy", "mean"), subjects=("subject_id", "count"))
    fold_frame.insert(1, "method", ARCH)
    fold_frame["selected_epoch"] = [records[index]["selected_epoch"] for index in range(5)]
    fold_frame["selected_inner_val_BA"] = [records[index]["best_inner_val_BA"] for index in range(5)]
    base.write_csv(OUT / "LITEBN_ONLY_OPENBMI_MI_OUTER_FOLD_RESULTS.csv", fold_frame)
    base.write_json(OUT / "OPENBMI_MI_LITEBN_FIRST_STATUS.json", {
        "status": "OPENBMI_MI_LITEBN_ONLY_COMPLETE", "seed": 0, "folds": 5, "outer_subjects": 40,
        "outer_subject_mean_BA": float(frame.BA.mean()), "outer_subject_mean_macro_F1": float(frame.macro_F1.mean()),
        "xng_openbmi_mi_trained": False, "comparison_evaluated": False, "internal_heldout_accessed": False,
    })
    print("OPENBMI_MI_LITEBN_ONLY_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
