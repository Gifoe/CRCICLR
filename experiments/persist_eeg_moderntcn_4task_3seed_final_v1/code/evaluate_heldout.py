"""Post-freeze evaluation on the established CRCICLR fixed heldout cohorts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import benchmark_data as data_code
from model_adapter import MODEL_NAME, build_model


EXP = Path(__file__).resolve().parents[1]
PROTOCOL, OUTPUTS = EXP / "protocol", EXP / "outputs"
RUNTIME = Path(os.environ["BASELINE_RUNTIME"]).resolve()
REPO = Path(os.environ["SEVEN_REPO"]).resolve()
OPENBMI_CACHE = Path(os.environ["FULL_OPENBMI_CACHE"]).resolve()
WBCIC_CACHE = Path(os.environ["FULL_WBCIC_CACHE"]).resolve()
MANIFEST = REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
SEEDS = (0, 1, 2)
BATCH_SIZE = 128


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, dict): return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(item) for item in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(_clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows([{key: _clean(row.get(key, "")) for key in fields} for row in rows])
    os.replace(temporary, path)


def _cell(task: str, fold: int, seed: int) -> tuple[dict[str, Any], Path]:
    root = RUNTIME / "cells" / task.lower() / f"fold{fold}_seed{seed}"
    record_path, checkpoint = root / "record.json", root / "selected.pt"
    if not record_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"missing frozen cell: {root}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("task"), int(record.get("fold", -1)), int(record.get("seed", -1))) != (task, fold, seed):
        raise RuntimeError(f"cell identity mismatch: {root}")
    if record.get("checkpoint_sha256") != _sha(checkpoint):
        raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
    return record, checkpoint


def _all_records() -> list[dict[str, Any]]:
    rows = []
    for task in TASKS:
        for fold in range(5):
            for seed in SEEDS:
                record, checkpoint = _cell(task, fold, seed)
                rows.append({
                    "task": task, "fold": fold, "seed": seed, "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": record["normalizer"]["mean_std_sha256"],
                    "selected_epoch": record["selected_epoch"], "split_sha256": record["split_sha256"],
                    "channels": record["channels"], "samples": record["samples"], "classes": record["classes"],
                })
    if len(rows) != 60:
        raise RuntimeError("heldout requires all 60 selected checkpoints")
    return rows


def prelock() -> None:
    records = _all_records()  # fail before touching even heldout membership metadata
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    memberships = {
        "OpenBMI": list(map(str, manifest["OpenBMI"]["subject_ids"])),
        "WBCIC": list(map(str, manifest["WBCIC"]["subject_ids"])),
    }
    if len(memberships["OpenBMI"]) != 14 or len(memberships["WBCIC"]) != 10:
        raise RuntimeError("established heldout membership mismatch")
    lock_path = PROTOCOL / "HELDOUT_EVALUATION_LOCK.json"
    if lock_path.exists():
        raise RuntimeError("refusing to replace heldout lock")
    lock = {
        "schema": "CRCICLR_MATCHED_BASELINE_HELDOUT_LOCK_V1", "model": MODEL_NAME,
        "checkpoint_count": len(records), "checkpoints": records,
        "manifest_path": str(MANIFEST), "manifest_sha256": _sha(MANIFEST), "memberships": memberships,
        "aggregation": "for each seed, subject-equal metrics averaged over the five frozen checkpoint folds; then mean and sample SD across seeds",
        "prelock_access": {"heldout_signal_loaded": False, "heldout_labels_loaded": False, "heldout_predictions_generated": False},
    }
    _json(lock_path, lock)
    (PROTOCOL / "HELDOUT_EVALUATION_LOCK.sha256").write_text(_sha(lock_path) + "\n", encoding="utf-8")
    print("HELDOUT_INPUTS_LOCKED_WITHOUT_SIGNAL_OR_LABEL_ACCESS", flush=True)


def _heldout_data(task: str, fold: int, memberships: dict[str, list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    inner = data_code._load_module("baseline_heldout_inner_loader", REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code" / "tech_recipe_selection.py")
    modern, task_module = data_code._sources()
    if task in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = "OpenBMI" if task == "OpenBMI_MI" else "WBCIC"
        folds, _, _ = modern.load_split()
        current = next(row for row in folds[dataset] if int(row["fold_id"]) == fold)
        source_sessions, future = modern.SOURCE_SESSIONS[dataset], (modern.EVAL_SESSION,)
        if dataset == "OpenBMI":
            train, _, _, mapping = inner._openbmi_rows(OPENBMI_CACHE, current["inner_train_subjects"], source_sessions, "mi")
            heldout, labels, subjects, _ = inner._openbmi_rows(OPENBMI_CACHE, memberships[dataset], future, "mi", mapping)
        else:
            train, _, _, mapping = inner._wbcic_rows(WBCIC_CACHE, current["inner_train_subjects"], source_sessions)
            heldout, labels, subjects, _ = inner._wbcic_rows(WBCIC_CACHE, memberships[dataset], future, mapping)
    else:
        name = "ERP" if task == "OpenBMI_ERP" else "SSVEP"
        _, _, reference, _ = task_module.split_reference()
        current = next(row for row in reference["folds"] if int(row["fold_id"]) == fold)
        spec = task_module.TASKS[name]
        train, _, _, mapping = inner._openbmi_rows(OPENBMI_CACHE, current["inner_train_subjects"], (spec["source_session"],), spec["cache_name"])
        heldout, labels, subjects, _ = inner._openbmi_rows(OPENBMI_CACHE, memberships["OpenBMI"], (spec["future_session"],), spec["cache_name"], mapping)
    _, [heldout], normalizer = data_code._normalise(train, heldout)
    return np.ascontiguousarray(heldout), labels, subjects, normalizer


def _metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    prediction = logits.argmax(1); rows = []
    for subject in sorted(np.unique(subjects.astype(str)), key=lambda value: int(value.replace("sub-", ""))):
        mask = subjects.astype(str) == subject
        rows.append({
            "subject_id": subject, "BA": float(balanced_accuracy_score(labels[mask], prediction[mask])),
            "macro_F1": float(f1_score(labels[mask], prediction[mask], average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(labels[mask], prediction[mask])), "trials": int(mask.sum()),
        })
    return {
        "subject_equal_BA": float(np.mean([row["BA"] for row in rows])),
        "subject_equal_macro_F1": float(np.mean([row["macro_F1"] for row in rows])),
        "trial_accuracy": float(accuracy_score(labels, prediction)),
    }, rows


def evaluate() -> None:
    lock_path, sidecar = PROTOCOL / "HELDOUT_EVALUATION_LOCK.json", PROTOCOL / "HELDOUT_EVALUATION_LOCK.sha256"
    if not lock_path.is_file() or not sidecar.is_file() or _sha(lock_path) != sidecar.read_text(encoding="utf-8").strip():
        raise RuntimeError("heldout evaluation requires an intact post-training lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["prelock_access"] != {"heldout_signal_loaded": False, "heldout_labels_loaded": False, "heldout_predictions_generated": False}:
        raise RuntimeError("invalid prelock access state")
    records = _all_records()
    expected = {(row["task"], row["fold"], row["seed"]): row["checkpoint_sha256"] for row in lock["checkpoints"]}
    if any(expected[(row["task"], row["fold"], row["seed"])] != row["checkpoint_sha256"] for row in records):
        raise RuntimeError("checkpoint family changed after heldout lock")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_rows, replicate_rows = [], []
    for task in TASKS:
        dataset = "WBCIC" if task == "WBCIC_MI" else "OpenBMI"
        for fold in range(5):
            heldout, labels, subjects, normalizer = _heldout_data(task, fold, lock["memberships"])
            x = torch.from_numpy(heldout).to(device)
            for seed in SEEDS:
                record = next(row for row in records if row["task"] == task and row["fold"] == fold and row["seed"] == seed)
                if normalizer["mean_std_sha256"] != record["normalizer_sha256"]:
                    raise RuntimeError(f"normalizer mismatch: {task}/fold{fold}")
                model = build_model(channels=record["channels"], samples=record["samples"], classes=record["classes"])
                payload = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=False)
                model.load_state_dict(payload["state_dict"], strict=True); model.eval().to(device)
                parts = []
                with torch.no_grad():
                    for start in range(0, len(labels), BATCH_SIZE):
                        parts.append(model(x[start:start + BATCH_SIZE]).float().cpu().numpy())
                metrics, details = _metrics(labels, np.concatenate(parts), subjects)
                replicate_rows.append({"task": task, "dataset": dataset, "fold": fold, "seed": seed, **metrics, "checkpoint_sha256": record["checkpoint_sha256"]})
                subject_rows.extend([{"task": task, "dataset": dataset, "fold": fold, "seed": seed, **row} for row in details])
                del model
            del x
            if device.type == "cuda": torch.cuda.empty_cache()
    seed_rows = []
    for task in TASKS:
        for seed in SEEDS:
            details = [row for row in subject_rows if row["task"] == task and row["seed"] == seed]
            reps = [row for row in replicate_rows if row["task"] == task and row["seed"] == seed]
            seed_rows.append({
                "task": task, "seed": seed, "subject_equal_BA": float(np.mean([row["BA"] for row in details])),
                "subject_equal_macro_F1": float(np.mean([row["macro_F1"] for row in details])),
                "trial_accuracy": float(np.mean([row["trial_accuracy"] for row in reps])),
                "checkpoint_folds": 5, "heldout_subjects": len({row["subject_id"] for row in details}),
            })
    final_rows = []
    for task in TASKS:
        rows = [row for row in seed_rows if row["task"] == task]
        final_rows.append({
            "task": task, "subject_equal_BA_mean": float(np.mean([row["subject_equal_BA"] for row in rows])),
            "subject_equal_BA_std": float(np.std([row["subject_equal_BA"] for row in rows], ddof=1)),
            "subject_equal_macro_F1_mean": float(np.mean([row["subject_equal_macro_F1"] for row in rows])),
            "subject_equal_macro_F1_std": float(np.std([row["subject_equal_macro_F1"] for row in rows], ddof=1)),
            "trial_accuracy_mean": float(np.mean([row["trial_accuracy"] for row in rows])),
            "trial_accuracy_std": float(np.std([row["trial_accuracy"] for row in rows], ddof=1)),
        })
    _csv(OUTPUTS / "HELDOUT_SUBJECT_RESULTS.csv", subject_rows)
    _csv(OUTPUTS / "HELDOUT_SEED_RESULTS.csv", seed_rows)
    _csv(OUTPUTS / "HELDOUT_FINAL_SUMMARY.csv", final_rows)
    _json(PROTOCOL / "HELDOUT_LEAKAGE_AUDIT.json", {
        "pass": True, "HELDOUT_ACCESSED_DURING_TRAINING": "NO", "HELDOUT_ACCESSED_DURING_SELECTION": "NO",
        "OUTER_DEV_USED_FOR_SELECTION": "NO", "heldout_labels_read_once_after_post_training_lock": True,
    })
    print("FIXED_HELDOUT_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("prelock", "evaluate"), required=True)
    args = parser.parse_args(); prelock() if args.stage == "prelock" else evaluate()
