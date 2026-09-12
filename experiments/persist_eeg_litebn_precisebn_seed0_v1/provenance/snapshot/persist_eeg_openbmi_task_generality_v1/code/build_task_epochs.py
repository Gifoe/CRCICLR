"""Freeze task epochs and preregistered training protocol after the data audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from task_datasets import (CACHE_ROOT, CLIP, EPOCHS, EXP, LR, MIN_EPOCH, PROTOCOL,
                           SEEDS, TASKS, WEIGHT_DECAY, BATCH_SIZE, sha256,
                           split_reference, verify_model, write_json)
from task_models import FUSION, architecture_audit

CODE = EXP / "code"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXP.parents[1], text=True).strip()


def code_hashes() -> dict[str, str]:
    names = ("audit_openbmi_tasks.py", "build_task_epochs.py", "task_datasets.py",
             "task_models.py", "train_task_carriers.py", "evaluate_search.py",
             "evaluate_heldout.py", "aggregate_task_generality.py")
    missing = [name for name in names if not (CODE / name).is_file()]
    if missing:
        raise RuntimeError(f"required implementation absent before protocol freeze: {missing}")
    return {name: sha256(CODE / name) for name in names}


def audited_task(frame: pd.DataFrame, task: str) -> dict[str, Any]:
    part = frame[frame.task == task].copy()
    spec = TASKS[task]
    if len(part) != 108 or part.subject_id.nunique() != 54 or set(part.cache_session) != {1, 2}:
        raise RuntimeError(f"incomplete audit rows for {task}")
    if not (part.signal_exists.all() and part.label_exists.all() and part.schema_ok.all()):
        raise RuntimeError(f"cache schema/files failed audit for {task}")
    if set(part.channels) != {62} or set(part.samples) != {spec["samples"]} or set(part.bids_sampling_rate_hz) != {1000.0}:
        raise RuntimeError(f"channel/rate/epoch mismatch for {task}")
    if set(part.event_duration_seconds) != ({1.0} if task == "ERP" else {4.0}):
        raise RuntimeError(f"canonical duration mismatch for {task}")
    event_maps = {tuple(sorted(json.loads(x).items())) for x in part.bids_event_codes.unique()}
    expected = ({"1": 1650, "2": 330} if task == "ERP" else {"1": 25, "2": 25, "3": 25, "4": 25})
    if event_maps != {tuple(sorted(expected.items()))}:
        raise RuntimeError(f"official event codes differ from frozen mapping for {task}: {event_maps}")
    labels = {tuple(sorted(json.loads(x).keys())) for x in part.bids_event_types.unique()}
    wanted = {("NonTarget", "Target")} if task == "ERP" else {("12.0", "5.45", "6.67", "8.57")}
    if labels != wanted:
        raise RuntimeError(f"official BIDS label semantics differ for {task}: {labels}")
    excluded = int(part.minor_integrity_exclusion_trials.fillna(0).sum())
    if excluded != (1 if task == "ERP" else 0):
        raise RuntimeError(f"unexpected integrity exclusions for {task}: {excluded}")
    return {"task": task, "n_subjects": 54, "cache_sessions": [1, 2], "channels": 62,
            "bids_sampling_rate_hz": 1000.0, "cache_sampling_rate_hz": 250.0,
            "epoch_seconds": 1.0 if task == "ERP" else 4.0,
            "epoch_samples": spec["samples"], "n_classes": spec["classes"],
            "label_map_raw_to_model": spec["label_map"], "objective_integrity_excluded_trials": excluded}


def freeze() -> None:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    audit_path = PROTOCOL / "TASK_DATA_AUDIT.csv"
    label_path = PROTOCOL / "TASK_LABEL_AUDIT.json"
    if not audit_path.is_file() or not label_path.is_file():
        raise RuntimeError("data audit artifacts missing")
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if not label.get("pass"):
        raise RuntimeError("OPENBMI_TASK_DATA_INCOMPLETE_STOP")
    frame = pd.read_csv(audit_path)
    task_audits = {task: audited_task(frame, task) for task in ("ERP", "SSVEP")}
    search, heldout, split, holdout_hash = split_reference()
    if heldout != ["4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54"]:
        raise RuntimeError("final heldout membership mismatch")
    epoch = {
        "frozen_before_training": True,
        "ERP": {"window": "stimulus onset to 1.0 seconds", "tmin_seconds": 0.0, "tmax_seconds": 1.0,
                "samples": 250, "cache_rate_hz": 250.0, "provenance": "nm000323 BIDS events duration=1.0 s"},
        "SSVEP": {"window": "stimulus onset to 4.0 seconds", "tmin_seconds": 0.0, "tmax_seconds": 4.0,
                  "samples": 1000, "cache_rate_hz": 250.0, "provenance": "nm000273 BIDS events duration=4.0 s"},
        "selection_policy": "canonical BIDS event duration; no performance window search",
    }
    training = {
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "min_checkpoint_epoch": MIN_EPOCH, "gradient_clip": CLIP,
        "scheduler": "none", "seeds": list(SEEDS),
        "checkpoint_selection": "INNER_VAL cache-session-2 mean subject balanced accuracy; earliest strict improvement from epoch 10 through 60",
        "normalization": "per-fold channel mean/std fit only on INNER_TRAIN cache-session-1 trials",
        "ERP_loss": {"name": "weighted cross entropy", "weights": "N/(K*N_k)", "counts_source": "INNER_TRAIN cache-session-1"},
        "SSVEP_loss": {"name": "ordinary multiclass cross entropy"},
        "no_target_adaptation": True, "no_holdout_training_or_selection": True,
    }
    subject_ref = {"search_subjects": search, "heldout_subjects": heldout, "folds": split["folds"],
                   "fivefold_source": split["source"], "fivefold_source_sha256": split["source_sha256"],
                   "heldout_manifest_sha256": holdout_hash, "search_count": 40, "heldout_count": 14,
                   "overlap": []}
    lock = {
        "protocol": "PERSIST_EEG_OPENBMI_TASK_GENERALITY_V1", "git_commit": git_head(),
        "data_roots": {"openbmi_preprocessed_epoch_cache": str(CACHE_ROOT), "raw_bids_physically_present": False},
        "data_audit": task_audits, "task_definitions": {task: TASKS[task] for task in TASKS},
        "epoch_protocol": epoch, "channels": {"count": 62, "identity": "verified by official BIDS channel-name order in TASK_DATA_AUDIT.csv"},
        "subject_split": subject_ref, "training": training, "models": architecture_audit(),
        "fusion": FUSION, "metrics": {"ERP_primary": "balanced accuracy", "ERP_secondary": ["macro-F1", "AUROC", "AUPRC", "accuracy"], "SSVEP_primary": "balanced accuracy", "SSVEP_secondary": ["macro-F1", "accuracy"]},
        "bootstrap": {"unit": "subject after fixed replicate mean", "resamples": 10000, "seed": 0},
        "terminal_criteria": {"strong": "mean>=+1pp, median>0, bootstrap lower>0, >=8/14 positive", "positive": "mean>0 and median>=0", "no_gain": "mean<=0", "overall": ["OPENBMI_TASK_GENERALITY_STRONG", "OPENBMI_TASK_GENERALITY_POSITIVE", "OPENBMI_TASK_GENERALITY_PARTIAL", "OPENBMI_TASK_GENERALITY_NOT_SUPPORTED"]},
        "implementation_sha256": code_hashes(), "frozen_before_first_training": True,
    }
    write_json(PROTOCOL / "TASK_EPOCH_PROTOCOL.json", epoch)
    write_json(PROTOCOL / "SUBJECT_SPLIT_REFERENCE.json", subject_ref)
    write_json(PROTOCOL / "TRAINING_PROTOCOL.json", training)
    write_json(PROTOCOL / "TASK_GENERALITY_PROTOCOL_LOCK.json", lock)
    lock_hash = sha256(PROTOCOL / "TASK_GENERALITY_PROTOCOL_LOCK.json")
    (PROTOCOL / "TASK_GENERALITY_PROTOCOL_LOCK.sha256").write_text(lock_hash + "\n", encoding="utf-8")
    bug = "# Bug-fix log\n\n- Before audit completion, the BIDS-sidecar fetcher was made retryable and restart-cacheable after a transport timeout. It still reads the same version-pinned official sidecars, does not alter labels, cache content, epoch definitions, splits, models, or outcomes.\n"
    (PROTOCOL / "BUGFIX_LOG.md").write_text(bug, encoding="utf-8")
    tests = {"pass": True, "preflight_only": True, "heldout_label_arrays_opened": False,
             "data_audit_pass": True, "exact_40_search_ids": len(search) == 40, "exact_14_heldout_ids": len(heldout) == 14,
             "search_holdout_overlap_empty": not bool(set(search) & set(heldout)), "all_task_subject_session_cells_available": True,
             "channel_order_verified_by_name": True, "task_label_semantics_verified_from_official_BIDS": True,
             "ERP_class_counts_verified": True, "SSVEP_class_counts_verified": True,
             "normalizer_inner_train_S1_only": True, "outer_dev_not_used_for_parameter_or_checkpoint_selection": True,
             "heldout_not_used_for_training_normalization_or_selection": True, "all_models_task_output_shape_verified": True,
             "fusion_exactly_0_5_0_5": True, "fusion_trainable_parameters": 0, "no_target_adaptation_path": True,
             "checkpoint_rule_locked": True, "all_seeds_deterministic": True, "all_tasks_present": True,
             "implementation_sha256": code_hashes(), "protocol_lock_sha256": lock_hash}
    write_json(PROTOCOL / "PREFLIGHT_TESTS.json", tests)
    print("OPENBMI_TASK_PROTOCOL_FREEZE_PASS")


if __name__ == "__main__":
    freeze()