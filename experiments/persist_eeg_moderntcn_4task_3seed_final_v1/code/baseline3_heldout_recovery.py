"""Crash-isolated, resume-safe heldout evaluation for the baseline3 models.

This is an engineering-only wrapper around each experiment's frozen
``evaluate_heldout.py`` implementation.  It evaluates exactly one
task/fold/seed checkpoint per Python process, then combines the resulting
shards with the original aggregation semantics.  Model definitions, frozen
checkpoints, normalizers, splits, heldout membership and metrics all come
from the target experiment module unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_module(code_dir: Path):
    sys.path.insert(0, str(code_dir))
    source = code_dir / "evaluate_heldout.py"
    spec = importlib.util.spec_from_file_location("baseline3_target_evaluate_heldout", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _lock(module):
    lock_path, sidecar, _ = module._artifact_paths(False)
    if not lock_path.is_file() or not sidecar.is_file():
        raise RuntimeError("heldout evaluation requires the frozen lock")
    lock_sha = module._sha(lock_path)
    if lock_sha != sidecar.read_text(encoding="utf-8").strip():
        raise RuntimeError("heldout lock sidecar mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_access = {
        "heldout_signal_loaded": False,
        "heldout_labels_loaded": False,
        "heldout_predictions_generated": False,
    }
    if lock.get("prelock_access") != expected_access:
        raise RuntimeError("invalid prelock access state")
    return lock_path, lock_sha, lock


def _evaluate_cell(module, shard_root: Path, task: str, fold: int, seed: int, cpu: bool) -> None:
    import torch

    lock_path, lock_sha, lock = _lock(module)
    record, checkpoint = module._cell(task, fold, seed)
    expected = next(
        row for row in lock["checkpoints"]
        if row["task"] == task and int(row["fold"]) == fold and int(row["seed"]) == seed
    )
    if expected["checkpoint_sha256"] != record["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed after heldout lock")

    shard = shard_root / task.lower() / f"fold{fold}_seed{seed}.json"
    if shard.is_file():
        prior = json.loads(shard.read_text(encoding="utf-8"))
        if (
            prior.get("lock_sha256") == lock_sha
            and prior.get("checkpoint_sha256") == record["checkpoint_sha256"]
            and prior.get("task") == task
            and int(prior.get("fold", -1)) == fold
            and int(prior.get("seed", -1)) == seed
        ):
            print(f"HELDOUT_SHARD_REUSED {task} fold={fold} seed={seed}", flush=True)
            return
        raise RuntimeError(f"refusing incompatible existing shard: {shard}")

    heldout, labels, subjects, normalizer = module._heldout_data(task, fold, lock["memberships"])
    if normalizer["mean_std_sha256"] != record["normalizer"]["mean_std_sha256"]:
        raise RuntimeError(f"normalizer mismatch: {task}/fold{fold}")
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    x = torch.from_numpy(heldout).to(device)
    model = module.build_model(
        channels=record["channels"], samples=record["samples"], classes=record["classes"]
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    parts = []
    with torch.no_grad():
        for start in range(0, len(labels), module.BATCH_SIZE):
            parts.append(model(x[start:start + module.BATCH_SIZE]).float().cpu().numpy())
    metrics, details = module._metrics(labels, np.concatenate(parts), subjects)
    dataset = "WBCIC" if task == "WBCIC_MI" else "OpenBMI"
    replicate = {
        "task": task, "dataset": dataset, "fold": fold, "seed": seed,
        **metrics, "checkpoint_sha256": record["checkpoint_sha256"],
    }
    subject_rows = [
        {"task": task, "dataset": dataset, "fold": fold, "seed": seed, **row}
        for row in details
    ]
    _write_json(shard, {
        "schema": "BASELINE3_CRASH_ISOLATED_HELDOUT_SHARD_V1",
        "model": module.MODEL_NAME,
        "task": task, "fold": fold, "seed": seed,
        "lock_path": str(lock_path), "lock_sha256": lock_sha,
        "checkpoint_sha256": record["checkpoint_sha256"],
        "device": str(device), "replicate": replicate, "subjects": subject_rows,
    })
    print(f"HELDOUT_SHARD_COMPLETE {task} fold={fold} seed={seed} device={device}", flush=True)


def _aggregate(module, shard_root: Path) -> None:
    _, lock_sha, _ = _lock(module)
    subject_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    expected_keys = {(task, fold, seed) for task in module.TASKS for fold in range(5) for seed in module.SEEDS}
    seen = set()
    for task, fold, seed in sorted(expected_keys):
        shard = shard_root / task.lower() / f"fold{fold}_seed{seed}.json"
        if not shard.is_file():
            raise FileNotFoundError(f"missing heldout shard: {shard}")
        row = json.loads(shard.read_text(encoding="utf-8"))
        key = (row["task"], int(row["fold"]), int(row["seed"]))
        if key != (task, fold, seed) or row.get("lock_sha256") != lock_sha:
            raise RuntimeError(f"heldout shard identity mismatch: {shard}")
        seen.add(key)
        replicate_rows.append(row["replicate"])
        subject_rows.extend(row["subjects"])
    if seen != expected_keys:
        raise RuntimeError("heldout shard matrix is incomplete")

    seed_rows = []
    for task in module.TASKS:
        for seed in module.SEEDS:
            details = [row for row in subject_rows if row["task"] == task and row["seed"] == seed]
            reps = [row for row in replicate_rows if row["task"] == task and row["seed"] == seed]
            seed_rows.append({
                "task": task, "seed": seed,
                "subject_equal_BA": float(np.mean([row["BA"] for row in details])),
                "subject_equal_macro_F1": float(np.mean([row["macro_F1"] for row in details])),
                "trial_accuracy": float(np.mean([row["trial_accuracy"] for row in reps])),
                "checkpoint_folds": 5,
                "heldout_subjects": len({row["subject_id"] for row in details}),
            })
    final_rows = []
    for task in module.TASKS:
        rows = [row for row in seed_rows if row["task"] == task]
        final_rows.append({
            "task": task,
            "subject_equal_BA_mean": float(np.mean([row["subject_equal_BA"] for row in rows])),
            "subject_equal_BA_std": float(np.std([row["subject_equal_BA"] for row in rows], ddof=1)),
            "subject_equal_macro_F1_mean": float(np.mean([row["subject_equal_macro_F1"] for row in rows])),
            "subject_equal_macro_F1_std": float(np.std([row["subject_equal_macro_F1"] for row in rows], ddof=1)),
            "trial_accuracy_mean": float(np.mean([row["trial_accuracy"] for row in rows])),
            "trial_accuracy_std": float(np.std([row["trial_accuracy"] for row in rows], ddof=1)),
        })
    module._csv(module.OUTPUTS / "HELDOUT_SUBJECT_RESULTS.csv", subject_rows)
    module._csv(module.OUTPUTS / "HELDOUT_SEED_RESULTS.csv", seed_rows)
    module._csv(module.OUTPUTS / "HELDOUT_FINAL_SUMMARY.csv", final_rows)
    module._json(module.PROTOCOL / "HELDOUT_LEAKAGE_AUDIT.json", {
        "pass": True,
        "HELDOUT_ACCESSED_DURING_TRAINING": "NO",
        "HELDOUT_ACCESSED_DURING_SELECTION": "NO",
        "OUTER_DEV_USED_FOR_SELECTION": "NO",
        "heldout_labels_read_once_after_post_training_lock": False,
        "heldout_labels_read_only_after_post_training_lock": True,
        "heldout_labels_read_in_independent_evaluation_processes": len(expected_keys),
        "evaluation_scope": "three_seed_final",
        "engineering_execution": "one frozen checkpoint per crash-isolated process; exact original aggregation",
    })
    print("FIXED_HELDOUT_EVALUATION_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("cell", "aggregate"), required=True)
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    module = _load_module(args.code_dir.resolve())
    if args.stage == "cell":
        if args.task is None or args.fold not in range(5) or args.seed not in range(3):
            parser.error("cell stage requires --task, --fold 0..4, --seed 0..2")
        _evaluate_cell(module, args.shard_root.resolve(), args.task, args.fold, args.seed, args.cpu)
    else:
        _aggregate(module, args.shard_root.resolve())


if __name__ == "__main__":
    main()
