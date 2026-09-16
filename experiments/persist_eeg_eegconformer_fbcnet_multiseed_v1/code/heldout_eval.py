"""Post-training, hash-locked all-session inference on final true holdouts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from filterbank import FILTER_SHA256, make_memmap
from models import build_model
from train_queue import (EXP, FOLDS, MODELS, PROTOCOL, REPO, RUNTIME, SCRATCH,
                         SEEDS, TASKS, cell_complete, cell_dir, sha, atomic_json)


LOCK = EXP / "protocol" / "HELDOUT_EVALUATION_LOCK.json"
SIDECAR = LOCK.with_suffix(".sha256")
CSGD_SOURCE = (Path(r"D:\nips-temp\TotalP\P1\CRCICLR_CROSSBACKBONE_CSGD_WORK") /
               "experiments" / "persist_eeg_crossbackbone_csgd_v1" / "code" /
               "run_crossbackbone_csgd.py")
EXPECTED_WBCIC = {"sub-4", "sub-8", "sub-10", "sub-15", "sub-20",
                  "sub-39", "sub-40", "sub-43", "sub-46", "sub-51"}


def csgd_module():
    os.environ["SEVEN_REPO"] = str(REPO)
    spec = importlib.util.spec_from_file_location("new_baseline_frozen_loader", CSGD_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(CSGD_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def all_records() -> list[dict]:
    rows = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                for seed in SEEDS:
                    if not cell_complete(model, task, fold, seed):
                        raise RuntimeError(f"all 120 frozen checkpoints required before heldout: {model} {task} {fold} {seed}")
                    root = cell_dir(model, task, fold, seed)
                    record = json.loads((root / "record.json").read_text(encoding="utf-8"))
                    rows.append({"model": model, "task": task, "fold": fold, "seed": seed,
                                 "checkpoint_path": str(root / "selected.pt"),
                                 "checkpoint_sha256": record["checkpoint_sha256"],
                                 "normalizer_sha256": record["normalizer"]["mean_std_sha256"],
                                 "split_sha256": record["split_sha256"],
                                 "channels": record["channels"], "samples": record["samples"],
                                 "classes": record["classes"], "selected_epoch": record["selected_epoch"]})
    if len(rows) != 120:
        raise RuntimeError("checkpoint count is not 120")
    return rows


def prelock() -> None:
    if LOCK.exists():
        raise RuntimeError("refusing to replace existing heldout lock")
    rows = all_records()  # this executes before reading heldout membership
    loader = csgd_module()
    manifest = json.loads(loader.HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    open_subjects = list(map(str, manifest["OpenBMI"]["subject_ids"]))
    wbcic = list(loader.TRUE_WBCIC_SUBJECTS)
    if len(open_subjects) != 14 or len(set(open_subjects)) != 14:
        raise RuntimeError("OpenBMI final heldout membership mismatch")
    if set(wbcic) != EXPECTED_WBCIC or len(wbcic) != 10:
        raise RuntimeError("WBCIC must use final true-outer 10, not V8 internal")
    for subject in wbcic:
        for session in (0, 1, 2):
            for suffix in ("epochs.npy", "labels.npy", "metadata.json"):
                if not (loader.TRUE_WBCIC_CACHE / subject / f"ses-{session}_{suffix}").is_file():
                    raise FileNotFoundError(f"missing true-outer cache {subject} ses-{session} {suffix}")
    lock = {"schema": "EEGCONFORMER_FBCNET_TRUE_OUTER_LOCK_V1",
            "checkpoint_count": 120, "checkpoints": rows,
            "protocol_sha256": sha(PROTOCOL),
            "frozen_loader_sha256": sha(CSGD_SOURCE),
            "manifest_sha256": sha(loader.HOLDOUT_MANIFEST),
            "cohorts": {"OpenBMI": open_subjects, "WBCIC": wbcic},
            "session_file_to_paper": {"OpenBMI": {"S1": "S1", "S2": "S2"},
                                      "WBCIC": {"S0": "S1", "S1": "S2", "S2": "S3"}},
            "filterbank_sha256": FILTER_SHA256,
            "heldout_signal_read_before_lock": False,
            "heldout_label_read_before_lock": False,
            "heldout_used_for_training": False,
            "heldout_used_for_selection": False}
    atomic_json(LOCK, lock)
    SIDECAR.write_text(sha(LOCK) + "\n", encoding="utf-8")
    print("HELDOUT_PRELOCK_COMPLETE checkpoints=120", flush=True)


def locked() -> tuple[dict, object]:
    if not LOCK.is_file() or not SIDECAR.is_file() or sha(LOCK) != SIDECAR.read_text().strip():
        raise RuntimeError("heldout inference requires intact pre-evaluation lock")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if len(lock["checkpoints"]) != 120 or lock["protocol_sha256"] != sha(PROTOCOL):
        raise RuntimeError("heldout checkpoint/protocol lock changed")
    if lock["frozen_loader_sha256"] != sha(CSGD_SOURCE):
        raise RuntimeError("frozen loader changed")
    current = {(r["model"], r["task"], r["fold"], r["seed"]): r
               for r in all_records()}
    for row in lock["checkpoints"]:
        key = row["model"], row["task"], row["fold"], row["seed"]
        if current[key] != row:
            raise RuntimeError(f"checkpoint changed after lock: {key}")
    loader = csgd_module()
    if lock["manifest_sha256"] != sha(loader.HOLDOUT_MANIFEST):
        raise RuntimeError("heldout cohort manifest changed")
    return lock, loader


def cell_output(model: str, task: str, fold: int, seed: int, session: str) -> Path:
    return RUNTIME / "heldout_sessions" / model.lower() / task.lower() / f"fold{fold}_seed{seed}_{session}.json"


def infer(model: torch.nn.Module, values, labels: np.ndarray, subjects: np.ndarray,
          device: torch.device, batch_size: int = 128) -> list[dict]:
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            tensor = torch.from_numpy(np.array(values[start:start + batch_size], copy=True, order="C")).to(device)
            parts.append(model(tensor).float().cpu().numpy())
    predictions = np.concatenate(parts).argmax(axis=1)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    rows = []
    for subject in sorted(set(map(str, subjects)), key=lambda x: int(x.replace("sub-", ""))):
        mask = subjects.astype(str) == subject
        rows.append({"subject_id": subject,
                     "BA": float(balanced_accuracy_score(labels[mask], predictions[mask])),
                     "macro_F1": float(f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)),
                     "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
                     "trials": int(mask.sum())})
    return rows


def run() -> None:
    lock, loader = locked()
    torch.set_num_threads(int(os.environ.get("NEW_BASELINE_CPU_THREADS", "16")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_key = {(r["model"], r["task"], r["fold"], r["seed"]): r for r in lock["checkpoints"]}
    for task in TASKS:
        dataset = "WBCIC" if task == "WBCIC_MI" else "OpenBMI"
        for fold in FOLDS:
            plan = loader.fold_plan(task, fold, lock["cohorts"]["OpenBMI"])
            for model_name in MODELS:
                for seed in SEEDS:
                    row = by_key[model_name, task, fold, seed]
                    if row["normalizer_sha256"] != plan["normalizer"]["mean_std_sha256"]:
                        raise RuntimeError(f"normalizer mismatch: {model_name} {task} f{fold}")
                    if row["split_sha256"] != plan["split_sha256"]:
                        raise RuntimeError(f"split mismatch: {model_name} {task} f{fold}")
            for session_index in plan["sessions"]:
                paper_session = lock["session_file_to_paper"][dataset][f"S{session_index}"]
                pending = []
                for model in MODELS:
                    for seed in SEEDS:
                        output = cell_output(model, task, fold, seed, paper_session)
                        if output.is_file():
                            found = json.loads(output.read_text(encoding="utf-8"))
                            expected_identity = {"model": model, "task": task, "fold": fold,
                                                 "seed": seed, "session": paper_session}
                            record = by_key[model, task, fold, seed]
                            if (found.get("identity") != expected_identity or
                                found.get("checkpoint_sha256") != record["checkpoint_sha256"] or
                                found.get("normalizer_sha256") != record["normalizer_sha256"] or
                                len(found.get("subject_rows", [])) != len(lock["cohorts"][dataset])):
                                raise RuntimeError(f"stale heldout session cache: {output}")
                        else:
                            pending.append((model, seed))
                if not pending:
                    continue
                raw, labels, subjects = loader.evaluation_session(plan, session_index)
                if set(map(str, subjects)) != set(lock["cohorts"][dataset]):
                    raise RuntimeError("heldout subject set changed")
                for model_name in MODELS:
                    seeds = [seed for model, seed in pending if model == model_name]
                    if not seeds:
                        continue
                    if model_name == "FBCNet":
                        destination = SCRATCH / "heldout" / task.lower() / f"fold{fold}" / f"{paper_session}.npy"
                        values = make_memmap(raw, destination,
                                             {"task": task, "fold": fold, "session": paper_session,
                                              "normalizer_sha256": plan["normalizer"]["mean_std_sha256"],
                                              "cohort_sha256": lock["manifest_sha256"] if dataset == "OpenBMI" else
                                              hashlib.sha256("|".join(lock["cohorts"]["WBCIC"]).encode()).hexdigest()})
                    else:
                        values = raw
                    for seed in seeds:
                        row = by_key[model_name, task, fold, seed]
                        model = build_model(model_name, row["channels"], row["samples"], row["classes"])
                        payload = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=False)
                        model.load_state_dict(payload["state_dict"], strict=True)
                        model.eval().to(device)
                        details = infer(model, values, labels, subjects, device)
                        atomic_json(cell_output(model_name, task, fold, seed, paper_session),
                                    {"identity": {"model": model_name, "task": task, "fold": fold,
                                                  "seed": seed, "session": paper_session},
                                     "checkpoint_sha256": row["checkpoint_sha256"],
                                     "normalizer_sha256": row["normalizer_sha256"],
                                     "subject_rows": details,
                                     "eval_mode": True, "training_performed": False,
                                     "selection_performed": False})
                        print(f"HELDOUT_SESSION_COMPLETE {model_name} {task} f{fold} s{seed} {paper_session}", flush=True)
                        del model
                        torch.cuda.empty_cache()
                    if model_name == "FBCNet":
                        del values
                        for path in (destination, destination.with_suffix(".json")):
                            if path.is_file():
                                path.unlink()
                del raw, labels, subjects
    expected = [cell_output(model, task, fold, seed, session)
                for model in MODELS for task in TASKS for fold in FOLDS for seed in SEEDS
                for session in (("S1", "S2", "S3") if task == "WBCIC_MI" else ("S1", "S2"))]
    if len(expected) != 270 or not all(path.is_file() for path in expected):
        raise RuntimeError("heldout session matrix incomplete")
    atomic_json(RUNTIME / "HELDOUT_COMPLETED.json", {"session_cells": len(expected), "models": list(MODELS)})
    print("ALL_TRUE_OUTER_SESSION_CELLS_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prelock", "run"))
    args = parser.parse_args()
    prelock() if args.stage == "prelock" else run()
