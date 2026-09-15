"""Frozen cross-backbone cross-session generalization-drop (CSGD) audit.

The runner is inference-only.  It pre-locks exact selected checkpoint hashes
before loading evaluation arrays or labels, recomputes the original train-only
channelwise normalizer and verifies its recorded hash, keeps every module in
eval mode, and writes resume-safe session/cell JSON outside git.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs" / "crossbackbone_csgd_v1"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("CSGD_RUNTIME", r"D:\nips-temp\TotalP\P1\crossbackbone_csgd_runtime"))
SEVEN_REPO = Path(os.environ.get("SEVEN_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_BACKBONE_GEN_WORK"))
SEVEN_CODE = SEVEN_REPO / "experiments" / "persist_eeg_seven_backbone_fourtask_3seed_v1" / "code"
SEVEN_RUNTIME = Path(os.environ.get("SEVEN_RUNTIME", r"D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime"))
OPENBMI_CACHE = Path(os.environ.get("FULL_OPENBMI_CACHE", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi"))
WBCIC_CACHE = Path(os.environ.get("FULL_WBCIC_CACHE", r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs"))
TRUE_WBCIC_CACHE = Path(os.environ.get("TRUE_OUTER_WBCIC_CACHE", r"D:\nips-temp\TotalP\P2\wbcic_outer_cache\wbcic_epochs"))
HOLDOUT_MANIFEST = SEVEN_REPO / "experiments" / "persist_eeg_final_heldout_confirmation_v1" / "protocol" / "FINAL_HOLDOUT_MANIFEST.json"

RECENT = {
    "ModernTCN": (
        Path(r"D:\nips-temp\TotalP\P1\CRCICLR_MODERNTCN_FINAL\experiments\persist_eeg_moderntcn_4task_3seed_final_v1\code"),
        Path(r"D:\nips-temp\TotalP\P1\baseline3_runtime\moderntcn"),
    ),
    "Medformer": (
        Path(r"D:\nips-temp\TotalP\P1\CRCICLR_MEDFORMER_FINAL\experiments\persist_eeg_medformer_4task_3seed_final_v1\code"),
        Path(r"D:\nips-temp\TotalP\P1\baseline3_runtime\medformer"),
    ),
    "SGN": (
        Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SGN_FINAL\experiments\persist_eeg_sgn_4task_3seed_final_v1\code"),
        Path(r"D:\nips-temp\TotalP\P1\baseline3_runtime\sgn"),
    ),
}

MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "SGN", "LiteBN", "TFFormer")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
OLD_MODELS = ("EEGNet", "CBraMod", "TeCh", "LiteBN")
OLD_SLUGS = {"EEGNet": "eegnet", "CBraMod": "cbramod", "TeCh": "tech", "LiteBN": "litebn"}
OPENBMI_SESSIONS = (1, 2)
WBCIC_SESSIONS = (0, 1, 2)
SESSION_NAMES = {"OpenBMI": {1: "S1", 2: "S2"}, "WBCIC": {0: "S0", 1: "S1", 2: "S2"}}
TRUE_WBCIC_SUBJECTS = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")
BOOTSTRAP_DRAWS = 20_000

if str(SEVEN_CODE) not in sys.path:
    sys.path.insert(0, str(SEVEN_CODE))
os.environ.setdefault("SEVEN_REPO", str(SEVEN_REPO))
os.environ.setdefault("OFFICIAL_BACKBONE_ROOT", str(SEVEN_RUNTIME / "official"))
os.environ.setdefault("FULL_OPENBMI_CACHE", str(OPENBMI_CACHE))
os.environ.setdefault("FULL_WBCIC_CACHE", str(WBCIC_CACHE))
os.environ.setdefault("MODERN_REPO", str(SEVEN_REPO))
os.environ.setdefault("TASK_GENERALITY_REPO", str(SEVEN_REPO))

_MODULES: dict[str, Any] = {}
_VERIFIED_CHECKPOINTS: set[str] = set()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    if not fields:
        raise RuntimeError(f"refusing empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: clean(row.get(key, "")) for key in fields} for row in rows])
    os.replace(temporary, path)


def module(name: str, path: Path) -> Any:
    key = f"{name}:{path}"
    if key in _MODULES: return _MODULES[key]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except Exception:
        sys.modules.pop(name, None)
        raise
    _MODULES[key] = value
    return value


def natural_subjects(values: Iterable[object]) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        digits = "".join(character for character in value if character.isdigit())
        return (int(digits) if digits else 10**9, value)
    return sorted({str(value) for value in values}, key=key)


def checkpoint_root(model: str, task: str, fold: int, seed: int) -> Path | None:
    # There is no exact TFFormer model/checkpoint identity in the repository.
    # TCFormer is a distinct recorded identity and is deliberately not aliased.
    if model == "TFFormer": return None
    if model in OLD_MODELS:
        return SEVEN_RUNTIME / "search_cells" / task.lower() / OLD_SLUGS[model] / f"fold{fold}_seed{seed}"
    return RECENT[model][1] / "cells" / task.lower() / f"fold{fold}_seed{seed}"


def audit_row(model: str, task: str, fold: int, seed: int) -> dict[str, Any]:
    base = {"Model": model, "Task": task, "fold": fold, "seed": seed}
    root = checkpoint_root(model, task, fold, seed)
    if root is None:
        return {**base, "status": "MISSING_NO_EXACT_MODEL_IDENTITY", "reason": "repository has TCFormer, not TFFormer; no alias permitted"}
    checkpoint, record_path = root / "selected.pt", root / "record.json"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0 or not record_path.is_file():
        missing = []
        if not checkpoint.is_file() or (checkpoint.is_file() and checkpoint.stat().st_size == 0): missing.append("selected.pt")
        if not record_path.is_file(): missing.append("record.json")
        return {**base, "status": "MISSING_CHECKPOINT_CELL", "reason": "+".join(missing), "checkpoint_path": str(checkpoint)}
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        identity = (record.get("model"), record.get("task"), int(record.get("fold", -1)), int(record.get("seed", -1)))
        if identity != (model, task, fold, seed):
            return {**base, "status": "INVALID_IDENTITY", "reason": repr(identity), "checkpoint_path": str(checkpoint)}
        actual = sha(checkpoint)
        recorded = str(record.get("checkpoint_sha256", ""))
        if actual != recorded:
            return {**base, "status": "INVALID_CHECKPOINT_HASH", "reason": f"recorded={recorded};actual={actual}", "checkpoint_path": str(checkpoint)}
        normalizer_hash = str(record.get("normalizer", {}).get("mean_std_sha256", ""))
        if len(normalizer_hash) != 64:
            return {**base, "status": "INVALID_NORMALIZER_AUDIT", "reason": "missing mean_std_sha256", "checkpoint_path": str(checkpoint)}
        return {
            **base, "status": "PRESENT_VERIFIED", "reason": "", "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": actual, "checkpoint_bytes": checkpoint.stat().st_size,
            "normalizer_sha256": normalizer_hash, "selected_epoch": int(record["selected_epoch"]),
            "split_sha256": record.get("split_sha256"), "recipe_name": record.get("recipe", {}).get("name"),
            "batch_size_recorded": record.get("recipe", {}).get("batch_size"),
            "channels": int(record.get("channels") or (58 if task == "WBCIC_MI" else 62)),
            "samples": int(record.get("samples") or (250 if task == "OpenBMI_ERP" else 1000)),
            "classes": int(record.get("classes") or (4 if task == "OpenBMI_SSVEP" else 2)),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }
    except Exception as error:
        return {**base, "status": "INVALID_AUDIT_EXCEPTION", "reason": f"{type(error).__name__}: {error}", "checkpoint_path": str(checkpoint)}


def prelock() -> None:
    lock_path = PROTOCOL / "CSGD_PROTOCOL_LOCK.json"
    sidecar = PROTOCOL / "CSGD_PROTOCOL_LOCK.sha256"
    if lock_path.is_file():
        if not sidecar.is_file() or sha(lock_path) != sidecar.read_text(encoding="utf-8").strip():
            raise RuntimeError("existing CSGD protocol lock is corrupt")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        print(f"CSGD_PROTOCOL_ALREADY_LOCKED checkpoints={len(lock['evaluation_checkpoints'])}", flush=True)
        return

    manifest = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    openbmi_subjects = tuple(map(str, manifest["OpenBMI"]["subject_ids"]))
    if len(openbmi_subjects) != 14:
        raise RuntimeError(f"expected 14 OpenBMI heldout subjects, found {len(openbmi_subjects)}")
    found_wbcic = tuple(natural_subjects(path.name for path in TRUE_WBCIC_CACHE.iterdir() if path.is_dir() and path.name.startswith("sub-")))
    if found_wbcic != tuple(natural_subjects(TRUE_WBCIC_SUBJECTS)):
        raise RuntimeError(f"true-outer WBCIC membership mismatch: {found_wbcic}")
    for subject in TRUE_WBCIC_SUBJECTS:
        for session in WBCIC_SESSIONS:
            for suffix in ("epochs.npy", "labels.npy", "metadata.json"):
                path = TRUE_WBCIC_CACHE / subject / f"ses-{session}_{suffix}"
                if not path.is_file(): raise FileNotFoundError(path)

    rows = [audit_row(model, task, fold, seed) for model in MODELS for task in TASKS for seed in SEEDS for fold in FOLDS]
    invalid = [row for row in rows if str(row["status"]).startswith("INVALID")]
    if invalid:
        write_csv(OUT / "CHECKPOINT_AUDIT.csv", rows)
        raise RuntimeError(f"checkpoint audit found {len(invalid)} invalid present artifacts; first={invalid[0]}")
    present = {(row["Model"], row["Task"], row["fold"], row["seed"]): row for row in rows if row["status"] == "PRESENT_VERIFIED"}
    seed_group_complete = {
        (model, task, seed): all((model, task, fold, seed) in present for fold in FOLDS)
        for model in MODELS for task in TASKS for seed in SEEDS
    }
    primary_groups = {(model, task) for model in MODELS for task in TASKS if seed_group_complete[(model, task, 0)]}
    multiseed_groups = {
        (model, task) for model in MODELS for task in TASKS
        if all(seed_group_complete[(model, task, seed)] for seed in SEEDS)
    }
    selected_keys: set[tuple[str, str, int, int]] = set()
    for model, task in primary_groups:
        selected_keys.update((model, task, fold, 0) for fold in FOLDS)
    for model, task in multiseed_groups:
        selected_keys.update((model, task, fold, seed) for seed in SEEDS for fold in FOLDS)
    evaluation_rows = [present[key] for key in sorted(selected_keys, key=lambda x: (MODELS.index(x[0]), TASKS.index(x[1]), x[3], x[2]))]
    coverage = []
    for model in MODELS:
        for task in TASKS:
            counts = {seed: sum((model, task, fold, seed) in present for fold in FOLDS) for seed in SEEDS}
            coverage.append({
                "Model": model, "Task": task,
                "seed0_present": counts[0], "seed1_present": counts[1], "seed2_present": counts[2],
                "primary_status": "COMPLETE" if (model, task) in primary_groups else "INCOMPLETE",
                "multiseed_status": "COMPLETE" if (model, task) in multiseed_groups else "INCOMPLETE",
            })
    lock = {
        "schema": "PERSIST_EEG_CROSSBACKBONE_CSGD_LOCK_V1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requested_models": list(MODELS), "tasks": list(TASKS), "folds": list(FOLDS), "primary_seed": 0,
        "secondary_seeds": list(SEEDS), "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "biological subject",
        "cohorts": {
            "OpenBMI": {"subjects": list(openbmi_subjects), "sessions": ["S1", "S2"], "session_indices": list(OPENBMI_SESSIONS)},
            "WBCIC": {"subjects": list(TRUE_WBCIC_SUBJECTS), "sessions": ["S0", "S1", "S2"], "session_indices": list(WBCIC_SESSIONS)},
        },
        "checkpoint_inventory_rows": len(rows), "verified_checkpoint_rows": len(present),
        "evaluation_checkpoint_count": len(evaluation_rows), "coverage": coverage,
        "primary_complete_model_tasks": [{"Model": model, "Task": task} for model, task in sorted(primary_groups, key=lambda x: (MODELS.index(x[0]), TASKS.index(x[1])))],
        "multiseed_complete_model_tasks": [{"Model": model, "Task": task} for model, task in sorted(multiseed_groups, key=lambda x: (MODELS.index(x[0]), TASKS.index(x[1])))],
        "evaluation_checkpoints": evaluation_rows,
        "frozen_rules": {
            "training": False, "finetuning": False, "calibration": False, "target_adaptation": False,
            "batchnorm_updates": False, "normalizer_refit": False, "classification_head": "actual frozen model head",
            "evaluation_labels_used_for_selection": False,
        },
        "known_blockers": {
            "TFFormer": "no exact TFFormer checkpoint/model identity exists; TCFormer is not silently aliased",
            "SGN_OpenBMI_ERP": "seed0 fold4 checkpoint cell missing (4/5 present)",
            "SGN_OpenBMI_SSVEP": "all seed0 checkpoint cells missing (0/5 present)",
        },
    }
    write_csv(OUT / "CHECKPOINT_AUDIT.csv", rows)
    write_json(lock_path, lock)
    sidecar.write_text(sha(lock_path) + "\n", encoding="utf-8")
    print(
        f"CSGD_PROTOCOL_LOCKED inventory={len(rows)} verified={len(present)} evaluation={len(evaluation_rows)} "
        f"primary_model_tasks={len(primary_groups)} multiseed_model_tasks={len(multiseed_groups)}",
        flush=True,
    )


def load_lock() -> dict[str, Any]:
    lock_path, sidecar = PROTOCOL / "CSGD_PROTOCOL_LOCK.json", PROTOCOL / "CSGD_PROTOCOL_LOCK.sha256"
    if not lock_path.is_file() or not sidecar.is_file() or sha(lock_path) != sidecar.read_text(encoding="utf-8").strip():
        raise RuntimeError("intact pre-evaluation CSGD lock required")
    return json.loads(lock_path.read_text(encoding="utf-8"))


def normalizer(train: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    observations = train.shape[0] * train.shape[2]
    total = train.sum(axis=(0, 2), dtype=np.float64)
    square = np.square(train, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
    mean = (total / observations).astype(np.float32)
    std = np.sqrt(np.maximum(square / observations - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    result = {
        "mean_std_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest(),
        "samples_per_channel": int(observations),
    }
    return mean, std, result


def fold_plan(task: str, fold: int, openbmi_subjects: list[str]) -> dict[str, Any]:
    benchmark = module("csgd_benchmark_data", SEVEN_CODE / "benchmark_data.py")
    inner = benchmark._load_module("csgd_inner_loader", SEVEN_CODE / "tech_recipe_selection.py")
    modern, task_code = benchmark._sources()
    if task in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = "OpenBMI" if task == "OpenBMI_MI" else "WBCIC"
        folds, _, split_hash = modern.load_split()
        current = next(row for row in folds[dataset] if int(row["fold_id"]) == fold)
        source_sessions = tuple(map(int, modern.SOURCE_SESSIONS[dataset]))
        if dataset == "OpenBMI":
            train, _, _, mapping = inner._openbmi_rows(OPENBMI_CACHE, current["inner_train_subjects"], source_sessions, "mi")
            subjects, sessions, cache_name = list(openbmi_subjects), OPENBMI_SESSIONS, "mi"
        else:
            train, _, _, mapping = inner._wbcic_rows(WBCIC_CACHE, current["inner_train_subjects"], source_sessions)
            subjects, sessions, cache_name = list(TRUE_WBCIC_SUBJECTS), WBCIC_SESSIONS, None
    else:
        dataset = "OpenBMI"
        task_name = "ERP" if task == "OpenBMI_ERP" else "SSVEP"
        _, _, reference, _ = task_code.split_reference()
        current = next(row for row in reference["folds"] if int(row["fold_id"]) == fold)
        spec = task_code.TASKS[task_name]
        source_sessions = (int(spec["source_session"]),)
        train, _, _, mapping = inner._openbmi_rows(OPENBMI_CACHE, current["inner_train_subjects"], source_sessions, spec["cache_name"])
        subjects, sessions, cache_name, split_hash = list(openbmi_subjects), OPENBMI_SESSIONS, spec["cache_name"], reference["source_sha256"]
    mean, std, norm = normalizer(train)
    del train
    return {
        "dataset": dataset, "task": task, "fold": fold, "subjects": subjects, "sessions": tuple(sessions),
        "cache_name": cache_name, "mapping": mapping, "mean": mean, "std": std,
        "normalizer": norm, "split_sha256": split_hash, "inner": inner,
    }


def evaluation_session(plan: dict[str, Any], session: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plan["dataset"] == "OpenBMI":
        value, labels, subjects, _ = plan["inner"]._openbmi_rows(
            OPENBMI_CACHE, plan["subjects"], (session,), plan["cache_name"], plan["mapping"]
        )
    else:
        value, labels, subjects, _ = plan["inner"]._wbcic_rows(
            TRUE_WBCIC_CACHE, plan["subjects"], (session,), plan["mapping"]
        )
    expected = tuple(natural_subjects(plan["subjects"]))
    actual = tuple(natural_subjects(subjects))
    if actual != expected: raise RuntimeError(f"evaluation cohort mismatch {plan['task']} session={session}: {actual}")
    value = ((value - plan["mean"][None, :, None]) / np.maximum(plan["std"][None, :, None], 1e-6)).astype(np.float32)
    return np.ascontiguousarray(value), labels.astype(np.int64), subjects.astype(str)


def purge_vendor_namespaces() -> None:
    for namespace in ("models", "layers", "utils", "sgnmodels"):
        for name in [key for key in sys.modules if key == namespace or key.startswith(namespace + ".")]:
            sys.modules.pop(name, None)


def build_model(row: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_name = row["Model"]
    if model_name in OLD_MODELS:
        models = module("csgd_old_models", SEVEN_CODE / "backbone_models.py")
        model = models.build_model(
            model_name, dataset="WBCIC" if row["Task"] == "WBCIC_MI" else "OpenBMI",
            channels=int(row["channels"]), samples=int(row["samples"]), classes=int(row["classes"]),
            tech_recipe=row.get("recipe_name") if model_name == "TeCh" else None,
        )
    else:
        purge_vendor_namespaces()
        code = RECENT[model_name][0]
        if str(code) not in sys.path: sys.path.insert(0, str(code))
        adapter = module(f"csgd_{model_name.lower()}_adapter", code / "model_adapter.py")
        model = adapter.build_model(channels=int(row["channels"]), samples=int(row["samples"]), classes=int(row["classes"]))
    checkpoint = Path(row["checkpoint_path"])
    checkpoint_key = str(checkpoint)
    if checkpoint_key not in _VERIFIED_CHECKPOINTS:
        actual = sha(checkpoint)
        if actual != row["checkpoint_sha256"]: raise RuntimeError(f"post-lock checkpoint hash mismatch: {checkpoint}")
        _VERIFIED_CHECKPOINTS.add(checkpoint_key)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if model.training or any(child.training for child in model.modules()):
        raise RuntimeError(f"eval-mode assertion failed for {model_name}")
    return model


def resample(model_name: str, value: np.ndarray) -> np.ndarray:
    if model_name == "CBraMod":
        runner = module("csgd_old_runner", SEVEN_CODE / "run_search.py")
        return runner._resample(model_name, value)
    return np.ascontiguousarray(value, dtype=np.float32)


def inference_batch(model_name: str, recorded: Any) -> int:
    safe = {"EEGNet": 256, "LiteBN": 256, "CBraMod": 64, "TeCh": 128, "ModernTCN": 64, "Medformer": 32, "SGN": 64}
    if recorded not in (None, ""):
        return min(int(recorded), safe[model_name])
    return safe[model_name]


def subject_metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> list[dict[str, Any]]:
    predictions = logits.argmax(1)
    rows = []
    for subject in natural_subjects(subjects):
        mask = subjects.astype(str) == subject
        truth, prediction = labels[mask], predictions[mask]
        rows.append({
            "subject": subject, "BA": float(balanced_accuracy_score(truth, prediction)),
            "macro_F1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, prediction)), "trials": int(mask.sum()),
        })
    return rows


def infer(model: torch.nn.Module, value: np.ndarray, labels: np.ndarray, subjects: np.ndarray, batch: int, device: torch.device) -> list[dict[str, Any]]:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(labels), batch):
            tensor = torch.from_numpy(value[start:start + batch]).to(device)
            outputs.append(model(tensor).float().cpu().numpy())
            del tensor
    if model.training or any(child.training for child in model.modules()):
        raise RuntimeError("model left eval mode during inference")
    return subject_metrics(labels, np.concatenate(outputs), subjects)


def session_path(model: str, task: str, fold: int, seed: int, session: int) -> Path:
    return RUNTIME / "session_cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}_session{session}.json"


def cell_path(model: str, task: str, fold: int, seed: int) -> Path:
    return RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}.json"


def valid_cached_session(path: Path, row: dict[str, Any], session: int) -> bool:
    if not path.is_file(): return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value["identity"] == {"Model": row["Model"], "Task": row["Task"], "fold": row["fold"], "seed": row["seed"], "session_index": session}
            and value["checkpoint_sha256"] == row["checkpoint_sha256"]
            and value["normalizer_sha256"] == row["normalizer_sha256"]
        )
    except Exception:
        return False


def run_group(model_name: str, task: str, fold: int, records: list[dict[str, Any]], lock: dict[str, Any], device: torch.device) -> None:
    if all(cell_path(model_name, task, fold, int(row["seed"])).is_file() for row in records):
        print(f"CSGD_GROUP_CACHED model={model_name} task={task} fold={fold}", flush=True)
        return
    plan = fold_plan(task, fold, lock["cohorts"]["OpenBMI"]["subjects"])
    for row in records:
        if plan["normalizer"]["mean_std_sha256"] != row["normalizer_sha256"]:
            raise RuntimeError(f"frozen normalizer hash mismatch: {model_name}/{task}/fold{fold}/seed{row['seed']}")
        if plan["split_sha256"] != row["split_sha256"]:
            raise RuntimeError(f"split hash mismatch: {model_name}/{task}/fold{fold}/seed{row['seed']}")
    for session in plan["sessions"]:
        pending = [row for row in records if not valid_cached_session(session_path(model_name, task, fold, int(row["seed"]), session), row, session)]
        if not pending: continue
        value, labels, subjects = evaluation_session(plan, session)
        value = resample(model_name, value)
        for row in pending:
            model = build_model(row, device)
            batch = inference_batch(model_name, row.get("batch_size_recorded"))
            details = infer(model, value, labels, subjects, batch, device)
            result = {
                "identity": {"Model": model_name, "Task": task, "fold": fold, "seed": int(row["seed"]), "session_index": session},
                "session": SESSION_NAMES[plan["dataset"]][session], "checkpoint_sha256": row["checkpoint_sha256"],
                "normalizer_sha256": row["normalizer_sha256"], "normalizer_samples_per_channel": plan["normalizer"]["samples_per_channel"],
                "batch_size": batch, "inference_mode": True, "eval_mode": True, "parameters_updated": False,
                "subject_rows": details,
            }
            write_json(session_path(model_name, task, fold, int(row["seed"]), session), result)
            print(f"CSGD_SESSION_COMPLETED model={model_name} task={task} fold={fold} seed={row['seed']} session={result['session']}", flush=True)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
        del value, labels, subjects
        gc.collect()
    for row in records:
        sessions = []
        for session in plan["sessions"]:
            path = session_path(model_name, task, fold, int(row["seed"]), session)
            if not valid_cached_session(path, row, session): raise RuntimeError(f"missing valid session result: {path}")
            sessions.append(json.loads(path.read_text(encoding="utf-8")))
        result = {
            "identity": {"Model": model_name, "Task": task, "fold": fold, "seed": int(row["seed"])},
            "checkpoint_sha256": row["checkpoint_sha256"], "normalizer_sha256": row["normalizer_sha256"],
            "sessions": sessions, "training_performed": False, "selection_performed": False,
            "normalizer_refit_on_evaluation": False, "batchnorm_updated": False,
        }
        write_json(cell_path(model_name, task, fold, int(row["seed"])), result)
        print(f"CSGD_CELL_COMPLETED model={model_name} task={task} fold={fold} seed={row['seed']}", flush=True)


def run_model(model_name: str) -> None:
    lock = load_lock()
    records = [row for row in lock["evaluation_checkpoints"] if row["Model"] == model_name]
    if not records:
        print(f"CSGD_MODEL_SKIPPED model={model_name} reason=no_complete_model_task_checkpoint_matrix", flush=True)
        return
    threads = int(os.environ.get("CSGD_CPU_THREADS", "4"))
    if threads > 0: torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for task in TASKS:
        task_rows = [row for row in records if row["Task"] == task]
        for fold in FOLDS:
            group = sorted([row for row in task_rows if int(row["fold"]) == fold], key=lambda row: int(row["seed"]))
            if group: run_group(model_name, task, fold, group, lock, device)
    print(f"CSGD_MODEL_COMPLETED model={model_name} cells={len(records)} device={device}", flush=True)


def bootstrap(values: list[float], *parts: object) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0: raise RuntimeError("empty bootstrap")
    rng = np.random.default_rng(stable_seed("CSGD_BOOTSTRAP", *parts))
    draws = array[rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))].mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def coverage_map(lock: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["Model"], row["Task"]): row for row in lock["coverage"]}


def compute_seed_subject_rows(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in session_rows:
        groups[(row["Model"], row["Task"], int(row["seed"]), row["subject"])].append(row)
    result = []
    for (model, task, seed, subject), rows in groups.items():
        expected_sessions = OPENBMI_SESSIONS if task.startswith("OpenBMI") else WBCIC_SESSIONS
        per_session: dict[int, dict[str, float]] = {}
        for session in expected_sessions:
            selected = [row for row in rows if int(row["session_index"]) == session]
            folds = {int(row["fold"]) for row in selected}
            if folds != set(FOLDS):
                raise RuntimeError(f"fold coverage mismatch {model}/{task}/seed{seed}/{subject}/session{session}: {folds}")
            per_session[session] = {
                "BA": float(np.mean([float(row["BA"]) for row in selected])),
                "macro_F1": float(np.mean([float(row["macro_F1"]) for row in selected])),
            }
        if task.startswith("OpenBMI"):
            source_ba, future_ba = per_session[1]["BA"], per_session[2]["BA"]
            source_f1, future_f1 = per_session[1]["macro_F1"], per_session[2]["macro_F1"]
            row = {
                "Model": model, "Task": task, "seed": seed, "subject": subject, "folds_averaged": 5,
                "S1_BA": per_session[1]["BA"], "S1_macro_F1": per_session[1]["macro_F1"],
                "S2_BA": per_session[2]["BA"], "S2_macro_F1": per_session[2]["macro_F1"],
                "source_BA": source_ba, "future_BA": future_ba, "source_macro_F1": source_f1, "future_macro_F1": future_f1,
                "CSGD_pp": 100.0 * (source_ba - future_ba), "S0_to_S2_drop_pp": "", "S1_to_S2_drop_pp": "",
            }
        else:
            source_ba = 0.5 * (per_session[0]["BA"] + per_session[1]["BA"])
            future_ba = per_session[2]["BA"]
            source_f1 = 0.5 * (per_session[0]["macro_F1"] + per_session[1]["macro_F1"])
            future_f1 = per_session[2]["macro_F1"]
            row = {
                "Model": model, "Task": task, "seed": seed, "subject": subject, "folds_averaged": 5,
                "S0_BA": per_session[0]["BA"], "S0_macro_F1": per_session[0]["macro_F1"],
                "S1_BA": per_session[1]["BA"], "S1_macro_F1": per_session[1]["macro_F1"],
                "S2_BA": per_session[2]["BA"], "S2_macro_F1": per_session[2]["macro_F1"],
                "source_BA": source_ba, "future_BA": future_ba, "source_macro_F1": source_f1, "future_macro_F1": future_f1,
                "CSGD_pp": 100.0 * (source_ba - future_ba),
                "S0_to_S2_drop_pp": 100.0 * (per_session[0]["BA"] - future_ba),
                "S1_to_S2_drop_pp": 100.0 * (per_session[1]["BA"] - future_ba),
            }
        result.append(row)
    return sorted(result, key=lambda row: (MODELS.index(row["Model"]), TASKS.index(row["Task"]), row["seed"], natural_subjects([row["subject"]])[0]))


def summary_rows(seed_subject_rows: list[dict[str, Any]], lock: dict[str, Any], multiseed: bool) -> list[dict[str, Any]]:
    coverage = coverage_map(lock)
    rows = []
    for model in MODELS:
        for task in TASKS:
            audit = coverage[(model, task)]
            status = audit["multiseed_status"] if multiseed else audit["primary_status"]
            base = {
                "Model": model, "Task": task, "status": status,
                "seed0_checkpoints": f"{audit['seed0_present']}/5",
                "seed1_checkpoints": f"{audit['seed1_present']}/5",
                "seed2_checkpoints": f"{audit['seed2_present']}/5",
            }
            if status != "COMPLETE":
                rows.append({**base, "subjects": "", "source_BA": "", "future_BA": "", "mean_CSGD_pp": "", "median_CSGD_pp": "", "CSGD_CI95_low_pp": "", "CSGD_CI95_high_pp": ""})
                continue
            selected = [row for row in seed_subject_rows if row["Model"] == model and row["Task"] == task]
            if multiseed:
                by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in selected: by_subject[row["subject"]].append(row)
                combined = []
                for subject, subject_rows in by_subject.items():
                    if {int(row["seed"]) for row in subject_rows} != set(SEEDS):
                        raise RuntimeError(f"multiseed subject coverage mismatch {model}/{task}/{subject}")
                    combined.append({
                        "subject": subject,
                        "source_BA": float(np.mean([float(row["source_BA"]) for row in subject_rows])),
                        "future_BA": float(np.mean([float(row["future_BA"]) for row in subject_rows])),
                        "CSGD_pp": float(np.mean([float(row["CSGD_pp"]) for row in subject_rows])),
                    })
                selected_values = combined
            else:
                selected_values = [row for row in selected if int(row["seed"]) == 0]
            expected_subjects = 14 if task.startswith("OpenBMI") else 10
            if len(selected_values) != expected_subjects:
                raise RuntimeError(f"subject coverage mismatch {model}/{task}: {len(selected_values)}")
            csgd = [float(row["CSGD_pp"]) for row in selected_values]
            mean, low, high = bootstrap(csgd, "multiseed" if multiseed else "seed0", model, task)
            rows.append({
                **base, "subjects": len(selected_values),
                "source_BA": float(np.mean([float(row["source_BA"]) for row in selected_values])),
                "future_BA": float(np.mean([float(row["future_BA"]) for row in selected_values])),
                "mean_CSGD_pp": mean, "median_CSGD_pp": float(np.median(csgd)),
                "CSGD_CI95_low_pp": low, "CSGD_CI95_high_pp": high,
            })
    return rows


def fmt_ba(value: Any) -> str:
    return "—" if value in ("", None) else f"{100 * float(value):.2f}"


def fmt_csgd(row: dict[str, Any]) -> str:
    if row.get("mean_CSGD_pp") in ("", None): return "—"
    return f"{float(row['mean_CSGD_pp']):.2f} [{float(row['CSGD_CI95_low_pp']):.2f}, {float(row['CSGD_CI95_high_pp']):.2f}]"


def write_report(primary: list[dict[str, Any]], multiseed: list[dict[str, Any]], lock: dict[str, Any]) -> None:
    lines = [
        "# Final Cross-Backbone CSGD Report", "",
        "CSGD is reported in percentage points; lower is better. Future-session BA and CSGD must be interpreted jointly: a weak model can have a small drop because it is weak in every session.", "",
        "All results use exact frozen selected checkpoints, the original train-only channelwise normalizer verified by hash, actual frozen classification heads, and eval/inference mode. No retraining, finetuning, calibration, target adaptation, BN update, normalization refit, or evaluation-label-based selection occurred.", "",
        "## Primary seed0 results", "",
        "| Model | Task | Coverage | Source BA | Future BA ↑ | CSGD ↓ pp [95% CI] | Median CSGD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        coverage = row["seed0_checkpoints"]
        if row["status"] != "COMPLETE": coverage += " incomplete"
        median = "—" if row["median_CSGD_pp"] in ("", None) else f"{float(row['median_CSGD_pp']):.2f}"
        lines.append(f"| {row['Model']} | {row['Task']} | {coverage} | {fmt_ba(row['source_BA'])} | {fmt_ba(row['future_BA'])} | {fmt_csgd(row)} | {median} |")
    by_key = {(row["Model"], row["Task"]): row for row in primary}
    lines.extend(["", "## Compact cross-task summary", "", "| Model | MI Future BA | MI CSGD | ERP Future BA | ERP CSGD | SSVEP Future BA | SSVEP CSGD | WBCIC Future BA | WBCIC CSGD | Mean CSGD |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for model in MODELS:
        values = [by_key[(model, task)] for task in TASKS]
        complete = [row for row in values if row["status"] == "COMPLETE"]
        mean_all = f"{np.mean([float(row['mean_CSGD_pp']) for row in complete]):.2f}" if len(complete) == 4 else "—"
        cells = []
        for row in values:
            cells.extend([fmt_ba(row["future_BA"]), "—" if row["mean_CSGD_pp"] in ("", None) else f"{float(row['mean_CSGD_pp']):.2f}"])
        lines.append("| " + " | ".join([model, *cells, mean_all]) + " |")
    lines.extend(["", "## Secondary complete three-seed results", "", "Only model-task combinations with all 15 checkpoints (five folds × seeds 0/1/2) are estimated here.", "", "| Model | Task | Coverage | Future BA ↑ | CSGD ↓ pp [95% CI] |", "|---|---|---:|---:|---:|"])
    for row in multiseed:
        if row["status"] == "COMPLETE":
            lines.append(f"| {row['Model']} | {row['Task']} | 15/15 | {fmt_ba(row['future_BA'])} | {fmt_csgd(row)} |")
    lines.extend(["", "## Incomplete checkpoint matrices", ""])
    for row in primary:
        if row["status"] != "COMPLETE":
            lines.append(f"- {row['Model']} / {row['Task']}: seed0 {row['seed0_checkpoints']}; excluded from metric aggregation.")
    lines.extend([
        "", "TFFormer has no exact recorded checkpoint/model identity. The existing TCFormer artifacts were not silently relabeled or substituted.",
        "", "SGN is reported only for complete OpenBMI MI and WBCIC MI matrices. OpenBMI ERP lacks seed0 fold4; OpenBMI SSVEP lacks all seed0 folds.",
        "", "## Interpretation guard", "",
        "Future BA ↑ measures retained predictive performance in the future session. CSGD ↓ measures degradation relative to the source session(s). Neither alone establishes the desired property; use both columns jointly. PEEH is not used for ranking because it answers a different question.",
        "",
    ])
    (OUT / "FINAL_CROSSBACKBONE_CSGD_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate() -> None:
    lock = load_lock()
    session_rows = []
    for row in lock["evaluation_checkpoints"]:
        path = cell_path(row["Model"], row["Task"], int(row["fold"]), int(row["seed"]))
        if not path.is_file(): raise RuntimeError(f"missing completed CSGD cell: {path}")
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell["checkpoint_sha256"] != row["checkpoint_sha256"] or cell["normalizer_sha256"] != row["normalizer_sha256"]:
            raise RuntimeError(f"cell lock mismatch: {path}")
        for session_result in cell["sessions"]:
            for detail in session_result["subject_rows"]:
                session_rows.append({
                    "Model": row["Model"], "Task": row["Task"], "fold": int(row["fold"]), "seed": int(row["seed"]),
                    "subject": detail["subject"], "session": session_result["session"],
                    "session_index": int(session_result["identity"]["session_index"]),
                    "BA": float(detail["BA"]), "macro_F1": float(detail["macro_F1"]),
                    "accuracy": float(detail["accuracy"]), "trials": int(detail["trials"]),
                    "checkpoint_sha256": row["checkpoint_sha256"], "normalizer_sha256": row["normalizer_sha256"],
                })
    for row in session_rows:
        for key in ("BA", "macro_F1", "accuracy"):
            if not np.isfinite(float(row[key])): raise RuntimeError(f"non-finite metric: {row}")
    expected_session_rows = sum((28 if row["Task"].startswith("OpenBMI") else 30) for row in lock["evaluation_checkpoints"])
    if len(session_rows) != expected_session_rows:
        raise RuntimeError(f"session-subject row count mismatch: {len(session_rows)} != {expected_session_rows}")
    session_rows.sort(key=lambda row: (MODELS.index(row["Model"]), TASKS.index(row["Task"]), row["seed"], row["fold"], row["session_index"], row["subject"]))
    write_csv(OUT / "SESSION_SUBJECT_RESULTS.csv", session_rows)
    seed_subject_rows = compute_seed_subject_rows(session_rows)
    primary_subject_rows = [row for row in seed_subject_rows if int(row["seed"]) == 0]
    write_csv(OUT / "SUBJECT_LEVEL_CSGD_SEED0.csv", primary_subject_rows)
    primary = summary_rows(seed_subject_rows, lock, multiseed=False)
    multiseed = summary_rows(seed_subject_rows, lock, multiseed=True)
    write_csv(OUT / "CROSSBACKBONE_CSGD_SEED0.csv", primary)
    write_csv(OUT / "CROSSBACKBONE_CSGD_MULTISEED.csv", multiseed)
    write_report(primary, multiseed, lock)
    validation = {
        "pass": True, "evaluation_checkpoint_cells": len(lock["evaluation_checkpoints"]),
        "session_subject_rows": len(session_rows), "expected_session_subject_rows": expected_session_rows,
        "seed0_subject_rows": len(primary_subject_rows),
        "primary_complete_model_tasks": sum(row["status"] == "COMPLETE" for row in primary),
        "multiseed_complete_model_tasks": sum(row["status"] == "COMPLETE" for row in multiseed),
        "finite_metrics": True, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "training_performed": False, "selection_performed": False, "evaluation_labels_used_for_selection": False,
        "normalizer_refit_on_evaluation": False, "batchnorm_updated": False,
        "protocol_lock_sha256": sha(PROTOCOL / "CSGD_PROTOCOL_LOCK.json"),
    }
    write_json(OUT / "VALIDATION.json", validation)
    print(f"CSGD_AGGREGATION_COMPLETE cells={len(lock['evaluation_checkpoints'])} rows={len(session_rows)}", flush=True)


def status() -> None:
    lock = load_lock()
    completed = 0
    by_model = defaultdict(lambda: [0, 0])
    for row in lock["evaluation_checkpoints"]:
        by_model[row["Model"]][1] += 1
        if cell_path(row["Model"], row["Task"], int(row["fold"]), int(row["seed"])).is_file():
            completed += 1
            by_model[row["Model"]][0] += 1
    print(json.dumps({"completed": completed, "total": len(lock["evaluation_checkpoints"]), "by_model": dict(by_model)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prelock", "run", "aggregate", "status"), required=True)
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "prelock": prelock()
    elif args.stage == "run":
        if not args.model: parser.error("--model is required for --stage run")
        run_model(args.model)
    elif args.stage == "aggregate": aggregate()
    else: status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
