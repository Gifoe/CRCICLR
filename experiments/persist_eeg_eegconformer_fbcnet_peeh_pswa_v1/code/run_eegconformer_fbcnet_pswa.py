"""Deterministic coordinate recovery and frozen PSWA for two new baselines.

This script is deliberately unable to run PEEH selection.  It reconstructs
only the deterministic canonical transform, reads final Protected indices from
the corrected PEEH cell JSON, and performs frozen canonical-space probes.
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
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score


EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs" / "eegconformer_fbcnet_peeh_pswa_v1"
RUNTIME = Path(os.environ.get("PSWA_RUNTIME", r"D:\nips-temp\TotalP\P1\eegconformer_fbcnet_pswa_runtime"))
PEEH_CODE = Path(os.environ.get("PEEH_CODE", str(EXP / "code" / "run_eegconformer_fbcnet_peeh.py")))
PEEH_RUNTIME = Path(os.environ.get("PEEH_RUNTIME", r"D:\nips-temp\TotalP\P1\eegconformer_fbcnet_peeh_runtime"))

PRIMARY_MODELS = ("EEGConformer", "FBCNet")
PRIORITY_MODELS = PRIMARY_MODELS
SECONDARY_MODELS: tuple[str, ...] = ()
MODELS = PRIMARY_MODELS
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
SEED = 0
RIDGE_ALPHA = 0.01
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 20_000
TRANSFORM_VERSION = "PEEH_DETERMINISTIC_PREFIX_V1_NO_NULL_NO_SELECTION"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


peeh = load_module("pswa_frozen_peeh_helpers", PEEH_CODE)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def subjects(values: Iterable[object]) -> list[str]:
    return peeh.natural_subjects(values)


def checkpoint_row(model: str, task: str, fold: int) -> tuple[dict[str, Any], Path]:
    record, checkpoint = peeh.cell(model, task, fold)
    return {
        "Model": model,
        "Task": task,
        "fold": fold,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "normalizer_sha256": record["normalizer"]["mean_std_sha256"],
        "split_sha256": record["split_sha256"],
        "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters"))),
        "recipe_name": record.get("recipe", {}).get("name"),
        "channels": int(record.get("channels") or (58 if task == "WBCIC_MI" else 62)),
        "samples": int(record.get("samples") or (250 if task == "OpenBMI_ERP" else 1000)),
        "classes": int(record["classes"]),
    }, checkpoint


def peeh_result_path(model: str, task: str, fold: int) -> Path:
    return PEEH_RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def load_arrays(task: str, fold: int) -> dict[str, Any]:
    """Recreate the exact PEEH normalization and add fixed heldout sessions."""
    data_code = peeh.module("pswa_recovery_benchmark_data", peeh.SEVEN_CODE / "benchmark_data.py")
    inner = data_code._load_module("pswa_recovery_inner_loader", peeh.SEVEN_CODE / "tech_recipe_selection.py")
    modern, task_module = data_code._sources()
    manifest = json.loads(peeh.HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    eval_raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    if task in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = "OpenBMI" if task == "OpenBMI_MI" else "WBCIC"
        folds, _, _ = modern.load_split()
        role = next(x for x in folds[dataset] if int(x["fold_id"]) == fold)
        train_subjects = list(map(str, role["inner_train_subjects"]))
        source_sessions = tuple(map(int, modern.SOURCE_SESSIONS[dataset]))
        future_session = int(modern.EVAL_SESSION)
        if dataset == "OpenBMI":
            src, sy, ss, mapping = inner._openbmi_rows(peeh.OPENBMI_CACHE, train_subjects, source_sessions, "mi")
            fut, fy, fs, _ = inner._openbmi_rows(peeh.OPENBMI_CACHE, train_subjects, (future_session,), "mi", mapping)
            held_subjects = list(map(str, manifest["OpenBMI"]["subject_ids"]))
            for session in (1, 2):
                x, y, s, _ = inner._openbmi_rows(peeh.OPENBMI_CACHE, held_subjects, (session,), "mi", mapping)
                eval_raw[f"S{session}"] = (x, y, s)
        else:
            src, sy, ss, mapping = inner._wbcic_rows(peeh.WBCIC_CACHE, train_subjects, source_sessions)
            fut, fy, fs, _ = inner._wbcic_rows(peeh.WBCIC_CACHE, train_subjects, (future_session,), mapping)
            held_subjects = list(peeh.TRUE_WBCIC_SUBJECTS)
            for session in (0, 1, 2):
                x, y, s, _ = inner._wbcic_rows(peeh.TRUE_WBCIC_CACHE, held_subjects, (session,), mapping)
                eval_raw[f"S{session}"] = (x, y, s)
    else:
        dataset = "OpenBMI"
        name = "ERP" if task == "OpenBMI_ERP" else "SSVEP"
        _, _, reference, _ = task_module.split_reference()
        role = next(x for x in reference["folds"] if int(x["fold_id"]) == fold)
        spec = task_module.TASKS[name]
        train_subjects = list(map(str, role["inner_train_subjects"]))
        source_sessions = (int(spec["source_session"]),)
        future_session = int(spec["future_session"])
        src, sy, ss, mapping = inner._openbmi_rows(peeh.OPENBMI_CACHE, train_subjects, source_sessions, spec["cache_name"])
        fut, fy, fs, _ = inner._openbmi_rows(peeh.OPENBMI_CACHE, train_subjects, (future_session,), spec["cache_name"], mapping)
        held_subjects = list(map(str, manifest["OpenBMI"]["subject_ids"]))
        for session in (1, 2):
            x, y, s, _ = inner._openbmi_rows(peeh.OPENBMI_CACHE, held_subjects, (session,), spec["cache_name"], mapping)
            eval_raw[f"S{session}"] = (x, y, s)

    ordered = list(eval_raw)
    src_norm, normalized, norm = data_code._normalise(src, fut, *[eval_raw[k][0] for k in ordered])
    fut_norm, *eval_norms = normalized
    eval_data = {
        key: {"x": value, "y": eval_raw[key][1].astype(np.int64), "subjects": eval_raw[key][2].astype(str)}
        for key, value in zip(ordered, eval_norms)
    }
    return {
        "dataset": dataset,
        "source_session": int(source_sessions[0]),
        "future_session": future_session,
        "src": src_norm,
        "sy": sy.astype(np.int64),
        "ss": ss.astype(str),
        "fut": fut_norm,
        "fy": fy.astype(np.int64),
        "fs": fs.astype(str),
        "eval": eval_data,
        "classes": int(max(np.max(sy), np.max(fy), *[np.max(eval_raw[k][1]) for k in ordered]) + 1),
        "normalizer": norm,
    }


def cap_data(data: dict[str, Any], task: str, fold: int) -> dict[str, Any]:
    a = peeh.capped_indices(data["ss"], data["sy"], data["source_session"], task, fold, "train-source")
    b = peeh.capped_indices(data["fs"], data["fy"], data["future_session"], task, fold, "train-future")
    evaluation: dict[str, dict[str, Any]] = {}
    for key, value in data["eval"].items():
        session = int(key[1:])
        take = peeh.capped_indices(value["subjects"], value["y"], session, task, fold, "evaluation")
        evaluation[key] = {"x": value["x"][take], "y": value["y"][take], "subjects": value["subjects"][take]}
    return {
        "source_x": data["src"][a], "source_y": data["sy"][a], "source_subjects": data["ss"][a],
        "future_x": data["fut"][b], "future_y": data["fy"][b], "future_subjects": data["fs"][b],
        "evaluation": evaluation,
    }


def recover_transform(h: np.ndarray, y: np.ndarray, owner: np.ndarray, session: np.ndarray,
                      task: str, model: str, fold: int) -> dict[str, Any]:
    """Exact deterministic prefix of PEEH spectrum(); no null loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(h).to(device)
    mu = x.mean(0)
    xc = x - mu
    q = min(32, xc.shape[0] - 1, xc.shape[1])
    torch.manual_seed(peeh.stable_seed("svd", model, task, fold))
    _, singular, basis = torch.pca_lowrank(xc, q=q, center=False, niter=6)
    eigen = (singular.square() / max(len(h) - 1, 1)).double()
    threshold = max(float(eigen[0]) * 1e-3, 1e-8)
    rank_lower = int((eigen > threshold).sum())
    rank = min(peeh.ACTIVE_RANK, rank_lower)
    if rank < 4:
        raise RuntimeError(f"active rank below 4: {model}/{task}/f{fold}")
    floor = max(float(eigen[rank - 1]) * 1e-4, 1e-8)
    active = torch.clamp(eigen[:rank], min=floor).float()
    basis = basis[:, :rank]
    z = (xc @ basis / torch.sqrt(active)).cpu().numpy()
    mu_np = mu.cpu().numpy()
    basis_np = basis.cpu().numpy()

    session_ids = sorted(np.unique(session).astype(int).tolist())
    labels = sorted(np.unique(y).astype(int).tolist())
    centroids: dict[tuple[str, int, int], np.ndarray] = {}
    for subject in subjects(owner):
        for session_id in session_ids:
            for label in labels:
                idx = np.flatnonzero((owner.astype(str) == subject) & (session == session_id) & (y == label))
                if len(idx):
                    centroids[(subject, session_id, label)] = z[idx].mean(0)
    operators = []
    biological = subjects(owner)
    for label in labels:
        left, right = [], []
        for subject in biological:
            if (subject, session_ids[0], label) in centroids and (subject, session_ids[1], label) in centroids:
                left.append(centroids[(subject, session_ids[0], label)])
                right.append(centroids[(subject, session_ids[1], label)])
        if len(left) >= 3:
            a = np.asarray(left); b = np.asarray(right)
            a -= a.mean(0); b -= b.mean(0)
            operators.append((a.T @ b + b.T @ a) / (2 * len(a)))
    cross = np.mean(operators, axis=0)
    rho, directions = np.linalg.eigh((cross + cross.T) / 2)
    order = np.argsort(rho)[::-1]
    directions = directions[:, order]
    result = {
        "mean": mu_np,
        "basis": basis_np,
        "scale": np.sqrt(active.cpu().numpy()),
        "directions": directions.astype(np.float32),
        "rank": rank,
        "numerical_rank_lower_bound": rank_lower,
    }
    del x, xc, basis
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def canonical(h: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    return (((h - transform["mean"]) @ transform["basis"] / transform["scale"]) @ transform["directions"]).astype(np.float32)


def array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def save_q_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def reconstruct_cell(model: str, task: str, fold: int, device: torch.device) -> dict[str, Any]:
    result_path = RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    ppath = peeh_result_path(model, task, fold)
    if not ppath.is_file():
        result = {"Model": model, "Task": task, "fold": fold, "seed": 0, "status": "NO_PROTECTED_ASSIGNMENT", "reason": str(ppath)}
        write_json(result_path, result); return result
    stored = json.loads(ppath.read_text(encoding="utf-8"))
    protected = sorted(set(map(int, stored["protected_blocks"])))
    if not protected:
        result = {"Model": model, "Task": task, "fold": fold, "seed": 0, "status": "EMPTY_PROTECTED",
                  "protected_rank": 0, "active_rank": int(stored["rank"]), "peeh_cell_sha256": sha(ppath)}
        write_json(result_path, result); return result

    row, checkpoint = checkpoint_row(model, task, fold)
    if sha(checkpoint) != row["checkpoint_sha256"] or stored["checkpoint_sha256"] != row["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA mismatch")
    data = load_arrays(task, fold)
    if data["normalizer"]["mean_std_sha256"] != row["normalizer_sha256"]:
        raise RuntimeError("normalizer SHA mismatch")
    capped = cap_data(data, task, fold)
    cache = RUNTIME / "q_cache" / model.lower() / task.lower() / f"fold{fold}_seed0.npz"
    meta_path = cache.with_suffix(".json")
    expected_meta = {
        "checkpoint_sha256": row["checkpoint_sha256"], "normalizer_sha256": row["normalizer_sha256"],
        "split_sha256": row["split_sha256"], "transform_version": TRANSFORM_VERSION,
        "stored_active_rank": int(stored["rank"]), "peeh_cell_sha256": sha(ppath),
    }
    arrays: dict[str, np.ndarray]
    recovered_rank: int
    transform_hash: str
    if cache.is_file() and meta_path.is_file() and all(json.loads(meta_path.read_text(encoding="utf-8")).get(k) == v for k, v in expected_meta.items()):
        with np.load(cache, allow_pickle=False) as payload:
            arrays = {k: payload[k] for k in payload.files}
        recovered_rank = int(arrays["q_train"].shape[1])
        transform_hash = json.loads(meta_path.read_text(encoding="utf-8"))["transform_sha256"]
    else:
        net, head = peeh.build_model(row, device)
        h_source = peeh.representations(net, head, capped["source_x"], model, device, batch=32)
        h_future = peeh.representations(net, head, capped["future_x"], model, device, batch=32)
        h = np.concatenate([h_source, h_future])
        y = np.concatenate([capped["source_y"], capped["future_y"]])
        owner = np.concatenate([capped["source_subjects"], capped["future_subjects"]])
        session = np.concatenate([
            np.full(len(h_source), data["source_session"]), np.full(len(h_future), data["future_session"])
        ]).astype(np.int64)
        transform = recover_transform(h, y, owner, session, task, model, fold)
        recovered_rank = int(transform["rank"])
        transform_hash = array_hash(transform["mean"], transform["basis"], transform["scale"], transform["directions"])
        arrays = {
            "q_train": canonical(h_source, transform), "y_train": capped["source_y"].astype(np.int64),
            "train_subjects": capped["source_subjects"].astype(str),
        }
        del h, h_source, h_future
        for session_name, value in capped["evaluation"].items():
            h_eval = peeh.representations(net, head, value["x"], model, device, batch=32)
            arrays[f"q_eval_{session_name}"] = canonical(h_eval, transform)
            arrays[f"y_eval_{session_name}"] = value["y"].astype(np.int64)
            arrays[f"subjects_eval_{session_name}"] = value["subjects"].astype(str)
            del h_eval
        save_q_cache(cache, arrays)
        write_json(meta_path, {**expected_meta, "recovered_active_rank": recovered_rank, "transform_sha256": transform_hash,
                               "representation_layer": "exact corrected-PEEH final pre-classifier hook", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        del net
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    if recovered_rank != int(stored["rank"]):
        result = {"Model": model, "Task": task, "fold": fold, "seed": 0, "status": "RECOVERY_MISMATCH",
                  "stored_active_rank": int(stored["rank"]), "recovered_active_rank": recovered_rank}
        write_json(result_path, result); return result
    if any(index < 0 or index >= recovered_rank for index in protected):
        raise RuntimeError("stored Protected coordinate out of bounds")

    controls = [np.random.default_rng(peeh.stable_seed("final-random", model, task, fold, draw)).choice(
        np.arange(recovered_rank), size=len(protected), replace=False).astype(int).tolist() for draw in range(RANDOM_DRAWS)]
    session_results, random_results = probe_cell(arrays, protected, controls, data["classes"])
    result = {
        "Model": model, "Task": task, "fold": fold, "seed": 0, "status": "RECOVERY_OK",
        "checkpoint_sha256": row["checkpoint_sha256"], "normalizer_sha256": row["normalizer_sha256"],
        "split_sha256": row["split_sha256"], "peeh_cell_sha256": sha(ppath), "q_cache_path": str(cache),
        "q_cache_sha256": sha(cache), "transform_sha256": transform_hash, "stored_active_rank": int(stored["rank"]),
        "recovered_active_rank": recovered_rank, "protected_indices": protected, "protected_rank": len(protected),
        "random_controls": controls, "protected_session_results": session_results, "random_session_results": random_results,
        "peeh_rerun": False, "selection_rerun": False, "recovery_checksum": "NOT_RUN_OPTIONAL",
    }
    write_json(result_path, result)
    print(f"PSWA_CELL_COMPLETE {model} {task} fold={fold} rank={len(protected)}", flush=True)
    return result


def ridge_scores(x: np.ndarray, y: np.ndarray, z: np.ndarray, classes: int) -> np.ndarray:
    mu = x.mean(0, dtype=np.float64); sd = x.std(0, dtype=np.float64); sd[sd < 1e-6] = 1.0
    train = ((x - mu) / sd).astype(np.float32); test = ((z - mu) / sd).astype(np.float32)
    targets = np.eye(classes, dtype=np.float64)[y]; target_mean = targets.mean(0); centered = targets - target_mean
    weight = np.linalg.solve(train.T @ train + RIDGE_ALPHA * np.eye(train.shape[1]), train.T @ centered)
    return test @ weight + target_mean


def probe_cell(arrays: dict[str, np.ndarray], protected: list[int], controls: list[list[int]], classes: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, labels = arrays["q_train"], arrays["y_train"]
    sessions = sorted(key.removeprefix("q_eval_") for key in arrays if key.startswith("q_eval_"))
    protected_rows, random_rows = [], []
    for session in sessions:
        q = arrays[f"q_eval_{session}"]; y = arrays[f"y_eval_{session}"]; owner = arrays[f"subjects_eval_{session}"].astype(str)
        pred = ridge_scores(train[:, protected], labels, q[:, protected], classes).argmax(1)
        for subject in subjects(owner):
            mask = owner == subject
            protected_rows.append({"session": session, "subject_id": subject, "BA": float(balanced_accuracy_score(y[mask], pred[mask]))})
        for draw, dims in enumerate(controls):
            random_pred = ridge_scores(train[:, dims], labels, q[:, dims], classes).argmax(1)
            for subject in subjects(owner):
                mask = owner == subject
                random_rows.append({"draw": draw, "session": session, "subject_id": subject,
                                    "BA": float(balanced_accuracy_score(y[mask], random_pred[mask]))})
    return protected_rows, random_rows


def bootstrap(values: Sequence[float], *parts: object) -> tuple[float, float, float, float]:
    value = np.asarray(values, dtype=float)
    rng = np.random.default_rng(peeh.stable_seed("pswa-bootstrap", *parts))
    means = value[rng.integers(0, len(value), size=(BOOTSTRAP_DRAWS, len(value)))].mean(1)
    return float(value.mean()), float(np.median(value)), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def aggregate() -> None:
    cells: list[dict[str, Any]] = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                path = RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"
                if path.is_file(): cells.append(json.loads(path.read_text(encoding="utf-8")))

    audit_rows, protected_rows, random_rows, subject_fold_rows = [], [], [], []
    random_control_payload: dict[str, Any] = {"schema": "PSWA_RECOVERED_FINAL_RANDOM_CONTROLS_V1", "cells": {}}
    for cell in cells:
        audit_rows.append({k: cell.get(k, "") for k in (
            "Model", "Task", "fold", "seed", "status", "checkpoint_sha256", "normalizer_sha256", "split_sha256",
            "peeh_cell_sha256", "q_cache_path", "q_cache_sha256", "transform_sha256", "stored_active_rank",
            "recovered_active_rank", "protected_rank", "recovery_checksum", "reason")})
        if cell.get("status") != "RECOVERY_OK": continue
        key = f"{cell['Model']}/{cell['Task']}/fold{cell['fold']}/seed0"
        random_control_payload["cells"][key] = {"active_rank": cell["recovered_active_rank"], "protected_indices": cell["protected_indices"], "draws": cell["random_controls"]}
        for row in cell["protected_session_results"]:
            protected_rows.append({"Model": cell["Model"], "Task": cell["Task"], "fold": cell["fold"], "seed": 0, **row})
        for row in cell["random_session_results"]:
            random_rows.append({"Model": cell["Model"], "Task": cell["Task"], "fold": cell["fold"], "seed": 0, **row})
        ids = subjects(x["subject_id"] for x in cell["protected_session_results"])
        for subject in ids:
            p = [x["BA"] for x in cell["protected_session_results"] if x["subject_id"] == subject]
            by_draw = []
            for draw in range(RANDOM_DRAWS):
                by_draw.append(min(x["BA"] for x in cell["random_session_results"] if x["subject_id"] == subject and x["draw"] == draw))
            protected_wsba = min(p); random_wsba = float(np.mean(by_draw))
            subject_fold_rows.append({"Model": cell["Model"], "Task": cell["Task"], "fold": cell["fold"], "subject_id": subject,
                                      "Protected_WSBA": protected_wsba, "Random_WSBA": random_wsba,
                                      "PSWA_pp": 100 * (protected_wsba - random_wsba)})

    subject_rows, primary_rows, session_rows = [], [], []
    for model in PRIMARY_MODELS:
        for task in TASKS:
            relevant = [c for c in cells if c.get("Model") == model and c.get("Task") == task]
            matrix_complete = len(relevant) == 5 and all(c.get("status") in {"RECOVERY_OK", "EMPTY_PROTECTED"} for c in relevant)
            valid = [c for c in relevant if c.get("status") == "RECOVERY_OK"]
            cohort = subjects(x["subject_id"] for x in subject_fold_rows if x["Model"] == model and x["Task"] == task)
            for subject in cohort:
                values = [x for x in subject_fold_rows if x["Model"] == model and x["Task"] == task and x["subject_id"] == subject]
                subject_rows.append({"Model": model, "Task": task, "subject_id": subject,
                                     "Protected_WSBA": float(np.mean([x["Protected_WSBA"] for x in values])),
                                     "Random_WSBA": float(np.mean([x["Random_WSBA"] for x in values])),
                                     "PSWA_pp": float(np.mean([x["PSWA_pp"] for x in values])), "folds": len(values)})
            srows = [x for x in subject_rows if x["Model"] == model and x["Task"] == task]
            ranks = [int(c.get("protected_rank", 0)) for c in relevant if c.get("status") in {"RECOVERY_OK", "EMPTY_PROTECTED"}]
            row = {"Model": model, "Task": task, "Protected rank": float(np.mean(ranks)) if ranks else "",
                   "Protected coverage": f"{len(valid)}/5", "valid_folds": len(valid), "Status": "COMPLETE" if matrix_complete and valid else "INCOMPLETE"}
            if row["Status"] == "COMPLETE":
                mean, median, lo, hi = bootstrap([x["PSWA_pp"] for x in srows], "primary", model, task)
                fp = np.mean([x["Protected_WSBA"] for x in srows]); fr = np.mean([x["Random_WSBA"] for x in srows])
                future = "S2"
                advantages = []
                for subject in cohort:
                    fold_values = []
                    for cell in valid:
                        p = next(x["BA"] for x in cell["protected_session_results"] if x["subject_id"] == subject and x["session"] == future)
                        r = np.mean([x["BA"] for x in cell["random_session_results"] if x["subject_id"] == subject and x["session"] == future])
                        fold_values.append(100 * (p - r))
                    advantages.append(float(np.mean(fold_values)))
                adv, _, advlo, advhi = bootstrap(advantages, "future", model, task)
                row.update({"Protected-only WS-BA": fp, "Random-only WS-BA": fr, "PSWA mean": mean, "PSWA median": median,
                            "PSWA CI low": lo, "PSWA CI high": hi, "Significant": "YES" if lo > 0 else "NO",
                            "Future-session advantage pp": adv, "Future CI low": advlo, "Future CI high": advhi,
                            "Recovery status": "RECOVERY_OK"})
            else:
                row.update({"Protected-only WS-BA": "", "Random-only WS-BA": "", "PSWA mean": "", "PSWA median": "",
                            "PSWA CI low": "", "PSWA CI high": "", "Significant": "NOT_ESTIMATED",
                            "Future-session advantage pp": "", "Future CI low": "", "Future CI high": "",
                            "Recovery status": "INCOMPLETE_OR_PENDING"})
            primary_rows.append(row)

            for session in (("S1", "S2") if task.startswith("OpenBMI") else ("S0", "S1", "S2")):
                per_subject = []
                for subject in cohort:
                    fold_adv = []
                    for cell in valid:
                        p = next(x["BA"] for x in cell["protected_session_results"] if x["subject_id"] == subject and x["session"] == session)
                        r = np.mean([x["BA"] for x in cell["random_session_results"] if x["subject_id"] == subject and x["session"] == session])
                        fold_adv.append(100 * (p - r))
                    if fold_adv: per_subject.append(float(np.mean(fold_adv)))
                if matrix_complete and per_subject:
                    mean, median, lo, hi = bootstrap(per_subject, "session", model, task, session)
                    session_rows.append({"Model": model, "Task": task, "session": session, "Advantage pp": mean,
                                         "Median pp": median, "CI low": lo, "CI high": hi, "Status": "COMPLETE"})
                else:
                    session_rows.append({"Model": model, "Task": task, "session": session, "Advantage pp": "",
                                         "Median pp": "", "CI low": "", "CI high": "", "Status": "INCOMPLETE"})

    write_csv(OUT / "RECOVERY_AUDIT.csv", audit_rows, ["Model", "Task", "fold", "seed", "status", "checkpoint_sha256", "normalizer_sha256", "split_sha256", "peeh_cell_sha256", "q_cache_path", "q_cache_sha256", "transform_sha256", "stored_active_rank", "recovered_active_rank", "protected_rank", "recovery_checksum", "reason"])
    write_json(OUT / "RECOVERED_RANDOM_CONTROLS.json", random_control_payload)
    write_csv(OUT / "PROTECTED_ONLY_SESSION_RESULTS.csv", protected_rows, ["Model", "Task", "fold", "seed", "session", "subject_id", "BA"])
    write_csv(OUT / "RANDOM_ONLY_SESSION_RESULTS.csv", random_rows, ["Model", "Task", "fold", "seed", "draw", "session", "subject_id", "BA"])
    write_csv(OUT / "SUBJECT_LEVEL_PSWA_SEED0.csv", subject_rows, ["Model", "Task", "subject_id", "Protected_WSBA", "Random_WSBA", "PSWA_pp", "folds"])
    write_csv(OUT / "CROSSBACKBONE_PSWA_SEED0.csv", primary_rows, list(primary_rows[0]))
    write_csv(OUT / "SESSION_SPECIFIC_PSWA.csv", session_rows, list(session_rows[0]))

    priority_complete = all(next(x for x in primary_rows if x["Model"] == m and x["Task"] == t)["Status"] == "COMPLETE"
                            for m in PRIMARY_MODELS for t in TASKS)
    validation = {"pass": priority_complete, "scientific_status": "EEGCONFORMER_FBCNET_COMPLETE" if priority_complete else "PARTIAL_RECOVERY_IN_PROGRESS",
                  "priority_models": list(PRIORITY_MODELS), "priority_two_complete": priority_complete,
                  "recovery_ok_cells": sum(c.get("status") == "RECOVERY_OK" for c in cells), "empty_protected_cells": sum(c.get("status") == "EMPTY_PROTECTED" for c in cells),
                  "no_assignment_cells": sum(c.get("status") == "NO_PROTECTED_ASSIGNMENT" for c in cells), "peeh_rerun": False,
                  "protected_selection_rerun": False, "persistence_permutations_run": 0, "random_utility_erasures_run": 0,
                  "frozen_inference_only": True, "ridge_alpha": RIDGE_ALPHA, "random_draws": RANDOM_DRAWS,
                  "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "biological subject"}
    write_json(OUT / "VALIDATION.json", validation)

    lines = ["# EEGConformer/FBCNet frozen PSWA", "", f"Scientific status: **{validation['scientific_status']}**.", "",
             "No PEEH selection, persistence null, utility split, or PEEH bootstrap was rerun.", "",
             "|Model|Task|Coverage|Protected WS-BA|Random WS-BA|PSWA pp [95% CI]|Future S2 advantage pp [95% CI]|Recovery|",
             "|---|---|---:|---:|---:|---|---|---|"]
    for row in primary_rows:
        if row["Status"] == "COMPLETE":
            lines.append(f"|{row['Model']}|{row['Task']}|{row['Protected coverage']}|{100*row['Protected-only WS-BA']:.2f}|{100*row['Random-only WS-BA']:.2f}|{row['PSWA mean']:.3f} [{row['PSWA CI low']:.3f}, {row['PSWA CI high']:.3f}]|{row['Future-session advantage pp']:.3f} [{row['Future CI low']:.3f}, {row['Future CI high']:.3f}]|RECOVERY_OK|")
        else:
            lines.append(f"|{row['Model']}|{row['Task']}|{row['Protected coverage']}|NA|NA|NA|NA|{row['Recovery status']}|")
    (OUT / "FINAL_CROSSBACKBONE_PSWA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PSWA_AGGREGATE_COMPLETE priority2={priority_complete}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("run", "aggregate"), required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    args = parser.parse_args()
    if args.stage == "aggregate": aggregate(); return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [(m, t, f) for m in MODELS for t in TASKS for f in FOLDS
            if (args.model is None or m == args.model) and (args.task is None or t == args.task) and (args.fold is None or f == args.fold)]
    for model, task, fold in jobs:
        reconstruct_cell(model, task, fold, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
