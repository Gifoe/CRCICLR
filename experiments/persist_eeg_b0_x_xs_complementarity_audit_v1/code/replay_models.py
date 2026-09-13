#!/usr/bin/env python3
"""Exact development-outer replay for historical LiteBN B0/X/XS cells.

Only already-trained checkpoints are evaluated. Heldout membership is never
imported or enumerated and no optimizer is instantiated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
REPLAY_COLUMNS = [
    "comparison", "task", "seed", "fold", "subject_id", "method",
    "historical_BA", "replay_BA", "abs_BA_diff", "historical_macro_F1",
    "replay_macro_F1", "abs_macro_F1_diff", "historical_accuracy",
    "replay_accuracy", "abs_accuracy_diff", "checkpoint_sha256",
    "normalizer_sha256", "status",
]
TRIAL_COLUMNS = [
    "comparison", "task", "seed", "fold", "subject_id", "session",
    "trial_id", "true_label", "method", "logits", "probabilities",
    "prediction", "correct", "checkpoint_sha256", "normalizer_sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False, **kwargs)
    os.replace(temp, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_model_module(repo: Path, cache: Path, seed0_runtime: Path) -> Any:
    os.environ["LITEBN_X_REPO"] = str(repo)
    os.environ["PERSIST_CACHE_ROOT"] = str(cache)
    os.environ["LITEBN_X_RUNTIME"] = str(seed0_runtime)
    source = repo / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py"
    spec = importlib.util.spec_from_file_location("complementarity_litebn_x", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def x_checkpoint(seed0_runtime: Path, task: str, fold: int, method: str) -> Path:
    return seed0_runtime / "checkpoints" / task.lower() / f"fold{fold}_{method.lower()}" / "selected_best.pt"


def xs_checkpoint(seed0_runtime: Path, xs_runtime: Path, erp_runtime: Path, task: str, fold: int, seed: int) -> Path:
    root = seed0_runtime if seed == 0 else (erp_runtime / f"seed{seed}" if task == "OpenBMI_ERP" else xs_runtime / f"seed{seed}")
    return root / "checkpoints" / task.lower() / f"fold{fold}_litebn_xs" / "selected_best.pt"


def xs_baseline(carrier_runtime: Path, task_runtime: Path, task: str, fold: int, seed: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        short = "erp" if task.endswith("ERP") else "ssvep"
        return task_runtime / f"{short}_fold{fold}_seed{seed}_litebn" / "selected_best.pt"
    short = "openbmi" if task == "OpenBMI_MI" else "wbcic"
    return carrier_runtime / f"{short}_fold{fold}_seed{seed}_litebn" / "selected_best.pt"


def normalizer_path(seed0_runtime: Path, xs_runtime: Path, erp_runtime: Path, comparison: str, task: str, fold: int, seed: int) -> Path:
    if comparison == "X" or seed == 0:
        root = seed0_runtime
    else:
        root = erp_runtime / f"seed{seed}" if task == "OpenBMI_ERP" else xs_runtime / f"seed{seed}"
    return root / "normalizers" / f"{task.lower()}_fold{fold}.npz"


def historical_rows(repo: Path) -> dict[str, pd.DataFrame]:
    x = pd.read_csv(repo / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/outputs/OUTER_SUBJECT_RESULTS.csv")
    x["seed"] = 0
    xs = pd.read_csv(repo / "experiments/persist_eeg_xs_full_multiseed_finaltest_v1/outputs/DEVELOPMENT_MULTISEED_SUBJECT_RESULTS.csv")
    required = {"task", "seed", "fold", "subject_id", "method", "BA", "macro_F1", "accuracy", "checkpoint_sha256", "normalizer_sha256"}
    for name, frame in (("X", x), ("XS", xs)):
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{name} historical table missing {sorted(missing)}")
    return {"X": x, "XS": xs}


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "BA": float(balanced_accuracy_score(labels, predictions)),
        "macro_F1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def infer_subject(mod: Any, model: torch.nn.Module, bundle: Any, subject: str, mean: np.ndarray, std: np.ndarray,
                  device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    session = int(mod.TASKS[bundle.task]["future_session"])
    indices = bundle.indices([subject], (session,))
    if not len(indices):
        raise RuntimeError(f"no development-outer trials for {bundle.task}/{subject}")
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None]
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None]
    logits_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            raw = bundle.signal_batch(indices[start:start + batch_size])
            value = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)).to(device, non_blocking=True)
            logits_parts.append(model((value - mean_t) / torch.clamp(std_t, min=1e-6))[0].float().cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted); probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    records = []
    for position, bundle_index in enumerate(indices):
        row = bundle.rows[int(bundle_index)]
        records.append({
            "subject_id": str(subject), "session": int(row.session), "trial_id": int(row.index),
            "true_label": int(row.label), "logits": json.dumps(logits[position].tolist(), separators=(",", ":")),
            "probabilities": json.dumps(probabilities[position].tolist(), separators=(",", ":")),
            "prediction": int(predictions[position]), "correct": bool(predictions[position] == row.label),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/root/rivermind-data/persist_eeg_cache"))
    parser.add_argument("--seed0-runtime", type=Path, default=Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime"))
    parser.add_argument("--xs-runtime", type=Path, default=Path("/root/rivermind-data/xs_full_multiseed_finaltest_runtime"))
    parser.add_argument("--erp-runtime", type=Path, default=Path("/root/rivermind-data/xs_erp_seed12_stability_runtime_correct_xs"))
    parser.add_argument("--carrier-runtime", type=Path, default=Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime"))
    parser.add_argument("--task-runtime", type=Path, default=Path("/root/rivermind-data/openbmi_task_generality_runtime"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if not (0 < args.tolerance <= 1e-6):
        raise ValueError("replay tolerance must be in (0, 1e-6]")
    repo, runtime = args.repo.resolve(), args.runtime.resolve()
    experiment = repo / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1"
    outputs, protocol, cells_dir = experiment / "outputs", experiment / "protocol", runtime / "replay_cells"
    outputs.mkdir(parents=True, exist_ok=True); protocol.mkdir(parents=True, exist_ok=True); cells_dir.mkdir(parents=True, exist_ok=True)
    mod = load_model_module(repo, args.cache.resolve(), args.seed0_runtime.resolve())
    _, folds, split_sha = mod.load_folds()
    historical = historical_rows(repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    replay_rows: list[dict[str, Any]] = []; cell_rows: list[dict[str, Any]] = []; provenance: list[dict[str, Any]] = []
    comparisons = [("X", 0, "LiteBN_X")] + [("XS", seed, "LiteBN_XS") for seed in (0, 1, 2)]

    for comparison, seed, candidate in comparisons:
        source = historical[comparison]
        for task in TASKS:
            dataset = mod.TASKS[task]["dataset"]
            for fold_record in folds[dataset]:
                fold = int(fold_record["fold_id"]); subjects = [str(v) for v in fold_record["outer_dev_subjects"]]
                norm_path = normalizer_path(args.seed0_runtime, args.xs_runtime, args.erp_runtime, comparison, task, fold, seed)
                paths = {
                    "LiteBN_BASELINE": x_checkpoint(args.seed0_runtime, task, fold, "LiteBN_BASELINE") if comparison == "X" else xs_baseline(args.carrier_runtime, args.task_runtime, task, fold, seed),
                    candidate: x_checkpoint(args.seed0_runtime, task, fold, candidate) if comparison == "X" else xs_checkpoint(args.seed0_runtime, args.xs_runtime, args.erp_runtime, task, fold, seed),
                }
                cell_name = f"{comparison}__{task}__seed{seed}__fold{fold}"
                cell_path = cells_dir / f"{cell_name}.csv.gz"
                tf32_allowed = False
                expected = source[(source.task == task) & (source.seed.astype(int) == seed) & (source.fold.astype(int) == fold)]
                checkpoint_info: dict[str, tuple[str, str]] = {}; unavailable = False
                for method, path in paths.items():
                    method_rows = expected[expected.method == method]
                    if len(method_rows) != len(subjects) or method_rows.checkpoint_sha256.nunique() != 1:
                        raise RuntimeError(f"historical cell cardinality mismatch: {cell_name}/{method}")
                    expected_hash = str(method_rows.checkpoint_sha256.iloc[0]); actual_hash = sha256(path) if path.is_file() else ""
                    status = "AVAILABLE" if actual_hash == expected_hash else "CHECKPOINT_UNAVAILABLE"
                    unavailable |= status != "AVAILABLE"; checkpoint_info[method] = (expected_hash, actual_hash)
                    provenance.append({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "method": method,
                        "checkpoint_path": str(path), "expected_sha256": expected_hash, "actual_sha256": actual_hash,
                        "cuda_tf32_allowed": tf32_allowed, "status": status})
                if not norm_path.is_file():
                    unavailable = True
                if unavailable:
                    for row in expected.itertuples(index=False):
                        replay_rows.append({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "subject_id": str(row.subject_id), "method": row.method,
                            "historical_BA": row.BA, "replay_BA": np.nan, "abs_BA_diff": np.nan, "historical_macro_F1": row.macro_F1,
                            "replay_macro_F1": np.nan, "abs_macro_F1_diff": np.nan, "historical_accuracy": row.accuracy,
                            "replay_accuracy": np.nan, "abs_accuracy_diff": np.nan, "checkpoint_sha256": row.checkpoint_sha256,
                            "normalizer_sha256": row.normalizer_sha256, "status": "CHECKPOINT_UNAVAILABLE"})
                    cell_rows.append({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "status": "CHECKPOINT_UNAVAILABLE", "trial_rows": 0})
                    continue
                mean, std, norm_meta = mod.load_tensor_pair(norm_path)
                if set(map(str, expected.normalizer_sha256.unique())) != {str(norm_meta["mean_std_sha256"])}:
                    raise RuntimeError(f"normalizer SHA mismatch: {cell_name}")
                bundle = mod.build_bundle(task, subjects); trial_frames = []; cell_pass = True
                for method, path in paths.items():
                    model = mod.build_model(method, task).to(device)
                    model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
                    subject_predictions = []
                    for subject in subjects:
                        records = infer_subject(mod, model, bundle, subject, mean, std, device, args.batch_size)
                        prediction = np.asarray([r["prediction"] for r in records], dtype=np.int64)
                        labels = np.asarray([r["true_label"] for r in records], dtype=np.int64)
                        replay = metric(labels, prediction)
                        hist = expected[(expected.method == method) & (expected.subject_id.astype(str) == str(subject))]
                        if len(hist) != 1:
                            raise RuntimeError(f"historical subject row mismatch: {cell_name}/{method}/{subject}")
                        hist_row = hist.iloc[0]; diffs = {name: abs(float(replay[name]) - float(hist_row[name])) for name in ("BA", "macro_F1", "accuracy")}
                        status = "PASS" if max(diffs.values()) <= args.tolerance else "PROTOCOL_FAIL_FOR_CELL"; cell_pass &= status == "PASS"
                        replay_rows.append({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "subject_id": str(subject), "method": method,
                            "historical_BA": float(hist_row.BA), "replay_BA": replay["BA"], "abs_BA_diff": diffs["BA"],
                            "historical_macro_F1": float(hist_row.macro_F1), "replay_macro_F1": replay["macro_F1"], "abs_macro_F1_diff": diffs["macro_F1"],
                            "historical_accuracy": float(hist_row.accuracy), "replay_accuracy": replay["accuracy"], "abs_accuracy_diff": diffs["accuracy"],
                            "checkpoint_sha256": checkpoint_info[method][0], "normalizer_sha256": norm_meta["mean_std_sha256"], "status": status})
                        for record in records:
                            record.update({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "method": method,
                                "checkpoint_sha256": checkpoint_info[method][0], "normalizer_sha256": norm_meta["mean_std_sha256"]})
                        subject_predictions.extend(records)
                    trial_frames.append(pd.DataFrame(subject_predictions, columns=TRIAL_COLUMNS)); del model
                    if device.type == "cuda": torch.cuda.empty_cache()
                trials = pd.concat(trial_frames, ignore_index=True); status = "PASS" if cell_pass else "PROTOCOL_FAIL_FOR_CELL"
                if cell_pass:
                    atomic_csv(cell_path, trials, compression="gzip")
                cell_rows.append({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "status": status,
                    "trial_rows": int(len(trials)) if cell_pass else 0, "normalizer_path": str(norm_path),
                    "normalizer_sha256": norm_meta["mean_std_sha256"], "cuda_tf32_allowed": tf32_allowed})
                del bundle, trials, trial_frames; gc.collect()
                print(f"REPLAY {cell_name} {status}", flush=True)

    replay = pd.DataFrame(replay_rows, columns=REPLAY_COLUMNS).sort_values(["comparison", "task", "seed", "fold", "subject_id", "method"])
    cell_status = pd.DataFrame(cell_rows).sort_values(["comparison", "task", "seed", "fold"])
    atomic_csv(outputs / "REPLAY_AUDIT.csv", replay); atomic_csv(outputs / "REPLAY_CELL_STATUS.csv", cell_status)
    atomic_csv(outputs / "CHECKPOINT_AVAILABILITY.csv", pd.DataFrame(provenance).sort_values(["comparison", "task", "seed", "fold", "method"]))
    atomic_json(protocol / "SOURCE_PROVENANCE.json", {
        "authoritative_development_sources": [
            "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/outputs/OUTER_SUBJECT_RESULTS.csv",
            "experiments/persist_eeg_xs_full_multiseed_finaltest_v1/outputs/DEVELOPMENT_MULTISEED_SUBJECT_RESULTS.csv",
        ],
        "checkpoint_records": provenance, "fivefold_split_sha256": split_sha,
        "replay_tolerance": args.tolerance, "device": str(device),
        "numerical_replay_mode": {"cuda_tf32_allowed": False, "cudnn_benchmark": False,
            "cudnn_deterministic": True},
        "optimizer_instantiated": False,
    })
    atomic_json(protocol / "DEVELOPMENT_SCOPE_AUDIT.json", {
        "EXPERIMENT_TYPE": "ANALYSIS_ONLY", "NEW_MODEL_TRAINED": "NO",
        "FINAL_HELDOUT_ACCESSED": "NO", "INTERNAL_HELDOUT_ACCESSED": "NO",
        "DEVELOPMENT_OUTER_ONLY": "YES", "final_test_result_artifacts_read": False,
        "development_outer_subjects_loaded": True, "optimizer_instantiated": False,
        "replay_cells_passed": int((cell_status.status == "PASS").sum()),
        "replay_cells_failed": int((cell_status.status == "PROTOCOL_FAIL_FOR_CELL").sum()),
        "checkpoint_unavailable_cells": int((cell_status.status == "CHECKPOINT_UNAVAILABLE").sum()),
    })
    print(f"REPLAY_COMPLETE pass={(cell_status.status == 'PASS').sum()} fail={(cell_status.status != 'PASS').sum()}", flush=True)


if __name__ == "__main__":
    main()
