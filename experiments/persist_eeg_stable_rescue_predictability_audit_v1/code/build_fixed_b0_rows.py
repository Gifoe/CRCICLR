#!/usr/bin/env python3
"""Repair ERP replay and construct fixed-B0 development-outer trial rows.

This program evaluates frozen checkpoints only. It never imports heldout/final
artifacts and never instantiates an optimizer.
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
KEY = ["task", "fold", "subject_id", "session", "trial_id"]
FAILED_ERP_CELLS = (("X", 0, 1), ("XS", 1, 0), ("XS", 1, 3), ("XS", 2, 0), ("XS", 2, 3))
REPLAY_MODES = (
    ("historical_pytorch_default_cudnn_tf32_on_matmul_tf32_off", True, False),
    ("all_tf32_on", True, True),
    ("full_fp32", False, False),
    ("cudnn_tf32_off_matmul_tf32_on", False, True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_sha(items: list[tuple[str, tuple[int, ...], str]]) -> str:
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


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


def load_py(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metric(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "BA": float(balanced_accuracy_score(labels, predictions)),
        "macro_F1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def infer_cell(mod: Any, model: torch.nn.Module, bundle: Any, cache: Any, subjects: list[str],
               mean: np.ndarray, std: np.ndarray) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    records: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for subject in mod.subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], (int(mod.TASKS[bundle.task]["future_session"]),))
            labels = bundle.labels(indices)
            parts = []
            for start in range(0, len(indices), 128):
                value, _ = cache.batch(indices[start:start + 128], mean, std)
                parts.append(model(value)[0].float().cpu().numpy())
            logits = np.concatenate(parts)
            probs = softmax(logits)
            pred = logits.argmax(axis=1)
            metrics[str(subject)] = metric(labels, pred)
            for pos, index in enumerate(indices):
                source = bundle.rows[int(index)]
                records.append({
                    "task": bundle.task, "fold": -1, "subject_id": str(subject),
                    "session": int(source.session), "trial_id": int(source.index),
                    "true_label": int(source.label),
                    "logits": json.dumps(logits[pos].tolist(), separators=(",", ":")),
                    "probabilities": json.dumps(probs[pos].tolist(), separators=(",", ":")),
                    "prediction": int(pred[pos]), "correct": bool(pred[pos] == source.label),
                })
    return pd.DataFrame(records), metrics


def checkpoint_path(replay: Any, comparison: str, seed: int, fold: int, method: str,
                    roots: dict[str, Path]) -> Path:
    if comparison == "X":
        return replay.x_checkpoint(roots["seed0"], "OpenBMI_ERP", fold, method)
    if method == "LiteBN_BASELINE":
        return replay.xs_baseline(roots["carrier"], roots["task"], "OpenBMI_ERP", fold, seed)
    return replay.xs_checkpoint(roots["seed0"], roots["xs"], roots["erp"], "OpenBMI_ERP", fold, seed)


def normalizer_path(replay: Any, comparison: str, seed: int, fold: int, roots: dict[str, Path]) -> Path:
    return replay.normalizer_path(roots["seed0"], roots["xs"], roots["erp"], comparison, "OpenBMI_ERP", fold, seed)


def selected_epoch(selected: Path) -> int:
    latest = selected.with_name("checkpoint_latest.pt")
    if not latest.is_file():
        return -1
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    value = payload.get("best_epoch")
    return int(value) if value is not None else -1


def model_schema(model: torch.nn.Module) -> tuple[str, str, str, str]:
    params = [(name, tuple(value.shape), str(value.dtype)) for name, value in model.named_parameters()]
    buffers = [(name, tuple(value.shape), str(value.dtype)) for name, value in model.named_buffers()]
    return model.__class__.__name__, json.dumps(params, separators=(",", ":")), schema_sha(params), schema_sha(buffers)


def historical_cell(historical: dict[str, pd.DataFrame], comparison: str, seed: int, fold: int) -> pd.DataFrame:
    frame = historical[comparison]
    return frame[(frame.task == "OpenBMI_ERP") & (frame.seed.astype(int) == seed) & (frame.fold.astype(int) == fold)]


def compare_metrics(expected: pd.DataFrame, method: str, got: dict[str, dict[str, float]]) -> tuple[float, list[str]]:
    maximum, failed = 0.0, []
    for subject, values in got.items():
        row = expected[(expected.method == method) & (expected.subject_id.astype(str) == str(subject))]
        if len(row) != 1:
            raise RuntimeError(f"historical row mismatch {method}/{subject}")
        diffs = [abs(float(values[k]) - float(row.iloc[0][k])) for k in ("BA", "macro_F1", "accuracy")]
        maximum = max(maximum, max(diffs))
        if max(diffs) > 1e-8:
            failed.append(str(subject))
    return maximum, failed


def derived_pair(base: pd.DataFrame, candidate: pd.DataFrame, candidate_name: str, seed: int,
                 b0_sha: str, candidate_sha: str, norm_sha: str) -> pd.DataFrame:
    base = base.sort_values(KEY).reset_index(drop=True)
    candidate = candidate.sort_values(KEY).reset_index(drop=True)
    if not base[KEY + ["true_label"]].equals(candidate[KEY + ["true_label"]]):
        raise RuntimeError(f"trial alignment failure {candidate_name}/seed{seed}")
    out = base[KEY + ["true_label"]].copy()
    out["seed"] = int(seed)
    out["B0_logits"], out["B0_probabilities"] = base.logits, base.probabilities
    out["B0_prediction"], out["B0_correct"] = base.prediction.astype(int), base.correct.astype(bool)
    out["candidate_name"] = candidate_name
    out["candidate_logits"], out["candidate_probabilities"] = candidate.logits, candidate.probabilities
    out["candidate_prediction"], out["candidate_correct"] = candidate.prediction.astype(int), candidate.correct.astype(bool)
    out["B0_checkpoint_sha256"], out["candidate_checkpoint_sha256"], out["normalizer_sha256"] = b0_sha, candidate_sha, norm_sha
    bp = np.vstack(out.B0_probabilities.map(json.loads)); cp = np.vstack(out.candidate_probabilities.map(json.loads))
    out["B0_margin"] = np.sort(bp, axis=1)[:, -1] - np.sort(bp, axis=1)[:, -2]
    out["candidate_margin"] = np.sort(cp, axis=1)[:, -1] - np.sort(cp, axis=1)[:, -2]
    out["B0_entropy"] = -(bp * np.log(np.clip(bp, 1e-12, 1))).sum(axis=1)
    out["candidate_entropy"] = -(cp * np.log(np.clip(cp, 1e-12, 1))).sum(axis=1)
    out["probability_L1_distance"] = np.abs(bp - cp).sum(axis=1)
    midpoint = .5 * (bp + cp)
    out["JS_divergence"] = .5 * ((bp * np.log(np.clip(bp / midpoint, 1e-12, None))).sum(axis=1) +
                                  (cp * np.log(np.clip(cp / midpoint, 1e-12, None))).sum(axis=1))
    out["error_state"] = np.select(
        [out.B0_correct & out.candidate_correct, ~out.B0_correct & out.candidate_correct,
         out.B0_correct & ~out.candidate_correct], ["CC", "RESCUE", "HARM"], default="WW")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/root/rivermind-data/persist_eeg_cache"))
    args = parser.parse_args()
    repo, runtime = args.repo.resolve(), args.runtime.resolve()
    exp = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1"
    prior = repo / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1"
    outputs, protocol = exp / "outputs", exp / "protocol"
    outputs.mkdir(parents=True, exist_ok=True); runtime.mkdir(parents=True, exist_ok=True)
    roots = {
        "seed0": Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime"),
        "xs": Path("/root/rivermind-data/xs_full_multiseed_finaltest_runtime"),
        "erp": Path("/root/rivermind-data/xs_erp_seed12_stability_runtime_correct_xs"),
        "carrier": Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime"),
        "task": Path("/root/rivermind-data/openbmi_task_generality_runtime"),
    }
    os.environ.update({"LITEBN_X_REPO": str(repo), "PERSIST_CACHE_ROOT": str(args.cache.resolve()), "LITEBN_X_RUNTIME": str(roots["seed0"])})
    replay = load_py("stable_replay_helpers", prior / "code/replay_models.py")
    mod = replay.load_model_module(repo, args.cache.resolve(), roots["seed0"])
    source_sha = sha256(repo / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py")
    if source_sha != "5fa225c30f1938146a71f7655d56c487f457b8ab4a35c1a724a527e606c7dbf5":
        raise RuntimeError(f"historical source mismatch: {source_sha}")
    _, folds, split_sha = mod.load_folds()
    historical = replay.historical_rows(repo)
    source_path = prior / "outputs/PAIRED_TRIAL_RESULTS.csv"
    reference_labels = pd.read_csv(source_path, usecols=KEY + ["true_label", "seed", "candidate_name"], dtype={"subject_id": str})
    reference_labels = reference_labels[(reference_labels.task == "OpenBMI_ERP") &
                                        (reference_labels.seed == 0) &
                                        (reference_labels.candidate_name == "LiteBN_XS")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("historical ERP replay requires CUDA semantics")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    attempts: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for comparison, seed, fold in FAILED_ERP_CELLS:
        candidate_method = "LiteBN_X" if comparison == "X" else "LiteBN_XS"
        expected = historical_cell(historical, comparison, seed, fold)
        subjects = [str(v) for v in folds["OpenBMI"][fold]["outer_dev_subjects"]]
        norm_path = normalizer_path(replay, comparison, seed, fold, roots)
        mean, std, norm_meta = mod.load_tensor_pair(norm_path)
        bundle = mod.build_bundle("OpenBMI_ERP", subjects)
        current_labels = []
        for index in bundle.indices(subjects, (2,)):
            item = bundle.rows[int(index)]
            current_labels.append({"task": "OpenBMI_ERP", "fold": fold, "subject_id": str(item.subject),
                                   "session": int(item.session), "trial_id": int(item.index), "true_label": int(item.label)})
        current_labels = pd.DataFrame(current_labels).sort_values(KEY).reset_index(drop=True)
        label_reference = reference_labels[reference_labels.fold == fold][KEY + ["true_label"]].sort_values(KEY).reset_index(drop=True)
        labels_match = current_labels.equals(label_reference)
        if not labels_match: raise RuntimeError(f"ERP label/trial reference mismatch {comparison}/{seed}/{fold}")
        for method in ("LiteBN_BASELINE", candidate_method):
            ckpt = checkpoint_path(replay, comparison, seed, fold, method, roots)
            if sha256(ckpt) != str(expected[expected.method == method].checkpoint_sha256.iloc[0]):
                raise RuntimeError(f"checkpoint provenance mismatch {comparison}/{seed}/{fold}/{method}")
            model = mod.build_model(method, "OpenBMI_ERP")
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
            arch, schema, param_sha, buffer_sha = model_schema(model)
            if method == "LiteBN_BASELINE":
                historical_experiment = ("persist_eeg_litebn_x_singlemodel_seed0_v1" if comparison == "X" else
                                         "persist_eeg_openbmi_task_generality_v1")
                model_source_path = repo / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py"
            else:
                historical_experiment = "persist_eeg_litebn_x_singlemodel_seed0_v1" if comparison == "X" else "persist_eeg_xs_erp_seed12_stability_v1"
                model_source_path = repo / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py"
            runner_source_path = (repo / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py" if comparison == "X" else
                                  repo / "experiments/persist_eeg_xs_erp_seed12_stability_v1/code/run_xs_erp_seed12_stability.py")
            provenance.append({
                "historical_experiment": historical_experiment,
                "comparison": comparison, "task": "OpenBMI_ERP", "seed": seed, "fold": fold, "method": method,
                "checkpoint_path": str(ckpt), "checkpoint_sha256": sha256(ckpt), "selected_epoch": selected_epoch(ckpt),
                "architecture": arch, "parameter_schema": schema, "parameter_schema_sha256": param_sha,
                "buffer_schema_sha256": buffer_sha, "normalizer_path": str(norm_path),
                "normalizer_sha256": norm_meta["mean_std_sha256"], "normalizer_file_sha256": sha256(norm_path),
                "historical_source_code_path": str(model_source_path), "historical_source_code_sha256": sha256(model_source_path),
                "historical_evaluation_runner_path": str(runner_source_path), "historical_evaluation_runner_sha256": sha256(runner_source_path),
                "evaluation_subjects": json.dumps(subjects, separators=(",", ":")),
                "evaluation_session": 2, "label_mapping": json.dumps({"1": 0, "2": 1}, separators=(",", ":")),
                "trial_count": int(len(bundle.indices(subjects, (2,)))), "checkpoint_strict_load": True,
                "subject_set_match": set(expected.subject_id.astype(str)) == set(subjects),
                "trial_count_match": int(expected[expected.method == method].trials.sum()) == len(current_labels),
                "labels_match": labels_match, "trial_keys_match": labels_match, "fold_assignment_match": True,
            })
        for mode, cudnn_tf32, matmul_tf32 in REPLAY_MODES:
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            cache = mod.RawGPUCache(bundle, device)
            mode_rows = []
            for method in ("LiteBN_BASELINE", candidate_method):
                ckpt = checkpoint_path(replay, comparison, seed, fold, method, roots)
                model = mod.build_model(method, "OpenBMI_ERP").to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False), strict=True)
                _, got = infer_cell(mod, model, bundle, cache, subjects, mean, std)
                maximum, failed = compare_metrics(expected, method, got)
                row = {
                    "comparison": comparison, "task": "OpenBMI_ERP", "seed": seed, "fold": fold, "method": method,
                    "configuration": mode, "cudnn_tf32_allowed": cudnn_tf32, "matmul_tf32_allowed": matmul_tf32,
                    "max_metric_difference": maximum, "max_logit_difference": np.nan,
                    "historical_logits_available": False, "failed_subjects": json.dumps(failed),
                    "status": "PASS" if maximum <= 1e-8 else "FAIL",
                }
                attempts.append(row); mode_rows.append(row); del model
            attempts.append({
                "comparison": comparison, "task": "OpenBMI_ERP", "seed": seed, "fold": fold, "method": "CELL",
                "configuration": mode, "cudnn_tf32_allowed": cudnn_tf32, "matmul_tf32_allowed": matmul_tf32,
                "max_metric_difference": max(x["max_metric_difference"] for x in mode_rows), "max_logit_difference": np.nan,
                "historical_logits_available": False, "failed_subjects": "[]",
                "status": "ERP_REPLAY_REPAIRED" if all(x["status"] == "PASS" for x in mode_rows) else "ERP_REPLAY_UNRESOLVED",
            })
            del cache; torch.cuda.empty_cache()
        del bundle; gc.collect()
    attempt_frame = pd.DataFrame(attempts)
    chosen = attempt_frame[(attempt_frame.method == "CELL") & (attempt_frame.configuration == REPLAY_MODES[0][0])]
    atomic_csv(outputs / "ERP_REPLAY_REPAIR_ATTEMPTS.csv", attempt_frame)
    atomic_csv(outputs / "ERP_REPLAY_REPAIR_PROVENANCE.csv", pd.DataFrame(provenance))
    if len(chosen) != 5 or not chosen.status.eq("ERP_REPLAY_REPAIRED").all():
        raise RuntimeError("ERP historical-default replay unresolved")

    # Preserve every formerly passing cell exactly as emitted by the preceding
    # full-FP32 replay. Only the five absent/failed cells are regenerated under
    # the recovered historical-default semantics.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    repaired_x: list[pd.DataFrame] = []; repaired_xs: list[pd.DataFrame] = []; repaired_status = []
    for comparison, seed, fold in FAILED_ERP_CELLS:
        candidate_method = "LiteBN_X" if comparison == "X" else "LiteBN_XS"
        expected = historical_cell(historical, comparison, seed, fold)
        subjects = [str(v) for v in folds["OpenBMI"][fold]["outer_dev_subjects"]]
        norm_path = normalizer_path(replay, comparison, seed, fold, roots)
        mean, std, norm_meta = mod.load_tensor_pair(norm_path)
        bundle = mod.build_bundle("OpenBMI_ERP", subjects); cache = mod.RawGPUCache(bundle, device)
        pair: dict[str, pd.DataFrame] = {}; passed = True
        for method in ("LiteBN_BASELINE", candidate_method):
            ckpt = checkpoint_path(replay, comparison, seed, fold, method, roots)
            model = mod.build_model(method, "OpenBMI_ERP").to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False), strict=True)
            frame, got = infer_cell(mod, model, bundle, cache, subjects, mean, std); frame["fold"] = fold
            maximum, failed = compare_metrics(expected, method, got); passed &= maximum <= 1e-8
            repaired_status.append({"comparison": comparison, "seed": seed, "fold": fold, "method": method,
                                    "max_metric_difference": maximum, "failed_subjects": json.dumps(failed),
                                    "status": "PASS" if maximum <= 1e-8 else "FAIL"})
            pair[method] = frame; del model
        if not passed: raise RuntimeError(f"ERP repair replay failed {comparison}/{seed}/fold{fold}")
        b0_ckpt = checkpoint_path(replay, comparison, seed, fold, "LiteBN_BASELINE", roots)
        candidate_ckpt = checkpoint_path(replay, comparison, seed, fold, candidate_method, roots)
        paired = derived_pair(pair["LiteBN_BASELINE"], pair[candidate_method], candidate_method, seed,
                              sha256(b0_ckpt), sha256(candidate_ckpt), norm_meta["mean_std_sha256"])
        (repaired_x if comparison == "X" else repaired_xs).append(paired)
        del pair, paired, cache, bundle; torch.cuda.empty_cache(); gc.collect()
        print(f"ERP_REPAIRED {comparison} seed={seed} fold={fold} PASS", flush=True)
    atomic_csv(runtime / "ERP_REPAIRED_CELL_STATUS.csv", pd.DataFrame(repaired_status))

    if source_path.stat().st_size < 1_000_000:
        raise RuntimeError("PAIRED_TRIAL_RESULTS is absent or an LFS pointer")
    source = pd.read_csv(source_path, dtype={"subject_id": str})
    x_rows = pd.concat([source[source.candidate_name == "LiteBN_X"], *repaired_x], ignore_index=True)
    xs_source = pd.concat([source[source.candidate_name == "LiteBN_XS"], *repaired_xs], ignore_index=True)
    expected_counts = {"X": 4_000 + 79_200 + 4_000 + 6_195, "XS": 3 * (4_000 + 79_200 + 4_000 + 6_195)}
    if len(x_rows) != expected_counts["X"] or len(xs_source) != expected_counts["XS"]:
        raise RuntimeError(f"repaired grid cardinality mismatch X={len(x_rows)} XS={len(xs_source)}")

    fixed_rows: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for task in TASKS:
        for fold in range(5):
            cells = {seed: xs_source[(xs_source.task == task) & (xs_source.fold == fold) & (xs_source.seed == seed)].sort_values(KEY).reset_index(drop=True) for seed in (0, 1, 2)}
            fixed = cells[0]
            if fixed.empty:
                raise RuntimeError(f"missing fixed B0 cell {task}/fold{fold}")
            for seed, candidate in cells.items():
                if len(candidate) != len(fixed) or not fixed[KEY + ["true_label"]].equals(candidate[KEY + ["true_label"]]):
                    raise RuntimeError(f"fixed-B0 alignment fail {task}/fold{fold}/seed{seed}")
                if set(candidate.normalizer_sha256) != set(fixed.normalizer_sha256):
                    raise RuntimeError(f"normalizer mismatch {task}/fold{fold}/seed{seed}")
                out = fixed[KEY + ["true_label", "B0_logits", "B0_probabilities", "B0_prediction", "B0_correct",
                                  "B0_checkpoint_sha256", "normalizer_sha256", "B0_margin", "B0_entropy"]].copy()
                out["seed"] = seed; out["candidate_name"] = "LiteBN_XS"
                for col in ("candidate_logits", "candidate_probabilities", "candidate_prediction", "candidate_correct",
                            "candidate_checkpoint_sha256", "candidate_margin", "candidate_entropy",
                            "probability_L1_distance", "JS_divergence"):
                    out[col] = candidate[col].to_numpy()
                out["error_state"] = np.select(
                    [out.B0_correct & out.candidate_correct, ~out.B0_correct & out.candidate_correct,
                     out.B0_correct & ~out.candidate_correct], ["CC", "RESCUE", "HARM"], default="WW")
                fixed_rows.append(out)
                audit.append({"task": task, "fold": fold, "method": "LiteBN_XS", "seed": seed,
                              "checkpoint_sha256": str(candidate.candidate_checkpoint_sha256.iloc[0]),
                              "normalizer_sha256": str(candidate.normalizer_sha256.iloc[0]),
                              "subjects": int(candidate.subject_id.nunique()), "trials": len(candidate), "replay_status": "PASS"})
            audit.append({"task": task, "fold": fold, "method": "B0_FIXED", "seed": 0,
                          "checkpoint_sha256": str(fixed.B0_checkpoint_sha256.iloc[0]),
                          "normalizer_sha256": str(fixed.normalizer_sha256.iloc[0]),
                          "subjects": int(fixed.subject_id.nunique()), "trials": len(fixed), "replay_status": "PASS"})
    fixed = pd.concat(fixed_rows, ignore_index=True).sort_values(KEY + ["seed"]).reset_index(drop=True)
    x_rows = x_rows.sort_values(KEY + ["seed"]).reset_index(drop=True)
    gzip_options = {"method": "gzip", "mtime": 0}
    atomic_csv(runtime / "FIXED_B0_XS_ROWS.csv.gz", fixed, compression=gzip_options)
    atomic_csv(runtime / "X_ROWS.csv.gz", x_rows, compression=gzip_options)
    atomic_csv(outputs / "FIXED_B0_REPLAY_AUDIT.csv", pd.DataFrame(audit).sort_values(["task", "fold", "method", "seed"]))
    atomic_json(protocol / "SOURCE_PROVENANCE.json", {
        "primary_source": "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1",
        "source_commit": "577b1c5910abb4aa65a438b5e88417bf9939281e",
        "paired_trial_results_sha256": sha256(source_path), "historical_model_source_sha256": source_sha,
        "fivefold_split_sha256": split_sha, "tasks": list(TASKS),
        "erp_replay_semantics": {"cudnn_tf32_allowed": True, "matmul_tf32_allowed": False,
                                 "cudnn_deterministic": True, "cudnn_benchmark": False},
        "erp_previously_failed_cells": 5, "erp_repaired_cells": 5, "erp_unresolved_cells": 0,
        "B0_FIXED": "exact seed0 XS-development LiteBN baseline per task/fold",
        "fixed_b0_equal_across_candidate_seeds": True, "normalizer_equal_across_candidate_seeds": True,
        "fixed_rows_runtime_path": str(runtime / "FIXED_B0_XS_ROWS.csv.gz"), "fixed_rows_sha256": sha256(runtime / "FIXED_B0_XS_ROWS.csv.gz"),
        "x_rows_runtime_path": str(runtime / "X_ROWS.csv.gz"), "x_rows_sha256": sha256(runtime / "X_ROWS.csv.gz"),
    })
    atomic_json(protocol / "DEVELOPMENT_SCOPE_AUDIT.json", {
        "EXPERIMENT_TYPE": "DEVELOPMENT_ANALYSIS_ONLY", "NEW_EEG_MODEL_TRAINED": "NO",
        "FINAL_HELDOUT_ACCESSED": "NO", "INTERNAL_HELDOUT_ACCESSED": "NO", "DEVELOPMENT_OUTER_ONLY": "YES",
        "DIAGNOSTIC_LINEAR_MODEL_FIT": "NOT_YET", "optimizer_instantiated": False,
        "ERP_REPLAY_STATUS": "ERP_FULLY_INCLUDED", "final_test_result_artifacts_read": False,
    })
    print(f"FIXED_B0_ROWS_COMPLETE xs_rows={len(fixed)} x_rows={len(x_rows)} ERP_FULLY_INCLUDED", flush=True)


if __name__ == "__main__":
    main()
