#!/usr/bin/env python3
"""Train and evaluate the exact W0R0 seed-0 candidate on Linux.

The already exposed development-model-selection cohort is opened only after all
20 W0R0 checkpoints are frozen. No sealed or new final-test cohort is accessed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("W0R0_REPO", "/root/rivermind-data/CRCICLR_W0R0_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_linux_w0r0_screen_v1"
CODE = EXP / "code"
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("W0R0_RUNTIME", "/root/rivermind-data/linux_w0r0_seed0_runtime")).resolve()
BASE_CODE = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py"
NORMALIZER_SOURCE = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime/normalizers")
CARRIER_RUNTIME = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")
TASK_RUNTIME = Path("/root/rivermind-data/openbmi_task_generality_runtime")
SEED = 0

INTERNAL_HELDOUT = {
    "OpenBMI": ["4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54"],
    "WBCIC": ["sub-2", "sub-3", "sub-17", "sub-19", "sub-21", "sub-25", "sub-31", "sub-33", "sub-38", "sub-42"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_modules():
    os.environ["LITEBN_X_REPO"] = str(REPO)
    spec = importlib.util.spec_from_file_location("w0r0_base", BASE_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import historical training implementation")
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    model_spec = importlib.util.spec_from_file_location("w0r0_model", CODE / "models_w0r0.py")
    if model_spec is None or model_spec.loader is None:
        raise RuntimeError("cannot import W0R0 implementation")
    models = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = models
    model_spec.loader.exec_module(models)
    base.RUNTIME = RUNTIME
    base.SEED = SEED
    return base, models


def build_w0r0(base, models, task: str):
    task_spec = base.TASKS[task]
    return models.W0R0(
        int(task_spec["channels"]),
        int(task_spec["classes"]),
        base.LiteBNStem,
        base.ResidualTemporalBlock,
    )


def baseline_path(task: str, fold: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        short = "erp" if task.endswith("ERP") else "ssvep"
        return TASK_RUNTIME / f"{short}_fold{fold}_seed0_litebn/selected_best.pt"
    short = "openbmi" if task == "OpenBMI_MI" else "wbcic"
    return CARRIER_RUNTIME / f"{short}_fold{fold}_seed0_litebn/selected_best.pt"


def normalizer_source(task: str, fold: int) -> Path:
    return NORMALIZER_SOURCE / f"{task.lower()}_fold{fold}.npz"


def candidate_path(base, task: str, fold: int) -> Path:
    return base.checkpoint_path(task, fold, "W0R0", "selected_best.pt")


def preflight(base, models) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    search, folds, split_hash = base.load_folds()
    baseline_rows: list[dict[str, Any]] = []
    normalizer_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for task in base.TASK_ORDER:
        b0 = base.build_model("LiteBN_BASELINE", task)
        w0r0 = build_w0r0(base, models, task)
        test = torch.zeros(2, base.TASKS[task]["channels"], base.TASKS[task]["samples"])
        with torch.no_grad():
            logits, hidden = w0r0(test)
        if tuple(logits.shape) != (2, base.TASKS[task]["classes"]) or tuple(hidden.shape) != (2, 64):
            raise RuntimeError(f"W0R0 shape mismatch: {task}")
        if float(w0r0.lambda_channel) != 0.0 or float(w0r0.lambda_scale) != 0.0:
            raise RuntimeError("gate lambda initialization mismatch")
        if any(float(block.gamma) != np.float32(1e-3) for block in w0r0.mixers):
            raise RuntimeError("mixer gamma initialization mismatch")
        parameter_rows.extend([
            {"task": task, "model": "B0", "parameter_count": base.parameter_count(b0), "representation_dimension": 64},
            {"task": task, "model": "W0R0", "parameter_count": base.parameter_count(w0r0), "representation_dimension": 64},
        ])
        dataset = base.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            checkpoint = baseline_path(task, fold_id)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            b0.load_state_dict(state, strict=True)
            baseline_rows.append({
                "task": task,
                "fold": fold_id,
                "seed": 0,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "strict_load": True,
                "depth1_weight_shape": str(tuple(state["depth1.weight"].shape)),
                "head_weight_shape": str(tuple(state["head.weight"].shape)),
                "architecture": "historical_64_feature_CompactLite_BN",
            })
            path = normalizer_source(task, fold_id)
            if not path.is_file():
                raise FileNotFoundError(path)
            mean, std, metadata = base.load_tensor_pair(path)
            allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            bundle = base.build_bundle(task, allowed)
            check_mean, check_std, check_metadata = base.normalizer(bundle, fold["inner_train_subjects"])
            exact = bool(np.array_equal(mean, check_mean) and np.array_equal(std, check_std))
            if not exact or metadata["mean_std_sha256"] != check_metadata["mean_std_sha256"]:
                raise RuntimeError(f"normalizer mismatch: {task} f{fold_id}")
            destination = RUNTIME / "normalizers" / f"{task.lower()}_fold{fold_id}.npz"
            base.save_tensor_pair(destination, mean, std, metadata)
            normalizer_rows.append({
                "task": task,
                "fold": fold_id,
                "seed": 0,
                "normalizer_sha256": metadata["mean_std_sha256"],
                "recomputed_exact": exact,
                "shared_by_B0_and_W0R0": True,
                "source_path": str(path),
            })
            del bundle
            gc.collect()
    write_csv(OUT / "BASELINE_ARCHITECTURE_AUDIT.csv", pd.DataFrame(baseline_rows))
    write_csv(OUT / "NORMALIZER_AUDIT.csv", pd.DataFrame(normalizer_rows))
    write_csv(OUT / "PARAMETER_COUNTS.csv", pd.DataFrame(parameter_rows))
    runtime = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }
    write_json(PROTOCOL / "LINUX_RUNTIME_AUDIT.json", runtime)
    write_json(PROTOCOL / "SOURCE_PROVENANCE.json", {
        "linux_adaptation_of": "persist_eeg_windows_wr_final_screen_v1 W0R0-only seed0 screen",
        "base_training_source": str(BASE_CODE),
        "base_training_source_sha256": sha256(BASE_CODE),
        "w0r0_source_sha256": sha256(CODE / "models_w0r0.py"),
        "runner_sha256": sha256(Path(__file__)),
        "fivefold_split_sha256": split_hash,
        "baseline": "strict replay of existing matched Linux historical 64-feature LiteBN seed0 checkpoints",
        "baseline_retrained": False,
    })
    write_json(PROTOCOL / "DATA_SCOPE_AUDIT.json", {
        "search_subject_counts": {key: len(value) for key, value in search.items()},
        "internal_heldout_status": "DEVELOPMENT_MODEL_SELECTION_DATA",
        "internal_heldout_subject_counts": {key: len(value) for key, value in INTERNAL_HELDOUT.items()},
        "new_sealed_test_accessed": False,
        "final_test_accessed": False,
        "heldout_labels_opened_during_preflight": False,
    })
    print("W0R0_PREFLIGHT_PASS", flush=True)


def train(base, models) -> None:
    _, folds, _ = base.load_folds()
    records_path = RUNTIME / "TRAINING_LOGS.json"
    existing = json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else []
    records = {(row["task"], int(row["fold"])): row for row in existing}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for task in base.TASK_ORDER:
        dataset = base.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            bundle = base.build_bundle(task, allowed)
            mean, std, norm_meta = base.load_tensor_pair(RUNTIME / "normalizers" / f"{task.lower()}_fold{fold_id}.npz")
            cache = base.RawGPUCache(bundle, device)
            if base.TASKS[task]["mi_protocol"]:
                episodes, manifest_meta = base.mi_manifest(bundle, fold, task)
                batch_info = {**manifest_meta, "episodes": episodes}
            else:
                batch_info = {"kind": "historical_task_full_permutation_batch64", "manifest_sha256": None, "steps_per_epoch": None}
            weight, weight_meta = base.class_weights(bundle, fold["inner_train_subjects"])
            base.set_seed(SEED)
            model = build_w0r0(base, models, task).to(device)
            initial_hash = base.state_hash(model)
            base.set_seed(SEED + 100_000)
            record = base.train_one(model, "W0R0", task, fold, bundle, cache, mean, std, norm_meta, batch_info, weight, weight_meta, device)
            record["initial_sha256"] = initial_hash
            records[(task, fold_id)] = record
            write_json(records_path, list(records.values()))
            print(f"W0R0_FOLD_COMPLETE {task} fold={fold_id} selected_epoch={record['selected_epoch']}", flush=True)
            del model, cache, bundle
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(records) != 20:
        raise RuntimeError(f"incomplete training grid: {len(records)}/20")
    compact = [{key: value for key, value in row.items() if key != "history"} for row in records.values()]
    write_csv(OUT / "W0R0_TRAINING_SUMMARY.csv", pd.DataFrame(compact).sort_values(["task", "fold"]))
    write_csv(OUT / "INITIALIZATION_AUDIT.csv", pd.DataFrame([{
        "task": row["task"], "fold": row["fold"], "seed": 0,
        "model": "W0R0", "initial_sha256": row["initial_sha256"],
    } for row in compact]).sort_values(["task", "fold"]))


def eval_scope(base, models, scope: str) -> None:
    training = pd.read_csv(OUT / "W0R0_TRAINING_SUMMARY.csv")
    if len(training) != 20:
        raise RuntimeError("all 20 W0R0 checkpoints must be frozen before evaluation")
    _, folds, _ = base.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for task in base.TASK_ORDER:
        dataset = base.TASKS[task]["dataset"]
        subjects = None
        if scope == "internal_heldout":
            subjects = INTERNAL_HELDOUT[dataset]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            evaluation_subjects = fold["outer_dev_subjects"] if scope == "outer" else subjects
            bundle = base.build_bundle(task, evaluation_subjects)
            mean, std, norm_meta = base.load_tensor_pair(RUNTIME / "normalizers" / f"{task.lower()}_fold{fold_id}.npz")
            cache = base.RawGPUCache(bundle, device)
            checkpoints = {"B0": baseline_path(task, fold_id), "W0R0": candidate_path(base, task, fold_id)}
            for method, checkpoint in checkpoints.items():
                model = base.build_model("LiteBN_BASELINE", task) if method == "B0" else build_w0r0(base, models, task)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
                model = model.to(device)
                metrics = base.evaluate(model, bundle, cache, evaluation_subjects, mean, std)
                for subject, value in metrics.items():
                    rows.append({
                        "scope": "DEVELOPMENT_OUTER" if scope == "outer" else "DEVELOPMENT_MODEL_SELECTION_DATA",
                        "task": task, "dataset": dataset, "fold": fold_id,
                        "seed": 0, "subject_id": str(subject), "model": method,
                        **value, "checkpoint_sha256": sha256(checkpoint),
                        "normalizer_sha256": norm_meta["mean_std_sha256"],
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del cache, bundle
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"{scope.upper()}_FOLD_COMPLETE {task} fold={fold_id}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["task", "fold", "subject_id", "model"])
    if scope == "outer":
        expected = 2 * (
            3 * sum(len(fold["outer_dev_subjects"]) for fold in folds["OpenBMI"])
            + sum(len(fold["outer_dev_subjects"]) for fold in folds["WBCIC"])
        )
        if len(frame) != expected:
            raise RuntimeError(f"outer cardinality mismatch: {len(frame)}/{expected}")
        write_csv(OUT / "STAGE_A_OUTER_SUBJECT_RESULTS.csv", frame)
    else:
        expected = 2 * 5 * (3 * len(INTERNAL_HELDOUT["OpenBMI"]) + len(INTERNAL_HELDOUT["WBCIC"]))
        if len(frame) != expected:
            raise RuntimeError(f"internal-heldout cardinality mismatch: {len(frame)}/{expected}")
        write_csv(OUT / "STAGE_A_HELDOUT_SUBJECT_RESULTS.csv", frame)


def summarize_scope(path: Path, output: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    per_subject = frame.groupby(["task", "subject_id", "model"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    task = per_subject.groupby(["task", "model"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean()
    wide = task.pivot(index="task", columns="model", values="BA")
    deltas = ((wide["W0R0"] - wide["B0"]) * 100.0).rename("delta_vs_B0_pp").reset_index()
    task = task.merge(deltas, on="task", how="left")
    task.loc[task.model == "B0", "delta_vs_B0_pp"] = 0.0
    write_csv(output, task.sort_values(["task", "model"]))
    return task


def aggregate(base) -> None:
    outer = summarize_scope(OUT / "STAGE_A_OUTER_SUBJECT_RESULTS.csv", OUT / "STAGE_A_OUTER_TASK_SUMMARY.csv")
    held = summarize_scope(OUT / "STAGE_A_HELDOUT_SUBJECT_RESULTS.csv", OUT / "STAGE_A_HELDOUT_TASK_SUMMARY.csv")
    outer_delta = outer[outer.model == "W0R0"].set_index("task").loc[list(base.TASK_ORDER), "delta_vs_B0_pp"]
    held_delta = held[held.model == "W0R0"].set_index("task").loc[list(base.TASK_ORDER), "delta_vs_B0_pp"]
    held_pass = bool(held_delta.mean() > 0.0 and (held_delta >= 0.0).sum() >= 3 and held_delta.min() > -0.75)
    outer_pass = bool(outer_delta.mean() > 0.0 and (outer_delta > 0.0).sum() >= 2 and outer_delta.min() > -1.0)
    passed = held_pass and outer_pass
    decision = {
        "terminal": "LINUX_W0R0_SEED0_PASSED" if passed else "LINUX_W0R0_SEED0_FAILED",
        "model": "W0R0", "seed": 0,
        "outer_equal_task_mean_delta_pp": float(outer_delta.mean()),
        "outer_positive_tasks": int((outer_delta > 0.0).sum()),
        "outer_worst_task_delta_pp": float(outer_delta.min()),
        "heldout_equal_task_mean_delta_pp": float(held_delta.mean()),
        "heldout_nonnegative_tasks": int((held_delta >= 0.0).sum()),
        "heldout_worst_task_delta_pp": float(held_delta.min()),
        "heldout_eligibility_pass": held_pass,
        "outer_safety_gate_pass": outer_pass,
        "protocol_audit_pass": True,
        "internal_heldout_status": "DEVELOPMENT_MODEL_SELECTION_DATA",
        "new_sealed_test_accessed": False,
        "final_test_accessed": False,
        "multiseed_run": False,
    }
    write_json(OUT / "STAGE_A_SELECTION_DECISION.json", decision)
    write_json(OUT / "FINAL_LINUX_W0R0_DECISION.json", decision)
    lines = [
        "# Linux W0R0 seed-0 screen", "",
        "This run evaluates only B0 and W0R0. It is a Linux adaptation requested after the Windows-only protocol was written; it is not represented as a Windows result.", "",
        "| Task | B0 outer BA | W0R0 outer BA | delta pp | B0 heldout BA | W0R0 heldout BA | delta pp |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task_name in base.TASK_ORDER:
        o = outer[outer.task == task_name].set_index("model")
        h = held[held.task == task_name].set_index("model")
        lines.append(f"| {task_name} | {o.loc['B0','BA']:.4f} | {o.loc['W0R0','BA']:.4f} | {o.loc['W0R0','delta_vs_B0_pp']:+.3f} | {h.loc['B0','BA']:.4f} | {h.loc['W0R0','BA']:.4f} | {h.loc['W0R0','delta_vs_B0_pp']:+.3f} |")
    lines += ["", f"Terminal: `{decision['terminal']}`.", "", f"Outer equal-task mean delta: {decision['outer_equal_task_mean_delta_pp']:+.3f} pp.", f"Internal-heldout equal-task mean delta: {decision['heldout_equal_task_mean_delta_pp']:+.3f} pp.", "", "INTERNAL_HELDOUT_STATUS = DEVELOPMENT_MODEL_SELECTION_DATA", "", "NEW_SEALED_TEST_ACCESSED = NO", "FINAL_TEST_ACCESSED = NO", "", "This seed-0 screen cannot establish a final model. No seed1/2 run was performed in this W0R0-only scope."]
    report = "\n".join(lines) + "\n"
    (OUT / "STAGE_A_SELECTION_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "FINAL_LINUX_W0R0_REPORT.md").write_text(report, encoding="utf-8")
    print(decision["terminal"], flush=True)


def efficiency(base, models) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for task in base.TASK_ORDER:
        shape = (1, base.TASKS[task]["channels"], base.TASKS[task]["samples"])
        for name in ("B0", "W0R0"):
            model = base.build_model("LiteBN_BASELINE", task) if name == "B0" else build_w0r0(base, models, task)
            model.eval()
            cpu_input = torch.zeros(shape)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
                with torch.no_grad():
                    model(cpu_input)
            macs = sum(int(event.flops or 0) for event in profile.key_averages()) / 2.0
            model = model.to(device)
            value = torch.zeros(shape, device=device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(20): model(value)
                if device.type == "cuda": torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(100): model(value)
                if device.type == "cuda": torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start) * 10.0
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            rows.append({"task": task, "model": name, "parameter_count": base.parameter_count(model), "approximate_MACs": int(macs), "batch1_latency_ms": latency_ms, "peak_allocated_bytes": peak, "device": str(device)})
            del model, value
            if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(OUT / "PARAMETER_MAC_LATENCY_SUMMARY.csv", pd.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "train", "outer", "heldout", "aggregate", "efficiency", "all"))
    args = parser.parse_args()
    base, models = load_modules()
    if args.stage in ("preflight", "all"): preflight(base, models)
    if args.stage in ("train", "all"): train(base, models)
    if args.stage in ("outer", "all"): eval_scope(base, models, "outer")
    if args.stage in ("heldout", "all"): eval_scope(base, models, "internal_heldout")
    if args.stage in ("aggregate", "all"): aggregate(base)
    if args.stage in ("efficiency", "all"): efficiency(base, models)


if __name__ == "__main__":
    main()
