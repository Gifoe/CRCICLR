#!/usr/bin/env python3
"""Evaluate one fully frozen W0R0 task against historical B0 immediately."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

RUNNER = Path(__file__).with_name("run_w0r0_seed0.py")
spec = importlib.util.spec_from_file_location("w0r0_runner", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import W0R0 runner")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    per_subject = frame.groupby(
        ["task", "subject_id", "model"], as_index=False
    )[["BA", "macro_F1", "accuracy"]].mean()
    task = per_subject.groupby(["task", "model"], as_index=False)[
        ["BA", "macro_F1", "accuracy"]
    ].mean()
    indexed = task.set_index("model")
    delta = float((indexed.loc["W0R0", "BA"] - indexed.loc["B0", "BA"]) * 100.0)
    task["delta_vs_B0_pp"] = task.model.map({"B0": 0.0, "W0R0": delta})
    return task, delta


def evaluate(task: str) -> None:
    base, models = runner.load_modules()
    if task not in base.TASK_ORDER:
        raise ValueError(task)
    records_path = runner.RUNTIME / "TRAINING_LOGS.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    frozen = [row for row in records if row["task"] == task]
    if len(frozen) != 5 or {int(row["fold"]) for row in frozen} != set(range(5)):
        raise RuntimeError(f"{task} is not frozen across all five folds")
    for row in frozen:
        checkpoint = Path(row["checkpoint_path"])
        if not checkpoint.is_file() or runner.sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError(f"candidate checkpoint audit failed: {checkpoint}")

    _, folds, _ = base.load_folds()
    dataset = base.TASKS[task]["dataset"]
    task_dir = runner.OUT / "task_immediate" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for scope in ("outer", "internal_heldout"):
        rows = []
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            subjects = (
                fold["outer_dev_subjects"]
                if scope == "outer"
                else runner.INTERNAL_HELDOUT[dataset]
            )
            bundle = base.build_bundle(task, subjects)
            mean, std, metadata = base.load_tensor_pair(
                runner.RUNTIME / "normalizers" / f"{task.lower()}_fold{fold_id}.npz"
            )
            cache = base.RawGPUCache(bundle, device)
            checkpoints = {
                "B0": runner.baseline_path(task, fold_id),
                "W0R0": runner.candidate_path(base, task, fold_id),
            }
            for method, checkpoint in checkpoints.items():
                model = (
                    base.build_model("LiteBN_BASELINE", task)
                    if method == "B0"
                    else runner.build_w0r0(base, models, task)
                )
                model.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=False),
                    strict=True,
                )
                model = model.to(device)
                metrics = base.evaluate(model, bundle, cache, subjects, mean, std)
                for subject, value in metrics.items():
                    rows.append({
                        "scope": "DEVELOPMENT_OUTER" if scope == "outer" else "DEVELOPMENT_MODEL_SELECTION_DATA",
                        "task": task,
                        "dataset": dataset,
                        "fold": fold_id,
                        "seed": 0,
                        "subject_id": str(subject),
                        "model": method,
                        **value,
                        "checkpoint_sha256": runner.sha256(checkpoint),
                        "normalizer_sha256": metadata["mean_std_sha256"],
                    })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del cache, bundle
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        frame = pd.DataFrame(rows).sort_values(["fold", "subject_id", "model"])
        expected_subjects = 40 if scope == "outer" and dataset == "OpenBMI" else 31 if scope == "outer" else len(runner.INTERNAL_HELDOUT[dataset]) * 5
        if len(frame) != expected_subjects * 2:
            raise RuntimeError(f"{task} {scope} cardinality mismatch: {len(frame)}")
        summary, delta = summarize(frame)
        runner.write_csv(task_dir / f"{scope.upper()}_SUBJECT_RESULTS.csv", frame)
        runner.write_csv(task_dir / f"{scope.upper()}_SUMMARY.csv", summary)
        results[scope] = {"summary": summary, "delta": delta}

    outer = results["outer"]["summary"].set_index("model")
    held = results["internal_heldout"]["summary"].set_index("model")
    decision = {
        "task": task,
        "seed": 0,
        "status": "TASK_IMMEDIATE_COMPARISON_COMPLETE",
        "outer_B0_BA": float(outer.loc["B0", "BA"]),
        "outer_W0R0_BA": float(outer.loc["W0R0", "BA"]),
        "outer_delta_vs_B0_pp": results["outer"]["delta"],
        "heldout_B0_BA": float(held.loc["B0", "BA"]),
        "heldout_W0R0_BA": float(held.loc["W0R0", "BA"]),
        "heldout_delta_vs_B0_pp": results["internal_heldout"]["delta"],
        "internal_heldout_status": "DEVELOPMENT_MODEL_SELECTION_DATA",
        "new_sealed_test_accessed": False,
        "final_test_accessed": False,
    }
    runner.write_json(task_dir / "TASK_DECISION.json", decision)
    print(json.dumps(decision, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"))
    evaluate(parser.parse_args().task)


if __name__ == "__main__":
    main()
