"""Matched B0/LiteBN-G2 evaluation on already-exposed development benchmarks."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from litebn_g2 import LiteBNG2

OPENBMI_V8_INTERNAL = ("4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54")
WBCIC_FINAL_TRUE_OUTER_EXPOSED = ("sub-2", "sub-3", "sub-17", "sub-19", "sub-21", "sub-25", "sub-31", "sub-33", "sub-38", "sub-42")
BENCHMARK = {"OpenBMI_SSVEP": (.9234, "TCFormer"), "OpenBMI_ERP": (.8557, "TCFormer"), "OpenBMI_MI": (.7555, "TCFormer"), "WBCIC_MI": (.7918, "EEGNet")}


def subjects(task: str) -> tuple[str, ...]:
    return OPENBMI_V8_INTERNAL if task.startswith("OpenBMI") else WBCIC_FINAL_TRUE_OUTER_EXPOSED


def evaluate_task(*, base, runner, task: str, folds: list[dict], checkpoints: dict[int, Path], device: torch.device, output: Path) -> tuple[dict, pd.DataFrame]:
    rows, selected_subjects = [], subjects(task)
    for fold in folds:
        fold_id = int(fold["fold_id"]); bundle = base.build_bundle(task, selected_subjects)
        if set(bundle.subjects) != set(selected_subjects): raise RuntimeError("exposed benchmark membership mismatch")
        mean, std, metadata = base.load_tensor_pair(runner.normalizer_source(task, fold_id)); cache = base.RawGPUCache(bundle, device)
        b0_path = runner.baseline_path(task, fold_id)
        b0 = base.build_model("LiteBN_BASELINE", task); b0.load_state_dict(torch.load(b0_path, map_location="cpu", weights_only=False), strict=True); b0 = b0.to(device).eval()
        g = LiteBNG2(base.build_model("LiteBN_BASELINE", task)); payload = torch.load(checkpoints[fold_id], map_location="cpu", weights_only=False); g.load_state_dict(payload["state_dict"], strict=True); g = g.to(device).eval()
        for name, model, path in (("B0", b0, b0_path), ("LiteBN-G2", g, checkpoints[fold_id])):
            for subject, metrics in base.evaluate(model, bundle, cache, selected_subjects, mean, std).items():
                rows.append({"scope": "EXPOSED_DEVELOPMENT_BENCHMARK", "task": task, "dataset": base.TASKS[task]["dataset"], "fold": fold_id,
                             "seed": 0, "subject_id": str(subject), "model": name, **metrics, "checkpoint_path": str(path),
                             "checkpoint_sha256": runner.sha256(path), "normalizer_sha256": metadata["mean_std_sha256"]})
        del b0, g, cache, bundle
        if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["fold", "subject_id", "model"])
    aggregate = frame.groupby(["subject_id", "model"], as_index=False)[["BA", "macro_F1", "accuracy"]].mean().pivot(index="subject_id", columns="model", values="BA")
    if set(aggregate.columns) != {"B0", "LiteBN-G2"}: raise RuntimeError("incomplete matched evaluation")
    b0, g = float(aggregate.B0.mean()), float(aggregate["LiteBN-G2"].mean()); reference, reference_model = BENCHMARK[task]
    result = {"task": task, "scope": "EXPOSED_DEVELOPMENT_BENCHMARK", "resource": "OpenBMI_V8_INTERNAL" if task.startswith("OpenBMI") else "WBCIC_FINAL_TRUE_OUTER_EXPOSED_DEVELOPMENT",
              "subjects": len(aggregate), "folds": len(folds), "benchmark_best": reference, "benchmark_model": reference_model,
              "matched_LiteBN_BA": b0, "LiteBN_G2_BA": g, "delta_vs_LiteBN_pp": 100 * (g - b0), "delta_vs_benchmark_pp": 100 * (g - reference)}
    output.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(output, index=False)
    return result, frame
