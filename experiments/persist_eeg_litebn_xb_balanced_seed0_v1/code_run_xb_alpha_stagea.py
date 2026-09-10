#!/usr/bin/env python3
"""Inner-validation-only LiteBN-XB fixed-alpha candidate runner.

The runner deliberately does not build an outer-development bundle.  It trains
the same single XB architecture under a named conservative channel-gate cap and
emits only Stage-A artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")
BASE_EXP = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1"
EXP = REPO / "experiments/persist_eeg_litebn_xb_balanced_seed0_v1"
HIST_CARRIER = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")
HIST_TASK = Path("/root/rivermind-data/openbmi_task_generality_runtime")


def module_for(runtime: Path, label: str, alpha: float):
    source = BASE_EXP / "code/litebn_x.py"
    spec = importlib.util.spec_from_file_location(f"xb_{label}", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LiteBN-X implementation")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.RUNTIME = runtime
    mod.SEED = 0
    original_build = mod.build_model

    def build(architecture: str, task: str):
        if architecture == label:
            model = original_build("LiteBN_XS", task)
            model.architecture = label
            return model
        return original_build(architecture, task)

    def conservative_gate(self, value: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
        difference = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + 1e-8)
        raw = self.channel_mlp(torch.stack((rms, difference), dim=-1)).squeeze(-1)
        return value * (1.0 + alpha * torch.tanh(self.lambda_channel) * torch.tanh(raw)).unsqueeze(-1)

    mod.build_model = build
    mod.LiteBNEnhanced.apply_channel_gate = conservative_gate
    return mod


def baseline_path(task: str, fold: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        prefix = "erp" if task.endswith("ERP") else "ssvep"
        return HIST_TASK / f"{prefix}_fold{fold}_seed0_litebn/selected_best.pt"
    prefix = "openbmi" if task.startswith("OpenBMI") else "wbcic"
    return HIST_CARRIER / f"{prefix}_fold{fold}_seed0_litebn/selected_best.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    label = f"LiteBN_XB_{args.tag.upper()}"
    runtime = Path(f"/root/rivermind-data/litebn_xb_{args.tag}_seed0_runtime")
    output = EXP / "outputs"
    protocol = EXP / "protocol"
    runtime.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    protocol.mkdir(parents=True, exist_ok=True)
    mod = module_for(runtime, label, args.alpha)
    source_config = {
        "architecture": label,
        "mother_architecture": "LiteBN_XS",
        "alpha_cap": args.alpha,
        "gate": "1 + alpha*tanh(lambda_channel)*tanh(shared_mlp(RMS,DiffRMS))",
        "stage": "A inner-validation only",
        "seed": 0,
    }
    config_hash = hashlib.sha256(json.dumps(source_config, sort_keys=True).encode()).hexdigest()
    search, folds, split_hash = mod.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, task_fold, diagnostics = [], [], []
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            if set(allowed) & set(fold["outer_dev_subjects"]):
                raise RuntimeError("Stage-A subject overlap")
            bundle = mod.build_bundle(task, allowed)
            mean, std, norm = mod.normalizer(bundle, fold["inner_train_subjects"])
            mod.save_tensor_pair(runtime / "normalizers" / f"{task.lower()}_fold{fold_id}.npz", mean, std, norm)
            cache = mod.RawGPUCache(bundle, device)
            if mod.TASKS[task]["mi_protocol"]:
                episodes, metadata = mod.mi_manifest(bundle, fold, task)
                batch_info = {**metadata, "episodes": episodes}
            else:
                batch_info = {"kind": "historical_task_full_permutation_batch64", "manifest_sha256": None, "steps_per_epoch": None}
            weight, weight_meta = mod.class_weights(bundle, fold["inner_train_subjects"])
            mod.set_seed(0)
            model = mod.build_model(label, task).to(device)
            mod.set_seed(100_000)
            record = mod.train_one(model, label, task, fold, bundle, cache, mean, std, norm, batch_info, weight, weight_meta, device)
            record = {key: value for key, value in record.items() if key != "history"}
            records.append(record)
            candidate_ba = float(record["best_inner_val_BA"])
            state = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=False)
            lam = float(state["lambda_channel"].item())
            diagnostics.append({"task": task, "fold": fold_id, "alpha_cap": args.alpha, "tanh_lambda_channel": float(np.tanh(lam)), "effective_gate_amplitude": float(args.alpha * np.tanh(lam))})
            baseline = mod.build_model("LiteBN_BASELINE", task).to(device)
            path = baseline_path(task, fold_id)
            baseline.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
            values = mod.evaluate(baseline, bundle, cache, fold["inner_val_subjects"], mean, std)
            baseline_ba = float(np.mean([row["BA"] for row in values.values()]))
            task_fold.append({"task": task, "fold": fold_id, "architecture": label, "alpha_cap": args.alpha, "LiteBN_inner_val_BA": baseline_ba, "candidate_inner_val_BA": candidate_ba, "delta_pp": (candidate_ba - baseline_ba) * 100.0, "selected_epoch": record["selected_epoch"], "parameter_count": record["parameter_count"]})
            print(f"XB_{args.tag}_TRAINED {task} fold={fold_id} epoch={record['selected_epoch']}", flush=True)
            del model, baseline, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    frame = pd.DataFrame(records).sort_values(["task", "fold"])
    detail = pd.DataFrame(task_fold).sort_values(["task", "fold"])
    by_task = detail.groupby("task", as_index=False).agg(candidate_inner_val_BA=("candidate_inner_val_BA", "mean"), LiteBN_inner_val_BA=("LiteBN_inner_val_BA", "mean"), delta_pp=("delta_pp", "mean"), positive_folds=("delta_pp", lambda value: int((value > 0).sum())))
    result = {"architecture": label, "tag": args.tag, "alpha_cap": args.alpha, "source_config_sha256": config_hash, "split_sha256": split_hash, "selection_metric": "maximize worst-task delta, then equal-task mean, then fold consistency"}
    for row in by_task.itertuples(index=False):
        result[f"{row.task}_inner_val_BA"] = row.candidate_inner_val_BA
        result[f"{row.task}_delta_pp"] = row.delta_pp
    result.update({"worst_task_delta_pp": float(by_task.delta_pp.min()), "equal_task_mean_delta_pp": float(by_task.delta_pp.mean()), "positive_tasks": int((by_task.delta_pp > 0).sum()), "positive_folds": int((detail.delta_pp > 0).sum()), "max_parameter_count": int(detail.parameter_count.max()), "eligible": bool((by_task.delta_pp > 0).all()), "outer_dev_predictions_generated": False, "final_holdout_accessed": False})
    frame.to_csv(output / f"XB_{args.tag}_STAGE_A_INNERVAL_RESULTS.csv", index=False)
    detail.to_csv(output / f"XB_{args.tag}_STAGE_A_INNERVAL_TASK_FOLD.csv", index=False)
    pd.DataFrame([result]).to_csv(output / f"XB_{args.tag}_STAGE_A_CANDIDATE_SUMMARY.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / f"XB_{args.tag}_GATE_DIAGNOSTICS_INNERVAL.csv", index=False)
    (protocol / f"XB_{args.tag}_STAGE_A_PROTOCOL.json").write_text(json.dumps({**source_config, "source_config_sha256": config_hash, "split_sha256": split_hash, "outer_dev_predictions_generated": False, "final_holdout_accessed": False}, indent=2, sort_keys=True) + "\n")
    print("XB_STAGE_A_CANDIDATE_DONE", json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
