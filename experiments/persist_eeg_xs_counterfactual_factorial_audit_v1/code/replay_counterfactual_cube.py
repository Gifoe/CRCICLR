#!/usr/bin/env python3
"""Replay the exact frozen XS checkpoints under the predeclared 2^3 C/S/M cube."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

KEY = ["task", "fold", "subject_id", "session", "trial_id"]
TASK_CODES = {"OpenBMI_MI": "OMI", "OpenBMI_ERP": "OERP", "OpenBMI_SSVEP": "OSSVEP", "WBCIC_MI": "WMI"}
STATES = (
    ("C0S0M0", 0, 0, 0), ("C1S0M0", 1, 0, 0),
    ("C0S1M0", 0, 1, 0), ("C0S0M1", 0, 0, 1),
    ("C1S1M0", 1, 1, 0), ("C1S0M1", 1, 0, 1),
    ("C0S1M1", 0, 1, 1), ("C1S1M1", 1, 1, 1),
)


def load_py(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack(series.map(json.loads))


def subject_equal_ba(frame: pd.DataFrame, prediction: str) -> float:
    return float(np.mean([balanced_accuracy_score(group.true_label, group[prediction])
                          for _, group in frame.groupby("subject_id", sort=True)]))


def set_state(model: torch.nn.Module, trained: dict[str, Any], c: int, s: int, m: int) -> None:
    with torch.no_grad():
        model.lambda_channel.copy_(trained["lambda_channel"] if c else torch.zeros_like(model.lambda_channel))
        model.lambda_scale.copy_(trained["lambda_scale"] if s else torch.zeros_like(model.lambda_scale))
        for block, value in zip(model.mixer, trained["mixer_gamma"]):
            block.gamma.copy_(value if m else torch.zeros_like(block.gamma))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    repo, runtime = args.repo.resolve(), args.runtime.resolve()
    exp = repo / "experiments/persist_eeg_xs_counterfactual_factorial_audit_v1"
    outputs = exp / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    prior = repo / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1"
    source_path = Path("/root/rivermind-data/stable_rescue_predictability_runtime_v1/FIXED_B0_XS_ROWS.csv.gz")
    roots = {
        "seed0": Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime"),
        "xs": Path("/root/rivermind-data/xs_full_multiseed_finaltest_runtime"),
        "erp": Path("/root/rivermind-data/xs_erp_seed12_stability_runtime_correct_xs"),
        "carrier": Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime"),
        "task": Path("/root/rivermind-data/openbmi_task_generality_runtime"),
    }
    os.environ.update({"LITEBN_X_REPO": str(repo), "PERSIST_CACHE_ROOT": "/root/rivermind-data/persist_eeg_cache",
                       "LITEBN_X_RUNTIME": str(roots["seed0"])})
    helper = load_py("counterfactual_replay_helpers", prior / "code/replay_models.py")
    mod = helper.load_model_module(repo, Path("/root/rivermind-data/persist_eeg_cache"), roots["seed0"])
    _, folds, _ = mod.load_folds()
    source = pd.read_csv(source_path, dtype={"subject_id": str})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for exact historical replay")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output = outputs / "COUNTERFACTUAL_TRIAL_RESULTS.csv"
    part = output.with_name(output.name + ".part")
    if part.exists():
        part.unlink()
    audits: list[dict[str, Any]] = []
    wrote_header = False
    valid_cells = 0
    for task in mod.TASK_ORDER:
        for fold in range(5):
            fold_spec = folds[mod.TASKS[task]["dataset"]][fold]
            subjects = [str(value) for value in fold_spec["outer_dev_subjects"]]
            bundle = mod.build_bundle(task, subjects)
            for seed in (0, 1, 2):
                cell = source[(source.task == task) & (source.fold == fold) & (source.seed == seed)].copy()
                candidate_path = helper.xs_checkpoint(roots["seed0"], roots["xs"], roots["erp"], task, fold, seed)
                normalizer_path = helper.normalizer_path(roots["seed0"], roots["xs"], roots["erp"], "XS", task, fold, seed)
                mean, std, _ = mod.load_tensor_pair(normalizer_path)
                historical_default = bool(task == "OpenBMI_ERP" and (seed, fold) in {(1, 0), (1, 3), (2, 0), (2, 3)})
                torch.backends.cudnn.allow_tf32 = historical_default
                torch.backends.cuda.matmul.allow_tf32 = False
                model = mod.build_model("LiteBN_XS", task).to(device)
                model.load_state_dict(torch.load(candidate_path, map_location=device, weights_only=False), strict=True)
                model.eval()
                trained = {
                    "lambda_channel": model.lambda_channel.detach().clone(),
                    "lambda_scale": model.lambda_scale.detach().clone(),
                    "mixer_gamma": [block.gamma.detach().clone() for block in model.mixer],
                }
                cache = mod.RawGPUCache(bundle, device) if historical_default else None
                mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None]
                std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None]
                cell_rows: dict[str, list[dict[str, Any]]] = {state[0]: [] for state in STATES}
                with torch.no_grad():
                    for subject in mod.subject_sort(subjects, bundle.name):
                        indices = bundle.indices([subject], (int(mod.TASKS[task]["future_session"]),))
                        for start in range(0, len(indices), 128):
                            batch_indices = indices[start:start + 128]
                            if cache is not None:
                                value, _ = cache.batch(batch_indices, mean, std)
                            else:
                                raw = bundle.signal_batch(batch_indices)
                                value = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)).to(device, non_blocking=True)
                                value = (value - mean_t) / torch.clamp(std_t, min=1e-6)
                            for state_id, c, s, m in STATES:
                                set_state(model, trained, c, s, m)
                                logits = model(value)[0].float().detach().cpu().numpy()
                                predictions = logits.argmax(axis=1)
                                for pos, index in enumerate(batch_indices):
                                    row = bundle.rows[int(index)]
                                    cell_rows[state_id].append({
                                        "task": task, "seed": seed, "fold": fold, "subject_id": str(subject),
                                        "session": int(row.session), "trial_id": int(row.index),
                                        "true_label_replayed": int(row.label),
                                        "counterfactual_prediction": int(predictions[pos]),
                                        "counterfactual_logit": json.dumps(logits[pos].tolist(), separators=(",", ":")),
                                        "state_id": state_id, "C": c, "S": s, "M": m,
                                    })

                full = pd.DataFrame(cell_rows["C1S1M1"])
                checked = cell[KEY + ["true_label", "B0_prediction", "B0_correct", "candidate_prediction", "candidate_logits"]].merge(
                    full, on=KEY, validate="one_to_one")
                predictions_exact = np.array_equal(checked.candidate_prediction.to_numpy(int), checked.counterfactual_prediction.to_numpy(int))
                max_logit_difference = float(np.abs(parse_matrix(checked.candidate_logits) - parse_matrix(checked.counterfactual_logit)).max())
                key_exact = len(checked) == len(cell) == len(full)
                label_exact = np.array_equal(checked.true_label.to_numpy(int), checked.true_label_replayed.to_numpy(int))
                stored_ba = subject_equal_ba(checked.rename(columns={"candidate_prediction": "stored_prediction"}), "stored_prediction")
                replayed_ba = subject_equal_ba(checked, "counterfactual_prediction")
                metric_difference = abs(stored_ba - replayed_ba)
                status = "PASS" if predictions_exact and key_exact and label_exact and metric_difference <= 1e-8 else "PROTOCOL_FAIL_FOR_CELL"
                audits.append({"task": task, "seed": seed, "fold": fold, "rows": len(cell),
                               "predictions_exact": predictions_exact, "labels_exact": label_exact, "trial_keys_exact": key_exact,
                               "subject_set_exact": set(cell.subject_id.astype(str)) == set(subjects),
                               "stored_subject_equal_BA": stored_ba, "replayed_subject_equal_BA": replayed_ba,
                               "absolute_metric_difference": metric_difference, "max_abs_logit_difference": max_logit_difference,
                               "historical_default_numerics": historical_default, "status": status})
                if status == "PASS":
                    assembled = []
                    reference = cell[KEY + ["true_label", "B0_prediction", "B0_correct"]]
                    for state_id, c, s, m in STATES:
                        state = pd.DataFrame(cell_rows[state_id]).drop(columns=["counterfactual_logit", "true_label_replayed"])
                        state = reference.merge(state, on=KEY, validate="one_to_one")
                        state["counterfactual_correct"] = state.counterfactual_prediction.eq(state.true_label)
                        state["error_state"] = np.select(
                            [state.B0_correct & state.counterfactual_correct,
                             ~state.B0_correct & state.counterfactual_correct,
                             state.B0_correct & ~state.counterfactual_correct],
                            ["CC", "RESCUE", "HARM"], default="WW")
                        assembled.append(state)
                    cube = pd.concat(assembled, ignore_index=True)
                    # Lossless compact categorical encoding keeps this required
                    # CSV below GitHub's 100 MB single-blob limit when LFS is
                    # unavailable. Downstream scripts restore canonical names.
                    cube["task"] = cube.task.map(TASK_CODES)
                    if task == "WBCIC_MI":
                        cube["subject_id"] = cube.subject_id.astype(str).str.removeprefix("sub-")
                    cube["B0_correct"] = cube.B0_correct.astype(np.int8)
                    cube["counterfactual_correct"] = cube.counterfactual_correct.astype(np.int8)
                    cube.to_csv(part, index=False, mode="a", header=not wrote_header)
                    wrote_header = True
                    valid_cells += 1
                print(f"CUBE {task} seed={seed} fold={fold} {status} max_logit_diff={max_logit_difference:.3g}", flush=True)
                del model, cache, cell_rows, full, checked
                torch.cuda.empty_cache(); gc.collect()
            del bundle
            torch.cuda.empty_cache(); gc.collect()
    audit = pd.DataFrame(audits)
    audit.to_csv(outputs / "FULL_XS_REPLAY_AUDIT.csv", index=False)
    if valid_cells != 60 or not audit.status.eq("PASS").all():
        raise RuntimeError(f"counterfactual cube incomplete: valid cells {valid_cells}/60")
    os.replace(part, output)
    print(f"COUNTERFACTUAL_CUBE_COMPLETE cells={valid_cells} rows={sum(1 for _ in open(output, encoding='utf-8'))-1}", flush=True)


if __name__ == "__main__":
    main()
