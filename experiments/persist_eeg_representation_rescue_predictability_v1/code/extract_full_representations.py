#!/usr/bin/env python3
"""Replay frozen B0/X/XS networks and retain full pre-head embeddings."""
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

KEY = ["task", "fold", "subject_id", "session", "trial_id"]


def load_py(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_csv(path: Path, frame: pd.DataFrame, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False, **kwargs)
    os.replace(temp, path)


def parse_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack(series.map(json.loads))


def checkpoint_paths(replay: Any, comparison: str, task: str, fold: int, seed: int,
                     roots: dict[str, Path]) -> tuple[Path, Path, Path]:
    if comparison == "X":
        return (replay.x_checkpoint(roots["seed0"], task, fold, "LiteBN_BASELINE"),
                replay.x_checkpoint(roots["seed0"], task, fold, "LiteBN_X"),
                replay.normalizer_path(roots["seed0"], roots["xs"], roots["erp"], "X", task, fold, 0))
    return (replay.xs_baseline(roots["carrier"], roots["task"], task, fold, 0),
            replay.xs_checkpoint(roots["seed0"], roots["xs"], roots["erp"], task, fold, seed),
            replay.normalizer_path(roots["seed0"], roots["xs"], roots["erp"], "XS", task, fold, seed))


def replay(mod: Any, model: torch.nn.Module, bundle: Any, subjects: list[str], mean: np.ndarray,
           std: np.ndarray, device: torch.device, historical_default: bool,
           embedding_module: torch.nn.Module) -> pd.DataFrame:
    captured: dict[str, torch.Tensor] = {}
    handle = embedding_module.register_forward_hook(lambda _m, _i, output: captured.__setitem__("embedding", output))
    cache = mod.RawGPUCache(bundle, device) if historical_default else None
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None]
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None]
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for subject in mod.subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], (int(mod.TASKS[bundle.task]["future_session"]),))
            for start in range(0, len(indices), 128):
                batch_indices = indices[start:start + 128]
                if cache is not None:
                    value, _ = cache.batch(batch_indices, mean, std)
                else:
                    raw = bundle.signal_batch(batch_indices)
                    value = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)).to(device, non_blocking=True)
                    value = (value - mean_t) / torch.clamp(std_t, min=1e-6)
                logits = model(value)[0].float().detach().cpu().numpy()
                embedding = captured["embedding"].float().flatten(1).detach().cpu().numpy()
                for pos, index in enumerate(batch_indices):
                    source = bundle.rows[int(index)]
                    rows.append({
                        "task": bundle.task, "fold": -1, "subject_id": str(subject),
                        "session": int(source.session), "trial_id": int(source.index),
                        "prediction_replayed": int(logits[pos].argmax()),
                        "logits_replayed": json.dumps(logits[pos].tolist(), separators=(",", ":")),
                        "embedding": json.dumps(embedding[pos].tolist(), separators=(",", ":")),
                        "embedding_dim": int(embedding.shape[1]),
                    })
    handle.remove()
    del cache
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    repo, runtime = args.repo.resolve(), args.runtime.resolve()
    source_exp = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1"
    prior = repo / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1"
    outputs = repo / "experiments/persist_eeg_representation_rescue_predictability_v1/outputs"
    roots = {
        "seed0": Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime"),
        "xs": Path("/root/rivermind-data/xs_full_multiseed_finaltest_runtime"),
        "erp": Path("/root/rivermind-data/xs_erp_seed12_stability_runtime_correct_xs"),
        "carrier": Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime"),
        "task": Path("/root/rivermind-data/openbmi_task_generality_runtime"),
    }
    os.environ.update({"LITEBN_X_REPO": str(repo), "PERSIST_CACHE_ROOT": "/root/rivermind-data/persist_eeg_cache",
                       "LITEBN_X_RUNTIME": str(roots["seed0"])})
    helper = load_py("representation_replay_helpers", prior / "code/replay_models.py")
    mod = helper.load_model_module(repo, Path("/root/rivermind-data/persist_eeg_cache"), roots["seed0"])
    _, folds, _ = mod.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for historical replay")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    all_rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    expected_dims = {"B0": 64, "candidate": 128}
    for comparison, source_path, seeds in (
        ("X", runtime / "X_ROWS.csv.gz", (0,)),
        ("XS", runtime / "FIXED_B0_XS_ROWS.csv.gz", (0, 1, 2)),
    ):
        source_rows = pd.read_csv(source_path, dtype={"subject_id": str})
        for task in mod.TASK_ORDER:
            for fold in range(5):
                fold_spec = folds[mod.TASKS[task]["dataset"]][fold]
                subjects = [str(value) for value in fold_spec["outer_dev_subjects"]]
                bundle = mod.build_bundle(task, subjects)
                b0_path, _, norm_path = checkpoint_paths(helper, comparison, task, fold, 0, roots)
                mean, std, norm_meta = mod.load_tensor_pair(norm_path)
                b0_default = bool(task == "OpenBMI_ERP" and comparison == "X" and fold == 1)
                torch.backends.cudnn.allow_tf32 = b0_default
                torch.backends.cuda.matmul.allow_tf32 = False
                b0 = mod.build_model("LiteBN_BASELINE", task).to(device)
                b0.load_state_dict(torch.load(b0_path, map_location=device, weights_only=False), strict=True)
                b0_replay = replay(mod, b0, bundle, subjects, mean, std, device, b0_default, b0.embedding)
                b0_replay["fold"] = fold
                ref = source_rows[(source_rows.task == task) & (source_rows.fold == fold) & (source_rows.seed == 0)]
                checked = ref[KEY + ["B0_prediction", "B0_logits"]].merge(b0_replay, on=KEY, validate="one_to_one")
                stored, observed = parse_matrix(checked.B0_logits), parse_matrix(checked.logits_replayed)
                pred_ok = np.array_equal(checked.B0_prediction.to_numpy(int), checked.prediction_replayed.to_numpy(int))
                max_diff = float(np.abs(stored - observed).max())
                dims = sorted(b0_replay.embedding_dim.unique().tolist())
                if not pred_ok or dims != [expected_dims["B0"]]:
                    raise RuntimeError(f"B0 replay invariant failed {comparison}/{task}/fold{fold}: pred={pred_ok}, dims={dims}")
                audits.append({"comparison": comparison, "task": task, "seed": 0, "fold": fold, "method": "B0",
                               "rows": len(checked), "predictions_exact": pred_ok, "max_abs_logit_difference": max_diff,
                               "embedding_dim": dims[0], "expected_embedding_dim": expected_dims["B0"],
                               "dimension_pass": True, "historical_default_numerics": b0_default,
                               "status": "PASS"})
                b0_embedding = b0_replay[KEY + ["embedding"]].rename(columns={"embedding": "B0_embedding"})
                del b0, b0_replay, checked
                torch.cuda.empty_cache()

                for seed in seeds:
                    candidate_default = bool(task == "OpenBMI_ERP" and
                                             ((comparison == "X" and fold == 1) or
                                              (comparison == "XS" and (seed, fold) in {(1, 0), (1, 3), (2, 0), (2, 3)})))
                    torch.backends.cudnn.allow_tf32 = candidate_default
                    cell = source_rows[(source_rows.task == task) & (source_rows.fold == fold) & (source_rows.seed == seed)].copy()
                    _, candidate_path, candidate_norm_path = checkpoint_paths(helper, comparison, task, fold, seed, roots)
                    _, _, candidate_norm = mod.load_tensor_pair(candidate_norm_path)
                    if candidate_norm["mean_std_sha256"] != norm_meta["mean_std_sha256"]:
                        raise RuntimeError(f"normalizer mismatch {comparison}/{task}/seed{seed}/fold{fold}")
                    method = "LiteBN_X" if comparison == "X" else "LiteBN_XS"
                    candidate = mod.build_model(method, task).to(device)
                    candidate.load_state_dict(torch.load(candidate_path, map_location=device, weights_only=False), strict=True)
                    candidate_replay = replay(mod, candidate, bundle, subjects, mean, std, device, candidate_default, candidate.embedding)
                    candidate_replay["fold"] = fold
                    checked = cell[KEY + ["candidate_prediction", "candidate_logits"]].merge(candidate_replay, on=KEY, validate="one_to_one")
                    stored, observed = parse_matrix(checked.candidate_logits), parse_matrix(checked.logits_replayed)
                    pred_ok = np.array_equal(checked.candidate_prediction.to_numpy(int), checked.prediction_replayed.to_numpy(int))
                    max_diff = float(np.abs(stored - observed).max())
                    dims = sorted(candidate_replay.embedding_dim.unique().tolist())
                    if not pred_ok or dims != [expected_dims["candidate"]]:
                        raise RuntimeError(f"candidate replay invariant failed {comparison}/{task}/seed{seed}/fold{fold}: pred={pred_ok}, dims={dims}")
                    audits.append({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "method": method,
                                   "rows": len(checked), "predictions_exact": pred_ok, "max_abs_logit_difference": max_diff,
                                   "embedding_dim": dims[0], "expected_embedding_dim": expected_dims["candidate"],
                                   "dimension_pass": True, "historical_default_numerics": candidate_default,
                                   "status": "PASS"})
                    disagreement = cell[cell.B0_prediction != cell.candidate_prediction][KEY + ["seed"]]
                    candidate_embedding = candidate_replay[KEY + ["embedding"]].rename(columns={"embedding": "candidate_embedding"})
                    disagreement = disagreement.merge(b0_embedding, on=KEY, validate="many_to_one")
                    disagreement = disagreement.merge(candidate_embedding, on=KEY, validate="one_to_one")
                    disagreement["comparison"] = comparison
                    all_rows.append(disagreement)
                    del candidate, candidate_replay, candidate_embedding, checked, disagreement
                    torch.cuda.empty_cache(); gc.collect()
                    print(f"REPRESENTATIONS {comparison} {task} seed={seed} fold={fold} PASS max_logit_diff={max_diff:.3g}", flush=True)
                del bundle, b0_embedding
                torch.cuda.empty_cache(); gc.collect()

    result = pd.concat(all_rows, ignore_index=True)
    expected = pd.read_csv(source_exp / "outputs/PREDICTABILITY_FEATURES.csv", usecols=["comparison", "task", "seed", "fold", "subject_id", "session", "trial_id"], dtype={"subject_id": str})
    key = ["comparison", "task", "seed", "fold", "subject_id", "session", "trial_id"]
    if len(result) != len(expected) or len(result.merge(expected, on=key, how="outer", indicator=True).query("_merge != 'both'")):
        raise RuntimeError("representation disagreement key set does not match authoritative prior feature rows")
    atomic_csv(runtime / "FULL_REPRESENTATIONS.csv.gz", result, compression="gzip")
    atomic_csv(outputs / "REPRESENTATION_REPLAY_AUDIT.csv", pd.DataFrame(audits))
    print(f"REPRESENTATION_REPLAY_COMPLETE rows={len(result)} audit_cells={len(audits)}", flush=True)


if __name__ == "__main__":
    main()
