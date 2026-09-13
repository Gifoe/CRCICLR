#!/usr/bin/env python3
"""Extract predeclared label-free confidence and internal features.

Frozen EEG checkpoints are evaluated in model.eval() with hooks only; weights
and forward functions are not modified. Rows are emitted only for B0/candidate
disagreements, although full subject batches are replayed to preserve semantics.
"""
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
CONF_FEATURES = [
    "b0_top1_probability", "b0_margin", "b0_entropy", "b0_logit_l2_norm",
    "candidate_top1_probability", "candidate_margin", "candidate_entropy", "candidate_logit_l2_norm",
    "top1_probability_difference", "margin_difference", "entropy_difference",
    "probability_L1_distance", "JS_divergence",
]
BASE_INTERNAL = ["b0_embedding_l2_norm", "b0_final_temporal_feature_l2_norm"]
CANDIDATE_INTERNAL = [
    "candidate_embedding_l2_norm", "candidate_prehead_feature_l2_norm",
    "mixer_block1_relative_update_norm", "mixer_block2_relative_update_norm", "mixer_block3_relative_update_norm",
    "scale_alpha_max", "scale_alpha_min", "scale_alpha_entropy", "scale_alpha_range",
]
XS_INTERNAL = ["channel_mod_mean_abs_delta", "channel_mod_std_delta", "channel_mod_max_abs_delta"]


def load_py(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_name(path.name + ".part")
    frame.to_csv(temp, index=False); os.replace(temp, path)


def checkpoint_paths(replay: Any, comparison: str, task: str, fold: int, seed: int,
                     roots: dict[str, Path]) -> tuple[Path, Path, Path]:
    if comparison == "X":
        return (replay.x_checkpoint(roots["seed0"], task, fold, "LiteBN_BASELINE"),
                replay.x_checkpoint(roots["seed0"], task, fold, "LiteBN_X"),
                replay.normalizer_path(roots["seed0"], roots["xs"], roots["erp"], "X", task, fold, 0))
    return (replay.xs_baseline(roots["carrier"], roots["task"], task, fold, 0),
            replay.xs_checkpoint(roots["seed0"], roots["xs"], roots["erp"], task, fold, seed),
            replay.normalizer_path(roots["seed0"], roots["xs"], roots["erp"], "XS", task, fold, seed))


def norm_rows(value: torch.Tensor) -> np.ndarray:
    return torch.linalg.vector_norm(value.float().flatten(1), dim=1).detach().cpu().numpy()


def install_hooks(model: torch.nn.Module, candidate: bool) -> tuple[dict[str, torch.Tensor], list[Any]]:
    cap: dict[str, torch.Tensor] = {}; handles = []
    if not candidate:
        handles.append(model.embedding.register_forward_hook(lambda _m, _i, o: cap.__setitem__("b0_embedding", o)))
        handles.append(model.pool.register_forward_pre_hook(lambda _m, i: cap.__setitem__("b0_temporal", i[0])))
        return cap, handles
    for index, block in enumerate(model.mixer, 1):
        def hook(_m: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor, idx: int = index) -> None:
            cap[f"mixer{idx}_input"] = inputs[0]; cap[f"mixer{idx}_output"] = output
        handles.append(block.register_forward_hook(hook))
    handles.append(model.scale_mlp.register_forward_hook(lambda _m, _i, o: cap.__setitem__("scale_logits", o)))
    if getattr(model, "channel_gate", False):
        handles.append(model.channel_mlp.register_forward_hook(lambda _m, _i, o: cap.__setitem__("channel_raw", o)))
    handles.append(model.embedding.register_forward_hook(lambda _m, _i, o: cap.__setitem__("embedding", o)))
    handles.append(model.head.register_forward_pre_hook(lambda _m, i: cap.__setitem__("prehead", i[0])))
    return cap, handles


def captured_features(model: torch.nn.Module, cap: dict[str, torch.Tensor], candidate: bool) -> dict[str, np.ndarray]:
    if not candidate:
        return {"b0_embedding_l2_norm": norm_rows(cap["b0_embedding"]),
                "b0_final_temporal_feature_l2_norm": norm_rows(cap["b0_temporal"])}
    result = {
        "candidate_embedding_l2_norm": norm_rows(cap["embedding"]),
        "candidate_prehead_feature_l2_norm": norm_rows(cap["prehead"]),
    }
    for index in (1, 2, 3):
        before, after = cap[f"mixer{index}_input"], cap[f"mixer{index}_output"]
        result[f"mixer_block{index}_relative_update_norm"] = norm_rows(after - before) / (norm_rows(before) + 1e-12)
    alpha = torch.softmax(cap["scale_logits"].float(), dim=1)
    alpha_np = alpha.detach().cpu().numpy()
    result.update({
        "scale_alpha_max": alpha_np.max(axis=1), "scale_alpha_min": alpha_np.min(axis=1),
        "scale_alpha_entropy": -(alpha_np * np.log(np.clip(alpha_np, 1e-12, 1))).sum(axis=1),
        "scale_alpha_range": alpha_np.max(axis=1) - alpha_np.min(axis=1),
    })
    if getattr(model, "channel_gate", False):
        delta = (torch.tanh(model.lambda_channel) * torch.tanh(cap["channel_raw"].squeeze(-1))).float().detach().cpu().numpy()
        result.update({"channel_mod_mean_abs_delta": np.abs(delta).mean(axis=1),
                       "channel_mod_std_delta": delta.std(axis=1),
                       "channel_mod_max_abs_delta": np.abs(delta).max(axis=1)})
    return result


def replay_internal(mod: Any, model: torch.nn.Module, bundle: Any, subjects: list[str], mean: np.ndarray, std: np.ndarray,
                    device: torch.device, candidate: bool, historical_default: bool) -> pd.DataFrame:
    cap, handles = install_hooks(model, candidate)
    cache = mod.RawGPUCache(bundle, device) if historical_default else None
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None]
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None]
    rows: list[dict[str, Any]] = []; model.eval()
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
                features = captured_features(model, cap, candidate)
                for pos, index in enumerate(batch_indices):
                    source = bundle.rows[int(index)]
                    row: dict[str, Any] = {"task": bundle.task, "fold": -1, "subject_id": str(subject),
                                           "session": int(source.session), "trial_id": int(source.index),
                                           "prediction_replayed": int(logits[pos].argmax()),
                                           "logits_replayed": json.dumps(logits[pos].tolist(), separators=(",", ":"))}
                    row.update({key: float(value[pos]) for key, value in features.items()})
                    rows.append(row)
    for handle in handles: handle.remove()
    del cache
    return pd.DataFrame(rows)


def parse_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack(series.map(json.loads))


def confidence_features(frame: pd.DataFrame) -> pd.DataFrame:
    bp, cp = parse_matrix(frame.B0_probabilities), parse_matrix(frame.candidate_probabilities)
    bl, cl = parse_matrix(frame.B0_logits), parse_matrix(frame.candidate_logits)
    frame = frame.copy()
    frame["b0_top1_probability"] = bp.max(axis=1)
    frame["b0_margin"] = np.sort(bp, axis=1)[:, -1] - np.sort(bp, axis=1)[:, -2]
    frame["b0_entropy"] = -(bp * np.log(np.clip(bp, 1e-12, 1))).sum(axis=1)
    frame["b0_logit_l2_norm"] = np.linalg.norm(bl, axis=1)
    frame["candidate_top1_probability"] = cp.max(axis=1)
    frame["candidate_margin"] = np.sort(cp, axis=1)[:, -1] - np.sort(cp, axis=1)[:, -2]
    frame["candidate_entropy"] = -(cp * np.log(np.clip(cp, 1e-12, 1))).sum(axis=1)
    frame["candidate_logit_l2_norm"] = np.linalg.norm(cl, axis=1)
    frame["top1_probability_difference"] = frame.candidate_top1_probability - frame.b0_top1_probability
    frame["margin_difference"] = frame.candidate_margin - frame.b0_margin
    frame["entropy_difference"] = frame.candidate_entropy - frame.b0_entropy
    frame["probability_L1_distance"] = np.abs(bp - cp).sum(axis=1)
    midpoint = .5 * (bp + cp)
    frame["JS_divergence"] = .5 * ((bp * np.log(np.clip(bp / midpoint, 1e-12, None))).sum(axis=1) +
                                    (cp * np.log(np.clip(cp / midpoint, 1e-12, None))).sum(axis=1))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", type=Path, required=True); parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(); repo, runtime = args.repo.resolve(), args.runtime.resolve()
    outputs = repo / "experiments/persist_eeg_stable_rescue_predictability_audit_v1/outputs"
    prior = repo / "experiments/persist_eeg_b0_x_xs_complementarity_audit_v1"
    roots = {"seed0": Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime"),
             "xs": Path("/root/rivermind-data/xs_full_multiseed_finaltest_runtime"),
             "erp": Path("/root/rivermind-data/xs_erp_seed12_stability_runtime_correct_xs"),
             "carrier": Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime"),
             "task": Path("/root/rivermind-data/openbmi_task_generality_runtime")}
    os.environ.update({"LITEBN_X_REPO": str(repo), "PERSIST_CACHE_ROOT": "/root/rivermind-data/persist_eeg_cache", "LITEBN_X_RUNTIME": str(roots["seed0"])})
    replay = load_py("feature_replay_helpers", prior / "code/replay_models.py")
    mod = replay.load_model_module(repo, Path("/root/rivermind-data/persist_eeg_cache"), roots["seed0"])
    _, folds, _ = mod.load_folds(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA required for historical replay")
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    all_features: list[pd.DataFrame] = []; audit: list[dict[str, Any]] = []
    for comparison, input_path, seeds in (("X", runtime / "X_ROWS.csv.gz", (0,)), ("XS", runtime / "FIXED_B0_XS_ROWS.csv.gz", (0, 1, 2))):
        raw_rows = pd.read_csv(input_path, dtype={"subject_id": str})
        for task in mod.TASK_ORDER:
            for fold in range(5):
                # The preceding audit's passing rows used full FP32. Stage 0
                # changes only the five repaired cells to the original default.
                b0_historical_default = bool(task == "OpenBMI_ERP" and comparison == "X" and fold == 1)
                torch.backends.cudnn.allow_tf32 = b0_historical_default
                torch.backends.cuda.matmul.allow_tf32 = False
                fold_spec = folds[mod.TASKS[task]["dataset"]][fold]
                subjects = [str(v) for v in fold_spec["outer_dev_subjects"]]
                bundle = mod.build_bundle(task, subjects)
                b0_path, _, norm_path = checkpoint_paths(replay, comparison, task, fold, 0, roots)
                mean, std, norm_meta = mod.load_tensor_pair(norm_path)
                b0 = mod.build_model("LiteBN_BASELINE", task).to(device)
                b0.load_state_dict(torch.load(b0_path, map_location=device, weights_only=False), strict=True)
                b0_internal = replay_internal(mod, b0, bundle, subjects, mean, std, device, False, b0_historical_default); b0_internal["fold"] = fold
                base_reference = raw_rows[(raw_rows.task == task) & (raw_rows.fold == fold) & (raw_rows.seed == 0)]
                check = base_reference[KEY + ["B0_prediction", "B0_logits"]].merge(b0_internal, on=KEY, validate="one_to_one")
                stored = parse_matrix(check.B0_logits); replayed = parse_matrix(check.logits_replayed)
                if not np.array_equal(check.B0_prediction.to_numpy(int), check.prediction_replayed.to_numpy(int)):
                    raise RuntimeError(f"B0 prediction replay mismatch {comparison}/{task}/fold{fold}")
                audit.append({"comparison": comparison, "task": task, "seed": 0, "fold": fold, "method": "B0",
                              "max_abs_logit_difference": float(np.abs(stored - replayed).max()), "predictions_exact": True,
                              "historical_default_numerics": b0_historical_default})
                b0_internal = b0_internal.drop(columns=["prediction_replayed", "logits_replayed"])
                del b0; torch.cuda.empty_cache()
                for seed in seeds:
                    candidate_historical_default = bool(task == "OpenBMI_ERP" and
                                                        ((comparison == "X" and fold == 1) or
                                                         (comparison == "XS" and (seed, fold) in {(1, 0), (1, 3), (2, 0), (2, 3)})))
                    torch.backends.cudnn.allow_tf32 = candidate_historical_default
                    cell = raw_rows[(raw_rows.task == task) & (raw_rows.fold == fold) & (raw_rows.seed == seed)].copy()
                    _, candidate_path, candidate_norm_path = checkpoint_paths(replay, comparison, task, fold, seed, roots)
                    _, _, candidate_norm = mod.load_tensor_pair(candidate_norm_path)
                    if candidate_norm["mean_std_sha256"] != norm_meta["mean_std_sha256"]:
                        raise RuntimeError(f"normalizer mismatch {comparison}/{task}/seed{seed}/fold{fold}")
                    method = "LiteBN_X" if comparison == "X" else "LiteBN_XS"
                    candidate = mod.build_model(method, task).to(device)
                    candidate.load_state_dict(torch.load(candidate_path, map_location=device, weights_only=False), strict=True)
                    internal = replay_internal(mod, candidate, bundle, subjects, mean, std, device, True, candidate_historical_default); internal["fold"] = fold
                    check = cell[KEY + ["candidate_prediction", "candidate_logits"]].merge(internal, on=KEY, validate="one_to_one")
                    stored = parse_matrix(check.candidate_logits); replayed = parse_matrix(check.logits_replayed)
                    if not np.array_equal(check.candidate_prediction.to_numpy(int), check.prediction_replayed.to_numpy(int)):
                        raise RuntimeError(f"candidate prediction replay mismatch {comparison}/{task}/seed{seed}/fold{fold}")
                    audit.append({"comparison": comparison, "task": task, "seed": seed, "fold": fold, "method": method,
                                  "max_abs_logit_difference": float(np.abs(stored - replayed).max()), "predictions_exact": True,
                                  "historical_default_numerics": candidate_historical_default})
                    internal = internal.drop(columns=["prediction_replayed", "logits_replayed"])
                    disagreement = cell[cell.B0_prediction != cell.candidate_prediction].copy()
                    disagreement = disagreement.merge(b0_internal, on=KEY, validate="many_to_one").merge(internal, on=KEY, validate="one_to_one")
                    disagreement["comparison"] = comparison
                    disagreement["outcome"] = np.select(
                        [disagreement.B0_correct.eq(False) & disagreement.candidate_correct.eq(True),
                         disagreement.B0_correct.eq(True) & disagreement.candidate_correct.eq(False)],
                        ["RESCUE", "HARM"], default="BOTH_WRONG_DIFFERENT")
                    disagreement = confidence_features(disagreement)
                    if comparison == "X":
                        for column in XS_INTERNAL: disagreement[column] = np.nan
                    all_features.append(disagreement)
                    del candidate, internal, check, disagreement; torch.cuda.empty_cache(); gc.collect()
                    print(f"FEATURES {comparison} {task} seed={seed} fold={fold} PASS", flush=True)
                del bundle, b0_internal; torch.cuda.empty_cache(); gc.collect()
    result = pd.concat(all_features, ignore_index=True).sort_values(["comparison", "task", "seed", "fold", "subject_id", "trial_id"])
    required = CONF_FEATURES + BASE_INTERNAL + CANDIDATE_INTERNAL
    if result[required].isna().any().any() or result[result.comparison == "XS"][XS_INTERNAL].isna().any().any():
        raise RuntimeError("required mechanistic features contain missing values")
    if not np.allclose(result.candidate_embedding_l2_norm, result.candidate_prehead_feature_l2_norm, rtol=0, atol=1e-7):
        raise RuntimeError("candidate embedding/prehead identity invariant failed")
    atomic_csv(outputs / "PREDICTABILITY_FEATURES.csv", result)
    atomic_csv(runtime / "FEATURE_REPLAY_AUDIT.csv", pd.DataFrame(audit))
    print(f"MECHANISTIC_FEATURES_COMPLETE rows={len(result)}", flush=True)


if __name__ == "__main__":
    main()
