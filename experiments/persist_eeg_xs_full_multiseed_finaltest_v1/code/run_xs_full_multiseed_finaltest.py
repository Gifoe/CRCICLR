#!/usr/bin/env python3
"""Frozen LiteBN-XS seed1/2 completion and one-shot final heldout evaluation.

The program is deliberately split into three separately-invoked modes:
development, final-preflight, and final-evaluate.  The last mode refuses to run
unless the full development grid and the immutable preflight lock already exist.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


REPO = Path(os.environ.get("PERSIST_EEG_REPO", "/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_xs_full_multiseed_finaltest_v1"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path("/root/rivermind-data/xs_full_multiseed_finaltest_runtime").resolve()
ERP_RUNTIME = Path("/root/rivermind-data/xs_erp_seed12_stability_runtime_correct_xs").resolve()
SEED0_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime").resolve()
TASK_RUNTIME = Path("/root/rivermind-data/openbmi_task_generality_runtime").resolve()
CARRIER_RUNTIME = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime").resolve()
BASE_EXP = REPO / "experiments" / "persist_eeg_litebn_x_singlemodel_seed0_v1"
BASE_CODE = BASE_EXP / "code" / "litebn_x.py"
FIVEFOLD = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"
V8_SPLIT = REPO / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"

TRAIN_TASKS = ("OpenBMI_MI", "OpenBMI_SSVEP", "WBCIC_MI")
ALL_TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
SEEDS = (1, 2)
MAX_EPOCHS, MIN_SELECTION_EPOCH, PATIENCE = 60, 10, 10
LR, WEIGHT_DECAY, CLIP, TOL = 3e-4, 5e-4, 5.0, 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("frozen_litebn_xs", BASE_CODE)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load LiteBN-XS source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.EPOCHS = MAX_EPOCHS
    module.MIN_EPOCH = MIN_SELECTION_EPOCH
    return module


def baseline_path(task: str, fold: int, seed: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        short = "erp" if task.endswith("ERP") else "ssvep"
        return TASK_RUNTIME / f"{short}_fold{fold}_seed{seed}_litebn" / "selected_best.pt"
    short = "openbmi" if task.startswith("OpenBMI") else "wbcic"
    return CARRIER_RUNTIME / f"{short}_fold{fold}_seed{seed}_litebn" / "selected_best.pt"


def xs_checkpoint_path(task: str, fold: int, seed: int) -> Path:
    root = SEED0_RUNTIME if seed == 0 else (ERP_RUNTIME / f"seed{seed}" if task == "OpenBMI_ERP" else RUNTIME / f"seed{seed}")
    return root / "checkpoints" / task.lower() / f"fold{fold}_litebn_xs" / "selected_best.pt"


def normalizer_path(task: str, fold: int, seed: int) -> Path:
    root = SEED0_RUNTIME if seed == 0 else (ERP_RUNTIME / f"seed{seed}" if task == "OpenBMI_ERP" else RUNTIME / f"seed{seed}")
    return root / "normalizers" / f"{task.lower()}_fold{fold}.npz"


def protocol_lock(mod: Any) -> dict[str, Any]:
    search, folds, split_sha = mod.load_folds()
    data = {
        "experiment": "persist_eeg_xs_full_multiseed_finaltest_v1",
        "model": "exact original LiteBN_XS",
        "training_tasks": list(TRAIN_TASKS), "all_final_tasks": list(ALL_TASKS),
        "new_training_seeds": list(SEEDS), "seed0": "historical frozen XS checkpoints/results reused",
        "fivefold_split_sha256": split_sha, "canonical_inner_split_only": True,
        "new_inner_splits_created": False, "optimizer": "AdamW", "lr": LR,
        "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS,
        "min_selection_epoch": MIN_SELECTION_EPOCH, "patience": PATIENCE,
        "selection_metric": "canonical inner-validation subject-mean balanced accuracy",
        "baseline": "exact verified historical LiteBN_BASELINE checkpoints reused",
        "source_code_sha256": sha256(BASE_CODE), "runner_sha256": sha256(Path(__file__)),
        "final_heldout_accessed": False, "final_test_used_for_tuning": False,
    }
    write_json(PROTOCOL / "MULTISEED_PROTOCOL_LOCK.json", data)
    return data


def train_one_patience(mod: Any, model: torch.nn.Module, task: str, fold: dict[str, Any], bundle: Any,
                       cache: Any, mean: np.ndarray, std: np.ndarray, norm_meta: dict[str, Any],
                       batch_info: dict[str, Any], class_weight: torch.Tensor | None, class_meta: dict[str, Any],
                       device: torch.device) -> dict[str, Any]:
    architecture = "LiteBN_XS"
    latest = mod.checkpoint_path(task, int(fold["fold_id"]), architecture, "checkpoint_latest.pt")
    selected = mod.checkpoint_path(task, int(fold["fold_id"]), architecture, "selected_best.pt")
    invariants = {
        "task": task, "fold": int(fold["fold_id"]), "architecture": architecture, "seed": int(mod.SEED),
        "initial_sha256": mod.state_hash(model), "runner_sha256": sha256(Path(__file__)),
        "normalizer_sha256": norm_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info.get("manifest_sha256"),
        "class_weight_info": class_meta, "max_epochs": MAX_EPOCHS,
        "min_selection_epoch": MIN_SELECTION_EPOCH, "patience": PATIENCE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "clip": CLIP,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_f1, best_epoch, best_state, bad = 1, [], -float("inf"), -float("inf"), None, None, 0
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants: raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"]); mod.restore_rng(saved["rng"])
        start, history, best, best_f1 = int(saved["epoch"]) + 1, list(saved["history"]), float(saved["best"]), float(saved["best_f1"])
        best_epoch, best_state, bad = saved["best_epoch"], saved["best_state"], int(saved["bad"])
    train_indices = bundle.indices(fold["inner_train_subjects"], mod.TASKS[task]["source_sessions"])
    weight = None if class_weight is None else class_weight.to(device)
    started = time.perf_counter()
    for epoch in range(start, MAX_EPOCHS + 1):
        model.train(); losses: list[float] = []
        batches = batch_info["episodes"][epoch - 1] if mod.TASKS[task]["mi_protocol"] else mod.task_epoch_batches(train_indices, task, int(fold["fold_id"]), epoch)
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(value); loss = F.cross_entropy(logits, labels, weight=weight)
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss: {task}/fold{fold['fold_id']}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = mod.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        val_ba = float(np.mean([r["BA"] for r in validation.values()]))
        val_f1 = float(np.mean([r["macro_F1"] for r in validation.values()]))
        eligible = epoch >= MIN_SELECTION_EPOCH
        improved = bool(eligible and (val_ba > best + TOL or (abs(val_ba - best) <= TOL and val_f1 > best_f1 + TOL)))
        if improved:
            best, best_f1, best_epoch, bad = val_ba, val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif eligible:
            bad += 1
        stopped = bool(eligible and bad >= PATIENCE)
        history.append({"epoch": epoch, "cross_entropy": float(np.mean(losses)), "inner_val_subject_BA": val_ba,
                        "inner_val_subject_macro_F1": val_f1, "selected": improved, "eligible": eligible,
                        "bad_epochs": bad, "stopped_early": stopped, "batches": len(losses)})
        mod.atomic_torch_save(latest, {"epoch": epoch, "history": history, "best": best, "best_f1": best_f1,
            "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": mod.rng_state(), "bad": bad,
            "invariants": invariants})
        if epoch == 1 or epoch % 5 == 0 or improved or stopped:
            print(f"[XS seed{mod.SEED} {task} f{fold['fold_id']}] e={epoch:02d} valBA={val_ba:.5f} best={best:.5f} bad={bad}/{PATIENCE}", flush=True)
        if stopped: break
    if best_state is None: raise RuntimeError("no eligible checkpoint selected")
    model.load_state_dict(best_state, strict=True); mod.atomic_torch_save(selected, model.state_dict())
    return {"task": task, "dataset": mod.TASKS[task]["dataset"], "fold": int(fold["fold_id"]), "architecture": architecture,
        "seed": int(mod.SEED), "parameter_count": mod.parameter_count(model), "selected_epoch": int(best_epoch),
        "best_inner_val_BA": best, "best_inner_val_macro_F1": best_f1, "checkpoint_path": str(selected),
        "checkpoint_sha256": sha256(selected), "normalizer_sha256": norm_meta["mean_std_sha256"],
        "batch_manifest_sha256": batch_info.get("manifest_sha256"), "class_weighted_ce": bool(class_meta["weighted_cross_entropy"]),
        "epochs_completed": len(history), "early_stopped": bool(history[-1]["stopped_early"]),
        "elapsed_seconds_this_invocation": time.perf_counter() - started, "history": history, "runner_sha256": sha256(Path(__file__))}


def train_seed(mod: Any, seed: int) -> list[dict[str, Any]]:
    runtime = RUNTIME / f"seed{seed}"; runtime.mkdir(parents=True, exist_ok=True); mod.RUNTIME = runtime; mod.SEED = seed
    existing = OUT / f"SEED{seed}_XS_INNERVAL.csv"
    if existing.is_file():
        frame = pd.read_csv(existing)
        if len(frame) == 15 and set(frame.task) == set(TRAIN_TASKS) and set(frame.architecture) == {"LiteBN_XS"} and all(Path(x).is_file() for x in frame.checkpoint_path):
            print(f"RESUME_SEED{seed}_XS", flush=True); return frame.to_dict("records")
    _, folds, _ = mod.load_folds(); rows: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for task in TRAIN_TASKS:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            bundle = mod.build_bundle(task, allowed); mean, std, norm_meta = mod.normalizer(bundle, fold["inner_train_subjects"])
            mod.save_tensor_pair(runtime / "normalizers" / f"{task.lower()}_fold{fold['fold_id']}.npz", mean, std, norm_meta)
            cache = mod.RawGPUCache(bundle, device)
            if mod.TASKS[task]["mi_protocol"]:
                episodes, manifest = mod.mi_manifest(bundle, fold, task); batch_info = {**manifest, "episodes": episodes}
            else:
                batch_info = {"kind": "historical_task_full_permutation_batch64", "manifest_sha256": None, "steps_per_epoch": None}
            weights, class_meta = mod.class_weights(bundle, fold["inner_train_subjects"])
            mod.set_seed(seed); model = mod.build_model("LiteBN_XS", task).to(device); mod.set_seed(seed + 100_000)
            record = train_one_patience(mod, model, task, fold, bundle, cache, mean, std, norm_meta, batch_info, weights, class_meta, device)
            rows.append({k: v for k, v in record.items() if k != "history"})
            print(f"XS_SEED{seed}_DONE {task} fold={fold['fold_id']} selected_epoch={record['selected_epoch']}", flush=True)
            del model, cache, bundle
            if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["task", "fold"])
    if len(frame) != 15: raise RuntimeError(f"incomplete seed{seed} XS grid: {len(frame)}")
    write_csv(existing, frame); write_json(runtime / "TRAINING_LOGS.json", rows)
    return frame.to_dict("records")


def evaluate_outer(mod: Any, seed: int, records: list[dict[str, Any]]) -> pd.DataFrame:
    runtime = RUNTIME / f"seed{seed}"; mod.RUNTIME = runtime; mod.SEED = seed
    _, folds, _ = mod.load_folds(); by = {(r["task"], int(r["fold"])): r for r in records}; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for task in TRAIN_TASKS:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fid, outer = int(fold["fold_id"]), fold["outer_dev_subjects"]
            bundle = mod.build_bundle(task, outer); mean, std, normalizer = mod.load_tensor_pair(normalizer_path(task, fid, seed)); cache = mod.RawGPUCache(bundle, device)
            paths = {"LiteBN_BASELINE": baseline_path(task, fid, seed), "LiteBN_XS": Path(by[(task, fid)]["checkpoint_path"])}
            for method, path in paths.items():
                if not path.is_file(): raise FileNotFoundError(path)
                model = mod.build_model(method, task).to(device); model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
                for subject, metrics in mod.evaluate(model, bundle, cache, outer, mean, std).items():
                    rows.append({"task": task, "dataset": mod.TASKS[task]["dataset"], "fold": fid, "seed": seed, "subject_id": str(subject), "method": method, **metrics,
                        "checkpoint_sha256": sha256(path), "normalizer_sha256": normalizer["mean_std_sha256"], "baseline_reused": method == "LiteBN_BASELINE"})
                del model
            del cache, bundle
            if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["task", "fold", "subject_id", "method"]).reset_index(drop=True)
    expected = 2 * sum(len(f["outer_dev_subjects"]) for task in TRAIN_TASKS for f in folds[mod.TASKS[task]["dataset"]])
    if len(frame) != expected: raise RuntimeError(f"outer evaluation cardinality {len(frame)} != {expected}")
    write_csv(OUT / f"SEED{seed}_OUTER_DEVELOPMENT_RESULTS.csv", frame)
    return frame


def seed0_frame(task: str) -> pd.DataFrame:
    sources = {
        "OpenBMI_MI": BASE_EXP / "outputs" / "POSTHOC_RG_XS_SEED0_MI_OUTER_SUBJECT_RESULTS.csv",
        "WBCIC_MI": BASE_EXP / "outputs" / "POSTHOC_RG_XS_SEED0_MI_OUTER_SUBJECT_RESULTS.csv",
        "OpenBMI_SSVEP": BASE_EXP / "outputs" / "POSTHOC_XS_SEED0_SSVEP_OUTER_SUBJECT_RESULTS.csv",
        "OpenBMI_ERP": REPO / "experiments" / "persist_eeg_xs_erp_seed12_stability_v1" / "outputs_correct_xs" / "SEED0_EXISTING_ERP_OUTER_SUBJECT_RESULTS.csv",
    }
    frame = pd.read_csv(sources[task]); frame = frame[(frame.task == task) & frame.method.isin(("LiteBN_BASELINE", "LiteBN_XS"))].copy()
    frame["seed"] = 0; frame["baseline_reused"] = frame.method.eq("LiteBN_BASELINE")
    expected_subjects = 31 if task == "WBCIC_MI" else 40
    if len(frame) != expected_subjects * 2: raise RuntimeError(f"seed0 source invalid {task}: {len(frame)}")
    return frame


def existing_erp_frames() -> list[pd.DataFrame]:
    """ERP seed1/2 was completed on its dedicated, provenance-matched branch."""
    root = REPO / "experiments" / "persist_eeg_xs_erp_seed12_stability_v1" / "outputs_correct_xs"
    frames: list[pd.DataFrame] = []
    for seed in SEEDS:
        path = root / f"SEED{seed}_OUTER_SUBJECT_RESULTS.csv"
        frame = pd.read_csv(path)
        frame = frame[(frame.task == "OpenBMI_ERP") & frame.method.isin(("LiteBN_BASELINE", "LiteBN_XS"))].copy()
        frame["seed"] = seed
        if len(frame) != 80: raise RuntimeError(f"historical ERP seed{seed} source invalid: {len(frame)}")
        frames.append(frame)
    return frames


def summarize_development(frames: list[pd.DataFrame]) -> None:
    all_rows = pd.concat(frames, ignore_index=True).sort_values(["task", "seed", "fold", "subject_id", "method"]).reset_index(drop=True)
    expected_tasks = set(ALL_TASKS)
    if set(all_rows.task) != expected_tasks or set(all_rows.seed) != {0, 1, 2}: raise RuntimeError("development grid not complete")
    write_csv(OUT / "DEVELOPMENT_MULTISEED_SUBJECT_RESULTS.csv", all_rows)
    summary_rows: list[dict[str, Any]] = []; fold_rows: list[dict[str, Any]] = []
    for task in ALL_TASKS:
        sub = all_rows[all_rows.task == task]
        p = sub.pivot_table(index=["seed", "fold", "subject_id"], columns="method", values=["BA", "macro_F1", "accuracy"])
        if set(p[("BA",)].columns) != {"LiteBN_BASELINE", "LiteBN_XS"}: raise RuntimeError(f"method pairing failure {task}")
        delta = p[("BA", "LiteBN_XS")] - p[("BA", "LiteBN_BASELINE")]
        for (seed, fold), values in delta.groupby(level=["seed", "fold"]):
            fold_rows.append({"task": task, "seed": int(seed), "fold": int(fold), "LiteBN_BA": float(p.loc[(seed, fold), ("BA", "LiteBN_BASELINE")].mean()),
                "XS_BA": float(p.loc[(seed, fold), ("BA", "LiteBN_XS")].mean()), "delta_pp": float(100 * values.mean()), "positive": bool(values.mean() > 0)})
        for seed, values in delta.groupby(level="seed"):
            summary_rows.append({"task": task, "seed": int(seed), "n_subjects": int(len(values)),
                "LiteBN_BA": float(p.loc[seed, ("BA", "LiteBN_BASELINE")].mean()), "XS_BA": float(p.loc[seed, ("BA", "LiteBN_XS")].mean()),
                "delta_pp": float(100 * values.mean()), "LiteBN_macro_F1": float(p.loc[seed, ("macro_F1", "LiteBN_BASELINE")].mean()),
                "XS_macro_F1": float(p.loc[seed, ("macro_F1", "LiteBN_XS")].mean()), "positive_folds": int((values.groupby(level="fold").mean() > 0).sum())})
    seed_summary, fold_summary = pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)
    aggregate_rows = []
    for task in ALL_TASKS:
        t = seed_summary[seed_summary.task == task].sort_values("seed")
        aggregate_rows.append({"task": task, "seed0_delta_pp": float(t[t.seed == 0].delta_pp.iloc[0]), "seed1_delta_pp": float(t[t.seed == 1].delta_pp.iloc[0]),
            "seed2_delta_pp": float(t[t.seed == 2].delta_pp.iloc[0]), "three_seed_mean_delta_pp": float(t.delta_pp.mean()),
            "LiteBN_three_seed_BA": float(t.LiteBN_BA.mean()), "XS_three_seed_BA": float(t.XS_BA.mean()),
            "LiteBN_three_seed_macro_F1": float(t.LiteBN_macro_F1.mean()), "XS_three_seed_macro_F1": float(t.XS_macro_F1.mean()),
            "positive_seeds": int((t.delta_pp > 0).sum()), "positive_folds": int((fold_summary[fold_summary.task == task].delta_pp > 0).sum())})
    aggregate = pd.DataFrame(aggregate_rows)
    write_csv(OUT / "DEVELOPMENT_SEED_TASK_SUMMARY.csv", seed_summary); write_csv(OUT / "DEVELOPMENT_FOLD_SUMMARY.csv", fold_summary); write_csv(OUT / "DEVELOPMENT_3SEED_SUMMARY.csv", aggregate)
    equal_mean = float(aggregate.three_seed_mean_delta_pp.mean())
    report = ["# Frozen original LiteBN-XS: three-seed outer-development summary", "", "| Task | Seed0 ΔBA pp | Seed1 ΔBA pp | Seed2 ΔBA pp | 3-seed mean ΔBA pp | positive seeds | positive folds |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggregate.itertuples(index=False): report.append(f"| {r.task} | {r.seed0_delta_pp:+.3f} | {r.seed1_delta_pp:+.3f} | {r.seed2_delta_pp:+.3f} | {r.three_seed_mean_delta_pp:+.3f} | {r.positive_seeds}/3 | {r.positive_folds}/15 |")
    report += ["", f"Equal-task mean gain: {equal_mean:+.3f} pp.", "Final heldout/test was not accessed during development completion."]
    write_text(OUT / "DEVELOPMENT_3SEED_SUMMARY.md", "\n".join(report))
    write_json(PROTOCOL / "DEVELOPMENT_COMPLETION.json", {"pass": True, "tasks": list(ALL_TASKS), "seeds": [0, 1, 2], "new_training_tasks": list(TRAIN_TASKS), "new_training_seeds": list(SEEDS), "equal_task_mean_delta_pp": equal_mean, "final_heldout_accessed": False})


def development() -> None:
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True)
    mod = load_module(); protocol_lock(mod)
    _, folds, _ = mod.load_folds()
    baseline_records = []
    for task in ALL_TASKS:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            for seed in (0, 1, 2):
                path = baseline_path(task, int(fold["fold_id"]), seed)
                if not path.is_file(): raise FileNotFoundError(path)
                model = mod.build_model("LiteBN_BASELINE", task); model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
                baseline_records.append({"task": task, "fold": int(fold["fold_id"]), "seed": seed, "path": str(path), "sha256": sha256(path), "strict_load": True})
    write_json(PROTOCOL / "BASELINE_REUSE_AUDIT.json", {"pass": True, "records": baseline_records})
    frames: list[pd.DataFrame] = [seed0_frame(task) for task in ALL_TASKS] + existing_erp_frames()
    for seed in SEEDS:
        records = train_seed(mod, seed); frames.append(evaluate_outer(mod, seed, records))
    summarize_development(frames)
    print("XS_FULL_MULTISEED_DEVELOPMENT_COMPLETE", flush=True)


def heldout_subjects() -> dict[str, list[str]]:
    raw = json.loads(V8_SPLIT.read_text(encoding="utf-8"))
    result = {"OpenBMI": [str(v) for v in raw["openbmi"]["V8_INTERNAL_HOLDOUT"]], "WBCIC": [str(v) for v in raw["wbcic"]["V8_INTERNAL_HOLDOUT"]]}
    if len(result["OpenBMI"]) != 14 or len(result["WBCIC"]) != 10: raise RuntimeError("final heldout membership mismatch")
    return result


def heldout_signal_path(mod: Any, task: str, subject: str) -> tuple[Path, Path]:
    if mod.TASKS[task]["dataset"] == "OpenBMI": return mod.openbmi_path(task, subject, 2, "signal"), mod.openbmi_path(task, subject, 2, "label")
    root = mod.CACHE / "wbcic" / "wbcic_epochs" / subject
    return root / "ses-2_epochs.npy", root / "ses-2_labels.npy"


def final_preflight() -> None:
    mod = load_module(); complete = json.loads((PROTOCOL / "DEVELOPMENT_COMPLETION.json").read_text(encoding="utf-8"))
    if not complete.get("pass"): raise RuntimeError("development completion missing")
    if (OUT / "FINAL_TEST_SUBJECT_RESULTS.csv").exists(): raise RuntimeError("final heldout has already been evaluated")
    memberships = heldout_subjects(); records = []; schema = []
    for task in ALL_TASKS:
        for seed in (0, 1, 2):
            for fold in range(5):
                for method, path in (("LiteBN_BASELINE", baseline_path(task, fold, seed)), ("LiteBN_XS", xs_checkpoint_path(task, fold, seed))):
                    if not path.is_file(): raise FileNotFoundError(path)
                    model = mod.build_model(method, task); model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True); model.eval()
                    if model.training: raise RuntimeError("model not eval-ready")
                    records.append({"task": task, "seed": seed, "fold": fold, "method": method, "checkpoint_path": str(path), "checkpoint_sha256": sha256(path), "strict_load": True})
        for subject in memberships[mod.TASKS[task]["dataset"]]:
            signal, label = heldout_signal_path(mod, task, subject)
            if not signal.is_file() or not label.is_file(): raise FileNotFoundError(f"missing final cache: {task}/{subject}")
            value = np.load(signal, mmap_mode="r", allow_pickle=False)
            expected = (mod.TASKS[task]["channels"], mod.TASKS[task]["samples"])
            if value.ndim != 3 or tuple(value.shape[1:]) != expected: raise RuntimeError(f"heldout schema mismatch: {signal}")
            schema.append({"task": task, "subject_id": subject, "signal_path": str(signal), "label_path": str(label), "shape": list(value.shape), "label_opened": False})
    if len(records) != 120: raise RuntimeError(f"checkpoint cardinality {len(records)}")
    lock = {"MODEL_FROZEN_BEFORE_FINAL_TEST": "YES", "FINAL_HELDOUT_ACCESSED": "NO", "FINAL_TEST_USED_FOR_TUNING": "NO", "development_complete": True,
        "final_membership_source": str(V8_SPLIT), "final_membership_sha256": sha256(V8_SPLIT), "canonical_inner_split_only": True,
        "checkpoint_records": records, "signal_schema": schema, "labels_opened_in_preflight": False, "optimizer_instantiated": False,
        "runner_sha256": sha256(Path(__file__)), "protocol_lock_sha256": sha256(PROTOCOL / "MULTISEED_PROTOCOL_LOCK.json")}
    write_json(PROTOCOL / "FINAL_TEST_FREEZE_LOCK.json", lock)
    print("XS_FINAL_TEST_PREFLIGHT_PASS", flush=True)


def metric(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(axis=1)
    return {"BA": float(balanced_accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, pred))}


def infer(model: torch.nn.Module, value: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    rows = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(value), 128):
            x = torch.from_numpy(np.ascontiguousarray(value[start:start + 128], dtype=np.float32)).to(device, non_blocking=True)
            x = (x - torch.as_tensor(mean, device=device)[None, :, None]) / torch.clamp(torch.as_tensor(std, device=device)[None, :, None], min=1e-6)
            rows.append(model(x)[0].float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def final_evaluate() -> None:
    lock_path = PROTOCOL / "FINAL_TEST_FREEZE_LOCK.json"
    if not lock_path.is_file(): raise RuntimeError("final-test preflight lock missing")
    if (OUT / "FINAL_TEST_SUBJECT_RESULTS.csv").exists(): raise RuntimeError("refusing repeat final heldout evaluation")
    lock = json.loads(lock_path.read_text(encoding="utf-8")); mod = load_module()
    if lock.get("runner_sha256") != sha256(Path(__file__)) or lock.get("labels_opened_in_preflight"): raise RuntimeError("final freeze lock invalid")
    memberships, device, rows = heldout_subjects(), torch.device("cuda" if torch.cuda.is_available() else "cpu"), []
    audited = {(r["task"], int(r["seed"]), int(r["fold"]), r["method"]): r for r in lock["checkpoint_records"]}
    for task in ALL_TASKS:
        dataset = mod.TASKS[task]["dataset"]
        arrays = []
        for subject in memberships[dataset]:
            signal, label = heldout_signal_path(mod, task, subject)
            x, raw_y = np.load(signal, mmap_mode="r", allow_pickle=False), np.load(label, mmap_mode="r", allow_pickle=False)
            mapping = mod.TASKS[task]["raw_codes"]; y = np.asarray([mapping[int(v)] for v in raw_y], dtype=np.int64)
            if len(x) != len(y): raise RuntimeError(f"heldout label shape mismatch: {task}/{subject}")
            arrays.append((subject, np.asarray(x, dtype=np.float32), y))
        for seed in (0, 1, 2):
            for fold in range(5):
                mean, std, norm_meta = mod.load_tensor_pair(normalizer_path(task, fold, seed))
                for method in ("LiteBN_BASELINE", "LiteBN_XS"):
                    record = audited[(task, seed, fold, method)]; path = Path(record["checkpoint_path"])
                    if sha256(path) != record["checkpoint_sha256"]: raise RuntimeError(f"checkpoint changed after freeze: {path}")
                    model = mod.build_model(method, task).to(device); model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
                    for subject, x, y in arrays:
                        rows.append({"task": task, "dataset": dataset, "seed": seed, "fold": fold, "subject_id": subject, "method": method,
                            **metric(y, infer(model, x, mean, std, device)), "trials": int(len(y)), "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm_meta["mean_std_sha256"]})
                    del model
                if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["task", "seed", "fold", "subject_id", "method"]).reset_index(drop=True)
    expected = 2 * 15 * (14 * 3 + 10)
    if len(frame) != expected or frame.duplicated(["task", "seed", "fold", "subject_id", "method"]).any(): raise RuntimeError(f"final cardinality invalid: {len(frame)}/{expected}")
    write_csv(OUT / "FINAL_TEST_SUBJECT_RESULTS.csv", frame); summarize_final(frame)
    print("XS_FINAL_HELDOUT_EVALUATION_COMPLETE", flush=True)


def summarize_final(frame: pd.DataFrame) -> None:
    rows = []
    for task in ALL_TASKS:
        sub = frame[frame.task == task]; p = sub.pivot_table(index=["seed", "fold", "subject_id"], columns="method", values=["BA", "macro_F1", "accuracy"])
        delta = p[("BA", "LiteBN_XS")] - p[("BA", "LiteBN_BASELINE")]
        rows.append({"task": task, "LiteBN_test_BA": float(p[("BA", "LiteBN_BASELINE")].mean()), "XS_test_BA": float(p[("BA", "LiteBN_XS")].mean()), "delta_BA_pp": float(100 * delta.mean()),
            "LiteBN_test_macro_F1": float(p[("macro_F1", "LiteBN_BASELINE")].mean()), "XS_test_macro_F1": float(p[("macro_F1", "LiteBN_XS")].mean()),
            "positive_seed_means": int((delta.groupby(level="seed").mean() > 0).sum()), "positive_fold_means": int((delta.groupby(level=["seed", "fold"]).mean() > 0).sum())})
    summary = pd.DataFrame(rows); write_csv(OUT / "FINAL_TEST_TASK_SUMMARY.csv", summary)
    equal_mean = float(summary.delta_BA_pp.mean()); openbmi_mean = float(summary[summary.task.str.startswith("OpenBMI")].delta_BA_pp.mean()); wbcic_mean = float(summary[summary.task == "WBCIC_MI"].delta_BA_pp.iloc[0]); dataset_balanced = (openbmi_mean + wbcic_mean) / 2
    report = ["# Frozen original LiteBN-XS final heldout/test", "", "| Task | LiteBN BA | XS BA | ΔBA pp | LiteBN macro-F1 | XS macro-F1 |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples(index=False): report.append(f"| {r.task} | {r.LiteBN_test_BA:.6f} | {r.XS_test_BA:.6f} | {r.delta_BA_pp:+.3f} | {r.LiteBN_test_macro_F1:.6f} | {r.XS_test_macro_F1:.6f} |")
    report += ["", f"Equal-task mean ΔBA: {equal_mean:+.3f} pp.", f"Dataset-balanced mean ΔBA: {dataset_balanced:+.3f} pp.", f"Positive tasks: {int((summary.delta_BA_pp > 0).sum())}/4.", "", "MODEL_FROZEN_BEFORE_FINAL_TEST = YES", "FINAL_HELDOUT_ACCESSED = YES", "FINAL_TEST_USED_FOR_TUNING = NO"]
    write_text(OUT / "FINAL_TEST_REPORT.md", "\n".join(report))
    write_json(OUT / "FINAL_TEST_SUMMARY.json", {"tasks": rows, "equal_task_mean_delta_pp": equal_mean, "dataset_balanced_mean_delta_pp": dataset_balanced, "positive_tasks": int((summary.delta_BA_pp > 0).sum()), "MODEL_FROZEN_BEFORE_FINAL_TEST": "YES", "FINAL_HELDOUT_ACCESSED": "YES", "FINAL_TEST_USED_FOR_TUNING": "NO"})


def main() -> None:
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--development", action="store_true"); group.add_argument("--final-preflight", action="store_true"); group.add_argument("--final-evaluate", action="store_true")
    args = parser.parse_args()
    if args.development: development()
    elif args.final_preflight: final_preflight()
    else: final_evaluate()


if __name__ == "__main__":
    main()
