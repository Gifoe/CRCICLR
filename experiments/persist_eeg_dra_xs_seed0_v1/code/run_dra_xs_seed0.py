#!/usr/bin/env python3
"""LiteBN-DRA-XS seed-0 canonical-inner screen.

The only intervention relative to the random-initialized LiteBN-XS recipe is
the epoch-dependent admission of the existing channel residual.  This script
does not load trained XS weights; it initializes the exact XS architecture
from seed 0 and uses only canonical inner train/validation subjects.  ERP is
run first and WBCIC is opened only when the pre-registered ERP signal rule is
met.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")
BASE_CODE = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py"
BASE_OUT = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/outputs"
BASE_PROTOCOL = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/protocol"
EXP = REPO / "experiments/persist_eeg_dra_xs_seed0_v1"
OUT = EXP / "outputs_screen"
PROTOCOL = EXP / "protocol"
# v2 runtime keeps the interrupted diagnostic run intact after the validation
# scale bug fix; no old DRA runtime/checkpoint is overwritten.
RUNTIME = Path("/root/rivermind-data/dra_xs_seed0_runtime_v2")

SEED = 0
MAX_EPOCHS = 60
MIN_SELECTION_EPOCH = 10
PATIENCE = 10
LR = 3e-4
WEIGHT_DECAY = 5e-4
CLIP = 5.0
TOL = 1e-12
ERP_MIN_MEAN_PP = 0.20


def load_module():
    spec = importlib.util.spec_from_file_location("dra_litebn_x", BASE_CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    import io
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if np.isfinite(item) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(value, tmp)
    os.replace(tmp, path)


def make_dra_class(module):
    class DRALiteBNXS(module.LiteBNEnhanced):
        def __init__(self, channels: int, classes: int):
            super().__init__(channels, classes, "LiteBN_XS")
            self.admission_scale = 1.0
            self._last_abs = 0.0
            self._last_rms = 0.0

        def set_admission_scale(self, scale: float) -> None:
            self.admission_scale = float(scale)

        def apply_channel_gate(self, value: torch.Tensor) -> torch.Tensor:
            scale = float(self.admission_scale)
            if scale == 0.0:
                self._last_abs = 0.0
                self._last_rms = 0.0
                return value
            rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
            diff_rms = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + 1e-8)
            raw = self.channel_mlp(torch.stack((rms, diff_rms), dim=-1)).squeeze(-1)
            residual = torch.tanh(self.lambda_channel) * torch.tanh(raw)
            residual = residual * scale
            self._last_abs = float(residual.detach().abs().mean().cpu())
            self._last_rms = float(torch.sqrt(residual.detach().square().mean()).cpu())
            return value * (1.0 + residual).unsqueeze(-1)

        @property
        def last_abs(self) -> float:
            return float(self._last_abs)

        @property
        def last_rms(self) -> float:
            return float(self._last_rms)

    return DRALiteBNXS


def admission_schedule(epoch: int) -> float:
    if epoch <= 5:
        return 0.0
    if epoch < 10:
        return (epoch - 5) / 5.0
    return 1.0


def baseline_references(module, task: str, fold: int, replay: pd.DataFrame) -> dict[str, Any]:
    stage = pd.read_csv(BASE_OUT / "STAGE_A_INNERVAL_RESULTS.csv")
    base = stage[(stage.task == task) & (stage.fold == fold) & (stage.architecture == "LiteBN_BASELINE")]
    if len(base) != 1:
        raise RuntimeError(f"missing LiteBN baseline inner reference {task}/f{fold}")
    xs = replay[(replay.task == task) & (replay.fold == fold)]
    if len(xs) != 1 or not bool(xs.iloc[0]["pass"]):
        raise RuntimeError(f"missing exact XS replay reference {task}/f{fold}")
    return {
        "LiteBN_BA": float(base.iloc[0].best_inner_val_BA),
        "LiteBN_macro_F1": float(base.iloc[0].best_inner_val_macro_F1),
        "XS_BA": float(xs.iloc[0].XS_BA),
        "XS_macro_F1": float(xs.iloc[0].XS_macro_F1),
        "XS_accuracy": float(xs.iloc[0].XS_accuracy),
        "XS_checkpoint": str(xs.iloc[0].XS_checkpoint),
        "XS_checkpoint_sha256": str(xs.iloc[0].XS_checkpoint_sha256),
        "replay_max_abs_logit_difference": float(xs.iloc[0].max_abs_logit_difference),
        "replay_prediction_mismatch_count": int(xs.iloc[0].prediction_mismatch_count),
    }


def load_initial_hashes() -> dict[tuple[str, int], str]:
    path = BASE_PROTOCOL / "CHECKPOINT_PROVENANCE.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in data.get("records", []):
        if row.get("architecture") == "LiteBN_XS":
            result[(str(row["task"]), int(row["fold"]))] = str(row.get("initial_sha256", ""))
    return result


def evaluate(module, model: nn.Module, bundle, cache, subjects: Iterable[str], mean: np.ndarray,
             std: np.ndarray, scale: float) -> dict[str, float]:
    model.eval()
    model.set_admission_scale(scale)
    rows, abs_values, rms_values = [], [], []
    future = (int(module.TASKS[bundle.task]["future_session"]),)
    with torch.no_grad():
        for subject in module.subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], future)
            labels, logits = [], []
            for start in range(0, len(indices), 128):
                value, y = cache.batch(indices[start:start + 128], mean, std)
                out, _ = model(value)
                labels.append(y.cpu().numpy())
                logits.append(out.float().cpu().numpy())
                abs_values.append(model.last_abs)
                rms_values.append(model.last_rms)
            rows.append(module.classification_metrics(np.concatenate(labels), np.concatenate(logits)))
    result = {key: float(np.mean([row[key] for row in rows])) for key in ("BA", "macro_F1", "accuracy")}
    result["mean_abs_residual"] = float(np.mean(abs_values)) if abs_values else 0.0
    result["rms_residual"] = float(np.mean(rms_values)) if rms_values else 0.0
    return result


def train_cell(module, task: str, fold: dict[str, Any], replay: pd.DataFrame,
               expected_initial: str, device: torch.device, DRAModel) -> dict[str, Any]:
    fid = int(fold["fold_id"])
    record_path = RUNTIME / "records" / task.lower() / f"fold{fid}.json"
    latest_path = RUNTIME / "checkpoints" / task.lower() / f"fold{fid}_dra_xs" / "checkpoint_latest.pt"
    selected_path = RUNTIME / "checkpoints" / task.lower() / f"fold{fid}_dra_xs" / "selected_best.pt"
    bundle = module.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
    mean, std, norm_meta = module.normalizer(bundle, fold["inner_train_subjects"])
    module.save_tensor_pair(RUNTIME / "normalizers" / f"{task.lower()}_fold{fid}.npz", mean, std, norm_meta)
    cache = module.RawGPUCache(bundle, device)
    train_indices = bundle.indices(fold["inner_train_subjects"], module.TASKS[task]["source_sessions"])
    references = baseline_references(module, task, fid, replay)
    batch_info = {"kind": "historical_task_full_permutation_batch64", "steps_per_epoch": int(math.ceil(len(train_indices) / module.TASK_BATCH))}

    module.set_seed(SEED)
    model = DRAModel(int(module.TASKS[task]["channels"]), int(module.TASKS[task]["classes"])).to(device)
    initial_hash = state_hash(model)
    if expected_initial and initial_hash != expected_initial:
        raise RuntimeError(f"DRA initial state mismatch {task}/f{fid}: {initial_hash} != {expected_initial}")
    invariants = {
        "task": task, "fold": fid, "architecture": "LiteBN_DRA_XS", "base_architecture": "LiteBN_XS",
        "seed": SEED, "initial_sha256": initial_hash, "original_xs_initial_sha256": expected_initial,
        "normalizer_sha256": norm_meta["mean_std_sha256"], "max_epochs": MAX_EPOCHS,
        "min_selection_epoch": MIN_SELECTION_EPOCH, "patience": PATIENCE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "clip": CLIP, "admission_schedule": "epoch1-5=0; epoch6-9=(epoch-5)/5; epoch10+=1",
        "source_code_sha256": sha256_file(Path(__file__)),
    }
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("invariants") == invariants:
            return record
        raise RuntimeError(f"DRA completed record invariant mismatch: {record_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    class_weight, _ = module.class_weights(bundle, fold["inner_train_subjects"])
    class_weight = None if class_weight is None else class_weight.to(device)
    start, history = 1, []
    best_ba, best_f1, best_epoch, best_state, bad_epochs = -float("inf"), -float("inf"), None, None, 0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants:
            raise RuntimeError(f"DRA resume invariant mismatch: {latest_path}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        module.restore_rng(saved["rng"])
        start = int(saved["epoch"]) + 1
        history = list(saved["history"])
        best_ba, best_f1, best_epoch = float(saved["best_ba"]), float(saved["best_f1"]), saved["best_epoch"]
        best_state, bad_epochs = saved["best_state"], int(saved.get("bad_epochs", 0))

    # Match the validated original XS runner's post-construction RNG reset;
    # fold identity enters only through its canonical batch schedule.
    module.set_seed(SEED + 100_000)
    started = time.perf_counter()
    for epoch in range(start, MAX_EPOCHS + 1):
        scale = admission_schedule(epoch)
        model.set_admission_scale(scale)
        model.train()
        batches = module.task_epoch_batches(train_indices, task, fid, epoch)
        losses, residual_abs, residual_rms = [], [], []
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(value)
                loss = F.cross_entropy(logits, labels, weight=class_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite DRA loss {task}/f{fid}/epoch{epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            residual_abs.append(model.last_abs)
            residual_rms.append(model.last_rms)
        # Selection observes the model under the same admission scale that was
        # used for this training epoch.  Thus epochs 1-5 are truly bypassed,
        # epochs 6-9 are ramp observations, and epoch 10+ is fully admitted.
        validation = evaluate(module, model, bundle, cache, fold["inner_val_subjects"], mean, std, scale)
        val_ba, val_f1 = float(validation["BA"]), float(validation["macro_F1"])
        eligible = epoch >= MIN_SELECTION_EPOCH
        improved = bool(eligible and (val_ba > best_ba + TOL or (abs(val_ba - best_ba) <= TOL and val_f1 > best_f1 + TOL)))
        if improved:
            best_ba, best_f1, best_epoch = val_ba, val_f1, int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        elif eligible:
            bad_epochs += 1
        stop_now = bool(eligible and bad_epochs >= PATIENCE)
        row = {
            "task": task, "dataset": module.TASKS[task]["dataset"], "fold": fid, "epoch": epoch,
            "admission_scale": scale, "training_loss": float(np.mean(losses)),
            "inner_val_BA": val_ba, "inner_val_macro_F1": val_f1, "inner_val_accuracy": float(validation["accuracy"]),
            "selected": improved, "eligible_for_selection": eligible, "bad_epochs": bad_epochs, "patience": PATIENCE,
            "lambda_channel": float(model.lambda_channel.detach().cpu()),
            "mean_abs_residual": float(np.mean(residual_abs)) if residual_abs else 0.0,
            "rms_residual": float(np.mean(residual_rms)) if residual_rms else 0.0, "stopped_early": stop_now,
        }
        history.append(row)
        atomic_torch_save(latest_path, {"epoch": epoch, "history": history, "best_ba": best_ba, "best_f1": best_f1,
                                        "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(),
                                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": module.rng_state(),
                                        "bad_epochs": bad_epochs, "invariants": invariants})
        if epoch == 1 or epoch % 5 == 0 or improved:
            print(f"[DRA {task} f{fid}] epoch={epoch:02d} scale={scale:.2f} valBA={val_ba:.5f} best={best_ba:.5f} bad={bad_epochs}/{PATIENCE}", flush=True)
        if stop_now:
            print(f"[DRA {task} f{fid}] EARLY_STOP epoch={epoch:02d} best_epoch={best_epoch}", flush=True)
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError(f"no eligible DRA checkpoint {task}/f{fid}")
    model.load_state_dict(best_state, strict=True)
    atomic_torch_save(selected_path, {"state_dict": model.state_dict(), "selected_epoch": best_epoch, "invariants": invariants})
    selected_row = next(row for row in history if int(row["epoch"]) == int(best_epoch))
    normal = evaluate(module, model, bundle, cache, fold["inner_val_subjects"], mean, std, 1.0)
    gateoff = evaluate(module, model, bundle, cache, fold["inner_val_subjects"], mean, std, 0.0)
    record = {
        "task": task, "dataset": module.TASKS[task]["dataset"], "fold": fid, "architecture": "LiteBN_DRA_XS", "seed": SEED,
        "invariants": invariants, "references": references, "history": history, "selected": selected_row,
        "selected_epoch": int(best_epoch), "selected_admission_scale": admission_schedule(int(best_epoch)),
        "selected_normal": normal, "selected_gateoff": gateoff, "gateoff_delta_BA": normal["BA"] - gateoff["BA"],
        "selected_checkpoint": str(selected_path), "selected_checkpoint_sha256": sha256_file(selected_path),
        "elapsed_seconds": time.perf_counter() - started, "epochs_completed": len(history),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_state_hash": initial_hash, "original_xs_initial_hash": expected_initial,
    }
    write_json(record_path, record)
    del model, cache, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def fold_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        ref, selected = record["references"], record["selected_normal"]
        rows.append({
            "task": record["task"], "dataset": record["dataset"], "fold": record["fold"],
            "LiteBN_BA": ref["LiteBN_BA"], "LiteBN_macro_F1": ref["LiteBN_macro_F1"],
            "XS_BA": ref["XS_BA"], "XS_macro_F1": ref["XS_macro_F1"],
            "DRA_BA": selected["BA"], "DRA_macro_F1": selected["macro_F1"], "DRA_accuracy": selected["accuracy"],
            "delta_BA_vs_XS": selected["BA"] - ref["XS_BA"], "delta_BA_vs_XS_pp": 100.0 * (selected["BA"] - ref["XS_BA"]),
            "delta_BA_vs_LiteBN_pp": 100.0 * (selected["BA"] - ref["LiteBN_BA"]),
            "selected_epoch": record["selected_epoch"], "selected_admission_scale": record["selected_admission_scale"],
            "lambda_channel": record["selected"]["lambda_channel"], "mean_abs_residual": record["selected"]["mean_abs_residual"],
            "rms_residual": record["selected"]["rms_residual"], "gateoff_BA": record["selected_gateoff"]["BA"],
            "gateoff_delta_BA": record["gateoff_delta_BA"], "epochs_completed": record["epochs_completed"],
            "early_stop_epoch": record["history"][-1]["epoch"], "parameter_count": record["parameter_count"],
            "initial_state_matches_original_xs": bool(record["initial_state_hash"] == record["original_xs_initial_hash"]),
        })
    return pd.DataFrame(rows).sort_values(["task", "fold"]).reset_index(drop=True)


def task_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in frame.groupby("task", sort=True):
        delta = group["delta_BA_vs_XS"].to_numpy(float)
        rows.append({
            "task": task, "dataset": group["dataset"].iloc[0], "folds": len(group),
            "LiteBN_BA": group["LiteBN_BA"].mean(), "XS_BA": group["XS_BA"].mean(), "DRA_BA": group["DRA_BA"].mean(),
            "mean_delta_vs_XS_pp": 100.0 * delta.mean(), "mean_delta_vs_LiteBN_pp": group["delta_BA_vs_LiteBN_pp"].mean(),
            "positive_folds_vs_XS": int((delta > TOL).sum()), "zero_folds_vs_XS": int((np.abs(delta) <= TOL).sum()),
            "negative_folds_vs_XS": int((delta < -TOL).sum()), "mean_selected_epoch": group["selected_epoch"].mean(),
            "mean_selected_admission_scale": group["selected_admission_scale"].mean(), "mean_abs_residual": group["mean_abs_residual"].mean(),
            "mean_gateoff_delta_BA": group["gateoff_delta_BA"].mean(),
        })
    return pd.DataFrame(rows)


def decide_erp(summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    mean_pp = float(row["mean_delta_vs_XS_pp"])
    positive = int(row["positive_folds_vs_XS"])
    if mean_pp >= ERP_MIN_MEAN_PP and positive >= 3:
        return "DRA_ERP_SIGNAL_FOUND"
    if mean_pp <= 0.0 or positive < 2:
        return "DRA_ERP_NO_SIGNAL"
    return "DRA_ERP_BORDERLINE"


def decide_wbcic(summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    if float(row["DRA_BA"]) > float(row["LiteBN_BA"]) and float(row["mean_delta_vs_XS_pp"]) >= -0.20:
        return "DRA_ERP_REPAIRED_WBCIC_PRESERVED"
    return "DRA_WBCIC_ADVANTAGE_LOST"


def run_task(module, task: str, folds: dict[str, list[dict[str, Any]]], replay: pd.DataFrame,
             initial_hashes: dict[tuple[str, int], str], device: torch.device, DRAModel):
    records = []
    dataset = module.TASKS[task]["dataset"]
    for fold in folds[dataset]:
        fid = int(fold["fold_id"])
        record = train_cell(module, task, fold, replay, initial_hashes.get((task, fid), ""), device, DRAModel)
        records.append(record)
        print(f"[DRA] completed {task} fold{fid} selected_epoch={record['selected_epoch']}", flush=True)
    frame = fold_rows(records)
    traj = pd.DataFrame([row for record in records for row in record["history"]]).sort_values(["task", "fold", "epoch"])
    diag = frame[["task", "fold", "DRA_BA", "gateoff_BA", "gateoff_delta_BA", "lambda_channel", "mean_abs_residual", "rms_residual"]].copy()
    summary = task_summary(frame)
    tag = "ERP" if task == "OpenBMI_ERP" else "WBCIC"
    write_csv(OUT / f"DRA_{tag}_FOLD_RESULTS.csv", frame)
    write_csv(OUT / f"DRA_{tag}_TASK_SUMMARY.csv", summary)
    write_csv(OUT / f"DRA_{tag}_TRAINING_TRAJECTORY.csv", traj)
    write_csv(OUT / f"DRA_{tag}_GATE_DIAGNOSTICS.csv", diag)
    return records, frame, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    module = load_module()
    _, folds, split_hash = module.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DRAModel = make_dra_class(module)
    replay = pd.read_csv(REPO / "experiments/persist_eeg_xs_residual_head_seed0_v1/outputs_single_inner/EPOCH0_EXACT_XS_REPLAY.csv")
    initial_hashes = load_initial_hashes()
    protocol = [
        "# LiteBN-DRA-XS seed0 protocol", "",
        "Exact LiteBN-XS architecture from random seed 0; no trained XS/X checkpoint is loaded.",
        "Single intervention: admission of the existing channel residual.",
        "Admission: epochs 1-5 scale 0; epochs 6-9 scale (epoch-5)/5; epoch 10+ scale 1.",
        "At scale 0 the channel MLP and lambda_channel path are bypassed completely; all other XS parameters train normally.",
        f"AdamW lr={LR}; weight_decay={WEIGHT_DECAY}; clip={CLIP}; max_epochs={MAX_EPOCHS}; min_selection_epoch={MIN_SELECTION_EPOCH}; patience={PATIENCE}.",
        "Selection: canonical inner-validation subject-mean BA; ties use higher macro-F1 then earlier epoch.",
        "Stage 1 opens OpenBMI ERP only. WBCIC MI opens only after the frozen ERP clear-signal rule.",
        f"ERP clear signal: mean DRA-vs-XS >= +{ERP_MIN_MEAN_PP:.2f} pp and at least 3/5 positive folds.",
        f"canonical split sha256: `{split_hash}`", "SEED = 0", "CANONICAL_INNER_SPLITS_ONLY = YES",
        "NEW_INNER_SPLITS_CREATED = NO", "OUTER_DEVELOPMENT_ACCESSED = NO", "FINAL_HELDOUT_ACCESSED = NO",
    ]
    write_text(PROTOCOL / "DRA_XS_PROTOCOL.md", "\n".join(protocol))
    erp_records, erp_frame, erp_summary = run_task(module, "OpenBMI_ERP", folds, replay, initial_hashes, device, DRAModel)
    erp_terminal = decide_erp(erp_summary)
    all_records, all_frames, all_summaries = list(erp_records), [erp_frame], [erp_summary]
    terminal = erp_terminal
    stage2_reason = "ERP clear-signal rule not met"
    if erp_terminal == "DRA_ERP_SIGNAL_FOUND":
        stage2_reason = "ERP clear-signal rule met"
        try:
            w_records, w_frame, w_summary = run_task(module, "WBCIC_MI", folds, replay, initial_hashes, device, DRAModel)
            all_records.extend(w_records); all_frames.append(w_frame); all_summaries.append(w_summary)
            terminal = decide_wbcic(w_summary)
        except Exception as exc:
            terminal = "DRA_ERP_SIGNAL_FOUND_WBCIC_NOT_RUN"
            write_text(OUT / "DRA_STAGE2_ERROR.txt", repr(exc))
            print(f"[DRA] stage2 engineering failure: {exc!r}", flush=True)

    full_frame = pd.concat(all_frames, ignore_index=True)
    full_summary = pd.concat(all_summaries, ignore_index=True)
    write_csv(OUT / "DRA_FOLD_RESULTS.csv", full_frame)
    write_csv(OUT / "DRA_TASK_SUMMARY.csv", full_summary)
    metadata = {
        "experiment": "persist_eeg_dra_xs_seed0_v1", "model": "LiteBN_DRA_XS", "seed": SEED,
        "stages_run": ["OpenBMI_ERP"] + (["WBCIC_MI"] if len(all_frames) > 1 else []),
        "stage2_reason": stage2_reason, "terminal": terminal, "max_epochs": MAX_EPOCHS,
        "min_selection_epoch": MIN_SELECTION_EPOCH, "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "admission_schedule": "epoch1-5=0; epoch6-9=(epoch-5)/5; epoch10+=1", "canonical_split_sha256": split_hash,
        "source_code_sha256": sha256_file(Path(__file__)), "parameter_count": int(full_frame.iloc[0]["parameter_count"]),
        "outer_development_accessed": False, "final_heldout_accessed": False,
    }
    write_json(OUT / "DRA_METADATA.json", metadata)
    lines = ["# LiteBN-DRA-XS seed0 result", "", f"Terminal: **{terminal}**", "", f"Stages run: {', '.join(metadata['stages_run'])}.", f"Stage 2: {stage2_reason}.", "", "| Task | LiteBN BA | XS BA | DRA BA | Δ vs XS (pp) | Δ vs LiteBN (pp) | positive folds vs XS |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in full_summary.itertuples(index=False):
        lines.append(f"| {row.task.replace('_', ' ')} | {row.LiteBN_BA:.4f} | {row.XS_BA:.4f} | {row.DRA_BA:.4f} | {row.mean_delta_vs_XS_pp:+.3f} | {row.mean_delta_vs_LiteBN_pp:+.3f} | {int(row.positive_folds_vs_XS)}/{int(row.folds)} |")
    lines += ["", "## ERP admission timeline", "", "The trajectory CSV retains epochs 1-5 (residual bypassed), 6-9 (0-to-1 ramp), and 10+ (fully admitted) for every ERP fold. The report does not make a causal claim from one seed.", "", "## Diagnostics", "", "Gate-off is diagnostic only and was not used for selection. `gateoff_delta_BA` is normal selected-checkpoint BA minus scale-0 BA.", "", f"`SEED = {SEED}`", "`CANONICAL_INNER_SPLITS_ONLY = YES`", "`NEW_INNER_SPLITS_CREATED = NO`", "`OUTER_DEVELOPMENT_ACCESSED = NO`", "`FINAL_HELDOUT_ACCESSED = NO`"]
    write_text(OUT / "DRA_RESULTS.md", "\n".join(lines))
    print(f"DRA_COMPLETE terminal={terminal}", flush=True)
    print("OUTER_DEVELOPMENT_ACCESSED = NO", flush=True)
    print("FINAL_HELDOUT_ACCESSED = NO", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
