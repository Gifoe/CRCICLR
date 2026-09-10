#!/usr/bin/env python3
"""LiteBN-BPXS seed0 canonical-inner first screen.

This experiment keeps the exact random-initialized LiteBN-XS architecture and
changes only the training objective: the normal channel-gated path is trained
alongside a gate-off base path and a directional KL preservation term.  Only
OpenBMI ERP and WBCIC MI are opened, using the frozen canonical inner split.
No outer-development or final-heldout rows are loaded.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import subprocess
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
MOD_PATH = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py"
EXP = REPO / "experiments/persist_eeg_base_preserving_xs_seed0_v1"
OUT = EXP / "outputs_first_screen"
PROTOCOL = EXP / "protocol"
RUNTIME = Path("/root/rivermind-data/bpxs_first_screen_runtime")
BASE_OUT = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/outputs"
SOURCE_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")

TASK_ORDER = ("OpenBMI_ERP", "WBCIC_MI")
ARCHITECTURE = "LiteBN_XS"
SEED = 0
EPOCHS = 60
MIN_EPOCH = 10
PATIENCE = 5
LR = 3e-4
WEIGHT_DECAY = 5e-4
CLIP = 5.0
ALPHA = 0.5
GAMMA = 0.1
TEMPERATURE = 1.0
TIE_TOL = 1e-12


def load_module():
    spec = importlib.util.spec_from_file_location("litebn_x_bpxs", MOD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MOD_PATH}")
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


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    os.replace(temporary, path)


class BPXSModel(nn.Module):
    """Exact LiteBN-XS core with a switch that disables only channel gating."""

    def __init__(self, module, task: str):
        super().__init__()
        spec = module.TASKS[task]
        self.core = module.LiteBNEnhanced(int(spec["channels"]), int(spec["classes"]), ARCHITECTURE)
        self._gate_on = True
        self._last_abs = 0.0
        self._last_rms = 0.0

        def channel_gate(value: torch.Tensor) -> torch.Tensor:
            if not self._gate_on:
                self._last_abs = 0.0
                self._last_rms = 0.0
                return value
            rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
            diff_rms = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + 1e-8)
            raw = self.core.channel_mlp(torch.stack((rms, diff_rms), dim=-1)).squeeze(-1)
            residual = torch.tanh(self.core.lambda_channel) * torch.tanh(raw)
            self._last_abs = float(residual.detach().abs().mean().cpu())
            self._last_rms = float(torch.sqrt(residual.detach().square().mean()).cpu())
            return value * (1.0 + residual).unsqueeze(-1)

        # LiteBNEnhanced.forward calls this instance method.  The replacement
        # changes only the XS channel residual path; scale, stem, mixers,
        # embedding, classifier, dropout and normalization remain untouched.
        self.core.apply_channel_gate = channel_gate

    def forward(self, value: torch.Tensor, channel_gate: bool = True):
        self._gate_on = bool(channel_gate)
        try:
            return self.core(value)
        finally:
            self._gate_on = True

    @property
    def last_abs(self) -> float:
        return float(self._last_abs)

    @property
    def last_rms(self) -> float:
        return float(self._last_rms)


def state_hash(model: nn.Module) -> str:
    import io

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def checkpoint_path(task: str, fold: int, which: str) -> Path:
    return RUNTIME / "checkpoints" / task.lower() / f"fold{fold}_bpxs" / which


def record_path(task: str, fold: int) -> Path:
    return RUNTIME / "records" / task.lower() / f"fold{fold}.json"


def resume_invariants_match(saved: dict[str, Any], current: dict[str, Any]) -> bool:
    """Allow only non-scientific legacy differences when resuming.

    The interrupted first implementation did not record ``patience`` and its
    source-code hash necessarily differs after this repair.  All scientific
    invariants must remain byte-for-byte equal.
    """
    for key, expected in current.items():
        if key == "source_code_sha256":
            continue
        if key == "patience" and key not in saved:
            continue
        if saved.get(key) != expected:
            return False
    return True


def derive_early_stop_state(history: list[dict[str, Any]]) -> tuple[float, int | None, int, int | None]:
    """Reconstruct best/patience state from a possibly legacy trajectory."""
    best = -float("inf")
    best_epoch: int | None = None
    bad_epochs = 0
    cutoff: int | None = None
    for row in sorted(history, key=lambda item: int(item["epoch"])):
        epoch = int(row["epoch"])
        if epoch < MIN_EPOCH:
            continue
        value = float(row["gate_on_val_BA"])
        if value > best + TIE_TOL:
            best = value
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                cutoff = epoch
                break
    return best, best_epoch, bad_epochs, cutoff


def set_seed(module, seed: int) -> None:
    module.set_seed(seed)


def batch_info(module, task: str, fold: dict[str, Any], bundle) -> dict[str, Any]:
    if module.TASKS[task]["mi_protocol"]:
        episodes, metadata = module.mi_manifest(bundle, fold, task)
        return {**metadata, "episodes": episodes}
    indices = bundle.indices(fold["inner_train_subjects"], module.TASKS[task]["source_sessions"])
    return {"kind": "historical_task_full_permutation_batch64", "manifest_sha256": None,
            "steps_per_epoch": int(np.ceil(len(indices) / module.TASK_BATCH))}


def evaluate(module, model: BPXSModel, bundle, cache, subjects: Iterable[str], mean: np.ndarray,
             std: np.ndarray, gate_on: bool) -> dict[str, float]:
    model.eval()
    rows = []
    residual_abs = []
    residual_rms = []
    future = (int(module.TASKS[bundle.task]["future_session"]),)
    with torch.no_grad():
        for subject in module.subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], future)
            labels, logits = [], []
            for start in range(0, len(indices), 128):
                idx = indices[start:start + 128]
                value, y = cache.batch(idx, mean, std)
                out, _ = model(value, channel_gate=gate_on)
                labels.append(y.cpu().numpy())
                logits.append(out.float().cpu().numpy())
                residual_abs.append(model.last_abs)
                residual_rms.append(model.last_rms)
            rows.append(module.classification_metrics(np.concatenate(labels), np.concatenate(logits)))
    result = {key: float(np.mean([row[key] for row in rows])) for key in ("BA", "macro_F1", "accuracy")}
    result["mean_abs_channel_residual"] = float(np.mean(residual_abs)) if residual_abs else 0.0
    result["rms_channel_residual"] = float(np.mean(residual_rms)) if residual_rms else 0.0
    return result


def reference_results(module) -> dict[tuple[str, int], dict[str, Any]]:
    path = BASE_OUT / "STAGE_A_INNERVAL_RESULTS.csv"
    frame = pd.read_csv(path)
    xs_replay = pd.read_csv(REPO / "experiments/persist_eeg_xs_residual_head_seed0_v1/outputs_single_inner/EPOCH0_EXACT_XS_REPLAY.csv")
    result = {}
    for task in TASK_ORDER:
        for fold in range(5):
            rows = frame[(frame.task == task) & (frame.fold == fold)]
            if "LiteBN_BASELINE" not in set(rows.architecture):
                raise RuntimeError(f"missing LiteBN baseline reference for {task}/f{fold}")
            base = rows[rows.architecture == "LiteBN_BASELINE"].iloc[0]
            xs_rows = xs_replay[(xs_replay.task == task) & (xs_replay.fold == fold)]
            if len(xs_rows) != 1:
                raise RuntimeError(f"missing exact XS replay reference for {task}/f{fold}")
            xs = xs_rows.iloc[0]
            base_path = SOURCE_RUNTIME / "checkpoints" / task.lower() / f"fold{fold}_litebn_baseline" / "selected_best.pt"
            xs_path = SOURCE_RUNTIME / "checkpoints" / task.lower() / f"fold{fold}_litebn_xs" / "selected_best.pt"
            if not base_path.is_file() or not xs_path.is_file():
                raise FileNotFoundError(f"reference checkpoint missing for {task}/f{fold}")
            result[(task, fold)] = {
                "LiteBN_BA": float(base.best_inner_val_BA),
                "LiteBN_macro_F1": float(base.best_inner_val_macro_F1),
                "XS_BA": float(xs.XS_BA),
                "XS_macro_F1": float(xs.XS_macro_F1),
                "LiteBN_checkpoint": str(base_path),
                "LiteBN_checkpoint_sha256": sha256_file(base_path),
                "XS_checkpoint": str(xs_path),
                "XS_checkpoint_sha256": sha256_file(xs_path),
            }
    return result


def train_cell(module, task: str, fold: dict[str, Any], device: torch.device,
               references: dict[str, Any]) -> dict[str, Any]:
    fid = int(fold["fold_id"])
    path = record_path(task, fid)
    bundle = module.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
    mean, std, norm_meta = module.normalizer(bundle, fold["inner_train_subjects"])
    module.save_tensor_pair(RUNTIME / "normalizers" / f"{task.lower()}_fold{fid}.npz", mean, std, norm_meta)
    cache = module.RawGPUCache(bundle, device)
    batch = batch_info(module, task, fold, bundle)
    weights, weight_meta = module.class_weights(bundle, fold["inner_train_subjects"])
    weight = None if weights is None else weights.to(device)

    set_seed(module, SEED)
    model = BPXSModel(module, task).to(device)
    initial_hash = state_hash(model)
    invariants = {
        "task": task, "fold": fid, "architecture": ARCHITECTURE, "seed": SEED,
        "initial_sha256": initial_hash, "normalizer_sha256": norm_meta["mean_std_sha256"],
        "batch_manifest_sha256": batch.get("manifest_sha256"), "class_weight_meta": weight_meta,
        "alpha": ALPHA, "gamma": GAMMA, "temperature": TEMPERATURE,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS, "min_epoch": MIN_EPOCH,
        "patience": PATIENCE,
        "source_code_sha256": sha256_file(Path(__file__)),
    }
    latest = checkpoint_path(task, fid, "checkpoint_latest.pt")
    selected = checkpoint_path(task, fid, "selected_best.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    bad_epochs = 0
    stopped_early = False
    resumed = False
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if not resume_invariants_match(saved.get("invariants", {}), invariants):
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        module.restore_rng(saved["rng"])
        start = int(saved["epoch"]) + 1
        history = list(saved["history"])
        best = float(saved["best"])
        best_epoch = saved["best_epoch"]
        best_state = saved["best_state"]
        bad_epochs = int(saved.get("bad_epochs", 0))
        stopped_early = bool(saved.get("stopped_early", False))
        # The old run was allowed to continue beyond the new patience rule.
        # Reconstruct the canonical trajectory and truncate it at the first
        # epoch where five eligible validation checks had failed to improve.
        legacy_patience = "patience" not in saved.get("invariants", {})
        if legacy_patience or "bad_epochs" not in saved:
            derived_best, derived_epoch, derived_bad, cutoff = derive_early_stop_state(history)
            if derived_epoch is not None:
                best, best_epoch, bad_epochs = derived_best, derived_epoch, derived_bad
            if cutoff is not None:
                history = [row for row in history if int(row["epoch"]) <= cutoff]
                start = EPOCHS + 1
                stopped_early = True
                # The legacy checkpoint contains the exact best model state;
                # normalize the resumable checkpoint to that state so a later
                # invocation cannot accidentally count epochs after cutoff.
                if best_state is None:
                    raise RuntimeError(f"legacy checkpoint has no best state: {latest}")
                model.load_state_dict(best_state, strict=True)
                atomic_torch_save(latest, {
                    "epoch": cutoff, "history": history, "best": best,
                    "best_epoch": best_epoch, "best_state": best_state,
                    "current_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "rng": module.rng_state(),
                    "bad_epochs": PATIENCE, "patience": PATIENCE,
                    "stopped_early": True, "invariants": invariants,
                })
        if stopped_early:
            start = EPOCHS + 1
        resumed = True

    if not resumed:
        set_seed(module, SEED + 100_000)
    train_indices = bundle.indices(fold["inner_train_subjects"], module.TASKS[task]["source_sessions"])
    started = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        batches = batch["episodes"][epoch - 1] if module.TASKS[task]["mi_protocol"] else module.task_epoch_batches(train_indices, task, fid, epoch)
        losses_on, losses_off, losses_kl, losses_total, residuals = [], [], [], [], []
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits_on, _ = model(value, channel_gate=True)
                residuals.append(model.last_abs)
                logits_off, _ = model(value, channel_gate=False)
                loss_on = F.cross_entropy(logits_on, labels, weight=weight)
                loss_off = F.cross_entropy(logits_off, labels, weight=weight)
                teacher = torch.softmax(logits_off.float() / TEMPERATURE, dim=1).detach()
                student_log = torch.log_softmax(logits_on.float() / TEMPERATURE, dim=1)
                loss_preserve = F.kl_div(student_log, teacher, reduction="batchmean") * (TEMPERATURE ** 2)
                loss = loss_on + ALPHA * loss_off + GAMMA * loss_preserve
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite BPXS loss {task}/f{fid}/epoch{epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses_on.append(float(loss_on.detach().cpu()))
            losses_off.append(float(loss_off.detach().cpu()))
            losses_kl.append(float(loss_preserve.detach().cpu()))
            losses_total.append(float(loss.detach().cpu()))

        gate_on = evaluate(module, model, bundle, cache, fold["inner_val_subjects"], mean, std, True)
        gate_off = evaluate(module, model, bundle, cache, fold["inner_val_subjects"], mean, std, False)
        val_ba = float(gate_on["BA"])
        chose = bool(epoch >= MIN_EPOCH and val_ba > best + TIE_TOL)
        if chose:
            best, best_epoch = val_ba, int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        elif epoch >= MIN_EPOCH:
            bad_epochs += 1
        stop_now = bool(epoch >= MIN_EPOCH and bad_epochs >= PATIENCE)
        row = {
            "task": task, "dataset": module.TASKS[task]["dataset"], "fold": fid, "epoch": epoch,
            "steps": len(batches), "L_on": float(np.mean(losses_on)), "L_off": float(np.mean(losses_off)),
            "L_preserve": float(np.mean(losses_kl)), "total_loss": float(np.mean(losses_total)),
            "gate_on_val_BA": gate_on["BA"], "gate_off_val_BA": gate_off["BA"],
            "gate_on_val_macro_F1": gate_on["macro_F1"], "gate_off_val_macro_F1": gate_off["macro_F1"],
            "gate_on_val_accuracy": gate_on["accuracy"], "gate_off_val_accuracy": gate_off["accuracy"],
            "BA_on_minus_off": gate_on["BA"] - gate_off["BA"],
            "lambda_channel": float(model.core.lambda_channel.detach().cpu()),
            "mean_abs_channel_residual": float(np.mean(residuals)) if residuals else 0.0,
            "rms_channel_residual": float(model.last_rms), "selected": chose,
            "eligible_for_selection": bool(epoch >= MIN_EPOCH), "bad_epochs": int(bad_epochs),
            "patience": PATIENCE, "stopped_early": stop_now,
        }
        history.append(row)
        atomic_torch_save(latest, {"epoch": epoch, "history": history, "best": best,
                                   "best_epoch": best_epoch, "best_state": best_state,
                                   "current_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "scaler": scaler.state_dict(), "rng": module.rng_state(),
                                   "bad_epochs": bad_epochs, "patience": PATIENCE,
                                   "stopped_early": stop_now,
                                   "invariants": invariants})
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[BPXS {task} f{fid}] epoch={epoch:02d} on={gate_on['BA']:.5f} off={gate_off['BA']:.5f} best={best:.5f} bad={bad_epochs}/{PATIENCE}", flush=True)
        if stop_now:
            stopped_early = True
            print(f"[BPXS {task} f{fid}] early-stop epoch={epoch:02d} best_epoch={best_epoch} best={best:.5f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"no BPXS checkpoint selected {task}/f{fid}")
    model.load_state_dict(best_state, strict=True)
    selected_row = history[next(index for index, row in enumerate(history) if int(row["epoch"]) == int(best_epoch))]
    atomic_torch_save(selected, {"state_dict": model.state_dict(), "selected_epoch": int(best_epoch),
                                  "invariants": invariants, "history": history})
    record = {
        "task": task, "dataset": module.TASKS[task]["dataset"], "fold": fid, "architecture": ARCHITECTURE,
        "seed": SEED, "invariants": invariants, "history": history, "selected": selected_row,
        "selected_epoch": int(best_epoch), "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": sha256_file(selected), "elapsed_seconds": time.perf_counter() - started,
        "references": references, "normalizer_sha256": norm_meta["mean_std_sha256"],
        "class_weight_meta": weight_meta, "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "patience": PATIENCE, "stopped_early": bool(stopped_early),
    }
    write_json(path, record)
    del model, cache, bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def fold_row(record: dict[str, Any]) -> dict[str, Any]:
    row = record["selected"]
    ref = record["references"]
    bpxs_ba = float(row["gate_on_val_BA"])
    return {
        "task": record["task"], "dataset": record["dataset"], "fold": record["fold"],
        "LiteBN_BA": ref["LiteBN_BA"], "LiteBN_macro_F1": ref["LiteBN_macro_F1"],
        "XS_BA": ref["XS_BA"], "XS_macro_F1": ref["XS_macro_F1"],
        "BPXS_BA": bpxs_ba, "BPXS_macro_F1": row["gate_on_val_macro_F1"],
        "BPXS_accuracy": row["gate_on_val_accuracy"],
        "delta_BA_vs_LiteBN": bpxs_ba - ref["LiteBN_BA"],
        "delta_BA_vs_LiteBN_pp": 100.0 * (bpxs_ba - ref["LiteBN_BA"]),
        "delta_BA_vs_XS": bpxs_ba - ref["XS_BA"],
        "delta_BA_vs_XS_pp": 100.0 * (bpxs_ba - ref["XS_BA"]),
        "selected_epoch": record["selected_epoch"],
        "gate_on_val_BA": row["gate_on_val_BA"], "gate_off_val_BA": row["gate_off_val_BA"],
        "BA_on_minus_off": row["BA_on_minus_off"], "lambda_channel": row["lambda_channel"],
        "mean_abs_channel_residual": row["mean_abs_channel_residual"],
        "rms_channel_residual": row["rms_channel_residual"],
        "selected_checkpoint": record["selected_checkpoint"],
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in frame.groupby("task", sort=True):
        rows.append({
            "task": task, "dataset": group["dataset"].iloc[0], "folds": int(len(group)),
            "LiteBN_BA": float(group["LiteBN_BA"].mean()), "XS_BA": float(group["XS_BA"].mean()),
            "BPXS_BA": float(group["BPXS_BA"].mean()),
            "mean_delta_vs_LiteBN_pp": float(group["delta_BA_vs_LiteBN_pp"].mean()),
            "mean_delta_vs_XS_pp": float(group["delta_BA_vs_XS_pp"].mean()),
            "positive_folds_vs_LiteBN": int((group["delta_BA_vs_LiteBN"] > TIE_TOL).sum()),
            "positive_folds_vs_XS": int((group["delta_BA_vs_XS"] > TIE_TOL).sum()),
            "zero_folds_vs_LiteBN": int((group["delta_BA_vs_LiteBN"].abs() <= TIE_TOL).sum()),
            "negative_folds_vs_LiteBN": int((group["delta_BA_vs_LiteBN"] < -TIE_TOL).sum()),
            "mean_gate_on_minus_off_BA": float(group["BA_on_minus_off"].mean()),
            "mean_lambda_channel": float(group["lambda_channel"].mean()),
            "mean_abs_channel_residual": float(group["mean_abs_channel_residual"].mean()),
        })
    return pd.DataFrame(rows)


def decide(summary: pd.DataFrame) -> str:
    erp = summary[summary.task == "OpenBMI_ERP"].iloc[0]
    wbcic = summary[summary.task == "WBCIC_MI"].iloc[0]
    erp_mean_ok = float(erp.mean_delta_vs_LiteBN_pp) >= -1e-9
    wbcic_mean_ok = float(wbcic.mean_delta_vs_LiteBN_pp) > 1e-9
    support_ok = int(erp.positive_folds_vs_LiteBN) >= 2 and int(wbcic.positive_folds_vs_LiteBN) >= 2
    if erp_mean_ok and wbcic_mean_ok and support_ok:
        return "BPXS_FIRST_SCREEN_PASS"
    if erp_mean_ok and not wbcic_mean_ok:
        return "BPXS_PRESERVATION_TOO_STRONG"
    if wbcic_mean_ok and not erp_mean_ok:
        return "BPXS_ERP_NOT_REPAIRED"
    return "BPXS_OBJECTIVE_NO_USEFUL_SIGNAL"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    module = load_module()
    search, folds, split_hash = module.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    references = reference_results(module)
    protocol = [
        "# BPXS first-screen protocol", "",
        "model: LiteBN_BPXS = exact original random-initialized LiteBN_XS architecture.",
        "Training objective: L_on + alpha*L_off + gamma*KL(stopgrad(p_off)||p_on).",
        f"seed: {SEED}; alpha: {ALPHA}; gamma: {GAMMA}; temperature: {TEMPERATURE}.",
        f"optimizer: AdamW lr={LR}; weight_decay={WEIGHT_DECAY}; clip={CLIP}; max_epochs={EPOCHS}; selection epochs >= {MIN_EPOCH}; patience={PATIENCE}.",
        "Gate-off disables only the XS channel residual correction; the scale gate, stem, mixers, embedding, classifier and dropout remain shared.",
        "Scope: OpenBMI ERP and WBCIC MI, five canonical inner splits each; no MI/SSVEP expansion.",
        f"canonical split sha256: `{split_hash}`", "CANONICAL_INNER_SPLITS_ONLY = YES",
        "OUTER_DEVELOPMENT_ACCESSED = NO", "FINAL_HELDOUT_ACCESSED = NO", "",
    ]
    write_text(PROTOCOL / "BPXS_FIRST_SCREEN_PROTOCOL.md", "\n".join(protocol))

    ordered = [("OpenBMI_ERP", folds["OpenBMI"][0]), ("WBCIC_MI", folds["WBCIC"][0])]
    for task in TASK_ORDER:
        for fold in folds[module.TASKS[task]["dataset"]]:
            if (task, int(fold["fold_id"])) not in {("OpenBMI_ERP", 0), ("WBCIC_MI", 0)}:
                ordered.append((task, fold))
    records = []
    for task, fold in ordered:
        key = (task, int(fold["fold_id"]))
        rec_path = record_path(*key)
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            if rec.get("invariants", {}).get("alpha") != ALPHA or rec.get("invariants", {}).get("gamma") != GAMMA:
                raise RuntimeError(f"existing BPXS record invariant mismatch: {rec_path}")
            print(f"[BPXS] reuse {task} fold{key[1]}", flush=True)
        else:
            rec = train_cell(module, task, fold, device, references[key])
        records.append(rec)

    if len(records) != 10:
        raise RuntimeError(f"incomplete BPXS cells: {len(records)}/10")
    fold_frame = pd.DataFrame([fold_row(record) for record in records]).sort_values(["task", "fold"]).reset_index(drop=True)
    trajectory = pd.DataFrame([row for record in records for row in record["history"]]).sort_values(["task", "fold", "epoch"])
    diagnostics = fold_frame[["task", "fold", "gate_on_val_BA", "gate_off_val_BA", "BA_on_minus_off", "lambda_channel", "mean_abs_channel_residual", "rms_channel_residual"]].copy()
    summary = summarize(fold_frame)
    terminal = decide(summary)
    write_csv(OUT / "BPXS_FOLD_RESULTS.csv", fold_frame)
    write_csv(OUT / "BPXS_TASK_SUMMARY.csv", summary)
    write_csv(OUT / "BPXS_TRAINING_TRAJECTORY.csv", trajectory)
    write_csv(OUT / "BPXS_GATE_DIAGNOSTICS.csv", diagnostics)
    metadata = {
        "experiment": "persist_eeg_base_preserving_xs_seed0_v1", "model": "LiteBN_BPXS",
        "tasks": list(TASK_ORDER), "folds": 10, "seed": SEED, "alpha": ALPHA, "gamma": GAMMA,
        "temperature": TEMPERATURE, "max_epochs": EPOCHS, "selection_min_epoch": MIN_EPOCH,
        "patience": PATIENCE, "canonical_split_sha256": split_hash,
        "terminal": terminal, "outer_development_accessed": False, "final_heldout_accessed": False,
        "parameter_count_by_task": {
            task: int(sum(p.numel() for p in BPXSModel(module, task).parameters()))
            for task in TASK_ORDER
        },
    }
    write_json(OUT / "BPXS_METADATA.json", metadata)
    report = [
        "# LiteBN-BPXS first-screen result", "",
        "Only OpenBMI ERP and WBCIC MI were run, five canonical inner folds each. No outer-development or final-heldout rows were accessed.", "",
        "| Task | LiteBN BA | Original XS BA | BPXS BA | Δ vs LiteBN pp | Δ vs XS pp | positive folds vs LiteBN |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(f"| {row.task} | {row.LiteBN_BA:.4f} | {row.XS_BA:.4f} | {row.BPXS_BA:.4f} | {row.mean_delta_vs_LiteBN_pp:+.3f} | {row.mean_delta_vs_XS_pp:+.3f} | {int(row.positive_folds_vs_LiteBN)}/5 |")
    report += ["", "## Gate diagnostics", "", "| Task | mean gate-on BA | mean gate-off BA | mean BA(on-off) |", "|---|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        report.append(f"| {row.task} | {row.BPXS_BA:.4f} | {row.BPXS_BA - row.mean_gate_on_minus_off_BA:.4f} | {row.mean_gate_on_minus_off_BA:+.4f} |")
    report += ["", f"Terminal: **{terminal}**", "", "`SEED = 0`", "`CANONICAL_INNER_SPLITS_ONLY = YES`", "`OUTER_DEVELOPMENT_ACCESSED = NO`", "`FINAL_HELDOUT_ACCESSED = NO`", ""]
    write_text(OUT / "BPXS_RESULTS.md", "\n".join(report))
    print(f"BPXS_COMPLETE terminal={terminal}", flush=True)
    print("OUTER_DEVELOPMENT_ACCESSED = NO", flush=True)
    print("FINAL_HELDOUT_ACCESSED = NO", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
