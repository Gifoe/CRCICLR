#!/usr/bin/env python3
"""Seed-0 X-initialized residual-adapter canonical-inner experiment.

Stage A runs only G1 on each frozen task/fold canonical inner train/validation
pair. No additional resplits are created. Epoch zero is the exact LiteBN-X
function; early stopping uses inner-validation BA with patience five. This
run stops after the 20 G1 inner runs and never opens outer-development.
Stage B remains a separate guarded entry point and is not invoked here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")
BASE_EXP = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1"
EXP = REPO / "experiments/persist_eeg_xinit_residual_adapter_seed0_v1"
OUT = EXP / "outputs_single_inner_g1"
PROTOCOL = EXP / "protocol"
RUNTIME = Path("/root/rivermind-data/xinit_residual_adapter_seed0_single_inner_runtime")
SOURCE_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
HIST_CARRIER = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")
HIST_TASK = Path("/root/rivermind-data/openbmi_task_generality_runtime")

SEED = 0
MAX_EPOCHS = 20
PATIENCE = 5
LR = 1e-4
WEIGHT_DECAY = 5e-4
BETA = 1e-4
TAU = 0.5
SCALE_ANCHOR = 1e-3
MIXER_ANCHOR = 1e-3
CLIP = 5.0
BOOTSTRAPS = 10_000
EPS = 1e-8
TOL = 1e-6


@dataclass(frozen=True)
class Regime:
    name: str
    train_scale: bool
    train_last_mixer: bool
    simplicity_rank: int

    @property
    def trainable(self) -> list[str]:
        items = ["residual_adapter.lambda_channel", "residual_adapter.channel_mlp"]
        if self.train_scale:
            items += ["X.scale_mlp", "X.lambda_scale"]
        if self.train_last_mixer:
            items += ["X.mixer[-1]"]
        return items


G0 = Regime("XRA_G0_ADAPTER_ONLY", False, False, 1)
G1 = Regime("XRA_G1_ADAPTER_SCALE", True, False, 2)
G2 = Regime("XRA_G2_ADAPTER_SCALE_LAST_MIXER", True, True, 3)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    stream = io.BytesIO()
    torch.save(state, stream)
    return sha_bytes(stream.getvalue())


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
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


def torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    output: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        output["cuda"] = torch.cuda.get_rng_state_all()
    return output


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].detach().cpu())
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in value["cuda"]])


def load_module():
    source = BASE_EXP / "code/litebn_x.py"
    spec = importlib.util.spec_from_file_location("xra_base_impl", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen LiteBN-X implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEED = SEED
    return module


def base_x_path(task: str, fold: int) -> Path:
    return SOURCE_RUNTIME / "checkpoints" / task.lower() / f"fold{fold}_litebn_x" / "selected_best.pt"


def baseline_path(task: str, fold: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        prefix = "erp" if task == "OpenBMI_ERP" else "ssvep"
        return HIST_TASK / f"{prefix}_fold{fold}_seed0_litebn" / "selected_best.pt"
    prefix = "openbmi" if task == "OpenBMI_MI" else "wbcic"
    return HIST_CARRIER / f"{prefix}_fold{fold}_seed0_litebn" / "selected_best.pt"


def frozen_normalizer_path(task: str, fold: int) -> Path:
    return SOURCE_RUNTIME / "normalizers" / f"{task.lower()}_fold{fold}.npz"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


class XResidualAdapter(nn.Module):
    """One X model plus a bounded, zero-effect shared channel residual path."""

    def __init__(self, base: nn.Module, tau: float = TAU):
        super().__init__()
        self.base = base
        self.channel_mlp = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
        self.lambda_channel = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.tau = float(tau)
        self.force_adapter_off = False

    def correction(self, value: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(value.square().mean(dim=-1) + EPS)
        diff = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + EPS)
        raw = self.channel_mlp(torch.stack((rms, diff), dim=-1)).squeeze(-1)
        amplitude = self.tau * torch.tanh(self.lambda_channel)
        return amplitude * torch.tanh(raw)

    def adapted(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.force_adapter_off or (not self.training and float(self.lambda_channel.detach().cpu()) == 0.0):
            return value, torch.zeros_like(value[..., 0])
        correction = self.correction(value)
        return value * (1.0 + correction).unsqueeze(-1), correction

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        corrected, _ = self.adapted(value)
        return self.base(corrected)


def build_xra(mod, task: str, checkpoint: Path, device: torch.device) -> tuple[XResidualAdapter, dict[str, torch.Tensor]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen X checkpoint missing: {checkpoint}")
    base = mod.build_model("LiteBN_X", task).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    base.load_state_dict(state, strict=True)
    raw = copy.deepcopy(base.state_dict())
    return XResidualAdapter(base).to(device), raw


def mean_metrics(per_subject: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in per_subject.values()])) for key in ("BA", "macro_F1", "accuracy")}


def subject_logits(mod, model: nn.Module, bundle, cache, subjects: Iterable[str], mean: np.ndarray, std: np.ndarray) -> dict[str, dict[str, Any]]:
    model.eval()
    output: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for subject in mod.subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], (int(mod.TASKS[bundle.task]["future_session"]),))
            labels = bundle.labels(indices)
            fragments = []
            for start in range(0, len(indices), 128):
                value, _ = cache.batch(indices[start:start + 128], mean, std)
                fragments.append(model(value)[0].float().cpu().numpy())
            logits = np.concatenate(fragments, axis=0)
            output[str(subject)] = {**mod.classification_metrics(labels, logits), "labels": labels, "logits": logits, "trials": int(len(labels))}
    return output


def epoch0_replay(mod, folds: dict[str, list[dict[str, Any]]], device: torch.device, tasks: Iterable[str] | None = None) -> pd.DataFrame:
    rows = []
    task_order = tuple(mod.TASK_ORDER if tasks is None else tasks)
    for task in task_order:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            source = base_x_path(task, fold_id)
            normalizer = frozen_normalizer_path(task, fold_id)
            bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
            mean, std, norm_meta = mod.load_tensor_pair(normalizer)
            cache = mod.RawGPUCache(bundle, device)
            x = mod.build_model("LiteBN_X", task).to(device)
            x.load_state_dict(torch.load(source, map_location=device, weights_only=False), strict=True)
            xra, _ = build_xra(mod, task, source, device)
            x_rows = subject_logits(mod, x, bundle, cache, fold["inner_val_subjects"], mean, std)
            xra_rows = subject_logits(mod, xra, bundle, cache, fold["inner_val_subjects"], mean, std)
            difference, mismatch = 0.0, 0
            for subject in x_rows:
                difference = max(difference, float(np.max(np.abs(x_rows[subject]["logits"] - xra_rows[subject]["logits"]))))
                mismatch += int(np.sum(x_rows[subject]["logits"].argmax(1) != xra_rows[subject]["logits"].argmax(1)))
            x_metric, xra_metric = mean_metrics(x_rows), mean_metrics(xra_rows)
            passed = bool(difference < TOL and mismatch == 0 and all(abs(x_metric[key] - xra_metric[key]) < 1e-12 for key in x_metric))
            rows.append({
                "task": task, "fold": fold_id, "X_checkpoint": str(source), "X_checkpoint_sha256": sha_file(source),
                "XRA_init_state_sha256": state_sha(xra.state_dict()), "normalizer_sha256": norm_meta["mean_std_sha256"],
                "max_abs_logit_difference": difference, "prediction_mismatch_count": mismatch,
                "X_BA": x_metric["BA"], "XRA_epoch0_BA": xra_metric["BA"],
                "X_macro_F1": x_metric["macro_F1"], "XRA_epoch0_macro_F1": xra_metric["macro_F1"],
                "X_accuracy": x_metric["accuracy"], "XRA_epoch0_accuracy": xra_metric["accuracy"], "pass": passed,
            })
            del x, xra, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["task", "fold"]).reset_index(drop=True)
    write_csv(OUT / "EPOCH0_EXACT_X_REPLAY.csv", frame)
    expected = sum(len(folds[mod.TASKS[task]["dataset"]]) for task in task_order)
    if len(frame) != expected or not bool(frame["pass"].all()):
        raise RuntimeError("epoch-0 exact-X replay invariant failed")
    print(f"EPOCH0_EXACT_X_REPLAY_PASS {expected}/{expected}", flush=True)
    return frame


def build_inner_splits(mod, task: str, fold: dict[str, Any], bundle) -> list[dict[str, Any]]:
    """Return exactly the historical canonical inner train/validation pair."""
    training = mod.subject_sort(list(fold["inner_train_subjects"]), bundle.name)
    validation = mod.subject_sort(list(fold["inner_val_subjects"]), bundle.name)
    outer = set(fold["outer_dev_subjects"])
    if set(training) & set(validation):
        raise RuntimeError(f"canonical inner partition overlaps in {task}/fold{fold['fold_id']}")
    if set(training) | set(validation) != set(fold["inner_train_subjects"]) | set(fold["inner_val_subjects"]):
        raise RuntimeError(f"canonical inner partition changed in {task}/fold{fold['fold_id']}")
    if (set(training) | set(validation)) & outer:
        raise RuntimeError(f"outer-development leakage in {task}/fold{fold['fold_id']}")
    mean, std, metadata = mod.normalizer(bundle, training)
    fold_id = int(fold["fold_id"])
    path = RUNTIME / "inner_normalizers" / task.lower() / f"fold{fold_id}.npz"
    mod.save_tensor_pair(path, mean, std, {**metadata, "inner_split_policy": "canonical_frozen", "outer_subjects_absent": True})
    canonical_seed = int(fold.get("inner_split_seed", fold.get("fold_seed", 0)))
    return [{
        "split": 0, "base_seed": canonical_seed, "effective_seed": canonical_seed,
        "train_subjects": training, "val_subjects": validation, "normalizer_path": str(path),
        "normalizer_sha256": metadata["mean_std_sha256"], "normalizer_mean": mean, "normalizer_std": std,
    }]


def residual_energy(model: XResidualAdapter, value: torch.Tensor) -> torch.Tensor:
    corrected, _ = model.adapted(value)
    return ((corrected - value).square() / (value.square() + EPS)).mean()


def configure_trainable(model: XResidualAdapter, regime: Regime) -> tuple[dict[str, torch.Tensor], torch.optim.Optimizer]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.lambda_channel.requires_grad_(True)
    for parameter in model.channel_mlp.parameters():
        parameter.requires_grad_(True)
    groups = [{"params": [model.lambda_channel, *model.channel_mlp.parameters()], "lr": LR}]
    if regime.train_scale:
        model.base.lambda_scale.requires_grad_(True)
        for parameter in model.base.scale_mlp.parameters():
            parameter.requires_grad_(True)
        groups.append({"params": [model.base.lambda_scale, *model.base.scale_mlp.parameters()], "lr": LR})
    if regime.train_last_mixer:
        for parameter in model.base.mixer[-1].parameters():
            parameter.requires_grad_(True)
        groups.append({"params": list(model.base.mixer[-1].parameters()), "lr": LR})
    anchor = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith("base.")}
    return anchor, torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def keep_base_batch_norms_frozen(model: XResidualAdapter) -> None:
    """BatchNorm buffers belong to the frozen X function in every regime."""
    for module in model.base.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def anchor_loss(model: XResidualAdapter, anchor: dict[str, torch.Tensor], regime: Regime) -> torch.Tensor:
    terms = []
    for name, parameter in model.named_parameters():
        if name in anchor:
            coefficient = MIXER_ANCHOR if name.startswith("base.mixer.2.") else SCALE_ANCHOR
            terms.append(coefficient * (parameter - anchor[name]).square().mean())
    return torch.stack(terms).sum() if terms else torch.zeros((), device=next(model.parameters()).device)


def train_batches(indices: np.ndarray, task: str, fold: int, split: int, epoch: int) -> list[np.ndarray]:
    task_code = {"OpenBMI_MI": 17, "OpenBMI_ERP": 29, "OpenBMI_SSVEP": 43, "WBCIC_MI": 59}[task]
    generator = np.random.default_rng(SEED + task_code * 1_000_003 + fold * 10_009 + split * 101 + epoch)
    shuffled = generator.permutation(indices)
    return [shuffled[start:start + 64] for start in range(0, len(shuffled), 64)]


def split_cell_path(regime: Regime, task: str, fold: int, split: int) -> Path:
    return RUNTIME / "stage_a" / regime.name.lower() / task.lower() / f"fold{fold}" / f"split{split}"


def train_one_split(mod, regime: Regime, task: str, fold: dict[str, Any], split: dict[str, Any], source_sha: str, device: torch.device) -> dict[str, Any]:
    fold_id, split_id = int(fold["fold_id"]), int(split["split"])
    directory = split_cell_path(regime, task, fold_id, split_id)
    record_path, latest_path, best_path = directory / "record.json", directory / "latest.pt", directory / "best.pt"
    source = base_x_path(task, fold_id)
    invariant = {
        "regime": regime.name, "task": task, "fold": fold_id, "split": split_id, "seed": SEED,
        "source_x_sha256": source_sha, "split_seed": split["effective_seed"],
        "train_subjects": split["train_subjects"], "val_subjects": split["val_subjects"],
        "normalizer_sha256": split["normalizer_sha256"], "lr": LR, "weight_decay": WEIGHT_DECAY,
        "beta": BETA, "tau": TAU, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "scale_anchor": SCALE_ANCHOR, "mixer_anchor": MIXER_ANCHOR,
        "inner_split_policy": "canonical_frozen",
    }
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("invariant") == invariant:
            return previous
        raise RuntimeError(f"completed inner cell invariant mismatch: {record_path}")
    bundle = mod.build_bundle(task, split["train_subjects"] + split["val_subjects"])
    cache = mod.RawGPUCache(bundle, device)
    mean, std = split["normalizer_mean"], split["normalizer_std"]
    source_indices = bundle.indices(split["train_subjects"], mod.TASKS[task]["source_sessions"])
    class_weight, _ = mod.class_weights(bundle, split["train_subjects"])
    class_weight = None if class_weight is None else class_weight.to(device)
    set_seed(SEED + fold_id * 100 + split_id)
    model, _ = build_xra(mod, task, source, device)
    anchor, optimizer = configure_trainable(model, regime)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_ba = float("-inf")
    bad_epochs = 0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved.get("invariant") != invariant:
            raise RuntimeError(f"resume invariant mismatch: {latest_path}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        history, start_epoch = list(saved["history"]), int(saved["epoch"]) + 1
        best_epoch = int(saved.get("best_epoch", max(history, key=lambda row: row["BA"])["epoch"]))
        best_ba = float(saved.get("best_ba", max(row["BA"] for row in history)))
        bad_epochs = int(saved.get("bad_epochs", 0))
    if not history:
        epoch0 = mean_metrics(mod.evaluate(model, bundle, cache, split["val_subjects"], mean, std))
        history.append({"epoch": 0, **epoch0, "ce": None, "residual_energy": 0.0, "anchor": 0.0, "best_epoch": 0, "best_ba": float(epoch0["BA"]), "bad_epochs": 0})
        best_ba = float(epoch0["BA"])
        best_epoch = 0
        torch_save(best_path, {"invariant": invariant, "epoch": 0, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "history": history, "best_epoch": best_epoch, "best_ba": best_ba, "bad_epochs": bad_epochs})
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        keep_base_batch_norms_frozen(model)
        ce_values, residual_values, anchor_values, fp32_recoveries = [], [], [], 0
        for indices in train_batches(source_indices, task, fold_id, split_id, epoch):
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            pre_forward_rng = rng_state()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(value)
                ce = F.cross_entropy(logits, labels, weight=class_weight)
                energy = residual_energy(model, value)
                anchor_term = anchor_loss(model, anchor, regime)
                loss = ce + BETA * energy + anchor_term
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                restore_rng(pre_forward_rng)
                with torch.autocast(device_type=device.type, enabled=False):
                    logits, _ = model(value)
                    ce = F.cross_entropy(logits, labels, weight=class_weight)
                    energy = residual_energy(model, value)
                    anchor_term = anchor_loss(model, anchor, regime)
                    loss = ce + BETA * energy + anchor_term
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss after FP32 recovery: {regime.name}/{task}/f{fold_id}/s{split_id}")
                fp32_recoveries += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], CLIP)
            scaler.step(optimizer)
            scaler.update()
            ce_values.append(float(ce.detach().float().cpu()))
            residual_values.append(float(energy.detach().float().cpu()))
            anchor_values.append(float(anchor_term.detach().float().cpu()))
        validation = mean_metrics(mod.evaluate(model, bundle, cache, split["val_subjects"], mean, std))
        improved = bool(validation["BA"] > best_ba + TOL)
        if improved:
            best_ba = float(validation["BA"])
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        snapshot = {"epoch": epoch, **validation, "ce": float(np.mean(ce_values)), "residual_energy": float(np.mean(residual_values)), "anchor": float(np.mean(anchor_values)), "fp32_recovery_batches": fp32_recoveries, "lambda_channel": float(model.lambda_channel.detach().cpu()), "effective_amplitude": float(TAU * torch.tanh(model.lambda_channel).detach().cpu()), "best_epoch": best_epoch, "best_ba": best_ba, "bad_epochs": bad_epochs}
        history.append(snapshot)
        state = {"invariant": invariant, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "history": history, "best_epoch": best_epoch, "best_ba": best_ba, "bad_epochs": bad_epochs}
        torch_save(latest_path, state)
        if improved:
            torch_save(best_path, state)
        print(f"[{regime.name} {task} f{fold_id} s{split_id}] epoch={epoch:02d} valBA={validation['BA']:.5f} best_epoch={best_epoch:02d} bestBA={best_ba:.5f} bad={bad_epochs}/{PATIENCE}", flush=True)
        if bad_epochs >= PATIENCE:
            print(f"[{regime.name} {task} f{fold_id} s{split_id}] EARLY_STOP epoch={epoch:02d} best_epoch={best_epoch:02d}", flush=True)
            break
    record = {"invariant": invariant, "regime": regime.name, "task": task, "dataset": mod.TASKS[task]["dataset"], "fold": fold_id, "split": split_id, "history": history, "best_epoch": int(best_epoch), "best_inner_val_BA": float(best_ba), "best_checkpoint": str(best_path), "source_x_checkpoint": str(source), "source_x_sha256": source_sha, "normalizer_sha256": split["normalizer_sha256"], "train_subjects": split["train_subjects"], "val_subjects": split["val_subjects"], "inner_split_policy": "canonical_frozen"}
    write_json(record_path, record)
    del model, cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def select_fold_checkpoint(records: list[dict[str, Any]], regime: Regime, task: str, fold: int) -> dict[str, Any]:
    group = [row for row in records if row["regime"] == regime.name and row["task"] == task and int(row["fold"]) == fold]
    if len(group) != 1:
        raise RuntimeError(f"expected exactly one canonical inner trajectory for {regime.name}/{task}/f{fold}")
    row = group[0]
    epoch0 = row["history"][0]
    selected_epoch = int(row["best_epoch"])
    selected = next(item for item in row["history"] if int(item["epoch"]) == selected_epoch)
    delta = float(selected["BA"] - epoch0["BA"])
    status = "positive" if delta > TOL else ("zero" if abs(delta) <= TOL else "negative")
    return {"regime": regime.name, "task": task, "fold": fold, "selected_epoch": selected_epoch, "epoch0_inner_val_BA": float(epoch0["BA"]), "selected_inner_val_BA": float(selected["BA"]), "selected_delta_BA": delta, "selected_delta_BA_pp": 100.0 * delta, "status": status, "early_stop_epoch": int(row["history"][-1]["epoch"]), "patience": PATIENCE, "selected_checkpoint": row["best_checkpoint"], "selected_eligible": bool(delta >= -TOL)}


def aggregate_regime(selections: list[dict[str, Any]], regime: Regime) -> dict[str, Any]:
    subset = [row for row in selections if row["regime"] == regime.name]
    if len(subset) != 20:
        raise RuntimeError(f"incomplete selections for {regime.name}")
    tasks = sorted({row["task"] for row in subset})
    task_means = {task: float(np.mean([row["selected_delta_BA"] for row in subset if row["task"] == task])) for task in tasks}
    openbmi = [task_means[task] for task in tasks if task.startswith("OpenBMI_")]
    wbcic = float(task_means.get("WBCIC_MI", float("nan")))
    return {"regime": regime.name, "trainable_modules": ";".join(regime.trainable), "inner_split_policy": "canonical_frozen", "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "positive_folds": int(sum(row["status"] == "positive" for row in subset)), "zero_folds": int(sum(row["status"] == "zero" for row in subset)), "negative_folds": int(sum(row["status"] == "negative" for row in subset)), "openbmi_nonnegative": bool(all(value >= -TOL for value in openbmi)), "wbcic_mean_delta_pp": 100.0 * wbcic, "passes_target": bool(all(value >= -TOL for value in openbmi) and wbcic > TOL), "outer_development_opened": False, "final_heldout_accessed": False, **{f"{task}_mean_delta_pp": 100.0 * value for task, value in task_means.items()}}


def stage_a_protocol(split_hash: str) -> str:
    return "\n".join([
        "# X-init residual adapter seed-0 — canonical-inner G1 protocol", "",
        "Stage A uses exactly each frozen fold's historical inner_train_subjects and inner_val_subjects. No additional resplits are created.",
        "Only XRA_G1_ADAPTER_SCALE is run in this amendment. G0/G2 are not run and no outer-development is opened.", "",
        f"Epoch 0 is the exact LiteBN-X checkpoint; tau={TAU}; adapter lambda_channel=0 at initialization.",
        f"Trainable modules: {', '.join(G1.trainable)}. All remaining LiteBN-X parameters and BatchNorm buffers are frozen.",
        f"AdamW lr={LR}, weight_decay={WEIGHT_DECAY}, beta={BETA}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}, selection metric=inner-val BA.",
        "Best checkpoint is saved after each strict BA improvement. Training stops after five consecutive non-improving epochs. If no epoch improves over epoch 0, epoch 0 is selected.",
        "Target summary: OpenBMI MI/ERP/SSVEP non-negative versus X; WBCIC MI positive mean delta. This run stops after the 20 G1 inner runs for inspection.",
        f"Frozen source code SHA256: {sha_file(Path(__file__))}", f"Frozen five-fold split SHA256: {split_hash}", "FINAL_HELDOUT_ACCESSED = NO", ""
    ])


def run_stage_a() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    _, folds, split_hash = mod.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (PROTOCOL / "EXPERIMENT_PROTOCOL_G1_SINGLE_CANONICAL.md").write_text(stage_a_protocol(split_hash), encoding="utf-8")
    replay = epoch0_replay(mod, folds, device)
    source_sha = {(row.task, int(row.fold)): str(row.X_checkpoint_sha256) for row in replay.itertuples(index=False)}
    all_records, all_selections = [], []
    regime = G1
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
            splits = build_inner_splits(mod, task, fold, bundle)
            if len(splits) != 1 or splits[0]["split"] != 0:
                raise RuntimeError("canonical-inner policy produced more than one split")
            all_records.append(train_one_split(mod, regime, task, fold, splits[0], source_sha[(task, fold_id)], device))
            all_selections.append(select_fold_checkpoint(all_records, regime, task, fold_id))
            del bundle
    summary = aggregate_regime(all_selections, regime)
    history_rows = []
    for row in all_records:
        for item in row["history"]:
            history_rows.append({"regime": row["regime"], "task": row["task"], "dataset": row["dataset"], "fold": row["fold"], "split": row["split"], "epoch": item["epoch"], "BA": item["BA"], "macro_F1": item["macro_F1"], "accuracy": item["accuracy"], "best_epoch": row["best_epoch"], "early_stop_epoch": row["history"][-1]["epoch"]})
    write_csv(OUT / "G1_CANONICAL_INNER_HISTORY.csv", pd.DataFrame(history_rows))
    write_csv(OUT / "G1_CANONICAL_INNER_FOLD_RESULTS.csv", pd.DataFrame(all_selections).sort_values(["task", "fold"]).reset_index(drop=True))
    task_rows = []
    for task in sorted({row["task"] for row in all_selections}):
        subset = [row for row in all_selections if row["task"] == task]
        deltas = np.asarray([row["selected_delta_BA"] for row in subset], dtype=float)
        task_rows.append({"task": task, "folds": len(subset), "task_mean_delta_pp": 100.0 * float(np.mean(deltas)), "positive_folds": int(np.sum(deltas > TOL)), "zero_folds": int(np.sum(np.abs(deltas) <= TOL)), "negative_folds": int(np.sum(deltas < -TOL))})
    write_csv(OUT / "G1_CANONICAL_INNER_TASK_SUMMARY.csv", pd.DataFrame(task_rows))
    write_json(OUT / "G1_CANONICAL_INNER_METADATA.json", {"experiment": "persist_eeg_xinit_residual_adapter_seed0_v1", "regime": regime.name, "summary": summary, "final_heldout_accessed": False, "outer_development_opened": False, "terminal": "G1_CANONICAL_INNER_COMPLETE"})
    report = ["# LiteBN-XRA seed0 — G1 canonical-inner result", "", "- Regime: `XRA_G1_ADAPTER_SCALE` only.", "- Runs: 4 tasks × 5 folds × 1 frozen canonical inner train/validation pair = 20 runs.", f"- Early stopping: max {MAX_EPOCHS} epochs, patience {PATIENCE}, selection metric inner-val BA.", "- Epoch 0 is exact LiteBN-X; if no improvement, epoch 0 is selected.", "- G2 and outer-development were not run.", "", "| Task | Mean delta vs X (pp) | Positive | Zero | Negative |", "|---|---:|---:|---:|---:|"]
    for row in task_rows:
        report.append(f"| {row['task']} | {row['task_mean_delta_pp']:+.4f} | {row['positive_folds']}/5 | {row['zero_folds']}/5 | {row['negative_folds']}/5 |")
    report += ["", f"- WBCIC MI mean delta: **{summary['wbcic_mean_delta_pp']:+.4f} pp**", f"- OpenBMI non-negative target: `{summary['openbmi_nonnegative']}`", f"- WBCIC positive target: `{summary['wbcic_mean_delta_pp'] > 0.0}`", "- This run stops here for inspection; no Stage-A lock is issued.", "", "`FINAL_HELDOUT_ACCESSED = NO`", ""]
    (OUT / "G1_CANONICAL_INNER_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"G1_CANONICAL_INNER_COMPLETE wbcic_mean={summary['wbcic_mean_delta_pp']:+.4f}pp openbmi_nonnegative={summary['openbmi_nonnegative']}", flush=True)
    return 0


def run_stage_g2_wbcic() -> int:
    """Run only G2 on WBCIC's five canonical inner folds."""
    global OUT, RUNTIME, PROTOCOL
    OUT = EXP / "outputs_single_inner_g2_wbcic"
    RUNTIME = Path("/root/rivermind-data/xinit_residual_adapter_seed0_g2_wbcic_runtime")
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    _, folds, split_hash = mod.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = "\n".join([
        "# X-init residual adapter seed-0 — canonical-inner G2 WBCIC-only protocol", "",
        "Only WBCIC_MI is evaluated: five frozen outer folds, one canonical inner train/validation pair per fold.",
        "No additional resplits are created. OpenBMI, G1, G0, G2 follow-up, outer-development and final held-out evaluation are not run.", "",
        f"Epoch 0 is the exact LiteBN-X checkpoint; tau={TAU}; adapter lambda_channel=0 at initialization.",
        f"Trainable modules: {', '.join(G2.trainable)}. All remaining LiteBN-X parameters and BatchNorm buffers are frozen.",
        f"AdamW lr={LR}, weight_decay={WEIGHT_DECAY}, beta={BETA}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}, selection metric=inner-val BA.",
        "Best checkpoint is saved after each strict BA improvement. Training stops after five consecutive non-improving epochs. If no epoch improves over epoch 0, epoch 0 is selected.",
        f"Frozen source code SHA256: {sha_file(Path(__file__))}", f"Frozen five-fold split SHA256: {split_hash}", "FINAL_HELDOUT_ACCESSED = NO", "",
    ])
    (PROTOCOL / "EXPERIMENT_PROTOCOL_G2_WBCIC_SINGLE_CANONICAL.md").write_text(protocol, encoding="utf-8")
    replay = epoch0_replay(mod, folds, device, tasks=("WBCIC_MI",))
    source_sha = {(row.task, int(row.fold)): str(row.X_checkpoint_sha256) for row in replay.itertuples(index=False)}
    all_records, all_selections = [], []
    regime = G2
    task = "WBCIC_MI"
    for fold in folds[mod.TASKS[task]["dataset"]]:
        fold_id = int(fold["fold_id"])
        bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
        splits = build_inner_splits(mod, task, fold, bundle)
        if len(splits) != 1 or splits[0]["split"] != 0:
            raise RuntimeError("canonical-inner policy produced more than one split")
        all_records.append(train_one_split(mod, regime, task, fold, splits[0], source_sha[(task, fold_id)], device))
        all_selections.append(select_fold_checkpoint(all_records, regime, task, fold_id))
        del bundle
    if len(all_selections) != 5:
        raise RuntimeError("incomplete WBCIC G2 canonical-inner run")
    deltas = np.asarray([row["selected_delta_BA"] for row in all_selections], dtype=float)
    summary = {
        "regime": regime.name, "task": task, "inner_split_policy": "canonical_frozen",
        "folds": 5, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "task_mean_delta_pp": 100.0 * float(np.mean(deltas)),
        "positive_folds": int(np.sum(deltas > TOL)), "zero_folds": int(np.sum(np.abs(deltas) <= TOL)),
        "negative_folds": int(np.sum(deltas < -TOL)),
        "outer_development_opened": False, "final_heldout_accessed": False,
    }
    history_rows = []
    for row in all_records:
        for item in row["history"]:
            history_rows.append({"regime": row["regime"], "task": row["task"], "dataset": row["dataset"], "fold": row["fold"], "split": row["split"], "epoch": item["epoch"], "BA": item["BA"], "macro_F1": item["macro_F1"], "accuracy": item["accuracy"], "best_epoch": row["best_epoch"], "early_stop_epoch": row["history"][-1]["epoch"]})
    write_csv(OUT / "G2_WBCIC_CANONICAL_INNER_HISTORY.csv", pd.DataFrame(history_rows))
    write_csv(OUT / "G2_WBCIC_CANONICAL_INNER_FOLD_RESULTS.csv", pd.DataFrame(all_selections).sort_values(["task", "fold"]).reset_index(drop=True))
    write_csv(OUT / "G2_WBCIC_CANONICAL_INNER_TASK_SUMMARY.csv", pd.DataFrame([summary]))
    write_json(OUT / "G2_WBCIC_CANONICAL_INNER_METADATA.json", {"experiment": "persist_eeg_xinit_residual_adapter_seed0_v1", "summary": summary, "final_heldout_accessed": False, "outer_development_opened": False, "terminal": "G2_WBCIC_CANONICAL_INNER_COMPLETE"})
    report = ["# LiteBN-XRA seed0 — G2 WBCIC canonical-inner result", "", "- Regime: `XRA_G2_ADAPTER_SCALE_LAST_MIXER` only.", "- Scope: WBCIC MI × 5 folds × 1 frozen canonical inner train/validation pair.", f"- Early stopping: max {MAX_EPOCHS} epochs, patience {PATIENCE}, selection metric inner-val BA.", "- Epoch 0 is exact LiteBN-X; if no improvement, epoch 0 is selected.", "- G1/G0, OpenBMI and outer-development were not run.", "", "| Task | Mean delta vs X (pp) | Positive | Zero | Negative |", "|---|---:|---:|---:|---:|"]
    report.append(f"| WBCIC_MI | {summary['task_mean_delta_pp']:+.4f} | {summary['positive_folds']}/5 | {summary['zero_folds']}/5 | {summary['negative_folds']}/5 |")
    report += ["", "Selected epoch per fold: " + ", ".join(str(row["selected_epoch"]) for row in all_selections), "", "This run stops here for inspection; no Stage-A lock is issued.", "", "`FINAL_HELDOUT_ACCESSED = NO`", ""]
    (OUT / "G2_WBCIC_CANONICAL_INNER_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"G2_WBCIC_CANONICAL_INNER_COMPLETE mean={summary['task_mean_delta_pp']:+.4f}pp", flush=True)
    return 0


def selected_regime(name: str) -> Regime:
    return {item.name: item for item in (G0, G1, G2)}[name]


def final_checkpoint_path(regime: Regime, task: str, fold: int) -> Path:
    return RUNTIME / "stage_b" / regime.name.lower() / task.lower() / f"fold{fold}" / "selected.pt"


def train_final(mod, regime: Regime, task: str, fold: dict[str, Any], epoch_count: int, device: torch.device) -> XResidualAdapter:
    fold_id = int(fold["fold_id"])
    path = final_checkpoint_path(regime, task, fold_id)
    source = base_x_path(task, fold_id)
    set_seed(SEED + 9_001 + fold_id)
    model, _ = build_xra(mod, task, source, device)
    if epoch_count == 0:
        torch_save(path, model.state_dict())
        return model
    bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
    mean, std, _ = mod.load_tensor_pair(frozen_normalizer_path(task, fold_id))
    cache = mod.RawGPUCache(bundle, device)
    all_train = mod.subject_sort(fold["inner_train_subjects"] + fold["inner_val_subjects"], bundle.name)
    train_indices = bundle.indices(all_train, mod.TASKS[task]["source_sessions"])
    class_weight, _ = mod.class_weights(bundle, all_train)
    class_weight = None if class_weight is None else class_weight.to(device)
    anchor, optimizer = configure_trainable(model, regime)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, epoch_count + 1):
        model.train()
        keep_base_batch_norms_frozen(model)
        for indices in train_batches(train_indices, task, fold_id, 99, epoch):
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(value)
                loss = F.cross_entropy(logits, labels, weight=class_weight) + BETA * residual_energy(model, value) + anchor_loss(model, anchor, regime)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite final training loss: {regime.name}/{task}/f{fold_id}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], CLIP)
            scaler.step(optimizer)
            scaler.update()
    torch_save(path, model.state_dict())
    del cache
    return model


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025) * 100.0), float(np.quantile(samples, 0.975) * 100.0)


def correction_diagnostics(model: XResidualAdapter, cache, bundle, subjects, mean, std, mod) -> dict[str, float]:
    indices = bundle.indices(subjects, (int(mod.TASKS[bundle.task]["future_session"]),))
    per_value, per_channel = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            value, _ = cache.batch(indices[start:start + 128], mean, std)
            correction = model.correction(value).abs().float().cpu().numpy()
            per_value.append(correction.reshape(-1))
            per_channel.append(correction.mean(axis=0))
    values, channels = np.concatenate(per_value), np.concatenate(per_channel)
    amplitude = float(TAU * torch.tanh(model.lambda_channel).detach().cpu())
    return {"lambda_channel": float(model.lambda_channel.detach().cpu()), "tanh_lambda_channel": float(torch.tanh(model.lambda_channel).detach().cpu()), "effective_amplitude": amplitude, "mean_abs_channel_correction": float(values.mean()), "median_abs_channel_correction": float(np.median(values)), "p95_abs_channel_correction": float(np.quantile(values, .95)), "max_abs_channel_correction": float(values.max()), "channel_mean_abs_min": float(channels.min()), "channel_mean_abs_std": float(channels.std()), "channel_mean_abs_max": float(channels.max())}


def functional_drift(x: nn.Module, xra: XResidualAdapter, cache, bundle, subjects, mean, std, mod) -> dict[str, float]:
    indices = bundle.indices(subjects, (int(mod.TASKS[bundle.task]["future_session"]),))
    x.eval(); xra.eval()
    x_logits, xra_logits, cosine = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), 128):
            value, _ = cache.batch(indices[start:start + 128], mean, std)
            logits_x, repr_x = x(value)
            logits_r, repr_r = xra(value)
            x_logits.append(logits_x.float().cpu().numpy())
            xra_logits.append(logits_r.float().cpu().numpy())
            cosine.append(F.cosine_similarity(repr_x.float(), repr_r.float(), dim=1).cpu().numpy())
    left, right = np.concatenate(x_logits), np.concatenate(xra_logits)
    return {"prediction_disagreement_rate_vs_X": float(np.mean(left.argmax(1) != right.argmax(1))), "logit_l2_drift": float(np.sqrt(np.mean((left - right) ** 2))), "mean_embedding_cosine_vs_X": float(np.concatenate(cosine).mean())}


def block_drift(raw: dict[str, torch.Tensor], final: dict[str, torch.Tensor], prefix: str) -> tuple[float, float]:
    keys = [name for name in raw if name.startswith(prefix) and torch.is_floating_point(raw[name])]
    if not keys:
        return 0.0, 0.0
    delta = sum(float((final[name].detach().cpu().float() - raw[name].detach().cpu().float()).square().sum()) for name in keys) ** .5
    norm = sum(float(raw[name].detach().cpu().float().square().sum()) for name in keys) ** .5
    return delta, delta / max(norm, 1e-12)


def stage_b(commit: str) -> int:
    if git_head() != commit:
        raise RuntimeError("Stage B requires the current committed Stage-A revision")
    mod = load_module()
    _, folds, _ = mod.load_folds()
    lock = json.loads((OUT / "XRA_STAGE_A_LOCK.json").read_text(encoding="utf-8"))
    if not lock.get("stage_a_complete") or lock.get("winner") is None or lock.get("outer_development_opened"):
        raise RuntimeError("Stage B cannot open outer-development without a frozen passing Stage-A lock")
    regime = selected_regime(lock["winner"])
    selection = {(row["task"], int(row["fold"])): row for row in lock["selected_checkpoints"]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subject_rows, fold_rows, diagnostic_rows, drift_rows = [], [], [], []
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id, outer = int(fold["fold_id"]), fold["outer_dev_subjects"]
            normalizer = frozen_normalizer_path(task, fold_id)
            mean, std, norm_meta = mod.load_tensor_pair(normalizer)
            bundle = mod.build_bundle(task, outer)
            cache = mod.RawGPUCache(bundle, device)
            baseline = mod.build_model("LiteBN_BASELINE", task).to(device)
            baseline.load_state_dict(torch.load(baseline_path(task, fold_id), map_location=device, weights_only=False), strict=True)
            x = mod.build_model("LiteBN_X", task).to(device)
            raw_x = torch.load(base_x_path(task, fold_id), map_location=device, weights_only=False)
            x.load_state_dict(raw_x, strict=True)
            chosen_epoch = int(selection[(task, fold_id)]["selected_epoch"])
            xra = train_final(mod, regime, task, fold, chosen_epoch, device)
            metric_sets = {"LiteBN_BASELINE": subject_logits(mod, baseline, bundle, cache, outer, mean, std), "LiteBN_X": subject_logits(mod, x, bundle, cache, outer, mean, std), "LiteBN_XRA": subject_logits(mod, xra, bundle, cache, outer, mean, std)}
            for method, metrics in metric_sets.items():
                for subject, row in metrics.items():
                    subject_rows.append({"task": task, "dataset": mod.TASKS[task]["dataset"], "fold": fold_id, "subject": subject, "method": method, "BA": row["BA"], "macro_F1": row["macro_F1"], "accuracy": row["accuracy"], "trials": row["trials"]})
            base_metric, x_metric, xra_metric = (mean_metrics(metric_sets[key]) for key in ("LiteBN_BASELINE", "LiteBN_X", "LiteBN_XRA"))
            paired_baseline = np.asarray([metric_sets["LiteBN_XRA"][s]["BA"] - metric_sets["LiteBN_BASELINE"][s]["BA"] for s in metric_sets["LiteBN_XRA"]])
            paired_x = np.asarray([metric_sets["LiteBN_XRA"][s]["BA"] - metric_sets["LiteBN_X"][s]["BA"] for s in metric_sets["LiteBN_XRA"]])
            fold_rows.append({"task": task, "dataset": mod.TASKS[task]["dataset"], "fold": fold_id, "selected_epoch": chosen_epoch, "LiteBN_BA": base_metric["BA"], "X_BA": x_metric["BA"], "XRA_BA": xra_metric["BA"], "delta_vs_LiteBN_pp": 100 * float(paired_baseline.mean()), "delta_vs_X_pp": 100 * float(paired_x.mean()), "positive_subjects_vs_LiteBN": int(np.sum(paired_baseline > 0)), "harmed_subjects_vs_LiteBN": int(np.sum(paired_baseline < 0)), "tied_subjects_vs_LiteBN": int(np.sum(paired_baseline == 0))})
            xra.force_adapter_off = True
            off_rows = subject_logits(mod, xra, bundle, cache, outer, mean, std)
            off_metric = mean_metrics(off_rows)
            xra.force_adapter_off = False
            off_logit_difference = max(float(np.max(np.abs(metric_sets["LiteBN_X"][subject]["logits"] - off_rows[subject]["logits"]))) for subject in off_rows)
            off_mismatch = sum(int(np.sum(metric_sets["LiteBN_X"][subject]["logits"].argmax(1) != off_rows[subject]["logits"].argmax(1))) for subject in off_rows)
            diagnostic_rows.append({"task": task, "dataset": mod.TASKS[task]["dataset"], "fold": fold_id, "selected_epoch": chosen_epoch, "normalizer_sha256": norm_meta["mean_std_sha256"], **correction_diagnostics(xra, cache, bundle, outer, mean, std, mod), "X_BA": x_metric["BA"], "XRA_BA": xra_metric["BA"], "adapter_off_BA": off_metric["BA"], "on_minus_off_pp": 100 * (xra_metric["BA"] - off_metric["BA"]), "off_minus_X_pp": 100 * (off_metric["BA"] - x_metric["BA"]), "adapter_off_max_abs_logit_difference_vs_X": off_logit_difference, "adapter_off_prediction_mismatch_vs_X": off_mismatch, **functional_drift(x, xra, cache, bundle, outer, mean, std, mod)})
            final_base = xra.base.state_dict()
            for name, prefix in (("stem", "stem."), ("scale_mlp", "scale_mlp."), ("lambda_scale", "lambda_scale"), ("mixer0", "mixer.0."), ("mixer1", "mixer.1."), ("mixer2", "mixer.2."), ("embedding", "embedding."), ("head", "head.")):
                absolute, relative = block_drift(raw_x, final_base, prefix)
                if regime == G0 and absolute != 0.0:
                    raise RuntimeError(f"G0 changed frozen X parameter block: {task}/f{fold_id}/{name}")
                drift_rows.append({"task": task, "fold": fold_id, "regime": regime.name, "block": name, "absolute_l2_drift": absolute, "relative_l2_drift": relative})
            drift_rows.append({"task": task, "fold": fold_id, "regime": regime.name, "block": "residual_adapter", "absolute_l2_drift": float(sum(float(value.detach().cpu().float().square().sum()) for value in xra.channel_mlp.parameters()) ** .5), "relative_l2_drift": np.nan})
            del baseline, x, xra, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    subject_frame, fold_frame = pd.DataFrame(subject_rows), pd.DataFrame(fold_rows)
    task_rows = []
    for task, subset in subject_frame.groupby("task"):
        pivot = subset.pivot(index="subject", columns="method", values=["BA", "macro_F1", "accuracy"])
        delta_base = (pivot[("BA", "LiteBN_XRA")] - pivot[("BA", "LiteBN_BASELINE")]).to_numpy()
        delta_x = (pivot[("BA", "LiteBN_XRA")] - pivot[("BA", "LiteBN_X")]).to_numpy()
        low, high = bootstrap(delta_base, SEED + len(task))
        task_rows.append({"task": task, "dataset": subset.dataset.iloc[0], "LiteBN_BA": float(pivot[("BA", "LiteBN_BASELINE")].mean()), "X_BA": float(pivot[("BA", "LiteBN_X")].mean()), "XRA_BA": float(pivot[("BA", "LiteBN_XRA")].mean()), "delta_vs_LiteBN_pp": 100 * float(delta_base.mean()), "delta_vs_X_pp": 100 * float(delta_x.mean()), "bootstrap_ci_low_pp": low, "bootstrap_ci_high_pp": high, "LiteBN_macro_F1": float(pivot[("macro_F1", "LiteBN_BASELINE")].mean()), "X_macro_F1": float(pivot[("macro_F1", "LiteBN_X")].mean()), "XRA_macro_F1": float(pivot[("macro_F1", "LiteBN_XRA")].mean()), "LiteBN_accuracy": float(pivot[("accuracy", "LiteBN_BASELINE")].mean()), "X_accuracy": float(pivot[("accuracy", "LiteBN_X")].mean()), "XRA_accuracy": float(pivot[("accuracy", "LiteBN_XRA")].mean()), "positive_folds": int(np.sum(fold_frame[fold_frame.task == task].delta_vs_LiteBN_pp > 0)), "harmed_folds": int(np.sum(fold_frame[fold_frame.task == task].delta_vs_LiteBN_pp < 0)), "positive_subjects": int(np.sum(delta_base > 0)), "harmed_subjects": int(np.sum(delta_base < 0)), "tied_subjects": int(np.sum(delta_base == 0))})
    task_frame = pd.DataFrame(task_rows).sort_values("task").reset_index(drop=True)
    write_csv(OUT / "XRA_OUTER_SUBJECT_RESULTS.csv", subject_frame)
    write_csv(OUT / "XRA_OUTER_FOLD_RESULTS.csv", fold_frame)
    write_csv(OUT / "XRA_OUTER_TASK_RESULTS.csv", task_frame)
    write_csv(OUT / "XRA_ADAPTER_DIAGNOSTICS.csv", pd.DataFrame(diagnostic_rows))
    write_csv(OUT / "XRA_PARAMETER_DRIFT.csv", pd.DataFrame(drift_rows))
    deltas = task_frame.delta_vs_LiteBN_pp.to_numpy()
    terminal = "XRA_SEED0_4OF4_SUCCESS" if bool(np.all(deltas > 0)) else ("XRA_SEED0_MIXED" if float(deltas.mean()) > 0 else "XRA_SEED0_FAIL")
    report = ["# XRA seed-0 decision", "", f"Regime: {regime.name}", "", "Epoch-0 exact-X replay: PASS (20/20).", "Architecture and training recipe were frozen and committed before outer-development was opened.", "", "## Outer-development task results", "", task_frame.to_csv(index=False), "", f"Four-task mean delta vs LiteBN: {float(deltas.mean()):+.3f} pp.", f"Worst-task delta vs LiteBN: {float(deltas.min()):+.3f} pp.", f"Positive tasks: {int(np.sum(deltas > 0))}/4.", f"Positive folds: {int(np.sum(fold_frame.delta_vs_LiteBN_pp > 0))}/20.", "", f"Terminal: `{terminal}`", "", "Outer-development did not influence any subsequent tuning.", "FINAL_HELDOUT_ACCESSED = NO", ""]
    (OUT / "FINAL_XRA_SEED0_DECISION.md").write_text("\n".join(report), encoding="utf-8")
    lock["outer_development_opened"] = True
    lock["outer_development_completed"] = True
    lock["stage_b_commit"] = commit
    lock["terminal"] = terminal
    write_json(OUT / "XRA_STAGE_A_LOCK.json", lock)
    print(f"XRA_STAGE_B_DONE {terminal}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("stage_a", "stage_b", "stage_g2_wbcic"), required=True)
    parser.add_argument("--stage-a-commit")
    args = parser.parse_args()
    if args.stage == "stage_a":
        return run_stage_a()
    if args.stage == "stage_g2_wbcic":
        return run_stage_g2_wbcic()
    if not args.stage_a_commit:
        raise SystemExit("stage_b requires --stage-a-commit")
    return stage_b(args.stage_a_commit)


if __name__ == "__main__":
    raise SystemExit(main())
