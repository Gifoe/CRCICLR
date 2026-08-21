from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (
    CHECKPOINTS,
    OUTPUTS,
    SMOKE,
    balanced_accuracy,
    ce_loss,
    clean,
    ensure_directories,
    load_config,
    macro_f1,
    seed_all,
    sha256_file,
    softmax,
    stable_seed,
    write_csv,
    write_json,
)
from data import (
    MIDataset,
    domain_map,
    load_development_split,
    load_manifest,
    make_loader,
    normalizer,
    select_frame,
    subject_sort,
)
from losses import (
    conditional_alignment_loss,
    coral_loss,
    expert_conditional_alignment_loss,
    expert_mmd_loss,
    marginal_mmd_loss,
    supervised_contrastive_loss,
)
from models import build_model, grl_lambda, method_family, parameter_count, roster


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def _run_root(mode: str, fold: int, seed: int, method_id: str) -> Path:
    base = SMOKE if mode == "smoke" else OUTPUTS / "runs"
    return base / f"fold-{fold}" / f"seed-{seed}" / method_id


def checkpoint_path(mode: str, fold: int, seed: int, method_id: str) -> Path:
    if mode == "smoke":
        return _run_root(mode, fold, seed, method_id) / "best.pt"
    return CHECKPOINTS / f"fold-{fold}" / f"seed-{seed}" / f"{method_id}.pt"


def representation_path(fold: int, seed: int, method_id: str) -> Path:
    return OUTPUTS / "cache" / "representations" / f"fold-{fold}" / f"seed-{seed}" / f"{method_id}.npz"


def _loss(
    method_id: str,
    output: Any,
    labels: torch.Tensor,
    domains: torch.Tensor,
    step: int,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    task = F.cross_entropy(output.logits.float(), labels)
    total = task
    parts: dict[str, torch.Tensor] = {"task_ce": task}
    family = method_family(method_id)

    if family == "A" and grl_lambda(method_id) > 0:
        domain = F.cross_entropy(output.domain_logits.float(), domains)
        total = total + domain
        parts["subject_ce"] = domain
    elif method_id == "B1_EEG_DG_FULL":
        weights = config["eeg_dg_weights"]
        if output.expert_features is None:
            raise RuntimeError("B1 requires source-special expert features")
        marginal = expert_mmd_loss(output.expert_features, step)
        conditional = expert_conditional_alignment_loss(output.expert_features, labels, step)
        domain = F.cross_entropy(output.domain_logits.float(), domains)
        total = (
            total
            + float(weights["marginal_mmd"]) * marginal
            + float(weights["conditional_alignment"]) * conditional
            + float(weights["domain_classification"]) * domain
        )
        parts.update(marginal_mmd=marginal, conditional_alignment=conditional, domain_ce=domain)
    elif method_id == "C1_SCLDGN_FULL":
        weights = config["scldgn_weights"]
        alignment = coral_loss(output.features.float(), labels, domains, step)
        contrastive = supervised_contrastive_loss(output.projection.float(), labels, step)
        total = total + float(weights["coral"]) * alignment + float(weights["supervised_contrastive"]) * contrastive
        parts.update(coral=alignment, supervised_contrastive=contrastive)

    values = {name: float(value.detach().cpu()) for name, value in parts.items()}
    values["total"] = float(total.detach().cpu())
    return total, values


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits, labels, positions = [], [], []
    for signals, target, _, position in loader:
        signals = signals.to(device, non_blocking=True)
        with _autocast(device):
            output = model(signals)
        logits.append(output.logits.float().cpu().numpy())
        labels.append(target.numpy())
        positions.append(position.numpy())
    y = np.concatenate(labels).astype(np.int64)
    score = np.concatenate(logits).astype(np.float32)
    pred = score.argmax(axis=1)
    probability = softmax(score)
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "cross_entropy": ce_loss(y, probability),
        "n": int(len(y)),
        "labels": y,
        "predictions": pred,
        "logits": score,
        "positions": np.concatenate(positions).astype(np.int64),
    }


def _checkpoint_payload(
    model: torch.nn.Module,
    method_id: str,
    fold: int,
    seed: int,
    epoch: int,
    calibration: Mapping[str, Any],
    history: list[dict[str, Any]],
    split_payload: Mapping[str, Any],
    normalizer_path: Path,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "method_id": method_id,
        "fold": int(fold),
        "seed": int(seed),
        "epoch": int(epoch),
        "calibration": {k: v for k, v in calibration.items() if k not in {"labels", "predictions", "logits", "positions"}},
        "history": history,
        "parameter_count": parameter_count(model),
        "split": dict(split_payload),
        "normalizer_sha256": sha256_file(normalizer_path),
        "outer_test_used": False,
    }


def train_one(
    method_id: str,
    fold: int,
    seed: int,
    device: torch.device,
    mode: str = "full",
    force: bool = False,
) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    ensure_directories()
    config = load_config()
    split = load_development_split(fold)
    # Training never materializes development-outcome rows or labels.
    manifest = load_manifest(split, split.model_fit_subjects + split.calibration_subjects)
    mean, std, norm_path = normalizer(fold, manifest, split.model_fit_subjects)
    dmap = domain_map(split.model_fit_subjects)
    train_frame = select_frame(manifest, split.model_fit_subjects, config["train_sessions"])
    calibration_frame = select_frame(manifest, split.calibration_subjects, [config["calibration_session"]])
    family = method_family(method_id)
    hyper = config["training"]["family_hyperparameters"][family]
    run_seed = stable_seed("train", mode, fold, seed, method_id)
    seed_all(run_seed)
    model = build_model(method_id, len(dmap), config).to(device)
    ckpt = checkpoint_path(mode, fold, seed, method_id)
    done_path = _run_root(mode, fold, seed, method_id) / "TRAINING_COMPLETE.json"
    if not force and ckpt.exists() and done_path.exists():
        saved = json.loads(done_path.read_text(encoding="utf-8"))
        if saved.get("status") == "COMPLETE" and saved.get("outer_test_used") is False:
            print(f"[train] reuse {mode} fold={fold} seed={seed} method={method_id}", flush=True)
            return saved

    train_set = MIDataset(train_frame, mean, std, dmap)
    calibration_set = MIDataset(calibration_frame, mean, std, None)
    train_loader = make_loader(train_set, int(hyper["batch_size"]), True, run_seed)
    calibration_loader = make_loader(calibration_set, int(hyper["batch_size"]), False, run_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(hyper["learning_rate"]), weight_decay=float(hyper["weight_decay"])
    )
    max_epochs = int(config["training"][f"{mode}_max_epochs"])
    min_epochs = int(config["training"][f"{mode}_min_epochs"])
    patience = int(config["training"][f"{mode}_patience"])
    clip = float(config["training"]["gradient_clip"])
    history: list[dict[str, Any]] = []
    best_ba = -np.inf
    best_ce = np.inf
    stale = 0
    global_step = 0
    started = time.time()
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    _run_root(mode, fold, seed, method_id).mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_parts: dict[str, list[float]] = {}
            for signals, labels, domains, _ in train_loader:
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                domains = domains.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    output = model(signals, grl_strength=grl_lambda(method_id))
                    loss, parts = _loss(method_id, output, labels, domains, global_step, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {global_step}: {parts}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                for name, value in parts.items():
                    epoch_parts.setdefault(name, []).append(value)
                global_step += 1
            calibration = evaluate(model, calibration_loader, device)
            row = {
                "epoch": epoch,
                **{f"train_{key}": float(np.mean(values)) for key, values in epoch_parts.items()},
                **{f"calibration_{key}": calibration[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "cross_entropy")},
            }
            history.append(row)
            better = (
                calibration["balanced_accuracy"] > best_ba + 1e-12
                or (
                    abs(calibration["balanced_accuracy"] - best_ba) <= 1e-12
                    and calibration["cross_entropy"] < best_ce
                )
            )
            if better:
                best_ba = float(calibration["balanced_accuracy"])
                best_ce = float(calibration["cross_entropy"])
                stale = 0
                torch.save(
                    _checkpoint_payload(
                        model, method_id, fold, seed, epoch, calibration, history, split.payload(), norm_path
                    ),
                    ckpt,
                )
            else:
                stale += 1
            print(
                f"[train] {mode} f{fold}s{seed} {method_id} epoch={epoch}/{max_epochs} "
                f"cal_BA={calibration['balanced_accuracy']:.4f} best={best_ba:.4f} stale={stale}",
                flush=True,
            )
            if epoch >= min_epochs and stale >= patience:
                break
    except Exception as error:
        failure = {
            "status": "FAILED",
            "mode": mode,
            "fold": fold,
            "seed": seed,
            "method_id": method_id,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
            "outer_test_used": False,
        }
        write_json(_run_root(mode, fold, seed, method_id) / "TRAINING_FAILED.json", failure)
        raise

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    payload = {
        "status": "COMPLETE",
        "mode": mode,
        "fold": fold,
        "seed": seed,
        "method_id": method_id,
        "family": family,
        "best_epoch": int(state["epoch"]),
        "best_calibration_BA": float(state["calibration"]["balanced_accuracy"]),
        "best_calibration_macro_f1": float(state["calibration"]["macro_f1"]),
        "best_calibration_CE": float(state["calibration"]["cross_entropy"]),
        "epochs_executed": len(history),
        "parameter_count": int(state["parameter_count"]),
        "checkpoint": str(ckpt),
        "elapsed_seconds": time.time() - started,
        "train_rows": len(train_frame),
        "calibration_rows": len(calibration_frame),
        "outcome_loader_constructed_during_training": False,
        "outer_test_used": False,
    }
    write_json(done_path, payload)
    write_csv(_run_root(mode, fold, seed, method_id) / "TRAINING_HISTORY.csv", pd.DataFrame(history))
    return payload


def load_trained_model(
    method_id: str, fold: int, seed: int, device: torch.device, mode: str = "full"
) -> torch.nn.Module:
    config = load_config()
    split = load_development_split(fold)
    model = build_model(method_id, len(split.model_fit_subjects), config)
    payload = torch.load(checkpoint_path(mode, fold, seed, method_id), map_location="cpu", weights_only=False)
    if payload["method_id"] != method_id or int(payload["fold"]) != fold or int(payload["seed"]) != seed:
        raise RuntimeError("checkpoint provenance mismatch")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def extract_one(method_id: str, fold: int, seed: int, device: torch.device, force: bool = False) -> Path:
    path = representation_path(fold, seed, method_id)
    provenance_path = path.with_suffix(".json")
    if path.exists() and provenance_path.exists() and not force:
        print(f"[extract] reuse f{fold}s{seed} {method_id}", flush=True)
        return path
    config = load_config()
    split = load_development_split(fold)
    manifest = load_manifest(split)
    mean, std, norm_path = normalizer(fold, manifest, split.model_fit_subjects)
    frame = select_frame(manifest, split.allowed_subjects, [1, 2])
    dataset = MIDataset(frame, mean, std, None)
    family = method_family(method_id)
    batch_size = int(config["training"]["family_hyperparameters"][family]["batch_size"])
    loader = make_loader(dataset, batch_size, False, stable_seed("extract", fold, seed, method_id))
    model = load_trained_model(method_id, fold, seed, device)
    features, logits, positions = [], [], []
    for signals, _, _, position in loader:
        signals = signals.to(device, non_blocking=True)
        with _autocast(device):
            output = model(signals)
        features.append(output.features.float().cpu().numpy())
        logits.append(output.logits.float().cpu().numpy())
        positions.append(position.numpy())
    h = np.concatenate(features).astype(np.float32)
    score = np.concatenate(logits).astype(np.float32)
    pos = np.concatenate(positions).astype(np.int64)
    if len(pos) != len(frame) or len(np.unique(pos)) != len(pos) or h.shape != (len(frame), int(config["embedding_dim"])):
        raise RuntimeError("representation coverage failure")
    order = np.argsort(pos)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(
        temporary,
        positions=pos[order],
        features=h[order],
        logits=score[order],
        outer_test_used=np.asarray(False),
    )
    os.replace(temporary, path)
    write_json(
        provenance_path,
        {
            "method_id": method_id,
            "fold": fold,
            "seed": seed,
            "representation": "natural penultimate pre-classifier embedding",
            "shape": list(h.shape),
            "manifest_positions_sha256": __import__("hashlib").sha256(pos[order].tobytes()).hexdigest(),
            "checkpoint_sha256": sha256_file(checkpoint_path("full", fold, seed, method_id)),
            "normalizer_sha256": sha256_file(norm_path),
            "allowed_roles": ["model_fit", "calibration", "development_outcome"],
            "outer_test_used": False,
        },
    )
    return path


def load_representation(method_id: str, fold: int, seed: int) -> dict[str, np.ndarray]:
    path = representation_path(fold, seed, method_id)
    if not path.exists():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False)
    if bool(value["outer_test_used"].item()):
        raise RuntimeError("outer lock violation in representation")
    return {name: value[name] for name in ("positions", "features", "logits")}


def run_smoke(device: torch.device, force: bool = False) -> list[dict[str, Any]]:
    config = load_config()
    results = [train_one(method, 0, 0, device, mode="smoke", force=force) for method in roster(config)]
    return results


def run_full(device: torch.device, force: bool = False) -> list[dict[str, Any]]:
    config = load_config()
    rows: list[dict[str, Any]] = []
    for fold in config["development_folds"]:
        for seed in config["seeds"]:
            for method in roster(config):
                result = train_one(method, int(fold), int(seed), device, mode="full", force=force)
                extract_one(method, int(fold), int(seed), device, force=force)
                rows.append(result)
    write_csv(OUTPUTS / "RUN_LEDGER.csv", pd.DataFrame(rows))
    return rows
