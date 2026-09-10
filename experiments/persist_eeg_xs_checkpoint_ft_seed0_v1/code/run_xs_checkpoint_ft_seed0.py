#!/usr/bin/env python3
"""Checkpoint-local LiteBN-XS fine-tuning for seed-0 development only.

The runner never enumerates a final held-out cohort. Stage A builds only
inner-train/inner-validation bundles, locks a single winner using those bundles,
and only then opens frozen outer-development bundles once for Stage B.
"""
from __future__ import annotations

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
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO = Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")
BASE_EXP = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1"
EXP = REPO / "experiments/persist_eeg_xs_checkpoint_ft_seed0_v1"
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path("/root/rivermind-data/xs_checkpoint_ft_seed0_runtime")
SOURCE_RUNTIME = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")
HIST_CARRIER = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")
HIST_TASK = Path("/root/rivermind-data/openbmi_task_generality_runtime")

SEED = 0
MAX_EPOCHS = 20
MIN_EPOCH = 3
PATIENCE = 5
GATE_LR = 1e-4
HEAD_LR = 1e-4
EMBEDDING_LR = 3e-5
MIXER_LR = 1e-5
WEIGHT_DECAY = 5e-4
CLIP = 5.0
ANCHOR = 1e-3
WEAK_GATE_SHRINK = 1e-4
BOOTSTRAPS = 10_000
TOL = 1e-10


@dataclass(frozen=True)
class Regime:
    name: str
    train_embedding: bool
    train_last_mixer: bool
    gate_shrinkage: float
    simplicity_rank: int

    @property
    def trainable_modules(self) -> list[str]:
        result = ["lambda_channel", "channel_mlp", "classifier_head"]
        if self.train_embedding:
            result.insert(-1, "embedding")
        if self.train_last_mixer:
            result.insert(-1, "last_residual_temporal_mixer")
        return result

    @property
    def frozen_modules(self) -> list[str]:
        result = [
            "temporal_multi_scale_stems",
            "spatial_blocks",
            "scale_gate",
            "temporal_feature_blocks",
            "pooling",
        ]
        if not self.train_embedding:
            result.append("embedding_projection")
        if not self.train_last_mixer:
            result.append("all_residual_temporal_mixers")
        else:
            result.append("residual_temporal_mixers_0_to_1")
        return result


FT1 = Regime("FT1_GATE_HEAD_WEAK_SHRINK", False, False, WEAK_GATE_SHRINK, 1)
FT2 = Regime("FT2_GATE_EMBEDDING_HEAD_WEAK_SHRINK", True, False, WEAK_GATE_SHRINK, 2)
FT3 = Regime("FT3_GATE_LAST_MIXER_EMBEDDING_HEAD_WEAK_SHRINK", True, True, WEAK_GATE_SHRINK, 3)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
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
    output: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        output["cuda"] = torch.cuda.get_rng_state_all()
    return output


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].detach().cpu())
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in value["cuda"]])


def state_hash(model: torch.nn.Module) -> str:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return digest_bytes(stream.getvalue())


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def load_module():
    source = BASE_EXP / "code/litebn_x.py"
    spec = importlib.util.spec_from_file_location("xs_checkpoint_ft_impl", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LiteBN-X source implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEED = SEED
    module.RUNTIME = RUNTIME
    return module


def source_xs_path(task: str, fold: int) -> Path:
    return SOURCE_RUNTIME / "checkpoints" / task.lower() / f"fold{fold}_litebn_xs" / "selected_best.pt"


def baseline_path(task: str, fold: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        prefix = "erp" if task == "OpenBMI_ERP" else "ssvep"
        return HIST_TASK / f"{prefix}_fold{fold}_seed0_litebn" / "selected_best.pt"
    prefix = "openbmi" if task == "OpenBMI_MI" else "wbcic"
    return HIST_CARRIER / f"{prefix}_fold{fold}_seed0_litebn" / "selected_best.pt"


def normalizer_path(task: str, fold: int) -> Path:
    return SOURCE_RUNTIME / "normalizers" / f"{task.lower()}_fold{fold}.npz"


def cell_dir(regime: Regime, task: str, fold: int) -> Path:
    return RUNTIME / "checkpoints" / regime.name.lower() / task.lower() / f"fold{fold}"


def load_source_history() -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    sources = [
        BASE_EXP / "outputs/STAGE_A_INNERVAL_RESULTS.csv",
        BASE_EXP / "outputs/POSTHOC_XS_SEED0_SSVEP_INNERVAL.csv",
    ]
    for path in sources:
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if "architecture" not in frame.columns:
            continue
        for _, row in frame[frame.architecture.eq("LiteBN_XS")].iterrows():
            result[(str(row.task), int(row.fold))] = {
                "stored_inner_val_BA": float(row.best_inner_val_BA),
                "stored_selected_epoch": int(row.selected_epoch),
                "stored_checkpoint_sha256": str(row.checkpoint_sha256) if "checkpoint_sha256" in row else None,
                "stored_source": str(path),
            }
    return result


def parameter_count(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad))


def results_fingerprint(results: dict[str, dict[str, Any]]) -> str:
    compact = {
        str(subject): {key: float(value) if isinstance(value, (float, np.floating)) else int(value)
                       for key, value in sorted(metrics.items())}
        for subject, metrics in sorted(results.items())
    }
    return digest_bytes(json.dumps(compact, sort_keys=True).encode("utf-8"))


def load_xs(mod, task: str, source: Path, device: torch.device) -> torch.nn.Module:
    model = mod.build_model("LiteBN_XS", task).to(device)
    model.load_state_dict(torch.load(source, map_location=device, weights_only=False), strict=True)
    model.eval()
    return model


def load_ft(mod, task: str, source: Path, fine_tuned: Path, device: torch.device) -> torch.nn.Module:
    model = load_xs(mod, task, source, device)
    model.load_state_dict(torch.load(fine_tuned, map_location=device, weights_only=False), strict=True)
    model.eval()
    return model


def make_manifest(mod, folds: dict[str, list[dict[str, Any]]], split_hash: str, device: torch.device) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    history, rows = load_source_history(), []
    source_code = BASE_EXP / "code/litebn_x.py"
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            source, normalizer = source_xs_path(task, fold_id), normalizer_path(task, fold_id)
            if not source.is_file() or not normalizer.is_file():
                raise FileNotFoundError(f"missing XS provenance: {task}/fold{fold_id}")
            _, _, metadata = mod.load_tensor_pair(normalizer)
            model = load_xs(mod, task, source, device)
            stored = history.get((task, fold_id), {})
            rows.append({
                "task": task,
                "dataset": mod.TASKS[task]["dataset"],
                "fold": fold_id,
                "seed": SEED,
                "checkpoint_path": str(source),
                "checkpoint_sha256": digest(source),
                "training_architecture": "LiteBN_XS",
                "selected_epoch": stored.get("stored_selected_epoch"),
                "stored_inner_val_BA": stored.get("stored_inner_val_BA"),
                "stored_checkpoint_sha256": stored.get("stored_checkpoint_sha256"),
                "stored_result_source": stored.get("stored_source"),
                "split_sha256": split_hash,
                "normalizer_path": str(normalizer),
                "normalizer_sha256": metadata["mean_std_sha256"],
                "code_source_path": str(source_code),
                "code_source_sha256": digest(source_code),
                "code_commit_before_ft": git_commit(),
                "total_parameter_count": parameter_count(model),
            })
            del model
    frame = pd.DataFrame(rows).sort_values(["task", "fold"])
    write_csv(OUT / "XS_SOURCE_CHECKPOINT_MANIFEST_SEED0.csv", frame)
    return frame, history


def replay_all(mod, folds: dict[str, list[dict[str, Any]]], history: dict[tuple[str, int], dict[str, Any]], device: torch.device) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    rows, replay = [], {}
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            source = source_xs_path(task, fold_id)
            mean, std, metadata = mod.load_tensor_pair(normalizer_path(task, fold_id))
            allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            bundle = mod.build_bundle(task, allowed)
            cache = mod.RawGPUCache(bundle, device)
            model = load_xs(mod, task, source, device)
            first = mod.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
            second = mod.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
            first_ba = float(np.mean([value["BA"] for value in first.values()]))
            second_ba = float(np.mean([value["BA"] for value in second.values()]))
            stored = history.get((task, fold_id), {}).get("stored_inner_val_BA")
            difference = None if stored is None else abs(first_ba - float(stored))
            deterministic = results_fingerprint(first) == results_fingerprint(second)
            passed = bool(deterministic and (difference is None or difference <= TOL))
            row = {
                "task": task,
                "dataset": mod.TASKS[task]["dataset"],
                "fold": fold_id,
                "stored_BA": stored,
                "replayed_BA": first_ba,
                "second_replayed_BA": second_ba,
                "absolute_difference": difference,
                "replay_fingerprint": results_fingerprint(first),
                "repeat_fingerprint": results_fingerprint(second),
                "deterministic_replay": deterministic,
                "replay_passed": passed,
                "stored_record_available": stored is not None,
                "status": "MATCHED_STORED_RESULT" if stored is not None else "DETERMINISTIC_REPLAY_NO_COMMITTED_INNER_RECORD",
                "checkpoint_sha256": digest(source),
                "normalizer_sha256": metadata["mean_std_sha256"],
            }
            rows.append(row)
            replay[(task, fold_id)] = row
            del model, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["task", "fold"])
    write_csv(OUT / "XS_CHECKPOINT_REPLAY_SEED0.csv", frame)
    if not bool(frame.replay_passed.all()):
        raise RuntimeError("an XS checkpoint replay did not pass; fine-tuning is forbidden")
    return frame, replay


def cached_replay_if_valid(mod, folds: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int], dict[str, Any]] | None:
    """Reuse a completed exact replay only when its source provenance still matches."""
    path = OUT / "XS_CHECKPOINT_REPLAY_SEED0.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    required = {"task", "fold", "replayed_BA", "replay_passed", "checkpoint_sha256", "normalizer_sha256"}
    if not required.issubset(frame.columns) or len(frame) != 20 or not bool(frame.replay_passed.all()):
        return None
    expected: dict[tuple[str, int], tuple[str, str]] = {}
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            _, _, metadata = mod.load_tensor_pair(normalizer_path(task, fold_id))
            expected[(task, fold_id)] = (digest(source_xs_path(task, fold_id)), metadata["mean_std_sha256"])
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = (str(row.task), int(row.fold))
        if key not in expected or (str(row.checkpoint_sha256), str(row.normalizer_sha256)) != expected[key]:
            return None
        output[key] = row.to_dict()
    return output if len(output) == 20 else None


def baseline_inner_all(mod, folds: dict[str, list[dict[str, Any]]], device: torch.device) -> dict[tuple[str, int], float]:
    output: dict[tuple[str, int], float] = {}
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id = int(fold["fold_id"])
            bundle = mod.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
            mean, std, _ = mod.load_tensor_pair(normalizer_path(task, fold_id))
            cache = mod.RawGPUCache(bundle, device)
            model = mod.build_model("LiteBN_BASELINE", task).to(device)
            checkpoint = baseline_path(task, fold_id)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing LiteBN baseline: {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
            metrics = mod.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
            output[(task, fold_id)] = float(np.mean([value["BA"] for value in metrics.values()]))
            del model, cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return output


def gate_correction(model: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
    diff_rms = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + 1e-8)
    raw = model.channel_mlp(torch.stack((rms, diff_rms), dim=-1)).squeeze(-1)
    return torch.tanh(model.lambda_channel) * torch.tanh(raw)


def gate_stats(model: torch.nn.Module, cache, bundle, subjects, mean, std, mod, task: str) -> dict[str, float]:
    indices = bundle.indices(subjects, mod.TASKS[task]["source_sessions"])
    corrections = []
    with torch.no_grad():
        for start in range(0, len(indices), 64):
            value, _ = cache.batch(indices[start:start + 64], mean, std)
            corrections.append(gate_correction(model, value).abs().flatten().detach().float().cpu())
    values = torch.cat(corrections).numpy()
    amplitude = float(torch.tanh(model.lambda_channel).detach().float().cpu())
    return {
        "lambda_channel": float(model.lambda_channel.detach().float().cpu()),
        "signed_amplitude": amplitude,
        "absolute_amplitude": abs(amplitude),
        "mean_absolute_channel_correction": float(values.mean()),
        "median_absolute_channel_correction": float(np.median(values)),
        "std_absolute_channel_correction": float(values.std()),
        "max_absolute_channel_correction": float(values.max()),
    }


def freeze_and_optimizer(model: torch.nn.Module, regime: Regime):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.lambda_channel.requires_grad_(True)
    for parameter in model.channel_mlp.parameters():
        parameter.requires_grad_(True)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    if regime.train_embedding:
        for parameter in model.embedding.parameters():
            parameter.requires_grad_(True)
    if regime.train_last_mixer:
        for parameter in model.mixer[-1].parameters():
            parameter.requires_grad_(True)
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("head.")
    }
    groups = [
        {"params": [model.lambda_channel, *model.channel_mlp.parameters()], "lr": GATE_LR},
        {"params": list(model.head.parameters()), "lr": HEAD_LR},
    ]
    if regime.train_embedding:
        groups.append({"params": list(model.embedding.parameters()), "lr": EMBEDDING_LR})
    if regime.train_last_mixer:
        groups.append({"params": list(model.mixer[-1].parameters()), "lr": MIXER_LR})
    return anchor, torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def keep_frozen_batch_norms_in_eval(model: torch.nn.Module) -> None:
    """Frozen stem BatchNorm buffers are part of the preserved XS state."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def anchor_loss(model: torch.nn.Module, anchor: dict[str, torch.Tensor]) -> torch.Tensor:
    values = [(parameter - anchor[name]).square().mean() for name, parameter in model.named_parameters() if name in anchor]
    return torch.stack(values).mean() if values else torch.zeros((), device=next(model.parameters()).device)


def module_drift(raw: dict[str, torch.Tensor], final: dict[str, torch.Tensor], prefix: str) -> tuple[float, float]:
    keys = [name for name in raw if name.startswith(prefix) and torch.is_floating_point(raw[name])]
    if not keys:
        return 0.0, 0.0
    numerator = sum(float((final[name].detach().cpu().float() - raw[name].detach().cpu().float()).square().sum()) for name in keys) ** 0.5
    denominator = sum(float(raw[name].detach().cpu().float().square().sum()) for name in keys) ** 0.5
    return numerator, numerator / max(denominator, 1e-12)


def train_cell(mod, task: str, fold: dict[str, Any], regime: Regime, replay: dict[str, Any],
               baseline_ba: float, split_hash: str, device: torch.device) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    directory = cell_dir(regime, task, fold_id)
    record_path, latest_path, selected_path = directory / "record.json", directory / "checkpoint_latest.pt", directory / "selected_best.pt"
    source, normalizer = source_xs_path(task, fold_id), normalizer_path(task, fold_id)
    invariant = {
        "regime": regime.name,
        "task": task,
        "fold": fold_id,
        "seed": SEED,
        "source_xs_sha256": digest(source),
        "normalizer_sha256": mod.load_tensor_pair(normalizer)[2]["mean_std_sha256"],
        "split_sha256": split_hash,
        "max_epochs": MAX_EPOCHS,
        "min_epoch": MIN_EPOCH,
        "patience": PATIENCE,
        "gate_lr": GATE_LR,
        "head_lr": HEAD_LR,
        "embedding_lr": EMBEDDING_LR,
        "mixer_lr": MIXER_LR,
        "anchor": ANCHOR,
        "gate_shrinkage": regime.gate_shrinkage,
        "runner_sha256": digest(Path(__file__)),
    }
    if record_path.is_file() and selected_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("invariant") == invariant:
            return previous
        raise RuntimeError(f"completed checkpoint invariant mismatch: {record_path}")

    allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
    bundle = mod.build_bundle(task, allowed)
    mean, std, norm_meta = mod.load_tensor_pair(normalizer)
    cache = mod.RawGPUCache(bundle, device)
    if mod.TASKS[task]["mi_protocol"]:
        episodes, metadata = mod.mi_manifest(bundle, fold, task)
        batch_info = {"episodes": episodes, **metadata}
    else:
        batch_info = {"episodes": None, "kind": "historical_task_full_permutation_batch64"}
    class_weight, _ = mod.class_weights(bundle, fold["inner_train_subjects"])
    class_weight = None if class_weight is None else class_weight.to(device)

    set_seed(SEED)
    model = load_xs(mod, task, source, device)
    raw_state = copy.deepcopy(model.state_dict())
    initial_state_sha = state_hash(model)
    initial_gate = gate_stats(model, cache, bundle, fold["inner_val_subjects"], mean, std, mod, task)
    anchor, optimizer = freeze_and_optimizer(model, regime)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_ba, best_epoch, best_state, history, start, stalled = -float("inf"), None, None, [], 1, 0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved.get("invariant") != invariant:
            raise RuntimeError(f"resume invariant mismatch: {latest_path}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        best_ba, best_epoch, best_state = float(saved["best_ba"]), saved["best_epoch"], saved["best_state"]
        history, start, stalled = list(saved["history"]), int(saved["epoch"]) + 1, int(saved["stalled"])
    train_indices = bundle.indices(fold["inner_train_subjects"], mod.TASKS[task]["source_sessions"])
    started = time.perf_counter()
    for epoch in range(start, MAX_EPOCHS + 1):
        model.train()
        keep_frozen_batch_norms_in_eval(model)
        ce_values, anchor_values, shrink_values = [], [], []
        batches = batch_info["episodes"][epoch - 1] if batch_info["episodes"] is not None else mod.task_epoch_batches(train_indices, task, fold_id, epoch)
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(value)
                ce = F.cross_entropy(logits, labels, weight=class_weight)
                raw_anchor = anchor_loss(model, anchor)
                shrink = torch.tanh(model.lambda_channel).square()
                loss = ce + ANCHOR * raw_anchor + regime.gate_shrinkage * shrink
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite FT loss: {regime.name}/{task}/fold{fold_id}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], CLIP)
            scaler.step(optimizer)
            scaler.update()
            ce_values.append(float(ce.detach().float().cpu()))
            anchor_values.append(float(raw_anchor.detach().float().cpu()))
            shrink_values.append(float(shrink.detach().float().cpu()))
        values = mod.evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        ba = float(np.mean([value["BA"] for value in values.values()]))
        selected = epoch >= MIN_EPOCH and ba > best_ba + TOL
        if selected:
            best_ba, best_epoch, best_state, stalled = ba, epoch, copy.deepcopy(model.state_dict()), 0
        elif epoch >= MIN_EPOCH:
            stalled += 1
        snapshot = {
            "epoch": epoch,
            "inner_val_BA": ba,
            "cross_entropy": float(np.mean(ce_values)),
            "anchor_raw": float(np.mean(anchor_values)),
            "gate_shrink_raw": float(np.mean(shrink_values)),
            "lambda_channel": float(model.lambda_channel.detach().float().cpu()),
            "tanh_lambda_channel": float(torch.tanh(model.lambda_channel).detach().float().cpu()),
            "selected": selected,
            "stalled": stalled,
        }
        history.append(snapshot)
        atomic_torch_save(latest_path, {
            "invariant": invariant,
            "epoch": epoch,
            "best_ba": best_ba,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "current_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_state(),
            "history": history,
            "stalled": stalled,
        })
        print(f"[{regime.name} {task} f{fold_id}] epoch={epoch:02d} valBA={ba:.5f} gate={snapshot['tanh_lambda_channel']:+.5f}", flush=True)
        if epoch >= MIN_EPOCH and stalled >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"no selected FT checkpoint: {regime.name}/{task}/fold{fold_id}")
    model.load_state_dict(best_state, strict=True)
    atomic_torch_save(selected_path, model.state_dict())
    final_state = torch.load(selected_path, map_location="cpu", weights_only=False)
    final_gate = gate_stats(model, cache, bundle, fold["inner_val_subjects"], mean, std, mod, task)
    total, trainable = parameter_count(model), parameter_count(model, trainable_only=True)
    drift_rows = []
    for component, prefix in (
        ("channel_mlp", "channel_mlp."),
        ("classifier_head", "head."),
        ("embedding", "embedding."),
        ("last_residual_temporal_mixer", "mixer.2."),
    ):
        absolute, relative = module_drift(raw_state, final_state, prefix)
        drift_rows.append({"component": component, "absolute_l2_drift": absolute, "relative_l2_drift": relative})
    output = {
        "invariant": invariant,
        "regime": regime.name,
        "task": task,
        "dataset": mod.TASKS[task]["dataset"],
        "fold": fold_id,
        "seed": SEED,
        "source_xs_checkpoint": str(source),
        "source_xs_sha256": digest(source),
        "fine_tune_initial_state_sha256": initial_state_sha,
        "fine_tuned_checkpoint": str(selected_path),
        "fine_tuned_checkpoint_sha256": digest(selected_path),
        "normalizer_sha256": norm_meta["mean_std_sha256"],
        "split_sha256": split_hash,
        "initial_XS_inner_val_BA": replay["replayed_BA"],
        "LiteBN_inner_val_BA": baseline_ba,
        "fine_tuned_inner_val_BA": best_ba,
        "selected_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "trainable_percent": 100.0 * trainable / total,
        "gate_shrinkage": regime.gate_shrinkage,
        "anchor_coefficient": ANCHOR,
        "initial_gate": initial_gate,
        "final_gate": final_gate,
        "parameter_drift": drift_rows,
        "history": history,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
    }
    write_json(record_path, output)
    del model, cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def stage_a_tables(records: list[dict[str, Any]], regimes: list[Regime]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.DataFrame(records)
    detail["delta_vs_LiteBN_pp"] = (detail.fine_tuned_inner_val_BA - detail.LiteBN_inner_val_BA) * 100.0
    detail["delta_vs_XS_pp"] = (detail.fine_tuned_inner_val_BA - detail.initial_XS_inner_val_BA) * 100.0
    rows = []
    for regime in regimes:
        subset = detail[detail.regime.eq(regime.name)]
        by_task = subset.groupby("task").delta_vs_LiteBN_pp.mean()
        row = {
            "regime": regime.name,
            "trainable_modules": ";".join(regime.trainable_modules),
            "frozen_modules": ";".join(regime.frozen_modules),
            "gate_shrinkage": regime.gate_shrinkage,
            "worst_task_delta_pp": float(by_task.min()),
            "equal_task_mean_delta_pp": float(by_task.mean()),
            "positive_tasks": int((by_task > 0).sum()),
            "positive_folds": int((subset.delta_vs_LiteBN_pp > 0).sum()),
            "eligible_4_of_4": bool((by_task > 0).all()),
            "simplicity_rank": regime.simplicity_rank,
        }
        for task in by_task.index:
            row[f"{task}_delta_pp"] = float(by_task.loc[task])
        rows.append(row)
    aggregate = pd.DataFrame(rows).sort_values(
        ["worst_task_delta_pp", "positive_tasks", "equal_task_mean_delta_pp", "positive_folds", "simplicity_rank"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return detail.sort_values(["regime", "task", "fold"]).reset_index(drop=True), aggregate


def select_winner(aggregate: pd.DataFrame) -> str:
    return str(aggregate.iloc[0].regime)


def stage_b(mod, records: list[dict[str, Any]], winner: Regime, device: torch.device) -> str:
    by_cell = {(record["task"], int(record["fold"])): record for record in records if record["regime"] == winner.name}
    _, folds, _ = mod.load_folds()
    rows, off_rows = [], []
    for task in mod.TASK_ORDER:
        for fold in folds[mod.TASKS[task]["dataset"]]:
            fold_id, outer = int(fold["fold_id"]), fold["outer_dev_subjects"]
            bundle = mod.build_bundle(task, outer)
            mean, std, normalizer_meta = mod.load_tensor_pair(normalizer_path(task, fold_id))
            cache = mod.RawGPUCache(bundle, device)
            paths = {
                "LiteBN_BASELINE": baseline_path(task, fold_id),
                "LiteBN_XS": source_xs_path(task, fold_id),
                "XS_FT": Path(by_cell[(task, fold_id)]["fine_tuned_checkpoint"]),
            }
            for method, path in paths.items():
                if method == "LiteBN_BASELINE":
                    model = mod.build_model("LiteBN_BASELINE", task).to(device)
                    model.load_state_dict(torch.load(path, map_location=device, weights_only=False), strict=True)
                    model.eval()
                elif method == "LiteBN_XS":
                    model = load_xs(mod, task, path, device)
                else:
                    model = load_ft(mod, task, source_xs_path(task, fold_id), path, device)
                results = mod.evaluate(model, bundle, cache, outer, mean, std)
                for subject, metrics in results.items():
                    rows.append({
                        "task": task,
                        "dataset": mod.TASKS[task]["dataset"],
                        "fold": fold_id,
                        "seed": SEED,
                        "subject_id": str(subject),
                        "method": method,
                        **metrics,
                        "checkpoint_sha256": digest(path),
                        "normalizer_sha256": normalizer_meta["mean_std_sha256"],
                    })
                if method == "XS_FT":
                    original = model.lambda_channel.detach().clone()
                    with torch.no_grad():
                        model.lambda_channel.zero_()
                    disabled = mod.evaluate(model, bundle, cache, outer, mean, std)
                    with torch.no_grad():
                        model.lambda_channel.copy_(original)
                    for subject, metrics in disabled.items():
                        off_rows.append({
                            "task": task,
                            "dataset": mod.TASKS[task]["dataset"],
                            "fold": fold_id,
                            "seed": SEED,
                            "subject_id": str(subject),
                            "method": "XS_FT_GATE_OFF",
                            **metrics,
                        })
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del cache
    subject = pd.DataFrame(rows).sort_values(["task", "fold", "subject_id", "method"])
    write_csv(OUT / "XS_FT_OUTER_SUBJECT_RESULTS.csv", subject)
    gate_off = pd.DataFrame(off_rows).sort_values(["task", "fold", "subject_id"])
    summary_rows, bootstrap_rows, off_summary_rows = [], [], []
    for task in mod.TASK_ORDER:
        subset = subject[subject.task.eq(task)]
        wide = subset.pivot(index=["fold", "subject_id"], columns="method", values=["BA", "macro_F1", "accuracy"])
        delta = (wide[("BA", "XS_FT")] - wide[("BA", "LiteBN_BASELINE")]).to_numpy(float) * 100.0
        delta_xs = (wide[("BA", "XS_FT")] - wide[("BA", "LiteBN_XS")]).to_numpy(float) * 100.0
        fold_delta = pd.Series(delta, index=wide.index).groupby(level=0).mean()
        rng = np.random.default_rng(SEED)
        samples = rng.choice(delta, size=(BOOTSTRAPS, len(delta)), replace=True).mean(axis=1)
        off = gate_off[gate_off.task.eq(task)].set_index(["fold", "subject_id"]).sort_index()
        normal = wide[("BA", "XS_FT")].sort_index()
        summary_rows.append({
            "task": task,
            "dataset": mod.TASKS[task]["dataset"],
            "LiteBN_BA": float(wide[("BA", "LiteBN_BASELINE")].mean()),
            "Original_XS_BA": float(wide[("BA", "LiteBN_XS")].mean()),
            "XS_FT_BA": float(wide[("BA", "XS_FT")].mean()),
            "delta_vs_LiteBN_pp": float(delta.mean()),
            "delta_vs_XS_pp": float(delta_xs.mean()),
            "ci_low_pp": float(np.quantile(samples, .025)),
            "ci_high_pp": float(np.quantile(samples, .975)),
            "LiteBN_macro_F1": float(wide[("macro_F1", "LiteBN_BASELINE")].mean()),
            "Original_XS_macro_F1": float(wide[("macro_F1", "LiteBN_XS")].mean()),
            "XS_FT_macro_F1": float(wide[("macro_F1", "XS_FT")].mean()),
            "LiteBN_accuracy": float(wide[("accuracy", "LiteBN_BASELINE")].mean()),
            "Original_XS_accuracy": float(wide[("accuracy", "LiteBN_XS")].mean()),
            "XS_FT_accuracy": float(wide[("accuracy", "XS_FT")].mean()),
            "positive_folds": int((fold_delta > 0).sum()),
            "harmed_folds": int((fold_delta < 0).sum()),
            "positive_subjects": int((delta > 0).sum()),
            "harmed_subjects": int((delta < 0).sum()),
            "tied_subjects": int((np.abs(delta) <= TOL).sum()),
        })
        bootstrap_rows.append({
            "task": task,
            "comparison": "XS_FT_minus_LiteBN",
            "mean_delta_pp": float(delta.mean()),
            "ci_low_pp": float(np.quantile(samples, .025)),
            "ci_high_pp": float(np.quantile(samples, .975)),
            "resamples": BOOTSTRAPS,
            "bootstrap_seed": SEED,
        })
        off_summary_rows.append({
            "task": task,
            "normal_gate_BA": float(normal.mean()),
            "gate_off_BA": float(off.BA.mean()),
            "gate_on_minus_off_pp": float((normal - off.BA).mean() * 100.0),
        })
    task_frame = pd.DataFrame(summary_rows)
    write_csv(OUT / "XS_FT_OUTER_TASK_RESULTS.csv", task_frame)
    write_csv(OUT / "XS_FT_PAIRED_BOOTSTRAP.csv", pd.DataFrame(bootstrap_rows))
    write_csv(OUT / "XS_FT_GATE_OFF_DIAGNOSTIC.csv", pd.DataFrame(off_summary_rows))
    write_csv(OUT / "XS_FT_GATE_OFF_SUBJECT_RESULTS.csv", gate_off)

    winner_cells = [item for item in records if item["regime"] == winner.name]
    diagnostics, drifts = [], []
    for item in winner_cells:
        before, after = item["initial_gate"], item["final_gate"]
        diagnostics.append({
            "task": item["task"], "dataset": item["dataset"], "fold": item["fold"], "selected_epoch": item["selected_epoch"],
            "lambda_before": before["lambda_channel"], "lambda_after": after["lambda_channel"],
            "signed_amplitude_before": before["signed_amplitude"], "signed_amplitude_after": after["signed_amplitude"],
            "absolute_amplitude_before": before["absolute_amplitude"], "absolute_amplitude_after": after["absolute_amplitude"],
            "amplitude_change": after["signed_amplitude"] - before["signed_amplitude"],
            "amplitude_percent_change": 100.0 * (after["absolute_amplitude"] - before["absolute_amplitude"]) / max(before["absolute_amplitude"], 1e-12),
            "mean_correction_before": before["mean_absolute_channel_correction"],
            "mean_correction_after": after["mean_absolute_channel_correction"],
            "median_correction_before": before["median_absolute_channel_correction"],
            "median_correction_after": after["median_absolute_channel_correction"],
            "max_correction_before": before["max_absolute_channel_correction"],
            "max_correction_after": after["max_absolute_channel_correction"],
        })
        for drift in item["parameter_drift"]:
            drifts.append({"task": item["task"], "dataset": item["dataset"], "fold": item["fold"], "regime": winner.name, **drift})
    write_csv(OUT / "XS_FT_GATE_DIAGNOSTICS_SEED0.csv", pd.DataFrame(diagnostics).sort_values(["task", "fold"]))
    write_csv(OUT / "XS_FT_PARAMETER_DRIFT.csv", pd.DataFrame(drifts).sort_values(["task", "fold", "component"]))

    positive_tasks = int((task_frame.delta_vs_LiteBN_pp > 0).sum())
    worst = float(task_frame.delta_vs_LiteBN_pp.min())
    mean_delta = float(task_frame.delta_vs_LiteBN_pp.mean())
    original_worst = float((task_frame.Original_XS_BA.sub(task_frame.LiteBN_BA) * 100.0).min())
    terminal = (
        "XS_FT_SEED0_BALANCED_SUCCESS" if positive_tasks == 4 and worst > 0
        else "XS_FT_SEED0_PARTIAL_SUCCESS" if positive_tasks >= 3 and worst > original_worst + 0.10
        else "XS_FT_SEED0_FAIL"
    )
    trainable = winner_cells[0]["trainable_parameter_count"]
    total = winner_cells[0]["total_parameter_count"]
    aggregate_gate = pd.DataFrame(diagnostics).groupby("task", as_index=False).agg(
        gate_before=("signed_amplitude_before", "mean"),
        gate_after=("signed_amplitude_after", "mean"),
        correction_before=("mean_correction_before", "mean"),
        correction_after=("mean_correction_after", "mean"),
    )
    report = [
        "# XS checkpoint fine-tuning seed-0 decision",
        "",
        "## 1. Checkpoint replay",
        "",
        "All 20 source checkpoints had deterministic repeated inner-validation replay. Stored historical inner metrics were matched whenever a committed historical row existed; the replay CSV records cells for which only checkpoint provenance was retained.",
        "",
        "## 2. Winning regime",
        "",
        winner.name,
        "",
        "## 3. Updated parameters",
        "",
        f"Total parameters: {total}. Trainable parameters: {trainable} ({100.0 * trainable / total:.3f}%).",
        "",
        "## 4. Seed-0 outer-development result",
        "",
        task_frame.to_csv(index=False),
        "",
        "## 5. Four-task robustness",
        "",
        f"Positive tasks: {positive_tasks}/4. Positive folds: {int(task_frame.positive_folds.sum())}/20. Equal-task mean delta: {mean_delta:+.3f} pp. Worst-task delta: {worst:+.3f} pp.",
        "",
        "## 6. Learned channel correction",
        "",
        aggregate_gate.to_csv(index=False),
        "",
        "## 7. Gate-off diagnostic",
        "",
        pd.DataFrame(off_summary_rows).to_csv(index=False),
        "",
        "## 8. Final terminal",
        "",
        terminal,
        "",
        "FINAL_HELDOUT_ACCESSED = NO",
    ]
    (OUT / "FINAL_XS_FT_SEED0_DECISION.md").write_text("\n".join(report), encoding="utf-8")
    return terminal


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    _, folds, split_hash = mod.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = {
        "name": "XS_CHECKPOINT_LOCAL_FINE_TUNING_SEED0_V1",
        "seed": SEED,
        "scope": "SEARCH_AND_OUTER_DEVELOPMENT_ONLY",
        "final_heldout_accessed": False,
        "source_initialization": "existing LiteBN_XS selected checkpoint only",
        "forbidden": ["random_initialization", "new_alpha", "fixed_alpha_sweep", "ensemble", "routing", "TTA", "final_heldout"],
        "stage_a_selection": "maximize worst-task delta vs LiteBN; then positive tasks; equal-task mean; positive folds; simpler regime",
        "ft1": {"trainable": FT1.trainable_modules, "frozen": FT1.frozen_modules},
        "hyperparameters": {
            "max_epochs": MAX_EPOCHS, "min_epoch": MIN_EPOCH, "patience": PATIENCE,
            "gate_lr": GATE_LR, "head_lr": HEAD_LR, "embedding_lr": EMBEDDING_LR, "mixer_lr": MIXER_LR,
            "gradient_clip": CLIP, "anchor": ANCHOR, "weak_gate_shrinkage": WEAK_GATE_SHRINK,
        },
        "split_sha256": split_hash,
        "code_commit": git_commit(),
    }
    write_json(PROTOCOL / "XS_FT_PROTOCOL.json", protocol)
    make_manifest(mod, folds, split_hash, device)
    replay = cached_replay_if_valid(mod, folds)
    if replay is None:
        _, replay = replay_all(mod, folds, load_source_history(), device)
    else:
        print("XS_CHECKPOINT_REPLAY_REUSED 20/20", flush=True)
    baseline = baseline_inner_all(mod, folds, device)

    records: list[dict[str, Any]] = []
    attempted: list[Regime] = []
    for regime in (FT1, FT2, FT3):
        attempted.append(regime)
        for task in mod.TASK_ORDER:
            for fold in folds[mod.TASKS[task]["dataset"]]:
                fold_id = int(fold["fold_id"])
                records.append(train_cell(mod, task, fold, regime, replay[(task, fold_id)], baseline[(task, fold_id)], split_hash, device))
        detail, aggregate = stage_a_tables(records, attempted)
        if regime == FT1:
            write_csv(OUT / "XS_FT1_INNERVAL_TASK_FOLD.csv", detail[detail.regime.eq(FT1.name)])
            write_csv(OUT / "XS_FT1_INNERVAL_AGGREGATE.csv", aggregate[aggregate.regime.eq(FT1.name)])
        if bool(aggregate[aggregate.regime.eq(regime.name)].iloc[0].eligible_4_of_4):
            break

    detail, aggregate = stage_a_tables(records, attempted)
    write_csv(OUT / "XS_FT_STAGE_A_ALL_CANDIDATES.csv", detail)
    write_csv(OUT / "XS_FT_STAGE_A_SUMMARY.csv", detail)
    write_csv(OUT / "XS_FT_STAGE_A_AGGREGATE.csv", aggregate)
    winner_name = select_winner(aggregate)
    winner = next(regime for regime in attempted if regime.name == winner_name)
    winner_cells = [item for item in records if item["regime"] == winner_name]
    lock = {
        **protocol,
        "stage_a_complete": True,
        "selected_regime": winner_name,
        "selected_aggregate": clean(aggregate[aggregate.regime.eq(winner_name)].iloc[0].to_dict()),
        "trainable_modules": winner.trainable_modules,
        "frozen_modules": winner.frozen_modules,
        "source_checkpoint_policy": "strict XS checkpoint replay then local adaptation",
        "source_xs_checkpoints": [{"task": item["task"], "fold": item["fold"], "path": item["source_xs_checkpoint"], "sha256": item["source_xs_sha256"]} for item in winner_cells],
        "normalizers": [{"task": item["task"], "fold": item["fold"], "sha256": item["normalizer_sha256"]} for item in winner_cells],
        "outer_development_predictions_generated": False,
        "final_heldout_accessed": False,
    }
    write_json(OUT / "XS_FT_SEED0_STAGE_A_LOCK.json", lock)
    print(f"XS_FT_STAGE_A_LOCKED {winner_name}", flush=True)
    terminal = stage_b(mod, records, winner, device)
    lock["outer_development_predictions_generated"] = True
    lock["terminal"] = terminal
    write_json(OUT / "XS_FT_SEED0_STAGE_A_LOCK.json", lock)
    print(f"XS_FT_STAGE_B_DONE {terminal}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
