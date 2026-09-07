"""Target-history encoder fine-tuning with Fisher-protected PERSIST control."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, log_loss

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import ABLATIONS, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V6_SEED, stable_seed, stage0_root, wbcic_source_root, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import OPENBMI_BEST_EPOCHS, load_fold


@dataclass
class RawSubject:
    history_x: np.ndarray
    history_y: np.ndarray
    future_x: np.ndarray
    future_y: np.ndarray
    future_uid: np.ndarray


class OpenBMIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 63), padding=(0, 31), bias=False),
            nn.BatchNorm2d(16),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(62, 1), groups=16, bias=False),
            nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(32, 128), nn.LayerNorm(128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.separable(self.spatial(self.temporal(x.unsqueeze(1))))))


class OpenBMIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = OpenBMIEncoder()
        self.heads = nn.ModuleDict({"mi": nn.Linear(128, 2), "erp": nn.Linear(128, 2), "ssvep": nn.Linear(128, 4)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads["mi"](self.encoder(x))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _openbmi_source(fold: int) -> tuple[nn.Module, dict[str, torch.Tensor], np.ndarray, np.ndarray, Path]:
    root = stage0_root()
    checkpoint = root / "outputs" / "persist_eeg_p2p3" / "backbone" / "checkpoints" / "eegnet" / f"fold-{fold}" / "seed-0" / "trajectory" / f"epoch-{OPENBMI_BEST_EPOCHS[fold]:03d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["fold"]) != fold or payload.get("label_maps", {}).get("mi") != {"left_hand": 0, "right_hand": 1}:
        raise RuntimeError("OpenBMI source checkpoint mismatch")
    model = OpenBMIModel()
    model.load_state_dict(payload["model"], strict=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return model, state, np.asarray(payload["channel_mean"], dtype=np.float32), np.asarray(payload["channel_std"], dtype=np.float32), checkpoint


def _wbcic_source(fold: int) -> tuple[nn.Module, dict[str, torch.Tensor], None, None, Path]:
    source = wbcic_source_root()
    core = _load_module(source / "experiments" / "persist_eeg_wbcic_actionability_v2" / "code" / "core.py", f"v6_finetune_wbcic_core_{fold}")
    checkpoint = source / "experiments" / "persist_eeg_wbcic_actionability_v2" / "outputs" / "model" / "competence" / f"EEGNET_STABLE_fold-{fold}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["fold_label"].rsplit("-", 1)[-1]) != fold or list(map(int, payload["train_sessions"])) != [0, 1]:
        raise RuntimeError("WBCIC source checkpoint mismatch")
    model = core.EEGNet(float(payload["config"]["dropout"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return model, state, None, None, checkpoint


def _load_openbmi_raw(subject: str, mean: np.ndarray, std: np.ndarray) -> RawSubject:
    root = stage0_root()
    manifest = pd.read_parquet(root / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet")
    rows = manifest.loc[manifest.paradigm.eq("mi") & manifest.subject_id.astype(str).eq(str(subject))].copy()
    arrays = []
    labels = []
    sessions = []
    uids = []
    for relative, group in rows.sort_values(["session_id", "cache_index"]).groupby("signal_cache_path", sort=False):
        source = np.load(root / str(relative), mmap_mode="r", allow_pickle=False)
        index = group.cache_index.to_numpy(int)
        arrays.extend(np.asarray(source[index], dtype=np.float32))
        labels.extend(group.event_label.astype(str).map({"left_hand": 0, "right_hand": 1}).astype(int).tolist())
        sessions.extend(group.session_id.astype(int).tolist())
        uids.extend(("OpenBMI_nm000273_MI:" + group.trial_id.astype(str)).tolist())
    x = np.stack(arrays)
    x = (x - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-8)
    labels_array = np.asarray(labels, dtype=np.int64)
    sessions_array = np.asarray(sessions, dtype=int)
    uid_array = np.asarray(uids)
    return RawSubject(x[sessions_array == 1], labels_array[sessions_array == 1], x[sessions_array == 2], labels_array[sessions_array == 2], uid_array[sessions_array == 2])


def _load_wbcic_raw(subject: str) -> RawSubject:
    root = wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2" / "outputs" / "cache" / "wbcic_epochs" / subject
    histories_x, histories_y = [], []
    for source_session in (0, 1):
        histories_x.append(np.asarray(np.load(root / f"ses-{source_session}_epochs.npy", mmap_mode="r", allow_pickle=False), dtype=np.float32))
        histories_y.append(np.load(root / f"ses-{source_session}_labels.npy", allow_pickle=False).astype(np.int64))
    future_x = np.asarray(np.load(root / "ses-2_epochs.npy", mmap_mode="r", allow_pickle=False), dtype=np.float32)
    future_y = np.load(root / "ses-2_labels.npy", allow_pickle=False).astype(np.int64)
    uid = np.asarray([f"WBCIC_nm000348_dev:{subject}:S3:{index}" for index in range(len(future_y))])
    return RawSubject(np.concatenate(histories_x), np.concatenate(histories_y), future_x, future_y, uid)


def _configs(benchmark: str) -> list[dict[str, Any]]:
    epoch_short, epoch_long = ((8, 24) if benchmark == "wbcic" else (10, 30))
    return [
        {"strategy": "frozen", "lr": 0.0, "epochs": 0, "augment": False},
        {"strategy": "adabn", "lr": 0.0, "epochs": 1, "augment": False},
        {"strategy": "head", "lr": 1e-3, "epochs": epoch_short, "augment": False},
        {"strategy": "head", "lr": 3e-4, "epochs": epoch_long, "augment": False},
        {"strategy": "tail", "lr": 1e-4, "epochs": epoch_short, "augment": False},
        {"strategy": "tail", "lr": 3e-4, "epochs": epoch_short, "augment": False},
        {"strategy": "tail", "lr": 1e-4, "epochs": epoch_long, "augment": True},
        {"strategy": "spatial_tail", "lr": 3e-5, "epochs": epoch_short, "augment": False},
        {"strategy": "spatial_tail", "lr": 1e-4, "epochs": epoch_short, "augment": False},
        {"strategy": "spatial_tail", "lr": 3e-5, "epochs": epoch_long, "augment": True},
        {"strategy": "full", "lr": 1e-5, "epochs": epoch_short, "augment": False},
        {"strategy": "full", "lr": 3e-5, "epochs": epoch_short, "augment": False},
        {"strategy": "full", "lr": 1e-4, "epochs": epoch_short, "augment": False},
        {"strategy": "full", "lr": 3e-5, "epochs": epoch_long, "augment": True},
    ]


def _trainable(model: nn.Module, benchmark: str, strategy: str) -> list[tuple[str, nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        if benchmark == "openbmi":
            keep = (
                strategy == "full"
                or (strategy == "head" and name.startswith("heads.mi"))
                or (strategy == "tail" and (name.startswith("encoder.projection") or name.startswith("heads.mi")))
                or (strategy == "spatial_tail" and (name.startswith("encoder.spatial") or name.startswith("encoder.separable") or name.startswith("encoder.projection") or name.startswith("heads.mi")))
            )
        else:
            keep = (
                strategy == "full"
                or (strategy == "head" and name.startswith("head"))
                or (strategy == "tail" and (name.startswith("embedding") or name.startswith("embedding_norm") or name.startswith("head")))
                or (strategy == "spatial_tail" and (name.startswith("spatial") or name.startswith("separable") or name.startswith("bn2") or name.startswith("bn3") or name.startswith("embedding") or name.startswith("head")))
            )
        if keep:
            parameter.requires_grad_(True)
            selected.append((name, parameter))
    return selected


def _augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    value = x.clone()
    if value.shape[-1] > 50:
        shift = int(torch.randint(-25, 26, (1,), generator=generator, device="cpu"))
        value = torch.roll(value, shift, dims=-1)
    noise = torch.randn(value.shape, generator=generator, device="cpu", dtype=value.dtype).to(value.device)
    value = value + 0.01 * noise
    channel_mask = torch.rand((value.shape[0], value.shape[1], 1), generator=generator, device="cpu").to(value.device) > 0.03
    return value * channel_mask


def _predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            logits = model(torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device))
            values.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(values)


def _fisher(model: nn.Module, trainable: list[tuple[str, nn.Parameter]], x: np.ndarray, y: np.ndarray, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    output = {name: torch.zeros_like(parameter) for name, parameter in trainable}
    model.eval()
    for start in range(0, len(x), 64):
        model.zero_grad(set_to_none=True)
        xb = torch.as_tensor(x[start : start + 64], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y[start : start + 64], dtype=torch.long, device=device)
        F.cross_entropy(model(xb), yb).backward()
        for name, parameter in trainable:
            if parameter.grad is not None:
                output[name] += parameter.grad.detach().square() * len(yb)
    for name in output:
        output[name] /= max(len(x), 1)
    flat_mean = torch.cat([value.flatten() for value in output.values()]).mean().clamp_min(1e-12)
    return {name: value / flat_mean for name, value in output.items()}


def _adapt(
    benchmark: str,
    source_model: nn.Module,
    source_state: dict[str, torch.Tensor],
    raw: RawSubject,
    configuration: dict[str, Any],
    device: torch.device,
    seed: int,
    protection: str = "none",
    strength: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = copy.deepcopy(source_model).to(device)
    model.load_state_dict(source_state, strict=True)
    strategy = str(configuration["strategy"])
    if strategy == "frozen":
        return _predict(model, raw.future_x, device), {"relative_parameter_change": 0.0}
    if strategy == "adabn":
        model.train()
        with torch.inference_mode():
            for start in range(0, len(raw.history_x), 64):
                model(torch.as_tensor(raw.history_x[start : start + 64], dtype=torch.float32, device=device))
        return _predict(model, raw.future_x, device), {"relative_parameter_change": 0.0}
    trainable = _trainable(model, benchmark, strategy)
    if not trainable:
        raise RuntimeError(f"No trainable parameters for {benchmark}/{strategy}")
    initial = {name: parameter.detach().clone() for name, parameter in trainable}
    importance = None
    if protection in {"fisher", "random_fisher", "uniform"}:
        importance = _fisher(model, trainable, raw.history_x, raw.history_y, device, seed)
        if protection == "uniform":
            importance = {name: torch.ones_like(value) for name, value in importance.items()}
        elif protection == "random_fisher":
            rng = np.random.default_rng(seed)
            importance = {
                name: value.flatten()[torch.as_tensor(rng.permutation(value.numel()), device=value.device)].reshape_as(value)
                for name, value in importance.items()
            }
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=float(configuration["lr"]), weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(int(configuration["epochs"])):
        model.train()
        permutation = torch.randperm(len(raw.history_y), generator=generator)
        for start in range(0, len(permutation), 32):
            index = permutation[start : start + 32]
            xb = torch.as_tensor(raw.history_x[index], dtype=torch.float32, device=device)
            yb = torch.as_tensor(raw.history_y[index], dtype=torch.long, device=device)
            if bool(configuration.get("augment", False)):
                xb = _augment(xb, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            if importance is not None and strength > 0:
                numerator = sum((importance[name] * (parameter - initial[name]).square()).sum() for name, parameter in trainable)
                denominator = sum(value.sum() for value in importance.values()).clamp_min(1e-12)
                loss = loss + float(strength) * numerator / denominator
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 5.0)
            optimizer.step()
    changed = torch.sqrt(sum((parameter.detach() - initial[name]).square().sum() for name, parameter in trainable))
    base = torch.sqrt(sum(initial[name].square().sum() for name, _ in trainable)).clamp_min(1e-12)
    return _predict(model, raw.future_x, device), {"relative_parameter_change": float((changed / base).cpu())}


def _score(records: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    ba = [balanced_accuracy_score(y, p >= 0.5) for y, p in records]
    nll = [log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]) for y, p in records]
    return float(np.mean(ba)), float(np.mean(nll))


# The first WBCIC search completed all five discovery folds on 2026-08-20, but
# the old summarizer rejected the protection controls because fold 2 selected
# the frozen model and therefore did not emit those method IDs.  These are the
# fold-wise choices printed by that completed search.  Recovery only replays
# outcome subjects; it does not re-select against outcome labels.
WBCIC_DISCOVERY_RECOVERY_SPECS: dict[int, dict[str, dict[str, Any]]] = {
    0: {
        "FT_GENERIC_SELECTED": {"configuration": {"strategy": "full", "lr": 1e-5, "epochs": 8, "augment": False}, "protection": "none", "strength": 0.0},
        "FT_UNIFORM_L2_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-5, "epochs": 8, "augment": False}, "protection": "uniform", "strength": 1000.0},
        "FT_RANDOM_FISHER_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-5, "epochs": 8, "augment": False}, "protection": "random_fisher", "strength": 10.0},
        "PERSIST_SA_FISHER_PROTECTED": {"configuration": {"strategy": "full", "lr": 1e-5, "epochs": 8, "augment": False}, "protection": "fisher", "strength": 0.1},
    },
    1: {
        "FT_GENERIC_SELECTED": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "none", "strength": 0.0},
        "FT_UNIFORM_L2_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "uniform", "strength": 10.0},
        "FT_RANDOM_FISHER_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "random_fisher", "strength": 0.1},
        "PERSIST_SA_FISHER_PROTECTED": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "fisher", "strength": 0.1},
    },
    2: {
        method: {"configuration": {"strategy": "frozen", "lr": 0.0, "epochs": 0, "augment": False}, "protection": "none", "strength": 0.0}
        for method in ("FT_GENERIC_SELECTED", "FT_UNIFORM_L2_CONTROL", "FT_RANDOM_FISHER_CONTROL", "PERSIST_SA_FISHER_PROTECTED")
    },
    3: {
        "FT_GENERIC_SELECTED": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "none", "strength": 0.0},
        "FT_UNIFORM_L2_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "uniform", "strength": 10.0},
        "FT_RANDOM_FISHER_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "random_fisher", "strength": 1000.0},
        "PERSIST_SA_FISHER_PROTECTED": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "fisher", "strength": 1.0},
    },
    4: {
        "FT_GENERIC_SELECTED": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "none", "strength": 0.0},
        "FT_UNIFORM_L2_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "uniform", "strength": 10.0},
        "FT_RANDOM_FISHER_CONTROL": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "random_fisher", "strength": 100.0},
        "PERSIST_SA_FISHER_PROTECTED": {"configuration": {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False}, "protection": "fisher", "strength": 1.0},
    },
}


def run_wbcic_outcome_recovery(device_name: str) -> None:
    """Replay only locked outcome configurations after the coverage-only crash."""
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    prediction_parts: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    checkpoints = []
    for fold in range(5):
        data = load_fold("wbcic", fold)
        source_model, source_state, _, _, checkpoint = _wbcic_source(fold)
        checkpoints.append({"fold": fold, "checkpoint": str(checkpoint)})
        raw_cache = {subject: _load_wbcic_raw(subject) for subject in data.outcome_subjects}
        fold_specs = {
            "FT_FROZEN_EEGNET": {"configuration": {"strategy": "frozen", "lr": 0.0, "epochs": 0, "augment": False}, "protection": "none", "strength": 0.0},
            **WBCIC_DISCOVERY_RECOVERY_SPECS[fold],
        }
        for method_id, specification in fold_specs.items():
            changes = []
            for subject in data.outcome_subjects:
                raw = raw_cache[subject]
                probability, audit = _adapt(
                    "wbcic", source_model, source_state, raw, specification["configuration"], device,
                    # Paired controls must share the identical minibatch order;
                    # otherwise the claimed Fisher increment is confounded by
                    # optimization noise.
                    stable_seed(V6_SEED, "wbcic", fold, "paired-protection-control", subject), specification["protection"], specification["strength"],
                )
                changes.append(audit["relative_parameter_change"])
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "benchmark": data.benchmark,
                            "method_id": method_id,
                            "trial_uid": raw.future_uid,
                            "subject_id": subject,
                            "outer_fold": fold,
                            "label": raw.future_y,
                            "probability": probability,
                            "prediction": (probability >= 0.5).astype(int),
                            "target_history_labels_used": method_id != "FT_FROZEN_EEGNET",
                            "target_future_labels_used_for_fit": False,
                            "exploratory": True,
                            "OUTER_TEST_USED": False,
                        }
                    )
                )
            selections.append(
                {
                    "benchmark": data.benchmark,
                    "outer_fold": fold,
                    "method_id": method_id,
                    "configuration": json.dumps(specification["configuration"], sort_keys=True),
                    "protection": specification["protection"],
                    "strength": specification["strength"],
                    "mean_outcome_parameter_change": float(np.mean(changes)),
                    "selection_source": "completed discovery search stdout before coverage-only summarization failure",
                    "target_future_labels_used_for_fit": False,
                    "OUTER_TEST_USED": False,
                }
            )
        print(f"[wbcic recovery] fold={fold} complete", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    baseline = pd.read_csv(DIAGNOSTICS / "WBCIC_BASELINE_PREDICTIONS.csv")
    reference = baseline.loc[baseline.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    rows, subject_parts, fold_parts = [], [], []
    for method in predictions.method_id.unique():
        frame = predictions.loc[predictions.method_id.eq(method)].copy()
        row, subject_result, fold_result = summarize(frame, reference=reference)
        rows.append(row); subject_parts.append(subject_result); fold_parts.append(fold_result)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / "WBCIC_ENCODER_FINETUNING.csv", table)
    write_csv(DIAGNOSTICS / "WBCIC_ENCODER_FINETUNING_SELECTIONS.csv", pd.DataFrame(selections))
    write_csv(DIAGNOSTICS / "WBCIC_ENCODER_FINETUNING_PREDICTIONS.csv", predictions)
    write_csv(DIAGNOSTICS / "WBCIC_ENCODER_FINETUNING_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_ENCODER_FINETUNING_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(ABLATIONS / "WBCIC_FISHER_PROTECTION_ABLATION.csv", table)
    write_json(
        PROTOCOL / "WBCIC_ENCODER_FINETUNING_AUDIT.json",
        {
            "source_checkpoints": checkpoints,
            "recovery_reason": "The complete discovery search finished, then summarization failed because frozen fold 2 omitted protection method IDs.",
            "recovery_scope": "locked outcome configurations only; no outcome label was used for selection",
            "paired_control_seed": "identical minibatch permutation for generic, uniform, random-Fisher, and Fisher methods within fold/subject",
            "target_history_labels_used_for_adaptation": True,
            "outcome_future_labels_used_for_adaptation_or_selection": False,
            "exploratory": True,
            "OUTER_TEST_USED": False,
        },
    )
    print(table.to_string(index=False), flush=True)


def run(benchmark: str, device_name: str) -> None:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    predictions, selections = [], []
    checkpoint_records = []
    for fold in range(5):
        data = load_fold(benchmark, fold)
        if benchmark == "openbmi":
            source_model, source_state, mean, std, checkpoint = _openbmi_source(fold)
            raw_cache = {subject: _load_openbmi_raw(subject, mean, std) for subject in set(data.discovery_subjects) | set(data.outcome_subjects)}
        else:
            source_model, source_state, _, _, checkpoint = _wbcic_source(fold)
            raw_cache = {subject: _load_wbcic_raw(subject) for subject in set(data.discovery_subjects) | set(data.outcome_subjects)}
        checkpoint_records.append({"fold": fold, "checkpoint": str(checkpoint), "source_future_target_subjects_used": False})
        generic_rows = []
        for order, configuration in enumerate(_configs(benchmark)):
            records = []
            changes = []
            for subject in data.discovery_subjects:
                raw = raw_cache[subject]
                probability, audit = _adapt(benchmark, source_model, source_state, raw, configuration, device, stable_seed(V6_SEED, benchmark, fold, "generic", order, subject))
                records.append((raw.future_y, probability))
                changes.append(audit["relative_parameter_change"])
            ba, nll = _score(records)
            generic_rows.append({"configuration": configuration, "order": order, "discovery_BA": ba, "discovery_NLL": nll, "mean_parameter_change": float(np.mean(changes))})
            print(f"[{benchmark}] fold={fold} generic={order + 1}/{len(_configs(benchmark))} BA={ba:.4f} {configuration}", flush=True)
        generic = max(generic_rows, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["order"]))
        protection_rows = []
        if generic["configuration"]["strategy"] not in {"frozen", "adabn"}:
            for protection in ("uniform", "random_fisher", "fisher"):
                for strength in (0.1, 1.0, 10.0, 100.0, 1000.0):
                    records = []
                    changes = []
                    for subject in data.discovery_subjects:
                        raw = raw_cache[subject]
                        probability, audit = _adapt(
                            benchmark, source_model, source_state, raw, generic["configuration"], device,
                            stable_seed(V6_SEED, benchmark, fold, protection, strength, subject), protection, strength,
                        )
                        records.append((raw.future_y, probability))
                        changes.append(audit["relative_parameter_change"])
                    ba, nll = _score(records)
                    protection_rows.append({"protection": protection, "strength": strength, "discovery_BA": ba, "discovery_NLL": nll, "mean_parameter_change": float(np.mean(changes))})
                    print(f"[{benchmark}] fold={fold} protection={protection}/{strength:g} BA={ba:.4f}", flush=True)
        persist_candidates = [row for row in protection_rows if row["protection"] == "fisher"]
        random_candidates = [row for row in protection_rows if row["protection"] == "random_fisher"]
        uniform_candidates = [row for row in protection_rows if row["protection"] == "uniform"]
        chosen = {
            "FT_FROZEN_EEGNET": {"configuration": {"strategy": "frozen", "lr": 0.0, "epochs": 0, "augment": False}, "protection": "none", "strength": 0.0},
            "FT_GENERIC_SELECTED": {"configuration": generic["configuration"], "protection": "none", "strength": 0.0},
        }
        for name, rows, protection in (
            ("FT_UNIFORM_L2_CONTROL", uniform_candidates, "uniform"),
            ("FT_RANDOM_FISHER_CONTROL", random_candidates, "random_fisher"),
            ("PERSIST_SA_FISHER_PROTECTED", persist_candidates, "fisher"),
        ):
            if rows:
                selected = max(rows, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["strength"]))
                chosen[name] = {"configuration": generic["configuration"], "protection": protection, "strength": selected["strength"]}
            else:
                # If discovery selects a non-trainable frozen/AdaBN strategy,
                # the protected controls collapse exactly to that strategy.
                # Emit them anyway so every method retains all five folds.
                chosen[name] = {"configuration": generic["configuration"], "protection": "none", "strength": 0.0}
        for method_id, specification in chosen.items():
            outcome_changes = []
            for subject in data.outcome_subjects:
                raw = raw_cache[subject]
                probability, audit = _adapt(
                    benchmark, source_model, source_state, raw, specification["configuration"], device,
                    stable_seed(V6_SEED, benchmark, fold, "outcome-paired-control", subject), specification["protection"], specification["strength"],
                )
                outcome_changes.append(audit["relative_parameter_change"])
                predictions.append(
                    pd.DataFrame(
                        {
                            "benchmark": data.benchmark,
                            "method_id": method_id,
                            "trial_uid": raw.future_uid,
                            "subject_id": subject,
                            "outer_fold": fold,
                            "label": raw.future_y,
                            "probability": probability,
                            "prediction": (probability >= 0.5).astype(int),
                            "target_history_labels_used": method_id != "FT_FROZEN_EEGNET",
                            "target_future_labels_used_for_fit": False,
                            "exploratory": True,
                            "OUTER_TEST_USED": False,
                        }
                    )
                )
            selection_entry = {
                "benchmark": data.benchmark,
                "outer_fold": fold,
                "method_id": method_id,
                "configuration": json.dumps(specification["configuration"], sort_keys=True),
                "protection": specification["protection"],
                "strength": specification["strength"],
                "mean_outcome_parameter_change": float(np.mean(outcome_changes)),
                "target_future_labels_used_for_fit": False,
                "OUTER_TEST_USED": False,
            }
            selections.append(selection_entry)
        for row in generic_rows:
            selections.append({"benchmark": data.benchmark, "outer_fold": fold, "method_id": "GENERIC_CANDIDATE", "configuration": json.dumps(row["configuration"], sort_keys=True), **{key: value for key, value in row.items() if key != "configuration"}, "target_future_labels_used_for_fit": False, "OUTER_TEST_USED": False})
        for row in protection_rows:
            selections.append({"benchmark": data.benchmark, "outer_fold": fold, "method_id": "PROTECTION_CANDIDATE", "configuration": json.dumps(generic["configuration"], sort_keys=True), **row, "target_future_labels_used_for_fit": False, "OUTER_TEST_USED": False})
    prediction_frame = pd.concat(predictions, ignore_index=True)
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    baseline = pd.read_csv(DIAGNOSTICS / f"{prefix}_BASELINE_PREDICTIONS.csv")
    reference = baseline.loc[baseline.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    rows, subject_parts, fold_parts = [], [], []
    for method in prediction_frame.method_id.unique():
        row, subject_result, fold_result = summarize(prediction_frame.loc[prediction_frame.method_id.eq(method)].copy(), reference=reference)
        rows.append(row)
        subject_parts.append(subject_result)
        fold_parts.append(fold_result)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / f"{prefix}_ENCODER_FINETUNING.csv", table)
    write_csv(DIAGNOSTICS / f"{prefix}_ENCODER_FINETUNING_SELECTIONS.csv", pd.DataFrame(selections))
    write_csv(DIAGNOSTICS / f"{prefix}_ENCODER_FINETUNING_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / f"{prefix}_ENCODER_FINETUNING_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / f"{prefix}_ENCODER_FINETUNING_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(ABLATIONS / f"{prefix}_FISHER_PROTECTION_ABLATION.csv", table)
    write_json(
        PROTOCOL / f"{prefix}_ENCODER_FINETUNING_AUDIT.json",
        {
            "source_checkpoints": checkpoint_records,
            "target_history_labels_used_for_adaptation": True,
            "outcome_future_labels_used_for_adaptation_or_selection": False,
            "hyperparameters_selected_on": "discovery-subject history-to-future episodes",
            "PERSIST_protection": "target-history diagonal empirical Fisher, normalized within trainable parameters",
            "equal_capacity_controls": ["unprotected", "uniform L2", "randomly permuted Fisher"],
            "exploratory": True,
            "OUTER_TEST_USED": False,
        },
    )
    (RESEARCH_LOG / f"ITERATION_002_{prefix}.md").write_text(
        f"# Iteration 002 — {prefix} encoder fine-tuning\n\n"
        "- Previous failure: frozen-embedding conditional adapters did not beat matched controls.\n"
        "- Structural change: adapt the EEG encoder itself from legal target-history trials.\n"
        "- PERSIST change: parameter-level empirical-Fisher protection with uniform and shuffled controls.\n\n"
        f"```text\n{table.to_string(index=False)}\n```\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recover-outcome-only", action="store_true")
    args = parser.parse_args()
    if args.recover_outcome_only:
        if args.benchmark != "wbcic":
            raise ValueError("Outcome-only recovery is defined only for the completed WBCIC search")
        run_wbcic_outcome_recovery(args.device)
    else:
        run(args.benchmark, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
