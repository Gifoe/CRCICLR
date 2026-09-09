"""Outcome-blind, single-model LiteBN-X seed-0 experiment.

Stage A only opens inner-train/inner-validation SEARCH rows. Stage B is an
explicit post-freeze operation and opens only the frozen outer-development rows.
No final held-out cohort is enumerated, loaded, or scored by this program.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
import json
import math
import os
import random
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

REPO = Path(os.environ.get("LITEBN_X_REPO", "/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_litebn_x_singlemodel_seed0_v1"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("LITEBN_X_RUNTIME", "/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")).resolve()
CACHE = Path(os.environ.get("PERSIST_CACHE_ROOT", "/root/rivermind-data/persist_eeg_cache")).resolve()
FIVEFOLD = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"
CARRIER_CODE = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"

SEED, EPOCHS, MIN_EPOCH = 0, 60, 10
LR, WEIGHT_DECAY, CLIP = 3e-4, 5e-4, 5.0
TASK_BATCH, MI_EPISODE_BATCH, BOOTSTRAPS, TIE_TOL = 64, 128, 10_000, 1e-12

TASKS: dict[str, dict[str, Any]] = {
    "OpenBMI_MI": {"dataset": "OpenBMI", "cache": "mi", "classes": 2, "samples": 1000, "channels": 62, "source_sessions": (1,), "future_session": 2, "weighted_ce": False, "mi_protocol": True, "raw_codes": {1: 0, 2: 1}},
    "OpenBMI_ERP": {"dataset": "OpenBMI", "cache": "erp", "classes": 2, "samples": 250, "channels": 62, "source_sessions": (1,), "future_session": 2, "weighted_ce": True, "mi_protocol": False, "raw_codes": {1: 0, 2: 1}},
    "OpenBMI_SSVEP": {"dataset": "OpenBMI", "cache": "ssvep", "classes": 4, "samples": 1000, "channels": 62, "source_sessions": (1,), "future_session": 2, "weighted_ce": False, "mi_protocol": False, "raw_codes": {1: 0, 2: 1, 3: 2, 4: 3}},
    "WBCIC_MI": {"dataset": "WBCIC", "cache": "mi", "classes": 2, "samples": 1000, "channels": 58, "source_sessions": (0, 1), "future_session": 2, "weighted_ce": False, "mi_protocol": True, "raw_codes": {0: 0, 1: 1}},
}
TASK_ORDER = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
ARCHITECTURES = ("LiteBN_BASELINE", "LiteBN_R", "LiteBN_RG", "LiteBN_X", "LiteBN_XS")
CANDIDATES = ARCHITECTURES[1:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].detach().cpu())
    if "cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in value["cuda"]])


def state_hash(model: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return sha256_bytes(buffer.getvalue())


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def subject_sort(values: Iterable[str], dataset: str) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        text = str(value).replace("sub-", "")
        return (int(text) if text.isdigit() else 10**9, str(value))
    return sorted((str(value) for value in values), key=key)


def historical_litebn_class() -> type[nn.Module]:
    os.environ.setdefault("R2EEG_REPO", str(REPO))
    source = str(CARRIER_CODE)
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module("run_carrier_screen").CompactLite


def LiteBN_BASELINE(channels: int, classes: int) -> nn.Module:
    model = historical_litebn_class()(channels, "bn")
    model.head = nn.Linear(64, classes)
    return model


class ResidualTemporalBlock(nn.Module):
    def __init__(self, features: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(features, features, kernel_size=9, dilation=dilation, padding=4 * dilation, groups=features, bias=True)
        self.norm = nn.LayerNorm(features)
        self.expand = nn.Linear(features, 2 * features)
        self.dropout = nn.Dropout(0.10)
        self.contract = nn.Linear(2 * features, features)
        self.gamma = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(value).transpose(1, 2)
        update = self.norm(update)
        update = self.contract(self.dropout(F.gelu(self.expand(update))))
        return value + self.gamma * update.transpose(1, 2)


class LiteBNStem(nn.Module):
    def __init__(self, channels: int, temporal_channels: int, spatial_channels: int, features: int):
        super().__init__()
        if spatial_channels % temporal_channels:
            raise ValueError("spatial_channels must divide temporal_channels")
        self.temporal = nn.ModuleList([nn.Conv2d(1, temporal_channels, (1, kernel), padding="same", bias=False) for kernel in (15, 63, 127)])
        self.temporal_norm = nn.ModuleList([nn.BatchNorm2d(temporal_channels) for _ in range(3)])
        self.spatial = nn.ModuleList([nn.Conv2d(temporal_channels, spatial_channels, (channels, 1), groups=temporal_channels, bias=False) for _ in range(3)])
        self.spatial_norm = nn.ModuleList([nn.BatchNorm2d(spatial_channels) for _ in range(3)])
        concat = 3 * spatial_channels
        self.depth1 = nn.Conv2d(concat, concat, (1, 15), groups=concat, padding="same", bias=False)
        self.point1 = nn.Conv2d(concat, features, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(features)
        self.depth2 = nn.Conv2d(features, features, (1, 31), groups=features, padding="same", bias=False)
        self.point2 = nn.Conv2d(features, features, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(features)

    def branch_outputs(self, value: torch.Tensor) -> list[torch.Tensor]:
        value = value.unsqueeze(1)
        outputs: list[torch.Tensor] = []
        for temporal, temporal_norm, spatial, spatial_norm in zip(self.temporal, self.temporal_norm, self.spatial, self.spatial_norm):
            branch = F.elu(temporal_norm(temporal(value)))
            branch = F.elu(spatial_norm(spatial(branch)))
            branch = F.avg_pool2d(branch, (1, 4))
            outputs.append(F.dropout(branch, 0.20, self.training))
        return outputs

    def temporal_blocks(self, branches: list[torch.Tensor]) -> torch.Tensor:
        value = torch.cat(branches, dim=1)
        value = F.dropout(F.avg_pool2d(F.elu(self.norm1(self.point1(self.depth1(value)))), (1, 2)), 0.15, self.training)
        return F.dropout(F.avg_pool2d(F.elu(self.norm2(self.point2(self.depth2(value)))), (1, 2)), 0.15, self.training)


class LiteBNEnhanced(nn.Module):
    def __init__(self, channels: int, classes: int, architecture: str):
        super().__init__()
        if architecture not in CANDIDATES:
            raise ValueError(architecture)
        wide = architecture in ("LiteBN_X", "LiteBN_XS")
        temporal_channels, spatial_channels, features, embedding = (12, 24, 96, 128) if wide else (8, 16, 64, 96)
        self.architecture = architecture
        self.stem = LiteBNStem(channels, temporal_channels, spatial_channels, features)
        self.scale_gate = architecture in ("LiteBN_RG", "LiteBN_X", "LiteBN_XS")
        self.channel_gate = architecture == "LiteBN_XS"
        if self.scale_gate:
            descriptor = 3 * spatial_channels
            self.scale_mlp = nn.Sequential(nn.Linear(descriptor, max(12, descriptor // 2)), nn.GELU(), nn.Linear(max(12, descriptor // 2), 3))
            self.lambda_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        if self.channel_gate:
            self.channel_mlp = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 1))
            self.lambda_channel = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.mixer = nn.ModuleList([ResidualTemporalBlock(features, dilation) for dilation in (1, 2, 4)])
        self.pool_bins = nn.AdaptiveAvgPool1d(4)
        self.embedding = nn.Sequential(nn.Linear(6 * features, embedding), nn.GELU(), nn.LayerNorm(embedding), nn.Dropout(0.25))
        self.head = nn.Linear(embedding, classes)

    def apply_channel_gate(self, value: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(value.square().mean(dim=-1) + 1e-8)
        diff_rms = torch.sqrt((value[..., 1:] - value[..., :-1]).square().mean(dim=-1) + 1e-8)
        raw = self.channel_mlp(torch.stack((rms, diff_rms), dim=-1)).squeeze(-1)
        return value * (1.0 + torch.tanh(self.lambda_channel) * torch.tanh(raw)).unsqueeze(-1)

    def apply_scale_gate(self, branches: list[torch.Tensor]) -> list[torch.Tensor]:
        descriptor = torch.cat([branch.mean(dim=(2, 3)) for branch in branches], dim=1)
        alpha = F.softmax(self.scale_mlp(descriptor), dim=1)
        amplitude = torch.tanh(self.lambda_scale)
        return [branch * (1.0 + amplitude * (3.0 * alpha[:, index] - 1.0)).view(-1, 1, 1, 1) for index, branch in enumerate(branches)]

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.channel_gate:
            value = self.apply_channel_gate(value)
        branches = self.stem.branch_outputs(value)
        if self.scale_gate:
            branches = self.apply_scale_gate(branches)
        value = self.stem.temporal_blocks(branches).squeeze(2)
        for block in self.mixer:
            value = block(value)
        pooled = torch.cat((value.mean(dim=-1), value.std(dim=-1, unbiased=False), self.pool_bins(value).flatten(1)), dim=1)
        representation = self.embedding(pooled)
        return self.head(representation), representation


def build_model(architecture: str, task: str) -> nn.Module:
    spec = TASKS[task]
    return LiteBN_BASELINE(int(spec["channels"]), int(spec["classes"])) if architecture == "LiteBN_BASELINE" else LiteBNEnhanced(int(spec["channels"]), int(spec["classes"]), architecture)


def architecture_description() -> str:
    return """# LiteBN-X single-model architecture ladder

All candidates use one normalized EEG input, one neural network, one post-fusion
representation stream, and one classifier head/checkpoint. No teacher, logits,
distillation, ensemble, routing, mixture of experts, or test-time adaptation is used.

## LiteBN_BASELINE
Exact historical CompactLite(channels, bn): three temporal branches with kernels
15/63/127; 8 temporal then 16 spatial channels per branch; two historical
depthwise-separable temporal blocks 48-to-64-to-64; adaptive 8-bin pooling;
64-d embedding; and one classifier. Only classifier output dimension follows task classes.

## LiteBN-R
The historical stem and both historical temporal blocks are unchanged. The
64-channel post-block sequence receives residual temporal mixers with dilations
1/2/4. Each mixer has depthwise Conv1d(k=9), LayerNorm, F-to-2F-to-F GELU MLP,
dropout 0.10, and gamma initialized to 1e-3. Pooling is mean + standard deviation
+ four adaptive temporal bins (6F), then 96-d GELU/LayerNorm/dropout and one head.

## LiteBN-RG
LiteBN-R plus residual sample-dependent three-branch scale weighting after the
unchanged temporal+spatial branches. Descriptors are time means. The gate scale is
1 + tanh(lambda_scale) * (3 alpha_i - 1), lambda_scale initialized to zero.

## LiteBN-X
LiteBN-RG widened only as specified: temporal branch 8-to-12, spatial branch
16-to-24, concatenate 72, temporal blocks 72-to-96-to-96, mixer F=96, 6F-to-128 embedding.

## LiteBN-XS
LiteBN-X plus shared per-electrode residual gating before temporal stems. Each
electrode maps [RMS, temporal-difference RMS] through shared 2-to-8-to-1 MLP.
Input scaling is 1+tanh(lambda_channel)*tanh(g_c), with lambda_channel initialized
to zero. No channel, subject, session, or dataset identifiers are used.
"""


@dataclass(frozen=True)
class Row:
    subject: str
    session: int
    signal_path: str
    index: int
    label: int


class SignalBundle:
    def __init__(self, task: str, subjects: Iterable[str], rows: list[Row]):
        self.task, self.spec = task, TASKS[task]
        self.name = self.spec["dataset"]
        self.subjects = subject_sort(subjects, self.name)
        self.rows, self.channels, self.samples, self.classes = rows, int(self.spec["channels"]), int(self.spec["samples"]), int(self.spec["classes"])
        self._arrays: dict[str, np.ndarray] = {}

    def indices(self, subjects: Iterable[str], sessions: Iterable[int]) -> np.ndarray:
        wanted, wanted_sessions = set(map(str, subjects)), set(map(int, sessions))
        return np.asarray([index for index, row in enumerate(self.rows) if row.subject in wanted and row.session in wanted_sessions], dtype=np.int64)

    def labels(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.rows[int(index)].label for index in indices], dtype=np.int64)

    def signal_batch(self, indices: np.ndarray) -> np.ndarray:
        values = []
        for index in np.asarray(indices, dtype=np.int64):
            row = self.rows[int(index)]
            if row.signal_path not in self._arrays:
                self._arrays[row.signal_path] = np.load(row.signal_path, mmap_mode="r", allow_pickle=False)
            values.append(np.asarray(self._arrays[row.signal_path][row.index], dtype=np.float32))
        return np.stack(values, axis=0) if values else np.empty((0, self.channels, self.samples), dtype=np.float32)


def openbmi_path(task: str, subject: str, session: int, kind: str) -> Path:
    cache_name = TASKS[task]["cache"]
    base = CACHE / "openbmi" / "openbmi" / f"sub-{int(subject):02d}" / f"ses-{session}" / f"{cache_name}_1train"
    return base.with_name(base.name + ("_signals.npy" if kind == "signal" else "_codes.npy"))


def build_bundle(task: str, subjects: Iterable[str]) -> SignalBundle:
    spec = TASKS[task]
    dataset = str(spec["dataset"])
    canonical = subject_sort(subjects, dataset)
    sessions = tuple(sorted(set(spec["source_sessions"]) | {int(spec["future_session"])}))
    rows: list[Row] = []
    if dataset == "OpenBMI":
        for subject in canonical:
            for session in sessions:
                signal, labels = openbmi_path(task, subject, int(session), "signal"), openbmi_path(task, subject, int(session), "label")
                if not signal.is_file() or not labels.is_file():
                    raise FileNotFoundError(f"OpenBMI cache missing: {task} sub-{subject} ses-{session}")
                x, y = np.load(signal, mmap_mode="r", allow_pickle=False), np.load(labels, mmap_mode="r", allow_pickle=False)
                if x.ndim != 3 or tuple(x.shape[1:]) != (62, int(spec["samples"])) or x.dtype != np.float32:
                    raise RuntimeError(f"OpenBMI schema mismatch: {signal} {x.shape}/{x.dtype}")
                if y.shape != (x.shape[0],) or set(map(int, np.unique(y))) != set(spec["raw_codes"]):
                    raise RuntimeError(f"OpenBMI label schema mismatch: {labels}")
                rows.extend(Row(str(subject), int(session), str(signal), index, int(spec["raw_codes"][int(code)])) for index, code in enumerate(y))
    else:
        root = CACHE / "wbcic" / "wbcic_epochs"
        for subject in canonical:
            for session in sessions:
                signal, labels = root / subject / f"ses-{session}_epochs.npy", root / subject / f"ses-{session}_labels.npy"
                if not signal.is_file() or not labels.is_file():
                    raise FileNotFoundError(f"WBCIC cache missing: {subject} ses-{session}")
                x, y = np.load(signal, mmap_mode="r", allow_pickle=False), np.load(labels, mmap_mode="r", allow_pickle=False)
                if x.ndim != 3 or tuple(x.shape[1:]) != (58, 1000) or x.dtype != np.float16:
                    raise RuntimeError(f"WBCIC schema mismatch: {signal} {x.shape}/{x.dtype}")
                if y.shape != (x.shape[0],) or set(map(int, np.unique(y))) != {0, 1}:
                    raise RuntimeError(f"WBCIC label schema mismatch: {labels}")
                rows.extend(Row(str(subject), int(session), str(signal), index, int(code)) for index, code in enumerate(y))
    if not rows:
        raise RuntimeError(f"empty bundle: {task}")
    return SignalBundle(task, canonical, rows)


class RawGPUCache:
    def __init__(self, bundle: SignalBundle, device: torch.device):
        pieces = [torch.from_numpy(np.ascontiguousarray(bundle.signal_batch(np.arange(start, min(start + 128, len(bundle.rows)), dtype=np.int64)))) for start in range(0, len(bundle.rows), 128)]
        self.x = torch.cat(pieces, dim=0).to(device, non_blocking=True)
        self.y = torch.as_tensor(bundle.labels(np.arange(len(bundle.rows), dtype=np.int64)), dtype=torch.long, device=device)
        self.device = device

    def batch(self, indices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        index = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long, device=self.device)
        value = self.x.index_select(0, index)
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=self.device)[None, :, None]
        std_t = torch.as_tensor(std, dtype=torch.float32, device=self.device)[None, :, None]
        return (value - mean_t) / torch.clamp(std_t, min=1e-6), self.y.index_select(0, index)


def normalizer(bundle: SignalBundle, train_subjects: Iterable[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = bundle.indices(train_subjects, TASKS[bundle.task]["source_sessions"])
    if not len(indices):
        raise RuntimeError("normalizer has no source samples")
    total, squared, elements = np.zeros(bundle.channels, np.float64), np.zeros(bundle.channels, np.float64), 0
    for start in range(0, len(indices), 64):
        value = bundle.signal_batch(indices[start:start + 64]).astype(np.float64)
        total += value.sum(axis=(0, 2))
        squared += np.square(value).sum(axis=(0, 2))
        elements += value.shape[0] * value.shape[2]
    mean = (total / elements).astype(np.float32)
    std = np.sqrt(np.maximum(squared / elements - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    return mean, std, {"subjects": subject_sort(train_subjects, bundle.name), "source_sessions": list(map(int, TASKS[bundle.task]["source_sessions"])), "trials": int(len(indices)), "mean_std_sha256": sha256_bytes(mean.tobytes() + std.tobytes())}


def class_weights(bundle: SignalBundle, train_subjects: Iterable[str]) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not TASKS[bundle.task]["weighted_ce"]:
        return None, {"weighted_cross_entropy": False, "counts": None, "weights": None}
    labels = bundle.labels(bundle.indices(train_subjects, TASKS[bundle.task]["source_sessions"]))
    counts = np.bincount(labels, minlength=bundle.classes)
    if np.any(counts == 0):
        raise RuntimeError("a weighted-CE class is absent from training subjects")
    weights = (len(labels) / (bundle.classes * counts)).astype(np.float32)
    return torch.from_numpy(weights), {"weighted_cross_entropy": True, "counts": counts.tolist(), "weights": weights.tolist(), "formula": "N/(K*N_k)"}


def load_folds() -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]], str]:
    raw = read_json(FIVEFOLD)
    if raw.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1":
        raise RuntimeError("unexpected carrier five-fold protocol")
    search, folds = {}, {}
    for dataset, expected in (("OpenBMI", 40), ("WBCIC", 31)):
        search[dataset] = subject_sort(raw["search_subjects"][dataset], dataset)
        if len(search[dataset]) != expected or len(set(search[dataset])) != expected:
            raise RuntimeError(f"{dataset} SEARCH membership mismatch")
        entries, seen = [], set()
        for item in raw["folds"][dataset]:
            entry = {"fold_id": int(item["fold_id"]), "fold_seed": int(item["fold_seed"]), "inner_split_seed": int(item["inner_split_seed"]), "inner_train_subjects": subject_sort(item["inner_train_subjects"], dataset), "inner_val_subjects": subject_sort(item["inner_val_subjects"], dataset), "outer_dev_subjects": subject_sort(item["outer_dev_subjects"], dataset)}
            train, val, outer = set(entry["inner_train_subjects"]), set(entry["inner_val_subjects"]), set(entry["outer_dev_subjects"])
            if train & val or train & outer or val & outer or train | val | outer != set(search[dataset]):
                raise RuntimeError(f"invalid frozen {dataset} fold {entry['fold_id']}")
            seen |= outer
            entries.append(entry)
        if len(entries) != 5 or seen != set(search[dataset]):
            raise RuntimeError(f"{dataset} outer-development coverage mismatch")
        folds[dataset] = entries
    return search, folds, sha256_file(FIVEFOLD)


def subject_index(bundle: SignalBundle) -> dict[tuple[str, int, int], list[int]]:
    result: dict[tuple[str, int, int], list[int]] = {}
    for index, row in enumerate(bundle.rows):
        result.setdefault((row.subject, row.session, row.label), []).append(index)
    return result


def sample_subject_trials(mapping: dict[tuple[str, int, int], list[int]], subject: str, sessions: Iterable[int], per_session_class: int, generator: np.random.Generator) -> list[int]:
    result = []
    for session in sessions:
        for cls in (0, 1):
            pool = mapping[(subject, int(session), cls)]
            if len(pool) < per_session_class:
                raise RuntimeError(f"insufficient MI trials: {subject} s{session} c{cls}")
            result.extend(int(value) for value in generator.choice(np.asarray(pool), size=per_session_class, replace=False))
    return result


def mi_manifest(bundle: SignalBundle, fold: dict[str, Any], task: str) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    mapping, source, train = subject_index(bundle), tuple(map(int, TASKS[task]["source_sessions"])), fold["inner_train_subjects"]
    steps = max(20, int(math.ceil(len(bundle.indices(train, source)) / MI_EPISODE_BATCH)))
    epochs: list[list[np.ndarray]] = []
    for epoch in range(1, EPOCHS + 1):
        current: list[np.ndarray] = []
        for step in range(steps):
            generator = np.random.default_rng(int(fold["fold_seed"]) + epoch * 1_000_003 + step * 97)
            order = [str(value) for value in generator.permutation(np.asarray(train, dtype=object))]
            support, query = order[:4], order[4:8]
            if len(support) != 4 or len(query) != 4 or set(support) & set(query):
                raise RuntimeError("invalid source/future MI episode")
            support_indices, query_indices = [], []
            per_source_class = 8 if task == "OpenBMI_MI" else 4
            for subject in support:
                support_indices.extend(sample_subject_trials(mapping, subject, source, per_source_class, generator))
            for subject in query:
                query_indices.extend(sample_subject_trials(mapping, subject, (2,), 8, generator))
            if len(support_indices) != 64 or len(query_indices) != 64:
                raise RuntimeError("MI episode cardinality is not 128")
            current.append(np.asarray(support_indices + query_indices, dtype=np.int64))
        epochs.append(current)
    path = RUNTIME / "episode_manifests" / f"{task.lower()}_fold{fold['fold_id']}.json"
    write_json(path, {"task": task, "fold": int(fold["fold_id"]), "steps_per_epoch": steps, "source_sessions": list(source), "query_session": 2, "training_subjects": train, "outer_rows_absent": True, "epochs": [[row.tolist() for row in epoch] for epoch in epochs]})
    return epochs, {"kind": "historical_MI_subject_disjoint_episodes", "steps_per_epoch": steps, "manifest_path": str(path), "manifest_sha256": sha256_file(path)}


def task_epoch_batches(indices: np.ndarray, task: str, fold_id: int, epoch: int) -> list[np.ndarray]:
    task_code = 17 if task == "OpenBMI_ERP" else 29
    shuffled = np.random.default_rng(1_000_003 * (epoch + 1) + 1009 * fold_id + task_code).permutation(indices)
    return [shuffled[start:start + TASK_BATCH] for start in range(0, len(shuffled), TASK_BATCH)]


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(logits).argmax(axis=1)
    return {"BA": float(balanced_accuracy_score(labels, prediction)), "macro_F1": float(f1_score(labels, prediction, average="macro", zero_division=0)), "accuracy": float(accuracy_score(labels, prediction))}


def evaluate(model: nn.Module, bundle: SignalBundle, cache: RawGPUCache, subjects: Iterable[str], mean: np.ndarray, std: np.ndarray) -> dict[str, dict[str, Any]]:
    model.eval()
    result: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for subject in subject_sort(subjects, bundle.name):
            indices = bundle.indices([subject], (int(TASKS[bundle.task]["future_session"]),))
            labels, logits = bundle.labels(indices), []
            for start in range(0, len(indices), 128):
                value, _ = cache.batch(indices[start:start + 128], mean, std)
                logits.append(model(value)[0].float().cpu().numpy())
            result[str(subject)] = {**classification_metrics(labels, np.concatenate(logits, axis=0)), "trials": int(len(labels))}
    return result


def save_tensor_pair(path: Path, mean: np.ndarray, std: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, mean=mean, std=std, metadata=json.dumps(clean(metadata), sort_keys=True))
    os.replace(temporary, path)


def load_tensor_pair(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["mean"], dtype=np.float32), np.asarray(archive["std"], dtype=np.float32), json.loads(str(archive["metadata"].item()))


def checkpoint_path(task: str, fold_id: int, architecture: str, which: str) -> Path:
    return RUNTIME / "checkpoints" / task.lower() / f"fold{fold_id}_{architecture.lower()}" / which


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_one(model: nn.Module, architecture: str, task: str, fold: dict[str, Any], bundle: SignalBundle, cache: RawGPUCache, mean: np.ndarray, std: np.ndarray, normalizer_meta: dict[str, Any], batch_info: dict[str, Any], class_weight: torch.Tensor | None, class_weight_meta: dict[str, Any], device: torch.device) -> dict[str, Any]:
    latest, selected = checkpoint_path(task, int(fold["fold_id"]), architecture, "checkpoint_latest.pt"), checkpoint_path(task, int(fold["fold_id"]), architecture, "selected_best.pt")
    initial_hash, source_hash = state_hash(model), sha256_file(Path(__file__))
    invariants = {"task": task, "fold": int(fold["fold_id"]), "architecture": architecture, "seed": SEED, "initial_sha256": initial_hash, "source_sha256": source_hash, "normalizer_sha256": normalizer_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info.get("manifest_sha256"), "class_weight_info": class_weight_meta}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp, scaler = device.type == "cuda", torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved.get("invariants") != invariants:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"])
        start, history, best, best_epoch, best_state = int(saved["epoch"]) + 1, list(saved["history"]), float(saved["best"]), saved["best_epoch"], saved["best_state"]
    train_indices = bundle.indices(fold["inner_train_subjects"], TASKS[task]["source_sessions"])
    weight, started = None if class_weight is None else class_weight.to(device), time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        losses = []
        batches: Iterable[np.ndarray] = batch_info["episodes"][epoch - 1] if TASKS[task]["mi_protocol"] else task_epoch_batches(train_indices, task, int(fold["fold_id"]), epoch)
        for indices in batches:
            value, labels = cache.batch(np.asarray(indices, dtype=np.int64), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(value)
                loss = F.cross_entropy(logits, labels, weight=weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite CE: {task}/{architecture}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, bundle, cache, fold["inner_val_subjects"], mean, std)
        validation_ba, validation_f1 = float(np.mean([row["BA"] for row in validation.values()])), float(np.mean([row["macro_F1"] for row in validation.values()]))
        chose = epoch >= MIN_EPOCH and validation_ba > best + TIE_TOL
        if chose:
            best, best_epoch, best_state = validation_ba, int(epoch), copy.deepcopy(model.state_dict())
        history.append({"epoch": int(epoch), "cross_entropy": float(np.mean(losses)), "inner_val_subject_BA": validation_ba, "inner_val_subject_macro_F1": validation_f1, "selected": bool(chose), "batches": int(len(losses))})
        atomic_torch_save(latest, {"epoch": int(epoch), "history": history, "best": best, "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "invariants": invariants})
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"[A {task} f{fold['fold_id']} {architecture}] epoch={epoch:02d} CE={history[-1]['cross_entropy']:.4f} valBA={validation_ba:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"no checkpoint selected after epoch {MIN_EPOCH}")
    model.load_state_dict(best_state, strict=True)
    atomic_torch_save(selected, model.state_dict())
    return {"task": task, "dataset": TASKS[task]["dataset"], "fold": int(fold["fold_id"]), "architecture": architecture, "seed": SEED, "parameter_count": parameter_count(model), "initial_sha256": initial_hash, "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best), "best_inner_val_macro_F1": float(next(row["inner_val_subject_macro_F1"] for row in history if row["epoch"] == best_epoch)), "checkpoint_path": str(selected), "checkpoint_sha256": sha256_file(selected), "normalizer_sha256": normalizer_meta["mean_std_sha256"], "batch_manifest_sha256": batch_info.get("manifest_sha256"), "class_weighted_ce": bool(class_weight_meta["weighted_cross_entropy"]), "elapsed_seconds_this_invocation": float(time.perf_counter() - started), "epochs_completed": int(len(history)), "history": history, "source_sha256": source_hash}


def baseline_audit() -> list[dict[str, Any]]:
    rows, historic = [], historical_litebn_class()
    for task in TASK_ORDER:
        spec = TASKS[task]
        set_seed(9123)
        actual = LiteBN_BASELINE(int(spec["channels"]), int(spec["classes"]))
        set_seed(9123)
        reference = historic(int(spec["channels"]), "bn")
        reference.head = nn.Linear(64, int(spec["classes"]))
        same_keys, same_params = tuple(actual.state_dict()) == tuple(reference.state_dict()), parameter_count(actual) == parameter_count(reference)
        reference.load_state_dict(actual.state_dict(), strict=True)
        set_seed(77)
        value = torch.randn(2, int(spec["channels"]), int(spec["samples"]))
        actual.eval(); reference.eval()
        with torch.no_grad():
            a_logits, a_hidden = actual(value)
            b_logits, b_hidden = reference(value)
        equivalent = bool(torch.equal(a_logits, b_logits) and torch.equal(a_hidden, b_hidden))
        if not (same_keys and same_params and equivalent):
            raise RuntimeError(f"historical LiteBN verification failed: {task}")
        rows.append({"task": task, "architecture": "LiteBN_BASELINE", "historical_source": str(CARRIER_CODE / "run_carrier_screen.py"), "historical_source_sha256": sha256_file(CARRIER_CODE / "run_carrier_screen.py"), "parameter_count": parameter_count(actual), "state_keys_identical": same_keys, "strict_checkpoint_roundtrip": equivalent})
    return rows


def parameter_rows() -> pd.DataFrame:
    rows, audit = [], {row["task"]: row for row in baseline_audit()}
    for task in TASK_ORDER:
        for architecture in ARCHITECTURES:
            model = build_model(architecture, task)
            value = torch.zeros((2, TASKS[task]["channels"], TASKS[task]["samples"]))
            logits, representation = model(value)
            if logits.shape != (2, TASKS[task]["classes"]) or representation.ndim != 2:
                raise RuntimeError(f"model shape audit failed: {task}/{architecture}")
            rows.append({"task": task, "dataset": TASKS[task]["dataset"], "architecture": architecture, "parameter_count": parameter_count(model), "input_shape": json.dumps([2, TASKS[task]["channels"], TASKS[task]["samples"]]), "logits_shape": json.dumps(list(logits.shape)), "representation_dimension": int(representation.shape[1]), "historical_baseline_verified": bool(audit[task]["strict_checkpoint_roundtrip"]) if architecture == "LiteBN_BASELINE" else None})
    return pd.DataFrame(rows)


def stage_a_preflight(search: dict[str, list[str]], folds: dict[str, list[dict[str, Any]]], split_hash: str) -> None:
    for path in (PROTOCOL, OUT, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)
    parameters = parameter_rows()
    if (parameters[parameters.architecture != "LiteBN_BASELINE"].parameter_count < 100_000).any():
        raise RuntimeError("a LiteBN-X candidate falls below the target scale")
    if (parameters[parameters.architecture.isin(("LiteBN_X", "LiteBN_XS"))].parameter_count > 300_000).any():
        raise RuntimeError("LiteBN-X width exceeds the declared parameter range")
    write_csv(OUT / "PARAMETER_COUNTS.csv", parameters)
    write_text(PROTOCOL / "ARCHITECTURE_SPEC.md", architecture_description())
    write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"scope": "SEARCH/development only", "final_heldout_subject_manifest_read": False, "final_heldout_signal_loaded": False, "final_heldout_labels_loaded": False, "final_heldout_predictions_generated": False, "outer_dev_access_during_stage_a": False, "stage_b_outer_dev_access_requires_explicit_flag": True, "allowed_search_subject_counts": {"OpenBMI": len(search["OpenBMI"]), "WBCIC": len(search["WBCIC"])}, "frozen_fivefold_split_sha256": split_hash})
    write_json(PROTOCOL / "STAGE_A_PREFLIGHT.json", {"pass": True, "seed": SEED, "tasks": list(TASK_ORDER), "architectures": list(ARCHITECTURES), "split_sha256": split_hash, "fold_counts": {dataset: len(value) for dataset, value in folds.items()}, "historical_litebn_checkpoint_behavior_verified": True, "no_final_holdout_access": True})


def selection_from_inner(frame: pd.DataFrame, parameters: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    baseline = frame[frame.architecture == "LiteBN_BASELINE"].groupby("task").best_inner_val_BA.mean()
    for architecture in CANDIDATES:
        means = frame[frame.architecture == architecture].groupby("task").best_inner_val_BA.mean()
        delta_pp = ((means - baseline) * 100.0).reindex(TASK_ORDER)
        counts = parameters[parameters.architecture == architecture].groupby("task").parameter_count.max()
        rows.append({"architecture": architecture, **{f"{task}_inner_val_BA": float(means.loc[task]) for task in TASK_ORDER}, **{f"{task}_delta_pp": float(delta_pp.loc[task]) for task in TASK_ORDER}, "equal_task_mean_delta_pp": float(delta_pp.mean()), "worst_task_delta_pp": float(delta_pp.min()), "max_parameter_count": int(counts.max()), "eligible": bool(delta_pp.mean() > 0.0 and delta_pp.min() >= -0.50)})
    summary = pd.DataFrame(rows).sort_values(["equal_task_mean_delta_pp", "max_parameter_count", "architecture"], ascending=[False, True, True]).reset_index(drop=True)
    eligible = summary[summary.eligible].copy()
    pool = eligible if len(eligible) else summary
    top = float(pool.equal_task_mean_delta_pp.max())
    chosen = pool[pool.equal_task_mean_delta_pp >= top - 0.10].sort_values(["max_parameter_count", "architecture"], ascending=[True, True]).iloc[0]
    return summary, {"selected_architecture": str(chosen.architecture), "stage_a_no_robust_winner": not bool(len(eligible)), "selection_rule": "eligible: equal-task mean delta > 0 and worst task delta >= -0.50 pp; choose highest equal-task mean delta, with candidates within 0.10 pp resolved by smaller parameter count", "selected_equal_task_mean_delta_pp": float(chosen.equal_task_mean_delta_pp), "selected_worst_task_delta_pp": float(chosen.worst_task_delta_pp), "selected_parameter_count_max": int(chosen.max_parameter_count)}


def stage_a() -> int:
    search, folds, split_hash = load_folds()
    stage_a_preflight(search, folds, split_hash)
    device, rows, provenance = torch.device("cuda" if torch.cuda.is_available() else "cpu"), [], []
    for task in TASK_ORDER:
        dataset = TASKS[task]["dataset"]
        for fold in folds[dataset]:
            allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            if set(allowed) & set(fold["outer_dev_subjects"]):
                raise RuntimeError("Stage A bundle would include an outer subject")
            bundle = build_bundle(task, allowed)
            if set(bundle.subjects) != set(allowed):
                raise RuntimeError("Stage A bundle membership mismatch")
            mean, std, norm_meta = normalizer(bundle, fold["inner_train_subjects"])
            save_tensor_pair(RUNTIME / "normalizers" / f"{task.lower()}_fold{fold['fold_id']}.npz", mean, std, norm_meta)
            cache = RawGPUCache(bundle, device)
            if TASKS[task]["mi_protocol"]:
                episodes, manifest_meta = mi_manifest(bundle, fold, task)
                batch_info = {**manifest_meta, "episodes": episodes}
            else:
                batch_info = {"kind": "historical_task_full_permutation_batch64", "manifest_sha256": None, "steps_per_epoch": None}
            weight, weight_meta = class_weights(bundle, fold["inner_train_subjects"])
            for architecture in ARCHITECTURES:
                set_seed(SEED)
                model = build_model(architecture, task).to(device)
                set_seed(SEED + 100_000)
                record = train_one(model, architecture, task, fold, bundle, cache, mean, std, norm_meta, batch_info, weight, weight_meta, device)
                rows.append({key: value for key, value in record.items() if key != "history"})
                provenance.append(record)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    frame, expected = pd.DataFrame(rows), len(TASK_ORDER) * 5 * len(ARCHITECTURES)
    if len(frame) != expected or frame.duplicated(["task", "fold", "architecture"]).any():
        raise RuntimeError(f"incomplete Stage A grid: {len(frame)}/{expected}")
    parameters = pd.read_csv(OUT / "PARAMETER_COUNTS.csv")
    summary, selection = selection_from_inner(frame, parameters)
    write_csv(OUT / "STAGE_A_INNERVAL_RESULTS.csv", frame.sort_values(["task", "fold", "architecture"]))
    write_csv(OUT / "STAGE_A_ARCHITECTURE_SUMMARY.csv", summary)
    write_json(PROTOCOL / "CHECKPOINT_PROVENANCE.json", {"stage": "A", "seed": SEED, "expected_cells": expected, "records": [{key: value for key, value in record.items() if key != "history"} for record in provenance]})
    baseline_means = frame[frame.architecture == "LiteBN_BASELINE"].groupby("task").best_inner_val_BA.mean()
    table = ["# Stage-A LiteBN-X selection (inner validation only)", "", "No outer-development, final-holdout, or test prediction was generated before this file.", "", f"Selected architecture: {selection['selected_architecture']}", f"Stage-A no robust winner: {selection['stage_a_no_robust_winner']}", f"Source hash: {sha256_file(Path(__file__))}", f"Split hash: {split_hash}", "", "## Selection rule", "", selection["selection_rule"], "", "## Candidate definitions", "", architecture_description(), "", "## Inner-validation task means and deltas", "", "| Architecture | MI BA / delta | ERP BA / delta | SSVEP BA / delta | WBCIC BA / delta | Equal mean delta | Worst delta | Eligible |", "|---|---:|---:|---:|---:|---:|---:|---|", f"| LiteBN_BASELINE | {baseline_means.loc['OpenBMI_MI']:.4f} / +0.000 | {baseline_means.loc['OpenBMI_ERP']:.4f} / +0.000 | {baseline_means.loc['OpenBMI_SSVEP']:.4f} / +0.000 | {baseline_means.loc['WBCIC_MI']:.4f} / +0.000 | +0.000 | +0.000 | reference |"]
    for row in summary.itertuples(index=False):
        table.append(f"| {row.architecture} | {row.OpenBMI_MI_inner_val_BA:.4f} / {row.OpenBMI_MI_delta_pp:+.3f} | {row.OpenBMI_ERP_inner_val_BA:.4f} / {row.OpenBMI_ERP_delta_pp:+.3f} | {row.OpenBMI_SSVEP_inner_val_BA:.4f} / {row.OpenBMI_SSVEP_delta_pp:+.3f} | {row.WBCIC_MI_inner_val_BA:.4f} / {row.WBCIC_MI_delta_pp:+.3f} | {row.equal_task_mean_delta_pp:+.3f} | {row.worst_task_delta_pp:+.3f} | {bool(row.eligible)} |")
    table += ["", "## Exact parameter counts", "", "| Architecture | OpenBMI MI | OpenBMI ERP | OpenBMI SSVEP | WBCIC MI |", "|---|---:|---:|---:|---:|"]
    for architecture in ARCHITECTURES:
        q = parameters[parameters.architecture == architecture].set_index("task")
        table.append(f"| {architecture} | {int(q.loc['OpenBMI_MI', 'parameter_count'])} | {int(q.loc['OpenBMI_ERP', 'parameter_count'])} | {int(q.loc['OpenBMI_SSVEP', 'parameter_count'])} | {int(q.loc['WBCIC_MI', 'parameter_count'])} |")
    table += ["", "## Fixed recipe", "", "AdamW lr=3e-4, weight_decay=5e-4, gradient clipping 5.0, 60 epochs, earliest-best inner future-session subject-equal BA at epochs 10..60, AMP, seed 0. MI preserves historical subject-disjoint source/future episode construction; ERP/SSVEP preserve historical full-permutation batch-64 training and ERP class weighting."]
    write_text(PROTOCOL / "STAGE_A_SELECTION.md", "\n".join(table))
    write_json(PROTOCOL / "STAGE_A_LOCK.json", {"stage": "A", "seed": SEED, "stage_a_complete": True, "outer_dev_predictions_generated": False, "final_holdout_predictions_generated": False, "source_sha256": sha256_file(Path(__file__)), "split_sha256": split_hash, **selection})
    print("STAGE_A_READY_FOR_FREEZE", json.dumps(selection, sort_keys=True), flush=True)
    return 0


def paired_bootstrap(delta_pp: np.ndarray) -> dict[str, Any]:
    draws = np.random.default_rng(0).choice(delta_pp, size=(BOOTSTRAPS, len(delta_pp)), replace=True).mean(axis=1)
    return {"n_subjects": int(len(delta_pp)), "bootstrap_seed": 0, "resamples": BOOTSTRAPS, "mean_delta_pp": float(delta_pp.mean()), "ci_low_pp": float(np.quantile(draws, 0.025)), "ci_high_pp": float(np.quantile(draws, 0.975)), "positive_subjects": int((delta_pp > TIE_TOL).sum()), "harmed_subjects": int((delta_pp < -TIE_TOL).sum()), "tied_subjects": int((np.abs(delta_pp) <= TIE_TOL).sum())}


def stage_b(stage_a_commit: str) -> int:
    lock_path = PROTOCOL / "STAGE_A_LOCK.json"
    if not lock_path.is_file() or not (PROTOCOL / "STAGE_A_SELECTION.md").is_file():
        raise RuntimeError("Stage A freeze artifacts are missing")
    lock = read_json(lock_path)
    if not lock.get("stage_a_complete") or lock.get("outer_dev_predictions_generated"):
        raise RuntimeError("invalid Stage A lock for a one-time reveal")
    if lock.get("source_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("source changed after Stage A freeze")
    selected_architecture = str(lock["selected_architecture"])
    if selected_architecture not in CANDIDATES:
        raise RuntimeError("invalid selected candidate")
    search, folds, split_hash = load_folds()
    if split_hash != lock.get("split_sha256"):
        raise RuntimeError("frozen split hash changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_json(PROTOCOL / "CHECKPOINT_PROVENANCE.json")["records"]
    by_cell = {(row["task"], int(row["fold"]), row["architecture"]): row for row in records}
    output_rows = []
    for task in TASK_ORDER:
        dataset = TASKS[task]["dataset"]
        for fold in folds[dataset]:
            outer = fold["outer_dev_subjects"]
            bundle = build_bundle(task, outer)
            if set(bundle.subjects) != set(outer):
                raise RuntimeError("Stage B outer bundle membership mismatch")
            mean, std, norm_meta = load_tensor_pair(RUNTIME / "normalizers" / f"{task.lower()}_fold{fold['fold_id']}.npz")
            cache = RawGPUCache(bundle, device)
            for architecture in ("LiteBN_BASELINE", selected_architecture):
                record = by_cell[(task, int(fold["fold_id"]), architecture)]
                checkpoint = Path(record["checkpoint_path"])
                if not checkpoint.is_file() or sha256_file(checkpoint) != record["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint provenance failure: {checkpoint}")
                model = build_model(architecture, task).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
                for subject, metrics in evaluate(model, bundle, cache, outer, mean, std).items():
                    output_rows.append({"task": task, "dataset": dataset, "fold": int(fold["fold_id"]), "subject_id": subject, "method": architecture, **metrics, "checkpoint_sha256": record["checkpoint_sha256"], "normalizer_sha256": norm_meta["mean_std_sha256"]})
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    subject_frame, expected = pd.DataFrame(output_rows), 2 * (40 + 40 + 40 + 31)
    if len(subject_frame) != expected or subject_frame.duplicated(["task", "subject_id", "method"]).any():
        raise RuntimeError(f"Stage B cardinality failure: {len(subject_frame)}/{expected}")
    task_rows, bootstrap_rows, task_deltas, fold_collapse = [], [], {}, False
    for task in TASK_ORDER:
        subset = subject_frame[subject_frame.task == task]
        baseline = subset[subset.method == "LiteBN_BASELINE"].set_index("subject_id")
        candidate = subset[subset.method == selected_architecture].set_index("subject_id")
        if set(baseline.index) != set(candidate.index):
            raise RuntimeError("paired subject alignment failure")
        delta_pp = (candidate.BA - baseline.BA).to_numpy(float) * 100.0
        boot = paired_bootstrap(delta_pp)
        pivot = subset.groupby(["fold", "method"], as_index=False).BA.mean().pivot(index="fold", columns="method", values="BA")
        fold_means = pivot[selected_architecture] - pivot["LiteBN_BASELINE"]
        chance = 1.0 / TASKS[task]["classes"]
        fold_collapse = fold_collapse or bool((candidate.groupby("fold").BA.mean() <= chance + 0.02).any())
        task_deltas[task] = float(delta_pp.mean())
        task_rows.append({"task": task, "dataset": TASKS[task]["dataset"], "n_outer_subjects": int(len(baseline)), "LiteBN_BA": float(baseline.BA.mean()), "selected_model": selected_architecture, "selected_BA": float(candidate.BA.mean()), "delta_pp": float(delta_pp.mean()), "LiteBN_macro_F1": float(baseline.macro_F1.mean()), "selected_macro_F1": float(candidate.macro_F1.mean()), "LiteBN_accuracy": float(baseline.accuracy.mean()), "selected_accuracy": float(candidate.accuracy.mean()), "positive_folds": int((fold_means > 0).sum()), "harmed_folds": int((fold_means < 0).sum()), **boot})
        bootstrap_rows.append({"task": task, "comparison": f"{selected_architecture}-LiteBN_BASELINE", **boot})
    equal_task = float(np.mean([task_deltas[task] for task in TASK_ORDER]))
    openbmi_mean = float(np.mean([task_deltas[task] for task in TASK_ORDER[:3]]))
    dataset_balanced = float(0.5 * (openbmi_mean + task_deltas["WBCIC_MI"]))
    positive_tasks, worsened_tasks = int(sum(value > 0 for value in task_deltas.values())), int(sum(value <= -0.50 for value in task_deltas.values()))
    if positive_tasks == 4 and equal_task >= 0.75 and sum(value >= 1.0 for value in task_deltas.values()) >= 2 and not fold_collapse:
        terminal = "STRONG_SINGLE_MODEL"
    elif positive_tasks >= 3 and equal_task >= 0.50 and min(task_deltas.values()) > -0.50:
        terminal = "PROMISING_SINGLE_MODEL"
    elif equal_task <= 0.0 or worsened_tasks >= 2:
        terminal = "NEGATIVE_SINGLE_MODEL"
    else:
        terminal = "MIXED_SINGLE_MODEL"
    task_frame = pd.DataFrame(task_rows)
    write_csv(OUT / "OUTER_SUBJECT_RESULTS.csv", subject_frame.sort_values(["task", "fold", "subject_id", "method"]))
    write_csv(OUT / "OUTER_TASK_RESULTS.csv", task_frame)
    write_csv(OUT / "PAIRED_BOOTSTRAP.csv", pd.DataFrame(bootstrap_rows))
    counts = pd.read_csv(OUT / "PARAMETER_COUNTS.csv")
    selected_params = counts[counts.architecture == selected_architecture].set_index("task")
    base_params = counts[counts.architecture == "LiteBN_BASELINE"].set_index("task")
    lines = ["# LiteBN-X single-model seed-0 decision", "", f"Selected before outer-dev reveal: {selected_architecture}.", f"Stage-A robust winner: {not lock['stage_a_no_robust_winner']}.", "", "| Task | LiteBN BA | New Model BA | Delta pp | 95% CI | LiteBN F1 | New F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in task_frame.itertuples(index=False):
        lines.append(f"| {row.task.replace('_', ' ')} | {row.LiteBN_BA:.4f} | {row.selected_BA:.4f} | {row.delta_pp:+.3f} | [{row.ci_low_pp:+.3f}, {row.ci_high_pp:+.3f}] | {row.LiteBN_macro_F1:.4f} | {row.selected_macro_F1:.4f} |")
    lines += ["", f"Equal-task mean delta: {equal_task:+.3f} pp.", f"OpenBMI three-task mean delta: {openbmi_mean:+.3f} pp.", f"Dataset-balanced mean delta: {dataset_balanced:+.3f} pp.", f"Task improvements: {positive_tasks}/4.", f"Decision terminal: {terminal}.", "", "## Required answers", "", f"1. Selected architecture: {selected_architecture}.", f"2. It was selected only by the frozen Stage-A inner-validation rule: {lock['selection_rule']}", "3. Parameters versus LiteBN are task-specific only through head dimension: " + "; ".join(f"{task} {int(base_params.loc[task, 'parameter_count'])} to {int(selected_params.loc[task, 'parameter_count'])}" for task in TASK_ORDER) + ".", f"4. OpenBMI MI gain: {task_deltas['OpenBMI_MI']:+.3f} pp.", f"5. OpenBMI ERP gain: {task_deltas['OpenBMI_ERP']:+.3f} pp.", f"6. OpenBMI SSVEP gain: {task_deltas['OpenBMI_SSVEP']:+.3f} pp.", f"7. WBCIC MI gain: {task_deltas['WBCIC_MI']:+.3f} pp.", f"8. Improved tasks: {positive_tasks}/4.", f"9. Equal-task mean gain: {equal_task:+.3f} pp.", f"10. Dataset-balanced mean gain: {dataset_balanced:+.3f} pp.", "11. Replacement support is limited to the terminal above; this remains seed-0 SEARCH/development evidence.", f"12. Seed1/2 are {'worth running' if terminal in ('STRONG_SINGLE_MODEL', 'PROMISING_SINGLE_MODEL') else 'not justified automatically'}.", "13. The inner-val ablation summary identifies the useful module descriptively; no outer result changed the architecture.", "14. Held-out/test data accessed: NO.", "", "Boundary: Stage B opened only frozen outer-development SEARCH subjects once after the Stage-A freeze. It did not read final held-out/test labels or predictions."]
    write_text(OUT / "FINAL_SEED0_DECISION.md", "\n".join(lines))
    write_json(PROTOCOL / "STAGE_B_REVEAL.json", {"stage_a_commit": stage_a_commit, "selected_architecture": selected_architecture, "outer_dev_predictions_generated": True, "final_holdout_predictions_generated": False, "final_holdout_data_accessed": False, "split_sha256": split_hash, "terminal": terminal, "equal_task_mean_delta_pp": equal_task, "dataset_balanced_mean_delta_pp": dataset_balanced})
    lock["outer_dev_predictions_generated"] = True
    write_json(lock_path, lock)
    print(terminal, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage_a", "stage_b"))
    parser.add_argument("--stage-a-commit")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        search, folds, split_hash = load_folds()
        stage_a_preflight(search, folds, split_hash)
        print("LITEBN_X_PREFLIGHT_VALID", flush=True)
        return 0
    if args.stage == "stage_a":
        return stage_a()
    if not args.stage_a_commit:
        raise SystemExit("stage_b requires --stage-a-commit after an externally committed Stage-A freeze")
    return stage_b(args.stage_a_commit)


if __name__ == "__main__":
    raise SystemExit(main())
