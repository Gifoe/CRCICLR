from __future__ import annotations

"""Outcome-blind seed-0 modern-backbone experiment helpers.

This module deliberately exposes only the frozen V8_SEARCH cache and the
five-fold carrier split.  Training/evaluation entry points pass an explicit
subject list, so Stage A never materialises outer-development rows.
"""

import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO = Path(os.environ.get("MODERN_REPO", "/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_outcome_blind_modern_backbone_seed0_v1"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("MODERN_RUNTIME", "/root/rivermind-data/modern_backbone_seed0_runtime")).resolve()
CARRIER_RUNTIME = Path(os.environ.get("CARRIER_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
CACHE_ROOT = Path(os.environ.get("PERSIST_CACHE_ROOT", "/root/rivermind-data/persist_eeg_cache")).resolve()
OFFICIAL_ROOT = Path(os.environ.get("OFFICIAL_BACKBONE_ROOT", "/root/rivermind-data/official_backbone_sources")).resolve()
SPLIT_PATH = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"

DATASETS = ("OpenBMI", "WBCIC")
MODELS = ("EEGNet", "LiteBN", "TCFormer", "ST-EEGFormer-small", "LaBraM-base", "CBraMod")
SOURCE_SESSIONS = {"OpenBMI": (1,), "WBCIC": (0, 1)}
EVAL_SESSION = 2
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = 12
MIN_EPOCHS = 4
PATIENCE = 3


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
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
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def subject_sort(values: Iterable[Any]) -> list[str]:
    def key(value: Any) -> tuple[int, str]:
        text = str(value).replace("sub-", "")
        return (int(text) if text.isdigit() else 10**9, text)
    return sorted([str(v) for v in values], key=key)


def load_split() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], str]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(SPLIT_PATH)
    payload = read_json(SPLIT_PATH)
    if payload.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1":
        raise RuntimeError("unexpected carrier split protocol")
    folds: dict[str, list[dict[str, Any]]] = {}
    search: dict[str, list[str]] = {}
    for dataset, expected in (("OpenBMI", 40), ("WBCIC", 31)):
        search[dataset] = subject_sort(payload["search_subjects"][dataset])
        if len(search[dataset]) != expected:
            raise RuntimeError(f"{dataset} search size mismatch: {len(search[dataset])}")
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in payload["folds"][dataset]:
            row = {k: raw[k] for k in ("fold_id", "fold_seed", "inner_split_seed", "inner_train_subjects", "inner_val_subjects", "outer_dev_subjects")}
            for k in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects"):
                row[k] = subject_sort(row[k])
            a, b, c = map(set, (row["inner_train_subjects"], row["inner_val_subjects"], row["outer_dev_subjects"]))
            if a & b or a & c or b & c or (a | b | c) != set(search[dataset]):
                raise RuntimeError(f"invalid {dataset} fold {row['fold_id']}")
            seen |= c
            rows.append(row)
        if len(rows) != 5 or seen != set(search[dataset]):
            raise RuntimeError(f"{dataset} five-fold coverage failure")
        folds[dataset] = rows
    return folds, search, sha256_file(SPLIT_PATH)


@dataclass(frozen=True)
class Sample:
    subject: str
    session: int
    signal_path: Path
    index: int
    label: int


@dataclass
class Subset:
    x: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    channels: int

    def indices_for(self, subjects: Sequence[str] | None = None, sessions: Sequence[int] | None = None) -> np.ndarray:
        mask = np.ones(len(self.labels), dtype=bool)
        if subjects is not None:
            mask &= np.isin(self.subjects.astype(str), np.asarray(list(map(str, subjects))))
        if sessions is not None:
            mask &= np.isin(self.sessions.astype(int), np.asarray(list(map(int, sessions))))
        return np.flatnonzero(mask).astype(np.int64)


def _sample_rows(dataset: str, subjects: Sequence[str], sessions: Sequence[int]) -> list[Sample]:
    rows: list[Sample] = []
    wanted = set(map(str, subjects))
    session_set = set(map(int, sessions))
    if dataset == "OpenBMI":
        root = CACHE_ROOT / "openbmi" / "openbmi"
        for subject in subject_sort(wanted):
            for session in sorted(session_set):
                base = root / f"sub-{int(subject):02d}" / f"ses-{session}" / "mi_1train"
                signal = base.with_name(base.name + "_signals.npy")
                labels = base.with_name(base.name + "_codes.npy")
                if not signal.is_file() or not labels.is_file():
                    raise FileNotFoundError(f"OpenBMI MI cache missing: {base}")
                x = np.load(signal, mmap_mode="r", allow_pickle=False)
                y = np.load(labels, mmap_mode="r", allow_pickle=False)
                if tuple(x.shape) != (100, 62, 1000) or x.dtype != np.float32 or tuple(y.shape) != (100,):
                    raise RuntimeError(f"OpenBMI schema mismatch: {signal} {x.shape}/{x.dtype}")
                if set(map(int, np.unique(y))) != {1, 2}:
                    raise RuntimeError(f"OpenBMI labels must be 1/2: {labels}")
                rows.extend(Sample(str(subject), session, signal, i, int(y[i]) - 1) for i in range(100))
        return rows
    root = CACHE_ROOT / "wbcic" / "wbcic_epochs"
    for subject in subject_sort(wanted):
        for session in sorted(session_set):
            signal = root / subject / f"ses-{session}_epochs.npy"
            labels = root / subject / f"ses-{session}_labels.npy"
            if not signal.is_file() or not labels.is_file():
                raise FileNotFoundError(f"WBCIC MI cache missing: {subject} session {session}")
            x = np.load(signal, mmap_mode="r", allow_pickle=False)
            y = np.load(labels, mmap_mode="r", allow_pickle=False)
            if x.ndim != 3 or tuple(x.shape[1:]) != (58, 1000) or x.dtype != np.float16 or y.shape != (x.shape[0],):
                raise RuntimeError(f"WBCIC schema mismatch: {signal} {x.shape}/{x.dtype}")
            if set(map(int, np.unique(y))) != {0, 1}:
                raise RuntimeError(f"WBCIC labels must be 0/1: {labels}")
            rows.extend(Sample(str(subject), session, signal, i, int(y[i])) for i in range(x.shape[0]))
    return rows


def load_subset(dataset: str, subjects: Sequence[str], sessions: Sequence[int]) -> Subset:
    rows = _sample_rows(dataset, subjects, sessions)
    if not rows:
        raise RuntimeError(f"empty subset: {dataset} {subjects} {sessions}")
    channels = 62 if dataset == "OpenBMI" else 58
    out = np.empty((len(rows), channels, 1000), dtype=np.float32)
    arrays: dict[str, np.ndarray] = {}
    for i, row in enumerate(rows):
        key = str(row.signal_path)
        if key not in arrays:
            arrays[key] = np.load(row.signal_path, mmap_mode="r", allow_pickle=False)
        out[i] = np.asarray(arrays[key][row.index], dtype=np.float32)
    return Subset(
        x=out,
        labels=np.asarray([r.label for r in rows], dtype=np.int64),
        subjects=np.asarray([r.subject for r in rows], dtype=object),
        sessions=np.asarray([r.session for r in rows], dtype=np.int64),
        channels=channels,
    )


def normalizer(dataset: str, train_subjects: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    subset = load_subset(dataset, train_subjects, SOURCE_SESSIONS[dataset])
    n = subset.x.shape[0] * subset.x.shape[2]
    total = subset.x.sum(axis=(0, 2), dtype=np.float64)
    square = np.square(subset.x, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
    mean = (total / n).astype(np.float32)
    std = np.sqrt(np.maximum(square / n - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    return mean, std, {
        "dataset": dataset,
        "subjects": subject_sort(train_subjects),
        "sessions": list(SOURCE_SESSIONS[dataset]),
        "trials": int(len(subset.labels)),
        "mean_std_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest(),
    }


def normalize(subset: Subset, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((subset.x - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32, copy=False)


class EEGNetExact(nn.Module):
    def __init__(self, channels: int, samples: int = 1000, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4)); self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False); self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8)); self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(16 * (samples // 4 // 8), 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.head(self.embedding(value.flatten(1)))


def _norm_layer(kind: str, channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels) if kind == "bn" else nn.GroupNorm(4 if channels in (8, 16) else 8, channels)


class LiteBNExact(nn.Module):
    def __init__(self, channels: int, kind: str = "bn"):
        super().__init__()
        self.temporal = nn.ModuleList([nn.Conv2d(1, 8, (1, k), padding="same", bias=False) for k in (15, 63, 127)])
        self.temporal_norm = nn.ModuleList([_norm_layer(kind, 8) for _ in range(3)])
        self.spatial = nn.ModuleList([nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False) for _ in range(3)])
        self.spatial_norm = nn.ModuleList([_norm_layer(kind, 16) for _ in range(3)])
        self.depth1 = nn.Conv2d(48, 48, (1, 15), groups=48, padding="same", bias=False)
        self.point1 = nn.Conv2d(48, 64, 1, bias=False); self.norm1 = _norm_layer(kind, 64)
        self.depth2 = nn.Conv2d(64, 64, (1, 31), groups=64, padding="same", bias=False)
        self.point2 = nn.Conv2d(64, 64, 1, bias=False); self.norm2 = _norm_layer(kind, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.embedding = nn.Sequential(nn.Linear(512, 64), nn.ELU(), nn.LayerNorm(64))
        self.drop = nn.Dropout(.25); self.head = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1); branches = []
        for temporal, temporal_norm, spatial, spatial_norm in zip(self.temporal, self.temporal_norm, self.spatial, self.spatial_norm):
            y = F.elu(temporal_norm(temporal(value))); y = F.elu(spatial_norm(spatial(y))); y = F.avg_pool2d(y, (1, 4))
            branches.append(F.dropout(y, .20, self.training))
        value = torch.cat(branches, 1)
        value = F.dropout(F.avg_pool2d(F.elu(self.norm1(self.point1(self.depth1(value)))), (1, 2)), .15, self.training)
        value = F.dropout(F.avg_pool2d(F.elu(self.norm2(self.point2(self.depth2(value)))), (1, 2)), .15, self.training)
        z = self.drop(self.embedding(self.pool(value).flatten(1)))
        return self.head(z)


class STWrapper(nn.Module):
    """Official ST-EEGFormer-small with an explicit 250 Hz -> 128 Hz adapter."""
    def __init__(self, channels: int):
        super().__init__()
        root = OFFICIAL_ROOT / "STEEGFormer" / "eeg_foundation_2025"
        sys.path.insert(0, str(root / "utils"))
        import models_vit_eeg as st_models  # type: ignore
        base = st_models.vit_small_patch16(global_pool="token", num_classes=2, head_drop_out=0.1)
        ckpt_path = OFFICIAL_ROOT / "STEEGFormer" / "checkpoints" / "checkpoint-300.pth"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        source = checkpoint.get("model", checkpoint)
        target = base.state_dict()
        compatible = {k: v for k, v in source.items() if k in target and tuple(target[k].shape) == tuple(v.shape)}
        # The released pretraining checkpoint has a 145-row channel table;
        # zero-pad it while using deterministic 0..C-1 channel indices.
        if "enc_channel_emd.channel_transformation.weight" in source:
            weight = torch.zeros_like(target["enc_channel_emd.channel_transformation.weight"])
            src_weight = source["enc_channel_emd.channel_transformation.weight"]
            weight[: min(weight.shape[0], src_weight.shape[0])] = src_weight[: weight.shape[0]]
            compatible["enc_channel_emd.channel_transformation.weight"] = weight
        base.load_state_dict(compatible, strict=False)
        base.default_chan_idx = torch.arange(channels, dtype=torch.long)
        self.base = base
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Carrier cache is 250 Hz × 4 s; ST-EEGFormer pretraining is 128 Hz.
        adapted = F.interpolate(x, size=512, mode="linear", align_corners=False)
        tokens = self.base._forward_tokens(adapted, None)
        return self.base.cls_head(tokens)


class LaBraMWrapper(nn.Module):
    """Official LaBraM-base with a declared 250 Hz -> 200 Hz adapter."""
    def __init__(self, channels: int):
        super().__init__()
        root = OFFICIAL_ROOT / "LaBraM"
        sys.path.insert(0, str(root))
        import modeling_finetune  # type: ignore
        encoder = modeling_finetune.labram_base_patch200_200(
            pretrained=False, num_classes=0, use_mean_pooling=True,
            use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1,
            qkv_bias=False,
        )
        checkpoint = torch.load(root / "checkpoints" / "labram-base.pth", map_location="cpu", weights_only=False)
        source = checkpoint.get("model", checkpoint)
        cleaned = {}
        for key, value in source.items():
            key = key.removeprefix("student.")
            if not key.startswith("head."):
                cleaned[key] = value
        encoder.load_state_dict(cleaned, strict=False)
        self.encoder = encoder
        self.input_chans = [0] + list(range(1, channels + 1))
        self.head = nn.Linear(200, 2)
        nn.init.trunc_normal_(self.head.weight, std=.02); nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapted = F.interpolate(x, size=800, mode="linear", align_corners=False)
        adapted = adapted.reshape(adapted.shape[0], adapted.shape[1], 4, 200)
        features = self.encoder.forward_features(adapted, input_chans=self.input_chans)
        return self.head(features)


class CBraModWrapper(nn.Module):
    """Official CBraMod with the released all-patch representation adapter."""
    def __init__(self, channels: int):
        super().__init__()
        root = OFFICIAL_ROOT / "CBraMod"
        sys.path.insert(0, str(root))
        from models.cbramod import CBraMod  # type: ignore
        encoder = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        checkpoint = torch.load(root / "pretrained_weights" / "pretrained_weights.pth", map_location="cpu", weights_only=True)
        encoder.load_state_dict(checkpoint, strict=True)
        encoder.proj_out = nn.Identity()
        self.encoder = encoder
        self.channels = channels
        self.head = nn.Linear(200, 2)
        nn.init.trunc_normal_(self.head.weight, std=.02); nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Five 200-sample patches preserve the carrier cache samples exactly;
        # this is explicitly a model-native patch adapter, not a new split.
        patches = x.reshape(x.shape[0], x.shape[1], 5, 200)
        reps = self.encoder(patches)
        features = reps.mean(dim=(1, 2))
        return self.head(features)


def build_model(name: str, dataset: str, channels: int) -> nn.Module:
    if name == "EEGNet":
        return EEGNetExact(channels)
    if name == "LiteBN":
        return LiteBNExact(channels)
    if name == "TCFormer":
        root = OFFICIAL_ROOT / "TCFormer"
        sys.path.insert(0, str(root))
        from models.tcformer import TCFormerModule  # type: ignore
        return TCFormerModule(n_channels=channels, n_classes=2, F1=16, temp_kernel_lengths=(16, 32, 64),
                              pool_length_1=8, pool_length_2=7, D=2, dropout_conv=.3, d_group=16,
                              tcn_depth=2, kernel_length_tcn=4, dropout_tcn=.3, use_group_attn=True,
                              q_heads=8, kv_heads=4, trans_depth=5, trans_dropout=.4)
    if name == "ST-EEGFormer-small":
        return STWrapper(channels)
    if name == "LaBraM-base":
        return LaBraMWrapper(channels)
    if name == "CBraMod":
        return CBraModWrapper(channels)
    raise KeyError(name)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def old_checkpoint(dataset: str, fold: int, model: str) -> Path:
    return CARRIER_RUNTIME / f"{dataset.lower()}_fold{fold}_seed0_{model.lower()}" / "selected_best.pt"


def modern_slug(name: str) -> str:
    return {"TCFormer": "tcformer", "ST-EEGFormer-small": "steegformer_small", "LaBraM-base": "labram_base", "CBraMod": "cbramod"}[name]


def modern_dir(dataset: str, fold: int, model: str) -> Path:
    return RUNTIME / "checkpoints" / dataset.lower() / f"fold{fold}_{modern_slug(model)}"


def metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    pred = logits.argmax(1)
    unique = subject_sort(np.unique(subjects.astype(str)))
    ba = [balanced_accuracy_score(labels[subjects.astype(str) == s], pred[subjects.astype(str) == s]) for s in unique]
    f1 = [f1_score(labels[subjects.astype(str) == s], pred[subjects.astype(str) == s], average="macro", zero_division=0) for s in unique]
    acc = [accuracy_score(labels[subjects.astype(str) == s], pred[subjects.astype(str) == s]) for s in unique]
    return {"BA": float(np.mean(ba)), "macro_F1": float(np.mean(f1)), "accuracy": float(np.mean(acc)), "subjects": int(len(unique)), "trials": int(len(labels))}


def load_state(model: nn.Module, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)


def load_model_for_inference(name: str, dataset: str, fold: int, device: torch.device) -> nn.Module:
    channels = 62 if dataset == "OpenBMI" else 58
    model = build_model(name, dataset, channels)
    if name in ("EEGNet", "LiteBN"):
        path = old_checkpoint(dataset, fold, name)
    else:
        path = modern_dir(dataset, fold, name) / "selected_best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    load_state(model, path)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def infer(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = BATCH_SIZE) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size])).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(batch)
            outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)

