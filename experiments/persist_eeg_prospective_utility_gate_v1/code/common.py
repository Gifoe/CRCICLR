"""Shared implementation for the frozen prospective-utility mechanism audit.

This module intentionally does not import the historical split/data loaders:
some of them enumerate the internal holdout before filtering.  Only the
already-materialized, exactly enumerated 40-subject development cache is read.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
PROTOCOL_PATH = EXP / "UTILITY_GATE_PROTOCOL_FROZEN.json"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"
RUNS = RUNTIME / "runs"

EPS = 1e-12
RIDGE_ALPHA = 1.0
METHODS = ("ERM",)
BACKBONES = ("eegnet", "eegconformer")


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol() -> dict[str, Any]:
    value = read_json(PROTOCOL_PATH)
    if value.get("schema") != "PERSIST_EEG_PROSPECTIVE_UTILITY_GATE_V1":
        raise RuntimeError("unexpected utility-gate protocol schema")
    if value.get("frozen_before_training") is not True or value.get("frozen_before_outcome_evaluation") is not True:
        raise RuntimeError("utility-gate protocol was not frozen before execution")
    if value.get("repository_start_sha") != "32f88afa76ace3a19ab9c2cfefc1c0b916fd3eb3":
        raise RuntimeError("starting provenance changed")
    return value


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, RUNTIME, RUNS):
        path.mkdir(parents=True, exist_ok=True)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass
class DevelopmentData:
    x: np.ndarray
    metadata: pd.DataFrame
    cache_root: Path


def subject_sort(values: Iterable[str]) -> list[str]:
    return sorted(map(str, values), key=lambda item: (int(item) if item.isdigit() else 10**9, item))


def frozen_subjects() -> tuple[str, ...]:
    return tuple(subject_sort(protocol()["dataset"]["subject_pool"]))


def frozen_fold(fold: int) -> dict[str, tuple[str, ...]]:
    row = next(item for item in protocol()["dataset"]["folds"] if int(item["fold"]) == int(fold))
    roles = {
        "fit_train": tuple(subject_sort(row["fit_train"])),
        "fit_validation": tuple(subject_sort(row["fit_validation"])),
        "pseudo_target": tuple(subject_sort(row["pseudo_target"])),
        "outcome": tuple(subject_sort(row["outcome"])),
    }
    source = tuple(subject_sort(set(roles["fit_train"]) | set(roles["fit_validation"]) | set(roles["pseudo_target"])))
    if tuple(map(len, (roles["fit_train"], roles["fit_validation"], roles["pseudo_target"], roles["outcome"], source))) != (20, 4, 8, 8, 32):
        raise RuntimeError(f"fold {fold} cardinality failure")
    role_sets = [set(value) for value in roles.values()]
    if set.union(*role_sets) != set(frozen_subjects()) or any(role_sets[i] & role_sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise RuntimeError(f"fold {fold} subject overlap")
    return {**roles, "source": source}


def _cache_candidates() -> list[Path]:
    values: list[Path] = []
    explicit = os.environ.get("PERSIST_UTILITY_DATA_CACHE", "").strip()
    if explicit:
        values.append(Path(explicit))
    legacy_lock = REPO / "experiments" / "persist_eeg_persist_net_final_v1" / "runtime" / "runs" / "fold-0" / "seed-0" / "RUN_LOCK.json"
    if legacy_lock.is_file():
        try:
            values.append(Path(str(read_json(legacy_lock)["normalizer"])).parent)
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
    values.append(REPO / "experiments" / "persist_eeg_persist_net_final_v1" / "runtime" / "cache")
    # Server-local authorized cache used by the frozen closure. This path is a
    # cache location, not a fallback raw-data or split loader.
    values.append(Path(r"D:\nips-temp\TotalP\P1\CRCICLR_EXP4_HEADROOM_AUDIT\experiments\persist_eeg_persist_net_final_v1\runtime\cache"))
    unique: list[Path] = []
    for value in values:
        resolved = value.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def locate_authorized_cache() -> Path:
    for root in _cache_candidates():
        lower = str(root).lower()
        if "wbcic" in lower or "holdout" in lower:
            continue
        signal = root / "OPENBMI_V8_SEARCH_MI_RAW.npy"
        metadata = root / "OPENBMI_V8_SEARCH_MI_METADATA.parquet"
        if signal.is_file() and metadata.is_file():
            return root
    raise FileNotFoundError("authorized 40-subject OpenBMI cache not found; set PERSIST_UTILITY_DATA_CACHE")


def load_data(label_subjects: Sequence[str] | None = None) -> DevelopmentData:
    """Load signals plus labels only for the caller-authorized subject role.

    The subject/session header is read without the label column so global row
    indices remain aligned with the signal array.  Predicate pushdown then
    materializes labels only for ``label_subjects``.  This makes the source and
    outer-outcome phases mechanically separable at the loader boundary.
    """
    root = locate_authorized_cache()
    meta_path = root / "OPENBMI_V8_SEARCH_MI_METADATA.parquet"
    x_path = root / "OPENBMI_V8_SEARCH_MI_RAW.npy"
    metadata = pd.read_parquet(meta_path, columns=["subject_id", "session_id"], engine="pyarrow")
    subject_dtype = metadata.subject_id.dtype
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = -1
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    expected = set(frozen_subjects())
    observed = set(metadata.subject_id.unique())
    valid = bool(
        len(metadata) == 8000
        and x.shape == (8000, 62, 1000)
        and observed == expected
        and set(metadata.session_id.unique()) == {1, 2}
        and set(map(int, metadata.groupby(["subject_id", "session_id"]).size().tolist())) == {100}
        and np.isfinite(np.asarray(x[::197], dtype=np.float32)).all()
    )
    if not valid:
        raise RuntimeError(f"authorized-cache header audit failed rows={len(metadata)} shape={x.shape} subjects={subject_sort(observed)}")
    if label_subjects is not None:
        allowed = set(map(str, label_subjects))
        if not allowed or not allowed <= expected:
            raise RuntimeError("invalid label-subject scope")
        native_allowed: list[Any] = [int(value) for value in allowed] if is_integer_dtype(subject_dtype) else sorted(allowed)
        filtered = pd.read_parquet(
            meta_path,
            columns=["subject_id", "session_id", "label"],
            filters=[("subject_id", "in", native_allowed)],
            engine="pyarrow",
        )
        filtered["subject_id"] = filtered.subject_id.astype(str)
        filtered["session_id"] = filtered.session_id.astype(int)
        selected = np.flatnonzero(metadata.subject_id.isin(allowed).to_numpy())
        if len(filtered) != len(selected):
            raise RuntimeError("filtered label cardinality mismatch")
        header = metadata.iloc[selected][["subject_id", "session_id"]].reset_index(drop=True)
        if not header.equals(filtered[["subject_id", "session_id"]].reset_index(drop=True)):
            raise RuntimeError("filtered label order mismatch")
        labels = filtered.label.to_numpy(np.int64)
        if set(map(int, np.unique(labels))) != {0, 1}:
            raise RuntimeError("filtered labels are not binary")
        metadata.loc[selected, "label"] = labels
        cells = metadata.iloc[selected].groupby(["subject_id", "session_id", "label"]).size()
        if set(map(int, cells.tolist())) != {50}:
            raise RuntimeError("filtered label cell cardinality changed")
    return DevelopmentData(x=x, metadata=metadata.reset_index(drop=True), cache_root=root)


def row_indices(metadata: pd.DataFrame, subjects: Sequence[str], sessions: Sequence[int]) -> np.ndarray:
    mask = metadata.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy(copy=True)
    mask &= metadata.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def materialize_subject_scope(data: DevelopmentData, subjects: Sequence[str]) -> DevelopmentData:
    """Copy only the explicitly authorized subject signals into process memory."""
    scope = set(map(str, subjects))
    indices = np.flatnonzero(data.metadata.subject_id.astype(str).isin(scope).to_numpy()).astype(np.int64)
    metadata = data.metadata.iloc[indices].reset_index(drop=True)
    if len(metadata) != 200 * len(scope) or np.any(metadata.label.to_numpy(np.int64) < 0):
        raise RuntimeError("subject-scoped materialization cardinality/label failure")
    signal = np.asarray(data.x[indices], dtype=np.float32).copy()
    return DevelopmentData(x=signal, metadata=metadata, cache_root=data.cache_root)


def compute_normalizer(raw: torch.Tensor, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(raw.shape[1], dtype=torch.float64, device=raw.device)
    square = torch.zeros_like(total)
    count = 0
    for start in range(0, len(indices), 256):
        idx = torch.as_tensor(indices[start:start + 256], dtype=torch.long, device=raw.device)
        value = raw[idx].float()
        total += value.sum(dim=(0, 2), dtype=torch.float64)
        square += value.square().sum(dim=(0, 2), dtype=torch.float64)
        count += int(value.shape[0] * value.shape[2])
    mean64 = total / count
    variance = torch.clamp(square / count - mean64.square(), min=1e-12)
    return mean64.float(), variance.sqrt().float()


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-6)


class StandardEEGNet(nn.Module):
    """Authoritative OpenBMI EEGNet baseline (F1=8, D=2, F2=16)."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (62, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        self.embedding = nn.Sequential(nn.Linear(16 * 31, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)
        self.representation_dim = 64

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class CompactEEGConformer(nn.Module):
    """Validated V7 OpenBMI CompactEEGConformer with a shared input boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.spatial = nn.Conv2d(40, 40, (62, 1), bias=False)
        self.norm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 25), stride=(1, 10))
        self.dropout = nn.Dropout(0.4)
        layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=160,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(40), enable_nested_tensor=False)
        self.position = nn.Parameter(torch.zeros(1, 100, 40))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.embedding = nn.Sequential(nn.Linear(40, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)
        self.representation_dim = 64

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # V7 used a 3% channel-drop augmentation for this backbone.
        if self.training:
            keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= 0.03).to(x.dtype)
            x = x * keep
        value = self.dropout(self.pool(F.elu(self.norm(self.spatial(self.temporal(x.unsqueeze(1)))))))
        token = value.squeeze(2).transpose(1, 2)
        token = token + self.position[:, : token.shape[1]]
        token = self.transformer(token)
        return self.embedding(token.mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(backbone: str, initialization_seed: int) -> nn.Module:
    set_seed(initialization_seed)
    if backbone == "eegnet":
        return StandardEEGNet()
    if backbone == "eegconformer":
        return CompactEEGConformer()
    raise KeyError(backbone)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (-grad_output,)


def coral_penalty(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    h = features.float()
    covariances: list[torch.Tensor] = []
    for domain in torch.unique(domains, sorted=True):
        value = h[domains == domain]
        if len(value) < 2:
            continue
        value = value - value.mean(dim=0, keepdim=True)
        covariances.append(value.T @ value / (len(value) - 1))
    if len(covariances) < 2:
        return h.new_zeros(())
    cov = torch.stack(covariances)
    centered = cov - cov.mean(dim=0, keepdim=True)
    # Mean over all unordered domain pairs, expressed without a pair loop.
    pair_mean_sq = 2.0 * centered.square().sum() / (len(covariances) - 1)
    dim = h.shape[1]
    return pair_mean_sq / (4.0 * dim * dim)


def mmd_penalty(features: torch.Tensor, domains: torch.Tensor, bandwidths: Sequence[float]) -> torch.Tensor:
    h = features.float()
    unique, inverse = torch.unique(domains, sorted=True, return_inverse=True)
    if len(unique) < 2:
        return h.new_zeros(())
    distance2 = torch.cdist(h, h, p=2).square().clamp_min(0.0)
    kernel = torch.zeros_like(distance2)
    for sigma in bandwidths:
        kernel += torch.exp(-distance2 / (2.0 * max(float(sigma), 1e-6) ** 2))
    kernel /= len(bandwidths)
    assignment = F.one_hot(inverse, num_classes=len(unique)).T.float()
    assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1.0)
    domain_kernel = assignment @ kernel @ assignment.T
    diagonal = torch.diag(domain_kernel)
    pair = diagonal[:, None] + diagonal[None, :] - 2.0 * domain_kernel
    upper = torch.triu_indices(len(unique), len(unique), offset=1, device=h.device)
    return pair[upper[0], upper[1]].clamp_min(0.0).mean()


def subject_mean_ba(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    prediction = np.asarray(logits).argmax(axis=1)
    values = []
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects.astype(str) == subject
        values.append(balanced_accuracy_score(labels[mask], prediction[mask]))
    return float(np.mean(values))


def evaluate_model(
    model: nn.Module,
    raw: torch.Tensor,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    if np.any(metadata.iloc[indices].label.to_numpy(np.int64) < 0):
        raise RuntimeError("evaluation attempted outside the authorized label scope")
    model.eval()
    all_logits: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx_np = indices[start:start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=raw.device)
            x = normalize(raw[idx].float(), mean, std)
            with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16, enabled=raw.device.type == "cuda"):
                features = model.forward_features(x)
                logits = model.head(features)
            all_features.append(features.float().cpu().numpy())
            all_logits.append(logits.float().cpu().numpy())
    selected = metadata.iloc[indices]
    return {
        "features": np.concatenate(all_features).astype(np.float32),
        "logits": np.concatenate(all_logits).astype(np.float32),
        "labels": selected.label.to_numpy(np.int64),
        "subjects": selected.subject_id.astype(str).to_numpy(),
        "sessions": selected.session_id.to_numpy(np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
    }


def determine_mmd_bandwidths(
    backbone: str,
    initialization_seed: int,
    raw: torch.Tensor,
    train_indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[float]:
    model = build_model(backbone, initialization_seed).to(raw.device)
    model.eval()
    rng = np.random.default_rng(stable_seed("mmd-bandwidth-sample", backbone, initialization_seed))
    count = min(512, len(train_indices))
    sample = np.sort(rng.choice(train_indices, size=count, replace=False))
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(sample), 256):
            idx = torch.as_tensor(sample[start:start + 256], dtype=torch.long, device=raw.device)
            x = normalize(raw[idx].float(), mean, std)
            values.append(model.forward_features(x).float())
    features = torch.cat(values)
    distances = torch.pdist(features, p=2)
    positive = distances[distances > 1e-8]
    median = float(torch.median(positive).cpu()) if len(positive) else 1.0
    del model, features, distances, positive
    return [0.5 * median, median, 2.0 * median]


def train_model(
    backbone: str,
    method: str,
    lam: float,
    raw: torch.Tensor,
    metadata: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    initialization_seed: int,
    loader_seed: int,
    bandwidths: Sequence[float],
) -> tuple[nn.Module, dict[str, Any]]:
    cfg = protocol()
    train_cfg = cfg["training"]
    backbone_cfg = cfg["backbones"][backbone]
    model = build_model(backbone, initialization_seed).to(raw.device)
    initial_sha = state_sha256(model)
    train_subjects = subject_sort(metadata.iloc[train_indices].subject_id.unique())
    subject_map = {subject: index for index, subject in enumerate(train_subjects)}
    subject_codes = np.asarray([subject_map.get(str(subject), -1) for subject in metadata.subject_id], dtype=np.int64)
    subject_head: nn.Module | None = None
    parameters = list(model.parameters())
    if method == "DANN":
        set_seed(stable_seed("subject-head", backbone, initialization_seed))
        subject_head = nn.Sequential(
            nn.Linear(int(model.representation_dim), 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(train_subjects)),
        ).to(raw.device)
        parameters += list(subject_head.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(backbone_cfg["lr"]),
        weight_decay=float(backbone_cfg["weight_decay"]),
    )
    batch_size = int(backbone_cfg["batch_size"])
    max_epochs = int(train_cfg["max_epochs"])
    min_epochs = int(train_cfg["minimum_epochs"])
    patience = int(train_cfg["patience"])
    labels = metadata.label.to_numpy(np.int64)
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float, int] | None = None
    best_epoch = max_epochs
    best_ba = float("nan")
    best_nll = float("nan")
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    epoch0_permutation_sha = ""
    for epoch in range(max_epochs):
        model.train()
        if subject_head is not None:
            subject_head.train()
        rng = np.random.default_rng(int(loader_seed) + epoch)
        permutation = rng.permutation(train_indices)
        if epoch == 0:
            epoch0_permutation_sha = array_sha256(permutation.astype(np.int64))
        task_total = 0.0
        penalty_total = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            idx_np = permutation[start:start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=raw.device)
            y = torch.as_tensor(labels[idx_np], dtype=torch.long, device=raw.device)
            domain = torch.as_tensor(subject_codes[idx_np], dtype=torch.long, device=raw.device)
            if torch.any(domain < 0):
                raise RuntimeError("training minibatch crossed the frozen inner-train subjects")
            x = normalize(raw[idx].float(), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16, enabled=raw.device.type == "cuda"):
                h = model.forward_features(x)
                task_logits = model.head(h)
                task_loss = F.cross_entropy(task_logits, y)
                if method == "ERM":
                    penalty = h.float().new_zeros(())
                elif method == "DANN":
                    assert subject_head is not None
                    penalty = F.cross_entropy(subject_head(GradientReverse.apply(h)), domain)
                elif method == "CORAL":
                    penalty = coral_penalty(h, domain)
                elif method == "MMD":
                    penalty = mmd_penalty(h, domain, bandwidths)
                else:
                    raise KeyError(method)
                loss = task_loss.float() + float(lam) * penalty.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            task_total += float(task_loss.detach().float().cpu()) * len(idx_np)
            penalty_total += float(penalty.detach().float().cpu()) * len(idx_np)
            seen += len(idx_np)
        validation = evaluate_model(model, raw, metadata, validation_indices, mean, std, batch_size=512)
        val_ba = subject_mean_ba(validation["labels"], validation["logits"], validation["subjects"])
        probability = torch.softmax(torch.from_numpy(validation["logits"]), dim=1)[:, 1].numpy()
        val_nll = float(log_loss(validation["labels"], np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]))
        row = {
            "epoch": epoch + 1,
            "train_task_CE": task_total / max(seen, 1),
            "train_invariance_penalty": penalty_total / max(seen, 1),
            "validation_mean_subject_BA": val_ba,
            "validation_NLL": val_nll,
        }
        history.append(row)
        key = (val_ba, -val_nll, -(epoch + 1))
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch = epoch + 1
            best_ba = val_ba
            best_nll = val_nll
            stale = 0
        else:
            stale += 1
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(
                f"[{backbone} {method} lambda={lam:g}] epoch={epoch + 1} "
                f"task={row['train_task_CE']:.4f} inv={row['train_invariance_penalty']:.4f} valBA={val_ba:.4f}",
                flush=True,
            )
        if epoch + 1 >= min_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metadata_out = {
        "backbone": backbone,
        "method": method,
        "lambda": float(lam),
        "initialization_seed": int(initialization_seed),
        "loader_seed": int(loader_seed),
        "initial_shared_state_sha256": initial_sha,
        "epoch0_minibatch_order_sha256": epoch0_permutation_sha,
        "best_epoch": int(best_epoch),
        "epochs_executed": int(len(history)),
        "best_validation_BA": float(best_ba),
        "best_validation_NLL": float(best_nll),
        "elapsed_seconds": float(time.time() - started),
        "history": history,
        "outcome_labels_used": False,
        "outcome_rows_used": False,
        "restricted_data_accessed": False,
        "DANN_subject_head": "Linear64_128_ReLU_Dropout0.2_Linear128_K" if method == "DANN" else None,
    }
    return model, metadata_out


def ridge_pack(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    z = np.c_[(x - mean) / std, np.ones(len(x))]
    target = np.eye(int(y.max()) + 1, dtype=np.float64)[y]
    penalty = np.eye(z.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    lhs = z.T @ z + RIDGE_ALPHA * penalty
    try:
        weight = np.linalg.solve(lhs, z.T @ target)
    except np.linalg.LinAlgError:
        weight = np.linalg.pinv(lhs) @ z.T @ target
    return weight, mean, std


def ridge_logits(x: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weight, mean, std = pack
    z = np.c_[(np.asarray(x, dtype=np.float64) - mean) / std, np.ones(len(x))]
    return z @ weight


def identity_direction(
    features: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    train_session: int,
    eval_session: int,
) -> dict[str, float]:
    ordered = subject_sort(np.unique(subjects.astype(str)))
    code = {subject: index for index, subject in enumerate(ordered)}
    train = np.flatnonzero(sessions.astype(int) == int(train_session))
    evaluate = np.flatnonzero(sessions.astype(int) == int(eval_session))
    y_train = np.asarray([code[subject] for subject in subjects[train].astype(str)], dtype=np.int64)
    y_eval = np.asarray([code[subject] for subject in subjects[evaluate].astype(str)], dtype=np.int64)
    pack = ridge_pack(features[train], y_train)
    logits = ridge_logits(features[evaluate], pack)
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(np.clip(logits, -60, 60))
    probability /= np.maximum(probability.sum(axis=1, keepdims=True), EPS)
    ce = -np.log(np.clip(probability[np.arange(len(y_eval)), y_eval], EPS, 1.0)).mean()
    prediction = probability.argmax(axis=1)
    per_subject = [np.mean(prediction[y_eval == index] == index) for index in range(len(ordered))]
    accuracy = float(np.mean(per_subject))
    return {"skill": float(math.log(len(ordered)) - ce), "ce": float(ce), "accuracy": accuracy}


def identity_probe(features: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> dict[str, float]:
    forward = identity_direction(features, subjects, sessions, 1, 2)
    reverse = identity_direction(features, subjects, sessions, 2, 1)
    symmetric_skill = float(np.mean([forward["skill"], reverse["skill"]]))
    symmetric_accuracy = float(np.mean([forward["accuracy"], reverse["accuracy"]]))
    chance = 1.0 / len(np.unique(subjects.astype(str)))
    return {
        "identity_S1_to_S2": forward["skill"],
        "identity_S2_to_S1": reverse["skill"],
        "identity_symmetric": symmetric_skill,
        "identity_accuracy_S1_to_S2": forward["accuracy"],
        "identity_accuracy_S2_to_S1": reverse["accuracy"],
        "identity_accuracy_symmetric": symmetric_accuracy,
        "chance_accuracy": chance,
        "chance_normalized_identity": (symmetric_accuracy - chance) / max(1.0 - chance, EPS),
        "subject_count": int(len(np.unique(subjects.astype(str)))),
    }


def per_subject_performance(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    prediction = logits.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for subject in subject_sort(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        rows.append(
            {
                "subject_id": subject,
                "BA": float(balanced_accuracy_score(labels[mask], prediction[mask])),
                "macro_f1": float(f1_score(labels[mask], prediction[mask], average="macro")),
            }
        )
    return pd.DataFrame(rows)


def per_subject_utility(
    labels: np.ndarray,
    intact_logits: np.ndarray,
    erased_logits: np.ndarray,
    subjects: np.ndarray,
) -> pd.DataFrame:
    """Suppression utility; positive BA/F1/CE values consistently mean help."""
    intact_prediction = np.asarray(intact_logits).argmax(axis=1)
    erased_prediction = np.asarray(erased_logits).argmax(axis=1)
    intact_ce = numpy_cross_entropy(intact_logits, labels)
    erased_ce = numpy_cross_entropy(erased_logits, labels)
    rows: list[dict[str, Any]] = []
    for subject in subject_sort(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        intact_ba = balanced_accuracy_score(labels[mask], intact_prediction[mask])
        erased_ba = balanced_accuracy_score(labels[mask], erased_prediction[mask])
        intact_f1 = f1_score(labels[mask], intact_prediction[mask], average="macro")
        erased_f1 = f1_score(labels[mask], erased_prediction[mask], average="macro")
        rows.append(
            {
                "subject_id": subject,
                "BA_intact": float(intact_ba),
                "BA_erased": float(erased_ba),
                "U_BA": float(erased_ba - intact_ba),
                "F1_intact": float(intact_f1),
                "F1_erased": float(erased_f1),
                "U_F1": float(erased_f1 - intact_f1),
                "CE_intact": float(np.mean(intact_ce[mask])),
                "CE_erased": float(np.mean(erased_ce[mask])),
                "U_CE": float(np.mean(intact_ce[mask] - erased_ce[mask])),
            }
        )
    return pd.DataFrame(rows)


def utility_summary(frame: pd.DataFrame, prefix: str, bootstrap_seed: int, draws: int = 2000) -> dict[str, Any]:
    values = frame.U_BA.to_numpy(np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    boot = values[rng.integers(0, len(values), size=(int(draws), len(values)))].mean(axis=1)
    return {
        f"{prefix}_BA": float(values.mean()),
        f"{prefix}_BA_median_subject": float(np.median(values)),
        f"{prefix}_BA_ci_low": float(np.quantile(boot, 0.025)),
        f"{prefix}_BA_ci_high": float(np.quantile(boot, 0.975)),
        f"{prefix}_F1": float(frame.U_F1.mean()),
        f"{prefix}_CE": float(frame.U_CE.mean()),
        f"{prefix}_positive_subjects": int((values > 0).sum()),
        f"{prefix}_negative_subjects": int((values < 0).sum()),
        f"{prefix}_tied_subjects": int((values == 0).sum()),
        f"{prefix}_subject_count": int(len(values)),
    }


def centered_logits_np(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array - array.mean(axis=-1, keepdims=True)


def exact_d_finite(clean_logits: np.ndarray, erased_logits: np.ndarray) -> float:
    delta = np.asarray(erased_logits, dtype=np.float64) - np.asarray(clean_logits, dtype=np.float64)
    centered = centered_logits_np(delta)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=-1))))


def numpy_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value -= value.max(axis=1, keepdims=True)
    log_probability = value - np.log(np.exp(value).sum(axis=1, keepdims=True))
    return -log_probability[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)]


def persistent_directions(
    features: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
    count: int = 8,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    value = np.asarray(features, dtype=np.float64)
    center = value.mean(axis=0)
    ordered = subject_sort(np.unique(subjects.astype(str)))
    means1 = np.stack([value[(subjects.astype(str) == subject) & (sessions == 1)].mean(axis=0) for subject in ordered])
    means2 = np.stack([value[(subjects.astype(str) == subject) & (sessions == 2)].mean(axis=0) for subject in ordered])
    subject_geometry = np.concatenate((means1 - center, means2 - center), axis=0)
    _, _, vt = np.linalg.svd(subject_geometry, full_matrices=False)
    pool = vt[: min(24, len(vt))].T
    rows: list[dict[str, float]] = []
    for index in range(pool.shape[1]):
        direction = pool[:, index]
        projection1 = (means1 - center) @ direction
        projection2 = (means2 - center) @ direction
        if np.std(projection1) < 1e-12 or np.std(projection2) < 1e-12:
            persistence = 0.0
        else:
            persistence = float(np.corrcoef(projection1, projection2)[0, 1])
        geometry = float(np.sqrt(np.mean(np.square((value - center) @ direction))))
        rows.append({"pool_index": index, "persistence": persistence, "geometry_strength": geometry})
    order = sorted(range(len(rows)), key=lambda index: (-rows[index]["persistence"], -rows[index]["geometry_strength"], index))[:count]
    basis = pool[:, order]
    selected = [rows[index] for index in order]
    return center.astype(np.float64), basis.astype(np.float64), selected


def erase_direction(features: np.ndarray, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    v = v / max(float(np.linalg.norm(v)), EPS)
    return value - ((value - center) @ v)[:, None] * v[None, :]


def direction_audit(
    model: nn.Module,
    source: Mapping[str, np.ndarray],
    outcome: Mapping[str, np.ndarray],
    primary_identity_subjects: Sequence[str],
    backbone: str,
    method: str,
    lam: float,
    fold: int,
    seed: int,
) -> pd.DataFrame:
    center, basis, direction_meta = persistent_directions(
        source["features"], source["subjects"], source["sessions"], count=int(protocol()["secondary_direction_audit"]["candidate_count"])
    )
    identity_mask = np.isin(source["subjects"].astype(str), list(map(str, primary_identity_subjects)))
    full_identity = identity_probe(
        source["features"][identity_mask], source["subjects"][identity_mask], source["sessions"][identity_mask]
    )["identity_symmetric"]
    weight = model.head.weight.detach().float().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().float().cpu().numpy().astype(np.float64)
    clean_source_logits = np.asarray(source["logits"], dtype=np.float64)
    clean_outcome_logits = np.asarray(outcome["logits"], dtype=np.float64)
    clean_outcome_ce = numpy_cross_entropy(clean_outcome_logits, outcome["labels"])
    clean_outcome_prediction = clean_outcome_logits.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for direction_id, (direction, meta) in enumerate(zip(basis.T, direction_meta)):
        erased_source_features = erase_direction(source["features"], center, direction)
        erased_outcome_features = erase_direction(outcome["features"], center, direction)
        erased_source_logits = erased_source_features @ weight.T + bias
        erased_outcome_logits = erased_outcome_features @ weight.T + bias
        erased_identity = identity_probe(
            erased_source_features[identity_mask], source["subjects"][identity_mask], source["sessions"][identity_mask]
        )["identity_symmetric"]
        erased_outcome_ce = numpy_cross_entropy(erased_outcome_logits, outcome["labels"])
        erased_prediction = erased_outcome_logits.argmax(axis=1)
        subject_ba_effect: list[float] = []
        for subject in subject_sort(np.unique(outcome["subjects"].astype(str))):
            mask = outcome["subjects"].astype(str) == subject
            clean_ba = balanced_accuracy_score(outcome["labels"][mask], clean_outcome_prediction[mask])
            erased_ba = balanced_accuracy_score(outcome["labels"][mask], erased_prediction[mask])
            subject_ba_effect.append(float(clean_ba - erased_ba))
        rows.append(
            {
                "backbone": backbone,
                "method": method,
                "lambda": float(lam),
                "fold": int(fold),
                "seed": int(seed),
                "direction_id": int(direction_id),
                "source_pool_index": int(meta["pool_index"]),
                "persistence": float(meta["persistence"]),
                "geometry_strength": float(meta["geometry_strength"]),
                "rank": 1,
                "identity_score": float(full_identity - erased_identity),
                "identity_full": float(full_identity),
                "identity_erased": float(erased_identity),
                "D_finite": exact_d_finite(clean_source_logits, erased_source_logits),
                "outcome_CE_effect": float(np.mean(erased_outcome_ce - clean_outcome_ce)),
                "outcome_BA_effect": float(np.mean(subject_ba_effect)),
                "outcome_subject_count": int(len(subject_ba_effect)),
                "direction_source_only": True,
                "outcome_used_to_define_direction": False,
                "D_finite_definition": "exact_exp3_centered_logit_RMS",
            }
        )
    return pd.DataFrame(rows)


def config_slug(method: str, lam: float) -> str:
    return f"{method.lower()}__lambda-{lam:.2f}"


def configuration_grid() -> list[tuple[str, float]]:
    return [("ERM", 0.0)]


def unit_dir(backbone: str, fold: int, seed: int) -> Path:
    return RUNS / backbone / f"fold-{fold}" / f"seed-{seed}"
