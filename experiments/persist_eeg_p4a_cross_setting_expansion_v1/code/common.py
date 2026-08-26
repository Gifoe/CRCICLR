from __future__ import annotations

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"
RUNS = RUNTIME / "runs"
PROTOCOL_PATH = EXP / "P4A_PROTOCOL_FROZEN.json"

STAGE0_ROOT = Path(r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full")
OPENBMI_MANIFEST = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
OPENBMI_FOLDS = REPO / "experiments" / "persist_eeg_persist_net_final_v1" / "protocol" / "DEVELOPMENT_FOLDS.json"
P3_ROOT = REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1"
P2_ROOT = REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1"
WBCIC_SCOPE = P3_ROOT / "provenance" / "DEVELOPMENT_SCOPE_LOCK.json"
WBCIC_CACHE = P3_ROOT / "runtime" / "cache"

EPS = 1e-12
RIDGE_ALPHA = 1.0
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
LAMBDAS = (0.01, 0.1, 1.0)
METHOD_GRID = (("ERM", 0.0),) + tuple((method, lam) for method in ("DANN", "CORAL", "MMD") for lam in LAMBDAS)

SETTINGS: dict[str, dict[str, Any]] = {
    "S1": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGNet", "status": "historical"},
    "S2": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGConformer", "status": "historical"},
    "S3": {"dataset": "WBCIC", "task": "MI", "backbone": "EEGNet", "status": "historical"},
    "S4": {"dataset": "WBCIC", "task": "MI", "backbone": "EEGConformer", "status": "new"},
    "S5": {"dataset": "OpenBMI", "task": "ERP", "backbone": "EEGNet", "status": "new"},
    "S6": {"dataset": "OpenBMI", "task": "ERP", "backbone": "EEGConformer", "status": "new"},
}


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
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
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, RUNTIME, RUNS):
        path.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def text_sha256(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def tree_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


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


def subject_sort(values: Iterable[str]) -> list[str]:
    def key(item: str) -> tuple[int, str]:
        token = str(item).replace("sub-", "")
        return (int(token) if token.isdigit() else 10**9, str(item))
    return sorted(map(str, values), key=key)


def protocol() -> dict[str, Any]:
    payload = read_json(PROTOCOL_PATH)
    if payload.get("schema") != "PERSIST_EEG_P4A_CROSS_SETTING_EXPANSION_V1":
        raise RuntimeError("P4A protocol schema mismatch")
    if payload.get("future_direction_utility_sealed") is not True:
        raise RuntimeError("P4A direction utility is not sealed")
    return payload


def openbmi_roles(fold: int) -> dict[str, tuple[str, ...]]:
    payload = read_json(OPENBMI_FOLDS)
    row = next(item for item in payload["folds"] if int(item["fold"]) == int(fold))
    roles = {
        "model_fit": tuple(subject_sort(row["inner_train_subjects"])),
        "validation": tuple(subject_sort(row["inner_validation_subjects"])),
        "outcome": tuple(subject_sort(row["outcome_subjects"])),
    }
    pool = set().union(*map(set, roles.values()))
    if len(pool) != 40 or any(set(a) & set(b) for i, a in enumerate(roles.values()) for b in list(roles.values())[i + 1:]):
        raise RuntimeError(f"OpenBMI fold {fold} role failure")
    return roles


def wbcic_roles(fold: int) -> dict[str, tuple[str, ...]]:
    payload = read_json(WBCIC_SCOPE)
    if payload.get("outer_subject_ids_present") is not False:
        raise RuntimeError("WBCIC sealed subject identifiers were materialized")
    row = payload["audit_roles"][str(fold)]
    roles = {
        "model_fit": tuple(subject_sort(str(x).replace("sub-", "") for x in row["model_fit"])),
        "validation": tuple(subject_sort(str(x).replace("sub-", "") for x in row["discovery_decision"])),
        "outcome": tuple(subject_sort(str(x).replace("sub-", "") for x in row["outcome"])),
    }
    pool = set().union(*map(set, roles.values()))
    if len(pool) != 41 or any(set(a) & set(b) for i, a in enumerate(roles.values()) for b in list(roles.values())[i + 1:]):
        raise RuntimeError(f"WBCIC fold {fold} role failure")
    return roles


def roles_for(setting: str, fold: int) -> dict[str, tuple[str, ...]]:
    return openbmi_roles(fold) if SETTINGS[setting]["dataset"] == "OpenBMI" else wbcic_roles(fold)


@dataclass
class DataBundle:
    x: np.ndarray
    metadata: pd.DataFrame
    dataset: str
    task: str
    channels: int
    times: int
    source_sessions: tuple[int, int]
    future_session: int
    cache_path: Path


def erp_cache_paths() -> tuple[Path, Path]:
    return RUNTIME / "cache" / "OPENBMI_ERP_DEVELOPMENT_FLOAT16.npy", RUNTIME / "cache" / "OPENBMI_ERP_DEVELOPMENT_METADATA.parquet"


def prepare_erp_cache() -> dict[str, Any]:
    ensure_dirs()
    signal_out, meta_out = erp_cache_paths()
    audit_out = RUNTIME / "cache" / "OPENBMI_ERP_CACHE_AUDIT.json"
    if signal_out.is_file() and meta_out.is_file() and audit_out.is_file():
        return read_json(audit_out)
    whitelist = subject_sort(set().union(*(set(openbmi_roles(f)[role]) for f in FOLDS for role in ("model_fit", "validation", "outcome"))))
    frame = pd.read_parquet(
        OPENBMI_MANIFEST,
        filters=[("subject_id", "in", whitelist), ("paradigm", "==", "erp")],
        engine="pyarrow",
    ).sort_values(["subject_id", "session_id", "signal_cache_path", "cache_index"]).reset_index(drop=True)
    if len(frame) != 158400 or frame.subject_id.nunique() != 40 or frame.signal_cache_path.nunique() != 80:
        raise RuntimeError("authorized OpenBMI ERP manifest cardinality failure")
    signal_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = signal_out.with_suffix(".npy.part")
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(len(frame), 62, 250))
    for relative, group in frame.groupby("signal_cache_path", sort=True):
        source_path = STAGE0_ROOT / relative
        source = np.load(source_path, mmap_mode="r", allow_pickle=False)
        positions = group.index.to_numpy(np.int64)
        cache_indices = group.cache_index.to_numpy(np.int64)
        target[positions] = np.asarray(source[cache_indices], dtype=np.float16)
    target.flush()
    del target
    os.replace(temporary, signal_out)
    metadata = pd.DataFrame({
        "subject_id": frame.subject_id.astype(str),
        "session_id": frame.session_id.astype(int) - 1,
        "label": frame.event_code.astype(int) - 1,
        "trial_id": frame.trial_id.astype(str),
    })
    write_tmp = meta_out.with_suffix(".parquet.part")
    metadata.to_parquet(write_tmp, index=False)
    os.replace(write_tmp, meta_out)
    audit = {
        "schema": "OPENBMI_ERP_AUTHORIZED_DEVELOPMENT_CACHE_V1",
        "subjects": 40,
        "subject_scope_hash": text_sha256(whitelist),
        "rows": len(metadata),
        "shape": list(np.load(signal_out, mmap_mode="r", allow_pickle=False).shape),
        "storage_dtype": "float16",
        "source_dtype": "float32",
        "precision_adaptation": "storage-only deterministic float16 cast; no scientific preprocessing changed",
        "sessions": [0, 1],
        "labels": {"NonTarget": 0, "Target": 1},
        "class_counts": metadata.label.value_counts().sort_index().to_dict(),
        "signal_sha256": file_sha256(signal_out),
        "metadata_sha256": file_sha256(meta_out),
        "sealed_internal_holdout_eeg_accessed": False,
    }
    write_json(audit_out, audit)
    return audit


def load_data(setting: str) -> DataBundle:
    spec = SETTINGS[setting]
    if spec["dataset"] == "WBCIC":
        x_path = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_RAW.npy"
        meta_path = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
        x = np.load(x_path, mmap_mode="r", allow_pickle=False)
        metadata = pd.read_parquet(meta_path, columns=["subject_id", "session_id", "label"], engine="pyarrow")
        metadata["subject_id"] = metadata.subject_id.astype(str).str.replace("sub-", "", regex=False)
        metadata["session_id"] = metadata.session_id.astype(int)
        metadata["label"] = metadata.label.astype(int)
        if x.shape != (24591, 58, 1000) or x.dtype != np.float16 or metadata.subject_id.nunique() != 41:
            raise RuntimeError("authorized WBCIC cache audit failed")
        return DataBundle(x, metadata.reset_index(drop=True), "WBCIC", "MI", 58, 1000, (0, 1), 2, x_path)
    prepare_erp_cache()
    x_path, meta_path = erp_cache_paths()
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(meta_path, engine="pyarrow")
    if x.shape != (158400, 62, 250) or x.dtype != np.float16 or metadata.subject_id.nunique() != 40:
        raise RuntimeError("authorized OpenBMI ERP cache audit failed")
    return DataBundle(x, metadata.reset_index(drop=True), "OpenBMI", "ERP", 62, 250, (0, 1), 1, x_path)


def row_indices(metadata: pd.DataFrame, subjects: Sequence[str], sessions: Sequence[int]) -> np.ndarray:
    mask = metadata.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy()
    mask &= metadata.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def compute_normalizer(setting: str, raw: torch.Tensor, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    if SETTINGS[setting]["dataset"] == "WBCIC":
        return (
            torch.zeros(raw.shape[1], dtype=torch.float32, device=raw.device),
            torch.ones(raw.shape[1], dtype=torch.float32, device=raw.device),
        )
    total = torch.zeros(raw.shape[1], dtype=torch.float64, device=raw.device)
    square = torch.zeros(raw.shape[1], dtype=torch.float64, device=raw.device)
    count = 0
    for start in range(0, len(indices), 512):
        idx = torch.as_tensor(indices[start:start + 512], dtype=torch.long, device=raw.device)
        value = raw[idx].float()
        total += value.sum(dim=(0, 2), dtype=torch.float64)
        square += value.square().sum(dim=(0, 2), dtype=torch.float64)
        count += int(value.shape[0] * value.shape[2])
    mean64 = total / max(count, 1)
    variance = torch.clamp(square / max(count, 1) - mean64.square(), min=1e-12)
    return mean64.float(), variance.sqrt().float()


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-6)


class StandardEEGNet(nn.Module):
    def __init__(self, channels: int, times: int, representation_dim: int = 64) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        self.embedding = nn.Sequential(nn.Linear(16 * (times // 32), representation_dim), nn.ELU(), nn.LayerNorm(representation_dim))
        self.head = nn.Linear(representation_dim, 2)
        self.representation_dim = representation_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = self.bn1(self.temporal(x.unsqueeze(1)))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class CompactEEGConformer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.spatial = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.norm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 25), stride=(1, 10))
        self.dropout = nn.Dropout(0.4)
        layer = nn.TransformerEncoderLayer(d_model=40, nhead=4, dim_feedforward=160, dropout=0.3, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(40), enable_nested_tensor=False)
        self.position = nn.Parameter(torch.zeros(1, 100, 40))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.embedding = nn.Sequential(nn.Linear(40, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)
        self.representation_dim = 64

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= 0.03).to(x.dtype)
            x = x * keep
        value = self.dropout(self.pool(F.elu(self.norm(self.spatial(self.temporal(x.unsqueeze(1)))))))
        token = value.squeeze(2).transpose(1, 2)
        token = self.transformer(token + self.position[:, : token.shape[1]])
        return self.embedding(token.mean(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(setting: str, initialization_seed: int) -> nn.Module:
    set_seed(initialization_seed)
    spec = SETTINGS[setting]
    channels = 58 if spec["dataset"] == "WBCIC" else 62
    times = 1000 if spec["dataset"] == "WBCIC" else 250
    if spec["backbone"] == "EEGNet":
        return StandardEEGNet(channels, times, 64)
    return CompactEEGConformer(channels)


def evaluate_model(
    model: nn.Module,
    raw: torch.Tensor,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    model.eval()
    all_features: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
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


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (-grad_output,)


def coral_penalty(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    h = features.float()
    covariances = []
    for domain in torch.unique(domains, sorted=True):
        value = h[domains == domain]
        if len(value) >= 2:
            value = value - value.mean(dim=0, keepdim=True)
            covariances.append(value.T @ value / (len(value) - 1))
    if len(covariances) < 2:
        return h.new_zeros(())
    cov = torch.stack(covariances)
    centered = cov - cov.mean(dim=0, keepdim=True)
    return 2.0 * centered.square().sum() / ((len(covariances) - 1) * 4.0 * h.shape[1] ** 2)


def mmd_penalty(features: torch.Tensor, domains: torch.Tensor, bandwidths: Sequence[float]) -> torch.Tensor:
    h = features.float()
    unique, inverse = torch.unique(domains, sorted=True, return_inverse=True)
    if len(unique) < 2:
        return h.new_zeros(())
    distance2 = torch.cdist(h, h).square().clamp_min(0.0)
    kernel = sum(torch.exp(-distance2 / (2.0 * max(float(sigma), 1e-6) ** 2)) for sigma in bandwidths) / len(bandwidths)
    assignment = F.one_hot(inverse, num_classes=len(unique)).T.float()
    assignment /= assignment.sum(dim=1, keepdim=True).clamp_min(1.0)
    domain_kernel = assignment @ kernel @ assignment.T
    diagonal = torch.diag(domain_kernel)
    pair = diagonal[:, None] + diagonal[None, :] - 2.0 * domain_kernel
    upper = torch.triu_indices(len(unique), len(unique), offset=1, device=h.device)
    return pair[upper[0], upper[1]].clamp_min(0.0).mean()


def determine_mmd_bandwidths(
    setting: str,
    initialization_seed: int,
    raw: torch.Tensor,
    train_indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[float]:
    model = build_model(setting, initialization_seed).to(raw.device)
    model.eval()
    rng = np.random.default_rng(stable_seed("P4A-mmd-bandwidth", setting, initialization_seed))
    sample = np.sort(rng.choice(train_indices, size=min(512, len(train_indices)), replace=False))
    values: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(sample), 256):
            idx = torch.as_tensor(sample[start:start + 256], dtype=torch.long, device=raw.device)
            values.append(model.forward_features(normalize(raw[idx].float(), mean, std)).float())
    features = torch.cat(values)
    positive = torch.pdist(features, p=2)
    positive = positive[positive > 1e-8]
    median = float(torch.median(positive).cpu()) if len(positive) else 1.0
    del model, values, features, positive
    return [0.5 * median, median, 2.0 * median]


def training_class_weight(setting: str, labels: np.ndarray, train_indices: np.ndarray, device: torch.device) -> torch.Tensor | None:
    if SETTINGS[setting]["task"] != "ERP":
        return None
    counts = np.bincount(labels[train_indices], minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def train_model(
    setting: str,
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
    backbone = SETTINGS[setting]["backbone"]
    backbone_cfg = cfg["training"]["backbones"][backbone]
    model = build_model(setting, initialization_seed).to(raw.device)
    initial_sha = state_sha256(model)
    train_subjects = subject_sort(metadata.iloc[train_indices].subject_id.unique())
    subject_map = {subject: index for index, subject in enumerate(train_subjects)}
    subject_codes = np.asarray([subject_map.get(str(subject), -1) for subject in metadata.subject_id], dtype=np.int64)
    parameters = list(model.parameters())
    subject_head: nn.Module | None = None
    if method == "DANN":
        set_seed(stable_seed("P4A-subject-head", setting, initialization_seed))
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
    labels = metadata.label.to_numpy(np.int64)
    class_weight = training_class_weight(setting, labels, train_indices, raw.device)
    batch_size = int(backbone_cfg["batch_size"])
    max_epochs = int(cfg["training"]["max_epochs"])
    min_epochs = int(cfg["training"]["minimum_epochs"])
    patience = int(cfg["training"]["patience"])
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float, int] | None = None
    best_epoch, stale = max_epochs, 0
    best_ba = best_nll = float("nan")
    history: list[dict[str, Any]] = []
    epoch0_sha = ""
    started = time.time()
    for epoch in range(max_epochs):
        model.train()
        if subject_head is not None:
            subject_head.train()
        permutation = np.random.default_rng(int(loader_seed) + epoch).permutation(train_indices)
        if epoch == 0:
            epoch0_sha = array_sha256(permutation.astype(np.int64))
        task_total = penalty_total = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            idx_np = permutation[start:start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=raw.device)
            y = torch.as_tensor(labels[idx_np], dtype=torch.long, device=raw.device)
            domain = torch.as_tensor(subject_codes[idx_np], dtype=torch.long, device=raw.device)
            if torch.any(domain < 0):
                raise RuntimeError("training minibatch crossed frozen model-fit subjects")
            x = normalize(raw[idx].float(), mean, std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16, enabled=raw.device.type == "cuda"):
                h = model.forward_features(x)
                task_logits = model.head(h)
                task_loss = F.cross_entropy(task_logits, y, weight=class_weight)
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
        val_metrics = mean_subject_metrics(validation["labels"], validation["logits"], validation["subjects"])
        val_nll = float(numpy_cross_entropy(validation["logits"], validation["labels"]).mean())
        row = {
            "epoch": epoch + 1,
            "train_task_CE": task_total / max(seen, 1),
            "train_invariance_penalty": penalty_total / max(seen, 1),
            "validation_mean_subject_BA": val_metrics["BA"],
            "validation_NLL": val_nll,
        }
        history.append(row)
        key = (val_metrics["BA"], -val_nll, -(epoch + 1))
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch, best_ba, best_nll, stale = epoch + 1, val_metrics["BA"], val_nll, 0
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"[{setting} {method} lambda={lam:g}] epoch={epoch + 1} "
                f"task={row['train_task_CE']:.4f} inv={row['train_invariance_penalty']:.4f} valBA={val_metrics['BA']:.4f}",
                flush=True,
            )
        if epoch + 1 >= min_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "setting_id": setting,
        "backbone": backbone,
        "method": method,
        "lambda": float(lam),
        "initialization_seed": int(initialization_seed),
        "loader_seed": int(loader_seed),
        "initial_shared_state_sha256": initial_sha,
        "epoch0_minibatch_order_sha256": epoch0_sha,
        "best_epoch": int(best_epoch),
        "epochs_executed": len(history),
        "best_validation_BA": float(best_ba),
        "best_validation_NLL": float(best_nll),
        "elapsed_seconds": float(time.time() - started),
        "history": history,
        "class_weight": None if class_weight is None else class_weight.detach().cpu().tolist(),
        "outcome_labels_used_for_training_or_selection": False,
        "sealed_subjects_accessed": False,
    }


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def ridge_pack(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    mean, std = x.mean(0), x.std(0)
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
    return np.c_[(np.asarray(x, dtype=np.float64) - mean) / std, np.ones(len(x))] @ weight


def identity_direction(features: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, train_session: int, eval_session: int) -> dict[str, float]:
    ordered = subject_sort(np.unique(subjects.astype(str)))
    code = {subject: index for index, subject in enumerate(ordered)}
    train = np.flatnonzero(sessions.astype(int) == train_session)
    evaluate = np.flatnonzero(sessions.astype(int) == eval_session)
    y_train = np.asarray([code[s] for s in subjects[train].astype(str)], dtype=np.int64)
    y_eval = np.asarray([code[s] for s in subjects[evaluate].astype(str)], dtype=np.int64)
    logits = ridge_logits(features[evaluate], ridge_pack(features[train], y_train))
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(np.clip(logits, -60, 60))
    probability /= np.maximum(probability.sum(1, keepdims=True), EPS)
    ce = float(-np.log(np.clip(probability[np.arange(len(y_eval)), y_eval], EPS, 1.0)).mean())
    prediction = probability.argmax(1)
    accuracy = float(np.mean([np.mean(prediction[y_eval == i] == i) for i in range(len(ordered))]))
    return {"skill": float(math.log(len(ordered)) - ce), "ce": ce, "accuracy": accuracy}


def identity_probe(features: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> dict[str, float]:
    if set(np.unique(sessions.astype(int))) != {0, 1}:
        raise RuntimeError("identity probe requires the two source sessions encoded 0/1")
    forward = identity_direction(features, subjects, sessions, 0, 1)
    reverse = identity_direction(features, subjects, sessions, 1, 0)
    symmetric_skill = float(np.mean([forward["skill"], reverse["skill"]]))
    symmetric_accuracy = float(np.mean([forward["accuracy"], reverse["accuracy"]]))
    chance = 1.0 / len(np.unique(subjects.astype(str)))
    return {
        "identity_S1_to_S2": forward["skill"], "identity_S2_to_S1": reverse["skill"],
        "identity_CE_S1_to_S2": forward["ce"], "identity_CE_S2_to_S1": reverse["ce"],
        "identity_symmetric": symmetric_skill,
        "identity_accuracy_S1_to_S2": forward["accuracy"], "identity_accuracy_S2_to_S1": reverse["accuracy"],
        "identity_accuracy_symmetric": symmetric_accuracy, "chance_accuracy": chance,
        "chance_normalized_identity": (symmetric_accuracy - chance) / max(1.0 - chance, EPS),
        "subject_count": len(np.unique(subjects.astype(str))),
    }


def persistent_directions(features: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, count: int = 8) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    value = np.asarray(features, dtype=np.float64)
    center = value.mean(0)
    ordered = subject_sort(np.unique(subjects.astype(str)))
    means1 = np.stack([value[(subjects.astype(str) == s) & (sessions == 0)].mean(0) for s in ordered])
    means2 = np.stack([value[(subjects.astype(str) == s) & (sessions == 1)].mean(0) for s in ordered])
    subject_geometry = np.concatenate((means1 - center, means2 - center), axis=0)
    _, _, vt = np.linalg.svd(subject_geometry, full_matrices=False)
    pool = vt[: min(24, len(vt))].T
    candidates: list[dict[str, float]] = []
    for index, direction in enumerate(pool.T):
        p1, p2 = (means1 - center) @ direction, (means2 - center) @ direction
        persistence = 0.0 if min(np.std(p1), np.std(p2)) < 1e-12 else float(np.corrcoef(p1, p2)[0, 1])
        geometry = float(np.sqrt(np.mean(np.square((value - center) @ direction))))
        candidates.append({"pool_index": index, "persistence": persistence, "geometry_strength": geometry})
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-candidates[index]["persistence"], -candidates[index]["geometry_strength"], index),
    )[:count]
    return center.astype(np.float64), pool[:, order].astype(np.float64), [candidates[index] for index in order]


def erase_direction(features: np.ndarray, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    v /= max(float(np.linalg.norm(v)), EPS)
    return value - ((value - center) @ v)[:, None] * v[None, :]


def numpy_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value -= value.max(1, keepdims=True)
    log_probability = value - np.log(np.exp(value).sum(1, keepdims=True))
    return -log_probability[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)]


def exact_d_finite(clean_logits: np.ndarray, erased_logits: np.ndarray) -> float:
    delta = np.asarray(erased_logits, dtype=np.float64) - np.asarray(clean_logits, dtype=np.float64)
    centered = delta - delta.mean(-1, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=-1))))


def mean_subject_metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(logits).argmax(1)
    ba, f1 = [], []
    for subject in subject_sort(np.unique(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        ba.append(balanced_accuracy_score(labels[mask], prediction[mask]))
        f1.append(f1_score(labels[mask], prediction[mask], average="macro", zero_division=0))
    return {"BA": float(np.mean(ba)), "macro_f1": float(np.mean(f1)), "subject_count": len(ba)}


def task_subspace_overlap(weight: np.ndarray, direction: np.ndarray) -> float:
    centered = np.asarray(weight, dtype=np.float64) - np.asarray(weight, dtype=np.float64).mean(0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0] if len(singular) else 0.0, 1.0) * 1e-10))
    if rank == 0:
        return 0.0
    task_basis = vh[:rank].T
    v = np.asarray(direction, dtype=np.float64)
    v /= max(np.linalg.norm(v), EPS)
    return float(np.sum((task_basis.T @ v) ** 2))


def direction_rows(setting: str, fold: int, seed: int, model: nn.Module, model_fit: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray], checkpoint_sha: str, normalizer_sha: str, scope_hashes: Mapping[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    center, basis, meta = persistent_directions(model_fit["features"], model_fit["subjects"], model_fit["sessions"], 8)
    full_identity = identity_probe(model_fit["features"], model_fit["subjects"], model_fit["sessions"])
    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
    clean_fit_logits = np.asarray(model_fit["features"], dtype=np.float64) @ weight.T + bias
    clean_val_logits = np.asarray(validation["features"], dtype=np.float64) @ weight.T + bias
    clean_val_ce = numpy_cross_entropy(clean_val_logits, validation["labels"])
    clean_val_metrics = mean_subject_metrics(validation["labels"], clean_val_logits, validation["subjects"])
    evidence, controls = [], []
    basis_sha = array_sha256(basis.astype(np.float64))
    centered_fit = np.asarray(model_fit["features"], dtype=np.float64) - center
    for rank, (direction, info) in enumerate(zip(basis.T, meta), start=1):
        erased_fit = erase_direction(model_fit["features"], center, direction)
        erased_val = erase_direction(validation["features"], center, direction)
        erased_identity = identity_probe(erased_fit, model_fit["subjects"], model_fit["sessions"])
        erased_fit_logits = erased_fit @ weight.T + bias
        erased_val_logits = erased_val @ weight.T + bias
        erased_val_metrics = mean_subject_metrics(validation["labels"], erased_val_logits, validation["subjects"])
        direction_sha = array_sha256(np.asarray(direction, dtype=np.float64))
        evidence.append({
            "setting_id": setting, "dataset": SETTINGS[setting]["dataset"], "task": SETTINGS[setting]["task"],
            "backbone": SETTINGS[setting]["backbone"], "fold": fold, "seed": seed,
            "representation_dim": int(model.representation_dim),
            "identity_full": full_identity["identity_symmetric"],
            "identity_raw_accuracy": full_identity["identity_accuracy_symmetric"],
            "identity_chance_normalized_accuracy": full_identity["chance_normalized_identity"],
            "identity_cross_entropy": float(np.mean([full_identity["identity_CE_S1_to_S2"], full_identity["identity_CE_S2_to_S1"]])),
            "identity_direction_effect": full_identity["identity_symmetric"] - erased_identity["identity_symmetric"],
            "persistence": info["persistence"], "geometry_strength": info["geometry_strength"], "direction_rank": rank,
            "D_finite": exact_d_finite(clean_fit_logits, erased_fit_logits),
            "C_src_CE": float(np.mean(numpy_cross_entropy(erased_val_logits, validation["labels"]) - clean_val_ce)),
            "C_src_BA": clean_val_metrics["BA"] - erased_val_metrics["BA"],
            "C_src_F1": clean_val_metrics["macro_f1"] - erased_val_metrics["macro_f1"],
            "O_task": task_subspace_overlap(weight, direction), "direction_sha256": direction_sha,
            "checkpoint_sha256": checkpoint_sha, "normalizer_sha256": normalizer_sha,
            "persistence_basis_sha256": basis_sha, "source_scope_hash": scope_hashes["source"],
            "validation_scope_hash": scope_hashes["validation"],
        })
        target_displacement = np.abs(centered_fit @ direction)
        for control_id in range(100):
            control_seed = stable_seed("P4A-control", setting, fold, seed, rank, control_id)
            rng = np.random.default_rng(control_seed)
            random_direction = rng.normal(size=basis.shape[0])
            random_direction /= max(np.linalg.norm(random_direction), EPS)
            controls.append({
                "setting_id": setting, "fold": fold, "seed": seed, "direction_rank": rank,
                "control_id": control_id, "control_rank": 1, "control_sha256": array_sha256(random_direction.astype(np.float64)),
                "target_direction_sha256": direction_sha, "target_mean_displacement": float(target_displacement.mean()),
                "matched_mean_displacement": float(target_displacement.mean()),
                "control_seed": control_seed,
                "matching_rule": "P3_full_space_random_rank1_per_trial_displacement_norm",
                "regeneration_rule": "q=normalize(N(0,I)); d=(h-center)qqT; scale_i=abs((h_i-center)v)/norm(d_i)",
            })
    artifact = {"center": center, "basis": basis, "meta": meta, "basis_sha256": basis_sha, "full_identity": full_identity}
    return pd.DataFrame(evidence), pd.DataFrame(controls), artifact


def config_slug(method: str, lam: float) -> str:
    return f"{method.lower()}__lambda-{lam:.2f}"


def run_dir(setting: str, fold: int, seed: int) -> Path:
    return RUNS / setting / f"fold-{fold}" / f"seed-{seed}"
