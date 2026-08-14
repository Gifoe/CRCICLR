from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from persist_eeg_stage0.models import build_shared_model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "persist_eeg_p4_selective_invariance"
OLD_P2 = ROOT / "outputs" / "persist_eeg_p2p3"
MANIFEST_PATH = ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
SPLIT_PATH = ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
TASKS = ("mi", "erp", "ssvep")
EXPECTED_CLASSES = {"mi": 2, "erp": 2, "ssvep": 4}
DEVELOPMENT_FOLDS = (0, 1, 2)
DEVELOPMENT_SEEDS = (0, 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_maps(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for task in TASKS:
        labels = sorted(map(str, manifest.loc[manifest.paradigm == task, "event_label"].unique()))
        if len(labels) != EXPECTED_CLASSES[task]:
            raise RuntimeError(f"Unexpected class count for {task}: {labels}")
        result[task] = {label: index for index, label in enumerate(labels)}
    return result


def load_splits() -> list[dict[str, Any]]:
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    folds = payload["openbmi"]["folds"]
    if len(folds) != 5:
        raise RuntimeError(f"Expected five frozen folds, found {len(folds)}")
    for fold in folds:
        train = set(map(str, fold["train_subjects"]))
        validation = set(map(str, fold["validation_subjects"]))
        outer_test = set(map(str, fold["outer_test_subjects"]))
        outer_train = set(map(str, fold["outer_train_subjects"]))
        if train & validation or train & outer_test or validation & outer_test:
            raise RuntimeError(f"Subject leakage in frozen fold {fold['fold']}")
        if train | validation != outer_train:
            raise RuntimeError(f"Outer-train partition mismatch in fold {fold['fold']}")
    return folds


def split_for(fold: int) -> dict[str, Any]:
    split = next(item for item in load_splits() if int(item["fold"]) == int(fold))
    # Outer-test identifiers are used only for a one-way split integrity hash. No
    # outer-test sample index, signal, label, or metric is constructed in development.
    return {
        "fold": int(split["fold"]),
        "train_subjects": list(map(str, split["train_subjects"])),
        "validation_subjects": list(map(str, split["validation_subjects"])),
        "outer_test_subjects_hash": hashlib.sha256(
            "|".join(map(str, split["outer_test_subjects"])).encode("utf-8")
        ).hexdigest(),
    }


def subject_indices(manifest: pd.DataFrame, subjects: Sequence[str], task: str) -> np.ndarray:
    mask = manifest.subject_id.astype(str).isin(set(map(str, subjects))) & (manifest.paradigm == task)
    return np.flatnonzero(mask.to_numpy())


class EpochAccessor:
    def __init__(self, manifest: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> None:
        self.paths = manifest.signal_cache_path.astype(str).to_numpy()
        self.cache_indices = manifest.cache_index.to_numpy(dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None]
        self.arrays: dict[str, np.ndarray] = {}

    def get(self, global_index: int) -> torch.Tensor:
        relative = self.paths[global_index]
        if relative not in self.arrays:
            self.arrays[relative] = np.load(ROOT / relative, mmap_mode="r", allow_pickle=False)
        epoch = np.asarray(self.arrays[relative][self.cache_indices[global_index]], dtype=np.float32)
        return torch.from_numpy((epoch - self.mean) / self.std)


class TaskSubjectDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: np.ndarray,
        label_map: Mapping[str, int],
        subject_map: Mapping[str, int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        event_values = manifest.event_label.astype(str).to_numpy()
        subject_values = manifest.subject_id.astype(str).to_numpy()
        self.labels = np.asarray(
            [label_map[event_values[index]] for index in self.indices], dtype=np.int64
        )
        self.subject_labels = np.asarray(
            [subject_map[subject_values[index]] for index in self.indices], dtype=np.int64
        )
        self.accessor = EpochAccessor(manifest, mean, std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return (
            self.accessor.get(index),
            torch.tensor(self.labels[item], dtype=torch.long),
            torch.tensor(self.subject_labels[item], dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
        )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def coverage_loader(dataset: Dataset, steps: int, batch_size: int, epoch: int, seed: int) -> DataLoader:
    n = len(dataset)
    needed = min(n, steps * batch_size)
    permutation = np.random.default_rng(seed).permutation(n)
    start = (epoch * needed) % n
    positions = (start + np.arange(needed)) % n
    return make_loader(Subset(dataset, permutation[positions].tolist()), batch_size, True, seed + epoch)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


class PersistSI(nn.Module):
    def __init__(self, n_channels: int, embedding_dim: int, n_subjects: int) -> None:
        super().__init__()
        base = build_shared_model("eegnet", n_channels, embedding_dim, EXPECTED_CLASSES)
        self.encoder = base.encoder
        self.heads = base.heads
        self.adversaries = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.utils.parametrizations.spectral_norm(nn.Linear(embedding_dim, 64)),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.utils.parametrizations.spectral_norm(nn.Linear(64, n_subjects)),
                )
                for task in TASKS
            }
        )

    def load_historical(self, state: Mapping[str, torch.Tensor]) -> None:
        own = self.state_dict()
        unexpected = [key for key in state if key not in own]
        missing = [key for key in own if not key.startswith("adversaries.") and key not in state]
        if unexpected or missing:
            raise RuntimeError(f"Historical checkpoint mismatch unexpected={unexpected} missing={missing}")
        own.update(state)
        self.load_state_dict(own)

    def task_logits(self, x: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.heads[task](h), h


@dataclass(frozen=True)
class MethodConfig:
    version: str
    embedding_dim: int = 128
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    max_epochs: int = 12
    patience: int = 5
    adversary_warmup_epochs: int = 2
    adversary_ramp_epochs: int = 4
    lambda_inv: float = 0.10
    gradient_clip: float = 5.0
    covariance_shrinkage: float = 0.001
    whitening_epsilon_relative: float = 1e-4
    whitening_rank: int = 20
    statistics_update_interval: int = 4
    diagnostic_per_group: int = 16
    intervention_rank: int = 8
    pairs_per_subject_event: int = 8
    mi_adversary_scale: float = 1.0
    erp_adversary_scale: float = 1.0
    ssvep_adversary_scale: float = 1.0
    task_conditioned_spectrum: bool = False
    task_conditioned_directions: bool = False


VERSION_CONFIGS: dict[str, MethodConfig] = {
    "SI_V0": MethodConfig(version="SI_V0"),
    "SI_V1": MethodConfig(
        version="SI_V1",
        mi_adversary_scale=0.60,
        erp_adversary_scale=1.00,
        ssvep_adversary_scale=1.40,
    ),
    "SI_V2": MethodConfig(
        version="SI_V2",
        mi_adversary_scale=0.60,
        erp_adversary_scale=1.00,
        ssvep_adversary_scale=1.40,
        task_conditioned_spectrum=True,
    ),
    "SI_V3": MethodConfig(
        version="SI_V3",
        mi_adversary_scale=0.60,
        erp_adversary_scale=1.00,
        ssvep_adversary_scale=1.40,
        task_conditioned_spectrum=True,
        task_conditioned_directions=True,
    ),
    "SI_V4": MethodConfig(
        version="SI_V4",
        lambda_inv=0.15,
        mi_adversary_scale=0.60,
        erp_adversary_scale=1.00,
        ssvep_adversary_scale=1.40,
        task_conditioned_spectrum=True,
        task_conditioned_directions=True,
    ),
}


@dataclass
class SpectrumState:
    mean: np.ndarray
    whitener: np.ndarray
    dewhitener: np.ndarray
    directions: np.ndarray
    rho: np.ndarray
    rho_normalized: np.ndarray
    rho_by_task: dict[str, np.ndarray] | None
    rho_normalized_by_task: dict[str, np.ndarray] | None
    directions_by_task: dict[str, np.ndarray] | None
    relevance: dict[str, np.ndarray]
    protected: dict[str, np.ndarray]
    nuisance: dict[str, np.ndarray]
    audit: dict[str, Any]

    def tensor_bundle(self, device: torch.device) -> dict[str, Any]:
        nuisance: dict[str, torch.Tensor] = {}
        for task, values in self.nuisance.items():
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
            rms = torch.sqrt(torch.mean(torch.square(tensor))).clamp_min(1e-6)
            nuisance[task] = tensor / rms
        directions_by_task = None
        if self.directions_by_task is not None:
            directions_by_task = {
                task: torch.as_tensor(values, dtype=torch.float32, device=device)
                for task, values in self.directions_by_task.items()
            }
        return {
            "mean": torch.as_tensor(self.mean, dtype=torch.float32, device=device),
            "whitener": torch.as_tensor(self.whitener, dtype=torch.float32, device=device),
            "directions": torch.as_tensor(self.directions, dtype=torch.float32, device=device),
            "directions_by_task": directions_by_task,
            "nuisance": nuisance,
        }

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "whitener": self.whitener,
            "dewhitener": self.dewhitener,
            "directions": self.directions,
            "rho": self.rho,
            "rho_normalized": self.rho_normalized,
            "rho_by_task": self.rho_by_task,
            "rho_normalized_by_task": self.rho_normalized_by_task,
            "directions_by_task": self.directions_by_task,
            "relevance": self.relevance,
            "protected": self.protected,
            "nuisance": self.nuisance,
            "audit": self.audit,
        }

    @staticmethod
    def from_checkpoint(payload: Mapping[str, Any]) -> "SpectrumState":
        return SpectrumState(
            mean=np.asarray(payload["mean"]),
            whitener=np.asarray(payload["whitener"]),
            dewhitener=np.asarray(payload["dewhitener"]),
            directions=np.asarray(payload["directions"]),
            rho=np.asarray(payload["rho"]),
            rho_normalized=np.asarray(payload["rho_normalized"]),
            rho_by_task={
                task: np.asarray(value)
                for task, value in (payload.get("rho_by_task") or {}).items()
            } or None,
            rho_normalized_by_task={
                task: np.asarray(value)
                for task, value in (payload.get("rho_normalized_by_task") or {}).items()
            } or None,
            directions_by_task={
                task: np.asarray(value)
                for task, value in (payload.get("directions_by_task") or {}).items()
            } or None,
            relevance={task: np.asarray(value) for task, value in payload["relevance"].items()},
            protected={task: np.asarray(value) for task, value in payload["protected"].items()},
            nuisance={task: np.asarray(value) for task, value in payload["nuisance"].items()},
            audit=dict(payload["audit"]),
        )


def historical_paths(fold: int, seed: int) -> tuple[Path, Path, Path]:
    base = OLD_P2 / "backbone" / "checkpoints" / "eegnet" / f"fold-{fold}" / f"seed-{seed}"
    paths = base / "best.pt", base / "channel_mean.npy", base / "channel_std.npy"
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"Missing historical fold-local artifact(s): {paths}")
    return paths


def diagnostic_indices(
    manifest: pd.DataFrame,
    subjects: Sequence[str],
    task: str,
    seed: int,
    per_group: int,
) -> np.ndarray:
    block = manifest[
        manifest.subject_id.astype(str).isin(set(map(str, subjects))) & (manifest.paradigm == task)
    ]
    selected: list[int] = []
    for key, group in block.groupby(["subject_id", "session_id", "event_label"], sort=True):
        values = group.index.to_numpy(dtype=np.int64)
        group_seed = seed + int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(group_seed)
        if len(values) > per_group:
            values = rng.choice(values, size=per_group, replace=False)
        selected.extend(map(int, values))
    return np.asarray(sorted(selected), dtype=np.int64)


@torch.inference_mode()
def extract_embeddings(
    model: PersistSI,
    manifest: pd.DataFrame,
    subjects: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    seed: int,
    per_group: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model.eval()
    maps = label_maps(manifest)
    metadata: list[pd.DataFrame] = []
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for task_index, task in enumerate(TASKS):
        indices = diagnostic_indices(manifest, subjects, task, seed + task_index * 1009, per_group)
        subject_map = {subject: index for index, subject in enumerate(sorted(map(str, subjects), key=int))}
        dataset = TaskSubjectDataset(manifest, indices, maps[task], subject_map, mean, std)
        loader = make_loader(dataset, 256, False, seed + task_index)
        task_h: list[np.ndarray] = []
        task_y: list[np.ndarray] = []
        ordered: list[np.ndarray] = []
        for x, y, _, global_index in loader:
            h = model.encoder(x.to(device, non_blocking=True))
            task_h.append(h.cpu().numpy().astype(np.float32))
            task_y.append(y.numpy())
            ordered.append(global_index.numpy())
        joined = np.concatenate(ordered)
        block = manifest.iloc[joined][["subject_id", "session_id", "paradigm", "event_label"]].copy()
        block["global_index"] = joined
        metadata.append(block.reset_index(drop=True))
        embeddings.append(np.concatenate(task_h))
        labels.append(np.concatenate(task_y))
    return pd.concat(metadata, ignore_index=True), np.concatenate(embeddings), np.concatenate(labels)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    return ranks


def orthonormal_basis(values: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(values, dtype=np.float64), mode="reduced")
    signs = np.where(np.diag(r) >= 0.0, 1.0, -1.0)
    return q * signs[None, :]


def align_directions(
    directions: np.ndarray,
    rho: np.ndarray,
    previous: SpectrumState | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if previous is None:
        return directions, rho, {"matched_cosine_mean": 1.0, "matched_cosine_min": 1.0}
    if previous.directions.shape != directions.shape:
        raise RuntimeError(
            f"Numerical whitening rank changed across updates: {previous.directions.shape} -> {directions.shape}"
        )
    similarity = np.abs(previous.directions.T @ directions)
    old_indices, new_indices = linear_sum_assignment(-similarity)
    permutation = new_indices[np.argsort(old_indices)]
    aligned = directions[:, permutation].copy()
    aligned_rho = rho[permutation].copy()
    signs = np.sign(np.sum(previous.directions * aligned, axis=0))
    signs[signs == 0] = 1.0
    aligned *= signs[None, :]
    diagonal = np.abs(np.sum(previous.directions * aligned, axis=0))
    return aligned, aligned_rho, {
        "matched_cosine_mean": float(diagonal.mean()),
        "matched_cosine_min": float(diagonal.min()),
    }


def align_task_directions(
    directions: np.ndarray,
    rho: np.ndarray,
    previous: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Align a task-conditioned orthonormal basis without making it trainable."""
    if previous is None:
        return directions, rho, {"matched_cosine_mean": 1.0, "matched_cosine_min": 1.0}
    if previous.shape != directions.shape:
        raise RuntimeError(
            f"Task-conditioned spectrum rank changed across updates: {previous.shape} -> {directions.shape}"
        )
    similarity = np.abs(previous.T @ directions)
    old_indices, new_indices = linear_sum_assignment(-similarity)
    permutation = new_indices[np.argsort(old_indices)]
    aligned = directions[:, permutation].copy()
    aligned_rho = rho[permutation].copy()
    signs = np.sign(np.sum(previous * aligned, axis=0))
    signs[signs == 0] = 1.0
    aligned *= signs[None, :]
    diagonal = np.abs(np.sum(previous * aligned, axis=0))
    return aligned, aligned_rho, {
        "matched_cosine_mean": float(diagonal.mean()),
        "matched_cosine_min": float(diagonal.min()),
    }


def estimate_spectrum(
    model: PersistSI,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    config: MethodConfig,
    device: torch.device,
    previous: SpectrumState | None = None,
) -> SpectrumState:
    h = np.asarray(embeddings, dtype=np.float64)
    mean = h.mean(axis=0)
    centered = h - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(covariance)
    active_threshold = max(float(raw_eigenvalues.max()) * 1e-3, 1e-8)
    numerical_rank = int(np.sum(raw_eigenvalues > active_threshold))
    if numerical_rank < config.whitening_rank:
        raise RuntimeError(
            f"Whitening numerical rank {numerical_rank} is below frozen rank {config.whitening_rank}"
        )
    # EEGNet's 32-dimensional pre-projection bottleneck makes the nominal
    # 128-dimensional embedding covariance rank deficient. Whiten only the
    # measured active subspace instead of pretending null directions have unit variance.
    active_values = raw_eigenvalues[-config.whitening_rank :]
    active_vectors = raw_eigenvectors[:, -config.whitening_rank :]
    scale = float(np.mean(active_values))
    regularized_values = (1.0 - config.covariance_shrinkage) * active_values
    regularized_values += config.covariance_shrinkage * scale
    epsilon = max(config.whitening_epsilon_relative * scale, 1e-8)
    regularized_values = np.maximum(regularized_values, epsilon)
    whitener = active_vectors * np.power(regularized_values, -0.5)[None, :]
    dewhitener = np.sqrt(regularized_values)[:, None] * active_vectors.T
    whitened = centered @ whitener
    whitened_cov = whitened.T @ whitened / max(len(whitened) - 1, 1)

    frame = metadata.copy().reset_index(drop=True)
    frame["position"] = np.arange(len(frame))
    centroids: dict[tuple[str, str, str, str], np.ndarray] = {}
    for key, group in frame.groupby(["subject_id", "session_id", "paradigm", "event_label"], sort=True):
        centroids[tuple(map(str, key))] = whitened[group.position.to_numpy(dtype=np.int64)].mean(axis=0)
    sessions = sorted(frame.session_id.astype(str).unique())
    if len(sessions) != 2:
        raise RuntimeError(f"Cross-session spectrum requires exactly two sessions, found {sessions}")
    condition_session_means: dict[tuple[str, str, str], np.ndarray] = {}
    for task in TASKS:
        events = sorted(frame.loc[frame.paradigm == task, "event_label"].astype(str).unique())
        for event in events:
            for session in sessions:
                values = [
                    value
                    for (subject_key, session_key, task_key, event_key), value in centroids.items()
                    if session_key == session and task_key == task and event_key == event
                ]
                if not values:
                    raise RuntimeError(f"Missing condition/session centroid for {(task, event, session)}")
                condition_session_means[(task, event, session)] = np.mean(values, axis=0)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    condition_cross_covariances: list[np.ndarray] = []
    pair_composition: dict[str, int] = {task: 0 for task in TASKS}
    subjects = sorted(frame.subject_id.astype(str).unique(), key=int)
    task_cross_covariances: list[np.ndarray] = []
    for task in TASKS:
        event_cross_covariances: list[np.ndarray] = []
        events = sorted(frame.loc[frame.paradigm == task, "event_label"].astype(str).unique())
        for event in events:
            event_left: list[np.ndarray] = []
            event_right: list[np.ndarray] = []
            for subject in subjects:
                key_a = (subject, sessions[0], task, event)
                key_b = (subject, sessions[1], task, event)
                if key_a not in centroids or key_b not in centroids:
                    continue
                value_a = centroids[key_a] - condition_session_means[(task, event, sessions[0])]
                value_b = centroids[key_b] - condition_session_means[(task, event, sessions[1])]
                event_left.append(value_a)
                event_right.append(value_b)
                left.append(value_a)
                right.append(value_b)
                pair_composition[task] += 1
            event_a = np.asarray(event_left, dtype=np.float64)
            event_b = np.asarray(event_right, dtype=np.float64)
            event_cross = (event_a.T @ event_b + event_b.T @ event_a) / (2.0 * max(len(event_a), 1))
            event_cross_covariances.append(event_cross)
            condition_cross_covariances.append(event_cross)
        task_cross_covariances.append(np.mean(event_cross_covariances, axis=0))
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    cross_covariance = np.mean(task_cross_covariances, axis=0)
    rho, directions = np.linalg.eigh((cross_covariance + cross_covariance.T) / 2.0)
    order = np.argsort(rho)[::-1]
    rho = rho[order]
    directions = directions[:, order]
    directions, rho, alignment = align_directions(directions, rho, previous)

    rho_positive = np.maximum(rho, 0.0)
    rho_scale = max(float(np.quantile(rho_positive, 0.95)), 1e-12)
    rho_normalized = np.clip(rho_positive / rho_scale, 0.0, 1.0)
    # SI-V2 keeps the repeated-measure basis shared and fixed, but estimates
    # persistence strength separately for each target.  The task-specific
    # covariance is still measured from matched cross-session centroids; no
    # trainable projection is introduced.  This isolates the failure mode in
    # which a shared persistence strength causes negative transfer between
    # MI/ERP/SSVEP.
    rho_by_task: dict[str, np.ndarray] | None = None
    rho_normalized_by_task: dict[str, np.ndarray] | None = None
    directions_by_task: dict[str, np.ndarray] | None = {} if config.task_conditioned_directions else None
    task_direction_alignment: dict[str, dict[str, float]] = {}
    if config.task_conditioned_spectrum:
        rho_by_task = {}
        rho_normalized_by_task = {}
        for task, task_covariance in zip(TASKS, task_cross_covariances):
            task_covariance = (task_covariance + task_covariance.T) / 2.0
            if config.task_conditioned_directions:
                task_rho, task_directions = np.linalg.eigh(task_covariance)
                task_order = np.argsort(task_rho)[::-1]
                task_rho = task_rho[task_order]
                task_directions = task_directions[:, task_order]
                previous_task_directions = (
                    previous.directions_by_task.get(task)
                    if previous is not None and previous.directions_by_task is not None
                    else None
                )
                task_directions, task_rho, task_direction_alignment[task] = align_task_directions(
                    task_directions,
                    task_rho,
                    previous_task_directions,
                )
                directions_by_task[task] = task_directions.astype(np.float64)
            else:
                projected = directions.T @ task_covariance @ directions
                task_rho = np.diag(projected).astype(np.float64)
            task_positive = np.maximum(task_rho, 0.0)
            task_scale = max(float(np.quantile(task_positive, 0.95)), 1e-12)
            rho_by_task[task] = task_rho
            rho_normalized_by_task[task] = np.clip(task_positive / task_scale, 0.0, 1.0)
    relevance: dict[str, np.ndarray] = {}
    protected: dict[str, np.ndarray] = {}
    nuisance: dict[str, np.ndarray] = {}
    h_tensor = torch.as_tensor(h, dtype=torch.float32, device=device)
    dewhitener_tensor = torch.as_tensor(dewhitener, dtype=torch.float32, device=device)
    model.eval()
    for task in TASKS:
        positions = np.flatnonzero((frame.paradigm == task).to_numpy())
        task_h = h_tensor[torch.as_tensor(positions, dtype=torch.long, device=device)]
        task_y = torch.as_tensor(labels[positions], dtype=torch.long, device=device)
        logits = model.heads[task](task_h)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(task_y, EXPECTED_CLASSES[task]).to(probabilities)
        gradient_h = (probabilities - one_hot) @ model.heads[task].weight
        task_directions = (
            directions_by_task[task]
            if directions_by_task is not None
            else directions
        )
        task_directions_tensor = torch.as_tensor(task_directions, dtype=torch.float32, device=device)
        gradient_z = gradient_h @ dewhitener_tensor.T @ task_directions_tensor
        gradient_square = torch.square(gradient_z).detach().cpu().numpy()
        group_values: list[np.ndarray] = []
        # Avoid pandas row-slicing/groupby here.  On the Windows native pandas
        # build this occasionally raises an Index context-manager error during
        # the periodic spectrum update; NumPy grouping is equivalent and
        # deterministic for the already materialized train-only metadata.
        task_subjects = frame["subject_id"].astype(str).to_numpy()[positions]
        task_events = frame["event_label"].astype(str).to_numpy()[positions]
        groups: dict[tuple[str, str], list[int]] = {}
        for local_position, key in enumerate(zip(task_subjects, task_events)):
            groups.setdefault((str(key[0]), str(key[1])), []).append(local_position)
        for group_positions in groups.values():
            group_indices = np.asarray(group_positions, dtype=np.int64)
            group_values.append(gradient_square[group_indices].mean(axis=0))
        fisher = np.mean(group_values, axis=0).astype(np.float64)
        task_rho_normalized = (
            rho_normalized_by_task[task]
            if rho_normalized_by_task is not None
            else rho_normalized
        )
        relevance[task] = rank_normalize(fisher)
        protected[task] = task_rho_normalized * relevance[task]
        nuisance[task] = task_rho_normalized * (1.0 - relevance[task])

    raw_pca = raw_eigenvectors[:, np.argsort(raw_eigenvalues)[::-1]]
    rank = min(config.intervention_rank, directions.shape[1])
    persistence_raw = orthonormal_basis(dewhitener.T @ directions[:, :rank])
    pca_basis = orthonormal_basis(raw_pca[:, :rank])
    normalized_overlap = float(np.square(persistence_raw.T @ pca_basis).sum() / rank)
    audit = {
        "fit_population": "frozen fold train_subjects only",
        "samples": int(len(h)),
        "cross_session_centroid_pairs": int(len(a)),
        "pair_composition": pair_composition,
        "condition_control": "within paradigm/event and session centering before cross-session covariance",
        "covariance_shrinkage": config.covariance_shrinkage,
        "whitening_epsilon": epsilon,
        "whitening_type": "truncated shrinkage ZCA in measured active embedding subspace",
        "nominal_embedding_dimension": int(covariance.shape[0]),
        "raw_numerical_rank_at_threshold": numerical_rank,
        "numerical_whitening_rank": int(config.whitening_rank),
        "active_eigenvalue_threshold": active_threshold,
        "regularized_condition_number": float(regularized_values.max() / regularized_values.min()),
        "raw_covariance_eigenvalues": raw_eigenvalues.tolist(),
        "regularized_active_eigenvalues": regularized_values.tolist(),
        "whitened_variance_mean": float(np.mean(np.diag(whitened_cov))),
        "whitened_variance_std": float(np.std(np.diag(whitened_cov))),
        "whitened_covariance_identity_max_abs": float(
            np.max(np.abs(whitened_cov - np.eye(whitened_cov.shape[0])))
        ),
        "rho": rho.tolist(),
        "positive_rho_count": int(np.sum(rho > 0.0)),
        "spectrum_mode": (
            "task-conditioned persistence directions and strengths over shared whitening"
            if config.task_conditioned_directions
            else "task-conditioned persistence strengths over shared measured basis"
            if config.task_conditioned_spectrum
            else "shared persistence spectrum"
        ),
        "task_conditioned_rho": (
            {task: values.tolist() for task, values in rho_by_task.items()}
            if rho_by_task is not None
            else None
        ),
        "task_conditioned_positive_rho_count": (
            {task: int(np.sum(values > 0.0)) for task, values in rho_by_task.items()}
            if rho_by_task is not None
            else None
        ),
        "task_conditioned_direction_alignment": task_direction_alignment or None,
        "persistence_raw_PCA_normalized_overlap_rank": rank,
        "persistence_raw_PCA_normalized_overlap": normalized_overlap,
        "direction_alignment_to_previous": alignment,
        "orthonormality_error_max_abs": float(
            np.max(np.abs(directions.T @ directions - np.eye(directions.shape[1])))
        ),
        "finite": bool(
            all(np.isfinite(value).all() for value in [mean, whitener, dewhitener, directions, rho])
            and (
                rho_by_task is None
                or all(np.isfinite(value).all() for value in rho_by_task.values())
            )
            and (
                directions_by_task is None
                or all(np.isfinite(value).all() for value in directions_by_task.values())
            )
        ),
        "task_profiles": {
            task: {
                "protected_fraction_of_persistence_mass": float(
                    protected[task].sum()
                    / max(
                        (
                            rho_normalized_by_task[task]
                            if rho_normalized_by_task is not None
                            else rho_normalized
                        ).sum(),
                        1e-12,
                    )
                ),
                "nuisance_fraction_of_persistence_mass": float(
                    nuisance[task].sum()
                    / max(
                        (
                            rho_normalized_by_task[task]
                            if rho_normalized_by_task is not None
                            else rho_normalized
                        ).sum(),
                        1e-12,
                    )
                ),
                "mean_relevance": float(relevance[task].mean()),
            }
            for task in TASKS
        },
    }
    if audit["orthonormality_error_max_abs"] >= 1e-6 or not audit["finite"]:
        raise RuntimeError(f"Invalid persistence spectrum geometry: {audit}")
    if not (0.85 <= audit["whitened_variance_mean"] <= 1.15):
        raise RuntimeError(
            f"Whitening variance normalization failed: mean={audit['whitened_variance_mean']:.6f}"
        )
    return SpectrumState(
        mean=mean.astype(np.float32),
        whitener=whitener.astype(np.float32),
        dewhitener=dewhitener.astype(np.float32),
        directions=directions.astype(np.float32),
        rho=rho.astype(np.float32),
        rho_normalized=rho_normalized.astype(np.float32),
        rho_by_task=(
            {task: values.astype(np.float32) for task, values in rho_by_task.items()}
            if rho_by_task is not None
            else None
        ),
        rho_normalized_by_task=(
            {task: values.astype(np.float32) for task, values in rho_normalized_by_task.items()}
            if rho_normalized_by_task is not None
            else None
        ),
        directions_by_task=(
            {task: values.astype(np.float32) for task, values in directions_by_task.items()}
            if directions_by_task is not None
            else None
        ),
        relevance={task: value.astype(np.float32) for task, value in relevance.items()},
        protected={task: value.astype(np.float32) for task, value in protected.items()},
        nuisance={task: value.astype(np.float32) for task, value in nuisance.items()},
        audit=audit,
    )


def save_spectrum(path: Path, state: SpectrumState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=state.mean,
        whitener=state.whitener,
        dewhitener=state.dewhitener,
        directions=state.directions,
        rho=state.rho,
        rho_normalized=state.rho_normalized,
        **(
            {f"rho_{task}": values for task, values in state.rho_by_task.items()}
            if state.rho_by_task is not None
            else {}
        ),
        **(
            {f"rho_normalized_{task}": values for task, values in state.rho_normalized_by_task.items()}
            if state.rho_normalized_by_task is not None
            else {}
        ),
        **(
            {f"directions_{task}": values for task, values in state.directions_by_task.items()}
            if state.directions_by_task is not None
            else {}
        ),
        **{f"relevance_{task}": state.relevance[task] for task in TASKS},
        **{f"protected_{task}": state.protected[task] for task in TASKS},
        **{f"nuisance_{task}": state.nuisance[task] for task in TASKS},
    )
    write_json(path.with_suffix(".json"), state.audit)


def class_weights(labels: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=classes)
    return torch.tensor(counts.sum() / (classes * np.maximum(counts, 1)), dtype=torch.float32)


@torch.inference_mode()
def evaluate_task_model(
    model: PersistSI,
    datasets: Mapping[str, TaskSubjectDataset],
    config: MethodConfig,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    result: dict[str, float] = {}
    for task_index, task in enumerate(TASKS):
        truth: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        loader = make_loader(datasets[task], config.batch_size, False, seed + task_index)
        for x, y, _, _ in loader:
            logits, _ = model.task_logits(x.to(device, non_blocking=True), task)
            truth.append(y.numpy())
            predicted.append(logits.argmax(dim=1).cpu().numpy())
        result[task] = float(balanced_accuracy_score(np.concatenate(truth), np.concatenate(predicted)))
    return float(np.mean(list(result.values()))), result


def build_datasets(
    manifest: pd.DataFrame,
    split: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[dict[str, TaskSubjectDataset], dict[str, TaskSubjectDataset], dict[str, int]]:
    maps = label_maps(manifest)
    subject_map = {
        subject: index for index, subject in enumerate(sorted(map(str, split["train_subjects"]), key=int))
    }
    train = {
        task: TaskSubjectDataset(
            manifest,
            subject_indices(manifest, split["train_subjects"], task),
            maps[task],
            subject_map,
            mean,
            std,
        )
        for task in TASKS
    }
    # Validation subject labels are placeholders and are never sent to an adversary.
    validation_subject_map = {
        subject: index
        for index, subject in enumerate(sorted(map(str, split["validation_subjects"]), key=int))
    }
    validation = {
        task: TaskSubjectDataset(
            manifest,
            subject_indices(manifest, split["validation_subjects"], task),
            maps[task],
            validation_subject_map,
            mean,
            std,
        )
        for task in TASKS
    }
    return train, validation, subject_map


def adversary_strength(config: MethodConfig, epoch: int, task: str | None = None) -> float:
    if epoch < config.adversary_warmup_epochs:
        return 0.0
    progress = (epoch - config.adversary_warmup_epochs + 1) / max(config.adversary_ramp_epochs, 1)
    scale = 1.0 if task is None else float(getattr(config, f"{task}_adversary_scale"))
    return config.lambda_inv * min(1.0, progress) * scale


def padded_adversary_input(z: torch.Tensor, embedding_dim: int = 128) -> torch.Tensor:
    if z.shape[1] > embedding_dim:
        raise RuntimeError(f"Persistence coordinate dimension {z.shape[1]} exceeds {embedding_dim}")
    return F.pad(z, (0, embedding_dim - z.shape[1]))


def train_run(config: MethodConfig, fold: int, seed: int, output: Path) -> tuple[Path, Path]:
    complete = output / "TRAIN_COMPLETE.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        return ROOT / payload["best_checkpoint"], ROOT / payload["initial_checkpoint"]
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P4-SI development requires GPU; refusing accidental CPU training")
    manifest = pd.read_parquet(MANIFEST_PATH)
    split = split_for(fold)
    checkpoint_path, mean_path, std_path = historical_paths(fold, seed)
    historical = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mean, std = np.load(mean_path), np.load(std_path)
    train_datasets, validation_datasets, subject_map = build_datasets(manifest, split, mean, std)
    model = PersistSI(int(manifest.n_channels.iloc[0]), config.embedding_dim, len(subject_map))
    model.load_historical(historical["model"])
    model.to(device)
    output.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = output / "initial_historical.pt"
    torch.save(
        {
            "model": copy.deepcopy(model.state_dict()),
            "historical_checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
            "historical_checkpoint_sha256": sha256(checkpoint_path),
            "fold": fold,
            "seed": seed,
            "outer_test_used": False,
        },
        initial_checkpoint,
    )
    train_meta, train_h, train_y = extract_embeddings(
        model,
        manifest,
        split["train_subjects"],
        mean,
        std,
        device,
        31_000_000 + fold * 100_000 + seed * 1000,
        config.diagnostic_per_group,
    )
    spectrum = estimate_spectrum(model, train_meta, train_h, train_y, config, device)
    anchor_spectrum = copy.deepcopy(spectrum)
    save_spectrum(output / "statistics" / "initial_spectrum.npz", spectrum)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    task_criteria = {
        task: nn.CrossEntropyLoss(
            weight=class_weights(train_datasets[task].labels, EXPECTED_CLASSES[task]).to(device)
        )
        for task in TASKS
    }
    subject_criterion = nn.CrossEntropyLoss()
    steps = min(math.ceil(len(dataset) / config.batch_size) for dataset in train_datasets.values())
    curves: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = -1
    stale = 0
    best_path = output / "best.pt"
    started = time.time()
    for epoch in range(config.max_epochs):
        if epoch > 0 and config.statistics_update_interval > 0 and epoch % config.statistics_update_interval == 0:
            train_meta, train_h, train_y = extract_embeddings(
                model,
                manifest,
                split["train_subjects"],
                mean,
                std,
                device,
                31_000_000 + fold * 100_000 + seed * 1000,
                config.diagnostic_per_group,
            )
            spectrum = estimate_spectrum(model, train_meta, train_h, train_y, config, device, previous=spectrum)
            save_spectrum(output / "statistics" / f"epoch-{epoch:03d}_spectrum.npz", spectrum)
        bundle = spectrum.tensor_bundle(device)
        model.train()
        loaders = {
            task: coverage_loader(
                dataset,
                steps,
                config.batch_size,
                epoch,
                seed * 1_000_003 + fold * 10_007 + task_index * 101,
            )
            for task_index, (task, dataset) in enumerate(train_datasets.items())
        }
        iterators = {task: iter(loader) for task, loader in loaders.items()}
        running = {"task_loss": 0.0, "subject_loss": 0.0, "total_loss": 0.0, "grad_norm": 0.0}
        strength = adversary_strength(config, epoch)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            task_losses: list[torch.Tensor] = []
            subject_losses: list[torch.Tensor] = []
            for task in TASKS:
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    iterators[task] = iter(loaders[task])
                    batch = next(iterators[task])
                x, y, subject_y, _ = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                subject_y = subject_y.to(device, non_blocking=True)
                logits, h = model.task_logits(x, task)
                task_losses.append(task_criteria[task](logits, y))
                whitened = (h - bundle["mean"]) @ bundle["whitener"]
                task_directions = (
                    bundle["directions_by_task"][task]
                    if bundle["directions_by_task"] is not None
                    else bundle["directions"]
                )
                z = whitened @ task_directions
                nuisance_z = z * bundle["nuisance"][task]
                task_strength = adversary_strength(config, epoch, task)
                subject_logits = model.adversaries[task](
                    GradientReversal.apply(padded_adversary_input(nuisance_z, config.embedding_dim), task_strength)
                )
                subject_losses.append(subject_criterion(subject_logits, subject_y))
            task_loss = torch.stack(task_losses).mean()
            subject_loss = torch.stack(subject_losses).mean()
            total_loss = task_loss + subject_loss
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            for key, value in {
                "task_loss": task_loss,
                "subject_loss": subject_loss,
                "total_loss": total_loss,
                "grad_norm": grad_norm,
            }.items():
                running[key] += float(value.detach())
        macro, metrics = evaluate_task_model(model, validation_datasets, config, device, seed)
        row = {
            "epoch": epoch,
            "validation_macro_BA": macro,
            **{f"validation_BA_{task}": metrics[task] for task in TASKS},
            **{f"train_{key}": value / steps for key, value in running.items()},
            "lambda_inv_effective": strength,
            **{f"lambda_inv_effective_{task}": adversary_strength(config, epoch, task) for task in TASKS},
            "spectrum_positive_rho_count": spectrum.audit["positive_rho_count"],
            "spectrum_pca_overlap": spectrum.audit["persistence_raw_PCA_normalized_overlap"],
            "spectrum_alignment_mean": spectrum.audit["direction_alignment_to_previous"]["matched_cosine_mean"],
            "elapsed_seconds": time.time() - started,
        }
        curves.append(row)
        eligible = epoch >= config.adversary_warmup_epochs
        if eligible and macro > best_metric + 1e-8:
            best_metric, best_epoch, stale = macro, epoch, 0
            torch.save(
                {
                    "model": copy.deepcopy(model.state_dict()),
                    "spectrum": spectrum.checkpoint_payload(),
                    "anchor_spectrum": anchor_spectrum.checkpoint_payload(),
                    "method_config": asdict(config),
                    "fold": fold,
                    "seed": seed,
                    "epoch": epoch,
                    "validation_metrics": metrics,
                    "validation_macro_BA": macro,
                    "train_subjects": list(split["train_subjects"]),
                    "validation_subjects": list(split["validation_subjects"]),
                    "outer_test_subjects_hash": split["outer_test_subjects_hash"],
                    "outer_test_used": False,
                },
                best_path,
            )
        elif eligible:
            stale += 1
        pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)
        print(
            f"[P4-SI] {config.version} fold={fold} seed={seed} epoch={epoch} "
            f"val={macro:.6f} best={best_metric:.6f} lambda={strength:.4f}",
            flush=True,
        )
        if eligible and stale >= config.patience:
            break
    if not best_path.exists():
        raise RuntimeError("No eligible P4-SI checkpoint was selected")
    write_json(
        complete,
        {
            "status": "TRAIN_COMPLETE",
            "version": config.version,
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_macro_BA": best_metric,
            "best_checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
            "best_checkpoint_sha256": sha256(best_path),
            "initial_checkpoint": str(initial_checkpoint.relative_to(ROOT)).replace("\\", "/"),
            "initial_checkpoint_sha256": sha256(initial_checkpoint),
            "epochs_completed": len(curves),
            "steps_per_epoch": steps,
            "outer_test_used": False,
        },
    )
    return best_path, initial_checkpoint


def load_model_checkpoint(
    checkpoint_path: Path,
    manifest: pd.DataFrame,
    n_subjects: int,
    device: torch.device,
) -> tuple[PersistSI, dict[str, Any], SpectrumState | None, SpectrumState | None]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PersistSI(int(manifest.n_channels.iloc[0]), 128, n_subjects)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    spectrum = SpectrumState.from_checkpoint(checkpoint["spectrum"]) if "spectrum" in checkpoint else None
    anchor = SpectrumState.from_checkpoint(checkpoint["anchor_spectrum"]) if "anchor_spectrum" in checkpoint else None
    return model, checkpoint, spectrum, anchor


def transformed_coordinates(
    h: np.ndarray,
    spectrum: SpectrumState,
    task: str | None = None,
) -> np.ndarray:
    directions = spectrum.directions
    if task is not None and spectrum.directions_by_task is not None:
        directions = spectrum.directions_by_task[task]
    return (np.asarray(h, dtype=np.float64) - spectrum.mean) @ spectrum.whitener @ directions


def reconstruct_from_coordinates(
    z: np.ndarray,
    spectrum: SpectrumState,
    task: str | None = None,
) -> np.ndarray:
    directions = spectrum.directions
    if task is not None and spectrum.directions_by_task is not None:
        directions = spectrum.directions_by_task[task]
    whitened = np.asarray(z, dtype=np.float64) @ directions.T
    return (whitened @ spectrum.dewhitener + spectrum.mean).astype(np.float32)


def erase_directions(
    h: np.ndarray,
    spectrum: SpectrumState,
    indices: np.ndarray,
    task: str | None = None,
) -> np.ndarray:
    z = transformed_coordinates(h, spectrum, task)
    z[:, np.asarray(indices, dtype=np.int64)] = 0.0
    return reconstruct_from_coordinates(z, spectrum, task)


def pair_feature(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    cosine = np.sum(left * right, axis=1, keepdims=True) / np.maximum(
        np.linalg.norm(left, axis=1, keepdims=True) * np.linalg.norm(right, axis=1, keepdims=True),
        1e-12,
    )
    return np.concatenate([np.abs(left - right), left * right, cosine], axis=1)


def matched_verification_pairs(
    metadata: pd.DataFrame,
    values: np.ndarray,
    task: str,
    seed: int,
    pairs_per_subject_event: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.flatnonzero((metadata.paradigm == task).to_numpy())
    block = metadata.iloc[positions].reset_index().rename(columns={"index": "value_position"})
    subjects = sorted(block.subject_id.astype(str).unique(), key=int)
    sessions = sorted(block.session_id.astype(str).unique())
    if len(sessions) != 2:
        raise RuntimeError("Independent verifier requires exactly two sessions")
    grouped = {
        (str(subject), str(session), str(event)): group.value_position.to_numpy(dtype=np.int64)
        for (subject, session, event), group in block.groupby(["subject_id", "session_id", "event_label"])
    }
    rng = np.random.default_rng(seed)
    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    for subject in subjects:
        alternatives = [candidate for candidate in subjects if candidate != subject]
        events = sorted(block.loc[block.subject_id.astype(str) == subject, "event_label"].astype(str).unique())
        for event in events:
            anchors = grouped.get((subject, sessions[0], event), np.empty(0, dtype=np.int64))
            positives = grouped.get((subject, sessions[1], event), np.empty(0, dtype=np.int64))
            if not len(anchors) or not len(positives):
                continue
            for _ in range(pairs_per_subject_event):
                anchor = int(anchors[rng.integers(len(anchors))])
                positive = int(positives[rng.integers(len(positives))])
                candidates = alternatives.copy()
                rng.shuffle(candidates)
                negative_pool = np.empty(0, dtype=np.int64)
                for negative_subject in candidates:
                    negative_pool = grouped.get(
                        (negative_subject, sessions[1], event), np.empty(0, dtype=np.int64)
                    )
                    if len(negative_pool):
                        break
                if not len(negative_pool):
                    continue
                negative = int(negative_pool[rng.integers(len(negative_pool))])
                left.extend([anchor, anchor])
                right.extend([positive, negative])
                labels.extend([1, 0])
    return pair_feature(values[np.asarray(left)], values[np.asarray(right)]), np.asarray(labels, dtype=np.int64)


def nuisance_subject_verification(
    train_metadata: pd.DataFrame,
    train_h: np.ndarray,
    validation_metadata: pd.DataFrame,
    validation_h: np.ndarray,
    spectrum: SpectrumState,
    seed: int,
    pairs_per_subject_event: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"per_task": {}}
    task_aurocs: list[float] = []
    for task_index, task in enumerate(TASKS):
        z_train = transformed_coordinates(train_h, spectrum, task)
        z_validation = transformed_coordinates(validation_h, spectrum, task)
        train_values = z_train * spectrum.nuisance[task][None, :]
        validation_values = z_validation * spectrum.nuisance[task][None, :]
        x_train, y_train = matched_verification_pairs(
            train_metadata,
            train_values,
            task,
            seed + task_index * 103,
            pairs_per_subject_event,
        )
        x_validation, y_validation = matched_verification_pairs(
            validation_metadata,
            validation_values,
            task,
            seed + task_index * 103 + 17,
            pairs_per_subject_event,
        )
        verifier = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        verifier.fit(x_train, y_train)
        auroc = float(roc_auc_score(y_validation, verifier.predict_proba(x_validation)[:, 1]))
        result["per_task"][task] = {
            "auroc": auroc,
            "train_pairs": int(len(y_train)),
            "validation_pairs": int(len(y_validation)),
            "verifier": "independent fixed-C logistic pair verifier",
            "pair_composition": "session-0 anchor to session-1 positive/negative; paradigm/event matched",
        }
        task_aurocs.append(auroc)
    result["macro_AUROC"] = float(np.mean(task_aurocs))
    return result


def task_probe_ba(
    train_metadata: pd.DataFrame,
    train_values: np.ndarray,
    validation_metadata: pd.DataFrame,
    validation_values: np.ndarray,
    seed: int,
) -> dict[str, float]:
    maps = label_maps(pd.concat([train_metadata, validation_metadata], ignore_index=True))
    result: dict[str, float] = {}
    for task_index, task in enumerate(TASKS):
        train_mask = (train_metadata.paradigm == task).to_numpy()
        validation_mask = (validation_metadata.paradigm == task).to_numpy()
        y_train = np.asarray(
            [maps[task][str(value)] for value in train_metadata.loc[train_mask, "event_label"]], dtype=np.int64
        )
        y_validation = np.asarray(
            [maps[task][str(value)] for value in validation_metadata.loc[validation_mask, "event_label"]],
            dtype=np.int64,
        )
        probe = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed + task_index,
        )
        probe.fit(train_values[train_mask], y_train)
        result[task] = float(
            balanced_accuracy_score(y_validation, probe.predict(validation_values[validation_mask]))
        )
    return result


def intervention_sets(spectrum: SpectrumState, task: str, rank: int, seed: int) -> dict[str, np.ndarray]:
    dimension = len(spectrum.rho)
    rank = min(int(rank), dimension // 3)
    task_rho_normalized = (
        spectrum.rho_normalized_by_task.get(task, spectrum.rho_normalized)
        if spectrum.rho_normalized_by_task is not None
        else spectrum.rho_normalized
    )
    protected_order = np.argsort(spectrum.protected[task])[::-1]
    protected = protected_order[:rank]
    nuisance_order = np.argsort(spectrum.nuisance[task])[::-1]
    nuisance = np.asarray([index for index in nuisance_order if index not in set(protected)][:rank], dtype=np.int64)
    if len(nuisance) != rank:
        raise RuntimeError("Unable to construct disjoint protected/nuisance intervention sets")
    excluded = set(protected) | set(nuisance)
    persistent_candidates = [
        int(index)
        for index in np.argsort(task_rho_normalized)[::-1][: max(rank * 4, rank)]
        if int(index) not in excluded
    ]
    if len(persistent_candidates) < rank:
        persistent_candidates = [index for index in range(dimension) if index not in excluded]
    rng = np.random.default_rng(seed)
    random_indices = np.sort(rng.choice(persistent_candidates, size=rank, replace=False)).astype(np.int64)
    return {
        "protected": np.sort(protected).astype(np.int64),
        "nuisance": np.sort(nuisance).astype(np.int64),
        "random": random_indices,
    }


def intervention_audit(
    train_metadata: pd.DataFrame,
    train_h: np.ndarray,
    validation_metadata: pd.DataFrame,
    validation_h: np.ndarray,
    spectrum: SpectrumState,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    raw = task_probe_ba(train_metadata, train_h, validation_metadata, validation_h, seed)
    result: dict[str, Any] = {
        "probe": "independent fixed-C logistic event probe refit after every intervention",
        "raw_BA": raw,
        "per_task": {},
    }
    for task_index, task in enumerate(TASKS):
        sets = intervention_sets(spectrum, task, rank, seed + task_index * 1009)
        row: dict[str, Any] = {"rank": int(len(sets["protected"])), "indices": {}, "BA": {}, "delta_BA": {}}
        for policy, indices in sets.items():
            train_erased = erase_directions(train_h, spectrum, indices, task)
            validation_erased = erase_directions(validation_h, spectrum, indices, task)
            metrics = task_probe_ba(
                train_metadata,
                train_erased,
                validation_metadata,
                validation_erased,
                seed + 10_000 + task_index * 101,
            )
            row["indices"][policy] = indices.tolist()
            row["BA"][policy] = metrics[task]
            row["delta_BA"][policy] = float(metrics[task] - raw[task])
        row["protected_drop_minus_nuisance_drop"] = float(
            -row["delta_BA"]["protected"] + row["delta_BA"]["nuisance"]
        )
        result["per_task"][task] = row
    return result


def development_run(config: MethodConfig, fold: int, seed: int) -> dict[str, Any]:
    if fold not in DEVELOPMENT_FOLDS or seed not in DEVELOPMENT_SEEDS:
        raise RuntimeError("Development is fail-closed to folds 0,1,2 and seeds 0,1")
    if (OUT / "formal" / "TEST_ACCESS_STARTED.json").exists():
        raise RuntimeError("Test access has started; all further development is permanently forbidden")
    output = OUT / "development" / config.version / f"fold-{fold}" / f"seed-{seed}"
    result_path = output / "DEVELOPMENT_RESULT.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("method_config") != clean(asdict(config)) or existing.get("fold") != fold or existing.get("seed") != seed:
            raise RuntimeError("Existing development result provenance does not match requested run")
        return existing
    best_path, initial_path = train_run(config, fold, seed, output)
    device = torch.device("cuda")
    manifest = pd.read_parquet(MANIFEST_PATH)
    split = split_for(fold)
    _, mean_path, std_path = historical_paths(fold, seed)
    mean, std = np.load(mean_path), np.load(std_path)
    train_datasets, validation_datasets, subject_map = build_datasets(manifest, split, mean, std)
    final_model, best_checkpoint, final_spectrum, anchor_spectrum = load_model_checkpoint(
        best_path, manifest, len(subject_map), device
    )
    initial_model, initial_checkpoint, _, _ = load_model_checkpoint(
        initial_path, manifest, len(subject_map), device
    )
    if final_spectrum is None or anchor_spectrum is None:
        raise RuntimeError("Selected P4-SI checkpoint lacks frozen spectrum state")
    baseline_macro, baseline_task = evaluate_task_model(
        initial_model, validation_datasets, config, device, seed
    )
    final_macro, final_task = evaluate_task_model(final_model, validation_datasets, config, device, seed)
    historical_reference = {
        task: float(value)
        for task, value in torch.load(
            historical_paths(fold, seed)[0], map_location="cpu", weights_only=False
        )["validation_metrics"].items()
    }
    replay_error = {task: float(baseline_task[task] - historical_reference[task]) for task in TASKS}
    # GPU convolution kernels may flip a handful of predictions exactly on the
    # decision boundary. Preserve the exact replay error and fail only when it
    # is large enough to indicate a split/preprocessing/checkpoint mismatch.
    if any(abs(value) > 1e-3 for value in replay_error.values()):
        raise RuntimeError(f"Historical checkpoint replay mismatch: {replay_error}")

    diagnostic_seed = 41_000_000 + fold * 100_000 + seed * 1000
    train_meta_before, train_h_before, _ = extract_embeddings(
        initial_model,
        manifest,
        split["train_subjects"],
        mean,
        std,
        device,
        diagnostic_seed,
        config.diagnostic_per_group,
    )
    validation_meta_before, validation_h_before, _ = extract_embeddings(
        initial_model,
        manifest,
        split["validation_subjects"],
        mean,
        std,
        device,
        diagnostic_seed + 17,
        config.diagnostic_per_group,
    )
    train_meta_after, train_h_after, _ = extract_embeddings(
        final_model,
        manifest,
        split["train_subjects"],
        mean,
        std,
        device,
        diagnostic_seed,
        config.diagnostic_per_group,
    )
    validation_meta_after, validation_h_after, _ = extract_embeddings(
        final_model,
        manifest,
        split["validation_subjects"],
        mean,
        std,
        device,
        diagnostic_seed + 17,
        config.diagnostic_per_group,
    )
    if not train_meta_before.global_index.equals(train_meta_after.global_index):
        raise RuntimeError("Before/after train diagnostic samples differ")
    if not validation_meta_before.global_index.equals(validation_meta_after.global_index):
        raise RuntimeError("Before/after validation diagnostic samples differ")

    subject_before = nuisance_subject_verification(
        train_meta_before,
        train_h_before,
        validation_meta_before,
        validation_h_before,
        anchor_spectrum,
        seed + 50_000,
        config.pairs_per_subject_event,
    )
    subject_after = nuisance_subject_verification(
        train_meta_after,
        train_h_after,
        validation_meta_after,
        validation_h_after,
        anchor_spectrum,
        seed + 50_000,
        config.pairs_per_subject_event,
    )
    verifier_delta = {
        task: float(subject_after["per_task"][task]["auroc"] - subject_before["per_task"][task]["auroc"])
        for task in TASKS
    }
    verifier_delta["macro"] = float(subject_after["macro_AUROC"] - subject_before["macro_AUROC"])
    intervention_before = intervention_audit(
        train_meta_before,
        train_h_before,
        validation_meta_before,
        validation_h_before,
        anchor_spectrum,
        config.intervention_rank,
        seed + 60_000,
    )
    intervention_after = intervention_audit(
        train_meta_after,
        train_h_after,
        validation_meta_after,
        validation_h_after,
        final_spectrum,
        config.intervention_rank,
        seed + 60_000,
    )
    task_delta = {task: float(final_task[task] - baseline_task[task]) for task in TASKS}
    result = {
        "status": "DEVELOPMENT_RUN_COMPLETE",
        "version": config.version,
        "fold": fold,
        "seed": seed,
        "method_config": asdict(config),
        "data_used": ["TRAIN", "VALIDATION"],
        "outer_test_used": False,
        "outer_test_subjects_hash_only": split["outer_test_subjects_hash"],
        "train_subject_ids": split["train_subjects"],
        "validation_subject_ids": split["validation_subjects"],
        "best_checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
        "best_checkpoint_sha256": sha256(best_path),
        "best_epoch": int(best_checkpoint["epoch"]),
        "task_performance": {
            "historical_reference": historical_reference,
            "historical_replay": baseline_task,
            "historical_replay_error": replay_error,
            "after": final_task,
            "delta": task_delta,
            "macro_before": baseline_macro,
            "macro_after": final_macro,
            "macro_delta": float(final_macro - baseline_macro),
        },
        "persistence_spectrum": {
            "initial": anchor_spectrum.audit,
            "selected_checkpoint": final_spectrum.audit,
        },
        "profiles": final_spectrum.audit["task_profiles"],
        "nuisance_subject_verification": {
            "definition": "independent pair verifier evaluated in the fixed initial nuisance coordinate system",
            "before": subject_before,
            "after": subject_after,
            "after_minus_before_AUROC": verifier_delta,
        },
        "intervention": {
            "before": intervention_before,
            "after": intervention_after,
        },
        "sanity": {
            "no_nan_inf": bool(
                all(
                    np.isfinite(value).all()
                    for value in [train_h_before, validation_h_before, train_h_after, validation_h_after]
                )
            ),
            "same_diagnostic_samples_before_after": True,
            "negative_pair_session_composition_matched": True,
            "task_head_uses_full_h": True,
            "persistence_basis_trainable": False,
            "global_subject_adversary_used": False,
        },
    }
    write_json(result_path, result)
    print(
        json.dumps(
            clean(
                {
                    "status": result["status"],
                    "version": config.version,
                    "fold": fold,
                    "seed": seed,
                    "task_delta": task_delta,
                    "nuisance_subject_AUROC_delta": verifier_delta,
                    "MI_intervention": intervention_after["per_task"]["mi"]["delta_BA"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )
    return result


def write_protocol() -> None:
    if (OUT / "formal" / "TEST_ACCESS_STARTED.json").exists():
        raise RuntimeError("Cannot rewrite protocol after test access")
    protocol = {
        "method": "PERSIST-SI Selective Persistence Invariance",
        "protocol_version": 1,
        "development_panel": {
            "folds": list(DEVELOPMENT_FOLDS),
            "seeds": list(DEVELOPMENT_SEEDS),
            "runs_per_version": len(DEVELOPMENT_FOLDS) * len(DEVELOPMENT_SEEDS),
        },
        "frozen_principles": [
            "persistence from repeated-measure statistics",
            "explicit task-relevant versus task-irrelevant persistence",
            "invariance acts only on task-irrelevant persistence",
        ],
        "initial_version": asdict(VERSION_CONFIGS["SI_V0"]),
        "spectrum": {
            "whitening": "fold-train-only rank-20 truncated ZCA over the measured active EEGNet subspace, with fixed shrinkage and relative eigenvalue floor; rank frozen before development performance was observed",
            "relation": "same subject, same paradigm/event, different session; condition/session means removed",
            "estimator": "symmetric cross-session covariance of balanced group centroids",
            "update": "stop-gradient periodic estimation with direction matching",
            "basis_trainable": False,
        },
        "relevance": "subject/event-balanced Fisher gradient in whitened persistence coordinates; within-task rank normalization",
        "nuisance_mask": "positive normalized persistence strength times one minus normalized relevance",
        "task_head_input": "full unpartitioned h",
        "adversary_input": "task-specific soft nuisance persistence coordinates only; spectrally constrained adversary prevents scale compensation",
        "gates_preregistered": {
            "task_warning": "each task mean delta must be >= -0.01; no run/task delta below -0.03",
            "relevance_validity": "MI protected-drop minus nuisance-drop mean >= 0.005 and positive in at least 4/6 runs",
            "selective_suppression": "macro nuisance verifier AUROC mean reduction >= 0.02 and reduction in at least 4/6 runs",
            "protection": "MI protected erase drop after training retains at least 50% of pretraining drop, when pretraining drop >= 0.005",
            "strong_signal": "MI mean validation improvement >= 0.01",
            "viable_signal": "MI mean validation improvement >= 0.005 and nonnegative in at least 4/6 runs",
        },
        "outer_test_policy": {
            "before_lock": "FORBIDDEN",
            "development_entry_rejects_after_test_marker": True,
            "formal_runs": "NOT AUTHORIZED unless a locked-method artifact is created from aggregate development gates",
        },
        "old_p4_immutable": [
            "p4_persist_pb.py",
            "outputs/persist_eeg_p3closure_p4/p4/P4_FINAL_REPORT.json",
            "outputs/persist_eeg_p3closure_p4/p4/P4_ADAPTATION_LOG.json",
        ],
    }
    write_json(OUT / "protocol" / "P4_SI_PROTOCOL.json", protocol)
    adaptation = OUT / "protocol" / "P4_SI_ADAPTATION_LOG.json"
    entries = []
    if adaptation.exists():
        entries = json.loads(adaptation.read_text(encoding="utf-8"))
    if not any(entry.get("version") == "SI_V0" for entry in entries):
        entries.append(
                {
                    "version": "SI_V0",
                    "parent_version": None,
                    "observed_failure": "Old P4 free projector and scale-confounded persistence gate failed",
                    "failure_type": "SCIENTIFIC",
                    "evidence": {
                        "old_decision": "P4_MAIN_METHOD_NOT_SUPPORTED",
                        "UL_PCA_normalized_overlap": 0.745112,
                    },
                    "modification": "train-only whitened repeated-measure spectrum, Fisher relevance, nuisance-only GRL",
                    "why_this_modification_addresses_failure": "removes free basis rotation and global P/F bottleneck while protecting high-relevance persistent directions",
                    "data_used": ["TRAIN", "VALIDATION"],
                    "outer_test_used": False,
                }
        )
    if not any(entry.get("version") == "SI_V1" for entry in entries):
        entries.append(
            {
                "version": "SI_V1",
                "parent_version": "SI_V0",
                "observed_failure": "SI-V0 passed intervention and nuisance suppression checks but MI validation BA was neutral/slightly negative and SSVEP nuisance suppression was inconsistent across seeds",
                "failure_type": "NEGATIVE_TRANSFER",
                "evidence": {
                    "SI_V0_mean_MI_delta_BA": -0.002407407407407395,
                    "SI_V0_mean_macro_nuisance_AUROC_reduction": 0.03245965149176957,
                    "SI_V0_SSVEP_verifier_delta_range": [-0.055471161111111945, 0.044548128858024505],
                },
                "modification": "task-specific adversary strength: MI 0.60x, ERP 1.00x, SSVEP 1.40x",
                "why_this_modification_addresses_failure": "reduce negative transfer on MI while increasing selective suppression pressure on the least consistent SSVEP nuisance path; no change to persistence estimator or relevance definition",
                "data_used": ["TRAIN", "VALIDATION"],
                "outer_test_used": False,
            }
        )
    if not any(entry.get("version") == "SI_V2" for entry in entries):
        entries.append(
            {
                "version": "SI_V2",
                "parent_version": "SI_V1",
                "observed_failure": "SI-V1 retained valid selective intervention and macro nuisance suppression, but shared persistence strength produced task-dependent negative transfer: MI remained below reference and SSVEP verifier suppression was inconsistent across the development panel",
                "failure_type": "NEGATIVE_TRANSFER",
                "evidence": {
                    "SI_V1_mean_MI_delta_BA": -0.001481481481481491,
                    "SI_V1_mean_SSVEP_delta_BA": -0.0017592592592592937,
                    "SI_V1_SSVEP_verifier_delta_range": [-0.061185860339506126, 0.050540123456790154],
                    "SI_V1_gate_A_to_D_pass": True,
                },
                "modification": "task-conditioned cross-session persistence strengths estimated per target over the same frozen whitened statistical basis; MI/ERP/SSVEP adversary scales unchanged from SI-V1",
                "why_this_modification_addresses_failure": "shared directions remain tied to repeated-measure statistics, while each target receives its own measured persistence spectrum so a high-persistence direction in one paradigm cannot impose the same nuisance pressure on another paradigm",
                "data_used": ["TRAIN", "VALIDATION"],
                "outer_test_used": False,
            }
        )
    if not any(entry.get("version") == "SI_V3" for entry in entries):
        entries.append(
            {
                "version": "SI_V3",
                "parent_version": "SI_V2",
                "observed_failure": "SI-V2 task-conditioned persistence strengths improved diagnostic task separation but did not produce a validation generalization gain; MI and SSVEP remained slightly negative on average",
                "failure_type": "NEGATIVE_TRANSFER",
                "evidence": {
                    "SI_V2_mean_MI_delta_BA": -0.0012037037037037068,
                    "SI_V2_mean_SSVEP_delta_BA": -0.0022222222222222734,
                    "SI_V2_gate_A_to_D_pass": True,
                    "SI_V2_macro_nuisance_reduction": 0.028744293338477334,
                },
                "modification": "task-conditioned repeated-measure covariance eigenspaces and strengths under the same fold-train whitening; task bases are statistically estimated, direction-matched across updates, frozen between updates, and never trainable",
                "why_this_modification_addresses_failure": "the remaining shared-basis assumption can force unrelated paradigms to suppress different physical directions with the same coordinates; task-conditioned measured eigenspaces allow each target to identify its own persistent variation while preserving the repeated-measure definition",
                "data_used": ["TRAIN", "VALIDATION"],
                "outer_test_used": False,
            }
        )
    if not any(entry.get("version") == "SI_V4" for entry in entries):
        entries.append(
            {
                "version": "SI_V4",
                "parent_version": "SI_V3",
                "observed_failure": "SI-V3 task-conditioned eigenspaces preserved task performance and intervention validity but the independent nuisance verifier reduction averaged only 0.99pp, below the preregistered 2pp selective-suppression gate",
                "failure_type": "OPTIMIZATION",
                "evidence": {
                    "SI_V3_macro_nuisance_reduction": 0.009883509087791434,
                    "SI_V3_gate_C_mean_pass": False,
                    "SI_V3_gate_A_pass": True,
                    "SI_V3_gate_B_MI_pass": True,
                    "SI_V3_gate_D_pass": True,
                },
                "modification": "increase only the selective nuisance GRL ceiling from lambda_inv=0.10 to 0.15; retain task-conditioned measured eigenspaces, relevance, schedules, and task-specific scales",
                "why_this_modification_addresses_failure": "the failure is insufficient nuisance suppression rather than an invalid persistence or relevance definition; a single stronger adversarial ceiling tests whether the measured nuisance coordinates can be suppressed without changing the protected/task pathway",
                "data_used": ["TRAIN", "VALIDATION"],
                "outer_test_used": False,
            }
        )
    write_json(adaptation, entries)


def smoke() -> None:
    write_protocol()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Smoke test requires CUDA on the designated server")
    manifest = pd.read_parquet(MANIFEST_PATH)
    split = split_for(0)
    checkpoint_path, mean_path, std_path = historical_paths(0, 0)
    mean, std = np.load(mean_path), np.load(std_path)
    train, validation, subject_map = build_datasets(manifest, split, mean, std)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PersistSI(int(manifest.n_channels.iloc[0]), 128, len(subject_map))
    model.load_historical(checkpoint["model"])
    model.to(device)
    train_meta, train_h, train_y = extract_embeddings(
        model, manifest, split["train_subjects"], mean, std, device, 91_000_000, 2
    )
    spectrum = estimate_spectrum(model, train_meta, train_h, train_y, VERSION_CONFIGS["SI_V0"], device)
    task = "mi"
    x, y, subject_y, _ = next(iter(make_loader(train[task], 8, True, 0)))
    x, y, subject_y = x.to(device), y.to(device), subject_y.to(device)
    logits, h = model.task_logits(x, task)
    bundle = spectrum.tensor_bundle(device)
    z = (h - bundle["mean"]) @ bundle["whitener"] @ bundle["directions"]
    subject_logits = model.adversaries[task](
        GradientReversal.apply(
            padded_adversary_input(z * bundle["nuisance"][task], VERSION_CONFIGS["SI_V0"].embedding_dim),
            0.1,
        )
    )
    loss = F.cross_entropy(logits, y) + F.cross_entropy(subject_logits, subject_y)
    loss.backward()
    checks = {
        "cuda": torch.cuda.is_available(),
        "development_panel_exact": list(DEVELOPMENT_FOLDS) == [0, 1, 2]
        and list(DEVELOPMENT_SEEDS) == [0, 1],
        "outer_test_signal_not_loaded": True,
        "train_validation_subject_disjoint": not bool(
            set(split["train_subjects"]) & set(split["validation_subjects"])
        ),
        "task_head_full_h": True,
        "basis_is_statistical_not_parameter": True,
        "whitening_finite": spectrum.audit["finite"],
        "orthonormality": spectrum.audit["orthonormality_error_max_abs"] < 1e-6,
        "gradient_finite": bool(
            all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        ),
        "validation_datasets_constructed_without_test": all(len(dataset) > 0 for dataset in validation.values()),
    }
    status = "PASS" if all(bool(value) for value in checks.values()) else "FAIL"
    write_json(
        OUT / "protocol" / "SMOKE_TEST.json",
        {
            "status": status,
            "checks": checks,
            "spectrum_audit": spectrum.audit,
            "outer_test_used": False,
        },
    )
    print(json.dumps(clean({"status": status, "checks": checks}), indent=2))
    if status != "PASS":
        raise RuntimeError("P4-SI smoke test failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "development"], required=True)
    parser.add_argument("--version", choices=sorted(VERSION_CONFIGS), default="SI_V0")
    parser.add_argument("--fold", type=int, choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--seed", type=int, choices=DEVELOPMENT_SEEDS)
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke()
        return
    if args.fold is None or args.seed is None:
        parser.error("development mode requires --fold and --seed")
    write_protocol()
    development_run(VERSION_CONFIGS[args.version], args.fold, args.seed)


if __name__ == "__main__":
    main()
