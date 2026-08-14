from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from persist_eeg_stage0.models import EEGNetEncoder


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "outputs" / "persist_eeg_p2p3"
OUT = ROOT / "outputs" / "persist_eeg_p3closure_p4" / "p4"
MANIFEST_PATH = ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
SPLIT_PATH = ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
TASKS = ("mi", "erp", "ssvep")
EXPECTED_CLASSES = {"mi": 2, "erp": 2, "ssvep": 4}


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
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
    mappings: dict[str, dict[str, int]] = {}
    for task in TASKS:
        labels = sorted(map(str, manifest.loc[manifest.paradigm == task, "event_label"].unique()))
        mappings[task] = {label: index for index, label in enumerate(labels)}
        if len(labels) != EXPECTED_CLASSES[task]:
            raise RuntimeError(f"Unexpected {task} label count: {len(labels)}")
    return mappings


def load_splits() -> list[dict[str, Any]]:
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    folds = payload["openbmi"]["folds"]
    if len(folds) != 5:
        raise RuntimeError(f"Expected five frozen folds, found {len(folds)}")
    for fold in folds:
        train = set(map(str, fold["train_subjects"]))
        validation = set(map(str, fold["validation_subjects"]))
        test = set(map(str, fold.get("test_subjects", fold.get("outer_test_subjects", []))))
        outer_train = set(map(str, fold["outer_train_subjects"]))
        if train & validation or train & test or validation & test:
            raise RuntimeError(f"Subject leakage in fold {fold['fold']}")
        if train | validation != outer_train:
            raise RuntimeError(f"Outer-train partition mismatch in fold {fold['fold']}")
    return folds


def split_for(fold: int) -> dict[str, Any]:
    return next(value for value in load_splits() if int(value["fold"]) == fold)


def subject_indices(manifest: pd.DataFrame, subjects: Sequence[str], task: str) -> np.ndarray:
    mask = manifest.subject_id.astype(str).isin(set(map(str, subjects))) & (manifest.paradigm == task)
    return np.flatnonzero(mask.to_numpy())


class EpochAccessor:
    def __init__(self, repository: Path, manifest: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        self.repository = repository
        self.paths = manifest.signal_cache_path.astype(str).to_numpy()
        self.cache_indices = manifest.cache_index.to_numpy(dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None]
        self.arrays: dict[str, np.ndarray] = {}

    def get(self, global_index: int) -> torch.Tensor:
        path = self.paths[global_index]
        if path not in self.arrays:
            self.arrays[path] = np.load(self.repository / path, mmap_mode="r", allow_pickle=False)
        epoch = np.asarray(self.arrays[path][self.cache_indices[global_index]], dtype=np.float32)
        return torch.from_numpy((epoch - self.mean) / self.std)


class SingleEpochDataset(Dataset):
    def __init__(
        self,
        repository: Path,
        manifest: pd.DataFrame,
        indices: np.ndarray,
        mapping: Mapping[str, int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray([mapping[str(manifest.iloc[i].event_label)] for i in self.indices], dtype=np.int64)
        self.accessor = EpochAccessor(repository, manifest, mean, std)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_index = int(self.indices[item])
        return self.accessor.get(global_index), torch.tensor(self.labels[item]), torch.tensor(global_index)


class MatchedPairDataset(Dataset):
    """Cross-session same-subject positives and class/paradigm-matched negatives."""

    def __init__(
        self,
        repository: Path,
        manifest: pd.DataFrame,
        indices: np.ndarray,
        mapping: Mapping[str, int],
        mean: np.ndarray,
        std: np.ndarray,
        seed: int,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seed = int(seed)
        self.epoch = 0
        self.accessor = EpochAccessor(repository, manifest, mean, std)
        self.subjects = manifest.subject_id.astype(str).to_numpy()
        self.sessions = manifest.session_id.astype(str).to_numpy()
        self.events = manifest.event_label.astype(str).to_numpy()
        self.labels = np.asarray([mapping[self.events[i]] for i in self.indices], dtype=np.int64)
        by_subject_session_event: dict[tuple[str, str, str], list[int]] = {}
        by_session_event: dict[tuple[str, str], list[int]] = {}
        for global_index in self.indices:
            i = int(global_index)
            by_subject_session_event.setdefault((self.subjects[i], self.sessions[i], self.events[i]), []).append(i)
            by_session_event.setdefault((self.sessions[i], self.events[i]), []).append(i)
        sessions = sorted(set(self.sessions[self.indices]))
        if len(sessions) != 2:
            raise RuntimeError(f"PERSIST-PB requires exactly two OpenBMI sessions, found {sessions}")
        self.other_session = {sessions[0]: sessions[1], sessions[1]: sessions[0]}
        self.by_subject_session_event = {
            key: np.asarray(values, dtype=np.int64) for key, values in by_subject_session_event.items()
        }
        self.by_session_event = {
            key: np.asarray(values, dtype=np.int64) for key, values in by_session_event.items()
        }
        for subject, session, event in self.by_subject_session_event:
            if (subject, self.other_session[session], event) not in self.by_subject_session_event:
                raise RuntimeError(f"No cross-session positive group for {(subject, session, event)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = int(self.indices[item])
        rng = np.random.default_rng(self.seed * 10_000_019 + self.epoch * 1_000_003 + anchor)
        positive_pool = self.by_subject_session_event[
            (self.subjects[anchor], self.other_session[self.sessions[anchor]], self.events[anchor])
        ]
        positive = int(positive_pool[rng.integers(len(positive_pool))])
        negative_pool = self.by_session_event[(self.sessions[anchor], self.events[anchor])]
        negative = anchor
        for _ in range(64):
            candidate = int(negative_pool[rng.integers(len(negative_pool))])
            if self.subjects[candidate] != self.subjects[anchor]:
                negative = candidate
                break
        if negative == anchor:
            valid = negative_pool[self.subjects[negative_pool] != self.subjects[anchor]]
            if not len(valid):
                raise RuntimeError(f"No matched negative for trial {anchor}")
            negative = int(valid[0])
        return (
            self.accessor.get(anchor),
            self.accessor.get(positive),
            self.accessor.get(negative),
            torch.tensor(self.labels[item], dtype=torch.long),
            torch.tensor(anchor, dtype=torch.long),
            torch.tensor(positive, dtype=torch.long),
            torch.tensor(negative, dtype=torch.long),
        )


class PersistPB(nn.Module):
    def __init__(self, n_channels: int, embedding_dim: int, rank: int, task_classes: Mapping[str, int], readout: str):
        super().__init__()
        if not 0 < rank <= embedding_dim:
            raise ValueError("Invalid persistence rank")
        self.embedding_dim = int(embedding_dim)
        self.rank = int(rank)
        self.readout = readout
        self.encoder = EEGNetEncoder(n_channels, embedding_dim)
        initial = torch.randn(embedding_dim, rank)
        initial, _ = torch.linalg.qr(initial, mode="reduced")
        self.basis_raw = nn.Parameter(initial)
        self.gate_logits = nn.ParameterDict({task: nn.Parameter(torch.zeros(rank)) for task in task_classes})
        if readout == "concat":
            self.fast_heads = nn.ModuleDict()
            self.persist_heads = nn.ModuleDict(
                {task: nn.Linear(embedding_dim * 2, count) for task, count in task_classes.items()}
            )
        elif readout == "residual":
            self.fast_heads = nn.ModuleDict(
                {task: nn.Linear(embedding_dim, count) for task, count in task_classes.items()}
            )
            self.persist_heads = nn.ModuleDict(
                {task: nn.Linear(rank, count, bias=False) for task, count in task_classes.items()}
            )
        else:
            raise ValueError(f"Unknown readout: {readout}")

    def basis(self) -> torch.Tensor:
        q, r = torch.linalg.qr(self.basis_raw, mode="reduced")
        sign = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0).detach()
        return q * sign.unsqueeze(0)

    def decompose(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.basis()
        zp = h @ u
        hp = zp @ u.T
        hf = h - hp
        return zp, hp, hf, u

    def logits_from_embedding(
        self,
        h: torch.Tensor,
        task: str,
        gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zp, hp, hf, u = self.decompose(h)
        gate = torch.sigmoid(self.gate_logits[task]) if gate_override is None else gate_override.to(h)
        gated_zp = zp * gate
        gated_hp = gated_zp @ u.T
        if self.readout == "concat":
            logits = self.persist_heads[task](torch.cat([hf, gated_hp], dim=1))
        else:
            logits = self.fast_heads[task](hf) + self.persist_heads[task](gated_zp)
        return logits, {"h": h, "zp": zp, "hp": hp, "hf": hf, "u": u, "gate": gate, "gated_hp": gated_hp}

    def forward(
        self,
        x: torch.Tensor,
        task: str,
        gate_override: torch.Tensor | None = None,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits, parts = self.logits_from_embedding(self.encoder(x), task, gate_override)
        return (logits, parts) if return_parts else logits


@dataclass(frozen=True)
class MethodConfig:
    version: str
    rank: int = 8
    embedding_dim: int = 128
    readout: str = "concat"
    lambda_p: float = 0.20
    lambda_order: float = 0.10
    lambda_budget: float = 0.01
    contrastive_temperature: float = 0.10
    order_margin: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 70
    patience: int = 12
    gradient_clip: float = 5.0
    task_warmup_epochs: int = 0
    persistence_ramp_epochs: int = 0
    budget_start_epoch: int = 0
    budget_ramp_epochs: int = 0


def method_config(version: str) -> MethodConfig:
    presets = {
        "V0": MethodConfig(version="V0"),
        "V1": MethodConfig(
            version="V1",
            lambda_p=0.35,
            lambda_order=0.20,
            lambda_budget=0.005,
            task_warmup_epochs=5,
            persistence_ramp_epochs=5,
            budget_start_epoch=10,
            budget_ramp_epochs=5,
        ),
        "V2": MethodConfig(
            version="V2",
            readout="residual",
            lambda_p=0.35,
            lambda_order=0.20,
            lambda_budget=0.005,
            task_warmup_epochs=5,
            persistence_ramp_epochs=5,
            budget_start_epoch=10,
            budget_ramp_epochs=5,
        ),
        "V3": MethodConfig(
            version="V3",
            rank=4,
            readout="residual",
            lambda_p=0.50,
            lambda_order=0.25,
            lambda_budget=0.0025,
            task_warmup_epochs=5,
            persistence_ramp_epochs=8,
            budget_start_epoch=13,
            budget_ramp_epochs=8,
        ),
    }
    if version not in presets:
        raise ValueError(f"Unknown version: {version}")
    return presets[version]


def loss_weights(config: MethodConfig, epoch: int) -> tuple[float, float, float]:
    if epoch < config.task_warmup_epochs:
        persistence_scale = 0.0
    elif config.persistence_ramp_epochs:
        persistence_scale = min(1.0, (epoch - config.task_warmup_epochs + 1) / config.persistence_ramp_epochs)
    else:
        persistence_scale = 1.0
    if epoch < config.budget_start_epoch:
        budget_scale = 0.0
    elif config.budget_ramp_epochs:
        budget_scale = min(1.0, (epoch - config.budget_start_epoch + 1) / config.budget_ramp_epochs)
    else:
        budget_scale = 1.0
    return (
        config.lambda_p * persistence_scale,
        config.lambda_order * persistence_scale,
        config.lambda_budget * budget_scale,
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
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


def next_cycling(iterator: Iterable[Any], loader: DataLoader) -> tuple[Any, Iterable[Any]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def class_weights(dataset: MatchedPairDataset) -> torch.Tensor:
    counts = np.bincount(dataset.labels)
    values = counts.sum() / (len(counts) * np.maximum(counts, 1))
    return torch.tensor(values, dtype=torch.float32)


@torch.inference_mode()
def evaluate_task(
    model: PersistPB,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    gate_mode: str = "learned",
) -> tuple[float, dict[str, float], pd.DataFrame]:
    model.eval()
    metrics: dict[str, float] = {}
    subject_rows: list[dict[str, Any]] = []
    for task, loader in loaders.items():
        truth: list[np.ndarray] = []
        prediction: list[np.ndarray] = []
        global_indices: list[np.ndarray] = []
        if gate_mode == "learned":
            override = None
        elif gate_mode == "zero":
            override = torch.zeros(model.rank, device=device)
        elif gate_mode == "full":
            override = torch.ones(model.rank, device=device)
        else:
            raise ValueError(gate_mode)
        for x, y, index in loader:
            logits = model(x.to(device, non_blocking=True), task, gate_override=override)
            truth.append(y.numpy())
            prediction.append(logits.argmax(dim=1).cpu().numpy())
            global_indices.append(index.numpy())
        y_true = np.concatenate(truth)
        y_pred = np.concatenate(prediction)
        indices = np.concatenate(global_indices)
        metrics[task] = float(balanced_accuracy_score(y_true, y_pred))
        subject_rows.extend(
            {
                "task": task,
                "global_index": int(index),
                "y_true": int(truth_value),
                "y_pred": int(prediction_value),
                "gate_mode": gate_mode,
            }
            for index, truth_value, prediction_value in zip(indices, y_true, y_pred)
        )
    return float(np.mean(list(metrics.values()))), metrics, pd.DataFrame(subject_rows)


def build_datasets(
    manifest: pd.DataFrame,
    split: Mapping[str, Any],
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    include_test: bool = False,
) -> tuple[dict[str, MatchedPairDataset], dict[str, SingleEpochDataset], dict[str, SingleEpochDataset]]:
    mappings = label_maps(manifest)
    train = {
        task: MatchedPairDataset(
            ROOT,
            manifest,
            subject_indices(manifest, split["train_subjects"], task),
            mappings[task],
            mean,
            std,
            seed * 100 + TASKS.index(task),
        )
        for task in TASKS
    }
    validation = {
        task: SingleEpochDataset(
            ROOT,
            manifest,
            subject_indices(manifest, split["validation_subjects"], task),
            mappings[task],
            mean,
            std,
        )
        for task in TASKS
    }
    test: dict[str, SingleEpochDataset] = {}
    if include_test:
        test_subjects = split.get("test_subjects", split.get("outer_test_subjects"))
        test = {
            task: SingleEpochDataset(
                ROOT,
                manifest,
                subject_indices(manifest, test_subjects, task),
                mappings[task],
                mean,
                std,
            )
            for task in TASKS
        }
    return train, validation, test


def scaler_for_fold(fold: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    base = OLD / "backbone" / "checkpoints" / "eegnet" / f"fold-{fold}" / "seed-0"
    mean_path, std_path = base / "channel_mean.npy", base / "channel_std.npy"
    if not mean_path.exists() or not std_path.exists():
        raise FileNotFoundError("P2 fold-local scaler is missing")
    return np.load(mean_path), np.load(std_path), {
        "mean_path": str(mean_path.relative_to(ROOT)).replace("\\", "/"),
        "std_path": str(std_path.relative_to(ROOT)).replace("\\", "/"),
        "mean_sha256": sha256(mean_path),
        "std_sha256": sha256(std_path),
        "fit_population": "frozen fold train_subjects only (reused from P2)",
    }


def train_one(config: MethodConfig, fold: int, seed: int, output: Path, allow_resume: bool = True) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    complete_path = output / "TRAIN_COMPLETE.json"
    if complete_path.exists():
        record = json.loads(complete_path.read_text(encoding="utf-8"))
        return ROOT / record["best_checkpoint"]
    seed_all(seed)
    manifest = pd.read_parquet(MANIFEST_PATH)
    split = split_for(fold)
    mean, std, scaler_audit = scaler_for_fold(fold)
    train_datasets, validation_datasets, _ = build_datasets(manifest, split, seed, mean, std)
    validation_loaders = {
        task: make_loader(dataset, config.batch_size, False, seed)
        for task, dataset in validation_datasets.items()
    }
    model = PersistPB(
        int(manifest.n_channels.iloc[0]),
        config.embedding_dim,
        config.rank,
        EXPECTED_CLASSES,
        config.readout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("P4 training requires the available GPU; refusing accidental CPU formal training")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criteria = {task: nn.CrossEntropyLoss(weight=class_weights(ds).to(device)) for task, ds in train_datasets.items()}
    steps = min(int(math.ceil(len(ds) / config.batch_size)) for ds in train_datasets.values())
    minimum_coverage_epochs = max(
        int(math.ceil(len(ds) / (steps * config.batch_size))) for ds in train_datasets.values()
    )
    curves: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_epoch = -1
    stale = 0
    start_epoch = 0
    best_path, last_path = output / "best.pt", output / "last.pt"
    if allow_resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        curves = state["curves"]
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale"])
        start_epoch = int(state["epoch"]) + 1
    started = time.time()
    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        for dataset in train_datasets.values():
            dataset.set_epoch(epoch)
        loaders = {
            task: coverage_loader(
                dataset,
                steps,
                config.batch_size,
                epoch,
                seed * 1_000_000 + fold * 10_000 + TASKS.index(task) * 101,
            )
            for task, dataset in train_datasets.items()
        }
        iterators = {task: iter(loader) for task, loader in loaders.items()}
        running = {name: 0.0 for name in ["task", "persistence", "order", "budget", "total", "grad_norm"]}
        lp_weight, lo_weight, lb_weight = loss_weights(config, epoch)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            task_losses: list[torch.Tensor] = []
            persistence_losses: list[torch.Tensor] = []
            order_losses: list[torch.Tensor] = []
            budget_losses: list[torch.Tensor] = []
            for task in TASKS:
                batch, iterators[task] = next_cycling(iterators[task], loaders[task])
                xa, xp, xn, labels, _, _, _ = batch
                batch_size = len(labels)
                x = torch.cat([xa, xp, xn], dim=0).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                h = model.encoder(x)
                ha, hp, hn = h[:batch_size], h[batch_size : 2 * batch_size], h[2 * batch_size :]
                logits_a, parts_a = model.logits_from_embedding(ha, task)
                logits_p, parts_p = model.logits_from_embedding(hp, task)
                logits_n, parts_n = model.logits_from_embedding(hn, task)
                repeated_labels = torch.cat([labels, labels, labels])
                task_logits = torch.cat([logits_a, logits_p, logits_n], dim=0)
                task_losses.append(criteria[task](task_logits, repeated_labels))
                za = F.normalize(parts_a["zp"], dim=1)
                zp = F.normalize(parts_p["zp"], dim=1)
                zn = F.normalize(parts_n["zp"], dim=1)
                contrastive_logits = torch.stack([(za * zp).sum(1), (za * zn).sum(1)], dim=1)
                persistence_losses.append(
                    F.cross_entropy(contrastive_logits / config.contrastive_temperature, torch.zeros(batch_size, dtype=torch.long, device=device))
                )
                sim_p = F.cosine_similarity(parts_a["hp"], parts_p["hp"], dim=1)
                sim_f = F.cosine_similarity(parts_a["hf"], parts_p["hf"], dim=1)
                order_losses.append(F.relu(config.order_margin - sim_p + sim_f).mean())
                budget_losses.append(parts_a["gate"].mean())
            task_loss = torch.stack(task_losses).mean()
            persistence_loss = torch.stack(persistence_losses).mean()
            order_loss = torch.stack(order_losses).mean()
            budget_loss = torch.stack(budget_losses).mean()
            total_loss = task_loss + lp_weight * persistence_loss + lo_weight * order_loss + lb_weight * budget_loss
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            for key, value in {
                "task": task_loss,
                "persistence": persistence_loss,
                "order": order_loss,
                "budget": budget_loss,
                "total": total_loss,
                "grad_norm": grad_norm,
            }.items():
                running[key] += float(value.detach())
        macro, validation_metrics, _ = evaluate_task(model, validation_loaders, device)
        basis = model.basis().detach()
        orthogonality_error = float(torch.max(torch.abs(basis.T @ basis - torch.eye(config.rank, device=device))).cpu())
        row: dict[str, Any] = {
            "epoch": epoch,
            "validation_macro_BA": macro,
            **{f"validation_BA_{task}": validation_metrics[task] for task in TASKS},
            **{f"train_{key}": value / steps for key, value in running.items()},
            "lambda_p_effective": lp_weight,
            "lambda_order_effective": lo_weight,
            "lambda_budget_effective": lb_weight,
            "orthogonality_error": orthogonality_error,
            **{f"gate_mean_{task}": float(torch.sigmoid(model.gate_logits[task]).mean().detach().cpu()) for task in TASKS},
            **{f"gate_active_{task}": int((torch.sigmoid(model.gate_logits[task]) > 0.5).sum().detach().cpu()) for task in TASKS},
            "elapsed_seconds": time.time() - started,
        }
        curves.append(row)
        coverage_complete = epoch + 1 >= minimum_coverage_epochs
        improved = coverage_complete and macro > best_metric + 1e-8
        checkpoint_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric if not improved else macro,
            "best_epoch": best_epoch if not improved else epoch,
            "stale": 0 if improved else (stale + 1 if coverage_complete else 0),
            "curves": curves,
            "method_config": asdict(config),
            "fold": fold,
            "seed": seed,
            "scaler_audit": scaler_audit,
            "train_subjects": list(map(str, split["train_subjects"])),
            "validation_subjects": list(map(str, split["validation_subjects"])),
            "outer_test_subjects": list(map(str, split.get("test_subjects", split.get("outer_test_subjects")))),
            "outer_test_used": False,
        }
        if improved:
            best_metric, best_epoch, stale = macro, epoch, 0
            torch.save(checkpoint_payload, best_path)
        elif coverage_complete:
            stale += 1
        else:
            stale = 0
        checkpoint_payload.update(best_metric=best_metric, best_epoch=best_epoch, stale=stale)
        torch.save(checkpoint_payload, last_path)
        pd.DataFrame(curves).to_csv(output / "curves.csv", index=False)
        print(
            f"[train] {config.version} fold={fold} seed={seed} epoch={epoch} val={macro:.5f} "
            f"best={best_metric:.5f} gates=" + ",".join(f"{task}:{row[f'gate_mean_{task}']:.3f}" for task in TASKS),
            flush=True,
        )
        if coverage_complete and stale >= config.patience:
            break
    if not best_path.exists():
        raise RuntimeError("No eligible best checkpoint was selected")
    write_json(
        complete_path,
        {
            "status": "TRAIN_COMPLETE",
            "version": config.version,
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_macro_BA": best_metric,
            "best_checkpoint": str(best_path.relative_to(ROOT)).replace("\\", "/"),
            "best_checkpoint_sha256": sha256(best_path),
            "epochs_completed": len(curves),
            "minimum_full_coverage_epochs": minimum_coverage_epochs,
            "steps_per_epoch": steps,
            "outer_test_used": False,
            "scaler_audit": scaler_audit,
        },
    )
    return best_path


def load_model(checkpoint_path: Path, manifest: pd.DataFrame, device: torch.device) -> tuple[PersistPB, MethodConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MethodConfig(**checkpoint["method_config"])
    model = PersistPB(
        int(manifest.n_channels.iloc[0]),
        config.embedding_dim,
        config.rank,
        EXPECTED_CLASSES,
        config.readout,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, config, checkpoint


def diagnostic_indices(
    manifest: pd.DataFrame,
    subjects: Sequence[str],
    task: str,
    seed: int,
    per_group: int = 4,
) -> np.ndarray:
    block = manifest[
        manifest.subject_id.astype(str).isin(set(map(str, subjects)))
        & (manifest.paradigm == task)
    ]
    selected: list[int] = []
    for key, group in block.groupby(["subject_id", "session_id", "event_label"], sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        rng = np.random.default_rng(seed + int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16))
        if len(indices) > per_group:
            indices = rng.choice(indices, size=per_group, replace=False)
        selected.extend(map(int, indices))
    return np.asarray(sorted(selected), dtype=np.int64)


@torch.inference_mode()
def extract_diagnostic_representations(
    model: PersistPB,
    manifest: pd.DataFrame,
    subjects: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metadata: list[pd.DataFrame] = []
    arrays: dict[str, list[np.ndarray]] = {"h": [], "zp": [], "hp": [], "hf": []}
    mappings = label_maps(manifest)
    for task in TASKS:
        indices = diagnostic_indices(manifest, subjects, task, seed + TASKS.index(task) * 1000)
        dataset = SingleEpochDataset(ROOT, manifest, indices, mappings[task], mean, std)
        loader = make_loader(dataset, 256, False, seed)
        task_arrays: dict[str, list[np.ndarray]] = {key: [] for key in arrays}
        ordered_indices: list[np.ndarray] = []
        for x, _, global_index in loader:
            h = model.encoder(x.to(device, non_blocking=True))
            zp, hp, hf, _ = model.decompose(h)
            for key, value in {"h": h, "zp": zp, "hp": hp, "hf": hf}.items():
                task_arrays[key].append(value.cpu().numpy().astype(np.float32))
            ordered_indices.append(global_index.numpy())
        joined_indices = np.concatenate(ordered_indices)
        meta = manifest.iloc[joined_indices][["subject_id", "session_id", "paradigm", "event_label"]].copy()
        meta["global_index"] = joined_indices
        metadata.append(meta.reset_index(drop=True))
        for key in arrays:
            arrays[key].append(np.concatenate(task_arrays[key]))
    combined = pd.concat(metadata, ignore_index=True)
    return combined, {key: np.concatenate(values) for key, values in arrays.items()}


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
    pairs_per_subject_event: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    block_positions = np.flatnonzero((metadata.paradigm == task).to_numpy())
    block = metadata.iloc[block_positions].reset_index().rename(columns={"index": "value_position"})
    subjects = sorted(block.subject_id.astype(str).unique(), key=int)
    sessions = sorted(block.session_id.astype(str).unique())
    if len(sessions) != 2:
        raise RuntimeError("Verification requires exactly two sessions")
    grouped = {
        (str(subject), str(session), str(event)): group.value_position.to_numpy(dtype=np.int64)
        for (subject, session, event), group in block.groupby(["subject_id", "session_id", "event_label"])
    }
    rng = np.random.default_rng(seed)
    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    for subject in subjects:
        events = sorted(block[block.subject_id.astype(str) == subject].event_label.astype(str).unique())
        alternatives = [candidate for candidate in subjects if candidate != subject]
        for event in events:
            a = grouped.get((subject, sessions[0], event), np.empty(0, dtype=np.int64))
            b = grouped.get((subject, sessions[1], event), np.empty(0, dtype=np.int64))
            if not len(a) or not len(b):
                continue
            for _ in range(pairs_per_subject_event):
                anchor = int(a[rng.integers(len(a))])
                positive = int(b[rng.integers(len(b))])
                negative_subject = alternatives[int(rng.integers(len(alternatives)))]
                negative_pool = grouped.get((negative_subject, sessions[1], event), np.empty(0, dtype=np.int64))
                if not len(negative_pool):
                    continue
                negative = int(negative_pool[rng.integers(len(negative_pool))])
                left.extend([anchor, anchor])
                right.extend([positive, negative])
                labels.extend([1, 0])
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    return pair_feature(values[left_array], values[right_array]), np.asarray(labels, dtype=np.int64), left_array


def semantic_verification(
    train_metadata: pd.DataFrame,
    train_arrays: Mapping[str, np.ndarray],
    evaluation_metadata: pd.DataFrame,
    evaluation_arrays: Mapping[str, np.ndarray],
    seed: int,
    validation_for_c: tuple[pd.DataFrame, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"per_task": {}}
    for representation in ("zp", "hf"):
        task_values: list[float] = []
        for task in TASKS:
            x_train, y_train, _ = matched_verification_pairs(
                train_metadata, train_arrays[representation], task, seed + TASKS.index(task) * 101
            )
            x_eval, y_eval, _ = matched_verification_pairs(
                evaluation_metadata, evaluation_arrays[representation], task, seed + TASKS.index(task) * 101 + 17
            )
            selected_c = 1.0
            if validation_for_c is not None:
                val_meta, val_arrays = validation_for_c
                x_val, y_val, _ = matched_verification_pairs(
                    val_meta, val_arrays[representation], task, seed + TASKS.index(task) * 101 + 11
                )
                best = (-math.inf, 1.0)
                for c in (0.01, 0.1, 1.0, 10.0, 100.0):
                    candidate = LogisticRegression(C=c, max_iter=2000, class_weight="balanced", solver="liblinear", random_state=seed)
                    candidate.fit(x_train, y_train)
                    auc = roc_auc_score(y_val, candidate.predict_proba(x_val)[:, 1])
                    if auc > best[0]:
                        best = (float(auc), float(c))
                selected_c = best[1]
            verifier = LogisticRegression(
                C=selected_c,
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
            )
            verifier.fit(x_train, y_train)
            auc = float(roc_auc_score(y_eval, verifier.predict_proba(x_eval)[:, 1]))
            result["per_task"].setdefault(task, {})[representation] = {
                "auroc": auc,
                "selected_c": selected_c,
                "train_pairs": int(len(y_train)),
                "evaluation_pairs": int(len(y_eval)),
            }
            task_values.append(auc)
        result[f"{representation}_macro_AUROC"] = float(np.mean(task_values))
    result["macro_gap_zp_minus_hf"] = result["zp_macro_AUROC"] - result["hf_macro_AUROC"]
    for task in TASKS:
        result["per_task"][task]["gap_zp_minus_hf"] = (
            result["per_task"][task]["zp"]["auroc"] - result["per_task"][task]["hf"]["auroc"]
        )
    return result


def geometry_diagnostics(model: PersistPB, arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    h = np.asarray(arrays["h"], dtype=np.float64)
    hp = np.asarray(arrays["hp"], dtype=np.float64)
    hf = np.asarray(arrays["hf"], dtype=np.float64)
    reconstruction = hp + hf
    denominator = np.maximum(np.linalg.norm(h, axis=1), 1e-12)
    relative = np.linalg.norm(reconstruction - h, axis=1) / denominator
    cross = np.sum(hp * hf, axis=1) / np.maximum(np.linalg.norm(hp, axis=1) * np.linalg.norm(hf, axis=1), 1e-12)
    with torch.no_grad():
        u = model.basis().cpu().numpy().astype(np.float64)
    return {
        "orthogonality_error_max_abs": float(np.max(np.abs(u.T @ u - np.eye(model.rank)))),
        "reconstruction_relative_error_max": float(np.max(relative)),
        "reconstruction_relative_error_mean": float(np.mean(relative)),
        "hp_hf_cosine_abs_max": float(np.max(np.abs(cross))),
        "captured_energy_ratio": float(np.square(hp).sum() / np.maximum(np.square(h).sum(), 1e-12)),
        "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
    }


def baseline_validation_reference(fold: int, seed: int) -> dict[str, float]:
    checkpoint = torch.load(
        OLD / "backbone" / "checkpoints" / "eegnet" / f"fold-{fold}" / f"seed-{seed}" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    return {task: float(checkpoint["validation_metrics"][task]) for task in TASKS}


def development(version: str) -> dict[str, Any]:
    config = method_config(version)
    fold, seed = 0, 0
    output = OUT / "development" / version / "fold-0" / "seed-0"
    checkpoint_path = train_one(config, fold, seed, output)
    manifest = pd.read_parquet(MANIFEST_PATH)
    split = split_for(fold)
    mean, std, scaler_audit = scaler_for_fold(fold)
    _, validation_datasets, _ = build_datasets(manifest, split, seed, mean, std)
    validation_loaders = {task: make_loader(ds, config.batch_size, False, seed) for task, ds in validation_datasets.items()}
    device = torch.device("cuda")
    model, _, checkpoint = load_model(checkpoint_path, manifest, device)
    macro, metrics, _ = evaluate_task(model, validation_loaders, device, "learned")
    zero_macro, zero_metrics, _ = evaluate_task(model, validation_loaders, device, "zero")
    train_meta, train_arrays = extract_diagnostic_representations(
        model, manifest, split["train_subjects"], mean, std, device, 120_000 + seed
    )
    val_meta, val_arrays = extract_diagnostic_representations(
        model, manifest, split["validation_subjects"], mean, std, device, 130_000 + seed
    )
    semantic = semantic_verification(train_meta, train_arrays, val_meta, val_arrays, 140_000 + seed)
    geometry = geometry_diagnostics(model, val_arrays)
    gates = {
        task: {
            "values": torch.sigmoid(model.gate_logits[task]).detach().cpu().numpy().tolist(),
            "mean": float(torch.sigmoid(model.gate_logits[task]).mean().detach().cpu()),
            "l1_budget": float(torch.sigmoid(model.gate_logits[task]).sum().detach().cpu()),
            "effective_active_dimensions_gt_0_5": int((torch.sigmoid(model.gate_logits[task]) > 0.5).sum().detach().cpu()),
            "learned_minus_zero_BA": float(metrics[task] - zero_metrics[task]),
        }
        for task in TASKS
    }
    reference = baseline_validation_reference(fold, seed)
    losses = {task: float(metrics[task] - reference[task]) for task in TASKS}
    gate_means = np.asarray([gates[task]["mean"] for task in TASKS])
    checks = {
        "semantic_gap_at_least_0_05": semantic["macro_gap_zp_minus_hf"] >= 0.05,
        "no_task_worse_than_historical_reference_by_over_1pp": all(delta >= -0.01 for delta in losses.values()),
        "noncollapsed_budgets": bool(np.all((gate_means > 0.05) & (gate_means < 0.95))),
        "task_dependent_budget_range_at_least_0_03": float(gate_means.max() - gate_means.min()) >= 0.03,
        "MI_uses_persistence": gates["mi"]["mean"] > 0.05 and gates["mi"]["learned_minus_zero_BA"] > 0.002,
        "geometry_valid": geometry["orthogonality_error_max_abs"] < 1e-5
        and geometry["reconstruction_relative_error_max"] < 1e-5
        and geometry["finite"],
    }
    result = {
        "status": "DEVELOPMENT_GATES_PASS" if all(checks.values()) else "DEVELOPMENT_GATES_FAIL",
        "version": version,
        "fold": fold,
        "seed": seed,
        "data_used": ["TRAIN", "VALIDATION"],
        "held_out_test_used": False,
        "method_config": asdict(config),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": {
            "macro_BA": macro,
            "task_BA": metrics,
            "zero_persistence_macro_BA": zero_macro,
            "zero_persistence_task_BA": zero_metrics,
            "historical_EEGNet_reference": reference,
            "PERSIST_minus_historical_reference": losses,
            "reference_warning": "NOT A FORMAL BASELINE COMPARISON",
        },
        "semantic": semantic,
        "gates": gates,
        "geometry": geometry,
        "checks": checks,
        "scaler_audit": scaler_audit,
        "outer_test_subject_ids_committed_but_not_loaded": list(map(str, split.get("test_subjects", split.get("outer_test_subjects")))),
    }
    write_json(output / "DEVELOPMENT_RESULT.json", result)
    print(json.dumps(clean({"status": result["status"], "checks": checks, "validation": result["validation"], "semantic": semantic, "gates": gates}), indent=2))
    return result


@torch.inference_mode()
def evaluate_gate_vectors(
    model: PersistPB,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    gate_vectors: Mapping[str, torch.Tensor | None],
    policy: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    metrics: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for task, loader in loaders.items():
        truth: list[np.ndarray] = []
        prediction: list[np.ndarray] = []
        indices: list[np.ndarray] = []
        gate = gate_vectors[task]
        for x, y, global_index in loader:
            logits = model(x.to(device, non_blocking=True), task, gate_override=gate)
            truth.append(y.numpy())
            prediction.append(logits.argmax(dim=1).cpu().numpy())
            indices.append(global_index.numpy())
        y_true, y_pred, joined = np.concatenate(truth), np.concatenate(prediction), np.concatenate(indices)
        metrics[task] = float(balanced_accuracy_score(y_true, y_pred))
        rows.extend(
            {
                "task": task,
                "global_index": int(index),
                "y_true": int(target),
                "y_pred": int(prediction_value),
                "gate_policy": policy,
            }
            for index, target, prediction_value in zip(joined, y_true, y_pred)
        )
    return metrics, pd.DataFrame(rows)


def topk_gate_vectors(model: PersistPB, fraction: float, device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    keep = int(round(fraction * model.rank))
    for task in TASKS:
        learned = torch.sigmoid(model.gate_logits[task]).detach()
        gate = torch.zeros(model.rank, device=device)
        if keep:
            selected = torch.topk(learned, k=keep).indices
            gate[selected] = 1.0
        result[task] = gate
    return result


def subject_metrics(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    lookup = manifest[["subject_id"]].copy()
    lookup["global_index"] = np.arange(len(lookup))
    frame = predictions.merge(lookup, on="global_index", how="left", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for (task, policy, subject), group in frame.groupby(["task", "gate_policy", "subject_id"]):
        rows.append(
            {
                "task": task,
                "gate_policy": policy,
                "subject_id": str(subject),
                "balanced_accuracy": float(balanced_accuracy_score(group.y_true, group.y_pred)),
                "trials": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def formal_run(fold: int, seed: int) -> dict[str, Any]:
    lock_path = OUT / "P4_LOCKED_METHOD.json"
    if not lock_path.exists():
        raise RuntimeError("P4 method is not locked")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "P4_METHOD_LOCKED" or lock.get("held_out_test_used_for_selection"):
        raise RuntimeError("Invalid P4 lock")
    config = MethodConfig(**lock["method_config"])
    output = OUT / "openbmi" / f"seed_{seed}" / f"fold-{fold}"
    test_complete = output / "TEST_COMPLETE.json"
    if test_complete.exists():
        return json.loads(test_complete.read_text(encoding="utf-8"))
    checkpoint_path = train_one(config, fold, seed, output)
    manifest = pd.read_parquet(MANIFEST_PATH)
    split = split_for(fold)
    mean, std, scaler_audit = scaler_for_fold(fold)
    device = torch.device("cuda")
    model, loaded_config, checkpoint = load_model(checkpoint_path, manifest, device)
    if asdict(loaded_config) != asdict(config):
        raise RuntimeError("Trained method differs from locked method")

    # The test split is instantiated only after the immutable lock and completed training.
    write_json(
        output / "TEST_ACCESS_STARTED.json",
        {
            "status": "TEST_ACCESS_STARTED_AFTER_LOCK",
            "lock_sha256": sha256(lock_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "test_subjects": list(map(str, split.get("test_subjects", split.get("outer_test_subjects")))),
            "adaptation_after_access_prohibited": True,
        },
    )
    _, validation_datasets, test_datasets = build_datasets(
        manifest, split, seed, mean, std, include_test=True
    )
    test_loaders = {task: make_loader(ds, config.batch_size, False, seed) for task, ds in test_datasets.items()}
    learned_vectors = {task: None for task in TASKS}
    learned_metrics, learned_predictions = evaluate_gate_vectors(model, test_loaders, device, learned_vectors, "learned")
    risk_metrics: list[dict[str, Any]] = []
    all_predictions = [learned_predictions]
    for fraction in (0.0, 0.25, 0.50, 0.75, 1.0):
        vectors = topk_gate_vectors(model, fraction, device)
        metrics, predictions = evaluate_gate_vectors(model, test_loaders, device, vectors, f"topk_{fraction:.2f}")
        all_predictions.append(predictions)
        for task in TASKS:
            risk_metrics.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "task": task,
                    "policy": f"topk_{fraction:.2f}",
                    "budget_fraction": fraction,
                    "active_dimensions": int(round(fraction * config.rank)),
                    "balanced_accuracy": metrics[task],
                }
            )
    for task in TASKS:
        learned_gate = torch.sigmoid(model.gate_logits[task]).detach().cpu().numpy()
        risk_metrics.append(
            {
                "fold": fold,
                "seed": seed,
                "task": task,
                "policy": "learned",
                "budget_fraction": float(learned_gate.mean()),
                "active_dimensions": int((learned_gate > 0.5).sum()),
                "balanced_accuracy": learned_metrics[task],
            }
        )
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(output / "test_predictions.parquet", index=False)
    subjects = subject_metrics(predictions, manifest)
    subjects.insert(0, "seed", seed)
    subjects.insert(0, "fold", fold)
    subjects.to_csv(output / "test_subject_metrics.csv", index=False)
    pd.DataFrame(risk_metrics).to_csv(output / "risk_curve.csv", index=False)

    train_meta, train_arrays = extract_diagnostic_representations(
        model, manifest, split["train_subjects"], mean, std, device, 210_000 + seed * 100 + fold
    )
    val_meta, val_arrays = extract_diagnostic_representations(
        model, manifest, split["validation_subjects"], mean, std, device, 220_000 + seed * 100 + fold
    )
    test_subject_ids = split.get("test_subjects", split.get("outer_test_subjects"))
    test_meta, test_arrays = extract_diagnostic_representations(
        model, manifest, test_subject_ids, mean, std, device, 230_000 + seed * 100 + fold
    )
    semantic = semantic_verification(
        train_meta,
        train_arrays,
        test_meta,
        test_arrays,
        240_000 + seed * 100 + fold,
        validation_for_c=(val_meta, val_arrays),
    )
    geometry = geometry_diagnostics(model, test_arrays)
    gates = {
        task: {
            "values": torch.sigmoid(model.gate_logits[task]).detach().cpu().numpy().tolist(),
            "mean": float(torch.sigmoid(model.gate_logits[task]).mean().detach().cpu()),
            "l1_budget": float(torch.sigmoid(model.gate_logits[task]).sum().detach().cpu()),
            "effective_active_dimensions_gt_0_5": int((torch.sigmoid(model.gate_logits[task]) > 0.5).sum().detach().cpu()),
            "learned_minus_zero_BA": float(
                learned_metrics[task]
                - next(row["balanced_accuracy"] for row in risk_metrics if row["task"] == task and row["policy"] == "topk_0.00")
            ),
        }
        for task in TASKS
    }
    result = {
        "status": "TEST_COMPLETE",
        "fold": fold,
        "seed": seed,
        "version": config.version,
        "lock_sha256": sha256(lock_path),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]),
        "test_task_BA": learned_metrics,
        "semantic": semantic,
        "geometry": geometry,
        "gates": gates,
        "risk_curve_rows": risk_metrics,
        "train_subjects": list(map(str, split["train_subjects"])),
        "validation_subjects": list(map(str, split["validation_subjects"])),
        "test_subjects": list(map(str, test_subject_ids)),
        "subject_disjoint": True,
        "scaler_audit": scaler_audit,
        "test_used_for_adaptation": False,
        "adaptation_after_test": False,
    }
    write_json(output / "RUN_RESULT.json", result)
    write_json(test_complete, result)
    print(json.dumps(clean({"status": "TEST_COMPLETE", "fold": fold, "seed": seed, "test_task_BA": learned_metrics, "semantic_gap": semantic["macro_gap_zp_minus_hf"], "gates": gates}), indent=2))
    return result


def smoke() -> None:
    manifest = pd.read_parquet(MANIFEST_PATH)
    split = split_for(0)
    mean, std, _ = scaler_for_fold(0)
    train, _, _ = build_datasets(manifest, split, 0, mean, std)
    model = PersistPB(62, 128, 8, EXPECTED_CLASSES, "concat").cuda()
    checks: dict[str, Any] = {}
    for task in TASKS:
        xa, xp, xn, labels, anchors, positives, negatives = next(iter(make_loader(Subset(train[task], list(range(4))), 4, False, 0)))
        logits, parts = model(torch.cat([xa, xp, xn]).cuda(), task, return_parts=True)
        anchor_subjects = manifest.iloc[anchors.numpy()].subject_id.astype(str).to_numpy()
        anchor_sessions = manifest.iloc[anchors.numpy()].session_id.astype(str).to_numpy()
        positive_subjects = manifest.iloc[positives.numpy()].subject_id.astype(str).to_numpy()
        positive_sessions = manifest.iloc[positives.numpy()].session_id.astype(str).to_numpy()
        negative_subjects = manifest.iloc[negatives.numpy()].subject_id.astype(str).to_numpy()
        anchor_events = manifest.iloc[anchors.numpy()].event_label.astype(str).to_numpy()
        positive_events = manifest.iloc[positives.numpy()].event_label.astype(str).to_numpy()
        negative_events = manifest.iloc[negatives.numpy()].event_label.astype(str).to_numpy()
        if not np.all(anchor_subjects == positive_subjects):
            raise RuntimeError("Positive-pair subject mismatch")
        if not np.all(anchor_sessions != positive_sessions):
            raise RuntimeError("Positive pair is not cross-session")
        if not np.all(anchor_subjects != negative_subjects):
            raise RuntimeError("Negative pair uses the same subject")
        if not (np.all(anchor_events == positive_events) and np.all(anchor_events == negative_events)):
            raise RuntimeError("Event-matched pair invariant failed")
        checks[task] = {
            "logits_shape": list(logits.shape),
            "parts_shapes": {key: list(value.shape) for key, value in parts.items() if key != "u"},
            "anchor_subjects": anchor_subjects.tolist(),
            "anchor_sessions": anchor_sessions.tolist(),
            "positive_sessions": positive_sessions.tolist(),
            "positive_subject_match": True,
            "negative_subject_mismatch": True,
            "event_matched": True,
            "labels": labels.numpy().tolist(),
        }
    u = model.basis()
    checks["orthogonality_error"] = float(torch.max(torch.abs(u.T @ u - torch.eye(8, device="cuda"))).detach().cpu())
    write_json(OUT / "development" / "SMOKE_TEST.json", {"status": "PASS", "checks": checks, "outer_test_used": False})
    print(json.dumps(clean(checks), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "development", "formal"], required=True)
    parser.add_argument("--version", choices=["V0", "V1", "V2", "V3"], default="V0")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, choices=range(5))
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke()
    elif args.mode == "development":
        development(args.version)
    else:
        if args.fold is None or args.seed is None:
            parser.error("--fold and --seed are required for formal mode")
        formal_run(args.fold, args.seed)


if __name__ == "__main__":
    main()
