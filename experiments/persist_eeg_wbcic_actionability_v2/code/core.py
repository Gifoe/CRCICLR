"""Shared numerical and EEGNet components for the WBCIC audit."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2"
OUT = EXP_ROOT / "outputs"
PROTOCOL = OUT / "protocol"
RESULTS = OUT / "results"
CACHE = OUT / "cache" / "wbcic_epochs"
MODEL = OUT / "model"
SCOPE_PATH = PROTOCOL / "DEVELOPMENT_SCOPE_LOCK.json"
ACTION_LOCK_PATH = PROTOCOL / "ACTIONABILITY_PROTOCOL_LOCK.json"
FROZEN_PATH = PROTOCOL / "REPRESENTATION_FROZEN.json"
IMPLEMENTATION_ID = "persist_eeg_wbcic_eegnet_actionability_v2_20260817"
SEED = 20260817
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 10_000
BLOCKS = (("P01_04", 0, 4), ("P05_08", 4, 8), ("P09_16", 8, 16), ("P17_32", 16, 32))
EPS = 1e-12


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pd.DataFrame([clean(dict(row)) for row in rows]).to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_lines(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def require_development_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        SCOPE_PATH,
        ACTION_LOCK_PATH,
        PROTOCOL / "CACHE_SCOPE_AUDIT.json",
        PROTOCOL / "REPRESENTATION_CANDIDATE_LOCK.json",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required prospective/cache lock missing: {path}")
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    action = json.loads(ACTION_LOCK_PATH.read_text(encoding="utf-8"))
    cache_scope = json.loads((PROTOCOL / "CACHE_SCOPE_AUDIT.json").read_text(encoding="utf-8"))
    subjects = list(map(str, scope.get("allowed_subjects", [])))
    if (
        len(subjects) != 41
        or scope.get("outer_subject_ids_present") is not False
        or scope.get("runtime_must_not_open") != "OUTER_SPLIT_LOCK.json"
        or action.get("outer_test_state") != "OUTER_TEST_LOCKED"
        or cache_scope.get("status") != "DEVELOPMENT_CACHE_COMPLETE"
        or cache_scope.get("outer_subject_ids_materialized") is not False
        or cache_scope.get("allowed_subjects_hash") != scope.get("allowed_subjects_hash")
    ):
        raise RuntimeError("DATA_SCOPE_VIOLATION")
    materialized = {path.name for path in CACHE.iterdir() if path.is_dir()}
    if materialized != set(subjects):
        raise RuntimeError("DATA_SCOPE_VIOLATION: cache materialization differs from scope")
    return scope, action


def audit_roles(scope: Mapping[str, Any], fold: int) -> tuple[list[str], list[str], list[str]]:
    value = scope["audit_roles"][str(fold)]
    allowed = set(map(str, scope["allowed_subjects"]))
    outcome = list(map(str, value["outcome"]))
    discovery = list(map(str, value["discovery_decision"]))
    model_fit = list(map(str, value["model_fit"]))
    groups = [set(outcome), set(discovery), set(model_fit)]
    if set.union(*groups) != allowed or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError(f"DATA_SCOPE_VIOLATION: fold {fold} roles are not disjoint/exhaustive")
    return outcome, discovery, model_fit


def cache_paths(subject: str, session: int, cache_root: Path = CACHE) -> tuple[Path, Path]:
    return (
        cache_root / subject / f"ses-{session}_epochs.npy",
        cache_root / subject / f"ses-{session}_labels.npy",
    )


class EpochDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Lazy mmap-backed session dataset that is safe with spawned workers."""

    def __init__(self, subjects: Sequence[str], sessions: Iterable[int], cache_root: Path = CACHE):
        self.subjects = list(map(str, subjects))
        self.sessions = list(map(int, sessions))
        self.cache_root = Path(cache_root)
        self.records: list[dict[str, Any]] = []
        self.ends: list[int] = []
        self._arrays: dict[int, np.ndarray] = {}
        total = 0
        for subject_index, subject in enumerate(self.subjects):
            for session in self.sessions:
                epochs_path, labels_path = cache_paths(subject, session, self.cache_root)
                if not epochs_path.is_file() or not labels_path.is_file():
                    raise FileNotFoundError(f"Missing cache for {subject} ses-{session}")
                labels = np.load(labels_path, allow_pickle=False)
                shape = np.load(epochs_path, mmap_mode="r", allow_pickle=False).shape
                if shape != (len(labels), 58, 1000) or set(labels.astype(int).tolist()) != {0, 1}:
                    raise RuntimeError(f"Malformed cached session: {subject} ses-{session}")
                total += len(labels)
                self.records.append(
                    {
                        "epochs": str(epochs_path),
                        "labels": labels.astype(np.int64),
                        "subject_index": subject_index,
                        "session": session,
                        "start": total - len(labels),
                    }
                )
                self.ends.append(total)

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record_index = bisect.bisect_right(self.ends, int(index))
        record = self.records[record_index]
        local = int(index) - int(record["start"])
        if record_index not in self._arrays:
            self._arrays[record_index] = np.load(record["epochs"], mmap_mode="r", allow_pickle=False)
        epoch = np.array(self._arrays[record_index][local], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(epoch),
            torch.tensor(int(record["labels"][local]), dtype=torch.long),
            torch.tensor(int(record["subject_index"]), dtype=torch.long),
            torch.tensor(int(record["session"]), dtype=torch.long),
        )


class EEGNet(nn.Module):
    """Frozen 58-channel, 4-second EEGNet configuration with a 32-D embedding."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, kernel_size=(1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, kernel_size=(58, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(float(dropout))
        self.separable_depth = nn.Conv2d(
            16, 16, kernel_size=(1, 16), padding="same", groups=16, bias=False
        )
        self.separable_point = nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(float(dropout))
        self.embedding = nn.Linear(16 * 31, 32)
        self.embedding_norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (58, 1000):
            raise ValueError(f"EEGNet expects (batch,58,1000), received {tuple(x.shape)}")
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.separable_depth(value)
        value = self.separable_point(value)
        value = self.drop2(self.pool2(F.elu(self.bn3(value))))
        value = value.flatten(1)
        return self.embedding_norm(F.elu(self.embedding(value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_model(
    subjects: Sequence[str],
    sessions: Sequence[int],
    config: Mapping[str, Any],
    checkpoint: Path,
    fold_label: str,
    device: torch.device,
    workers: int,
) -> tuple[EEGNet, dict[str, Any]]:
    seed = stable_seed(SEED, "train", fold_label, config["id"])
    seed_all(seed)
    dataset = EpochDataset(subjects, sessions)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
        generator=generator,
    )
    model = EEGNet(float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history: list[dict[str, Any]] = []
    for epoch in range(int(config["epochs"])):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite EEGNet training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(y)
            correct += int((logits.detach().argmax(1) == y).sum())
            seen += len(y)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(seen, 1),
                "train_accuracy": correct / max(seen, 1),
            }
        )
        print(
            f"[train {fold_label} {config['id']}] epoch={epoch + 1}/{config['epochs']} "
            f"loss={history[-1]['train_loss']:.5f} acc={history[-1]['train_accuracy']:.4f}",
            flush=True,
        )
    payload = {
        "implementation_id": IMPLEMENTATION_ID,
        "fold_label": fold_label,
        "config": dict(config),
        "seed": seed,
        "train_subjects": list(subjects),
        "train_subjects_hash": sha_lines(list(subjects)),
        "train_sessions": list(map(int, sessions)),
        "model_state_sha256": model_state_sha256(model),
        "history": history,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".part")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    payload["checkpoint_sha256"] = sha256_file(checkpoint)
    return model, payload


def load_model(checkpoint: Path, device: torch.device) -> tuple[EEGNet, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = EEGNet(float(config["dropout"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise RuntimeError(f"Checkpoint state hash mismatch: {checkpoint}")
    model.to(device).eval()
    return model, payload


def infer(
    model: EEGNet,
    subjects: Sequence[str],
    sessions: Sequence[int],
    device: torch.device,
    workers: int,
    batch_size: int = 128,
    cache_root: Path = CACHE,
) -> dict[str, np.ndarray]:
    dataset = EpochDataset(subjects, sessions, cache_root=cache_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subject_index: list[np.ndarray] = []
    session: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, y, sid, ses in loader:
            x = x.to(device, non_blocking=True)
            h = model.forward_features(x)
            z = model.head(h)
            embeddings.append(h.float().cpu().numpy())
            logits.append(z.float().cpu().numpy())
            labels.append(y.numpy())
            subject_index.append(sid.numpy())
            session.append(ses.numpy())
    return {
        "embeddings": np.concatenate(embeddings).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.int64),
        "subject_index": np.concatenate(subject_index).astype(np.int16),
        "session": np.concatenate(session).astype(np.int8),
        "subjects": np.asarray(list(subjects)),
    }


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    recalls = [float(np.mean(prediction[truth == label] == label)) for label in np.unique(truth)]
    return float(np.mean(recalls)) if recalls else float("nan")


def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value -= value.max(axis=-1, keepdims=True)
    exp = np.exp(value)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), EPS)


def ce_rows(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probability = softmax(logits)
    return -np.log(np.clip(probability[np.arange(len(labels)), labels], EPS, 1.0))


def true_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    own = value[np.arange(len(labels)), labels]
    other = value.copy()
    other[np.arange(len(labels)), labels] = -np.inf
    return own - other.max(axis=1)


def centered_rms(delta: np.ndarray) -> float:
    value = np.asarray(delta, dtype=np.float64)
    centered = value - value.mean(axis=-1, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=-1))))


def exact_matched_delta(residual: np.ndarray, target_delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projection = np.asarray(residual, dtype=np.float64) @ basis @ basis.T
    target_norm = np.linalg.norm(target_delta, axis=1)
    random_norm = np.linalg.norm(projection, axis=1)
    output = projection.copy()
    bad = random_norm <= EPS
    if np.any(bad):
        output[bad] = basis[:, 0][None, :]
        random_norm[bad] = 1.0
    output *= (target_norm / np.maximum(random_norm, EPS))[:, None]
    error = float(np.max(np.abs(np.linalg.norm(output, axis=1) - target_norm)))
    if error > 5e-6:
        raise RuntimeError(f"Matched-random displacement norm error {error}")
    return output


def random_bases(rank: int, fold: int, block: str, excluded: np.ndarray | None = None) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    projector = None
    if excluded is not None and excluded.size:
        q_excluded, _ = np.linalg.qr(np.asarray(excluded, dtype=np.float64))
        projector = np.eye(32) - q_excluded @ q_excluded.T
    for draw in range(RANDOM_DRAWS):
        rng = np.random.default_rng(stable_seed(IMPLEMENTATION_ID, "random-basis", fold, block, draw))
        value = rng.normal(size=(32, max(rank, 1)))
        if projector is not None:
            value = projector @ value
        q, _ = np.linalg.qr(value)
        if q.shape[1] < rank or np.linalg.matrix_rank(q[:, :rank]) < rank:
            raise RuntimeError("Unable to construct requested random control rank")
        result.append(np.asarray(q[:, :rank], dtype=np.float64))
    return result


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap_mean requires a finite nonempty vector")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    samples = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def signflip_p(values: np.ndarray, direction: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    n = len(array)
    if n <= 20:
        numbers = np.arange(2**n, dtype=np.uint32)[:, None]
        bits = ((numbers >> np.arange(n, dtype=np.uint32)[None, :]) & 1).astype(np.float64)
        signs = 2 * bits - 1
    else:
        rng = np.random.default_rng(stable_seed("signflip", direction, *array.tolist()))
        signs = rng.choice((-1.0, 1.0), size=(100_000, n))
    distribution = (signs * array[None, :]).mean(axis=1)
    observed = float(array.mean())
    if direction == "positive":
        return float((1 + np.sum(distribution >= observed - 1e-15)) / (1 + len(distribution)))
    if direction == "negative":
        return float((1 + np.sum(distribution <= observed + 1e-15)) / (1 + len(distribution)))
    raise ValueError(direction)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * float(value)))
        adjusted[name] = running
    return adjusted


def protected_projection(weight: np.ndarray, harmful: np.ndarray, alpha: float) -> np.ndarray:
    basis = np.asarray(harmful, dtype=np.float64)
    projector = basis @ basis.T if basis.size else np.zeros((weight.shape[1], weight.shape[1]))
    return np.asarray(weight, dtype=np.float64) @ (np.eye(weight.shape[1]) - float(alpha) * projector)


def protected_relative_error(
    weight_before: np.ndarray, weight_after: np.ndarray, protected: np.ndarray
) -> float:
    if protected.size == 0:
        return 0.0
    before = np.asarray(weight_before, dtype=np.float64) @ protected
    after = np.asarray(weight_after, dtype=np.float64) @ protected
    return float(np.linalg.norm(after - before) / max(np.linalg.norm(before), EPS))
