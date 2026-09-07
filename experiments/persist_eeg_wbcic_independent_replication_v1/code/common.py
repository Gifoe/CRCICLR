"""Shared implementation for the frozen WBCIC independent replication.

Only the historical 41-subject development whitelist is materialized.  The
sealed 10-subject WBCIC outer split and the OpenBMI holdout are never imported,
enumerated, or addressed by this module.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
PROTOCOL_PATH = EXP / "WBCIC_REPLICATION_PROTOCOL_FROZEN.json"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"
RUNS = RUNTIME / "runs"

EPS = 1e-12
RIDGE_ALPHA = 1.0
METHODS = ("ERM", "DANN", "CORAL", "MMD")
BACKBONES = ("eegnet",)


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
    if value.get("schema") != "PERSIST_EEG_WBCIC_INDEPENDENT_REPLICATION_V1":
        raise RuntimeError("unexpected WBCIC replication protocol schema")
    if value.get("frozen_before_training") is not True or value.get("frozen_before_outcome_evaluation") is not True:
        raise RuntimeError("stress-test protocol was not frozen before execution")
    if value.get("repository_start_sha") != "3654486141c91333e0507e95be98f4bdc41c0254":
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
    pool = set(frozen_subjects())
    outcome = tuple(subject_sort(row["outcome"]))
    validation = tuple(subject_sort(row["validation_discovery"]))
    train = tuple(subject_sort(row["model_fit"]))
    if set(outcome) | set(validation) | set(train) != pool:
        raise RuntimeError(f"fold {fold} is not exhaustive")
    if len(outcome) not in (8, 9) or len(validation) not in (8, 9) or len(train) not in (24, 25):
        raise RuntimeError(f"fold {fold} cardinality failure")
    if set(outcome) & set(train) or set(outcome) & set(validation) or set(train) & set(validation):
        raise RuntimeError(f"fold {fold} subject overlap")
    return {
        "outcome": outcome,
        "validation_discovery": validation,
        "model_fit": train,
        "source": train,
    }


def _cache_candidates() -> list[Path]:
    values: list[Path] = [EXP / "runtime" / "cache"]
    explicit = os.environ.get("PERSIST_WBCIC_REPLICATION_CACHE", "").strip()
    if explicit:
        values.insert(0, Path(explicit))
    unique: list[Path] = []
    for value in values:
        resolved = value.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def locate_authorized_cache() -> Path:
    for root in _cache_candidates():
        signal = root / "WBCIC_DEVELOPMENT_MI_RAW.npy"
        metadata = root / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
        if signal.is_file() and metadata.is_file():
            return root
    raise FileNotFoundError("authorized 41-subject WBCIC cache not found")


def load_data() -> DevelopmentData:
    root = locate_authorized_cache()
    meta_path = root / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
    x_path = root / "WBCIC_DEVELOPMENT_MI_RAW.npy"
    metadata = pd.read_parquet(meta_path, columns=["subject_id", "session_id", "label"], engine="pyarrow")
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = metadata.label.astype(int)
    x = np.load(x_path, mmap_mode="r", allow_pickle=False)
    expected = set(frozen_subjects())
    observed = set(metadata.subject_id.unique())
    cells = metadata.groupby(["subject_id", "session_id", "label"]).size()
    valid = bool(
        x.shape == (len(metadata), 58, 1000)
        and x.dtype == np.float16
        and observed == expected
        and set(metadata.session_id.unique()) == {0, 1, 2}
        and set(metadata.label.unique()) == {0, 1}
        and len(cells) == 41 * 3 * 2
        and int(cells.min()) >= 20
        and np.isfinite(np.asarray(x[::197], dtype=np.float32)).all()
    )
    if not valid:
        raise RuntimeError(
            f"authorized-cache audit failed rows={len(metadata)} shape={x.shape} "
            f"subjects={subject_sort(observed)} cells={sorted(set(cells.tolist()))}"
        )
    return DevelopmentData(x=x, metadata=metadata.reset_index(drop=True), cache_root=root)


def row_indices(metadata: pd.DataFrame, subjects: Sequence[str], sessions: Sequence[int]) -> np.ndarray:
    mask = metadata.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy(copy=True)
    mask &= metadata.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def compute_normalizer(raw: torch.Tensor, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    # Frozen WBCIC preprocessing already applies uV/20 and clipping.  The
    # historical protocol explicitly forbids cross-subject amplitude statistics.
    del indices
    return (
        torch.zeros(raw.shape[1], dtype=torch.float32, device=raw.device),
        torch.ones(raw.shape[1], dtype=torch.float32, device=raw.device),
    )


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-6)


class StandardEEGNet(nn.Module):
    """Frozen historical WBCIC EEGNet with a 32-D representation."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (58, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        self.embedding = nn.Sequential(nn.Linear(16 * 31, 32), nn.ELU(), nn.LayerNorm(32))
        self.head = nn.Linear(32, 2)
        self.representation_dim = 32

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(backbone: str, initialization_seed: int) -> nn.Module:
    set_seed(initialization_seed)
    if backbone == "eegnet":
        return StandardEEGNet()
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
                raise RuntimeError("training minibatch crossed the frozen model-fit subjects")
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
        "DANN_subject_head": "Linear32_128_ReLU_Dropout0.2_Linear128_K" if method == "DANN" else None,
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
    observed_sessions = set(np.unique(np.asarray(sessions, dtype=np.int64)).tolist())
    if observed_sessions != {0, 1}:
        raise RuntimeError(f"identity probe requires source S1/S2 encoded as ses-0/ses-1, got {sorted(observed_sessions)}")
    forward = identity_direction(features, subjects, sessions, 0, 1)
    reverse = identity_direction(features, subjects, sessions, 1, 0)
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
    means1 = np.stack([value[(subjects.astype(str) == subject) & (sessions == 0)].mean(axis=0) for subject in ordered])
    means2 = np.stack([value[(subjects.astype(str) == subject) & (sessions == 1)].mean(axis=0) for subject in ordered])
    centered1 = means1 - means1.mean(axis=0, keepdims=True)
    centered2 = means2 - means2.mean(axis=0, keepdims=True)
    cross = (centered1.T @ centered2 + centered2.T @ centered1) / (2.0 * max(len(ordered) - 1, 1))
    eigenvalues, eigenvectors = np.linalg.eigh((cross + cross.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    pool = eigenvectors[:, order]
    eigenvalues = eigenvalues[order]
    if np.linalg.norm(pool.T @ pool - np.eye(pool.shape[1]), ord="fro") > 1e-8:
        raise RuntimeError("persistent eigenspace is not orthonormal")
    rows: list[dict[str, float]] = []
    for index in range(min(count, pool.shape[1])):
        direction = pool[:, index]
        projection1 = centered1 @ direction
        projection2 = centered2 @ direction
        if np.std(projection1) < 1e-12 or np.std(projection2) < 1e-12:
            persistence = 0.0
        else:
            persistence = float(np.corrcoef(projection1, projection2)[0, 1])
        rows.append(
            {
                "pool_index": index,
                "persistence": persistence,
                "geometry_strength": float(eigenvalues[index]),
            }
        )
    return center.astype(np.float64), pool[:, :count].astype(np.float64), rows


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
    frozen_center: np.ndarray | None = None,
    frozen_basis: np.ndarray | None = None,
    frozen_meta: list[dict[str, float]] | None = None,
) -> pd.DataFrame:
    if frozen_center is None or frozen_basis is None or frozen_meta is None:
        center, basis, direction_meta = persistent_directions(
            source["features"], source["subjects"], source["sessions"], count=int(protocol()["secondary_direction_audit"]["candidate_count"])
        )
    else:
        center = np.asarray(frozen_center, dtype=np.float64)
        basis = np.asarray(frozen_basis, dtype=np.float64)
        direction_meta = frozen_meta
    identity_mask = np.isin(source["subjects"].astype(str), list(map(str, primary_identity_subjects)))
    full_identity = identity_probe(
        source["features"][identity_mask], source["subjects"][identity_mask], source["sessions"][identity_mask]
    )["identity_symmetric"]
    weight = model.head.weight.detach().float().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().float().cpu().numpy().astype(np.float64)
    # Recompute both intact and intervened logits through the same frozen head
    # in float64.  Mixing autocast inference logits with float64 erased logits
    # would create a small, purely numerical pseudo-intervention effect.
    clean_source_logits = np.asarray(source["features"], dtype=np.float64) @ weight.T + bias
    clean_outcome_logits = np.asarray(outcome["features"], dtype=np.float64) @ weight.T + bias
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
        subject_f1_effect: list[float] = []
        for subject in subject_sort(np.unique(outcome["subjects"].astype(str))):
            mask = outcome["subjects"].astype(str) == subject
            clean_ba = balanced_accuracy_score(outcome["labels"][mask], clean_outcome_prediction[mask])
            erased_ba = balanced_accuracy_score(outcome["labels"][mask], erased_prediction[mask])
            subject_ba_effect.append(float(clean_ba - erased_ba))
            clean_f1 = f1_score(outcome["labels"][mask], clean_outcome_prediction[mask], average="macro")
            erased_f1 = f1_score(outcome["labels"][mask], erased_prediction[mask], average="macro")
            subject_f1_effect.append(float(clean_f1 - erased_f1))
        rows.append(
            {
                "backbone": backbone,
                "method": method,
                "lambda": float(lam),
                "fold": int(fold),
                "seed": int(seed),
                "direction_id": int(direction_id),
                "direction_rank": int(direction_id + 1),
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
                "outcome_F1_effect": float(np.mean(subject_f1_effect)),
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
    values = [("ERM", 0.0)]
    for method in ("DANN", "CORAL", "MMD"):
        values.extend((method, float(lam)) for lam in protocol()["methods"][method]["lambda_grid"])
    return values


def unit_dir(backbone: str, fold: int, seed: int) -> Path:
    return RUNS / backbone / f"fold-{fold}" / f"seed-{seed}"
