"""Shared scope, training, inference, and numerical routines."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models import build_model


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_multibackbone_final_closure"
OUT = EXP_ROOT / "outputs"
PROTOCOL = OUT / "protocol"
RESULTS = OUT / "results"
MODEL = OUT / "model"
REFERENCE_EXP = REPO_ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2"
REFERENCE_OUT = REFERENCE_EXP / "outputs"
CACHE = Path(os.environ.get("PERSIST_WBCIC_CACHE", REFERENCE_OUT / "cache" / "wbcic_epochs"))
SCOPE_PATH = REFERENCE_OUT / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json"
REFERENCE_ACTION_PATH = REFERENCE_OUT / "protocol" / "ACTIONABILITY_PROTOCOL_LOCK.json"
IMPLEMENTATION_ID = "persist_eeg_multibackbone_final_closure_20260818"
PRIMARY_SEED = 20260817
REPLICATION_SEEDS = (20260823, 20260829)
BACKBONES = ("FBCNet", "EEGConformer", "DeepConvNet", "TeCh")
BLOCKS = (("P01_04", 0, 4), ("P05_08", 4, 8), ("P09_16", 8, 16), ("P17_32", 16, 32))
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 10_000
EPS = 1e-12


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_once(path: Path, payload: Any) -> None:
    text = json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Frozen artifact mismatch; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pd.DataFrame([clean(dict(row)) for row in rows]).to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=5
        ).strip()
    except Exception:
        return None


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def device_from_argument(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def require_development_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        SCOPE_PATH,
        REFERENCE_ACTION_PATH,
        REFERENCE_OUT / "protocol" / "CACHE_SCOPE_AUDIT.json",
        PROTOCOL / "BACKBONE_ROSTER_LOCK.json",
        PROTOCOL / "MULTIBACKBONE_PROTOCOL_LOCK.json",
        PROTOCOL / "MULTIBACKBONE_MULTIPLICITY_LOCK.json",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required prospective/cache lock missing: {path}")
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    action = json.loads(REFERENCE_ACTION_PATH.read_text(encoding="utf-8"))
    cache_scope = json.loads(
        (REFERENCE_OUT / "protocol" / "CACHE_SCOPE_AUDIT.json").read_text(encoding="utf-8")
    )
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
        raise RuntimeError("DATA_SCOPE_VIOLATION: cache materialization differs from frozen scope")
    return scope, action


def audit_roles(scope: Mapping[str, Any], fold: int) -> tuple[list[str], list[str], list[str]]:
    value = scope["audit_roles"][str(fold)]
    allowed = set(map(str, scope["allowed_subjects"]))
    outcome = list(map(str, value["outcome"]))
    discovery = list(map(str, value["discovery_decision"]))
    model_fit = list(map(str, value["model_fit"]))
    groups = [set(outcome), set(discovery), set(model_fit)]
    if set.union(*groups) != allowed or any(
        groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)
    ):
        raise RuntimeError(f"DATA_SCOPE_VIOLATION: fold {fold} roles are invalid")
    return outcome, discovery, model_fit


def cache_paths(subject: str, session: int) -> tuple[Path, Path]:
    return (
        CACHE / subject / f"ses-{session}_epochs.npy",
        CACHE / subject / f"ses-{session}_labels.npy",
    )


class EpochDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Lazy mmap-backed development-only WBCIC epoch dataset."""

    def __init__(self, subjects: Sequence[str], sessions: Iterable[int]):
        self.subjects = list(map(str, subjects))
        self.sessions = list(map(int, sessions))
        self.records: list[dict[str, Any]] = []
        self.ends: list[int] = []
        self._arrays: dict[int, np.ndarray] = {}
        total = 0
        for subject_index, subject in enumerate(self.subjects):
            for session in self.sessions:
                epochs_path, labels_path = cache_paths(subject, session)
                if not epochs_path.is_file() or not labels_path.is_file():
                    raise FileNotFoundError(f"Missing cache for {subject} ses-{session}")
                labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
                shape = np.load(epochs_path, mmap_mode="r", allow_pickle=False).shape
                if shape != (len(labels), 58, 1000) or set(labels.tolist()) != {0, 1}:
                    raise RuntimeError(f"Malformed cached session: {subject} ses-{session}")
                total += len(labels)
                self.records.append(
                    {
                        "epochs": str(epochs_path),
                        "labels": labels,
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


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_model(
    backbone: str,
    subjects: Sequence[str],
    config: Mapping[str, Any],
    checkpoint: Path,
    fold_label: str,
    device: torch.device,
    workers: int,
    experiment_seed: int = PRIMARY_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    seed = stable_seed(experiment_seed, "train", backbone, fold_label, config["id"])
    seed_all(seed)
    dataset = EpochDataset(subjects, (0, 1))
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
    model = build_model(backbone, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history: list[dict[str, Any]] = []
    for epoch in range(int(config["epochs"])):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=float(config.get("label_smoothing", 0.0)))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {backbone} training loss")
            scaler.scale(loss).backward()
            if float(config.get("gradient_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(y)
            correct += int((logits.detach().argmax(1) == y).sum())
            seen += len(y)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(seen, 1),
                "train_accuracy": correct / max(seen, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[train {backbone} {fold_label} {config['id']}] {epoch + 1}/{config['epochs']} "
            f"loss={history[-1]['train_loss']:.5f} acc={history[-1]['train_accuracy']:.4f}",
            flush=True,
        )
    payload = {
        "implementation_id": IMPLEMENTATION_ID,
        "backbone": backbone,
        "fold_label": fold_label,
        "config": dict(config),
        "experiment_seed": int(experiment_seed),
        "seed": seed,
        "train_subjects": list(subjects),
        "train_subjects_hash": sha_lines(list(subjects)),
        "train_sessions": [0, 1],
        "representation_dim": int(model.representation_dim),
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


def load_model(checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(str(payload["backbone"]), payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise RuntimeError(f"Checkpoint state hash mismatch: {checkpoint}")
    model.to(device).eval()
    return model, payload


def get_or_train(
    backbone: str,
    checkpoint: Path,
    subjects: Sequence[str],
    config: Mapping[str, Any],
    fold_label: str,
    device: torch.device,
    workers: int,
    experiment_seed: int = PRIMARY_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    if checkpoint.exists():
        model, payload = load_model(checkpoint, device)
        if (
            payload.get("implementation_id") != IMPLEMENTATION_ID
            or payload.get("backbone") != backbone
            or payload.get("fold_label") != fold_label
            or payload.get("config") != dict(config)
            or payload.get("experiment_seed") != int(experiment_seed)
            or payload.get("train_subjects") != list(subjects)
            or payload.get("train_sessions") != [0, 1]
        ):
            raise RuntimeError(f"Existing checkpoint does not match frozen job: {checkpoint}")
        payload["checkpoint_sha256"] = sha256_file(checkpoint)
        print(f"[resume] {checkpoint}", flush=True)
        return model, payload
    return train_model(
        backbone, subjects, config, checkpoint, fold_label, device, workers, experiment_seed
    )


def infer(
    model: nn.Module,
    subjects: Sequence[str],
    sessions: Sequence[int],
    device: torch.device,
    workers: int,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    dataset = EpochDataset(subjects, sessions)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=max(0, int(workers)),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    embeddings, logits, labels, subject_index, session = [], [], [], [], []
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
    truth, prediction = np.asarray(y_true, int), np.asarray(y_pred, int)
    return float(np.mean([np.mean(prediction[truth == label] == label) for label in np.unique(truth)]))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, prediction = np.asarray(y_true, int), np.asarray(y_pred, int)
    values = []
    for label in (0, 1):
        tp = np.sum((truth == label) & (prediction == label))
        fp = np.sum((truth != label) & (prediction == label))
        fn = np.sum((truth == label) & (prediction != label))
        values.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    return float(np.mean(values))


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


def project_rows(value: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (np.asarray(value, np.float64) @ basis) @ basis.T


def exact_matched_delta(residual: np.ndarray, target_delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projection = project_rows(residual, basis)
    target_norm = np.linalg.norm(target_delta, axis=1)
    random_norm = np.linalg.norm(projection, axis=1)
    output = projection.copy()
    bad = random_norm <= EPS
    if np.any(bad):
        output[bad] = basis[:, 0][None, :]
        random_norm[bad] = 1.0
    output *= (target_norm / np.maximum(random_norm, EPS))[:, None]
    if float(np.max(np.abs(np.linalg.norm(output, axis=1) - target_norm))) > 5e-6:
        raise RuntimeError("Matched-random displacement norm error")
    return output


def random_bases(dim: int, rank: int, backbone: str, fold: int, block: str) -> list[np.ndarray]:
    output = []
    for draw in range(RANDOM_DRAWS):
        rng = np.random.default_rng(
            stable_seed(IMPLEMENTATION_ID, "random-basis", backbone, fold, block, draw)
        )
        q, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
        output.append(np.asarray(q[:, :rank], np.float64))
    return output


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
    rng = np.random.default_rng(stable_seed("signflip", direction, *array.tolist()))
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(array)))
    distribution = (signs * array[None, :]).mean(axis=1)
    observed = float(array.mean())
    if direction == "positive":
        return float((1 + np.sum(distribution >= observed - 1e-15)) / (1 + len(distribution)))
    if direction == "negative":
        return float((1 + np.sum(distribution <= observed + 1e-15)) / (1 + len(distribution)))
    raise ValueError(direction)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted, running = {}, 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * float(value)))
        adjusted[name] = running
    return adjusted


def discovered_basis(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    subjects = arrays["subjects"].astype(str).tolist()
    h = arrays["embeddings"].astype(np.float64)
    sid, session = arrays["subject_index"].astype(int), arrays["session"].astype(int)
    dim = h.shape[1]
    centroids: dict[tuple[int, int], np.ndarray] = {}
    session_means: dict[int, np.ndarray] = {}
    for ses in (0, 1):
        values = []
        for index in range(len(subjects)):
            mask = (sid == index) & (session == ses)
            if not np.any(mask):
                raise RuntimeError(f"Discovery subject lacks ses-{ses}: {subjects[index]}")
            centroids[(index, ses)] = h[mask].mean(axis=0)
            values.append(centroids[(index, ses)])
        session_means[ses] = np.mean(values, axis=0)
    a = np.stack([centroids[(index, 0)] - session_means[0] for index in range(len(subjects))])
    b = np.stack([centroids[(index, 1)] - session_means[1] for index in range(len(subjects))])
    cross = (a.T @ b + b.T @ a) / (2 * max(len(subjects) - 1, 1))
    eigenvalues, eigenvectors = np.linalg.eigh((cross + cross.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, basis = eigenvalues[order], eigenvectors[:, order].astype(np.float64)
    error = float(np.linalg.norm(basis[:, :32].T @ basis[:, :32] - np.eye(32), ord="fro"))
    if dim < 32 or error > 1e-8 or not np.isfinite(basis[:, :32]).all():
        raise RuntimeError(f"Invalid persistent basis dim={dim} error={error}")
    tolerance = max(abs(float(eigenvalues[0])), 1.0) * dim * np.finfo(float).eps
    positive_rank = int(np.sum(eigenvalues > tolerance))
    rank_for_angles = min(32, max(len(subjects) - 1, 1))
    _, _, va = np.linalg.svd(a, full_matrices=False)
    _, _, vb = np.linalg.svd(b, full_matrices=False)
    qa = va[:rank_for_angles].T
    qb = vb[:rank_for_angles].T
    singular_angles = np.linalg.svd(qa.T @ qb, compute_uv=False)
    principal_angles = np.degrees(np.arccos(np.clip(singular_angles, -1.0, 1.0)))
    diag = {
        "representation_dim": dim,
        "effective_positive_rank": positive_rank,
        "rank_tolerance": tolerance,
        "top32_orthogonality_error": error,
        "cross_covariance_condition_positive": (
            float(eigenvalues[0] / eigenvalues[positive_rank - 1]) if positive_rank else None
        ),
        "s1_s2_centroid_correlation": float(
            np.corrcoef((a @ basis[:, 0]), (b @ basis[:, 0]))[0, 1]
        ),
        "s1_s2_principal_angle_mean_deg": float(principal_angles.mean()),
        "s1_s2_principal_angle_max_deg": float(principal_angles.max()),
    }
    return basis[:, :32], eigenvalues[:32], h.mean(axis=0), session_means, diag


def persistence_values(
    arrays: Mapping[str, np.ndarray],
    basis: np.ndarray,
    random: Sequence[np.ndarray],
    session_means: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    subjects = arrays["subjects"].astype(str).tolist()
    h = arrays["embeddings"].astype(np.float64)
    sid, session = arrays["subject_index"].astype(int), arrays["session"].astype(int)
    centroids = {}
    for index in range(len(subjects)):
        for ses in (0, 1):
            mask = (sid == index) & (session == ses)
            centroids[(index, ses)] = h[mask].mean(axis=0) - session_means[ses]

    def advantage(vectors: Mapping[tuple[int, int], np.ndarray]) -> np.ndarray:
        values = np.empty(len(subjects), np.float64)
        for index in range(len(subjects)):
            anchor = vectors[(index, 0)]
            genuine = float(np.sum((anchor - vectors[(index, 1)]) ** 2))
            impostor = [
                float(np.sum((anchor - vectors[(other, 1)]) ** 2))
                for other in range(len(subjects))
                if other != index
            ]
            values[index] = float(np.mean(impostor) - genuine)
        return values

    keys = sorted(centroids)
    residual = np.stack([centroids[key] for key in keys])
    target = project_rows(residual, basis)
    candidate = advantage({key: target[index] for index, key in enumerate(keys)})
    controls = []
    for random_basis in random:
        delta = exact_matched_delta(residual, target, random_basis)
        controls.append(advantage({key: delta[index] for index, key in enumerate(keys)}))
    return candidate, np.stack(controls)


def local_binary_dependence(weight: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    """Exact Haar-null test for a binary linear margin and random rank-r span."""
    from scipy.stats import beta

    margin = np.asarray(weight[1] - weight[0], np.float64)
    dim, rank = len(margin), basis.shape[1]
    total = float(np.sum(margin * margin))
    captured = float(np.sum((margin @ basis) ** 2))
    fraction = captured / max(total, EPS)
    null_mean_local = total / dim
    candidate_local = captured / rank
    distribution = beta(rank / 2.0, (dim - rank) / 2.0)
    q025, q975 = map(float, distribution.ppf((0.025, 0.975)))
    return {
        "candidate": candidate_local,
        "random_mean": null_mean_local,
        "ratio": candidate_local / max(null_mean_local, EPS),
        "ratio_CI95_L": fraction / max(q975, EPS),
        "ratio_CI95_U": fraction / max(q025, EPS),
        "p_raw": float(distribution.sf(fraction)),
        "captured_margin_fraction": fraction,
    }


def residual_harmful(protected: np.ndarray, harmful: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Remove protected overlap before a conditional AGDI intervention."""
    protected = np.asarray(protected, np.float64)
    harmful = np.asarray(harmful, np.float64)
    dim = harmful.shape[0] if harmful.ndim == 2 else protected.shape[0]
    if harmful.size == 0:
        return np.empty((dim, 0), np.float64), {
            "protected_rank": int(protected.shape[1]) if protected.ndim == 2 else 0,
            "raw_harmful_rank": 0,
            "overlap_energy_fraction": 0.0,
            "residual_harmful_rank": 0,
            "residual_energy_fraction": 0.0,
        }
    if protected.size:
        qp, _ = np.linalg.qr(protected)
        qp = qp[:, : protected.shape[1]]
        raw = harmful - qp @ (qp.T @ harmful)
        overlap = harmful - raw
    else:
        raw, overlap = harmful.copy(), np.zeros_like(harmful)
    u, singular, _ = np.linalg.svd(raw, full_matrices=False)
    tolerance = max(float(singular[0]) if len(singular) else 0.0, 1.0) * 1e-10
    residual = u[:, singular > tolerance]
    energy = max(float(np.sum(harmful * harmful)), EPS)
    return residual, {
        "protected_rank": int(protected.shape[1]) if protected.ndim == 2 else 0,
        "raw_harmful_rank": int(harmful.shape[1]),
        "overlap_energy_fraction": float(np.sum(overlap * overlap) / energy),
        "residual_harmful_rank": int(residual.shape[1]),
        "residual_energy_fraction": float(np.sum(raw * raw) / energy),
    }


def agdi_projection(weight: np.ndarray, harmful_residual: np.ndarray, alpha: float) -> np.ndarray:
    basis = np.asarray(harmful_residual, np.float64)
    projector = basis @ basis.T if basis.size else np.zeros((weight.shape[1], weight.shape[1]))
    return np.asarray(weight, np.float64) @ (np.eye(weight.shape[1]) - float(alpha) * projector)
