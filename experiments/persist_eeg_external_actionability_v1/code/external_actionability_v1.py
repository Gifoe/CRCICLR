"""PERSIST-EEG prospective EEGMMIDB external actionability audit V1.

The command operates only on the 90-subject DEVELOPMENT_SCOPE_LOCK.  The
sealed outer split is never opened.  Confirmatory inference uses the 15
calibration subjects, with subject (not trial) as the statistical unit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_external_actionability_v1"
OUT = EXP_ROOT / "outputs"
PROTOCOL = OUT / "protocol"
RESULTS = OUT / "results"
CACHE = OUT / "cache"
SCOPE_PATH = PROTOCOL / "DEVELOPMENT_SCOPE_LOCK.json"
SEALED_SPLIT_PATH = PROTOCOL / "EXTERNAL_SPLIT_LOCK.json"
ACTION_LOCK_PATH = PROTOCOL / "ACTIONABILITY_PROTOCOL_LOCK.json"
REFERENCE_COMMIT = "1eca3976d62d38fb4291e217ca06add484babd41"
IMPLEMENTATION_ID = "persist_external_actionability_v1_20260817"
SEED = 20260817
CONTEXT_RUNS = (4, 6)
FUTURE_RUNS = (8, 10, 12, 14)
TARGET_RUNS = CONTEXT_RUNS + FUTURE_RUNS
BLOCKS = (("P01_04", 0, 4), ("P05_08", 4, 8), ("P09_16", 8, 16), ("P17_32", 16, 32))
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 10_000
EPS = 1e-12


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dependency-free sklearn-equivalent BA over classes present in y_true."""
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    values = [float(np.mean(prediction[truth == label] == label)) for label in np.unique(truth)]
    return float(np.mean(values)) if values else float("nan")


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
    temporary.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def require_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    required = [
        PROTOCOL / "EXTERNAL_AUDIT_PROTOCOL_LOCK.json",
        PROTOCOL / "EXTERNAL_DATASET_SELECTION_LOCK.json",
        ACTION_LOCK_PATH,
        SCOPE_PATH,
        PROTOCOL / "DATA_SCOPE_AUDIT.json",
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Required prospective/data-scope lock missing: {path}")
    # This program deliberately never reads SEALED_SPLIT_PATH.  All permitted
    # IDs are supplied by the outer-ID-free development lock.
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    action = json.loads(ACTION_LOCK_PATH.read_text(encoding="utf-8"))
    data_scope = json.loads((PROTOCOL / "DATA_SCOPE_AUDIT.json").read_text(encoding="utf-8"))
    if scope.get("outer_subject_ids_present") is not False or data_scope.get("status") != "DATA_SCOPE_PASS":
        raise RuntimeError("DATA_SCOPE_VIOLATION")
    if len(scope.get("allowed_subjects", [])) != 90:
        raise RuntimeError("DATA_SCOPE_VIOLATION: invalid allowed-subject count")
    if action.get("outer_test_state") != "OUTER_TEST_LOCKED":
        raise RuntimeError("DATA_SCOPE_VIOLATION: outer test is not locked")
    return scope, action


def role(scope: Mapping[str, Any], name: str) -> list[str]:
    values = [str(item) for item in scope["allowed_roles"][name]]
    if not set(values).issubset(set(map(str, scope["allowed_subjects"]))):
        raise RuntimeError(f"DATA_SCOPE_VIOLATION: role {name} escapes development scope")
    return sorted(values)


def feature_path(subject: str) -> Path:
    return CACHE / "eegmmidb_features" / f"{subject}.npz"


def embedding_path(subject: str) -> Path:
    return CACHE / "eegmmidb_embeddings" / f"{subject}.npz"


def load_feature_subject(subject: str) -> dict[str, np.ndarray]:
    path = feature_path(subject)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as item:
        if str(item["subject"].item()) != subject:
            raise RuntimeError(f"DATA_SCOPE_VIOLATION: cache identity mismatch {path}")
        return {key: np.asarray(item[key]).copy() for key in ("features", "labels", "runs", "trial_index")}


def load_embedding_subject(subject: str) -> dict[str, np.ndarray]:
    path = embedding_path(subject)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as item:
        if str(item["subject"].item()) != subject:
            raise RuntimeError(f"DATA_SCOPE_VIOLATION: embedding identity mismatch {path}")
        return {key: np.asarray(item[key]).copy() for key in ("embeddings", "logits", "labels", "runs", "trial_index")}


def concatenate_features(subjects: Sequence[str], runs: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wanted = set(map(int, runs))
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subject_index: list[np.ndarray] = []
    run_values: list[np.ndarray] = []
    for index, subject in enumerate(subjects):
        item = load_feature_subject(str(subject))
        mask = np.isin(item["runs"], sorted(wanted))
        features.append(item["features"][mask].astype(np.float32))
        labels.append(item["labels"][mask].astype(np.int64))
        subject_index.append(np.full(int(mask.sum()), index, dtype=np.int16))
        run_values.append(item["runs"][mask].astype(np.int16))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(subject_index), np.concatenate(run_values)


class FeatureNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.head = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return embedding, self.head(embedding)


def subject_ba(y: np.ndarray, pred: np.ndarray, subject_index: np.ndarray, n_subjects: int) -> np.ndarray:
    output = np.empty(n_subjects, dtype=np.float64)
    for index in range(n_subjects):
        mask = subject_index == index
        output[index] = balanced_accuracy_score(y[mask], pred[mask])
    return output


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_task_head(device: torch.device) -> dict[str, Any]:
    scope, _ = require_protocol()
    fit_subjects = role(scope, "task_head_fit")
    validation_subjects = role(scope, "task_head_validation")
    x_fit, y_fit, _, _ = concatenate_features(fit_subjects, CONTEXT_RUNS)
    x_validation, y_validation, validation_subject_index, _ = concatenate_features(validation_subjects, CONTEXT_RUNS)
    mean = x_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x_fit.std(axis=0, dtype=np.float64), 1e-5).astype(np.float32)
    x_fit = ((x_fit - mean) / std).astype(np.float32)
    x_validation = ((x_validation - mean) / std).astype(np.float32)
    x_fit_tensor, y_fit_tensor = torch.from_numpy(x_fit), torch.from_numpy(y_fit)
    x_validation_tensor = torch.from_numpy(x_validation).to(device)
    counts = np.bincount(y_fit, minlength=4).astype(np.float64)
    weights = counts.sum() / np.maximum(4 * counts, 1)
    grid = list(itertools.product((3e-4, 1e-3), (1e-4, 1e-3)))
    candidates: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for candidate, (learning_rate, weight_decay) in enumerate(grid):
        seed_all(stable_seed(IMPLEMENTATION_ID, "train", candidate))
        model = FeatureNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        generator = torch.Generator().manual_seed(stable_seed("batch-order", candidate))
        best_ba, best_loss, best_epoch, stale = -np.inf, np.inf, 0, 0
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(1, 81):
            model.train()
            order = torch.randperm(len(y_fit_tensor), generator=generator)
            batch_losses: list[float] = []
            for start in range(0, len(order), 256):
                index = order[start : start + 256]
                xb = x_fit_tensor[index].to(device)
                yb = y_fit_tensor[index].to(device)
                optimizer.zero_grad(set_to_none=True)
                _, logits = model(xb)
                loss = criterion(logits, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                batch_losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                _, validation_logits = model(x_validation_tensor)
                validation_probability = F.softmax(validation_logits, dim=1)
                validation_ce = float(F.cross_entropy(validation_logits, torch.from_numpy(y_validation).to(device)))
                prediction = validation_probability.argmax(1).cpu().numpy()
            per_subject = subject_ba(y_validation, prediction, validation_subject_index, len(validation_subjects))
            mean_ba = float(per_subject.mean())
            training_rows.append({
                "candidate": candidate, "learning_rate": learning_rate, "weight_decay": weight_decay,
                "epoch": epoch, "train_loss": float(np.mean(batch_losses)),
                "validation_subject_mean_BA": mean_ba, "validation_CE": validation_ce,
            })
            if (mean_ba > best_ba + 1e-12) or (abs(mean_ba - best_ba) <= 1e-12 and validation_ce < best_loss - 1e-12):
                best_ba, best_loss, best_epoch, stale = mean_ba, validation_ce, epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 10:
                    break
        if best_state is None:
            raise AssertionError("Task-head training produced no state")
        candidates.append({
            "candidate": candidate, "learning_rate": learning_rate, "weight_decay": weight_decay,
            "best_epoch": best_epoch, "validation_subject_mean_BA": best_ba,
            "validation_CE": best_loss, "state_dict": best_state,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    chosen = sorted(
        candidates,
        key=lambda row: (-row["validation_subject_mean_BA"], row["validation_CE"], row["learning_rate"], row["weight_decay"]),
    )[0]
    model = FeatureNet()
    model.load_state_dict(chosen["state_dict"])
    digest = state_hash(model)
    checkpoint = {
        "model_state": chosen["state_dict"], "feature_mean": mean, "feature_std": std,
        "fit_subjects_hash": hashlib.sha256(("\n".join(fit_subjects) + "\n").encode()).hexdigest(),
        "validation_subjects_hash": hashlib.sha256(("\n".join(validation_subjects) + "\n").encode()).hexdigest(),
        "selected": {key: value for key, value in chosen.items() if key != "state_dict"},
        "model_state_sha256": digest, "seed": SEED, "outer_test_used": False,
    }
    target = OUT / "model" / "eegmmidb_feature_net.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.part")
    torch.save(checkpoint, temporary)
    os.replace(temporary, target)
    write_frame(RESULTS / "TASK_HEAD_TRAINING_LOG.csv", pd.DataFrame(training_rows))
    report = {
        "status": "TASK_HEAD_FROZEN", "checkpoint": str(target.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(target), "model_state_sha256": digest,
        "selected": checkpoint["selected"],
        "all_candidates": [{key: value for key, value in row.items() if key != "state_dict"} for row in candidates],
        "fit_subject_count": len(fit_subjects), "validation_subject_count": len(validation_subjects),
        "training_runs": list(CONTEXT_RUNS), "outer_test_used": False,
    }
    write_json(RESULTS / "TASK_HEAD_RESULT.json", report)
    print(json.dumps(clean(report), indent=2))
    return report


def load_model(device: torch.device) -> tuple[FeatureNet, dict[str, Any]]:
    path = OUT / "model" / "eegmmidb_feature_net.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = FeatureNet()
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), payload


def extract_embeddings(device: torch.device) -> dict[str, Any]:
    scope, _ = require_protocol()
    subjects = sorted(role(scope, "block_discovery") + role(scope, "confirmatory_calibration"))
    model, payload = load_model(device)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    std = np.asarray(payload["feature_std"], dtype=np.float32)
    output = CACHE / "eegmmidb_embeddings"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        item = load_feature_subject(subject)
        x = torch.from_numpy(((item["features"] - mean) / std).astype(np.float32)).to(device)
        embeddings, logits = [], []
        with torch.inference_mode():
            for start in range(0, len(x), 512):
                h, z = model(x[start : start + 512])
                embeddings.append(h.cpu().numpy().astype(np.float32))
                logits.append(z.cpu().numpy().astype(np.float32))
        target = output / f"{subject}.npz"
        temporary = target.with_suffix(".npz.part")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, subject=np.asarray(subject), embeddings=np.concatenate(embeddings), logits=np.concatenate(logits),
                labels=item["labels"].astype(np.int16), runs=item["runs"].astype(np.int16),
                trial_index=item["trial_index"].astype(np.int16), model_state_sha256=np.asarray(payload["model_state_sha256"]),
            )
        os.replace(temporary, target)
        rows.append({"subject": subject, "n_trials": len(item["labels"]), "sha256": sha256(target)})
        print(f"[embedding] {subject} n={len(item['labels'])}", flush=True)
    expected = set(subjects)
    materialized = {path.stem for path in output.glob("S*.npz")}
    if materialized != expected:
        raise RuntimeError(f"DATA_SCOPE_VIOLATION: embedding cache scope mismatch {materialized ^ expected}")
    report = {
        "status": "FILTERED_EMBEDDINGS_COMPLETE", "subject_count": len(rows),
        "block_discovery_subjects": 30, "confirmatory_calibration_subjects": 15,
        "outer_embeddings_materialized": False, "model_state_sha256": payload["model_state_sha256"],
        "files": rows,
    }
    write_json(PROTOCOL / "EMBEDDING_SCOPE_AUDIT.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    return report


def subject_run_centroids(subjects: Sequence[str]) -> tuple[dict[tuple[str, int], np.ndarray], dict[int, np.ndarray]]:
    raw: dict[tuple[str, int], np.ndarray] = {}
    for subject in subjects:
        item = load_embedding_subject(subject)
        for run in TARGET_RUNS:
            mask = item["runs"] == run
            if not np.any(mask):
                raise RuntimeError(f"Missing run {run} for {subject}")
            raw[(subject, run)] = item["embeddings"][mask].mean(axis=0, dtype=np.float64)
    run_means = {
        run: np.mean([raw[(subject, run)] for subject in subjects], axis=0)
        for run in TARGET_RUNS
    }
    return raw, run_means


def discover_blocks() -> dict[str, Any]:
    scope, _ = require_protocol()
    subjects = role(scope, "block_discovery")
    centroids, run_means = subject_run_centroids(subjects)
    pairs = ((4, 8), (4, 12), (6, 10), (6, 14), (8, 12), (10, 14))
    cross = np.zeros((128, 128), dtype=np.float64)
    for left, right in pairs:
        a = np.stack([centroids[(subject, left)] - run_means[left] for subject in subjects])
        b = np.stack([centroids[(subject, right)] - run_means[right] for subject in subjects])
        cross += (a.T @ b + b.T @ a) / (2 * max(len(subjects) - 1, 1))
    cross /= len(pairs)
    eigenvalues, eigenvectors = np.linalg.eigh((cross + cross.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    basis = np.asarray(eigenvectors[:, :32], dtype=np.float32)
    orthogonality_error = float(np.linalg.norm(basis.T @ basis - np.eye(32), ord="fro"))
    if orthogonality_error > 1e-4 or not np.isfinite(basis).all():
        raise RuntimeError("Invalid discovered basis")
    global_center = np.mean(
        [centroids[(subject, run)] for subject in subjects for run in TARGET_RUNS], axis=0
    ).astype(np.float32)
    target = OUT / "model" / "eegmmidb_persistent_basis.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle, basis=basis, eigenvalues=eigenvalues.astype(np.float64), global_center=global_center,
            run_ids=np.asarray(TARGET_RUNS, dtype=np.int16),
            run_means=np.stack([run_means[run] for run in TARGET_RUNS]).astype(np.float32),
            discovery_subject_hash=np.asarray(hashlib.sha256(("\n".join(subjects) + "\n").encode()).hexdigest()),
        )
    os.replace(temporary, target)
    rows = []
    for name, start, end in BLOCKS:
        rows.append({
            "block": name, "start_component_1based": start + 1, "end_component_1based": end,
            "rank": end - start, "eigenvalue_sum": float(eigenvalues[start:end].sum()),
            "minimum_eigenvalue": float(eigenvalues[start:end].min()),
        })
    report = {
        "status": "PERSISTENT_BLOCKS_DISCOVERED_TRAIN_ONLY", "basis_path": str(target.relative_to(REPO_ROOT)).replace("\\", "/"),
        "basis_sha256": sha256(target), "discovery_subject_count": len(subjects),
        "run_pairs": [list(pair) for pair in pairs], "target_labels_used_for_centering": False,
        "orthogonality_error": orthogonality_error, "blocks": rows, "outer_test_used": False,
    }
    write_json(RESULTS / "BLOCK_DISCOVERY_RESULT.json", report)
    print(json.dumps(clean(report), indent=2))
    return report


def softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value = value - value.max(axis=-1, keepdims=True)
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


def random_bases(rank: int, block: str) -> list[np.ndarray]:
    result = []
    for draw in range(RANDOM_DRAWS):
        rng = np.random.default_rng(stable_seed(IMPLEMENTATION_ID, "random-basis", block, draw))
        q, _ = np.linalg.qr(rng.normal(size=(128, rank)))
        result.append(np.asarray(q[:, :rank], dtype=np.float64))
    return result


def exact_matched_delta(residual: np.ndarray, target_delta: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projection = residual @ basis @ basis.T
    target_norm = np.linalg.norm(target_delta, axis=1)
    random_norm = np.linalg.norm(projection, axis=1)
    output = projection.copy()
    bad = random_norm <= EPS
    if np.any(bad):
        output[bad] = basis[:, 0][None, :]
        random_norm[bad] = 1.0
    output *= (target_norm / np.maximum(random_norm, EPS))[:, None]
    error = np.max(np.abs(np.linalg.norm(output, axis=1) - target_norm))
    if error > 5e-6:
        raise RuntimeError(f"Matched-random displacement norm error {error}")
    return output


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
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
        return float(np.mean(distribution >= observed - 1e-15))
    if direction == "negative":
        return float(np.mean(distribution <= observed + 1e-15))
    raise ValueError(direction)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return adjusted


def calibration_arrays(subjects: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    embeddings, labels, runs, subject_index = [], [], [], []
    for index, subject in enumerate(subjects):
        item = load_embedding_subject(subject)
        embeddings.append(item["embeddings"].astype(np.float64))
        labels.append(item["labels"].astype(np.int64))
        runs.append(item["runs"].astype(np.int16))
        subject_index.append(np.full(len(item["labels"]), index, dtype=np.int16))
    return np.concatenate(embeddings), np.concatenate(labels), np.concatenate(runs), np.concatenate(subject_index), list(subjects)


def persistence_subject_values(
    embeddings: np.ndarray, runs: np.ndarray, subject_index: np.ndarray, n_subjects: int,
    run_means: Mapping[int, np.ndarray], basis: np.ndarray, random: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    centroids: dict[tuple[int, int], np.ndarray] = {}
    for subject in range(n_subjects):
        for run in TARGET_RUNS:
            mask = (subject_index == subject) & (runs == run)
            centroids[(subject, run)] = embeddings[mask].mean(axis=0) - run_means[run]
    pairs = ((4, 8), (4, 12), (6, 10), (6, 14))

    def advantage(vectors: Mapping[tuple[int, int], np.ndarray]) -> np.ndarray:
        output = np.zeros(n_subjects, dtype=np.float64)
        for subject in range(n_subjects):
            values = []
            for left, right in pairs:
                anchor = vectors[(subject, left)]
                genuine = float(np.sum((anchor - vectors[(subject, right)]) ** 2))
                impostors = [
                    float(np.sum((anchor - vectors[(other, right)]) ** 2))
                    for other in range(n_subjects) if other != subject
                ]
                values.append(float(np.mean(impostors) - genuine))
            output[subject] = float(np.mean(values))
        return output

    candidate_vectors = {key: value @ basis @ basis.T for key, value in centroids.items()}
    candidate = advantage(candidate_vectors)
    random_values = []
    keys = sorted(centroids)
    residual = np.stack([centroids[key] for key in keys])
    target = np.stack([candidate_vectors[key] for key in keys])
    for random_basis in random:
        delta = exact_matched_delta(residual, target, random_basis)
        random_values.append(advantage({key: delta[index] for index, key in enumerate(keys)}))
    return candidate, np.stack(random_values, axis=0)


def audit_blocks() -> dict[str, Any]:
    scope, _ = require_protocol()
    subjects = role(scope, "confirmatory_calibration")
    embeddings, labels, runs, subject_index, subjects = calibration_arrays(subjects)
    model, payload = load_model(torch.device("cpu"))
    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
    base_logits = embeddings @ weight.T + bias
    del model
    with np.load(OUT / "model" / "eegmmidb_persistent_basis.npz", allow_pickle=False) as item:
        basis_all = np.asarray(item["basis"], dtype=np.float64)
        center = np.asarray(item["global_center"], dtype=np.float64)
        run_ids = item["run_ids"].astype(int).tolist()
        run_means = {run: value.astype(np.float64) for run, value in zip(run_ids, item["run_means"])}
    future = np.isin(runs, FUTURE_RUNS)
    h = embeddings[future]
    y = labels[future]
    r = runs[future]
    sid = subject_index[future]
    z0 = base_logits[future]
    residual = h - center[None, :]
    base_ce = ce_rows(z0, y)
    base_probability = softmax(z0)
    base_prediction = z0.argmax(1)
    centered_weight = weight - weight.mean(axis=0, keepdims=True)
    persistence_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    actionability_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    p_raw = {"H1": {}, "H2": {}, "H3": {}, "H4": {}}
    block_cache: dict[str, dict[str, Any]] = {}
    for block_name, start, end in BLOCKS:
        block = basis_all[:, start:end]
        rank = end - start
        random = random_bases(rank, block_name)
        target_delta = residual @ block @ block.T
        candidate_logits = z0 - target_delta @ weight.T
        candidate_ce = ce_rows(candidate_logits, y)
        candidate_probability = softmax(candidate_logits)
        candidate_prediction = candidate_logits.argmax(1)
        random_logits: list[np.ndarray] = []
        random_ce: list[np.ndarray] = []
        random_delta: list[np.ndarray] = []
        for draw, random_basis in enumerate(random):
            delta = exact_matched_delta(residual, target_delta, random_basis)
            logits = z0 - delta @ weight.T
            random_delta.append(delta)
            random_logits.append(logits)
            random_ce.append(ce_rows(logits, y))
        random_ce_array = np.stack(random_ce)
        candidate_persistence, random_persistence = persistence_subject_values(
            embeddings, runs, subject_index, len(subjects), run_means, block, random,
        )
        persistence_specific = candidate_persistence - random_persistence.mean(axis=0)
        persistence_mean, persistence_lcb, persistence_ucb = bootstrap_mean(
            persistence_specific, stable_seed("bootstrap", block_name, "H1")
        )
        p_raw["H1"][block_name] = signflip_p(persistence_specific, "positive")

        per_subject_u = np.empty(len(subjects), dtype=np.float64)
        per_subject_u_abs = np.empty(len(subjects), dtype=np.float64)
        per_subject_finite_ratio = np.empty(len(subjects), dtype=np.float64)
        per_subject_ba_specific = np.empty(len(subjects), dtype=np.float64)
        per_subject_ba_delta = np.empty(len(subjects), dtype=np.float64)
        per_subject_random_ba_delta = np.empty(len(subjects), dtype=np.float64)
        run_specific: dict[tuple[int, int], float] = {}
        for subject_index_value, subject in enumerate(subjects):
            mask = sid == subject_index_value
            base_ba = balanced_accuracy_score(y[mask], base_prediction[mask])
            candidate_ba = balanced_accuracy_score(y[mask], candidate_prediction[mask])
            random_ba_values = [
                balanced_accuracy_score(y[mask], logits.argmax(1)[mask]) for logits in random_logits
            ]
            u_abs = float(np.mean(candidate_ce[mask] - base_ce[mask]))
            u_random = float(np.mean(random_ce_array[:, mask] - base_ce[None, mask]))
            per_subject_u_abs[subject_index_value] = u_abs
            per_subject_u[subject_index_value] = u_abs - u_random
            candidate_finite = centered_rms(candidate_logits[mask] - z0[mask])
            random_finite = float(np.mean([centered_rms(logits[mask] - z0[mask]) for logits in random_logits]))
            per_subject_finite_ratio[subject_index_value] = candidate_finite / max(random_finite, EPS)
            per_subject_ba_delta[subject_index_value] = candidate_ba - base_ba
            per_subject_random_ba_delta[subject_index_value] = float(np.mean(random_ba_values) - base_ba)
            per_subject_ba_specific[subject_index_value] = per_subject_ba_delta[subject_index_value] - per_subject_random_ba_delta[subject_index_value]
            for run in FUTURE_RUNS:
                run_mask = mask & (r == run)
                base_run_ba = balanced_accuracy_score(y[run_mask], base_prediction[run_mask])
                candidate_run_ba = balanced_accuracy_score(y[run_mask], candidate_prediction[run_mask])
                random_run_ba = float(np.mean([
                    balanced_accuracy_score(y[run_mask], logits.argmax(1)[run_mask]) for logits in random_logits
                ]))
                run_specific[(subject_index_value, run)] = (candidate_run_ba - base_run_ba) - (random_run_ba - base_run_ba)
            subject_rows.append({
                "block": block_name, "subject": subject, "n_future_trials": int(mask.sum()),
                "persistence_advantage": float(candidate_persistence[subject_index_value]),
                "persistence_random": float(random_persistence[:, subject_index_value].mean()),
                "persistence_specific": float(persistence_specific[subject_index_value]),
                "u_abs": u_abs, "u_random": u_random, "u_spec": float(per_subject_u[subject_index_value]),
                "finite_ratio": float(per_subject_finite_ratio[subject_index_value]),
                "base_BA": float(base_ba), "candidate_BA": float(candidate_ba),
                "random_BA_mean": float(np.mean(random_ba_values)),
                "delta_BA": float(per_subject_ba_delta[subject_index_value]),
                "delta_BA_random": float(per_subject_random_ba_delta[subject_index_value]),
                "delta_BA_specific": float(per_subject_ba_specific[subject_index_value]),
            })
        for draw, logits in enumerate(random_logits):
            for subject_index_value, subject in enumerate(subjects):
                mask = sid == subject_index_value
                random_rows.append({
                    "block": block_name, "draw": draw, "subject": subject,
                    "u_random": float(np.mean(random_ce_array[draw, mask] - base_ce[mask])),
                    "finite_logit_rms": centered_rms(logits[mask] - z0[mask]),
                    "delta_BA_random": float(
                        balanced_accuracy_score(y[mask], logits.argmax(1)[mask])
                        - balanced_accuracy_score(y[mask], base_prediction[mask])
                    ),
                    "persistence_random": float(random_persistence[draw, subject_index_value]),
                })

        u_mean, u_lcb, u_ucb = bootstrap_mean(per_subject_u, stable_seed("bootstrap", block_name, "H2"))
        u_abs_mean, u_abs_lcb, u_abs_ucb = bootstrap_mean(per_subject_u_abs, stable_seed("bootstrap", block_name, "u_abs"))
        finite_mean, finite_lcb, finite_ucb = bootstrap_mean(per_subject_finite_ratio, stable_seed("bootstrap", block_name, "H3"))
        ba_mean, ba_lcb, ba_ucb = bootstrap_mean(per_subject_ba_specific, stable_seed("bootstrap", block_name, "H4"))
        p_raw["H2"][block_name] = signflip_p(per_subject_u, "negative")
        p_raw["H3"][block_name] = signflip_p(np.log(np.maximum(per_subject_finite_ratio, EPS)), "positive")
        p_raw["H4"][block_name] = signflip_p(per_subject_ba_specific, "positive")
        candidate_local = float(np.sum((centered_weight @ block) ** 2) / rank)
        random_local_values = np.asarray([
            np.sum((centered_weight @ random_basis) ** 2) / rank for random_basis in random
        ], dtype=np.float64)
        local_ratios = candidate_local / np.maximum(random_local_values, EPS)
        local_ratio_mean = float(candidate_local / max(random_local_values.mean(), EPS))
        local_ratio_lcb, local_ratio_ucb = map(float, np.quantile(local_ratios, [0.025, 0.975]))
        local_p = float((1 + np.sum(random_local_values >= candidate_local)) / (1 + len(random_local_values)))
        loso = [float(np.delete(per_subject_ba_specific, index).mean()) for index in range(len(subjects))]
        leave_run_out = []
        for held_run in FUTURE_RUNS:
            values = [
                float(np.mean([run_specific[(subject_index_value, run)] for run in FUTURE_RUNS if run != held_run]))
                for subject_index_value in range(len(subjects))
            ]
            leave_run_out.append(float(np.mean(values)))
        positive_fraction = float(np.mean(per_subject_ba_specific >= 0))
        stability = bool(min(loso) > 0 and min(leave_run_out) > 0 and positive_fraction >= 0.60)
        persistence_rows.append({
            "block": block_name, "rank": rank, "mean_specific_advantage": persistence_mean,
            "CI95_L": persistence_lcb, "CI95_U": persistence_ucb, "p_raw": p_raw["H1"][block_name],
        })
        utility_rows.append({
            "block": block_name, "rank": rank, "u_abs_mean": u_abs_mean,
            "u_abs_CI95_L": u_abs_lcb, "u_abs_CI95_U": u_abs_ucb,
            "u_spec_mean": u_mean, "u_spec_CI95_L": u_lcb, "u_spec_CI95_U": u_ucb,
            "p_raw_harmful": p_raw["H2"][block_name],
        })
        decision_rows.append({
            "block": block_name, "rank": rank, "local_energy": candidate_local,
            "local_random_mean": float(random_local_values.mean()), "local_ratio": local_ratio_mean,
            "local_ratio_CI95_L": local_ratio_lcb, "local_ratio_CI95_U": local_ratio_ucb,
            "local_randomization_p": local_p, "finite_ratio_mean": finite_mean,
            "finite_ratio_CI95_L": finite_lcb, "finite_ratio_CI95_U": finite_ucb,
            "finite_p_raw": p_raw["H3"][block_name],
            "candidate_logit_rms": centered_rms(candidate_logits - z0),
            "candidate_margin_displacement": float(np.mean(np.abs(true_margin(candidate_logits, y) - true_margin(z0, y)))),
            "candidate_flip_rate": float(np.mean(candidate_prediction != base_prediction)),
            "candidate_total_variation": float(np.mean(0.5 * np.sum(np.abs(candidate_probability - base_probability), axis=1))),
        })
        actionability_rows.append({
            "block": block_name, "rank": rank, "delta_BA_mean": float(per_subject_ba_delta.mean()),
            "delta_BA_random_mean": float(per_subject_random_ba_delta.mean()),
            "delta_BA_specific_mean": ba_mean, "delta_BA_specific_CI95_L": ba_lcb,
            "delta_BA_specific_CI95_U": ba_ucb, "p_raw": p_raw["H4"][block_name],
            "minimum_LOSO_mean": min(loso), "minimum_leave_one_run_out_mean": min(leave_run_out),
            "nonnegative_subject_fraction": positive_fraction, "stability_preliminary": stability,
        })
        block_cache[block_name] = {
            "u": (u_mean, u_lcb, u_ucb), "finite": (finite_mean, finite_lcb, finite_ucb),
            "ba": (ba_mean, ba_lcb, ba_ucb), "persistence": (persistence_mean, persistence_lcb, persistence_ucb),
            "local": (local_ratio_mean, local_ratio_lcb, local_ratio_ucb, local_p),
            "stability": stability, "positive_fraction": positive_fraction,
            "minimum_loso": min(loso), "minimum_leave_run_out": min(leave_run_out),
        }
        print(f"[audit] {block_name} u_spec={u_mean:.6f} ba_specific={ba_mean:.6f}", flush=True)

    p_adjusted = {family: holm(values) for family, values in p_raw.items()}
    for row in persistence_rows:
        row["p_holm"] = p_adjusted["H1"][row["block"]]
    for row in utility_rows:
        row["p_holm_harmful"] = p_adjusted["H2"][row["block"]]
    for row in decision_rows:
        row["finite_p_holm"] = p_adjusted["H3"][row["block"]]
    for row in actionability_rows:
        row["p_holm"] = p_adjusted["H4"][row["block"]]
    assignments = []
    for block_name, start, end in BLOCKS:
        value = block_cache[block_name]
        h1 = bool(value["persistence"][1] > 0 and p_adjusted["H1"][block_name] < 0.05)
        h2 = bool(value["u"][2] < 0 and p_adjusted["H2"][block_name] < 0.05)
        h3 = bool(
            value["local"][1] > 1 and value["local"][3] < 0.05
            and value["finite"][1] > 1 and p_adjusted["H3"][block_name] < 0.05
        )
        h4 = bool(value["ba"][1] > 0 and value["ba"][0] >= 0.005 and p_adjusted["H4"][block_name] < 0.05)
        h5 = bool(value["stability"])
        actionable = bool(h1 and h2 and h3 and h4 and h5)
        if actionable:
            assignment, action = "ACTIONABLE-HARMFUL", "SUPPRESS_CANDIDATE"
        elif h1 and h3 and value["u"][1] > 0:
            assignment, action = "PROTECTED", "PRESERVE"
        elif h1 and not h3:
            assignment, action = "DECISION-NULL / WEAKLY ACTIVE", "NO_OP"
        elif h1 and h3:
            assignment, action = "DECISION-ACTIVE BUT NON-ACTIONABLE", "NO_OP"
        else:
            assignment, action = "UNCERTAIN", "NO_OP"
        assignments.append({
            "block": block_name, "rank": end - start, "H1": h1, "H2": h2, "H3": h3,
            "H4": h4, "H5": h5, "all_H1_H5": actionable,
            "assignment": assignment, "action": action,
        })
    write_frame(RESULTS / "PERSISTENCE_RESULTS.csv", pd.DataFrame(persistence_rows))
    write_frame(RESULTS / "SIGNED_UTILITY_RESULTS.csv", pd.DataFrame(utility_rows))
    write_frame(RESULTS / "DECISION_DEPENDENCE_RESULTS.csv", pd.DataFrame(decision_rows))
    write_frame(RESULTS / "ACTIONABILITY_RESULTS.csv", pd.DataFrame(actionability_rows))
    write_frame(RESULTS / "BLOCK_ASSIGNMENTS.csv", pd.DataFrame(assignments))
    write_frame(RESULTS / "EXTERNAL_AUDIT_SUBJECT.csv", pd.DataFrame(subject_rows))
    write_frame(RESULTS / "EXTERNAL_AUDIT_RANDOM_SUBJECT.csv", pd.DataFrame(random_rows))
    found = [row["block"] for row in assignments if row["all_H1_H5"]]
    final = {
        "terminal_state": "EXTERNAL_AUDIT_ACTIONABLE_HARMFUL_FOUND" if found else "EXTERNAL_AUDIT_NO_ACTIONABLE_HARMFUL",
        "scientific_conclusion": "NO_REAL_ACTIONABLE_HARMFUL_CASE" if not found else "REAL_ACTIONABLE_HARMFUL_CASE_FOUND",
        "actionable_harmful_blocks": found,
        "agdi_training_authorized": bool(found),
        "next_action": "AGDI_TRAINING_AUTHORIZED" if found else "STOP_AGDI_NO_ACTIONABLE_TARGET",
        "outer_test_state": "OUTER_TEST_LOCKED", "outer_test_used": False,
        "dataset_scope": "EEGMMIDB repeated-run MI; not multisession/site/device",
        "confirmatory_subjects": len(subjects), "random_draws": RANDOM_DRAWS,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "multiplicity": "Holm within gate family across four blocks",
        "limitations": [
            "the selected task head is weak: validation context subject-mean BA is 0.346 and confirmatory future-run baseline BA is about 0.286 versus 0.25 chance",
            "confirmatory inference uses 15 subjects and therefore has wide confidence intervals",
            "EEGMMIDB supplies repeated runs, not independent sessions, sites, or devices",
            "official records include sampling-rate and trial-count anomalies handled prospectively by the recorded data amendment",
            "failure to detect an actionable block under this audit is not proof that no such block exists under a stronger frozen representation",
        ],
        "assignments": assignments,
    }
    write_json(OUT / "FINAL_DECISION.json", final)
    print(json.dumps(clean(final), indent=2))
    return final


def report() -> dict[str, Any]:
    final = json.loads((OUT / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    persistence = pd.read_csv(RESULTS / "PERSISTENCE_RESULTS.csv")
    utility = pd.read_csv(RESULTS / "SIGNED_UTILITY_RESULTS.csv")
    decision = pd.read_csv(RESULTS / "DECISION_DEPENDENCE_RESULTS.csv")
    action = pd.read_csv(RESULTS / "ACTIONABILITY_RESULTS.csv")
    assignments = pd.read_csv(RESULTS / "BLOCK_ASSIGNMENTS.csv")
    subject_frame = pd.read_csv(RESULTS / "EXTERNAL_AUDIT_SUBJECT.csv")
    baseline_future_ba = float(subject_frame.groupby("subject", as_index=False).base_BA.first().base_BA.mean())
    task_result = json.loads((RESULTS / "TASK_HEAD_RESULT.json").read_text(encoding="utf-8"))
    validation_ba = float(task_result["selected"]["validation_subject_mean_BA"])
    merged = persistence.merge(utility, on=["block", "rank"]).merge(decision, on=["block", "rank"]).merge(action, on=["block", "rank"])
    merged = merged.merge(assignments, on=["block", "rank"])
    lines = [
        "# PERSIST-EEG External Actionability Audit V1", "",
        f"Terminal state: `{final['terminal_state']}`", "",
        f"Next action: `{final['next_action']}`", "",
        "## Scope", "",
        "The prospective external dataset is PhysioNet EEGMMIDB v1.0.0: 109 subjects, 64 channels, 160 Hz, and six motor-imagery runs. The audit used 45 task-head subjects, 30 block-discovery subjects, and 15 confirmatory calibration subjects. The 19-subject outer test remained locked and was not materialized.", "",
        "This is a repeated-run motor-imagery replication. It is not evidence for true multisession, multisite, or multidevice persistence.", "",
        f"The frozen task head is weak: subject-mean validation BA on context runs was {validation_ba:.3f}, and mean baseline BA on confirmatory future runs was {baseline_future_ba:.3f} (four-class chance = 0.250). This materially limits the strength of a negative actionability conclusion.", "",
        "## Frozen DDA interpretation", "",
        "DDA-A remains permanently failed and its behavioral-null explanation remains falsified. DDA-B and DDA-C authorized this prospective external audit; they did not authorize AGDI by themselves.", "",
        "## Confirmatory block results", "",
        "| Block | H1 | H2 | H3 | H4 | H5 | u_spec | finite ratio | BA specific | Assignment |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in merged.iterrows():
        lines.append(
            f"| {row.block} | {bool(row.H1)} | {bool(row.H2)} | {bool(row.H3)} | {bool(row.H4)} | {bool(row.H5)} | "
            f"{row.u_spec_mean:.6f} [{row.u_spec_CI95_L:.6f}, {row.u_spec_CI95_U:.6f}] | "
            f"{row.finite_ratio_mean:.3f} [{row.finite_ratio_CI95_L:.3f}, {row.finite_ratio_CI95_U:.3f}] | "
            f"{row.delta_BA_specific_mean:.4f} [{row.delta_BA_specific_CI95_L:.4f}, {row.delta_BA_specific_CI95_U:.4f}] | {row.assignment} |"
        )
    cf_path = RESULTS / "CF_RESCUE_HARM_RESULT.json"
    if cf_path.exists():
        cf = json.loads(cf_path.read_text(encoding="utf-8"))
        summary = cf["summaries"]
        lines += [
            "", "## Supporting PERSIST-CF rescue/harm decomposition", "",
            f"Frozen interpretation: `{cf['case']}`. This analysis is not an authorization gate and does not change `DDA_A_FAIL`.", "",
            f"CF rescue rate was {summary['rescue_rate']['mean']:.4f} and harm rate was {summary['harm_rate']['mean']:.4f}; net rescue was {summary['net_rescue']['mean']:.4f} "
            f"(95% cluster-bootstrap CI [{summary['net_rescue']['CI95'][0]:.4f}, {summary['net_rescue']['CI95'][1]:.4f}]).",
            "",
            f"Relative to exact matched-random offsets, net rescue changed by {summary['specific_net_rescue']['mean']:.4f} "
            f"(95% CI [{summary['specific_net_rescue']['CI95'][0]:.4f}, {summary['specific_net_rescue']['CI95'][1]:.4f}]).",
        ]
    if final["agdi_training_authorized"]:
        conclusion = "At least one real block passed H1-H5. AGDI is authorized, but outer test remains locked."
    else:
        conclusion = "No real block passed all H1-H5. AGDI is not authorized; constructive model search stops at this falsifiable negative boundary."
    lines += [
        "", "## Limitations", "",
        "The confirmatory sample has 15 subjects, giving wide intervals. EEGMMIDB has repeated runs rather than independent sessions/sites/devices. Official sampling-rate and trial-count anomalies were retained under the recorded pre-outcome data amendment. Most importantly, the task head is only modestly above chance. Therefore the terminal state means that this frozen audit did not identify an actionable target; it does not establish that no actionable persistent structure could exist under a stronger frozen representation.",
        "", "## Conclusion", "", conclusion, "",
        "All confidence intervals and sign-flip tests use subjects as the inference unit. Holm correction was applied separately within each gate family across the four pre-registered blocks.", ""
    ]
    (OUT / "scientific_report.md").write_text("\n".join(lines), encoding="utf-8")
    readme = """# PERSIST-EEG External Actionability V1

Prospective EEGMMIDB repeated-run audit following frozen DDA-V1.  Run order:

1. `freeze_protocol.py`
2. `extract_features.py inventory`
3. `extract_features.py extract`
4. `external_actionability_v1.py train`
5. `external_actionability_v1.py embed`
6. `external_actionability_v1.py discover`
7. `external_actionability_v1.py audit`
8. `external_actionability_v1.py report`

The sealed outer split is not opened by the extractor or audit program.
"""
    (EXP_ROOT / "README.md").write_text(readme, encoding="utf-8")
    result = {"status": "REPORT_COMPLETE", "terminal_state": final["terminal_state"], "outer_test_used": False}
    write_json(OUT / "REPORT_STATUS.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "embed", "discover", "audit", "report"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    started = time.time()
    if args.command == "train":
        train_task_head(device)
    elif args.command == "embed":
        extract_embeddings(device)
    elif args.command == "discover":
        discover_blocks()
    elif args.command == "audit":
        audit_blocks()
    else:
        report()
    print(f"completed command={args.command} elapsed_seconds={time.time()-started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
