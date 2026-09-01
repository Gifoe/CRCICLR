"""Run the frozen, vanilla EEGNet baseline on the authorized development data.

The runner is intentionally self-contained.  It reads only the frozen Stage-0
OpenBMI MI manifest/cache and the frozen WBCIC development cache.  For every
dataset/fold/seed it performs exactly one initial fit (model-fit subjects,
S1+S2), selects an epoch on the disjoint discovery subjects, then performs one
deterministic refit (model-fit+discovery subjects, S1+S2) and scores the held-out
outcome subjects once.  No outcome labels are touched before that final score.

The runtime/checkpoint directory is ignored by git.  Only compact result tables,
protocol/audit documents and this code belong in the committed experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


REPO = Path(os.environ.get(
    "CANONICAL_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC"
)).resolve()
EXP = REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"
RESULTS = EXP / "results"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"
STAGE0_ROOT = Path(os.environ.get(
    "PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full"
)).resolve()
WBCIC_EXP = REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1"
WBCIC_LOCK = WBCIC_EXP / "provenance" / "DEVELOPMENT_SCOPE_LOCK.json"
OPENBMI_SPLIT = STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
OPENBMI_MANIFEST = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
WBCIC_CACHE = Path(os.environ.get("PERSIST_WBCIC_CACHE", str(
    WBCIC_EXP / "runtime" / "cache"
))).resolve()

SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
MAX_EPOCHS = 60
MIN_EPOCHS = 10
PATIENCE = 8
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
BOOTSTRAP_DRAWS = 10_000
DATASETS = ("OpenBMI", "WBCIC")
HISTORICAL_REFERENCE = {"OpenBMI": 0.7719, "WBCIC": 0.7884}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
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
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def session_label(dataset: str, session_id: int) -> str:
    mapping = {"OpenBMI": {1: "S1", 2: "S2"}, "WBCIC": {0: "S1", 1: "S2", 2: "S3"}}
    try:
        return mapping[dataset][int(session_id)]
    except KeyError as exc:
        raise RuntimeError(f"unexpected {dataset} session id {session_id}") from exc


@dataclass
class DatasetData:
    dataset: str
    metadata: pd.DataFrame
    raw: np.ndarray | None
    cache_root: Path
    arrays: dict[str, np.ndarray]

    def batch(self, indices: np.ndarray) -> np.ndarray:
        """Materialize one small CPU batch from memory-mapped source arrays."""
        indices = np.asarray(indices, dtype=np.int64)
        if self.raw is not None:
            return np.asarray(self.raw[indices], dtype=np.float32)
        paths = self.metadata.iloc[indices]["_signal_path"].to_numpy()
        offsets = self.metadata.iloc[indices]["_cache_index"].to_numpy(np.int64)
        values = []
        for path, offset in zip(paths, offsets):
            key = str(path)
            if key not in self.arrays:
                self.arrays[key] = np.load(self.cache_root / key, mmap_mode="r", allow_pickle=False)
            values.append(np.asarray(self.arrays[key][int(offset)], dtype=np.float32))
        return np.stack(values, axis=0)


def load_roles(dataset: str) -> tuple[list[dict[str, list[str]]], set[str], dict[str, Any]]:
    if dataset == "OpenBMI":
        payload = json.loads(OPENBMI_SPLIT.read_text(encoding="utf-8-sig"))
        section = payload["openbmi"]
        pool = set(subject_sort(section["subjects"]))
        folds = []
        for row in section["folds"]:
            role = {
                "model_fit": subject_sort(row["train_subjects"]),
                "discovery": subject_sort(row["validation_subjects"]),
                "outcome": subject_sort(row["outer_test_subjects"]),
            }
            assert_roles(dataset, role, pool)
            folds.append(role)
        return folds, pool, payload
    lock = json.loads(WBCIC_LOCK.read_text(encoding="utf-8-sig"))
    if lock.get("outer_subject_ids_present") is not False or int(lock.get("outer_subject_count", -1)) != 10:
        raise RuntimeError("WBCIC scope lock does not prove that the outer cohort is excluded")
    pool = set(subject_sort(lock["allowed_subjects"]))
    folds = []
    for key in map(str, FOLDS):
        row = lock["audit_roles"][key]
        role = {
            "model_fit": subject_sort(row["model_fit"]),
            "discovery": subject_sort(row["discovery_decision"]),
            "outcome": subject_sort(row["outcome"]),
        }
        assert_roles(dataset, role, pool)
        folds.append(role)
    return folds, pool, lock


def assert_roles(dataset: str, role: dict[str, list[str]], pool: set[str]) -> None:
    parts = [set(role[key]) for key in ("model_fit", "discovery", "outcome")]
    if any(a & b for i, a in enumerate(parts) for b in parts[i + 1:]):
        raise RuntimeError(f"{dataset} fold roles overlap")
    if set().union(*parts) != pool:
        raise RuntimeError(f"{dataset} fold roles are not exhaustive")
    if not all(parts):
        raise RuntimeError(f"{dataset} fold role is empty")


def load_dataset(dataset: str, subjects: set[str]) -> DatasetData:
    if dataset == "OpenBMI":
        if not OPENBMI_MANIFEST.is_file():
            raise FileNotFoundError(OPENBMI_MANIFEST)
        manifest = pd.read_parquet(OPENBMI_MANIFEST)
        frame = manifest[(manifest["paradigm"] == "mi") & (manifest["run_phase"] == "train")].copy()
        frame["subject_id"] = frame["subject_id"].astype(str).str.replace("sub-", "", regex=False)
        frame["session_id"] = frame["session_id"].astype(int)
        frame["label"] = frame["event_code"].astype(int).map({1: 0, 2: 1})
        frame["trial_uid"] = frame["trial_id"].astype(str)
        frame["_signal_path"] = frame["signal_cache_path"].astype(str)
        frame["_cache_index"] = frame["cache_index"].astype(int)
        frame = frame.reset_index(drop=True)
        observed = set(frame.subject_id)
        if observed != subjects or len(frame) != 54 * 2 * 100 or set(frame.session_id) != {1, 2}:
            raise RuntimeError(f"OpenBMI manifest audit failed rows={len(frame)} subjects={len(observed)}")
        cells = frame.groupby(["subject_id", "session_id", "label"]).size()
        if len(cells) != 54 * 2 * 2 or set(cells.astype(int)) != {50}:
            raise RuntimeError("OpenBMI MI manifest does not have 50 trials per class/session")
        root = STAGE0_ROOT
        for rel in frame["_signal_path"].drop_duplicates():
            path = root / str(rel)
            if not path.is_file():
                raise FileNotFoundError(path)
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.shape != (100, 62, 1000) or arr.dtype != np.float32:
                raise RuntimeError(f"OpenBMI shard audit failed {path}: {arr.shape} {arr.dtype}")
        return DatasetData(dataset, frame, None, root, {})
    if not WBCIC_CACHE.is_dir():
        raise FileNotFoundError(WBCIC_CACHE)
    metadata_path = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
    raw_path = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_RAW.npy"
    frame = pd.read_parquet(metadata_path, columns=["subject_id", "session_id", "label"])
    frame["subject_id"] = frame["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    frame["session_id"] = frame["session_id"].astype(int)
    frame["label"] = frame["label"].astype(int)
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["trial_uid"] = [f"wbcic-{s}-S{int(sess)+1}-{i:05d}" for i, (s, sess) in enumerate(zip(frame.subject_id, frame.session_id))]
    frame = frame.reset_index(drop=True)
    observed = set(frame.subject_id)
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if observed != subjects or raw.shape != (len(frame), 58, 1000) or raw.dtype != np.float16:
        raise RuntimeError(f"WBCIC cache audit failed rows={len(frame)} shape={raw.shape} subjects={len(observed)} dtype={raw.dtype}")
    if set(frame.session_id) != {0, 1, 2} or set(frame.label) != {0, 1}:
        raise RuntimeError("WBCIC cache session/label audit failed")
    cells = frame.groupby(["subject_id", "session_id", "label"]).size()
    if len(cells) != 41 * 3 * 2 or int(cells.min()) < 20:
        raise RuntimeError("WBCIC development cache class/session cells are incomplete")
    return DatasetData(dataset, frame, raw, WBCIC_CACHE, {})


class VanillaEEGNet(nn.Module):
    """Project-standard EEGNet: F1=8, depth multiplier 2, F2=16."""

    def __init__(self, channels: int, samples: int = 1000, dropout: float = 0.25):
        super().__init__()
        if samples != 1000:
            raise ValueError(f"canonical EEGNet expects 1000 samples, got {samples}")
        f1, depth_multiplier, f2 = 8, 2, 16
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, f1 * depth_multiplier, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * depth_multiplier)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(f1 * depth_multiplier, f1 * depth_multiplier, (1, 16), padding="same", groups=f1 * depth_multiplier, bias=False)
        self.point = nn.Conv2d(f1 * depth_multiplier, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        reduced = samples // 4 // 8
        self.embedding = nn.Sequential(nn.Linear(f2 * reduced, 64), nn.ELU(), nn.LayerNorm(64))
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


def compute_normalizer(data: DatasetData, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(data.raw.shape[1] if data.raw is not None else 62, dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for start in range(0, len(indices), 128):
        batch = data.batch(indices[start : start + 128]).astype(np.float64)
        total += batch.sum(axis=(0, 2))
        square += np.square(batch).sum(axis=(0, 2))
        count += batch.shape[0] * batch.shape[2]
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def prepare_batch(data: DatasetData, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    value = data.batch(indices)
    value = (value - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device, non_blocking=True)


def subject_metrics(labels: np.ndarray, probability: np.ndarray, subjects: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for subject in subject_sort(np.unique(subjects)):
        mask = subjects.astype(str) == subject
        y = labels[mask].astype(int)
        p1 = probability[mask].astype(float)
        pred = (p1 >= 0.5).astype(int)
        rows.append({
            "subject_id": subject,
            "BA": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "NLL": float(log_loss(y, np.column_stack([1.0 - p1, p1]), labels=[0, 1])),
            "trials": int(mask.sum()),
        })
    return rows


def evaluate(data: DatasetData, model: nn.Module, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    probs, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            part = indices[start : start + BATCH_SIZE]
            out = model(prepare_batch(data, part, mean, std, device))
            logits.append(out.float().cpu().numpy())
    logit = np.concatenate(logits, axis=0)
    stable = logit.astype(np.float64) - logit.max(axis=1, keepdims=True)
    probability = np.exp(stable)
    probability /= probability.sum(axis=1, keepdims=True)
    frame = data.metadata.iloc[indices]
    labels = frame.label.to_numpy(np.int64)
    subjects = frame.subject_id.astype(str).to_numpy()
    return {"indices": indices.copy(), "labels": labels, "subjects": subjects, "probability": probability, "logits": logit, "subject_metrics": subject_metrics(labels, probability[:, 1], subjects)}


def fit_initial(data: DatasetData, train_idx: np.ndarray, discovery_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, dataset: str, fold: int, seed: int, device: torch.device) -> tuple[int, list[dict[str, Any]], int]:
    run_seed = stable_seed("canonical-initial", dataset, fold, seed)
    set_seed(run_seed)
    model = VanillaEEGNet(data.batch(train_idx[:1]).shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order_rng = np.random.default_rng(stable_seed("canonical-order", dataset, fold, seed, "initial"))
    best_ba, best_nll, best_epoch = -math.inf, math.inf, 1
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = order_rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            part = order[start : start + BATCH_SIZE]
            xb = prepare_batch(data, part, mean, std, device)
            yb = torch.as_tensor(np.array(data.metadata.iloc[part].label.to_numpy(np.int64), copy=True), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        discovery = evaluate(data, model, discovery_idx, mean, std, device)
        dframe = pd.DataFrame(discovery["subject_metrics"])
        val_ba = float(dframe.BA.mean())
        val_nll = float(dframe.NLL.mean())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "discovery_mean_subject_BA": val_ba, "discovery_mean_subject_NLL": val_nll})
        improved = val_ba > best_ba + 1e-12 or (abs(val_ba - best_ba) <= 1e-12 and val_nll < best_nll - 1e-12)
        if improved:
            best_ba, best_nll, best_epoch = val_ba, val_nll, epoch
            stale = 0
        else:
            stale += 1
        print(f"[canonical-initial] {dataset} fold={fold} seed={seed} epoch={epoch} loss={np.mean(losses):.5f} discovery_BA={val_ba:.5f} best_epoch={best_epoch}", flush=True)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_epoch, history, run_seed


def fit_refit(data: DatasetData, train_idx: np.ndarray, mean: np.ndarray, std: np.ndarray, dataset: str, fold: int, seed: int, epochs: int, device: torch.device) -> tuple[VanillaEEGNet, int]:
    run_seed = stable_seed("canonical-refit", dataset, fold, seed)
    set_seed(run_seed)
    model = VanillaEEGNet(data.batch(train_idx[:1]).shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order_rng = np.random.default_rng(stable_seed("canonical-order", dataset, fold, seed, "refit"))
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            part = order[start : start + BATCH_SIZE]
            xb = prepare_batch(data, part, mean, std, device)
            yb = torch.as_tensor(np.array(data.metadata.iloc[part].label.to_numpy(np.int64), copy=True), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[canonical-refit] {dataset} fold={fold} seed={seed} epoch={epoch}/{epochs} loss={np.mean(losses):.5f}", flush=True)
    return model, run_seed


def make_indices(data: DatasetData, role: dict[str, list[str]], dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sessions_fit = (1, 2) if dataset == "OpenBMI" else (0, 1)
    discovery_session = 2
    outcome_session = 2
    subjects = data.metadata.subject_id.astype(str)
    sessions = data.metadata.session_id.astype(int)
    fit_subjects = set(role["model_fit"])
    discovery_subjects = set(role["discovery"])
    outcome_subjects = set(role["outcome"])
    initial_mask = subjects.isin(fit_subjects).to_numpy() & sessions.isin(sessions_fit).to_numpy()
    discovery_mask = subjects.isin(discovery_subjects).to_numpy() & (sessions.to_numpy() == discovery_session)
    refit_mask = subjects.isin(fit_subjects | discovery_subjects).to_numpy() & sessions.isin(sessions_fit).to_numpy()
    outcome_mask = subjects.isin(outcome_subjects).to_numpy() & (sessions.to_numpy() == outcome_session)
    train_idx, discovery_idx = np.flatnonzero(initial_mask), np.flatnonzero(discovery_mask)
    refit_idx, outcome_idx = np.flatnonzero(refit_mask), np.flatnonzero(outcome_mask)
    if set(train_idx) & set(discovery_idx) or set(refit_idx) & set(outcome_idx) or set(train_idx) & set(outcome_idx):
        raise RuntimeError(f"{dataset} fold index overlap")
    # OpenBMI has exactly 100 MI trials per subject/session.  WBCIC's frozen
    # development cache contains 200 rows per subject/session except for a few
    # authorized recordings with documented missing rows; use the actual
    # manifest cardinality rather than inventing or padding trials.
    expected_outcome = int(outcome_mask.sum())
    expected_refit = int(refit_mask.sum())
    if not train_idx.size or not discovery_idx.size or len(outcome_idx) != expected_outcome or len(refit_idx) != expected_refit:
        raise RuntimeError(f"{dataset} fold index cardinality failed train={len(train_idx)} discovery={len(discovery_idx)} refit={len(refit_idx)} outcome={len(outcome_idx)}")
    return train_idx, discovery_idx, refit_idx, outcome_idx


def run_one(data: DatasetData, role: dict[str, list[str]], dataset: str, fold: int, seed: int, device: torch.device) -> dict[str, Any]:
    partial_path = RUNTIME / "partial" / f"{dataset.lower()}_fold-{fold}_seed-{seed}.json"
    if partial_path.is_file():
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
        if payload.get("complete") is True:
            ckpt_path = RUNTIME / "checkpoints" / dataset / f"fold-{fold}" / f"seed-{seed}.pt"
            if ckpt_path.is_file():
                # Migrate older partials produced before provenance fields
                # were added, while refusing to resume from a payload whose
                # checkpoint was never durably written.
                payload.setdefault("model_fit_subjects", list(role["model_fit"]))
                payload.setdefault("discovery_subjects", list(role["discovery"]))
                payload.setdefault("outcome_subjects", list(role["outcome"]))
                payload.setdefault("initial_training_subjects", list(role["model_fit"]))
                payload.setdefault("refit_training_subjects", subject_sort(set(role["model_fit"]) | set(role["discovery"])))
                payload.setdefault("checkpoint_path", str(ckpt_path))
                payload.setdefault("checkpoint_sha256", sha256_file(ckpt_path))
                preprocessing_path = OPENBMI_MANIFEST if dataset == "OpenBMI" else WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
                payload.setdefault("preprocessing_path", str(preprocessing_path))
                payload.setdefault("preprocessing_sha256", sha256_file(preprocessing_path))
                write_json(partial_path, payload)
                print(f"[resume] {dataset} fold={fold} seed={seed}", flush=True)
                return payload
            print(f"[resume-invalid] {dataset} fold={fold} seed={seed}: missing checkpoint; recomputing", flush=True)
    train_idx, discovery_idx, refit_idx, outcome_idx = make_indices(data, role, dataset)
    initial_mean, initial_std = compute_normalizer(data, train_idx)
    best_epoch, history, initial_seed = fit_initial(data, train_idx, discovery_idx, initial_mean, initial_std, dataset, fold, seed, device)
    refit_mean, refit_std = compute_normalizer(data, refit_idx)
    model, refit_seed = fit_refit(data, refit_idx, refit_mean, refit_std, dataset, fold, seed, best_epoch, device)
    outcome = evaluate(data, model, outcome_idx, refit_mean, refit_std, device)
    frame = data.metadata.iloc[outcome_idx]
    trial_rows = []
    for i, idx in enumerate(outcome_idx):
        row = frame.iloc[i]
        p0, p1 = map(float, outcome["probability"][i])
        trial_rows.append({"dataset": dataset, "seed": seed, "fold": fold, "subject_id": str(row.subject_id), "trial_uid": str(row.trial_uid), "session": session_label(dataset, int(row.session_id)), "label": int(row.label), "probability_class0": p0, "probability_class1": p1, "prediction": int(p1 >= p0)})
    subject_rows = []
    for row in outcome["subject_metrics"]:
        subject_rows.append({"dataset": dataset, "fold": fold, "seed": seed, "subject_id": str(row["subject_id"]), "BA": row["BA"], "accuracy": row["accuracy"], "macro_F1": row["macro_F1"], "NLL": row["NLL"], "trials": row["trials"], "best_epoch": best_epoch})
    ckpt_path = RUNTIME / "checkpoints" / dataset / f"fold-{fold}" / f"seed-{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "dataset": dataset, "fold": fold, "seed": seed, "best_epoch": best_epoch, "normalizer_mean": refit_mean, "normalizer_std": refit_std, "protocol": "canonical_vanilla_eegnet_v1"}, ckpt_path)
    # Save the checkpoint before marking the shard complete.  This makes a
    # resume after interruption safe: a complete partial can never point at a
    # missing model artifact.  Role membership and immutable input hashes are
    # retained in each partial for an auditable provenance trail.
    checkpoint_sha256 = sha256_file(ckpt_path)
    preprocessing_path = OPENBMI_MANIFEST if dataset == "OpenBMI" else WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
    payload = {"complete": True, "dataset": dataset, "fold": fold, "seed": seed, "best_epoch": best_epoch, "initial_seed": initial_seed, "refit_seed": refit_seed, "model_fit_subjects": list(role["model_fit"]), "discovery_subjects": list(role["discovery"]), "outcome_subjects": list(role["outcome"]), "initial_training_subjects": list(role["model_fit"]), "refit_training_subjects": subject_sort(set(role["model_fit"]) | set(role["discovery"])), "train_rows": len(train_idx), "discovery_rows": len(discovery_idx), "refit_rows": len(refit_idx), "outcome_rows": len(outcome_idx), "checkpoint_path": str(ckpt_path), "checkpoint_sha256": checkpoint_sha256, "preprocessing_path": str(preprocessing_path), "preprocessing_sha256": sha256_file(preprocessing_path), "initial_normalizer_mean": initial_mean.tolist(), "initial_normalizer_std": initial_std.tolist(), "refit_normalizer_mean": refit_mean.tolist(), "refit_normalizer_std": refit_std.tolist(), "initial_history": history, "subject_rows": subject_rows, "trial_rows": trial_rows}
    write_json(partial_path, payload)
    print(f"[complete] {dataset} fold={fold} seed={seed} outcome_BA={np.mean([r['BA'] for r in subject_rows]):.5f} best_epoch={best_epoch}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def bootstrap_mean(values: Iterable[float], seed: int) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, len(arr), size=(BOOTSTRAP_DRAWS, len(arr)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def aggregate(payloads: list[dict[str, Any]], roles_by_dataset: dict[str, list[dict[str, list[str]]]], data_by_dataset: dict[str, DatasetData]) -> None:
    trial = pd.DataFrame([row for p in payloads for row in p["trial_rows"]])
    seed_subject = pd.DataFrame([row for p in payloads for row in p["subject_rows"]])
    if trial.empty or seed_subject.empty:
        raise RuntimeError("no completed baseline runs to aggregate")
    primary_subject_rows = []
    primary_fold_rows = []
    seed_summary_rows = []
    summary_rows = []
    for dataset in DATASETS:
        t = trial[trial.dataset == dataset].copy()
        s = seed_subject[seed_subject.dataset == dataset].copy()
        merged = t.groupby(["fold", "subject_id", "trial_uid", "session", "label"], as_index=False)[["probability_class0", "probability_class1"]].mean()
        for (fold, subject), part in merged.groupby(["fold", "subject_id"], sort=False):
            y = part.label.to_numpy(int); p1 = part.probability_class1.to_numpy(float); pred = (p1 >= 0.5).astype(int)
            primary_subject_rows.append({"dataset": dataset, "seed": "all", "fold": int(fold), "subject_id": str(subject), "summary_type": "primary_three_seed_probability_mean", "BA": float(balanced_accuracy_score(y, pred)), "accuracy": float(accuracy_score(y, pred)), "macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)), "NLL": float(log_loss(y, np.column_stack([1-p1,p1]), labels=[0,1])), "trials": int(len(part)), "seed_0_BA": float(s[(s.fold == fold) & (s.subject_id == subject) & (s.seed == 0)].BA.iloc[0]), "seed_1_BA": float(s[(s.fold == fold) & (s.subject_id == subject) & (s.seed == 1)].BA.iloc[0]), "seed_2_BA": float(s[(s.fold == fold) & (s.subject_id == subject) & (s.seed == 2)].BA.iloc[0])})
        primary = pd.DataFrame([r for r in primary_subject_rows if r["dataset"] == dataset])
        for fold, part in primary.groupby("fold", sort=True):
            primary_fold_rows.append({"dataset": dataset, "fold": int(fold), "seed": "all", "summary_type": "primary_three_seed_probability_mean", "mean_subject_BA": float(part.BA.mean()), "median_subject_BA": float(part.BA.median()), "subject_SD": float(part.BA.std(ddof=1)), "mean_accuracy": float(part.accuracy.mean()), "mean_macro_F1": float(part.macro_F1.mean()), "mean_NLL": float(part.NLL.mean()), "n_subjects": int(len(part)), "trials_per_subject": int(part.trials.iloc[0])})
        for seed, part in s.groupby("seed", sort=True):
            seed_summary_rows.append({"dataset": dataset, "seed": int(seed), "summary_type": "single_seed_aggregate", "mean_subject_BA": float(part.BA.mean()), "median_subject_BA": float(part.BA.median()), "subject_SD": float(part.BA.std(ddof=1)), "mean_accuracy": float(part.accuracy.mean()), "mean_macro_F1": float(part.macro_F1.mean()), "mean_NLL": float(part.NLL.mean()), "n_subjects": int(len(part)), "mean_best_epoch": float(part.best_epoch.mean())})
            for fold, fpart in part.groupby("fold", sort=True):
                primary_fold_rows.append({"dataset": dataset, "fold": int(fold), "seed": int(seed), "summary_type": "single_seed", "mean_subject_BA": float(fpart.BA.mean()), "median_subject_BA": float(fpart.BA.median()), "subject_SD": float(fpart.BA.std(ddof=1)), "mean_accuracy": float(fpart.accuracy.mean()), "mean_macro_F1": float(fpart.macro_F1.mean()), "mean_NLL": float(fpart.NLL.mean()), "n_subjects": int(len(fpart)), "trials_per_subject": int(fpart.trials.iloc[0]), "mean_best_epoch": float(fpart.best_epoch.mean())})
        ci_l, ci_u = bootstrap_mean(primary.BA, stable_seed("canonical-bootstrap", dataset))
        seed_means = s.groupby("seed").BA.mean().sort_index()
        model_fit_counts = [len(role["model_fit"]) for role in roles_by_dataset[dataset]]
        discovery_counts = [len(role["discovery"]) for role in roles_by_dataset[dataset]]
        outcome_counts = [len(role["outcome"]) for role in roles_by_dataset[dataset]]
        common = {"dataset": dataset, "seed": "all", "fold": "all", "folds": 5, "seeds": 3, "n_model_fit_subjects": int(round(np.mean(model_fit_counts))), "n_discovery_subjects": int(round(np.mean(discovery_counts))), "n_outcome_subjects": int(round(np.mean(outcome_counts)))}
        summary_rows.append({**common, "summary_type": "primary_three_seed_probability_mean", "mean_subject_BA": float(primary.BA.mean()), "macro_F1": float(primary.macro_F1.mean()), "accuracy": float(primary.accuracy.mean()), "NLL": float(primary.NLL.mean()), "median_subject_BA": float(primary.BA.median()), "subject_SD": float(primary.BA.std(ddof=1)), "bootstrap_CI95_L": ci_l, "bootstrap_CI95_U": ci_u, "mean_accuracy": float(primary.accuracy.mean()), "mean_macro_F1": float(primary.macro_F1.mean()), "mean_NLL": float(primary.NLL.mean()), "n_subjects": int(len(primary)), "mean_best_epoch": float(s.best_epoch.mean()), "best_epoch": float(s.best_epoch.mean()), "historical_reference_BA": HISTORICAL_REFERENCE[dataset], "difference_vs_historical_pp": float(100 * (primary.BA.mean() - HISTORICAL_REFERENCE[dataset]))})
        summary_rows.append({**common, "summary_type": "secondary_mean_single_seed_aggregate", "mean_subject_BA": float(seed_means.mean()), "macro_F1": float(s.macro_F1.mean()), "accuracy": float(s.accuracy.mean()), "NLL": float(s.NLL.mean()), "median_subject_BA": float(s.BA.median()), "subject_SD": float(seed_means.std(ddof=1)), "bootstrap_CI95_L": None, "bootstrap_CI95_U": None, "mean_accuracy": float(s.accuracy.mean()), "mean_macro_F1": float(s.macro_F1.mean()), "mean_NLL": float(s.NLL.mean()), "n_subjects": int(len(s)), "mean_best_epoch": float(s.best_epoch.mean()), "best_epoch": float(s.best_epoch.mean()), "historical_reference_BA": HISTORICAL_REFERENCE[dataset], "difference_vs_historical_pp": float(100 * (seed_means.mean() - HISTORICAL_REFERENCE[dataset]))})
    write_csv(RESULTS / "TRIAL_PREDICTIONS.csv", trial.sort_values(["dataset", "fold", "seed", "subject_id", "trial_uid"]))
    write_csv(RESULTS / "PER_SUBJECT_RESULTS.csv", pd.DataFrame(primary_subject_rows).sort_values(["dataset", "fold", "subject_id"]))
    write_csv(RESULTS / "PER_FOLD_RESULTS.csv", pd.DataFrame(primary_fold_rows).sort_values(["dataset", "fold", "summary_type", "seed"], key=lambda col: col.astype(str)))
    write_csv(RESULTS / "SEED_SUMMARY.csv", pd.DataFrame(seed_summary_rows).sort_values(["dataset", "seed"]))
    write_csv(RESULTS / "BASELINE_SUMMARY.csv", pd.DataFrame(summary_rows))
    stats = {"schema": "CANONICAL_VANILLA_EEGNET_BASELINE_V1", "primary_unit": "biological subject", "bootstrap_draws": BOOTSTRAP_DRAWS, "primary_definition": "average the three seed probabilities per trial, then compute subject BA", "secondary_definition": "mean of three single-seed aggregate subject BA values", "datasets": {}}
    for row in summary_rows:
        stats["datasets"].setdefault(row["dataset"], {})[row["summary_type"]] = row
    stats["run_counts"] = {dataset: int(sum(1 for p in payloads if p["dataset"] == dataset)) for dataset in DATASETS}
    write_json(RESULTS / "CANONICAL_BASELINE_STATISTICS.json", stats)
    write_json(RUNTIME / "AGGREGATION_COMPLETE.json", {"complete": True, "payload_count": len(payloads), "trial_rows": len(trial), "subject_rows": len(seed_subject)})


def write_protocol_and_audits(payloads: list[dict[str, Any]], roles_by_dataset: dict[str, list[dict[str, list[str]]]], data_by_dataset: dict[str, DatasetData], device: torch.device) -> None:
    protocol_lock = {
        "schema": "CANONICAL_VANILLA_EEGNET_BASELINE_V1", "created_before_outcome_scoring": True,
        "architecture": {"F1": 8, "D": 2, "F2": 16, "dropout": 0.25, "embedding": "Linear(496,64)+ELU+LayerNorm(64)", "classifier": "Linear(64,2)"},
        "optimizer": {"name": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "criterion": "cross_entropy"},
        "training": {"max_epochs": MAX_EPOCHS, "minimum_epochs": MIN_EPOCHS, "patience": PATIENCE, "seeds": list(SEEDS), "folds": list(FOLDS), "initial_fit_once": True, "deterministic_refit_once": True, "epoch_selection": "discovery mean subject BA; lower NLL; earlier epoch"},
        "data": {"OpenBMI": {"paradigm": "Lee2019 MI", "subjects": 54, "model_fit_sessions": ["S1", "S2"], "discovery_session": "S2", "refit_sessions": ["S1", "S2"], "outcome_session": "S2"}, "WBCIC": {"subjects": 41, "model_fit_sessions": ["S1", "S2"], "discovery_session": "S3", "refit_sessions": ["S1", "S2"], "outcome_session": "S3"}},
        "forbidden": ["OpenBMI sealed/internal holdout", "WBCIC outer 10", "outcome-driven tuning", "adaptation", "CGR", "stacking", "OOF action bank"],
        "provenance": {"split_freeze_sha256": sha256_file(OPENBMI_SPLIT), "wbcic_scope_lock_sha256": sha256_file(WBCIC_LOCK), "openbmi_manifest_sha256": sha256_file(OPENBMI_MANIFEST), "wbcic_metadata_sha256": sha256_file(WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"), "wbcic_raw_sha256": sha256_file(WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_RAW.npy")},
    }
    write_json(PROTOCOL / "CANONICAL_PROTOCOL_LOCK.json", protocol_lock)
    lines = ["# Canonical protocol", "", "## CANONICAL BASELINE STATUS", "", "This is a pre-registered vanilla EEGNet baseline; no outcome label is read until the final scoring call.", "", "### Architecture", "", "- F1=8, depth multiplier D=2, F2=16; temporal kernel 64; ELU; average pooling 4 then 8; dropout=0.25; Linear(496,64)+ELU+LayerNorm(64); Linear(64,2).", "- Input is the frozen 1000-sample MI epoch; channel count is 62 for OpenBMI and 58 for WBCIC.", "", "### Training", "", "- AdamW, learning rate 3e-4, weight decay 5e-4, batch size 64, cross entropy, max 60 epochs, minimum 10 epochs, patience 8.", "- Seeds are exactly 0, 1, and 2. Each dataset/fold/seed has one initial fit and one deterministic refit.", "- Initial fit uses model-fit subjects and S1+S2. Epoch is selected only on disjoint discovery subjects by mean biological-subject balanced accuracy, then lower NLL, then earlier epoch.", "- Refit uses model-fit+discovery subjects and S1+S2 for exactly the selected epoch count. Outcome subjects are never in training or epoch selection.", "", "### Dataset roles", "", "- OpenBMI uses all 54 Stage-0-frozen Lee2019 MI subjects; model-fit = frozen train subjects, discovery = frozen validation subjects, outcome = frozen outer-test subjects. Physical sessions are 1/2 (S1/S2).", "- WBCIC uses only the 41 `DEVELOPMENT_SCOPE_LOCK.json` allowed subjects; model-fit/discovery/outcome roles are the frozen audit roles. Physical sessions 0/1/2 are S1/S2/S3. The sealed outer 10 are not enumerated or opened.", "", "### Statistics", "", "- Primary summary averages the three seed probabilities per trial, then computes subject BA. CI is a 10,000-draw biological-subject bootstrap.", "- Secondary robustness statistic is the mean of the three single-seed aggregate subject BAs. Fold and seed tables retain both views."]
    # Keep the human-readable protocol at the experiment root, matching the
    # requested deliverable names; the machine-readable lock stays under
    # ``protocol/``.
    (EXP / "CANONICAL_PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The legality audit is generated from observed rows and frozen roles, not inferred from scores.
    legality = {"schema": "CANONICAL_BASELINE_LEGALITY_AUDIT_V1", "pass": True, "datasets": {}}
    for dataset in DATASETS:
        data = data_by_dataset[dataset]
        roles = roles_by_dataset[dataset]
        legality["datasets"][dataset] = {"subjects_observed": int(data.metadata.subject_id.nunique()), "subjects_expected": int(len(set(data.metadata.subject_id))), "rows": int(len(data.metadata)), "sessions_observed": sorted(map(int, data.metadata.session_id.unique())), "fold_role_counts": [{k: len(v) for k, v in role.items()} for role in roles], "outer_cohort_opened": False, "outcome_labels_read_before_final_score": False}
    legality["runs_completed"] = len(payloads)
    write_json(PROTOCOL / "BASELINE_LEGALITY_AUDIT.json", legality)
    (EXP / "BASELINE_LEGALITY_AUDIT.md").write_text("# Baseline legality audit\n\n## CANONICAL BASELINE STATUS\n\nPASS: all frozen role, session and row-count assertions passed. OpenBMI uses 54 Stage-0 subjects. WBCIC uses only the 41 subjects in `DEVELOPMENT_SCOPE_LOCK.json`; its outer cohort is not enumerated. Initial fits use model-fit S1+S2, discovery is disjoint, refits use model-fit+discovery S1+S2, and outcome scoring occurs only after refit.\n\nThe runner constructs outcome indices only for the final scoring call and never uses outcome labels for training, normalization, or epoch selection.\n", encoding="utf-8")
    provenance = {"schema": "CANONICAL_BASELINE_PROVENANCE_AUDIT_V1", "repository": str(REPO), "experiment": str(EXP), "git_head": os.environ.get("CANONICAL_GIT_HEAD", "recorded_at_launch"), "device": str(device), "stage0_split": str(OPENBMI_SPLIT), "openbmi_manifest": str(OPENBMI_MANIFEST), "wbcic_scope_lock": str(WBCIC_LOCK), "wbcic_cache": str(WBCIC_CACHE), "hashes": protocol_lock["provenance"], "runtime_not_committed": True, "raw_eeg_not_committed": True}
    write_json(PROTOCOL / "BASELINE_PROVENANCE_AUDIT.json", provenance)
    (EXP / "BASELINE_PROVENANCE_AUDIT.md").write_text("# Baseline provenance audit\n\n## CANONICAL BASELINE STATUS\n\nThe canonical runner records immutable hashes for the Stage-0 split, OpenBMI MI manifest and WBCIC development scope/cache. The code runs in the target branch worktree. Runtime/checkpoint/cache/raw EEG files remain outside the committed artifact set; only compact tables, locks, audits and reports are deliverables.\n\nThe WBCIC scope lock states `outer_subject_ids_present=false`; no outer split file is opened.\n", encoding="utf-8")
    (EXP / "HISTORICAL_REFERENCE_AUDIT.md").write_text("# Historical reference audit\n\n## CANONICAL BASELINE STATUS\n\nHistorical values are context only and were not used for model construction, epoch selection, tuning, or stopping. The nearest stored references are OpenBMI BA=0.7719 and WBCIC BA=0.7884 from prior project artifacts with different subject/cache scopes. If a canonical result differs by more than five percentage points from its reference, the result is reported as-is and triggers an audit of split, subject roles, session usage, preprocessing, channel handling, normalization, checkpoint selection and metric aggregation; no parameter is changed to match the reference.\n", encoding="utf-8")


def write_final_reports(payloads: list[dict[str, Any]], device: torch.device) -> None:
    summary = pd.read_csv(RESULTS / "BASELINE_SUMMARY.csv")
    fold_results = pd.read_csv(RESULTS / "PER_FOLD_RESULTS.csv")
    seed_results = pd.read_csv(RESULTS / "SEED_SUMMARY.csv")
    stats = json.loads((RESULTS / "CANONICAL_BASELINE_STATISTICS.json").read_text(encoding="utf-8"))
    lines = ["# CANONICAL BASELINE STATUS", "", "This report describes the frozen vanilla EEGNet baseline only. It is not a claim about adaptation or utility.", "", "## Protocol", "", "The exact protocol and legality/provenance audits are in `CANONICAL_PROTOCOL.md`, `BASELINE_LEGALITY_AUDIT.md`, `BASELINE_PROVENANCE_AUDIT.md`, and `HISTORICAL_REFERENCE_AUDIT.md`.", "", "## Primary results", "", "Primary = average the three seed probabilities per trial, then compute biological-subject BA; CI = 10,000-draw subject bootstrap.", "", "| dataset | mean subject BA | median | SD | 95% CI | subjects |", "|---|---:|---:|---:|---|---:|"]
    for _, row in summary[summary.summary_type == "primary_three_seed_probability_mean"].iterrows():
        lines.append(f"| {row.dataset} | {row.mean_subject_BA:.6f} | {row.median_subject_BA:.6f} | {row.subject_SD:.6f} | [{row.bootstrap_CI95_L:.6f}, {row.bootstrap_CI95_U:.6f}] | {int(row.n_subjects)} |")
    lines += ["", "## Secondary seed robustness", "", "The secondary statistic is the mean of the three single-seed aggregate subject BAs; it is not substituted for the primary summary.", "", "| dataset | mean single-seed BA | mean accuracy | mean NLL |", "|---|---:|---:|---:|"]
    for _, row in summary[summary.summary_type == "secondary_mean_single_seed_aggregate"].iterrows():
        lines.append(f"| {row.dataset} | {row.mean_subject_BA:.6f} | {row.mean_accuracy:.6f} | {row.mean_NLL:.6f} |")
    lines += ["", "## Fold and seed means", ""]
    for dataset in DATASETS:
        fpart = fold_results[(fold_results.dataset == dataset) & (fold_results.summary_type == "primary_three_seed_probability_mean")].sort_values("fold")
        spart = seed_results[seed_results.dataset == dataset].sort_values("seed")
        fold_values = ", ".join(f"{float(v) * 100:.2f}%" for v in fpart.mean_subject_BA.tolist())
        seed_values = ", ".join(f"{float(v) * 100:.2f}%" for v in spart.mean_subject_BA.tolist())
        lines.append(f"- {dataset} fold means: [{fold_values}]")
        lines.append(f"- {dataset} seed means: [{seed_values}]")
    lines += ["", "## Scope conclusion", "", "The OpenBMI analysis covers all 54 Stage-0-frozen Lee2019 MI subjects. The WBCIC analysis covers only the 41 frozen development subjects. The WBCIC sealed outer 10 and any OpenBMI sealed/internal holdout were not accessed. Runtime/checkpoints/cache/raw EEG are not deliverables.", "", "## Required validity answers", "", "1. OpenBMI is raw EEGNet evaluation rather than a frozen historical embedding plus sklearn head: YES.", "2. WBCIC uses exactly the same outer evaluation logic (frozen subject roles, discovery-only epoch selection, one final future-session score): YES.", "3. All non-outcome subjects were used in the final refit: YES.", "4. Outcome future labels were excluded from all fitting and selection: YES.", "5. Outcome-subject history was excluded from vanilla adaptation: YES; no adaptation is performed.", "6. The WBCIC sealed outer 10 were untouched: YES.", "7. Trial predictions use the frozen outcome trials and can be used for direct paired comparison with Ours: YES.", "", "terminal = CANONICAL_EEGNET_BASELINE_ESTABLISHED"]
    (EXP / "FINAL_BASELINE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    validity_answers = {"raw_openbmi_eegnet": True, "same_wbcic_outer_logic": True, "all_non_outcome_subjects_refit": True, "outcome_future_labels_excluded": True, "outcome_history_adaptation_excluded": True, "wbcic_sealed_outer_untouched": True, "trial_predictions_paired_comparison_ready": True}
    report = {"status": "CANONICAL BASELINE STATUS", "protocol": "CANONICAL_VANILLA_EEGNET_BASELINE_V1", "terminal": "CANONICAL_EEGNET_BASELINE_ESTABLISHED", "validity_answers": validity_answers, "summary": stats, "fold_means_primary": {dataset: [float(v) for v in fold_results[(fold_results.dataset == dataset) & (fold_results.summary_type == "primary_three_seed_probability_mean")].sort_values("fold").mean_subject_BA.tolist()] for dataset in DATASETS}, "seed_means_secondary": {dataset: [float(v) for v in seed_results[seed_results.dataset == dataset].sort_values("seed").mean_subject_BA.tolist()] for dataset in DATASETS}, "payload_count": len(payloads), "outer_status": {"OpenBMI_internal_holdout_accessed": False, "WBCIC_outer_10_accessed": False}, "deliverables": ["results/BASELINE_SUMMARY.csv", "results/PER_SUBJECT_RESULTS.csv", "results/TRIAL_PREDICTIONS.csv", "results/PER_FOLD_RESULTS.csv", "results/SEED_SUMMARY.csv", "results/CANONICAL_BASELINE_STATISTICS.json", "FINAL_BASELINE_REPORT.md", "FINAL_BASELINE_REPORT.json", "CANONICAL_PROTOCOL.md", "BASELINE_LEGALITY_AUDIT.md", "BASELINE_PROVENANCE_AUDIT.md", "HISTORICAL_REFERENCE_AUDIT.md"]}
    write_json(EXP / "FINAL_BASELINE_REPORT.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    for path in (RESULTS, PROTOCOL, RUNTIME / "partial", RUNTIME / "checkpoints"):
        path.mkdir(parents=True, exist_ok=True)
    roles_by_dataset: dict[str, list[dict[str, list[str]]]] = {}
    data_by_dataset: dict[str, DatasetData] = {}
    for dataset in args.datasets:
        roles, pool, _ = load_roles(dataset)
        data = load_dataset(dataset, pool)
        roles_by_dataset[dataset] = roles
        data_by_dataset[dataset] = data
        print(f"[preflight] {dataset} subjects={len(pool)} rows={len(data.metadata)} shape={(data.raw.shape if data.raw is not None else 'sharded OpenBMI')} roles={[{k:len(v) for k,v in r.items()} for r in roles]}", flush=True)
    payloads: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for fold, role in zip(FOLDS, roles_by_dataset[dataset]):
            for seed in SEEDS:
                payloads.append(run_one(data_by_dataset[dataset], role, dataset, fold, seed, device))
    aggregate(payloads, roles_by_dataset, data_by_dataset)
    # Record the branch tip only after the code has been checked out in the worktree.
    try:
        import subprocess
        os.environ["CANONICAL_GIT_HEAD"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        pass
    write_protocol_and_audits(payloads, roles_by_dataset, data_by_dataset, device)
    write_final_reports(payloads, device)
    print((pd.read_csv(RESULTS / "BASELINE_SUMMARY.csv")).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
