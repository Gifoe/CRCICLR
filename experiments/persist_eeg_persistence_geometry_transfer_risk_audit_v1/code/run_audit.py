"""PERSIST-EEG persistence-geometry transfer-risk audit (seed 0 only).

This runner has two explicitly separated phases.  ``preflight`` trains a
source-only EEGNet for every canonical dataset/fold, reconstructs the
train-only persistence spectrum and Protected geometry, freezes descriptor
support and writes a pre-outcome lock.  ``outcome`` requires that lock and
only then reads the discovery query labels.  Canonical outcome subjects are
never loaded or indexed by this program.

All scientific constants are intentionally explicit and are inherited from
the registered Signed V3.1 / Shared Geometry definitions.  Runtime files are
written below ``runtime/`` (ignored by git); only compact tables and locks are
deliverables.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr, kendalltau, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


# ---------------------------------------------------------------------------
# Immutable protocol constants
# ---------------------------------------------------------------------------
EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
RESULTS.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)

SEED = 0
FOLDS = (0, 1, 2, 3, 4)
DATASETS = ("OpenBMI", "WBCIC")
CAP_CANDIDATES = (32, 16, 8)
Q_MIN = 16
MAX_EPOCHS = 60
MIN_EPOCHS = 10
PATIENCE = 8
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
PERSISTENCE_RANK = 20
PER_GROUP = 32
INNER_SPLITS = 5
NULL_DRAWS = 100
BOOTSTRAP_DRAWS = 10_000
EPS = 1e-12
PROBE_DIM = 24  # Signed V3.1 probe convention

CANONICAL_REPO = Path(os.environ.get(
    "CANONICAL_REPO", r"D:\\nips-temp\\TotalP\\P1\\CRCICLR_CANONICAL_EEGNET"
)).resolve()
SOURCE_REPO = Path(os.environ.get(
    "SOURCE_REPO", r"D:\\nips-temp\\TotalP\\P1\\CRCICLR_SOURCE_ONLY_DIAGNOSTIC"
)).resolve()
STAGE0_ROOT = Path(os.environ.get(
    "PERSIST_STAGE0_REPO", r"D:\\nips-temp\\TotalP\\P1\\persist_eeg_stage0_repo_full"
)).resolve()
WBCIC_CACHE = Path(os.environ.get(
    "PERSIST_WBCIC_CACHE",
    str(SOURCE_REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1" / "runtime" / "cache"),
)).resolve()
OPENBMI_MANIFEST = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
OPENBMI_SPLIT = STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
WBCIC_SCOPE = SOURCE_REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1" / "provenance" / "DEVELOPMENT_SCOPE_LOCK.json"
WBCIC_META = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
WBCIC_RAW = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_RAW.npy"

try:
    torch.set_num_threads(int(os.environ.get("PERSIST_TORCH_THREADS", "24")))
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
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


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
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
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False) % (2**63 - 1)


def set_seed(seed: int) -> None:
    if int(seed) != 0:
        raise RuntimeError("this audit is registered for seed 0 only")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    def key(v: object) -> tuple[int, str]:
        s = str(v).replace("sub-", "")
        return (int(s), s) if s.isdigit() else (10**9, s)
    return sorted((str(v).replace("sub-", "") for v in values), key=key)


# ---------------------------------------------------------------------------
# Frozen roles and safe A/B-only data access
# ---------------------------------------------------------------------------
def load_roles(dataset: str) -> tuple[list[dict[str, list[str]]], set[str], dict[str, Any]]:
    if dataset == "OpenBMI":
        payload = json.loads(OPENBMI_SPLIT.read_text(encoding="utf-8-sig"))
        section = payload["openbmi"]
        pool = set(subject_sort(section["subjects"]))
        folds = []
        for row in section["folds"]:
            role = {"model_fit": subject_sort(row["train_subjects"]),
                    "discovery": subject_sort(row["validation_subjects"]),
                    "outcome": subject_sort(row["outer_test_subjects"])}
            assert_roles(dataset, role, pool)
            folds.append(role)
        return folds, pool, payload
    lock = json.loads(WBCIC_SCOPE.read_text(encoding="utf-8-sig"))
    if lock.get("outer_subject_ids_present") is not False or int(lock.get("outer_subject_count", -1)) != 10:
        raise RuntimeError("WBCIC scope lock does not prove outer cohort exclusion")
    pool = set(subject_sort(lock["allowed_subjects"]))
    folds = []
    for k in map(str, FOLDS):
        row = lock["audit_roles"][k]
        role = {"model_fit": subject_sort(row["model_fit"]),
                "discovery": subject_sort(row["discovery_decision"]),
                "outcome": subject_sort(row["outcome"])}
        assert_roles(dataset, role, pool)
        folds.append(role)
    return folds, pool, lock


def assert_roles(dataset: str, role: Mapping[str, Sequence[str]], pool: set[str]) -> None:
    parts = [set(role[k]) for k in ("model_fit", "discovery", "outcome")]
    if any(a & b for i, a in enumerate(parts) for b in parts[i + 1:]):
        raise RuntimeError(f"{dataset} role overlap")
    if set().union(*parts) != pool:
        raise RuntimeError(f"{dataset} role not exhaustive")
    if any(not p for p in parts):
        raise RuntimeError(f"{dataset} empty role")


@dataclass
class SafeData:
    dataset: str
    metadata: pd.DataFrame
    root: Path
    raw: np.ndarray | None
    arrays: dict[str, np.ndarray]

    def batch(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 0:
            channels = 62 if self.dataset == "OpenBMI" else 58
            return np.empty((0, channels, 1000), dtype=np.float32)
        if self.raw is not None:
            return np.asarray(self.raw[self.metadata.iloc[idx]["_source_row"].to_numpy(np.int64)], dtype=np.float32)
        frame = self.metadata.iloc[idx]
        paths = frame["_signal_path"].astype(str).to_numpy()
        offsets = frame["_cache_index"].to_numpy(np.int64)
        first_key = str(paths[0])
        if first_key not in self.arrays:
            self.arrays[first_key] = np.load(self.root / first_key, mmap_mode="r", allow_pickle=False)
        first = self.arrays[first_key]
        out = np.empty((len(idx), int(first.shape[1]), int(first.shape[2])), dtype=np.float32)
        for key in np.unique(paths):
            key = str(key)
            if key not in self.arrays:
                self.arrays[key] = np.load(self.root / key, mmap_mode="r", allow_pickle=False)
            mask = paths == key
            out[mask] = np.asarray(self.arrays[key][offsets[mask]], dtype=np.float32)
        return out


def _openbmi_ab_metadata(subjects: set[str]) -> pd.DataFrame:
    if not OPENBMI_MANIFEST.is_file():
        raise FileNotFoundError(OPENBMI_MANIFEST)
    cols = ["subject_id", "session_id", "paradigm", "run_phase", "trial_id", "signal_cache_path", "cache_index"]
    tab = pq.read_table(OPENBMI_MANIFEST, columns=cols,
                        filters=[("paradigm", "=", "mi"), ("run_phase", "=", "train"),
                                 ("subject_id", "in", sorted(subjects))])
    frame = tab.to_pandas()
    frame["subject_id"] = frame["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    frame["session_id"] = frame["session_id"].astype(int)
    frame["trial_uid"] = frame["trial_id"].astype(str)
    # Label column is deliberately read only for A/B subjects via the same filter.
    labels = pq.read_table(OPENBMI_MANIFEST, columns=["subject_id", "session_id", "trial_id", "event_code"],
                           filters=[("paradigm", "=", "mi"), ("run_phase", "=", "train"),
                                    ("subject_id", "in", sorted(subjects))]).to_pandas()
    labels["subject_id"] = labels["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    labels["session_id"] = labels["session_id"].astype(int)
    key = ["subject_id", "session_id", "trial_id"]
    frame = frame.merge(labels[key + ["event_code"]], on=key, how="left", validate="one_to_one")
    frame["label"] = frame["event_code"].astype(int).map({1: 0, 2: 1})
    if frame["label"].isna().any() or set(frame["label"].astype(int)) != {0, 1}:
        raise RuntimeError("OpenBMI A/B label mapping failed")
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.drop(columns=["event_code"]).sort_values(["subject_id", "session_id", "trial_uid"]).reset_index(drop=True)
    # Cache shards are checked only for A/B subjects; no outer path is enumerated.
    for rel in frame["signal_cache_path"].astype(str).drop_duplicates():
        p = STAGE0_ROOT / rel
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        if arr.shape != (100, 62, 1000) or arr.dtype != np.float32:
            raise RuntimeError(f"OpenBMI shard audit failed: {p} {arr.shape} {arr.dtype}")
    cells = frame.groupby(["subject_id", "session_id", "label"]).size()
    if len(cells) != len(subjects) * 2 * 2 or set(cells.astype(int)) != {50}:
        raise RuntimeError("OpenBMI A/B cell support failed")
    frame["_signal_path"] = frame["signal_cache_path"].astype(str)
    frame["_cache_index"] = frame["cache_index"].astype(np.int64)
    return frame


def _wbcic_ab_metadata(subjects: set[str]) -> pd.DataFrame:
    if not WBCIC_META.is_file() or not WBCIC_RAW.is_file():
        raise FileNotFoundError(WBCIC_META)
    # Read subject/session/trial identity for the complete development cache but
    # never read its label column for a C subject.  Labels are fetched below with
    # a predicate restricted to A/B subject IDs.
    identity = pq.read_table(WBCIC_META, columns=["subject_id", "session_id", "trial_in_session"]).to_pandas()
    identity["subject_id"] = identity["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    identity["session_id"] = identity["session_id"].astype(int)
    identity["_source_row"] = np.arange(len(identity), dtype=np.int64)
    identity = identity[identity["subject_id"].isin(subjects)].copy()
    tab = pq.read_table(WBCIC_META, columns=["subject_id", "session_id", "label", "trial_in_session"],
                        filters=[("subject_id", "in", sorted(subjects))])
    labels = tab.to_pandas()
    labels["subject_id"] = labels["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    labels["session_id"] = labels["session_id"].astype(int)
    key = ["subject_id", "session_id", "trial_in_session"]
    frame = identity.merge(labels[key + ["label"]], on=key, how="left", validate="one_to_one")
    if frame["label"].isna().any():
        raise RuntimeError("WBCIC A/B label predicate failed")
    frame["label"] = frame["label"].astype(int)
    frame["trial_uid"] = [f"wbcic-{s}-S{int(se)+1}-{int(t):05d}" for s, se, t in zip(frame.subject_id, frame.session_id, frame.trial_in_session)]
    frame = frame.sort_values(["subject_id", "session_id", "trial_in_session"]).reset_index(drop=True)
    raw = np.load(WBCIC_RAW, mmap_mode="r", allow_pickle=False)
    if raw.shape != (len(identity) + int((pq.read_table(WBCIC_META, columns=["subject_id"]).num_rows - len(identity))), 58, 1000) or raw.dtype != np.float16:
        # The shape assertion intentionally avoids enumerating any outer IDs;
        # only the physical array shape is checked.
        if raw.ndim != 3 or raw.shape[1:] != (58, 1000) or raw.dtype != np.float16:
            raise RuntimeError(f"WBCIC raw cache audit failed: {raw.shape} {raw.dtype}")
    cells = frame.groupby(["subject_id", "session_id", "label"]).size()
    if set(frame.session_id) != {0, 1, 2} or set(frame.label) != {0, 1} or int(cells.min()) < 16:
        raise RuntimeError("WBCIC A/B cell support failed")
    return frame


def load_ab_data(dataset: str, subjects: set[str]) -> SafeData:
    if dataset == "OpenBMI":
        return SafeData(dataset, _openbmi_ab_metadata(subjects), STAGE0_ROOT, None, {})
    return SafeData(dataset, _wbcic_ab_metadata(subjects), WBCIC_CACHE, np.load(WBCIC_RAW, mmap_mode="r", allow_pickle=False), {})


def stable_inner_split(subjects: Sequence[str], dataset: str, fold: int) -> tuple[list[str], list[str]]:
    values = np.asarray(subject_sort(subjects), dtype=object)
    rng = np.random.default_rng(stable_seed("a-only-inner-split", dataset, fold, SEED))
    values = values[rng.permutation(len(values))]
    n_train = max(1, int(math.floor(0.8 * len(values))))
    n_train = min(n_train, len(values) - 1)
    return subject_sort(values[:n_train]), subject_sort(values[n_train:])


def indices_for(data: SafeData, subjects: Sequence[str], sessions: Sequence[int]) -> np.ndarray:
    m = data.metadata
    mask = m["subject_id"].astype(str).isin(set(map(str, subjects))) & m["session_id"].isin(list(map(int, sessions)))
    return np.flatnonzero(mask.to_numpy()).astype(np.int64)


def indices_subject_session_label(data: SafeData, subjects: Sequence[str], sessions: Sequence[int]) -> dict[tuple[str, int, int], np.ndarray]:
    m = data.metadata
    mask = m["subject_id"].astype(str).isin(set(map(str, subjects))) & m["session_id"].isin(list(map(int, sessions)))
    frame = m.loc[mask]
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for (s, se, lab), g in frame.groupby(["subject_id", "session_id", "label"], sort=True):
        out[(str(s), int(se), int(lab))] = g.index.to_numpy(np.int64)
    return out


def compute_normalizer(data: SafeData, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = 62 if data.dataset == "OpenBMI" else 58
    total = np.zeros(channels, dtype=np.float64)
    square = np.zeros(channels, dtype=np.float64)
    count = 0
    for start in range(0, len(idx), 128):
        x = data.batch(idx[start:start + 128]).astype(np.float64)
        total += x.sum(axis=(0, 2)); square += np.square(x).sum(axis=(0, 2)); count += x.shape[0] * x.shape[2]
    mean = total / max(count, 1)
    var = np.maximum(square / max(count, 1) - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def prepare(data: SafeData, idx: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = data.batch(idx)
    x = (x - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


class VanillaEEGNet(nn.Module):
    def __init__(self, channels: int, samples: int = 1000, dropout: float = 0.25):
        super().__init__()
        if samples != 1000:
            raise ValueError("EEGNet expects 1000 samples")
        f1, d, f2 = 8, 2, 16
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((1, 4)); self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(f1 * d, f1 * d, (1, 16), padding="same", groups=f1 * d, bias=False)
        self.point = nn.Conv2d(f1 * d, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8)); self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(f2 * (samples // 4 // 8), 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        v = x.unsqueeze(1)
        v = self.bn1(self.temporal(v))
        v = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(v)))))
        v = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(v))))))
        return self.embedding(v.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def eval_subject_metrics(data: SafeData, model: nn.Module, idx: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model.eval(); logits = []
    with torch.inference_mode():
        for start in range(0, len(idx), BATCH_SIZE):
            logits.append(model(prepare(data, idx[start:start + BATCH_SIZE], mean, std)).numpy())
    logit = np.concatenate(logits, axis=0)
    stable = logit - logit.max(axis=1, keepdims=True); prob = np.exp(stable); prob /= prob.sum(axis=1, keepdims=True)
    frame = data.metadata.iloc[idx].reset_index(drop=True)
    y = frame.label.to_numpy(np.int64); pred = prob.argmax(1)
    rows = []
    for subject, g in frame.groupby(frame.subject_id.astype(str), sort=True):
        loc = g.index.to_numpy(np.int64); yy, pp = y[loc], prob[loc, 1]; pr = (pp >= 0.5).astype(int)
        rows.append({"subject_id": str(subject), "BA": float(balanced_accuracy_score(yy, pr)),
                     "accuracy": float(accuracy_score(yy, pr)), "macro_F1": float(f1_score(yy, pr, average="macro", zero_division=0)),
                     "NLL": float(log_loss(yy, prob[loc], labels=[0, 1])), "trials": int(len(loc))})
    return pd.DataFrame(rows), prob, y


def train_epoch(model: nn.Module, data: SafeData, idx: np.ndarray, mean: np.ndarray, std: np.ndarray,
                optimizer: torch.optim.Optimizer, order: np.ndarray) -> float:
    model.train(); losses = []
    labels = data.metadata["label"].to_numpy(np.int64)
    for start in range(0, len(order), BATCH_SIZE):
        part = order[start:start + BATCH_SIZE]
        xb = prepare(data, part, mean, std); yb = torch.from_numpy(labels[part]).long()
        optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(xb), yb); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step(); losses.append(float(loss.detach()))
    return float(np.mean(losses))


def fit_initial(data: SafeData, train_idx: np.ndarray, val_idx: np.ndarray, mean: np.ndarray, std: np.ndarray,
                dataset: str, fold: int) -> tuple[int, list[dict[str, Any]], int]:
    run_seed = stable_seed("a-only-initial", dataset, fold, SEED); set_seed(0); torch.manual_seed(run_seed)
    model = VanillaEEGNet(data.batch(train_idx[:1]).shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(stable_seed("a-only-minibatch", dataset, fold, SEED, "initial")); labels = data.metadata["label"].to_numpy(np.int64)
    best_ba, best_nll, best_epoch, stale = -math.inf, math.inf, 1, 0; history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        order = train_idx[rng.permutation(len(train_idx))]; loss = train_epoch(model, data, train_idx, mean, std, optimizer, order)
        metrics, _, _ = eval_subject_metrics(data, model, val_idx, mean, std); ba = float(metrics.BA.mean()); nll = float(metrics.NLL.mean())
        history.append({"epoch": epoch, "train_loss": loss, "inner_validation_mean_subject_BA": ba, "inner_validation_mean_subject_NLL": nll})
        improved = ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and nll < best_nll - 1e-12)
        if improved: best_ba, best_nll, best_epoch, stale = ba, nll, epoch, 0
        else: stale += 1
        print(f"[A-only initial] {dataset} fold={fold} epoch={epoch} loss={loss:.5f} val_BA={ba:.5f} best={best_epoch}", flush=True)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE: break
    del model; gc.collect(); return int(best_epoch), history, int(run_seed)


def fit_refit(data: SafeData, idx: np.ndarray, mean: np.ndarray, std: np.ndarray, dataset: str, fold: int, epochs: int) -> tuple[VanillaEEGNet, int]:
    run_seed = stable_seed("a-only-refit", dataset, fold, SEED); set_seed(0); torch.manual_seed(run_seed)
    model = VanillaEEGNet(data.batch(idx[:1]).shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(stable_seed("a-only-minibatch", dataset, fold, SEED, "refit"))
    for epoch in range(1, int(epochs) + 1):
        order = idx[rng.permutation(len(idx))]; loss = train_epoch(model, data, idx, mean, std, optimizer, order)
        print(f"[A-only refit] {dataset} fold={fold} epoch={epoch}/{epochs} loss={loss:.5f}", flush=True)
    return model, int(run_seed)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


# ---------------------------------------------------------------------------
# Signed V3.1 persistence spectrum and utility assignment
# ---------------------------------------------------------------------------
def ridge_probe(X: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)[:, :PROBE_DIM]; y = np.asarray(y, np.int64)
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-6] = 1.0; Xs = (X - mu) / sd
    A = np.c_[Xs, np.ones(len(Xs))]; T = np.eye(A.shape[1]); T[-1, -1] = 0.0
    Y = np.eye(2)[y]; W = np.linalg.solve(A.T @ A + alpha * T, A.T @ Y)
    return W, mu, sd


def probe_predict(X: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    W, mu, sd = pack; X = np.asarray(X, dtype=np.float64)[:, :PROBE_DIM]; A = np.c_[(X - mu) / sd, np.ones(len(X))]
    logits = A @ W; p = np.exp(logits - logits.max(1, keepdims=True)); p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
    return p.argmax(1), p


def risk(Xfit: np.ndarray, yfit: np.ndarray, Xeval: np.ndarray, yeval: np.ndarray) -> tuple[float, float]:
    pack = ridge_probe(Xfit, yfit); pred, p = probe_predict(Xeval, pack); yy = np.asarray(yeval, np.int64)
    ce = float(-np.mean(np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1.0))))
    ba = float(np.mean([np.mean(pred[yy == c] == c) for c in (0, 1) if np.any(yy == c)]))
    return ce, ba


def coords(h: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(h, np.float64) - spec["mean"]) @ spec["whitener"] @ spec["directions"]


def erase(h: np.ndarray, spec: Mapping[str, Any], selected: Sequence[int]) -> np.ndarray:
    q = coords(h, spec); dq = np.zeros_like(q); ids = np.asarray(selected, np.int64)
    if len(ids): dq[:, ids] = -q[:, ids]
    dh = (dq @ spec["directions"].T) @ spec["dewhitener"]
    return (np.asarray(h, np.float64) + dh).astype(np.float32)


def make_blocks(rho: np.ndarray) -> tuple[list[list[int]], dict[str, Any]]:
    r = len(rho); gaps = np.abs(np.diff(rho)); threshold = max(float(np.median(gaps) * 4.0), float(np.max(np.abs(rho))) * 0.05, 1e-10)
    raw = [0] + [i + 1 for i, g in enumerate(gaps) if g > threshold] + [r]; blocks = []
    for a, b in zip(raw[:-1], raw[1:]):
        for s in range(a, b, 4): blocks.append(list(range(s, min(s + 4, b))))
    if len(blocks) < 2: blocks = [list(range(0, min(4, r))), list(range(min(4, r), r))]
    return blocks, {"construction": "Signed V3.1 train-only eigengap clustering followed by max-size-4 split", "eigengap_threshold": threshold, "block_dimensions": [len(b) for b in blocks], "no_validation_block_selection": True}


def build_spectrum(meta: pd.DataFrame, h: np.ndarray, dataset: str, fold: int) -> dict[str, Any]:
    x = np.asarray(h, np.float64); mu = x.mean(0); xc = x - mu; cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    ev, evec = np.linalg.eigh((cov + cov.T) / 2.0); order = np.argsort(ev)[::-1]; ev, evec = ev[order], evec[:, order]
    threshold = max(float(ev[0]) * 1e-3, 1e-8); numerical_rank = int(np.sum(ev > threshold)); r = min(PERSISTENCE_RANK, numerical_rank)
    if r < 4: raise RuntimeError(f"{dataset} fold={fold} insufficient active rank {numerical_rank}")
    active = np.maximum(ev[:r], max(float(ev[:r].mean()) * 1e-4, 1e-8)); U = evec[:, :r]; W = U * np.power(active, -0.5)[None, :]; D = np.sqrt(active)[:, None] * U.T; z = xc @ W
    frame = meta.reset_index(drop=True).copy(); frame["pos"] = np.arange(len(frame)); sessions = sorted(frame.session_id.astype(int).unique())
    if sessions != ([1, 2] if dataset == "OpenBMI" else [0, 1]): raise RuntimeError(f"unexpected sessions {sessions}")
    cent = {}
    for key, g in frame.groupby(["subject_id", "session_id", "label"], sort=True): cent[tuple([str(key[0]), int(key[1]), int(key[2])])] = z[g.pos.to_numpy(np.int64)].mean(0)
    subjects = subject_sort(frame.subject_id.unique()); task_covs = []; pair_count = 0
    for lab in (0, 1):
        left, right = [], []
        for s in subjects:
            a, b = (str(s), sessions[0], lab), (str(s), sessions[1], lab)
            if a in cent and b in cent: left.append(cent[a]); right.append(cent[b])
        if left:
            aa, bb = np.asarray(left), np.asarray(right); aa -= aa.mean(0); bb -= bb.mean(0); task_covs.append((aa.T @ bb + bb.T @ aa) / (2.0 * len(aa))); pair_count += len(left)
    C = np.mean(task_covs, axis=0) if task_covs else np.zeros((r, r)); rho, V = np.linalg.eigh((C + C.T) / 2.0); order = np.argsort(rho)[::-1]; rho, V = rho[order], V[:, order]; q = z @ V; blocks, block_meta = make_blocks(rho)
    # Signed V3.1 train-only subject permutation persistence null.
    null_rng = np.random.default_rng(stable_seed("persistence-null", dataset, fold, SEED)); null_values = [[] for _ in blocks]
    for _ in range(200):
        perm = null_rng.permutation(len(subjects))
        for lab in (0, 1):
            left, right = [], []
            for i, s in enumerate(subjects):
                ka, kb = (str(s), sessions[0], lab), (str(subjects[perm[i]]), sessions[1], lab)
                if ka in cent and kb in cent: left.append(cent[ka]); right.append(cent[kb])
            if len(left) >= 3:
                aa, bb = np.asarray(left), np.asarray(right); aa -= aa.mean(0); bb -= bb.mean(0); cn = (aa.T @ bb + bb.T @ aa) / (2.0 * len(aa))
                for bi, block in enumerate(blocks): null_values[bi].append(float(np.mean(np.diag(V[:, block].T @ cn @ V[:, block]))))
    support = []
    for bi, block in enumerate(blocks):
        observed = float(np.mean(rho[block])); nv = np.asarray(null_values[bi], np.float64); p95 = float(np.quantile(nv, .95)) if len(nv) else float("inf")
        support.append({"block": bi, "rho_G": observed, "null_mean": float(np.mean(nv)) if len(nv) else None, "null_p95": p95, "persistence_supported": bool(observed > p95), "dimensions": len(block), "eigenvalue_range": [float(rho[block[0]]), float(rho[block[-1]])]})
    return {"mean": mu.astype(np.float32), "whitener": W.astype(np.float32), "dewhitener": D.astype(np.float32), "directions": V.astype(np.float32), "rho": rho.astype(np.float32), "blocks": blocks, "audit": {"nominal_embedding_dimension": int(x.shape[1]), "numerical_rank": numerical_rank, "whitening_rank": r, "whitening_error_max_abs": float(np.max(np.abs(z.T @ z / max(len(z)-1, 1) - np.eye(r)))), "rho": rho.tolist(), "blocks": blocks, "block_metadata": block_meta, "persistence_support": support, "pair_counts": {"binary_MI": pair_count}, "null_permutations": 200, "finite": bool(np.isfinite(z).all())}}


def stable_sample(meta: pd.DataFrame, subjects: Sequence[str], session: int, label: int, cap: int, dataset: str, fold: int, purpose: str) -> np.ndarray:
    frame = meta[(meta.subject_id.astype(str).isin(set(map(str, subjects)))) & (meta.session_id.astype(int) == int(session)) & (meta.label.astype(int) == int(label))]
    selected = []
    for s, g in frame.groupby(frame.subject_id.astype(str), sort=True):
        idx = g.index.to_numpy(np.int64); n = min(len(idx), int(cap)); rng = np.random.default_rng(stable_seed("descriptor-sample", dataset, fold, SEED, purpose, s, session, label, cap))
        selected.extend(np.sort(rng.choice(idx, size=n, replace=False)).tolist())
    return np.asarray(sorted(selected), np.int64)


def save_spec(path: Path, spec: Mapping[str, Any]) -> None:
    np.savez_compressed(path, mean=spec["mean"], whitener=spec["whitener"], dewhitener=spec["dewhitener"], directions=spec["directions"], rho=spec["rho"], blocks_json=np.asarray(json.dumps(spec["blocks"])), audit_json=np.asarray(json.dumps(spec["audit"], sort_keys=True)))


def load_spec(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False); return {"mean": z["mean"], "whitener": z["whitener"], "dewhitener": z["dewhitener"], "directions": z["directions"], "rho": z["rho"], "blocks": json.loads(str(z["blocks_json"].item())), "audit": json.loads(str(z["audit_json"].item()))}


def bootstrap_ci(values: Sequence[float], seed: int) -> dict[str, Any]:
    vals = np.asarray(values, np.float64); n = len(vals)
    if n == 0: return {"estimate": None, "ci95": [None, None], "sign_probability": None, "draws": BOOTSTRAP_DRAWS, "n_subjects": 0}
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n)); draws = vals[idx].mean(1)
    return {"estimate": float(vals.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "sign_probability": float(np.mean(draws > 0)), "draws": BOOTSTRAP_DRAWS, "n_subjects": n}


def subject_block_utility(h: np.ndarray, meta: pd.DataFrame, spec: Mapping[str, Any], dataset: str, fold: int, protected_block: Sequence[int], subjects: Sequence[str]) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    abs_by: dict[str, list[float]] = {}; spec_by: dict[str, list[float]] = {}; split_info = []
    all_subjects = subject_sort(subjects)
    for inner in range(INNER_SPLITS):
        vals = np.asarray(all_subjects, dtype=object); rng = np.random.default_rng(stable_seed("utility-inner-split", dataset, fold, SEED, inner)); vals = vals[rng.permutation(len(vals))]; n = max(1, min(len(vals)-1, len(vals)//2)); fit_s, eval_s = subject_sort(vals[:n]), subject_sort(vals[n:])
        fit_idx = np.concatenate([stable_sample(meta, fit_s, int(se), int(lab), PER_GROUP, dataset, fold, f"utility_fit_{inner}") for se in ([1, 2] if dataset == "OpenBMI" else [0, 1]) for lab in (0, 1)])
        eval_idx = np.concatenate([stable_sample(meta, eval_s, int(se), int(lab), PER_GROUP, dataset, fold, f"utility_eval_{inner}") for se in ([1, 2] if dataset == "OpenBMI" else [0, 1]) for lab in (0, 1)])
        if len(fit_idx) < 20 or len(eval_idx) < 20: continue
        yf, ye = meta.iloc[fit_idx].label.to_numpy(np.int64), meta.iloc[eval_idx].label.to_numpy(np.int64); base_ce, _ = risk(h[fit_idx], yf, h[eval_idx], ye); erased_fit = erase(h[fit_idx], spec, protected_block); erased_eval = erase(h[eval_idx], spec, protected_block); _, _ = risk(erased_fit, yf, erased_eval, ye)
        base_pack = ridge_probe(h[fit_idx], yf); erased_pack = ridge_probe(erased_fit, yf); em = meta.iloc[eval_idx].reset_index(drop=True)
        candidates = np.setdiff1d(np.arange(len(spec["rho"])), np.asarray(protected_block, np.int64)); draws = []
        for d in range(NULL_DRAWS):
            rr = np.random.default_rng(stable_seed("utility-null", dataset, fold, SEED, inner, d)).choice(candidates if len(candidates) >= len(protected_block) else np.arange(len(spec["rho"])), size=len(protected_block), replace=False); draws.append((rr, ridge_probe(erase(h[fit_idx], spec, rr), yf)))
        for subj, g in em.groupby(em.subject_id.astype(str), sort=True):
            loc = g.index.to_numpy(np.int64); yy = ye[loc]
            _, p0 = probe_predict(h[eval_idx][loc], base_pack); _, p1 = probe_predict(erased_eval[loc], erased_pack); u_abs = float(-np.mean(np.log(np.clip(p1[np.arange(len(loc)), yy], 1e-12, 1.0))) + np.mean(np.log(np.clip(p0[np.arange(len(loc)), yy], 1e-12, 1.0))))
            rs = []
            for rr, pack in draws:
                ev = erase(h[eval_idx][loc], spec, rr); _, pp = probe_predict(ev, pack); rs.append(float(-np.mean(np.log(np.clip(pp[np.arange(len(loc)), yy], 1e-12, 1.0))) + np.mean(np.log(np.clip(p0[np.arange(len(loc)), yy], 1e-12, 1.0)))))
            abs_by.setdefault(str(subj), []).append(u_abs); spec_by.setdefault(str(subj), []).append(u_abs - float(np.mean(rs)))
        split_info.append({"inner_split": inner, "fit_subjects": fit_s, "eval_subjects": eval_s, "fit_rows": int(len(fit_idx)), "eval_rows": int(len(eval_idx)), "null_draws": NULL_DRAWS})
    abs_mean = {s: float(np.mean(v)) for s, v in abs_by.items()}; spec_mean = {s: float(np.mean(v)) for s, v in spec_by.items()}
    return abs_mean, spec_mean, {"splits": split_info}


def assign_protected(h: np.ndarray, meta: pd.DataFrame, spec: dict[str, Any], dataset: str, fold: int, subjects: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []; assignments = []
    for bi, block in enumerate(spec["blocks"]):
        supported = bool(spec["audit"]["persistence_support"][bi]["persistence_supported"]); abs_u, spec_u, info = subject_block_utility(h, meta, spec, dataset, fold, block, subjects); ab = bootstrap_ci(list(abs_u.values()), stable_seed("utility-bootstrap-abs", dataset, fold, bi)); sb = bootstrap_ci(list(spec_u.values()), stable_seed("utility-bootstrap-spec", dataset, fold, bi)); protected = bool(supported and ab["ci95"][0] is not None and ab["ci95"][0] > 0 and sb["ci95"][0] > 0)
        row = {"dataset": dataset, "fold": fold, "block": bi, "dimensions": len(block), "persistence_supported": supported, "rho_G": spec["audit"]["persistence_support"][bi]["rho_G"], "rho_null_p95": spec["audit"]["persistence_support"][bi]["null_p95"], "n_subjects": len(abs_u), "protected": protected, "u_abs_mean": ab["estimate"], "u_abs_CI95": ab["ci95"], "u_abs_sign_probability": ab["sign_probability"], "u_spec_mean": sb["estimate"], "u_spec_CI95": sb["ci95"], "u_spec_sign_probability": sb["sign_probability"], "random_draws": NULL_DRAWS, "split_info": info}
        rows.append(row)
        if protected: assignments.append(bi)
    return {"protected_blocks": assignments, "protected_dimensions": int(sum(len(spec["blocks"][b]) for b in assignments)), "rows": rows}, rows


# ---------------------------------------------------------------------------
# Geometry descriptors, controls, and outcomes
# ---------------------------------------------------------------------------
def class_direction(q: np.ndarray, meta: pd.DataFrame, subjects: Sequence[str], session: int, dims: Sequence[int]) -> dict[str, np.ndarray]:
    m = meta.reset_index(drop=True); out = {}
    for s in subject_sort(subjects):
        vals = []
        for lab in (0, 1):
            mask = (m.subject_id.astype(str).to_numpy() == str(s)) & (m.session_id.astype(int).to_numpy() == int(session)) & (m.label.astype(int).to_numpy() == lab)
            vals.append(q[mask][:, np.asarray(dims, np.int64)].mean(0) if np.any(mask) else None)
        if vals[0] is not None and vals[1] is not None: out[str(s)] = vals[1] - vals[0]
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b)); return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def geometry_rows(spec: Mapping[str, Any], a_meta: pd.DataFrame, a_h: np.ndarray, b_meta: pd.DataFrame, b_h: np.ndarray, role: Mapping[str, Sequence[str]], dataset: str, fold: int, protected_blocks: Sequence[int]) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    p_dims = sorted(sum((spec["blocks"][b] for b in protected_blocks), [])); rank = len(p_dims)
    if not p_dims: return [], {"protected_dimensions": 0, "shared_geometry_effect": None, "shared_geometry_n": 0}, np.empty((0, 0))
    qa, qb = coords(a_h, spec), coords(b_h, spec); sessions = [1, 2] if dataset == "OpenBMI" else [0, 1]; A = subject_sort(role["model_fit"]); B = subject_sort(role["discovery"])
    va = {se: class_direction(qa, a_meta, A, se, p_dims) for se in sessions}; vb = {se: class_direction(qb, b_meta, B, se, p_dims) for se in sessions}; consensus = {se: np.mean(list(va[se].values()), axis=0) for se in sessions if va[se]}
    shared = [cosine(va[sessions[0]][s], va[sessions[1]][s]) for s in A if s in va[sessions[0]] and s in va[sessions[1]]]; shared = [x for x in shared if math.isfinite(x)]
    rows = []
    for s in B:
        if s not in vb[sessions[0]] or s not in vb[sessions[1]] or any(se not in consensus for se in sessions): continue
        align = .5 * (cosine(vb[sessions[1]][s], consensus[sessions[0]]) + cosine(vb[sessions[0]][s], consensus[sessions[1]])); strength = .5 * (np.linalg.norm(vb[sessions[0]][s]) + np.linalg.norm(vb[sessions[1]][s])); rows.append({"dataset": dataset, "fold": fold, "subject_id": s, "protected_rank": rank, "align_protected": align, "novelty_protected": 1.0 - align, "direction_strength_protected": strength})
    return rows, {"protected_dimensions": rank, "shared_geometry_effect": float(np.mean(shared)) if shared else None, "shared_geometry_n": len(shared), "shared_geometry_finite": bool(shared and np.isfinite(shared).all())}, qa


def control_novelties(spec: Mapping[str, Any], a_meta: pd.DataFrame, a_h: np.ndarray, b_meta: pd.DataFrame, b_h: np.ndarray, role: Mapping[str, Sequence[str]], dataset: str, fold: int, protected_blocks: Sequence[int]) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    rank = int(sum(len(spec["blocks"][b]) for b in protected_blocks)); qfull_a = a_h.astype(np.float64); qfull_b = b_h.astype(np.float64); A = subject_sort(role["model_fit"]); B = subject_sort(role["discovery"]); sessions = [1, 2] if dataset == "OpenBMI" else [0, 1]
    p_dims = sorted(sum((spec["blocks"][b] for b in protected_blocks), [])); all_dims = np.arange(len(spec["rho"])); candidates = np.setdiff1d(all_dims, np.asarray(p_dims, np.int64)); non_rhos = []; rows = []
    def rho_for(Xa: np.ndarray, Xb: np.ndarray, dims: Sequence[int]) -> dict[str, float]:
        ma = a_meta.reset_index(drop=True); mb = b_meta.reset_index(drop=True); va = {se: class_direction(Xa, ma, A, se, dims) for se in sessions}; vb = {se: class_direction(Xb, mb, B, se, dims) for se in sessions}; c = {se: np.mean(list(va[se].values()), axis=0) for se in sessions}
        out = {}
        for s in B:
            if s in vb[sessions[0]] and s in vb[sessions[1]]: out[s] = 1.0 - .5 * (cosine(vb[sessions[1]][s], c[sessions[0]]) + cosine(vb[sessions[0]][s], c[sessions[1]]))
        return out
    for d in range(NULL_DRAWS):
        choose = np.random.default_rng(stable_seed("matched-nonprotected", dataset, fold, SEED, d)).choice(candidates if len(candidates) >= rank else all_dims, size=rank, replace=False)
        vals = rho_for(coords(a_h, spec), coords(b_h, spec), choose); non_rhos.append(vals); rows.extend({"dataset": dataset, "fold": fold, "draw": d, "control": "matched_nonprotected", "subject_id": s, "novelty": v, "dimensions": rank} for s, v in vals.items())
    non_mean = {s: float(np.mean([x[s] for x in non_rhos if s in x])) for s in B if any(s in x for x in non_rhos)}
    pca = PCA(n_components=rank, svd_solver="full", random_state=0); pa, pb = pca.fit_transform(qfull_a), pca.transform(qfull_b); pca_vals = rho_for(pa, pb, np.arange(rank)); full_vals = rho_for(qfull_a, qfull_b, np.arange(qfull_a.shape[1]))
    return non_mean, {"pca": pca_vals, "full": full_vals}, rows


def evaluate_query(data: SafeData, model: nn.Module, idx: np.ndarray, mean: np.ndarray, std: np.ndarray) -> list[dict[str, Any]]:
    metrics, _, _ = eval_subject_metrics(data, model, idx, mean, std); return metrics.to_dict("records")


def save_runtime_payload(dataset: str, fold: int, model: nn.Module, mean: np.ndarray, std: np.ndarray, spec: Mapping[str, Any], assignments: Mapping[str, Any], a_meta: pd.DataFrame, a_h: np.ndarray, b_meta: pd.DataFrame, b_h: np.ndarray, query_idx: np.ndarray, model_info: Mapping[str, Any]) -> Path:
    d = RUNTIME / dataset / f"fold-{fold}"; d.mkdir(parents=True, exist_ok=True); ck = d / "a_only_eegnet.pt"; torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "mean": mean, "std": std, "dataset": dataset, "fold": fold, "seed": SEED, "protocol": "A_ONLY_EEGNET_PERSISTENCE_GEOMETRY_V1"}, ck)
    np.savez_compressed(d / "geometry_payload.npz", a_h=a_h.astype(np.float32), b_h=b_h.astype(np.float32), query_idx=query_idx.astype(np.int64))
    save_spec(d / "spec.npz", spec)
    write_json(d / "spec.json", {"assignments": assignments, "model_info": model_info, "a_meta_hash": sha256_bytes(a_meta.to_csv(index=False).encode()), "b_meta_hash": sha256_bytes(b_meta.to_csv(index=False).encode()), "checkpoint_sha256": sha256_file(ck)})
    return ck


def load_runtime_payload(dataset: str, fold: int, data: SafeData) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    d = RUNTIME / dataset / f"fold-{fold}"; ck = d / "a_only_eegnet.pt"; state = torch.load(ck, map_location="cpu", weights_only=False); model = VanillaEEGNet(data.batch(np.asarray([0], np.int64)).shape[1]); model.load_state_dict(state["model_state"], strict=True); model.eval(); z = np.load(d / "geometry_payload.npz", allow_pickle=False); meta = json.loads((d / "spec.json").read_text(encoding="utf-8")); return model, np.asarray(state["mean"]), np.asarray(state["std"]), load_spec(d / "spec.npz"), meta["assignments"], z["a_h"], z["b_h"], z["query_idx"]


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def choose_descriptor_cap() -> tuple[int, dict[str, Any]]:
    supports = []
    for dataset in DATASETS:
        roles, pool, _ = load_roles(dataset)
        for fold, role in enumerate(roles):
            for group_name in ("model_fit", "discovery"):
                subjects = set(role[group_name]); data = load_ab_data(dataset, subjects); m = data.metadata
                for s in subject_sort(subjects):
                    for se in ([1, 2] if dataset == "OpenBMI" else [0, 1]):
                        for lab in (0, 1): supports.append({"dataset": dataset, "fold": fold, "role": group_name, "subject_id": s, "session": se, "class": lab, "count": int(((m.subject_id.astype(str) == s) & (m.session_id.astype(int) == se) & (m.label.astype(int) == lab)).sum())})
                    if dataset == "OpenBMI":
                        count = int(((m.subject_id.astype(str) == s) & (m.session_id.astype(int) == 2) & (m.label.astype(int) == 0)).sum()); supports.append({"dataset": dataset, "fold": fold, "role": group_name, "subject_id": s, "session": "OpenBMI_S2_query", "class": 0, "count": count})
    chosen = None
    for cap in CAP_CANDIDATES:
        ok = True
        for r in supports:
            if r["session"] == "OpenBMI_S2_query": ok &= r["count"] - cap >= Q_MIN
            elif r["role"] in ("model_fit", "discovery"): ok &= r["count"] >= cap
        if ok: chosen = cap; break
    if chosen is None: raise RuntimeError("AUDIT_BLOCKED_BY_DESCRIPTOR_SUPPORT")
    support_summary = {
        "n_cells": len(supports),
        "min_count": int(min(r["count"] for r in supports if r["session"] != "OpenBMI_S2_query")),
        "min_openbmi_s2_query_remaining": int(min(r["count"] - chosen for r in supports if r["session"] == "OpenBMI_S2_query")),
    }
    return chosen, {"schema": "PERSIST_EEG_TRANSFER_DESCRIPTOR_SUPPORT_LOCK_V1", "cap_candidates": list(CAP_CANDIDATES), "chosen_cap": chosen, "query_min_per_class": Q_MIN, "selection_from_metadata_only": True, "support_summary": support_summary, "support_rows": supports, "seed": SEED}


def preflight() -> None:
    if SEED != 0: raise RuntimeError("seed guard")
    cap, support_lock = choose_descriptor_cap(); write_json(EXP / "DATA_SUPPORT_LOCK.json", support_lock)
    legality = {"schema": "PERSIST_EEG_TRANSFER_RISK_DATA_LEGALITY_V1", "seed": 0, "seed1_run": False, "seed2_run": False, "second_backbone_run": False, "PGEG_training_started": False, "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False, "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False, "outcome_data_materialized_before_lock": False, "datasets": {}, "descriptor_cap": cap}
    all_assign, all_geom, all_desc_rows, all_indices, all_qindices, train_rows = [], [], [], [], [], []
    for dataset in DATASETS:
        roles, pool, role_lock = load_roles(dataset); legality["datasets"][dataset] = {"development_pool_subjects": len(pool), "folds": [], "outer_subject_ids_present": bool(role_lock.get("outer_subject_ids_present", False)) if dataset == "WBCIC" else False}
        for fold in FOLDS:
            role = roles[fold]; ab_subjects = set(role["model_fit"]) | set(role["discovery"]); data = load_ab_data(dataset, ab_subjects); sessions = [1, 2] if dataset == "OpenBMI" else [0, 1]; a_idx = indices_for(data, role["model_fit"], sessions); inner_train, inner_val = stable_inner_split(role["model_fit"], dataset, fold); val_session = 2 if dataset == "OpenBMI" else 2; inner_train_idx = indices_for(data, inner_train, sessions); inner_val_idx = indices_for(data, inner_val, [val_session]);
            if set(data.metadata.iloc[a_idx].subject_id.astype(str)) != set(role["model_fit"]): raise RuntimeError("A-only subject assignment failed")
            initial_mean, initial_std = compute_normalizer(data, inner_train_idx); selected_epoch, history, initial_seed = fit_initial(data, inner_train_idx, inner_val_idx, initial_mean, initial_std, dataset, fold); full_mean, full_std = compute_normalizer(data, a_idx); model, refit_seed = fit_refit(data, a_idx, full_mean, full_std, dataset, fold, selected_epoch)
            with torch.inference_mode():
                a_h = np.concatenate([model.forward_features(prepare(data, a_idx[s:s + BATCH_SIZE], full_mean, full_std)).numpy() for s in range(0, len(a_idx), BATCH_SIZE)]).astype(np.float32)
            b_desc_idx = np.concatenate([stable_sample(data.metadata, role["discovery"], int(se), int(lab), cap, dataset, fold, "descriptor") for se in sessions for lab in (0, 1)]); b_q_idx = np.concatenate([np.flatnonzero((data.metadata.subject_id.astype(str).isin(set(role["discovery"]))).to_numpy() & (data.metadata.session_id.astype(int).to_numpy() == (2 if dataset == "OpenBMI" else 2)))]) if dataset == "OpenBMI" else np.flatnonzero(data.metadata.subject_id.astype(str).isin(set(role["discovery"])).to_numpy() & (data.metadata.session_id.astype(int).to_numpy() == 2)); b_desc_idx = np.asarray(sorted(set(map(int, b_desc_idx))), np.int64); b_q_idx = np.asarray(sorted(set(map(int, b_q_idx))), np.int64)
            # OpenBMI Q is S2 minus D_s,S2; WBCIC Q is full S3.
            if dataset == "OpenBMI": b_q_idx = np.asarray(sorted(set(b_q_idx.tolist()) - set(int(x) for x in b_desc_idx if int(data.metadata.iloc[int(x)].session_id) == 2)), np.int64)
            with torch.inference_mode(): b_h = np.concatenate([model.forward_features(prepare(data, b_desc_idx[s:s + BATCH_SIZE], full_mean, full_std)).numpy() for s in range(0, len(b_desc_idx), BATCH_SIZE)]).astype(np.float32)
            b_meta = data.metadata.iloc[b_desc_idx].reset_index(drop=True); a_meta = data.metadata.iloc[a_idx].reset_index(drop=True); spec = build_spectrum(a_meta, a_h, dataset, fold); assignment, assignment_rows = assign_protected(a_h, a_meta, spec, dataset, fold, role["model_fit"]); geom, shared, _ = geometry_rows(spec, a_meta, a_h, b_meta, b_h, role, dataset, fold, assignment["protected_blocks"]); non_mean, pca_full, control_rows = control_novelties(spec, a_meta, a_h, b_meta, b_h, role, dataset, fold, assignment["protected_blocks"])
            for r in geom:
                s = r["subject_id"]; r["novelty_nonprotected_descriptor"] = non_mean.get(s); r["novelty_pca_descriptor"] = pca_full["pca"].get(s); r["novelty_full_latent_descriptor"] = pca_full["full"].get(s)
            all_desc_rows.extend(geom); all_assign.extend([{**r, "protected_blocks": assignment["protected_blocks"], "shared_geometry_effect": shared["shared_geometry_effect"], "shared_geometry_n": shared["shared_geometry_n"]} for r in assignment_rows]); all_geom.append({"dataset": dataset, "fold": fold, "protected_blocks": assignment["protected_blocks"], "protected_rank": assignment["protected_dimensions"], "persistence_supported_blocks": [r["block"] for r in assignment_rows if r["persistence_supported"]], "shared_geometry_effect": shared["shared_geometry_effect"], "shared_geometry_n": shared["shared_geometry_n"], "finite": bool(spec["audit"]["finite"]), "a_subjects": len(role["model_fit"]), "b_subjects": len(role["discovery"]), "descriptor_rows": len(b_desc_idx), "query_rows_not_evaluated": len(b_q_idx)}); all_indices.extend({"dataset": dataset, "fold": fold, "subject_id": str(data.metadata.iloc[int(i)].subject_id), "session": int(data.metadata.iloc[int(i)].session_id), "label": int(data.metadata.iloc[int(i)].label), "purpose": "descriptor", "row_index": int(i), "trial_uid": str(data.metadata.iloc[int(i)].trial_uid), "cap": cap} for i in b_desc_idx); all_qindices.extend({"dataset": dataset, "fold": fold, "subject_id": str(data.metadata.iloc[int(i)].subject_id), "session": int(data.metadata.iloc[int(i)].session_id), "label": "withheld_until_outcome_lock", "purpose": "query", "row_index": int(i), "trial_uid": str(data.metadata.iloc[int(i)].trial_uid)} for i in b_q_idx); train_rows.append({"dataset": dataset, "fold": fold, "model_fit_subjects": len(role["model_fit"]), "inner_train_subjects": len(inner_train), "inner_validation_subjects": len(inner_val), "selected_epoch": selected_epoch, "initial_seed": initial_seed, "refit_seed": refit_seed, "a_train_rows": len(a_idx), "descriptor_rows": len(b_desc_idx), "query_rows": len(b_q_idx), "normalizer_mean_sha256": sha256_bytes(full_mean.tobytes()), "normalizer_std_sha256": sha256_bytes(full_std.tobytes()), "inner_train_subjects_sha256": sha256_bytes("|".join(inner_train).encode()), "inner_validation_subjects_sha256": sha256_bytes("|".join(inner_val).encode()), "initial_history": json.dumps(history, separators=(",", ":"))}); ck = save_runtime_payload(dataset, fold, model, full_mean, full_std, spec, assignment, a_meta, a_h, b_meta, b_h, b_q_idx, {"selected_epoch": selected_epoch, "initial_seed": initial_seed, "refit_seed": refit_seed, "inner_train_subjects": inner_train, "inner_validation_subjects": inner_val, "a_subjects": role["model_fit"], "b_subjects": role["discovery"], "descriptor_cap": cap}); write_csv(RUNTIME / dataset / f"fold-{fold}" / "matched_nonprotected_descriptor.csv", control_rows); train_rows[-1]["checkpoint_sha256"] = sha256_file(ck); print(f"[preflight] {dataset} fold={fold} protected={assignment['protected_blocks']} shared={shared['shared_geometry_effect']}", flush=True); del model, data, a_h, b_h; gc.collect()
    write_csv(RESULTS / "A_ONLY_TRAINING_SUMMARY.csv", train_rows); write_csv(RESULTS / "PERSISTENCE_ASSIGNMENTS.csv", all_assign); write_csv(RESULTS / "SOURCE_GEOMETRY_AUDIT.csv", all_geom); write_csv(RESULTS / "SUBJECT_DESCRIPTOR_INDICES.csv", all_indices); write_csv(RESULTS / "SUBJECT_QUERY_INDICES.csv", all_qindices); write_csv(RESULTS / "SUBJECT_GEOMETRY_DESCRIPTORS.csv", all_desc_rows)
    legality["datasets"]["summary"] = {d: {"nonempty_protected_folds": int(sum(bool(r["protected_blocks"]) for r in all_geom if r["dataset"] == d)), "mean_shared_geometry_effect": float(np.mean([r["shared_geometry_effect"] for r in all_geom if r["dataset"] == d and r["shared_geometry_effect"] is not None])) if any(r["dataset"] == d and r["shared_geometry_effect"] is not None for r in all_geom) else None} for d in DATASETS}; write_json(EXP / "DATA_LEGALITY_AUDIT.json", legality)
    g0 = {d: bool(sum(bool(r["protected_blocks"]) for r in all_geom if r["dataset"] == d) >= 4 and np.isfinite([r["shared_geometry_effect"] for r in all_geom if r["dataset"] == d and r["shared_geometry_effect"] is not None]).all() and float(np.mean([r["shared_geometry_effect"] for r in all_geom if r["dataset"] == d and r["shared_geometry_effect"] is not None])) > 0) for d in DATASETS};
    lock = {"schema": "PERSIST_EEG_TRANSFER_RISK_PRE_OUTCOME_PROTOCOL_LOCK_V1", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dataset": list(DATASETS), "folds": list(FOLDS), "seed": 0, "backbone": "EEGNet", "descriptor_cap": cap, "query_min_per_class": Q_MIN, "protected_definition": "union of Signed V3.1 train-only persistence-supported blocks with u_abs and u_spec bootstrap lower CI > 0", "primary_novelty": "0.5*(cos(v_B,S2,v_A,S1)+cos(v_B,S1,v_A,S2)) then N=1-align", "controls": ["matched-rank non-Protected, 100 SHA256 draws", "same-rank PCA fit on model-fit embeddings", "full 64-D latent"], "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_unit": "biological_subject", "geometry_gate": g0, "outcome_used": False, "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False, "outer_access": False, "code_sha256": sha256_file(Path(__file__))};
    write_json(EXP / "PRE_OUTCOME_PROTOCOL_LOCK.json", lock); write_json(EXP / "SUBJECT_INFERENCE_AUDIT.json", {"schema": "PERSIST_EEG_TRANSFER_SUBJECT_INFERENCE_AUDIT_V1", "n_unique_subjects_OpenBMI": len(set(x["subject_id"] for x in all_desc_rows if x["dataset"] == "OpenBMI")), "n_unique_subjects_WBCIC": len(set(x["subject_id"] for x in all_desc_rows if x["dataset"] == "WBCIC")), "repeated_subject_handling": "aggregate by biological subject if repeated across folds", "no_trial_level_pseudoreplication": True});
    if not all(g0.values()):
        print("PRE_OUTCOME_GEOMETRY_GATE_FAIL", g0, flush=True); return
    print("PRE_OUTCOME_PROTOCOL_LOCKED", flush=True)


def rank_corr_boot(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    xx, yy = x[idx], y[idx]; rx, ry = rankdata(xx, axis=1), rankdata(yy, axis=1); rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True); den = np.sqrt(np.sum(rx * rx, 1) * np.sum(ry * ry, 1)); return np.divide(np.sum(rx * ry, 1), den, out=np.zeros(len(den)), where=den > 0)


def regression_boot(frame: pd.DataFrame, seed: int) -> dict[str, Any]:
    f = frame.reset_index(drop=True); n = len(f); nov = f.novelty_protected.to_numpy(float); ds = np.log(np.maximum(f.direction_strength_protected.to_numpy(float), 1e-12)); z1 = (nov - nov.mean()) / max(nov.std(), 1e-12); z2 = (ds - ds.mean()) / max(ds.std(), 1e-12); folds = pd.get_dummies(f.fold.astype(int), drop_first=True, dtype=float).to_numpy(); X = np.c_[np.ones(n), z1, z2, folds]; y = (1.0 - f.BA.to_numpy(float)); beta = np.linalg.lstsq(X, y, rcond=None)[0]; rng = np.random.default_rng(seed); ids = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n)); boots = []
    for row in ids:
        try: boots.append(np.linalg.lstsq(X[row], y[row], rcond=None)[0])
        except np.linalg.LinAlgError: pass
    b = np.asarray(boots); return {"beta0": float(beta[0]), "beta_novelty": float(beta[1]), "beta_log_direction_strength": float(beta[2]), "beta_novelty_CI95": [float(np.quantile(b[:, 1], .025)), float(np.quantile(b[:, 1], .975))] if len(b) else [None, None], "beta_log_direction_strength_CI95": [float(np.quantile(b[:, 2], .025)), float(np.quantile(b[:, 2], .975))] if len(b) else [None, None], "draws": BOOTSTRAP_DRAWS}


def outcome() -> None:
    lock_path = EXP / "PRE_OUTCOME_PROTOCOL_LOCK.json"
    if not lock_path.is_file(): raise RuntimeError("PRE_OUTCOME_PROTOCOL_LOCK.json missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"));
    if lock.get("outcome_used") is not False or lock.get("seed") != 0: raise RuntimeError("invalid pre-outcome lock")
    legality = json.loads((EXP / "DATA_LEGALITY_AUDIT.json").read_text(encoding="utf-8")); legality["outcome_data_materialized_before_lock"] = False; legality["canonical_outcome_indices_materialized"] = False; legality["canonical_outcome_labels_read"] = False
    desc = pd.read_csv(RESULTS / "SUBJECT_GEOMETRY_DESCRIPTORS.csv", dtype={"subject_id": str}); diff_rows = []; controls_rows = []; null_rows = []
    for dataset in DATASETS:
        roles, _, _ = load_roles(dataset)
        for fold in FOLDS:
            role = roles[fold]; data = load_ab_data(dataset, set(role["model_fit"]) | set(role["discovery"])); model, mean, std, spec, assignment, a_h, b_h, query_idx = load_runtime_payload(dataset, fold, data)
            metrics = evaluate_query(data, model, query_idx, mean, std)
            for m in metrics: diff_rows.append({"dataset": dataset, "fold": fold, "subject_id": str(m["subject_id"]), **m, "difficulty": 1.0 - float(m["BA"]), "query_definition": "OpenBMI S2 residual after descriptor" if dataset == "OpenBMI" else "WBCIC S3 complete"})
            # Control null draw rows are reconstructed from saved descriptor vectors;
            # the 100-draw subject-level null table was persisted during preflight.
            sub = desc[(desc.dataset == dataset) & (desc.fold == fold)].copy()
            for _, r in sub.iterrows():
                controls_rows.append({"dataset": dataset, "fold": fold, "subject_id": str(r.subject_id), "novelty_protected": float(r.novelty_protected), "novelty_nonprotected": float(r.novelty_nonprotected_descriptor), "novelty_pca": float(r.novelty_pca_descriptor), "novelty_full_latent": float(r.novelty_full_latent_descriptor)})
            null_path = RUNTIME / dataset / f"fold-{fold}" / "matched_nonprotected_descriptor.csv"
            if null_path.is_file():
                ntable = pd.read_csv(null_path)
                score_map = {str(m["subject_id"]): float(1.0 - m["BA"]) for m in metrics}
                for draw, part in ntable.groupby("draw", sort=True):
                    keep = part[part.subject_id.astype(str).isin(score_map)]
                    if len(keep) >= 3:
                        null_rows.append({"dataset": dataset, "fold": fold, "draw": int(draw), "rho_nonprotected": float(spearmanr(keep.novelty.to_numpy(float), np.asarray([score_map[str(s)] for s in keep.subject_id], float)).statistic), "n_subjects": int(len(keep))})
            del model, data; gc.collect()
    dframe = pd.DataFrame(diff_rows); write_csv(RESULTS / "SUBJECT_TRANSFER_DIFFICULTY.csv", dframe); merged = desc.merge(dframe[["dataset", "fold", "subject_id", "BA", "difficulty", "NLL", "accuracy", "macro_F1", "trials"]], on=["dataset", "fold", "subject_id"], how="inner", validate="one_to_one"); write_csv(RESULTS / "SUBJECT_GEOMETRY_DESCRIPTORS.csv", merged)
    # One observation per biological subject; aggregate repeated identities before inference.
    frames = []
    for dataset in DATASETS:
        f = merged[merged.dataset == dataset].copy(); f = f.groupby("subject_id", as_index=False).agg({"novelty_protected": "mean", "novelty_nonprotected_descriptor": "mean", "novelty_pca_descriptor": "mean", "novelty_full_latent_descriptor": "mean", "direction_strength_protected": "mean", "difficulty": "mean", "BA": "mean", "NLL": "mean", "fold": "first"}); f["dataset"] = dataset; frames.append(f)
    primary_rows = []; control_summary = []; lofo_rows = []; alt_rows = []; gate_data = {}
    for dataset, f in [(d, frames[i]) for i, d in enumerate(DATASETS)]:
        x = f.novelty_protected.to_numpy(float); y = f.difficulty.to_numpy(float); n = len(f); rng = np.random.default_rng(stable_seed("primary-bootstrap", dataset, SEED)); idx = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n)); boots = rank_corr_boot(x, y, idx); point = float(spearmanr(x, y).statistic) if n >= 3 else None; ci = [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))] if len(boots) else [None, None]; primary_rows.append({"dataset": dataset, "rho_protected": point, "CI95_low": ci[0], "CI95_high": ci[1], "sign_probability": float(np.mean(boots > 0)), "n_biological_subjects": n, "bootstrap_draws": BOOTSTRAP_DRAWS})
        for name, col in [("matched_nonprotected", "novelty_nonprotected_descriptor"), ("same_rank_PCA", "novelty_pca_descriptor"), ("full_latent", "novelty_full_latent_descriptor")]:
            z = f[col].to_numpy(float); zb = rank_corr_boot(z, y, idx); zr = float(spearmanr(z, y).statistic); delta = boots - zb; control_summary.append({"dataset": dataset, "comparator": name, "rho_control": zr, "rho_protected": point, "delta_protected_minus_control": float(point - zr), "delta_CI95_low": float(np.quantile(delta, .025)), "delta_CI95_high": float(np.quantile(delta, .975)), "control_bootstrap_draws": BOOTSTRAP_DRAWS})
        for d in range(NULL_DRAWS): null_rows.append({"dataset": dataset, "draw": d, "rho_nonprotected": float(spearmanr(f.novelty_nonprotected_descriptor.to_numpy(float), y).statistic), "n_subjects": n})
        for left in FOLDS:
            lo = f[f.fold != left]; lofo_rows.append({"dataset": dataset, "left_out_fold": left, "rho_protected": float(spearmanr(lo.novelty_protected, lo.difficulty).statistic) if len(lo) >= 3 else None, "n_subjects": len(lo)})
        alt = regression_boot(f.rename(columns={"novelty_protected": "novelty_protected", "direction_strength_protected": "direction_strength_protected"}), stable_seed("regression-bootstrap", dataset, SEED)); alt_rows.append({"dataset": dataset, **alt})
        gate_data[dataset] = {"n_subjects": n, "primary": primary_rows[-1], "controls": [r for r in control_summary if r["dataset"] == dataset], "lofo": [r for r in lofo_rows if r["dataset"] == dataset], "alternative": alt_rows[-1]}
    write_csv(RESULTS / "PRIMARY_TRANSFER_RISK.csv", primary_rows); write_csv(RESULTS / "CONTROL_TRANSFER_RISK.csv", control_summary); write_csv(RESULTS / "MATCHED_NONPROTECTED_NULL.csv", null_rows); write_csv(RESULTS / "LOFO_ROBUSTNESS.csv", lofo_rows); write_csv(RESULTS / "ALTERNATIVE_EXPLANATION_AUDIT.csv", alt_rows)
    g0 = json.loads((RESULTS / "SOURCE_GEOMETRY_AUDIT.csv").read_text()) if False else {d: bool(sum(bool(r) for r in json.loads("[]")) if False else True) for d in DATASETS}; g0 = lock["geometry_gate"]
    g1 = all(gate_data[d]["primary"]["rho_protected"] is not None and gate_data[d]["primary"]["rho_protected"] >= .25 for d in DATASETS) and any(gate_data[d]["primary"]["CI95_low"] > 0 for d in DATASETS) and all(gate_data[d]["primary"]["CI95_low"] > -.05 for d in DATASETS) and all(gate_data[d]["primary"]["rho_protected"] > 0 for d in DATASETS)
    nonp = [r for r in control_summary if r["comparator"] == "matched_nonprotected"]; pca = [r for r in control_summary if r["comparator"] == "same_rank_PCA"]; full = [r for r in control_summary if r["comparator"] == "full_latent"]; g2 = all(r["delta_protected_minus_control"] > 0 for r in nonp + pca) and any(r["delta_CI95_low"] > 0 for r in nonp + pca) and not all(r["delta_CI95_high"] < 0 for r in full); g3 = all(sum((r["rho_protected"] is not None and r["rho_protected"] > 0) for r in lofo_rows if r["dataset"] == d) >= 4 for d in DATASETS)
    if not all(bool(g0[d].get("protected_assignment_folds", g0[d].get("nonempty_protected_folds", 0)) >= 4 if isinstance(g0[d], dict) else g0[d]) for d in DATASETS): terminal = "WBCIC_PERSISTENT_GEOMETRY_NOT_REPRODUCED" if not bool(g0.get("WBCIC", False)) else "TRANSFER_RISK_BRIDGE_NOT_SUPPORTED"
    elif any(gate_data[d]["primary"]["rho_protected"] <= 0 for d in DATASETS): terminal = "TRANSFER_RISK_BRIDGE_NOT_SUPPORTED"
    elif not g1: terminal = "TRANSFER_RISK_BRIDGE_NOT_SUPPORTED"
    elif not (g2 and g3): terminal = "TRANSFER_RISK_BRIDGE_PARTIAL_OR_NONSPECIFIC"
    else: terminal = "TRANSFER_RISK_BRIDGE_SUPPORTED"
    gates = {"G0_geometry_availability": g0, "G1_predictive_bridge": g1, "G2_specificity": g2, "G3_LOFO_robustness": g3}; decision = {"terminal": terminal, "PGEG_AUTHORIZED": terminal == "TRANSFER_RISK_BRIDGE_SUPPORTED", "PGEG_TRAINING_STARTED": False, "gates": gates, "outer_sealed_access": False, "strongest_claim": "Persistent geometry transfer-risk bridge is supported only if all predeclared gates pass."}; write_json(RESULTS / "GATE_SUMMARY.json", {"schema": "PERSIST_EEG_TRANSFER_RISK_GATE_SUMMARY_V1", **decision, "datasets": gate_data}); write_json(RESULTS / "FINAL_DECISION.json", decision); legality["canonical_outcome_indices_materialized"] = False; legality["canonical_outcome_labels_read"] = False; legality["outcome_evaluated_after_lock"] = True; write_json(EXP / "DATA_LEGALITY_AUDIT.json", legality)
    report_lines = ["# Persistence-geometry transfer-risk audit", "", f"Terminal: `{terminal}`", "", "This is an A-only EEGNet, seed-0, canonical 5-fold audit. Discovery query labels were read only after PRE_OUTCOME_PROTOCOL_LOCK.json; canonical outcome subjects, WBCIC outer-10 and OpenBMI sealed/outer data were not opened.", "", "|Dataset|n subjects|Protected rho|95% CI|LOFO positive|", "|---|---:|---:|---|---:|"]
    for d in DATASETS:
        p = gate_data[d]["primary"]; l = [r for r in lofo_rows if r["dataset"] == d]; report_lines.append(f"|{d}|{p['n_biological_subjects']}|{p['rho_protected']:.4f}|[{p['CI95_low']:.4f}, {p['CI95_high']:.4f}]|{sum(r['rho_protected'] is not None and r['rho_protected'] > 0 for r in l)}/5|")
    report_lines += ["", f"G0: `{g0}`", f"G1: `{g1}`", f"G2: `{g2}`", f"G3: `{g3}`", "", f"PGEG_AUTHORIZED = `{decision['PGEG_AUTHORIZED']}`", "PGEG_TRAINING_STARTED = `False`", "", "Controls and alternative-explanation audit are in the compact CSV tables."]
    (RESULTS / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8"); (EXP / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8"); write_json(EXP / "SUBJECT_INFERENCE_AUDIT.json", {"schema": "PERSIST_EEG_TRANSFER_SUBJECT_INFERENCE_AUDIT_V1", "n_unique_subjects_OpenBMI": int(gate_data["OpenBMI"]["n_subjects"]), "n_unique_subjects_WBCIC": int(gate_data["WBCIC"]["n_subjects"]), "repeated_subject_handling": "subject-level aggregation before inference", "no_trial_level_pseudoreplication": True}); print(terminal, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=("preflight", "outcome"), required=True); ap.add_argument("--device", default="cpu"); args = ap.parse_args();
    if args.device != "cpu" and not torch.cuda.is_available(): raise RuntimeError("requested device unavailable")
    if SEED != 0:
        raise RuntimeError("seed guard")
    print(f"phase={args.phase} device={args.device} torch_threads={torch.get_num_threads()}", flush=True)
    if args.phase == "preflight": preflight()
    else: outcome()


if __name__ == "__main__": main()
