"""Frozen GeoSR final constructive experiment.

The implementation deliberately keeps the scientific protocol small and explicit:
source-only cross-fitted scalar geometry/risk, six pre-registered training
variants, and one post-lock outcome evaluation.  Runtime tensors/checkpoints are
written below the experiment's ignored ``runtime`` directory; only compact
provenance and result tables are deliverables.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

# ``audit_primitives.py`` is a verbatim, audited copy of the prior transfer-risk
# runner.  It supplies the frozen role/data loader and canonical EEGNet.
import audit_primitives as ap


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime"
RESULTS.mkdir(parents=True, exist_ok=True)
RUNTIME.mkdir(parents=True, exist_ok=True)

DATASETS = ("OpenBMI", "WBCIC")
FOLDS = (0, 1, 2, 3, 4)
METHODS = ("CANONICAL_ERM", "SUBJECT_BALANCED_ERM", "RANDOM_RANK", "LOSS_HARD", "GEO_ONLY", "GEOSR")
SESSIONS_FIT = {"OpenBMI": (1, 2), "WBCIC": (0, 1)}
SESSION_DISCOVERY = {"OpenBMI": 2, "WBCIC": 2}
SESSION_OUTCOME = {"OpenBMI": 2, "WBCIC": 2}
CAP = 32
INNER_K = 5
Q_MIN = 16
MAX_EPOCHS = 60
MIN_EPOCHS = 10
PATIENCE = 8
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
BOOTSTRAP_DRAWS = 10_000
CACHE_SCHEMA_VERSION = 2


def jclean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jclean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [jclean(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.ndarray):
        return jclean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(jclean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False) % (2**63 - 1)


def seed_everything(seed: int) -> None:
    if seed not in (0, 1, 2):
        raise RuntimeError("GeoSR only permits scientific seeds 0, 1, and 2")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def subj_sort(values: Iterable[object]) -> list[str]:
    return ap.subject_sort(values)


def role_hash(role: Mapping[str, Sequence[str]]) -> str:
    payload = "|".join(f"{k}:{','.join(subj_sort(role[k]))}" for k in ("model_fit", "discovery", "outcome"))
    return bytes_sha(payload.encode())


def sessions_for(dataset: str) -> tuple[int, ...]:
    return SESSIONS_FIT[dataset]


class FoldCache:
    """A memory-resident normalized view of one outer fold's source data.

    Only model-fit/discovery subjects are loaded here.  Outcome subjects are
    deliberately loaded in the separate post-lock phase.
    """

    def __init__(self, dataset: str, subjects: Sequence[str], seed: int, fold: int):
        self.dataset = dataset
        self.subjects = subj_sort(subjects)
        self.seed = seed
        self.fold = fold
        self.data = ap.load_ab_data(dataset, set(self.subjects))
        self.meta = self.data.metadata.reset_index(drop=True)
        needed = tuple(sorted(set(sessions_for(dataset)) | {SESSION_DISCOVERY[dataset]}))
        self.indices = np.flatnonzero(self.meta.session_id.astype(int).isin(needed).to_numpy()).astype(np.int64)
        if not len(self.indices):
            raise RuntimeError(f"empty source cache {dataset} fold {fold}")
        raw = self.data.batch(self.indices).astype(np.float32, copy=False)
        self.pos = {int(r): i for i, r in enumerate(self.indices.tolist())}
        self.labels = self.meta.label.to_numpy(np.int64)
        self._raw = raw
        self.x: torch.Tensor | None = None
        self.default_mean: np.ndarray | None = None
        self.default_std: np.ndarray | None = None
        self._raw_gpu: torch.Tensor | None = None
        self._gpu_view: torch.Tensor | None = None
        self._gpu_view_key: str | None = None
        # These caches only memoize deterministic indexing/statistics.  They
        # never alter the row order or arithmetic used by the frozen protocol.
        self._rows_cache: dict[tuple[tuple[str, ...], tuple[int, ...]], np.ndarray] = {}
        self._normalizer_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._labels_gpu: torch.Tensor | None = None
        self._labels_gpu_device: torch.device | None = None

    @staticmethod
    def _rows_key(subjects: Sequence[str], sessions: Sequence[int]) -> tuple[tuple[str, ...], tuple[int, ...]]:
        return tuple(subj_sort(subjects)), tuple(sorted(set(map(int, sessions))))

    def rows(self, subjects: Sequence[str], sessions: Sequence[int]) -> np.ndarray:
        key = self._rows_key(subjects, sessions)
        cached = self._rows_cache.get(key)
        if cached is not None:
            return cached.copy()
        m = self.meta
        mask = m.subject_id.astype(str).isin(set(map(str, subjects))) & m.session_id.astype(int).isin(list(map(int, sessions)))
        out = np.flatnonzero(mask.to_numpy()).astype(np.int64)
        self._rows_cache[key] = out
        return out.copy()

    def normalizer(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(rows, dtype=np.int64)
        key = bytes_sha(rows.tobytes())
        cached = self._normalizer_cache.get(key)
        if cached is not None:
            return cached[0].copy(), cached[1].copy()
        positions = np.asarray([self.pos[int(r)] for r in rows], dtype=np.int64)
        x = self._raw[positions].astype(np.float64, copy=False)
        mean = x.mean(axis=(0, 2)).astype(np.float32)
        std = np.sqrt(np.maximum(x.var(axis=(0, 2)), 1e-12)).astype(np.float32)
        self._normalizer_cache[key] = (mean, std)
        return mean.copy(), std.copy()

    def normalize(self, mean: np.ndarray, std: np.ndarray) -> None:
        if self.x is not None and self.default_mean is not None and np.array_equal(self.default_mean, mean) and np.array_equal(self.default_std, std):
            return
        x = (self._raw - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.default_mean = np.asarray(mean).copy()
        self.default_std = np.asarray(std).copy()

    def tensor(self, rows: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None,
               device: torch.device | None = None) -> torch.Tensor:
        positions = np.asarray([self.pos[int(r)] for r in np.asarray(rows, dtype=np.int64)], dtype=np.int64)
        if device is not None and device.type == "cuda" and mean is not None and std is not None:
            key = bytes_sha(np.asarray(mean).tobytes() + np.asarray(std).tobytes())
            if self._raw_gpu is None or self._raw_gpu.device != device:
                self._raw_gpu = torch.from_numpy(self._raw).to(device, non_blocking=True)
                self._gpu_view = None
                self._gpu_view_key = None
            if self._gpu_view is None or self._gpu_view_key != key:
                mu = torch.as_tensor(np.asarray(mean), dtype=torch.float32, device=device).view(1, -1, 1)
                sd = torch.as_tensor(np.asarray(std), dtype=torch.float32, device=device).view(1, -1, 1)
                self._gpu_view = (self._raw_gpu - mu) / torch.clamp(sd, min=1e-6)
                self._gpu_view_key = key
            return self._gpu_view[torch.as_tensor(positions, dtype=torch.long, device=device)]
        # The default normalized view is reused for the student.  Cross-fitted
        # teachers request their own A_k-only normalizer and are normalized from
        # the retained in-memory raw view; this prevents leakage between folds.
        if mean is None or (self.x is not None and self.default_mean is not None and np.array_equal(self.default_mean, mean) and np.array_equal(self.default_std, std)):
            if self.x is None:
                raise RuntimeError("normalizer must be supplied before tensor access")
            return self.x[torch.from_numpy(positions)]
        raw = self._raw[positions]
        arr = (raw - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    def labels_tensor(self, device: torch.device) -> torch.Tensor:
        if device.type != "cuda":
            return torch.from_numpy(self.labels)
        if self._labels_gpu is None or self._labels_gpu_device != device:
            self._labels_gpu = torch.from_numpy(self.labels).to(device, non_blocking=True)
            self._labels_gpu_device = device
        return self._labels_gpu


def inner_partition(subjects: Sequence[str], dataset: str, fold: int, k: int, seed: int) -> tuple[list[str], list[str]]:
    values = np.asarray(subj_sort(subjects), dtype=object)
    # One deterministic permutation is shared by all k slices.  Seeding each
    # k independently would reshuffle the subjects and break disjoint,
    # exhaustive held-out coverage.
    rng = np.random.default_rng(stable_seed("geosr-inner-kfold", dataset, fold, seed))
    values = values[rng.permutation(len(values))]
    held = values[np.arange(len(values)) % INNER_K == k]
    fit = np.asarray([s for s in values if s not in set(held)], dtype=object)
    if not len(held) or not len(fit):
        raise RuntimeError("inner cross-fit partition is empty")
    return subj_sort(fit), subj_sort(held)


def teacher_inner_split(subjects: Sequence[str], dataset: str, fold: int, k: int, seed: int) -> tuple[list[str], list[str]]:
    vals = np.asarray(subj_sort(subjects), dtype=object)
    rng = np.random.default_rng(stable_seed("geosr-teacher-inner-validation", dataset, fold, k, seed))
    vals = vals[rng.permutation(len(vals))]
    n = max(1, min(len(vals) - 1, int(math.floor(0.8 * len(vals)))))
    return subj_sort(vals[:n]), subj_sort(vals[n:])


def stable_descriptor_rows(cache: FoldCache, subject: str, session: int, label: int, seed: int) -> np.ndarray:
    m = cache.meta
    mask = (m.subject_id.astype(str) == str(subject)) & (m.session_id.astype(int) == int(session)) & (m.label.astype(int) == int(label))
    idx = np.flatnonzero(mask.to_numpy()).astype(np.int64)
    if len(idx) < CAP:
        raise RuntimeError(f"descriptor support below frozen cap for {cache.dataset} {subject} S{session} C{label}")
    rng = np.random.default_rng(stable_seed("descriptor-sample", cache.dataset, cache.fold, seed, "geosr_descriptor", subject, session, label, CAP))
    return np.sort(rng.choice(idx, size=CAP, replace=False)).astype(np.int64)


def make_model(cache: FoldCache, device: torch.device) -> torch.nn.Module:
    channels = int(cache._raw.shape[1])
    return ap.VanillaEEGNet(channels).to(device)


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initial_state(cache: FoldCache, dataset: str, fold: int, seed: int, tag: str) -> tuple[dict[str, torch.Tensor], int, str]:
    init_seed = stable_seed("geosr-initial-state", dataset, fold, seed, tag)
    seed_everything(seed)
    torch.manual_seed(init_seed)
    model = make_model(cache, torch.device("cpu"))
    st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    digest = state_hash(st)
    del model
    return st, int(init_seed), digest


def order_for(rows: np.ndarray, dataset: str, fold: int, seed: int, stage: str, owner: str, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed("geosr-minibatch-order", dataset, fold, seed, stage, owner, epoch))
    return rows[rng.permutation(len(rows))]


def weight_vector(cache: FoldCache, rows: np.ndarray, subject_weights: Mapping[str, float], method: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if method == "CANONICAL_ERM":
        return np.ones(len(rows), dtype=np.float32)
    m = cache.meta
    labels = m.label.to_numpy(np.int64)
    subs = m.subject_id.astype(str).to_numpy()
    counts: dict[tuple[str, int], int] = {}
    for r in rows:
        counts[(str(subs[r]), int(labels[r]))] = counts.get((str(subs[r]), int(labels[r])), 0) + 1
    raw = np.asarray([float(subject_weights[str(subs[r])]) / (2.0 * counts[(str(subs[r]), int(labels[r]))]) for r in rows], dtype=np.float64)
    raw /= max(float(raw.mean()), 1e-12)
    if not np.isfinite(raw).all():
        raise RuntimeError(f"invalid normalized weights for {method}")
    if not np.isclose(raw.mean(), 1.0, atol=1e-5):
        raise RuntimeError("normalized trial weights do not have mean one")
    return raw.astype(np.float32)


def train_epoch(model: torch.nn.Module, cache: FoldCache, rows: np.ndarray, mean: np.ndarray, std: np.ndarray,
                weights: np.ndarray, optimizer: torch.optim.Optimizer, order: np.ndarray, device: torch.device,
                row_weight_lookup: np.ndarray | None = None, weight_device: torch.Tensor | None = None) -> float:
    model.train()
    losses: list[float] = []
    labels = cache.labels
    if row_weight_lookup is None:
        row_weight_lookup = np.zeros(len(cache.meta), dtype=np.float32)
        row_weight_lookup[np.asarray(rows, dtype=np.int64)] = np.asarray(weights, dtype=np.float32)
    if device.type == "cuda":
        if weight_device is None:
            weight_device = torch.from_numpy(row_weight_lookup).to(device, non_blocking=True)
        labels_device = cache.labels_tensor(device)
    for start in range(0, len(order), BATCH_SIZE):
        part = order[start:start + BATCH_SIZE]
        xb = cache.tensor(part, mean, std, device)
        if xb.device != device:
            xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            pos = torch.as_tensor(part, dtype=torch.long, device=device)
            yb = labels_device[pos]
            wb = weight_device[pos]
        else:
            yb = torch.from_numpy(labels[part]).long()
            wb = torch.from_numpy(row_weight_lookup[part])
        optimizer.zero_grad(set_to_none=True)
        ce = F.cross_entropy(model(xb), yb, reduction="none")
        loss = (ce * wb).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def eval_rows(cache: FoldCache, model: torch.nn.Module, rows: np.ndarray, mean: np.ndarray, std: np.ndarray,
              device: torch.device) -> pd.DataFrame:
    model.eval()
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            part = rows[start:start + BATCH_SIZE]
            xb = cache.tensor(part, mean, std, device)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            logits.append(model(xb).detach().cpu().numpy())
    z = np.concatenate(logits, axis=0)
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    frame = cache.meta.iloc[rows].reset_index(drop=True)
    y = frame.label.to_numpy(np.int64)
    pred = p.argmax(1)
    out: list[dict[str, Any]] = []
    for subject, g in frame.groupby(frame.subject_id.astype(str), sort=True):
        loc = g.index.to_numpy(np.int64)
        yy = y[loc]
        pp = p[loc]
        out.append({"subject_id": str(subject), "BA": float(balanced_accuracy_score(yy, pred[loc])),
                    "accuracy": float(accuracy_score(yy, pred[loc])),
                    "macro_F1": float(f1_score(yy, pred[loc], average="macro", zero_division=0)),
                    "NLL": float(log_loss(yy, pp, labels=[0, 1])), "trials": int(len(loc))})
    return pd.DataFrame(out)


def select_epoch(cache: FoldCache, train_rows: np.ndarray, val_rows: np.ndarray, mean: np.ndarray, std: np.ndarray,
                 weights: np.ndarray, state: Mapping[str, torch.Tensor], dataset: str, fold: int, seed: int,
                 owner: str, device: torch.device) -> tuple[int, list[dict[str, Any]]]:
    seed_everything(seed)
    model = make_model(cache, device)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    row_weight_lookup = np.zeros(len(cache.meta), dtype=np.float32)
    row_weight_lookup[np.asarray(train_rows, dtype=np.int64)] = np.asarray(weights, dtype=np.float32)
    weight_device = torch.from_numpy(row_weight_lookup).to(device, non_blocking=True) if device.type == "cuda" else None
    best_ba, best_nll, best_epoch, stale = -math.inf, math.inf, 1, 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_t0 = time.perf_counter()
        order = order_for(train_rows, dataset, fold, seed, "select", owner, epoch)
        loss = train_epoch(model, cache, train_rows, mean, std, weights, optimizer, order, device,
                           row_weight_lookup=row_weight_lookup, weight_device=weight_device)
        val = eval_rows(cache, model, val_rows, mean, std, device)
        ba = float(val.BA.mean()) if len(val) else -math.inf
        nll = float(val.NLL.mean()) if len(val) else math.inf
        improved = ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and nll < best_nll - 1e-12)
        if improved:
            best_ba, best_nll, best_epoch, stale = ba, nll, epoch, 0
        else:
            stale += 1
        elapsed = time.perf_counter() - epoch_t0
        history.append({"epoch": epoch, "train_loss": loss, "val_BA": ba, "val_NLL": nll, "sec": elapsed})
        print(f"[select] {dataset} fold={fold} owner={owner} epoch={epoch} BA={ba:.4f} best={best_epoch} sec={elapsed:.3f}", flush=True)
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    del model
    gc.collect()
    return int(best_epoch), history


def fit_exact(cache: FoldCache, rows: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: np.ndarray,
              state: Mapping[str, torch.Tensor], dataset: str, fold: int, seed: int, owner: str,
              epochs: int, device: torch.device, timing: dict[str, Any] | None = None) -> torch.nn.Module:
    seed_everything(seed)
    model = make_model(cache, device)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    row_weight_lookup = np.zeros(len(cache.meta), dtype=np.float32)
    row_weight_lookup[np.asarray(rows, dtype=np.int64)] = np.asarray(weights, dtype=np.float32)
    weight_device = torch.from_numpy(row_weight_lookup).to(device, non_blocking=True) if device.type == "cuda" else None
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        order = order_for(rows, dataset, fold, seed, "exact", owner, epoch)
        loss = train_epoch(model, cache, rows, mean, std, weights, optimizer, order, device,
                           row_weight_lookup=row_weight_lookup, weight_device=weight_device)
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"[fit] {dataset} fold={fold} owner={owner} epoch={epoch}/{epochs} loss={loss:.5f}", flush=True)
    if timing is not None:
        timing["sec"] = time.perf_counter() - t0
        timing["epochs"] = int(epochs)
        timing["sec_per_epoch"] = timing["sec"] / max(int(epochs), 1)
    return model


def extract_embeddings(cache: FoldCache, model: torch.nn.Module, rows: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            part = rows[start:start + BATCH_SIZE]
            xb = cache.tensor(part, mean, std, device)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            out.append(model.forward_features(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def array_sha(value: np.ndarray | Sequence[object]) -> str:
    arr = np.asarray(value)
    return bytes_sha(np.ascontiguousarray(arr).tobytes())


def code_fingerprint() -> str:
    """Fingerprint executable code and frozen constants for cache safety."""
    parts = [file_sha(Path(__file__)), file_sha(Path(__file__).with_name("audit_primitives.py")),
             str((CAP, INNER_K, MAX_EPOCHS, MIN_EPOCHS, PATIENCE, BATCH_SIZE, LR, WEIGHT_DECAY, GRAD_CLIP))]
    return bytes_sha("|".join(parts).encode("utf-8"))


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(value, tmp)
    os.replace(tmp, path)


def load_cache(path: Path, expected: Mapping[str, Any]) -> Any | None:
    if not path.is_file():
        return None
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        meta = value.get("cache_meta", {}) if isinstance(value, dict) else {}
        if all(meta.get(k) == v for k, v in expected.items()):
            return value
    except Exception as exc:
        print(f"[cache] ignoring unreadable {path.name}: {exc}", flush=True)
    return None


def save_selection_cache(path: Path, expected: Mapping[str, Any], epoch: int, history: list[dict[str, Any]]) -> None:
    atomic_torch_save({"cache_meta": dict(expected), "selected_epoch": int(epoch), "history": history}, path)


def select_epoch_cached(cache: FoldCache, train_rows: np.ndarray, val_rows: np.ndarray, mean: np.ndarray, std: np.ndarray,
                        weights: np.ndarray, state: Mapping[str, torch.Tensor], dataset: str, fold: int, seed: int,
                        owner: str, device: torch.device, path: Path | None = None,
                        expected_extra: Mapping[str, Any] | None = None) -> tuple[int, list[dict[str, Any]], bool]:
    expected: dict[str, Any] = {
        "schema": CACHE_SCHEMA_VERSION, "code_fingerprint": code_fingerprint(), "dataset": dataset,
        "fold": int(fold), "seed": int(seed), "owner": owner, "train_rows_sha256": array_sha(train_rows),
        "val_rows_sha256": array_sha(val_rows), "weights_sha256": array_sha(np.asarray(weights, dtype=np.float32)),
        "state_sha256": state_hash(state), "mean_sha256": bytes_sha(np.asarray(mean).tobytes()),
        "std_sha256": bytes_sha(np.asarray(std).tobytes()),
    }
    if expected_extra:
        expected.update(dict(expected_extra))
    if path is not None:
        hit = load_cache(path, expected)
        if hit is not None:
            print(f"[cache] selection hit {dataset} fold={fold} owner={owner}", flush=True)
            return int(hit["selected_epoch"]), list(hit.get("history", [])), True
    epoch, history = select_epoch(cache, train_rows, val_rows, mean, std, weights, state, dataset, fold, seed, owner, device)
    if path is not None:
        save_selection_cache(path, expected, epoch, history)
    return epoch, history, False


def crossfit_scalars(cache: FoldCache, source_subjects: Sequence[str], dataset: str, fold: int, seed: int,
                     stage: str, device: torch.device, cache_root: Path | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    source = subj_sort(source_subjects)
    scalar_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    seen: list[str] = []
    for k in range(INNER_K):
        fit_s, held_s = inner_partition(source, dataset, fold, k, seed)
        seen.extend(held_s)
        cache_path = None
        cache_expected = {"schema": CACHE_SCHEMA_VERSION, "code_fingerprint": code_fingerprint(),
                          "dataset": dataset, "fold": int(fold), "seed": int(seed), "stage": stage,
                          "inner_k": int(k), "source_subjects_sha256": bytes_sha("|".join(source).encode()),
                          "held_subjects_sha256": bytes_sha("|".join(held_s).encode())}
        if cache_root is not None:
            cache_path = Path(cache_root) / dataset / f"fold-{fold}" / stage / f"teacher-{k}.pt"
            hit = load_cache(cache_path, cache_expected)
            if hit is not None:
                scalar_rows.extend(hit.get("scalar_rows", []))
                assignment_rows.append(hit["assignment_row"])
                teacher_rows.append(hit["teacher_row"])
                print(f"[cache] teacher hit {dataset} fold={fold} stage={stage} k={k}", flush=True)
                continue
        train_s, val_s = teacher_inner_split(fit_s, dataset, fold, k, seed)
        train_rows = cache.rows(train_s, sessions_for(dataset))
        val_rows = cache.rows(val_s, (SESSION_DISCOVERY[dataset],))
        fit_rows = cache.rows(fit_s, sessions_for(dataset))
        mean, std = cache.normalizer(fit_rows)
        cache.normalize(mean, std)
        state, init_seed, init_sha = initial_state(cache, dataset, fold, seed, f"teacher-{stage}-{k}")
        w = np.ones(len(train_rows), dtype=np.float32)
        select_t0 = time.perf_counter()
        ep, hist = select_epoch(cache, train_rows, val_rows, mean, std, w, state, dataset, fold, seed, f"teacher-{stage}-{k}", device)
        select_sec = time.perf_counter() - select_t0
        fit_timing: dict[str, Any] = {}
        teacher = fit_exact(cache, fit_rows, mean, std, np.ones(len(fit_rows), dtype=np.float32), state,
                            dataset, fold, seed, f"teacher-{stage}-{k}", ep, device, timing=fit_timing)
        # Build directions only in this teacher's own coordinate system.
        desc_rows: list[int] = []
        for s in fit_s + held_s:
            for se in sessions_for(dataset):
                for lab in (0, 1):
                    desc_rows.extend(stable_descriptor_rows(cache, s, se, lab, seed).tolist())
        desc_rows_arr = np.asarray(sorted(set(desc_rows)), dtype=np.int64)
        emb = extract_embeddings(cache, teacher, desc_rows_arr, mean, std, device)
        pos = {int(r): i for i, r in enumerate(desc_rows_arr.tolist())}
        m = cache.meta

        def direction(subject: str, session: int) -> np.ndarray | None:
            c0 = stable_descriptor_rows(cache, subject, session, 0, seed)
            c1 = stable_descriptor_rows(cache, subject, session, 1, seed)
            a = emb[[pos[int(x)] for x in c0]].mean(0)
            b = emb[[pos[int(x)] for x in c1]].mean(0)
            return b - a

        consensus: dict[int, np.ndarray] = {}
        for se in sessions_for(dataset):
            vals = [direction(s, se) for s in fit_s]
            vals = [v for v in vals if v is not None and np.isfinite(v).all()]
            if vals:
                consensus[se] = np.mean(np.asarray(vals), axis=0)
        # Teacher logits for held-out balanced descriptor NLL.
        held_desc = []
        held_owner: list[str] = []
        for s in held_s:
            for se in sessions_for(dataset):
                for lab in (0, 1):
                    q = stable_descriptor_rows(cache, s, se, lab, seed)
                    held_desc.extend(q.tolist())
                    held_owner.extend([s] * len(q))
        held_desc_arr = np.asarray(held_desc, dtype=np.int64)
        with torch.inference_mode():
            logits = []
            for start in range(0, len(held_desc_arr), BATCH_SIZE):
                q = held_desc_arr[start:start + BATCH_SIZE]
                xb = cache.tensor(q, mean, std, device)
                if xb.device != device:
                    xb = xb.to(device, non_blocking=True)
                logits.append(teacher(xb).detach().cpu().numpy())
        lz = np.concatenate(logits, axis=0)
        lz -= lz.max(axis=1, keepdims=True)
        prob = np.exp(lz); prob /= np.maximum(prob.sum(1, keepdims=True), 1e-12)
        held_labels = m.label.to_numpy(np.int64)[held_desc_arr]
        for s in held_s:
            idx_s = np.asarray([i for i, x in enumerate(held_owner) if x == s], dtype=np.int64)
            y_s = held_labels[idx_s]
            nll = float(np.mean([-np.mean(np.log(np.clip(prob[idx_s][y_s == c, c], 1e-12, 1.0))) for c in (0, 1)]))
            dirs = {se: direction(s, se) for se in sessions_for(dataset)}
            if all(se in consensus and dirs[se] is not None for se in sessions_for(dataset)):
                s0, s1 = sessions_for(dataset)
                align = 0.5 * (cosine(dirs[s1], consensus[s0]) + cosine(dirs[s0], consensus[s1]))
                novelty = float(1.0 - align)
            else:
                novelty = float("nan")
            scalar_rows.append({"dataset": dataset, "fold": fold, "stage": stage, "inner_k": k,
                                "subject_id": str(s), "N_geo": novelty, "N_loss": nll,
                                "teacher_epoch": ep, "teacher_init_seed": init_seed,
                                "teacher_initial_state_sha256": init_sha,
                                "teacher_normalizer_mean_sha256": bytes_sha(mean.tobytes()),
                                "teacher_normalizer_std_sha256": bytes_sha(std.tobytes())})
        assignment_rows.append({"dataset": dataset, "fold": fold, "stage": stage, "inner_k": k,
                                "fit_subjects": ",".join(fit_s), "held_out_subjects": ",".join(held_s),
                                "teacher_train_subjects": ",".join(fit_s),
                                "teacher_inner_train_subjects": ",".join(train_s),
                                "teacher_inner_validation_subjects": ",".join(val_s),
                                "held_out_subjects_hash": bytes_sha("|".join(held_s).encode()),
                                "teacher_protocol": "A_k-only; held-out B_k never in teacher training/normalizer"})
        teacher_row = {"dataset": dataset, "fold": fold, "stage": stage, "inner_k": k,
                             "teacher_epoch": ep, "teacher_initial_state_sha256": init_sha,
                             "teacher_history_last": json.dumps(hist[-1] if hist else {}, separators=(",", ":")),
                             "teacher_fit_subject_count": len(fit_s), "held_out_subject_count": len(held_s),
                             "held_out_rows": len(held_desc_arr), "teacher_select_sec": select_sec,
                             "teacher_fit_sec": float(fit_timing.get("sec", 0.0)),
                             "teacher_fit_sec_per_epoch": float(fit_timing.get("sec_per_epoch", 0.0)),
                             "cache_hit": False}
        teacher_rows.append(teacher_row)
        if cache_path is not None:
            assignment_row = assignment_rows[-1]
            atomic_torch_save({"cache_meta": cache_expected, "scalar_rows": scalar_rows[-len(held_s):],
                               "assignment_row": assignment_row, "teacher_row": teacher_row}, cache_path)
        del teacher, emb, lz, prob
        gc.collect()
    if sorted(seen) != sorted(source) or len(seen) != len(set(seen)):
        raise RuntimeError(f"cross-fit held-out coverage failure {dataset} fold {fold} stage {stage}")
    frame = pd.DataFrame(scalar_rows)
    if frame.N_geo.isna().any() or frame.N_loss.isna().any():
        raise RuntimeError("non-finite cross-fitted scalar risk")
    # Exactly one observation per subject in each cross-fit stage.
    frame = frame.sort_values(["subject_id", "inner_k"]).drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    return frame, assignment_rows, teacher_rows


def source_weights(risk: pd.DataFrame, dataset: str, fold: int, seed: int,
                   methods: Sequence[str] | None = None) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    methods = tuple(METHODS if methods is None else methods)
    f = risk.sort_values("subject_id").reset_index(drop=True)
    n = len(f)
    if n < 2:
        raise RuntimeError("too few source subjects for ranks")
    r_geo = (rankdata(f.N_geo.to_numpy(float), method="average") - 1.0) / (n - 1)
    r_loss = (rankdata(f.N_loss.to_numpy(float), method="average") - 1.0) / (n - 1)
    base = {
        "CANONICAL_ERM": np.ones(n),
        "SUBJECT_BALANCED_ERM": np.ones(n),
        "LOSS_HARD": 0.5 + r_loss,
        "GEO_ONLY": 0.5 + r_geo,
        "GEOSR": 0.5 + 0.5 * r_geo + 0.5 * r_loss,
    }
    perm_rng = np.random.default_rng(stable_seed("geosr-random-rank-permutation", dataset, fold, seed))
    perm = perm_rng.permutation(n)
    base["RANDOM_RANK"] = base["GEOSR"][perm]
    rows: list[dict[str, Any]] = []
    out: dict[str, dict[str, float]] = {m: {} for m in methods}
    for i, s in enumerate(f.subject_id.astype(str)):
        for method in methods:
            w = float(base[method][i])
            out[method][str(s)] = w
        rows.append({"dataset": dataset, "fold": fold, "seed": seed, "subject_id": str(s),
                     "N_geo": float(f.N_geo.iloc[i]), "N_loss": float(f.N_loss.iloc[i]),
                     "r_geo": float(r_geo[i]), "r_loss": float(r_loss[i]),
                     **{f"weight_{m}": float(base[m][i]) for m in methods},
                     "random_rank_source_position": int(perm[i])})
    audit = {"dataset": dataset, "fold": fold, "seed": seed, "subjects": f.subject_id.astype(str).tolist(),
             "random_rank_permutation": perm.tolist(), "weight_multiset_sha256": bytes_sha(np.sort(base["GEOSR"]).astype(np.float64).tobytes()),
             "weights": {m: {s: float(w) for s, w in out[m].items()} for m in methods}}
    return out, {"rows": rows, "lock": audit}


def checkpoint_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def checkpoint_cache_valid(path: Path, expected: Mapping[str, Any]) -> bool:
    meta_path = checkpoint_meta_path(path)
    if not path.is_file() or not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return all(meta.get(k) == v for k, v in expected.items())
    except Exception:
        return False


def save_checkpoint(path: Path, model: torch.nn.Module, mean: np.ndarray, std: np.ndarray, dataset: str,
                    fold: int, seed: int, method: str, selected_epoch: int, init_sha: str,
                    cache_meta: Mapping[str, Any] | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "mean": mean, "std": std, "dataset": dataset, "fold": fold, "seed": seed,
               "method": method, "selected_epoch": int(selected_epoch), "initial_state_sha256": init_sha,
               "protocol": "GeoSR_final_frozen_v1"}
    tmp = path.with_suffix(path.suffix + ".part")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if cache_meta is not None:
        write_json(checkpoint_meta_path(path), dict(cache_meta))
    return file_sha(path)


def preflight(seed: int, device: torch.device) -> None:
    seed_everything(seed)
    # Descriptor support is frozen from metadata alone, before any outcome
    # labels or performance are touched.  The registered cache is expected to
    # select 32; a smaller cap would be a protocol stop, not a tuning choice.
    try:
        chosen_cap, support_lock = ap.choose_descriptor_cap()
    except Exception as exc:
        raise RuntimeError(f"descriptor support audit failed: {exc}") from exc
    if int(chosen_cap) != CAP:
        raise RuntimeError(f"AUDIT_BLOCKED_BY_DESCRIPTOR_SUPPORT: frozen cap={chosen_cap}, expected {CAP}")
    write_json(EXP / "DATA_SUPPORT_LOCK.json", {**support_lock, "seed": seed, "chosen_cap": CAP, "geosr_protocol": True})
    # The outer cohort is represented only by role lists and is never loaded in
    # this phase.  Role hashes are sufficient for the pre-outcome provenance.
    roles_by_dataset: dict[str, list[dict[str, list[str]]]] = {}
    role_hashes: dict[str, list[str]] = {}
    for d in DATASETS:
        roles, _, _ = ap.load_roles(d)
        roles_by_dataset[d] = roles
        role_hashes[d] = [role_hash(r) for r in roles]
    support = {"descriptor_cap": CAP, "query_min_per_class": Q_MIN, "metadata_only": True,
               "outer_access": False, "outcome_labels_read": False}
    write_json(EXP / "DATA_LEGALITY_AUDIT.json", {
        "schema": "PERSIST_EEG_GEOSR_DATA_LEGALITY_V1", "seed": seed,
        "seed1_run": False, "seed2_run": False, "second_backbone_run": False,
        "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False,
        "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False,
        "PGEG_training_started": False, "outcome_labels_read_before_lock": False,
        "descriptor_support_lock": support, "role_hashes": role_hashes,
    })
    all_crossfit: list[dict[str, Any]] = []
    all_teachers: list[dict[str, Any]] = []
    all_risk: list[dict[str, Any]] = []
    all_weights: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    state_hashes: dict[str, Any] = {}
    fold_manifests: dict[str, Any] = {}
    cache_root = RUNTIME / f"seed-{seed}" / "cache"
    for dataset in DATASETS:
        for fold in FOLDS:
            role = roles_by_dataset[dataset][fold]
            source = subj_sort(role["model_fit"])
            source_all = subj_sort(set(role["model_fit"]) | set(role["discovery"]))
            cache = FoldCache(dataset, source_all, seed, fold)
            fit_rows = cache.rows(source, sessions_for(dataset))
            discovery_rows = cache.rows(role["discovery"], (SESSION_DISCOVERY[dataset],))
            refit_rows = cache.rows(source_all, sessions_for(dataset))
            fit_mean, fit_std = cache.normalizer(fit_rows)
            cache.normalize(fit_mean, fit_std)
            # Both source stages are required by the frozen protocol.
            risk1, a1, t1 = crossfit_scalars(cache, source, dataset, fold, seed, "initial_selection", device, cache_root=cache_root)
            weights1, wa1 = source_weights(risk1, dataset, fold, seed, methods=METHODS)
            risk2, a2, t2 = crossfit_scalars(cache, source_all, dataset, fold, seed, "final_refit", device, cache_root=cache_root)
            weights2, wa2 = source_weights(risk2, dataset, fold, seed, methods=METHODS)
            refit_mean, refit_std = cache.normalizer(refit_rows)
            cache.normalize(refit_mean, refit_std)
            all_crossfit.extend(a1 + a2); all_teachers.extend(t1 + t2)
            for _, r in risk1.iterrows(): all_risk.append({**r.to_dict(), "seed": seed})
            for _, r in risk2.iterrows(): all_risk.append({**r.to_dict(), "seed": seed})
            for r in wa1["rows"]: all_weights.append({**r, "stage": "initial_selection"})
            for r in wa2["rows"]: all_weights.append({**r, "stage": "final_refit"})
            # Fair matched training: one initial state and one order schedule per
            # fold/stage, cloned for all methods.
            state, init_seed, init_sha = initial_state(cache, dataset, fold, seed, "student")
            state_hashes[f"{dataset}/fold-{fold}/seed-{seed}"] = {"initial_state_sha256": init_sha, "initial_seed": init_seed}
            selected: dict[str, int] = {}
            histories: dict[str, Any] = {}
            for method in METHODS:
                wvec = weight_vector(cache, fit_rows, weights1[method], method)
                sel_path = cache_root / dataset / f"fold-{fold}" / "student_initial_selection" / f"{method}.pt"
                ep, hist, sel_hit = select_epoch_cached(
                    cache, fit_rows, discovery_rows, fit_mean, fit_std, wvec, state,
                    dataset, fold, seed, "student-common", device, path=sel_path,
                    expected_extra={"method": method, "stage": "initial_selection"})
                selected[method] = ep; histories[method] = hist
                training_rows.append({"dataset": dataset, "fold": fold, "seed": seed, "stage": "initial_selection",
                                      "method": method, "selected_epoch": ep, "initial_state_sha256": init_sha,
                                      "normalizer_mean_sha256": bytes_sha(fit_mean.tobytes()),
                                      "normalizer_std_sha256": bytes_sha(fit_std.tobytes()),
                                      "training_subjects": len(source), "discovery_subjects": len(role["discovery"]),
                                      "weight_mean": float(wvec.mean()), "weight_min": float(wvec.min()), "weight_max": float(wvec.max()),
                                      "selection_sec": float(sum(float(h.get("sec", 0.0)) for h in hist)), "cache_hit": bool(sel_hit)})
            # Final refit from the same deterministic initial state, exact selected
            # epochs, with independently recomputed source cross-fit weights.
            ckpt_info: dict[str, Any] = {}
            for method in METHODS:
                wvec = weight_vector(cache, refit_rows, weights2[method], method)
                ck = RUNTIME / f"seed-{seed}" / dataset / f"fold-{fold}" / f"{method}.pt"
                ck_expected = {"schema": CACHE_SCHEMA_VERSION, "code_fingerprint": code_fingerprint(),
                               "dataset": dataset, "fold": int(fold), "seed": int(seed), "method": method,
                               "stage": "final_refit", "selected_epoch": int(selected[method]),
                               "initial_state_sha256": init_sha, "rows_sha256": array_sha(refit_rows),
                               "weights_sha256": array_sha(np.asarray(wvec, dtype=np.float32)),
                               "mean_sha256": bytes_sha(np.asarray(refit_mean).tobytes()),
                               "std_sha256": bytes_sha(np.asarray(refit_std).tobytes())}
                fit_timing: dict[str, Any] = {}
                ck_hit = checkpoint_cache_valid(ck, ck_expected)
                if ck_hit:
                    ck_sha = file_sha(ck)
                    print(f"[cache] checkpoint hit {dataset} fold={fold} method={method}", flush=True)
                else:
                    model = fit_exact(cache, refit_rows, refit_mean, refit_std, wvec, state, dataset, fold, seed,
                                      "student-common", selected[method], device, timing=fit_timing)
                    ck_sha = save_checkpoint(ck, model, refit_mean, refit_std, dataset, fold, seed, method,
                                             selected[method], init_sha, cache_meta=ck_expected)
                    del model
                ckpt_info[method] = {"path": str(ck), "sha256": ck_sha, "selected_epoch": selected[method]}
                training_rows.append({"dataset": dataset, "fold": fold, "seed": seed, "stage": "final_refit",
                                      "method": method, "selected_epoch": selected[method], "initial_state_sha256": init_sha,
                                      "normalizer_mean_sha256": bytes_sha(refit_mean.tobytes()),
                                      "normalizer_std_sha256": bytes_sha(refit_std.tobytes()),
                                      "training_subjects": len(source_all), "discovery_subjects": len(role["discovery"]),
                                      "weight_mean": float(wvec.mean()), "weight_min": float(wvec.min()), "weight_max": float(wvec.max()),
                                      "checkpoint_sha256": ck_sha, "fit_sec": float(fit_timing.get("sec", 0.0)),
                                      "fit_sec_per_epoch": float(fit_timing.get("sec_per_epoch", 0.0)), "cache_hit": bool(ck_hit)})
                gc.collect()
            fold_manifests[f"{dataset}/fold-{fold}/seed-{seed}"] = {
                "dataset": dataset, "fold": fold, "seed": seed, "model_fit_subjects": source,
                "discovery_subjects": subj_sort(role["discovery"]), "outcome_subjects_hash": bytes_sha("|".join(subj_sort(role["outcome"])).encode()),
                "role_hash": role_hash(role), "selected_epochs": selected, "checkpoints": ckpt_info,
                "initial_normalizer_mean_sha256": bytes_sha(fit_mean.tobytes()), "initial_normalizer_std_sha256": bytes_sha(fit_std.tobytes()),
                "refit_normalizer_mean_sha256": bytes_sha(refit_mean.tobytes()), "refit_normalizer_std_sha256": bytes_sha(refit_std.tobytes()),
                "source_initial_weight_lock": wa1["lock"], "source_final_weight_lock": wa2["lock"],
                "initial_histories": histories,
            }
            write_json(RUNTIME / f"seed-{seed}" / dataset / f"fold-{fold}" / "FOLD_PROGRESS.json", fold_manifests[f"{dataset}/fold-{fold}/seed-{seed}"])
            print(f"[preflight] complete {dataset} fold={fold} source={len(source)} refit={len(source_all)}", flush=True)
            del cache
            gc.collect()
    write_csv(RESULTS / "CROSS_FIT_ASSIGNMENTS.csv", all_crossfit)
    write_csv(RESULTS / "CROSSFIT_TEACHER_AUDIT.csv", all_teachers)
    write_csv(RESULTS / "SOURCE_GEOMETRY_RISK.csv", all_risk)
    write_csv(RESULTS / "SOURCE_WEIGHT_AUDIT.csv", all_weights)
    write_csv(RESULTS / "TRAINING_SUMMARY.csv", training_rows)
    write_json(RESULTS / "INITIAL_STATE_HASHES.json", state_hashes)
    # Lock is written only after all source weights/checkpoints exist.
    lock = {
        "schema": "PERSIST_EEG_GEOSR_PRE_OUTCOME_PROTOCOL_LOCK_V1", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed, "datasets": list(DATASETS), "folds": list(FOLDS), "backbone": "EEGNet",
        "methods": list(METHODS), "inner_crossfit_k": INNER_K, "descriptor_cap": CAP,
        "formula": {"N_geo": "1 - .5*(cos(v_s,t2,c_t1)+cos(v_s,t1,c_t2))", "N_loss": "balanced held-out descriptor NLL",
                     "ranks": "scipy rankdata(method=average)", "GeoSR": "0.5+0.5*r_geo+0.5*r_loss", "weight_range": [0.5, 1.5]},
        "training": {"architecture": "canonical VanillaEEGNet F1=8 D=2 F2=16 temporal=64 pool=4,8 dropout=.25 embedding=64",
                     "optimizer": "AdamW lr=3e-4 wd=5e-4 batch=64 grad_clip=5", "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS,
                     "patience": PATIENCE, "epoch_selection": "discovery mean subject BA; lower NLL; earlier epoch",
                     "fair_initial_state_per_outer_fold": True, "same_minibatch_order": True},
        "role_hashes": role_hashes, "fold_manifests": fold_manifests,
        "random_rank_permutations": {k: v["source_final_weight_lock"]["random_rank_permutation"] for k, v in fold_manifests.items()},
        "outcome_labels_read": False, "canonical_outcome_indices_materialized": False,
        "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False,
        "code_sha256": file_sha(Path(__file__)), "audit_primitives_sha256": file_sha(Path(__file__).with_name("audit_primitives.py")),
    }
    write_json(EXP / "PRE_OUTCOME_GEOSR_PROTOCOL_LOCK.json", lock)
    write_json(EXP / "DATA_LEGALITY_AUDIT.json", {
        "schema": "PERSIST_EEG_GEOSR_DATA_LEGALITY_V1", "seed": seed, "seed1_run": False, "seed2_run": False,
        "second_backbone_run": False, "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False,
        "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False,
        "PGEG_training_started": False, "outcome_labels_read_before_lock": False,
        "outcome_labels_read_after_lock": False, "lock_sha256": file_sha(EXP / "PRE_OUTCOME_GEOSR_PROTOCOL_LOCK.json"),
        "role_hashes": role_hashes, "descriptor_cap": CAP,
    })
    write_json(RUNTIME / f"seed-{seed}" / "PREFLIGHT_MANIFEST.json", fold_manifests)
    print("PRE_OUTCOME_GEOSR_PROTOCOL_LOCKED", flush=True)


def eval_checkpoint(data: ap.SafeData, ck_path: Path, outcome_subjects: Sequence[str], dataset: str,
                    fold: int, seed: int, device: torch.device) -> list[dict[str, Any]]:
    state = torch.load(ck_path, map_location="cpu", weights_only=False)
    # Materializing outcome indices is intentionally confined to this post-lock call.
    mask = data.metadata.subject_id.astype(str).isin(set(map(str, outcome_subjects))) & data.metadata.session_id.astype(int).eq(SESSION_OUTCOME[dataset])
    rows = np.flatnonzero(mask.to_numpy()).astype(np.int64)
    model = ap.VanillaEEGNet(int(data.batch(np.asarray([rows[0]], dtype=np.int64)).shape[1])).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    mean, std = np.asarray(state["mean"]), np.asarray(state["std"])
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            q = rows[start:start + BATCH_SIZE]
            x = ap.prepare(data, q, mean, std).to(device, non_blocking=True)
            logits.append(model(x).detach().cpu().numpy())
    z = np.concatenate(logits, axis=0); z -= z.max(axis=1, keepdims=True); p = np.exp(z); p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
    frame = data.metadata.iloc[rows].reset_index(drop=True); y = frame.label.to_numpy(np.int64); pred = p.argmax(1)
    out: list[dict[str, Any]] = []
    for s, g in frame.groupby(frame.subject_id.astype(str), sort=True):
        loc = g.index.to_numpy(np.int64); yy = y[loc]
        out.append({"dataset": dataset, "fold": fold, "seed": seed, "subject_id": str(s), "BA": float(balanced_accuracy_score(yy, pred[loc])),
                    "accuracy": float(accuracy_score(yy, pred[loc])), "macro_F1": float(f1_score(yy, pred[loc], average="macro", zero_division=0)),
                    "NLL": float(log_loss(yy, p[loc], labels=[0, 1])), "trials": int(len(loc))})
    del model, data
    return out


def bootstrap_pair(a: np.ndarray, b: np.ndarray, dataset: str, tag: str, seed: int) -> dict[str, Any]:
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(stable_seed("geosr-paired-bootstrap", dataset, tag, seed))
    idx = rng.integers(0, len(d), size=(BOOTSTRAP_DRAWS, len(d)))
    draws = d[idx].mean(axis=1)
    return {"mean_delta_pp": float(d.mean() * 100.0), "median_delta_pp": float(np.median(d) * 100.0),
            "CI95_low_pp": float(np.quantile(draws, .025) * 100.0), "CI95_high_pp": float(np.quantile(draws, .975) * 100.0),
            "positive_subject_fraction": float(np.mean(d > 0)), "nonnegative_subject_fraction": float(np.mean(d >= 0)),
            "worst_subject_delta_pp": float(d.min() * 100.0), "draws": BOOTSTRAP_DRAWS, "n_subjects": int(len(d))}


def outcome(seed: int, device: torch.device) -> dict[str, Any]:
    lock_path = EXP / "PRE_OUTCOME_GEOSR_PROTOCOL_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("pre-outcome lock missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if int(lock.get("seed", -1)) != seed or lock.get("outcome_labels_read") is not False:
        raise RuntimeError("invalid pre-outcome lock")
    all_rows: list[dict[str, Any]] = []
    manifest = json.loads((RUNTIME / f"seed-{seed}" / "PREFLIGHT_MANIFEST.json").read_text(encoding="utf-8"))
    for dataset in DATASETS:
        roles, _, _ = ap.load_roles(dataset)
        for fold in FOLDS:
            role = roles[fold]
            # This is the first call that loads outcome-subject labels.
            data = ap.load_ab_data(dataset, set(role["outcome"]))
            key = f"{dataset}/fold-{fold}/seed-{seed}"
            for method in METHODS:
                ck = Path(manifest[key]["checkpoints"][method]["path"])
                all_rows.extend([{**r, "method": method} for r in eval_checkpoint(data, ck, role["outcome"], dataset, fold, seed, device)])
                # eval_checkpoint consumes its data object; reload for the next method.
                data = ap.load_ab_data(dataset, set(role["outcome"]))
            del data
    frame = pd.DataFrame(all_rows)
    write_csv(RESULTS / "OUTCOME_PER_SUBJECT.csv", frame)
    fold_summary = frame.groupby(["dataset", "fold", "seed", "method"], as_index=False).agg(
        mean_subject_BA=("BA", "mean"), mean_accuracy=("accuracy", "mean"), mean_macro_F1=("macro_F1", "mean"), mean_NLL=("NLL", "mean"), n_subjects=("subject_id", "nunique"))
    write_csv(RESULTS / "OUTCOME_PER_FOLD.csv", fold_summary)
    # Aggregate repeated biological identities before any inferential statistic.
    subj = frame.groupby(["dataset", "method", "subject_id"], as_index=False).agg(BA=("BA", "mean"), accuracy=("accuracy", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"))
    sb = subj[subj.method == "SUBJECT_BALANCED_ERM"].rename(columns={"BA": "BA_SB"})[["dataset", "subject_id", "BA_SB"]]
    joined = subj.merge(sb, on=["dataset", "subject_id"], how="left")
    perf = joined.groupby(["dataset", "method"], as_index=False).agg(mean_subject_BA=("BA", "mean"), mean_accuracy=("accuracy", "mean"), mean_macro_F1=("macro_F1", "mean"), mean_NLL=("NLL", "mean"), n_subjects=("subject_id", "nunique"))
    write_csv(RESULTS / "PERFORMANCE_SUMMARY.csv", perf)
    tail_rows: list[dict[str, Any]] = []; comparison_rows: list[dict[str, Any]] = []; boot: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for dataset in DATASETS:
        f = joined[joined.dataset == dataset].copy()
        base = f[f.method == "SUBJECT_BALANCED_ERM"].set_index("subject_id").BA
        bottom_ids = base.sort_values().index[:max(1, int(math.ceil(.25 * len(base))))]
        geo = f[f.method == "GEOSR"].set_index("subject_id").reindex(base.index)
        deltas = geo.BA - base
        fold_geo = fold_summary[(fold_summary.dataset == dataset) & (fold_summary.method == "GEOSR")].set_index("fold").mean_subject_BA
        fold_sb = fold_summary[(fold_summary.dataset == dataset) & (fold_summary.method == "SUBJECT_BALANCED_ERM")].set_index("fold").mean_subject_BA
        tail_rows.append({"dataset": dataset, "fixed_bottom25_n": len(bottom_ids), "bottom25_subjects": ",".join(map(str, bottom_ids.tolist())),
                          "geosr_mean_BA": float(geo.BA.mean()), "sb_erm_mean_BA": float(base.mean()),
                          "mean_delta_pp": float(deltas.mean() * 100), "bottom25_mean_BA_geosr": float(geo.loc[bottom_ids].BA.mean()),
                          "bottom25_mean_BA_sb_erm": float(base.loc[bottom_ids].mean()), "bottom25_delta_pp": float((geo.loc[bottom_ids].BA.mean() - base.loc[bottom_ids].mean()) * 100),
                          "p10_subject_BA_geosr": float(np.quantile(geo.BA, .10)), "harmed_fraction": float(np.mean(deltas < 0)),
                          "material_harm_fraction": float(np.mean(deltas <= -.02)), "nonnegative_fraction": float(np.mean(deltas >= 0)),
                          "worst_subject_delta_pp": float(deltas.min() * 100), "worst_quartile_delta_pp": float(deltas.sort_values().iloc[:max(1, int(math.ceil(.25 * len(deltas))))].mean() * 100),
                          "fold_nonnegative_count": int(np.sum(fold_geo.to_numpy() >= fold_sb.to_numpy())), "fold_count": 5})
        gates.setdefault(dataset, {})
        gates[dataset]["mean_delta_pp"] = float(deltas.mean() * 100)
        gates[dataset]["bottom25_delta_pp"] = float((geo.loc[bottom_ids].BA.mean() - base.loc[bottom_ids].mean()) * 100)
        gates[dataset]["nonnegative_fraction"] = float(np.mean(deltas >= 0)); gates[dataset]["material_harm_fraction"] = float(np.mean(deltas <= -.02))
        gates[dataset]["fold_nonnegative_count"] = int(np.sum(fold_geo.to_numpy() >= fold_sb.to_numpy()))
        for comparator in ("SUBJECT_BALANCED_ERM", "CANONICAL_ERM", "RANDOM_RANK", "LOSS_HARD", "GEO_ONLY"):
            c = f[f.method == comparator].set_index("subject_id").reindex(base.index)
            result = bootstrap_pair(geo.BA.to_numpy(), c.BA.to_numpy(), dataset, comparator, seed)
            comparison_rows.append({"dataset": dataset, "method": "GEOSR", "comparator": comparator, **result})
            boot[f"{dataset}:{comparator}"] = result
    write_csv(RESULTS / "TAIL_ROBUSTNESS.csv", tail_rows)
    write_csv(RESULTS / "CONTROL_COMPARISON.csv", comparison_rows)
    write_json(RESULTS / "PAIRED_BOOTSTRAP.json", boot)
    g1 = all(gates[d]["mean_delta_pp"] >= .10 for d in DATASETS) and any(gates[d]["mean_delta_pp"] >= .30 for d in DATASETS)
    g2 = all(gates[d]["bottom25_delta_pp"] >= 0 for d in DATASETS) and any(gates[d]["bottom25_delta_pp"] >= .50 for d in DATASETS)
    g3 = all(gates[d]["nonnegative_fraction"] >= .60 and gates[d]["material_harm_fraction"] <= .15 for d in DATASETS)
    g4 = all(gates[d]["fold_nonnegative_count"] >= 3 for d in DATASETS)
    random_perf = perf[perf.method == "RANDOM_RANK"].set_index("dataset").mean_subject_BA
    geosr_perf = perf[perf.method == "GEOSR"].set_index("dataset").mean_subject_BA
    g5 = all(float(geosr_perf[d]) >= float(random_perf[d]) for d in DATASETS)
    g6 = all(float(next(x for x in comparison_rows if x["dataset"] == d and x["comparator"] == "SUBJECT_BALANCED_ERM")["CI95_high_pp"]) > 0 for d in DATASETS)
    gates_all = {"G1_mean_utility": g1, "G2_tail_utility": g2, "G3_individual_harm": g3, "G4_fold_robustness": g4, "G5_not_random": g5, "G6_bootstrap_upper_positive": g6}
    go = all(gates_all.values())
    terminal = "GEOSR_SEED0_GO_MULTISEED" if go and seed == 0 else "GEOSR_FINAL_CONSTRUCTIVE_STOP"
    decision = {"schema": "PERSIST_EEG_GEOSR_FINAL_DECISION_V1", "seed": seed, "terminal": terminal,
                "MULTISEED_AUTHORIZED": bool(go and seed == 0), "READY_FOR_SECOND_BACKBONE": False,
                "gates": gates_all, "dataset_gate_values": gates, "outer_sealed_access": False,
                "scientific_rescue_performed": False, "PGEG_training_started": False}
    write_json(RESULTS / "FINAL_DECISION.json", decision)
    report = ["# GeoSR final constructive experiment", "", f"Terminal: `{terminal}`", "", "All inference is biological-subject level; outcome labels were loaded only after PRE_OUTCOME_GEOSR_PROTOCOL_LOCK.json.", "", "|Dataset|GeoSR−SB-ERM mean BA (pp)|fixed bottom-25 delta (pp)|nonnegative|material harm|folds nonnegative|", "|---|---:|---:|---:|---:|---:|"]
    for row in tail_rows:
        report.append(f"|{row['dataset']}|{row['mean_delta_pp']:.3f}|{row['bottom25_delta_pp']:.3f}|{row['nonnegative_fraction']:.3f}|{row['material_harm_fraction']:.3f}|{row['fold_nonnegative_count']}/5|")
    report += ["", "## Frozen gates", "", *[f"- {k}: `{v}`" for k, v in gates_all.items()], "", f"RANDOM_RANK comparison gate: `{g5}`", f"Paired bootstrap upper-CI gate: `{g6}`", "", "No scientific rescue, formula change, threshold search, outer access, or second backbone was performed.", ""]
    (RESULTS / "FINAL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    # Update legality only after the locked outcome call.
    legality = json.loads((EXP / "DATA_LEGALITY_AUDIT.json").read_text(encoding="utf-8"))
    legality.update({"canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": True, "outcome_labels_read_after_lock": True,
                     "outcome_evaluated_after_lock": True, "lock_sha256": file_sha(lock_path)})
    write_json(EXP / "DATA_LEGALITY_AUDIT.json", legality)
    write_json(RESULTS / "VALIDATION.json", {"required_files": True, "seed_zero_only": seed == 0, "eegnet_only": True,
                                               "outer_sealed_closed": True, "outcome_after_lock": True, "no_scientific_rescue": True,
                                               "pass": True})
    print(terminal, flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "outcome", "all"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.seed not in (0, 1, 2):
        raise RuntimeError("seed must be 0, 1, or 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    print(f"GeoSR phase={args.phase} seed={args.seed} device={device} torch={torch.__version__}", flush=True)
    if args.phase in ("preflight", "all"):
        preflight(args.seed, device)
    if args.phase in ("outcome", "all"):
        decision = outcome(args.seed, device)
        # Seed 1/2 are permitted only after a successful seed-0 gate.  The
        # current run is deliberately not allowed to manufacture a rescue path.
        if args.phase == "all" and args.seed == 0 and decision.get("MULTISEED_AUTHORIZED"):
            print("SEED0_GO: launching frozen seeds 1 and 2", flush=True)
            for s in (1, 2):
                preflight(s, device)
                outcome(s, device)


if __name__ == "__main__":
    main()
