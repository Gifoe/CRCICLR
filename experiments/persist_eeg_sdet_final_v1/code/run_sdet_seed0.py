"""PERSIST-EEG Population-Consensus SDET seed-0 viability experiment.

The script is deliberately one-shot and outcome-gated.  It trains an exact
canonical EEGNet on the V8 SEARCH population, learns one low-rank population
intervention from subject-disjoint SEARCH episodes, freezes every trainable
object, and only then opens internal-holdout labels for paired scoring.
Runtime/checkpoints are kept outside the Git worktree; this file and compact
protocol/results are the only repository artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


REPO = Path(os.environ.get("SDET_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_SDET_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_sdet_final_v1"
CODE = EXP / "code"
OUT = EXP / "outputs"
PROTOCOL = OUT / "protocol"
RUNTIME = Path(os.environ.get("SDET_RUNTIME", r"D:\nips-temp\TotalP\P2\sdet_runtime_seed0")).resolve()
STAGE0_ROOT = Path(os.environ.get("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full")).resolve()
WBCIC_DEV_ROOT = Path(os.environ.get(
    "PERSIST_WBCIC_REPO", str(REPO)
)).resolve()
WBCIC_CACHE = Path(os.environ.get(
    "PERSIST_WBCIC_CACHE",
    r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache",
)).resolve()
WBCIC_OUTER_CACHE = Path(os.environ.get(
    "PERSIST_WBCIC_OUTER_CACHE",
    r"D:\nips-temp\TotalP\P2\wbcic_outer_cache\wbcic_epochs",
)).resolve()
SPLIT_PATH = REPO / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"
OPENBMI_MANIFEST = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
WBCIC_META = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
WBCIC_RAW = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_RAW.npy"

SEED = 0
RANK = 4
RHO = 0.05
LAMBDA_DOWNSIDE = 1.0
BETA = 1e-4
EPS = 1e-8
TIE_TOL = 1e-8
BASELINE_MAX_EPOCHS = 60
BASELINE_MIN_EPOCHS = 10
BASELINE_PATIENCE = 8
BASELINE_BATCH_SIZE = 64
BASELINE_LR = 3e-4
BASELINE_WEIGHT_DECAY = 5e-4
SDET_EPOCHS = 30
SDET_LR = 1e-2
SDET_WEIGHT_DECAY = 0.0
SDET_EPISODES = 5


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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subject_sort(values: Iterable[object], dataset: str) -> list[str]:
    def key(value: str):
        text = str(value)
        if dataset == "WBCIC":
            text = text.replace("sub-", "")
        return (int(text) if text.isdigit() else 10**9, text)
    return sorted([str(v) for v in values], key=key)


def load_split() -> tuple[dict[str, Any], str]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(SPLIT_PATH)
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8-sig"))
    digest = sha256_file(SPLIT_PATH)
    for dataset in ("openbmi", "wbcic"):
        section = payload[dataset]
        for key in ("V8_SEARCH", "V8_INTERNAL_HOLDOUT"):
            if not section.get(key):
                raise RuntimeError(f"empty frozen split: {dataset} {key}")
    return payload, digest


@dataclass
class Row:
    subject: str
    session: int
    signal_path: str
    cache_index: int
    label: int | None = None
    label_path: str | None = None


class SignalAccessor:
    def __init__(self, rows: list[Row], root: Path, channels: int):
        self.rows = rows
        self.root = root
        self.channels = channels
        self._arrays: dict[str, np.ndarray] = {}

    def _array(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            path = Path(key)
            if not path.is_absolute():
                path = self.root / path
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            self._arrays[key] = arr
        return self._arrays[key]

    def batch(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        values: list[np.ndarray] = []
        for index in indices:
            row = self.rows[int(index)]
            values.append(np.asarray(self._array(row.signal_path)[row.cache_index], dtype=np.float32))
        if not values:
            return np.empty((0, self.channels, 1000), dtype=np.float32)
        return np.stack(values, axis=0)


@dataclass
class DatasetBundle:
    name: str
    search_subjects: list[str]
    test_subjects: list[str]
    search_rows: list[Row]
    test_rows: list[Row]
    search_accessor: SignalAccessor
    test_accessor: SignalAccessor
    channels: int

    def search_indices(self, subjects: Iterable[str], sessions: Iterable[int] | None = None) -> np.ndarray:
        wanted = set(map(str, subjects))
        sess = None if sessions is None else set(map(int, sessions))
        return np.asarray([
            i for i, row in enumerate(self.search_rows)
            if row.subject in wanted and (sess is None or row.session in sess)
        ], dtype=np.int64)

    def test_indices(self, subjects: Iterable[str], sessions: Iterable[int] | None = None) -> np.ndarray:
        wanted = set(map(str, subjects))
        sess = None if sessions is None else set(map(int, sessions))
        return np.asarray([
            i for i, row in enumerate(self.test_rows)
            if row.subject in wanted and (sess is None or row.session in sess)
        ], dtype=np.int64)

    def load_search_labels(self, indices: np.ndarray) -> np.ndarray:
        values = [self.search_rows[int(i)].label for i in indices]
        if any(v is None for v in values):
            raise RuntimeError(f"missing SEARCH labels for {self.name}")
        return np.asarray(values, dtype=np.int64)

    def load_test_labels(self, indices: np.ndarray) -> np.ndarray:
        """Open TEST labels only from the final scoring call."""
        values: list[int] = []
        grouped: dict[tuple[str, int, str], list[tuple[int, int]]] = {}
        for pos, index in enumerate(indices):
            row = self.test_rows[int(index)]
            if row.label_path is None:
                raise RuntimeError(f"missing TEST label path: {self.name}")
            grouped.setdefault((row.subject, row.session, row.label_path), []).append((pos, row.cache_index))
        for key, positions in grouped.items():
            path = Path(key[2])
            if not path.is_absolute():
                path = self.test_accessor.root / path
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            for pos, offset in positions:
                code = int(arr[offset])
                # OpenBMI stores MI labels with the raw event codes {1, 2},
                # while the EEGNet head and WBCIC cache use the canonical
                # zero-based class IDs {0, 1}.  Keep the sealed TEST file
                # closed until this final scoring call, then apply the same
                # mapping used for SEARCH rows before metrics are computed.
                if self.name == "OpenBMI":
                    if code not in (1, 2):
                        raise RuntimeError(f"unexpected OpenBMI TEST label code {code}")
                    code -= 1
                elif code not in (0, 1):
                    raise RuntimeError(f"unexpected WBCIC TEST label code {code}")
                values.append((pos, code))
        ordered = [code for _, code in sorted(values)]
        return np.asarray(ordered, dtype=np.int64)


def _openbmi_rows(root: Path, subjects: list[str], test_subjects: list[str]) -> tuple[list[Row], list[Row]]:
    if not OPENBMI_MANIFEST.is_file():
        raise FileNotFoundError(OPENBMI_MANIFEST)
    columns = ["subject_id", "session_id", "paradigm", "run_phase", "trial_id", "signal_cache_path", "label_cache_path", "cache_index"]
    frame = pd.read_parquet(OPENBMI_MANIFEST, columns=columns)
    frame["subject_id"] = frame["subject_id"].astype(str).str.replace("sub-", "", regex=False)
    frame["session_id"] = frame["session_id"].astype(int)
    frame = frame[(frame["paradigm"] == "mi") & (frame["run_phase"] == "train")].copy()
    frame = frame[frame["subject_id"].isin(set(subjects) | set(test_subjects))].reset_index(drop=True)
    if len(frame) != 54 * 2 * 100:
        raise RuntimeError(f"OpenBMI MI manifest rows mismatch: {len(frame)}")
    # Label files are opened only for SEARCH subjects at this stage.
    labels_by_path: dict[str, np.ndarray] = {}
    search_rows: list[Row] = []
    test_rows: list[Row] = []
    search_set = set(subjects)
    test_set = set(test_subjects)
    for row in frame.itertuples(index=False):
        subject = str(row.subject_id)
        label_path = str(root / str(row.label_cache_path))
        if subject in search_set:
            if label_path not in labels_by_path:
                labels_by_path[label_path] = np.load(label_path, mmap_mode="r", allow_pickle=False)
            code = int(labels_by_path[label_path][int(row.cache_index)])
            if code not in (1, 2):
                raise RuntimeError(f"unexpected OpenBMI MI label code {code}")
            search_rows.append(Row(subject, int(row.session_id), str(row.signal_cache_path), int(row.cache_index), code - 1, label_path))
        elif subject in test_set:
            # Do not open test label files until the final scoring phase.
            test_rows.append(Row(subject, int(row.session_id), str(row.signal_cache_path), int(row.cache_index), None, label_path))
    return search_rows, test_rows


def _wbcic_rows(subjects: list[str], test_subjects: list[str]) -> tuple[list[Row], list[Row]]:
    # The development cache contains one independent epochs/labels pair per
    # subject/session.  This is the V8 SEARCH + internal-holdout resource; the
    # separate P2 outer cache contains only the true outer cohort and must not
    # be used here.  Reading a per-subject labels file for SEARCH is safe, while
    # TEST label files are recorded as paths and opened only by load_test_labels
    # after both dataset interventions have been frozen.
    epochs_root = WBCIC_CACHE / "wbcic_epochs"
    if not epochs_root.is_dir():
        raise FileNotFoundError(f"WBCIC per-subject development cache missing: {epochs_root}")
    search_set, test_set = set(subjects), set(test_subjects)
    search_rows: list[Row] = []
    test_rows: list[Row] = []
    for subject in sorted(search_set | test_set, key=lambda x: int(x.replace("sub-", ""))):
        for session in (0, 1, 2):
            epochs_path = epochs_root / subject / f"ses-{session}_epochs.npy"
            labels_path = epochs_root / subject / f"ses-{session}_labels.npy"
            if not epochs_path.is_file() or not labels_path.is_file():
                raise FileNotFoundError(f"WBCIC development cache missing: {subject} ses-{session}")
            epochs = np.load(epochs_path, mmap_mode="r", allow_pickle=False)
            if epochs.ndim != 3 or tuple(epochs.shape[1:]) != (58, 1000) or epochs.dtype != np.float16:
                raise RuntimeError(f"unexpected WBCIC epochs schema: {epochs_path} {epochs.shape} {epochs.dtype}")
            if subject in search_set:
                labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
                if labels.ndim != 1 or labels.shape[0] != epochs.shape[0] or labels.dtype.kind not in "iu":
                    raise RuntimeError(f"unexpected WBCIC labels schema: {labels_path} {labels.shape} {labels.dtype}")
                for index, code in enumerate(labels):
                    value = int(code)
                    if value not in (0, 1):
                        raise RuntimeError(f"unexpected WBCIC SEARCH label code {value}")
                    search_rows.append(Row(subject, session, str(epochs_path), index, value, None))
            if subject in test_set:
                # Do not load the TEST labels here; only retain the path.
                for index in range(int(epochs.shape[0])):
                    test_rows.append(Row(subject, session, str(epochs_path), index, None, str(labels_path)))
    return search_rows, test_rows


def load_bundle(name: str, split: dict[str, Any]) -> DatasetBundle:
    section = split["openbmi" if name == "OpenBMI" else "wbcic"]
    if name == "OpenBMI":
        search = subject_sort(section["V8_SEARCH"], name)
        test = subject_sort(section["V8_INTERNAL_HOLDOUT"], name)
        search_rows, test_rows = _openbmi_rows(STAGE0_ROOT, search, test)
        channels = 62
        root = STAGE0_ROOT
    else:
        search = subject_sort(section["V8_SEARCH"], name)
        test = subject_sort(section["V8_INTERNAL_HOLDOUT"], name)
        search_rows, test_rows = _wbcic_rows(search, test)
        channels = 58
        root = WBCIC_CACHE
    if not search_rows or not test_rows:
        raise RuntimeError(f"empty {name} rows")
    return DatasetBundle(
        name=name,
        search_subjects=search,
        test_subjects=test,
        search_rows=search_rows,
        test_rows=test_rows,
        search_accessor=SignalAccessor(search_rows, root, channels),
        test_accessor=SignalAccessor(test_rows, root if name == "OpenBMI" else WBCIC_OUTER_CACHE, channels),
        channels=channels,
    )


class VanillaEEGNet(nn.Module):
    """Exact canonical EEGNet backbone used by the committed baseline."""

    def __init__(self, channels: int, samples: int = 1000, dropout: float = 0.25):
        super().__init__()
        if samples != 1000:
            raise ValueError("canonical EEGNet expects 1000 samples")
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
        self.embedding = nn.Sequential(nn.Linear(f2 * (samples // 4 // 8), 64), nn.ELU(), nn.LayerNorm(64))
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


def compute_normalizer(accessor: SignalAccessor, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(accessor.channels, dtype=np.float64)
    square = np.zeros(accessor.channels, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 128):
        batch = accessor.batch(indices[start : start + 128]).astype(np.float64)
        total += batch.sum(axis=(0, 2))
        square += np.square(batch).sum(axis=(0, 2))
        count += batch.shape[0] * batch.shape[2]
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def prepare(accessor: SignalAccessor, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    value = accessor.batch(indices)
    value = (value - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device, non_blocking=True)


def subject_ba(labels: np.ndarray, preds: np.ndarray, subjects: np.ndarray) -> float:
    values = []
    for subject in sorted(set(subjects), key=str):
        mask = subjects == subject
        values.append(float(balanced_accuracy_score(labels[mask], preds[mask])))
    return float(np.mean(values)) if values else float("nan")


def evaluate_model(model: VanillaEEGNet, accessor: SignalAccessor, indices: np.ndarray, labels: np.ndarray, subjects: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), BASELINE_BATCH_SIZE):
            x = prepare(accessor, indices[start : start + BASELINE_BATCH_SIZE], mean, std, device)
            probabilities.append(torch.softmax(model(x), dim=1).cpu().numpy())
    probability = np.concatenate(probabilities, axis=0)
    return subject_ba(labels, np.argmax(probability, axis=1), subjects), probability


def train_one_epoch(model: VanillaEEGNet, accessor: SignalAccessor, indices: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray, optimizer: torch.optim.Optimizer, device: torch.device, rng: np.random.Generator) -> float:
    model.train()
    order = rng.permutation(indices)
    # ``indices`` are global row IDs, while ``labels`` is aligned to the
    # supplied index array (not to the global row-ID space).  Build the
    # explicit lookup once per epoch so shuffling sparse/global IDs cannot
    # index the compact label vector out of bounds.
    label_lookup = np.full(int(np.max(indices)) + 1, -1, dtype=np.int64)
    label_lookup[np.asarray(indices, dtype=np.int64)] = np.asarray(labels, dtype=np.int64)
    losses = []
    for start in range(0, len(order), BASELINE_BATCH_SIZE):
        batch_indices = order[start : start + BASELINE_BATCH_SIZE]
        x = prepare(accessor, batch_indices, mean, std, device)
        y_np = label_lookup[np.asarray(batch_indices, dtype=np.int64)]
        if np.any(y_np < 0):
            raise RuntimeError("baseline label lookup missing a training row")
        y = torch.from_numpy(y_np).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite baseline loss")
        loss.backward()
        # Match the canonical EEGNet trainer's fixed global gradient clip.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def fit_baseline(bundle: DatasetBundle, mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[VanillaEEGNet, dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(np.asarray(bundle.search_subjects, dtype=object))
    n_disc = max(1, int(math.ceil(len(perm) * 0.2)))
    discovery_subjects = [str(x) for x in perm[:n_disc]]
    fit_subjects = [str(x) for x in perm[n_disc:]]
    fit_indices = bundle.search_indices(fit_subjects)
    disc_indices = bundle.search_indices(discovery_subjects)
    fit_labels = bundle.load_search_labels(fit_indices)
    disc_labels = bundle.load_search_labels(disc_indices)
    disc_subject_array = np.asarray([bundle.search_rows[int(i)].subject for i in disc_indices], dtype=object)
    set_seed(SEED)
    model = VanillaEEGNet(bundle.channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASELINE_LR, weight_decay=BASELINE_WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    best_epoch = BASELINE_MAX_EPOCHS
    best_score = -float("inf")
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    for epoch in range(1, BASELINE_MAX_EPOCHS + 1):
        loss = train_one_epoch(model, bundle.search_accessor, fit_indices, fit_labels, mean, std, optimizer, device, np.random.default_rng(SEED + epoch))
        val_ba, val_probability = evaluate_model(model, bundle.search_accessor, disc_indices, disc_labels, disc_subject_array, mean, std, device)
        val_nll = float(log_loss(disc_labels, val_probability, labels=[0, 1]))
        history.append({"epoch": epoch, "loss": loss, "discovery_subject_BA": val_ba, "discovery_subject_NLL": val_nll})
        improved = val_ba > best_score + 1e-12 or (abs(val_ba - best_score) <= 1e-12 and val_nll < best_nll - 1e-12)
        if epoch >= BASELINE_MIN_EPOCHS and improved:
            best_score = val_ba
            best_nll = val_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        elif epoch >= BASELINE_MIN_EPOCHS:
            bad_epochs += 1
            if bad_epochs >= BASELINE_PATIENCE:
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    # Deterministic final refit on the complete SEARCH population.
    set_seed(SEED)
    final_model = VanillaEEGNet(bundle.channels).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=BASELINE_LR, weight_decay=BASELINE_WEIGHT_DECAY)
    all_indices = bundle.search_indices(bundle.search_subjects)
    all_labels = bundle.load_search_labels(all_indices)
    for epoch in range(1, best_epoch + 1):
        train_one_epoch(final_model, bundle.search_accessor, all_indices, all_labels, mean, std, optimizer, device, np.random.default_rng(SEED + 10_000 + epoch))
    final_model.eval()
    state_path = RUNTIME / f"{bundle.name.lower()}_baseline_seed0.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), state_path)
    info = {
        "fit_subjects": fit_subjects,
        "discovery_subjects": discovery_subjects,
        "selected_epoch": best_epoch,
        "discovery_best_subject_BA": best_score,
        "discovery_best_subject_NLL": best_nll,
        "history": history,
        "final_refit_subjects": bundle.search_subjects,
        "final_refit_sessions": [1, 2] if bundle.name == "OpenBMI" else [0, 1, 2],
        "checkpoint_path": str(state_path),
    }
    return final_model, info, all_indices


def extract_latents(model: VanillaEEGNet, accessor: SignalAccessor, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), BASELINE_BATCH_SIZE):
            x = prepare(accessor, indices[start : start + BASELINE_BATCH_SIZE], mean, std, device)
            latents.append(model.forward_features(x).cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0), np.asarray(indices, dtype=np.int64)


def normalize_columns(value: torch.Tensor) -> None:
    with torch.no_grad():
        value.div_(value.norm(dim=0, keepdim=True).clamp_min(1e-12))


class SDETModule(nn.Module):
    def __init__(self, dim: int = 64, rank: int = RANK):
        super().__init__()
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(dim, rank) * 0.02)
        self.p = nn.Parameter(torch.zeros(rank))
        normalize_columns(self.U)
        normalize_columns(self.V)

    def transformed(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + (z @ self.V * a) @ self.U.t()

    def adapter(self, g: torch.Tensor) -> torch.Tensor:
        return -RHO * torch.tanh(self.p) * g


def class_balanced_risk(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    pieces = []
    for cls in (0, 1):
        mask = labels == cls
        if int(mask.sum()) == 0:
            raise RuntimeError("subject-level class balance missing class")
        pieces.append(F.cross_entropy(logits[mask], labels[mask]))
    return 0.5 * (pieces[0] + pieces[1])


def proposal_gradient(module: SDETModule, head: nn.Linear, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    a0 = torch.zeros(RANK, dtype=z.dtype, device=z.device, requires_grad=True)
    logits = head(module.transformed(z, a0))
    risk = class_balanced_risk(logits, labels)
    gradient = torch.autograd.grad(risk, a0, create_graph=False, retain_graph=False)[0]
    if not torch.isfinite(gradient).all():
        raise RuntimeError("non-finite proposal gradient")
    return gradient.detach()


def build_episodes(subjects: list[str], dataset: str) -> list[dict[str, list[str]]]:
    rng = np.random.default_rng(SEED)
    order = [str(x) for x in rng.permutation(np.asarray(subjects, dtype=object))]
    groups = [list(map(str, x)) for x in np.array_split(order, SDET_EPISODES)]
    episodes = []
    for query_group in groups:
        query = set(query_group)
        support = [s for s in order if s not in query]
        if set(support) & query:
            raise RuntimeError("episode support/query overlap")
        episodes.append({"support_subjects": support, "query_subjects": query_group})
    return episodes


def subject_tensors(bundle: DatasetBundle, latents: np.ndarray, latent_indices: np.ndarray, include_sessions: Iterable[int], subjects: Iterable[str], labels: np.ndarray) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pos = {int(index): i for i, index in enumerate(latent_indices.tolist())}
    sessions = set(map(int, include_sessions))
    output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for subject in subjects:
        indices = [i for i, row in enumerate(bundle.search_rows) if row.subject == subject and row.session in sessions]
        if not indices:
            raise RuntimeError(f"no latent rows for {bundle.name} {subject}")
        z = np.stack([latents[pos[int(i)]] for i in indices], axis=0)
        y = np.asarray([labels[pos[int(i)]] for i in indices], dtype=np.int64)
        output[subject] = (torch.from_numpy(z), torch.from_numpy(y))
    return output


def train_sdet(bundle: DatasetBundle, model: VanillaEEGNet, latents: np.ndarray, latent_indices: np.ndarray, search_labels: np.ndarray, device: torch.device, episodes: list[dict[str, list[str]]]) -> tuple[SDETModule, dict[str, Any], dict[str, Any]]:
    module = SDETModule().to(device)
    head = copy.deepcopy(model.head).to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    proposal_sessions = (1,) if bundle.name == "OpenBMI" else (0, 1)
    query_sessions = (2,) if bundle.name == "OpenBMI" else (2,)
    tensors = subject_tensors(bundle, latents, latent_indices, sorted(set(proposal_sessions) | set(query_sessions)), bundle.search_subjects, search_labels)
    optimizer = torch.optim.Adam(module.parameters(), lr=SDET_LR, weight_decay=SDET_WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, SDET_EPOCHS + 1):
        epoch_losses = []
        epoch_g = []
        epoch_a = []
        for episode_index, episode in enumerate(episodes):
            optimizer.zero_grad(set_to_none=True)
            gradients = []
            for subject in episode["support_subjects"]:
                z_all, y_all = tensors[subject]
                mask = torch.zeros(len(y_all), dtype=torch.bool)
                for row_index, row in enumerate(bundle.search_rows):
                    if row.subject == subject and row.session in proposal_sessions:
                        local = sum(1 for j in range(row_index + 1) if bundle.search_rows[j].subject == subject and bundle.search_rows[j].session in proposal_sessions) - 1
                        if local >= 0 and local < len(mask):
                            mask[local] = True
                # The tensor order is proposal sessions followed by query sessions;
                # reconstruct a direct mask by collecting the subject rows.
                proposal_rows = [i for i, row in enumerate(bundle.search_rows) if row.subject == subject and row.session in proposal_sessions]
                all_rows = [i for i, row in enumerate(bundle.search_rows) if row.subject == subject and row.session in sorted(set(proposal_sessions) | set(query_sessions))]
                position = {row_index: j for j, row_index in enumerate(all_rows)}
                local_positions = [position[i] for i in proposal_rows]
                gradients.append(proposal_gradient(module, head, z_all[local_positions].to(device), y_all[local_positions].to(device)))
            g_pop = torch.stack(gradients, dim=0).div(torch.stack(gradients, dim=0).norm(dim=1, keepdim=True).clamp_min(EPS)).mean(dim=0).detach()
            a = module.adapter(g_pop)
            risks_a = []
            risks_0 = []
            for subject in episode["query_subjects"]:
                z_all, y_all = tensors[subject]
                query_rows = [i for i, row in enumerate(bundle.search_rows) if row.subject == subject and row.session in query_sessions]
                all_rows = [i for i, row in enumerate(bundle.search_rows) if row.subject == subject and row.session in sorted(set(proposal_sessions) | set(query_sessions))]
                position = {row_index: j for j, row_index in enumerate(all_rows)}
                local_positions = [position[i] for i in query_rows]
                zq = z_all[local_positions].to(device)
                yq = y_all[local_positions].to(device)
                r0 = class_balanced_risk(head(zq), yq)
                ra = class_balanced_risk(head(module.transformed(zq, a)), yq)
                risks_0.append(r0.detach())
                risks_a.append(ra)
            ra_mean = torch.stack(risks_a).mean()
            downside = torch.stack([torch.relu(ra - r0) for ra, r0 in zip(risks_a, risks_0)]).mean()
            objective = ra_mean + LAMBDA_DOWNSIDE * downside + BETA * torch.sum(a * a)
            if not torch.isfinite(objective):
                raise RuntimeError("non-finite SDET objective")
            objective.backward()
            optimizer.step()
            normalize_columns(module.U)
            normalize_columns(module.V)
            epoch_losses.append(float(objective.detach().cpu()))
            epoch_g.append(float(g_pop.norm().detach().cpu()))
            epoch_a.append(float(a.norm().detach().cpu()))
        history.append({"epoch": epoch, "objective": float(np.mean(epoch_losses)), "g_norm": float(np.mean(epoch_g)), "a_norm": float(np.mean(epoch_a))})
        if epoch == 1 or epoch % 5 == 0 or epoch == SDET_EPOCHS:
            print(f"[{bundle.name}] SDET epoch {epoch}/{SDET_EPOCHS} objective={history[-1]['objective']:.6f} a_norm={history[-1]['a_norm']:.6f}", flush=True)
    elapsed = time.perf_counter() - started
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    # Final population proposal: all legal SEARCH subjects, no second normalize.
    final_grads = []
    for subject in bundle.search_subjects:
        z_all, y_all = tensors[subject]
        all_rows = [i for i, row in enumerate(bundle.search_rows) if row.subject == subject and row.session in sorted(set(proposal_sessions) | set(query_sessions))]
        position = {row_index: j for j, row_index in enumerate(all_rows)}
        proposal_rows = [i for i, row in enumerate(bundle.search_rows) if row.subject == subject and row.session in proposal_sessions]
        local_positions = [position[i] for i in proposal_rows]
        g = proposal_gradient(module, head, z_all[local_positions].to(device), y_all[local_positions].to(device))
        final_grads.append(g / g.norm().clamp_min(EPS))
    g_pop = torch.stack(final_grads).mean(dim=0).detach()
    a_pop = module.adapter(g_pop).detach()
    if not torch.isfinite(g_pop).all() or not torch.isfinite(a_pop).all():
        raise RuntimeError("non-finite final intervention")
    diagnostics = {
        "U": module.U.detach().cpu().numpy(),
        "V": module.V.detach().cpu().numpy(),
        "p": module.p.detach().cpu().numpy(),
        "g_pop": g_pop.cpu().numpy(),
        "a_pop": a_pop.cpu().numpy(),
        "g_pop_norm": float(g_pop.norm().cpu()),
        "a_pop_norm": float(a_pop.norm().cpu()),
        "U_column_norms": module.U.detach().norm(dim=0).cpu().numpy(),
        "V_column_norms": module.V.detach().norm(dim=0).cpu().numpy(),
        "runtime_seconds": elapsed,
    }
    return module, {"history": history, "runtime_seconds": elapsed, "episodes": episodes}, diagnostics


def score_test(bundle: DatasetBundle, model: VanillaEEGNet, module: SDETModule, diagnostics: dict[str, Any], mean: np.ndarray, std: np.ndarray, device: torch.device) -> list[dict[str, Any]]:
    # This function is the only place where TEST labels are opened.
    model.eval(); module.eval()
    rows: list[dict[str, Any]] = []
    for subject in bundle.test_subjects:
        # The preregistered unseen-subject endpoint is future-session only:
        # OpenBMI physical S2 and WBCIC physical S3 (both encoded as 2).
        indices = bundle.test_indices([subject], sessions=(2,))
        x = prepare(bundle.test_accessor, indices, mean, std, device)
        labels = bundle.load_test_labels(indices)
        with torch.no_grad():
            z = model.forward_features(x)
            baseline_probability = torch.softmax(model.head(z), dim=1).cpu().numpy()
            a = torch.from_numpy(np.asarray(diagnostics["a_pop"], dtype=np.float32)).to(device)
            U = module.U
            V = module.V
            z_sdet = z + (z @ V * a) @ U.t()
            sdet_probability = torch.softmax(model.head(z_sdet), dim=1).cpu().numpy()
        base_pred = np.argmax(baseline_probability, axis=1)
        sdet_pred = np.argmax(sdet_probability, axis=1)
        base_ba = float(balanced_accuracy_score(labels, base_pred))
        sdet_ba = float(balanced_accuracy_score(labels, sdet_pred))
        base_f1 = float(f1_score(labels, base_pred, average="macro"))
        sdet_f1 = float(f1_score(labels, sdet_pred, average="macro"))
        rows.append({"dataset": bundle.name, "subject_id": subject, "baseline_BA": base_ba, "sdet_BA": sdet_ba, "delta_BA": sdet_ba - base_ba, "baseline_F1": base_f1, "sdet_F1": sdet_f1, "delta_F1": sdet_f1 - base_f1, "trials": int(len(labels))})
    return rows


def aggregate(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    delta = frame["delta_BA"].to_numpy(float)
    improved = delta > TIE_TOL
    harmed = delta < -TIE_TOL
    tied = np.abs(delta) <= TIE_TOL
    return {
        "dataset": dataset,
        "n_subjects": int(len(frame)),
        "baseline_mean_subject_BA": float(frame["baseline_BA"].mean()),
        "sdet_mean_subject_BA": float(frame["sdet_BA"].mean()),
        "delta_BA_pp": float(frame["delta_BA"].mean() * 100.0),
        "baseline_macro_F1": float(frame["baseline_F1"].mean()),
        "sdet_macro_F1": float(frame["sdet_F1"].mean()),
        "delta_macro_F1": float(frame["delta_F1"].mean()),
        "median_subject_delta_BA_pp": float(np.median(delta) * 100.0),
        "mean_subject_delta_BA_pp": float(np.mean(delta) * 100.0),
        "improved_subjects": int(np.sum(improved)),
        "harmed_subjects": int(np.sum(harmed)),
        "tied_subjects": int(np.sum(tied)),
        "improved_subject_ratio": float(np.mean(improved)),
        "harmed_subject_ratio": float(np.mean(harmed)),
        "tied_subject_ratio": float(np.mean(tied)),
        "NTR0": float(np.mean(delta >= -TIE_TOL)),
        "NTR0_5": float(np.mean(delta >= 0.005 - TIE_TOL)),
        "worst_quartile_subject_delta_BA_pp": float(np.quantile(delta, 0.25) * 100.0),
        "best_quartile_subject_delta_BA_pp": float(np.quantile(delta, 0.75) * 100.0),
        "minimum_subject_delta_BA_pp": float(np.min(delta) * 100.0),
        "maximum_subject_delta_BA_pp": float(np.max(delta) * 100.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    set_seed(SEED)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    split, split_sha = load_split()
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CODE.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    test_payload = {
        "protocol": "V8_SEARCH_SPLIT.json",
        "split_manifest_path": str(SPLIT_PATH),
        "split_manifest_sha256": split_sha,
        "V8_INTERNAL_HOLDOUT_OPENED_FOR_SDET_SEED0": True,
        "OpenBMI": {"search_subjects": subject_sort(split["openbmi"]["V8_SEARCH"], "OpenBMI"), "test_subjects": subject_sort(split["openbmi"]["V8_INTERNAL_HOLDOUT"], "OpenBMI")},
        "WBCIC": {"search_subjects": subject_sort(split["wbcic"]["V8_SEARCH"], "WBCIC"), "test_subjects": subject_sort(split["wbcic"]["V8_INTERNAL_HOLDOUT"], "WBCIC")},
        "test_labels_read_before_freeze": False,
    }
    write_json(PROTOCOL / "TEST_SUBJECTS.json", test_payload)
    config = {
        "method": "Population-Consensus Subject-Disjoint Effect Transport (SDET)",
        "seed": SEED,
        "rank": RANK,
        "rho": RHO,
        "lambda_downside": LAMBDA_DOWNSIDE,
        "beta": BETA,
        "stop_gradient_proposal": True,
        "scale_control": "unit L2 norm per U/V column after every optimizer step",
        "baseline": {"architecture": "canonical vanilla EEGNet", "F1": 8, "D": 2, "F2": 16, "temporal_kernel": 64, "dropout": 0.25, "latent_dim": 64, "optimizer": "AdamW", "lr": BASELINE_LR, "weight_decay": BASELINE_WEIGHT_DECAY, "gradient_clip_norm": 5.0, "epoch_selection": "discovery mean subject BA; lower NLL; earlier epoch", "max_epochs": BASELINE_MAX_EPOCHS, "min_epochs": BASELINE_MIN_EPOCHS, "patience": BASELINE_PATIENCE, "batch_size": BASELINE_BATCH_SIZE},
        "sdet_optimizer": {"optimizer": "Adam", "lr": SDET_LR, "weight_decay": SDET_WEIGHT_DECAY, "epochs": SDET_EPOCHS, "episodes": SDET_EPISODES},
        "session_semantics": {"OpenBMI": {"support": [1], "query": [2]}, "WBCIC": {"support": [0, 1], "query": [2]}},
        "test_time_adaptation": False,
        "test_subject_adapter": False,
        "test_labels_used_for_training_or_selection": False,
        "seed1_run": False,
        "seed2_run": False,
        "sweep": False,
        "runtime": {"repo": str(REPO), "runtime_root": str(RUNTIME), "device": str(device)},
    }
    write_json(PROTOCOL / "SDET_SEED0_CONFIG.json", config)
    write_json(PROTOCOL / "BASELINE_REUSE_AUDIT.json", {"reused": False, "seed": 0, "candidate_artifact": str(REPO / "experiments" / "persist_eeg_canonical_eegnet_baseline"), "reason": "Existing canonical artifacts use different five-fold model-fit/discovery/outcome roles and do not prove exact V8 SEARCH=40/31 training population; exact paired test predictions cannot be recovered.", "rerun": True})
    write_json(PROTOCOL / "INFORMATION_MATCHING.json", {"OpenBMI": {"SDET_labels": "SEARCH S1 support and S2 query", "baseline_labels": "SEARCH S1+S2", "matched": True}, "WBCIC": {"SDET_labels": "SEARCH S1+S2 support and S3 query", "baseline_labels": "SEARCH S1+S2+S3", "matched": True}, "test_labels": "opened only after baseline, U, V, p and a_pop freeze"})

    all_results: list[dict[str, Any]] = []
    training_log: dict[str, Any] = {"seed": SEED, "device": str(device), "datasets": [], "test_labels_opened_after_freeze": True}
    interventions: dict[str, Any] = {}
    prepared_runs: dict[str, tuple[DatasetBundle, VanillaEEGNet, SDETModule, dict[str, Any], np.ndarray, np.ndarray]] = {}
    for dataset_name in ("OpenBMI", "WBCIC"):
        print(f"[{dataset_name}] loading frozen SEARCH and TEST metadata", flush=True)
        bundle = load_bundle(dataset_name, split)
        all_search = bundle.search_indices(bundle.search_subjects)
        mean, std = compute_normalizer(bundle.search_accessor, all_search)
        print(f"[{dataset_name}] rows search={len(bundle.search_rows)} test={len(bundle.test_rows)} channels={bundle.channels}", flush=True)
        model, baseline_info, _ = fit_baseline(bundle, mean, std, device)
        # Check A at a=0 is exact by construction before any SDET training.
        search_latents, latent_indices = extract_latents(model, bundle.search_accessor, all_search, mean, std, device)
        z_probe = torch.from_numpy(search_latents[: min(8, len(search_latents))]).to(device)
        zero = torch.zeros(RANK, device=device)
        probe_module = SDETModule().to(device)
        probe_module.eval()
        with torch.no_grad():
            baseline_logits = model.head(z_probe)
            zero_logits = model.head(probe_module.transformed(z_probe, zero))
            zero_diff = float(torch.max(torch.abs(baseline_logits - zero_logits)).cpu())
        del probe_module
        if zero_diff > 1e-6:
            raise RuntimeError(f"Check A failed for {dataset_name}: {zero_diff}")
        labels = bundle.load_search_labels(latent_indices)
        episodes = build_episodes(bundle.search_subjects, dataset_name)
        module, sdet_info, diagnostics = train_sdet(bundle, model, search_latents, latent_indices, labels, device, episodes)
        interventions[dataset_name] = {key: value for key, value in diagnostics.items() if key not in {"U", "V", "p", "g_pop", "a_pop"}}
        interventions[dataset_name].update({"U": diagnostics["U"], "V": diagnostics["V"], "p": diagnostics["p"], "g_pop": diagnostics["g_pop"], "a_pop": diagnostics["a_pop"], "search_subjects": bundle.search_subjects})
        # Keep every TEST label file closed while either dataset is still
        # training.  Scoring is deferred until both baselines and both frozen
        # population interventions have been constructed.
        prepared_runs[dataset_name] = (bundle, model, module, diagnostics, mean, std)
        training_log["datasets"].append({"dataset": dataset_name, "n_search_subjects": len(bundle.search_subjects), "n_test_subjects": len(bundle.test_subjects), "search_trials": len(bundle.search_rows), "test_trials": len(bundle.test_rows), "normalizer_mean": mean, "normalizer_std": std, "baseline": baseline_info, "sdet": sdet_info, "check_A_max_abs_logit_diff": zero_diff, "episode_schedule": episodes, "check_B_episode_subject_disjoint": all(not (set(ep["support_subjects"]) & set(ep["query_subjects"])) for ep in episodes), "check_C_test_excluded_from_episodes": all(not (set(bundle.test_subjects) & (set(ep["support_subjects"]) | set(ep["query_subjects"]))) for ep in episodes), "check_D_final_population_subjects": bundle.search_subjects, "check_F_finite": True, "check_G_U_column_norms": diagnostics["U_column_norms"], "check_G_V_column_norms": diagnostics["V_column_norms"], "check_H_intervention_norm": diagnostics["a_pop_norm"]})
        print(f"[{dataset_name}] training and intervention frozen; TEST labels remain closed", flush=True)
    # Only now, after both dataset models and interventions are frozen, open
    # each authorized V8 internal-holdout label file for final paired scoring.
    for dataset_name in ("OpenBMI", "WBCIC"):
        bundle, model, module, diagnostics, mean, std = prepared_runs[dataset_name]
        all_results.extend(score_test(bundle, model, module, diagnostics, mean, std, device))
        print(f"[{dataset_name}] test scoring complete", flush=True)
    write_csv(OUT / "SEED0_TEST_PER_SUBJECT.csv", all_results)
    aggregates = [aggregate(name, [row for row in all_results if row["dataset"] == name]) for name in ("OpenBMI", "WBCIC")]
    write_csv(OUT / "SEED0_TEST_RESULTS.csv", aggregates)
    write_json(OUT / "SEED0_TRAINING_LOG.json", {**training_log, "runtime_seconds_total": time.perf_counter() - started})
    intervention_payload = {"method": config["method"], "seed": SEED, "frozen_before_test_labels": True, "datasets": interventions}
    write_json(OUT / "SEED0_INTERVENTION.json", intervention_payload)
    valid_protocol = all(item["check_B_episode_subject_disjoint"] and item["check_C_test_excluded_from_episodes"] and item["check_A_max_abs_logit_diff"] <= 1e-6 and item["check_F_finite"] and max(item["check_G_U_column_norms"]) < 1.000001 and min(item["check_G_U_column_norms"]) > 0.999999 and max(item["check_G_V_column_norms"]) < 1.000001 and min(item["check_G_V_column_norms"]) > 0.999999 for item in training_log["datasets"])
    if not valid_protocol:
        terminal = "SDET_SEED0_PROTOCOL_INVALID"
    else:
        deltas = [float(item["delta_BA_pp"]) for item in aggregates]
        collapses = [float(item["sdet_mean_subject_BA"]) < 0.5 for item in aggregates]
        obvious_harm = any(float(item["harmed_subject_ratio"]) > 0.5 or float(item["worst_quartile_subject_delta_BA_pp"]) < -10.0 for item in aggregates)
        if any(collapses) or all(delta < 0 for delta in deltas):
            terminal = "SDET_SEED0_TEST_NEGATIVE_STOP"
        elif all(delta > 0 for delta in deltas) and not obvious_harm:
            terminal = "SDET_SEED0_TEST_POSITIVE_STOP"
        else:
            terminal = "SDET_SEED0_TEST_MIXED_STOP"
    lookup = {row["dataset"]: row for row in aggregates}
    decision = [
        "# SDET seed-0 decision",
        "",
        "This is an authorized V8 internal holdout evaluation, not independent external confirmation.",
        "",
        f"1. OpenBMI baseline BA: {lookup['OpenBMI']['baseline_mean_subject_BA']:.6f}",
        f"2. OpenBMI SDET BA: {lookup['OpenBMI']['sdet_mean_subject_BA']:.6f}",
        f"3. OpenBMI delta: {lookup['OpenBMI']['delta_BA_pp']:+.3f} pp",
        f"4. WBCIC baseline BA: {lookup['WBCIC']['baseline_mean_subject_BA']:.6f}",
        f"5. WBCIC SDET BA: {lookup['WBCIC']['sdet_mean_subject_BA']:.6f}",
        f"6. WBCIC delta: {lookup['WBCIC']['delta_BA_pp']:+.3f} pp",
        f"7. Direction consistent: {lookup['OpenBMI']['delta_BA_pp'] * lookup['WBCIC']['delta_BA_pp'] > 0}",
        f"8. Improved subjects: OpenBMI {lookup['OpenBMI']['improved_subjects']}/{lookup['OpenBMI']['n_subjects']}; WBCIC {lookup['WBCIC']['improved_subjects']}/{lookup['WBCIC']['n_subjects']}",
        f"9. Worst quartile delta (pp): OpenBMI {lookup['OpenBMI']['worst_quartile_subject_delta_BA_pp']:+.3f}; WBCIC {lookup['WBCIC']['worst_quartile_subject_delta_BA_pp']:+.3f}",
        "10. Mean-improvement concentration: inspect paired table; no subject-specific adapter was used.",
        f"11. Intervention norms: OpenBMI {interventions['OpenBMI']['a_pop_norm']:.6g}; WBCIC {interventions['WBCIC']['a_pop_norm']:.6g}",
        f"12. Protocol validity: {valid_protocol}",
        "",
        f"Terminal state: **{terminal}**",
    ]
    (OUT / "SEED0_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    (EXP / "METHOD.md").write_text("# Population-Consensus SDET\n\nSDET uses the exact frozen canonical EEGNet backbone and a rank-4 low-rank latent intervention. Per-support-subject class-balanced gradients are unit-normalized and averaged without a second normalization; a global tanh gate constructs one frozen population adapter. Support/query episodes are subject-disjoint and test subjects are excluded. The internal V8 holdout is opened only after baseline and intervention freezing.\n", encoding="utf-8")
    (EXP / "README.md").write_text("# PERSIST-EEG SDET seed 0\n\nThis directory contains the one-shot seed-0 viability experiment. Runtime models, raw EEG and caches remain outside Git. See `outputs/SEED0_DECISION.md` and `outputs/SEED0_TEST_PER_SUBJECT.csv`.\n", encoding="utf-8")
    print(f"TERMINAL={terminal}", flush=True)
    print(f"TOTAL_RUNTIME_SECONDS={time.perf_counter() - started:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException:
        # Detached server runs must preserve the traceback even if the SSH
        # channel or redirected stderr disappears.  This is diagnostics only;
        # it does not alter the scientific configuration or retry anything.
        try:
            RUNTIME.mkdir(parents=True, exist_ok=True)
            (RUNTIME / "sdet_exception.log").write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        raise
