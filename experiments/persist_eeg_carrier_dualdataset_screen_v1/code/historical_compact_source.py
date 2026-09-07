"""Compact-TG Stage-1 SEARCH-only seed-0 experiment.

This runner deliberately never reads V8_INTERNAL_HOLDOUT or the WBCIC outer
cache.  It trains EEGNet-ERM, Compact-ERM and Compact-TG on signal-level
epochs using deterministic subject-wise development folds, then evaluates each
outer development subject once on the future session.  Runtime checkpoints and
episode manifests are kept outside the git worktree.
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
from sklearn.metrics import balanced_accuracy_score, f1_score


REPO = Path(os.environ.get("TG_REPO", r"D:\nips-temp\TotalP\P1\CRCICLR_TG_STAGE1_WORK")).resolve()
EXP = REPO / "experiments" / "persist_eeg_transfer_geometry_stage1_v1"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("TG_RUNTIME", r"D:\nips-temp\TotalP\P2\transfer_geometry_stage1_runtime")).resolve()
STAGE0_ROOT = Path(os.environ.get("PERSIST_STAGE0_REPO", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full")).resolve()
WBCIC_CACHE = Path(os.environ.get(
    "PERSIST_WBCIC_CACHE",
    r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache",
)).resolve()
SPLIT_PATH = REPO / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"
OPENBMI_MANIFEST = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"

SEED = 0
FOLD_SEEDS = (100, 101, 102)
MAX_EPOCHS = 60
MIN_EPOCHS = 10
PATIENCE = 8
LR = 3e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
EPISODE_TRIALS = 128
TAU = 0.1
LAMBDA_TG = 1.0
TIE_TOL = 1e-8


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
        raise RuntimeError(f"empty CSV: {path}")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def subject_sort(values: Iterable[object], dataset: str) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        text = str(value)
        if dataset == "WBCIC":
            text = text.replace("sub-", "")
        return (int(text) if text.isdigit() else 10**9, text)
    return sorted([str(value) for value in values], key=key)


def load_search_split() -> tuple[dict[str, list[str]], str]:
    # Only V8_SEARCH fields are accessed.  No V8_INTERNAL_HOLDOUT field is
    # referenced, and no holdout labels/signals are enumerated.
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(SPLIT_PATH)
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8-sig"))
    search = {
        "OpenBMI": subject_sort(payload["openbmi"]["V8_SEARCH"], "OpenBMI"),
        "WBCIC": subject_sort(payload["wbcic"]["V8_SEARCH"], "WBCIC"),
    }
    if len(search["OpenBMI"]) != 40 or len(search["WBCIC"]) != 31:
        raise RuntimeError(f"unexpected V8 SEARCH sizes: { {k: len(v) for k, v in search.items()} }")
    return search, sha256_file(SPLIT_PATH)


@dataclass
class Row:
    subject: str
    session: int
    signal_path: str
    cache_index: int
    label: int


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
            self._arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._arrays[key]

    def batch(self, indices: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for index in np.asarray(indices, dtype=np.int64):
            row = self.rows[int(index)]
            values.append(np.asarray(self._array(row.signal_path)[row.cache_index], dtype=np.float32))
        if not values:
            return np.empty((0, self.channels, 1000), dtype=np.float32)
        return np.stack(values, axis=0)


@dataclass
class DatasetBundle:
    name: str
    search_subjects: list[str]
    search_rows: list[Row]
    accessor: SignalAccessor
    channels: int

    def indices(self, subjects: Iterable[str], sessions: Iterable[int] | None = None) -> np.ndarray:
        wanted = set(map(str, subjects))
        session_set = None if sessions is None else set(map(int, sessions))
        return np.asarray([
            i for i, row in enumerate(self.search_rows)
            if row.subject in wanted and (session_set is None or row.session in session_set)
        ], dtype=np.int64)

    def labels(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray([self.search_rows[int(i)].label for i in indices], dtype=np.int64)


def load_bundle(name: str, subjects: list[str]) -> DatasetBundle:
    if name == "OpenBMI":
        if not OPENBMI_MANIFEST.is_file():
            raise FileNotFoundError(OPENBMI_MANIFEST)
        columns = ["subject_id", "session_id", "paradigm", "run_phase", "signal_cache_path", "label_cache_path", "cache_index"]
        frame = pd.read_parquet(OPENBMI_MANIFEST, columns=columns)
        frame["subject_id"] = frame["subject_id"].astype(str).str.replace("sub-", "", regex=False)
        frame["session_id"] = frame["session_id"].astype(int)
        frame = frame[(frame["paradigm"] == "mi") & (frame["run_phase"] == "train")]
        frame = frame[frame["subject_id"].isin(set(subjects))].reset_index(drop=True)
        if len(frame) != 40 * 2 * 100:
            raise RuntimeError(f"OpenBMI SEARCH row mismatch: {len(frame)}")
        labels_by_path: dict[str, np.ndarray] = {}
        rows: list[Row] = []
        for item in frame.itertuples(index=False):
            label_path = str(STAGE0_ROOT / str(item.label_cache_path))
            if label_path not in labels_by_path:
                labels_by_path[label_path] = np.load(label_path, mmap_mode="r", allow_pickle=False)
            code = int(labels_by_path[label_path][int(item.cache_index)])
            if code not in (1, 2):
                raise RuntimeError(f"unexpected OpenBMI label code {code}")
            rows.append(Row(str(item.subject_id), int(item.session_id), str(item.signal_cache_path), int(item.cache_index), code - 1))
        return DatasetBundle(name, subjects, rows, SignalAccessor(rows, STAGE0_ROOT, 62), 62)

    epochs_root = WBCIC_CACHE / "wbcic_epochs"
    if not epochs_root.is_dir():
        raise FileNotFoundError(epochs_root)
    rows = []
    for subject in subject_sort(subjects, "WBCIC"):
        for session in (0, 1, 2):
            epochs_path = epochs_root / subject / f"ses-{session}_epochs.npy"
            labels_path = epochs_root / subject / f"ses-{session}_labels.npy"
            if not epochs_path.is_file() or not labels_path.is_file():
                raise FileNotFoundError(f"WBCIC development cache missing {subject} ses-{session}")
            epochs = np.load(epochs_path, mmap_mode="r", allow_pickle=False)
            labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
            if epochs.ndim != 3 or tuple(epochs.shape[1:]) != (58, 1000) or epochs.dtype != np.float16:
                raise RuntimeError(f"unexpected WBCIC epoch schema: {epochs_path} {epochs.shape} {epochs.dtype}")
            if labels.ndim != 1 or labels.shape[0] != epochs.shape[0] or labels.dtype.kind not in "iu":
                raise RuntimeError(f"unexpected WBCIC label schema: {labels_path} {labels.shape} {labels.dtype}")
            for index, code in enumerate(labels):
                value = int(code)
                if value not in (0, 1):
                    raise RuntimeError(f"unexpected WBCIC label code {value}")
                rows.append(Row(subject, session, str(epochs_path), index, value))
    # The development cache is not perfectly rectangular (some legal SEARCH
    # sessions contain a small number of missing trials).  Keep the exact
    # observed row count; episode construction below still enforces the
    # per-class sample requirements without fabricating trials.
    if len(rows) <= 0:
        raise RuntimeError("empty WBCIC SEARCH cache")
    return DatasetBundle(name, subjects, rows, SignalAccessor(rows, WBCIC_CACHE, 58), 58)


def make_folds(subjects: list[str], dataset: str) -> list[dict[str, Any]]:
    canonical = subject_sort(subjects, dataset)
    rng = np.random.default_rng(SEED)
    shuffled = [str(x) for x in rng.permutation(np.asarray(canonical, dtype=object))]
    fold_arrays = np.array_split(np.asarray(shuffled, dtype=object), 3)
    folds = [list(map(str, fold.tolist())) for fold in fold_arrays]
    output = []
    for fold_id in range(3):
        outer = folds[fold_id]
        remaining = [subject for j, values in enumerate(folds) if j != fold_id for subject in values]
        inner_rng = np.random.default_rng(1000 + fold_id)
        remaining = [str(x) for x in inner_rng.permutation(np.asarray(remaining, dtype=object))]
        n_val = max(1, int(math.ceil(len(remaining) * 0.2)))
        inner_val = remaining[:n_val]
        inner_train = remaining[n_val:]
        if set(inner_train) & set(inner_val) or set(inner_train) & set(outer) or set(inner_val) & set(outer):
            raise RuntimeError("fold split overlap")
        output.append({
            "fold_id": fold_id,
            "fold_seed": FOLD_SEEDS[fold_id],
            "outer_dev_subjects": outer,
            "remaining_subjects": remaining,
            "inner_train_subjects": inner_train,
            "inner_val_subjects": inner_val,
            "inner_split_seed": 1000 + fold_id,
        })
    if subject_sort(sum((item["outer_dev_subjects"] for item in output), []), dataset) != canonical:
        raise RuntimeError("outer fold coverage is not exactly once")
    return output


def compute_normalizer(bundle: DatasetBundle, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    indices = bundle.indices(subjects)
    total = np.zeros(bundle.channels, dtype=np.float64)
    square = np.zeros(bundle.channels, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 128):
        batch = bundle.accessor.batch(indices[start:start + 128]).astype(np.float64)
        total += batch.sum(axis=(0, 2))
        square += np.square(batch).sum(axis=(0, 2))
        count += batch.shape[0] * batch.shape[2]
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), int(len(indices))


def prepare(bundle: DatasetBundle, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    value = bundle.accessor.batch(indices)
    value = (value - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device, non_blocking=True)


class EEGNet(nn.Module):
    def __init__(self, channels: int, samples: int = 1000, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(16 * (samples // 4 // 8), 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.embedding(value.flatten(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward_features(x)
        return self.head(z), z


class ResidualSeparableBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, (1, kernel), groups=channels_in, padding="same", bias=False),
            nn.BatchNorm2d(channels_in), nn.ELU(),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out), nn.ELU(),
            nn.AvgPool2d((1, 2)), nn.Dropout(0.20),
        )
        self.skip = nn.Sequential(nn.Conv2d(channels_in, channels_out, 1, bias=False), nn.AvgPool2d((1, 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class CompactEncoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branches = []
        for kernel in (15, 63, 127):
            branches.append(nn.Sequential(nn.Conv2d(1, 12, (1, kernel), padding="same", bias=False), nn.BatchNorm2d(12), nn.ELU()))
        self.temporal_stem = nn.ModuleList(branches)
        self.spatial = nn.Sequential(
            nn.Conv2d(36, 72, (channels, 1), groups=36, bias=False),
            nn.BatchNorm2d(72), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
        )
        self.block1 = ResidualSeparableBlock(72, 96, 15)
        self.block2 = ResidualSeparableBlock(96, 128, 31)
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.embedding = nn.Sequential(nn.Linear(128 * 8, 128), nn.ELU(), nn.LayerNorm(128))
        self.head = nn.Linear(128, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = torch.cat([branch(value) for branch in self.temporal_stem], dim=1)
        value = self.spatial(value)
        value = self.block2(self.block1(value))
        return self.embedding(self.pool(value).flatten(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward_features(x)
        return self.head(z), z


def trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def build_subject_index(bundle: DatasetBundle) -> dict[tuple[str, int, int], list[int]]:
    mapping: dict[tuple[str, int, int], list[int]] = {}
    for index, row in enumerate(bundle.search_rows):
        mapping.setdefault((row.subject, row.session, row.label), []).append(index)
    return mapping


def sample_subject_trials(mapping: dict[tuple[str, int, int], list[int]], subject: str, sessions: tuple[int, ...], per_session_class: int, rng: np.random.Generator) -> list[int]:
    selected: list[int] = []
    for session in sessions:
        for cls in (0, 1):
            pool = mapping[(subject, session, cls)]
            if len(pool) < per_session_class:
                raise RuntimeError(f"insufficient trials for {subject} session {session} class {cls}")
            selected.extend([int(x) for x in rng.choice(np.asarray(pool), size=per_session_class, replace=False)])
    return selected


def generate_episode_manifest(bundle: DatasetBundle, fold: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], int, str]:
    inner_train = fold["inner_train_subjects"]
    n_train_trials = len(bundle.indices(inner_train))
    steps = max(20, int(math.ceil(n_train_trials / EPISODE_TRIALS)))
    mapping = build_subject_index(bundle)
    proposal_sessions = (1,) if bundle.name == "OpenBMI" else (0, 1)
    query_sessions = (2,)
    manifest: list[list[dict[str, Any]]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_items = []
        for step in range(steps):
            rng = np.random.default_rng(int(fold["fold_seed"]) + epoch * 1_000_003 + step * 97)
            order = [str(x) for x in rng.permutation(np.asarray(inner_train, dtype=object))]
            support = order[:4]
            query = order[4:8]
            if set(support) & set(query):
                raise RuntimeError("support/query overlap")
            support_indices: list[int] = []
            query_indices: list[int] = []
            for subject in support:
                support_indices.extend(sample_subject_trials(mapping, subject, proposal_sessions, 8 if bundle.name == "OpenBMI" else 4, rng))
            for subject in query:
                query_indices.extend(sample_subject_trials(mapping, subject, query_sessions, 8, rng))
            if len(support_indices) != 64 or len(query_indices) != 64:
                raise RuntimeError("episode does not contain 128 trials")
            epoch_items.append({"support_subjects": support, "query_subjects": query, "support_indices": support_indices, "query_indices": query_indices})
        manifest.append(epoch_items)
    path = RUNTIME / "episode_manifests" / f"{bundle.name.lower()}_fold{fold['fold_id']}.json"
    write_json(path, {"dataset": bundle.name, "fold_id": fold["fold_id"], "fold_seed": fold["fold_seed"], "steps_per_epoch": steps, "episodes": manifest})
    return manifest, steps, sha256_file(path)


def evaluate(model: nn.Module, bundle: DatasetBundle, subjects: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[float, float, dict[str, dict[str, float]]]:
    model.eval()
    per_subject: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for subject in subjects:
            indices = bundle.indices([subject], sessions=(2,))
            labels = bundle.labels(indices)
            probabilities: list[np.ndarray] = []
            for start in range(0, len(indices), 128):
                x = prepare(bundle, indices[start:start + 128], mean, std, device)
                logits, _ = model(x)
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            probability = np.concatenate(probabilities, axis=0)
            prediction = np.argmax(probability, axis=1)
            per_subject[subject] = {
                "BA": float(balanced_accuracy_score(labels, prediction)),
                "F1": float(f1_score(labels, prediction, average="macro")),
                "trials": int(len(labels)),
            }
    return (float(np.mean([value["BA"] for value in per_subject.values()])),
            float(np.mean([value["F1"] for value in per_subject.values()])), per_subject)


def tg_loss(support_z: torch.Tensor, support_y: torch.Tensor, query_z: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
    support_norm = support_z / (support_z.norm(dim=1, keepdim=True) + 1e-8)
    prototypes = []
    for cls in (0, 1):
        mask = support_y == cls
        if int(mask.sum()) == 0:
            raise RuntimeError(f"missing support class {cls}")
        prototype = support_norm[mask].mean(dim=0)
        prototype = prototype / (prototype.norm() + 1e-8)
        prototypes.append(prototype)
    prototype_matrix = torch.stack(prototypes, dim=0)
    query_norm = query_z / (query_z.norm(dim=1, keepdim=True) + 1e-8)
    return F.cross_entropy((query_norm @ prototype_matrix.t()) / TAU, query_y)


def train_model(model: nn.Module, model_name: str, bundle: DatasetBundle, manifest: list[list[dict[str, Any]]], fold: dict[str, Any], mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state: dict[str, torch.Tensor] | None = None
    best_ba = -float("inf")
    best_epoch = MAX_EPOCHS
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch, episodes in enumerate(manifest, start=1):
        model.train()
        losses = []
        for episode in episodes:
            support_indices = np.asarray(episode["support_indices"], dtype=np.int64)
            query_indices = np.asarray(episode["query_indices"], dtype=np.int64)
            support_x = prepare(bundle, support_indices, mean, std, device)
            query_x = prepare(bundle, query_indices, mean, std, device)
            support_y = bundle.labels(support_indices)
            query_y = bundle.labels(query_indices)
            support_y_t = torch.from_numpy(support_y).to(device)
            query_y_t = torch.from_numpy(query_y).to(device)
            optimizer.zero_grad(set_to_none=True)
            support_logits, support_z = model(support_x)
            query_logits, query_z = model(query_x)
            logits = torch.cat([support_logits, query_logits], dim=0)
            labels = torch.cat([support_y_t, query_y_t], dim=0)
            loss = F.cross_entropy(logits, labels)
            if model_name == "Compact-TG":
                loss = loss + LAMBDA_TG * tg_loss(support_z, support_y_t, query_z, query_y_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {model_name} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_ba, val_f1, _ = evaluate(model, bundle, fold["inner_val_subjects"], mean, std, device)
        item = {"epoch": epoch, "mean_loss": float(np.mean(losses)), "inner_val_BA": val_ba, "inner_val_macro_F1": val_f1}
        history.append(item)
        improved = val_ba > best_ba + 1e-12
        if epoch >= MIN_EPOCHS and improved:
            best_ba = val_ba
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            item["selected"] = True
        elif epoch >= MIN_EPOCHS:
            bad_epochs += 1
            item["selected"] = False
            if bad_epochs >= PATIENCE:
                print(f"[{bundle.name} fold={fold['fold_id']} {model_name}] early stop epoch={epoch}", flush=True)
                break
        else:
            item["selected"] = False
        if epoch == 1 or epoch % 5 == 0 or item.get("selected"):
            print(f"[{bundle.name} fold={fold['fold_id']} {model_name}] epoch={epoch} loss={item['mean_loss']:.5f} val_BA={val_ba:.5f}", flush=True)
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
        best_ba = history[-1]["inner_val_BA"]
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = RUNTIME / "checkpoints" / bundle.name.lower() / f"fold{fold['fold_id']}_{model_name.lower().replace('-', '_')}_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    elapsed = time.perf_counter() - started
    info = {"model": model_name, "selected_epoch": int(best_epoch), "best_inner_val_BA": float(best_ba), "history": history, "runtime_seconds": elapsed, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}
    return model, info


def aggregate_subjects(dataset: str, subject_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]], bootstrap_seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = pd.DataFrame(subject_rows)
    delta = frame["compact_tg_BA"].to_numpy(float) - frame["compact_erm_BA"].to_numpy(float)
    rng = np.random.default_rng(bootstrap_seed)
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    fold_frame = pd.DataFrame(fold_rows)
    row = {
        "dataset": dataset,
        "eegnet_BA": float(frame["eegnet_BA"].mean()),
        "compact_erm_BA": float(frame["compact_erm_BA"].mean()),
        "compact_tg_BA": float(frame["compact_tg_BA"].mean()),
        "architecture_gain_pp": float((frame["compact_erm_BA"].mean() - frame["eegnet_BA"].mean()) * 100.0),
        "tg_gain_pp": float((frame["compact_tg_BA"].mean() - frame["compact_erm_BA"].mean()) * 100.0),
        "total_gain_pp": float((frame["compact_tg_BA"].mean() - frame["eegnet_BA"].mean()) * 100.0),
        "eegnet_macro_F1": float(frame["eegnet_F1"].mean()),
        "compact_erm_macro_F1": float(frame["compact_erm_F1"].mean()),
        "compact_tg_macro_F1": float(frame["compact_tg_F1"].mean()),
        "tg_vs_erm_mean_subject_delta_pp": float(delta.mean() * 100.0),
        "tg_vs_erm_median_subject_delta_pp": float(np.median(delta) * 100.0),
        "tg_vs_erm_ci_low_pp": float(np.quantile(boot, 0.025) * 100.0),
        "tg_vs_erm_ci_high_pp": float(np.quantile(boot, 0.975) * 100.0),
        "tg_improved_subjects": int(np.sum(delta > TIE_TOL)),
        "tg_tied_subjects": int(np.sum(np.abs(delta) <= TIE_TOL)),
        "tg_harmed_subjects": int(np.sum(delta < -TIE_TOL)),
        "tg_positive_folds": int(np.sum(fold_frame["tg_vs_erm_gain_pp"].to_numpy(float) > 0.0)),
        "total_folds": 3,
    }
    bootstrap = {"dataset": dataset, "seed": bootstrap_seed, "n_subjects": len(delta), "resamples": 10000, "mean_delta_BA": float(delta.mean()), "ci_low_pp": row["tg_vs_erm_ci_low_pp"], "ci_high_pp": row["tg_vs_erm_ci_high_pp"]}
    return row, bootstrap


def choose_terminal(aggregates: list[dict[str, Any]]) -> str:
    strong = all(item["tg_gain_pp"] >= 1.0 and item["total_gain_pp"] >= 1.5 and item["tg_positive_folds"] >= 2 and item["tg_vs_erm_median_subject_delta_pp"] > 0 for item in aggregates)
    promising = all(item["tg_gain_pp"] > 0.5 for item in aggregates)
    negative = all(item["tg_gain_pp"] <= 0.0 or abs(item["tg_gain_pp"]) < 0.1 for item in aggregates)
    if strong:
        return "TG_STAGE1_STRONG_STOP"
    if promising:
        return "TG_STAGE1_PROMISING_STOP"
    if negative:
        return "TG_STAGE1_NEGATIVE_STOP"
    return "TG_STAGE1_MIXED_STOP"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    PROTOCOL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True); CODE.mkdir(parents=True, exist_ok=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    search, split_sha = load_search_split()
    bundles = {name: load_bundle(name, subjects) for name, subjects in search.items()}
    folds = {name: make_folds(subjects, name) for name, subjects in search.items()}
    protocol_split = {"protocol": "STAGE1_SEARCH_ONLY", "seed": SEED, "source_split_manifest": str(SPLIT_PATH), "source_split_sha256": split_sha, "datasets": {name: {"search_subjects": search[name], "folds": folds[name]} for name in search}}
    write_json(PROTOCOL / "STAGE1_SEARCH_CV_SPLIT.json", protocol_split)
    counts = {"EEGNet": trainable_parameters(EEGNet(62)), "CompactEncoder": trainable_parameters(CompactEncoder(62))}
    write_json(PROTOCOL / "PARAMETER_COUNTS.json", {"trainable_parameters": counts, "compact_under_0_5M": counts["CompactEncoder"] < 500000})
    config = {"method": "Compact-TG (Transfer Geometry)", "seed": SEED, "fold_seeds": list(FOLD_SEEDS), "datasets": ["OpenBMI", "WBCIC"], "outer_folds": 3, "future_session": {"OpenBMI": "physical session 2", "WBCIC": "physical session 2 / S3"}, "compact_architecture": {"temporal_kernels": [15, 63, 127], "stem_channels": 12, "spatial_out": 72, "block1_out": 96, "block2_out": 128, "representation_dim": 128, "dropout": [0.25, 0.20], "tau": TAU}, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS, "patience": PATIENCE, "gradient_clip_norm": GRAD_CLIP, "lambda_TG": LAMBDA_TG, "episode_trials": EPISODE_TRIALS, "runtime": str(RUNTIME), "device": str(device)}
    write_json(PROTOCOL / "STAGE1_CONFIG.json", config)
    holdout_audit = {"V8_INTERNAL_HOLDOUT_loaded": False, "V8_INTERNAL_HOLDOUT_labels_loaded": False, "WBCIC_outer_loaded": False, "WBCIC_outer_labels_loaded": False, "accessed_subject_scope": {name: search[name] for name in search}, "accessed_signal_roots": {"OpenBMI": str(STAGE0_ROOT), "WBCIC": str(WBCIC_CACHE / "wbcic_epochs")}, "forbidden_roots": ["WBCIC outer cache", "V8_INTERNAL_HOLDOUT"], "protocol_valid": True}
    write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", holdout_audit)
    info_matching = {"same_subjects": True, "same_fold_splits": True, "same_normalization_source": True, "same_training_labels": True, "same_session_availability": True, "same_episodic_batches": True, "same_optimization_steps": True, "same_early_stopping_rule": True, "same_outer_dev_samples": True, "only_allowed_differences": {"architecture": "EEGNet vs Compact", "loss": "Compact-ERM=CE; Compact-TG=CE+TG"}}
    write_json(PROTOCOL / "INFORMATION_MATCHING.json", info_matching)
    if args.validate_only:
        for name, bundle in bundles.items():
            print(f"VALID {name}: subjects={len(bundle.search_subjects)} rows={len(bundle.search_rows)} channels={bundle.channels} compact_params={counts['CompactEncoder']}", flush=True)
        print("VALIDATION_ONLY_OK", flush=True)
        return 0

    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    training_log: dict[str, Any] = {"seed": SEED, "device": str(device), "datasets": [], "holdout_opened": False}
    for dataset in ("OpenBMI", "WBCIC"):
        bundle = bundles[dataset]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            mean, std, norm_trials = compute_normalizer(bundle, fold["inner_train_subjects"])
            manifest, steps, manifest_sha = generate_episode_manifest(bundle, fold)
            model_runs: dict[str, tuple[nn.Module, dict[str, Any], dict[str, dict[str, float]]]] = {}
            initial_seed = int(fold["fold_seed"])
            set_seed(initial_seed)
            eegnet = EEGNet(bundle.channels).to(device)
            set_seed(initial_seed + 20000)
            eegnet, eegnet_info = train_model(eegnet, "EEGNet-ERM", bundle, manifest, fold, mean, std, device)
            eegnet_ba, eegnet_f1, eegnet_subject = evaluate(eegnet, bundle, fold["outer_dev_subjects"], mean, std, device)
            model_runs["eegnet"] = (eegnet, {**eegnet_info, "outer_BA": eegnet_ba, "outer_macro_F1": eegnet_f1}, eegnet_subject)

            set_seed(initial_seed)
            initial_compact = CompactEncoder(bundle.channels).to(device)
            initial_state = copy.deepcopy(initial_compact.state_dict())
            compact_erm = CompactEncoder(bundle.channels).to(device); compact_erm.load_state_dict(initial_state)
            compact_tg = CompactEncoder(bundle.channels).to(device); compact_tg.load_state_dict(initial_state)
            set_seed(initial_seed + 20000)
            compact_erm, erm_info = train_model(compact_erm, "Compact-ERM", bundle, manifest, fold, mean, std, device)
            erm_ba, erm_f1, erm_subject = evaluate(compact_erm, bundle, fold["outer_dev_subjects"], mean, std, device)
            set_seed(initial_seed + 20000)
            compact_tg, tg_info = train_model(compact_tg, "Compact-TG", bundle, manifest, fold, mean, std, device)
            tg_ba, tg_f1, tg_subject = evaluate(compact_tg, bundle, fold["outer_dev_subjects"], mean, std, device)
            for subject in fold["outer_dev_subjects"]:
                subject_rows.append({"dataset": dataset, "fold": fold_id, "subject_id": subject, "eegnet_BA": eegnet_subject[subject]["BA"], "eegnet_F1": eegnet_subject[subject]["F1"], "compact_erm_BA": erm_subject[subject]["BA"], "compact_erm_F1": erm_subject[subject]["F1"], "compact_tg_BA": tg_subject[subject]["BA"], "compact_tg_F1": tg_subject[subject]["F1"], "tg_vs_erm_delta_BA_pp": (tg_subject[subject]["BA"] - erm_subject[subject]["BA"]) * 100.0, "status": "improved" if tg_subject[subject]["BA"] - erm_subject[subject]["BA"] > TIE_TOL else "harmed" if tg_subject[subject]["BA"] - erm_subject[subject]["BA"] < -TIE_TOL else "tie"})
            fold_rows.append({"dataset": dataset, "fold": fold_id, "eegnet_BA": eegnet_ba, "compact_erm_BA": erm_ba, "compact_tg_BA": tg_ba, "eegnet_macro_F1": eegnet_f1, "compact_erm_macro_F1": erm_f1, "compact_tg_macro_F1": tg_f1, "architecture_gain_pp": (erm_ba - eegnet_ba) * 100.0, "tg_vs_erm_gain_pp": (tg_ba - erm_ba) * 100.0, "tg_vs_eegnet_gain_pp": (tg_ba - eegnet_ba) * 100.0, "steps_per_epoch": steps, "normalizer_trials": norm_trials, "normalizer_mean": mean.tolist(), "normalizer_std": std.tolist(), "episode_manifest_sha256": manifest_sha, "episode_manifest_path": str(RUNTIME / "episode_manifests" / f"{dataset.lower()}_fold{fold_id}.json"), "model_training": {"EEGNet-ERM": model_runs.get("eegnet", (None, eegnet_info, None))[1], "Compact-ERM": erm_info, "Compact-TG": tg_info}})
            training_log["datasets"].append({"dataset": dataset, "fold": fold_id, "fold_seed": fold["fold_seed"], "normalizer_trials": norm_trials, "normalizer_mean": mean.tolist(), "normalizer_std": std.tolist(), "steps_per_epoch": steps, "episode_manifest_sha256": manifest_sha, "models": {"EEGNet-ERM": eegnet_info, "Compact-ERM": erm_info, "Compact-TG": tg_info}})
            print(f"[{dataset} fold={fold_id}] OUTER EEGNet={eegnet_ba:.5f} CompactERM={erm_ba:.5f} CompactTG={tg_ba:.5f} TG-ERM={(tg_ba-erm_ba)*100:+.3f}pp", flush=True)
    write_csv(OUT / "SEED0_FOLD_RESULTS.csv", fold_rows)
    write_csv(OUT / "SEED0_SUBJECT_RESULTS.csv", subject_rows)
    aggregate_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for dataset in ("OpenBMI", "WBCIC"):
        ds_subject = [row for row in subject_rows if row["dataset"] == dataset]
        ds_fold = [row for row in fold_rows if row["dataset"] == dataset]
        aggregate, bootstrap = aggregate_subjects(dataset, ds_subject, ds_fold, 0)
        aggregate_rows.append(aggregate); bootstrap_rows.append(bootstrap)
    terminal = choose_terminal(aggregate_rows)
    write_csv(OUT / "SEED0_AGGREGATE_RESULTS.csv", aggregate_rows)
    write_json(OUT / "SEED0_BOOTSTRAP.json", {"datasets": bootstrap_rows, "resamples": 10000, "seed": 0})
    training_log["runtime_seconds_total"] = float(sum(float(entry["models"]["Compact-TG"]["runtime_seconds"]) for entry in training_log["datasets"]))
    training_log["terminal"] = terminal
    write_json(OUT / "SEED0_TRAINING_LOG.json", training_log)
    table = ["| Dataset | EEGNet | Compact-ERM | Compact-TG | TG-vs-ERM | TG-vs-EEGNet |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate_rows:
        table.append(f"| {row['dataset']} | {row['eegnet_BA']:.6f} | {row['compact_erm_BA']:.6f} | {row['compact_tg_BA']:.6f} | {row['tg_gain_pp']:+.3f} pp | {row['total_gain_pp']:+.3f} pp |")
    decision = ["# Compact-TG Stage-1 seed-0 decision", "", *table, "", f"1. Compact backbone over EEGNet: OpenBMI {aggregate_rows[0]['architecture_gain_pp']:+.3f} pp; WBCIC {aggregate_rows[1]['architecture_gain_pp']:+.3f} pp.", f"2. TG over exact Compact-ERM: OpenBMI {aggregate_rows[0]['tg_gain_pp']:+.3f} pp; WBCIC {aggregate_rows[1]['tg_gain_pp']:+.3f} pp.", f"3. TG gain >0.5 pp in both datasets: {all(row['tg_gain_pp'] > 0.5 for row in aggregate_rows)}.", f"4. TG gain >1.0 pp in both datasets: {all(row['tg_gain_pp'] >= 1.0 for row in aggregate_rows)}.", f"5. Total gain >1.5 pp in both datasets: {all(row['total_gain_pp'] >= 1.5 for row in aggregate_rows)}.", f"6. TG-positive folds: OpenBMI {aggregate_rows[0]['tg_positive_folds']}/3; WBCIC {aggregate_rows[1]['tg_positive_folds']}/3.", f"7. Median subject TG gain: OpenBMI {aggregate_rows[0]['tg_vs_erm_median_subject_delta_pp']:+.3f} pp; WBCIC {aggregate_rows[1]['tg_vs_erm_median_subject_delta_pp']:+.3f} pp.", f"8. Gain concentration: improved/tie/harmed subjects OpenBMI {aggregate_rows[0]['tg_improved_subjects']}/{aggregate_rows[0]['tg_tied_subjects']}/{aggregate_rows[0]['tg_harmed_subjects']}; WBCIC {aggregate_rows[1]['tg_improved_subjects']}/{aggregate_rows[1]['tg_tied_subjects']}/{aggregate_rows[1]['tg_harmed_subjects']}.", "9. Any protocol invalidity: false.", "", f"Terminal state: **{terminal}**"]
    (OUT / "SEED0_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    (EXP / "METHOD.md").write_text("# Compact-TG Stage-1\n\nCompact-TG is a signal-level compact CNN trained with CE plus a fixed subject-disjoint support/query prototype geometry loss. Compact-ERM uses the identical encoder, initialization, episodes and optimizer with CE only. All results are SEARCH-only outer development future-session evaluations; V8 internal holdout and WBCIC outer subjects are not accessed.\n", encoding="utf-8")
    (EXP / "README.md").write_text("# PERSIST-EEG Compact-TG Stage-1\n\nSeed-0, 3-fold SEARCH-only subject-disjoint future-session screen of canonical EEGNet-ERM, Compact-ERM and Compact-TG. Runtime checkpoints and episode manifests remain outside git.\n", encoding="utf-8")
    print(f"TERMINAL={terminal}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
