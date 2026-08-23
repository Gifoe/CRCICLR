"""Core implementation for the frozen PERSIST-Net constructive experiment.

The module intentionally has no WBCIC or sealed-holdout loader.  OpenBMI rows
are filtered by parquet predicates before any signal path or label is
materialized.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = EXPERIMENT / "PROTOCOL_FROZEN.json"
RESULTS = EXPERIMENT / "results"
FIGURES = EXPERIMENT / "figures"
RUNTIME = EXPERIMENT / "runtime"
RUNTIME_CACHE = RUNTIME / "cache"
RUNTIME_RUNS = RUNTIME / "runs"
PROTOCOL_DIR = EXPERIMENT / "protocol"

STAGE0_ROOT = Path(
    os.environ.get(
        "PERSIST_STAGE0_REPO",
        r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full",
    )
)
V8_SPLIT = Path(
    os.environ.get(
        "PERSIST_V8_SPLIT",
        r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST\experiments\persist_eeg_final_model_v8\outputs\protocol\V8_SEARCH_SPLIT.json",
    )
)
EXP3_REPORT = Path(
    os.environ.get(
        "PERSIST_EXP3_REPORT",
        r"D:\nips-temp\TotalP\P1\CRCICLR_EXP3_DECISION_GROUNDING_CLOSURE_V1\experiments\persist_eeg_exp3_decision_grounding_closure_v1\EXP3_FINAL_REPORT.json",
    )
)

EPS = 1e-10


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, RUNTIME, RUNTIME_CACHE, RUNTIME_RUNS, PROTOCOL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


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
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def deterministic_reinitialize(model: nn.Module, seed: int) -> None:
    """Reset every parameterized submodule after applying the stable seed."""
    set_determinism(seed)
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def validate_deterministic_initialization() -> dict[str, Any]:
    """Unit-check that the training seed controls constructor-time weights."""
    config = protocol()["baseline_candidates"][0]
    first = EEGNetClassifier(config)
    second = EEGNetClassifier(config)
    third = EEGNetClassifier(config)
    deterministic_reinitialize(first, stable_seed("initialization-unit", 0))
    deterministic_reinitialize(second, stable_seed("initialization-unit", 0))
    deterministic_reinitialize(third, stable_seed("initialization-unit", 1))

    first_state = first.state_dict()
    second_state = second.state_dict()
    third_state = third.state_dict()
    same_seed_exact = all(torch.equal(first_state[key], second_state[key]) for key in first_state)
    different_seed_changes_parameters = any(
        not torch.equal(first_state[name], third_state[name])
        for name, _ in first.named_parameters()
    )
    payload = {
        "same_seed_state_dict_exact": same_seed_exact,
        "different_seed_changes_parameters": different_seed_changes_parameters,
        "validated": bool(same_seed_exact and different_seed_changes_parameters),
    }
    if not payload["validated"]:
        raise RuntimeError(f"Deterministic initialization validation failed: {payload}")
    return payload


def subject_sort(values: Iterable[str]) -> list[str]:
    return sorted(map(str, values), key=lambda x: int(x) if x.isdigit() else x)


def centered_logits(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=-1, keepdim=True)


def centered_logits_np(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array - array.mean(axis=-1, keepdims=True)


def exact_centered_logit_sq(delta: np.ndarray) -> np.ndarray:
    centered = centered_logits_np(delta)
    return np.sum(centered * centered, axis=-1)


def exact_d_finite(clean_logits: np.ndarray, erased_logits: np.ndarray) -> float:
    delta = np.asarray(erased_logits, dtype=np.float64) - np.asarray(clean_logits, dtype=np.float64)
    return float(np.sqrt(np.mean(exact_centered_logit_sq(delta))))


def validate_exact_d() -> dict[str, Any]:
    rng = np.random.default_rng(20260824)
    margin_delta = rng.normal(size=2048)
    delta_logits = np.stack([-0.5 * margin_delta, 0.5 * margin_delta], axis=-1)
    numeric = float(np.sqrt(np.mean(exact_centered_logit_sq(delta_logits))))
    analytic = float(np.sqrt(np.mean(margin_delta**2) / 2.0))
    frozen_reference = {
        "decision_protected_mean": 0.9982230109222217,
        "decision_control_mean": 0.2467850870938559,
        "M0_RMSE": 0.04597839942134,
        "MI_RMSE": 0.0457441624640147,
        "MD_RMSE": 0.0314928431971294,
        "MID_RMSE": 0.0315332767866679,
        "M0_minus_MD": 0.014791624471360496,
        "MD_beats_MI_positive_runs": 6,
    }
    report_candidates = (
        EXP3_REPORT,
        REPO / "experiments" / "persist_eeg_exp3_decision_grounding_closure_v1" / "EXP3_FINAL_REPORT.json",
        REPO / "experiments" / "persist_eeg_exp3_decision_grounding_closure_v1" / "results" / "EXP3_FINAL_REPORT.json",
    )
    report_path = next((path for path in report_candidates if path.is_file()), None)
    if report_path is None:
        raise FileNotFoundError(f"Frozen Exp3 report missing; checked {report_candidates}")
    archived = json.loads(report_path.read_text(encoding="utf-8"))
    observed = {
        "decision_protected_mean": float(archived["decision_protected_mean"]),
        "decision_control_mean": float(archived["decision_control_mean"]),
        "M0_RMSE": float(archived["model_rmse"]["M0"]),
        "MI_RMSE": float(archived["model_rmse"]["MI"]),
        "MD_RMSE": float(archived["model_rmse"]["MD"]),
        "MID_RMSE": float(archived["model_rmse"]["MID"]),
        "M0_minus_MD": float(archived["tests"]["A"]["mean"]),
        "MD_beats_MI_positive_runs": int(archived["tests"]["B"]["positive_runs"]),
    }
    archive_matches = all(
        observed[key] == expected
        if isinstance(expected, int)
        else abs(observed[key] - expected) <= 1e-12
        for key, expected in frozen_reference.items()
    )
    payload = {
        "definition": "sqrt(mean(sum((delta_logits-mean_class(delta_logits))^2,class)))",
        "binary_margin_equivalent": "sqrt(mean(delta_margin^2)/2)",
        "numeric": numeric,
        "analytic": analytic,
        "absolute_error": abs(numeric - analytic),
        "archived_report": str(report_path),
        "archived_values": observed,
        "archive_matches_frozen_reference": archive_matches,
        "validated": bool(abs(numeric - analytic) <= 1e-12 and archive_matches),
    }
    if not payload["validated"]:
        raise RuntimeError(f"Exact D_finite validation failed: {payload}")
    return payload


@dataclass(frozen=True)
class DataPaths:
    x: Path
    metadata: Path
    audit: Path


@dataclass
class DevelopmentData:
    x: np.ndarray
    metadata: pd.DataFrame
    search_subjects: tuple[str, ...]
    holdout_count: int


def authorized_cache_paths() -> DataPaths:
    return DataPaths(
        x=RUNTIME_CACHE / "OPENBMI_V8_SEARCH_MI_RAW.npy",
        metadata=RUNTIME_CACHE / "OPENBMI_V8_SEARCH_MI_METADATA.parquet",
        audit=PROTOCOL_DIR / "HOLDOUT_RUNTIME_AUDIT.json",
    )


def _split_payload() -> dict[str, Any]:
    if not V8_SPLIT.is_file():
        raise FileNotFoundError(V8_SPLIT)
    payload = json.loads(V8_SPLIT.read_text(encoding="utf-8"))
    openbmi = payload["openbmi"]
    search = tuple(map(str, openbmi["V8_SEARCH"]))
    holdout = tuple(map(str, openbmi["V8_INTERNAL_HOLDOUT"]))
    if len(search) != 40 or len(holdout) != 14 or set(search) & set(holdout):
        raise RuntimeError("Malformed V8 OpenBMI split")
    return payload


def build_authorized_cache(force: bool = False) -> DataPaths:
    """Materialize only the predicate-filtered 40-subject MI rows."""
    ensure_dirs()
    paths = authorized_cache_paths()
    payload = _split_payload()
    search = tuple(map(str, payload["openbmi"]["V8_SEARCH"]))
    holdout = tuple(map(str, payload["openbmi"]["V8_INTERNAL_HOLDOUT"]))
    if paths.x.is_file() and paths.metadata.is_file() and paths.audit.is_file() and not force:
        audit = json.loads(paths.audit.read_text(encoding="utf-8"))
        if audit.get("all_checks_passed") is True:
            return paths

    manifest_path = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
    columns = [
        "subject_id",
        "session_id",
        "paradigm",
        "trial_id",
        "event_label",
        "sampling_rate",
        "n_channels",
        "n_times",
        "signal_cache_path",
        "cache_index",
    ]
    # Predicate pushdown is the purity boundary: holdout rows and their label
    # columns are never returned to pandas.
    frame = pd.read_parquet(
        manifest_path,
        columns=columns,
        filters=[("paradigm", "==", "mi"), ("subject_id", "in", list(search))],
        engine="pyarrow",
    )
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["session_id"] = frame.session_id.astype(int)
    frame = frame.sort_values(
        ["subject_id", "session_id", "event_label", "signal_cache_path", "cache_index"],
        key=lambda s: s.map(lambda v: int(v) if str(v).isdigit() else v) if s.name == "subject_id" else s,
    ).reset_index(drop=True)
    mapping = {"left_hand": 0, "right_hand": 1}
    frame["label"] = frame.event_label.astype(str).map(mapping)
    per_cell = frame.groupby(["subject_id", "session_id", "label"]).size()
    checks = {
        "manifest_exists": manifest_path.is_file(),
        "split_read_for_exclusion_only": True,
        "predicate_pushdown_used_before_materialization": True,
        "rows": int(len(frame)),
        "subjects": int(frame.subject_id.nunique()),
        "sessions": sorted(frame.session_id.unique().tolist()),
        "labels": sorted(frame.label.dropna().unique().astype(int).tolist()),
        "per_cell_counts": sorted(set(map(int, per_cell.tolist()))),
        "materialized_subjects_equal_search": set(frame.subject_id) == set(search),
        "materialized_subjects_intersect_holdout": bool(set(frame.subject_id) & set(holdout)),
        "holdout_eeg_materialized": False,
        "holdout_labels_materialized": False,
        "holdout_count": len(holdout),
        "outer_test_used": False,
    }
    valid = bool(
        len(frame) == 8000
        and frame.subject_id.nunique() == 40
        and set(frame.session_id) == {1, 2}
        and set(frame.label.dropna().astype(int)) == {0, 1}
        and set(per_cell.tolist()) == {50}
        and set(frame.subject_id) == set(search)
        and not (set(frame.subject_id) & set(holdout))
    )
    if not valid:
        raise RuntimeError(f"Authorized OpenBMI coverage failure: {checks}")
    frame["label"] = frame.label.astype(int)

    first = frame.iloc[0]
    first_array = np.load(STAGE0_ROOT / str(first.signal_cache_path), mmap_mode="r", allow_pickle=False)
    sample_shape = tuple(first_array[int(first.cache_index)].shape)
    if sample_shape != (62, 1000):
        raise RuntimeError(f"Unexpected OpenBMI MI shape {sample_shape}")
    temporary = paths.x.with_suffix(".npy.part")
    if temporary.exists():
        temporary.unlink()
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(frame),) + sample_shape)
    for relative, group in frame.groupby("signal_cache_path", sort=False):
        source = np.load(STAGE0_ROOT / str(relative), mmap_mode="r", allow_pickle=False)
        positions = group.index.to_numpy(dtype=np.int64)
        indices = group.cache_index.to_numpy(dtype=np.int64)
        target[positions] = np.asarray(source[indices], dtype=np.float32)
    target.flush()
    del target
    os.replace(temporary, paths.x)
    frame.to_parquet(paths.metadata, index=False, engine="pyarrow")
    checks.update(
        {
            "signal_shape": [len(frame), *sample_shape],
            "signal_dtype": "float32",
            "finite_sample_check": bool(np.isfinite(np.load(paths.x, mmap_mode="r")[::97]).all()),
            "split_sha256": sha256_file(V8_SPLIT),
            "manifest_sha256": sha256_file(manifest_path),
            "cache_sha256": sha256_file(paths.x),
            "metadata_sha256": sha256_file(paths.metadata),
            "all_checks_passed": True,
        }
    )
    write_json(paths.audit, checks)
    return paths


def load_development_data() -> DevelopmentData:
    paths = build_authorized_cache(force=False)
    split = _split_payload()["openbmi"]
    metadata = pd.read_parquet(paths.metadata, engine="pyarrow")
    x = np.load(paths.x, mmap_mode="r", allow_pickle=False)
    search = tuple(map(str, split["V8_SEARCH"]))
    if len(x) != 8000 or set(metadata.subject_id.astype(str)) != set(search):
        raise RuntimeError("Runtime cache no longer matches V8_SEARCH")
    return DevelopmentData(
        x=x,
        metadata=metadata,
        search_subjects=search,
        holdout_count=int(split["internal_holdout_subjects"]),
    )


def outer_folds(subjects: Sequence[str]) -> list[dict[str, tuple[str, ...]]]:
    cfg = protocol()
    values = np.asarray(subject_sort(subjects), dtype=object)
    rng = np.random.default_rng(int(cfg["protocol_seed"]))
    values = values[rng.permutation(len(values))]
    chunks = np.array_split(values, 5)
    rows: list[dict[str, tuple[str, ...]]] = []
    for fold, chunk in enumerate(chunks):
        outcome = tuple(subject_sort(chunk.tolist()))
        source = tuple(subject_sort(set(map(str, values)) - set(outcome)))
        inner_rng = np.random.default_rng(stable_seed(cfg["protocol_seed"], "inner", fold))
        shuffled = np.asarray(source, dtype=object)[inner_rng.permutation(len(source))]
        inner_val = tuple(subject_sort(shuffled[:8].tolist()))
        inner_train = tuple(subject_sort(shuffled[8:].tolist()))
        if len(outcome) != 8 or len(source) != 32 or len(inner_val) != 8 or len(inner_train) != 24:
            raise RuntimeError("Fold cardinality failure")
        if set(outcome) & set(source) or set(inner_val) & set(inner_train):
            raise RuntimeError("Fold subject overlap")
        rows.append(
            {
                "outcome": outcome,
                "source": source,
                "inner_validation": inner_val,
                "inner_train": inner_train,
            }
        )
    if set().union(*(set(row["outcome"]) for row in rows)) != set(map(str, subjects)):
        raise RuntimeError("Outer folds do not cover development subjects")
    return rows


def row_indices(
    metadata: pd.DataFrame,
    subjects: Sequence[str],
    sessions: Sequence[int] = (1, 2),
) -> np.ndarray:
    # Arrow-backed pandas booleans may expose a read-only NumPy view.  Copy
    # before the in-place conjunction; this changes no row-selection rule.
    mask = metadata.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy(copy=True)
    mask &= metadata.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def compute_normalizer(data: DevelopmentData, subjects: Sequence[str], cache_key: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = RUNTIME_CACHE / f"NORMALIZER_{cache_key}.npz"
    if path.is_file():
        values = np.load(path, allow_pickle=False)
        return values["mean"].astype(np.float32), values["std"].astype(np.float32), path
    indices = row_indices(data.metadata, subjects, (1, 2))
    total = np.zeros(data.x.shape[1], dtype=np.float64)
    square = np.zeros(data.x.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 16):
        batch = np.asarray(data.x[indices[start : start + 16]], dtype=np.float64)
        total += batch.sum(axis=(0, 2))
        square += np.square(batch).sum(axis=(0, 2))
        count += batch.shape[0] * batch.shape[2]
    mean = total / count
    variance = np.maximum(square / count - mean * mean, 1e-12)
    std = np.sqrt(variance)
    np.savez_compressed(
        path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        subjects=np.asarray(subject_sort(subjects)),
        sessions=np.asarray([1, 2]),
    )
    return mean.astype(np.float32), std.astype(np.float32), path


class IndexDataset(Dataset):
    def __init__(self, x: np.ndarray, metadata: pd.DataFrame, indices: Sequence[int]):
        self.x = x
        self.metadata = metadata
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[int(item)])
        row = self.metadata.iloc[index]
        signal = np.array(self.x[index], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(signal),
            torch.tensor(int(row.label), dtype=torch.long),
            torch.tensor(index, dtype=torch.long),
            str(row.subject_id),
        )


def make_loader(
    data: DevelopmentData,
    indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        IndexDataset(data.x, data.metadata, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _same_pad(x: torch.Tensor, kernel: int) -> torch.Tensor:
    total = kernel - 1
    left = total // 2
    return F.pad(x, (left, total - left, 0, 0))


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 62,
        samples: int = 1000,
        f1: int = 8,
        f2: int = 16,
        embedding_dim: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.channels = int(channels)
        self.samples = int(samples)
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, 2 * f1, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * f1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(2 * f1, 2 * f1, (1, 16), padding=0, groups=2 * f1, bias=False)
        self.point = nn.Conv2d(2 * f1, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        with torch.no_grad():
            probe = torch.zeros(1, 1, channels, samples)
            probe = self.bn1(self.temporal(_same_pad(probe, 64)))
            probe = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(probe)))))
            probe = self.drop2(
                self.pool2(F.elu(self.bn3(self.point(self.depth(_same_pad(probe, 16))))))
            )
            flattened = int(probe.flatten(1).shape[1])
        self.embedding = nn.Sequential(
            nn.Linear(flattened, embedding_dim),
            nn.ELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.embedding_dim = int(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(_same_pad(value, 64)))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(
            self.pool2(F.elu(self.bn3(self.point(self.depth(_same_pad(value, 16))))))
        )
        return self.embedding(value.flatten(1))


class EEGNetClassifier(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.encoder = EEGNetEncoder(
            f1=int(config["f1"]),
            f2=int(config["f2"]),
            embedding_dim=int(config.get("embedding_dim", 64)),
            dropout=float(config["dropout"]),
        )
        self.head = nn.Linear(self.encoder.embedding_dim, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class DualPathEEGNet(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        kwargs = {
            "f1": int(config["f1"]),
            "f2": int(config["f2"]),
            "embedding_dim": int(config["embedding_dim"]),
            "dropout": float(config["dropout"]),
        }
        self.protected = EEGNetEncoder(**kwargs)
        self.adaptive = EEGNetEncoder(**kwargs)
        self.protected_head = nn.Linear(self.protected.embedding_dim, 2)
        self.adaptive_head = nn.Linear(self.adaptive.embedding_dim, 2)

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zp = self.protected(x)
        za = self.adaptive(x)
        lp = self.protected_head(zp)
        la = self.adaptive_head(za)
        return lp, la, zp, za

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lp, la, _, _ = self.forward_parts(x)
        return lp + la


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def normalize_tensor(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-8)


def _amp_context(device: torch.device):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    )


def _scaler(device: torch.device):
    return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")


@dataclass
class Evaluation:
    logits: np.ndarray
    features: np.ndarray | None
    labels: np.ndarray
    subjects: np.ndarray
    indices: np.ndarray


def evaluate_single(
    model: EEGNetClassifier,
    data: DevelopmentData,
    indices: Sequence[int],
    device: torch.device,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    include_features: bool = True,
    batch_size: int = 512,
) -> Evaluation:
    loader = make_loader(data, indices, batch_size, False, 0)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    model.eval()
    logits: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[str] = []
    positions: list[np.ndarray] = []
    with torch.inference_mode():
        for x, y, idx, subject in loader:
            x = normalize_tensor(x.to(device, non_blocking=True), mean, std)
            with _amp_context(device):
                h = model.forward_features(x)
                value = model.head(h)
            logits.append(value.float().cpu().numpy())
            if include_features:
                features.append(h.float().cpu().numpy())
            labels.append(y.numpy())
            subjects.extend(map(str, subject))
            positions.append(idx.numpy())
    return Evaluation(
        logits=np.concatenate(logits),
        features=np.concatenate(features) if include_features else None,
        labels=np.concatenate(labels),
        subjects=np.asarray(subjects),
        indices=np.concatenate(positions),
    )


def evaluate_dual(
    model: DualPathEEGNet,
    data: DevelopmentData,
    indices: Sequence[int],
    device: torch.device,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    loader = make_loader(data, indices, batch_size, False, 0)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    model.eval()
    fields: dict[str, list[Any]] = {
        "protected_logits": [],
        "adaptive_logits": [],
        "protected_features": [],
        "adaptive_features": [],
        "labels": [],
        "indices": [],
        "subjects": [],
    }
    with torch.inference_mode():
        for x, y, idx, subject in loader:
            x = normalize_tensor(x.to(device, non_blocking=True), mean, std)
            with _amp_context(device):
                lp, la, zp, za = model.forward_parts(x)
            fields["protected_logits"].append(lp.float().cpu().numpy())
            fields["adaptive_logits"].append(la.float().cpu().numpy())
            fields["protected_features"].append(zp.float().cpu().numpy())
            fields["adaptive_features"].append(za.float().cpu().numpy())
            fields["labels"].append(y.numpy())
            fields["indices"].append(idx.numpy())
            fields["subjects"].extend(map(str, subject))
    return {
        key: np.asarray(value) if key == "subjects" else np.concatenate(value)
        for key, value in fields.items()
    }


def subject_mean_ba(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> float:
    pred = np.asarray(logits).argmax(axis=1)
    values = [
        balanced_accuracy_score(labels[subjects == subject], pred[subjects == subject])
        for subject in subject_sort(set(map(str, subjects)))
    ]
    return float(np.mean(values))


def per_subject_metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    pred = np.asarray(logits).argmax(axis=1)
    rows = []
    for subject in subject_sort(set(map(str, subjects))):
        mask = np.asarray(subjects, dtype=str) == subject
        rows.append(
            {
                "subject_id": subject,
                "BA": float(balanced_accuracy_score(labels[mask], pred[mask])),
                "macro_f1": float(f1_score(labels[mask], pred[mask], average="macro")),
                "n_trials": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def train_single(
    model: EEGNetClassifier,
    data: DevelopmentData,
    train_indices: Sequence[int],
    validation_indices: Sequence[int] | None,
    device: torch.device,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    seed: int,
    config: Mapping[str, Any],
    fixed_epochs: int | None = None,
) -> tuple[EEGNetClassifier, int, list[dict[str, Any]]]:
    deterministic_reinitialize(model, seed)
    model = model.to(device)
    batch_size = int(protocol()["baseline_training"]["batch_size"])
    loader = make_loader(data, train_indices, batch_size, True, seed)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = _scaler(device)
    rule = protocol()["baseline_training"]
    maximum = int(fixed_epochs if fixed_epochs is not None else rule["max_epochs"])
    minimum = int(rule["minimum_epochs"])
    patience = int(rule["patience"])
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float, int] | None = None
    best_epoch = maximum
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(maximum):
        model.train()
        total = 0.0
        seen = 0
        for x, y, _, _ in loader:
            x = normalize_tensor(x.to(device, non_blocking=True), mean, std)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * len(y)
            seen += len(y)
        row: dict[str, Any] = {"epoch": epoch + 1, "train_loss": total / max(seen, 1)}
        if validation_indices is not None:
            evaluation = evaluate_single(model, data, validation_indices, device, mean_np, std_np, include_features=False)
            score = subject_mean_ba(evaluation.labels, evaluation.logits, evaluation.subjects)
            row["validation_mean_subject_BA"] = score
            key = (score, -row["train_loss"], -(epoch + 1))
            if best_key is None or key > best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
        history.append(row)
        if validation_indices is not None and epoch + 1 >= minimum and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def numpy_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    shifted = value - value.max(axis=1, keepdims=True)
    logsum = np.log(np.exp(shifted).sum(axis=1))
    return logsum - shifted[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)]


def subject_bootstrap_ci(values: Mapping[str, float], seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    subjects = subject_sort(values)
    array = np.asarray([values[s] for s in subjects], dtype=np.float64)
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[sampled].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@dataclass
class Certificate:
    mean: np.ndarray
    whitener: np.ndarray
    dewhitener: np.ndarray
    directions: np.ndarray
    rho: np.ndarray
    rows: pd.DataFrame
    bases: dict[str, np.ndarray]
    audit: dict[str, Any]

    @property
    def rank(self) -> int:
        return int(self.bases["PUD"].shape[1])


def _corr_or_zero(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def _orthonormal(matrix: np.ndarray, rank: int | None = None) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.size == 0:
        return np.zeros((value.shape[0], 0), dtype=np.float32)
    q, _ = np.linalg.qr(value)
    if rank is not None:
        q = q[:, :rank]
    return q.astype(np.float32)


def _delta_logits_for_basis(
    q: np.ndarray,
    basis: np.ndarray,
    directions: np.ndarray,
    dewhitener: np.ndarray,
    head_weight: np.ndarray,
) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((len(q), head_weight.shape[0]), dtype=np.float64)
    delta_q = (q @ basis) @ basis.T
    delta_h = (delta_q @ directions.T) @ dewhitener
    return delta_h @ np.asarray(head_weight, dtype=np.float64).T


def fit_certificate(
    teacher: EEGNetClassifier,
    data: DevelopmentData,
    source_subjects: Sequence[str],
    device: torch.device,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    fold: int,
    seed: int,
) -> tuple[Certificate, Evaluation]:
    cfg = protocol()["certification"]
    indices = row_indices(data.metadata, source_subjects, (1, 2))
    evaluation = evaluate_single(teacher, data, indices, device, mean_np, std_np, include_features=True)
    if evaluation.features is None:
        raise RuntimeError("Teacher features missing")
    h = np.asarray(evaluation.features, dtype=np.float64)
    logits = np.asarray(evaluation.logits, dtype=np.float64)
    y = np.asarray(evaluation.labels, dtype=np.int64)
    meta = data.metadata.iloc[evaluation.indices].reset_index(drop=True).copy()
    subjects = subject_sort(source_subjects)

    mu = h.mean(axis=0)
    hc = h - mu
    cov = hc.T @ hc / max(len(hc) - 1, 1)
    ev, u = np.linalg.eigh((cov + cov.T) / 2.0)
    order = np.argsort(ev)[::-1]
    ev, u = ev[order], u[:, order]
    threshold = max(
        float(ev[0]) * float(cfg["embedding_eigenvalue_relative_threshold"]),
        float(cfg["embedding_eigenvalue_absolute_floor"]),
    )
    numerical_rank = int(np.sum(ev > threshold))
    rank = min(int(cfg["whitening_rank_max"]), numerical_rank)
    if rank < 4:
        raise RuntimeError(f"Insufficient teacher embedding rank: {rank}")
    active = np.maximum(ev[:rank], max(float(ev[:rank].mean()) * 1e-4, 1e-8))
    up = u[:, :rank]
    whitener = up * np.power(active, -0.5)[None, :]
    dewhitener = np.sqrt(active)[:, None] * up.T
    z = hc @ whitener

    centroids: dict[tuple[str, int, int], np.ndarray] = {}
    meta["position"] = np.arange(len(meta))
    for key, group in meta.groupby(["subject_id", "session_id", "label"], sort=True):
        centroids[(str(key[0]), int(key[1]), int(key[2]))] = z[group.position.to_numpy(dtype=np.int64)].mean(axis=0)
    cross_covariances = []
    pair_counts = 0
    for label in (0, 1):
        left = np.stack([centroids[(s, 1, label)] for s in subjects])
        right = np.stack([centroids[(s, 2, label)] for s in subjects])
        left = left - left.mean(axis=0, keepdims=True)
        right = right - right.mean(axis=0, keepdims=True)
        cross_covariances.append((left.T @ right + right.T @ left) / (2.0 * max(len(subjects) - 1, 1)))
        pair_counts += len(subjects)
    cross = np.mean(cross_covariances, axis=0)
    rho, directions = np.linalg.eigh((cross + cross.T) / 2.0)
    order = np.argsort(rho)[::-1]
    rho, directions = rho[order], directions[:, order]
    q = z @ directions

    left_all = []
    right_all = []
    for label in (0, 1):
        left_all.extend(centroids[(s, 1, label)] for s in subjects)
        right_all.extend(centroids[(s, 2, label)] for s in subjects)
    left_all = np.stack(left_all)
    right_all = np.stack(right_all)
    persistence_corr = np.asarray(
        [_corr_or_zero(left_all @ directions[:, j], right_all @ directions[:, j]) for j in range(rank)]
    )

    null = np.zeros((int(cfg["permutation_draws"]), rank), dtype=np.float64)
    rng = np.random.default_rng(stable_seed("P-null", fold, seed))
    for draw in range(len(null)):
        perm = rng.permutation(len(subjects))
        covs = []
        for label in (0, 1):
            left = np.stack([centroids[(s, 1, label)] for s in subjects])
            right = np.stack([centroids[(subjects[perm[i]], 2, label)] for i, s in enumerate(subjects)])
            left -= left.mean(axis=0, keepdims=True)
            right -= right.mean(axis=0, keepdims=True)
            covs.append((left.T @ right + right.T @ left) / (2.0 * max(len(subjects) - 1, 1)))
        cn = np.mean(covs, axis=0)
        null[draw] = np.diag(directions.T @ cn @ directions)
    null_p95 = np.quantile(null, float(cfg["persistence_permutation_quantile"]), axis=0)

    head_weight = teacher.head.weight.detach().cpu().numpy().astype(np.float64)
    full_ce = numpy_cross_entropy(logits, y)
    random_count = int(cfg["utility_random_directions"])
    random_rng = np.random.default_rng(stable_seed("U-random", fold, seed))
    random_unit = random_rng.normal(size=(random_count, rank))
    random_unit /= np.maximum(np.linalg.norm(random_unit, axis=1, keepdims=True), EPS)
    h_metric = directions.T @ dewhitener @ dewhitener.T @ directions

    rows: list[dict[str, Any]] = []
    pass_p: list[int] = []
    pass_pu: list[int] = []
    pass_pd: list[int] = []
    pass_pud: list[int] = []
    meta_subject = meta.subject_id.astype(str).to_numpy()
    for j in range(rank):
        target_basis = np.eye(rank, dtype=np.float64)[:, [j]]
        target_delta = _delta_logits_for_basis(q, target_basis, directions, dewhitener, head_weight)
        erased = logits - target_delta
        target_ce = numpy_cross_entropy(erased, y)
        target_by_subject = {
            subject: float(np.mean(target_ce[meta_subject == subject] - full_ce[meta_subject == subject]))
            for subject in subjects
        }
        random_utilities: list[dict[str, float]] = []
        random_ds: list[float] = []
        target_energy_norm = math.sqrt(max(float(h_metric[j, j]), EPS))
        target_amplitude = np.abs(q[:, j]) * target_energy_norm
        for random_direction in random_unit:
            random_norm = math.sqrt(max(float(random_direction @ h_metric @ random_direction), EPS))
            coefficient = np.sign(q @ random_direction) * target_amplitude / random_norm
            random_delta_q = coefficient[:, None] * random_direction[None, :]
            random_delta_h = (random_delta_q @ directions.T) @ dewhitener
            random_delta_logits = random_delta_h @ head_weight.T
            random_erased = logits - random_delta_logits
            random_ce = numpy_cross_entropy(random_erased, y)
            random_utilities.append(
                {
                    subject: float(np.mean(random_ce[meta_subject == subject] - full_ce[meta_subject == subject]))
                    for subject in subjects
                }
            )
            random_ds.append(exact_d_finite(logits, random_erased))
        random_by_subject = {
            subject: float(np.mean([entry[subject] for entry in random_utilities]))
            for subject in subjects
        }
        utility_difference = {
            subject: target_by_subject[subject] - random_by_subject[subject]
            for subject in subjects
        }
        u_mean, u_low, u_high = subject_bootstrap_ci(
            utility_difference,
            stable_seed("U-bootstrap", fold, seed, j),
            int(cfg["utility_bootstrap_draws"]),
        )
        target_d = exact_d_finite(logits, erased)
        random_d_mean = float(np.mean(random_ds))
        d_ratio = target_d / max(random_d_mean, EPS)
        p_ok = bool(rho[j] > null_p95[j] and persistence_corr[j] >= float(cfg["persistence_threshold"]))
        u_ok = bool(float(np.mean(list(target_by_subject.values()))) > 0.0 and u_low > 0.0)
        d_ok = bool(d_ratio > float(cfg["decision_ratio_threshold"]))
        if p_ok:
            pass_p.append(j)
        if p_ok and u_ok:
            pass_pu.append(j)
        if p_ok and d_ok:
            pass_pd.append(j)
        if p_ok and u_ok and d_ok:
            pass_pud.append(j)
        rows.append(
            {
                "fold": fold,
                "seed": seed,
                "direction": j,
                "rho": float(rho[j]),
                "persistence_correlation": float(persistence_corr[j]),
                "permutation_p95": float(null_p95[j]),
                "P_pass": p_ok,
                "utility_harm_mean": float(np.mean(list(target_by_subject.values()))),
                "matched_random_utility_mean": float(np.mean(list(random_by_subject.values()))),
                "utility_specific_mean": u_mean,
                "utility_specific_CI95_L": u_low,
                "utility_specific_CI95_U": u_high,
                "U_pass": u_ok,
                "D_finite": target_d,
                "D_finite_matched_random_mean": random_d_mean,
                "D_finite_ratio": d_ratio,
                "D_pass": d_ok,
                "PUD_pass": bool(p_ok and u_ok and d_ok),
                "D_flip_used": False,
                "identity_used": False,
            }
        )

    def coordinate_basis(selected: Sequence[int]) -> np.ndarray:
        if not selected:
            return np.zeros((rank, 0), dtype=np.float32)
        return np.eye(rank, dtype=np.float32)[:, np.asarray(selected, dtype=np.int64)]

    protected_rank = len(pass_pud)
    # Identity control: source-only between-subject covariance in the exact q space.
    subject_means = np.stack([q[meta_subject == subject].mean(axis=0) for subject in subjects])
    subject_means -= subject_means.mean(axis=0, keepdims=True)
    identity_cov = subject_means.T @ subject_means / max(len(subjects) - 1, 1)
    _, identity_vectors = np.linalg.eigh((identity_cov + identity_cov.T) / 2.0)
    identity_vectors = identity_vectors[:, ::-1]
    identity_basis = _orthonormal(identity_vectors[:, :protected_rank])

    random_control_rng = np.random.default_rng(stable_seed("matched-random-basis", fold, seed))
    random_basis = _orthonormal(random_control_rng.normal(size=(rank, max(protected_rank, 1))))[:, :protected_rank]
    # q = z @ directions.  Rows of directions are original whitened-PCA axes in q coordinates.
    pca_basis = _orthonormal(directions[: max(protected_rank, 1), :].T)[:, :protected_rank]
    bases = {
        "P": coordinate_basis(pass_p),
        "PU": coordinate_basis(pass_pu),
        "PD": coordinate_basis(pass_pd),
        "PUD": coordinate_basis(pass_pud),
        "IDENTITY": identity_basis,
        "RANDOM": random_basis,
        "PCA": pca_basis,
    }
    nonpersistent = [j for j in range(rank) if j not in pass_p]
    nonpersistent_basis = coordinate_basis(nonpersistent[:protected_rank])

    def control_metrics(basis: np.ndarray) -> dict[str, Any]:
        delta_logits = _delta_logits_for_basis(q, basis, directions, dewhitener, head_weight)
        erased_logits = logits - delta_logits
        prediction = logits.argmax(axis=1)
        erased_prediction = erased_logits.argmax(axis=1)
        return {
            "rank": int(basis.shape[1]),
            "CE_harm": float(np.mean(numpy_cross_entropy(erased_logits, y) - full_ce)),
            "BA_harm": float(
                balanced_accuracy_score(y, prediction)
                - balanced_accuracy_score(y, erased_prediction)
            ),
            "D_finite": exact_d_finite(logits, erased_logits),
        }

    matched_control_summary = {
        "PUD_union": control_metrics(bases["PUD"]),
        "matched_rank_random": control_metrics(bases["RANDOM"]),
        "matched_rank_nonpersistent": control_metrics(nonpersistent_basis),
        "matched_rank_PCA": control_metrics(bases["PCA"]),
        "per_direction_matched_energy_random_used_for_U_and_D": True,
    }
    overlap = float(
        np.square(np.linalg.svd(bases["PUD"].T @ pca_basis, compute_uv=False)).sum() / max(protected_rank, 1)
    ) if protected_rank else 0.0
    audit = {
        "fold": fold,
        "seed": seed,
        "source_subjects": subjects,
        "source_rows": int(len(h)),
        "embedding_dimension": int(h.shape[1]),
        "numerical_rank": numerical_rank,
        "whitening_rank": rank,
        "whitening_error_max_abs": float(np.max(np.abs(z.T @ z / max(len(z) - 1, 1) - np.eye(rank)))),
        "class_conditioned_session_pairs": pair_counts,
        "candidate_directions": rank,
        "P_rank": len(pass_p),
        "PU_rank": len(pass_pu),
        "PD_rank": len(pass_pd),
        "PUD_rank": protected_rank,
        "identity_rank": int(identity_basis.shape[1]),
        "random_rank": int(random_basis.shape[1]),
        "pca_rank": int(pca_basis.shape[1]),
        "PUD_PCA_overlap_fraction": overlap,
        "matched_control_summary": matched_control_summary,
        "rank_cap": None,
        "D_flip_used": False,
        "identity_used_in_primary_definition": False,
        "finite": bool(np.isfinite(q).all() and np.isfinite(logits).all()),
    }
    certificate = Certificate(
        mean=mu.astype(np.float32),
        whitener=whitener.astype(np.float32),
        dewhitener=dewhitener.astype(np.float32),
        directions=directions.astype(np.float32),
        rho=rho.astype(np.float32),
        rows=pd.DataFrame(rows),
        bases=bases,
        audit=audit,
    )
    return certificate, evaluation


def save_certificate(certificate: Certificate, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "PUD_CERTIFICATION.csv", certificate.rows)
    write_json(directory / "PUD_CERTIFICATION_AUDIT.json", certificate.audit)
    np.savez_compressed(
        directory / "PUD_CERTIFICATE.npz",
        mean=certificate.mean,
        whitener=certificate.whitener,
        dewhitener=certificate.dewhitener,
        directions=certificate.directions,
        rho=certificate.rho,
        **{f"basis_{key}": value for key, value in certificate.bases.items()},
    )


def teacher_targets(
    teacher: EEGNetClassifier,
    certificate: Certificate,
    evaluation: Evaluation,
    basis_name: str,
) -> dict[str, np.ndarray | float]:
    if evaluation.features is None:
        raise RuntimeError("Teacher features missing")
    h = np.asarray(evaluation.features, dtype=np.float64)
    full = np.asarray(evaluation.logits, dtype=np.float64)
    z = (h - certificate.mean) @ certificate.whitener
    q = z @ certificate.directions
    head_weight = teacher.head.weight.detach().cpu().numpy().astype(np.float64)
    delta = _delta_logits_for_basis(
        q,
        certificate.bases[basis_name],
        certificate.directions,
        certificate.dewhitener,
        head_weight,
    )
    protected = centered_logits_np(delta)
    centered_full = centered_logits_np(full)
    residual = centered_full - protected
    scale = float(np.sqrt(np.mean(exact_centered_logit_sq(centered_full))))
    scale = max(scale, 1e-4)
    return {
        "indices": evaluation.indices.astype(np.int64),
        "full": full.astype(np.float32),
        "protected": protected.astype(np.float32),
        "residual": residual.astype(np.float32),
        "scale": scale,
    }


class PrototypeSampler:
    def __init__(self, data: DevelopmentData, indices: Sequence[int], seed: int):
        self.data = data
        self.seed = int(seed)
        self.calls = 0
        selected = data.metadata.iloc[np.asarray(indices, dtype=np.int64)].copy()
        self.cells: list[tuple[np.ndarray, np.ndarray]] = []
        for (_, _), group in selected.groupby(["subject_id", "label"], sort=True):
            s1 = group.loc[group.session_id.astype(int).eq(1)].index.to_numpy(dtype=np.int64)
            s2 = group.loc[group.session_id.astype(int).eq(2)].index.to_numpy(dtype=np.int64)
            if len(s1) and len(s2):
                self.cells.append((s1, s2))
        if not self.cells:
            raise RuntimeError("No class-conditioned cross-session prototype cells")

    def sample(self, cell_count: int = 4, trials_per_session: int = 4) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(stable_seed(self.seed, "prototype", self.calls))
        self.calls += 1
        chosen = rng.choice(len(self.cells), size=min(cell_count, len(self.cells)), replace=False)
        left, right = [], []
        for cell in chosen:
            s1, s2 = self.cells[int(cell)]
            i1 = rng.choice(s1, size=min(trials_per_session, len(s1)), replace=False)
            i2 = rng.choice(s2, size=min(trials_per_session, len(s2)), replace=False)
            left.append(np.asarray(self.data.x[i1], dtype=np.float32))
            right.append(np.asarray(self.data.x[i2], dtype=np.float32))
        return np.stack(left), np.stack(right)


def _target_lookup(targets: Mapping[str, Any], total_rows: int) -> tuple[np.ndarray, np.ndarray]:
    protected = np.full((total_rows, 2), np.nan, dtype=np.float32)
    residual = np.full((total_rows, 2), np.nan, dtype=np.float32)
    idx = np.asarray(targets["indices"], dtype=np.int64)
    protected[idx] = np.asarray(targets["protected"], dtype=np.float32)
    residual[idx] = np.asarray(targets["residual"], dtype=np.float32)
    return protected, residual


def train_dual(
    model: DualPathEEGNet,
    data: DevelopmentData,
    train_indices: Sequence[int],
    validation_indices: Sequence[int] | None,
    targets: Mapping[str, Any] | None,
    device: torch.device,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    seed: int,
    fixed_epochs: int | None = None,
    task_only: bool = False,
) -> tuple[DualPathEEGNet, int, list[dict[str, Any]], dict[str, Any]]:
    deterministic_reinitialize(model, seed)
    model = model.to(device)
    cfg = protocol()
    train_cfg = cfg["student_training"]
    loss_cfg = cfg["loss"]
    loader = make_loader(data, train_indices, int(train_cfg["batch_size"]), True, seed)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scaler = _scaler(device)
    maximum = int(fixed_epochs if fixed_epochs is not None else train_cfg["max_epochs"])
    minimum = int(train_cfg["minimum_epochs"])
    patience = int(train_cfg["patience"])
    protected_lookup: np.ndarray | None = None
    residual_lookup: np.ndarray | None = None
    target_scale = 1.0
    prototype_sampler: PrototypeSampler | None = None
    if not task_only:
        if targets is None:
            raise ValueError("Distillation targets required")
        protected_lookup, residual_lookup = _target_lookup(targets, len(data.metadata))
        target_scale = float(targets["scale"])
        prototype_sampler = PrototypeSampler(data, train_indices, seed)

    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float, int] | None = None
    best_epoch = maximum
    stale = 0
    history: list[dict[str, Any]] = []
    first_gradient = {"protected": None, "adaptive": None}
    for epoch in range(maximum):
        model.train()
        totals = {"loss": 0.0, "task": 0.0, "protected": 0.0, "residual": 0.0, "persistence": 0.0}
        seen = 0
        for x, y, idx, _ in loader:
            x = normalize_tensor(x.to(device, non_blocking=True), mean, std)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device):
                lp, la, _, _ = model.forward_parts(x)
                task_loss = F.cross_entropy(lp + la, y)
                protected_loss = torch.zeros((), device=device)
                residual_loss = torch.zeros((), device=device)
                persistence_loss = torch.zeros((), device=device)
                if not task_only:
                    if protected_lookup is None or residual_lookup is None or prototype_sampler is None:
                        raise RuntimeError("Missing distillation state")
                    tp = torch.as_tensor(protected_lookup[idx.numpy()], dtype=torch.float32, device=device)
                    tr = torch.as_tensor(residual_lookup[idx.numpy()], dtype=torch.float32, device=device)
                    if not torch.isfinite(tp).all() or not torch.isfinite(tr).all():
                        raise RuntimeError("Target lookup crossed an unauthorized row")
                    protected_loss = F.mse_loss(centered_logits(lp) / target_scale, tp / target_scale)
                    residual_loss = F.mse_loss(centered_logits(la) / target_scale, tr / target_scale)
                    x1_np, x2_np = prototype_sampler.sample()
                    cells, per = x1_np.shape[:2]
                    x1 = normalize_tensor(
                        torch.as_tensor(x1_np.reshape(-1, *x1_np.shape[2:]), device=device), mean, std
                    )
                    x2 = normalize_tensor(
                        torch.as_tensor(x2_np.reshape(-1, *x2_np.shape[2:]), device=device), mean, std
                    )
                    p1 = model.protected(x1).reshape(cells, per, -1).mean(dim=1)
                    p2 = model.protected(x2).reshape(cells, per, -1).mean(dim=1)
                    persistence_loss = F.mse_loss(p1, p2)
                loss = (
                    float(loss_cfg["task"]) * task_loss
                    + float(loss_cfg["lambda_D"]) * protected_loss
                    + float(loss_cfg["lambda_R"]) * residual_loss
                    + float(loss_cfg["lambda_P"]) * persistence_loss
                )
            scaler.scale(loss).backward()
            if first_gradient["protected"] is None:
                scaler.unscale_(optimizer)
                first_gradient["protected"] = float(
                    math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in model.protected.parameters() if p.grad is not None))
                )
                first_gradient["adaptive"] = float(
                    math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in model.adaptive.parameters() if p.grad is not None))
                )
            scaler.step(optimizer)
            scaler.update()
            count = len(y)
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["task"] += float(task_loss.detach().cpu()) * count
            totals["protected"] += float(protected_loss.detach().cpu()) * count
            totals["residual"] += float(residual_loss.detach().cpu()) * count
            totals["persistence"] += float(persistence_loss.detach().cpu()) * count
            seen += count
        row: dict[str, Any] = {"epoch": epoch + 1, **{key: value / max(seen, 1) for key, value in totals.items()}}
        if validation_indices is not None:
            evaluation = evaluate_dual(model, data, validation_indices, device, mean_np, std_np)
            logits = evaluation["protected_logits"] + evaluation["adaptive_logits"]
            score = subject_mean_ba(evaluation["labels"], logits, evaluation["subjects"])
            row["validation_mean_subject_BA"] = score
            key = (score, -row["loss"], -(epoch + 1))
            if best_key is None or key > best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
        history.append(row)
        if validation_indices is not None and epoch + 1 >= minimum and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    diagnostics = {"first_batch_gradient_norm": first_gradient, "target_scale": target_scale, "task_only": task_only}
    return model, best_epoch, history, diagnostics


def _adapt_parameter_names_single(model: EEGNetClassifier, strategy: str) -> list[str]:
    names = []
    for name, _ in model.named_parameters():
        if strategy == "head" and name.startswith("head"):
            names.append(name)
        elif strategy == "tail" and (name.startswith("encoder.embedding") or name.startswith("head")):
            names.append(name)
        elif strategy == "full":
            names.append(name)
    return names


def adapt_single(
    source: EEGNetClassifier,
    x_np: np.ndarray,
    y_np: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    seed: int,
) -> tuple[EEGNetClassifier, dict[str, Any]]:
    set_determinism(seed)
    model = copy.deepcopy(source).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    names = _adapt_parameter_names_single(model, str(config["strategy"]))
    for name, p in model.named_parameters():
        if name in names:
            p.requires_grad_(True)
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["lr"]),
        weight_decay=1e-4,
    )
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    for _ in range(int(config["epochs"])):
        model.train()
        permutation = torch.randperm(len(y_np), generator=generator)
        for start in range(0, len(y_np), 32):
            index = permutation[start : start + 32].numpy()
            xb = normalize_tensor(torch.as_tensor(x_np[index], dtype=torch.float32, device=device), mean, std)
            yb = torch.as_tensor(y_np[index], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started
    update = math.sqrt(
        sum(float(((p.detach().cpu() - before[name]) ** 2).sum()) for name, p in model.named_parameters())
    )
    return model, {
        "strategy": str(config["strategy"]),
        "trainable_parameter_names": names,
        "trainable_parameters": trainable_parameter_count(model),
        "parameter_update_l2": update,
        "adaptation_time_s": elapsed,
    }


def configure_dual_adaptation(model: DualPathEEGNet, strategy: str, all_adapt: bool) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    names = []
    for name, p in model.named_parameters():
        branch_allowed = name.startswith("adaptive") or (all_adapt and name.startswith("protected"))
        head_allowed = name.startswith("adaptive_head") or (all_adapt and name.startswith("protected_head"))
        if strategy == "head":
            allowed = head_allowed
        elif strategy == "tail":
            allowed = branch_allowed and ("embedding" in name or "head" in name)
        elif strategy == "full":
            allowed = branch_allowed or head_allowed
        else:
            raise ValueError(strategy)
        if allowed:
            p.requires_grad_(True)
            names.append(name)
    return names


def adapt_dual(
    source: DualPathEEGNet,
    x_np: np.ndarray,
    y_np: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    seed: int,
    all_adapt: bool,
) -> tuple[DualPathEEGNet, dict[str, Any]]:
    set_determinism(seed)
    model = copy.deepcopy(source).to(device)
    names = configure_dual_adaptation(model, str(config["strategy"]), all_adapt=all_adapt)
    before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    before_protected_buffers = {
        name: value.detach().cpu().clone()
        for name, value in model.named_buffers()
        if name.startswith("protected")
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["lr"]),
        weight_decay=1e-4,
    )
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    for _ in range(int(config["epochs"])):
        if all_adapt:
            model.train()
        else:
            model.adaptive.train()
            model.adaptive_head.train()
            model.protected.eval()
            model.protected_head.eval()
        permutation = torch.randperm(len(y_np), generator=generator)
        for start in range(0, len(y_np), 32):
            index = permutation[start : start + 32].numpy()
            xb = normalize_tensor(torch.as_tensor(x_np[index], dtype=torch.float32, device=device), mean, std)
            yb = torch.as_tensor(y_np[index], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - started
    updates = {}
    for prefix in ("protected", "adaptive"):
        updates[prefix] = math.sqrt(
            sum(
                float(((p.detach().cpu() - before[name]) ** 2).sum())
                for name, p in model.named_parameters()
                if name.startswith(prefix)
            )
        )
    protected_buffer_drift = math.sqrt(
        sum(
            float(((value.detach().cpu() - before_protected_buffers[name]) ** 2).sum())
            for name, value in model.named_buffers()
            if name.startswith("protected") and name in before_protected_buffers
        )
    )
    return model, {
        "strategy": str(config["strategy"]),
        "all_adapt": bool(all_adapt),
        "trainable_parameter_names": names,
        "trainable_parameters": trainable_parameter_count(model),
        "protected_parameter_update_l2": updates["protected"],
        "adaptive_parameter_update_l2": updates["adaptive"],
        "protected_buffer_update_l2": protected_buffer_drift,
        "adaptation_time_s": elapsed,
    }


def raw_subject(data: DevelopmentData, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = data.metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
    s1 = mask & data.metadata.session_id.astype(int).eq(1).to_numpy()
    s2 = mask & data.metadata.session_id.astype(int).eq(2).to_numpy()
    i1, i2 = np.flatnonzero(s1), np.flatnonzero(s2)
    return (
        np.asarray(data.x[i1], dtype=np.float32),
        data.metadata.iloc[i1].label.to_numpy(dtype=np.int64),
        np.asarray(data.x[i2], dtype=np.float32),
        data.metadata.iloc[i2].label.to_numpy(dtype=np.int64),
        i1,
        i2,
    )


def functional_agreement(student_logits: np.ndarray, teacher_target: np.ndarray) -> tuple[float, float]:
    a = centered_logits_np(student_logits).reshape(-1)
    b = centered_logits_np(teacher_target).reshape(-1)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    corr = _corr_or_zero(a, b)
    return rmse, corr


def branch_mechanism_metrics(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    teacher_protected_target: np.ndarray | None,
) -> dict[str, Any]:
    lp0 = np.asarray(before["protected_logits"])
    lp1 = np.asarray(after["protected_logits"])
    la0 = np.asarray(before["adaptive_logits"])
    la1 = np.asarray(after["adaptive_logits"])
    zp0 = np.asarray(before["protected_features"])
    zp1 = np.asarray(after["protected_features"])
    za0 = np.asarray(before["adaptive_features"])
    za1 = np.asarray(after["adaptive_features"])
    payload: dict[str, Any] = {
        "protected_representation_drift": float(np.sqrt(np.mean((zp1 - zp0) ** 2))),
        "adaptive_representation_drift": float(np.sqrt(np.mean((za1 - za0) ** 2))),
        "protected_decision_logit_drift": exact_d_finite(lp0, lp1),
        "adaptive_decision_logit_drift": exact_d_finite(la0, la1),
        "protected_D_finite_after": exact_d_finite(np.zeros_like(lp1), lp1),
    }
    if teacher_protected_target is not None:
        before_rmse, before_corr = functional_agreement(lp0, teacher_protected_target)
        after_rmse, after_corr = functional_agreement(lp1, teacher_protected_target)
        payload.update(
            {
                "functional_distillation_RMSE_before": before_rmse,
                "functional_distillation_correlation_before": before_corr,
                "functional_distillation_RMSE_after": after_rmse,
                "functional_distillation_correlation_after": after_corr,
            }
        )
    return payload


def approximate_macs(model: nn.Module, input_shape: tuple[int, int, int] = (1, 62, 1000)) -> int:
    macs = 0
    hooks = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            out = output
            kernel = module.kernel_size[0] * module.kernel_size[1]
            per_output = (module.in_channels // module.groups) * kernel
            macs += int(out.numel() * per_output)
        elif isinstance(module, nn.Linear):
            macs += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(input_shape, device=device))
    if was_training:
        model.train()
    for handle in hooks:
        handle.remove()
    return int(macs)


def model_sha256(model: nn.Module, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)
    return sha256_file(path)
