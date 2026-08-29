"""Shared utilities for the prospective SCST Stage-1 experiment.

This module deliberately separates source-only model screening from future-session
utility.  It only reads the already-authorized development caches and never opens
the sealed WBCIC outer subjects or the OpenBMI internal holdout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_utility_stage1"
CODE = EXP / "code"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"

DATASETS = ("OpenBMI", "WBCIC")
FOLDS = tuple(range(5))
SEEDS = tuple(range(3))
SOURCE_SESSIONS = {"OpenBMI": (1, 2), "WBCIC": (0, 1)}
FUTURE_SESSION = {"WBCIC": 2}
THRESHOLDS = {"OpenBMI": 0.7519166667, "WBCIC": 0.7684300821}
HISTORICAL_ANCHORS = {"OpenBMI": 0.7719, "WBCIC": 0.7884}
MANIFOLD_THRESHOLDS = (1.20, 1.25, 1.30, 1.35)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P2 = load_module(
    "stage1_p2_common",
    REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1" / "code" / "common.py",
)
P3 = load_module(
    "stage1_p3_common",
    REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1" / "code" / "common.py",
)
P4A = load_module(
    "stage1_p4a_common",
    REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1" / "code" / "common.py",
)
OLD_MODELS = load_module(
    "stage1_old_specialist_models",
    REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "code" / "specialist_models.py",
)


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.tobytes())
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted((str(v).replace("sub-", "") for v in values), key=lambda x: (int(x) if x.isdigit() else 10**9, x))


def load_data(dataset: str):
    data = P2.load_data() if dataset == "OpenBMI" else P3.load_data()
    metadata = data.metadata.copy()
    metadata["subject_id"] = metadata.subject_id.astype(str).str.replace("sub-", "", regex=False)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = metadata.label.astype(int)
    return data.x, metadata.reset_index(drop=True), Path(data.cache_root)


def roles(dataset: str, fold: int) -> dict[str, tuple[str, ...]]:
    role = P2.frozen_fold(fold) if dataset == "OpenBMI" else P3.frozen_fold(fold)
    if dataset == "OpenBMI":
        result = {"model_fit": role["inner_train"], "validation": role["inner_validation"], "outcome": role["outcome"]}
    else:
        result = {"model_fit": role["model_fit"], "validation": role["validation_discovery"], "outcome": role["outcome"]}
    return {key: tuple(subject_sort(value)) for key, value in result.items()}


def row_indices(metadata: pd.DataFrame, subjects: Iterable[object], sessions: Iterable[int]) -> np.ndarray:
    mask = metadata.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy(copy=True)
    mask &= metadata.session_id.astype(int).isin(set(map(int, sessions))).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def normalize_raw(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, np.float32)
    value = value - value.mean(axis=-1, keepdims=True)
    scale = value.std(axis=-1, keepdims=True)
    return value / np.maximum(scale, 1e-6)


def channel_normalizer(raw: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # All statistics are source model-fit only, preserving the old protocol.
    value = np.asarray(raw[indices], np.float32)
    mean = value.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = value.std(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_channel_normalizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, np.float32) - mean[None, :, None]) / std[None, :, None]


def metrics(labels: np.ndarray, logits: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, np.int64)
    logits = np.asarray(logits, np.float64)
    pred = logits.argmax(1)
    per_ba = [balanced_accuracy_score(labels[subjects.astype(str) == s], pred[subjects.astype(str) == s]) for s in subject_sort(np.unique(subjects))]
    per_f1 = [f1_score(labels[subjects.astype(str) == s], pred[subjects.astype(str) == s], average="macro", zero_division=0) for s in subject_sort(np.unique(subjects))]
    stable = logits - logits.max(1, keepdims=True)
    p = np.exp(stable)
    p /= p.sum(1, keepdims=True)
    return {
        "BA": float(np.mean(per_ba)),
        "trial_BA": float(balanced_accuracy_score(labels, pred)),
        "macro_F1": float(np.mean(per_f1)),
        "NLL": float(log_loss(labels, p, labels=[0, 1])),
    }


def save_rep(path: Path, value: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_value = {key: (np.asarray(item).astype("U") if np.asarray(item).dtype == object else np.asarray(item)) for key, item in value.items()}
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **clean_value)
    os.replace(temporary, path)


def load_rep(path: Path) -> dict[str, np.ndarray]:
    # Frozen Stage-0 archives stored subject strings as object arrays. These are
    # trusted local artifacts created by this repository; convert them to plain
    # Unicode immediately so object values never propagate downstream.
    with np.load(path, allow_pickle=True) as values:
        out = {key: values[key] for key in values.files}
    if out.get("subjects", np.empty(0)).dtype == object:
        out["subjects"] = out["subjects"].astype("U")
    return out


def rep_path(model: str, dataset: str, fold: int, seed: int, role: str) -> Path:
    return RUNTIME / "representations" / model / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"


def model_checkpoint_path(model: str, dataset: str, fold: int, seed: int) -> Path:
    return RUNTIME / "checkpoints" / model / dataset / f"fold-{fold}" / f"seed-{seed}.pt"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def ensure_dirs() -> None:
    for path in (CODE, RESULTS, FIGURES, PROTOCOL, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits_a.float(), dim=-1)
    log_q = torch.log_softmax(logits_b.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    return 0.5 * ((p * (log_p - log_q)).sum(-1) + (q * (log_q - log_p)).sum(-1)).mean()


def model_kind(model: str) -> str:
    if model == "ATCNet-CleanRoom":
        return "cleanroom"
    if model == "ATCNet-Official":
        return "braindecode_atcnet"
    if model == "EEGNeX":
        return "braindecode_eegnex"
    if model == "EEGNet":
        return "historical_eegnet"
    if model == "EEGConformer":
        return "historical_eegconformer"
    raise KeyError(model)


def build_model(model: str, channels: int, n_times: int = 1000) -> nn.Module:
    kind = model_kind(model)
    if kind == "cleanroom":
        return OLD_MODELS.ATCNet(channels)
    if kind == "braindecode_atcnet":
        from braindecode.models import ATCNet

        return ATCNet(n_chans=channels, n_outputs=2, n_times=n_times, sfreq=250.0, input_window_seconds=n_times / 250.0)
    if kind == "braindecode_eegnex":
        from braindecode.models import EEGNeX

        return EEGNeX(
            n_chans=channels,
            n_outputs=2,
            n_times=n_times,
            sfreq=250.0,
        )
    if kind == "historical_eegnet":
        return P3.StandardEEGNet() if hasattr(P3, "StandardEEGNet") else P2.StandardEEGNet()
    if kind == "historical_eegconformer":
        # WBCIC's frozen historical control is P4A setting S4 (58 channels),
        # not the 62-channel OpenBMI P2 implementation.
        return P4A.build_model("S4", stable_seed("stage1-eegconformer-build", channels, n_times))
    raise KeyError(model)


def model_features(model: str, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    kind = model_kind(model)
    if kind == "cleanroom":
        return net.forward_features(x)
    if kind == "braindecode_atcnet":
        # Exact braindecode ATCNet forward up to the per-window TCN feature.
        value = net.ensuredims(x)
        value = net.dimshuffle(value)
        conv = net.conv_block(value).view(-1, net.F2, net.Tc)
        outputs = []
        for idx, (attention, tcn) in enumerate(zip(net.attention_blocks, net.temporal_conv_nets)):
            window = conv[..., idx : idx + net.Tw]
            attended = attention(window)
            outputs.append(tcn(attended)[..., -1])
        # Concatenating all five window features preserves the mature model's
        # exact classifier factorization while exposing one vector for the
        # geometry audit (5 * F2 dimensions with default settings).
        return torch.cat(outputs, dim=1)
    if kind == "braindecode_eegnex":
        value = net.block_1(x)
        value = net.block_2(value)
        value = net.block_3(value)
        value = net.block_4(value)
        return net.block_5(value)
    if kind in {"historical_eegnet", "historical_eegconformer"}:
        return net.forward_features(x)
    raise KeyError(model)


def feature_logits(model: str, net: nn.Module, h: torch.Tensor) -> torch.Tensor:
    if model_kind(model) == "braindecode_atcnet":
        chunks = h.split(net.F2, dim=1)
        if len(chunks) != len(net.final_layer):
            raise RuntimeError("ATCNet window-feature decomposition mismatch")
        return torch.stack([head(value) for head, value in zip(net.final_layer, chunks)], dim=0).mean(0)
    if model_kind(model) == "braindecode_eegnex":
        return net.final_layer(h)
    return net.head(h)


def model_logits(model: str, net: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = model_features(model, net, x)
    return h, feature_logits(model, net, h)


def infer_model(model: str, net: nn.Module, raw: np.ndarray, metadata: pd.DataFrame, indices: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None, device: torch.device, batch_size: int = 128) -> dict[str, np.ndarray]:
    net.eval()
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            # Match the frozen Stage-0 specialist protocol: per-trial,
            # per-channel temporal standardization. ``mean``/``std`` remain in
            # the signature for checkpoint compatibility but are not used.
            value = normalize_raw(raw[idx])
            tensor = torch.from_numpy(value).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                h, z = model_logits(model, net, tensor)
            features.append(h.float().cpu().numpy())
            logits.append(z.float().cpu().numpy())
    picked = metadata.iloc[indices]
    return {
        "indices": np.asarray(indices, np.int64),
        "features": np.concatenate(features).astype(np.float32),
        "logits": np.concatenate(logits).astype(np.float32),
        "labels": picked.label.to_numpy(np.int64),
        "subjects": picked.subject_id.astype(str).to_numpy(),
        "sessions": picked.session_id.to_numpy(np.int64),
    }
