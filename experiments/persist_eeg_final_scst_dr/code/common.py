from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
PROTOCOL = EXP / "protocol" / "STAGE0_PROTOCOL_LOCK.json"
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
RUNTIME = EXP / "runtime"

P2 = REPO / "experiments" / "persist_eeg_subject_invariance_stress_test_v1"
P3 = REPO / "experiments" / "persist_eeg_wbcic_independent_replication_v1"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P2_COMMON = _load_module("scst_p2_common", P2 / "code" / "common.py")
P3_COMMON = _load_module("scst_p3_common", P3 / "code" / "common.py")
P4A_COMMON = _load_module("scst_p4a_common", P4A / "code" / "common.py")


SETTINGS: dict[str, dict[str, str]] = {
    "OPENBMI_MI_EEGNET": {"dataset": "OpenBMI", "backbone": "EEGNet", "source": "P2", "key": "eegnet"},
    "OPENBMI_MI_EEGCONFORMER": {"dataset": "OpenBMI", "backbone": "EEGConformer", "source": "P2", "key": "eegconformer"},
    "WBCIC_MI_EEGNET": {"dataset": "WBCIC", "backbone": "EEGNet", "source": "P3", "key": "eegnet"},
    "WBCIC_MI_EEGCONFORMER": {"dataset": "WBCIC", "backbone": "EEGConformer", "source": "P4A", "key": "S4"},
}


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
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
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_status_porcelain() -> list[str]:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).splitlines()


def ensure_dirs() -> None:
    for path in (RESULTS, FIGURES, RUNTIME, EXP / "protocol"):
        path.mkdir(parents=True, exist_ok=True)


def protocol() -> dict[str, Any]:
    value = read_json(PROTOCOL)
    if value.get("schema") != "PERSIST_EEG_SCST_DR_STAGE0_PROTOCOL_V1":
        raise RuntimeError("Stage-0 protocol schema mismatch")
    if value.get("frozen_before_transport_outcome_access") is not True:
        raise RuntimeError("Stage-0 protocol is not frozen")
    return value


def subject_sort(values: Iterable[str]) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        token = str(value).replace("sub-", "")
        return (int(token) if token.isdigit() else 10**9, str(value))
    return sorted(map(str, values), key=key)


@dataclass
class SettingData:
    setting_id: str
    spec: dict[str, str]
    x: np.ndarray
    metadata: pd.DataFrame
    source_sessions: tuple[int, int]
    cache_root: Path


def load_setting_data(setting_id: str) -> SettingData:
    spec = SETTINGS[setting_id]
    if spec["dataset"] == "OpenBMI":
        data = P2_COMMON.load_data()
        metadata = data.metadata.copy()
        return SettingData(setting_id, spec, data.x, metadata, (1, 2), Path(data.cache_root))
    data = P3_COMMON.load_data()
    metadata = data.metadata.copy()
    metadata["subject_id"] = metadata.subject_id.astype(str).str.replace("sub-", "", regex=False)
    return SettingData(setting_id, spec, data.x, metadata, (0, 1), Path(data.cache_root))


def fold_roles(setting_id: str, fold: int) -> dict[str, tuple[str, ...]]:
    spec = SETTINGS[setting_id]
    if spec["dataset"] == "OpenBMI":
        role = P2_COMMON.frozen_fold(fold)
        return {
            "model_fit": tuple(subject_sort(role["inner_train"])),
            "validation": tuple(subject_sort(role["inner_validation"])),
            "outcome": tuple(subject_sort(role["outcome"])),
        }
    role = P3_COMMON.frozen_fold(fold)
    return {
        "model_fit": tuple(subject_sort(str(x).replace("sub-", "") for x in role["model_fit"])),
        "validation": tuple(subject_sort(str(x).replace("sub-", "") for x in role["validation_discovery"])),
        "outcome": tuple(subject_sort(str(x).replace("sub-", "") for x in role["outcome"])),
    }


def row_indices(metadata: pd.DataFrame, subjects: Iterable[str], sessions: Iterable[int]) -> np.ndarray:
    subject_set = set(map(str, subjects))
    session_set = set(map(int, sessions))
    # Arrow-backed pandas columns can expose a read-only NumPy view.  The mask
    # is updated in-place below, so request an explicit writable copy.
    mask = metadata.subject_id.astype(str).isin(subject_set).to_numpy(copy=True)
    mask &= metadata.session_id.astype(int).isin(session_set).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def unit_context(setting_id: str, fold: int, seed: int = 0) -> Path:
    spec = SETTINGS[setting_id]
    if spec["source"] == "P2":
        return P2_COMMON.unit_dir(spec["key"], fold, seed)
    if spec["source"] == "P3":
        return P3_COMMON.unit_dir(spec["key"], fold, seed)
    return P4A_COMMON.run_dir(spec["key"], fold, seed)


def checkpoint_path(setting_id: str, fold: int, seed: int = 0) -> Path:
    return unit_context(setting_id, fold, seed) / "checkpoints" / "erm__lambda-0.00.pt"


def normalizer_path(setting_id: str, fold: int, seed: int = 0) -> Path:
    return unit_context(setting_id, fold, seed) / "normalizer.npz"


def load_model(setting_id: str, fold: int, device: torch.device, seed: int = 0):
    spec = SETTINGS[setting_id]
    context = unit_context(setting_id, fold, seed)
    unit = read_json(context / "UNIT_PROTOCOL.json")
    initialization_seed = int(unit["initialization_seed"])
    if spec["source"] == "P2":
        model = P2_COMMON.build_model(spec["key"], initialization_seed)
    elif spec["source"] == "P3":
        model = P3_COMMON.build_model(spec["key"], initialization_seed)
    else:
        model = P4A_COMMON.build_model(spec["key"], initialization_seed)
    payload = torch.load(checkpoint_path(setting_id, fold, seed), map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model.to(device), unit


def load_normalizer(setting_id: str, fold: int, device: torch.device, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    payload = np.load(normalizer_path(setting_id, fold, seed), allow_pickle=False)
    mean = torch.as_tensor(payload["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(payload["std"], dtype=torch.float32, device=device)
    return mean, std


def normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-6)


def forward_layers(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    if hasattr(model, "transformer"):
        value = model.dropout(model.pool(F.elu(model.norm(model.spatial(model.temporal(x.unsqueeze(1)))))))
        token = value.squeeze(2).transpose(1, 2)
        token = model.transformer(token + model.position[:, : token.shape[1]])
        pre = token.mean(dim=1)
        final = model.embedding(pre)
    else:
        value = model.bn1(model.temporal(x.unsqueeze(1)))
        value = model.drop1(model.pool1(F.elu(model.bn2(model.spatial(value)))))
        value = model.drop2(model.pool2(F.elu(model.bn3(model.point(model.depth(value))))))
        pre = value.flatten(1)
        final = model.embedding(pre)
    return {"pre_embedding": pre, "final_embedding": final}


def extract_layers(
    model,
    raw: torch.Tensor,
    metadata: pd.DataFrame,
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int = 512,
) -> dict[str, Any]:
    chunks: dict[str, list[np.ndarray]] = {"pre_embedding": [], "final_embedding": []}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            idx_np = indices[start:start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=raw.device)
            x = normalize(raw[idx].float(), mean, std)
            with torch.autocast(device_type=raw.device.type, dtype=torch.bfloat16, enabled=raw.device.type == "cuda"):
                values = forward_layers(model, x)
            for layer, value in values.items():
                chunks[layer].append(value.float().cpu().numpy())
    selected = metadata.iloc[indices]
    return {
        "layers": {layer: np.concatenate(parts).astype(np.float32) for layer, parts in chunks.items()},
        "labels": selected.label.to_numpy(np.int64),
        "subjects": selected.subject_id.astype(str).to_numpy(),
        "sessions": selected.session_id.to_numpy(np.int64),
        "indices": indices.copy(),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b) / denominator)


def centered_logits(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    return value - value.mean(axis=-1, keepdims=True)
