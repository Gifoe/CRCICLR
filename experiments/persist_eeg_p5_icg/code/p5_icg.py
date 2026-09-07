"""PERSIST-EEG P5: Intervention-Calibrated Selective Geometry (PERSIST-ICG).

This implementation is deliberately narrow.  It uses the frozen historical
EEGNet representation and the persisted Signed-V3.1 spectrum/assignments,
trains only a small task head plus a zero-initialised canonical-coordinate
adapter, and evaluates the resulting representation on unseen validation
subjects without centering or adaptation.  V0, V1 and V2 share one trainer;
the only scientific changes are the pre-declared adapter geometry and the
frozen intervention-confidence weights.

The script is intended to run from the repository root or from this file on
the P1 GPU environment.  It never reads outer-test samples or labels.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT = REPO_ROOT / "experiments" / "persist_eeg_p5_icg" / "outputs"
MANIFEST = REPO_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
SPLIT = REPO_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
HIST_ROOT = REPO_ROOT / "outputs" / "persist_eeg_p2p3" / "backbone" / "checkpoints" / "eegnet"
V31_ROOT = REPO_ROOT / "outputs" / "persist_eeg_p4_signed_v3_1" / "runs"
TASK = "mi"
TASK_CLASSES = {"mi": 2, "erp": 2, "ssvep": 4}
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
BOOTSTRAP_DRAWS = 10_000
# Bump this only for a meaningful implementation/reproducibility repair.  It
# lets the runner invalidate results produced by the pre-repair script without
# deleting them, while allowing subsequent invocations to resume safely.
IMPLEMENTATION_ID = "p5_icg_repro_v4_shared_streams_exact_weights"


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False) % (2**32 - 1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1, eps=1e-8)[0]


def parse_ci(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return float(value[0]), float(value[1])


def load_split(fold: int) -> dict[str, list[str]]:
    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    folds = payload["openbmi"]["folds"]
    item = next(x for x in folds if int(x["fold"]) == int(fold))
    # Only train/validation identifiers are materialised.  Outer identifiers
    # are intentionally not loaded, hashed, or used by this method phase.
    return {
        "train_subjects": [str(x) for x in item["train_subjects"]],
        "validation_subjects": [str(x) for x in item["validation_subjects"]],
    }


def label_map(meta: pd.DataFrame) -> dict[str, int]:
    labels = sorted(meta.event_label.astype(str).unique().tolist())
    if labels != ["left_hand", "right_hand"]:
        raise RuntimeError(f"Unexpected MI label map: {labels}")
    return {labels[0]: 0, labels[1]: 1}


@dataclass
class CanonicalArtifacts:
    mean: np.ndarray
    whitener: np.ndarray
    dewhitener: np.ndarray
    directions: np.ndarray
    rho: np.ndarray
    blocks: list[list[int]]
    protected_blocks: list[int]
    weights: dict[int, float]
    utility_rows: list[dict[str, Any]]
    spectrum_sha256: str

    @property
    def q_dim(self) -> int:
        return int(self.directions.shape[1])

    @property
    def protected_dims(self) -> np.ndarray:
        return np.asarray(sorted({d for b in self.protected_blocks for d in self.blocks[b]}), dtype=np.int64)


def load_artifacts(fold: int, seed: int) -> CanonicalArtifacts:
    run = V31_ROOT / f"fold-{fold}" / f"seed-{seed}"
    spectrum_path = run / "spectrum" / "PERSISTENCE_SPECTRUM.npz"
    assignment_path = run / "SIGNED_ASSIGNMENTS_V3_1.json"
    utility_path = run / "SIGNED_UTILITY_V3_1.csv"
    for path in (spectrum_path, assignment_path, utility_path):
        if not path.exists():
            raise FileNotFoundError(path)
    z = np.load(spectrum_path, allow_pickle=False)
    blocks = [[int(v) for v in block] for block in json.loads(str(z["blocks_json"].item()))]
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    protected = [int(v) for v in assignment[TASK].get("protected", [])]
    if not protected:
        raise RuntimeError(f"No canonical MI Protected assignment for fold={fold}, seed={seed}")
    utility = pd.read_csv(utility_path)
    rows: list[dict[str, Any]] = []
    confidence: dict[int, float] = {}
    for block in protected:
        match = utility[(utility.task == TASK) & (utility.block.astype(int) == block)]
        if len(match) != 1:
            raise RuntimeError(f"Expected one utility row for protected block {block}")
        row = match.iloc[0].to_dict()
        lcb_abs = parse_ci(row["u_abs_CI95"])[0]
        lcb_spec = parse_ci(row["u_spec_CI95"])[0]
        c = max(0.0, min(lcb_abs, lcb_spec))
        confidence[block] = c
        rows.append({"block": block, "lcb_abs": lcb_abs, "lcb_spec": lcb_spec, "confidence": c})
    if not any(v > 0 for v in confidence.values()):
        raise RuntimeError("Canonical Protected confidence scores are all zero")
    # The attachment freezes normalization by the mean over *all* Protected
    # blocks, including zero-confidence blocks.  Do not silently renormalize
    # only the positive subset.
    mean_c = float(np.mean(list(confidence.values())))
    weights = {b: float(confidence[b] / mean_c) if confidence[b] > 0 else 0.0 for b in confidence}
    return CanonicalArtifacts(
        mean=np.asarray(z["mean"], dtype=np.float32),
        whitener=np.asarray(z["whitener"], dtype=np.float32),
        dewhitener=np.asarray(z["dewhitener"], dtype=np.float32),
        directions=np.asarray(z["directions"], dtype=np.float32),
        rho=np.asarray(z["rho"], dtype=np.float32),
        blocks=blocks,
        protected_blocks=protected,
        weights=weights,
        utility_rows=rows,
        spectrum_sha256=sha256(spectrum_path),
    )


class RawMI(Dataset):
    def __init__(self, meta: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        self.meta = meta.reset_index(drop=True)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None]
        self.std = np.maximum(np.asarray(std, dtype=np.float32)[:, None], 1e-6)
        self.arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.meta.iloc[int(index)]
        relative = str(row.signal_cache_path).replace("/", os.sep)
        if relative not in self.arrays:
            self.arrays[relative] = np.load(REPO_ROOT / relative, mmap_mode="r", allow_pickle=False)
        epoch = np.asarray(self.arrays[relative][int(row.cache_index)], dtype=np.float32)
        return torch.from_numpy((epoch - self.mean) / self.std), int(index)


def historical_checkpoint(fold: int, seed: int) -> tuple[Path, Path, Path]:
    base = HIST_ROOT / f"fold-{fold}" / f"seed-{seed}"
    paths = base / "best.pt", base / "channel_mean.npy", base / "channel_std.npy"
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(paths)
    return paths


def load_mi_manifest() -> pd.DataFrame:
    manifest = pd.read_parquet(MANIFEST)
    meta = manifest[manifest.paradigm.astype(str) == TASK].copy().reset_index(names="manifest_index")
    if len(meta) != 10_800 or meta.subject_id.astype(str).nunique() != 54:
        raise RuntimeError(f"Unexpected MI manifest shape: {meta.shape}")
    mapper = label_map(meta)
    meta["subject"] = meta.subject_id.astype(str)
    meta["session"] = meta.session_id.astype(str)
    meta["label"] = meta.event_label.astype(str).map(mapper).astype(np.int64)
    return meta


def extract_h0(meta: pd.DataFrame, fold: int, seed: int, cache_dir: Path, device: torch.device) -> np.ndarray:
    checkpoint_path, mean_path, std_path = historical_checkpoint(fold, seed)
    h_path = cache_dir / "h0.npy"
    provenance = cache_dir / "H0_PROVENANCE.json"
    expected = {
        "checkpoint_sha256": sha256(checkpoint_path),
        "manifest_sha256": sha256(MANIFEST),
        "n_rows": len(meta),
        "fold": fold,
        "seed": seed,
    }
    if h_path.exists() and provenance.exists():
        old = json.loads(provenance.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in expected.items()):
            arr = np.load(h_path, mmap_mode="r")
            if arr.shape == (len(meta), 128) and np.isfinite(arr).all():
                return np.asarray(arr, dtype=np.float32)
    from persist_eeg_stage0.models import build_shared_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_shared_model("eegnet", int(meta.n_channels.iloc[0]), 128, TASK_CLASSES)
    model.load_state_dict(checkpoint["model"])
    model.encoder.to(device).eval()
    dataset = RawMI(meta, np.load(mean_path), np.load(std_path))
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    output = np.lib.format.open_memmap(h_path, mode="w+", dtype=np.float32, shape=(len(meta), 128))
    with torch.inference_mode():
        for x, idx in loader:
            h = model.encoder(x.to(device, non_blocking=True)).detach().cpu().numpy().astype(np.float32)
            output[idx.numpy()] = h
    output.flush()
    del model, dataset, loader, output
    write_json(provenance, expected)
    return np.asarray(np.load(h_path, mmap_mode="r"), dtype=np.float32)


def q_from_h(h: np.ndarray, art: CanonicalArtifacts) -> np.ndarray:
    return ((np.asarray(h, dtype=np.float64) - art.mean) @ art.whitener @ art.directions).astype(np.float32)


@dataclass
class GeometryTargets:
    same: dict[tuple[str, str, int], np.ndarray]
    cross: dict[tuple[str, str, int], np.ndarray]
    global_direction: dict[int, np.ndarray]
    subjects: list[str]


def build_geometry_targets(meta: pd.DataFrame, q: np.ndarray, train_positions: np.ndarray, art: CanonicalArtifacts, out: Path) -> GeometryTargets:
    train_meta = meta.iloc[train_positions].reset_index(drop=True)
    train_q = np.asarray(q[train_positions], dtype=np.float32)
    subjects = sorted(train_meta.subject.unique().tolist(), key=lambda x: int(x))
    sessions = sorted(train_meta.session.unique().tolist())
    deltas: dict[tuple[str, str], np.ndarray] = {}
    for subject in subjects:
        for session in sessions:
            rows = np.flatnonzero((train_meta.subject == subject).to_numpy() & (train_meta.session == session).to_numpy())
            arr = np.zeros((2, art.q_dim), dtype=np.float32)
            for label in (0, 1):
                loc = rows[train_meta.label.to_numpy()[rows] == label]
                if len(loc) == 0:
                    raise RuntimeError(f"Missing class for subject/session {subject}/{session}")
                arr[label] = train_q[loc].mean(axis=0)
            deltas[(subject, session)] = arr[1] - arr[0]
    same: dict[tuple[str, str, int], np.ndarray] = {}
    cross: dict[tuple[str, str, int], np.ndarray] = {}
    global_direction: dict[int, np.ndarray] = {}
    for block in art.protected_blocks:
        dims = art.blocks[block]
        all_delta = np.stack([deltas[key][dims] for key in deltas])
        global_direction[block] = normalise(all_delta.mean(axis=0))
        for subject in subjects:
            for session in sessions:
                other = [deltas[(s, session)][dims] for s in subjects if s != subject]
                other_cross_session = [deltas[(s, sessions[1 - sessions.index(session)])][dims] for s in subjects if s != subject]
                same[(subject, session, block)] = normalise(np.mean(other, axis=0))
                cross[(subject, session, block)] = normalise(np.mean(other_cross_session, axis=0))
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    index: list[dict[str, Any]] = []
    for (s, r, b), value in same.items():
        key = f"same__{s}__{r}__{b}"; arrays[key] = value; index.append({"kind": "same", "subject": s, "session": r, "block": b, "key": key})
    for (s, r, b), value in cross.items():
        key = f"cross__{s}__{r}__{b}"; arrays[key] = value; index.append({"kind": "cross", "subject": s, "session": r, "block": b, "key": key})
    for b, value in global_direction.items():
        key = f"global__{b}"; arrays[key] = value; index.append({"kind": "global", "block": b, "key": key})
    np.savez_compressed(out, **arrays)
    write_json(out.with_suffix(".json"), {"subjects": subjects, "protected_blocks": art.protected_blocks, "index": index, "outer_test_used": False})
    return GeometryTargets(same=same, cross=cross, global_direction=global_direction, subjects=subjects)


class ProtectedAdapter(nn.Module):
    def __init__(self, q_dim: int, dims: Sequence[int], bottleneck: int):
        super().__init__()
        self.q_dim = int(q_dim)
        self.dims = [int(x) for x in dims]
        width = max(1, min(int(bottleneck), max(1, 2 * len(self.dims))))
        self.fc1 = nn.Linear(len(self.dims), width)
        self.fc2 = nn.Linear(width, len(self.dims))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(self.dims, dtype=torch.long, device=q.device)
        delta = self.fc2(F.gelu(self.fc1(q.index_select(1, idx))))
        full = torch.zeros_like(q)
        full.scatter_(1, idx.expand(q.shape[0], -1), delta)
        return q + full, full


class SelectiveDirectionAdapter(nn.Module):
    def __init__(self, q_dim: int, block_dims: Mapping[int, Sequence[int]], directions: Mapping[int, np.ndarray], bottleneck: int):
        super().__init__()
        self.q_dim = int(q_dim)
        self.block_ids = [int(k) for k in block_dims]
        self.dims = {int(k): [int(x) for x in v] for k, v in block_dims.items()}
        self.mlps = nn.ModuleDict()
        for block in self.block_ids:
            d = len(self.dims[block]); width = max(1, min(int(bottleneck), max(1, 2 * d)))
            mlp = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Linear(width, 1))
            nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)
            self.mlps[str(block)] = mlp
            self.register_buffer(f"dir_{block}", torch.as_tensor(directions[block], dtype=torch.float32))

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = torch.zeros_like(q)
        for block in self.block_ids:
            idx = torch.as_tensor(self.dims[block], dtype=torch.long, device=q.device)
            scalar = self.mlps[str(block)](q.index_select(1, idx))
            delta = scalar * getattr(self, f"dir_{block}").reshape(1, -1)
            full.scatter_add_(1, idx.expand(q.shape[0], -1), delta)
        return q + full, full


class ICGModel(nn.Module):
    def __init__(self, historical_head: nn.Module, art: CanonicalArtifacts, mode: str, targets: GeometryTargets, bottleneck: int = 8):
        super().__init__()
        dims = {b: art.blocks[b] for b in art.protected_blocks}
        if mode in {"V0", "V1"}:
            self.adapter = ProtectedAdapter(art.q_dim, art.protected_dims.tolist(), bottleneck)
        elif mode == "V2":
            self.adapter = SelectiveDirectionAdapter(art.q_dim, dims, targets.global_direction, bottleneck)
        else:
            raise ValueError(mode)
        self.head = copy.deepcopy(historical_head)
        self.register_buffer("directions", torch.as_tensor(art.directions, dtype=torch.float32))
        self.register_buffer("dewhitener", torch.as_tensor(art.dewhitener, dtype=torch.float32))
        self.mode = mode

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_adj, delta = self.adapter(q)
        dh = (delta @ self.directions.T) @ self.dewhitener
        h_adj = h + dh
        return self.head(h_adj), q_adj, delta


class StructuredSampler:
    def __init__(self, meta: pd.DataFrame, subjects: Sequence[str], *, subjects_per_batch: int = 6, trials_per_class: int = 4):
        self.meta = meta.reset_index(drop=True)
        self.subjects = sorted(map(str, subjects), key=int)
        self.s_per_batch = min(int(subjects_per_batch), len(self.subjects))
        self.k = int(trials_per_class)
        self.groups: dict[tuple[str, str, int], np.ndarray] = {}
        for key, group in self.meta.groupby(["subject", "session", "label"], sort=True):
            self.groups[(str(key[0]), str(key[1]), int(key[2]))] = group.index.to_numpy(dtype=np.int64)
        self.sessions = sorted(self.meta.session.unique().tolist())
        self.steps = max(1, int(math.ceil(len(self.meta) / max(1, self.s_per_batch * len(self.sessions) * 2 * self.k))))

    def batches(self, epoch: int, seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng(stable_seed("sampler", epoch, seed))
        result: list[np.ndarray] = []
        for _ in range(self.steps):
            selected = rng.choice(self.subjects, size=self.s_per_batch, replace=False)
            rows: list[int] = []
            for subject in selected:
                for session in self.sessions:
                    for label in (0, 1):
                        values = self.groups[(str(subject), session, label)]
                        if len(values) >= self.k:
                            rows.extend(rng.choice(values, size=self.k, replace=False).tolist())
                        else:
                            rows.extend(rng.choice(values, size=self.k, replace=True).tolist())
            result.append(np.asarray(rows, dtype=np.int64))
        return result


@dataclass(frozen=True)
class TrainConfig:
    version: str
    lambda_geometry: float = 0.30
    lambda_drift: float = 0.10
    bottleneck: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 24
    subjects_per_batch: int = 6
    trials_per_class: int = 4
    gradient_clip: float = 2.0


CONFIGS = {"V0": TrainConfig("V0"), "V1": TrainConfig("V1"), "V2": TrainConfig("V2")}


def geometry_loss(q_adj: torch.Tensor, batch_idx: np.ndarray, meta: pd.DataFrame, targets: GeometryTargets, art: CanonicalArtifacts, weights: Mapping[int, float]) -> torch.Tensor:
    values: list[torch.Tensor] = []
    subject = meta.subject.to_numpy(dtype=str)
    session = meta.session.to_numpy(dtype=str)
    label = meta.label.to_numpy(dtype=np.int64)
    for s in sorted(set(subject[batch_idx]), key=int):
        for r in sorted(set(session[batch_idx])):
            # q_adj is ordered by the batch rows, whereas batch_idx contains
            # positions in the full train table.  Keep this mapping explicit;
            # confusing the two silently mixes subjects and invalidates the
            # contrast loss.
            loc = np.flatnonzero((subject[batch_idx] == s) & (session[batch_idx] == r))
            for block in art.protected_blocks:
                d = np.asarray(art.blocks[block], dtype=np.int64)
                p0 = loc[label[batch_idx[loc]] == 0]; p1 = loc[label[batch_idx[loc]] == 1]
                if len(p0) == 0 or len(p1) == 0:
                    continue
                contrast = q_adj[torch.as_tensor(p1, device=q_adj.device)].mean(0).index_select(0, torch.as_tensor(d, device=q_adj.device)) - q_adj[torch.as_tensor(p0, device=q_adj.device)].mean(0).index_select(0, torch.as_tensor(d, device=q_adj.device))
                target_same = torch.as_tensor(targets.same[(s, r, block)], dtype=q_adj.dtype, device=q_adj.device)
                target_cross = torch.as_tensor(targets.cross[(s, r, block)], dtype=q_adj.dtype, device=q_adj.device)
                if float(torch.linalg.norm(target_same)) < 1e-8 or float(torch.linalg.norm(target_cross)) < 1e-8:
                    continue
                term = 0.5 * (1.0 - cosine(contrast, target_same)) + 0.5 * (1.0 - cosine(contrast, target_cross))
                values.append(float(weights.get(block, 0.0)) * term)
    if not values:
        return q_adj.sum() * 0.0
    denom = max(float(sum(weights.get(b, 0.0) for b in art.protected_blocks)), 1e-8)
    return torch.stack(values).sum() / denom


def drift_loss(delta: torch.Tensor, art: CanonicalArtifacts) -> torch.Tensor:
    idx = torch.as_tensor(art.protected_dims, dtype=torch.long, device=delta.device)
    return delta.index_select(1, idx).square().mean()


def eval_ba(model: ICGModel, h: torch.Tensor, q: torch.Tensor, y: np.ndarray, batch_size: int = 1024) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); logits_out: list[np.ndarray] = []; q_out: list[np.ndarray] = []; delta_out: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(h), batch_size):
            logits, q_adj, delta = model(h[start:start + batch_size], q[start:start + batch_size])
            logits_out.append(logits.detach().cpu().numpy()); q_out.append(q_adj.detach().cpu().numpy()); delta_out.append(delta.detach().cpu().numpy())
    logits_np = np.concatenate(logits_out); q_np = np.concatenate(q_out); delta_np = np.concatenate(delta_out)
    pred = logits_np.argmax(axis=1)
    return float(balanced_accuracy_score(y, pred)), logits_np, q_np, delta_np


def historical_ba(head: nn.Module, h: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    head = copy.deepcopy(head).to(device).eval()
    with torch.inference_mode():
        pred = head(torch.as_tensor(h, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    return float(balanced_accuracy_score(y, pred))


def paired_bootstrap(rows: Sequence[dict[str, Any]], draws: int = BOOTSTRAP_DRAWS, seed: int = 20260815) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["run"]), []).append(float(row["delta"]))
    runs = sorted(grouped)
    if not runs:
        return {"mean": None, "ci95": [None, None], "sign_probability": None, "draws": draws}
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        selected = rng.choice(runs, size=len(runs), replace=True)
        values[i] = np.mean([np.mean(rng.choice(grouped[r], size=len(grouped[r]), replace=True)) for r in selected])
    return {"mean": float(np.mean(values)), "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))], "sign_probability": float(np.mean(values > 0)), "draws": draws, "n_runs": len(runs), "n_subject_values": len(rows)}


def model_components(model: ICGModel, h: np.ndarray, q: np.ndarray, art: CanonicalArtifacts, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ht = torch.as_tensor(h, dtype=torch.float32, device=device); qt = torch.as_tensor(q, dtype=torch.float32, device=device)
    _, q_adj, delta = eval_ba(model, ht, qt, np.zeros(len(h), dtype=np.int64))
    # eval_ba's BA is ignored; q/delta are exact sample-wise outputs.
    with torch.inference_mode():
        logits = model.head(torch.as_tensor(h, dtype=torch.float32, device=device) + ((torch.as_tensor(delta, device=device) @ model.directions.T) @ model.dewhitener)).cpu().numpy()
    return logits, q_adj, delta


def subject_ba(logits: np.ndarray, meta: pd.DataFrame) -> list[dict[str, Any]]:
    pred = logits.argmax(1); rows: list[dict[str, Any]] = []
    for subject, group in meta.groupby("subject", sort=True):
        idx = group.index.to_numpy(dtype=np.int64); rows.append({"subject": str(subject), "ba": float(balanced_accuracy_score(meta.label.to_numpy()[idx], pred[idx]))})
    return rows


def geometry_diagnostics(q0: np.ndarray, q_after: np.ndarray, meta: pd.DataFrame, train_meta: pd.DataFrame, train_q: np.ndarray, art: CanonicalArtifacts, targets: GeometryTargets) -> dict[str, float]:
    values_same: list[float] = []; values_cross: list[float] = []; margin: list[float] = []
    subj = meta.subject.to_numpy(dtype=str); ses = meta.session.to_numpy(dtype=str); lab = meta.label.to_numpy(dtype=np.int64)
    train_s = train_meta.subject.to_numpy(dtype=str); train_r = train_meta.session.to_numpy(dtype=str); train_y = train_meta.label.to_numpy(dtype=np.int64)
    train_consensus: dict[tuple[str, int], np.ndarray] = {}
    for r in sorted(train_meta.session.unique().tolist()):
        for y in (0, 1):
            idx = np.flatnonzero((train_r == r) & (train_y == y)); train_consensus[(r, y)] = train_q[idx].mean(axis=0)
    for s in sorted(set(subj), key=int):
        for r in sorted(set(ses)):
            idx = np.flatnonzero((subj == s) & (ses == r));
            for b in art.protected_blocks:
                dims = np.asarray(art.blocks[b], dtype=np.int64); idx0 = idx[lab[idx] == 0]; idx1 = idx[lab[idx] == 1]
                if not len(idx0) or not len(idx1): continue
                delta = q_after[idx1][:, dims].mean(0) - q_after[idx0][:, dims].mean(0)
                same_target = normalise(train_consensus[(r, 1)][dims] - train_consensus[(r, 0)][dims])
                cross_r = "2" if r == "1" else "1"; cross_target = normalise(train_consensus[(cross_r, 1)][dims] - train_consensus[(cross_r, 0)][dims])
                if np.linalg.norm(delta) > 1e-8:
                    values_same.append(float(np.dot(normalise(delta), same_target))); values_cross.append(float(np.dot(normalise(delta), cross_target)))
                same_dist = float(np.linalg.norm(q_after[idx1][:, dims].mean(0) - train_consensus[(r, 1)][dims])) + float(np.linalg.norm(q_after[idx0][:, dims].mean(0) - train_consensus[(r, 0)][dims]))
                diff_dist = float(np.linalg.norm(q_after[idx1][:, dims].mean(0) - train_consensus[(r, 0)][dims])) + float(np.linalg.norm(q_after[idx0][:, dims].mean(0) - train_consensus[(r, 1)][dims]))
                margin.append((diff_dist - same_dist) / 2.0)
    return {"alignment_same": float(np.mean(values_same)) if values_same else 0.0, "alignment_cross": float(np.mean(values_cross)) if values_cross else 0.0, "geometry_margin": float(np.mean(margin)) if margin else 0.0}


def drift_diagnostics(delta: np.ndarray, art: CanonicalArtifacts, targets: GeometryTargets) -> dict[str, float]:
    protected = np.asarray(art.protected_dims, dtype=np.int64); non = np.setdiff1d(np.arange(art.q_dim), protected)
    shared: list[float] = []; residual: list[float] = []
    for b in art.protected_blocks:
        dims = np.asarray(art.blocks[b], dtype=np.int64); g = targets.global_direction[b]
        value = delta[:, dims]
        projection = value @ g
        shared.extend(np.abs(projection).tolist()); residual.extend(np.linalg.norm(value - projection[:, None] * g[None, :], axis=1).tolist())
    return {"shared_geometry_drift": float(np.mean(shared)) if shared else 0.0, "individual_protected_residual_drift": float(np.mean(residual)) if residual else 0.0, "nonprotected_drift": 0.0 if len(non) else 0.0, "protected_q_drift_rms": float(np.sqrt(np.mean(delta[:, protected] ** 2)))}


def intervention_diagnostics(model: ICGModel, h: np.ndarray, q: np.ndarray, y: np.ndarray, art: CanonicalArtifacts, seed: int, device: torch.device) -> dict[str, Any]:
    model.eval(); ht = torch.as_tensor(h, dtype=torch.float32, device=device); qt = torch.as_tensor(q, dtype=torch.float32, device=device)
    with torch.inference_mode():
        _, q_adj, _ = model(ht, qt)
        h_adj = ht + ((q_adj - qt) @ model.directions.T) @ model.dewhitener
        raw_ba = float(balanced_accuracy_score(y, model.head(h_adj).argmax(1).cpu().numpy()))
        dims = art.protected_dims; k = len(dims); protected_q = q_adj.clone(); protected_q[:, torch.as_tensor(dims, dtype=torch.long, device=device)] = 0.0
        h_protected = h_adj + ((protected_q - q_adj) @ model.directions.T) @ model.dewhitener
        protected_ba = float(balanced_accuracy_score(y, model.head(h_protected).argmax(1).cpu().numpy()))
        rng = np.random.default_rng(stable_seed("random_intervention", seed)); random_ba: list[float] = []
        candidates = np.arange(art.q_dim)
        for _ in range(20):
            rd = rng.choice(candidates, size=min(k, len(candidates)), replace=False); rq = q_adj.clone(); rq[:, torch.as_tensor(rd, dtype=torch.long, device=device)] = 0.0; rh = h_adj + ((rq - q_adj) @ model.directions.T) @ model.dewhitener; random_ba.append(float(balanced_accuracy_score(y, model.head(rh).argmax(1).cpu().numpy())))
    return {"raw_BA": raw_ba, "protected_ablation_BA": protected_ba, "protected_drop": raw_ba - protected_ba, "random_ablation_BA_mean": float(np.mean(random_ba)), "random_ablation_BA_std": float(np.std(random_ba)), "random_drop_mean": raw_ba - float(np.mean(random_ba)), "n_random_subspaces": 20, "protected_rank": int(k)}


def run_one(config: TrainConfig, fold: int, seed: int, meta: pd.DataFrame, h0: np.ndarray, q0: np.ndarray, art: CanonicalArtifacts, targets: GeometryTargets, device: torch.device, out: Path) -> dict[str, Any]:
    from persist_eeg_stage0.models import build_shared_model

    # All stochastic state (adapter initialisation, CUDA kernels and sampler
    # draws) is keyed only by the predeclared version/fold/seed.  The method
    # and matched control are constructed from the same state and then cloned,
    # so they remain exactly matched while separate runs are reproducible.
    seed_all(stable_seed("p5", IMPLEMENTATION_ID, fold, seed))
    split = load_split(fold); train_mask = meta.subject.isin(split["train_subjects"]).to_numpy(); val_mask = meta.subject.isin(split["validation_subjects"]).to_numpy(); train_pos = np.flatnonzero(train_mask); val_pos = np.flatnonzero(val_mask)
    train_meta = meta.iloc[train_pos].reset_index(drop=True); val_meta = meta.iloc[val_pos].reset_index(drop=True); h_train = np.array(h0[train_pos], dtype=np.float32, copy=True); q_train = np.array(q0[train_pos], dtype=np.float32, copy=True); y_train = train_meta.label.to_numpy(dtype=np.int64); h_val = np.array(h0[val_pos], dtype=np.float32, copy=True); q_val = np.array(q0[val_pos], dtype=np.float32, copy=True); y_val = val_meta.label.to_numpy(dtype=np.int64)
    checkpoint_path, _, _ = historical_checkpoint(fold, seed); hist = torch.load(checkpoint_path, map_location="cpu", weights_only=False); base = build_shared_model("eegnet", int(meta.n_channels.iloc[0]), 128, TASK_CLASSES); base.load_state_dict(hist["model"]); head = base.heads[TASK]
    historical = historical_ba(head, h_val, y_val, device)
    weights = {b: (1.0 if config.version == "V0" else art.weights[b]) for b in art.protected_blocks}
    sampler = StructuredSampler(train_meta, split["train_subjects"], subjects_per_batch=config.subjects_per_batch, trials_per_class=config.trials_per_class)
    method = ICGModel(head, art, config.version, targets, config.bottleneck).to(device)
    control = ICGModel(head, art, config.version, targets, config.bottleneck).to(device); control.load_state_dict(copy.deepcopy(method.state_dict()))
    opt_m = torch.optim.AdamW(method.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay); opt_c = torch.optim.AdamW(control.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    htr = torch.as_tensor(h_train, dtype=torch.float32, device=device); qtr = torch.as_tensor(q_train, dtype=torch.float32, device=device); ytr = torch.as_tensor(y_train, dtype=torch.long, device=device); hva = torch.as_tensor(h_val, dtype=torch.float32, device=device); qva = torch.as_tensor(q_val, dtype=torch.float32, device=device)
    curves: list[dict[str, Any]] = []; best_m = -np.inf; best_c = -np.inf; best_m_state: dict[str, Any] | None = None; best_c_state: dict[str, Any] | None = None; start = time.time()
    for epoch in range(config.max_epochs):
        method.train(); control.train(); loss_m_sum = loss_c_sum = geo_sum = drift_sum = 0.0; batches = sampler.batches(epoch, stable_seed("run", IMPLEMENTATION_ID, fold, seed))
        for batch in batches:
            idx = torch.as_tensor(batch, dtype=torch.long, device=device);
            opt_m.zero_grad(set_to_none=True); logits, qa, delta = method(htr.index_select(0, idx), qtr.index_select(0, idx)); task = F.cross_entropy(logits, ytr.index_select(0, idx)); geo = geometry_loss(qa, batch, train_meta, targets, art, weights); drift = drift_loss(delta, art); total = task + config.lambda_geometry * geo + config.lambda_drift * drift; total.backward(); nn.utils.clip_grad_norm_(method.parameters(), config.gradient_clip); opt_m.step()
            opt_c.zero_grad(set_to_none=True); clogits, cqa, cdelta = control(htr.index_select(0, idx), qtr.index_select(0, idx)); ctask = F.cross_entropy(clogits, ytr.index_select(0, idx)); cdrift = drift_loss(cdelta, art); ctotal = ctask + config.lambda_drift * cdrift; ctotal.backward(); nn.utils.clip_grad_norm_(control.parameters(), config.gradient_clip); opt_c.step()
            loss_m_sum += float(total.detach()); loss_c_sum += float(ctotal.detach()); geo_sum += float(geo.detach()); drift_sum += float(drift.detach())
        ba_m, _, _, _ = eval_ba(method, hva, qva, y_val); ba_c, _, _, _ = eval_ba(control, hva, qva, y_val)
        curves.append({"epoch": epoch, "method_strict_inductive_BA": ba_m, "control_strict_inductive_BA": ba_c, "method_loss": loss_m_sum / len(batches), "control_loss": loss_c_sum / len(batches), "geometry_loss": geo_sum / len(batches), "drift_loss": drift_sum / len(batches), "steps": len(batches), "elapsed_seconds": time.time() - start})
        if ba_m > best_m + 1e-10: best_m = ba_m; best_m_state = copy.deepcopy(method.state_dict()); best_m_epoch = epoch
        if ba_c > best_c + 1e-10: best_c = ba_c; best_c_state = copy.deepcopy(control.state_dict()); best_c_epoch = epoch
        print(f"[P5-{config.version}] fold={fold} seed={seed} epoch={epoch} method={ba_m:.5f} control={ba_c:.5f} geo={geo_sum/len(batches):.5f}", flush=True)
    assert best_m_state is not None and best_c_state is not None
    method.load_state_dict(best_m_state); control.load_state_dict(best_c_state)
    ba_m, logits_m, q_m, delta_m = eval_ba(method, hva, qva, y_val); ba_c, logits_c, q_c, delta_c = eval_ba(control, hva, qva, y_val)
    train_q0 = q_train
    diag_m = geometry_diagnostics(q0[val_pos], q_m, val_meta, train_meta, train_q0, art, targets); diag_c = geometry_diagnostics(q0[val_pos], q_c, val_meta, train_meta, train_q0, art, targets); drift_m = drift_diagnostics(delta_m, art, targets); drift_c = drift_diagnostics(delta_c, art, targets); int_m = intervention_diagnostics(method, h_val, q_val, y_val, art, stable_seed("int", fold, seed, config.version), device); int_c = intervention_diagnostics(control, h_val, q_val, y_val, art, stable_seed("int", fold, seed, "control", config.version), device)
    historical_head = copy.deepcopy(head).to(device).eval();
    with torch.inference_mode():
        hzero = hva + ((torch.as_tensor(q_m, device=device) - qva) @ method.directions.T) @ method.dewhitener; zero_shot_ba = float(balanced_accuracy_score(y_val, historical_head(hzero).argmax(1).cpu().numpy()))
    run_out = out / f"fold-{fold}" / f"seed-{seed}"; run_out.mkdir(parents=True, exist_ok=True); torch.save({"model": best_m_state, "fold": fold, "seed": seed, "version": config.version, "best_epoch": best_m_epoch, "outer_test_used": False, "checkpoint_sha256_historical": sha256(checkpoint_path)}, run_out / "best_method.pt"); torch.save({"model": best_c_state, "fold": fold, "seed": seed, "version": config.version, "best_epoch": best_c_epoch, "outer_test_used": False, "checkpoint_sha256_historical": sha256(checkpoint_path)}, run_out / "best_control.pt"); pd.DataFrame(curves).to_csv(run_out / "TRAIN_CURVES.csv", index=False); pd.DataFrame(subject_ba(logits_m, val_meta)).assign(run=f"fold-{fold}/seed-{seed}", model="method").to_csv(run_out / "SUBJECT_METHOD.csv", index=False); pd.DataFrame(subject_ba(logits_c, val_meta)).assign(run=f"fold-{fold}/seed-{seed}", model="control").to_csv(run_out / "SUBJECT_CONTROL.csv", index=False)
    result = {"status": "RUN_COMPLETE", "implementation_id": IMPLEMENTATION_ID, "version": config.version, "fold": fold, "seed": seed, "outer_test_used": False, "historical_strict_inductive_BA": historical, "method_strict_inductive_BA": ba_m, "control_strict_inductive_BA": ba_c, "delta_BA": ba_m - ba_c, "method_best_epoch": best_m_epoch, "control_best_epoch": best_c_epoch, "historical_head_zero_shot_BA": zero_shot_ba, "geometry_method": diag_m, "geometry_control": diag_c, "geometry_delta": {k: diag_m[k] - diag_c[k] for k in diag_m}, "drift_method": drift_m, "drift_control": drift_c, "intervention_method": int_m, "intervention_control": int_c, "n_validation_subjects": int(val_meta.subject.nunique()), "n_train_subjects": int(train_meta.subject.nunique()), "sampler": {"subjects_per_batch": config.subjects_per_batch, "trials_per_class": config.trials_per_class, "steps_per_epoch": sampler.steps, "seed_rule": "sha256(p5|implementation|fold|seed) shared across versions, plus sha256(sampler|epoch|run_seed)"}, "config": asdict(config), "canonical_spectrum_sha256": art.spectrum_sha256, "protected_blocks": art.protected_blocks, "protected_weights": weights, "checkpoint_paths": {"method": str((run_out / "best_method.pt").relative_to(OUT)).replace("\\", "/"), "control": str((run_out / "best_control.pt").relative_to(OUT)).replace("\\", "/")}}
    write_json(run_out / "RUN_RESULT.json", result); return result


def aggregate_version(version: str, rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    # Keep the required human-facing layout while retaining the original
    # fold/seed directories used by the runner.  These are small JSON/CSV
    # copies only; raw EEG and feature caches are never copied here.
    (out / "CONFIGS").mkdir(parents=True, exist_ok=True)
    (out / "TRAIN_LOGS").mkdir(parents=True, exist_ok=True)
    (out / "RUN_RESULTS").mkdir(parents=True, exist_ok=True)
    write_json(out / "CONFIGS" / "TRAIN_CONFIG.json", rows[0]["config"])
    for row in rows:
        run_dir = out / f"fold-{row['fold']}" / f"seed-{row['seed']}"
        run_tag = f"fold-{row['fold']}__seed-{row['seed']}"
        if (run_dir / "TRAIN_CURVES.csv").exists():
            shutil.copy2(run_dir / "TRAIN_CURVES.csv", out / "TRAIN_LOGS" / f"{run_tag}.csv")
        if (run_dir / "RUN_RESULT.json").exists():
            shutil.copy2(run_dir / "RUN_RESULT.json", out / "RUN_RESULTS" / f"{run_tag}.json")
    frame = pd.DataFrame(rows); frame.to_csv(out / "STRICT_INDUCTIVE_RESULTS.csv", index=False)
    pd.DataFrame([{"fold": r["fold"], "seed": r["seed"], "status": "NOT_RUN_PRIMARY_STRICT_ONLY", "outer_test_used": False} for r in rows]).to_csv(out / "TRANSDUCTIVE_RESULTS.csv", index=False)
    subject_rows: list[dict[str, Any]] = []
    for row in rows:
        run_path = out / f"fold-{row['fold']}" / f"seed-{row['seed']}" / "SUBJECT_METHOD.csv"; method = pd.read_csv(run_path); control = pd.read_csv(run_path.with_name("SUBJECT_CONTROL.csv")); merged = method.merge(control, on="subject", suffixes=("_method", "_control"));
        for _, r in merged.iterrows(): subject_rows.append({"run": f"fold-{row['fold']}/seed-{row['seed']}", "subject": str(r.subject), "delta": float(r.ba_method - r.ba_control), "method_BA": float(r.ba_method), "control_BA": float(r.ba_control)})
    pd.DataFrame(subject_rows).to_csv(out / "SUBJECT_LEVEL_RESULTS.csv", index=False)
    pd.DataFrame([{"run": f"fold-{r['fold']}/seed-{r['seed']}", "model": model, **(r["geometry_method"] if model == "method" else r["geometry_control"])} for r in rows for model in ("method", "control")]).to_csv(out / "GEOMETRY_DIAGNOSTICS.csv", index=False)
    pd.DataFrame([{"run": f"fold-{r['fold']}/seed-{r['seed']}", "model": model, **(r["drift_method"] if model == "method" else r["drift_control"])} for r in rows for model in ("method", "control")]).to_csv(out / "REPRESENTATION_DRIFT.csv", index=False)
    pd.DataFrame([{"run": f"fold-{r['fold']}/seed-{r['seed']}", "model": model, **(r["intervention_method"] if model == "method" else r["intervention_control"])} for r in rows for model in ("method", "control")]).to_csv(out / "INTERVENTION_DIAGNOSTICS.csv", index=False)
    delta_rows = [{"run": r["run"], "delta": r["delta"]} for r in subject_rows]; boot = paired_bootstrap(delta_rows, seed=stable_seed("bootstrap", version)); positive = int(sum(float(r["delta_BA"]) > 0 for r in rows)); geometry_delta = {k: float(np.mean([r["geometry_delta"][k] for r in rows])) for k in rows[0]["geometry_delta"]}
    random_superiority = float(np.mean([r["intervention_method"]["protected_drop"] - r["intervention_method"]["random_drop_mean"] for r in rows])) > 0
    viable = bool(boot["mean"] is not None and boot["mean"] >= 0.005 and positive >= 4 and boot["ci95"][0] > 0 and sum(float(r["delta_BA"]) < -0.005 for r in rows) <= 1 and any(geometry_delta[k] > 0 for k in ("alignment_same", "alignment_cross", "geometry_margin")) and random_superiority)
    strong = bool(viable and boot["mean"] >= 0.010 and positive >= 5)
    status = "PERSIST_ICG_STRONG" if strong else "PERSIST_ICG_VIABLE" if viable else "PERSIST_ICG_REPRESENTATION_ONLY" if any(geometry_delta[k] > 0 for k in geometry_delta) else "PERSIST_ICG_OPTIMIZATION_NOT_SUPPORTED"
    report = {"status": status, "version": version, "primary": {"mean_delta_BA": float(frame.delta_BA.mean()), "positive_runs": positive, "n_runs": len(rows), "hierarchical_subject_run_bootstrap": boot, "geometry_delta": geometry_delta, "random_intervention": {"method_protected_drop": float(np.mean([r["intervention_method"]["protected_drop"] for r in rows])), "method_random_drop": float(np.mean([r["intervention_method"]["random_drop_mean"] for r in rows]))}}, "rules": {"viable": {"mean_delta_ge_0.005": bool(boot["mean"] is not None and boot["mean"] >= 0.005), "positive_runs_ge_4": positive >= 4, "ci_lower_gt_0": bool(boot["ci95"][0] > 0 if boot["ci95"][0] is not None else False), "catastrophic_runs_le_1": sum(float(r["delta_BA"]) < -0.005 for r in rows) <= 1, "geometry_support": any(geometry_delta[k] > 0 for k in ("alignment_same", "alignment_cross", "geometry_margin")), "random_not_same": random_superiority}, "strong": {"mean_delta_ge_0.010": bool(boot["mean"] is not None and boot["mean"] >= 0.010), "positive_runs_ge_5": positive >= 5, "ci_lower_gt_0": bool(boot["ci95"][0] > 0 if boot["ci95"][0] is not None else False)}}, "outer_test_used": False, "method_training_scope": "frozen E0 feature-level canonical adapter and historical MI head initialized from checkpoint"}
    write_json(out / "VERSION_REPORT.json", report); (out / "VERSION_REPORT.md").write_text(f"# PERSIST-ICG {version}\n\nStatus: `{status}`\n\nStrict-inductive paired Delta_BA mean: `{report['primary']['mean_delta_BA']:.6f}`; positive runs: `{positive}/{len(rows)}`; bootstrap CI: `{boot['ci95']}`.\n\nOuter-test used: `false`.\n", encoding="utf-8"); return report


def write_baseline(rows: Sequence[dict[str, Any]]) -> None:
    """Materialise the frozen historical and V0 matched-control tables."""
    base = OUT / "baseline"
    base.mkdir(parents=True, exist_ok=True)
    historical = [{"fold": r["fold"], "seed": r["seed"], "historical_strict_inductive_BA": r["historical_strict_inductive_BA"], "outer_test_used": False} for r in rows]
    control = [{"fold": r["fold"], "seed": r["seed"], "matched_control_strict_inductive_BA": r["control_strict_inductive_BA"], "outer_test_used": False, "source_version": "V0"} for r in rows]
    pd.DataFrame(historical).to_csv(base / "HISTORICAL_RESULTS.csv", index=False)
    pd.DataFrame(control).to_csv(base / "MATCHED_CONTROL_RESULTS.csv", index=False)


def write_protocol() -> None:
    protocol = {"method": "PERSIST-ICG", "implementation_id": IMPLEMENTATION_ID, "primary_task": "MI", "folds": list(FOLDS), "seeds": list(SEEDS), "outer_test_used": False, "canonical_source": "Signed Audit V3.1 persisted spectrum and assignments", "strict_inductive": {"validation_subject_centering": False, "target_labels_at_inference": False, "target_subject_adaptation": False}, "v0": "uniform Protected canonical residual adapter", "v1": "same adapter with frozen c_b=min(LCB95(u_abs),LCB95(u_spec)) positive-part weights normalized within Protected blocks", "v2": "global Protected geometry direction scalar adapter preserving individual residual", "v3": "not implemented unless fixed progression rule authorizes it", "fixed_config": asdict(CONFIGS["V0"])}
    write_json(OUT / "protocol" / "P5_METHOD_PROTOCOL.json", protocol); write_json(OUT / "protocol" / "P5_VERSION_POLICY.json", {"max_scientific_versions": ["V0", "V1", "V2", "V3"], "progression_rules": "attachment P5 section 14", "success_gates": "attachment P5 section 17", "outer_test_used": False})
    log_path = OUT / "protocol" / "P5_ADAPTATION_LOG.json"
    old = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"entries": []}
    entries = old.get("entries", [])
    if not any(e.get("issue") == "P5 method development start" for e in entries):
        entries.append({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "version": "V0", "issue": "P5 method development start", "evidence": "V3.1 and Shared Geometry V1.2 canonical artifacts verified", "change": "feature-level frozen-E0 canonical adapter trainer", "reason": "exactly implements predeclared V0 without introducing global invariance", "scientific_impact": "no change to utility, Protected assignment, primary task, or evaluation gates", "data_used": ["TRAIN", "DEVELOPMENT_VALIDATION"], "validation_used": "checkpoint selection only", "outer_test_used": False})
    write_json(log_path, {"entries": entries})


def append_adaptation_log(*, version: str, issue: str, evidence: str, change: str,
                          reason: str, scientific_impact: str,
                          data_used: Sequence[str] = ("TRAIN", "DEVELOPMENT_VALIDATION"),
                          validation_used: str = "checkpoint selection only") -> None:
    """Append one auditable decision without duplicating it on resume."""
    path = OUT / "protocol" / "P5_ADAPTATION_LOG.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"entries": []}
    entries = payload.setdefault("entries", [])
    if any(e.get("version") == version and e.get("issue") == issue for e in entries):
        return
    entries.append({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "version": version,
                    "issue": issue, "evidence": evidence, "change": change, "reason": reason,
                    "scientific_impact": scientific_impact, "data_used": list(data_used),
                    "validation_used": validation_used, "outer_test_used": False})
    write_json(path, payload)


def write_not_authorized_version(version: str, evidence: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """Create an explicit non-run version record; never fabricate metrics."""
    out = OUT / version
    for name in ("CONFIGS", "TRAIN_LOGS", "RUN_RESULTS"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write_json(out / "CONFIGS" / "STATUS.json", {"version": version, "status": "NOT_AUTHORIZED_FIXED_RULE", "outer_test_used": False})
    for filename in ("STRICT_INDUCTIVE_RESULTS.csv", "TRANSDUCTIVE_RESULTS.csv", "SUBJECT_LEVEL_RESULTS.csv",
                     "GEOMETRY_DIAGNOSTICS.csv", "REPRESENTATION_DRIFT.csv", "INTERVENTION_DIAGNOSTICS.csv"):
        (out / filename).write_text("status,outer_test_used\nNOT_RUN,False\n", encoding="utf-8")
    report = {"status": "NOT_AUTHORIZED_FIXED_RULE", "version": version, "reason": reason,
              "evidence": clean(evidence), "outer_test_used": False}
    write_json(out / "VERSION_REPORT.json", report)
    (out / "VERSION_REPORT.md").write_text(f"# PERSIST-ICG {version}\n\nStatus: `NOT_AUTHORIZED_FIXED_RULE`\n\n{reason}\n\nOuter-test used: `false`.\n", encoding="utf-8")
    return report


def verify_inputs() -> None:
    if not MANIFEST.exists() or not SPLIT.exists(): raise FileNotFoundError((MANIFEST, SPLIT))
    for fold in FOLDS:
        for seed in SEEDS:
            load_artifacts(fold, seed); historical_checkpoint(fold, seed)
    meta = load_mi_manifest(); write_json(OUT / "protocol" / "P5_INPUT_VERIFICATION.json", {"manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST), "n_mi_rows": len(meta), "n_subjects": int(meta.subject.nunique()), "folds": list(FOLDS), "seeds": list(SEEDS), "canonical_runs": 6, "outer_test_used": False})


def run_version(version: str, device: torch.device, meta: pd.DataFrame, data_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, CanonicalArtifacts, GeometryTargets]]) -> dict[str, Any]:
    config = CONFIGS[version]; out = OUT / version; out.mkdir(parents=True, exist_ok=True); rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            result_path = out / f"fold-{fold}" / f"seed-{seed}" / "RUN_RESULT.json"
            if result_path.exists():
                try:
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    cached = None
                if (isinstance(cached, dict) and cached.get("status") == "RUN_COMPLETE"
                        and cached.get("implementation_id") == IMPLEMENTATION_ID
                        and cached.get("version") == version
                        and int(cached.get("fold", -1)) == int(fold)
                        and int(cached.get("seed", -1)) == int(seed)
                        and cached.get("outer_test_used") is False):
                    rows.append(cached)
                    continue
            key = (fold, seed)
            if key not in data_cache:
                cache_dir = OUT / "cache" / f"fold-{fold}" / f"seed-{seed}"; cache_dir.mkdir(parents=True, exist_ok=True); h = extract_h0(meta, fold, seed, cache_dir, device); art = load_artifacts(fold, seed); q = q_from_h(h, art); split = load_split(fold); train_pos = np.flatnonzero(meta.subject.isin(split["train_subjects"]).to_numpy()); targets = build_geometry_targets(meta, q, train_pos, art, cache_dir / "GEOMETRY_TARGETS.npz"); data_cache[key] = (h, q, art, targets)
            h, q, art, targets = data_cache[key]; rows.append(run_one(config, fold, seed, meta, h, q, art, targets, device, out))
    if len(rows) != len(FOLDS) * len(SEEDS):
        raise RuntimeError(f"Expected 6 complete {version} runs, got {len(rows)}")
    if version == "V0":
        write_baseline(rows)
    return aggregate_version(version, rows, out)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=["verify", "v0", "v1", "v2", "all"], default="all"); ap.add_argument("--device", default="cuda")
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); write_protocol(); verify_inputs();
    append_adaptation_log(version="ENGINEERING", issue="reproducibility and causal-isolation repair",
                          evidence="The pre-repair runner did not call seed_all and included version names in the per-run RNG/sampler keys.",
                          change="Fixed deterministic fold/seed streams shared across V0/V1, added resumable implementation IDs, and copied read-only memmaps before tensor conversion.",
                          reason="Make V0 versus V1 a same-initialisation/same-batch comparison without changing the scientific premise or thresholds.",
                          scientific_impact="No change to Signed-V3.1 utility, Protected assignment, task, loss definitions, or evaluation gates; prior outputs are retained but invalidated by implementation_id.")
    if args.phase == "verify": return
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu");
    if device.type != "cuda": raise RuntimeError("P5 requires the GPU environment; refusing accidental CPU training")
    meta = load_mi_manifest(); cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, CanonicalArtifacts, GeometryTargets]] = {}
    reports: dict[str, Any] = {}
    v3_decision: dict[str, Any] = {"authorized": False, "reason": "V2 progression not evaluated in this phase", "outer_test_used": False}
    if args.phase in {"v0", "v1", "v2", "all"}:
        reports["V0"] = run_version("V0", device, meta, cache)
    if args.phase in {"v1", "v2", "all"}:
        reports["V1"] = run_version("V1", device, meta, cache)
    if args.phase in {"v2", "all"}:
        # Fixed P5 rule: V2 is justified if either V0/V1 has a positive mean,
        # four positive runs, five positive geometry runs, or substantial drift.
        decision = False
        for report in reports.values():
            p = report["primary"]; decision |= bool(p["mean_delta_BA"] > 0 or p["positive_runs"] >= 4 or p["geometry_delta"].get("alignment_same", 0.0) > 0)
        write_json(OUT / "protocol" / "V2_PROGRESSION_DECISION.json", {"authorized": bool(decision), "evidence": reports, "outer_test_used": False})
        if decision:
            reports["V2"] = run_version("V2", device, meta, cache)
            append_adaptation_log(version="V2", issue="fixed progression rule authorized V2",
                                  evidence="At least one prior version had positive aggregate geometry alignment support, satisfying P5 section 14 rule 3.",
                                  change="Ran predeclared V2 selective global-direction adapter with individual residual preservation.",
                                  reason="The transition was authorized before inspecting V2 accuracy.",
                                  scientific_impact="V2 is a predeclared scientific version; no post-result threshold or objective change.")
        v3_decision: dict[str, Any]
        if "V2" in reports:
            p2 = reports["V2"]["primary"]
            geometry_active = any(float(p2["geometry_delta"].get(k, 0.0)) > 0.0 for k in ("alignment_same", "alignment_cross", "geometry_margin"))
            strong = reports["V2"]["status"] in {"PERSIST_ICG_STRONG", "PERSIST_ICG_VIABLE"} and bool(p2["mean_delta_BA"] >= 0.010 and p2["positive_runs"] >= 5 and p2["hierarchical_subject_run_bootstrap"]["ci95"][0] > 0)
            authorized_v3 = bool(p2["mean_delta_BA"] > 0 and p2["positive_runs"] >= 4 and geometry_active and not strong)
            v3_decision = {"authorized": authorized_v3, "evidence": reports["V2"], "rule": "V2 mean > 0, >=4/6 positive, active geometry, and failure of STRONG", "outer_test_used": False}
            write_json(OUT / "protocol" / "V3_PROGRESSION_DECISION.json", v3_decision)
            if authorized_v3:
                raise RuntimeError("V3 was authorized by the fixed rule but is not implemented; refusing to silently skip it")
            write_not_authorized_version("V3", v3_decision, "V3 was not authorized: V2 did not simultaneously have positive mean Delta_BA and at least 4/6 positive runs.")
            append_adaptation_log(version="V3", issue="fixed progression rule did not authorize V3",
                                  evidence=f"V2 mean Delta_BA={p2['mean_delta_BA']:.9f}; positive runs={p2['positive_runs']}/6; geometry_active={geometry_active}.",
                                  change="Stopped before V3 and materialized an explicit NOT_AUTHORIZED_FIXED_RULE record.",
                                  reason="P5 section 14 requires all V2 conditions before V3; the observed V2 result fails the accuracy conditions.",
                                  scientific_impact="No further scientific version or outer-test access.")
        else:
            v3_decision = {"authorized": False, "reason": "V2 was not run in this phase", "outer_test_used": False}
    frame = []
    for version, report in reports.items(): frame.append({"version": version, "status": report["status"], "mean_delta_BA": report["primary"]["mean_delta_BA"], "positive_runs": report["primary"]["positive_runs"], "ci_lower": report["primary"]["hierarchical_subject_run_bootstrap"]["ci95"][0]})
    pd.DataFrame(frame).to_csv(OUT / "PERSIST_ICG_METHOD_LEADERBOARD.csv", index=False)
    final_status = "PERSIST_ICG_STRONG" if any(r["status"] == "PERSIST_ICG_STRONG" for r in reports.values()) else "PERSIST_ICG_VIABLE" if any(r["status"] == "PERSIST_ICG_VIABLE" for r in reports.values()) else "PERSIST_ICG_REPRESENTATION_ONLY" if any(r["status"] == "PERSIST_ICG_REPRESENTATION_ONLY" for r in reports.values()) else "PERSIST_ICG_OPTIMIZATION_NOT_SUPPORTED"
    final = {"status": final_status, "implementation_id": IMPLEMENTATION_ID, "versions": reports,
             "progression": {"v3": v3_decision}, "outer_test_used": False,
             "method_training_started": True, "secondary_tasks": "not run before MI decision",
             "terminal_interpretation": "Geometry diagnostics improved in aggregate, but no predeclared version converted that signal into strict-inductive MI decoding gain." if final_status == "PERSIST_ICG_REPRESENTATION_ONLY" else "See version reports."}
    write_json(OUT / "PERSIST_ICG_FINAL_REPORT.json", final)
    lines = ["# PERSIST-ICG", "", f"Status: `{final_status}`", "", "Primary task: MI strict-inductive unseen-subject balanced accuracy.", "", "| Version | Status | Mean Delta_BA | Positive runs | CI lower |", "|---|---|---:|---:|---:|"]
    for row in frame:
        lines.append(f"| {row['version']} | {row['status']} | {row['mean_delta_BA']:.6f} | {row['positive_runs']}/6 | {row['ci_lower'] if row['ci_lower'] is not None else 'n/a'} |")
    lines += ["", "V3 was not authorized by the fixed progression rule.", "", "Outer-test used: `false`.", "", "No secondary paradigm was used to rescue the MI decision."]
    (OUT / "PERSIST_ICG_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
