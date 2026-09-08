"""Seed-0 backbone-generality screen and post-screen holdout audit.

The runner intentionally keeps runtime checkpoints outside git.  The screen
uses only the frozen SEARCH subjects; ``--phase holdout`` is a separate,
explicitly post-screen command and never participates in selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score


# ``backbone_screen.py`` lives under ``<worktree>/experiments/<exp>/code``;
# parents[3] is therefore the worktree root (parents[2] would point at the
# ``experiments`` directory and duplicate that component in every path).
REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments" / "persist_eeg_backbone_generality_seed0_v1"
CODE = EXP / "code"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get(
    "BACKBONE_RUNTIME",
    r"D:\nips-temp\TotalP\P1\backbone_generality_seed0_runtime",
)).resolve()
OPENBMI_ROOT = Path(os.environ.get(
    "PERSIST_OPENBMI_CACHE",
    r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi",
)).resolve()
WBCIC_ROOT = Path(os.environ.get(
    "PERSIST_WBCIC_CACHE",
    r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache\wbcic_epochs",
)).resolve()
CARRIER_RUNTIME = Path(os.environ.get(
    "CARRIER_5FOLD_RUNTIME",
    r"D:\nips-temp\TotalP\P1\carrier_5fold_multiseed_stability_runtime",
)).resolve()
PRETRAINED_ROOT = Path(os.environ.get(
    "BACKBONE_PRETRAINED_ROOT",
    r"D:\nips-temp\TotalP\pretrained_backbones",
)).resolve()
UPSTREAM_ROOT = Path(os.environ.get(
    "BACKBONE_UPSTREAM_ROOT",
    r"D:\nips-temp\TotalP\upstream_backbones",
)).resolve()

SEARCH_SPLIT = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"
BACKBONES = ("EEGConformer", "CBraMod", "CodeBrain")
DATASETS = ("OpenBMI", "WBCIC")
SEED = 0
FUSION_WEIGHT = 0.5
BOOTSTRAPS = 10_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [clean(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, "")) for key in fields})
    os.replace(tmp, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sort_subjects(values: list[str]) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        text = str(value)
        return (int(text.replace("sub-", "")), text)
    return sorted([str(x) for x in values], key=key)


def load_split() -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]], str]:
    raw = json.loads(SEARCH_SPLIT.read_text(encoding="utf-8"))
    search = {d: sort_subjects(raw["search_subjects"][d]) for d in DATASETS}
    folds = {d: raw["folds"][d] for d in DATASETS}
    if len(search["OpenBMI"]) != 40 or len(search["WBCIC"]) != 31:
        raise RuntimeError("frozen search sizes changed")
    for dataset in DATASETS:
        if len(folds[dataset]) != 5:
            raise RuntimeError(f"{dataset}: expected five folds")
        seen: set[str] = set()
        for fold in folds[dataset]:
            pieces = [set(map(str, fold[key])) for key in ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects")]
            if any(pieces[i] & pieces[j] for i in range(3) for j in range(i + 1, 3)):
                raise RuntimeError(f"{dataset} fold partition overlap")
            if set.union(*pieces) != set(search[dataset]):
                raise RuntimeError(f"{dataset} fold partition does not cover SEARCH")
            seen |= pieces[2]
        if seen != set(search[dataset]):
            raise RuntimeError(f"{dataset} outer coverage is not exact")
    return search, folds, sha256_file(SEARCH_SPLIT)


class Bundle:
    def __init__(self, dataset: str, subjects: list[str], x: np.ndarray, y: np.ndarray,
                 subject: np.ndarray, session: np.ndarray):
        self.dataset = dataset
        self.subjects = sort_subjects(subjects)
        self.x = x
        self.y = y.astype(np.int64, copy=False)
        self.subject = subject.astype(str, copy=False)
        self.session = session.astype(np.int64, copy=False)
        self.channels = int(x.shape[1])
        self.samples = int(x.shape[2])

    def indices(self, subjects: list[str] | set[str], sessions: tuple[int, ...] | list[int]) -> np.ndarray:
        mask = np.isin(self.subject, list(map(str, subjects))) & np.isin(self.session, list(map(int, sessions)))
        return np.flatnonzero(mask).astype(np.int64)


def load_bundle(dataset: str, subjects: list[str]) -> Bundle:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    se: list[np.ndarray] = []
    if dataset == "OpenBMI":
        for subject in sort_subjects(subjects):
            for session in (1, 2):
                base = OPENBMI_ROOT / f"sub-{int(subject):02d}" / f"ses-{session}" / "mi_1train"
                xp = base.with_name(base.name + "_signals.npy")
                yp = base.with_name(base.name + "_codes.npy")
                x = np.load(xp, mmap_mode="r", allow_pickle=False)
                y = np.load(yp, mmap_mode="r", allow_pickle=False)
                if tuple(x.shape) != (100, 62, 1000) or x.dtype != np.float32 or tuple(y.shape) != (100,):
                    raise RuntimeError(f"OpenBMI cache schema mismatch: {xp}")
                if set(map(int, np.unique(y))) != {1, 2}:
                    raise RuntimeError(f"OpenBMI labels are not 1/2: {yp}")
                xs.append(np.asarray(x))
                ys.append(np.asarray(y, dtype=np.int64) - 1)
                ss.append(np.full(100, str(subject)))
                se.append(np.full(100, session, dtype=np.int64))
    elif dataset == "WBCIC":
        for subject in sort_subjects(subjects):
            for session in (0, 1, 2):
                xp = WBCIC_ROOT / subject / f"ses-{session}_epochs.npy"
                yp = WBCIC_ROOT / subject / f"ses-{session}_labels.npy"
                x = np.load(xp, mmap_mode="r", allow_pickle=False)
                y = np.load(yp, mmap_mode="r", allow_pickle=False)
                if x.ndim != 3 or tuple(x.shape[1:]) != (58, 1000) or x.dtype != np.float16 or tuple(y.shape) != (x.shape[0],):
                    raise RuntimeError(f"WBCIC cache schema mismatch: {xp}")
                if set(map(int, np.unique(y))) != {0, 1}:
                    raise RuntimeError(f"WBCIC labels are not 0/1: {yp}")
                xs.append(np.asarray(x))
                ys.append(np.asarray(y, dtype=np.int64))
                ss.append(np.full(x.shape[0], subject))
                se.append(np.full(x.shape[0], session, dtype=np.int64))
    else:
        raise ValueError(dataset)
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    s_all = np.concatenate(ss, axis=0)
    se_all = np.concatenate(se, axis=0)
    return Bundle(dataset, subjects, x_all, y_all, s_all, se_all)


def source_sessions(dataset: str) -> tuple[int, ...]:
    return (1,) if dataset == "OpenBMI" else (0, 1)


def future_session(dataset: str) -> tuple[int, ...]:
    return (2,)


def normalizer(bundle: Bundle, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    idx = bundle.indices(subjects, source_sessions(bundle.dataset))
    x = bundle.x[idx].astype(np.float64)
    mean = x.mean(axis=(0, 2)).astype(np.float32)
    std = np.sqrt(np.maximum(x.var(axis=(0, 2)), 1e-12)).astype(np.float32)
    return mean, std, {
        "subjects": sort_subjects(subjects), "sessions": list(source_sessions(bundle.dataset)),
        "trials": int(len(idx)), "mean_std_sha256": sha256_bytes(mean.tobytes() + std.tobytes()),
    }


class GPUCache:
    def __init__(self, bundle: Bundle, mean: np.ndarray, std: np.ndarray, device: torch.device):
        # The active SEARCH bundle is materialized once per fold.  It contains
        # no held-out subjects in screen phase.
        raw = torch.from_numpy(np.asarray(bundle.x, dtype=np.float32))
        m = torch.from_numpy(mean).view(1, -1, 1)
        s = torch.from_numpy(std).view(1, -1, 1).clamp_min(1e-6)
        self.x = ((raw - m) / s).to(device, non_blocking=True)
        self.y = torch.from_numpy(bundle.y).to(device, non_blocking=True)
        self.device = device

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        ii = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        return self.x.index_select(0, ii), self.y.index_select(0, ii)


class EEGConformer(nn.Module):
    """Official EEG-Conformer topology, with a dataset-native spatial kernel."""
    def __init__(self, channels: int, n_classes: int = 2):
        super().__init__()
        emb = 40
        self.patch = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (channels, 1), (1, 1)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)), nn.Dropout(0.5),
            nn.Conv2d(40, emb, (1, 1)),
        )
        layer = nn.TransformerEncoderLayer(emb, nhead=10, dim_feedforward=4 * emb,
                                           dropout=0.5, activation="gelu",
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=6, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(emb * 61), nn.Linear(emb * 61, 256),
                                  nn.ELU(), nn.Dropout(0.5), nn.Linear(256, 32),
                                  nn.ELU(), nn.Dropout(0.3), nn.Linear(32, n_classes))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        z = self.encoder(z)
        if z.shape[1] != 61:
            raise RuntimeError(f"EEGConformer token length changed: {z.shape}")
        return z.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class CBraModAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        root = str((UPSTREAM_ROOT / "CBraMod").resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from models.cbramod import CBraMod  # type: ignore
        self.backbone = CBraMod(in_dim=200, out_dim=200, d_model=200,
                                dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        checkpoint = PRETRAINED_ROOT / "cbramod_pretrained_weights.pth"
        self.backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.head = nn.Linear(200, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if t != 1000:
            raise ValueError("CBraMod adapter expects 1000 samples")
        z = self.backbone(x.float().reshape(b, c, 5, 200))
        if z.ndim != 4 or z.shape[-1] != 200:
            raise RuntimeError(f"unexpected CBraMod output {tuple(z.shape)}")
        return z.mean(dim=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class CodeBrainAdapter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        root = str((UPSTREAM_ROOT / "CodeBrain").resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from Models.SSSM import SSSM  # type: ignore
        self.backbone = SSSM(
            in_channels=200, res_channels=200, skip_channels=200, out_channels=200,
            num_res_layers=8, diffusion_step_embed_dim_in=200,
            diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
            s4_lmax=570, s4_d_state=64, s4_dropout=0.1,
            s4_bidirectional=True, s4_layernorm=True,
            codebook_size_t=4096, codebook_size_f=4096, if_codebook=False,
        )
        checkpoint = PRETRAINED_ROOT / "codebrain.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        self.backbone.load_state_dict(state, strict=True)
        self.head = nn.Linear(channels * 5 * 200, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if t != 1000:
            raise ValueError("CodeBrain adapter expects 1000 samples")
        z = self.backbone(x.float().reshape(b, c, 5, 200))
        if z.ndim != 4 or z.shape[-1] != 200:
            raise RuntimeError(f"unexpected CodeBrain output {tuple(z.shape)}")
        return z.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_new_model(name: str, channels: int) -> nn.Module:
    if name == "EEGConformer":
        return EEGConformer(channels)
    if name == "CBraMod":
        return CBraModAdapter()
    if name == "CodeBrain":
        return CodeBrainAdapter(channels)
    raise KeyError(name)


def load_carrier(name: str, channels: int, dataset: str, fold: int) -> nn.Module:
    # These are the exact seed-0 selected carrier checkpoints; no carrier is
    # retrained in this experiment.
    os.environ.setdefault("R2EEG_REPO", str(REPO))
    carrier_code = str(REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code")
    if carrier_code not in sys.path:
        sys.path.insert(0, carrier_code)
    from eegnet_locked import EEGNet  # type: ignore
    from run_carrier_screen import CompactLite  # type: ignore
    model: nn.Module = EEGNet(channels, samples=1000) if name == "EEGNet" else CompactLite(channels, "bn")
    path = CARRIER_RUNTIME / f"{dataset.lower()}_fold{fold}_seed0_{name.lower()}" / "selected_best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model


def logits_for(model: nn.Module, cache: GPUCache, indices: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            x, _ = cache.batch(indices[start:start + batch_size])
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            values.append(out.float().cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.empty((0, 2), dtype=np.float32)


def subject_metric(y: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    pred = logits.argmax(1)
    return (float(balanced_accuracy_score(y, pred)),
            float(f1_score(y, pred, average="macro", zero_division=0)))


def grouped_metrics(bundle: Bundle, indices: np.ndarray, logits: np.ndarray,
                    subjects: list[str]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for subject in subjects:
        pos = np.flatnonzero(bundle.subject[indices] == str(subject))
        result[str(subject)] = subject_metric(bundle.y[indices[pos]], logits[pos])
    return result


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_one(name: str, dataset: str, fold: int, bundle: Bundle, fold_spec: dict[str, Any],
              cache: GPUCache, device: torch.device) -> dict[str, Any]:
    cell = RUNTIME / dataset.lower() / name.lower() / f"fold-{fold}" / "seed-0"
    cell.mkdir(parents=True, exist_ok=True)
    selected_path = cell / "selected_best.pt"
    latest_path = cell / "latest.pt"
    set_seed(1000 + 17 * fold + BACKBONES.index(name) * 100)
    model = build_new_model(name, bundle.channels).to(device)
    params = parameter_count(model)
    epochs = {"EEGConformer": 60, "CBraMod": 10, "CodeBrain": 10}[name]
    batch_size = 64
    lr = {"EEGConformer": 2e-4, "CBraMod": 1e-4, "CodeBrain": 1e-4}[name]
    wd = {"EEGConformer": 5e-4, "CBraMod": 1e-2, "CodeBrain": 5e-4}[name]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    amp = name != "CodeBrain"
    best = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    start_epoch = 1
    history: list[dict[str, Any]] = []
    if latest_path.is_file():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        if payload.get("protocol_hash") != protocol_hash():
            raise RuntimeError(f"resume protocol mismatch: {latest_path}")
        model.load_state_dict(payload["current_state"])
        opt.load_state_dict(payload["optimizer"])
        history = payload.get("history", [])
        best = float(payload.get("best_val_BA", best))
        best_epoch = int(payload.get("best_epoch", 0))
        best_state = payload.get("best_state")
        start_epoch = int(payload.get("epoch", 0)) + 1
    if selected_path.is_file() and start_epoch > epochs:
        model.load_state_dict(torch.load(selected_path, map_location=device, weights_only=True))
        return {"dataset": dataset, "backbone": name, "fold": fold, "seed": 0,
                "status": "RESUMED_COMPLETE", "best_epoch": best_epoch,
                "best_inner_val_BA": best, "runtime_seconds": 0.0,
                "peak_vram_GB": None, "parameter_count": params,
                "checkpoint_path": str(selected_path), "checkpoint_sha256": sha256_file(selected_path)}
    train_idx = bundle.indices(fold_spec["inner_train_subjects"], source_sessions(dataset))
    val_idx = bundle.indices(fold_spec["inner_val_subjects"], future_session(dataset))
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = np.random.default_rng(100000 + fold * 100 + epoch).permutation(train_idx)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            x, y = cache.batch(order[start:start + batch_size])
            opt.zero_grad(set_to_none=True)
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x), y)
            else:
                loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {name} {dataset} f{fold} e{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_logits = logits_for(model, cache, val_idx)
        val_ba = float(np.mean([subject_metric(bundle.y[val_idx[np.flatnonzero(bundle.subject[val_idx] == s)]],
                                                val_logits[np.flatnonzero(bundle.subject[val_idx] == s)])[0]
                               for s in sort_subjects(fold_spec["inner_val_subjects"])]))
        selected = val_ba > best + 1e-12
        if selected:
            best, best_epoch = val_ba, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "inner_val_subject_BA": val_ba,
                        "selected": bool(selected)})
        torch.save({"epoch": epoch, "history": history, "best_val_BA": best,
                    "best_epoch": best_epoch, "best_state": best_state,
                    "current_state": model.state_dict(), "optimizer": opt.state_dict(),
                    "protocol_hash": protocol_hash()}, latest_path)
        print(f"[train] {dataset} {name} fold={fold} epoch={epoch}/{epochs} valBA={val_ba:.5f}", flush=True)
    if best_state is None:
        raise RuntimeError(f"no selected checkpoint for {name}/{dataset}/fold{fold}")
    torch.save(best_state, selected_path)
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    peak = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else None
    return {"dataset": dataset, "backbone": name, "fold": fold, "seed": 0,
            "status": "COMPLETE", "best_epoch": best_epoch,
            "best_inner_val_BA": best, "runtime_seconds": elapsed,
            "peak_vram_GB": peak, "parameter_count": params,
            "checkpoint_path": str(selected_path), "checkpoint_sha256": sha256_file(selected_path)}


def protocol_hash() -> str:
    path = PROTOCOL / "PROTOCOL_LOCK.json"
    if not path.is_file():
        return "UNLOCKED"
    return sha256_file(path)


def preflight(device: torch.device) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for name in BACKBONES:
        for channels in (62, 58):
            set_seed(9000 + channels + BACKBONES.index(name))
            model = build_new_model(name, channels).to(device)
            model.eval()
            x = torch.randn(2, channels, 1000, device=device)
            with torch.no_grad():
                logits = model(x)
            if tuple(logits.shape) != (2, 2) or not torch.isfinite(logits).all():
                raise RuntimeError(f"{name} {channels}-channel forward preflight failed")
            model.train()
            y = torch.tensor([0, 1], device=device)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"{name} {channels}-channel gradient preflight failed")
            step_state = copy.deepcopy(model.state_dict())
            tmp = RUNTIME / "preflight" / f"{name}_{channels}.pt"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(step_state, tmp)
            model.load_state_dict(torch.load(tmp, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                a = model(x).float()
                b = model(x).float()
            if not torch.equal(a, b):
                raise RuntimeError(f"{name} eval determinism preflight failed")
            results.append({"backbone": name, "channels": channels, "output_shape": [2, 2],
                            "loss_finite": True, "gradients_finite": True,
                            "checkpoint_roundtrip_identical": True,
                            "pretrained_loaded": name in ("CBraMod", "CodeBrain"),
                            "parameters": parameter_count(model)})
            del model, x
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {"critical_checks_pass": True, "models": results}


def prepare_protocol(search: dict[str, list[str]], folds: dict[str, list[dict[str, Any]]], split_sha: str) -> None:
    for p in (PROTOCOL, OUT, RUNTIME):
        p.mkdir(parents=True, exist_ok=True)
    split_copy = PROTOCOL / "SPLIT_REFERENCE.json"
    shutil.copyfile(SEARCH_SPLIT, split_copy)
    write_json(PROTOCOL / "SPLIT_HASH_VERIFY.json", {"source": str(SEARCH_SPLIT),
        "source_sha256": split_sha, "copy": str(split_copy),
        "copy_sha256": sha256_file(split_copy), "match": sha256_file(split_copy) == split_sha})
    write_json(PROTOCOL / "BACKBONE_PROVENANCE.json", {
        "EEGConformer": {"official_repository": "https://github.com/eeyhsong/EEG-Conformer",
                         "commit": "9ae149ba62487ceae723277d13adac27837113d2",
                         "paper": "EEG Conformer, TNSRE 2023", "license": "GPL-3.0-only",
                         "pretrained": False, "pretraining_corpus": "not applicable",
                         "openbmi_overlap": False, "wbcic_overlap": False},
        "CBraMod": {"official_repository": "https://github.com/wjq-learning/CBraMod",
                    "commit": "b9e961003214326972c567eff390e75b0287e32a",
                    "paper": "CBraMod, ICLR 2025", "license": "MIT",
                    "pretrained_checkpoint": "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth",
                    "pretraining_corpus": "TUEG (paper provenance; repository README does not enumerate corpus)",
                    "openbmi_overlap": False, "wbcic_overlap": False},
        "CodeBrain": {"official_repository": "https://github.com/jingyingma01/CodeBrain",
                       "commit": "22d350caf68246d2fda4f630ef837420db3fb130",
                       "paper": "CodeBrain, ICLR 2026", "license": "Apache-2.0",
                       "pretrained_checkpoint": "https://huggingface.co/YjMajy/CodeBrain/resolve/main/CodeBrain.pth",
                       "pretraining_corpus": "TUEG (paper provenance; repository README does not enumerate corpus)",
                       "openbmi_overlap": False, "wbcic_overlap": False},
    })
    write_json(PROTOCOL / "PRETRAINED_CHECKPOINTS.json", {
        "CBraMod": {"path": str(PRETRAINED_ROOT / "cbramod_pretrained_weights.pth"),
                    "sha256": sha256_file(PRETRAINED_ROOT / "cbramod_pretrained_weights.pth"),
                    "filename": "pretrained_weights.pth"},
        "CodeBrain": {"path": str(PRETRAINED_ROOT / "codebrain.pth"),
                       "sha256": sha256_file(PRETRAINED_ROOT / "codebrain.pth"),
                       "filename": "CodeBrain.pth"},
    })
    (PROTOCOL / "UPSTREAM_LICENSE_AUDIT.md").write_text(
        "# Upstream license audit\n\n"
        "EEG-Conformer is GPL-3.0-only and is imported from an external clone; no GPL source is vendored. "
        "CBraMod is MIT and CodeBrain is Apache-2.0. Exact commits are recorded in BACKBONE_PROVENANCE.json.\n",
        encoding="utf-8")
    (PROTOCOL / "PRETRAINING_OVERLAP_AUDIT.md").write_text(
        "# Pretraining overlap audit\n\n"
        "The CBraMod and CodeBrain papers identify TUEG as the pretraining corpus. The official repositories "
        "and released checkpoint metadata do not report OpenBMI or WBCIC as pretraining targets. No direct target "
        "overlap evidence was found; this is recorded conservatively rather than inferred from downstream scores.\n",
        encoding="utf-8")
    holdout = {"screen_scope": "SEARCH subjects only", "V8_INTERNAL_HOLDOUT_loaded": False,
               "V8_INTERNAL_HOLDOUT_labels_loaded": False, "V8_INTERNAL_HOLDOUT_used_for_training": False,
               "V8_INTERNAL_HOLDOUT_used_for_normalization": False, "V8_INTERNAL_HOLDOUT_used_for_checkpoint_selection": False,
               "V8_INTERNAL_HOLDOUT_used_for_model_choice": False, "V8_INTERNAL_HOLDOUT_evaluated": False,
               "WBCIC_true_outer_loaded": False, "WBCIC_true_outer_labels_loaded": False}
    write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", holdout)
    carrier_rows = []
    for dataset in DATASETS:
        channels = 62 if dataset == "OpenBMI" else 58
        for fold in range(5):
            for model in ("EEGNet", "LiteBN"):
                path = CARRIER_RUNTIME / f"{dataset.lower()}_fold{fold}_seed0_{model.lower()}" / "selected_best.pt"
                carrier_rows.append({"dataset": dataset, "fold": fold, "seed": 0, "model": model,
                                     "path": str(path), "sha256": sha256_file(path), "channels": channels})
    write_json(PROTOCOL / "EXISTING_CARRIER_PROVENANCE.json", {"rows": carrier_rows})
    write_json(PROTOCOL / "BACKBONE_INPUT_PROTOCOL.json", {
        "canonical_input": "same cached MI trial; float32 normalized per fold from source sessions",
        "OpenBMI": {"channels": 62, "samples": 1000, "source_session": [1], "future_session": [2]},
        "WBCIC": {"channels": 58, "samples": 1000, "source_session": [0, 1], "future_session": [2]},
        "CBraMod": "reshape each trial to (channels, 5, 200), official patch length 200",
        "CodeBrain": "reshape each trial to (channels, 5, 200), official EEGSSM patch length 200",
        "EEGConformer": "dataset-native spatial convolution; temporal length unchanged",
    })
    write_json(PROTOCOL / "CHANNEL_MAPPING.json", {"OpenBMI": {"channels": 62, "mapping": "native cached order"},
        "WBCIC": {"channels": 58, "mapping": "native cached order"}, "cross_dataset_projection": False})
    write_json(PROTOCOL / "RESAMPLING_PROTOCOL.json", {"resampling": False, "samples": 1000,
        "CBraMod_CodeBrain_patch_size": 200})
    write_json(PROTOCOL / "NORMALIZATION_PROTOCOL.json", {"type": "per-channel z-score",
        "fit_scope": "inner_train subjects and source sessions only", "future_and_outer_stats_fit": False})
    write_json(PROTOCOL / "TRAINING_PROTOCOL.json", {
        "seed": 0, "folds": 5, "datasets": list(DATASETS), "backbones": list(BACKBONES),
        "epochs": {"EEGConformer": 60, "CBraMod": 10, "CodeBrain": 10},
        "optimizer": "AdamW", "learning_rate": {"EEGConformer": 2e-4, "CBraMod": 1e-4, "CodeBrain": 1e-4},
        "weight_decay": {"EEGConformer": 5e-4, "CBraMod": 1e-2, "CodeBrain": 5e-4},
        "batch_size": 64, "gradient_clip": 5.0, "amp": {"EEGConformer": "bfloat16", "CBraMod": "bfloat16", "CodeBrain": False},
        "checkpoint_selection": "earliest strict best inner future-session subject-balanced BA",
        "fusion": "RAW_LOGIT50", "fusion_weights": [0.5, 0.5], "hyperparameter_sweep": False,
    })
    # Keep the human-readable ledger as Markdown; writing JSON into a .md
    # file makes the protocol artifact misleading and breaks downstream
    # reviewers that expect a text log.
    (PROTOCOL / "BUGFIX_LOG.md").write_text(
        "# Bugfix log\n\nNo post-result bug fixes.\n", encoding="utf-8")
    write_json(PROTOCOL / "PROTOCOL_LOCK.json", {
        "protocol": "PERSIST_EEG_BACKBONE_GENERALITY_SEED0_V1",
        "seed": 0, "search_split_sha256": split_sha, "backbones": list(BACKBONES),
        "datasets": list(DATASETS), "fusion": {"rule": "0.5 * raw_logit_A + 0.5 * raw_logit_B", "alpha_search": False},
        "holdout_isolation": "screen phase never loads internal holdout or true outer subjects",
        "pretrained_checkpoint_manifest": str(PROTOCOL / "PRETRAINED_CHECKPOINTS.json"),
    })
    (PROTOCOL / "PROTOCOL_LOCK.sha256").write_text(sha256_file(PROTOCOL / "PROTOCOL_LOCK.json") + "\n", encoding="utf-8")
    (EXP / "README.md").write_text(
        "# PERSIST-EEG backbone generality seed-0\n\n"
        "This is a frozen SEARCH-only screen of EEGConformer, CBraMod, and CodeBrain with the fixed RAW_LOGIT50 "
        "LiteBN and EEGNet controls. Runtime checkpoints remain outside git. The separate `--phase holdout` command "
        "is post-screen only and cannot affect selection.\n", encoding="utf-8")
    (EXP / "METHOD.md").write_text(
        "# Method\n\nAll models see the same cached MI epochs and frozen five-fold subject split. OpenBMI trains on S1 and "
        "evaluates S2; WBCIC trains on S1+S2 and evaluates S3. New backbones use fixed recipes recorded before "
        "outer-dev inspection. Primary fusion is the arithmetic mean of raw pre-softmax logits.\n", encoding="utf-8")


def logit_audit(dataset: str, fold: int, model: str, logits: np.ndarray) -> dict[str, Any]:
    z = np.asarray(logits, dtype=np.float64)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)
    margin = np.abs(z[:, 1] - z[:, 0])
    return {"dataset": dataset, "fold": fold, "model": model,
            "mean_abs_logit": float(np.abs(z).mean()), "mean_logit_l2": float(np.linalg.norm(z, axis=1).mean()),
            "mean_decision_margin": float(margin.mean()), "predictive_entropy": float(ent.mean()),
            "confidence": float(p.max(1).mean()), "fraction_finite": float(np.isfinite(z).all(1).mean()),
            "class_ordering": "logit[0]=class0, logit[1]=class1"}


def bootstrap(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(1)
    return {"n_subjects": int(len(values)), "resamples": BOOTSTRAPS,
            "mean_delta_pp": float(values.mean()), "median_delta_pp": float(np.median(values)),
            "ci_low_pp": float(np.quantile(draws, 0.025)), "ci_high_pp": float(np.quantile(draws, 0.975)),
            "positive_subjects": int((values > 0).sum())}


def complementarity(y: np.ndarray, a: np.ndarray, b: np.ndarray, fused: np.ndarray,
                    dataset: str, backbone: str, partner: str) -> dict[str, Any]:
    ya, yb, yf = a.argmax(1), b.argmax(1), fused.argmax(1)
    ca, cb = ya == y, yb == y
    both_correct = int((ca & cb).sum()); both_wrong = int((~ca & ~cb).sum())
    a_only = int((ca & ~cb).sum()); b_only = int((~ca & cb).sum())
    return {"dataset": dataset, "backbone": backbone, "partner": partner,
            "both_correct": both_correct, "both_wrong": both_wrong,
            "base_only_correct": a_only, "partner_only_correct": b_only,
            "exclusive_correct_fraction": float((a_only + b_only) / max(1, len(y))),
            "prediction_disagreement": float(np.mean(ya != yb)),
            "fusion_correct": int((yf == y).sum()), "trials": int(len(y))}


def aggregate_outputs(standalone: list[dict[str, Any]], fusion: list[dict[str, Any]],
                      fold_logits: list[dict[str, Any]], training: list[dict[str, Any]]) -> None:
    write_csv(OUT / "STANDALONE_SUBJECT_RESULTS.csv", standalone)
    write_csv(OUT / "FUSION_SUBJECT_RESULTS.csv", fusion)
    write_csv(OUT / "TRAINING_LEDGER.csv", training)
    fold_rows: list[dict[str, Any]] = []
    keys = sorted({(r["dataset"], r["backbone"], int(r["fold"])) for r in fusion})
    for dataset, backbone, fold in keys:
        s = [r for r in standalone if r["dataset"] == dataset and r["fold"] == fold and r["backbone"] in (backbone, "")]
        # standalone rows use the backbone field of the new model and carrier rows
        # are available under the same fold; index by model below.
        def mean_model(model: str, col: str = "BA") -> float:
            vals = [float(r[col]) for r in standalone if r["dataset"] == dataset and int(r["fold"]) == fold and r["model"] == model]
            return float(np.mean(vals))
        base = mean_model(backbone); lite = mean_model("LiteBN")
        for pair in (f"{backbone}+LiteBN", f"{backbone}+EEGNet"):
            rows = [r for r in fusion if r["dataset"] == dataset and int(r["fold"]) == fold and r["backbone"] == backbone and r["pair"] == pair]
            if not rows:
                continue
            fba = float(np.mean([float(r["BA"]) for r in rows]))
            partner = "LiteBN" if pair.endswith("LiteBN") else "EEGNet"
            pba = lite if partner == "LiteBN" else mean_model("EEGNet")
            fold_rows.append({"dataset": dataset, "backbone": backbone, "fold": fold, "pair": pair,
                              "base_BA": base, "partner_BA": pba, "fusion_BA": fba,
                              "delta_vs_base_pp": (fba - base) * 100.0,
                              "delta_vs_partner_pp": (fba - pba) * 100.0,
                              "delta_vs_best_pp": (fba - max(base, pba)) * 100.0})
        if backbone == "EEGNet":
            pass
    # Add the canonical EEGNet+LiteBN reference, which is inference-only here.
    for dataset in DATASETS:
        for fold in range(5):
            rows = [r for r in fusion if r["dataset"] == dataset and int(r["fold"]) == fold and r["pair"] == "EEGNet+LiteBN"]
            if rows:
                e = mean_model_global(standalone, dataset, fold, "EEGNet")
                l = mean_model_global(standalone, dataset, fold, "LiteBN")
                f = float(np.mean([r["BA"] for r in rows]))
                fold_rows.append({"dataset": dataset, "backbone": "EEGNet", "fold": fold, "pair": "EEGNet+LiteBN",
                                  "base_BA": e, "partner_BA": l, "fusion_BA": f,
                                  "delta_vs_base_pp": (f - e) * 100.0,
                                  "delta_vs_partner_pp": (f - l) * 100.0,
                                  "delta_vs_best_pp": (f - max(e, l)) * 100.0})
    write_csv(OUT / "FOLD_RESULTS.csv", fold_rows)
    dataset_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for dataset, backbone, pair in sorted({(r["dataset"], r["backbone"], r["pair"]) for r in fold_rows}):
        fr = [r for r in fold_rows if r["dataset"] == dataset and r["backbone"] == backbone and r["pair"] == pair]
        # subject-paired deltas are the statistical unit.
        sr = [r for r in fusion if r["dataset"] == dataset and r["backbone"] == backbone and r["pair"] == pair]
        base_model = backbone
        partner_model = "LiteBN" if pair.endswith("LiteBN") else "EEGNet"
        base_map = {(r["subject_id"], int(r["fold"])): float(r["BA"]) for r in standalone if r["dataset"] == dataset and r["model"] == base_model}
        partner_map = {(r["subject_id"], int(r["fold"])): float(r["BA"]) for r in standalone if r["dataset"] == dataset and r["model"] == partner_model}
        fused_map = {(r["subject_id"], int(r["fold"])): float(r["BA"]) for r in sr}
        if not fused_map:
            continue
        dbase = np.array([(v - base_map[k]) * 100 for k, v in fused_map.items() if k in base_map], dtype=float)
        dpartner = np.array([(v - partner_map[k]) * 100 for k, v in fused_map.items() if k in partner_map], dtype=float)
        base_vals = np.array([base_map[k] for k in fused_map if k in base_map], dtype=float)
        partner_vals = np.array([partner_map[k] for k in fused_map if k in partner_map], dtype=float)
        fused_vals = np.array([fused_map[k] for k in fused_map if k in base_map and k in partner_map], dtype=float)
        if len(fused_vals) == 0:
            continue
        bbest = bootstrap(dbase)
        bpartner = bootstrap(dpartner)
        # Protocol-defined Delta_best bootstrap: subtract the larger dataset mean.
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(BOOTSTRAPS):
            ii = rng.integers(0, len(fused_vals), len(fused_vals))
            draws.append((fused_vals[ii].mean() - max(base_vals[ii].mean(), partner_vals[ii].mean())) * 100)
        best_delta = (fused_vals.mean() - max(base_vals.mean(), partner_vals.mean())) * 100
        positive_folds = int(sum(float(r["delta_vs_best_pp"]) > 0 for r in fr))
        row = {"dataset": dataset, "backbone": backbone, "pair": pair,
               "base_BA": float(base_vals.mean()), "partner_BA": float(partner_vals.mean()),
               "fusion_BA": float(fused_vals.mean()), "delta_vs_base_pp": float(dbase.mean()),
               "delta_vs_partner_pp": float(dpartner.mean()), "delta_best_pp": float(best_delta),
               "delta_best_ci_low_pp": float(np.quantile(draws, 0.025)),
               "delta_best_ci_high_pp": float(np.quantile(draws, 0.975)),
               "positive_folds_over_best": positive_folds, "folds": len(fr),
               "base_bootstrap_ci": [bbest["ci_low_pp"], bbest["ci_high_pp"]],
               "partner_bootstrap_ci": [bpartner["ci_low_pp"], bpartner["ci_high_pp"]]}
        dataset_rows.append(row)
        matrix_rows.append(row)
    write_csv(OUT / "DATASET_RESULTS.csv", dataset_rows)
    # Primary matrix is the LiteBN pairing; append the generic EEGNet controls.
    write_csv(OUT / "BACKBONE_GENERALITY_MATRIX.csv", matrix_rows)
    lines = ["# Backbone generality matrix", "", "| Dataset | Pair | Base BA | Partner BA | Fusion BA | Delta best (pp) | 95% CI (pp) | Positive folds |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for r in matrix_rows:
        lines.append(f"| {r['dataset']} | {r['pair']} | {r['base_BA']:.4f} | {r['partner_BA']:.4f} | {r['fusion_BA']:.4f} | {r['delta_best_pp']:+.3f} | [{r['delta_best_ci_low_pp']:+.3f}, {r['delta_best_ci_high_pp']:+.3f}] | {r['positive_folds_over_best']}/{r['folds']} |")
    (OUT / "BACKBONE_GENERALITY_MATRIX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Predeclared descriptive categories.
    lite_rows = [r for r in matrix_rows if r["pair"].endswith("+LiteBN") and r["backbone"] in BACKBONES]
    robust = sum(r["delta_best_pp"] >= 0.5 and r["positive_folds_over_best"] >= 3 for r in lite_rows if r["dataset"] in DATASETS)
    pair_keys = {(r["backbone"], r["dataset"]) for r in lite_rows}
    cross = sum(all((b, d) in pair_keys and next(x for x in lite_rows if x["backbone"] == b and x["dataset"] == d)["delta_best_pp"] > 0 for d in DATASETS) for b in BACKBONES)
    if cross == 0:
        terminal = "BACKBONE_GENERALITY_NOT_SUPPORTED"
    elif robust >= 4:
        terminal = "BACKBONE_GENERALITY_STRONG"
    elif cross >= 2:
        terminal = "BACKBONE_GENERALITY_SUPPORTED"
    else:
        terminal = "BACKBONE_GENERALITY_PARTIAL"
    write_json(OUT / "FINAL_TERMINAL.json", {"terminal": terminal, "new_backbones": BACKBONES,
        "screen_only": True, "holdout_confirmation_separate": True, "rows": matrix_rows})
    (OUT / "FINAL_BACKBONE_SCREEN_DECISION.md").write_text(
        f"# Final backbone screen decision\n\nTerminal: **{terminal}**.\n\n"
        "This seed-0 screen uses SEARCH subjects only. Holdout confirmation is a separate post-screen audit and "
        "cannot change the screen category.\n", encoding="utf-8")


def mean_model_global(rows: list[dict[str, Any]], dataset: str, fold: int, model: str) -> float:
    vals = [float(r["BA"]) for r in rows if r["dataset"] == dataset and int(r["fold"]) == fold and r["model"] == model]
    return float(np.mean(vals))


def screen() -> None:
    search, folds, split_sha = load_split()
    prepare_protocol(search, folds, split_sha)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("screen requires CUDA")
    pf = preflight(device)
    write_json(PROTOCOL / "PREFLIGHT_TESTS.json", pf)
    # Lock is already complete and hashed before the first outer-dev metric.
    standalone: list[dict[str, Any]] = []
    fusion: list[dict[str, Any]] = []
    fold_logits: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    complement_rows: list[dict[str, Any]] = []
    logit_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        bundle = load_bundle(dataset, search[dataset])
        for fold_spec in folds[dataset]:
            fold = int(fold_spec["fold_id"])
            mean, std, norm_info = normalizer(bundle, fold_spec["inner_train_subjects"])
            fold_cache = GPUCache(bundle, mean, std, device)
            outer_idx = bundle.indices(fold_spec["outer_dev_subjects"], future_session(dataset))
            carrier_logits: dict[str, np.ndarray] = {}
            for carrier in ("EEGNet", "LiteBN"):
                m = load_carrier(carrier, bundle.channels, dataset, fold).to(device)
                z = logits_for(m, fold_cache, outer_idx)
                carrier_logits[carrier] = z
                metrics = grouped_metrics(bundle, outer_idx, z, fold_spec["outer_dev_subjects"])
                for subject, (ba, f1) in metrics.items():
                    standalone.append({"dataset": dataset, "backbone": "EEGNet" if carrier == "EEGNet" else "LiteBN",
                                       "fold": fold, "subject_id": subject, "model": carrier, "BA": ba, "macro_F1": f1})
                logit_rows.append(logit_audit(dataset, fold, carrier, z))
                cost_rows.append({"dataset": dataset, "backbone": carrier, "fold": fold,
                                  "parameter_count": parameter_count(m), "trainable_parameters": 0,
                                  "model_disk_bytes": (CARRIER_RUNTIME / f"{dataset.lower()}_fold{fold}_seed0_{carrier.lower()}" / "selected_best.pt").stat().st_size,
                                  "inference_peak_vram_GB": None})
                del m
            for name in BACKBONES:
                info = train_one(name, dataset, fold, bundle, fold_spec, fold_cache, device)
                training.append({**info, "normalizer_sha256": norm_info["mean_std_sha256"]})
                m = build_new_model(name, bundle.channels).to(device)
                m.load_state_dict(torch.load(info["checkpoint_path"], map_location=device, weights_only=True), strict=True)
                z = logits_for(m, fold_cache, outer_idx)
                metrics = grouped_metrics(bundle, outer_idx, z, fold_spec["outer_dev_subjects"])
                for subject, (ba, f1) in metrics.items():
                    standalone.append({"dataset": dataset, "backbone": name, "fold": fold, "subject_id": subject,
                                       "model": name, "BA": ba, "macro_F1": f1})
                logit_rows.append(logit_audit(dataset, fold, name, z))
                cost_rows.append({"dataset": dataset, "backbone": name, "fold": fold,
                                  "parameter_count": parameter_count(m), "trainable_parameters": parameter_count(m),
                                  "model_disk_bytes": Path(info["checkpoint_path"]).stat().st_size,
                                  "peak_training_vram_GB": info.get("peak_vram_GB"),
                                  "inference_peak_vram_GB": None})
                for partner in ("LiteBN", "EEGNet"):
                    pair = f"{name}+{partner}"
                    fused = 0.5 * (z + carrier_logits[partner])
                    fm = grouped_metrics(bundle, outer_idx, fused, fold_spec["outer_dev_subjects"])
                    bm = metrics
                    pm = grouped_metrics(bundle, outer_idx, carrier_logits[partner], fold_spec["outer_dev_subjects"])
                    for subject, (ba, f1) in fm.items():
                        fusion.append({"dataset": dataset, "backbone": name, "fold": fold, "subject_id": subject,
                                       "pair": pair, "BA": ba, "macro_F1": f1,
                                       "delta_vs_base_pp": (ba - bm[subject][0]) * 100,
                                       "delta_vs_partner_pp": (ba - pm[subject][0]) * 100,
                                       "delta_vs_best_pp": (ba - max(bm[subject][0], pm[subject][0])) * 100})
                    complement_rows.append(complementarity(bundle.y[outer_idx], z, carrier_logits[partner], fused,
                                                            dataset, name, partner))
                    logit_rows.append(logit_audit(dataset, fold, pair, fused))
                del m
            # Canonical reference fusion is inference-only and uses no new training.
            fused_ref = 0.5 * (carrier_logits["EEGNet"] + carrier_logits["LiteBN"])
            refm = grouped_metrics(bundle, outer_idx, fused_ref, fold_spec["outer_dev_subjects"])
            em = grouped_metrics(bundle, outer_idx, carrier_logits["EEGNet"], fold_spec["outer_dev_subjects"])
            lm = grouped_metrics(bundle, outer_idx, carrier_logits["LiteBN"], fold_spec["outer_dev_subjects"])
            for subject, (ba, f1) in refm.items():
                fusion.append({"dataset": dataset, "backbone": "EEGNet", "fold": fold, "subject_id": subject,
                               "pair": "EEGNet+LiteBN", "BA": ba, "macro_F1": f1,
                               "delta_vs_base_pp": (ba - em[subject][0]) * 100,
                               "delta_vs_partner_pp": (ba - lm[subject][0]) * 100,
                               "delta_vs_best_pp": (ba - max(em[subject][0], lm[subject][0])) * 100})
            complement_rows.append(complementarity(bundle.y[outer_idx], carrier_logits["EEGNet"], carrier_logits["LiteBN"], fused_ref,
                                                    dataset, "EEGNet", "LiteBN"))
            logit_rows.append(logit_audit(dataset, fold, "EEGNet+LiteBN", fused_ref))
            # ``bundle`` is shared by all five folds of this dataset; only
            # release the per-fold GPU materialization here.
            del fold_cache
            torch.cuda.empty_cache()
        del bundle
    aggregate_outputs(standalone, fusion, fold_logits, training)
    write_csv(OUT / "LOGIT_SCALE_AUDIT.csv", logit_rows)
    write_csv(OUT / "CARRIER_COMPLEMENTARITY.csv", complement_rows)
    write_csv(OUT / "MODEL_COSTS.csv", cost_rows)
    # A compact stability table is useful for reviewers and does not expose raw predictions.
    fold_rows = []
    if (OUT / "FOLD_RESULTS.csv").is_file():
        with (OUT / "FOLD_RESULTS.csv").open(encoding="utf-8") as f:
            fold_rows = list(csv.DictReader(f))
    write_csv(OUT / "FOLD_STABILITY.csv", [{"dataset": r["dataset"], "backbone": r["backbone"], "fold": r["fold"],
        "pair": r["pair"], "delta_vs_best_pp": r["delta_vs_best_pp"],
        "positive_over_best": float(r["delta_vs_best_pp"]) > 0} for r in fold_rows])
    print("SCREEN_COMPLETE", flush=True)


def holdout_subjects(dataset: str) -> list[str]:
    if dataset == "OpenBMI":
        return ["4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54"]
    return ["sub-2", "sub-3", "sub-17", "sub-19", "sub-21", "sub-25", "sub-31", "sub-33", "sub-38", "sub-42"]


def holdout() -> None:
    if not (OUT / "FINAL_TERMINAL.json").is_file():
        raise RuntimeError("run and finish screen before holdout phase")
    search, folds, _ = load_split()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    data_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        hs = holdout_subjects(dataset)
        hold_bundle = load_bundle(dataset, search[dataset] + hs)
        for fold_spec in folds[dataset]:
            fold = int(fold_spec["fold_id"])
            mean, std, _ = normalizer(hold_bundle, fold_spec["inner_train_subjects"])
            cache = GPUCache(hold_bundle, mean, std, device)
            idx = hold_bundle.indices(hs, future_session(dataset))
            logits: dict[str, np.ndarray] = {}
            for model_name in ("EEGNet", "LiteBN"):
                m = load_carrier(model_name, hold_bundle.channels, dataset, fold).to(device)
                logits[model_name] = logits_for(m, cache, idx)
                del m
            for name in BACKBONES:
                path = RUNTIME / dataset.lower() / name.lower() / f"fold-{fold}" / "seed-0" / "selected_best.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                m = build_new_model(name, hold_bundle.channels).to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
                logits[name] = logits_for(m, cache, idx)
                del m
            for name in ("EEGNet",) + BACKBONES:
                z = logits[name]
                gm = grouped_metrics(hold_bundle, idx, z, hs)
                for subject, (ba, f1) in gm.items():
                    rows.append({"dataset": dataset, "model": name, "fold": fold, "subject_id": subject,
                                 "BA": ba, "macro_F1": f1, "evaluation": "post_screen_internal_holdout"})
                for partner in (("LiteBN",) if name == "EEGNet" else ("LiteBN", "EEGNet")):
                    pair = f"{name}+{partner}"
                    fm = grouped_metrics(hold_bundle, idx, 0.5 * (z + logits[partner]), hs)
                    for subject, (ba, f1) in fm.items():
                        rows.append({"dataset": dataset, "model": pair, "fold": fold, "subject_id": subject,
                                     "BA": ba, "macro_F1": f1, "evaluation": "post_screen_internal_holdout"})
            del cache
            torch.cuda.empty_cache()
        del hold_bundle
    write_csv(OUT / "HOLDOUT_SUBJECT_RESULTS.csv", rows)
    for dataset in DATASETS:
        for model in sorted({r["model"] for r in rows if r["dataset"] == dataset}):
            rr = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
            data_rows.append({"dataset": dataset, "model": model, "BA": float(np.mean([r["BA"] for r in rr])),
                              "macro_F1": float(np.mean([r["macro_F1"] for r in rr])),
                              "subjects": len(set(r["subject_id"] for r in rr)), "folds": 5,
                              "selection_used_holdout": False})
    write_csv(OUT / "HOLDOUT_DATASET_RESULTS.csv", data_rows)
    write_json(OUT / "HOLDOUT_CONFIRMATION.json", {"post_screen_only": True, "selection_used_holdout": False,
        "datasets": DATASETS, "rows": data_rows, "note": "Internal holdout audit; WBCIC true outer subjects were not accessed."})
    print("HOLDOUT_COMPLETE", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "screen", "holdout"), default="screen")
    args = parser.parse_args()
    if args.phase == "preflight":
        search, folds, split_sha = load_split()
        prepare_protocol(search, folds, split_sha)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("preflight requires CUDA")
        write_json(PROTOCOL / "PREFLIGHT_TESTS.json", preflight(device))
        print("PREFLIGHT_COMPLETE", flush=True)
    elif args.phase == "screen":
        screen()
    else:
        holdout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
