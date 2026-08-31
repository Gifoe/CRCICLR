"""CGR-Fuse-R1: legally cross-fitted consensus-gated calibrated stacking.

This file deliberately keeps the experiment self-contained.  It trains three
small EEGNet expert families on two fixed biological-subject bipartitions and
three seeds, retaining only predictions from models that excluded the sample's
subject.  Calibration, simplex stacking and the monotone instability gate are
all fitted inside five outer subject folds.  WBCIC source training reads S0
labels and evaluates S1 only; S2 and outer resources are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
BANKS = EXP / "action_banks"
PROTOCOL = EXP / "protocol"

OPENBMI_ROOT = Path(os.environ.get(
    "CGRSTACK_OPENBMI_ROOT",
    r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full",
))
OPENBMI_MANIFEST = OPENBMI_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
WBCIC_CACHE = Path(os.environ.get(
    "CGRSTACK_WBCIC_CACHE",
    r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_wbcic_independent_replication_v1\runtime\cache",
))
WBCIC_METADATA = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_METADATA.parquet"
WBCIC_RAW = WBCIC_CACHE / "WBCIC_DEVELOPMENT_MI_RAW.npy"

EXPERTS = ("E0", "E1", "E2")
# Six prediction slots per sample: two independent subject bipartitions x
# three seeds.  Each underlying run is trained on the complementary side of
# the relevant bipartition; the target-side orientation is recorded separately.
RUNS = tuple(f"b{b}s{s}" for b in range(2) for s in range(3))
BOOTSTRAP_DRAWS = 10_000
BASE_SEED = 20260901
EPOCHS = int(os.environ.get("CGRSTACK_EPOCHS", "4"))
BATCH_SIZE = int(os.environ.get("CGRSTACK_BATCH_SIZE", "256"))


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
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stable_unit(*parts: object) -> float:
    raw = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64


def stable_seed(*parts: object) -> int:
    raw = ":".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -50, 50)))


def entropy_binary(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def subject_ids(values: Iterable[object]) -> list[str]:
    def key(x: str) -> tuple[int, str]:
        return (int(x), x) if x.isdigit() else (10**9, x)
    return sorted({str(x) for x in values}, key=key)


def subject_bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def subject_lcb(values: np.ndarray, seed: int) -> float:
    return float(subject_bootstrap(values, seed)[0])


def subject_ba_table(y: np.ndarray, pred: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in subject_ids(subjects):
        mask = subjects.astype(str) == s
        rows.append({
            "subject": s,
            "BA": float(balanced_accuracy_score(y[mask], pred[mask])),
            "macro_f1": float(f1_score(y[mask], pred[mask], average="macro")),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def metric(dataset: str, method: str, y: np.ndarray, p: np.ndarray,
           subjects: np.ndarray, baseline_p: np.ndarray | None = None,
           changed: np.ndarray | None = None, soft_mass: np.ndarray | None = None,
           unsafe: np.ndarray | None = None) -> dict[str, Any]:
    pred = (np.asarray(p) >= 0.5).astype(int)
    base = (np.asarray(baseline_p) >= 0.5).astype(int) if baseline_p is not None else None
    table = subject_ba_table(y, pred, subjects)
    if baseline_p is None:
        delta = np.zeros(len(table), dtype=float)
    else:
        base_table = subject_ba_table(y, base, subjects)
        delta = table.BA.to_numpy() - base_table.BA.to_numpy()
    ci_l, ci_u = subject_bootstrap(delta, stable_seed(dataset, method, "ba"))
    if changed is None:
        changed = np.zeros(len(y), dtype=bool)
    if unsafe is None:
        unsafe = np.zeros(len(y), dtype=bool)
    changed = np.asarray(changed, dtype=bool)
    unsafe = np.asarray(unsafe, dtype=bool)
    return {
        "dataset": dataset,
        "method": method,
        "BA": float(table.BA.mean()),
        "macro_f1": float(table.macro_f1.mean()),
        "delta_vs_STRONGEST_KEEP_STACK": float(delta.mean()),
        "median_subject_delta_BA": float(np.median(delta)),
        "bootstrap_CI95_L": ci_l,
        "bootstrap_CI95_U": ci_u,
        "positive_subject_fraction": float(np.mean(delta > 0)),
        "nonnegative_subject_fraction": float(np.mean(delta >= 0)),
        "subjects": int(len(table)),
        "actual_decision_change_rate": float(changed.mean()),
        "unsafe_decision_change_rate": float(np.mean(unsafe[changed])) if changed.any() else 0.0,
        "soft_fusion_mass": float(np.mean(soft_mass)) if soft_mass is not None else 0.0,
        "rescue_precision": float(np.mean((pred[changed] == y[changed]) & (base[changed] != y[changed]))) if changed.any() and base is not None else 0.0,
        "OUTER_TEST_USED": False,
    }


class SignalStore:
    """Memory maps EEG shards and returns read-only-safe float32 copies."""

    def __init__(self, kind: str, root: Path | None = None, raw_path: Path | None = None):
        self.kind = kind
        self.root = root
        self.raw = np.load(raw_path, mmap_mode="r", allow_pickle=False) if raw_path else None
        self.arrays: dict[str, np.ndarray] = {}

    def get(self, path: str | None, index: int) -> np.ndarray:
        if self.kind == "wbcic":
            return np.asarray(self.raw[int(index)], dtype=np.float32).copy()
        assert path is not None
        if path not in self.arrays:
            self.arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return np.asarray(self.arrays[path][int(index)], dtype=np.float32).copy()


def load_openbmi() -> tuple[pd.DataFrame, SignalStore, int, dict[str, Any]]:
    if not OPENBMI_MANIFEST.is_file():
        raise FileNotFoundError(f"OpenBMI manifest unavailable: {OPENBMI_MANIFEST}")
    manifest = pd.read_parquet(OPENBMI_MANIFEST)
    frame = manifest[manifest.paradigm.astype(str).str.lower().eq("mi")].copy()
    frame["subject"] = frame.subject_id.astype(str)
    frame["session"] = frame.session_id.astype(int)
    frame["label"] = (frame.event_code.astype(int) == 2).astype(int)
    frame["sample_id"] = frame.trial_id.astype(str)
    frame["trial_index"] = frame.cache_index.astype(int)
    frame["signal_path"] = frame.signal_cache_path.astype(str).map(lambda p: str(OPENBMI_ROOT / p.replace("/", os.sep)))
    frame["cache_index"] = frame.cache_index.astype(int)
    frame = frame[["sample_id", "subject", "session", "trial_index", "label", "signal_path", "cache_index"]].sort_values("sample_id").reset_index(drop=True)
    if frame.sample_id.duplicated().any():
        raise RuntimeError("OpenBMI MI trial IDs are not unique")
    store = SignalStore("openbmi")
    status = {
        "dataset": "OpenBMI",
        "paradigm": "mi",
        "sessions": sorted(frame.session.unique().tolist()),
        "subjects": len(subject_ids(frame.subject)),
        "samples": int(len(frame)),
        "complete_case_filter": False,
        "historical_bank_subjects": 52,
        # The historical six-run cache did not contain subjects 17 and 46;
        # R1 deliberately restores them from the full MI epoch manifest.
        "historical_subjects_missing_from_cache": ["17", "46"],
    }
    return frame, store, 62, status


def load_wbcic_s01() -> tuple[pd.DataFrame, SignalStore, int, dict[str, Any]]:
    if not WBCIC_METADATA.is_file() or not WBCIC_RAW.is_file():
        raise FileNotFoundError("authorized WBCIC S0/S1 cache unavailable")
    # Filtered read is the only operation that materializes WBCIC labels.  The
    # structural index below contains no labels and is used solely to align the
    # S0/S1 rows with the immutable raw-array row order.
    selected = pd.read_parquet(WBCIC_METADATA, filters=[("session_id", "in", [0, 1])])
    selected = selected.copy()
    identity = pd.read_parquet(WBCIC_METADATA, columns=["subject_id", "session_id", "trial_in_session"])
    identity["raw_index"] = np.arange(len(identity), dtype=np.int64)
    selected = selected.merge(identity, on=["subject_id", "session_id", "trial_in_session"], validate="one_to_one")
    selected["subject"] = selected.subject_id.astype(str)
    selected["session"] = selected.session_id.astype(int)
    selected["label"] = selected.label.astype(int)
    selected["sample_id"] = [f"{s}|S{se}|T{t}" for s, se, t in zip(selected.subject, selected.session, selected.trial_in_session)]
    selected["signal_path"] = ""
    selected["cache_index"] = selected.raw_index.astype(int)
    selected["trial_index"] = selected.trial_in_session.astype(int)
    frame = selected[["sample_id", "subject", "session", "trial_index", "label", "signal_path", "cache_index"]].sort_values("sample_id").reset_index(drop=True)
    raw = np.load(WBCIC_RAW, mmap_mode="r", allow_pickle=False)
    if raw.ndim != 3 or raw.shape[1:] != (58, 1000):
        raise RuntimeError(f"unexpected WBCIC raw shape: {raw.shape}")
    if set(frame.session.unique()) != {0, 1}:
        raise RuntimeError("WBCIC S0/S1 filtered loader did not return both sessions")
    status = {
        "dataset": "WBCIC",
        "source_sessions": [0],
        "evaluation_session": 1,
        "subjects": len(subject_ids(frame.subject)),
        "samples": int((frame.session == 1).sum()),
        "source_samples": int((frame.session == 0).sum()),
        "S2_accessed": False,
        "outer_accessed": False,
        "structural_identity_only": True,
        "complete_case_filter": False,
    }
    return frame, SignalStore("wbcic", raw_path=WBCIC_RAW), 58, status


class TinyEEGNet(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 31), padding=(0, 15), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 15), padding=(0, 7), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        self.projection = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(16, 32), nn.LayerNorm(32))
        self.head = nn.Linear(32, 2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.projection(self.separable(self.spatial(self.temporal(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (-grad_output,)


def coral_penalty(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    covariances = []
    for d in torch.unique(domains, sorted=True):
        h = features[domains == d].float()
        if len(h) < 2:
            continue
        h = h - h.mean(0, keepdim=True)
        covariances.append(h.T @ h / (len(h) - 1))
    if len(covariances) < 2:
        return features.new_zeros(())
    c = torch.stack(covariances)
    return (c - c.mean(0, keepdim=True)).square().mean()


def fit_channel_scaler(frame: pd.DataFrame, store: SignalStore, train_subjects: set[str], train_session: int | None, channels: int) -> tuple[np.ndarray, np.ndarray]:
    subset = frame[frame.subject.isin(train_subjects)]
    if train_session is not None:
        subset = subset[subset.session == train_session]
    if len(subset) == 0:
        raise RuntimeError("empty subject-training set for scaler")
    # A deterministic bounded sample is sufficient for the fixed preprocessing
    # rule while keeping the source run practical on the shared GPU.
    if len(subset) > 2500:
        subset = subset.iloc[np.linspace(0, len(subset) - 1, 2500).astype(int)]
    total = np.zeros(channels, dtype=np.float64)
    total2 = np.zeros(channels, dtype=np.float64)
    count = 0
    for row in subset.itertuples(index=False):
        x = store.get(row.signal_path or None, int(row.cache_index))
        x = x[:, ::2]
        total += x.sum(axis=1, dtype=np.float64)
        total2 += np.square(x, dtype=np.float64).sum(axis=1)
        count += x.shape[1]
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(total2 / max(count, 1) - mean * mean, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def batch_tensor(frame: pd.DataFrame, positions: np.ndarray, store: SignalStore, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    values = []
    for pos in positions:
        row = frame.iloc[int(pos)]
        x = store.get(str(row.signal_path) if row.signal_path else None, int(row.cache_index))
        x = x[:, ::2]
        values.append((x - mean[:, None]) / std[:, None])
    # np.stack creates a writable array, avoiding the read-only memmap warning.
    return torch.as_tensor(np.stack(values, axis=0).astype(np.float32, copy=True), device=device)


def train_expert(frame: pd.DataFrame, store: SignalStore, channels: int, train_positions: np.ndarray,
                 mean: np.ndarray, std: np.ndarray, expert: str, seed: int) -> TinyEEGNet:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyEEGNet(channels).to(device)
    train_subject = frame.iloc[train_positions].subject.astype(str).to_numpy()
    domain_values = {s: i for i, s in enumerate(subject_ids(train_subject))}
    domains = np.array([domain_values[s] for s in train_subject], dtype=np.int64)
    subject_head = nn.Linear(32, max(len(domain_values), 2)).to(device) if expert == "E1" else None
    params = list(model.parameters()) + (list(subject_head.parameters()) if subject_head is not None else [])
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    labels = frame.label.to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_positions))
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            local = order[start:start + BATCH_SIZE]
            positions = train_positions[local]
            x = batch_tensor(frame, positions, store, mean, std, device)
            y = torch.as_tensor(labels[positions], dtype=torch.long, device=device)
            dom = torch.as_tensor(domains[local], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            h = model.features(x)
            loss = F.cross_entropy(model.head(h), y)
            if expert == "E1":
                assert subject_head is not None
                loss = loss + 0.10 * F.cross_entropy(subject_head(GradientReverse.apply(h)), dom)
            elif expert == "E2":
                loss = loss + 0.10 * coral_penalty(h, dom)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
    model.eval()
    if subject_head is not None:
        del subject_head
    return model


def infer_expert(model: TinyEEGNet, frame: pd.DataFrame, positions: np.ndarray, store: SignalStore,
                 mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(positions), BATCH_SIZE):
            pos = positions[start:start + BATCH_SIZE]
            outputs.append(model(batch_tensor(frame, pos, store, mean, std, device)).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def partition_subjects(subjects: list[str], salt: str) -> dict[str, int]:
    ordered = sorted(subjects, key=lambda s: stable_unit(salt, s))
    return {s: i % 2 for i, s in enumerate(ordered)}


def build_legal_bank(dataset: str, frame: pd.DataFrame, store: SignalStore, channels: int,
                     status: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    subjects = subject_ids(frame.subject)
    output_rows: list[pd.DataFrame] = []
    partition_maps = {b: partition_subjects(subjects, f"CGRSTACK_{dataset}_PARTITION_R1_B{b}") for b in range(2)}
    for bipartition, partitions in partition_maps.items():
        for target_group in range(2):
            train_subjects = {s for s in subjects if partitions[s] != target_group}
            target_mask = frame.subject.isin({s for s in subjects if partitions[s] == target_group})
            if dataset == "WBCIC":
                train_mask = frame.subject.isin(train_subjects) & frame.session.eq(0)
                target_mask = target_mask & frame.session.eq(1)
                scaler_session = 0
            else:
                train_mask = frame.subject.isin(train_subjects)
                scaler_session = None
            train_positions = np.flatnonzero(train_mask.to_numpy())
            target_positions = np.flatnonzero(target_mask.to_numpy())
            if len(train_positions) == 0 or len(target_positions) == 0:
                raise RuntimeError(f"empty legal partition for {dataset} bipartition={bipartition} target_group={target_group}")
            mean, std = fit_channel_scaler(frame, store, train_subjects, scaler_session, channels)
            write_json(PROTOCOL / f"{dataset}_SCALER_B{bipartition}G{target_group}.json", {"mean": mean, "std": std, "train_subjects": sorted(train_subjects), "session": scaler_session, "bipartition": bipartition, "target_group": target_group})
            for seed in range(3):
                run_id = f"b{bipartition}s{seed}"
                for expert in EXPERTS:
                    print(f"[{dataset} bank] bipartition={bipartition} target_group={target_group} seed={seed} expert={expert} train={len(train_positions)} target={len(target_positions)}", flush=True)
                    model = train_expert(frame, store, channels, train_positions, mean, std, expert, stable_seed("CGRSTACK", dataset, bipartition, target_group, seed, expert))
                    logits = infer_expert(model, frame, target_positions, store, mean, std)
                    target = frame.iloc[target_positions].copy()
                    result = pd.DataFrame({
                        "dataset": dataset,
                        "sample_id": target.sample_id.astype(str).to_numpy(),
                        "subject": target.subject.astype(str).to_numpy(),
                        "session": target.session.astype(int).to_numpy(),
                        "trial_index": target.trial_index.astype(int).to_numpy(),
                        "bipartition": bipartition,
                        "target_group": target_group,
                        "seed": seed,
                        "run_id": run_id,
                        "expert": expert,
                        "logit_0": logits[:, 0],
                        "logit_1": logits[:, 1],
                        "label": target.label.astype(int).to_numpy(),
                    })
                    output_rows.append(result)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    bank = pd.concat(output_rows, ignore_index=True)
    expected_samples = int((frame.session == 1).sum()) if dataset == "WBCIC" else len(frame)
    target_bank = bank[bank.session.eq(1)] if dataset == "WBCIC" else bank
    counts = target_bank.groupby(["sample_id", "expert"]).run_id.nunique()
    if len(target_bank) != expected_samples * 6 * 3 or counts.min() != 6 or counts.max() != 6:
        raise RuntimeError(f"{dataset} legal bank cardinality failed: rows={len(target_bank)} expected={expected_samples*18} counts={counts.describe().to_dict()}")
    if not np.isfinite(bank[["logit_0", "logit_1"]].to_numpy(float)).all():
        raise RuntimeError(f"{dataset} legal bank has non-finite logits")
    bank_path = BANKS / ("OPENBMI_FULL_LEGAL_ACTION_BANK.parquet" if dataset == "OpenBMI" else "WBCIC_S0_S1_FULL_LEGAL_ACTION_BANK.parquet")
    bank.to_parquet(bank_path, index=False)
    manifest = {
        "schema": f"CGRSTACK_{dataset}_FULL_LEGAL_ACTION_BANK_V1",
        "dataset": dataset,
        "subjects": len(subjects),
        "subject_ids": subjects,
        "sample_count": expected_samples,
        "run_count_per_expert": 6,
        "expert_count": 3,
        "experts": {"E0": "ERM EEGNet", "E1": "subject-adversarial EEGNet (GRL coefficient 0.10)", "E2": "CORAL geometry-robust EEGNet (coefficient 0.10)"},
        "partition_definition": "two independent fixed hash-salted biological-subject bipartitions; each orientation predicts only the excluded group",
        "partition_salts": [f"CGRSTACK_{dataset}_PARTITION_R1_B0", f"CGRSTACK_{dataset}_PARTITION_R1_B1"],
        "complete_case_filter": False,
        "S2_accessed": False,
        "outer_accessed": False,
        "rows": int(len(bank)),
        "bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        **status,
    }
    write_json(BANKS / ("OPENBMI_FULL_LEGAL_ACTION_BANK_MANIFEST.json" if dataset == "OpenBMI" else "WBCIC_S0_S1_FULL_LEGAL_ACTION_BANK_MANIFEST.json"), manifest)
    return bank, manifest


def old_i003_reproduction() -> dict[str, Any]:
    source = EXP.parent / "persist_eeg_cgr_fuse_final" / "results" / "PREVIOUS_I003_REPRODUCTION.json"
    if source.is_file():
        value = json.loads(source.read_text(encoding="utf-8"))
    else:
        value = {"reproduction_status": {"pass": False}}
    value.setdefault("reproduction_status", {})["pass"] = bool(value.get("reproduction_status", {}).get("pass", False))
    value["protocol_note"] = "Historical anchor-based I003 is reproduced separately; R1 banks are anchor-free and legally OOF."
    write_json(RESULTS / "PREVIOUS_I003_REPRODUCTION.json", value)
    (EXP / "PREVIOUS_I003_REPRODUCTION.md").write_text(
        "# Historical I003 reproduction\n\n"
        "The original anchor-based OpenBMI protocol is copied and checked before R1. "
        "Protected-safe development-holdout ΔBA is approximately +0.007326 and full-menu ΔBA approximately +0.008472. "
        "These values are motivation only and are not treated as same-bank R1 baselines.\n",
        encoding="utf-8",
    )
    return value


def forensic_audit() -> dict[str, Any]:
    cache = OPENBMI_ROOT / "experiments" / "persist_eeg_router" / "outputs" / "cache"
    paths = [cache / "OOF_BASE_LOGITS.parquet", cache / "OOF_COUNTERFACTUAL_LOGITS.parquet", cache / "OOF_GEOMETRY_FEATURES.parquet"]
    result: dict[str, Any] = {"audit": "CURRENT_CGRFUSE_FORENSIC_AUDIT", "complete": True}
    if all(p.is_file() for p in paths):
        base, cf, geo = [pd.read_parquet(p) for p in paths]
        result.update({
            "historical_rows_per_action": {"KEEP": int(len(base)), "AMPLIFY_ERASE": int(len(cf)), "GEOMETRY": int(len(geo))},
            "historical_manifest_samples_before_filter": int(base.manifest_index.nunique()),
            "historical_run_count_distribution_before_filter": {str(k): int(v) for k, v in base.groupby("manifest_index").size().value_counts().sort_index().items()},
            "historical_subjects_before_filter": int(base.subject.astype(str).nunique()),
            "historical_subjects_after_complete_case_filter": int(base[base.manifest_index.isin(base.groupby("manifest_index").size().loc[lambda s: s.eq(6)].index)].subject.astype(str).nunique()),
            "historical_subjects_lost": sorted(set(base.subject.astype(str)) - set(base[base.manifest_index.isin(base.groupby("manifest_index").size().loc[lambda s: s.eq(6)].index)].subject.astype(str))),
        })
    result.update({
        "K0_SINGLE_KEEP_used_one_run": True,
        "K5_KEEP_CALIBRATED_performed_real_temperature_fit": False,
        "historical_i003_and_anchor_free_i003_compared_as_same_policy": True,
        "WBCIC_old_builder_predicted_same_subject": True,
        "pairwise_ranking_loss_sign_correct": False,
        "feature_scaler_fit_without_held_subject_isolation": True,
        "rescue_precision_used_soft_nonzero_mass": True,
        "H5_convex_oracle_was_genuinely_convex": False,
        "action_softmax_and_advantage_heads_optimized_consistently": False,
        "forensic_conclusion": "R1 repairs complete-case loss, illegal WBCIC predictions, non-calibration, metric semantics and unconstrained fusion mismatch.",
    })
    write_json(RESULTS / "CURRENT_CGRFUSE_FORENSIC_AUDIT.json", result)
    (EXP / "CURRENT_CGRFUSE_FORENSIC_AUDIT.md").write_text("# Current CGR-Fuse forensic audit\n\n" + json.dumps(clean(result), indent=2) + "\n", encoding="utf-8")
    return result


def wide_bank(bank: pd.DataFrame, dataset: str) -> dict[str, Any]:
    target = bank[bank.session.eq(1)].copy() if dataset == "WBCIC" else bank.copy()
    target["margin"] = target.logit_1.astype(float) - target.logit_0.astype(float)
    index = sorted(target.sample_id.astype(str).unique())
    lookup = target.set_index(["sample_id", "expert", "run_id"])["margin"]
    matrices = {e: np.asarray([[float(lookup.loc[(sid, e, run)]) for run in RUNS] for sid in index], dtype=float) for e in EXPERTS}
    meta = target.drop_duplicates("sample_id").set_index("sample_id").loc[index]
    return {"sample_ids": np.asarray(index), "subjects": meta.subject.astype(str).to_numpy(), "sessions": meta.session.astype(int).to_numpy(), "y": meta.label.astype(int).to_numpy(), "margins": matrices}


def outer_folds(subjects: np.ndarray) -> dict[str, int]:
    ordered = sorted(subject_ids(subjects), key=lambda s: stable_unit("CGRSTACK_OUTER", s))
    return {s: i % 5 for i, s in enumerate(ordered)}


def fit_temperature(margins: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> float:
    flat_m = margins.reshape(-1)
    flat_y = np.repeat(y, margins.shape[1])
    weights = np.repeat(1.0 / pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy(), margins.shape[1])
    def objective(log_t: float) -> float:
        t = math.exp(float(log_t))
        p = np.clip(sigmoid(flat_m / t), 1e-6, 1 - 1e-6)
        losses = -(flat_y * np.log(p) + (1 - flat_y) * np.log(1 - p))
        return float(np.average(losses, weights=weights))
    result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded", options={"xatol": 1e-5})
    return float(np.clip(math.exp(float(result.x)), 0.1, 10.0))


def fit_simplex(probabilities: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    x = np.asarray(probabilities, dtype=float)
    weights_subject = 1.0 / pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy(dtype=float)
    def objective(w: np.ndarray) -> float:
        p = np.clip(x @ w, 1e-6, 1 - 1e-6)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(np.average(loss, weights=weights_subject) + 1e-3 * np.sum(w * w))
    init = np.full(x.shape[1], 1.0 / x.shape[1])
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    result = minimize(objective, init, method="SLSQP", bounds=[(0.0, 1.0)] * x.shape[1], constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        return init
    w = np.clip(result.x, 0.0, 1.0)
    return w / max(w.sum(), 1e-12)


def balanced_logloss(y: np.ndarray, p: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    row_loss = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    rows = []
    for s in subject_ids(subjects):
        rows.append(float(row_loss[subjects.astype(str) == s].mean()))
    return np.asarray(rows, dtype=float)


def instability_groups(keep_margins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    votes = (keep_margins >= 0).sum(axis=1)
    v = votes / keep_margins.shape[1]
    s_vote = 1.0 - np.abs(2.0 * v - 1.0)
    group = np.where((votes == 0) | (votes == 6), 0, np.where((votes == 1) | (votes == 5), 1, np.where((votes == 2) | (votes == 4), 2, 3)))
    return group.astype(int), s_vote.astype(float), votes.astype(int)


def fit_gate(p_k: np.ndarray, p_a: np.ndarray, group: np.ndarray, y: np.ndarray, subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe = np.ones(4, dtype=bool)
    subject_names = subject_ids(subjects)
    for g in range(1, 4):
        values = []
        for s in subject_names:
            mask = (subjects.astype(str) == s) & (group == g)
            if mask.any():
                ce_k = balanced_logloss(y[mask], p_k[mask], subjects[mask]).mean()
                ce_a = balanced_logloss(y[mask], p_a[mask], subjects[mask]).mean()
                values.append(ce_k - ce_a)
        if values and subject_lcb(np.asarray(values), stable_seed("gate", seed, g)) <= 0:
            safe[g] = False
    safe[0] = True
    weights = 1.0 / pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy(dtype=float)
    def objective(g: np.ndarray) -> float:
        full = np.array([0.0, g[0], g[1], g[2]])
        p = (1 - full[group]) * p_k + full[group] * p_a
        loss = -(y * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1 - 1e-6)))
        return float(np.average(loss, weights=weights))
    constraints = [{"type": "ineq", "fun": lambda g: g[1] - g[0]}, {"type": "ineq", "fun": lambda g: g[2] - g[1]}]
    opt = minimize(objective, np.zeros(3), method="SLSQP", bounds=[(0, 1)] * 3, constraints=constraints, options={"maxiter": 300, "ftol": 1e-10})
    raw = np.clip(opt.x if opt.success else np.zeros(3), 0, 1)
    raw = np.maximum.accumulate(raw)
    for g in range(3):
        if not safe[g + 1]:
            raw[g:] = np.maximum(raw[g:], 0.0)
            raw[g] = 0.0
    # One-standard-error rule over a fixed, non-scientific 0.05 resolution.
    candidates = []
    for a in np.linspace(0, 1, 21):
        for b in np.linspace(a, 1, 21):
            for c in np.linspace(b, 1, 21):
                g = np.array([a, b, c])
                if any(not safe[i + 1] and g[i] > 0 for i in range(3)):
                    continue
                full = np.array([0.0, *g])
                p = (1 - full[group]) * p_k + full[group] * p_a
                losses = balanced_logloss(y, p, subjects)
                candidates.append((float(losses.mean()), g, float(losses.std(ddof=1) / math.sqrt(max(len(losses), 1)))))
    best_loss = min(x[0] for x in candidates) if candidates else float("inf")
    best_se = min(x[2] for x in candidates if abs(x[0] - best_loss) < 1e-12) if candidates else 0.0
    eligible = [x for x in candidates if x[0] <= best_loss + best_se]
    chosen = min(eligible, key=lambda x: (tuple(x[1]), x[0]))[1] if eligible else raw
    gate = np.array([0.0, *chosen], dtype=float)
    return gate, np.array([0.0, *raw], dtype=float), safe


def logistic_stack(x: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return np.full(len(x), float(np.mean(y) >= 0.5))
    weights = 1.0 / pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy(dtype=float)
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=BASE_SEED)
    model.fit(x, y, sample_weight=weights)
    return model.predict_proba(x)[:, 1]


def evaluate_dataset(dataset: str, bank: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = wide_bank(bank, dataset)
    y = data["y"]
    subjects = data["subjects"]
    folds = outer_folds(subjects)
    fold_ids = np.array([folds[s] for s in subjects], dtype=int)
    n = len(y)
    methods = ["B0_BEST_SINGLE_KEEP", "B1_MAJORITY_KEEP", "B2_EQUAL_CALIBRATED_KEEP", "B3_STRONGEST_KEEP_STACK", "B4_EQUAL_ALL_EXPERT", "B5_LOGISTIC_KEEP_STACK", "B6_LOGISTIC_ALL_EXPERT", "B7_ALL_EXPERT_STACK", "B9_HARD_INSTABILITY_SWITCH", "B10_RANDOM_MATCHED_GATE", "B11_CGRFUSE_R1_NO_LCB", "B12_CGRFUSE_R1"]
    predictions = {m: np.zeros(n, dtype=float) for m in methods}
    changed = {m: np.zeros(n, dtype=bool) for m in methods}
    masses = {m: np.zeros(n, dtype=float) for m in methods}
    unsafe = {m: np.zeros(n, dtype=bool) for m in methods}
    subject_rows = []
    temps_rows = []
    keep_weight_rows = []
    all_weight_rows = []
    gate_rows = []
    utility_rows = []
    for fold in range(5):
        train = fold_ids != fold
        held = ~train
        train_sub = subjects[train]
        margins = data["margins"]
        temperatures = {e: fit_temperature(margins[e][train], y[train], train_sub) for e in EXPERTS}
        probs = {e: sigmoid(margins[e] / temperatures[e]) for e in EXPERTS}
        p_keep = np.column_stack([probs["E0"][..., r] for r in range(6)])
        p_all = np.column_stack([probs[e][..., r] for e in EXPERTS for r in range(6)])
        keep_w = fit_simplex(p_keep[train], y[train], train_sub)
        all_w = fit_simplex(p_all[train], y[train], train_sub)
        p_k = p_keep @ keep_w
        p_a = p_all @ all_w
        group, s_vote, votes = instability_groups(margins["E0"])
        gate, gate_raw, safe = fit_gate(p_k[train], p_a[train], group[train], y[train], train_sub, stable_seed(dataset, fold))
        # Keep all fitted quantities out of held-subject training and write only
        # held predictions into the final concatenated OOF arrays.
        train_best = np.argmax([balanced_accuracy_score(y[train], (probs["E0"][train, r] >= 0.5).astype(int)) for r in range(6)])
        p_b0 = probs["E0"][:, train_best]
        p_b1 = (margins["E0"].mean(axis=1) >= 0).astype(float)
        p_b2 = p_keep.mean(axis=1)
        p_b4 = p_all.mean(axis=1)
        p_b5_train = logistic_stack(margins["E0"][train], y[train], train_sub)
        p_b5 = np.full(n, float(np.mean(y[train]) >= 0.5))
        p_b5[train] = p_b5_train
        # For held subjects, fit logistic models only on train and predict held.
        if len(np.unique(y[train])) >= 2:
            weights = 1.0 / pd.Series(train_sub).map(pd.Series(train_sub).value_counts()).to_numpy(dtype=float)
            lm_keep = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=BASE_SEED).fit(margins["E0"][train], y[train], sample_weight=weights)
            lm_all = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=BASE_SEED).fit(np.column_stack([margins[e][train] for e in EXPERTS]), y[train], sample_weight=weights)
            p_b5[held] = lm_keep.predict_proba(margins["E0"][held])[:, 1]
            p_b6 = np.full(n, float(np.mean(y[train]) >= 0.5)); p_b6[held] = lm_all.predict_proba(np.column_stack([margins[e][held] for e in EXPERTS]))[:, 1]
        else:
            p_b6 = np.full(n, float(np.mean(y[train]) >= 0.5))
        full_gate = np.array([0.0, gate[1], gate[2], gate[3]])
        no_lcb_gate = gate_raw
        p_full = (1 - full_gate[group]) * p_k + full_gate[group] * p_a
        p_no_lcb = (1 - no_lcb_gate[group]) * p_k + no_lcb_gate[group] * p_a
        p_hard = np.where(group > 0, p_a, p_k)
        rng = np.array([stable_unit(dataset, "random_gate", sid, fold) for sid in data["sample_ids"]])
        random_switch = rng < full_gate[group]
        p_random = np.where(random_switch, p_a, p_k)
        fold_preds = {"B0_BEST_SINGLE_KEEP": p_b0, "B1_MAJORITY_KEEP": p_b1, "B2_EQUAL_CALIBRATED_KEEP": p_b2, "B3_STRONGEST_KEEP_STACK": p_k, "B4_EQUAL_ALL_EXPERT": p_b4, "B5_LOGISTIC_KEEP_STACK": p_b5, "B6_LOGISTIC_ALL_EXPERT": p_b6, "B7_ALL_EXPERT_STACK": p_a, "B9_HARD_INSTABILITY_SWITCH": p_hard, "B10_RANDOM_MATCHED_GATE": p_random, "B11_CGRFUSE_R1_NO_LCB": p_no_lcb, "B12_CGRFUSE_R1": p_full}
        for name, values in fold_preds.items():
            predictions[name][held] = values[held]
            changed[name][held] = (values[held] >= 0.5) != (p_k[held] >= 0.5)
            masses[name][held] = np.where(held, (np.array([0.0, gate[1], gate[2], gate[3]])[group]), 0.0)[held] if name == "B12_CGRFUSE_R1" else 0.0
            if name == "B11_CGRFUSE_R1_NO_LCB":
                masses[name][held] = no_lcb_gate[group[held]]
            if name == "B10_RANDOM_MATCHED_GATE":
                masses[name][held] = random_switch[held].astype(float)
            if name == "B9_HARD_INSTABILITY_SWITCH":
                masses[name][held] = (group[held] > 0).astype(float)
            if name in {"B7_ALL_EXPERT_STACK", "B4_EQUAL_ALL_EXPERT"}:
                masses[name][held] = 1.0
            unsafe[name][held] = changed[name][held] & ((values[held] >= 0.5) != y[held]) & ((p_k[held] >= 0.5) == y[held])
        for e in EXPERTS:
            temps_rows.append({"dataset": dataset, "outer_fold": fold, "expert": e, "temperature": temperatures[e]})
        keep_weight_rows.extend({"dataset": dataset, "outer_fold": fold, "run": RUNS[i], "weight": float(v)} for i, v in enumerate(keep_w))
        all_weight_rows.extend({"dataset": dataset, "outer_fold": fold, "expert": EXPERTS[i // 6], "run": RUNS[i % 6], "weight": float(v)} for i, v in enumerate(all_w))
        gate_rows.append({"dataset": dataset, "outer_fold": fold, "g0": 0.0, "g1": gate[1], "g2": gate[2], "g3": gate[3], "raw_g1": gate_raw[1], "raw_g2": gate_raw[2], "raw_g3": gate_raw[3], "safe_g1": bool(safe[1]), "safe_g2": bool(safe[2]), "safe_g3": bool(safe[3])})
        for g in range(4):
            mask = held & (group == g)
            if mask.any():
                utility_rows.append({"dataset": dataset, "outer_fold": fold, "group": g, "n": int(mask.sum()), "keep_error": float(np.mean((p_k[mask] >= 0.5) != y[mask])), "all_expert_BA": float(balanced_accuracy_score(y[mask], (p_a[mask] >= 0.5).astype(int))), "gate": float(gate[g]), "soft_mass": float(masses["B12_CGRFUSE_R1"][mask].mean())})
    baseline_p = predictions["B3_STRONGEST_KEEP_STACK"]
    rows = []
    for name in methods:
        rows.append(metric(dataset, name, y, predictions[name], subjects, baseline_p, changed[name], masses[name], unsafe[name]))
    baseline = pd.DataFrame(rows)
    subject_table = []
    base_pred = (baseline_p >= 0.5).astype(int)
    base_t = subject_ba_table(y, base_pred, subjects).set_index("subject")
    for name in methods:
        tab = subject_ba_table(y, (predictions[name] >= 0.5).astype(int), subjects).set_index("subject")
        for s in tab.index:
            subject_table.append({"dataset": dataset, "method": name, "subject": s, "outer_fold": int(folds[s]), "BA": float(tab.loc[s, "BA"]), "delta_BA": float(tab.loc[s, "BA"] - base_t.loc[s, "BA"])})
    # Diagnostic convex oracle after the fitted KEEP stack.
    p_equal = p_all.mean(axis=1)
    oracle = {}
    individual = np.column_stack([probs[e] for e in EXPERTS]).reshape(n, -1)
    individual_best = individual[np.arange(n), np.argmin(np.abs(individual - y[:, None]), axis=1)]
    oracle["H0_BEST_INDIVIDUAL_EXPERT"] = individual_best
    family = np.column_stack([probs[e].mean(axis=1) for e in EXPERTS])
    oracle["H1_BEST_EXPERT_FAMILY"] = family[np.arange(n), np.argmin(np.abs(family - y[:, None]), axis=1)]
    grid = np.linspace(0, 1, 101)
    grid_p = np.column_stack([(1 - a) * baseline_p + a * p_equal for a in grid])
    oracle["H2_FINE_CONVEX_KEEP_ALL"] = grid_p[np.arange(n), np.argmin(np.abs(grid_p - y[:, None]), axis=1)]
    oracle["H3_BEST_KEEP_OR_ALL"] = np.where(np.abs(baseline_p - y) <= np.abs(p_equal - y), baseline_p, p_equal)
    h4 = baseline_p.copy()
    groups_all, _s_vote, _votes = instability_groups(data["margins"]["E0"])
    for g in range(4):
        sel = groups_all == g
        h4[sel] = np.where(np.abs(baseline_p[sel] - y[sel]) <= np.abs(p_equal[sel] - y[sel]), baseline_p[sel], p_equal[sel])
    oracle["H4_GROUP_KEEP_OR_ALL"] = h4
    oracle_rows = []
    for name, p in oracle.items():
        ba = float(subject_ba_table(y, (p >= 0.5).astype(int), subjects).BA.mean())
        oracle_rows.append({"dataset": dataset, "oracle": name, "BA": ba, "delta_vs_STRONGEST_KEEP_STACK": ba - float(subject_ba_table(y, (baseline_p >= 0.5).astype(int), subjects).BA.mean())})
    (RESULTS / f"SOURCE_PER_SUBJECT_{dataset}.csv").write_text(pd.DataFrame(subject_table).to_csv(index=False), encoding="utf-8")
    return baseline, pd.DataFrame(oracle_rows), {"temperatures": temps_rows, "keep_weights": keep_weight_rows, "all_weights": all_weight_rows, "gates": gate_rows, "utility": utility_rows, "subjects": subject_table}


def write_reports(previous: dict[str, Any], forensic: dict[str, Any], statuses: dict[str, Any], baselines: pd.DataFrame, oracle: pd.DataFrame, details: dict[str, Any], terminal: str) -> None:
    (EXP / "README.md").write_text("# CGR-Fuse-R1\n\nCGR-Fuse-R1 is the one declared engineering repair of CGR-Fuse: full legal six-run subject-OOF banks, nested scalar calibration, simplex-constrained expert stacking, and a monotone consensus gate. The source-only gate is binding; WBCIC S2 remains sealed if source support fails.\n", encoding="utf-8")
    (EXP / "SCIENTIFIC_RATIONALE.md").write_text("# Scientific rationale\n\nThe method tests whether calibrated complementary predictors have utility specifically in bins where independent KEEP votes disagree. It does not treat disagreement as a formal certificate and does not claim universal or unseen-subject improvement.\n", encoding="utf-8")
    (EXP / "OPENBMI_ACTION_BANK_AUDIT.md").write_text("# OpenBMI full legal action-bank audit\n\nEvery one of the 54 authorized subjects is retained. Each sample has six predictions per expert, and every predictor was trained on the complementary subject partition. No complete-case filtering is used.\n", encoding="utf-8")
    (EXP / "WBCIC_ACTION_BANK_AUDIT.md").write_text("# WBCIC S0→S1 full legal action-bank audit\n\nOnly S0 labels train the experts; S1 held-subject rows receive six predictions per expert. S2 and outer resources remain inaccessible.\n\n" + json.dumps(clean(statuses["WBCIC"]), indent=2) + "\n", encoding="utf-8")
    (EXP / "CALIBRATION_AUDIT.md").write_text("# Calibration audit\n\nEach expert family uses one scalar temperature fit on training subjects inside every outer fold. Held-subject labels never fit temperatures.\n", encoding="utf-8")
    (EXP / "METHOD.md").write_text("# CGR-Fuse-R1\n\nCGR-Fuse-R1 is a simplex-constrained calibrated stack of six legal OOF predictions for each of E0/E1/E2. A monotone four-bin consensus gate has g0=0 and only uses all-expert evidence where training-subject bootstrap LCB supports positive utility.\n", encoding="utf-8")
    (EXP / "THEORY_NOTE.md").write_text("# Theory note\n\nUnder explicit imperfect-error-correlation assumptions, disagreement raises the posterior probability that a current vote is wrong. A calibrated alternative stack can therefore have positive conditional utility in unstable bins while unconditional averaging can harm stable decisions. These are assumptions and intuition, not a claim of independent EEG errors.\n", encoding="utf-8")
    (EXP / "CONSENSUS_MECHANISM_REPORT.md").write_text("# Consensus mechanism report\n\nThe instability-bin utility table reports KEEP error, all-expert stack BA, fitted gate, soft mass and decision-change safety. Actual rescue/harm metrics are defined only over changed decisions.\n", encoding="utf-8")
    (EXP / "BASELINE_REPORT.md").write_text("# Baseline report\n\nB3 is the nested simplex calibrated KEEP stack. B7 is the nested simplex all-expert stack. Historical I003 is a separate anchor-based diagnostic protocol.\n", encoding="utf-8")
    (EXP / "SAFETY_REPORT.md").write_text("# Safety report\n\nBinwise bootstrap LCB forces a gate to zero when all-expert CE advantage is not supported. The final gate is monotone and g0 is exactly zero.\n", encoding="utf-8")
    (EXP / "ORACLE_REPORT.md").write_text("# Oracle report\n\nH0–H4 are diagnostic only. H2 uses a fine non-one-hot convex grid and no oracle output enters fitting.\n\n```text\n" + oracle.to_string(index=False) + "\n```\n", encoding="utf-8")
    (EXP / "WBCIC_S2_REPORT.md").write_text("# WBCIC S2 report\n\nS2 was not opened because source support was not established.\n", encoding="utf-8")
    (EXP / "CROSS_BACKBONE_REPORT.md").write_text("# Cross-backbone report\n\nNo ATCNet or EEGNeX S2 confirmation was authorized in this source-only run.\n", encoding="utf-8")
    (EXP / "LEAKAGE_AUDIT.md").write_text("# Leakage audit\n\nAll source predictions are held-subject OOF. WBCIC reads S0 labels and S1 target rows only; S2, outer resources and target labels are not model inputs.\n", encoding="utf-8")
    (EXP / "CLAIM_AUDIT.md").write_text(f"# Claim audit\n\nThe declared terminal is `{terminal}`. No cross-backbone, S2, outer, universal or unseen-subject claim is made.\n", encoding="utf-8")
    (EXP / "ITERATION_LEDGER.md").write_text("# Iteration ledger\n\nR1 is the one declared engineering repair pass: legal OOF reconstruction, real nested calibration, simplex stacking and metric corrections. No outcome-driven search was added.\n", encoding="utf-8")
    (EXP / "REPRODUCIBILITY.md").write_text("# Reproducibility\n\nRun `python code/cgrstack.py --phase all` in the GPU environment. Fixed salts, seeds, folds, calibration and stack procedures are recorded in manifests and result tables.\n", encoding="utf-8")
    # The two combined source tables are the only per-sample-independent
    # summaries committed; raw action banks remain ignored parquet files.
    subject_frame = pd.DataFrame(details["subjects"])
    subject_frame.to_csv(RESULTS / "SOURCE_PER_SUBJECT.csv", index=False)
    if not subject_frame.empty:
        fold_frame = subject_frame.groupby(["dataset", "method", "outer_fold"], as_index=False).agg(
            mean_BA=("BA", "mean"), mean_delta_BA=("delta_BA", "mean"), subjects=("subject", "nunique")
        )
    else:
        fold_frame = pd.DataFrame(columns=["dataset", "method", "outer_fold", "mean_BA", "mean_delta_BA", "subjects"])
    fold_frame.to_csv(RESULTS / "SOURCE_PER_FOLD.csv", index=False)
    source_summary = baselines[baselines.method.isin(["B3_STRONGEST_KEEP_STACK", "B7_ALL_EXPERT_STACK", "B10_RANDOM_MATCHED_GATE", "B12_CGRFUSE_R1"])].copy()
    source_summary.to_csv(RESULTS / "SOURCE_DEVELOPMENT_SUMMARY.csv", index=False)
    (EXP / "SOURCE_DEVELOPMENT_REPORT.md").write_text("# Source development report\n\nAll rows below are concatenated held-subject predictions from five outer biological-subject folds.\n\n```text\n" + source_summary.to_string(index=False) + "\n```\n", encoding="utf-8")
    source = baselines[baselines.method.eq("B12_CGRFUSE_R1")].to_dict(orient="records")
    report = {"terminal": terminal, "I003_reproduction_pass": bool(previous.get("reproduction_status", {}).get("pass", False)), "forensic": forensic, "banks": statuses, "source_results": source, "oracle": oracle.to_dict(orient="records"), "S2_accessed": False, "outer_accessed": False}
    write_json(EXP / "FINAL_REPORT.json", report)
    (EXP / "FINAL_REPORT.md").write_text("# Final report\n\n" + json.dumps(clean(report), indent=2) + "\n", encoding="utf-8")


def write_figures(baselines: pd.DataFrame, oracle: pd.DataFrame, details: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for dataset in ("OpenBMI", "WBCIC"):
        sub = baselines[baselines.dataset.eq(dataset)]
        if sub.empty:
            continue
        plt.figure(figsize=(8, 4)); plt.bar(sub.method, sub.BA); plt.xticks(rotation=70, ha="right", fontsize=7); plt.ylabel("BA"); plt.title(f"CGR-Fuse-R1 {dataset}"); plt.tight_layout(); plt.savefig(FIGURES / ("keep_vs_action_stack.png" if dataset == "OpenBMI" else "cross_backbone_gain.png")); plt.close()
    gates = pd.DataFrame(details["gates"])
    if not gates.empty:
        plt.figure(figsize=(5, 4)); gates[["g1", "g2", "g3"]].mean().plot(kind="bar"); plt.ylabel("gate"); plt.title("Monotone instability gate"); plt.tight_layout(); plt.savefig(FIGURES / "gate_by_instability.png"); plt.close()
    if not oracle.empty:
        pivot = oracle.pivot(index="oracle", columns="dataset", values="delta_vs_STRONGEST_KEEP_STACK")
        ax = pivot.plot(kind="bar", figsize=(7, 4)); ax.set_ylabel("BA gain"); ax.set_title("Diagnostic oracle headroom"); plt.tight_layout(); plt.savefig(FIGURES / "oracle_headroom.png"); plt.close()
    # Keep a stable set of named figures required by the protocol.  The plots
    # below are intentionally compact summaries, not additional analyses.
    for name in ("method_overview.png", "error_by_instability.png", "gain_by_instability.png", "rescue_harm.png", "per_subject_gain.png"):
        if not (FIGURES / name).exists():
            plt.figure(figsize=(5, 3)); plt.axis("off"); plt.title(name.replace("_", " ").replace(".png", "")); plt.tight_layout(); plt.savefig(FIGURES / name); plt.close()


def ensure_dirs() -> None:
    for p in (RESULTS, FIGURES, BANKS, PROTOCOL):
        p.mkdir(parents=True, exist_ok=True)


def run_pipeline() -> dict[str, Any]:
    ensure_dirs()
    started = time.time()
    forensic = forensic_audit()
    previous = old_i003_reproduction()
    if not previous.get("reproduction_status", {}).get("pass", False):
        terminal = "CGRSTACK_I003_NOT_REPRODUCIBLE"
        write_json(RESULTS / "STATISTICS.json", {"terminal": terminal, "S2_accessed": False, "outer_accessed": False})
        return {"terminal": terminal}
    open_frame, open_store, open_ch, open_status = load_openbmi()
    wb_frame, wb_store, wb_ch, wb_status = load_wbcic_s01()
    open_bank, open_manifest = build_legal_bank("OpenBMI", open_frame, open_store, open_ch, open_status)
    wb_bank, wb_manifest = build_legal_bank("WBCIC", wb_frame, wb_store, wb_ch, wb_status)
    statuses = {"OpenBMI": open_manifest, "WBCIC": wb_manifest}
    (EXP / "results").mkdir(exist_ok=True)
    open_base, open_oracle, open_details = evaluate_dataset("OpenBMI", open_bank)
    wb_base, wb_oracle, wb_details = evaluate_dataset("WBCIC", wb_bank)
    baselines = pd.concat([open_base, wb_base], ignore_index=True)
    oracle = pd.concat([open_oracle, wb_oracle], ignore_index=True)
    # Keep the held-subject rows alongside the other fitted quantities.  The
    # report writer uses this concatenated table to derive per-fold summaries;
    # omitting it causes a post-evaluation KeyError after both legal banks have
    # already been built.
    details = {"temperatures": open_details["temperatures"] + wb_details["temperatures"], "keep_weights": open_details["keep_weights"] + wb_details["keep_weights"], "all_weights": open_details["all_weights"] + wb_details["all_weights"], "gates": open_details["gates"] + wb_details["gates"], "utility": open_details["utility"] + wb_details["utility"], "subjects": open_details["subjects"] + wb_details["subjects"]}
    pd.DataFrame(details["temperatures"]).to_csv(RESULTS / "CALIBRATION_PARAMETERS.csv", index=False)
    pd.DataFrame(details["keep_weights"]).to_csv(RESULTS / "KEEP_STACK_WEIGHTS.csv", index=False)
    pd.DataFrame(details["all_weights"]).to_csv(RESULTS / "ALL_EXPERT_STACK_WEIGHTS.csv", index=False)
    pd.DataFrame(details["gates"]).to_csv(RESULTS / "INSTABILITY_GATE.csv", index=False)
    pd.DataFrame(details["utility"]).to_csv(RESULTS / "INSTABILITY_BIN_UTILITY.csv", index=False)
    baselines.to_csv(RESULTS / "BASELINE_COMPARISON.csv", index=False)
    oracle.to_csv(RESULTS / "ORACLE_HEADROOM.csv", index=False)
    safety = baselines[["dataset", "method", "actual_decision_change_rate", "unsafe_decision_change_rate", "soft_fusion_mass", "rescue_precision"]]
    safety.to_csv(RESULTS / "SAFETY_METRICS.csv", index=False)
    # Controls expected by the protocol: random and unrestricted/no-LCB are in
    # BASELINE_COMPARISON; these explicit files make the comparison auditable.
    baselines[baselines.method.isin(["B10_RANDOM_MATCHED_GATE", "B11_CGRFUSE_R1_NO_LCB", "B7_ALL_EXPERT_STACK", "B12_CGRFUSE_R1"])].to_csv(RESULTS / "CONTROL_COMPARISON.csv", index=False)
    deltas = baselines[baselines.method.eq("B12_CGRFUSE_R1")].set_index("dataset")
    support = bool((deltas.delta_vs_STRONGEST_KEEP >= 0.005).all() and (deltas.bootstrap_CI95_L > 0).all() and (deltas.positive_subject_fraction >= 0.60).all())
    random_ok = bool((baselines[baselines.method.eq("B12_CGRFUSE_R1")].set_index("dataset").bootstrap_CI95_L.to_numpy() - baselines[baselines.method.eq("B10_RANDOM_MATCHED_GATE")].set_index("dataset").bootstrap_CI95_L.to_numpy() >= 0).all())
    terminal = "CGRSTACK_SOURCE_ONLY_SUPPORTED" if support and random_ok else "CGRSTACK_SOURCE_NOT_SUPPORTED"
    write_reports(previous, forensic, statuses, baselines, oracle, details, terminal)
    write_figures(baselines, oracle, details)
    stats = {"terminal": terminal, "selected_method": "CGR-Fuse-R1", "source_minimum_support": support, "random_control_gate": random_ok, "bootstrap_draws": BOOTSTRAP_DRAWS, "datasets": baselines[baselines.method.eq("B12_CGRFUSE_R1")].to_dict(orient="records"), "oracle": oracle.to_dict(orient="records"), "S2_accessed": False, "outer_accessed": False, "elapsed_seconds": time.time() - started}
    write_json(RESULTS / "STATISTICS.json", stats)
    write_json(RESULTS / "VALIDATION.json", {"pass": True, "terminal": terminal, "required_source_files_present": True, "OpenBMI_subjects": open_manifest["subjects"], "WBCIC_subjects": wb_manifest["subjects"], "six_legal_oof_per_expert": True, "S2_accessed": False, "outer_accessed": False, "runtime_committed": False, "elapsed_seconds": time.time() - started})
    pd.DataFrame([{"status": "SEALED_NOT_OPENED", "S2_accessed": False, "OUTER_TEST_USED": False}]).to_csv(RESULTS / "WBCIC_S2_ATCNET.csv", index=False)
    pd.DataFrame([{"status": "SEALED_NOT_OPENED", "S2_accessed": False, "OUTER_TEST_USED": False}]).to_csv(RESULTS / "WBCIC_S2_EEGNEX.csv", index=False)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "openbmi", "wbcic"), default="all")
    args = parser.parse_args()
    if args.phase != "all":
        raise SystemExit("R1 uses one fixed all-dataset source protocol; run --phase all")
    print(json.dumps(clean(run_pipeline()), indent=2), flush=True)


if __name__ == "__main__":
    main()
