#!/usr/bin/env python3
"""Frozen LiteBN/TFFormer Protected-coordinate erasure diagnostic (PEEH v1).

This is a representation diagnostic only.  It never trains or mutates either
neural network.  Protected coordinates and every ridge probe are fitted using
only the canonical inner-train biological subjects.  Evaluation uses the
OpenBMI internal-heldout subjects and the WBCIC true-outer subjects.

The persistence spectrum, block construction, subject-correspondence null,
subject-half utility selection, and canonical-coordinate erasure reproduce the
existing Signed-V3.1 implementation in
``persist_eeg_geosr_final_v1/code/audit_primitives.py``.  Protected selection
keeps its historical 24-d probe convention; the requested PEEH A/B/C metrics
use full 64-d probes and 20,000 subject bootstrap draws.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


REPO = Path("/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")
EXP = REPO / "experiments/persist_eeg_litebn_tfformer_peeh_v1"
OUT = EXP / "outputs/peeh_v1"
PROTOCOL = EXP / "protocol"
RUNTIME = Path("/root/rivermind-data/litebn_tfformer_peeh_v1_runtime")
CACHE = RUNTIME / "embeddings"
RUN_RESULTS = RUNTIME / "run_results"
REFERENCE = REPO / "experiments/persist_eeg_geosr_final_v1/code/audit_primitives.py"
MULTI_EXP = REPO / "experiments/persist_eeg_litebn_tfformer_multiseed_v1"
SEED0_EXP = REPO / "experiments/persist_eeg_litebn_tfformer_remaining_tasks_seed0_v1"
SEED0_SSVEP_EXP = REPO / "experiments/persist_eeg_litebn_tfformer_v1"
NORMALIZERS = Path("/root/rivermind-data/litebn_x_singlemodel_seed0_runtime/normalizers")
CARRIER = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")
TASK_RUNTIME = Path("/root/rivermind-data/openbmi_task_generality_runtime")
TRUE_OUTER = Path("/root/rivermind-data/persist_eeg_cache/wbcic_true_outer_v1/wbcic_epochs")

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MODELS = ("LiteBN", "TFFormer")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
OPENBMI_EVAL = ("4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54")
WBCIC_EVAL = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")

ACTIVE_RANK_MAX = 20
MAX_BLOCK_RANK = 4
PERSISTENCE_PERMUTATIONS = 200
INNER_HALF_SPLITS = 5
UTILITY_RANDOM_DRAWS = 100
ERASURE_RANDOM_DRAWS = 100
UTILITY_BOOTSTRAP_DRAWS = 10_000
FINAL_BOOTSTRAP_DRAWS = 20_000
PER_SUBJECT_SESSION_CLASS_CAP = 32
RIDGE_ALPHA = 0.01
EPS = 1e-12


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
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False) % (2**63 - 1)


def subject_key(value: object) -> tuple[int, str]:
    text = str(value).replace("sub-", "")
    return (int(text), text) if text.isdigit() else (10**9, text)


def subject_sort(values: Iterable[object]) -> list[str]:
    return sorted(map(str, values), key=subject_key)


def load_runtime():
    os.environ["TFFREM_SEED"] = "1"
    code = MULTI_EXP / "code"
    sys.path.insert(0, str(code))
    source = code / "run_multiseed.py"
    spec = importlib.util.spec_from_file_location("peeh_frozen_runtime", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen TFFormer runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def litebn_path(task: str, fold: int, seed: int) -> Path:
    if task in ("OpenBMI_ERP", "OpenBMI_SSVEP"):
        short = "erp" if task.endswith("ERP") else "ssvep"
        return TASK_RUNTIME / f"{short}_fold{fold}_seed{seed}_litebn/selected_best.pt"
    short = "openbmi" if task == "OpenBMI_MI" else "wbcic"
    return CARRIER / f"{short}_fold{fold}_seed{seed}_litebn/selected_best.pt"


def tfformer_path(task: str, fold: int, seed: int) -> Path:
    if seed == 0 and task == "OpenBMI_SSVEP":
        return SEED0_SSVEP_EXP / f"runtime/checkpoints/OpenBMI_SSVEP/fold{fold}.pt"
    if seed == 0:
        return SEED0_EXP / f"runtime/checkpoints/{task}/fold{fold}/selected.pt"
    return MULTI_EXP / f"runtime/seed{seed}/checkpoints/{task}/fold{fold}/selected.pt"


def checkpoint_path(model: str, task: str, fold: int, seed: int) -> Path:
    return litebn_path(task, fold, seed) if model == "LiteBN" else tfformer_path(task, fold, seed)


def normalizer_path(task: str, fold: int) -> Path:
    return NORMALIZERS / f"{task.lower()}_fold{fold}.npz"


def state_dict(path: Path, model: str) -> Mapping[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if model == "TFFormer":
        if not isinstance(value, dict) or "state_dict" not in value:
            raise RuntimeError(f"TFFormer checkpoint schema mismatch: {path}")
        return value["state_dict"]
    return value


def build_model(runtime, model_name: str, task: str, path: Path, device: torch.device):
    base = runtime.base.build_model("LiteBN_BASELINE", task)
    if model_name == "LiteBN":
        base.load_state_dict(state_dict(path, model_name), strict=True)
        model = base
    else:
        model = runtime.CachedTFFormer(base, 250)
        model.load_state_dict(state_dict(path, model_name), strict=True)
    model = model.to(device).eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("eval-mode lock failed")
    return model


def subset_bundle(runtime, bundle, sessions: Sequence[int]):
    wanted = set(map(int, sessions))
    rows = [row for row in bundle.rows if int(row.session) in wanted]
    return runtime.base.SignalBundle(bundle.task, bundle.subjects, rows)


def true_outer_bundle(runtime):
    rows = []
    for subject in WBCIC_EVAL:
        signal = TRUE_OUTER / subject / "ses-2_epochs.npy"
        labels = TRUE_OUTER / subject / "ses-2_labels.npy"
        x = np.load(signal, mmap_mode="r", allow_pickle=False)
        y = np.load(labels, mmap_mode="r", allow_pickle=False)
        if x.shape != (200, 58, 1000) or x.dtype != np.float16 or y.shape != (200,) or set(map(int, np.unique(y))) != {0, 1}:
            raise RuntimeError(f"WBCIC true-outer schema mismatch: {subject}")
        rows.extend(runtime.base.Row(subject, 2, str(signal), i, int(label)) for i, label in enumerate(y))
    return runtime.base.SignalBundle("WBCIC_MI", WBCIC_EVAL, rows)


def bundle_metadata(bundle) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [str(row.subject) for row in bundle.rows],
        "session_id": [int(row.session) for row in bundle.rows],
        "label": [int(row.label) for row in bundle.rows],
    })


def infer_representations(model, raw, mean: np.ndarray, std: np.ndarray, device: torch.device, model_name: str, task: str) -> np.ndarray:
    values = []
    batch_size = 128 if model_name == "LiteBN" else (64 if task == "OpenBMI_ERP" else 24)
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(raw.x.shape[0]), batch_size):
            idx = np.arange(start, min(start + batch_size, int(raw.x.shape[0])), dtype=np.int64)
            x, _ = raw.batch(idx, mean, std)
            output = model(x)
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("model does not expose pre-classifier representation")
            h = output[1]
            if h.ndim != 2 or h.shape[1] != 64:
                raise RuntimeError(f"expected 64-d representation, got {tuple(h.shape)}")
            values.append(h.float().cpu().numpy())
    result = np.concatenate(values, axis=0).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite representation")
    return result


def embedding_cache_path(task: str, model: str, fold: int, seed: int) -> Path:
    return CACHE / task / model / f"fold{fold}_seed{seed}.npz"


def save_embeddings(path: Path, train_h: np.ndarray, train_meta: pd.DataFrame, eval_h: np.ndarray, eval_meta: pd.DataFrame, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_h=train_h,
            train_y=train_meta.label.to_numpy(np.int64),
            train_subject=train_meta.subject_id.astype(str).to_numpy(dtype="U16"),
            train_session=train_meta.session_id.to_numpy(np.int64),
            eval_h=eval_h,
            eval_y=eval_meta.label.to_numpy(np.int64),
            eval_subject=eval_meta.subject_id.astype(str).to_numpy(dtype="U16"),
            eval_session=eval_meta.session_id.to_numpy(np.int64),
            metadata=np.asarray(json.dumps(clean(metadata), sort_keys=True)),
        )
    os.replace(temporary, path)


def load_embeddings(path: Path, expected_checkpoint_sha: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("checkpoint_sha256") != expected_checkpoint_sha:
            raise RuntimeError(f"stale embedding cache: {path}")
        result = {name: np.asarray(archive[name]) for name in ("train_h", "train_y", "train_subject", "train_session", "eval_h", "eval_y", "eval_subject", "eval_session")}
        result["metadata"] = metadata
    return result


def ridge_fit(x: np.ndarray, y: np.ndarray, classes: int, dimensions: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != 64:
        raise RuntimeError(f"ridge input must be full 64D, got {x.shape}")
    if dimensions not in (24, 64):
        raise RuntimeError(f"unsupported ridge dimension: {dimensions}")
    x = x[:, :dimensions]
    y = np.asarray(y, np.int64)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0
    design = np.c_[(x - mean) / std, np.ones(len(x))]
    penalty = np.eye(dimensions + 1); penalty[-1, -1] = 0.0
    target = np.eye(classes)[y]
    weight = np.linalg.solve(design.T @ design + RIDGE_ALPHA * penalty, design.T @ target)
    return weight, mean, std


def ridge_predict(x: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    weight, mean, std = pack
    x = np.asarray(x, np.float64)[:, :len(mean)]
    logits = np.c_[(x - mean) / std, np.ones(len(x))] @ weight
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= np.maximum(prob.sum(1, keepdims=True), EPS)
    return prob.argmax(1), prob


def balanced_accuracy(y: np.ndarray, prediction: np.ndarray, classes: int) -> float:
    y, prediction = np.asarray(y, np.int64), np.asarray(prediction, np.int64)
    recalls = [np.mean(prediction[y == label] == label) for label in range(classes) if np.any(y == label)]
    return float(np.mean(recalls))


def coordinates(h: np.ndarray, spectrum: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(h, np.float64) - spectrum["mean"]) @ spectrum["whitener"] @ spectrum["directions"]


def erase(h: np.ndarray, spectrum: Mapping[str, Any], selected: Sequence[int]) -> np.ndarray:
    # Exact Signed-V3.1 canonical-coordinate erasure and inverse map.
    q = coordinates(h, spectrum)
    delta_q = np.zeros_like(q)
    ids = np.asarray(selected, np.int64)
    if len(ids):
        delta_q[:, ids] = -q[:, ids]
    delta_h = (delta_q @ spectrum["directions"].T) @ spectrum["dewhitener"]
    return (np.asarray(h, np.float64) + delta_h).astype(np.float32)


def make_blocks(rho: np.ndarray) -> tuple[list[list[int]], dict[str, Any]]:
    # Exact Signed-V3.1 eigengap/max-size-4 rule.
    rank = len(rho)
    gaps = np.abs(np.diff(rho))
    threshold = max(float(np.median(gaps) * 4.0), float(np.max(np.abs(rho))) * 0.05, 1e-10)
    boundaries = [0] + [i + 1 for i, gap in enumerate(gaps) if gap > threshold] + [rank]
    blocks = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        for start in range(left, right, MAX_BLOCK_RANK):
            blocks.append(list(range(start, min(start + MAX_BLOCK_RANK, right))))
    if len(blocks) < 2 and rank > MAX_BLOCK_RANK:
        blocks = [list(range(0, MAX_BLOCK_RANK)), list(range(MAX_BLOCK_RANK, rank))]
    return blocks, {
        "construction": "Signed V3.1 train-only eigengap clustering followed by max-size-4 split",
        "eigengap_threshold": threshold,
        "block_dimensions": [len(block) for block in blocks],
        "no_evaluation_block_selection": True,
    }


def build_spectrum(meta: pd.DataFrame, h: np.ndarray, run_key: Sequence[Any]) -> dict[str, Any]:
    x = np.asarray(h, np.float64)
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    threshold = max(float(eigenvalues[0]) * 1e-3, 1e-8)
    numerical_rank = int(np.sum(eigenvalues > threshold))
    active_rank = min(ACTIVE_RANK_MAX, numerical_rank)
    if active_rank < 4:
        raise RuntimeError(f"insufficient active representation rank: {numerical_rank}")
    active = np.maximum(eigenvalues[:active_rank], max(float(eigenvalues[:active_rank].mean()) * 1e-4, 1e-8))
    active_vectors = eigenvectors[:, :active_rank]
    whitener = active_vectors * np.power(active, -0.5)[None, :]
    dewhitener = np.sqrt(active)[:, None] * active_vectors.T
    whitened = centered @ whitener

    frame = meta.reset_index(drop=True).copy()
    frame["position"] = np.arange(len(frame))
    sessions = sorted(frame.session_id.astype(int).unique())
    if len(sessions) != 2:
        raise RuntimeError(f"persistence requires exactly two train sessions, got {sessions}")
    classes = sorted(frame.label.astype(int).unique())
    centroids = {}
    for key, group in frame.groupby(["subject_id", "session_id", "label"], sort=True):
        centroids[(str(key[0]), int(key[1]), int(key[2]))] = whitened[group.position.to_numpy(np.int64)].mean(0)
    subjects = subject_sort(frame.subject_id.unique())
    class_covariances = []
    pair_count = 0
    for label in classes:
        left, right = [], []
        for subject in subjects:
            a, b = (subject, sessions[0], label), (subject, sessions[1], label)
            if a in centroids and b in centroids:
                left.append(centroids[a]); right.append(centroids[b])
        if left:
            aa, bb = np.asarray(left), np.asarray(right)
            aa -= aa.mean(0); bb -= bb.mean(0)
            class_covariances.append((aa.T @ bb + bb.T @ aa) / (2.0 * len(aa)))
            pair_count += len(left)
    persistence = np.mean(class_covariances, axis=0)
    rho, directions = np.linalg.eigh((persistence + persistence.T) / 2.0)
    order = np.argsort(rho)[::-1]
    rho, directions = rho[order], directions[:, order]
    blocks, block_metadata = make_blocks(rho)

    null_values = [[] for _ in blocks]
    rng = np.random.default_rng(stable_seed("persistence-null", *run_key))
    for _ in range(PERSISTENCE_PERMUTATIONS):
        permutation = rng.permutation(len(subjects))
        for label in classes:
            left, right = [], []
            for index, subject in enumerate(subjects):
                a = (subject, sessions[0], label)
                b = (subjects[permutation[index]], sessions[1], label)
                if a in centroids and b in centroids:
                    left.append(centroids[a]); right.append(centroids[b])
            if len(left) >= 3:
                aa, bb = np.asarray(left), np.asarray(right)
                aa -= aa.mean(0); bb -= bb.mean(0)
                covariance_null = (aa.T @ bb + bb.T @ aa) / (2.0 * len(aa))
                for block_index, block in enumerate(blocks):
                    null_values[block_index].append(float(np.mean(np.diag(directions[:, block].T @ covariance_null @ directions[:, block]))))
    support = []
    for block_index, block in enumerate(blocks):
        observed = float(np.mean(rho[block]))
        null = np.asarray(null_values[block_index], np.float64)
        null_p95 = float(np.quantile(null, 0.95)) if len(null) else float("inf")
        support.append({
            "block": block_index,
            "rho_G": observed,
            "null_mean": float(null.mean()) if len(null) else None,
            "null_p95": null_p95,
            "persistence_supported": bool(observed > null_p95),
            "dimensions": len(block),
        })
    return {
        "mean": mean.astype(np.float32),
        "whitener": whitener.astype(np.float32),
        "dewhitener": dewhitener.astype(np.float32),
        "directions": directions.astype(np.float32),
        "rho": rho.astype(np.float32),
        "blocks": blocks,
        "audit": {
            "nominal_embedding_dimension": 64,
            "numerical_rank": numerical_rank,
            "active_rank": active_rank,
            "whitening_error_max_abs": float(np.max(np.abs(whitened.T @ whitened / max(len(whitened) - 1, 1) - np.eye(active_rank)))),
            "block_metadata": block_metadata,
            "persistence_support": support,
            "subject_class_pairs": pair_count,
            "classes": classes,
            "sessions": sessions,
            "null_permutations": PERSISTENCE_PERMUTATIONS,
        },
    }


def bootstrap(values: Sequence[float], seed: int, draws: int) -> dict[str, Any]:
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "draws": draws, "n_subjects": 0}
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "draws": draws,
        "n_subjects": len(values),
    }


def stable_sample(meta: pd.DataFrame, subjects: Sequence[str], session: int, label: int, run_key: Sequence[Any], purpose: str) -> np.ndarray:
    frame = meta[
        meta.subject_id.astype(str).isin(set(map(str, subjects)))
        & (meta.session_id.astype(int) == int(session))
        & (meta.label.astype(int) == int(label))
    ]
    selected = []
    for subject, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        idx = group.index.to_numpy(np.int64)
        count = min(len(idx), PER_SUBJECT_SESSION_CLASS_CAP)
        rng = np.random.default_rng(stable_seed("descriptor-sample", *run_key, purpose, subject, session, label))
        selected.extend(np.sort(rng.choice(idx, size=count, replace=False)).tolist())
    return np.asarray(sorted(selected), np.int64)


def ce_for(x: np.ndarray, y: np.ndarray, pack) -> np.ndarray:
    _, probability = ridge_predict(x, pack)
    y = np.asarray(y, np.int64)
    return -np.log(np.clip(probability[np.arange(len(y)), y], EPS, 1.0))


def block_utility(h: np.ndarray, meta: pd.DataFrame, spectrum: Mapping[str, Any], block: Sequence[int], run_key: Sequence[Any], classes: int) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    absolute_by_subject: dict[str, list[float]] = {}
    excess_by_subject: dict[str, list[float]] = {}
    subjects = subject_sort(meta.subject_id.unique())
    sessions = sorted(meta.session_id.astype(int).unique())
    labels = sorted(meta.label.astype(int).unique())
    split_rows = []
    for inner in range(INNER_HALF_SPLITS):
        values = np.asarray(subjects, dtype=object)
        rng = np.random.default_rng(stable_seed("utility-inner-split", *run_key, inner))
        values = values[rng.permutation(len(values))]
        count = max(1, min(len(values) - 1, len(values) // 2))
        fit_subjects, evaluation_subjects = subject_sort(values[:count]), subject_sort(values[count:])
        fit_indices = np.concatenate([
            stable_sample(meta, fit_subjects, session, label, run_key, f"utility-fit-{inner}")
            for session in sessions for label in labels
        ])
        eval_indices = np.concatenate([
            stable_sample(meta, evaluation_subjects, session, label, run_key, f"utility-eval-{inner}")
            for session in sessions for label in labels
        ])
        fit_y = meta.iloc[fit_indices].label.to_numpy(np.int64)
        eval_y = meta.iloc[eval_indices].label.to_numpy(np.int64)
        # Preserve the manuscript MI-only Protected-selection convention:
        # selection uses the first 24 hidden dimensions. The final PEEH A/B/C
        # probes below are separately fit on all 64 dimensions as requested.
        intact_pack = ridge_fit(h[fit_indices], fit_y, classes, dimensions=24)
        protected_fit = erase(h[fit_indices], spectrum, block)
        protected_eval = erase(h[eval_indices], spectrum, block)
        protected_pack = ridge_fit(protected_fit, fit_y, classes, dimensions=24)
        active = np.arange(len(spectrum["rho"]), dtype=np.int64)
        candidates = np.setdiff1d(active, np.asarray(block, np.int64))
        random_packs = []
        for draw in range(UTILITY_RANDOM_DRAWS):
            ids = np.random.default_rng(stable_seed("utility-null", *run_key, inner, draw)).choice(candidates, size=len(block), replace=False)
            random_packs.append((ids, ridge_fit(erase(h[fit_indices], spectrum, ids), fit_y, classes, dimensions=24)))
        eval_frame = meta.iloc[eval_indices].reset_index(drop=True)
        for subject, group in eval_frame.groupby(eval_frame.subject_id.astype(str), sort=True):
            loc = group.index.to_numpy(np.int64)
            y = eval_y[loc]
            intact_ce = ce_for(h[eval_indices][loc], y, intact_pack)
            protected_ce = ce_for(protected_eval[loc], y, protected_pack)
            absolute_harm = float(np.mean(protected_ce - intact_ce))
            random_harms = []
            for ids, pack in random_packs:
                random_ce = ce_for(erase(h[eval_indices][loc], spectrum, ids), y, pack)
                random_harms.append(float(np.mean(random_ce - intact_ce)))
            absolute_by_subject.setdefault(str(subject), []).append(absolute_harm)
            excess_by_subject.setdefault(str(subject), []).append(absolute_harm - float(np.mean(random_harms)))
        split_rows.append({
            "inner_split": inner,
            "fit_subjects": fit_subjects,
            "evaluation_subjects": evaluation_subjects,
            "fit_rows": int(len(fit_indices)),
            "evaluation_rows": int(len(eval_indices)),
            "random_draws": UTILITY_RANDOM_DRAWS,
        })
    absolute = {subject: float(np.mean(values)) for subject, values in absolute_by_subject.items()}
    excess = {subject: float(np.mean(values)) for subject, values in excess_by_subject.items()}
    return absolute, excess, split_rows


def select_protected(h: np.ndarray, meta: pd.DataFrame, spectrum: Mapping[str, Any], run_key: Sequence[Any], classes: int) -> tuple[list[int], list[dict[str, Any]]]:
    selected_blocks = []
    rows = []
    for block_index, block in enumerate(spectrum["blocks"]):
        support = spectrum["audit"]["persistence_support"][block_index]
        absolute, excess, split_rows = block_utility(h, meta, spectrum, block, run_key, classes)
        absolute_boot = bootstrap(list(absolute.values()), stable_seed("utility-bootstrap-absolute", *run_key, block_index), UTILITY_BOOTSTRAP_DRAWS)
        excess_boot = bootstrap(list(excess.values()), stable_seed("utility-bootstrap-excess", *run_key, block_index), UTILITY_BOOTSTRAP_DRAWS)
        protected = bool(
            support["persistence_supported"]
            and absolute_boot["ci95"][0] is not None
            and absolute_boot["ci95"][0] > 0
            and excess_boot["ci95"][0] > 0
        )
        if protected:
            selected_blocks.append(block_index)
        rows.append({
            "block": block_index,
            "coordinates": block,
            "rank": len(block),
            "persistence_supported": bool(support["persistence_supported"]),
            "rho_G": support["rho_G"],
            "rho_null_p95": support["null_p95"],
            "absolute_CE_harm_mean": absolute_boot["mean"],
            "absolute_CE_harm_CI95": absolute_boot["ci95"],
            "excess_CE_harm_mean": excess_boot["mean"],
            "excess_CE_harm_CI95": excess_boot["ci95"],
            "protected": protected,
            "selection_subjects": len(absolute),
            "inner_half_splits": split_rows,
        })
    protected_union = sorted({coordinate for block_index in selected_blocks for coordinate in spectrum["blocks"][block_index]})
    return protected_union, rows


def evaluate_erasure(train_h: np.ndarray, train_y: np.ndarray, eval_h: np.ndarray, eval_y: np.ndarray, eval_subject: np.ndarray, spectrum: Mapping[str, Any], protected: Sequence[int], run_key: Sequence[Any], classes: int) -> list[dict[str, Any]]:
    intact_pack = ridge_fit(train_h, train_y, classes)
    intact_prediction, _ = ridge_predict(eval_h, intact_pack)
    protected_train = erase(train_h, spectrum, protected)
    protected_eval = erase(eval_h, spectrum, protected)
    protected_pack = ridge_fit(protected_train, train_y, classes)
    protected_prediction, _ = ridge_predict(protected_eval, protected_pack)
    subjects = subject_sort(np.unique(eval_subject))
    intact_ba = {}
    protected_ba = {}
    random_ba = {subject: [] for subject in subjects}
    for subject in subjects:
        mask = np.asarray(eval_subject).astype(str) == str(subject)
        intact_ba[subject] = balanced_accuracy(eval_y[mask], intact_prediction[mask], classes)
        protected_ba[subject] = balanced_accuracy(eval_y[mask], protected_prediction[mask], classes)
    active = np.arange(len(spectrum["rho"]), dtype=np.int64)
    for draw in range(ERASURE_RANDOM_DRAWS):
        if len(protected):
            ids = np.random.default_rng(stable_seed("peeh-random-erasure", *run_key, draw)).choice(active, size=len(protected), replace=False)
        else:
            ids = np.empty(0, dtype=np.int64)
        random_pack = ridge_fit(erase(train_h, spectrum, ids), train_y, classes)
        random_prediction, _ = ridge_predict(erase(eval_h, spectrum, ids), random_pack)
        for subject in subjects:
            mask = np.asarray(eval_subject).astype(str) == str(subject)
            random_ba[subject].append(balanced_accuracy(eval_y[mask], random_prediction[mask], classes))
    rows = []
    for subject in subjects:
        random_mean = float(np.mean(random_ba[subject]))
        h_protected = intact_ba[subject] - protected_ba[subject]
        h_random = intact_ba[subject] - random_mean
        rows.append({
            "subject_id": subject,
            "intact_BA": intact_ba[subject],
            "protected_erased_BA": protected_ba[subject],
            "random_erased_BA": random_mean,
            "protected_harm_pp": 100.0 * h_protected,
            "random_harm_pp": 100.0 * h_random,
            "PEEH_pp": 100.0 * (h_protected - h_random),
        })
    return rows


def cache_embeddings(runtime, task: str, model_name: str, fold: int, seed: int, train_bundle, train_raw, eval_bundle, eval_raw, mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, Any]:
    checkpoint = checkpoint_path(model_name, task, fold, seed)
    checkpoint_sha = sha256_file(checkpoint)
    path = embedding_cache_path(task, model_name, fold, seed)
    cached = load_embeddings(path, checkpoint_sha)
    if cached is not None:
        return cached
    model = build_model(runtime, model_name, task, checkpoint, device)
    train_h = infer_representations(model, train_raw, mean, std, device, model_name, task)
    eval_h = infer_representations(model, eval_raw, mean, std, device, model_name, task)
    train_meta, eval_meta = bundle_metadata(train_bundle), bundle_metadata(eval_bundle)
    metadata = {
        "task": task,
        "model": model_name,
        "fold": fold,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "normalizer": str(normalizer_path(task, fold)),
        "normalizer_sha256": sha256_file(normalizer_path(task, fold)),
        "representation": "64-d immediately before final classifier",
        "eval_mode": True,
        "bn_state_updates": False,
        "training": False,
    }
    save_embeddings(path, train_h, train_meta, eval_h, eval_meta, metadata)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return load_embeddings(path, checkpoint_sha)


def run_result_path(task: str, model_name: str, fold: int, seed: int) -> Path:
    return RUN_RESULTS / task / model_name / f"fold{fold}_seed{seed}.json"


def run_one(embeddings: Mapping[str, Any], task: str, model_name: str, fold: int, seed: int, classes: int) -> dict[str, Any]:
    key = ("PEEH-v1", task, model_name, fold, seed)
    train_meta = pd.DataFrame({
        "subject_id": embeddings["train_subject"].astype(str),
        "session_id": embeddings["train_session"].astype(int),
        "label": embeddings["train_y"].astype(int),
    })
    spectrum = build_spectrum(train_meta, embeddings["train_h"], key)
    protected, selection_rows = select_protected(embeddings["train_h"], train_meta, spectrum, key, classes)
    subject_rows = evaluate_erasure(
        embeddings["train_h"], embeddings["train_y"], embeddings["eval_h"], embeddings["eval_y"],
        embeddings["eval_subject"], spectrum, protected, key, classes,
    )
    for row in subject_rows:
        row.update({"task": task, "model": model_name, "fold": fold, "seed": seed, "protected_rank": len(protected)})
    frame = pd.DataFrame(subject_rows)
    return {
        "task": task,
        "model": model_name,
        "fold": fold,
        "seed": seed,
        "protected_coordinates": protected,
        "protected_rank": len(protected),
        "active_rank": len(spectrum["rho"]),
        "intact_BA": float(frame.intact_BA.mean()),
        "protected_erased_BA": float(frame.protected_erased_BA.mean()),
        "random_erased_BA": float(frame.random_erased_BA.mean()),
        "protected_harm_pp": float(frame.protected_harm_pp.mean()),
        "random_harm_pp": float(frame.random_harm_pp.mean()),
        "PEEH_pp": float(frame.PEEH_pp.mean()),
        "subject_rows": subject_rows,
        "spectrum_audit": spectrum["audit"],
        "block_selection": selection_rows,
        "checkpoint_sha256": embeddings["metadata"]["checkpoint_sha256"],
        "normalizer_sha256": embeddings["metadata"]["normalizer_sha256"],
        "no_training": True,
        "bn_state_updates": False,
    }


def protocol_payload(checkpoints: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    return {
        "schema": "PERSIST_EEG_LITEBN_TFFORMER_PEEH_V1",
        "models": list(MODELS),
        "tasks": list(TASKS),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "representation": "64-d immediately before final classifier",
        "selection_scope": "inner_train biological subjects only",
        "evaluation": {
            "OpenBMI": {"scope": "internal_heldout_diagnostic", "subjects": list(OPENBMI_EVAL), "session": "S2"},
            "WBCIC": {"scope": "true_outer_confirmation_already_accessed", "subjects": list(WBCIC_EVAL), "session": "S3"},
        },
        "method_source": str(REFERENCE),
        "method_source_sha256": sha256_file(REFERENCE),
        "whitening": "train-only full 64-d basis",
        "active_rank_max": ACTIVE_RANK_MAX,
        "eigengap_max_block_rank": MAX_BLOCK_RANK,
        "persistence_subject_correspondence_permutations": PERSISTENCE_PERMUTATIONS,
        "predictive_consequence_subject_half_splits": INNER_HALF_SPLITS,
        "utility_random_erasure_draws": UTILITY_RANDOM_DRAWS,
        "protected_rule": "persistence supported AND lower95 absolute CE harm > 0 AND lower95 excess CE harm > 0",
        "ridge": {"protected_selection_dimensions": 24, "final_erasure_dimensions": 64, "standardized": True, "alpha": RIDGE_ALPHA, "refit_after_each_erasure": True},
        "final_random_erasure_draws": ERASURE_RANDOM_DRAWS,
        "final_subject_bootstrap_draws": FINAL_BOOTSTRAP_DRAWS,
        "erasure": "canonical coordinates with inverse map; raw hidden coordinates are never directly zeroed",
        "no_retraining": True,
        "bn_eval_locked": True,
        "device": str(device),
        "checkpoints": checkpoints,
    }


def finalize(results: list[dict[str, Any]]) -> None:
    run_rows = [{key: value for key, value in result.items() if key not in ("subject_rows", "spectrum_audit", "block_selection")} for result in results]
    subject_rows = [row for result in results for row in result["subject_rows"]]
    raw_subject = pd.DataFrame(subject_rows)
    subject = raw_subject.groupby(["task", "model", "subject_id"], as_index=False).agg(
        intact_BA=("intact_BA", "mean"),
        protected_erased_BA=("protected_erased_BA", "mean"),
        random_erased_BA=("random_erased_BA", "mean"),
        protected_harm_pp=("protected_harm_pp", "mean"),
        random_harm_pp=("random_harm_pp", "mean"),
        PEEH_pp=("PEEH_pp", "mean"),
        repeated_runs=("PEEH_pp", "size"),
        protected_rank_mean=("protected_rank", "mean"),
        protected_rank_min=("protected_rank", "min"),
        protected_rank_max=("protected_rank", "max"),
    )
    summaries = []
    for (task, model_name), group in subject.groupby(["task", "model"], sort=False):
        run_group = pd.DataFrame(run_rows)
        run_group = run_group[(run_group.task == task) & (run_group.model == model_name)]
        boot = bootstrap(group.PEEH_pp.to_numpy(float), stable_seed("final-subject-bootstrap", task, model_name), FINAL_BOOTSTRAP_DRAWS)
        summaries.append({
            "task": task,
            "model": model_name,
            "evaluation_scope": "true_outer_confirmation" if task == "WBCIC_MI" else "internal_heldout_diagnostic",
            "n_biological_subjects": int(len(group)),
            "intact_BA": float(group.intact_BA.mean()),
            "protected_erased_BA": float(group.protected_erased_BA.mean()),
            "random_erased_BA": float(group.random_erased_BA.mean()),
            "protected_harm_pp": float(group.protected_harm_pp.mean()),
            "random_harm_pp": float(group.random_harm_pp.mean()),
            "PEEH_pp": float(group.PEEH_pp.mean()),
            "PEEH_CI95_L_pp": boot["ci95"][0],
            "PEEH_CI95_U_pp": boot["ci95"][1],
            "median_PEEH_pp": float(group.PEEH_pp.median()),
            "protected_rank_mean": float(run_group.protected_rank.mean()),
            "protected_rank_median": float(run_group.protected_rank.median()),
            "protected_rank_min": int(run_group.protected_rank.min()),
            "protected_rank_max": int(run_group.protected_rank.max()),
            "bootstrap_draws": FINAL_BOOTSTRAP_DRAWS,
        })
    summary = pd.DataFrame(summaries)
    write_csv(OUT / "SUBJECT_LEVEL_PEEH.csv", subject)
    write_csv(OUT / "RUN_LEVEL_PEEH.csv", run_rows)
    write_csv(OUT / "TASK_LEVEL_PEEH.csv", summary)
    selection_rows = []
    for result in results:
        for row in result["block_selection"]:
            selection_rows.append({
                "task": result["task"], "model": result["model"], "fold": result["fold"], "seed": result["seed"],
                **{key: value for key, value in row.items() if key != "inner_half_splits"},
            })
    write_csv(OUT / "PROTECTED_BLOCK_SELECTION.csv", selection_rows)

    lines = [
        "# LiteBN / TFFormer PEEH v1",
        "",
        "No neural model was retrained. Protected selection and all ridge probes use inner-train biological subjects only.",
        "OpenBMI rows use the 14-subject internal-heldout diagnostic cohort; WBCIC uses the 10-subject true-outer cohort.",
        "",
        "| Task | Model | Intact BA | Protected-erased BA | Random-erased BA | Protected harm | Random harm | PEEH [95% CI] | Protected rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        rank = f"{row.protected_rank_mean:.2f} [{int(row.protected_rank_min)}-{int(row.protected_rank_max)}]"
        lines.append(
            f"| {row.task} | {row.model} | {100*row.intact_BA:.3f}% | {100*row.protected_erased_BA:.3f}% | "
            f"{100*row.random_erased_BA:.3f}% | {row.protected_harm_pp:.3f} pp | {row.random_harm_pp:.3f} pp | "
            f"{row.PEEH_pp:.3f} [{row.PEEH_CI95_L_pp:.3f}, {row.PEEH_CI95_U_pp:.3f}] pp | {rank} |"
        )
    lines += [
        "",
        "PEEH > 0 means TRAIN-selected persistent Protected coordinates are more predictively consequential than equal-rank random coordinates.",
        "Larger PEEH does not imply a better classifier; model superiority remains determined by BA and Macro-F1.",
        "Folds and seeds were averaged within biological subject before the 20,000-draw subject bootstrap.",
    ]
    (OUT / "FINAL_PEEH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-runs", type=int, default=None, help="debug/resume limit; finalization requires all runs")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen representation extraction")
    device = torch.device("cuda")
    for directory in (OUT, PROTOCOL, RUNTIME, CACHE, RUN_RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime()
    _, folds_by_dataset, split_sha = runtime.base.load_folds()
    checkpoints = []
    for task in TASKS:
        for model_name in MODELS:
            for fold in FOLDS:
                for seed in SEEDS:
                    path = checkpoint_path(model_name, task, fold, seed)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    checkpoints.append({"task": task, "model": model_name, "fold": fold, "seed": seed, "path": str(path), "sha256": sha256_file(path)})
    write_json(PROTOCOL / "PROTOCOL_LOCK.json", {**protocol_payload(checkpoints, device), "fivefold_split_sha256": split_sha})
    write_csv(OUT / "CHECKPOINT_AUDIT.csv", checkpoints)

    completed_new = 0
    for task in TASKS:
        dataset = runtime.base.TASKS[task]["dataset"]
        classes = int(runtime.base.TASKS[task]["classes"])
        persistence_sessions = (1, 2) if dataset == "OpenBMI" else (0, 1)
        eval_subjects = OPENBMI_EVAL if dataset == "OpenBMI" else WBCIC_EVAL
        for fold_record in folds_by_dataset[dataset]:
            fold = int(fold_record["fold_id"])
            train_subjects = fold_record["inner_train_subjects"]
            train_full = runtime.base.build_bundle(task, train_subjects)
            train_bundle = subset_bundle(runtime, train_full, persistence_sessions)
            if dataset == "OpenBMI":
                eval_full = runtime.base.build_bundle(task, eval_subjects)
                eval_bundle = subset_bundle(runtime, eval_full, (2,))
            else:
                eval_bundle = true_outer_bundle(runtime)
            train_raw = runtime.base.RawGPUCache(train_bundle, device)
            eval_raw = runtime.base.RawGPUCache(eval_bundle, device)
            mean, std, _ = runtime.base.load_tensor_pair(normalizer_path(task, fold))
            for seed in SEEDS:
                for model_name in MODELS:
                    result_path = run_result_path(task, model_name, fold, seed)
                    if result_path.is_file():
                        continue
                    embeddings = cache_embeddings(runtime, task, model_name, fold, seed, train_bundle, train_raw, eval_bundle, eval_raw, mean, std, device)
                    result = run_one(embeddings, task, model_name, fold, seed, classes)
                    write_json(result_path, result)
                    completed_new += 1
                    print(
                        f"PEEH_RUN_DONE task={task} model={model_name} fold={fold} seed={seed} "
                        f"rank={result['protected_rank']} PEEH_pp={result['PEEH_pp']:.6f}",
                        flush=True,
                    )
                    if args.max_new_runs is not None and completed_new >= args.max_new_runs:
                        print("PEEH_DEBUG_LIMIT_REACHED", flush=True)
                        return
            del train_raw, eval_raw, train_bundle, eval_bundle, train_full
            if dataset == "OpenBMI":
                del eval_full
            gc.collect(); torch.cuda.empty_cache()

    result_files = [run_result_path(task, model, fold, seed) for task in TASKS for model in MODELS for fold in FOLDS for seed in SEEDS]
    if not all(path.is_file() for path in result_files):
        raise RuntimeError("not all 120 run results exist")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    finalize(results)
    write_json(OUT / "COMPLETION.json", {
        "status": "COMPLETE",
        "run_count": len(results),
        "models_retrained": False,
        "bn_state_updates": False,
        "evaluation_subjects_used_for_selection": False,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print("PEEH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
