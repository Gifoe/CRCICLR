from __future__ import annotations

"""PERSIST-EEG Experiment 3: matched identity-removal causal test.

The script deliberately keeps the data boundary explicit.  It reads only
development-train and development-validation subjects from the frozen
OpenBMI manifest, reuses the V3.1 EEGNet checkpoints/canonical spectra, and
never asks the split object for the outer-subject field.  All design decisions
are train-only and deterministic; validation outcomes are touched only in the
``final`` phase after ``freeze``.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXP_ROOT = Path(__file__).resolve().parents[1]
OUT = EXP_ROOT / "outputs"
FIGURES = EXP_ROOT / "figures"
CONFIG_PATH = EXP_ROOT / "PROTOCOL_FROZEN.json"

STAGE0_ROOT = Path(os.environ.get(
    "PERSIST_STAGE0_ROOT",
    r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full",
))
SIGNED_ROOT = Path(os.environ.get(
    "PERSIST_SIGNED_ROOT",
    r"D:\nips-temp\TotalP\P1\CRCICLR_INVARIANCE_RESCUE_V1\experiments\persist_eeg_p4_signed_v3_1",
))

if str(STAGE0_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE0_ROOT))
import p4_persist_ct as upstream  # noqa: E402


FOLDS = (0, 1, 2)
SEEDS = (0, 1)
DOSES = {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.75}
ALPHAS = np.linspace(0.0, 1.0, 21)
BOOTSTRAP_DRAWS = 10_000
MI_CLASSES = 2
MATCH_MAX = 50
MATCH_MIN = 20
# The structural features are correlated (for example, task-margin and
# dewhitened direction norm).  A hard per-feature z-score cutoff can therefore
# reject every legal candidate even when the joint train-only distance is
# small.  Matching is consequently a deterministic top-K ranking over the
# complete exact-rank candidate set.  The per-feature extrema remain in the
# diagnostics and are not hidden.
MATCH_SELECTION = "top_k_train_only_standardized_structural_distance"
IDENTITY_TOLERANCE = 0.01


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
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else float(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False) % (2**31 - 1)


def subject_sort(values: Iterable[object]) -> list[str]:
    def key(value: str) -> tuple[int, str]:
        digits = "".join(c for c in str(value) if c.isdigit())
        return (int(digits) if digits else 10**9, str(value))
    return sorted({str(v) for v in values}, key=key)


def ensure_dirs() -> None:
    for path in (OUT, FIGURES, OUT / "feature_cache", OUT / "train_only", OUT / "final"):
        path.mkdir(parents=True, exist_ok=True)


def flags() -> dict[str, bool]:
    return {"outer_test_used": False, "outer_membership_enumerated": False}


def load_frozen_config() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_VALIDATION_OUTCOME":
        raise RuntimeError("Experiment-3 protocol is not frozen")
    return payload


def load_development_splits() -> tuple[dict[int, dict[str, Any]], str]:
    """Read only train/validation fields; never extract outer membership."""
    path = STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    records: dict[int, dict[str, Any]] = {}
    for item in payload["openbmi"]["folds"]:
        fold = int(item["fold"])
        if fold not in FOLDS:
            continue
        train = subject_sort(item["train_subjects"])
        validation = subject_sort(item["validation_subjects"])
        if len(train) != 34 or len(validation) != 9 or set(train) & set(validation):
            raise RuntimeError(f"invalid development split fold={fold}")
        records[fold] = {"fold": fold, "train_subjects": train, "validation_subjects": validation}
    if set(records) != set(FOLDS):
        raise RuntimeError("missing development folds")
    return records, sha256_bytes(raw)


def load_development_manifest(subjects: Sequence[str]) -> pd.DataFrame:
    """Predicate-read only authorized development subjects from the manifest."""
    path = STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
    cols = ["subject_id", "session_id", "paradigm", "trial_id", "event_label",
            "sampling_rate", "n_channels", "n_times", "signal_cache_path", "cache_index"]
    frame = pd.read_parquet(path, columns=cols, filters=[("subject_id", "in", list(map(str, subjects)))])
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["session_id"] = frame.session_id.astype(int)
    frame["paradigm"] = frame.paradigm.astype(str)
    frame["event_label"] = frame.event_label.astype(str)
    if not set(frame.subject_id).issubset(set(map(str, subjects))):
        raise RuntimeError("manifest predicate returned unauthorized subject")
    if set(frame.paradigm) != {"mi", "erp", "ssvep"}:
        raise RuntimeError("development manifest is not OpenBMI complete")
    # Sorting is deterministic and preserves the semantic rows; the canonical
    # spectrum itself is read from the V3.1 artifact and is not rebuilt here.
    frame = frame.sort_values(["subject_id", "session_id", "paradigm", "trial_id"]).reset_index(drop=True)
    frame["global_index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def phase0() -> dict[str, Any]:
    ensure_dirs()
    splits, split_sha = load_development_splits()
    rows = []
    for fold in FOLDS:
        allowed = splits[fold]["train_subjects"] + splits[fold]["validation_subjects"]
        frame = load_development_manifest(allowed)
        rows.append({
            "fold": fold,
            "train_subjects": len(splits[fold]["train_subjects"]),
            "validation_subjects": len(splits[fold]["validation_subjects"]),
            "materialized_subjects": len(frame.subject_id.unique()),
            "materialized_rows": len(frame),
            "paradigms": sorted(frame.paradigm.unique().tolist()),
            "sessions": sorted(map(int, frame.session_id.unique())),
            "channels": sorted(map(int, frame.n_channels.unique())),
            "sampling_rates": sorted(map(float, frame.sampling_rate.unique())),
            "outer_test_used": False,
            "outer_membership_enumerated": False,
        })
    payload = {
        "dataset": "OpenBMI_nm000273_MI",
        "stage0_root": str(STAGE0_ROOT),
        "manifest_path": str(STAGE0_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"),
        "split_path": str(STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"),
        "split_file_sha256": split_sha,
        "folds": rows,
        "authorized_scope": "development_train_and_development_validation_only",
        **flags(),
    }
    write_json(OUT / "DATA_ACCESS_AUDIT.json", payload)
    return payload


def run_paths(fold: int, seed: int) -> dict[str, Path]:
    run = SIGNED_ROOT / "results_v3_1" / "runs" / f"fold-{fold}" / f"seed-{seed}"
    return {
        "run": run,
        "spectrum": run / "spectrum" / "PERSISTENCE_SPECTRUM.npz",
        "fingerprint": run / "spectrum" / "PERSISTENCE_SPECTRUM_FINGERPRINT.json",
        "assignment": run / "SIGNED_ASSIGNMENTS_V3_1.json",
        "provenance": run / "RUN_PROVENANCE.json",
        "features": OUT / "feature_cache" / f"fold-{fold}_seed-{seed}.npz",
        "feature_meta": OUT / "feature_cache" / f"fold-{fold}_seed-{seed}.json",
    }


def extract_or_load_features(fold: int, seed: int, split: Mapping[str, Any], manifest: pd.DataFrame, device: Any) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    paths = run_paths(fold, seed)
    if paths["features"].exists() and paths["feature_meta"].exists():
        meta = json.loads(paths["feature_meta"].read_text(encoding="utf-8"))
        if meta.get("checkpoint_sha256") == sha256_file(Path(meta["checkpoint"])):
            z = np.load(paths["features"], allow_pickle=False)
            ntr = int(z["n_train"].item())
            train_meta = pd.DataFrame({
                "subject_id": z["train_subject_id"].astype(str),
                "session_id": z["train_session_id"].astype(int),
                "paradigm": z["train_paradigm"].astype(str),
                "event_label": z["train_event_label"].astype(str),
            })
            val_meta = pd.DataFrame({
                "subject_id": z["val_subject_id"].astype(str),
                "session_id": z["val_session_id"].astype(int),
                "paradigm": z["val_paradigm"].astype(str),
                "event_label": z["val_event_label"].astype(str),
            })
            return train_meta, z["train_features"].astype(np.float32), val_meta, z["val_features"].astype(np.float32), meta

    ckpt, mean, std = upstream.historical(fold, seed)
    model = upstream.load_model(ckpt, manifest, device)
    train_meta, train_features, _ = upstream.extract(model, manifest, split["train_subjects"], mean, std, device, 190000 + fold * 101 + seed, cap=0)
    val_meta, val_features, _ = upstream.extract(model, manifest, split["validation_subjects"], mean, std, device, 200000 + fold * 101 + seed, cap=0)
    checkpoint_sha = sha256_file(ckpt)
    meta = {
        "fold": fold, "seed": seed, "checkpoint": str(ckpt), "checkpoint_sha256": checkpoint_sha,
        "train_rows": len(train_meta), "validation_rows": len(val_meta), "embedding_dim": int(train_features.shape[1]),
        "manifest_subjects": int(manifest.subject_id.nunique()), **flags(),
    }
    paths["features"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["features"].with_suffix(".part.npz")
    np.savez_compressed(tmp,
        n_train=np.asarray(len(train_meta), dtype=np.int64),
        train_features=train_features.astype(np.float32),
        val_features=val_features.astype(np.float32),
        train_subject_id=np.asarray(train_meta.subject_id.astype(str).tolist(), dtype="<U32"),
        train_session_id=train_meta.session_id.astype(np.int64).to_numpy(),
        train_paradigm=np.asarray(train_meta.paradigm.astype(str).tolist(), dtype="<U16"),
        train_event_label=np.asarray(train_meta.event_label.astype(str).tolist(), dtype="<U64"),
        val_subject_id=np.asarray(val_meta.subject_id.astype(str).tolist(), dtype="<U32"),
        val_session_id=val_meta.session_id.astype(np.int64).to_numpy(),
        val_paradigm=np.asarray(val_meta.paradigm.astype(str).tolist(), dtype="<U16"),
        val_event_label=np.asarray(val_meta.event_label.astype(str).tolist(), dtype="<U64"),
        outer_test_used=np.asarray(False), outer_membership_enumerated=np.asarray(False))
    os.replace(tmp, paths["features"])
    write_json(paths["feature_meta"], meta)
    return train_meta, train_features.astype(np.float32), val_meta, val_features.astype(np.float32), meta


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_upstream_run(fold: int, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = run_paths(fold, seed)
    if not all(paths[k].exists() for k in ("spectrum", "fingerprint", "assignment", "provenance")):
        raise FileNotFoundError(f"missing Signed-V3.1 artifacts fold={fold} seed={seed}")
    z = np.load(paths["spectrum"], allow_pickle=False)
    spec = {k: z[k].astype(np.float32) for k in ("mean", "whitener", "dewhitener", "directions", "rho")}
    spec["blocks"] = json.loads(str(z["blocks_json"].item()))
    spec["audit"] = json.loads(str(z["audit_json"].item()))
    assignment = json.loads(paths["assignment"].read_text(encoding="utf-8"))
    fingerprint = json.loads(paths["fingerprint"].read_text(encoding="utf-8"))
    for key in ("mean", "whitener", "dewhitener", "directions", "rho"):
        if fingerprint["arrays"][key] != array_sha(spec[key]):
            raise RuntimeError(f"V3.1 spectrum fingerprint mismatch: {key}")
    # V3.1's canonical spectrum predates the explicit outer-lock keys.  A
    # missing key is treated as the historical false value; an explicit true
    # value is rejected.  The experiment's own artifacts always emit both
    # keys explicitly.
    if spec["audit"].get("outer_test_used", False) is not False or spec["audit"].get("outer_membership_enumerated", False) is not False:
        raise RuntimeError("upstream spectrum violates outer lock")
    if not assignment.get("mi") or not isinstance(assignment["mi"].get("protected"), list):
        raise RuntimeError("missing frozen MI Protected assignment")
    if any(int(b) < 0 or int(b) >= len(spec["blocks"]) for b in assignment["mi"]["protected"]):
        raise RuntimeError("Protected block ID outside canonical spectrum")
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    return spec, assignment, {"fingerprint": fingerprint, "provenance": provenance, "paths": {k: str(v) for k, v in paths.items()}}


def coordinates(features: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - spec["mean"]) @ spec["whitener"] @ spec["directions"]


def suppress_coordinates(features: np.ndarray, q: np.ndarray, spec: Mapping[str, Any], dims: Sequence[int], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(sorted(set(map(int, dims))), dtype=np.int64)
    delta = np.zeros_like(q, dtype=np.float64)
    delta[:, selected] = -float(alpha) * q[:, selected]
    delta_h = (delta @ spec["directions"].T) @ spec["dewhitener"]
    return (np.asarray(features, dtype=np.float64) + delta_h).astype(np.float32), (q + delta).astype(np.float32)


def ridge_pack(X: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(X, dtype=np.float64)
    yy = np.asarray(y, dtype=np.int64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = np.c_[(x - mu) / sd, np.ones(len(x))]
    penalty = np.eye(z.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    classes = int(np.max(yy)) + 1
    target = np.eye(classes, dtype=np.float64)[yy]
    w = np.linalg.pinv(z.T @ z + float(alpha) * penalty) @ z.T @ target
    return w, mu, sd


def ridge_predict(X: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    w, mu, sd = pack
    z = np.c_[(np.asarray(X, dtype=np.float64) - mu) / sd, np.ones(len(X))]
    logits = z @ w
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs.argmax(axis=1), probs


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.asarray(y_true, dtype=np.int64))
    return float(np.mean([np.mean(y_pred[y_true == k] == k) for k in labels])) if len(labels) else float("nan")


def mi_label(meta: pd.DataFrame) -> np.ndarray:
    labels = {name: i for i, name in enumerate(sorted(meta.loc[meta.paradigm == "mi", "event_label"].astype(str).unique()))}
    return meta.event_label.astype(str).map(labels).to_numpy(dtype=np.int64)


def subject_id_score(meta: pd.DataFrame, representation: np.ndarray, subjects: Sequence[str], return_per_subject: bool = False) -> tuple[float, dict[str, float]]:
    allowed = set(map(str, subjects))
    train_mask = meta.subject_id.astype(str).isin(allowed).to_numpy() & (meta.session_id.to_numpy() == 1)
    eval_mask = meta.subject_id.astype(str).isin(allowed).to_numpy() & (meta.session_id.to_numpy() == 2)
    ordered = subject_sort(allowed)
    code = {s: i for i, s in enumerate(ordered)}
    y_train = meta.loc[train_mask, "subject_id"].astype(str).map(code).to_numpy(dtype=np.int64)
    y_eval = meta.loc[eval_mask, "subject_id"].astype(str).map(code).to_numpy(dtype=np.int64)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
        return float("nan"), {}
    pack = ridge_pack(representation[train_mask], y_train)
    pred, _ = ridge_predict(representation[eval_mask], pack)
    per: dict[str, float] = {}
    eval_subject = meta.loc[eval_mask, "subject_id"].astype(str).to_numpy()
    for s in ordered:
        mask = eval_subject == s
        per[s] = float(np.mean(pred[mask] == code[s])) if np.any(mask) else float("nan")
    return float(np.nanmean(list(per.values()))), per


def task_score(train_meta: pd.DataFrame, train_h: np.ndarray, val_meta: pd.DataFrame, val_h: np.ndarray, fit_subjects: Sequence[str], eval_subjects: Sequence[str]) -> tuple[float, dict[str, float]]:
    train_mask = (train_meta.paradigm == "mi").to_numpy() & train_meta.subject_id.astype(str).isin(set(map(str, fit_subjects))).to_numpy()
    eval_mask = (val_meta.paradigm == "mi").to_numpy() & (val_meta.session_id.to_numpy() == 2) & val_meta.subject_id.astype(str).isin(set(map(str, eval_subjects))).to_numpy()
    y_train = mi_label(train_meta)[train_mask]
    y_eval = mi_label(val_meta)[eval_mask]
    pack = ridge_pack(train_h[train_mask], y_train)
    pred, _ = ridge_predict(val_h[eval_mask], pack)
    per: dict[str, float] = {}
    eval_subject = val_meta.loc[eval_mask, "subject_id"].astype(str).to_numpy()
    for s in subject_sort(eval_subjects):
        mask = eval_subject == s
        per[s] = balanced_accuracy(y_eval[mask], pred[mask]) if np.any(mask) else float("nan")
    return float(np.nanmean(list(per.values()))), per


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def persistence_rows(val_meta: pd.DataFrame, val_q: np.ndarray, subjects: Sequence[str], fold: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = subject_sort(subjects)
    shifted = {s: ordered[(i + 1) % len(ordered)] for i, s in enumerate(ordered)}
    for s in ordered:
        per_class = []
        for label in sorted(val_meta.loc[val_meta.paradigm == "mi", "event_label"].astype(str).unique()):
            a = (val_meta.subject_id.astype(str) == s).to_numpy() & (val_meta.session_id.to_numpy() == 1) & (val_meta.paradigm == "mi").to_numpy() & (val_meta.event_label.astype(str) == label).to_numpy()
            b = (val_meta.subject_id.astype(str) == s).to_numpy() & (val_meta.session_id.to_numpy() == 2) & (val_meta.paradigm == "mi").to_numpy() & (val_meta.event_label.astype(str) == label).to_numpy()
            m = (val_meta.subject_id.astype(str) == shifted[s]).to_numpy() & (val_meta.session_id.to_numpy() == 2) & (val_meta.paradigm == "mi").to_numpy() & (val_meta.event_label.astype(str) == label).to_numpy()
            if not (np.any(a) and np.any(b) and np.any(m)):
                continue
            ca, cb, cm = val_q[a].mean(axis=0), val_q[b].mean(axis=0), val_q[m].mean(axis=0)
            per_class.append((cosine(ca, cb), cosine(ca, cm)))
        if per_class:
            same = float(np.mean([x[0] for x in per_class])); mismatch = float(np.mean([x[1] for x in per_class]))
            rows.append({"fold": fold, "seed": seed, "subject_id": s, "same_similarity": same, "mismatched_similarity": mismatch, "R_persist": same - mismatch, "mismatch_subject": shifted[s], **flags()})
    return rows


def group_metrics(meta: pd.DataFrame, q: np.ndarray, spec: Mapping[str, Any], dims: Sequence[int], train_subjects: Sequence[str]) -> dict[str, float]:
    d = np.asarray(sorted(set(map(int, dims))), dtype=np.int64)
    x = q[:, d]
    rho = np.asarray(spec["rho"], dtype=float)[d]
    train_mask = (meta.paradigm == "mi").to_numpy() & meta.subject_id.astype(str).isin(set(map(str, train_subjects))).to_numpy()
    id_ba, _ = subject_id_score(meta.loc[train_mask].reset_index(drop=True), x[train_mask], train_subjects)
    y = mi_label(meta)[train_mask]
    task_pack = ridge_pack(x[train_mask], y)
    margin = float(np.linalg.norm(task_pack[0][:-1], axis=0).mean())
    dewhite = np.asarray(spec["directions"], float)[:, d].T @ np.asarray(spec["dewhitener"], float)
    norms = np.linalg.norm(dewhite, axis=1)
    variances = np.var(x[train_mask], axis=0)
    energy = np.mean(np.square(x[train_mask]), axis=0)
    return {
        "rank": float(len(d)), "rho_mean": float(np.mean(rho)), "rho_std": float(np.std(rho)),
        "coordinate_variance_mean": float(np.mean(variances)), "coordinate_energy_mean": float(np.mean(energy)),
        "train_identity_balanced_accuracy": float(id_ba), "dewhitened_direction_norm_mean": float(np.mean(norms)),
        "train_task_margin_magnitude": margin,
    }


MATCH_FEATURES = ["rho_mean", "rho_std", "coordinate_variance_mean", "coordinate_energy_mean", "train_identity_balanced_accuracy", "dewhitened_direction_norm_mean", "train_task_margin_magnitude"]


def candidate_sets(pool: Sequence[int], rank: int, fold: int, seed: int) -> list[tuple[int, ...]]:
    pool = sorted(map(int, pool))
    count = math.comb(len(pool), rank) if len(pool) >= rank else 0
    if count <= 50_000:
        return list(itertools.combinations(pool, rank))
    rng = np.random.default_rng(stable_seed("exp3-candidates", fold, seed, rank, *pool))
    seen: set[tuple[int, ...]] = set()
    target = min(50_000, count)
    while len(seen) < target:
        seen.add(tuple(sorted(rng.choice(pool, size=rank, replace=False).tolist())))
    return sorted(seen)


def match_controls(train_meta: pd.DataFrame, train_q: np.ndarray, spec: Mapping[str, Any], protected_blocks: Sequence[int], fold: int, seed: int, train_subjects: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    protected = sorted(set(sum((list(map(int, spec["blocks"][int(b)])) for b in protected_blocks), [])))
    rank = len(protected)
    supported = [int(row["block"]) for row in spec["audit"].get("persistence_support", []) if bool(row.get("persistence_supported"))]
    pool = sorted(set(sum((list(map(int, spec["blocks"][b])) for b in supported), [])) - set(protected))
    if len(pool) < rank:
        return pd.DataFrame(), pd.DataFrame(), {"status": "MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE", "protected": protected, "pool": pool, "rank": rank}
    p_metrics = group_metrics(train_meta, train_q, spec, protected, train_subjects)
    candidates = candidate_sets(pool, rank, fold, seed)
    candidate_rows = []
    for dims in candidates:
        m = group_metrics(train_meta, train_q, spec, dims, train_subjects)
        candidate_rows.append({"coordinate_ids": json.dumps(list(dims)), **m})
    cand = pd.DataFrame(candidate_rows)
    all_values = cand[MATCH_FEATURES].to_numpy(float)
    scale = np.nanstd(np.vstack([all_values, np.asarray([[p_metrics[f] for f in MATCH_FEATURES]], float)]), axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    target = np.asarray([p_metrics[f] for f in MATCH_FEATURES], float)
    z = (cand[MATCH_FEATURES].to_numpy(float) - target[None, :]) / scale[None, :]
    cand["match_distance"] = np.sqrt(np.nanmean(z * z, axis=1))
    cand["match_max_abs_z"] = np.nanmax(np.abs(z), axis=1)
    cand = cand.sort_values(["match_distance", "match_max_abs_z", "coordinate_ids"], kind="mergesort").reset_index(drop=True)
    accepted = cand.head(MATCH_MAX).copy()
    accepted_distance_max = float(accepted.match_distance.max()) if len(accepted) else float("nan")
    accepted_max_abs_z_max = float(accepted.match_max_abs_z.max()) if len(accepted) else float("nan")
    diagnostics = [{"fold": fold, "seed": seed, "protected_block_ids": json.dumps(list(map(int, protected_blocks))), "protected_coordinate_ids": json.dumps(protected), "candidate_count": len(cand), "accepted_count": len(accepted), "protected_rank": rank, "pool_size": len(pool), "accepted_distance_max": accepted_distance_max, "accepted_max_abs_z_max": accepted_max_abs_z_max, "matching_selection": MATCH_SELECTION, "protected_" + key: value, **flags()} for key, value in p_metrics.items()]
    # The repeated rows above are convenient for long-format diagnostics but
    # the actual table below has one summary row.
    diag_row = {"fold": fold, "seed": seed, "protected_block_ids": json.dumps(list(map(int, protected_blocks))), "protected_coordinate_ids": json.dumps(protected), "candidate_count": len(cand), "accepted_count": len(accepted), "protected_rank": rank, "pool_size": len(pool), "accepted_distance_max": accepted_distance_max, "accepted_max_abs_z_max": accepted_max_abs_z_max, "matching_selection": MATCH_SELECTION, **{"protected_" + key: value for key, value in p_metrics.items()}, **flags()}
    selected_rows = []
    for index, row in accepted.iterrows():
        selected_rows.append({"fold": fold, "seed": seed, "control_id": f"N{len(selected_rows)+1:03d}", "protected_block_ids": json.dumps(list(map(int, protected_blocks))), "protected_coordinate_ids": json.dumps(protected), "coordinate_ids": row.coordinate_ids, "rank": rank, "match_distance": float(row.match_distance), "match_max_abs_z": float(row.match_max_abs_z), **{f"N_{key}": float(row[key]) for key in MATCH_FEATURES}, **{f"P_{key}": float(p_metrics[key]) for key in MATCH_FEATURES}, **flags()})
    selected = pd.DataFrame(selected_rows)
    diagnostics_frame = pd.DataFrame([diag_row])
    payload = {"fold": fold, "seed": seed, "protected_block_ids": list(map(int, protected_blocks)), "protected_coordinate_ids": protected, "rank": rank, "pool": pool, "candidate_count": len(cand), "accepted_count": len(accepted), "accepted_distance_max": accepted_distance_max, "accepted_max_abs_z_max": accepted_max_abs_z_max, "controls": [json.loads(x) for x in accepted.coordinate_ids.tolist()], "matching_rule": MATCH_SELECTION, "matching_features": list(MATCH_FEATURES), **flags()}
    return selected, diagnostics_frame, payload


def interpolate_alpha(curve: pd.DataFrame, target_drop: float) -> tuple[float, float]:
    drops = curve["drop"].to_numpy(float)
    alphas = curve.alpha.to_numpy(float)
    finite = np.isfinite(drops)
    if not np.any(finite):
        return float("nan"), float("nan")
    distance = np.abs(drops[finite] - float(target_drop))
    idx = np.flatnonzero(finite)[int(np.argmin(distance))]
    return float(alphas[idx]), float(drops[idx])


def identity_curve(meta: pd.DataFrame, q: np.ndarray, dims: Sequence[int], subjects: Sequence[str], fold: int, seed: int, group_id: str) -> pd.DataFrame:
    baseline, _ = subject_id_score(meta, q, subjects)
    rows = []
    for alpha in ALPHAS:
        q2 = q.copy()
        q2[:, np.asarray(dims, dtype=np.int64)] *= (1.0 - float(alpha))
        score, _ = subject_id_score(meta, q2, subjects)
        rows.append({"fold": fold, "seed": seed, "group_id": group_id, "alpha": float(alpha), "identity_BA": float(score), "identity_drop": float(baseline - score), "baseline_identity_BA": float(baseline), **flags()})
    return pd.DataFrame(rows).rename(columns={"identity_drop": "drop"})


def build_train_only() -> dict[str, Any]:
    ensure_dirs(); config = load_frozen_config(); splits, split_sha = load_development_splits()
    # Importing torch is delayed until this phase so phase0 remains a cheap
    # audit.  The upstream loader is deterministic and uses the server GPU.
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Experiment 3 requires the server CUDA environment for V3.1 feature extraction")
    all_controls, all_diags, all_curves, all_params, provenance = [], [], [], [], []
    failed_runs: list[dict[str, Any]] = []
    eligible_runs: list[dict[str, int]] = []
    for fold in FOLDS:
        allowed = splits[fold]["train_subjects"] + splits[fold]["validation_subjects"]
        manifest = load_development_manifest(allowed)
        train_meta, train_h, _, _, feat_prov = (None, None, None, None, None)
        for seed in SEEDS:
            train_meta, train_h, val_meta, val_h, feat_prov = extract_or_load_features(fold, seed, splits[fold], manifest, device)
            spec, assignment, upstream_prov = load_upstream_run(fold, seed)
            train_q = coordinates(train_h, spec).astype(np.float32)
            mi_train_mask = (train_meta.paradigm == "mi").to_numpy()
            mi_meta = train_meta.loc[mi_train_mask].reset_index(drop=True)
            mi_q = train_q[mi_train_mask]
            protected_blocks = list(map(int, assignment["mi"]["protected"]))
            controls, diagnostics, control_payload = match_controls(mi_meta, mi_q, spec, protected_blocks, fold, seed, splits[fold]["train_subjects"])
            write_csv(OUT / "train_only" / f"MATCHED_CONTROLS_FOLD_{fold}_SEED_{seed}.csv", controls)
            write_json(OUT / "train_only" / f"MATCHED_CONTROLS_FOLD_{fold}_SEED_{seed}.json", control_payload)
            if len(controls):
                all_controls.append(controls)
            all_diags.append(diagnostics)
            protected_dims = sorted(set(sum((list(map(int, spec["blocks"][b])) for b in protected_blocks), [])))
            run_provenance = {"fold": fold, "seed": seed, "protected_block_ids": protected_blocks, "protected_coordinate_ids": protected_dims, "spectrum_fingerprint": upstream_prov["fingerprint"], "spectrum_path": upstream_prov["paths"]["spectrum"], "assignment_path": upstream_prov["paths"]["assignment"], "checkpoint": feat_prov["checkpoint"], "checkpoint_sha256": feat_prov["checkpoint_sha256"], "train_rows": len(train_meta), "validation_rows": len(val_meta), "accepted_controls": len(controls), **flags()}
            if len(controls) < MATCH_MIN:
                run_provenance["matching_status"] = "MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE"
                provenance.append(run_provenance)
                failed_runs.append({"fold": fold, "seed": seed, "accepted_controls": len(controls), "required_controls": MATCH_MIN, "protected_block_ids": protected_blocks, "protected_rank": len(protected_dims), "pool_size": int(control_payload.get("pool", []) and len(control_payload["pool"]) or 0), "candidate_count": int(control_payload.get("candidate_count", 0)), "reason": "frozen non-Protected persistence-supported pool has fewer than the required 20 exact-rank controls", **flags()})
                continue
            run_provenance["matching_status"] = "OK"
            provenance.append(run_provenance)
            eligible_runs.append({"fold": fold, "seed": seed, "controls": len(controls)})
            p_curve = identity_curve(mi_meta, mi_q, protected_dims, splits[fold]["train_subjects"], fold, seed, "P")
            all_curves.append(p_curve)
            n_curves: dict[str, pd.DataFrame] = {}
            for _, row in controls.iterrows():
                dims = json.loads(row.coordinate_ids)
                c = identity_curve(mi_meta, mi_q, dims, splits[fold]["train_subjects"], fold, seed, str(row.control_id))
                n_curves[str(row.control_id)] = c
                all_curves.append(c)
            p_max = float(np.nanmax(p_curve["drop"]))
            n_maxes = {key: float(np.nanmax(value["drop"])) for key, value in n_curves.items()}
            common_max = float(min([p_max] + list(n_maxes.values())))
            if not np.isfinite(common_max) or common_max <= 0:
                raise RuntimeError(f"IDENTITY_MATCH_FAILED fold={fold} seed={seed}: common train range={common_max}")
            for dose, fraction in DOSES.items():
                target = common_max * float(fraction)
                alpha_p, drop_p = interpolate_alpha(p_curve, target)
                all_params.append({"fold": fold, "seed": seed, "group_id": "P", "dose": dose, "target_drop_train": target, "alpha": alpha_p, "drop_train": drop_p, "common_max_drop_train": common_max, **flags()})
                for control_id, curve in n_curves.items():
                    alpha_n, drop_n = interpolate_alpha(curve, target)
                    all_params.append({"fold": fold, "seed": seed, "group_id": control_id, "dose": dose, "target_drop_train": target, "alpha": alpha_n, "drop_train": drop_n, "common_max_drop_train": common_max, **flags()})
    controls_frame = pd.concat(all_controls, ignore_index=True) if all_controls else pd.DataFrame()
    diags_frame = pd.concat(all_diags, ignore_index=True)
    curves_frame = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    params_frame = pd.DataFrame(all_params)
    write_csv(OUT / "MATCHED_CONTROL_TABLE.csv", controls_frame)
    write_csv(OUT / "MATCHING_DIAGNOSTICS.csv", diags_frame)
    write_csv(OUT / "IDENTITY_RESPONSE_CURVES.csv", curves_frame)
    write_csv(OUT / "IDENTITY_MATCHING_PARAMETERS.csv", params_frame)
    write_json(OUT / "MATCHED_CONTROL_PROVENANCE.json", {"runs": provenance, "failed_runs": failed_runs, "eligible_runs": eligible_runs, "minimum_controls": MATCH_MIN, "maximum_controls": MATCH_MAX, "matching_selection": MATCH_SELECTION, "matching_distance_cutoff": None, **flags()})
    if failed_runs:
        payload = {"status": "MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE", "runs": len(provenance), "eligible_runs": eligible_runs, "failed_runs": failed_runs, "controls_per_run_min": int(diags_frame.accepted_count.min()) if len(diags_frame) else 0, "common_train_identity_range_min": float(params_frame.common_max_drop_train.min()) if len(params_frame) else None, "config_sha256": sha256_file(CONFIG_PATH), "split_sha256": split_sha, "device": str(device), "validation_outcome_used": False, **flags()}
    else:
        payload = {"status": "TRAIN_ONLY_DESIGN_READY", "runs": len(provenance), "controls_per_run_min": int(controls_frame.groupby(["fold", "seed"]).control_id.nunique().min()), "common_train_identity_range_min": float(params_frame.common_max_drop_train.min()), "config_sha256": sha256_file(CONFIG_PATH), "split_sha256": split_sha, "device": str(device), "validation_outcome_used": False, **flags()}
    write_json(OUT / "TRAIN_ONLY_DESIGN.json", payload)
    return payload


def freeze() -> dict[str, Any]:
    design_path = OUT / "TRAIN_ONLY_DESIGN.json"
    if not design_path.exists():
        raise RuntimeError("run train_only before freeze")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("status") != "TRAIN_ONLY_DESIGN_READY":
        raise RuntimeError(f"cannot freeze terminal train-only design: {design.get('status')}")
    code_hashes = {str(p.relative_to(EXP_ROOT)): sha256_file(p) for p in sorted((EXP_ROOT / "code").glob("*.py"))}
    payload = {"status": "FROZEN_BEFORE_FINAL_VALIDATION", "protocol_sha256": sha256_file(CONFIG_PATH), "code_sha256": code_hashes, "design_sha256": sha256_file(design_path), "frozen_features": ["matching_features", "candidate_rule", "identity_metric", "alpha_grid", "dose_fractions", "identity_tolerance", "task_probe", "bootstrap", "gates"], "validation_outcome_used": False, **flags()}
    path = OUT / "PROTOCOL_FREEZE_RECORD.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != payload["protocol_sha256"] or old.get("code_sha256") != payload["code_sha256"]:
            raise RuntimeError("post-freeze protocol/code differs")
        return old
    write_json(path, payload)
    return payload


def require_frozen() -> dict[str, Any]:
    path = OUT / "PROTOCOL_FREEZE_RECORD.json"
    if not path.exists():
        raise RuntimeError("run freeze before final")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_FINAL_VALIDATION":
        raise RuntimeError("invalid freeze record")
    return payload


def lookup_params(params: pd.DataFrame, fold: int, seed: int, group_id: str, dose: str) -> tuple[float, float]:
    row = params[(params.fold == fold) & (params.seed == seed) & (params.group_id.astype(str) == str(group_id)) & (params.dose == dose)]
    if len(row) != 1:
        raise RuntimeError(f"missing alpha assignment {fold}/{seed}/{group_id}/{dose}")
    return float(row.iloc[0].alpha), float(row.iloc[0].drop_train)


def run_final() -> dict[str, Any]:
    freeze_record = require_frozen(); ensure_dirs(); splits, split_sha = load_development_splits()
    params = pd.read_csv(OUT / "IDENTITY_MATCHING_PARAMETERS.csv")
    controls = pd.read_csv(OUT / "MATCHED_CONTROL_TABLE.csv")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Experiment 3 final requires CUDA server")
    persistence_rows_all, id_rows_all, task_rows_all, dose_rows_all, random_rows_all, provenance = [], [], [], [], [], []
    for fold in FOLDS:
        allowed = splits[fold]["train_subjects"] + splits[fold]["validation_subjects"]
        manifest = load_development_manifest(allowed)
        for seed in SEEDS:
            train_meta, train_h, val_meta, val_h, feat_prov = extract_or_load_features(fold, seed, splits[fold], manifest, device)
            spec, assignment, upstream_prov = load_upstream_run(fold, seed)
            train_q = coordinates(train_h, spec).astype(np.float32); val_q = coordinates(val_h, spec).astype(np.float32)
            persistence_rows_all.extend(persistence_rows(val_meta, val_q, splits[fold]["validation_subjects"], fold, seed))
            mi_train = (train_meta.paradigm == "mi").to_numpy(); mi_val = (val_meta.paradigm == "mi").to_numpy()
            trm, trh, trq = train_meta.loc[mi_train].reset_index(drop=True), train_h[mi_train], train_q[mi_train]
            vam, vah, vaq = val_meta.loc[mi_val].reset_index(drop=True), val_h[mi_val], val_q[mi_val]
            protected_blocks = list(map(int, assignment["mi"]["protected"]))
            protected_dims = sorted(set(sum((list(map(int, spec["blocks"][b])) for b in protected_blocks), [])))
            run_controls = controls[(controls.fold == fold) & (controls.seed == seed)].copy()
            if len(run_controls) < MATCH_MIN:
                raise RuntimeError(f"G1 controls unexpectedly missing fold={fold} seed={seed}")
            # Baseline identity and task scores use exactly the same subject
            # and session masks as their intervention counterparts.
            id_base, id_base_per = subject_id_score(vam, vaq, splits[fold]["validation_subjects"], return_per_subject=True)
            base_ba, base_per = task_score(trm, trh, vam, vah, splits[fold]["train_subjects"], splits[fold]["validation_subjects"])
            for dose in DOSES:
                alpha_p, _ = lookup_params(params, fold, seed, "P", dose)
                _, p_q = suppress_coordinates(np.zeros_like(trq), trq, spec, protected_dims, alpha_p)
                _, vq_p = suppress_coordinates(np.zeros_like(vaq), vaq, spec, protected_dims, alpha_p)
                id_p, id_p_per = subject_id_score(vam, vq_p, splits[fold]["validation_subjects"], return_per_subject=True)
                for subject in subject_sort(splits[fold]["validation_subjects"]):
                    id_rows_all.append({"fold": fold, "seed": seed, "dose": dose, "group_id": "P", "subject_id": subject, "baseline_identity_BA": id_base_per.get(subject), "post_identity_BA": id_p_per.get(subject), "identity_drop": id_base_per.get(subject, np.nan) - id_p_per.get(subject, np.nan), "alpha": alpha_p, **flags()})
                p_h, p_per = task_score(trm, suppress_coordinates(trh, trq, spec, protected_dims, alpha_p)[0], vam, suppress_coordinates(vah, vaq, spec, protected_dims, alpha_p)[0], splits[fold]["train_subjects"], splits[fold]["validation_subjects"])
                for subject in subject_sort(splits[fold]["validation_subjects"]):
                    hp = base_per.get(subject, np.nan) - p_per.get(subject, np.nan)
                    task_rows_all.append({"fold": fold, "seed": seed, "dose": dose, "group_id": "P", "subject_id": subject, "baseline_BA": base_per.get(subject), "post_BA": p_per.get(subject), "harm": hp, "alpha": alpha_p, "head": "refit", **flags()})
                # Secondary fixed-head consequence for P.
                fixed_pred, _ = ridge_predict(suppress_coordinates(vah, vaq, spec, protected_dims, alpha_p)[0], ridge_pack(trh, mi_label(trm)))
                # N ensemble: each control is retained; primary N is the
                # arithmetic ensemble mean, fixed before seeing validation BA.
                for _, ctrl in run_controls.iterrows():
                    gid = str(ctrl.control_id); dims = json.loads(ctrl.coordinate_ids); alpha_n, _ = lookup_params(params, fold, seed, gid, dose)
                    htr_n, qtr_n = suppress_coordinates(trh, trq, spec, dims, alpha_n); hva_n, qva_n = suppress_coordinates(vah, vaq, spec, dims, alpha_n)
                    id_n, id_n_per = subject_id_score(vam, qva_n, splits[fold]["validation_subjects"], return_per_subject=True)
                    n_ba, n_per = task_score(trm, htr_n, vam, hva_n, splits[fold]["train_subjects"], splits[fold]["validation_subjects"])
                    for subject in subject_sort(splits[fold]["validation_subjects"]):
                        id_rows_all.append({"fold": fold, "seed": seed, "dose": dose, "group_id": gid, "subject_id": subject, "baseline_identity_BA": id_base_per.get(subject), "post_identity_BA": id_n_per.get(subject), "identity_drop": id_base_per.get(subject, np.nan) - id_n_per.get(subject, np.nan), "alpha": alpha_n, **flags()})
                        task_rows_all.append({"fold": fold, "seed": seed, "dose": dose, "group_id": gid, "subject_id": subject, "baseline_BA": base_per.get(subject), "post_BA": n_per.get(subject), "harm": base_per.get(subject, np.nan) - n_per.get(subject, np.nan), "alpha": alpha_n, "head": "refit", **flags()})
                    dose_rows_all.append({"fold": fold, "seed": seed, "dose": dose, "group_id": gid, "control_id": gid, "identity_drop": id_base - id_n, "task_harm": base_ba - n_ba, "alpha": alpha_n, **flags()})
                dose_rows_all.append({"fold": fold, "seed": seed, "dose": dose, "group_id": "P", "control_id": "P", "identity_drop": id_base - id_p, "task_harm": base_ba - p_h, "alpha": alpha_p, **flags()})
            # Same-rank random control is a secondary sanity diagnostic only.
            rng = np.random.default_rng(stable_seed("exp3-random-control", fold, seed))
            all_dims = np.arange(len(spec["rho"]), dtype=np.int64)
            rank = len(protected_dims)
            for ri in range(20):
                rdims = np.sort(rng.choice(all_dims, size=rank, replace=False)).tolist()
                alpha_r, _ = lookup_params(params, fold, seed, "P", "MEDIUM")
                htr_r, _ = suppress_coordinates(trh, trq, spec, rdims, alpha_r); hva_r, _ = suppress_coordinates(vah, vaq, spec, rdims, alpha_r)
                rb, rp = task_score(trm, htr_r, vam, hva_r, splits[fold]["train_subjects"], splits[fold]["validation_subjects"])
                random_rows_all.append({"fold": fold, "seed": seed, "random_id": f"R{ri+1:03d}", "coordinate_ids": json.dumps(rdims), "rank": rank, "alpha": alpha_r, "baseline_BA": base_ba, "post_BA": rb, "task_harm": base_ba - rb, **flags()})
            provenance.append({"fold": fold, "seed": seed, "checkpoint": feat_prov["checkpoint"], "checkpoint_sha256": feat_prov["checkpoint_sha256"], "spectrum_fingerprint": upstream_prov["fingerprint"], "protected_block_ids": protected_blocks, "protected_coordinate_ids": protected_dims, "train_subjects": splits[fold]["train_subjects"], "validation_subjects": splits[fold]["validation_subjects"], **flags()})
    write_csv(OUT / "PERSISTENCE_REPLICATION.csv", pd.DataFrame(persistence_rows_all))
    write_csv(OUT / "IDENTITY_MANIPULATION_CHECK.csv", pd.DataFrame(id_rows_all))
    write_csv(OUT / "TASK_CONSEQUENCE_RUN_LEVEL.csv", pd.DataFrame(task_rows_all))
    write_csv(OUT / "DOSE_RESPONSE.csv", pd.DataFrame(dose_rows_all))
    write_csv(OUT / "RANDOM_CONTROL_DIAGNOSTIC.csv", pd.DataFrame(random_rows_all))
    write_json(OUT / "PROVENANCE_AUDIT.json", {"runs": provenance, "freeze_record": freeze_record, "split_sha256": split_sha, "validation_outcome_used_for_design": False, **flags()})
    write_json(OUT / "DATA_ACCESS_AUDIT_FINAL.json", {"authorized_subject_scope": "development_train_and_development_validation_only", "outer_data_loaded": False, "outer_membership_enumerated": False, "outer_test_used": False})
    return {"status": "FINAL_RAW_OUTPUTS_READY", "persistence_rows": len(persistence_rows_all), "identity_rows": len(id_rows_all), "task_rows": len(task_rows_all), "dose_rows": len(dose_rows_all), **flags()}


def terminal_after_train_only() -> dict[str, Any]:
    """Close the experiment honestly when G1 is impossible in train-only data.

    This path computes the held-out persistence audit, but deliberately does
    not run identity-dose or task-consequence outcomes.  It is therefore not
    a workaround for the G1 gate and cannot produce a causal positive claim.
    """
    design_path = OUT / "TRAIN_ONLY_DESIGN.json"
    if not design_path.exists():
        raise RuntimeError("run train_only before terminal")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("status") != "MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE":
        raise RuntimeError(f"terminal phase requires G1-unavailable design, got {design.get('status')}")
    ensure_dirs(); splits, split_sha = load_development_splits()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Experiment 3 terminal audit requires the server CUDA environment")
    persistence_rows_all: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for fold in FOLDS:
        allowed = splits[fold]["train_subjects"] + splits[fold]["validation_subjects"]
        manifest = load_development_manifest(allowed)
        for seed in SEEDS:
            _, _, val_meta, val_h, feat_prov = extract_or_load_features(fold, seed, splits[fold], manifest, device)
            spec, assignment, upstream_prov = load_upstream_run(fold, seed)
            val_q = coordinates(val_h, spec).astype(np.float32)
            persistence_rows_all.extend(persistence_rows(val_meta, val_q, splits[fold]["validation_subjects"], fold, seed))
            protected_blocks = list(map(int, assignment["mi"]["protected"]))
            protected_dims = sorted(set(sum((list(map(int, spec["blocks"][b])) for b in protected_blocks), [])))
            provenance.append({"fold": fold, "seed": seed, "checkpoint": feat_prov["checkpoint"], "checkpoint_sha256": feat_prov["checkpoint_sha256"], "spectrum_fingerprint": upstream_prov["fingerprint"], "protected_block_ids": protected_blocks, "protected_coordinate_ids": protected_dims, "train_subjects": splits[fold]["train_subjects"], "validation_subjects": splits[fold]["validation_subjects"], **flags()})
    persistence = pd.DataFrame(persistence_rows_all)
    write_csv(OUT / "PERSISTENCE_REPLICATION.csv", persistence)
    if len(persistence):
        p_subject = persistence.groupby("subject_id", as_index=False).agg(same_similarity=("same_similarity", "mean"), mismatched_similarity=("mismatched_similarity", "mean"), R_persist=("R_persist", "mean"))
    else:
        p_subject = pd.DataFrame(columns=["subject_id", "same_similarity", "mismatched_similarity", "R_persist"])
    write_csv(OUT / "PERSISTENCE_REPLICATION_SUBJECT_LEVEL.csv", p_subject)
    p_stats = bootstrap(p_subject, "R_persist", stable_seed("g0"))
    write_json(OUT / "PERSISTENCE_REPLICATION_STATS.json", p_stats)
    g0 = bool(p_stats.get("ci95", [None, None])[0] is not None and p_stats["ci95"][0] > 0)
    reason = "G1 failed in train-only construction: at least one frozen Protected assignment has fewer than 20 exact-rank non-Protected persistence-supported controls."
    marker = pd.DataFrame([{"status": "NOT_EVALUATED", "reason": reason, **flags()}])
    for name in ("IDENTITY_MANIPULATION_CHECK.csv", "IDENTITY_MANIPULATION_SUBJECT_LEVEL.csv", "IDENTITY_MANIPULATION_PAIRED.csv", "TASK_CONSEQUENCE_SUBJECT_LEVEL.csv", "TASK_CONSEQUENCE_RUN_LEVEL.csv", "DOSE_RESPONSE.csv", "RANDOM_CONTROL_DIAGNOSTIC.csv"):
        write_csv(OUT / name, marker)
    empty_metric = {"mean": None, "median": None, "ci95": [None, None], "sign_probability": None, "n_unique_subjects": 0, "draws": BOOTSTRAP_DRAWS}
    primary = {"status": "NOT_EVALUATED_G1_FAILED", "H_P": empty_metric, "H_N": empty_metric, "Delta_H": empty_metric, "gate_G3": None, "primary_dose": "MEDIUM", **flags()}
    write_json(OUT / "PRIMARY_CAUSAL_EFFECT.json", primary)
    write_json(OUT / "IDENTITY_MANIPULATION_STATS.json", {"status": "NOT_EVALUATED_G1_FAILED", "reason": reason, **flags()})
    write_json(OUT / "DOSE_RESPONSE_STATS.json", {"status": "NOT_EVALUATED_G1_FAILED", "reason": reason, **flags()})
    write_json(OUT / "PROVENANCE_AUDIT.json", {"runs": provenance, "train_only_design": design, "split_sha256": split_sha, "validation_outcome_used_for_design": False, **flags()})
    write_json(OUT / "DATA_ACCESS_AUDIT_FINAL.json", {"authorized_subject_scope": "development_train_and_development_validation_only", "outer_data_loaded": False, "outer_membership_enumerated": False, "outer_test_used": False})
    failed = design.get("failed_runs", [])
    decision = {"terminal_state": "MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE", "G0_heldout_persistence": g0, "G1_matched_controls": False, "G2_identity_equivalence": None, "G3_utility_causal_effect": None, "mean_delta_ID_P": None, "mean_delta_ID_N": None, "identity_difference": None, "primary": primary, "matching_failures": failed, "utility_not_identity_claim": "PARTIAL" if g0 else "NO", "enter_experiment_4": "NO", "validation_outcome_used_for_design": False, **flags()}
    write_json(OUT / "FINAL_DECISION.json", decision)
    p_ci = p_stats.get("ci95")
    report = "\n".join([
        "# PERSIST-EEG Experiment 3 scientific report", "", "## Scope", "",
        "This prospectively frozen closure stopped before causal validation outcomes because the train-only matched-control gate was impossible for one frozen assignment. It is not an untouched independent replication and it is not outer validation.", "",
        "## Gate results", "",
        f"- G0 held-out persistence replication: `{g0}`; R_persist mean={p_stats.get('mean')}, 95% CI={p_ci}.",
        f"- G1 matched controls: `False`; required at least {MATCH_MIN} controls for every assignment. Train-only failed runs: {json.dumps(failed, ensure_ascii=False)}.",
        "- G2 identity-dose equivalence: not evaluated because G1 failed before validation outcome.",
        "- G3 primary causal consequence: not evaluated because no valid six-run matched-control design was frozen.", "",
        "## Scientific interpretation", "",
        "The natural non-Protected persistence-supported pool cannot supply the required exact-rank control ensemble for fold-2/seed-1 (Protected rank 8, non-Protected pool rank 8, one legal combination). Creating duplicate rotations of the same full-rank span would not be independent controls, and admitting unsupported coordinates would change the frozen control source; neither was used.", "",
        "No H_P, H_N, Delta_H, identity-dose equivalence, or dose-response causal quantity is reported as measured. The terminal state is `MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE`.", "",
        f"Utility-not-identity claim: `{'PARTIAL' if g0 else 'NO'}`. Experiment-4 entry: `NO`.", "",
        "## Leakage and outer audit", "",
        "Validation representations were used only for the held-out persistence audit after the train-only design failed. No validation task BA, identity-dose outcome, or task consequence was used to select matching. All artifacts set `outer_test_used=false` and `outer_membership_enumerated=false`.", "",
    ])
    (OUT / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    # Only the applicable held-out persistence figure is generated.  No causal
    # figure is fabricated for an endpoint that was not evaluated.
    try:
        import matplotlib.pyplot as plt
        if len(persistence):
            fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(persistence.same_similarity, persistence.mismatched_similarity, alpha=.7); lo = min(persistence.same_similarity.min(), persistence.mismatched_similarity.min()); hi = max(persistence.same_similarity.max(), persistence.mismatched_similarity.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=.8); ax.set_xlabel("same-subject cosine"); ax.set_ylabel("mismatched-subject cosine"); ax.set_title("Held-out persistence replication"); fig.tight_layout(); fig.savefig(FIGURES / "01_heldout_persistence.png", dpi=160); plt.close(fig)
    except Exception:
        pass
    expected = ["PERSISTENCE_REPLICATION.csv", "PERSISTENCE_REPLICATION_STATS.json", "MATCHED_CONTROL_TABLE.csv", "MATCHING_DIAGNOSTICS.csv", "MATCHED_CONTROL_PROVENANCE.json", "IDENTITY_RESPONSE_CURVES.csv", "IDENTITY_MATCHING_PARAMETERS.csv", "IDENTITY_MANIPULATION_CHECK.csv", "IDENTITY_MANIPULATION_STATS.json", "TASK_CONSEQUENCE_SUBJECT_LEVEL.csv", "TASK_CONSEQUENCE_RUN_LEVEL.csv", "PRIMARY_CAUSAL_EFFECT.json", "DOSE_RESPONSE.csv", "DOSE_RESPONSE_STATS.json", "RANDOM_CONTROL_DIAGNOSTIC.csv", "PROVENANCE_AUDIT.json", "DATA_ACCESS_AUDIT_FINAL.json", "FINAL_DECISION.json", "SCIENTIFIC_REPORT.md"]
    hashes = {name: sha256_file(OUT / name) for name in expected if (OUT / name).exists()}
    write_json(OUT / "REPRODUCIBILITY_AUDIT.json", {"terminal_state": decision["terminal_state"], "lightweight_output_sha256": hashes, "expected_files": expected, "bootstrap_draws": BOOTSTRAP_DRAWS, "finalize_deterministic": True, **flags()})
    return decision


def bootstrap(values: pd.DataFrame, column: str, seed: int) -> dict[str, Any]:
    data = values[["subject_id", column]].dropna().groupby("subject_id", sort=True)[column].mean()
    arr = data.to_numpy(float)
    if len(arr) == 0:
        return {"mean": None, "median": None, "ci95": [None, None], "sign_probability": None, "n_unique_subjects": 0, "draws": BOOTSTRAP_DRAWS}
    rng = np.random.default_rng(stable_seed("exp3-bootstrap", seed, column, len(arr)))
    draw = rng.choice(arr, size=(BOOTSTRAP_DRAWS, len(arr)), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "ci95": [float(np.quantile(draw, .025)), float(np.quantile(draw, .975))], "sign_probability": float(np.mean(draw > 0)), "n_unique_subjects": int(len(arr)), "positive_subject_fraction": float(np.mean(arr > 0)), "nonnegative_subject_fraction": float(np.mean(arr >= 0)), "worst_subject": float(arr.min()), "draws": BOOTSTRAP_DRAWS}


def finalize() -> dict[str, Any]:
    require_frozen()
    p = pd.read_csv(OUT / "PERSISTENCE_REPLICATION.csv")
    i = pd.read_csv(OUT / "IDENTITY_MANIPULATION_CHECK.csv")
    t = pd.read_csv(OUT / "TASK_CONSEQUENCE_RUN_LEVEL.csv")
    d = pd.read_csv(OUT / "DOSE_RESPONSE.csv")
    r = pd.read_csv(OUT / "RANDOM_CONTROL_DIAGNOSTIC.csv")
    # Aggregate repeated runs/controls inside unique subject before inferential
    # bootstrap.  This is deliberately separate from run-level diagnostics.
    p_subject = p.groupby("subject_id", as_index=False).agg(same_similarity=("same_similarity", "mean"), mismatched_similarity=("mismatched_similarity", "mean"), R_persist=("R_persist", "mean"))
    write_csv(OUT / "PERSISTENCE_REPLICATION_SUBJECT_LEVEL.csv", p_subject)
    p_stats = bootstrap(p_subject, "R_persist", stable_seed("g0"))
    write_json(OUT / "PERSISTENCE_REPLICATION_STATS.json", p_stats)
    medium_i = i[i.dose == "MEDIUM"].copy()
    id_subject = medium_i.groupby(["subject_id", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), baseline_identity_BA=("baseline_identity_BA", "mean"), post_identity_BA=("post_identity_BA", "mean"))
    piv = id_subject.pivot(index="subject_id", columns="group_id", values="identity_drop")
    id_diff = pd.DataFrame({"subject_id": piv.index.astype(str), "delta_ID_P_minus_N": piv.get("P", np.nan) - piv.drop(columns=["P"], errors="ignore").mean(axis=1)})
    write_csv(OUT / "IDENTITY_MANIPULATION_SUBJECT_LEVEL.csv", id_subject)
    write_csv(OUT / "IDENTITY_MANIPULATION_PAIRED.csv", id_diff)
    id_stats = {"P": bootstrap(id_subject[id_subject.group_id == "P"], "identity_drop", stable_seed("id-p")), "N": bootstrap(id_subject[id_subject.group_id != "P"], "identity_drop", stable_seed("id-n")), "P_minus_N": bootstrap(id_diff.rename(columns={"delta_ID_P_minus_N": "value"}), "value", stable_seed("id-diff")), "tolerance": IDENTITY_TOLERANCE}
    write_json(OUT / "IDENTITY_MANIPULATION_STATS.json", id_stats)
    medium_t = t[t.dose == "MEDIUM"].copy()
    p_t = medium_t[medium_t.group_id == "P"]
    n_t = medium_t[medium_t.group_id != "P"]
    p_sub = p_t.groupby("subject_id", as_index=False).agg(baseline_BA=("baseline_BA", "mean"), P_BA=("post_BA", "mean"), H_P=("harm", "mean"))
    n_sub = n_t.groupby("subject_id", as_index=False).agg(N_BA=("post_BA", "mean"), H_N=("harm", "mean"))
    task_subject = p_sub.merge(n_sub, on="subject_id", how="inner")
    task_subject["Delta_H"] = task_subject.H_P - task_subject.H_N
    write_csv(OUT / "TASK_CONSEQUENCE_SUBJECT_LEVEL.csv", task_subject)
    primary = bootstrap(task_subject.rename(columns={"Delta_H": "value"}), "value", stable_seed("primary-delta-h"))
    hp = bootstrap(task_subject.rename(columns={"H_P": "value"}), "value", stable_seed("primary-hp"))
    hn = bootstrap(task_subject.rename(columns={"H_N": "value"}), "value", stable_seed("primary-hn"))
    run_positive = int((medium_t.groupby(["fold", "seed", "group_id"]).harm.mean().unstack("group_id").assign(delta=lambda x: x.get("P", np.nan) - x.drop(columns=["P"], errors="ignore").mean(axis=1)).delta > 0).sum()) if len(medium_t) else 0
    primary_payload = {"H_P": hp, "H_N": hn, "Delta_H": primary, "run_level_delta_positive_count": run_positive, "unique_subject_robustness_positive": primary.get("positive_subject_fraction"), "gate_G3": bool(primary.get("mean") is not None and primary["mean"] >= 0.01 and primary["ci95"][0] > 0), "primary_dose": "MEDIUM", **flags()}
    write_json(OUT / "PRIMARY_CAUSAL_EFFECT.json", primary_payload)
    dose_stats = {}
    for dose in DOSES:
        dd = d[d.dose == dose]
        pp = dd[dd.group_id == "P"].groupby("fold", as_index=False).agg(identity_drop=("identity_drop", "mean"), task_harm=("task_harm", "mean"))
        nn = dd[dd.group_id != "P"].groupby("fold", as_index=False).agg(identity_drop=("identity_drop", "mean"), task_harm=("task_harm", "mean"))
        dose_stats[dose] = {"P": {"identity_drop": float(pp.identity_drop.mean()) if len(pp) else None, "task_harm": float(pp.task_harm.mean()) if len(pp) else None}, "N": {"identity_drop": float(nn.identity_drop.mean()) if len(nn) else None, "task_harm": float(nn.task_harm.mean()) if len(nn) else None}, "n_rows": int(len(dd))}
    write_json(OUT / "DOSE_RESPONSE_STATS.json", dose_stats)
    # Gate 0 and G2 are evaluated only after the frozen outcome measurements.
    g0 = bool(p_stats.get("ci95", [None, None])[0] is not None and p_stats["ci95"][0] > 0)
    g1_rows = pd.read_csv(OUT / "MATCHING_DIAGNOSTICS.csv")
    g1 = bool(len(g1_rows) == 6 and (g1_rows.accepted_count >= MATCH_MIN).all() and (g1_rows.protected_rank > 0).all())
    id_mean_p = float(id_subject[id_subject.group_id == "P"].identity_drop.mean()) if len(id_subject[id_subject.group_id == "P"]) else float("nan")
    id_mean_n = float(id_subject[id_subject.group_id != "P"].identity_drop.mean()) if len(id_subject[id_subject.group_id != "P"]) else float("nan")
    g2 = bool(np.isfinite(id_mean_p) and np.isfinite(id_mean_n) and abs(id_mean_p - id_mean_n) <= IDENTITY_TOLERANCE)
    g3 = bool(primary_payload["gate_G3"])
    if not g0:
        terminal = "PERSISTENCE_DOES_NOT_REPLICATE_ON_HELDOUT_SUBJECTS"
    elif not g1:
        terminal = "MATCHED_NONPROTECTED_CONTROL_UNAVAILABLE"
    elif not g2:
        terminal = "IDENTITY_MATCH_FAILED"
    elif not g3:
        terminal = "PROTECTED_CAUSAL_EFFECT_NOT_SUPPORTED"
    else:
        terminal = "EXP3_UTILITY_NOT_IDENTITY_SUPPORTED"
    utility_claim = "YES" if terminal == "EXP3_UTILITY_NOT_IDENTITY_SUPPORTED" else ("NO" if g0 and g1 and g2 else "PARTIAL")
    decision = {"terminal_state": terminal, "G0_heldout_persistence": g0, "G1_matched_controls": g1, "G2_identity_equivalence": g2, "G3_utility_causal_effect": g3, "mean_delta_ID_P": id_mean_p, "mean_delta_ID_N": id_mean_n, "identity_difference": id_mean_p - id_mean_n if np.isfinite(id_mean_p) and np.isfinite(id_mean_n) else None, "primary": primary_payload, "utility_not_identity_claim": utility_claim, "enter_experiment_4": "YES" if g0 and g1 and g2 and g3 else ("CONDITIONAL" if g0 and g1 and g2 else "NO"), "validation_outcome_used_for_design": False, **flags()}
    write_json(OUT / "FINAL_DECISION.json", decision)
    report = make_report(decision, p_stats, id_stats, primary_payload, dose_stats, r)
    (OUT / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    make_figures(p, medium_i, task_subject, d)
    expected = ["PERSISTENCE_REPLICATION.csv", "PERSISTENCE_REPLICATION_STATS.json", "MATCHED_CONTROL_TABLE.csv", "MATCHING_DIAGNOSTICS.csv", "MATCHED_CONTROL_PROVENANCE.json", "IDENTITY_RESPONSE_CURVES.csv", "IDENTITY_MATCHING_PARAMETERS.csv", "IDENTITY_MANIPULATION_CHECK.csv", "IDENTITY_MANIPULATION_STATS.json", "TASK_CONSEQUENCE_SUBJECT_LEVEL.csv", "TASK_CONSEQUENCE_RUN_LEVEL.csv", "PRIMARY_CAUSAL_EFFECT.json", "DOSE_RESPONSE.csv", "DOSE_RESPONSE_STATS.json", "RANDOM_CONTROL_DIAGNOSTIC.csv", "PROVENANCE_AUDIT.json", "FINAL_DECISION.json", "SCIENTIFIC_REPORT.md"]
    hashes = {name: sha256_file(OUT / name) for name in expected if (OUT / name).exists()}
    write_json(OUT / "REPRODUCIBILITY_AUDIT.json", {"lightweight_output_sha256": hashes, "expected_files": expected, "bootstrap_draws": BOOTSTRAP_DRAWS, "finalize_deterministic": True, **flags()})
    return decision


def make_report(decision: Mapping[str, Any], p_stats: Mapping[str, Any], id_stats: Mapping[str, Any], primary: Mapping[str, Any], dose_stats: Mapping[str, Any], random_frame: pd.DataFrame) -> str:
    hp, hn, dh = primary["H_P"], primary["H_N"], primary["Delta_H"]
    lines = ["# PERSIST-EEG Experiment 3 scientific report", "", "## Scope", "", "This is a prospectively frozen Experiment-3 closure on a development resource reused by earlier experiments. It is not an untouched independent replication and it is not outer validation.", "", "## Gate results", "", f"- G0 held-out persistence replication: `{decision['G0_heldout_persistence']}`; R_persist mean={p_stats.get('mean')}, 95% CI={p_stats.get('ci95')}.", f"- G1 matched controls: `{decision['G1_matched_controls']}`; accepted controls per run are recorded in `MATCHING_DIAGNOSTICS.csv`.", f"- G2 identity-dose equivalence: `{decision['G2_identity_equivalence']}`; ΔID_P={decision.get('mean_delta_ID_P')}, ΔID_N={decision.get('mean_delta_ID_N')}, difference={decision.get('identity_difference')}.", f"- G3 primary causal consequence: `{decision['G3_utility_causal_effect']}`.", "", "## Primary MEDIUM endpoint", "", f"- H_P mean={hp.get('mean')}, median={hp.get('median')}, 95% CI={hp.get('ci95')}.", f"- H_N mean={hn.get('mean')}, median={hn.get('median')}, 95% CI={hn.get('ci95')}.", f"- ΔH mean={dh.get('mean')}, median={dh.get('median')}, 95% CI={dh.get('ci95')}, sign probability={dh.get('sign_probability')}, positive-subject fraction={dh.get('positive_subject_fraction')}, nonnegative-subject fraction={dh.get('nonnegative_subject_fraction')}, worst subject={dh.get('worst_subject')}.", "", "## Dose-response", ""]
    for dose, value in dose_stats.items():
        lines.append(f"- {dose}: P identity drop/task harm={value['P']}; N identity drop/task harm={value['N']}.")
    lines += ["", f"Secondary random-control rows: {len(random_frame)}.", "", f"Terminal scientific state: `{decision['terminal_state']}`.", f"Utility-not-identity claim: `{decision['utility_not_identity_claim']}`.", f"Experiment 4 entry: `{decision['enter_experiment_4']}`.", "", "## Leakage and outer audit", "", "All artifacts set `outer_test_used=false` and `outer_membership_enumerated=false`; validation outcomes were not used for matching, alpha, dose, metric, or gate selection.", ""]
    return "\n".join(lines)


def make_figures(persistence: pd.DataFrame, identity: pd.DataFrame, task_subject: pd.DataFrame, dose: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(persistence.same_similarity, persistence.mismatched_similarity, alpha=.7); lo = min(persistence.same_similarity.min(), persistence.mismatched_similarity.min()); hi = max(persistence.same_similarity.max(), persistence.mismatched_similarity.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=.8); ax.set_xlabel("same-subject cosine"); ax.set_ylabel("mismatched-subject cosine"); ax.set_title("Held-out persistence replication"); fig.tight_layout(); fig.savefig(FIGURES / "01_heldout_persistence.png", dpi=160); plt.close(fig)
    medium = identity[identity.dose == "MEDIUM"]; fig, ax = plt.subplots(figsize=(6, 4)); medium.boxplot(column="identity_drop", by="group_id", ax=ax); ax.set_title("Identity removal at MEDIUM dose"); ax.set_ylabel("cross-session subject-ID BA drop"); fig.suptitle(""); fig.tight_layout(); fig.savefig(FIGURES / "02_identity_matching.png", dpi=160); plt.close(fig)
    if len(task_subject):
        fig, ax = plt.subplots(figsize=(5, 4)); ax.bar(["Protected", "Matched N"], [task_subject.H_P.mean(), task_subject.H_N.mean()], yerr=[task_subject.H_P.std(ddof=1), task_subject.H_N.std(ddof=1)]); ax.set_ylabel("task BA harm"); ax.set_title("Primary matched causal endpoint"); fig.tight_layout(); fig.savefig(FIGURES / "03_primary_task_harm.png", dpi=160); plt.close(fig)
    if len(dose):
        summary = dose.groupby(["dose", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), task_harm=("task_harm", "mean")); fig, ax = plt.subplots(figsize=(6, 4));
        for gid, group in summary.groupby("group_id"):
            ax.plot(group.identity_drop, group.task_harm, marker="o", label=gid)
        ax.set_xlabel("actual identity reduction"); ax.set_ylabel("task BA harm"); ax.legend(title="group"); ax.set_title("Dose response"); fig.tight_layout(); fig.savefig(FIGURES / "04_dose_response.png", dpi=160); plt.close(fig)


def _legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["phase0", "train_only", "freeze", "final", "terminal", "finalize", "all"])
    args = parser.parse_args()
    if args.phase in {"phase0", "all"}:
        phase0()
    if args.phase in {"train_only", "all"}:
        build_train_only()
    if args.phase in {"freeze", "all"}:
        freeze()
    if args.phase in {"final", "all"}:
        run_final()
    if args.phase == "terminal":
        print(json.dumps(clean(terminal_after_train_only()), ensure_ascii=False, indent=2))
    if args.phase in {"finalize", "all"}:
        print(json.dumps(clean(finalize()), ensure_ascii=False, indent=2))


if False and __name__ == "__main__":
    _legacy_main()


# ---------------------------------------------------------------------------
# Experiment 3 V1.1: block-wise causal unit.
# The V1 implementation above is retained as an audit reference, but the
# definitions below are the only entry points used by this V1.1 script.
# ---------------------------------------------------------------------------

V11_MATCH_MAX = 50
V11_MATCH_MIN = 20
V11_ID_TOLERANCE = 0.01
V11_RANDOM_COUNT = 20
V11_MATCH_FEATURES = [
    "rho_mean", "rho_std", "coordinate_variance_mean", "coordinate_energy_mean",
    "train_identity_balanced_accuracy", "dewhitened_direction_norm_mean",
    "train_task_margin_magnitude",
]


def symmetric_subject_id_score(meta: pd.DataFrame, representation: np.ndarray, subjects: Sequence[str], return_per_subject: bool = False) -> tuple[float, dict[str, float]]:
    """Average S1->S2 and S2->S1 subject-ID BA on the same frozen rows."""
    score12, per12 = subject_id_score(meta, representation, subjects, return_per_subject=True)
    reverse = meta.copy()
    reverse["session_id"] = 3 - reverse["session_id"].astype(int)
    score21, per21 = subject_id_score(reverse, representation, subjects, return_per_subject=True)
    score = float(np.nanmean([score12, score21]))
    per = {s: float(np.nanmean([per12.get(s, np.nan), per21.get(s, np.nan)])) for s in subject_sort(subjects)}
    return (score, per) if return_per_subject else (score, {})


def identity_curve_v11(meta: pd.DataFrame, q: np.ndarray, dims: Sequence[int], subjects: Sequence[str], fold: int, seed: int, group_id: str, protected_block_id: int) -> pd.DataFrame:
    baseline, _ = symmetric_subject_id_score(meta, q, subjects, return_per_subject=True)
    rows = []
    for alpha in ALPHAS:
        q2 = q.copy()
        q2[:, np.asarray(dims, dtype=np.int64)] *= (1.0 - float(alpha))
        score, _ = symmetric_subject_id_score(meta, q2, subjects, return_per_subject=True)
        rows.append({"fold": fold, "seed": seed, "protected_block_id": int(protected_block_id), "group_id": str(group_id), "alpha": float(alpha), "identity_BA": float(score), "identity_drop": float(baseline - score), "baseline_identity_BA": float(baseline), **flags()})
    return pd.DataFrame(rows).rename(columns={"identity_drop": "drop"})


def make_persistence_cache(meta: pd.DataFrame, q: np.ndarray, subjects: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute train same/mismatched class means once per frozen run."""
    ordered = subject_sort(subjects)
    shifted = {s: ordered[(i + 1) % len(ordered)] for i, s in enumerate(ordered)}
    same_a, same_b, mismatch = [], [], []
    labels = sorted(meta.loc[meta.paradigm == "mi", "event_label"].astype(str).unique())
    for subject in ordered:
        for label in labels:
            a = (meta.subject_id.astype(str) == subject).to_numpy() & (meta.session_id.to_numpy() == 1) & (meta.paradigm == "mi").to_numpy() & (meta.event_label.astype(str) == label).to_numpy()
            b = (meta.subject_id.astype(str) == subject).to_numpy() & (meta.session_id.to_numpy() == 2) & (meta.paradigm == "mi").to_numpy() & (meta.event_label.astype(str) == label).to_numpy()
            m = (meta.subject_id.astype(str) == shifted[subject]).to_numpy() & (meta.session_id.to_numpy() == 2) & (meta.paradigm == "mi").to_numpy() & (meta.event_label.astype(str) == label).to_numpy()
            if np.any(a) and np.any(b) and np.any(m):
                same_a.append(q[a].mean(axis=0)); same_b.append(q[b].mean(axis=0)); mismatch.append(q[m].mean(axis=0))
    if not same_a:
        empty = np.empty((0, q.shape[1]), dtype=np.float64); return empty, empty, empty
    return np.asarray(same_a, dtype=np.float64), np.asarray(same_b, dtype=np.float64), np.asarray(mismatch, dtype=np.float64)


def persistence_metrics_train(meta: pd.DataFrame, q: np.ndarray, subjects: Sequence[str], dims: Sequence[int], fold: int, seed: int, cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> dict[str, float]:
    same_a, same_b, mismatch = cache if cache is not None else make_persistence_cache(meta, q, subjects)
    if len(same_a) == 0:
        return {"train_same_similarity": float("nan"), "train_mismatched_similarity": float("nan"), "train_R_persist": float("nan")}
    d = np.asarray(sorted(set(map(int, dims))), dtype=np.int64); a = same_a[:, d]; b = same_b[:, d]; m = mismatch[:, d]
    def row_cos(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1); return np.divide(np.sum(x * y, axis=1), den, out=np.zeros(len(x), dtype=float), where=den > 1e-12)
    same = row_cos(a, b); mis = row_cos(a, m)
    return {"train_same_similarity": float(np.mean(same)), "train_mismatched_similarity": float(np.mean(mis)), "train_R_persist": float(np.mean(same - mis))}


def v11_control_ids(block: int, count: int) -> str:
    return f"N_B{int(block)}_{int(count):03d}"


def match_block_controls_v11(train_meta: pd.DataFrame, train_q: np.ndarray, spec: Mapping[str, Any], all_protected_blocks: Sequence[int], protected_block: int, fold: int, seed: int, train_subjects: Sequence[str], persistence_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    protected_dims = sorted(set(map(int, spec["blocks"][int(protected_block)])))
    all_protected_dims = sorted(set(sum((list(map(int, spec["blocks"][int(b)])) for b in all_protected_blocks), [])))
    supported_blocks = [int(row["block"]) for row in spec["audit"].get("persistence_support", []) if bool(row.get("persistence_supported"))]
    pool = sorted(set(sum((list(map(int, spec["blocks"][b])) for b in supported_blocks), [])) - set(all_protected_dims))
    rank = len(protected_dims)
    candidates = candidate_sets(pool, rank, fold, seed)
    p_metrics = group_metrics(train_meta, train_q, spec, protected_dims, train_subjects)
    p_persist = persistence_metrics_train(train_meta, train_q, train_subjects, protected_dims, fold, seed, persistence_cache)
    rows = []
    for dims in candidates:
        metrics = group_metrics(train_meta, train_q, spec, dims, train_subjects)
        persist = persistence_metrics_train(train_meta, train_q, train_subjects, dims, fold, seed, persistence_cache)
        rows.append({"fold": fold, "seed": seed, "protected_block_id": int(protected_block), "candidate_coordinate_ids": json.dumps(list(map(int, dims))), "rank": rank, **metrics, **persist, "train_persistence_supported": bool(np.isfinite(persist["train_R_persist"]) and persist["train_R_persist"] > 0), **flags()})
    cand = pd.DataFrame(rows)
    if len(cand):
        good = cand[cand.train_persistence_supported.astype(bool)].copy()
    else:
        good = cand.copy()
    if len(good):
        values = good[V11_MATCH_FEATURES].to_numpy(float)
        target = np.asarray([p_metrics[f] for f in V11_MATCH_FEATURES], dtype=float)
        scale = np.nanstd(np.vstack([values, target]), axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        z = (values - target[None, :]) / scale[None, :]
        good["match_distance"] = np.sqrt(np.nanmean(z * z, axis=1))
        good["match_max_abs_z"] = np.nanmax(np.abs(z), axis=1)
        good = good.sort_values(["match_distance", "match_max_abs_z", "candidate_coordinate_ids"], kind="mergesort").reset_index(drop=True)
        accepted = good.head(V11_MATCH_MAX).copy()
    else:
        good["match_distance"] = np.nan; good["match_max_abs_z"] = np.nan
        accepted = good.copy()
    accepted_ids = set(accepted.candidate_coordinate_ids.astype(str).tolist()) if len(accepted) else set()
    if len(cand):
        cand["match_distance"] = cand.candidate_coordinate_ids.map(dict(zip(good.candidate_coordinate_ids.astype(str), good.match_distance)))
        cand["match_max_abs_z"] = cand.candidate_coordinate_ids.map(dict(zip(good.candidate_coordinate_ids.astype(str), good.match_max_abs_z)))
        cand["accepted"] = cand.candidate_coordinate_ids.astype(str).isin(accepted_ids)
    selected_rows = []
    for i, (_, row) in enumerate(accepted.iterrows(), start=1):
        selected_rows.append({"fold": fold, "seed": seed, "protected_block_id": int(protected_block), "control_id": v11_control_ids(protected_block, i), "protected_coordinate_ids": json.dumps(protected_dims), "coordinate_ids": row.candidate_coordinate_ids, "rank": rank, "match_distance": float(row.match_distance), "match_max_abs_z": float(row.match_max_abs_z), "train_same_similarity": float(row.train_same_similarity), "train_mismatched_similarity": float(row.train_mismatched_similarity), "train_R_persist": float(row.train_R_persist), **{f"N_{key}": float(row[key]) for key in V11_MATCH_FEATURES}, **{f"P_{key}": float(p_metrics[key]) for key in V11_MATCH_FEATURES}, **flags()})
    selected = pd.DataFrame(selected_rows)
    diag = pd.DataFrame([{"fold": fold, "seed": seed, "protected_block_id": int(protected_block), "protected_coordinate_ids": json.dumps(protected_dims), "protected_rank": rank, "pool_size": len(pool), "candidate_count": len(cand), "persistence_supported_count": len(good), "accepted_count": len(selected), "accepted_distance_max": float(selected.match_distance.max()) if len(selected) else None, "accepted_max_abs_z_max": float(selected.match_max_abs_z.max()) if len(selected) else None, "protected_train_R_persist": p_persist["train_R_persist"], "block_eligible": bool(len(selected) >= V11_MATCH_MIN), "matching_selection": "top_k_train_only_standardized_structural_distance_after_persistence_certification", **flags()}])
    payload = {"fold": fold, "seed": seed, "protected_block_id": int(protected_block), "protected_coordinate_ids": protected_dims, "all_protected_block_ids": list(map(int, all_protected_blocks)), "pool": pool, "rank": rank, "candidate_count": len(cand), "persistence_supported_count": len(good), "accepted_count": len(selected), "controls": [json.loads(x) for x in selected.coordinate_ids.tolist()] if len(selected) else [], "candidate_generation_seed": stable_seed("exp3-v11-candidates", fold, seed, protected_block, rank, *pool), "matching_rule": "top_k_train_only_standardized_structural_distance_after_persistence_certification", **flags()}
    return selected, cand, diag, payload


def _concat_v11(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()


def build_train_only_v11() -> dict[str, Any]:
    ensure_dirs(); config = load_frozen_config(); splits, split_sha = load_development_splits()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Experiment 3 V1.1 requires CUDA for V3.1 feature extraction")
    assignments, candidates_all, controls_all, diagnostics_all, persistence_all = [], [], [], [], []
    curves_all, targets_all, params_all, provenance = [], [], [], []
    eligible_blocks: list[tuple[int, int, int]] = []; failures = []
    for fold in FOLDS:
        allowed = splits[fold]["train_subjects"] + splits[fold]["validation_subjects"]
        manifest = load_development_manifest(allowed)
        for seed in SEEDS:
            train_meta, train_h, val_meta, val_h, feat_prov = extract_or_load_features(fold, seed, splits[fold], manifest, device)
            spec, assignment, upstream_prov = load_upstream_run(fold, seed)
            train_q = coordinates(train_h, spec).astype(np.float32)
            mi_mask = (train_meta.paradigm == "mi").to_numpy(); mi_meta = train_meta.loc[mi_mask].reset_index(drop=True); mi_q = train_q[mi_mask]; persistence_cache = make_persistence_cache(mi_meta, mi_q, splits[fold]["train_subjects"])
            protected_blocks = list(map(int, assignment["mi"]["protected"]))
            for block in protected_blocks:
                protected_dims = sorted(set(map(int, spec["blocks"][block])))
                assignments.append({"fold": fold, "seed": seed, "protected_block_id": block, "protected_coordinate_ids": json.dumps(protected_dims), "rank": len(protected_dims), "all_protected_block_ids": json.dumps(protected_blocks), "outer_test_used": False, "outer_membership_enumerated": False})
                selected, cand, diag, payload = match_block_controls_v11(mi_meta, mi_q, spec, protected_blocks, block, fold, seed, splits[fold]["train_subjects"], persistence_cache)
                candidates_all.append(cand); controls_all.append(selected); diagnostics_all.append(diag)
                if len(cand):
                    persistence_all.append(cand[["fold", "seed", "protected_block_id", "candidate_coordinate_ids", "rank", "train_same_similarity", "train_mismatched_similarity", "train_R_persist", "train_persistence_supported", "outer_test_used", "outer_membership_enumerated"]])
                p_persist = persistence_metrics_train(mi_meta, mi_q, splits[fold]["train_subjects"], protected_dims, fold, seed, persistence_cache)
                block_ok = len(selected) >= V11_MATCH_MIN
                medium_eligible = 0; p_max = float("nan")
                if block_ok:
                    p_gid = f"P_B{block}"
                    p_curve = identity_curve_v11(mi_meta, mi_q, protected_dims, splits[fold]["train_subjects"], fold, seed, p_gid, block)
                    curves_all.append(p_curve)
                    p_max = float(np.nanmax(p_curve["drop"].to_numpy(float)))
                    if not np.isfinite(p_max) or p_max <= 0:
                        block_ok = False
                    else:
                        for dose, fraction in DOSES.items():
                            target = p_max * float(fraction); alpha_p, drop_p = interpolate_alpha(p_curve, target)
                            targets_all.append({"fold": fold, "seed": seed, "protected_block_id": block, "dose": dose, "target_identity_drop_train": target, "protected_Dmax_train": p_max, **flags()})
                            params_all.append({"fold": fold, "seed": seed, "protected_block_id": block, "group_id": p_gid, "dose": dose, "target_identity_drop_train": target, "alpha": alpha_p, "drop_train": drop_p, "train_max_drop": p_max, "eligible": True, **flags()})
                            for _, row in selected.iterrows():
                                dims = json.loads(row.coordinate_ids); c_gid = str(row.control_id); c = identity_curve_v11(mi_meta, mi_q, dims, splits[fold]["train_subjects"], fold, seed, c_gid, block); curves_all.append(c); n_max = float(np.nanmax(c["drop"].to_numpy(float))); eligible = bool(np.isfinite(n_max) and n_max >= target and np.isfinite(target))
                                if eligible:
                                    alpha_n, drop_n = interpolate_alpha(c, target)
                                else:
                                    alpha_n, drop_n = float("nan"), float("nan")
                                params_all.append({"fold": fold, "seed": seed, "protected_block_id": block, "group_id": c_gid, "dose": dose, "target_identity_drop_train": target, "alpha": alpha_n, "drop_train": drop_n, "train_max_drop": n_max, "eligible": eligible, **flags()})
                        medium_eligible = int(sum(1 for row in params_all if row["fold"] == fold and row["seed"] == seed and row["protected_block_id"] == block and row["dose"] == "MEDIUM" and str(row["group_id"]).startswith("N_") and bool(row["eligible"])))
                        block_ok = bool(medium_eligible >= V11_MATCH_MIN)
                diagnostics_all[-1] = diag.assign(protected_train_R_persist=p_persist["train_R_persist"], protected_Dmax_train=p_max, medium_eligible_controls=medium_eligible, block_eligible=block_ok)
                if block_ok:
                    eligible_blocks.append((fold, seed, block))
                else:
                    failures.append({"fold": fold, "seed": seed, "protected_block_id": block, "accepted_controls": len(selected), "medium_eligible_controls": medium_eligible, "reason": "fewer than 20 valid block-wise controls after train-only persistence certification and P-anchored MEDIUM feasibility", **flags()})
                provenance.append({"fold": fold, "seed": seed, "protected_block_id": block, "protected_coordinate_ids": protected_dims, "protected_train_persistence": p_persist, "spectrum_fingerprint": upstream_prov["fingerprint"], "spectrum_path": upstream_prov["paths"]["spectrum"], "assignment_path": upstream_prov["paths"]["assignment"], "checkpoint": feat_prov["checkpoint"], "checkpoint_sha256": feat_prov["checkpoint_sha256"], "train_rows": len(train_meta), "validation_rows": len(val_meta), "control_payload": payload, "block_eligible": block_ok, **flags()})
    assignments_frame = pd.DataFrame(assignments); candidates_frame = _concat_v11(candidates_all); controls_frame = _concat_v11(controls_all); diagnostics_frame = _concat_v11(diagnostics_all); persistence_frame = _concat_v11(persistence_all); curves_frame = _concat_v11(curves_all); targets_frame = pd.DataFrame(targets_all); params_frame = pd.DataFrame(params_all)
    write_csv(OUT / "BLOCKWISE_PROTECTED_ASSIGNMENTS.csv", assignments_frame); write_csv(OUT / "BLOCKWISE_CONTROL_CANDIDATES.csv", candidates_frame); write_csv(OUT / "BLOCKWISE_MATCHED_CONTROLS.csv", controls_frame); write_csv(OUT / "BLOCKWISE_MATCHING_DIAGNOSTICS.csv", diagnostics_frame); write_json(OUT / "CONTROL_PROVENANCE.json", {"runs": provenance, "failures": failures, "minimum_controls_per_block": V11_MATCH_MIN, "maximum_controls_per_block": V11_MATCH_MAX, "matching_features": V11_MATCH_FEATURES, **flags()}); write_csv(OUT / "CONTROL_PERSISTENCE_AUDIT.csv", persistence_frame); write_csv(OUT / "TRAIN_IDENTITY_RESPONSE_CURVES.csv", curves_frame); write_csv(OUT / "IDENTITY_TARGETS.csv", targets_frame); write_csv(OUT / "IDENTITY_MATCHING_PARAMETERS.csv", params_frame)
    total_blocks = len(assignments_frame); eligible_block_fraction = float(len(eligible_blocks) / total_blocks) if total_blocks else 0.0; eligible_run_ids = sorted(set((f, s) for f, s, _ in eligible_blocks)); status = "TRAIN_ONLY_DESIGN_READY" if len(eligible_run_ids) >= 5 and eligible_block_fraction >= 0.8 else "EXP3_INSUFFICIENT_BLOCKWISE_CONTROL_COVERAGE"
    payload = {"status": status, "total_runs": len(FOLDS) * len(SEEDS), "eligible_runs": [{"fold": f, "seed": s} for f, s in eligible_run_ids], "total_protected_blocks": total_blocks, "eligible_protected_blocks": len(eligible_blocks), "eligible_block_fraction": eligible_block_fraction, "failures": failures, "config_sha256": sha256_file(CONFIG_PATH), "split_sha256": split_sha, "device": str(device), "validation_outcome_used": False, **flags()}
    write_json(OUT / "TRAIN_ONLY_DESIGN.json", payload); return payload


def freeze_v11() -> dict[str, Any]:
    path = OUT / "TRAIN_ONLY_DESIGN.json"
    if not path.exists(): raise RuntimeError("run train_only before freeze")
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("status") != "TRAIN_ONLY_DESIGN_READY": raise RuntimeError(f"cannot freeze V1.1 design: {design.get('status')}")
    code_hashes = {str(p.relative_to(EXP_ROOT)): sha256_file(p) for p in sorted((EXP_ROOT / "code").glob("*.py"))}
    payload = {"status": "FROZEN_BEFORE_VALIDATION_OUTCOME", "protocol_sha256": sha256_file(CONFIG_PATH), "code_sha256": code_hashes, "train_only_design_sha256": sha256_file(path), "frozen_features": ["blockwise_causal_unit", "persistence_certification", "matching_rule", "P_anchored_identity_targets", "identity_metric", "coverage_rule", "task_probe", "bootstrap", "gates"], "validation_outcome_used": False, **flags()}
    out = OUT / "PROTOCOL_FREEZE_AUDIT.json"
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != payload["protocol_sha256"] or old.get("code_sha256") != payload["code_sha256"] or old.get("train_only_design_sha256") != payload["train_only_design_sha256"]: raise RuntimeError("post-freeze V1.1 protocol differs")
        return old
    write_json(out, payload); return payload


def require_freeze_v11() -> dict[str, Any]:
    path = OUT / "PROTOCOL_FREEZE_AUDIT.json"
    if not path.exists(): raise RuntimeError("run freeze before validation")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_VALIDATION_OUTCOME": raise RuntimeError("invalid V1.1 freeze audit")
    return payload


def lookup_v11(params: pd.DataFrame, fold: int, seed: int, block: int, group_id: str, dose: str) -> tuple[float, float]:
    rows = params[(params.fold == fold) & (params.seed == seed) & (params.protected_block_id == block) & (params.group_id.astype(str) == str(group_id)) & (params.dose == dose) & params.eligible.astype(bool)]
    if len(rows) != 1: raise RuntimeError(f"missing V1.1 alpha {fold}/{seed}/B{block}/{group_id}/{dose}")
    return float(rows.iloc[0].alpha), float(rows.iloc[0].drop_train)


def run_final_v11() -> dict[str, Any]:
    freeze_record = require_freeze_v11(); design = json.loads((OUT / "TRAIN_ONLY_DESIGN.json").read_text(encoding="utf-8")); ensure_dirs(); splits, split_sha = load_development_splits(); params = pd.read_csv(OUT / "IDENTITY_MATCHING_PARAMETERS.csv"); controls = pd.read_csv(OUT / "BLOCKWISE_MATCHED_CONTROLS.csv"); diagnostics = pd.read_csv(OUT / "BLOCKWISE_MATCHING_DIAGNOSTICS.csv")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("V1.1 final requires CUDA")
    persistence_all, id_raw, task_raw, random_rows, provenance = [], [], [], [], []
    for fold in FOLDS:
        allowed = splits[fold]["train_subjects"] + splits[fold]["validation_subjects"]; manifest = load_development_manifest(allowed)
        for seed in SEEDS:
            train_meta, train_h, val_meta, val_h, feat_prov = extract_or_load_features(fold, seed, splits[fold], manifest, device); spec, assignment, upstream_prov = load_upstream_run(fold, seed); train_q = coordinates(train_h, spec).astype(np.float32); val_q = coordinates(val_h, spec).astype(np.float32); persistence_all.extend(persistence_rows(val_meta, val_q, splits[fold]["validation_subjects"], fold, seed))
            mi_train = (train_meta.paradigm == "mi").to_numpy(); mi_val = (val_meta.paradigm == "mi").to_numpy(); trm, trh, trq = train_meta.loc[mi_train].reset_index(drop=True), train_h[mi_train], train_q[mi_train]; vam, vah, vaq = val_meta.loc[mi_val].reset_index(drop=True), val_h[mi_val], val_q[mi_val]; protected_blocks = list(map(int, assignment["mi"]["protected"])); id_base, id_base_per = symmetric_subject_id_score(vam, vaq, splits[fold]["validation_subjects"], return_per_subject=True); base_ba, base_per = task_score(trm, trh, vam, vah, splits[fold]["train_subjects"], splits[fold]["validation_subjects"])
            run_diags = diagnostics[(diagnostics.fold == fold) & (diagnostics.seed == seed) & diagnostics.block_eligible.astype(bool)]
            for _, drow in run_diags.iterrows():
                block = int(drow.protected_block_id); p_dims = sorted(set(map(int, spec["blocks"][block]))); p_gid = f"P_B{block}"; run_controls = controls[(controls.fold == fold) & (controls.seed == seed) & (controls.protected_block_id == block)].copy()
                for dose in DOSES:
                    alpha_p, _ = lookup_v11(params, fold, seed, block, p_gid, dose); htr_p, qtr_p = suppress_coordinates(trh, trq, spec, p_dims, alpha_p); hva_p, qva_p = suppress_coordinates(vah, vaq, spec, p_dims, alpha_p); id_p, id_p_per = symmetric_subject_id_score(vam, qva_p, splits[fold]["validation_subjects"], return_per_subject=True); ba_p, per_p = task_score(trm, htr_p, vam, hva_p, splits[fold]["train_subjects"], splits[fold]["validation_subjects"])
                    for subject in subject_sort(splits[fold]["validation_subjects"]):
                        id_raw.append({"fold": fold, "seed": seed, "protected_block_id": block, "dose": dose, "group_id": "P", "control_id": "P", "subject_id": subject, "baseline_identity_BA": id_base_per.get(subject), "post_identity_BA": id_p_per.get(subject), "identity_drop": id_base_per.get(subject, np.nan) - id_p_per.get(subject, np.nan), "alpha": alpha_p, **flags()}); task_raw.append({"fold": fold, "seed": seed, "protected_block_id": block, "dose": dose, "group_id": "P", "control_id": "P", "subject_id": subject, "baseline_BA": base_per.get(subject), "post_BA": per_p.get(subject), "harm": base_per.get(subject, np.nan) - per_p.get(subject, np.nan), "alpha": alpha_p, "head": "refit", **flags()})
                    for _, crow in run_controls.iterrows():
                        gid = str(crow.control_id); dims = json.loads(crow.coordinate_ids)
                        try: alpha_n, _ = lookup_v11(params, fold, seed, block, gid, dose)
                        except RuntimeError: continue
                        htr_n, qtr_n = suppress_coordinates(trh, trq, spec, dims, alpha_n); hva_n, qva_n = suppress_coordinates(vah, vaq, spec, dims, alpha_n); id_n, id_n_per = symmetric_subject_id_score(vam, qva_n, splits[fold]["validation_subjects"], return_per_subject=True); ba_n, per_n = task_score(trm, htr_n, vam, hva_n, splits[fold]["train_subjects"], splits[fold]["validation_subjects"])
                        for subject in subject_sort(splits[fold]["validation_subjects"]):
                            id_raw.append({"fold": fold, "seed": seed, "protected_block_id": block, "dose": dose, "group_id": "N", "control_id": gid, "subject_id": subject, "baseline_identity_BA": id_base_per.get(subject), "post_identity_BA": id_n_per.get(subject), "identity_drop": id_base_per.get(subject, np.nan) - id_n_per.get(subject, np.nan), "alpha": alpha_n, **flags()}); task_raw.append({"fold": fold, "seed": seed, "protected_block_id": block, "dose": dose, "group_id": "N", "control_id": gid, "subject_id": subject, "baseline_BA": base_per.get(subject), "post_BA": per_n.get(subject), "harm": base_per.get(subject, np.nan) - per_n.get(subject, np.nan), "alpha": alpha_n, "head": "refit", **flags()})
                rng = np.random.default_rng(stable_seed("exp3-v11-random", fold, seed, block)); rank = len(p_dims); all_dims = np.arange(len(spec["rho"]), dtype=np.int64); alpha_r, _ = lookup_v11(params, fold, seed, block, p_gid, "MEDIUM")
                for ri in range(V11_RANDOM_COUNT):
                    rdims = np.sort(rng.choice(all_dims, size=rank, replace=False)).tolist(); htr_r, _ = suppress_coordinates(trh, trq, spec, rdims, alpha_r); hva_r, _ = suppress_coordinates(vah, vaq, spec, rdims, alpha_r); rb, _ = task_score(trm, htr_r, vam, hva_r, splits[fold]["train_subjects"], splits[fold]["validation_subjects"]); random_rows.append({"fold": fold, "seed": seed, "protected_block_id": block, "random_id": f"R{ri+1:03d}", "coordinate_ids": json.dumps(rdims), "rank": rank, "alpha": alpha_r, "baseline_BA": base_ba, "post_BA": rb, "task_harm": base_ba - rb, **flags()})
                provenance.append({"fold": fold, "seed": seed, "protected_block_id": block, "checkpoint": feat_prov["checkpoint"], "checkpoint_sha256": feat_prov["checkpoint_sha256"], "spectrum_fingerprint": upstream_prov["fingerprint"], "protected_coordinate_ids": p_dims, "train_subjects": splits[fold]["train_subjects"], "validation_subjects": splits[fold]["validation_subjects"], **flags()})
    id_raw_frame = pd.DataFrame(id_raw); task_raw_frame = pd.DataFrame(task_raw); write_csv(OUT / "IDENTITY_MANIPULATION_RAW.csv", id_raw_frame); write_csv(OUT / "TASK_CONSEQUENCE_RAW.csv", task_raw_frame); write_csv(OUT / "HELDOUT_PERSISTENCE_REPLICATION.csv", pd.DataFrame(persistence_all)); write_csv(OUT / "RANDOM_CONTROL_DIAGNOSTICS.csv", pd.DataFrame(random_rows));
    def block_subject(frame: pd.DataFrame, value: str) -> pd.DataFrame:
        if not len(frame): return pd.DataFrame()
        return frame.groupby(["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id"], as_index=False).agg(**{value: (value, "mean")}, baseline=("baseline_" + ("identity_BA" if value == "identity_drop" else "BA"), "mean"))
    id_block = id_raw_frame.groupby(["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), baseline_identity_BA=("baseline_identity_BA", "mean"), post_identity_BA=("post_identity_BA", "mean")); task_block = task_raw_frame.groupby(["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id"], as_index=False).agg(harm=("harm", "mean"), baseline_BA=("baseline_BA", "mean"), post_BA=("post_BA", "mean")); block_join = task_block.merge(id_block[["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id", "identity_drop"]], on=["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id"], how="left"); write_csv(OUT / "IDENTITY_MANIPULATION_RUN_LEVEL.csv", id_block.groupby(["fold", "seed", "protected_block_id", "dose", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), baseline_identity_BA=("baseline_identity_BA", "mean"), post_identity_BA=("post_identity_BA", "mean"))); write_csv(OUT / "TASK_CONSEQUENCE_BLOCK_LEVEL.csv", block_join.groupby(["fold", "seed", "protected_block_id", "dose", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), task_harm=("harm", "mean"), baseline_BA=("baseline_BA", "mean"), post_BA=("post_BA", "mean"))); write_csv(OUT / "TASK_CONSEQUENCE_RUN_LEVEL.csv", block_join.groupby(["fold", "seed", "dose", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), task_harm=("harm", "mean"), baseline_BA=("baseline_BA", "mean"), post_BA=("post_BA", "mean"))); write_csv(OUT / "DOSE_RESPONSE.csv", block_join.groupby(["fold", "seed", "protected_block_id", "dose", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), task_harm=("harm", "mean")))
    write_json(OUT / "PROVENANCE_AUDIT.json", {"runs": provenance, "freeze_record": freeze_record, "split_sha256": split_sha, "validation_outcome_used_for_design": False, **flags()}); write_json(OUT / "DATA_ACCESS_AUDIT_FINAL.json", {"authorized_subject_scope": "development_train_and_development_validation_only", "outer_data_loaded": False, "outer_membership_enumerated": False, "outer_test_used": False}); return {"status": "FINAL_RAW_OUTPUTS_READY", "persistence_rows": len(persistence_all), "identity_rows": len(id_raw_frame), "task_rows": len(task_raw_frame), **flags()}


def bootstrap_v11(frame: pd.DataFrame, column: str, seed: int) -> dict[str, Any]:
    if not len(frame) or column not in frame: return {"mean": None, "median": None, "ci95": [None, None], "sign_probability": None, "n_unique_subjects": 0, "draws": BOOTSTRAP_DRAWS}
    values = frame[["subject_id", column]].dropna().groupby("subject_id", sort=True)[column].mean().to_numpy(float)
    if len(values) == 0: return {"mean": None, "median": None, "ci95": [None, None], "sign_probability": None, "n_unique_subjects": 0, "draws": BOOTSTRAP_DRAWS}
    rng = np.random.default_rng(stable_seed("exp3-v11-bootstrap", seed, column, len(values))); draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "median": float(np.median(values)), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], "sign_probability": float(np.mean(draws > 0)), "n_unique_subjects": int(len(values)), "positive_subject_fraction": float(np.mean(values > 0)), "nonnegative_subject_fraction": float(np.mean(values >= 0)), "worst_subject": float(values.min()), "draws": BOOTSTRAP_DRAWS}


def finalize_v11() -> dict[str, Any]:
    freeze_record = require_freeze_v11(); p = pd.read_csv(OUT / "HELDOUT_PERSISTENCE_REPLICATION.csv"); id_raw = pd.read_csv(OUT / "IDENTITY_MANIPULATION_RAW.csv"); task_raw = pd.read_csv(OUT / "TASK_CONSEQUENCE_RAW.csv"); design = json.loads((OUT / "TRAIN_ONLY_DESIGN.json").read_text(encoding="utf-8")); diag = pd.read_csv(OUT / "BLOCKWISE_MATCHING_DIAGNOSTICS.csv")
    p_sub = p.groupby("subject_id", as_index=False).agg(same_similarity=("same_similarity", "mean"), mismatched_similarity=("mismatched_similarity", "mean"), R_persist=("R_persist", "mean")); write_csv(OUT / "HELDOUT_PERSISTENCE_REPLICATION_SUBJECT_LEVEL.csv", p_sub); p_stats = bootstrap_v11(p_sub, "R_persist", stable_seed("g0-v11")); write_json(OUT / "HELDOUT_PERSISTENCE_STATS.json", p_stats)
    id_block = id_raw.groupby(["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), baseline_identity_BA=("baseline_identity_BA", "mean"), post_identity_BA=("post_identity_BA", "mean")); id_run = id_block.groupby(["fold", "seed", "dose", "group_id", "subject_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), baseline_identity_BA=("baseline_identity_BA", "mean"), post_identity_BA=("post_identity_BA", "mean")); id_subject = id_run.groupby(["subject_id", "dose", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), baseline_identity_BA=("baseline_identity_BA", "mean"), post_identity_BA=("post_identity_BA", "mean")); write_csv(OUT / "IDENTITY_MANIPULATION_RUN_LEVEL.csv", id_run); write_csv(OUT / "IDENTITY_MANIPULATION_SUBJECT_LEVEL.csv", id_subject)
    medium_id = id_subject[id_subject.dose == "MEDIUM"]; id_p = medium_id[medium_id.group_id == "P"]; id_n = medium_id[medium_id.group_id == "N"]; id_piv = medium_id.pivot_table(index="subject_id", columns="group_id", values="identity_drop", aggfunc="mean"); id_diff = pd.DataFrame({"subject_id": id_piv.index.astype(str), "delta_ID_P_minus_N": id_piv.get("P", np.nan) - id_piv.get("N", np.nan)}); id_stats = {"P": bootstrap_v11(id_p.rename(columns={"identity_drop": "value"}), "value", stable_seed("id-p-v11")), "N": bootstrap_v11(id_n.rename(columns={"identity_drop": "value"}), "value", stable_seed("id-n-v11")), "P_minus_N": bootstrap_v11(id_diff.rename(columns={"delta_ID_P_minus_N": "value"}), "value", stable_seed("id-diff-v11")), "tolerance": V11_ID_TOLERANCE}; write_csv(OUT / "IDENTITY_MANIPULATION_PAIRED.csv", id_diff); write_json(OUT / "IDENTITY_MANIPULATION_STATS.json", id_stats)
    task_block = task_raw.groupby(["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id"], as_index=False).agg(harm=("harm", "mean"), baseline_BA=("baseline_BA", "mean"), post_BA=("post_BA", "mean")); task_block = task_block.merge(id_block[["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id", "identity_drop"]], on=["fold", "seed", "protected_block_id", "dose", "group_id", "subject_id"], how="left"); task_run = task_block.groupby(["fold", "seed", "dose", "group_id", "subject_id"], as_index=False).agg(harm=("harm", "mean"), baseline_BA=("baseline_BA", "mean"), post_BA=("post_BA", "mean"), identity_drop=("identity_drop", "mean")); task_subject = task_run.groupby(["subject_id", "dose", "group_id"], as_index=False).agg(harm=("harm", "mean"), baseline_BA=("baseline_BA", "mean"), post_BA=("post_BA", "mean"), identity_drop=("identity_drop", "mean")); task_block.to_csv(OUT / "TASK_CONSEQUENCE_BLOCK_LEVEL.csv", index=False); task_run.to_csv(OUT / "TASK_CONSEQUENCE_RUN_LEVEL.csv", index=False); write_csv(OUT / "TASK_CONSEQUENCE_SUBJECT_LEVEL.csv", task_subject)
    medium_task = task_subject[task_subject.dose == "MEDIUM"]; p_t = medium_task[medium_task.group_id == "P"]; n_t = medium_task[medium_task.group_id == "N"]; piv = medium_task.pivot_table(index="subject_id", columns="group_id", values="harm", aggfunc="mean"); primary_frame = pd.DataFrame({"subject_id": piv.index.astype(str), "Delta_H": piv.get("P", np.nan) - piv.get("N", np.nan)}); hp = bootstrap_v11(p_t.rename(columns={"harm": "value"}), "value", stable_seed("hp-v11")); hn = bootstrap_v11(n_t.rename(columns={"harm": "value"}), "value", stable_seed("hn-v11")); dh = bootstrap_v11(primary_frame.rename(columns={"Delta_H": "value"}), "value", stable_seed("dh-v11")); run_medium = task_run[task_run.dose == "MEDIUM"].groupby(["fold", "seed", "group_id"], as_index=False).agg(harm=("harm", "mean")).pivot_table(index=["fold", "seed"], columns="group_id", values="harm"); run_medium["Delta_H"] = run_medium.get("P", np.nan) - run_medium.get("N", np.nan); run_positive = int(np.sum(run_medium["Delta_H"].to_numpy(float) > 0)); g3 = bool(dh.get("mean") is not None and dh["mean"] >= 0.01 and dh["ci95"][0] > 0 and run_positive >= 5 and float(dh.get("nonnegative_subject_fraction", 0.0)) >= 0.60); primary = {"H_P": hp, "H_N": hn, "Delta_H": dh, "run_level_delta_positive_count": run_positive, "run_level_count": int(len(run_medium)), "gate_G3": g3, "primary_dose": "MEDIUM", **flags()}; write_json(OUT / "PRIMARY_CAUSAL_EFFECT.json", primary); dose_stats = {}
    dose = pd.read_csv(OUT / "DOSE_RESPONSE.csv") if (OUT / "DOSE_RESPONSE.csv").exists() else task_block[["fold", "seed", "protected_block_id", "dose", "group_id", "identity_drop", "harm"]].rename(columns={"harm": "task_harm"})
    for name, frame in dose.groupby("dose"):
        dose_stats[str(name)] = {"P": {"identity_drop": float(frame[frame.group_id == "P"].identity_drop.mean()) if len(frame[frame.group_id == "P"]) else None, "task_harm": float(frame[frame.group_id == "P"].task_harm.mean()) if len(frame[frame.group_id == "P"]) else None}, "N": {"identity_drop": float(frame[frame.group_id == "N"].identity_drop.mean()) if len(frame[frame.group_id == "N"]) else None, "task_harm": float(frame[frame.group_id == "N"].task_harm.mean()) if len(frame[frame.group_id == "N"]) else None}, "n_rows": int(len(frame))}
    write_json(OUT / "DOSE_RESPONSE_STATS.json", dose_stats); g0 = bool(p_stats.get("ci95", [None, None])[0] is not None and p_stats["ci95"][0] > 0); g1 = bool(design.get("status") == "TRAIN_ONLY_DESIGN_READY" and len(diag) == int(design.get("total_protected_blocks", 0)) and float(design.get("eligible_block_fraction", 0.0)) >= 0.8 and len(design.get("eligible_runs", [])) >= 5); id_mean_p = float(id_p.identity_drop.mean()) if len(id_p) else float("nan"); id_mean_n = float(id_n.identity_drop.mean()) if len(id_n) else float("nan"); g2 = bool(np.isfinite(id_mean_p) and np.isfinite(id_mean_n) and abs(id_mean_p - id_mean_n) <= V11_ID_TOLERANCE); terminal = "PERSISTENCE_REPLICATION_FAILED" if not g0 else ("EXP3_INSUFFICIENT_BLOCKWISE_CONTROL_COVERAGE" if not g1 else ("IDENTITY_MATCH_FAILED" if not g2 else ("EXP3_UTILITY_NOT_IDENTITY_SUPPORTED_DEVELOPMENT" if g3 else "PROTECTED_CAUSAL_EFFECT_NOT_SUPPORTED"))); claim = "YES" if terminal == "EXP3_UTILITY_NOT_IDENTITY_SUPPORTED_DEVELOPMENT" else ("NO" if g0 and g1 and g2 else "PARTIAL"); decision = {"terminal_state": terminal, "G0_heldout_persistence": g0, "G1_blockwise_controls": g1, "G2_identity_equivalence": g2, "G3_utility_causal_effect": g3, "mean_delta_ID_P": id_mean_p, "mean_delta_ID_N": id_mean_n, "identity_difference": id_mean_p - id_mean_n if np.isfinite(id_mean_p) and np.isfinite(id_mean_n) else None, "primary": primary, "utility_not_identity_claim": claim, "enter_experiment_4": "YES" if terminal == "EXP3_UTILITY_NOT_IDENTITY_SUPPORTED_DEVELOPMENT" else "NO", "validation_outcome_used_for_design": False, "freeze_record": freeze_record, **flags()}; write_json(OUT / "FINAL_DECISION.json", decision); write_json(OUT / "DATA_ACCESS_AUDIT_FINAL.json", {"authorized_subject_scope": "development_train_and_development_validation_only", "outer_data_loaded": False, "outer_membership_enumerated": False, "outer_test_used": False}); report = make_report_v11(decision, p_stats, id_stats, primary, dose_stats, run_medium, design); (OUT / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8"); make_figures_v11(p, id_subject, task_subject, dose); expected = ["PHASE0_AUDIT.json", "BLOCKWISE_PROTECTED_ASSIGNMENTS.csv", "BLOCKWISE_CONTROL_CANDIDATES.csv", "BLOCKWISE_MATCHED_CONTROLS.csv", "BLOCKWISE_MATCHING_DIAGNOSTICS.csv", "CONTROL_PROVENANCE.json", "CONTROL_PERSISTENCE_AUDIT.csv", "TRAIN_IDENTITY_RESPONSE_CURVES.csv", "IDENTITY_TARGETS.csv", "IDENTITY_MATCHING_PARAMETERS.csv", "PROTOCOL_FREEZE_AUDIT.json", "HELDOUT_PERSISTENCE_REPLICATION.csv", "HELDOUT_PERSISTENCE_STATS.json", "IDENTITY_MANIPULATION_SUBJECT_LEVEL.csv", "IDENTITY_MANIPULATION_RUN_LEVEL.csv", "IDENTITY_MANIPULATION_STATS.json", "TASK_CONSEQUENCE_BLOCK_LEVEL.csv", "TASK_CONSEQUENCE_RUN_LEVEL.csv", "TASK_CONSEQUENCE_SUBJECT_LEVEL.csv", "PRIMARY_CAUSAL_EFFECT.json", "DOSE_RESPONSE.csv", "DOSE_RESPONSE_STATS.json", "RANDOM_CONTROL_DIAGNOSTICS.csv", "DATA_ACCESS_AUDIT.json", "DATA_ACCESS_AUDIT_FINAL.json", "PROVENANCE_AUDIT.json", "FINAL_DECISION.json", "SCIENTIFIC_REPORT.md"]; hashes = {name: sha256_file(OUT / name) for name in expected if (OUT / name).exists()}; write_json(OUT / "REPRODUCIBILITY_AUDIT.json", {"lightweight_output_sha256": hashes, "expected_files": expected, "bootstrap_draws": BOOTSTRAP_DRAWS, "finalize_deterministic": True, **flags()}); return decision


def make_report_v11(decision: Mapping[str, Any], p_stats: Mapping[str, Any], id_stats: Mapping[str, Any], primary: Mapping[str, Any], dose_stats: Mapping[str, Any], run_medium: pd.DataFrame, design: Mapping[str, Any]) -> str:
    hp, hn, dh = primary["H_P"], primary["H_N"], primary["Delta_H"]; failures = design.get("failures", [])
    lines = ["# PERSIST-EEG Experiment 3 V1.1 scientific report", "", "## 1. Redesign rationale", "", "V1 did not obtain a negative causal result: its G1 failure was a train-only union-level feasibility problem (fold 2 / seed 1 had rank(P_union)=8 and only one exact-rank non-Protected supported combination). V1.1 therefore changes only the causal unit to one frozen Protected block; Protected definition, persistence definition and the estimand are unchanged.", "", "## 2. Freeze and scope", "", "The block-wise control rule, train persistence certification, P-anchored identity targets, symmetric identity metric, coverage rule and primary statistics were frozen before validation task outcomes. This is a development-resource closure, not an untouched or independent replication.", f"Frozen Protected blocks: {design.get('total_protected_blocks')}; eligible blocks: {design.get('eligible_protected_blocks')}; eligible runs: {len(design.get('eligible_runs', []))}/6; failures: {json.dumps(failures, ensure_ascii=False)}.", "", "## 3. Gate answers", f"- G0 held-out persistence: `{decision['G0_heldout_persistence']}`; mean={p_stats.get('mean')}, 95% CI={p_stats.get('ci95')}, subjects={p_stats.get('n_unique_subjects')}.", f"- G1 block-wise controls: `{decision['G1_blockwise_controls']}`.", f"- G2 identity equivalence: `{decision['G2_identity_equivalence']}`; ΔID_P={decision.get('mean_delta_ID_P')}, ΔID_N={decision.get('mean_delta_ID_N')}, difference={decision.get('identity_difference')}.", f"- G3 causal task effect: `{decision['G3_utility_causal_effect']}`.", "", "## 4. Primary MEDIUM endpoint", f"H_P={hp.get('mean')} (median {hp.get('median')}, CI {hp.get('ci95')}); H_N={hn.get('mean')} (median {hn.get('median')}, CI {hn.get('ci95')}); ΔH={dh.get('mean')} (median {dh.get('median')}, CI {dh.get('ci95')}, sign probability {dh.get('sign_probability')}, positive-subject fraction {dh.get('positive_subject_fraction')}, nonnegative-subject fraction {dh.get('nonnegative_subject_fraction')}, worst subject {dh.get('worst_subject')}); positive run means={primary.get('run_level_delta_positive_count')}/{primary.get('run_level_count')}.", "", "## 5. Dose and secondary controls", f"Dose-response: {json.dumps(dose_stats, ensure_ascii=False)}.", "Same-rank random-control diagnostics are in `RANDOM_CONTROL_DIAGNOSTICS.csv`; Neutral-only is not a primary comparator and was not used to alter coverage or gates.", "", "## 6. Leakage and final claim", "All artifacts set `outer_test_used=false` and `outer_membership_enumerated=false`; no validation task outcome was used for matching, control eligibility, alpha, dose, metric or gate selection.", f"Terminal state: `{decision['terminal_state']}`.", f"Utility-not-identity claim: `{decision['utility_not_identity_claim']}`.", f"READY_FOR_EXPERIMENT_4: `{decision['enter_experiment_4']}`.", ""]
    if decision["G3_utility_causal_effect"] is False and decision["G1_blockwise_controls"] and decision["G2_identity_equivalence"]: lines.append("After matching subject-identity removal, the frozen Protected assignment does not meet the predeclared evidence threshold for greater task harm than matched Non-Protected persistence.")
    return "\n".join(lines)


def make_figures_v11(persistence: pd.DataFrame, identity: pd.DataFrame, task: pd.DataFrame, dose: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    if len(persistence):
        fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(persistence.same_similarity, persistence.mismatched_similarity, alpha=.7); lo = min(persistence.same_similarity.min(), persistence.mismatched_similarity.min()); hi = max(persistence.same_similarity.max(), persistence.mismatched_similarity.max()); ax.plot([lo, hi], [lo, hi], "k--", lw=.8); ax.set_xlabel("same-subject cosine"); ax.set_ylabel("mismatched-subject cosine"); ax.set_title("Held-out persistence replication"); fig.tight_layout(); fig.savefig(FIGURES / "figure_A_heldout_persistence.png", dpi=180); plt.close(fig)
    if len(identity):
        med = identity[identity.dose == "MEDIUM"]; fig, ax = plt.subplots(figsize=(6, 4)); med.boxplot(column="identity_drop", by="group_id", ax=ax); ax.set_title("Identity removal at matched MEDIUM dose"); ax.set_ylabel("cross-session subject-ID BA drop"); fig.suptitle(""); fig.tight_layout(); fig.savefig(FIGURES / "figure_B_identity_matching.png", dpi=180); plt.close(fig)
    if len(task):
        med = task[task.dose == "MEDIUM"]; fig, ax = plt.subplots(figsize=(5, 4)); vals = [med[med.group_id == "P"].harm.mean(), med[med.group_id == "N"].harm.mean()]; ax.bar(["Protected", "Matched N"], vals); ax.set_ylabel("task BA harm"); ax.set_title("Primary matched causal endpoint"); fig.tight_layout(); fig.savefig(FIGURES / "figure_C_primary_task_harm.png", dpi=180); plt.close(fig)
    if len(dose):
        summary = dose.groupby(["dose", "group_id"], as_index=False).agg(identity_drop=("identity_drop", "mean"), task_harm=("task_harm", "mean")); fig, ax = plt.subplots(figsize=(6, 4));
        for gid, group in summary.groupby("group_id"): ax.plot(group.identity_drop, group.task_harm, marker="o", label=gid)
        ax.set_xlabel("actual identity reduction"); ax.set_ylabel("task BA harm"); ax.legend(); ax.set_title("Dose response"); fig.tight_layout(); fig.savefig(FIGURES / "figure_D_dose_response.png", dpi=180); plt.close(fig)


def terminal_v11() -> dict[str, Any]:
    design = json.loads((OUT / "TRAIN_ONLY_DESIGN.json").read_text(encoding="utf-8")); decision = {"terminal_state": "EXP3_INSUFFICIENT_BLOCKWISE_CONTROL_COVERAGE", "G0_heldout_persistence": None, "G1_blockwise_controls": False, "G2_identity_equivalence": None, "G3_utility_causal_effect": None, "utility_not_identity_claim": "PARTIAL", "enter_experiment_4": "NO", "failures": design.get("failures", []), **flags()}; write_json(OUT / "FINAL_DECISION.json", decision); (OUT / "SCIENTIFIC_REPORT.md").write_text("# PERSIST-EEG Experiment 3 V1.1\n\nTrain-only block coverage failed before validation outcome evaluation.\n\n" + json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"); return decision


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["phase0", "train_only", "freeze", "final", "finalize", "terminal", "all"]); args = parser.parse_args()
    if args.phase in {"phase0", "all"}: phase0(); write_json(OUT / "PHASE0_AUDIT.json", json.loads((OUT / "DATA_ACCESS_AUDIT.json").read_text(encoding="utf-8")))
    if args.phase in {"train_only", "all"}: build_train_only_v11()
    if args.phase in {"freeze", "all"}: freeze_v11()
    if args.phase in {"final", "all"}: run_final_v11()
    if args.phase in {"finalize", "all"}: print(json.dumps(clean(finalize_v11()), ensure_ascii=False, indent=2))
    if args.phase == "terminal": print(json.dumps(clean(terminal_v11()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
