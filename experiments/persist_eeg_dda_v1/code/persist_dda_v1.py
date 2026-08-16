"""PERSIST-EEG Decision Dependence Audit V1.

The implementation is intentionally fail-closed.  It reuses frozen
Signed-V3.1 canonical blocks, P5.1 V2 matched controls, and PERSIST-CF stress
banks.  It never loads an OpenBMI development-validation or outer-test row.

Primary statistical units are subjects inside cross-fits and run/block cells
in the incremental analysis.  Trials are aggregated before inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_dda_v1"
OUT = EXP_ROOT / "outputs"
P5_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_icg"
P56_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_1_p6"
ROUTER_ROOT = REPO_ROOT / "experiments" / "persist_eeg_router"
CF_ROOT = REPO_ROOT / "experiments" / "persist_eeg_cf"
SHARED_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p4_shared_geometry" / "results_v1_2"
REFERENCE_PARENT_COMMIT = "39c35b43f4a5112e1a774190f4ec444e36719b15"
IMPLEMENTATION_ID = "persist_dda_v1_confirmatory_20260817"
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
AUDIT_FOLDS = 5
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 5_000
PERMUTATION_DRAWS = 5_000
EPS = 1e-12


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P5 = _import_file("dda_frozen_p5", P5_ROOT / "code" / "p5_icg.py")
P56 = _import_file("dda_frozen_p56", P56_ROOT / "code" / "p5_1_p6.py")
ROUTER = _import_file("dda_frozen_router", ROUTER_ROOT / "code" / "persist_router.py")
CF = _import_file("dda_frozen_cf", CF_ROOT / "code" / "persist_cf.py")


PROTOCOL = {
    "implementation_id": IMPLEMENTATION_ID,
    "scientific_scope": "OpenBMI MI TRAIN-only decision-dependence audit",
    "frozen_classifier": "P5.1 V2 matched continued-training control",
    "canonical_assignment": "Signed V3.1; no reassignment permitted",
    "primary_finite_metric": "subject-mean RMS centered-logit displacement under finite erasure/offset",
    "primary_local_metric": "centered-logit Jacobian energy per canonical dimension",
    "primary_probability_metric": "total variation",
    "statistical_units": {
        "dda_a": "held subject; hierarchically aggregated within run",
        "dda_b": "run x frozen Protected block after subject aggregation",
        "dda_c": "run x audit-fold x block; outcome and decision subjects disjoint",
    },
    "controls": {
        "dda_a": "100 exact-norm random offsets in the same TRAIN-derived blockwise G-perp admissible space",
        "dda_b": "100 same-rank random subspaces with per-sample representation-displacement norm matching, plus same-rank non-PROTECTED blocks",
    },
    "dda_a_gate": {
        "representation_q_movement_relative_lower_onesided_95_min": 0.05,
        "equivalence_logit_rms_over_train_margin_upper_onesided_95_max": 0.10,
        "equivalence_flip_rate_upper_onesided_95_max": 0.01,
        "equivalence_total_variation_upper_onesided_95_max": 0.01,
        "alternative_matched_random_logit_ratio_upper_onesided_95_max": 0.50,
        "rule": "representation movement AND (all three absolute equivalence tests OR matched-random ratio test)",
    },
    "dda_b_gate": {
        "jacobian_ratio_bootstrap_lower_95_min": 1.0,
        "finite_logit_ratio_bootstrap_lower_95_min": 1.0,
        "minimum_positive_runs_of_6": 4,
        "minimum_protected_gt_matched_nonprotected_runs_of_6": 4,
        "minimum_directionally_concordant_runs_of_6": 4,
    },
    "dda_c_gate": {
        "model": "ridge(alpha=1) with train-fold standardization",
        "baseline": ["persistence_strength", "geometry_strength", "rank"],
        "incremental_feature": "decision_logit_rms",
        "target": "held-subject mean signed CE consequence of block erasure",
        "loro_relative_rmse_improvement_min": 0.05,
        "minimum_improved_runs_of_6": 4,
        "run_cluster_bootstrap_improvement_lower_95_min": 0.0,
        "permutation_p_max": 0.05,
        "minimum_positive_coefficients_of_6": 5,
    },
    "crossfit": {
        "dda_a": "original frozen CF 5-fold train/held banks",
        "dda_bc": "for k=0..4: decision=k, outcome=(k+1)%5, model-fit=remaining 3 subject folds",
    },
    "outer_test_used": False,
    "development_validation_used": False,
    "agdi_training_authorized_before_results": False,
}


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**32 - 1)


def seed_all(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), EPS)


def ce_per_row(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = softmax(logits)
    flat = p.reshape(-1, p.shape[-1])
    yy = np.broadcast_to(np.asarray(y, dtype=np.int64), p.shape[:-1]).reshape(-1)
    return (-np.log(np.clip(flat[np.arange(len(flat)), yy], EPS, 1.0))).reshape(p.shape[:-1])


def true_margin(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    yy = np.broadcast_to(np.asarray(y, dtype=np.int64), z.shape[:-1])
    own = np.take_along_axis(z, yy[..., None], axis=-1)[..., 0]
    masked = z.copy()
    np.put_along_axis(masked, yy[..., None], -np.inf, axis=-1)
    return own - masked.max(axis=-1)


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64); pred = np.asarray(pred, dtype=np.int64)
    values = [float(np.mean(pred[y == c] == c)) for c in np.unique(y) if np.any(y == c)]
    return float(np.mean(values)) if values else float("nan")


def centered_logit_sq(delta: np.ndarray) -> np.ndarray:
    value = np.asarray(delta, dtype=np.float64)
    centered = value - value.mean(axis=-1, keepdims=True)
    return np.sum(centered * centered, axis=-1)


def js_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa = np.clip(softmax(a), EPS, 1.0); pb = np.clip(softmax(b), EPS, 1.0)
    m = 0.5 * (pa + pb)
    return 0.5 * np.sum(pa * np.log(pa / m), axis=-1) + 0.5 * np.sum(pb * np.log(pb / m), axis=-1)


def current_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except Exception:
        return None


def artifact_row(kind: str, path: Path, split: str, train_only: bool,
                 classifier_frozen: bool, regenerate: bool = False) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "kind": kind,
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "split": split,
        "train_only_construction": bool(train_only),
        "classifier_frozen": bool(classifier_frozen),
        "test_subject_information": False,
        "regeneration_required": bool(regenerate),
    }


def assignment_path(fold: int, seed: int) -> Path:
    return P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_ASSIGNMENTS_V3_1.json"


def utility_path(fold: int, seed: int) -> Path:
    return P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_UTILITY_V3_1.csv"


def spectrum_path(fold: int, seed: int) -> Path:
    return P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "spectrum" / "PERSISTENCE_SPECTRUM.npz"


def audit_provenance() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    result_markers = list((OUT / "results").glob("*.csv")) if (OUT / "results").exists() else []
    lock_path = OUT / "protocol" / "DDA_PROTOCOL_LOCK.json"
    if result_markers and not lock_path.exists():
        raise RuntimeError("DDA results exist without a prior protocol lock; refusing retrospective freeze")
    meta = P5.load_mi_manifest()
    artifacts: list[dict[str, Any]] = []
    artifacts.append(artifact_row("OpenBMI trial manifest", P5.MANIFEST, "all rows indexed; only outer-TRAIN selected", True, True))
    artifacts.append(artifact_row("split freeze", P5.SPLIT, "frozen subject split", True, True))
    split_rows = []
    assignment_rows = []
    for fold in FOLDS:
        split = P5.load_split(fold)
        tr = set(map(str, split["train_subjects"])); va = set(map(str, split["validation_subjects"]))
        if tr & va or len(tr) != 34 or not va or len(tr | va) >= int(meta.subject.nunique()):
            raise RuntimeError(f"Invalid fold split {fold}: train={len(tr)} validation={len(va)} overlap={len(tr & va)}")
        split_rows.append({"fold": fold, "train_subjects": sorted(tr, key=int),
                           "development_validation_subjects": sorted(va, key=int),
                           "n_manifest_subjects": int(meta.subject.nunique()),
                           "n_outer_test_subjects_locked_not_loaded": int(meta.subject.nunique()) - len(tr | va)})
        for seed in SEEDS:
            checkpoint, mean, std = P5.historical_checkpoint(fold, seed)
            artifacts.extend([
                artifact_row("historical EEGNet checkpoint", checkpoint, f"fold-{fold} TRAIN", True, True),
                artifact_row("channel normalization", mean, f"fold-{fold} TRAIN", True, True),
                artifact_row("channel normalization", std, f"fold-{fold} TRAIN", True, True),
                artifact_row("Signed V3.1 spectrum", spectrum_path(fold, seed), f"fold-{fold} TRAIN", True, True),
                artifact_row("Signed V3.1 assignment", assignment_path(fold, seed), f"fold-{fold} TRAIN", True, True),
                artifact_row("Signed V3.1 utility", utility_path(fold, seed), f"fold-{fold} inner subject CV", True, True),
            ])
            h_path = P5.OUT / "cache" / f"fold-{fold}" / f"seed-{seed}" / "h0.npy"
            h = np.load(h_path, mmap_mode="r", allow_pickle=False)
            if h.shape != (10_800, 128) or not np.isfinite(np.asarray(h[::500])).all():
                raise RuntimeError(f"Invalid frozen representation cache {h_path}: {h.shape}")
            artifacts.append(artifact_row("frozen h0 cache", h_path, "manifest-aligned; selection by outer-TRAIN only", True, True))
            ckpt = P56_ROOT / "outputs" / "V2" / f"fold-{fold}" / f"seed-{seed}" / "best_control.pt"
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            if payload.get("outer_test_used") is not False or payload.get("version") != "V2":
                raise RuntimeError(f"Non-frozen or invalid V2 checkpoint {ckpt}")
            artifacts.append(artifact_row("P5.1 V2 matched-control checkpoint", ckpt, f"fold-{fold} TRAIN", True, True))
            assignment = json.loads(assignment_path(fold, seed).read_text(encoding="utf-8"))
            if not assignment.get("mi", {}).get("protected"):
                raise RuntimeError(f"Missing frozen MI Protected assignment fold={fold} seed={seed}")
            assignment_rows.append({"fold": fold, "seed": seed, **assignment["mi"]})
    selection_path = P56_ROOT / "outputs" / "P5_1_SELECTED_CONFIGS.csv"
    base_selection_path = P56_ROOT / "outputs" / "protocol" / "P6_BASE_VERSION_SELECTION.json"
    stress_path = CF_ROOT / "outputs" / "protocol" / "STRESS_BANK_FREEZE.json"
    cf_report_path = CF_ROOT / "outputs" / "final" / "PERSIST_CF_FINAL_REPORT.json"
    artifacts.extend([
        artifact_row("P5.1 selected configs", selection_path, "nested outer-TRAIN subject CV", True, True),
        artifact_row("P6 base-version lock", base_selection_path, "TRAIN-only", True, True),
        artifact_row("PERSIST-CF stress-bank freeze", stress_path, "inner-TRAIN donors", True, True),
        artifact_row("PERSIST-CF final report", cf_report_path, "TRAIN-only", True, True),
    ])
    geometry_csv = SHARED_ROOT / "BLOCK_GEOMETRY_UTILITY.csv"
    if geometry_csv.exists():
        artifacts.append(artifact_row("Shared Geometry V1.2 block table", geometry_csv, "TRAIN/development audit; no outer test", False, True))
    stress = json.loads(stress_path.read_text(encoding="utf-8"))
    if stress.get("outer_test_used") is not False or stress.get("held_subject_statistics_or_labels_used") is not False:
        raise RuntimeError("PERSIST-CF stress bank is not a legal TRAIN-only freeze")
    if len(stress.get("banks", [])) != 30:
        raise RuntimeError(f"Expected 30 CF banks, found {len(stress.get('banks', []))}")
    for bank in stress["banks"]:
        tr, held = set(map(str, bank["train_subjects"])), set(map(str, bank["held_subjects"]))
        outer_tr = set(P5.load_split(int(bank["fold"]))["train_subjects"])
        if tr & held or tr | held != outer_tr or len(bank["pairs"]) < 1:
            raise RuntimeError(f"Illegal CF bank fold={bank['fold']} seed={bank['seed']} inner={bank['inner_fold']}")
    payload = {
        "status": "DDA_PROVENANCE_AUDIT_PASS",
        "implementation_id": IMPLEMENTATION_ID,
        "reference_parent_commit": REFERENCE_PARENT_COMMIT,
        "runtime_git_commit": current_git_commit(),
        "code_sha256": sha256(Path(__file__)),
        "artifacts": artifacts,
        "splits": split_rows,
        "frozen_mi_assignments": assignment_rows,
        "checks": {
            "manifest_rows": len(meta), "manifest_subjects": int(meta.subject.nunique()),
            "all_h0_shapes_10800x128": True, "all_assignments_frozen": True,
            "cf_banks": len(stress["banks"]), "development_validation_loaded": False,
            "outer_test_loaded": False, "backbone_retraining_required": False,
        },
        "known_limitation": "Canonical discovery and Signed-V3.1 assignment predate DDA and use the full outer-TRAIN split; DDA cross-fitting separates classifier fit, decision measurement, and consequence measurement but does not rediscover the canonical basis.",
    }
    write_json(OUT / "protocol" / "PROVENANCE_AUDIT.json", payload)
    lock_payload = {
        **PROTOCOL,
        "frozen_before_any_dda_result": not bool(result_markers),
        "provenance_audit_sha256": sha256(OUT / "protocol" / "PROVENANCE_AUDIT.json"),
        "source_code_sha256": sha256(Path(__file__)),
        "created_utc_unix": int(time.time()),
    }
    if lock_path.exists():
        old = json.loads(lock_path.read_text(encoding="utf-8"))
        for key, value in PROTOCOL.items():
            if old.get(key) != value:
                raise RuntimeError(f"Existing protocol lock differs at {key}; refusing mutation")
        current_code_sha = sha256(Path(__file__))
        if old.get("source_code_sha256") != current_code_sha:
            if result_markers:
                raise RuntimeError("Source changed after DDA results existed; refusing silent protocol mutation")
            adaptation_path = OUT / "protocol" / "DDA_ADAPTATION_LOG.json"
            adaptation = json.loads(adaptation_path.read_text(encoding="utf-8")) if adaptation_path.exists() else {"events": []}
            adaptation["events"].append({
                "stage": "pre-result provenance repair",
                "old_source_code_sha256": old.get("source_code_sha256"),
                "new_source_code_sha256": current_code_sha,
                "scientific_gates_changed": False,
                "reason": "Aggregate multiple frozen PROTECTED blocks to the declared six run-level DDA-B units.",
            })
            write_json(adaptation_path, adaptation)
            old["source_code_sha256"] = current_code_sha
            old["implementation_repaired_before_any_dda_result"] = True
            write_json(lock_path, old)
    else:
        write_json(lock_path, lock_payload)
    print(json.dumps(clean(payload), indent=2), flush=True)
    return payload


def require_lock() -> dict[str, Any]:
    path = OUT / "protocol" / "DDA_PROTOCOL_LOCK.json"
    if not path.exists():
        raise RuntimeError("Run the audit phase first; DDA protocol is not locked")
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key, value in PROTOCOL.items():
        if lock.get(key) != value:
            raise RuntimeError(f"Protocol lock mismatch at {key}")
    return lock


def positions(meta: pd.DataFrame, subjects: Sequence[str]) -> np.ndarray:
    wanted = set(map(str, subjects))
    return np.flatnonzero(meta.subject.astype(str).isin(wanted).to_numpy())


def fit_v2_control(run: Any, cfg: Any, train_pos: np.ndarray, tag: str,
                   device: torch.device) -> Any:
    cache = OUT / "cache" / "geometry_targets" / f"fold-{run.fold}" / f"seed-{run.seed}" / f"{tag}.npz"
    targets = P5.build_geometry_targets(run.meta, run.q, train_pos, run.art, cache)
    model = ROUTER.initialise_v2_control(run, cfg, targets, f"dda-{tag}", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3)
    train_meta = run.meta.iloc[train_pos].reset_index(drop=True)
    h = torch.as_tensor(np.asarray(run.h[train_pos], dtype=np.float32), device=device)
    q = torch.as_tensor(np.asarray(run.q[train_pos], dtype=np.float32), device=device)
    y = torch.as_tensor(train_meta.label.to_numpy(dtype=np.int64), dtype=torch.long, device=device)
    sampler = P5.StructuredSampler(train_meta, train_meta.subject.unique().tolist(),
                                   subjects_per_batch=6, trials_per_class=4)
    for epoch in range(int(cfg.epochs)):
        model.train()
        batches = sampler.batches(epoch, stable_seed(IMPLEMENTATION_ID, run.fold, run.seed, tag, "sampler"))
        for batch in batches:
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, delta = model(h.index_select(0, idx), q.index_select(0, idx))
            loss = F.cross_entropy(logits, y.index_select(0, idx)) + cfg.lambda_drift * P5.drift_loss(delta, run.art)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model.eval()


def forward(model: Any, h: np.ndarray, q: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return ROUTER.forward_model(model, h, q, device)


def full_shift_logits(model: Any, h: np.ndarray, q: np.ndarray, deltas: np.ndarray,
                      device: torch.device) -> np.ndarray:
    deltas = np.asarray(deltas, dtype=np.float32)
    n_shift, n = len(deltas), len(h)
    shifted_q = (np.asarray(q, dtype=np.float32)[None, :, :] + deltas[:, None, :]).reshape(n_shift * n, -1)
    dh = inverse_q_delta(deltas, model)
    shifted_h = (np.asarray(h, dtype=np.float32)[None, :, :] + dh[:, None, :]).reshape(n_shift * n, -1)
    logits, _, _ = forward(model, shifted_h, shifted_q, device)
    return logits.reshape(n_shift, n, -1)


def matched_random_offsets(real: np.ndarray, geometry: Any, art: Any, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros_like(real, dtype=np.float32)
    for i, delta in enumerate(np.asarray(real, dtype=np.float32)):
        for block in sorted(geometry.block_dims):
            dims = geometry.block_dims[block]
            target = delta[dims]
            target_norm = float(np.linalg.norm(target))
            if target_norm <= EPS:
                continue
            g = np.asarray(geometry.directions[block], dtype=np.float64)
            value = rng.normal(size=len(dims))
            value = value - g * float(value @ g)
            norm = float(np.linalg.norm(value))
            if norm <= EPS:
                value = np.roll(g, 1); value = value - g * float(value @ g); norm = float(np.linalg.norm(value))
            out[i, dims] = (value / max(norm, EPS) * target_norm).astype(np.float32)
        real_h = CF.inverse_delta(delta[None, :], art)[0]
        random_h = CF.inverse_delta(out[i:i + 1], art)[0]
        scale = float(np.linalg.norm(real_h) / max(np.linalg.norm(random_h), EPS))
        out[i] *= scale
    return out


def finite_subject_rows(clean_logits: np.ndarray, shifted_logits: np.ndarray, y: np.ndarray,
                        subjects: np.ndarray, run_info: Mapping[str, Any], margin_scale: float,
                        q_movement_relative: float | None = None) -> list[dict[str, Any]]:
    clean = np.asarray(clean_logits, dtype=np.float64)
    shifted = np.asarray(shifted_logits, dtype=np.float64)
    rows = []
    for subject in sorted(set(map(str, subjects)), key=int):
        idx = np.flatnonzero(np.asarray(subjects, dtype=str) == subject)
        base = clean[idx]
        alt = shifted[:, idx, :]
        delta = alt - base[None, :, :]
        p0 = softmax(base)[None, :, :]; p1 = softmax(alt)
        rows.append({
            **run_info, "subject": subject, "n_trials": int(len(idx)),
            "logit_rms": float(np.sqrt(np.mean(centered_logit_sq(delta)))),
            "logit_rms_over_train_margin": float(np.sqrt(np.mean(centered_logit_sq(delta))) / max(margin_scale, EPS)),
            "margin_displacement": float(np.mean(np.abs(true_margin(alt, y[idx]) - true_margin(base, y[idx])[None, :]))),
            "flip_rate": float(np.mean(alt.argmax(-1) != base.argmax(-1)[None, :])),
            "total_variation": float(np.mean(0.5 * np.sum(np.abs(p1 - p0), axis=-1))),
            "js_divergence": float(np.mean(js_divergence(alt, base[None, :, :]))),
            "q_movement_relative": q_movement_relative,
        })
    return rows


def hierarchical_bootstrap(frame: pd.DataFrame, column: str, seed: int,
                           draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    run_keys = sorted(frame.run.unique().tolist())
    groups = {run: frame[frame.run == run][column].to_numpy(dtype=np.float64) for run in run_keys}
    rng = np.random.default_rng(int(seed)); out = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        picked = rng.choice(run_keys, size=len(run_keys), replace=True)
        values = []
        for run in picked:
            arr = groups[str(run)]
            values.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        out[b] = float(np.mean(values))
    return out


def run_dda_a(device: torch.device) -> dict[str, Any]:
    require_lock()
    meta = P5.load_mi_manifest(); bases = ROUTER.selected_bases()
    stress = json.loads((CF_ROOT / "outputs" / "protocol" / "STRESS_BANK_FREEZE.json").read_text(encoding="utf-8"))
    real_rows: list[dict[str, Any]] = []; random_rows: list[dict[str, Any]] = []
    for bank_index, bank in enumerate(stress["banks"]):
        fold, seed, inner = int(bank["fold"]), int(bank["seed"]), int(bank["inner_fold"])
        run = ROUTER.load_run(fold, seed, meta)
        train_pos = positions(meta, bank["train_subjects"]); held_pos = positions(meta, bank["held_subjects"])
        geometry = CF.fit_geometry(run, train_pos)
        model = fit_v2_control(run, bases[(fold, seed)], train_pos, f"dda-a-inner-{inner}", device)
        clean_logits, _, _ = forward(model, run.h[held_pos], run.q[held_pos], device)
        train_logits, _, _ = forward(model, run.h[train_pos], run.q[train_pos], device)
        train_y = meta.iloc[train_pos].label.to_numpy(dtype=np.int64)
        margin_scale = float(np.median(np.abs(true_margin(train_logits, train_y))))
        deltas = np.asarray([pair["delta_q"] for pair in bank["pairs"]], dtype=np.float32)
        q_scale = float(np.median(np.linalg.norm(run.q[train_pos], axis=1)))
        q_move = float(np.mean(np.linalg.norm(deltas, axis=1)) / max(q_scale, EPS))
        shifted = full_shift_logits(model, run.h[held_pos], run.q[held_pos], deltas, device)
        held_meta = meta.iloc[held_pos].reset_index(drop=True)
        info = {"fold": fold, "seed": seed, "inner_fold": inner, "run": f"{fold}_{seed}", "kind": "real_cf"}
        real_rows.extend(finite_subject_rows(
            clean_logits, shifted, held_meta.label.to_numpy(dtype=np.int64), held_meta.subject.to_numpy(dtype=str),
            info, margin_scale, q_move,
        ))
        for draw in range(RANDOM_DRAWS):
            random_delta = matched_random_offsets(
                deltas, geometry, run.art, stable_seed(IMPLEMENTATION_ID, "dda-a-random", fold, seed, inner, draw),
            )
            random_shifted = full_shift_logits(model, run.h[held_pos], run.q[held_pos], random_delta, device)
            rinfo = {"fold": fold, "seed": seed, "inner_fold": inner, "run": f"{fold}_{seed}",
                     "kind": "matched_random", "draw": draw}
            random_rows.extend(finite_subject_rows(
                clean_logits, random_shifted, held_meta.label.to_numpy(dtype=np.int64), held_meta.subject.to_numpy(dtype=str),
                rinfo, margin_scale, q_move,
            ))
        print(f"[DDA-A] bank={bank_index+1}/30 fold={fold} seed={seed} inner={inner}", flush=True)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    real = pd.DataFrame(real_rows); random_frame = pd.DataFrame(random_rows)
    keys = ["fold", "seed", "inner_fold", "run", "subject"]
    random_mean = random_frame.groupby(keys, as_index=False).agg(
        random_logit_rms=("logit_rms", "mean"), random_flip_rate=("flip_rate", "mean"),
        random_total_variation=("total_variation", "mean"),
    )
    real = real.merge(random_mean, on=keys, how="left", validate="one_to_one")
    real["logit_ratio_to_random"] = real.logit_rms / np.maximum(real.random_logit_rms, EPS)
    result_dir = OUT / "results"; result_dir.mkdir(parents=True, exist_ok=True)
    real.to_csv(result_dir / "DDA_A_SUBJECT.csv", index=False)
    random_frame.to_csv(result_dir / "DDA_A_RANDOM_SUBJECT.csv", index=False)
    boot = {column: hierarchical_bootstrap(real, column, stable_seed("dda-a-bootstrap", column))
            for column in ("q_movement_relative", "logit_rms_over_train_margin", "flip_rate", "total_variation", "logit_ratio_to_random")}
    ci = {column: {"mean": float(real[column].mean()),
                   "ci90_onesided_equivalent": [float(np.quantile(values, .05)), float(np.quantile(values, .95))],
                   "ci95_twosided": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}
          for column, values in boot.items()}
    gate = PROTOCOL["dda_a_gate"]
    movement = bool(ci["q_movement_relative"]["ci90_onesided_equivalent"][0] > gate["representation_q_movement_relative_lower_onesided_95_min"])
    equivalence = bool(
        ci["logit_rms_over_train_margin"]["ci90_onesided_equivalent"][1] <= gate["equivalence_logit_rms_over_train_margin_upper_onesided_95_max"]
        and ci["flip_rate"]["ci90_onesided_equivalent"][1] <= gate["equivalence_flip_rate_upper_onesided_95_max"]
        and ci["total_variation"]["ci90_onesided_equivalent"][1] <= gate["equivalence_total_variation_upper_onesided_95_max"]
    )
    relative = bool(ci["logit_ratio_to_random"]["ci90_onesided_equivalent"][1] <= gate["alternative_matched_random_logit_ratio_upper_onesided_95_max"])
    passed = bool(movement and (equivalence or relative))
    report = {
        "status": "DDA_A_PASS" if passed else "DDA_A_FAIL", "passed": passed,
        "n_subject_run_rows": len(real), "n_random_subject_rows": len(random_frame),
        "bootstrap": ci, "movement_nontrivial": movement, "absolute_equivalence": equivalence,
        "significantly_below_matched_random": relative,
        "equivalence_test": "TOST-equivalent one-sided 95% upper bounds; nonsignificance is not used as evidence of null",
        "outer_test_used": False,
    }
    write_json(result_dir / "DDA_A_RESULT.json", report)
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def random_basis(q_dim: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed)); matrix = rng.normal(size=(q_dim, rank))
    q, _ = np.linalg.qr(matrix)
    return np.asarray(q[:, :rank], dtype=np.float64)


def inverse_q_delta(delta_q: np.ndarray, model: Any) -> np.ndarray:
    directions = model.directions.detach().cpu().numpy().astype(np.float64)
    dewhitener = model.dewhitener.detach().cpu().numpy().astype(np.float64)
    value = np.asarray(delta_q, dtype=np.float64)
    return ((value @ directions.T) @ dewhitener).astype(np.float32)


def linear_erase_logits(clean_logits: np.ndarray, delta_q: np.ndarray, model: Any) -> np.ndarray:
    dh = inverse_q_delta(delta_q, model)
    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
    return np.asarray(clean_logits, dtype=np.float64) - np.asarray(dh, dtype=np.float64) @ weight.T


def jacobian_margin(model: Any, h: np.ndarray, q: np.ndarray, device: torch.device) -> np.ndarray:
    output = []
    for start in range(0, len(h), 2048):
        ht = torch.as_tensor(np.asarray(h[start:start + 2048], dtype=np.float32), device=device)
        qt = torch.as_tensor(np.asarray(q[start:start + 2048], dtype=np.float32), device=device).requires_grad_(True)
        logits, _, _ = model(ht, qt)
        if logits.shape[1] != 2:
            raise RuntimeError("Confirmatory DDA V1 Jacobian implementation expects binary OpenBMI MI")
        grad = torch.autograd.grad((logits[:, 1] - logits[:, 0]).sum(), qt, create_graph=False)[0]
        output.append(grad.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(output)


def block_geometry_alignment(meta: pd.DataFrame, q: np.ndarray, pos: np.ndarray,
                             dims: Sequence[int]) -> float:
    m = meta.iloc[pos].reset_index(drop=True); x = np.asarray(q[pos], dtype=np.float64)
    contrasts: dict[tuple[str, str], np.ndarray] = {}
    for (subject, session), group in m.groupby(["subject", "session"], sort=True):
        idx = group.index.to_numpy(dtype=np.int64); y = m.label.to_numpy(dtype=np.int64)[idx]
        if not (np.any(y == 0) and np.any(y == 1)): continue
        contrasts[(str(subject), str(session))] = x[idx[y == 1]][:, dims].mean(0) - x[idx[y == 0]][:, dims].mean(0)
    scores = []
    for (subject, session), value in contrasts.items():
        others = [v for (s, r), v in contrasts.items() if s != subject and r == session]
        if not others: continue
        consensus = np.mean(others, axis=0)
        den = float(np.linalg.norm(value) * np.linalg.norm(consensus))
        if den > EPS: scores.append(float(value @ consensus / den))
    return float(np.mean(scores)) if scores else 0.0


def observed_delta_q(q_adj: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    delta = np.zeros_like(q_adj, dtype=np.float32)
    delta[:, np.asarray(dims, dtype=np.int64)] = q_adj[:, np.asarray(dims, dtype=np.int64)]
    return delta


def matched_random_delta(q_adj: np.ndarray, target_delta: np.ndarray, basis: np.ndarray,
                         model: Any) -> np.ndarray:
    projected = (np.asarray(q_adj, dtype=np.float64) @ basis) @ basis.T
    target_h = inverse_q_delta(target_delta, model)
    random_h = inverse_q_delta(projected, model)
    target_norm = np.linalg.norm(target_h, axis=1)
    random_norm = np.linalg.norm(random_h, axis=1)
    scale = target_norm / np.maximum(random_norm, EPS)
    return (projected * scale[:, None]).astype(np.float32)


def subject_decision_metrics(clean: np.ndarray, erased: np.ndarray, y: np.ndarray,
                             subjects: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for subject in sorted(set(map(str, subjects)), key=int):
        idx = np.flatnonzero(np.asarray(subjects, dtype=str) == subject)
        delta = erased[idx] - clean[idx]
        p0, p1 = softmax(clean[idx]), softmax(erased[idx])
        rows.append({
            "subject": subject, "n_trials": int(len(idx)),
            "decision_logit_rms": float(np.sqrt(np.mean(centered_logit_sq(delta)))),
            "decision_margin_displacement": float(np.mean(np.abs(true_margin(erased[idx], y[idx]) - true_margin(clean[idx], y[idx])))),
            "decision_flip_rate": float(np.mean(erased[idx].argmax(1) != clean[idx].argmax(1))),
            "decision_total_variation": float(np.mean(0.5 * np.sum(np.abs(p1 - p0), axis=1))),
        })
    return rows


def subject_outcome_metrics(clean: np.ndarray, erased: np.ndarray, y: np.ndarray,
                            subjects: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for subject in sorted(set(map(str, subjects)), key=int):
        idx = np.flatnonzero(np.asarray(subjects, dtype=str) == subject)
        rows.append({
            "subject": subject, "n_trials": int(len(idx)),
            "outcome_ce_effect": float(np.mean(ce_per_row(erased[idx], y[idx]) - ce_per_row(clean[idx], y[idx]))),
            "outcome_ba_change": float(balanced_accuracy(y[idx], erased[idx].argmax(1)) - balanced_accuracy(y[idx], clean[idx].argmax(1))),
        })
    return rows


def run_dda_bc(device: torch.device) -> dict[str, Any]:
    require_lock()
    meta = P5.load_mi_manifest(); bases = ROUTER.selected_bases()
    cell_rows: list[dict[str, Any]] = []; subject_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            run = ROUTER.load_run(fold, seed, meta); cfg = bases[(fold, seed)]
            parts = ROUTER.router_subject_folds(run.split["train_subjects"], fold, seed)
            utility = pd.read_csv(utility_path(fold, seed)); assignment = json.loads(assignment_path(fold, seed).read_text(encoding="utf-8"))["mi"]
            assignment_map = {int(b): "neutral" for b in assignment.get("neutral", [])}
            for name in ("uncertain", "harmful", "protected"):
                for b in assignment.get(name, []): assignment_map[int(b)] = name
            for audit_fold in range(AUDIT_FOLDS):
                decision_subjects = parts[audit_fold]
                outcome_subjects = parts[(audit_fold + 1) % AUDIT_FOLDS]
                excluded = set(decision_subjects) | set(outcome_subjects)
                fit_subjects = [s for s in run.split["train_subjects"] if s not in excluded]
                if set(fit_subjects) & excluded or len(fit_subjects) + len(excluded) != len(run.split["train_subjects"]):
                    raise RuntimeError("Illegal three-way DDA cross-fit")
                fit_pos = positions(meta, fit_subjects); dpos = positions(meta, decision_subjects); opos = positions(meta, outcome_subjects)
                model = fit_v2_control(run, cfg, fit_pos, f"dda-bc-{audit_fold}", device)
                dclean, dq_adj, _ = forward(model, run.h[dpos], run.q[dpos], device)
                oclean, oq_adj, _ = forward(model, run.h[opos], run.q[opos], device)
                grad = jacobian_margin(model, run.h[dpos], run.q[dpos], device)
                dm = meta.iloc[dpos].reset_index(drop=True); om = meta.iloc[opos].reset_index(drop=True)
                dy = dm.label.to_numpy(dtype=np.int64); oy = om.label.to_numpy(dtype=np.int64)
                for block, dims_list in enumerate(run.art.blocks):
                    dims = np.asarray(dims_list, dtype=np.int64); rank = len(dims)
                    ddelta = observed_delta_q(dq_adj, dims); odelta = observed_delta_q(oq_adj, dims)
                    derased = linear_erase_logits(dclean, ddelta, model); oerased = linear_erase_logits(oclean, odelta, model)
                    drows = subject_decision_metrics(dclean, derased, dy, dm.subject.to_numpy(dtype=str))
                    orows = subject_outcome_metrics(oclean, oerased, oy, om.subject.to_numpy(dtype=str))
                    jac_trial = np.sum(grad[:, dims] ** 2, axis=1) / (2.0 * rank)
                    jac_subject = []
                    for subject in sorted(dm.subject.unique().tolist(), key=int):
                        idx = np.flatnonzero(dm.subject.to_numpy(dtype=str) == str(subject))
                        jac_subject.append(float(np.mean(jac_trial[idx])))
                    random_local = []; random_logit = []
                    for draw in range(RANDOM_DRAWS):
                        basis = random_basis(run.art.q_dim, rank, stable_seed(IMPLEMENTATION_ID, "dda-b-random", fold, seed, audit_fold, block, draw))
                        rdelta = matched_random_delta(dq_adj, ddelta, basis, model)
                        rerased = linear_erase_logits(dclean, rdelta, model)
                        rlocal = float(np.mean(np.sum((grad @ basis) ** 2, axis=1) / (2.0 * rank)))
                        rlogit = float(np.mean([r["decision_logit_rms"] for r in subject_decision_metrics(
                            dclean, rerased, dy, dm.subject.to_numpy(dtype=str)
                        )]))
                        random_local.append(rlocal); random_logit.append(rlogit)
                        random_rows.append({"fold": fold, "seed": seed, "run": f"{fold}_{seed}", "audit_fold": audit_fold,
                                            "block": block, "rank": rank, "draw": draw,
                                            "random_jacobian_energy": rlocal, "random_logit_rms": rlogit})
                    observed_local = float(np.mean(jac_subject)); observed_logit = float(np.mean([r["decision_logit_rms"] for r in drows]))
                    urow = utility[(utility.task == "mi") & (utility.block.astype(int) == block)]
                    if len(urow) != 1: raise RuntimeError(f"Missing utility fold={fold} seed={seed} block={block}")
                    u = urow.iloc[0]
                    cell = {
                        "fold": fold, "seed": seed, "run": f"{fold}_{seed}", "audit_fold": audit_fold,
                        "block": block, "rank": rank, "assignment": assignment_map.get(block, "unassigned"),
                        "n_fit_subjects": len(fit_subjects), "n_decision_subjects": len(decision_subjects),
                        "n_outcome_subjects": len(outcome_subjects),
                        "persistence_strength": float(np.mean(run.art.rho[dims])),
                        "geometry_strength": block_geometry_alignment(meta, run.q, fit_pos, dims),
                        "jacobian_energy": observed_local,
                        "jacobian_random_mean": float(np.mean(random_local)),
                        "jacobian_ratio": observed_local / max(float(np.mean(random_local)), EPS),
                        "decision_logit_rms": observed_logit,
                        "decision_random_logit_mean": float(np.mean(random_logit)),
                        "decision_logit_ratio": observed_logit / max(float(np.mean(random_logit)), EPS),
                        "decision_margin_displacement": float(np.mean([r["decision_margin_displacement"] for r in drows])),
                        "decision_flip_rate": float(np.mean([r["decision_flip_rate"] for r in drows])),
                        "decision_total_variation": float(np.mean([r["decision_total_variation"] for r in drows])),
                        "outcome_ce_effect": float(np.mean([r["outcome_ce_effect"] for r in orows])),
                        "outcome_ba_change": float(np.mean([r["outcome_ba_change"] for r in orows])),
                        "signed_u_abs": float(u.u_abs_mean), "signed_u_spec": float(u.u_spec_mean),
                        "outer_test_used": False,
                    }
                    cell_rows.append(cell)
                    for row in drows:
                        subject_rows.append({**{k: cell[k] for k in ("fold", "seed", "run", "audit_fold", "block", "rank", "assignment")},
                                             "role": "decision", **row})
                    for row in orows:
                        subject_rows.append({**{k: cell[k] for k in ("fold", "seed", "run", "audit_fold", "block", "rank", "assignment")},
                                             "role": "outcome", **row})
                print(f"[DDA-BC] fold={fold} seed={seed} audit_fold={audit_fold}/4", flush=True)
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
    result_dir = OUT / "results"; result_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame(cell_rows); subjects = pd.DataFrame(subject_rows); random_frame = pd.DataFrame(random_rows)
    cells.to_csv(result_dir / "DDA_BLOCK_CROSSFIT.csv", index=False)
    subjects.to_csv(result_dir / "DDA_BC_SUBJECT.csv", index=False)
    random_frame.to_csv(result_dir / "DDA_B_RANDOM_CONTROLS.csv", index=False)
    b_report = finalize_dda_b(cells)
    c_report = finalize_dda_c(cells)
    return {"dda_b": b_report, "dda_c": c_report}


def bootstrap_simple(values: np.ndarray, seed: int, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(int(seed))
    return np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)


def finalize_dda_b(cells: pd.DataFrame) -> dict[str, Any]:
    summary = cells.groupby(["fold", "seed", "run", "block", "rank", "assignment"], as_index=False).mean(numeric_only=True)
    protected = summary[summary.assignment == "protected"].copy()
    matched_values = []
    for _, row in protected.iterrows():
        controls = summary[(summary.run == row.run) & (summary["rank"] == row["rank"]) & (summary.assignment != "protected")]
        matched_values.append(float(controls.decision_logit_rms.mean()) if len(controls) else float("nan"))
    protected["matched_nonprotected_logit"] = matched_values
    protected["protected_gt_matched_nonprotected"] = protected.decision_logit_rms > protected.matched_nonprotected_logit
    protected["directionally_concordant"] = ((protected.signed_u_abs > 0) &
                                               (protected.outcome_ce_effect > 0) &
                                               (protected.outcome_ba_change < 0))
    run_protected = protected.groupby(["fold", "seed", "run"], as_index=False).agg(
        jacobian_ratio=("jacobian_ratio", "mean"),
        decision_logit_ratio=("decision_logit_ratio", "mean"),
        protected_gt_matched_fraction=("protected_gt_matched_nonprotected", "mean"),
        signed_u_abs=("signed_u_abs", "mean"),
        outcome_ce_effect=("outcome_ce_effect", "mean"),
        outcome_ba_change=("outcome_ba_change", "mean"),
        concordant_fraction=("directionally_concordant", "mean"),
        n_protected_blocks=("block", "count"),
    )
    local_boot = bootstrap_simple(run_protected.jacobian_ratio.to_numpy(), stable_seed("dda-b", "local"))
    finite_boot = bootstrap_simple(run_protected.decision_logit_ratio.to_numpy(), stable_seed("dda-b", "finite"))
    local_positive = int(np.sum(run_protected.jacobian_ratio > 1.0))
    finite_positive = int(np.sum(run_protected.decision_logit_ratio > 1.0))
    nonprotected_positive = int(np.sum(run_protected.protected_gt_matched_fraction > 0.5))
    concordant = int(np.sum((run_protected.signed_u_abs > 0) &
                            (run_protected.outcome_ce_effect > 0) &
                            (run_protected.outcome_ba_change < 0)))
    gate = PROTOCOL["dda_b_gate"]
    passed = bool(
        np.quantile(local_boot, .025) > gate["jacobian_ratio_bootstrap_lower_95_min"]
        and np.quantile(finite_boot, .025) > gate["finite_logit_ratio_bootstrap_lower_95_min"]
        and local_positive >= gate["minimum_positive_runs_of_6"]
        and finite_positive >= gate["minimum_positive_runs_of_6"]
        and nonprotected_positive >= gate["minimum_protected_gt_matched_nonprotected_runs_of_6"]
        and concordant >= gate["minimum_directionally_concordant_runs_of_6"]
    )
    protected.to_csv(OUT / "results" / "DDA_B_PROTECTED_RUNS.csv", index=False)
    run_protected.to_csv(OUT / "results" / "DDA_B_PROTECTED_RUN_AGGREGATE.csv", index=False)
    summary.to_csv(OUT / "results" / "DDA_B_BLOCK_SUMMARY.csv", index=False)
    report = {
        "status": "DDA_B_PASS" if passed else "DDA_B_FAIL", "passed": passed,
        "n_protected_run_blocks": len(protected), "n_run_level_units": len(run_protected),
        "local_ratio_mean": float(run_protected.jacobian_ratio.mean()),
        "local_ratio_bootstrap_ci95": [float(np.quantile(local_boot, .025)), float(np.quantile(local_boot, .975))],
        "finite_ratio_mean": float(run_protected.decision_logit_ratio.mean()),
        "finite_ratio_bootstrap_ci95": [float(np.quantile(finite_boot, .025)), float(np.quantile(finite_boot, .975))],
        "local_positive_runs": local_positive, "finite_positive_runs": finite_positive,
        "protected_gt_matched_nonprotected_runs": nonprotected_positive,
        "signed_utility_and_held_consequence_concordant_runs": concordant,
        "outer_test_used": False,
    }
    write_json(OUT / "results" / "DDA_B_RESULT.json", report)
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                  alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0); std = np.maximum(x_train.std(axis=0), 1e-8)
    xt = (x_train - mean) / std; xe = (x_test - mean) / std
    design = np.column_stack([np.ones(len(xt)), xt]); test = np.column_stack([np.ones(len(xe)), xe])
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y_train)
    return test @ beta, beta


def loro_predictions(cells: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, list[float]]:
    rows = []; coefficients = []
    for held_run in sorted(cells.run.unique().tolist()):
        train = cells.run != held_run; test = ~train
        pred, beta = ridge_predict(cells.loc[train, list(features)].to_numpy(dtype=np.float64),
                                  cells.loc[train, "outcome_ce_effect"].to_numpy(dtype=np.float64),
                                  cells.loc[test, list(features)].to_numpy(dtype=np.float64), alpha=1.0)
        part = cells.loc[test, ["fold", "seed", "run", "audit_fold", "block", "outcome_ce_effect"]].copy()
        part["prediction"] = pred; rows.append(part)
        if "decision_logit_rms" in features:
            coefficients.append(float(beta[1 + list(features).index("decision_logit_rms")]))
    return pd.concat(rows, ignore_index=True), coefficients


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def incremental_score(cells: pd.DataFrame) -> tuple[float, pd.DataFrame, pd.DataFrame, list[float]]:
    baseline_features = PROTOCOL["dda_c_gate"]["baseline"]
    full_features = [*baseline_features, PROTOCOL["dda_c_gate"]["incremental_feature"]]
    base, _ = loro_predictions(cells, baseline_features); full, coefficients = loro_predictions(cells, full_features)
    keys = ["fold", "seed", "run", "audit_fold", "block", "outcome_ce_effect"]
    pred = base.rename(columns={"prediction": "baseline_prediction"}).merge(
        full.rename(columns={"prediction": "full_prediction"}), on=keys, validate="one_to_one"
    )
    score = (rmse(pred.outcome_ce_effect, pred.baseline_prediction) - rmse(pred.outcome_ce_effect, pred.full_prediction)) / max(
        rmse(pred.outcome_ce_effect, pred.baseline_prediction), EPS
    )
    per_run = []
    for run, group in pred.groupby("run", sort=True):
        rb = rmse(group.outcome_ce_effect, group.baseline_prediction); rf = rmse(group.outcome_ce_effect, group.full_prediction)
        per_run.append({"run": run, "baseline_rmse": rb, "full_rmse": rf,
                        "relative_improvement": (rb - rf) / max(rb, EPS)})
    return float(score), pred, pd.DataFrame(per_run), coefficients


def loro_numpy(x: np.ndarray, y: np.ndarray, runs: np.ndarray) -> tuple[np.ndarray, list[float]]:
    prediction = np.empty(len(y), dtype=np.float64); coefficients: list[float] = []
    for held_run in sorted(set(runs.tolist())):
        test = runs == held_run; train = ~test
        pred, beta = ridge_predict(x[train], y[train], x[test], alpha=1.0)
        prediction[test] = pred
        coefficients.append(float(beta[-1]))
    return prediction, coefficients


def incremental_score_numpy(base_x: np.ndarray, decision: np.ndarray, y: np.ndarray,
                            runs: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    base_pred, _ = loro_numpy(base_x, y, runs)
    full_pred, coefficients = loro_numpy(np.column_stack([base_x, decision]), y, runs)
    rb = rmse(y, base_pred); rf = rmse(y, full_pred)
    return float((rb - rf) / max(rb, EPS)), base_pred, full_pred, coefficients


def finalize_dda_c(cells: pd.DataFrame) -> dict[str, Any]:
    baseline_features = PROTOCOL["dda_c_gate"]["baseline"]
    base_x = cells[baseline_features].to_numpy(dtype=np.float64)
    decision = cells.decision_logit_rms.to_numpy(dtype=np.float64)
    target = cells.outcome_ce_effect.to_numpy(dtype=np.float64)
    run_array = np.asarray(cells.run.astype(str).to_numpy(), dtype=object)
    observed, base_prediction, full_prediction, coefficients = incremental_score_numpy(
        base_x, decision, target, run_array,
    )
    pred = cells[["fold", "seed", "run", "audit_fold", "block", "outcome_ce_effect"]].copy()
    pred["baseline_prediction"] = base_prediction; pred["full_prediction"] = full_prediction
    per_run_rows = []
    for run in sorted(set(run_array.tolist())):
        idx = np.flatnonzero(run_array == run)
        rb = rmse(target[idx], base_prediction[idx]); rf = rmse(target[idx], full_prediction[idx])
        per_run_rows.append({"run": run, "baseline_rmse": rb, "full_rmse": rf,
                             "relative_improvement": (rb - rf) / max(rb, EPS)})
    per_run = pd.DataFrame(per_run_rows)
    rng = np.random.default_rng(stable_seed("dda-c-cluster-bootstrap")); runs = sorted(set(run_array.tolist()))
    improvements = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    by_run = {run: np.flatnonzero(run_array == run) for run in runs}
    for b in range(BOOTSTRAP_DRAWS):
        sample = np.concatenate([by_run[str(run)] for run in rng.choice(runs, size=len(runs), replace=True)])
        rb = rmse(target[sample], base_prediction[sample]); rf = rmse(target[sample], full_prediction[sample])
        improvements[b] = (rb - rf) / max(rb, EPS)
    null = np.empty(PERMUTATION_DRAWS, dtype=np.float64)
    audit_fold_array = cells.audit_fold.to_numpy(dtype=np.int64)
    permutation_groups = [np.flatnonzero((run_array == run) & (audit_fold_array == audit_fold))
                          for run in runs for audit_fold in range(AUDIT_FOLDS)]
    for draw in range(PERMUTATION_DRAWS):
        values = decision.copy()
        prng = np.random.default_rng(stable_seed("dda-c-permutation", draw))
        for idx in permutation_groups:
            values[idx] = prng.permutation(values[idx])
        null[draw] = incremental_score_numpy(base_x, values, target, run_array)[0]
    p_value = float((1 + np.sum(null >= observed)) / (PERMUTATION_DRAWS + 1))
    positive_runs = int(np.sum(per_run.relative_improvement > 0)); positive_coefficients = int(np.sum(np.asarray(coefficients) > 0))
    gate = PROTOCOL["dda_c_gate"]
    passed = bool(
        observed >= gate["loro_relative_rmse_improvement_min"]
        and positive_runs >= gate["minimum_improved_runs_of_6"]
        and np.quantile(improvements, .025) > gate["run_cluster_bootstrap_improvement_lower_95_min"]
        and p_value <= gate["permutation_p_max"]
        and positive_coefficients >= gate["minimum_positive_coefficients_of_6"]
    )
    pred.to_csv(OUT / "results" / "DDA_C_LORO_PREDICTIONS.csv", index=False)
    per_run.to_csv(OUT / "results" / "DDA_C_RUN_SENSITIVITY.csv", index=False)
    pd.DataFrame({"relative_rmse_improvement_null": null}).to_csv(OUT / "results" / "DDA_C_PERMUTATION_NULL.csv", index=False)
    report = {
        "status": "DDA_C_PASS" if passed else "DDA_C_FAIL", "passed": passed,
        "n_run_fold_block_cells": len(cells), "n_runs": int(cells.run.nunique()),
        "baseline_rmse": rmse(target, base_prediction),
        "full_rmse": rmse(target, full_prediction),
        "relative_rmse_improvement": observed,
        "run_cluster_bootstrap_ci95": [float(np.quantile(improvements, .025)), float(np.quantile(improvements, .975))],
        "improved_runs": positive_runs, "positive_incremental_coefficients": positive_coefficients,
        "incremental_coefficients": coefficients, "permutation_p": p_value,
        "permutation_draws": PERMUTATION_DRAWS,
        "interpretation_constraint": "Pooled in-sample correlation is not a success criterion.",
        "outer_test_used": False,
    }
    write_json(OUT / "results" / "DDA_C_RESULT.json", report)
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def make_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(OUT / "results" / "DDA_A_SUBJECT.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.2)); ax.scatter(a.q_movement_relative, a.logit_rms_over_train_margin, s=16, alpha=.65)
    ax.axhline(PROTOCOL["dda_a_gate"]["equivalence_logit_rms_over_train_margin_upper_onesided_95_max"], color="tab:red", ls="--")
    ax.set(xlabel="Representation movement / median TRAIN q norm", ylabel="Centered-logit RMS / TRAIN margin")
    fig.tight_layout(); fig.savefig(fig_dir / "dda_a_representation_vs_decision.png", dpi=180); plt.close(fig)
    b = pd.read_csv(OUT / "results" / "DDA_B_BLOCK_SUMMARY.csv")
    order = [x for x in ("protected", "neutral", "uncertain", "harmful", "unassigned") if x in set(b.assignment)]
    means = [b[b.assignment == x].decision_logit_rms.mean() for x in order]
    fig, ax = plt.subplots(figsize=(5.2, 4.2)); ax.bar(order, means, color=["#d95f02" if x == "protected" else "#7570b3" for x in order])
    ax.set(ylabel="Finite centered-logit RMS", xlabel="Frozen Signed-V3.1 assignment"); fig.tight_layout()
    fig.savefig(fig_dir / "dda_b_assignment_decision_dependence.png", dpi=180); plt.close(fig)
    cells = pd.read_csv(OUT / "results" / "DDA_BLOCK_CROSSFIT.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.2)); colors = np.where(cells.assignment == "protected", "#d95f02", "#666666")
    ax.scatter(cells.decision_logit_rms, cells.outcome_ce_effect, c=colors, s=18, alpha=.7)
    ax.set(xlabel="Decision dependence (held subjects)", ylabel="CE erasure consequence (disjoint held subjects)")
    fig.tight_layout(); fig.savefig(fig_dir / "dda_c_dependence_vs_consequence.png", dpi=180); plt.close(fig)
    p = pd.read_csv(OUT / "results" / "DDA_C_LORO_PREDICTIONS.csv")
    fig, ax = plt.subplots(figsize=(5.2, 4.2)); ax.scatter(p.outcome_ce_effect, p.full_prediction, s=18, alpha=.65)
    lo = min(p.outcome_ce_effect.min(), p.full_prediction.min()); hi = max(p.outcome_ce_effect.max(), p.full_prediction.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1); ax.set(xlabel="Observed held consequence", ylabel="LORO predicted consequence")
    fig.tight_layout(); fig.savefig(fig_dir / "dda_c_predicted_vs_observed.png", dpi=180); plt.close(fig)


def finalize() -> dict[str, Any]:
    require_lock()
    a = json.loads((OUT / "results" / "DDA_A_RESULT.json").read_text(encoding="utf-8"))
    b = json.loads((OUT / "results" / "DDA_B_RESULT.json").read_text(encoding="utf-8"))
    c = json.loads((OUT / "results" / "DDA_C_RESULT.json").read_text(encoding="utf-8"))
    all_pass = bool(a["passed"] and b["passed"] and c["passed"])
    if all_pass:
        terminal = "DDA_PASS_EXTERNAL_AUDIT_REQUIRED"; stop = None
    elif not any((a["passed"], b["passed"], c["passed"])):
        terminal = "DDA_FAIL_NO_DECISION_MECHANISM"
        stop = ("STOP_AGDI_DECISION_DEPENDENCE_NOT_INCREMENTAL" if not c["passed"]
                else "STOP_AGDI_DDA_CHAIN_INCOMPLETE")
    else:
        terminal = "DDA_PARTIAL_MECHANISM_ONLY"
        stop = ("STOP_AGDI_DECISION_DEPENDENCE_NOT_INCREMENTAL" if not c["passed"]
                else "STOP_AGDI_DDA_CHAIN_INCOMPLETE")
    report = {
        "terminal_state": terminal, "dda_a": a["status"], "dda_b": b["status"], "dda_c": c["status"],
        "consistent_mechanism_chain": all_pass,
        "external_actionability_audit_authorized": all_pass,
        "agdi_training_authorized": False,
        "stop_state": stop,
        "outer_test_state": "OUTER_TEST_LOCKED",
        "development_validation_used": False, "outer_test_used": False,
        "outer_test_container_scope_caveat": "The inherited P5/Router loader materialized the all-54-subject manifest and aligned h0 container; only frozen outer-TRAIN positions were ever indexed for fitting, measurement, statistics, or gates.",
        "reason": ("DDA-A/B/C all passed; an external actionability audit is required before AGDI."
                   if all_pass else "The frozen DDA chain is incomplete; protocol forbids external constructive search and AGDI."),
    }
    lock = json.loads((OUT / "protocol" / "DDA_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    repair_log = {
        "scientific_gates_changed": False,
        "result_selection_changed": False,
        "locked_source_sha256": lock.get("source_code_sha256"),
        "final_source_sha256": sha256(Path(__file__)),
        "events": [
            {"phase": "DDA-A post-write reporting", "repair": "renamed a local clean-logits variable that shadowed the JSON clean function", "numerical_results_affected": False},
            {"phase": "DDA-C permutation", "repair": "replaced repeated pandas/Arrow operations with algebraically equivalent NumPy arrays after the server native warnings state failed", "crossfit_cells_recomputed": False, "numerical_results_affected": False},
            {"phase": "terminal labeling", "repair": "use chain-incomplete stop code when DDA-C passes but another mandatory DDA gate fails", "gate_outcomes_affected": False},
        ],
        "dda_a_result_sha256": sha256(OUT / "results" / "DDA_A_RESULT.json"),
        "dda_bc_cells_sha256": sha256(OUT / "results" / "DDA_BLOCK_CROSSFIT.csv"),
    }
    write_json(OUT / "protocol" / "DDA_POSTRUN_REPAIR_LOG.json", repair_log)
    provenance_correction = {
        "status": "PROVENANCE_SCOPE_WORDING_CORRECTED",
        "original_file_preserved": "protocol/PROVENANCE_AUDIT.json",
        "inaccurate_original_wording": ["outer_test_loaded=false", "test_subject_information=false for the full-row manifest/h0 container"],
        "container_scope": {
            "all_54_subject_manifest_loaded_to_memory": True,
            "all_54_subject_aligned_h0_loaded_by_inherited_router_api": True,
            "event_label_column_present_in_manifest_container": True,
        },
        "computational_use_scope": {
            "development_validation_positions_indexed": False,
            "outer_test_positions_indexed": False,
            "development_validation_or_outer_labels_used": False,
            "development_validation_or_outer_representations_used": False,
            "all_model_fit_decision_measurement_outcome_measurement_and_gates_restricted_to_frozen_outer_train_subjects": True,
        },
        "scientific_effect": "No inferential leakage was found, but the inherited all-row container API does not satisfy a literal never-materialize-metadata interpretation of the lock.",
        "result_or_gate_changed": False,
    }
    write_json(OUT / "protocol" / "PROVENANCE_SCOPE_CORRECTION.json", provenance_correction)
    write_json(OUT / "DDA_FINAL_REPORT.json", report)
    make_figures()
    markdown = f"""# PERSIST-EEG Decision Dependence Audit V1

**Terminal state: `{terminal}`**

## Executive decision

- DDA-A: `{a['status']}`
- DDA-B: `{b['status']}`
- DDA-C: `{c['status']}`
- AGDI training authorized: `false`
- Outer test: `OUTER_TEST_LOCKED`
{f'- Stop state: `{stop}`' if stop else '- Next state: `EXTERNAL_ACTIONABILITY_AUDIT`'}

The confirmatory audit used OpenBMI MI outer-TRAIN subjects only.  Signed-V3.1
assignments and the P5.1 V2 matched classifier choice were frozen before this
audit and were not redefined.

## Provenance

The machine-readable provenance map is `protocol/PROVENANCE_AUDIT.json`; the
pre-result gate freeze is `protocol/DDA_PROTOCOL_LOCK.json`.  No backbone was
retrained.  The inherited loader materialized the all-subject manifest and h0
container, but no development-validation or outer-test position, label, or
representation was indexed for fitting, measurement, statistics, or gates.
`protocol/PROVENANCE_SCOPE_CORRECTION.json` preserves this distinction and
corrects the original audit's overly broad `loaded=false` wording.

## DDA-A — CF behavioral-null explanation

Status: `{a['status']}`.  Relative q movement mean:
{a['bootstrap']['q_movement_relative']['mean']:.6f}.  Centered-logit RMS divided
by the TRAIN margin mean: {a['bootstrap']['logit_rms_over_train_margin']['mean']:.6f}.
Flip rate mean: {a['bootstrap']['flip_rate']['mean']:.6f}.  Formal equivalence
uses frozen one-sided bounds; a nonsignificant difference is never treated as
evidence of null.

## DDA-B — Protected decision activity

Status: `{b['status']}`.  Protected/random Jacobian ratio mean:
{b['local_ratio_mean']:.6f}; finite-logit ratio mean:
{b['finite_ratio_mean']:.6f}.  Protected exceeded matched non-PROTECTED blocks
in {b['protected_gt_matched_nonprotected_runs']}/6 runs, with signed-utility and
held-consequence concordance in {b['signed_utility_and_held_consequence_concordant_runs']}/6.

## DDA-C — incremental held-out explanation

Status: `{c['status']}`.  Baseline LORO RMSE: {c['baseline_rmse']:.8f}; full
RMSE: {c['full_rmse']:.8f}; relative improvement:
{c['relative_rmse_improvement']:.4%}; improved runs: {c['improved_runs']}/6;
permutation p={c['permutation_p']:.6f}.  Decision subjects and intervention-
outcome subjects are disjoint within each audit fold.  Trial-level
pseudo-replication is not used.

## Statistical uncertainty and limitations

Inference aggregates trials to subjects before gate evaluation.  DDA-C uses
run/block cells with run-level held prediction and run-cluster bootstrap.  Six
runs limit precision.  Canonical discovery and Signed assignment used the
full outer-TRAIN split in the earlier frozen phase; DDA cross-fitting separates
classifier fitting, decision measurement, and consequence measurement, but it
does not claim an independently rediscovered basis.  The confirmatory scope is
MI because PERSIST-CF and the P5.1 V2 matched classifier are MI-specific.  The
all-row container behavior means the run satisfies a no-use lock, not a literal
never-materialize-metadata lock; this is a provenance limitation, not evidence
that a held-out row influenced any computation.

## Exact decision

`{terminal}`

`{stop or 'EXTERNAL_ACTIONABILITY_AUDIT_REQUIRED_BEFORE_AGDI'}`
"""
    (OUT / "scientific_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "dda-a", "dda-bc", "dda-c-stats", "finalize", "all"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    seed_all(stable_seed(IMPLEMENTATION_ID, "global"))
    if args.phase in ("audit", "all"): audit_provenance()
    if args.phase in ("dda-a", "all"): run_dda_a(device)
    if args.phase in ("dda-bc", "all"): run_dda_bc(device)
    if args.phase == "dda-c-stats":
        require_lock(); finalize_dda_c(pd.read_csv(OUT / "results" / "DDA_BLOCK_CROSSFIT.csv"))
    if args.phase in ("finalize", "all"): finalize()


if __name__ == "__main__":
    main()
