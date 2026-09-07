"""PERSIST-CF: geometry-preserving subject-offset counterfactual training.

The command line is split at the method lock.  ``audit`` and ``cf0`` only
materialise outer-TRAIN rows.  ``evaluate`` refuses to run until a complete
TRAIN-only lock exists.  The OpenBMI outer-test split is never loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
EXP_ROOT = REPO_ROOT / "experiments" / "persist_eeg_cf"
OUT = EXP_ROOT / "outputs"
P5_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_icg"
P56_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_1_p6"
P56_OUT = P56_ROOT / "outputs"
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
INNER_FOLDS = 5
TASK = "mi"
IMPLEMENTATION_ID = "persist_cf_v1_train_local_blockwise"
REFERENCE_COMMIT = "b0a8c184619b697fd1b3342c5acdfa033a5df85d"
EPS = 1e-8
INVARIANT_RTOL = 1e-6


def _import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


P5 = _import_file("persist_cf_frozen_p5", P5_ROOT / "code" / "p5_icg.py")
P56 = _import_file("persist_cf_frozen_p56", P56_ROOT / "code" / "p5_1_p6.py")


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def normalise(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return np.zeros_like(value, dtype=np.float32)
    return (value / norm).astype(np.float32)


def positions(meta: pd.DataFrame, subjects: Sequence[str]) -> np.ndarray:
    wanted = {str(x) for x in subjects}
    return np.flatnonzero(meta.subject.astype(str).isin(wanted).to_numpy())


def outer_train_subjects(fold: int) -> list[str]:
    payload = json.loads(P5.SPLIT.read_text(encoding="utf-8"))
    item = next(x for x in payload["openbmi"]["folds"] if int(x["fold"]) == int(fold))
    # No validation or outer subject list is retained by this function.
    return [str(x) for x in item["train_subjects"]]


def labels(meta: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
    return np.asarray(meta.iloc[pos].label.to_numpy(dtype=np.int64), dtype=np.int64)


def subject_folds(subjects: Sequence[str], fold: int, seed: int) -> list[list[str]]:
    ordered = sorted({str(x) for x in subjects}, key=int)
    rng = np.random.default_rng(stable_seed("persist-cf-inner-subject-folds", fold, seed, ordered))
    perm = [ordered[i] for i in rng.permutation(len(ordered))]
    result = [perm[i::INNER_FOLDS] for i in range(INNER_FOLDS)]
    flat = [s for part in result for s in part]
    if len(flat) != len(set(flat)) or set(flat) != set(ordered):
        raise RuntimeError("Inner subject folds are not a disjoint complete partition")
    return result


@dataclass(frozen=True)
class SelectedBase:
    fold: int
    seed: int
    lambda_drift: float
    learning_rate: float
    bottleneck: int
    epochs: int
    candidate: str


@dataclass
class AuditRun:
    fold: int
    seed: int
    meta: pd.DataFrame
    h: np.ndarray
    q: np.ndarray
    art: Any
    train_subjects: list[str]


@dataclass
class TrainGeometry:
    subjects: list[str]
    centers: dict[str, np.ndarray]
    directions: dict[int, np.ndarray]
    block_dims: dict[int, np.ndarray]
    cell_counts: dict[str, int]

    @property
    def protected_dims(self) -> np.ndarray:
        values = [self.block_dims[b] for b in sorted(self.block_dims)]
        return np.concatenate(values) if values else np.zeros(0, dtype=np.int64)


@dataclass(frozen=True)
class CFConfig:
    name: str
    alpha: float
    p_cf: float
    lambda_cf: float
    lambda_cons: float
    scale_policy: str = "constant"
    clip_quantile: float | None = None
    donor_policy: str = "uniform"


CF0_CONFIGS = (
    CFConfig("cf0_a025_p025_l05_c0", 0.25, 0.25, 0.5, 0.0),
    CFConfig("cf0_a025_p050_l10_c01", 0.25, 0.50, 1.0, 0.1),
    CFConfig("cf0_a050_p025_l05_c01", 0.50, 0.25, 0.5, 0.1),
    CFConfig("cf0_a050_p050_l10_c01", 0.50, 0.50, 1.0, 0.1),
    CFConfig("cf0_a100_p025_l05_c01", 1.00, 0.25, 0.5, 0.1),
    CFConfig("cf0_a100_p050_l10_c01", 1.00, 0.50, 1.0, 0.1),
)


def selected_bases() -> dict[tuple[int, int], SelectedBase]:
    selection_path = P56_OUT / "protocol" / "P6_BASE_VERSION_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("base_version") != "V2" or selection.get("outer_test_used") is not False:
        raise RuntimeError("Frozen P6 selection is not the TRAIN-only V2 base")
    frame = pd.read_csv(P56_OUT / "P5_1_SELECTED_CONFIGS.csv")
    frame = frame[frame.version.astype(str) == "V2"].copy()
    if len(frame) != 6:
        raise RuntimeError(f"Expected six V2 selected configs, found {len(frame)}")
    result: dict[tuple[int, int], SelectedBase] = {}
    for _, row in frame.iterrows():
        item = SelectedBase(
            fold=int(row.fold), seed=int(row.seed), lambda_drift=float(row.lambda_drift),
            learning_rate=float(row.learning_rate), bottleneck=int(row.bottleneck),
            epochs=int(P56.median_epoch(row.median_pair_epoch)), candidate=str(row.candidate),
        )
        result[(item.fold, item.seed)] = item
    return result


def load_run(fold: int, seed: int, meta: pd.DataFrame) -> AuditRun:
    train_subjects = outer_train_subjects(fold)
    h_path = P5.OUT / "cache" / f"fold-{fold}" / f"seed-{seed}" / "h0.npy"
    h = np.asarray(np.load(h_path, mmap_mode="r"), dtype=np.float32)
    if h.shape != (len(meta), 128) or not np.isfinite(h).all():
        raise RuntimeError(f"Invalid frozen h0: {h_path} {h.shape}")
    art = P5.load_artifacts(fold, seed)
    q = P5.q_from_h(h, art)
    return AuditRun(fold, seed, meta, h, q, art, train_subjects)


def fit_geometry(run: AuditRun, train_pos: np.ndarray) -> TrainGeometry:
    meta = run.meta.iloc[train_pos].reset_index(drop=True)
    q = np.asarray(run.q[train_pos], dtype=np.float64)
    subjects = sorted(meta.subject.astype(str).unique().tolist(), key=int)
    sessions = sorted(meta.session.astype(str).unique().tolist())
    block_dims = {int(b): np.asarray(run.art.blocks[b], dtype=np.int64) for b in run.art.protected_blocks}
    centers: dict[str, np.ndarray] = {}
    contrasts: dict[int, list[np.ndarray]] = {b: [] for b in block_dims}
    cell_counts: dict[str, int] = {}
    subject_values = meta.subject.astype(str).to_numpy()
    session_values = meta.session.astype(str).to_numpy()
    y = meta.label.to_numpy(dtype=np.int64)
    for subject in subjects:
        cells: list[np.ndarray] = []
        for session in sessions:
            class_means: list[np.ndarray] = []
            for label in (0, 1):
                loc = np.flatnonzero((subject_values == subject) & (session_values == session) & (y == label))
                if len(loc) == 0:
                    raise RuntimeError(f"Missing TRAIN cell subject={subject} session={session} class={label}")
                class_means.append(q[loc].mean(axis=0))
                cells.append(class_means[-1])
                cell_counts[f"{subject}|{session}|{label}"] = int(len(loc))
            contrast = class_means[1] - class_means[0]
            for block, dims in block_dims.items():
                contrasts[block].append(contrast[dims])
        center = np.mean(np.stack(cells), axis=0)
        protected_center = np.zeros(run.art.q_dim, dtype=np.float32)
        for dims in block_dims.values():
            protected_center[dims] = center[dims].astype(np.float32)
        centers[subject] = protected_center
    directions: dict[int, np.ndarray] = {}
    for block in sorted(block_dims):
        # Binary Shared-Geometry V1.2 consensus: mean of equally weighted
        # subject/session class contrasts, then unit normalization.
        directions[block] = normalise(np.mean(np.stack(contrasts[block]), axis=0))
        if not np.any(directions[block]):
            raise RuntimeError(f"Degenerate TRAIN shared geometry in block {block}")
    return TrainGeometry(subjects, centers, directions, block_dims, cell_counts)


def project_shift(delta: np.ndarray, geometry: TrainGeometry, *, keep_geometry: bool) -> np.ndarray:
    source = np.asarray(delta, dtype=np.float32)
    one = source.ndim == 1
    value = source.reshape(1, -1).copy() if one else source.copy()
    output = np.zeros_like(value)
    for block in sorted(geometry.block_dims):
        dims = geometry.block_dims[block]
        part = value[:, dims]
        if keep_geometry:
            g = geometry.directions[block].reshape(1, -1)
            part = part - (part @ g.T) @ g
        output[:, dims] = part
    return output[0] if one else output


def geometry_projection(delta: np.ndarray, geometry: TrainGeometry) -> np.ndarray:
    value = np.asarray(delta, dtype=np.float32)
    one = value.ndim == 1
    value = value.reshape(1, -1) if one else value
    parts = []
    for block in sorted(geometry.block_dims):
        dims = geometry.block_dims[block]
        g = geometry.directions[block].reshape(-1, 1)
        parts.append(value[:, dims] @ g)
    out = np.concatenate(parts, axis=1)
    return out[0] if one else out


def inverse_delta(delta_q: np.ndarray, art: Any) -> np.ndarray:
    value = np.asarray(delta_q, dtype=np.float64)
    return ((value @ np.asarray(art.directions, dtype=np.float64).T) @
            np.asarray(art.dewhitener, dtype=np.float64)).astype(np.float32)


def donor_shifts(meta: pd.DataFrame, pos: np.ndarray, geometry: TrainGeometry,
                 seed: int, *, keep_geometry: bool = True) -> tuple[np.ndarray, list[str]]:
    subjects = meta.iloc[pos].subject.astype(str).to_numpy()
    rng = np.random.default_rng(int(seed))
    shifts = np.zeros((len(pos), next(iter(geometry.centers.values())).shape[0]), dtype=np.float32)
    donors: list[str] = []
    for i, source in enumerate(subjects):
        candidates = [x for x in geometry.subjects if x != source]
        donor = str(candidates[int(rng.integers(len(candidates)))])
        donors.append(donor)
        raw = geometry.centers[donor] - geometry.centers[str(source)]
        shifts[i] = project_shift(raw, geometry, keep_geometry=keep_geometry)
    return shifts, donors


def phase_a(meta: pd.DataFrame, bases: Mapping[tuple[int, int], SelectedBase]) -> dict[str, Any]:
    required = [
        P5_ROOT / "code" / "p5_icg.py",
        P56_ROOT / "code" / "p5_1_p6.py",
        P56_OUT / "P5_1_SELECTED_CONFIGS.csv",
        P56_OUT / "protocol" / "P6_BASE_VERSION_SELECTION.json",
        P5.MANIFEST,
        P5.SPLIT,
    ]
    for fold in FOLDS:
        for seed in SEEDS:
            required.extend([
                P5.OUT / "cache" / f"fold-{fold}" / f"seed-{seed}" / "h0.npy",
                P56_OUT / "V2" / f"fold-{fold}" / f"seed-{seed}" / "best_control.pt",
                P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "spectrum" / "PERSISTENCE_SPECTRUM.npz",
                P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_ASSIGNMENTS_V3_1.json",
            ])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen prerequisites: {missing}")
    payload = {
        "status": "PHASE_A_VERIFIED",
        "implementation_id": IMPLEMENTATION_ID,
        "reference_commit": REFERENCE_COMMIT,
        "primary": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGNet"},
        "manifest": {"rows": len(meta), "subjects": int(meta.subject.nunique()), "sha256": sha256(P5.MANIFEST)},
        "canonical_signed_v3_1": True,
        "matched_base": {"version": "P5.1 V2 control", "runs": [asdict(bases[(f, s)]) for f in FOLDS for s in SEEDS]},
        "development_validation_rows_or_labels_materialised": False,
        "development_validation_used_for_design_or_selection": False,
        "outer_test_subject_list_retained": False,
        "outer_test_samples_or_labels_loaded": False,
        "outer_test_used": False,
        "required_artifacts_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256(path)
            for path in required if path.is_file() and path.stat().st_size < 100 * 1024 * 1024
        },
    }
    write_json(OUT / "protocol" / "PHASE_A_AUDIT.json", payload)
    return payload


def unit_tests(runs: Mapping[tuple[int, int], AuditRun]) -> dict[str, Any]:
    rows = []
    for (fold, seed), run in runs.items():
        rng = np.random.default_rng(stable_seed("persist-cf-unit", fold, seed))
        dq = np.zeros((16, run.art.q_dim), dtype=np.float32)
        protected = run.art.protected_dims
        dq[:, protected] = rng.normal(0, 0.1, size=(len(dq), len(protected))).astype(np.float32)
        sample = np.asarray(run.h[:16], dtype=np.float32)
        q0 = P5.q_from_h(sample, run.art)
        dh = inverse_delta(dq, run.art)
        q1 = P5.q_from_h(sample + dh, run.art)
        recovered = q1 - q0
        residual = dh - inverse_delta(recovered, run.art)
        zero_error = float(np.max(np.abs(sample + inverse_delta(np.zeros_like(dq), run.art) - sample)))
        coordinate_error = float(np.max(np.abs(recovered - dq)))
        residual_error = float(np.max(np.abs(residual)))
        assignment = json.loads((P5.V31_ROOT / f"fold-{fold}" / f"seed-{seed}" / "SIGNED_ASSIGNMENTS_V3_1.json").read_text(encoding="utf-8"))
        indexing_exact = [int(x) for x in assignment[TASK]["protected"]] == [int(x) for x in run.art.protected_blocks]
        deterministic_error = float(np.max(np.abs(inverse_delta(dq, run.art) - inverse_delta(dq, run.art))))
        passed = bool(zero_error <= 1e-12 and coordinate_error <= 2e-5 and residual_error <= 2e-5 and indexing_exact and deterministic_error == 0.0)
        rows.append({
            "fold": fold, "seed": seed, "zero_delta_error": zero_error,
            "canonical_coordinate_error_max": coordinate_error,
            "residual_change_max": residual_error,
            "protected_indexing_exact": indexing_exact,
            "deterministic_error_max": deterministic_error,
            "passed": passed,
        })
    payload = {"status": "PHASE_B_UNIT_TESTS_PASS" if all(x["passed"] for x in rows) else "PHASE_B_UNIT_TESTS_FAIL", "runs": rows}
    write_json(OUT / "protocol" / "TRANSFORM_UNIT_TESTS.json", payload)
    if not all(x["passed"] for x in rows):
        raise RuntimeError(f"Canonical reconstruction unit test failed: {payload}")
    return payload


def headroom_row(run: AuditRun, geometry: TrainGeometry) -> tuple[dict[str, Any], np.ndarray]:
    raw, perp = [], []
    for source in geometry.subjects:
        for donor in geometry.subjects:
            if donor == source:
                continue
            value = geometry.centers[donor] - geometry.centers[source]
            raw.append(value)
            perp.append(project_shift(value, geometry, keep_geometry=True))
    raw_arr = np.stack(raw)
    perp_arr = np.stack(perp)
    proj = raw_arr - perp_arr
    total = np.sum(raw_arr * raw_arr, axis=1)
    g_energy = np.sum(proj * proj, axis=1)
    perp_energy = np.sum(perp_arr * perp_arr, axis=1)
    norms = np.linalg.norm(perp_arr, axis=1)
    protected_dim = int(sum(len(x) for x in geometry.block_dims.values()))
    geometry_rank = int(sum(int(np.linalg.norm(geometry.directions[b]) > 0) for b in geometry.block_dims))
    row = {
        "fold": run.fold, "seed": run.seed,
        "protected_blocks": json.dumps(sorted(geometry.block_dims)),
        "protected_dimension": protected_dim,
        "task_geometry_rank": geometry_rank,
        "orthogonal_augmentable_dimension": protected_dim - geometry_rank,
        "subject_offset_energy_total": float(np.mean(total)),
        "subject_offset_energy_G": float(np.mean(g_energy)),
        "subject_offset_energy_Gperp": float(np.mean(perp_energy)),
        "rho_offset": float(np.mean(perp_energy) / max(float(np.mean(total)), EPS)),
        "delta_perp_norm_min": float(np.min(norms)),
        "delta_perp_norm_q25": float(np.quantile(norms, 0.25)),
        "delta_perp_norm_median": float(np.median(norms)),
        "delta_perp_norm_q75": float(np.quantile(norms, 0.75)),
        "delta_perp_norm_q90": float(np.quantile(norms, 0.90)),
        "delta_perp_norm_max": float(np.max(norms)),
        "n_ordered_donor_pairs": int(len(raw_arr)),
    }
    row["headroom_pass"] = bool(row["orthogonal_augmentable_dimension"] > 0 and row["rho_offset"] >= 0.05 and row["delta_perp_norm_median"] > 1e-6)
    return row, perp_arr


def validity_row(run: AuditRun, train_pos: np.ndarray, geometry: TrainGeometry) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed("persist-cf-validity-subsample", run.fold, run.seed))
    selected = np.asarray(train_pos, dtype=np.int64)
    if len(selected) > 4096:
        selected = np.sort(rng.choice(selected, size=4096, replace=False))
    projected, donors = donor_shifts(run.meta, selected, geometry, stable_seed("persist-cf-validity-donors", run.fold, run.seed), keep_geometry=True)
    raw, _ = donor_shifts(run.meta, selected, geometry, stable_seed("persist-cf-validity-donors", run.fold, run.seed), keep_geometry=False)
    q = np.asarray(run.q[selected], dtype=np.float32)
    q_cf = q + projected
    source = run.meta.iloc[selected].subject.astype(str).to_numpy()
    source_centers = np.stack([geometry.centers[str(x)] for x in source])
    donor_centers = np.stack([geometry.centers[str(x)] for x in donors])
    q_perp = project_shift(q, geometry, keep_geometry=True)
    cf_perp = project_shift(q_cf, geometry, keep_geometry=True)
    source_perp = project_shift(source_centers, geometry, keep_geometry=True)
    donor_perp = project_shift(donor_centers, geometry, keep_geometry=True)
    donor_before = np.linalg.norm(q_perp - donor_perp, axis=1)
    donor_after = np.linalg.norm(cf_perp - donor_perp, axis=1)
    source_before = np.linalg.norm(q_perp - source_perp, axis=1)
    source_after = np.linalg.norm(cf_perp - source_perp, axis=1)
    projection_error = np.linalg.norm(geometry_projection(projected, geometry), axis=1)
    scale = np.maximum(np.linalg.norm(projected, axis=1), 1.0)
    relative_projection_error = projection_error / scale
    full_projection_change = np.linalg.norm(geometry_projection(raw, geometry), axis=1)
    protected_projection_change = np.linalg.norm(geometry_projection(q_cf, geometry) - geometry_projection(q, geometry), axis=1)
    # The shared-geometry task margin is a function only of Pi_G q and is
    # therefore exactly preserved by a valid projected counterfactual.
    margin_error = np.abs(protected_projection_change)
    row = {
        "fold": run.fold, "seed": run.seed, "n_samples": int(len(selected)),
        "geometry_projection_change_mean": float(np.mean(projection_error)),
        "geometry_projection_change_max": float(np.max(projection_error)),
        "geometry_projection_relative_max": float(np.max(relative_projection_error)),
        "donor_Gperp_distance_before_mean": float(np.mean(donor_before)),
        "donor_Gperp_distance_after_mean": float(np.mean(donor_after)),
        "donor_Gperp_distance_decrease_mean": float(np.mean(donor_before - donor_after)),
        "fraction_moved_toward_donor": float(np.mean(donor_after < donor_before)),
        "source_Gperp_distance_increase_mean": float(np.mean(source_after - source_before)),
        "task_geometry_margin_absolute_error_mean": float(np.mean(margin_error)),
        "task_geometry_margin_absolute_error_max": float(np.max(margin_error)),
        "full_unprojected_geometry_change_mean": float(np.mean(full_projection_change)),
        "projected_geometry_change_mean": float(np.mean(protected_projection_change)),
        "full_vs_projected_geometry_change_ratio": float(np.mean(full_projection_change) / max(float(np.mean(protected_projection_change)), 1e-12)),
    }
    row["validity_pass"] = bool(
        row["geometry_projection_relative_max"] <= INVARIANT_RTOL and
        row["donor_Gperp_distance_decrease_mean"] > 0 and
        row["fraction_moved_toward_donor"] > 0.5 and
        row["source_Gperp_distance_increase_mean"] > 0 and
        row["task_geometry_margin_absolute_error_max"] <= 2e-6 and
        row["full_unprojected_geometry_change_mean"] > row["projected_geometry_change_mean"] + 1e-6
    )
    return row


def audit(device: torch.device) -> dict[str, Any]:
    del device  # h0 and q audits are intentionally CPU/numpy only.
    meta = P5.load_mi_manifest()
    bases = selected_bases()
    phase_a_payload = phase_a(meta, bases)
    runs = {(f, s): load_run(f, s, meta) for f in FOLDS for s in SEEDS}
    tests = unit_tests(runs)
    fold_freeze = []
    energy_rows, validity_rows = [], []
    for (fold, seed), run in runs.items():
        folds = subject_folds(run.train_subjects, fold, seed)
        fold_freeze.append({"fold": fold, "seed": seed, "inner_folds": folds})
        train_pos = positions(meta, run.train_subjects)
        geometry = fit_geometry(run, train_pos)
        energy, _ = headroom_row(run, geometry)
        validity = validity_row(run, train_pos, geometry)
        energy_rows.append(energy)
        validity_rows.append(validity)
        print(f"[AUDIT] fold={fold} seed={seed} rho={energy['rho_offset']:.4f} "
              f"median_norm={energy['delta_perp_norm_median']:.4f} validity={validity['validity_pass']}", flush=True)
    write_json(OUT / "protocol" / "PERSIST_CF_SUBJECT_FOLDS.json", {
        "seed_rule": "SHA256(persist-cf-inner-subject-folds|fold|seed|ordered_subjects)",
        "n_folds": INNER_FOLDS, "runs": fold_freeze,
        "development_validation_used": False, "outer_test_used": False,
    })
    protocol = {
        "implementation_id": IMPLEMENTATION_ID,
        "coordinate_transform": "Signed-V3.1 q=(h-mean)@whitener@directions",
        "inverse_linear": "delta_h=(delta_q@directions.T)@dewhitener",
        "representation_location": "frozen historical h0 before P5.1 V2 adapter/head",
        "subject_center": "equal mean over subject x session x class cell means",
        "geometry": "blockwise normalized TRAIN-subject/session binary class-contrast consensus",
        "projector": "one rank-1 projector per frozen Protected block",
        "inner_cv": "deterministic five-fold subject-disjoint",
        "primary_comparator": "matched P5.1 V2 continued-training control",
        "development_validation_used_before_lock": False,
        "outer_test_used": False,
    }
    write_json(OUT / "protocol" / "PERSIST_CF_PROTOCOL.json", protocol)
    version_policy = {
        "CF0_max_serious_configs_per_run": 12,
        "max_refinement_families": 2,
        "authorized_refinements": ["CF1-SCALE", "CF1-HARD", "CF1-BLOCK-JOINT", "CF1-LOSS"],
        "forbidden": ["GRL", "global suppression", "PB/SI/old CT", "ICG continuation", "Router", "TTA", "target centering"],
    }
    write_json(OUT / "protocol" / "PERSIST_CF_VERSION_POLICY.json", version_policy)
    (OUT / "headroom").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(energy_rows).to_csv(OUT / "headroom" / "OFFSET_ENERGY_AUDIT.csv", index=False)
    pd.DataFrame(validity_rows).to_csv(OUT / "headroom" / "COUNTERFACTUAL_VALIDITY.csv", index=False)
    headroom_pass = all(x["headroom_pass"] for x in energy_rows)
    validity_pass = all(x["validity_pass"] for x in validity_rows)
    if not headroom_pass:
        status = "PERSIST_CF_NO_AUGMENTABLE_OFFSET_HEADROOM"
    elif not validity_pass:
        status = "PERSIST_CF_COUNTERFACTUAL_INVALID"
    else:
        status = "PERSIST_CF_AUDIT_PASS_READY_FOR_CF0"
    report = {
        "status": status,
        "phase_a": phase_a_payload["status"],
        "phase_b": tests["status"],
        "headroom_pass_all_runs": headroom_pass,
        "counterfactual_validity_pass_all_runs": validity_pass,
        "mean_rho_offset": float(np.mean([x["rho_offset"] for x in energy_rows])),
        "min_rho_offset": float(np.min([x["rho_offset"] for x in energy_rows])),
        "mean_donor_distance_decrease": float(np.mean([x["donor_Gperp_distance_decrease_mean"] for x in validity_rows])),
        "max_relative_geometry_projection_error": float(np.max([x["geometry_projection_relative_max"] for x in validity_rows])),
        "development_validation_used": False,
        "outer_test_used": False,
        "decision_rule": {"rho_offset_min": 0.05, "median_delta_perp_norm_min": 1e-6, "invariant_relative_max": INVARIANT_RTOL},
    }
    write_json(OUT / "headroom" / "COUNTERFACTUAL_VALIDITY_REPORT.json", report)
    write_json(OUT / "protocol" / "PERSIST_CF_ADAPTATION_LOG.json", {
        "events": [{"phase": "A-D", "decision": status, "evidence": "TRAIN-only structural and validity audits"}],
        "development_validation_used": False, "outer_test_used": False,
    })
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def softmax_np(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    value = value - value.max(axis=1, keepdims=True)
    exp = np.exp(value)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def js_np(prob_a: np.ndarray, prob_b: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(prob_a, dtype=np.float64), EPS, 1.0)
    b = np.clip(np.asarray(prob_b, dtype=np.float64), EPS, 1.0)
    middle = 0.5 * (a + b)
    return 0.5 * np.sum(a * np.log(a / middle), axis=1) + 0.5 * np.sum(b * np.log(b / middle), axis=1)


def js_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    a = log_a.exp()
    b = log_b.exp()
    middle = 0.5 * (a + b)
    log_middle = torch.log(torch.clamp(middle, min=EPS))
    return 0.5 * (
        F.kl_div(log_middle, a, reduction="batchmean") +
        F.kl_div(log_middle, b, reduction="batchmean")
    )


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    p = np.asarray(prob, dtype=np.float64)
    target = np.asarray(y, dtype=np.int64)
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == target
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence >= edges[i]) & (confidence < edges[i + 1])
        if np.any(mask):
            result += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def classification_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=np.int64)
    prob = softmax_np(logits)
    pred = prob.argmax(axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "accuracy": float(accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, average="macro")),
        "nll": float(log_loss(target, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(target, prob[:, 1])),
        "ece": expected_calibration_error(prob, target),
    }


def initialise_model(run: AuditRun, base: SelectedBase, geometry: TrainGeometry,
                     stream_tag: str, device: torch.device):
    from persist_eeg_stage0.models import build_shared_model

    # This exactly follows the P5.1 V2 matched-control initialization rule.
    seed_all(P56.stable_seed("p5.1-init", P56.IMPLEMENTATION_ID, run.fold, run.seed, stream_tag))
    checkpoint_path, _, _ = P5.historical_checkpoint(run.fold, run.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    historical = build_shared_model("eegnet", int(run.meta.n_channels.iloc[0]), 128, P56.TASK_CLASSES)
    historical.load_state_dict(checkpoint["model"])
    targets = SimpleNamespace(global_direction=geometry.directions)
    return P5.ICGModel(historical.heads[TASK], run.art, "V2", targets, base.bottleneck).to(device)


def torch_inverse_delta(delta_q: torch.Tensor, model: Any) -> torch.Tensor:
    return (delta_q @ model.directions.T) @ model.dewhitener


def make_training_shift(
    mode: str,
    batch_subjects: Sequence[str],
    geometry: TrainGeometry,
    config: CFConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    n = len(batch_subjects)
    q_dim = next(iter(geometry.centers.values())).shape[0]
    raw = np.zeros((n, q_dim), dtype=np.float32)
    for i, source_value in enumerate(batch_subjects):
        source = str(source_value)
        candidates = [x for x in geometry.subjects if x != source]
        donor = candidates[int(rng.integers(len(candidates)))]
        raw[i] = geometry.centers[donor] - geometry.centers[source]
    real = project_shift(raw, geometry, keep_geometry=True)
    if mode == "duplicate":
        shift = np.zeros_like(raw)
    elif mode == "full":
        shift = raw
    elif mode == "cf":
        shift = real
    elif mode == "random":
        shift = np.zeros_like(real)
        for block in sorted(geometry.block_dims):
            dims = geometry.block_dims[block]
            g = geometry.directions[block].reshape(1, -1)
            random_part = rng.normal(size=(n, len(dims))).astype(np.float32)
            random_part = random_part - (random_part @ g.T) @ g
            random_norm = np.linalg.norm(random_part, axis=1, keepdims=True)
            target_norm = np.linalg.norm(real[:, dims], axis=1, keepdims=True)
            random_part = random_part / np.maximum(random_norm, EPS) * target_norm
            shift[:, dims] = random_part
    else:
        raise ValueError(mode)
    mask = rng.random(n) < float(config.p_cf)
    applied = float(config.alpha) * shift
    applied[~mask] = 0.0
    projection = geometry_projection(applied, geometry)
    norms = np.linalg.norm(applied, axis=1)
    diag = {
        "n": float(n),
        "n_augmented": float(mask.sum()),
        "shift_norm_sum": float(norms.sum()),
        "geometry_projection_norm_sum": float(np.linalg.norm(projection, axis=1).sum()),
        "geometry_projection_norm_max": float(np.linalg.norm(projection, axis=1).max(initial=0.0)),
    }
    return torch.as_tensor(applied, dtype=torch.float32, device=device), diag


def make_hard_training_shift(
    batch_subjects: Sequence[str],
    geometry: TrainGeometry,
    config: CFConfig,
    rng: np.random.Generator,
    model: Any,
    clean_h: torch.Tensor,
    clean_q: torch.Tensor,
    target: torch.Tensor,
    pool_size: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Choose the highest-current-CE empirical donor from four TRAIN donors."""
    n = len(batch_subjects)
    q_dim = next(iter(geometry.centers.values())).shape[0]
    candidates = np.zeros((pool_size, n, q_dim), dtype=np.float32)
    for i, source_value in enumerate(batch_subjects):
        source = str(source_value)
        available = [x for x in geometry.subjects if x != source]
        order = rng.permutation(len(available))[:pool_size]
        for j, donor_index in enumerate(order):
            raw = geometry.centers[available[int(donor_index)]] - geometry.centers[source]
            candidates[j, i] = project_shift(raw, geometry, keep_geometry=True)
    candidates *= float(config.alpha)
    losses = []
    with torch.no_grad():
        for j in range(pool_size):
            dq = torch.as_tensor(candidates[j], dtype=torch.float32, device=clean_h.device)
            logits, _, _ = model(clean_h + torch_inverse_delta(dq, model), clean_q + dq)
            losses.append(F.cross_entropy(logits, target, reduction="none"))
    chosen = torch.stack(losses, dim=0).argmax(dim=0).detach().cpu().numpy()
    selected = candidates[chosen, np.arange(n)]
    mask = rng.random(n) < float(config.p_cf)
    selected[~mask] = 0.0
    projection = geometry_projection(selected, geometry)
    norms = np.linalg.norm(selected, axis=1)
    diag = {
        "n": float(n),
        "n_augmented": float(mask.sum()),
        "shift_norm_sum": float(norms.sum()),
        "geometry_projection_norm_sum": float(np.linalg.norm(projection, axis=1).sum()),
        "geometry_projection_norm_max": float(np.linalg.norm(projection, axis=1).max(initial=0.0)),
    }
    return torch.as_tensor(selected, dtype=torch.float32, device=clean_h.device), diag


def stress_bank(geometry: TrainGeometry, fold: int, seed: int, inner: int,
                count: int = 12) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    pairs = [(source, donor) for source in geometry.subjects for donor in geometry.subjects if donor != source]
    rng = np.random.default_rng(stable_seed("persist-cf-stress-bank", fold, seed, inner))
    chosen = [pairs[i] for i in rng.permutation(len(pairs))[:min(count, len(pairs))]]
    shifts, rows = [], []
    for index, (source, donor) in enumerate(chosen):
        raw = geometry.centers[donor] - geometry.centers[source]
        delta = project_shift(raw, geometry, keep_geometry=True).astype(np.float32)
        shifts.append(delta)
        rows.append({
            "index": index, "source": source, "donor": donor,
            "norm": float(np.linalg.norm(delta)),
            "sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
            "delta_q": delta.tolist(),
        })
    return shifts, rows


def evaluate_stress(model: Any, h: np.ndarray, q: np.ndarray, y: np.ndarray,
                    shifts: Sequence[np.ndarray], clean_logits: np.ndarray,
                    device: torch.device) -> dict[str, Any]:
    clean_prob = softmax_np(clean_logits)
    bas, nlls, changes, rows = [], [], [], []
    for index, delta in enumerate(shifts):
        dq = np.broadcast_to(np.asarray(delta, dtype=np.float32), (len(q), len(delta))).copy()
        shifted_q = np.asarray(q, dtype=np.float32) + dq
        shifted_h = np.asarray(h, dtype=np.float32) + inverse_delta(dq, SimpleNamespace(
            directions=model.directions.detach().cpu().numpy(),
            dewhitener=model.dewhitener.detach().cpu().numpy(),
        ))
        ba, logits, _, _ = P56.eval_model(model, shifted_h, shifted_q, y, device)
        prob = softmax_np(logits)
        nll = float(log_loss(y, prob, labels=[0, 1]))
        sensitivity = float(js_np(clean_prob, prob).mean())
        bas.append(float(ba)); nlls.append(nll); changes.append(sensitivity)
        rows.append({"stress_index": index, "balanced_accuracy": float(ba), "nll": nll, "prediction_js": sensitivity})
    ordered = np.sort(np.asarray(bas, dtype=np.float64))
    count = max(1, int(math.ceil(len(ordered) / 4)))
    clean_ba = float(balanced_accuracy_score(y, np.asarray(clean_logits).argmax(1)))
    return {
        "robust_ba": float(np.mean(bas)),
        "delta_ba_under_shift": float(np.mean(bas) - clean_ba),
        "worst_quartile_ba": float(np.mean(ordered[:count])),
        "worst_shift_ba": float(np.min(bas)),
        "robust_nll": float(np.mean(nlls)),
        "prediction_js_sensitivity": float(np.mean(changes)),
        "stress_rows": rows,
    }


def subject_metric_rows(logits: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> list[dict[str, Any]]:
    pred = np.asarray(logits).argmax(1)
    result = []
    for subject, group in meta.reset_index(drop=True).groupby("subject", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        result.append({
            "subject": str(subject),
            "balanced_accuracy": float(balanced_accuracy_score(y[idx], pred[idx])),
            "n": int(len(idx)),
        })
    return result


def train_and_score(
    run: AuditRun,
    base: SelectedBase,
    geometry: TrainGeometry,
    train_pos: np.ndarray,
    held_pos: np.ndarray,
    inner: int,
    config: CFConfig,
    mode: str,
    shifts: Sequence[np.ndarray],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    stream_tag = f"persist-cf-inner-{inner}"
    model = initialise_model(run, base, geometry, stream_tag, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.learning_rate, weight_decay=1e-3)
    train_meta = run.meta.iloc[train_pos].reset_index(drop=True)
    held_meta = run.meta.iloc[held_pos].reset_index(drop=True)
    h_train = torch.as_tensor(np.asarray(run.h[train_pos], dtype=np.float32), device=device)
    q_train = torch.as_tensor(np.asarray(run.q[train_pos], dtype=np.float32), device=device)
    y_train = torch.as_tensor(labels(run.meta, train_pos), dtype=torch.long, device=device)
    sampler = P5.StructuredSampler(
        train_meta, train_meta.subject.astype(str).unique().tolist(),
        subjects_per_batch=6, trials_per_class=4,
    )
    train_subject_values = train_meta.subject.astype(str).to_numpy()
    curves: list[dict[str, Any]] = []
    diagnostics = {"n": 0.0, "n_augmented": 0.0, "shift_norm_sum": 0.0,
                   "geometry_projection_norm_sum": 0.0, "geometry_projection_norm_max": 0.0}
    started = time.time()
    for epoch in range(base.epochs):
        model.train()
        batches = sampler.batches(
            epoch,
            P56.stable_seed("p5.1-sampler", P56.IMPLEMENTATION_ID, run.fold, run.seed, stream_tag),
        )
        totals = {"loss": 0.0, "clean_ce": 0.0, "branch_ce": 0.0, "consistency": 0.0, "drift": 0.0}
        for step, batch in enumerate(batches):
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            clean_logits, _, adapter_delta = model(h_train.index_select(0, idx), q_train.index_select(0, idx))
            clean_ce = F.cross_entropy(clean_logits, y_train.index_select(0, idx))
            drift = P5.drift_loss(adapter_delta, run.art)
            if mode == "base":
                branch_ce = clean_ce.detach() * 0.0
                consistency = clean_ce.detach() * 0.0
                loss = clean_ce + base.lambda_drift * drift
            else:
                rng = np.random.default_rng(stable_seed(
                    "persist-cf-augmentation-stream", run.fold, run.seed, inner, epoch, step,
                ))
                clean_h_batch = h_train.index_select(0, idx)
                clean_q_batch = q_train.index_select(0, idx)
                target_batch = y_train.index_select(0, idx)
                if mode == "hard":
                    dq, diag = make_hard_training_shift(
                        train_subject_values[batch].tolist(), geometry, config, rng,
                        model, clean_h_batch, clean_q_batch, target_batch,
                    )
                else:
                    dq, diag = make_training_shift(
                        mode, train_subject_values[batch].tolist(), geometry, config, rng, device,
                    )
                for key in diagnostics:
                    if key.endswith("_max"):
                        diagnostics[key] = max(diagnostics[key], diag[key])
                    else:
                        diagnostics[key] += diag[key]
                branch_h = clean_h_batch + torch_inverse_delta(dq, model)
                branch_q = clean_q_batch + dq
                branch_logits, _, _ = model(branch_h, branch_q)
                branch_ce = F.cross_entropy(branch_logits, target_batch)
                consistency = js_loss(clean_logits, branch_logits)
                loss = (clean_ce + float(config.lambda_cf) * branch_ce +
                        float(config.lambda_cons) * consistency + base.lambda_drift * drift)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["clean_ce"] += float(clean_ce.detach())
            totals["branch_ce"] += float(branch_ce.detach())
            totals["consistency"] += float(consistency.detach())
            totals["drift"] += float(drift.detach())
        curves.append({"epoch": epoch, **{k: v / max(len(batches), 1) for k, v in totals.items()}})
    h_held = np.asarray(run.h[held_pos], dtype=np.float32)
    q_held = np.asarray(run.q[held_pos], dtype=np.float32)
    y_held = labels(run.meta, held_pos)
    _, logits, _, _ = P56.eval_model(model, h_held, q_held, y_held, device)
    natural = classification_metrics(logits, y_held)
    robust = evaluate_stress(model, h_held, q_held, y_held, shifts, logits, device)
    n_aug = max(diagnostics["n_augmented"], 1.0)
    result = {
        "fold": run.fold, "seed": run.seed, "inner_fold": inner,
        "mode": mode, "config": config.name,
        **{f"clean_{key}": value for key, value in natural.items()},
        **{key: value for key, value in robust.items() if key != "stress_rows"},
        "epochs": base.epochs, "optimizer_steps": int(base.epochs * sampler.steps),
        "elapsed_seconds": time.time() - started,
        "n_augmented": int(diagnostics["n_augmented"]),
        "mean_applied_shift_norm": float(diagnostics["shift_norm_sum"] / n_aug),
        "mean_geometry_projection_error": float(diagnostics["geometry_projection_norm_sum"] / n_aug),
        "max_geometry_projection_error": float(diagnostics["geometry_projection_norm_max"]),
        "alpha": config.alpha, "p_cf": config.p_cf,
        "lambda_cf": config.lambda_cf, "lambda_cons": config.lambda_cons,
        "scale_policy": config.scale_policy, "donor_policy": config.donor_policy,
        "development_validation_used": False, "outer_test_used": False,
    }
    subjects = subject_metric_rows(logits, y_held, held_meta)
    del model, optimizer, h_train, q_train, y_train
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, subjects, curves


def score_historical(
    run: AuditRun,
    base: SelectedBase,
    geometry: TrainGeometry,
    held_pos: np.ndarray,
    inner: int,
    shifts: Sequence[np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    path = cv_result_path(run.fold, run.seed, inner, "historical")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("implementation_id") == IMPLEMENTATION_ID:
            return payload
    model = initialise_model(run, base, geometry, f"persist-cf-inner-{inner}", device)
    h = np.asarray(run.h[held_pos], dtype=np.float32)
    q = np.asarray(run.q[held_pos], dtype=np.float32)
    y = labels(run.meta, held_pos)
    held_meta = run.meta.iloc[held_pos].reset_index(drop=True)
    _, logits, _, _ = P56.eval_model(model, h, q, y, device)
    natural = classification_metrics(logits, y)
    robust = evaluate_stress(model, h, q, y, shifts, logits, device)
    payload = {
        "implementation_id": IMPLEMENTATION_ID,
        "fold": run.fold, "seed": run.seed, "inner_fold": inner,
        "mode": "historical", "config": "historical_eegnet",
        **{f"clean_{key}": value for key, value in natural.items()},
        **{key: value for key, value in robust.items() if key != "stress_rows"},
        "epochs": 0, "optimizer_steps": 0, "elapsed_seconds": 0.0,
        "n_augmented": 0, "mean_applied_shift_norm": 0.0,
        "mean_geometry_projection_error": 0.0, "max_geometry_projection_error": 0.0,
        "alpha": 0.0, "p_cf": 0.0, "lambda_cf": 0.0, "lambda_cons": 0.0,
        "scale_policy": "none", "donor_policy": "none",
        "development_validation_used": False, "outer_test_used": False,
        "subject_rows": subject_metric_rows(logits, y, held_meta),
    }
    write_json(path, payload)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[HISTORICAL] fold={run.fold} seed={run.seed} inner={inner} clean={payload['clean_balanced_accuracy']:.4f}", flush=True)
    return payload


def cv_result_path(fold: int, seed: int, inner: int, label: str) -> Path:
    return OUT / "CF0" / "CONFIGS" / f"fold-{fold}" / f"seed-{seed}" / f"inner-{inner}" / label / "RESULT.json"


def run_cv_cell(run: AuditRun, base: SelectedBase, geometry: TrainGeometry,
                train_pos: np.ndarray, held_pos: np.ndarray, inner: int,
                config: CFConfig, mode: str, shifts: Sequence[np.ndarray],
    device: torch.device) -> dict[str, Any]:
    label = config.name if mode == "cf" else mode
    if mode == "hard":
        path = OUT / "CF1" / "CONFIGS" / f"fold-{run.fold}" / f"seed-{run.seed}" / f"inner-{inner}" / label / "RESULT.json"
    else:
        path = cv_result_path(run.fold, run.seed, inner, label)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("implementation_id") == IMPLEMENTATION_ID:
            return payload
    result, subjects, curves = train_and_score(
        run, base, geometry, train_pos, held_pos, inner, config, mode, shifts, device,
    )
    payload = {"implementation_id": IMPLEMENTATION_ID, **result, "subject_rows": subjects}
    write_json(path, payload)
    version_dir = "CF1" if mode == "hard" else "CF0"
    log_dir = OUT / version_dir / "TRAIN_LOGS" / f"fold-{run.fold}" / f"seed-{run.seed}" / f"inner-{inner}"
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(log_dir / f"{label}.csv", index=False)
    print(f"[CF0] fold={run.fold} seed={run.seed} inner={inner} mode={label} "
          f"clean={result['clean_balanced_accuracy']:.4f} robust={result['robust_ba']:.4f}", flush=True)
    return payload


def select_cf0(rows: Sequence[dict[str, Any]], base_rows: Sequence[dict[str, Any]]) -> tuple[CFConfig, dict[str, Any]]:
    base_map = {(int(x["fold"]), int(x["seed"]), int(x["inner_fold"])): x for x in base_rows}
    aggregates = []
    for config in CF0_CONFIGS:
        group = [x for x in rows if x["config"] == config.name and x["mode"] == "cf"]
        natural_delta = [float(x["clean_balanced_accuracy"]) - float(base_map[(x["fold"], x["seed"], x["inner_fold"])]["clean_balanced_accuracy"]) for x in group]
        robust_delta = [float(x["robust_ba"]) - float(base_map[(x["fold"], x["seed"], x["inner_fold"])]["robust_ba"]) for x in group]
        aggregates.append({
            **asdict(config), "n_inner_folds": len(group),
            "mean_clean_ba": float(np.mean([x["clean_balanced_accuracy"] for x in group])),
            "mean_natural_delta_ba": float(np.mean(natural_delta)),
            "positive_natural_inner_folds": int(np.sum(np.asarray(natural_delta) > 0)),
            "mean_robust_ba": float(np.mean([x["robust_ba"] for x in group])),
            "mean_robust_delta_ba": float(np.mean(robust_delta)),
            "positive_robust_inner_folds": int(np.sum(np.asarray(robust_delta) > 0)),
        })
    if any(x["mean_natural_delta_ba"] >= 0.002 for x in aggregates):
        eligible = [x for x in aggregates if x["mean_natural_delta_ba"] >= 0.002]
        selected = max(eligible, key=lambda x: (x["mean_natural_delta_ba"], x["mean_robust_delta_ba"], -x["alpha"], -x["p_cf"]))
        rule = "highest natural Delta BA among configs reaching +0.002 TRAIN-CV"
    elif any(x["mean_natural_delta_ba"] >= -0.001 for x in aggregates):
        eligible = [x for x in aggregates if x["mean_natural_delta_ba"] >= -0.001]
        selected = max(eligible, key=lambda x: (x["mean_robust_delta_ba"], x["mean_natural_delta_ba"], -x["alpha"], -x["p_cf"]))
        rule = "highest robustness Delta BA among clean-noninferior configs"
    else:
        selected = max(aggregates, key=lambda x: (x["mean_natural_delta_ba"], x["mean_robust_delta_ba"], -x["alpha"], -x["p_cf"]))
        rule = "least natural degradation because no config was clean-noninferior"
    config = next(x for x in CF0_CONFIGS if x.name == selected["name"])
    return config, {"selection_rule": rule, "selected": selected, "configs": aggregates}


def cf0(device: torch.device) -> dict[str, Any]:
    audit_path = OUT / "headroom" / "COUNTERFACTUAL_VALIDITY_REPORT.json"
    if not audit_path.exists() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "PERSIST_CF_AUDIT_PASS_READY_FOR_CF0":
        raise RuntimeError("Phase A-D did not pass; refusing CF0 training")
    meta = P5.load_mi_manifest()
    bases = selected_bases()
    all_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    stress_freeze: list[dict[str, Any]] = []
    run_selections: list[dict[str, Any]] = []
    # First stage: six serious CF0 configurations and one shared matched base.
    for fold in FOLDS:
        for seed in SEEDS:
            run = load_run(fold, seed, meta)
            inner_parts = subject_folds(run.train_subjects, fold, seed)
            run_candidate_rows, run_base_rows = [], []
            cells: list[tuple[int, TrainGeometry, np.ndarray, np.ndarray, list[np.ndarray]]] = []
            for inner, held_subjects in enumerate(inner_parts):
                train_subjects = [x for x in run.train_subjects if x not in set(held_subjects)]
                train_pos = positions(meta, train_subjects)
                held_pos = positions(meta, held_subjects)
                geometry = fit_geometry(run, train_pos)
                shifts, freeze_rows = stress_bank(geometry, fold, seed, inner)
                stress_freeze.append({
                    "fold": fold, "seed": seed, "inner_fold": inner,
                    "train_subjects": train_subjects, "held_subjects": held_subjects,
                    "bank_seed": stable_seed("persist-cf-stress-bank", fold, seed, inner),
                    "pairs": freeze_rows,
                })
                cells.append((inner, geometry, train_pos, held_pos, shifts))
                neutral = CFConfig("matched_base", 0.0, 0.0, 0.0, 0.0)
                base_result = run_cv_cell(run, bases[(fold, seed)], geometry, train_pos, held_pos, inner, neutral, "base", shifts, device)
                run_base_rows.append(base_result); base_rows.append(base_result); all_rows.append(base_result)
                for config in CF0_CONFIGS:
                    result = run_cv_cell(run, bases[(fold, seed)], geometry, train_pos, held_pos, inner, config, "cf", shifts, device)
                    run_candidate_rows.append(result); all_rows.append(result)
            selected_config, selection = select_cf0(run_candidate_rows, run_base_rows)
            run_selections.append({"fold": fold, "seed": seed, **selection})
            # Required controls are run only for the selected CF0 config; this
            # keeps compute bounded while matching its streams and loss weights.
            for inner, geometry, train_pos, held_pos, shifts in cells:
                for mode in ("duplicate", "full", "random"):
                    result = run_cv_cell(run, bases[(fold, seed)], geometry, train_pos, held_pos, inner, selected_config, mode, shifts, device)
                    all_rows.append(result)
    write_json(OUT / "protocol" / "STRESS_BANK_FREEZE.json", {
        "seed_rule": "SHA256(persist-cf-stress-bank|fold|seed|inner)",
        "construction": "fixed inner-TRAIN donor-pair blockwise projected offsets",
        "held_subject_statistics_or_labels_used": False,
        "banks": stress_freeze, "outer_test_used": False,
    })
    write_json(OUT / "protocol" / "RANDOM_CONTROL_FREEZE.json", {
        "rule": "per-sample Gaussian direction projected into the same Protected G_perp block and rescaled to the empirical projected donor-offset norm",
        "seed_rule": "shared SHA256 augmentation stream plus numpy PCG64",
        "matched": ["block rank", "per-block norm", "p_cf", "optimizer steps", "loss weights"],
        "outer_test_used": False,
    })
    flat_rows = [{k: v for k, v in row.items() if k not in {"subject_rows", "stress_rows", "implementation_id"}} for row in all_rows]
    frame = pd.DataFrame(flat_rows)
    (OUT / "CF0").mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "CF0" / "TRAIN_CV_RESULTS.csv", index=False)
    write_json(OUT / "CF0" / "RUN_RESULTS" / "CF0_SELECTIONS.json", {"runs": run_selections})
    selected_names = {(int(x["fold"]), int(x["seed"])): x["selected"]["name"] for x in run_selections}
    selected_rows = [x for x in all_rows if x["mode"] == "cf" and x["config"] == selected_names[(int(x["fold"]), int(x["seed"]))]]
    matched_map = {(int(x["fold"]), int(x["seed"]), int(x["inner_fold"])): x for x in base_rows}
    control_maps = {
        mode: {(int(x["fold"]), int(x["seed"]), int(x["inner_fold"])): x for x in all_rows if x["mode"] == mode}
        for mode in ("duplicate", "full", "random")
    }
    natural_delta = np.asarray([x["clean_balanced_accuracy"] - matched_map[(x["fold"], x["seed"], x["inner_fold"])]["clean_balanced_accuracy"] for x in selected_rows], dtype=np.float64)
    robust_delta = np.asarray([x["robust_ba"] - matched_map[(x["fold"], x["seed"], x["inner_fold"])]["robust_ba"] for x in selected_rows], dtype=np.float64)
    cf_vs_controls = {}
    for mode, mapping in control_maps.items():
        cf_vs_controls[mode] = {
            "clean_delta": float(np.mean([x["clean_balanced_accuracy"] - mapping[(x["fold"], x["seed"], x["inner_fold"])]["clean_balanced_accuracy"] for x in selected_rows])),
            "robust_delta": float(np.mean([x["robust_ba"] - mapping[(x["fold"], x["seed"], x["inner_fold"])]["robust_ba"] for x in selected_rows])),
        }
    mechanism_specific = bool(cf_vs_controls["random"]["robust_delta"] > 0 and cf_vs_controls["full"]["robust_delta"] > 0)
    if float(natural_delta.mean()) < -0.003 and float(robust_delta.mean()) < 0.010:
        decision = "PERSIST_CF_NOT_SUPPORTED"
    elif float(natural_delta.mean()) >= 0.002 or (float(robust_delta.mean()) >= 0.010 and mechanism_specific):
        decision = "PERSIST_CF_CF0_SUPPORTS_TRAIN_ONLY_LOCK"
    elif float(robust_delta.mean()) > 0 and float(natural_delta.mean()) < -0.001:
        decision = "PERSIST_CF_CF0_JUSTIFIES_CF1_SCALE"
    elif not mechanism_specific:
        decision = "PERSIST_CF_NO_MECHANISM_SPECIFICITY"
    else:
        decision = "PERSIST_CF_CF0_WEAK_SIGNAL"
    report = {
        "status": decision,
        "implementation_id": IMPLEMENTATION_ID,
        "n_serious_configs_per_run": len(CF0_CONFIGS),
        "n_inner_fold_selected_results": len(selected_rows),
        "mean_natural_delta_ba_vs_matched": float(natural_delta.mean()),
        "positive_natural_inner_folds": int((natural_delta > 0).sum()),
        "mean_robust_delta_ba_vs_matched": float(robust_delta.mean()),
        "positive_robust_inner_folds": int((robust_delta > 0).sum()),
        "cf_vs_controls": cf_vs_controls,
        "mechanism_specific_against_random_and_full": mechanism_specific,
        "run_selections": run_selections,
        "development_validation_used": False,
        "outer_test_used": False,
    }
    write_json(OUT / "CF0" / "VERSION_REPORT.json", report)
    markdown = (
        "# PERSIST-CF CF0 TRAIN-only report\n\n"
        f"Status: `{decision}`\n\n"
        f"Mean natural ΔBA vs matched: {natural_delta.mean():+.6f}.  "
        f"Mean stress robust ΔBA: {robust_delta.mean():+.6f}.\n\n"
        "Development validation and outer test were not used.\n"
    )
    (OUT / "CF0" / "VERSION_REPORT.md").write_text(markdown, encoding="utf-8")
    mechanism = frame[frame["mode"].isin(["base", "duplicate", "full", "random", "cf"])][[
        "fold", "seed", "inner_fold", "mode", "config", "mean_applied_shift_norm",
        "mean_geometry_projection_error", "max_geometry_projection_error",
        "prediction_js_sensitivity", "robust_ba", "delta_ba_under_shift",
    ]]
    mechanism.to_csv(OUT / "CF0" / "MECHANISM_METRICS.csv", index=False)
    adaptation_path = OUT / "protocol" / "PERSIST_CF_ADAPTATION_LOG.json"
    adaptation = json.loads(adaptation_path.read_text(encoding="utf-8"))
    adaptation["events"].append({"phase": "E-F", "decision": decision, "evidence": {
        "mean_natural_delta_ba": float(natural_delta.mean()),
        "mean_robust_delta_ba": float(robust_delta.mean()),
        "mechanism_specific": mechanism_specific,
    }})
    write_json(adaptation_path, adaptation)
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def selected_cf0_configs() -> dict[tuple[int, int], CFConfig]:
    path = OUT / "CF0" / "VERSION_REPORT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload["run_selections"]:
        name = row["selected"]["name"]
        result[(int(row["fold"]), int(row["seed"]))] = next(x for x in CF0_CONFIGS if x.name == name)
    if len(result) != 6:
        raise RuntimeError("Incomplete CF0 selection map")
    return result


def load_cv_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cf1_hard(device: torch.device) -> dict[str, Any]:
    cf0_report = json.loads((OUT / "CF0" / "VERSION_REPORT.json").read_text(encoding="utf-8"))
    if cf0_report["status"] not in {"PERSIST_CF_CF0_WEAK_SIGNAL", "PERSIST_CF_CF0_JUSTIFIES_CF1_SCALE", "PERSIST_CF_CF0_SUPPORTS_TRAIN_ONLY_LOCK"}:
        raise RuntimeError(f"CF0 evidence does not authorize refinement: {cf0_report['status']}")
    cf0_frame = pd.read_csv(OUT / "CF0" / "TRAIN_CV_RESULTS.csv")
    base_frame = cf0_frame[cf0_frame["mode"] == "base"]
    mean_worst_damage = float((base_frame.worst_shift_ba - base_frame.clean_balanced_accuracy).mean())
    if not (float(cf0_report["mean_robust_delta_ba_vs_matched"]) > 0 and mean_worst_damage < -0.005):
        raise RuntimeError("CF1-HARD prerequisites absent: uniform CF must help robustness and stress offsets must be heterogeneous/damaging")
    selected = selected_cf0_configs()
    meta = P5.load_mi_manifest()
    bases = selected_bases()
    hard_rows, comparison_rows = [], []
    for fold in FOLDS:
        for seed in SEEDS:
            run = load_run(fold, seed, meta)
            config = selected[(fold, seed)]
            inner_parts = subject_folds(run.train_subjects, fold, seed)
            for inner, held_subjects in enumerate(inner_parts):
                train_subjects = [x for x in run.train_subjects if x not in set(held_subjects)]
                train_pos = positions(meta, train_subjects)
                held_pos = positions(meta, held_subjects)
                geometry = fit_geometry(run, train_pos)
                shifts, _ = stress_bank(geometry, fold, seed, inner)
                hard = run_cv_cell(run, bases[(fold, seed)], geometry, train_pos, held_pos, inner, config, "hard", shifts, device)
                hard_rows.append(hard)
                base = load_cv_payload(cv_result_path(fold, seed, inner, "base"))
                cf = load_cv_payload(cv_result_path(fold, seed, inner, config.name))
                controls = {mode: load_cv_payload(cv_result_path(fold, seed, inner, mode)) for mode in ("duplicate", "full", "random")}
                comparison_rows.append({
                    "fold": fold, "seed": seed, "inner_fold": inner,
                    "config": config.name,
                    "hard_clean_ba": hard["clean_balanced_accuracy"],
                    "base_clean_ba": base["clean_balanced_accuracy"],
                    "cf0_clean_ba": cf["clean_balanced_accuracy"],
                    "hard_natural_delta_vs_base": hard["clean_balanced_accuracy"] - base["clean_balanced_accuracy"],
                    "hard_natural_delta_vs_cf0": hard["clean_balanced_accuracy"] - cf["clean_balanced_accuracy"],
                    "hard_robust_ba": hard["robust_ba"],
                    "base_robust_ba": base["robust_ba"],
                    "cf0_robust_ba": cf["robust_ba"],
                    "hard_robust_delta_vs_base": hard["robust_ba"] - base["robust_ba"],
                    "hard_robust_delta_vs_cf0": hard["robust_ba"] - cf["robust_ba"],
                    **{f"hard_clean_delta_vs_{mode}": hard["clean_balanced_accuracy"] - controls[mode]["clean_balanced_accuracy"] for mode in controls},
                    **{f"hard_robust_delta_vs_{mode}": hard["robust_ba"] - controls[mode]["robust_ba"] for mode in controls},
                })
    out_dir = OUT / "CF1"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in x.items() if k not in {"subject_rows", "stress_rows", "implementation_id"}} for x in hard_rows]).to_csv(out_dir / "TRAIN_CV_RESULTS.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "HARD_COMPARISON.csv", index=False)
    natural = comparison.hard_natural_delta_vs_base.to_numpy(dtype=np.float64)
    robust = comparison.hard_robust_delta_vs_base.to_numpy(dtype=np.float64)
    mechanism_specific = bool(comparison.hard_robust_delta_vs_random.mean() > 0 and comparison.hard_robust_delta_vs_full.mean() > 0)
    if natural.mean() >= 0.002 or (robust.mean() >= 0.010 and mechanism_specific):
        decision = "PERSIST_CF_CF1_HARD_SUPPORTS_TRAIN_ONLY_LOCK"
    elif natural.mean() < -0.003 and robust.mean() < 0.010:
        decision = "PERSIST_CF_NOT_SUPPORTED"
    elif natural.mean() < 0.002 and robust.mean() < 0.010:
        decision = "PERSIST_CF_NOT_SUPPORTED"
    elif not mechanism_specific:
        decision = "PERSIST_CF_NO_MECHANISM_SPECIFICITY"
    else:
        decision = "PERSIST_CF_NOT_SUPPORTED"
    report = {
        "status": decision,
        "family": "CF1-HARD",
        "authorization_evidence": {
            "cf0_mean_robust_delta_ba": float(cf0_report["mean_robust_delta_ba_vs_matched"]),
            "matched_mean_worst_offset_damage_ba": mean_worst_damage,
            "donor_candidate_pool": 4,
        },
        "mean_natural_delta_ba_vs_matched": float(natural.mean()),
        "positive_natural_inner_folds": int((natural > 0).sum()),
        "mean_robust_delta_ba_vs_matched": float(robust.mean()),
        "positive_robust_inner_folds": int((robust > 0).sum()),
        "mean_natural_delta_vs_cf0": float(comparison.hard_natural_delta_vs_cf0.mean()),
        "mean_robust_delta_vs_cf0": float(comparison.hard_robust_delta_vs_cf0.mean()),
        "mean_clean_delta_vs_duplicate": float(comparison.hard_clean_delta_vs_duplicate.mean()),
        "mean_robust_delta_vs_random": float(comparison.hard_robust_delta_vs_random.mean()),
        "mean_robust_delta_vs_full": float(comparison.hard_robust_delta_vs_full.mean()),
        "mechanism_specific_against_random_and_full": mechanism_specific,
        "development_validation_used": False,
        "outer_test_used": False,
        "stop_rule": "stop when best TRAIN-CV natural gain < +0.002 and robustness gain < +0.010",
    }
    write_json(out_dir / "VERSION_REPORT.json", report)
    (out_dir / "VERSION_REPORT.md").write_text(
        "# PERSIST-CF CF1-HARD TRAIN-only report\n\n"
        f"Status: `{decision}`\n\n"
        f"Mean natural ΔBA: {natural.mean():+.6f}; mean robust ΔBA: {robust.mean():+.6f}.\n\n"
        "Development validation and outer test were not used.\n",
        encoding="utf-8",
    )
    adaptation_path = OUT / "protocol" / "PERSIST_CF_ADAPTATION_LOG.json"
    adaptation = json.loads(adaptation_path.read_text(encoding="utf-8"))
    adaptation["events"].append({"phase": "F", "family": "CF1-HARD", "decision": decision,
                                  "authorization_evidence": report["authorization_evidence"]})
    write_json(adaptation_path, adaptation)
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def hierarchical_subject_ci(delta_rows: Sequence[dict[str, Any]], draws: int = 10_000,
                            seed: int = 20260816) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in delta_rows:
        grouped.setdefault(str(row["run"]), []).append(float(row["delta"]))
    runs = sorted(grouped)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        selected_runs = rng.choice(runs, size=len(runs), replace=True)
        samples[i] = np.mean([
            np.mean(rng.choice(grouped[run], size=len(grouped[run]), replace=True))
            for run in selected_runs
        ])
    return {
        "estimate": float(np.mean([np.mean(grouped[run]) for run in runs])),
        "bootstrap_mean": float(samples.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "sign_probability": float(np.mean(samples > 0)),
        "draws": draws, "n_runs": len(runs), "n_subject_values": int(sum(map(len, grouped.values()))),
    }


def finalize_negative(device: torch.device) -> dict[str, Any]:
    audit_report = json.loads((OUT / "headroom" / "COUNTERFACTUAL_VALIDITY_REPORT.json").read_text(encoding="utf-8"))
    cf0_report = json.loads((OUT / "CF0" / "VERSION_REPORT.json").read_text(encoding="utf-8"))
    cf1_report = json.loads((OUT / "CF1" / "VERSION_REPORT.json").read_text(encoding="utf-8"))
    if cf1_report.get("status") != "PERSIST_CF_NOT_SUPPORTED":
        raise RuntimeError("Negative finalization requires the frozen TRAIN-only stop state")
    selected = selected_cf0_configs()
    meta = P5.load_mi_manifest()
    bases = selected_bases()
    methods = ("historical", "base", "duplicate", "full", "random", "cf", "hard")
    rows_by_method: dict[str, list[dict[str, Any]]] = {x: [] for x in methods}
    subject_deltas: dict[str, list[dict[str, Any]]] = {x: [] for x in methods if x != "base"}
    for fold in FOLDS:
        for seed in SEEDS:
            run = load_run(fold, seed, meta)
            config = selected[(fold, seed)]
            inner_parts = subject_folds(run.train_subjects, fold, seed)
            for inner in range(INNER_FOLDS):
                held_subjects = inner_parts[inner]
                train_subjects = [x for x in run.train_subjects if x not in set(held_subjects)]
                train_pos = positions(meta, train_subjects)
                held_pos = positions(meta, held_subjects)
                geometry = fit_geometry(run, train_pos)
                shifts, _ = stress_bank(geometry, fold, seed, inner)
                payloads = {
                    "historical": score_historical(run, bases[(fold, seed)], geometry, held_pos, inner, shifts, device),
                    "base": load_cv_payload(cv_result_path(fold, seed, inner, "base")),
                    "duplicate": load_cv_payload(cv_result_path(fold, seed, inner, "duplicate")),
                    "full": load_cv_payload(cv_result_path(fold, seed, inner, "full")),
                    "random": load_cv_payload(cv_result_path(fold, seed, inner, "random")),
                    "cf": load_cv_payload(cv_result_path(fold, seed, inner, config.name)),
                    "hard": load_cv_payload(OUT / "CF1" / "CONFIGS" / f"fold-{fold}" / f"seed-{seed}" / f"inner-{inner}" / "hard" / "RESULT.json"),
                }
                for method, payload in payloads.items():
                    rows_by_method[method].append(payload)
                base_subject = {str(x["subject"]): float(x["balanced_accuracy"]) for x in payloads["base"]["subject_rows"]}
                for method in subject_deltas:
                    for row in payloads[method]["subject_rows"]:
                        subject = str(row["subject"])
                        subject_deltas[method].append({
                            "run": f"fold-{fold}/seed-{seed}", "inner_fold": inner,
                            "subject": subject,
                            "delta": float(row["balanced_accuracy"]) - base_subject[subject],
                        })
    base_clean = float(np.mean([x["clean_balanced_accuracy"] for x in rows_by_method["base"]]))
    base_robust = float(np.mean([x["robust_ba"] for x in rows_by_method["base"]]))
    labels_out = {
        "historical": "Historical EEGNet",
        "base": "Matched continued-training base",
        "duplicate": "Duplicate-clean control",
        "full": "Generic/full Protected subject swap",
        "random": "Random same-norm G_perp augmentation",
        "cf": "PERSIST-CF (CF0 selected)",
        "hard": "PERSIST-CF CF1-HARD",
    }
    summaries = []
    ci_payload = {}
    for method in methods:
        values = rows_by_method[method]
        if method == "base":
            ci = {"estimate": 0.0, "ci95": [0.0, 0.0], "sign_probability": None, "draws": 0,
                  "n_runs": 6, "n_subject_values": 0}
        else:
            ci = hierarchical_subject_ci(subject_deltas[method], seed=stable_seed("persist-cf-final-ci", method))
        ci_payload[method] = ci
        summaries.append({
            "method": labels_out[method], "key": method,
            "evaluation_scope": "OpenBMI outer-TRAIN five-fold subject-disjoint inner CV",
            "balanced_accuracy": float(np.mean([x["clean_balanced_accuracy"] for x in values])),
            "delta_ba_vs_matched": float(np.mean([x["clean_balanced_accuracy"] for x in values]) - base_clean),
            "ci95_subject_run": json.dumps(ci["ci95"]),
            "positive_inner_folds": int(np.sum([x["clean_balanced_accuracy"] - rows_by_method["base"][i]["clean_balanced_accuracy"] > 0 for i, x in enumerate(values)])),
            "macro_f1": float(np.mean([x["clean_macro_f1"] for x in values])),
            "nll": float(np.mean([x["clean_nll"] for x in values])),
            "robust_ba": float(np.mean([x["robust_ba"] for x in values])),
            "delta_robust_ba_vs_matched": float(np.mean([x["robust_ba"] for x in values]) - base_robust),
            "geometry_preservation_error": float(np.mean([x["mean_geometry_projection_error"] for x in values])),
            "offset_sensitivity": float(np.mean([x["prediction_js_sensitivity"] for x in values])),
        })
    summary_frame = pd.DataFrame(summaries)
    final_dir = OUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    ablations = summary_frame.copy()
    ablations.to_csv(final_dir / "PERSIST_CF_ABLATIONS.csv", index=False)
    mechanism = summary_frame[[
        "method", "geometry_preservation_error", "offset_sensitivity", "robust_ba",
        "delta_robust_ba_vs_matched", "balanced_accuracy", "delta_ba_vs_matched",
    ]].copy()
    mechanism.to_csv(final_dir / "MECHANISM_SUMMARY_TABLE.csv", index=False)
    required_rows = [
        {"method": "Historical EEGNet", "key": "historical"},
        {"method": "Matched continued-training base", "key": "base"},
        {"method": "Duplicate-clean control", "key": "duplicate"},
        {"method": "Generic/full Protected subject swap", "key": "full"},
        {"method": "Random same-norm G_perp augmentation", "key": "random"},
        {"method": "PERSIST-ICG best prior method", "note": "Prior result not mixed into TRAIN-only CF table"},
        {"method": "PERSIST-Router prior result", "note": "Prior result not mixed into TRAIN-only CF table"},
        {"method": "PERSIST-CF", "key": "cf"},
    ]
    indexed = {x["key"]: x for x in summaries}
    main_rows = []
    for item in required_rows:
        if item.get("key") in indexed:
            row = dict(indexed[item["key"]])
            row["method"] = item["method"]
            row["note"] = "TRAIN-only diagnostic; not a development result"
            main_rows.append(row)
        else:
            main_rows.append({
                "method": item["method"], "key": item.get("key"),
                "evaluation_scope": "not evaluated in PERSIST-CF before stop",
                "balanced_accuracy": None, "delta_ba_vs_matched": None,
                "ci95_subject_run": None, "positive_inner_folds": None,
                "macro_f1": None, "nll": None, "robust_ba": None,
                "delta_robust_ba_vs_matched": None,
                "geometry_preservation_error": None, "offset_sensitivity": None,
                "note": item["note"],
            })
    pd.DataFrame(main_rows).to_csv(final_dir / "MAIN_METHOD_TABLE.csv", index=False)
    write_json(final_dir / "TRAIN_ONLY_HIERARCHICAL_BOOTSTRAP.json", ci_payload)
    report = {
        "status": "PERSIST_CF_NOT_SUPPORTED",
        "implementation_id": IMPLEMENTATION_ID,
        "primary": {"dataset": "OpenBMI", "task": "MI", "backbone": "EEGNet"},
        "structural_audit": audit_report,
        "cf0": {
            "mean_natural_delta_ba": cf0_report["mean_natural_delta_ba_vs_matched"],
            "mean_robust_delta_ba": cf0_report["mean_robust_delta_ba_vs_matched"],
            "status": cf0_report["status"],
        },
        "cf1_hard": cf1_report,
        "final_train_only_methods": summaries,
        "success_gates": {
            "natural_required_mean_delta": 0.005,
            "robustness_required_mean_delta": 0.020,
            "minimum_signal_to_continue_natural": 0.002,
            "minimum_signal_to_continue_robustness": 0.010,
            "passed": False,
        },
        "refinement_families_run": ["CF1-HARD"],
        "second_refinement_run": False,
        "second_refinement_reason": "No specific correctable failure remained: hard donors produced only +0.000411 robust Delta BA and reduced clean performance relative to CF0.",
        "lock_created": False,
        "development_validation_evaluated": False,
        "outer_test_evaluated": False,
        "conditional_openbmi_erp_ssvep_run": False,
        "conditional_eegmmidb_run": False,
        "claim": "The frozen decomposition has real geometric headroom, but it does not provide sufficient train-time augmentation leverage for improved EEGNet/OpenBMI generalization under the tested protocol.",
    }
    write_json(final_dir / "PERSIST_CF_FINAL_REPORT.json", report)
    markdown = f"""# PERSIST-CF final report

Terminal state: `PERSIST_CF_NOT_SUPPORTED`.

The structural mechanism is valid but the learning result is not useful.  The
six TRAIN-only audits retained 73.9%--90.1% of subject-offset energy in
Protected $G^\\perp$, while keeping the relative geometry-projection error
below `{audit_report['max_relative_geometry_projection_error']:.3e}`.

CF0 improved natural BA by only `{cf0_report['mean_natural_delta_ba_vs_matched']:+.6f}`
and stress robust BA by `{cf0_report['mean_robust_delta_ba_vs_matched']:+.6f}`.
CF1-HARD changed these to `{cf1_report['mean_natural_delta_ba_vs_matched']:+.6f}`
and `{cf1_report['mean_robust_delta_ba_vs_matched']:+.6f}`.  Both robustness
effects are far below the `+0.010` continuation threshold and the `+0.020`
ROBUSTNESS_PASS threshold.  Duplicate-clean explains most of CF0's tiny
natural change.

No method lock was created.  Development validation, outer test, ERP, SSVEP,
and EEGMMIDB were not evaluated.  Continuing with another refinement would be
unsupported search rather than a protocol-driven adjustment.
"""
    (final_dir / "PERSIST_CF_FINAL_REPORT.md").write_text(markdown, encoding="utf-8")
    openbmi_dir = OUT / "openbmi_mi"
    write_json(openbmi_dir / "OPENBMI_MI_REPORT.json", {
        "status": "NOT_RUN_METHOD_STOPPED_BEFORE_LOCK",
        "reason": "TRAIN-only signal failed continuation thresholds",
        "development_validation_evaluated": False, "outer_test_evaluated": False,
    })
    adaptation_path = OUT / "protocol" / "PERSIST_CF_ADAPTATION_LOG.json"
    adaptation = json.loads(adaptation_path.read_text(encoding="utf-8"))
    adaptation["events"].append({"phase": "I", "decision": "PERSIST_CF_NOT_SUPPORTED",
                                  "action": "stopped before method lock and development evaluation"})
    write_json(adaptation_path, adaptation)
    print(json.dumps(clean(report), indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "cf0", "cf1-hard", "lock", "evaluate", "finalize", "all"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.command in {"audit", "all"}:
        result = audit(device)
        if result["status"] != "PERSIST_CF_AUDIT_PASS_READY_FOR_CF0":
            return
    if args.command in {"cf0", "all"}:
        result = cf0(device)
        if args.command == "cf0":
            return
        if result["status"] == "PERSIST_CF_CF0_WEAK_SIGNAL":
            cf1_hard(device)
        return
    if args.command == "cf1-hard":
        cf1_hard(device)
        return
    if args.command == "finalize":
        finalize_negative(device)
        return
    if args.command in {"lock", "evaluate"}:
        raise NotImplementedError(f"{args.command} will be enabled after the fail-closed Phase A-D gate passes")


if __name__ == "__main__":
    main()
