"""Frozen development-only C-direction actionability and oracle audit."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
V1_EXP = REPO / "experiments" / "persist_eeg_pc_refine_v1_seed0"
V1_CODE = V1_EXP / "code" / "run.py"
V2_EXP = REPO / "experiments" / "persist_eeg_native_cp_transfer_gate_v2_seed0"
V2_CODE = V2_EXP / "code" / "run.py"
V25_EXP = REPO / "experiments" / "persist_eeg_cp_direction_utility_v1_seed0"
V25_CODE = V25_EXP / "code" / "run.py"
V3_EXP = REPO / "experiments" / "persist_eeg_selective_cp_routing_v3_seed0"
V3_CODE = V3_EXP / "code" / "run.py"
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("CP_ACTIONABILITY_RUNTIME", str(REPO.parent / "cp_actionability_oracle_v1_runtime"))).resolve()
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
ALPHAS = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), dtype=np.float32)
BOOTSTRAPS = 20_000
TOP_K = 3
N_RANDOM_ALPHA = 20
LOW_ORACLE1 = 0.005
LOW_ORACLE3 = 0.01
CURVE_BATCH = int(os.environ.get("CP_ACTIONABILITY_CURVE_BATCH", "64"))
DEVICE = None


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


V3 = import_file("cp_actionability_v3", V3_CODE)
B = V3.B
DEVICE = V3.DEVICE
torch.set_num_threads(min(int(os.environ.get("CP_ACTIONABILITY_CPU_THREADS", "8")), os.cpu_count() or 1))
torch.backends.cudnn.benchmark = False


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arr_sha(*values: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return V3.stable_seed("CP_ACTIONABILITY_ORACLE_V1", *parts)


def jwrite(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def csvwrite(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
    os.replace(temp, path)


def runtime_cell(task: str, fold: int) -> Path:
    return RUNTIME / "cells" / task.lower() / f"fold{fold}_seed0"


def output_cell(task: str, fold: int) -> Path:
    return OUT / "cells" / task.lower() / f"fold{fold}_seed0"


def subject_key(values) -> str:
    return hashlib.sha256("|".join(sorted(map(str, values))).encode()).hexdigest()


def ordered_subjects(values) -> list[str]:
    return sorted(set(map(str, values)), key=lambda value: int(value.replace("sub-", "")))


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS, dtype=np.float32)
    for start in range(0, BOOTSTRAPS, 512):
        stop = min(BOOTSTRAPS, start + 512)
        index = rng.integers(0, len(x), size=(stop - start, len(x)))
        draws[start:stop] = x[index].mean(axis=1)
    return float(x.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def subject_summary(values: np.ndarray, seed: int) -> dict:
    x = np.asarray(values, dtype=np.float64)
    mean, lo, hi = bootstrap_mean_ci(x, seed)
    return {"mean": mean, "ci95_low": lo, "ci95_high": hi,
            "median": float(np.median(x)) if len(x) else math.nan,
            "positive_subject_fraction": float(np.mean(x > 0)) if len(x) else math.nan,
            "biological_subjects": int(len(x))}


def train_utility_for_cell(task: str, fold: int, v3_record: dict, role: dict, geometry: dict) -> dict:
    train_ids = ordered_subjects(role["inner_train_subjects"])
    discovery_ids = ordered_subjects(role["inner_val_subjects"])
    outer_ids = ordered_subjects(role["outer_dev_subjects"])
    all_ids = ordered_subjects(train_ids + discovery_ids + outer_ids)
    subject_matrix = np.asarray(geometry["train_subject_utility"], dtype=np.float64)
    if subject_matrix.shape[0] != len(all_ids):
        raise RuntimeError(f"V3 subject-utility ordering/count mismatch: {task}/{fold}")
    ix = [all_ids.index(subject) for subject in train_ids]
    matrix = subject_matrix[ix]
    means = matrix.mean(axis=0)
    seed = stable_seed("TRAIN_UTILITY_BOOTSTRAP", task, fold, subject_key(train_ids))
    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAPS, matrix.shape[1]), dtype=np.float32)
    for start in range(0, BOOTSTRAPS, 256):
        stop = min(BOOTSTRAPS, start + 256)
        rows = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        draws[start:stop] = matrix[rows].mean(axis=1)
    lower = np.quantile(draws, 0.025, axis=0)
    upper = np.quantile(draws, 0.975, axis=0)
    rank = sorted(range(len(means)), key=lambda j: (-abs(float(means[j])), j))
    k_top = rank[:min(TOP_K, len(rank))]
    return {"train_subjects": train_ids, "discovery_subjects": discovery_ids,
            "outer_dev_subject_count_excluded": len(outer_ids), "all_refit_subject_order": all_ids,
            "train_subject_utility_mean": means.tolist(), "train_subject_utility_ci_low": lower.tolist(),
            "train_subject_utility_ci_high": upper.tolist(), "train_utility_bootstrap_seed": seed,
            "top3_directions_abs_utility_rank": k_top,
            "top3_ranking_rule": "descending absolute inner_train subject-equal U_P mean, ties by fold-local direction index"}


def preflight() -> None:
    if (PROTOCOL / "PROTOCOL_LOCK.json").exists():
        raise RuntimeError("protocol is already locked")
    v3_lock = V3.verify_protocol()
    v3_lock_path = V3_EXP / "protocol" / "V3_PROTOCOL_LOCK.json"
    v25_lock_path = V25_EXP / "protocol" / "PROTOCOL_LOCK.json"
    v25_sum = V25_EXP / "outputs" / "DIRECTION_UTILITY_SUMMARY.csv"
    v25_lock = json.loads(v25_lock_path.read_text(encoding="utf-8"))
    if sha(V25_CODE) != v25_lock.get("code_sha256"):
        raise RuntimeError("V2.5 utility implementation differs from its frozen lock")
    if not v25_sum.is_file():
        raise RuntimeError("V2.5 direction utility summary is missing")
    with v25_sum.open(newline="", encoding="utf-8") as stream:
        v25_rows = list(csv.DictReader(stream))
    v25_cell_keys = {(r["task"], int(r["fold"])) for r in v25_rows}
    preflight_cells = []
    for task in TASKS:
        for fold in FOLDS:
            role, split, cache_name, source_sessions, future_session = B.role(task, fold)
            train = ordered_subjects(role["inner_train_subjects"])
            discovery = ordered_subjects(role["inner_val_subjects"])
            outer = ordered_subjects(role["outer_dev_subjects"])
            if set(train) & set(discovery) or set(train) & set(outer) or set(discovery) & set(outer):
                raise RuntimeError(f"split role overlap: {task}/{fold}")
            if (task, fold) not in v25_cell_keys:
                raise RuntimeError(f"V2.5 utility row missing: {task}/{fold}")
            v3_cell = next(c for c in v3_lock["baseline"]["cells"]
                           if c["task"] == task and int(c["fold"]) == fold)
            geometry_path = V3.output_cell(task, fold) / "geometry.npz"
            geometry = np.load(geometry_path, allow_pickle=False)
            ck_record, checkpoint, _ = V3.baseline_record(task, fold)
            if sha(geometry_path) != v3_cell["geometry_file_sha256"]:
                raise RuntimeError(f"V3 geometry hash mismatch: {task}/{fold}")
            if sha(checkpoint) != v3_cell["baseline_checkpoint_sha256"]:
                raise RuntimeError(f"V3 baseline checkpoint hash mismatch: {task}/{fold}")
            if int(geometry["c_basis"].shape[1]) != int(v3_cell["c_basis_rank"]):
                raise RuntimeError(f"V3 C rank mismatch: {task}/{fold}")
            train_record = train_utility_for_cell(task, fold, v3_cell, role, geometry)
            random0 = np.asarray(geometry["random_bases"][0], dtype=np.float32)
            random_hash = arr_sha(random0)
            locked_random = v3_cell["random_bases"][0]
            if random_hash != locked_random["basis_sha256"]:
                raise RuntimeError(f"prelocked random complement basis hash mismatch: {task}/{fold}")
            mu, sd = geometry["normalizer_mu"], geometry["normalizer_sd"]
            norm_hash = hashlib.sha256(np.ascontiguousarray(mu).tobytes() + np.ascontiguousarray(sd).tobytes()).hexdigest()
            if norm_hash != v3_cell["normalizer_sha256"]:
                raise RuntimeError(f"V3 frozen normalizer hash mismatch: {task}/{fold}")
            q_s = np.asarray(geometry["qs"]); q_d = np.asarray(geometry["qd"]); basis = np.asarray(geometry["c_basis"])
            if np.max(np.abs(q_s.T @ q_s - np.eye(q_s.shape[1]))) >= 1e-5 or np.max(np.abs(q_d.T @ q_d - np.eye(q_d.shape[1]))) >= 1e-5:
                raise RuntimeError(f"V3 projector orthogonality failed: {task}/{fold}")
            if np.max(np.abs(q_s.T @ basis)) >= 1e-5 or np.max(np.abs(q_s.T @ random0)) >= 1e-5:
                raise RuntimeError(f"V3/random basis leaves the spatial complement: {task}/{fold}")
            preflight_cells.append({"task": task, "fold": fold, "split_sha256": split,
                "cache_name": cache_name, "train_subjects": train, "train_subjects_sha256": subject_key(train),
                "discovery_subjects": discovery, "discovery_subjects_sha256": subject_key(discovery),
                "outer_dev_subject_count_excluded": len(outer),
                "sessions": sorted(set(map(int, tuple(source_sessions) + (int(future_session),)))),
                "geometry_sha256": sha(geometry_path), "checkpoint_sha256": sha(checkpoint),
                "normalizer_sha256": norm_hash, "random_basis0_sha256": random_hash,
                "spatial_rank": int(q_s.shape[1]), "successor_rank": int(q_d.shape[1]),
                "C_rank": int(basis.shape[1]), **train_record})
            geometry.close()
    record = {"status": "PASS", "source_commit": source_commit(), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata_only": True, "current_experiment_eeg_array_reads": 0, "final_heldout_array_reads": 0,
        "outer_dev_array_reads": 0, "curve_batch_size": CURVE_BATCH, "cells": preflight_cells,
        "v3_protocol_lock_sha256": sha(v3_lock_path), "v3_code_sha256": sha(V3_CODE),
        "v25_protocol_lock_sha256": sha(v25_lock_path), "v25_code_sha256": sha(V25_CODE),
        "v25_direction_utility_summary_sha256": sha(v25_sum)}
    RUNTIME.mkdir(parents=True, exist_ok=True)
    jwrite(RUNTIME / "PREFLIGHT.json", record)
    print("PREFLIGHT_PASS", len(preflight_cells), "cells; current EEG reads=0; final-heldout reads=0", flush=True)


def source_commit() -> str:
    if os.environ.get("CP_ACTIONABILITY_SOURCE_COMMIT"):
        return os.environ["CP_ACTIONABILITY_SOURCE_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def lock_protocol() -> None:
    path = PROTOCOL / "PROTOCOL_LOCK.json"
    if path.exists():
        raise RuntimeError("protocol lock already exists")
    pre = json.loads((RUNTIME / "PREFLIGHT.json").read_text(encoding="utf-8"))
    if pre.get("status") != "PASS" or pre.get("current_experiment_eeg_array_reads") != 0:
        raise RuntimeError("successful metadata-only preflight required")
    code_files = {str(p.relative_to(REPO)).replace("\\", "/"): sha(p)
                  for p in sorted((EXP / "code").glob("*.py"))}
    dependency_paths = [V1_CODE, V2_CODE, V25_CODE, V3_CODE,
        REPO / "experiments/persist_eeg_seven_backbone_fourtask_3seed_v1/code/backbone_models.py",
        REPO / "experiments/persist_eeg_seven_backbone_fourtask_3seed_v1/code/tech_recipe_selection.py",
        REPO / "experiments/persist_eeg_crossbackbone_peeh_v1/code/run_crossbackbone_peeh.py",
        V3_EXP / "protocol/V3_PROTOCOL_LOCK.json", V25_EXP / "protocol/PROTOCOL_LOCK.json",
        V25_EXP / "outputs/DIRECTION_UTILITY_SUMMARY.csv"]
    dependencies = {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in dependency_paths}
    record = {"schema": "PERSIST_EEG_CP_ACTIONABILITY_ORACLE_V1_SEED0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": source_commit(), "tasks": list(TASKS), "folds": list(FOLDS), "seed": 0,
        "backbone": "canonical frozen V3 EEGNet final-refit checkpoint", "new_backbone_training": False,
        "new_EEGNet_classifier_training": False, "all_EEGNet_parameters_bn_buffers_and_projectors_frozen": True,
        "conditional_diagnostic_router_training": "allowed only after development oracle BA gain >= 0.01",
        "new_trainable_parameters_in_backbone": 0,
        "FINAL_HELDOUT_ACCESSED": False, "final_heldout_array_reads": 0,
        "outer_dev_array_reads": 0, "analysis_population": "inner_train subjects plus inner_val discovery subjects only",
        "outer_dev_eeg_arrays_loaded": False, "outer_dev_subject_metadata_read_for_exclusion": True,
        "final_heldout_ids_or_arrays_loaded": False,
        "v3_protocol_lock_sha256": pre["v3_protocol_lock_sha256"],
        "v25_protocol_lock_sha256": pre["v25_protocol_lock_sha256"],
        "v25_direction_utility_summary_sha256": pre["v25_direction_utility_summary_sha256"],
        "reused_cells": pre["cells"], "runtime": {"curve_batch_size": int(pre["curve_batch_size"])},
        "code_files": code_files, "dependency_hashes": dependencies,
        "intervention": {"source_layer": "spatial_elu_pool1", "successor_layer": "depth_point_elu_pool2",
            "alpha_grid": ALPHAS.tolist(), "alpha_1_is_native_identity": True,
            "formula": "h_s(alpha)=mu_s+p_s+C_residual+sum_{k!=j} a_k c_k+alpha a_j c_j",
            "successor_route": "y_tilde=mu_d+Q_d Q_d^T(y_alpha-mu_d)+c_d_native",
            "downstream_C": "full native successor complement preserved for every intervention",
            "identity_tolerance_max_abs": 1e-5, "prediction_identity_required": True,
            "margin": "true_class logit minus largest non-true logit", "CE": "cross entropy on true label",
            "native_predicted_probability": "probability assigned to native baseline argmax under each alpha",
            "P_movement": "absolute L2 norm from the native successor P coordinates at alpha=1",
            "local_derivative": "(metric(alpha=1.25)-metric(alpha=0.75))/0.5",
            "argmax_tie": "scores within 1e-8 tie; choose alpha closest to 1, then smaller alpha; direction ties use lower fold-local index"},
        "oracles": {"one_direction": "per trial choose best direction and alpha by true-class margin; all others native",
            "top3": "visit the three prelocked inner_train utility-ranked directions in descending rank order; at each step choose a trial-specific alpha by true-class margin conditional on earlier steps; at most one change per direction",
            "top3_k": TOP_K, "utility_rank": "descending absolute V3 U_P mean computed only from inner_train subject rows; ties by direction index",
            "independent_direction_upper_bound": "choose each direction alpha independently from its one-direction response curve, then apply all selected alphas jointly; LOOSE_UPPER_BOUND_NOT_DEPLOYABLE",
            "random_direction_oracle": "one-direction oracle over the fold's V3-prelocked random complement basis draw 0; same rank as C basis",
            "random_alpha_control": {"draws": N_RANDOM_ALPHA, "rule": "per trial draw one structured C direction and one alpha uniformly; no outcome-dependent selection"}},
        "statistics": {"pooling": "trial metrics within subject-session; average sessions within biological subject; average folds within subject; equal biological-subject weighting",
            "bootstrap_unit": "biological subject", "bootstrap_draws": BOOTSTRAPS,
            "router_trigger": "run only if discovery ORACLE_1DIR or ORACLE_TOP3 BA gain is at least 0.01",
            "low_headroom": {"oracle1_ba_gain_below": LOW_ORACLE1, "oracle3_ba_gain_below": LOW_ORACLE3},
            "moderate_top3_ba_gain": [0.01, 0.02], "strong_top3_ba_gain_above": 0.02,
            "peak_near_native": "group mean margin argmax in {0.75,1,1.25} and paired subject-bootstrap lower CI exceeds both alpha=0 and alpha=2",
            "paired_bootstrap_delta": "resample matched biological subjects for both variants; 20,000 draws",
            "direction_trial_conditional": "both P(alpha*<1) and P(alpha*>1) are at least 0.10 for a direction",
            "local_slope_zero_tolerance": 0.001, "monotonicity_difference_tolerance": 0.0001,
            "flat_margin_range_tolerance": 0.001,
            "router_task_gate": "for each task trigger if pooled discovery ORACLE_1DIR or ORACLE_TOP3 BA gain is at least 0.01",
            "router_headroom_recovery_denominator": "the larger of the same task's discovery ORACLE_1DIR and ORACLE_TOP3 BA gains over baseline",
            "NATIVE_STRENGTH_NEAR_OPTIMAL": "all four tasks satisfy LOW_ACTIONABILITY_HEADROOM and more than half of utility-positive trial optima lie in alpha {0.75,1,1.25}",
            "UTILITY_NONMONOTONIC": "at least half of utility-positive subject-session-direction rows have local margin slope <=0, or at least half of direction profiles are PEAK_NEAR_NATIVE or NONMONOTONIC",
            "TRIAL_CONDITIONAL_ACTIONABILITY": "at least one task passes the 1 pp oracle gate and at least half of utility-positive direction profiles meet the locked two-sided trial-conditional rule"},
        "router": {"feature_inputs": ["native p_s coordinates", "native C coefficients", "native p_d coordinates",
                "native logits", "native confidence", "native entropy", "native P/C and transfer norm descriptors"],
            "prohibited_inputs": ["true label", "oracle margin", "future correctness", "future label"],
            "fit_population": "inner_train subjects", "evaluation_population": "inner_val discovery subjects, subject-disjoint",
            "target": "alpha class for the prelocked top-1 inner_train utility direction",
            "classes": {"SUPPRESS": "alpha* <= 0.5", "KEEP": "0.75 <= alpha* <= 1.25", "ENHANCE": "alpha* >= 1.5"},
            "applied_alpha": {"SUPPRESS": 0.5, "KEEP": 1.0, "ENHANCE": 1.5},
            "model": "StandardScaler plus multinomial LogisticRegression C=1, class_weight=balanced, max_iter=500",
            "training_validation": "5-fold GroupKFold by biological subject inside inner_train; final fit on all inner_train"}}
    jwrite(path, record)
    (PROTOCOL / "PROTOCOL_LOCK.sha256").write_text(sha(path) + "\n", encoding="utf-8")
    audit = {"FINAL_HELDOUT_ACCESSED": False, "current_experiment_eeg_array_reads": 0, "final_heldout_array_reads": 0,
        "OpenBMI_final_heldout_array_reads": 0, "WBCIC_true_outer_array_reads": 0,
        "outer_dev_array_reads": 0, "outer_dev_eeg_arrays_loaded": False,
        "outer_dev_subject_count_in_splits_but_not_loaded": sum(c["outer_dev_subject_count_excluded"] for c in pre["cells"]),
        "inner_train_and_inner_val_only": True, "all_B_rows_calls_final_false": True,
        "final_heldout_ids_or_arrays_loaded": False, "V3_heldout_predictions_reused": False,
        "V3_geometry_and_checkpoint_reused": True, "policy": "development-only audit; no final loader or heldout arrays called"}
    jwrite(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", audit)
    print("PROTOCOL_LOCKED", sha(path), "FINAL_HELDOUT_ACCESSED=False", flush=True)


def verify_protocol() -> dict:
    path = PROTOCOL / "PROTOCOL_LOCK.json"
    digest = PROTOCOL / "PROTOCOL_LOCK.sha256"
    if not path.is_file() or not digest.is_file() or sha(path) != digest.read_text(encoding="utf-8").strip():
        raise RuntimeError("protocol lock missing or hash mismatch")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("FINAL_HELDOUT_ACCESSED") is not False or rec.get("final_heldout_array_reads") != 0:
        raise RuntimeError("heldout policy mismatch")
    if int(rec.get("runtime", {}).get("curve_batch_size", -1)) != CURVE_BATCH:
        raise RuntimeError("curve batch size changed after protocol lock")
    for rel, h in rec["code_files"].items():
        if sha(REPO / rel) != h:
            raise RuntimeError(f"experiment source changed after lock: {rel}")
    for rel, h in rec["dependency_hashes"].items():
        if sha(REPO / rel) != h:
            raise RuntimeError(f"locked dependency changed: {rel}")
    V3.verify_protocol()
    for c in rec["reused_cells"]:
        p = V3.output_cell(c["task"], int(c["fold"])) / "geometry.npz"
        if sha(p) != c["geometry_sha256"]:
            raise RuntimeError(f"locked V3 geometry changed: {c['task']}/{c['fold']}")
        _, ck, _ = V3.baseline_record(c["task"], int(c["fold"]))
        if sha(ck) != c["checkpoint_sha256"]:
            raise RuntimeError(f"locked baseline changed: {c['task']}/{c['fold']}")
    return rec


def pick_index(scores: np.ndarray, maximize: bool = True) -> np.ndarray:
    """Stable alpha argmax/argmin with the prelocked native-strength tie rule."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    result = np.empty(len(x), dtype=np.int64)
    for i, row in enumerate(x):
        best = float(np.max(row) if maximize else np.min(row))
        tied = np.flatnonzero(np.abs(row - best) <= 1e-8)
        result[i] = min(tied, key=lambda j: (abs(float(ALPHAS[j]) - 1.0), float(ALPHAS[j])))
    return result


def pick_joint_index(scores: np.ndarray, directions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64)
    n = len(x)
    chosen_d = np.empty(n, dtype=np.int64); chosen_a = np.empty(n, dtype=np.int64)
    for i in range(n):
        row = x[i].reshape(-1)
        best = float(np.max(row))
        tied = np.flatnonzero(np.abs(row - best) <= 1e-8)
        ix = min(tied, key=lambda flat: (abs(float(ALPHAS[flat % len(ALPHAS)]) - 1.0),
            float(ALPHAS[flat % len(ALPHAS)]), int(directions[flat // len(ALPHAS)])))
        chosen_d[i] = int(directions[ix // len(ALPHAS)])
        chosen_a[i] = int(ix % len(ALPHAS))
    return chosen_d, chosen_a


def torch_margins(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], -torch.inf)
    return true - masked.max(dim=1).values


def routed_logits_and_p(model, hs: torch.Tensor, qd: torch.Tensor, md: torch.Tensor,
                        native_c: torch.Tensor, samples: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    y = V3.f0(model, hs, samples).float()
    p = (y - md) @ qd
    hd = md + p @ qd.T + native_c
    centered = hd - md
    cnow = centered - (centered @ qd) @ qd.T
    error = float(torch.max(torch.abs(cnow - native_c)).item()) if cnow.numel() else 0.0
    return V3.suffix(model, hd).float(), p, error


def routed_logits(model, hs: torch.Tensor, qd: torch.Tensor, md: torch.Tensor,
                  native_c: torch.Tensor, samples: int) -> tuple[torch.Tensor, float]:
    z, _, error = routed_logits_and_p(model, hs, qd, md, native_c, samples)
    return z, error


def evaluate_curve_bank(model, hs: np.ndarray, labels: np.ndarray, base: dict,
                        qs: np.ndarray, ms: np.ndarray, qd: np.ndarray, md: np.ndarray,
                        directions: np.ndarray, samples: int, batch_size: int,
                        capture_direction: int | None = None):
    n, k = len(hs), directions.shape[1]
    names = ("margin", "CE", "prediction", "max_softmax", "native_predicted_probability", "entropy", "p_movement")
    result = {name: np.empty((n, k, len(ALPHAS)), dtype=np.float32 if name != "prediction" else np.int16) for name in names}
    captured_logits = (np.empty((n, len(ALPHAS), int(base["logits"].shape[1])), dtype=np.float32)
                       if capture_direction is not None else None)
    if capture_direction is not None and not 0 <= capture_direction < k:
        raise ValueError("capture_direction is outside the response-curve bank")
    qst = torch.as_tensor(qs, dtype=torch.float32, device=DEVICE)
    mst = torch.as_tensor(ms, dtype=torch.float32, device=DEVICE)
    qdt = torch.as_tensor(qd, dtype=torch.float32, device=DEVICE)
    mdt = torch.as_tensor(md, dtype=torch.float32, device=DEVICE)
    basis_t = torch.as_tensor(directions, dtype=torch.float32, device=DEVICE)
    ut = basis_t.T
    alpha_t = torch.as_tensor(ALPHAS, dtype=torch.float32, device=DEVICE)
    identity_error = 0.0; preservation_error = 0.0
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size); b = stop - start
        h = torch.from_numpy(np.ascontiguousarray(hs[start:stop])).to(DEVICE)
        y0 = torch.from_numpy(np.ascontiguousarray(base["representation"][start:stop])).to(DEVICE)
        z0 = torch.from_numpy(np.ascontiguousarray(base["logits"][start:stop])).to(DEVICE)
        p0 = torch.from_numpy(np.ascontiguousarray(base["protected"][start:stop])).to(DEVICE)
        c0 = torch.from_numpy(np.ascontiguousarray(base["complement"][start:stop])).to(DEVICE)
        y = torch.as_tensor(labels[start:stop], dtype=torch.long, device=DEVICE)
        centered = h - mst
        ps = (centered @ qst) @ qst.T
        hc = centered - ps
        coeff = hc @ basis_t
        scaled = h[:, None, None, :] + (alpha_t[None, None, :, None] - 1.0) * coeff[:, :, None, None] * ut[None, :, None, :]
        zflat, pflat, preserve = routed_logits_and_p(model, scaled.reshape(-1, h.shape[1]), qdt, mdt,
            c0[:, None, None, :].expand(-1, k, len(ALPHAS), -1).reshape(-1, c0.shape[1]), samples)
        preservation_error = max(preservation_error, preserve)
        z = zflat.reshape(b, k, len(ALPHAS), -1)
        p_scaled = pflat.reshape(b, k, len(ALPHAS), -1)
        yrep = y[:, None, None].expand(-1, k, len(ALPHAS)).reshape(-1)
        margin = torch_margins(z.reshape(-1, z.shape[-1]), yrep).reshape(b, k, len(ALPHAS))
        ce = F.cross_entropy(z.reshape(-1, z.shape[-1]), yrep, reduction="none").reshape(b, k, len(ALPHAS))
        prob = z.softmax(dim=-1)
        pred = prob.argmax(dim=-1)
        native_pred = z0.argmax(dim=-1)
        native_prob = prob.gather(-1, native_pred[:, None, None, None].expand(-1, k, len(ALPHAS), 1)).squeeze(-1)
        max_prob = prob.max(dim=-1).values
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        # Successor P coordinates are measured relative to the native alpha=1 coordinates.
        pmove = torch.linalg.vector_norm(p_scaled - p0[:, None, None, :], dim=-1)
        native_curve = z[:, :, int(np.where(ALPHAS == 1.0)[0][0]), :]
        identity_error = max(identity_error, float(torch.max(torch.abs(native_curve - z0[:, None, :])).item()))
        native_pred = z0.argmax(dim=-1)[:, None].expand(-1, k)
        if not torch.equal(native_curve.argmax(dim=-1), native_pred):
            raise RuntimeError("alpha=1 routed predictions differ from the native EEGNet baseline")
        result["margin"][start:stop] = margin.cpu().numpy()
        result["CE"][start:stop] = ce.cpu().numpy()
        result["prediction"][start:stop] = pred.to(torch.int16).cpu().numpy()
        result["max_softmax"][start:stop] = max_prob.cpu().numpy()
        result["native_predicted_probability"][start:stop] = native_prob.cpu().numpy()
        result["entropy"][start:stop] = entropy.cpu().numpy()
        result["p_movement"][start:stop] = pmove.cpu().numpy()
        if captured_logits is not None:
            captured_logits[start:stop] = z[:, capture_direction].cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if identity_error >= 1e-5:
        raise RuntimeError(f"alpha=1 identity routing failed: {identity_error}")
    return result, identity_error, preservation_error, captured_logits


def evaluate_static_v3(model, hs: np.ndarray, base: dict, qs: np.ndarray, ms: np.ndarray,
                       qd: np.ndarray, md: np.ndarray, basis: np.ndarray,
                       coefficients: np.ndarray, samples: int, batch_size: int = 256):
    outputs = []
    qst = torch.as_tensor(qs, dtype=torch.float32, device=DEVICE)
    mst = torch.as_tensor(ms, dtype=torch.float32, device=DEVICE)
    qdt = torch.as_tensor(qd, dtype=torch.float32, device=DEVICE)
    mdt = torch.as_tensor(md, dtype=torch.float32, device=DEVICE)
    ut = torch.as_tensor(basis, dtype=torch.float32, device=DEVICE)
    rt = torch.as_tensor(coefficients, dtype=torch.float32, device=DEVICE)
    max_preserve = 0.0
    with torch.inference_mode():
        for start in range(0, len(hs), batch_size):
            stop = min(len(hs), start + batch_size)
            h = torch.from_numpy(np.ascontiguousarray(hs[start:stop])).to(DEVICE)
            centered = h - mst; ps = (centered @ qst) @ qst.T; hc = centered - ps
            a = hc @ ut
            routed_h = mst + ps + (hc + (a * (rt - 1.0)) @ ut.T)
            native_c = torch.from_numpy(np.ascontiguousarray(base["complement"][start:stop])).to(DEVICE)
            z, err = routed_logits(model, routed_h, qdt, mdt, native_c, samples)
            max_preserve = max(max_preserve, err)
            outputs.append(z.cpu().numpy())
    return np.concatenate(outputs), max_preserve


def sequential_top3(model, h: torch.Tensor, coeff: torch.Tensor, basis: torch.Tensor,
                    qd: torch.Tensor, md: torch.Tensor, native_c: torch.Tensor,
                    labels: torch.Tensor, directions: list[int], samples: int):
    current = h.clone()
    alpha_t = torch.as_tensor(ALPHAS, dtype=torch.float32, device=DEVICE)
    for direction in directions:
        variants = current[:, None, :] + (alpha_t[None, :, None] - 1.0) * coeff[:, direction, None, None] * basis[:, direction][None, None, :]
        z, _ = routed_logits(model, variants.reshape(-1, h.shape[1]), qd, md,
            native_c[:, None, :].expand(-1, len(ALPHAS), -1).reshape(-1, native_c.shape[1]), samples)
        z = z.reshape(len(h), len(ALPHAS), -1)
        margin = torch_margins(z.reshape(-1, z.shape[-1]), labels[:, None].expand(-1, len(ALPHAS)).reshape(-1)).reshape(len(h), len(ALPHAS))
        alpha_index = pick_index(margin.detach().cpu().numpy(), maximize=True)
        chosen = alpha_t[torch.as_tensor(alpha_index, dtype=torch.long, device=DEVICE)]
        current = current + (chosen[:, None] - 1.0) * coeff[:, direction, None] * basis[:, direction][None, :]
    z_final, preserve_error = routed_logits(model, current, qd, md, native_c, samples)
    ce = F.cross_entropy(z_final, labels, reduction="none")
    return z_final, ce, preserve_error


def metric_row(task, fold, role_name, session, subject, variant, labels, pred, ce):
    y = np.asarray(labels, dtype=np.int64); p = np.asarray(pred, dtype=np.int64); loss = np.asarray(ce, dtype=np.float64)
    base = metric_row._baseline[(task, fold, role_name, int(session), str(subject))]
    rescued = int(np.sum((base != y) & (p == y)))
    damaged = int(np.sum((base == y) & (p != y)))
    return {"task": task, "fold": int(fold), "role": role_name, "session": int(session),
        "subject": str(subject), "variant": variant,
        "BA": float(balanced_accuracy_score(y, p)), "macro_F1": float(f1_score(y, p, average="macro", zero_division=0)),
        "NLL": float(loss.mean()), "rescued_errors": rescued, "damaged_correct": damaged,
        "trials": int(len(y)), "label": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}


def performance_row(task, fold, role_name, session, subject, variant, labels, pred, ce, baseline_pred):
    y = np.asarray(labels, dtype=np.int64); p = np.asarray(pred, dtype=np.int64)
    loss = np.asarray(ce, dtype=np.float64); base = np.asarray(baseline_pred, dtype=np.int64)
    oracle_status = ("LOOSE_UPPER_BOUND_NOT_DEPLOYABLE" if variant == "ORACLE_UPPER" else
        "LABEL_CONDITIONAL_ORACLE_NOT_DEPLOYABLE" if variant in ("ORACLE_1DIR", "CE_ORACLE_1DIR", "ORACLE_TOP3", "RANDOM_DIRECTION_ORACLE") else
        "RANDOMIZED_LABEL_FREE_CONTROL" if variant.startswith("RANDOM_ALPHA_DRAW_") else "FROZEN_REFERENCE_OR_ROUTER")
    return {"task": task, "fold": int(fold), "role": role_name, "session": int(session),
        "subject": str(subject), "variant": variant,
        "BA": float(balanced_accuracy_score(y, p)),
        "macro_F1": float(f1_score(y, p, average="macro", zero_division=0)),
        "NLL": float(loss.mean()), "rescued_errors": int(np.sum((base != y) & (p == y))),
        "damaged_correct": int(np.sum((base == y) & (p != y))), "trials": int(len(y)),
        "oracle_status": oracle_status,
        "label": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}


def alpha_tag(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def curve_fields() -> list[str]:
    fields = ["task", "fold", "role", "session", "subject", "trial_index", "label", "direction"]
    for metric in ("margin", "CE", "prediction", "max_softmax", "native_predicted_probability", "entropy", "p_movement"):
        fields.extend(f"{metric}_a{alpha_tag(a)}" for a in ALPHAS)
    fields.append("label_scope")
    return fields


def write_trial_curves(writer: csv.DictWriter, task: str, fold: int, role_name: str,
                       session: int, subject: str, labels: np.ndarray, curves: dict) -> None:
    n, k, _ = curves["margin"].shape
    for trial in range(n):
        for direction in range(k):
            row = {"task": task, "fold": fold, "role": role_name, "session": session,
                   "subject": subject, "trial_index": trial, "label": int(labels[trial]),
                   "direction": direction, "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}
            for metric in ("margin", "CE", "prediction", "max_softmax", "native_predicted_probability", "entropy", "p_movement"):
                for aidx, alpha in enumerate(ALPHAS):
                    row[f"{metric}_a{alpha_tag(alpha)}"] = float(curves[metric][trial, direction, aidx])
            writer.writerow(row)


def router_features(hs: np.ndarray, base: dict, qs: np.ndarray, ms: np.ndarray,
                    basis: np.ndarray) -> np.ndarray:
    centered = hs - ms[None, :]
    ps = centered @ qs
    hc = centered - ps @ qs.T
    coeff = hc @ basis
    pd = base["protected"]
    logits = base["logits"]
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)
    sorted_logits = np.sort(logits, axis=1)
    top_gap = sorted_logits[:, -1] - (sorted_logits[:, -2] if logits.shape[1] > 1 else 0.0)
    conf = prob.max(axis=1)
    entropy = -(prob * np.log(np.maximum(prob, 1e-12))).sum(axis=1)
    pnorm = np.linalg.norm(ps, axis=1); cnorm = np.linalg.norm(hc, axis=1); dnorm = np.linalg.norm(pd, axis=1)
    scalar = np.stack((conf, entropy, top_gap, pnorm, cnorm, dnorm,
        pnorm / np.maximum(cnorm, 1e-8), dnorm / np.maximum(pnorm, 1e-8)), axis=1)
    # Labels, oracle outcomes, and future correctness are kept outside this matrix.
    return np.concatenate((ps, coeff, pd, logits, scalar), axis=1).astype(np.float32)


def _variant_arrays(curves: dict, labels: np.ndarray, choose_margin: bool = True):
    score = curves["margin"] if choose_margin else -curves["CE"]
    chosen_dir, chosen_alpha = pick_joint_index(score.reshape(len(labels), -1), list(range(score.shape[1])))
    rows = np.arange(len(labels))
    return chosen_dir, chosen_alpha, curves["prediction"][rows, chosen_dir, chosen_alpha], curves["CE"][rows, chosen_dir, chosen_alpha]


def _alpha_action(index: np.ndarray) -> np.ndarray:
    values = ALPHAS[np.asarray(index, dtype=np.int64)]
    action = np.full(len(values), -1, dtype=np.int64)
    action[values <= 0.5] = 0
    action[(values >= 0.75) & (values <= 1.25)] = 1
    action[values >= 1.5] = 2
    if np.any(action < 0):
        raise RuntimeError("locked alpha grid contains an unassigned router action")
    return action


def _normalized_entropy(counts: np.ndarray) -> float:
    p = np.asarray(counts, dtype=np.float64); p = p[p > 0]
    if not len(p): return 0.0
    return -float(np.sum(p * np.log(p))) / math.log(len(ALPHAS))


def _monotonicity_label(curve: np.ndarray, subject_curves: np.ndarray, seed: int):
    mean_curve = np.asarray(curve, dtype=np.float64); diffs = np.diff(mean_curve)
    peak_idx = int(np.argmax(mean_curve)); flat = float(np.ptp(mean_curve)) <= 1e-3
    peak_ci_low = math.nan; extreme2_ci_low = math.nan; peak_sig = False
    if peak_idx in (3, 4, 5) and not flat and len(subject_curves):
        _, peak_ci_low, _ = bootstrap_mean_ci(subject_curves[:, peak_idx] - subject_curves[:, 0], seed)
        _, extreme2_ci_low, _ = bootstrap_mean_ci(subject_curves[:, peak_idx] - subject_curves[:, -1], seed + 1)
        peak_sig = peak_ci_low > 0 and extreme2_ci_low > 0
    if flat: return "FLAT", peak_idx, False, peak_ci_low, extreme2_ci_low
    if peak_sig: return "PEAK_NEAR_NATIVE", peak_idx, True, peak_ci_low, extreme2_ci_low
    tol = 1e-4
    if np.all(diffs >= -tol) and np.any(diffs > tol): return "MONOTONIC_INCREASING", peak_idx, False, peak_ci_low, extreme2_ci_low
    if np.all(diffs <= tol) and np.any(diffs < -tol): return "MONOTONIC_DECREASING", peak_idx, False, peak_ci_low, extreme2_ci_low
    min_idx = int(np.argmin(mean_curve))
    if 0 < min_idx < len(mean_curve) - 1 and mean_curve[0] > mean_curve[min_idx] + tol and mean_curve[-1] > mean_curve[min_idx] + tol:
        return "U_SHAPED", peak_idx, False, peak_ci_low, extreme2_ci_low
    return "NONMONOTONIC", peak_idx, False, peak_ci_low, extreme2_ci_low


def _load_model(raw: np.ndarray, mapping: dict, checkpoint: Path):
    model = B.eegnet({"channels": int(raw.shape[1]), "samples": int(raw.shape[2]), "classes": int(len(mapping))})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    for param in model.parameters(): param.requires_grad_(False)
    if any(param.requires_grad for param in model.parameters()): raise RuntimeError("EEGNet is not frozen")
    if not np.isfinite(raw).all(): raise RuntimeError("nonfinite EEG in an allowed split")
    return model, int(raw.shape[2]), int(len(mapping))


def model_state_sha(model) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); arr = value.detach().cpu().contiguous().numpy()
        h.update(str(arr.shape).encode()); h.update(str(arr.dtype).encode()); h.update(arr.tobytes())
    return h.hexdigest()


def response_summary_rows(task: str, fold: int, role_name: str, session: int,
                          subject: str, labels: np.ndarray, curves: dict) -> list[dict]:
    rows = []
    for d in range(curves["margin"].shape[1]):
        for aidx, alpha in enumerate(ALPHAS):
            prediction = curves["prediction"][:, d, aidx]
            rows.append({"task": task, "fold": fold, "role": role_name, "session": session,
                "subject": subject, "direction": d, "alpha": float(alpha),
                "mean_true_class_margin": float(curves["margin"][:, d, aidx].mean()),
                "mean_CE": float(curves["CE"][:, d, aidx].mean()),
                "accuracy": float(np.mean(prediction == labels)),
                "mean_max_softmax": float(curves["max_softmax"][:, d, aidx].mean()),
                "mean_native_predicted_probability": float(curves["native_predicted_probability"][:, d, aidx].mean()),
                "mean_entropy": float(curves["entropy"][:, d, aidx].mean()),
                "mean_successor_P_movement": float(curves["p_movement"][:, d, aidx].mean()),
                "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return rows


def local_action_rows(task: str, fold: int, role_name: str, session: int,
                      subject: str, curves: dict, utility: np.ndarray) -> list[dict]:
    i075 = int(np.where(np.isclose(ALPHAS, 0.75))[0][0]); i125 = int(np.where(np.isclose(ALPHAS, 1.25))[0][0])
    rows = []
    for d in range(curves["margin"].shape[1]):
        dm = float((curves["margin"][:, d, i125].mean() - curves["margin"][:, d, i075].mean()) / 0.5)
        dc = float((curves["CE"][:, d, i125].mean() - curves["CE"][:, d, i075].mean()) / 0.5)
        u = float(utility[d])
        group = "A_USEFUL_AMPLIFY" if u > 0 and dm > 0.001 else (
            "B_USEFUL_PLATEAU" if u > 0 and abs(dm) <= 0.001 else (
            "C_USEFUL_AMPLIFY_HARM" if u > 0 and dm < -0.001 else "D_NONPOSITIVE_UTILITY"))
        rows.append({"task": task, "fold": fold, "role": role_name, "session": session,
            "subject": subject, "direction": d, "train_U_P": u, "D_margin": dm, "D_CE": dc,
            "local_actionability": "AMPLIFY" if dm > 0.001 else "SUPPRESS" if dm < -0.001 else "PLATEAU",
            "utility_action_quadrant": group, "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return rows


def utility_action_rows(task: str, fold: int, role_name: str, session: int,
                        subject: str, curves: dict, utility: np.ndarray) -> list[dict]:
    rows = []
    for d in range(curves["margin"].shape[1]):
        m_idx = pick_index(curves["margin"][:, d, :], maximize=True)
        ce_idx = pick_index(curves["CE"][:, d, :], maximize=False)
        low = float(np.mean(ALPHAS[m_idx] < 1.0)); high = float(np.mean(ALPHAS[m_idx] > 1.0))
        rows.append({"task": task, "fold": fold, "role": role_name, "session": session,
            "subject": subject, "direction": d, "train_U_P": float(utility[d]),
            "margin_alpha_star_le_1_fraction": float(np.mean(ALPHAS[m_idx] <= 1.0)),
            "margin_alpha_star_near_native_fraction": float(np.mean((ALPHAS[m_idx] >= 0.75) & (ALPHAS[m_idx] <= 1.25))),
            "margin_alpha_star_lt_1_fraction": low, "margin_alpha_star_gt_1_fraction": high,
            "direction_trial_conditional": bool(low >= 0.10 and high >= 0.10),
            "margin_alpha_star_distribution": json.dumps({str(float(a)): int(np.sum(ALPHAS[m_idx] == a)) for a in ALPHAS}, sort_keys=True),
            "ce_alpha_star_distribution": json.dumps({str(float(a)): int(np.sum(ALPHAS[ce_idx] == a)) for a in ALPHAS}, sort_keys=True),
            "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return rows


def conditionality_rows(task: str, fold: int, role_name: str, session: int,
                        subject: str, curves: dict) -> list[dict]:
    rows = []
    for d in range(curves["margin"].shape[1]):
        m_idx = pick_index(curves["margin"][:, d, :], maximize=True)
        ce_idx = pick_index(curves["CE"][:, d, :], maximize=False)
        counts = np.bincount(m_idx, minlength=len(ALPHAS)); ce_counts = np.bincount(ce_idx, minlength=len(ALPHAS))
        n = max(int(counts.sum()), 1); low = float(counts[:3].sum() / n)
        mid = float(counts[3:6].sum() / n); high = float(counts[6:].sum() / n)
        rows.append({"task": task, "fold": fold, "role": role_name, "session": session,
            "subject": subject, "direction": d, "margin_alpha_star_lt_0p75": low,
            "margin_alpha_star_0p75_to_1p25": mid, "margin_alpha_star_gt_1p25": high,
            "margin_alpha_star_below_1_fraction": float(counts[:4].sum() / n),
            "margin_alpha_star_above_1_fraction": float(counts[5:].sum() / n),
            "margin_conditional_entropy_normalized": _normalized_entropy(counts),
            "ce_conditional_entropy_normalized": _normalized_entropy(ce_counts),
            "both_sides_at_least_0p10": bool(counts[:4].sum() / n >= 0.10 and counts[5:].sum() / n >= 0.10),
            "margin_alpha_star_counts": json.dumps(counts.tolist()), "ce_alpha_star_counts": json.dumps(ce_counts.tolist()),
            "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return rows


def evaluate_cell(task: str, fold: int) -> None:
    lock = verify_protocol()
    if task not in TASKS or fold not in FOLDS:
        raise ValueError(f"unlocked task/fold: {task}/{fold}")
    odir = output_cell(task, fold); odir.mkdir(parents=True, exist_ok=True)
    done_path = odir / "CELL_COMPLETE.json"
    curve_path = odir / "RESPONSE_CURVES.csv.gz"
    metric_path = odir / "METRICS.csv"
    if done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if (done.get("protocol_sha256") == sha(PROTOCOL / "PROTOCOL_LOCK.json") and
                done.get("curve_sha256") == sha(curve_path) and done.get("metrics_sha256") == sha(metric_path)):
            print("CELL_ALREADY_COMPLETE", task, fold, flush=True); return
        raise RuntimeError(f"incomplete or changed cell outputs refuse overwrite: {task}/{fold}")

    _, checkpoint, _ = V3.baseline_record(task, fold)
    role, split, cache_name, source_sessions, future_session = B.role(task, fold)
    train_ids = ordered_subjects(role["inner_train_subjects"])
    discovery_ids = ordered_subjects(role["inner_val_subjects"])
    outer_ids = ordered_subjects(role["outer_dev_subjects"])
    if set(train_ids) & set(discovery_ids) or set(train_ids) & set(outer_ids) or set(discovery_ids) & set(outer_ids):
        raise RuntimeError("split roles overlap before EEG load")
    if set(train_ids) & set(discovery_ids):
        raise RuntimeError("router fit/evaluation subject overlap")

    locked_cell = next(c for c in lock["reused_cells"] if c["task"] == task and int(c["fold"]) == fold)
    train_utility = np.asarray(locked_cell["train_subject_utility_mean"], dtype=np.float64)
    top3 = list(map(int, locked_cell["top3_directions_abs_utility_rank"]))
    top1 = top3[0]
    geom = np.load(V3.output_cell(task, fold) / "geometry.npz", allow_pickle=False)
    qs, qd, ms, md = (np.asarray(geom[k], dtype=np.float32) for k in ("qs", "qd", "ms", "md"))
    basis = np.asarray(geom["c_basis"], dtype=np.float32)
    random_basis = np.asarray(geom["random_bases"][0], dtype=np.float32)
    r_primary = np.asarray(geom["r_primary"], dtype=np.float32)
    norm_mu = np.asarray(geom["normalizer_mu"], dtype=np.float32)
    norm_sd = np.asarray(geom["normalizer_sd"], dtype=np.float32)
    if basis.shape[1] < TOP_K or random_basis.shape != basis.shape or len(train_utility) != basis.shape[1]:
        raise RuntimeError(f"locked basis/utility dimensions disagree: {task}/{fold}")
    session_ids = sorted(set(tuple(source_sessions) + (int(future_session),)))
    results = {"metrics": [], "response_summary": [], "local": [], "utility": [], "conditionality": [], "monotonicity": []}
    profile_acc: dict[tuple[str, str, int], list[np.ndarray]] = {}
    router_acc = {name: {key: [] for key in ("X", "alpha", "y", "subject", "session", "logits")}
                  for name in ("train", "discovery")}
    mapping = None; model = None; samples = None; state_before = None
    max_identity = max_cpreserve = 0.0
    with gzip.open(curve_path, "wt", encoding="utf-8", newline="", compresslevel=6) as compressed:
        writer = csv.DictWriter(compressed, fieldnames=curve_fields(), extrasaction="ignore")
        writer.writeheader()
        for role_name, subjects in (("inner_train", train_ids), ("discovery", discovery_ids)):
            for subject in subjects:
                for session in session_ids:
                    # This is the only EEG loader used by the experiment. final=False is explicit.
                    raw, labels, owners, next_mapping = B.rows(task, [subject], (int(session),), cache_name,
                        mapping, final=False)
                    if mapping is None:
                        mapping = dict(next_mapping)
                    elif mapping != next_mapping:
                        raise RuntimeError(f"label mapping drift: {task}/{fold}/{subject}/S{session}")
                    if not len(raw) or not np.all(np.asarray(owners).astype(str) == str(subject)):
                        raise RuntimeError(f"cache slice subject mismatch: {task}/{fold}/{subject}/S{session}")
                    if model is None:
                        model, samples, classes = _load_model(raw, mapping, checkpoint)
                        if classes != len(mapping):
                            raise RuntimeError("frozen checkpoint and raw-label map class counts differ")
                        state_before = model_state_sha(model)
                    if int(raw.shape[2]) != samples:
                        raise RuntimeError(f"sample count changed in cache: {task}/{fold}")
                    x = ((raw - norm_mu[None, :, None]) / np.maximum(norm_sd[None, :, None], 1e-6)).astype(np.float32)
                    del raw
                    hs = V3.spatial(model, x, batch=128)
                    base = V3.full_forward(model, hs, qd, md, samples, batch=128)
                    curves, identity_err, c_preserve, captured = evaluate_curve_bank(model, hs, labels, base,
                        qs, ms, qd, md, basis, samples, CURVE_BATCH, capture_direction=top1)
                    random_curves, random_identity, random_preserve, _ = evaluate_curve_bank(model, hs, labels, base,
                        qs, ms, qd, md, random_basis, samples, CURVE_BATCH)
                    identity_err = max(identity_err, random_identity); c_preserve = max(c_preserve, random_preserve)
                    max_identity = max(max_identity, identity_err); max_cpreserve = max(max_cpreserve, c_preserve)
                    if identity_err >= 1e-5 or c_preserve >= 1e-5:
                        raise RuntimeError(f"curve identity/complement audit failed: {task}/{fold} {identity_err=} {c_preserve=}")
                    y = np.asarray(labels, dtype=np.int64); base_logits = base["logits"]
                    base_pred = base_logits.argmax(axis=1)
                    base_ce = F.cross_entropy(torch.as_tensor(base_logits), torch.as_tensor(y), reduction="none").numpy()
                    static_logits, static_preserve = evaluate_static_v3(model, hs, base, qs, ms, qd, md,
                        basis, r_primary, samples, batch_size=128)
                    max_cpreserve = max(max_cpreserve, static_preserve)
                    if static_preserve >= 1e-5: raise RuntimeError("V3 static route successor-C preservation failed")
                    static_pred = static_logits.argmax(axis=1)
                    static_ce = F.cross_entropy(torch.as_tensor(static_logits), torch.as_tensor(y), reduction="none").numpy()
                    _, _, one_pred, one_ce = _variant_arrays(curves, y, choose_margin=True)
                    _, _, ce_pred, ce_ce = _variant_arrays(curves, y, choose_margin=False)
                    _, _, rand_pred, rand_ce = _variant_arrays(random_curves, y, choose_margin=True)

                    h_t = torch.as_tensor(hs, dtype=torch.float32, device=DEVICE)
                    centered_t = h_t - torch.as_tensor(ms, dtype=torch.float32, device=DEVICE)
                    ps_t = (centered_t @ torch.as_tensor(qs, dtype=torch.float32, device=DEVICE)) @ torch.as_tensor(qs.T, dtype=torch.float32, device=DEVICE)
                    coeff_t = (centered_t - ps_t) @ torch.as_tensor(basis, dtype=torch.float32, device=DEVICE)
                    qd_t = torch.as_tensor(qd, dtype=torch.float32, device=DEVICE)
                    md_t = torch.as_tensor(md, dtype=torch.float32, device=DEVICE)
                    native_c = torch.as_tensor(base["complement"], dtype=torch.float32, device=DEVICE)
                    labels_t = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
                    z_top3, ce_top3, top3_preserve = sequential_top3(model, h_t, coeff_t,
                        torch.as_tensor(basis, dtype=torch.float32, device=DEVICE), qd_t, md_t, native_c,
                        labels_t, top3, samples)
                    max_cpreserve = max(max_cpreserve, top3_preserve)
                    if top3_preserve >= 1e-5: raise RuntimeError("top-three route successor-C preservation failed")
                    top3_logits = z_top3.detach().cpu().numpy(); top3_pred = top3_logits.argmax(axis=1)
                    top3_ce = ce_top3.detach().cpu().numpy()
                    alpha_each = pick_index(curves["margin"].reshape(-1, len(ALPHAS)), maximize=True).reshape(len(y), -1)
                    alpha_values = ALPHAS[alpha_each]
                    scaled_coeff = coeff_t * torch.as_tensor(alpha_values - 1.0, dtype=torch.float32, device=DEVICE)
                    upper_h = h_t + scaled_coeff @ torch.as_tensor(basis.T, dtype=torch.float32, device=DEVICE)
                    upper_logits, upper_preserve = routed_logits(model, upper_h, qd_t, md_t, native_c, samples)
                    max_cpreserve = max(max_cpreserve, upper_preserve)
                    if upper_preserve >= 1e-5: raise RuntimeError("independent-direction upper bound C preservation failed")
                    upper_logits_np = upper_logits.detach().cpu().numpy(); upper_pred = upper_logits_np.argmax(axis=1)
                    upper_ce = F.cross_entropy(upper_logits, labels_t, reduction="none").detach().cpu().numpy()
                    variants = [("BASELINE", base_pred, base_ce), ("STATIC_V3", static_pred, static_ce),
                        ("ORACLE_1DIR", one_pred, one_ce), ("CE_ORACLE_1DIR", ce_pred, ce_ce),
                        ("ORACLE_TOP3", top3_pred, top3_ce), ("ORACLE_UPPER", upper_pred, upper_ce),
                        ("RANDOM_DIRECTION_ORACLE", rand_pred, rand_ce)]
                    for variant, pred, loss in variants:
                        results["metrics"].append(performance_row(task, fold, role_name, session, subject,
                            variant, y, pred, loss, base_pred))
                    rng = np.random.default_rng(stable_seed("RANDOM_ALPHA", task, fold, role_name, subject, session))
                    trial_ix = np.arange(len(y))
                    for draw in range(N_RANDOM_ALPHA):
                        rd = rng.integers(0, basis.shape[1], size=len(y)); ra = rng.integers(0, len(ALPHAS), size=len(y))
                        rp = curves["prediction"][trial_ix, rd, ra]; rce = curves["CE"][trial_ix, rd, ra]
                        results["metrics"].append(performance_row(task, fold, role_name, session, subject,
                            f"RANDOM_ALPHA_DRAW_{draw:02d}", y, rp, rce, base_pred))
                    results["response_summary"].extend(response_summary_rows(task, fold, role_name, session, subject, y, curves))
                    results["local"].extend(local_action_rows(task, fold, role_name, session, subject, curves, train_utility))
                    results["utility"].extend(utility_action_rows(task, fold, role_name, session, subject, curves, train_utility))
                    results["conditionality"].extend(conditionality_rows(task, fold, role_name, session, subject, curves))
                    for direction in range(basis.shape[1]):
                        profile_acc.setdefault((role_name, str(subject), direction), []).append(
                            curves["margin"][:, direction, :].mean(axis=0))
                    write_trial_curves(writer, task, fold, role_name, session, subject, y, curves)
                    X = router_features(hs, base, qs, ms, basis)
                    oracle_alpha = pick_index(curves["margin"][:, top1, :], maximize=True)
                    bucket = router_acc["train" if role_name == "inner_train" else "discovery"]
                    bucket["X"].append(X); bucket["alpha"].append(oracle_alpha.astype(np.int16)); bucket["y"].append(y.astype(np.int16))
                    bucket["subject"].append(np.full(len(y), str(subject), dtype="U16"))
                    bucket["session"].append(np.full(len(y), int(session), dtype=np.int16))
                    bucket["logits"].append(captured.astype(np.float32))
                    del x, hs, base, curves, random_curves, static_logits, h_t, coeff_t, upper_h
                    print("CELL_SESSION_COMPLETE", task, fold, role_name, subject, f"S{session}",
                        "trials", len(y), "identity", f"{identity_err:.2e}", flush=True)

    if model is None or state_before != model_state_sha(model):
        raise RuntimeError(f"frozen EEGNet state changed: {task}/{fold}")
    if mapping is None: raise RuntimeError("no allowed inner_train or discovery EEG arrays were loaded")
    profile_by_subject = {}
    for (role_name, subject, direction), session_curves in profile_acc.items():
        profile_by_subject.setdefault((role_name, subject), {})[direction] = np.mean(np.stack(session_curves), axis=0)
    for role_name in ("inner_train", "discovery"):
        keys = sorted((key for key in profile_by_subject if key[0] == role_name), key=lambda key: int(key[1].replace("sub-", "")))
        for direction in range(basis.shape[1]):
            profiles = np.stack([profile_by_subject[key][direction] for key in keys])
            mean_curve = profiles.mean(axis=0)
            label, peak_idx, peak_sig, ci0, ci2 = _monotonicity_label(mean_curve, profiles,
                stable_seed("PEAK_BOOTSTRAP", task, fold, role_name, direction))
            subject_argmax = np.argmax(profiles, axis=1)
            results["monotonicity"].append({"task": task, "fold": fold, "role": role_name, "direction": direction,
                "classification": label, "group_argmax_alpha": float(ALPHAS[peak_idx]),
                "peak_near_native_significant": peak_sig, "peak_minus_alpha0_ci95_low": ci0,
                "peak_minus_alpha2_ci95_low": ci2,
                "subject_argmax_alpha_distribution": json.dumps({str(float(a)): int(np.sum(ALPHAS[subject_argmax] == a)) for a in ALPHAS}, sort_keys=True),
                "mean_margin_curve": json.dumps(mean_curve.tolist()), "biological_subjects": len(keys),
                "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})

    router_payload = {}
    for name, parts in router_acc.items():
        for key in ("X", "alpha", "y", "subject", "session", "logits"):
            router_payload[f"{name}_{'alpha_index' if key == 'alpha' else 'top1_logits_curve' if key == 'logits' else key}"] = np.concatenate(parts[key], axis=0)
    router_path = runtime_cell(task, fold) / "router_dataset.npz"
    router_path.parent.mkdir(parents=True, exist_ok=True)
    temp_router = router_path.with_suffix(".npz.part")
    with temp_router.open("wb") as stream: np.savez_compressed(stream, **router_payload)
    os.replace(temp_router, router_path)
    for name, values in results.items(): csvwrite(odir / f"{name.upper()}.csv", values)
    done = {"status": "COMPLETE", "task": task, "fold": fold,
        "protocol_sha256": sha(PROTOCOL / "PROTOCOL_LOCK.json"), "curve_sha256": sha(curve_path),
        "metrics_sha256": sha(metric_path), "router_dataset_sha256": sha(router_path),
        "source_split_sha256": split, "train_subjects": train_ids, "discovery_subjects": discovery_ids,
        "allowed_inner_train_array_reads": len(train_ids) * len(session_ids),
        "allowed_discovery_array_reads": len(discovery_ids) * len(session_ids),
        "current_experiment_eeg_array_reads": (len(train_ids) + len(discovery_ids)) * len(session_ids),
        "outer_dev_subject_count_excluded": len(outer_ids), "outer_dev_arrays_read": 0, "final_heldout_arrays_read": 0,
        "eegnet_state_sha256_before": state_before, "eegnet_state_sha256_after": model_state_sha(model),
        "trainable_parameter_count": 0, "max_alpha1_identity_error": max_identity,
        "alpha1_prediction_identity_checked": True,
        "max_successor_C_preservation_error": max_cpreserve, "top3_utility_directions": top3,
        "top1_router_direction": top1, "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}
    jwrite(done_path, done); geom.close()
    print("CELL_COMPLETE", task, fold, "train", len(train_ids), "discovery", len(discovery_ids), flush=True)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def combine_csv_files(paths: list[Path], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    fields = None
    with temp.open("w", newline="", encoding="utf-8") as out:
        writer = None
        for path in paths:
            if not path.is_file(): continue
            with path.open(newline="", encoding="utf-8") as inp:
                reader = csv.DictReader(inp)
                if fields is None:
                    fields = reader.fieldnames
                    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                for row in reader: writer.writerow(row)
    if fields is None:
        temp.write_text("status\nNO_ROWS\n", encoding="utf-8")
    os.replace(temp, target)


def collapse_metric_rows(rows: list[dict], task: str, role_name: str, variant: str, field: str) -> dict[str, float]:
    # Equal sessions within fold, then equal folds within biological subject.
    fold_subject: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if row["task"] != task or row["role"] != role_name or row["variant"] != variant: continue
        key = (str(row["subject"]), int(row["fold"]))
        fold_subject.setdefault(key, []).append(float(row[field]))
    subject_values: dict[str, list[float]] = {}
    for (subject, _fold), values in fold_subject.items():
        subject_values.setdefault(subject, []).append(float(np.mean(values)))
    return {subject: float(np.mean(values)) for subject, values in subject_values.items()}


def collapse_fold_metric(rows: list[dict], task: str, fold: int, role_name: str,
                         variant: str, field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["task"] == task and int(row["fold"]) == fold and row["role"] == role_name and row["variant"] == variant:
            grouped.setdefault(str(row["subject"]), []).append(float(row[field]))
    return {subject: float(np.mean(values)) for subject, values in grouped.items()}


def paired_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    subjects = sorted(set(a) & set(b))
    return {subject: float(a[subject] - b[subject]) for subject in subjects}


def summary_fields(values: dict[str, float], seed: int) -> dict:
    return subject_summary(np.asarray([values[key] for key in sorted(values)], dtype=np.float64), seed)


def action_model(y: np.ndarray):
    classes = np.unique(y)
    if len(classes) == 1:
        class Constant:
            def __init__(self, value): self.value = int(value)
            def fit(self, x, labels): return self
            def predict(self, x): return np.full(len(x), self.value, dtype=np.int64)
        return Constant(classes[0])
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
        max_iter=500, solver="lbfgs", random_state=0))


def action_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = [0, 1, 2]
    return {"macro_F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=labels).tolist())}


def run_router_for_cell(task: str, fold: int, all_metrics: list[dict]):
    path = runtime_cell(task, fold) / "router_dataset.npz"
    with np.load(path, allow_pickle=False) as data:
        train_X = data["train_X"].astype(np.float32); train_alpha = data["train_alpha_index"].astype(np.int64)
        train_y = _alpha_action(train_alpha); train_subject = data["train_subject"].astype(str)
        disc_X = data["discovery_X"].astype(np.float32); disc_alpha = data["discovery_alpha_index"].astype(np.int64)
        disc_y = _alpha_action(disc_alpha); disc_subject = data["discovery_subject"].astype(str)
        disc_session = data["discovery_session"].astype(np.int64); labels = data["discovery_y"].astype(np.int64)
        disc_logits = data["discovery_top1_logits_curve"].astype(np.float32)
    if set(train_subject) & set(disc_subject): raise RuntimeError("router split is not subject-disjoint")
    unique_subjects = np.unique(train_subject)
    if len(unique_subjects) < 2: raise RuntimeError("subject-grouped router CV needs at least two TRAIN subjects")
    splits = GroupKFold(n_splits=min(5, len(unique_subjects)))
    oof = np.full(len(train_y), -1, dtype=np.int64)
    for tr, va in splits.split(train_X, train_y, groups=train_subject):
        estimator = action_model(train_y[tr]); estimator.fit(train_X[tr], train_y[tr]); oof[va] = estimator.predict(train_X[va])
    if np.any(oof < 0): raise RuntimeError("router grouped CV left unpredicted training rows")
    oof_metrics = action_scores(train_y, oof)
    model = action_model(train_y); model.fit(train_X, train_y); pred_action = model.predict(disc_X).astype(np.int64)
    disc_metrics = action_scores(disc_y, pred_action)
    index_for_action = np.asarray((2, 4, 6), dtype=np.int64)
    applied_idx = index_for_action[pred_action]
    chosen_logits = disc_logits[np.arange(len(labels)), applied_idx]
    chosen_pred = chosen_logits.argmax(axis=1)
    chosen_ce = F.cross_entropy(torch.as_tensor(chosen_logits), torch.as_tensor(labels), reduction="none").numpy()
    base_map = {(r["task"], int(r["fold"]), r["role"], int(r["session"]), str(r["subject"])): r
                for r in all_metrics if r["variant"] == "BASELINE"}
    perf_rows = []
    for sub in sorted(set(disc_subject), key=lambda x: int(x.replace("sub-", ""))):
        for session in sorted(set(disc_session[disc_subject == sub])):
            ix = (disc_subject == sub) & (disc_session == session)
            key = (task, fold, "discovery", int(session), str(sub))
            if key not in base_map: raise RuntimeError(f"router has no baseline metrics row: {key}")
            # alpha=1 in the top-1 curve is the frozen native baseline by the identity audit.
            base_pred = disc_logits[ix, int(np.where(ALPHAS == 1.0)[0][0])].argmax(axis=1)
            perf_rows.append(performance_row(task, fold, "discovery", int(session), str(sub), "ROUTER",
                labels[ix], chosen_pred[ix], chosen_ce[ix], base_pred))
    action_by_subject = []
    for sub in sorted(set(disc_subject), key=lambda x: int(x.replace("sub-", ""))):
        ix = disc_subject == sub
        action_by_subject.append(float(np.mean(pred_action[ix] == disc_y[ix])))
    action_subject_accuracy = float(np.mean(action_by_subject))
    session_rows = []
    for session in sorted(set(disc_session)):
        ix = disc_session == session
        session_rows.append({"task": task, "fold": fold, "session": int(session),
            "action_accuracy": float(np.mean(pred_action[ix] == disc_y[ix])), "trials": int(ix.sum())})
    prediction_rows = [{"task": task, "fold": fold, "population": "inner_train_grouped_OOF",
            **oof_metrics, "accuracy": float(np.mean(oof == train_y)), "biological_subjects": len(unique_subjects),
            "feature_count": int(train_X.shape[1]), "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"},
        {"task": task, "fold": fold, "population": "discovery_subject_disjoint",
            **disc_metrics, "accuracy": float(np.mean(pred_action == disc_y)),
            "subject_level_accuracy": action_subject_accuracy, "biological_subjects": len(np.unique(disc_subject)),
            "feature_count": int(disc_X.shape[1]), "session_transfer_accuracy": json.dumps(session_rows, sort_keys=True),
            "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}]
    return perf_rows, prediction_rows


def aggregate() -> None:
    lock = verify_protocol()
    cells = [(task, fold) for task in TASKS for fold in FOLDS]
    metric_paths = []
    cell_records = []
    for task, fold in cells:
        odir = output_cell(task, fold); done_path = odir / "CELL_COMPLETE.json"
        metric_path = odir / "METRICS.csv"; curve_path = odir / "RESPONSE_CURVES.csv.gz"
        if not done_path.is_file() or not metric_path.is_file() or not curve_path.is_file():
            raise RuntimeError(f"cell incomplete: {task}/{fold}")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("status") != "COMPLETE" or done.get("protocol_sha256") != sha(PROTOCOL / "PROTOCOL_LOCK.json"):
            raise RuntimeError(f"cell lock mismatch: {task}/{fold}")
        if done.get("final_heldout_arrays_read") != 0 or done.get("outer_dev_arrays_read") != 0:
            raise RuntimeError(f"heldout/outer-dev array read recorded: {task}/{fold}")
        if done.get("curve_sha256") != sha(curve_path) or done.get("metrics_sha256") != sha(metric_path):
            raise RuntimeError(f"cell output hash mismatch: {task}/{fold}")
        metric_paths.append(metric_path); cell_records.append(done)
    if len(cell_records) != 20:
        raise RuntimeError("all 20 locked task/fold cells are required")

    # Concatenate the full trial-level curves and compact subject/session summaries.
    combine_csv_files([output_cell(t, f) / "RESPONSE_SUMMARY.csv" for t, f in cells], OUT / "RESPONSE_CURVES.csv")
    combine_csv_files([output_cell(t, f) / "LOCAL.csv" for t, f in cells], OUT / "LOCAL_ACTIONABILITY.csv")
    combine_csv_files([output_cell(t, f) / "MONOTONICITY.csv" for t, f in cells], OUT / "MONOTONICITY_SUMMARY.csv")
    combine_csv_files([output_cell(t, f) / "UTILITY.csv" for t, f in cells], OUT / "UTILITY_VS_ACTIONABILITY.csv")
    combine_csv_files([output_cell(t, f) / "CONDITIONALITY.csv" for t, f in cells], OUT / "TRIAL_CONDITIONALITY.csv")
    metrics = [row for path in metric_paths for row in read_csv(path)]
    oracle1_rows = [r for r in metrics if r["variant"] in ("ORACLE_1DIR", "CE_ORACLE_1DIR")]
    top3_rows = [r for r in metrics if r["variant"] == "ORACLE_TOP3"]
    upper_rows = [r for r in metrics if r["variant"] == "ORACLE_UPPER"]
    random_rows = [r for r in metrics if r["variant"].startswith("RANDOM_")]
    csvwrite(OUT / "ORACLE_1DIR_RESULTS.csv", oracle1_rows)
    csvwrite(OUT / "ORACLE_TOP3_RESULTS.csv", top3_rows)
    csvwrite(OUT / "ORACLE_UPPER_BOUND.csv", upper_rows)

    primary_variants = ("BASELINE", "STATIC_V3", "ORACLE_1DIR", "CE_ORACLE_1DIR", "ORACLE_TOP3", "ORACLE_UPPER", "RANDOM_DIRECTION_ORACLE")
    fold_rows = []
    for task in TASKS:
        for fold in FOLDS:
            for role_name in ("inner_train", "discovery"):
                for variant in primary_variants + tuple(f"RANDOM_ALPHA_DRAW_{i:02d}" for i in range(N_RANDOM_ALPHA)):
                    by_subject = collapse_fold_metric(metrics, task, fold, role_name, variant, "BA")
                    if not by_subject: continue
                    seed = stable_seed("TASK_FOLD_BA", task, fold, role_name, variant)
                    pack = subject_summary(np.asarray(list(by_subject.values())), seed) if variant in primary_variants else {
                        "mean": float(np.mean(list(by_subject.values()))), "ci95_low": math.nan, "ci95_high": math.nan,
                        "median": float(np.median(list(by_subject.values()))), "positive_subject_fraction": math.nan,
                        "biological_subjects": len(by_subject)}
                    f1 = collapse_fold_metric(metrics, task, fold, role_name, variant, "macro_F1")
                    nll = collapse_fold_metric(metrics, task, fold, role_name, variant, "NLL")
                    rescue = [int(r["rescued_errors"]) for r in metrics if r["task"] == task and int(r["fold"]) == fold and r["role"] == role_name and r["variant"] == variant]
                    damage = [int(r["damaged_correct"]) for r in metrics if r["task"] == task and int(r["fold"]) == fold and r["role"] == role_name and r["variant"] == variant]
                    fold_rows.append({"task": task, "fold": fold, "role": role_name, "variant": variant,
                        "BA_subject_equal_mean": pack["mean"], "BA_CI95_low": pack["ci95_low"], "BA_CI95_high": pack["ci95_high"],
                        "BA_subject_median": pack["median"], "MacroF1_subject_equal_mean": float(np.mean(list(f1.values()))),
                        "NLL_subject_equal_mean": float(np.mean(list(nll.values()))), "positive_subject_fraction": pack["positive_subject_fraction"],
                        "rescued_errors_total": int(sum(rescue)), "correct_predictions_damaged_total": int(sum(damage)),
                        "biological_subjects": pack["biological_subjects"], "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    csvwrite(OUT / "TASK_FOLD_SUMMARY.csv", fold_rows)

    headroom_rows = []
    headroom_by_task = {}
    for task in TASKS:
        base = collapse_metric_rows(metrics, task, "discovery", "BASELINE", "BA")
        task_record = {"task": task, "role": "discovery", "baseline_BA": float(np.mean(list(base.values()))) if base else math.nan,
            "biological_subjects": len(base), "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}
        task_variant_values = {}
        for variant in ("STATIC_V3", "ORACLE_1DIR", "ORACLE_TOP3", "ORACLE_UPPER", "RANDOM_DIRECTION_ORACLE"):
            vals = collapse_metric_rows(metrics, task, "discovery", variant, "BA")
            task_variant_values[variant] = vals
            common = paired_delta(vals, base)
            pack = summary_fields(common, stable_seed("HEADROOM", task, variant))
            task_record[f"{variant}_BA"] = float(np.mean(list(vals.values()))) if vals else math.nan
            for field, value in pack.items(): task_record[f"{variant}_delta_BA_{field}"] = value
        g1 = task_record["ORACLE_1DIR_delta_BA_mean"]
        g3 = task_record["ORACLE_TOP3_delta_BA_mean"]
        if g1 < LOW_ORACLE1 and g3 < LOW_ORACLE3: state = "LOW_ACTIONABILITY_HEADROOM"
        elif g3 > 0.02: state = "STRONG_ACTIONABILITY_HEADROOM"
        elif 0.01 <= g3 <= 0.02: state = "MODERATE_ACTIONABILITY_HEADROOM"
        else: state = "INTERMEDIATE_ACTIONABILITY_HEADROOM"
        task_record["headroom_state"] = state
        task_record["router_triggered"] = bool(g1 >= 0.01 or g3 >= 0.01)
        headroom_by_task[task] = task_record
        headroom_rows.append(task_record)
    csvwrite(OUT / "ORACLE_HEADROOM_SUMMARY.csv", headroom_rows)
    random_summary = []
    for task in TASKS:
        base = collapse_metric_rows(metrics, task, "discovery", "BASELINE", "BA")
        variants = ["RANDOM_DIRECTION_ORACLE"] + [f"RANDOM_ALPHA_DRAW_{i:02d}" for i in range(N_RANDOM_ALPHA)]
        for variant in variants:
            vals = collapse_metric_rows(metrics, task, "discovery", variant, "BA")
            delta = paired_delta(vals, base)
            pack = summary_fields(delta, stable_seed("RANDOM_CONTROL", task, variant))
            random_summary.append({"task": task, "variant": variant, "row_type": "SUBJECT_EQUAL_DISCOVERY_SUMMARY",
                **{f"delta_BA_{k}": v for k, v in pack.items()},
                "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    random_rows.extend(random_summary)
    csvwrite(OUT / "RANDOM_ORACLE_CONTROLS.csv", random_rows)

    router_audit = []; router_predictions = []; router_performance = []
    triggered_tasks = {task for task, row in headroom_by_task.items() if row["router_triggered"]}
    for task, fold in cells:
        with np.load(runtime_cell(task, fold) / "router_dataset.npz", allow_pickle=False) as data:
            train_target = _alpha_action(data["train_alpha_index"].astype(np.int64))
            disc_target = _alpha_action(data["discovery_alpha_index"].astype(np.int64))
            train_ids = np.unique(data["train_subject"].astype(str)); disc_ids = np.unique(data["discovery_subject"].astype(str))
            row = {"task": task, "fold": fold, "triggered": task in triggered_tasks,
                "status": "READY" if task in triggered_tasks else "ROUTER_PREDICTION_SKIPPED_LOW_HEADROOM",
                "train_trial_count": int(len(train_target)), "train_subject_count": int(len(train_ids)),
                "train_session_count": int(len(np.unique(data["train_session"]))),
                "discovery_trial_count": int(len(disc_target)), "discovery_subject_count": int(len(disc_ids)),
                "discovery_session_count": int(len(np.unique(data["discovery_session"]))),
                "feature_count": int(data["train_X"].shape[1]),
                "train_action_class_counts": json.dumps(np.bincount(train_target, minlength=3).tolist()),
                "discovery_oracle_action_class_counts": json.dumps(np.bincount(disc_target, minlength=3).tolist()),
                "subject_disjoint": not bool(set(train_ids) & set(disc_ids)),
                "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}
        router_audit.append(row)
        if task not in triggered_tasks:
            continue
        perfs, preds = run_router_for_cell(task, fold, metrics)
        router_performance.extend(perfs); router_predictions.extend(preds)
    csvwrite(OUT / "ROUTER_DATASET_AUDIT.csv", router_audit)
    if triggered_tasks:
        csvwrite(OUT / "ROUTER_PREDICTION_RESULTS.csv", router_predictions)
        # Include subject/session measurements and one task-level subject-equal summary row per method.
        for task in sorted(triggered_tasks):
            existing = [r for r in router_performance if r["task"] == task]
            task_base = collapse_metric_rows(metrics, task, "discovery", "BASELINE", "BA")
            task_static = collapse_metric_rows(metrics, task, "discovery", "STATIC_V3", "BA")
            task_oracle1 = collapse_metric_rows(metrics, task, "discovery", "ORACLE_1DIR", "BA")
            task_oracle3 = collapse_metric_rows(metrics, task, "discovery", "ORACLE_TOP3", "BA")
            routed_rows = [r for r in existing if r["variant"] == "ROUTER"]
            by_fold_sub = {}
            for r in routed_rows:
                by_fold_sub.setdefault((str(r["subject"]), int(r["fold"])), []).append(float(r["BA"]))
            per_subject = {}
            for (sub, _fold), values in by_fold_sub.items(): per_subject.setdefault(sub, []).append(float(np.mean(values)))
            router_subject = {sub: float(np.mean(values)) for sub, values in per_subject.items()}
            router_mean = float(np.mean(list(router_subject.values()))) if router_subject else math.nan
            base_mean = float(np.mean(list(task_base.values()))) if task_base else math.nan
            oracle1_mean = float(np.mean(list(task_oracle1.values()))) if task_oracle1 else math.nan
            oracle3_mean = float(np.mean(list(task_oracle3.values()))) if task_oracle3 else math.nan
            oracle_mean = max(oracle1_mean, oracle3_mean)
            oracle_source = "ORACLE_1DIR" if oracle1_mean >= oracle3_mean else "ORACLE_TOP3"
            headroom = oracle_mean - base_mean
            router_gain = router_mean - base_mean
            recovered = router_gain / headroom if abs(headroom) > 1e-12 else math.nan
            router_performance.append({"task": task, "fold": "ALL_FOLDS", "role": "discovery", "session": "ALL",
                "subject": "SUBJECT_EQUAL_AGGREGATE", "variant": "ROUTER_SUMMARY", "BA": router_mean,
                "baseline_BA": base_mean, "static_V3_BA": float(np.mean(list(task_static.values()))) if task_static else math.nan,
                "oracle_1dir_BA": float(np.mean(list(task_oracle1.values()))) if task_oracle1 else math.nan,
                "oracle_top3_BA": oracle3_mean, "oracle_headroom_source": oracle_source,
                "oracle_BA_used_for_headroom": oracle_mean, "router_BA_gain": router_gain,
                "oracle_BA_headroom": headroom, "fraction_oracle_headroom_recovered": recovered,
                "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
        csvwrite(OUT / "ROUTER_PERFORMANCE.csv", router_performance)

    # Reassert the exclusion policy after every cell and before writing conclusions.
    audit = json.loads((OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json").read_text(encoding="utf-8"))
    audit.update({"FINAL_HELDOUT_ACCESSED": False, "final_heldout_array_reads": 0,
        "outer_dev_array_reads": 0, "completed_locked_cells": len(cell_records),
        "current_experiment_eeg_array_reads": int(sum(r["current_experiment_eeg_array_reads"] for r in cell_records)),
        "allowed_inner_train_array_reads": int(sum(r["allowed_inner_train_array_reads"] for r in cell_records)),
        "allowed_discovery_array_reads": int(sum(r["allowed_discovery_array_reads"] for r in cell_records)),
        "current_experiment_reads_only_inner_train_and_discovery": True,
        "outer_dev_arrays_loaded": False, "final_heldout_ids_or_arrays_loaded": False,
        "verified_cell_read_audits": True, "protocol_lock_sha256": sha(PROTOCOL / "PROTOCOL_LOCK.json"),
        "label_scope": "DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    jwrite(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", audit)
    write_final_report(metrics, headroom_by_task, triggered_tasks)
    print("AGGREGATE_COMPLETE", "cells=20", "router_tasks=" + (",".join(sorted(triggered_tasks)) or "none"),
        "FINAL_HELDOUT_ACCESSED=False", flush=True)


def _subject_equal_fraction(rows: list[dict], subject_field: str, predicate, denominator=None) -> float:
    per_subject = {}
    for row in rows:
        subject = str(row[subject_field])
        per_subject.setdefault(subject, [0.0, 0.0])
        if denominator is None or denominator(row):
            per_subject[subject][1] += 1.0
            if predicate(row): per_subject[subject][0] += 1.0
    vals = [num / den for num, den in per_subject.values() if den > 0]
    return float(np.mean(vals)) if vals else math.nan


def _subject_equal_value(rows: list[dict], subject_field: str, value_field: str, denominator=None) -> float:
    per_subject = {}
    for row in rows:
        if denominator is not None and not denominator(row): continue
        per_subject.setdefault(str(row[subject_field]), []).append(float(row[value_field]))
    vals = [float(np.mean(values)) for values in per_subject.values() if values]
    return float(np.mean(vals)) if vals else math.nan


def _subject_equal_joint_fraction(rows: list[dict], subject_field: str, value_field: str,
                                  condition) -> float:
    per_subject = {}
    for row in rows:
        value = float(row[value_field]) if condition(row) else 0.0
        per_subject.setdefault(str(row[subject_field]), []).append(value)
    vals = [float(np.mean(values)) for values in per_subject.values() if values]
    return float(np.mean(vals)) if vals else math.nan


def write_final_report(metrics: list[dict], headroom: dict[str, dict], triggered_tasks: set[str]) -> None:
    local_rows = read_csv(OUT / "LOCAL_ACTIONABILITY.csv")
    mono_rows = read_csv(OUT / "MONOTONICITY_SUMMARY.csv")
    conditional_rows = read_csv(OUT / "TRIAL_CONDITIONALITY.csv")
    utility_rows = read_csv(OUT / "UTILITY_VS_ACTIONABILITY.csv")
    utility_by_key = {(r["task"], r["fold"], r["role"], r["session"], r["subject"], r["direction"]): float(r["train_U_P"])
                      for r in utility_rows}
    table1 = []
    table2 = []
    low_states = []
    task_evidence = {}
    for task in TASKS:
        h = headroom[task]
        base = float(h["baseline_BA"])
        table1.append(f"| {task} | {base:.4f} | {float(h['STATIC_V3_BA']):.4f} | {float(h['ORACLE_1DIR_BA']):.4f} | {float(h['ORACLE_TOP3_BA']):.4f} | {100*float(h['ORACLE_1DIR_delta_BA_mean']):+.2f} pp | {100*float(h['ORACLE_TOP3_delta_BA_mean']):+.2f} pp |")
        if h["headroom_state"] == "LOW_ACTIONABILITY_HEADROOM": low_states.append(task)
        lr = [r for r in local_rows if r["task"] == task and r["role"] == "discovery"]
        useful_neg_all = _subject_equal_fraction(lr, "subject",
            lambda r: float(r["train_U_P"]) > 0 and float(r["D_margin"]) < -0.001)
        useful_neg_within = _subject_equal_fraction(lr, "subject",
            lambda r: float(r["D_margin"]) < -0.001,
            denominator=lambda r: float(r["train_U_P"]) > 0)
        mr = [r for r in mono_rows if r["task"] == task and r["role"] == "discovery"]
        peak = float(np.mean([r["classification"] == "PEAK_NEAR_NATIVE" for r in mr])) if mr else math.nan
        nonmono = float(np.mean([r["classification"] == "NONMONOTONIC" for r in mr])) if mr else math.nan
        cr = [r for r in conditional_rows if r["task"] == task and r["role"] == "discovery"]
        cond = _subject_equal_fraction(cr, "subject", lambda r: str(r["both_sides_at_least_0p10"]).lower() == "true")
        cr_useful = []
        for row in cr:
            key = (row["task"], row["fold"], row["role"], row["session"], row["subject"], row["direction"])
            cr_useful.append({**row, "train_U_P": utility_by_key[key]})
        ur = [r for r in utility_rows if r["task"] == task and r["role"] == "discovery"]
        useful_le_native = _subject_equal_value(ur, "subject", "margin_alpha_star_le_1_fraction",
            denominator=lambda r: float(r["train_U_P"]) > 0)
        useful_le_native_joint = _subject_equal_joint_fraction(ur, "subject", "margin_alpha_star_le_1_fraction",
            condition=lambda r: float(r["train_U_P"]) > 0)
        useful_near_native = _subject_equal_value(ur, "subject", "margin_alpha_star_near_native_fraction",
            denominator=lambda r: float(r["train_U_P"]) > 0)
        cond_useful = _subject_equal_fraction(cr_useful, "subject",
            lambda r: str(r["both_sides_at_least_0p10"]).lower() == "true",
            denominator=lambda r: float(r["train_U_P"]) > 0)
        task_evidence[task] = {"useful_neg_within": useful_neg_within, "useful_neg_all": useful_neg_all,
            "useful_le_native": useful_le_native, "useful_le_native_joint": useful_le_native_joint,
            "useful_near_native": useful_near_native, "peak": peak, "nonmono": nonmono,
            "cond": cond, "cond_useful": cond_useful}
        table2.append(f"| {task} | {100*useful_neg_within:.1f}% | {100*useful_neg_all:.1f}% | {100*useful_le_native_joint:.1f}% | {100*peak:.1f}% | {100*nonmono:.1f}% | {100*cond:.1f}% |")

    lines = ["# Frozen C-direction Actionability and Oracle Audit", "",
        "## Scope and exclusion audit", "",
        "This is a development-only audit of four seed-zero EEGNet tasks across folds 0–4. All current experiment EEG reads were explicit `B.rows(..., final=False)` calls for inner-train or discovery subjects. It completed all 20 cells with zero outer-dev EEG reads and zero final-heldout EEG reads. No final-heldout predictions were loaded.", "",
        "The reused V3 checkpoint, projectors, means, C basis, and utility geometry were frozen before this audit. V3 geometry provenance includes its prior non-final refit pool, which included outer-dev subjects; this audit adds no outer-dev arrays, but its reused geometry is not an independent train-only estimate. This limitation is part of the evidence boundary.", "",
        f"Protocol lock SHA-256: `{sha(PROTOCOL / 'PROTOCOL_LOCK.json')}`. Source commit: `{lock_source_commit()}`.", "",
        "## Table 1. Discovery performance and oracle headroom", "",
        "| Task | Baseline BA | Static V3 BA | Oracle-1dir BA | Oracle-top3 BA | ΔOracle1 | ΔOracle3 |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(table1)
    lines += ["", "Oracle results use labels only to select oracle actions and are not deployable estimates. Each Δ has a paired 20,000-draw biological-subject bootstrap in `ORACLE_HEADROOM_SUMMARY.csv`. Macro-F1, NLL, rescued errors, and correct predictions damaged are recorded in the decision-level results.", "",
        "## Table 2. Utility, local response, and trial conditionality", "",
        "| Task | U+ with negative slope among U+ | P(U+ and negative slope) | P(U+ and α*≤1) | Peak near native | Nonmonotonic | Trial-conditional |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(table2)
    lines += ["", "A negative local slope is computed from the fixed centered difference between α=0.75 and α=1.25. Peak-near-native requires the group subject-equal curve maximum in that interval and paired subject-bootstrap lower bounds above both α=0 and α=2. Trial-conditional means the same direction has at least 10% of its trial-level margin optima below α=1 and at least 10% above α=1 within a subject/session.", "",
        "The four local-actionability quadrants are in `LOCAL_ACTIONABILITY.csv`; the direction-utility relation and margin/CE oracle-alpha distributions are in `UTILITY_VS_ACTIONABILITY.csv` and `TRIAL_CONDITIONALITY.csv`. The detailed trial × direction curves, including margin, CE, prediction, confidence, entropy, and successor-P movement at every locked α, are in each cell’s compressed `RESPONSE_CURVES.csv.gz`; the top-level `RESPONSE_CURVES.csv` is their compact subject/session summary.", "",
        "## Conditional router", ""]
    if triggered_tasks:
        router_rows = read_csv(OUT / "ROUTER_PERFORMANCE.csv")
        for task in sorted(triggered_tasks):
            row = next((r for r in router_rows if r.get("task") == task and r.get("variant") == "ROUTER_SUMMARY"), None)
            if row:
                lines.append(f"- **{task}:** router BA {float(row['BA']):.4f}, baseline {float(row['baseline_BA']):.4f}, static V3 {float(row['static_V3_BA']):.4f}; triggered oracle `{row['oracle_headroom_source']}` BA {float(row['oracle_BA_used_for_headroom']):.4f}; router gain {100*float(row['router_BA_gain']):+.2f} pp; headroom recovered {100*float(row['fraction_oracle_headroom_recovered']):.1f}%.")
        lines += ["", "The router fits only inner-train subjects with five-fold subject-grouped OOF diagnostics, then evaluates on disjoint discovery subjects. Inputs are native label-free features. It predicts SUPPRESS/KEEP/ENHANCE for the prelocked top-1 direction and applies α=0.5/1.0/1.5 using the stored frozen response curve. Detailed action metrics and confusion matrices are in `ROUTER_PREDICTION_RESULTS.csv`; applied-model results are in `ROUTER_PERFORMANCE.csv`."]
    else:
        lines.append("No task reached the prelocked discovery gate (Oracle-1dir or Oracle-top3 BA gain ≥1.0 pp). Router fitting and evaluation were skipped. `ROUTER_DATASET_AUDIT.csv` records `ROUTER_PREDICTION_SKIPPED_LOW_HEADROOM` for every task/fold.")
    lines += ["", "## Interpretation", ""]
    mean_near_native = float(np.mean([task_evidence[t]["useful_near_native"] for t in TASKS]))
    mean_useful_negative = float(np.mean([task_evidence[t]["useful_neg_within"] for t in TASKS]))
    mean_nonmonotonic = float(np.mean([task_evidence[t]["peak"] + task_evidence[t]["nonmono"] for t in TASKS]))
    mean_useful_conditional = float(np.mean([task_evidence[t]["cond_useful"] for t in TASKS]))
    if len(low_states) == len(TASKS):
        lines.append("- `NO_ROUTING_PERFORMANCE_PATH`: all four tasks meet the locked low-headroom rule for Oracle-1dir (<0.5 pp) and Oracle-top3 (<1.0 pp). Under this audit, stronger oracle selection does not create enough development BA headroom to justify a performance architecture direction.")
        if mean_near_native > 0.5:
            lines.append("- `NATIVE_STRENGTH_NEAR_OPTIMAL`: more than half of utility-positive trial optima lie in α∈{0.75,1,1.25}, alongside low oracle headroom.")
    if mean_useful_negative >= 0.5 or mean_nonmonotonic >= 0.5:
        lines.append("- `UTILITY_NONMONOTONIC`: at least half of utility-positive subject-session-direction rows have nonpositive local margin slope, or at least half of the audited curves are peak-near-native/nonmonotonic.")
    if triggered_tasks and mean_useful_conditional >= 0.5:
        lines.append("- `TRIAL_CONDITIONAL_ACTIONABILITY`: oracle BA reaches the 1 pp trigger in at least one task; this establishes development oracle headroom, not deployability.")
    if triggered_tasks:
        router_rows = read_csv(OUT / "ROUTER_PERFORMANCE.csv")
        recoveries = [float(r["fraction_oracle_headroom_recovered"]) for r in router_rows if r.get("variant") == "ROUTER_SUMMARY" and r.get("fraction_oracle_headroom_recovered") not in ("", "nan")]
        if recoveries and max(recoveries) > 0.30:
            lines.append("- `PREDICTABLE_CONDITIONAL_ACTIONABILITY`: at least one diagnostic router recovers more than 30% of the triggered oracle BA headroom on discovery subjects.")
        else:
            lines.append("- `UNPREDICTABLE_ORACLE_HEADROOM`: oracle headroom passed the trigger, but no task router recovered more than 30% on subject-disjoint discovery subjects.")
    lines += ["", "`U_P>0` is an erasure-necessity result. Its relationship to amplification is measured directly by the sign of the local margin slope, the full response curve, and the distribution of trial-specific α optima. Static V3 performance is reported beside both conservative and loose oracle bounds, so a useful direction is not interpreted as a reason to amplify it globally.", "",
        "## Required artifacts", "",
        "See `ORACLE_1DIR_RESULTS.csv`, `ORACLE_TOP3_RESULTS.csv`, `ORACLE_UPPER_BOUND.csv`, `ORACLE_HEADROOM_SUMMARY.csv`, `RANDOM_ORACLE_CONTROLS.csv`, `TASK_FOLD_SUMMARY.csv`, `FINAL_HELDOUT_EXCLUSION_AUDIT.json`, and `ROUTER_DATASET_AUDIT.csv`.", "",
        "Every table is development-only. This report makes no final-heldout performance claim.", ""]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def lock_source_commit() -> str:
    try:
        lock = json.loads((PROTOCOL / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
        return str(lock.get("source_commit", "UNAVAILABLE"))
    except Exception:
        return source_commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("lock")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--task", choices=TASKS, required=True)
    ev.add_argument("--fold", type=int, choices=FOLDS, required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "preflight": preflight()
    elif args.command == "lock": lock_protocol()
    elif args.command == "evaluate": evaluate_cell(args.task, args.fold)
    elif args.command == "aggregate": aggregate()


if __name__ == "__main__":
    main()
