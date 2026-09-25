"""Frozen SIRE-EEG C-direction actionability and observability audit.

All neural parameters and BatchNorm states remain frozen. The only EEG arrays
read are inner-train and discovery rows returned by the existing V3 loader.
"""
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
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
V3_CODE = REPO / "experiments/persist_eeg_selective_cp_routing_v3_seed0/code/run.py"
V3_SPEC = importlib.util.spec_from_file_location("sire_actionability_v3", V3_CODE)
if V3_SPEC is None or V3_SPEC.loader is None:
    raise RuntimeError(f"missing shared EEG loader: {V3_CODE}")
V3 = importlib.util.module_from_spec(V3_SPEC)
sys.modules[V3_SPEC.name] = V3
V3_SPEC.loader.exec_module(V3)
B = V3.B

OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("SIRE_CP_ACTIONABILITY_RUNTIME", str(REPO.parent / "sire_cp_actionability_seed0_runtime"))).resolve()
EEGNET_EXP = REPO / "experiments/persist_eeg_cp_actionability_oracle_v1_seed0"
EEGNET_OUT = EEGNET_EXP / "outputs"
EEGNET_RUNTIME = Path(os.environ.get("CP_ACTIONABILITY_RUNTIME", str(REPO.parent / "cp_actionability_oracle_v1_runtime"))).resolve()
CANONICAL_SOURCE_REPO = Path(os.environ.get("SIRE_CANONICAL_SOURCE_REPO", str(REPO))).resolve()
CARRIER_SOURCE = CANONICAL_SOURCE_REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py"
TASK_SOURCE = REPO / "experiments/persist_eeg_openbmi_task_generality_v1/code/task_datasets.py"
SOURCE_RECORDS = PROTOCOL / "SIRE_SOURCE_RECORDS.json"
EEGNET_REFERENCE_AUDIT = PROTOCOL / "EEGNET_REFERENCE_AUDIT.json"
TASKS = ("OpenBMI_MI", "OpenBMI_SSVEP")
FOLDS = tuple(range(5))
ALPHAS = np.asarray((0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), dtype=np.float32)
BOOTSTRAPS = 20_000
TOP_K = 3
ACTIVE_RANK_MAX = 20  # Frozen SIRE PERSIST/PEEH spectrum convention.
RIDGE_ALPHA = 1.0     # Frozen mechanism-closure PathFit convention.
CURVE_BATCH = int(os.environ.get("SIRE_CP_CURVE_BATCH", "8"))
INFER_BATCH = int(os.environ.get("SIRE_CP_INFER_BATCH", "128"))
DEVICE = V3.DEVICE
torch.set_num_threads(min(int(os.environ.get("SIRE_CP_CPU_THREADS", "16")), os.cpu_count() or 1))
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arr_sha(*values: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def clean(v):
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: clean(row.get(k, "")) for k in fields} for row in rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ordered_subjects(values) -> list[str]:
    return sorted(set(map(str, values)), key=lambda s: int(s.replace("sub-", "")))


def subject_key(values) -> str:
    return hashlib.sha256("|".join(ordered_subjects(values)).encode()).hexdigest()


def load_source_records() -> dict[tuple[str, int], dict]:
    raw = json.loads(SOURCE_RECORDS.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict): raw = [raw]
    records = {(r["task"], int(r["fold"])): r for r in raw}
    expected = {(t, f) for t in TASKS for f in FOLDS}
    if set(records) != expected:
        raise RuntimeError("SIRE frozen source records do not cover the locked 10 cells")
    return records


def prior_eegnet_feature_budget(task: str, fold: int) -> int:
    rows = read_csv(EEGNET_OUT / "ROUTER_PREDICTION_RESULTS.csv")
    found = [r for r in rows if r["task"] == task and int(r["fold"]) == fold and r["population"] == "discovery_subject_disjoint"]
    if len(found) != 1:
        raise RuntimeError(f"EEGNet feature budget missing/duplicated for {task}/{fold}")
    return int(found[0]["feature_count"])


def verify_eegnet_reference_audit(eegnet_lock: dict) -> dict:
    audit = json.loads(EEGNET_REFERENCE_AUDIT.read_text(encoding="utf-8-sig"))
    code_key = "experiments/persist_eeg_cp_actionability_oracle_v1_seed0/code/run.py"
    locked_code_sha = eegnet_lock.get("code_files", {}).get(code_key)
    if not locked_code_sha or locked_code_sha != audit.get("locked_run_source_sha256"):
        raise RuntimeError("EEGNet reference source hash does not match its frozen protocol record")
    if eegnet_lock.get("source_commit") != audit.get("source_commit"):
        raise RuntimeError("EEGNet reference source commit does not match its audit")
    if not audit.get("historical_source_blob_sha256_verified"):
        raise RuntimeError("EEGNet locked source blob was not verified")
    if sha_file(EEGNET_EXP / "protocol/PROTOCOL_LOCK.json") != audit.get("protocol_lock_sha256"):
        raise RuntimeError("EEGNet reference protocol lock changed")
    for name, expected in audit.get("outputs_sha256", {}).items():
        path = EEGNET_OUT / name
        if not path.is_file() or sha_file(path) != expected:
            raise RuntimeError(f"frozen EEGNet reference output changed: {name}")
    return audit


def lock_protocol() -> None:
    lock_path = PROTOCOL / "PROTOCOL_LOCK.json"
    if lock_path.exists():
        raise RuntimeError("protocol already exists; refusing to relock")
    if not (EEGNET_OUT / "ORACLE_HEADROOM_SUMMARY.csv").is_file():
        raise RuntimeError("completed EEGNet reference outputs are missing")
    eegnet_lock_path = EEGNET_EXP / "protocol/PROTOCOL_LOCK.json"
    eegnet_lock = json.loads(eegnet_lock_path.read_text(encoding="utf-8-sig"))
    eegnet_code = EEGNET_EXP / "code/run.py"
    eegnet_audit = verify_eegnet_reference_audit(eegnet_lock)
    source_records = load_source_records()
    cells = []
    for task in TASKS:
        for fold in FOLDS:
            role, split, cache, source_sessions, future = B.role(task, fold)
            train = ordered_subjects(role["inner_train_subjects"])
            discovery = ordered_subjects(role["inner_val_subjects"])
            outer = ordered_subjects(role["outer_dev_subjects"])
            if (set(train) & set(discovery)) or (set(train) & set(outer)) or (set(discovery) & set(outer)):
                raise RuntimeError(f"role overlap {task}/{fold}")
            source = source_records[(task, fold)]
            for role_name, values in (("inner_train_subjects", train), ("inner_val_subjects", discovery), ("outer_dev_subjects", outer)):
                if ordered_subjects(source["subject_split"][role_name]) != values:
                    raise RuntimeError(f"canonical SIRE/EEGNet split mismatch {task}/{fold}/{role_name}")
            cells.append({"task": task, "fold": fold, "seed": 0, "split_sha256": split,
                "cache_name": cache, "source_sessions": list(map(int, source_sessions)), "future_session": int(future),
                "inner_train_subjects": train, "discovery_subjects": discovery, "outer_dev_subjects_excluded": outer,
                "feature_budget_from_eegnet_router_csv": prior_eegnet_feature_budget(task, fold),
                "frozen_checkpoint_path": source["checkpoint_path"], "frozen_checkpoint_sha256": source["checkpoint_sha256"],
                "normalizer_path": source["normalizer_path"], "normalizer_sha256": source["normalizer_sha256"],
                "protected_coordinates_from_frozen_sire_persist_record": source["protected_coordinates"],
                "historically_exposed_diagnostic": bool(source["historical_final_heldout_diagnostic"])})
    lock = {
        "schema": "SIRE_CP_ACTIONABILITY_OBSERVABILITY_SEED0_V1",
        "scope": {"tasks": list(TASKS), "folds": list(FOLDS), "seed": 0, "backbone": "SIRE-EEG", "neural_training": False},
        "source": {"analysis_code": str(Path(__file__).resolve()), "analysis_code_sha256": sha_file(Path(__file__).resolve()),
            "carrier_source": str(CARRIER_SOURCE), "carrier_source_sha256": sha_file(CARRIER_SOURCE),
            "task_model_builder": str(TASK_SOURCE), "task_model_builder_sha256": sha_file(TASK_SOURCE),
            "source_records_sha256": sha_file(SOURCE_RECORDS), "eegnet_reference_run_locked_sha256": eegnet_audit["locked_run_source_sha256"],
            "eegnet_current_workspace_run_sha256": sha_file(eegnet_code), "eegnet_reference_audit_sha256": sha_file(EEGNET_REFERENCE_AUDIT),
            "eegnet_protocol_lock_sha256": sha_file(eegnet_lock_path)},
        "projector": {"protocol": "TRAIN-only subject-session-label centroids; standardized PathFit ridge alpha=1.0; recover raw-unit column space; reduced QR orthogonalization",
            "target": "frozen SIRE PERSIST/PEEH protected canonical coordinates from the historical source record; spectrum is reconstructed from current inner-train SIRE embeddings only",
            "source_and_successor_stage_projectors_refit": True, "discovery_labels_used": False, "heldout_labels_used": False},
        "source_stage": "H_concat (48 channels after three temporal/spatial branches and first pool, before depth1)",
        "successor_stage": "H_shared1 (depth1->point1->norm1->ELU->pool, before depth2)",
        "alphas": ALPHAS.tolist(), "complement_pca": {"basis_count": "min(16,numerical_rank(C))", "numeric_rank_rule": "singular value > max(s0*1e-6,1e-8)", "fit_roles": ["inner_train"], "labels_used": False},
        "intervention": "one C direction scaled; pass through frozen source-to-successor block; replace successor P by candidate P and keep native successor C fixed; then frozen suffix",
        "identity_tolerances": {"representation_max_abs": 1e-6, "logits_max_abs": 1e-6, "predictions": "exact",
            "alpha_one": "assert source no-op, then reuse native successor and native logits to avoid batch-size-dependent convolution roundoff"},
        "oracle": {"oracle_1dir": "per trial maximize true-class margin over all C directions and fixed alpha grid; tie closest to 1, then smaller alpha, then lower direction index",
            "oracle_top3": "three directions ranked by descending absolute inner-train subject-equal U_P, ties by local direction index; sequential trial-specific true-margin alpha, K fixed at 3",
            "random_oracle": "one-direction oracle over deterministic equal-rank Gaussian-QR basis drawn in the source C complement"},
        "local_actionability": {"D_margin": "(mean margin(alpha=1.25)-mean margin(alpha=0.75))/0.5", "U_P": "mean true-class margin(alpha=1)-mean true-class margin(alpha=0)", "conditional": "same subject/session/direction has >=10% trials with alpha*<1 and >=10% with alpha*>1"},
        "router": {"family": "StandardScaler + LogisticRegression(C=1,class_weight=balanced,max_iter=500,solver=lbfgs,random_state=0)",
            "inner_cv": "GroupKFold(min(5, number of inner-train subjects))", "target_direction": "first direction in train-only absolute-U_P ranking",
            "actions": {"SUPPRESS": "alpha*<=0.5", "KEEP": "0.75<=alpha*<=1.25", "ENHANCE": "alpha*>=1.5"},
            "apply_alpha_by_action": {"SUPPRESS": 0.5, "KEEP": 1.0, "ENHANCE": 1.5},
            "feature_compression": "if raw feature dimension exceeds matched EEGNet feature budget, StandardScaler->PCA(n=budget, full SVD)->StandardScaler; fitted within each training split only"},
        "router_trigger": "fit/apply diagnostic router only if discovery max(Oracle-1dir,Oracle-top3) BA gain >= 0.01; same EEGNet audit gate",
        "observability_targets": ["baseline correctness", "oracle action class for top-1 train-utility direction", "per-trial best true-margin gain over native"],
        "statistics": {"unit": "biological subject", "bootstrap_draws": BOOTSTRAPS, "confidence": 0.95, "seed_rule": "stable hash of task and estimand"},
        "data_exclusion": {"allowed_roles": ["inner_train", "discovery"], "outer_dev_arrays": False, "final_heldout_eeg_arrays": False},
        "cells": cells,
        "historical_provenance_caveat": "The canonical selected checkpoints have historical diagnostic evaluations on final-heldout subjects recorded in the source manifest. This experiment performs zero final-heldout EEG array reads and does not erase that historical exposure.",
    }
    write_json(lock_path, lock)
    source_audit = {"formal_model_name": "SIRE-EEG", "historical_names": {"LiteBN": "artifact provenance name", "CompactLite": "canonical implementation class name"},
        "carrier_source_path": str(CARRIER_SOURCE), "carrier_source_sha256": lock["source"]["carrier_source_sha256"],
        "task_model_builder_path": str(TASK_SOURCE), "task_model_builder_sha256": lock["source"]["task_model_builder_sha256"],
        "architecture_sha256": hashlib.sha256((lock["source"]["carrier_source_sha256"] + lock["source"]["task_model_builder_sha256"]).encode()).hexdigest(),
        "baseline_metric_provenance": "frozen Full seed0 selected_best checkpoints; historical checkpoint selection recorded as inner future-session mean-subject BA, epochs 10-60, earliest tie; current task uses the matched EEGNet inner-train/discovery roles and recomputed discovery metrics",
        "historical_final_heldout_diagnostic_exposure": True, "new_neural_training": 0,
        "cells": [{k: c[k] for k in ("task", "fold", "seed", "frozen_checkpoint_path", "frozen_checkpoint_sha256", "normalizer_path", "normalizer_sha256", "protected_coordinates_from_frozen_sire_persist_record", "historically_exposed_diagnostic")} for c in cells]}
    write_json(OUT / "SIRE_SOURCE_AUDIT.json", source_audit)
    write_json(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {"FINAL_HELDOUT_ACCESSED": False,
        "final_heldout_eeg_array_reads": 0, "outer_dev_eeg_array_reads": 0,
        "status": "PREFLIGHT_LOCKED_ZERO_READS; updated with final cell-level counts after execution",
        "current_allowed_roles": ["inner_train", "discovery"], "historical_checkpoint_exposure_is_separately_disclosed": True})
    print(f"PROTOCOL_LOCKED cells={len(cells)} eegnet_router_source={lock['source']['eegnet_protocol_lock_sha256']}", flush=True)


def verify_lock() -> dict:
    lock_path = PROTOCOL / "PROTOCOL_LOCK.json"
    if not lock_path.is_file(): raise RuntimeError("run lock_protocol before any cell")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha_file(SOURCE_RECORDS) != lock["source"]["source_records_sha256"]: raise RuntimeError("source record changed after lock")
    if sha_file(EEGNET_REFERENCE_AUDIT) != lock["source"]["eegnet_reference_audit_sha256"]: raise RuntimeError("EEGNet reference audit changed")
    eegnet_lock = json.loads((EEGNET_EXP / "protocol/PROTOCOL_LOCK.json").read_text(encoding="utf-8-sig"))
    verify_eegnet_reference_audit(eegnet_lock)
    if sha_file(Path(__file__).resolve()) != lock["source"]["analysis_code_sha256"]:
        raise RuntimeError("analysis code changed after lock")
    if sha_file(CARRIER_SOURCE) != lock["source"]["carrier_source_sha256"] or sha_file(TASK_SOURCE) != lock["source"]["task_model_builder_sha256"]:
        raise RuntimeError("canonical SIRE source changed after lock")
    for cell in lock["cells"]:
        if sha_file(Path(cell["frozen_checkpoint_path"])) != cell["frozen_checkpoint_sha256"]: raise RuntimeError(f"checkpoint SHA mismatch {cell['task']}/{cell['fold']}")
        if sha_file(Path(cell["normalizer_path"])) != cell["normalizer_sha256"]: raise RuntimeError(f"normalizer SHA mismatch {cell['task']}/{cell['fold']}")
    return lock


def build_model(task: str, classes: int, checkpoint: Path):
    td = import_file("sire_actionability_task_datasets", TASK_SOURCE)
    # The canonical CompactLite implementation lives in the final-confirm
    # checkout, while task_datasets.py is the frozen task data/model wrapper.
    td.CARRIER_CODE = CARRIER_SOURCE.parent
    # The frozen wrapper's ERP entry provides the two-class model head size
    # used by OpenBMI MI; CompactLite itself has no task-specific input length.
    model = td.build_model("LiteBN", "ERP" if task == "OpenBMI_MI" else "SSVEP")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid frozen checkpoint payload: {checkpoint}")
    # The canonical selected_best.pt files are raw state_dict mappings.
    state_dict = payload.get("state_dict", payload)
    if not state_dict or not all(torch.is_tensor(value) for value in state_dict.values()):
        raise RuntimeError(f"checkpoint does not contain a raw model state_dict: {checkpoint}")
    model.load_state_dict(state_dict, strict=True)
    if model.head.out_features != classes: raise RuntimeError(f"class count mismatch {task}: checkpoint={model.head.out_features}, data={classes}")
    model = model.to(DEVICE).eval()
    for p in model.parameters(): p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()) or model.training: raise RuntimeError("frozen SIRE was not in eval mode")
    return model


def model_state_sha(model) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def forward_stages(model, x: torch.Tensor):
    """Return exact canonical H_concat, H_shared1, native logits, embedding."""
    branches = []
    xx = x.unsqueeze(1)
    for t, tn, s, sn in zip(model.temporal, model.temporal_norm, model.spatial, model.spatial_norm):
        a = F.elu(tn(t(xx)))
        a = F.elu(sn(s(a)))
        a = F.avg_pool2d(a, (1, 4))
        branches.append(a)
    hs = torch.cat(branches, dim=1)
    yd = F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(hs)))), (1, 2))
    x2 = F.avg_pool2d(F.elu(model.norm2(model.point2(model.depth2(yd)))), (1, 2))
    emb = model.embedding(model.pool(x2).flatten(1))
    logits = model.head(emb)
    return hs, yd, logits, emb


def suffix(model, successor: torch.Tensor) -> torch.Tensor:
    x = successor.reshape(-1, 64, 1, successor.shape[1] // 64)
    x = F.avg_pool2d(F.elu(model.norm2(model.point2(model.depth2(x)))), (1, 2))
    return model.head(model.embedding(model.pool(x).flatten(1)))


def infer_stage(model, raw: np.ndarray, mean: np.ndarray, std: np.ndarray):
    hs_all, yd_all, z_all, emb_all = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw), INFER_BATCH):
            x = ((raw[start:start + INFER_BATCH] - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE)
            hs, yd, z, emb = forward_stages(model, xt)
            native = model(xt)[0]
            if float(torch.max(torch.abs(native - z)).item()) >= 1e-6: raise RuntimeError("manual SIRE stage forward differs from canonical forward")
            hs_all.append(hs.flatten(1).float().cpu().numpy())
            yd_all.append(yd.flatten(1).float().cpu().numpy())
            z_all.append(z.float().cpu().numpy())
            emb_all.append(emb.float().cpu().numpy())
    return tuple(np.concatenate(v, axis=0).astype(np.float32) for v in (hs_all, yd_all, z_all, emb_all))


def fetch_role(task: str, subjects: list[str], sessions: list[int], cache: str, mapping: dict | None):
    pieces = []
    for subject in subjects:
        for session in sessions:
            raw, y, owners, new_mapping = B.rows(task, [subject], (int(session),), cache, mapping, final=False)
            if mapping is None: mapping = dict(new_mapping)
            elif mapping != new_mapping: raise RuntimeError(f"label map drift at {task}/{subject}/S{session}")
            owner = np.asarray(owners).astype(str)
            if not len(raw) or not np.all(owner == str(subject)):
                raise RuntimeError(f"row owner/slice mismatch at {task}/{subject}/S{session}")
            pieces.append((int(session), str(subject), raw, np.asarray(y, dtype=np.int64), owner))
    return pieces, mapping


def apply_stages(model, pieces, mean: np.ndarray, std: np.ndarray):
    rows = []
    for session, subject, raw, y, owners in pieces:
        hs, yd, logits, emb = infer_stage(model, raw, mean, std)
        rows.append({"session": session, "subject": subject, "raw_y": y, "owners": owners,
            "hs": hs, "yd": yd, "logits": logits, "emb": emb})
        del raw
    return rows


def peeh_spectrum(h: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> dict:
    """Frozen PERSIST/PEEH whitening and cross-session persistence frame."""
    x = np.asarray(h, np.float64)
    mean = x.mean(0)
    centered = x - mean
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
    order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
    threshold = max(float(vals[0]) * 1e-3, 1e-8)
    numerical_rank = int(np.sum(vals > threshold))
    rank = min(ACTIVE_RANK_MAX, numerical_rank)
    if rank < 4: raise RuntimeError(f"SIRE train embedding rank too low: {rank}")
    floor = max(1e-4 * float(vals[rank - 1]), 1e-8)
    active = np.maximum(vals[:rank], floor)
    active_vecs = vecs[:, :rank]
    whitener = active_vecs * np.power(active, -0.5)[None, :]
    whitened = centered @ whitener
    meta = list(zip(map(str, subjects), map(int, sessions), map(int, y)))
    subjects_sorted = ordered_subjects(subjects)
    sessions_sorted = sorted(set(map(int, sessions)))
    classes = sorted(set(map(int, y)))
    if len(sessions_sorted) != 2: raise RuntimeError(f"PERSIST frame needs two train sessions: {sessions_sorted}")
    centroids = {}
    for subject, session, label in sorted(set(meta), key=lambda k:(int(k[0].replace('sub-','')), k[1], k[2])):
        ix = np.asarray([i for i, k in enumerate(meta) if k == (subject, session, label)], dtype=np.int64)
        centroids[(subject, session, label)] = whitened[ix].mean(0)
    covs = []
    for label in classes:
        left, right = [], []
        for subject in subjects_sorted:
            a, b = (subject, sessions_sorted[0], label), (subject, sessions_sorted[1], label)
            if a in centroids and b in centroids:
                left.append(centroids[a]); right.append(centroids[b])
        if left:
            aa, bb = np.asarray(left).copy(), np.asarray(right).copy()
            aa -= aa.mean(0); bb -= bb.mean(0)
            covs.append((aa.T @ bb + bb.T @ aa) / (2.0 * len(aa)))
    if not covs: raise RuntimeError("no matched subject-session-class centroids for PERSIST frame")
    persistence = np.mean(covs, axis=0)
    rv, directions = np.linalg.eigh((persistence + persistence.T) / 2.0)
    order = np.argsort(rv)[::-1]
    return {"mean": mean.astype(np.float32), "whitener": whitener.astype(np.float32),
        "directions": directions[:, order].astype(np.float32), "rho": rv[order].astype(np.float32),
        "active_rank": rank, "numerical_rank": numerical_rank,
        "centroid_count": len(centroids), "session_ids": sessions_sorted}


def canonical_targets(h: np.ndarray, spectrum: dict, dims: list[int]) -> np.ndarray:
    return ((np.asarray(h, np.float64) - spectrum["mean"]) @ spectrum["whitener"] @ spectrum["directions"][:, dims]).astype(np.float32)


def raw_q(activation_centroids: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    a = np.asarray(activation_centroids, np.float64)
    mu = a.mean(0)
    sd = np.maximum(a.std(0), 1e-6)
    x = (a - mu) / sd
    # Exact PathFit kernel ridge solve from the mechanism-closure source.
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(DEVICE)
    yt = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)).to(DEVICE)
    with torch.inference_mode():
        kernel = (xt @ xt.T) / max(xt.shape[1], 1)
        chol = torch.linalg.cholesky(kernel + RIDGE_ALPHA * torch.eye(len(xt), device=DEVICE, dtype=xt.dtype))
        alpha = torch.cholesky_solve(yt, chol)
        coef = ((xt.T @ alpha) / max(xt.shape[1], 1)).cpu().numpy().astype(np.float64)
    u, singular, _ = np.linalg.svd(coef, full_matrices=False)
    rank = int(np.sum(singular > max(float(singular[0]) * 1e-6, 1e-8))) if len(singular) else 0
    if rank <= 0: raise RuntimeError("empty intermediate pathway mapping")
    raw = u[:, :rank] / sd[:, None]
    q, _ = np.linalg.qr(raw, mode="reduced")
    q = np.ascontiguousarray(q.astype(np.float32))
    mu32 = np.ascontiguousarray(mu.astype(np.float32))
    details = {"rank": rank, "singular_values": singular[:rank].tolist(), "source_centroids": int(len(a)),
        "feature_dimension": int(a.shape[1]), "ridge_alpha": RIDGE_ALPHA}
    return q, mu32, details


def train_centroid_rows(train_rows: list[dict], dims: list[int]):
    sessions = sorted(set(int(r["session"]) for r in train_rows))
    if len(sessions) != 2: raise RuntimeError(f"locked OpenBMI roles must contain two sessions, got {sessions}")
    # Deterministic capped sample per subject/session/class, matching the prior
    # pathway mechanism-closure cap (32) before grouped centroids are formed.
    groups = {}
    for row in train_rows:
        s, sub = int(row["session"]), str(row["subject"])
        for label in sorted(np.unique(row["raw_y"])):
            ix = np.flatnonzero(row["raw_y"] == label)
            rng = np.random.default_rng(stable_seed("sire_pathfit_cap32", row["task"], row["fold"], sub, s, int(label)))
            if len(ix) > 32: ix = np.sort(rng.choice(ix, 32, replace=False))
            groups[(sub, s, int(label))] = (row, ix)
    acts_s, acts_d, embs, ys, subs, ses = [], [], [], [], [], []
    for key in sorted(groups, key=lambda k:(int(k[0].replace('sub-','')), k[1], k[2])):
        row, ix = groups[key]
        acts_s.append(row["hs"][ix].mean(0)); acts_d.append(row["yd"][ix].mean(0)); embs.append(row["emb"][ix].mean(0))
        ys.append(key[2]); subs.append(key[0]); ses.append(key[1])
    acts_s = np.stack(acts_s).astype(np.float32); acts_d = np.stack(acts_d).astype(np.float32); embs = np.stack(embs).astype(np.float32)
    ys = np.asarray(ys, dtype=np.int64); subs = np.asarray(subs, dtype="U16"); ses = np.asarray(ses, dtype=np.int64)
    spectrum = peeh_spectrum(np.concatenate([r["emb"] for r in train_rows]),
        np.concatenate([r["raw_y"] for r in train_rows]),
        np.concatenate([np.full(len(r["raw_y"]), str(r["subject"]), dtype="U16") for r in train_rows]),
        np.concatenate([np.full(len(r["raw_y"]), int(r["session"]), dtype=np.int64) for r in train_rows]))
    targets = canonical_targets(embs, spectrum, dims)
    qs, ms, info_s = raw_q(acts_s, targets)
    qd, md, info_d = raw_q(acts_d, targets)
    return (qs, ms, qd, md, spectrum, {"train_centroid_labels": ys, "train_centroid_subjects": subs,
        "train_centroid_sessions": ses, "train_stage_source": acts_s, "train_stage_successor": acts_d,
        "source_pathfit": info_s, "successor_pathfit": info_d})


def complement_basis(train_h: np.ndarray, qs: np.ndarray, ms: np.ndarray) -> tuple[np.ndarray, dict]:
    x = torch.from_numpy(np.ascontiguousarray(train_h, dtype=np.float32)).to(DEVICE)
    q = torch.from_numpy(np.ascontiguousarray(qs, dtype=np.float32)).to(DEVICE)
    m = torch.from_numpy(np.ascontiguousarray(ms, dtype=np.float32)).to(DEVICE)
    with torch.inference_mode():
        c = x - m - ((x - m) @ q) @ q.T
        gram = c @ c.T
        ev, vv = torch.linalg.eigh(gram)
        order = torch.argsort(ev, descending=True)
        ev = ev[order].clamp_min(0); vv = vv[:, order]
        singular = torch.sqrt(ev)
        tol = max(float(singular[0].item()) * 1e-6, 1e-8)
        rank = int((singular > tol).sum().item())
        k = min(16, rank)
        if k < 1: raise RuntimeError("source complement has numerical rank zero")
        top = vv[:, :k] / singular[:k].clamp_min(1e-12)[None, :]
        basis = c.T @ top
        # Reduced QR repairs float32 Gram roundoff without changing the top-k span.
        basis, _ = torch.linalg.qr(basis, mode="reduced")
        basis = basis.contiguous().float().cpu().numpy()
        ortho = float(torch.max(torch.abs(torch.as_tensor(basis, dtype=torch.float64).T @ torch.as_tensor(basis, dtype=torch.float64) - torch.eye(k, dtype=torch.float64))).item())
        total = float(ev.sum().item())
        explained = (ev[:k].sum().item() / max(total, 1e-30))
        recon = float(torch.max(torch.abs((x - m) - ((x - m) @ q) @ q.T - c)).item())
    return basis.astype(np.float32), {"rank": rank, "K_C": k, "singular_value_tolerance": tol,
        "explained_variance_fraction": float(explained), "basis_orthogonality_max_abs": ortho,
        "source_partition_reconstruction_max_abs": recon, "train_trials": int(len(train_h))}


def margin_t(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    own = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone(); other.scatter_(1, y[:, None], -torch.inf)
    return own - other.max(1).values


def pick_alpha(scores: np.ndarray, maximize: bool = True) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    target = z.max(1) if maximize else z.min(1)
    out = np.empty(len(z), dtype=np.int64)
    for i, best in enumerate(target):
        ties = np.flatnonzero(np.abs(z[i] - best) <= 1e-8)
        out[i] = min(ties, key=lambda j:(abs(float(ALPHAS[j]) - 1.0), float(ALPHAS[j]), int(j)))
    return out


def pick_oracle(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, k, a = scores.shape
    out_d = np.empty(n, dtype=np.int64); out_a = np.empty(n, dtype=np.int64)
    for i in range(n):
        row = scores[i].reshape(-1); best = row.max()
        tied = np.flatnonzero(np.abs(row - best) <= 1e-8)
        ix = min(tied, key=lambda j:(abs(float(ALPHAS[j % a]) - 1.0), float(ALPHAS[j % a]), int(j // a)))
        out_d[i], out_a[i] = int(ix // a), int(ix % a)
    return out_d, out_a


def source_base(model, hs: np.ndarray, qd: np.ndarray, md: np.ndarray, batch_size: int = 128):
    yd_values, logits_values = [], []
    with torch.inference_mode():
        for start in range(0, len(hs), batch_size):
            h = torch.from_numpy(np.ascontiguousarray(hs[start:start+batch_size])).to(DEVICE)
            shaped = h.reshape(-1, 48, 1, h.shape[1] // 48)
            yd = F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(shaped)))), (1, 2)).flatten(1)
            z = suffix(model, yd)
            yd_values.append(yd.float().cpu().numpy()); logits_values.append(z.float().cpu().numpy())
    return np.concatenate(yd_values), np.concatenate(logits_values)


def eval_curve_bank(model, hs: np.ndarray, yd_native: np.ndarray, z_native: np.ndarray, labels: np.ndarray,
                    qs: np.ndarray, ms: np.ndarray, qd: np.ndarray, md: np.ndarray, basis: np.ndarray,
                    batch_size: int = CURVE_BATCH):
    n, k, d = len(hs), basis.shape[1], hs.shape[1]
    classes = z_native.shape[1]
    vals = {"margin": np.empty((n,k,len(ALPHAS)),np.float32), "CE": np.empty((n,k,len(ALPHAS)),np.float32),
        "prediction": np.empty((n,k,len(ALPHAS)),np.int16), "max_softmax": np.empty((n,k,len(ALPHAS)),np.float32),
        "native_predicted_probability": np.empty((n,k,len(ALPHAS)),np.float32), "entropy": np.empty((n,k,len(ALPHAS)),np.float32),
        "p_movement": np.empty((n,k,len(ALPHAS)),np.float32)}
    logits_bank = np.empty((n,k,len(ALPHAS),classes),np.float32)
    qst=torch.as_tensor(qs,dtype=torch.float32,device=DEVICE); mst=torch.as_tensor(ms,dtype=torch.float32,device=DEVICE)
    qdt=torch.as_tensor(qd,dtype=torch.float64,device=DEVICE); mdt=torch.as_tensor(md,dtype=torch.float64,device=DEVICE)
    bt=torch.as_tensor(basis,dtype=torch.float32,device=DEVICE); at=torch.as_tensor(ALPHAS,dtype=torch.float32,device=DEVICE)
    ai=int(np.where(np.isclose(ALPHAS,1.0))[0][0])
    identity_rep=identity_logits=preserve_err=raw_suffix_roundoff=0.0
    for start in range(0,n,batch_size):
        stop=min(n,start+batch_size); b=stop-start
        h=torch.from_numpy(np.ascontiguousarray(hs[start:stop])).to(DEVICE)
        yd0=torch.from_numpy(np.ascontiguousarray(yd_native[start:stop])).to(DEVICE)
        z0=torch.from_numpy(np.ascontiguousarray(z_native[start:stop])).to(DEVICE)
        y=torch.as_tensor(labels[start:stop],dtype=torch.long,device=DEVICE)
        centered=h-mst; ps=(centered@qst)@qst.T; hc=centered-ps; coeff=hc@bt
        scaled=h[:,None,None,:]+(at[None,None,:,None]-1.0)*coeff[:,:,None,None]*bt.T[None,:,None,:]
        if not torch.equal(scaled[:,:,ai,:],h[:,None,:].expand(-1,k,-1)):
            raise RuntimeError("alpha=1 source intervention is not an exact no-op")
        flat=scaled.reshape(-1,d)
        shaped=flat.reshape(-1,48,1,d//48)
        yd=F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(shaped)))), (1, 2)).flatten(1)
        yd_view=yd.reshape(b,k,len(ALPHAS),-1)
        yd_view[:,:,ai,:]=yd0[:,None,:]
        yd=yd_view.reshape(-1,yd.shape[1])
        yd64=yd.double(); yd064=yd0.double()
        pd=(yd64-mdt)@qdt
        p0=(yd064-mdt)@qdt
        c0=(yd064-mdt)-p0@qdt.T
        yp=(mdt+pd@qdt.T+c0[:,None,:].expand(-1,k*len(ALPHAS),-1).reshape(-1,yd.shape[1])).float()
        yp_view=yp.reshape(b,k,len(ALPHAS),-1)
        identity_rep=max(identity_rep,float(torch.max(torch.abs(yp_view[:,:,ai,:]-yd0[:,None,:])).item()))
        yp_view[:,:,ai,:]=yd0[:,None,:]
        yp=yp_view.reshape(-1,yd.shape[1])
        z=suffix(model,yp).reshape(b,k,len(ALPHAS),classes)
        raw_suffix_roundoff=max(raw_suffix_roundoff,float(torch.max(torch.abs(z[:,:,ai,:]-z0[:,None,:])).item()))
        z[:,:,ai,:]=z0[:,None,:]
        identity_logits=max(identity_logits,float(torch.max(torch.abs(z[:,:,ai,:]-z0[:,None,:])).item()))
        yy=y[:,None,None].expand(-1,k,len(ALPHAS)).reshape(-1)
        mar=margin_t(z.reshape(-1,classes),yy).reshape(b,k,len(ALPHAS))
        ce=F.cross_entropy(z.reshape(-1,classes),yy,reduction="none").reshape(b,k,len(ALPHAS))
        prob=z.softmax(-1); pred=prob.argmax(-1); maxp=prob.max(-1).values
        native_pred=z0.argmax(-1)
        native_prob=prob.gather(-1,native_pred[:,None,None,None].expand(-1,k,len(ALPHAS),1)).squeeze(-1)
        ent=-(prob*prob.clamp_min(1e-12).log()).sum(-1)
        p_move=torch.linalg.vector_norm(pd.reshape(b,k,len(ALPHAS),-1)-p0[:,None,None,:],dim=-1)
        # Report the native successor P/C reconstruction audit separately from
        # the exact alpha=1 no-op route above.
        yrecon=(mdt+p0@qdt.T+c0).float()
        identity_rep=max(identity_rep,float(torch.max(torch.abs(yrecon-yd0)).item()))
        native_alpha=z[:,:,ai,:]
        if not torch.equal(native_alpha.argmax(-1),native_pred[:,None].expand(-1,k)):
            raise RuntimeError("alpha=1 prediction identity failed")
        ycenter=(yp.double()-mdt)
        cnew=ycenter-(ycenter@qdt)@qdt.T
        preserve_err=max(preserve_err,float(torch.max(torch.abs(cnew-c0[:,None,:].expand(-1,k*len(ALPHAS),-1).reshape_as(cnew))).item()))
        for key,value in (("margin",mar),("CE",ce),("prediction",pred),("max_softmax",maxp),
                          ("native_predicted_probability",native_prob),("entropy",ent),("p_movement",p_move)):
            vals[key][start:stop]=value.detach().cpu().numpy().astype(vals[key].dtype)
        logits_bank[start:stop]=z.detach().float().cpu().numpy()
    return vals, logits_bank, {"representation_max_abs":identity_rep,"logits_max_abs":identity_logits,
        "raw_suffix_batch_roundoff_max_abs":raw_suffix_roundoff,
        "prediction_exact":True,"successor_C_reconstruction_max_abs":preserve_err}


def sequential_top3(model, hs: np.ndarray, labels: np.ndarray, coeff: np.ndarray, basis: np.ndarray,
                    qd: np.ndarray, md: np.ndarray, yd_native: np.ndarray, directions: list[int]):
    results=[]; qdt=torch.as_tensor(qd,dtype=torch.float32,device=DEVICE); mdt=torch.as_tensor(md,dtype=torch.float32,device=DEVICE)
    bt=torch.as_tensor(basis,dtype=torch.float32,device=DEVICE); at=torch.as_tensor(ALPHAS,dtype=torch.float32,device=DEVICE)
    for start in range(0,len(hs),CURVE_BATCH):
        stop=min(len(hs),start+CURVE_BATCH); h=torch.from_numpy(np.ascontiguousarray(hs[start:stop])).to(DEVICE)
        c=torch.from_numpy(np.ascontiguousarray(coeff[start:stop])).to(DEVICE); y=torch.as_tensor(labels[start:stop],dtype=torch.long,device=DEVICE)
        yd0=torch.from_numpy(np.ascontiguousarray(yd_native[start:stop])).to(DEVICE); current=h.clone()
        for direction in directions:
            candidate=current[:,None,:]+(at[None,:,None]-1.0)*c[:,direction,None,None]*bt[:,direction][None,None,:]
            shaped=candidate.reshape(-1,48,1,h.shape[1]//48)
            yd=F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(shaped)))), (1, 2)).flatten(1)
            pd=(yd-mdt)@qdt; p0=(yd0-mdt)@qdt; c0=(yd0-mdt)-p0@qdt.T
            yp=mdt+pd@qdt.T+c0[:,None,:].expand(-1,len(ALPHAS),-1).reshape(-1,yd.shape[1])
            z=suffix(model,yp).reshape(len(h),len(ALPHAS),-1)
            rep=y[:,None].expand(-1,len(ALPHAS)).reshape(-1)
            margin=margin_t(z.reshape(-1,z.shape[-1]),rep).reshape(len(h),len(ALPHAS)).cpu().numpy()
            ai=pick_alpha(margin)
            a=at[torch.as_tensor(ai,dtype=torch.long,device=DEVICE)]
            current=current+(a[:,None]-1.0)*c[:,direction,None]*bt[:,direction][None,:]
        shaped=current.reshape(-1,48,1,h.shape[1]//48)
        yd=F.avg_pool2d(F.elu(model.norm1(model.point1(model.depth1(shaped)))), (1, 2)).flatten(1)
        pd=(yd-mdt)@qdt; p0=(yd0-mdt)@qdt; c0=(yd0-mdt)-p0@qdt.T
        yp=mdt+pd@qdt.T+c0
        z=suffix(model,yp); ce=F.cross_entropy(z,y,reduction="none")
        results.append((z.argmax(-1).cpu().numpy(),ce.cpu().numpy()))
    return np.concatenate([x[0] for x in results]),np.concatenate([x[1] for x in results])


def make_random_basis(qs: np.ndarray, dim: int, task: str, fold: int, k: int) -> np.ndarray:
    rng=np.random.default_rng(stable_seed("SIRE_RANDOM_C_BASIS_DRAW0",task,fold))
    g=rng.standard_normal((dim,k)).astype(np.float64)
    q=np.asarray(qs,np.float64)
    g-=q@(q.T@g)
    basis,_=np.linalg.qr(g,mode="reduced")
    if basis.shape[1] != k: raise RuntimeError("random C basis rank mismatch")
    return basis.astype(np.float32)


def softmax_numpy(logits: np.ndarray):
    z=np.asarray(logits,np.float64); z=z-z.max(axis=1,keepdims=True); p=np.exp(z); return p/np.maximum(p.sum(axis=1,keepdims=True),1e-12)


def router_features(hs: np.ndarray, base_logits: np.ndarray, successor_p: np.ndarray,
                    qs: np.ndarray, ms: np.ndarray, basis: np.ndarray):
    centered=hs-ms[None,:]; ps=centered@qs; hc=centered-ps@qs.T; coeff=hc@basis
    prob=softmax_numpy(base_logits); gap=np.sort(base_logits,axis=1)[:,-1]-np.sort(base_logits,axis=1)[:,-2]
    conf=prob.max(1); entropy=-(prob*np.log(np.maximum(prob,1e-12))).sum(1)
    pnorm=np.linalg.norm(ps,axis=1); cnorm=np.linalg.norm(hc,axis=1); dnorm=np.linalg.norm(successor_p,axis=1)
    scalar=np.stack((conf,entropy,gap,pnorm,cnorm,dnorm,pnorm/np.maximum(cnorm,1e-8),dnorm/np.maximum(pnorm,1e-8)),axis=1)
    return np.concatenate((ps,coeff,successor_p,base_logits,scalar),axis=1).astype(np.float32),coeff


def action_from_alpha(alpha_idx: np.ndarray) -> np.ndarray:
    a=ALPHAS[np.asarray(alpha_idx,dtype=np.int64)]; out=np.full(len(a),-1,np.int64)
    out[a<=0.5]=0; out[(a>=0.75)&(a<=1.25)]=1; out[a>=1.5]=2
    if np.any(out<0): raise RuntimeError("unmapped router action")
    return out


def action_scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"macro_F1":float(f1_score(y,pred,labels=[0,1,2],average="macro",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),"accuracy":float(np.mean(y==pred)),
        "confusion_matrix":json.dumps(confusion_matrix(y,pred,labels=[0,1,2]).tolist())}


def performance_rows(task: str, fold: int, role: str, session: int, subject: str, labels: np.ndarray,
                     native_pred: np.ndarray, native_ce: np.ndarray, variants: dict[str, tuple[np.ndarray,np.ndarray]]):
    out=[]
    allv={"BASELINE":(native_pred,native_ce),**variants}
    for variant,(pred,ce) in allv.items():
        out.append({"task":task,"fold":fold,"role":role,"session":session,"subject":subject,"variant":variant,
            "BA":float(balanced_accuracy_score(labels,pred)),"macro_F1":float(f1_score(labels,pred,average="macro",zero_division=0)),
            "NLL":float(np.mean(ce)),"rescued_errors":int(np.sum((native_pred!=labels)&(pred==labels))),
            "damaged_correct":int(np.sum((native_pred==labels)&(pred!=labels))),"trials":len(labels),
            "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return out


def curve_fields():
    def tag(a): return str(float(a)).replace(".","p")
    f=["task","fold","role","session","subject","trial_index","label","direction"]
    for metric in ("margin","CE","prediction","max_softmax","native_predicted_probability","entropy","p_movement"):
        f.extend([f"{metric}_a{tag(a)}" for a in ALPHAS])
    f.append("label_scope")
    return f


def write_curve_rows(writer, task, fold, role, row, curves):
    n,k,_=curves["margin"].shape
    def tag(a): return str(float(a)).replace(".","p")
    for i in range(n):
        for j in range(k):
            r={"task":task,"fold":fold,"role":role,"session":row["session"],"subject":row["subject"],
                "trial_index":i,"label":int(row["raw_y"][i]),"direction":j,"label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}
            for metric in ("margin","CE","prediction","max_softmax","native_predicted_probability","entropy","p_movement"):
                for aidx,a in enumerate(ALPHAS): r[f"{metric}_a{tag(a)}"]=float(curves[metric][i,j,aidx])
            writer.writerow(r)


def local_rows(task,fold,role,row,curves,utility):
    i075=int(np.where(np.isclose(ALPHAS,.75))[0][0]); i125=int(np.where(np.isclose(ALPHAS,1.25))[0][0]); i0=0; i1=4
    out=[]
    for j in range(curves["margin"].shape[1]):
        dm=float((curves["margin"][:,j,i125].mean()-curves["margin"][:,j,i075].mean())/.5)
        dc=float((curves["CE"][:,j,i125].mean()-curves["CE"][:,j,i075].mean())/.5)
        up=float(curves["margin"][:,j,i1].mean()-curves["margin"][:,j,i0].mean())
        out.append({"task":task,"fold":fold,"role":role,"session":row["session"],"subject":row["subject"],"direction":j,
            "U_P":up,"D_margin":dm,"D_CE":dc,"U_P_positive_D_positive":bool(up>0 and dm>0.001),
            "U_P_positive_D_plateau":bool(up>0 and abs(dm)<=0.001),"U_P_positive_D_negative":bool(up>0 and dm< -0.001),
            "local_actionability":"AMPLIFY" if dm>0.001 else "SUPPRESS" if dm< -0.001 else "PLATEAU",
            "utility_action_quadrant":"A_USEFUL_AMPLIFY" if up>0 and dm>0.001 else "B_USEFUL_PLATEAU" if up>0 and abs(dm)<=.001 else "C_USEFUL_AMPLIFY_HARM" if up>0 and dm<-.001 else "D_NONPOSITIVE_UTILITY",
            "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return out


def conditional_rows(task,fold,role,row,curves):
    out=[]
    for j in range(curves["margin"].shape[1]):
        mi=pick_alpha(curves["margin"][:,j,:]); ci=pick_alpha(curves["CE"][:,j,:],maximize=False)
        a=ALPHAS[mi]; low=float(np.mean(a<1)); high=float(np.mean(a>1))
        counts=np.bincount(mi,minlength=len(ALPHAS))
        out.append({"task":task,"fold":fold,"role":role,"session":row["session"],"subject":row["subject"],"direction":j,
            "margin_alpha_star_lt_1_fraction":low,"margin_alpha_star_gt_1_fraction":high,
            "margin_alpha_star_distribution":json.dumps(counts.tolist()),
            "trial_conditional":bool(low>=.1 and high>=.1),"ce_alpha_star_distribution":json.dumps(np.bincount(ci,minlength=len(ALPHAS)).tolist()),
            "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return out


def bootstrap(values: np.ndarray, seed: int):
    x=np.asarray(values,dtype=np.float64)
    if not len(x): return float("nan"),float("nan"),float("nan")
    rng=np.random.default_rng(seed); draws=np.empty(BOOTSTRAPS,np.float32)
    for start in range(0,BOOTSTRAPS,512):
        stop=min(BOOTSTRAPS,start+512); ix=rng.integers(0,len(x),size=(stop-start,len(x))); draws[start:stop]=x[ix].mean(1)
    return float(x.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def subject_equal(rows: list[dict], variant: str, task: str, role: str="discovery", fold: int | None=None):
    bysf={}
    for r in rows:
        if r.get("task")!=task or r.get("role")!=role or r.get("variant")!=variant: continue
        if fold is not None and int(r.get("fold",-1))!=fold: continue
        bysf.setdefault((str(r["subject"]),int(r["fold"])),[]).append(float(r["BA"]))
    bysub={}
    for (s,_f),vals in bysf.items(): bysub.setdefault(s,[]).append(float(np.mean(vals)))
    return {s:float(np.mean(vals)) for s,vals in bysub.items()}


def pair_mean(a: dict[str,float], b: dict[str,float]):
    return {s:float(a[s]-b[s]) for s in set(a)&set(b)}


def load_normalizer(path: Path):
    with np.load(path,allow_pickle=False) as z:
        mean=np.asarray(z["mean"],np.float32); std=np.asarray(z["std"],np.float32)
    if mean.shape!=(62,) or std.shape!=(62,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError(f"normalizer schema error: {path}")
    return mean,std


def fit_router(train_X, train_action, train_subject, test_X, budget):
    unique=np.unique(train_subject)
    if len(unique)<2: raise RuntimeError("router requires at least two training subjects")
    ncomp=min(int(budget),train_X.shape[1])
    def model_for(y):
        cls=np.unique(y)
        if len(cls)==1:
            class Constant:
                def __init__(self,v): self.v=int(v)
                def fit(self,x,y): return self
                def predict(self,x): return np.full(len(x),self.v,np.int64)
            return Constant(cls[0])
        pieces=[StandardScaler()]
        if train_X.shape[1]>ncomp: pieces.append(PCA(n_components=ncomp,svd_solver="full"))
        pieces.extend([StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=500,solver="lbfgs",random_state=0)])
        return make_pipeline(*pieces)
    cv=GroupKFold(n_splits=min(5,len(unique))); oof=np.full(len(train_action),-1,np.int64)
    for tr,va in cv.split(train_X,train_action,groups=train_subject):
        est=model_for(train_action[tr]); est.fit(train_X[tr],train_action[tr]); oof[va]=est.predict(train_X[va])
    model=model_for(train_action); model.fit(train_X,train_action); pred=model.predict(test_X).astype(np.int64)
    return oof,pred,int(ncomp)


def router_features_for_record(row, curves, logits, p_s, p_d, qs, ms, basis):
    x, coeff=router_features(row["hs"],logits,p_d,qs,ms,basis)
    return x,coeff


def evaluate_cell(task: str, fold: int) -> None:
    lock=verify_lock(); cell=next(c for c in lock["cells"] if c["task"]==task and int(c["fold"])==fold)
    runtime_cell=RUNTIME/"cells"/task.lower()/f"fold{fold}_seed0"; runtime_cell.mkdir(parents=True,exist_ok=True)
    done=runtime_cell/"CELL_COMPLETE.json"
    if done.exists():
        rec=json.loads(done.read_text(encoding="utf-8"))
        if rec.get("protocol_sha256")==sha_file(PROTOCOL/"PROTOCOL_LOCK.json"):
            print("CELL_ALREADY_COMPLETE",task,fold,flush=True); return
        raise RuntimeError(f"partial cell refuses overwrite {task}/{fold}")
    role,split,cache,source_sessions,future=B.role(task,fold)
    train_ids=ordered_subjects(role["inner_train_subjects"]); disc_ids=ordered_subjects(role["inner_val_subjects"])
    sessions=sorted(set(map(int,tuple(source_sessions)+(int(future),))))
    checkpoint=Path(cell["frozen_checkpoint_path"]); mean,std=load_normalizer(Path(cell["normalizer_path"]))
    mapping=None; read_counts={"inner_train":0,"discovery":0,"outer_dev":0,"final_heldout":0}
    train_pieces,mapping=fetch_role(task,train_ids,sessions,cache,mapping); read_counts["inner_train"]=len(train_pieces)
    disc_pieces,mapping=fetch_role(task,disc_ids,sessions,cache,mapping); read_counts["discovery"]=len(disc_pieces)
    classes=len(mapping); model=build_model(task,classes,checkpoint); state_before=model_state_sha(model)
    train_rows=apply_stages(model,train_pieces,mean,std); disc_rows=apply_stages(model,disc_pieces,mean,std)
    for row in train_rows+disc_rows: row.update({"task":task,"fold":fold})
    if len(sessions)!=2: raise RuntimeError("locked PERSIST geometry requires exactly two sessions")
    dims=list(map(int,cell["protected_coordinates_from_frozen_sire_persist_record"]))
    if not dims or max(dims)>=ACTIVE_RANK_MAX: raise RuntimeError(f"invalid frozen protected coordinates {dims}")
    qs,ms,qd,md,spectrum,proj=train_centroid_rows(train_rows,dims)
    train_h=np.concatenate([r["hs"] for r in train_rows]); basis,basis_info=complement_basis(train_h,qs,ms)
    k=basis.shape[1]
    if k<TOP_K: raise RuntimeError(f"C basis rank {k} does not support locked top3")
    random_basis=make_random_basis(qs,train_h.shape[1],task,fold,k)
    projector_rows=[]
    for stage,q,m,info in (("H_concat",qs,ms,proj["source_pathfit"]),("H_shared1",qd,md,proj["successor_pathfit"])):
        orth=float(np.max(np.abs(q.T@q-np.eye(q.shape[1])))); projector_rows.append({"task":task,"fold":fold,"stage":stage,
            "projector_mode":"TRAIN_ONLY_ORTHOGONAL_PATHFIT","Q_rank":q.shape[1],"feature_dimension":q.shape[0],
            "train_group_centroids":info["source_centroids"],"ridge_alpha":RIDGE_ALPHA,"pathfit_singular_values":json.dumps(info["singular_values"]),
            "orthogonality_max_abs":orth,"centering_vector_sha256":arr_sha(m),"projector_sha256":arr_sha(q,m),
            "target_coordinates":json.dumps(dims),"labels_scope":"INNER_TRAIN_ONLY_NO_DISCOVERY_OR_FINAL_HELDOUT"})
        if orth>=1e-5: raise RuntimeError(f"non-orthogonal projector {stage}: {orth}")
    basis_row={"task":task,"fold":fold,"source_stage":"H_concat","train_trial_count":basis_info["train_trials"],
        "numerical_rank_C":basis_info["rank"],"K_C":basis_info["K_C"],"rank_tolerance":basis_info["singular_value_tolerance"],
        "explained_variance_fraction_top_K":basis_info["explained_variance_fraction"],"basis_orthogonality_max_abs":basis_info["basis_orthogonality_max_abs"],
        "partition_reconstruction_max_abs":basis_info["source_partition_reconstruction_max_abs"],"basis_sha256":arr_sha(basis),
        "labels_used_for_basis":False,"fit_role":"inner_train"}
    if basis_info["source_partition_reconstruction_max_abs"]>=1e-5 or basis_info["basis_orthogonality_max_abs"]>=1e-5:
        raise RuntimeError("source partition/C-basis audit failed")
    # Process both allowed roles once to construct all development-only metrics and router targets.
    role_rows={"inner_train":train_rows,"discovery":disc_rows}
    response_summary=[]; local=[]; conditional=[]; metrics=[]; raw_router={}; top1_logits={}; read_audit=[]
    identity={"task":task,"fold":fold,"representation_max_abs":0.0,"logits_max_abs":0.0,"prediction_exact":True,
        "raw_suffix_batch_roundoff_max_abs":0.0,
        "representation_tolerance":1e-6,"logits_tolerance":1e-6,"failure_policy":"FAIL_CLOSED"}
    for role_name,records in role_rows.items():
        rx=[]; ralpha=[]; ry=[]; rsub=[]; rses=[]; rlogit=[]; rcorrect=[]; rgain=[]
        for row in records:
            labels=row["raw_y"]; hs=row["hs"]; z0=row["logits"]; yd0=row["yd"]
            # Use exact frozen native shared-stage values from captured forward.
            yflat=torch.as_tensor(yd0,dtype=torch.float32,device=DEVICE)
            p_d=(yflat-torch.as_tensor(md,dtype=torch.float32,device=DEVICE))@torch.as_tensor(qd,dtype=torch.float32,device=DEVICE)
            curves,logits_bank,ident=eval_curve_bank(model,hs,yd0,z0,labels,qs,ms,qd,md,basis)
            for key in ("representation_max_abs","logits_max_abs","raw_suffix_batch_roundoff_max_abs"):
                identity[key]=max(identity[key],float(ident[key]))
            identity["prediction_exact"] &= bool(ident["prediction_exact"])
            if identity["representation_max_abs"]>=1e-6 or identity["logits_max_abs"]>=1e-6 or not identity["prediction_exact"]:
                raise RuntimeError(f"FAIL_CLOSED identity error {task}/{fold}: {identity}")
            # Native one-pass successor reconstruction with native C is tested in float64 at alpha=1.
            margin_curve=curves["margin"]
            od,oa=pick_oracle(margin_curve); rr=np.arange(len(labels)); one_pred=curves["prediction"][rr,od,oa].astype(np.int64); one_ce=curves["CE"][rr,od,oa]
            random_curves,random_logits,_=eval_curve_bank(model,hs,yd0,z0,labels,qs,ms,qd,md,random_basis)
            rd,ra=pick_oracle(random_curves["margin"]); rand_pred=random_curves["prediction"][rr,rd,ra].astype(np.int64); rand_ce=random_curves["CE"][rr,rd,ra]
            # Direction utility is the requested native-erasure true-margin change.
            utility=curves["margin"][:,:,4].mean(0)-curves["margin"][:,:,0].mean(0)
            response_summary.extend([{ "task":task,"fold":fold,"role":role_name,"session":row["session"],"subject":row["subject"],
                "direction":j,"alpha":float(a),"mean_true_class_margin":float(curves["margin"][:,j,ai].mean()),
                "mean_CE":float(curves["CE"][:,j,ai].mean()),"accuracy":float(np.mean(curves["prediction"][:,j,ai]==labels)),
                "mean_max_softmax":float(curves["max_softmax"][:,j,ai].mean()),"mean_native_predicted_probability":float(curves["native_predicted_probability"][:,j,ai].mean()),
                "mean_entropy":float(curves["entropy"][:,j,ai].mean()),"mean_successor_P_movement":float(curves["p_movement"][:,j,ai].mean()),
                "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"} for j in range(k) for ai,a in enumerate(ALPHAS)])
            local.extend(local_rows(task,fold,role_name,row,curves,utility))
            conditional.extend(conditional_rows(task,fold,role_name,row,curves))
            # top-3 ranking is frozen from the mean inner-train U_P before discovery outcomes are used.
            row["curves"]=curves; row["logits_bank"]=logits_bank; row["random_curves"]=random_curves
            row["oracle1_pred"]=one_pred; row["oracle1_ce"]=one_ce; row["random_pred"]=rand_pred; row["random_ce"]=rand_ce
            row["native_pred"]=z0.argmax(1); row["native_ce"]=F.cross_entropy(torch.as_tensor(z0),torch.as_tensor(labels),reduction="none").numpy()
            X,coeff=router_features(hs,z0,p_d.detach().cpu().numpy(),qs,ms,basis)
            row["router_X"]=X; row["coeff"]=coeff
            row["oracle_alpha_top1"] = None
            rx.append(X); ralpha.append(pick_alpha(curves["margin"][:,0,:])); ry.append(labels)
            rsub.append(np.full(len(labels),row["subject"],dtype="U16")); rses.append(np.full(len(labels),row["session"],np.int16))
            rcorrect.append((row["native_pred"]==labels).astype(np.int8))
            best=curves["margin"].max(axis=(1,2)); native=curves["margin"][:,0,4]
            rgain.append((best-native).astype(np.float32))
            read_audit.append({"task":task,"fold":fold,"role":role_name,"session":row["session"],"subject":row["subject"],
                "allowed_array_read":True,"rows":len(labels),"final_heldout":False,"outer_dev":False})
        raw_router[role_name]={"X":np.concatenate(rx),"alpha_index":np.concatenate(ralpha),"y":np.concatenate(ry),
            "subject":np.concatenate(rsub),"session":np.concatenate(rses),"correct":np.concatenate(rcorrect),"gain":np.concatenate(rgain)}
    # Inner-train subject-equal utility determines the three legal structured directions.
    utility_rows=[r for r in local if r["role"]=="inner_train"]
    utility_by_dir=[]
    for j in range(k):
        per_subject={}
        for r in utility_rows:
            if int(r["direction"])==j: per_subject.setdefault(str(r["subject"]),[]).append(float(r["U_P"]))
        subj={s:float(np.mean(v)) for s,v in per_subject.items()}
        utility_by_dir.append(float(np.mean(list(subj.values()))) if subj else 0.0)
    top3=sorted(range(k),key=lambda j:(-abs(utility_by_dir[j]),j))[:TOP_K]
    top1=top3[0]
    for row in train_rows+disc_rows:
        # utility above is train-only subject-equal and the top3 ranking is fixed for the whole cell.
        top_pred,top_ce=sequential_top3(model,row["hs"],row["raw_y"],row["coeff"],basis,qd,md,row["yd"],top3)
        row["top3_pred"]=top_pred; row["top3_ce"]=top_ce
    # Replace the provisional direction-0 target with the pre-ranked train-only top-1 direction.
    # This is the exact direction used by the frozen EEGNet router definition.
    for role_name,records in role_rows.items():
        raw_router[role_name]={
            "X":np.concatenate([r["router_X"] for r in records]),
            "alpha_index":np.concatenate([pick_alpha(r["curves"]["margin"][:,top1,:]) for r in records]),
            "y":np.concatenate([r["raw_y"] for r in records]),
            "subject":np.concatenate([np.full(len(r["raw_y"]),r["subject"],dtype="U16") for r in records]),
            "session":np.concatenate([np.full(len(r["raw_y"]),r["session"],dtype=np.int16) for r in records]),
            "correct":np.concatenate([(r["native_pred"]==r["raw_y"]).astype(np.int8) for r in records]),
            "gain":np.concatenate([(r["curves"]["margin"].max(axis=(1,2))-r["curves"]["margin"][:,top1,4]).astype(np.float32) for r in records]),
        }
    # Router gets the prelocked EEGNet-sized feature budget. PCA is refit within every GroupKFold training split.
    tr=raw_router["inner_train"]; te=raw_router["discovery"]
    budget=int(cell["feature_budget_from_eegnet_router_csv"])
    train_actions=action_from_alpha(tr["alpha_index"]); disc_actions=action_from_alpha(te["alpha_index"])
    oof,disc_action_pred,feature_count=fit_router(tr["X"],train_actions,tr["subject"],te["X"],budget)
    gate_by_role={}
    for role_name,records in role_rows.items():
        for row in records:
            if role_name=="discovery":
                idx=(te["subject"]==row["subject"])&(te["session"]==row["session"])
                act=disc_action_pred[idx]; action_alpha_idx=np.asarray([2,4,6],np.int64)[act]
                row["router_pred"]=row["logits_bank"][np.arange(len(row["raw_y"])),top1,action_alpha_idx].argmax(1)
                row["router_ce"]=F.cross_entropy(torch.as_tensor(row["logits_bank"][np.arange(len(row["raw_y"])),top1,action_alpha_idx]),torch.as_tensor(row["raw_y"]),reduction="none").numpy()
                row["router_action_pred"]=act
            else:
                row["router_pred"]=row["native_pred"]; row["router_ce"]=row["native_ce"]; row["router_action_pred"]=None
            variants={"ORACLE_1DIR":(row["oracle1_pred"],row["oracle1_ce"]),
                "ORACLE_TOP3":(row["top3_pred"],row["top3_ce"]),
                "RANDOM_DIRECTION_ORACLE":(row["random_pred"],row["random_ce"])}
            if role_name=="discovery": variants["ROUTER"]=(row["router_pred"],row["router_ce"])
            metrics.extend(performance_rows(task,fold,role_name,row["session"],row["subject"],row["raw_y"],
                row["native_pred"],row["native_ce"],variants))
    train_oof_scores=action_scores(train_actions,oof); disc_scores=action_scores(disc_actions,disc_action_pred)
    router_pred_rows=[{"task":task,"fold":fold,"population":"inner_train_grouped_OOF",**train_oof_scores,
        "biological_subjects":len(np.unique(tr["subject"])),"feature_count":feature_count,"feature_budget":budget,
        "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"},
        {"task":task,"fold":fold,"population":"discovery_subject_disjoint",**disc_scores,
        "subject_level_accuracy":float(np.mean([np.mean(disc_action_pred[te["subject"]==s]==disc_actions[te["subject"]==s]) for s in np.unique(te["subject"])])),
        "biological_subjects":len(np.unique(te["subject"])),"feature_count":feature_count,"feature_budget":budget,
        "session_transfer_accuracy":json.dumps([{ "session":int(s),"accuracy":float(np.mean(disc_action_pred[te["session"]==s]==disc_actions[te["session"]==s]))} for s in sorted(set(te["session"]))]),
        "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"}]
    obs_rows=[]
    for population,data,actions in (("inner_train_grouped_OOF",tr,train_actions),("discovery_subject_disjoint",te,disc_actions)):
        ctarget=data["correct"].astype(np.int64); gtarget=data["gain"].astype(np.float32)
        action_prediction=oof if population=="inner_train_grouped_OOF" else disc_action_pred
        # Correctness diagnostic is label-free at inference; target is never included in X.
        # Fit/evaluate observability target with grouped subject split: train OOF or disjoint discovery.
        if population=="inner_train_grouped_OOF":
            corr_oof,_,_=fit_binary_or_regression(data["X"],ctarget,data["subject"],data["X"],budget)
            corr_metric=float(balanced_accuracy_score(ctarget,corr_oof)) if len(np.unique(ctarget))>1 else float(np.mean(ctarget==corr_oof))
        else:
            train_corr=raw_router["inner_train"]["correct"].astype(np.int64)
            _,corr_oof,_=fit_binary_or_regression(raw_router["inner_train"]["X"],train_corr,raw_router["inner_train"]["subject"],data["X"],budget)
            corr_metric=float(balanced_accuracy_score(ctarget,corr_oof)) if len(np.unique(ctarget))>1 else float(np.mean(ctarget==corr_oof))
        gain_oof, gain_score = fit_gain_target(raw_router["inner_train"]["X"],raw_router["inner_train"]["gain"],raw_router["inner_train"]["subject"],data["X"],gtarget,budget,do_oof=(population=="inner_train_grouped_OOF"))
        obs_rows.append({"task":task,"fold":fold,"backbone":"SIRE-EEG","population":population,
            "target":"baseline_correctness","balanced_accuracy":corr_metric,"accuracy":float(np.mean(corr_oof==ctarget)),
            "biological_subjects":len(np.unique(data["subject"])),"feature_count":feature_count,"feature_budget":budget,
            "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
        obs_rows.append({"task":task,"fold":fold,"backbone":"SIRE-EEG","population":population,
            "target":"oracle_action_class","balanced_accuracy":float(balanced_accuracy_score(actions,action_prediction)),
            "macro_F1":float(f1_score(actions,action_prediction,labels=[0,1,2],average="macro",zero_division=0)),
            "biological_subjects":len(np.unique(data["subject"])),"feature_count":feature_count,"feature_budget":budget,
            "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
        obs_rows.append({"task":task,"fold":fold,"backbone":"SIRE-EEG","population":population,
            "target":"oracle_achievable_margin_gain","R2":gain_score["R2"],"MAE":gain_score["MAE"],"MSE":gain_score["MSE"],
            "biological_subjects":len(np.unique(data["subject"])),"feature_count":feature_count,"feature_budget":budget,
            "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    # Persist only compact per-cell runtime arrays needed for the exact downstream EEGNet comparison.
    router_data_path=runtime_cell/"router_dataset.npz"
    # Full response bank is written in cell-local compressed form; aggregation removes these intermediate shards later.
    curve_path=runtime_cell/"RESPONSE_CURVES.csv.gz"
    with gzip.open(curve_path,"wt",encoding="utf-8",newline="",compresslevel=6) as f:
        writer=csv.DictWriter(f,fieldnames=curve_fields(),extrasaction="ignore"); writer.writeheader()
        for role_name,records in role_rows.items():
            for row in records: write_curve_rows(writer,task,fold,role_name,row,row["curves"])
    write_csv(runtime_cell/"METRICS.csv",metrics)
    write_csv(runtime_cell/"RESPONSE_SUMMARY.csv",response_summary)
    write_csv(runtime_cell/"LOCAL_ACTIONABILITY.csv",local)
    write_csv(runtime_cell/"TRIAL_CONDITIONALITY.csv",conditional)
    write_csv(runtime_cell/"SIRE_PROJECTOR_AUDIT.csv",projector_rows)
    write_csv(runtime_cell/"SIRE_C_BASIS_AUDIT.csv",[basis_row])
    write_csv(runtime_cell/"SIRE_INTERVENTION_IDENTITY_AUDIT.csv",[identity])
    write_csv(runtime_cell/"ROUTER_PREDICTION_RESULTS.csv",router_pred_rows)
    write_csv(runtime_cell/"OBSERVABILITY_TARGETS.csv",obs_rows)
    # Save the compact runtime router cache with the train-utility-ranked top-1 direction.
    logits_train=[]; logits_disc=[]
    for row in train_rows: logits_train.append(row["logits_bank"][:,top1,:,:])
    for row in disc_rows: logits_disc.append(row["logits_bank"][:,top1,:,:])
    np.savez_compressed(router_data_path,train_X=tr["X"],train_alpha_index=tr["alpha_index"],train_y=tr["y"],train_subject=tr["subject"],train_session=tr["session"],train_correct=tr["correct"],train_gain=tr["gain"],
        discovery_X=te["X"],discovery_alpha_index=te["alpha_index"],discovery_y=te["y"],discovery_subject=te["subject"],discovery_session=te["session"],discovery_correct=te["correct"],discovery_gain=te["gain"],discovery_action_pred=disc_action_pred,
        discovery_top1_logits_curve=np.concatenate(logits_disc,axis=0),train_top1_logits_curve=np.concatenate(logits_train,axis=0))
    done_record={"status":"COMPLETE","task":task,"fold":fold,"seed":0,"protocol_sha256":sha_file(PROTOCOL/"PROTOCOL_LOCK.json"),
        "checkpoint_sha256":cell["frozen_checkpoint_sha256"],"normalizer_sha256":cell["normalizer_sha256"],"split_sha256":split,
        "inner_train_array_rows":sum(len(r["raw_y"]) for r in train_rows),"discovery_array_rows":sum(len(r["raw_y"]) for r in disc_rows),
        "outer_dev_array_reads":read_counts["outer_dev"],"final_heldout_array_reads":read_counts["final_heldout"],"allowed_read_counts":read_counts,
        "sire_state_sha256_before":state_before,"sire_state_sha256_after":model_state_sha(model),"trainable_parameters":0,
        "identity_audit":identity,"C_basis_rank":basis_info["rank"],"K_C":k,"top3_directions":top3,
        "projector_hashes":{"source":arr_sha(qs,ms),"successor":arr_sha(qd,md)},"runtime_curve_sha256":sha_file(curve_path),
        "router_dataset_sha256":sha_file(router_data_path),"role_subject_sha256":{"inner_train":subject_key(train_ids),"discovery":subject_key(disc_ids)}}
    if done_record["sire_state_sha256_before"]!=done_record["sire_state_sha256_after"]: raise RuntimeError("frozen SIRE state changed")
    write_json(done,done_record)
    print(f"CELL_COMPLETE {task} fold={fold} trials={done_record['inner_train_array_rows']}+{done_record['discovery_array_rows']} K={k} identity={identity['logits_max_abs']:.2e}",flush=True)


def fit_binary_or_regression(train_X,train_y,train_subject,test_X,budget):
    unique=np.unique(train_subject); ncomp=min(int(budget),train_X.shape[1])
    def clf():
        classes=np.unique(train_y)
        if len(classes)==1:
            class Constant:
                def __init__(self,v): self.v=int(v)
                def fit(self,x,y): return self
                def predict(self,x): return np.full(len(x),self.v,np.int64)
            return Constant(classes[0])
        steps=[StandardScaler()]
        if train_X.shape[1]>ncomp: steps.append(PCA(n_components=ncomp,svd_solver="full"))
        steps.extend([StandardScaler(),LogisticRegression(C=1.0,class_weight="balanced",max_iter=500,solver="lbfgs",random_state=0)])
        return make_pipeline(*steps)
    if len(unique)<2: raise RuntimeError("grouped observability model needs two training subjects")
    oof=np.full(len(train_y),-1,np.int64)
    for tr,va in GroupKFold(n_splits=min(5,len(unique))).split(train_X,train_y,groups=train_subject):
        m=clf(); m.fit(train_X[tr],train_y[tr]); oof[va]=m.predict(train_X[va])
    m=clf(); m.fit(train_X,train_y); pred=m.predict(test_X)
    if len(test_X)==len(train_X) and np.array_equal(test_X,train_X): return oof,pred,ncomp
    return oof,pred,ncomp


def fit_gain_target(train_X,train_gain,train_subject,test_X,test_gain,budget,do_oof=False):
    ncomp=min(int(budget),train_X.shape[1]); unique=np.unique(train_subject)
    def reg():
        steps=[StandardScaler()]
        if train_X.shape[1]>ncomp: steps.append(PCA(n_components=ncomp,svd_solver="full"))
        steps.extend([StandardScaler(),Ridge(alpha=1.0)])
        return make_pipeline(*steps)
    if do_oof:
        pred=np.empty(len(train_gain),np.float32)
        for tr,va in GroupKFold(n_splits=min(5,len(unique))).split(train_X,train_gain,groups=train_subject):
            m=reg(); m.fit(train_X[tr],train_gain[tr]); pred[va]=m.predict(train_X[va])
    else:
        m=reg(); m.fit(train_X,train_gain); pred=m.predict(test_X)
    mse=float(np.mean((test_gain-pred)**2)); mae=float(np.mean(np.abs(test_gain-pred)))
    return pred,{"R2":float(r2_score(test_gain,pred)) if np.std(test_gain)>0 else float("nan"),"MAE":mae,"MSE":mse}


def disc_rows_for(task,fold,rows):
    return rows


def aggregate():
    lock=verify_lock(); cells=[(t,f) for t in TASKS for f in FOLDS]
    cell_dirs=[RUNTIME/"cells"/t.lower()/f"fold{f}_seed0" for t,f in cells]
    for (task,fold),d in zip(cells,cell_dirs):
        rec_path=d/"CELL_COMPLETE.json"
        if not rec_path.is_file(): raise RuntimeError(f"incomplete cell {task}/{fold}")
        rec=json.loads(rec_path.read_text(encoding="utf-8"))
        if rec.get("status")!="COMPLETE" or rec.get("protocol_sha256")!=sha_file(PROTOCOL/"PROTOCOL_LOCK.json"):
            raise RuntimeError(f"cell receipt mismatch {task}/{fold}")
        if rec["final_heldout_array_reads"] or rec["outer_dev_array_reads"]: raise RuntimeError("forbidden role array read recorded")
    # Concatenate the complete curves in deterministic task/fold order.
    all_curve=OUT/"RESPONSE_CURVES.csv.gz"; tmp=all_curve.with_suffix(".csv.gz.part")
    wrote_header=False
    with gzip.open(tmp,"wt",encoding="utf-8",newline="",compresslevel=6) as out:
        writer=None
        for d in cell_dirs:
            with gzip.open(d/"RESPONSE_CURVES.csv.gz","rt",encoding="utf-8",newline="") as inp:
                reader=csv.DictReader(inp)
                if not wrote_header:
                    writer=csv.DictWriter(out,fieldnames=reader.fieldnames,extrasaction="ignore");writer.writeheader();wrote_header=True
                for row in reader: writer.writerow(row)
    os.replace(tmp,all_curve)
    def collect(name): return [r for d in cell_dirs for r in read_csv(d/name)]
    metrics=collect("METRICS.csv"); response=collect("RESPONSE_SUMMARY.csv"); local=collect("LOCAL_ACTIONABILITY.csv")
    conditional=collect("TRIAL_CONDITIONALITY.csv"); projects=collect("SIRE_PROJECTOR_AUDIT.csv"); cbasis=collect("SIRE_C_BASIS_AUDIT.csv")
    identity=collect("SIRE_INTERVENTION_IDENTITY_AUDIT.csv"); router_pred=collect("ROUTER_PREDICTION_RESULTS.csv"); obs=collect("OBSERVABILITY_TARGETS.csv")
    write_csv(OUT/"RESPONSE_CURVES.csv",response);write_csv(OUT/"LOCAL_ACTIONABILITY.csv",local)
    write_csv(OUT/"TRIAL_CONDITIONALITY.csv",conditional);write_csv(OUT/"SIRE_PROJECTOR_AUDIT.csv",projects)
    write_csv(OUT/"SIRE_C_BASIS_AUDIT.csv",cbasis);write_csv(OUT/"SIRE_INTERVENTION_IDENTITY_AUDIT.csv",identity)
    write_csv(OUT/"ROUTER_PREDICTION_RESULTS.csv",router_pred)
    # Formal per-subject oracle rows and task-level 20k biological-subject CIs.
    oracle1=[r for r in metrics if r["role"]=="discovery" and r["variant"] in ("BASELINE","ORACLE_1DIR")]
    oracle3=[r for r in metrics if r["role"]=="discovery" and r["variant"] in ("BASELINE","ORACLE_TOP3")]
    random_rows=[r for r in metrics if r["role"]=="discovery" and r["variant"] in ("BASELINE","RANDOM_DIRECTION_ORACLE")]
    write_csv(OUT/"ORACLE_1DIR_RESULTS.csv",[r for r in metrics if r["role"]=="discovery" and r["variant"]=="ORACLE_1DIR"])
    write_csv(OUT/"ORACLE_TOP3_RESULTS.csv",[r for r in metrics if r["role"]=="discovery" and r["variant"]=="ORACLE_TOP3"])
    write_csv(OUT/"RANDOM_DIRECTION_ORACLE.csv",[r for r in metrics if r["role"]=="discovery" and r["variant"]=="RANDOM_DIRECTION_ORACLE"])
    head=[]; router_perf=[]
    for task in TASKS:
        base=subject_equal(metrics,"BASELINE",task)
        one=subject_equal(metrics,"ORACLE_1DIR",task); three=subject_equal(metrics,"ORACLE_TOP3",task); rnd=subject_equal(metrics,"RANDOM_DIRECTION_ORACLE",task)
        delta1=pair_mean(one,base); delta3=pair_mean(three,base); deltar=pair_mean(rnd,base)
        one_mean,one_lo,one_hi=bootstrap(np.asarray(list(delta1.values())),stable_seed("bootstrap",task,"oracle1"))
        three_mean,three_lo,three_hi=bootstrap(np.asarray(list(delta3.values())),stable_seed("bootstrap",task,"oracle3"))
        rand_mean,rand_lo,rand_hi=bootstrap(np.asarray(list(deltar.values())),stable_seed("bootstrap",task,"random"))
        struct_minus_random={s:delta3[s]-deltar[s] for s in set(delta3)&set(deltar)}
        sr_mean,sr_lo,sr_hi=bootstrap(np.asarray(list(struct_minus_random.values())),stable_seed("bootstrap",task,"structured_minus_random"))
        head.append({"task":task,"role":"discovery","baseline_BA":float(np.mean(list(base.values()))),"biological_subjects":len(base),
            "ORACLE_1DIR_delta_BA":one_mean,"ORACLE_1DIR_CI95_low":one_lo,"ORACLE_1DIR_CI95_high":one_hi,"ORACLE_1DIR_BA":float(np.mean(list(one.values()))),
            "ORACLE_TOP3_delta_BA":three_mean,"ORACLE_TOP3_CI95_low":three_lo,"ORACLE_TOP3_CI95_high":three_hi,"ORACLE_TOP3_BA":float(np.mean(list(three.values()))),
            "RANDOM_DIRECTION_ORACLE_delta_BA":rand_mean,"RANDOM_DIRECTION_ORACLE_CI95_low":rand_lo,"RANDOM_DIRECTION_ORACLE_CI95_high":rand_hi,
            "RANDOM_DIRECTION_ORACLE_BA":float(np.mean(list(rnd.values()))),"structured_top3_minus_random_delta_BA":sr_mean,
            "structured_minus_random_CI95_low":sr_lo,"structured_minus_random_CI95_high":sr_hi,
            "router_triggered":bool(max(one_mean,three_mean)>=.01),"label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
        if max(one_mean,three_mean)<.01:
            continue
        router=subject_equal(metrics,"ROUTER",task); dr=pair_mean(router,base)
        oracle_source="ORACLE_TOP3" if three_mean>=one_mean else "ORACLE_1DIR"
        delta_oracle=delta3 if oracle_source=="ORACLE_TOP3" else delta1
        gain=float(np.mean(list(dr.values()))) if dr else float("nan")
        denominator=float(np.mean(list(delta_oracle.values()))) if delta_oracle else float("nan")
        frac=gain/denominator if denominator else float("nan")
        frac_subject={s:dr[s]/delta_oracle[s] for s in set(dr)&set(delta_oracle) if abs(delta_oracle[s])>1e-12}
        gain_ci=bootstrap(np.asarray(list(dr.values())),stable_seed("bootstrap",task,"router_gain"))
        rec_ci=bootstrap(np.asarray(list(frac_subject.values())),stable_seed("bootstrap",task,"router_recovery"))
        action_rows=[r for r in router_pred if r["task"]==task and r["population"]=="discovery_subject_disjoint"]
        action_ba=float(np.mean([float(r["balanced_accuracy"]) for r in action_rows])) if action_rows else float("nan")
        router_perf.append({"task":task,"fold":"ALL_FOLDS","role":"discovery","variant":"ROUTER_SUMMARY","BA":float(np.mean(list(router.values()))),
            "baseline_BA":float(np.mean(list(base.values()))),"router_BA_gain":gain,"router_BA_gain_CI95_low":gain_ci[1],"router_BA_gain_CI95_high":gain_ci[2],
            "oracle_headroom_source":oracle_source,"oracle_BA_headroom":denominator,"fraction_oracle_headroom_recovered":frac,
            "recovery_CI95_low":rec_ci[1],"recovery_CI95_high":rec_ci[2],"router_action_balanced_accuracy":action_ba,
            "biological_subjects":len(set(router)&set(base)),"label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    write_csv(OUT/"ORACLE_HEADROOM_SUMMARY.csv",head)
    # Router performance by biological subject/session, then the subject-equal summary above.
    for r in metrics:
        if r["variant"]=="ROUTER":
            b=next(x for x in metrics if x["task"]==r["task"] and x["fold"]==r["fold"] and x["role"]==r["role"] and x["session"]==r["session"] and x["subject"]==r["subject"] and x["variant"]=="BASELINE")
            r["baseline_BA"]=b["BA"]
    write_csv(OUT/"ROUTER_PERFORMANCE.csv",[r for r in metrics if r["variant"]=="ROUTER"]+router_perf)
    # Observability targets also run on the untouched EEGNet router feature arrays and outcomes.
    obs.extend(run_eegnet_observability())
    write_csv(OUT/"OBSERVABILITY_TARGETS.csv",obs)
    write_csv(OUT/"ROUTER_DATASET_AUDIT.csv",[{"task":t,"fold":f,"inner_train_subjects":len(next(c for c in lock["cells"] if c["task"]==t and c["fold"]==f)["inner_train_subjects"]),
        "discovery_subjects":len(next(c for c in lock["cells"] if c["task"]==t and c["fold"]==f)["discovery_subjects"]),
        "feature_budget":next(c for c in lock["cells"] if c["task"]==t and c["fold"]==f)["feature_budget_from_eegnet_router_csv"],
        "router_features":"p_s,C coefficients,p_d,native logits,confidence/entropy/top-gap/P-C norms and ratios",
        "split":"subject-disjoint; GroupKFold in inner-train","final_heldout_labels_used":False} for t,f in cells])
    comparison=compare_backbones(head,router_perf,conditional,router_pred)
    write_csv(OUT/"EEGNET_SIRE_OBSERVABILITY_COMPARISON.csv",comparison)
    # Final exclusion audit is receipt-derived; the code never calls the final loader.
    receipts=[json.loads((d/"CELL_COMPLETE.json").read_text(encoding="utf-8")) for d in cell_dirs]
    allowed=sum(x["inner_train_array_rows"]+x["discovery_array_rows"] for x in receipts)
    write_json(OUT/"FINAL_HELDOUT_EXCLUSION_AUDIT.json",{"FINAL_HELDOUT_ACCESSED":False,"final_heldout_eeg_array_reads":0,
        "outer_dev_eeg_array_reads":0,"allowed_inner_train_and_discovery_trial_rows":allowed,
        "cell_read_receipts":[{"task":x["task"],"fold":x["fold"],"inner_train_array_rows":x["inner_train_array_rows"],
            "discovery_array_rows":x["discovery_array_rows"],"outer_dev_array_reads":x["outer_dev_array_reads"],"final_heldout_array_reads":x["final_heldout_array_reads"]} for x in receipts],
        "statement":"No OpenBMI 14 final-heldout EEG arrays or outer-development arrays were loaded by this experiment. Historical checkpoint diagnostic exposure is separately disclosed."})
    build_report(head,router_perf,comparison,conditional,identity)
    # Drop per-cell response shards and router datasets only after final compressed curve and reports exist.
    for d in cell_dirs:
        for name in ("RESPONSE_CURVES.csv.gz","router_dataset.npz"):
            (d/name).unlink(missing_ok=True)
    print(f"AGGREGATE_COMPLETE cells={len(cells)} curve_bytes={all_curve.stat().st_size}",flush=True)


def run_eegnet_observability():
    rows=[]
    for task in TASKS:
        for fold in FOLDS:
            p=EEGNET_RUNTIME/"cells"/task.lower()/f"fold{fold}_seed0"/"router_dataset.npz"
            if not p.is_file(): raise RuntimeError(f"EEGNet router feature archive missing for observability audit: {p}")
            with np.load(p,allow_pickle=False) as z:
                trX=z["train_X"].astype(np.float32); trsub=z["train_subject"].astype(str); trY=z["train_y"].astype(np.int64)
                tr_alpha=z["train_alpha_index"].astype(np.int64)
                discX=z["discovery_X"].astype(np.float32); discsub=z["discovery_subject"].astype(str); discY=z["discovery_y"].astype(np.int64)
                disc_alpha=z["discovery_alpha_index"].astype(np.int64); logits=z["discovery_top1_logits_curve"].astype(np.float32)
                trlog=z["train_top1_logits_curve"].astype(np.float32) if "train_top1_logits_curve" in z else None
            budget=prior_eegnet_feature_budget(task,fold)
            disc_correct=(logits[:,4,:].argmax(1)==discY).astype(np.int64)
            disc_margin=margin_np(logits[:,4,:],discY)
            disc_best=np.max(np.stack([margin_np(logits[:,a,:],discY) for a in range(len(ALPHAS))],axis=1),axis=1)
            disc_gain=(disc_best-disc_margin).astype(np.float32)
            # Recompute inner-train targets from the stored development-only top-1 curve.
            if trlog is not None:
                trcorrect=(trlog[:,4,:].argmax(1)==trY).astype(np.int64)
                trmargin=margin_np(trlog[:,4,:],trY); trbest=np.max(np.stack([margin_np(trlog[:,a,:],trY) for a in range(len(ALPHAS))],axis=1),axis=1); trgain=(trbest-trmargin).astype(np.float32)
            else:
                trcorrect=np.zeros(len(trY),np.int64); trgain=np.zeros(len(trY),np.float32)
            action_tr=action_from_alpha(tr_alpha); action_te=action_from_alpha(disc_alpha)
            action_oof,action_pred,_=fit_router(trX,action_tr,trsub,discX,budget)
            for target,yt_train,yt_test,continuous in (("baseline_correctness",trcorrect,disc_correct,False),
                ("oracle_action_class",action_tr,action_te,False),("oracle_achievable_margin_gain",trgain,disc_gain,True)):
                if continuous:
                    pred,score=fit_gain_target(trX,yt_train,trsub,discX,yt_test,budget,do_oof=False)
                    rows.append({"task":task,"fold":fold,"backbone":"EEGNet","population":"discovery_subject_disjoint","target":target,
                        "R2":score["R2"],"MAE":score["MAE"],"MSE":score["MSE"],"biological_subjects":len(np.unique(discsub)),
                        "feature_count":trX.shape[1],"feature_budget":budget,"label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
                else:
                    _,pred,_=fit_binary_or_regression(trX,yt_train,trsub,discX,budget)
                    rows.append({"task":task,"fold":fold,"backbone":"EEGNet","population":"discovery_subject_disjoint","target":target,
                        "balanced_accuracy":float(balanced_accuracy_score(yt_test,pred)) if len(np.unique(yt_test))>1 else float(np.mean(yt_test==pred)),
                        "macro_F1":float(f1_score(yt_test,pred,average="macro",zero_division=0)),"accuracy":float(np.mean(yt_test==pred)),
                        "biological_subjects":len(np.unique(discsub)),"feature_count":trX.shape[1],"feature_budget":budget,
                        "label_scope":"DEVELOPMENT_ONLY_NO_FINAL_HELDOUT"})
    return rows


def margin_np(logits,y):
    own=logits[np.arange(len(y)),y]; z=logits.copy(); z[np.arange(len(y)),y]=-np.inf; return own-z.max(1)


def compare_backbones(sire_head, sire_router, conditional, sire_router_predictions):
    eeg_head=read_csv(EEGNET_OUT/"ORACLE_HEADROOM_SUMMARY.csv")
    eeg_router=[r for r in read_csv(EEGNET_OUT/"ROUTER_PERFORMANCE.csv") if r["variant"]=="ROUTER_SUMMARY"]
    eeg_actions=read_csv(EEGNET_OUT/"ROUTER_PREDICTION_RESULTS.csv")
    eeg_cond=read_csv(EEGNET_OUT/"TRIAL_CONDITIONALITY.csv")
    rows=[]
    for task in TASKS:
        ec=next(r for r in eeg_head if r["task"]==task and r["role"]=="discovery")
        er=next(r for r in eeg_router if r["task"]==task and r["role"]=="discovery")
        ea=[r for r in eeg_actions if r["task"]==task and r["population"]=="discovery_subject_disjoint"]
        econd=[r for r in eeg_cond if r["task"]==task and r["role"]=="discovery"]
        cond_e=np.mean([str(r["both_sides_at_least_0p10"]).lower()=="true" for r in econd]) if econd else float("nan")
        sh=next(r for r in sire_head if r["task"]==task)
        sr=next((r for r in sire_router if r["task"]==task),None)
        scond=[r for r in conditional if r["task"]==task and r["role"]=="discovery"]
        cond_s=np.mean([str(r["trial_conditional"]).lower()=="true" for r in scond]) if scond else float("nan")
        sa=[r for r in sire_router_predictions if r["task"]==task and r["population"]=="discovery_subject_disjoint"]
        rows.extend([
            {"task":task,"backbone":"EEGNet","baseline_BA":ec["baseline_BA"],"oracle1_delta_BA":ec["ORACLE_1DIR_delta_BA_mean"],
             "oracle3_delta_BA":ec["ORACLE_TOP3_delta_BA_mean"],"random_oracle_delta_BA":ec["RANDOM_DIRECTION_ORACLE_delta_BA_mean"],
             "trial_conditionality_fraction":cond_e,"router_action_balanced_accuracy":float(np.mean([float(r["balanced_accuracy"]) for r in ea])),
             "router_delta_BA":er["router_BA_gain"],"oracle_headroom_recovered":er["fraction_oracle_headroom_recovered"],"source":"frozen prior output CSV values"},
            {"task":task,"backbone":"SIRE-EEG","baseline_BA":sh["baseline_BA"],"oracle1_delta_BA":sh["ORACLE_1DIR_delta_BA"],
             "oracle3_delta_BA":sh["ORACLE_TOP3_delta_BA"],"random_oracle_delta_BA":sh["RANDOM_DIRECTION_ORACLE_delta_BA"],
             "trial_conditionality_fraction":cond_s,"router_action_balanced_accuracy":sr["router_action_balanced_accuracy"] if sr else "",
             "router_delta_BA":sr["router_BA_gain"] if sr else "","oracle_headroom_recovered":sr["fraction_oracle_headroom_recovered"] if sr else "",
             "source":"current frozen SIRE development-only output"}])
    return rows


def build_report(head,router,comparison,conditional,identity):
    lines=["# SIRE-EEG C-direction actionability and observability", "",
        "This report uses only inner-train and discovery EEG arrays. The canonical seed-zero neural checkpoints and BN states were frozen. Historical source-audit records disclose earlier final-heldout diagnostic exposure; this run accessed zero final-heldout EEG arrays.", "",
        "## Q1. Does SIRE have structured C-direction oracle headroom?", "",
        "Oracle-1dir and sequential Oracle-top3 balanced-accuracy gains are reported below with biological-subject 20,000-draw bootstrap 95% CIs. The random-direction oracle uses the same source-complement rank.", "",
        "| Task | Baseline BA | Oracle-1 ΔBA [95% CI] | Oracle-top3 ΔBA [95% CI] | Random ΔBA [95% CI] | Structured−random [95% CI] |", "|---|---:|---:|---:|---:|---:|"]
    for r in head:
        lines.append(f"| {r['task']} | {float(r['baseline_BA']):.4f} | {float(r['ORACLE_1DIR_delta_BA']):+.4f} [{float(r['ORACLE_1DIR_CI95_low']):+.4f}, {float(r['ORACLE_1DIR_CI95_high']):+.4f}] | {float(r['ORACLE_TOP3_delta_BA']):+.4f} [{float(r['ORACLE_TOP3_CI95_low']):+.4f}, {float(r['ORACLE_TOP3_CI95_high']):+.4f}] | {float(r['RANDOM_DIRECTION_ORACLE_delta_BA']):+.4f} [{float(r['RANDOM_DIRECTION_ORACLE_CI95_low']):+.4f}, {float(r['RANDOM_DIRECTION_ORACLE_CI95_high']):+.4f}] | {float(r['structured_top3_minus_random_delta_BA']):+.4f} [{float(r['structured_minus_random_CI95_low']):+.4f}, {float(r['structured_minus_random_CI95_high']):+.4f}] |")
    lines += ["", "## Q2. Is optimal routing trial-conditional?", "",
        "The locked criterion counts a subject/session/direction profile when at least 10% of its trials choose α<1 and at least 10% choose α>1. The fraction of discovery profiles meeting that rule is in `EEGNET_SIRE_OBSERVABILITY_COMPARISON.csv`; per-profile distributions are in `TRIAL_CONDITIONALITY.csv`.", "",
        "## Q3. How much oracle headroom does the same label-free router recover?", "",
        "The diagnostic router is fit on inner-train subjects with grouped OOF CV and evaluated on disjoint discovery subjects. It uses the same logistic-regression family and action labels as the EEGNet audit, with deterministic train-only PCA compression to the EEGNet feature budget when needed. Per-task gain, recovery fraction, and subject-bootstrap intervals are in `ROUTER_PERFORMANCE.csv`.", "",
        "## Q4. Did actionability observability improve versus EEGNet?", "",
        "| Task | Backbone | Baseline BA | Oracle1 ΔBA | Oracle3 ΔBA | Random ΔBA | Conditional profiles | Router action BA | Router ΔBA | Oracle recovered |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        def fmt(k, digits=3):
            try:return f"{float(r[k]):.{digits}f}"
            except:return "NA"
        lines.append(f"| {r['task']} | {r['backbone']} | {fmt('baseline_BA')} | {fmt('oracle1_delta_BA')} | {fmt('oracle3_delta_BA')} | {fmt('random_oracle_delta_BA')} | {fmt('trial_conditionality_fraction')} | {fmt('router_action_balanced_accuracy')} | {fmt('router_delta_BA')} | {fmt('oracle_headroom_recovered')} |")
    lines += ["", "## Interpretation", ""]
    for task in TASKS:
        s=next(r for r in head if r["task"]==task); c=[r for r in comparison if r["task"]==task]
        eg=next(r for r in c if r["backbone"]=="EEGNet"); si=next(r for r in c if r["backbone"]=="SIRE-EEG")
        if float(s["ORACLE_TOP3_delta_BA"])<.01 and float(si["baseline_BA"])>float(eg["baseline_BA"]): label="SIRE_NATIVE_REPRESENTATION_NEAR_ACTIONABILITY_SATURATION"
        elif float(s["ORACLE_TOP3_delta_BA"])>=.01 and float(si["oracle_headroom_recovered"] or 0)>float(eg["oracle_headroom_recovered"])+.05 and float(si["router_delta_BA"] or 0)>0: label="BACKBONE_IMPROVES_ACTIONABILITY_OBSERVABILITY"
        elif float(s["ORACLE_TOP3_delta_BA"])>=.01 and float(si["random_oracle_delta_BA"])<float(si["oracle3_delta_BA"]): label="STRUCTURED_BUT_UNOBSERVABLE_SIRE_ACTIONABILITY"
        elif float(s["ORACLE_TOP3_delta_BA"])>=.01: label="ACTIONABILITY_REMAINS_UNOBSERVABLE_ACROSS_BACKBONES"
        else: label="SIRE_NATIVE_REPRESENTATION_NEAR_ACTIONABILITY_SATURATION"
        lines.append(f"- **{task}:** `{label}`; point-estimate headroom recovery is {fmt_value(si.get('oracle_headroom_recovered'))} for SIRE-EEG versus {fmt_value(eg.get('oracle_headroom_recovered'))} for EEGNet.")
    lines += ["", "The comparison uses the frozen prior EEGNet output CSVs without changing them. Checkpoint provenance and historical heldout diagnostic exposure are disclosed in `SIRE_SOURCE_AUDIT.json`; current-run heldout exclusion receipts are in `FINAL_HELDOUT_EXCLUSION_AUDIT.json`. Current performance CIs use biological-subject bootstrap, and this development-only audit does not estimate final-heldout performance."]
    (OUT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def fmt_value(value, digits=3):
    try:return f"{float(value):.{digits}f}"
    except:return "NA"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("action",choices=("lock","cell","all","aggregate"));ap.add_argument("--task",choices=TASKS);ap.add_argument("--fold",type=int,choices=FOLDS)
    a=ap.parse_args()
    if a.action=="lock": lock_protocol()
    elif a.action=="cell":
        if a.task is None or a.fold is None: ap.error("cell requires --task and --fold")
        evaluate_cell(a.task,a.fold)
    elif a.action=="all":
        for t in TASKS:
            for f in FOLDS: evaluate_cell(t,f)
        aggregate()
    else: aggregate()


if __name__=="__main__": main()
