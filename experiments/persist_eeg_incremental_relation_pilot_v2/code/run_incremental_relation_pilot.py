"""Fast frozen-feature audit for incremental cross-session discrimination.

The script is deliberately split into a source-only phase and a post-lock
outcome phase.  It reuses the previously completed fold-0 SUBJECT_BALANCED_ERM
checkpoint, trains no new neural backbone, and keeps all feature arrays under
the ignored runtime directory.  The four scores are fixed before outcome
access and are evaluated with the same frozen feature extractor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

HERE = Path(__file__).resolve()
PILOT = Path(os.environ.get("INCREMENTAL_RELATION_PILOT_ROOT", str(HERE.parents[1]))).resolve()
BASE = PILOT.parent / "persist_eeg_geosr_final_v1"
sys.path.insert(0, str(BASE / "code"))
import audit_primitives as ap  # noqa: E402
import run_geosr as geo  # noqa: E402


DATASETS = ("OpenBMI", "WBCIC")
METHODS = ("SUBJECT_BALANCED_ERM", "GENERIC_RESIDUAL", "GENERIC_PROTOTYPE", "CROSS_SESSION_RELATION")
SEED = 0
FOLD = 0
BATCH_SIZE = 256
RESIDUAL_ALPHA = 1.0
RESIDUAL_SCALE = 0.25
EPS = 1e-8
CLEAR_PP = 0.5


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def code_hash() -> str:
    return sha(HERE)


def build_locks() -> str:
    amendment_path = PILOT / "INCREMENTAL_RELATION_PROTOCOL_AMENDMENT.json"
    amendment = {
        "schema": "PERSIST_EEG_INCREMENTAL_RELATION_AMENDMENT_V1",
        "status": "ACTIVE",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "fast frozen-feature pilot for incremental cross-session discriminative information",
        "seed": SEED, "datasets": list(DATASETS), "outer_folds": [FOLD],
        "backbone": "canonical EEGNet frozen fold0 SUBJECT_BALANCED_ERM checkpoint",
        "methods": list(METHODS),
        "controls": {"generic_residual": True, "generic_prototype": True,
                      "relation_uses_full_latent": True, "protected_only": False},
        "training": {"neural_backbone_training": False, "residual_alpha": RESIDUAL_ALPHA,
                     "residual_scale": RESIDUAL_SCALE, "prototype_metric": "standardized_squared_euclidean",
                     "relation_score": "centered_full_latent_dot_mean_unit_cross_session_direction"},
        "data_scope": {"source_only_before_lock": True, "outcome_after_both_dataset_locks": True,
                       "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False},
        "stopping_rule": {"both_dataset_relation_delta_BA_pp_min": CLEAR_PP,
                           "both_dataset_relation_delta_macro_F1_pp_min": 0.0,
                           "nonnegative_subject_fraction_min": 0.5,
                           "relation_over_strongest_generic_delta_BA_pp_min": CLEAR_PP},
        "scientific_definition_changed": True, "final_claim_authorized": False,
        "all_existing_cache_retained": True,
    }
    write_json(amendment_path, amendment)
    amendment_sha = sha(amendment_path)
    training_lock = {
        "schema": "PERSIST_EEG_INCREMENTAL_RELATION_TRAINING_LOCK_V1",
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "amendment_sha256": amendment_sha, "code_sha256": code_hash(),
        "seed": SEED, "datasets": list(DATASETS), "outer_folds": [FOLD], "methods": list(METHODS),
        "feature_view": "full_latent_64d", "protected_only": False,
        "outcome_labels_read": False, "outcome_labels_read_before_lock": False,
        "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False,
        "scientific_definition_changed": True, "final_claim_authorized": False,
    }
    write_json(PILOT / "INCREMENTAL_RELATION_TRAINING_LOCK.json", training_lock)
    execution_lock = {
        "schema": "PERSIST_EEG_INCREMENTAL_RELATION_EXECUTION_LOCK_V1",
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "amendment_sha256": amendment_sha, "training_lock_sha256": sha(PILOT / "INCREMENTAL_RELATION_TRAINING_LOCK.json"),
        "code_sha256": code_hash(), "device": "cuda:0", "execution_mode": "sequential_single_gpu",
        "outcome_labels_read": False, "all_existing_cache_retained": True,
        "scientific_definition_changed": False, "final_claim_authorized": False,
    }
    write_json(PILOT / "INCREMENTAL_RELATION_EXECUTION_LOCK.json", execution_lock)
    return amendment_sha


def checkpoint_path(dataset: str) -> Path:
    manifest = BASE / "rapid_triage" / "evidence" / "PREFLIGHT_MANIFEST.json"
    if not manifest.is_file():
        manifest = BASE / "rapid_triage" / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json"
    entry = read_json(manifest)[f"{dataset}/fold-0/seed-0"]
    path = Path(entry["checkpoints"]["SUBJECT_BALANCED_ERM"]["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_frozen(dataset: str, root: Path, device: torch.device) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    roles, _, _ = ap.load_roles(dataset)
    role = roles[FOLD]
    source = geo.subj_sort(role["model_fit"])
    data = ap.load_ab_data(dataset, set(source))
    sessions = geo.SESSIONS_FIT[dataset]
    rows = ap.indices_for(data, source, sessions)
    ck = checkpoint_path(dataset)
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    mean, std = np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32)
    channels = int(data.batch(np.asarray([rows[0]], np.int64)).shape[1])
    model = ap.VanillaEEGNet(channels).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    t0 = time.perf_counter()
    hs, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            q = rows[start:start + BATCH_SIZE]
            x = ap.prepare(data, q, mean, std).to(device, non_blocking=True)
            hs.append(model.forward_features(x).detach().cpu().numpy())
            logits.append(model(x).detach().cpu().numpy())
    h = np.concatenate(hs, axis=0).astype(np.float32)
    z = np.concatenate(logits, axis=0).astype(np.float32)
    meta = data.metadata.iloc[rows].reset_index(drop=True).copy()
    source_sha = hashlib.sha256("|".join(source).encode()).hexdigest()
    runtime = root / "runtime" / f"{dataset}_fold0"
    runtime.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(runtime / "source_frozen_features.npz", h=h, logits=z,
                        label=meta.label.to_numpy(np.int64),
                        subject=meta.subject_id.astype(str).to_numpy(),
                        session=meta.session_id.astype(np.int64).to_numpy())
    write_json(runtime / "source_manifest.json", {
        "dataset": dataset, "fold": FOLD, "seed": SEED, "source_subjects": source,
        "source_subjects_sha256": source_sha, "source_rows": int(len(rows)),
        "feature_dim": int(h.shape[1]), "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
        "normalizer_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "normalizer_std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
        "feature_extraction_sec": time.perf_counter() - t0, "outcome_labels_read": False,
    })
    return data, role, h, z, meta, mean, {"std": std, "checkpoint": str(ck), "checkpoint_sha256": sha(ck)}


def standardize_fit(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = h.mean(0).astype(np.float64)
    sd = h.std(0).astype(np.float64)
    sd[sd < 1e-6] = 1.0
    return ((h - mu) / sd).astype(np.float64), mu, sd


def ridge_residual(h: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x, mu, sd = standardize_fit(h)
    a = np.c_[x, np.ones(len(x))]
    target = np.where(y.astype(int) > 0, 1.0, -1.0)
    weight = np.where(y == 0, 0.5 / max(np.mean(y == 0), EPS), 0.5 / max(np.mean(y == 1), EPS))
    reg = np.eye(a.shape[1], dtype=np.float64) * RESIDUAL_ALPHA
    reg[-1, -1] = 0.0
    ata = a.T @ (a * weight[:, None]) + reg
    aty = a.T @ (target * weight)
    w = np.linalg.solve(ata, aty)
    return {"mu": mu, "sd": sd, "w": w}


def ridge_score(h: np.ndarray, pack: dict[str, Any]) -> np.ndarray:
    x = (h.astype(np.float64) - pack["mu"]) / pack["sd"]
    return np.c_[x, np.ones(len(x))] @ pack["w"]


def source_specs(h: np.ndarray, z: np.ndarray, meta: pd.DataFrame, dataset: str) -> dict[str, Any]:
    y = meta.label.to_numpy(np.int64)
    residual = ridge_residual(h, y)
    x, mu, sd = standardize_fit(h)
    proto = {str(c): x[y == c].mean(0) for c in (0, 1)}
    group = meta.copy()
    group["pos"] = np.arange(len(group))
    cells: dict[tuple[str, int, int], np.ndarray] = {}
    for (subject, session, label), frame in group.groupby(["subject_id", "session_id", "label"], sort=True):
        cells[(str(subject), int(session), int(label))] = x[frame.pos.to_numpy(np.int64)].mean(0)
    subjects = geo.subj_sort(group.subject_id.astype(str).unique())
    sessions = list(geo.SESSIONS_FIT[dataset])
    dirs = []
    paired = 0
    for subject in subjects:
        ds = []
        for session in sessions:
            a, b = cells.get((subject, session, 0)), cells.get((subject, session, 1))
            if a is not None and b is not None:
                d = b - a
                n = float(np.linalg.norm(d))
                if n > EPS:
                    ds.append(d / n)
        if len(ds) == len(sessions):
            dirs.append(np.mean(ds, axis=0)); paired += 1
    if not dirs:
        raise RuntimeError(f"no paired cross-session directions for {dataset}")
    # Explicit leave-one-source-subject-out construction.  Each source
    # subject's direction is excluded from the relation direction used for
    # that subject's inner evaluation; the outer outcome subjects are disjoint
    # from this source pool.  Averaging the five-fold LOO directions gives one
    # fixed relation direction for the unseen outer subjects.
    direction_array = np.asarray(dirs, dtype=np.float64)
    loo = []
    for leave in range(len(direction_array)):
        pool = np.delete(direction_array, leave, axis=0)
        candidate = pool.mean(axis=0)
        candidate /= max(float(np.linalg.norm(candidate)), EPS)
        loo.append(candidate)
    relation_dir = np.mean(np.asarray(loo), axis=0)
    relation_dir /= max(float(np.linalg.norm(relation_dir)), EPS)
    relation_mid = 0.5 * (proto["0"] + proto["1"])
    return {"residual": residual, "feature_mu": mu, "feature_sd": sd, "prototype": proto,
            "relation_dir": relation_dir, "relation_mid": relation_mid,
            "paired_subjects": paired, "fit_sessions": sessions,
            "relation_formula": "leave-one-source-subject-out mean(unit(d_s,t)) then unit-normalize"}


def scores(h: np.ndarray, z: np.ndarray, specs: dict[str, Any]) -> dict[str, np.ndarray]:
    x = (h.astype(np.float64) - specs["feature_mu"]) / specs["feature_sd"]
    baseline = z[:, 1] - z[:, 0]
    residual = baseline + RESIDUAL_SCALE * ridge_score(h, specs["residual"])
    p0, p1 = specs["prototype"]["0"], specs["prototype"]["1"]
    prototype = -np.sum((x - p1) ** 2, axis=1) + np.sum((x - p0) ** 2, axis=1)
    relation = (x - specs["relation_mid"]) @ specs["relation_dir"]
    return {"SUBJECT_BALANCED_ERM": baseline, "GENERIC_RESIDUAL": residual,
            "GENERIC_PROTOTYPE": prototype, "CROSS_SESSION_RELATION": relation}


def score_metrics(meta: pd.DataFrame, score: np.ndarray, method: str, dataset: str) -> list[dict[str, Any]]:
    y = meta.label.to_numpy(np.int64)
    stable = np.clip(score.astype(np.float64), -60.0, 60.0)
    prob1 = 1.0 / (1.0 + np.exp(-stable))
    pred = (prob1 >= 0.5).astype(np.int64)
    rows = []
    for subject, frame in meta.assign(_pos=np.arange(len(meta))).groupby(meta.subject_id.astype(str), sort=True):
        idx = frame._pos.to_numpy(np.int64)
        yy, pp, pr = y[idx], prob1[idx], pred[idx]
        p = np.c_[1.0 - pp, pp]
        rows.append({"dataset": dataset, "fold": FOLD, "seed": SEED, "subject_id": str(subject),
                     "method": method, "BA": float(balanced_accuracy_score(yy, pr)),
                     "accuracy": float(accuracy_score(yy, pr)),
                     "macro_F1": float(f1_score(yy, pr, average="macro", zero_division=0)),
                     "NLL": float(log_loss(yy, p, labels=[0, 1])), "trials": int(len(idx))})
    return rows


def train_dataset(dataset: str, root: Path, device: torch.device, amendment_sha: str) -> dict[str, Any]:
    source_data, role, h, z, meta, mean, info = load_frozen(dataset, root, device)
    specs = source_specs(h, z, meta, dataset)
    runtime = root / "runtime" / f"{dataset}_fold0"
    np.savez_compressed(runtime / "source_specs.npz", feature_mu=specs["feature_mu"], feature_sd=specs["feature_sd"],
                        proto0=specs["prototype"]["0"], proto1=specs["prototype"]["1"],
                        relation_dir=specs["relation_dir"], relation_mid=specs["relation_mid"],
                        residual_mu=specs["residual"]["mu"], residual_sd=specs["residual"]["sd"], residual_w=specs["residual"]["w"])
    audit = {"dataset": dataset, "fold": FOLD, "seed": SEED, "methods": list(METHODS),
             "source_subject_count": int(meta.subject_id.astype(str).nunique()), "source_rows": int(len(meta)),
             "feature_dim": int(h.shape[1]), "paired_subjects": specs["paired_subjects"],
             "fit_sessions": specs["fit_sessions"], "relation_formula": specs["relation_formula"],
             "relation_crossfit": "leave_one_source_subject_out",
             "amendment_sha256": amendment_sha, "outcome_labels_read": False,
             "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False,
             "checkpoint": info, "feature_extraction_sec": json.loads((runtime / "source_manifest.json").read_text())["feature_extraction_sec"]}
    write_json(root / "results" / f"SOURCE_{dataset}_AUDIT.json", audit)
    del source_data, role, h, z, meta, specs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return audit


def write_pre_outcome_lock(root: Path, amendment_sha: str, audits: dict[str, Any]) -> Path:
    artifact_hashes = {}
    for dataset in DATASETS:
        runtime = root / "runtime" / f"{dataset}_fold0"
        for name in ("source_frozen_features.npz", "source_specs.npz", "source_manifest.json"):
            artifact_hashes[f"{dataset}/{name}"] = sha(runtime / name)
    lock = {
        "schema": "PERSIST_EEG_INCREMENTAL_RELATION_PRE_OUTCOME_LOCK_V1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "amendment_sha256": amendment_sha, "training_lock_sha256": sha(PILOT / "INCREMENTAL_RELATION_TRAINING_LOCK.json"),
        "code_sha256": code_hash(), "datasets": list(DATASETS), "folds": [FOLD], "seed": SEED,
        "methods": list(METHODS), "source_audit_sha256": {d: sha(root / "results" / f"SOURCE_{d}_AUDIT.json") for d in DATASETS},
        "artifact_sha256": artifact_hashes, "outcome_labels_read": False,
        "outcome_labels_read_before_lock": False, "WBCIC_outer_10_opened": False,
        "OpenBMI_sealed_holdout_opened": False, "final_claim_authorized": False,
    }
    path = root / "INCREMENTAL_RELATION_PRE_OUTCOME_LOCK.json"
    write_json(path, lock)
    return path


def load_specs(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    return {"feature_mu": z["feature_mu"], "feature_sd": z["feature_sd"],
            "prototype": {"0": z["proto0"], "1": z["proto1"]},
            "relation_dir": z["relation_dir"], "relation_mid": z["relation_mid"],
            "residual": {"mu": z["residual_mu"], "sd": z["residual_sd"], "w": z["residual_w"]}}


def evaluate_dataset(dataset: str, root: Path, device: torch.device) -> list[dict[str, Any]]:
    # This is the first function in the script allowed to materialize outcome data.
    roles, _, _ = ap.load_roles(dataset)
    role = roles[FOLD]
    data = ap.load_ab_data(dataset, set(role["outcome"]))
    ck = checkpoint_path(dataset)
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    mean, std = np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32)
    mask = data.metadata.subject_id.astype(str).isin(set(map(str, role["outcome"]))) & data.metadata.session_id.astype(int).eq(geo.SESSION_OUTCOME[dataset])
    rows = np.flatnonzero(mask.to_numpy()).astype(np.int64)
    model = ap.VanillaEEGNet(int(data.batch(np.asarray([rows[0]], np.int64)).shape[1])).to(device)
    model.load_state_dict(payload["model_state"], strict=True); model.eval()
    hs, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            q = rows[start:start + BATCH_SIZE]
            x = ap.prepare(data, q, mean, std).to(device, non_blocking=True)
            hs.append(model.forward_features(x).detach().cpu().numpy()); logits.append(model(x).detach().cpu().numpy())
    h = np.concatenate(hs, axis=0).astype(np.float32); z = np.concatenate(logits, axis=0).astype(np.float32)
    meta = data.metadata.iloc[rows].reset_index(drop=True).copy()
    specs = load_specs(root / "runtime" / f"{dataset}_fold0" / "source_specs.npz")
    out = []
    for method, score in scores(h, z, specs).items():
        out.extend(score_metrics(meta, score, method, dataset))
    del data, model, h, z, specs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def summarize(root: Path, frame: pd.DataFrame, pre_lock: dict[str, Any]) -> dict[str, Any]:
    summary, deltas, decisions = [], [], {}
    for dataset in DATASETS:
        f = frame[frame.dataset == dataset]
        sb = f[f.method == "SUBJECT_BALANCED_ERM"].set_index("subject_id").sort_index()
        methods = {}
        for method in METHODS:
            z = f[f.method == method]
            summary.append({"dataset": dataset, "fold": FOLD, "seed": SEED, "method": method,
                            "mean_subject_BA": float(z.BA.mean()), "mean_macro_F1": float(z.macro_F1.mean()),
                            "mean_accuracy": float(z.accuracy.mean()), "mean_NLL": float(z.NLL.mean()),
                            "n_subjects": int(z.subject_id.nunique())})
            methods[method] = z.set_index("subject_id").sort_index()
        relation = methods["CROSS_SESSION_RELATION"]
        generic_best = max(methods["GENERIC_RESIDUAL"].BA.mean(), methods["GENERIC_PROTOTYPE"].BA.mean())
        d_ba = (relation.BA - sb.BA) * 100.0; d_f1 = (relation.macro_F1 - sb.macro_F1) * 100.0
        for subject in sb.index:
            deltas.append({"dataset": dataset, "fold": FOLD, "seed": SEED, "subject_id": subject,
                           "SB_ERM_BA": float(sb.loc[subject, "BA"]), "SB_ERM_macro_F1": float(sb.loc[subject, "macro_F1"]),
                           "GENERIC_RESIDUAL_BA": float(methods["GENERIC_RESIDUAL"].loc[subject, "BA"]),
                           "GENERIC_PROTOTYPE_BA": float(methods["GENERIC_PROTOTYPE"].loc[subject, "BA"]),
                           "RELATION_BA": float(relation.loc[subject, "BA"]),
                           "delta_relation_BA_pp": float(d_ba.loc[subject]),
                           "delta_relation_macro_F1_pp": float(d_f1.loc[subject])})
        relation_delta_ba = float(d_ba.mean()); relation_delta_f1 = float(d_f1.mean())
        generic_delta = float(generic_best - sb.BA.mean()) * 100.0
        clear = bool(relation_delta_ba >= CLEAR_PP and relation_delta_f1 >= 0.0 and float(np.mean(d_ba >= 0.0)) >= 0.5 and relation_delta_ba - generic_delta >= CLEAR_PP)
        decisions[dataset] = {"relation_clear_positive": clear, "relation_delta_BA_pp": relation_delta_ba,
                              "relation_delta_macro_F1_pp": relation_delta_f1,
                              "relation_nonnegative_subject_fraction": float(np.mean(d_ba >= 0.0)),
                              "relation_positive_subject_fraction": float(np.mean(d_ba > 0.0)),
                              "strongest_generic_delta_BA_pp": generic_delta,
                              "relation_over_strongest_generic_delta_BA_pp": relation_delta_ba - generic_delta,
                              "worst_relation_subject_delta_BA_pp": float(d_ba.min())}
    write_csv(root / "results" / "INCREMENTAL_RELATION_OUTCOME_PER_SUBJECT.csv", frame)
    write_csv(root / "results" / "INCREMENTAL_RELATION_PERFORMANCE_SUMMARY.csv", summary)
    write_csv(root / "results" / "INCREMENTAL_RELATION_SUBJECT_DELTAS.csv", deltas)
    both_clear = all(decisions[d]["relation_clear_positive"] for d in DATASETS)
    generic_explains = all(decisions[d]["strongest_generic_delta_BA_pp"] >= CLEAR_PP and not decisions[d]["relation_clear_positive"] for d in DATASETS)
    terminal = "INCREMENTAL_RELATION_RESTORE_NEXT_STAGE" if both_clear else ("INCREMENTAL_RELATION_STOP_GENERIC_CONTROL_EXPLAINS_GAIN" if generic_explains else "INCREMENTAL_RELATION_STOP_NO_CLEAR_GAIN")
    result = {"schema": "PERSIST_EEG_INCREMENTAL_RELATION_RESULT_V1", "terminal": terminal,
              "restore_next_stage": both_clear, "dataset_decisions": decisions,
              "methods": list(METHODS), "screen_only": True, "outcome_after_lock": True,
              "outcome_labels_read_before_lock": False, "final_claim_authorized": False,
              "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False,
              "scientific_definition_changed": True}
    write_json(root / "results" / "INCREMENTAL_RELATION_RESULT.json", result)
    rows = ["# Incremental relation frozen-feature pilot", "", f"Terminal: `{terminal}`", "",
            "|Dataset|SB-ERM BA|Residual BA|Prototype BA|Relation BA|Relation ΔBA pp|Relation ΔMacro-F1 pp|Relation vs generic pp|Clear|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lookup = {(r["dataset"], r["method"]): r for r in summary}
    for d in DATASETS:
        x = decisions[d]
        rows.append(f"|{d}|{lookup[(d,'SUBJECT_BALANCED_ERM')]['mean_subject_BA']:.4f}|{lookup[(d,'GENERIC_RESIDUAL')]['mean_subject_BA']:.4f}|{lookup[(d,'GENERIC_PROTOTYPE')]['mean_subject_BA']:.4f}|{lookup[(d,'CROSS_SESSION_RELATION')]['mean_subject_BA']:.4f}|{x['relation_delta_BA_pp']:.3f}|{x['relation_delta_macro_F1_pp']:.3f}|{x['relation_over_strongest_generic_delta_BA_pp']:.3f}|{x['relation_clear_positive']}|")
    rows += ["", "This is a one-fold seed-0 frozen-feature pilot, not a formal multi-fold claim.", ""]
    (root / "results" / "INCREMENTAL_RELATION_REPORT.md").write_text("\n".join(rows), encoding="utf-8")
    write_json(root / "results" / "VALIDATION.json", {"pass": True, "terminal": terminal,
                                                        "outcome_after_lock": True, "screen_only": True,
                                                        "final_claim_authorized": False,
                                                        "WBCIC_outer_10_opened": False,
                                                        "OpenBMI_sealed_holdout_opened": False})
    return result


def phase_train(root: Path, device: torch.device) -> dict[str, Any]:
    amendment_sha = build_locks()
    audits = {}
    for dataset in DATASETS:
        audits[dataset] = train_dataset(dataset, root, device, amendment_sha)
    pre_path = write_pre_outcome_lock(root, amendment_sha, audits)
    return {"amendment_sha256": amendment_sha, "pre_outcome_lock": str(pre_path), "audits": audits}


def phase_outcome(root: Path, device: torch.device) -> dict[str, Any]:
    pre_path = root / "INCREMENTAL_RELATION_PRE_OUTCOME_LOCK.json"
    if not pre_path.is_file():
        raise RuntimeError("pre-outcome lock missing")
    pre = read_json(pre_path)
    if pre.get("outcome_labels_read") is not False or pre.get("methods") != list(METHODS):
        raise RuntimeError("pre-outcome lock invalid")
    access = {
        "schema": "PERSIST_EEG_INCREMENTAL_RELATION_OUTCOME_ACCESS_LOCK_V1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_outcome_lock_sha256": sha(pre_path), "amendment_sha256": pre["amendment_sha256"],
        "datasets": list(DATASETS), "folds": [FOLD], "methods": list(METHODS),
        "outcome_labels_read": False, "outcome_labels_read_before_lock": False,
        "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False,
        "screen_only": True, "final_claim_authorized": False,
    }
    access_path = root / "INCREMENTAL_RELATION_OUTCOME_ACCESS_LOCK.json"
    write_json(access_path, access)
    rows = []
    for dataset in DATASETS:
        rows.extend(evaluate_dataset(dataset, root, device))
    result = summarize(root, pd.DataFrame(rows), pre)
    access["outcome_labels_read"] = True; access["outcome_labels_read_after_lock"] = True
    access["result_sha256"] = sha(root / "results" / "INCREMENTAL_RELATION_RESULT.json")
    write_json(access_path, access)
    legal = {"schema": "PERSIST_EEG_INCREMENTAL_RELATION_LEGALITY_V1", "seed": SEED,
             "datasets": list(DATASETS), "folds": [FOLD], "methods": list(METHODS),
             "amendment_sha256": pre["amendment_sha256"], "outcome_labels_read_before_lock": False,
             "outcome_labels_read_after_lock": True, "WBCIC_outer_10_opened": False,
             "OpenBMI_sealed_holdout_opened": False, "access_lock_sha256": sha(access_path)}
    write_json(root / "DATA_LEGALITY_AUDIT.json", legal)
    print(result["terminal"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "outcome", "all"), default="all")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.phase in ("train", "all"):
        phase_train(root, device)
    if args.phase in ("outcome", "all"):
        phase_outcome(root, device)


if __name__ == "__main__":
    main()
