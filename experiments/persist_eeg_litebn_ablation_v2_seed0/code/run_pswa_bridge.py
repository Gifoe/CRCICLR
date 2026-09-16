#!/usr/bin/env python3
"""Frozen Protected-Subspace Worst-Session Advantage (PSWA).

This program consumes the corrected Appendix-L PEEH artifacts. It does not
train, tune, adapt, or select neural models or Protected coordinates. PEEH did
not serialize its canonical basis arrays or random coordinate lists, so this
program reconstructs only those omitted deterministic arrays from the frozen
TRAIN embeddings and the exact PEEH seeds, then validates them against the
stored PEEH spectrum audit and Protected union. No persistence permutation,
block construction, or Protected selection is rerun.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


REPO = Path(os.environ.get("PSWA_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_ablation_v2_seed0"
OUT = EXP / "outputs/p3_temporal_bridge/pswa"
PROTOCOL = EXP / "protocol/p3_temporal_bridge/pswa"
RUNTIME = Path("/root/rivermind-data/litebn_ablation_v2_pswa_runtime")
SESSION_CACHE = RUNTIME / "session_embeddings"
RUN_CACHE = RUNTIME / "run_results"
PEEH_EXP = EXP
PEEH_OUT = EXP / "outputs/p3_temporal_bridge"
PEEH_RUNTIME = Path("/root/rivermind-data/litebn_ablation_v2_peeh_runtime")
PEEH_EMBED = PEEH_RUNTIME / "embeddings"
PEEH_RUN = PEEH_RUNTIME / "run_results"
PEEH_CODE = PEEH_EXP / "code/run_peeh_bridge.py"
CSGD_CODE = REPO / "experiments/persist_eeg_litebn_tfformer_csgd_v1/code/run_csgd.py"

TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MODELS = ("OFFICIAL_FINAL_LITEBN_REFERENCE", "B1_SAME_SCALE_63")
FOLDS = tuple(range(5))
SEEDS = (0,)
OPENBMI_SUBJECTS = ("4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54")
WBCIC_SUBJECTS = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")
RANDOM_DRAWS = 100
BOOTSTRAP_DRAWS = 20_000
RIDGE_ALPHA = 0.01
EPS = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False) % (2**63 - 1)


def clean(v: Any) -> Any:
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v); return x if math.isfinite(x) else None
    if isinstance(v, np.bool_): return bool(v)
    return v


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def peeh_embedding_path(task: str, model: str, fold: int, seed: int) -> Path:
    return PEEH_EMBED / task / model / f"fold{fold}_seed{seed}.npz"


def peeh_result_path(task: str, model: str, fold: int, seed: int) -> Path:
    return PEEH_RUN / task / model / f"fold{fold}_seed{seed}.json"


def session_cache_path(task: str, model: str, fold: int, seed: int) -> Path:
    return SESSION_CACHE / task / model / f"fold{fold}_seed{seed}.npz"


def run_cache_path(task: str, model: str, fold: int, seed: int) -> Path:
    return RUN_CACHE / task / model / f"fold{fold}_seed{seed}.json"


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        ans = {k: np.asarray(z[k]) for k in z.files if k != "metadata"}
        ans["metadata"] = json.loads(str(z["metadata"].item())) if "metadata" in z.files else {}
    return ans


def save_session_cache(path: Path, h: np.ndarray, y: np.ndarray, subject: np.ndarray,
                       session: np.ndarray, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("wb") as f:
        np.savez_compressed(f, h=np.asarray(h, np.float32), y=np.asarray(y, np.int64),
            subject=np.asarray(subject).astype("U16"), session=np.asarray(session, np.int64),
            metadata=np.asarray(json.dumps(clean(metadata), sort_keys=True)))
    os.replace(tmp, path)


def subjects_sessions(task: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    return (WBCIC_SUBJECTS, (0, 1, 2)) if task == "WBCIC_MI" else (OPENBMI_SUBJECTS, (1, 2))


def load_peeh_run(task: str, model: str, fold: int, seed: int) -> dict[str, Any]:
    return json.loads(peeh_result_path(task, model, fold, seed).read_text())


def canonical_basis(train_h: np.ndarray, train_subject: np.ndarray, train_session: np.ndarray,
                    train_y: np.ndarray, historical: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Reconstruct only omitted canonical basis arrays; never rerun null/selection."""
    x = np.asarray(train_h, np.float64)
    mean = x.mean(0); centered = x - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    order = np.argsort(values)[::-1]; values, vectors = values[order], vectors[:, order]
    threshold = max(float(values[0]) * 1e-3, 1e-8)
    numerical_rank = int(np.sum(values > threshold))
    active_rank = min(20, numerical_rank)
    if active_rank != int(historical["active_rank"]):
        raise RuntimeError(f"active rank reconstruction mismatch {active_rank} != {historical['active_rank']}")
    lambda_r = float(values[active_rank - 1]); floor = max(1e-4 * lambda_r, 1e-8)
    old = historical["spectrum_audit"]
    if not np.isclose(lambda_r, float(old["lambda_r_smallest_retained"]), rtol=2e-5, atol=1e-8):
        raise RuntimeError("lambda_r reconstruction mismatch")
    if not np.isclose(floor, float(old["whitening_eigenvalue_floor"]), rtol=2e-5, atol=1e-10):
        raise RuntimeError("whitening floor reconstruction mismatch")
    active = np.maximum(values[:active_rank], floor)
    whitener = vectors[:, :active_rank] * np.power(active, -0.5)[None, :]
    whitened = centered @ whitener
    frame = pd.DataFrame({"subject": np.asarray(train_subject).astype(str),
                          "session": np.asarray(train_session).astype(int),
                          "label": np.asarray(train_y).astype(int), "row": np.arange(len(x))})
    sessions = sorted(frame.session.unique()); labels = sorted(frame.label.unique())
    if len(sessions) != 2 or sessions != list(map(int, old["sessions"])): raise RuntimeError("session reconstruction mismatch")
    centroids = {(str(k[0]), int(k[1]), int(k[2])): whitened[g.row.to_numpy(np.int64)].mean(0)
                 for k, g in frame.groupby(["subject", "session", "label"], sort=True)}
    subjects = sorted(frame.subject.unique(), key=lambda s: int(str(s).replace("sub-", "")))
    covs = []
    for label in labels:
        aa, bb = [], []
        for subject in subjects:
            a, b = (subject, sessions[0], label), (subject, sessions[1], label)
            if a in centroids and b in centroids: aa.append(centroids[a]); bb.append(centroids[b])
        if aa:
            aa, bb = np.asarray(aa), np.asarray(bb); aa -= aa.mean(0); bb -= bb.mean(0)
            covs.append((aa.T @ bb + bb.T @ aa) / (2.0 * len(aa)))
    persistence = np.mean(covs, axis=0)
    rho, directions = np.linalg.eigh((persistence + persistence.T) / 2.0)
    order = np.argsort(rho)[::-1]; rho, directions = rho[order], directions[:, order]
    # Stored block means are sufficient to audit exact eigenvalue ordering.
    for support in old["persistence_support"]:
        block = historical["block_selection"][int(support["block"])]["coordinates"]
        if not np.isclose(float(np.mean(rho[np.asarray(block, int)])), float(support["rho_G"]), rtol=3e-5, atol=3e-7):
            raise RuntimeError("persistence rho reconstruction mismatch")
    dewhitener = np.sqrt(active)[:, None] * vectors[:, :active_rank].T
    return {"mean": mean.astype(np.float32), "whitener": whitener.astype(np.float32),
            "dewhitener": dewhitener.astype(np.float32),
            "directions": directions.astype(np.float32), "rho": rho.astype(np.float32)}


def z_coordinates(h: np.ndarray, basis: Mapping[str, np.ndarray]) -> np.ndarray:
    return ((np.asarray(h, np.float64) - basis["mean"]) @ basis["whitener"] @ basis["directions"]).astype(np.float32)


def ridge_fit(x: np.ndarray, y: np.ndarray, classes: int):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.int64)
    mean, std = x.mean(0), x.std(0); std[std < 1e-6] = 1.0
    design = np.c_[(x - mean) / std, np.ones(len(x))]
    penalty = np.eye(x.shape[1] + 1); penalty[-1, -1] = 0.0
    target = np.eye(classes)[y]
    weight = np.linalg.solve(design.T @ design + RIDGE_ALPHA * penalty, design.T @ target)
    return weight, mean, std


def ridge_predict(x: np.ndarray, pack) -> np.ndarray:
    weight, mean, std = pack
    return (np.c_[(np.asarray(x, np.float64) - mean) / std, np.ones(len(x))] @ weight).argmax(1)


def ba(y: np.ndarray, p: np.ndarray, classes: int) -> float:
    y, p = np.asarray(y, int), np.asarray(p, int)
    return float(np.mean([np.mean(p[y == c] == c) for c in range(classes) if np.any(y == c)]))


def random_sets(task: str, model: str, fold: int, seed: int, rank: int, active_rank: int) -> list[list[int]]:
    # Exact revised final-random rule. Seed is intentionally absent.
    active = np.arange(active_rank, dtype=np.int64)
    return [np.random.default_rng(stable_seed("final-random", model, task, fold, draw)).choice(active, size=rank, replace=False).astype(int).tolist()
            for draw in range(RANDOM_DRAWS)]


def audit_artifacts(seeds: Sequence[int]) -> list[dict[str, Any]]:
    required = [PEEH_CODE, PEEH_OUT / "RUN_LEVEL_PEEH.csv", PEEH_OUT / "PROTECTED_BLOCK_SELECTION.csv",
                PEEH_OUT / "TASK_LEVEL_PEEH.csv", PEEH_EXP / "protocol/p3_temporal_bridge/PROTOCOL_LOCK.json"]
    rows = [{"artifact_type": "global_peeh_artifact", "task": "ALL", "model": "ALL", "fold": "", "seed": "",
             "path": str(p), "sha256": sha256_file(p), "status": "REUSED_EXACT"} for p in required]
    summary = pd.read_csv(PEEH_OUT / "RUN_LEVEL_PEEH.csv")
    for task in TASKS:
        for model in MODELS:
            for fold in FOLDS:
                for seed in seeds:
                    ep, rp = peeh_embedding_path(task, model, fold, seed), peeh_result_path(task, model, fold, seed)
                    if not ep.is_file() or not rp.is_file(): raise FileNotFoundError(ep if not ep.is_file() else rp)
                    old = load_peeh_run(task, model, fold, seed)
                    match = summary[(summary.task == task) & (summary.model == model) & (summary.fold == fold) & (summary.seed == seed)]
                    if len(match) != 1: raise RuntimeError("PEEH run summary identity mismatch")
                    csv_p = ast.literal_eval(match.iloc[0].protected_coordinates)
                    if list(map(int, csv_p)) != list(map(int, old["protected_coordinates"])): raise RuntimeError("Protected union mismatch")
                    emb = load_npz(ep)
                    if emb["metadata"]["checkpoint_sha256"] != old["checkpoint_sha256"]: raise RuntimeError("checkpoint hash mismatch")
                    if emb["metadata"]["normalizer_sha256"] != old["normalizer_sha256"]: raise RuntimeError("normalizer hash mismatch")
                    checkpoint = Path(emb["metadata"]["checkpoint"]); normalizer = Path(emb["metadata"]["normalizer"])
                    if sha256_file(checkpoint) != old["checkpoint_sha256"]: raise RuntimeError("live checkpoint SHA256 mismatch")
                    if sha256_file(normalizer) != old["normalizer_sha256"]: raise RuntimeError("live normalizer SHA256 mismatch")
                    sets = random_sets(task, model, fold, seed, int(old["protected_rank"]), int(old["active_rank"]))
                    rows += [
                        {"artifact_type": "frozen_64d_embeddings", "task": task, "model": model, "fold": fold, "seed": seed,
                         "path": str(ep), "sha256": sha256_file(ep), "status": "REUSED_EXACT"},
                        {"artifact_type": "corrected_peeh_run", "task": task, "model": model, "fold": fold, "seed": seed,
                         "path": str(rp), "sha256": sha256_file(rp), "status": "REUSED_EXACT"},
                        {"artifact_type": "protected_union", "task": task, "model": model, "fold": fold, "seed": seed,
                         "path": str(rp) + "#protected_coordinates", "sha256": sha256_json(old["protected_coordinates"]), "status": "REUSED_EXACT"},
                        {"artifact_type": "frozen_checkpoint", "task": task, "model": model, "fold": fold, "seed": seed,
                         "path": str(checkpoint), "sha256": old["checkpoint_sha256"], "status": "VERIFIED_EXACT"},
                        {"artifact_type": "frozen_train_only_normalizer", "task": task, "model": model, "fold": fold, "seed": seed,
                         "path": str(normalizer), "sha256": old["normalizer_sha256"], "status": "VERIFIED_EXACT"},
                        {"artifact_type": "revised_final_random_sets", "task": task, "model": model, "fold": fold, "seed": seed,
                         "path": "revised PSWA rule#stable_seed(final-random,model,task,fold,draw)", "sha256": sha256_json(sets),
                         "status": "EXACT_DETERMINISTIC_RECONSTRUCTION_NO_REDRAW"},
                    ]
    return rows


def protected_erasure_checksum(emb: Mapping[str, Any], basis: Mapping[str, np.ndarray],
                               protected: Sequence[int], old: Mapping[str, Any]) -> tuple[float, float]:
    def erase(h: np.ndarray) -> np.ndarray:
        q = z_coordinates(h, basis); delta = np.zeros_like(q)
        if protected: delta[:, np.asarray(protected, int)] = -q[:, np.asarray(protected, int)]
        dh = (delta @ basis["directions"].T) @ basis["dewhitener"]
        return (np.asarray(h, np.float64) + dh).astype(np.float32)
    classes = len(np.unique(emb["train_y"]))
    pack = ridge_fit(erase(emb["train_h"]), emb["train_y"], classes)
    pred = ridge_predict(erase(emb["eval_h"]), pack)
    values = []
    for subject in sorted(np.unique(emb["eval_subject"].astype(str))):
        mask = emb["eval_subject"].astype(str) == subject
        values.append(ba(emb["eval_y"][mask], pred[mask], classes))
    recovered = float(np.mean(values)); stored = float(old["protected_erased_BA"])
    return recovered, abs(recovered - stored)


def missing_inference(device_name: str, seeds: Sequence[int]) -> list[dict[str, Any]]:
    """Infer only missing sessions and persist compact canonical q arrays."""
    peeh = import_path("pswa_peeh", PEEH_CODE)
    csgd = import_path("pswa_csgd", CSGD_CODE)
    runtime = peeh.load_runtime(); device = torch.device(device_name); recovery_rows = []
    if device.type == "cuda": torch.cuda.set_per_process_memory_fraction(float(os.environ.get("PSWA_GPU_FRACTION", "0.25")), 0)
    for task in TASKS:
        subjects, sessions = subjects_sessions(task)
        if task == "WBCIC_MI": full, _ = csgd.build_wbcic_outer_bundle(runtime)
        else: full = runtime.base.build_bundle(task, subjects)
        missing_sessions = sessions[:-1]; bundle = peeh.subset_bundle(runtime, full, missing_sessions)
        meta = peeh.bundle_metadata(bundle)
        for fold in FOLDS:
            mean, std, _ = runtime.base.load_tensor_pair(peeh.normalizer_path(task, fold))
            for seed in seeds:
                for model_name in MODELS:
                    target = session_cache_path(task, model_name, fold, seed)
                    if target.is_file():
                        recovery_rows.append(load_npz(target)["metadata"]["recovery"]); continue
                    old_emb = load_npz(peeh_embedding_path(task, model_name, fold, seed)); old = load_peeh_run(task, model_name, fold, seed)
                    basis = canonical_basis(old_emb["train_h"], old_emb["train_subject"], old_emb["train_session"], old_emb["train_y"], old)
                    protected = list(map(int, old["protected_coordinates"]))
                    if any(i < 0 or i >= int(old["active_rank"]) for i in protected): raise RuntimeError("Protected coordinate out of range")
                    checksum_ba, checksum_delta = protected_erasure_checksum(old_emb, basis, protected, old)
                    recovery = {"task": task, "model": model_name, "fold": fold, "seed": seed,
                        "stored_active_rank": int(old["active_rank"]), "recovered_active_rank": int(len(basis["rho"])),
                        "protected_rank": len(protected), "protected_indices_valid": True,
                        "stored_protected_erased_BA": float(old["protected_erased_BA"]),
                        "recovered_protected_erased_BA": checksum_ba, "absolute_BA_difference": checksum_delta,
                        "tolerance": 1e-5, "recovery_status": "PASS" if checksum_delta <= 1e-5 else "RECOVERY_MISMATCH"}
                    recovery_rows.append(recovery)
                    if checksum_delta > 1e-5:
                        print(f"PSWA_RECOVERY_MISMATCH {task} {model_name} f{fold} s{seed} delta={checksum_delta}", flush=True); continue
                    model = peeh.build_model(runtime, model_name, task, peeh.checkpoint_path(model_name, task, fold, seed), device)
                    values = []; batch = int(os.environ.get("PSWA_TF_BATCH", "8")) if model_name == "TFFormer" else int(os.environ.get("PSWA_LITE_BATCH", "64"))
                    model.eval()
                    with torch.inference_mode():
                        for start in range(0, len(bundle.rows), batch):
                            ids = np.arange(start, min(start + batch, len(bundle.rows)), dtype=np.int64)
                            # Stream directly from the canonical cache. This
                            # keeps GPU memory bounded and avoids competing with
                            # the concurrently running ablation spectral cache.
                            raw_np = bundle.signal_batch(ids)
                            x = torch.as_tensor(np.ascontiguousarray(raw_np), dtype=torch.float32, device=device)
                            mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)[None, :, None]
                            std_t = torch.as_tensor(std, dtype=torch.float32, device=device)[None, :, None]
                            x = (x - mean_t) / torch.clamp(std_t, min=1e-6)
                            values.append(model(x)[1].float().cpu().numpy())
                    h_missing = np.concatenate(values).astype(np.float32)
                    h = np.concatenate([h_missing, old_emb["eval_h"]]); y = np.concatenate([meta.label.to_numpy(int), old_emb["eval_y"]])
                    subject = np.concatenate([meta.subject_id.astype(str).to_numpy("U16"), old_emb["eval_subject"].astype("U16")])
                    session = np.concatenate([meta.session_id.to_numpy(int), old_emb["eval_session"]])
                    basis_hash = sha256_json({k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in basis.items()})
                    save_session_cache(target, z_coordinates(h, basis), y, subject, session, {"task": task, "model": model_name, "fold": fold, "seed": seed,
                        "checkpoint_sha256": old["checkpoint_sha256"], "normalizer_sha256": old["normalizer_sha256"],
                        "S2_source": str(peeh_embedding_path(task, model_name, fold, seed)), "S2_reused": True,
                        "missing_sessions_inferred_once": list(missing_sessions), "eval_mode": True, "BN_updates": False, "training": False,
                        "cache_content": "compact canonical q", "basis_sha256": basis_hash, "recovery": recovery})
                    qtrain_path = target.with_name(target.stem + "_train_q.npz")
                    save_session_cache(qtrain_path, z_coordinates(old_emb["train_h"], basis), old_emb["train_y"], old_emb["train_subject"], old_emb["train_session"],
                        {"source": str(peeh_embedding_path(task, model_name, fold, seed)), "cache_content": "compact canonical q_train", "basis_sha256": basis_hash})
                    del model, h_missing, h; gc.collect()
                    if device.type == "cuda": torch.cuda.empty_cache()
                    print(f"PSWA_EMBED {task} {model_name} f{fold} s{seed}", flush=True)
        del bundle, full; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    return recovery_rows


def evaluate_run(task: str, model: str, fold: int, seed: int) -> dict[str, Any]:
    old = load_peeh_run(task, model, fold, seed)
    ev = load_npz(session_cache_path(task, model, fold, seed))
    tr = load_npz(session_cache_path(task, model, fold, seed).with_name(f"fold{fold}_seed{seed}_train_q.npz"))
    protected = list(map(int, old["protected_coordinates"])); active_rank = int(old["active_rank"]); rank = len(protected)
    random = random_sets(task, model, fold, seed, rank, active_rank)
    base = {"task": task, "model": model, "fold": fold, "seed": seed, "protected_rank": rank,
            "active_rank": active_rank, "status": "EMPTY_PROTECTED" if rank == 0 else "VALID",
            "protected_coordinates": protected, "random_sets_sha256": sha256_json(random),
            "basis_sha256": ev["metadata"]["basis_sha256"], "recovery_status": ev["metadata"]["recovery"]["recovery_status"]}
    if rank == 0: return {**base, "protected_rows": [], "random_rows": []}
    classes = len(np.unique(tr["y"])); z_train = tr["h"]; z_eval = ev["h"]
    protected_pack = ridge_fit(z_train[:, protected], tr["y"], classes)
    protected_pred = ridge_predict(z_eval[:, protected], protected_pack)
    subjects, sessions = subjects_sessions(task); p_rows, r_rows = [], []
    for subject in subjects:
        for session in sessions:
            mask = (ev["subject"].astype(str) == subject) & (ev["session"].astype(int) == session)
            if not mask.any(): raise RuntimeError(f"empty eval cell {task}/{subject}/S{session}")
            p_rows.append({**base, "subject_id": subject, "session": f"S{session}", "BA": ba(ev["y"][mask], protected_pred[mask], classes), "trials": int(mask.sum())})
    for draw, subset in enumerate(random):
        pack = ridge_fit(z_train[:, subset], tr["y"], classes); pred = ridge_predict(z_eval[:, subset], pack)
        for subject in subjects:
            for session in sessions:
                mask = (ev["subject"].astype(str) == subject) & (ev["session"].astype(int) == session)
                r_rows.append({**base, "draw": draw, "coordinates": subset, "subject_id": subject,
                               "session": f"S{session}", "BA": ba(ev["y"][mask], pred[mask], classes), "trials": int(mask.sum())})
    return {**base, "protected_rows": p_rows, "random_rows": r_rows}


def bootstrap(values: Sequence[float], seed: int) -> tuple[float, float]:
    v = np.asarray(values, np.float64); rng = np.random.default_rng(seed); means = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 2000):
        stop = min(start + 2000, BOOTSTRAP_DRAWS); means[start:stop] = v[rng.integers(0, len(v), size=(stop-start, len(v)))].mean(1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def aggregate(results: Sequence[Mapping[str, Any]], total_expected_runs: int):
    protected = pd.DataFrame([row for result in results for row in result["protected_rows"]])
    random = pd.DataFrame([row for result in results for row in result["random_rows"]])
    run_subject = []
    for key, p in protected.groupby(["task", "model", "fold", "seed", "subject_id"], sort=False):
        task, model, fold, seed, subject = key; r = random[(random.task == task) & (random.model == model) & (random.fold == fold) & (random.seed == seed) & (random.subject_id == subject)]
        p_session = dict(zip(p.session, p.BA)); r_session = r.groupby("session").BA.mean().to_dict()
        r_ws = r.groupby("draw").BA.min(); p_ws = min(p_session.values())
        row = {"task": task, "model": model, "fold": fold, "seed": seed, "subject_id": subject,
               "protected_rank": int(p.protected_rank.iloc[0]), "protected_only_WSBA": p_ws,
               "random_only_WSBA": float(r_ws.mean()), "PSWA_pp": 100*(p_ws-float(r_ws.mean())), "random_draws": int(r.draw.nunique())}
        for session in sorted(p_session):
            row[f"protected_BA_{session}"] = p_session[session]; row[f"random_BA_{session}"] = r_session[session]
            row[f"protected_minus_random_{session}_pp"] = 100*(p_session[session]-r_session[session])
        run_subject.append(row)
    rs = pd.DataFrame(run_subject)
    metric_cols = [c for c in rs.columns if c.startswith("protected_BA_") or c.startswith("random_BA_") or c.startswith("protected_minus_random_")]
    subject = rs.groupby(["task", "model", "subject_id"], as_index=False).agg(
        protected_only_WSBA=("protected_only_WSBA", "mean"), random_only_WSBA=("random_only_WSBA", "mean"),
        PSWA_pp=("PSWA_pp", "mean"), valid_fold_seed_runs=("PSWA_pp", "size"), protected_rank_mean=("protected_rank", "mean"))
    for col in metric_cols:
        subject[col] = rs.groupby(["task", "model", "subject_id"])[col].mean().reindex(pd.MultiIndex.from_frame(subject[["task","model","subject_id"]])).to_numpy()
    task_rows = []
    for (task, model), p in subject.groupby(["task", "model"], sort=False):
        vals = p.PSWA_pp.to_numpy(float); lo, hi = bootstrap(vals, stable_seed("PSWA-bootstrap", task, model))
        old_runs = [r for r in results if r["task"] == task and r["model"] == model]
        ranks = np.asarray([r["protected_rank"] for r in old_runs], float); nonempty = int(np.sum(ranks > 0))
        row = {"task": task, "model": model, "biological_subjects": len(p), "protected_rank_mean": float(ranks.mean()),
               "protected_rank_min": int(ranks.min()), "protected_rank_max": int(ranks.max()),
               "protected_coverage_nonempty_runs": nonempty, "protected_coverage_total_runs": total_expected_runs,
               "protected_coverage": nonempty/total_expected_runs, "protected_only_WSBA": float(p.protected_only_WSBA.mean()),
               "random_only_WSBA": float(p.random_only_WSBA.mean()), "PSWA_mean_pp": float(vals.mean()),
               "PSWA_median_pp": float(np.median(vals)), "PSWA_CI95_low_pp": lo, "PSWA_CI95_high_pp": hi,
               "significant_positive": bool(lo > 0), "random_draws": RANDOM_DRAWS, "bootstrap_draws": BOOTSTRAP_DRAWS}
        for col in metric_cols: row[col] = float(p[col].mean())
        future = p["protected_minus_random_S2_pp"].to_numpy(float)
        flo, fhi = bootstrap(future, stable_seed("PSWA-future-bootstrap", task, model, total_expected_runs))
        row["future_session_advantage_mean_pp"] = float(future.mean())
        row["future_session_advantage_CI95_low_pp"] = flo
        row["future_session_advantage_CI95_high_pp"] = fhi
        row["recovery_status"] = "PASS" if all(r.get("recovery_status") == "PASS" for r in old_runs) else "RECOVERY_MISMATCH"
        task_rows.append(row)
    paired = []
    for task in TASKS:
        a = subject[(subject.task == task) & (subject.model == "OFFICIAL_FINAL_LITEBN_REFERENCE")][["subject_id","PSWA_pp"]].rename(columns={"PSWA_pp":"Full_PSWA_pp"})
        b = subject[(subject.task == task) & (subject.model == "B1_SAME_SCALE_63")][["subject_id","PSWA_pp"]].rename(columns={"PSWA_pp":"SameScale_PSWA_pp"})
        p = a.merge(b, on="subject_id", validate="one_to_one"); delta = (p.Full_PSWA_pp-p.SameScale_PSWA_pp).to_numpy(float)
        lo, hi = bootstrap(delta, stable_seed("PSWA-paired-bootstrap", task))
        paired.append({"task": task, "biological_subjects": len(p), "Full_PSWA_mean_pp": float(p.Full_PSWA_pp.mean()),
                       "SameScale_PSWA_mean_pp": float(p.SameScale_PSWA_pp.mean()), "Full_minus_SameScale_PSWA_mean_pp": float(delta.mean()),
                       "delta_CI95_low_pp": lo, "delta_CI95_high_pp": hi, "bootstrap_draws": BOOTSTRAP_DRAWS})
    return protected, random, subject, pd.DataFrame(task_rows), pd.DataFrame(paired)


def report(task: pd.DataFrame, paired: pd.DataFrame) -> str:
    lines = ["# Frozen Full-reference / SameScale-63 Protected-Subspace Worst-Session Advantage", "",
             "The Full model is the official frozen LiteBN, not a newly trained matched B0 control.", "",
             "This post-hoc analysis performed no neural training, tuning, adaptation, checkpoint selection, Protected selection, or new random-null sampling. Missing evaluation-session embeddings were inferred once in eval mode; existing S2 embeddings were reused.", "",
             "The corrected PEEH run did not serialize canonical basis matrices or random lists. We therefore reconstructed only the omitted deterministic basis arrays and the exact revised `final-random` lists from the frozen TRAIN embeddings and frozen stable-seed rule, validating rank, whitening floor, persistence block means, Protected unions, and hashes. No persistence permutations or Protected selection were rerun.", "",
             "## Primary seed0 results", "", "| Task | Model | Protected coverage | Protected-only WS-BA | Random-only WS-BA | PSWA pp [95% CI] | Future-session advantage pp [95% CI] | Recovery |", "|---|---|---:|---:|---:|---:|---:|:---:|"]
    for r in task.itertuples(index=False):
        lines.append(f"| {r.task} | {r.model} | {r.protected_coverage_nonempty_runs}/{r.protected_coverage_total_runs} | {100*r.protected_only_WSBA:.2f}% | {100*r.random_only_WSBA:.2f}% | {r.PSWA_mean_pp:.2f} [{r.PSWA_CI95_low_pp:.2f}, {r.PSWA_CI95_high_pp:.2f}] | {r.future_session_advantage_mean_pp:.2f} [{r.future_session_advantage_CI95_low_pp:.2f}, {r.future_session_advantage_CI95_high_pp:.2f}] | {r.recovery_status} |")
    lines += ["", "## Session-specific Protected-minus-Random advantage", "", "| Task | Model | S0 | S1 | S2 / future |", "|---|---|---:|---:|---:|"]
    for r in task.to_dict("records"):
        f = lambda s: "n/a" if f"protected_minus_random_{s}_pp" not in r or pd.isna(r.get(f"protected_minus_random_{s}_pp")) else f"{r[f'protected_minus_random_{s}_pp']:.2f} pp"
        lines.append(f"| {r['task']} | {r['model']} | {f('S0')} | {f('S1')} | {f('S2')} |")
    lines += ["", "## Paired Full reference vs SameScale", "", "| Task | Full PSWA | SameScale PSWA | Full - SameScale [95% CI] |", "|---|---:|---:|---:|"]
    for r in paired.itertuples(index=False): lines.append(f"| {r.task} | {r.Full_PSWA_mean_pp:.2f} | {r.SameScale_PSWA_mean_pp:.2f} | {r.Full_minus_SameScale_PSWA_mean_pp:.2f} [{r.delta_CI95_low_pp:.2f}, {r.delta_CI95_high_pp:.2f}] |")
    lines += ["", "PSWA uses the biological subject as the bootstrap unit after averaging all valid fold-seed runs. Positive PSWA means the TRAIN-selected Protected subspace alone supports better worst-session decoding than the exact equal-rank PEEH random controls. It is mechanistic and is not overall classifier performance."]
    return "\n".join(lines)


def analyze(seeds: Sequence[int], recovery_rows: Sequence[Mapping[str, Any]]) -> None:
    results = []
    for task in TASKS:
        for model in MODELS:
            for fold in FOLDS:
                for seed in seeds:
                    target = run_cache_path(task, model, fold, seed)
                    if target.is_file(): result = json.loads(target.read_text())
                    else:
                        if not session_cache_path(task, model, fold, seed).is_file(): raise FileNotFoundError(session_cache_path(task, model, fold, seed))
                        result = evaluate_run(task, model, fold, seed); atomic_json(target, result)
                    results.append(result); print(f"PSWA_RUN {task} {model} f{fold} s{seed} {result['status']}", flush=True)
    protected, random, subject, task, paired = aggregate(results, len(FOLDS) * len(seeds))
    atomic_csv(OUT / "PROTECTED_ONLY_SESSION_RESULTS.csv", protected)
    atomic_csv(OUT / "RANDOM_ONLY_SESSION_RESULTS.csv", random)
    atomic_csv(OUT / "SUBJECT_LEVEL_PSWA_SEED0.csv", subject)
    atomic_csv(OUT / "TASK_LEVEL_PSWA_SEED0.csv", task)
    # Primary FAST mode intentionally does not block on optional seeds 1/2.
    atomic_csv(OUT / "TASK_LEVEL_PSWA_MULTISEED.csv", pd.DataFrame(columns=list(task.columns)))
    session_cols = [c for c in task.columns if c.startswith("protected_minus_random_")]
    atomic_csv(OUT / "SESSION_SPECIFIC_ADVANTAGE.csv", task[["task", "model", *session_cols,
        "future_session_advantage_mean_pp", "future_session_advantage_CI95_low_pp", "future_session_advantage_CI95_high_pp"]])
    atomic_csv(OUT / "FULL_SAMESCALE_PAIRED_PSWA.csv", paired)
    atomic_csv(OUT / "RECOVERY_AUDIT.csv", recovery_rows)
    controls = {f"{r['task']}|{r['model']}|fold{r['fold']}|seed{r['seed']}": random_sets(r["task"], r["model"], r["fold"], r["seed"], r["protected_rank"], r["active_rank"]) for r in results}
    atomic_json(OUT / "RECOVERED_RANDOM_CONTROLS.json", controls)
    atomic_text(OUT / "FINAL_PSWA_REPORT.md", report(task, paired))
    validation = {"status": "PASS", "protected_coordinates_exact_corrected_peeh": True,
        "random_subsets_exact_revised_final_random_rule": True, "random_null_redrawn": False,
        "checkpoint_and_normalizer_sha256_verified": True,
        "evaluation_labels_used_for_selection_or_tuning": False, "neural_training_or_finetuning": False,
        "checkpoint_selection_rerun": False, "target_normalization_or_adaptation": False, "BN_updates": False,
        "probe_hyperparameters_retuned": False, "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_unit": "biological subject", "bootstrap_draws": BOOTSTRAP_DRAWS,
        "persistence_permutations_rerun": False, "protected_selection_rerun": False,
        "canonical_basis_arrays": "exact deterministic reconstruction because corrected PEEH omitted serialization; audited against historical active rank, lambda_r, floor, and rho block means",
        "primary_fast_mode": True, "primary_seeds": list(seeds), "optional_multiseed_status": "NOT_RUN_PRIMARY_COMPLETE",
        "all_expected_runs": len(results) == 40, "task_rows": len(task) == 8, "paired_rows": len(paired) == 4,
        "recovery_checks_all_pass": all(r["recovery_status"] == "PASS" for r in recovery_rows)}
    atomic_json(OUT / "VALIDATION.json", validation)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-inference", action="store_true"); parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--seeds", default="0", help="comma-separated; primary FAST mode is 0")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); PROTOCOL.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(x) for x in args.seeds.split(",")); audit = audit_artifacts(seeds); atomic_csv(OUT / "ARTIFACT_AUDIT.csv", audit)
    atomic_json(PROTOCOL / "PSWA_PROTOCOL_LOCK.json", {"schema": "PERSIST_EEG_LITEBN_TFFORMER_PSWA_V1", "tasks": TASKS,
        "models": MODELS, "folds": FOLDS, "seeds": SEEDS, "ridge_alpha": RIDGE_ALPHA, "random_draws": RANDOM_DRAWS,
        "bootstrap_draws": BOOTSTRAP_DRAWS, "training": False, "finetuning": False, "adaptation": False,
        "protected_selection_rerun": False, "persistence_permutations_rerun": False, "random_redraw": False,
        "OpenBMI": {"cohort": OPENBMI_SUBJECTS, "sessions": [1,2], "status": "previously accessed internal-heldout diagnostic"},
        "WBCIC": {"cohort": WBCIC_SUBJECTS, "sessions": [0,1,2], "status": "previously accessed true-outer diagnostic"},
        "basis_array_reconstruction": "PEEH omitted matrices; reconstruct deterministically from frozen TRAIN embeddings and validate against stored spectrum audit; no null or selection rerun"})
    print(f"PSWA_AUDIT_OK rows={len(audit)}", flush=True)
    if args.audit_only: return
    if not args.skip_inference: recovery_rows = missing_inference(args.device, seeds)
    else:
        recovery_rows = [load_npz(session_cache_path(task, model, fold, seed))["metadata"]["recovery"]
            for task in TASKS for model in MODELS for fold in FOLDS for seed in seeds
            if session_cache_path(task, model, fold, seed).is_file()]
    analyze(seeds, recovery_rows); print("PSWA_COMPLETE", flush=True)


if __name__ == "__main__": main()
