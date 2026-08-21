from __future__ import annotations

"""Signed-V3.1-compatible train-only geometry and identity audits.

The protected assignment is constructed only from T_anchor and the 34 frozen
development-train subjects.  Outcome subjects enter this module only in the
final, non-selection identity/task audit.
"""

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import CACHE, OUTPUTS, balanced_accuracy, ce_loss, clean, load_config, macro_f1, softmax, stable_seed, stable_uint64, write_csv, write_json
from data import load_development_split, load_manifest, select_frame
from train import load_representation


EPSILON_NEUTRAL = 0.005


def aligned_representation(method_id: str, fold: int, seed: int, role: str = "T_anchor") -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    split = load_development_split(fold); manifest = load_manifest(split); cached = load_representation(method_id, fold, seed, role); lookup = manifest.set_index("manifest_position", drop=False); positions = cached["positions"].astype(np.int64)
    if not set(map(int, positions)).issubset(set(map(int, lookup.index))): raise RuntimeError("representation position outside development manifest")
    frame = lookup.loc[positions].reset_index(drop=True)
    if not np.array_equal(frame.manifest_position.to_numpy(dtype=np.int64), positions): raise RuntimeError("representation/manifest alignment failure")
    return frame, cached["features"].astype(np.float32), cached["logits"].astype(np.float32)


def _bootstrap(values: Sequence[float], seed: int, draws: int) -> dict[str, Any]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not len(arr): return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed); sample = rng.choice(arr, size=(int(draws), len(arr)), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))], "n": int(len(arr)), "draws": int(draws)}


def _blocks(rho: np.ndarray, rank: int) -> list[list[int]]:
    order = np.argsort(-rho); return [order[i:i + int(rank)].astype(int).tolist() for i in range(0, len(order), int(rank)) if len(order[i:i + int(rank)])]


def build_spectrum(meta: pd.DataFrame, features: np.ndarray, fold: int, seed: int) -> dict[str, Any]:
    cfg = load_config()["spectrum"]; x = np.asarray(features, dtype=np.float64); mu = x.mean(axis=0); centered = x - mu; cov = centered.T @ centered / max(len(x) - 1, 1); eva, evec = np.linalg.eigh((cov + cov.T) / 2); order = np.argsort(eva)[::-1]; eva, evec = eva[order], evec[:, order]; numerical = int(np.sum(eva > max(float(eva[0]) * 1e-3, 1e-8))); rank = min(int(cfg["whitening_rank"]), numerical); rank = max(rank, min(x.shape[1], 4)); active = np.maximum(eva[:rank], 1e-8); pca = evec[:, :rank]; whitener = pca * np.power(active, -.5)[None, :]; dewhitener = np.sqrt(active)[:, None] * pca.T; white = centered @ whitener
    frame = meta.reset_index(drop=True).copy(); frame["local_position"] = np.arange(len(frame)); subjects = sorted(frame.subject_id.astype(str).unique(), key=lambda s: (int(''.join(c for c in s if c.isdigit()) or 10**9), s)); centroids = {}
    for key, group in frame.groupby(["subject_id", "session_id", "label"], sort=True): centroids[(str(key[0]), int(key[1]), int(key[2]))] = white[group.local_position.to_numpy()].mean(axis=0)
    covars = []
    for label in (0, 1):
        a, b = [], []
        for subject in subjects:
            if (subject, 1, label) in centroids and (subject, 2, label) in centroids: a.append(centroids[(subject, 1, label)]); b.append(centroids[(subject, 2, label)])
        if len(a) >= 3:
            aa, bb = np.asarray(a), np.asarray(b); aa -= aa.mean(0); bb -= bb.mean(0); covars.append((aa.T @ bb + bb.T @ aa) / (2 * len(aa)))
    persistence = np.mean(covars, axis=0) if covars else np.zeros((rank, rank)); rho, directions = np.linalg.eigh((persistence + persistence.T) / 2); order = np.argsort(rho)[::-1]; rho, directions = rho[order], directions[:, order]; blocks = _blocks(rho, int(cfg["max_block_rank"]))
    rng = np.random.default_rng(stable_seed("v11-persistence-null", fold, seed)); support = []
    for bi, block in enumerate(blocks):
        null = []
        for _ in range(int(cfg.get("permutation_draws", 100))):
            perm = rng.permutation(subjects); vals = []
            for label in (0, 1):
                a, b = [], []
                for subject, paired in zip(subjects, perm):
                    if (subject, 1, label) in centroids and (paired, 2, label) in centroids: a.append(centroids[(subject, 1, label)]); b.append(centroids[(paired, 2, label)])
                if len(a) >= 3:
                    aa, bb = np.asarray(a), np.asarray(b); aa -= aa.mean(0); bb -= bb.mean(0); c = (aa.T @ bb + bb.T @ aa) / (2 * len(aa)); vals.append(float(np.mean(np.diag(directions[:, block].T @ c @ directions[:, block]))))
            if vals: null.append(float(np.mean(vals)))
        observed = float(np.mean(rho[block])); q95 = float(np.quantile(null, .95)) if null else float("inf"); support.append({"block": bi, "rho_G": observed, "null_p95": q95, "persistence_supported": bool(observed > q95), "dimensions": len(block), "eigenvalue_range": [float(rho[block[0]]), float(rho[block[-1]])]})
    return {"mean": mu.astype(np.float32), "whitener": whitener.astype(np.float32), "dewhitener": dewhitener.astype(np.float32), "pca_vectors": pca.astype(np.float32), "directions": directions.astype(np.float32), "rho": rho.astype(np.float32), "blocks": blocks, "audit": {"fold": fold, "seed": seed, "nominal_embedding_dimension": int(x.shape[1]), "numerical_rank": numerical, "whitening_rank": rank, "persistence_support": support, "fit_roles": ["T_anchor_model_fit_session_1", "T_anchor_model_fit_session_2"], "calibration_used": False, "outcome_used": False, "outer_test_used": False, "outer_membership_enumerated": False}}


def coordinates(features: np.ndarray, spectrum: Mapping[str, Any], dimensions: Sequence[int] | None = None) -> np.ndarray:
    value = (np.asarray(features, dtype=np.float64) - spectrum["mean"]) @ spectrum["whitener"] @ spectrum["directions"]
    return value[:, np.asarray(dimensions, dtype=np.int64)] if dimensions is not None else value


def erase(features: np.ndarray, spectrum: Mapping[str, Any], dimensions: Sequence[int]) -> np.ndarray:
    q = coordinates(features, spectrum); selected = np.asarray(dimensions, dtype=np.int64); delta = np.zeros_like(q); delta[:, selected] = -q[:, selected]; return (np.asarray(features, dtype=np.float64) + (delta @ spectrum["directions"].T) @ spectrum["dewhitener"]).astype(np.float32)


def save_spectrum(path: Path, spectrum: Mapping[str, Any], assignments: Mapping[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp = path.with_suffix(".part.npz"); np.savez_compressed(temp, mean=spectrum["mean"], whitener=spectrum["whitener"], dewhitener=spectrum["dewhitener"], pca_vectors=spectrum["pca_vectors"], directions=spectrum["directions"], rho=spectrum["rho"], blocks_json=np.asarray(json.dumps(spectrum["blocks"])), audit_json=np.asarray(json.dumps(clean(spectrum["audit"]))), assignments_json=np.asarray(json.dumps(clean(assignments or {})))); os.replace(temp, path)


def load_spectrum(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = np.load(path, allow_pickle=False); spectrum = {name: value[name] for name in ("mean", "whitener", "dewhitener", "pca_vectors", "directions", "rho")}; spectrum["blocks"] = json.loads(str(value["blocks_json"].item())); spectrum["audit"] = json.loads(str(value["audit_json"].item())); return spectrum, json.loads(str(value["assignments_json"].item()))


def _ridge_probe(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.int64); mean = x.mean(0); std = x.std(0); std[std < 1e-6] = 1; z = np.c_[(x - mean) / std, np.ones(len(x))]; penalty = np.eye(z.shape[1]); penalty[-1, -1] = 0; target = np.eye(int(y.max()) + 1)[y]; w = np.linalg.pinv(z.T @ z + alpha * penalty) @ z.T @ target; return w, mean, std


def _predict_probe(x: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    w, mean, std = pack; z = np.c_[(np.asarray(x) - mean) / std, np.ones(len(x))]; return softmax(z @ w)


def subject_probe(meta: pd.DataFrame, features: np.ndarray, subjects: Sequence[str]) -> tuple[dict[str, Any], dict[str, float], np.ndarray]:
    allowed = set(map(str, subjects)); train = meta.subject_id.astype(str).isin(allowed).to_numpy() & (meta.session_id.to_numpy() == 1); test = meta.subject_id.astype(str).isin(allowed).to_numpy() & (meta.session_id.to_numpy() == 2); ordered = sorted(allowed, key=lambda s: (int(''.join(c for c in s if c.isdigit()) or 10**9), s)); codes = {s: i for i, s in enumerate(ordered)}; y_train = meta.loc[train, "subject_id"].astype(str).map(codes).to_numpy(); y_test = meta.loc[test, "subject_id"].astype(str).map(codes).to_numpy(); pack = _ridge_probe(features[train], y_train); pred = _predict_probe(features[test], pack).argmax(1); per = {s: float(np.mean(pred[y_test == c] == c)) for s, c in codes.items()}; return {"balanced_accuracy": float(np.mean(list(per.values()))) if per else np.nan, "chance": 1 / max(len(ordered), 1), "n_subjects": len(ordered), "train_rows": int(train.sum()), "eval_rows": int(test.sum())}, per, np.flatnonzero(test)


def identity_from_cached_inner(fold: int, seed: int, eval_subjects: Sequence[str], task_method: str, inv_method: str) -> float:
    task_meta, task, _ = aligned_representation(task_method, fold, seed, "T_anchor"); inv_meta, inv, _ = aligned_representation(inv_method, fold, seed, "I_invariant"); subjects = list(map(str, eval_subjects)); a = subject_probe(task_meta, task, subjects)[0]["balanced_accuracy"]; b = subject_probe(inv_meta, inv, subjects)[0]["balanced_accuracy"]; return float(b - a)


def spectrum_path(family: str, fold: int, seed: int) -> Path:
    return CACHE / "protected_spectra" / family / f"fold-{fold}" / f"seed-{seed}.npz"


def _signed_assignment(meta: pd.DataFrame, features: np.ndarray, spectrum: Mapping[str, Any], fold: int, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = load_config(); train_subjects = sorted(meta.subject_id.astype(str).unique()); rows = []; rng = np.random.default_rng(stable_seed("v11-signed", fold, seed));
    for bi, block in enumerate(spectrum["blocks"]):
        deltas, random_deltas = [], []; candidates = [i for i in range(len(spectrum["rho"])) if i not in block]
        for inner in range(int(cfg["spectrum"].get("inner_splits", 5))):
            shuffled = train_subjects.copy(); rng.shuffle(shuffled); cut = max(1, len(shuffled) // 2); fit_s, eval_s = shuffled[:cut], shuffled[cut:]; fi = meta.subject_id.astype(str).isin(fit_s).to_numpy(); ei = meta.subject_id.astype(str).isin(eval_s).to_numpy(); yfit = meta.loc[fi, "label"].to_numpy(); yeval = meta.loc[ei, "label"].to_numpy(); base = _predict_probe(features[ei], _ridge_probe(features[fi], yfit)); erased = _predict_probe(erase(features[ei], spectrum, block), _ridge_probe(erase(features[fi], spectrum, block), yfit)); d = ce_loss(yeval, erased) - ce_loss(yeval, base); deltas.append(d); rand = []
            for _ in range(int(cfg["spectrum"].get("random_erasures", 100))):
                choice = np.sort(rng.choice(candidates if len(candidates) >= len(block) else np.arange(len(spectrum["rho"])), size=len(block), replace=False)); rand.append(ce_loss(yeval, _predict_probe(erase(features[ei], spectrum, choice), _ridge_probe(erase(features[fi], spectrum, choice), yfit))) - ce_loss(yeval, base))
            random_deltas.append(float(np.mean(rand)))
        rows.append({"fold": fold, "seed": seed, "block": bi, "dimensions": len(block), "coordinate_ids": json.dumps(block), "persistence_supported": bool(spectrum["audit"]["persistence_support"][bi]["persistence_supported"]), "u_abs_mean": float(np.mean(deltas)), "u_spec_mean": float(np.mean(np.asarray(deltas) - np.asarray(random_deltas))), "u_abs_CI95_low": _bootstrap(deltas, stable_seed("u", fold, seed, bi), int(cfg["bootstrap_draws"]))["ci95"][0], "u_spec_CI95_low": _bootstrap(np.asarray(deltas) - np.asarray(random_deltas), stable_seed("us", fold, seed, bi), int(cfg["bootstrap_draws"]))["ci95"][0], "outer_test_used": False})
    frame = pd.DataFrame(rows); protected_blocks = frame[(frame.persistence_supported) & (frame.u_abs_CI95_low > 0) & (frame.u_spec_CI95_low > 0)].block.astype(int).tolist(); protected = sorted({d for b in protected_blocks for d in spectrum["blocks"][b]}); assignment = {"protected_blocks": protected_blocks, "protected_dimensions": protected, "rank": len(protected), "definition": "Signed-V3.1-compatible persistence support plus positive absolute and random-subtracted utility lower bounds", "fit_role": "T_anchor_model_fit_only", "outcome_used": False, "outer_test_used": False, "outer_membership_enumerated": False}; return assignment, frame


def ensure_family_spectrum(family: str, task_method: str, fold: int, seed: int, force: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    path = spectrum_path(family, fold, seed)
    if path.exists() and not force: return load_spectrum(path)
    split = load_development_split(fold); meta, feat, _ = aligned_representation(task_method, fold, seed, "T_anchor"); mask = meta.subject_id.astype(str).isin(split.model_fit_subjects).to_numpy(); fit_meta = meta.loc[mask].reset_index(drop=True); fit_feat = feat[mask]; spectrum = build_spectrum(fit_meta, fit_feat, fold, seed); assignment, utility = _signed_assignment(fit_meta, fit_feat, spectrum, fold, seed); save_spectrum(path, spectrum, assignment); directory = OUTPUTS / "audit" / family / f"fold-{fold}" / f"seed-{seed}"; write_csv(directory / "SIGNED_UTILITY.csv", utility); write_json(directory / "SIGNED_ASSIGNMENTS.json", assignment); write_json(directory / "SPECTRUM_AUDIT.json", spectrum["audit"]); return spectrum, assignment


def _native(meta: pd.DataFrame, logits: np.ndarray, subjects: Sequence[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    mask = meta.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy() & (meta.session_id.to_numpy() == 2); frame = meta.loc[mask].reset_index(drop=True); truth = frame.label.to_numpy(); pred = logits[mask].argmax(1); per = {}
    for subject, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        idx = group.index.to_numpy(); per[str(subject)] = {"balanced_accuracy": balanced_accuracy(truth[idx], pred[idx]), "accuracy": float(np.mean(truth[idx] == pred[idx])), "macro_f1": macro_f1(truth[idx], pred[idx])}
    return {"balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in per.values()])), "accuracy": float(np.mean(pred == truth)), "macro_f1": float(np.mean([v["macro_f1"] for v in per.values()])), "n_subjects": len(per)}, per


def audit_pair(family: str, task_method: str, invariant_method: str, fold: int, seed: int, force_spectrum: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split = load_development_split(fold); meta, anchor, anchor_logits = aligned_representation(task_method, fold, seed, "T_anchor"); _, replica, replica_logits = aligned_representation(task_method, fold, seed, "T_replica"); _, invariant, invariant_logits = aligned_representation(invariant_method, fold, seed, "I_invariant"); spectrum, assignment = ensure_family_spectrum(family, task_method, fold, seed, force_spectrum); anchor_nat, anchor_subject = _native(meta, anchor_logits, split.outcome_subjects); replica_nat, replica_subject = _native(meta, replica_logits, split.outcome_subjects); inv_nat, inv_subject = _native(meta, invariant_logits, split.outcome_subjects); anchor_id, anchor_id_subject, _ = subject_probe(meta, anchor, split.outcome_subjects); replica_id, replica_id_subject, _ = subject_probe(meta, replica, split.outcome_subjects); inv_id, inv_id_subject, _ = subject_probe(meta, invariant, split.outcome_subjects); rows = []
    for subject in split.outcome_subjects:
        s = str(subject); rows.append({"family": family, "fold": fold, "seed": seed, "subject_id": s, "T_anchor_BA": anchor_subject[s]["balanced_accuracy"], "T_replica_BA": replica_subject[s]["balanced_accuracy"], "invariant_BA": inv_subject[s]["balanced_accuracy"], "delta_BA_INV": inv_subject[s]["balanced_accuracy"] - anchor_subject[s]["balanced_accuracy"], "T_anchor_ID": anchor_id_subject[s], "T_replica_ID": replica_id_subject[s], "invariant_ID": inv_id_subject[s], "delta_ID": inv_id_subject[s] - anchor_id_subject[s], "replica_delta_ID": replica_id_subject[s] - anchor_id_subject[s], "outer_test_used": False, "outer_membership_enumerated": False})
    run = {"family": family, "fold": fold, "seed": seed, "task_method": task_method, "invariant_method": invariant_method, "T_anchor_BA": anchor_nat["balanced_accuracy"], "T_replica_BA": replica_nat["balanced_accuracy"], "invariant_BA": inv_nat["balanced_accuracy"], "delta_BA_INV": inv_nat["balanced_accuracy"] - anchor_nat["balanced_accuracy"], "T_anchor_ID": anchor_id["balanced_accuracy"], "T_replica_ID": replica_id["balanced_accuracy"], "invariant_ID": inv_id["balanced_accuracy"], "delta_ID": inv_id["balanced_accuracy"] - anchor_id["balanced_accuracy"], "replica_delta_ID": replica_id["balanced_accuracy"] - anchor_id["balanced_accuracy"], "selected_lambda": None, "protected_rank": len(assignment.get("protected_dimensions", [])), "protected_assignment_exists": bool(assignment.get("protected_dimensions")), "outer_test_used": False, "outer_membership_enumerated": False}
    return run, rows


def audit_all(force_spectrum: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(); selections = pd.read_csv(OUTPUTS / "HYPERPARAM_SELECTION.csv"); runs, subjects = [], []
    for fold in map(int, config["development_folds"]):
        lam = float(selections[selections.fold == fold].iloc[0].selected_lambda); task = "A0_TASK_ONLY_EEGNET"; inv = f"A1_SUBJECT_GRL_EEGNET_L{int(round(lam * 1000)):04d}"
        for seed in map(int, config["seeds"]):
            for family, t, i in [("A_SUBJECT_GRL_EEGNET", task, inv), ("B_EEG_DG", "B0_EEG_DG_TASK_ONLY", "B1_EEG_DG_FULL"), ("C_SCLDGN", "C0_SCLDGN_TASK_ONLY", "C1_SCLDGN_FULL")]:
                run, sub = audit_pair(family, t, i, fold, seed, force_spectrum); run["selected_lambda"] = lam if family.startswith("A_") else None; runs.append(run); subjects.extend(sub)
    run_frame, sub_frame = pd.DataFrame(runs), pd.DataFrame(subjects); write_csv(OUTPUTS / "IDENTITY_AUDIT.csv", run_frame); write_csv(OUTPUTS / "TASK_HARM.csv", sub_frame); write_csv(OUTPUTS / "INVARIANCE_AUDIT.csv", run_frame); write_csv(OUTPUTS / "SUBJECT_LEVEL_AUDIT.csv", sub_frame); return run_frame, sub_frame, pd.DataFrame()
