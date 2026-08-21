from __future__ import annotations

"""Replica-normalized functional Protected-task-evidence measurement."""

import json
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from common import OUTPUTS, load_config, stable_seed, write_csv, write_json
from data import load_development_split
from spectrum import aligned_representation, ensure_family_spectrum
from train import checkpoint_path


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64); mean = x.mean(0); std = x.std(0); std[std < 1e-6] = 1.; z = np.c_[(x - mean) / std, np.ones(len(x))]; penalty = np.eye(z.shape[1]); penalty[-1, -1] = 0.; w = np.linalg.pinv(z.T @ z + float(alpha) * penalty) @ z.T @ y; return {"w": w, "mean": mean, "std": std}


def _ridge_predict(x: np.ndarray, pack: Mapping[str, np.ndarray]) -> np.ndarray:
    z = np.c_[(np.asarray(x, dtype=np.float64) - pack["mean"]) / pack["std"], np.ones(len(x))]; return z @ pack["w"]


def _subject_cv_alpha(meta: pd.DataFrame, x: np.ndarray, y: np.ndarray, train_subjects: Sequence[str], grid: Sequence[float]) -> float:
    subjects = sorted(map(str, train_subjects)); folds = np.array_split(np.asarray(subjects), 3); scores = []
    for alpha in map(float, grid):
        losses = []
        for held in folds:
            ev = set(map(str, held)); fi = meta.subject_id.astype(str).isin(set(subjects) - ev).to_numpy() & (meta.session_id.to_numpy() == 1); ei = meta.subject_id.astype(str).isin(ev).to_numpy() & (meta.session_id.to_numpy() == 2)
            if not fi.any() or not ei.any(): continue
            pred = _ridge_predict(x[ei], _ridge_fit(x[fi], y[fi], alpha)); losses.append(float(np.mean((pred - y[ei]) ** 2)))
        scores.append((float(np.mean(losses)) if losses else np.inf, alpha))
    return min(scores, key=lambda t: (t[0], t[1]))[1]


def _orthogonal_subspace(spectrum: Mapping[str, Any], dims: Sequence[int]) -> np.ndarray:
    raw = np.asarray(spectrum["whitener"], dtype=np.float64) @ np.asarray(spectrum["directions"], dtype=np.float64)[:, list(map(int, dims))]
    q, _ = np.linalg.qr(raw); return q[:, :len(dims)]


def _target_q(features: np.ndarray, mu: np.ndarray, projector: np.ndarray, margin: np.ndarray) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float64) - np.asarray(mu, dtype=np.float64); return (centered @ projector @ margin).astype(np.float64)


def _teacher_margin(fold: int, seed: int, method_id: str, role: str) -> np.ndarray:
    import torch
    payload = torch.load(checkpoint_path("full", fold, seed, method_id, role), map_location="cpu", weights_only=False); state = payload["model"]
    if "head.weight" not in state: raise RuntimeError("teacher classifier head is missing")
    return (state["head.weight"][1].numpy() - state["head.weight"][0].numpy()).astype(np.float64)


def _make_controls(spectrum: Mapping[str, Any], protected: Sequence[int], teacher_features: np.ndarray, meta: pd.DataFrame, margin: np.ndarray, fold: int, seed: int, count: int) -> list[list[int]]:
    rank = len(protected); all_dims = list(range(len(spectrum["rho"]))); forbidden = set(map(int, protected)); candidates = []
    if not rank: return []
    if rank <= 3 and len(all_dims) <= 20:
        combos = combinations([d for d in all_dims if d not in forbidden], rank)
    else:
        rng = np.random.default_rng(stable_seed("matched-control", fold, seed)); pool = [d for d in all_dims if d not in forbidden]; sampled = set(); combos_list = []
        for _ in range(max(1000, count * 20)):
            if len(pool) < rank: break
            choice = tuple(sorted(rng.choice(pool, size=rank, replace=False).tolist()));
            if choice not in sampled: sampled.add(choice); combos_list.append(choice)
        combos = iter(combos_list)
    prot_u = _orthogonal_subspace(spectrum, protected); centered = teacher_features - teacher_features[meta.subject_id.astype(str).isin([]).to_numpy()].mean(0) if False else teacher_features - teacher_features.mean(0)
    q_p = _target_q(teacher_features, teacher_features.mean(0), prot_u @ prot_u.T, margin); target = np.array([np.var(q_p), np.mean(np.abs(q_p)), np.linalg.norm(prot_u)])
    records = []
    for dims in combos:
        u = _orthogonal_subspace(spectrum, dims); q = _target_q(teacher_features, teacher_features.mean(0), u @ u.T, margin); records.append((float(np.linalg.norm(np.array([np.var(q), np.mean(np.abs(q)), np.linalg.norm(u)]) - target)), list(map(int, dims))))
    records.sort(key=lambda t: (t[0], t[1])); return [dims for _, dims in records[:count]]


def _fr_by_subject(meta: pd.DataFrame, prediction: np.ndarray, target: np.ndarray, outcome_subjects: Sequence[str]) -> dict[str, float]:
    mask = meta.subject_id.astype(str).isin(set(map(str, outcome_subjects))).to_numpy() & (meta.session_id.to_numpy() == 2); frame = meta.loc[mask].reset_index(drop=True); pred = np.asarray(prediction)[mask]; truth = np.asarray(target)[mask]; result = {}
    for subject, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        idx = group.index.to_numpy(); result[str(subject)] = float(1.0 - np.mean((pred[idx] - truth[idx]) ** 2))
    return result


def analyze_one(family: str, task_method: str, invariant_method: str, fold: int, seed: int, force: bool = False) -> dict[str, Any]:
    config = load_config(); split = load_development_split(fold); meta, anchor, _ = aligned_representation(task_method, fold, seed, "T_anchor"); _, replica, _ = aligned_representation(task_method, fold, seed, "T_replica"); _, invariant, _ = aligned_representation(invariant_method, fold, seed, "I_invariant"); spectrum, assignment = ensure_family_spectrum(family, task_method, fold, seed, force); protected = list(map(int, assignment.get("protected_dimensions", []))); train_mask = meta.subject_id.astype(str).isin(split.model_fit_subjects).to_numpy(); margin = _teacher_margin(fold, seed, task_method, "T_anchor"); mu = anchor[train_mask].mean(0); projector = _orthogonal_subspace(spectrum, protected) if protected else np.zeros((anchor.shape[1], 0)); q_p = _target_q(anchor, mu, projector @ projector.T if protected else np.zeros((anchor.shape[1], anchor.shape[1])), margin) if protected else np.zeros(len(anchor)); q_p_train = q_p[train_mask]; q_std = float(np.std(q_p_train)); valid = bool(protected and np.isfinite(q_std) and q_std > float(config["functional"]["q_variance_floor"]))
    if valid: q_p = (q_p - float(np.mean(q_p_train))) / q_std
    controls = _make_controls(spectrum, protected, anchor[train_mask], meta.loc[train_mask].reset_index(drop=True), margin, fold, seed, int(config["functional"]["matched_controls"])) if valid else []
    rows_p, rows_n = [], []
    if valid:
        for role, features in (("T_replica", replica), ("I_invariant", invariant)):
            alpha = _subject_cv_alpha(meta.loc[train_mask].reset_index(drop=True), features[train_mask], q_p[train_mask], split.model_fit_subjects, config["functional"]["ridge_grid"]); pack = _ridge_fit(features[train_mask], q_p[train_mask], alpha); fr = _fr_by_subject(meta, _ridge_predict(features, pack), q_p, split.outcome_subjects); mean_fr = float(np.mean(list(fr.values())))
            for subject, value in fr.items(): rows_p.append({"family": family, "fold": fold, "seed": seed, "subject_id": subject, "role": role, "FR": value, "ridge": alpha, "q_variance_train": q_std, "competence": role == "T_replica" and mean_fr > float(config["functional"]["competence_floor_FR_replica"]), "outer_test_used": False})
        for control_id, dims in enumerate(controls):
            u = _orthogonal_subspace(spectrum, dims); q = _target_q(anchor, mu, u @ u.T, margin); q_train = q[train_mask]; s = float(np.std(q_train));
            if s <= 1e-8: continue
            q = (q - float(np.mean(q_train))) / s
            for role, features in (("T_replica", replica), ("I_invariant", invariant)):
                alpha = _subject_cv_alpha(meta.loc[train_mask].reset_index(drop=True), features[train_mask], q[train_mask], split.model_fit_subjects, config["functional"]["ridge_grid"]); fr = _fr_by_subject(meta, _ridge_predict(features, _ridge_fit(features[train_mask], q[train_mask], alpha)), q, split.outcome_subjects)
                for subject, value in fr.items(): rows_n.append({"family": family, "fold": fold, "seed": seed, "subject_id": subject, "control_id": control_id, "rank": len(dims), "dimensions": json.dumps(dims), "role": role, "FR": value, "ridge": alpha, "outer_test_used": False})
    p_frame, n_frame = pd.DataFrame(rows_p), pd.DataFrame(rows_n); lp, ln, spl = [], [], []
    if valid and len(p_frame) and len(n_frame):
        for subject in split.outcome_subjects:
            s = str(subject); p = p_frame[p_frame.subject_id == s].set_index("role"); n = n_frame[n_frame.subject_id == s].pivot(index="control_id", columns="role", values="FR"); l_p = float(p.loc["T_replica", "FR"] - p.loc["I_invariant", "FR"]); l_n = float(np.median(n["T_replica"] - n["I_invariant"])); lp.append({"family": family, "fold": fold, "seed": seed, "subject_id": s, "L_P": l_p, "L_N": l_n, "SPL": l_p - l_n, "q_variance_train": q_std, "outer_test_used": False})
    result = {"family": family, "fold": fold, "seed": seed, "protected_assignment_exists": bool(protected), "protected_rank": len(protected), "q_variance_train": q_std, "measurement_valid": valid and len(controls) >= int(config["functional"]["matched_controls"]), "replica_FR_mean": float(p_frame[p_frame.role == "T_replica"].FR.mean()) if len(p_frame) else None, "functional_rows": p_frame, "control_rows": n_frame, "spl_rows": pd.DataFrame(lp), "outer_test_used": False, "outer_membership_enumerated": False}; return result


def analyze_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(); selections = pd.read_csv(OUTPUTS / "HYPERPARAM_SELECTION.csv"); p_rows, n_rows, spl_rows, assignment_rows = [], [], [], []
    for fold in map(int, config["development_folds"]):
        selected = selections[selections.fold == fold]; lam = float(selected[selected.selected.astype(bool)].iloc[0].candidate_lambda); pairs = [("A_SUBJECT_GRL_EEGNET", "A0_TASK_ONLY_EEGNET", f"A1_SUBJECT_GRL_EEGNET_L{int(round(lam*1000)):04d}"), ("B_EEG_DG", "B0_EEG_DG_TASK_ONLY", "B1_EEG_DG_FULL"), ("C_SCLDGN", "C0_SCLDGN_TASK_ONLY", "C1_SCLDGN_FULL")]
        for seed in map(int, config["seeds"]):
            for family, task, inv in pairs:
                result = analyze_one(family, task, inv, fold, seed); p_rows.append(result["functional_rows"]); n_rows.append(result["control_rows"]); spl_rows.append(result["spl_rows"]); assignment_rows.append({"family": family, "fold": fold, "seed": seed, "protected_assignment_exists": result["protected_assignment_exists"], "protected_rank": result["protected_rank"], "q_variance_train": result["q_variance_train"], "measurement_valid": result["measurement_valid"], "outer_test_used": False})
    p = pd.concat([x for x in p_rows if len(x)], ignore_index=True) if any(len(x) for x in p_rows) else pd.DataFrame(columns=["family","fold","seed","subject_id","role","FR"]); n = pd.concat([x for x in n_rows if len(x)], ignore_index=True) if any(len(x) for x in n_rows) else pd.DataFrame(columns=["family","fold","seed","subject_id","control_id","role","FR"]); spl = pd.concat([x for x in spl_rows if len(x)], ignore_index=True) if any(len(x) for x in spl_rows) else pd.DataFrame(columns=["family","fold","seed","subject_id","L_P","L_N","SPL"]); write_csv(OUTPUTS / "FUNCTIONAL_RETENTION.csv", p); write_csv(OUTPUTS / "MATCHED_NONPROTECTED_RETENTION.csv", n); write_csv(OUTPUTS / "SELECTIVE_PROTECTED_LOSS.csv", spl); write_csv(OUTPUTS / "PROTECTED_ASSIGNMENT.csv", pd.DataFrame(assignment_rows)); return p, n, spl
