from __future__ import annotations

"""Frozen, rank-matched residual restoration diagnostics."""

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from common import OUTPUTS, balanced_accuracy, ce_loss, load_config, macro_f1, softmax, stable_seed, write_csv, write_json
from data import load_development_split
from spectrum import aligned_representation, coordinates, ensure_family_spectrum, subject_probe


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64); mean = x.mean(0); std = x.std(0); std[std < 1e-6] = 1.; z = np.c_[(x - mean) / std, np.ones(len(x))]; p = np.eye(z.shape[1]); p[-1, -1] = 0.; w = np.linalg.pinv(z.T @ z + float(alpha) * p) @ z.T @ y; return w, mean, std


def _predict(x: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    w, mean, std = pack; return np.c_[(np.asarray(x) - mean) / std, np.ones(len(x))] @ w


def _head(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.eye(2)[np.asarray(y, dtype=np.int64)]; return _ridge(x, target, alpha)


def _score(meta: pd.DataFrame, logits: np.ndarray, outcome: Sequence[str]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    mask = meta.subject_id.astype(str).isin(set(map(str, outcome))).to_numpy() & (meta.session_id.to_numpy() == 2); frame = meta.loc[mask].reset_index(drop=True); truth = frame.label.to_numpy(); pred = logits[mask].argmax(1); per = {}
    for subject, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        idx = group.index.to_numpy(); per[str(subject)] = {"balanced_accuracy": balanced_accuracy(truth[idx], pred[idx]), "accuracy": float(np.mean(pred[idx] == truth[idx])), "macro_f1": macro_f1(truth[idx], pred[idx])}
    return {"balanced_accuracy": float(np.mean([v["balanced_accuracy"] for v in per.values()])), "accuracy": float(np.mean(pred == truth)), "macro_f1": float(np.mean([v["macro_f1"] for v in per.values()])), "cross_entropy": ce_loss(truth, softmax(logits[mask]))}, per


def _residual_fit(z: np.ndarray, delta: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _ridge(z, delta, alpha)


def _inner_select(meta: pd.DataFrame, invariant: np.ndarray, teacher: np.ndarray, z: np.ndarray, labels: np.ndarray, train_subjects: Sequence[str], grid: Mapping[str, Sequence[float]]) -> tuple[float, float]:
    subjects = sorted(map(str, train_subjects)); folds = np.array_split(np.asarray(subjects), 3); candidates = []
    for ridge in map(float, grid["ridge_grid"]):
        for alpha in map(float, grid["alpha_grid"]):
            values = []
            for held in folds:
                ev = set(map(str, held)); fit = set(subjects) - ev; fi = meta.subject_id.astype(str).isin(fit).to_numpy() & (meta.session_id.to_numpy() == 1); ei = meta.subject_id.astype(str).isin(ev).to_numpy() & (meta.session_id.to_numpy() == 2)
                if not fi.any() or not ei.any(): continue
                b = _residual_fit(z[fi], teacher[fi] - invariant[fi], ridge); restored = invariant + float(alpha) * _predict(z, b); h = _head(restored[fi], labels[fi], ridge); pred = _predict(restored[ei], h).argmax(1); values.append(float(balanced_accuracy(labels[ei], pred)))
            candidates.append((float(np.mean(values)) if values else -np.inf, -ridge, -alpha, ridge, alpha))
    best = max(candidates, key=lambda x: x[:3]); return float(best[3]), float(best[4])


def _score_method(method: str, family: str, fold: int, seed: int, meta: pd.DataFrame, representation: np.ndarray, logits: np.ndarray, outcome: Sequence[str], selection: Mapping[str, Any], draw: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    aggregate, per = _score(meta, logits, outcome); probe, probe_subject, _ = subject_probe(meta, representation, outcome); rows = []
    for subject, values in per.items(): rows.append({"family": family, "fold": fold, "seed": seed, "subject_id": subject, "rescue_method": method, "random_draw": draw, **values, "subject_probe_accuracy": probe_subject[subject], "outer_test_used": False})
    return {"family": family, "fold": fold, "seed": seed, "rescue_method": method, "random_draw": draw, **aggregate, "subject_probe_accuracy": probe["balanced_accuracy"], "selected_ridge": selection.get("ridge"), "selected_alpha": selection.get("alpha"), "residual_rank": selection.get("rank", 0), "residual_parameters": selection.get("parameters", 0), "selection_role": "nested_train_subject_cv", "outcome_used_for_selection": False, "outer_test_used": False}, rows


def run_one(family: str, task_method: str, invariant_method: str, fold: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config(); split = load_development_split(fold); meta, teacher, _ = aligned_representation(task_method, fold, seed, "T_anchor"); _, invariant, _ = aligned_representation(invariant_method, fold, seed, "I_invariant"); spectrum, assignment = ensure_family_spectrum(family, task_method, fold, seed); protected = list(map(int, assignment.get("protected_dimensions", []))); rank = len(protected)
    if not rank: return [], []
    train_mask = meta.subject_id.astype(str).isin(split.model_fit_subjects).to_numpy(); labels = meta.label.to_numpy(dtype=np.int64); grid = config["rescue"]; rows, subjects = [], []
    # Invariant-only and task-only reference heads are both selected using the
    # same nested training-subject budget.
    ridge_grid = grid["ridge_grid"]; alpha_i = _inner_select(meta, invariant, teacher, coordinates(teacher, spectrum, protected), labels, split.model_fit_subjects, {"ridge_grid": ridge_grid, "alpha_grid": [0.0]});
    h_inv = _head(invariant[train_mask], labels[train_mask], alpha_i[0]); inv_logits = _predict(invariant, h_inv); row, sub = _score_method("R0_INVARIANT_ONLY", family, fold, seed, meta, invariant, inv_logits, split.outcome_subjects, {"ridge": alpha_i[0], "alpha": 0.0, "rank": 0}); rows.append(row); subjects.extend(sub)
    protected_z = coordinates(teacher, spectrum, protected); generic = [int(i) for i in np.argsort(-np.asarray(spectrum["rho"])) if int(i) not in set(protected)][:rank]; generic_z = coordinates(teacher, spectrum, generic); pca_z = (teacher - spectrum["mean"]) @ spectrum["pca_vectors"][:, :rank]
    rng = np.random.default_rng(stable_seed("v11-random-rescue", family, fold, seed)); random_z = []
    for _ in range(int(grid["random_draws"])):
        basis, _ = np.linalg.qr(rng.normal(size=(teacher.shape[1], rank))); random_z.append((teacher - teacher[train_mask].mean(0)) @ basis[:, :rank])
    methods = [("R4_PERSIST_PROTECTED_RESIDUAL", protected_z), ("R3_GENERIC_PERSISTENT_RESIDUAL", generic_z), ("R2_PCA_RESIDUAL", pca_z)]
    for name, z in methods:
        ridge, alpha = _inner_select(meta, invariant, teacher, z, labels, split.model_fit_subjects, grid); b = _residual_fit(z[train_mask], (teacher - invariant)[train_mask], ridge); restored = invariant + alpha * _predict(z, b); head = _head(restored[train_mask], labels[train_mask], ridge); row, sub = _score_method(name, family, fold, seed, meta, restored, _predict(restored, head), split.outcome_subjects, {"ridge": ridge, "alpha": alpha, "rank": rank, "parameters": int(restored.shape[1] * rank)}); rows.append(row); subjects.extend(sub)
    for draw, z in enumerate(random_z):
        ridge, alpha = _inner_select(meta, invariant, teacher, z, labels, split.model_fit_subjects, grid); b = _residual_fit(z[train_mask], (teacher - invariant)[train_mask], ridge); restored = invariant + alpha * _predict(z, b); head = _head(restored[train_mask], labels[train_mask], ridge); row, sub = _score_method("R1_RANDOM_RESIDUAL", family, fold, seed, meta, restored, _predict(restored, head), split.outcome_subjects, {"ridge": ridge, "alpha": alpha, "rank": rank, "parameters": int(restored.shape[1] * rank)}, draw); rows.append(row); subjects.extend(sub)
    ridge, alpha = _inner_select(meta, invariant, teacher, teacher - teacher[train_mask].mean(0), labels, split.model_fit_subjects, grid); full_head = _head(teacher[train_mask], labels[train_mask], ridge); row, sub = _score_method("R5_FULL_TEACHER_DIAGNOSTIC_UPPER_BOUND", family, fold, seed, meta, teacher, _predict(teacher, full_head), split.outcome_subjects, {"ridge": ridge, "alpha": 1.0, "rank": teacher.shape[1], "parameters": teacher.shape[1] * teacher.shape[1]}); rows.append(row); subjects.extend(sub)
    return rows, subjects


def run_eligible_rescues() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligibility = json.loads((OUTPUTS / "ELIGIBILITY.json").read_text(encoding="utf-8")); config = load_config(); rows, subjects = [], []
    selections = pd.read_csv(OUTPUTS / "HYPERPARAM_SELECTION.csv")
    for fold in map(int, config["development_folds"]):
        selected = selections[selections.fold == fold]; lam = float(selected[selected.selected.astype(bool)].iloc[0].candidate_lambda); pairs = [("A_SUBJECT_GRL_EEGNET", "A0_TASK_ONLY_EEGNET", f"A1_SUBJECT_GRL_EEGNET_L{int(round(lam*1000)):04d}"), ("B_EEG_DG", "B0_EEG_DG_TASK_ONLY", "B1_EEG_DG_FULL"), ("C_SCLDGN", "C0_SCLDGN_TASK_ONLY", "C1_SCLDGN_FULL")]
        for seed in map(int, config["seeds"]):
            for family, task, inv in pairs:
                entry = eligibility["families"].get(family, {});
                if not entry.get("rescue_allowed", False): continue
                r, s = run_one(family, task, inv, fold, seed); rows.extend(r); subjects.extend(s)
    result = pd.DataFrame(rows); subject = pd.DataFrame(subjects)
    if not len(result): result = pd.DataFrame(columns=["family","fold","seed","rescue_method","balanced_accuracy","outer_test_used"])
    if not len(subject): subject = pd.DataFrame(columns=["family","fold","seed","subject_id","rescue_method","balanced_accuracy","outer_test_used"])
    write_csv(OUTPUTS / "RESCUE_RESULTS.csv", result); write_csv(OUTPUTS / "RESCUE_SUBJECT_RESULTS.csv", subject); return result, subject, eligibility
