from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from common import (
    OUTPUTS,
    balanced_accuracy,
    ce_loss,
    load_config,
    macro_f1,
    softmax,
    stable_seed,
    write_csv,
    write_json,
)
from data import load_development_split
from models import primary_pairs
from spectrum import (
    aligned_representation,
    coordinates,
    load_spectrum,
    spectrum_path,
    subject_probe,
)


@dataclass
class RidgeHead:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ResidualMap:
    weights: np.ndarray
    coordinate_scale: np.ndarray


def _fit_head(features: np.ndarray, labels: np.ndarray, alpha: float) -> RidgeHead:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    design = np.concatenate([(x - mean) / std, np.ones((len(x), 1))], axis=1)
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    counts = np.bincount(y, minlength=2).astype(np.float64)
    sample_weight = 1.0 / np.maximum(counts[y], 1.0)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    weighted = design * np.sqrt(sample_weight)[:, None]
    targets = np.eye(2)[y] * np.sqrt(sample_weight)[:, None]
    system = weighted.T @ weighted + float(alpha) * penalty
    try:
        weights = np.linalg.solve(system, weighted.T @ targets)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ weighted.T @ targets
    return RidgeHead(weights=weights, mean=mean, std=std)


def _head_logits(features: np.ndarray, head: RidgeHead) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    design = np.concatenate([(x - head.mean) / head.std, np.ones((len(x), 1))], axis=1)
    return (design @ head.weights).astype(np.float32)


def _fit_residual(coordinates_fit: np.ndarray, delta_fit: np.ndarray, alpha: float) -> ResidualMap:
    z = np.asarray(coordinates_fit, dtype=np.float64)
    delta = np.asarray(delta_fit, dtype=np.float64)
    scale = z.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = z / scale
    system = normalized.T @ normalized + float(alpha) * np.eye(normalized.shape[1])
    try:
        weights = np.linalg.solve(system, normalized.T @ delta)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ normalized.T @ delta
    return ResidualMap(weights=weights, coordinate_scale=scale)


def _apply_residual(
    invariant_features: np.ndarray,
    coordinates_all: np.ndarray,
    residual: ResidualMap,
    scale: float,
) -> np.ndarray:
    prediction = (np.asarray(coordinates_all, dtype=np.float64) / residual.coordinate_scale) @ residual.weights
    return (np.asarray(invariant_features, dtype=np.float64) + float(scale) * prediction).astype(np.float32)


def _calibration_score(logits: np.ndarray, labels: np.ndarray, meta: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]:
    truth = labels[mask]
    probability = softmax(logits[mask])
    prediction = probability.argmax(axis=1)
    frame = meta.loc[mask].reset_index(drop=True)
    values = []
    for _, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        values.append(balanced_accuracy(truth[positions], prediction[positions]))
    return float(np.mean(values)), ce_loss(truth, probability)


def _select_plain_head(
    features: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    ridge_values: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = []
    for alpha in ridge_values:
        head = _fit_head(features[fit_mask], labels[fit_mask], float(alpha))
        logits = _head_logits(features, head)
        ba, ce = _calibration_score(logits, labels, meta, calibration_mask)
        candidates.append((ba, -ce, -float(alpha), logits, {"ridge": float(alpha), "calibration_BA": ba, "calibration_CE": ce}))
    best = max(candidates, key=lambda item: item[:3])
    return best[3], best[4]


def _select_rescue(
    invariant_features: np.ndarray,
    teacher_features: np.ndarray,
    coordinate_values: np.ndarray,
    labels: np.ndarray,
    meta: pd.DataFrame,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
    ridge_values: Sequence[float],
    residual_scales: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidates = []
    delta_fit = teacher_features[fit_mask] - invariant_features[fit_mask]
    for alpha in ridge_values:
        residual = _fit_residual(coordinate_values[fit_mask], delta_fit, float(alpha))
        for residual_scale in residual_scales:
            rescued = _apply_residual(invariant_features, coordinate_values, residual, float(residual_scale))
            head = _fit_head(rescued[fit_mask], labels[fit_mask], float(alpha))
            logits = _head_logits(rescued, head)
            ba, ce = _calibration_score(logits, labels, meta, calibration_mask)
            selection = {
                "ridge": float(alpha),
                "residual_scale": float(residual_scale),
                "calibration_BA": ba,
                "calibration_CE": ce,
                "residual_rank": int(coordinate_values.shape[1]),
                "residual_parameters": int(coordinate_values.shape[1] * invariant_features.shape[1]),
            }
            candidates.append((ba, -ce, -float(residual_scale), -float(alpha), rescued, logits, selection))
    best = max(candidates, key=lambda item: item[:4])
    return best[4], best[5], best[6]


def _score_outcome(
    method: str,
    family: str,
    fold: int,
    seed: int,
    meta: pd.DataFrame,
    representation: np.ndarray,
    logits: np.ndarray,
    selection: Mapping[str, Any],
    outcome_subjects: Sequence[str],
    draw: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed = set(map(str, outcome_subjects))
    mask = meta.subject_id.astype(str).isin(allowed).to_numpy() & (meta.session_id.to_numpy(dtype=int) == 2)
    frame = meta.loc[mask].reset_index(drop=True)
    truth = frame.label.to_numpy(dtype=np.int64)
    probability = softmax(logits[mask])
    prediction = probability.argmax(axis=1)
    probe, probe_subject, _ = subject_probe(meta, representation, outcome_subjects)
    subject_rows = []
    for subject, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        subject_rows.append(
            {
                "family": family,
                "fold": fold,
                "seed": seed,
                "subject_id": str(subject),
                "rescue_method": method,
                "random_draw": draw,
                "balanced_accuracy": balanced_accuracy(truth[positions], prediction[positions]),
                "accuracy": float(np.mean(truth[positions] == prediction[positions])),
                "macro_f1": macro_f1(truth[positions], prediction[positions]),
                "subject_probe_accuracy": probe_subject[str(subject)],
                "outer_test_used": False,
            }
        )
    result = {
        "family": family,
        "fold": fold,
        "seed": seed,
        "rescue_method": method,
        "random_draw": draw,
        "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in subject_rows])),
        "accuracy": float(np.mean(prediction == truth)),
        "macro_f1": float(np.mean([row["macro_f1"] for row in subject_rows])),
        "cross_entropy": ce_loss(truth, probability),
        "subject_probe_accuracy": probe["balanced_accuracy"],
        "subject_probe_chance": probe["chance"],
        "selection": json.dumps(dict(selection), sort_keys=True),
        "selected_ridge": selection.get("ridge"),
        "selected_residual_scale": selection.get("residual_scale"),
        "residual_rank": selection.get("residual_rank", 0),
        "residual_parameters": selection.get("residual_parameters", 0),
        "selection_role": "calibration_subjects_session_2_only",
        "outcome_used_for_selection": False,
        "outer_test_used": False,
    }
    return result, subject_rows


def _primary_audit_name(family: str, config: Mapping[str, Any]) -> str:
    if family == "A_SUBJECT_GRL_EEGNET":
        value = int(round(float(config["primary_grl_lambda"]) * 1000))
        return f"A_SUBJECT_GRL_EEGNET_L{value:04d}"
    return family


def determine_eligibility(audit: pd.DataFrame) -> dict[str, Any]:
    config = load_config()
    results = {}
    for family in primary_pairs(config):
        audit_name = _primary_audit_name(family, config)
        frame = audit[audit.family == audit_name]
        complete = len(frame) == len(config["development_folds"]) * len(config["seeds"])
        finite_id = frame.delta_ID.notna().all() and complete
        finite_prs = frame.delta_PRS.notna().all() and complete
        finite_ba = frame.delta_BA_INV.notna().all() and complete
        mean_id = float(frame.delta_ID.mean()) if finite_id else None
        mean_prs = float(frame.delta_PRS.mean()) if finite_prs else None
        mean_ba = float(frame.delta_BA_INV.mean()) if finite_ba else None
        i1 = bool(finite_id and mean_id < 0)
        i2 = bool(finite_prs and mean_prs < 0)
        i3 = bool(finite_ba and mean_ba < 0)
        if not i1:
            status = "NO_MEASURABLE_INVARIANCE_EFFECT"
        elif not i2:
            status = "INVARIANCE_PRESERVES_PROTECTED_STRUCTURE"
        elif not i3:
            status = "PROTECTED_LOSS_WITHOUT_TASK_HARM"
        else:
            status = "ELIGIBLE_PROTECTED_LOSS"
        results[family] = {
            "audit_family": audit_name,
            "runs_complete": int(len(frame)),
            "expected_runs": int(len(config["development_folds"]) * len(config["seeds"])),
            "mean_delta_ID": mean_id,
            "mean_delta_PRS": mean_prs,
            "mean_delta_BA_INV": mean_ba,
            "I1": i1,
            "I2": i2,
            "I3": i3,
            "status": status,
            "eligible": status == "ELIGIBLE_PROTECTED_LOSS",
        }
    payload = {"families": results, "outer_test_used": False}
    write_json(OUTPUTS / "ELIGIBILITY.json", payload)
    return payload


def _generic_dimensions(spectrum: Mapping[str, Any], protected: Sequence[int]) -> list[int]:
    excluded = set(map(int, protected))
    ordered = [int(index) for index in np.argsort(-np.asarray(spectrum["rho"])) if int(index) not in excluded]
    return ordered[: len(protected)]


def run_rescue_one(
    family: str,
    task_method: str,
    invariant_method: str,
    fold: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config()
    rescue_config = config["rescue"]
    split = load_development_split(fold)
    meta, teacher, _ = aligned_representation(task_method, fold, seed)
    inv_meta, invariant, _ = aligned_representation(invariant_method, fold, seed)
    if not np.array_equal(meta.manifest_position.to_numpy(), inv_meta.manifest_position.to_numpy()):
        raise RuntimeError("teacher/invariant alignment failure")
    spectrum, assignment = load_spectrum(spectrum_path(family, fold, seed))
    protected = list(map(int, assignment.get("protected_dimensions", [])))
    if not protected:
        raise RuntimeError(f"eligible family {family} has no protected dimensions f{fold}s{seed}")
    generic = _generic_dimensions(spectrum, protected)
    rank = len(protected)
    labels = meta.label.to_numpy(dtype=np.int64)
    model_fit = set(split.model_fit_subjects)
    calibration = set(split.calibration_subjects)
    fit_mask = meta.subject_id.astype(str).isin(model_fit).to_numpy()
    calibration_mask = (
        meta.subject_id.astype(str).isin(calibration).to_numpy()
        & (meta.session_id.to_numpy(dtype=int) == int(config["calibration_session"]))
    )
    ridge_values = list(map(float, rescue_config["ridge_values"]))
    residual_scales = list(map(float, rescue_config["residual_scales"]))
    rows, subject_rows = [], []

    invariant_logits, invariant_selection = _select_plain_head(
        invariant, labels, meta, fit_mask, calibration_mask, ridge_values
    )
    result, subjects = _score_outcome(
        "R0_INVARIANT_ONLY", family, fold, seed, meta, invariant, invariant_logits,
        invariant_selection, split.outcome_subjects
    )
    rows.append(result)
    subject_rows.extend(subjects)

    persist_coordinates = coordinates(teacher, spectrum, protected)
    persist_representation, persist_logits, persist_selection = _select_rescue(
        invariant, teacher, persist_coordinates, labels, meta, fit_mask, calibration_mask,
        ridge_values, residual_scales
    )
    result, subjects = _score_outcome(
        "R4_PERSIST_PROTECTED_RESIDUAL", family, fold, seed, meta, persist_representation,
        persist_logits, persist_selection, split.outcome_subjects
    )
    rows.append(result)
    subject_rows.extend(subjects)

    generic_coordinates = coordinates(teacher, spectrum, generic)
    generic_representation, generic_logits, generic_selection = _select_rescue(
        invariant, teacher, generic_coordinates, labels, meta, fit_mask, calibration_mask,
        ridge_values, residual_scales
    )
    result, subjects = _score_outcome(
        "R2_GENERIC_PERSISTENT_RESIDUAL", family, fold, seed, meta, generic_representation,
        generic_logits, generic_selection, split.outcome_subjects
    )
    rows.append(result)
    subject_rows.extend(subjects)

    pca_coordinates = (teacher - spectrum["mean"]) @ spectrum["pca_vectors"][:, :rank]
    pca_representation, pca_logits, pca_selection = _select_rescue(
        invariant, teacher, pca_coordinates, labels, meta, fit_mask, calibration_mask,
        ridge_values, residual_scales
    )
    result, subjects = _score_outcome(
        "R3_PCA_RESIDUAL", family, fold, seed, meta, pca_representation,
        pca_logits, pca_selection, split.outcome_subjects
    )
    rows.append(result)
    subject_rows.extend(subjects)

    centered_teacher = teacher - teacher[fit_mask].mean(axis=0, keepdims=True)
    rng = np.random.default_rng(stable_seed("random-rescue", family, fold, seed))
    for draw in range(int(rescue_config["random_draws"])):
        matrix = rng.normal(size=(teacher.shape[1], rank))
        basis, _ = np.linalg.qr(matrix)
        random_coordinates = centered_teacher @ basis[:, :rank]
        random_representation, random_logits, random_selection = _select_rescue(
            invariant, teacher, random_coordinates, labels, meta, fit_mask, calibration_mask,
            ridge_values, residual_scales
        )
        result, subjects = _score_outcome(
            "R1_RANDOM_RESIDUAL", family, fold, seed, meta, random_representation,
            random_logits, random_selection, split.outcome_subjects, draw=draw
        )
        rows.append(result)
        subject_rows.extend(subjects)

    task_logits, task_selection = _select_plain_head(teacher, labels, meta, fit_mask, calibration_mask, ridge_values)
    result, subjects = _score_outcome(
        "R5_TASK_ONLY_UPPER_REFERENCE", family, fold, seed, meta, teacher, task_logits,
        task_selection, split.outcome_subjects
    )
    rows.append(result)
    subject_rows.extend(subjects)

    full_representation, full_logits, full_selection = _select_rescue(
        invariant, teacher, centered_teacher, labels, meta, fit_mask, calibration_mask,
        ridge_values, residual_scales
    )
    result, subjects = _score_outcome(
        "R6_FULL_TASK_TEACHER_UPPER_BOUND", family, fold, seed, meta, full_representation,
        full_logits, full_selection, split.outcome_subjects
    )
    rows.append(result)
    subject_rows.extend(subjects)
    for row in rows:
        row["protected_rank"] = rank
        row["protected_dimensions"] = json.dumps(protected)
        row["generic_dimensions"] = json.dumps(generic)
    return rows, subject_rows


def run_eligible_rescues() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = load_config()
    audit = pd.read_csv(OUTPUTS / "INVARIANCE_AUDIT.csv")
    eligibility = determine_eligibility(audit)
    rows, subjects = [], []
    for family, (task, invariant) in primary_pairs(config).items():
        if not eligibility["families"][family]["eligible"]:
            print(f"[rescue] skip {family}: {eligibility['families'][family]['status']}", flush=True)
            continue
        for fold in config["development_folds"]:
            for seed in config["seeds"]:
                print(f"[rescue] f{fold}s{seed} {family}", flush=True)
                run_rows, run_subjects = run_rescue_one(family, task, invariant, int(fold), int(seed))
                rows.extend(run_rows)
                subjects.extend(run_subjects)
    result_frame = pd.DataFrame(rows)
    subject_frame = pd.DataFrame(subjects)
    if not len(result_frame):
        result_frame = pd.DataFrame(columns=[
            "family", "fold", "seed", "rescue_method", "random_draw", "balanced_accuracy",
            "accuracy", "macro_f1", "subject_probe_accuracy", "outer_test_used"
        ])
    if not len(subject_frame):
        subject_frame = pd.DataFrame(columns=[
            "family", "fold", "seed", "subject_id", "rescue_method", "random_draw",
            "balanced_accuracy", "accuracy", "macro_f1", "subject_probe_accuracy", "outer_test_used"
        ])
    write_csv(OUTPUTS / "RESCUE_RESULTS.csv", result_frame)
    write_csv(OUTPUTS / "RESCUE_SUBJECT_RESULTS.csv", subject_frame)
    return result_frame, subject_frame, eligibility

