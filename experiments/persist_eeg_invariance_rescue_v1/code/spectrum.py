from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    CACHE,
    OUTPUTS,
    balanced_accuracy,
    ce_loss,
    clean,
    load_config,
    macro_f1,
    sha256_bytes,
    softmax,
    stable_seed,
    stable_uint64,
    subject_sort,
    write_csv,
    write_json,
)
from data import load_development_split, load_manifest, select_frame
from models import primary_pairs
from train import load_representation


PROBE_DIM = 16
EPSILON_NEUTRAL = 0.005


def aligned_representation(method_id: str, fold: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    split = load_development_split(fold)
    manifest = load_manifest(split)
    cached = load_representation(method_id, fold, seed)
    lookup = manifest.set_index("manifest_position", drop=False)
    positions = cached["positions"].astype(np.int64)
    if not set(map(int, positions)).issubset(set(map(int, lookup.index))):
        raise RuntimeError("representation contains a position outside the development manifest")
    frame = lookup.loc[positions].reset_index(drop=True)
    if not np.array_equal(frame.manifest_position.to_numpy(dtype=np.int64), positions):
        raise RuntimeError("representation/manifest alignment failure")
    return frame, cached["features"].astype(np.float32), cached["logits"].astype(np.float32)


def make_blocks(rho: np.ndarray, maximum_rank: int) -> tuple[list[list[int]], dict[str, Any]]:
    gaps = np.abs(np.diff(rho))
    threshold = max(float(np.median(gaps) * 4.0), float(np.max(np.abs(rho))) * 0.05, 1e-10)
    bounds = [0]
    for index, gap in enumerate(gaps):
        if gap > threshold:
            bounds.append(index + 1)
    bounds.append(len(rho))
    blocks: list[list[int]] = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        for start in range(left, right, int(maximum_rank)):
            blocks.append(list(range(start, min(start + int(maximum_rank), right))))
    if len(blocks) < 2:
        midpoint = min(int(maximum_rank), max(1, len(rho) // 2))
        blocks = [list(range(0, midpoint)), list(range(midpoint, len(rho)))]
        blocks = [block for block in blocks if block]
    return blocks, {
        "construction": "train-only eigengap clustering followed by max-rank split",
        "eigengap_threshold": threshold,
        "max_block_rank": int(maximum_rank),
        "block_dimensions": [len(block) for block in blocks],
        "no_calibration_or_outcome_block_selection": True,
    }


def build_spectrum(meta: pd.DataFrame, features: np.ndarray, fold: int, seed: int) -> dict[str, Any]:
    config = load_config()
    spectrum_config = config["spectrum"]
    x = np.asarray(features, dtype=np.float64)
    mu = x.mean(axis=0)
    centered = x - mu
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    threshold = max(float(eigenvalues[0]) * 1e-3, 1e-8)
    numerical_rank = int(np.sum(eigenvalues > threshold))
    rank = min(int(spectrum_config["whitening_rank"]), numerical_rank)
    if rank < 4:
        raise RuntimeError(f"insufficient active whitening rank: {numerical_rank}")
    active = np.maximum(eigenvalues[:rank], max(float(eigenvalues[:rank].mean()) * 1e-4, 1e-8))
    pca_vectors = eigenvectors[:, :rank]
    whitener = pca_vectors * np.power(active, -0.5)[None, :]
    dewhitener = np.sqrt(active)[:, None] * pca_vectors.T
    whitened = centered @ whitener

    frame = meta.reset_index(drop=True).copy()
    frame["local_position"] = np.arange(len(frame), dtype=np.int64)
    sessions = sorted(frame.session_id.astype(int).unique())
    if sessions != [1, 2]:
        raise RuntimeError(f"two sessions required, got {sessions}")
    centroids: dict[tuple[str, int, int], np.ndarray] = {}
    for key, group in frame.groupby(["subject_id", "session_id", "label"], sort=True):
        centroids[(str(key[0]), int(key[1]), int(key[2]))] = whitened[
            group.local_position.to_numpy(dtype=np.int64)
        ].mean(axis=0)
    subjects = subject_sort(frame.subject_id.astype(str).unique())
    class_covariances = []
    pair_counts: dict[str, int] = {}
    for label in (0, 1):
        first, second = [], []
        for subject in subjects:
            left = (subject, 1, label)
            right = (subject, 2, label)
            if left in centroids and right in centroids:
                first.append(centroids[left])
                second.append(centroids[right])
        if first:
            a, b = np.asarray(first), np.asarray(second)
            a -= a.mean(axis=0)
            b -= b.mean(axis=0)
            class_covariances.append((a.T @ b + b.T @ a) / (2.0 * max(len(a), 1)))
            pair_counts[str(label)] = len(a)
    persistence_covariance = np.mean(class_covariances, axis=0)
    rho, directions = np.linalg.eigh((persistence_covariance + persistence_covariance.T) / 2.0)
    order = np.argsort(rho)[::-1]
    rho, directions = rho[order], directions[:, order]
    blocks, block_metadata = make_blocks(rho, int(spectrum_config["max_block_rank"]))

    null_values: list[list[float]] = [[] for _ in blocks]
    rng = np.random.default_rng(stable_seed("persistence-null", fold, seed))
    for _ in range(int(spectrum_config["permutation_draws"])):
        permutation = rng.permutation(len(subjects))
        for label in (0, 1):
            first, second = [], []
            for subject_index, subject in enumerate(subjects):
                left = (subject, 1, label)
                right = (subjects[int(permutation[subject_index])], 2, label)
                if left in centroids and right in centroids:
                    first.append(centroids[left])
                    second.append(centroids[right])
            if len(first) >= 3:
                a, b = np.asarray(first), np.asarray(second)
                a -= a.mean(axis=0)
                b -= b.mean(axis=0)
                permuted_covariance = (a.T @ b + b.T @ a) / (2.0 * len(a))
                for block_index, block in enumerate(blocks):
                    projected = directions[:, block].T @ permuted_covariance @ directions[:, block]
                    null_values[block_index].append(float(np.mean(np.diag(projected))))
    support = []
    for block_index, block in enumerate(blocks):
        values = np.asarray(null_values[block_index], dtype=np.float64)
        observed = float(np.mean(rho[block]))
        null_p95 = float(np.quantile(values, 0.95)) if len(values) else float("inf")
        support.append(
            {
                "block": block_index,
                "rho_G": observed,
                "null_mean": float(values.mean()) if len(values) else None,
                "null_p95": null_p95,
                "persistence_supported": bool(observed > null_p95),
                "dimensions": len(block),
                "eigenvalue_range": [float(rho[block[0]]), float(rho[block[-1]])],
            }
        )
    whitening_error = whitened.T @ whitened / max(len(whitened) - 1, 1) - np.eye(rank)
    return {
        "mean": mu.astype(np.float32),
        "whitener": whitener.astype(np.float32),
        "dewhitener": dewhitener.astype(np.float32),
        "pca_vectors": pca_vectors.astype(np.float32),
        "pca_eigenvalues": active.astype(np.float32),
        "directions": directions.astype(np.float32),
        "rho": rho.astype(np.float32),
        "blocks": blocks,
        "audit": {
            "fold": fold,
            "seed": seed,
            "nominal_embedding_dimension": int(x.shape[1]),
            "numerical_rank": numerical_rank,
            "whitening_rank": rank,
            "whitening_error_max_abs": float(np.max(np.abs(whitening_error))),
            "pair_counts": pair_counts,
            "rho": rho.tolist(),
            "blocks": blocks,
            "block_metadata": block_metadata,
            "persistence_support": support,
            "null_permutations": int(spectrum_config["permutation_draws"]),
            "fit_roles": ["model_fit_session_1", "model_fit_session_2"],
            "calibration_used": False,
            "outcome_used": False,
            "outer_test_used": False,
        },
    }


def coordinates(features: np.ndarray, spectrum: Mapping[str, Any], dimensions: Sequence[int] | None = None) -> np.ndarray:
    value = (np.asarray(features, dtype=np.float64) - spectrum["mean"]) @ spectrum["whitener"] @ spectrum["directions"]
    if dimensions is not None:
        value = value[:, np.asarray(dimensions, dtype=np.int64)]
    return value.astype(np.float32)


def erase(features: np.ndarray, spectrum: Mapping[str, Any], dimensions: Sequence[int]) -> np.ndarray:
    selected = np.asarray(dimensions, dtype=np.int64)
    q = coordinates(features, spectrum).astype(np.float64)
    delta = np.zeros_like(q)
    delta[:, selected] = -q[:, selected]
    reconstructed = (delta @ spectrum["directions"].T) @ spectrum["dewhitener"]
    return (np.asarray(features, dtype=np.float64) + reconstructed).astype(np.float32)


def save_spectrum(path: Path, spectrum: Mapping[str, Any], assignments: Mapping[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.npz")
    np.savez_compressed(
        temporary,
        mean=spectrum["mean"],
        whitener=spectrum["whitener"],
        dewhitener=spectrum["dewhitener"],
        pca_vectors=spectrum["pca_vectors"],
        pca_eigenvalues=spectrum["pca_eigenvalues"],
        directions=spectrum["directions"],
        rho=spectrum["rho"],
        blocks_json=np.asarray(json.dumps(spectrum["blocks"], sort_keys=True)),
        audit_json=np.asarray(json.dumps(clean(spectrum["audit"]), sort_keys=True)),
        assignments_json=np.asarray(json.dumps(clean(assignments or {}), sort_keys=True)),
    )
    os.replace(temporary, path)


def load_spectrum(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = np.load(path, allow_pickle=False)
    spectrum = {
        name: value[name]
        for name in ("mean", "whitener", "dewhitener", "pca_vectors", "pca_eigenvalues", "directions", "rho")
    }
    spectrum["blocks"] = json.loads(str(value["blocks_json"].item()))
    spectrum["audit"] = json.loads(str(value["audit_json"].item()))
    assignments = json.loads(str(value["assignments_json"].item()))
    return spectrum, assignments


def _ridge_probe(features: np.ndarray, labels: np.ndarray, alpha: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    if x.shape[1] > PROBE_DIM:
        x = x[:, :PROBE_DIM]
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    design = np.concatenate([(x - mean) / std, np.ones((len(x), 1))], axis=1)
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    target = np.eye(2)[np.asarray(labels, dtype=np.int64)]
    system = design.T @ design + float(alpha) * penalty
    try:
        weights = np.linalg.solve(system, design.T @ target)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ design.T @ target
    return weights, mean, std


def _probe_probability(features: np.ndarray, pack: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean, std = pack
    x = np.asarray(features, dtype=np.float64)
    if x.shape[1] > PROBE_DIM:
        x = x[:, :PROBE_DIM]
    design = np.concatenate([(x - mean) / std, np.ones((len(x), 1))], axis=1)
    return softmax(design @ weights)


def _risk(features_fit: np.ndarray, labels_fit: np.ndarray, features_eval: np.ndarray, labels_eval: np.ndarray) -> tuple[float, float]:
    probability = _probe_probability(features_eval, _ridge_probe(features_fit, labels_fit))
    return ce_loss(labels_eval, probability), balanced_accuracy(labels_eval, probability.argmax(axis=1))


def _bootstrap(values: Sequence[float], seed: int, draws: int) -> dict[str, Any]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"mean": None, "median": None, "ci95": [None, None], "sign_probability": None, "draws": draws, "n_subjects": 0}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(int(draws), len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "sign_probability": float(np.mean(sampled > 0)),
        "draws": int(draws),
        "n_subjects": int(len(array)),
    }


def _split_subjects(subjects: Sequence[str], fold: int, seed: int, inner: int) -> tuple[list[str], list[str]]:
    values = subject_sort(subjects)
    rng = np.random.default_rng(stable_seed("signed-inner", fold, seed, inner))
    rng.shuffle(values)
    midpoint = max(1, len(values) // 2)
    return subject_sort(values[:midpoint]), subject_sort(values[midpoint:])


def _stable_sample(
    meta: pd.DataFrame,
    subjects: Sequence[str],
    fold: int,
    seed: int,
    inner: int,
    purpose: str,
    cap: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    selected, ledger = [], []
    frame = meta[meta.subject_id.astype(str).isin(set(map(str, subjects)))]
    for (subject, session, label), group in frame.groupby(["subject_id", "session_id", "label"], sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        count_before = len(indices)
        if cap and len(indices) > cap:
            rng = np.random.default_rng(stable_uint64("signed-sample", fold, seed, inner, purpose, subject, session, label))
            indices = np.sort(rng.choice(indices, size=cap, replace=False))
        selected.extend(map(int, indices))
        for index in indices:
            ledger.append(
                {
                    "frame_index": int(index),
                    "manifest_position": int(meta.loc[int(index), "manifest_position"]),
                    "trial_uid": str(meta.loc[int(index), "trial_uid"]),
                    "subject_id": str(subject),
                    "session_id": int(session),
                    "label": int(label),
                    "fold": fold,
                    "seed": seed,
                    "inner_split": inner,
                    "purpose": purpose,
                    "group_count_before": count_before,
                    "selected_count": len(indices),
                    "outer_test_used": False,
                }
            )
    return np.asarray(sorted(selected), dtype=np.int64), ledger


def signed_utility_audit(
    meta: pd.DataFrame,
    features: np.ndarray,
    spectrum: Mapping[str, Any],
    fold: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = load_config()
    spec_config = config["spectrum"]
    subjects = subject_sort(meta.subject_id.astype(str).unique())
    inner_records = []
    samples: list[dict[str, Any]] = []
    for inner in range(int(spec_config["inner_splits"])):
        fit_subjects, eval_subjects = _split_subjects(subjects, fold, seed, inner)
        fit_indices, fit_rows = _stable_sample(
            meta, fit_subjects, fold, seed, inner, "fit", int(spec_config["per_group_cap"])
        )
        eval_indices, eval_rows = _stable_sample(
            meta, eval_subjects, fold, seed, inner, "eval", int(spec_config["per_group_cap"])
        )
        inner_records.append((inner, fit_indices, eval_indices))
        samples.extend(fit_rows)
        samples.extend(eval_rows)

    utility_rows: list[dict[str, Any]] = []
    for block_index, block in enumerate(spectrum["blocks"]):
        absolute_by_subject: dict[str, list[float]] = {}
        specific_by_subject: dict[str, list[float]] = {}
        ba_by_subject: dict[str, list[float]] = {}
        random_ba_by_subject: dict[str, list[float]] = {}
        for inner, fit_indices, eval_indices in inner_records:
            y_fit = meta.loc[fit_indices, "label"].to_numpy(dtype=np.int64)
            y_eval = meta.loc[eval_indices, "label"].to_numpy(dtype=np.int64)
            baseline_pack = _ridge_probe(features[fit_indices], y_fit)
            baseline_probability = _probe_probability(features[eval_indices], baseline_pack)
            erased_fit = erase(features[fit_indices], spectrum, block)
            erased_eval = erase(features[eval_indices], spectrum, block)
            erased_probability = _probe_probability(erased_eval, _ridge_probe(erased_fit, y_fit))
            candidates = np.setdiff1d(np.arange(len(spectrum["rho"])), np.asarray(block, dtype=np.int64))
            if len(candidates) < len(block):
                candidates = np.arange(len(spectrum["rho"]))
            rng = np.random.default_rng(stable_seed("signed-random", fold, seed, inner, block_index))
            random_probabilities = []
            for _ in range(int(spec_config["random_erasures"])):
                choice = np.sort(rng.choice(candidates, size=len(block), replace=False))
                random_fit = erase(features[fit_indices], spectrum, choice)
                random_eval = erase(features[eval_indices], spectrum, choice)
                random_probabilities.append(_probe_probability(random_eval, _ridge_probe(random_fit, y_fit)))
            eval_meta = meta.loc[eval_indices].reset_index(drop=True)
            for subject, group in eval_meta.groupby(eval_meta.subject_id.astype(str), sort=True):
                locations = group.index.to_numpy(dtype=np.int64)
                truth = y_eval[locations]
                baseline_ce = ce_loss(truth, baseline_probability[locations])
                erased_ce = ce_loss(truth, erased_probability[locations])
                baseline_ba = balanced_accuracy(truth, baseline_probability[locations].argmax(axis=1))
                erased_ba = balanced_accuracy(truth, erased_probability[locations].argmax(axis=1))
                random_ce_change, random_ba_change = [], []
                for probability in random_probabilities:
                    random_ce_change.append(ce_loss(truth, probability[locations]) - baseline_ce)
                    random_ba_change.append(
                        balanced_accuracy(truth, probability[locations].argmax(axis=1)) - baseline_ba
                    )
                absolute = erased_ce - baseline_ce
                specific = absolute - float(np.mean(random_ce_change))
                key = str(subject)
                absolute_by_subject.setdefault(key, []).append(float(absolute))
                specific_by_subject.setdefault(key, []).append(float(specific))
                ba_by_subject.setdefault(key, []).append(float(erased_ba - baseline_ba))
                random_ba_by_subject.setdefault(key, []).append(float(np.mean(random_ba_change)))
        absolute = {subject: float(np.mean(values)) for subject, values in absolute_by_subject.items()}
        specific = {subject: float(np.mean(values)) for subject, values in specific_by_subject.items()}
        ba_change = {subject: float(np.mean(values)) for subject, values in ba_by_subject.items()}
        random_ba = {subject: float(np.mean(values)) for subject, values in random_ba_by_subject.items()}
        absolute_boot = _bootstrap(absolute.values(), stable_seed("u-abs", fold, seed, block_index), int(config["bootstrap_draws"]))
        specific_boot = _bootstrap(specific.values(), stable_seed("u-spec", fold, seed, block_index), int(config["bootstrap_draws"]))
        support = spectrum["audit"]["persistence_support"][block_index]
        utility_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "block": block_index,
                "dimensions": len(block),
                "coordinate_ids": json.dumps(block),
                "persistence_supported": bool(support["persistence_supported"]),
                "rho_G": float(support["rho_G"]),
                "n_unique_subjects": len(absolute),
                "u_abs_mean": absolute_boot["mean"],
                "u_abs_CI95_low": absolute_boot["ci95"][0],
                "u_abs_CI95_high": absolute_boot["ci95"][1],
                "u_spec_mean": specific_boot["mean"],
                "u_spec_CI95_low": specific_boot["ci95"][0],
                "u_spec_CI95_high": specific_boot["ci95"][1],
                "raw_BA_change": float(np.mean(list(ba_change.values()))) if ba_change else None,
                "same_rank_random_BA_change": float(np.mean(list(random_ba.values()))) if random_ba else None,
                "random_interventions": int(spec_config["random_erasures"]),
                "outer_test_used": False,
            }
        )
    frame = pd.DataFrame(utility_rows)
    protected = frame[
        frame.persistence_supported
        & (frame.u_abs_CI95_low > 0)
        & (frame.u_spec_CI95_low > 0)
    ].block.astype(int).tolist()
    harmful = frame[
        frame.persistence_supported
        & (frame.u_abs_CI95_high < 0)
        & (frame.u_spec_CI95_high < 0)
    ].block.astype(int).tolist()
    neutral = frame[
        frame.persistence_supported
        & (frame.u_abs_CI95_low >= -EPSILON_NEUTRAL)
        & (frame.u_abs_CI95_high <= EPSILON_NEUTRAL)
    ].block.astype(int).tolist()
    all_blocks = set(frame.block.astype(int))
    assignment = {
        "protected_blocks": protected,
        "harmful_blocks": harmful,
        "neutral_blocks": neutral,
        "uncertain_blocks": sorted(all_blocks - set(protected) - set(harmful) - set(neutral)),
        "protected_dimensions": sorted({dimension for block in protected for dimension in spectrum["blocks"][block]}),
        "definition": "Signed-V3.1 persistence support plus u_abs LCB>0 and u_spec LCB>0",
        "epsilon_neutral": EPSILON_NEUTRAL,
        "fit_role": "model_fit_only",
        "outcome_used": False,
        "outer_test_used": False,
    }
    return assignment, frame, pd.DataFrame(samples)


def spectrum_path(family: str, fold: int, seed: int) -> Path:
    return CACHE / "protected_spectra" / family / f"fold-{fold}" / f"seed-{seed}.npz"


def ensure_family_spectrum(
    family: str, task_method: str, fold: int, seed: int, force: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = spectrum_path(family, fold, seed)
    if path.exists() and not force:
        return load_spectrum(path)
    split = load_development_split(fold)
    meta, features, _ = aligned_representation(task_method, fold, seed)
    mask = meta.subject_id.isin(set(split.model_fit_subjects)).to_numpy()
    fit_meta = meta.loc[mask].reset_index(drop=True)
    fit_features = features[mask]
    spectrum = build_spectrum(fit_meta, fit_features, fold, seed)
    assignments, utility, sampling = signed_utility_audit(fit_meta, fit_features, spectrum, fold, seed)
    save_spectrum(path, spectrum, assignments)
    audit_dir = OUTPUTS / "audit" / family / f"fold-{fold}" / f"seed-{seed}"
    write_csv(audit_dir / "SIGNED_UTILITY.csv", utility)
    write_csv(audit_dir / "SIGNED_SAMPLING.csv", sampling)
    write_json(audit_dir / "SIGNED_ASSIGNMENTS.json", assignments)
    write_json(audit_dir / "SPECTRUM_AUDIT.json", spectrum["audit"])
    return spectrum, assignments


def subject_probe(
    meta: pd.DataFrame, features: np.ndarray, subjects: Sequence[str]
) -> tuple[dict[str, Any], dict[str, float], np.ndarray]:
    allowed = set(map(str, subjects))
    train_mask = meta.subject_id.astype(str).isin(allowed).to_numpy() & (meta.session_id.to_numpy(dtype=int) == 1)
    eval_mask = meta.subject_id.astype(str).isin(allowed).to_numpy() & (meta.session_id.to_numpy(dtype=int) == 2)
    ordered = subject_sort(subjects)
    codes = {subject: index for index, subject in enumerate(ordered)}
    y_train = meta.loc[train_mask, "subject_id"].astype(str).map(codes).to_numpy(dtype=np.int64)
    y_eval = meta.loc[eval_mask, "subject_id"].astype(str).map(codes).to_numpy(dtype=np.int64)
    scaler = StandardScaler().fit(features[train_mask])
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0)
    classifier.fit(scaler.transform(features[train_mask]), y_train)
    prediction = classifier.predict(scaler.transform(features[eval_mask])).astype(np.int64)
    per_subject = {
        subject: float(np.mean(prediction[y_eval == code] == code)) for subject, code in codes.items()
    }
    metrics = {
        "accuracy": float(np.mean(prediction == y_eval)),
        "balanced_accuracy": float(np.mean(list(per_subject.values()))),
        "chance": 1.0 / len(ordered),
        "n_subjects": len(ordered),
        "train_rows": int(train_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
        "capacity": "standardized multinomial logistic C=1.0",
    }
    eval_positions = np.flatnonzero(eval_mask)
    return metrics, per_subject, eval_positions


def _ridge_map_fit(source: np.ndarray, target: np.ndarray, alpha: float = 1e-3) -> dict[str, np.ndarray]:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    design = np.concatenate([(x - x_mean) / x_std, np.ones((len(x), 1))], axis=1)
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    system = design.T @ design + float(alpha) * penalty
    try:
        weights = np.linalg.solve(system, design.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(system) @ design.T @ y
    return {"weights": weights, "x_mean": x_mean, "x_std": x_std, "target_mean": y.mean(axis=0)}


def _ridge_map_predict(source: np.ndarray, pack: Mapping[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(source, dtype=np.float64)
    design = np.concatenate([(x - pack["x_mean"]) / pack["x_std"], np.ones((len(x), 1))], axis=1)
    return design @ pack["weights"]


def _canonical_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(a) < 3 or a.shape[1] == 0:
        return float("nan")
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    ca = a.T @ a / max(len(a) - 1, 1) + 1e-5 * np.eye(a.shape[1])
    cb = b.T @ b / max(len(b) - 1, 1) + 1e-5 * np.eye(b.shape[1])
    cross = a.T @ b / max(len(a) - 1, 1)
    eva, eua = np.linalg.eigh(ca)
    evb, eub = np.linalg.eigh(cb)
    wa = eua @ np.diag(np.power(np.maximum(eva, 1e-8), -0.5)) @ eua.T
    wb = eub @ np.diag(np.power(np.maximum(evb, 1e-8), -0.5)) @ eub.T
    singular = np.linalg.svd(wa @ cross @ wb, compute_uv=False)
    return float(np.mean(np.clip(singular, 0.0, 1.0)))


def protected_retention(
    meta: pd.DataFrame,
    source_features: np.ndarray,
    teacher_features: np.ndarray,
    spectrum: Mapping[str, Any],
    dimensions: Sequence[int],
    model_fit_subjects: Sequence[str],
    outcome_subjects: Sequence[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    dimensions = list(map(int, dimensions))
    if not dimensions:
        return {
            "score": None,
            "canonical_correlation": None,
            "cosine_geometry": None,
            "rank": 0,
            "status": "NO_PROTECTED_ASSIGNMENT",
        }, {str(subject): float("nan") for subject in outcome_subjects}
    target = coordinates(teacher_features, spectrum, dimensions).astype(np.float64)
    fit_mask = (
        meta.subject_id.astype(str).isin(set(map(str, model_fit_subjects))).to_numpy()
        & (meta.session_id.to_numpy(dtype=int) == 1)
    )
    eval_mask = (
        meta.subject_id.astype(str).isin(set(map(str, outcome_subjects))).to_numpy()
        & (meta.session_id.to_numpy(dtype=int) == 2)
    )
    pack = _ridge_map_fit(source_features[fit_mask], target[fit_mask], alpha=1e-3)
    predicted = _ridge_map_predict(source_features[eval_mask], pack)
    observed = target[eval_mask]
    denominator = np.square(observed - pack["target_mean"]).sum()
    score = 1.0 - float(np.square(observed - predicted).sum()) / max(float(denominator), 1e-12)
    centered_observed = observed - pack["target_mean"]
    centered_predicted = predicted - pack["target_mean"]
    cosine = np.sum(centered_observed * centered_predicted, axis=1) / np.maximum(
        np.linalg.norm(centered_observed, axis=1) * np.linalg.norm(centered_predicted, axis=1), 1e-12
    )
    eval_meta = meta.loc[eval_mask].reset_index(drop=True)
    per_subject = {}
    for subject, group in eval_meta.groupby(eval_meta.subject_id.astype(str), sort=True):
        locations = group.index.to_numpy(dtype=np.int64)
        subject_denominator = np.square(observed[locations] - pack["target_mean"]).sum()
        per_subject[str(subject)] = 1.0 - float(np.square(observed[locations] - predicted[locations]).sum()) / max(
            float(subject_denominator), 1e-12
        )
    return {
        "score": score,
        "canonical_correlation": _canonical_correlation(predicted, observed),
        "cosine_geometry": float(np.mean(cosine)),
        "rank": len(dimensions),
        "status": "OK",
        "fit_rows": int(fit_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
    }, per_subject


def _native_subject_metrics(meta: pd.DataFrame, logits: np.ndarray, subjects: Sequence[str]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    mask = (
        meta.subject_id.astype(str).isin(set(map(str, subjects))).to_numpy()
        & (meta.session_id.to_numpy(dtype=int) == 2)
    )
    frame = meta.loc[mask].reset_index(drop=True)
    score = logits[mask]
    truth = frame.label.to_numpy(dtype=np.int64)
    prediction = score.argmax(axis=1)
    probability = softmax(score)
    per_subject = {}
    for subject, group in frame.groupby(frame.subject_id.astype(str), sort=True):
        locations = group.index.to_numpy(dtype=np.int64)
        per_subject[str(subject)] = {
            "balanced_accuracy": balanced_accuracy(truth[locations], prediction[locations]),
            "accuracy": float(np.mean(truth[locations] == prediction[locations])),
            "macro_f1": macro_f1(truth[locations], prediction[locations]),
        }
    aggregate = {
        "balanced_accuracy": float(np.mean([value["balanced_accuracy"] for value in per_subject.values()])),
        "accuracy": float(np.mean(prediction == truth)),
        "macro_f1": float(np.mean([value["macro_f1"] for value in per_subject.values()])),
        "cross_entropy": ce_loss(truth, probability),
        "n_subjects": len(per_subject),
        "n_trials": len(truth),
    }
    return aggregate, per_subject


def _generic_dimensions(spectrum: Mapping[str, Any], protected: Sequence[int]) -> list[int]:
    rank = len(protected)
    excluded = set(map(int, protected))
    ordered = [int(index) for index in np.argsort(-np.asarray(spectrum["rho"])) if int(index) not in excluded]
    return ordered[:rank]


def audit_pair(
    family: str,
    task_method: str,
    invariant_method: str,
    fold: int,
    seed: int,
    force_spectrum: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    split = load_development_split(fold)
    task_meta, task_features, task_logits = aligned_representation(task_method, fold, seed)
    inv_meta, inv_features, inv_logits = aligned_representation(invariant_method, fold, seed)
    if not np.array_equal(task_meta.manifest_position.to_numpy(), inv_meta.manifest_position.to_numpy()):
        raise RuntimeError("matched representation trial alignment failure")
    spectrum, assignment = ensure_family_spectrum(family, task_method, fold, seed, force=force_spectrum)
    protected = list(map(int, assignment.get("protected_dimensions", [])))
    generic = _generic_dimensions(spectrum, protected)

    task_native, task_subject = _native_subject_metrics(task_meta, task_logits, split.outcome_subjects)
    inv_native, inv_subject = _native_subject_metrics(task_meta, inv_logits, split.outcome_subjects)
    task_probe, task_probe_subject, _ = subject_probe(task_meta, task_features, split.outcome_subjects)
    inv_probe, inv_probe_subject, _ = subject_probe(task_meta, inv_features, split.outcome_subjects)
    task_retention, task_retention_subject = protected_retention(
        task_meta, task_features, task_features, spectrum, protected, split.model_fit_subjects, split.outcome_subjects
    )
    inv_retention, inv_retention_subject = protected_retention(
        task_meta, inv_features, task_features, spectrum, protected, split.model_fit_subjects, split.outcome_subjects
    )
    task_nonprotected, _ = protected_retention(
        task_meta, task_features, task_features, spectrum, generic, split.model_fit_subjects, split.outcome_subjects
    )
    inv_nonprotected, _ = protected_retention(
        task_meta, inv_features, task_features, spectrum, generic, split.model_fit_subjects, split.outcome_subjects
    )
    row = {
        "family": family,
        "fold": fold,
        "seed": seed,
        "task_method": task_method,
        "invariant_method": invariant_method,
        "task_only_BA": task_native["balanced_accuracy"],
        "invariant_BA": inv_native["balanced_accuracy"],
        "delta_BA_INV": inv_native["balanced_accuracy"] - task_native["balanced_accuracy"],
        "task_only_accuracy": task_native["accuracy"],
        "invariant_accuracy": inv_native["accuracy"],
        "task_only_macro_f1": task_native["macro_f1"],
        "invariant_macro_f1": inv_native["macro_f1"],
        "task_only_subject_probe": task_probe["balanced_accuracy"],
        "invariant_subject_probe": inv_probe["balanced_accuracy"],
        "delta_ID": inv_probe["balanced_accuracy"] - task_probe["balanced_accuracy"],
        "subject_probe_chance": task_probe["chance"],
        "task_only_PRS": task_retention["score"],
        "invariant_PRS": inv_retention["score"],
        "delta_PRS": (
            inv_retention["score"] - task_retention["score"]
            if inv_retention["score"] is not None and task_retention["score"] is not None
            else None
        ),
        "task_only_PRS_CCA": task_retention["canonical_correlation"],
        "invariant_PRS_CCA": inv_retention["canonical_correlation"],
        "task_only_PRS_cosine": task_retention["cosine_geometry"],
        "invariant_PRS_cosine": inv_retention["cosine_geometry"],
        "task_only_matched_nonprotected_R2": task_nonprotected["score"],
        "invariant_matched_nonprotected_R2": inv_nonprotected["score"],
        "protected_rank": len(protected),
        "protected_dimensions": json.dumps(protected),
        "generic_dimensions": json.dumps(generic),
        "protected_assignment_exists": bool(protected),
        "protected_assignment_fit_role": "model_fit_only",
        "outcome_labels_used_for_assignment": False,
        "outer_test_used": False,
    }
    subject_rows, retention_rows = [], []
    for subject in split.outcome_subjects:
        key = str(subject)
        subject_rows.append(
            {
                "family": family,
                "fold": fold,
                "seed": seed,
                "subject_id": key,
                "task_method": task_method,
                "invariant_method": invariant_method,
                "task_only_BA": task_subject[key]["balanced_accuracy"],
                "invariant_BA": inv_subject[key]["balanced_accuracy"],
                "delta_BA_INV": inv_subject[key]["balanced_accuracy"] - task_subject[key]["balanced_accuracy"],
                "task_only_macro_f1": task_subject[key]["macro_f1"],
                "invariant_macro_f1": inv_subject[key]["macro_f1"],
                "task_only_ID_accuracy": task_probe_subject[key],
                "invariant_ID_accuracy": inv_probe_subject[key],
                "delta_ID": inv_probe_subject[key] - task_probe_subject[key],
                "task_only_PRS": task_retention_subject[key],
                "invariant_PRS": inv_retention_subject[key],
                "delta_PRS": inv_retention_subject[key] - task_retention_subject[key],
                "outer_test_used": False,
            }
        )
        for role, method, metrics, values in (
            ("task_only", task_method, task_retention, task_retention_subject),
            ("invariant", invariant_method, inv_retention, inv_retention_subject),
        ):
            retention_rows.append(
                {
                    "family": family,
                    "fold": fold,
                    "seed": seed,
                    "subject_id": key,
                    "role": role,
                    "method_id": method,
                    "protected_rank": len(protected),
                    "protected_retention_R2_subject": values[key],
                    "protected_retention_R2_run": metrics["score"],
                    "canonical_correlation_run": metrics["canonical_correlation"],
                    "cosine_geometry_run": metrics["cosine_geometry"],
                    "outer_test_used": False,
                }
            )
    return row, subject_rows, retention_rows


def audit_all(force_spectrum: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config()
    pairs = primary_pairs(config)
    pair_list: list[tuple[str, str, str]] = []
    a_task = pairs["A_SUBJECT_GRL_EEGNET"][0]
    for value in config["grl_lambdas"]:
        method = f"A1_SUBJECT_GRL_EEGNET_L{int(round(float(value) * 1000)):04d}"
        pair_list.append((f"A_SUBJECT_GRL_EEGNET_L{int(round(float(value) * 1000)):04d}", a_task, method))
    pair_list.extend((family, task, invariant) for family, (task, invariant) in pairs.items() if not family.startswith("A_"))
    rows, subjects, retention = [], [], []
    for fold in config["development_folds"]:
        for seed in config["seeds"]:
            for family, task, invariant in pair_list:
                spectrum_family = "A_SUBJECT_GRL_EEGNET" if family.startswith("A_SUBJECT") else family
                print(f"[audit] f{fold}s{seed} {family}", flush=True)
                row, subject_rows, retention_rows = audit_pair(
                    spectrum_family, task, invariant, int(fold), int(seed), force_spectrum=force_spectrum
                )
                row["family"] = family
                for item in subject_rows:
                    item["family"] = family
                for item in retention_rows:
                    item["family"] = family
                rows.append(row)
                subjects.extend(subject_rows)
                retention.extend(retention_rows)
    audit_frame = pd.DataFrame(rows)
    subject_frame = pd.DataFrame(subjects)
    retention_frame = pd.DataFrame(retention)
    write_csv(OUTPUTS / "INVARIANCE_AUDIT.csv", audit_frame)
    write_csv(OUTPUTS / "SUBJECT_LEVEL_AUDIT.csv", subject_frame)
    write_csv(OUTPUTS / "PROTECTED_RETENTION.csv", retention_frame)
    return audit_frame, subject_frame, retention_frame

