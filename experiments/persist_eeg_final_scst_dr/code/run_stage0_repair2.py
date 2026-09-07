from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

import common as c
import run_stage0 as v0


LAYER = "final_embedding"
ALPHA_GRID = np.arange(17, dtype=np.float64) / 64.0
ALPHA_MAX = 0.25
METHODS_SUBJECT_MANIFOLD = (
    "no_transport",
    "scst",
    "norm_matched_random",
    "unconditional_subject_transport",
    "wrong_class",
    "subject_permutation",
    "same_class_mixup",
)
METHODS_CLASS = (
    "no_transport",
    "scst",
    "norm_matched_random",
    "unconditional_subject_transport",
    "same_class_mixup",
)


def unit_output(setting: str, fold: int) -> Path:
    return c.RUNTIME / "stage0_repair2_units" / setting / f"fold-{fold}"


def verify_repair2_freeze() -> dict[str, Any]:
    path = c.EXP / "protocol" / "PRE_STAGE0_REPAIR2_FREEZE.json"
    if not path.is_file():
        raise RuntimeError("Repair 2 is blocked until PRE_STAGE0_REPAIR2_FREEZE.json exists")
    freeze = c.read_json(path)
    if freeze.get("pass") is not True or freeze.get("frozen_before_repair2_outcomes") is not True:
        raise RuntimeError("invalid Repair-2 freeze")
    for relative, expected in freeze.get("file_sha256", {}).items():
        target = c.EXP / relative
        if not target.is_file() or c.sha256(target) != expected:
            raise RuntimeError(f"post-freeze Repair-2 input changed: {relative}")
    lock = c.read_json(c.EXP / "protocol" / "STAGE0_REPAIR2_PROTOCOL_LOCK.json")
    if lock.get("sole_layer") != LAYER:
        raise RuntimeError("Repair-2 layer differs from lock")
    grid = np.asarray(lock["step_solver"]["grid"], dtype=np.float64)
    if not np.array_equal(grid, ALPHA_GRID) or float(lock["step_solver"]["alpha_max"]) != ALPHA_MAX:
        raise RuntimeError("Repair-2 solver differs from lock")
    if lock.get("outer_access_allowed") is not False:
        raise RuntimeError("Repair-2 lock does not forbid outer access")
    return freeze


def support_distance(query: np.ndarray, support: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    if support.ndim != 2 or len(support) < 3:
        raise RuntimeError("source support requires at least three centroids")
    support_sq = np.sum(support * support, axis=1)[None, :]
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), batch_size):
        value = query[start:start + batch_size]
        distance_sq = np.sum(value * value, axis=1)[:, None] + support_sq - 2.0 * value @ support.T
        np.maximum(distance_sq, 0.0, out=distance_sq)
        nearest_sq = np.partition(distance_sq, kth=2, axis=1)[:, :3]
        result[start:start + len(value)] = np.sqrt(nearest_sq).mean(axis=1)
    return result


def support_radius(support: np.ndarray) -> tuple[float, np.ndarray]:
    support = np.asarray(support, dtype=np.float64)
    if len(support) < 4:
        raise RuntimeError("leave-one-subject-out 3NN radius requires at least four subjects")
    distance = np.linalg.norm(support[:, None, :] - support[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    clean = np.partition(distance, kth=2, axis=1)[:, :3].mean(axis=1)
    return float(np.quantile(clean, 0.95)), clean


def solve_alpha(
    query: np.ndarray,
    delta: np.ndarray,
    support: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if query.shape != delta.shape:
        raise RuntimeError(f"query/delta shape mismatch: {query.shape} versus {delta.shape}")
    selected = np.zeros(len(query), dtype=np.float64)
    clean_distance = support_distance(query, support)
    for alpha in ALPHA_GRID[1:]:
        distance = support_distance(query + alpha * delta, support)
        selected[distance <= radius] = alpha
    realized_distance = support_distance(query + selected[:, None] * delta, support)
    return selected, clean_distance, realized_distance


def alpha_summary(
    values: np.ndarray,
    norm_ratio: np.ndarray,
    setting: str,
    fold: int,
    source_subject: str,
    target_subject: str,
    label: int,
    query_unit: str,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    norm_ratio = np.asarray(norm_ratio, dtype=np.float64)
    return {
        "setting_id": setting,
        "fold": fold,
        "layer": LAYER,
        "source_subject": source_subject,
        "target_subject": target_subject,
        "class_label": label,
        "query_unit": query_unit,
        "candidate_count": int(len(values)),
        "fraction_alpha_zero": float(np.mean(values == 0.0)),
        "alpha_mean": float(values.mean()),
        "alpha_median": float(np.median(values)),
        "alpha_q25": float(np.quantile(values, 0.25)),
        "alpha_q75": float(np.quantile(values, 0.75)),
        "fraction_alpha_max": float(np.mean(values == ALPHA_MAX)),
        "realized_norm_ratio_mean": float(norm_ratio.mean()),
        "realized_norm_ratio_median": float(np.median(norm_ratio)),
    }


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def run_unit(
    setting: str,
    fold: int,
    source: dict[str, Any],
    validation: dict[str, Any],
    source_sessions: tuple[int, int],
) -> None:
    target = unit_output(setting, fold)
    complete_path = target / "UNIT_COMPLETE.json"
    if complete_path.is_file() and c.read_json(complete_path).get("pass") is True:
        print(f"[cached-repair2] {setting} fold={fold}", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)

    v0_complete_path = c.RUNTIME / "stage0_units" / setting / f"fold-{fold}" / LAYER / "UNIT_COMPLETE.json"
    if not v0_complete_path.is_file():
        raise RuntimeError(f"missing V0 unit: {v0_complete_path}")
    v0_complete = c.read_json(v0_complete_path)

    h_source = np.asarray(source["layers"][LAYER], dtype=np.float64)
    h_validation = np.asarray(validation["layers"][LAYER], dtype=np.float64)
    s_bank, s_eval = source_sessions
    bank_mask = source["sessions"].astype(int) == int(s_bank)
    center = h_source[bank_mask].mean(axis=0)
    scale = h_source[bank_mask].std(axis=0)
    scale[scale < 1e-6] = 1.0
    current_hashes = {
        "feature_scope_sha256": c.array_sha256(source["indices"]),
        "scaling_center_sha256": c.array_sha256(center),
        "scaling_scale_sha256": c.array_sha256(scale),
    }
    for key, value in current_hashes.items():
        if value != v0_complete.get(key):
            raise RuntimeError(f"{setting} fold={fold}: Repair-2 {key} differs from V0")

    z_source = (h_source - center) / scale
    z_validation = (h_validation - center) / scale
    labels = sorted(map(int, np.unique(source["labels"])))
    subjects = c.subject_sort(np.unique(source["subjects"].astype(str)))
    validation_subjects = c.subject_sort(np.unique(validation["subjects"].astype(str)))
    subject_code = {subject: index for index, subject in enumerate(subjects)}
    centroids = v0.centroid_map(z_source, source["subjects"], source["labels"], source["sessions"])
    validation_centroids = v0.centroid_map(
        z_validation, validation["subjects"], validation["labels"], validation["sessions"]
    )
    residual, _ = v0.population_residuals(centroids, subjects, labels, source_sessions)

    support: dict[int, np.ndarray] = {}
    source_radius: dict[int, float] = {}
    source_clean_loo: dict[int, np.ndarray] = {}
    for label in labels:
        support[label] = np.stack([centroids[(subject, label, s_bank)] for subject in subjects])
        source_radius[label], source_clean_loo[label] = support_radius(support[label])

    validation_bank = validation["sessions"].astype(int) == int(s_bank)
    validation_eval = validation["sessions"].astype(int) == int(s_eval)
    probe = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=c.stable_seed("stage0-class-probe", setting, fold, LAYER),
    )
    probe.fit(z_validation[validation_bank], validation["labels"][validation_bank])
    validation_prediction = probe.predict(z_validation[validation_eval])
    probe_ba = float(balanced_accuracy_score(validation["labels"][validation_eval], validation_prediction))
    if not np.isclose(probe_ba, float(v0_complete["independent_probe_BA"]), atol=1e-12, rtol=0.0):
        raise RuntimeError(f"{setting} fold={fold}: independent class probe differs from V0")

    unconditional: dict[str, np.ndarray] = {}
    global_subject_mean = np.stack(
        [np.mean([centroids[(subject, label, s_bank)] for label in labels], axis=0) for subject in subjects]
    )
    global_mean = global_subject_mean.mean(axis=0)
    for index, subject in enumerate(subjects):
        unconditional[subject] = global_subject_mean[index] - global_mean
    permutation = v0.derangement(subjects, c.stable_seed("stage0-permutation", setting, fold, LAYER))

    manifold_model: dict[int, NearestNeighbors] = {}
    manifold_threshold: dict[int, float] = {}
    for label in labels:
        real = np.stack(
            [centroids[(subject, label, s_eval)] for subject in subjects]
            + [validation_centroids[(subject, label, s_eval)] for subject in validation_subjects]
        )
        loo = NearestNeighbors(n_neighbors=min(4, len(real)), metric="euclidean").fit(real)
        loo_distance, _ = loo.kneighbors(real)
        manifold_threshold[label] = float(np.quantile(loo_distance[:, 1:].mean(axis=1), 0.95))
        manifold_model[label] = NearestNeighbors(n_neighbors=min(3, len(real)), metric="euclidean").fit(real)

    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    manifold_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    raw_alpha: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "centroid_alpha", "centroid_norm_ratio", "centroid_source_code", "centroid_target_code", "centroid_class",
            "trial_alpha", "trial_norm_ratio", "trial_source_code", "trial_target_code", "trial_class",
        )
    }

    for source_subject in subjects:
        target_subjects = [subject for subject in subjects if subject != source_subject]
        for label in labels:
            clean_centroid = centroids[(source_subject, label, s_eval)]
            clean_batch = np.broadcast_to(clean_centroid, (len(target_subjects), len(clean_centroid)))
            target_centroids = np.stack([centroids[(target_subject, label, s_eval)] for target_subject in target_subjects])
            deltas = np.stack(
                [residual[(target_subject, label, s_bank)] - residual[(source_subject, label, s_bank)] for target_subject in target_subjects]
            )
            unconditional_deltas = np.stack(
                [unconditional[target_subject] - unconditional[source_subject] for target_subject in target_subjects]
            )
            wrong_deltas = np.stack(
                [residual[(target_subject, 1 - label, s_bank)] - residual[(source_subject, 1 - label, s_bank)] for target_subject in target_subjects]
            )
            permuted_deltas = np.stack(
                [residual[(permutation[target_subject], label, s_bank)] - residual[(source_subject, label, s_bank)] for target_subject in target_subjects]
            )
            random_units: list[np.ndarray] = []
            for target_subject, delta in zip(target_subjects, deltas):
                rng = np.random.default_rng(
                    c.stable_seed("stage0-repair2-norm-random", setting, fold, LAYER, source_subject, target_subject, label)
                )
                direction = rng.normal(size=len(delta))
                direction /= max(np.linalg.norm(direction), 1e-12)
                random_units.append(direction)
            random_units_array = np.stack(random_units)

            centroid_alpha, centroid_source_clean, centroid_source_realized = solve_alpha(
                clean_batch, deltas, support[label], source_radius[label]
            )
            realized_norm = centroid_alpha * np.linalg.norm(deltas, axis=1)
            centroid_norm_ratio = realized_norm / np.maximum(np.linalg.norm(clean_batch, axis=1), 1e-12)
            centroid_transports = {
                "no_transport": clean_batch,
                "scst": clean_batch + centroid_alpha[:, None] * deltas,
                "norm_matched_random": clean_batch + realized_norm[:, None] * random_units_array,
                "unconditional_subject_transport": clean_batch + centroid_alpha[:, None] * unconditional_deltas,
                "wrong_class": clean_batch + centroid_alpha[:, None] * wrong_deltas,
                "subject_permutation": clean_batch + centroid_alpha[:, None] * permuted_deltas,
                "same_class_mixup": 0.5 * clean_batch + 0.5 * target_centroids,
            }
            clean_target_distance = np.linalg.norm(clean_batch - target_centroids, axis=1)
            for method, transported in centroid_transports.items():
                distances = np.linalg.norm(transported - target_centroids, axis=1)
                independent_manifold_distance = v0.mean_knn(manifold_model[label], transported)
                perturbation_norm = np.linalg.norm(transported - clean_batch, axis=1)
                for index, target_subject in enumerate(target_subjects):
                    common = {
                        "setting_id": setting,
                        "fold": fold,
                        "layer": LAYER,
                        "source_subject": source_subject,
                        "target_subject": target_subject,
                        "class_label": label,
                        "method": method,
                        "alpha_star": float(centroid_alpha[index]) if method != "same_class_mixup" else np.nan,
                        "perturbation_norm": float(perturbation_norm[index]),
                    }
                    subject_rows.append(
                        {
                            **common,
                            "target_distance": float(distances[index]),
                            "clean_target_distance": float(clean_target_distance[index]),
                            "relative_target_affinity_improvement": float(
                                (clean_target_distance[index] - distances[index]) / max(clean_target_distance[index], 1e-12)
                            ),
                            "delta_norm": float(np.linalg.norm(deltas[index])),
                            "source_support_clean_distance": float(centroid_source_clean[index]),
                            "source_support_realized_distance": float(centroid_source_realized[index]) if method == "scst" else np.nan,
                            "source_support_radius": source_radius[label],
                        }
                    )
                    manifold_rows.append(
                        {
                            **common,
                            "knn_distance": float(independent_manifold_distance[index]),
                            "off_manifold": bool(independent_manifold_distance[index] > manifold_threshold[label]),
                            "real_support_q95": manifold_threshold[label],
                        }
                    )

            source_trial_mask = (
                (source["subjects"].astype(str) == source_subject)
                & (source["sessions"].astype(int) == int(s_eval))
                & (source["labels"].astype(int) == int(label))
            )
            trial = z_source[source_trial_mask]
            clean_probability, clean_prediction = v0.class_probability(probe, trial, label)
            clean_logp = np.log(np.clip(clean_probability, 1e-12, 1.0))
            clean_accuracy = float(np.mean(clean_prediction == label))
            trial_batch = np.broadcast_to(trial[None, :, :], (len(target_subjects), len(trial), trial.shape[1]))
            trial_delta = np.broadcast_to(deltas[:, None, :], trial_batch.shape)
            flat_alpha, _, _ = solve_alpha(
                trial_batch.reshape(-1, trial.shape[1]),
                trial_delta.reshape(-1, trial.shape[1]),
                support[label],
                source_radius[label],
            )
            trial_alpha = flat_alpha.reshape(len(target_subjects), len(trial))
            delta_norm = np.linalg.norm(deltas, axis=1)
            trial_realized_norm = trial_alpha * delta_norm[:, None]
            trial_norm_ratio = trial_realized_norm / np.maximum(np.linalg.norm(trial_batch, axis=2), 1e-12)
            class_transports = {
                "no_transport": trial_batch,
                "scst": trial_batch + trial_alpha[:, :, None] * deltas[:, None, :],
                "norm_matched_random": trial_batch + trial_realized_norm[:, :, None] * random_units_array[:, None, :],
                "unconditional_subject_transport": trial_batch + trial_alpha[:, :, None] * unconditional_deltas[:, None, :],
                "same_class_mixup": 0.5 * trial_batch + 0.5 * target_centroids[:, None, :],
            }
            for method, transported in class_transports.items():
                probability, prediction = v0.class_probability(
                    probe, transported.reshape(-1, transported.shape[-1]), label
                )
                probability = probability.reshape(len(target_subjects), len(trial))
                prediction = prediction.reshape(len(target_subjects), len(trial))
                perturbation_norm = np.linalg.norm(transported - trial_batch, axis=2)
                for index, target_subject in enumerate(target_subjects):
                    transported_accuracy = float(np.mean(prediction[index] == label))
                    class_rows.append(
                        {
                            "setting_id": setting,
                            "fold": fold,
                            "layer": LAYER,
                            "source_subject": source_subject,
                            "target_subject": target_subject,
                            "class_label": label,
                            "method": method,
                            "independent_probe_BA": probe_ba,
                            "clean_accuracy": clean_accuracy,
                            "transported_accuracy": transported_accuracy,
                            "accuracy_change": transported_accuracy - clean_accuracy,
                            "clean_true_probability": float(clean_probability.mean()),
                            "transported_true_probability": float(probability[index].mean()),
                            "true_log_probability_change": float(
                                np.mean(np.log(np.clip(probability[index], 1e-12, 1.0)) - clean_logp)
                            ),
                            "trial_count": int(len(trial)),
                            "alpha_star_mean": float(trial_alpha[index].mean()) if method != "same_class_mixup" else np.nan,
                            "mean_perturbation_norm": float(perturbation_norm[index].mean()),
                        }
                    )

            for index, target_subject in enumerate(target_subjects):
                alpha_rows.append(
                    alpha_summary(
                        np.asarray([centroid_alpha[index]]),
                        np.asarray([centroid_norm_ratio[index]]),
                        setting,
                        fold,
                        source_subject,
                        target_subject,
                        label,
                        "centroid",
                    )
                )
                alpha_rows.append(
                    alpha_summary(
                        trial_alpha[index],
                        trial_norm_ratio[index],
                        setting,
                        fold,
                        source_subject,
                        target_subject,
                        label,
                        "trial",
                    )
                )

            target_codes = np.asarray([subject_code[value] for value in target_subjects], dtype=np.uint16)
            raw_alpha["centroid_alpha"].append(centroid_alpha.astype(np.float32))
            raw_alpha["centroid_norm_ratio"].append(centroid_norm_ratio.astype(np.float32))
            raw_alpha["centroid_source_code"].append(np.full(len(target_subjects), subject_code[source_subject], dtype=np.uint16))
            raw_alpha["centroid_target_code"].append(target_codes)
            raw_alpha["centroid_class"].append(np.full(len(target_subjects), label, dtype=np.int8))
            raw_alpha["trial_alpha"].append(trial_alpha.reshape(-1).astype(np.float32))
            raw_alpha["trial_norm_ratio"].append(trial_norm_ratio.reshape(-1).astype(np.float32))
            raw_alpha["trial_source_code"].append(np.full(trial_alpha.size, subject_code[source_subject], dtype=np.uint16))
            raw_alpha["trial_target_code"].append(np.repeat(target_codes, len(trial)).astype(np.uint16))
            raw_alpha["trial_class"].append(np.full(trial_alpha.size, label, dtype=np.int8))

    outputs = {
        "SUBJECT_FIDELITY.csv": pd.DataFrame(subject_rows),
        "CLASS_FIDELITY.csv": pd.DataFrame(class_rows),
        "MANIFOLD_VALIDITY.csv": pd.DataFrame(manifold_rows),
        "ALPHA_DISTRIBUTION.csv": pd.DataFrame(alpha_rows),
    }
    for name, frame in outputs.items():
        c.write_csv(target / name, frame)
    npz_path = target / "ALPHA_VALUES.npz"
    write_npz_atomic(
        npz_path,
        subject_values=np.asarray(subjects),
        **{key: np.concatenate(parts) for key, parts in raw_alpha.items()},
    )
    c.write_json(
        complete_path,
        {
            "schema": "SCST_DR_STAGE0_REPAIR2_UNIT_V1",
            "pass": True,
            "setting_id": setting,
            "fold": fold,
            "layer": LAYER,
            "source_subject_count": len(subjects),
            "validation_subject_count": len(validation_subjects),
            "representation_dim": h_source.shape[1],
            "source_rows": len(h_source),
            "validation_rows": len(h_validation),
            "bank_session": s_bank,
            "evaluation_session": s_eval,
            "independent_probe_BA": probe_ba,
            "alpha_grid": ALPHA_GRID.tolist(),
            "alpha_max": ALPHA_MAX,
            "support_radius": {str(label): source_radius[label] for label in labels},
            "support_sha256": {str(label): c.array_sha256(support[label]) for label in labels},
            "support_clean_loo_sha256": {str(label): c.array_sha256(source_clean_loo[label]) for label in labels},
            "outcome_rows_loaded": 0,
            "future_session_rows_loaded": 0,
            **current_hashes,
            "v0_unit_complete_sha256": c.sha256(v0_complete_path),
            "output_sha256": {
                **{name: c.sha256(target / name) for name in outputs},
                "ALPHA_VALUES.npz": c.sha256(npz_path),
            },
        },
    )
    print(
        f"[complete-repair2] {setting} fold={fold} probeBA={probe_ba:.4f} "
        f"alpha_centroid_mean={np.mean(np.concatenate(raw_alpha['centroid_alpha'])):.4f}",
        flush=True,
    )


def run_setting(setting: str, folds: list[int]) -> None:
    data = c.load_setting_data(setting)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-0 Repair 2 must run on the server GPU")
    raw = torch.from_numpy(np.asarray(data.x)).to(device=device)
    print(f"[{setting}] data={tuple(raw.shape)} gpu={torch.cuda.get_device_name(0)}", flush=True)
    for fold in folds:
        roles = c.fold_roles(setting, fold)
        source_indices = c.row_indices(data.metadata, roles["model_fit"], data.source_sessions)
        validation_indices = c.row_indices(data.metadata, roles["validation"], data.source_sessions)
        outcome_indices = c.row_indices(data.metadata, roles["outcome"], data.source_sessions)
        if set(source_indices) & set(validation_indices) or set(source_indices) & set(outcome_indices):
            raise RuntimeError(f"{setting} fold={fold}: role row overlap")
        model, _ = c.load_model(setting, fold, device, seed=0)
        mean, std = c.load_normalizer(setting, fold, device, seed=0)
        source = c.extract_layers(model, raw, data.metadata, source_indices, mean, std)
        validation = c.extract_layers(model, raw, data.metadata, validation_indices, mean, std)
        run_unit(setting, fold, source, validation, data.source_sessions)
        del model, source, validation
        gc.collect()
        torch.cuda.empty_cache()
    del raw
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", choices=list(c.SETTINGS), default=list(c.SETTINGS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    verify_repair2_freeze()
    lock = c.read_json(c.EXP / "protocol" / "STAGE0_REPAIR2_PROTOCOL_LOCK.json")
    if any(fold not in lock["folds"] for fold in args.folds):
        raise RuntimeError("requested fold differs from Repair-2 lock")
    for setting in args.settings:
        run_setting(setting, args.folds)
    print("SCST_DR_STAGE0_REPAIR2_RAW_METRICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
