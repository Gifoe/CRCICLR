from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

import common as c


LAYERS = ("pre_embedding", "final_embedding")


def centroid_map(z: np.ndarray, subjects: np.ndarray, labels: np.ndarray, sessions: np.ndarray) -> dict[tuple[str, int, int], np.ndarray]:
    result: dict[tuple[str, int, int], np.ndarray] = {}
    for subject in c.subject_sort(np.unique(subjects.astype(str))):
        for session in sorted(map(int, np.unique(sessions))):
            for label in sorted(map(int, np.unique(labels))):
                mask = (subjects.astype(str) == subject) & (sessions.astype(int) == session) & (labels.astype(int) == label)
                if not np.any(mask):
                    raise RuntimeError(f"missing centroid subject={subject} session={session} class={label}")
                result[(subject, label, session)] = z[mask].mean(axis=0)
    return result


def population_residuals(
    centroids: dict[tuple[str, int, int], np.ndarray],
    subjects: list[str],
    labels: list[int],
    sessions: tuple[int, int],
) -> tuple[dict[tuple[str, int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    population: dict[tuple[int, int], np.ndarray] = {}
    residual: dict[tuple[str, int, int], np.ndarray] = {}
    for session in sessions:
        for label in labels:
            population[(label, session)] = np.stack([centroids[(s, label, session)] for s in subjects]).mean(axis=0)
            for subject in subjects:
                residual[(subject, label, session)] = centroids[(subject, label, session)] - population[(label, session)]
    return residual, population


def derangement(values: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    base = np.asarray(values, dtype=object)
    for _ in range(1000):
        candidate = rng.permutation(base)
        if np.all(candidate != base):
            return dict(zip(values, map(str, candidate)))
    return {value: values[(index + 1) % len(values)] for index, value in enumerate(values)}


def unit_output(setting: str, fold: int, layer: str) -> Path:
    return c.RUNTIME / "stage0_units" / setting / f"fold-{fold}" / layer


def class_probability(probe: LogisticRegression, x: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray]:
    probability = probe.predict_proba(x)
    class_index = int(np.flatnonzero(probe.classes_.astype(int) == int(label))[0])
    return probability[:, class_index], probe.predict(x).astype(int)


def mean_knn(model: NearestNeighbors, query: np.ndarray) -> np.ndarray:
    distance, _ = model.kneighbors(np.asarray(query, dtype=np.float64), return_distance=True)
    return distance.mean(axis=1)


def run_layer(
    setting: str,
    fold: int,
    layer: str,
    source: dict[str, Any],
    validation: dict[str, Any],
    source_sessions: tuple[int, int],
) -> None:
    target = unit_output(setting, fold, layer)
    complete = target / "UNIT_COMPLETE.json"
    if complete.is_file() and c.read_json(complete).get("pass") is True:
        print(f"[cached] {setting} fold={fold} layer={layer}", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    h_source = np.asarray(source["layers"][layer], dtype=np.float64)
    h_validation = np.asarray(validation["layers"][layer], dtype=np.float64)
    labels = sorted(map(int, np.unique(source["labels"])))
    s_bank, s_eval = source_sessions
    bank_mask = source["sessions"].astype(int) == int(s_bank)
    center = h_source[bank_mask].mean(axis=0)
    scale = h_source[bank_mask].std(axis=0)
    scale[scale < 1e-6] = 1.0
    z_source = (h_source - center) / scale
    z_validation = (h_validation - center) / scale

    subjects = c.subject_sort(np.unique(source["subjects"].astype(str)))
    validation_subjects = c.subject_sort(np.unique(validation["subjects"].astype(str)))
    centroids = centroid_map(z_source, source["subjects"], source["labels"], source["sessions"])
    validation_centroids = centroid_map(z_validation, validation["subjects"], validation["labels"], validation["sessions"])
    residual, _ = population_residuals(centroids, subjects, labels, source_sessions)

    permutation = derangement(subjects, c.stable_seed("stage0-permutation", setting, fold, layer))
    stability_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for label in labels:
            base = residual[(subject, label, s_bank)]
            comparisons = {
                "matched_subject_same_class": residual[(subject, label, s_eval)],
                "wrong_class": residual[(subject, 1 - label, s_eval)],
                "subject_permutation": residual[(permutation[subject], label, s_eval)],
            }
            rng = np.random.default_rng(c.stable_seed("stage0-random-stability", setting, fold, layer, subject, label))
            random_value = rng.normal(size=len(base))
            random_value *= np.linalg.norm(residual[(subject, label, s_eval)]) / max(np.linalg.norm(random_value), 1e-12)
            comparisons["norm_matched_random"] = random_value
            for control, other in comparisons.items():
                stability_rows.append({
                    "setting_id": setting, "fold": fold, "layer": layer, "source_subject": subject,
                    "target_subject": subject if control in {"matched_subject_same_class", "wrong_class"} else permutation[subject] if control == "subject_permutation" else "RANDOM",
                    "class_label": label, "control": control, "cosine": c.cosine(base, other),
                })
            for target_subject in subjects:
                if target_subject == subject:
                    continue
                stability_rows.append({
                    "setting_id": setting, "fold": fold, "layer": layer, "source_subject": subject,
                    "target_subject": target_subject, "class_label": label,
                    "control": "mismatched_subject_same_class",
                    "cosine": c.cosine(base, residual[(target_subject, label, s_eval)]),
                })

    validation_bank = validation["sessions"].astype(int) == int(s_bank)
    validation_eval = validation["sessions"].astype(int) == int(s_eval)
    probe = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000,
        random_state=c.stable_seed("stage0-class-probe", setting, fold, layer),
    )
    probe.fit(z_validation[validation_bank], validation["labels"][validation_bank])
    validation_prediction = probe.predict(z_validation[validation_eval])
    probe_ba = float(balanced_accuracy_score(validation["labels"][validation_eval], validation_prediction))

    unconditional: dict[str, np.ndarray] = {}
    global_subject_mean = np.stack([
        np.mean([centroids[(subject, label, s_bank)] for label in labels], axis=0) for subject in subjects
    ])
    global_mean = global_subject_mean.mean(axis=0)
    for index, subject in enumerate(subjects):
        unconditional[subject] = global_subject_mean[index] - global_mean

    real_centroids_by_class: dict[int, np.ndarray] = {}
    manifold_model: dict[int, NearestNeighbors] = {}
    manifold_threshold: dict[int, float] = {}
    for label in labels:
        real = np.stack(
            [centroids[(subject, label, s_eval)] for subject in subjects]
            + [validation_centroids[(subject, label, s_eval)] for subject in validation_subjects]
        )
        real_centroids_by_class[label] = real
        neighbors = min(4, len(real))
        loo = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(real)
        loo_distance, _ = loo.kneighbors(real)
        real_score = loo_distance[:, 1:].mean(axis=1)
        manifold_threshold[label] = float(np.quantile(real_score, 0.95))
        manifold_model[label] = NearestNeighbors(n_neighbors=min(3, len(real)), metric="euclidean").fit(real)

    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    manifold_rows: list[dict[str, Any]] = []
    for source_subject in subjects:
        for target_subject in subjects:
            if source_subject == target_subject:
                continue
            for label in labels:
                clean_centroid = centroids[(source_subject, label, s_eval)]
                target_centroid = centroids[(target_subject, label, s_eval)]
                delta = residual[(target_subject, label, s_bank)] - residual[(source_subject, label, s_bank)]
                delta_unconditional = unconditional[target_subject] - unconditional[source_subject]
                delta_wrong = residual[(target_subject, 1 - label, s_bank)] - residual[(source_subject, 1 - label, s_bank)]
                permuted_target = permutation[target_subject]
                delta_permuted = residual[(permuted_target, label, s_bank)] - residual[(source_subject, label, s_bank)]
                rng = np.random.default_rng(c.stable_seed("stage0-norm-random", setting, fold, layer, source_subject, target_subject, label))
                random_delta = rng.normal(size=len(delta))
                random_delta *= np.linalg.norm(delta) / max(np.linalg.norm(random_delta), 1e-12)
                transports = {
                    "no_transport": clean_centroid,
                    "scst": clean_centroid + delta,
                    "norm_matched_random": clean_centroid + random_delta,
                    "unconditional_subject_transport": clean_centroid + delta_unconditional,
                    "wrong_class": clean_centroid + delta_wrong,
                    "subject_permutation": clean_centroid + delta_permuted,
                    "same_class_mixup": 0.5 * clean_centroid + 0.5 * target_centroid,
                }
                clean_distance = float(np.linalg.norm(clean_centroid - target_centroid))
                for method, transported in transports.items():
                    distance = float(np.linalg.norm(transported - target_centroid))
                    subject_rows.append({
                        "setting_id": setting, "fold": fold, "layer": layer,
                        "source_subject": source_subject, "target_subject": target_subject,
                        "class_label": label, "method": method,
                        "target_distance": distance,
                        "clean_target_distance": clean_distance,
                        "relative_target_affinity_improvement": (clean_distance - distance) / max(clean_distance, 1e-12),
                        "delta_norm": float(np.linalg.norm(delta)),
                    })
                    manifold_distance = float(mean_knn(manifold_model[label], transported[None, :])[0])
                    manifold_rows.append({
                        "setting_id": setting, "fold": fold, "layer": layer,
                        "source_subject": source_subject, "target_subject": target_subject,
                        "class_label": label, "method": method,
                        "knn_distance": manifold_distance,
                        "off_manifold": bool(manifold_distance > manifold_threshold[label]),
                        "real_support_q95": manifold_threshold[label],
                    })

                source_trial_mask = (
                    (source["subjects"].astype(str) == source_subject)
                    & (source["sessions"].astype(int) == int(s_eval))
                    & (source["labels"].astype(int) == int(label))
                )
                trial = z_source[source_trial_mask]
                class_transports = {
                    "no_transport": trial,
                    "scst": trial + delta,
                    "norm_matched_random": trial + random_delta,
                    "unconditional_subject_transport": trial + delta_unconditional,
                    "same_class_mixup": 0.5 * trial + 0.5 * target_centroid,
                }
                clean_probability, clean_prediction = class_probability(probe, trial, label)
                clean_logp = np.log(np.clip(clean_probability, 1e-12, 1.0))
                for method, transported in class_transports.items():
                    probability, prediction = class_probability(probe, transported, label)
                    class_rows.append({
                        "setting_id": setting, "fold": fold, "layer": layer,
                        "source_subject": source_subject, "target_subject": target_subject,
                        "class_label": label, "method": method,
                        "independent_probe_BA": probe_ba,
                        "clean_accuracy": float(np.mean(clean_prediction == label)),
                        "transported_accuracy": float(np.mean(prediction == label)),
                        "accuracy_change": float(np.mean(prediction == label) - np.mean(clean_prediction == label)),
                        "clean_true_probability": float(clean_probability.mean()),
                        "transported_true_probability": float(probability.mean()),
                        "true_log_probability_change": float(np.mean(np.log(np.clip(probability, 1e-12, 1.0)) - clean_logp)),
                        "trial_count": int(len(trial)),
                    })

    c.write_csv(target / "TRANSPORT_STABILITY.csv", pd.DataFrame(stability_rows))
    c.write_csv(target / "SUBJECT_FIDELITY.csv", pd.DataFrame(subject_rows))
    c.write_csv(target / "CLASS_FIDELITY.csv", pd.DataFrame(class_rows))
    c.write_csv(target / "MANIFOLD_VALIDITY.csv", pd.DataFrame(manifold_rows))
    c.write_json(target / "UNIT_COMPLETE.json", {
        "schema": "SCST_DR_STAGE0_UNIT_V1", "pass": True, "setting_id": setting,
        "fold": fold, "layer": layer, "source_subject_count": len(subjects),
        "validation_subject_count": len(validation_subjects), "representation_dim": h_source.shape[1],
        "independent_probe_BA": probe_ba, "bank_session": s_bank, "evaluation_session": s_eval,
        "source_rows": len(h_source), "validation_rows": len(h_validation),
        "outcome_rows_loaded": 0, "future_session_rows_loaded": 0,
        "feature_scope_sha256": c.array_sha256(source["indices"]),
        "scaling_center_sha256": c.array_sha256(center), "scaling_scale_sha256": c.array_sha256(scale),
    })
    print(f"[complete] {setting} fold={fold} layer={layer} probeBA={probe_ba:.4f}", flush=True)


def run_setting(setting: str, folds: list[int]) -> None:
    data = c.load_setting_data(setting)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage 0 must run on the server GPU")
    raw = torch.from_numpy(np.asarray(data.x)).to(device=device)
    print(f"[{setting}] data={tuple(raw.shape)} dtype={raw.dtype} gpu={torch.cuda.get_device_name(0)}", flush=True)
    for fold in folds:
        roles = c.fold_roles(setting, fold)
        source_indices = c.row_indices(data.metadata, roles["model_fit"], data.source_sessions)
        validation_indices = c.row_indices(data.metadata, roles["validation"], data.source_sessions)
        outcome_indices = c.row_indices(data.metadata, roles["outcome"], data.source_sessions)
        if set(source_indices) & set(validation_indices) or set(source_indices) & set(outcome_indices):
            raise RuntimeError(f"{setting} fold={fold}: role row overlap")
        # The outcome indices are constructed only as a role-overlap guard and
        # are never dereferenced. Future-session rows are never indexed.
        model, unit = c.load_model(setting, fold, device, seed=0)
        mean, std = c.load_normalizer(setting, fold, device, seed=0)
        source = c.extract_layers(model, raw, data.metadata, source_indices, mean, std)
        validation = c.extract_layers(model, raw, data.metadata, validation_indices, mean, std)
        for layer in LAYERS:
            run_layer(setting, fold, layer, source, validation, data.source_sessions)
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
    protocol = c.protocol()
    freeze = c.EXP / "protocol" / "PRE_STAGE0_FREEZE.json"
    if not freeze.is_file() or c.read_json(freeze).get("pass") is not True:
        raise RuntimeError("Stage 0 blocked until PRE_STAGE0_FREEZE exists")
    if set(args.settings) != set(protocol["development_settings"]) and args.settings == list(c.SETTINGS):
        raise RuntimeError("requested settings differ from frozen protocol")
    for setting in args.settings:
        run_setting(setting, args.folds)
    print("SCST_DR_STAGE0_RAW_METRICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
