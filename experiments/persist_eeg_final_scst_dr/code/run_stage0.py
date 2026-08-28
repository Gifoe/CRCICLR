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
    # Batch all target-subject queries for each source-subject/class cell.  This
    # is numerically equivalent to the original pair loop but avoids thousands
    # of one-row sklearn calls.
    for source_subject in subjects:
        target_subjects = [subject for subject in subjects if subject != source_subject]
        for label in labels:
            clean_centroid = centroids[(source_subject, label, s_eval)]
            target_centroids = np.stack([centroids[(target_subject, label, s_eval)] for target_subject in target_subjects])
            deltas = np.stack([
                residual[(target_subject, label, s_bank)] - residual[(source_subject, label, s_bank)]
                for target_subject in target_subjects
            ])
            unconditional_deltas = np.stack([
                unconditional[target_subject] - unconditional[source_subject] for target_subject in target_subjects
            ])
            wrong_deltas = np.stack([
                residual[(target_subject, 1 - label, s_bank)] - residual[(source_subject, 1 - label, s_bank)]
                for target_subject in target_subjects
            ])
            permuted_deltas = np.stack([
                residual[(permutation[target_subject], label, s_bank)] - residual[(source_subject, label, s_bank)]
                for target_subject in target_subjects
            ])
            random_deltas = []
            for target_subject, delta in zip(target_subjects, deltas):
                rng = np.random.default_rng(c.stable_seed("stage0-norm-random", setting, fold, layer, source_subject, target_subject, label))
                random_delta = rng.normal(size=len(delta))
                random_delta *= np.linalg.norm(delta) / max(np.linalg.norm(random_delta), 1e-12)
                random_deltas.append(random_delta)
            random_deltas = np.stack(random_deltas)
            clean_batch = np.broadcast_to(clean_centroid, target_centroids.shape)
            centroid_transports = {
                "no_transport": clean_batch,
                "scst": clean_batch + deltas,
                "norm_matched_random": clean_batch + random_deltas,
                "unconditional_subject_transport": clean_batch + unconditional_deltas,
                "wrong_class": clean_batch + wrong_deltas,
                "subject_permutation": clean_batch + permuted_deltas,
                "same_class_mixup": 0.5 * clean_batch + 0.5 * target_centroids,
            }
            clean_distances = np.linalg.norm(clean_batch - target_centroids, axis=1)
            delta_norms = np.linalg.norm(deltas, axis=1)
            for method, transported in centroid_transports.items():
                distances = np.linalg.norm(transported - target_centroids, axis=1)
                manifold_distances = mean_knn(manifold_model[label], transported)
                for index, target_subject in enumerate(target_subjects):
                    subject_rows.append({
                        "setting_id": setting, "fold": fold, "layer": layer,
                        "source_subject": source_subject, "target_subject": target_subject,
                        "class_label": label, "method": method,
                        "target_distance": float(distances[index]),
                        "clean_target_distance": float(clean_distances[index]),
                        "relative_target_affinity_improvement": float((clean_distances[index] - distances[index]) / max(clean_distances[index], 1e-12)),
                        "delta_norm": float(delta_norms[index]),
                    })
                    manifold_rows.append({
                        "setting_id": setting, "fold": fold, "layer": layer,
                        "source_subject": source_subject, "target_subject": target_subject,
                        "class_label": label, "method": method,
                        "knn_distance": float(manifold_distances[index]),
                        "off_manifold": bool(manifold_distances[index] > manifold_threshold[label]),
                        "real_support_q95": manifold_threshold[label],
                    })

            source_trial_mask = (
                (source["subjects"].astype(str) == source_subject)
                & (source["sessions"].astype(int) == int(s_eval))
                & (source["labels"].astype(int) == int(label))
            )
            trial = z_source[source_trial_mask]
            clean_probability, clean_prediction = class_probability(probe, trial, label)
            clean_logp = np.log(np.clip(clean_probability, 1e-12, 1.0))
            trial_batch = np.broadcast_to(trial[None, :, :], (len(target_subjects), len(trial), trial.shape[1]))
            class_transports = {
                "no_transport": trial_batch,
                "scst": trial_batch + deltas[:, None, :],
                "norm_matched_random": trial_batch + random_deltas[:, None, :],
                "unconditional_subject_transport": trial_batch + unconditional_deltas[:, None, :],
                "same_class_mixup": 0.5 * trial_batch + 0.5 * target_centroids[:, None, :],
            }
            for method, transported in class_transports.items():
                probability, prediction = class_probability(probe, transported.reshape(-1, transported.shape[-1]), label)
                probability = probability.reshape(len(target_subjects), len(trial))
                prediction = prediction.reshape(len(target_subjects), len(trial))
                for index, target_subject in enumerate(target_subjects):
                    transported_accuracy = float(np.mean(prediction[index] == label))
                    class_rows.append({
                        "setting_id": setting, "fold": fold, "layer": layer,
                        "source_subject": source_subject, "target_subject": target_subject,
                        "class_label": label, "method": method,
                        "independent_probe_BA": probe_ba,
                        "clean_accuracy": float(np.mean(clean_prediction == label)),
                        "transported_accuracy": transported_accuracy,
                        "accuracy_change": transported_accuracy - float(np.mean(clean_prediction == label)),
                        "clean_true_probability": float(clean_probability.mean()),
                        "transported_true_probability": float(probability[index].mean()),
                        "true_log_probability_change": float(np.mean(np.log(np.clip(probability[index], 1e-12, 1.0)) - clean_logp)),
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
