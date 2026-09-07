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
import run_stage0 as v0


LAYERS = ("pre_embedding", "final_embedding")
ALPHAS = (0.25, 0.50)


def alpha_method(alpha: float) -> str:
    return f"scst_alpha_{alpha:.2f}"


def unit_output(setting: str, fold: int, layer: str) -> Path:
    return c.RUNTIME / "stage0_repair1_units" / setting / f"fold-{fold}" / layer


def verify_repair_freeze() -> dict[str, Any]:
    freeze_path = c.EXP / "protocol" / "PRE_STAGE0_REPAIR1_FREEZE.json"
    if not freeze_path.is_file():
        raise RuntimeError("Repair 1 is blocked until PRE_STAGE0_REPAIR1_FREEZE.json exists")
    freeze = c.read_json(freeze_path)
    if freeze.get("pass") is not True or freeze.get("frozen_before_repair1_metrics") is not True:
        raise RuntimeError("invalid Repair-1 freeze")
    for relative, expected in freeze.get("file_sha256", {}).items():
        path = c.EXP / relative
        if not path.is_file() or c.sha256(path) != expected:
            raise RuntimeError(f"post-freeze Repair-1 input changed: {relative}")
    lock = c.read_json(c.EXP / "protocol" / "STAGE0_REPAIR1_PROTOCOL_LOCK.json")
    if tuple(map(float, lock["repair"]["global_alpha_candidates"])) != ALPHAS:
        raise RuntimeError("Repair-1 alpha candidates differ from the frozen lock")
    if lock["future_or_outer_performance_access_allowed"] is not False:
        raise RuntimeError("Repair-1 lock does not forbid future/outer performance access")
    return freeze


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
        print(f"[cached-repair1] {setting} fold={fold} layer={layer}", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)

    v0_dir = c.RUNTIME / "stage0_units" / setting / f"fold-{fold}" / layer
    v0_complete_path = v0_dir / "UNIT_COMPLETE.json"
    if not v0_complete_path.is_file():
        raise RuntimeError(f"missing validated V0 unit: {v0_complete_path}")
    v0_complete = c.read_json(v0_complete_path)
    if v0_complete.get("pass") is not True:
        raise RuntimeError(f"V0 unit is not complete: {v0_complete_path}")

    h_source = np.asarray(source["layers"][layer], dtype=np.float64)
    h_validation = np.asarray(validation["layers"][layer], dtype=np.float64)
    labels = sorted(map(int, np.unique(source["labels"])))
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
            raise RuntimeError(f"{setting} fold={fold} layer={layer}: Repair-1 {key} differs from V0")

    z_source = (h_source - center) / scale
    z_validation = (h_validation - center) / scale
    subjects = c.subject_sort(np.unique(source["subjects"].astype(str)))
    validation_subjects = c.subject_sort(np.unique(validation["subjects"].astype(str)))
    centroids = v0.centroid_map(z_source, source["subjects"], source["labels"], source["sessions"])
    validation_centroids = v0.centroid_map(
        z_validation, validation["subjects"], validation["labels"], validation["sessions"]
    )
    residual, _ = v0.population_residuals(centroids, subjects, labels, source_sessions)

    validation_bank = validation["sessions"].astype(int) == int(s_bank)
    validation_eval = validation["sessions"].astype(int) == int(s_eval)
    probe = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=c.stable_seed("stage0-class-probe", setting, fold, layer),
    )
    probe.fit(z_validation[validation_bank], validation["labels"][validation_bank])
    validation_prediction = probe.predict(z_validation[validation_eval])
    probe_ba = float(balanced_accuracy_score(validation["labels"][validation_eval], validation_prediction))
    if not np.isclose(probe_ba, float(v0_complete["independent_probe_BA"]), atol=1e-12, rtol=0.0):
        raise RuntimeError(f"{setting} fold={fold} layer={layer}: independent probe differs from V0")

    manifold_model: dict[int, NearestNeighbors] = {}
    manifold_threshold: dict[int, float] = {}
    for label in labels:
        real = np.stack(
            [centroids[(subject, label, s_eval)] for subject in subjects]
            + [validation_centroids[(subject, label, s_eval)] for subject in validation_subjects]
        )
        neighbors = min(4, len(real))
        loo = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(real)
        loo_distance, _ = loo.kneighbors(real)
        real_score = loo_distance[:, 1:].mean(axis=1)
        manifold_threshold[label] = float(np.quantile(real_score, 0.95))
        manifold_model[label] = NearestNeighbors(
            n_neighbors=min(3, len(real)), metric="euclidean"
        ).fit(real)

    subject_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    manifold_rows: list[dict[str, Any]] = []
    for source_subject in subjects:
        target_subjects = [subject for subject in subjects if subject != source_subject]
        for label in labels:
            clean_centroid = centroids[(source_subject, label, s_eval)]
            target_centroids = np.stack(
                [centroids[(target_subject, label, s_eval)] for target_subject in target_subjects]
            )
            deltas = np.stack(
                [
                    residual[(target_subject, label, s_bank)]
                    - residual[(source_subject, label, s_bank)]
                    for target_subject in target_subjects
                ]
            )
            clean_batch = np.broadcast_to(clean_centroid, target_centroids.shape)
            clean_distances = np.linalg.norm(clean_batch - target_centroids, axis=1)
            delta_norms = np.linalg.norm(deltas, axis=1)

            source_trial_mask = (
                (source["subjects"].astype(str) == source_subject)
                & (source["sessions"].astype(int) == int(s_eval))
                & (source["labels"].astype(int) == int(label))
            )
            trial = z_source[source_trial_mask]
            clean_probability, clean_prediction = v0.class_probability(probe, trial, label)
            clean_logp = np.log(np.clip(clean_probability, 1e-12, 1.0))
            clean_accuracy = float(np.mean(clean_prediction == label))
            trial_batch = np.broadcast_to(
                trial[None, :, :], (len(target_subjects), len(trial), trial.shape[1])
            )

            for alpha in ALPHAS:
                method = alpha_method(alpha)
                transported_centroids = clean_batch + alpha * deltas
                distances = np.linalg.norm(transported_centroids - target_centroids, axis=1)
                manifold_distances = v0.mean_knn(manifold_model[label], transported_centroids)
                transported_trials = trial_batch + alpha * deltas[:, None, :]
                probability, prediction = v0.class_probability(
                    probe,
                    transported_trials.reshape(-1, transported_trials.shape[-1]),
                    label,
                )
                probability = probability.reshape(len(target_subjects), len(trial))
                prediction = prediction.reshape(len(target_subjects), len(trial))

                for index, target_subject in enumerate(target_subjects):
                    common = {
                        "setting_id": setting,
                        "fold": fold,
                        "layer": layer,
                        "source_subject": source_subject,
                        "target_subject": target_subject,
                        "class_label": label,
                        "method": method,
                        "alpha": alpha,
                    }
                    subject_rows.append(
                        {
                            **common,
                            "target_distance": float(distances[index]),
                            "clean_target_distance": float(clean_distances[index]),
                            "relative_target_affinity_improvement": float(
                                (clean_distances[index] - distances[index])
                                / max(clean_distances[index], 1e-12)
                            ),
                            "delta_norm": float(delta_norms[index]),
                        }
                    )
                    manifold_rows.append(
                        {
                            **common,
                            "knn_distance": float(manifold_distances[index]),
                            "off_manifold": bool(
                                manifold_distances[index] > manifold_threshold[label]
                            ),
                            "real_support_q95": manifold_threshold[label],
                        }
                    )
                    transported_accuracy = float(np.mean(prediction[index] == label))
                    class_rows.append(
                        {
                            **common,
                            "independent_probe_BA": probe_ba,
                            "clean_accuracy": clean_accuracy,
                            "transported_accuracy": transported_accuracy,
                            "accuracy_change": transported_accuracy - clean_accuracy,
                            "clean_true_probability": float(clean_probability.mean()),
                            "transported_true_probability": float(probability[index].mean()),
                            "true_log_probability_change": float(
                                np.mean(
                                    np.log(np.clip(probability[index], 1e-12, 1.0))
                                    - clean_logp
                                )
                            ),
                            "trial_count": int(len(trial)),
                        }
                    )

    outputs = {
        "SUBJECT_FIDELITY.csv": pd.DataFrame(subject_rows),
        "CLASS_FIDELITY.csv": pd.DataFrame(class_rows),
        "MANIFOLD_VALIDITY.csv": pd.DataFrame(manifold_rows),
    }
    for name, frame in outputs.items():
        c.write_csv(target / name, frame)
    c.write_json(
        complete,
        {
            "schema": "SCST_DR_STAGE0_REPAIR1_UNIT_V1",
            "pass": True,
            "setting_id": setting,
            "fold": fold,
            "layer": layer,
            "alphas": list(ALPHAS),
            "methods": [alpha_method(alpha) for alpha in ALPHAS],
            "only_change_from_v0": "transport_magnitude",
            "source_subject_count": len(subjects),
            "validation_subject_count": len(validation_subjects),
            "representation_dim": h_source.shape[1],
            "independent_probe_BA": probe_ba,
            "bank_session": s_bank,
            "evaluation_session": s_eval,
            "source_rows": len(h_source),
            "validation_rows": len(h_validation),
            "outcome_rows_loaded": 0,
            "future_session_rows_loaded": 0,
            **current_hashes,
            "v0_unit_complete_sha256": c.sha256(v0_complete_path),
            "output_sha256": {name: c.sha256(target / name) for name in outputs},
        },
    )
    print(
        f"[complete-repair1] {setting} fold={fold} layer={layer} probeBA={probe_ba:.4f}",
        flush=True,
    )


def run_setting(setting: str, folds: list[int]) -> None:
    data = c.load_setting_data(setting)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage-0 Repair 1 must run on the server GPU")
    raw = torch.from_numpy(np.asarray(data.x)).to(device=device)
    print(
        f"[{setting}] data={tuple(raw.shape)} dtype={raw.dtype} gpu={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for fold in folds:
        roles = c.fold_roles(setting, fold)
        source_indices = c.row_indices(data.metadata, roles["model_fit"], data.source_sessions)
        validation_indices = c.row_indices(
            data.metadata, roles["validation"], data.source_sessions
        )
        outcome_indices = c.row_indices(data.metadata, roles["outcome"], data.source_sessions)
        if set(source_indices) & set(validation_indices) or set(source_indices) & set(
            outcome_indices
        ):
            raise RuntimeError(f"{setting} fold={fold}: role row overlap")
        model, _ = c.load_model(setting, fold, device, seed=0)
        mean, std = c.load_normalizer(setting, fold, device, seed=0)
        source = c.extract_layers(model, raw, data.metadata, source_indices, mean, std)
        validation = c.extract_layers(
            model, raw, data.metadata, validation_indices, mean, std
        )
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
    parser.add_argument(
        "--settings", nargs="+", choices=list(c.SETTINGS), default=list(c.SETTINGS)
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    verify_repair_freeze()
    protocol = c.protocol()
    if any(fold not in protocol["folds"] for fold in args.folds):
        raise RuntimeError("requested folds differ from the frozen protocol")
    if args.settings == list(c.SETTINGS) and set(args.settings) != set(
        protocol["development_settings"]
    ):
        raise RuntimeError("requested settings differ from the frozen protocol")
    for setting in args.settings:
        run_setting(setting, args.folds)
    print("SCST_DR_STAGE0_REPAIR1_RAW_METRICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
