from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

import stage1_common as c


ALPHA_GRID = np.arange(17, dtype=np.float64) / 64.0
EPS = 1e-12
MODELS = ("ATCNet-CleanRoom", "ATCNet-Official", "EEGNeX")


def source_rep_path(model: str, dataset: str, fold: int, seed: int, role: str) -> Path:
    if model == "ATCNet-CleanRoom":
        old = c.REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "runtime" / "specialist_representations" / "ATCNet"
        return old / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"
    return c.rep_path(model, dataset, fold, seed, role)


def load_rep(model: str, dataset: str, fold: int, seed: int, role: str) -> dict[str, np.ndarray]:
    path = source_rep_path(model, dataset, fold, seed, role)
    if not path.is_file():
        raise FileNotFoundError(path)
    return c.load_rep(path)


def centroids(rep: dict[str, np.ndarray]) -> dict[tuple[str, int, int], np.ndarray]:
    out: dict[tuple[str, int, int], np.ndarray] = {}
    subjects = c.subject_sort(np.unique(rep["subjects"].astype(str)))
    labels = sorted(np.unique(rep["labels"].astype(int)).tolist())
    sessions = sorted(np.unique(rep["sessions"].astype(int)).tolist())
    for subject in subjects:
        for label in labels:
            for session in sessions:
                mask = (
                    (rep["subjects"].astype(str) == subject)
                    & (rep["labels"].astype(int) == label)
                    & (rep["sessions"].astype(int) == session)
                )
                if mask.any():
                    out[(subject, label, session)] = rep["features"][mask].mean(axis=0).astype(np.float64)
    return out


def support_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    q = np.asarray(query, np.float64)
    s = np.asarray(support, np.float64)
    if len(s) < 3:
        raise RuntimeError("3NN support requires at least three source subjects")
    distance = np.sum(q * q, axis=1)[:, None] + np.sum(s * s, axis=1)[None, :] - 2.0 * q @ s.T
    np.maximum(distance, 0.0, out=distance)
    return np.sqrt(np.partition(distance, kth=2, axis=1)[:, :3]).mean(axis=1)


def support_radius(support: np.ndarray) -> float:
    value = np.asarray(support, np.float64)
    distance = np.linalg.norm(value[:, None] - value[None, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    clean = np.partition(distance, kth=2, axis=1)[:, :3].mean(axis=1)
    return float(np.quantile(clean, 0.95))


def solve_alpha(query: np.ndarray, delta: np.ndarray, support: np.ndarray, radius: float) -> np.ndarray:
    selected = np.zeros(len(query), dtype=np.float64)
    for alpha in ALPHA_GRID[1:]:
        selected[support_distance(query + alpha * delta, support) <= radius] = alpha
    return selected


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    arr = np.asarray(values, np.float64)
    point = float(np.mean(arr)) if len(arr) else float("nan")
    if len(arr) < 4 or not np.isfinite(arr).all():
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arr), size=(10_000, len(arr)))
    dist = arr[draws].mean(axis=1)
    return point, float(np.quantile(dist, 0.025)), float(np.quantile(dist, 0.975))


def task_row(model: str, dataset: str, fold: int, seed: int, outcome: dict[str, np.ndarray]) -> dict[str, object]:
    score = c.metrics(outcome["labels"], outcome["logits"], outcome["subjects"])
    return {
        "model": model,
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "BA": score["BA"],
        "macro_F1": score["macro_F1"],
        "NLL": score["NLL"],
        "threshold": c.THRESHOLDS[dataset],
        "competent": bool(score["BA"] >= c.THRESHOLDS[dataset]),
        "future_session_used": False,
    }


def audit_unit(model: str, dataset: str, fold: int, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    source = load_rep(model, dataset, fold, seed, "model_fit")
    validation = load_rep(model, dataset, fold, seed, "validation")
    outcome = load_rep(model, dataset, fold, seed, "outcome")
    s_bank, s_eval = c.SOURCE_SESSIONS[dataset]
    center = source["features"][source["sessions"].astype(int) == s_bank].mean(axis=0)
    scale = source["features"][source["sessions"].astype(int) == s_bank].std(axis=0)
    scale[scale < 1e-6] = 1.0
    source = {**source, "features": ((source["features"] - center) / scale).astype(np.float32)}
    validation = {**validation, "features": ((validation["features"] - center) / scale).astype(np.float32)}

    subjects = c.subject_sort(np.unique(source["subjects"].astype(str)))
    validation_subjects = c.subject_sort(np.unique(validation["subjects"].astype(str)))
    labels = sorted(np.unique(source["labels"].astype(int)).tolist())
    source_centroids = centroids(source)
    validation_centroids = centroids(validation)
    population = {
        (label, session): np.mean([source_centroids[(s, label, session)] for s in subjects], axis=0)
        for label in labels
        for session in (s_bank, s_eval)
    }
    residual = {
        (s, label, session): source_centroids[(s, label, session)] - population[(label, session)]
        for s in subjects
        for label in labels
        for session in (s_bank, s_eval)
    }

    stability_by_subject: dict[str, list[float]] = {s: [] for s in subjects}
    for subject in subjects:
        for label in labels:
            matched = cosine(residual[(subject, label, s_bank)], residual[(subject, label, s_eval)])
            mismatched = np.mean(
                [cosine(residual[(subject, label, s_bank)], residual[(other, label, s_eval)]) for other in subjects if other != subject]
            )
            stability_by_subject[subject].append(matched - mismatched)

    affinity_by_subject: dict[str, list[float]] = {s: [] for s in subjects}
    random_advantage_by_subject: dict[str, list[float]] = {s: [] for s in subjects}
    class_change_by_subject: dict[str, list[float]] = {s: [] for s in subjects}
    logprob_change_by_subject: dict[str, list[float]] = {s: [] for s in subjects}
    transported_knn: list[float] = []
    clean_knn: list[float] = []
    off_values: list[bool] = []
    random_off_values: list[bool] = []
    rng = np.random.default_rng(c.stable_seed("stage1-random", model, dataset, fold, seed))

    validation_bank = validation["sessions"].astype(int) == s_bank
    validation_eval = validation["sessions"].astype(int) == s_eval
    probe = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=c.stable_seed("stage1-probe", model, dataset, fold, seed),
    ).fit(validation["features"][validation_bank], validation["labels"][validation_bank])
    probe_ba = float(balanced_accuracy_score(validation["labels"][validation_eval], probe.predict(validation["features"][validation_eval])))

    for label in labels:
        support = np.stack([source_centroids[(s, label, s_bank)] for s in subjects])
        radius = support_radius(support)
        real = np.stack(
            [source_centroids[(s, label, s_eval)] for s in subjects]
            + [validation_centroids[(s, label, s_eval)] for s in validation_subjects]
        )
        knn = NearestNeighbors(n_neighbors=3).fit(real)
        loo = NearestNeighbors(n_neighbors=4).fit(real)
        leave_one_out = loo.kneighbors(real, return_distance=True)[0][:, 1:].mean(axis=1)
        off_threshold = float(np.quantile(leave_one_out, 0.95))
        for source_subject in subjects:
            targets = [s for s in subjects if s != source_subject]
            query = np.repeat(source_centroids[(source_subject, label, s_eval)][None, :], len(targets), axis=0)
            target_centroid = np.stack([source_centroids[(target, label, s_eval)] for target in targets])
            delta = np.stack(
                [residual[(target, label, s_bank)] - residual[(source_subject, label, s_bank)] for target in targets]
            )
            alpha = solve_alpha(query, delta, support, radius)
            transported = query + alpha[:, None] * delta
            random_delta = []
            for value in delta:
                random = rng.normal(size=len(value))
                random *= np.linalg.norm(value) / max(np.linalg.norm(random), EPS)
                random_delta.append(random)
            random_delta = np.asarray(random_delta)
            random_transport = query + alpha[:, None] * random_delta
            clean_distance = np.linalg.norm(query - target_centroid, axis=1)
            transported_distance = np.linalg.norm(transported - target_centroid, axis=1)
            random_distance = np.linalg.norm(random_transport - target_centroid, axis=1)
            relative = (clean_distance - transported_distance) / np.maximum(clean_distance, EPS)
            random_relative = (clean_distance - random_distance) / np.maximum(clean_distance, EPS)
            affinity_by_subject[source_subject].extend(relative.tolist())
            random_advantage_by_subject[source_subject].extend((relative - random_relative).tolist())
            transported_knn.extend(knn.kneighbors(transported, return_distance=True)[0].mean(axis=1).tolist())
            clean_knn.extend(knn.kneighbors(query, return_distance=True)[0].mean(axis=1).tolist())
            random_knn = knn.kneighbors(random_transport, return_distance=True)[0].mean(axis=1)
            off_values.extend((np.asarray(transported_knn[-len(targets) :]) > off_threshold).tolist())
            random_off_values.extend((random_knn > off_threshold).tolist())

            trials = np.flatnonzero(
                (source["subjects"].astype(str) == source_subject)
                & (source["labels"].astype(int) == label)
                & (source["sessions"].astype(int) == s_eval)
            )
            clean = source["features"][trials].astype(np.float64)
            clean_pred = probe.predict(clean)
            class_index = int(np.flatnonzero(probe.classes_.astype(int) == label)[0])
            clean_probability = probe.predict_proba(clean)[:, class_index]
            clean_accuracy = float(np.mean(clean_pred == label))
            clean_log_probability = np.log(np.clip(clean_probability, 1e-12, 1.0))
            for target_index, _target in enumerate(targets):
                trial_delta = np.repeat(delta[target_index][None, :], len(clean), axis=0)
                trial_alpha = solve_alpha(clean, trial_delta, support, radius)
                moved = clean + trial_alpha[:, None] * trial_delta
                prediction = probe.predict(moved)
                probability = probe.predict_proba(moved)[:, class_index]
                class_change_by_subject[source_subject].append(float(np.mean(prediction == label) - clean_accuracy))
                logprob_change_by_subject[source_subject].append(
                    float(np.mean(np.log(np.clip(probability, 1e-12, 1.0)) - clean_log_probability))
                )

    subject_rows: list[dict[str, object]] = []
    for subject in subjects:
        subject_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "fold": fold,
                "seed": seed,
                "source_subject": subject,
                "stability": float(np.mean(stability_by_subject[subject])),
                "affinity": float(np.mean(affinity_by_subject[subject])),
                "random_advantage": float(np.mean(random_advantage_by_subject[subject])),
                "class_accuracy_change": float(np.mean(class_change_by_subject[subject])),
                "class_true_log_probability_change": float(np.mean(logprob_change_by_subject[subject])),
            }
        )
    frame = pd.DataFrame(subject_rows)
    stability, stability_low, stability_high = bootstrap_ci(frame["stability"].to_numpy(), c.stable_seed("ci", model, dataset, fold, seed, "stability"))
    affinity, affinity_low, affinity_high = bootstrap_ci(frame["affinity"].to_numpy(), c.stable_seed("ci", model, dataset, fold, seed, "affinity"))
    advantage, advantage_low, advantage_high = bootstrap_ci(frame["random_advantage"].to_numpy(), c.stable_seed("ci", model, dataset, fold, seed, "random"))
    task = task_row(model, dataset, fold, seed, outcome)
    unit = {
        **task,
        "independent_probe_BA": probe_ba,
        "residual_stability": stability,
        "stability_CI_low": stability_low,
        "stability_CI_high": stability_high,
        "subject_fidelity": affinity,
        "subject_fidelity_CI_low": affinity_low,
        "subject_fidelity_CI_high": affinity_high,
        "random_advantage": advantage,
        "random_advantage_CI_low": advantage_low,
        "random_advantage_CI_high": advantage_high,
        "class_accuracy_change": float(frame["class_accuracy_change"].mean()),
        "class_accuracy_loss": float(-frame["class_accuracy_change"].mean()),
        "class_true_log_probability_change": float(frame["class_true_log_probability_change"].mean()),
        "manifold_transport_mean": float(np.mean(transported_knn)),
        "manifold_clean_mean": float(np.mean(clean_knn)),
        "independent_session_3NN_ratio": float(np.mean(transported_knn) / max(np.mean(clean_knn), EPS)),
        "off_manifold_rate": float(np.mean(off_values)),
        "random_off_manifold_rate": float(np.mean(random_off_values)),
        "off_manifold_excess_vs_random": float(np.mean(off_values) - np.mean(random_off_values)),
        "historical_strict_pass": bool(float(np.mean(transported_knn) / max(np.mean(clean_knn), EPS)) <= 1.25),
        "stage1_manifold_pass": bool(float(np.mean(transported_knn) / max(np.mean(clean_knn), EPS)) <= 1.30),
        "source_subjects": len(subjects),
    }
    return unit, subject_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args()
    c.ensure_dirs()
    units: list[dict[str, object]] = []
    subjects: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    for model in args.models:
        for dataset in c.DATASETS:
            for fold in c.FOLDS:
                for seed in c.SEEDS:
                    unit, rows = audit_unit(model, dataset, fold, seed)
                    units.append(unit)
                    subjects.extend(rows)
                    task_rows.append({key: unit[key] for key in ("model", "dataset", "fold", "seed", "BA", "macro_F1", "NLL", "threshold", "competent", "future_session_used")})
                    print(f"[stage1-audit] {model} {dataset} fold={fold} seed={seed} BA={float(unit['BA']):.5f} 3NN={float(unit['independent_session_3NN_ratio']):.5f}", flush=True)

    unit_frame = pd.DataFrame(units)
    task_frame = pd.DataFrame(task_rows)
    subject_frame = pd.DataFrame(subjects)
    c.write_csv(c.RESULTS / "STAGE1_ADMISSIBILITY_PER_FOLD.csv", unit_frame)
    c.write_csv(c.RESULTS / "MODEL_COMPETENCE_PER_FOLD.csv", task_frame)
    c.write_csv(RUNTIME / "STAGE1_SOURCE_SUBJECT_METRICS.csv", subject_frame)

    summary_rows: list[dict[str, object]] = []
    for (model, dataset), group in unit_frame.groupby(["model", "dataset"], sort=True):
        task_ba = float(group["BA"].mean())
        probe = float(group["independent_probe_BA"].mean())
        stability = float(group["residual_stability"].mean())
        subject_fidelity = float(group["subject_fidelity"].mean())
        random_advantage = float(group["random_advantage"].mean())
        class_change = float(group["class_accuracy_change"].mean())
        logprob = float(group["class_true_log_probability_change"].mean())
        ratio = float(group["manifold_transport_mean"].mean() / max(group["manifold_clean_mean"].mean(), EPS))
        off = float(group["off_manifold_excess_vs_random"].mean())
        # CIs are aggregated at the source-subject level, not at trial level.
        subset = subject_frame[(subject_frame.model == model) & (subject_frame.dataset == dataset)]
        _, stab_low, _ = bootstrap_ci(subset["stability"].to_numpy(), c.stable_seed("summary", model, dataset, "stability"))
        _, fid_low, _ = bootstrap_ci(subset["affinity"].to_numpy(), c.stable_seed("summary", model, dataset, "affinity"))
        _, rand_low, _ = bootstrap_ci(subset["random_advantage"].to_numpy(), c.stable_seed("summary", model, dataset, "random"))
        task_gate = bool(task_ba >= c.THRESHOLDS[dataset])
        stable_gate = bool(stability > 0 and stab_low > 0)
        fidelity_gate = bool(subject_fidelity > 0 and fid_low > 0)
        random_gate = bool(random_advantage > 0 and rand_low > 0)
        class_gate = bool(-class_change <= 0.02 and logprob >= -0.05)
        probe_gate = bool(probe >= 0.55)
        manifold_gate = bool(ratio <= 1.30 and off <= 0.02)
        summary_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "BA": task_ba,
                "macro_F1": float(group["macro_F1"].mean()),
                "threshold": c.THRESHOLDS[dataset],
                "competent": task_gate,
                "independent_probe_BA": probe,
                "residual_stability": stability,
                "stability_CI_low": stab_low,
                "subject_fidelity": subject_fidelity,
                "subject_fidelity_CI_low": fid_low,
                "random_advantage": random_advantage,
                "random_advantage_CI_low": rand_low,
                "class_accuracy_change": class_change,
                "class_accuracy_loss": -class_change,
                "class_true_log_probability_change": logprob,
                "independent_session_3NN_ratio": ratio,
                "historical_strict_pass": bool(ratio <= 1.25),
                "stage1_manifold_pass": bool(ratio <= 1.30),
                "off_manifold_excess_vs_random": off,
                "gate_task_competence": task_gate,
                "gate_residual_stability": stable_gate,
                "gate_subject_fidelity": fidelity_gate,
                "gate_random_advantage": random_gate,
                "gate_class_fidelity": class_gate,
                "gate_independent_probe_competence": probe_gate,
                "gate_manifold": manifold_gate,
                "all_stage1_gates": bool(task_gate and stable_gate and fidelity_gate and random_gate and class_gate and probe_gate and manifold_gate),
            }
        )
    summary = pd.DataFrame(summary_rows)
    c.write_csv(c.RESULTS / "STAGE1_ADMISSIBILITY.csv", summary)
    c.write_csv(c.RESULTS / "MODEL_COMPETENCE.csv", task_frame.groupby(["model", "dataset"], as_index=False).agg(BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), NLL=("NLL", "mean"), threshold=("threshold", "first"), competent=("competent", "all")))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
