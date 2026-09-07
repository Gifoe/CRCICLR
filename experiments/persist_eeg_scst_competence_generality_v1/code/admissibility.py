from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
RUNTIME = EXP / "runtime"; RESULTS = EXP / "results"; EPS = 1e-12
ALPHA_GRID = np.arange(17, dtype=np.float64) / 64.0
DATASETS = ("OpenBMI", "WBCIC"); FOLDS = range(5); SEEDS = range(3)
SOURCE_SESSIONS = {"OpenBMI": (1, 2), "WBCIC": (0, 1)}


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def load_rep(model: str, dataset: str, fold: int, seed: int, role: str) -> dict[str, np.ndarray]:
    if model == "CBraMod-R1": root = RUNTIME / "cbramod_repaired_representations"
    else: root = RUNTIME / "specialist_representations" / model
    path = root / dataset / f"fold-{fold}" / f"seed-{seed}" / f"{role}.npz"
    # Current experiment runtime is trusted and may contain pandas object
    # subject IDs from the first engineering pass.  Convert immediately to a
    # non-object Unicode array; no numeric representation is modified.
    with np.load(path, allow_pickle=True) as value:
        output = {key: value[key] for key in value.files}
    if output.get("subjects", np.empty(0)).dtype == object:
        output["subjects"] = output["subjects"].astype("U")
    return output


def support_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    q = np.asarray(query, np.float64); s = np.asarray(support, np.float64); distance = np.sum(q * q, 1)[:, None] + np.sum(s * s, 1)[None, :] - 2 * q @ s.T; np.maximum(distance, 0, out=distance); return np.sqrt(np.partition(distance, 2, axis=1)[:, :3]).mean(1)


def support_radius(support: np.ndarray) -> float:
    value = np.asarray(support, np.float64); distance = np.linalg.norm(value[:, None] - value[None, :], axis=2); np.fill_diagonal(distance, np.inf); clean = np.partition(distance, 2, axis=1)[:, :3].mean(1); return float(np.quantile(clean, .95))


def solve_alpha(query: np.ndarray, delta: np.ndarray, support: np.ndarray, radius: float) -> np.ndarray:
    selected = np.zeros(len(query))
    for alpha in ALPHA_GRID[1:]:
        selected[support_distance(query + alpha * delta, support) <= radius] = alpha
    return selected


def centroids(rep: dict[str, np.ndarray]) -> dict[tuple[str, int, int], np.ndarray]:
    output = {}
    for subject in sorted(np.unique(rep["subjects"].astype(str)), key=lambda x: (int(x) if x.isdigit() else 10**9, x)):
        for label in sorted(np.unique(rep["labels"])):
            for session in sorted(np.unique(rep["sessions"])):
                mask = (rep["subjects"].astype(str) == subject) & (rep["labels"] == label) & (rep["sessions"] == session)
                if mask.any(): output[(subject, int(label), int(session))] = rep["features"][mask].mean(0).astype(np.float64)
    return output


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS))


def subject_ci(frame: pd.DataFrame, value: str, seed: int) -> tuple[float, float, float]:
    per = frame.groupby("source_subject", as_index=False)[value].mean(); values = per[value].to_numpy(np.float64)
    if len(values) < 4 or not np.isfinite(values).all(): return float(np.nanmean(values)), float("nan"), float("nan")
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(values), size=(10000, len(values))); distribution = values[draws].mean(1); return float(values.mean()), float(np.quantile(distribution, .025)), float(np.quantile(distribution, .975))


def audit_unit(model: str, dataset: str, fold: int, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    cache_root = RUNTIME / "admissibility_units" / model / dataset / f"fold-{fold}" / f"seed-{seed}"
    unit_path = cache_root / "unit.json"; subject_path = cache_root / "subjects.csv"
    if unit_path.is_file() and subject_path.is_file():
        return json.loads(unit_path.read_text(encoding="utf-8")), pd.read_csv(subject_path).to_dict("records")
    source = load_rep(model, dataset, fold, seed, "model_fit"); validation = load_rep(model, dataset, fold, seed, "validation"); session_a, session_b = SOURCE_SESSIONS[dataset]
    bank = source["sessions"].astype(int) == session_a; center = source["features"][bank].mean(0); scale = source["features"][bank].std(0); scale[scale < 1e-6] = 1.0
    source = {**source, "features": ((source["features"] - center) / scale).astype(np.float32)}; validation = {**validation, "features": ((validation["features"] - center) / scale).astype(np.float32)}
    source_centroids = centroids(source); validation_centroids = centroids(validation); subjects = sorted(np.unique(source["subjects"].astype(str)), key=lambda x: (int(x) if x.isdigit() else 10**9, x)); validation_subjects = sorted(np.unique(validation["subjects"].astype(str)), key=lambda x: (int(x) if x.isdigit() else 10**9, x)); labels = sorted(np.unique(source["labels"]).astype(int))
    population = {(label, session): np.mean([source_centroids[(subject, label, session)] for subject in subjects], axis=0) for label in labels for session in (session_a, session_b)}
    residual = {(subject, label, session): source_centroids[(subject, label, session)] - population[(label, session)] for subject in subjects for label in labels for session in (session_a, session_b)}
    validation_bank = validation["sessions"].astype(int) == session_a; validation_eval = validation["sessions"].astype(int) == session_b
    probe = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, random_state=stable_seed("probe", model, dataset, fold, seed)).fit(validation["features"][validation_bank], validation["labels"][validation_bank])
    probe_ba = float(balanced_accuracy_score(validation["labels"][validation_eval], probe.predict(validation["features"][validation_eval])))
    metrics = {subject: {key: [] for key in ("stability_effect", "affinity_improvement", "advantage_over_random", "class_accuracy_change", "class_true_log_probability_change")} for subject in subjects}
    for subject in subjects:
        for label in labels:
            base = residual[(subject, label, session_a)]; matched = cosine(base, residual[(subject, label, session_b)]); mismatch = float(np.mean([cosine(base, residual[(other, label, session_b)]) for other in subjects if other != subject])); metrics[subject]["stability_effect"].append(matched - mismatch)
    rng = np.random.default_rng(stable_seed("scst-audit", model, dataset, fold, seed)); knn_transport = []; knn_clean = []; off = []; off_random = []
    for label in labels:
        support = np.stack([source_centroids[(subject, label, session_a)] for subject in subjects]); radius = support_radius(support); real = np.stack([source_centroids[(subject, label, session_b)] for subject in subjects] + [validation_centroids[(subject, label, session_b)] for subject in validation_subjects]); knn = NearestNeighbors(n_neighbors=3).fit(real); loo = NearestNeighbors(n_neighbors=4).fit(real); threshold = float(np.quantile(loo.kneighbors(real, return_distance=True)[0][:, 1:].mean(1), .95))
        for source_subject in subjects:
            targets = [target for target in subjects if target != source_subject]; query = np.repeat(source_centroids[(source_subject, label, session_b)][None, :], len(targets), 0); target_centroid = np.stack([source_centroids[(target, label, session_b)] for target in targets]); delta = np.stack([residual[(target, label, session_a)] - residual[(source_subject, label, session_a)] for target in targets]); alpha = solve_alpha(query, delta, support, radius); transported = query + alpha[:, None] * delta
            random_delta = []
            for value in delta:
                random = rng.normal(size=len(value)); random *= np.linalg.norm(value) / max(np.linalg.norm(random), EPS); random_delta.append(random)
            random_delta = np.asarray(random_delta); random_transport = query + alpha[:, None] * random_delta; clean_distance = np.linalg.norm(query - target_centroid, axis=1); transported_distance = np.linalg.norm(transported - target_centroid, axis=1); random_distance = np.linalg.norm(random_transport - target_centroid, axis=1); relative = (clean_distance - transported_distance) / np.maximum(clean_distance, EPS); random_relative = (clean_distance - random_distance) / np.maximum(clean_distance, EPS); metrics[source_subject]["affinity_improvement"].extend(relative.tolist()); metrics[source_subject]["advantage_over_random"].extend((relative - random_relative).tolist())
            transported_knn = knn.kneighbors(transported, return_distance=True)[0].mean(1); clean_knn = knn.kneighbors(query, return_distance=True)[0].mean(1); random_knn = knn.kneighbors(random_transport, return_distance=True)[0].mean(1); knn_transport.extend(transported_knn.tolist()); knn_clean.extend(clean_knn.tolist()); off.extend((transported_knn > threshold).tolist()); off_random.extend((random_knn > threshold).tolist())
            trials = np.flatnonzero((source["subjects"].astype(str) == source_subject) & (source["labels"] == label) & (source["sessions"] == session_b)); clean = source["features"][trials].astype(np.float64); clean_pred = probe.predict(clean); class_index = list(probe.classes_).index(label); clean_probability = probe.predict_proba(clean)[:, class_index]; clean_accuracy = float(np.mean(clean_pred == label)); clean_log_probability = np.log(np.clip(clean_probability, 1e-12, 1))
            for index, _ in enumerate(targets):
                trial_delta = np.repeat(delta[index][None, :], len(clean), axis=0); trial_alpha = solve_alpha(clean, trial_delta, support, radius); moved = clean + trial_alpha[:, None] * trial_delta; prediction = probe.predict(moved); probability = probe.predict_proba(moved)[:, class_index]; metrics[source_subject]["class_accuracy_change"].append(float(np.mean(prediction == label) - clean_accuracy)); metrics[source_subject]["class_true_log_probability_change"].append(float(np.mean(np.log(np.clip(probability, 1e-12, 1)) - clean_log_probability)))
    subject_rows = [{"dataset": dataset, "model": model, "fold": fold, "seed": seed, "source_subject": subject, **{key: float(np.mean(value)) for key, value in values.items()}} for subject, values in metrics.items()]
    frame = pd.DataFrame(subject_rows); stability, stability_low, stability_high = subject_ci(frame, "stability_effect", stable_seed("unit-stability", model, dataset, fold, seed)); affinity, affinity_low, affinity_high = subject_ci(frame, "affinity_improvement", stable_seed("unit-affinity", model, dataset, fold, seed)); advantage, advantage_low, advantage_high = subject_ci(frame, "advantage_over_random", stable_seed("unit-random", model, dataset, fold, seed))
    unit = {"dataset": dataset, "model": model, "fold": fold, "seed": seed, "independent_probe_BA": probe_ba, "residual_stability": stability, "stability_CI_low": stability_low, "stability_CI_high": stability_high, "affinity_improvement": affinity, "affinity_CI_low": affinity_low, "affinity_CI_high": affinity_high, "advantage_over_random": advantage, "advantage_over_random_CI_low": advantage_low, "advantage_over_random_CI_high": advantage_high, "class_accuracy_change": float(frame.class_accuracy_change.mean()), "class_accuracy_loss": float(-frame.class_accuracy_change.mean()), "class_true_log_probability_change": float(frame.class_true_log_probability_change.mean()), "manifold_transport_mean": float(np.mean(knn_transport)), "manifold_clean_mean": float(np.mean(knn_clean)), "independent_session_3NN_ratio": float(np.mean(knn_transport) / max(np.mean(knn_clean), EPS)), "off_manifold_rate": float(np.mean(off)), "random_off_manifold_rate": float(np.mean(off_random)), "off_manifold_excess_vs_random": float(np.mean(off) - np.mean(off_random)), "source_subjects": len(frame)}
    cache_root.mkdir(parents=True, exist_ok=True); unit_path.write_text(json.dumps(unit, indent=2) + "\n", encoding="utf-8"); frame.to_csv(subject_path, index=False)
    return unit, subject_rows


def competence_table() -> pd.DataFrame:
    repaired = pd.read_csv(RESULTS / "CBRAMOD_REPAIR_COMPETENCE.csv"); specialists = pd.read_csv(RESULTS / "SPECIALIST_SCREEN.csv"); return pd.concat([repaired, specialists], ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--precompute-model"); args = parser.parse_args()
    if args.precompute_model:
        model = args.precompute_model
        for dataset in DATASETS:
            for fold in FOLDS:
                for seed in SEEDS:
                    audit_unit(model, dataset, fold, seed); print(f"[admissibility-precompute] {model} {dataset} fold={fold} seed={seed}", flush=True)
        return
    competence = competence_table(); models = ["CBraMod-R1"] + sorted(competence[competence.type == "Specialist"].model.unique().tolist()); rows = []; subject_rows = []; skipped = []
    for model in models:
        for dataset in DATASETS:
            task = competence[(competence.model == model) & (competence.dataset == dataset)]
            if task.empty: raise RuntimeError((model, dataset))
            task_ok = bool(task.competent.iloc[0])
            if model != "CBraMod-R1" and not task_ok:
                skipped.append({"model": model, "dataset": dataset, "reason": "SPECIALIST_NOT_COMPETENT"}); continue
            for fold in FOLDS:
                for seed in SEEDS:
                    unit, cells = audit_unit(model, dataset, fold, seed); rows.append(unit); subject_rows.extend(cells); print(f"[admissibility] {model} {dataset} fold={fold} seed={seed}", flush=True)
    per_fold = pd.DataFrame(rows); subjects = pd.DataFrame(subject_rows); per_fold.to_csv(RESULTS / "SCST_VALIDITY_PER_FOLD.csv", index=False); subjects.to_csv(RUNTIME / "SCST_VALIDITY_PER_SOURCE_SUBJECT.csv", index=False)
    summaries = []
    for (model, dataset), units in per_fold.groupby(["model", "dataset"]):
        cells = subjects[(subjects.model == model) & (subjects.dataset == dataset)]; stability, stability_low, stability_high = subject_ci(cells, "stability_effect", stable_seed("summary-stability", model, dataset)); affinity, affinity_low, affinity_high = subject_ci(cells, "affinity_improvement", stable_seed("summary-affinity", model, dataset)); advantage, advantage_low, advantage_high = subject_ci(cells, "advantage_over_random", stable_seed("summary-random", model, dataset)); class_change = float(cells.groupby("source_subject").class_accuracy_change.mean().mean()); log_probability = float(cells.groupby("source_subject").class_true_log_probability_change.mean().mean()); ratio = float(units.manifold_transport_mean.mean() / max(units.manifold_clean_mean.mean(), EPS)); off = float(units.off_manifold_rate.mean() - units.random_off_manifold_rate.mean()); task_ok = bool(competence[(competence.model == model) & (competence.dataset == dataset)].competent.iloc[0]); probe_ok = bool(units.independent_probe_BA.mean() >= .55); gate_stability = bool(stability > 0 and stability_low > 0); gate_subject = bool(affinity > 0 and affinity_low > 0 and advantage > 0 and advantage_low > 0); gate_class = bool(-class_change <= .02 and log_probability >= -.05); gate_manifold = bool(ratio <= 1.25 and off <= .02)
        summaries.append({"dataset": dataset, "model": model, "task_BA": float(competence[(competence.model == model) & (competence.dataset == dataset)].BA.iloc[0]), "task_competent": task_ok, "independent_probe_BA": float(units.independent_probe_BA.mean()), "residual_stability": stability, "stability_CI_low": stability_low, "stability_CI_high": stability_high, "affinity_improvement": affinity, "affinity_CI_low": affinity_low, "affinity_CI_high": affinity_high, "advantage_over_random": advantage, "advantage_over_random_CI_low": advantage_low, "advantage_over_random_CI_high": advantage_high, "class_accuracy_change": class_change, "class_accuracy_loss": -class_change, "class_true_log_probability_change": log_probability, "independent_session_3NN_ratio": ratio, "off_manifold_excess_vs_random": off, "gate_task_competence": task_ok, "gate_independent_probe_competence": probe_ok, "gate_residual_stability": gate_stability, "gate_subject_fidelity": gate_subject, "gate_random_advantage": gate_subject, "gate_class_fidelity": gate_class, "gate_manifold": gate_manifold, "all_admissibility_gates": bool(task_ok and probe_ok and gate_stability and gate_subject and gate_class and gate_manifold)})
    summary = pd.DataFrame(summaries)
    for row in skipped:
        task = competence[(competence.model == row["model"]) & (competence.dataset == row["dataset"])].iloc[0]
        summaries.append({"dataset": row["dataset"], "model": row["model"], "task_BA": float(task.BA), "task_competent": False, "all_admissibility_gates": False, "audit_skipped_reason": row["reason"]})
    summary = pd.DataFrame(summaries); summary.to_csv(RESULTS / "SCST_VALIDITY_PER_MODEL.csv", index=False)
    terminals = []
    for model in models:
        part = summary[summary.model == model]; both_competent = len(part) == 2 and bool(part.task_competent.fillna(False).all()); both_valid = len(part) == 2 and bool(part.all_admissibility_gates.fillna(False).all())
        if model == "CBraMod-R1": terminal = "CBRAMOD_SCST_ADMISSIBLE" if both_valid else ("CBRAMOD_COMPETENT_BUT_NOT_ADMISSIBLE" if both_competent else "CBRAMOD_COMPETENCE_NOT_RECOVERED")
        else: terminal = "SPECIALIST_COMPETENT_AND_ADMISSIBLE" if both_valid else ("SPECIALIST_COMPETENT_NOT_ADMISSIBLE" if both_competent else "SPECIALIST_NOT_COMPETENT")
        terminals.append({"model": model, "terminal": terminal, "competent_both_datasets": both_competent, "admissible_both_datasets": both_valid})
    terminal_frame = pd.DataFrame(terminals); terminal_frame.to_csv(RESULTS / "SCST_MODEL_TERMINALS.csv", index=False)
    fm = terminal_frame[(terminal_frame.model == "CBraMod-R1") & terminal_frame.admissible_both_datasets]; specialist = terminal_frame[(terminal_frame.model != "CBraMod-R1") & terminal_frame.admissible_both_datasets]; level1 = bool(len(fm) + len(specialist)); level2 = bool(len(fm) and len(specialist)); authorization = {"schema": "SCST_ADMISSIBILITY_AUTHORIZATION_V1", "level1_SCST_DISCOVERY_TRAINING_AUTHORIZED": level1, "level2_SCST_GENERAL_METHOD_DEVELOPMENT_AUTHORIZED": level2, "eligible_models": terminal_frame[terminal_frame.admissible_both_datasets].model.tolist(), "terminals": terminals, "sealed_resources_untouched": True}
    (RESULTS / "SCST_AUTHORIZATION.json").write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8"); print(json.dumps(authorization, indent=2), flush=True)


if __name__ == "__main__":
    main()
