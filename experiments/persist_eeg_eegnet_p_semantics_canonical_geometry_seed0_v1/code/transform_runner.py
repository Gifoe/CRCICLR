"""Frozen-decoder evaluation of TRAIN-fitted canonical transforms."""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import GroupKFold

import analysis_runner as ar
import canonical
import data_geometry as dg
import decoders
import semantics


def label_free_cap_held(pop: dg.Population, fold: int, limit: int = 128) -> dg.Population:
    chosen = []
    for subject in sorted(set(pop.subject)):
        for session in sorted(set(pop.session)):
            ix = np.flatnonzero((pop.subject == subject) & (pop.session == session))
            rng = np.random.default_rng(dg.seed("transform-held-unlabeled-cap", fold, subject, session))
            chosen.extend(np.sort(rng.choice(ix, size=min(limit, len(ix)), replace=False)).tolist())
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    return dg.Population(pop.x[chosen], pop.y[chosen], pop.subject[chosen], pop.session[chosen], pop.role)


def unlabeled_subject_means(x: np.ndarray, pop: dg.Population, session: int,
                            subjects: set[str] | None = None) -> np.ndarray:
    owners = sorted(set(pop.subject) if subjects is None else subjects)
    return np.stack([x[(pop.subject == owner) & (pop.session == session)].mean(axis=0)
                     for owner in owners]).astype(np.float32)


def paired_class_centroids(x: np.ndarray, pop: dg.Population, sessions: tuple[int, int],
                           subjects: set[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    grid = semantics.centroid_grid(x, pop)
    owners = sorted(set(pop.subject) if subjects is None else subjects)
    source = np.stack([grid[(owner, sessions[0], label)] for owner in owners for label in (0, 1)])
    target = np.stack([grid[(owner, sessions[1], label)] for owner in owners for label in (0, 1)])
    return source.astype(np.float32), target.astype(np.float32)


def frozen_scores(scale, decoder, y: np.ndarray, x: np.ndarray) -> dict:
    probabilities = decoder.predict_proba(scale.transform(x))
    return decoders.metrics(y, probabilities, decoder.classes_)


def alignment_metrics(reference: np.ndarray, transformed: np.ndarray, pop: dg.Population,
                      sessions: tuple[int, int]) -> dict:
    # Both arrays are full population matrices; transformed applies only to S2.
    grid_ref = semantics.centroid_grid(reference, pop)
    grid_tr = semantics.centroid_grid(transformed, pop)
    owners = sorted(set(pop.subject))
    centroid_residuals, relation_cosines, relation_residuals = [], [], []
    for subject in owners:
        first = [grid_ref[(subject, sessions[0], label)] for label in (0, 1)]
        second = [grid_tr[(subject, sessions[1], label)] for label in (0, 1)]
        centroid_residuals.append(float(np.mean([np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-8)
                                                 for a, b in zip(first, second)])))
        d1, d2 = first[1] - first[0], second[1] - second[0]
        pair = semantics.sim(d1, d2)
        relation_cosines.append(pair["cosine"])
        relation_residuals.append(pair["optimal_scalar_residual"])
    source_centroids = np.stack([grid_ref[(s, sessions[0], y)] for s in owners for y in (0, 1)])
    target_centroids = np.stack([grid_tr[(s, sessions[1], y)] for s in owners for y in (0, 1)])
    # Gram-space discrepancy equals a covariance comparison without forming D^2.
    a, b = source_centroids - source_centroids.mean(0), target_centroids - target_centroids.mean(0)
    gram_a = a @ a.T / max(len(a) - 1, 1)
    gram_b = b @ b.T / max(len(b) - 1, 1)
    discrepancy = float(np.linalg.norm(gram_b - gram_a) / max(np.linalg.norm(gram_a), 1e-8))
    return {"centroid_residual": float(np.mean(centroid_residuals)),
            "class_relation_cosine": float(np.mean(relation_cosines)),
            "class_relation_residual": float(np.mean(relation_residuals)),
            "centroid_gram_covariance_discrepancy": discrepancy,
            "subject_equal_count": len(owners)}


def fit_transform(x: np.ndarray, pop: dg.Population, sessions: tuple[int, int],
                  transform_family: str, mode: str, subjects: set[str] | None = None):
    src = unlabeled_subject_means(x, pop, sessions[0], subjects)
    tgt = unlabeled_subject_means(x, pop, sessions[1], subjects)
    if mode == "UNSUPERVISED":
        return canonical.fit_unsupervised(src, tgt, transform_family)
    if mode == "LABEL_ASSISTED_ORACLE":
        pair_s, pair_t = paired_class_centroids(x, pop, sessions, subjects)
        return canonical.fit_oracle(pair_s, pair_t, transform_family,
                                    unlabeled_support_pool=np.concatenate((src, tgt)))
    raise KeyError(mode)


def rank_cv(x: np.ndarray, pop: dg.Population, sessions: tuple[int, int], mode: str) -> dict[str, float]:
    # Fold by biological TRAIN subject. Only TRAIN labels enter the validation
    # score; UNSUPERVISED transform *fitting* still receives unlabeled means.
    subjects = np.asarray(sorted(set(pop.subject)))
    scores = {rank: [] for rank in canonical.RANKS}
    splitter = GroupKFold(n_splits=5)
    for fit_idx, eval_idx in splitter.split(subjects, groups=subjects):
        fit_subjects, held_subjects = set(subjects[fit_idx]), set(subjects[eval_idx])
        held_mask = np.isin(pop.subject, list(held_subjects))
        held = dg.Population(pop.x[held_mask], pop.y[held_mask], pop.subject[held_mask], pop.session[held_mask], pop.role)
        held_x = x[held_mask]
        target = held.session == sessions[1]
        for rank in canonical.RANKS:
            transform = fit_transform(x, pop, sessions, f"lowrank_residual_affine_r{rank}", mode, fit_subjects)
            changed = held_x.copy()
            changed[target] = transform.apply(held_x[target])
            value = alignment_metrics(held_x, changed, held, sessions)["class_relation_residual"]
            scores[rank].append(value)
    return {str(rank): float(np.mean(scores[rank])) for rank in canonical.RANKS}


def subject_unlabeled_rows(fold: int, stage: str, family: str, draw: int, x: np.ndarray,
                           train: dg.Population, held: dg.Population, source_train: np.ndarray,
                           held_x: np.ndarray, sessions: tuple[int, int], scale, decoder) -> list[dict]:
    rows = []
    for subject in sorted(set(held.subject)):
        subset = (held.subject == subject) & (held.session == sessions[1])
        indices = np.flatnonzero(subset)
        if len(indices) < 8:
            raise RuntimeError("target subject lacks unlabeled calibration/evaluation rows")
        rng = np.random.default_rng(dg.seed("target-unlabeled-calibration", fold, stage, family, draw, subject))
        order = rng.permutation(indices)
        calibration = order[:len(order)//2]
        evaluation = order[len(order)//2:]
        for name in ("translation", "diagonal_affine", "orthogonal_translation"):
            transform = canonical.fit_unsupervised(source_train, held_x[calibration], name,
                                                   support_pool=np.concatenate((source_train, held_x[calibration])))
            baseline = frozen_scores(scale, decoder, held.y[evaluation], held_x[evaluation])
            moved = frozen_scores(scale, decoder, held.y[evaluation], transform.apply(held_x[evaluation]))
            rows.append({"fold": fold, "stage": stage, "family": family,
                         "random_draw": draw if family == "RANDOM" else "",
                         "population": held.role, "subject": subject, "transform": name,
                         "arm": "TARGET_UNLABELED_ADAPTATION_DIAGNOSTIC",
                         "target_labels_used_for_transform": False,
                         "target_unlabeled_calibration_rows": len(calibration),
                         "independent_evaluation_rows": len(evaluation),
                         "untransformed_BA": baseline["BA"], "transformed_BA": moved["BA"],
                         "BA_recovery": moved["BA"] - baseline["BA"],
                         "macro_F1_recovery": moved["macro_F1"] - baseline["macro_F1"],
                         "NLL_recovery": moved["NLL"] - baseline["NLL"],
                         **transform.complexity()})
    return rows


def analyze_one(fold: int, stage: str, geometry: dict, populations: dict,
                sessions: tuple[int, int], family: str, draw: int) -> dict[str, list[dict]]:
    q = geometry["q"]
    if family == "RANDOM":
        q = dg.random_subspace(len(geometry["mu"]), q.shape[1], fold, stage, draw)
    train, train_raw = populations["TRAIN_GEOMETRY"]
    train_x = decoders.coordinates(train_raw, geometry["mu"], q, family)
    cent, labels, subjects, sess = decoders.centroid_training(train, train_raw, sessions)
    cent = decoders.coordinates(cent, geometry["mu"], q, family)
    fit_session = sess == sessions[0]
    scaler, decoder = decoders.fit_linear(cent[fit_session], labels[fit_session])
    output = {key: [] for key in ("GLOBAL_SESSION_TRANSFORM", "CROSS_SUBJECT_TRANSFORM",
                                 "TRANSFORM_DECODER_RECOVERY", "TRANSFORMATION_COMPLEXITY_CURVE")}
    for mode in ("UNSUPERVISED", "LABEL_ASSISTED_ORACLE"):
        cv = rank_cv(train_x, train, sessions, mode)
        chosen_rank = min(canonical.RANKS, key=lambda rank: (cv[str(rank)], rank))
        for transform_family in canonical.families():
            fitted = fit_transform(train_x, train, sessions, transform_family, mode)
            for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT"):
                held, raw = populations[role]
                held_x = decoders.coordinates(raw, geometry["mu"], q, family)
                target = held.session == sessions[1]
                original = frozen_scores(scaler, decoder, held.y[target], held_x[target])
                moved = fitted.apply(held_x[target])
                transformed = frozen_scores(scaler, decoder, held.y[target], moved)
                changed = held_x.copy()
                changed[target] = moved
                alignment = alignment_metrics(held_x, changed, held, sessions)
                base = {"fold": fold, "stage": stage, "family": family,
                        "random_draw": draw if family == "RANDOM" else "",
                        "mode": mode, "transform": transform_family,
                        "fit_population": "TRAIN_GEOMETRY_ONLY", "eval_population": role,
                        "transform_uses_target_labels": False,
                        "oracle_train_class_correspondence": mode == "LABEL_ASSISTED_ORACLE",
                        "target_decoder_refit": False,
                        "train_CV_lowrank_rank": chosen_rank,
                        "train_CV_rank_scores": json.dumps(cv, sort_keys=True),
                        **fitted.complexity(), **alignment}
                score = {"baseline_BA": original["BA"], "transformed_BA": transformed["BA"],
                         "BA_recovery": transformed["BA"] - original["BA"],
                         "baseline_macro_F1": original["macro_F1"],
                         "transformed_macro_F1": transformed["macro_F1"],
                         "macro_F1_recovery": transformed["macro_F1"] - original["macro_F1"],
                         "baseline_NLL": original["NLL"], "transformed_NLL": transformed["NLL"],
                         "NLL_recovery": transformed["NLL"] - original["NLL"],
                         "evaluation_rows": int(target.sum())}
                output["GLOBAL_SESSION_TRANSFORM"].append({**base, **score})
                output["TRANSFORM_DECODER_RECOVERY"].append({**base, **score})
                output["TRANSFORMATION_COMPLEXITY_CURVE"].append({**base, **score})
                output["CROSS_SUBJECT_TRANSFORM"].append({**base, **score,
                                                           "arm": "ZERO_SHOT_GLOBAL" if mode == "UNSUPERVISED" else "TRAIN_LABEL_ORACLE_GLOBAL"})
    for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT"):
        held, raw = populations[role]
        held_x = decoders.coordinates(raw, geometry["mu"], q, family)
        reference = unlabeled_subject_means(train_x, train, sessions[0])
        output["CROSS_SUBJECT_TRANSFORM"].extend(subject_unlabeled_rows(
            fold, stage, family, draw, train_x, train, held, reference, held_x, sessions, scaler, decoder))
    return output


def run(fold: int, stage: str):
    target = dg.RUNTIME / "analysis" / f"fold{fold}_seed0" / stage / "transform"
    if target.exists():
        raise FileExistsError(target)
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    before = dg.model_sha(model)
    geometry, provenance = ar.verify_geometry(fold, stage, dg.file_sha(ckpt), context)
    held = {role: label_free_cap_held(dg.load_held(role, context, een, up), fold)
            for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT")}
    stages = pw.Stages(model, head, "EEGNet", device)
    populations = {"TRAIN_GEOMETRY": (train, ar.extract_stage(train, stages, stage))}
    populations.update({role: (pop, ar.extract_stage(pop, stages, stage)) for role, pop in held.items()})
    output = {key: [] for key in ("GLOBAL_SESSION_TRANSFORM", "CROSS_SUBJECT_TRANSFORM",
                                  "TRANSFORM_DECODER_RECOVERY", "TRANSFORMATION_COMPLEXITY_CURVE")}
    for family in ("FULL", "P", "C", "RANDOM"):
        for draw in (range(20) if family == "RANDOM" else (-1,)):
            result = analyze_one(fold, stage, geometry, populations, context["sessions"], family, draw)
            for key, rows in result.items():
                output[key].extend(rows)
            print(f"TRANSFORM_PROGRESS fold={fold} stage={stage} family={family} draw={draw}", flush=True)
    if dg.model_sha(model) != before:
        raise RuntimeError("frozen neural parameter/BatchNorm state changed")
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in output.items():
        path = target / f"{name}.csv"
        ar.write_rows(path, rows)
        hashes[path.name] = {"sha256": dg.file_sha(path), "rows": len(rows)}
    audit = {"schema": "P_SEMANTICS_FOLD_STAGE_TRANSFORM_V1", "fold": fold, "stage": stage,
             "checkpoint_sha256": dg.file_sha(ckpt),
             "geometry_provenance_sha256": dg.file_sha(dg.RUNTIME / "geometry" / f"fold{fold}_seed0" / "GEOMETRY_PROVENANCE.json"),
             "stage_geometry_sha256": provenance["stage_geometry_sha256"][stage],
             "model_state_sha256_before_after": before,
             "transform_fit_population": "TRAIN_GEOMETRY_ONLY",
             "target_labels_used_for_unsupervised_fit": False,
             "oracle_is_train_label_assisted_diagnostic": True,
             "target_decoder_refit": False,
             "final_heldout_eeg_reads": 0, "files": hashes}
    with (target / "AUDIT.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"fold": fold, "stage": stage, "files": hashes,
                      "final_heldout_eeg_reads": 0}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--stage", choices=dg.STAGES, required=True)
    args = parser.parse_args()
    run(args.fold, args.stage)
