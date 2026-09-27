"""Algebraically equivalent cached-centroid/batched V2 transform audit.

V1's fold0 embedding output is preserved. V2 avoids repeatedly rescanning the
large temporal/spatial activation matrices: all linear-transform fitting and
geometry scores use TRAIN or held centroids/moments computed once per family.
Frozen decoder evaluation still scores every held S2 trial, in batches.
"""
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
import transform_runner as v1


def tables(x: np.ndarray, pop: dg.Population, sessions: tuple[int, int]) -> tuple[dict, dict]:
    owners = sorted(set(pop.subject))
    means = {(subject, session): x[(pop.subject == subject) & (pop.session == session)].mean(axis=0)
             for subject in owners for session in sessions}
    centroids = semantics.centroid_grid(x, pop)
    return means, centroids


def fit_from_tables(means: dict, centroids: dict, sessions: tuple[int, int],
                    family: str, mode: str, subjects: set[str]):
    owners = sorted(subjects)
    source = np.stack([means[(subject, sessions[0])] for subject in owners]).astype(np.float32)
    target = np.stack([means[(subject, sessions[1])] for subject in owners]).astype(np.float32)
    if mode == "UNSUPERVISED":
        return canonical.fit_unsupervised(source, target, family)
    if mode == "LABEL_ASSISTED_ORACLE":
        first = np.stack([centroids[(subject, sessions[0], label)] for subject in owners for label in (0, 1)])
        second = np.stack([centroids[(subject, sessions[1], label)] for subject in owners for label in (0, 1)])
        return canonical.fit_oracle(first, second, family,
                                    unlabeled_support_pool=np.concatenate((source, target)))
    raise KeyError(mode)


def alignment_from_grid(grid: dict, owners: list[str], sessions: tuple[int, int], fitted) -> dict:
    first, second = [], []
    centroid_residuals, relation_cosines, relation_residuals = [], [], []
    for subject in owners:
        a = [grid[(subject, sessions[0], label)] for label in (0, 1)]
        b = [fitted.apply(grid[(subject, sessions[1], label)][None])[0] for label in (0, 1)]
        first.extend(a)
        second.extend(b)
        centroid_residuals.append(float(np.mean([
            np.linalg.norm(p - q) / max(np.linalg.norm(p), 1e-8) for p, q in zip(a, b)])))
        pair = semantics.sim(a[1] - a[0], b[1] - b[0])
        relation_cosines.append(pair["cosine"])
        relation_residuals.append(pair["optimal_scalar_residual"])
    a, b = np.stack(first), np.stack(second)
    a, b = a - a.mean(0), b - b.mean(0)
    gram_a = a @ a.T / max(len(a) - 1, 1)
    gram_b = b @ b.T / max(len(b) - 1, 1)
    return {"centroid_residual": float(np.mean(centroid_residuals)),
            "class_relation_cosine": float(np.mean(relation_cosines)),
            "class_relation_residual": float(np.mean(relation_residuals)),
            "centroid_gram_covariance_discrepancy": float(
                np.linalg.norm(gram_b - gram_a) / max(np.linalg.norm(gram_a), 1e-8)),
            "subject_equal_count": len(owners)}


def rank_cv_from_tables(means: dict, centroids: dict, sessions: tuple[int, int], mode: str) -> dict[str, float]:
    subjects = np.asarray(sorted({key[0] for key in means}))
    scores = {rank: [] for rank in canonical.RANKS}
    splitter = GroupKFold(n_splits=5)
    for fit_idx, eval_idx in splitter.split(subjects, groups=subjects):
        fit = set(subjects[fit_idx])
        held = sorted(subjects[eval_idx])
        for rank in canonical.RANKS:
            fitted = fit_from_tables(means, centroids, sessions,
                                     f"lowrank_residual_affine_r{rank}", mode, fit)
            scores[rank].append(alignment_from_grid(
                centroids, held, sessions, fitted)["class_relation_residual"])
    return {str(rank): float(np.mean(scores[rank])) for rank in canonical.RANKS}


def scores_batched(scaler, decoder, y: np.ndarray, x: np.ndarray, fitted=None, batch: int = 64) -> dict:
    pieces = []
    for start in range(0, len(y), batch):
        segment = x[start:start + batch]
        if fitted is not None:
            segment = fitted.apply(segment)
        pieces.append(decoder.predict_proba(scaler.transform(segment)))
    return decoders.metrics(y, np.concatenate(pieces), decoder.classes_)


def analyze_one(fold: int, stage: str, geometry: dict, populations: dict,
                sessions: tuple[int, int], family: str, draw: int) -> dict[str, list[dict]]:
    q = geometry["q"]
    if family == "RANDOM":
        q = dg.random_subspace(len(geometry["mu"]), q.shape[1], fold, stage, draw)
    train, train_raw = populations["TRAIN_GEOMETRY"]
    train_x = decoders.coordinates(train_raw, geometry["mu"], q, family)
    means, centroids = tables(train_x, train, sessions)
    del train_x
    raw_cent, labels, _, session = decoders.centroid_training(train, train_raw, sessions)
    decoder_cent = decoders.coordinates(raw_cent, geometry["mu"], q, family)
    fit_session = session == sessions[0]
    scaler, decoder = decoders.fit_linear(decoder_cent[fit_session], labels[fit_session])
    held = {}
    for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT"):
        pop, raw = populations[role]
        x = decoders.coordinates(raw, geometry["mu"], q, family)
        target = pop.session == sessions[1]
        held[role] = (pop, x, target, semantics.centroid_grid(x, pop),
                      scores_batched(scaler, decoder, pop.y[target], x[target]))
    output = {key: [] for key in ("GLOBAL_SESSION_TRANSFORM", "CROSS_SUBJECT_TRANSFORM",
                                  "TRANSFORM_DECODER_RECOVERY", "TRANSFORMATION_COMPLEXITY_CURVE")}
    subjects = {key[0] for key in means}
    for mode in ("UNSUPERVISED", "LABEL_ASSISTED_ORACLE"):
        cv = rank_cv_from_tables(means, centroids, sessions, mode)
        chosen_rank = min(canonical.RANKS, key=lambda rank: (cv[str(rank)], rank))
        for transform_family in canonical.families():
            fitted = fit_from_tables(means, centroids, sessions, transform_family, mode, subjects)
            for role, (pop, x, target, grid, original) in held.items():
                changed = scores_batched(scaler, decoder, pop.y[target], x[target], fitted)
                alignment = alignment_from_grid(grid, sorted(set(pop.subject)), sessions, fitted)
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
                score = {"baseline_BA": original["BA"], "transformed_BA": changed["BA"],
                         "BA_recovery": changed["BA"] - original["BA"],
                         "baseline_macro_F1": original["macro_F1"],
                         "transformed_macro_F1": changed["macro_F1"],
                         "macro_F1_recovery": changed["macro_F1"] - original["macro_F1"],
                         "baseline_NLL": original["NLL"], "transformed_NLL": changed["NLL"],
                         "NLL_recovery": changed["NLL"] - original["NLL"],
                         "evaluation_rows": int(target.sum())}
                output["GLOBAL_SESSION_TRANSFORM"].append({**base, **score})
                output["TRANSFORM_DECODER_RECOVERY"].append({**base, **score})
                output["TRANSFORMATION_COMPLEXITY_CURVE"].append({**base, **score})
                output["CROSS_SUBJECT_TRANSFORM"].append({**base, **score,
                    "arm": "ZERO_SHOT_GLOBAL" if mode == "UNSUPERVISED" else "TRAIN_LABEL_ORACLE_GLOBAL"})
    source_train = np.stack([means[(subject, sessions[0])] for subject in sorted(subjects)]).astype(np.float32)
    for role, (pop, x, _, _, _) in held.items():
        output["CROSS_SUBJECT_TRANSFORM"].extend(v1.subject_unlabeled_rows(
            fold, stage, family, draw, source_train, train, pop, source_train, x, sessions, scaler, decoder))
    return output


def run(fold: int, stage: str) -> None:
    target = dg.RUNTIME / "analysis" / f"fold{fold}_seed0" / stage / "transform"
    if target.exists():
        raise FileExistsError(target)
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    _, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    before = dg.model_sha(model)
    geometry, provenance = ar.verify_geometry(fold, stage, dg.file_sha(ckpt), context)
    held = {role: v1.label_free_cap_held(dg.load_held(role, context, een, up), fold)
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
            print(f"TRANSFORM_V2_PROGRESS fold={fold} stage={stage} family={family} draw={draw}", flush=True)
    if dg.model_sha(model) != before:
        raise RuntimeError("frozen neural parameter/BatchNorm state changed")
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in output.items():
        path = target / f"{name}.csv"
        ar.write_rows(path, rows)
        hashes[path.name] = {"sha256": dg.file_sha(path), "rows": len(rows)}
    audit = {"schema": "P_SEMANTICS_FOLD_STAGE_TRANSFORM_V2", "fold": fold, "stage": stage,
             "checkpoint_sha256": dg.file_sha(ckpt),
             "geometry_provenance_sha256": dg.file_sha(
                 dg.RUNTIME / "geometry" / f"fold{fold}_seed0" / "GEOMETRY_PROVENANCE.json"),
             "stage_geometry_sha256": provenance["stage_geometry_sha256"][stage],
             "model_state_sha256_before_after": before,
             "transform_fit_population": "TRAIN_GEOMETRY_ONLY",
             "target_labels_used_for_unsupervised_fit": False,
             "oracle_is_train_label_assisted_diagnostic": True,
             "target_decoder_refit": False,
             "optimization": "cached group moments/centroids; affine centroid equivalence; batched frozen decoder",
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
