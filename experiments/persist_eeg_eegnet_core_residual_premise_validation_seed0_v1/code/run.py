"""Frozen EEGNet core/residual premise audit; no neural fitting or final-heldout access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
SOURCE = REPO / "experiments/persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1"
PARENT = REPO / "experiments/persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1"
PARENT_OUTPUT = PARENT / os.environ.get("TASKCORE_OUTPUT_FOLDER", "outputs")
sys.path.insert(0, str(SOURCE / "code"))
sys.path.insert(0, str(PARENT / "code"))
import data_geometry as dg  # noqa: E402
import analysis_runner as ar  # noqa: E402
import decoders as dec  # noqa: E402
import semantics as sem  # noqa: E402

# The parent experiment's run.py is loaded under a non-conflicting name.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "taskcore_frozen_parent", Path(os.environ.get("TASKCORE_PARENT_CODE", str(PARENT / "code/run.py"))))
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

RUNTIME = Path(os.environ.get("CORE_RESIDUAL_RUNTIME", str(REPO.parent / "core_residual_premise_runtime"))).resolve()
STAGE = "embedding_64d"
FOLDS = tuple(range(5))
FAMILIES = ("FULL", "G", "P_ALL", "P_NOT_U", "C_CURRENT", "NEITHER",
            "PCA_R", "SUPERVISED_DECISION_R", "FISHER_1D")
RESIDUALS = ("P_NOT_U", "NEITHER", "C_CURRENT")
SCOPES = ("TRAIN_GEOMETRY", "CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT")


def sha(path: Path) -> str:
    return dg.file_sha(path)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def unit(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(z))
    if norm < 1e-10:
        raise RuntimeError("zero task or discriminant direction")
    return z / norm


def cosine(a, b) -> float:
    return float(unit(a) @ unit(b))


def finite(value, label: str):
    if not np.isfinite(value).all():
        raise RuntimeError(f"nonfinite {label}")
    return value


def source_gate(fold: int):
    if fold not in FOLDS:
        raise ValueError(fold)
    lock = load(EXP / "protocol/SOURCE_PROVENANCE.json")
    if lock["source_commits"] != {"p_semantics": "8a8b708b30e19a20a83902d272fbe53fe8279f12",
                                    "taskcore": "d5184c5423806693a3b218f2d0329cd6b29d5d9a"}:
        raise RuntimeError("source commit lock drift")
    for relative, expected in lock["source_file_sha256"].items():
        path = REPO / relative
        if relative.startswith("experiments/persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1/outputs/"):
            path = PARENT_OUTPUT / Path(relative).name
        if relative.endswith("persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1/protocol/PROTOCOL_LOCK.json"):
            path = Path(os.environ.get("TASKCORE_PROTOCOL_SNAPSHOT", str(path)))
        if sha(path) != expected:
            raise RuntimeError(f"source file drift: {relative}")
    parent_recon = load(PARENT_OUTPUT / "PU_RECONSTRUCTION_AUDIT.json")
    if not parent_recon["all_G_exact"] or parent_recon["final_heldout_eeg_reads"]:
        raise RuntimeError("parent reconstruction gate")
    prior = tc.source_fold(fold)
    frozen = load(tc.RUNTIME / "reconstruction" / f"fold{fold}.json")
    if (not frozen["G_exact_source_geometry"] or frozen["final_heldout_eeg_reads"]
            or frozen["source_geometry_sha256"] != prior["stage_geometry_sha256"][STAGE]):
        raise RuntimeError("prior geometry mismatch")
    audit = load(PARENT_OUTPUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json")
    if sha(tc.RUNTIME / "reconstruction" / f"fold{fold}.json") != audit["fold_audit_sha256"]["reconstruction"][str(fold)]:
        raise RuntimeError("prior reconstruction runtime hash drift")
    return prior, frozen


def extract_train(fold: int):
    torch.set_num_threads(4)
    prior, frozen = source_gate(fold)
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    if sha(ckpt) != prior["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA drift")
    state = dg.model_sha(model)
    stages = pw.Stages(model, head, "EEGNet", device)
    train_h = ar.extract_stage(train, stages, STAGE)
    if dg.array_sha(train_h) != prior["train_feature_sha256"][STAGE]:
        raise RuntimeError("TRAIN features drift")
    if state != prior["model_state_sha256_before_after"] or dg.model_sha(model) != state:
        raise RuntimeError("neural parameter/BatchNorm drift")
    spec, gate = tc.gated_spec(fold, train_h, train, een)
    if gate["families"] != frozen["families"]:
        raise RuntimeError("source family support drift")
    return prior, frozen, een, up, train, context, model, stages, train_h, spec, state


def supervised_basis(train, h: np.ndarray, rank: int):
    weights, keys = [], []
    for subject in sorted(set(train.subject)):
        for session in sorted(set(train.session)):
            ix = (train.subject == subject) & (train.session == session)
            if set(np.unique(train.y[ix])) != {0, 1}:
                raise RuntimeError("TRAIN unit lacks binary classes")
            scale, classifier = dec.fit_linear(h[ix], train.y[ix])
            w = unit(classifier.coef_[0] / scale.scale_)
            weights.append(w)
            keys.append((subject, int(session)))
    matrix = np.stack(weights)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if rank > vt.shape[0] or singular[rank - 1] < 1e-8:
        raise RuntimeError("supervised decision SVD insufficient rank")
    q = vt[:rank].T.astype(np.float32)
    for k in range(rank):
        pivot = int(np.argmax(np.abs(q[:, k])))
        if q[pivot, k] < 0:
            q[:, k] *= -1
    return q, singular.tolist(), keys


def fisher_basis(train, h: np.ndarray):
    scale = StandardScaler().fit(h)
    model = LinearDiscriminantAnalysis(solver="svd")
    model.fit(scale.transform(h), train.y)
    w = unit(model.coef_[0] / scale.scale_)
    return w.astype(np.float32)[:, None]


def construct(fold: int) -> None:
    folder = RUNTIME / "construction" / f"fold{fold}"
    if folder.exists():
        raise FileExistsError(folder)
    prior, frozen, een, up, train, context, model, stages, h, spec, state = extract_train(fold)
    dims = frozen["families"]
    q = {"G": tc.family_basis(spec, dims["PROTECTED_G"]),
         "P_ALL": tc.family_basis(spec, dims["P_ALL"]),
         "P_NOT_U": tc.family_basis(spec, dims["P_NOT_U"]),
         "NEITHER": tc.family_basis(spec, dims["NEITHER"])}
    rank = q["G"].shape[1]
    mu = h.mean(0, dtype=np.float64).astype(np.float32)
    if dg.array_sha(mu, q["G"]) != prior["stage_geometry_sha256"][STAGE]:
        raise RuntimeError("G geometry not exact")
    covariance = np.cov((h - mu).astype(np.float64), rowvar=False)
    eig, vec = np.linalg.eigh(covariance)
    q["PCA_R"] = vec[:, np.argsort(eig)[::-1][:rank]].astype(np.float32)
    q["SUPERVISED_DECISION_R"], singular, unit_keys = supervised_basis(train, h - mu, rank)
    q["FISHER_1D"] = fisher_basis(train, h - mu)
    for name, basis in q.items():
        finite(basis, name)
        if np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))) > 1e-5:
            raise RuntimeError(f"nonorthonormal {name}")
    if dg.model_sha(model) != state:
        raise RuntimeError("neural parameter/BatchNorm changed during construction")
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "BASES.npz", mu=mu, **q)
    audit = {"schema": "CORE_RESIDUAL_CONSTRUCTION_FOLD_V1", "fold": fold,
             "source_geometry_sha256": prior["stage_geometry_sha256"][STAGE],
             "source_checkpoint_sha256": prior["checkpoint_sha256"],
             "source_train_feature_sha256": prior["train_feature_sha256"][STAGE],
             "model_state_sha256_before_after": state,
             "source_reconstruction_sha256": sha(tc.RUNTIME / "reconstruction" / f"fold{fold}.json"),
             "G_exact_source_geometry": True,
             "basis_sha256": {name: dg.array_sha(basis) for name, basis in q.items()},
             "rank": {name: int(basis.shape[1]) for name, basis in q.items()},
             "G_rank": rank, "FISHER_1D_rank_matched": False,
             "pca_fit_population": "TRAIN_GEOMETRY", "supervised_fit_population": "TRAIN_GEOMETRY",
             "supervised_unit": "subject_x_session", "supervised_unit_keys": unit_keys,
             "supervised_singular_spectrum": singular,
             "outer_rows_seen_at_construction": 0, "final_heldout_eeg_reads": 0,
             "basis_file_sha256": sha(folder / "BASES.npz")}
    write_new(folder / "AUDIT.json", audit)
    print(json.dumps({"fold": fold, "rank_G": rank, "status": "CONSTRUCTED"}), flush=True)


def features(h: np.ndarray, mu: np.ndarray, basis: dict[str, np.ndarray], name: str) -> np.ndarray:
    x = h - mu
    if name == "FULL":
        return np.ascontiguousarray(x, dtype=np.float32)
    if name == "C_CURRENT":
        q = basis["G"]
        return np.ascontiguousarray(x - (x @ q) @ q.T, dtype=np.float32)
    return np.ascontiguousarray(x @ basis[name], dtype=np.float32)


def view_all(populations, mu, basis, name):
    return {role: (pop, features(h, mu, basis, name)) for role, (pop, h) in populations.items()}


def fitted_centroid_decoder(train_pop, x, sessions, fit_session=None):
    cent, y, subject, session = dec.centroid_training(train_pop, x, sessions)
    fit = np.ones(len(y), dtype=bool) if fit_session is None else session == fit_session
    if not fit.any():
        raise RuntimeError("no TRAIN centroids for decoder")
    return dec.fit_linear(cent[fit], y[fit])


def predict(scale, model, x):
    # The immutable source decoder casts each probability batch to float32
    # before sklearn log_loss; preserve that exact NLL parameterization.
    return model.predict_proba(scale.transform(x)).astype(np.float32)


def score(y, probability):
    return dec.metrics(y, probability, np.array([0, 1], dtype=np.int64))


def subject_scores(pop, probability):
    rows = []
    for subject in sorted(set(pop.subject)):
        mask = pop.subject == subject
        m = score(pop.y[mask], probability[mask])
        rows.append({"subject": subject, **m})
    return rows


def decoder_protocol(fold, family, views, sessions):
    train, train_x = views["TRAIN_GEOMETRY"]
    rows, probabilities = [], {}
    for fit_session in (*sessions, None):
        scale, model = fitted_centroid_decoder(train, train_x, sessions, fit_session)
        for role in SCOPES:
            pop, x = views[role]
            if fit_session is None:
                masks = [("both", np.ones(len(pop.y), dtype=bool))]
                if role != "TRAIN_GEOMETRY":
                    masks += [(str(s), pop.session == s) for s in sessions]
                transfer = "cross_subject"
            else:
                target = next(s for s in sessions if s != fit_session)
                masks = [(str(target), pop.session == target)]
                transfer = "session"
            for eval_session, mask in masks:
                probability = predict(scale, model, x[mask])
                m = score(pop.y[mask], probability)
                rows.append({"fold": fold, "family": family, "rank": x.shape[1],
                             "fit_population": "TRAIN_GEOMETRY", "fit_unit": "subject_session_class_centroid",
                             "fit_session": str(fit_session) if fit_session is not None else "both",
                             "eval_population": role, "eval_session": eval_session,
                             "transfer_type": transfer, "C": 1.0, "outer_fit_rows": 0,
                             **m})
                if role == "OUTER_DEVELOPMENT" and fit_session is None and eval_session == "both":
                    probabilities["OUTER_DEVELOPMENT"] = probability
                if role == "CHECKPOINT_VALIDATION" and fit_session is None and eval_session == "both":
                    probabilities["CHECKPOINT_VALIDATION"] = probability
    return rows, probabilities


def relation_vectors(pop, x, sessions):
    grid = sem.centroid_grid(x, pop)
    return sem.baseline_relation(grid, sessions)


def geometry_rows(fold, family, views, sessions):
    train, tx = views["TRAIN_GEOMETRY"]
    _, train_d = relation_vectors(train, tx, sessions)
    train_subjects = sorted(set(train.subject))
    rows, subject_map = [], {}
    for role in SCOPES:
        pop, x = views[role]
        baseline, relation = relation_vectors(pop, x, sessions)
        subjects = sorted(set(pop.subject))
        for subject in subjects:
            refs = [s for s in train_subjects if role != "TRAIN_GEOMETRY" or s != subject]
            if not refs:
                raise RuntimeError("empty consensus reference")
            consensus = unit(np.mean([unit(train_d[(s, t)]) for s in refs for t in sessions], axis=0))
            task_unit = unit(np.mean([unit(relation[(subject, t)]) for t in sessions], axis=0))
            shared = float(np.mean([cosine(relation[(subject, t)], consensus) for t in sessions]))
            heterogeneity = 1 - cosine(task_unit, consensus)
            instability = 1 - cosine(relation[(subject, sessions[0])], relation[(subject, sessions[1])])
            context = cosine(baseline[(subject, sessions[0])], baseline[(subject, sessions[1])])
            cross = float(np.mean([cosine(relation[(subject, t)],
                             np.mean([train_d[(s, t)] for s in refs], axis=0)) for t in sessions]))
            row = {"fold": fold, "family": family, "population": role, "subject": subject,
                   "rank": x.shape[1], "shared_relation_alignment": shared,
                   "subject_task_heterogeneity": heterogeneity,
                   "session_task_instability": instability,
                   "subject_baseline_persistence": context,
                   "cross_subject_relation_cosine": cross,
                   "fit_population": "TRAIN_GEOMETRY_LOSO" if role == "TRAIN_GEOMETRY" else "TRAIN_GEOMETRY"}
            rows.append(row)
            if role == "OUTER_DEVELOPMENT":
                subject_map[subject] = row
    return rows, subject_map


def shared_fraction(fold, family, outer, x, sessions, ambient_basis):
    # All terms live in the original 64-D ambient embedding coordinates.
    _, relation = relation_vectors(outer, x, sessions)
    subjects = sorted(set(outer.subject))
    vectors = np.stack([[relation[(s, t)] @ ambient_basis.T if ambient_basis is not None
                         else relation[(s, t)] for t in sessions] for s in subjects]).astype(np.float64)
    direction = unit(np.mean([unit(v) for v in vectors.reshape(-1, vectors.shape[-1])], axis=0))
    shared_signal = float(np.square(vectors @ direction).mean())
    subject_mean = vectors.mean(axis=1)
    subject_variation = float(np.square(subject_mean - subject_mean.mean(axis=0)).sum(axis=1).mean())
    session_variation = float(np.square(vectors - subject_mean[:, None, :]).sum(axis=2).mean())
    denom = shared_signal + subject_variation + session_variation + 1e-12
    return {"fold": fold, "family": family, "population": "OUTER_DEVELOPMENT",
            "shared_signal": shared_signal, "subject_task_variation": subject_variation,
            "session_task_variation": session_variation, "shared_fraction": shared_signal / denom,
            "scope": "descriptive_only_ambient_64d_energy_decomposition"}


def variance_summary(fold, family, views, sessions):
    outer, ox = views["OUTER_DEVELOPMENT"]
    grid = sem.centroid_grid(ox, outer)
    rows = sem.variance_rows(grid, role="OUTER_DEVELOPMENT", fold=fold, stage=STAGE,
                             family=family, draw=-1, sessions=sessions)
    return {row["effect"]: row["fraction"] for row in rows}


def metric_row(rows, role, fit, eval_session="both"):
    matches = [r for r in rows if r["eval_population"] == role and r["fit_session"] == fit
               and r["eval_session"] == eval_session]
    if len(matches) != 1:
        raise RuntimeError(f"decoder row nonunique: {role} {fit} {eval_session}")
    return matches[0]


def summary_row(fold, family, rank, geometry, decoder, variance, sessions):
    outer = [r for r in geometry if r["population"] == "OUTER_DEVELOPMENT"]
    if not outer:
        raise RuntimeError("outer geometry absent")
    cross = metric_row(decoder, "OUTER_DEVELOPMENT", "both")
    forward = metric_row(decoder, "OUTER_DEVELOPMENT", str(sessions[0]), str(sessions[1]))
    mean = lambda key: float(np.mean([r[key] for r in outer]))
    return {"fold": fold, "family": family, "rank": rank,
            "cross_subject_relation_cosine": mean("cross_subject_relation_cosine"),
            "within_subject_cross_session_relation_cosine": 1 - mean("session_task_instability"),
            "shared_relation_alignment": mean("shared_relation_alignment"),
            "subject_task_heterogeneity": mean("subject_task_heterogeneity"),
            "session_task_instability": mean("session_task_instability"),
            "subject_baseline_persistence": mean("subject_baseline_persistence"),
            "class_variance_fraction": variance["class"],
            "subject_variance_fraction": variance["subject"],
            "frozen_cross_subject_BA": cross["BA"],
            "frozen_cross_subject_macro_F1": cross["macro_F1"],
            "frozen_cross_subject_NLL": cross["NLL"],
            "frozen_S1_to_S2_BA": forward["BA"],
            "fisher_rank_mismatch": family == "FISHER_1D"}


def paired_bootstrap(fold, comparator, metric, by_subject):
    subjects = sorted(by_subject)
    diffs = np.asarray([by_subject[s]["G"][metric] - by_subject[s][comparator][metric]
                        for s in subjects], dtype=np.float64)
    finite(diffs, f"bootstrap {metric}")
    rng = np.random.default_rng(dg.seed("core-residual-subject-bootstrap", fold, comparator, metric))
    draw = rng.integers(0, len(subjects), size=(20_000, len(subjects)))
    dist = diffs[draw].mean(axis=1)
    return {"fold": fold, "contrast": f"G_MINUS_{comparator}", "metric": metric,
            "subject_equal_difference": float(diffs.mean()),
            "bootstrap_CI_low": float(np.quantile(dist, .025)),
            "bootstrap_CI_high": float(np.quantile(dist, .975)),
            "bootstrap_draws": 20_000, "biological_subjects": len(subjects)}


def joint_views(populations, mu, basis, residual):
    return {role: (pop, np.concatenate((features(h, mu, basis, "G"),
                                        features(h, mu, basis, residual)), axis=1))
            for role, (pop, h) in populations.items()}


def trial_error_rows(fold, outer, probabilities, sessions):
    rows, oracle = [], []
    g_pred = probabilities["G"].argmax(axis=1)
    full_pred = probabilities["FULL"].argmax(axis=1)
    for residual in RESIDUALS:
        r_pred = probabilities[residual].argmax(axis=1)
        gc = g_pred == outer.y
        rc = r_pred == outer.y
        fc = full_pred == outer.y
        for i in range(len(outer.y)):
            rows.append({"fold": fold, "residual": residual, "row_type": "OUTER_TRIAL",
                         "subject": str(outer.subject[i]), "session": int(outer.session[i]),
                         "trial_index_within_capped_fold": i, "class": int(outer.y[i]),
                         "G_correct": bool(gc[i]), "residual_correct": bool(rc[i]),
                         "FULL_correct": bool(fc[i]),
                         "G_correct_R_correct": bool(gc[i] and rc[i]),
                         "G_correct_R_wrong": bool(gc[i] and not rc[i]),
                         "G_wrong_R_correct": bool(not gc[i] and rc[i]),
                         "G_wrong_R_wrong": bool(not gc[i] and not rc[i])})
        for subject in sorted(set(outer.subject)):
            for session in sessions:
                mask = (outer.subject == subject) & (outer.session == session)
                wrong = mask & ~gc
                correct = mask & gc
                if not wrong.any() or not correct.any():
                    raise RuntimeError("rescue/redundancy denominator zero")
                oracle_prediction_correct = gc[mask] | rc[mask]
                # The oracle conditions on ground truth after two predictions.
                oracle_prob = np.where(oracle_prediction_correct, outer.y[mask], 1 - outer.y[mask])
                oracle.append({"fold": fold, "residual": residual, "subject": str(subject),
                               "session": int(session), "oracle_union_BA": float(balanced_accuracy_score(
                                   outer.y[mask], oracle_prob)),
                               "label": "NONDEPLOYABLE_COMPLEMENTARITY_UPPER_BOUND",
                               "rescue_fraction": float(rc[wrong].mean()),
                               "redundancy_fraction": float(rc[correct].mean()),
                               "G_wrong_rows": int(wrong.sum()), "G_correct_rows": int(correct.sum()),
                               "rows": int(mask.sum())})
    return rows, oracle


def margin(probability):
    # Source-compatible float32 decoder probabilities can round to exactly 1.
    # Promote before clipping so the logit bound remains representable.
    p = np.clip(probability[:, 1].astype(np.float64), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def stacked_fusion(fold, residual, train, outer, train_g, train_r, outer_g, outer_r, sessions):
    splitter = GroupKFold(n_splits=5)
    oof = np.full((len(train.y), 2), np.nan, dtype=np.float64)
    for fit, eva in splitter.split(train_g, train.y, train.subject):
        fit_pop = dg.Population(train.x[fit], train.y[fit], train.subject[fit], train.session[fit], "TRAIN_GEOMETRY")
        for col, x in enumerate((train_g, train_r)):
            scale, model = fitted_centroid_decoder(fit_pop, x[fit], sessions)
            oof[eva, col] = margin(predict(scale, model, x[eva]))
    finite(oof, "OOF TRAIN margins")
    oof_subjects = len(set(train.subject))
    if oof_subjects < 5:
        raise RuntimeError("insufficient TRAIN subjects for grouped OOF")
    stack_scale, stack = dec.fit_linear(oof, train.y)
    held_margins = []
    for tx, ox in ((train_g, outer_g), (train_r, outer_r)):
        scale, model = fitted_centroid_decoder(train, tx, sessions)
        held_margins.append(margin(predict(scale, model, ox)))
    held = np.stack(held_margins, axis=1)
    probabilities = stack.predict_proba(stack_scale.transform(held))
    result = score(outer.y, probabilities)
    rows = [{"fold": fold, "residual": residual, "scope": "OUTER_DEVELOPMENT_BOTH_SESSIONS",
             "fit_population": "TRAIN_GEOMETRY", "OOF_group": "biological_subject",
             "OOF_folds": 5, "stacker_inputs": "G_margin,residual_margin",
             "stacker_C": 1.0, "outer_fit_rows": 0, **result}]
    rows += [{"fold": fold, "residual": residual, "scope": "OUTER_SUBJECT",
              "subject": r["subject"], "fit_population": "TRAIN_GEOMETRY",
              "OOF_group": "biological_subject", "OOF_folds": 5,
              "stacker_inputs": "G_margin,residual_margin", "outer_fit_rows": 0,
              **{k: r[k] for k in ("BA", "macro_F1", "NLL", "rows")}}
             for r in subject_scores(outer, probabilities)]
    return rows


def capacity_views(populations, mu, covariance, dimension):
    eig, vec = np.linalg.eigh(covariance)
    q = vec[:, np.argsort(eig)[::-1][:min(dimension, 64)]].astype(np.float32)
    result = {}
    for role, (pop, h) in populations.items():
        x = (h - mu) @ q
        if dimension > 64:
            x = np.pad(x, ((0, 0), (0, dimension - 64)))
        result[role] = (pop, np.ascontiguousarray(x, dtype=np.float32))
    return result, min(dimension, 64)


def source_numeric_gate(fold, summary):
    with (PARENT_OUTPUT / "PU_FAMILY_SUMMARY.csv").open("r", encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    mapping = {"G": "PROTECTED_G", "P_ALL": "P_ALL", "P_NOT_U": "P_NOT_U",
               "NEITHER": "NEITHER", "FULL": "FULL", "C_CURRENT": "C_CURRENT"}
    for name, old_name in mapping.items():
        old = [r for r in source if int(r["fold"]) == fold and r["family"] == old_name]
        new = [r for r in summary if r["family"] == name]
        if len(old) != 1 or len(new) != 1:
            raise RuntimeError("source summary row missing")
        if name == "NEITHER" and old[0].get("status") == "NOT_ESTIMABLE":
            continue
        for key, new_key in (("cross_subject_relation_cosine", "cross_subject_relation_cosine"),
                             ("cross_subject_decoder_BA", "frozen_cross_subject_BA"),
                             ("cross_subject_decoder_NLL", "frozen_cross_subject_NLL")):
            if abs(float(old[0][key]) - float(new[0][new_key])) > 2e-5:
                raise RuntimeError(f"source semantic/decoder drift {fold} {name} {key}")


def csv_new(path: Path, rows: list[dict]):
    if not rows:
        raise RuntimeError(f"empty output {path.name}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(fold: int):
    target = RUNTIME / "evaluation" / f"fold{fold}"
    if target.exists():
        raise FileExistsError(target)
    # No held population is read until all five TRAIN-only constructions pass.
    for f in FOLDS:
        audit_path = RUNTIME / "construction" / f"fold{f}" / "AUDIT.json"
        audit = load(audit_path)
        basis_path = audit_path.parent / "BASES.npz"
        if (not audit["G_exact_source_geometry"] or audit["outer_rows_seen_at_construction"] != 0
                or audit["final_heldout_eeg_reads"] != 0 or sha(basis_path) != audit["basis_file_sha256"]):
            raise RuntimeError(f"all-fold construction gate failure {f}")
    prior, frozen, een, up, train, context, model, stages, train_h, spec, state = extract_train(fold)
    audit = load(RUNTIME / "construction" / f"fold{fold}" / "AUDIT.json")
    with np.load(RUNTIME / "construction" / f"fold{fold}" / "BASES.npz", allow_pickle=False) as artifact:
        mu = artifact["mu"]
        basis = {name: artifact[name] for name in artifact.files if name != "mu"}
    if dg.array_sha(mu, basis["G"]) != audit["source_geometry_sha256"]:
        raise RuntimeError("loaded G geometry differs from source")
    if any(dg.array_sha(q) != audit["basis_sha256"][name] for name, q in basis.items()):
        raise RuntimeError("constructed basis content SHA mismatch")
    if not np.array_equal(mu, train_h.mean(0, dtype=np.float64).astype(np.float32)):
        raise RuntimeError("TRAIN mean drift")
    held = {role: dg.load_held(role, context, een, up) for role in SCOPES[1:]}
    populations = {"TRAIN_GEOMETRY": (train, train_h)}
    for role, pop in held.items():
        capped = ar.cap_held(pop, fold, een)
        populations[role] = (capped, ar.extract_stage(capped, stages, STAGE))
        if set(capped.subject) != set(context["ids"][role]):
            raise RuntimeError("held subject role drift")
    if dg.model_sha(model) != state or state != audit["model_state_sha256_before_after"]:
        raise RuntimeError("neural parameter/BatchNorm state changed")
    sessions = context["sessions"]
    result = {name: [] for name in (
        "TASK_GEOMETRY_DECOMPOSITION", "TASK_GEOMETRY_SHARED_FRACTION",
        "SUBSPACE_FROZEN_DECODER_TRANSFER", "GEOMETRY_VS_PREDICTION_SUMMARY",
        "G_VS_PCA_CONTRAST", "G_VS_SUPERVISED_CONTRAST",
        "RESIDUAL_ONLY_DECODER_RESULTS", "RESIDUAL_ERROR_COMPLEMENTARITY",
        "RESIDUAL_ORACLE_UNION", "CORE_RESIDUAL_JOINT_DECODER",
        "JOINT_CAPACITY_CONTROL", "STACKED_LOGIT_FUSION")}
    primary_probs, subject_values, subject_geometry, views_by_name = {}, {}, {}, {}
    outer = populations["OUTER_DEVELOPMENT"][0]
    for name in FAMILIES:
        views = view_all(populations, mu, basis, name)
        views_by_name[name] = views
        geometry, subject_geometry[name] = geometry_rows(fold, name, views, sessions)
        decoder, predictions = decoder_protocol(fold, name, views, sessions)
        variance = variance_summary(fold, name, views, sessions)
        summary = summary_row(fold, name, views["TRAIN_GEOMETRY"][1].shape[1],
                              geometry, decoder, variance, sessions)
        result["TASK_GEOMETRY_DECOMPOSITION"].extend(geometry)
        result["SUBSPACE_FROZEN_DECODER_TRANSFER"].extend(decoder)
        result["GEOMETRY_VS_PREDICTION_SUMMARY"].append(summary)
        ambient = basis[name] if name in basis else None
        result["TASK_GEOMETRY_SHARED_FRACTION"].append(shared_fraction(
            fold, name, outer, views["OUTER_DEVELOPMENT"][1], sessions, ambient))
        primary_probs[name] = predictions["OUTER_DEVELOPMENT"]
        by_subj = {row["subject"]: row for row in subject_scores(outer, primary_probs[name])}
        subject_values[name] = {s: {**subject_geometry[name][s],
                                    "frozen_cross_subject_BA": by_subj[s]["BA"],
                                    "frozen_cross_subject_NLL": by_subj[s]["NLL"]}
                                for s in by_subj}
        if name in ("G", *RESIDUALS):
            result["RESIDUAL_ONLY_DECODER_RESULTS"].append({**summary, "population": "OUTER_DEVELOPMENT"})
    source_numeric_gate(fold, result["GEOMETRY_VS_PREDICTION_SUMMARY"])

    # Source-convention 100 deterministic, rank-matched random subspaces.
    for draw in range(100):
        qb = dg.random_subspace(64, basis["G"].shape[1], fold, STAGE, draw)
        random_basis = {**basis, "RANK_RANDOM_R": qb}
        views = view_all(populations, mu, random_basis, "RANK_RANDOM_R")
        geometry, _ = geometry_rows(fold, "RANK_RANDOM_R", views, sessions)
        decoder, _ = decoder_protocol(fold, "RANK_RANDOM_R", views, sessions)
        variance = variance_summary(fold, "RANK_RANDOM_R", views, sessions)
        summary = summary_row(fold, "RANK_RANDOM_R", qb.shape[1], geometry, decoder, variance, sessions)
        result["TASK_GEOMETRY_DECOMPOSITION"].extend({**r, "random_draw": draw} for r in geometry)
        result["SUBSPACE_FROZEN_DECODER_TRANSFER"].extend({**r, "random_draw": draw} for r in decoder)
        result["GEOMETRY_VS_PREDICTION_SUMMARY"].append({**summary, "random_draw": draw})

    for comparator, output in (("PCA_R", "G_VS_PCA_CONTRAST"),
                               ("SUPERVISED_DECISION_R", "G_VS_SUPERVISED_CONTRAST")):
        by_subject = {subject: {"G": subject_values["G"][subject],
                                comparator: subject_values[comparator][subject]}
                      for subject in sorted(subject_values["G"])}
        for metric in ("shared_relation_alignment", "cross_subject_relation_cosine",
                       "subject_task_heterogeneity", "session_task_instability",
                       "frozen_cross_subject_BA", "frozen_cross_subject_NLL"):
            result[output].append(paired_bootstrap(fold, comparator, metric, by_subject))

    error_rows, oracle_rows = trial_error_rows(fold, outer, primary_probs, sessions)
    result["RESIDUAL_ERROR_COMPLEMENTARITY"].extend(error_rows)
    result["RESIDUAL_ORACLE_UNION"].extend(oracle_rows)
    train_cov = np.cov((train_h - mu).astype(np.float64), rowvar=False)
    for residual in RESIDUALS:
        joint = joint_views(populations, mu, basis, residual)
        joint_rows, _ = decoder_protocol(fold, "G_PLUS_" + residual, joint, sessions)
        result["CORE_RESIDUAL_JOINT_DECODER"].extend(joint_rows)
        dimension = joint["TRAIN_GEOMETRY"][1].shape[1]
        capacity, effective_rank = capacity_views(populations, mu, train_cov, dimension)
        control_rows, _ = decoder_protocol(fold, "TRAIN_PCA_DIMENSION_CONTROL_" + residual,
                                           capacity, sessions)
        result["JOINT_CAPACITY_CONTROL"].extend({**r, "joint_dimension": dimension,
                                                 "capacity_effective_rank": effective_rank,
                                                 "zero_padded_if_D_gt_64": dimension > 64,
                                                 "PCA_fit_population": "TRAIN_GEOMETRY"}
                                                for r in control_rows)
        result["STACKED_LOGIT_FUSION"].extend(stacked_fusion(
            fold, residual, train, outer,
            views_by_name["G"]["TRAIN_GEOMETRY"][1],
            views_by_name[residual]["TRAIN_GEOMETRY"][1],
            views_by_name["G"]["OUTER_DEVELOPMENT"][1],
            views_by_name[residual]["OUTER_DEVELOPMENT"][1], sessions))
    if dg.model_sha(model) != state:
        raise RuntimeError("neural parameter/BatchNorm changed during evaluation")
    target.mkdir(parents=True, exist_ok=False)
    index = {}
    for name, rows in result.items():
        path = target / (name + ".csv")
        csv_new(path, rows)
        index[path.name] = {"sha256": sha(path), "rows": len(rows)}
    write_new(target / "AUDIT.json", {"schema": "CORE_RESIDUAL_EVALUATION_FOLD_V1",
        "fold": fold, "construction_audit_sha256": sha(RUNTIME / "construction" / f"fold{fold}" / "AUDIT.json"),
        "checkpoint_sha256": prior["checkpoint_sha256"], "model_state_sha256_before_after": state,
        "source_family_numeric_reproduction": True, "fit_population": "TRAIN_GEOMETRY",
        "outer_fit_rows": 0, "checkpoint_validation_fit_rows": 0,
        "random_draws": 100, "final_heldout_eeg_reads": 0,
        "files": index})
    print(json.dumps({"fold": fold, "status": "EVALUATED", "files": len(index)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("construct", "evaluate"))
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    args = parser.parse_args()
    if args.phase == "construct":
        construct(args.fold)
    else:
        evaluate(args.fold)
