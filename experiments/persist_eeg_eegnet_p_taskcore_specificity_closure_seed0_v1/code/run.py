"""Fail-closed frozen EEGNet P/U reconstruction and specificity audit.

Run ``reconstruct --fold F`` for every fold before any ``evaluate``.  The
source selector already measured TRAIN utility for *all* atomic blocks; only
its final conjunction with persistence was restricted.  This audit reuses
those immutable per-block measurements, never outer labels for selection.
"""
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

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
SOURCE = REPO / "experiments/persist_eeg_eegnet_p_semantics_canonical_geometry_seed0_v1"
sys.path.insert(0, str(SOURCE / "code"))
import data_geometry as dg  # noqa: E402
import decoders as dec  # noqa: E402
import permutations as perm  # noqa: E402
import semantics as sem  # noqa: E402

RUNTIME = Path(os.environ.get("P_TASKCORE_RUNTIME", str(REPO.parent / "p_taskcore_specificity_runtime"))).resolve()
EXPECTED_COMMIT = "8a8b708b30e19a20a83902d272fbe53fe8279f12"
STAGE = "embedding_64d"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rows_new(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def source_fold(fold: int) -> dict:
    lock = read(EXP / "protocol/SOURCE_PROVENANCE.json")
    if lock["source_commit"] != EXPECTED_COMMIT:
        raise RuntimeError("source commit lock mismatch")
    path = dg.RUNTIME / "geometry" / f"fold{fold}_seed0" / "GEOMETRY_PROVENANCE.json"
    if dg.file_sha(path) != lock["fold_geometry_source_sha256"][str(fold)]:
        raise RuntimeError("frozen source geometry provenance hash mismatch")
    item = read(path)
    if item["fold"] != fold or item["final_heldout_eeg_reads"] != 0:
        raise RuntimeError("source fold or heldout mismatch")
    return item


def select_independently(spec: dict, prior: dict) -> tuple[dict, dict]:
    """Evaluate P and U separately on every atomic source block."""
    old = prior["final_P_selection_assignment"]
    if len(old) != len(spec["blocks"]):
        raise RuntimeError("source atomic block count changed")
    decisions, families = [], {key: [] for key in ("P_ALL", "U_ALL", "PROTECTED_G", "P_NOT_U", "U_NOT_P", "NEITHER")}
    for bi, (block, row, support) in enumerate(zip(spec["blocks"], old, spec["support"])):
        if bi != row["block"] or len(block) != row["dimensions"] or bi != support["block"]:
            raise RuntimeError("atomic block partition mismatch")
        p = bool(support["rho"] > support["null_p95"])
        u = bool(row["absolute_CI_low"] > 0 and row["excess_CI_low"] > 0)
        if p != row["persistence_supported"]:
            raise RuntimeError("persistence reconstruction mismatch")
        if (p and u) != row["protected"]:
            raise RuntimeError("Protected conjunction mismatch")
        for name, include in (("P_ALL", p), ("U_ALL", u), ("PROTECTED_G", p and u),
                              ("P_NOT_U", p and not u), ("U_NOT_P", u and not p),
                              ("NEITHER", not (p or u))):
            if include:
                families[name].extend(block)
        decisions.append({"block": bi, "coordinates": block, "rank": len(block),
                          "rho": support["rho"], "null_p95": support["null_p95"],
                          "P_support": p, "absolute_CI_low": row["absolute_CI_low"],
                          "excess_CI_low": row["excess_CI_low"], "U_support": u,
                          "G_support": p and u})
    families = {key: sorted(value) for key, value in families.items()}
    if families["PROTECTED_G"] != prior["final_P_selected_coordinates"]:
        raise RuntimeError("new G differs from source Protected coordinate IDs")
    return {"blocks": decisions, "families": families}, families


def family_basis(spec: dict, dims: list[int]) -> np.ndarray | None:
    if not dims:
        return None
    raw = ((spec["directions"][:, dims].T * spec["scale"][None, :]) @ spec["basis"].T).T
    return dg.orthonormal(raw, len(dims))


def reconstruct(fold: int) -> None:
    target = RUNTIME / "reconstruction" / f"fold{fold}.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(4)
    prior = source_fold(fold)
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    if dg.file_sha(ckpt) != prior["checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint hash mismatch")
    state = dg.model_sha(model)
    stages = pw.Stages(model, head, "EEGNet", device)
    features, _ = dg.extract(train, stages)
    h = features[STAGE]
    if dg.array_sha(h) != prior["train_feature_sha256"][STAGE]:
        raise RuntimeError("TRAIN feature hash mismatch")
    spec = een.spectrum(h, train.y, train.subject, train.session, "OpenBMI_MI", "EEGNet", fold)
    if dg.array_sha(spec["mean"]) != prior["final_whitening_mean_sha256"]:
        raise RuntimeError("whitening mean mismatch")
    if dg.array_sha(spec["basis"], spec["scale"], spec["directions"]) != prior["final_whitening_basis_sha256"]:
        raise RuntimeError("whitening basis mismatch")
    decisions, families = select_independently(spec, prior)
    qg = family_basis(spec, families["PROTECTED_G"])
    artifact = np.load(dg.RUNTIME / "geometry" / f"fold{fold}_seed0" / f"{STAGE}.npz", allow_pickle=False)
    if dg.array_sha(artifact["mu"], artifact["q"]) != prior["stage_geometry_sha256"][STAGE]:
        raise RuntimeError("source geometry artifact hash mismatch")
    if dg.array_sha(h.mean(0, dtype=np.float64).astype(np.float32), qg) != prior["stage_geometry_sha256"][STAGE]:
        raise RuntimeError("reconstructed G differs from source Protected geometry")
    if dg.model_sha(model) != state or state != prior["model_state_sha256_before_after"]:
        raise RuntimeError("neural parameter or BatchNorm state changed")
    result = {"schema": "PU_RECONSTRUCTION_FOLD_V1", "fold": fold, **decisions,
              "ranks": {name: len(dims) for name, dims in families.items()},
              "active_rank": int(spec["rank"]), "checkpoint_sha256": dg.file_sha(ckpt),
              "source_geometry_sha256": prior["stage_geometry_sha256"][STAGE],
              "G_exact_source_coordinates": True, "G_exact_source_geometry": True,
              "P_uses_utility": False, "U_uses_persistence": False,
              "utility_all_blocks_measured_in_source": True,
              "train_feature_sha256": dg.array_sha(h), "model_state_sha256_before_after": state,
              "outer_development_read_before_reconstruction": False,
              "checkpoint_validation_read_before_reconstruction": False,
              "final_heldout_eeg_reads": 0}
    write_new(target, result)
    print(json.dumps({"fold": fold, "ranks": result["ranks"], "status": "PASS"}), flush=True)


def gated_spec(fold: int, h: np.ndarray, train, een):
    for k in range(5):
        check = read(RUNTIME / "reconstruction" / f"fold{k}.json")
        if not check["G_exact_source_geometry"] or check["final_heldout_eeg_reads"] != 0:
            raise RuntimeError("all-fold reconstruction gate failed")
    prior = source_fold(fold)
    if dg.array_sha(h) != prior["train_feature_sha256"][STAGE]:
        raise RuntimeError("TRAIN feature drift")
    spec = een.spectrum(h, train.y, train.subject, train.session, "OpenBMI_MI", "EEGNet", fold)
    if dg.array_sha(spec["basis"], spec["scale"], spec["directions"]) != prior["final_whitening_basis_sha256"]:
        raise RuntimeError("whitening drift")
    return spec, read(RUNTIME / "reconstruction" / f"fold{fold}.json")


def complement_basis(q: np.ndarray) -> np.ndarray:
    # Full QR of a rank-r basis yields an exact orthogonal complement.
    return np.linalg.qr(q.astype(np.float64), mode="complete")[0][:, q.shape[1]:].astype(np.float32)


def metric_pair(train_x, train_pop, held_x, held_pop, sessions):
    train_grid = sem.centroid_grid(train_x, train_pop)
    held_grid = sem.centroid_grid(held_x, held_pop)
    _, td = sem.baseline_relation(train_grid, sessions)
    _, hd = sem.baseline_relation(held_grid, sessions)
    values = []
    for s in sorted(set(held_pop.subject)):
        for t in sessions:
            ref = np.mean([td[(j, t)] for j in sorted(set(train_pop.subject))], axis=0)
            values.append(sem.sim(hd[(s, t)], ref)["cosine"])
    return float(np.mean(values))


def native_views(populations: dict, mu: np.ndarray, q: np.ndarray, transform=None):
    out = {}
    for role, (pop, h) in populations.items():
        x = (h - mu) @ q
        if transform is not None:
            x = x @ transform
        out[role] = (pop, np.ascontiguousarray(x, dtype=np.float32))
    return out


def exact_family(fold, name, views, sessions, *, expensive=True):
    """Source semantic and frozen-decoder protocol on selected coordinates."""
    result = {key: [] for key in ("baseline", "within", "cross", "variance", "decoder", "probes")}
    grid = {role: sem.centroid_grid(x, pop) for role, (pop, x) in views.items()}
    relations = {role: sem.baseline_relation(g, sessions) for role, g in grid.items()}
    train_d = relations["TRAIN_GEOMETRY"][1]
    for role, (b, d) in relations.items():
        kw = dict(role=role, fold=fold, stage=STAGE, family=name, draw=-1, sessions=sessions)
        result["baseline"].extend(sem.persistence_rows(b, kind="class_common_baseline", **kw))
        result["within"].extend(sem.persistence_rows(d, kind="class_relation", **kw))
        result["cross"].extend(sem.cross_subject_rows(train_d, d, **kw))
        result["variance"].extend(sem.variance_rows(grid[role], **kw))
    zero = np.zeros(views["TRAIN_GEOMETRY"][1].shape[1], dtype=np.float32)
    eye = np.eye(len(zero), dtype=np.float32)
    geometry = {"mu": zero, "q": eye}
    a, b, _ = dec.decoder_rows(fold, STAGE, geometry, views, sessions, "P", -1)
    result["decoder"] = [{**row, "family": name, "transfer_type": "session"} for row in a] + [
        {**row, "family": name, "transfer_type": "cross_subject"} for row in b]
    if expensive:
        result["probes"] = [{**row, "family": name} for row in dec.probe_rows(
            fold, STAGE, geometry, views, sessions, "P", -1)]
    return result


def held_summary(fold, name, views, sessions):
    train, tx = views["TRAIN_GEOMETRY"]
    outer, ox = views["OUTER_DEVELOPMENT"]
    relation = metric_pair(tx, train, ox, outer, sessions)
    zero = np.zeros(tx.shape[1], dtype=np.float32)
    eye = np.eye(len(zero), dtype=np.float32)
    geometry = {"mu": zero, "q": eye}
    a, b, _ = dec.decoder_rows(fold, STAGE, geometry, views, sessions, "P", -1)
    cross = next(r for r in b if r["eval_population"] == "OUTER_DEVELOPMENT" and r["eval_session"] == "both")
    s1, s2 = sessions
    forward = next(r for r in a if r["eval_population"] == "OUTER_DEVELOPMENT" and r["fit_session"] == s1)
    reverse = next(r for r in a if r["eval_population"] == "OUTER_DEVELOPMENT" and r["fit_session"] == s2)
    grid = sem.centroid_grid(ox, outer)
    _, d = sem.baseline_relation(grid, sessions)
    within = float(np.mean([sem.sim(d[(s, s1)], d[(s, s2)])["cosine"] for s in sorted(set(outer.subject))]))
    var = sem.variance_rows(grid, role="OUTER_DEVELOPMENT", fold=fold, stage=STAGE,
                            family=name, draw=-1, sessions=sessions)
    effects = {r["effect"]: r["fraction"] for r in var}
    return {"fold": fold, "family": name, "rank": tx.shape[1], "cross_subject_relation_cosine": relation,
            "within_subject_relation_cosine": within, "cross_subject_decoder_BA": cross["BA"],
            "cross_subject_decoder_macro_F1": cross["macro_F1"], "cross_subject_decoder_NLL": cross["NLL"],
            "S1_to_S2_BA": forward["BA"], "S2_to_S1_BA": reverse["BA"],
            "class_variance_fraction": effects["class"], "subject_variance_fraction": effects["subject"]}


def control_pool(fold, train_h, mu, qg, *, pool_size=2000, retain=20):
    """All matching uses unlabeled TRAIN embeddings only."""
    x = (train_h - mu).astype(np.float64)
    covariance = x.T @ x / max(len(x) - 1, 1)
    target_cov = qg.T @ covariance @ qg
    target_lambda = np.linalg.eigvalsh(target_cov)[::-1]
    target_trace = float(np.trace(target_cov))
    eigvals, eigvecs = np.linalg.eigh(covariance)
    pca = eigvecs[:, np.argsort(eigvals)[::-1][:qg.shape[1]]].astype(np.float32)
    rng = np.random.default_rng(dg.seed("taskcore-matched-pool", fold, 0))
    rows = []
    bases = []
    eps = 1e-12
    for draw in range(pool_size):
        q, _ = np.linalg.qr(rng.standard_normal((x.shape[1], qg.shape[1])), mode="reduced")
        c = q.T @ covariance @ q
        lam = np.linalg.eigvalsh(c)[::-1]
        trace = float(np.trace(c))
        energy_gap = abs(float(np.log((trace + eps) / (target_trace + eps))))
        shape_gap = float(np.sqrt(np.mean((np.log(lam + eps) - np.log(target_lambda + eps)) ** 2)))
        overlap = float(np.square(qg.T @ q).sum() / qg.shape[1])
        rows.append({"fold": fold, "pool_index": draw, "rank": qg.shape[1], "energy_ratio": trace / target_trace,
                     "energy_log_gap": energy_gap, "spectrum_log_RMS": shape_gap,
                     "covariance_score": shape_gap + energy_gap, "overlap_with_G": overlap})
        bases.append(q.astype(np.float32))
    energy_ids = sorted(range(pool_size), key=lambda i: (rows[i]["energy_log_gap"], i))[:retain]
    cov_ids = sorted(range(pool_size), key=lambda i: (rows[i]["covariance_score"], i))[:retain]
    for i, row in enumerate(rows):
        row["selected_energy"] = i in energy_ids
        row["selected_covariance"] = i in cov_ids
    # Deterministic low-overlap diagnostic: no OUTER quantity enters this ranking.
    low_energy = sorted(range(pool_size), key=lambda i: (rows[i]["overlap_with_G"], rows[i]["energy_log_gap"], i))[:retain]
    for i in low_energy:
        rows[i]["low_overlap_descriptive_subset"] = True
    return pca, rows, bases, energy_ids, cov_ids, target_lambda


def synthetic_map(train_h, mu, qr, target_lambda):
    z = (train_h - mu) @ qr
    c = np.cov(z.astype(np.float64), rowvar=False)
    lam, vec = np.linalg.eigh(c)
    if np.min(lam) <= 1e-12 or np.min(target_lambda) <= 1e-12:
        raise RuntimeError("synthetic covariance map not invertible")
    # Cov(z A)=diag(target_lambda). This is a coordinate transform, not an
    # orthogonal projection.  Its use is descriptive confound control only.
    a = vec @ np.diag(np.sqrt(target_lambda / lam))
    return a.astype(np.float32)


def evaluate(fold: int) -> None:
    target = RUNTIME / os.environ.get("P_TASKCORE_EVAL_FOLDER", "evaluation") / f"fold{fold}"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(4)
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    state = dg.model_sha(model)
    stages = pw.Stages(model, head, "EEGNet", device)
    train_h = dg.extract(train, stages)[0][STAGE]
    spec, gate = gated_spec(fold, train_h, train, een)
    if dg.file_sha(ckpt) != gate["checkpoint_sha256"] or state != gate["model_state_sha256_before_after"]:
        raise RuntimeError("evaluation checkpoint/model drift")
    # This is the first held-role access, strictly after the all-fold gate.
    held = {role: dg.load_held(role, context, een, up) for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT")}
    from analysis_runner import cap_held, extract_stage
    populations = {"TRAIN_GEOMETRY": (train, train_h)}
    for role, pop in held.items():
        capped = cap_held(pop, fold, een)
        populations[role] = (capped, extract_stage(capped, stages, STAGE))
    if any(set(pop.subject) != set(context["ids"][role]) for role, (pop, _) in populations.items()):
        raise RuntimeError("role inventory mismatch")
    if dg.model_sha(model) != state:
        raise RuntimeError("neural parameter/BatchNorm change")
    mu = train_h.mean(0, dtype=np.float64).astype(np.float32)
    primary = {}
    for name, dims in gate["families"].items():
        if not dims:
            primary[name] = None
            continue
        primary[name] = family_basis(spec, dims)
    qg = primary["PROTECTED_G"]
    if dg.array_sha(mu, qg) != gate["source_geometry_sha256"]:
        raise RuntimeError("G geometry drift")
    primary["FULL"] = np.eye(len(mu), dtype=np.float32)
    primary["C_CURRENT"] = complement_basis(qg)
    outputs = {name: [] for name in ("PU_BASELINE_PERSISTENCE", "PU_TASK_RELATION_WITHIN_SUBJECT",
                                   "PU_TASK_RELATION_CROSS_SUBJECT", "PU_FROZEN_DECODER_TRANSFER",
                                   "PU_SEMANTIC_PROBES", "PU_VARIANCE_DECOMPOSITION",
                                   "PROTECTED_SPECIFICITY_CONTROLS", "CONTROL_MATCHING_AUDIT",
                                   "PU_PERMUTATION_CONTROLS")}
    summaries = []
    for name, q in primary.items():
        if q is None:
            summaries.append({"fold": fold, "family": name, "rank": 0, "status": "NOT_ESTIMABLE"})
            continue
        if name == "C_CURRENT":
            # Match the source decoder exactly: it standardizes the ambient
            # 64-D residual featurewise.  Rotating into an orthogonal 60-D
            # complement preserves cosines but changes L2 regularization.
            views = {}
            for role, (pop, h) in populations.items():
                centered = h - mu
                residual = centered - (centered @ qg) @ qg.T
                views[role] = (pop, np.ascontiguousarray(residual, dtype=np.float32))
        else:
            views = native_views(populations, mu, q)
        details = exact_family(fold, name, views, context["sessions"])
        for key, dest in (("baseline", "PU_BASELINE_PERSISTENCE"), ("within", "PU_TASK_RELATION_WITHIN_SUBJECT"),
                          ("cross", "PU_TASK_RELATION_CROSS_SUBJECT"), ("decoder", "PU_FROZEN_DECODER_TRANSFER"),
                          ("probes", "PU_SEMANTIC_PROBES"), ("variance", "PU_VARIANCE_DECOMPOSITION")):
            outputs[dest].extend(details[key])
        result_summary = held_summary(fold, name, views, context["sessions"])
        if name == "C_CURRENT":
            result_summary["rank"] = q.shape[1]
        summaries.append(result_summary)
        if name in ("P_ALL", "U_ALL", "PROTECTED_G", "P_NOT_U", "U_NOT_P"):
            train_pop, tx = views["TRAIN_GEOMETRY"]
            outer_pop, ox = views["OUTER_DEVELOPMENT"]
            for role, (pop, x) in views.items():
                grid = sem.centroid_grid(x, pop)
                baseline, relation = sem.baseline_relation(grid, context["sessions"])
                for quantity, vectors in (("baseline", baseline), ("task_relation", relation)):
                    obs, nm, n95, p = perm._paired_null(vectors, context["sessions"], 200,
                                                       (fold, STAGE, role, name, -1, quantity))
                    outputs["PU_PERMUTATION_CONTROLS"].append({"fold": fold, "family": name, "population": role,
                        "control": "SESSION2_SUBJECT_ID_PERMUTATION", "quantity": quantity,
                        "observed": obs, "null_mean": nm, "null_p95": n95, "one_sided_p": p, "permutations": 200})
            _, td = sem.baseline_relation(sem.centroid_grid(tx, train_pop), context["sessions"])
            obs, nm, n95, p = perm._label_null(td, ox, outer_pop, context["sessions"], 200,
                                               (fold, STAGE, name, -1))
            outputs["PU_PERMUTATION_CONTROLS"].append({"fold": fold, "family": name,
                "population": "OUTER_DEVELOPMENT", "control": "WITHIN_CELL_LABEL_PERMUTATION",
                "quantity": "cross_subject_task_relation", "observed": obs, "null_mean": nm,
                "null_p95": n95, "one_sided_p": p, "permutations": 200,
                "target_labels_used_for_fit": False})
    pca, pool_rows, bases, energy_ids, cov_ids, target_lambda = control_pool(fold, train_h, mu, qg)
    outputs["CONTROL_MATCHING_AUDIT"].extend(pool_rows)
    controls = [("PCA_ENERGY_CONTROL", -1, pca, None)]
    controls += [("RANK_RANDOM", i, dg.random_subspace(len(mu), qg.shape[1], fold, STAGE, i), None)
                 for i in range(100)]
    controls += [("ENERGY_MATCHED_RANDOM", i, bases[i], None) for i in energy_ids]
    controls += [("COVARIANCE_MATCHED_RANDOM", i, bases[i], None) for i in cov_ids]
    controls += [("COVARIANCE_SHAPED_RANDOM", -1, bases[0], synthetic_map(train_h, mu, bases[0], target_lambda))]
    for name, draw, q, transform in controls:
        views = native_views(populations, mu, q, transform)
        summary = held_summary(fold, name, views, context["sessions"])
        summary.update({"pool_index": draw, "control_type": "SYNTHETIC_ENERGY_COVARIANCE_CONFOUND_CONTROL" if transform is not None else "ORTHOGONAL_SUBSPACE"})
        outputs["PROTECTED_SPECIFICITY_CONTROLS"].append(summary)
        if name in ("PCA_ENERGY_CONTROL", "COVARIANCE_SHAPED_RANDOM"):
            details = exact_family(fold, name, views, context["sessions"])
            for key, dest in (("baseline", "PU_BASELINE_PERSISTENCE"), ("within", "PU_TASK_RELATION_WITHIN_SUBJECT"),
                              ("cross", "PU_TASK_RELATION_CROSS_SUBJECT"), ("decoder", "PU_FROZEN_DECODER_TRANSFER"),
                              ("probes", "PU_SEMANTIC_PROBES"), ("variance", "PU_VARIANCE_DECOMPOSITION")):
                outputs[dest].extend(details[key])
    outputs["PROTECTED_SPECIFICITY_CONTROLS"].append(next(r for r in summaries if r["family"] == "PROTECTED_G"))
    if dg.model_sha(model) != state:
        raise RuntimeError("frozen model/BatchNorm drift after evaluation")
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, rows in outputs.items():
        path = target / f"{name}.csv"
        rows_new(path, rows)
        hashes[path.name] = {"sha256": dg.file_sha(path), "rows": len(rows)}
    rows_new(target / "FAMILY_SUMMARY.csv", summaries)
    hashes["FAMILY_SUMMARY.csv"] = {"sha256": dg.file_sha(target / "FAMILY_SUMMARY.csv"), "rows": len(summaries)}
    write_new(target / "AUDIT.json", {"fold": fold, "files": hashes, "checkpoint_sha256": dg.file_sha(ckpt),
        "G_exact_source_geometry": True, "model_state_sha256_before_after": state,
        "fit_population": "TRAIN_GEOMETRY", "outer_fit_rows": 0, "final_heldout_eeg_reads": 0})
    print(json.dumps({"fold": fold, "status": "EVALUATION_COMPLETE", "files": len(hashes)}), flush=True)


def p_all_controls(fold: int) -> None:
    """Rank/energy/covariance controls at P_ALL's own (usually larger) rank."""
    target = RUNTIME / "p_all_controls" / f"fold{fold}"
    if target.exists():
        raise FileExistsError(target)
    if not (RUNTIME / "evaluation" / f"fold{fold}" / "AUDIT.json").is_file():
        raise RuntimeError("main evaluation must complete first")
    torch.set_num_threads(4)
    een, pw, up = dg.upstream()
    train, context = dg.load_train(fold, een, up)
    record, ckpt, model, head, device = dg.checkpoint(fold, train, context, een)
    state = dg.model_sha(model)
    stages = pw.Stages(model, head, "EEGNet", device)
    h = dg.extract(train, stages)[0][STAGE]
    spec, gate = gated_spec(fold, h, train, een)
    if dg.file_sha(ckpt) != gate["checkpoint_sha256"]:
        raise RuntimeError("checkpoint mismatch")
    qp = family_basis(spec, gate["families"]["P_ALL"])
    mu = h.mean(0, dtype=np.float64).astype(np.float32)
    from analysis_runner import cap_held, extract_stage
    populations = {"TRAIN_GEOMETRY": (train, h)}
    for role in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT"):
        pop = cap_held(dg.load_held(role, context, een, up), fold, een)
        populations[role] = (pop, extract_stage(pop, stages, STAGE))
    pca, match_rows, bases, energy_ids, cov_ids, _ = control_pool(fold, h, mu, qp)
    controls = [("P_ALL_PCA_ENERGY_CONTROL", -1, pca)]
    controls += [("P_ALL_RANK_RANDOM", i, dg.random_subspace(len(mu), qp.shape[1], fold,
                    "embedding_64d_p_all", i)) for i in range(100)]
    controls += [("P_ALL_ENERGY_MATCHED_RANDOM", i, bases[i]) for i in energy_ids]
    controls += [("P_ALL_COVARIANCE_MATCHED_RANDOM", i, bases[i]) for i in cov_ids]
    output = []
    for name, draw, q in controls:
        row = held_summary(fold, name, native_views(populations, mu, q), context["sessions"])
        row["pool_index"] = draw
        output.append(row)
    if dg.model_sha(model) != state:
        raise RuntimeError("frozen model/BatchNorm drift")
    target.mkdir(parents=True, exist_ok=False)
    rows_new(target / "P_ALL_SPECIFICITY_CONTROLS.csv", output)
    rows_new(target / "P_ALL_CONTROL_MATCHING_AUDIT.csv", match_rows)
    files = {name: {"sha256": dg.file_sha(target / name), "rows": len(rows)} for name, rows in (
        ("P_ALL_SPECIFICITY_CONTROLS.csv", output), ("P_ALL_CONTROL_MATCHING_AUDIT.csv", match_rows))}
    write_new(target / "AUDIT.json", {"fold": fold, "files": files, "fit_population": "TRAIN_GEOMETRY",
        "outer_fit_rows": 0, "checkpoint_sha256": dg.file_sha(ckpt),
        "model_state_sha256_before_after": state, "final_heldout_eeg_reads": 0})
    print(json.dumps({"fold": fold, "status": "P_ALL_CONTROLS_COMPLETE", "rank": qp.shape[1]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("reconstruct", "evaluate", "p-all-controls"))
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    if args.phase == "reconstruct":
        reconstruct(args.fold)
    elif args.phase == "evaluate":
        evaluate(args.fold)
    else:
        p_all_controls(args.fold)
