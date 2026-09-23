"""Frozen late-stage subject/session signature and descriptive drift audit.

The stage is the preregistered EEGNet late stage. Both sessions use the same
cap-8 pair rule. Outer labels define post-outcome diagnostics only; nothing is
trained, selected, or tuned on outer outcomes. No final-heldout data are read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("MECHANISM_RUNTIME", str(ROOT.parent / "pc_mechanism_closure_runtime")))
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from cell_data_v1 import materialize_stage  # noqa: E402
from sampling_core import trial_view  # noqa: E402
from run_mediation_cell_v1 import verified_context, ORIGINAL_IMPL_SHA, ANALYSIS_SHA  # noqa: E402

MODEL, TASK, FOLD = "EEGNet", "OpenBMI_MI", 0
STAGE = "depth_point_elu_pool2"
BOOTSTRAPS = 2000
SIGNATURE_COMPONENTS = (
    "P_effect_norm", "C_effect_norm", "interaction_norm",
    "P_to_P_norm", "P_to_C_norm", "C_to_P_norm", "C_to_C_norm",
    "local_P_context_sensitivity", "local_C_context_sensitivity",
    "PC_prediction_flip_disagreement_rate", "protected_top10_interaction_norm",
    "P_support_C_oppose_rate", "P_oppose_C_support_rate",
    "final_P_true_margin", "final_C_true_margin",
)


def grouped_mean(values: np.ndarray, subjects: np.ndarray,
                 sessions: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    groups = {}
    for subject in C.PW.natural(subjects):
        for session in (1, 2):
            index = np.flatnonzero((subjects == subject) & (sessions == session))
            if len(index):
                groups[(str(subject), session)] = np.asarray(values[index], np.float64).mean(0)
    return groups


def correlation(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float | None:
    a, b = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if rank:
        a, b = rankdata(a), rankdata(b)
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, tag: str) -> dict[str, object]:
    rng = np.random.default_rng(C.PW.stable_seed(
        "mechanism-subject-bootstrap", MODEL, TASK, FOLD, tag))
    draws = {"pearson": [], "spearman": []}
    for _ in range(BOOTSTRAPS):
        take = rng.integers(0, len(x), size=len(x))
        for name, rank in (("pearson", False), ("spearman", True)):
            value = correlation(x[take], y[take], rank)
            if value is not None:
                draws[name].append(value)
    return {
        "biological_subject_count": len(x), "bootstrap_resamples": BOOTSTRAPS,
        "pearson": correlation(x, y), "spearman": correlation(x, y, True),
        "pearson_valid_resamples": len(draws["pearson"]),
        "spearman_valid_resamples": len(draws["spearman"]),
        "pearson_ci95": np.percentile(draws["pearson"], [2.5, 97.5]).tolist()
                        if draws["pearson"] else None,
        "spearman_ci95": np.percentile(draws["spearman"], [2.5, 97.5]).tolist()
                         if draws["spearman"] else None,
    }


def historical_model_and_spec(source: dict[str, object], device: torch.device):
    record, stored, checkpoint, data = (source[key] for key in
                                        ("record", "stored", "checkpoint", "data"))
    net, head = C.UP.helper(MODEL).build_model({
        "Model": MODEL, "Task": TASK, "fold": FOLD, "seed": 0,
        "channels": int(record.get("channels") or 62),
        "samples": int(record.get("samples") or 1000),
        "classes": int(record["classes"]), "checkpoint_path": str(checkpoint),
        "recipe_name": record.get("recipe", {}).get("name"),
        "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
    }, device)
    capped = C.UP.capped_train(data, TASK, MODEL, FOLD)
    h, z, hz = C.UP.hook_representations(net, head, capped[0], MODEL, device)
    if np.max(np.abs(z - hz)) >= 1e-5:
        raise RuntimeError("frozen native head mismatch")
    spec = C.UP.helper(MODEL).spectrum(h, *capped[1:], TASK, MODEL, FOLD)
    if (C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"])
            != source["hashes"]["basis_sha256"] or
            int(spec["rank"]) != int(stored["rank"])):
        raise RuntimeError("frozen historical canonical basis/rank mismatch")
    return net, head, spec, np.asarray(stored["protected_blocks"], dtype=int)


def functional_signatures(runner, source, spec, dims, cell, gate_protocol):
    data = source["data"]
    centroid = C.PW.train_centroids(runner, data, MODEL, TASK, FOLD)
    canonical = C.PW.canonical(centroid["h"], spec)
    q, mu = C.raw_q(C.PW.PathFit(centroid["acts"][STAGE]), canonical[:, dims])
    cache = C.RUNTIME / "stage_cache" / MODEL / TASK / f"fold{FOLD}" / f"{STAGE}.json"
    cached = json.loads(cache.read_text(encoding="utf-8"))
    if (cached.get("mapping_sha256") != C.AR.array_digest(q, mu) or
            cached.get("protocol_sha256") != gate_protocol or
            cached.get("implementation_sha256") != ORIGINAL_IMPL_SHA):
        raise RuntimeError("frozen TRAIN late-stage mapping mismatch")
    med_manifest = json.loads((cell / "MEDIATION_V2.json").read_text(encoding="utf-8"))
    dir_manifest = json.loads((cell / "DIRECTIONAL_V4.json").read_text(encoding="utf-8"))
    mpath, dpath = (cell / "mediation_v2" / f"{STAGE}.json",
                    cell / "directional_v4" / f"{STAGE}.json")
    mediation = json.loads(mpath.read_text(encoding="utf-8"))
    directional = json.loads(dpath.read_text(encoding="utf-8"))
    if (med_manifest.get("status") != "MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES" or
            dir_manifest.get("status") != "DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES" or
            C.digest(mpath) != med_manifest["stage_sha256"][STAGE] or
            C.digest(dpath) != dir_manifest["stage_sha256"][STAGE] or
            mediation.get("status") != "STAGE_COMPLETE" or
            directional.get("status") != "STAGE_COMPLETE" or
            len(directional["random_controls"]) != 20 or
            mediation.get("final_heldout_accessed") is not False or
            directional.get("final_heldout_accessed") is not False):
        raise RuntimeError("frozen mediation/directional evidence invalid")
    med_lookup = {(row["split"], str(row["subject"]), int(row["session"])): row
                  for row in mediation["subject_session_rows"]
                  if row["transition_kind"] == "ADJACENT"}
    protected = defaultdict(list)
    for field in ("protected_train_subject_session_rows", "protected_outer_subject_session_rows"):
        for row in directional[field]:
            protected[(row["split"], str(row["subject"]), int(row["session"]))].append(
                float(row["interaction_norm"]))
    random = defaultdict(dict)
    if sorted(int(control["draw"]) for control in directional["random_controls"]) != list(range(20)):
        raise RuntimeError("frozen first-20 random draw IDs mismatch")
    for control in directional["random_controls"]:
        draw = int(control["draw"])
        by_group = defaultdict(list)
        for row in control["outer_subject_session_rows"]:
            by_group[(str(row["subject"]), int(row["session"]))].append(
                float(row["interaction_norm"]))
        for key, values in by_group.items():
            random[key][draw] = float(np.mean(values))
    signatures = {}
    pair_counts = {}
    for split in ("TRAIN", "OUTER_DEVELOPMENT"):
        view = trial_view(data, split)
        selected = materialize_stage(runner, view, model=MODEL, task=TASK,
                                     fold=FOLD, stage=STAGE)
        pairs = list(zip(selected.recipient_local.tolist(), selected.donor_local.tolist()))
        functional = C.functional_pairs(
            runner, STAGE, centroid["shapes"][STAGE], selected.activations,
            selected.logits, view.labels[selected.row_indices],
            view.subjects[selected.row_indices], pairs, q, mu, q,
            view.sessions[selected.row_indices], local=False)
        if len(functional) != len(pairs):
            raise RuntimeError(f"functional pair count mismatch: {split}")
        groups = defaultdict(list)
        for row, index in zip(functional, selected.recipient_global):
            groups[(str(view.subjects[index]), int(view.sessions[index]))].append(row)
        for (subject, session), part in groups.items():
            key = (split, subject, session)
            if key not in med_lookup or key not in protected:
                raise RuntimeError(f"missing frozen mediation/directional subject: {key}")
            means = {name: float(np.mean([float(row[name]) for row in part]))
                     for name in ("P_effect_norm", "C_effect_norm", "interaction_norm")}
            means["local_P_context_sensitivity"] = float(np.mean([
                1.0 - float(row["P_effect_context_cosine"]) for row in part]))
            means["local_C_context_sensitivity"] = float(np.mean([
                1.0 - float(row["C_effect_context_cosine"]) for row in part]))
            means["PC_prediction_flip_disagreement_rate"] = float(np.mean([
                row["P_prediction_flip"] != row["C_prediction_flip"] for row in part]))
            for metric in ("P_to_P_norm", "P_to_C_norm", "C_to_P_norm", "C_to_C_norm"):
                means[metric] = float(med_lookup[key][metric])
            means["protected_top10_interaction_norm"] = float(np.mean(protected[key]))
            if split == "OUTER_DEVELOPMENT":
                draw = random.get((subject, session), {})
                if sorted(draw) != list(range(20)):
                    raise RuntimeError(f"random subject/session coverage mismatch: {key}")
                means["rank_matched_random_interaction_norm"] = float(np.mean(list(draw.values())))
                means["interaction_random_excess"] = (means["protected_top10_interaction_norm"] -
                                                      means["rank_matched_random_interaction_norm"])
            else:
                means["rank_matched_random_interaction_norm"] = None
                means["interaction_random_excess"] = None
                means["random_excess_missing_reason"] = "V4_NO_TRAIN_SUBJECT_SESSION_RANDOM_ROWS"
            if not np.isfinite([value for value in means.values() if isinstance(value, float)]).all():
                raise RuntimeError(f"nonfinite subject signature: {key}")
            signatures[key] = {
                "split": split, "subject": subject, "session": session,
                "signature_stage": STAGE, "functional_pair_count": len(part),
                "local_sensitivity_definition": "ONE_MINUS_FINITE_SWAP_EFFECT_CONTEXT_COSINE_NOT_JACOBIAN",
                "outer_scope": "POST_OUTCOME_DIAGNOSTIC_NOT_PROSPECTIVE" if split == "OUTER_DEVELOPMENT" else "TRAIN_ONLY",
                **means,
            }
        pair_counts[split] = len(pairs)
    hashes = {"phase1_stage_cache": C.digest(cache),
              "mediation_manifest": C.digest(cell / "MEDIATION_V2.json"),
              "mediation_stage": C.digest(mpath),
              "directional_manifest": C.digest(cell / "DIRECTIONAL_V4.json"),
              "directional_stage": C.digest(dpath)}
    return signatures, pair_counts, hashes


def final_rows(net, head, spec, dims, source):
    data = source["data"]
    tx, ty, ts, tse = C.BR.trial_train(data, MODEL, TASK, FOLD)
    sx, sy, ss = C.PW.capped_outer(data, MODEL, TASK, FOLD, "source")
    fx, fy, fs = C.PW.capped_outer(data, MODEL, TASK, FOLD, "future")
    views = {
        "TRAIN": (tx, ty, ts.astype(str), tse),
        "OUTER_DEVELOPMENT": (
            np.concatenate((sx, fx)), np.concatenate((sy, fy)),
            np.concatenate((ss, fs)).astype(str),
            np.concatenate((np.full(len(sx), int(data["source_session"])),
                            np.full(len(fx), int(data["future_session"])))),
        ),
    }
    result, reps, future = {}, {}, {}
    raw_base = C.UP.raw_base(spec)[dims]
    for split, (x, labels, subjects, sessions) in views.items():
        h, z, hz = C.UP.hook_representations(net, head, x, MODEL, net.device if hasattr(net, "device") else next(net.parameters()).device)
        if np.max(np.abs(z - hz)) >= 1e-5:
            raise RuntimeError(f"native head mismatch: {split}")
        q, z0, zp, zc = C.BR.decompose(h, z, spec, dims, head)
        if np.max(np.abs(z - (z0 + zp + zc))) >= 1e-5:
            raise RuntimeError(f"frozen final P+C reconstruction mismatch: {split}")
        p = q[:, dims] @ raw_base
        c = h - spec["mean"] - p
        reps[split] = {name: grouped_mean(value, subjects, sessions)
                       for name, value in (("P", p), ("C", c), ("H", h))}
        rival = z.copy()
        rival[np.arange(len(z)), labels] = -np.inf
        other = rival.argmax(1)
        p_margin = zp[np.arange(len(z)), labels] - zp[np.arange(len(z)), other]
        c_margin = zc[np.arange(len(z)), labels] - zc[np.arange(len(z)), other]
        values = np.column_stack((p_margin, c_margin,
                                  (p_margin > 0) & (c_margin < 0),
                                  (p_margin < 0) & (c_margin > 0)))
        for (subject, session), mean in grouped_mean(values, subjects, sessions).items():
            result[(split, subject, session)] = {
                "final_P_true_margin": float(mean[0]),
                "final_C_true_margin": float(mean[1]),
                "P_support_C_oppose_rate": float(mean[2]),
                "P_oppose_C_support_rate": float(mean[3]),
            }
        if split == "OUTER_DEVELOPMENT":
            prob = C.AR.sm(z)
            ce = -np.log(np.maximum(prob[np.arange(len(z)), labels], 1e-12))
            pred = z.argmax(1)
            for subject in C.PW.natural(subjects):
                mask = (subjects == subject) & (sessions == 2)
                if not mask.any():
                    continue
                recalls = [float(np.mean(pred[mask & (labels == label)] == label))
                           for label in np.unique(labels[mask])]
                future[str(subject)] = {
                    "subject": str(subject), "future_trials": int(mask.sum()),
                    "future_BA": float(np.mean(recalls)),
                    "future_CE": float(np.mean(ce[mask])),
                    "future_error_rate": float(np.mean(pred[mask] != labels[mask])),
                }
    return result, reps, future


def calculate_drift(signatures, representations):
    vectors = {}
    for key, row in signatures.items():
        vector = np.asarray([row[name] for name in SIGNATURE_COMPONENTS], np.float64)
        if not np.isfinite(vector).all():
            raise RuntimeError(f"nonfinite signature vector: {key}")
        vectors[key] = vector
    train = np.stack([value for key, value in vectors.items() if key[0] == "TRAIN"])
    mu, sd = train.mean(0), np.maximum(train.std(0), 1e-6)
    rep_sd = {name: np.maximum(np.stack(list(representations["TRAIN"][name].values())).std(0), 1e-6)
              for name in ("P", "C", "H")}
    drift = []
    for split in ("TRAIN", "OUTER_DEVELOPMENT"):
        subjects = sorted({subject for sp, subject, _ in vectors if sp == split},
                          key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        for subject in subjects:
            first, second = (split, subject, 1), (split, subject, 2)
            if first not in vectors or second not in vectors:
                drift.append({"split": split, "subject": subject,
                              "status": "MISSING_PAIRED_SESSION"})
                continue
            a, b = ((vectors[key] - mu) / sd for key in (first, second))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            row = {"split": split, "subject": subject, "status": "PAIRED",
                   "mechanism_cosine": float(np.dot(a, b) / denominator) if denominator > 1e-12 else None,
                   "mechanism_standardized_euclidean": float(np.sqrt(np.mean((b - a) ** 2))),
                   "mechanism_component_drift": {
                       name: float(value) for name, value in zip(SIGNATURE_COMPONENTS, b - a)},
                   "random_excess_excluded_from_drift": True}
            for name in ("P", "C", "H"):
                x, y = (representations[split][name][(subject, session)] for session in (1, 2))
                row[f"{name}_standardized_euclidean"] = float(np.sqrt(np.mean(((y - x) / rep_sd[name]) ** 2)))
                row[f"{name}_cosine"] = float(np.dot(x, y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
            drift.append(row)
    return drift


def main() -> None:
    global TASK, FOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    TASK, FOLD = args.task, args.fold
    cell = RUNTIME / "cells" / MODEL.lower() / TASK.lower() / f"fold{FOLD}_seed0"
    result = cell / "SUBJECT_SESSION_AUDIT_V1.json"
    failure = cell / "SUBJECT_SESSION_AUDIT_V1.FAIL_CLOSED.json"
    if result.exists() or failure.exists():
        raise RuntimeError("subject/session audit has terminal evidence")
    try:
        gate_path, _, source, protocol_sha = verified_context(MODEL, TASK, FOLD)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head, spec, dims = historical_model_and_spec(source, device)
        runner = C.PW.Stages(net, head, MODEL, device)
        signatures, pair_counts, hashes = functional_signatures(
            runner, source, spec, dims, cell, protocol_sha)
        margins, representations, future = final_rows(net, head, spec, dims, source)
        for key, row in signatures.items():
            if key not in margins:
                raise RuntimeError(f"final margin subject/session absent: {key}")
            row.update(margins[key])
        drift = calculate_drift(signatures, representations)
        outer = {row["subject"]: row for row in drift
                 if row["split"] == "OUTER_DEVELOPMENT" and row["status"] == "PAIRED"}
        if set(outer) != set(future):
            raise RuntimeError("outer drift/future performance subjects mismatch")
        subjects = sorted(outer, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        relation = {}
        for measure in ("mechanism_standardized_euclidean", "P_standardized_euclidean",
                        "C_standardized_euclidean", "H_standardized_euclidean"):
            x = np.asarray([outer[s][measure] for s in subjects], np.float64)
            for target in ("future_BA", "future_CE", "future_error_rate"):
                y = np.asarray([future[s][target] for s in subjects], np.float64)
                relation[f"{measure}__{target}"] = bootstrap_correlation(x, y, f"{measure}/{target}")
        if (len([key for key in signatures if key[0] == "TRAIN"]) != 52 or
                len([key for key in signatures if key[0] == "OUTER_DEVELOPMENT"]) != 16 or
                len(drift) != 34 or len(future) != 8):
            raise RuntimeError("biological subject/session coverage incomplete")
        C.write_json(result, {
            "status": "SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE",
            "model": MODEL, "task": TASK, "fold": FOLD, "seed": 0,
            "signature_stage": STAGE,
            "outer_scope": "EVALUATION_ONLY_POST_OUTCOME_DIAGNOSTIC_NOT_PROSPECTIVE",
            "cap8_pair_counts": pair_counts,
            "subject_session_signature_rows": list(signatures.values()),
            "subject_session_drift_rows": drift,
            "future_subject_performance": list(future.values()),
            "drift_performance_descriptive_correlations": relation,
            "drift_components": SIGNATURE_COMPONENTS,
            "standardization_unit": "TRAIN_BIOLOGICAL_SUBJECT_SESSION",
            "bootstrap_unit": "BIOLOGICAL_SUBJECT",
            "bootstrap_resamples": BOOTSTRAPS,
            "missing_train_random_excess_reason": "V4_NO_TRAIN_SUBJECT_SESSION_RANDOM_ROWS",
            "frozen_evidence_sha256": hashes,
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_accessed": True,
            "final_heldout_accessed": False,
        })
        print("SUBJECT_SESSION_AUDIT_V1_COMPLETE", len(signatures), len(drift),
              len(future), flush=True)
    except Exception as exc:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
                               "implementation_sha256": C.digest(Path(__file__)),
                               "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
