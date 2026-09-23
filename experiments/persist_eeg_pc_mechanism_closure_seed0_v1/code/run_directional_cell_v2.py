"""TRAIN-ranked P×C directional phase v4 with resource-safe process recycling.

This is a required component of a mechanism cell, not a complete cell by
itself. All axes, PCA fits, SDs and top-10 ranks come only from TRAIN.
Outer development supplies evaluation rows after each direction is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("MECHANISM_RUNTIME", str(ROOT.parent / "pc_mechanism_closure_runtime")))
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from cell_data_v1 import materialize_stage, stage_norms  # noqa: E402
from directional_core_v3 import fit_directions, frozen_top_pairs, mixed_difference  # noqa: E402
from directional_factorized_eegnet_v4 import factorized_mixed_difference, precompute_spatial_base  # noqa: E402
from directional_linear_tail import eligible as affine_tail_eligible, affine_tail_mixed_difference  # noqa: E402
from sampling_core import trial_view  # noqa: E402
from run_mediation_cell_v1 import verified_context, AMENDMENT_SHA, ANALYSIS_SHA, ORIGINAL_IMPL_SHA  # noqa: E402


def unique_recipient_rows(selection) -> np.ndarray:
    return np.unique(selection.recipient_local)


def selected_identity(selection, view, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original = selection.row_indices[rows]
    return (view.labels[original], view.subjects[original].astype(str),
            view.sessions[original], selection.logits[rows])


def subject_equal_mean(values: np.ndarray, subjects: np.ndarray) -> float:
    return float(np.mean([np.mean(values[subjects == subject]) for subject in sorted(set(subjects))]))


def group_rows(effect: dict[str, np.ndarray], labels: np.ndarray, subjects: np.ndarray,
               sessions: np.ndarray, native: np.ndarray) -> list[dict[str, object]]:
    correct = native.argmax(1) == labels
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (subject, session) in enumerate(zip(subjects, sessions)):
        groups[(str(subject), int(session))].append(i)
    rows: list[dict[str, object]] = []
    for (subject, session), positions in sorted(groups.items()):
        idx = np.asarray(positions, np.int64)
        row = {"subject": subject, "session": session, "trial_count": int(len(idx)),
               "native_error_rate": float(np.mean(~correct[idx]))}
        for key in ("norm", "true_margin", "top1_margin"):
            row[f"interaction_{key}"] = float(np.mean(effect[key][idx]))
        row["positive_true_margin_rate"] = float(np.mean(effect["true_margin"][idx] > 0))
        for label, mask in (("native_correct", correct[idx]), ("native_error", ~correct[idx])):
            row[f"{label}_n"] = int(mask.sum())
            row[f"{label}_interaction_norm"] = float(np.mean(effect["norm"][idx][mask])) if mask.any() else None
            row[f"{label}_true_margin"] = float(np.mean(effect["true_margin"][idx][mask])) if mask.any() else None
        rows.append(row)
    return rows


def matrix_and_pairs(runner, stage: str, shape: tuple[int, ...], selection, view,
                     axes, *, pair_subset: tuple[tuple[int, int], ...] | None = None,
                     epsilon: float = 0.5, spatial_base=None):
    rows = unique_recipient_rows(selection)
    a = selection.activations[rows]
    labels, subjects, sessions, native = selected_identity(selection, view, rows)
    pairs = tuple(np.ndindex((axes.p_rank, axes.c_rank))) if pair_subset is None else pair_subset
    means = np.full((axes.p_rank, axes.c_rank), np.nan, np.float64)
    effects = {}
    for p, c in pairs:
        if affine_tail_eligible(runner, stage):
            effect = affine_tail_mixed_difference(runner, stage, shape, a, labels,
                                                  native, axes, (p, c), epsilon=epsilon)
        elif spatial_base is None:
            effect = mixed_difference(runner, stage, shape, a, labels, native, axes, (p, c), epsilon=epsilon)
        else:
            effect = factorized_mixed_difference(runner, stage, shape, a, labels, native,
                                                 axes, (p, c), epsilon=epsilon,
                                                 spatial_base=spatial_base)
        means[p, c] = subject_equal_mean(effect["norm"], subjects)
        effects[(p, c)] = effect
    return means, effects, (labels, subjects, sessions, native)


def evaluated_rows(effects, identity, pairs, *, split: str, train_means: np.ndarray | None = None):
    labels, subjects, sessions, native = identity
    rows = []
    for rank, (p, c) in enumerate(pairs, 1):
        effect = effects[(p, c)]
        for row in group_rows(effect, labels, subjects, sessions, native):
            row.update({"split": split, "p_index": p, "c_index": c, "train_rank": rank})
            if train_means is not None:
                row["train_subject_equal_interaction_norm"] = float(train_means[p, c])
            rows.append(row)
    return rows


def summarize_control(train_matrix: np.ndarray, outer_rows: list[dict[str, object]], pairs):
    flat = train_matrix.ravel()
    mass = float(sum(train_matrix[p, c] for p, c in pairs))
    total = float(np.sum(flat))
    per_subject: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in outer_rows:
        if row["native_error_interaction_norm"] is not None and row["native_correct_interaction_norm"] is not None:
            per_subject[str(row["subject"])]["error_minus_correct"].append(
                float(row["native_error_interaction_norm"] - row["native_correct_interaction_norm"]))
    association = [float(np.mean(values["error_minus_correct"])) for values in per_subject.values() if values["error_minus_correct"]]
    return {"train_max_interaction": float(np.max(flat)), "train_top10_mass": mass,
            "train_concentration_ratio": mass / total if total > 0 else None,
            "outer_subject_error_association": float(np.mean(association)) if association else None,
            "outer_error_association_subjects": len(association)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("EEGNet", "EEGConformer"), required=True)
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    model, task, fold = args.model, args.task, args.fold
    cell_dir = RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0"
    manifest = cell_dir / "DIRECTIONAL_V4.json"
    failure = cell_dir / "DIRECTIONAL_V4_FAIL_CLOSED.json"
    if manifest.exists() or failure.exists():
        raise RuntimeError("directional phase already has terminal evidence; no overwrite")
    try:
        gate_path, _, source, protocol_sha = verified_context(model, task, fold)
        record, stored, checkpoint, data = source["record"], source["stored"], source["checkpoint"], source["data"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = C.UP.helper(model).build_model({
            "Model": model, "Task": task, "fold": fold, "seed": 0,
            "channels": int(record.get("channels") or 62), "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]), "checkpoint_path": str(checkpoint),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        runner = C.PW.Stages(net, head, model, device)
        centroid = C.PW.train_centroids(runner, data, model, task, fold)
        capped = C.UP.capped_train(data, task, model, fold)
        bh, _, _ = C.UP.hook_representations(net, head, capped[0], model, device)
        spec = C.UP.helper(model).spectrum(bh, *capped[1:], task, model, fold)
        dims = np.asarray(stored["protected_blocks"], dtype=int)
        if C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"]) != source["hashes"]["basis_sha256"]:
            raise RuntimeError("frozen canonical basis SHA mismatch")
        canonical = C.PW.canonical(centroid["h"], spec)
        views = {split: trial_view(data, split) for split in ("TRAIN", "OUTER_DEVELOPMENT")}
        norms = {split: stage_norms(runner, view, tuple(runner.names)) for split, view in views.items()}
        print("DIRECTIONAL_NORMS_COMPLETE", model, task, fold, flush=True)
        randoms = source["randoms"][:20]
        if len(randoms) != 20:
            raise RuntimeError("fewer than 20 locked equal-rank random draws")
        stage_files = []
        for stage in runner.names:
            destination = cell_dir / "directional_v4" / f"{stage}.json"
            partial_path = cell_dir / "directional_v4" / f"{stage}.partial.json"
            if destination.exists():
                cached = json.loads(destination.read_text(encoding="utf-8"))
                if cached.get("status") != "STAGE_COMPLETE" or cached.get("implementation_sha256") != C.digest(Path(__file__)):
                    raise RuntimeError(f"stale phase-2 directional stage evidence: {stage}")
                stage_files.append(destination)
                print("DIRECTIONAL_STAGE_CACHED", stage, flush=True)
                continue
            shape = tuple(centroid["shapes"][stage])
            fit = C.PW.PathFit(centroid["acts"][stage]) if stage != "classifier_input" else None
            if fit is None:
                q, mu = C.final_q(spec, dims)
                left = np.ascontiguousarray(((spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]).astype(np.float32))
                right = np.ascontiguousarray(C.UP.raw_base(spec)[dims].astype(np.float32))
                projector = (left, right)
            else:
                q, mu = C.raw_q(fit, canonical[:, dims])
                projector = None
            stage_cache = C.RUNTIME / "stage_cache" / model / task / f"fold{fold}" / f"{stage}.json"
            frozen = json.loads(stage_cache.read_text(encoding="utf-8"))
            if (frozen.get("mapping_sha256") != C.AR.array_digest(q, mu)
                    or frozen.get("implementation_sha256") != ORIGINAL_IMPL_SHA):
                raise RuntimeError(f"frozen TRAIN mapping mismatch: {stage}")
            selections = {split: materialize_stage(runner, view, model=model, task=task, fold=fold,
                                                    stage=stage, precomputed_norms=norms[split][stage])
                          for split, view in views.items()}
            spatial_bases = {}
            if model == "EEGNet" and stage == "temporal_bn":
                for split, selection in selections.items():
                    recipients = unique_recipient_rows(selection)
                    spatial_bases[split] = precompute_spatial_base(
                        runner, shape, selection.activations[recipients], batch_size=64)
                print("DIRECTIONAL_V4_SPATIAL_BASE_CACHED", model, task, fold, stage,
                      "train_rows", len(spatial_bases["TRAIN"]),
                      "outer_rows", len(spatial_bases["OUTER_DEVELOPMENT"]), flush=True)
            train_a = selections["TRAIN"].activations[unique_recipient_rows(selections["TRAIN"])]
            stage_started = time.perf_counter()
            axes = fit_directions(train_a, q, mu, c_limit=16, final_projector=projector, device=device)
            print("DIRECTIONAL_V4_AXES_FIT", model, task, fold, stage,
                  "seconds", round(time.perf_counter() - stage_started, 2), flush=True)
            equivalence_max = None
            if (spatial_bases or affine_tail_eligible(runner, stage)) and not partial_path.exists():
                train_selection = selections["TRAIN"]
                verify_rows = unique_recipient_rows(train_selection)
                verify_a = train_selection.activations[verify_rows]
                verify_labels, _, _, verify_native = selected_identity(
                    train_selection, views["TRAIN"], verify_rows)
                verification_pairs = ((0, 0), (axes.p_rank - 1, axes.c_rank - 1))
                differences = []
                for pair in verification_pairs:
                    reference = mixed_difference(runner, stage, shape, verify_a, verify_labels,
                                                 verify_native, axes, pair, epsilon=0.5)
                    if affine_tail_eligible(runner, stage):
                        accelerated = affine_tail_mixed_difference(
                            runner, stage, shape, verify_a, verify_labels, verify_native,
                            axes, pair, epsilon=0.5)
                    else:
                        accelerated = factorized_mixed_difference(
                            runner, stage, shape, verify_a, verify_labels, verify_native,
                            axes, pair, epsilon=0.5, spatial_base=spatial_bases["TRAIN"])
                    differences.extend(float(np.max(np.abs(reference[key] - accelerated[key])))
                                       for key in reference)
                equivalence_max = max(differences)
                if not np.isfinite(equivalence_max) or equivalence_max >= 1e-5:
                    raise RuntimeError(f"full TRAIN affine-prefix equivalence failed: {equivalence_max}")
                print("DIRECTIONAL_V4_FULL_TRAIN_EQUIVALENCE", model, task, fold, stage,
                      "trials", len(verify_rows), "max_abs", equivalence_max, flush=True)
            if partial_path.exists():
                partial = json.loads(partial_path.read_text(encoding="utf-8"))
                if (partial.get("status") != "PARTIAL_RESOURCE_SAFE"
                        or partial.get("implementation_sha256") != C.digest(Path(__file__))
                        or partial.get("analysis_lock_sha256") != ANALYSIS_SHA
                        or partial.get("upstream_gate_sha256") != C.digest(gate_path)
                        or partial.get("train_mapping_sha256") != C.AR.array_digest(q, mu)
                        or partial.get("p_rank") != axes.p_rank
                        or partial.get("c_rank") != axes.c_rank
                        or partial.get("projector_mode") != axes.projector_mode
                        or partial.get("final_heldout_accessed") is not False
                        or partial.get("stage") != stage):
                    raise RuntimeError(f"directional partial evidence mismatch: {stage}")
                primary = np.asarray(partial["protected_train_subject_equal_matrix"], np.float64)
                top = tuple(tuple(pair) for pair in partial["protected_top10"])
                protected_train_rows = partial["protected_train_subject_session_rows"]
                protected_outer_rows = partial["protected_outer_subject_session_rows"]
                protected_summary = partial["protected_summary"]
                controls = list(partial["random_controls"])
                equivalence_max = partial["full_train_equivalence_max_abs"]
                if primary.shape != (axes.p_rank, axes.c_rank) or len(top) != 10 or len(controls) > 20:
                    raise RuntimeError("directional partial shape/count invalid")
                print("DIRECTIONAL_V4_PARTIAL_REUSED", model, task, fold, stage,
                      "controls", len(controls), flush=True)
            else:
                primary, effects, train_id = matrix_and_pairs(runner, stage, shape,
                                                               selections["TRAIN"], views["TRAIN"], axes,
                                                               spatial_base=spatial_bases.get("TRAIN"))
                print("DIRECTIONAL_V4_PROTECTED_MATRIX", model, task, fold, stage,
                      "seconds", round(time.perf_counter() - stage_started, 2), flush=True)
                top = frozen_top_pairs(primary, 10)
                _, outer_effects, outer_id = matrix_and_pairs(runner, stage, shape,
                                                               selections["OUTER_DEVELOPMENT"],
                                                               views["OUTER_DEVELOPMENT"], axes,
                                                               pair_subset=top,
                                                               spatial_base=spatial_bases.get("OUTER_DEVELOPMENT"))
                protected_train_rows = evaluated_rows(effects, train_id, top, split="TRAIN", train_means=primary)
                protected_outer_rows = evaluated_rows(outer_effects, outer_id, top,
                                                       split="OUTER_DEVELOPMENT", train_means=primary)
                protected_summary = summarize_control(primary, protected_outer_rows, top)
                controls = []
            new_draws = 0
            for draw, subset in enumerate(randoms):
                subset_sha = hashlib.sha256(np.ascontiguousarray(subset).tobytes()).hexdigest()
                if draw < len(controls):
                    if controls[draw].get("draw") != draw or controls[draw].get("final_subset_sha256") != subset_sha:
                        raise RuntimeError(f"cached random control identity mismatch: {stage}/{draw}")
                    continue
                draw_started = time.perf_counter()
                if fit is None:
                    qr, mur = C.final_q(spec, subset)
                    left_r = np.ascontiguousarray(((spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, subset]).astype(np.float32))
                    right_r = np.ascontiguousarray(C.UP.raw_base(spec)[subset].astype(np.float32))
                    projector_r = (left_r, right_r)
                else:
                    qr, mur = C.raw_q(fit, canonical[:, subset])
                    projector_r = None
                random_axes = fit_directions(train_a, qr, mur, c_limit=16,
                                             final_projector=projector_r, device=device)
                random_matrix, random_effects, random_id = matrix_and_pairs(
                    runner, stage, shape, selections["TRAIN"], views["TRAIN"], random_axes,
                    spatial_base=spatial_bases.get("TRAIN"))
                random_top = frozen_top_pairs(random_matrix, 10)
                _, random_outer_effects, random_outer_id = matrix_and_pairs(
                    runner, stage, shape, selections["OUTER_DEVELOPMENT"],
                    views["OUTER_DEVELOPMENT"], random_axes, pair_subset=random_top,
                    spatial_base=spatial_bases.get("OUTER_DEVELOPMENT"))
                random_outer_rows = evaluated_rows(random_outer_effects, random_outer_id,
                                                   random_top, split="OUTER_DEVELOPMENT",
                                                   train_means=random_matrix)
                controls.append({
                    "draw": draw, "final_subset": subset.tolist(),
                    "final_subset_sha256": subset_sha,
                    "mapping_sha256": C.AR.array_digest(qr, mur),
                    "p_rank": random_axes.p_rank, "c_rank": random_axes.c_rank,
                    "train_top10": [list(pair) for pair in random_top],
                    **summarize_control(random_matrix, random_outer_rows, random_top),
                    "outer_subject_session_rows": random_outer_rows,
                })
                C.write_json(partial_path, {
                    "status": "PARTIAL_RESOURCE_SAFE", "schema": "PC_MECHANISM_DIRECTIONAL_PARTIAL_V4",
                    "model": model, "task": task, "fold": fold, "stage": stage,
                    "projector_mode": axes.projector_mode,
                    "p_rank": axes.p_rank, "c_rank": axes.c_rank,
                    "train_mapping_sha256": C.AR.array_digest(q, mu),
                    "protected_train_subject_equal_matrix": primary,
                    "protected_top10": [list(pair) for pair in top],
                    "protected_summary": protected_summary,
                    "protected_train_subject_session_rows": protected_train_rows,
                    "protected_outer_subject_session_rows": protected_outer_rows,
                    "random_controls": controls,
                    "full_train_equivalence_max_abs": equivalence_max,
                    "upstream_gate_sha256": C.digest(gate_path),
                    "analysis_lock_sha256": ANALYSIS_SHA,
                    "implementation_sha256": C.digest(Path(__file__)),
                    "final_heldout_accessed": False,
                })
                print("DIRECTIONAL_RANDOM_DONE", model, task, fold, stage, draw + 1,
                      "seconds", round(time.perf_counter() - draw_started, 2), flush=True)
                new_draws += 1
                free_gb = psutil.virtual_memory().available / (1024 ** 3)
                if draw < 19 and (free_gb < 24 or (stage == "temporal_bn" and new_draws >= 2)):
                    print("DIRECTIONAL_V4_PARTIAL_RESOURCE_SAFE", model, task, fold, stage,
                          "controls", len(controls), "free_gb", round(free_gb, 2), flush=True)
                    return
            C.write_json(destination, {
                "status": "STAGE_COMPLETE", "schema": "PC_MECHANISM_DIRECTIONAL_STAGE_V4",
                "model": model, "task": task, "fold": fold, "stage": stage,
                "projector_mode": axes.projector_mode,
                "p_rank": axes.p_rank, "c_rank": axes.c_rank,
                "train_mapping_sha256": C.AR.array_digest(q, mu),
                "protected_train_subject_equal_matrix": primary,
                "protected_top10": [list(pair) for pair in top],
                "protected_summary": protected_summary,
                "protected_train_subject_session_rows": protected_train_rows,
                "protected_outer_subject_session_rows": protected_outer_rows,
                "random_controls": controls,
                "upstream_gate_sha256": C.digest(gate_path),
                "upstream_protocol_sha256": protocol_sha,
                "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
                "analysis_lock_sha256": ANALYSIS_SHA, "projector_amendment_sha256": AMENDMENT_SHA,
                "numeric_batch_size": 64,
                "full_train_equivalence_max_abs": equivalence_max,
                "forward_schedule": ("AFFINE_CLASSIFIER_TAIL_EXACT"
                                     if affine_tail_eligible(runner, stage) else
                                     "EEGNET_SPATIAL_AFFINE_PREFIX_CACHED"
                                     if spatial_bases else "NATIVE_STAGE_FORWARD"),
                "directional_core_sha256": C.digest(EXP / "code" / "directional_core_v3.py"),
                "factorized_core_sha256": C.digest(EXP / "code" / "directional_factorized_eegnet_v4.py"),
                "affine_tail_core_sha256": C.digest(EXP / "code" / "directional_linear_tail.py"),
                "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
            })
            stage_files.append(destination)
            print("DIRECTIONAL_STAGE_COMPLETE", model, task, fold, stage, flush=True)
        C.write_json(manifest, {
            "status": "DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES",
            "model": model, "task": task, "fold": fold, "seed": 0,
            "stage_count": len(stage_files), "stage_sha256": {p.stem: C.digest(p) for p in stage_files},
            "random_draws_per_stage": 20,
            "numeric_batch_size": 64,
            "directional_core_sha256": C.digest(EXP / "code" / "directional_core_v3.py"),
            "factorized_core_sha256": C.digest(EXP / "code" / "directional_factorized_eegnet_v4.py"),
            "affine_tail_core_sha256": C.digest(EXP / "code" / "directional_linear_tail.py"),
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False,
        })
        print("DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES", model, task, fold, flush=True)
    except Exception as error:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(error).__name__}: {error}",
                               "implementation_sha256": C.digest(Path(__file__)), "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
