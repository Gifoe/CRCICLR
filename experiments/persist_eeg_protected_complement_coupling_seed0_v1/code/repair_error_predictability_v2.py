"""Separately provenanced, label-free repair of coupling error-prediction features.

The frozen original cells, their stage caches, and the 2x2 mechanism analysis
are never modified. Only the retrospective-to-prospective feature leak in the
error-prediction analysis is repaired. This is a post-outcome engineering
correction, so corrected prediction evidence remains explicitly exploratory.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_coupling as C


ORIGINAL_IMPL_SHA = "4aab984969f4ba43128ed738a38ac138d94ee1c9b1252662af91b0c44e3f8eee"
PROTOCOL_SHA = "3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c"
RULE = "same subject/session; frozen native predicted class differs; nearest activation norm; all recipients; no true label in feature construction"
REPAIR_ROOT = C.RUNTIME / "prediction_repair_v2"
REPAIR_OUT = C.EXP / "outputs_protocol_repair_v2"


def repaired_cell_path(model: str, task: str, fold: int) -> Path:
    return REPAIR_ROOT / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def feature_stage_path(model: str, task: str, fold: int, stage: str) -> Path:
    return REPAIR_ROOT / "stage_features" / model / task / f"fold{fold}" / f"{stage}.json"


def label_free_pairs(activations: np.ndarray, native_predictions: np.ndarray,
                     subjects: np.ndarray, sessions: np.ndarray) -> list[tuple[int, int]]:
    """No task labels, error indicators, or outer outcomes enter this rule."""
    if not (len(activations) == len(native_predictions) == len(subjects) == len(sessions)):
        raise ValueError("pair arrays have unequal length")
    magnitude = np.linalg.norm(activations.reshape(len(activations), -1), axis=1)
    result: list[tuple[int, int]] = []
    for recipient in range(len(activations)):
        eligible = np.flatnonzero((subjects == subjects[recipient]) &
                                  (sessions == sessions[recipient]) &
                                  (native_predictions != native_predictions[recipient]))
        if len(eligible) == 0:
            continue
        # np.flatnonzero is index sorted; argmin therefore fixes tie breaking.
        donor = int(eligible[np.argmin(np.abs(magnitude[eligible] - magnitude[recipient]))])
        result.append((recipient, donor))
    return result


def original_cell(model: str, task: str, fold: int) -> dict[str, Any]:
    path = C.cell_path(model, task, fold)
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("status") != "COMPLETE" or row.get("protocol_sha256") != PROTOCOL_SHA or
            row.get("implementation_sha256") != ORIGINAL_IMPL_SHA or row.get("final_heldout_accessed") or
            row.get("backbone_training") or row.get("native_head_refit") or row.get("protected_reselection")):
        raise RuntimeError(f"original cell provenance/status mismatch: {model}/{task}/{fold}")
    if len(row.get("stages", [])) != (5 if model == "EEGNet" else 7):
        raise RuntimeError("original stage count mismatch")
    return row


def stage_features(runner: Any, model: str, task: str, fold: int, stage: str,
                   original: dict[str, Any], train_centroids: dict[str, Any],
                   canonical_centroids: np.ndarray, spec: dict[str, Any], dims: np.ndarray,
                   tx: np.ndarray, ts: np.ndarray, tse: np.ndarray,
                   ox: np.ndarray, osub: np.ndarray, ose: np.ndarray,
                   source_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    original_stage = next(s for s in original["stages"] if s["stage"] == stage)
    train = C.PW.outer_stage(runner, tx, stage)
    outer = C.PW.outer_stage(runner, ox, stage)
    a_train, a_outer = train["a"], outer["a"]
    centroid_a = train_centroids["acts"][stage]
    fit = C.PW.PathFit(centroid_a) if stage != "classifier_input" else None
    if fit is None:
        q, mu = C.final_q(spec, dims)
        transform = (spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]
        projector = (np.ascontiguousarray(transform.astype(np.float32)),
                     np.ascontiguousarray(C.UP.raw_base(spec)[dims].astype(np.float32)))
    else:
        q, mu = C.raw_q(fit, canonical_centroids[:, dims])
        projector = None
    if C.AR.array_digest(q, mu) != original_stage["mapping_sha256"]:
        raise RuntimeError(f"frozen stage mapping mismatch: {stage}")
    if C.check_partition(a_train[:min(16, len(a_train))], q, mu) >= 1e-5:
        raise RuntimeError("TRAIN P+C reconstruction failure")
    if C.check_partition(a_outer[:min(16, len(a_outer))], q, mu) >= 1e-5:
        raise RuntimeError("outer P+C reconstruction failure")
    centroid_pca = C.GramPCA(centroid_a, centroid_a[:1], mu, runner.device)
    cdir = C.complement_directions(centroid_a, q, mu, projector=projector, context=centroid_pca)
    train_native_pred = np.asarray(train["z"]).argmax(1)
    outer_native_pred = np.asarray(outer["z"]).argmax(1)
    train_pairs = label_free_pairs(a_train, train_native_pred, ts, tse)
    outer_pairs = label_free_pairs(a_outer, outer_native_pred, osub, ose)
    if any(ose[i] != ose[j] or osub[i] != osub[j] for i, j in outer_pairs):
        raise RuntimeError("label-free outer pair crossed subject/session")
    train_rows = C.functional_pairs(runner, stage, train_centroids["shapes"][stage],
                                    a_train, train["z"], train_native_pred, ts,
                                    train_pairs, q, mu, cdir, tse, projector=projector)
    future_pairs = [(i, j) for i, j in outer_pairs if i >= source_count]
    if any(j < source_count for _, j in future_pairs):
        raise RuntimeError("outer source trial became future donor")
    future_rows = C.functional_pairs(runner, stage, train_centroids["shapes"][stage],
                                     a_outer, outer["z"], outer_native_pred, osub,
                                     future_pairs, q, mu, cdir, ose, projector=projector)
    future_rows = [{**row, "recipient": row["recipient"] - source_count,
                    "donor": row["donor"] - source_count} for row in future_rows]
    train_features = C.trial_coupling_features(train_rows, len(tx))
    future_features = C.trial_coupling_features(future_rows, len(ox) - source_count)
    counts = {"train_pairs": len(train_pairs), "future_pairs": len(future_pairs),
              "train_recipients_with_pair": len({i for i, _ in train_pairs}),
              "future_recipients_with_pair": len({i for i, _ in future_pairs})}
    return train_features, future_features, counts


def repair_cell(model: str, task: str, fold: int) -> None:
    destination = repaired_cell_path(model, task, fold)
    if destination.exists():
        prior = json.loads(destination.read_text(encoding="utf-8"))
        if prior.get("status") != "COMPLETE" or prior.get("repair_implementation_sha256") != C.digest(Path(__file__)):
            raise RuntimeError("existing repair cell is not eligible for reuse")
        print("REPAIR_CELL_CACHED", model, task, fold, flush=True)
        return
    source_cell = original_cell(model, task, fold)
    source_hash = C.digest(C.cell_path(model, task, fold))
    source = C.prior_record(model, task, fold)
    if C.assert_lock(model, task, fold, source) != PROTOCOL_SHA:
        raise RuntimeError("frozen protocol mismatch")
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
    tx, ty, ts, tse = C.BR.trial_train(data, model, task, fold)
    C.PW.stage_check(runner, tx)
    train_centroids = C.PW.train_centroids(runner, data, model, task, fold)
    bh, _, _ = C.UP.hook_representations(net, head, C.UP.capped_train(data, task, model, fold)[0], model, device)
    spec = C.UP.helper(model).spectrum(bh, *C.UP.capped_train(data, task, model, fold)[1:], task, model, fold)
    dims = np.asarray(stored["protected_blocks"], int)
    basis_sha = C.UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"])
    if basis_sha != source["hashes"]["basis_sha256"] or basis_sha != source_cell["basis_sha256"]:
        raise RuntimeError("frozen basis mismatch")
    canonical_centroids = C.PW.canonical(train_centroids["h"], spec)
    sx, sy, ss = C.PW.capped_outer(data, model, task, fold, "source")
    fx, fy, fs = C.PW.capped_outer(data, model, task, fold, "future")
    ox = np.concatenate((sx, fx))
    osub = np.concatenate((ss, fs)).astype(str)
    ose = np.concatenate((np.full(len(sx), int(data["source_session"])),
                          np.full(len(fx), int(data["future_session"])))).astype(int)
    th, tz, _ = C.UP.hook_representations(net, head, tx, model, device)
    oh, oz, _ = C.UP.hook_representations(net, head, ox, model, device)
    tq, tz0, tzp, tzc = C.BR.decompose(th, tz, spec, dims, head)
    oq, oz0, ozp, ozc = C.BR.decompose(oh, oz, spec, dims, head)
    if max(np.max(np.abs(tz - (tz0 + tzp + tzc))), np.max(np.abs(oz - (oz0 + ozp + ozc)))) >= 1e-5:
        raise RuntimeError("final exact P+C decomposition mismatch")
    coupling_train: dict[str, np.ndarray] = {}
    coupling_future: dict[str, np.ndarray] = {}
    pair_counts: dict[str, dict[str, int]] = {}
    for stage in runner.names:
        cached_path = feature_stage_path(model, task, fold, stage)
        if cached_path.exists():
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            if (cached.get("source_cell_sha256") != source_hash or
                    cached.get("repair_implementation_sha256") != C.digest(Path(__file__)) or
                    cached.get("protocol_sha256") != PROTOCOL_SHA or cached.get("donor_rule") != RULE):
                raise RuntimeError(f"stale repair stage cache: {stage}")
            train_feature = np.asarray(cached["train_features"], np.float32)
            future_feature = np.asarray(cached["future_features"], np.float32)
            counts = cached["pair_counts"]
            print("REPAIR_STAGE_CACHED", model, task, fold, stage, flush=True)
        else:
            print("REPAIR_STAGE_START", model, task, fold, stage, flush=True)
            train_feature, future_feature, counts = stage_features(
                runner, model, task, fold, stage, source_cell, train_centroids,
                canonical_centroids, spec, dims, tx, ts, tse, ox, osub, ose, len(sx))
            C.write_json(cached_path, {"source_cell_sha256": source_hash,
                "repair_implementation_sha256": C.digest(Path(__file__)),
                "protocol_sha256": PROTOCOL_SHA, "donor_rule": RULE,
                "train_features": train_feature, "future_features": future_feature,
                "pair_counts": counts})
            print("REPAIR_STAGE_COMPLETE", model, task, fold, stage, flush=True)
        coupling_train[stage] = train_feature
        coupling_future[stage] = future_feature
        pair_counts[stage] = counts
    geometry_train = C.final_geometry(tq, tz0, tzp, tzc, tz, dims)
    geometry_future = C.final_geometry(oq[len(sx):], oz0, ozp[len(sx):], ozc[len(sx):], oz[len(sx):], dims)
    targets_train = C.error_targets(ty, tz, tz0 + tzp, tz0 + tzc,
                                    C.previous_grid_nonrecoverable(ty, tz, tz0, tzp, tzc))
    targets_future = C.error_targets(fy, oz[len(sx):], oz0 + ozp[len(sx):], oz0 + ozc[len(sx):],
                                     C.previous_grid_nonrecoverable(fy, oz[len(sx):], oz0, ozp[len(sx):], ozc[len(sx):]))
    prediction_rows = C.error_predictability(geometry_train, geometry_future,
        coupling_train, coupling_future, targets_train, targets_future, ts, fs,
        (model, task, fold, "label-free-predicted-class-v2"))
    C.write_json(destination, {"status": "COMPLETE", "model": model, "task": task,
        "fold": fold, "seed": 0, "protocol_sha256": PROTOCOL_SHA,
        "original_implementation_sha256": ORIGINAL_IMPL_SHA,
        "repair_implementation_sha256": C.digest(Path(__file__)),
        "source_cell_sha256": source_hash, "donor_rule": RULE,
        "outer_true_labels_used_for_features": False,
        "outer_true_labels_used_for_targets_and_metrics_only": True,
        "post_outcome_engineering_correction": True,
        "final_heldout_accessed": False, "backbone_training": False,
        "native_head_refit": False, "protected_reselection": False,
        "pair_counts": pair_counts, "prediction_rows": prediction_rows})
    print("REPAIR_CELL_COMPLETE", model, task, fold, flush=True)


def add_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {(r["model"], r["task"], r["fold"], r["target"], r["subject_id"]): r
                 for r in rows if r["family"] == "BASE"}
    for row in rows:
        base = baselines[(row["model"], row["task"], row["fold"], row["target"], row["subject_id"])]
        for metric in ("AUROC", "AUPRC", "Brier", "ECE_10bin"):
            value, original = row[metric], base[metric]
            row[f"delta_{metric}_over_BASE"] = (value - original) if value is not None and original is not None else None


def aggregate() -> None:
    if REPAIR_OUT.exists() and any(REPAIR_OUT.iterdir()):
        raise RuntimeError("corrected aggregate already exists; preserve and inspect it")
    original_out = C.OUT
    if not (original_out / "PROVENANCE.json").exists():
        raise RuntimeError("original aggregate missing")
    rows: list[dict[str, Any]] = []
    provenance: dict[str, str] = {}
    for model in C.MODELS:
        for task in C.TASKS:
            for fold in C.FOLDS:
                original_cell = C.cell_path(model, task, fold)
                repair_path = repaired_cell_path(model, task, fold)
                repair = json.loads(repair_path.read_text(encoding="utf-8"))
                if (repair.get("status") != "COMPLETE" or repair.get("protocol_sha256") != PROTOCOL_SHA or
                        repair.get("original_implementation_sha256") != ORIGINAL_IMPL_SHA or
                        repair.get("repair_implementation_sha256") != C.digest(Path(__file__)) or
                        repair.get("source_cell_sha256") != C.digest(original_cell) or
                        repair.get("outer_true_labels_used_for_features") is not False or
                        repair.get("final_heldout_accessed")):
                    raise RuntimeError(f"repaired cell invalid: {model}/{task}/{fold}")
                provenance[f"{model}/{task}/{fold}"] = C.digest(repair_path)
                rows.extend({"model": model, "task": task, "fold": fold, "seed": 0, **r}
                            for r in repair["prediction_rows"])
    add_deltas(rows)
    with (original_out / "MODEL_TASK_SUMMARY.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    report = (original_out / "REPORT.md").read_text(encoding="utf-8")
    for item in summary:
        model, task = item["model"], item["task"]
        group = [r for r in rows if (r["model"], r["task"], r["target"]) == (model, task, "native_error")]
        base = [r["AUROC"] for r in group if r["family"] == "BASE" and r["AUROC"] is not None]
        full = [r["AUROC"] for r in group if r["family"] == "ALL_LAYER" and r["AUROC"] is not None]
        by_subject: dict[str, list[float]] = defaultdict(list)
        for r in group:
            if r["family"] == "ALL_LAYER" and r["delta_AUROC_over_BASE"] is not None:
                by_subject[r["subject_id"]].append(r["delta_AUROC_over_BASE"])
        point, low, high = C.subject_bootstrap(by_subject, (model, task, "native-error", "label-free-v2"))
        item.update({"native_error_AUROC_BASE": float(np.mean(base)) if base else None,
                     "native_error_AUROC_ALL_LAYER": float(np.mean(full)) if full else None,
                     "native_error_delta_AUROC": point,
                     "native_error_delta_AUROC_CI_low": low,
                     "native_error_delta_AUROC_CI_high": high,
                     "error_prediction_provenance": "label_free_predicted_class_v2_post_outcome_exploratory"})
        pattern = rf"(## {re.escape(model)} / {re.escape(task)}.*?\n9\. )[^\n]*"
        replacement = (rf"\g<1>Corrected label-free future-session native-error AUROC: geometry BASE "
                       f"{item['native_error_AUROC_BASE']:.6g}, ALL_LAYER {item['native_error_AUROC_ALL_LAYER']:.6g}; "
                       f"paired subject delta {point:.6g} (95% CI [{low:.6g}, {high:.6g}]). "
                       "This post-outcome engineering correction is exploratory, not a pre-registered success-criterion C test.")
        report, count = re.subn(pattern, replacement, report, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError(f"report prediction section missing: {model}/{task}")
    intro = ("\n## Prediction-feature engineering correction v2\n\n"
             "The original outer-development error-prediction features used true class labels to select donors, "
             "so their AUROC values are invalid as prospective evidence and are preserved only in the original aggregate. "
             "The corrected analysis selects same-subject/session donors by frozen native predicted class and nearest "
             "activation norm, without true labels in feature construction. True labels enter only TRAIN targets and "
             "outer evaluation metrics. The correction was specified after the original outer outcomes had been viewed; "
             "therefore its future-error gains are exploratory and do not independently satisfy success criterion C. "
             "The original class-conditioned 2x2 mechanism and random-partition results are unchanged.\n\n")
    report = report.replace("## EEGNet / OpenBMI_MI", intro + "## EEGNet / OpenBMI_MI", 1)
    C.write_csv(REPAIR_OUT / "PC_ERROR_PREDICTABILITY.csv", rows)
    C.write_csv(REPAIR_OUT / "MODEL_TASK_SUMMARY.csv", summary)
    (REPAIR_OUT / "REPORT.md").write_text(report, encoding="utf-8")
    C.write_json(REPAIR_OUT / "PROVENANCE.json", {
        "schema": "PERSIST_EEG_PROTECTED_COMPLEMENT_COUPLING_PREDICTION_REPAIR_V2",
        "frozen_protocol_sha256": PROTOCOL_SHA,
        "original_implementation_sha256": ORIGINAL_IMPL_SHA,
        "repair_implementation_sha256": C.digest(Path(__file__)),
        "original_aggregate_provenance_sha256": C.digest(original_out / "PROVENANCE.json"),
        "donor_rule": RULE, "outer_true_labels_used_for_features": False,
        "post_outcome_engineering_correction": True,
        "criterion_C_confirmatory_eligible": False,
        "final_heldout_accessed": False, "cells_complete": 20,
        "repair_cell_sha256": provenance,
        "output_sha256": {name: C.digest(REPAIR_OUT / name) for name in
            ("PC_ERROR_PREDICTABILITY.csv", "MODEL_TASK_SUMMARY.csv", "REPORT.md")},
    })
    print("PREDICTION_REPAIR_AGGREGATE_COMPLETE", len(provenance), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("cell", "aggregate"))
    parser.add_argument("--model", choices=C.MODELS)
    parser.add_argument("--task", choices=C.TASKS)
    parser.add_argument("--fold", type=int, choices=C.FOLDS)
    args = parser.parse_args()
    if C.digest(C.EXP / "protocol" / "PROVENANCE.json") != PROTOCOL_SHA:
        raise RuntimeError("frozen protocol lock mismatch")
    if C.digest(C.EXP / "code" / "run_coupling.py") != ORIGINAL_IMPL_SHA:
        raise RuntimeError("original implementation was modified")
    if args.mode == "aggregate":
        aggregate()
    else:
        if args.model is None or args.task is None or args.fold is None:
            parser.error("cell mode needs model, task, fold")
        try:
            repair_cell(args.model, args.task, args.fold)
        except Exception as error:
            failed = repaired_cell_path(args.model, args.task, args.fold).with_suffix(".FAIL_CLOSED.json")
            if not failed.exists():
                C.write_json(failed, {"status": "FAIL_CLOSED", "reason": f"{type(error).__name__}: {error}",
                    "model": args.model, "task": args.task, "fold": args.fold,
                    "protocol_sha256": PROTOCOL_SHA,
                    "repair_implementation_sha256": C.digest(Path(__file__)),
                    "final_heldout_accessed": False})
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
