#!/usr/bin/env python3
"""Outcome-blind MI-first Stage-A amendment for LiteBN-X seed0.

This runner preserves the frozen data/split/training recipe while reducing the
remaining search by applying a predeclared MI-only inner-validation gate before
ERP/SSVEP training. It never opens final held-out/test cohorts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(os.environ.get("LITEBN_X_REPO", "/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")).resolve()
CODE = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py"
EXP = REPO / "experiments/persist_eeg_litebn_x_singlemodel_seed0_v1"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("LITEBN_X_RUNTIME", "/root/rivermind-data/litebn_x_singlemodel_seed0_runtime")).resolve()
ARCHES = ("LiteBN_BASELINE", "LiteBN_R", "LiteBN_RG", "LiteBN_X", "LiteBN_XS")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")


def load_impl():
    spec = importlib.util.spec_from_file_location("litebn_x_impl", CODE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUNTIME = RUNTIME
    return module


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def old_openbmi_mi_records(mod):
    """Recover completed OpenBMI-MI inner-val records from latest checkpoints."""
    rows = []
    root = RUNTIME / "checkpoints" / "openbmi_mi"
    search, folds, split_hash = mod.load_folds()
    for fold in folds["OpenBMI"]:
        for architecture in ARCHES:
            directory = root / f"fold{int(fold['fold_id'])}_{architecture.lower()}"
            latest = directory / "checkpoint_latest.pt"
            selected = directory / "selected_best.pt"
            if not latest.is_file() or not selected.is_file():
                raise RuntimeError(f"missing completed OpenBMI MI checkpoint: {directory}")
            payload = torch.load(latest, map_location="cpu", weights_only=False)
            history = list(payload.get("history", []))
            eligible = [row for row in history if int(row["epoch"]) >= mod.MIN_EPOCH]
            if not eligible or int(payload.get("epoch", 0)) < mod.EPOCHS:
                raise RuntimeError(f"OpenBMI MI cell is not complete: {directory}")
            best = max(eligible, key=lambda row: float(row["inner_val_subject_BA"]))
            rows.append({
                "task": "OpenBMI_MI", "dataset": "OpenBMI", "fold": int(fold["fold_id"]),
                "architecture": architecture, "seed": int(mod.SEED),
                "parameter_count": int(mod.parameter_count(mod.build_model(architecture, 62 if False else "OpenBMI_MI"))),
                "selected_epoch": int(best["epoch"]),
                "best_inner_val_BA": float(best["inner_val_subject_BA"]),
                "best_inner_val_macro_F1": float(best["inner_val_subject_macro_F1"]),
                "checkpoint_path": str(selected),
                "checkpoint_sha256": mod.sha256_file(selected),
                "normalizer_sha256": "recovered_from_existing_stage_a_checkpoint",
                "batch_manifest_sha256": None,
                "class_weighted_ce": False,
                "source_sha256": mod.sha256_file(CODE),
                "recovered": True,
            })
    return rows, split_hash


def train_cells(mod, tasks, architectures, split_hash):
    search, folds, _ = mod.load_folds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, records = [], []
    for task in tasks:
        dataset = mod.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
            bundle = mod.build_bundle(task, allowed)
            mean, std, norm_meta = mod.normalizer(bundle, fold["inner_train_subjects"])
            mod.save_tensor_pair(RUNTIME / "normalizers" / f"{task.lower()}_fold{fold['fold_id']}.npz", mean, std, norm_meta)
            cache = mod.RawGPUCache(bundle, device)
            if mod.TASKS[task]["mi_protocol"]:
                episodes, manifest_meta = mod.mi_manifest(bundle, fold, task)
                batch_info = {**manifest_meta, "episodes": episodes}
            else:
                batch_info = {"kind": "historical_task_full_permutation_batch64", "manifest_sha256": None, "steps_per_epoch": None}
            weight, weight_meta = mod.class_weights(bundle, fold["inner_train_subjects"])
            for architecture in architectures:
                mod.set_seed(mod.SEED)
                model = mod.build_model(architecture, task).to(device)
                mod.set_seed(mod.SEED + 100_000)
                record = mod.train_one(model, architecture, task, fold, bundle, cache, mean, std, norm_meta, batch_info, weight, weight_meta, device)
                records.append(record)
                rows.append({key: value for key, value in record.items() if key != "history"})
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows, records


def mi_phase(mod):
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    old_rows, split_hash = old_openbmi_mi_records(mod)
    wbcic_rows, wbcic_records = train_cells(mod, ("WBCIC_MI",), ARCHES, split_hash)
    all_rows = pd.DataFrame(old_rows + wbcic_rows).sort_values(["task", "fold", "architecture"])
    mod.write_csv(OUT / "MI_TRIAGE_RESULTS.csv", all_rows)
    base = all_rows[all_rows.architecture == "LiteBN_BASELINE"].groupby("task").best_inner_val_BA.mean()
    summary = []
    for architecture in ARCHES[1:]:
        means = all_rows[all_rows.architecture == architecture].groupby("task").best_inner_val_BA.mean()
        delta = ((means - base) * 100.0).reindex(("OpenBMI_MI", "WBCIC_MI"))
        keep = bool(float(delta.mean()) > 0.0 and float(delta.min()) >= -0.50)
        summary.append({"architecture": architecture, "OpenBMI_MI_delta_pp": float(delta.iloc[0]), "WBCIC_MI_delta_pp": float(delta.iloc[1]), "mi_equal_mean_delta_pp": float(delta.mean()), "mi_worst_delta_pp": float(delta.min()), "retained_for_remaining_tasks": keep})
    summary_frame = pd.DataFrame(summary).sort_values(["retained_for_remaining_tasks", "mi_equal_mean_delta_pp", "architecture"], ascending=[False, False, True])
    retained = ["LiteBN_BASELINE"] + summary_frame.loc[summary_frame.retained_for_remaining_tasks, "architecture"].tolist()
    no_robust = len(retained) == 1
    if no_robust:
        retained.append(str(summary_frame.iloc[0].architecture))
    mod.write_csv(OUT / "MI_ARCHITECTURE_SUMMARY.csv", summary_frame)
    lock = {"amendment": "MI_FIRST_TRIAGE", "seed": int(mod.SEED), "split_sha256": split_hash, "mi_cells_complete": True, "mi_outcomes_read": True, "retained_architectures": retained, "mi_no_robust_survivor": no_robust, "rule": "retain candidate iff mean(OpenBMI_MI,WBCIC_MI) inner-val delta > 0 pp and each MI-task delta >= -0.50 pp; always retain exact LiteBN_BASELINE; if none survive, retain top candidate diagnostically", "final_holdout_accessed": False, "outer_dev_accessed": False, "source_sha256": mod.sha256_file(CODE)}
    write_json(PROTOCOL / "MI_TRIAGE_LOCK.json", lock)
    provenance = {"stage": "MI_TRIAGE", "seed": int(mod.SEED), "records": [{k: v for k, v in row.items() if k != "history"} for row in old_rows + wbcic_records]}
    write_json(PROTOCOL / "MI_TRIAGE_CHECKPOINT_PROVENANCE.json", provenance)
    print("MI_TRIAGE_READY", json.dumps({"retained_architectures": retained, "summary": summary}, sort_keys=True), flush=True)


def remaining_phase(mod):
    lock = read_json(PROTOCOL / "MI_TRIAGE_LOCK.json")
    if not lock.get("mi_cells_complete") or lock.get("final_holdout_accessed"):
        raise RuntimeError("MI triage lock missing or invalid")
    split_hash = lock["split_sha256"]
    old_rows, old_hash = old_openbmi_mi_records(mod)
    if old_hash != split_hash:
        raise RuntimeError("split hash mismatch")
    triage_prov = read_json(PROTOCOL / "MI_TRIAGE_CHECKPOINT_PROVENANCE.json")["records"]
    retained = tuple(lock.get("downstream_architectures", lock["retained_architectures"]))
    remaining_rows, remaining_records = train_cells(mod, ("OpenBMI_ERP", "OpenBMI_SSVEP"), retained, split_hash)
    all_records = old_rows + triage_prov + remaining_records
    frame = pd.DataFrame(old_rows + [r for r in triage_prov if r["task"] == "WBCIC_MI"] + remaining_rows)
    frame = frame.sort_values(["task", "fold", "architecture"])
    if frame.duplicated(["task", "fold", "architecture"]).any():
        raise RuntimeError("duplicate combined Stage-A cell")
    base = frame[frame.architecture == "LiteBN_BASELINE"].groupby("task").best_inner_val_BA.mean()
    rows = []
    for architecture in retained[1:]:
        means = frame[frame.architecture == architecture].groupby("task").best_inner_val_BA.mean()
        delta = ((means - base) * 100.0).reindex(TASKS)
        rows.append({"architecture": architecture, **{f"{task}_inner_val_BA": float(means.loc[task]) for task in TASKS}, **{f"{task}_delta_pp": float(delta.loc[task]) for task in TASKS}, "equal_task_mean_delta_pp": float(delta.mean()), "worst_task_delta_pp": float(delta.min()), "eligible": bool(delta.mean() > 0.0 and delta.min() >= -0.50)})
    summary = pd.DataFrame(rows).sort_values(["eligible", "equal_task_mean_delta_pp", "architecture"], ascending=[False, False, True]).reset_index(drop=True)
    pool = summary[summary.eligible] if len(summary[summary.eligible]) else summary
    top = float(pool.equal_task_mean_delta_pp.max())
    chosen = pool[pool.equal_task_mean_delta_pp >= top - 0.10].sort_values("architecture").iloc[0]
    selection = {"selected_architecture": str(chosen.architecture), "stage_a_no_robust_winner": not bool(len(summary[summary.eligible])), "selection_rule": "MI-first amendment: MI survivor gate was predeclared; among retained architectures, full four-task inner-val eligibility requires mean delta >0 and worst task >=-0.50 pp; highest equal-task mean, 0.10 pp tie resolved lexicographically.", "selected_equal_task_mean_delta_pp": float(chosen.equal_task_mean_delta_pp), "selected_worst_task_delta_pp": float(chosen.worst_task_delta_pp)}
    mod.write_csv(OUT / "STAGE_A_INNERVAL_RESULTS.csv", frame)
    mod.write_csv(OUT / "STAGE_A_ARCHITECTURE_SUMMARY.csv", summary)
    write_json(PROTOCOL / "CHECKPOINT_PROVENANCE.json", {"stage": "A_MI_FIRST", "seed": int(mod.SEED), "expected_cells": int(len(frame)), "records": [{k: v for k, v in r.items() if k != "history"} for r in all_records]})
    table = ["# Stage-A selection (MI-first amendment)", "", "The predeclared MI-first gate retained the architectures listed in MI_TRIAGE_LOCK.json before ERP/SSVEP training.", "", f"Selected architecture: {selection['selected_architecture']}", f"Stage-A no robust winner: {selection['stage_a_no_robust_winner']}", f"Source hash: {mod.sha256_file(CODE)}", f"Split hash: {split_hash}", "", "| Architecture | Equal-task delta (pp) | Worst-task delta (pp) | Eligible |", "|---|---:|---:|---|"]
    for row in summary.itertuples(index=False):
        table.append(f"| {row.architecture} | {row.equal_task_mean_delta_pp:+.3f} | {row.worst_task_delta_pp:+.3f} | {bool(row.eligible)} |")
    mod.write_text(PROTOCOL / "STAGE_A_SELECTION.md", "\n".join(table))
    write_json(PROTOCOL / "STAGE_A_LOCK.json", {"stage": "A_MI_FIRST", "seed": int(mod.SEED), "stage_a_complete": True, "outer_dev_predictions_generated": False, "final_holdout_predictions_generated": False, "source_sha256": mod.sha256_file(CODE), "split_sha256": split_hash, **selection, "mi_first_amendment": True})
    print("STAGE_A_MI_FIRST_READY", json.dumps(selection, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("mi", "remaining"))
    args = parser.parse_args()
    mod = load_impl()
    if args.phase == "mi":
        return mi_phase(mod)
    return remaining_phase(mod)


if __name__ == "__main__":
    main()
