#!/usr/bin/env python3
"""Extend the frozen LiteBN CE versus CE+PRD study to seeds 0, 1, 2."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(os.environ.get("PRD_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_prd_multiseed_v1"
OUT, RUN, PROTOCOL = EXP / "outputs", EXP / "runtime", EXP / "protocol"
SEED0_EXP = REPO / "experiments/persist_eeg_litebn_prd_seed0_v1"
SOURCE = SEED0_EXP / "code/run_litebn_prd_seed0.py"
SEEDS = (0, 1, 2)
BOOTSTRAPS = 20_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prd = load_module("litebn_prd_seed0_source", SOURCE)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stable(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def seed0_paths() -> dict[tuple, Path]:
    paths = {}
    for task in prd.TASKS:
        for fold in prd.FOLDS:
            for condition in prd.CONDITIONS:
                path = SEED0_EXP / f"runtime/checkpoints/{task}/{condition}/fold{fold}/selected.pt"
                if not path.is_file():
                    raise FileNotFoundError(path)
                paths[(task, fold, condition)] = path
    return paths


def train_seed(seed: int, device, fold_map) -> tuple[dict, list[dict]]:
    if seed == 0:
        paths = seed0_paths()
        audit = []
        for (task, fold, condition), path in paths.items():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            audit.append({"task": task, "fold": fold, "seed": seed, "condition": condition,
                          "initial_hash": payload["initial_hash"], "manifest_hash": payload["manifest_hash"],
                          "selected_epoch": payload["selected_epoch"], "inner_val_BA": payload["inner_val_BA"],
                          "checkpoint": str(path), "reused_completed_seed0": True})
        return paths, audit
    prd.RUN = RUN / f"seed{seed}"
    paths, audit = {}, []
    for task in prd.TASKS:
        dataset = prd.base_global.TASKS[task]["dataset"]
        for fold in fold_map[dataset]:
            fold_id = int(fold["fold_id"])
            bundle = prd.base_global.build_bundle(task, fold["inner_train_subjects"] + fold["inner_val_subjects"])
            mean, std, _ = prd.base_global.load_tensor_pair(prd.csgd_global.normalizer_path(task, fold_id))
            raw = prd.base_global.RawGPUCache(bundle, device)
            cache = prd.base_runner.NormalizedCache(raw, mean, std)
            manifest = prd.make_manifest(bundle, fold, task)
            prd.seed(seed)
            template = prd.base_global.build_model("LiteBN_BASELINE", task)
            initial = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
            for condition in prd.CONDITIONS:
                prd.seed(seed)
                path = prd.train_one(task, fold, bundle, cache, manifest, condition, initial, device)
                paths[(task, fold_id, condition)] = path
                payload = torch.load(path, map_location="cpu", weights_only=False)
                audit.append({"task": task, "fold": fold_id, "seed": seed, "condition": condition,
                              "initial_hash": payload["initial_hash"], "manifest_hash": payload["manifest_hash"],
                              "selected_epoch": payload["selected_epoch"], "inner_val_BA": payload["inner_val_BA"],
                              "checkpoint": str(path), "reused_completed_seed0": False})
            pair = audit[-2:]
            if pair[0]["initial_hash"] != pair[1]["initial_hash"] or pair[0]["manifest_hash"] != pair[1]["manifest_hash"]:
                raise RuntimeError(f"matching invariant failed seed={seed} task={task} fold={fold_id}")
            del cache, raw, bundle
            gc.collect(); torch.cuda.empty_cache()
    return paths, audit


def evaluate_seed(seed: int, paths: dict, runtime, device):
    rows, alignment = prd.evaluate_all(prd.csgd_global, runtime, paths, device)
    for row in rows:
        row["seed"] = seed
    for row in alignment:
        row["seed"] = seed
    return rows, alignment


def bootstrap(values, key: str):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable("prd-multiseed-bootstrap", key))
    means = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 2000):
        stop = min(BOOTSTRAPS, start + 2000)
        idx = rng.integers(0, len(values), size=(stop-start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return float(values.mean()), float(low), float(high)


def aggregate(rows, alignment):
    frame = pd.DataFrame(rows)
    session = frame.groupby(["task", "condition", "seed", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), folds=("fold", "nunique"))
    if not (session.folds == 5).all():
        raise RuntimeError("incomplete PRD fold coverage")
    af = pd.DataFrame(alignment).groupby(["task", "condition", "seed", "subject_id"], as_index=False).alignment.mean()
    subject_seed = []
    for (task, condition, seed, subject), part in session.groupby(["task", "condition", "seed", "subject_id"]):
        cells = {r.session: r for r in part.itertuples(index=False)}
        required = ("S0", "S1", "S2") if task == "WBCIC_MI" else ("S1", "S2")
        value = af[(af.task == task) & (af.condition == condition) & (af.seed == seed) & (af.subject_id == subject)]
        subject_seed.append({"task": task, "condition": condition, "seed": seed, "subject_id": subject,
                             "alignment": float(value.alignment.iloc[0]), "future_BA": cells["S2"].BA,
                             "future_macro_F1": cells["S2"].macro_F1, "WS_BA": min(cells[s].BA for s in required)})
    ss = pd.DataFrame(subject_seed)
    seed_summary = ss.groupby(["task", "condition", "seed"], as_index=False).agg(
        biological_subjects=("subject_id", "nunique"), alignment=("alignment", "mean"),
        future_BA=("future_BA", "mean"), future_macro_F1=("future_macro_F1", "mean"), WS_BA=("WS_BA", "mean"))
    biological_session = session.groupby(["task", "condition", "subject_id", "session"], as_index=False).agg(
        BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), seeds=("seed", "nunique"))
    if not (biological_session.seeds == 3).all():
        raise RuntimeError("not every PRD subject has three seeds")
    biological_rows = []
    for (task, condition, subject), part in biological_session.groupby(["task", "condition", "subject_id"]):
        cells = {r.session: r for r in part.itertuples(index=False)}
        required = ("S0", "S1", "S2") if task == "WBCIC_MI" else ("S1", "S2")
        if set(cells) != set(required):
            raise RuntimeError(f"biological session mismatch {task}/{condition}/{subject}")
        alignment_mean = ss[(ss.task == task) & (ss.condition == condition) & (ss.subject_id == subject)].alignment.mean()
        biological_rows.append({"task": task, "condition": condition, "subject_id": subject,
                                "seeds": 3, "alignment": float(alignment_mean),
                                "future_BA": cells["S2"].BA, "future_macro_F1": cells["S2"].macro_F1,
                                "WS_BA": min(cells[s].BA for s in required)})
    biological = pd.DataFrame(biological_rows)
    summary = []
    for task in prd.TASKS:
        for condition in prd.CONDITIONS:
            part = biological[(biological.task == task) & (biological.condition == condition)]
            row = {"task": task, "condition": condition, "biological_subjects": len(part)}
            for metric in ("alignment", "future_BA", "future_macro_F1", "WS_BA"):
                mean, low, high = bootstrap(part[metric].to_numpy(), f"{task}/{condition}/{metric}/absolute")
                row[metric] = mean; row[f"{metric}_CI95_low"] = low; row[f"{metric}_CI95_high"] = high
            summary.append(row)
        ce = biological[(biological.task == task) & (biological.condition == "CE")].set_index("subject_id")
        pp = biological[(biological.task == task) & (biological.condition == "CE_PLUS_PRD")].set_index("subject_id")
        ids = sorted(set(ce.index) & set(pp.index))
        row = {"task": task, "condition": "DELTA_PRD_MINUS_CE", "biological_subjects": len(ids)}
        for metric in ("alignment", "future_BA", "future_macro_F1", "WS_BA"):
            scale = 1 if metric == "alignment" else 100
            delta = scale * (pp.loc[ids, metric].to_numpy() - ce.loc[ids, metric].to_numpy())
            mean, low, high = bootstrap(delta, f"{task}/{metric}/delta")
            row[f"delta_{metric}"] = mean; row[f"delta_{metric}_CI95_low"] = low; row[f"delta_{metric}_CI95_high"] = high
        summary.append(row)
    return session.to_dict("records"), subject_seed, seed_summary.to_dict("records"), biological.to_dict("records"), summary


def report(summary):
    lines = ["# LiteBN CE versus CE+PRD, seeds 0, 1, 2", "", "PRD lambda=0.25 was reused without tuning. Fold and seed repetitions were averaged within each biological subject and session; WS-BA is the minimum of those session means. The 20,000-draw bootstrap resamples biological subjects.", "",
             "OpenBMI V8 is an internal-heldout diagnostic cohort. The WBCIC true-outer cohort has already been accessed for prior ablation diagnostics and is therefore an exposed benchmark, not untouched final confirmation.", "",
             "| Task | Condition | Alignment | Future BA | Macro-F1 | WS-BA |", "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        if r["condition"] != "DELTA_PRD_MINUS_CE":
            lines.append(f"| {r['task']} | {r['condition']} | {r['alignment']:.4f} | {r['future_BA']:.4f} | {r['future_macro_F1']:.4f} | {r['WS_BA']:.4f} |")
    lines += ["", "| Task | Delta alignment [95% CI] | Delta BA pp [95% CI] | Delta Macro-F1 pp [95% CI] | Delta WS-BA pp [95% CI] |", "|---|---:|---:|---:|---:|"]
    for r in summary:
        if r["condition"] == "DELTA_PRD_MINUS_CE":
            lines.append(f"| {r['task']} | {r['delta_alignment']:+.4f} [{r['delta_alignment_CI95_low']:+.4f}, {r['delta_alignment_CI95_high']:+.4f}] | {r['delta_future_BA']:+.3f} [{r['delta_future_BA_CI95_low']:+.3f}, {r['delta_future_BA_CI95_high']:+.3f}] | {r['delta_future_macro_F1']:+.3f} [{r['delta_future_macro_F1_CI95_low']:+.3f}, {r['delta_future_macro_F1_CI95_high']:+.3f}] | {r['delta_WS_BA']:+.3f} [{r['delta_WS_BA_CI95_low']:+.3f}, {r['delta_WS_BA_CI95_high']:+.3f}] |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    requested = tuple(int(x) for x in args.seeds.split(",") if x)
    if any(seed not in SEEDS for seed in requested):
        raise ValueError(requested)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    for directory in (OUT, RUN, PROTOCOL):
        directory.mkdir(parents=True, exist_ok=True)
    prd.base_runner = prd.module("prd_multiseed_base_runner", prd.BASE_RUN)
    prd.base_global = prd.base_runner.base
    prd.csgd_global = prd.module("prd_multiseed_csgd", prd.CSGD_RUN)
    runtime = prd.csgd_global.load_runtime()
    _, fold_map, split_sha = prd.base_global.load_folds()
    if args.preflight:
        print(f"PRD_MULTISEED_PREFLIGHT_OK seed0_cells={len(seed0_paths())} split={split_sha}", flush=True)
        return
    if args.aggregate_only:
        session_path, alignment_path = RUN / "SESSION_RESULTS.csv", RUN / "ALIGNMENT_RESULTS.csv"
        if not session_path.is_file() or not alignment_path.is_file():
            raise FileNotFoundError("frozen inference cache missing")
        all_rows = pd.read_csv(session_path).to_dict("records")
        all_alignment = pd.read_csv(alignment_path).to_dict("records")
        audits = pd.read_csv(PROTOCOL / "MATCHING_AUDIT.csv").to_dict("records")
    else:
        device = torch.device("cuda")
        all_rows, all_alignment, audits = [], [], []
        path_by_seed = {}
        for seed in requested:
            paths, audit = train_seed(seed, device, fold_map)
            path_by_seed[seed] = paths; audits.extend(audit)
        if set(path_by_seed) != set(SEEDS):
            raise RuntimeError("all three seeds are required before heldout evaluation")
        print("PRD_MULTISEED_ALL_CHECKPOINTS_FROZEN", flush=True)
        for seed in SEEDS:
            rows, alignment = evaluate_seed(seed, path_by_seed[seed], runtime, device)
            all_rows.extend(rows); all_alignment.extend(alignment)
            atomic_csv(RUN / "SESSION_RESULTS.csv", all_rows)
            atomic_csv(RUN / "ALIGNMENT_RESULTS.csv", all_alignment)
            print(f"PRD_MULTISEED_EVAL seed={seed}", flush=True)
    session, subject_seed, seed_summary, biological, summary = aggregate(all_rows, all_alignment)
    atomic_csv(OUT / "SESSION_SUBJECT_RESULTS.csv", session)
    atomic_csv(OUT / "SUBJECT_SEED_RESULTS.csv", subject_seed)
    atomic_csv(OUT / "TASK_SEED_SUMMARY.csv", seed_summary)
    atomic_csv(OUT / "BIOLOGICAL_SUBJECT_RESULTS.csv", biological)
    atomic_csv(OUT / "MULTISEED_TASK_SUMMARY.csv", summary)
    atomic_csv(PROTOCOL / "MATCHING_AUDIT.csv", audits)
    atomic_json(PROTOCOL / "PROTOCOL.json", {"seeds": SEEDS, "tasks": prd.TASKS, "folds": list(prd.FOLDS),
        "conditions": prd.CONDITIONS, "architecture": "exact final LiteBN", "lambda": prd.LAMBDA,
        "lambda_source": str(prd.PRD_CODE / "run_fold0.py"), "prd_definition": str(prd.PRD_CODE / "prd_loss.py"),
        "optimizer": "AdamW", "lr": prd.LR, "weight_decay": prd.WD, "gradient_clip": prd.CLIP,
        "max_epochs": prd.MAX_EPOCHS, "min_checkpoint_epoch": prd.MIN_EPOCH, "patience": prd.PATIENCE,
        "identical_initialization_within_seed": True, "identical_CE_batches_within_seed": True,
        "seed0_reused": True, "heldout_used_for_training_or_selection": False,
        "bootstrap_unit": "biological subject", "bootstrap_draws": BOOTSTRAPS})
    atomic_text(OUT / "FINAL_LITEBN_PRD_MULTISEED_REPORT.md", report(summary))
    atomic_json(OUT / "COMPLETION.json", {"status": "COMPLETE", "training_cells": 60, "missing_cells": 0,
        "new_training_cells": 40, "reused_seed0_cells": 20, "seeds": SEEDS,
        "heldout_evaluated_after_all_checkpoints_frozen": True})
    print("LITEBN_PRD_MULTISEED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
