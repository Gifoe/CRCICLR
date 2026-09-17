#!/usr/bin/env python3
"""Run only the missing Full EEGNet/SIRE BNCI seeds under final semantics.

Seed 0 is frozen and reused.  Seeds 1/2 use the final benchmark convention:
the predeclared five subject folds and the authoritative episode manifests stay
fixed, while initialization and training RNG are respectively ``seed`` and
``100000 + seed``.  No architecture ablation is instantiated here.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("BNCI_P4_ROOT", "/root/p4_bnci2015_001_seed0_matched_episodic_v1"))
BASE_PATH = ROOT / "code/run_seed0_matched_episodic.py"
OUT = ROOT / "three_seed_nonablation/primary"
SEEDS = (0, 1, 2)


def imported(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fixed_folds(base):
    old = base.SEED
    try:
        base.SEED = 0
        return base.make_splits()
    finally:
        base.SEED = old


def run_missing(seed: int) -> None:
    base = imported(f"bnci_final_seed_{seed}", BASE_PATH)
    folds, expected_rows = fixed_folds(base)
    frozen_rows = list(csv.DictReader((ROOT / "subject_split_manifest.csv").open(encoding="utf-8")))
    expected_membership = sorted((str(r["fold"]), r["role"], r["subject"]) for r in expected_rows)
    frozen_membership = sorted((r["fold"], r["role"], r["subject"]) for r in frozen_rows)
    if expected_membership != frozen_membership:
        raise RuntimeError("final fixed seed-0 split differs from frozen BNCI split")
    original_build = base.build_manifest
    def fixed_manifest(stage1, bundle, fold):
        old = base.SEED
        try:
            base.SEED = 0
            return original_build(stage1, bundle, fold)
        finally:
            base.SEED = old
    base.make_splits = lambda: (folds, [{**r, "seed": seed} for r in expected_rows])
    base.build_manifest = fixed_manifest
    base.SEED, base.TRAINING_SEED = seed, 100000 + seed
    base.ROOT = OUT / "runtime" / f"seed{seed}"
    base.run()


def source_for(seed: int) -> Path:
    return ROOT if seed == 0 else OUT / "runtime" / f"seed{seed}"


def aggregate() -> None:
    metrics, records = [], []
    for seed in SEEDS:
        source = source_for(seed)
        subject = pd.read_csv(source / "seed0_subject_metrics.csv")
        train = json.loads((source / "seed0_training_records.json").read_text(encoding="utf-8"))
        if set(subject.seed.astype(int)) != {seed} or len(subject) != 48:
            raise RuntimeError(f"subject metric coverage failure seed={seed}")
        if len(train["records"]) != 10 or any(int(r["seed"]) != seed for r in train["records"]):
            raise RuntimeError(f"training record coverage failure seed={seed}")
        subject = subject.copy(); subject["source"] = "FROZEN_REUSED" if seed == 0 else "NEW_FINAL_SEMANTICS"
        metrics.append(subject); records.extend(train["records"])
        shutil.copy2(source / "seed0_subject_metrics.csv", OUT / f"seed{seed}_subject_metrics.csv")
        shutil.copy2(source / "seed0_training_records.json", OUT / f"seed{seed}_training_records.json")
    all_rows = pd.concat(metrics, ignore_index=True)
    write_csv(OUT / "multiseed_subject_metrics.csv", all_rows)
    summary = []
    values = {}
    for model in ("EEGNet", "SIRE-EEG"):
        per_seed = []
        for seed in SEEDS:
            one = all_rows[(all_rows.model == model) & (all_rows.seed == seed)]
            table = one.pivot(index="subject", columns="session", values=["BA", "Macro_F1"])
            future = table[("BA", "S2")]; macro = table[("Macro_F1", "S2")]; ws = table["BA"].min(axis=1)
            values[(model, seed)] = {"future_S2_BA": future, "future_S2_Macro_F1": macro, "WS_BA": ws}
            per_seed.append({"row_type": "seed", "seed": seed, "model": model, "future_S2_BA": float(future.mean()), "future_S2_Macro_F1": float(macro.mean()), "WS_BA": float(ws.mean()), "subjects": len(future)})
        summary.extend(per_seed)
        for metric in ("future_S2_BA", "future_S2_Macro_F1", "WS_BA"):
            by_subject = np.stack([values[(model, seed)][metric].sort_index().to_numpy(float) for seed in SEEDS]).mean(axis=0)
            summary.append({"row_type": "three_seed_subject_mean", "seed": "0,1,2", "model": model, "metric": metric, "mean": float(by_subject.mean()), "subjects": len(by_subject)})
    for metric in ("future_S2_BA", "WS_BA"):
        delta = np.stack([values[("SIRE-EEG", seed)][metric].sort_index().to_numpy(float) - values[("EEGNet", seed)][metric].sort_index().to_numpy(float) for seed in SEEDS]).mean(axis=0)
        rng = np.random.default_rng(int.from_bytes(f"BNCI-three-seed-{metric}".encode()[:8].ljust(8, b"0"), "little"))
        draws = delta[rng.integers(0, len(delta), size=(20000, len(delta)))].mean(1)
        summary.append({"row_type": "paired_three_seed_subject_mean", "seed": "0,1,2", "model": "SIRE-EEG minus EEGNet", "metric": metric, "mean": float(delta.mean()), "CI95_low": float(np.quantile(draws,.025)), "CI95_high": float(np.quantile(draws,.975)), "subjects": len(delta), "bootstrap_draws": 20000})
    write_csv(OUT / "multiseed_summary.csv", pd.DataFrame(summary))
    write_json(OUT / "multiseed_training_records.json", {"seeds": list(SEEDS), "seed0": "frozen reused", "seed1_2": "newly trained", "records": records,
        "final_seed_semantics": {"same_fixed_outer_folds_across_seeds": True, "same_authoritative_episode_manifests_across_seeds": True, "initialization_seed": "seed", "training_rng_seed": "100000 + seed"}})
    lines = ["# BNCI2015-001 Full-model matched protocol: three seeds", "", "Seed 0 is frozen reuse. Seeds 1/2 were trained with the authoritative final convention: fixed seed-0 folds and manifests, initialization RNG = seed, training RNG = 100000 + seed.", "",
             "| Seed | Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |", "|---:|---|---:|---:|---:|"]
    for row in summary:
        if row["row_type"] == "seed": lines.append(f"| {row['seed']} | {row['model']} | {row['future_S2_BA']:.4f} | {row['future_S2_Macro_F1']:.4f} | {row['WS_BA']:.4f} |")
    lines += ["", "Three-seed SIRE−EEGNet paired effects are in `multiseed_summary.csv`; the bootstrap unit is the biological subject after within-subject seed averaging."]
    (OUT / "MULTISEED_PRIMARY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run: raise RuntimeError("pass --run")
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in (1, 2):
        target = OUT / "runtime" / f"seed{seed}" / "seed0_training_records.json"
        if not target.is_file(): run_missing(seed)
    aggregate()
    print("BNCI_THREE_SEED_PRIMARY_COMPLETE", flush=True)


if __name__ == "__main__": main()
