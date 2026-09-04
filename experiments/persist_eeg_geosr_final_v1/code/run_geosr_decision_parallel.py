"""Run the seed-0 two-fold decision screen with two bounded GPU workers.

OpenBMI and WBCIC workers are independent deterministic processes.  Pairing one
fold from each dataset keeps the measured peak below the RTX 5090 memory limit;
workers are never launched four-at-a-time.  Their compact artifacts are merged
only after both preflight locks exist, then outcome evaluation runs once in the
parent process after the merged lock.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_geosr_decision as d


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "decision_seed0"
WORKERS = ROOT / "workers"
DATASETS = ("OpenBMI", "WBCIC")
FOLDS = (0, 1)
METHODS = d.METHODS
MEMORY_LIMIT_MIB = 31_000


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(d.jclean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def gpu_sample() -> tuple[float, float]:
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        util, mem = line.split(",", 1)
        return float(util), float(mem)
    except Exception:
        return float("nan"), 0.0


def worker_root(dataset: str, fold: int) -> Path:
    return WORKERS / f"{dataset}_fold{fold}"


def worker_complete(root: Path) -> bool:
    marker = root / "WORKER_PREFLIGHT_COMPLETE.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return payload.get("code_fingerprint") == d.g.code_fingerprint() and payload.get("seed") == 0
    except Exception:
        return False


def launch_pair(pair: list[tuple[str, int]], metrics: dict[str, Any]) -> None:
    processes: list[tuple[tuple[str, int], subprocess.Popen[str], Any, Any]] = []
    env = os.environ.copy()
    env.setdefault("PERSIST_TORCH_THREADS", "8")
    # This flag was verified bitwise-equivalent on the target CUDA stack; keep
    # it opt-in so a different host falls back to the canonical deterministic
    # cuDNN setting.
    env.setdefault("PERSIST_CUDNN_BENCHMARK", "0")
    for dataset, fold in pair:
        root = worker_root(dataset, fold)
        root.mkdir(parents=True, exist_ok=True)
        out = (root / "worker.log").open("a", encoding="utf-8")
        err = (root / "worker.err").open("a", encoding="utf-8")
        cmd = [sys.executable, str(Path(__file__).with_name("run_geosr_decision.py")),
               "--phase", "preflight", "--dataset", dataset, "--fold", str(fold),
               "--root", str(root), "--device", "cuda"]
        if worker_complete(root):
            out.write("[orchestrator] worker cache marker valid; skipped launch\n")
            out.close(); err.close()
            continue
        p = subprocess.Popen(cmd, cwd=str(EXP), env=env, stdout=out, stderr=err, text=True)
        processes.append(((dataset, fold), p, out, err))
        print(f"[orchestrator] launched {dataset} fold={fold} pid={p.pid}", flush=True)
    if not processes:
        return
    t0 = time.perf_counter()
    samples: list[tuple[float, float]] = []
    while processes:
        time.sleep(5.0)
        util, mem = gpu_sample(); samples.append((util, mem))
        metrics["gpu_samples"].append({"t_sec": time.perf_counter() - t0, "util_pct": util, "vram_mib": mem})
        if mem > MEMORY_LIMIT_MIB:
            for _, p, out, err in processes:
                if p.poll() is None:
                    p.terminate()
                out.close(); err.close()
            raise RuntimeError(f"GPU memory guard exceeded: {mem:.0f} MiB")
        still: list[tuple[tuple[str, int], subprocess.Popen[str], Any, Any]] = []
        for item in processes:
            if item[1].poll() is None:
                still.append(item)
            else:
                item[2].close(); item[3].close()
                print(f"[orchestrator] finished {item[0][0]} fold={item[0][1]} rc={item[1].returncode}", flush=True)
        processes = still
    metrics["pair_wall_sec"].append(time.perf_counter() - t0)
    for dataset, fold in pair:
        root = worker_root(dataset, fold)
        if not worker_complete(root):
            raise RuntimeError(f"worker did not produce a valid completion marker: {dataset} fold {fold}")


def merge_preflight() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    ROOT_RESULTS = ROOT / "results"
    ROOT_RUNTIME = ROOT / "runtime"
    ROOT_RESULTS.mkdir(parents=True, exist_ok=True)
    ROOT_RUNTIME.mkdir(parents=True, exist_ok=True)
    csv_names = ["CROSS_FIT_ASSIGNMENTS.csv", "CROSSFIT_TEACHER_AUDIT.csv", "SOURCE_GEOMETRY_RISK.csv",
                 "SOURCE_WEIGHT_AUDIT.csv", "TRAINING_SUMMARY.csv"]
    for name in csv_names:
        parts = []
        for dataset in DATASETS:
            for fold in FOLDS:
                p = worker_root(dataset, fold) / "results" / name
                if p.is_file():
                    parts.append(pd.read_csv(p))
        if not parts:
            raise RuntimeError(f"no worker artifact for {name}")
        d.write_csv(ROOT_RESULTS / name, pd.concat(parts, ignore_index=True))
    state_hashes: dict[str, Any] = {}
    fold_manifests: dict[str, Any] = {}
    role_hashes: dict[str, list[str]] = {}
    support_lock: dict[str, Any] | None = None
    first_lock: dict[str, Any] | None = None
    for dataset in DATASETS:
        for fold in FOLDS:
            wr = worker_root(dataset, fold)
            worker_runtime = wr / "runtime" / "seed-0"
            worker_manifest = json.loads((worker_runtime / "PREFLIGHT_MANIFEST.json").read_text(encoding="utf-8"))
            fold_manifests.update(worker_manifest)
            state_path = wr / "results" / "INITIAL_STATE_HASHES.json"
            if state_path.is_file():
                state_hashes.update(json.loads(state_path.read_text(encoding="utf-8")))
            lock = json.loads((wr / "PRE_OUTCOME_GEOSR_DECISION_LOCK.json").read_text(encoding="utf-8"))
            if first_lock is None:
                first_lock = lock
            for ds, values in lock.get("role_hashes", {}).items():
                role_hashes.setdefault(ds, [None] * 5)
                # Worker locks contain all role hashes for their dataset; keep
                # the canonical fold-indexed list.  In single-fold worker mode
                # the one-element list corresponds to this worker's `fold`,
                # rather than index zero.
                if len(values) == 1 and len(FOLDS) == 1:
                    role_hashes[ds][int(fold)] = values[0]
                else:
                    for i, value in enumerate(values):
                        role_hashes[ds][i] = value
            if support_lock is None:
                p = wr / "DATA_SUPPORT_LOCK.json"
                support_lock = json.loads(p.read_text(encoding="utf-8"))
    if first_lock is None or support_lock is None:
        raise RuntimeError("worker locks missing")
    json_write(ROOT_RESULTS / "INITIAL_STATE_HASHES.json", state_hashes)
    json_write(ROOT / "DATA_SUPPORT_LOCK.json", {**support_lock, "decision_parallel_workers": 2})
    lock = dict(first_lock)
    lock.update({"datasets": list(DATASETS), "folds": list(FOLDS), "methods": list(METHODS),
                 "role_hashes": role_hashes, "fold_manifests": fold_manifests,
                 "decision_parallel_workers": 2, "outcome_labels_read": False,
                 "canonical_outcome_indices_materialized": False,
                 "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False})
    json_write(ROOT / "PRE_OUTCOME_GEOSR_DECISION_LOCK.json", lock)
    json_write(ROOT_RUNTIME / "seed-0" / "PREFLIGHT_MANIFEST.json", fold_manifests)
    json_write(ROOT / "DATA_LEGALITY_AUDIT.json", {
        "schema": "PERSIST_EEG_GEOSR_DECISION_DATA_LEGALITY_V1", "seed": 0,
        "decision_run": True, "folds": list(FOLDS), "methods": list(METHODS),
        "WBCIC_outer_10_opened": False, "OpenBMI_outer_test_opened": False,
        "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False,
        "outcome_labels_read_before_lock": False, "outcome_labels_read_after_lock": False,
        "lock_sha256": d.g.file_sha(ROOT / "PRE_OUTCOME_GEOSR_DECISION_LOCK.json"),
        "role_hashes": role_hashes, "descriptor_cap": d.g.CAP,
    })
    print("[orchestrator] merged preflight lock", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "outcome", "all"), default="all")
    args = parser.parse_args()
    if args.phase in ("preflight", "all"):
        d.ROOT = ROOT; d.RESULTS = ROOT / "results"; d.RUNTIME = ROOT / "runtime"
        d.DATASETS = DATASETS; d.FOLDS = FOLDS
        d.write_screen_protocol()
        metrics = {"started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "gpu_samples": [], "pair_wall_sec": [], "workers": 2}
        t0 = time.perf_counter()
        for fold in FOLDS:
            launch_pair([("OpenBMI", fold), ("WBCIC", fold)], metrics)
        merge_preflight()
        metrics["wall_sec"] = time.perf_counter() - t0
        metrics["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        finite = [x for x in metrics["gpu_samples"] if np.isfinite(x["util_pct"])]
        metrics["gpu_util_mean_pct"] = float(np.mean([x["util_pct"] for x in finite])) if finite else None
        metrics["gpu_vram_peak_mib"] = float(max((x["vram_mib"] for x in metrics["gpu_samples"]), default=0.0))
        json_write(ROOT / "PARALLEL_RUN_METRICS.json", metrics)
    if args.phase in ("outcome", "all"):
        d.ROOT = ROOT; d.RESULTS = ROOT / "results"; d.RUNTIME = ROOT / "runtime"
        d.DATASETS = DATASETS; d.FOLDS = FOLDS
        result = d.evaluate_outcome(torch.device("cuda"))
        print(result["terminal"], flush=True)


if __name__ == "__main__":
    main()
