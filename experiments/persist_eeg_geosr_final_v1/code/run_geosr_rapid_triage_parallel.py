"""Run the two fold-0 RAPID_TRIAGE workers sequentially and merge locks.

The orchestrator never imports outcome data.  It only waits for the two
worker completion markers, verifies their hashes/scopes, and writes a compact
pre-outcome lock for the subsequent outcome-only evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_geosr as g


EXP = Path(__file__).resolve().parents[1]
DATASETS = ("OpenBMI", "WBCIC")
FOLD = 0
METHODS = ("SUBJECT_BALANCED_ERM", "GEOSR")
MEMORY_LIMIT_MIB = 31_000


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def worker_root(root: Path, dataset: str) -> Path:
    return root / "workers" / f"{dataset}_fold0"


def worker_valid(root: Path, dataset: str, amendment_sha: str) -> bool:
    wr = worker_root(root, dataset)
    marker = wr / "RAPID_TRIAGE_WORKER_COMPLETE.json"
    lock_path = wr / "PRE_OUTCOME_RAPID_TRIAGE_WORKER_LOCK.json"
    manifest = wr / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json"
    if not marker.is_file() or not lock_path.is_file() or not manifest.is_file():
        return False
    try:
        m = json.loads(marker.read_text(encoding="utf-8")); lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if m.get("dataset") != dataset or m.get("fold") != FOLD or m.get("seed") != 0 or m.get("methods") != list(METHODS):
            return False
        if m.get("amendment_sha256") != amendment_sha or lock.get("amendment_sha256") != amendment_sha:
            return False
        if lock.get("outcome_labels_read") is not False or lock.get("exact_refit") is not False:
            return False
        info = json.loads(manifest.read_text(encoding="utf-8"))[f"{dataset}/fold-0/seed-0"]
        for method in METHODS:
            p = Path(info["checkpoints"][method]["path"])
            if not p.is_file() or not g.checkpoint_meta_path(p).is_file():
                return False
        return True
    except Exception:
        return False


def gpu_sample() -> tuple[float, float]:
    try:
        text = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
        util, mem = text.strip().split(",", 1)
        return float(util), float(mem)
    except Exception:
        return float("nan"), 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="full decision_seed0/workers directory containing initial-selection caches")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve(); source_root = args.source_root.resolve()
    amendment_path = EXP / "RAPID_TRIAGE_PROTOCOL_AMENDMENT.json"
    amendment_lock_path = EXP / "RAPID_TRIAGE_LOCK.json"
    if not amendment_path.is_file() or not amendment_lock_path.is_file():
        raise RuntimeError("RAPID_TRIAGE amendment/hash lock missing")
    amendment_sha = file_sha(amendment_path)
    amendment_lock = json.loads(amendment_lock_path.read_text(encoding="utf-8-sig"))
    if amendment_lock.get("amendment_sha256") != amendment_sha or amendment_lock.get("outcome_labels_read") is not False:
        raise RuntimeError("RAPID_TRIAGE lock is not valid pre-outcome")

    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PERSIST_TORCH_THREADS"] = env.get("PERSIST_TORCH_THREADS", "8")
    env["PERSIST_CUDNN_BENCHMARK"] = "0"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "8")
    py = sys.executable
    metrics: dict[str, Any] = {"execution_mode": "sequential_single_gpu", "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu_samples": [], "workers": {}}
    t0 = time.perf_counter()
    for dataset in DATASETS:
        wr = worker_root(root, dataset); wr.mkdir(parents=True, exist_ok=True)
        if worker_valid(root, dataset, amendment_sha):
            print(f"[rapid-parent] cache hit {dataset} fold=0", flush=True)
            continue
        out = (wr / "worker.log").open("a", encoding="utf-8")
        err = (wr / "worker.err").open("a", encoding="utf-8")
        cmd = [py, "-u", str(Path(__file__).with_name("run_geosr_rapid_triage.py")),
               "--dataset", dataset, "--fold", "0", "--source-root", str(source_root / f"{dataset}_fold0"),
               "--root", str(wr), "--device", args.device]
        p = subprocess.Popen(cmd, cwd=str(EXP), env=env, stdout=out, stderr=err, text=True)
        started = time.perf_counter()
        print(f"[rapid-parent] launched {dataset} pid={p.pid}", flush=True)
        try:
            while p.poll() is None:
                time.sleep(5.0)
                util, mem = gpu_sample()
                metrics["gpu_samples"].append({"t_sec": time.perf_counter() - t0, "dataset": dataset, "util_pct": util, "vram_mib": mem})
                write_json(root / "RAPID_TRIAGE_RUN_METRICS.json", metrics)
                if mem > MEMORY_LIMIT_MIB:
                    p.terminate()
                    p.wait()
                    raise RuntimeError(f"GPU memory guard exceeded: {mem:.0f} MiB")
        finally:
            out.close(); err.close()
        metrics["workers"][dataset] = {"returncode": p.returncode, "wall_sec": time.perf_counter() - started}
        print(f"[rapid-parent] finished {dataset} rc={p.returncode}", flush=True)
        if p.returncode != 0 or not worker_valid(root, dataset, amendment_sha):
            raise RuntimeError(f"rapid worker failed validation: {dataset}")

    # Merge only compact pre-outcome artifacts; no outcome loader is imported.
    merged = root / "results"; merged.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    for name in ("CROSS_FIT_ASSIGNMENTS.csv", "CROSSFIT_TEACHER_AUDIT.csv", "SOURCE_GEOMETRY_RISK.csv", "SOURCE_WEIGHT_AUDIT.csv", "TRAINING_SUMMARY.csv"):
        frames = [pd.read_csv(worker_root(root, d) / "results" / name) for d in DATASETS]
        pd.concat(frames, ignore_index=True).to_csv(merged / name, index=False)
    manifests: dict[str, Any] = {}; lock_hashes: dict[str, str] = {}; role_hashes: dict[str, str] = {}
    for dataset in DATASETS:
        wr = worker_root(root, dataset)
        manifests.update(json.loads((wr / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json").read_text(encoding="utf-8")))
        lock_path = wr / "PRE_OUTCOME_RAPID_TRIAGE_WORKER_LOCK.json"
        lock_hashes[dataset] = file_sha(lock_path)
        role_hashes[dataset] = json.loads(lock_path.read_text(encoding="utf-8"))["role_hash"]
    write_json(root / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json", manifests)
    pre_lock = {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_PRE_OUTCOME_LOCK_V1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "amendment_sha256": amendment_sha, "seed": 0, "datasets": list(DATASETS), "folds": [FOLD],
        "methods": list(METHODS), "backbone": "EEGNet", "inner_crossfit_k": 5, "descriptor_cap": 32,
        "worker_lock_sha256": lock_hashes, "role_hashes": role_hashes, "manifest_sha256": file_sha(root / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json"),
        "initial_selection_weights_reused": True, "wbcic_final_refit_teacher": False, "exact_refit": False,
        "student_discovery_selected": True, "outcome_labels_read": False, "outcome_labels_read_before_lock": False,
        "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False,
        "scientific_definition_changed": True, "final_claim_authorized": False,
    }
    write_json(root / "RAPID_TRIAGE_PRE_OUTCOME_LOCK.json", pre_lock)
    write_json(root / "DATA_LEGALITY_AUDIT.json", {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_LEGALITY_V1", "seed": 0, "datasets": list(DATASETS), "folds": [FOLD], "methods": list(METHODS),
        "amendment_sha256": amendment_sha, "outcome_labels_read_before_lock": False, "canonical_outcome_labels_read": False,
        "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False, "lock_sha256": file_sha(root / "RAPID_TRIAGE_PRE_OUTCOME_LOCK.json"),
    })
    metrics["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); metrics["wall_sec"] = time.perf_counter() - t0
    finite = [x["util_pct"] for x in metrics["gpu_samples"] if np.isfinite(x["util_pct"])]
    metrics["gpu_util_mean_pct"] = float(np.mean(finite)) if finite else None
    metrics["gpu_vram_peak_mib"] = float(max((x["vram_mib"] for x in metrics["gpu_samples"]), default=0.0))
    write_json(root / "RAPID_TRIAGE_RUN_METRICS.json", metrics)
    print("RAPID_TRIAGE_PRE_OUTCOME_LOCKED", flush=True)


if __name__ == "__main__":
    main()
