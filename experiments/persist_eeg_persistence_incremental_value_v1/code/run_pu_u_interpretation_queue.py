"""Low-CPU resumable queue for frozen PU/U explanatory cells."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

from run_pu_u_interpretation import MODELS, TASKS, path_for


CODE = Path(__file__).resolve().parent
# ``CODE`` is the code directory itself, whereas run_cell derives ROOT from
# a file path.  Therefore its repository parent is index 2, not index 3.
ROOT = CODE.parents[2]
RUNTIME = Path(os.environ.get("PERSIST_ANALYSIS_ROOT", str(ROOT.parent / "persist_incremental_value_runtime/analysis"))).resolve()


def complete_original_model(model: str) -> bool:
    for task in TASKS:
        for fold in range(5):
            for seed in range(3):
                path = RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed{seed}.json"
                if not path.is_file():
                    return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = [m for m in args.models if complete_original_model(m)]
    excluded = [m for m in args.models if m not in selected]
    print("INTERPRETATION_MODELS", ",".join(selected), "excluded_incomplete=", ",".join(excluded), flush=True)
    jobs = [(model, task, fold, seed) for model in selected for task in TASKS for fold in range(5) for seed in range(3)
            if not path_for(model, task, fold, seed).is_file()]
    # ERP's full heldout session matrix is much larger than the other tasks.
    # Parallel ERP cells duplicate several GB of normalized arrays and can hit
    # the host memory ceiling.  Serialization changes only process timing, not
    # cached data, seeds, model inference, or any numerical result.
    parallel_jobs = [job for job in jobs if job[1] != "OpenBMI_ERP"]
    erp_jobs = [job for job in jobs if job[1] == "OpenBMI_ERP"]
    print("INTERPRETATION_JOBS", len(jobs), "parallel_non_erp=", len(parallel_jobs),
          "erp_serial=", len(erp_jobs), "workers=", args.workers, flush=True)
    log_root = RUNTIME / "pu_u_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def execute(job):
        model, task, fold, seed = job
        name = f"{model}_{task}_fold{fold}_seed{seed}.log"
        log_path = log_root / name
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", str(CODE / "run_pu_u_interpretation.py"),
                 "cell", model, task, str(fold), str(seed)],
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            raise RuntimeError(f"interpretation cell failed: {model} {task} {fold} {seed}; log={log_path}")
    if parallel_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(execute, parallel_jobs))
    if erp_jobs:
        # Do not overlap memory-heavy ERP normalization with another ERP cell.
        for job in erp_jobs:
            execute(job)
    print("INTERPRETATION_QUEUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
