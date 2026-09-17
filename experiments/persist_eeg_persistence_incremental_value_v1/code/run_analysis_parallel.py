"""Run independent frozen analysis cells with two bounded workers.

Each cell remains an unchanged run_cell.py subprocess and owns a unique output
path. The runner changes scheduling only, never the scientific computation.
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path


CODE = Path(__file__).resolve().parent
RUNTIME = CODE.parents[3] / "persist_incremental_value_runtime"
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "EEGConformer", "FBCNet")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
CELLS = [(m, t, f, s) for m in MODELS for t in TASKS for f in range(5) for s in range(3)]


def run_cell(cell: tuple[str, str, int, int]) -> tuple[int, float, str]:
    model, task, fold, seed = cell
    name = f"{model}_{task}_fold{fold}_seed{seed}"
    log_path = RUNTIME / "analysis" / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["INCREMENTAL_DIRECT_ANALYSIS"] = "1"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        for attempt in range(2):
            result = subprocess.run(
                [sys.executable, "-u", str(CODE / "run_cell.py"), model, task, str(fold), str(seed)],
                env=env, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            if result.returncode not in (-1073741819, 0xC0000005):
                break
            log.write(f"\nNATIVE_ACCESS_VIOLATION_RETRY attempt={attempt + 1}\n")
            log.flush()
    return result.returncode, time.monotonic() - start, name


def main() -> None:
    if os.environ.get("INCREMENTAL_DIRECT_ANALYSIS") != "1":
        raise RuntimeError("direct-analysis amendment not enabled")
    pending = iter(CELLS)
    done = 0
    failure = None
    workers = int(os.environ.get("INCREMENTAL_ANALYSIS_WORKERS", "3"))
    if workers < 1 or workers > 3:
        raise RuntimeError("worker count must be in [1, 3]")
    print(f"PARALLEL_WORKERS {workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        active = {pool.submit(run_cell, next(pending)) for _ in range(workers)}
        while active:
            completed = next(concurrent.futures.as_completed(active))
            active.remove(completed)
            code, seconds, name = completed.result()
            done += 1
            print(f"CELL_STATUS {done}/420 {name} exit={code} seconds={seconds:.1f}", flush=True)
            if code and failure is None:
                failure = (name, code)
            if failure is None:
                try:
                    active.add(pool.submit(run_cell, next(pending)))
                except StopIteration:
                    pass
    if failure is not None:
        raise RuntimeError(f"cell failure: {failure[0]} exit={failure[1]}")
    result = subprocess.run([sys.executable, "-u", str(CODE / "finalize.py")], check=False)
    if result.returncode:
        raise RuntimeError(f"finalization failed exit={result.returncode}")


if __name__ == "__main__":
    main()
