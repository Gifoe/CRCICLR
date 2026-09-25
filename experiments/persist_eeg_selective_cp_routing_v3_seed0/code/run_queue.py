"""Bounded parallel launcher for independent UGCR-V3 task/fold cells."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

RUN = Path(__file__).with_name("run.py")
EXP = RUN.parents[1]
REPO = EXP.parents[1]
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FOLDS = tuple(range(5))
RUNTIME = Path(os.environ.get("UGCR_V3_RUNTIME", str(REPO.parent / "ugcr_v3_seed0_runtime")))


def one(stage: str, task: str, fold: int, threads: int):
    log_dir = RUNTIME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{stage}_{task.lower()}_fold{fold}.log"
    env = os.environ.copy()
    env["UGCR_V3_CPU_THREADS"] = str(threads)
    command = [sys.executable, str(RUN), stage, "--task", task, "--fold", str(fold)]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    tail = ""
    try:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:])
    except OSError:
        pass
    return task, fold, result.returncode, str(log), tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "final-eval"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if (args.task is None) != (args.fold is None):
        parser.error("task and fold must be provided together")
    cells = [(args.task, args.fold)] if args.task else [(t, f) for t in TASKS for f in FOLDS]
    cpus = os.cpu_count() or 1
    threads = max(1, min(8, cpus // args.workers))
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, args.stage, task, fold, threads) for task, fold in cells]
        for future in concurrent.futures.as_completed(futures):
            task, fold, code, log, tail = future.result()
            print(f"{args.stage.upper()} {'OK' if code == 0 else 'FAILED'} {task} fold{fold} rc={code} log={log}", flush=True)
            if tail:
                print(tail, flush=True)
            if code:
                failures.append((task, fold, code))
    if failures:
        raise SystemExit(f"failed cells: {failures}")
    print(f"QUEUE_COMPLETE {args.stage} cells={len(cells)} workers={args.workers} cpu_threads_per_worker={threads}", flush=True)


if __name__ == "__main__":
    main()
