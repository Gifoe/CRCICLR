"""Server-only bounded scheduler for independent Phase-B fold/seed jobs."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RUNTIME = EXP / "runtime"
SCRIPT = HERE / "matched_aux.py"


def complete(fold: int, seed: int) -> bool:
    path = RUNTIME / "matched_runs" / f"fold-{fold}" / f"seed-{seed}" / "RUN_COMPLETE.json"
    if not path.is_file():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("pass") is True


def detect_existing_failure(fold: int, seed: int) -> None:
    candidates = [
        RUNTIME / f"matched_fold{fold}_seed{seed}.log",
        RUNTIME / f"matched_fold{fold}_seed{seed}.err.log",
    ]
    for path in candidates:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if "Traceback (most recent call last)" in text:
                raise RuntimeError(f"existing pilot failed: fold={fold} seed={seed}; see {path}")


def wait_for_pilot() -> None:
    pilot = [(0, 0), (0, 1), (0, 2)]
    started = time.time()
    while not all(complete(fold, seed) for fold, seed in pilot):
        for fold, seed in pilot:
            if not complete(fold, seed):
                detect_existing_failure(fold, seed)
        if time.time() - started > 24 * 3600:
            raise TimeoutError("fold-0 pilot did not complete within 24 hours")
        time.sleep(30)
    print("[scheduler] fold-0 pilot complete; launching remaining jobs", flush=True)


def main() -> None:
    wait_for_pilot()
    pending = [(fold, seed) for fold in range(1, 5) for seed in range(3) if not complete(fold, seed)]
    active: dict[tuple[int, int], tuple[subprocess.Popen[bytes], object]] = {}
    while pending or active:
        while pending and len(active) < 3:
            fold, seed = pending.pop(0)
            log_path = RUNTIME / f"matched_fold{fold}_seed{seed}.log"
            handle = log_path.open("ab")
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--fold", str(fold), "--seed", str(seed)],
                cwd=REPO,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[(fold, seed)] = (process, handle)
            print(f"[scheduler] started fold={fold} seed={seed} pid={process.pid}", flush=True)
        time.sleep(10)
        finished = []
        for key, (process, handle) in active.items():
            code = process.poll()
            if code is None:
                continue
            handle.close()
            fold, seed = key
            if code != 0 or not complete(fold, seed):
                raise RuntimeError(f"matched job failed fold={fold} seed={seed} exit={code}")
            print(f"[scheduler] completed fold={fold} seed={seed}", flush=True)
            finished.append(key)
        for key in finished:
            del active[key]
    print("[scheduler] ALL 15 FOLD/SEED RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
