"""Run the frozen Phase-B aggregate after every matched run is complete."""
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
MATCHED = HERE / "matched_aux.py"


def run_complete(fold: int, seed: int) -> bool:
    marker = RUNTIME / "matched_runs" / f"fold-{fold}" / f"seed-{seed}" / "RUN_COMPLETE.json"
    if not marker.is_file():
        return False
    return json.loads(marker.read_text(encoding="utf-8")).get("pass") is True


def scheduler_failed() -> bool:
    error_log = RUNTIME / "scheduler.err.log"
    if not error_log.is_file():
        return False
    text = error_log.read_text(encoding="utf-8", errors="replace")
    return "Traceback (most recent call last)" in text


def main() -> None:
    started = time.time()
    expected = [(fold, seed) for fold in range(5) for seed in range(3)]
    while not all(run_complete(fold, seed) for fold, seed in expected):
        if scheduler_failed():
            raise RuntimeError("bounded scheduler failed; see runtime/scheduler.err.log")
        if time.time() - started > 36 * 3600:
            raise TimeoutError("matched runs did not complete within 36 hours")
        time.sleep(30)

    print("[finalizer] all 15 matched runs complete; starting frozen aggregate", flush=True)
    result = subprocess.run(
        [sys.executable, str(MATCHED), "--aggregate"],
        cwd=REPO,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"matched aggregate failed with exit code {result.returncode}")
    print("[finalizer] aggregate complete", flush=True)


if __name__ == "__main__":
    main()
