"""Sequential, resumable launcher for the 20 frozen actionability audit cells."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
RUN = EXP / "code" / "run.py"
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        raise SystemExit("This GPU memory bound is locked to one worker; use --workers 1.")
    for task in TASKS:
        for fold in range(5):
            cmd = [sys.executable, str(RUN), "evaluate", "--task", task, "--fold", str(fold)]
            print("RUN_CELL", task, fold, flush=True)
            subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
