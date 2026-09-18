"""Low-CPU resumable queue for frozen PU/U explanatory cells."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_pu_u_interpretation import MODELS, TASKS, path_for


CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[3]
RUNTIME = ROOT.parent / "persist_incremental_value_runtime/analysis"


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
    args = parser.parse_args()
    selected = [m for m in args.models if complete_original_model(m)]
    excluded = [m for m in args.models if m not in selected]
    print("INTERPRETATION_MODELS", ",".join(selected), "excluded_incomplete=", ",".join(excluded), flush=True)
    for model in selected:
        for task in TASKS:
            for fold in range(5):
                for seed in range(3):
                    if path_for(model, task, fold, seed).is_file():
                        continue
                    result = subprocess.run([sys.executable, "-u", str(CODE / "run_pu_u_interpretation.py"), "cell", model, task, str(fold), str(seed)], check=False)
                    if result.returncode:
                        raise RuntimeError(f"interpretation cell failed: {model} {task} {fold} {seed}")
    print("INTERPRETATION_QUEUE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
