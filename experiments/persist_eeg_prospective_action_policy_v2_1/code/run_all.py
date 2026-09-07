from __future__ import annotations

import argparse
import json
import unittest
import warnings
from pathlib import Path

import pandas as pd

from evaluate import evaluate_v2_1
from reconstruct_v2 import EXPERIMENT_ROOT, reconstruct_v2, v2_common


def run_tests() -> None:
    suite = unittest.defaultTestLoader.discover(str(EXPERIMENT_ROOT / "code"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("V2.1 unit tests failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed PERSIST-EEG V2.1 falsification audit")
    parser.add_argument(
        "--phase",
        choices=("reconstruct", "test", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--cache-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root or v2_common.default_cache_root()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    if args.phase in ("reconstruct", "all"):
        reconstruction = reconstruct_v2(cache_root)
        print(json.dumps({"phase": "reconstruct", "status": reconstruction["status"]}))
    if args.phase in ("test", "all"):
        run_tests()
        print(json.dumps({"phase": "test", "status": "PASS"}))
    if args.phase in ("evaluate", "all"):
        decision = evaluate_v2_1(cache_root)
        print(
            json.dumps(
                {
                    "phase": "evaluate",
                    "status": "PASS",
                    "primary_state": decision["primary_state"],
                    "best_ensemble": decision["best_ensemble_selected_on_exploration"],
                    "OUTER_TEST_USED": decision["OUTER_TEST_USED"],
                }
            )
        )


if __name__ == "__main__":
    main()

