#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7-0B expert qualification repair")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--dataset", choices=["hmc", "eegmmidb"])
    parser.add_argument("--model", choices=["cbramod", "labram", "biot"])
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.repo_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from hsc_tta.fm_routing_repair import RepairPipeline

    pipeline = RepairPipeline(root)
    # Partial selectors are accepted only for resumability. Pipeline gates remain
    # authoritative, so none can bypass a failed predecessor phase.
    return pipeline.run()


if __name__ == "__main__":
    raise SystemExit(main())
