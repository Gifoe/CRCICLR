#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage")
    parser.add_argument("--dataset", choices=("hmc", "eegmmidb"))
    parser.add_argument("--model", choices=("cbramod", "labram", "biot"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--embedding-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = pathlib.Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "src"))
    from hsc_tta.fm_routing.pipeline import Pipeline

    shortcuts = [args.audit_only, args.embedding_only, args.probe_only, args.oracle_only]
    if any(shortcuts) or any(value is not None for value in (args.stage, args.dataset, args.model, args.fold, args.seed)):
        state_path = repo / "outputs/fm_routing_v7/RUN_STATE.json"
        if not state_path.exists():
            raise SystemExit("partial execution cannot bypass uncompleted prerequisite gates")
        import json
        state = json.loads(state_path.read_text())
        if state.get("terminal"):
            pipeline = Pipeline(repo)
            pipeline.build_manifest()
            print(f"terminal state retained: {state.get('verdict')}")
            return 2 if state.get("verdict") == "V7_STAGE0A_TECHNICAL_BLOCK" else 0
        raise SystemExit("partial execution is disabled until its prerequisite state is complete")
    return Pipeline(repo).run()


if __name__ == "__main__":
    raise SystemExit(main())
