from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import default_cache_root
from evaluate_holdout import evaluate_holdout_once
from explore import run_exploration
from freeze import freeze_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("explore", "freeze", "holdout"))
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    args = parser.parse_args()
    if args.phase == "explore":
        result = run_exploration(args.cache_root)
    elif args.phase == "freeze":
        result = freeze_candidates()
    else:
        result = evaluate_holdout_once(args.cache_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

