from __future__ import annotations

import argparse
import pathlib
import sys


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="V7 full gated pipeline")
    value.add_argument("--repo-root", required=True)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--stage")
    value.add_argument("--dataset")
    value.add_argument("--model")
    value.add_argument("--fold", type=int)
    value.add_argument("--seed", type=int)
    value.add_argument("--max-workers", type=int, default=1)
    for name in ["compatibility-only", "experts-only", "oracle-only", "routing-only", "method-only", "external-only"]:
        value.add_argument(f"--{name}", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    repo = pathlib.Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo / "src"))
    from hsc_tta.fm_routing_full import FullPipeline
    return FullPipeline(repo).run(resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
