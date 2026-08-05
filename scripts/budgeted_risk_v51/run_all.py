from __future__ import annotations

import argparse

from hsc_tta.budgeted_risk.diagnostics.pipeline import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the closed CPU-only V5.1 diagnostic")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", choices=["hmc", "eegmmidb"])
    parser.add_argument("--budget", type=int, choices=[5, 10, 20, 50])
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--scheme")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--output-tag", default="v51")
    args = parser.parse_args()
    if any(v is not None for v in (args.dataset, args.budget, args.seed, args.fold, args.scheme)):
        parser.error("partial selectors are reserved for resumable shard diagnostics; the frozen final run must be complete")
    print(run_all(args.project_root, resume=args.resume))


if __name__ == "__main__":
    main()

