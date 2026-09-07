#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from _common import parser
from hsc_tta.simulation import run_simulations
from hsc_tta.utils import load_yaml, require_cpu


def main() -> int:
    args = parser("Run CPU critical-index HSC-TTA simulations A-G").parse_args()
    require_cpu(args.device)
    cfg = load_yaml(args.config)
    repetitions = int(cfg.get("simulation_repetitions", 500))
    if args.limit_subjects:
        repetitions = int(args.limit_subjects)
    output = Path("/root/autodl-tmp/hsc_tta_eeg/outputs/cpu_critical_index_simulation")
    results = run_simulations(
        output,
        seed=args.seed,
        repetitions=repetitions,
        n_nontrivial=int(cfg["n_nontrivial_lambdas"]),
        enforce_go=not bool(args.limit_subjects),
    )
    print(results["simulation_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
