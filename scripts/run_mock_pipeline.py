#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
from _common import parser
from hsc_tta.simulation import run_simulations
from hsc_tta.schemas import write_mock_gpu_interface
from hsc_tta.utils import load_yaml, require_cpu


def main() -> int:
    args = parser("Run CPU synthetic HSC-TTA simulations").parse_args(); require_cpu(args.device)
    cfg = load_yaml(args.config)
    out = Path("/root/autodl-tmp/hsc_tta_eeg/outputs/cpu_simulation")
    if args.dry_run: print(out); return 0
    results = run_simulations(out, seed=args.seed, n_subjects=min(args.limit_subjects or cfg.get("simulation_subjects",120), cfg.get("simulation_subjects",120)))
    interface = write_mock_gpu_interface(
        Path("/root/autodl-tmp/hsc_tta_eeg/mock_features"),
        seed=args.seed,
        n_subjects=min(args.limit_subjects or cfg.get("simulation_subjects",120), cfg.get("simulation_subjects",120)),
    )
    print("mock_interface=" + ",".join(str(path) for path in interface.values()))
    print(results["simulation_summary"].to_string(index=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
