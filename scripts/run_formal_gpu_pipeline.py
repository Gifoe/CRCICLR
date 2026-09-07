#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hsc_tta.gpu.experiment import evaluate_history_seed
from hsc_tta.gpu.formal import (evaluate_final_outcomes, freeze_decisions, freeze_methods,
                                make_final_decisions, train_predictors_and_calibrate)
from hsc_tta.gpu.reporting import generate_reports


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("history", "predictors", "freeze", "decisions", "decision-freeze", "outcomes", "reports"))
    parser.add_argument("--datasets", default="hmc,cap,eegmmidb")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    datasets, seeds = args.datasets.split(","), list(map(int, args.seeds.split(",")))
    if args.stage == "history":
        # HMC configurations must exist before CAP inheritance.
        order = [x for x in ("hmc", "eegmmidb", "cap") if x in datasets]
        for dataset in order:
            for seed in seeds:
                features, diagnostics, outcomes = evaluate_history_seed(ROOT, dataset, seed, args.device, args.resume)
                print(dataset, seed, len(features), len(diagnostics), len(outcomes), flush=True)
    elif args.stage == "predictors":
        order = [x for x in ("hmc", "eegmmidb", "cap") if x in datasets]
        for dataset in order:
            for seed in seeds:
                predictions, quantiles = train_predictors_and_calibrate(ROOT, dataset, seed, args.device, args.resume)
                print(dataset, seed, len(predictions), quantiles, flush=True)
    elif args.stage == "freeze":
        print(freeze_methods(ROOT))
    elif args.stage == "decisions":
        for dataset in datasets:
            for seed in seeds:
                candidates, decisions = make_final_decisions(ROOT, dataset, seed, args.device)
                print(dataset, seed, len(candidates), len(decisions), flush=True)
    elif args.stage == "decision-freeze":
        freeze_decisions(ROOT)
        print("decision freeze verified")
    elif args.stage == "outcomes":
        for dataset in datasets:
            for seed in seeds:
                counter, selected, joined = evaluate_final_outcomes(ROOT, dataset, seed, args.device)
                print(dataset, seed, len(counter), len(selected), len(joined), flush=True)
    elif args.stage == "reports":
        by_seed, with_ci = generate_reports(ROOT)
        print(len(by_seed), len(with_ci))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
