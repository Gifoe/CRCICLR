#!/usr/bin/env python
from __future__ import annotations

import argparse

from hsc_tta.v3.simulation import run_simulations

from _common import load_yaml, project_root


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default="."); parser.add_argument("--config",required=True)
    args=parser.parse_args(); root=project_root(args.root); result=run_simulations(load_yaml(args.config))
    out=root/"outputs/v3_probecert/simulations"; out.mkdir(parents=True,exist_ok=True)
    result.to_parquet(out/"SIMULATION_RESULTS.parquet",index=False); result.to_csv(out/"SIMULATION_SUMMARY.csv",index=False)
    print({"rows":len(result),"minimum_exchangeable_validity_gap":result[result.scenario=='exchangeable_grid'].validity_gap.min()})


if __name__=="__main__": main()
