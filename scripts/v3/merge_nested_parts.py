#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import project_root


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default="."); args=parser.parse_args()
    root=project_root(args.root); base=root/"outputs/v3_probecert/nested_dev"; parts=base/"parts"
    mappings={"decisions/POLICY_DECISIONS.parquet":("decisions/POLICY_DECISIONS.parquet",["dataset","seed","outer_fold","alpha","subject_id","role"]),
              "calibration/CALIBRATION_JOINT_INDICES.parquet":("calibration/CALIBRATION_JOINT_INDICES.parquet",["dataset","seed","outer_fold","alpha","subject_id","policy"]),
              "counterfactuals/OUTER_COUNTERFACTUALS.parquet":("counterfactuals/OUTER_COUNTERFACTUALS.parquet",["dataset","seed","outer_fold","alpha","subject_id","policy"]),
              "NESTED_ACTION_CONFIG_RESULTS.parquet":("NESTED_ACTION_CONFIG_RESULTS.parquet",["dataset","seed","outer_fold","action","config_id","stage"])}
    for relative,(target,keys) in mappings.items():
        files=sorted(parts.glob(f"*/{relative}")); existing=base/target
        frames=([pd.read_parquet(existing)] if existing.exists() else [])+[pd.read_parquet(path) for path in files]
        if frames:
            frame=pd.concat(frames,ignore_index=True).drop_duplicates(subset=keys,keep="last")
            destination=base/target; destination.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(destination,index=False)
    fold_files=sorted(parts.glob("*/metrics/RESULTS_BY_FOLD.csv")); existing=base/"metrics/RESULTS_BY_FOLD.csv"
    frames=([pd.read_csv(existing)] if existing.exists() else [])+[pd.read_csv(path) for path in fold_files]
    if frames:
        frame=pd.concat(frames,ignore_index=True).drop_duplicates(subset=["dataset","seed","outer_fold","alpha","policy"],keep="last"); existing.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(existing,index=False)
        print({"folds":frame.groupby(["dataset","seed","outer_fold"]).ngroups,"rows":len(frame)})


if __name__=="__main__": main()
