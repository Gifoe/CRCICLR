#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from hsc_tta.v3.baselines import evaluate_baselines
from _common import load_yaml,project_root
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--config",required=True);p.add_argument("--device",default="cuda");p.add_argument("--resume",action="store_true");a=p.parse_args();r=project_root(a.root);c=load_yaml(a.config)
 files=sorted((r/"outputs/v3_probecert/nested_dev/surfaces").glob("NESTED_ACTION_SURFACES_*.parquet")); surfaces=pd.concat([pd.read_parquet(x) for x in files],ignore_index=True)
 results=[];settings=[]
 for alpha in c["alphas"]:
  x,y=evaluate_baselines(surfaces,float(alpha),float(c["epsilon"]),float(c["delta"]));results.append(x);settings.append(y)
 out=r/"outputs/v3_probecert/baselines";out.mkdir(parents=True,exist_ok=True);result=pd.concat(results)
 core=pd.read_parquet(r/"outputs/v3_probecert/nested_dev/counterfactuals/OUTER_COUNTERFACTUALS.parquet")
 for column in result.columns:
  if column not in core: core[column]=pd.NA
 result=pd.concat([result,core[result.columns]],ignore_index=True,sort=False)
 result.to_parquet(out/"BASELINE_SUBJECT_RESULTS.parquet",index=False);pd.concat(settings).to_csv(out/"BASELINE_SETTINGS.csv",index=False)
 summary=result.groupby(["dataset","seed","alpha","policy"],as_index=False).agg(joint_violation_rate=("joint_violation","mean"),average_set_size=("average_set_size","mean"),intervention_rate=("intervention","mean"),sentinel_rate=("sentinel","mean"));summary.to_csv(out/"BASELINE_SUMMARY.csv",index=False);print({"rows":len(result)})
if __name__=="__main__":main()
