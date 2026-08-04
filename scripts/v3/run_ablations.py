#!/usr/bin/env python
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from hsc_tta.v3.ablations import evaluate_ablation
from _common import load_yaml,project_root

_SURFACES=None
_POLICY_GRID=None

def _initialize_worker(surfaces,policy_grid):
 global _SURFACES,_POLICY_GRID
 _SURFACES=surfaces;_POLICY_GRID=policy_grid

def _evaluate_task(task):
 alpha,epsilon,delta,variant,calibration_limit,actions=task
 return evaluate_ablation(_SURFACES,_POLICY_GRID,alpha,epsilon,delta,variant,calibration_limit,actions)

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--config",required=True);p.add_argument("--policy-config",default="configs/v3/probe_policy.yaml");p.add_argument("--device",default="cuda");p.add_argument("--resume",action="store_true");p.add_argument("--workers",type=int,default=8);a=p.parse_args();r=project_root(a.root);c=load_yaml(a.config);g=load_yaml(a.policy_config)
 surfaces=pd.concat([pd.read_parquet(x) for x in sorted((r/"outputs/v3_probecert/nested_dev/surfaces").glob("NESTED_ACTION_SURFACES_*.parquet"))],ignore_index=True);tasks=[]
 variants=["probecert_v3","g_set_only","without_augmentation_consistency","without_temporal_stability","without_source_drift","without_collapse_gate","without_update_gate","risk_only_without_noninferiority","no_pseudo_episodes"]
 for alpha in c["alphas"]:
  for variant in variants: tasks.append((float(alpha),float(c["epsilon"]),float(c["delta"]),variant,None,None))
  for eps in c["epsilon_sensitivity"]: tasks.append((float(alpha),float(eps),float(c["delta"]),f"epsilon_{eps}",None,None))
  for size in [10,12]: tasks.append((float(alpha),float(c["epsilon"]),float(c["delta"]),f"calibration_size_{size}",size,None))
  action_libraries=[("number_actions_1_adapter",["robust_residual_adapter"]),("number_actions_1_t3a",["official_t3a"]),("number_actions_2",["official_t3a","robust_residual_adapter"])]
  for label,actions in action_libraries: tasks.append((float(alpha),float(c["epsilon"]),float(c["delta"]),label,None,actions))
 with ProcessPoolExecutor(max_workers=max(1,min(a.workers,len(tasks))),initializer=_initialize_worker,initargs=(surfaces,g)) as pool:
  results=list(pool.map(_evaluate_task,tasks))
 result=pd.concat(results,ignore_index=True)
 baseline_path=r/"outputs/v3_probecert/baselines/BASELINE_SUBJECT_RESULTS.parquet"
 if baseline_path.exists():
  actionwise=pd.read_parquet(baseline_path);actionwise=actionwise[actionwise.policy=="v2_actionwise_joint"].copy();actionwise["ablation"]="actionwise_simultaneous_certificate"
  for column in result.columns:
   if column not in actionwise:actionwise[column]=pd.NA
  result=pd.concat([result,actionwise[result.columns]],ignore_index=True)
 out=r/"outputs/v3_probecert/ablations";out.mkdir(parents=True,exist_ok=True);result.to_parquet(out/"ABLATION_SUBJECT_RESULTS.parquet",index=False)
 summary=result.groupby(["dataset","alpha","ablation"],as_index=False).agg(joint_violation_rate=("joint_violation","mean"),average_set_size=("average_set_size","mean"),intervention_rate=("intervention","mean"),sentinel_rate=("sentinel","mean"));
 missing=pd.DataFrame([{"dataset":"all","alpha":alpha,"ablation":name,"status":"not_comparable_from_frozen_1:1_surfaces"} for alpha in c["alphas"] for name in ["no_adapt_probe_split","high_dimensional_v2_features","adapt_probe_ratio_sensitivity"]]);summary["status"]="measured";summary=pd.concat([summary,missing],ignore_index=True);summary.to_csv(out/"ABLATION_SUMMARY.csv",index=False);print({"rows":len(result),"summary":len(summary)})
 summary[summary.ablation.astype(str).str.startswith("calibration_size_")].to_csv(out/"CALIBRATION_SIZE_SENSITIVITY.csv",index=False)
if __name__=="__main__":main()
