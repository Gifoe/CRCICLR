#!/usr/bin/env python
from __future__ import annotations

import argparse

from hsc_tta.v3.nested_policy_evaluation import run_nested_policy_evaluation

from _common import load_yaml, project_root


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default="."); parser.add_argument("--config",required=True)
    parser.add_argument("--action-config",default="configs/v3/action_search.yaml"); parser.add_argument("--policy-config",default="configs/v3/probe_policy.yaml")
    parser.add_argument("--device",default="cuda"); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--datasets",nargs="+"); parser.add_argument("--seeds",nargs="+",type=int); parser.add_argument("--output-tag")
    args=parser.parse_args(); result=run_nested_policy_evaluation(project_root(args.root),load_yaml(args.config),load_yaml(args.action_config),
        load_yaml(args.policy_config),args.device,args.resume,args.datasets,args.seeds,args.output_tag); print({key:len(value) for key,value in result.items()})


if __name__=="__main__": main()
