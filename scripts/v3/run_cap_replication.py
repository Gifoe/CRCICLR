#!/usr/bin/env python
from __future__ import annotations

import argparse
from hsc_tta.v3.external_replication import merge_cap_parts,run_cap_replication
from _common import load_yaml,project_root

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--config",required=True);p.add_argument("--policy-config",default="configs/v3/probe_policy.yaml");p.add_argument("--device",default="cuda");p.add_argument("--seeds",nargs="+",type=int);p.add_argument("--output-tag");p.add_argument("--merge-parts",action="store_true")
    a=p.parse_args();root=project_root(a.root);main=load_yaml(a.config)
    if a.merge_parts:
        result=merge_cap_parts(root,[int(x) for x in main["seeds"]])
    else:
        result=run_cap_replication(root,main,load_yaml(a.policy_config),a.device,a.seeds,a.output_tag)
    print({"rows":len(result)})
if __name__=="__main__":main()
