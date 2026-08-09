#!/usr/bin/env python
from __future__ import annotations
import argparse
from hsc_tta.v3.nested_surfaces import build_nested_surfaces
from _common import load_yaml,project_root
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--config",required=True);p.add_argument("--dataset",required=True);p.add_argument("--seed",required=True,type=int);p.add_argument("--output-tag",required=True);p.add_argument("--device",default="cuda")
 a=p.parse_args();x=build_nested_surfaces(project_root(a.root),load_yaml(a.config),a.dataset,a.seed,a.output_tag,a.device);print({"rows":len(x)})
if __name__=="__main__":main()
