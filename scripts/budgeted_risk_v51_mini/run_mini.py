from __future__ import annotations
import argparse
from hsc_tta.budgeted_risk.diagnostics.mini import run_mini

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--project-root",required=True);p.add_argument("--resume",action="store_true");a=p.parse_args();print(run_mini(a.project_root,a.resume))

