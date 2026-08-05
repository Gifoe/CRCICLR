from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from hsc_tta.budgeted_risk.diagnostics.mini import _atomic_json,_pass,_sha

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--project-root",required=True);a=p.parse_args()
    repo=Path(a.project_root)/"repo";out=repo/"outputs/budgeted_risk_v51_mini";delivery=repo/"delivery/budgeted_risk_v51_mini"
    summary=pd.read_csv(out/"MINI_SUMMARY.csv");summary["pass"]=[_pass(r) for r in summary.itertuples()];summary.to_csv(out/"MINI_SUMMARY.csv",index=False)
    files=[]
    for root in (delivery,out):
        for path in sorted(x for x in root.rglob("*") if x.is_file() and x.name!="DELIVERY_MANIFEST.json"):
            files.append({"path":str(path.relative_to(repo)),"sha256":_sha(path),"bytes":path.stat().st_size})
    verdict=(delivery/"MINI_DECISION.json").read_text(encoding="utf-8")
    import json
    _atomic_json({"verdict":json.loads(verdict)["verdict"],"files":files},delivery/"DELIVERY_MANIFEST.json")
