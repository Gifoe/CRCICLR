from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",default=str(Path(__file__).resolve().parents[2]));args=parser.parse_args();repo=Path(args.repo).resolve();out=repo/"outputs/budgeted_risk"
    state=json.loads((out/"RUN_STATE.json").read_text());decision=json.loads((repo/"delivery/budgeted_risk/stage0/STAGE0_DECISION.json").read_text())
    assert state["state"]=="STOPPED_NO_GO"
    assert decision["verdict"]=="STAGE0_NO_GO" and decision["full_context_pass"] and not decision["budget_pass"]
    for key in ("formal_calibration_opened","internal_final_opened","cap_opened"):assert state[key] is False and decision[key] is False
    source=pd.read_parquet(out/"source_models/STAGE0_SOURCE_MODEL_MANIFEST.parquet");cache=pd.read_parquet(out/"source_cache/STAGE0_CACHE_MANIFEST.parquet")
    assert len(source)==50 and len(cache)==3875
    overlap=["evaluation_overlap","calibration_overlap","formal_overlap","internal_final_overlap","cap_overlap"]
    assert int(source[overlap].to_numpy().sum())==0
    full=pd.read_parquet(out/"stage0/FULL_CONTEXT_RESULTS.parquet");budgets=pd.read_parquet(out/"stage0/BUDGET_RESULTS.parquet");tuning=pd.read_parquet(out/"stage0/BUDGET_TUNING.parquet");gate=pd.read_csv(out/"stage0/BUDGET_GATE_SUMMARY.csv")
    assert len(full)==775 and len(tuning)==350 and len(gate)==6 and not gate.budget_pass.any()
    temporal=budgets[budgets.strategy=="temporal"];random=budgets[budgets.strategy=="random"]
    assert len(temporal)==5425 and len(random)==108500 and len(budgets)==113925
    assert set(random.repeat.unique())==set(range(20)) and not (out/"stage0/ACQUISITION_RESULTS.parquet").exists()
    batch_files=list((out/"risk_decisions/stage0_random_eval_batch").rglob("*.json"));assert len(batch_files)==5425
    for path in (batch_files[0],batch_files[len(batch_files)//2],batch_files[-1]):
        payload=json.loads(path.read_text());canonical=json.dumps(payload["decisions"],sort_keys=True,separators=(",",":"));assert len(payload["decisions"])==20 and hashlib.sha256(canonical.encode()).hexdigest()==payload["batch_hash"]
    assert not any((out/"full_method").rglob("*")) if (out/"full_method").exists() else True
    print(json.dumps({"status":"VALID","full_context_rows":len(full),"budget_rows":len(budgets),"temporal_rows":len(temporal),"random_rows":len(random),"source_heads":len(source),"cache_files":len(cache),"decision_batches":len(batch_files),"formal_calibration_opened":False,"internal_final_opened":False,"cap_opened":False},indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
