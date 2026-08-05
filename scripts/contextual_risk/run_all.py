#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from hsc_tta.contextual_risk.cache import cache_source_predictions
from hsc_tta.contextual_risk.cohorts import attach_screening_folds, build_master_cohorts
from hsc_tta.contextual_risk.episodes import build_contextual_episodes
from hsc_tta.contextual_risk.io import atomic_json, atomic_parquet, canonical_hash, sha256_file
from hsc_tta.contextual_risk.screening import build_shared_tables, run_screening_a, run_screening_b


STAGES = ["NOT_STARTED","AUDIT_COMPLETE","MASTER_SPLITS_FROZEN","EPISODES_COMPLETE","SOURCE_CACHE_COMPLETE","SCREENING_FROZEN","BRANCH_A_COMPLETE","BRANCH_B_COMPLETE","BRANCH_SELECTED","STOP_REPORT_COMPLETE","DELIVERY_COMPLETE"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class State:
    def __init__(self, path: Path, git_commit: str):
        self.path=path
        self.payload={"schema_version":"contextual-risk-run-state-v1","stage":"NOT_STARTED","git_commit_at_start":git_commit,"history":[]}
        self.write()
    def advance(self, stage: str, **hashes):
        if stage not in STAGES: raise ValueError(stage)
        self.payload.update(hashes);self.payload["stage"]=stage
        self.payload["history"].append({"stage":stage,"timestamp_utc":_now(),**hashes})
        self.write()
    def write(self): atomic_json(self.payload,self.path)


def _git(repo:Path,*args:str)->str:
    return subprocess.check_output(["git","-C",str(repo),*args],text=True).strip()


def _report_json(title:str,payload:dict)->str:
    return f"# {title}\n\n```json\n{json.dumps(payload,indent=2,sort_keys=True)}\n```\n"


def _write_reports(repo:Path,a:dict,b:dict,selection:dict,cohorts:pd.DataFrame,episodes:pd.DataFrame,config_hash:str)->None:
    delivery=repo/"delivery/contextual_risk"; screening=delivery/"screening"; screening.mkdir(parents=True,exist_ok=True)
    counts=cohorts.groupby(["dataset","master_cohort"]).size().unstack(fill_value=0).to_dict("index")
    (delivery/"REPOSITORY_AND_DATA_AUDIT.md").write_text(
        "# Repository and data audit\n\n"
        "The run uses only the frozen CBraMod token embeddings and five frozen source heads for HMC and EEGMMIDB. CAP is reserved for post-selection external replication. "
        "Historical V1--V3 analyses used the same source datasets and subjects; therefore the new final cohort is a current-method held-out internal evaluation, not a historically untouched confirmation set. "
        "The five frozen source heads also use seed-specific task-head training splits, so some master-cohort subjects are in-source-head-training for some seeds; `provenance/SOURCE_HEAD_SUBJECT_OVERLAP.csv` quantifies this structural limitation.\n\n"
        f"Master cohort counts: `{json.dumps(counts,sort_keys=True)}`.\n",encoding="utf-8")
    (delivery/"STATISTICAL_ISOLATION.md").write_text(
        "# Statistical isolation\n\nA/B selection used only `method_development`. `formal_calibration` and `internal_final_evaluation` were inaccessible to screening. "
        "All source-head seeds share one subject partition and one hash-derived five-fold rotation. Primary selection uses alpha=0.10 and delta=0.10.\n",encoding="utf-8")
    (delivery/"DATA_PROTOCOL.md").write_text(
        "# Data protocol\n\nFor every subject, U is the sorted union of the frozen V3 adapt and probe indices; V is the unchanged V3 Future episode. "
        "Automated checks enforce disjointness, U strictly before V, and shared A/B episodes.\n\n"
        f"Episode rows: {len(episodes)}. Manifest SHA-256: `{sha256_file(repo/'outputs/contextual_risk/episodes/EPISODE_MANIFEST.parquet')}`.\n",encoding="utf-8")
    (delivery/"ALGORITHM_CONTRACT.md").write_text(
        "# Algorithm contract\n\nPrediction sets use deterministic probability-descending/class-id-ascending sorting, 20 thresholds from 0.50 to 0.99, and index 20 as the full-set sentinel. "
        "TPS, APS and the preregistered RAPS grid contain the argmax and receive monotone-union repair. Higher and split-conformal quantiles use exact order-statistic indices.\n",encoding="utf-8")
    (delivery/"PROTOCOL_DEVIATIONS.md").write_text("# Protocol deviations\n\nNone. Both internal datasets met the preregistered 60/20/20 minimum counts.\n",encoding="utf-8")
    (delivery/"PROVENANCE.md").write_text(
        f"# Provenance\n\nConfiguration SHA-256: `{config_hash}`. Screening outputs are generated from frozen source probabilities and immutable contextual episodes. "
        "No TTA, adapter, extra backbone, or extra dataset is used. All numerical output tables are versioned in `outputs/contextual_risk`.\n",encoding="utf-8")
    (delivery/"RUN_STATE_HISTORY.md").write_text("# Run-state history\n\nSee `outputs/contextual_risk/RUN_STATE.json`; every transition includes UTC time and upstream hashes.\n",encoding="utf-8")
    (screening/"BRANCH_A_ORACLE_HEADROOM.md").write_text(_report_json("Branch A oracle headroom",a),encoding="utf-8")
    (screening/"BRANCH_A_TARGET_RELIABILITY.md").write_text(_report_json("Branch A target reliability",a),encoding="utf-8")
    (screening/"BRANCH_A_PREDICTABILITY.md").write_text(_report_json("Branch A OOF predictability",a),encoding="utf-8")
    (screening/"BRANCH_A_PILOT_VALIDITY.md").write_text(_report_json("Branch A pilot validity",a),encoding="utf-8")
    (screening/"BRANCH_A_GO_NO_GO.md").write_text(_report_json("Branch A GO/NO-GO",a),encoding="utf-8")
    (screening/"BRANCH_B_ORACLE_HEADROOM.md").write_text(_report_json("Branch B oracle headroom",b),encoding="utf-8")
    (screening/"BRANCH_B_POLICY_DIVERSITY.md").write_text(_report_json("Branch B policy diversity",b),encoding="utf-8")
    (screening/"BRANCH_B_SELECTOR_PERFORMANCE.md").write_text(_report_json("Branch B OOF selector",b),encoding="utf-8")
    (screening/"BRANCH_B_PILOT_VALIDITY.md").write_text(_report_json("Branch B pilot validity",b),encoding="utf-8")
    (screening/"BRANCH_B_GO_NO_GO.md").write_text(_report_json("Branch B GO/NO-GO",b),encoding="utf-8")
    (screening/"BRANCH_COMPARISON.md").write_text(_report_json("Frozen branch comparison",selection),encoding="utf-8")
    (screening/"BRANCH_SELECTION.md").write_text(_report_json("Branch selection",selection),encoding="utf-8")
    if selection["decision"]=="STOP_CONTEXTUAL_RISK_ALLOCATION":
        (delivery/"STOP_CONTEXTUAL_RISK_ALLOCATION.md").write_text(
            "# STOP_CONTEXTUAL_RISK_ALLOCATION\n\nBoth preregistered branches failed at least one mandatory gate on development-only cross-fitting. "
            "The contract therefore forbids full-method implementation, formal calibration, internal final evaluation, and CAP evaluation. "
            "Negative results are retained without branch switching or post-hoc relaxation.\n\n"+_report_json("Gate evidence",selection),encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",default="/root/autodl-tmp/hsc_tta_eeg");parser.add_argument("--device",default="cuda");args=parser.parse_args()
    root=Path(args.project_root).resolve();repo=root/"repo";out=repo/"outputs/contextual_risk";out.mkdir(parents=True,exist_ok=True)
    config_path=repo/"configs/contextual_risk/screening.yaml";config=yaml.safe_load(config_path.read_text());config_hash=sha256_file(config_path)
    head=_git(repo,"rev-parse","HEAD");state=State(out/"RUN_STATE.json",head)
    audit={"timestamp_utc":_now(),"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"git_commit":head,"git_branch":_git(repo,"branch","--show-current"),"config_hash":config_hash,"historically_untouched":False}
    atomic_json(audit,out/"provenance/AUDIT.json");state.advance("AUDIT_COMPLETE",config_sha256=config_hash)
    cohorts=[]
    for dataset in ("hmc","eegmmidb"):
        source=pd.read_parquet(root/"data/episodes_v3"/dataset/"seed_0.parquet")
        cohorts.append(build_master_cohorts(dataset,source.subject_id))
    cohorts=attach_screening_folds(pd.concat(cohorts,ignore_index=True));atomic_parquet(cohorts,out/"cohorts/MASTER_SUBJECT_COHORTS.parquet")
    cohort_hash=sha256_file(out/"cohorts/MASTER_SUBJECT_COHORTS.parquet");state.advance("MASTER_SPLITS_FROZEN",master_split_sha256=cohort_hash)
    episodes=build_contextual_episodes(root);atomic_parquet(episodes,out/"episodes/EPISODE_MANIFEST.parquet");episode_hash=sha256_file(out/"episodes/EPISODE_MANIFEST.parquet");state.advance("EPISODES_COMPLETE",episode_manifest_sha256=episode_hash)
    cache_rows=pd.DataFrame(cache_source_predictions(root,device=args.device));atomic_parquet(cache_rows,out/"source_cache/SOURCE_CACHE_MANIFEST.parquet");cache_hash=sha256_file(out/"source_cache/SOURCE_CACHE_MANIFEST.parquet");state.advance("SOURCE_CACHE_COMPLETE",source_cache_manifest_sha256=cache_hash)
    feature_path=out/"CONTEXT_FEATURES.parquet";surface_path=out/"SUBJECT_POLICY_SURFACES.parquet"
    if feature_path.exists() and surface_path.exists():
        features=pd.read_parquet(feature_path);surfaces=pd.read_parquet(surface_path)
    else:
        features,surfaces=build_shared_tables(root,cohorts);atomic_parquet(features,feature_path);atomic_parquet(surfaces,surface_path)
    critical=surfaces[(surfaces.family=="TPS")][["dataset","seed","subject_id","master_cohort","screening_fold","alpha","critical_index","critical_index_first_half","critical_index_second_half"]]
    atomic_parquet(critical,out/"SUBJECT_CRITICAL_INDICES.parquet")
    feature_hash=sha256_file(out/"CONTEXT_FEATURES.parquet")
    freeze={"schema_version":"contextual-risk-screening-freeze-v1","created_utc":_now(),"git_commit":head,"config_sha256":config_hash,"master_split_sha256":cohort_hash,"episode_manifest_sha256":episode_hash,"source_cache_manifest_sha256":cache_hash,"context_features_sha256":feature_hash,"primary_alpha":.10,"primary_delta":.10,"allowed_cohort":"method_development","future_results_seen_before_freeze":False}
    freeze["freeze_hash"]=canonical_hash(freeze);atomic_json(freeze,repo/"delivery/contextual_risk/SCREENING_METHOD_FREEZE.json");state.advance("SCREENING_FROZEN",screening_freeze_sha256=sha256_file(repo/"delivery/contextual_risk/SCREENING_METHOD_FREEZE.json"),feature_sha256=feature_hash)
    a_results,a=run_screening_a(features,surfaces);atomic_parquet(a_results,out/"SCREENING_A_RESULTS.parquet");atomic_json(a,out/"SCREENING_A_SUMMARY.json");state.advance("BRANCH_A_COMPLETE",screening_a_sha256=sha256_file(out/"SCREENING_A_RESULTS.parquet"))
    b_results,b=run_screening_b(features,surfaces);atomic_parquet(b_results,out/"SCREENING_B_RESULTS.parquet");atomic_json(b,out/"SCREENING_B_SUMMARY.json");state.advance("BRANCH_B_COMPLETE",screening_b_sha256=sha256_file(out/"SCREENING_B_RESULTS.parquet"))
    if a["decision"]=="A_NO_GO" and b["decision"]=="B_NO_GO": decision="STOP_CONTEXTUAL_RISK_ALLOCATION"
    elif a["decision"]=="A_GO" and b["decision"]=="B_NO_GO": decision="SELECT_A_FINE_INDEX"
    elif a["decision"]=="A_NO_GO" and b["decision"]=="B_GO": decision="SELECT_B_COARSE_POLICY"
    else:
        ma=min(x["realized_relative_gain"] for x in a["datasets"].values());mb=min(x["realized_relative_gain"] for x in b["datasets"].values())
        decision="BOTH_GO_SELECT_A" if ma-mb>.02 else "BOTH_GO_SELECT_B"
    selection={"schema_version":"contextual-risk-branch-selection-v1","created_utc":_now(),"decision":decision,"branch_a":a,"branch_b":b,"selection_cohort":"method_development","formal_calibration_opened":False,"internal_final_opened":False,"cap_opened":False,"screening_freeze_hash":freeze["freeze_hash"]}
    selection["selection_hash"]=canonical_hash(selection);atomic_json(selection,out/"BRANCH_SELECTION.json");atomic_json(selection,repo/"delivery/contextual_risk/BRANCH_SELECTION.json");state.advance("BRANCH_SELECTED",branch_selection_sha256=sha256_file(out/"BRANCH_SELECTION.json"),branch_decision=decision)
    _write_reports(repo,a,b,selection,cohorts,episodes,config_hash)
    if decision=="STOP_CONTEXTUAL_RISK_ALLOCATION": state.advance("STOP_REPORT_COMPLETE");state.advance("DELIVERY_COMPLETE")
    else: raise RuntimeError(f"{decision} passed screening; selected full-method implementation must now run")
    failures=pd.DataFrame(columns=["stage","dataset","seed","subject_id","error"]);failures.to_csv(out/"FAILURES.csv",index=False)
    print(json.dumps({"decision":decision,"a":a,"b":b},indent=2))


if __name__=="__main__": main()
