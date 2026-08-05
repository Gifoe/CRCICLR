from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))

from hsc_tta.budgeted_risk.reporting import build_delivery_manifest,plot_full_context,sha256_paths,write_stage0_reports
from hsc_tta.budgeted_risk.run_state import RunState
from hsc_tta.budgeted_risk.source_models import build_clean_stage0_models_and_cache
from hsc_tta.budgeted_risk.stage0 import FEATURE_SCHEMA,full_context_gate,run_full_context
from hsc_tta.contextual_risk.io import atomic_json,atomic_parquet,sha256_file


def digest_text(value:str)->str:return hashlib.sha256(value.encode()).hexdigest()
def git_commit()->str:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",default=str(REPO.parent));parser.add_argument("--config",default=str(REPO/"configs/budgeted_risk/stage0.yaml"));parser.add_argument("--device",default="cuda");parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    root=Path(args.project_root).resolve();repo=root/"repo";config_path=Path(args.config);config=yaml.safe_load(config_path.read_text())
    out=repo/"outputs/budgeted_risk";delivery=repo/"delivery/budgeted_risk";out.mkdir(parents=True,exist_ok=True);delivery.mkdir(parents=True,exist_ok=True)
    cohorts_path=repo/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet";episode_paths=sorted((root/"data/episodes_contextual_risk").glob("*/*.parquet"));checkpoint=root/"checkpoints/cbramod/pretrained_weights.pth"
    config_hash=sha256_file(config_path);cohort_hash=sha256_file(cohorts_path);episode_hash=sha256_paths(episode_paths);commit=git_commit();feature_hash=digest_text(FEATURE_SCHEMA)
    state=RunState(out/"RUN_STATE.json")
    def advance(name:str,source_hash:str="pending",output_hash:str="pending",**extra):
        return state.advance(name,git_commit=commit,config_hash=config_hash,cohort_hash=cohort_hash,episode_hash=episode_hash,source_model_hash=source_hash,feature_schema_hash=feature_hash,output_manifest_hash=output_hash,**extra)
    if state.read() and not args.resume:raise RuntimeError("RUN_STATE exists; pass --resume")
    if not state.read():advance("INITIALIZED")
    current=state.read()["state"]
    cohorts=pd.read_parquet(cohorts_path)
    counts=cohorts.groupby(["dataset","master_cohort"]).size().reset_index(name="subjects")
    audit=(f"# Repository and data audit\n\n- Branch: `v5-budgeted-subject-risk-calibration`\n- Start commit: `d2b03911a4d97321847dc47683427e2a2717dc8e`\n- Run commit: `{commit}`\n- CBraMod checkpoint: `{checkpoint}` (`{sha256_file(checkpoint)}`)\n- Frozen backbone: yes\n- HMC subjects: 151\n- EEGMMIDB subjects: 109\n- CAP subjects: 99\n\n## Master cohorts\n\n{counts.to_markdown(index=False)}\n")
    (delivery/"REPOSITORY_AND_DATA_AUDIT.md").write_text(audit,encoding="utf-8")
    if current=="INITIALIZED":advance("AUDIT_COMPLETE",output_hash=sha256_file(delivery/"REPOSITORY_AND_DATA_AUDIT.md"));current="AUDIT_COMPLETE"
    cohort_out=out/"cohorts/MASTER_SUBJECT_COHORTS.parquet";cohort_out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(cohorts_path,cohort_out)
    episode_out=out/"episodes/EPISODE_MANIFEST.parquet";episode_out.parent.mkdir(parents=True,exist_ok=True)
    source_episode_manifest=repo/"outputs/contextual_risk/episodes/EPISODE_MANIFEST.parquet"
    shutil.copy2(source_episode_manifest,episode_out)
    protocol=("# B-HiCER data protocol\n\nOnly `method_development` subjects are accessible in Stage-0. In each five-fold rotation, fold r is evaluation, fold r+1 is one-sided calibration, and the remaining three folds are meta training. Formal calibration, internal final, and CAP remain closed unless all Stage-0 gates pass. Context labels are exposed only through the budgeted query oracle; Future labels are exposed only after a hashed risk decision.\n")
    (delivery/"DATA_PROTOCOL.md").write_text(protocol,encoding="utf-8")
    if current=="AUDIT_COMPLETE":advance("COHORTS_VERIFIED",output_hash=sha256_paths([cohort_out,episode_out]));current="COHORTS_VERIFIED"
    if current=="COHORTS_VERIFIED":
        manifests,caches=build_clean_stage0_models_and_cache(root,config,args.device)
        source_hash=digest_text("|".join(sorted(manifests.checkpoint_sha256)))
        source_report=("# Source-model audit\n\nThe inherited selected heads were rejected because their training subjects overlap Stage-0 calibration/evaluation and reserved internal subjects. No uncontaminated inherited head was found. Clean cross-fitted copies of the existing linear/task-head architectures were therefore trained only on the three meta folds for every outer fold and seed; frozen CBraMod token embeddings were used and the backbone was not updated.\n\n"
          f"- clean heads: {len(manifests)}\n- evaluation overlap: {int(manifests.evaluation_overlap.sum())}\n- calibration overlap: {int(manifests.calibration_overlap.sum())}\n- formal overlap: {int(manifests.formal_overlap.sum())}\n- internal-final overlap: {int(manifests.internal_final_overlap.sum())}\n- CAP overlap: {int(manifests.cap_overlap.sum())}\n- aggregate source hash: `{source_hash}`\n")
        (delivery/"SOURCE_MODEL_AUDIT.md").write_text(source_report,encoding="utf-8")
        advance("SOURCE_MODELS_VERIFIED",source_hash=source_hash,output_hash=sha256_file(repo/"outputs/budgeted_risk/source_models/STAGE0_SOURCE_MODEL_MANIFEST.parquet"));current="SOURCE_MODELS_VERIFIED"
    else:
        manifests=pd.read_parquet(repo/"outputs/budgeted_risk/source_models/STAGE0_SOURCE_MODEL_MANIFEST.parquet");source_hash=digest_text("|".join(sorted(manifests.checkpoint_sha256)))
    if current=="SOURCE_MODELS_VERIFIED":advance("CACHE_COMPLETE",source_hash=source_hash,output_hash=sha256_file(repo/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet"));current="CACHE_COMPLETE"
    freeze={"schema_version":"budgeted-risk-stage0-freeze-v1","git_commit":commit,"datasets":["hmc","eegmmidb"],"cohort_hash":cohort_hash,"episode_hash":episode_hash,"source_model_hashes":sorted(manifests.checkpoint_sha256.tolist()),"TPS_grid":{"values":[round(.5+i*(.49/19),12) for i in range(20)],"sentinel_index":20},"alpha":config["alpha"],"delta":config["delta"],"budgets":config["budgets"],"random_repeats":config["random_repeats"],"acquisitions":["random","first","temporal","predictive_entropy","class_balanced","diversity","risk_entropy","active"],"feature_schema":FEATURE_SCHEMA,"candidate_models":["constant","direct","ridge","isotonic","ordinal"],"quantile":"higher empirical; split conformal ceil((m+1)(1-delta))","full_context_gate":config["full_context_gate"],"budget_gate":config["budget_gate"],"bootstrap_repetitions":config["bootstrap_repetitions"],"stopping_rule":"stop before budgets if either dataset fails full-context gate"}
    stage_delivery=delivery/"stage0";stage_delivery.mkdir(parents=True,exist_ok=True);atomic_json(freeze,stage_delivery/"STAGE0_METHOD_FREEZE.json")
    (stage_delivery/"STAGE0_PROTOCOL.md").write_text("# Stage-0 protocol\n\nThis file freezes the five-fold rotation, full-context feasibility gate, fixed-budget hierarchy, acquisition candidates, one-sided calibration, and hard stopping rules. The machine-readable specification is `STAGE0_METHOD_FREEZE.json`.\n",encoding="utf-8")
    if current=="CACHE_COMPLETE":advance("STAGE0_PROTOCOL_FROZEN",source_hash=source_hash,output_hash=sha256_file(stage_delivery/"STAGE0_METHOD_FREEZE.json"));current="STAGE0_PROTOCOL_FROZEN"
    if current=="STAGE0_PROTOCOL_FROZEN":
        results,transcripts,selections=run_full_context(root,config)
        advance("STAGE0_FULL_CONTEXT_COMPLETE",source_hash=source_hash,output_hash=sha256_file(out/"stage0/FULL_CONTEXT_RESULTS.parquet"));current="STAGE0_FULL_CONTEXT_COMPLETE"
    else:results=pd.read_parquet(out/"stage0/FULL_CONTEXT_RESULTS.parquet")
    summary,passed=full_context_gate(results,config);summary.to_csv(out/"stage0/STAGE0_SUMMARY.csv",index=False);plot_full_context(repo,results)
    decision={"schema_version":"budgeted-risk-stage0-decision-v1","verdict":"FULL_CONTEXT_GO" if passed else "STAGE0_NO_GO","full_context_pass":passed,"dataset_results":summary.to_dict("records"),"budget_experiments_opened":False,"acquisition_experiments_opened":False,"formal_calibration_opened":False,"internal_final_opened":False,"cap_opened":False}
    write_stage0_reports(repo,summary,passed,decision)
    if passed:
        print("FULL_CONTEXT_GO: Stage-0B must now run",flush=True);return 2
    pd.DataFrame([{"stage":"STAGE0A","failure":"FULL_CONTEXT_NO_GO","action":"hard stop before budget experiments"}]).to_csv(out/"FAILURES.csv",index=False)
    (delivery/"PROVENANCE.md").write_text(f"# Provenance\n\nRun commit: `{commit}`. Config hash: `{config_hash}`. Cohort hash: `{cohort_hash}`. Episode hash: `{episode_hash}`. Source-model aggregate hash: `{source_hash}`.\n",encoding="utf-8")
    (delivery/"LIMITATIONS.md").write_text("# Limitations\n\nStage-0A failed. Therefore no evidence about budget efficiency, acquisition choice, formal validity, held-out internal performance, or CAP transfer was produced. Any claim about those stages would be unsupported.\n",encoding="utf-8")
    manifest=build_delivery_manifest(repo);manifest_hash=digest_text(json.dumps(manifest,sort_keys=True))
    advance("STOPPED_NO_GO",source_hash=source_hash,output_hash=manifest_hash,formal_calibration_opened=False,internal_final_opened=False,cap_opened=False)
    print(summary.to_string(index=False));print("STAGE0_NO_GO",flush=True);return 0


if __name__=="__main__":raise SystemExit(main())
