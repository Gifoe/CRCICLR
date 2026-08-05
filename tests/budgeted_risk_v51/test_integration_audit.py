import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest


REPO=Path(__file__).resolve().parents[2]


def test_ordinal_fix_is_run_commit_ancestor():
    state_path=REPO/"outputs/budgeted_risk/RUN_STATE.json"
    if not state_path.exists():pytest.skip("server result tree unavailable")
    run=json.loads(state_path.read_text())["git_commit"]
    ordinal=subprocess.check_output(["git","-C",str(REPO),"log","--all","--format=%H","--grep=implement_stage0_budget_gate_and_true_ordinal_candidate","-1"],text=True).strip()
    assert subprocess.run(["git","-C",str(REPO),"merge-base","--is-ancestor",ordinal,run]).returncode==0


def test_protected_cohort_not_in_source_manifest():
    cohort_path=REPO/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet";manifest_path=REPO/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet"
    if not cohort_path.exists() or not manifest_path.exists():pytest.skip("server result tree unavailable")
    cohort=pd.read_parquet(cohort_path);manifest=pd.read_parquet(manifest_path)
    allowed=set(zip(cohort[cohort.master_cohort=="method_development"].dataset,cohort[cohort.master_cohort=="method_development"].subject_id))
    assert set(zip(manifest.dataset,manifest.subject_id))<=allowed


def test_s1_result_reproduction_if_complete():
    path=REPO/"outputs/budgeted_risk_v51/results/S1_REPRODUCTION.json"
    if not path.exists():pytest.skip("diagnostic not yet complete")
    payload=json.loads(path.read_text());assert payload["passed"]


def test_input_hashes_match_if_audited():
    path=REPO/"outputs/budgeted_risk_v51/audit/INPUT_HASHES.json"
    if not path.exists():pytest.skip("diagnostic not yet audited")
    import hashlib
    for relative,expected in json.loads(path.read_text()).items():
        assert hashlib.sha256((REPO/relative).read_bytes()).hexdigest()==expected

