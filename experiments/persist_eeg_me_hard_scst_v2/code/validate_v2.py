"""Fail-closed compact artifact validator."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import v2_common as c


DOCS = ("README.md","CODE_MAP.md","V1_REPRODUCTION_AUDIT.md","SCIENTIFIC_RATIONALE.md","METHOD.md","LEAKAGE_AUDIT.md","MIXED_EFFECTS_BANK_AUDIT.md","BANK_STALENESS_AUDIT.md","ADMISSIBILITY_AUDIT.md","HARDNESS_AUDIT.md","HARD_RANDOM_MATCHING_AUDIT.md","SOURCE_DEVELOPMENT_REPORT.md","DISCOVERY_REPORT.md","CONFIRMATION_REPORT.md","CONTROL_REPORT.md","CLAIM_AUDIT.md","ITERATION_LEDGER.md","REPRODUCIBILITY.md","FINAL_REPORT.md","FINAL_REPORT.json")
RESULTS=("V1_REPRODUCTION.json","SOURCE_RECIPE_SEARCH.csv","BANK_DECOMPOSITION.csv","CANDIDATE_COVERAGE.csv","HARDNESS_DISTRIBUTION.csv","HARD_RANDOM_MATCHING.csv","DISCOVERY_PER_SUBJECT.csv","DISCOVERY_PER_FOLD.csv","DISCOVERY_SUMMARY.csv","CONFIRMATION_PER_SUBJECT.csv","CONFIRMATION_SUMMARY.csv","CONTROL_COMPARISON.csv","STATISTICS.json")
FIGURES=("main_effect_vs_interaction","valid_candidate_coverage","hardness_distribution","structured_vs_hard_random","subject_level_gain","cross_architecture_gain")
TERMINALS={"ME_HARD_SCST_CROSS_ARCH_SUPPORTED","ME_HARD_SCST_DISCOVERY_SUPPORTED_CONFIRMATION_FAILED","ME_HARD_SCST_NOT_SUPPORTED","ME_HARD_SCST_MECHANISM_NOT_REALIZED"}


def main()->None:
    checks={}
    checks["documents"]=all((c.EXP/name).is_file() for name in DOCS)
    checks["results"]=all((c.RESULTS/name).is_file() for name in RESULTS)
    checks["figures"]=all((c.FIGURES/f"{name}.{suffix}").is_file() for name in FIGURES for suffix in ("png","pdf"))
    report=c.read_json(c.EXP/"FINAL_REPORT.json")
    checks["terminal"]=report.get("terminal") in TERMINALS
    checks["v1_negative_preserved"]=report.get("immutable_v1_terminal")=="SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE"
    checks["v1_reproduced"]=c.read_json(c.RESULTS/"V1_REPRODUCTION.json").get("artifact_backed_reproduction_pass") is True
    source=pd.read_csv(c.RESULTS/"SOURCE_RECIPE_SEARCH.csv")
    checks["source_grid"]=len(source)==24 and (source.units==15).all()
    checks["outer_unopened"]=report.get("outer_resource_status")=="NOT_OPENED"
    if report["terminal"]=="ME_HARD_SCST_MECHANISM_NOT_REALIZED":
        checks["s3_stop_obeyed"]=c.read_json(c.RESULTS/"SOURCE_DECISION.json").get("s3_opened") is False
    else:
        checks["lock_present"]=all((c.PROTOCOL/name).is_file() for name in ("DATA_ACCESS_LOCK.json","SOURCE_DEVELOPMENT_LOCK.json","ME_HARD_SCST_V2_LOCK.json","OUTER_CONFIRMATION_LOCK_TEMPLATE.json"))
    validation={"pass":bool(all(checks.values())),"checks":checks,"terminal":report["terminal"]}
    c.write_json(c.RESULTS/"VALIDATION.json",validation)
    print(json.dumps(validation,indent=2))
    if not validation["pass"]:raise RuntimeError(validation)


if __name__=="__main__":main()

