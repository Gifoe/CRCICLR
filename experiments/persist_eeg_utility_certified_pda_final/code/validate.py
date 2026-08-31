"""Package and protocol integrity validation (scientific gate may be false)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

import upda_core as c


REQUIRED_DOCS = ["README.md","PREVIOUS_PDA_FORENSIC_AUDIT.md","SCIENTIFIC_RATIONALE.md","METHOD.md","RELATED_METHOD_BOUNDARY.md","ACTIONABILITY_CERTIFICATE_REPORT.md","SOURCE_DEVELOPMENT_REPORT.md","SOURCE_MECHANISM_REPORT.md","WBCIC_S2_REPORT.md","EEGNEX_REPORT.md","CONTROL_REPORT.md","ORACLE_HEADROOM_REPORT.md","LEAKAGE_AUDIT.md","CLAIM_AUDIT.md","ITERATION_LEDGER.md","REPRODUCIBILITY.md","FINAL_REPORT.md","FINAL_REPORT.json"]
REQUIRED_RESULTS = ["PREVIOUS_PDA_UTILITY_PREDICTIVENESS.csv","SOURCE_RECIPE_SEARCH.csv","SOURCE_PER_SUBJECT.csv","SOURCE_PER_FOLD.csv","HISTORICAL_CROSSFIT_UTILITY.csv","SUBJECT_ALPHA.csv","ALPHA_CALIBRATION.csv","CERTIFICATE_FUTURE_ASSOCIATION.csv","CORRECT_WRONG_SHUFFLED.csv","RANDOM_GATE_CONTROL.csv","ORACLE_HEADROOM.csv","WBCIC_S2_PER_SUBJECT.csv","WBCIC_S2_PER_FOLD.csv","EEGNEX_PER_SUBJECT.csv","EEGNEX_PER_FOLD.csv","STATISTICS.json","VALIDATION.json"]
REQUIRED_PROTOCOL = ["RESOURCE_LEDGER.md","DATA_ACCESS_LOCK.json","U_PDA_SOURCE_LOCK.json","U_PDA_FINAL_METHOD_LOCK.json","U_PDA_OUTER_CONFIRMATION_LOCK.json"]


def check() -> dict:
    errors=[]; checks={}
    for f in REQUIRED_DOCS: checks[f]=bool((c.EXP/f).is_file()); errors.extend([] if checks[f] else [f"missing document {f}"])
    # VALIDATION.json is the output of this command, so it is not a
    # prerequisite on the first invocation.
    for f in REQUIRED_RESULTS:
        if f == "VALIDATION.json":
            continue
        checks[f]=bool((c.RESULTS/f).is_file()); errors.extend([] if checks[f] else [f"missing result {f}"])
    for f in REQUIRED_PROTOCOL: checks[f]=bool((c.EXP/"protocol"/f).is_file()); errors.extend([] if checks[f] else [f"missing protocol {f}"])
    try:
        sp=pd.read_csv(c.RESULTS/"SOURCE_PER_SUBJECT.csv"); sa=pd.read_csv(c.RESULTS/"SUBJECT_ALPHA.csv"); hist=pd.read_csv(c.RESULTS/"HISTORICAL_CROSSFIT_UTILITY.csv")
        required_methods={"population","intercept_only","ordinary_adapter","previous_full_pda","persistent_ce","u_pda","eb_u_pda","random_gate","correct_adapter","wrong_adapter","shuffled_adapter","oracle_alpha"}
        checks["all_methods_present"]=required_methods.issubset(set(sp.method)); errors.extend([] if checks["all_methods_present"] else ["required method missing"])
        curve=pd.read_csv(c.RESULTS/"ALPHA_CALIBRATION.csv")
        observed=set(round(float(x),2) for x in pd.concat([sa.alpha,curve.alpha]).dropna().unique())
        checks["alpha_set_exact"]=observed==set(c.ALPHAS); errors.extend([] if checks["alpha_set_exact"] else [f"alpha set mismatch: {sorted(observed)}"])
        checks["no_future_fit_flags"]=not bool(sp.future_labels_used_for_fit.any() or sp.future_session_used_for_fit.any()); errors.extend([] if checks["no_future_fit_flags"] else ["future fit flag set"])
        checks["no_future_selection_flags"]="future_labels_used_for_selection" in sa and not bool(sa.future_labels_used_for_selection.any()); errors.extend([] if checks["no_future_selection_flags"] else ["future selection flag set"])
        checks["held_block_excluded"]=not bool(hist.held_block_used_for_fit.any()); errors.extend([] if checks["held_block_excluded"] else ["held block fit flag set"])
        checks["checkpoint_unchanged"]=bool(sp.population_checkpoint_unchanged.all()); errors.extend([] if checks["checkpoint_unchanged"] else ["population checkpoint changed"])
        checks["future_placeholders_empty"]=all(pd.read_csv(c.RESULTS/f).empty for f in ["WBCIC_S2_PER_SUBJECT.csv","WBCIC_S2_PER_FOLD.csv","EEGNEX_PER_SUBJECT.csv","EEGNEX_PER_FOLD.csv"]); errors.extend([] if checks["future_placeholders_empty"] else ["sealed placeholder contains observations"])
    except Exception as e: errors.append(f"result parse failure: {e}")
    try:
        gate=json.loads((c.RESULTS/"SOURCE_GATE.json").read_text()); checks["terminal_consistent"]=gate.get("terminal")=="U_PDA_SOURCE_NOT_SUPPORTED"; errors.extend([] if checks["terminal_consistent"] else ["unexpected terminal"])
        checks["bootstrap_10000"]=json.loads((c.RESULTS/"STATISTICS.json").read_text()).get("n_bootstrap")==10000; errors.extend([] if checks["bootstrap_10000"] else ["bootstrap count mismatch"])
    except Exception as e: errors.append(f"json parse failure: {e}")
    # No large/runtime artifact is admissible in the package directory.
    bad=[]
    for p in c.EXP.rglob("*"):
        if not p.is_file(): continue
        rel=str(p.relative_to(c.EXP)).lower()
        if rel.startswith("runtime\\") or rel.endswith((".npz",".npy",".pt",".pth",".ckpt")): bad.append(rel)
    checks["no_runtime_or_checkpoints"]=not bad; errors.extend([] if not bad else [f"forbidden packaged artifacts: {bad[:5]}"])
    checks["true_ce_primary"]= "class-balanced" in (c.EXP/"METHOD.md").read_text(encoding="utf-8") and "artificial desired-logit" in (c.EXP/"METHOD.md").read_text(encoding="utf-8"); errors.extend([] if checks["true_ce_primary"] else ["true CE method statement missing"])
    result={"pass":not errors,"package_integrity_pass":not errors,"scientific_gate_pass":json.loads((c.RESULTS/"SOURCE_GATE.json").read_text()).get("source_gate_pass",False),"terminal":json.loads((c.RESULTS/"SOURCE_GATE.json").read_text()).get("terminal"),"checks":checks,"errors":errors}
    return result


def main():
    result=check(); c.write_json(c.RESULTS/"VALIDATION.json",result); print(json.dumps(result,indent=2,sort_keys=True)); raise SystemExit(0 if result["pass"] else 1)


if __name__=="__main__": main()
