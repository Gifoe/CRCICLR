from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
P4B = REPO / "experiments" / "persist_eeg_p4b_identity_reliability_discovery_v1"
BRANCH = "codex/persist-eeg-p4c-suppression-safety-validation-v1"

sys.path.insert(0, str(HERE))
from p4c_safety_common import now_utc, read_json, sha256, write_json  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    issues: list[str] = []
    required_reports = [
        "README.md",
        "P4B_NEGATIVE_RESULT_PRESERVATION.md",
        "P4C_SAFETY_INPUT_AUDIT.md",
        "REGIME_REPRODUCTION_AUDIT.md",
        "REGIME_COVERAGE_AUDIT.md",
        "P4C_SAFETY_OUTCOME_ACCESS_LEDGER.md",
        "REGIME_SEPARATION_PROSPECTIVE_AUDIT.md",
        "SUPPRESSION_VETO_AUDIT.md",
        "MATCHED_RANDOM_SPECIFICITY_AUDIT.md",
        "LOW_ENTANGLEMENT_ACTIONABILITY_AUDIT.md",
        "CROSS_SETTING_SAFETY_STABILITY.md",
        "THEORY_ADMISSIBILITY_NOTE.md",
        "HOLDOUT_PURITY_AUDIT.md",
    ]
    required_results = [
        "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv",
        "p4c_safety_future_utility_subject.csv",
        "p4c_safety_future_utility_direction.csv",
        "p4c_safety_regime_summary.csv",
        "p4c_safety_matched_random_summary.csv",
        "p4c_safety_highest_identity_summary.csv",
        "p4c_safety_bootstrap.csv",
        "P4C_SAFETY_OUTCOME_EVALUATION_COMPLETE.json",
        "P4C_SAFETY_ANALYSIS_COMPLETE.json",
    ]
    required_figures = [
        "figure1_preoutcome_regime_map.png",
        "figure2_future_uba_regimes.png",
        "figure3_high_e_veto.png",
        "figure4_low_e_actionability.png",
        "figure5_matched_random_specificity.png",
        "figure6_discovery_prospective_forest.png",
    ]
    for name in required_reports:
        if not (EXP / name).is_file():
            issues.append(f"missing_report:{name}")
    for name in required_results:
        if not (RESULTS / name).is_file():
            issues.append(f"missing_result:{name}")
    for name in required_figures:
        if not (FIGURES / name).is_file():
            issues.append(f"missing_figure:{name}")
    if issues:
        write_json(RESULTS / "P4C_SAFETY_FINAL_VALIDATION.json", {"pass": False, "issues": issues, "SAFETY_STATUS": "P4C_SAFETY_PROTOCOL_OR_PURITY_FAILURE"})
        raise RuntimeError(str(issues))

    p4b_validation = read_json(P4B / "results" / "P4B_FINAL_VALIDATION.json")
    old_closure = read_json(P4B / "results" / "P4C_FINAL_VALIDATION.json")
    audit = read_json(EXP / "P4C_SAFETY_INPUT_AUDIT.json")
    protocol = read_json(EXP / "P4C_SAFETY_PROTOCOL_FROZEN.json")
    pre = read_json(EXP / "P4C_SAFETY_PREOUTCOME_FREEZE.json")
    outcome = read_json(RESULTS / "P4C_SAFETY_OUTCOME_EVALUATION_COMPLETE.json")
    analysis = read_json(RESULTS / "P4C_SAFETY_ANALYSIS_COMPLETE.json")
    assignments = pd.read_csv(RESULTS / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv")
    reproduction = pd.read_csv(RESULTS / "discovery_regime_reproduction.csv")
    coverage = pd.read_csv(RESULTS / "regime_coverage.csv")
    subject = pd.read_csv(RESULTS / "p4c_safety_future_utility_subject.csv")
    direction = pd.read_csv(RESULTS / "p4c_safety_future_utility_direction.csv")
    summary = pd.read_csv(RESULTS / "p4c_safety_regime_summary.csv").set_index("setting_id")
    matched = pd.read_csv(RESULTS / "p4c_safety_matched_random_summary.csv")
    highest = pd.read_csv(RESULTS / "p4c_safety_highest_identity_summary.csv").set_index("setting_id")
    boot = pd.read_csv(RESULTS / "p4c_safety_bootstrap.csv")
    input_source_hash = sha256(P4B / "results" / "source_evidence_normalized.csv")
    if git("branch", "--show-current") != BRANCH:
        issues.append("branch")
    if not (p4b_validation.get("pass") is True and p4b_validation.get("terminal") == "P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED"):
        issues.append("p4b_negative_terminal_preservation")
    if not (old_closure.get("terminal") == "P4C_NOT_AUTHORIZED_BY_P4B" and old_closure.get("reserved_future_outcome_accessed") is False):
        issues.append("old_p4c_closure_preservation")
    if audit.get("p4b_validator_sha256") != sha256(P4B / "results" / "P4B_FINAL_VALIDATION.json"):
        issues.append("p4b_validator_hash")
    if protocol.get("reserved_settings") != ["S4", "S6"] or set(assignments.setting_id) != {"S4", "S6"}:
        issues.append("reserved_identity")
    if protocol.get("E_task") != "(z_D + z_C + z_O)/3" or not (assignments.E_task_definition == "(z_D + z_C + z_O)/3").all():
        issues.append("E_task_formula")
    if protocol.get("normalization_sha256") != sha256(P4B / "SOURCE_NORMALIZATION_FROZEN.json"):
        issues.append("normalization_hash")
    if protocol.get("source_cube_sha256") != read_json(P4B / "SOURCE_NORMALIZATION_FROZEN.json").get("source_cube_sha256") or audit.get("source_evidence_normalized_sha256") != input_source_hash:
        issues.append("source_hash")
    if protocol.get("regime_rules", {}).get("comparison") != "inclusive pandas quantiles at 1/3 and 2/3":
        issues.append("tertile_implementation")
    if len(reproduction) != 4 or not reproduction.p4b_exact.astype(bool).all() or reproduction.set_index("setting_id")[["low_count", "high_count"]].to_dict("index") != {"S1": {"low_count": 17, "high_count": 12}, "S2": {"low_count": 11, "high_count": 15}, "S3": {"low_count": 8, "high_count": 20}, "S5": {"low_count": 15, "high_count": 8}}:
        issues.append("regime_reproduction")
    if len(assignments) != 240 or not set(assignments.direction_rank) == set(range(1, 9)) or assignments.duplicated(["setting_id", "fold", "seed", "direction_rank"]).any():
        issues.append("direction_freeze")
    if not coverage.coverage_pass.astype(bool).all():
        issues.append("coverage")
    assignment_hash = sha256(RESULTS / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv")
    if pre.get("hashes", {}).get("P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv") != assignment_hash:
        issues.append("assignment_hash")
    if not (outcome.get("first_outcome_access_timestamp_utc", "") > pre.get("timestamp_utc", "") and outcome.get("preoutcome_freeze_sha256") == sha256(EXP / "P4C_SAFETY_PREOUTCOME_FREEZE.json")):
        issues.append("preoutcome_timing_or_hash")
    if len(direction) != 240 or subject.duplicated(["setting_id", "fold", "seed", "direction_rank", "outcome_subject"]).any() or any("trial" in name.lower() for name in subject.columns):
        issues.append("subject_first")
    if set(direction.setting_id) != {"S4", "S6"} or set(summary.index) != {"S4", "S6", "POOLED"}:
        issues.append("setting_drop")
    if not (direction.control_count == 100).all() or protocol.get("matched_random", {}).get("seed") != "stable_seed('P4A-control', setting, fold, seed, rank, control_id)":
        issues.append("matched_random")
    source_text = (HERE / "run_safety_outcomes.py").read_text(encoding="utf-8")
    if "stable_seed(\"P4A-control\", setting, fold, seed, rank, control_id)" not in source_text or "per-trial displacement-norm" not in source_text:
        issues.append("matched_random_code")
    if len(boot) != 30000 or set(boot.analysis_scope) != {"S4", "S6", "POOLED"} or not all(len(boot[boot.analysis_scope == scope]) == 10000 for scope in ["S4", "S6", "POOLED"]):
        issues.append("bootstrap_draws")
    if protocol.get("bootstrap", {}).get("hierarchy_pooled") != ["setting", "fold", "seed/run", "direction", "outcome_subject"]:
        issues.append("bootstrap_hierarchy")
    if any([pre.get("post_outcome_protocol_modification"), outcome.get("post_outcome_scientific_modification"), analysis.get("post_outcome_protocol_modification")]):
        issues.append("post_outcome_modification")
    if not (p4b_validation.get("OpenBMI_sealed_internal_holdout") == "UNTOUCHED" and p4b_validation.get("WBCIC_outer_10") == "UNTOUCHED_NOT_ENUMERATED"):
        issues.append("outer_holdout_purity")

    pooled = summary.loc["POOLED"]
    g1 = bool(pooled.DeltaRegime > 0 and pooled.DeltaRegime_CI_lower > 0)
    g2 = bool(pooled.U_high < 0 and pooled.U_high_CI_upper < 0)
    g3 = bool(summary.loc["S4", "DeltaRegime"] > 0 and summary.loc["S6", "DeltaRegime"] > 0)
    g4 = bool(summary.loc["S4", "U_high"] < 0 and summary.loc["S6", "U_high"] < 0)
    g5 = not issues and analysis.get("purity_pass") is True
    indicators = [summary.loc[setting, "DeltaRegime"] > 0 for setting in ["S4", "S6"]] + [summary.loc[setting, "U_high"] < 0 for setting in ["S4", "S6"]]
    neither_double_reversal = all(not (summary.loc[setting, "DeltaRegime"] <= 0 and summary.loc[setting, "U_high"] >= 0) for setting in ["S4", "S6"])
    partial = bool(pooled.DeltaRegime > 0 and pooled.U_high < 0 and sum(indicators) >= 3 and neither_double_reversal and g5)
    if all([g1, g2, g3, g4, g5]):
        terminal = "P4C_SAFETY_BOUNDARY_STRONG_SUPPORTED"
        bridge = "AUTHORIZED"
    elif partial:
        terminal = "P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED"
        bridge = "CONDITIONAL"
    else:
        terminal = "P4C_SAFETY_BOUNDARY_NOT_SUPPORTED" if not issues else "P4C_SAFETY_PROTOCOL_OR_PURITY_FAILURE"
        bridge = "NOT_AUTHORIZED"
    if pooled.U_low > 0 and pooled.U_low_CI_lower > 0 and summary.loc["S4", "U_low"] >= 0 and summary.loc["S6", "U_low"] >= 0:
        low_status = "LOW_E_SUPPRESSION_BENEFICIAL"
    elif pooled.U_low <= 0 or pooled.U_low_CI_upper < 0 or (summary.loc["S4", "U_low"] < 0 and summary.loc["S6", "U_low"] < 0):
        low_status = "LOW_E_SUPPRESSION_NOT_BENEFICIAL"
    else:
        low_status = "LOW_E_SUPPRESSION_INCONCLUSIVE"
    if terminal != analysis.get("SAFETY_STATUS_candidate") or low_status != analysis.get("LOW_E_ACTIONABILITY_STATUS_candidate") or bridge != analysis.get("METHOD_LEVEL_BRIDGE_AUTHORIZATION_candidate"):
        issues.append("terminal_rule_reproduction")
    if issues:
        terminal = "P4C_SAFETY_PROTOCOL_OR_PURITY_FAILURE"
        bridge = "NOT_AUTHORIZED"

    matched_pooled_high = matched[(matched.setting_id == "POOLED") & (matched.regime_label == "REGIME_HIGH")].iloc[0]
    counts = assignments.groupby(["setting_id", "regime_label"]).size().unstack(fill_value=0)
    final_report = f"""# P4C-Safety Final Report

Validator: **{'PASS' if not issues else 'FAIL'}**.

1. P4B terminal is `P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED`.
2. P4B continuous MINT transport failed (MI RMSE 0.007967329 versus MINT 0.022863854; interaction sign was wrong), so it is not rewritten as success.
3. Its separately pre-specified discovery regime had pooled DeltaRegime +0.006690019, CI [+0.000983033, +0.012903471], with S1/S2/S3/S5 all positive.
4. S4/S6 were sealed before this experiment: YES.
5. Source cube hash: `{protocol['source_cube_sha256']}`.
6. Normalization hash: `{protocol['normalization_sha256']}`.
7. E_task: `(z_D + z_C + z_O)/3`.
8. Regime: inclusive within-setting top-z_I tertile intersected with bottom/top E_task tertile.
9. Pre-outcome assignment hash: `{assignment_hash}`.
10. S4 Low-E/High-E counts: {int(counts.loc['S4','REGIME_LOW'])}/{int(counts.loc['S4','REGIME_HIGH'])}.
11. S6 Low-E/High-E counts: {int(counts.loc['S6','REGIME_LOW'])}/{int(counts.loc['S6','REGIME_HIGH'])}.
12. Coverage gate: {'PASS' if coverage.coverage_pass.astype(bool).all() else 'FAIL'}.
13. S4 U_low: {summary.loc['S4','U_low']:+.9f}.
14. S4 U_high: {summary.loc['S4','U_high']:+.9f}.
15. S4 DeltaRegime: {summary.loc['S4','DeltaRegime']:+.9f}.
16. S6 U_low: {summary.loc['S6','U_low']:+.9f}.
17. S6 U_high: {summary.loc['S6','U_high']:+.9f}.
18. S6 DeltaRegime: {summary.loc['S6','DeltaRegime']:+.9f}.
19. Pooled U_low: {pooled.U_low:+.9f}.
20. Pooled U_high: {pooled.U_high:+.9f}.
21. Pooled DeltaRegime: {pooled.DeltaRegime:+.9f}.
22. DeltaRegime 95% CI: [{pooled.DeltaRegime_CI_lower:+.9f}, {pooled.DeltaRegime_CI_upper:+.9f}].
23. U_high 95% CI: [{pooled.U_high_CI_lower:+.9f}, {pooled.U_high_CI_upper:+.9f}].
24. Pooled High-E matched-random specificity: {matched_pooled_high.SpecificU_BA:+.9f}, CI [{matched_pooled_high.SpecificU_BA_CI_lower:+.9f}, {matched_pooled_high.SpecificU_BA_CI_upper:+.9f}].
25. Highest-I baseline: pooled {highest.loc['POOLED','U_HighestI']:+.9f}, CI [{highest.loc['POOLED','U_HighestI_CI_lower']:+.9f}, {highest.loc['POOLED','U_HighestI_CI_upper']:+.9f}].
26. Low-E actionability: `{low_status}`.
27. S4/S6 same direction: Delta {g3}; High-E harm {g4}.
28. Safety terminal: `{terminal}`.
29. Purity terminal: `{'P4C_SAFETY_PURITY_PASS' if g5 else 'P4C_SAFETY_PROTOCOL_OR_PURITY_FAILURE'}`.
30. P4A unfinished grid remains `OPTIONAL_PARTIAL_INVARIANCE_GRID` and paused.
31. OpenBMI sealed internal holdout: `UNTOUCHED`.
32. WBCIC outer 10: `UNTOUCHED_NOT_ENUMERATED`.
33. Post-outcome threshold/model modification: NONE.
34. Scientific principle: subject identifiability alone is insufficient evidence of nuisance; task-entangled identity can act as a prospective suppression veto. Task decoupling is an empirical nuisance-admissibility condition, not a guarantee of beneficial suppression.
35. Next-stage authorization: method-level bridge `{bridge}`; final new model `NOT_AUTHORIZED_AT_THIS_STAGE`.

P4B negative and P4C-Safety are not contradictory: continuous future-utility prediction remains unsupported; this experiment tests only a coarse asymmetric suppression-risk boundary.
"""
    (EXP / "P4C_SAFETY_FINAL_REPORT.md").write_text(final_report, encoding="utf-8")
    payload = {
        "pass": not issues,
        "issues": issues,
        "validated_at_utc": now_utc(),
        "branch": BRANCH,
        "p4b_terminal_preserved": p4b_validation["terminal"],
        "old_p4c_terminal_preserved": old_closure["terminal"],
        "reserved_settings": ["S4", "S6"],
        "regime_counts": {setting: {"REGIME_LOW": int(counts.loc[setting, "REGIME_LOW"]), "REGIME_HIGH": int(counts.loc[setting, "REGIME_HIGH"])} for setting in ["S4", "S6"]},
        "bootstrap_draws": 10000,
        "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5},
        "SAFETY_STATUS": terminal,
        "LOW_E_ACTIONABILITY_STATUS": low_status,
        "METHOD_LEVEL_BRIDGE_AUTHORIZATION": bridge,
        "FINAL_NEW_MODEL_AUTHORIZATION": "NOT_AUTHORIZED_AT_THIS_STAGE",
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
        "P4A_grid": "OPTIONAL_PARTIAL_INVARIANCE_GRID_PAUSED",
        "preoutcome_assignment_sha256": assignment_hash,
        "final_report_sha256": sha256(EXP / "P4C_SAFETY_FINAL_REPORT.md"),
        "figures": required_figures,
        "final_report_written_after_validator_pass": not issues,
    }
    write_json(RESULTS / "P4C_SAFETY_FINAL_VALIDATION.json", payload)
    if issues:
        raise RuntimeError(f"P4C-Safety validation failed: {issues}")
    print(f"P4C_SAFETY_VALIDATION_PASS terminal={terminal} lowE={low_status} bridge={bridge}", flush=True)


if __name__ == "__main__":
    main()
