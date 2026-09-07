from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC")
EXP = REPO / "experiments" / "persist_eeg_scst_competence_generality_v1"
RESULTS = EXP / "results"; PROTOCOL = EXP / "protocol"; FIGURES = EXP / "figures"


def main() -> None:
    errors = []
    required_docs = ["README.md", "PROTOCOL.md", "REPOSITORY_AUDIT.md", "DATA_AUDIT.md", "CBRAMOD_COMPETENCE_REPORT.md", "SPECIALIST_SCREEN_REPORT.md", "SCST_ADMISSIBILITY_REPORT.md", "SCST_TRAINING_REPORT.md", "INVALID_SPACE_CONTROL.md", "GENERALITY_ANALYSIS.md", "CLAIM_AUDIT.md", "REPRODUCIBILITY.md", "FINAL_REPORT.md", "FINAL_REPORT.json", "COMPETENCE_ITERATION_LEDGER.md", "SPECIALIST_TRAINING_LEDGER.md"]
    required_protocol = ["DATA_ACCESS_LOCK.json", "CBRAMOD_GEOMETRY_PRESERVATION_LOCK.json", "COMPETENCE_PROTOCOL_LOCK.json", "SPECIALIST_SCREEN_LOCK.json", "SCST_TRAINING_PROTOCOL_LOCK.json"]
    required_results = ["TASK_COMPETENCE.csv", "SCST_VALIDITY_PER_MODEL.csv", "SCST_VALIDITY_PER_FOLD.csv", "SPECIALIST_SCREEN.csv", "SCST_TRAINING_PER_SUBJECT.csv", "SCST_TRAINING_SUMMARY.csv", "SCST_CONTROL_COMPARISON.csv", "ADMISSIBILITY_UTILITY_RELATION.csv", "STATISTICAL_TESTS.json", "FINAL_REPORT.json"]
    required_figures = [f"{stem}.{suffix}" for stem in ("competence_vs_manifold", "admissibility_gates", "admissibility_vs_gain", "method_comparison", "subject_level_gain", "generality_summary") for suffix in ("png", "pdf")]
    for path in [*(EXP / value for value in required_docs), *(PROTOCOL / value for value in required_protocol), *(RESULTS / value for value in required_results), *(FIGURES / value for value in required_figures)]:
        if not path.is_file() or path.stat().st_size == 0: errors.append(f"missing_or_empty:{path.relative_to(EXP)}")
    final_path = RESULTS / "FINAL_REPORT.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8")); allowed = {"SCST_GENERAL_METHOD_SUPPORTED", "SCST_REPRESENTATION_DEPENDENT_SUPPORTED", "ADMISSIBILITY_SUPPORTED_BUT_SCST_UTILITY_NOT_SUPPORTED", "NO_ADMISSIBLE_COMPETENT_REPRESENTATION_FOUND", "SCST_GENERALITY_HYPOTHESIS_NOT_SUPPORTED"}
        if final.get("overall_terminal") not in allowed: errors.append("invalid_terminal")
        if final.get("sealed_resources_untouched") is not True: errors.append("sealed_resource_purity_failed")
        answers = final.get("answers", {})
        if len(answers) != 34: errors.append(f"final_answers_count:{len(answers)}")
    authorization_path = RESULTS / "SCST_AUTHORIZATION.json"
    if authorization_path.is_file():
        authorization = json.loads(authorization_path.read_text(encoding="utf-8")); terminals = pd.read_csv(RESULTS / "SCST_MODEL_TERMINALS.csv")
        if authorization.get("level1_SCST_DISCOVERY_TRAINING_AUTHORIZED") is False and terminals.admissible_both_datasets.fillna(False).any(): errors.append("level1_closed_despite_eligible_model")
    data_lock = json.loads((PROTOCOL / "DATA_ACCESS_LOCK.json").read_text(encoding="utf-8")) if (PROTOCOL / "DATA_ACCESS_LOCK.json").is_file() else {}
    if data_lock.get("sealed_resources_accessed") is not False: errors.append("data_lock_not_pure")
    validation = {"schema": "SCST_COMPETENCE_GENERALITY_VALIDATION_V1", "pass": not errors, "errors": errors, "overall_terminal": json.loads(final_path.read_text(encoding="utf-8")).get("overall_terminal") if final_path.is_file() else None, "sealed_resources_untouched": not any("sealed" in value or "data_lock" in value for value in errors), "required_docs": len(required_docs), "required_protocol_files": len(required_protocol), "required_result_files": len(required_results), "required_figures": len(required_figures)}
    (RESULTS / "VALIDATION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
