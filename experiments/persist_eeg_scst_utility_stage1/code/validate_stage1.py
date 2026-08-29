"""Strict compact-artifact validator for SCST utility Stage-1."""
from __future__ import annotations

import json

import pandas as pd

import stage1_common as c


def main() -> None:
    errors = []
    docs = ["README.md", "PROTOCOL.md", "STAGE1_CRITERION_REVISION.md", "IMPLEMENTATION_AUDIT.md", "ATCNET_REPORT.md", "ATCNET_OFFICIAL_REPORT.md", "EEGNEX_REPORT.md", "SCST_TRAINING_REPORT.md", "CONTROL_REPORT.md", "MANIFOLD_UTILITY_ANALYSIS.md", "CLAIM_AUDIT.md", "STAGE1_ITERATION_LEDGER.md", "REPRODUCIBILITY.md", "FINAL_REPORT.md", "FINAL_REPORT.json"]
    protocols = ["DATA_ACCESS_LOCK.json", "SCST_STAGE1_TRAINING_LOCK.json"]
    results = ["MODEL_COMPETENCE.csv", "STAGE1_ADMISSIBILITY.csv", "SCST_PER_SUBJECT.csv", "SCST_PER_FOLD.csv", "SCST_SUMMARY.csv", "CONTROL_COMPARISON.csv", "MANIFOLD_UTILITY.csv", "STATISTICS.json", "FINAL_REPORT.json"]
    figures = ["task_vs_manifold", "method_comparison", "subject_gain", "manifold_vs_scst_gain", "cross_architecture_summary"]
    for name in docs:
        if not (c.EXP / name).is_file(): errors.append(f"missing_doc:{name}")
    for name in protocols:
        if not (c.PROTOCOL / name).is_file(): errors.append(f"missing_protocol:{name}")
    for name in results:
        if not (c.RESULTS / name).is_file(): errors.append(f"missing_result:{name}")
    for stem in figures:
        for suffix in (".png", ".pdf"):
            if not (c.FIGURES / f"{stem}{suffix}").is_file(): errors.append(f"missing_figure:{stem}{suffix}")
    final_path = c.EXP / "FINAL_REPORT.json"
    allowed = {"SCST_CROSS_ARCHITECTURE_SUPPORTED", "SCST_ATCNET_SPECIFIC_SUPPORTED", "SCST_UTILITY_SUPPORTED_BUT_ADMISSIBILITY_RELATION_WEAK", "SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE", "SCST_GENERALITY_NOT_SUPPORTED"}
    if final_path.is_file():
        final = c.read_json(final_path)
        if final.get("terminal") not in allowed: errors.append("invalid_terminal")
        if final.get("outer_resources_untouched") is not True: errors.append("outer_purity_failed")
        if len(final.get("answers", {})) != 33: errors.append("final_answers_not_33")
    lock_path = c.PROTOCOL / "SCST_STAGE1_TRAINING_LOCK.json"
    if lock_path.is_file():
        lock = c.read_json(lock_path)
        if lock.get("future_utility_accessed_before_lock") is not False: errors.append("lock_not_prospective")
        for relative, digest in lock.get("source_artifact_hashes", {}).items():
            path = c.EXP / relative
            if not path.is_file() or c.sha256(path) != digest: errors.append(f"source_hash_mismatch:{relative}")
    per_path = c.RESULTS / "SCST_PER_SUBJECT.csv"
    if per_path.is_file():
        frame = pd.read_csv(per_path)
        if set(frame.future_session.astype(int)) != {2}: errors.append("unexpected_future_session")
        if set(frame.method) != {"ERM", "Mixup", "RandomTransport", "SCST-NoConsistency", "Full-SCST"}: errors.append("method_grid_incomplete")
    value = {"schema": "SCST_UTILITY_STAGE1_VALIDATION_V1", "pass": not errors, "errors": errors, "outer_resources_untouched": not any("outer" in item for item in errors)}
    c.write_json(c.RESULTS / "VALIDATION.json", value)
    print(json.dumps(value, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__": main()
