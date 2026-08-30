"""Fail-closed validator for the compact V3 delivery."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as c


DOCS = ("README.md", "V2_FORENSIC_AUDIT.md", "CODE_MAP.md", "SCIENTIFIC_RATIONALE.md", "METHOD.md", "CROSS_FIT_AUDIT.md", "COVARIANCE_AUDIT.md", "BURES_MAP_AUDIT.md", "ANCHOR_EXCLUSION_AUDIT.md", "TARGET_AFFINITY_AUDIT.md", "RANDOM_AFFINE_AUDIT.md", "SOURCE_DEVELOPMENT_REPORT.md", "ATCNET_OFFICIAL_REPORT.md", "EEGNEX_CONFIRMATION_REPORT.md", "CONTROL_REPORT.md", "CLAIM_AUDIT.md", "ITERATION_LEDGER.md", "REPRODUCIBILITY.md", "FINAL_REPORT.md", "FINAL_REPORT.json")
RESULTS = ("V2_FORENSIC_METRICS.csv", "SOURCE_RECIPE_SEARCH.csv", "BURES_STATISTICS.csv", "CANDIDATE_VALIDITY.csv", "TARGET_AFFINITY.csv", "RANDOM_AFFINE_MATCHING.csv", "ATCNET_OFFICIAL_PER_SUBJECT.csv", "ATCNET_OFFICIAL_PER_FOLD.csv", "EEGNEX_PER_SUBJECT.csv", "EEGNEX_PER_FOLD.csv", "METHOD_SUMMARY.csv", "CONTROL_COMPARISON.csv", "STATISTICS.json", "VALIDATION.json", "SOURCE_GATE.json", "SOURCE_GATE_DETAIL.csv", "GEOMETRY_PER_SUBJECT.csv")
FIGURES = ("v2_self_neighbor_audit", "mean_vs_second_order_transport", "target_affinity", "displacement_vs_margin", "method_comparison", "subject_level_gain", "cross_architecture_gain")
TERMINALS = {"BURES_SCST_CROSS_ARCH_SUPPORTED", "BURES_SCST_DISCOVERY_ONLY_GENERALITY_NOT_SUPPORTED", "BURES_SCST_NOT_SUPPORTED_ON_OFFICIAL_ATCNET", "BURES_SCST_SOURCE_GATE_FAILED", "BURES_SCST_TRANSPORT_NOT_REALIZED", "BURES_SCST_IMPLEMENTATION_INVALID"}


def main() -> None:
    checks: dict[str, bool] = {}
    checks["documents"] = all((c.EXP / name).is_file() for name in DOCS)
    checks["results"] = all((c.RESULTS / name).is_file() for name in RESULTS)
    checks["figures"] = all((c.FIGURES / f"{name}.{suffix}").is_file() for name in FIGURES for suffix in ("png", "pdf"))
    report = c.read_json(c.EXP / "FINAL_REPORT.json") if (c.EXP / "FINAL_REPORT.json").is_file() else {}
    checks["terminal"] = report.get("terminal") in TERMINALS
    checks["v1_negative_preserved"] = report.get("immutable_v1_terminal") == "SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE"
    checks["v2_negative_preserved"] = report.get("immutable_v2_terminal") == "ME_HARD_SCST_NOT_SUPPORTED"
    checks["outer_unopened"] = report.get("outer_resource_status") == "NOT_OPENED" and report.get("s3_opened") is False
    gate = c.read_json(c.RESULTS / "SOURCE_GATE.json") if (c.RESULTS / "SOURCE_GATE.json").is_file() else {}
    checks["gate_terminal_consistent"] = (not gate.get("source_gate_pass", False)) or report.get("terminal") in {"BURES_SCST_CROSS_ARCH_SUPPORTED", "BURES_SCST_DISCOVERY_ONLY_GENERALITY_NOT_SUPPORTED", "BURES_SCST_NOT_SUPPORTED_ON_OFFICIAL_ATCNET"}
    lock_path = c.PROTOCOL / "BURES_SCST_V3_LOCK.json"
    if lock_path.is_file():
        lock = c.read_json(lock_path); files = [c.CODE / name for name in lock.get("code_files", [])]
        checks["protocol_lock_hash"] = bool(files) and all(path.is_file() for path in files) and c.code_tree_sha256(files) == lock.get("code_tree_sha256")
        checks["protocol_no_s3_before_lock"] = lock.get("s3_authorized") is False and lock.get("outer_or_sealed_opened") is False
    else:
        checks["protocol_lock_hash"] = False; checks["protocol_no_s3_before_lock"] = False
    matching_path = c.RESULTS / "RANDOM_AFFINE_MATCHING.csv"
    matching = pd.read_csv(matching_path) if matching_path.is_file() else pd.DataFrame()
    if len(matching):
        checks["matching_nonempty"] = True
        checks["euclidean_tolerance"] = "euclidean_norm_mismatch" in matching and float(matching.euclidean_norm_mismatch.abs().mean()) < 1e-5
        checks["whitened_tolerance"] = "whitened_norm_mismatch" in matching and float(matching.whitened_norm_mismatch.abs().mean()) < 1e-5
        paired = matching[matching.matched_pair.astype(bool)] if "matched_pair" in matching else matching
        checks["alpha_tolerance"] = len(paired) > 0 and "alpha_mismatch" in paired and float(paired.alpha_mismatch.dropna().abs().max()) < 1e-8
        checks["per_anchor_count_match"] = "structured_matched_count" in matching and "random_matched_count" in matching and bool((matching.structured_matched_count == matching.random_matched_count).all())
    else:
        checks["matching_nonempty"] = bool(report.get("terminal") in {"BURES_SCST_TRANSPORT_NOT_REALIZED", "BURES_SCST_SOURCE_GATE_FAILED"})
        checks["euclidean_tolerance"] = checks["matching_nonempty"]; checks["whitened_tolerance"] = checks["matching_nonempty"]; checks["alpha_tolerance"] = checks["matching_nonempty"]; checks["per_anchor_count_match"] = checks["matching_nonempty"]
    stats = c.read_json(c.RESULTS / "STATISTICS.json") if (c.RESULTS / "STATISTICS.json").is_file() else {}
    checks["source_grid_or_explicit_stop"] = bool(stats.get("source_grid_complete", False) or report.get("terminal") in {"BURES_SCST_TRANSPORT_NOT_REALIZED", "BURES_SCST_SOURCE_GATE_FAILED"})
    validation = {"schema": "BURES_SCST_V3_VALIDATION_V1", "pass": bool(all(checks.values())), "checks": checks, "terminal": report.get("terminal")}
    c.write_json(c.RESULTS / "VALIDATION.json", validation); print(json.dumps(validation, indent=2))
    if not validation["pass"]: raise RuntimeError("V3_VALIDATION_FAILED")


if __name__ == "__main__": main()
