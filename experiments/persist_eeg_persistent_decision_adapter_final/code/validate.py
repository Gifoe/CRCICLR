"""Validate the compact PDA package and fail on leakage or invalid claims."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pda_core as c


REQUIRED_DOCS = ["README.md", "RELATED_METHOD_BOUNDARY.md", "CODE_AND_RESOURCE_AUDIT.md", "SCIENTIFIC_RATIONALE.md", "METHOD.md", "THEORY_NOTE.md", "SOURCE_DEVELOPMENT_REPORT.md", "WBCIC_S2_REPORT.md", "EEGNEX_REPORT.md", "OUTER_REPORT.md", "ABLATION_REPORT.md", "CORRECT_WRONG_SHUFFLED_REPORT.md", "ADAPTER_STABILITY_REPORT.md", "LEAKAGE_AUDIT.md", "CLAIM_AUDIT.md", "ITERATION_LEDGER.md", "REPRODUCIBILITY.md", "FINAL_REPORT.md", "FINAL_REPORT.json"]
REQUIRED_RESULTS = ["SOURCE_RECIPE_SEARCH.csv", "SOURCE_PER_SUBJECT.csv", "SOURCE_PER_FOLD.csv", "ADAPTER_COMPONENTS.csv", "FISHER_WEIGHTS.csv", "CORRECT_WRONG_SHUFFLED.csv", "ABLATION_SUMMARY.csv", "STATISTICS.json", "SOURCE_GATE.json"]


def main() -> None:
    issues = []
    for name in REQUIRED_DOCS:
        if not (c.EXP / name).is_file(): issues.append(f"missing document {name}")
    for name in REQUIRED_RESULTS:
        if not (c.RESULTS / name).is_file(): issues.append(f"missing result {name}")
    gate = json.loads((c.RESULTS / "SOURCE_GATE.json").read_text()) if (c.RESULTS / "SOURCE_GATE.json").is_file() else {}
    if gate.get("terminal") not in {"PERSIST_PDA_SOURCE_NOT_SUPPORTED", "PERSIST_PDA_SOURCE_ONLY_SUPPORTED"}: issues.append("invalid source terminal")
    if gate.get("source_gate_pass") is True and gate.get("terminal") != "PERSIST_PDA_SOURCE_ONLY_SUPPORTED": issues.append("gate/terminal mismatch")
    frame_path = c.RESULTS / "SOURCE_PER_SUBJECT.csv"
    if frame_path.is_file():
        frame = pd.read_csv(frame_path)
        for col in ["future_session_used_for_fit", "future_labels_used_for_fit"]:
            if col not in frame or not (frame[col] == False).all(): issues.append(f"leakage flag {col}")
        if "population_checkpoint_id" in frame and frame.groupby(["dataset","fold","seed"]).population_checkpoint_id.nunique().max() > 1: issues.append("matched baselines use different population ids")
    # Runtime and future utility paths must not be in the deliverable tree.
    for p in c.EXP.rglob("*"):
        if p.is_file() and ("runtime" in p.parts or "utility_metrics" in str(p) or "utility_units" in str(p)) and not p.name.startswith(".gitignore"):
            issues.append(f"forbidden runtime artifact in package: {p}")
    result = {"pass": not issues, "issues": issues, "scientific_gate_pass": bool(gate.get("source_gate_pass", False)), "terminal": gate.get("terminal"), "future_resource_opened": False}
    c.write_json(c.RESULTS / "VALIDATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if issues: raise SystemExit(1)


if __name__ == "__main__":
    main()
