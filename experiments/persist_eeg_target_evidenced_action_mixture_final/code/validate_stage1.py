"""Validate TEA-EEG artifact integrity without changing scientific results."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
PROTOCOL = EXP / "protocol"

REQUIRED_DOCS = [
    "README.md", "PREVIOUS_POLICY_REPRODUCTION.md", "SCIENTIFIC_RATIONALE.md", "METHOD.md",
    "ACTION_BANK_AUDIT.md", "ORACLE_HEADROOM_REPORT.md", "SOURCE_DEVELOPMENT_REPORT.md",
    "WBCIC_S2_REPORT.md", "CROSS_BACKBONE_REPORT.md", "TARGET_CONTEXT_REPORT.md", "SAFETY_REPORT.md",
    "CONTROL_REPORT.md", "ABLATION_REPORT.md", "LEAKAGE_AUDIT.md", "CLAIM_AUDIT.md",
    "ITERATION_LEDGER.md", "REPRODUCIBILITY.md", "FINAL_REPORT.md", "FINAL_REPORT.json",
]
REQUIRED_RESULTS = [
    "PREVIOUS_POLICY_REPRODUCTION.json", "ACTION_BANK_ORACLE.csv", "SOURCE_RECIPE_SEARCH.csv",
    "SOURCE_PER_SUBJECT.csv", "SOURCE_PER_FOLD.csv", "TARGET_CONTEXT_FEATURES.csv", "ACTION_WEIGHTS.csv",
    "REGRET_CALIBRATION.csv", "SAFETY_METRICS.csv", "CONTROL_COMPARISON.csv", "WBCIC_S2_ATCNET.csv",
    "WBCIC_S2_EEGNEX.csv", "STATISTICS.json",
]
REQUIRED_LOCKS = ["RESOURCE_LEDGER.md", "DATA_ACCESS_LOCK.json", "TEA_SOURCE_LOCK.json", "TEA_FINAL_METHOD_LOCK.json", "TEA_OUTER_CONFIRMATION_LOCK.json"]
ALLOWED_TERMINALS = {
    "TEA_CROSS_BACKBONE_SUPPORTED", "TEA_SINGLE_BACKBONE_ONLY", "TEA_SOURCE_ONLY_SUPPORTED",
    "TEA_WBCIC_S2_NOT_SUPPORTED", "TEA_SOURCE_NOT_SUPPORTED", "TEA_ACTION_BANK_HEADROOM_INSUFFICIENT",
    "TEA_PREVIOUS_RESULT_NOT_REPRODUCIBLE", "TEA_IMPLEMENTATION_INVALID",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: dict[str, Any] = {}
    missing = [p for p in REQUIRED_DOCS if not (EXP / p).exists()]
    missing += [f"results/{p}" for p in REQUIRED_RESULTS if not (RESULTS / p).exists()]
    missing += [f"protocol/{p}" for p in REQUIRED_LOCKS if not (PROTOCOL / p).exists()]
    checks["required_files"] = {"pass": not missing, "missing": missing}
    errors: list[str] = []
    stats = {}
    final = {}
    repro = {}
    try:
        stats = _json(RESULTS / "STATISTICS.json")
        final = _json(EXP / "FINAL_REPORT.json")
        repro = _json(RESULTS / "PREVIOUS_POLICY_REPRODUCTION.json")
    except Exception as exc:
        errors.append(f"json_load: {exc}")
    terminal = stats.get("terminal")
    if terminal not in ALLOWED_TERMINALS:
        errors.append(f"invalid terminal: {terminal}")
    if repro.get("status") != "PREVIOUS_POLICY_REPRODUCTION_PASS":
        errors.append("previous policy reproduction did not pass")
    if final.get("OUTER_TEST_USED", True) or stats.get("OUTER_TEST_USED", True):
        errors.append("outer resource marked as used")
    s2_tables = []
    for name in ("WBCIC_S2_ATCNET.csv", "WBCIC_S2_EEGNEX.csv"):
        path = RESULTS / name
        if path.exists():
            frame = pd.read_csv(path)
            s2_tables.append({"name": name, "rows": len(frame), "opened": bool(frame.get("OUTER_TEST_USED", pd.Series(False)).astype(bool).any())})
            if bool(frame.get("OUTER_TEST_USED", pd.Series(False)).astype(bool).any()):
                errors.append(f"{name} marks outer use")
    checks["s2_tables"] = s2_tables
    context = RESULTS / "TARGET_CONTEXT_FEATURES.csv"
    if context.exists():
        c = pd.read_csv(context)
        bad = [col for col in c.columns if re.search(r"(^|_)(label|effect|dce|subject_id|fold|seed)(_|$)", col, flags=re.I)]
        checks["context_columns"] = {"pass": not bad, "forbidden": bad, "rows": len(c)}
        if bad:
            errors.append(f"forbidden context columns: {bad}")
    else:
        checks["context_columns"] = {"pass": False}
    checks["source_gate"] = stats.get("source_gate", {})
    checks["scientific_terminal"] = terminal
    checks["pass"] = not missing and not errors
    checks["errors"] = errors
    out = RESULTS / "VALIDATION.json"
    out.write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
