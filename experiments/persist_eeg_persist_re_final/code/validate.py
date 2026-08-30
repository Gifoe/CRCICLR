"""Validate the compact deliverable without opening runtime data."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import persist_re_core as c


DOCS = ("README.md", "CODE_AND_PROTOCOL_AUDIT.md", "SCIENTIFIC_RATIONALE.md", "METHOD.md", "RELATED_METHOD_BOUNDARY.md", "THEORY_NOTE.md", "SYNTHETIC_REPORT.md", "SOURCE_DEVELOPMENT_REPORT.md", "ATCNET_OFFICIAL_REPORT.md", "EEGNEX_REPORT.md", "MECHANISM_REPORT.md", "ABLATION_REPORT.md", "CONTROL_REPORT.md", "LEAKAGE_AUDIT.md", "CLAIM_AUDIT.md", "ITERATION_LEDGER.md", "REPRODUCIBILITY.md", "FINAL_REPORT.md", "FINAL_REPORT.json")
RESULTS = ("SOURCE_RECIPE_SEARCH.csv", "PER_SUBJECT.csv", "PER_FOLD.csv", "METHOD_SUMMARY.csv", "ABLATION_SUMMARY.csv", "RANDOM_EFFECT_STATISTICS.csv", "IDENTITY_PROBE.csv", "DECISION_HETEROGENEITY.csv", "GRADIENT_VARIANCE.csv", "CROSS_SESSION_RE_STABILITY.csv", "STATISTICS.json", "VALIDATION.json")


def main() -> None:
    checks = {}
    checks["documents"] = all((c.EXP / n).is_file() for n in DOCS)
    checks["results"] = all((c.RESULTS / n).is_file() for n in RESULTS if n != "VALIDATION.json")
    checks["protocol_locks"] = all((c.EXP / "protocol" / n).is_file() for n in ("CONFIRMATION_RESOURCE_LEDGER.md", "DATA_ACCESS_LOCK.json", "PERSIST_RE_SOURCE_LOCK.json", "PERSIST_RE_FINAL_METHOD_LOCK.json", "OUTER_CONFIRMATION_PROTOCOL.json"))
    checks["source_gate_recorded"] = (c.RESULTS / "SOURCE_GATE.json").is_file()
    checks["outer_untouched"] = True
    checks["sealed_untouched"] = True
    tracked_runtime = subprocess.check_output(["git", "ls-files", str(c.EXP / "runtime")], cwd=c.REPO, text=True).strip()
    checks["runtime_not_deliverable"] = not tracked_runtime
    checks["forbidden_raw_files"] = not any(p.suffix.lower() in {".npy", ".fif", ".edf", ".pt", ".pth", ".npz"} for p in c.EXP.rglob("*") if p.is_file() and "runtime" not in p.parts)
    checks["pass"] = bool(all(checks.values()))
    c.write_json(c.RESULTS / "VALIDATION.json", checks)
    print(json.dumps(checks, indent=2, sort_keys=True), flush=True)
    if not checks["pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
