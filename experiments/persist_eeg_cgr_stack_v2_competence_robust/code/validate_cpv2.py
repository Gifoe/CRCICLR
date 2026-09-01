"""Strict compact-artifact validation for CPV2."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
BANKS = EXP / "action_banks"
REQUIRED = [
    "R1_HANDOFF_AUDIT.json", "COMPETENCE_AUDIT.json", "CALIBRATION_PARAMETERS.csv",
    "FAMILY_STACK_WEIGHTS.csv", "COMPLEMENTARITY_METRICS.csv", "ROBUST_RECIPE_SEARCH.csv",
    "ROBUST_STACK_PER_SUBJECT.csv", "ROBUST_STACK_PER_FOLD.csv", "SUBJECT_HETEROGENEITY.csv",
    "INSTABILITY_BINS.csv", "CONSENSUS_ALPHA.csv", "BASELINE_COMPARISON.csv",
    "CONTROL_COMPARISON.csv", "ABLATION_SUMMARY.csv", "WBCIC_S2_ATCNET.csv",
    "WBCIC_S2_EEGNEX.csv", "STATISTICS.json", "VALIDATION.json",
]


def main() -> int:
    missing = [name for name in REQUIRED if not (RESULTS / name).is_file()]
    manifests = [BANKS / "OPENBMI_CPV2_MANIFEST.json", BANKS / "WBCIC_S0_S1_CPV2_MANIFEST.json"]
    missing += [str(path.relative_to(EXP)) for path in manifests if not path.is_file()]
    validation = json.loads((RESULTS / "VALIDATION.json").read_text(encoding="utf-8")) if not missing else {}
    ok = not missing and validation.get("pass") is True and validation.get("S2_accessed") is False and validation.get("outer_accessed") is False
    if not missing:
        for path in manifests:
            value = json.loads(path.read_text(encoding="utf-8"))
            ok = ok and value.get("six_predictions_per_sample_per_expert") is True and value.get("complete_case_filter") is False
    result = {"pass": bool(ok), "missing": missing, "validation": validation, "runtime_committed": False}
    (RESULTS / "VALIDATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
