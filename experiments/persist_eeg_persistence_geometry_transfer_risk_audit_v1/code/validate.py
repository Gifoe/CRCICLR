"""Compact post-run validator for the persistence-geometry audit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"


def main() -> None:
    required = [
        EXP / "SCIENTIFIC_CONTEXT_AUDIT.md",
        EXP / "PERSISTENCE_DEFINITION_PROVENANCE.md",
        EXP / "DATA_LEGALITY_AUDIT.json",
        EXP / "DATA_SUPPORT_LOCK.json",
        EXP / "PRE_OUTCOME_PROTOCOL_LOCK.json",
        EXP / "SUBJECT_INFERENCE_AUDIT.json",
        RESULTS / "A_ONLY_TRAINING_SUMMARY.csv",
        RESULTS / "PERSISTENCE_ASSIGNMENTS.csv",
        RESULTS / "SOURCE_GEOMETRY_AUDIT.csv",
        RESULTS / "SUBJECT_DESCRIPTOR_INDICES.csv",
        RESULTS / "SUBJECT_QUERY_INDICES.csv",
        RESULTS / "SUBJECT_GEOMETRY_DESCRIPTORS.csv",
        RESULTS / "SUBJECT_TRANSFER_DIFFICULTY.csv",
        RESULTS / "PRIMARY_TRANSFER_RISK.csv",
        RESULTS / "CONTROL_TRANSFER_RISK.csv",
        RESULTS / "MATCHED_NONPROTECTED_NULL.csv",
        RESULTS / "LOFO_ROBUSTNESS.csv",
        RESULTS / "ALTERNATIVE_EXPLANATION_AUDIT.csv",
        RESULTS / "GATE_SUMMARY.json",
        RESULTS / "FINAL_DECISION.json",
        RESULTS / "FINAL_REPORT.md",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    legality = json.loads((EXP / "DATA_LEGALITY_AUDIT.json").read_text(encoding="utf-8")) if not missing else {}
    lock = json.loads((EXP / "PRE_OUTCOME_PROTOCOL_LOCK.json").read_text(encoding="utf-8")) if (EXP / "PRE_OUTCOME_PROTOCOL_LOCK.json").is_file() else {}
    checks = {
        "required_files": not missing,
        "missing": missing,
        "seed_zero_only": legality.get("seed") == 0 and not legality.get("seed1_run") and not legality.get("seed2_run"),
        "eegnet_only": not legality.get("second_backbone_run", True),
        "outer_sealed_closed": not legality.get("WBCIC_outer_10_opened", True) and not legality.get("OpenBMI_outer_test_opened", True),
        "outcome_after_lock": bool(lock) and lock.get("outcome_used") is False,
        "no_pgeg_training": not legality.get("PGEG_training_started", True),
    }
    if checks["required_files"]:
        for name in ("PRIMARY_TRANSFER_RISK.csv", "CONTROL_TRANSFER_RISK.csv", "LOFO_ROBUSTNESS.csv"):
            checks[name + "_readable"] = not pd.read_csv(RESULTS / name).empty
    checks["pass"] = (not missing) and all(bool(v) for k, v in checks.items() if k not in {"missing", "pass"})
    (RESULTS / "VALIDATION.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    raise SystemExit(0 if checks["pass"] else 1)


if __name__ == "__main__":
    main()
