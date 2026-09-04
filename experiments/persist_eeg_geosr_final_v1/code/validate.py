"""Compact protocol/result validator for the GeoSR final experiment."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"


def main() -> None:
    required = [
        "README.md", "FROZEN_PROTOCOL.md", "FROZEN_PROTOCOL.json", "DATA_LEGALITY_AUDIT.json",
        "BUG_REPAIR_LEDGER.md", "PRE_OUTCOME_GEOSR_PROTOCOL_LOCK.json",
        "results/CROSS_FIT_ASSIGNMENTS.csv", "results/CROSSFIT_TEACHER_AUDIT.csv",
        "results/SOURCE_GEOMETRY_RISK.csv", "results/SOURCE_WEIGHT_AUDIT.csv",
        "results/INITIAL_STATE_HASHES.json", "results/TRAINING_SUMMARY.csv",
        "results/OUTCOME_PER_FOLD.csv", "results/OUTCOME_PER_SUBJECT.csv",
        "results/PERFORMANCE_SUMMARY.csv", "results/TAIL_ROBUSTNESS.csv",
        "results/CONTROL_COMPARISON.csv", "results/PAIRED_BOOTSTRAP.json",
        "results/FINAL_DECISION.json", "results/FINAL_REPORT.md", "results/VALIDATION.json",
    ]
    missing = [p for p in required if not (EXP / p).is_file()]
    if missing:
        raise SystemExit(f"missing artifacts: {missing}")
    legality = json.loads((EXP / "DATA_LEGALITY_AUDIT.json").read_text(encoding="utf-8"))
    lock = json.loads((EXP / "PRE_OUTCOME_GEOSR_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    decision = json.loads((RESULTS / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    if int(lock.get("seed", -1)) != 0:
        raise SystemExit("final validation expects seed-0 lock")
    if lock.get("backbone") != "EEGNet" or lock.get("WBCIC_outer_10_opened") is not False:
        raise SystemExit("scope lock failure")
    if legality.get("WBCIC_outer_10_opened") is not False or legality.get("OpenBMI_outer_test_opened") is not False:
        raise SystemExit("outer access failure")
    if legality.get("outcome_labels_read_before_lock") is not False:
        raise SystemExit("outcome labels preceded lock")
    assignments = pd.read_csv(RESULTS / "CROSS_FIT_ASSIGNMENTS.csv")
    for (dataset, fold, stage), frame in assignments.groupby(["dataset", "fold", "stage"]):
        held = []
        for text in frame.held_out_subjects.astype(str):
            held.extend([x for x in text.split(",") if x])
        if len(held) != len(set(held)):
            raise SystemExit(f"cross-fit held-out duplication: {dataset} fold {fold} {stage}")
    weights = pd.read_csv(RESULTS / "SOURCE_WEIGHT_AUDIT.csv")
    for col in ["weight_GEOSR", "weight_RANDOM_RANK", "weight_LOSS_HARD", "weight_GEO_ONLY"]:
        if not ((weights[col] >= 0.5 - 1e-8) & (weights[col] <= 1.5 + 1e-8)).all():
            raise SystemExit(f"subject weight range failure: {col}")
    val = json.loads((RESULTS / "VALIDATION.json").read_text(encoding="utf-8"))
    if val.get("pass") is not True or decision.get("scientific_rescue_performed") is not False:
        raise SystemExit("validation status failure")
    print(json.dumps({"pass": True, "terminal": decision.get("terminal"), "gates": decision.get("gates")}, indent=2))


if __name__ == "__main__":
    main()
