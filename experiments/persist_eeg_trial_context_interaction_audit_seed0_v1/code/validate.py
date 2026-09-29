"""Independent compact-output inventory and arithmetic checks."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"
EXPECTED = {"TRAIN_MODEL_SELECTION.csv": 250, "OUTER_MODEL_RESULTS.csv": 280,
            "MODEL_FAMILY_COMPARISON.csv": 35, "INTERACTION_TERM_ABLATION.csv": 40,
            "CONTEXT_BUDGET_CURVE.csv": 20, "WRONG_CONTEXT_NULL.csv": 1000,
            "TRAIN_CONTEXT_PAIRING_NULL.csv": 1000, "HISTORICAL_CONTEXT_CONTROL.csv": 40,
            "PARAMETER_MATCHED_CONTROLS.csv": 1505, "CONTEXT_ONLY_RESULTS.csv": 40,
            "INTERACTION_COMPATIBILITY.csv": 240, "CONTEXT_SENSITIVITY.csv": 40,
            "CONTEXT_CLASS_BALANCE_AUDIT.csv": 40, "SUBJECT_LEVEL_EFFECTS.csv": 40,
            "BOOTSTRAP_CONTRASTS.csv": 6}


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding="utf-8", newline="") as f: return list(csv.DictReader(f))


def main():
    inventory = sorted(x.name for x in OUT.iterdir() if x.is_file())
    assert inventory == sorted([*EXPECTED, "DECISION_SUMMARY.json", "FINAL_HELDOUT_EXCLUSION_AUDIT.json", "FINAL_REPORT.md"]), inventory
    for name, n in EXPECTED.items():
        data = rows(OUT / name)
        assert len(data) == n, (name, len(data), n)
    dec = json.loads((OUT / "DECISION_SUMMARY.json").read_text(encoding="utf-8"))
    held = json.loads((OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json").read_text(encoding="utf-8"))
    rep = json.loads((HERE / "protocol" / "REPRESENTATION_PROVENANCE.json").read_text(encoding="utf-8"))
    assert held["formal_final_heldout_eeg_reads"] == dec["formal_final_heldout_eeg_reads"] == rep["final_heldout_eeg_reads"] == 0
    assert dec["outer_historically_exposed"] and not held.get("outer_untouched", False)
    assert len(dec["fold_table"]) == 5 and all(len(rep["folds"][str(f)]["TRAIN_GEOMETRY"]["role_subjects"]) == 26 for f in range(5))
    assert all(len(rep["folds"][str(f)]["OUTER_DEVELOPMENT"]["role_subjects"]) == 8 for f in range(5))
    for role in ("TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"):
        assert all(rep["folds"][str(f)][role]["model_state_before"] == rep["folds"][str(f)][role]["model_state_after"] for f in range(5))
    selection = rows(OUT / "TRAIN_MODEL_SELECTION.csv")
    fold_rows = rows(OUT / "OUTER_MODEL_RESULTS.csv")
    effects = rows(OUT / "SUBJECT_LEVEL_EFFECTS.csv")
    assert all(sum(x["selected"] == "True" for x in selection if int(x["fold"]) == f and x["arm"] == arm) == 1 for f in range(5) for arm in {x["arm"] for x in selection})
    for fold in range(5):
        fr = [x for x in fold_rows if int(x["fold"]) == fold]
        assert len(fr) == 56
        outcome = {(x["subject"], x["outcome_count"]) for x in fr}
        assert len(outcome) == 8
        assert all(len({x["outcome_count"] for x in fr if x["subject"] == s}) == 1 for s, _ in outcome)
        row = dec["fold_table"][fold]
        for arm in ("TRIAL_ONLY", "RELATIVE_SUBTRACTION", "ADDITIVE_CONTEXT", "ADDITIVE_RELATIVE", "ELEMENTWISE_INTERACTION", "FULL_SIMPLE_INTERACTION", "LOWRANK_BILINEAR"):
            actual = np.mean([float(x["BA"]) for x in fr if x["arm"] == arm])
            assert abs(actual-row[arm+"_BA"]) < 1e-10, (fold, arm)
    contrast = np.mean([float(x["interaction_minus_noninteraction"]) for x in effects])
    assert abs(contrast-dec["contrasts"]["interaction_minus_noninteraction"]["point_BA"]) < 1e-10
    assert dec["next_action"] in ("PROCEED_CONTEXT_INTERACTION_BLOCK", "DO_NOT_BUILD_INTERACTION_ARCHITECTURE", "STOP_CONTEXT_INTERACTION_DIRECTION")
    report = (OUT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert all(f"Q{i}." in report for i in range(1,13)) and "## Final fold table" in report
    assert dec["next_action"] in report and "final-heldout EEG reads 0" in report
    hashes = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    print(json.dumps({"files": len(hashes), "csv": len(EXPECTED), "rows": EXPECTED,
                      "decision": dec["next_action"], "hashes": hashes}, sort_keys=True))


if __name__ == "__main__": main()
