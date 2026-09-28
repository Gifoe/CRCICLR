"""Read-only, independent structural validation of the compact delivery."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
PROTOCOL = ROOT / "protocol"
COUNTS = {
    "BOOTSTRAP_CONTRASTS": 6,
    "CONTEXT_BUDGET_CURVE": 40,
    "CONTEXT_CLASS_BALANCE_AUDIT": 40,
    "CONTEXT_REFERENCE_AUDIT": 680,
    "CONTEXT_SHRINKAGE_RESULTS": 20,
    "CROSS_SESSION_GEOMETRY": 680,
    "FULL_SESSION_LOO_DIAGNOSTIC": 40,
    "GLOBAL_CENTER_CONTROL": 20,
    "INFORMATION_RETENTION": 10,
    "LAYER_LOCALIZATION": 20,
    "OUTER_TRANSFER_RESULTS": 40,
    "PCA_REMOVAL_CONTROL": 20,
    "RANDOM_REFERENCE_NULL": 1000,
    "RELATIVE_Z_RESULTS": 20,
    "REPRESENTATION_VARIANCE_DECOMPOSITION": 40,
    "SUBJECT_LEVEL_EFFECTS": 400,
    "SUBJECT_SESSION_CLASS_PROBES": 10,
    "TRAIN_CONTEXT_SELECTION": 5,
    "TRIAL_ORDER_AUDIT": 13260,
    "WRONG_SUBJECT_REFERENCE_NULL": 1000,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    with (OUTPUT / f"{name}.csv").open(newline="", encoding="utf-8-sig") as stream:
        result = list(csv.DictReader(stream))
    assert len(result) == COUNTS[name], (name, len(result))
    return result


def main() -> None:
    expected = {f"{name}.csv" for name in COUNTS} | {
        "DECISION_SUMMARY.json", "FINAL_HELDOUT_EXCLUSION_AUDIT.json", "FINAL_REPORT.md"
    }
    actual = {p.name for p in OUTPUT.iterdir() if p.is_file()}
    assert actual == expected, (sorted(expected - actual), sorted(actual - expected))
    tables = {name: rows(name) for name in COUNTS}
    summary = json.loads((OUTPUT / "DECISION_SUMMARY.json").read_text(encoding="utf-8"))
    heldout = json.loads((OUTPUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json").read_text(encoding="utf-8"))
    source = json.loads((PROTOCOL / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    stage = json.loads((PROTOCOL / "REPRESENTATION_STAGE_AUDIT.json").read_text(encoding="utf-8"))
    assert summary["formal_final_heldout_eeg_reads"] == heldout["formal_final_heldout_eeg_reads"] == 0
    assert source["formal_final_heldout_eeg_reads"] == stage["formal_final_heldout_eeg_reads"] == 0
    assert summary["outer_selection"] is False
    assert stage["primary_stage"] == "EMBEDDING"
    assert source["protocol_lock_sha256"] == sha(PROTOCOL / "PROTOCOL_LOCK.json")
    assert source["protocol_amendment_sha256"] == sha(PROTOCOL / "PROTOCOL_AMENDMENT_V2.json")
    for local, deployed in (("run.py", "run_v2.py"), ("aggregate.py", "aggregate.py"),
                            ("geometry.py", "geometry_v2.py"), ("localization.py", "localization_v1.py"),
                            ("finalize.py", "finalize_v2.py")):
        assert source["analysis_code_sha256"][deployed] == sha(ROOT / "code" / local)
    assert sorted(source["folds"]) == [str(i) for i in range(5)]
    for fold in range(5):
        for role in ("TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"):
            meta = source["folds"][str(fold)][role]
            assert meta["model_state_before"] == meta["model_state_after"]
            assert len(meta["stage_files"]) == 8
    primary = tables["OUTER_TRANSFER_RESULTS"]
    def mean(arm: str) -> float:
        picked = [float(r["subject_equal_BA"]) for r in primary if r["arm"] == arm]
        assert len(picked) == 5
        return sum(picked) / 5
    assert abs(mean("ABSOLUTE") - summary["folds"]["absolute_BA"]) < 1e-12
    assert abs(mean("REL_MEAN") - summary["folds"]["relative_BA"]) < 1e-12
    assert abs(mean("REL_SHRINK") - summary["folds"]["shrink_BA"]) < 1e-12
    contrast = next(r for r in tables["BOOTSTRAP_CONTRASTS"] if r["contrast"] == "REL_MEAN_MINUS_ABSOLUTE")
    assert int(contrast["biological_subjects"]) == 40
    assert int(contrast["draws"]) == 20000
    assert abs(float(contrast["mean"]) - (mean("REL_MEAN") - mean("ABSOLUTE"))) < 1e-12
    geometry = [r for r in tables["CROSS_SESSION_GEOMETRY"]
                if r["stage"] == "EMBEDDING" and r["role"] == "OUTER_DEVELOPMENT"]
    assert len(geometry) == 40
    assert max(abs(float(r["class_relation_cosine_delta"])) for r in geometry) < 1e-7
    assert summary["gates"]["A"] is False and summary["gates"]["B"] is False
    assert summary["next_action"] == "STOP_ARCHITECTURE_DIRECTION"
    print(json.dumps({"files": len(expected), "csv_rows": sum(COUNTS.values()),
                      "summary_sha256": sha(OUTPUT / "DECISION_SUMMARY.json"),
                      "report_sha256": sha(OUTPUT / "FINAL_REPORT.md"),
                      "formal_final_heldout_eeg_reads": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
