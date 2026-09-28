"""Independent checks of the reviewed compact delivery; never opens EEG data."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PROTOCOL = ROOT / "protocol"
EXPECTED_OUTPUTS = {
    "ADAPTATION_RESPONSE_SURFACE.csv": 1500,
    "DIFFERENTIAL_NECESSITY.csv": 15,
    "PC_ADAPTATION_TOLERANCE.csv": 1200,
    "PC_ROLLBACK_INTERVENTION.csv": 1200,
    "PC_DECISION_LEVERAGE.csv": 120,
    "PARTITION_SPECIFICITY_RESULTS.csv": 75,
    "RANDOM_PARTITION_DISTRIBUTION.csv": 1500,
    "OUTER_DEPLOYABLE_RESULTS.csv": 150,
    "UPDATE_NORM_MATCHED_CONTROLS.csv": 120,
    "CONTEXT_BUDGET_SENSITIVITY.csv": 480,
    "SUBJECT_LEVEL_EFFECTS.csv": 120,
    "BOOTSTRAP_CONTRASTS.csv": 18,
    "CROSS_TASK_REPLICATION.csv": 5,
    "CROSS_BACKBONE_REPLICATION.csv": 5,
}
EXPECTED_JSON = {"BASE_ADAPTATION_OPERATOR.json", "FINAL_HELDOUT_EXCLUSION_AUDIT.json",
                 "DECISION_SUMMARY.json"}
EXPECTED_PROTOCOL = {"PROTOCOL_LOCK.json", "ANALYSIS_LOCK.json", "SOURCE_PROVENANCE.json",
                     "REPRESENTATION_PROVENANCE.json", "PC_GEOMETRY_AUDIT.json"}
CELLS = {("EEGNet", "OpenBMI_MI", f) for f in range(5)} | {
    ("EEGNet", "OpenBMI_ERP", f) for f in range(5)} | {
    ("EEGConformer", "OpenBMI_MI", f) for f in range(5)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    assert len(result) == EXPECTED_OUTPUTS[name], (name, len(result))
    return result


def cell(row: dict) -> tuple[str, str, int]:
    key = row["backbone"], row["task"], int(row["fold"])
    assert key in CELLS, key
    return key


def close(a: float, b: float, tol: float = 1e-10) -> None:
    assert abs(float(a) - float(b)) <= tol, (a, b)


def main() -> None:
    assert {p.name for p in OUT.iterdir() if p.is_file()} == set(EXPECTED_OUTPUTS) | EXPECTED_JSON | {"FINAL_REPORT.md"}
    assert {p.name for p in PROTOCOL.iterdir() if p.is_file()} == EXPECTED_PROTOCOL
    tables = {name: rows(name) for name in EXPECTED_OUTPUTS}
    for name, table in tables.items():
        if name != "BOOTSTRAP_CONTRASTS.csv":
            assert {cell(row) for row in table} == CELLS if name not in {
                "CROSS_TASK_REPLICATION.csv", "CROSS_BACKBONE_REPLICATION.csv"} else len({cell(row) for row in table}) == 5
    necessity = {cell(r): r for r in tables["DIFFERENTIAL_NECESSITY.csv"]}
    assert len(necessity) == 15
    deployable = defaultdict(dict)
    for row in tables["OUTER_DEPLOYABLE_RESULTS.csv"]:
        key = cell(row)
        assert row["method"] not in deployable[key]
        deployable[key][row["method"]] = row
    assert all(len(methods) == 10 for methods in deployable.values())
    for key, row in necessity.items():
        methods = deployable[key]
        close(float(row["OUTER_delta_BA"]), float(methods["DIFFERENTIAL_P_C"]["BA"]) -
              float(methods["BEST_UNIFORM"]["BA"]))
        close(float(row["distance_from_diagonal"]), abs(float(row["alpha_P"]) - float(row["alpha_C"])))
        assert row["status"] == "COMPLETE"
    subject = tables["SUBJECT_LEVEL_EFFECTS.csv"]
    assert len({(*cell(r), r["subject"]) for r in subject}) == 120
    for row in subject:
        close(row["delta_vs_BestUniform"], float(row["Differential_BA"]) - float(row["BestUniform_BA"]))
        close(row["delta_vs_NoAdapt"], float(row["Differential_BA"]) - float(row["NoAdapt_BA"]))
    for row in tables["BOOTSTRAP_CONTRASTS.csv"]:
        assert int(row["bootstrap_draws"]) == 20000
        assert int(row["biological_subjects"]) >= 8
        assert float(row["CI95_low"]) <= float(row["CI95_high"])
        if row["contrast"] == "Differential_minus_BestUniform":
            relevant = [r for r in subject if (r["backbone"], r["task"]) == (row["backbone"], row["task"])]
            by_subject = defaultdict(list)
            for r in relevant: by_subject[r["subject"]].append(float(r["delta_vs_BestUniform"]))
            pooled = sum(sum(v) / len(v) for v in by_subject.values()) / len(by_subject)
            close(row["mean_delta_BA"], pooled)
    summary = json.loads((OUT / "DECISION_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["formal_final_heldout_eeg_reads"] == 0
    assert summary["primary_interpretation"] in {
        "PERSISTENCE_GUIDED_DIFFERENTIAL_ADAPTATION_SUPPORTED",
        "DIFFERENTIAL_ADAPTATION_SUPPORTED_BUT_NOT_PERSIST_SPECIFIC",
        "OFF_DIAGONAL_ORACLE_EXISTS_BUT_DOES_NOT_TRANSFER",
        "UNIFORM_SHRINKAGE_EXPLAINS_THE_GAIN", "PC_DIFFERENTIAL_ADAPTATION_NOT_SUPPORTED"}
    assert summary["protocol_lock_sha256"] == digest(PROTOCOL / "PROTOCOL_LOCK.json")
    assert summary["analysis_lock_sha256"] == digest(PROTOCOL / "ANALYSIS_LOCK.json")
    for name in EXPECTED_JSON | {"REPRESENTATION_PROVENANCE.json", "PC_GEOMETRY_AUDIT.json", "SOURCE_PROVENANCE.json"}:
        path = (OUT if name in EXPECTED_JSON else PROTOCOL) / name
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["formal_final_heldout_eeg_reads"] == 0
    source = json.loads((PROTOCOL / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    representation = json.loads((PROTOCOL / "REPRESENTATION_PROVENANCE.json").read_text(encoding="utf-8"))
    geometry = json.loads((PROTOCOL / "PC_GEOMETRY_AUDIT.json").read_text(encoding="utf-8"))
    for remote_name, local_name in {"run_prepare_v1.py": "run.py", "analysis_select_v1.py": "analysis.py",
                                    "outer_v2.py": "outer.py", "aggregate_v1.py": "aggregate.py"}.items():
        assert source["experiment_code_sha256"][remote_name] == digest(ROOT / "code" / local_name)
    assert len(source["cells"]) == len(representation["cells"]) == len(geometry["cells"]) == 15
    for item in source["cells"]:
        assert item["train_proof"]["model_state_sha256_before"] == item["train_proof"]["model_state_sha256_after"]
        assert item["outer_proof"]["model_state_sha256_before"] == item["outer_proof"]["model_state_sha256_after"]
        assert item["outer_proof"]["checkpoint_sha256"] == item["train_proof"]["checkpoint_sha256"]
    report = (OUT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert report.count("| EEGNet | OpenBMI_MI |") == 5
    assert report.count("| EEGNet | OpenBMI_ERP |") == 5
    assert report.count("| EEGConformer | OpenBMI_MI |") == 5
    assert summary["primary_interpretation"] in report and summary["exact_next_action"] in report
    print("COMPACT_VALIDATION_PASS", "15 cells", "18 outputs", "3 provenance audits")


if __name__ == "__main__": main()
