"""Independent compact-output/provenance validator; never opens EEG or runtime caches."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXPECTED = {
    "GLOBAL_HEAD_AUDIT.csv": 340,
    "SUBJECT_DECISION_CORRECTIONS.csv": 130,
    "FUNCTION_CORRECTION_TARGETS.csv": 27040,
    "PERSONALIZATION_HEADROOM.csv": 40,
    "CORRECTION_SPECTRUM.csv": 130,
    "CONTEXT_DESCRIPTOR_AUDIT.csv": 20,
    "TRAIN_LOSO_CONTEXT_RESULTS.csv": 650,
    "OUTER_CONTEXT_RESULTS.csv": 360,
    "CONTEXT_CORRECTION_PREDICTION.csv": 40,
    "CONTEXT_BUDGET_CURVE.csv": 200,
    "CONTEXT_SHUFFLE_NULL.csv": 8000,
    "CONTEXT_BASELINE_COMPARISON.csv": 45,
    "PC_CONTEXT_ABLATION.csv": 265,
    "SUBJECT_LEVEL_EFFECTS.csv": 40,
    "BOOTSTRAP_CONTRASTS.csv": 4,
}
OTHER = {"FINAL_HELDOUT_EXCLUSION_AUDIT.json", "DECISION_SUMMARY.json", "FINAL_REPORT.md", "HASH_INDEX.json"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    actual = {p.name for p in OUT.iterdir() if p.is_file()}
    assert actual == set(EXPECTED) | OTHER, (actual - (set(EXPECTED) | OTHER), (set(EXPECTED) | OTHER) - actual)
    declared = json.loads((OUT / "HASH_INDEX.json").read_text())
    assert set(declared) == actual - {"HASH_INDEX.json"}
    assert all(sha(OUT / n) == value for n, value in declared.items())
    data = {n: rows(n) for n in EXPECTED}
    assert all(len(data[n]) == count for n, count in EXPECTED.items()), {n: len(data[n]) for n in EXPECTED}
    lock = json.loads((ROOT / "protocol/PROTOCOL_LOCK.json").read_text())
    analysis = json.loads((ROOT / "protocol/ANALYSIS_LOCK.json").read_text())
    source = json.loads((ROOT / "protocol/SOURCE_PROVENANCE.json").read_text())
    rep = json.loads((ROOT / "protocol/REPRESENTATION_PROVENANCE.json").read_text())
    summary = json.loads((OUT / "DECISION_SUMMARY.json").read_text())
    exclusion = json.loads((OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json").read_text())
    assert source["protocol_sha256"] == sha(ROOT / "protocol/PROTOCOL_LOCK.json")
    assert source["analysis_lock_sha256"] == sha(ROOT / "protocol/ANALYSIS_LOCK.json")
    for name, key in (("run.py", "run_code_sha256"), ("outer.py", "outer_code_sha256"), ("aggregate.py", "aggregate_code_sha256")):
        assert source[key] == sha(ROOT / "code" / name), name
    assert lock["formal_final_heldout_eeg_reads"] == summary["formal_final_heldout_eeg_reads"] == \
           rep["formal_final_heldout_eeg_reads"] == exclusion["formal_final_heldout_eeg_reads"] == 0
    assert analysis["locked_before_new_outer_access"] and rep["neural_weights_and_BN_buffers_unchanged"]
    assert len(rep["folds"]) == len(rep["outer_extraction"]) == 5
    for fold in range(5):
        proof = rep["folds"][fold]; outerproof = rep["outer_extraction"][fold]
        roles = proof["role_subjects"]
        assert len(roles["TRAIN_GEOMETRY"]) == 26 and len(roles["OUTER_DEVELOPMENT"]) == 8
        assert not set(roles["TRAIN_GEOMETRY"]) & set(roles["OUTER_DEVELOPMENT"])
        assert not set(roles["CHECKPOINT_VALIDATION"]) & set(roles["OUTER_DEVELOPMENT"])
        assert set(outerproof["outer_subjects"]) == set(roles["OUTER_DEVELOPMENT"])
        assert outerproof["model_state_sha256_before"] == outerproof["model_state_sha256_after"] == proof["model_state_sha256"]
        assert outerproof["checkpoint_sha256"] == proof["checkpoint_sha256"]
        assert outerproof["normalizer_sha256"] == proof["normalizer_sha256"]
        assert outerproof["split_sha256"] == proof["split_sha256"]
        assert outerproof["heldout_eeg_reads"] == 0
        out_rows = [r for r in data["OUTER_CONTEXT_RESULTS.csv"] if int(r["fold"]) == fold]
        assert {r["subject"] for r in out_rows} == set(roles["OUTER_DEVELOPMENT"])
        assert Counter((r["subject"], r["method"]) for r in out_rows) == Counter({(s, m): 1 for s in roles["OUTER_DEVELOPMENT"] for m in
            ("NATIVE_POPULATION_HEAD", "GLOBAL_REFIT_HEAD", "BEST_GLOBAL", "MEAN_CORRECTION", "NEAREST_CONTEXT",
             "CONTEXT_PREDICTED", "CTX_COMBINED_MATCHED", "CTX_COMBINED_PLUS_PC", "LABEL_ASSISTED_ORACLE_S1_HEAD")})
        assert all(r["label_assisted"] == ("True" if r["method"] == "LABEL_ASSISTED_ORACLE_S1_HEAD" else "False") for r in out_rows)
        budget = [r for r in data["CONTEXT_BUDGET_CURVE.csv"] if int(r["fold"]) == fold]
        assert Counter((r["subject"], r["budget"]) for r in budget) == Counter({(s, str(b)): 1 for s in roles["OUTER_DEVELOPMENT"] for b in lock["context_budgets_trials"]})
        null = [r for r in data["CONTEXT_SHUFFLE_NULL.csv"] if int(r["fold"]) == fold]
        assert Counter((r["subject"], r["draw"]) for r in null) == Counter({(s, str(d)): 1 for s in roles["OUTER_DEVELOPMENT"] for d in range(200)})
        train_rows = [r for r in data["TRAIN_LOSO_CONTEXT_RESULTS.csv"] if int(r["fold"]) == fold]
        assert {r["subject"] for r in train_rows} == set(roles["TRAIN_GEOMETRY"])
    out = data["OUTER_CONTEXT_RESULTS.csv"]
    grouped = defaultdict(list)
    for r in out: grouped[r["method"]].append(float(r["BA"]))
    oracle = sum(grouped["LABEL_ASSISTED_ORACLE_S1_HEAD"]) / 40
    glob = sum(grouped["BEST_GLOBAL"]) / 40
    context = sum(grouped["CONTEXT_PREDICTED"]) / 40
    mean = sum(grouped["MEAN_CORRECTION"]) / 40
    assert abs((oracle - glob) - summary["pooled_oracle_headroom_BA"]) < 1e-12
    assert abs((context - glob) - summary["pooled_context_gain_BA"]) < 1e-12
    assert abs((context - mean) - summary["pooled_context_minus_mean_BA"]) < 1e-12
    assert summary["primary_interpretation"] == "NO_MEANINGFUL_PERSONALIZATION_HEADROOM"
    assert summary["exact_next_action"] == "STOP_CONTEXT_MODEL_DIRECTION"
    report = (OUT / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert all(f"Q{i}." in report for i in range(1, 11))
    assert "Formal final-heldout EEG reads: 0" in report
    assert "STOP_CONTEXT_MODEL_DIRECTION" in report
    print("VALIDATED", len(EXPECTED), "CSVs;", len(rep["folds"]), "folds;", len(data["CONTEXT_SHUFFLE_NULL.csv"]),
          "null rows; oracle-global", round(oracle - glob, 6), "context-global", round(context - glob, 6))


if __name__ == "__main__": main()
