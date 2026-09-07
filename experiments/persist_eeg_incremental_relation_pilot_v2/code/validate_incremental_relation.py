"""Independent compact-artifact validator; it never imports a data loader."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = ("OpenBMI", "WBCIC")
METHODS = ("SUBJECT_BALANCED_ERM", "GENERIC_RESIDUAL", "GENERIC_PROTOTYPE", "CROSS_SESSION_RELATION")

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def read(p: Path): return json.loads(p.read_text(encoding="utf-8-sig"))

def validate(pilot: Path, require_runtime: bool = False) -> dict:
    results = pilot / "results"
    amendment = read(pilot / "INCREMENTAL_RELATION_PROTOCOL_AMENDMENT.json")
    train = read(pilot / "INCREMENTAL_RELATION_TRAINING_LOCK.json")
    execution = read(pilot / "INCREMENTAL_RELATION_EXECUTION_LOCK.json")
    pre = read(pilot / "INCREMENTAL_RELATION_PRE_OUTCOME_LOCK.json")
    access = read(pilot / "INCREMENTAL_RELATION_OUTCOME_ACCESS_LOCK.json")
    legal = read(pilot / "DATA_LEGALITY_AUDIT.json")
    assert sha(pilot / "INCREMENTAL_RELATION_PROTOCOL_AMENDMENT.json") == train["amendment_sha256"] == pre["amendment_sha256"] == access["amendment_sha256"] == legal["amendment_sha256"]
    assert sha(pilot / "INCREMENTAL_RELATION_TRAINING_LOCK.json") == pre["training_lock_sha256"]
    assert sha(pilot / "INCREMENTAL_RELATION_PRE_OUTCOME_LOCK.json") == access["pre_outcome_lock_sha256"]
    assert train["outcome_labels_read"] is False and train["outcome_labels_read_before_lock"] is False
    assert pre["outcome_labels_read"] is False and pre["outcome_labels_read_before_lock"] is False
    assert access["outcome_labels_read_before_lock"] is False and access["outcome_labels_read_after_lock"] is True
    assert legal["outcome_labels_read_before_lock"] is False and legal["outcome_labels_read_after_lock"] is True
    for x in (train, pre, access, legal):
        assert x["WBCIC_outer_10_opened"] is False and x["OpenBMI_sealed_holdout_opened"] is False
    script = pilot / "code" / "run_incremental_relation_pilot.py"
    assert sha(script) == train["code_sha256"] == pre["code_sha256"]
    assert execution["scientific_definition_changed"] is False and execution["outcome_labels_read"] is False
    assert execution["device"] == "cuda:0" and execution["execution_mode"] == "sequential_single_gpu"
    assert pre["methods"] == list(METHODS) and pre["datasets"] == list(DATASETS) and pre["folds"] == [0] and pre["seed"] == 0
    frame = pd.read_csv(results / "INCREMENTAL_RELATION_OUTCOME_PER_SUBJECT.csv")
    summary = pd.read_csv(results / "INCREMENTAL_RELATION_PERFORMANCE_SUMMARY.csv")
    deltas = pd.read_csv(results / "INCREMENTAL_RELATION_SUBJECT_DELTAS.csv")
    assert set(frame.dataset) == set(DATASETS) and set(frame.method) == set(METHODS)
    assert set(frame.fold) == {0} and set(frame.seed) == {0}
    assert not frame.duplicated(["dataset", "subject_id", "method"]).any()
    assert np.isfinite(frame[["BA", "macro_F1"]].to_numpy()).all()
    clear, means, counts = {}, {}, {}
    for d in DATASETS:
        f = frame[frame.dataset == d]
        by = {m: f[f.method == m].set_index("subject_id").sort_index() for m in METHODS}
        sb, rel = by["SUBJECT_BALANCED_ERM"], by["CROSS_SESSION_RELATION"]
        assert sb.index.equals(rel.index)
        db = (rel.BA - sb.BA) * 100; df = (rel.macro_F1 - sb.macro_F1) * 100
        generic = max(by["GENERIC_RESIDUAL"].BA.mean(), by["GENERIC_PROTOTYPE"].BA.mean())
        generic_delta = (generic - sb.BA.mean()) * 100
        ds = deltas[deltas.dataset == d].set_index("subject_id").sort_index()
        np.testing.assert_allclose(db, ds.delta_relation_BA_pp, atol=1e-10, rtol=0)
        np.testing.assert_allclose(df, ds.delta_relation_macro_F1_pp, atol=1e-10, rtol=0)
        for m in METHODS:
            row = summary[(summary.dataset == d) & (summary.method == m)].iloc[0]
            assert np.isclose(by[m].BA.mean(), row.mean_subject_BA, atol=1e-12, rtol=0)
            assert np.isclose(by[m].macro_F1.mean(), row.mean_macro_F1, atol=1e-12, rtol=0)
        means[d] = float(db.mean())
        clear[d] = bool(db.mean() >= .5 and df.mean() >= 0 and (db >= 0).mean() >= .5 and db.mean() - generic_delta >= .5)
        counts[d] = {"positive": int((db > 1e-10).sum()), "tied": int((db.abs() <= 1e-10).sum()), "negative": int((db < -1e-10).sum()), "n": len(db)}
    terminal = "INCREMENTAL_RELATION_RESTORE_NEXT_STAGE" if all(clear.values()) else "INCREMENTAL_RELATION_STOP_GENERIC_CONTROL_EXPLAINS_GAIN" if all(means[d] >= .5 and not clear[d] for d in DATASETS) else "INCREMENTAL_RELATION_STOP_NO_CLEAR_GAIN"
    result = read(results / "INCREMENTAL_RELATION_RESULT.json")
    assert result["terminal"] == terminal and result["final_claim_authorized"] is False
    if require_runtime:
        for d in DATASETS:
            for name in ("source_frozen_features.npz", "source_specs.npz", "source_manifest.json"):
                p = pilot / "runtime" / f"{d}_fold0" / name
                assert p.is_file() and sha(p) == pre["artifact_sha256"][f"{d}/{name}"]
        # Feature files are hashed only; they are not loaded.
    return {"pass": True, "terminal": terminal, "subject_counts": counts, "runtime_hashes_verified": bool(require_runtime), "numeric_tables_recomputed": True, "raw_EEG_loaded": False, "outcome_labels_read_before_lock": False}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", type=Path, default=Path(__file__).resolve().parents[1]); ap.add_argument("--require-runtime", action="store_true"); ap.add_argument("--output", type=Path)
    args = ap.parse_args(); out = validate(args.pilot.resolve(), args.require_runtime)
    if args.output: args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
