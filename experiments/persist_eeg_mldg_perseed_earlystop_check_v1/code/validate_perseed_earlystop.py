"""Independent validator for the per-seed early-stop MLDG check."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(sys.argv[1]).resolve()
SCHEMA = "PERSIST_EEG_MLDG_PERSEED_EARLYSTOP_CHECK_V1"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: dict[str, bool] = {}
    lock = load_json(ROOT / "PROTOCOL_LOCK.json")
    split = load_json(ROOT / "SPLIT_REUSE_AUDIT.json")
    gate = load_json(ROOT / "PER_SEED_GATE.json")
    sel = pd.read_csv(ROOT / "PER_SEED_SELECTION.csv")
    res = pd.read_csv(ROOT / "PAIRED_RESULTS.csv")
    init = pd.read_csv(ROOT / "INITIAL_STATE_HASH.csv")
    traj = pd.read_csv(ROOT / "MLDG_VALIDATION_TRAJECTORY.csv")
    checks["schema"] = lock.get("schema") == SCHEMA and gate.get("schema") == SCHEMA and split.get("schema") == SCHEMA
    checks["split_reuse"] = bool(split.get("pass")) and bool(split["rows"][0].get("matches_parent_subject_lists"))
    row = split["rows"][0]
    checks["held_not_train_or_validation"] = bool(row["held_disjoint_train"] and row["held_disjoint_validation"] and row["validation_disjoint_selection"])
    checks["protocol_scope"] = bool(lock.get("dataset") == "WBCIC" and int(lock.get("outer_fold")) == 0 and lock.get("backbone") == "EEGNet" and lock.get("methods") == ["B0_SUBJECT_BALANCED_ERM", "B2_SUBJECT_EPISODIC_MLDG"] and lock.get("opt_seeds") == [0, 1, 2] and float(lock.get("beta")) == 1.0)
    checks["parent_order_and_rng_reused"] = bool(lock.get("order_seed") == "sha256(mldg-robustness|WBCIC|fold0|opt_seed|epoch)" and lock.get("base_rng_seed") == "sha256(mldg-robustness|WBCIC|fold0|opt_seed|train)" and lock.get("mldg_meta_seed") == "sha256(mldg-perseed-es-meta|WBCIC|fold0|opt_seed|epoch)")
    es = lock.get("early_stopping", {})
    checks["early_stop_frozen"] = bool(es.get("MAX_EPOCHS") == 60 and es.get("MIN_EPOCHS") == 10 and es.get("PATIENCE") == 8 and es.get("metric") == ["highest source-validation BA", "lowest source-validation NLL", "earliest epoch"])
    checks["selected_epochs_source_only"] = bool(len(sel) == 6 and set(sel.method) == {"ERM", "MLDG"} and set(sel.opt_seed) == {0, 1, 2} and "held_BA" not in sel.columns and "held_labels_used_for_selection" not in sel.columns)
    best_ok = True
    for _, r in sel.iterrows():
        q = traj[traj.opt_seed == int(r.opt_seed)] if r.method == "MLDG" else pd.DataFrame()
        if r.method == "MLDG":
            h = q[q.epoch == int(r.selected_epoch)]
            if len(h) != 1 or abs(float(h.val_BA.iloc[0]) / 100.0 - float(r.val_BA)) > 1e-12: best_ok = False
            higher = q[(q.val_BA > float(r.val_BA) * 100.0 + 1e-8)]
            if len(higher): best_ok = False
    checks["selection_metrics_match_trajectory"] = best_ok
    checks["paired_initial_state"] = bool(len(init) == 3 and bool(init.identical.all()) and (init.ERM_hash == init.MLDG_hash).all() and set(init.opt_seed) == {0, 1, 2})
    checks["paired_rng"] = bool(len(res) == 3 and (res.ERM_base_rng_seed == res.MLDG_base_rng_seed).all())
    checks["delta_recomputes"] = bool(len(res) == 3 and np.allclose(res.delta_BA_pp.to_numpy(float), res.MLDG_BA.to_numpy(float) - res.ERM_BA.to_numpy(float), atol=1e-10, rtol=0))
    vals = res.delta_BA_pp.to_numpy(float)
    if len(vals) == 3:
        mean, med, pos, collapse = float(np.mean(vals)), float(np.median(vals)), int(np.sum(vals >= 0)), bool(np.any(vals < -5))
        checks["gate_recomputes"] = bool(gate["mean_delta_BA_pp"] == mean and gate["median_delta_BA_pp"] == med and gate["positive_seeds"] == pos and gate["any_delta_below_minus5pp"] == collapse)
        checks["gate_definition"] = bool(gate["gates"]["G1_mean_ge_0.5"] == (mean >= 0.5) and gate["gates"]["G2_median_ge_0.25"] == (med >= 0.25) and gate["gates"]["G3_positive_ge_2_of_3"] == (pos >= 2) and gate["gates"]["G4_no_delta_below_minus5"] == (not collapse))
    else:
        checks["gate_recomputes"] = False; checks["gate_definition"] = False
    checks["forbidden_scopes_closed"] = bool(lock.get("canonical_outcome_labels_read") is False and lock.get("OpenBMI_sealed_holdout_opened") is False and lock.get("WBCIC_outer_10_opened") is False and load_json(ROOT / "NO_CANONICAL_OUTCOME_ACCESS_AUDIT.json").get("held_labels_used_for_selection") is False)
    checks["complete_cells"] = bool(set(res.opt_seed) == {0, 1, 2} and len(res) == 3)
    payload = {"schema": SCHEMA, "checks": checks, "pass": bool(all(checks.values())), "validated_files": [p.name for p in sorted(ROOT.iterdir()) if p.is_file() and p.suffix in {".csv", ".json", ".md"}]}
    (ROOT / "INDEPENDENT_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True)); return 0 if payload["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
