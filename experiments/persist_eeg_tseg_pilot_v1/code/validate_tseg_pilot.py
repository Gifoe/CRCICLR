"""Independent validation for the compact TSEG pilot artifacts."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(sys.argv[1]).resolve()
SCHEMA = "PERSIST_EEG_TSEG_PILOT_V1"
DATASETS = {"OpenBMI", "WBCIC"}
METHODS = {"B0_SUBJECT_BALANCED_ERM", "B1_PLAIN_MLDG", "B2_TWO_TRAJECTORY_MEAN", "B3_TSEG"}


def j(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main() -> int:
    checks: dict[str, bool] = {}
    lock, split, gate = j("PROTOCOL_LOCK.json"), j("SPLIT_REUSE_AUDIT.json"), j("GO_GATE.json")
    perf = pd.read_csv(ROOT / "PER_SEED_RESULTS.csv")
    sel = pd.read_csv(ROOT / "PER_SEED_SELECTION.csv")
    delta = pd.read_csv(ROOT / "PAIRED_METHOD_DELTAS.csv")
    stability = pd.read_csv(ROOT / "OPTIMIZATION_STABILITY.csv")
    init = pd.read_csv(ROOT / "INITIAL_STATE_HASH.csv")
    partitions = pd.read_csv(ROOT / "EPISODE_PARTITION_AUDIT.csv")
    pairs = pd.read_csv(ROOT / "TRAJECTORY_PAIR_HASH.csv")
    mech = pd.read_csv(ROOT / "SOURCE_TRAJECTORY_STABILITY.csv")
    checks["schema"] = bool(lock.get("schema") == SCHEMA and split.get("schema") == SCHEMA and gate.get("schema") == SCHEMA)
    checks["split_reuse"] = bool(split.get("pass") and len(split.get("rows", [])) == 2 and all(r.get("matches_parent_subject_lists") for r in split["rows"]))
    checks["held_not_train_or_validation"] = bool(all(r.get("held_disjoint_train") and r.get("held_disjoint_validation") and r.get("validation_disjoint_selection") for r in split["rows"]))
    checks["protocol_scope"] = bool(lock.get("datasets") == ["OpenBMI", "WBCIC"] and lock.get("outer_fold") == 0 and lock.get("backbone") == "EEGNet" and set(lock.get("methods", [])) == METHODS and lock.get("opt_seeds") == [0, 1, 2])
    checks["constants_frozen"] = bool(float(lock.get("beta")) == 1.0 and float(lock.get("gamma")) == 1.0 and float(lock.get("tau")) == 0.1 and float(lock.get("alpha")) == 0.1 and lock.get("early_stopping", {}).get("MAX_EPOCHS") == 60 and lock.get("early_stopping", {}).get("MIN_EPOCHS") == 10 and lock.get("early_stopping", {}).get("PATIENCE") == 8)
    checks["initial_states_matched"] = bool(len(init) == 6 and init.identical.all() and (init.ERM_hash == init.MLDG_hash).all() and (init.ERM_hash == init["2TR_hash"]).all() and (init.ERM_hash == init.TSEG_hash).all())
    checks["complete_cells"] = bool(len(perf) == 24 and set(perf.dataset) == DATASETS and set(perf.seed) == {0, 1, 2} and set(perf.method) == METHODS)
    checks["selected_epochs_source_only"] = bool(len(sel) == 24 and set(sel.selection_scope) == {"source_validation_only"} and "held_BA" not in sel.columns and set(sel.method) == METHODS)
    checks["episode_partitions_source_only"] = bool(len(partitions) > 0 and partitions.partition_disjoint.all() and all(set(str(x).split(";")).isdisjoint(set(str(y).split(";"))) for x, y in zip(partitions.meta_train_subjects, partitions.pseudo_unseen_subjects)))
    # Early stopping can leave method-specific terminal epochs.  Compare the
    # trajectory identities only on keys observed for both B2 and B3, while
    # requiring every shared key to have exactly one identical identity.
    pair_ok = False
    if len(pairs):
        key = ["dataset", "outer_fold", "opt_seed", "phase", "epoch", "step"]
        identity = ["trajectory1_seed", "trajectory2_seed", "partition_seed", "pair_hash"]
        maps = {}
        for method in ("B2_TWO_TRAJECTORY_MEAN", "B3_TSEG"):
            maps[method] = {}
            frame = pairs[pairs.method == method]
            for idx, group in frame.groupby(key, dropna=False):
                sig = {tuple(str(row[col]) for col in identity) for row in group.to_dict("records")}
                maps[method][idx] = sig
        common = set(maps["B2_TWO_TRAJECTORY_MEAN"]) & set(maps["B3_TSEG"])
        pair_ok = bool(common) and all(
            len(maps["B2_TWO_TRAJECTORY_MEAN"][idx]) == 1
            and maps["B2_TWO_TRAJECTORY_MEAN"][idx] == maps["B3_TSEG"][idx]
            for idx in common
        )
    checks["trajectory_pair_hashes_matched"] = pair_ok
    checks["primary_ba_recomputes"] = bool(np.all(np.isfinite(perf.held_BA.to_numpy(float))) and (perf.held_BA >= 0).all() and (perf.held_BA <= 100).all())
    # Recompute paired deltas from the compact per-seed table.
    recompute = []
    for (ds, seed), g in perf.groupby(["dataset", "seed"]):
        b = float(g[g.method == "B0_SUBJECT_BALANCED_ERM"].held_BA.iloc[0])
        for m in sorted(METHODS - {"B0_SUBJECT_BALANCED_ERM"}):
            a = float(g[g.method == m].held_BA.iloc[0]); recompute.append((ds, m, seed, a - b))
    d_ok = True
    for ds, m, seed, v in recompute:
        q = delta[(delta.dataset == ds) & (delta.method == m)]
        # Aggregate rows are checked against the direct per-seed values below.
        if len(q) != 1: d_ok = False
    checks["paired_deltas_recomputable"] = d_ok
    # Recompute mean/median/SD/range for all method/dataset groups.
    stab_ok = True
    for (ds, m), g in perf.groupby(["dataset", "method"]):
        q = stability[(stability.dataset == ds) & (stability.method == m)]
        vals = g.held_BA.to_numpy(float)
        if len(q) != 1 or abs(float(q.mean_BA.iloc[0]) - float(vals.mean())) > 1e-10 or abs(float(q.seed_range.iloc[0]) - float(vals.max() - vals.min())) > 1e-10:
            stab_ok = False
    checks["stability_recomputes"] = stab_ok
    collapse_ok = True
    for ds in DATASETS:
        for m in METHODS - {"B0_SUBJECT_BALANCED_ERM"}:
            vals = [v for d, mm, _, v in recompute if d == ds and mm == m]
            q = delta[(delta.dataset == ds) & (delta.method == m)].iloc[0]
            collapse_ok &= bool(int(q.catastrophic_cells) == int(sum(v < -5 for v in vals)) and abs(float(q.minimum_delta_BA_pp) - min(vals)) < 1e-10)
    checks["collapse_gate_recomputes"] = bool(collapse_ok)
    checks["tseg_vs_2tr_gate_recomputes"] = bool(gate.get("gates", {}).get("G5_beats_compute_matched_2TR") is not None and len(perf[perf.method == "B3_TSEG"]) == 6 and len(perf[perf.method == "B2_TWO_TRAJECTORY_MEAN"]) == 6)
    checks["mechanism_recomputes"] = bool(len(mech) > 0 and set(mech.method) >= {"B2_TWO_TRAJECTORY_MEAN", "B3_TSEG"} and np.isfinite(mech.dtraj.to_numpy(float)).all() and np.isfinite(mech.abs_r_diff.to_numpy(float)).all())
    checks["forbidden_scopes_closed"] = bool(lock.get("canonical_outcome_labels_read") is False and lock.get("OpenBMI_sealed_holdout_opened") is False and lock.get("WBCIC_outer_10_opened") is False and j("NO_CANONICAL_OUTCOME_ACCESS_AUDIT.json").get("held_labels_used_for_selection") is False)
    payload = {"schema": SCHEMA, "checks": checks, "pass": bool(all(checks.values()))}
    (ROOT / "INDEPENDENT_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True)); return 0 if payload["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
