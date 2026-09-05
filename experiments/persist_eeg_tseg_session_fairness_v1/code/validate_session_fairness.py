"""Independent, outcome-blind validator for the session-fair TSEG closure."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]).resolve()
SCHEMA = "PERSIST_EEG_TSEG_SESSION_FAIRNESS_V1"
BRANCH = "codex/persist-eeg-tseg-session-fairness-v1"
PARENT = "codex/persist-eeg-tseg-pilot-v1"
DATASETS = ("OpenBMI", "WBCIC")
METHODS = ("B0_SUBJECT_BALANCED_ERM", "B2_TWO_TRAJECTORY_MEAN", "B3_TSEG")
BASELINE = METHODS[0]

def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))

def finite(x):
    try: return math.isfinite(float(x))
    except Exception: return False

def approx(a, b, tol=1e-9):
    return finite(a) and finite(b) and abs(float(a)-float(b)) <= tol

def boolish(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

def split_ids(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return set()
    return {str(v) for v in str(x).split(";") if str(v).strip()}

def main():
    checks = {}
    lock = read_json("PROTOCOL_LOCK.json")
    split = read_json("SPLIT_REUSE_AUDIT.json")
    gate = read_json("GO_GATE.json")
    fairness = read_json("SESSION_FAIRNESS_GATE.json")
    no_outcome = read_json("NO_CANONICAL_OUTCOME_ACCESS_AUDIT.json")
    perf = pd.read_csv(ROOT / "PER_SEED_RESULTS.csv")
    sel = pd.read_csv(ROOT / "PER_SEED_SELECTION.csv")
    delta = pd.read_csv(ROOT / "PAIRED_METHOD_DELTAS.csv")
    stability = pd.read_csv(ROOT / "OPTIMIZATION_STABILITY.csv")
    init = pd.read_csv(ROOT / "INITIAL_STATE_HASH.csv")
    partitions = pd.read_csv(ROOT / "EPISODE_PARTITION_AUDIT.csv")
    pairs = pd.read_csv(ROOT / "TRAJECTORY_PAIR_HASH.csv")
    exposure = pd.read_csv(ROOT / "SESSION_EXPOSURE_AUDIT.csv")
    mech = pd.read_csv(ROOT / "SOURCE_TRAJECTORY_STABILITY.csv")
    matched = pd.read_csv(ROOT / "MATCHED_MECHANISM_DELTAS.csv")

    checks["schema_and_parent"] = bool(lock.get("schema") == SCHEMA and split.get("schema") == SCHEMA and gate.get("schema") == SCHEMA and fairness.get("schema") == SCHEMA and lock.get("branch") == BRANCH and lock.get("parent") == PARENT)
    checks["protocol_scope"] = bool(lock.get("datasets") == list(DATASETS) and lock.get("outer_fold") == 0 and lock.get("backbone") == "EEGNet" and tuple(lock.get("methods", [])) == METHODS and lock.get("opt_seeds") == [0,1,2])
    checks["frozen_coefficients"] = bool(float(lock.get("alpha")) == .1 and float(lock.get("beta")) == 1.0 and float(lock.get("gamma")) == 1.0 and float(lock.get("tau")) == .1 and lock.get("early_stopping", {}).get("MAX_EPOCHS") == 60 and lock.get("early_stopping", {}).get("MIN_EPOCHS") == 10 and lock.get("early_stopping", {}).get("PATIENCE") == 8)

    # Parent split reuse and explicit disjointness.
    rows = split.get("rows", [])
    split_ok = bool(split.get("pass") is True and len(rows) == 2 and {r.get("dataset") for r in rows} == set(DATASETS))
    for r in rows:
        tr, va, he = split_ids(r.get("train_subjects")), split_ids(r.get("inner_validation_subjects")), split_ids(r.get("held_subjects"))
        split_ok &= bool(r.get("matches_parent_subject_lists") is True and r.get("held_disjoint_train") is True and r.get("held_disjoint_validation") is True and r.get("validation_disjoint_selection") is True and not (tr & he) and not (va & he))
    checks["split_reuse_and_disjointness"] = split_ok

    # Equal initialization across all methods and all dataset/seed cells.
    state_ok = bool(len(init) == 6 and set(init.dataset) == set(DATASETS) and set(init.opt_seed.astype(int)) == {0,1,2} and init.identical.map(boolish).all())
    hash_cols = [c for c in ("ERM_hash", "2TR_hash", "TSEG_hash") if c in init.columns]
    for _, r in init.iterrows():
        vals = [str(r[c]) for c in hash_cols]
        state_ok &= bool(vals and all(v == vals[0] for v in vals) and ("MLDG_hash" not in init.columns or str(r.get("MLDG_hash")) == vals[0]))
    checks["initial_state_equality"] = state_ok

    # Exact cell completeness and source-only selection.
    checks["complete_seed_cells"] = bool(len(perf) == 18 and set(perf.dataset) == set(DATASETS) and set(perf.seed.astype(int)) == {0,1,2} and set(perf.method) == set(METHODS) and not perf.duplicated(["dataset","seed","method"]).any())
    checks["source_only_selection"] = bool(len(sel) == 18 and set(sel.dataset) == set(DATASETS) and set(sel.seed.astype(int)) == {0,1,2} and set(sel.method) == set(METHODS) and set(sel.selection_scope) == {"source_validation_only"} and "held_BA" not in sel.columns and not sel.duplicated(["dataset","seed","method"]).any())
    join = perf.merge(sel[["dataset","seed","method","selected_epoch"]], on=["dataset","seed","method"], suffixes=("_perf","_sel"))
    checks["selected_epochs_match"] = bool(len(join) == 18 and (join.selected_epoch_perf.astype(int) == join.selected_epoch_sel.astype(int)).all())

    # Every pseudo-unseen episode must be disjoint and only involve source rows.
    part_ok = bool(len(partitions) > 0 and partitions.partition_disjoint.map(boolish).all())
    for _, r in partitions.iterrows(): part_ok &= not bool(split_ids(r.meta_train_subjects) & split_ids(r.pseudo_unseen_subjects))
    checks["episode_partition_disjoint"] = bool(part_ok)

    # Session exposure: no WBCIC session-2 labels in any training stage; all methods share universe.
    exp = exposure.copy(); exp.labels_used = exp.labels_used.map(boolish)
    required = {(d,m,s) for d in DATASETS for m in METHODS for s in ("selection_training","source_validation","refit_training","held_evaluation")}
    present = set(zip(exp.dataset, exp.method, exp.stage))
    exposure_ok = required.issubset(present)
    train_stages = {"selection_training","refit_training","episodic_pseudo_unseen_training"}
    for _, r in exp.iterrows():
        ses = split_ids(r.sessions_used)
        if r.dataset == "WBCIC" and r.stage in train_stages:
            exposure_ok &= bool(r.labels_used and ses.issubset({"0","1"}))
        if r.stage == "held_evaluation":
            exposure_ok &= bool(not r.labels_used and ses == {"2"})
    for d in DATASETS:
        universes = []
        for m in METHODS:
            q = exp[(exp.dataset == d) & (exp.method == m) & exp.stage.isin(list(train_stages)) & exp.labels_used]
            universes.append(set().union(*(split_ids(x) for x in q.sessions_used)) if len(q) else set())
        exposure_ok &= bool(all(u == universes[0] for u in universes) and universes[0] == ({"0","1"} if d == "WBCIC" else {"1","2"}))
    checks["session_exposure_and_fairness"] = bool(exposure_ok and fairness.get("pass") is True and fairness.get("WBCIC_source_session2_labels_used_by_ERM") is False and fairness.get("WBCIC_source_session2_labels_used_by_2TR") is False and fairness.get("WBCIC_source_session2_labels_used_by_TSEG") is False and fairness.get("same_labeled_session_universe") is True)

    # B2/TSEG paired trajectory identities must match on every common key.
    pair_ok = False
    if len(pairs):
        key = ["dataset","outer_fold","opt_seed","phase","epoch","step"]
        ident = ["trajectory1_seed","trajectory2_seed","partition_seed","pair_hash"]
        maps = {}
        for m in ("B2_TWO_TRAJECTORY_MEAN","B3_TSEG"):
            maps[m] = {}
            for idx, g in pairs[pairs.method == m].groupby(key, dropna=False):
                maps[m][idx] = {tuple(str(v) for v in row) for row in g[ident].itertuples(index=False, name=None)}
        common = set(maps["B2_TWO_TRAJECTORY_MEAN"]) & set(maps["B3_TSEG"])
        pair_ok = bool(common) and all(len(maps["B2_TWO_TRAJECTORY_MEAN"][k]) == 1 and maps["B2_TWO_TRAJECTORY_MEAN"][k] == maps["B3_TSEG"][k] for k in common)
    checks["trajectory_pair_hash_intersection"] = pair_ok

    # Recompute the compact paired deltas and stability statistics independently.
    finite_ba = np.isfinite(perf.held_BA.astype(float).to_numpy())
    checks["held_ba_finite_and_bounded"] = bool(finite_ba.all() and ((perf.held_BA >= 0) & (perf.held_BA <= 100)).all())
    recomputed = {}
    delta_ok = True
    for (ds, seed), g in perf.groupby(["dataset","seed"]):
        b = float(g[g.method == BASELINE].held_BA.iloc[0])
        for m in METHODS[1:]: recomputed[(ds,m,int(seed))] = float(g[g.method == m].held_BA.iloc[0]) - b
    for (ds,m), g in delta.groupby(["dataset","method"]):
        vals = np.array([recomputed[(ds,m,s)] for s in (0,1,2)], dtype=float)
        q = g.iloc[0]
        delta_ok &= bool(len(g) == 1 and approx(q.mean_delta_BA_pp, vals.mean()) and approx(q.median_delta_BA_pp, np.median(vals)) and int(q.positive_seeds) == int((vals >= 0).sum()) and approx(q.minimum_delta_BA_pp, vals.min()) and approx(q.maximum_delta_BA_pp, vals.max()) and int(q.catastrophic_cells) == int((vals < -5).sum()))
    checks["paired_deltas_recomputed"] = bool(delta_ok and set(zip(delta.dataset,delta.method)) == {(d,m) for d in DATASETS for m in METHODS[1:]})
    stab_ok = True
    for (ds,m), g in perf.groupby(["dataset","method"]):
        q = stability[(stability.dataset == ds) & (stability.method == m)]
        vals = g.held_BA.astype(float).to_numpy()
        stab_ok &= bool(len(q) == 1 and approx(q.mean_BA.iloc[0], vals.mean()) and approx(q.median_BA.iloc[0], np.median(vals)) and approx(q.seed_SD.iloc[0], vals.std(ddof=1)) and approx(q.seed_range.iloc[0], vals.max()-vals.min()))
    checks["stability_recomputed"] = bool(stab_ok)

    # Gate A/B/C/D and terminal recomputation from per-seed values only.
    def D(ds,m): return delta[(delta.dataset == ds)&(delta.method == m)].iloc[0]
    gate_a = bool(float(D("OpenBMI","B2_TWO_TRAJECTORY_MEAN").mean_delta_BA_pp) >= .5 and int(D("OpenBMI","B2_TWO_TRAJECTORY_MEAN").positive_seeds) >= 2)
    gate_b = bool(float(D("WBCIC","B2_TWO_TRAJECTORY_MEAN").mean_delta_BA_pp) >= .5 and float(D("WBCIC","B2_TWO_TRAJECTORY_MEAN").median_delta_BA_pp) >= .25 and int(D("WBCIC","B2_TWO_TRAJECTORY_MEAN").positive_seeds) >= 2)
    gate_c = bool(all(int(D(ds,"B2_TWO_TRAJECTORY_MEAN").catastrophic_cells) == 0 for ds in DATASETS))
    npos = sum(int(D(ds,"B2_TWO_TRAJECTORY_MEAN").positive_seeds) for ds in DATASETS)
    gate_d = bool(npos >= 5 or (npos == 4 and all(float(D(ds,"B2_TWO_TRAJECTORY_MEAN").mean_delta_BA_pp) >= 1.0 for ds in DATASETS)))
    two_go = bool(exposure_ok and gate_a and gate_b and gate_c and gate_d)
    terminal = "STRONG_SESSION_FAIR_TWO_TRAJECTORY_SIGNAL" if (two_go and all(float(D(ds,"B2_TWO_TRAJECTORY_MEAN").mean_delta_BA_pp) >= 1 for ds in DATASETS) and npos >= 5) else ("SESSION_FAIR_TWO_TRAJECTORY_SIGNAL_SUPPORTED" if two_go else ("TWO_TRAJECTORY_GAIN_EXPLAINED_OR_NOT_GENERAL" if float(D("WBCIC","B2_TWO_TRAJECTORY_MEAN").mean_delta_BA_pp) <= 0 else "TWO_TRAJECTORY_SIGNAL_DATASET_SPECIFIC_ONLY"))
    checks["gates_recomputed"] = bool(gate.get("gate_A_openbmi_replication") is gate_a and gate.get("gate_B_wbcic_session_fair") is gate_b and gate.get("gate_C_no_catastrophic_collapse") is gate_c and gate.get("gate_D_pooled_two_trajectory") is gate_d and gate.get("two_trajectory_go") is two_go and gate.get("terminal") == terminal)

    # Recompute matched mechanism intersection and summary.
    mk = ["dataset","opt_seed","phase","epoch","step","pair_hash"]
    mech_ok = bool(set(["dataset","opt_seed","phase","epoch","step","pair_hash","method","dtraj","abs_r_diff"]).issubset(mech.columns))
    mm = pd.DataFrame()
    if mech_ok:
        a = mech[mech.method == "B2_TWO_TRAJECTORY_MEAN"]; b = mech[mech.method == "B3_TSEG"]
        mm = a.merge(b, on=mk, suffixes=("_2TR","_TSEG"))
        mech_ok = bool(len(mm) == len(matched) and len(mm) > 0)
        if mech_ok:
            js = mm.dtraj_TSEG.astype(float) - mm.dtraj_2TR.astype(float)
            rd = mm.abs_r_diff_TSEG.astype(float) - mm.abs_r_diff_2TR.astype(float)
            mech_ok &= bool(np.allclose(matched.delta_JS_TSEG_minus_2TR.astype(float), js, atol=1e-12, rtol=0) and np.allclose(matched.delta_risk_dispersion_TSEG_minus_2TR.astype(float), rd, atol=1e-12, rtol=0) and approx(matched.delta_JS_TSEG_minus_2TR.mean(), js.mean(), 1e-12) and approx(matched.delta_JS_TSEG_minus_2TR.median(), np.median(js), 1e-12) and approx(matched.delta_risk_dispersion_TSEG_minus_2TR.mean(), rd.mean(), 1e-12) and approx(matched.delta_risk_dispersion_TSEG_minus_2TR.median(), np.median(rd), 1e-12))
    checks["matched_mechanism_intersection_recomputed"] = bool(mech_ok)

    checks["forbidden_scopes_closed"] = bool(lock.get("canonical_outcome_labels_read") is False and lock.get("OpenBMI_sealed_holdout_opened") is False and lock.get("WBCIC_outer_10_opened") is False and no_outcome.get("canonical_outcome_labels_read") is False and no_outcome.get("OpenBMI_sealed_holdout_opened") is False and no_outcome.get("WBCIC_outer_10_opened") is False and no_outcome.get("held_labels_used_for_selection") is False)
    payload = {"schema": SCHEMA, "pass": bool(all(checks.values())), "checks": checks, "terminal_recomputed": terminal}
    (ROOT / "INDEPENDENT_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 1

if __name__ == "__main__": raise SystemExit(main())
