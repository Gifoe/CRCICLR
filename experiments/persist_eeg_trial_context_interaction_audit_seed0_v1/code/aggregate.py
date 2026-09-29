"""Fail-closed compact aggregation of the five frozen interaction-audit folds."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("INTERACTION_RUNTIME", r"D:\nips-temp\TotalP\P1\trial_context_interaction_runtime"))
OUT = HERE / "outputs"
LOCK = json.loads((HERE / "protocol" / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
ARMS = LOCK["arms"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def json_new(p: Path, data) -> None:
    with p.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def csv_new(p: Path, rows: list[dict]) -> None:
    if not rows: raise RuntimeError(f"empty table {p.name}")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def avg(a) -> float:
    return float(np.mean(list(a)))


def bootstrap(effect: list[dict], draws=20000) -> tuple[float, float, float]:
    """Cluster all fold appearances by biological subject ID."""
    by_subject = defaultdict(list)
    for row in effect: by_subject[str(row["subject"])].append(float(row["value"]))
    people = sorted(by_subject, key=int)
    values = np.asarray([avg(by_subject[s]) for s in people])
    rng = np.random.default_rng(20260929)
    result = np.empty(draws, np.float64)
    for i in range(draws): result[i] = np.mean(values[rng.integers(len(values), size=len(values))])
    return float(np.mean(values)), float(np.quantile(result, .025)), float(np.quantile(result, .975))


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    train, pairing, outer = [], [], []
    lp = HERE / "protocol" / "PROTOCOL_LOCK.json"
    for fold in range(5):
        tp, pp, op = (RUNTIME / "train_lock" / f"fold{fold}.json",
                      RUNTIME / "train_pairing" / f"fold{fold}.json",
                      RUNTIME / "outer" / f"fold{fold}.json")
        t, n, o = load(tp), load(pp), load(op)
        if t["protocol_lock_sha256"] != sha(lp): raise RuntimeError("TRAIN protocol mismatch")
        if n["train_lock_sha256"] != sha(tp) or o["train_lock_sha256"] != sha(tp) or o["pairing_sha256"] != sha(pp):
            raise RuntimeError("role chain hash mismatch")
        if o["role"] != "OUTER_DEVELOPMENT" or t["role"] != "TRAIN_GEOMETRY" or n["role"] != "TRAIN_GEOMETRY":
            raise RuntimeError("population role mismatch")
        if any(x["final_heldout_eeg_reads"] != 0 for x in (t,n,o)) or o["outer_selection"]:
            raise RuntimeError("forbidden heldout/selection read")
        if o["B"] != t["B"] or o["best_noninteraction"] != t["best_noninteraction"] or o["best_interaction"] != t["best_interaction"]:
            raise RuntimeError("OUTER configuration differs from TRAIN lock")
        for s in o["subjects"]:
            expected = o["outcome_inventory"][s]["n"]
            if any(o["results"][a][s]["n"] != expected for a in ARMS):
                raise RuntimeError("arm outcome-count mismatch")
        if len(o["wrong_context"]) != 200 or len(n["null"]) != 200:
            raise RuntimeError("insufficient permutation count")
        if any(len(o["matched_controls"][a]["random_linear"]) != 100 for a in ARMS[4:]):
            raise RuntimeError("insufficient capacity control count")
        train.append(t); pairing.append(n); outer.append(o)
    upstream = load(HERE / "protocol" / "SOURCE_PROVENANCE.json")
    representation = {"schema": "FROZEN_EEGNET_EMBEDDING_REUSE_V1", "backbone": "EEGNet",
                      "task": "OpenBMI_MI", "seed": 0, "stage": "EMBEDDING", "feature_dimension": 64,
                      "final_affine_interface": "frozen canonical EEGNet model.head(h): 64D embedding to binary logits",
                      "source_provenance_original_sha256": LOCK["upstream"]["source_provenance_sha256"],
                      "source_cache_extraction_only": True, "new_model_extractions": 0,
                      "folds": {}, "final_heldout_eeg_reads": 0}
    for f in range(5):
        source = upstream["folds"][str(f)]
        if outer[f]["source_checkpoint_sha256"] != source["TRAIN_GEOMETRY"]["checkpoint_sha256"] or outer[f]["target_checkpoint_sha256"] != source["OUTER_DEVELOPMENT"]["checkpoint_sha256"]:
            raise RuntimeError("representation checkpoint provenance mismatch")
        representation["folds"][str(f)] = {
            role: {k: source[role][k] for k in ("checkpoint_sha256", "split_sha256", "normalizer_sha256",
                   "model_source_sha256", "model_state_before", "model_state_after", "role_subjects")}
            for role in ("TRAIN_GEOMETRY", "OUTER_DEVELOPMENT")}
    json_new(HERE / "protocol" / "REPRESENTATION_PROVENANCE.json", representation)
    OUT.mkdir(parents=True, exist_ok=False)

    train_rows, budget_rows, outer_rows, comparison = [], [], [], []
    ablation, wrong_rows, pairing_rows, historical = [], [], [], []
    controls, context_only, compatibility, sensitivity, balance, effects = [], [], [], [], [], []
    fold_table = []
    by_contrast = defaultdict(list)
    for f, (t,n,o) in enumerate(zip(train,pairing,outer)):
        B, non, inter = t["B"], t["best_noninteraction"], t["best_interaction"]
        people = o["subjects"]
        for entry in t["budget_curve"]:
            budget_rows.append({"fold": f, **entry})
        for curve in t["cv_curves"]:
            train_rows.append({"fold": f, "arm": curve["arm"], "B": curve["B"], "C": curve["C"],
                               "K": curve["K"], "train_pseudotarget_BA": curve["BA"],
                               "train_pseudotarget_macro_f1": curve["macro_f1"],
                               "train_pseudotarget_NLL": curve["NLL"],
                               "selected": curve["C"] == t["selected"][curve["arm"]]["C"] and curve["K"] == t["selected"][curve["arm"]]["K"]})
        arm_mean = {}
        for arm in ARMS:
            rows = o["results"][arm]
            arm_mean[arm] = avg(rows[s]["BA"] for s in people)
            comparison.append({"fold": f, "arm": arm, "B": B, "C": t["selected"][arm]["C"],
                               "K": t["selected"][arm]["K"], "outer_subject_equal_BA": arm_mean[arm],
                               "TRAIN_selected_best_noninteraction": arm == non,
                               "TRAIN_selected_best_interaction": arm == inter})
            for s in people:
                x = rows[s]
                outer_rows.append({"fold": f, "subject": s, "arm": arm, "B": B,
                                   "BA": x["BA"], "macro_f1": x["macro_f1"], "NLL": x["NLL"], "outcome_count": x["n"]})
        for s in people:
            cur, no = o["results"][inter][s], o["results"][non][s]
            abl, hist = o["interaction_ablated"][s], o["historical"][s]
            wrong_mean = avg(x["subject_BA"][s] for x in o["wrong_context"])
            random_mean = avg(x["subject_BA"][s] for x in o["matched_controls"][inter]["random_linear"])
            honly = next(x["BA"] for x in o["h_only_expanded"] if x["subject"] == s)
            effect = {"fold": f, "subject": s, "B": B, "selected_interaction": inter,
                      "selected_noninteraction": non, "interaction_BA": cur["BA"],
                      "noninteraction_BA": no["BA"], "trial_only_BA": o["results"]["TRIAL_ONLY"][s]["BA"],
                      "additive_context_BA": o["results"]["ADDITIVE_CONTEXT"][s]["BA"],
                      "ablation_BA": abl["BA"], "historical_BA": hist["BA"],
                      "wrong_context_mean_BA": wrong_mean, "random_expansion_mean_BA": random_mean,
                      "h_only_expanded_BA": honly,
                      "interaction_minus_noninteraction": cur["BA"]-no["BA"],
                      "interaction_term_gain": cur["BA"]-abl["BA"],
                      "true_minus_wrong": cur["BA"]-wrong_mean,
                      "current_minus_historical": cur["BA"]-hist["BA"],
                      "interaction_minus_random": cur["BA"]-random_mean,
                      "interaction_minus_honly": cur["BA"]-honly}
            effects.append(effect)
            for key in ("interaction_minus_noninteraction", "interaction_term_gain", "true_minus_wrong",
                        "current_minus_historical", "interaction_minus_random", "interaction_minus_honly"):
                by_contrast[key].append({"subject": s, "fold": f, "value": effect[key]})
            ablation.append({"fold": f, "subject": s, "arm": inter, "full_BA": cur["BA"],
                             "zero_interaction_BA": abl["BA"], "gain": cur["BA"]-abl["BA"]})
            historical.append({"fold": f, "subject": s, "current_BA": cur["BA"],
                               "historical_S1_BA": hist["BA"], "delta": cur["BA"]-hist["BA"]})
            context_only.append({"fold": f, "subject": s, "BA": o["context_only"][s]["BA"],
                                 "NLL": o["context_only"][s]["NLL"]})
        for draw in o["wrong_context"]:
            wrong_rows.append({"fold": f, "draw": draw["draw"], "subject_equal_BA": draw["BA"],
                               "true_BA": arm_mean[inter], "true_minus_wrong": arm_mean[inter]-draw["BA"]})
        for draw in n["null"]:
            pairing_rows.append({"fold": f, "draw": draw["draw"], "TRAIN_held_BA": draw["BA"],
                                 "true_pairing_TRAIN_held_BA": n["true_pairing_BA"],
                                 "true_minus_null": n["true_pairing_BA"]-draw["BA"]})
        for family in ARMS[4:]:
            for draw in o["matched_controls"][family]["random_linear"]:
                controls.append({"fold": f, "family": family, "control": "RANDOM_FEATURE_EXPANSION",
                                 "draw": draw["draw"], "dimension": o["matched_controls"][family]["dimension"],
                                 "subject_equal_BA": draw["BA"],
                                 "interaction_BA": arm_mean[family]})
        controls.append({"fold": f, "family": inter, "control": "H_ONLY_EXPANDED", "draw": 0,
                         "dimension": o["matched_controls"][inter]["dimension"],
                         "subject_equal_BA": avg(x["BA"] for x in o["h_only_expanded"]),
                         "interaction_BA": arm_mean[inter]})
        for x in o["lowrank_compatibility"]:
            compatibility.append({"fold": f, **x, "mean_q": json.dumps(x["mean_q"]), "sd_q": json.dumps(x["sd_q"])})
        for x in o["sensitivity"]: sensitivity.append({"fold": f, **x})
        for x in o["context_balance"]:
            d = o["balanced_context_diagnostic"][x["subject"]]
            balance.append({"fold": f, **x, "label_assisted_BA": d["BA"],
                            "label_assisted_delta": d["delta_from_unlabeled_first_B"],
                            "label_assisted_not_deployable": True})
        fold_effect = [x for x in effects if x["fold"] == f]
        fold_table.append({"fold": f, "train_subject_count": len({x["subject"] for x in t["cv_curves"][0]["subject_rows"]}),
                           "outer_subject_count": len(people), "B": B,
                           **{a+"_BA": arm_mean[a] for a in ARMS},
                           "selected_bilinear_rank": t["selected"]["LOWRANK_BILINEAR"]["K"],
                           "best_noninteraction": non, "best_noninteraction_BA": arm_mean[non],
                           "best_interaction": inter, "best_interaction_BA": arm_mean[inter],
                           "interaction_minus_noninteraction": avg(x["interaction_minus_noninteraction"] for x in fold_effect),
                           "interaction_ablation_BA": avg(x["ablation_BA"] for x in fold_effect),
                           "interaction_term_gain": avg(x["interaction_term_gain"] for x in fold_effect),
                           "wrong_mean_BA": avg(x["BA"] for x in o["wrong_context"]),
                           "wrong_p95_BA": float(np.quantile([x["BA"] for x in o["wrong_context"]], .95)),
                           "current_minus_wrong": avg(x["true_minus_wrong"] for x in fold_effect),
                           "historical_BA": avg(x["historical_BA"] for x in fold_effect),
                           "current_minus_historical": avg(x["current_minus_historical"] for x in fold_effect),
                           "random_matched_BA": avg(x["random_expansion_mean_BA"] for x in fold_effect),
                           "h_only_expanded_BA": avg(x["h_only_expanded_BA"] for x in fold_effect),
                           "context_only_BA": avg(o["context_only"][s]["BA"] for s in people),
                           "improved_subject_fraction": avg(x["interaction_minus_noninteraction"] > 0 for x in fold_effect),
                           "harmed_subject_fraction": avg(x["interaction_minus_noninteraction"] < 0 for x in fold_effect),
                           "fold_interpretation": "positive_descriptive" if arm_mean[inter] > arm_mean[non] else ("tie" if arm_mean[inter] == arm_mean[non] else "negative_descriptive")})

    boot = []
    for name, rows in by_contrast.items():
        point, lo, hi = bootstrap(rows, LOCK["paired_bootstrap_draws"])
        boot.append({"contrast": name, "point_BA": point, "ci95_low": lo, "ci95_high": hi,
                     "cluster_unit": "biological_subject", "draws": LOCK["paired_bootstrap_draws"],
                     "unique_subjects": len({x["subject"] for x in rows})})
    stat = {x["contrast"]: x for x in boot}
    pos = lambda key: sum(x[key] > 0 for x in fold_table)
    gA = stat["interaction_minus_noninteraction"]["point_BA"] >= .01 and pos("interaction_minus_noninteraction") >= 4 and stat["interaction_minus_noninteraction"]["ci95_low"] > 0
    gB = stat["interaction_term_gain"]["point_BA"] >= .005 and pos("interaction_term_gain") >= 4
    gC = stat["true_minus_wrong"]["point_BA"] >= .01 and stat["true_minus_wrong"]["ci95_low"] > 0
    gD = stat["interaction_minus_random"]["point_BA"] > 0 and stat["interaction_minus_honly"]["point_BA"] > 0
    gE = stat["current_minus_historical"]["point_BA"] >= 0
    additive_delta = avg(x["additive_context_BA"]-x["trial_only_BA"] for x in effects)
    if all((gA,gB,gC,gD,gE)):
        case, action = "TRIAL_CONTEXT_INTERACTION_SUPPORTED", "PROCEED_CONTEXT_INTERACTION_BLOCK"
    elif not gD and stat["interaction_minus_noninteraction"]["point_BA"] > 0:
        case, action = "CAPACITY_EXPLAINS_INTERACTION_GAIN", "STOP_CONTEXT_INTERACTION_DIRECTION"
    elif not gC and stat["interaction_minus_noninteraction"]["point_BA"] >= .01:
        case, action = "INTERACTION_HEADROOM_EXISTS_BUT_NOT_CONTEXT_SPECIFIC", "STOP_CONTEXT_INTERACTION_DIRECTION"
    elif additive_delta > 0 and (not gA or not gB):
        case, action = "CONTEXT_USEFUL_BUT_ADDITIVE_SUFFICIENT", "DO_NOT_BUILD_INTERACTION_ARCHITECTURE"
    elif stat["interaction_minus_noninteraction"]["point_BA"] > 0:
        case, action = "INTERACTION_GAIN_SMALL_OR_UNSTABLE", "STOP_CONTEXT_INTERACTION_DIRECTION"
    else:
        case, action = "NO_TRIAL_CONTEXT_INTERACTION_ADVANTAGE", "STOP_CONTEXT_INTERACTION_DIRECTION"
    gates = {"A": bool(gA), "B": bool(gB), "C": bool(gC), "D": bool(gD), "E": bool(gE),
             "E_strong": stat["current_minus_historical"]["point_BA"] >= .005}
    summary = {"schema": "TRIAL_CONTEXT_INTERACTION_DECISION_V1", "case": case, "next_action": action,
               "gates": gates, "contrasts": stat, "additive_minus_trial_only": additive_delta,
               "fold_table": fold_table, "outer_historically_exposed": True,
               "formal_final_heldout_eeg_reads": 0,
               "branch": "codex/persist-eeg-trial-context-interaction-audit-seed0-v1"}
    tables = {"TRAIN_MODEL_SELECTION.csv": train_rows, "OUTER_MODEL_RESULTS.csv": outer_rows,
              "MODEL_FAMILY_COMPARISON.csv": comparison, "INTERACTION_TERM_ABLATION.csv": ablation,
              "CONTEXT_BUDGET_CURVE.csv": budget_rows, "WRONG_CONTEXT_NULL.csv": wrong_rows,
              "TRAIN_CONTEXT_PAIRING_NULL.csv": pairing_rows, "HISTORICAL_CONTEXT_CONTROL.csv": historical,
              "PARAMETER_MATCHED_CONTROLS.csv": controls, "CONTEXT_ONLY_RESULTS.csv": context_only,
              "INTERACTION_COMPATIBILITY.csv": compatibility, "CONTEXT_SENSITIVITY.csv": sensitivity,
              "CONTEXT_CLASS_BALANCE_AUDIT.csv": balance, "SUBJECT_LEVEL_EFFECTS.csv": effects,
              "BOOTSTRAP_CONTRASTS.csv": boot}
    for name, rows in tables.items(): csv_new(OUT / name, rows)
    json_new(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {"formal_final_heldout_eeg_reads": 0,
             "roles_loaded": ["TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"],
             "checkpoint_validation": "inventory_only", "outer_historically_exposed": True,
             "no_neural_training": True, "no_P_or_C_manipulation": True})
    json_new(OUT / "DECISION_SUMMARY.json", summary)
    print(json.dumps({"case": case, "next_action": action,
                      "summary_sha256": sha(OUT / "DECISION_SUMMARY.json")}), flush=True)


if __name__ == "__main__": main()
