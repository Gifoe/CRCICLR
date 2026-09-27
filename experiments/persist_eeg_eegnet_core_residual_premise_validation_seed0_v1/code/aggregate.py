"""Verify immutable fold products and emit compact mechanism conclusions."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

EXP = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("CORE_RESIDUAL_RUNTIME", str(EXP.parents[2] / "core_residual_premise_runtime"))).resolve()
OUT = EXP / os.environ.get("CORE_RESIDUAL_OUTPUT_FOLDER", "outputs")
FILES = ("TASK_GEOMETRY_DECOMPOSITION", "TASK_GEOMETRY_SHARED_FRACTION",
         "SUBSPACE_FROZEN_DECODER_TRANSFER", "GEOMETRY_VS_PREDICTION_SUMMARY",
         "G_VS_PCA_CONTRAST", "G_VS_SUPERVISED_CONTRAST",
         "RESIDUAL_ONLY_DECODER_RESULTS", "RESIDUAL_ERROR_COMPLEMENTARITY",
         "RESIDUAL_ORACLE_UNION", "CORE_RESIDUAL_JOINT_DECODER",
         "JOINT_CAPACITY_CONTROL", "STACKED_LOGIT_FUSION")
RESIDUALS = ("P_NOT_U", "NEITHER", "C_CURRENT")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"empty {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def val(row, key):
    return float(row[key])


def one(rows, fold, family, **extra):
    selected = [r for r in rows if int(r["fold"]) == fold and r["family"] == family
                and all(r.get(k) == v for k, v in extra.items())]
    if len(selected) != 1:
        raise RuntimeError(f"row not unique {fold} {family} {extra}: {len(selected)}")
    return selected[0]


def decoder(rows, fold, family):
    return one(rows, fold, family, eval_population="OUTER_DEVELOPMENT",
               fit_session="both", eval_session="both")


def positive_folds(summary, comparator, metric, minimum):
    return sum(val(one(summary, f, "G"), metric) - val(one(summary, f, comparator), metric) >= minimum
               for f in range(5))


def means(summary, family, metric):
    return mean(val(one(summary, f, family), metric) for f in range(5))


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    source = load(EXP / "protocol/SOURCE_PROVENANCE.json")
    if source["source_commits"]["taskcore"] != "d5184c5423806693a3b218f2d0329cd6b29d5d9a":
        raise RuntimeError("source lock drift")
    parent_reconstruction = (EXP.parents[1] / "experiments" /
        "persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1" /
        os.environ.get("TASKCORE_OUTPUT_FOLDER", "outputs") / "PU_RECONSTRUCTION_AUDIT.json")
    key = "experiments/persist_eeg_eegnet_p_taskcore_specificity_closure_seed0_v1/outputs/PU_RECONSTRUCTION_AUDIT.json"
    if sha(parent_reconstruction) != source["source_file_sha256"][key]:
        raise RuntimeError("immutable reconstruction audit drift")
    parent_folds = load(parent_reconstruction)["folds"]
    rows = {name: [] for name in FILES}
    audits = {"construction": {}, "evaluation": {}}
    construction = []
    for f in range(5):
        cf = RUNTIME / "construction" / f"fold{f}"
        ca = load(cf / "AUDIT.json")
        if (ca["fold"] != f or not ca["G_exact_source_geometry"]
                or ca["outer_rows_seen_at_construction"] != 0
                or ca["final_heldout_eeg_reads"] != 0
                or sha(cf / "BASES.npz") != ca["basis_file_sha256"]):
            raise RuntimeError(f"construction audit failed fold{f}")
        parent = parent_folds[f]
        if (parent["fold"] != f or parent["ranks"]["U_NOT_P"] != 0
                or parent["families"]["U_NOT_P"] or parent["ranks"]["PROTECTED_G"] != ca["G_rank"]
                or parent["checkpoint_sha256"] != ca["source_checkpoint_sha256"]):
            raise RuntimeError(f"source support mismatch fold{f}")
        audits["construction"][str(f)] = sha(cf / "AUDIT.json")
        ca["U_NOT_P"] = {"rank": 0, "status": "NOT_ESTIMABLE",
                         "reason": "immutable source established U subset P in this fold"}
        construction.append(ca)
        ef = RUNTIME / "evaluation" / f"fold{f}"
        ea = load(ef / "AUDIT.json")
        if (ea["fold"] != f or ea["construction_audit_sha256"] != audits["construction"][str(f)]
                or not ea["source_family_numeric_reproduction"] or ea["outer_fit_rows"] != 0
                or ea["checkpoint_validation_fit_rows"] != 0 or ea["final_heldout_eeg_reads"] != 0
                or ea["random_draws"] < 100):
            raise RuntimeError(f"evaluation audit failed fold{f}")
        audits["evaluation"][str(f)] = sha(ef / "AUDIT.json")
        for name in FILES:
            path = ef / (name + ".csv")
            item = ea["files"][path.name]
            part = read_csv(path)
            if sha(path) != item["sha256"] or len(part) != item["rows"]:
                raise RuntimeError(f"row/hash failure: {path}")
            rows[name].extend(part)
    summary = rows["GEOMETRY_VS_PREDICTION_SUMMARY"]
    if len([r for r in summary if r["family"] == "G"]) != 5:
        raise RuntimeError("G summary incomplete")
    constructions = {"schema": "CORE_RESIDUAL_ALL_FOLD_CONSTRUCTION_V1",
                     "source_commits": source["source_commits"], "folds": construction,
                     "fold_audit_sha256": audits["construction"],
                     "all_G_exact": True, "final_heldout_eeg_reads": 0}

    relation = {alt: positive_folds(summary, alt, "cross_subject_relation_cosine", .03)
                for alt in ("PCA_R", "SUPERVISED_DECISION_R")}
    variability = {}
    for alt in relation:
        variability[alt] = {metric: sum(val(one(summary, f, "G"), metric) <
                                         val(one(summary, f, alt), metric) for f in range(5))
                            for metric in ("subject_task_heterogeneity", "session_task_instability")}
    core_role = (all(count >= 4 for count in relation.values()) and
                 all(max(by_metric.values()) >= 4 for by_metric in variability.values()))
    relation_only = all(count >= 4 for count in relation.values())
    predictive = {}
    for alt in relation:
        predictive[alt] = {
            "BA_better_by_at_least_0_005": sum(val(one(summary, f, alt), "frozen_cross_subject_BA") -
                                                val(one(summary, f, "G"), "frozen_cross_subject_BA") >= .005
                                                for f in range(5)),
            "NLL_lower": sum(val(one(summary, f, alt), "frozen_cross_subject_NLL") <
                             val(one(summary, f, "G"), "frozen_cross_subject_NLL") for f in range(5))}
    not_predictive_maximal = any(x["BA_better_by_at_least_0_005"] >= 3 or x["NLL_lower"] >= 4
                                 for x in predictive.values())
    # Case C needs the *same* generic family to match or exceed both
    # reproducibility and prediction; all-five-fold dominance is sufficient.
    generic_dominates = {alt: all(
        val(one(summary, f, alt), "cross_subject_relation_cosine") >=
        val(one(summary, f, "G"), "cross_subject_relation_cosine") and
        val(one(summary, f, alt), "frozen_cross_subject_BA") >=
        val(one(summary, f, "G"), "frozen_cross_subject_BA")
        for f in range(5)) for alt in relation}

    # Add fold subject-equal rows; trial-level rows remain available for exact audit.
    for f in range(5):
        for residual in RESIDUALS:
            matching = [r for r in rows["RESIDUAL_ORACLE_UNION"] if int(r["fold"]) == f
                        and r["residual"] == residual]
            if not matching:
                raise RuntimeError("oracle rows absent")
            for name in ("rescue_fraction", "redundancy_fraction", "oracle_union_BA"):
                by_subject = defaultdict(list)
                for r in matching:
                    by_subject[r["subject"]].append(val(r, name))
                equal = mean(mean(v) for v in by_subject.values())
                rows["RESIDUAL_ERROR_COMPLEMENTARITY"].append({
                    "fold": f, "residual": residual, "row_type": "SUBJECT_EQUAL_FOLD_SUMMARY",
                    "quantity": name, "subject_equal_mean": equal, "subjects": len(by_subject)})

    residual_decisions = {}
    for residual in RESIDUALS:
        joint_name = "G_PLUS_" + residual
        capacity_name = "TRAIN_PCA_DIMENSION_CONTROL_" + residual
        joint = rows["CORE_RESIDUAL_JOINT_DECODER"]
        capacity = rows["JOINT_CAPACITY_CONTROL"]
        stack = rows["STACKED_LOGIT_FUSION"]
        joint_BA = sum(val(decoder(joint, f, joint_name), "BA") -
                       val(one(summary, f, "G"), "frozen_cross_subject_BA") >= .005 for f in range(5))
        joint_NLL = sum(val(decoder(joint, f, joint_name), "NLL") <
                        val(one(summary, f, "G"), "frozen_cross_subject_NLL") for f in range(5))
        capacity_BA = sum(val(decoder(joint, f, joint_name), "BA") -
                          val(decoder(capacity, f, capacity_name), "BA") >= .005 for f in range(5))
        capacity_NLL = sum(val(decoder(joint, f, joint_name), "NLL") <
                           val(decoder(capacity, f, capacity_name), "NLL") for f in range(5))
        rescue = mean(val(r, "subject_equal_mean") for r in rows["RESIDUAL_ERROR_COMPLEMENTARITY"]
                      if r.get("row_type") == "SUBJECT_EQUAL_FOLD_SUMMARY"
                      and r.get("residual") == residual and r.get("quantity") == "rescue_fraction")
        stack_rows = [r for r in stack if r["residual"] == residual and r["scope"] == "OUTER_DEVELOPMENT_BOTH_SESSIONS"]
        if len(stack_rows) != 5:
            raise RuntimeError("stacked fold row count")
        stack_ba = mean(val(r, "BA") for r in stack_rows)
        stack_nll = mean(val(r, "NLL") for r in stack_rows)
        g_ba = means(summary, "G", "frozen_cross_subject_BA")
        g_nll = means(summary, "G", "frozen_cross_subject_NLL")
        deployable = joint_BA >= 3 or joint_NLL >= 4
        capacity_specific = capacity_BA >= 3 or capacity_NLL >= 4
        stack_favorable = stack_ba > g_ba or stack_nll < g_nll
        # Nontrivial rescue threshold is frozen in the protocol before OUTER.
        complement = deployable and rescue >= .10 and stack_favorable and capacity_specific
        residual_decisions[residual] = {"joint_BA_gain_at_least_0_005_folds": joint_BA,
            "joint_NLL_gain_folds": joint_NLL, "capacity_BA_advantage_folds": capacity_BA,
            "capacity_NLL_advantage_folds": capacity_NLL,
            "rescue_fraction_equal_fold_subject_equal": rescue,
            "stacked_BA_equal_fold": stack_ba, "stacked_NLL_equal_fold": stack_nll,
            "stack_favorable": stack_favorable, "capacity_specific": capacity_specific,
            "RESIDUAL_COMPLEMENTS_G": complement}
    role_labels = []
    if residual_decisions["P_NOT_U"]["RESIDUAL_COMPLEMENTS_G"]:
        role_labels.append("MISSING_UTILITY_IN_PERSISTENT_RESIDUAL")
    if residual_decisions["NEITHER"]["RESIDUAL_COMPLEMENTS_G"]:
        role_labels.append("MISSING_UTILITY_IN_NONPERSISTENT_RESIDUAL")
    if residual_decisions["C_CURRENT"]["RESIDUAL_COMPLEMENTS_G"]:
        role_labels.append("MISSING_UTILITY_DISTRIBUTED_ACROSS_COMPLEMENT")
    if not role_labels:
        role_labels.append("RESIDUAL_COMPLEMENTARITY_NOT_ESTABLISHED")
    any_complement = any(row["RESIDUAL_COMPLEMENTS_G"] for row in residual_decisions.values())
    if core_role and any_complement:
        interpretation, next_action = "REPRODUCIBLE_CORE_PLUS_COMPLEMENTARY_UTILITY", "PROCEED_CORE_RESIDUAL_EEGNET"
    elif core_role and not any_complement and not not_predictive_maximal:
        interpretation, next_action = "G_IS_PREDICTIVE_CORE_WITH_LITTLE_COMPLEMENT", "PROCEED_CORE_DOMINANT_EEGNET"
    elif core_role and not any_complement:
        interpretation, next_action = "G_REPRODUCIBLE_BUT_RESIDUAL_NOT_USEFUL", "DO_NOT_BUILD_TWO_BRANCH_MODEL_YET"
    elif not core_role and any(generic_dominates.values()):
        interpretation, next_action = "G_NOT_DISTINCT_FROM_GENERIC_PREDICTIVE_SUBSPACE", "STOP_P_SPECIFIC_MODEL_DESIGN"
    else:
        interpretation, next_action = "MECHANISM_INCONCLUSIVE", "NO_NEW_MODEL_YET"
    decision = {"schema": "CORE_RESIDUAL_PREMISE_DECISION_V1",
        "source_commits": source["source_commits"],
        "core_role": "G_REPRODUCIBLE_GEOMETRY_ADVANTAGE" if core_role else "CORE_ROLE_NOT_ESTABLISHED",
        "G_relation_plus_0_03_folds": relation, "lower_variability_folds": variability,
        "relation_only_gate": relation_only, "G_NOT_PREDICTIVE_MAXIMAL": not_predictive_maximal,
        "generic_family_dominates_G_relation_and_BA_all_folds": generic_dominates,
        "alternative_predictive_comparison": predictive,
        "residual_roles": role_labels, "residual_decisions": residual_decisions,
        "CORE_RESIDUAL_PREMISE_SUPPORTED": bool(core_role and any_complement),
        "primary_interpretation": interpretation, "next_action": next_action,
        "final_heldout_eeg_reads": 0}

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "SUBSPACE_CONSTRUCTION_AUDIT.json", constructions)
    for name, data in rows.items():
        write_csv(OUT / (name + ".csv"), data)
    write_json(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {
        "formal_final_heldout_eeg_reads": 0, "fit_population": "TRAIN_GEOMETRY",
        "outer_fit_rows": 0, "checkpoint_validation_fit_rows": 0,
        "checkpoint_validation_historical_selection_exposed": True,
        "oracle_union_is_nondeployable": True, "all_G_exact_source_geometry": True,
        "fold_audit_sha256": audits})
    write_json(OUT / "DECISION_SUMMARY.json", decision)
    report = f"""# EEGNet Core–Residual premise validation

Frozen canonical EEGNet / OpenBMI_MI / seed 0 / five folds, 64-D embedding. TRAIN_GEOMETRY alone fitted all subspaces, decoders and grouped-OOF stackers. OUTER_DEVELOPMENT was evaluation-only. CHECKPOINT_VALIDATION is separately selection-exposed. Formal final-heldout EEG reads: **0**.

The source `U_NOT_P` rank is zero in all five folds and is retained as `NOT_ESTIMABLE` in the construction audit; no metric is fabricated for it.

## Q1–Q2. Geometry versus prediction

G held cross-subject relation **{means(summary,'G','cross_subject_relation_cosine'):.3f}**, PCA_R **{means(summary,'PCA_R','cross_subject_relation_cosine'):.3f}**, supervised-decision_R **{means(summary,'SUPERVISED_DECISION_R','cross_subject_relation_cosine'):.3f}**. Shared-relation alignment: G **{means(summary,'G','shared_relation_alignment'):.3f}**, PCA_R **{means(summary,'PCA_R','shared_relation_alignment'):.3f}**, supervised-decision_R **{means(summary,'SUPERVISED_DECISION_R','shared_relation_alignment'):.3f}**. G subject-task heterogeneity **{means(summary,'G','subject_task_heterogeneity'):.3f}** and session-task instability **{means(summary,'G','session_task_instability'):.3f}**. The locked reproducible-geometry gate is **{'met' if core_role else 'not met'}**; relation +0.03 fold counts versus PCA/supervised are **{relation}**. Do not infer an advantage merely from the mean.

Frozen held cross-subject BA: G **{means(summary,'G','frozen_cross_subject_BA'):.3f}**, PCA_R **{means(summary,'PCA_R','frozen_cross_subject_BA'):.3f}**, supervised-decision_R **{means(summary,'SUPERVISED_DECISION_R','frozen_cross_subject_BA'):.3f}**. The predictive-maximality contrast is **{'supported' if not_predictive_maximal else 'not established'}**. FISHER_1D is explicitly rank one, not a matched competitor. Paired 20,000-draw biological-subject intervals are in the G-versus-PCA and G-versus-supervised CSVs.

## Q3–Q4. Residual location and error rescue

Residual-only BA (P_NOT_U / NEITHER / C_CURRENT): **{means(summary,'P_NOT_U','frozen_cross_subject_BA'):.3f} / {means(summary,'NEITHER','frozen_cross_subject_BA'):.3f} / {means(summary,'C_CURRENT','frozen_cross_subject_BA'):.3f}**. Subject-equal rescue fractions P(residual correct | G wrong): **{residual_decisions['P_NOT_U']['rescue_fraction_equal_fold_subject_equal']:.3f} / {residual_decisions['NEITHER']['rescue_fraction_equal_fold_subject_equal']:.3f} / {residual_decisions['C_CURRENT']['rescue_fraction_equal_fold_subject_equal']:.3f}**. Error transitions are trial-level in `RESIDUAL_ERROR_COMPLEMENTARITY.csv`. `ORACLE_UNION` is a **NONDEPLOYABLE_COMPLEMENTARITY_UPPER_BOUND**; it is not an ensemble or achievable held predictor. Descriptive residual-role labels: **{', '.join(role_labels)}**.

## Q5–Q6. Deployable joint, capacity, and stacked tests

G-only BA **{means(summary,'G','frozen_cross_subject_BA'):.3f}**. G+P_NOT_U **{mean(val(decoder(rows['CORE_RESIDUAL_JOINT_DECODER'], f, 'G_PLUS_P_NOT_U'),'BA') for f in range(5)):.3f}**, G+NEITHER **{mean(val(decoder(rows['CORE_RESIDUAL_JOINT_DECODER'], f, 'G_PLUS_NEITHER'),'BA') for f in range(5)):.3f}**, and G+C_CURRENT **{mean(val(decoder(rows['CORE_RESIDUAL_JOINT_DECODER'], f, 'G_PLUS_C_CURRENT'),'BA') for f in range(5)):.3f}**. TRAIN-PCA dimension-matched BA for the corresponding residuals: **{mean(val(decoder(rows['JOINT_CAPACITY_CONTROL'], f, 'TRAIN_PCA_DIMENSION_CONTROL_P_NOT_U'),'BA') for f in range(5)):.3f} / {mean(val(decoder(rows['JOINT_CAPACITY_CONTROL'], f, 'TRAIN_PCA_DIMENSION_CONTROL_NEITHER'),'BA') for f in range(5)):.3f} / {mean(val(decoder(rows['JOINT_CAPACITY_CONTROL'], f, 'TRAIN_PCA_DIMENSION_CONTROL_C_CURRENT'),'BA') for f in range(5)):.3f}**. G+C_CURRENT spans the full embedding; its D>64 feature-count comparator is TRAIN PCA64 zero-padded to D, with effective rank 64 disclosed rather than falsely called a novel higher-rank representation. The two-logit stacker used five biological-subject-grouped OOF TRAIN folds. Its BA for P_NOT_U / NEITHER / C_CURRENT: **{residual_decisions['P_NOT_U']['stacked_BA_equal_fold']:.3f} / {residual_decisions['NEITHER']['stacked_BA_equal_fold']:.3f} / {residual_decisions['C_CURRENT']['stacked_BA_equal_fold']:.3f}**. Residual-specific deployable/capacity/stacked gates: **{json.dumps(residual_decisions, sort_keys=True)}**. A trial-level oracle rescue fraction alone never supports a model premise.

## Q7–Q8. Mechanism and next action

Core–Residual premise: **{'SUPPORTED' if core_role and any_complement else 'NOT_ESTABLISHED'}**. Primary interpretation: **`{interpretation}`**. Exactly one next action: **`{next_action}`**. No neural architecture was implemented or trained here.

Case C's generic-subspace sufficiency check asks whether the *same* alternative has at least G's relation cosine and BA in every fold: **{generic_dominates}**. The first aggregation incorrectly returned Case E despite this pattern. The Case C mapping was corrected after aggregate inspection; no representation, decoder, fold outcome or frozen scientific threshold changed. Both aggregate versions are retained on the server.

## Integrity boundary

All five G geometries and pre-existing family held metrics were reproduced against the immutable source, with fold-level hashes in the construction and evaluation audits. All PCA/supervised/Fisher bases were fitted before any held-role read. Normalizer, checkpoint, parameters and BatchNorm state were checked. Source-specificity results already inspected OUTER_DEVELOPMENT historically; this is a frozen follow-up and the OUTER evidence is descriptive, not an untouched confirmatory test. Formal final-heldout EEG reads: 0.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    index = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write_json(OUT / "HASH_INDEX.json", {"schema": "CORE_RESIDUAL_OUTPUT_HASH_INDEX_V1",
        "files": index, "final_heldout_eeg_reads": 0})
    print(json.dumps({"interpretation": interpretation, "next_action": next_action,
                      "outputs": len(index)}), flush=True)


if __name__ == "__main__":
    main()
