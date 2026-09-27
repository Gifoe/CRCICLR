"""Hash-check every frozen fold artifact, then build the compact Q1–Q7 report."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

EXP = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("P_TASKCORE_RUNTIME", str(EXP.parents[2] / "p_taskcore_specificity_runtime"))).resolve()
OUT = EXP / os.environ.get("P_TASKCORE_OUTPUT_NAME", "outputs")
FILES = ("PU_BASELINE_PERSISTENCE", "PU_TASK_RELATION_WITHIN_SUBJECT", "PU_TASK_RELATION_CROSS_SUBJECT",
         "PU_FROZEN_DECODER_TRANSFER", "PU_SEMANTIC_PROBES", "PU_VARIANCE_DECOMPOSITION",
         "CONTROL_MATCHING_AUDIT", "PROTECTED_SPECIFICITY_CONTROLS", "PU_PERMUTATION_CONTROLS")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]):
    if not rows or path.exists():
        raise RuntimeError(f"empty or existing output {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def number(row: dict, key: str):
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def checked_folder(folder: Path, mandatory: tuple[str, ...]) -> dict[str, list[dict]]:
    audit = read_json(folder / "AUDIT.json")
    if audit["final_heldout_eeg_reads"] != 0 or audit["outer_fit_rows"] != 0 or audit["fit_population"] != "TRAIN_GEOMETRY":
        raise RuntimeError(f"role/heldout audit failure {folder}")
    result = {}
    for name in mandatory:
        filename = name + ".csv"
        path = folder / filename
        row = audit["files"][filename]
        data = read_csv(path)
        if sha(path) != row["sha256"] or len(data) != row["rows"]:
            raise RuntimeError(f"file hash/row failure {path}")
        result[name] = data
    return result


def lookup(rows: list[dict], fold: int, family: str) -> dict:
    matches = [r for r in rows if int(r["fold"]) == fold and r["family"] == family]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row {fold} {family}, got {len(matches)}")
    return matches[0]


def control_mean(rows, fold, family, metric):
    vals = [number(r, metric) for r in rows if int(r["fold"]) == fold and r["family"] == family]
    if not vals or any(v is None for v in vals):
        raise RuntimeError(f"control absent/nonfinite {fold} {family} {metric}")
    return mean(vals)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"output already populated {OUT}")
    recon = [read_json(RUNTIME / "reconstruction" / f"fold{f}.json") for f in range(5)]
    for f, row in enumerate(recon):
        if row["fold"] != f or not row["G_exact_source_geometry"] or row["final_heldout_eeg_reads"] != 0:
            raise RuntimeError(f"reconstruction gate failure fold{f}")
    evaluation = {name: [] for name in FILES + ("FAMILY_SUMMARY",)}
    pcontrols = {name: [] for name in ("P_ALL_SPECIFICITY_CONTROLS", "P_ALL_CONTROL_MATCHING_AUDIT")}
    audit_hashes = {"reconstruction": {}, "evaluation": {}, "p_all_controls": {}, "subject_bootstrap": {}}
    bootstrap_rows = []
    for f in range(5):
        rp = RUNTIME / "reconstruction" / f"fold{f}.json"
        audit_hashes["reconstruction"][str(f)] = sha(rp)
        ef = RUNTIME / os.environ.get("P_TASKCORE_EVAL_FOLDER", "evaluation") / f"fold{f}"
        erows = checked_folder(ef, FILES + ("FAMILY_SUMMARY",))
        audit_hashes["evaluation"][str(f)] = sha(ef / "AUDIT.json")
        if not read_json(ef / "AUDIT.json")["G_exact_source_geometry"]:
            raise RuntimeError("evaluation G gate failed")
        for name, rows in erows.items():
            evaluation[name].extend(rows)
        pf = RUNTIME / "p_all_controls" / f"fold{f}"
        prows = checked_folder(pf, tuple(pcontrols))
        audit_hashes["p_all_controls"][str(f)] = sha(pf / "AUDIT.json")
        for name, rows in prows.items():
            pcontrols[name].extend(rows)
        bp = RUNTIME / "subject_bootstrap" / f"fold{f}.json"
        bj = read_json(bp)
        if (bj["fold"] != f or bj["final_heldout_eeg_reads"] != 0 or bj["outer_fit_rows"] != 0
                or bj["fit_population"] != "TRAIN_GEOMETRY" or len(bj["rows"]) != 2):
            raise RuntimeError(f"subject-bootstrap audit failure fold{f}")
        audit_hashes["subject_bootstrap"][str(f)] = sha(bp)
        bootstrap_rows.extend([{"fold": f, **row} for row in bj["rows"]])
    overlap = []
    for f, row in enumerate(recon):
        ranks = row["ranks"]
        pb = sum(d["P_support"] for d in row["blocks"])
        ub = sum(d["U_support"] for d in row["blocks"])
        gb = sum(d["G_support"] for d in row["blocks"])
        overlap.append({"fold": f, "rank_P": ranks["P_ALL"], "rank_U": ranks["U_ALL"],
                        "rank_G": ranks["PROTECTED_G"], "rank_P_NOT_U": ranks["P_NOT_U"],
                        "rank_U_NOT_P": ranks["U_NOT_P"], "rank_NEITHER": ranks["NEITHER"],
                        "eta_utility_rank_persistence_overlap": ranks["PROTECTED_G"] / ranks["U_ALL"],
                        "U_blocks_supported_by_P_fraction": gb / ub,
                        "P_blocks_selected_by_U_fraction": gb / pb,
                        "U_NOT_P_status": "NOT_ESTIMABLE_DUE_TO_SUPPORT_OVERLAP" if ranks["U_NOT_P"] < 2 else "ESTIMABLE"})
    summary = evaluation["FAMILY_SUMMARY"]
    control = evaluation["PROTECTED_SPECIFICITY_CONTROLS"]
    pcontrol = pcontrols["P_ALL_SPECIFICITY_CONTROLS"]
    metrics = ("cross_subject_relation_cosine", "cross_subject_decoder_BA", "class_variance_fraction",
               "subject_variance_fraction", "within_subject_relation_cosine", "S1_to_S2_BA", "S2_to_S1_BA")
    contrasts = []
    for f in range(5):
        g = lookup(summary, f, "PROTECTED_G")
        for opponent in ("P_NOT_U", "U_NOT_P", "P_ALL", "U_ALL"):
            other = lookup(summary, f, opponent)
            for metric in metrics:
                a, b = number(g, metric), number(other, metric)
                contrasts.append({"fold": f, "comparison": "PROTECTED_G_minus_" + opponent,
                                  "metric": metric, "rank_G": g["rank"], "rank_other": other["rank"],
                                  "G_value": a if a is not None else "", "other_value": b if b is not None else "",
                                  "difference": a-b if a is not None and b is not None else "",
                                  "status": "ESTIMABLE" if a is not None and b is not None else "NOT_ESTIMABLE_DUE_TO_SUPPORT_OVERLAP"})
    specificity = {}
    comparator_names = ("PCA_ENERGY_CONTROL", "ENERGY_MATCHED_RANDOM", "COVARIANCE_MATCHED_RANDOM")
    primary_metrics = ("cross_subject_relation_cosine", "cross_subject_decoder_BA")
    for comparator in comparator_names:
        specificity[comparator] = {}
        for metric in primary_metrics:
            positive = sum(number(lookup(summary, f, "PROTECTED_G"), metric) > control_mean(control, f, comparator, metric)
                           for f in range(5))
            specificity[comparator][metric] = {"positive_folds": positive, "pass_4_of_5": positive >= 4}
    survives = all(entry[metric]["pass_4_of_5"] for entry in specificity.values() for metric in primary_metrics)
    random_reference = {}
    for metric in primary_metrics:
        fold_rows = []
        for f in range(5):
            values = sorted(number(r, metric) for r in control if int(r["fold"]) == f and r["family"] == "RANK_RANDOM")
            if len(values) != 100:
                raise RuntimeError("rank-random count is not 100")
            gvalue = number(lookup(summary, f, "PROTECTED_G"), metric)
            fold_rows.append({"fold": f, "random_mean": mean(values), "random_p95": values[94],
                              "G_value": gvalue, "G_percentile_among_random": sum(v <= gvalue for v in values) / 100})
        random_reference[metric] = fold_rows
    matching = {}
    for flag, name in (("selected_energy", "ENERGY_MATCHED_RANDOM"),
                       ("selected_covariance", "COVARIANCE_MATCHED_RANDOM")):
        selected = [r for r in evaluation["CONTROL_MATCHING_AUDIT"] if r[flag] == "True"]
        if len(selected) != 100:
            raise RuntimeError(f"matched control count mismatch {name}")
        matching[name] = {"achieved_energy_ratio_range": [min(number(r, "energy_ratio") for r in selected),
                                                         max(number(r, "energy_ratio") for r in selected)],
                          "achieved_energy_ratio_mean": mean(number(r, "energy_ratio") for r in selected),
                          "spectrum_log_RMS_mean": mean(number(r, "spectrum_log_RMS") for r in selected),
                          "overlap_with_G_mean": mean(number(r, "overlap_with_G") for r in selected),
                          "overlap_with_G_range": [min(number(r, "overlap_with_G") for r in selected),
                                                   max(number(r, "overlap_with_G") for r in selected)]}
    p_structure = {}
    for comparator in ("P_ALL_RANK_RANDOM", "P_ALL_ENERGY_MATCHED_RANDOM", "P_ALL_COVARIANCE_MATCHED_RANDOM"):
        p_structure[comparator] = {metric: sum(number(lookup(summary, f, "P_ALL"), metric) >
                                                 control_mean(pcontrol, f, comparator, metric) for f in range(5))
                                   for metric in primary_metrics}
    p_contains_task = all(v >= 4 for by_metric in p_structure.values() for v in by_metric.values())
    g_pnu = {metric: [number(lookup(summary, f, "PROTECTED_G"), metric) -
                      number(lookup(summary, f, "P_NOT_U"), metric) for f in range(5)] for metric in primary_metrics}
    material_utility = (sum(v > 0.05 for v in g_pnu["cross_subject_relation_cosine"]) >= 4 and
                        sum(v > 0.02 for v in g_pnu["cross_subject_decoder_BA"]) >= 4)
    # All five folds have U_NOT_P rank zero. U_ALL==G is an identity, not an
    # independent utility-only counterfactual; never infer persistence effect.
    direct_persistence = all(r["rank_U_NOT_P"] >= 2 for r in overlap)
    if survives and p_contains_task and material_utility:
        interpretation = "PERSISTENCE_IS_BROAD; UTILITY_EXTRACTS_TRANSFERABLE_TASK_CORE"
        next_model = "PROCEED_FACTORIZED_TASK_CONTEXT_MODEL"
    elif (not survives and all(specificity["PCA_ENERGY_CONTROL"][m]["positive_folds"] <= 1
                               for m in primary_metrics)):
        # Call energy/covariance an explanation only if top PCA absorbs BOTH
        # the held task relation and decoder transfer, not merely one metric.
        interpretation = "APPARENT_TASK_CORE_EXPLAINED_BY_ENERGY_COVARIANCE"
        next_model = "DO_NOT_BUILD_MODEL_FROM_PROTECTED_SPECIFICITY"
    else:
        interpretation = "MIXED_OR_INCONCLUSIVE"
        next_model = "MECHANISM_NOT_IDENTIFIED_YET"
    if direct_persistence:
        # This branch is not expected for the frozen source, but must not be
        # silently interpreted without its own predeclared direct rule.
        interpretation = "MIXED_OR_INCONCLUSIVE"
        next_model = "MECHANISM_NOT_IDENTIFIED_YET"
    source_labels = []
    if p_contains_task:
        source_labels.append("PERSISTENCE_CONTAINS_TASK_STRUCTURE")
    if material_utility:
        source_labels.append("UTILITY_SELECTION_LOCALIZES_TASK_CORE_WITHIN_PERSISTENCE")
    source_labels.append("PERSISTENCE_BEYOND_UTILITY_NOT_DIRECTLY_ESTIMABLE")
    # Failure of the joint gate is not itself proof that energy/covariance
    # explains both task metrics, especially when random matching is poor.
    source_labels.append("PROTECTED_TASK_CORE_SURVIVES_ENERGY_CONTROLS" if survives else
                         "PROTECTED_SPECIFICITY_NOT_ESTABLISHED")
    decisions = {"schema": "EEGNET_P_TASKCORE_SPECIFICITY_DECISION_V1",
                 "folds": 5, "source_commit": "8a8b708b30e19a20a83902d272fbe53fe8279f12",
                 "utility_rank_overlap": sum(r["rank_G"] for r in overlap) / sum(r["rank_U"] for r in overlap),
                 "P_ALL_above_own_rank_energy_covariance_controls": p_structure,
                 "G_specificity_4_of_5_rule": specificity, "specificity_survives": survives,
                 "G_rank_random_reference": random_reference,
                 "matched_control_quality": matching,
                 "G_minus_P_NOT_U": {k: {"equal_fold_mean": mean(v), "positive_folds": sum(x > 0 for x in v)}
                                     for k, v in g_pnu.items()},
                 "material_utility_localization_rule_passed": material_utility,
                 "direct_persistence_beyond_utility": "NOT_ESTIMABLE_DUE_TO_SUPPORT_OVERLAP",
                 "source_attribution_labels": source_labels,
                 "mechanism_interpretation": interpretation, "next_model_decision": next_model,
                 "final_heldout_eeg_reads": 0}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "PU_RECONSTRUCTION_AUDIT.json", {"schema": "PU_ALL_FOLD_RECONSTRUCTION_V1",
        "source_commit": decisions["source_commit"], "folds": recon, "source_audit_sha256": audit_hashes["reconstruction"],
        "all_G_exact": True, "final_heldout_eeg_reads": 0})
    write_csv(OUT / "PU_OVERLAP_SUMMARY.csv", overlap)
    for name in FILES:
        write_csv(OUT / (name + ".csv"), evaluation[name])
    write_csv(OUT / "PU_SOURCE_ATTRIBUTION_CONTRASTS.csv", contrasts)
    for name, rows in pcontrols.items():
        write_csv(OUT / (name + ".csv"), rows)
    write_csv(OUT / "SUBJECT_BOOTSTRAP_G_MINUS_RANK_RANDOM.csv", bootstrap_rows)
    write_csv(OUT / "PU_FAMILY_SUMMARY.csv", summary)
    write_json(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {"formal_final_heldout_eeg_reads": 0,
        "all_fold_fit_population": "TRAIN_GEOMETRY", "outer_development_fit_rows": 0,
        "checkpoint_validation_is_historical_selection_exposed": True,
        "reconstruction_before_held_role_reads": True, "all_G_exact": True,
        "fold_audit_sha256": audit_hashes})
    write_json(OUT / "DECISION_SUMMARY.json", decisions)
    def avg(family, metric):
        return mean(number(lookup(summary, f, family), metric) for f in range(5))
    def cavg(family, metric):
        return mean(control_mean(control, f, family, metric) for f in range(5))
    report = f"""# EEGNet P task-core specificity closure

Frozen EEGNet / OpenBMI_MI / seed 0 / folds 0–4. Equal-fold descriptive means; the detailed 20,000-draw subject-bootstrap intervals, 200-permutation controls and per-fold rows are in the CSVs. All selection, matching and decoder fitting used TRAIN_GEOMETRY only. OUTER_DEVELOPMENT was evaluation-only. CHECKPOINT_VALIDATION is separately reported and historically exposed to checkpoint selection. Formal final-heldout EEG reads: **0**.

## Q1. Utility-rank persistence overlap

The rank-weighted overlap is **{decisions['utility_rank_overlap']:.3f}**. Fold P ranks are {[r['rank_P'] for r in overlap]}, U/G ranks {[r['rank_G'] for r in overlap]}, P_NOT_U ranks {[r['rank_P_NOT_U'] for r in overlap]}, and U_NOT_P ranks {[r['rank_U_NOT_P'] for r in overlap]}. All five U_NOT_P contrasts are `NOT_ESTIMABLE_DUE_TO_SUPPORT_OVERLAP`; U_ALL=G is a set identity, not evidence that persistence causally creates utility.

## Q2. Persistence alone

P_ALL held cross-subject relation cosine **{avg('P_ALL','cross_subject_relation_cosine'):.3f}** and frozen cross-subject decoder BA **{avg('P_ALL','cross_subject_decoder_BA'):.3f}**. Its own-rank rank/energy/covariance matched random controls and per-fold directional counts are in `P_ALL_SPECIFICITY_CONTROLS.csv` and `DECISION_SUMMARY.json`. The locked 4/5 two-metric rule is **{'met' if p_contains_task else 'not met'}**. This is an observational geometry comparison, not a causal persistence ablation.

## Q3. Utility within persistence

G held relation cosine **{avg('PROTECTED_G','cross_subject_relation_cosine'):.3f}** versus P_NOT_U **{avg('P_NOT_U','cross_subject_relation_cosine'):.3f}**. Frozen cross-subject BA: G **{avg('PROTECTED_G','cross_subject_decoder_BA'):.3f}**, P_NOT_U **{avg('P_NOT_U','cross_subject_decoder_BA'):.3f}**. Class variance fraction: G **{avg('PROTECTED_G','class_variance_fraction'):.3f}**, P_NOT_U **{avg('P_NOT_U','class_variance_fraction'):.3f}**; subject fraction: G **{avg('PROTECTED_G','subject_variance_fraction'):.3f}**, P_NOT_U **{avg('P_NOT_U','subject_variance_fraction'):.3f}**. The locked materiality rule requiring both +0.05 relation and +0.02 BA in at least four folds is **{'met' if material_utility else 'not met'}**. Rank differs; do not interpret this as an isolated intervention.

## Q4. Persistence within utility

`NOT_ESTIMABLE_DUE_TO_SUPPORT_OVERLAP`: U_NOT_P has rank zero in every fold. U_ALL and G coincide exactly. No direct claim that persistence filters transferable utility is identified.

## Q5. Protected specificity after controls

G held relation **{avg('PROTECTED_G','cross_subject_relation_cosine'):.3f}** and cross-subject BA **{avg('PROTECTED_G','cross_subject_decoder_BA'):.3f}**. Rank-random means: **{cavg('RANK_RANDOM','cross_subject_relation_cosine'):.3f} / {cavg('RANK_RANDOM','cross_subject_decoder_BA'):.3f}**; top PCA: **{cavg('PCA_ENERGY_CONTROL','cross_subject_relation_cosine'):.3f} / {cavg('PCA_ENERGY_CONTROL','cross_subject_decoder_BA'):.3f}**; energy-selected means: **{cavg('ENERGY_MATCHED_RANDOM','cross_subject_relation_cosine'):.3f} / {cavg('ENERGY_MATCHED_RANDOM','cross_subject_decoder_BA'):.3f}**; covariance-selected means: **{cavg('COVARIANCE_MATCHED_RANDOM','cross_subject_relation_cosine'):.3f} / {cavg('COVARIANCE_MATCHED_RANDOM','cross_subject_decoder_BA'):.3f}**. The native orthogonal-control 4/5 rule is **{'passed' if survives else 'not passed'}**. Specifically, G exceeds top PCA in held relation on {specificity['PCA_ENERGY_CONTROL']['cross_subject_relation_cosine']['positive_folds']}/5 folds but exceeds it in frozen cross-subject BA on only {specificity['PCA_ENERGY_CONTROL']['cross_subject_decoder_BA']['positive_folds']}/5. Thus failing the joint rule does not imply that PCA explains every semantic observation. Matching achieved energy-ratio ranges are {matching['ENERGY_MATCHED_RANDOM']['achieved_energy_ratio_range']} for energy-selected controls and {matching['COVARIANCE_MATCHED_RANDOM']['achieved_energy_ratio_range']} for covariance-selected controls: **none of these orthogonal draws truly matched G's energy** (their best ratio is below 0.5). Their favorable/unfavorable comparisons are therefore descriptive, not a successful energy-adjusted identification. Mean overlaps with G are {matching['ENERGY_MATCHED_RANDOM']['overlap_with_G_mean']:.3f} and {matching['COVARIANCE_MATCHED_RANDOM']['overlap_with_G_mean']:.3f}. Per-candidate spectrum discrepancy and overlap, plus 100-random means, 95th percentiles and G percentiles, are in the matching CSV and decision JSON. `SUBJECT_BOOTSTRAP_G_MINUS_RANK_RANDOM.csv` gives paired 20,000-draw biological-subject intervals for G minus the 100-draw rank-random mean. The synthetic shaped random control is invertibly rescaled, not an orthogonal erasure control, and is not used in this gate.

## Q6. Task versus context

G class and subject fractions are **{avg('PROTECTED_G','class_variance_fraction'):.3f} / {avg('PROTECTED_G','subject_variance_fraction'):.3f}**, compared with P_ALL **{avg('P_ALL','class_variance_fraction'):.3f} / {avg('P_ALL','subject_variance_fraction'):.3f}**. Task structure is present, but subject context is not absent. Labels from prior utility selection make this an operational, not independently discovered, task core.

## Q7. Model-design decision

Mechanism interpretation: **`{interpretation}`**. Exactly one next action: **`{next_model}`**. {'If pursued later, the conceptual decomposition is z = z_task + z_context: only z_task receives cross-subject relational constraints; z_context remains available to native prediction and is not adversarially deleted by default. No model is implemented or trained here.' if next_model == 'PROCEED_FACTORIZED_TASK_CONTEXT_MODEL' else 'No new architecture is trained in this branch.'}

## Integrity limits

The source P and U decisions were reconstructed separately from per-block TRAIN-only persistence statistics and absolute/excess 95% utility intervals; every G coordinate and fresh geometry hash matched source commit `8a8b708b30e19a20a83902d272fbe53fe8279f12`. Utility was measured on every atomic block in the source selector before conjunction with persistence. The all-zero U_NOT_P support makes direct P-versus-utility source attribution unidentified. Matched-control achieved energy ratios, spectrum discrepancies, and G overlaps must be inspected before treating the control labels as exact matches. The optional shared-layer replication was not run: the source cached geometry does not include directly reusable independent P/U per-block activation inventories at that layer. No neural parameter or BatchNorm buffer changed. Evaluation V2 used an equivalent 60-D rotated complement for C_CURRENT, which could change featurewise-standardized decoder regularization. The corrected V4 evaluation restored the source 64-D ambient residual; the V2 artifacts remain preserved on the server and are not substituted into this report. An intermediate V3 aggregation used an overstrong negative source-attribution label despite inadequate energy matching; this report corrects the label to specificity not established, and the V3 artifact remains preserved on the server. The failed duplicate scheduled-task trigger on reconstruction fold 4 occurred after a successful immutable scientific output and was blocked by the no-overwrite guard; its failure was preserved in the runtime log, not hidden.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    index = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write_json(OUT / "HASH_INDEX.json", {"schema": "EEGNET_P_TASKCORE_OUTPUT_HASH_INDEX_V1",
        "files": index, "final_heldout_eeg_reads": 0})
    print(json.dumps({"interpretation": interpretation, "next_model": next_model,
                      "specificity_survives": survives, "outputs": len(index)}), flush=True)


if __name__ == "__main__":
    main()
