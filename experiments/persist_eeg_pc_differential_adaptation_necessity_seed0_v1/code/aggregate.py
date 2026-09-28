"""Compact, subject-unit aggregation of the 15 frozen differential-adaptation folds."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ANALYSIS_PATH = Path(os.environ.get("PCDA_ANALYSIS_CODE", str(Path(__file__).with_name("analysis.py"))))
spec = importlib.util.spec_from_file_location("pcda_aggregate_analysis", ANALYSIS_PATH)
assert spec and spec.loader
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
core = analysis.core
OUT = core.RUNTIME / "outputs"
LOCK = json.loads((core.EXP / "protocol/ANALYSIS_LOCK.json").read_text(encoding="utf-8"))
BRANCH = "codex/persist-eeg-pc-differential-adaptation-necessity-seed0-v1"


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if not rows: raise RuntimeError(f"empty CSV {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{k: core.clean(row.get(k, "")) for k in keys} for row in rows])


def mean_rows(rows: list[dict], keys=("BA", "macro_F1", "NLL", "P_drift", "C_drift", "total_drift", "update_norm_sq", "classifier_margin_drift")) -> dict:
    return {key: float(np.mean([r[key] for r in rows])) for key in keys}


def subject_method(cell: dict, name: str) -> dict[str, dict]:
    rows = [r for r in cell["method_subject_rows"] if r["method"] == name]
    if len(rows) != cell["outer_subject_count"]: raise RuntimeError((cell["model"], cell["task"], cell["fold"], name))
    return {str(r["subject"]): r for r in rows}


def fold_method(cell: dict, name: str) -> dict:
    return mean_rows(list(subject_method(cell, name).values()))


def random_fold(cell: dict) -> list[dict]:
    grouped = defaultdict(list)
    for row in cell["random_subject_rows"]:
        grouped[int(row["draw"])].append(row)
    if len(grouped) != 100 or any(len(rows) != cell["outer_subject_count"] for rows in grouped.values()):
        raise RuntimeError("random partition inventory mismatch")
    return [{"draw": draw, **mean_rows(rows)} for draw, rows in sorted(grouped.items())]


def contrast_subject_values(cells: list[dict], task: str, model: str, left: str, right: str) -> dict[str, float]:
    by_subject = defaultdict(list)
    for cell in cells:
        if (cell["model"], cell["task"]) != (model, task): continue
        lhs = subject_method(cell, left)
        if right == "RANDOM_MEAN":
            rr = defaultdict(list)
            for row in cell["random_subject_rows"]: rr[str(row["subject"])].append(float(row["BA"]))
            rhs = {s: float(np.mean(v)) for s, v in rr.items()}
        else:
            rhs = {s: float(row["BA"]) for s, row in subject_method(cell, right).items()}
        for subject, row in lhs.items(): by_subject[subject].append(float(row["BA"]) - rhs[subject])
    return {subject: float(np.mean(values)) for subject, values in by_subject.items()}


def bootstrap(values: dict[str, float], *parts: object) -> dict:
    if not values: raise RuntimeError("empty bootstrap")
    arr = np.asarray(list(values.values()), np.float64)
    rng = np.random.default_rng(core.seed("paired-bootstrap", *parts))
    draws = rng.integers(0, len(arr), size=(int(LOCK["subject_bootstrap_draws"]), len(arr)))
    means = arr[draws].mean(axis=1)
    return {"mean_delta_BA": float(arr.mean()), "CI95_low": float(np.quantile(means, .025)),
            "CI95_high": float(np.quantile(means, .975)), "biological_subjects": len(arr),
            "bootstrap_draws": len(means)}


def group_key(cell: dict) -> tuple[str, str]: return cell["model"], cell["task"]


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError("compact outputs already exist")
    if core.sha(core.EXP / "protocol/ANALYSIS_LOCK.json") != core.sha(core.RUNTIME / "protocol/ANALYSIS_LOCK.json"):
        raise RuntimeError("pre-OUTER analysis lock SHA drift")
    cells, selections, train_audits = [], [], []
    for model, task in core.CELLS:
        for fold in core.FOLDS:
            name = core.cell_id(model, task, fold)
            cp = core.RUNTIME / "outer" / f"{name}.json"
            sp = core.RUNTIME / "selection" / f"{name}.json"
            ap = core.RUNTIME / "train" / f"{name}.json"
            if not cp.is_file() or not sp.is_file() or not ap.is_file(): raise RuntimeError(f"missing cell {name}")
            cell = json.loads(cp.read_text(encoding="utf-8"))
            select = json.loads(sp.read_text(encoding="utf-8"))
            audit = json.loads(ap.read_text(encoding="utf-8"))
            if cell["selection_sha256"] != core.sha(sp) or cell["train_audit_sha256"] != core.sha(ap):
                raise RuntimeError(f"cell provenance mismatch {name}")
            if cell["formal_final_heldout_eeg_reads"] or cell["outer_used_for_selection"] or select["outer_development_accessed"]:
                raise RuntimeError("forbidden final-heldout or OUTER selection")
            cells.append(cell); selections.append(select); train_audits.append(audit)
    if len(cells) != 15: raise RuntimeError("not 15 cells")
    provenance = core.RUNTIME / "protocol"
    sources = []
    representations = []
    geometries = []
    for cell, select, audit in zip(cells, selections, train_audits):
        ident = {"backbone": cell["model"], "task": cell["task"], "fold": cell["fold"]}
        name = core.cell_id(cell["model"], cell["task"], cell["fold"])
        proof = audit["proof"]
        if proof["model_state_sha256_before"] != proof["model_state_sha256_after"] or cell["proof"]["model_state_sha256_before"] != cell["proof"]["model_state_sha256_after"]:
            raise RuntimeError(f"frozen state mutated: {name}")
        if cell["proof"]["checkpoint_sha256"] != proof["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed between TRAIN and OUTER: {name}")
        if int(cell["proof"]["embedding_dim"]) != int(proof["embedding_dim"]) or cell["proof"]["model_source_sha256"] != proof["model_source_sha256"]:
            raise RuntimeError(f"penultimate interface or source changed: {name}")
        sources.append({**ident, "train_proof": proof, "outer_proof": cell["proof"],
                        "train_audit_sha256": core.sha(core.RUNTIME / "train" / f"{name}.json"),
                        "selection_sha256": core.sha(core.RUNTIME / "selection" / f"{name}.json"),
                        "outer_result_sha256": core.sha(core.RUNTIME / "outer" / f"{name}.json")})
        representations.append({**ident, "interface": "PENULTIMATE_EMBEDDING",
                                "dimension": int(proof["embedding_dim"]),
                                "native_capture": proof["penultimate_hook"],
                                "head_weight_sha256": proof["head_weight_sha256"],
                                "logit_identity_test": "Each extraction batch checked native logits equals the final affine applied to the captured head input (rtol=1e-5, atol=1e-6)",
                                "logit_identity_source_sha256": proof["sdg_source_sha256"],
                                "native_head_unchanged": True, "neural_state_unchanged": True})
        geometries.append({**ident, "status": audit["status"], "full": audit["full"],
                           "crossfit_half_subjects": audit["crossfit_half_subjects"],
                           "crossfit": audit["crossfit"],
                           "historical_peeh_sha256": audit["historical_peeh_sha256"],
                           "train_cache_sha256": audit["train_cache_sha256"],
                           "selection_status": select["status"],
                           "outer_rank_P": cell["rank_P"], "outer_rank_C": cell["rank_C"]})
    core.write_json_new(provenance / "SOURCE_PROVENANCE.json", {
        "schema": "PCDA_SOURCE_PROVENANCE_V1", "cells": sources,
        "experiment_code_sha256": {
            name: core.sha(core.RUNTIME / "code" / name)
            for name in ("run_prepare_v1.py", "analysis_select_v1.py", "outer_v2.py", "aggregate_v1.py")},
        "protocol_lock_sha256": core.sha(core.EXP / "protocol/PROTOCOL_LOCK.json"),
        "analysis_lock_sha256": core.sha(core.EXP / "protocol/ANALYSIS_LOCK.json"),
        "formal_final_heldout_eeg_reads": 0})
    core.write_json_new(provenance / "REPRESENTATION_PROVENANCE.json", {
        "schema": "PCDA_REPRESENTATION_PROVENANCE_V1", "cells": representations,
        "native_classifier_and_backbone_frozen": True, "formal_final_heldout_eeg_reads": 0})
    core.write_json_new(provenance / "PC_GEOMETRY_AUDIT.json", {
        "schema": "PCDA_PC_GEOMETRY_AUDIT_V1", "primary_P": "PROTECTED",
        "primary_C": "orthogonal complement of PROTECTED in the full penultimate space",
        "PERSIST_ALL": "persistence-supported source representation family",
        "cells": geometries, "formal_final_heldout_eeg_reads": 0})
    response, necessity, tolerance, rollback, leverage = [], [], [], [], []
    partition, random_dist, deployable, normmatch, context = [], [], [], [], []
    subject_effects, completion = [], []
    for cell, select, audit in zip(cells, selections, train_audits):
        model, task, fold = cell["model"], cell["task"], cell["fold"]
        ident = {"backbone": model, "task": task, "fold": fold}
        for row in select["response_surface"]:
            response.append({**ident, **row, "distance_from_diagonal": abs(row["alpha_P"] - row["alpha_C"])})
        diff = fold_method(cell, "DIFFERENTIAL_P_C")
        uniform = fold_method(cell, "BEST_UNIFORM")
        norm = fold_method(cell, "NORM_MATCHED_UNIFORM")
        no = fold_method(cell, "NO_ADAPT")
        full = fold_method(cell, "FULL_GENERIC")
        strict = fold_method(cell, "STRICT_P_PROTECT")
        only = fold_method(cell, "P_ONLY_ADAPT")
        pca = fold_method(cell, "PCA_R_DIFFERENTIAL")
        supervised = fold_method(cell, "SUPERVISED_DECISION_R_DIFFERENTIAL")
        randoms = random_fold(cell)
        random_p95 = float(np.quantile([r["BA"] for r in randoms], .95))
        selected_p = select["partition_selections"]["PROTECTED"]["best"]
        selected_u = select["partition_selections"]["PROTECTED"]["uniform"]
        necessity.append({**ident, "status": cell["status"], "rank_P": cell["rank_P"],
                          "alpha_P": cell["selected_alpha_P"], "alpha_C": cell["selected_alpha_C"],
                          "distance_from_diagonal": abs(cell["selected_alpha_P"] - cell["selected_alpha_C"]),
                          "TRAIN_differential_BA": selected_p["BA"], "TRAIN_best_uniform_BA": selected_u["BA"],
                          "TRAIN_delta_BA": selected_p["BA"] - selected_u["BA"],
                          "OUTER_differential_BA": diff["BA"], "OUTER_best_uniform_BA": uniform["BA"],
                          "OUTER_delta_BA": diff["BA"] - uniform["BA"],
                          "OUTER_normmatched_delta_BA": diff["BA"] - norm["BA"]})
        for method_name in sorted(set(r["method"] for r in cell["method_subject_rows"])):
            rows = list(subject_method(cell, method_name).values())
            deployable.append({**ident, "method": method_name, "subjects": len(rows), **mean_rows(rows)})
        for row in cell["tolerance_subject_rows"]: tolerance.append({**ident, **row})
        full_by_s = subject_method(cell, "FULL_GENERIC")
        no_by_s = subject_method(cell, "NO_ADAPT")
        for row in cell["rollback_subject_rows"]:
            s = str(row["subject"])
            rollback.append({**ident, **row, "recovery_BA_vs_full": row["BA"] - full_by_s[s]["BA"],
                             "generic_loss_BA_vs_NoAdapt": no_by_s[s]["BA"] - full_by_s[s]["BA"]})
        for row in cell["leverage_subject_rows"]: leverage.append({**ident, **row})
        for family in analysis.REAL:
            method = "DIFFERENTIAL_P_C" if family == "PROTECTED" else family + "_DIFFERENTIAL"
            score = fold_method(cell, method)
            choice = select["partition_selections"][family]["best"]
            partition.append({**ident, "partition": family, "rank": (cell["rank_PERSIST_ALL"] if family == "PERSIST_ALL" else cell["rank_P"]),
                              "selected_alpha_P": choice["alpha_P"], "selected_alpha_C": choice["alpha_C"], **score})
        for row in randoms: random_dist.append({**ident, "partition": "RANDOM_R", "rank": cell["rank_P"], **row})
        partition.append({**ident, "partition": "RANDOM_R_DISTRIBUTION", "rank": cell["rank_P"],
                          "BA": float(np.mean([r["BA"] for r in randoms])), "BA_p95": random_p95})
        diff_by_s = subject_method(cell, "DIFFERENTIAL_P_C")
        uniform_by_s = subject_method(cell, "BEST_UNIFORM")
        norm_by_s = subject_method(cell, "NORM_MATCHED_UNIFORM")
        for s in sorted(diff_by_s, key=int):
            row = diff_by_s[s]
            normmatch.append({**ident, "subject": s, "differential_BA": row["BA"],
                              "normmatched_uniform_BA": norm_by_s[s]["BA"],
                              "delta_BA": row["BA"] - norm_by_s[s]["BA"],
                              "differential_update_norm_sq": row["update_norm_sq"],
                              "normmatched_update_norm_sq": norm_by_s[s]["update_norm_sq"],
                              "TRAIN_normmatched_alpha": cell["normmatched_uniform_alpha"]})
            subject_effects.append({**ident, "subject": s, "NoAdapt_BA": no_by_s[s]["BA"],
                                    "FullGeneric_BA": full_by_s[s]["BA"],
                                    "Differential_BA": row["BA"], "BestUniform_BA": uniform_by_s[s]["BA"],
                                    "NormMatched_BA": norm_by_s[s]["BA"],
                                    "delta_vs_NoAdapt": row["BA"] - no_by_s[s]["BA"],
                                    "delta_vs_BestUniform": row["BA"] - uniform_by_s[s]["BA"],
                                    "delta_vs_NormMatched": row["BA"] - norm_by_s[s]["BA"],
                                    "material_negative_transfer": row["BA"] - no_by_s[s]["BA"] < -.01,
                                    "material_improvement": row["BA"] - no_by_s[s]["BA"] > .01})
        for row in cell["context_subject_rows"]: context.append({**ident, **row})
        prollback = np.mean([r["recovery_BA_vs_full"] for r in rollback if
                             (r["backbone"], r["task"], r["fold"], r["component_rolled_back"], r["remaining_strength"]) ==
                             (model, task, fold, "P", 0.0)])
        crollback = np.mean([r["recovery_BA_vs_full"] for r in rollback if
                             (r["backbone"], r["task"], r["fold"], r["component_rolled_back"], r["remaining_strength"]) ==
                             (model, task, fold, "C", 0.0)])
        lev = analysis.avg(cell["leverage_subject_rows"], ("P_sensitivity_fraction", "C_sensitivity_fraction"))
        fold_effects = [r for r in subject_effects if
                        (r["backbone"], r["task"], r["fold"]) == (model, task, fold)]
        harmed = [r["material_negative_transfer"] for r in fold_effects]
        sorted_delta = sorted(r["delta_vs_NoAdapt"] for r in fold_effects)
        worst_count = max(1, int(np.ceil(len(sorted_delta) / 4)))
        completion.append({**ident, "rank_P": cell["rank_P"], "rank_C": cell["rank_C"],
                           "context_budget": cell["source_only_budget"], "NoAdapt_BA": no["BA"],
                           "FullGeneric_BA": full["BA"], "alpha_P": cell["selected_alpha_P"],
                           "alpha_C": cell["selected_alpha_C"], "alpha_gap": abs(cell["selected_alpha_P"] - cell["selected_alpha_C"]),
                           "Differential_BA": diff["BA"], "BestUniform_alpha": cell["selected_uniform_alpha"],
                           "BestUniform_BA": uniform["BA"], "Delta_vs_uniform_BA": diff["BA"] - uniform["BA"],
                           "NormMatchedUniform_BA": norm["BA"], "StrictProtect_BA": strict["BA"],
                           "POnly_BA": only["BA"], "P_rollback_recovery": float(prollback),
                           "C_rollback_recovery": float(crollback), "P_drift": diff["P_drift"],
                           "C_drift": diff["C_drift"], "P_classifier_leverage": lev["P_sensitivity_fraction"],
                           "C_classifier_leverage": lev["C_sensitivity_fraction"],
                           "PCA_differential_BA": pca["BA"], "SupervisedDecision_differential_BA": supervised["BA"],
                           "Random_differential_p95": random_p95,
                           "negative_transfer_any": bool(any(harmed)),
                           "fraction_harmed": float(np.mean(harmed)),
                           "fraction_improved": float(np.mean([r["material_improvement"] for r in fold_effects])),
                           "worst_quartile_delta_BA": float(np.mean(sorted_delta[:worst_count])),
                           "primary_fold_result": "DIFFERENTIAL_WINS" if diff["BA"] - uniform["BA"] >= .005 else "NOT_SUPPORTED"})
    contrasts = []
    contrast_map = [("BEST_UNIFORM", "Differential_minus_BestUniform"),
                    ("NORM_MATCHED_UNIFORM", "Differential_minus_NormMatchedUniform"),
                    ("PCA_R_DIFFERENTIAL", "Differential_minus_PCA"),
                    ("SUPERVISED_DECISION_R_DIFFERENTIAL", "Differential_minus_SupervisedDecision"),
                    ("RANDOM_MEAN", "Differential_minus_mean_Random"),
                    ("NO_ADAPT", "Differential_minus_NoAdapt")]
    for model, task in core.CELLS:
        for right, label in contrast_map:
            values = contrast_subject_values(cells, task, model, "DIFFERENTIAL_P_C", right)
            contrasts.append({"backbone": model, "task": task, "contrast": label,
                              **bootstrap(values, model, task, label)})
    primary_by_fold = {r["fold"]: r for r in necessity if (r["backbone"], r["task"]) == ("EEGNet", "OpenBMI_MI")}
    cross_task = [{**r, "comparison": "EEGNet_MI_vs_EEGNet_ERP",
                   "primary_EEGNet_MI_delta_BA": primary_by_fold[r["fold"]]["OUTER_delta_BA"],
                   "replication_minus_primary_delta_BA": r["OUTER_delta_BA"] - primary_by_fold[r["fold"]]["OUTER_delta_BA"]}
                  for r in necessity if (r["backbone"], r["task"]) == ("EEGNet", "OpenBMI_ERP")]
    cross_backbone = [{**r, "comparison": "EEGNet_MI_vs_EEGConformer_MI",
                       "primary_EEGNet_MI_delta_BA": primary_by_fold[r["fold"]]["OUTER_delta_BA"],
                       "replication_minus_primary_delta_BA": r["OUTER_delta_BA"] - primary_by_fold[r["fold"]]["OUTER_delta_BA"]}
                      for r in necessity if (r["backbone"], r["task"]) == ("EEGConformer", "OpenBMI_MI")]
    summary = decision(cells, completion, contrasts, subject_effects)
    outputs = {
        "ADAPTATION_RESPONSE_SURFACE.csv": response,
        "DIFFERENTIAL_NECESSITY.csv": necessity,
        "PC_ADAPTATION_TOLERANCE.csv": tolerance,
        "PC_ROLLBACK_INTERVENTION.csv": rollback,
        "PC_DECISION_LEVERAGE.csv": leverage,
        "PARTITION_SPECIFICITY_RESULTS.csv": partition,
        "RANDOM_PARTITION_DISTRIBUTION.csv": random_dist,
        "OUTER_DEPLOYABLE_RESULTS.csv": deployable,
        "UPDATE_NORM_MATCHED_CONTROLS.csv": normmatch,
        "CONTEXT_BUDGET_SENSITIVITY.csv": context,
        "SUBJECT_LEVEL_EFFECTS.csv": subject_effects,
        "BOOTSTRAP_CONTRASTS.csv": contrasts,
        "CROSS_TASK_REPLICATION.csv": cross_task,
        "CROSS_BACKBONE_REPLICATION.csv": cross_backbone,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        write_csv_new(OUT / name, rows)
        print("OUTPUT", name, len(rows), core.sha(OUT / name), flush=True)
    core.write_json_new(OUT / "BASE_ADAPTATION_OPERATOR.json", {
        "schema": "PCDA_BASE_ADAPTATION_OPERATOR_V1", "formula": "mu_hat=(1-lambda)mu_src+lambda*mu_tgt; std_hat=(1-lambda)std_src+lambda*std_tgt; h_adapt=mu_src+std_src/(std_hat+eps)*(h-mu_hat)",
        "selection": [{"backbone": s["model"], "task": s["task"], "fold": s["fold"],
                       **s["source_only_selected_base"]} for s in selections],
        "source": "TRAIN-only source-session embedding statistics", "target": "unlabeled source-session context only",
        "neural_parameter_updates": 0, "native_buffer_updates": 0, "formal_final_heldout_eeg_reads": 0})
    core.write_json_new(OUT / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {
        "schema": "PCDA_FINAL_HELDOUT_EXCLUSION_AUDIT_V1", "fold_cells": 15,
        "formal_final_heldout_eeg_reads": 0, "outer_used_for_selection": False,
        "outer_population": "OUTER_DEVELOPMENT only", "checkpoint_validation_status": "historically selection-exposed; not used for tuning"})
    core.write_json_new(OUT / "DECISION_SUMMARY.json", summary)
    report = final_report(summary, completion, contrasts, cells)
    report_path = OUT / "FINAL_REPORT.md"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream: stream.write(report)
    print("AGGREGATED", summary["primary_interpretation"], summary["exact_next_action"], flush=True)


def get_contrast(rows: list[dict], model: str, task: str, label: str) -> dict:
    return next(r for r in rows if (r["backbone"], r["task"], r["contrast"]) == (model, task, label))


def decision(cells: list[dict], table: list[dict], contrasts: list[dict], subject_effects: list[dict]) -> dict:
    base = [r for r in table if (r["backbone"], r["task"]) == ("EEGNet", "OpenBMI_MI")]
    du = get_contrast(contrasts, "EEGNet", "OpenBMI_MI", "Differential_minus_BestUniform")
    dn = get_contrast(contrasts, "EEGNet", "OpenBMI_MI", "Differential_minus_NormMatchedUniform")
    dp = get_contrast(contrasts, "EEGNet", "OpenBMI_MI", "Differential_minus_PCA")
    ds = get_contrast(contrasts, "EEGNet", "OpenBMI_MI", "Differential_minus_SupervisedDecision")
    A = sum(r["alpha_gap"] >= .25 for r in base) >= 4
    B = sum(r["Delta_vs_uniform_BA"] >= .005 for r in base) >= 4 and du["CI95_low"] > 0
    C = sum(r["Differential_BA"] > r["NormMatchedUniform_BA"] for r in base) >= 4 and dn["CI95_low"] > -.005
    Dr = sum(r["Differential_BA"] > r["Random_differential_p95"] for r in base) >= 4
    Dp = sum(r["Differential_BA"] > r["PCA_differential_BA"] for r in base) >= 3 or dp["CI95_low"] > 0
    Ds = ds["CI95_low"] > 0 or ds["mean_delta_BA"] >= 0
    D = Dr and Dp and Ds
    structured_alternative_competitive = not Dp or not Ds
    rep = []
    for model, task in core.CELLS[1:]:
        rows = [r for r in table if (r["backbone"], r["task"]) == (model, task)]
        mean = get_contrast(contrasts, model, task, "Differential_minus_BestUniform")["mean_delta_BA"]
        rep.append({"backbone": model, "task": task, "positive_folds": sum(r["Delta_vs_uniform_BA"] > 0 for r in rows),
                    "pooled_mean_delta_BA": mean, "qualitatively_consistent": sum(r["Delta_vs_uniform_BA"] > 0 for r in rows) >= 3 and mean > 0})
    E = any(r["qualitatively_consistent"] for r in rep)
    oracle = [r for cell in cells if (cell["model"], cell["task"]) == ("EEGNet", "OpenBMI_MI")
              for r in cell["evaluation_only_oracle_subject_rows"]]
    oracle_by_subject = defaultdict(list)
    for row in oracle:
        oracle_by_subject[str(row["subject"])].append(row["oracle_offdiagonal_headroom_BA"])
    oracle_headroom = np.mean([np.mean(values) >= .005 for values in oracle_by_subject.values()]) > .5
    generic_gain = np.mean([r["Differential_BA"] - r["FullGeneric_BA"] for r in base]) > 0
    if A and B and C and D and E:
        interpretation, action = "PERSISTENCE_GUIDED_DIFFERENTIAL_ADAPTATION_SUPPORTED", "PROCEED_PERSISTENCE_AWARE_ADAPTATION_MODULE"
    elif A and B and C and E and structured_alternative_competitive:
        interpretation, action = "DIFFERENTIAL_ADAPTATION_SUPPORTED_BUT_NOT_PERSIST_SPECIFIC", "PROCEED_GENERIC_STRUCTURED_DIFFERENTIAL_ADAPTATION"
    elif oracle_headroom and not B:
        interpretation, action = "OFF_DIAGONAL_ORACLE_EXISTS_BUT_DOES_NOT_TRANSFER", "STOP_FIXED_PC_ADAPTATION"
    elif generic_gain and (not B or not C):
        interpretation, action = "UNIFORM_SHRINKAGE_EXPLAINS_THE_GAIN", "STOP_PC_SPECIFIC_ADAPTATION"
    else:
        interpretation, action = "PC_DIFFERENTIAL_ADAPTATION_NOT_SUPPORTED", "STOP_PC_MODEL_DIRECTION"
    return {"schema": "PCDA_DECISION_SUMMARY_V1", "primary_interpretation": interpretation,
            "exact_next_action": action, "gate_A": A, "gate_B": B, "gate_C": C,
            "gate_D_random": Dr, "gate_D_PCA": Dp, "gate_D_supervised_honesty": Ds, "gate_D": D,
            "structured_alternative_competitive": structured_alternative_competitive,
            "gate_E": E, "replications": rep, "oracle_headroom_majority": bool(oracle_headroom),
            "generic_gain_over_full": bool(generic_gain), "primary_bootstrap": {"uniform": du, "normmatched": dn,
            "PCA": dp, "supervised": ds}, "formal_final_heldout_eeg_reads": 0,
            "branch": BRANCH, "analysis_lock_sha256": core.sha(core.EXP / "protocol/ANALYSIS_LOCK.json"),
            "protocol_lock_sha256": core.sha(core.EXP / "protocol/PROTOCOL_LOCK.json")}


def final_report(summary: dict, table: list[dict], contrasts: list[dict], cells: list[dict]) -> str:
    primary = [r for r in table if (r["backbone"], r["task"]) == ("EEGNet", "OpenBMI_MI")]
    primary_cells = [cell for cell in cells if (cell["model"], cell["task"]) == ("EEGNet", "OpenBMI_MI")]
    sweep_ranges = {"P": [], "C": []}
    harmed_generic = []
    for cell in primary_cells:
        by_axis_subject = defaultdict(list)
        for row in cell["tolerance_subject_rows"]:
            by_axis_subject[(row["axis"], str(row["subject"]))].append(row["BA"])
        for (axis, _subject), values in by_axis_subject.items():
            sweep_ranges[axis].append(max(values) - min(values))
        baseline = subject_method(cell, "NO_ADAPT")
        generic = subject_method(cell, "FULL_GENERIC")
        p_rollback = {str(r["subject"]): r for r in cell["rollback_subject_rows"] if
                      r["component_rolled_back"] == "P" and r["remaining_strength"] == 0}
        c_rollback = {str(r["subject"]): r for r in cell["rollback_subject_rows"] if
                      r["component_rolled_back"] == "C" and r["remaining_strength"] == 0}
        for subject in baseline:
            if generic[subject]["BA"] - baseline[subject]["BA"] < -.01:
                harmed_generic.append((p_rollback[subject]["BA"] - generic[subject]["BA"],
                                       c_rollback[subject]["BA"] - generic[subject]["BA"]))
    mean = lambda key, rows=primary: float(np.mean([r[key] for r in rows]))
    harmed_recovery = (f"{np.mean([v[0] for v in harmed_generic]):.6f}/{np.mean([v[1] for v in harmed_generic]):.6f} BA"
                       if harmed_generic else "not applicable (no generic-harmed subject-folds)")
    lines = ["# P/C differential adaptation necessity — seed 0", "",
             "Frozen native backbone and final affine classifier; TRAIN-only source geometry/selection; OUTER_DEVELOPMENT evaluation only; formal final-heldout EEG reads: 0.",
             "The historical strict freeze-P/C-only route failed. This experiment asks whether TRAIN-selected unequal P/C adaptation strengths transfer beyond the best homogeneous and update-norm-matched alternatives; it does not claim P invariance or C nuisance.", "",
             "## Q1 — generic recalibration", "",
             f"Primary mean full-generic BA {mean('FullGeneric_BA'):.6f} versus NoAdapt {mean('NoAdapt_BA'):.6f}. Differential adaptation harms {mean('fraction_harmed'):.3f} of held subjects by >0.01 BA and improves {mean('fraction_improved'):.3f} by >0.01 BA; mean worst-quartile delta BA {mean('worst_quartile_delta_BA'):.6f}. All five fold and subject effects are in OUTER_DEPLOYABLE_RESULTS.csv and SUBJECT_LEVEL_EFFECTS.csv.", "",
             "## Q2 — off-diagonal response", "",
             f"TRAIN-selected alpha gap >=0.25 in {sum(r['alpha_gap'] >= .25 for r in primary)}/5 primary folds; gate A={summary['gate_A']}. The full 25-point TRAIN surfaces for every real partition are in ADAPTATION_RESPONSE_SURFACE.csv.", "",
             "## Q3 — differential versus best homogeneous", "",
             f"Primary pooled paired BA delta {summary['primary_bootstrap']['uniform']['mean_delta_BA']:.6f}, 95% subject bootstrap [{summary['primary_bootstrap']['uniform']['CI95_low']:.6f}, {summary['primary_bootstrap']['uniform']['CI95_high']:.6f}]; gate B={summary['gate_B']}.", "",
             "## Q4 — norm-matched homogeneous", "",
             f"Primary pooled paired BA delta {summary['primary_bootstrap']['normmatched']['mean_delta_BA']:.6f}, 95% subject bootstrap [{summary['primary_bootstrap']['normmatched']['CI95_low']:.6f}, {summary['primary_bootstrap']['normmatched']['CI95_high']:.6f}]; gate C={summary['gate_C']}.", "",
             "## Q5 — P/C tolerance", "",
             f"Mean held subject BA sweep range: P {np.mean(sweep_ranges['P']):.6f}, C {np.mean(sweep_ranges['C']):.6f}; these are descriptive tolerance differences, not a standalone claim of transfer. PC_ADAPTATION_TOLERANCE.csv gives every five-point sweep with BA/F1/NLL, representation drift and classifier-margin drift.", "",
             "## Q6 — rollback", "",
             f"Primary full-rollback recovery relative to full generic across all subjects: P {mean('P_rollback_recovery'):.6f} BA, C {mean('C_rollback_recovery'):.6f} BA. Among {len(harmed_generic)} generic-harmed subject-folds (FullGeneric minus NoAdapt < -0.01 BA), mean P/C rollback recovery is {harmed_recovery}. These evaluation-only curves never select the deployed rates.", "",
             "## Q7 — classifier actionability", "",
             f"Primary mean frozen-head P/C sensitivity fractions {mean('P_classifier_leverage'):.6f}/{mean('C_classifier_leverage'):.6f}. PC_DECISION_LEVERAGE.csv also gives class-margin operator norms and realized P/C logit deltas; information is not equated with actionability.", "",
             "## Q8 — partition specificity", "",
             f"Random p95 criterion={summary['gate_D_random']}; PCA criterion={summary['gate_D_PCA']}; supervised-decision honesty criterion={summary['gate_D_supervised_honesty']}. Primary mean Protected-minus-PCA and Protected-minus-supervised BA are {summary['primary_bootstrap']['PCA']['mean_delta_BA']:.6f} and {summary['primary_bootstrap']['supervised']['mean_delta_BA']:.6f}. Each family received the same alpha/context/operator/pseudo-target protocol; all 100 random selected controls per fold are retained in RANDOM_PARTITION_DISTRIBUTION.csv.", "",
             "## Q9 — replication", "",
             *[f"{r['backbone']}/{r['task']}: positive folds {r['positive_folds']}/5; pooled BA delta {r['pooled_mean_delta_BA']:.6f}; consistent={r['qualitatively_consistent']}." for r in summary['replications']], "",
             "## Q10 — proceed?", "",
             f"Interpretation: `{summary['primary_interpretation']}`. Exact next action: `{summary['exact_next_action']}`. No new module was trained. Evaluation-only per-subject oracle results are nondeployable and not used to choose alphas.", "",
             "## Final 15-fold completion table", ""]
    columns = ["backbone", "task", "fold", "rank_P", "rank_C", "context_budget", "NoAdapt_BA", "FullGeneric_BA",
               "alpha_P", "alpha_C", "alpha_gap", "Differential_BA", "BestUniform_alpha", "BestUniform_BA",
               "Delta_vs_uniform_BA", "NormMatchedUniform_BA", "StrictProtect_BA", "POnly_BA", "P_rollback_recovery",
               "C_rollback_recovery", "P_drift", "C_drift", "P_classifier_leverage", "C_classifier_leverage",
               "PCA_differential_BA", "SupervisedDecision_differential_BA", "Random_differential_p95",
               "negative_transfer_any", "primary_fold_result"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in table:
        lines.append("| " + " | ".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k]) for k in columns) + " |")
    lines += ["", "## Three cell-level summaries", ""]
    for model, task in core.CELLS:
        rows = [r for r in table if (r["backbone"], r["task"]) == (model, task)]
        diff = get_contrast(contrasts, model, task, "Differential_minus_BestUniform")
        lines.append(f"- {model}/{task}: mean differential BA {np.mean([r['Differential_BA'] for r in rows]):.6f}, best-uniform BA {np.mean([r['BestUniform_BA'] for r in rows]):.6f}, paired delta {diff['mean_delta_BA']:.6f} [95% CI {diff['CI95_low']:.6f}, {diff['CI95_high']:.6f}], selected off-diagonal {sum(r['alpha_gap'] > 0 for r in rows)}/5.")
    lines += ["", f"Primary interpretation: `{summary['primary_interpretation']}`.",
              f"Exact next action: `{summary['exact_next_action']}`.",
              "Formal final-heldout EEG reads: `0`.", f"Branch: `{BRANCH}`.",
              "Final commit SHA: recorded in the GitHub delivery response (a report cannot contain its own commit hash).", ""]
    return "\n".join(lines)


if __name__ == "__main__": main()
