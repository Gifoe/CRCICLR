#!/usr/bin/env python3
"""Aggregate frozen heldout intervention results at the biological-subject level.

This file deliberately never selects an arm, a coordinate set, an epoch, or a
hyperparameter.  It only summarizes the pre-frozen heldout inference files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common
from run_stage2_interventions import expected_stage2


METRICS = {"BA": "future_BA", "macro_F1": "future_macro_F1", "WS_BA": "WS_BA"}
SESSIONS = {"OpenBMI_MI": ("S1", "S2"), "WBCIC_MI": ("S0", "S1", "S2")}


def require_columns(frame: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {sorted(missing)}")


def endpoint_rows(frame: pd.DataFrame, condition: str, allowed_cells: pd.DataFrame | None = None,
                  random_mean: bool = False) -> pd.DataFrame:
    """Average checkpoint repetitions within biological subject/session first."""
    part = frame[frame["condition"].str.startswith("RANDOM_PRESERVE_R")].copy() if random_mean else frame[frame["condition"] == condition].copy()
    if allowed_cells is not None:
        part = part.merge(allowed_cells[["task", "fold", "seed"]].drop_duplicates(), on=["task", "fold", "seed"], how="inner")
    if part.empty:
        return pd.DataFrame()
    if random_mean:
        # Fold/seed repetitions are averaged within a fixed random arm before
        # the three frozen controls are averaged inside each biological subject.
        part = part.groupby(["task", "subject_id", "session", "condition"], as_index=False).agg(
            BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), repetitions=("fold", "count"))
        part = part.groupby(["task", "subject_id", "session"], as_index=False).agg(
            BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), random_arms=("condition", "count"),
            repetition_min=("repetitions", "min"), repetition_max=("repetitions", "max"))
        part["condition"] = "RANDOM_PRESERVE_MEAN"
        if not (part.random_arms == 3).all():
            raise RuntimeError("each random-preservation endpoint must average exactly R0/R1/R2")
    else:
        part = part.groupby(["task", "subject_id", "session"], as_index=False).agg(
            BA=("BA", "mean"), macro_F1=("macro_F1", "mean"), repetitions=("fold", "count"))
        part["condition"] = condition
        part["random_arms"] = 1
        part["repetition_min"] = part.repetitions
        part["repetition_max"] = part.repetitions

    rows = []
    for (task, subject), group in part.groupby(["task", "subject_id"], sort=True):
        expected = SESSIONS[str(task)]
        cell = group.set_index("session")
        if not set(expected).issubset(cell.index):
            raise RuntimeError(f"incomplete subject/session endpoint: {task}/{subject}/{condition}")
        rows.append({"task": task, "subject_id": subject, "condition": group["condition"].iloc[0],
                     "future_BA": float(cell.loc["S2", "BA"]),
                     "future_macro_F1": float(cell.loc["S2", "macro_F1"]),
                     "WS_BA": float(cell.loc[list(expected), "BA"].min()),
                     "repetition_min": int(group.repetition_min.min()),
                     "repetition_max": int(group.repetition_max.max()),
                     "random_arms": int(group.random_arms.max())})
    return pd.DataFrame(rows)


def paired(task: str, left: pd.DataFrame, right: pd.DataFrame, comparison: str, primary: bool) -> dict:
    merged = left.merge(right, on=["task", "subject_id"], suffixes=("_left", "_right"), validate="one_to_one")
    row = {"task": task, "comparison": comparison, "primary": primary, "biological_subjects": len(merged)}
    for printed, column in METRICS.items():
        values = (merged[f"{column}_left"] - merged[f"{column}_right"]).to_numpy(dtype=np.float64) * 100.0
        mean, low, high = common.bootstrap(values, task, comparison, printed)
        row[f"delta_{printed}_pp"] = mean
        row[f"delta_{printed}_ci95_low_pp"] = low
        row[f"delta_{printed}_ci95_high_pp"] = high
    return row


def condition_summary(task: str, condition: str, values: pd.DataFrame, p_drift: float,
                      coverage: str, random_spread: dict | None = None) -> dict:
    if values.empty:
        return {"task": task, "condition": condition, "estimable": False, "coverage": coverage}
    row = {"task": task, "condition": condition, "estimable": True, "coverage": coverage,
           "biological_subjects": len(values), "P_drift": p_drift}
    for printed, column in METRICS.items():
        observed = values[column].to_numpy(dtype=np.float64) * 100.0
        mean, low, high = common.bootstrap(observed, task, condition, printed)
        row[f"{printed}_pp"] = mean
        row[f"{printed}_ci95_low_pp"] = low
        row[f"{printed}_ci95_high_pp"] = high
    if random_spread:
        row.update(random_spread)
    return row


def drift_summary(drift: pd.DataFrame, targets: pd.DataFrame, task: str) -> tuple[dict[str, float], list[dict]]:
    valid = targets[(targets.task == task) & (targets.protected_target_estimable.astype(bool))][["task", "fold", "seed"]]
    valid_drift = drift.merge(valid, on=["task", "fold", "seed"], how="inner")
    result = {}
    sanity = []
    for condition in ("CE_REFERENCE", "PRD_ONLY", "PROTECTED_PRESERVE"):
        part = valid_drift[valid_drift["condition"] == condition]
        result[condition] = float(part["P_drift"].mean()) if len(part) else np.nan
    protected = valid_drift[valid_drift["condition"] == "PROTECTED_PRESERVE"][["task", "fold", "seed", "P_drift"]]
    prd_protected = valid_drift[valid_drift["condition"] == "PRD_ONLY"][["task", "fold", "seed", "P_drift"]]
    if len(protected):
        joined = protected.merge(prd_protected, on=["task", "fold", "seed"], suffixes=("_preserve", "_prd"), validate="one_to_one")
        delta = joined.P_drift_preserve - joined.P_drift_prd
        sanity.append({"task": task, "check": "PROTECTED_PRESERVE P drift minus PRD_ONLY P drift",
                       "mean_delta": float(delta.mean()), "all_lower": bool((delta < 0).all()), "cells": len(joined)})
    random = valid_drift[valid_drift["condition"].str.startswith("RANDOM_PRESERVE_R")]
    result["RANDOM_PRESERVE_MEAN"] = float(random.groupby(["task", "fold", "seed"])["P_drift"].mean().mean()) if len(random) else np.nan
    for rid in common.RANDOM_CONTROL_IDS:
        arm = f"RANDOM_PRESERVE_R{rid}"
        col = f"R{rid}_drift"
        prd = valid_drift[valid_drift["condition"] == "PRD_ONLY"][["task", "fold", "seed", col]]
        preserve = valid_drift[valid_drift["condition"] == arm][["task", "fold", "seed", "target_drift"]]
        joined = preserve.merge(prd, on=["task", "fold", "seed"], validate="one_to_one")
        delta = joined.target_drift - joined[col]
        sanity.append({"task": task, "check": f"{arm} target drift minus PRD-only same R{rid}",
                       "mean_delta": float(delta.mean()), "all_lower": bool((delta < 0).all()),
                       "cells": len(joined)})
    return result, sanity


def seed_sensitivity(raw: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    all_conditions = ["CE_REFERENCE", "PRD_ONLY", "PROTECTED_PRESERVE"]
    for task in common.TASKS:
        task_valid = valid[valid.task == task]
        for seed in common.SEEDS:
            for condition in all_conditions:
                part = raw[(raw.task == task) & (raw.seed == seed) & (raw["condition"] == condition)]
                if condition == "PROTECTED_PRESERVE":
                    part = part.merge(task_valid[["task", "fold", "seed"]], on=["task", "fold", "seed"], how="inner")
                if part.empty:
                    continue
                values = endpoint_rows(part, condition)
                if values.empty:
                    continue
                pieces.append({"task": task, "seed": seed, "condition": condition, "subjects": len(values),
                               "future_BA_pp": float(values.future_BA.mean() * 100),
                               "future_macro_F1_pp": float(values.future_macro_F1.mean() * 100),
                               "WS_BA_pp": float(values.WS_BA.mean() * 100)})
            random = raw[(raw.task == task) & (raw.seed == seed) & raw["condition"].str.startswith("RANDOM_PRESERVE_R")]
            random = random.merge(task_valid[["task", "fold", "seed"]], on=["task", "fold", "seed"], how="inner")
            if not random.empty:
                values = endpoint_rows(random, "RANDOM_PRESERVE_R0", random_mean=True)
                pieces.append({"task": task, "seed": seed, "condition": "RANDOM_PRESERVE_MEAN", "subjects": len(values),
                               "future_BA_pp": float(values.future_BA.mean() * 100),
                               "future_macro_F1_pp": float(values.future_macro_F1.mean() * 100),
                               "WS_BA_pp": float(values.WS_BA.mean() * 100)})
    return pd.DataFrame(pieces)


def markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free Markdown rendering; the runtime intentionally lacks tabulate."""
    if frame.empty:
        return "No estimable rows."
    columns = list(frame.columns)
    def render(value):
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.4f}"
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(render(row[column]) for column in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, effects: pd.DataFrame, coverage: pd.DataFrame, drift_sanity: pd.DataFrame, completion: dict) -> None:
    lines = ["# Protected-Subspace Preservation Intervention", "",
             "Frozen heldout intervention evaluation. OpenBMI is an internal-heldout cohort and WBCIC is the frozen true-outer cohort; neither is described as an untouched prospective confirmation.", "",
             "The primary comparison is **ProtectedPreserve − mean(RandomPreserve R0/R1/R2)**. Random arms are averaged within each biological subject before paired bootstrap; folds, seeds, and random arms are not treated as independent biological samples.", "",
             "## Task summary", "", markdown_table(summary), "",
             "## Paired biological-subject effects (percentage points)", "", markdown_table(effects), "",
             "## Target coverage", "", markdown_table(coverage), "",
             "## Representation-drift sanity", "", markdown_table(drift_sanity), "",
             "## Scope", "", "A lower protected-coordinate drift validates the intervention manipulation only. Preferential actionability is supported only when the primary heldout paired effect favors ProtectedPreserve with a 95% CI lower bound above zero. ProtectedPreserve is not required to exceed the frozen CE reference; that would answer a different question.", "",
             "## Completion", "", "`COMPLETE`" if completion["status"] == "COMPLETE" else "`INCOMPLETE`"]
    (common.OUT / "FINAL_PROTECTED_PRESERVATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = pd.read_csv(common.PROTOCOL / "PROTECTED_TARGETS.csv")
    raw = pd.read_csv(common.OUT / "SUBJECT_SESSION_RESULTS.csv")
    drift = pd.read_csv(common.OUT / "REPRESENTATION_DRIFT.csv")
    reference = pd.read_csv(common.OUT / "REFERENCE_REPLAY_AUDIT.csv")
    matching = pd.read_csv(common.PROTOCOL / "STAGE2_MATCHING_AUDIT.csv")
    randoms = pd.read_csv(common.PROTOCOL / "RANDOM_TARGET_MANIFEST.csv")
    freeze = json.loads((common.PROTOCOL / "PRE_HOLDOUT_FREEZE.json").read_text())
    require_columns(raw, common.OUT / "SUBJECT_SESSION_RESULTS.csv", {"task", "fold", "seed", "subject_id", "session", "condition", "BA", "macro_F1"})
    require_columns(drift, common.OUT / "REPRESENTATION_DRIFT.csv", {"task", "fold", "seed", "condition", "P_drift", "R0_drift", "R1_drift", "R2_drift"})
    if len(targets) != 30 or len(reference) != 6 or not reference["pass"].all():
        raise RuntimeError("reference audit/target coverage does not permit final summary")
    if len(randoms) != 3000 or not randoms.groupby(["task", "fold", "seed"]).size().eq(100).all():
        raise RuntimeError("random target pool is incomplete")
    rank_check = randoms.merge(targets[["task", "fold", "seed", "protected_rank"]], on=["task", "fold", "seed"], validate="many_to_one")
    if not (rank_check["rank"] == rank_check["protected_rank"]).all() or not all(len(json.loads(v)) == int(r) for v, r in zip(rank_check.coordinate_ids, rank_check["rank"])):
        raise RuntimeError("frozen random targets are not exact rank matches")
    if freeze.get("status") != "FROZEN_BEFORE_INTERVENTION_HELDOUT_INFERENCE":
        raise RuntimeError("pre-heldout freeze status is invalid")
    expected = set(expected_stage2(targets))
    actual = {(str(x.task), int(x.fold), int(x.seed), str(x.arm)) for x in matching.itertuples(index=False)}
    if expected != actual or not matching.matches_preflight.all() or not matching.checkpoint_frozen.all():
        raise RuntimeError("post-training matching audit failed")

    valid = targets[targets.protected_target_estimable.astype(bool)][["task", "fold", "seed"]].copy()
    coverage = targets.groupby("task", as_index=False).agg(total_cells=("task", "size"), estimable_cells=("protected_target_estimable", "sum"),
        protected_rank_mean=("protected_rank", "mean"), protected_rank_min=("protected_rank", "min"), protected_rank_max=("protected_rank", "max"))
    coverage["protected_assignment_coverage"] = coverage.estimable_cells.astype(str) + "/" + coverage.total_cells.astype(str)

    all_subject = []
    effects = []
    task_summary = []
    drift_checks = []
    for task in common.TASKS:
        task_valid = valid[valid.task == task]
        ce = endpoint_rows(raw[raw.task == task], "CE_REFERENCE")
        prd_all = endpoint_rows(raw[raw.task == task], "PRD_ONLY")
        # P/R primary outcomes use only cells where the protected target was
        # nonempty, exactly as frozen before outcome access.
        prd = endpoint_rows(raw[raw.task == task], "PRD_ONLY", task_valid)
        protected = endpoint_rows(raw[raw.task == task], "PROTECTED_PRESERVE", task_valid)
        random_raw = raw[(raw.task == task) & raw["condition"].str.startswith("RANDOM_PRESERVE_R")]
        random_mean = endpoint_rows(random_raw, "RANDOM_PRESERVE_R0", task_valid, random_mean=True)
        for values in (ce, prd_all, prd, protected, random_mean):
            if not values.empty:
                all_subject.append(values)
        drifts, sanity = drift_summary(drift, targets, task)
        drift_checks.extend(sanity)
        cov = coverage.loc[coverage.task == task, "protected_assignment_coverage"].iloc[0]
        task_summary.extend([
            condition_summary(task, "CE_REFERENCE", ce, drifts["CE_REFERENCE"], "15/15 all cells"),
            condition_summary(task, "PRD_ONLY", prd_all, drifts["PRD_ONLY"], "15/15 all cells"),
            condition_summary(task, "RANDOM_PRESERVE_MEAN", random_mean, drifts["RANDOM_PRESERVE_MEAN"], f"{cov} estimable target cells"),
            condition_summary(task, "PROTECTED_PRESERVE", protected, drifts["PROTECTED_PRESERVE"], f"{cov} estimable target cells"),
        ])
        if protected.empty:
            continue
        effects.append(paired(task, protected, random_mean, "PROTECTED_PRESERVE_MINUS_RANDOM_PRESERVE", True))
        effects.append(paired(task, protected, prd, "PROTECTED_PRESERVE_MINUS_PRD_ONLY", False))
        effects.append(paired(task, random_mean, prd, "RANDOM_PRESERVE_MINUS_PRD_ONLY", False))
        effects.append(paired(task, prd_all, ce, "PRD_ONLY_MINUS_CE_REFERENCE", False))
        effects.append(paired(task, protected, ce, "PROTECTED_PRESERVE_MINUS_CE_REFERENCE", False))

    subject = pd.concat(all_subject, ignore_index=True) if all_subject else pd.DataFrame()
    # The PRD restricted-match row has the same label as full PRD.  Preserve
    # both traceability and the user-facing all-cell PRD table by keeping only
    # the full row in SUBJECT_CONDITION_RESULTS; paired effects used `prd`.
    subject = subject.drop_duplicates(["task", "subject_id", "condition"], keep="first")
    effects_frame = pd.DataFrame(effects)
    summary_frame = pd.DataFrame(task_summary)
    random_spread = []
    for task in common.TASKS:
        part = raw[(raw.task == task) & raw["condition"].str.startswith("RANDOM_PRESERVE_R")].merge(valid[valid.task == task], on=["task", "fold", "seed"], how="inner")
        if part.empty:
            continue
        per_arm = endpoint_rows(part, "RANDOM_PRESERVE_R0", random_mean=False)
        # Endpoint construction per random arm is explicit rather than treating
        # the random id as a biological replicate.
        arm_rows = []
        for arm in sorted(part["condition"].unique()):
            values = endpoint_rows(part, arm)
            for value in values.to_dict("records"):
                value["arm"] = arm; arm_rows.append(value)
        arm_frame = pd.DataFrame(arm_rows)
        for printed, column in METRICS.items():
            pivot = arm_frame.pivot(index="subject_id", columns="arm", values=column)
            random_spread.append({"task": task, "metric": printed, "subjects": len(pivot),
                                  "mean_between_random_sd_pp": float(pivot.std(axis=1, ddof=0).mean() * 100),
                                  "mean_between_random_range_pp": float((pivot.max(axis=1) - pivot.min(axis=1)).mean() * 100)})
    random_spread_frame = pd.DataFrame(random_spread)
    if not random_spread_frame.empty:
        for task in common.TASKS:
            fields = random_spread_frame[(random_spread_frame.task == task) & (random_spread_frame.metric == "BA")]
            if len(fields):
                summary_frame.loc[(summary_frame.task == task) & (summary_frame["condition"] == "RANDOM_PRESERVE_MEAN"), "between_random_BA_sd_pp"] = fields.mean_between_random_sd_pp.iloc[0]
                summary_frame.loc[(summary_frame.task == task) & (summary_frame["condition"] == "RANDOM_PRESERVE_MEAN"), "between_random_BA_range_pp"] = fields.mean_between_random_range_pp.iloc[0]

    sensitivity = seed_sensitivity(raw, valid)
    completion = {
        "status": "COMPLETE", "reference_cells_audited": 30, "reference_replay_pass": True,
        "new_CE_reference_training": False, "protected_train_only": True, "reference_coordinate_system_fixed_before_stage2": True,
        "protected_and_random_targets_frozen_before_stage2": True, "random_pool_size": common.RANDOM_POOL_SIZE,
        "primary_random_ids": list(common.RANDOM_CONTROL_IDS), "rank_match_100_percent": True,
        "same_theta0_initialization": True, "same_stage2_manifests": True, "fixed_update_count": common.STAGE2_EPOCHS * common.STEPS_PER_EPOCH,
        "same_optimizer": True, "PRD_lambda": common.PRD_LAMBDA, "preserve_lambda": common.PRESERVE_LAMBDA,
        "all_intervention_checkpoints_frozen_before_heldout": True, "stage2_early_stopping": False,
        "heldout_hyperparameter_tuning": False, "selector_retuning": False,
        "biological_subject_bootstrap_draws": common.BOOTSTRAPS, "both_tasks_reported": True,
        "protected_assignment_coverage": coverage.to_dict("records"),
        "pre_heldout_freeze_status": freeze["status"],
        "interpretation": "primary effect is a paired biological-subject contrast; folds/seeds/random arms are not independent subjects",
    }
    common.atomic_csv(common.OUT / "SUBJECT_CONDITION_RESULTS.csv", subject)
    common.atomic_csv(common.OUT / "TASK_SUMMARY.csv", summary_frame)
    common.atomic_csv(common.OUT / "PAIRED_EFFECTS.csv", effects_frame)
    common.atomic_csv(common.OUT / "SEED_SENSITIVITY.csv", sensitivity)
    common.atomic_csv(common.OUT / "RANDOM_CONTROL_SPREAD.csv", random_spread_frame)
    common.atomic_csv(common.OUT / "REPRESENTATION_DRIFT_SANITY.csv", drift_checks)
    common.atomic_json(common.OUT / "COMPLETION.json", completion)
    write_report(summary_frame, effects_frame, coverage, pd.DataFrame(drift_checks), completion)
    print("PROTECTED_PRESERVATION_SUMMARY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
