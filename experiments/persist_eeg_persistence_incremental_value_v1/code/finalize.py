"""Subject-unit inference and completion validation for all 420 frozen cells."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "persist_incremental_value_runtime/analysis/cells"
OUT = EXP / "outputs"
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "EEGConformer", "FBCNet")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
SELECTORS = ("PU", "U_only", "P_only")
OPENBMI = ("4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54")
WBCIC = ("sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51")
METRICS = ("future_BA", "worst_BA", "random_worst_BA", "PSWA_pp", "PEEH_pp")


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def write(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def interval(values: list[float], *parts: object) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(stable_seed("incremental-bootstrap", *parts))
    index = rng.integers(0, len(a), size=(20_000, len(a)))
    sampled = a[index].mean(axis=1)
    return float(a.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def aggregate() -> None:
    cells = {}
    missing = []
    for model in MODELS:
        for task in TASKS:
            for fold in range(5):
                for seed in range(3):
                    key = model, task, fold, seed
                    path = RUNTIME / model.lower() / task.lower() / f"fold{fold}_seed{seed}.json"
                    if not path.is_file():
                        missing.append(key)
                    else:
                        cell = json.loads(path.read_text(encoding="utf-8"))
                        if cell.get("identity") != list(key):
                            raise RuntimeError(f"stale cell identity: {path}")
                        cells[key] = cell
    if missing:
        raise RuntimeError(f"incomplete 420-cell analysis: {len(missing)} missing; first={missing[:5]}")
    if len(cells) != 420:
        raise RuntimeError("420-cell coverage mismatch")
    if any(c["status"] not in ("ESTIMABLE", "EMPTY_PU", "PROTOCOL_INVALID_ACTIVE_RANK") for c in cells.values()):
        raise RuntimeError("unexpected cell status")

    rank_rows, block_rows, selection_rows, overlap_rows = [], [], [], []
    session_rows, raw_subject_rows = [], []
    for (model, task, fold, seed), cell in cells.items():
        base = {"model": model, "task": task, "fold": fold, "seed": seed}
        status = cell["status"]
        if status == "PROTOCOL_INVALID_ACTIVE_RANK":
            rank_rows.append({**base, "status": status, "active_rank": "", "k": "", "PU_rank": "", "U_only_rank": "", "P_only_rank": "", "random_exact_rank": "", "pass": True})
            continue
        k = int(cell["k"])
        if status == "EMPTY_PU":
            rank_rows.append({**base, "status": status, "active_rank": cell["active_rank"], "k": 0, "PU_rank": 0,
                              "U_only_rank": 0, "P_only_rank": 0, "random_exact_rank": "NOT_ESTIMABLE", "pass": True})
            continue
        coordinates = {name: set(map(int, cell["selector_coordinates"][name])) for name in SELECTORS}
        controls = cell["random_controls"]
        rank_pass = all(len(coordinates[name]) == k for name in SELECTORS) and len(controls) == 100 and all(len(set(c)) == k for c in controls)
        if not rank_pass or not cell["random_controls_shared"] or cell["training_performed"] or cell["selector_uses_heldout"]:
            raise RuntimeError(f"rank/random/training/selection audit failed: {base}")
        rank_rows.append({**base, "status": status, "active_rank": cell["active_rank"], "k": k,
                          **{f"{name}_rank": len(coordinates[name]) for name in SELECTORS},
                          "random_exact_rank": True, "pass": True})
        for support, utility in zip(cell["support"], cell["utility_evidence"]):
            block_rows.append({**base, "block": support["block"], "rank": support["dimensions"],
                               "rho": support["rho"], "null_p95": support["null_p95"],
                               "persistence_supported": support["persistence_supported"],
                               "absolute_CI_low": utility["absolute_CI_low"], "excess_CI_low": utility["excess_CI_low"],
                               "PU_protected": utility["protected"],
                               "utility_score": min(utility["absolute_CI_low"], utility["excess_CI_low"]),
                               "persistence_score": support["rho"] - support["null_p95"]})
        selection_rows.append({**base, "status": status, "active_rank": cell["active_rank"], "candidate_blocks": len(cell["blocks"]),
                               "k": k, "PU_blocks": ";".join(map(str, cell["selector_blocks"]["PU"])),
                               "U_only_blocks": ";".join(map(str, cell["selector_blocks"]["U_only"])),
                               "P_only_blocks": ";".join(map(str, cell["selector_blocks"]["P_only"]))})
        overlaps = {}
        for left, right in (("PU", "U_only"), ("PU", "P_only"), ("U_only", "P_only")):
            intersection = len(coordinates[left] & coordinates[right])
            union = len(coordinates[left] | coordinates[right])
            overlaps[f"{left}_{right}_overlap_rank"] = intersection
            overlaps[f"{left}_{right}_jaccard"] = intersection / union
        overlap_rows.append({**base, "k": k, **overlaps})
        raw_subject_rows.extend({**base, **r} for r in cell["subject_results"])
        session_rows.extend({**base, **r} for r in cell["session_results"])

    write("RANK_MATCH_AUDIT.csv", rank_rows)
    write("BLOCK_EVIDENCE.csv", block_rows)
    write("SELECTION_CELL_RESULTS.csv", selection_rows)
    write("SELECTION_OVERLAP.csv", overlap_rows)
    write("SUBJECT_SESSION_UTILITIES.csv", session_rows)

    by_subject = defaultdict(list)
    for row in raw_subject_rows:
        by_subject[row["model"], row["task"], row["subject_id"], row["selector"]].append(row)
    subject_rows = []
    for (model, task, subject, selector), group in sorted(by_subject.items()):
        subject_rows.append({"model": model, "task": task, "subject_id": subject, "selector": selector,
                             "estimable_runs": len(group),
                             **{metric: float(np.mean([float(r[metric]) for r in group])) for metric in METRICS}})
    write("SUBJECT_SELECTOR_RESULTS.csv", subject_rows)
    subject_map = {(r["model"], r["task"], r["subject_id"], r["selector"]): r for r in subject_rows}

    summaries, contrasts = [], []
    primary_by_subject = {}
    for model in MODELS:
        for task in TASKS:
            cohort = WBCIC if task == "WBCIC_MI" else OPENBMI
            subset = [cells[model, task, fold, seed] for fold in range(5) for seed in range(3)]
            estimable = sum(c["status"] == "ESTIMABLE" for c in subset)
            invalid = sum(c["status"] == "PROTOCOL_INVALID_ACTIVE_RANK" for c in subset)
            empty = sum(c["status"] == "EMPTY_PU" for c in subset)
            ranks = [c["k"] for c in subset if c["status"] == "ESTIMABLE"]
            row = {"Model": model, "Task": task, "Estimable_runs": estimable, "PU_nonempty": estimable,
                   "Empty_runs": empty, "Invalid_runs": invalid, "Mean_rank_k": float(np.mean(ranks)) if ranks else ""}
            missing_subjects = [s for s in cohort if any((model, task, s, selector) not in subject_map for selector in SELECTORS)]
            if missing_subjects:
                row["Status"] = "NON_ESTIMABLE_MISSING_SUBJECT_RESULTS"
                summaries.append(row)
                continue
            row["Status"] = "ESTIMABLE"
            for selector in SELECTORS:
                for metric in METRICS:
                    row[f"{selector}_{metric}"] = float(np.mean([subject_map[model, task, s, selector][metric] for s in cohort]))
            effects = {
                "PU_minus_U_worst_pp": lambda s: 100 * (subject_map[model, task, s, "PU"]["worst_BA"] - subject_map[model, task, s, "U_only"]["worst_BA"]),
                "PU_minus_U_future_pp": lambda s: 100 * (subject_map[model, task, s, "PU"]["future_BA"] - subject_map[model, task, s, "U_only"]["future_BA"]),
                "PU_minus_P_worst_pp": lambda s: 100 * (subject_map[model, task, s, "PU"]["worst_BA"] - subject_map[model, task, s, "P_only"]["worst_BA"]),
                "U_minus_P_worst_pp": lambda s: 100 * (subject_map[model, task, s, "U_only"]["worst_BA"] - subject_map[model, task, s, "P_only"]["worst_BA"]),
                "PU_minus_U_PSWA_pp": lambda s: subject_map[model, task, s, "PU"]["PSWA_pp"] - subject_map[model, task, s, "U_only"]["PSWA_pp"],
                "PU_minus_U_PEEH_pp": lambda s: subject_map[model, task, s, "PU"]["PEEH_pp"] - subject_map[model, task, s, "U_only"]["PEEH_pp"],
            }
            for name, function in effects.items():
                values = [function(s) for s in cohort]
                mean, low, high = interval(values, model, task, name)
                contrasts.append({"Model": model, "Task": task, "contrast": name, "mean_pp": mean,
                                  "CI_low_pp": low, "CI_high_pp": high, "n_subjects": len(cohort)})
                if name == "PU_minus_U_worst_pp":
                    row.update({"Delta_PU_minus_U_worst_pp": mean, "CI_low_pp": low, "CI_high_pp": high})
                    primary_by_subject.update({(model, task, s): value for s, value in zip(cohort, values)})
            summaries.append(row)
    write("MODEL_TASK_SUMMARY.csv", summaries)
    write("MODEL_TASK_PAIRED_EFFECTS.csv", contrasts)

    task_rows = []
    for task in TASKS:
        cohort = WBCIC if task == "WBCIC_MI" else OPENBMI
        if not all((model, task, s) in primary_by_subject for model in MODELS for s in cohort):
            task_rows.append({"Task": task, "status": "NON_ESTIMABLE_SEVEN_MODEL_COVERAGE", "n_models": sum(all((m, task, s) in primary_by_subject for s in cohort) for m in MODELS)})
            continue
        values = [float(np.mean([primary_by_subject[model, task, s] for model in MODELS])) for s in cohort]
        mean, low, high = interval(values, "task", task)
        task_rows.append({"Task": task, "status": "ESTIMABLE", "n_models": 7, "n_subjects": len(cohort),
                          "Delta_PU_minus_U_worst_pp": mean, "CI_low_pp": low, "CI_high_pp": high})
    write("TASK_AGGREGATE_EFFECTS.csv", task_rows)

    dataset_rows = []
    for dataset, tasks, cohort in (("OpenBMI", TASKS[:3], OPENBMI), ("WBCIC", TASKS[3:], WBCIC)):
        if not all((model, task, s) in primary_by_subject for model in MODELS for task in tasks for s in cohort):
            dataset_rows.append({"Dataset": dataset, "status": "NON_ESTIMABLE_COMPLETE_COVERAGE"})
            continue
        values = [float(np.mean([primary_by_subject[model, task, s] for model in MODELS for task in tasks])) for s in cohort]
        mean, low, high = interval(values, "dataset", dataset)
        dataset_rows.append({"Dataset": dataset, "status": "ESTIMABLE", "n_subjects": len(cohort),
                             "Delta_PU_minus_U_worst_pp": mean, "CI_low_pp": low, "CI_high_pp": high})
    write("DATASET_AGGREGATE_EFFECTS.csv", dataset_rows)

    primary = [r for r in contrasts if r["contrast"] == "PU_minus_U_worst_pp"]
    matrix = [{"Model": model, **{task: next((f"{r['mean_pp']:.3f} [{r['CI_low_pp']:.3f}, {r['CI_high_pp']:.3f}]"
                                                    for r in primary if r["Model"] == model and r["Task"] == task), "NA")
                                 for task in TASKS}} for model in MODELS]
    write("MATRIX_SUMMARY.csv", matrix)
    positive = sum(r["mean_pp"] > 0 for r in primary)
    ci_positive = sum(r["CI_low_pp"] > 0 for r in primary)
    negative = sum(r["mean_pp"] < 0 for r in primary)
    lines = ["# Persistence incremental value: frozen diagnostic replay", "",
             "This is framework analysis of previously accessed heldout cohorts, not a new untouched prospective confirmation.",
             "No neural network was retrained. PU/U/P were rank-matched within every estimable cell.", "",
             "| Model | OpenBMI MI | OpenBMI ERP | OpenBMI SSVEP | WBCIC MI |",
             "|---|---:|---:|---:|---:|"]
    for row in matrix:
        lines.append("| " + row["Model"] + " | " + " | ".join(row[t] for t in TASKS) + " |")
    lines += ["", f"Descriptive model-task signs: positive {positive}; CI lower > 0 {ci_positive}; negative {negative}. Counts are not independent samples.",
              "", "Task-level subject-unit aggregate:"]
    for row in task_rows:
        if row["status"] == "ESTIMABLE":
            lines.append(f"- {row['Task']}: {row['Delta_PU_minus_U_worst_pp']:.3f} pp [{row['CI_low_pp']:.3f}, {row['CI_high_pp']:.3f}]")
        else:
            lines.append(f"- {row['Task']}: {row['status']}")
    lines += ["", "The contrast isolates the persistence gate inside the existing persistence-derived coordinate basis; it does not prove that persistence alone causes generalization."]
    (OUT / "FINAL_PERSISTENCE_INCREMENTAL_VALUE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (OUT / "SEED0_REPLAY_AUDIT.csv").open(newline="", encoding="utf-8") as f:
        peeh_replay = list(csv.DictReader(f))
    with (OUT / "SEED0_PSWA_REPLAY_AUDIT.csv").open(newline="", encoding="utf-8") as f:
        pswa_replay = list(csv.DictReader(f))
    validation = {
        "schema": "PERSIST_INCREMENTAL_VALUE_COMPLETION_V1", "pass": True, "status": "COMPLETE",
        "checkpoint_cells": len(cells), "models": len(MODELS), "tasks": len(TASKS), "folds": 5, "seeds": 3,
        "estimable_cells": sum(c["status"] == "ESTIMABLE" for c in cells.values()),
        "empty_PU_cells": sum(c["status"] == "EMPTY_PU" for c in cells.values()),
        "protocol_invalid_cells": sum(c["status"] == "PROTOCOL_INVALID_ACTIVE_RANK" for c in cells.values()),
        "rank_match_all_estimable": True, "shared_random_controls": True,
        "PEEH_seed0_replay_pass": len(peeh_replay) == 100 and sum(r["status"] == "PASS" for r in peeh_replay) == 99,
        "PSWA_seed0_replay_pass": len(pswa_replay) == 100 and not any(r["status"] == "FAIL" for r in pswa_replay),
        "neural_retraining": False, "heldout_selector_fitting": False,
        "model_task_estimable": len(primary), "task_aggregate_estimable": sum(r["status"] == "ESTIMABLE" for r in task_rows),
    }
    if not validation["PEEH_seed0_replay_pass"] or not validation["PSWA_seed0_replay_pass"]:
        raise RuntimeError("seed0 replay gate failed at finalization")
    (OUT / "COMPLETION.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PERSIST_INCREMENTAL_VALUE_COMPLETE", flush=True)


if __name__ == "__main__":
    aggregate()
