"""Strict all-cell true-outer aggregation and biological-subject bootstrap."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from heldout_eval import EXPECTED_WBCIC, cell_output, locked
from train_queue import EXP, FOLDS, MODELS, RUNTIME, SEEDS, TASKS, cell_dir, sha


OUT = EXP / "outputs"
DRAW_COUNT = 20_000
BOOT_SEED = 20260916
METRICS = ("BA", "macro_F1", "WS_BA")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    target = OUT / name
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if len(vector) not in (10, 14) or not np.all(np.isfinite(vector)):
        raise RuntimeError("bootstrap requires 10 or 14 finite biological subjects")
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, len(vector), size=(DRAW_COUNT, len(vector)))
    draws = vector[indices].mean(axis=1)
    return float(vector.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sessions_for(task: str) -> tuple[str, ...]:
    return ("S1", "S2", "S3") if task == "WBCIC_MI" else ("S1", "S2")


def subject_sort(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def collect() -> tuple[dict, list[dict], list[dict]]:
    lock, _ = locked()
    if not (RUNTIME / "TRAINING_COMPLETED.json").is_file() or not (RUNTIME / "HELDOUT_COMPLETED.json").is_file():
        raise RuntimeError("training or all-session heldout completion marker absent")
    manifest, session_rows = [], []
    by_key = {(item["model"], item["task"], item["fold"], item["seed"]): item
              for item in lock["checkpoints"]}
    for model in MODELS:
        for task in TASKS:
            cohort = sorted(lock["cohorts"]["WBCIC" if task == "WBCIC_MI" else "OpenBMI"], key=subject_sort)
            if task == "WBCIC_MI" and set(cohort) != EXPECTED_WBCIC:
                raise RuntimeError("WBCIC cohort is not final true outer")
            for fold in FOLDS:
                for seed in SEEDS:
                    record = by_key[model, task, fold, seed]
                    root = cell_dir(model, task, fold, seed)
                    detail = json.loads((root / "record.json").read_text(encoding="utf-8"))
                    if sha(root / "selected.pt") != record["checkpoint_sha256"]:
                        raise RuntimeError("checkpoint changed")
                    manifest.append({"model": model, "task": task, "fold": fold, "seed": seed,
                                     "checkpoint_sha256": record["checkpoint_sha256"],
                                     "normalizer_sha256": record["normalizer_sha256"],
                                     "split_sha256": record["split_sha256"],
                                     "selected_epoch": record["selected_epoch"],
                                     "epochs_completed": detail["epochs_completed"],
                                     "early_stopped": detail["early_stopped"],
                                     "parameters": detail["parameters"],
                                     "train_seconds": detail["seconds"],
                                     "checkpoint_complete": True})
                    for session in sessions_for(task):
                        output = cell_output(model, task, fold, seed, session)
                        if not output.is_file():
                            raise RuntimeError(f"missing session cell {output}")
                        payload = json.loads(output.read_text(encoding="utf-8"))
                        expected_identity = {"model": model, "task": task, "fold": fold,
                                             "seed": seed, "session": session}
                        if (payload["identity"] != expected_identity or
                            payload["checkpoint_sha256"] != record["checkpoint_sha256"] or
                            payload["normalizer_sha256"] != record["normalizer_sha256"] or
                            set(row["subject_id"] for row in payload["subject_rows"]) != set(cohort) or
                            len(payload["subject_rows"]) != len(cohort)):
                            raise RuntimeError(f"invalid session cell {output}")
                        for row in payload["subject_rows"]:
                            session_rows.append({"model": model, "task": task,
                                                 "dataset": "WBCIC" if task == "WBCIC_MI" else "OpenBMI",
                                                 "subject_id": row["subject_id"], "session": session,
                                                 "fold": fold, "seed": seed, "BA": row["BA"],
                                                 "macro_F1": row["macro_F1"],
                                                 "accuracy": row["accuracy"], "trials": row["trials"],
                                                 "checkpoint_sha256": record["checkpoint_sha256"],
                                                 "normalizer_sha256": record["normalizer_sha256"]})
    if len(manifest) != 120 or len(session_rows) != 3420:
        raise RuntimeError(f"matrix size mismatch {len(manifest)} {len(session_rows)}")
    return lock, manifest, session_rows


def aggregate(session_rows: list[dict], manifest: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in session_rows:
        grouped[row["model"], row["task"], row["subject_id"], row["session"]].append(row)
    means = {}
    for key, rows in grouped.items():
        if len(rows) != 15 or {(r["fold"], r["seed"]) for r in rows} != {(f, s) for f in FOLDS for s in SEEDS}:
            raise RuntimeError(f"not 15 unique checkpoint estimates {key}")
        means[key] = {metric: float(np.mean([r[metric] for r in rows])) for metric in ("BA", "macro_F1", "accuracy")}
    subject_rows, seed_rows, task_rows = [], [], []
    for model in MODELS:
        for task in TASKS:
            subjects = sorted({key[2] for key in means if key[:2] == (model, task)}, key=subject_sort)
            sessions = sessions_for(task)
            future = sessions[-1]
            for subject in subjects:
                values = [means[model, task, subject, session]["BA"] for session in sessions]
                subject_rows.append({"model": model, "task": task, "subject_id": subject,
                                     "future_BA": means[model, task, subject, future]["BA"],
                                     "future_macro_F1": means[model, task, subject, future]["macro_F1"],
                                     "WS_BA": min(values), "worst_session": sessions[int(np.argmin(values))],
                                     "session_BA": json.dumps(dict(zip(sessions, values)), sort_keys=True),
                                     "n_checkpoints": 15})
            current = [row for row in subject_rows if row["model"] == model and row["task"] == task]
            stats = {metric: bootstrap([row[metric] for row in current]) for metric in ("future_BA", "future_macro_F1", "WS_BA")}
            counts = {row["parameters"] for row in manifest if row["model"] == model and row["task"] == task}
            if len(counts) != 1:
                raise RuntimeError(f"inconsistent parameter count {model} {task}")
            task_rows.append({"model": model, "task": task, "n_subjects": len(subjects),
                              "n_checkpoints": 15, "BA": stats["future_BA"][0],
                              "BA_ci_low": stats["future_BA"][1], "BA_ci_high": stats["future_BA"][2],
                              "macro_F1": stats["future_macro_F1"][0],
                              "macro_F1_ci_low": stats["future_macro_F1"][1],
                              "macro_F1_ci_high": stats["future_macro_F1"][2],
                              "WS_BA": stats["WS_BA"][0], "WS_BA_ci_low": stats["WS_BA"][1],
                              "WS_BA_ci_high": stats["WS_BA"][2],
                              "parameters": next(iter(counts)), "MACs": "not fully counted"})
            for seed in SEEDS:
                vectors = defaultdict(list)
                for row in session_rows:
                    if row["model"] == model and row["task"] == task and row["seed"] == seed:
                        vectors[row["subject_id"], row["session"]].append(row)
                per_subject = []
                for subject in subjects:
                    session_ba = {session: float(np.mean([r["BA"] for r in vectors[subject, session]]))
                                  for session in sessions}
                    per_subject.append({"BA": session_ba[future],
                                        "macro_F1": float(np.mean([r["macro_F1"] for r in vectors[subject, future]])),
                                        "WS_BA": min(session_ba.values())})
                if any(len(vectors[subject, session]) != 5 for subject in subjects for session in sessions):
                    raise RuntimeError("seed-level fold count not five")
                seed_rows.append({"model": model, "task": task, "seed": seed, "folds": 5,
                                  "subjects": len(subjects),
                                  **{metric: float(np.mean([row[metric] for row in per_subject])) for metric in METRICS}})
    return subject_rows, seed_rows, task_rows


def pairwise(subject_rows: list[dict], lock: dict) -> list[dict]:
    source = os.environ.get("FORMAL_SIRE_SUBJECT_ROWS", "")
    sire = {}
    if source:
        with open(source, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("model") != "SIRE-EEG":
                    raise RuntimeError("formal SIRE source identity missing")
                key = row["task"], row["subject_id"]
                if key in sire:
                    raise RuntimeError("duplicate formal SIRE subject")
                sire[key] = row
    current = {(row["model"], row["task"], row["subject_id"]): row for row in subject_rows}
    results = []
    for model in MODELS:
        for task in TASKS:
            cohort = sorted(lock["cohorts"]["WBCIC" if task == "WBCIC_MI" else "OpenBMI"], key=subject_sort)
            for metric in ("future_BA", "future_macro_F1", "WS_BA"):
                base = {subject for dataset, subject in sire if dataset == task}
                if not source:
                    status = "UNAVAILABLE_FORMAL_SIRE_SOURCE_NOT_PROVIDED"
                elif base != set(cohort):
                    status = "UNAVAILABLE_FORMAL_SIRE_COHORT_MISMATCH"
                else:
                    status = "COMPLETE"
                if status == "COMPLETE":
                    delta = [float(current[model, task, subject][metric]) - float(sire[task, subject][metric])
                             for subject in cohort]
                    result = bootstrap(delta)
                else:
                    result = ("", "", "")
                results.append({"model": model, "reference": "SIRE-EEG", "task": task,
                                "metric": metric, "direction": "baseline minus SIRE-EEG",
                                "delta": result[0], "ci_low": result[1], "ci_high": result[2],
                                "n_subjects": len(cohort) if status == "COMPLETE" else "",
                                "bootstrap_draws": DRAW_COUNT, "status": status})
    return results


def main() -> None:
    lock, manifest, session_rows = collect()
    subject_rows, seed_rows, task_rows = aggregate(session_rows, manifest)
    paired = pairwise(subject_rows, lock)
    write_csv("RUN_MANIFEST.csv", manifest)
    write_csv("SESSION_RESULTS.csv", session_rows)
    write_csv("SUBJECT_RESULTS.csv", subject_rows)
    write_csv("SEED_TASK_SUMMARY.csv", seed_rows)
    write_csv("TASK_SUMMARY.csv", task_rows)
    write_csv("PAIRED_EFFECTS_VS_SIRE.csv", paired)
    write_csv("CHECKPOINT_AUDIT.csv", manifest)
    write_csv("MODEL_COMPLEXITY.csv", [{"model": row["model"], "task": row["task"],
                                        "parameters": row["parameters"], "MACs": row["MACs"],
                                        "MAC_status": "not fully counted"} for row in task_rows])
    closure_fields = ("model", "task", "future_BA", "future_BA_ci_low", "future_BA_ci_high",
                      "future_macro_F1", "future_macro_F1_ci_low", "future_macro_F1_ci_high",
                      "WS_BA", "WS_BA_ci_low", "WS_BA_ci_high", "CSGD", "CSGD_ci_low",
                      "CSGD_ci_high", "parameters", "MACs", "complete_3seed", "notes")
    final_rows = []
    for row in task_rows:
        item = {field: "" for field in closure_fields}
        item.update({"model": row["model"], "task": row["task"], "parameters": row["parameters"],
                     "MACs": "not fully counted", "complete_3seed": True,
                     "notes": "true outer; formal SIRE comparison unavailable unless verified source provided"})
        for metric, source in (("future_BA", "BA"), ("future_macro_F1", "macro_F1"), ("WS_BA", "WS_BA")):
            item[metric] = row[source]
            item[metric + "_ci_low"] = row[source + "_ci_low"]
            item[metric + "_ci_high"] = row[source + "_ci_high"]
        final_rows.append(item)
    write_csv("FINAL_BASELINE_ROWS.csv", final_rows)
    completion = {"schema": "EEGCONFORMER_FBCNET_COMPLETION_V1", "pass": True,
                  "expected_training_cells": 120, "completed_training_cells": len(manifest),
                  "missing_training_cells": 120 - len(manifest),
                  "expected_heldout_session_cells": 270,
                  "completed_heldout_session_cells": 270,
                  "model_task_checkpoint_coverage": {f"{model}/{task}": 15 for model in MODELS for task in TASKS},
                  "heldout_used_for_training": False, "heldout_used_for_selection": False,
                  "WBCIC_cohort": "final true outer 10",
                  "formal_SIRE_paired_status": paired[0]["status"],
                  "biological_subject_bootstrap_draws": DRAW_COUNT}
    (OUT / "COMPLETION.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final EEG Conformer / FBCNet report", "",
             "| Model | Task | Future BA [95% CI] | Future Macro-F1 [95% CI] | WS-BA [95% CI] | Params |",
             "|---|---|---:|---:|---:|---:|"]
    for row in task_rows:
        def pct_ci(metric: str) -> str:
            return (f"{row[metric]*100:.2f}% "
                    f"[{row[metric + '_ci_low']*100:.2f}, {row[metric + '_ci_high']*100:.2f}]")
        lines.append(f"| {row['model']} | {row['task']} | {pct_ci('BA')} | "
                     f"{pct_ci('macro_F1')} | {pct_ci('WS_BA')} | {row['parameters']:,} |")
    lines.extend(["", "All eight cells contain 5 folds x 3 seeds (15 selected checkpoints).",
                  "Future-session BA/F1 and WS-BA are subject-equal; 95% CIs use 20,000 biological-subject bootstrap draws.",
                  "Fold/seed estimates are averaged within a subject/session before any bootstrap; WS-BA is the within-subject minimum over sessions.",
                  "", f"Formal SIRE-EEG paired comparison: {paired[0]['status']}. The different local LiteBN is not substituted.",
                  "", "MACs: not fully counted, because the available profiler does not account for the complete fixed filter bank and all attention operations.",
                  "", "Frozen split/normalizer and checkpoint hashes are in RUN_MANIFEST.csv and CHECKPOINT_AUDIT.csv.",
                  "Final WBCIC cohort is the true-outer 10, not the V8 internal cohort. No heldout training or checkpoint selection occurred."])
    (OUT / "FINAL_EEGCONFORMER_FBCNET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ALL_120_TRAINING_AND_270_HELDOUT_SESSIONS_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
