"""Aggregate frozen true-outer rows with biological-subject uncertainty.

Checkpoint predictions are never ensembled. The order for WS-BA is:
checkpoint mean within subject/session -> session minimum -> subject mean.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
RUNTIME = P1 / "baseline_metrics_closure_v1_runtime"
NATIVE_RUNTIME = P1 / "baseline_metrics_closure_v1_native_batch_runtime"
LOCK = EXP / "protocol" / "FROZEN_INFERENCE_LOCK.json"
LOCK_SHA = LOCK.with_suffix(".sha256")
OUT = EXP / "outputs"
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "LiteBN")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
BOOT_SEED = 20260916
N_BOOT = 20_000
EXPECTED_OPENBMI = {
    ("ModernTCN", "OpenBMI_MI"): (0.616952380952381, 0.6131284240652964),
    ("ModernTCN", "OpenBMI_ERP"): (0.7981471861471863, 0.7683683543195192),
    ("ModernTCN", "OpenBMI_SSVEP"): (0.8441904761904762, 0.8424344193334482),
    ("Medformer", "OpenBMI_MI"): (0.6715714285714287, 0.6663734978222521),
    ("Medformer", "OpenBMI_ERP"): (0.8007344877344877, 0.7889534338698844),
    ("Medformer", "OpenBMI_SSVEP"): (0.8860952380952379, 0.8846645384445041),
}
TOLERANCE = 1e-8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refuse empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError(f"CSV schema mismatch: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_lock() -> dict:
    if digest(LOCK) != LOCK_SHA.read_text(encoding="utf-8").strip():
        raise RuntimeError("inference lock hash mismatch")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["training_performed"] or lock["checkpoint_reselection"]:
        raise RuntimeError("inference-only lock violated")
    return lock


def session_path(model: str, task: str, fold: int, seed: int, session_index: int) -> Path:
    runtime = NATIVE_RUNTIME if model in ("ModernTCN", "Medformer") else RUNTIME
    return (runtime / "session_cells" / model.lower() / task.lower()
            / f"fold{fold}_seed{seed}_session{session_index}.json")


def session_map(task: str) -> dict[int, str]:
    return {0: "S1", 1: "S2", 2: "S3"} if task == "WBCIC_MI" else {1: "S1", 2: "S2"}


def validated_rows(lock: dict) -> tuple[list[dict], list[dict]]:
    records = {(row["Model"], row["Task"], int(row["fold"]), int(row["seed"])): row
               for row in lock["evaluation_checkpoints"]}
    all_rows: list[dict] = []
    coverage: list[dict] = []
    for model in MODELS:
        for task in TASKS:
            keys = [(model, task, fold, seed) for fold in range(5) for seed in range(3)]
            missing_ckpt = [key for key in keys if key not in records]
            subjects = set(lock["cohorts"]["WBCIC" if task == "WBCIC_MI" else "OpenBMI"]["subjects"])
            expected_sessions = session_map(task)
            missing_session = []
            current: list[dict] = []
            for key in keys:
                if key not in records:
                    continue
                record = records[key]
                _, _, fold, seed = key
                for index, paper_session in expected_sessions.items():
                    path = session_path(model, task, fold, seed, index)
                    if not path.is_file():
                        missing_session.append(str(path))
                        continue
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value["identity"] != {"Model": model, "Task": task, "fold": fold,
                                               "seed": seed, "session_index": index}:
                        raise RuntimeError(f"session identity mismatch: {path}")
                    if value["checkpoint_sha256"] != record["checkpoint_sha256"]:
                        raise RuntimeError(f"checkpoint hash mismatch: {path}")
                    if value["normalizer_sha256"] != record["normalizer_sha256"]:
                        raise RuntimeError(f"normalizer hash mismatch: {path}")
                    if value["session"] != (f"S{index}"):
                        raise RuntimeError(f"file session mismatch: {path}")
                    if model in ("ModernTCN", "Medformer") and int(value["batch_size"]) != 128:
                        raise RuntimeError(f"native frozen evaluator batch mismatch: {path}")
                    data = value["subject_rows"]
                    actual_subjects = {str(row["subject"]) for row in data}
                    if len(data) != len(subjects) or actual_subjects != subjects:
                        raise RuntimeError(f"biological subject mismatch: {path}")
                    for subject_row in data:
                        row = {
                            "model": model, "dataset": "WBCIC" if task == "WBCIC_MI" else "OpenBMI",
                            "task": task, "subject_id": str(subject_row["subject"]),
                            "session": paper_session, "fold": fold, "seed": seed,
                            "BA": float(subject_row["BA"]),
                            "macro_F1": float(subject_row["macro_F1"]),
                            "accuracy": float(subject_row["accuracy"]),
                            "trials": int(subject_row["trials"]),
                            "checkpoint_identifier": str(record["checkpoint_path"]),
                            "checkpoint_hash": record["checkpoint_sha256"],
                            "normalizer_identifier": f"{task}:fold{fold}:train_only_channelwise",
                            "normalizer_hash": record["normalizer_sha256"],
                        }
                        if not all(np.isfinite(row[field]) for field in ("BA", "macro_F1", "accuracy")):
                            raise RuntimeError(f"non-finite metric: {path}")
                        current.append(row)
            unique = {(row["model"], row["task"], row["subject_id"], row["session"],
                       row["fold"], row["seed"]) for row in current}
            if len(unique) != len(current):
                raise RuntimeError(f"duplicate subject/session/checkpoint: {model} {task}")
            expected_count = len(subjects) * len(expected_sessions) * 15
            complete = not missing_ckpt and not missing_session and len(current) == expected_count
            coverage.append({
                "model": model, "task": task, "checkpoints": 15 - len(missing_ckpt),
                "expected_checkpoints": 15, "session_rows": len(current),
                "expected_session_rows": expected_count, "n_subjects": len(subjects),
                "complete_3seed": complete,
                "status": "COMPLETE" if complete else "INCOMPLETE",
                "notes": f"missing_checkpoint={len(missing_ckpt)};missing_session_files={len(missing_session)}",
            })
            all_rows.extend(current)
    return all_rows, coverage


def subject_means(rows: list[dict], coverage: list[dict]) -> list[dict]:
    complete = {(row["model"], row["task"]) for row in coverage if row["complete_3seed"]}
    grouped = defaultdict(list)
    for row in rows:
        if (row["model"], row["task"]) in complete:
            grouped[row["model"], row["task"], row["subject_id"], row["session"]].append(row)
    result = []
    for (model, task, subject, session), source in sorted(grouped.items()):
        if len(source) != 15 or {(row["fold"], row["seed"]) for row in source} != {
                (fold, seed) for fold in range(5) for seed in range(3)}:
            raise RuntimeError(f"incomplete subject replicate matrix: {model} {task} {subject} {session}")
        result.append({
            "model": model, "task": task, "subject_id": subject, "session": session,
            "BA": float(np.mean([row["BA"] for row in source])),
            "macro_F1": float(np.mean([row["macro_F1"] for row in source])),
            "accuracy": float(np.mean([row["accuracy"] for row in source])),
            "n_checkpoints": 15,
        })
    return result


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) not in (10, 14) or not np.all(np.isfinite(data)):
        raise RuntimeError("invalid biological-subject bootstrap vector")
    rng = np.random.default_rng(BOOT_SEED)
    indices = rng.integers(0, len(data), size=(N_BOOT, len(data)))
    draws = data[indices].mean(axis=1)
    return float(data.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def regression_gate(subject_rows: list[dict], coverage: list[dict]) -> list[dict]:
    complete = {(row["model"], row["task"]) for row in coverage if row["complete_3seed"]}
    results = []
    for (model, task), (target_ba, target_f1) in EXPECTED_OPENBMI.items():
        if (model, task) not in complete:
            results.append({"model": model, "task": task, "status": "INCOMPLETE_STOP",
                            "expected_BA": target_ba, "observed_BA": "", "expected_macro_F1": target_f1,
                            "observed_macro_F1": "", "max_abs_error": ""})
            continue
        source = [row for row in subject_rows if row["model"] == model and row["task"] == task and row["session"] == "S2"]
        if len(source) != 14:
            raise RuntimeError(f"wrong OpenBMI final cohort: {model} {task}")
        observed_ba = float(np.mean([row["BA"] for row in source]))
        observed_f1 = float(np.mean([row["macro_F1"] for row in source]))
        error = max(abs(observed_ba - target_ba), abs(observed_f1 - target_f1))
        results.append({"model": model, "task": task,
                        "status": "PASS" if error <= TOLERANCE else "NUMERIC_MISMATCH_STOP",
                        "expected_BA": target_ba, "observed_BA": observed_ba,
                        "expected_macro_F1": target_f1, "observed_macro_F1": observed_f1,
                        "max_abs_error": error})
    return results


def summarize(subject_rows: list[dict], coverage: list[dict], lock: dict) -> None:
    by_subject = defaultdict(dict)
    for row in subject_rows:
        by_subject[row["model"], row["task"], row["subject_id"]][row["session"]] = row
    session_summary, ws_rows, csgd_rows, final_rows = [], [], [], []
    for model in MODELS:
        for task in TASKS:
            current_coverage = next(row for row in coverage if row["model"] == model and row["task"] == task)
            if not current_coverage["complete_3seed"]:
                known_params = {int(row["trainable_parameters"]) for row in lock["evaluation_checkpoints"]
                                if row["Model"] == model and row["Task"] == task}
                final_rows.append({"model": model, "task": task, "future_BA": "", "future_BA_ci_low": "",
                                   "future_BA_ci_high": "", "future_macro_F1": "", "future_macro_F1_ci_low": "",
                                   "future_macro_F1_ci_high": "", "WS_BA": "", "WS_BA_ci_low": "",
                                   "WS_BA_ci_high": "", "CSGD": "", "CSGD_ci_low": "", "CSGD_ci_high": "",
                                   "parameters": next(iter(known_params)) if len(known_params) == 1 else "",
                                   "MACs": "", "complete_3seed": False,
                                   "notes": current_coverage["notes"]})
                continue
            subjects = list(lock["cohorts"]["WBCIC" if task == "WBCIC_MI" else "OpenBMI"]["subjects"])
            sessions = tuple(session_map(task).values())
            values = {(subject, session): by_subject[model, task, subject][session]
                      for subject in subjects for session in sessions}
            summaries = {}
            for session in sessions:
                ba = bootstrap([values[subject, session]["BA"] for subject in subjects])
                f1 = bootstrap([values[subject, session]["macro_F1"] for subject in subjects])
                summaries[session] = (ba, f1)
                session_summary.append({"model": model, "task": task, "session": session,
                                        "BA": ba[0], "BA_ci_low": ba[1], "BA_ci_high": ba[2],
                                        "macro_F1": f1[0], "macro_F1_ci_low": f1[1], "macro_F1_ci_high": f1[2],
                                        "n_subjects": len(subjects), "n_checkpoints": 15})
            ws_values, drop_values = [], []
            for subject in subjects:
                ba_by_session = {session: values[subject, session]["BA"] for session in sessions}
                worst = min(sessions, key=lambda session: ba_by_session[session])
                worst_ba = ba_by_session[worst]
                drop = ((ba_by_session["S1"] + ba_by_session["S2"]) / 2 - ba_by_session["S3"]
                        if task == "WBCIC_MI" else ba_by_session["S1"] - ba_by_session["S2"])
                ws_values.append(worst_ba)
                drop_values.append(drop)
                ws_rows.append({"model": model, "task": task, "subject_id": subject,
                                "session1_mean_BA": ba_by_session["S1"],
                                "session2_mean_BA": ba_by_session["S2"],
                                "session3_mean_BA_if_applicable": ba_by_session.get("S3", ""),
                                "worst_session": worst, "WS_BA_subject": worst_ba})
                csgd_rows.append({"model": model, "task": task, "subject_id": subject, "CSGD_subject": drop})
            ws = bootstrap(ws_values)
            csgd = bootstrap(drop_values)
            future = summaries[sessions[-1]]
            params = {int(row["trainable_parameters"]) for row in lock["evaluation_checkpoints"]
                      if row["Model"] == model and row["Task"] == task}
            if len(params) != 1:
                raise RuntimeError(f"parameter count varies across frozen checkpoints: {model} {task} {params}")
            final_rows.append({"model": model, "task": task,
                               "future_BA": future[0][0], "future_BA_ci_low": future[0][1],
                               "future_BA_ci_high": future[0][2],
                               "future_macro_F1": future[1][0], "future_macro_F1_ci_low": future[1][1],
                               "future_macro_F1_ci_high": future[1][2],
                               "WS_BA": ws[0], "WS_BA_ci_low": ws[1], "WS_BA_ci_high": ws[2],
                               "CSGD": csgd[0], "CSGD_ci_low": csgd[1], "CSGD_ci_high": csgd[2],
                               "parameters": next(iter(params)), "MACs": "", "complete_3seed": True,
                               "notes": ""})
    write_csv(OUT / "SESSION_METRIC_SUMMARY.csv", session_summary)
    write_csv(OUT / "WSBA_SUBJECT_RESULTS.csv", ws_rows)
    write_csv(OUT / "CSGD_SUBJECT_RESULTS.csv", csgd_rows)
    write_csv(OUT / "FINAL_FULLMODEL_METRICS.csv", final_rows)
    paired_rows = []
    session_lookup = {(row["model"], row["task"], row["subject_id"], row["session"]): row
                      for row in subject_rows}
    ws_lookup = {(row["model"], row["task"], row["subject_id"]): row["WS_BA_subject"]
                 for row in ws_rows}
    complete = {(row["model"], row["task"]) for row in coverage if row["complete_3seed"]}
    for task in TASKS:
        future_name = tuple(session_map(task).values())[-1]
        subjects = list(lock["cohorts"]["WBCIC" if task == "WBCIC_MI" else "OpenBMI"]["subjects"])
        for baseline in MODELS:
            if baseline == "LiteBN":
                continue
            for metric in ("future_BA", "future_macro_F1", "WS_BA"):
                if ("LiteBN", task) not in complete or (baseline, task) not in complete:
                    paired_rows.append({"baseline": baseline, "task": task, "metric": metric,
                                        "delta_LiteBN_minus_baseline": "", "CI_low": "", "CI_high": "",
                                        "n_subjects": "", "bootstrap_draws": N_BOOT,
                                        "status": "NOT_ESTIMABLE_INCOMPLETE_3SEED_CELL"})
                    continue
                if metric == "WS_BA":
                    differences = [ws_lookup["LiteBN", task, subject] - ws_lookup[baseline, task, subject]
                                   for subject in subjects]
                else:
                    key = "BA" if metric == "future_BA" else "macro_F1"
                    differences = [session_lookup["LiteBN", task, subject, future_name][key]
                                   - session_lookup[baseline, task, subject, future_name][key]
                                   for subject in subjects]
                estimate = bootstrap(differences)
                paired_rows.append({"baseline": baseline, "task": task, "metric": metric,
                                    "delta_LiteBN_minus_baseline": estimate[0],
                                    "CI_low": estimate[1], "CI_high": estimate[2],
                                    "n_subjects": len(subjects), "bootstrap_draws": N_BOOT, "status": "COMPLETE"})
    write_csv(OUT / "PAIRED_LITEBN_BASELINE_CI.csv", paired_rows)
    (EXP / "protocol" / "BOOTSTRAP_PROTOCOL.json").write_text(json.dumps({
        "draws": N_BOOT, "seed": BOOT_SEED, "unit": "biological_subject",
        "interval": "percentile_2.5_97.5", "replicate_handling": "average_15_frozen_checkpoints_before_subject_bootstrap",
        "WS_BA_order": "checkpoint_mean_then_session_minimum_then_subject_mean",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    lock = load_lock()
    raw, coverage = validated_rows(lock)
    write_csv(OUT / "FROZEN_SESSION_COVERAGE.csv", coverage)
    write_csv(OUT / "HELDOUT_SUBJECT_SESSION_RESULTS.csv", raw)
    subject_rows = subject_means(raw, coverage)
    if not subject_rows:
        raise RuntimeError("no complete model/task to summarize")
    write_csv(OUT / "SUBJECT_SESSION_CHECKPOINT_MEANS.csv", subject_rows)
    gate = regression_gate(subject_rows, coverage)
    write_csv(OUT / "TRUE_OUTER_FUTURE_REGRESSION_AUDIT.csv", gate)
    bad = [row for row in gate if row["status"] != "PASS"]
    if bad:
        raise RuntimeError(f"OpenBMI frozen future regression failed before WS-BA: {bad}")
    summarize(subject_rows, coverage, lock)
    print("FROZEN_METRICS_AGGREGATED complete_cells=", sum(row["complete_3seed"] for row in coverage))


if __name__ == "__main__":
    main()
