"""Validate amended closure and render qualified manuscript artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
OUT = EXP / "outputs"
PROTO = EXP / "protocol"
PSWA_RUNTIME = P1 / "baseline_metrics_closure_v1_pswa_runtime"
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "LiteBN")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
TRUE_WBCIC = {"sub-4", "sub-8", "sub-10", "sub-15", "sub-20", "sub-39", "sub-40", "sub-43", "sub-46", "sub-51"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, values: list[dict]) -> None:
    if not values:
        raise RuntimeError(f"empty output {path}")
    fields = list(values[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_percent(value: str, n: int = 2) -> str:
    return "NA" if value in ("", None) else f"{100*float(value):.{n}f}"


def metric_cell(row: dict, name: str) -> str:
    return as_percent(row[name]) if row["complete_3seed"] == "True" else "INCOMPLETE"


def validate_pswa(lock: dict) -> tuple[list[dict], dict]:
    if digest(PROTO / "PSWA_EXTENSION_LOCK.json") != (PROTO / "PSWA_EXTENSION_LOCK.sha256").read_text().strip():
        raise RuntimeError("PSWA extension lock altered")
    if not lock["published_regression_pass"] or len(lock["published_regression"]) != 8:
        raise RuntimeError("published PSWA regression missing")
    published = {(row["model"], row["task"]): row for row in rows(PROTO / "PSWA_PUBLISHED_REFERENCE.csv")
                 if row["model"] in ("EEGNet", "TeCh")}
    aggregate = {(row["Model"], row["Task"]): row for row in rows(
        OUT / "crossbackbone_pswa_v1" / "CROSSBACKBONE_PSWA_SEED0.csv")}
    for key, reference in published.items():
        value = aggregate[key]
        for left, right in (("PSWA_pp", "PSWA mean"), ("CI_low_pp", "PSWA CI low"),
                            ("CI_high_pp", "PSWA CI high")):
            if abs(float(reference[left]) - float(value[right])) > 1e-10:
                raise RuntimeError(f"published PSWA regression changed: {key}")
        if reference["status"] != value["Status"] or reference["coverage"] != value["Protected coverage"]:
            raise RuntimeError(f"published PSWA status changed: {key}")
    output = []
    audit = {"nonempty": 0, "empty": 0, "cells": 0}
    for model in ("ModernTCN", "Medformer"):
        for task in TASKS:
            expected = [cell for cell in lock["cells"] if cell["model"] == model and cell["task"] == task]
            if len(expected) != 5:
                raise RuntimeError(f"PSWA lock fold matrix incomplete {model} {task}")
            rank_by_fold = []
            subjects = set()
            for fixed in sorted(expected, key=lambda item: item["fold"]):
                path = (PSWA_RUNTIME / "cells" / model.lower() / task.lower()
                        / f"fold{fixed['fold']}_seed0.json")
                if not path.is_file():
                    raise RuntimeError(f"PSWA cell missing {path}")
                cell = json.loads(path.read_text(encoding="utf-8"))
                audit["cells"] += 1
                rank_by_fold.append(fixed["protected_rank"])
                if fixed["status"] == "EMPTY_UNDEFINED_PSWA":
                    if cell["status"] != "EMPTY_PROTECTED":
                        raise RuntimeError(f"empty Protected assignment became a numeric PSWA: {path}")
                    audit["empty"] += 1
                    continue
                if cell["status"] != "RECOVERY_OK":
                    raise RuntimeError(f"PSWA cell not reproducibly recovered: {path} status={cell['status']}")
                if cell["peeh_cell_sha256"] != fixed["peeh_cell_sha256"]:
                    raise RuntimeError(f"PEEH assignment hash mismatch: {path}")
                if cell["checkpoint_sha256"] != fixed["checkpoint_sha256"] or cell["normalizer_sha256"] != fixed["normalizer_sha256"]:
                    raise RuntimeError(f"frozen model/normalizer mismatch: {path}")
                if cell["protected_indices"] != fixed["protected_indices"]:
                    raise RuntimeError(f"Protected coordinates changed: {path}")
                controls = hashlib.sha256(json.dumps(cell["random_controls"], separators=(",", ":")).encode()).hexdigest()
                if controls != fixed["random_controls_sha256"] or len(cell["random_controls"]) != 100:
                    raise RuntimeError(f"exact random controls changed: {path}")
                expected_subjects = TRUE_WBCIC if task == "WBCIC_MI" else set(json.loads(
                    (PROTO / "FROZEN_INFERENCE_LOCK.json").read_text(encoding="utf-8"))["cohorts"]["OpenBMI"]["subjects"])
                actual = {str(item["subject_id"]) for item in cell["protected_session_results"]}
                if actual != expected_subjects:
                    raise RuntimeError(f"PSWA heldout subject set mismatch: {path}")
                subjects.update(actual)
                audit["nonempty"] += 1
            value = aggregate[model, task]
            coverage = f"{sum(rank > 0 for rank in rank_by_fold)}/5"
            if value["Status"] != "COMPLETE" or value["Protected coverage"] != coverage:
                raise RuntimeError(f"PSWA aggregate coverage mismatch: {model} {task}")
            output.append({"model": model, "task": task, "seed": 0, "coverage": coverage,
                           "protected_rank_by_fold": ";".join(map(str, rank_by_fold)),
                           "PSWA_pp": value["PSWA mean"], "CI_low_pp": value["PSWA CI low"],
                           "CI_high_pp": value["PSWA CI high"], "n_subjects": len(subjects),
                           "n_random_controls": 100, "status": "COMPLETE_WITH_EMPTY_FOLDS_EXCLUDED"})
    if audit["cells"] != 40 or audit["nonempty"] + audit["empty"] != 40:
        raise RuntimeError("PSWA cell count mismatch")
    return output, audit


def main() -> None:
    if not (P1 / "baseline_metrics_closure_v1_runtime" / "INFERENCE_COMPLETED.txt").is_file():
        raise RuntimeError("frozen full-model inference queue not complete")
    if not (P1 / "baseline_metrics_closure_v1_native_batch_runtime" / "INFERENCE_COMPLETED.txt").is_file():
        raise RuntimeError("native-batch ModernTCN/Medformer inference queue not complete")
    if not (PSWA_RUNTIME / "PSWA_COMPLETED.txt").is_file():
        raise RuntimeError("PSWA extension queue not complete")
    full = rows(OUT / "FINAL_FULLMODEL_METRICS.csv")
    sessions = rows(OUT / "SESSION_METRIC_SUMMARY.csv")
    coverage = rows(OUT / "FROZEN_SESSION_COVERAGE.csv")
    macs = {(row["model"], row["task"]): row for row in rows(OUT / "MACS_AUDIT.csv")}
    regression = rows(OUT / "TRUE_OUTER_FUTURE_REGRESSION_AUDIT.csv")
    independent = rows(OUT / "TRUE_OUTER_SEED0_INDEPENDENT_REGRESSION.csv")
    scope = json.loads((PROTO / "TRUE_OUTER_REGRESSION_SCOPE.json").read_text(encoding="utf-8"))
    precision = json.loads((PROTO / "NATIVE_BATCH_PRECISION_AUDIT.json").read_text(encoding="utf-8"))
    pswa_lock = json.loads((PROTO / "PSWA_EXTENSION_LOCK.json").read_text(encoding="utf-8"))
    if len(full) != 24 or len(coverage) != 24 or len(regression) != 6:
        raise RuntimeError("main model-task matrix or OpenBMI regression incomplete")
    if any(row["status"] != "PASS" for row in regression):
        raise RuntimeError("OpenBMI future-session regression failed")
    if len(independent) != 100 or any(row["status"] != "PASS" for row in independent):
        raise RuntimeError("independent WBCIC true-outer seed0 regression failed")
    if scope["status"] != "SEED0_INDEPENDENT_PASS":
        raise RuntimeError("WBCIC regression scope unverified")
    if precision["OpenBMI_exact_rows"] != precision["OpenBMI_total_rows"]:
        raise RuntimeError("native-batch OpenBMI subject-level regression incomplete")
    complete = {(row["model"], row["task"]) for row in coverage if row["complete_3seed"] == "True"}
    expected_complete = {(model, task) for model in MODELS for task in TASKS} - {
        ("LiteBN", "OpenBMI_SSVEP"), ("LiteBN", "WBCIC_MI")}
    if complete != expected_complete:
        raise RuntimeError(f"unexpected primary completion matrix: {sorted(complete ^ expected_complete)}")
    frozen_lock = json.loads((PROTO / "FROZEN_INFERENCE_LOCK.json").read_text(encoding="utf-8"))
    if set(frozen_lock["cohorts"]["WBCIC"]["subjects"]) != TRUE_WBCIC:
        raise RuntimeError("wrong final WBCIC cohort")
    pswa, pswa_audit = validate_pswa(pswa_lock)
    write_rows(OUT / "PSWA_MODERNTCN_MEDFORMER.csv", pswa)
    for row in full:
        m = macs[row["model"], row["task"]]
        if m["status"] == "COUNTED_ATEN_CONV_MATMUL":
            row["MACs"] = m["MACs"]
        else:
            row["MACs"] = ""
            row["notes"] += f";MACs={m['status']}"
        if row["parameters"] and int(row["parameters"]) != int(m["trainable_parameters"]):
            raise RuntimeError(f"trainable parameter mismatch: {row['model']} {row['task']}")
    write_rows(OUT / "FINAL_FULLMODEL_METRICS.csv", full)

    index = {(row["model"], row["task"]): row for row in full}
    predictive = ["# Final true-outer predictive metrics", "",
                  "Values are subject-equal percentages; each complete cell averages 15 independent frozen checkpoints.",
                  "WBCIC uses the disjoint final true-outer 10, never the V8 internal 10.", "",
                  "| Method | Task | BA | Macro-F1 | WS-BA | Completion |",
                  "|---|---|---:|---:|---:|---|"]
    for model in MODELS:
        for task in TASKS:
            row = index[model, task]
            predictive.append(f"| {model} | {task} | {metric_cell(row, 'future_BA')} | "
                              f"{metric_cell(row, 'future_macro_F1')} | {metric_cell(row, 'WS_BA')} | "
                              f"{'15/15' if row['complete_3seed']=='True' else 'INCOMPLETE'} |")
    (OUT / "MAIN_PREDICTIVE_TABLE.md").write_text("\n".join(predictive) + "\n", encoding="utf-8")

    efficiency = ["# Efficiency and robustness", "",
                  "Params are recorded trainable parameters. MACs use one verified Conv/Matmul convention;",
                  "CBraMod contains an FFT major operator not counted by that profiler, so its MACs are not estimated.", "",
                  "| Method | Task | Params | MACs (M) | WS-BA (%) | CSGD (pp) |",
                  "|---|---|---:|---:|---:|---:|"]
    for model in MODELS:
        for task in TASKS:
            row = index[model, task]
            m = macs[model, task]
            efficiency.append(f"| {model} | {task} | {row['parameters'] or 'NA'} | "
                              f"{float(row['MACs'])/1e6:.2f}" if row["MACs"] else
                              f"| {model} | {task} | {row['parameters'] or 'NA'} | NE")
            efficiency[-1] += f" | {metric_cell(row, 'WS_BA')} | {metric_cell(row, 'CSGD')} |"
    (OUT / "EFFICIENCY_ROBUSTNESS_TABLE.md").write_text("\n".join(efficiency) + "\n", encoding="utf-8")

    by_model = []
    for model in MODELS:
        row = [model]
        for task in TASKS:
            value = index[model, task]
            row.extend(metric_cell(value, field) for field in ("future_BA", "future_macro_F1", "WS_BA"))
        by_model.append("| " + " | ".join(row) + " |")
    main_table = ["# Main benchmark table: final true outer", "",
                  "All numeric cells are subject-equal percent. `INC` means incomplete 3-seed checkpoint matrix.",
                  "CBraMod MACs are not reported because FFT is unsupported by the unified counter.", "",
                  "| Method | MI BA | MI F1 | MI WS | ERP BA | ERP F1 | ERP WS | SSVEP BA | SSVEP F1 | SSVEP WS | WBCIC BA | WBCIC F1 | WBCIC WS |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"] + by_model
    (OUT / "MAIN_BENCHMARK_TABLE.md").write_text("\n".join(main_table) + "\n", encoding="utf-8")

    report = ["# Baseline metrics closure — amended final true-outer scope", "",
              "Status: **AMENDED-SCOPE ANALYSIS COMPLETE; full 6×4 three-seed table remains incomplete.**", "",
              "## A. Completion audit", "",
              "ModernTCN, Medformer, EEGNet, CBraMod and TeCh each have 15/15 verified frozen checkpoints in all four tasks. "
              "LiteBN is 15/15 for OpenBMI MI and ERP, 13/15 for SSVEP and 5/15 for WBCIC MI. "
              "The last two cells are not presented as full three-seed results. No training or checkpoint reselection occurred.", "",
              "## B–D. Unified retained-baseline benchmark", "",
              "See `MAIN_PREDICTIVE_TABLE.md` and `FINAL_FULLMODEL_METRICS.csv`. "
              "The table uses 14 OpenBMI final heldout subjects and 10 WBCIC final true-outer subjects. "
              "The originally supplied WBCIC numerical targets belonged to the disjoint V8 internal cohort; "
              "they were retired as final-outer targets by the approved amendment.", "",
              *predictive[5:], "",
              "## B. ModernTCN frozen-session detail", "",
              "| Task | Session | BA (%) | Macro-F1 (%) | WS-BA (%) | CSGD (pp) | Params | MACs (M) |",
              "|---|---|---:|---:|---:|---:|---:|---:|",
              *[f"| {row['task']} | {row['session']} | {as_percent(row['BA'])} | {as_percent(row['macro_F1'])} | "
                f"{as_percent(index['ModernTCN', row['task']]['WS_BA'])} | "
                f"{as_percent(index['ModernTCN', row['task']]['CSGD'])} | "
                f"{index['ModernTCN', row['task']]['parameters']} | "
                f"{float(macs['ModernTCN', row['task']]['MACs_M']):.2f} |"
                for row in sessions if row['model'] == 'ModernTCN'], "",
              "## C. Medformer frozen-session detail", "",
              "| Task | Session | BA (%) | Macro-F1 (%) | WS-BA (%) | CSGD (pp) | Params | MACs (M) |",
              "|---|---|---:|---:|---:|---:|---:|---:|",
              *[f"| {row['task']} | {row['session']} | {as_percent(row['BA'])} | {as_percent(row['macro_F1'])} | "
                f"{as_percent(index['Medformer', row['task']]['WS_BA'])} | "
                f"{as_percent(index['Medformer', row['task']]['CSGD'])} | "
                f"{index['Medformer', row['task']]['parameters']} | "
                f"{float(macs['Medformer', row['task']]['MACs_M']):.2f} |"
                for row in sessions if row['model'] == 'Medformer'], "",
              "## E. Representation-only PSWA extension", "",
              "Published EEGNet/TeCh rows reproduce exactly (8/8). The exact published recovery script, "
              "frozen PEEH assignments, and deterministic original random controls were used. "
              "Empty Protected assignments are excluded as undefined, not converted to zero.", "",
              "| Model | Task | Coverage | PSWA pp [95% subject CI] |", "|---|---|---:|---:|"]
    for row in pswa:
        report.append(f"| {row['model']} | {row['task']} | {row['coverage']} | "
                      f"{float(row['PSWA_pp']):.2f} [{float(row['CI_low_pp']):.2f}, {float(row['CI_high_pp']):.2f}] |")
    report += ["", "## F. Statistical uncertainty", "",
               "20,000 deterministic percentile bootstrap draws resample biological subjects only. "
               "Checkpoint/session replicates are averaged within a subject first. Paired LiteBN contrasts are "
               "reported only for complete identical-cohort cells; a CI crossing zero is not equivalence.", "",
               "## G. Leakage and regression audit", "",
               "OpenBMI future-session frozen re-evaluation matches all six pre-existing numerical targets within 1e-8. "
               "For WBCIC, an independent frozen seed0 rerun matches 100 committed true-outer subject/fold rows; "
               "there is no independent published 15-checkpoint true-outer target, so this remains a documented scope limitation. "
               "All per-session rows retain checkpoint and normalizer hashes. WBCIC file S0/S1/S2 maps to paper S1/S2/S3.", "",
               f"Native-batch OpenBMI per-subject future rows match the published evaluator exactly "
               f"({precision['OpenBMI_exact_rows']}/{precision['OpenBMI_total_rows']}). "
               f"WBCIC seed0 comparison to the old batch-32 CSGD artifact is exact for "
               f"{precision['WBCIC_batch32_reference_exact_rows']}/{precision['WBCIC_seed0_total_rows']} rows; "
               f"maximum BA difference is {precision['WBCIC_max_abs_BA_difference']:.8f}.", "",
               "ModernTCN/Medformer were re-evaluated in a preserved, separate runtime using the original frozen "
               "evaluators' batch size 128. A single Medformer ERP subject/checkpoint prediction differed at CSGD batch 32; "
               "batch 128 reproduces the published OpenBMI target exactly. The batch-32 runtime was retained for audit.", "",
               "Source commits: seven-backbone true-outer `097c7f5016fb6690800c57452eb1ec9657a2fc7b`; "
               "ModernTCN `a9c6a6a4ecd363242a5cd7a0608abe3ae7d73b79`; "
               "Medformer `68ef63a182ee250b0b6865182ed94fcebec6e89d`; "
               "cross-backbone CSGD `4020f0bb0a24d5ef70bb9ba0e2116b7b0f257ebb`; "
               "PEEH `bd2cc4346041d16f544963703e02bf93e262fbd3`; "
               "published PSWA `3fd6aad50c08144c95a58454ca76d58f21aedc25`. "
               "Per-cell hashes are in the frozen inference and PSWA locks.", "",
               "## H. Manuscript recommendation and remaining gaps", "",
               "Use future BA, macro-F1, WS-BA and Params in the main predictive table. Put session-wise BA/F1, CSGD, "
               "subject bootstrap CIs, paired contrasts and MAC audit in appendix. PSWA is a representation diagnostic, "
               "not a substitute for full-model WS-BA. Do not claim a complete six-model four-task three-seed table: "
               "LiteBN has two incomplete cells. Do not report CBraMod MACs as numeric under the current profiler: FFT is uncounted.", "",
               f"PSWA extension cells: {pswa_audit['cells']}; nonempty: {pswa_audit['nonempty']}; "
               f"empty: {pswa_audit['empty']}.", ""]
    (OUT / "BASELINE_METRICS_CLOSURE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    validation = {"pass_amended_scope": True, "full_6x4_3seed_complete": False,
                  "complete_primary_cells": len(complete), "incomplete_primary_cells": 24-len(complete),
                  "OpenBMI_regression_pass": True, "WBCIC_true_outer_seed0_independent_regression_pass": True,
                  "WBCIC_full_15_independent_regression_available": False,
                  "PSWA_published_regression_rows": 8, "PSWA_extension_cells": pswa_audit,
                  "MAC_numeric_cells": sum(row["status"] == "COUNTED_ATEN_CONV_MATMUL" for row in macs.values()),
                  "MAC_not_estimable_cells": sum(row["status"] != "COUNTED_ATEN_CONV_MATMUL" for row in macs.values()),
                  "training_performed": False, "checkpoint_reselection": False,
                  "WBCIC_final_true_outer_subjects": sorted(TRUE_WBCIC),
                  "terminal": "AMENDED_BASELINE_CLOSURE_COMPLETE_WITH_EXPLICIT_INCOMPLETE_CELLS"}
    (OUT / "CLOSURE_VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(validation["terminal"], "complete_cells=", len(complete), "PSWA=", len(pswa))


if __name__ == "__main__":
    main()
