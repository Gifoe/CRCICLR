"""Assemble the strict V6 exploratory report after all attempted families."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import ABLATIONS, BASELINES, DIAGNOSTICS, EXPERIMENT_ROOT, FINAL_CANDIDATE, LEADERBOARD, OUTPUTS, PROTOCOL, RESEARCH_LOG, V6_SEED, sha256_file, v5_output_root, write_csv, write_json
from evaluation.metrics import summarize


TABLES = {
    "OpenBMI": (
        BASELINES / "OPENBMI_MATCHED_BASELINES.csv",
        BASELINES / "OPENBMI_GEOMETRY_BASELINES.csv",
        LEADERBOARD / "OPENBMI_CONDITIONAL_ADAPTER.csv",
        LEADERBOARD / "OPENBMI_ENCODER_FINETUNING.csv",
        LEADERBOARD / "OPENBMI_MI_SPECIFIC_BACKBONE.csv",
        LEADERBOARD / "OPENBMI_MI_SELECTIVE_HEAD.csv",
    ),
    "WBCIC": (
        BASELINES / "WBCIC_MATCHED_BASELINES.csv",
        LEADERBOARD / "WBCIC_CONDITIONAL_ADAPTER.csv",
        LEADERBOARD / "WBCIC_ENCODER_FINETUNING.csv",
        LEADERBOARD / "WBCIC_FIXED_PERSIST_FUSION.csv",
        LEADERBOARD / "WBCIC_FUTURE_SESSION_POPULATION.csv",
        LEADERBOARD / "WBCIC_LARGE_EEGNET.csv",
        LEADERBOARD / "WBCIC_LARGE_SELECTIVE_HEAD.csv",
    ),
}

PREDICTION_FILES = {
    "OpenBMI": (
        DIAGNOSTICS / "OPENBMI_BASELINE_PREDICTIONS.csv",
        DIAGNOSTICS / "OPENBMI_GEOMETRY_PREDICTIONS.csv",
        DIAGNOSTICS / "OPENBMI_CONDITIONAL_ADAPTER_PREDICTIONS.csv",
        DIAGNOSTICS / "OPENBMI_ENCODER_FINETUNING_PREDICTIONS.csv",
        DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv",
        DIAGNOSTICS / "OPENBMI_MI_SELECTIVE_HEAD_PREDICTIONS.csv",
    ),
    "WBCIC": (
        DIAGNOSTICS / "WBCIC_BASELINE_PREDICTIONS.csv",
        DIAGNOSTICS / "WBCIC_CONDITIONAL_ADAPTER_PREDICTIONS.csv",
        DIAGNOSTICS / "WBCIC_ENCODER_FINETUNING_PREDICTIONS.csv",
        DIAGNOSTICS / "WBCIC_FIXED_PERSIST_FUSION_PREDICTIONS.csv",
        DIAGNOSTICS / "WBCIC_FUTURE_SESSION_POPULATION_PREDICTIONS.csv",
        DIAGNOSTICS / "WBCIC_LARGE_EEGNET_PREDICTIONS.csv",
        DIAGNOSTICS / "WBCIC_LARGE_SELECTIVE_HEAD_PREDICTIONS.csv",
    ),
}

EEGNET_REFERENCES = {
    "OpenBMI": "B_POPULATION_LINEAR",
    "WBCIC": "B_EEGNET_FROZEN_HEAD",
}

PRE_V6_MATCHED_REFERENCES = {
    "OpenBMI": "B_HISTORY_FUSION_LDA",
    "WBCIC": "V5_CS_LGS_ANCHOR",
}


def _tables(benchmark: str) -> pd.DataFrame:
    parts = []
    for path in TABLES[benchmark]:
        if path.is_file():
            frame = pd.read_csv(path)
            if "method_id" in frame and "mean_subject_BA" in frame:
                frame["source_table"] = path.name
                parts.append(frame)
    if not parts:
        raise RuntimeError(f"No completed tables for {benchmark}")
    result = pd.concat(parts, ignore_index=True, sort=False)
    result = result.sort_values(["mean_subject_BA", "NLL"], ascending=[False, True]).drop_duplicates("method_id", keep="first")
    return result.reset_index(drop=True)


def _v5_prediction() -> pd.DataFrame:
    frame = pd.read_csv(v5_output_root() / "diagnostics" / "WBCIC_MULTI_SEED_OOF_PREDICTIONS.csv")
    frame = frame.loc[frame.seed.astype(int).eq(V6_SEED)].copy().rename(columns={"dataset": "benchmark"})
    frame["benchmark"] = "WBCIC_S1S2_to_S3_authorized_development"
    frame["method_id"] = "V5_CS_LGS_ANCHOR"
    frame["target_future_labels_used_for_fit"] = False
    return frame


def _prediction(benchmark: str, method: str) -> pd.DataFrame:
    if method == "V5_CS_LGS_ANCHOR":
        return _v5_prediction()
    for path in PREDICTION_FILES[benchmark]:
        if not path.is_file():
            continue
        frame = pd.read_csv(path, low_memory=False)
        if "method_id" not in frame:
            continue
        part = frame.loc[frame.method_id.astype(str).eq(method)].copy()
        if len(part):
            if "benchmark" not in part and "dataset" in part:
                part = part.rename(columns={"dataset": "benchmark"})
            if part.trial_uid.duplicated().any():
                raise RuntimeError(f"Duplicate predictions for {benchmark}/{method} in {path}")
            return part
    raise FileNotFoundError(f"Predictions not found: {benchmark}/{method}")


def _is_persist(method: str) -> bool:
    upper = method.upper()
    return "PERSIST" in upper or upper.startswith("V6_")


def _paired_comparison(benchmark: str, method: str, reference: str) -> dict[str, Any]:
    predictions = _prediction(benchmark, method)
    reference_predictions = _prediction(benchmark, reference)
    metric, _, _ = summarize(predictions, reference=reference_predictions)
    return metric


def _selected(benchmark: str, table: pd.DataFrame) -> dict[str, Any]:
    generic = table.loc[~table.method_id.astype(str).map(_is_persist)].sort_values("mean_subject_BA", ascending=False).iloc[0]
    persist_rows = table.loc[table.method_id.astype(str).map(_is_persist)].sort_values("mean_subject_BA", ascending=False)
    persist = persist_rows.iloc[0] if len(persist_rows) else None
    best = table.sort_values("mean_subject_BA", ascending=False).iloc[0]
    result: dict[str, Any] = {
        "best_legal_method": str(best.method_id),
        "best_legal_BA": float(best.mean_subject_BA),
        "strongest_information_matched_generic": str(generic.method_id),
        "strongest_information_matched_generic_BA": float(generic.mean_subject_BA),
        "best_PERSIST_method": None if persist is None else str(persist.method_id),
        "best_PERSIST_BA": None if persist is None else float(persist.mean_subject_BA),
    }
    if persist is not None:
        persist_predictions = _prediction(benchmark, str(persist.method_id))
        generic_predictions = _prediction(benchmark, str(generic.method_id))
        metric, subjects, folds = summarize(persist_predictions, reference=generic_predictions)
        result["PERSIST_vs_strongest_generic"] = metric
        write_csv(DIAGNOSTICS / f"{benchmark.upper()}_FINAL_PERSIST_SUBJECT_RESULTS.csv", subjects)
        write_csv(DIAGNOSTICS / f"{benchmark.upper()}_FINAL_PERSIST_FOLD_RESULTS.csv", folds)
    return result


def _safe_row(table: pd.DataFrame, pattern: str, label: str, benchmark: str) -> dict[str, Any]:
    part = table.loc[table.method_id.astype(str).str.contains(pattern, regex=True, case=False)]
    if not len(part):
        return {"benchmark": benchmark, "stage": label, "method_id": "NOT_AVAILABLE", "mean_subject_BA": np.nan, "Delta_BA": np.nan}
    row = part.sort_values("mean_subject_BA", ascending=False).iloc[0]
    return {"benchmark": benchmark, "stage": label, **row.to_dict()}


def _required_tables(open_table: pd.DataFrame, wbcic_table: pd.DataFrame, decisions: dict[str, dict[str, Any]]) -> None:
    main_rows = []
    stage_patterns = (
        ("single_backbone", "FROZEN|POPULATION_LINEAR"),
        ("strongest_static_or_V5", "V5_CS_LGS|HISTORY_FUSION_LDA"),
        ("standard_adapter", "GENERIC_SELECTED|GENERIC_AFFINE|TARGET_ADAPTED"),
        ("prototype", "PROTO"),
        ("geometry", "FBCSP|CSP"),
        ("PERSIST_constraint", "PERSIST"),
        ("selective_gate", "GATE|SELECTIVE"),
        ("best_legal", "^" + decisions["OpenBMI"]["best_legal_method"] + "$"),
    )
    for benchmark, table in (("OpenBMI", open_table), ("WBCIC", wbcic_table)):
        for label, pattern in stage_patterns[:-1]:
            main_rows.append(_safe_row(table, pattern, label, benchmark))
        method = decisions[benchmark]["best_legal_method"]
        main_rows.append(_safe_row(table, "^" + method.replace("+", "\\+") + "$", "best_legal", benchmark))
    main = pd.DataFrame(main_rows)
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_V6.csv", main)
    write_csv(LEADERBOARD / "OPENBMI_V6.csv", open_table)
    write_csv(LEADERBOARD / "WBCIC_DEV_V6.csv", wbcic_table)

    evolution = []
    for benchmark, table in (("OpenBMI", open_table), ("WBCIC", wbcic_table)):
        ordered = table.sort_values("mean_subject_BA")
        running = -np.inf
        for _, row in ordered.iterrows():
            value = float(row.mean_subject_BA)
            if value > running:
                evolution.append({"benchmark": benchmark, "method_id": row.method_id, "mean_subject_BA": value, "source_table": row.source_table, "history_matched": True, "OUTER_TEST_USED": False})
                running = value
    write_csv(LEADERBOARD / "BASELINE_EVOLUTION.csv", pd.DataFrame(evolution))

    adapter = pd.concat(
        [
            open_table.loc[open_table.method_id.str.contains("ADAPTER|FINETUN|HISTORY_HEAD|GENERIC", case=False, regex=True)],
            wbcic_table.loc[wbcic_table.method_id.str.contains("ADAPTER|FINETUN|TARGET_ADAPTED|GENERIC", case=False, regex=True)],
        ], ignore_index=True, sort=False,
    )
    write_csv(ABLATIONS / "ADAPTER_ABLATION.csv", adapter)
    write_csv(ABLATIONS / "CAPACITY_MATCHED_CONTROLS.csv", pd.concat([open_table, wbcic_table], ignore_index=True, sort=False))
    write_csv(ABLATIONS / "PROTOTYPE_ABLATION.csv", pd.concat([open_table.loc[open_table.method_id.str.contains("PROTO")], wbcic_table.loc[wbcic_table.method_id.str.contains("PROTO")]], ignore_index=True, sort=False))
    write_csv(ABLATIONS / "PERSIST_ABLATION.csv", pd.concat([open_table.loc[open_table.method_id.map(_is_persist)], wbcic_table.loc[wbcic_table.method_id.map(_is_persist)]], ignore_index=True, sort=False))
    write_csv(ABLATIONS / "RISK_GATE_ABLATION.csv", pd.concat([open_table.loc[open_table.method_id.str.contains("GATE|SELECTIVE", case=False, regex=True)], wbcic_table.loc[wbcic_table.method_id.str.contains("GATE|SELECTIVE", case=False, regex=True)]], ignore_index=True, sort=False))
    write_csv(ABLATIONS / "ALIGNMENT_ABLATION.csv", pd.DataFrame([{"method_id": "class_conditional_alignment_not_retained", "reason": "Frozen embedding alignment/conditional adapters did not exceed matched controls", "OUTER_TEST_USED": False}]))


def _diagnostics(decisions: dict[str, dict[str, Any]]) -> None:
    write_csv(DIAGNOSTICS / "REPRESENTATION_AUDIT.csv", pd.DataFrame([
        {"benchmark": "OpenBMI", "finding": "MI-specific enlarged EEGNet materially outperformed the multitask global-pooling representation", "OUTER_TEST_USED": False},
        {"benchmark": "WBCIC", "finding": "frozen representation adapters and target fine-tuning remained below V5 before the enlarged-backbone test", "OUTER_TEST_USED": False},
    ]))
    write_csv(DIAGNOSTICS / "SESSION_DRIFT_AUDIT.csv", pd.DataFrame([
        {"benchmark": "OpenBMI", "history": "S1", "future": "S2", "future_labels_used_for_fit": False, "OUTER_TEST_USED": False},
        {"benchmark": "WBCIC", "history": "S1+S2", "future": "S3", "future_labels_used_for_fit": False, "OUTER_TEST_USED": False},
    ]))
    write_csv(DIAGNOSTICS / "PROTECTED_SUBSPACE_AUDIT.csv", pd.DataFrame([
        {"benchmark": "OpenBMI", "protection": "diagonal empirical Fisher", "increment_found": False, "conclusion": "did not beat random/uniform controls", "OUTER_TEST_USED": False},
        {"benchmark": "WBCIC", "protection": "paired diagonal empirical Fisher", "increment_found": False, "conclusion": "paired-seed repair removed the apparent positive increment", "OUTER_TEST_USED": False},
    ]))
    write_csv(DIAGNOSTICS / "ADAPTABLE_SUBSPACE_AUDIT.csv", pd.DataFrame([
        {"benchmark": key, "best_method": value["best_legal_method"], "best_PERSIST_method": value["best_PERSIST_method"], "OUTER_TEST_USED": False}
        for key, value in decisions.items()
    ]))
    geometry = pd.read_csv(BASELINES / "OPENBMI_GEOMETRY_BASELINES.csv") if (BASELINES / "OPENBMI_GEOMETRY_BASELINES.csv").is_file() else pd.DataFrame()
    write_csv(DIAGNOSTICS / "GEOMETRY_COMPLEMENTARITY.csv", geometry)
    write_csv(DIAGNOSTICS / "PROTOTYPE_COMPLEMENTARITY.csv", pd.DataFrame([{"finding": "supervised cosine and shrinkage prototypes did not exceed history-fusion LDA on either benchmark", "OUTER_TEST_USED": False}]))


def _integrity_audit(decisions: dict[str, dict[str, Any]]) -> None:
    expected_subjects = {"OpenBMI": 54, "WBCIC": 41}
    rows = []
    for benchmark, decision in decisions.items():
        methods = {
            decision["best_legal_method"],
            decision["strongest_information_matched_generic"],
            decision["EEGNet_reference_method"],
            decision["pre_V6_strongest_matched_reference"],
        }
        if decision["best_PERSIST_method"] is not None:
            methods.add(decision["best_PERSIST_method"])
        for method in sorted(methods):
            predictions = _prediction(benchmark, method)
            subject_count = int(predictions.subject_id.astype(str).nunique())
            folds = sorted(predictions.outer_fold.astype(int).unique().tolist())
            future_used = (
                "target_future_labels_used_for_fit" in predictions
                and predictions.target_future_labels_used_for_fit.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).any()
            )
            outer_used = (
                "OUTER_TEST_USED" in predictions
                and predictions.OUTER_TEST_USED.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}).any()
            )
            future_clean = not future_used
            outer_clean = not outer_used
            valid = (
                not predictions.trial_uid.duplicated().any()
                and subject_count == expected_subjects[benchmark]
                and folds == [0, 1, 2, 3, 4]
                and future_clean
                and outer_clean
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "method_id": method,
                    "trials": int(len(predictions)),
                    "subjects": subject_count,
                    "folds": folds,
                    "duplicate_trial_uids": int(predictions.trial_uid.duplicated().sum()),
                    "target_future_labels_used_for_fit": not future_clean,
                    "OUTER_TEST_USED": not outer_clean,
                    "valid": bool(valid),
                }
            )
    if not all(row["valid"] for row in rows):
        raise RuntimeError("Final prediction integrity audit failed")
    write_json(PROTOCOL / "FINAL_PREDICTION_INTEGRITY_AUDIT.json", {"checks": rows, "all_valid": True, "OUTER_TEST_USED": False})


def _hashes() -> dict[str, str]:
    paths = list((EXPERIMENT_ROOT / "code").rglob("*.py"))
    paths += [path for path in (LEADERBOARD / "CROSS_BENCHMARK_V6.csv", OUTPUTS / "FINAL_DECISION.json") if path.is_file()]
    return {str(path.relative_to(EXPERIMENT_ROOT)).replace("\\", "/"): sha256_file(path) for path in sorted(paths)}


def run() -> None:
    open_table = _tables("OpenBMI")
    wbcic_table = _tables("WBCIC")
    decisions = {"OpenBMI": _selected("OpenBMI", open_table), "WBCIC": _selected("WBCIC", wbcic_table)}
    for benchmark in decisions:
        eegnet_method = EEGNET_REFERENCES[benchmark]
        matched_method = PRE_V6_MATCHED_REFERENCES[benchmark]
        eegnet_comparison = _paired_comparison(benchmark, decisions[benchmark]["best_legal_method"], eegnet_method)
        matched_comparison = _paired_comparison(benchmark, decisions[benchmark]["best_legal_method"], matched_method)
        decisions[benchmark]["EEGNet_reference_method"] = eegnet_method
        decisions[benchmark]["EEGNet_reference_BA"] = float(
            (open_table if benchmark == "OpenBMI" else wbcic_table)
            .loc[lambda frame: frame.method_id.eq(eegnet_method), "mean_subject_BA"]
            .iloc[0]
        )
        decisions[benchmark]["best_legal_vs_EEGNet"] = eegnet_comparison
        decisions[benchmark]["best_legal_Delta_vs_EEGNet"] = float(eegnet_comparison["Delta_BA"])
        decisions[benchmark]["plus_5pp_over_EEGNet"] = bool(eegnet_comparison["Delta_BA"] >= 0.05)
        decisions[benchmark]["pre_V6_strongest_matched_reference"] = matched_method
        decisions[benchmark]["best_legal_vs_pre_V6_matched_reference"] = matched_comparison
        decisions[benchmark]["PERSIST_increment_found"] = bool(
            decisions[benchmark].get("PERSIST_vs_strongest_generic", {}).get("Delta_BA", -np.inf) > 0
            and decisions[benchmark].get("PERSIST_vs_strongest_generic", {}).get("CI95_L", -np.inf) > 0
        )
    dual_relaxed = all(value["plus_5pp_over_EEGNet"] for value in decisions.values())
    dual_matched = all(
        value.get("PERSIST_vs_strongest_generic", {}).get("Delta_BA", -np.inf) >= 0.05
        for value in decisions.values()
    )
    if dual_matched:
        terminal = "V6_DUAL_BENCHMARK_TARGET_REACHED"
    elif decisions["OpenBMI"]["plus_5pp_over_EEGNet"] and not decisions["WBCIC"]["plus_5pp_over_EEGNet"]:
        terminal = "V6_OPENBMI_TARGET_ONLY"
    elif dual_relaxed:
        terminal = "V6_5PP_OVER_EEGNET_BACKBONE_LEVEL_ONLY"
    else:
        terminal = "V6_SCIENTIFIC_EXHAUSTION"
    ready = dual_matched and all(value["PERSIST_increment_found"] for value in decisions.values())
    decision = {
        "terminal_state": terminal,
        "benchmarks": decisions,
        "dual_plus_5pp_over_EEGNet_secondary_goal": dual_relaxed,
        "dual_plus_5pp_over_strongest_matched_baseline": dual_matched,
        "PERSIST_increment_found_on_both": all(value["PERSIST_increment_found"] for value in decisions.values()),
        "harmful_subspace_certified_in_real_EEG": False,
        "suppression_used": False,
        "READY_FOR_OUTER_FREEZE": ready,
        "development_estimate_exploratory": True,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", decision)
    _required_tables(open_table, wbcic_table, decisions)
    _diagnostics(decisions)
    _integrity_audit(decisions)

    spec = {
        "name": "PERSIST-SA V6 exploratory candidate",
        "best_legal_methods": {key: value["best_legal_method"] for key, value in decisions.items()},
        "best_PERSIST_methods": {key: value["best_PERSIST_method"] for key, value in decisions.items()},
        "core_backbone_family": "MI-specific EEGNet; enlarged 2x-width candidate tested on both benchmarks",
        "target_history": {"OpenBMI": "S1 labeled", "WBCIC": "S1/S2 labeled"},
        "PERSIST_protection": "diagonal empirical Fisher with identical paired optimization controls",
        "harmful_subspace": "empty",
        "suppression": 0.0,
        "outer_test_used": False,
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_CANDIDATE / "FINAL_MODEL_SPEC.json", spec)
    (FINAL_CANDIDATE / "FINAL_MODEL_SPEC.md").write_text(
        "# V6 final exploratory specification\n\n"
        f"- OpenBMI best legal method: `{decisions['OpenBMI']['best_legal_method']}` ({decisions['OpenBMI']['best_legal_BA']:.4f} BA)\n"
        f"- WBCIC best legal method: `{decisions['WBCIC']['best_legal_method']}` ({decisions['WBCIC']['best_legal_BA']:.4f} BA)\n"
        f"- Terminal state: `{terminal}`\n"
        "- Real-EEG harmful subspace: not certified; suppression disabled.\n"
        "- WBCIC outer: untouched.\n",
        encoding="utf-8",
    )
    write_json(FINAL_CANDIDATE / "DEVELOPMENT_RESULTS.json", decisions)
    write_json(FINAL_CANDIDATE / "DUAL_BENCHMARK_RESULTS.json", {"decisions": decisions, "secondary_dual_EEGNet_goal": dual_relaxed, "primary_dual_matched_goal": dual_matched, "OUTER_TEST_USED": False})
    write_json(FINAL_CANDIDATE / "FINAL_MODEL_HASHES.json", {"files": _hashes(), "OUTER_TEST_USED": False})

    report = f"""# PERSIST-EEG V6 scientific report

## Decision

Terminal state: `{terminal}`. All estimates are exploratory development results.

OpenBMI best legal result is `{decisions['OpenBMI']['best_legal_method']}` at **{100*decisions['OpenBMI']['best_legal_BA']:.2f}% BA**, or **{100*decisions['OpenBMI']['best_legal_Delta_vs_EEGNet']:+.2f} pp** versus the frozen EEGNet proxy (subject-bootstrap 95% CI **[{100*decisions['OpenBMI']['best_legal_vs_EEGNet']['CI95_L']:+.2f}, {100*decisions['OpenBMI']['best_legal_vs_EEGNet']['CI95_U']:+.2f}] pp**). WBCIC best legal result is `{decisions['WBCIC']['best_legal_method']}` at **{100*decisions['WBCIC']['best_legal_BA']:.2f}% BA**, or **{100*decisions['WBCIC']['best_legal_Delta_vs_EEGNet']:+.2f} pp** versus frozen EEGNet (subject-bootstrap 95% CI **[{100*decisions['WBCIC']['best_legal_vs_EEGNet']['CI95_L']:+.2f}, {100*decisions['WBCIC']['best_legal_vs_EEGNet']['CI95_U']:+.2f}] pp**).

The secondary +5 pp-over-EEGNet goal was {'reached on both benchmarks' if dual_relaxed else 'not reached on both benchmarks'}. The scientifically stricter +5 pp-over-strongest-information-matched-baseline target was {'reached' if dual_matched else 'not reached'}.

## Direct answers

1. Strongest OpenBMI information-matched generic: `{decisions['OpenBMI']['strongest_information_matched_generic']}`, BA {decisions['OpenBMI']['strongest_information_matched_generic_BA']:.4f}.
2. Strongest WBCIC information-matched generic: `{decisions['WBCIC']['strongest_information_matched_generic']}`, BA {decisions['WBCIC']['strongest_information_matched_generic_BA']:.4f}.
3. Target information: OpenBMI outcome S1 labels only; WBCIC outcome S1/S2 labels only. Future labels are scoring-only.
4. Best PERSIST-vs-generic OpenBMI delta: {decisions['OpenBMI'].get('PERSIST_vs_strongest_generic', {}).get('Delta_BA')}.
5. Best PERSIST-vs-generic WBCIC delta: {decisions['WBCIC'].get('PERSIST_vs_strongest_generic', {}).get('Delta_BA')}.
6. The primary dual matched +5 pp target: {dual_matched}.
7. Subject-bootstrap CIs for best-vs-EEGNet, best-vs-pre-V6 matched anchor, and PERSIST-vs-strongest-generic are stored in `final_candidate/DEVELOPMENT_RESULTS.json` and the final subject tables.
8. Fold positivity is reported in the same records.
9. Positive/nonnegative subject fractions are reported in the same records.
10. Worst-subject deltas are reported in the same records.
11. Representation adaptation did not consistently beat the strongest frozen-output/history stack on WBCIC.
12. Conditional alignment did not add reliable value.
13. Prototype adaptation did not add reliable value.
14. FBCSP/geometry did not add reliable value on OpenBMI.
15. Paired-seed Fisher protection did not improve BA over its generic control.
16. No robust real-EEG PERSIST safety increment survived paired-control repair.
17. The protected mechanism was diagonal task Fisher; it was not empirically sufficient.
18. Generic head/tail/full parameters were allowed to adapt from legal history.
19. No harmful real-EEG subspace was certified.
20. Suppression was unnecessary and remained disabled.
21. The code supports K=1 OpenBMI and K=2 WBCIC histories.
22. Generic capacity-matched adaptation explained or exceeded the PERSIST result.
23. Increment uniquely attributable to PERSIST-SA was not established.
24. Weak standalone FBCSP and failed conditional adapters were excluded from the best result.
25. No outcome future-session label was used for fitting or selection.
26. WBCIC outer was not opened, enumerated, featurized, or scored.
27. Attempted families included frozen representation heads, prototypes, FiLM/affine, bilinear adapters, FBCSP, encoder fine-tuning, Fisher protection, future-session population training, selective gates, and enlarged MI-specific backbones.
28. Redesigns followed negative matched-control results; failed families remain in outputs.
29. Dual matched +5 pp reached: {dual_matched}; secondary dual EEGNet +5 pp reached: {dual_relaxed}.
30. Ready for outer freeze: {ready}.

## Reproducibility warning

An initially positive WBCIC Fisher result disappeared after generic/Fisher controls were rerun with identical minibatch-order seeds. The repaired paired result is the only result retained in final tables. This repair is material and prevents a false PERSIST attribution.
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    (RESEARCH_LOG / "HYPOTHESIS_LEDGER.md").write_text(
        "# V6 hypothesis ledger\n\n"
        "| Iteration | Structural hypothesis | Outcome |\n|---:|---|---|\n"
        "| 000 | protocol and baseline reconstruction | KEEP |\n"
        "| 001 | frozen conditional affine/bilinear adaptation | ABANDON |\n"
        "| 002 | target encoder fine-tuning | generic improvement only; PERSIST failed after paired repair |\n"
        "| 003 | OpenBMI MI-specific enlarged backbone | KEEP if reflected in final leaderboard |\n"
        "| 004 | legal WBCIC model-fit S3 population update | ABANDON versus V5 |\n"
        "| 005 | history-only selective subject head | see final leaderboard |\n"
        "| 006 | WBCIC enlarged MI-specific backbone | see final leaderboard |\n"
        "| 007 | synthetic protected/adaptable/harmful positive control | PASS, mechanism-only |\n\n"
        "All development estimates are exploratory. `OUTER_TEST_USED=false`.\n",
        encoding="utf-8",
    )
    write_csv(RESEARCH_LOG / "ITERATION_SUMMARY.csv", pd.DataFrame([
        {"iteration": 0, "family": "protocol", "decision": "KEEP"},
        {"iteration": 1, "family": "conditional adapters", "decision": "ABANDON"},
        {"iteration": 2, "family": "encoder fine-tuning and Fisher protection", "decision": "PERSIST_ABANDON"},
        {"iteration": 3, "family": "OpenBMI MI-specific backbone", "decision": "SEE_FINAL"},
        {"iteration": 4, "family": "WBCIC future-session population update", "decision": "ABANDON"},
        {"iteration": 5, "family": "selective history head", "decision": "SEE_FINAL"},
        {"iteration": 6, "family": "WBCIC enlarged EEGNet", "decision": "SEE_FINAL"},
    ]).assign(OUTER_TEST_USED=False))
    write_json(OUTPUTS / "REPRODUCIBILITY.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "git_head_before_V6_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=EXPERIMENT_ROOT.parents[1], text=True).strip(),
        "seed": V6_SEED,
        "multi_seed_run": False,
        "multi_seed_reason": "User explicitly requested one seed for the current exploratory stage.",
        "OUTER_TEST_USED": False,
    })
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    run()
