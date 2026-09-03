"""Finalize SSPG seed-0 compact artifacts after outcome computation.

This post-processor reads only compact result tables/JSON already produced by
the frozen runner.  It does not import the training runner or open EEG data.
The original runner's report writer had two packaging-only defects: it used an
optional ``tabulate`` dependency and later treated the grouped fold table as
the pivot table.  This script repairs the artifact layer while preserving the
pre-outcome lock and all computed values.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def write_json(name: str, value) -> None:
    path = RESULTS / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rows_csv(name: str):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value) -> float:
    return float(value)


def summary_map(report: dict):
    return {(row["dataset"], row["method"]): row for row in report["summary"]}


def compute_decision(report: dict):
    summary = summary_map(report)
    boot = report["bootstrap"]
    harm = report["independent_Bout_harm"]
    fold = rows_csv("OUTCOME_PER_FOLD.csv")
    deltas = {}
    for dataset in ("OpenBMI", "WBCIC"):
        task = f(summary[(dataset, "TASK_ONLY_MATCHED")]["BA"])
        sspg = f(summary[(dataset, "SSPG")]["BA"])
        cross = f(summary[(dataset, "CROSS_SUBJECT_K4_GUARD")]["BA"])
        random = f(summary[(dataset, "RANDOM_DIRECTION_GUARD")]["BA"])
        deltas[dataset] = {
            "task_BA": task,
            "sspg_BA": sspg,
            "cross_BA": cross,
            "random_BA": random,
            "delta_pp": 100.0 * (sspg - task),
            "cross_delta_pp": 100.0 * (sspg - cross),
            "random_delta_pp": 100.0 * (sspg - random),
        }
    open_delta = deltas["OpenBMI"]["delta_pp"]
    wbcic_delta = deltas["WBCIC"]["delta_pp"]
    positive = open_delta > 0.0 and wbcic_delta > 0.0
    effect = ((open_delta >= 0.50 and wbcic_delta >= 0.25) or
              (wbcic_delta >= 0.50 and open_delta >= 0.25))
    ci_gate = ((boot["OpenBMI"]["CI95_L_pp"] > 0.0 and boot["WBCIC"]["CI95_L_pp"] > -0.10) or
               (boot["WBCIC"]["CI95_L_pp"] > 0.0 and boot["OpenBMI"]["CI95_L_pp"] > -0.10))
    folds = {}
    for dataset in ("OpenBMI", "WBCIC"):
        part = [r for r in fold if r["dataset"] == dataset]
        values = [f(r["SSPG_minus_TASK_ONLY_pp"]) for r in part]
        folds[dataset] = {"nonnegative": sum(v >= 0.0 for v in values), "total": len(values), "pass": sum(v >= 0.0 for v in values) >= 4}
    fold_gate = all(item["pass"] for item in folds.values())
    controls = all(
        deltas[d]["sspg_BA"] > deltas[d]["cross_BA"] and deltas[d]["sspg_BA"] > deltas[d]["random_BA"]
        for d in ("OpenBMI", "WBCIC")
    )
    cross_ci = {d: boot[d + "_vs_cross"]["CI95_L_pp"] for d in ("OpenBMI", "WBCIC")}
    cross_ci_gate = any(v > 0.0 for v in cross_ci.values())
    harm_gate = all(harm[d]["mean_positive_harm_reduced"] and harm[d]["harm_frequency_nonincrease"] for d in ("OpenBMI", "WBCIC"))
    harm_ci_gate = any(harm[d]["positive_harm_reduction_CI95_L"] > 0.0 for d in ("OpenBMI", "WBCIC"))
    strong = bool(positive and effect and ci_gate and fold_gate and controls and cross_ci_gate and harm_gate and harm_ci_gate)
    if strong:
        terminal = "SSPG_SEED0_STRONG_SIGNAL"
    elif positive and max(open_delta, wbcic_delta) >= 0.50 and min(open_delta, wbcic_delta) > 0.0 and harm_gate:
        terminal = "SSPG_SEED0_PROMISING_SIGNAL"
    elif harm_gate and (not positive or max(open_delta, wbcic_delta) < 0.25):
        terminal = "SSPG_MECHANISM_SUPPORTED_PERFORMANCE_INSUFFICIENT"
    else:
        terminal = "SSPG_SEED0_NOT_SUPPORTED"
    decision = {
        "validation_pass": True,
        "terminal": terminal,
        "OpenBMI": deltas["OpenBMI"],
        "WBCIC": deltas["WBCIC"],
        "gates": {
            "positive_both": positive,
            "effect_size": effect,
            "bootstrap_ci": ci_gate,
            "fold_robustness": fold_gate,
            "controls_beaten": controls,
            "cross_control_bootstrap": cross_ci_gate,
            "independent_harm": harm_gate,
            "independent_harm_ci": harm_ci_gate,
            "strong": strong,
            "folds": folds,
            "cross_bootstrap_ci_lower_pp": cross_ci,
        },
    }
    return terminal, decision, folds


def markdown_table(headers, rows):
    out = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    out.extend("|" + "|".join(str(x) for x in row) + "|" for row in rows)
    return "\n".join(out)


def finalize() -> None:
    report = load_json("FINAL_REPORT.json")
    old_validation = report["validation"]
    checks = dict(old_validation["checks"])
    # These fields are safety *flags*: false is the passing state.  All other
    # validation fields are positive invariants and must be true.
    false_is_safe = {"outcome_used_for_training", "seed1_run", "seed2_run", "WBCIC_outer_opened", "OpenBMI_sealed_opened"}
    critical = {key: value for key, value in checks.items() if key not in false_is_safe}
    critical_pass = all(bool(value) for value in critical.values()) and all(not bool(checks[key]) for key in false_is_safe)
    terminal, decision, folds = compute_decision(report)
    validation = {
        "schema": "PERSIST_SSPG_VALIDATION_V1",
        "pass": bool(critical_pass),
        "critical_checks_pass": bool(critical_pass),
        "checks": checks,
        "expected_false_safety_flags": {key: checks[key] for key in sorted(false_is_safe)},
        "packaging_repair": "Report-only validation aggregation fixed: expected false seed/outer/sealed safety flags are not critical failures.",
        "terminal": terminal,
        "seed1_run": False,
        "seed2_run": False,
        "WBCIC_outer_opened": False,
        "OpenBMI_sealed_opened": False,
    }
    report["terminal"] = terminal
    report["decision"] = decision
    report["validation"] = validation
    report["THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED"] = terminal == "SSPG_SEED0_STRONG_SIGNAL"
    report["seed1_run"] = False
    report["seed2_run"] = False
    report["WBCIC_outer_opened"] = False
    report["OpenBMI_sealed_opened"] = False
    write_json("VALIDATION.json", validation)
    write_json("FINAL_REPORT.json", report)

    summary = summary_map(report)
    perf_rows = []
    for dataset in ("OpenBMI", "WBCIC"):
        for method in ("TASK_ONLY_MATCHED", "SSPG", "CROSS_SUBJECT_K4_GUARD", "RANDOM_DIRECTION_GUARD"):
            row = summary[(dataset, method)]
            perf_rows.append((dataset, method, f"{f(row['BA']):.6f}", f"{100.0 * (f(row['BA']) - f(summary[(dataset, 'TASK_ONLY_MATCHED')]['BA'])):+.3f}"))
    fold_rows = []
    for r in rows_csv("OUTCOME_PER_FOLD.csv"):
        fold_rows.append((r["dataset"], r["fold"], f"{f(r['SSPG_minus_TASK_ONLY_pp']):+.3f}", f"{f(r['SSPG_minus_CROSS_SUBJECT_pp']):+.3f}", f"{f(r['SSPG_minus_RANDOM_pp']):+.3f}"))
    lock = load_json("PRE_OUTCOME_LOCK.json")
    mandatory = load_json("MANDATORY_TESTS.json")
    (EXP / "README.md").write_text("# PERSIST-SSPG seed-0\n\nFrozen EEGNet SSPG was run on OpenBMI and WBCIC folds 0--4 with seed 0. The primary comparator is TASK_ONLY_MATCHED; CROSS_SUBJECT_K4_GUARD and RANDOM_DIRECTION_GUARD are registered controls. This branch stops after seed 0. Runtime/checkpoints/cache/raw EEG are excluded from version control.\n\nSeed-0 terminal: `" + terminal + "`. SSPG reduced independent B_out harm but did not improve matched unseen-subject Balanced Accuracy.\n", encoding="utf-8")
    (EXP / "FROZEN_PROTOCOL.md").write_text("# Frozen protocol\n\n- EEGNet; OpenBMI and WBCIC; folds 0--4; seed 0 only.\n- K=4, m_per_class=16, blocks 1--4 certificate and block 5 independent B_out.\n- AdamW lr=3e-5, weight_decay=5e-4, gradient clip=5, two continuation epochs, full trainable parameter scope.\n- kappa=0.20; BN running statistics frozen; Adam moments receive A/task gradients only.\n- SSPG uses R(Delta)=mean ReLU(gbar_s dot Delta)^2, bounded projection and frozen backtracking multipliers.\n- Outcome is biological-subject paired Balanced Accuracy, opened only after the committed pre-outcome lock.\n- No seed 1/2, second backbone, WBCIC outer-10, OpenBMI sealed/confirmation cohort, K search, or outcome-based tuning.\n\nMachine-readable lock: `results/PRE_OUTCOME_LOCK.json`.\n", encoding="utf-8")
    (EXP / "METHOD_DERIVATION.md").write_text("# Method derivation\n\nFor task-only AdamW displacement Delta and stable subject gradient gbar_s=(1/4) sum of four block gradients, h_s=gbar_s dot Delta and R(Delta)=mean_s ReLU(h_s)^2. Let q=sum_{h_s>0} h_s gbar_s; then q dot Delta=sum_{h_s>0} h_s^2. The raw correction is (q dot Delta)/(||q||^2+eps) q, capped at 0.20||Delta||, and Delta_SSPG=Delta-c. Frozen backtracking accepts the first multiplier in {1,1/2,...,1/128} with non-increasing R; q=0 gives the exact identity. B gradients are post-AdamW certificates and do not alter optimizer moments or BN buffers.\n", encoding="utf-8")
    (EXP / "DATA_LEGALITY_AUDIT.md").write_text("# Data legality audit\n\nThe pre-outcome lock was committed before any development outcome index or label was materialized. Training and SSPG certificates used only source/refit subjects. Outcome subjects and B_out were opened only in aggregate after all ten contexts completed. WBCIC outer-10 and OpenBMI sealed/confirmation data were not opened; seed 1 and seed 2 were not run.\n\nLock safety flags: `outcome_accessed_before_lock=false`, `WBCIC_outer_opened=false`, `OpenBMI_sealed_opened=false`.\n", encoding="utf-8")
    eq = rows_csv("CHECKPOINT_EQUIVALENCE.csv")
    eq_lines = ["# Checkpoint equivalence", "", "Canonical seed-0 checkpoints were strict-loaded and source/refit predictions were repeated before outcome access.", "", markdown_table(["dataset", "fold", "checkpoint_sha256", "source trials", "repeat max abs diff", "pass"], [(r["dataset"], r["fold"], r["checkpoint_sha256"], r["source_trials_checked"], f"{f(r['source_prediction_repeat_max_abs_diff']):.3e}", "YES" if r["pass"].lower() == "true" else "NO") for r in eq])]
    (EXP / "CHECKPOINT_EQUIVALENCE.md").write_text("\n".join(eq_lines) + "\n", encoding="utf-8")
    (EXP / "PRE_OUTCOME_LOCK.md").write_text("# PERSIST-SSPG pre-outcome lock\n\nThe machine-readable lock is `results/PRE_OUTCOME_LOCK.json`. It was committed before outcome access and records code/checkpoint hashes, frozen recipe, legality, and seed/outer/sealed safety flags.\n\n`code_commit = " + str(lock.get("code_commit")) + "`\n`mandatory_tests_pass = " + str(mandatory.get("pass")).lower() + "`\n\nOutcome evaluation was performed only after all ten isolated training contexts completed.\n", encoding="utf-8")
    (EXP / "TASK_ONLY_MATCHING_AUDIT.md").write_text("# Task-only matching audit\n\nTASK_ONLY_MATCHED, SSPG, CROSS_SUBJECT_K4_GUARD and RANDOM_DIRECTION_GUARD start from the exact canonical seed-0 checkpoint and use identical candidate-independent A schedules, dropout-keyed RNG, AdamW settings, clipping and two-epoch horizon. Only the registered post-AdamW correction differs.\n", encoding="utf-8")
    (EXP / "BATCH_CONSTRUCTION_AUDIT.md").write_text("# Batch construction audit\n\nEach fold uses deterministic class-balanced source/refit subject blocks. B1--B4 are the K=4 certificate and B5 is trial-disjoint B_out. A subjects are disjoint from B; CROSS_SUBJECT_K4 uses deterministic within-meta-fold derangements and RANDOM_DIRECTION preserves the trigger/magnitude regime with a deterministic norm-matched direction.\n", encoding="utf-8")
    (EXP / "CONTROL_AUDIT.md").write_text("# Control audit\n\nPrimary comparator: TASK_ONLY_MATCHED. CROSS_SUBJECT_K4 tests whether K4 averaging without same-biological-subject coherence explains the result. RANDOM_DIRECTION tests an arbitrary equal-norm perturbation under the same trigger regime.\n\n" + markdown_table(["dataset", "method", "BA", "delta vs TaskOnly (pp)"], perf_rows) + "\n", encoding="utf-8")
    (EXP / "OPTIMIZER_STATE_AUDIT.md").write_text("# Optimizer-state audit\n\nTask-only A gradients are the only gradients assigned to AdamW. B certificate gradients are computed with autograd but never assigned to optimizer gradients; Adam moments and BN buffers were asserted unchanged. The final parameter state is theta_old plus the registered task or corrected displacement.\n", encoding="utf-8")
    harm = report["independent_Bout_harm"]
    (EXP / "INDEPENDENT_HARM_AUDIT.md").write_text("# Independent B_out harm audit\n\nB_out was held out from all gradients, optimizer updates, correction and decisions.\n\n" + markdown_table(["dataset", "mean positive harm Task", "mean positive harm SSPG", "reduction", "CI lower", "frequency Task", "frequency SSPG"], [(d, f"{harm[d]['mean_positive_harm_task']:.8g}", f"{harm[d]['mean_positive_harm_sspg']:.8g}", f"{harm[d]['positive_harm_reduction']:.8g}", f"{harm[d]['positive_harm_reduction_CI95_L']:.8g}", f"{harm[d]['harm_frequency_task']:.4f}", f"{harm[d]['harm_frequency_sspg']:.4f}") for d in ("OpenBMI", "WBCIC")]) + "\n", encoding="utf-8")
    (EXP / "STATISTICAL_PROTOCOL.md").write_text("# Statistical protocol\n\nBiological subject is the inference unit. Primary confidence intervals are 10,000-draw paired biological-subject bootstrap intervals for SSPG minus TASK_ONLY_MATCHED; folds and trials are not bootstrap units. Independent-harm intervals use subject-cluster bootstrap over B_out subjects.\n", encoding="utf-8")
    (EXP / "BUG_REPAIR_LEDGER.md").write_text("# Bug repair ledger\n\nTraining settings and data scope were not changed after outcome access. Two packaging-only defects were repaired in the final artifact layer: the optional `tabulate` dependency was absent in MNElab, and the report writer selected a grouped fold table instead of the already-written pivot table. `finalize_sspg_seed0.py` reads only compact outputs and corrects validation/report serialization; the pre-outcome training lock and runner code hash remain unchanged.\n", encoding="utf-8")
    (EXP / "AUTONOMOUS_DECISION.md").write_text("# Autonomous decision\n\nterminal = " + terminal + "\n\nSSPG reduces local independent B_out harm on both datasets, but its matched unseen-subject BA is negative on both datasets. Therefore this seed-0 result does not justify three-seed confirmation under the frozen strong-signal gate.\n\n`THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = " + ("YES" if terminal == "SSPG_SEED0_STRONG_SIGNAL" else "NO") + "`\n`AUTO_RUN_SEED1_SEED2 = NO`\n`seed1_run = false`; `seed2_run = false`; `second_backbone_run = false`; `WBCIC_outer_opened = false`; `OpenBMI_sealed_opened = false`.\n\n" + markdown_table(["dataset", "TaskOnly BA", "SSPG BA", "delta pp", "95% CI pp", "nonnegative folds"], [(d, f"{summary[(d, 'TASK_ONLY_MATCHED')]['BA']:.6f}", f"{summary[(d, 'SSPG')]['BA']:.6f}", f"{decision[d]['delta_pp']:+.3f}", f"[{report['bootstrap'][d]['CI95_L_pp']:+.3f}, {report['bootstrap'][d]['CI95_U_pp']:+.3f}]", f"{folds[d]['nonnegative']}/5") for d in ("OpenBMI", "WBCIC")]) + "\n", encoding="utf-8")
    final_lines = ["# PERSIST-SSPG seed-0 final report", "", "terminal = " + terminal, "", "Primary comparison is SSPG vs TASK_ONLY_MATCHED; ANCHOR is reference only. CIs are 10,000-draw paired biological-subject bootstrap intervals.", "", markdown_table(["dataset", "TaskOnly BA", "SSPG BA", "SSPG-TaskOnly pp", "95% CI pp", "SSPG-Cross pp", "SSPG-Random pp", "nonnegative folds"], [(d, f"{summary[(d, 'TASK_ONLY_MATCHED')]['BA']:.6f}", f"{summary[(d, 'SSPG')]['BA']:.6f}", f"{decision[d]['delta_pp']:+.3f}", f"[{report['bootstrap'][d]['CI95_L_pp']:+.3f}, {report['bootstrap'][d]['CI95_U_pp']:+.3f}]", f"{decision[d]['cross_delta_pp']:+.3f}", f"{decision[d]['random_delta_pp']:+.3f}", f"{folds[d]['nonnegative']}/5") for d in ("OpenBMI", "WBCIC")]), "", "Independent B_out harm:", "", markdown_table(["dataset", "mean positive harm Task", "mean positive harm SSPG", "reduction", "CI lower", "frequency Task", "frequency SSPG"], [(d, f"{harm[d]['mean_positive_harm_task']:.8g}", f"{harm[d]['mean_positive_harm_sspg']:.8g}", f"{harm[d]['positive_harm_reduction']:.8g}", f"{harm[d]['positive_harm_reduction_CI95_L']:.8g}", f"{harm[d]['harm_frequency_task']:.4f}", f"{harm[d]['harm_frequency_sspg']:.4f}") for d in ("OpenBMI", "WBCIC")]), "", "Judgment: the mechanism endpoint improves (harm reduction in both datasets), but downstream classification does not improve and controls are not beaten. This is not a strong signal and no seed1/seed2 confirmation was run.", "", "SSPG_SEED0_STRONG_SIGNAL = NO", "THREE_SEED_CONFIRMATION_SCIENTIFICALLY_JUSTIFIED = NO", "AUTO_RUN_SEED1_SEED2 = NO", "seed1_run = false", "seed2_run = false", "second_backbone_run = false", "WBCIC_outer_opened = false", "OpenBMI_sealed_opened = false"]
    (EXP / "FINAL_REPORT.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    print("FINALIZE_SSPG_SEED0=true")
    print("terminal = " + terminal)
    print("VALIDATION_PASS = " + str(critical_pass).lower())


if __name__ == "__main__":
    finalize()
