from __future__ import annotations

from typing import Any

import pandas as pd

import common as c


FINAL_TERMINAL = "FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED"
REPAIR2_TERMINAL = "TRANSPORT_VALIDITY_NOT_SUPPORTED"
NOT_RUN = "NOT_RUN_BY_STAGE0_GATE"


def append_once(path, marker: str, addition: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if marker not in existing:
        c.write_text(path, existing.rstrip() + "\n\n" + addition.strip())


def main() -> None:
    validation = c.read_json(c.RESULTS / "STAGE0_REPAIR2_VALIDATION.json")
    result = c.read_json(c.RESULTS / "STAGE0_REPAIR2_FINAL_RESULT.json")
    if validation.get("pass") is not True:
        raise RuntimeError("cannot close before Repair-2 validator passes")
    if validation.get("all_setting_pass_count") != 2:
        raise RuntimeError("unexpected Repair-2 pass cardinality")
    if result.get("terminal") != REPAIR2_TERMINAL:
        raise RuntimeError("closure is only valid for the frozen Repair-2 negative terminal")
    if result.get("stage1_authorized") is not False:
        raise RuntimeError("negative Repair-2 closure conflicts with Stage-1 authorization")
    if result.get("outer_or_future_performance_accessed") is not False:
        raise RuntimeError("future/outer access conflicts with the Repair-2 protocol")

    summary = pd.read_csv(c.RESULTS / "STAGE0_REPAIR2_LAYER_SUMMARY.csv")
    alpha = pd.read_csv(c.RESULTS / "STAGE0_REPAIR2_ALPHA_DISTRIBUTION.csv")
    setting_alpha = alpha[
        (alpha.query_unit == "centroid")
        & (alpha.scope == "setting")
        & (alpha.scope_value.astype(str) == "ALL")
    ].copy()
    if len(summary) != 4 or len(setting_alpha) != 4:
        raise RuntimeError("Repair-2 compact result cardinality mismatch")
    if int(summary.all_gates_pass.sum()) != 2:
        raise RuntimeError("Repair-2 summary no longer matches validated terminal")

    compact = summary.merge(
        setting_alpha[
            [
                "setting_id",
                "fraction_alpha_zero",
                "alpha_mean",
                "alpha_median",
                "alpha_q25",
                "alpha_q75",
                "fraction_alpha_max",
                "realized_norm_ratio_mean",
            ]
        ],
        on="setting_id",
        validate="one_to_one",
    )
    order = [
        "OPENBMI_MI_EEGNET",
        "OPENBMI_MI_EEGCONFORMER",
        "WBCIC_MI_EEGNET",
        "WBCIC_MI_EEGCONFORMER",
    ]
    compact["_order"] = compact.setting_id.map({name: i for i, name in enumerate(order)})
    compact = compact.sort_values("_order").drop(columns="_order")
    table_columns = [
        "setting_id",
        "fraction_alpha_zero",
        "alpha_mean",
        "alpha_median",
        "fraction_alpha_max",
        "subject_affinity_improvement_mean",
        "subject_affinity_CI_low",
        "subject_advantage_over_random_mean",
        "subject_advantage_over_random_CI_low",
        "class_accuracy_change",
        "class_logp_change",
        "manifold_knn_ratio_to_clean",
        "scst_off_manifold_rate",
        "random_off_manifold_rate",
        "all_gates_pass",
    ]
    result_table = compact[table_columns].to_markdown(index=False, floatfmt=".5f")

    rows = {row["setting_id"]: row for row in compact.to_dict(orient="records")}
    alpha_answers = {
        setting: {
            "fraction_zero": rows[setting]["fraction_alpha_zero"],
            "mean": rows[setting]["alpha_mean"],
            "median": rows[setting]["alpha_median"],
            "q25": rows[setting]["alpha_q25"],
            "q75": rows[setting]["alpha_q75"],
            "fraction_alpha_max": rows[setting]["fraction_alpha_max"],
            "realized_norm_ratio_mean": rows[setting]["realized_norm_ratio_mean"],
        }
        for setting in order
    }
    transport_answers = {
        setting: {
            "subject_affinity_mean": rows[setting]["subject_affinity_improvement_mean"],
            "subject_affinity_ci_low": rows[setting]["subject_affinity_CI_low"],
            "advantage_over_random_mean": rows[setting]["subject_advantage_over_random_mean"],
            "advantage_over_random_ci_low": rows[setting]["subject_advantage_over_random_CI_low"],
            "class_accuracy_change": rows[setting]["class_accuracy_change"],
            "class_logp_change": rows[setting]["class_logp_change"],
            "knn_ratio": rows[setting]["manifold_knn_ratio_to_clean"],
            "scst_off_manifold_rate": rows[setting]["scst_off_manifold_rate"],
            "random_off_manifold_rate": rows[setting]["random_off_manifold_rate"],
            "all_gates_pass": bool(rows[setting]["all_gates_pass"]),
        }
        for setting in order
    }

    answers: dict[str, Any] = {
        "1_repair2_frozen_before_outcomes": True,
        "1_evidence": {
            "protocol_commit": "ad86b25",
            "pre_outcome_hash_freeze_commit": "2ca4440",
            "pre_outcome_freeze": "protocol/PRE_STAGE0_REPAIR2_FREEZE.json",
        },
        "2_sealed_resources_untouched": True,
        "3_alpha_star_distribution": alpha_answers,
        "4_target_subject_affinity": "POSITIVE_WITH_CI_LOWER_ABOVE_ZERO_IN_ALL_FOUR_SETTINGS",
        "5_advantage_over_norm_matched_random": "POSITIVE_WITH_CI_LOWER_ABOVE_ZERO_IN_ALL_FOUR_SETTINGS",
        "6_class_fidelity": "PRESERVED_IN_ALL_FOUR_SETTINGS",
        "7_independent_session_3nn_ratio": {
            setting: transport_answers[setting]["knn_ratio"] for setting in order
        },
        "8_all_four_ratio_le_1_25": False,
        "9_binary_off_manifold_rate": {
            setting: {
                "scst": transport_answers[setting]["scst_off_manifold_rate"],
                "matched_random": transport_answers[setting]["random_off_manifold_rate"],
            }
            for setting in order
        },
        "10_transport_validity_supported": False,
        "11_exact_failure": "Both WBCIC 3NN-to-clean ratios exceeded the unchanged 1.25 gate (EEGNet 1.3079565485; EEGConformer 1.3407995963). All other frozen gates passed.",
        "12_trainable_scope": NOT_RUN,
        "13_bank_staleness_prevention": "NOT_APPLICABLE; no SCST model was trained",
        "14_matched_erm_future_ba": NOT_RUN,
        "15_scst_dr_future_ba": NOT_RUN,
        "16_ba_delta": NOT_RUN,
        "17_subject_level_ci": NOT_RUN,
        "18_positive_primary_settings": NOT_RUN,
        "19_scst_vs_mixup": NOT_RUN,
        "20_scst_vs_random_training": NOT_RUN,
        "21_class_conditioning_matters": "NOT_ESTABLISHED_AS_A_TRAINING_EFFECT; Stage-0 class compatibility was preserved",
        "22_decision_consistency_matters": NOT_RUN,
        "23_subject_identity_I": NOT_RUN,
        "24_transport_decision_sensitivity_D_T": NOT_RUN,
        "25_I_same_D_T_down_G_up": "NOT_SUPPORTED",
        "26_outer_authorized": False,
        "27_openbmi_outer_confirmation": "NOT_OPENED",
        "28_wbcic_outer_confirmation": "NOT_OPENED",
        "29_strongest_supported_claim": "Source-estimated subject-class residual directions are cross-session stable, and source-support-constrained transport improves target-subject affinity over norm-matched random while preserving class fidelity; validity does not generalize to the frozen WBCIC manifold criterion.",
        "30_stronger_unsupported_claim": "SCST-DR improves future-session or unseen-subject generalization, reduces decision sensitivity, retains identity during useful training, or beats ERM/Mixup/random augmentation.",
        "31_repair2_terminal": REPAIR2_TERMINAL,
        "31_final_terminal": FINAL_TERMINAL,
    }

    final_json = {
        "schema": "SCST_DR_FINAL_REPORT_V2",
        "final_terminal": FINAL_TERMINAL,
        "stage0_v0_terminal": "TRANSPORT_NOT_SUBJECT_FAITHFUL",
        "stage0_repair1_terminal": "TRANSPORT_OFF_MANIFOLD",
        "stage0_repair2_terminal": REPAIR2_TERMINAL,
        "repair2_validation_pass": True,
        "repair2_setting_pass_count": 2,
        "stage1_authorized": False,
        "stage1_status": NOT_RUN,
        "development_performance_status": NOT_RUN,
        "outer_status": "UNTOUCHED_UNENUMERATED_NOT_AUTHORIZED",
        "final_constructive_protocol_lock_created": False,
        "final_model_created": False,
        "outer_results_created": False,
        "closure_base_commit": c.git_head(),
        "repair2_validation_sha256": c.sha256(c.RESULTS / "STAGE0_REPAIR2_VALIDATION.json"),
        "repair2_result_sha256": c.sha256(c.RESULTS / "STAGE0_REPAIR2_FINAL_RESULT.json"),
        "answers": answers,
        "repair2_setting_results": transport_answers,
    }
    c.write_json(c.EXP / "SCST_DR_FINAL_REPORT.json", final_json)

    stage0_report = f"""# SCST-DR Stage-0 Repair-2 report

## Terminal

`{REPAIR2_TERMINAL}`

The protocol and execution hashes were frozen before Repair-2 outcomes.  The
operator kept the original residual direction, used only `final_embedding`,
capped alpha at 0.25, and selected the largest legal value on the fixed 1/64
grid using Session-1-only same-class source support.  Session 2 remained the
independent validity partition.  All 20 setting-by-fold units completed.

## Frozen-gate results

{result_table}

Both OpenBMI settings passed every gate.  Both WBCIC settings retained positive
target-subject affinity, positive advantage over norm-matched random, class
fidelity, and the binary off-manifold gate, but failed the unchanged absolute
3NN ratio gate: 1.30796 and 1.34080 versus the maximum 1.25.  Therefore 2/4,
not 4/4, settings passed and transport validity is not supported.

The centroid alpha distributions were not trivial zero operators.  Median alpha
was 0.25 in every setting; the fraction at alpha=0.25 ranged from 0.91942 to
0.99112.  Source support therefore rarely shortened the centroid transport
enough to repair the independent WBCIC manifold distance.

No Repair-3, SCST training, future-performance evaluation, or sealed outer
evaluation is authorized.
"""
    c.write_text(c.EXP / "STAGE0_REPAIR2_REPORT.md", stage0_report)

    final_report = f"""# SCST-DR final report

## Final terminal

`{FINAL_TERMINAL}`

Repair-2 ended at `{REPAIR2_TERMINAL}`.  The validator passed with 20/20 units
and 2/4 settings satisfying every frozen gate.  This is a validated scientific
failure, not a runtime failure.  The protocol forbids Repair-3, SCST training,
future-performance inspection, and sealed outer evaluation.

## Four-setting result

{result_table}

## Required 31 answers

1. **Frozen before outcomes:** yes; protocol commit `ad86b25`, pre-outcome hash
   freeze commit `2ca4440`.
2. **Sealed resources:** untouched and unenumerated.
3. **Alpha-star distribution:** setting summaries are in the table; all medians
   are 0.25, zero fractions are 0.00326-0.04167, and alpha-max fractions are
   0.91942-0.99112.  Full class/subject strata are preserved in
   `results/STAGE0_REPAIR2_ALPHA_DISTRIBUTION.csv`.
4. **Target-subject affinity:** positive with CI lower above zero in all four.
5. **Versus matched random:** positive with CI lower above zero in all four.
6. **Class fidelity:** passed in all four.
7. **Independent 3NN ratios:** OpenBMI EEGNet 1.16285; OpenBMI EEGConformer
   1.14937; WBCIC EEGNet 1.30796; WBCIC EEGConformer 1.34080.
8. **All four <=1.25:** no; both WBCIC settings failed.
9. **Binary off-manifold rates:** SCST rates were 0, 0, 0.01670, and 0.01119;
   matched-random rates were 0.00145, 0.00145, 0.05888, and 0.05337.
10. **Transport validity supported:** no.
11. **Exact failure:** the two WBCIC 3NN ratios exceeded 1.25 despite every
    other gate passing.
12. **Trainable scope:** not selected; training was prohibited.
13. **Bank staleness:** not applicable because no model was trained.
14. **Matched ERM future BA:** not run.
15. **SCST-DR future BA:** not run.
16. **Delta BA:** not run.
17. **Subject-level CI:** not run.
18. **Positive primary settings:** not evaluated as trained methods.
19. **SCST versus Mixup:** not run.
20. **SCST versus random augmentation:** not run.
21. **Class conditioning:** Stage-0 class compatibility is supported; a training
    contribution is not established.
22. **Decision consistency:** not run.
23. **Subject identity I:** not run after training.
24. **Decision sensitivity D_T:** not run.
25. **I retained / D_T down / G up:** not supported.
26. **Outer authorized:** no.
27. **OpenBMI outer confirmation:** not opened.
28. **WBCIC outer confirmation:** not opened.
29. **Strongest supported claim:** source residual directions are stable and
    constrained transport is subject-faithful, better than norm-matched random,
    and class-compatible, but is not manifold-valid across datasets.
30. **Unsupported stronger claim:** no SCST-DR generalization, mechanism, or
    superiority claim is justified.
31. **Terminal:** Repair-2 `{REPAIR2_TERMINAL}`; final
    `{FINAL_TERMINAL}`.

## Most serious limitation

The source-support rule rarely reduced centroid steps: at least 91.9% reached
alpha=0.25 in every setting.  It consequently did not enforce enough support on
the independent WBCIC geometry to satisfy the predeclared 1.25 criterion.  Any
further reduction, radius change, k change, setting exclusion, or outcome-aware
selection would be a new hypothesis, not an implementation repair.
"""
    c.write_text(c.EXP / "SCST_DR_FINAL_REPORT.md", final_report)

    c.write_text(
        c.EXP / "CLAIM_AUDIT.md",
        """# Claim audit

## Authorized

Across all four retained settings, source-estimated subject-class residual
directions are cross-session stable.  Source-support-constrained transport at
the final embedding improves target-subject affinity, beats realized-norm-
matched random displacement, and preserves the independent class probe.

## Not authorized

Transport validity across datasets is not supported because both WBCIC 3NN
ratios exceed the frozen 1.25 limit.  No claim about SCST-DR future-session BA,
unseen-subject generalization, decision sensitivity, identity retention,
superiority to ERM/Mixup/random/DANN, or sealed-outer confirmation is
authorized.  The final constructive hypothesis is not supported.
""",
    )
    c.write_text(
        c.EXP / "REPRODUCIBILITY.md",
        f"""# Reproducibility

- Parent experiment commit: `57d5e4f1ae0a7c80d95ca27983fedad2ec3f690c`.
- Server Python: `D:\\nips-temp\\TotalP\\P2\\.conda\\gpu-baseline-v1\\python.exe`.
- Development folds: frozen OpenBMI 40-subject and WBCIC 41-subject five-fold
  protocols; historical ERM seed 0.
- Repair-2 layer: `final_embedding` only.
- Repair-2 alpha solver: fixed grid 0..0.25 in 1/64 increments; largest
  Session-1 source-support-admissible value.
- Independent validity: Session 2 only; subject-cluster bootstrap with 10,000
  deterministic resamples.
- Runtime features are not committed.  Compact metrics, hashes, reports, and
  figures are committed.

`protocol/STAGE0_REPAIR2_PROTOCOL_LOCK.json` freezes the scientific protocol.
`protocol/PRE_STAGE0_REPAIR2_FREEZE.json` hashes the lock and execution code
before outcomes.  All 20 units verify their historical feature scope, scaling,
probe BA, and V0 hash.  `protocol/STAGE0_REPAIR2_E1_ENGINEERING_FREEZE.json`
locks the seven numerical outputs produced before the presentation-only E1
figure fix and requires byte-identical values after rerun.

`results/STAGE0_REPAIR2_VALIDATION.json` reports validator pass, 20 units, 2/4
all-gate settings, no future-session access, and sealed resources untouched.
Repair-2 result SHA256: `{c.sha256(c.RESULTS / 'STAGE0_REPAIR2_FINAL_RESULT.json')}`.
Repair-2 validation SHA256: `{c.sha256(c.RESULTS / 'STAGE0_REPAIR2_VALIDATION.json')}`.

Commands:

```powershell
& $python code\\freeze_stage0_repair2.py
& $python code\\run_stage0_repair2.py
& $python code\\analyze_stage0_repair2.py
& $python code\\validate_stage0_repair2.py
& $python code\\finalize_stage0_repair2_failure.py
& $python code\\validate_final_closure.py
```
""",
    )

    append_once(
        c.EXP / "ITERATION_LEDGER.md",
        "### V0.2 validated outcome",
        f"""### V0.2 validated outcome

- Actual result: validator pass with 20/20 units and 2/4 all-gate settings.
  OpenBMI ratios were 1.16285 and 1.14937; WBCIC ratios were 1.30796 and
  1.34080, above the unchanged 1.25 limit.  Subject affinity, advantage over
  matched random, class fidelity, and binary off-manifold gates passed in all
  four settings.
- Decision: `{REPAIR2_TERMINAL}`.  Stop the transport line permanently.  Do not
  create Repair-3, train SCST, inspect future performance, or open outer.
""",
    )
    append_once(
        c.EXP / "REPAIR_LOG.md",
        "### E1 verification complete",
        """### E1 verification complete

- Before/after verification: `STAGE0_REPAIR2_E1_ENGINEERING_FREEZE.json` locked
  the seven pre-fix numerical outputs; analyzer rerun required exact SHA256
  equality for every file.  The rerun passed, generated figures, and did not
  alter any numerical result or terminal.
""",
    )

    reason = (
        "Stage-0 Repair-2 validated TRANSPORT_VALIDITY_NOT_SUPPORTED: both "
        "WBCIC 3NN ratios exceeded 1.25; training and future/outer evaluation prohibited"
    )
    c.write_csv(
        c.RESULTS / "DEVELOPMENT_MAIN_RESULTS.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": reason, "setting_id": None, "method": None, "balanced_accuracy": None, "macro_f1": None, "nll": None}]),
    )
    c.write_csv(
        c.RESULTS / "PER_SUBJECT_RESULTS.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": reason, "setting_id": None, "subject_id": None, "method": None, "balanced_accuracy": None}]),
    )
    c.write_csv(
        c.RESULTS / "MECHANISM_RESULTS.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": reason, "setting_id": None, "method": None, "identity_evidence": None, "transport_decision_sensitivity": None}]),
    )
    c.write_csv(
        c.RESULTS / "BASELINE_COMPARISON.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": reason, "setting_id": None, "baseline": None, "scst_minus_baseline_ba": None}]),
    )
    c.write_json(
        c.RESULTS / "STATISTICAL_TESTS.json",
        {
            "schema": "SCST_DR_STATISTICAL_TESTS_V2",
            "status": NOT_RUN,
            "reason": reason,
            "tests": [],
            "trial_level_pseudoreplication_used": False,
        },
    )
    print(FINAL_TERMINAL, flush=True)


if __name__ == "__main__":
    main()
