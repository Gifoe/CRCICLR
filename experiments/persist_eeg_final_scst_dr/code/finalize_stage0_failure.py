from __future__ import annotations

import json
from typing import Any

import pandas as pd

import common as c


FINAL_TERMINAL = "FINAL_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED"
NOT_RUN = "NOT_RUN_BY_STAGE0_GATE"


def append_once(path, marker: str, addition: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if marker not in existing:
        c.write_text(path, existing.rstrip() + "\n\n" + addition.strip())


def main() -> None:
    validation = c.read_json(c.RESULTS / "STAGE0_REPAIR1_VALIDATION.json")
    result = c.read_json(c.RESULTS / "STAGE0_REPAIR1_FINAL_RESULT.json")
    if validation.get("pass") is not True:
        raise RuntimeError("cannot close the hypothesis before Repair-1 validation passes")
    if result.get("terminal") != "TRANSPORT_OFF_MANIFOLD":
        raise RuntimeError("this closure is only valid for the frozen off-manifold terminal")
    if result.get("stage1_authorized") is not False or result.get("selected_alpha") is not None:
        raise RuntimeError("negative closure conflicts with the Repair-1 result")

    summary = pd.read_csv(c.RESULTS / "STAGE0_REPAIR1_LAYER_SUMMARY.csv")
    final_rows = summary[summary.layer == "final_embedding"].copy()
    compact_columns = [
        "setting_id",
        "alpha",
        "stability_effect_mean",
        "stability_CI_low",
        "subject_affinity_improvement_mean",
        "subject_affinity_CI_low",
        "class_accuracy_change",
        "class_logp_change",
        "manifold_knn_ratio_to_clean",
        "gate_subject_fidelity",
        "gate_class_fidelity",
        "gate_manifold",
        "all_gates_pass",
    ]
    final_table = final_rows[compact_columns].to_markdown(index=False, floatfmt=".5f")

    answers: dict[str, Any] = {
        "1_legal_development_resources": {
            "OpenBMI": "40 development subjects; model-fit and validation subjects; Sessions 1 and 2 only",
            "WBCIC": "41 development subjects; model-fit and validation subjects; Sessions 1 and 2 only",
            "forbidden": "outcome-subject rows, future-session performance rows, OpenBMI sealed internal 14, WBCIC outer 10",
        },
        "2_sealed_untouched": True,
        "3_residual_stability": "SUPPORTED in all four settings and both layers; all subject-bootstrap lower bounds exceeded zero",
        "4_target_subject_fidelity": "SUPPORTED at final_embedding for alpha=0.25 and alpha=0.50 in all four settings",
        "5_original_class_preservation": "SUPPORTED at final_embedding for both frozen alphas in all four settings by the independent probe gates",
        "6_on_manifold_vs_random": "Binary off-manifold rate was no worse than norm-matched random, but the absolute 3NN-to-clean ratio exceeded the frozen 1.25 gate on WBCIC at alpha=0.25 and on every setting at alpha=0.50; transport validity therefore failed",
        "7_selected_layer": None,
        "7_layer_reason": "No globally valid layer/alpha exists. final_embedding was the strongest candidate but failed the frozen manifold gate",
        "8_representation_drift": "NOT_APPLICABLE; Stage 1 was prohibited before any continuation training",
        "9_strongest_matched_erm": "NOT_RUN; historical ERM checkpoints were used only as frozen Stage-0 representations",
        "10_scst_future_session_ba": NOT_RUN,
        "11_paired_ba_delta_vs_erm": NOT_RUN,
        "12_uncertainty_interval": NOT_RUN,
        "13_settings_favoring_scst": NOT_RUN,
        "14_scst_beats_mixup": NOT_RUN,
        "15_scst_beats_random": NOT_RUN,
        "16_class_conditioning_matters": "NOT_ESTABLISHED_AS_A_TRAINING_EFFECT; Stage-0 class conditioning preserved probe semantics but no model comparison was authorized",
        "17_decision_consistency_matters": NOT_RUN,
        "18_subject_identity_evidence": NOT_RUN,
        "19_transport_decision_sensitivity": NOT_RUN,
        "20_signature_I_retained_D_down_G_up": "NOT_SUPPORTED",
        "21_outer_authorized": False,
        "22_outer_confirmation": "NOT_OPENED",
        "23_supported_claim": "Source-estimated subject-class residual directions are cross-session stable and, at reduced magnitude, increase target-subject affinity while preserving an independent class probe; they are not certified as manifold-valid across datasets",
        "24_unsupported_stronger_claim": "SCST-DR improves future-session or unseen-subject generalization, reduces decision sensitivity, or shows that subject robustness does not require subject invariance",
        "25_final_terminal": FINAL_TERMINAL,
    }
    final_json = {
        "schema": "SCST_DR_FINAL_REPORT_V1",
        "final_terminal": FINAL_TERMINAL,
        "stage0_v0_terminal": "TRANSPORT_NOT_SUBJECT_FAITHFUL",
        "stage0_repair1_terminal": result["terminal"],
        "repair1_validation_pass": True,
        "eligible_global_alphas": result.get("eligible_alphas", []),
        "selected_alpha": None,
        "selected_layers": {},
        "stage1_status": NOT_RUN,
        "development_performance_status": NOT_RUN,
        "outer_status": "UNTOUCHED_UNENUMERATED_NOT_AUTHORIZED",
        "final_constructive_protocol_lock_created": False,
        "final_model_created": False,
        "outer_results_created": False,
        "closure_base_commit": c.git_head(),
        "repair1_validation_sha256": c.sha256(c.RESULTS / "STAGE0_REPAIR1_VALIDATION.json"),
        "answers": answers,
        "final_embedding_stage0_summary": final_rows[compact_columns].to_dict(orient="records"),
    }
    c.write_json(c.EXP / "SCST_DR_FINAL_REPORT.json", final_json)

    report = f"""# SCST-DR final report

## Final terminal

`{FINAL_TERMINAL}`

The constructive method was stopped at the validated Stage-0 terminal
`TRANSPORT_OFF_MANIFOLD`.  No SCST-DR continuation model, matched continuation
baseline, future-session performance analysis, mechanism analysis, final model
lock, or sealed-outer evaluation was run.

## What the repair established

V0 showed stable residual direction but alpha=1 overshoot.  The only authorized
scientific repair prelocked one global magnitude from {{0.25, 0.50}} without
changing layers, folds, seed, representations, centroids, controls, probes,
manifold estimator, bootstrap, or gates.  Alpha=0.25 repaired target-subject and
class fidelity at the final embedding in all four settings.  It still produced
3NN-to-clean manifold ratios of 1.34457 (WBCIC EEGNet) and 1.36594 (WBCIC
EEGConformer), above the frozen 1.25 maximum.  Alpha=0.50 violated the manifold
gate in every setting (ratios 1.33823-1.73031 across candidate layers).

The binary off-manifold rate relative to a norm-matched random perturbation was
not the failing component.  The failure is the absolute distance from real
same-class centroid support.  Passing subject affinity and class-probe gates is
therefore insufficient to certify the counterfactual representation.

## Final-embedding frozen-gate results

{final_table}

## Required answers

1. **Legal development resources:** OpenBMI 40 development subjects and WBCIC
   41 development subjects, using only frozen model-fit/validation roles and
   legal source Sessions 1/2.
2. **Sealed resources:** untouched and unenumerated.  OpenBMI internal 14 and
   WBCIC outer 10 were not opened.
3. **Residual stability:** supported in every setting/layer with subject-level
   bootstrap lower bounds above zero.
4. **Target-subject fidelity:** supported at `final_embedding` for both reduced
   alphas in all four settings.
5. **Class preservation:** supported at `final_embedding` for both reduced
   alphas by the independent probe gates.
6. **On-manifold validity:** not supported.  Although better than the random
   control on binary outlier rate, absolute 3NN support distance failed.
7. **Selected layer:** none; no layer/global-alpha combination passed all gates.
8. **Representation drift:** not applicable because training was prohibited.
9. **Strongest matched ERM:** not run as a continuation baseline.
10. **SCST future-session BA:** not run.
11. **Paired BA delta:** not run.
12. **BA uncertainty:** not run.
13. **Settings favoring SCST:** not evaluated.
14. **SCST versus Mixup:** not evaluated as trained methods.
15. **SCST versus random perturbation:** not evaluated as trained methods.
16. **Class conditioning:** source-side class fidelity is supported; a training
    contribution is not established.
17. **Decision consistency:** not evaluated.
18. **Subject identity:** not evaluated after training.
19. **Transport decision sensitivity:** not evaluated after training.
20. **I retained / D_T down / G up:** not supported.
21. **Outer authorization:** denied by Stage 0.
22. **Outer confirmation:** not opened.
23. **Supported claim:** reduced residual transport is directionally
    subject-faithful and class-compatible on source data, but not certified as
    manifold-valid across datasets.
24. **Unsupported claim:** no generalization, decision-robustness, or
    subject-invariance conclusion is justified.
25. **Final state:** `{FINAL_TERMINAL}`.

## Most serious limitation

The proposed arithmetic changes the target-subject and class-probe diagnostics
in the desired direction but does not stay sufficiently close to real WBCIC
same-class representation support.  Training on those points would test an
uncertified latent augmentation, not the stated subject-transport mechanism.
"""
    c.write_text(c.EXP / "SCST_DR_FINAL_REPORT.md", report)

    c.write_text(
        c.EXP / "CLAIM_AUDIT.md",
        """# Claim audit

## Authorized

Source-side subject-class residual direction is cross-session stable.  Reduced
magnitude transport improves target-subject affinity and preserves an
independent class probe at the final embedding across the four retained
settings.

## Not authorized

The transport is not certified on-manifold across datasets.  No claim about
SCST-DR future-session BA, unseen-subject generalization, decision sensitivity,
identity retention, superiority to ERM/Mixup/random/DANN, or sealed-outer
confirmation is authorized.  The final constructive hypothesis is not
supported under the frozen protocol.
""",
    )
    append_once(
        c.EXP / "README.md",
        "## Final status",
        f"""## Final status

`{FINAL_TERMINAL}`.  V0 failed subject fidelity at alpha=1.  The prelocked
magnitude-only repair restored subject/class fidelity at alpha=0.25 but failed
the WBCIC absolute manifold gate.  Stage 1 and sealed outer evaluation were not
authorized.  See `SCST_DR_FINAL_REPORT.md`.

Repair and closure commands:

```powershell
& $python code\\freeze_stage0_repair1.py
& $python code\\run_stage0_repair1.py
& $python code\\analyze_stage0_repair1.py
& $python code\\validate_stage0_repair1.py
& $python code\\finalize_stage0_failure.py
& $python code\\validate_final_closure.py
```
""",
    )
    append_once(
        c.EXP / "REPRODUCIBILITY.md",
        "## Repair-1 hash lock",
        """## Repair-1 hash lock

`protocol/PRE_STAGE0_REPAIR1_FREEZE.json` hashes the magnitude-only lock, all
three Repair-1 execution/analysis/validation programs, the unchanged common
implementation, and the validated V0 compact results before any Repair-1 unit
was computed.  Each of 40 Repair-1 units verifies its feature-scope, scaling
center, scaling scale, probe BA, and V0 unit hash before writing metrics.  The
raw pair-level Repair-1 CSVs remain under `runtime/stage0_repair1_units` and are
not committed; compact summaries, reports, and figures are committed.
""",
    )

    not_run_reason = "Stage-0 Repair 1 validated TRANSPORT_OFF_MANIFOLD; training and future/outer evaluation prohibited"
    c.write_csv(
        c.RESULTS / "DEVELOPMENT_MAIN_RESULTS.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": not_run_reason, "setting_id": None, "method": None, "balanced_accuracy": None, "macro_f1": None, "nll": None}]),
    )
    c.write_csv(
        c.RESULTS / "PER_SUBJECT_RESULTS.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": not_run_reason, "setting_id": None, "subject_id": None, "method": None, "balanced_accuracy": None}]),
    )
    c.write_csv(
        c.RESULTS / "MECHANISM_RESULTS.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": not_run_reason, "setting_id": None, "method": None, "identity_evidence": None, "transport_decision_sensitivity": None}]),
    )
    c.write_csv(
        c.RESULTS / "BASELINE_COMPARISON.csv",
        pd.DataFrame([{"status": NOT_RUN, "reason": not_run_reason, "setting_id": None, "baseline": None, "scst_minus_baseline_ba": None}]),
    )
    c.write_json(
        c.RESULTS / "STATISTICAL_TESTS.json",
        {"schema": "SCST_DR_STATISTICAL_TESTS_V1", "status": NOT_RUN, "reason": not_run_reason, "tests": [], "trial_level_pseudoreplication_used": False},
    )
    print(FINAL_TERMINAL, flush=True)


if __name__ == "__main__":
    main()
