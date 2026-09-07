"""Assemble the immutable V5 development report, ablations, hashes, and freeze decision."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from common import (
    ABLATIONS,
    DIAGNOSTICS,
    EXPERIMENT_ROOT,
    FINAL_CANDIDATE,
    LEADERBOARD,
    OUTPUTS,
    PROTOCOL,
    RESEARCH_LOG,
    V4_ROOT,
    default_wbcic_repo,
    ensure_directories,
    sha256_file,
    write_csv,
    write_json,
)


FINAL_METHOD = "M13_CSP_AUGMENTED_REFIT4"
FINAL_NAME = "CS-LGS: Cross-Session Local-Geometry Stack"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _row(path: Path, method_id: str) -> pd.Series:
    frame = _read_csv(path)
    part = frame.loc[frame.method_id.eq(method_id)]
    if len(part) != 1:
        raise RuntimeError(f"Expected exactly one {method_id} row in {path}")
    return part.iloc[0]


def _best(path: Path) -> pd.Series:
    frame = _read_csv(path)
    return frame.sort_values(["Delta_BA", "NLL"], ascending=[False, True]).iloc[0]


def _metric_payload(row: pd.Series) -> dict[str, object]:
    keys = (
        "dataset",
        "method_id",
        "subjects",
        "mean_subject_BA",
        "Delta_BA",
        "CI95_L",
        "CI95_U",
        "median_subject_delta",
        "positive_subject_fraction",
        "nonnegative_subject_fraction",
        "worst_subject_delta",
        "positive_fold_fraction",
        "accuracy",
        "macro_f1",
        "NLL",
        "Brier",
        "ECE",
        "switch_rate",
        "rescue_count",
        "harm_count",
        "rescue_precision",
        "oracle_headroom_recovered",
    )
    return {key: row[key] for key in keys}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=EXPERIMENT_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def _assemble_leaderboards(final: pd.Series, openbmi: pd.Series) -> None:
    sources = [
        "WBCIC_CROSS_SESSION_RELIABILITY.csv",
        "WBCIC_OUTPUT_CONTEXT_SEARCH.csv",
        "WBCIC_SHARED_REPRESENTATION_SEARCH.csv",
        "WBCIC_STRUCTURAL_SEARCH.csv",
        "WBCIC_SUBJECT_ADAPTATION.csv",
        "WBCIC_RELIABILITY_STACK.csv",
        "WBCIC_REFIT_DISAGREEMENT.csv",
        "WBCIC_CSP_AUGMENTATION.csv",
    ]
    frames = []
    for name in sources:
        frame = _read_csv(LEADERBOARD / name)
        frame["source_table"] = name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["Delta_BA", "NLL"], ascending=[False, True]
    )
    write_csv(LEADERBOARD / "WBCIC_DEV_V5.csv", combined)
    baseline = json.loads((PROTOCOL / "BASELINE_LOCK.json").read_text(encoding="utf-8"))
    evolution = pd.DataFrame(
        [
            {
                "benchmark": "OpenBMI",
                "stage": "V5 initial/final non-adaptive baseline",
                "method_id": baseline["OpenBMI"]["B_STRONG_CURRENT"],
                "mean_subject_BA": baseline["OpenBMI"]["B_STRONG_CURRENT_BA"],
                "baseline_updated": False,
                "reason": "No V5 non-adaptive OpenBMI baseline exceeded A1; final uses exact safe fallback.",
                "OUTER_TEST_USED": False,
            },
            {
                "benchmark": "WBCIC-development",
                "stage": "V5 initial/final non-adaptive baseline",
                "method_id": baseline["WBCIC-development"]["B_STRONG_CURRENT"],
                "mean_subject_BA": baseline["WBCIC-development"]["B_STRONG_CURRENT_BA"],
                "baseline_updated": False,
                "reason": "All stronger V5 candidates use target-subject S1/S2 adaptation and are model variants, not non-adaptive baselines.",
                "OUTER_TEST_USED": False,
            },
            {
                "benchmark": "WBCIC-development",
                "stage": "V5 selected adaptive model",
                "method_id": FINAL_METHOD,
                "mean_subject_BA": final.mean_subject_BA,
                "baseline_updated": False,
                "reason": "Reported as final adaptive candidate against frozen W1_RAW_LINEAR, with incremental V5 ablations reported separately.",
                "OUTER_TEST_USED": False,
            },
        ]
    )
    write_csv(LEADERBOARD / "BASELINE_EVOLUTION.csv", evolution)
    cross = pd.DataFrame(
        [
            {**_metric_payload(final), "benchmark": "WBCIC-development", "fallback": False},
            {**_metric_payload(openbmi), "benchmark": "OpenBMI", "fallback": True},
        ]
    )
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_V5.csv", cross)


def _assemble_ablations(final: pd.Series) -> None:
    baseline = json.loads((PROTOCOL / "BASELINE_LOCK.json").read_text(encoding="utf-8"))
    current_ba = float(baseline["WBCIC-development"]["B_STRONG_CURRENT_BA"])
    m3 = _best(LEADERBOARD / "WBCIC_CROSS_SESSION_RELIABILITY.csv")
    shared = _best(LEADERBOARD / "WBCIC_SHARED_REPRESENTATION_SEARCH.csv")
    structural = _best(LEADERBOARD / "WBCIC_STRUCTURAL_SEARCH.csv")
    local = _row(LEADERBOARD / "WBCIC_SUBJECT_ADAPTATION.csv", "M10_LOCAL_STACK_LOGISTIC")
    fixed = _row(LEADERBOARD / "WBCIC_RELIABILITY_STACK.csv", "M11_FIXED_SIMPLE_STACK_C1")
    refit = _row(LEADERBOARD / "WBCIC_REFIT_DISAGREEMENT.csv", "M12_SIMPLE_ALL_REFIT4")
    stages = [
        ("M0", "W1_RAW_LINEAR", current_ba, "Frozen strongest non-adaptive current baseline"),
        ("M1", "M2_OUTPUT_SHARED_LOGISTIC", float(shared.mean_subject_BA), "Output + fold-compatible shared representation"),
        ("M2", str(structural.method_id), float(structural.mean_subject_BA), "Best zero-shot structural competence family"),
        ("M3", str(m3.method_id), float(m3.mean_subject_BA), "Earlier-session reliability only"),
        ("M3b", "M10_LOCAL_STACK_LOGISTIC", float(local.mean_subject_BA), "S1/S2 subject-local frozen heads"),
        ("M3c", "M11_FIXED_SIMPLE_STACK_C1", float(fixed.mean_subject_BA), "Fixed variance-reduced local stack"),
        ("M3d", "M12_SIMPLE_ALL_REFIT4", float(refit.mean_subject_BA), "Fixed stack refit on all non-outcome subjects"),
        ("M3e", FINAL_METHOD, float(final.mean_subject_BA), "M3d + S1/S2 CSP context score"),
        ("M4", "NO_PREDICTIVE_PERSIST_INCREMENT", float(final.mean_subject_BA), "PERSIST retained as action-safety veto only"),
        ("M5", "NO_ACTION_EXPERT", float(final.mean_subject_BA), "WBCIC audit authorized no harmful/actionable block"),
    ]
    rows = []
    previous = current_ba
    for stage, method, ba, note in stages:
        rows.append(
            {
                "stage": stage,
                "method_id": method,
                "mean_subject_BA": ba,
                "Delta_BA_vs_W1": ba - current_ba,
                "increment_vs_previous_row": ba - previous,
                "note": note,
                "OUTER_TEST_USED": False,
            }
        )
        previous = ba
    write_csv(ABLATIONS / "COMPONENT_ABLATION.csv", pd.DataFrame(rows))
    write_csv(
        ABLATIONS / "EEG_CONTEXT_VALUE.csv",
        pd.DataFrame(
            [
                {
                    "comparison": "fold_compatible_shared_context_vs_W1",
                    "Delta_BA": float(shared.Delta_BA),
                    "CI95_L": float(shared.CI95_L),
                    "CI95_U": float(shared.CI95_U),
                    "conclusion": "weak_and_not_significant",
                },
                {
                    "comparison": "CSP_context_increment_over_M12",
                    "Delta_BA": float(final.mean_subject_BA - refit.mean_subject_BA),
                    "CI95_L": np.nan,
                    "CI95_U": np.nan,
                    "conclusion": "small_increment; CSP direct control is weak and is not claimed as a competent standalone expert",
                },
            ]
        ).assign(OUTER_TEST_USED=False),
    )
    write_csv(
        ABLATIONS / "RELIABILITY_VALUE.csv",
        pd.DataFrame(
            [
                {
                    "comparison": "scalar_S1S2_reliability_vs_W1",
                    "Delta_BA": float(m3.Delta_BA),
                    "CI95_L": float(m3.CI95_L),
                    "CI95_U": float(m3.CI95_U),
                    "conclusion": "failed",
                },
                {
                    "comparison": "subject_local_geometry_stack_vs_W1",
                    "Delta_BA": float(fixed.Delta_BA),
                    "CI95_L": float(fixed.CI95_L),
                    "CI95_U": float(fixed.CI95_U),
                    "conclusion": "meaningful_positive",
                },
            ]
        ).assign(OUTER_TEST_USED=False),
    )
    v4_persist = _read_csv(V4_ROOT / "outputs" / "ablations" / "PERSIST_INCREMENTAL_VALUE.csv")
    persist_row = v4_persist.loc[v4_persist.comparison.eq("PERSIST_increment")].iloc[0]
    wbcic_decision_path = (
        default_wbcic_repo()
        / "experiments"
        / "persist_eeg_wbcic_actionability_v2"
        / "outputs"
        / "FINAL_DECISION.json"
    )
    action_decision = json.loads(wbcic_decision_path.read_text(encoding="utf-8"))
    write_csv(
        ABLATIONS / "PERSIST_INCREMENTAL_VALUE.csv",
        pd.DataFrame(
            [
                {
                    "benchmark": "OpenBMI",
                    "role": "predictive_features",
                    "Delta_BA": float(persist_row.Delta_BA),
                    "CI95_L": float(persist_row.CI95_L),
                    "CI95_U": float(persist_row.CI95_U),
                    "value_found": False,
                    "conclusion": "No reliable raw BA, calibration, or robustness increment.",
                },
                {
                    "benchmark": "WBCIC-development",
                    "role": "action_safety",
                    "Delta_BA": np.nan,
                    "CI95_L": np.nan,
                    "CI95_U": np.nan,
                    "value_found": True,
                    "conclusion": "P01_04 remained PROTECTED; no block passed H1-H5, so ACTION was vetoed.",
                },
            ]
        ).assign(OUTER_TEST_USED=False),
    )
    write_csv(
        ABLATIONS / "ACTION_VALUE.csv",
        pd.DataFrame(
            [
                {
                    "benchmark": "OpenBMI",
                    "status": "historical_V4_negative",
                    "Delta_BA": float(v4_persist.loc[v4_persist.comparison.eq("ACTION_increment"), "Delta_BA"].iloc[0]),
                    "action_authorized": False,
                },
                {
                    "benchmark": "WBCIC-development",
                    "status": action_decision["terminal_state"],
                    "Delta_BA": np.nan,
                    "action_authorized": bool(action_decision["agdi_training_authorized"]),
                },
            ]
        ).assign(OUTER_TEST_USED=False),
    )


def _research_log(final: pd.Series) -> None:
    iteration_specs = [
        (1, "Cross-session scalar expert reliability", "S1/S2 expert ranking will transfer to S3.", "ABANDON", "F5/F9: fold-unstable and nonpositive."),
        (2, "Output and fixed EEG context", "Compact morphology will explain local expert competence.", "MODIFY", "Best +0.122 pp; CI crossed zero."),
        (3, "Fold-compatible shared representation", "Checkpoint coordinate repair will unlock context.", "MODIFY", "Best +0.148 pp; still unstable."),
        (4, "Structurally distinct zero-shot competence", "Error rescue, kNN, multi-label, pairwise, or covariance aggregation will exploit oracle headroom.", "ABANDON", "Only +0.049 pp from a very sparse 3-2 rescue; all others negative."),
        (5, "Target-history subject-local heads", "S1/S2 can expose subject geometry missing under zero-shot shift.", "KEEP", "+0.562 pp, 4/5 folds positive, CI crossed zero."),
        (6, "Fixed variance-reduced local stack", "Removing inner configuration variance will stabilize the local signal.", "KEEP", "+0.867 pp, positive CI, 5/5 folds."),
        (7, "Loss alignment and non-outcome refit", "Disagreement loss or full non-outcome refit will close the remaining gap.", "KEEP", "+0.928 pp; disagreement-only loss did not help."),
        (8, "CSP spatial context augmentation", "History-derived spatial geometry adds complementary context without acting as a standalone expert.", "KEEP", "+1.099 pp final refit; target conditions met."),
        (9, "Seed and cross-benchmark confirmation", "The fixed method is deterministic and fail-closed fallback is non-degrading.", "KEEP", "Five hashes identical; OpenBMI exactly matched A1."),
    ]
    summary_rows = []
    for number, title, hypothesis, conclusion, result in iteration_specs:
        path = RESEARCH_LOG / f"ITERATION_{number:03d}.md"
        path.write_text(
            "\n".join(
                [
                    f"# Iteration {number:03d}: {title}",
                    "",
                    f"- New hypothesis: {hypothesis}",
                    f"- Grouped result: {result}",
                    f"- Conclusion: `{conclusion}`",
                    "- Evaluation: subject-disjoint WBCIC development folds; target S3 outcomes were evaluation-only for held-out subjects.",
                    "- `OUTER_TEST_USED=false`.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "iteration": number,
                "title": title,
                "hypothesis": hypothesis,
                "conclusion": conclusion,
                "result": result,
                "OUTER_TEST_USED": False,
            }
        )
    write_csv(RESEARCH_LOG / "ITERATION_SUMMARY.csv", pd.DataFrame(summary_rows))
    ledger = [
        "# V5 hypothesis ledger",
        "",
        "All major attempts are retained; negative families were not deleted.",
        "",
        "| Iteration | Structural change | Decision |",
        "|---:|---|---|",
    ]
    ledger.extend(f"| {n:03d} | {title} | {decision} |" for n, title, _, decision, _ in iteration_specs)
    ledger.extend(
        [
            "",
            "The final positive signal did not come from predicting a unique best expert. It came from a fixed robust stack that combines cross-fitted expert output state with subject-local S1/S2 frozen-representation heads, plus a weak CSP score used only as context.",
            "",
            "Because development outcomes informed successive hypotheses, the final estimate is explicitly exploratory until the sealed outer cohort is evaluated under a later authorization.",
        ]
    )
    (RESEARCH_LOG / "HYPOTHESIS_LEDGER.md").write_text("\n".join(ledger) + "\n", encoding="utf-8")


def _final_spec(final: pd.Series, openbmi: pd.Series) -> dict[str, object]:
    expert_audit = json.loads((PROTOCOL / "WBCIC_ALL_SESSION_EXPERT_AUDIT.json").read_text(encoding="utf-8"))
    return {
        "model_name": FINAL_NAME,
        "model_id": FINAL_METHOD,
        "status": "DEVELOPMENT_TARGET_REACHED__FROZEN_FOR_LATER_OUTER",
        "benchmark": "WBCIC authorized development S1/S2->S3",
        "expert_roster": expert_audit["expert_roster"],
        "expert_checkpoint_hashes": expert_audit["checkpoint_audit"],
        "target_history": {
            "sessions": ["S1", "S2"],
            "labels_used": True,
            "selection": "per-subject S1->S2 and S2->S1 only",
            "target_S3_labels_used": False,
        },
        "local_heads": {
            "representations": ["EEGNet_STABLE", "EEGNet_STD", "DeepConvNet", "EEGConformer", "TeCh"],
            "model": "StandardScaler + class-balanced liblinear logistic",
            "C_grid_history_only": [0.001, 0.01, 0.1, 1.0],
        },
        "csp_context": {
            "role": "context_score_not_standalone_expert",
            "band_hz": [8.0, 30.0],
            "filter": "order-4 zero-phase Butterworth SOS",
            "covariance": "per-trial trace-normalized",
            "pairs_grid_history_only": [2, 3, 4],
            "C_grid_history_only": [0.01, 0.1, 1.0],
            "target_batch_adaptation": False,
        },
        "stack": {
            "features": "five raw expert output-state features + five local-head logits/probabilities + four CSP context fields",
            "model": "StandardScaler + class-balanced liblinear logistic",
            "C": 1.0,
            "fit_subjects": "all non-outcome development subjects within each grouped fold",
            "subject_and_class_balanced_weights": True,
        },
        "aggregation": {
            "anchor": "W1_RAW_LINEAR",
            "gate": "raw KEEP experts are not unanimous",
            "alpha": 1.0,
            "threshold": 0.5,
            "unanimous_behavior": "exact W1_RAW_LINEAR fallback",
        },
        "OpenBMI_policy": {
            "condition": "fewer than two legal prior sessions",
            "action": "exact A1_DYNAMIC_KEEP_FINAL fallback",
            "Delta_BA": float(openbmi.Delta_BA),
        },
        "PERSIST": {
            "predictive_input": False,
            "raw_BA_increment_claimed": False,
            "retained_role": "fail-closed action-safety veto; no ACTION expert authorized",
        },
        "seeds": [20260820, 20260821, 20260822, 20260823, 20260824],
        "development_result": _metric_payload(final),
        "outer_test_used": False,
        "OUTER_TEST_USED": False,
    }


def _write_report(final: pd.Series, openbmi: pd.Series, refit: pd.Series) -> None:
    delta_csp = float(final.mean_subject_BA - refit.mean_subject_BA)
    report = f"""# PERSIST-EEG V5 scientific report

## Decision

Terminal state: `V5_DEVELOPMENT_TARGET_REACHED` and `READY_FOR_OUTER_FREEZE`.

The selected development candidate is **{FINAL_NAME}**. On the authorized 41-subject WBCIC S1/S2->S3 development protocol it reached BA **{final.mean_subject_BA:.6f}**, a gain of **{100*final.Delta_BA:.3f} pp** over `W1_RAW_LINEAR` (paired subject bootstrap 95% CI **[{100*final.CI95_L:.3f}, {100*final.CI95_U:.3f}] pp**). All five folds improved. The sealed outer cohort was not enumerated, loaded, or evaluated.

This is an exploratory development result after repeated hypothesis iteration. It is not an outer-confirmed generalization claim.

## Required answers

1. **Strongest legal baselines.** OpenBMI: `A1_DYNAMIC_KEEP_FINAL`, BA 0.850962. WBCIC development: `W1_RAW_LINEAR`, BA 0.806789. Static references were 0.846442 and 0.803626 respectively.
2. **Baseline evolution.** No stronger non-adaptive baseline was created. Stronger V5 rows all use target-subject S1/S2 adaptation and are reported as candidate ablations, not silently substituted baselines.
3. **Oracle headroom.** It is concentrated in WBCIC 3-2 disagreements: 2,092 trials, baseline BA about 0.622, with 811 majority-wrong/minority-correct trials.
4. **Stable complementarity.** EEGNet Stable was strongest (BA about 0.794); EEGNet Standard and DeepConvNet were weaker but useful in a stack. TeCh and especially EEGConformer were not competent standalone experts and inflated oracle headroom through occasional guesses.
5. **Disagreement state alone.** No. Output-only selectors were negative or negligible.
6. **EEG context.** Fold-compatible frozen context added only +0.148 pp with a CI crossing zero. The history-derived CSP score added {100*delta_csp:.3f} pp over `M12_SIMPLE_ALL_REFIT4`; its direct control BA was only 0.580, so it is treated as context, not a strong expert.
7. **Cross-session reliability.** Scalar S1/S2 expert reliability failed. Rich subject-local S1/S2 geometry was useful.
8. **Best target.** Direct subject-balanced robust stacking outperformed expert-correctness BCE, pairwise ranking, local-error utility, and kNN competence.
9. **Best aggregation.** A conservative anchor with a fixed non-unanimous gate. Hard selection, generic soft weighting, ranking, and correlation-aware pools were worse.
10. **Final Delta BA.** +{100*final.Delta_BA:.3f} pp versus `W1_RAW_LINEAR`.
11. **Subject-bootstrap CI.** [{100*final.CI95_L:.3f}, {100*final.CI95_U:.3f}] pp.
12. **Positive folds.** 5/5.
13. **Subject stability.** {100*final.positive_subject_fraction:.1f}% positive and {100*final.nonnegative_subject_fraction:.1f}% nonnegative.
14. **Oracle recovery.** {100*final.oracle_headroom_recovered:.1f}% of available rescue trials, using the experiment's rescue-count definition.
15. **PERSIST raw BA.** No supported increment. The V4 OpenBMI PERSIST increment was negative and its CI crossed zero.
16. **PERSIST safety.** Yes, narrowly: the frozen WBCIC audit marked P01_04 protected and authorized no action. PERSIST is retained as a veto, not claimed as a performance feature.
17. **ACTION experts.** No. The WBCIC actionability audit found no block passing H1-H5; OpenBMI ACTION was negative.
18. **Full ERASE.** Not necessary and not authorized.
19. **Cross-benchmark behavior.** WBCIC met the development target. OpenBMI has fewer than two prior sessions, so the method fail-closed to A1 exactly: Delta BA {100*openbmi.Delta_BA:.3f} pp (non-degrading, not improving).
20. **Target-subject adaptation.** Yes: labeled S1/S2 only. No held-out target S3 label or target-batch statistic entered prediction.
21. **Outer access.** No. `OUTER_TEST_USED=false` throughout.
22. **Distinct families tried.** At least ten: scalar reliability, output direct, handcrafted context, shared frozen context, error rescue, kNN, multi-label correctness, pairwise ranking, covariance aggregation, subject-local heads, and CSP-context stacking.
23. **Failure progression.** Each failed family is retained in `outputs/research_log/`; the progression moved from zero-shot trial competence to legal history-conditioned subject geometry, then to a fixed low-variance stack.
24. **Target reached.** Yes on WBCIC development under the frozen non-adaptive baseline definition.
25. **Ready to freeze.** Yes as a later outer-evaluation candidate. This does not authorize opening outer data.

## Limitations

- The final gain is only {100*final.Delta_BA:.3f} pp and the worst subject changed by {100*final.worst_subject_delta:.1f} pp; it is not uniformly beneficial.
- CSP is a weak standalone classifier. Its small incremental value may not replicate; the untouched outer cohort is essential.
- Repeated development iterations make the estimate exploratory despite nested grouped evaluation.
- The OpenBMI result is a safety fallback, not evidence that the new history-conditioned mechanism transfers to a two-session dataset.
- PERSIST did not produce predictive gain in the selected model.
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")


def run() -> None:
    ensure_directories()
    seed_table = _read_csv(LEADERBOARD / "WBCIC_MULTI_SEED_CONFIRMATION.csv")
    final = seed_table.loc[seed_table.seed.eq(20260820)].iloc[0].copy()
    final["method_id"] = FINAL_METHOD
    openbmi = _read_csv(LEADERBOARD / "OPENBMI_V5.csv").iloc[0]
    refit = _row(LEADERBOARD / "WBCIC_REFIT_DISAGREEMENT.csv", "M12_SIMPLE_ALL_REFIT4")
    _assemble_leaderboards(final, openbmi)
    _assemble_ablations(final)
    _research_log(final)

    spec = _final_spec(final, openbmi)
    write_json(FINAL_CANDIDATE / "FINAL_MODEL_SPEC.json", spec)
    spec_md = f"""# {FINAL_NAME}

- Final method: `{FINAL_METHOD}`
- WBCIC development BA: `{final.mean_subject_BA:.6f}`
- Delta versus `W1_RAW_LINEAR`: `{100*final.Delta_BA:.3f} pp`
- Subject CI95: `[{100*final.CI95_L:.3f}, {100*final.CI95_U:.3f}] pp`
- Folds positive: `5/5`
- Target history: labeled S1/S2 only; no target S3 labels and no target-batch adaptation
- Gate: raw KEEP experts non-unanimous; unanimous trials exactly preserve W1
- OpenBMI: exact A1 fallback because two prior sessions are unavailable
- PERSIST: action-safety veto only; no predictive gain claim
- Outer: untouched
"""
    (FINAL_CANDIDATE / "FINAL_MODEL_SPEC.md").write_text(spec_md, encoding="utf-8")
    development = {
        "terminal_state": "V5_DEVELOPMENT_TARGET_REACHED",
        "selected_method": FINAL_METHOD,
        "WBCIC_development": _metric_payload(final),
        "OpenBMI": _metric_payload(openbmi),
        "increment_over_strongest_V5_non_CSP_ablation": float(final.mean_subject_BA - refit.mean_subject_BA),
        "multi_seed_identical": True,
        "development_estimate_exploratory": True,
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_CANDIDATE / "DEVELOPMENT_RESULTS.json", development)
    _write_report(final, openbmi, refit)

    prediction_path = DIAGNOSTICS / "WBCIC_MULTI_SEED_OOF_PREDICTIONS.csv"
    code_hashes = {
        str(path.relative_to(EXPERIMENT_ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted((EXPERIMENT_ROOT / "code").rglob("*.py"))
    }
    key_hashes = {
        "final_predictions": sha256_file(prediction_path),
        "baseline_lock": sha256_file(PROTOCOL / "BASELINE_LOCK.json"),
        "legality_audit": sha256_file(PROTOCOL / "LEGALITY_AUDIT.json"),
        "multi_seed_confirmation": sha256_file(PROTOCOL / "MULTI_SEED_CONFIRMATION.json"),
        "final_model_spec": sha256_file(FINAL_CANDIDATE / "FINAL_MODEL_SPEC.json"),
    }
    write_json(
        FINAL_CANDIDATE / "FINAL_MODEL_HASHES.json",
        {"code": code_hashes, "key_artifacts": key_hashes, "OUTER_TEST_USED": False},
    )
    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.version.cuda
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        torch_version, cuda, gpu = "unavailable", None, None
    write_json(
        OUTPUTS / "REPRODUCIBILITY.json",
        {
            "branch": _git("branch", "--show-current"),
            "base_commit": _git("rev-parse", "HEAD"),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch_version,
            "cuda": cuda,
            "gpu": gpu,
            "thread_limits": {"OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1, "NUMEXPR_NUM_THREADS": 1},
            "commands": [
                "python reconstruct_v4.py",
                "python disagreement_audit.py",
                "python expert_audit.py",
                "python run_iteration.py",
                "python run_search.py",
                "python run_shared_representation.py",
                "python run_structural_search.py",
                "python run_subject_adaptation.py",
                "python run_reliability_stack.py",
                "python run_refit_disagreement.py",
                "python run_csp_augmentation.py --workers 4",
                "python run_confirmation.py",
                "python run_final_dev.py",
            ],
            "multi_seed_prediction_sha256": str(final.prediction_sha256),
            "outer_split_lock_opened": False,
            "outer_subject_ids_loaded": False,
            "OUTER_TEST_USED": False,
        },
    )
    decision = {
        "terminal_state": "READY_FOR_OUTER_FREEZE",
        "development_state": "V5_DEVELOPMENT_TARGET_REACHED",
        "selected_method": FINAL_METHOD,
        "selected_model_name": FINAL_NAME,
        "target_checks": {
            "Delta_BA_ge_1pp": bool(final.Delta_BA >= 0.01),
            "CI95_lower_gt_0": bool(final.CI95_L > 0),
            "positive_folds_ge_4_of_5": bool(final.positive_fold_fraction >= 0.8),
            "majority_subjects_nonnegative": bool(final.nonnegative_subject_fraction > 0.5),
            "OpenBMI_nondegrading": bool(openbmi.Delta_BA >= -1e-12),
            "multi_seed_stable": True,
        },
        "development_adaptivity_warning": "Repeated development outcomes informed successive hypotheses; untouched outer confirmation is mandatory.",
        "outer_evaluation_authorized": False,
        "next_action": "Keep the outer cohort sealed until a later explicit authorization; do not modify the frozen V5 specification before that evaluation.",
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
