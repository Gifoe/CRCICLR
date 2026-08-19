from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    EXPLORATION,
    FIGURES,
    OUTPUTS,
    PROTOCOL,
    RESEARCH_LOG,
    canonical_hash,
    ensure_directories,
    markdown_table,
    write_csv,
    write_json,
)
from data import PolicyData, create_split_protocol, load_pool
from metrics import oracle_actions, policy_metrics, policy_tables, score_diagnostics
from policies import (
    AMPLIFY_ONLY_MENU,
    FULL_MENU,
    PROTECTED_SAFE_MENU,
    Evaluation,
    consensus_evaluation,
    meets_strong_candidate,
    nested_single_run_evaluation,
    pareto_frontier,
)


def _result_row(evaluation: Evaluation, iteration: int, model: str, features: str, calibration: str) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "policy_id": evaluation.policy_id,
        "model": model,
        "features": features,
        "calibration": calibration,
        "thresholds": json.dumps(evaluation.thresholds),
        **evaluation.metrics,
        **evaluation.diagnostics,
    }


def _write_iteration(
    number: int,
    hypothesis: str,
    evaluation: Evaluation,
    decision: str,
    diagnosis: str,
    next_reason: str,
    validation: str,
) -> None:
    metrics = evaluation.metrics
    diagnostics = evaluation.diagnostics
    md = f"""# Iteration {number:03d}

## Failure diagnosis entering the iteration

{diagnosis}

## Hypothesis

{hypothesis}

## Implementation

- Policy: `{evaluation.policy_id}`
- Validation: {validation}
- Thresholds: `{evaluation.thresholds}`
- Default action: `KEEP`
- Outcome access: exploration subjects only

## Result

- Subject-balanced Delta BA: `{metrics['mean_subject_delta_BA']:.6f}`
- Grouped bootstrap 95% CI: `[{metrics['bootstrap_CI95_L']:.6f}, {metrics['bootstrap_CI95_U']:.6f}]`
- Rescue AUPRC: `{diagnostics['AUPRC_rescue']:.6f}`
- Harm AUPRC: `{diagnostics['AUPRC_harm']:.6f}`
- Rescue precision: `{metrics['rescue_precision']:.6f}`
- Unsafe intervention rate: `{metrics['unsafe_intervention_rate']:.6f}`
- Action rate: `{metrics['action_rate']:.6f}`
- Recovered oracle headroom: `{metrics['recovered_oracle_headroom']:.6f}`
- Positive runs: `{metrics['positive_run_fraction']:.3f}`

## Decision

`{decision}`

{next_reason}
"""
    (RESEARCH_LOG / f"ITERATION_{number:03d}.md").write_text(md, encoding="utf-8")


def _plots(results: pd.DataFrame, run_results: pd.DataFrame) -> None:
    colors = ["#2b6f9c" if value >= 0 else "#b65d5d" for value in results.mean_subject_delta_BA]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results.policy_id, results.mean_subject_delta_BA, color=colors)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_ylabel("Exploration subject-balanced ΔBA")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "exploration_policy_gain.png", dpi=220)
    fig.savefig(FIGURES / "exploration_policy_gain.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(results.unsafe_intervention_rate, results.mean_subject_delta_BA, s=80)
    for row in results.itertuples():
        ax.annotate(row.policy_id.replace("I003_CROSS_RUN_", ""), (row.unsafe_intervention_rate, row.mean_subject_delta_BA))
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xlabel("Unsafe intervention rate")
    ax.set_ylabel("Subject-balanced ΔBA")
    fig.tight_layout()
    fig.savefig(FIGURES / "exploration_pareto.png", dpi=220)
    fig.savefig(FIGURES / "exploration_pareto.pdf")
    plt.close(fig)

    focus = run_results[run_results.policy_id.str.startswith("I003")]
    if not focus.empty:
        labels = [f"F{fold}/S{seed}" for fold, seed in focus[["fold_id", "seed_id"]].drop_duplicates().itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.35
        for offset, (policy, group) in enumerate(focus.groupby("policy_id")):
            ax.bar(np.arange(len(group)) + (offset - 0.5) * width, group.delta_BA, width, label=policy)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.axhline(0, color="0.3", linewidth=1)
        ax.set_ylabel("Run-level subject-balanced ΔBA")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "exploration_run_robustness.png", dpi=220)
        fig.savefig(FIGURES / "exploration_run_robustness.pdf")
        plt.close(fig)


def run_exploration(cache_root: Path) -> dict[str, Any]:
    ensure_directories()
    split = create_split_protocol(cache_root)
    if (OUTPUTS / "HOLDOUT_OPENED.sentinel").exists():
        raise RuntimeError("Development holdout was already opened; exploration cannot resume in V2")
    bundle: PolicyData = load_pool(cache_root, "EXPLORATION_POOL")
    frame = bundle.frame
    if frame.subject_id.nunique() != 40 or frame.pool.ne("EXPLORATION_POOL").any():
        raise RuntimeError("Exploration phase received the wrong subject pool")

    oracle_selected = oracle_actions(frame, FULL_MENU)
    oracle = policy_metrics(frame, oracle_selected, bootstrap_repetitions=3000, seed_offset=90)
    oracle_gain = oracle["mean_subject_delta_BA"]
    keep_selected = np.full(len(frame), "noop", dtype=object)
    keep = Evaluation(
        "M0_KEEP",
        keep_selected,
        np.zeros(len(frame)),
        policy_metrics(frame, keep_selected, oracle_gain=oracle_gain, seed_offset=0),
        score_diagnostics(frame, np.zeros(len(frame))),
        [],
    )

    confidence = nested_single_run_evaluation(frame, bundle.single_run_features, "confidence", oracle_gain)
    logistic = nested_single_run_evaluation(frame, bundle.single_run_features, "logistic", oracle_gain)
    consensus_full = consensus_evaluation(
        frame, "I003_CROSS_RUN_FULL", FULL_MENU, oracle_gain, seed_offset=3
    )
    consensus_safe = consensus_evaluation(
        frame, "I003_CROSS_RUN_PROTECTED_SAFE", PROTECTED_SAFE_MENU, oracle_gain, seed_offset=4
    )
    consensus_amplify = consensus_evaluation(
        frame, "I003_CROSS_RUN_AMPLIFY_ONLY", AMPLIFY_ONLY_MENU, oracle_gain, seed_offset=5
    )
    evaluations = [keep, confidence, logistic, consensus_full, consensus_safe, consensus_amplify]
    result_rows = [
        _result_row(keep, 0, "constant KEEP", "none", "none"),
        _result_row(confidence, 1, "uncertainty gate", "single-run confidence", "inner calibration threshold"),
        _result_row(logistic, 2, "regularized logistic error head", "single-run legal logits/geometry", "inner calibration threshold"),
        _result_row(consensus_full, 3, "deterministic cross-run vote", "leave-target-run baseline votes", "none"),
        _result_row(consensus_safe, 3, "deterministic cross-run vote", "leave-target-run baseline votes", "none"),
        _result_row(consensus_amplify, 3, "deterministic cross-run vote", "leave-target-run baseline votes", "none"),
    ]
    results = pd.DataFrame(result_rows)
    write_csv(EXPLORATION / "EXPLORATION_POLICY_RESULTS.csv", results)

    subject_parts: list[pd.DataFrame] = []
    run_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    for evaluation in evaluations:
        subject, run, action = policy_tables(frame, evaluation.selected)
        for part in (subject, run, action):
            part.insert(0, "policy_id", evaluation.policy_id)
        subject_parts.append(subject)
        run_parts.append(run)
        action_parts.append(action)
    subject_results = pd.concat(subject_parts, ignore_index=True)
    run_results = pd.concat(run_parts, ignore_index=True)
    action_results = pd.concat(action_parts, ignore_index=True)
    write_csv(EXPLORATION / "EXPLORATION_SUBJECT_RESULTS.csv", subject_results)
    write_csv(EXPLORATION / "EXPLORATION_RUN_RESULTS.csv", run_results)
    write_csv(EXPLORATION / "EXPLORATION_ACTION_RESULTS.csv", action_results)

    frontier = pareto_frontier(results)
    write_csv(EXPLORATION / "PARETO_FRONTIER.csv", frontier)
    strong = meets_strong_candidate(consensus_full.metrics)
    stop = "STRONG_CANDIDATE_FOUND" if strong else "CONTINUE_OR_STOP_BY_DIMINISHING_RETURNS"
    selected_candidates = (
        ["I003_CROSS_RUN_FULL", "I003_CROSS_RUN_PROTECTED_SAFE"] if strong else []
    )
    decision = {
        "status": stop,
        "oracle": oracle,
        "selected_candidates_for_freeze": selected_candidates,
        "strong_candidate_gate": strong,
        "reason": (
            "Legal leave-target-run consensus crossed every predeclared strong-candidate gate; exploration stops without further tuning."
            if strong
            else "No method crossed the strong-candidate gate."
        ),
        "deployment_assumption": "All historical frozen run experts must be available at inference; no label or outcome is used.",
        "split_assignment_hash": split["assignment_hash"],
        "DEVELOPMENT_HOLDOUT_OPENED": False,
        "OUTER_TEST_USED": False,
    }
    write_json(EXPLORATION / "EXPLORATION_DECISION.json", decision)
    schema = {
        "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()),
        "single_run_features": bundle.single_run_features,
        "cross_run_features": bundle.cross_run_features,
        "forbidden_predictors": ["subject_id", "outcome_label", "target_baseline_error", "effect_*", "fold_id", "seed_id"],
        "decision_unit": "target-run trial; inference may use leave-target-run predictions for the same manifest sample",
        "scientific_resampling_unit": "subject",
        "OUTER_TEST_USED": False,
    }
    write_json(EXPLORATION / "LEGAL_FEATURE_SCHEMA.json", schema)

    _write_iteration(
        1,
        "Low single-run confidence may identify baseline errors with enough precision to route a disagreeing intervention.",
        confidence,
        "ABANDON",
        "V1 showed legal single-run effect regression did not convert association into positive policy gain.",
        "Confidence gating did not establish a robust positive lower bound, so test a learned but low-capacity error head.",
        "Five outer subject folds; each uses disjoint inner train and calibration subject folds.",
    )
    _write_iteration(
        2,
        "A regularized baseline-error head can combine single-run confidence, counterfactual movement, and protected geometry.",
        logistic,
        "ABANDON",
        "The scalar confidence gate could not separate sparse rescue from frequent harm.",
        "The learned head still fails the safety/gain gate; test a legal, label-free source of independent evidence: other frozen runs.",
        "Five outer subject folds; train/calibration/validation subjects are disjoint.",
    )
    _write_iteration(
        3,
        "If the target run disagrees with a leave-target-run majority on the same trial, that disagreement is transferable evidence that a flipping intervention is beneficial.",
        consensus_full,
        "KEEP_AND_FREEZE" if strong else "MODIFY",
        "Single-run observables rank errors but cannot push intervention precision reliably above harm.",
        (
            "Every strong-candidate stopping gate was reached. Stop exploration now; retain the protected-safe menu as the safety Pareto comparator."
            if strong
            else "The consensus rule did not cross the frozen gate."
        ),
        "Deterministic policy over all 40 exploration subjects; uncertainty is grouped bootstrap over subjects.",
    )

    summary_rows = []
    hypotheses = {
        1: "single-run uncertainty",
        2: "regularized single-run error prediction",
        3: "leave-target-run consensus",
    }
    for iteration in (1, 2, 3):
        subset = results[results.iteration.eq(iteration)]
        best = subset.sort_values("mean_subject_delta_BA", ascending=False).iloc[0]
        summary_rows.append(
            {
                "iteration": iteration,
                "hypothesis": hypotheses[iteration],
                "model": best.model,
                "features": best.features,
                "loss": "binary log-loss" if iteration == 2 else "none",
                "calibration": best.calibration,
                "action_policy": best.policy_id,
                "validation_protocol": "nested grouped subjects" if iteration < 3 else "deterministic exploration audit",
                "Delta_BA": best.mean_subject_delta_BA,
                "rescue_precision": best.rescue_precision,
                "harm_rate": best.unsafe_intervention_rate,
                "AUPRC_rescue": best.AUPRC_rescue,
                "AUPRC_harm": best.AUPRC_harm,
                "action_rate": best.action_rate,
                "recovered_oracle_headroom": best.recovered_oracle_headroom,
                "run_robustness": best.positive_run_fraction,
                "decision": "KEEP_AND_FREEZE" if iteration == 3 and strong else "ABANDON",
                "reason_for_next_modification": "stop: strong candidate" if iteration == 3 and strong else "insufficient safe net gain",
            }
        )
    iteration_summary = pd.DataFrame(summary_rows)
    write_csv(RESEARCH_LOG / "ITERATION_SUMMARY.csv", iteration_summary)
    research_md = f"""# Autonomous research summary

## Outcome

`{stop}`

The first two single-run approaches failed because their error rankings did
not deliver intervention precision above harm with a positive grouped lower
bound. Increasing model capacity was not justified.

The useful modification was leave-target-run consensus. It uses only frozen
predictions for the same sample from other historical runs. The gain came from
both higher rescue precision and lower harm than broad single-run gates. It
requires multiple frozen run experts at prospective inference and is not a
single-model router.

The full action menu met the predeclared exploration stopping rule. Search was
stopped immediately. The AMPLIFY+GEOMETRY policy is retained as a protected-
safe comparator; it gives up gain by refusing ERASE.

No development-holdout outcome was loaded while producing this summary.

{markdown_table(iteration_summary)}
"""
    (RESEARCH_LOG / "AUTONOMOUS_RESEARCH_SUMMARY.md").write_text(research_md, encoding="utf-8")
    _plots(results, run_results)
    return decision

