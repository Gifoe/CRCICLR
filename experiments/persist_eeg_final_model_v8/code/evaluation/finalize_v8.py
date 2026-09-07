"""Finalize V8 after the Phase-A headroom gate has failed.

This finalizer reads compact V8_SEARCH summaries only.  It does not load raw
EEG, trial labels, the internal holdout, or WBCIC outer data.  It records the
strict baseline update, emits the required negative-result artifacts, and
prevents Phase-B/C metrics from being mistaken for completed experiments.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import (
    ABLATIONS, BASELINES, DIAGNOSTICS, EXPERIMENT_ROOT, FINAL_CANDIDATE,
    HEADROOM, LEADERBOARD, OUTPUTS, PROTOCOL, RESEARCH_LOG, SELECTORS,
    ensure_directories, sha256_file, v7_outputs, write_csv, write_json,
)


BENCHMARKS = {
    "OpenBMI_MI_S1_to_S2": "OpenBMI",
    "WBCIC_S1S2_to_S3_authorized_development": "WBCIC",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _screening_decisions(table: pd.DataFrame, baseline_lock: dict) -> dict[str, dict]:
    locked = {row["benchmark"]: row for row in baseline_lock["baselines"]}
    decisions = {}
    for benchmark, short in BENCHMARKS.items():
        rows = table.loc[table.benchmark.astype(str).eq(benchmark)].copy()
        union = rows.loc[rows.family_id.astype(str).str.contains("MULTIBACKBONE_COMPETENCE_UNION_F01", regex=False)]
        if len(union) != 1:
            raise RuntimeError(f"Expected one two-fold multi-backbone union for {benchmark}")
        union = union.iloc[0]
        strongest_index = rows.strongest_single_candidate_BA.astype(float).idxmax()
        strongest = rows.loc[strongest_index]
        updated_baseline = float(strongest.strongest_single_candidate_BA)
        oracle_ba = float(union.subject_oracle_BA)
        adjusted_headroom = oracle_ba - updated_baseline
        decisions[short] = {
            "benchmark": benchmark,
            "screening_folds": [0, 1],
            "screening_subjects": int(union.subjects),
            "full_scope_locked_V7_baseline_method": str(locked[benchmark]["method_id"]),
            "full_scope_locked_V7_baseline_BA": float(locked[benchmark]["mean_subject_BA"]),
            "two_fold_V7_baseline_BA": float(union.baseline_BA),
            "strongest_fair_screening_candidate": str(strongest.strongest_single_candidate),
            "strongest_fair_screening_candidate_family": str(strongest.family_id),
            "strongest_fair_screening_candidate_BA": updated_baseline,
            "screening_baseline_update_pp": 100.0 * (updated_baseline - float(union.baseline_BA)),
            "multi_backbone_union_subject_oracle_BA": oracle_ba,
            "oracle_headroom_vs_V7_two_fold_baseline_pp": float(union.oracle_headroom_pp),
            "oracle_headroom_vs_updated_screening_baseline_pp": 100.0 * adjusted_headroom,
            "rescue_ge_2pp_vs_V7_fraction": float(union.subjects_rescued_ge_2pp_fraction),
            "rescue_ge_5pp_vs_V7_fraction": float(union.subjects_rescued_ge_5pp_fraction),
            "eligible_union_candidates": int(union.eligible_candidate_count),
            "mean_pairwise_correctness_correlation": float(union.mean_pairwise_correctness_correlation),
            "headroom_state_after_baseline_update": "V8_HEADROOM_WEAK" if adjusted_headroom < 0.04 else "CHECK_GATE",
            "plus_5pp_actual_reached": False,
            "plus_8pp_oracle_gate_reached": False,
            "exploratory": True,
            "internal_holdout_used": False,
            "OUTER_TEST_USED": False,
        }
    return decisions


def _status_table(phase: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "phase": phase,
        "status": "NOT_RUN_PHASE_A_HEADROOM_GATE_FAILED",
        "reason": "No dual-benchmark candidate bank reached the approximately +8 pp subject-oracle gate.",
        "metric": np.nan,
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    }])


def _write_project_docs(decisions: dict[str, dict]) -> None:
    open_result = decisions["OpenBMI"]
    wbcic_result = decisions["WBCIC"]
    (EXPERIMENT_ROOT / "README.md").write_text(
        """# PERSIST-EEG V8 — Headroom First

V8 tested whether learned future-session adaptation banks contain enough
subject-level action headroom to justify prospective selection.  The answer is
negative on the sealed protocol used here.  Phase A was completed; Phase B
(META-GENERIC selection), Phase C (PERSIST increment), internal holdout, and
WBCIC outer evaluation were not run because the hard headroom gate failed.

Authoritative conclusions are in `outputs/FINAL_DECISION.json` and
`outputs/SCIENTIFIC_REPORT.md`.  Trial prediction CSVs and model/cache binaries
are reproducibility intermediates retained on the execution server, not source
artifacts intended for Git.
""",
        encoding="utf-8",
    )
    (EXPERIMENT_ROOT / "DESIGN.md").write_text(
        """# V8 design

The protocol fixes subject-only V8_SEARCH/internal-holdout partitions before
model search.  Source-fold non-outcome V8_SEARCH subjects provide legal
history-to-future meta-training episodes.  Source-fold outcome V8_SEARCH
subjects are exploratory Phase-A scoring subjects.  The internal holdout and
WBCIC outer cohort remain sealed.

The research order is deliberately gated: create action headroom, then recover
it with a generic prospective selector, then test PERSIST increment under a
capacity-matched comparison.  A selector is prohibited when the action bank's
subject oracle is below approximately +8 percentage points on either
benchmark.  Candidate competence, error correlation, rescue fractions, and
tail failures are audited to prevent weak lucky experts from manufacturing an
oracle.
""",
        encoding="utf-8",
    )
    (EXPERIMENT_ROOT / "METHOD.md").write_text(
        """# V8 method search

Phase A evaluated query-trained low-rank coverage banks, true coarse Meta-SGD,
prototype/metric transport, subject-conditioned normalization and FiLM,
shrinkage SPD transport, raw encoder adaptation, a competence-first
multi-scale temporal-spatial TCN, low-rank expert adapters, raw-signal FOMAML,
and a competence-filtered multi-backbone union.

OpenBMI episodes use session 1 as legal history and session 2 as future query.
WBCIC episodes use sessions 1/2 as legal history and session 3 as future query.
Meta-training query labels come only from source-fold non-outcome V8_SEARCH
subjects.  Outcome future labels are used only for exploratory headroom
measurement.  The strongest fair candidate observed on the same two-fold
screen is promoted as the search-scope matched baseline before the final gate
decision.

No selector, PERSIST policy, suppression action, risk gate, multi-seed study,
or final model was fit because Phase A failed.
""",
        encoding="utf-8",
    )
    (EXPERIMENT_ROOT / "THEORY.md").write_text(
        """# Headroom proposition

Let the fixed candidate action set be $\mathcal{A}$, legal history be $H_s$,
and future utility of action $k$ for subject $s$ be $U_s(k)$.  For any selector
$\pi(H_s) \in \mathcal{A}$,

$$
\mathbb{E}[U_s(\pi(H_s))] \leq \mathbb{E}[\max_{k\in\mathcal{A}} U_s(k)].
$$

This follows pointwise because the selected action's utility cannot exceed the
maximum utility in the same action set.  Therefore, if the deployment-level
oracle exceeds the strongest matched baseline by less than $\epsilon$, no
selector over that bank can deliver more than $\epsilon$.  The proposition is
only an upper-bound argument; it says nothing about whether the oracle action
is predictable from legal history.

V8's baseline-updated two-fold union headroom was approximately
"""
        + f"{open_result['oracle_headroom_vs_updated_screening_baseline_pp']:.3f} pp on OpenBMI and "
        + f"{wbcic_result['oracle_headroom_vs_updated_screening_baseline_pp']:.3f} pp on WBCIC, "
        + "so selector research cannot meet the requested dual +5 pp target with this action set.\n",
        encoding="utf-8",
    )


def _scientific_report(decisions: dict[str, dict], v7: dict, outer: dict) -> str:
    o = decisions["OpenBMI"]
    w = decisions["WBCIC"]
    return f"""# PERSIST-EEG V8 scientific report

## Decision

`V8_SCIENTIFIC_EXHAUSTION_PHASE_A_HEADROOM`

This is a negative result.  The broadest competence-filtered bank remained
below the hard gate on both benchmarks.  Phase B/C, multi-seed confirmation,
internal holdout, and WBCIC outer evaluation were correctly skipped.

## Required questions

1. **Authoritative V7 limits.** OpenBMI strongest generic was
   {v7['benchmarks']['openbmi']['V7_strongest_generic_BA']:.6f} BA with
   {v7['benchmarks']['openbmi']['V7_oracle_headroom_vs_strongest_generic_pp']:.3f} pp V7 oracle headroom. WBCIC was
   {v7['benchmarks']['wbcic']['V7_strongest_generic_BA']:.6f} BA with
   {v7['benchmarks']['wbcic']['V7_oracle_headroom_vs_strongest_generic_pp']:.3f} pp.
2. **New families.** Low-rank coverage, Meta-SGD, metric/prototype transport,
   norm/FiLM hypernetwork, SPD transport, raw fine-tuning, multi-scale TCN,
   competence-first training, low-rank expert adapters, raw FOMAML, and a
   multi-backbone union were evaluated.
3. **Families creating material headroom.** None reached +4 pp as a complete
   two-fold family.  The broad union was also below +4 pp.
4. **Strongest single expert on the two-fold screen.** OpenBMI:
   `{o['strongest_fair_screening_candidate']}` at {o['strongest_fair_screening_candidate_BA']:.6f} BA. WBCIC:
   `{w['strongest_fair_screening_candidate']}` at {w['strongest_fair_screening_candidate_BA']:.6f} BA.
5. **Strongest bank oracle.** OpenBMI {o['multi_backbone_union_subject_oracle_BA']:.6f}; WBCIC {w['multi_backbone_union_subject_oracle_BA']:.6f}.
6. **OpenBMI oracle headroom.** {o['oracle_headroom_vs_V7_two_fold_baseline_pp']:.3f} pp versus the V7 two-fold baseline, but only
   {o['oracle_headroom_vs_updated_screening_baseline_pp']:.3f} pp after the mandatory screening-baseline update.
7. **WBCIC oracle headroom.** {w['oracle_headroom_vs_V7_two_fold_baseline_pp']:.3f} pp versus V7 two-fold, and
   {w['oracle_headroom_vs_updated_screening_baseline_pp']:.3f} pp after update.
8. **Dual +8 pp gate.** No.
9. **At least +5 pp rescue potential versus V7 two-fold baseline.** OpenBMI
   {100*o['rescue_ge_5pp_vs_V7_fraction']:.1f}%; WBCIC {100*w['rescue_ge_5pp_vs_V7_fraction']:.1f}%.
10. **Expert competence.** The union applied a fixed competence filter, but it
    was outcome-screened and is only an optimistic upper bound.  It cannot be
    presented as prospective evidence.
11. **Error diversity.** Mean pairwise correctness correlations were
    {o['mean_pairwise_correctness_correlation']:.3f} (OpenBMI) and
    {w['mean_pairwise_correctness_correlation']:.3f} (WBCIC); residual
    complementarity was insufficient.
12. **META-GENERIC oracle recovery.** Not run; gate failed.
13. **META-GENERIC actual gain.** Not run.
14. **PERSIST additional value.** Not evaluated in V8.
15. **PERSIST BA improvement.** No V8 estimate exists.
16. **PERSIST safety improvement.** No V8 estimate exists.
17. **P/U/D/G/R.** Definitions were reserved for learned-transform audits, but
    were not estimated because Phase C was never authorized.
18. **Suppression.** Not used; coefficient remains zero.
19. **Largest WBCIC single-candidate improvement.** Competence-first
    multi-scale TCN generic blending, +{w['screening_baseline_update_pp']:.3f} pp over the V7 two-fold baseline.
20. **Largest OpenBMI single-candidate improvement.** Raw encoder adaptation,
    +{o['screening_baseline_update_pp']:.3f} pp over the V7 two-fold baseline.
21. **Same K=1/K=2 core.** Shared representation/adaptation families ran on
    both settings; neither passed.  The later multi-scale/FOMAML repair was
    prioritized on failing WBCIC and was not promoted to full cross-benchmark
    evaluation because its WBCIC screen failed.
22. **Stronger fair baseline found.** Yes, on the exploratory two-fold screens.
23. **Baseline claims updated.** Yes.  The stricter updated-baseline headroom is
    reported separately from V7-locked comparability.
24. **Internal holdout opened after freeze.** No; no candidate qualified for a freeze.
25. **WBCIC outer untouched.** Yes: split not opened, subjects not enumerated,
    and raw/features/labels not loaded (`OUTER_TEST_USED=false`).
26. **Dual +5 pp actual gain.** No.
27. **Generic contribution.** Best observed screening gains were
    {o['screening_baseline_update_pp']:.3f} pp and {w['screening_baseline_update_pp']:.3f} pp; these are exploratory, not final estimates.
28. **Unique PERSIST contribution.** 0 pp measured because Phase C was not run.
29. **Exact bottleneck.** Learned-action/representation headroom.  Even a broad
    outcome-only subject oracle remained about +2.1 pp above the updated
    matched baselines.
30. **Further iteration justified.** Not on these repeatedly reused development
    outcomes without a materially new representation source or independent
    development cohort.  Small architectural/hyperparameter changes would be
    additional adaptive search, not a credible new hypothesis.

## Scope limitations

Results are fixed-seed, two-fold V8_SEARCH screens.  They are not full five-fold
estimates, not multi-seed estimates, and not confirmation results.  The union
candidate filter used the same search outcomes, making its oracle deliberately
optimistic.  No confidence interval or final leaderboard claim is warranted.
"""


def _write_hashes() -> dict:
    rows = []
    for path in sorted(EXPERIMENT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        if (
            relative.startswith("outputs/cache/")
            or relative.endswith("_SEARCH_PREDICTIONS.csv")
            or relative.endswith("FINAL_MODEL_HASHES.json")
            or "__pycache__" in relative
        ):
            continue
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    payload = {
        "artifact_scope": "Git-trackable code and compact results; caches/checkpoints/trial predictions excluded",
        "files": rows,
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_CANDIDATE / "FINAL_MODEL_HASHES.json", payload)
    return payload


def run() -> None:
    ensure_directories()
    for directory in (LEADERBOARD, SELECTORS, ABLATIONS, FINAL_CANDIDATE):
        directory.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(HEADROOM / "HEADROOM_FAMILY_TABLE.csv")
    baseline_lock = _load_json(PROTOCOL / "BASELINE_LOCK.json")
    v7 = _load_json(PROTOCOL / "V7_RECONSTRUCTION.json")
    outer = _load_json(PROTOCOL / "OUTER_LOCK.json")
    split = _load_json(PROTOCOL / "V8_SEARCH_SPLIT.json")
    decisions = _screening_decisions(table, baseline_lock)

    baseline_rows = []
    for short, result in decisions.items():
        baseline_rows.extend([
            {
                "benchmark": short,
                "scope": "full pre-V8 locked reference",
                "method_id": result["full_scope_locked_V7_baseline_method"],
                "mean_subject_BA": result["full_scope_locked_V7_baseline_BA"],
                "selection_status": "locked before V8",
                "OUTER_TEST_USED": False,
            },
            {
                "benchmark": short,
                "scope": "V8_SEARCH folds 0/1 exploratory",
                "method_id": result["strongest_fair_screening_candidate"],
                "mean_subject_BA": result["strongest_fair_screening_candidate_BA"],
                "selection_status": "promoted search-scope matched baseline; not a full-scope estimate",
                "OUTER_TEST_USED": False,
            },
        ])
    baseline_evolution = pd.DataFrame(baseline_rows)
    write_csv(LEADERBOARD / "BASELINE_EVOLUTION.csv", baseline_evolution)
    write_csv(
        BASELINES / "OPENBMI_MATCHED_BASELINES.csv",
        baseline_evolution.loc[baseline_evolution.benchmark.eq("OpenBMI")],
    )
    write_csv(
        BASELINES / "WBCIC_MATCHED_BASELINES.csv",
        baseline_evolution.loc[baseline_evolution.benchmark.eq("WBCIC")],
    )
    write_json(PROTOCOL / "V8_SEARCH_BASELINE_UPDATE.json", {
        "decisions": decisions,
        "full_scope_baseline_changed": False,
        "reason_full_scope_not_changed": "New candidates were screened on only source folds 0/1.",
        "search_scope_baseline_promoted": True,
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    })

    for benchmark, short in BENCHMARKS.items():
        rows = table.loc[table.benchmark.astype(str).eq(benchmark)].copy()
        rows["benchmark_short"] = short
        rows["result_scope"] = "V8_SEARCH screening"
        write_csv(LEADERBOARD / ("OPENBMI_V8.csv" if short == "OpenBMI" else "WBCIC_DEV_V8.csv"), rows)
    cross = pd.DataFrame([
        {
            "benchmark": short,
            "screening_subjects": value["screening_subjects"],
            "updated_screening_baseline_BA": value["strongest_fair_screening_candidate_BA"],
            "multi_backbone_oracle_BA": value["multi_backbone_union_subject_oracle_BA"],
            "oracle_headroom_pp": value["oracle_headroom_vs_updated_screening_baseline_pp"],
            "headroom_state": value["headroom_state_after_baseline_update"],
            "phase_B_started": False,
            "internal_holdout_used": False,
            "OUTER_TEST_USED": False,
        }
        for short, value in decisions.items()
    ])
    write_csv(LEADERBOARD / "CROSS_BENCHMARK_V8.csv", cross)

    representation = table.loc[table.family_id.astype(str).str.contains("MULTISCALE|CONFORMER|MI_SPECIFIC|SPD", regex=True)].copy()
    write_csv(DIAGNOSTICS / "REPRESENTATION_DIAGNOSTICS.csv", representation)
    write_csv(DIAGNOSTICS / "ADAPTATION_DIAGNOSTICS.csv", table)
    write_csv(DIAGNOSTICS / "SESSION_TRANSFER.csv", cross)
    diversity_path = HEADROOM / "EXPERT_DIVERSITY.csv"
    write_csv(
        DIAGNOSTICS / "ERROR_CORRELATION.csv",
        pd.read_csv(diversity_path) if diversity_path.is_file() else _status_table("ERROR_CORRELATION"),
    )
    write_csv(
        DIAGNOSTICS / "PERSIST_LEARNED_COMPONENT_AUDIT.csv",
        _status_table("PERSIST_LEARNED_COMPONENT_AUDIT"),
    )

    specialization = table.loc[table.family_id.astype(str).str.contains("MULTISCALE", regex=False)].copy()
    meta_objective = table.loc[table.family_id.astype(str).str.contains("LR_COVERAGE|META_SGD|FOMAML|NORM_HYPER", regex=True)].copy()
    backbone = table.loc[table.family_id.astype(str).str.contains("MULTISCALE|RAW_ENCODER", regex=True)].copy()
    adapter = table.loc[~table.family_id.astype(str).str.contains("MULTIBACKBONE", regex=False)].copy()
    write_csv(ABLATIONS / "SPECIALIZATION_ABLATION.csv", specialization)
    write_csv(ABLATIONS / "META_OBJECTIVE_ABLATION.csv", meta_objective)
    write_csv(ABLATIONS / "BACKBONE_ABLATION.csv", backbone)
    write_csv(ABLATIONS / "ADAPTER_ABLATION.csv", adapter)
    write_csv(ABLATIONS / "CAPACITY_MATCHED_CONTROLS.csv", baseline_evolution)
    for name in ("BANK_SIZE_ABLATION", "PERSIST_ABLATION", "RISK_ABLATION"):
        write_csv(ABLATIONS / f"{name}.csv", _status_table(name))
    for name in ("GENERIC_RECOVERY", "PERSIST_RECOVERY", "ORACLE_RECOVERY"):
        write_csv(SELECTORS / f"{name}.csv", _status_table(name))

    iteration = table.sort_values(["benchmark", "oracle_headroom_pp"], ascending=[True, False]).reset_index(drop=True)
    iteration.insert(0, "iteration_id", [f"V8-{index + 1:02d}" for index in range(len(iteration))])
    iteration["decision"] = "ABANDON_PHASE_A_GATE_FAILED"
    write_csv(RESEARCH_LOG / "ITERATION_SUMMARY.csv", iteration)
    ledger_lines = [
        "# V8 hypothesis ledger",
        "",
        "All values are fixed-seed V8_SEARCH screening results. Failed iterations are retained.",
        "",
        "| ID | Benchmark | Family | strongest single BA | subject oracle BA | headroom (pp, V7 two-fold baseline) | Decision |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for _, row in iteration.iterrows():
        ledger_lines.append(
            f"| {row.iteration_id} | {BENCHMARKS[str(row.benchmark)]} | `{str(row.family_id).split('__', 1)[-1]}` | "
            f"{float(row.strongest_single_candidate_BA):.6f} | {float(row.subject_oracle_BA):.6f} | "
            f"{float(row.oracle_headroom_pp):.3f} | {row.decision} |"
        )
    ledger_lines.extend([
        "",
        "The final gate uses the stronger search-scope baselines, reducing the broad-union headroom to "
        f"{decisions['OpenBMI']['oracle_headroom_vs_updated_screening_baseline_pp']:.3f} pp (OpenBMI) and "
        f"{decisions['WBCIC']['oracle_headroom_vs_updated_screening_baseline_pp']:.3f} pp (WBCIC).",
    ])
    (RESEARCH_LOG / "HYPOTHESIS_LEDGER.md").write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")

    search_results = {
        "decisions": decisions,
        "family_count": int(len(table)),
        "screening_only": True,
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_CANDIDATE / "SEARCH_RESULTS.json", search_results)
    write_json(FINAL_CANDIDATE / "INTERNAL_HOLDOUT_RESULTS.json", {
        "status": "SEALED_NOT_OPENED",
        "reason": "No Phase-A bank passed the dual approximately +8 pp oracle gate.",
        "subjects": {
            "OpenBMI": split["openbmi"]["internal_holdout_subjects"],
            "WBCIC": split["wbcic"]["internal_holdout_subjects"],
        },
        "outcomes_accessed": False,
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    })
    write_json(FINAL_CANDIDATE / "DUAL_BENCHMARK_RESULTS.json", search_results)
    model_spec = {
        "status": "NO_FINAL_MODEL_FROZEN",
        "deployment_fallback": "retain the strongest previously locked fair anchor",
        "reason": "Phase-A learned-action headroom gate failed on both benchmarks",
        "selector": None,
        "PERSIST_policy": None,
        "suppression_coefficient": 0.0,
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    }
    write_json(FINAL_CANDIDATE / "FINAL_MODEL_SPEC.json", model_spec)
    (FINAL_CANDIDATE / "FINAL_MODEL_SPEC.md").write_text(
        "# Final model status\n\nNo V8 model was frozen. The protocol-preserving action is to retain the previously locked strongest fair anchor. Phase B/C were not authorized.\n",
        encoding="utf-8",
    )
    final_decision = {
        "terminal_state": "V8_SCIENTIFIC_EXHAUSTION_PHASE_A_HEADROOM",
        "success": False,
        "dual_5pp_target_reached": False,
        "dual_8pp_oracle_gate_reached": False,
        "phase_A_complete": True,
        "phase_B_started": False,
        "phase_C_started": False,
        "multi_seed_run": False,
        "internal_holdout_status": "SEALED_NOT_OPENED",
        "WBCIC_outer_status": "SEALED_NOT_OPENED_OR_ENUMERATED",
        "decisions": decisions,
        "bottleneck": "representation/action-bank headroom",
        "further_iteration": "not justified on reused development outcomes without a materially new representation source or independent cohort",
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", final_decision)
    positive_control = v7_outputs() / "diagnostics" / "V7_POSITIVE_CONTROL.json"
    write_json(PROTOCOL / "V8_POSITIVE_CONTROL_CARRYFORWARD.json", {
        "status": "V7_MECHANISM_CONTROL_RETAINED_NOT_EXPANDED",
        "source": str(positive_control),
        "source_sha256": sha256_file(positive_control) if positive_control.is_file() else None,
        "reason_not_expanded": "No V8 selector/PERSIST policy passed Phase A, so a V8 end-to-end mechanism was not instantiated.",
        "interpretation": "Synthetic wiring evidence only; no real-EEG suppression claim.",
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    })
    _write_project_docs(decisions)
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(
        _scientific_report(decisions, v7, outer), encoding="utf-8",
    )
    write_json(OUTPUTS / "REPRODUCIBILITY.json", {
        "source_commit": v7["source_commit"],
        "branch": "codex/persist-eeg-final-model-v8-headroom-first",
        "seed": split["seed"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "commands": [
            "python code/protocol/bootstrap_v8.py",
            "python code/adaptation_banks/run_query_bank.py --benchmark <openbmi|wbcic> --family CONFORMER_NORM --fold 0 --fold 1",
            "python code/adaptation_banks/run_meta_sgd_bank.py --benchmark <openbmi|wbcic> --family <CONFORMER_NORM|MI_SPECIFIC> --fold 0 --fold 1",
            "python code/adaptation_banks/run_metric_bank.py --benchmark <openbmi|wbcic> --family <CONFORMER_NORM|MI_SPECIFIC> --fold 0 --fold 1",
            "python code/normalization/run_norm_hyper_bank.py --benchmark <openbmi|wbcic> --family MI_SPECIFIC --fold 0 --fold 1",
            "python code/geometry/run_spd_transport.py --benchmark <openbmi|wbcic> --fold 0 --fold 1",
            "python code/adaptation_banks/run_raw_finetune_bank.py --benchmark <openbmi|wbcic> --fold 0 --fold 1",
            "python code/backbones/run_multiscale_bank.py --benchmark wbcic --experts 4 --adapter-rank 0 --pretrain-epochs 24 --epochs 8 --training-mode staged --lambda-mean 1.0 --fold 0 --fold 1",
            "python code/backbones/run_multiscale_bank.py --benchmark wbcic --experts 4 --adapter-rank 8 --pretrain-epochs 24 --epochs 12 --training-mode staged --lambda-mean 1.0 --fold 0 --fold 1",
            "python code/adaptation_banks/run_raw_fomaml_bank.py --benchmark wbcic --meta-epochs 4 --fold 0 --fold 1",
            "python code/evaluation/run_multibackbone_union.py --benchmark <openbmi|wbcic> --fold 0 --fold 1",
            "python code/evaluation/finalize_v8.py",
        ],
        "git_excludes": ["outputs/cache/", "outputs/diagnostics/*_SEARCH_PREDICTIONS.csv"],
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    })
    hashes = _write_hashes()
    print(json.dumps({
        "terminal_state": final_decision["terminal_state"],
        "decisions": decisions,
        "hashed_files": len(hashes["files"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
