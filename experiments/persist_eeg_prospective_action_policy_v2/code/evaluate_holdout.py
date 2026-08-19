from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from common import (
    EXPERIMENT_ROOT,
    FIGURES,
    FREEZE,
    HOLDOUT,
    NEXT_STAGE,
    OUTPUTS,
    PROTOCOL,
    canonical_hash,
    ensure_directories,
    markdown_table,
    sha256_file,
    write_csv,
    write_json,
)
from data import load_pool
from freeze import verify_lock
from metrics import oracle_actions, policy_metrics, policy_tables
from policies import FULL_MENU, PROTECTED_SAFE_MENU, consensus_evaluation


SENTINEL = OUTPUTS / "HOLDOUT_OPENED.sentinel"


def _holdout_plots(results: pd.DataFrame, run_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b6f9c" if value > 0 else "#b65d5d" if value < 0 else "#777777" for value in results.mean_subject_delta_BA]
    ax.bar(results.policy_id, results.mean_subject_delta_BA, color=colors)
    ax.errorbar(
        np.arange(len(results)),
        results.mean_subject_delta_BA,
        yerr=[
            results.mean_subject_delta_BA - results.bootstrap_CI95_L,
            results.bootstrap_CI95_U - results.mean_subject_delta_BA,
        ],
        fmt="none",
        color="black",
        capsize=4,
    )
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_ylabel("Development-holdout subject-balanced ΔBA")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "holdout_policy_gain.png", dpi=220)
    fig.savefig(FIGURES / "holdout_policy_gain.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    policies = list(run_results.policy_id.unique())
    width = 0.8 / len(policies)
    labels_frame = run_results[["fold_id", "seed_id"]].drop_duplicates().sort_values(["fold_id", "seed_id"])
    labels = [f"F{row.fold_id}/S{row.seed_id}" for row in labels_frame.itertuples()]
    for index, policy in enumerate(policies):
        group = run_results[run_results.policy_id.eq(policy)].sort_values(["fold_id", "seed_id"])
        ax.bar(np.arange(len(group)) + (index - (len(policies) - 1) / 2) * width, group.delta_BA, width, label=policy)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_ylabel("Run-level subject-balanced ΔBA")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "holdout_run_robustness.png", dpi=220)
    fig.savefig(FIGURES / "holdout_run_robustness.pdf")
    plt.close(fig)


def _output_hashes() -> dict[str, str]:
    files = [
        path
        for path in OUTPUTS.rglob("*")
        if path.is_file() and path.name not in ("REPRODUCIBILITY.json", "HOLDOUT_OPENED.sentinel")
    ]
    return {str(path.relative_to(EXPERIMENT_ROOT)).replace("\\", "/"): sha256_file(path) for path in sorted(files)}


def evaluate_holdout_once(cache_root: Path) -> dict[str, Any]:
    ensure_directories()
    if SENTINEL.exists():
        raise RuntimeError("Development holdout has already been opened; V2 forbids a second evaluation")
    lock = verify_lock()
    authorization_path = FREEZE / "HOLDOUT_OPEN_AUTHORIZATION.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization["remaining_openings"] != 1 or authorization["policy_lock_hash"] != lock["policy_lock_hash"]:
        raise RuntimeError("Invalid holdout authorization")

    # The sentinel is written before the label-bearing predicate scan. A crash
    # cannot silently grant a second opening.
    SENTINEL.write_text(
        json.dumps(
            {
                "status": "DEVELOPMENT_HOLDOUT_OPENED_ONCE",
                "policy_lock_hash": lock["policy_lock_hash"],
                "post_open_retuning_allowed": False,
                "OUTER_TEST_USED": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bundle = load_pool(cache_root, "DEVELOPMENT_HOLDOUT")
    frame = bundle.frame
    if frame.subject_id.nunique() != 12 or frame.pool.ne("DEVELOPMENT_HOLDOUT").any():
        raise RuntimeError("Holdout predicate returned the wrong pool")

    oracle_selected = oracle_actions(frame, FULL_MENU)
    oracle = policy_metrics(frame, oracle_selected, bootstrap_repetitions=5000, seed_offset=90)
    oracle_gain = oracle["mean_subject_delta_BA"]
    keep_selected = np.full(len(frame), "noop", dtype=object)
    policies = [
        (
            "M0_KEEP",
            keep_selected,
            policy_metrics(frame, keep_selected, oracle_gain=oracle_gain, bootstrap_repetitions=5000, seed_offset=0),
        ),
        (
            "I003_CROSS_RUN_FULL",
            None,
            None,
        ),
        (
            "I003_CROSS_RUN_PROTECTED_SAFE",
            None,
            None,
        ),
        (
            "ORACLE_FULL_MENU",
            oracle_selected,
            oracle,
        ),
    ]
    full_eval = consensus_evaluation(frame, "I003_CROSS_RUN_FULL", FULL_MENU, oracle_gain, seed_offset=3)
    safe_eval = consensus_evaluation(
        frame, "I003_CROSS_RUN_PROTECTED_SAFE", PROTECTED_SAFE_MENU, oracle_gain, seed_offset=4
    )
    policies[1] = (full_eval.policy_id, full_eval.selected, full_eval.metrics)
    policies[2] = (safe_eval.policy_id, safe_eval.selected, safe_eval.metrics)

    result_rows: list[dict[str, Any]] = []
    subject_parts: list[pd.DataFrame] = []
    run_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    for policy_id, selected, metrics in policies:
        result_rows.append({"policy_id": policy_id, **metrics})
        subject, run, action = policy_tables(frame, selected)
        for part in (subject, run, action):
            part.insert(0, "policy_id", policy_id)
        subject_parts.append(subject)
        run_parts.append(run)
        action_parts.append(action)
    results = pd.DataFrame(result_rows)
    subjects = pd.concat(subject_parts, ignore_index=True)
    runs = pd.concat(run_parts, ignore_index=True)
    actions = pd.concat(action_parts, ignore_index=True)
    write_csv(HOLDOUT / "DEVELOPMENT_HOLDOUT_POLICY_RESULTS.csv", results)
    write_csv(HOLDOUT / "DEVELOPMENT_HOLDOUT_SUBJECT_RESULTS.csv", subjects)
    write_csv(HOLDOUT / "DEVELOPMENT_HOLDOUT_RUN_RESULTS.csv", runs)
    write_csv(HOLDOUT / "DEVELOPMENT_HOLDOUT_ACTION_RESULTS.csv", actions)

    candidate_results = results[results.policy_id.isin(("I003_CROSS_RUN_FULL", "I003_CROSS_RUN_PROTECTED_SAFE"))]
    best = candidate_results.sort_values("mean_subject_delta_BA", ascending=False).iloc[0]
    holdout_success = bool(
        best.mean_subject_delta_BA > 0
        and best.bootstrap_CI95_L > 0
        and best.action_rate > 0
        and best.positive_run_fraction >= 4 / 6
    )
    terminal = (
        "DEVELOPMENT_HOLDOUT_SUCCESS_NEW_PROTOCOL_REQUIRED"
        if holdout_success
        else "DEVELOPMENT_HOLDOUT_FAILED_NO_RETUNING"
    )
    final = {
        "terminal_state": terminal,
        "best_frozen_candidate": best.to_dict(),
        "oracle": oracle,
        "holdout_success": holdout_success,
        "policy_lock_hash": lock["policy_lock_hash"],
        "holdout_subjects": int(frame.subject_id.nunique()),
        "holdout_rows": int(len(frame)),
        "holdout_openings_completed": 1,
        "post_holdout_retuning_allowed": False,
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", final)
    split = json.loads((PROTOCOL / "AUTONOMOUS_RESEARCH_SPLIT.json").read_text(encoding="utf-8"))
    audit = {
        "status": "ONE_TIME_HOLDOUT_AUDIT_PASS",
        "split_assignment_hash": split["assignment_hash"],
        "policy_lock_hash": lock["policy_lock_hash"],
        "exploration_subject_count": split["counts"]["exploration_pool"],
        "holdout_subject_count": split["counts"]["development_holdout"],
        "holdout_opened_after_lock": True,
        "holdout_openings": 1,
        "outer_subject_ids_loaded": False,
        "outer_samples_materialized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "HOLDOUT_AND_OUTER_ISOLATION_AUDIT.json", audit)

    if holdout_success:
        next_md = f"""# Next independent prospective experiment

V2 development holdout passed without retuning. A new experiment/version must
independently freeze `{best.policy_id}` with lock `{lock['policy_lock_hash']}`.
This file does not authorize WBCIC outer access. The deployment must provide
all frozen run experts needed for leave-target-run consensus before outcomes.
"""
    else:
        next_md = """# No next policy freeze

The one-time development holdout failed the frozen success gate. V2 is closed.
No retuning and no WBCIC outer evaluation are authorized.
"""
    (NEXT_STAGE / "NEXT_INDEPENDENT_EXPERIMENT.md").write_text(next_md, encoding="utf-8")

    report_table = results[
        [
            "policy_id",
            "mean_subject_delta_BA",
            "bootstrap_CI95_L",
            "bootstrap_CI95_U",
            "action_rate",
            "rescue_precision",
            "unsafe_intervention_rate",
            "positive_run_fraction",
            "recovered_oracle_headroom",
        ]
    ].copy()
    for column in report_table.columns[1:]:
        report_table[column] = report_table[column].map(lambda value: f"{value:.6f}")
    report = f"""# PERSIST-EEG prospective action policy V2

## Terminal state

`{terminal}`

The autonomous search used 40 exploration subjects. Twelve development
subjects were opened once only after candidate specifications and code hashes
were locked. WBCIC outer data were never accessed.

## Scientific result

The single-run confidence and regularized error models failed. A deterministic
leave-target-run consensus rule passed the exploration stopping gate, so search
stopped. Its development-holdout result is reported below without subsequent
method changes.

{markdown_table(report_table)}

## Interpretation

- Best frozen candidate: `{best.policy_id}`
- Development-holdout Delta BA: `{best.mean_subject_delta_BA:.6f}`
- Grouped bootstrap LCB95: `{best.bootstrap_CI95_L:.6f}`
- Rescue precision / harm: `{best.rescue_precision:.3f}` / `{best.unsafe_intervention_rate:.3f}`
- Positive-run fraction: `{best.positive_run_fraction:.3f}`
- Recovered holdout oracle headroom: `{best.recovered_oracle_headroom:.3f}`

The method is not a single-model PERSIST router. It requires multiple frozen
run experts at inference, and the full-menu variant may select ERASE. The
protected-safe result must therefore be read separately. A positive V2 result
would still be exploratory and would require a new independent protocol.

`OUTER_TEST_USED = false`
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    _holdout_plots(candidate_results, runs[runs.policy_id.isin(candidate_results.policy_id)])

    reproducibility = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "commands": [
            "python experiments/persist_eeg_prospective_action_policy_v2/code/run_all.py --phase explore",
            "python experiments/persist_eeg_prospective_action_policy_v2/code/run_all.py --phase freeze",
            "python experiments/persist_eeg_prospective_action_policy_v2/code/run_all.py --phase holdout",
        ],
        "policy_lock_hash": lock["policy_lock_hash"],
        "source_sha256": split["source_sha256"],
        "output_sha256": _output_hashes(),
        "holdout_openings": 1,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "REPRODUCIBILITY.json", reproducibility)
    return final

