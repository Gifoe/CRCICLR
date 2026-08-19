from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from build_ensemble_actions import build_action_candidates
from build_unique_trials import serialize_unique_trial_reference
from keep_oracle_control import build_keep_candidates
from reconstruct import reconstruct_v21, write_residual_action_spec
from residual_oracle import run_headroom_audit
from v3_common import (
    DIAGNOSTICS,
    EXPERIMENT_ROOT,
    FIGURES,
    NEXT_STAGE,
    OUTPUTS,
    PROTOCOL,
    RESEARCH_LOG,
    RESULTS,
    default_cache_root,
    ensure_directories,
    sha256_file,
    write_csv,
    write_json,
)


REQUIRED_DIAGNOSTICS = (
    "B6_UNIQUE_TRIAL_RESULTS.csv",
    "ACTION_ENSEMBLE_RESULTS.csv",
    "ACTION_UNIQUENESS.csv",
    "RESIDUAL_ORACLE_RESULTS.csv",
    "KEEP_ONLY_ORACLE_RESULTS.csv",
    "UNIQUE_ACTION_ORACLE_RESULTS.csv",
    "RESIDUAL_HEADROOM_BY_SUBJECT.csv",
    "RESIDUAL_HEADROOM_BY_ACTION.csv",
    "RESIDUAL_CONCENTRATION.csv",
    "HEADROOM_DECISION.json",
)
REQUIRED_RESULTS = (
    "RESIDUAL_LEARNABILITY.csv",
    "RESIDUAL_POLICY_RESULTS.csv",
    "SUBJECT_RESULTS.csv",
    "FOLD_RESULTS.csv",
    "ACTION_RESULTS.csv",
    "ORACLE_RECOVERY.csv",
)
REQUIRED_POLICY_AUDIT_RESULTS = (
    "CALIBRATION_SELECTION.csv",
    "FEATURE_IMPORTANCE.csv",
    "RESIDUAL_LEARNABILITY_PREDICTIONS.csv",
    "OOF_POLICY_PREDICTIONS.csv",
)


def _pp(value: float) -> str:
    return f"{100 * float(value):+.3f} pp"


def _write_not_authorized_results(state: str) -> None:
    row = pd.DataFrame(
        [
            {
                "status": "NOT_AUTHORIZED_BY_PHASE_7_HEADROOM_GATE",
                "headroom_state": state,
                "reason": "Phase 8-15 policy learning is prohibited unless STRUCTURAL_ACTION_RESIDUAL_EXISTS",
                "OUTER_TEST_USED": False,
            }
        ]
    )
    for name in REQUIRED_RESULTS:
        write_csv(RESULTS / name, row)
    write_csv(RESEARCH_LOG / "ITERATION_SUMMARY.csv", row)
    (RESEARCH_LOG / "ITERATION_000_NOT_AUTHORIZED.md").write_text(
        f"""# Iteration 000: not authorized

The frozen Phase 7 decision was `{state}`. No rescue/harm model, threshold,
MLP, boosting model, soft correction, or autonomous iteration was run.
Continuing the intervention search would violate the predeclared gate.

`OUTER_TEST_USED=false`
""",
        encoding="utf-8",
    )


def _finalize_stopped(
    reconstruction: dict[str, Any],
    audit: dict[str, Any],
    cache_root: Path,
) -> dict[str, Any]:
    decision = audit["decision"]
    state = decision["state"]
    _write_not_authorized_results(state)
    action = audit["action_table"]
    unique = audit["unique_table"]
    b6 = pd.read_csv(DIAGNOSTICS / "B6_UNIQUE_TRIAL_RESULTS.csv")
    b6_all = b6[b6.pool.eq("all_52_exploratory")].iloc[0]
    all_action = action[action.pool.eq("all_52_exploratory")].set_index("method_id")
    global_safe = all_action.loc["ORACLE_ACTION_PROTECTED_SAFE_GLOBAL"]
    global_full = all_action.loc["ORACLE_ACTION_FULL_GLOBAL"]
    single_safe = all_action.loc["ORACLE_ACTION_PROTECTED_SAFE_SINGLE_REPLACEMENT"]
    single_full = all_action.loc["ORACLE_ACTION_FULL_SINGLE_REPLACEMENT"]
    all_unique = unique[unique.pool.eq("all_52_exploratory")].set_index("menu_id")
    uniqueness = audit["uniqueness_table"]
    family_unique = uniqueness.groupby("action_family").unique_rescue_not_available_from_KEEP_menu.sum().sort_values(ascending=False)
    top_unique_action = str(family_unique.index[0]) if len(family_unique) else "NONE"
    erase_unique = int(family_unique.get("ERASE", 0))
    protected_meaningful = max(
        float(global_safe.mean_subject_delta_BA_vs_B6),
        float(single_safe.mean_subject_delta_BA_vs_B6),
    ) >= 0.005
    erase_necessary = (
        max(float(global_full.mean_subject_delta_BA_vs_B6), float(single_full.mean_subject_delta_BA_vs_B6))
        - max(float(global_safe.mean_subject_delta_BA_vs_B6), float(single_safe.mean_subject_delta_BA_vs_B6))
        >= 0.005
    )
    mechanism = (
        "generic model diversity above an ordinary ensemble"
        if state == "RESIDUAL_HEADROOM_IS_GENERIC_DIVERSITY"
        else "ordinary ensemble; no decision-relevant residual intervention headroom"
    )
    final = {
        "terminal_state": state,
        "phase_8_plus_executed": False,
        "v2_1_B6_reproduced": reconstruction["status"] == "V2_1_RECONSTRUCTION_PASS",
        "deployment_B6_mean_subject_BA": float(b6_all.mean_subject_BA),
        "global_protected_safe_oracle_delta_BA": float(global_safe.mean_subject_delta_BA_vs_B6),
        "global_full_oracle_delta_BA": float(global_full.mean_subject_delta_BA_vs_B6),
        "single_replacement_protected_safe_oracle_delta_BA": float(single_safe.mean_subject_delta_BA_vs_B6),
        "single_replacement_full_oracle_delta_BA": float(single_full.mean_subject_delta_BA_vs_B6),
        "keep_only_oracle_delta_BA": decision["keep_only_oracle_delta_BA_vs_B6"],
        "strongest_action_minus_keep_only_delta_BA": decision["action_oracle_minus_keep_only_delta_BA"],
        "combined_action_plus_keep_minus_keep_delta_BA": decision[
            "combined_keep_plus_action_minus_keep_only_delta_BA"
        ],
        "residual_distributed": {
            "positive_subjects": decision["positive_subjects"],
            "positive_subject_fraction": decision["positive_subject_fraction"],
            "positive_sessions": decision["positive_sessions"],
            "top20_subject_gain_concentration": decision["top20_subject_gain_concentration"],
        },
        "top_unique_action_family": top_unique_action,
        "erase_unique_rescue_candidate_count_sum": erase_unique,
        "erase_necessary_under_gate": erase_necessary,
        "protected_safe_meaningful_headroom_ge_0_5pp": protected_meaningful,
        "residual_predictability": "NOT_TESTED_NOT_AUTHORIZED",
        "features_predicting_residual": "NOT_TESTED_NOT_AUTHORIZED",
        "persist_features_incremental_value": "NOT_TESTED_NOT_AUTHORIZED",
        "prospective_method_beats_B6": False,
        "prospective_delta_BA_vs_B6": None,
        "unique_action_oracle_recovered_fraction": 0.0,
        "supported_mechanism": mechanism,
        "continue_intervention_research": False,
        "recommended_constructive_line": "B6-style strong ensemble -> compression/distillation -> one deployable student; retain PERSIST for audit, utility, decision-dependence, and safety analysis",
        "new_independent_protocol_authorized": False,
        "previous_52_subjects_confirmatory": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "FINAL_DECISION.json", final)
    report = f"""# PERSIST-EEG residual actionability V3

## Terminal state

`{state}`

All 52 subjects are historical development data. This is exploratory discovery,
not confirmatory evidence. WBCIC outer was not accessed.

## Direct answers

1. V2.1 B6 reproduced exactly: **{final['v2_1_B6_reproduced']}**.
2. Deployment-level B6 mean subject BA: **{final['deployment_B6_mean_subject_BA']:.6f}**.
3. Global protected-safe oracle above B6: **{_pp(final['global_protected_safe_oracle_delta_BA'])}**.
4. Full global oracle above B6: **{_pp(final['global_full_oracle_delta_BA'])}**.
5. Single-expert replacement oracle: protected-safe
   **{_pp(final['single_replacement_protected_safe_oracle_delta_BA'])}**;
   full **{_pp(final['single_replacement_full_oracle_delta_BA'])}**.
6. KEEP-only diversity oracle above B6: **{_pp(final['keep_only_oracle_delta_BA'])}**.
7. Strongest action oracle minus KEEP-only oracle:
   **{_pp(final['strongest_action_minus_keep_only_delta_BA'])}**. Combined
   KEEP+action minus KEEP-only: **{_pp(final['combined_action_plus_keep_minus_keep_delta_BA'])}**.
8. Residual distribution: {final['residual_distributed']['positive_subjects']}/52
   positive subjects, {final['residual_distributed']['positive_sessions']} positive
   sessions, top-20% subject concentration
   {final['residual_distributed']['top20_subject_gain_concentration']:.3f}.
9. Largest candidate-level unique-rescue family: `{top_unique_action}`.
10. ERASE necessary under the frozen gate: **{erase_necessary}**.
11. Protected-safe retains at least +0.5 pp oracle headroom:
    **{protected_meaningful}**.
12. Residual rescue predictability: **not tested; Phase 8 was not authorized**.
13. Predictive legal features: **not tested**.
14. Incremental value of P/D/protected features: **not tested**.
15. Prospective method beating B6: **none; no policy was legally trained**.
16. Prospective Delta BA and CI: **not applicable**.
17. Unique action oracle recovered: **0%**, because no prospective policy was authorized.
18. Supported mechanism: **{mechanism}**.
19. Continue intervention research: **no** under this development protocol.
20. Next constructive line: **ensemble compression/distillation**, with PERSIST
    retained as an audit and safety framework.

## Why development stopped

The Phase 7 gate was frozen before the oracle audit. It is not legitimate to
train increasingly flexible rescue models when the intervention-specific
headroom requirement fails. The result cannot be repaired by comparing against
a single EEGNet run; B6 is the required reference.

`OUTER_TEST_USED=false`
"""
    (OUTPUTS / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    (NEXT_STAGE / "ENSEMBLE_COMPRESSION_DISTILLATION_PLAN.md").write_text(
        """# Recommended constructive next line

Freeze B6 as teacher, train a single deployable student using only development
data, and evaluate the student under a genuinely new independent protocol.
Report teacher-student BA, calibration, compute, latency, and subject-level
robustness. PERSIST remains an audit/mechanistic layer, not a claimed residual
intervention gain.

WBCIC outer remains unauthorized.
""",
        encoding="utf-8",
    )
    _write_reproducibility(cache_root)
    return final


def _write_reproducibility(cache_root: Path) -> None:
    code = Path(__file__).resolve().parent
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(EXPERIMENT_ROOT.parents[1]), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_head = None
    artifacts = [path for path in OUTPUTS.rglob("*") if path.is_file() and path.name != "REPRODUCIBILITY.json"]
    payload = {
        "status": "V3_REPRODUCIBLE_ARTIFACT_SET",
        "server_execution_only": True,
        "command": "python experiments/persist_eeg_residual_actionability_v3/code/run_all.py --phase all",
        "git_head_at_execution": git_head,
        "code_sha256": {path.name: sha256_file(path) for path in sorted(code.glob("*.py"))},
        "source_cache_sha256": {path.name: sha256_file(path) for path in sorted(cache_root.glob("*.parquet"))},
        "artifact_sha256": {
            str(path.relative_to(OUTPUTS)).replace("\\", "/"): sha256_file(path)
            for path in sorted(artifacts)
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "OUTER_TEST_USED": False,
    }
    write_json(OUTPUTS / "REPRODUCIBILITY.json", payload)


def _validate_outputs(final_required: bool) -> None:
    required = [
        PROTOCOL / "V2_1_RECONSTRUCTION.md",
        PROTOCOL / "V2_1_RECONSTRUCTION.json",
        PROTOCOL / "PROVENANCE_AUDIT.json",
        PROTOCOL / "RESIDUAL_ACTION_SPEC.json",
        *[DIAGNOSTICS / name for name in REQUIRED_DIAGNOSTICS],
    ]
    if final_required:
        required.extend(
            [
                *[RESULTS / name for name in REQUIRED_RESULTS],
                RESEARCH_LOG / "ITERATION_SUMMARY.csv",
                OUTPUTS / "FINAL_DECISION.json",
                OUTPUTS / "SCIENTIFIC_REPORT.md",
                OUTPUTS / "REPRODUCIBILITY.json",
            ]
        )
        final_path = OUTPUTS / "FINAL_DECISION.json"
        if final_path.exists():
            final = json.loads(final_path.read_text(encoding="utf-8"))
            if final.get("phase_8_plus_executed", False):
                required.extend(
                    [
                        PROTOCOL / "GROUPED_NESTED_CV.json",
                        PROTOCOL / "LEGAL_FEATURE_SCHEMA.json",
                        *[RESULTS / name for name in REQUIRED_POLICY_AUDIT_RESULTS],
                    ]
                )
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty required V3 artifacts: {missing}")
    for path in list(DIAGNOSTICS.glob("*.csv")) + list(RESULTS.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            raise RuntimeError(f"Empty result table: {path}")
        outer_columns = [column for column in frame if column.lower() == "outer_test_used"]
        for column in outer_columns:
            if frame[column].astype(str).str.lower().isin(("true", "1")).any():
                raise RuntimeError(f"Outer-test flag in {path}")

    if not final_required:
        return

    def contains_outer_true(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                (str(key).lower() == "outer_test_used" and item is True)
                or contains_outer_true(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(contains_outer_true(item) for item in value)
        return False

    for path in OUTPUTS.rglob("*.json"):
        if contains_outer_true(json.loads(path.read_text(encoding="utf-8"))):
            raise RuntimeError(f"Outer-test flag in {path}")

    final = json.loads((OUTPUTS / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    if final.get("OUTER_TEST_USED") is not False or final.get("outer_test_authorized", False):
        raise RuntimeError("Final decision does not preserve the sealed outer test")
    if not final.get("phase_8_plus_executed", False):
        return

    protocol = json.loads((PROTOCOL / "GROUPED_NESTED_CV.json").read_text(encoding="utf-8"))
    heldout_seen: set[str] = set()
    for fold in protocol["folds"]:
        train = set(map(str, fold["model_training_subjects"]))
        calibration = set(map(str, fold["calibration_subjects"]))
        heldout = set(map(str, fold["heldout_subjects"]))
        if train & calibration or train & heldout or calibration & heldout:
            raise RuntimeError(f"Subject leakage in grouped fold {fold['outer_fold']}")
        if heldout_seen & heldout:
            raise RuntimeError("A held-out subject occurs in multiple outer folds")
        heldout_seen |= heldout
    if len(heldout_seen) != int(protocol["subjects"]):
        raise RuntimeError("Grouped folds do not cover every historical development subject")

    schema = json.loads((PROTOCOL / "LEGAL_FEATURE_SCHEMA.json").read_text(encoding="utf-8"))
    feature_names = schema["full_legal_features"]
    forbidden = ("label", "outcome", "correct", "rescue", "harm", "effect", "target_baseline_error")
    offenders = [name for name in feature_names if any(token in name.lower() for token in forbidden)]
    if offenders:
        raise RuntimeError(f"Outcome-dependent names in legal feature schema: {offenders}")

    oof = pd.read_csv(RESULTS / "OOF_POLICY_PREDICTIONS.csv")
    if oof.duplicated(["trial_uid", "model_id"]).any():
        raise RuntimeError("Duplicate trial/model rows in OOF predictions")
    model_counts = oof.groupby("model_id").trial_uid.nunique()
    if len(model_counts) != 6 or not (model_counts == 10_400).all() or len(oof) != 62_400:
        raise RuntimeError(f"Incomplete OOF coverage: {model_counts.to_dict()}, rows={len(oof)}")

    selection = pd.read_csv(RESULTS / "CALIBRATION_SELECTION.csv")
    if selection.heldout_subjects_read_for_selection.astype(str).str.lower().isin(("true", "1")).any():
        raise RuntimeError("Held-out subjects were read during model/threshold selection")
    selected = selection[
        selection.selected_on_inner_calibration.astype(str).str.lower().isin(("true", "1"))
    ]
    selected_counts = selected.groupby(["model_id", "outer_fold"]).size()
    if len(selected_counts) != 25 or not (selected_counts == 1).all():
        raise RuntimeError(f"Expected one inner-calibration choice per policy/fold: {selected_counts.to_dict()}")


def diagnose(cache_root: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    ensure_directories()
    spec = write_residual_action_spec()
    reconstruction, _, trials = reconstruct_v21(cache_root)
    unique_reference = serialize_unique_trial_reference(trials)
    write_csv(DIAGNOSTICS / "B6_UNIQUE_TRIAL_REFERENCE.csv", unique_reference)
    action_candidates = build_action_candidates(trials)
    keep_candidates = build_keep_candidates(trials)
    audit = run_headroom_audit(trials, action_candidates, keep_candidates, spec)
    _validate_outputs(final_required=False)
    return reconstruction, audit, trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PERSIST-EEG residual actionability V3")
    parser.add_argument("--phase", choices=("diagnose", "all"), default="all")
    parser.add_argument("--cache-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root or default_cache_root()
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    reconstruction, audit, trials = diagnose(cache_root)
    state = audit["decision"]["state"]
    print(json.dumps({"phase": "diagnose", "status": "PASS", "headroom_state": state}))
    if args.phase == "diagnose":
        return
    if state == "STRUCTURAL_ACTION_RESIDUAL_EXISTS":
        from policy import run_residual_policy_research

        final = run_residual_policy_research(trials=trials, audit=audit, cache_root=cache_root)
        _write_reproducibility(cache_root)
        _validate_outputs(final_required=True)
        print(
            json.dumps(
                {
                    "phase": "all",
                    "status": "PASS",
                    "terminal_state": final["terminal_state"],
                    "OUTER_TEST_USED": final["OUTER_TEST_USED"],
                }
            )
        )
        return
    final = _finalize_stopped(reconstruction, audit, cache_root)
    _validate_outputs(final_required=True)
    print(
        json.dumps(
            {
                "phase": "all",
                "status": "PASS",
                "terminal_state": final["terminal_state"],
                "OUTER_TEST_USED": final["OUTER_TEST_USED"],
            }
        )
    )


if __name__ == "__main__":
    main()
