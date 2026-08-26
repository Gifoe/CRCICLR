"""Independent fail-closed validation of the completed Phase-3 evidence package."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
FIGURES = EXP / "figures"


def require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RuntimeError(f"{name} missing columns: {sorted(missing)}")


def main() -> None:
    performance = pd.read_csv(RESULTS / "main_performance.csv")
    identity = pd.read_csv(RESULTS / "identity_probe.csv")
    stress = pd.read_csv(RESULTS / "invariance_stress.csv")
    directions = pd.read_csv(RESULTS / "persistent_direction_audit.csv")
    blocks = pd.read_csv(RESULTS / "persistent_block_controls.csv")
    prediction = pd.read_csv(RESULTS / "decision_vs_identity_prediction.csv")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    purity = json.loads((RESULTS / "holdout_purity.json").read_text(encoding="utf-8"))

    require_columns(performance, {"backbone", "method", "lambda", "fold", "seed", "subject_id", "session", "BA", "macro_f1"}, "main_performance")
    require_columns(identity, {"backbone", "method", "lambda", "fold", "seed", "identity_S1_to_S2", "identity_S2_to_S1", "identity_symmetric", "identity_accuracy", "chance_normalized_identity"}, "identity_probe")
    require_columns(stress, {"backbone", "method", "lambda", "fold", "seed", "identity_suppression_vs_ERM", "BA_delta_vs_ERM", "F1_delta_vs_ERM"}, "invariance_stress")
    require_columns(directions, {"backbone", "fold", "seed", "direction_id", "direction_rank", "persistence", "geometry_strength", "identity_score", "D_finite", "outcome_CE_effect", "outcome_BA_effect", "outcome_F1_effect"}, "persistent_direction_audit")
    require_columns(blocks, {"fold", "seed", "block", "block_rank", "persistent_BA_harm", "random_control_id", "random_BA_harm", "specific_delta"}, "persistent_block_controls")
    require_columns(prediction, {"M0", "MI", "MD", "MID", "held_run_error_M0", "held_run_error_MI", "held_run_error_MD", "held_run_error_MID"}, "decision_vs_identity_prediction")

    expected = {
        "main_performance": (len(performance), 1230),
        "identity_probe": (len(identity), 150),
        "invariance_stress": (len(stress), 135),
        "persistent_direction_audit": (len(directions), 120),
        "persistent_block_controls": (len(blocks), 3000),
        "decision_vs_identity_prediction": (len(prediction), 120),
    }
    failures = {name: values for name, values in expected.items() if values[0] != values[1]}
    if failures:
        raise RuntimeError(f"final table cardinality failures: {failures}")
    if set(performance.backbone) != {"eegnet"} or set(performance.seed.astype(int)) != {0, 1, 2} or set(performance.fold.astype(int)) != set(range(5)):
        raise RuntimeError("primary backbone/fold/seed support differs from frozen protocol")
    if set(performance.method) != {"ERM", "DANN", "CORAL", "MMD"}:
        raise RuntimeError("method grid differs from frozen protocol")
    fixed = performance.groupby("method")["lambda"].unique().to_dict()
    if set(map(float, fixed["ERM"])) != {0.0}:
        raise RuntimeError("ERM lambda mismatch")
    for method in ("DANN", "CORAL", "MMD"):
        if set(map(float, fixed[method])) != {0.01, 0.1, 1.0}:
            raise RuntimeError(f"{method} lambda grid mismatch")
    if purity.get("pass") is not True:
        raise RuntimeError("holdout purity failed")
    if any(
        purity.get(key) is not False
        for key in (
            "sealed_WBCIC_outer_accessed",
            "sealed_WBCIC_outer_enumerated",
            "OpenBMI_holdout_accessed",
            "outcome_S3_used_for_training",
            "outcome_S3_used_for_lambda_selection",
            "outcome_S3_used_for_direction_construction",
            "outcome_S3_used_for_identity_probe",
            "outcome_evaluation_before_source_freeze",
        )
    ):
        raise RuntimeError("purity audit contains a restricted-data or leakage flag")
    terminal_states = {
        "WBCIC_INDEPENDENT_REPLICATION_STRONG_SUPPORTED",
        "WBCIC_INDEPENDENT_REPLICATION_PARTIAL_SUPPORTED",
        "WBCIC_INDEPENDENT_REPLICATION_NOT_SUPPORTED",
        "WBCIC_INVARIANCE_MANIPULATION_INCONCLUSIVE",
        "WBCIC_REPRESENTATION_COMPETENCE_FAIL",
    }
    if summary.get("terminal_state") not in terminal_states:
        raise RuntimeError("missing or invalid unique primary terminal state")
    required_reports = [
        "README.md", "WBCIC_REPLICATION_PROTOCOL_FROZEN.json", "HISTORICAL_WBCIC_PROVENANCE.md",
        "DATA_SCOPE_AUDIT.md", "PREPROCESSING_AUDIT.md", "FOLD_ROLE_AUDIT.md", "TRAINING_LEDGER.md",
        "REPRESENTATION_COMPETENCE.md", "IDENTITY_PROBE_AUDIT.md", "PERSISTENCE_REPLICATION_AUDIT.md",
        "INVARIANCE_MANIPULATION_AUDIT.md", "INVARIANCE_GENERALIZATION_AUDIT.md",
        "DECISION_VS_IDENTITY_REPLICATION.md", "HOLDOUT_PURITY_AUDIT.md", "ENGINEERING_REPAIR_LOG.md",
        "FINAL_REPORT.md",
    ]
    missing_reports = [name for name in required_reports if not (EXP / name).is_file()]
    required_figures = [
        f"figure{index}_{name}.{suffix}"
        for index, name in enumerate(
            ["identity_vs_generalization", "identity_manipulation", "decision_vs_identity", "persistent_block_controls"],
            start=1,
        )
        for suffix in ("png", "pdf")
    ]
    missing_figures = [name for name in required_figures if not (FIGURES / name).is_file()]
    if missing_reports or missing_figures:
        raise RuntimeError(f"missing reports={missing_reports}, figures={missing_figures}")

    result = {
        "schema": "WBCIC_PHASE3_FINAL_VALIDATION_V1",
        "pass": True,
        "terminal_state": summary["terminal_state"],
        "valid_fold_seed_runs": 15,
        "fixed_training_configurations": 150,
        "table_rows": {name: actual for name, (actual, _) in expected.items()},
        "required_reports_present": len(required_reports),
        "required_figures_present": len(required_figures),
        "holdout_purity_pass": True,
    }
    (RESULTS / "FINAL_VALIDATION.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
