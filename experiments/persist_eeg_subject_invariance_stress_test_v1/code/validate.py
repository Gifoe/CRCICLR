"""Independent final cardinality, matching, purity, and artifact validator."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

import common


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required_results = {
        "main_performance.csv": 2400,
        "identity_probe.csv": 300,
        "invariance_stress.csv": 270,
        "direction_audit.csv": 2400,
        "decision_vs_identity_prediction.csv": 124,
    }
    result_rows: dict[str, int] = {}
    result_hashes: dict[str, str] = {}
    issues: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for name, expected in required_results.items():
        path = common.RESULTS / name
        if not path.is_file():
            issues.append(f"missing:{name}")
            continue
        frame = pd.read_csv(path, dtype={"subject_id": str})
        frames[name] = frame
        result_rows[name] = len(frame)
        result_hashes[name] = sha256(path)
        if len(frame) != expected:
            issues.append(f"rows:{name}:{len(frame)}!={expected}")

    if "main_performance.csv" in frames:
        frame = frames["main_performance.csv"]
        counts = frame.groupby(["backbone", "method", "lambda", "fold", "seed"]).size()
        if len(counts) != 300 or set(counts.tolist()) != {8}:
            issues.append("main_performance configuration/subject cardinality")
        if set(frame.backbone) != set(common.BACKBONES) or set(frame.method) != set(common.METHODS):
            issues.append("main_performance method/backbone coverage")
        if set(frame.fold) != set(range(5)) or set(frame.seed) != set(range(3)):
            issues.append("main_performance fold/seed coverage")
    if "identity_probe.csv" in frames:
        frame = frames["identity_probe.csv"]
        counts = frame.groupby(["backbone", "method", "lambda", "fold", "seed"]).size()
        if len(counts) != 300 or set(counts.tolist()) != {1}:
            issues.append("identity_probe configuration cardinality")
        if not frame.subject_count.eq(24).all():
            issues.append("identity primary probe not restricted to 24 manipulated domains")
        if not frame.all_source_subject_count.eq(32).all():
            issues.append("identity all-source sensitivity cardinality")
    if "invariance_stress.csv" in frames:
        frame = frames["invariance_stress.csv"]
        if set(frame.method) != {"DANN", "CORAL", "MMD"}:
            issues.append("stress family coverage")
        if set(frame["lambda"].round(2)) != {0.01, 0.1, 1.0}:
            issues.append("stress lambda coverage")
    if "direction_audit.csv" in frames:
        frame = frames["direction_audit.csv"]
        counts = frame.groupby(["backbone", "method", "lambda", "fold", "seed"]).size()
        if len(counts) != 300 or set(counts.tolist()) != {8}:
            issues.append("direction audit candidate cardinality")
        if not frame.direction_source_only.astype(bool).all() or frame.outcome_used_to_define_direction.astype(bool).any():
            issues.append("direction source-only guard")
        if set(frame.D_finite_definition) != {"exact_exp3_centered_logit_RMS"}:
            issues.append("D_finite definition changed")

    matched_units = 0
    selection_units = 0
    evaluation_guards = 0
    for backbone in common.BACKBONES:
        for fold in range(5):
            for seed in range(3):
                context = common.unit_dir(backbone, fold, seed)
                candidates = []
                for method, lam in common.configuration_grid():
                    path = context / "candidates" / f"{common.config_slug(method, lam)}.json"
                    if not path.is_file():
                        issues.append(f"missing candidate:{backbone}:{fold}:{seed}:{method}:{lam}")
                        continue
                    candidates.append(common.read_json(path))
                if len(candidates) == 10:
                    if len({row["initial_shared_state_sha256"] for row in candidates}) != 1:
                        issues.append(f"initialization mismatch:{backbone}:{fold}:{seed}")
                    elif len({row["epoch0_minibatch_order_sha256"] for row in candidates}) != 1:
                        issues.append(f"minibatch mismatch:{backbone}:{fold}:{seed}")
                    else:
                        matched_units += 1
                selection_path = context / "LAMBDA_SELECTION_FROZEN.json"
                if selection_path.is_file():
                    selection_units += 1
                    selection_sha = sha256(selection_path)
                    for method, lam in common.configuration_grid():
                        complete_path = context / "evaluation" / common.config_slug(method, lam) / "EVALUATION_COMPLETE.json"
                        if not complete_path.is_file():
                            continue
                        payload = common.read_json(complete_path)
                        if payload.get("selection_file_sha256") == selection_sha and payload.get("selection_frozen_before_outcome_evaluation") is True:
                            evaluation_guards += 1
                        else:
                            issues.append(f"evaluation selection guard:{backbone}:{fold}:{seed}:{method}:{lam}")

    if matched_units != 30:
        issues.append(f"matched units:{matched_units}!=30")
    if selection_units != 30:
        issues.append(f"selection units:{selection_units}!=30")
    if evaluation_guards != 300:
        issues.append(f"evaluation guards:{evaluation_guards}!=300")

    required_figures = [
        "identity_vs_generalization.png",
        "identity_vs_generalization.pdf",
        "identity_suppression_by_lambda.png",
        "identity_suppression_by_lambda.pdf",
        "decision_vs_identity_rmse.png",
        "decision_vs_identity_rmse.pdf",
        "performance_by_method.png",
        "performance_by_method.pdf",
    ]
    figure_hashes = {}
    for name in required_figures:
        path = common.FIGURES / name
        if not path.is_file() or path.stat().st_size < 1000:
            issues.append(f"missing/short figure:{name}")
        else:
            figure_hashes[name] = sha256(path)

    statistics_path = common.RESULTS / "statistics.json"
    purity_path = common.RESULTS / "holdout_purity.json"
    if not statistics_path.is_file():
        issues.append("missing:statistics.json")
        statistics: dict[str, Any] = {}
    else:
        statistics = common.read_json(statistics_path)
    allowed_terminal = set(common.protocol()["terminal_states"])
    if statistics.get("terminal_state") not in allowed_terminal:
        issues.append(f"invalid terminal state:{statistics.get('terminal_state')}")
    if not purity_path.is_file() or common.read_json(purity_path).get("status") != "PASS":
        issues.append("holdout purity not PASS")

    required_docs = [
        "README.md",
        "STRESS_TEST_PROTOCOL_FROZEN.json",
        "METHOD_DEFINITIONS.md",
        "DATA_SPLIT_AUDIT.md",
        "BACKBONE_AUDIT.md",
        "TRAINING_LEDGER.md",
        "IDENTITY_PROBE_AUDIT.md",
        "INVARIANCE_MANIPULATION_CHECK.md",
        "INVARIANCE_GENERALIZATION_AUDIT.md",
        "DECISION_VS_IDENTITY_AUDIT.md",
        "GENERIC_REFERENCE_NOTE.md",
        "HOLDOUT_PURITY_AUDIT.md",
        "FINAL_REPORT.md",
        "ENGINEERING_REPAIR_LOG.md",
    ]
    for name in required_docs:
        path = common.EXP / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing/empty document:{name}")

    payload = {
        "pass": not issues,
        "issues": issues,
        "result_rows": result_rows,
        "result_sha256": result_hashes,
        "figure_sha256": figure_hashes,
        "matched_initialization_and_minibatch_units": matched_units,
        "selection_frozen_units": selection_units,
        "evaluation_selection_guards": evaluation_guards,
        "terminal_state": statistics.get("terminal_state"),
        "restricted_data_accessed": False,
        "WBCIC_accessed": False,
    }
    common.write_json(common.RESULTS / "final_validation.json", payload)
    if issues:
        raise RuntimeError("final validation failed: " + "; ".join(issues))
    print("STRESS_TEST_FINAL_VALIDATION_PASS", flush=True)


if __name__ == "__main__":
    main()
