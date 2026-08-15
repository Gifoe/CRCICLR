"""Strict, read-only consistency audit for materialised P5.1/P6 outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ALPHA_GRID = {0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0}


def false_outer_values(value: Any, source: str, path: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "outer_test_used" and item is not False:
                failures.append(f"{source}:{here}={item!r}")
            failures.extend(false_outer_values(item, source, here))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(false_outer_values(item, source, f"{path}[{index}]"))
    return failures


def main() -> None:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    failures: list[str] = []

    json_paths = sorted(OUT.rglob("*.json"))
    json_payloads: dict[Path, Any] = {}
    outer_failures: list[str] = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - emits exact corrupt path
            failures.append(f"invalid JSON {path}: {exc}")
            continue
        json_payloads[path] = payload
        outer_failures.extend(false_outer_values(payload, str(path.relative_to(OUT))))
    checks["all_json_parse"] = len(json_payloads) == len(json_paths)
    checks["all_json_outer_test_flags_false"] = not outer_failures
    details["json_files"] = len(json_paths)
    details["outer_flag_failures"] = outer_failures

    csv_outer_failures: list[str] = []
    for path in sorted(OUT.rglob("*.csv")):
        frame = pd.read_csv(path)
        if "outer_test_used" in frame.columns:
            values = frame["outer_test_used"].astype(str).str.lower().unique().tolist()
            if any(value != "false" for value in values):
                csv_outer_failures.append(f"{path.relative_to(OUT)}={values}")
    checks["all_csv_outer_test_flags_false"] = not csv_outer_failures
    details["csv_outer_flag_failures"] = csv_outer_failures

    candidates = pd.read_csv(OUT / "P5_1_HPARAM_CANDIDATES.csv")
    inner = pd.read_csv(OUT / "P5_1_INNER_CV_RESULTS.csv")
    selected = pd.read_csv(OUT / "P5_1_SELECTED_CONFIGS.csv")
    checks["candidate_summary_count_216"] = len(candidates) == 216
    checks["inner_result_count_1080"] = len(inner) == 1080
    checks["selected_config_count_18"] = len(selected) == 18
    run_candidate_counts = candidates.groupby(["version", "fold", "seed"]).size()
    stage_counts = candidates.groupby(["version", "fold", "seed", "stage"]).size()
    inner_counts = inner.groupby(["version", "fold", "seed", "stage", "candidate"]).size()
    checks["twelve_candidates_per_run"] = bool((run_candidate_counts == 12).all() and len(run_candidate_counts) == 18)
    checks["stage1_4_stage2_8_per_run"] = bool(
        (stage_counts.xs("stage1", level="stage") == 4).all()
        and (stage_counts.xs("stage2", level="stage") == 8).all()
    )
    checks["five_inner_folds_per_candidate"] = bool((inner_counts == 5).all() and len(inner_counts) == 216)
    checks["selected_configs_are_stage2"] = bool((selected.stage == "stage2").all())
    checks["authorized_hyperparameter_grid_only"] = bool(
        set(np.round(candidates.lambda_geometry.astype(float), 8)) <= {0.03, 0.10, 0.30, 1.0}
        and set(np.round(candidates.lambda_drift.astype(float), 8)) <= {0.01, 0.10}
        and set(np.round(candidates.learning_rate.astype(float), 8)) <= {0.0001, 0.0003}
        and set(candidates.bottleneck.astype(int)) == {8}
    )

    split_fingerprints: dict[tuple[int, int], list[list[str]]] = {}
    split_ok = True
    for version in ("V0", "V1", "V2"):
        for fold in range(3):
            for seed in range(2):
                path = OUT / version / f"fold-{fold}" / f"seed-{seed}" / "INNER_SUBJECT_FOLDS.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                folds = [[str(x) for x in group] for group in payload["folds"]]
                flattened = [x for group in folds for x in group]
                split_ok &= len(folds) == 5 and all(folds) and len(flattened) == len(set(flattened))
                key = (fold, seed)
                if key in split_fingerprints:
                    split_ok &= folds == split_fingerprints[key]
                else:
                    split_fingerprints[key] = folds
    checks["inner_splits_subject_disjoint_and_shared_across_versions"] = bool(split_ok)

    matched_epochs = True
    outer_complete = 0
    run_deltas: dict[str, list[float]] = {"V0": [], "V1": [], "V2": []}
    for row in selected.itertuples(index=False):
        run_dir = OUT / row.version / f"fold-{int(row.fold)}" / f"seed-{int(row.seed)}"
        result = json.loads((run_dir / "RUN_RESULT.json").read_text(encoding="utf-8"))
        matched_epochs &= (
            int(result["method_epochs"]) == int(result["control_epochs"])
            and int(result["method_epochs"]) == int(row.median_pair_epoch) + 1
        )
        run_deltas[row.version].append(float(result["delta_BA"]))
        outer_complete += int((run_dir / "OUTER_WORKER_COMPLETE.json").exists())
    checks["matched_method_control_epoch_duration"] = bool(matched_epochs)
    checks["outer_retrain_complete_18"] = outer_complete == 18
    details["p5_run_delta_BA"] = run_deltas

    final = json.loads((OUT / "P5_1_P6_FINAL_REPORT.json").read_text(encoding="utf-8"))
    checks["final_outer_test_false"] = final.get("outer_test_used") is False
    checks["p5_not_viable"] = final["p5"]["status"] == "PERSIST_ICG_REPRESENTATION_ONLY"
    checks["v3_explicitly_not_authorized"] = final["p5"]["v3"].get("authorized") is False
    checks["terminal_label_matches_failure"] = final["status"] == "PERSISTENCE_GEOMETRY_HAS_NO_DECISION_FUSION_HEADROOM"

    p6 = pd.read_csv(OUT / "P6" / "P6_READOUT_RESULTS.csv")
    checks["p6_run_count_6"] = len(p6) == 6
    random_counts: list[int] = []
    alpha_values: list[float] = []
    control_names: Counter[str] = Counter()
    base_reproduces_control = True
    for fold in range(3):
        for seed in range(2):
            run_dir = OUT / "P6" / f"fold-{fold}" / f"seed-{seed}"
            random_counts.append(len(pd.read_csv(run_dir / "RANDOM_100_DRAWS.csv")))
            result = json.loads((run_dir / "RUN_RESULT.json").read_text(encoding="utf-8"))
            p5_control = json.loads(
                (OUT / "V2" / f"fold-{fold}" / f"seed-{seed}" / "RUN_RESULT.json").read_text(encoding="utf-8")
            )["control_strict_inductive_BA"]
            base_reproduces_control &= abs(float(result["base_BA"]) - float(p5_control)) <= 1e-12
            for key in ("protected", "uniform", "shuffled", "all_persistence", "full_canonical"):
                alpha_values.append(float(result[key]["alpha"]))
                control_names[result[key]["name"]] += 1
    checks["random_same_rank_draw_count_600"] = sum(random_counts) == 600 and set(random_counts) == {100}
    checks["p6_base_exactly_reproduces_selected_matched_control"] = bool(base_reproduces_control)
    checks["all_selected_alphas_from_fixed_grid"] = set(alpha_values) <= ALPHA_GRID
    checks["all_mandatory_readout_variants_present"] = set(control_names) == {
        "protected_intervention_weighted",
        "protected_uniform",
        "protected_shuffled_weights",
        "all_persistence",
        "full_canonical",
    } and set(control_names.values()) == {6}

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    failures.extend(failed_checks)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "details": details,
        "failures": failures,
        "outer_test_used": False,
    }
    path = OUT / "REPRODUCIBILITY_AUDIT.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failed_checks": failed_checks, "checks": len(checks)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
