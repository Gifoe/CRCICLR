"""Independent structural validation for the completed closure artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
RUNTIME = EXP / "runtime" / "matched_runs"
METHODS = [
    "Matched-TaskOnly",
    "Random-Aux",
    "Identity-Aux",
    "Full-Teacher-KD-Aux",
    "P-only-Aux",
    "PUD-Aux",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    per = pd.read_csv(RESULTS / "matched_aux_per_subject.csv", dtype={"subject_id": str})
    stats = read_json(RESULTS / "matched_aux_statistics.json")
    integrity = read_json(RESULTS / "matched_aux_integrity.json")

    checks: dict[str, object] = {}
    checks["rows_720"] = len(per) == 720
    checks["six_methods_120_rows_each"] = per.method.value_counts().to_dict() == {
        method: 120 for method in METHODS
    }
    checks["no_duplicate_method_fold_seed_subject"] = not per.duplicated(
        ["method", "fold", "seed", "subject_id"]
    ).any()
    checks["forty_subjects"] = per.subject_id.nunique() == 40
    checks["three_seeds_per_method_subject"] = bool(
        (per.groupby(["method", "subject_id"]).seed.nunique() == 3).all()
    )
    checks["one_outer_fold_per_subject"] = bool(
        (per.groupby("subject_id").fold.nunique() == 1).all()
    )
    checks["eight_subjects_per_method_fold"] = bool(
        (per.groupby(["method", "fold"]).subject_id.nunique() == 8).all()
    )

    inner_sha_match = True
    outer_sha_match = True
    nested_subject_guard = True
    run_markers_pass = True
    restricted_flags_false = True
    for fold in range(5):
        for seed in range(3):
            run = RUNTIME / f"fold-{fold}" / f"seed-{seed}"
            marker = read_json(run / "RUN_COMPLETE.json")
            run_markers_pass &= marker.get("pass") is True
            restricted_flags_false &= not marker.get("internal_holdout_accessed", False)
            restricted_flags_false &= not marker.get("WBCIC_outer_accessed", False)

            selection = read_json(run / "SELECTION_FROZEN.json")
            selected = list(selection["selected"].values())
            inner_sha_match &= len({item["initial_full_state_sha256"] for item in selected}) == 1
            inner_sha_match &= len({item["initial_main_state_sha256"] for item in selected}) == 1
            inner_sha_match &= len({item["loader_seed"] for item in selected}) == 1
            train_subjects = set(map(str, selection["inner_train_subjects"]))
            validation_subjects = set(map(str, selection["inner_validation_subjects"]))
            nested_subject_guard &= train_subjects.isdisjoint(validation_subjects)
            nested_subject_guard &= all(
                set(map(str, item["nested_target_subjects"])) == train_subjects
                and item.get("outcome_subjects_used") is False
                for item in selected
            )
            nested_subject_guard &= selection.get("outcome_session_2_labels_used") is False
            nested_subject_guard &= selection.get(
                "selection_frozen_before_outer_teacher_or_outcome_evaluation"
            ) is True

            final_metadata = [read_json(path) for path in sorted((run / "final").glob("*.json"))]
            require(len(final_metadata) == 6, f"expected six final metadata files in {run}")
            outer_sha_match &= len({item["initial_full_state_sha256"] for item in final_metadata}) == 1
            outer_sha_match &= len({item["initial_main_state_sha256"] for item in final_metadata}) == 1
            outer_sha_match &= len({item["loader_seed"] for item in final_metadata}) == 1
            outer_sha_match &= len({item["initialization_seed"] for item in final_metadata}) == 1
            nested_subject_guard &= all(
                item.get("outcome_labels_used_for_selection") is False
                and item.get("inference_uses_aux_head") is False
                for item in final_metadata
            )

    checks["all_15_run_markers_pass"] = run_markers_pass
    checks["inner_initialization_and_order_match"] = inner_sha_match
    checks["outer_initialization_and_order_match"] = outer_sha_match
    checks["strict_nested_subject_and_outcome_guard"] = nested_subject_guard
    checks["restricted_flags_false"] = restricted_flags_false

    frozen_hashes = pd.read_csv(RESULTS / "phase_a_frozen_hash_audit.csv")
    checks["phase_a_frozen_hashes_unchanged"] = bool(
        frozen_hashes.unchanged.astype(str).str.lower().eq("true").all()
        and frozen_hashes.sha256_before.eq(frozen_hashes.sha256_after).all()
    )

    fold = pd.read_csv(RESULTS / "matched_aux_per_fold.csv").pivot(
        index="fold", columns="method", values="BA"
    )
    seed = pd.read_csv(RESULTS / "matched_aux_per_seed.csv").pivot(
        index="seed", columns="method", values="BA"
    )
    checks["reported_positive_folds_match"] = int(
        (fold["PUD-Aux"] > fold["Matched-TaskOnly"]).sum()
    ) == stats["primary"]["positive_folds"]
    checks["reported_positive_seeds_match"] = int(
        (seed["PUD-Aux"] > seed["Matched-TaskOnly"]).sum()
    ) == stats["primary"]["positive_seeds"]
    checks["terminal_matches_prefrozen_gates"] = (
        stats["terminal"] == "PUD_AUX_MATCHED_NOT_SUPPORTED"
        and not stats["matched_success"]
        and not all(stats["gates"].values())
    )
    checks["aggregate_integrity_pass"] = integrity.get("pass") is True
    checks["aggregate_restricted_access_false"] = (
        integrity.get("internal_holdout_accessed") is False
        and integrity.get("WBCIC_outer_accessed") is False
    )

    required = [
        "FINAL_CLOSURE_REPAIRED.md",
        "PHASE_A_REPAIRED_FINAL.md",
        "PHASE_B_MATCHED_FINAL.md",
        "HOLDOUT_PURITY_AUDIT.md",
        "results/closure_repair_summary.json",
        "results/matched_aux_statistics.json",
        "results/matched_aux_per_subject.csv",
        "figures/matched_aux_main.png",
        "figures/matched_aux_main.pdf",
        "figures/matched_aux_subject_delta.png",
        "figures/matched_aux_subject_delta.pdf",
    ]
    checks["required_final_artifacts_exist"] = all((EXP / relative).is_file() for relative in required)

    for name, passed in checks.items():
        require(bool(passed), f"final validation failed: {name}")
    output = {
        "pass": True,
        "checks": checks,
        "rows": len(per),
        "subjects": per.subject_id.nunique(),
        "methods": per.method.value_counts().to_dict(),
        "terminal": stats["terminal"],
        "OpenBMI_internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    path = RESULTS / "final_validation.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
