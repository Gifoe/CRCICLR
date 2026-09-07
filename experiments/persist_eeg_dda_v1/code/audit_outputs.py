"""Independent materialized-output audit for PERSIST-EEG DDA V1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    a = pd.read_csv(ROOT / "results" / "DDA_A_SUBJECT.csv")
    ar = pd.read_csv(ROOT / "results" / "DDA_A_RANDOM_SUBJECT.csv")
    cells = pd.read_csv(ROOT / "results" / "DDA_BLOCK_CROSSFIT.csv")
    br = pd.read_csv(ROOT / "results" / "DDA_B_RANDOM_CONTROLS.csv")
    subjects = pd.read_csv(ROOT / "results" / "DDA_BC_SUBJECT.csv")
    pred = pd.read_csv(ROOT / "results" / "DDA_C_LORO_PREDICTIONS.csv")
    null = pd.read_csv(ROOT / "results" / "DDA_C_PERMUTATION_NULL.csv")
    sensitivity = pd.read_csv(ROOT / "results" / "DDA_C_RUN_SENSITIVITY.csv")
    with (ROOT / "results" / "DDA_C_RESULT.json").open(encoding="utf-8") as handle:
        c_result = json.load(handle)
    with (ROOT / "DDA_FINAL_REPORT.json").open(encoding="utf-8") as handle:
        final = json.load(handle)

    intersections = []
    for _, group in subjects.groupby(["run", "audit_fold", "block"]):
        decision = set(group[group.role == "decision"].subject.astype(str))
        outcome = set(group[group.role == "outcome"].subject.astype(str))
        intersections.append(len(decision & outcome))

    y = pred.outcome_ce_effect.to_numpy(dtype=np.float64)
    baseline_rmse = float(np.sqrt(np.mean((y - pred.baseline_prediction.to_numpy(dtype=np.float64)) ** 2)))
    full_rmse = float(np.sqrt(np.mean((y - pred.full_prediction.to_numpy(dtype=np.float64)) ** 2)))
    improvement = float((baseline_rmse - full_rmse) / baseline_rmse)
    permutation = null.iloc[:, 0].to_numpy(dtype=np.float64)
    permutation_p = float((1 + np.sum(permutation >= improvement)) / (len(permutation) + 1))

    checks = {
        "dda_a_subject_rows_204": len(a) == 204,
        "dda_a_random_rows_20400": len(ar) == 20_400,
        "dda_a_100_draws_per_subject_run": bool((ar.groupby(["run", "inner_fold", "subject"]).draw.nunique() == 100).all()),
        "dda_bc_cells_215": len(cells) == 215,
        "dda_b_random_rows_21500": len(br) == 21_500,
        "dda_b_100_draws_per_cell": bool((br.groupby(["run", "audit_fold", "block"]).draw.nunique() == 100).all()),
        "decision_outcome_subjects_disjoint_every_cell": max(intersections, default=1) == 0,
        "loro_predictions_215": len(pred) == 215,
        "permutation_draws_5000": len(null) == 5_000,
        "run_sensitivity_units_6": len(sensitivity) == 6,
        "all_primary_numeric_values_finite": bool(
            np.isfinite(a.select_dtypes("number")).all().all()
            and np.isfinite(ar.select_dtypes("number")).all().all()
            and np.isfinite(cells.select_dtypes("number")).all().all()
            and np.isfinite(br.select_dtypes("number")).all().all()
        ),
        "baseline_rmse_reproduced": abs(baseline_rmse - float(c_result["baseline_rmse"])) < 1e-12,
        "full_rmse_reproduced": abs(full_rmse - float(c_result["full_rmse"])) < 1e-12,
        "relative_improvement_reproduced": abs(improvement - float(c_result["relative_rmse_improvement"])) < 1e-12,
        "permutation_p_reproduced": abs(permutation_p - float(c_result["permutation_p"])) < 1e-12,
        "terminal_state_exact": final.get("terminal_state") == "DDA_PARTIAL_MECHANISM_ONLY",
        "outer_test_used_false": final.get("outer_test_used") is False,
        "agdi_not_authorized": final.get("agdi_training_authorized") is False,
        "four_nonempty_figures": len(list((ROOT / "figures").glob("*.png"))) == 4 and all(
            path.stat().st_size > 10_000 for path in (ROOT / "figures").glob("*.png")
        ),
    }
    core = [
        ROOT / "DDA_FINAL_REPORT.json", ROOT / "scientific_report.md",
        ROOT / "protocol" / "DDA_PROTOCOL_LOCK.json",
        ROOT / "protocol" / "PROVENANCE_AUDIT.json",
        ROOT / "protocol" / "PROVENANCE_SCOPE_CORRECTION.json",
        ROOT / "results" / "DDA_A_RESULT.json",
        ROOT / "results" / "DDA_B_RESULT.json",
        ROOT / "results" / "DDA_C_RESULT.json",
        ROOT / "results" / "DDA_BLOCK_CROSSFIT.csv",
    ]
    payload = {
        "status": "DDA_OUTPUT_AUDIT_PASS" if all(checks.values()) else "DDA_OUTPUT_AUDIT_FAIL",
        "checks": checks,
        "recomputed": {
            "baseline_rmse": baseline_rmse, "full_rmse": full_rmse,
            "relative_rmse_improvement": improvement, "permutation_p": permutation_p,
        },
        "core_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in core},
        # Exclude this audit's own target so rerunning is idempotent.
        "materialized_file_count_before_audit": sum(
            1 for path in ROOT.rglob("*")
            if path.is_file() and path.name != "OUTPUT_INTEGRITY_AUDIT.json"
        ),
    }
    target = ROOT / "OUTPUT_INTEGRITY_AUDIT.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "DDA_OUTPUT_AUDIT_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
