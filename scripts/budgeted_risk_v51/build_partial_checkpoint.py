from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from hsc_tta.budgeted_risk.diagnostics.decision import raw_pass
from hsc_tta.budgeted_risk.diagnostics.pipeline import markdown_table, summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    repo = Path(args.project_root) / "repo"
    result_dir = repo / "outputs/budgeted_risk_v51/results"
    delivery = repo / "delivery/budgeted_risk_v51"
    config = yaml.safe_load((repo / "configs/budgeted_risk_v51/diagnostic.yaml").read_text())
    s1 = pd.read_parquet(result_dir / "S1_RESULTS.parquet")
    residuals = pd.read_parquet(result_dir / "CALIBRATION_RESIDUALS_S1.parquet")
    summary, by_seed = summarize(s1, residuals, config)
    summary["raw_gate_pass"] = summary.apply(raw_pass, axis=1)
    summary.to_csv(result_dir / "PARTIAL_S1_SUMMARY.csv", index=False)
    by_seed.to_csv(result_dir / "PARTIAL_S1_RESULTS_BY_SEED.csv", index=False)
    temporal = summary[(summary.strategy == "temporal") & summary.requested_budget.isin([5, 10, 20, 50])].copy()
    columns = [
        "dataset", "requested_budget", "raw_spearman", "raw_mae_improvement",
        "raw_gain", "raw_gain_ci_low", "calibrated_violation_mean",
        "calibrated_gain", "calibrated_gain_ci_low", "calibrated_oracle_recovery",
        "sentinel_delta", "sentinel_transition_rate", "q_mean", "raw_gate_pass",
    ]
    state = json.loads((repo / "outputs/budgeted_risk_v51/RUN_STATE.json").read_text())
    reproduction = json.loads((result_dir / "S1_REPRODUCTION.json").read_text())
    payload = {
        "schema_version": "budgeted-risk-v51-partial-checkpoint-v1",
        "status": "USER_PAUSED_AFTER_RAW_DIAGNOSTIC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_state": state["state"],
        "s1_reproduction_passed": reproduction["passed"],
        "completed": ["input_audit", "protocol_freeze", "S1_reproduction", "raw_diagnostic"],
        "not_completed": ["S2", "S3", "S4", "outlier_analysis", "final_V51_decision"],
        "provisional_frozen_gate_read": "V51_STOP_RAW_PREDICTOR_FAILURE",
        "formal_calibration_opened": False,
        "internal_final_opened": False,
        "cap_opened": False,
        "active_acquisition_run": False,
        "full_method_entered": False,
    }
    (delivery / "V51_PARTIAL_CHECKPOINT.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report = f"""# V5.1 partial checkpoint — paused by user

Status: `USER_PAUSED_AFTER_RAW_DIAGNOSTIC`

The user requested termination while S2/S3/S4 were running. Those processes were stopped and their incomplete in-memory results were discarded. This is not a completed V5.1 decision package and must not be cited as a full S1–S4 comparison.

Completed and saved: repository/input audit, protocol freeze, ordinal ancestry audit, protected-cohort audit, exact restoration of the hash-locked S1 output, independent S1 index equivalence, raw predictor diagnostic, and subject-level S1 summaries. Current state remains `{state['state']}`.

S1 reproduction matched {reproduction['matched_rows']:,}/{reproduction['old_rows_in_scope']:,} rows. Official continuous and index mismatches are zero. The independently refitted ordinal raw prediction drifted by at most {reproduction['independent_refit_raw_max_abs_drift']:.9f}, without changing any certified index.

## Saved temporal results

{markdown_table(temporal[columns])}

The frozen raw gate fails for both datasets at every budget <=20 because raw set-size gain and its subject-bootstrap lower bound are negative. Therefore the provisional gate read is `V51_STOP_RAW_PREDICTOR_FAILURE`; S4 could have provided mechanism diagnostics but could not override the earlier raw gate. This is a provisional inference, not the missing full V5.1 comparison.

Not completed: S2 exact two-fold calibration, S3 scaled exact calibration, S4 pooled cross-fit diagnostic, calibration/evaluation LOO analyses, final figures, and the formal V5.1 verdict file.

Formal calibration was not opened. Internal final was not opened. CAP was not opened. Active acquisition was not run. The full method stage was not entered.
"""
    (delivery / "V51_PAUSED_BY_USER.md").write_text(report, encoding="utf-8")
    files = []
    for root in (delivery, repo / "outputs/budgeted_risk_v51"):
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if path.name == "PARTIAL_DELIVERY_MANIFEST.json":
                continue
            files.append({"path": str(path.relative_to(repo)), "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {"status": payload["status"], "files": files, "complete_v51_delivery": False}
    (delivery / "PARTIAL_DELIVERY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
