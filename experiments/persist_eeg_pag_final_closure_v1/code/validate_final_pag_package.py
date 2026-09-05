from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def source(rel: str):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: dict[str, bool] = {}
    blocked = load("SEALED_CONFIRMATION_BLOCKED.json")
    audit = load("SEALED_RESOURCE_AUDIT.json")
    decision = load("FINAL_PAG_DECISION.json")
    stage0 = load("STAGE0_AUDIT.json")
    second = load("SECOND_BACKBONE_DECISION.json")
    lock = load("PREREGISTRATION_LOCK.json")
    manifest = load("FINAL_RESULT_MANIFEST.json")
    p3 = source("experiments/persist_eeg_p3closure_p4/outputs/persist_eeg_p3closure_p4/p3_closure/P3_FINAL_REPORT_V2.json")
    pud = source("experiments/persist_eeg_final_failure_localization_and_aux_v1/results/pud_aux_statistics.json")
    matched = source("experiments/persist_eeg_final_closure_repair_v1/results/matched_aux_statistics.json")

    checks["sealed_blocked_terminal"] = blocked.get("status") == "SEALED_CONFIRMATION_NOT_RUN_RESOURCE_UNAVAILABLE"
    checks["sealed_outcome_labels_false"] = blocked.get("outcome_labels_read") is False and audit.get("outcome_labels_read") is False
    checks["no_post_unblinding_repair"] = blocked.get("post_unblinding_repair") is False and audit.get("post_unblinding_repair") is False
    checks["resources_false"] = audit.get("resources_available") is False
    checks["stage0_complete_without_sealed"] = stage0.get("status") == "COMPLETE" and stage0.get("sealed_outcomes_read") is False
    checks["second_backbone_fail_closed"] = second.get("training_started") is False and second.get("decision") == "SECOND_BACKBONE_GEOMETRY_NOT_FEASIBLE_UNDER_FROZEN_PROTOCOL"
    checks["final_partial_state"] = decision.get("terminal_state") == "PAG_PARTIAL_CONFIRMATION"
    checks["ready_for_manuscript_no"] = decision.get("ready_for_iclr_manuscript") == "NO"
    checks["primary_method_frozen"] = decision.get("constructive_primary", {}).get("method") == "PUD-Aux"
    checks["meaningful_gain_gate_frozen"] = decision.get("constructive_primary", {}).get("minimum_meaningful_gain_pp") == 0.5

    try:
        checks["preregistration_anchored"] = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", lock["preregistration_commit"], "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except Exception:
        checks["preregistration_anchored"] = False

    rows = list(csv.DictReader((ROOT / "FINAL_AUTHORITATIVE_RESULTS.csv").open(encoding="utf-8")))
    def row(family: str, comparison: str):
        return next(r for r in rows if r["family"] == family and r["comparison"] == comparison)
    historical = row("PUD-Aux", "PUD-Aux minus historical Vanilla")
    matched_row = row("PUD-Aux", "PUD-Aux minus matched task-only")
    checks["historical_delta_trace"] = abs(float(historical["value"]) - pud["delta"]) < 1e-12
    checks["matched_delta_trace"] = abs(float(matched_row["value"]) - matched["primary"]["mean"]) < 1e-12
    checks["p3_delta_trace"] = abs(float(row("P3", "MI epoch0 to best")["value"]) - p3["headline"]["MI_epoch0_to_best_BA"]["mean"]) < 1e-12
    checks["sealed_rows_have_no_value"] = all(r["evidence_scope"] != "sealed" or r["value"] == "" for r in rows)
    checks["manifest_files_match"] = all(sha256(ROOT / rel) == expected for rel, expected in manifest["files_sha256"].items())

    forbidden_names = {"runtime", "checkpoint", "cache", "raw_eeg", "raw-eeg", "embeddings"}
    found_forbidden = []
    for path in ROOT.rglob("*"):
        if any(token in path.name.lower() for token in forbidden_names):
            found_forbidden.append(path.relative_to(ROOT).as_posix())
    checks["no_runtime_or_raw_artifact_in_final_dir"] = not found_forbidden

    payload = {
        "schema": "PERSIST_EEG_PAG_INDEPENDENT_VALIDATION_V1",
        "pass": all(checks.values()),
        "checks": checks,
        "forbidden_artifact_names": found_forbidden,
        "source_outcome_labels_read": False,
        "validation_scope": "compact final package and provenance only; no sealed labels/features/predictions",
    }
    (ROOT / "INDEPENDENT_VALIDATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "INDEPENDENT_VALIDATION.md").write_text(
        "# Independent validation\n\n"
        + ("`pass=true`. All compact-package, provenance, hash, and fail-closed checks passed.\n" if payload["pass"] else "`pass=false`. One or more checks failed; inspect `INDEPENDENT_VALIDATION.json`.\n")
        + "\nNo sealed labels, features, predictions, or outcomes were read.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
