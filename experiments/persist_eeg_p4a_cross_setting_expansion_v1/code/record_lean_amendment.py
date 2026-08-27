from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    snapshot_path = EXP / "P4A_GRID_PAUSE_SNAPSHOT.json"
    protocol_path = EXP / "P4A_PROTOCOL_FROZEN.json"
    snapshot = read_json(snapshot_path)
    if snapshot["progress"]["erm_completed"] != 45:
        raise RuntimeError("Lean amendment requires 45/45 mandatory ERM runs")
    if snapshot["scheduler"]["State"] != "Ready":
        raise RuntimeError("non-ERM launcher was not stopped")
    if snapshot["pause_boundary"]["current_training_process_count"] != 0:
        raise RuntimeError("training process remained active at amendment time")

    unchanged = [
        "datasets",
        "tasks",
        "backbones",
        "folds",
        "seeds",
        "ERM definition",
        "I/P/D/C_src/O_task definitions",
        "competence rule",
        "holdout policy",
    ]
    amendment = {
        "schema": "P4A_PROTOCOL_AMENDMENT_LEAN_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_type": "COMPUTATIONAL_SCOPE_AMENDMENT",
        "scientific_outcome_amendment": False,
        "original_protocol": {
            "path": str(protocol_path),
            "sha256": sha256(protocol_path),
            "required_full_invariance_grid": True,
            "planned_non_erm_grid_count": 405,
        },
        "pause_snapshot": {
            "path": str(snapshot_path),
            "sha256": sha256(snapshot_path),
            "erm_completed": snapshot["progress"]["erm_completed"],
            "non_erm_grid_completed": snapshot["progress"]["grid_completed"],
            "non_erm_grid_complete": False,
            "counts_by_setting": snapshot["progress"]["grid_counts_by_setting"],
        },
        "lean_primary_asset": "cross-setting competent-ERM source evidence cube",
        "mandatory_erm_complete_before_amendment": True,
        "amendment_before_p4b_future_direction_utility_discovery": True,
        "reason": "The non-ERM DANN/CORAL/MMD grid is computationally redundant for P4B direction-level identity-reliability discovery, whose frozen I/P/D/C_src/O_task primitives are available from competent ERM representations.",
        "pause_decision_used_grid_scientific_outcomes": False,
        "partial_grid_label": "OPTIONAL_PARTIAL_INVARIANCE_GRID",
        "partial_grid_excluded_from_lean_primary_gate": True,
        "unchanged_scientific_elements": unchanged,
        "purity_boundary": {
            "new_setting_outcome_access": "ERM_COMPETENCE_ONLY",
            "invariance_outcome_deltas_sealed": True,
            "direction_level_future_utilities_sealed": True,
            "openbmi_internal_holdout_untouched": True,
            "wbcic_outer_holdout_untouched_not_enumerated": True,
        },
        "forbidden_claim": "The original full-grid P4A terminal must not be claimed or synthesized.",
        "allowed_lean_terminals": [
            "P4A_LEAN_CROSS_SETTING_CUBE_COMPLETE",
            "P4A_LEAN_PARTIAL_SETTING_FAILURE",
            "P4A_LEAN_PROTOCOL_OR_PURITY_FAILURE",
        ],
    }
    json_out = EXP / "P4A_PROTOCOL_AMENDMENT_LEAN_V1.json"
    json_out.write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")

    counts = amendment["pause_snapshot"]["counts_by_setting"]
    md = f"""# P4A Protocol Amendment — Lean V1

Amendment type: `COMPUTATIONAL_SCOPE_AMENDMENT`.

This is **not** a scientific-outcome amendment. The original frozen P4A protocol required the full 405-configuration DANN/CORAL/MMD grid. That grid is incomplete and must not be reported as complete: **{amendment['pause_snapshot']['non_erm_grid_completed']}/405** configurations were retained (S4={counts['S4']}, S5={counts['S5']}, S6={counts['S6']}).

The amendment was recorded after all mandatory new-setting ERM runs reached **45/45**, and before any P4B new-setting direction-level future utility was accessed. The pause decision did not inspect grid outcome deltas or direction-level future utilities. It was made solely because the non-ERM grid is not required to construct the competent-ERM I/P/D/C_src/O_task evidence used by P4B.

The following remain unchanged: datasets, tasks, backbones, folds, seeds, ERM definition, I/P/D/C_src/O_task definitions, competence rule, and holdout policy. New-setting outcome access remains limited to ERM competence. Invariance outcome deltas and direction-level future utilities remain sealed. OpenBMI sealed 14 and WBCIC outer 10 remain untouched; WBCIC outer membership remains unenumerated.

The partial grid is labeled `OPTIONAL_PARTIAL_INVARIANCE_GRID`, retained in runtime, and excluded from the Lean primary gate and all P4B hypothesis/predictor/threshold/setting-selection decisions. The original full-grid terminal is not claimed. Exact pause details and hashes are in `P4A_GRID_PAUSE_SNAPSHOT.json`.
"""
    (EXP / "P4A_PROTOCOL_AMENDMENT_LEAN_V1.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
