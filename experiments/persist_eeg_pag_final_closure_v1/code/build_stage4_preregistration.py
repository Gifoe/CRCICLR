from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    # This is a resource audit, not a data loader. It deliberately checks only
    # the frozen repository metadata and never opens labels/features/predictions.
    audit = {
        "schema": "PERSIST_EEG_PAG_SEALED_RESOURCE_AUDIT_V1",
        "resources_available": False,
        "openbmi_internal_sealed_holdout_materialized": False,
        "wbcic_outer10_materialized": False,
        "sealed_identifiers_enumerated": False,
        "outcome_labels_read": False,
        "post_unblinding_repair": False,
        "checked_metadata": [
            {
                "path": "DATA_AUDIT.md",
                "finding": "development caches only; sealed identifiers/resources absent",
            },
            {
                "path": "V8 OUTER_LOCK.json",
                "finding": "outer resource flags remain false in the frozen audit",
            },
            {
                "path": "PUD-Aux loader",
                "finding": "loader has no internal-holdout or WBCIC-outer adapter",
            },
        ],
        "interpretation": "No sealed confirmation is authorized until independently audited sealed resources are materialized.",
    }
    write_json(ROOT / "SEALED_RESOURCE_AUDIT.json", audit)

    prereg = {
        "schema": "PERSIST_EEG_PAG_PREREGISTRATION_V1",
        "document": "PREREGISTRATION.md",
        "primary_method": "PUD-Aux",
        "comparator": "exactly matched task-only control",
        "optimization_seeds": [0, 1, 2, 3, 4],
        "primary_metric": "subject-balanced balanced accuracy",
        "minimum_meaningful_actionability_gain_pp": 0.5,
        "sealed_cohorts": ["OpenBMI internal sealed holdout", "WBCIC outer sealed subjects"],
        "sealed_outcome_access_before_lock": False,
        "protocol_change_after_lock": False,
        "decision_states": [
            "PAG_STRONG_CONFIRMATION",
            "PAG_PARTIAL_CONFIRMATION",
            "PAG_PREMISE_NOT_REPLICATED",
            "PAG_FALSIFIED_BY_ACTIONABILITY",
            "PAG_MIXED",
        ],
    }
    write_json(ROOT / "PREREGISTRATION.json", prereg)

    excluded = {
        "PREREGISTRATION_LOCK.json",
        "CODE_HASH_MANIFEST.json",
        "SEALED_CONFIRMATION_BLOCKED.json",
        "EXECUTION_LOG.md",
    }
    files: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in excluded:
            continue
        files[rel] = sha256(path)

    manifest = {
        "schema": "PERSIST_EEG_PAG_CODE_HASH_MANIFEST_V1",
        "hash_algorithm": "sha256",
        "files": files,
    }
    write_json(ROOT / "CODE_HASH_MANIFEST.json", manifest)

    lock = {
        "schema": "PERSIST_EEG_PAG_PREREGISTRATION_LOCK_V1",
        "preregistration_commit": "PENDING_FIRST_PREREGISTRATION_COMMIT",
        "protocol_document": "PREREGISTRATION.md",
        "protocol_hash": sha256(ROOT / "PREREGISTRATION.md"),
        "code_hash_manifest": "CODE_HASH_MANIFEST.json",
        "file_sha256": files,
        "sealed_outcome_labels_read": False,
        "post_unblinding_repair": False,
        "resources_available_at_lock": False,
    }
    write_json(ROOT / "PREREGISTRATION_LOCK.json", lock)


if __name__ == "__main__":
    main()
