"""Fail-closed preflight for the frozen WBCIC Phase-3 experiment."""
from __future__ import annotations

import hashlib
import json
import py_compile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import common


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROVENANCE = EXP / "provenance"
EXPECTED_HASH = "dae8e7ec00cbcf6dcc8c5b25829f2148fd0b5fdf162f75a0cddc18b096af7db4"


def sha_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    cfg = common.protocol()
    scope = json.loads((PROVENANCE / "DEVELOPMENT_SCOPE_LOCK.json").read_text(encoding="utf-8"))
    allowed = list(map(str, scope["allowed_subjects"]))
    cache_audit = json.loads((PROVENANCE / "CONSOLIDATED_CACHE_AUDIT.json").read_text(encoding="utf-8"))
    if len(allowed) != 41 or sha_lines(allowed) != EXPECTED_HASH:
        raise RuntimeError("41-subject scope mismatch")
    if cache_audit.get("pass") is not True or cache_audit.get("subject_hash") != EXPECTED_HASH:
        raise RuntimeError("consolidated cache did not pass its scope/integrity audit")
    data = common.load_data()
    if set(data.metadata.subject_id.astype(str)) != set(common.frozen_subjects()):
        raise RuntimeError("loaded cache subject set differs from frozen protocol")

    outcome_counts = {subject: 0 for subject in common.frozen_subjects()}
    validation_counts = {subject: 0 for subject in common.frozen_subjects()}
    fold_rows = []
    for fold in range(5):
        roles = common.frozen_fold(fold)
        for subject in roles["outcome"]:
            outcome_counts[subject] += 1
        for subject in roles["validation_discovery"]:
            validation_counts[subject] += 1
        train = common.row_indices(data.metadata, roles["model_fit"], (0, 1))
        validation = common.row_indices(data.metadata, roles["validation_discovery"], (2,))
        outcome = common.row_indices(data.metadata, roles["outcome"], (2,))
        if set(train) & set(validation) or set(train) & set(outcome) or set(validation) & set(outcome):
            raise RuntimeError(f"row overlap in fold {fold}")
        fold_rows.append(
            {
                "fold": fold,
                "model_fit_subjects": len(roles["model_fit"]),
                "validation_discovery_subjects": len(roles["validation_discovery"]),
                "outcome_subjects": len(roles["outcome"]),
                "model_fit_S1_S2_rows": len(train),
                "validation_discovery_S3_rows": len(validation),
                "outcome_S3_rows": len(outcome),
            }
        )
    if set(outcome_counts.values()) != {1} or set(validation_counts.values()) != {1}:
        raise RuntimeError("subjects do not appear exactly once in outcome and validation roles")

    for path in sorted(HERE.glob("*.py")):
        py_compile.compile(str(path), doraise=True)
    runtime_files = [HERE / name for name in ("cache.py", "assemble_cache.py", "common.py", "run_unit.py") if (HERE / name).is_file()]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
    forbidden_runtime_tokens = ["outer.py" + " evaluate", "FINAL_OUTER" + "_EVALUATION_LOCK"]
    found = [token for token in forbidden_runtime_tokens if token in source_text]
    if found:
        raise RuntimeError(f"forbidden restricted-data runtime token in Phase-3 code: {found}")

    payload = {
        "schema": "WBCIC_REPLICATION_PREFLIGHT_V1",
        "pass": True,
        "protocol_schema": cfg["schema"],
        "repository_start_sha": cfg["repository_start_sha"],
        "authorized_subject_count": 41,
        "authorized_subject_hash": EXPECTED_HASH,
        "materialized_session_count": 123,
        "materialized_trial_count": int(len(data.metadata)),
        "cache_shape": list(data.x.shape),
        "cache_dtype": str(data.x.dtype),
        "folds": fold_rows,
        "each_subject_outcome_exactly_once": True,
        "each_subject_validation_exactly_once": True,
        "all_code_compiles": True,
        "restricted_runtime_tokens_absent": True,
        "sealed_WBCIC_outer_accessed": False,
        "sealed_WBCIC_outer_enumerated": False,
        "OpenBMI_holdout_accessed": False,
        "outcome_evaluation_started": False,
    }
    write_json(EXP / "results" / "PREFLIGHT.json", payload)
    pd.DataFrame(fold_rows).to_csv(EXP / "results" / "preflight_fold_cardinalities.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
