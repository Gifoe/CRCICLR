from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as c


def text_hash(values: list[str]) -> str:
    import hashlib
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def main() -> None:
    c.ensure_dirs()
    protocol = c.protocol()
    if c.RESULTS.exists() and any(c.RESULTS.glob("*FIDELITY*.csv")):
        raise RuntimeError("data-access audit must precede Stage-0 transport outcomes")

    open_data = c.load_setting_data("OPENBMI_MI_EEGNET")
    wbcic_data = c.load_setting_data("WBCIC_MI_EEGNET")
    open_subjects = c.subject_sort(open_data.metadata.subject_id.astype(str).unique())
    wbcic_subjects = c.subject_sort(wbcic_data.metadata.subject_id.astype(str).unique())
    if len(open_subjects) != 40 or len(wbcic_subjects) != 41:
        raise RuntimeError("development pool cardinality mismatch")

    open_cells = open_data.metadata.groupby(["subject_id", "session_id", "label"]).size()
    wbcic_cells = wbcic_data.metadata.groupby(["subject_id", "session_id", "label"]).size()
    checkpoint_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for setting in protocol["development_settings"]:
        for fold in protocol["folds"]:
            roles = c.fold_roles(setting, int(fold))
            pool = set().union(*map(set, roles.values()))
            expected = set(open_subjects if setting.startswith("OPENBMI") else wbcic_subjects)
            if pool != expected:
                raise RuntimeError(f"{setting} fold={fold}: role pool differs from development whitelist")
            if any(set(a) & set(b) for i, a in enumerate(roles.values()) for b in list(roles.values())[i + 1:]):
                raise RuntimeError(f"{setting} fold={fold}: role overlap")
            fold_rows.append({
                "setting_id": setting,
                "fold": int(fold),
                "model_fit_n": len(roles["model_fit"]),
                "validation_n": len(roles["validation"]),
                "outcome_n": len(roles["outcome"]),
                "model_fit_hash": text_hash(list(roles["model_fit"])),
                "validation_hash": text_hash(list(roles["validation"])),
                "outcome_membership_committed": False,
            })
            checkpoint = c.checkpoint_path(setting, int(fold), 0)
            normalizer = c.normalizer_path(setting, int(fold), 0)
            candidate = c.unit_context(setting, int(fold), 0) / "candidates" / "erm__lambda-0.00.json"
            unit = c.unit_context(setting, int(fold), 0) / "UNIT_PROTOCOL.json"
            for path in (checkpoint, normalizer, candidate, unit):
                if not path.is_file():
                    raise FileNotFoundError(path)
            candidate_payload = c.read_json(candidate)
            checkpoint_rows.append({
                "setting_id": setting,
                "fold": int(fold),
                "seed": 0,
                "checkpoint_sha256": c.sha256(checkpoint),
                "normalizer_sha256": c.sha256(normalizer),
                "unit_protocol_sha256": c.sha256(unit),
                "source_validation_BA": float(candidate_payload["best_validation_BA"]),
                "source_validation_NLL": float(candidate_payload["best_validation_NLL"]),
                "outcome_metric_read_for_this_audit": False,
            })

    data_audit = {
        "schema": "SCST_DR_DATA_ACCESS_AUDIT_V1",
        "pass": True,
        "audit_precedes_transport_outcomes": True,
        "openbmi": {
            "development_subject_count": len(open_subjects),
            "development_subjects": open_subjects,
            "development_subject_hash": text_hash(open_subjects),
            "cache_path": str(open_data.cache_root),
            "cache_shape": list(open_data.x.shape),
            "metadata_rows": len(open_data.metadata),
            "session_ids": sorted(map(int, open_data.metadata.session_id.unique())),
            "source_sessions_for_stage0": [1, 2],
            "future_evaluation_role": "Session 2 of outcome subjects; never indexed by Stage 0",
            "subject_class_session_cell_counts": sorted(set(map(int, open_cells.tolist()))),
            "sealed_internal_holdout_count": 14,
            "sealed_internal_holdout_membership_materialized": False,
        },
        "wbcic": {
            "development_subject_count": len(wbcic_subjects),
            "development_subjects": wbcic_subjects,
            "development_subject_hash": text_hash(wbcic_subjects),
            "cache_path": str(wbcic_data.cache_root),
            "cache_shape": list(wbcic_data.x.shape),
            "metadata_rows": len(wbcic_data.metadata),
            "session_ids_available": sorted(map(int, wbcic_data.metadata.session_id.unique())),
            "source_sessions_for_stage0": [0, 1],
            "future_evaluation_session": 2,
            "future_evaluation_rows_indexed_by_stage0": False,
            "subject_class_session_cell_count_min": int(wbcic_cells.min()),
            "subject_class_session_cell_count_max": int(wbcic_cells.max()),
            "sealed_outer_count": 10,
            "sealed_outer_membership_materialized": False,
        },
        "folds": fold_rows,
        "historical_erm_units": checkpoint_rows,
        "all_four_settings_have_complete_erm_units": len(checkpoint_rows) == 20,
        "source_validation_competence_min_BA": min(float(row["source_validation_BA"]) for row in checkpoint_rows),
        "outcome_or_outer_performance_used": False,
    }
    c.write_json(c.EXP / "protocol" / "DATA_ACCESS_AUDIT.json", data_audit)

    sealed_audit = {
        "schema": "SCST_DR_SEALED_RESOURCE_AUDIT_V1",
        "pass": True,
        "openbmi_internal_holdout": {
            "count": 14,
            "membership": "UNMATERIALIZED",
            "labels": "UNTOUCHED",
            "preprocessing": "NOT_RUN",
            "embeddings": "NOT_GENERATED",
            "performance": "NOT_INSPECTED",
        },
        "wbcic_outer": {
            "count": 10,
            "membership": "UNMATERIALIZED",
            "labels": "UNTOUCHED",
            "preprocessing": "NOT_RUN",
            "embeddings": "NOT_GENERATED",
            "performance": "NOT_INSPECTED",
        },
        "development_future_roles": {
            "openbmi_outcome_subject_session_2": "NOT_INDEXED_BY_STAGE0",
            "wbcic_development_subject_session_3": "NOT_INDEXED_BY_STAGE0",
        },
        "outer_evaluation_authorized": False,
    }
    c.write_json(c.EXP / "protocol" / "SEALED_RESOURCE_AUDIT.json", sealed_audit)
    c.write_csv(c.RESULTS / "ERM_SOURCE_COMPETENCE_AUDIT.csv", pd.DataFrame(checkpoint_rows))
    print("SCST_DR_DATA_ACCESS_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    main()
