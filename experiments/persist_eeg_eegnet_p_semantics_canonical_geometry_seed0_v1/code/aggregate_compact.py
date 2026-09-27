"""Verify and combine the 5 x 4 fold-stage compact mechanism audits.

The input tree is a copy of the server's ``analysis`` directory containing
only CSV and AUDIT.json files. Raw EEG, model weights, and geometry arrays are
not needed or exported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTS = ("semantics", "decoders", "transform", "permutations")
EXPECTED_PART_FILES = {
    "semantics": {"BASELINE_PERSISTENCE.csv", "TASK_RELATION_WITHIN_SUBJECT.csv",
                  "TASK_RELATION_CROSS_SUBJECT.csv", "REPRESENTATION_VARIANCE_DECOMPOSITION.csv",
                  "ENERGY_COVARIANCE.csv"},
    "decoders": {"P_SEMANTIC_PROBES.csv", "FROZEN_DECODER_SESSION_TRANSFER.csv",
                 "FROZEN_DECODER_CROSS_SUBJECT_TRANSFER.csv", "DECODER_BOUNDARY_STABILITY.csv"},
    "transform": {"GLOBAL_SESSION_TRANSFORM.csv", "CROSS_SUBJECT_TRANSFORM.csv",
                  "TRANSFORM_DECODER_RECOVERY.csv", "TRANSFORMATION_COMPLEXITY_CURVE.csv"},
    "permutations": {"PERMUTATION_CONTROL_SUMMARY.csv"},
}
REQUIRED = {
    "BASELINE_PERSISTENCE.csv", "TASK_RELATION_WITHIN_SUBJECT.csv",
    "TASK_RELATION_CROSS_SUBJECT.csv", "P_SEMANTIC_PROBES.csv",
    "REPRESENTATION_VARIANCE_DECOMPOSITION.csv",
    "FROZEN_DECODER_SESSION_TRANSFER.csv", "FROZEN_DECODER_CROSS_SUBJECT_TRANSFER.csv",
    "DECODER_BOUNDARY_STABILITY.csv", "ORIGINAL_HEAD_P_FUNCTION.csv",
    "GLOBAL_SESSION_TRANSFORM.csv", "CROSS_SUBJECT_TRANSFORM.csv",
    "TRANSFORM_DECODER_RECOVERY.csv", "TRANSFORMATION_COMPLEXITY_CURVE.csv",
    "PERMUTATION_CONTROL_SUMMARY.csv",
}
OPTIONAL = {"ENERGY_COVARIANCE.csv"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise RuntimeError(f"invalid CSV header: {path}")
        data = list(reader)
    if any(None in item or any(value is None for value in item.values()) for item in data):
        raise RuntimeError(f"malformed CSV row: {path}")
    return reader.fieldnames, data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    lock_path = ROOT / "protocol" / "PROTOCOL_LOCK.json"
    geometry_path = ROOT / "protocol" / "GEOMETRY_PROVENANCE.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry["protocol_lock_sha256"] != sha256(lock_path):
        raise RuntimeError("geometry/protocol lock mismatch")
    fold_geometry = {item["fold"]: item for item in geometry["folds"]}
    collected: dict[str, list[dict[str, str]]] = defaultdict(list)
    headers: dict[str, list[str]] = {}
    manifest = []
    audit_manifest = []
    for fold in lock["folds"]:
        g = fold_geometry[fold]
        for stage in lock["analysis"]["stages"]:
            for part in PARTS:
                location = args.analysis / f"fold{fold}_seed0" / stage / part
                audit_path = location / "AUDIT.json"
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if audit["fold"] != fold or audit["stage"] != stage:
                    raise RuntimeError(f"fold/stage mismatch: {audit_path}")
                if audit["checkpoint_sha256"] != g["checkpoint_sha256"]:
                    raise RuntimeError(f"checkpoint mismatch: {audit_path}")
                if audit["geometry_provenance_sha256"] != g["source_sha256"]:
                    raise RuntimeError(f"geometry provenance mismatch: {audit_path}")
                if audit["stage_geometry_sha256"] != g["stage_geometry_sha256"][stage]:
                    raise RuntimeError(f"stage geometry mismatch: {audit_path}")
                if audit["model_state_sha256_before_after"] != g["model_state_sha256_before_after"]:
                    raise RuntimeError(f"frozen model/BatchNorm state mismatch: {audit_path}")
                if audit["final_heldout_eeg_reads"] != 0:
                    raise RuntimeError(f"final-heldout access: {audit_path}")
                if part in ("semantics", "decoders"):
                    if audit["outer_development_fit_rows"] != 0 or audit["checkpoint_validation_fit_rows"] != 0:
                        raise RuntimeError(f"held population used for fit: {audit_path}")
                    if audit["fits_use_train_only"] is not True:
                        raise RuntimeError(f"non-TRAIN fit: {audit_path}")
                    if audit["role_subject_counts"] != {
                        "TRAIN_GEOMETRY": 26, "CHECKPOINT_VALIDATION": 6, "OUTER_DEVELOPMENT": 8}:
                        raise RuntimeError(f"subject-role count mismatch: {audit_path}")
                    if audit["role_trial_rows"] != {
                        "TRAIN_GEOMETRY": 3328, "CHECKPOINT_VALIDATION": 768, "OUTER_DEVELOPMENT": 1024}:
                        raise RuntimeError(f"trial-role count mismatch: {audit_path}")
                if part == "transform":
                    if (audit["schema"] != "P_SEMANTICS_FOLD_STAGE_TRANSFORM_V3"
                            or audit["transform_fit_population"] != "TRAIN_GEOMETRY_ONLY"
                            or audit["target_labels_used_for_unsupervised_fit"] is not False
                            or audit["TRAIN_labels_used_for_unsupervised_rank_CV"] is not False
                            or audit["unsupervised_rank_CV_metric"] != "unlabeled_subject_mean_plus_gram_residual"
                            or audit["oracle_is_train_label_assisted_diagnostic"] is not True
                            or audit["target_decoder_refit"] is not False):
                        raise RuntimeError(f"transform integrity failure: {audit_path}")
                if part == "permutations" and (
                        audit["session_identity_permutations"] < 200 or
                        audit["within_cell_trial_label_permutations"] < 200 or
                        audit["random_subspace_draws"] < 20):
                    raise RuntimeError(f"insufficient permutation control: {audit_path}")
                audit_manifest.append({"fold": fold, "stage": stage, "part": part,
                                       "sha256": sha256(audit_path),
                                       "final_heldout_eeg_reads": audit["final_heldout_eeg_reads"]})
                declared = set(audit["files"])
                actual = {p.name for p in location.iterdir() if p.suffix.lower() == ".csv"}
                if declared != actual:
                    raise RuntimeError(f"file inventory mismatch: {location}")
                expected = EXPECTED_PART_FILES[part].copy()
                if part == "decoders" and stage == "embedding_64d":
                    expected.add("ORIGINAL_HEAD_P_FUNCTION.csv")
                if declared != expected:
                    raise RuntimeError(f"part file set mismatch: {location}; {declared ^ expected}")
                for name, item in audit["files"].items():
                    if name not in REQUIRED | OPTIONAL:
                        raise RuntimeError(f"unexpected scientific output: {location / name}")
                    path = location / name
                    if sha256(path) != item["sha256"]:
                        raise RuntimeError(f"output hash mismatch: {path}")
                    header, data = rows(path)
                    if len(data) != item["rows"]:
                        raise RuntimeError(f"output row count mismatch: {path}")
                    if name == "ORIGINAL_HEAD_P_FUNCTION.csv" and stage != "embedding_64d":
                        raise RuntimeError(f"head output outside embedding: {path}")
                    if name == "ORIGINAL_HEAD_P_FUNCTION.csv" and any(
                            row["causal_attribution_claim"].lower() != "false" or
                            row["head_frozen"].lower() != "true" for row in data):
                        raise RuntimeError(f"head claim/frozen-state violation: {path}")
                    if name == "GLOBAL_SESSION_TRANSFORM.csv" and any(
                            row["fit_population"] != "TRAIN_GEOMETRY_ONLY" or
                            row["target_decoder_refit"].lower() != "false" or
                            row["transform_uses_target_labels"].lower() != "false" or
                            int(row["train_CV_lowrank_rank"]) not in (1, 2, 4, 8) or
                            (row["mode"] == "UNSUPERVISED" and
                             row["train_CV_rank_selection_metric"] != "unlabeled_subject_mean_plus_gram_residual") or
                            (row["mode"] == "LABEL_ASSISTED_ORACLE" and
                             row["train_CV_rank_selection_metric"] != "TRAIN_class_relation_residual") or
                            (row["mode"] == "UNSUPERVISED" and
                             row["oracle_train_class_correspondence"].lower() != "false")
                            for row in data):
                        raise RuntimeError(f"global transform row-level role violation: {path}")
                    if name == "CROSS_SUBJECT_TRANSFORM.csv" and any(
                            (row["arm"] == "TARGET_UNLABELED_ADAPTATION_DIAGNOSTIC" and
                             (row["target_labels_used_for_transform"].lower() != "false" or
                              int(row["target_unlabeled_calibration_rows"]) < 1 or
                              int(row["independent_evaluation_rows"]) < 1)) or
                            (row["arm"] in ("ZERO_SHOT_GLOBAL", "TRAIN_LABEL_ORACLE_GLOBAL") and
                             (row["fit_population"] != "TRAIN_GEOMETRY_ONLY" or
                              row["target_decoder_refit"].lower() != "false"))
                            for row in data):
                        raise RuntimeError(f"cross-subject transform role violation: {path}")
                    if name == "PERMUTATION_CONTROL_SUMMARY.csv":
                        session_rows = [row for row in data if row["control"] == "SESSION2_SUBJECT_ID_PERMUTATION"]
                        label_rows = [row for row in data if row["control"] == "WITHIN_SUBJECT_SESSION_TRIAL_LABEL_PERMUTATION"]
                        if (len(data) != 161 or len(session_rows) != 138 or len(label_rows) != 23 or
                                any(int(row["permutations"]) < 200 for row in data) or
                                any(row["population"] != "OUTER_DEVELOPMENT" or
                                    row["target_labels_used_for_fit"].lower() != "false"
                                    for row in label_rows)):
                            raise RuntimeError(f"permutation inventory violation: {path}")
                    if name in headers and headers[name] != header:
                        raise RuntimeError(f"CSV schema drift: {path}")
                    headers[name] = header
                    if any(row.get("fold") != str(fold) or row.get("stage") != stage for row in data):
                        raise RuntimeError(f"row provenance mismatch: {path}")
                    collected[name].extend(data)
                    manifest.append({"fold": fold, "stage": stage, "part": part,
                                     "file": name, "sha256": item["sha256"], "rows": len(data)})
    if set(collected) != REQUIRED | OPTIONAL:
        raise RuntimeError(f"missing required outputs: {(REQUIRED | OPTIONAL) - set(collected)}")
    energy = collected["ENERGY_COVARIANCE.csv"]
    for fold in lock["folds"]:
        for stage in lock["analysis"]["stages"]:
            group = [row for row in energy if row["fold"] == str(fold) and row["stage"] == stage
                     and row["population"] == "TRAIN_GEOMETRY"]
            p_rank = {row["rank"] for row in group if row["family"] == "P"}
            random_rows = [row for row in group if row["family"] == "RANDOM"]
            if len(p_rank) != 1 or len(random_rows) != 20 or any(row["rank"] not in p_rank for row in random_rows):
                raise RuntimeError(f"random/P rank mismatch fold={fold} stage={stage}")
            if {row["random_draw"] for row in random_rows} != {str(i) for i in range(20)}:
                raise RuntimeError(f"random draw inventory mismatch fold={fold} stage={stage}")
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    destinations = [output / name for name in collected] + [
        output / "HASH_INDEX.json", output / "FINAL_HELDOUT_EXCLUSION_AUDIT.json"]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite existing output: {existing}")
    for name, data in collected.items():
        path = output / name
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers[name], lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)
    index = {"schema": "P_SEMANTICS_COMPACT_HASH_INDEX_V1",
             "protocol_lock_sha256": sha256(lock_path),
             "geometry_provenance_sha256": sha256(geometry_path),
             "upstream_source_audit_sha256": sha256(ROOT / "protocol" / "UPSTREAM_SOURCE_AUDIT.json"),
             "aborted_preexecution_tasks_sha256": sha256(ROOT / "protocol" / "ABORTED_PREEXECUTION_TASKS.md"),
             "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                             for path in sorted((ROOT / "code").iterdir())
                             if path.is_file() and path.suffix.lower() in (".py", ".ps1")},
             "stage_file_manifest": manifest,
             "aggregate_files": {name: {"sha256": sha256(output / name), "rows": len(data)}
                                 for name, data in sorted(collected.items())},
             "final_heldout_eeg_reads": 0}
    (output / "HASH_INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    exclusion = {
        "schema": "P_SEMANTICS_FINAL_HELDOUT_EXCLUSION_AUDIT_V1",
        "scope": "current frozen-mechanism experiment only; historical checkpoint/outer-dev exposure disclosed separately",
        "geometry_folds_verified": len(fold_geometry),
        "analysis_fold_stage_parts_verified": len(audit_manifest),
        "all_geometry_final_heldout_eeg_reads_zero": all(
            item["final_heldout_eeg_reads"] == 0 for item in geometry["folds"]),
        "all_analysis_final_heldout_eeg_reads_zero": all(
            item["final_heldout_eeg_reads"] == 0 for item in audit_manifest),
        "analysis_audit_manifest": audit_manifest,
        "protocol_lock_sha256": sha256(lock_path),
        "geometry_provenance_sha256": sha256(geometry_path),
        "upstream_source_audit_sha256": sha256(ROOT / "protocol" / "UPSTREAM_SOURCE_AUDIT.json"),
        "aborted_preexecution_tasks_sha256": sha256(ROOT / "protocol" / "ABORTED_PREEXECUTION_TASKS.md"),
        "final_heldout_eeg_reads": 0,
    }
    exclusion_path = output / "FINAL_HELDOUT_EXCLUSION_AUDIT.json"
    exclusion_path.write_text(json.dumps(exclusion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(index["aggregate_files"], sort_keys=True))


if __name__ == "__main__":
    main()
