"""Fail-closed review and compact publication of the protected-coupling audit.

The frozen full outputs and the separately provenanced prediction correction
remain untouched. This script creates only a new reviewed compact directory.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import run_coupling as C
import repair_error_predictability_v2 as R


REQUIRED = (
    "PROVENANCE.json", "FINAL_PC_DECOMPOSITION_AUDIT.csv",
    "LAYER_PC_DECOMPOSITION_AUDIT.csv", "PC_MAIN_EFFECTS.csv",
    "PC_INTERACTION.csv", "PC_LOCAL_CONDITIONAL_SENSITIVITY.csv",
    "PC_RANDOM_SPECIFICITY.csv", "PC_ERROR_LINKAGE.csv",
    "PC_ERROR_PREDICTABILITY.csv", "PC_FUNCTIONAL_SURROGATE.csv",
    "PC_RANDOM_SURROGATE_CONTROL.csv", "PC_SESSION_STABILITY.csv",
    "MODEL_TASK_SUMMARY.csv", "REPORT.md",
)
COMPACT_ORIGINAL = (
    "FINAL_PC_DECOMPOSITION_AUDIT.csv", "LAYER_PC_DECOMPOSITION_AUDIT.csv",
    "PC_MAIN_EFFECTS.csv", "PC_LOCAL_CONDITIONAL_SENSITIVITY.csv",
    "PC_RANDOM_SPECIFICITY.csv", "PC_FUNCTIONAL_SURROGATE.csv",
)
COMPACT_REPAIRED = (
    "PC_ERROR_PREDICTABILITY.csv", "MODEL_TASK_SUMMARY.csv", "PROVENANCE.json",
)
DESTINATION = C.EXP / "outputs_compact"


def fail(message: str) -> None:
    raise RuntimeError(message)


def survey(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing or empty full output: {path.name}")
    result = {"sha256": C.digest(path), "bytes": path.stat().st_size}
    if path.suffix != ".csv":
        return result
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {"model", "task"}.issubset(fields):
            fail(f"invalid CSV schema: {path.name}")
        combinations = Counter()
        count = 0
        for row in reader:
            count += 1
            model, task = row["model"], row["task"]
            if model not in C.MODELS or task not in C.TASKS:
                fail(f"out-of-scope model/task in {path.name}")
            if "fold" in fields and (row["fold"] not in {"0", "1", "2", "3", "4"} or row.get("seed") != "0"):
                fail(f"out-of-scope fold/seed in {path.name}")
            if "subject_id" in fields and not row["subject_id"]:
                fail(f"missing biological subject in {path.name}")
            if path.name == "PC_INTERACTION.csv" and row["recipient_label"] == row["donor_label"]:
                fail("cross-label interaction contains a same-class pair")
            if path.name == "FINAL_PC_DECOMPOSITION_AUDIT.csv":
                if row["status"] != "COMPLETE" or row["final_heldout_accessed"] != "False":
                    fail("final audit status/heldout failure")
                if max(float(row["max_abs_z_error"]), float(row["max_abs_h_error"])) >= 1e-5:
                    fail("final P+C reconstruction failure")
            if path.name == "LAYER_PC_DECOMPOSITION_AUDIT.csv" and float(row["max_abs_reconstruction"]) >= 1e-5:
                fail("layer P+C reconstruction failure")
            combinations[(model, task)] += 1
        if not count or len(combinations) != 4:
            fail(f"incomplete CSV: {path.name}")
        result.update({"rows": count, "columns": fields, "model_task_row_counts":
                       {"/".join(k): v for k, v in sorted(combinations.items())}})
    return result


def validate_cells(original_provenance: dict, repaired_provenance: dict) -> dict:
    if original_provenance.get("cells_complete") != 20 or original_provenance.get("final_heldout_accessed") is not False:
        fail("original aggregate provenance invalid")
    if repaired_provenance.get("cells_complete") != 20 or repaired_provenance.get("final_heldout_accessed") is not False:
        fail("repaired aggregate provenance invalid")
    if repaired_provenance.get("criterion_C_confirmatory_eligible") is not False:
        fail("correction wrongly claims confirmatory criterion C")
    if repaired_provenance.get("frozen_protocol_sha256") != R.PROTOCOL_SHA:
        fail("repaired aggregate lock mismatch")
    baseline_rows = 0
    minimum_future_pairs = None
    for model in C.MODELS:
        for task in C.TASKS:
            for fold in C.FOLDS:
                key = f"{model}/{task}/{fold}"
                original_path = C.cell_path(model, task, fold)
                repaired_path = R.repaired_cell_path(model, task, fold)
                original = R.original_cell(model, task, fold)
                repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
                if original_provenance["cell_sha256"].get(key) != C.digest(original_path):
                    fail(f"original cell hash mismatch: {key}")
                if repaired_provenance["repair_cell_sha256"].get(key) != C.digest(repaired_path):
                    fail(f"repaired cell hash mismatch: {key}")
                if (repaired.get("status") != "COMPLETE" or repaired.get("source_cell_sha256") != C.digest(original_path)
                        or repaired.get("repair_implementation_sha256") != C.digest(Path(R.__file__))
                        or repaired.get("outer_true_labels_used_for_features") is not False
                        or repaired.get("outer_true_labels_used_for_targets_and_metrics_only") is not True
                        or repaired.get("final_heldout_accessed") is not False):
                    fail(f"repaired cell validation failed: {key}")
                if original.get("final_decomposition_max_abs", 1) >= 1e-5:
                    fail(f"original final decomposition failure: {key}")
                for stage in original["stages"]:
                    if stage["reconstruction_max_abs"] >= 1e-5:
                        fail(f"original stage reconstruction failure: {key}")
                    if len({row["draw"] for row in stage["random_subject_effects"]}) != 100:
                        fail(f"missing 100 random partitions: {key}")
                    if len({row["draw"] for row in stage["random_surrogate"]}) != 20:
                        fail(f"missing 20 surrogate controls: {key}")
                if len(repaired["pair_counts"]) != len(original["stages"]):
                    fail(f"repaired stage count mismatch: {key}")
                for counts in repaired["pair_counts"].values():
                    if counts["future_pairs"] < 1:
                        fail(f"no label-free future donor pairs: {key}")
                    minimum_future_pairs = min(minimum_future_pairs or counts["future_pairs"], counts["future_pairs"])
                original_base = {(r["target"], r["family"], r["subject_id"]): r
                                 for r in original["error_predictability"] if r["family"] == "BASE"}
                repaired_base = {(r["target"], r["family"], r["subject_id"]): r
                                 for r in repaired["prediction_rows"] if r["family"] == "BASE"}
                if original_base.keys() != repaired_base.keys():
                    fail(f"unchanged geometric baseline keys differ: {key}")
                for row_key in original_base:
                    for metric in ("AUROC", "AUPRC", "Brier", "ECE_10bin"):
                        old, new = original_base[row_key][metric], repaired_base[row_key][metric]
                        if (old is None) != (new is None) or (old is not None and abs(old - new) > 1e-10):
                            fail(f"unchanged geometric baseline differs: {key}/{row_key}/{metric}")
                baseline_rows += len(original_base)
    return {"original_cells_valid": 20, "repaired_cells_valid": 20,
            "unchanged_baseline_subject_target_rows": baseline_rows,
            "minimum_future_label_free_pairs_per_stage_cell": minimum_future_pairs}


def main() -> None:
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        fail("reviewed compact output already exists; no overwrite")
    if C.digest(C.EXP / "protocol" / "PROVENANCE.json") != R.PROTOCOL_SHA:
        fail("frozen protocol lock mismatch")
    if C.digest(C.EXP / "code" / "run_coupling.py") != R.ORIGINAL_IMPL_SHA:
        fail("original source hash mismatch")
    if not R.REPAIR_OUT.is_dir():
        fail("prediction repair aggregate missing")
    original_provenance = json.loads((C.OUT / "PROVENANCE.json").read_text(encoding="utf-8"))
    repaired_provenance = json.loads((R.REPAIR_OUT / "PROVENANCE.json").read_text(encoding="utf-8"))
    cell_audit = validate_cells(original_provenance, repaired_provenance)
    original_manifest = {name: survey(C.OUT / name) for name in REQUIRED}
    if original_manifest["FINAL_PC_DECOMPOSITION_AUDIT.csv"]["rows"] != 20:
        fail("final audit does not cover 20 cells")
    if original_manifest["LAYER_PC_DECOMPOSITION_AUDIT.csv"]["rows"] != 120:
        fail("layer audit does not cover 120 stage-cells")
    repaired_names = ("PROVENANCE.json", "PC_ERROR_PREDICTABILITY.csv", "MODEL_TASK_SUMMARY.csv", "REPORT.md")
    repaired_manifest = {name: survey(R.REPAIR_OUT / name) for name in repaired_names}
    for name in repaired_provenance["output_sha256"]:
        if repaired_provenance["output_sha256"][name] != repaired_manifest[name]["sha256"]:
            fail(f"repaired output hash mismatch: {name}")
    if repaired_manifest["MODEL_TASK_SUMMARY.csv"]["rows"] != 4:
        fail("repaired summary lacks four model/tasks")
    with (C.OUT / "PC_RANDOM_SPECIFICITY.csv").open(newline="", encoding="utf-8") as stream:
        specific = list(csv.DictReader(stream))
    criterion_a = [r for r in specific if float(r["P_minus_random"]) > 0
                   and float(r["subject_CI_low"]) > 0 and int(r["positive_folds"]) >= 4]
    if not criterion_a:
        fail("locked success criterion A is unsupported")
    report = (R.REPAIR_OUT / "REPORT.md").read_text(encoding="utf-8")
    old = "All mapping, pairing, PCA, surrogate fitting, and regularization choices use TRAIN only. Outer-development is evaluation only."
    updated = ("Frozen mappings, PCA, surrogate fitting, and predictor regularization are TRAIN-only. "
               "Class-conditioned outer 2x2 pairs use true classes for retrospective mechanism evaluation. "
               "The separately provenanced label-free prediction repair uses frozen native predictions and "
               "unlabeled same-subject/session batches; outer true labels are reserved for target evaluation. "
               "This is transductive batch evaluation, not online isolated-trial forecasting.")
    if report.count(old) != 1:
        fail("report methods sentence not found exactly once")
    report = report.replace(old, updated)
    if report.count("This post-outcome engineering correction is exploratory, not a pre-registered success-criterion C test.") != 4:
        fail("report does not qualify all four corrected prediction results")
    if "Future-session native-error AUROC:" in report:
        fail("original leaked AUROC remains in reviewed report")
    report += ("\n## Publication gate and retained evidence\n\n"
               "The original label-conditioned future-error AUROC and criterion C are invalid. "
               "Corrected prediction results are post-outcome exploratory. The fixed criterion A is met in "
               f"{len(criterion_a)} model/task/layer summaries with biological-subject CI above zero and "
               "at least four positive folds; this is the mechanism-based progression gate, not an error-prediction claim. "
               "Layer-wise tests are not multiplicity-adjusted. Full pair-level and random-control files remain "
               "on the original server; their SHA256 values and row counts are in PUBLISH_VALIDATION.json. "
               "Final-heldout accessed: NO.\n")
    DESTINATION.mkdir(parents=True, exist_ok=False)
    for name in COMPACT_ORIGINAL:
        shutil.copy2(C.OUT / name, DESTINATION / name)
    for name in COMPACT_REPAIRED:
        shutil.copy2(R.REPAIR_OUT / name, DESTINATION / name)
    (DESTINATION / "REPORT.md").write_text(report, encoding="utf-8")
    publication = {
        "schema": "PERSIST_EEG_PROTECTED_COMPLEMENT_COUPLING_REVIEWED_COMPACT_V2",
        "frozen_protocol_sha256": R.PROTOCOL_SHA,
        "original_implementation_sha256": R.ORIGINAL_IMPL_SHA,
        "repair_implementation_sha256": C.digest(Path(R.__file__)),
        "original_prediction_claim_valid": False,
        "corrected_prediction_is_post_outcome_exploratory": True,
        "criterion_C_confirmatory_eligible": False,
        "criterion_A_qualifying_layer_summaries": len(criterion_a),
        "criterion_A_qualifying_model_tasks": sorted({f"{r['model']}/{r['task']}" for r in criterion_a}),
        "final_heldout_accessed": False,
        **cell_audit,
        "original_full_output_manifest": original_manifest,
        "repaired_full_output_manifest": repaired_manifest,
        "compact_sha256": {p.name: C.digest(p) for p in DESTINATION.iterdir() if p.is_file()},
    }
    C.write_json(DESTINATION / "PUBLISH_VALIDATION.json", publication)
    print("REVIEWED_COMPACT_COMPLETE", len(criterion_a), cell_audit["repaired_cells_valid"], flush=True)


if __name__ == "__main__":
    main()
