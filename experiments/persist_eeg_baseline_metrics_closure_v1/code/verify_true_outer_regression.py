"""Independent seed-0 WBCIC true-outer rerun against committed CSGD rows."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
SOURCE_CSV = (P1 / "CRCICLR_CROSSBACKBONE_CSGD_WORK" / "experiments"
              / "persist_eeg_crossbackbone_csgd_v1" / "outputs" / "crossbackbone_csgd_v1"
              / "SESSION_SUBJECT_RESULTS.csv")
WRAPPER = EXP / "code" / "run_frozen_sessions.py"
OUT = EXP / "outputs" / "TRUE_OUTER_SEED0_INDEPENDENT_REGRESSION.csv"
MODELS = ("ModernTCN", "Medformer")
TOLERANCE = 1e-8


def main() -> None:
    spec = importlib.util.spec_from_file_location("closure_run_wrapper", WRAPPER)
    if spec is None or spec.loader is None:
        raise ImportError(WRAPPER)
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    csgd = wrapper.source()
    lock = wrapper.read_lock(csgd)
    with SOURCE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        reference = {(row["Model"], row["Task"], int(row["fold"]), int(row["seed"]),
                      str(row["subject"]), row["session"]): row
                     for row in csv.DictReader(handle)
                     if row["Model"] in MODELS and row["Task"] == "WBCIC_MI"
                     and int(row["seed"]) == 0 and row["session"] == "S2"}
    expected = {(model, "WBCIC_MI", fold, 0, subject, "S2")
                for model in MODELS for fold in range(5)
                for subject in lock["cohorts"]["WBCIC"]["subjects"]}
    if set(reference) != expected:
        raise RuntimeError(f"published true-outer seed0 reference coverage differs: {len(reference)} vs {len(expected)}")
    records = {(row["Model"], row["Task"], int(row["fold"]), int(row["seed"])): row
               for row in lock["evaluation_checkpoints"]}
    device = csgd.torch.device("cuda" if csgd.torch.cuda.is_available() else "cpu")
    csgd.torch.set_num_threads(4)
    rows = []
    for model in MODELS:
        for fold in range(5):
            record = records[model, "WBCIC_MI", fold, 0]
            plan = csgd.fold_plan("WBCIC_MI", fold, lock["cohorts"]["OpenBMI"]["subjects"])
            if plan["normalizer"]["mean_std_sha256"] != record["normalizer_sha256"]:
                raise RuntimeError("train-only normalizer hash differs")
            if plan["split_sha256"] != record["split_sha256"]:
                raise RuntimeError("frozen split hash differs")
            value, labels, subjects = csgd.evaluation_session(plan, 2)
            model_instance = csgd.build_model(record, device)
            fresh = csgd.infer(model_instance, value, labels, subjects,
                               csgd.inference_batch(model, record.get("batch_size_recorded")), device)
            del model_instance, value, labels, subjects
            if device.type == "cuda":
                csgd.torch.cuda.empty_cache()
            for result in fresh:
                key = (model, "WBCIC_MI", fold, 0, str(result["subject"]), "S2")
                old = reference[key]
                if old["checkpoint_sha256"] != record["checkpoint_sha256"]:
                    raise RuntimeError("published checkpoint hash differs")
                if old["normalizer_sha256"] != record["normalizer_sha256"]:
                    raise RuntimeError("published normalizer hash differs")
                errors = {field: abs(float(result[field]) - float(old[field]))
                          for field in ("BA", "macro_F1", "accuracy")}
                status = "PASS" if max(errors.values()) <= TOLERANCE and int(result["trials"]) == int(old["trials"]) else "FAIL"
                rows.append({"model": model, "task": "WBCIC_MI", "fold": fold, "seed": 0,
                             "subject_id": str(result["subject"]), "paper_session": "S3",
                             "fresh_BA": result["BA"], "published_BA": old["BA"],
                             "fresh_macro_F1": result["macro_F1"], "published_macro_F1": old["macro_F1"],
                             "max_abs_metric_error": max(errors.values()), "trials": result["trials"],
                             "checkpoint_hash": record["checkpoint_sha256"],
                             "normalizer_hash": record["normalizer_sha256"], "status": status})
            print(f"INDEPENDENT_REGRESSION model={model} fold={fold} subjects={len(fresh)}", flush=True)
    if len(rows) != len(expected) or any(row["status"] != "PASS" for row in rows):
        raise RuntimeError("true-outer seed0 independent regression failed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (EXP / "protocol" / "TRUE_OUTER_REGRESSION_SCOPE.json").write_text(json.dumps({
        "status": "SEED0_INDEPENDENT_PASS",
        "source_branch": "codex/persist-eeg-crossbackbone-csgd-v1",
        "source_commit": "4020f0bb0a24d5ef70bb9ba0e2116b7b0f257ebb",
        "comparison": "fresh frozen inference versus committed per-subject WBCIC true-outer future-session CSGD rows",
        "models": MODELS, "folds": 5, "seed": 0, "subjects": 10,
        "rows": len(rows), "tolerance": TOLERANCE,
        "limitation": "No independent published 15-checkpoint true-outer reference exists; this verifies seed0 only.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"TRUE_OUTER_INDEPENDENT_REGRESSION_PASS rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
