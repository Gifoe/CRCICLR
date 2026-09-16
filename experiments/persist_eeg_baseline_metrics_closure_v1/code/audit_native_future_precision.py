"""Compare batch-128 frozen inference with historical future-session rows.

OpenBMI uses the identical heldout cohort and all 15 checkpoint rows. WBCIC
compares only seed0 against the committed true-outer CSGD session artifact;
the older baseline3 WBCIC output is V8 internal and is never compared.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
NATIVE = P1 / "baseline_metrics_closure_v1_native_batch_runtime" / "session_cells"
CSGD = (P1 / "CRCICLR_CROSSBACKBONE_CSGD_WORK" / "experiments"
        / "persist_eeg_crossbackbone_csgd_v1" / "outputs" / "crossbackbone_csgd_v1"
        / "SESSION_SUBJECT_RESULTS.csv")
OUT = EXP / "outputs" / "NATIVE_BATCH128_FUTURE_PRECISION_AUDIT.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    csgd = {(row["Model"], row["Task"], int(row["fold"]), int(row["seed"]), row["subject"]): row
            for row in read_csv(CSGD) if row["Model"] in ("ModernTCN", "Medformer")
            and row["Task"] == "WBCIC_MI" and row["session"] == "S2" and int(row["seed"]) == 0}
    result = []
    for model, slug in (("ModernTCN", "moderntcn"), ("Medformer", "medformer")):
        old_path = (P1 / f"CRCICLR_{slug.upper()}_FINAL" / "experiments"
                    / f"persist_eeg_{slug}_4task_3seed_final_v1" / "outputs"
                    / "HELDOUT_SUBJECT_RESULTS.csv")
        historical = {(row["task"], int(row["fold"]), int(row["seed"]), row["subject_id"]): row
                      for row in read_csv(old_path) if row["task"].startswith("OpenBMI_")}
        for task in ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI"):
            for fold in range(5):
                for seed in range(3):
                    if task == "WBCIC_MI" and seed != 0:
                        continue
                    path = NATIVE / model.lower() / task.lower() / f"fold{fold}_seed{seed}_session2.json"
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if value["batch_size"] != 128:
                        raise RuntimeError(f"native batch identity mismatch {path}")
                    for new in value["subject_rows"]:
                        subject = str(new["subject"])
                        if task == "WBCIC_MI":
                            old = csgd[model, task, fold, seed, subject]
                            source = "PUBLISHED_TRUE_OUTER_CSGD_BATCH32_SEED0"
                        else:
                            old = historical[task, fold, seed, subject]
                            source = "PUBLISHED_ORIGINAL_HELDOUT_BATCH128"
                        errors = {metric: float(new[metric]) - float(old[metric])
                                  for metric in ("BA", "macro_F1", "accuracy")}
                        if int(new["trials"]) != int(old["trials"]):
                            raise RuntimeError(f"trial count changed {model} {task} f{fold} s{seed} {subject}")
                        result.append({"model": model, "task": task, "fold": fold, "seed": seed,
                                       "subject_id": subject, "paper_future_session": "S3" if task == "WBCIC_MI" else "S2",
                                       "reference": source, "new_BA": new["BA"], "old_BA": old["BA"],
                                       "BA_difference": errors["BA"],
                                       "new_macro_F1": new["macro_F1"], "old_macro_F1": old["macro_F1"],
                                       "macro_F1_difference": errors["macro_F1"],
                                       "accuracy_difference": errors["accuracy"],
                                       "trials": new["trials"],
                                       "status": "EXACT" if max(abs(x) for x in errors.values()) <= 1e-12 else "BATCH_PRECISION_DIFFERENCE"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    open_rows = [row for row in result if row["task"] != "WBCIC_MI"]
    wbcic_rows = [row for row in result if row["task"] == "WBCIC_MI"]
    summary = {"OpenBMI_exact_rows": sum(row["status"] == "EXACT" for row in open_rows),
               "OpenBMI_total_rows": len(open_rows),
               "WBCIC_batch32_reference_exact_rows": sum(row["status"] == "EXACT" for row in wbcic_rows),
               "WBCIC_seed0_total_rows": len(wbcic_rows),
               "WBCIC_max_abs_BA_difference": max(abs(row["BA_difference"]) for row in wbcic_rows),
               "WBCIC_reference_limitation": "CSGD uses batch32; native original baseline evaluator uses batch128. Differences, if any, are diagnostic precision differences, not model selection."}
    (EXP / "protocol" / "NATIVE_BATCH_PRECISION_AUDIT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)
    if summary["OpenBMI_exact_rows"] != summary["OpenBMI_total_rows"]:
        raise RuntimeError("native batch128 does not exactly reproduce OpenBMI published subject rows")


if __name__ == "__main__":
    main()
