"""Check frozen future-session numerical targets and cohort identity.

The gate deliberately stops before new inference or WS-BA aggregation when
the supplied numerical target belongs to a different WBCIC subject cohort.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
EXPECTED = {
    "ModernTCN": {
        "OpenBMI_MI": (0.616952380952381, 0.6131284240652964),
        "OpenBMI_ERP": (0.7981471861471863, 0.7683683543195192),
        "OpenBMI_SSVEP": (0.8441904761904762, 0.8424344193334482),
        "WBCIC_MI": (0.7177333333333333, 0.716095195124561),
    },
    "Medformer": {
        "OpenBMI_MI": (0.6715714285714287, 0.6663734978222521),
        "OpenBMI_ERP": (0.8007344877344877, 0.7889534338698844),
        "OpenBMI_SSVEP": (0.8860952380952379, 0.8846645384445041),
        "WBCIC_MI": (0.7673, 0.7634576065238258),
    },
}
RECENT = {
    "ModernTCN": "moderntcn",
    "Medformer": "medformer",
}
INTERNAL_WBCIC = {
    "sub-2", "sub-3", "sub-17", "sub-19", "sub-21",
    "sub-25", "sub-31", "sub-33", "sub-38", "sub-42",
}
TRUE_OUTER_WBCIC = {
    "sub-4", "sub-8", "sub-10", "sub-15", "sub-20",
    "sub-39", "sub-40", "sub-43", "sub-46", "sub-51",
}
TOLERANCE = 1e-8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    rows: list[dict[str, object]] = []
    for model, targets in EXPECTED.items():
        slug = RECENT[model]
        name = f"persist_eeg_{slug}_4task_3seed_final_v1"
        path = P1 / f"CRCICLR_{slug.upper()}_FINAL" / "experiments" / name / "outputs" / "HELDOUT_SUBJECT_RESULTS.csv"
        source_rows = read_csv(path)
        for task, (ba_target, f1_target) in targets.items():
            subset = [row for row in source_rows if row["task"] == task]
            subjects = {row["subject_id"] for row in subset}
            observed_ba = mean([float(row["BA"]) for row in subset])
            observed_f1 = mean([float(row["macro_F1"]) for row in subset])
            numeric_match = abs(observed_ba - ba_target) <= TOLERANCE and abs(observed_f1 - f1_target) <= TOLERANCE
            if task == "WBCIC_MI":
                cohort = "V8_INTERNAL" if subjects == INTERNAL_WBCIC else "TRUE_OUTER" if subjects == TRUE_OUTER_WBCIC else "OTHER"
                status = "COHORT_MISMATCH_STOP" if cohort != "TRUE_OUTER" else "VALID_FINAL_COHORT"
            else:
                cohort = "OPENBMI_14"
                status = "EXISTING_FUTURE_RESULT_MATCH_ONLY" if numeric_match else "NUMERIC_MISMATCH_STOP"
            rows.append({
                "model": model, "task": task, "expected_BA": ba_target,
                "observed_BA": observed_ba, "abs_BA_error": abs(observed_ba - ba_target),
                "expected_macro_F1": f1_target, "observed_macro_F1": observed_f1,
                "abs_macro_F1_error": abs(observed_f1 - f1_target),
                "numeric_match": numeric_match, "source_subject_count": len(subjects),
                "source_cohort": cohort, "status": status,
                "note": "This checks existing committed-style output, not an independent all-session rerun."
                if task != "WBCIC_MI" else
                "Prompt target is V8 internal 10 subjects, disjoint from final WBCIC true outer 10; cannot use as final regression target.",
            })
    output = EXP / "outputs" / "FUTURE_RESULT_REGRESSION_AUDIT.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (EXP / "protocol" / "REGRESSION_GATE.json").write_text(json.dumps({
        "pass": False,
        "stop_reason": "WBCIC expected values and ModernTCN/Medformer existing future output use V8 internal subjects, not final true outer subjects",
        "internal_subjects": sorted(INTERNAL_WBCIC),
        "true_outer_subjects": sorted(TRUE_OUTER_WBCIC),
        "cohort_overlap": sorted(INTERNAL_WBCIC & TRUE_OUTER_WBCIC),
        "tolerance": TOLERANCE,
        "new_inference_performed": False,
        "training_performed": False,
        "checkpoint_reselection_performed": False,
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["model"], row["task"], row["status"], row["numeric_match"])


if __name__ == "__main__":
    main()
