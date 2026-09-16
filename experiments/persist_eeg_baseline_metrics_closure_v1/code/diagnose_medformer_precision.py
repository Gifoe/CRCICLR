"""One-cell batch-partition diagnosis for frozen Medformer ERP regression."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
P1 = Path(r"D:\nips-temp\TotalP\P1")
WRAPPER = EXP / "code" / "run_frozen_sessions.py"
REFERENCE = (P1 / "CRCICLR_MEDFORMER_FINAL" / "experiments"
             / "persist_eeg_medformer_4task_3seed_final_v1" / "outputs"
             / "HELDOUT_SUBJECT_RESULTS.csv")


def main() -> None:
    spec = importlib.util.spec_from_file_location("precision_wrapper", WRAPPER)
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    csgd = wrapper.source()
    lock = wrapper.read_lock(csgd)
    record = next(row for row in lock["evaluation_checkpoints"]
                  if (row["Model"], row["Task"], row["fold"], row["seed"]) ==
                  ("Medformer", "OpenBMI_ERP", 4, 2))
    with REFERENCE.open(newline="", encoding="utf-8-sig") as handle:
        expected = {row["subject_id"]: row for row in csv.DictReader(handle)
                    if (row["task"], int(row["fold"]), int(row["seed"])) ==
                    ("OpenBMI_ERP", 4, 2)}
    plan = csgd.fold_plan("OpenBMI_ERP", 4, lock["cohorts"]["OpenBMI"]["subjects"])
    if plan["normalizer"]["mean_std_sha256"] != record["normalizer_sha256"]:
        raise RuntimeError("normalizer mismatch")
    value, labels, subjects = csgd.evaluation_session(plan, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = csgd.build_model(record, device)
    for batch in (32, 128):
        actual = csgd.infer(model, value, labels, subjects, batch, device)
        mismatches = []
        for row in actual:
            old = expected[str(row["subject"])]
            if abs(row["BA"] - float(old["BA"])) > 1e-12 or abs(row["macro_F1"] - float(old["macro_F1"])) > 1e-12:
                mismatches.append((row["subject"], row["BA"] - float(old["BA"]),
                                   row["macro_F1"] - float(old["macro_F1"])))
        print(f"BATCH={batch} mismatches={mismatches}", flush=True)


if __name__ == "__main__":
    main()
