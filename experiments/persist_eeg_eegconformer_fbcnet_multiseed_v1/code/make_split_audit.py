"""Write subject-membership-only audit before any model training."""
from __future__ import annotations

import hashlib
import json

from train_queue import EXP, FOLDS, REPO, TASKS, atomic_json, sha
from benchmark_data import _sources

OPENBMI_HELDOUT = {"4", "12", "13", "17", "18", "24", "25", "29", "36", "37", "39", "42", "51", "54"}
WBCIC_TRUE_OUTER = {"sub-4", "sub-8", "sub-10", "sub-15", "sub-20",
                    "sub-39", "sub-40", "sub-43", "sub-46", "sub-51"}


def main() -> None:
    modern, task_code = _sources()
    common, _, common_sha = modern.load_split()
    _, _, task_reference, _ = task_code.split_reference()
    rows = []
    for task in TASKS:
        for fold in FOLDS:
            if task in ("OpenBMI_MI", "WBCIC_MI"):
                dataset = task.split("_")[0]
                candidates = common[dataset]
                split_sha = common_sha
            else:
                dataset = "OpenBMI"
                candidates = task_reference["folds"]
                split_sha = task_reference["source_sha256"]
            found = next(item for item in candidates if int(item["fold_id"]) == fold)
            keys = ("inner_train_subjects", "inner_val_subjects", "outer_dev_subjects")
            groups = [set(map(str, found[key])) for key in keys]
            if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise RuntimeError(f"subject overlap {task} fold{fold}")
            heldout = WBCIC_TRUE_OUTER if dataset == "WBCIC" else OPENBMI_HELDOUT
            if any(group & heldout for group in groups):
                raise RuntimeError(f"final-heldout subject in development split {task} fold{fold}")
            rows.append({"task": task, "dataset": dataset, "fold": fold,
                         "source_split_sha256": split_sha,
                         **{key: sorted(map(str, found[key]), key=lambda s: int(s.replace("sub-", "")))
                            for key in keys},
                         "source_sessions": [0, 1] if dataset == "WBCIC" else [1],
                         "inner_validation_session": 2,
                         "outer_development_session": 2})
    if len(rows) != 20:
        raise RuntimeError("expected 20 task-fold split rows")
    audit = {"schema": "EEGCONFORMER_FBCNET_SPLIT_AUDIT_V1", "rows": rows,
             "row_count": len(rows), "all_partitions_subject_disjoint": True,
             "heldout_subjects_in_training": False,
             "final_heldout_overlap_checked_in_each_task_fold": True,
             "OpenBMI_final_heldout_ids": sorted(OPENBMI_HELDOUT, key=int),
             "WBCIC_final_true_outer_ids": sorted(WBCIC_TRUE_OUTER, key=lambda s: int(s.replace("sub-", ""))),
             "source_files": {
                 "carrier_fivefold": "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/FIVEFOLD_SPLIT.json",
                 "task_reference": "experiments/persist_eeg_openbmi_task_generality_v1/code/task_datasets.py",
                 "loader": "experiments/persist_eeg_seven_backbone_fourtask_3seed_v1/code/benchmark_data.py"}}
    atomic_json(EXP / "protocol" / "SPLIT_AUDIT.json", audit)
    print("SPLIT_AUDIT_PASS rows=20 disjoint=20", flush=True)


if __name__ == "__main__":
    main()
