#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from _common import parser
from hsc_tta.splits import make_subject_split
from hsc_tta.utils import require_cpu


def main() -> int:
    args = parser("Create deterministic subject-disjoint splits").parse_args(); require_cpu(args.device)
    root = Path("/root/autodl-tmp/hsc_tta_eeg")
    subjects = pd.read_parquet(root / "data/manifests/subjects.parquet")
    for dataset, group in subjects[subjects.eligible].groupby("dataset"):
        subject_ids = group.subject_id.tolist()
        if dataset == "cap":
            protocol_path = root / "repo/CHANNEL_PROTOCOL.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            if protocol.get("protocol_hash") != "2e35eff22ad71af3cf30612602934a97f5b0cb610ce60fa25fed87f7b5bc71eb":
                raise RuntimeError("CAP channel protocol hash mismatch")
            if protocol.get("selected_channel") != "C4":
                raise RuntimeError("formal CAP cohort requires frozen C4 protocol")
            subject_ids = sorted(set(subject_ids) & set(protocol["cap_subject_ids"]))
            if len(subject_ids) != 99:
                raise RuntimeError(f"formal CAP C4 cohort must contain 99 subjects, found {len(subject_ids)}")
        for seed in range(5):
            split = make_subject_split(subject_ids, dataset, seed)
            if dataset == "cap" and {
                role: len(ids) for role, ids in split.items()
            } != {"target_site_calibration": 25, "external_final_test": 74}:
                raise RuntimeError("formal CAP split must be 25 calibration + 74 external test")
            path = root / f"data/splits/{dataset}/seed_{seed}.json"; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"dataset":dataset,"seed":seed,"roles":split}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0
if __name__ == "__main__": raise SystemExit(main())
