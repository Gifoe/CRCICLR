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
        for seed in range(5):
            split = make_subject_split(group.subject_id.tolist(), dataset, seed)
            path = root / f"data/splits/{dataset}/seed_{seed}.json"; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"dataset":dataset,"seed":seed,"roles":split}, indent=2), encoding="utf-8")
    return 0
if __name__ == "__main__": raise SystemExit(main())

