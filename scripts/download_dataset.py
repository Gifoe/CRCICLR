#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from _common import parser
from hsc_tta.data.download import download_records, read_records
from hsc_tta.utils import load_yaml, require_cpu


def main() -> int:
    p = parser("Resumable official PhysioNet dataset download")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    require_cpu(args.device)
    cfg = load_yaml(args.config)
    dataset, base = cfg["dataset"], cfg["official_url"]
    root = Path("/root/autodl-tmp/hsc_tta_eeg")
    target = root / "data" / "raw" / dataset
    target.mkdir(parents=True, exist_ok=True)
    records = read_records(base)
    if dataset == "eegmmidb":
        target_runs = tuple(f"R{run:02d}.edf" for run in (4, 6, 8, 10, 12, 14))
        records = [record for record in records if record.endswith(target_runs)]
    if args.smoke:
        if dataset == "eegmmidb":
            records = [record for record in records if record.startswith("S001/")][:3]
        else:
            records = records[: min(1, len(records))]
    frame = download_records(base, target, records, resume=args.resume, dry_run=args.dry_run, verify_only=args.verify_only, minimum_free_gb=60, num_workers=args.num_workers)
    manifest_path = root / "data" / "manifests" / f"{dataset}_download_manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(dataset=dataset).to_parquet(manifest_path, index=False)
    print(frame.status.value_counts(dropna=False).to_string())
    return 1 if frame.status.isin(["missing", "failed"]).any() else 0


if __name__ == "__main__": raise SystemExit(main())
