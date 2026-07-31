#!/usr/bin/env python
from __future__ import annotations
from pathlib import Path
import pandas as pd
from _common import parser
from hsc_tta.data.download import sha256_file
from hsc_tta.utils import require_cpu


def main() -> int:
    args = parser("Verify download manifests").parse_args(); require_cpu(args.device)
    root = Path("/root/autodl-tmp/hsc_tta_eeg/data/manifests")
    failures = []
    for manifest in root.glob("*_download_manifest.parquet"):
        frame = pd.read_parquet(manifest)
        for row in frame.itertuples():
            path = Path(row.filepath)
            if not path.exists() or (getattr(row, "sha256", None) and sha256_file(path) != row.sha256): failures.append(str(path))
    print(f"verification_failures={len(failures)}")
    return 1 if failures else 0
if __name__ == "__main__": raise SystemExit(main())

