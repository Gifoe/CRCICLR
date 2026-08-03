#!/usr/bin/env python
from pathlib import Path

from hsc_tta.v2.ablations import run_ablations


if __name__ == "__main__":
    result = run_ablations(Path("/root/autodl-tmp/hsc_tta_eeg"))
    print(f"wrote {len(result)} ablation metric rows")
