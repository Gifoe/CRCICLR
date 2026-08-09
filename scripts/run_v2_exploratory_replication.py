#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hsc_tta.v2.exploratory import run_exploratory_replication


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    outputs = run_exploratory_replication(Path("/root/autodl-tmp/hsc_tta_eeg"), args.device, not args.no_resume)
    print({key: len(value) for key, value in outputs.items()})
