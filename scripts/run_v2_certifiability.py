#!/usr/bin/env python
from pathlib import Path

from hsc_tta.v2.certifiability import run_certifiability


if __name__ == "__main__":
    sample, actions = run_certifiability(Path("/root/autodl-tmp/hsc_tta_eeg"))
    print(f"wrote {len(sample)} sample-size and {len(actions)} action-count rows")
