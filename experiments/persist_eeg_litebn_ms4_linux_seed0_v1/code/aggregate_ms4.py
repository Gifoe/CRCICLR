"""Read-only integrity summary for a completed LiteBN-MS4 run."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


repo = Path(os.environ.get("MS4_REPO", "/root/rivermind-data/CRCICLR_SC_CONTROLLED_V2_WORK"))
out = repo / "experiments" / "persist_eeg_litebn_ms4_linux_seed0_v1" / "outputs"
required = [
    "MS4_INITIALIZATION_AUDIT.csv", "MS4_C0_PREFIX_EQUIVALENCE.csv",
    "MS4_K2_REDUCTION_AUDIT.csv", "MS4_BN_AUTOGRAD_AUDIT.csv", "MANIFEST_AUDIT.csv",
]
for name in required:
    frame = pd.read_csv(out / name)
    if frame.empty or not (frame.status == "PASS").all():
        raise RuntimeError(f"invalid audit: {name}")
decision = json.loads((out / "FINAL_MS4_DECISION.json").read_text())
print(decision["status"])
