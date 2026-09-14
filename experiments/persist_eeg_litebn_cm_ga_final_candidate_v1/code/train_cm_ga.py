#!/usr/bin/env python3
"""Fail-closed Stage-1 entry point."""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
decision = json.loads((EXP / "outputs/GRADIENT_TRANSFER_DECISION.json").read_text())
if not decision.get("stage1_authorized"):
    print("CM_GA_GRADIENT_TRANSFER_NOT_SUPPORTED: Stage 1 is forbidden")
    raise SystemExit(0)
raise RuntimeError("Stage 1 authorization exists but this stopped branch must be reviewed before execution")

