#!/usr/bin/env python3
"""Fail-closed exposed-benchmark entry point."""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
decision = json.loads((EXP / "outputs/GRADIENT_TRANSFER_DECISION.json").read_text())
if not decision.get("stage1_authorized"):
    print("BENCHMARK_EVALUATION_NOT_RUN_DUE_STAGE0_FAIL")
    raise SystemExit(0)
raise RuntimeError("Benchmark evaluation requires completed frozen Stage-1 checkpoints")

