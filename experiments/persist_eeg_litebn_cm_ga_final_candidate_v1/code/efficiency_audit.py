#!/usr/bin/env python3
"""Architecture efficiency at the audited identity checkpoint."""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
os.environ["CM_GA_REPO"] = str(REPO)

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    assert spec.loader is not None; spec.loader.exec_module(module)
    return module

audit = load("cm_ga_efficiency_source", EXP / "code/gradient_transfer_audit.py")
base, models, runner = audit.base, audit.models, audit.runner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for task in base.TASK_ORDER:
    model, _ = audit.make_model(task, 0, device)
    model.eval()
    parameters_base = sum(p.numel() for p in model.base.parameters())
    parameters_c = model.lambda_channel.numel() + sum(p.numel() for p in model.channel_mlp.parameters())
    parameters_m = sum(p.numel() for p in model.mixers.parameters())
    shape = (1, base.TASKS[task]["channels"], base.TASKS[task]["samples"])
    channels, samples = shape[1], shape[2]
    length_4, length_8, length_16 = samples // 4, samples // 8, samples // 16
    # Conv/linear multiplication-accumulations; pooling, normalization,
    # activations, square roots, and scalar residual operations are omitted.
    base_macs = (
        channels * samples * 8 * (15 + 63 + 127)
        + 3 * 16 * samples * channels
        + 48 * length_4 * 15 + 64 * 48 * length_4
        + 64 * length_8 * 31 + 64 * 64 * length_8
        + 512 * 64 + 64 * base.TASKS[task]["classes"]
    )
    c_macs = channels * (2 * 8 + 8)
    mixer_macs = 3 * (64 * length_16 * 9 + (64 * 128 + 128 * 64) * length_16)
    macs = base_macs + c_macs + mixer_macs
    value = torch.zeros(shape, device=device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(20): model(value)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(100): model(value)
        if device.type == "cuda": torch.cuda.synchronize()
    rows.append({
        "task": task, "LiteBN_parameters": parameters_base, "C_parameters": parameters_c,
        "M_parameters": parameters_m, "total_CM_GA_parameters": parameters_base + parameters_c + parameters_m,
        "trainable_CM_parameters": parameters_c + parameters_m, "approximate_MACs": int(macs),
        "batch1_latency_ms": (time.perf_counter() - started) * 10.0,
        "peak_inference_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "inference_gradient": False, "target_adaptation": False, "router": False,
    })
    del model, value
    if device.type == "cuda": torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(EXP / "outputs/MODEL_EFFICIENCY.csv", index=False)
print("EFFICIENCY_AUDIT_COMPLETE")
