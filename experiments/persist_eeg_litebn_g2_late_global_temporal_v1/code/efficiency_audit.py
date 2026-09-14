#!/usr/bin/env python3
"""Deterministic one-model LiteBN-G2 inference efficiency audit."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

import train_litebn_g2 as train

REPO = Path(os.environ.get("LITEBN_G2_REPO", "/root/rivermind-data/CRCICLR_G2_WORK")).resolve()
OUT = REPO / "experiments/persist_eeg_litebn_g2_late_global_temporal_v1/outputs"


def count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _checkpoint = train.build_model("OpenBMI_SSVEP", 0, device)
    selected = REPO / "experiments/persist_eeg_litebn_g2_late_global_temporal_v1/runtime/checkpoints/openbmi_ssvep/fold0_seed0/selected_best.pt"
    if selected.is_file():
        model.load_state_dict(torch.load(selected, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model = model.to(device).eval(); value = torch.zeros((1, 62, train.base.TASKS["OpenBMI_SSVEP"]["samples"]), device=device)
    cpu_model, cpu_value = model.cpu().eval(), value.cpu()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as profile:
        with torch.inference_mode():
            cpu_model(cpu_value)
    flops = int(sum(event.flops or 0 for event in profile.key_averages()))
    model, value = cpu_model.to(device).eval(), cpu_value.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(20):
            model(value)
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter(); model(value)
        if device.type == "cuda": torch.cuda.synchronize()
    report = {"input_shape": list(value.shape), "inference_mode": "eval/no-grad/one deterministic model/no labels/no adaptation", "LiteBN_parameters": count(model.base), "global_block_parameters": count(model.global_block), "LiteBN_G2_parameters": count(model), "FLOPs_if_profiled": flops, "MACs_estimate": flops / 2.0 if flops else None, "single_forward_latency_ms": (time.perf_counter() - started) * 1000.0, "peak_inference_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}
    path = OUT / "EFFICIENCY_REPORT.json"; temporary = path.with_suffix(".json.part"); temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(temporary, path)
    print("LITEBN_G2_EFFICIENCY_COMPLETE", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
