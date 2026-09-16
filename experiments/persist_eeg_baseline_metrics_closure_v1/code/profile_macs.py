"""Unified batch-1 MAC audit using PyTorch ATen operator profiling.

PyTorch reports FLOPs for Conv/Matmul as two operations per multiply-accumulate.
Toy Conv/Linear/BMM checks must pass before any model number is reported.
Unsupported major operators produce a blank MAC value, never an undersized one.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


EXP = Path(__file__).resolve().parents[1]
WRAPPER = EXP / "code" / "run_frozen_sessions.py"
OUT = EXP / "outputs" / "MACS_AUDIT.csv"
MODELS = ("EEGNet", "CBraMod", "TeCh", "ModernTCN", "Medformer", "LiteBN")
TASKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
MAJOR_UNSUPPORTED = ("scaled_dot_product_attention", "_fft_", "fft_")
MATMUL_ATEN = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm",
               "aten::mv", "aten::addmv", "aten::matmul", "aten::dot"}


def import_wrapper():
    spec = importlib.util.spec_from_file_location("closure_macs_wrapper", WRAPPER)
    if spec is None or spec.loader is None:
        raise ImportError(WRAPPER)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def flops_of(function, *args) -> tuple[int, list[dict]]:
    conv_macs: list[int] = []
    hooks = []
    conv_types = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                  torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)
    if isinstance(function, torch.nn.Module):
        def count_conv(module, inputs, output):
            kernel = int(np.prod(module.kernel_size))
            if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                macs = int(inputs[0].numel() * (module.out_channels // module.groups) * kernel)
            else:
                macs = int(output.numel() * (module.in_channels // module.groups) * kernel)
            conv_macs.append(macs)
        for module in function.modules():
            if isinstance(module, conv_types):
                hooks.append(module.register_forward_hook(count_conv))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True, with_flops=True) as profile:
        with torch.inference_mode():
            function(*args)
    for hook in hooks:
        hook.remove()
    events = [{"name": event.key, "flops": int(event.flops), "count": int(event.count)}
              for event in profile.key_averages()]
    # ATen Conv FLOPs are inconsistent across 1D/2D backends in this PyTorch
    # build. Count convolution modules by exact output/input geometry once.
    nonconv_flops = sum(row["flops"] for row in events if row["name"] in MATMUL_ATEN)
    events = [{**row, "flops": row["flops"] if row["name"] in MATMUL_ATEN else 0}
              for row in events]
    events.append({"name": "closure::module_convolution", "flops": 2 * sum(conv_macs),
                   "count": len(conv_macs)})
    return nonconv_flops + 2 * sum(conv_macs), events


def toy_validation() -> None:
    linear = torch.nn.Linear(3, 4)
    conv = torch.nn.Conv1d(2, 4, 3, bias=False)
    a, b = torch.ones(1, 2, 3), torch.ones(1, 3, 4)
    tests = (
        ("linear", linear, (torch.ones(2, 3),), 24),
        ("conv1d", conv, (torch.ones(1, 2, 6),), 96),
        ("conv2d", torch.nn.Conv2d(2, 4, 3, bias=False), (torch.ones(1, 2, 6, 6),), 1152),
        ("bmm", torch.bmm, (a, b), 24),
    )
    for name, function, args, expected_macs in tests:
        flops, events = flops_of(function, *args)
        if flops != 2 * expected_macs:
            raise RuntimeError(f"ATen FLOP convention failed {name}: flops={flops}, expected={2*expected_macs}, events={events}")
    print("MAC_TOY_CONVENTION_PASS linear=24 conv1d=96 conv2d=1152 bmm=24", flush=True)


def profile_one(csgd, row: dict) -> dict:
    model_name, task = row["Model"], row["Task"]
    c, t = int(row["channels"]), int(row["samples"])
    model = csgd.build_model(row, torch.device("cpu"))
    x = np.zeros((1, c, t), dtype=np.float32)
    x = csgd.resample(model_name, x)
    tensor = torch.from_numpy(x)
    try:
        flops, events = flops_of(model, tensor)
        event_names = {event["name"] for event in events}
        unsupported = sorted(name for name in event_names if any(part in name for part in MAJOR_UNSUPPORTED))
        has_bmm = any("bmm" in name for name in event_names)
        if any("scaled_dot_product_attention" in name for name in unsupported) and not has_bmm:
            status = "UNRELIABLE_FUSED_ATTENTION_UNCOUNTED"
        elif any("fft" in name for name in unsupported):
            status = "UNRELIABLE_FFT_UNCOUNTED"
        elif flops <= 0:
            status = "UNRELIABLE_NO_COUNTED_OPERATORS"
        else:
            status = "COUNTED_ATEN_CONV_MATMUL"
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = int(row["trainable_parameters"])
        if n_trainable > n_params:
            raise RuntimeError(f"recorded trainable parameters exceed total: {model_name} {task}")
        macs = flops // 2 if status == "COUNTED_ATEN_CONV_MATMUL" else ""
        return {"model": model_name, "task": task, "input_shape": str(tuple(tensor.shape)),
                "parameters": n_params, "trainable_parameters": n_trainable,
                "MACs": macs, "MACs_M": macs / 1e6 if macs != "" else "",
                "MACs_G": macs / 1e9 if macs != "" else "",
                "profiler": f"torch.profiler/{torch.__version__}/CPU/with_flops",
                "counting_convention": "one multiply-add pair = 1 MAC; counted ATen Conv/Matmul FLOPs divided by 2 after toy verification; elementwise/norm/pool/softmax excluded",
                "unsupported_major_ops": ";".join(unsupported), "status": status,
                "counted_operator_events": ";".join(f"{event['name']}:{event['flops']}" for event in events if event["flops"]),
                "all_operator_names": ";".join(sorted(event_names))}
    finally:
        del model, tensor


def main() -> None:
    torch.set_num_threads(2)
    toy_validation()
    wrapper = import_wrapper()
    csgd = wrapper.source()
    lock = wrapper.read_lock(csgd)
    records = {(row["Model"], row["Task"]): row for row in lock["evaluation_checkpoints"]
               if int(row["fold"]) == 0 and int(row["seed"]) == 0}
    rows = []
    for model in MODELS:
        for task in TASKS:
            row = records.get((model, task))
            if row is None:
                rows.append({"model": model, "task": task, "input_shape": "", "parameters": "",
                             "trainable_parameters": "", "MACs": "", "MACs_M": "", "MACs_G": "",
                             "profiler": "", "counting_convention": "", "unsupported_major_ops": "",
                             "status": "MISSING_FROZEN_CHECKPOINT", "counted_operator_events": "", "all_operator_names": ""})
                continue
            try:
                result = profile_one(csgd, row)
            except Exception as error:
                result = {"model": model, "task": task, "input_shape": "", "parameters": row["trainable_parameters"],
                          "trainable_parameters": row["trainable_parameters"], "MACs": "", "MACs_M": "", "MACs_G": "",
                          "profiler": f"torch.profiler/{torch.__version__}/CPU/with_flops",
                          "counting_convention": "one multiply-add pair = 1 MAC", "unsupported_major_ops": str(error),
                          "status": "UNRELIABLE_PROFILE_ERROR", "counted_operator_events": "", "all_operator_names": ""}
            rows.append(result)
            print(f"MAC_PROFILE model={model} task={task} status={result['status']} macs={result['MACs']}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (EXP / "protocol" / "MAC_CONVENTION.json").write_text(json.dumps({
        "profiler": f"torch.profiler/{torch.__version__}/CPU/with_flops",
        "toy_validation": "PASS: linear 24, conv1d 96, conv2d 1152, bmm 24 MACs",
        "counting_convention": "1 multiply-accumulate pair = 1 MAC; convolution module geometry plus ATen non-convolution matmul FLOPs / 2",
        "batch_size": 1, "excluded": ["activation", "dropout", "pooling", "normalization", "softmax", "simple elementwise"],
        "unsupported_major_ops_policy": "mark blank and UNRELIABLE, never report underestimated MACs",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
