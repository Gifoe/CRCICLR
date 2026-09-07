"""Numerical-equivalence and one-epoch throughput benchmark.

The reference path mirrors the pre-optimization per-batch implementation.  It
uses source-only OpenBMI fold 0 metadata/data and never opens outcome subjects.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import argparse

import numpy as np
import torch

import run_geosr as g


def reference_train_epoch(model, cache, rows, mean, std, weights, optimizer, order, device):
    model.train()
    losses = []
    row_to_weight = {int(r): float(w) for r, w in zip(rows.tolist(), weights.tolist())}
    labels = cache.labels
    for start in range(0, len(order), g.BATCH_SIZE):
        part = order[start : start + g.BATCH_SIZE]
        xb = cache.tensor(part, mean, std, device).to(device, non_blocking=True)
        yb = torch.from_numpy(labels[part]).long().to(device, non_blocking=True)
        wb = torch.tensor([row_to_weight[int(r)] for r in part], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        ce = torch.nn.functional.cross_entropy(model(xb), yb, reduction="none")
        loss = (ce * wb).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), g.GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def sample_gpu(stop: threading.Event, values: list[tuple[float, float]]) -> None:
    while not stop.is_set():
        try:
            line = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            util, mem = line.split(",", 1)
            values.append((float(util), float(mem)))
        except Exception:
            pass
        time.sleep(0.2)


def run_one(kind, cache, rows, mean, std, weights, order, state, device):
    # Match the runner's deterministic per-training initialization so dropout
    # consumes the same RNG stream in both implementations.
    g.seed_everything(0)
    torch.manual_seed(g.stable_seed("benchmark-training-rng"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(g.stable_seed("benchmark-training-rng"))
    model = g.make_model(cache, device)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=g.LR, weight_decay=g.WEIGHT_DECAY)
    if device.type == "cuda":
        torch.cuda.synchronize()
    values: list[tuple[float, float]] = []
    stop = threading.Event()
    thread = threading.Thread(target=sample_gpu, args=(stop, values), daemon=True)
    thread.start()
    t0 = time.perf_counter()
    if kind == "reference":
        loss = reference_train_epoch(model, cache, rows, mean, std, weights, optimizer, order, device)
    else:
        loss = g.train_epoch(model, cache, rows, mean, std, weights, optimizer, order, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    stop.set()
    thread.join(timeout=2)
    util = float(np.mean([v[0] for v in values])) if values else float("nan")
    mem = float(max((v[1] for v in values), default=0.0))
    digest = g.state_hash(model.state_dict())
    return {"loss": loss, "sec": elapsed, "gpu_util_mean": util, "vram_peak_mib": mem, "state_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="OpenBMI")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roles, _, _ = g.ap.load_roles(args.dataset)
    role = roles[args.fold]
    source = g.subj_sort(role["model_fit"])
    source_all = g.subj_sort(set(source) | set(role["discovery"]))
    cache = g.FoldCache(args.dataset, source_all, 0, args.fold)
    rows = cache.rows(source, g.sessions_for(args.dataset))
    mean, std = cache.normalizer(rows)
    cache.normalize(mean, std)
    weights = np.ones(len(rows), dtype=np.float32)
    order = g.order_for(rows, args.dataset, args.fold, 0, "benchmark", "benchmark", 1)
    state, _, _ = g.initial_state(cache, args.dataset, args.fold, 0, "benchmark")
    # One warm-up of each path removes first-kernel/JIT startup from the report.
    run_one("reference", cache, rows, mean, std, weights, order, state, device)
    run_one("optimized", cache, rows, mean, std, weights, order, state, device)
    reference = run_one("reference", cache, rows, mean, std, weights, order, state, device)
    optimized = run_one("optimized", cache, rows, mean, std, weights, order, state, device)
    print("device", device)
    print("reference", reference)
    print("optimized", optimized)
    print("exact_loss", reference["loss"] == optimized["loss"])
    print("exact_state", reference["state_sha256"] == optimized["state_sha256"])


if __name__ == "__main__":
    main()
