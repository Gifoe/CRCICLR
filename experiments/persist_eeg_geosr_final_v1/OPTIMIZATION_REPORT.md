# Seed-0 execution optimization record

The following measurements were taken on the authorized RTX 5090 server with
the canonical EEGNet, float32, deterministic cuDNN, batch size 64, and the
unchanged frozen hyperparameters.  The reference path is the pre-optimization
per-batch weight/label staging; the optimized path uses memoized row indices,
GPU-resident labels/weights, and the existing normalized CUDA view.

| Benchmark | Reference | Optimized |
|---|---:|---:|
| OpenBMI fold 0, one epoch (sec) | 24.889 | 24.969 |
| OpenBMI fold 0, GPU utilization (%) | 51.03 | 51.50 |
| WBCIC fold 0, one epoch (sec) | 68.606 | 68.733 |
| WBCIC fold 0, GPU utilization (%) | 51.17 | 51.07 |
| Peak VRAM (MiB) | 10,353 | 10,353 |

Both benchmarks produced identical loss and identical complete model
`state_sha256` (bitwise equality).  The per-batch staging optimization is
therefore retained for lower Python/transfer overhead, although the canonical
deterministic convolution dominates elapsed time.

Two independent workers (one OpenBMI and one WBCIC fold) were also measured:
the wall time for the four one-epoch reference/optimized passes per worker was
327.035 sec versus about 375 sec when run sequentially, with a 27,942 MiB
peak and roughly 76% aggregate GPU utilization.  The orchestrator therefore
never launches more than this bounded pair and aborts above 31,000 MiB.

The explicit-padding convolution rewrite was tested and rejected: it changed
CUDA outputs (maximum absolute logit difference about 3.98e-4) and gradients.
No AMP, TF32, batch-size, epoch, fold, seed, inner-cross-fit, descriptor-cap,
weight-formula, or data-split change was made.

The decision run writes per-epoch timings, teacher selection/fit durations,
checkpoint/cache hit flags, GPU samples, and final fold-level metrics under
the ignored runtime tree.  These measurements are provenance only and do not
alter the scientific gates.
