# H1 XS representation-level residual bottleneck — canonical-inner result

- Exact original LiteBN-XS checkpoint per task/fold; all XS parameters and original classifier frozen.
- Trainable adapter only: `z_new = z + Up(GELU(Down(z)))`, 128→16→128.
- Four tasks × five folds × one canonical inner split; no outer/test access.
- LR `3e-05`, weight decay `0.0005`, beta `0.001`, max steps `128`.
- Epoch0 replay: **20/20 passed**. ERP clear-signal threshold: **0.10 pp**.

| Task | Mean ΔBA vs XS (pp) | Positive | Zero | Negative |
|---|---:|---:|---:|---:|
| OpenBMI_ERP | +0.0313 | 4/5 | 1/5 | 0/5 |
| OpenBMI_MI | +0.2000 | 4/5 | 1/5 | 0/5 |
| OpenBMI_SSVEP | +0.0333 | 1/5 | 4/5 | 0/5 |
| WBCIC_MI | +0.1808 | 3/5 | 2/5 | 0/5 |

OpenBMI ERP mean delta vs XS: **+0.0313 pp** (4/5 positive folds).

ERP per-fold and all-step trajectories are in `H1_REPRESENTATION_TRAJECTORY.csv`.

XS_LOCAL_CHECKPOINT_REPAIR_FAILED
`FINAL_HELDOUT_ACCESSED = NO`
