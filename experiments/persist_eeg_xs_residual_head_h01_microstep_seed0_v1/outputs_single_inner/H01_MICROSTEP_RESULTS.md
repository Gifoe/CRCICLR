# H0.1 micro-step residual head — canonical-inner result

- Exact original LiteBN-XS checkpoint per task/fold; all XS parameters frozen.
- Only zero-initialized residual dW/db; XS embeddings/logits precomputed once per fold.
- Four tasks × five folds × one canonical inner split; no outer/test access.
- Candidate learning rates: `[1e-05, 3e-06]`; evaluation steps: `[0, 1, 2, 4, 8, 16, 32, 64]`.
- Selected global LR: **1e-05** (ERP-first inner-only rule).

## Selected global-LR summary

| Task | Mean delta vs XS (pp) | Positive | Zero | Negative |
|---|---:|---:|---:|---:|
| OpenBMI_ERP | +0.0192 | 4/5 | 1/5 | 0/5 |
| OpenBMI_MI | +0.1667 | 4/5 | 1/5 | 0/5 |
| OpenBMI_SSVEP | +0.0333 | 1/5 | 4/5 | 0/5 |
| WBCIC_MI | +0.0600 | 3/5 | 2/5 | 0/5 |

OpenBMI ERP mean delta vs XS: **+0.0192 pp**
Operational clear-signal threshold: **0.10 pp**; smaller means are treated as essentially zero.

## ERP trajectories

See `H01_MICROSTEP_TRAJECTORY.csv` for all five folds, both learning rates, and steps 0/1/2/4/8/16/32/64.

RESIDUAL_HEAD_ERP_SIGNAL_NOT_FOUND
`FINAL_HELDOUT_ACCESSED = NO`
