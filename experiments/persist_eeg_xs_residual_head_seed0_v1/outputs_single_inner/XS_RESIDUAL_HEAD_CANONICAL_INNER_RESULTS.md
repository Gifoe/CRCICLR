# LiteBN-XS residual head seed0 — canonical-inner result

- Exact original LiteBN-XS checkpoint per task/fold; all XS parameters frozen.
- Only zero-initialized residual dW/db trained on precomputed XS embeddings.
- 4 tasks × 5 folds × 1 frozen canonical inner split; no outer/test.

| Task | Mean delta vs XS (pp) | Positive | Zero | Negative |
|---|---:|---:|---:|---:|
| OpenBMI_ERP | +0.0000 | 0/5 | 5/5 | 0/5 |
| OpenBMI_MI | +0.2333 | 4/5 | 1/5 | 0/5 |
| OpenBMI_SSVEP | +0.0333 | 1/5 | 4/5 | 0/5 |
| WBCIC_MI | +0.2608 | 3/5 | 2/5 | 0/5 |

OpenBMI ERP mean delta vs XS: **+0.0000 pp**

Selected epochs: OpenBMI_ERP/f0=0; OpenBMI_ERP/f1=0; OpenBMI_ERP/f2=0; OpenBMI_ERP/f3=0; OpenBMI_ERP/f4=0; OpenBMI_MI/f0=1; OpenBMI_MI/f1=1; OpenBMI_MI/f2=3; OpenBMI_MI/f3=2; OpenBMI_MI/f4=0; OpenBMI_SSVEP/f0=0; OpenBMI_SSVEP/f1=1; OpenBMI_SSVEP/f2=0; OpenBMI_SSVEP/f3=0; OpenBMI_SSVEP/f4=0; WBCIC_MI/f0=1; WBCIC_MI/f1=0; WBCIC_MI/f2=4; WBCIC_MI/f3=0; WBCIC_MI/f4=3

`FINAL_HELDOUT_ACCESSED = NO`
