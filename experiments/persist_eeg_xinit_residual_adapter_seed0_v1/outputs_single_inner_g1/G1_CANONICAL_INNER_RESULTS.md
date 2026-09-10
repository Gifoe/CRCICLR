# LiteBN-XRA seed0 — G1 canonical-inner result

- Regime: `XRA_G1_ADAPTER_SCALE` only.
- Runs: 4 tasks × 5 folds × 1 frozen canonical inner train/validation pair = 20 runs.
- Early stopping: max 20 epochs, patience 5, selection metric inner-val BA.
- Epoch 0 is exact LiteBN-X; if no improvement, epoch 0 is selected.
- G2 and outer-development were not run.

| Task | Mean delta vs X (pp) | Positive | Zero | Negative |
|---|---:|---:|---:|---:|
| OpenBMI_ERP | +0.0778 | 3/5 | 2/5 | 0/5 |
| OpenBMI_MI | +0.0000 | 0/5 | 5/5 | 0/5 |
| OpenBMI_SSVEP | +0.1000 | 1/5 | 4/5 | 0/5 |
| WBCIC_MI | +0.0600 | 3/5 | 2/5 | 0/5 |

- WBCIC MI mean delta: **+0.0600 pp**
- OpenBMI non-negative target: `True`
- WBCIC positive target: `True`
- This run stops here for inspection; no Stage-A lock is issued.

`FINAL_HELDOUT_ACCESSED = NO`
