# LiteBN-XRA seed0 — G2 WBCIC canonical-inner result

- Regime: `XRA_G2_ADAPTER_SCALE_LAST_MIXER` only.
- Scope: WBCIC MI × 5 folds × 1 frozen canonical inner train/validation pair.
- Early stopping: max 20 epochs, patience 5, selection metric inner-val BA.
- Epoch 0 is exact LiteBN-X; if no improvement, epoch 0 is selected.
- G1/G0, OpenBMI and outer-development were not run.

| Task | Mean delta vs X (pp) | Positive | Zero | Negative |
|---|---:|---:|---:|---:|
| WBCIC_MI | +0.0996 | 3/5 | 2/5 | 0/5 |

Selected epoch per fold: 0, 5, 4, 4, 0

This run stops here for inspection; no Stage-A lock is issued.

`FINAL_HELDOUT_ACCESSED = NO`
