# LiteBN-BPXS first-screen result

Only OpenBMI ERP and WBCIC MI were run, five canonical inner folds each. No outer-development or final-heldout rows were accessed.

| Task | LiteBN BA | Original XS BA | BPXS BA | Δ vs LiteBN pp | Δ vs XS pp | positive folds vs LiteBN |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_ERP | 0.8376 | 0.8320 | 0.8299 | -0.761 | -0.206 | 0/5 |
| WBCIC_MI | 0.7957 | 0.7982 | 0.7948 | -0.085 | -0.335 | 3/5 |

## Gate diagnostics

| Task | mean gate-on BA | mean gate-off BA | mean BA(on-off) |
|---|---:|---:|---:|
| OpenBMI_ERP | 0.8299 | 0.8246 | +0.0053 |
| WBCIC_MI | 0.7948 | 0.7932 | +0.0016 |

Terminal: **BPXS_OBJECTIVE_NO_USEFUL_SIGNAL**

`SEED = 0`
`CANONICAL_INNER_SPLITS_ONLY = YES`
`OUTER_DEVELOPMENT_ACCESSED = NO`
`FINAL_HELDOUT_ACCESSED = NO`
