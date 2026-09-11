# LiteBN-DRA-XS seed0 result

Terminal: **DRA_ERP_NO_SIGNAL**

Stages run: OpenBMI_ERP.
Stage 2: ERP clear-signal rule not met.

| Task | LiteBN BA | XS BA | DRA BA | Δ vs XS (pp) | Δ vs LiteBN (pp) | positive folds vs XS |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI ERP | 0.8376 | 0.8320 | 0.8316 | -0.040 | -0.595 | 3/5 |

## ERP admission timeline

The trajectory CSV retains epochs 1-5 (residual bypassed), 6-9 (0-to-1 ramp), and 10+ (fully admitted) for every ERP fold. The report does not make a causal claim from one seed.

## Diagnostics

Gate-off is diagnostic only and was not used for selection. `gateoff_delta_BA` is normal selected-checkpoint BA minus scale-0 BA.

`SEED = 0`
`CANONICAL_INNER_SPLITS_ONLY = YES`
`NEW_INNER_SPLITS_CREATED = NO`
`OUTER_DEVELOPMENT_ACCESSED = NO`
`FINAL_HELDOUT_ACCESSED = NO`
