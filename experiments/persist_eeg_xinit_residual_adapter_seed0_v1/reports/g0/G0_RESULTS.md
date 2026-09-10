# LiteBN-XRA seed0 — G0 adapter-only result

- Regime: `XRA_G0_ADAPTER_ONLY`
- Scope: 4 tasks × 5 frozen outer folds × 3 subject-disjoint inner splits; inner-only selection.
- Source model: frozen LiteBN-X checkpoint; residual channel adapter initialized at exact zero effect.
- Protocol: tau=0.5; epoch 0 is exact X replay; no final held-out/test data accessed.
- Epoch-0 replay: 20/20 folds passed; max logit difference 0; prediction mismatches 0.
- G0 inner gate: **FAIL / continue to G1** (`XRA_G0_ADAPTER_ONLY_INNER_SUMMARY worst=+0.0000pp pass=False`).

## Task-level robust gain

Values are percentage points (pp), using `median(delta_BA) - 0.5 * std(delta_BA)` and the frozen checkpoint rule.

| Task | Median robust gain (pp) | Mean robust gain (pp) | Positive folds |
|---|---:|---:|---:|
| OpenBMI_ERP | +0.0062 | +0.0514 | 3/5 |
| OpenBMI_MI | +0.0000 | +0.0000 | 0/5 |
| OpenBMI_SSVEP | +0.0000 | +0.0000 | 0/5 |
| WBCIC_MI | +0.0833 | +0.1039 | 3/5 |

## Gate summary

- `worst_task_robust_gain_pp`: 0.0
- `nonnegative_tasks`: 4/4
- `positive_tasks`: 2/4
- `equal_task_mean_robust_gain_pp`: 0.03882177792493973
- `positive_fold_robust_gain`: 6/20
- `passes_all_tasks`: `False`

G0 does not qualify for the Stage-A lock. The runner advanced to G1; this snapshot is G0-only and is not an outer-development result.

`FINAL_HELDOUT_ACCESSED = NO`
