# P4-SI Development Summary

- Version: `SI_V1`
- Development runs: `6` (folds 0,1,2 × seeds 0,1)
- Decision: `P4_SI_REPRESENTATION_ONLY`
- Outer-test used: `false`

## Task deltas

| Task | Mean SI−reference BA | Std | Min | Max |
|---|---:|---:|---:|---:|
| MI | -0.0015 | 0.0115 | -0.0217 | +0.0111 |
| ERP | +0.0096 | 0.0101 | -0.0032 | +0.0226 |
| SSVEP | -0.0018 | 0.0060 | -0.0133 | +0.0044 |

## Pre-registered checks

- `gate_A_mean_task_delta_at_least_minus_1pp`: `True`
- `gate_A_no_task_catastrophic_in_two_or_more_runs`: `True`
- `gate_B_MI_relevance_mean_at_least_0_005`: `True`
- `gate_B_MI_relevance_positive_in_at_least_4_of_6`: `True`
- `gate_B_MI_protected_harm_exceeds_random_mean`: `True`
- `gate_C_macro_nuisance_reduction_mean_at_least_0_02`: `True`
- `gate_C_macro_nuisance_reduction_positive_in_at_least_4_of_6`: `True`
- `gate_D_MI_protected_retention_mean_at_least_0_50`: `True`
- `gate_D_MI_protected_retention_positive_in_at_least_4_of_6`: `True`

## Interpretation

The decision is based only on train/validation development artifacts. No outer-test signal, label, embedding, or metric was accessed.
Protected/nuisance intervention sets and independent event probes are reported per run; the random intervention is a same-rank control.
