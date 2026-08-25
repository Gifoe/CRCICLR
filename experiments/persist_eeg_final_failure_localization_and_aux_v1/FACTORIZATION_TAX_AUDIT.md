# Factorization and supervision tax audit

The paired subject decomposition uses the frozen five-fold subject table; no model was retrained in Phase A.

| component | mean Δ BA | 95% CI | harmed/improved |
|---|---:|---|---:|
| T_factorization_Dual_minus_Vanilla | -0.0085 | [-0.0202, 0.0022] | 18/19 |
| T_PUD_PUD_minus_Dual | -0.0212 | [-0.0296, -0.0132] | 31/6 |
| T_adaptation_Adapted_minus_PUD | 0.0074 | [0.0032, 0.0118] | 10/26 |
| PUD_minus_Strong | -0.0350 | [-0.0472, -0.0237] | 33/5 |

Fold and seed consistency is retained in canonical_run_table.csv. The frozen totals are Vanilla 0.7861667, Dual 0.7776667, PUD source 0.7565000, and PUD adapted 0.7639167.
