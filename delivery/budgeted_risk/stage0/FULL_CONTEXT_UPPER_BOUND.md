# Stage-0A full-context upper bound

Metrics are subject-level after averaging the five source-head seeds. The gate was applied separately to each dataset.

| dataset   |   n_subjects |   spearman |    mae |   constant_mae |   relative_mae_improvement |   predicted_vs_true_slope |   underestimation_rate |   paired_improvement_mean |   paired_ci_low |   paired_ci_high | full_context_pass   |
|:----------|-------------:|-----------:|-------:|---------------:|---------------------------:|--------------------------:|-----------------------:|--------------------------:|----------------:|-----------------:|:--------------------|
| eegmmidb  |           65 |     0.6127 | 1.0468 |        10.5923 |                     0.9012 |                    0.8497 |                 0.6615 |                    9.5455 |          8.6927 |          10.3446 | True                |
| hmc       |           90 |     0.5921 | 2.2173 |         2.6600 |                     0.1664 |                    0.4560 |                 0.5556 |                    0.4427 |          0.0415 |           0.8509 | True                |

Decision: **FULL_CONTEXT_GO**.
