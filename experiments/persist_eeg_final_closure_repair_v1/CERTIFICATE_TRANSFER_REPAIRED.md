# Certificate transfer — repaired hierarchical inference

The earlier join treated PUD basis column `j` as certificate direction `j`. The repaired join reads `basis_PUD`, maps each selected basis column to its actual certified coordinate, and corrected 680 repeated direction-subject source-score cells.

Direction consequence: **DIRECTION_CONSEQUENCE_TRANSFER_SUPPORTED**.

| future consequence | mean | hierarchical 95% CI |
|---|---:|---:|
| future_BA_erasure_harm | 0.0283 | [0.0194, 0.0394] |
| future_CE_erasure_harm | 0.0568 | [0.0433, 0.0747] |
| future_D_finite | 0.7130 | [0.6272, 0.8213] |

Certificate-score transfer: **CERTIFICATE_SCORE_TRANSFER_POSITIVE**.

| source → future | Pearson [95% CI] | Spearman [95% CI] | mapping status |
|---|---:|---:|---|
| source_U_repaired → future_CE_erasure_harm | 0.938 [0.793, 0.947] | 0.836 [0.578, 0.883] | POSITIVE |
| source_D_finite_repaired → future_D_finite | 0.992 [0.979, 0.994] | 0.991 [0.963, 0.991] | POSITIVE |
| source_P_repaired → future_BA_erasure_harm | 0.664 [0.490, 0.745] | 0.733 [0.461, 0.801] | POSITIVE |

Mean future consequence and predictive transfer of source certificate scores are separate hypotheses. The former cannot be used to label the latter as supported.
