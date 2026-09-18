# Protected-Subspace Preservation Intervention

Frozen heldout intervention evaluation. OpenBMI is an internal-heldout cohort and WBCIC is the frozen true-outer cohort; neither is described as an untouched prospective confirmation.

The primary comparison is **ProtectedPreserve − mean(RandomPreserve R0/R1/R2)**. Random arms are averaged within each biological subject before paired bootstrap; folds, seeds, and random arms are not treated as independent biological samples.

## Task summary

| task | condition | estimable | coverage | biological_subjects | P_drift | BA_pp | BA_ci95_low_pp | BA_ci95_high_pp | macro_F1_pp | macro_F1_ci95_low_pp | macro_F1_ci95_high_pp | WS_BA_pp | WS_BA_ci95_low_pp | WS_BA_ci95_high_pp | between_random_BA_sd_pp | between_random_BA_range_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenBMI_MI | CE_REFERENCE | yes | 15/15 all cells | 14 | 0.0000 | 70.6143 | 65.3333 | 76.7192 | 69.8047 | 64.2806 | 76.1451 | 67.0762 | 63.3094 | 71.5239 |  |  |
| OpenBMI_MI | PRD_ONLY | yes | 15/15 all cells | 14 | 0.4101 | 71.8333 | 66.5095 | 77.8476 | 71.0038 | 65.4331 | 77.1692 | 68.3238 | 64.3952 | 72.8715 |  |  |
| OpenBMI_MI | RANDOM_PRESERVE_MEAN | yes | 15/15 estimable target cells | 14 | 0.4723 | 71.6556 | 66.2825 | 77.6921 | 70.7974 | 65.0403 | 77.1191 | 68.2968 | 64.2460 | 73.0350 | 0.2001 | 0.4714 |
| OpenBMI_MI | PROTECTED_PRESERVE | yes | 15/15 estimable target cells | 14 | 0.0935 | 71.6762 | 66.2856 | 77.5571 | 70.8352 | 65.1333 | 77.1284 | 68.3238 | 64.2048 | 72.8431 |  |  |
| WBCIC_MI | CE_REFERENCE | yes | 15/15 all cells | 10 | 0.0000 | 76.4000 | 68.3930 | 83.3401 | 75.9183 | 67.7536 | 83.1142 | 68.8367 | 59.5098 | 78.3135 |  |  |
| WBCIC_MI | PRD_ONLY | yes | 15/15 all cells | 10 | 56.3810 | 67.5000 | 62.2200 | 72.1167 | 60.7009 | 55.2930 | 65.4575 | 62.8700 | 57.0600 | 68.7933 |  |  |
| WBCIC_MI | RANDOM_PRESERVE_MEAN | yes | 15/15 estimable target cells | 10 | 36.8049 | 67.6044 | 62.3766 | 72.1578 | 60.9542 | 55.6391 | 65.6665 | 63.0156 | 57.0778 | 69.1278 | 0.1391 | 0.3300 |
| WBCIC_MI | PROTECTED_PRESERVE | yes | 15/15 estimable target cells | 10 | 0.6230 | 67.5800 | 62.3800 | 72.2133 | 60.7810 | 55.4544 | 65.4563 | 62.9133 | 57.0667 | 68.9467 |  |  |

## Paired biological-subject effects (percentage points)

| task | comparison | primary | biological_subjects | delta_BA_pp | delta_BA_ci95_low_pp | delta_BA_ci95_high_pp | delta_macro_F1_pp | delta_macro_F1_ci95_low_pp | delta_macro_F1_ci95_high_pp | delta_WS_BA_pp | delta_WS_BA_ci95_low_pp | delta_WS_BA_ci95_high_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenBMI_MI | PROTECTED_PRESERVE_MINUS_RANDOM_PRESERVE | yes | 14 | 0.0206 | -0.0984 | 0.1333 | 0.0378 | -0.0866 | 0.1569 | 0.0270 | -0.1302 | 0.1683 |
| OpenBMI_MI | PROTECTED_PRESERVE_MINUS_PRD_ONLY | no | 14 | -0.1571 | -0.2810 | -0.0333 | -0.1686 | -0.3371 | -0.0109 | 0.0000 | -0.1381 | 0.1524 |
| OpenBMI_MI | RANDOM_PRESERVE_MINUS_PRD_ONLY | no | 14 | -0.1778 | -0.3190 | -0.0159 | -0.2064 | -0.3945 | -0.0152 | -0.0270 | -0.2190 | 0.1794 |
| OpenBMI_MI | PRD_ONLY_MINUS_CE_REFERENCE | no | 14 | 1.2190 | 0.6143 | 1.8714 | 1.1991 | 0.4827 | 1.9561 | 1.2476 | 0.6000 | 1.8905 |
| OpenBMI_MI | PROTECTED_PRESERVE_MINUS_CE_REFERENCE | no | 14 | 1.0619 | 0.4476 | 1.7333 | 1.0305 | 0.3194 | 1.7744 | 1.2476 | 0.6143 | 1.8905 |
| WBCIC_MI | PROTECTED_PRESERVE_MINUS_RANDOM_PRESERVE | yes | 10 | -0.0244 | -0.1022 | 0.0533 | -0.1732 | -0.2797 | -0.0645 | -0.1022 | -0.2167 | 0.0067 |
| WBCIC_MI | PROTECTED_PRESERVE_MINUS_PRD_ONLY | no | 10 | 0.0800 | -0.0567 | 0.2300 | 0.0801 | -0.0485 | 0.2127 | 0.0433 | -0.1567 | 0.2467 |
| WBCIC_MI | RANDOM_PRESERVE_MINUS_PRD_ONLY | no | 10 | 0.1044 | -0.0444 | 0.2511 | 0.2533 | 0.0931 | 0.4078 | 0.1456 | -0.0700 | 0.3478 |
| WBCIC_MI | PRD_ONLY_MINUS_CE_REFERENCE | no | 10 | -8.9000 | -11.3201 | -6.2800 | -15.2174 | -17.7705 | -12.3826 | -5.9667 | -9.5467 | -2.4133 |
| WBCIC_MI | PROTECTED_PRESERVE_MINUS_CE_REFERENCE | no | 10 | -8.8200 | -11.3101 | -6.0700 | -15.1373 | -17.7765 | -12.2439 | -5.9233 | -9.5800 | -2.4133 |

## Target coverage

| task | total_cells | estimable_cells | protected_rank_mean | protected_rank_min | protected_rank_max | protected_assignment_coverage |
| --- | --- | --- | --- | --- | --- | --- |
| OpenBMI_MI | 15 | 15 | 5.4667 | 3 | 8 | 15/15 |
| WBCIC_MI | 15 | 15 | 3.2667 | 1 | 4 | 15/15 |

## Representation-drift sanity

| task | check | mean_delta | all_lower | cells |
| --- | --- | --- | --- | --- |
| OpenBMI_MI | PROTECTED_PRESERVE P drift minus PRD_ONLY P drift | -0.3166 | yes | 15 |
| OpenBMI_MI | RANDOM_PRESERVE_R0 target drift minus PRD-only same R0 | -0.3223 | yes | 15 |
| OpenBMI_MI | RANDOM_PRESERVE_R1 target drift minus PRD-only same R1 | -0.3471 | yes | 15 |
| OpenBMI_MI | RANDOM_PRESERVE_R2 target drift minus PRD-only same R2 | -0.3411 | yes | 15 |
| WBCIC_MI | PROTECTED_PRESERVE P drift minus PRD_ONLY P drift | -55.7580 | no | 15 |
| WBCIC_MI | RANDOM_PRESERVE_R0 target drift minus PRD-only same R0 | -22.5326 | no | 15 |
| WBCIC_MI | RANDOM_PRESERVE_R1 target drift minus PRD-only same R1 | -43.4601 | yes | 15 |
| WBCIC_MI | RANDOM_PRESERVE_R2 target drift minus PRD-only same R2 | -15.3211 | no | 15 |

## Scope

A lower protected-coordinate drift validates the intervention manipulation only. Preferential actionability is supported only when the primary heldout paired effect favors ProtectedPreserve with a 95% CI lower bound above zero. ProtectedPreserve is not required to exceed the frozen CE reference; that would answer a different question.

## Completion

`COMPLETE`
