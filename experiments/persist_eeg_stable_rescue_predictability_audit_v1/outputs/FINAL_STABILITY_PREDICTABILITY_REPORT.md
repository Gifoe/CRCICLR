# Stable rescue and label-free predictability audit

EXPERIMENT_TYPE = DEVELOPMENT_ANALYSIS_ONLY
NEW_EEG_MODEL_TRAINED = NO
DIAGNOSTIC_LINEAR_MODEL_FIT = YES
FINAL_HELDOUT_ACCESSED = NO
INTERNAL_HELDOUT_ACCESSED = NO
DEVELOPMENT_OUTER_ONLY = YES

## ERP REPLAY STATUS

number of previously failed cells = 5

repaired cells = 5
unresolved cells = 0
X seed0 valid folds / 5 = 5/5
XS seed0 valid folds / 5 = 5/5
XS seed1 valid folds / 5 = 5/5
XS seed2 valid folds / 5 = 5/5

ERP_FULLY_INCLUDED

The repair restored the historically plausible PyTorch default numerical semantics: cuDNN TF32 enabled, matrix-multiplication TF32 disabled. No checkpoint, weight, normalizer, split, label, preprocessing path, or historical metric was changed.

## XS stability

| Task | B0-wrong trials | Any rescue % | Majority rescue % | Unanimous rescue % | Majority given any rescue % | Mean rescue-set Jaccard | Any harm % | Majority harm % | Unanimous harm % | Mean harm-set Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | 834 | 65.35 | 39.45 | 17.87 | 60.37 | 0.4409 | 24.29 | 10.36 | 3.47 | 0.2939 |
| OpenBMI_ERP | 10839 | 63.36 | 41.60 | 21.75 | 65.65 | 0.5055 | 10.57 | 3.48 | 0.99 | 0.2224 |
| OpenBMI_SSVEP | 432 | 73.84 | 56.48 | 34.26 | 76.49 | 0.6116 | 7.74 | 2.80 | 0.92 | 0.2527 |
| WBCIC_MI | 1320 | 61.21 | 36.29 | 16.14 | 59.28 | 0.4322 | 18.75 | 7.53 | 2.54 | 0.2802 |

Amounts (any/majority/unanimous rates) and exact trial-identity stability (Jaccard) are distinct; a low Jaccard does not negate complementary information.

## Predictability and selective policy

| Task | Candidate | Feature set | SAFE AUC | CLEAN AUC | Policy delta pp | 95% CI | Nonnegative folds | Worst fold pp | Switch rate | Intervention precision | Oracle pp | Fraction recovered | Interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenBMI_MI | X seed0 | CONF_ONLY | 0.5701 | 0.5701 | +0.350 | [-0.700, +1.350] | 4/5 | -2.125 | 8.55% | 0.5205 | 8.975 | 0.039 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_MI | X seed0 | MECH | 0.5125 | 0.5125 | +0.400 | [-0.500, +1.225] | 4/5 | -2.250 | 6.65% | 0.5301 | 8.975 | 0.045 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_MI | XS equal_seed_mean | CONF_ONLY | 0.6131 | 0.6131 | +0.633 | [-0.067, +1.308] | 4/5 | -0.083 | 7.33% | 0.5396 | 8.525 | 0.074 | PREDICTABLE_WEAK_SIGNAL |
| OpenBMI_MI | XS seed0 | CONF_ONLY | 0.6280 | 0.6280 | +1.700 | [+0.925, +2.525] | 5/5 | +0.125 | 7.95% | 0.6069 | 9.275 | 0.183 | PREDICTABLE_WEAK_SIGNAL |
| OpenBMI_MI | XS seed1 | CONF_ONLY | 0.6213 | 0.6213 | -0.150 | [-0.950, +0.650] | 2/5 | -1.000 | 6.55% | 0.4885 | 7.275 | -0.021 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_MI | XS seed2 | CONF_ONLY | 0.5900 | 0.5900 | +0.350 | [-0.650, +1.275] | 4/5 | -0.250 | 7.50% | 0.5233 | 9.025 | 0.039 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_MI | XS equal_seed_mean | MECH | 0.5791 | 0.5791 | +0.300 | [-0.375, +0.917] | 5/5 | +0.042 | 6.48% | 0.5142 | 8.525 | 0.035 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_MI | XS seed0 | MECH | 0.5882 | 0.5882 | +1.375 | [+0.650, +2.100] | 5/5 | +0.500 | 8.48% | 0.5811 | 9.275 | 0.148 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_MI | XS seed1 | MECH | 0.5811 | 0.5811 | -0.625 | [-1.500, +0.225] | 1/5 | -1.625 | 5.88% | 0.4468 | 7.275 | -0.086 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_MI | XS seed2 | MECH | 0.5680 | 0.5680 | +0.150 | [-0.700, +0.950] | 4/5 | -1.125 | 5.10% | 0.5147 | 9.025 | 0.017 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| OpenBMI_ERP | X seed0 | CONF_ONLY | 0.6603 | 0.6603 | +0.863 | [+0.313, +1.455] | 4/5 | -0.087 | 6.25% | 0.6438 | 5.188 | 0.166 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | X seed0 | MECH | 0.6463 | 0.6463 | +0.527 | [+0.045, +1.026] | 4/5 | -0.610 | 5.59% | 0.6138 | 5.188 | 0.102 | PREDICTABLE_WEAK_SIGNAL |
| OpenBMI_ERP | XS equal_seed_mean | CONF_ONLY | 0.6795 | 0.6795 | +1.023 | [+0.510, +1.560] | 5/5 | +0.691 | 5.29% | 0.6910 | 5.482 | 0.187 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | XS seed0 | CONF_ONLY | 0.6688 | 0.6688 | +0.770 | [+0.054, +1.420] | 5/5 | +0.455 | 5.59% | 0.6640 | 5.414 | 0.142 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | XS seed1 | CONF_ONLY | 0.6778 | 0.6778 | +1.302 | [+0.791, +1.848] | 5/5 | +0.705 | 4.87% | 0.6973 | 5.655 | 0.230 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | XS seed2 | CONF_ONLY | 0.6920 | 0.6920 | +0.996 | [+0.347, +1.633] | 5/5 | +0.258 | 5.41% | 0.7118 | 5.379 | 0.185 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | XS equal_seed_mean | MECH | 0.6877 | 0.6877 | +0.939 | [+0.417, +1.488] | 5/5 | +0.346 | 5.16% | 0.7026 | 5.482 | 0.171 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | XS seed0 | MECH | 0.6715 | 0.6715 | +0.716 | [+0.047, +1.359] | 5/5 | +0.455 | 5.19% | 0.6742 | 5.414 | 0.132 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | XS seed1 | MECH | 0.6947 | 0.6947 | +1.244 | [+0.681, +1.841] | 5/5 | +0.515 | 5.40% | 0.7000 | 5.655 | 0.220 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_ERP | XS seed2 | MECH | 0.6968 | 0.6968 | +0.856 | [+0.245, +1.477] | 5/5 | +0.045 | 4.89% | 0.7336 | 5.379 | 0.159 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | X seed0 | CONF_ONLY | 0.7709 | 0.7610 | +2.875 | [+1.125, +5.025] | 5/5 | +1.125 | 5.83% | 0.7889 | 5.525 | 0.520 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | X seed0 | MECH | 0.7137 | 0.7039 | +2.425 | [+0.650, +4.550] | 5/5 | +0.250 | 6.85% | 0.7118 | 5.525 | 0.439 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS equal_seed_mean | CONF_ONLY | 0.7683 | 0.7869 | +3.142 | [+1.500, +5.250] | 5/5 | +1.208 | 6.08% | 0.7824 | 5.925 | 0.530 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS seed0 | CONF_ONLY | 0.7678 | 0.7891 | +4.400 | [+2.050, +7.626] | 5/5 | +1.375 | 6.85% | 0.8548 | 6.650 | 0.662 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS seed1 | CONF_ONLY | 0.7511 | 0.7776 | +2.225 | [+1.100, +3.450] | 5/5 | +1.375 | 5.50% | 0.7306 | 5.500 | 0.405 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS seed2 | CONF_ONLY | 0.7859 | 0.7941 | +2.800 | [+1.150, +4.975] | 5/5 | +0.375 | 5.90% | 0.7617 | 5.625 | 0.498 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS equal_seed_mean | MECH | 0.7617 | 0.7999 | +3.033 | [+1.233, +5.500] | 5/5 | +1.167 | 6.59% | 0.7506 | 5.925 | 0.512 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS seed0 | MECH | 0.7828 | 0.8176 | +4.325 | [+1.825, +7.975] | 5/5 | +1.500 | 7.92% | 0.8035 | 6.650 | 0.650 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS seed1 | MECH | 0.7273 | 0.7835 | +2.100 | [+1.000, +3.400] | 5/5 | +1.125 | 5.53% | 0.7188 | 5.500 | 0.382 | PREDICTABLE_STRONG_SIGNAL |
| OpenBMI_SSVEP | XS seed2 | MECH | 0.7749 | 0.7986 | +2.675 | [+0.700, +5.450] | 5/5 | +0.750 | 6.33% | 0.7296 | 5.625 | 0.476 | PREDICTABLE_STRONG_SIGNAL |
| WBCIC_MI | X seed0 | CONF_ONLY | 0.5850 | 0.5850 | +0.307 | [-0.274, +0.887] | 4/5 | -0.833 | 4.83% | 0.5318 | 7.583 | 0.040 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | X seed0 | MECH | 0.5363 | 0.5363 | +0.049 | [-0.484, +0.678] | 3/5 | -0.332 | 5.63% | 0.5043 | 7.583 | 0.006 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS equal_seed_mean | CONF_ONLY | 0.5718 | 0.5718 | +0.915 | [-0.010, +1.769] | 5/5 | +0.056 | 10.63% | 0.5437 | 8.067 | 0.113 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS seed0 | CONF_ONLY | 0.5767 | 0.5767 | +0.711 | [-0.403, +1.776] | 4/5 | -0.408 | 11.65% | 0.5305 | 8.438 | 0.084 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS seed1 | CONF_ONLY | 0.5662 | 0.5662 | +1.049 | [-0.064, +2.018] | 4/5 | -0.583 | 9.93% | 0.5528 | 7.744 | 0.135 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS seed2 | CONF_ONLY | 0.5726 | 0.5726 | +0.984 | [-0.033, +1.984] | 5/5 | +0.417 | 10.31% | 0.5477 | 8.018 | 0.123 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS equal_seed_mean | MECH | 0.5622 | 0.5622 | +0.741 | [-0.092, +1.522] | 5/5 | +0.111 | 9.47% | 0.5405 | 8.067 | 0.092 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS seed0 | MECH | 0.5741 | 0.5741 | +0.484 | [-0.613, +1.532] | 4/5 | -0.917 | 10.75% | 0.5225 | 8.438 | 0.057 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS seed1 | MECH | 0.5498 | 0.5498 | +0.855 | [-0.258, +1.774] | 4/5 | -0.667 | 8.41% | 0.5509 | 7.744 | 0.110 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |
| WBCIC_MI | XS seed2 | MECH | 0.5626 | 0.5626 | +0.885 | [+0.046, +1.770] | 4/5 | -0.167 | 9.25% | 0.5480 | 8.018 | 0.110 | HIGH_ORACLE_BUT_NOT_PREDICTABLE |

## Scientific answers

1. XS rescue sets are moderately stable but not seed-invariant: the four-task mean exact-trial rescue Jaccard is 0.4976, while majority-given-any recurrence is task dependent.
2. Harm sets are less stable on average than rescue sets (0.2623 versus 0.4976 mean Jaccard).
3. CONF_ONLY is predictive on ERP and SSVEP but not broadly on both MI tasks; its four-task mean SAFE_SWITCH AUC is 0.6582.
4. MECH does not materially improve predictability: it changes mean SAFE_SWITCH AUC by -0.0105 relative to CONF_ONLY. This is descriptive, not a post-hoc feature selection rule.
5. The fully cross-fitted XS MECH policy improves point-estimate B0 BA on all four tasks (OpenBMI_MI +0.300 pp; OpenBMI_ERP +0.939 pp; OpenBMI_SSVEP +3.033 pp; WBCIC_MI +0.741 pp), but only ERP and SSVEP meet every strong-signal gate; MI uncertainty remains material.
6. Predictability is not fully consistent across XS seeds: ERP and SSVEP are positive for all seeds, WBCIC policy deltas are positive but AUC remains weak, and OpenBMI MI includes a negative seed1 policy delta. Every seed row uses the same fold-specific pooled XS model, not an ensemble.
7. A future conditional residual architecture is not justified by broad strong evidence: 2/4 tasks meet every predeclared strong-signal gate. The logistic policy is diagnostic only.

## Four-task conclusion

ERP is validly included, but intervention-selection predictability is not broadly shared across MI, ERP and SSVEP or across OpenBMI/WBCIC: only ERP and SSVEP satisfy the strong criteria, while both MI analyses have SAFE_SWITCH AUC below 0.60 under MECH.

No EEG model or neural router was trained. Logistic C and feature subsets were not tuned. No internal-heldout, final-heldout, final-test, or sealed-test artifact was opened.
