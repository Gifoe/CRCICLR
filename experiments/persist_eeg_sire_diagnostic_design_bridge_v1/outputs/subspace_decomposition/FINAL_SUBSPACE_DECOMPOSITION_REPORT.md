# SIRE-EEG Protected/complement-retained decomposition

40/40 frozen seed0 analyses; no neural retraining. The official Full checkpoint is not a matched B0.
Prior PEEH/PSWA and full-decoder Table-12 values replay exactly at subject and task level.
Complement-retained denotes E_P(h): it retains non-P active coordinates and residual, not a strict raw P orthogonal complement.
Probe utilities are not additive; no causal mediation claim follows.

| Task | Model | rank(P) | Protected-only WSBA | Complement-retained WSBA | Intact-probe WSBA | PEEH pp | PSWA pp | Full-model WSBA |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_MI | Full SIRE-EEG (historical) | 7.20 | 68.77 | 55.10 | 70.43 | 10.38 | 8.86 | 71.13 |
| OpenBMI_MI | B1 SameScale63 | 5.20 | 64.30 | 57.10 | 67.69 | 7.44 | 8.39 | 67.67 |
| OpenBMI_ERP | Full SIRE-EEG (historical) | 5.80 | 61.95 | 60.68 | 74.32 | 8.04 | 7.49 | 81.50 |
| OpenBMI_ERP | B1 SameScale63 | 4.80 | 62.23 | 61.23 | 74.79 | 9.72 | 9.48 | 81.89 |
| OpenBMI_SSVEP | Full SIRE-EEG (historical) | 5.60 | 74.87 | 74.54 | 90.13 | 7.83 | 14.06 | 89.79 |
| OpenBMI_SSVEP | B1 SameScale63 | 5.60 | 74.34 | 73.57 | 89.49 | 8.01 | 14.52 | 89.50 |
| WBCIC_MI | Full SIRE-EEG (historical) | 3.80 | 71.47 | 53.89 | 73.18 | 18.67 | 15.72 | 73.52 |
| WBCIC_MI | B1 SameScale63 | 2.40 | 70.95 | 52.98 | 71.93 | 19.82 | 18.83 | 71.91 |

Paired B1−Full effects and 20,000-draw biological-subject CIs: `FULL_VS_B1_PAIRED_EFFECTS.csv`.
All four tasks are retained regardless of direction. Protected rank is descriptive, not a quality score.
The full-decoder task/subject WSBA follows the historical Table-12 estimator (fold-average session BA, then min); cell-level columns are per-fold min.

## Paired biological-subject effects

| Task | Metric | B1−Full [95% CI], pp |
|---|---|---:|
| OpenBMI_MI | protected_only_WSBA | -4.47 [-6.19, -2.67] pp |
| OpenBMI_MI | complement_retained_WSBA | +2.00 [+0.11, +4.61] pp |
| OpenBMI_MI | intact_probe_WSBA | -2.74 [-4.84, -0.44] pp |
| OpenBMI_MI | full_model_WSBA | -3.46 [-5.61, -1.13] pp |
| OpenBMI_MI | PEEH_pp | -2.93 [-4.88, -0.95] pp |
| OpenBMI_MI | PSWA_pp | -0.47 [-1.77, +0.84] pp |
| OpenBMI_ERP | protected_only_WSBA | +0.28 [-0.59, +1.13] pp |
| OpenBMI_ERP | complement_retained_WSBA | +0.55 [-0.27, +1.38] pp |
| OpenBMI_ERP | intact_probe_WSBA | +0.47 [-0.25, +1.20] pp |
| OpenBMI_ERP | full_model_WSBA | +0.38 [-0.01, +0.77] pp |
| OpenBMI_ERP | PEEH_pp | +1.68 [+0.66, +2.76] pp |
| OpenBMI_ERP | PSWA_pp | +1.99 [+0.80, +3.19] pp |
| OpenBMI_SSVEP | protected_only_WSBA | -0.53 [-1.33, +0.33] pp |
| OpenBMI_SSVEP | complement_retained_WSBA | -0.97 [-2.60, +0.69] pp |
| OpenBMI_SSVEP | intact_probe_WSBA | -0.64 [-1.16, -0.17] pp |
| OpenBMI_SSVEP | full_model_WSBA | -0.29 [-1.29, +0.76] pp |
| OpenBMI_SSVEP | PEEH_pp | +0.18 [-1.36, +1.66] pp |
| OpenBMI_SSVEP | PSWA_pp | +0.46 [-0.68, +1.66] pp |
| WBCIC_MI | protected_only_WSBA | -0.52 [-1.76, +0.63] pp |
| WBCIC_MI | complement_retained_WSBA | -0.91 [-2.27, +0.70] pp |
| WBCIC_MI | intact_probe_WSBA | -1.25 [-2.67, +0.03] pp |
| WBCIC_MI | full_model_WSBA | -1.61 [-3.18, -0.31] pp |
| WBCIC_MI | PEEH_pp | +1.15 [-0.02, +2.45] pp |
| WBCIC_MI | PSWA_pp | +3.12 [+1.13, +5.00] pp |

## WBCIC interpretation

B1 has higher relative PSWA, but its absolute Protected-only WSBA is lower and its random-only WSBA is also lower. The higher PSWA therefore must not be described as stronger absolute Protected-only prediction. The B1−Full complement-retained point estimate is negative, but its paired CI includes zero; this decomposition does not attribute the decoder gap to complement utility alone.
