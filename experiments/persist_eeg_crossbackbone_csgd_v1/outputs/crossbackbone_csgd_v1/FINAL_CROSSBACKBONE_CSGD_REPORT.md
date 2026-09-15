# Final Cross-Backbone CSGD Report

CSGD is reported in percentage points; lower is better. Future-session BA and CSGD must be interpreted jointly: a weak model can have a small drop because it is weak in every session.

All results use exact frozen selected checkpoints, the original train-only channelwise normalizer verified by hash, actual frozen classification heads, and eval/inference mode. No retraining, finetuning, calibration, target adaptation, BN update, normalization refit, or evaluation-label-based selection occurred.

## Primary seed0 results

| Model | Task | Coverage | Source BA | Future BA ↑ | CSGD ↓ pp [95% CI] | Median CSGD |
|---|---|---:|---:|---:|---:|---:|
| EEGNet | OpenBMI_MI | 5/5 | 69.27 | 68.10 | 1.17 [-3.19, 5.16] | 2.30 |
| EEGNet | OpenBMI_ERP | 5/5 | 84.64 | 84.83 | -0.19 [-4.81, 4.52] | -1.20 |
| EEGNet | OpenBMI_SSVEP | 5/5 | 93.70 | 89.67 | 4.03 [-1.60, 11.39] | 0.70 |
| EEGNet | WBCIC_MI | 5/5 | 78.66 | 79.66 | -1.00 [-9.34, 9.90] | -0.80 |
| CBraMod | OpenBMI_MI | 5/5 | 50.36 | 50.64 | -0.29 [-1.59, 1.03] | -0.80 |
| CBraMod | OpenBMI_ERP | 5/5 | 76.55 | 75.07 | 1.49 [-2.12, 5.36] | 0.37 |
| CBraMod | OpenBMI_SSVEP | 5/5 | 87.89 | 84.80 | 3.09 [-2.00, 10.47] | -1.60 |
| CBraMod | WBCIC_MI | 5/5 | 74.76 | 75.39 | -0.63 [-8.56, 9.70] | -2.50 |
| TeCh | OpenBMI_MI | 5/5 | 71.80 | 71.46 | 0.34 [-4.51, 4.99] | -0.50 |
| TeCh | OpenBMI_ERP | 5/5 | 80.62 | 80.43 | 0.18 [-4.77, 5.25] | -1.86 |
| TeCh | OpenBMI_SSVEP | 5/5 | 86.39 | 83.04 | 3.34 [-2.37, 10.99] | -1.00 |
| TeCh | WBCIC_MI | 5/5 | 74.07 | 74.20 | -0.13 [-8.08, 9.12] | -1.32 |
| ModernTCN | OpenBMI_MI | 5/5 | 60.44 | 61.34 | -0.90 [-5.91, 3.61] | 0.20 |
| ModernTCN | OpenBMI_ERP | 5/5 | 79.81 | 79.69 | 0.13 [-4.45, 4.50] | -0.43 |
| ModernTCN | OpenBMI_SSVEP | 5/5 | 88.16 | 84.51 | 3.64 [-2.04, 11.56] | -0.50 |
| ModernTCN | WBCIC_MI | 5/5 | 71.23 | 70.74 | 0.49 [-6.37, 8.74] | -1.23 |
| Medformer | OpenBMI_MI | 5/5 | 66.80 | 67.50 | -0.70 [-5.19, 3.73] | -2.00 |
| Medformer | OpenBMI_ERP | 5/5 | 80.39 | 79.59 | 0.80 [-4.13, 5.83] | 0.62 |
| Medformer | OpenBMI_SSVEP | 5/5 | 92.37 | 88.50 | 3.87 [-1.47, 11.16] | -0.60 |
| Medformer | WBCIC_MI | 5/5 | 75.94 | 75.43 | 0.51 [-7.62, 10.52] | -1.38 |
| SGN | OpenBMI_MI | 5/5 | 51.14 | 52.29 | -1.14 [-3.36, 0.91] | -0.90 |
| SGN | OpenBMI_ERP | 4/5 incomplete | — | — | — | — |
| SGN | OpenBMI_SSVEP | 0/5 incomplete | — | — | — | — |
| SGN | WBCIC_MI | 5/5 | 68.22 | 66.24 | 1.97 [-5.80, 11.46] | -5.72 |
| LiteBN | OpenBMI_MI | 5/5 | 73.00 | 73.40 | -0.40 [-5.23, 4.41] | 1.00 |
| LiteBN | OpenBMI_ERP | 5/5 | 85.12 | 84.99 | 0.13 [-4.37, 4.53] | -0.41 |
| LiteBN | OpenBMI_SSVEP | 5/5 | 90.39 | 87.70 | 2.69 [-2.11, 9.37] | 0.00 |
| LiteBN | WBCIC_MI | 5/5 | 73.44 | 73.15 | 0.29 [-4.84, 6.24] | 0.70 |
| TFFormer | OpenBMI_MI | 0/5 incomplete | — | — | — | — |
| TFFormer | OpenBMI_ERP | 0/5 incomplete | — | — | — | — |
| TFFormer | OpenBMI_SSVEP | 0/5 incomplete | — | — | — | — |
| TFFormer | WBCIC_MI | 0/5 incomplete | — | — | — | — |

## Compact cross-task summary

| Model | MI Future BA | MI CSGD | ERP Future BA | ERP CSGD | SSVEP Future BA | SSVEP CSGD | WBCIC Future BA | WBCIC CSGD | Mean CSGD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 68.10 | 1.17 | 84.83 | -0.19 | 89.67 | 4.03 | 79.66 | -1.00 | 1.00 |
| CBraMod | 50.64 | -0.29 | 75.07 | 1.49 | 84.80 | 3.09 | 75.39 | -0.63 | 0.91 |
| TeCh | 71.46 | 0.34 | 80.43 | 0.18 | 83.04 | 3.34 | 74.20 | -0.13 | 0.93 |
| ModernTCN | 61.34 | -0.90 | 79.69 | 0.13 | 84.51 | 3.64 | 70.74 | 0.49 | 0.84 |
| Medformer | 67.50 | -0.70 | 79.59 | 0.80 | 88.50 | 3.87 | 75.43 | 0.51 | 1.12 |
| SGN | 52.29 | -1.14 | — | — | — | — | 66.24 | 1.97 | — |
| LiteBN | 73.40 | -0.40 | 84.99 | 0.13 | 87.70 | 2.69 | 73.15 | 0.29 | 0.68 |
| TFFormer | — | — | — | — | — | — | — | — | — |

## Secondary complete three-seed results

Only model-task combinations with all 15 checkpoints (five folds × seeds 0/1/2) are estimated here.

| Model | Task | Coverage | Future BA ↑ | CSGD ↓ pp [95% CI] |
|---|---|---:|---:|---:|
| EEGNet | OpenBMI_MI | 15/15 | 69.84 | 0.99 [-3.56, 5.33] |
| EEGNet | OpenBMI_ERP | 15/15 | 84.68 | -0.13 [-4.78, 4.52] |
| EEGNet | OpenBMI_SSVEP | 15/15 | 89.02 | 4.11 [-1.59, 11.78] |
| EEGNet | WBCIC_MI | 15/15 | 79.18 | -1.12 [-9.15, 9.36] |
| CBraMod | OpenBMI_MI | 15/15 | 50.68 | -0.50 [-1.35, 0.39] |
| CBraMod | OpenBMI_ERP | 15/15 | 74.74 | 1.47 [-2.09, 5.32] |
| CBraMod | OpenBMI_SSVEP | 15/15 | 85.03 | 2.99 [-2.03, 10.38] |
| CBraMod | WBCIC_MI | 15/15 | 75.45 | -0.63 [-8.30, 9.05] |
| TeCh | OpenBMI_MI | 15/15 | 71.18 | -0.03 [-5.00, 4.73] |
| TeCh | OpenBMI_ERP | 15/15 | 80.46 | 0.38 [-4.57, 5.54] |
| TeCh | OpenBMI_SSVEP | 15/15 | 82.35 | 3.77 [-2.28, 11.47] |
| TeCh | WBCIC_MI | 15/15 | 74.37 | -0.15 [-7.95, 9.25] |
| ModernTCN | OpenBMI_MI | 15/15 | 61.70 | -1.00 [-6.23, 3.66] |
| ModernTCN | OpenBMI_ERP | 15/15 | 79.81 | 0.33 [-4.21, 4.79] |
| Medformer | OpenBMI_MI | 15/15 | 67.16 | -0.69 [-5.25, 3.69] |
| SGN | OpenBMI_MI | 15/15 | 52.26 | -1.41 [-3.32, 0.50] |
| LiteBN | OpenBMI_MI | 15/15 | 72.13 | 0.44 [-3.95, 4.60] |
| LiteBN | OpenBMI_ERP | 15/15 | 85.04 | 0.20 [-4.33, 4.64] |

## Incomplete checkpoint matrices

- SGN / OpenBMI_ERP: seed0 4/5; excluded from metric aggregation.
- SGN / OpenBMI_SSVEP: seed0 0/5; excluded from metric aggregation.
- TFFormer / OpenBMI_MI: seed0 0/5; excluded from metric aggregation.
- TFFormer / OpenBMI_ERP: seed0 0/5; excluded from metric aggregation.
- TFFormer / OpenBMI_SSVEP: seed0 0/5; excluded from metric aggregation.
- TFFormer / WBCIC_MI: seed0 0/5; excluded from metric aggregation.

TFFormer has no exact recorded checkpoint/model identity. The existing TCFormer artifacts were not silently relabeled or substituted.

SGN is reported only for complete OpenBMI MI and WBCIC MI matrices. OpenBMI ERP lacks seed0 fold4; OpenBMI SSVEP lacks all seed0 folds.

## Interpretation guard

Future BA ↑ measures retained predictive performance in the future session. CSGD ↓ measures degradation relative to the source session(s). Neither alone establishes the desired property; use both columns jointly. PEEH is not used for ranking because it answers a different question.
