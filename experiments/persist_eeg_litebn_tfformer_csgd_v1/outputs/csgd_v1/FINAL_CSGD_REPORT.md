# Frozen LiteBN / TFFormer cross-session generalization drop

No model was trained, fine-tuned, recalibrated, adapted, or used with updated BN statistics. Each selected checkpoint used its original fold-specific train-only channelwise normalizer unchanged for every session.

OpenBMI uses the previously exposed 14-subject internal-heldout diagnostic cohort on S1 and S2. WBCIC uses the previously exposed 10-subject true-outer cohort on S0, S1, and S2. Labels were used only for post-hoc metrics; these cohorts are not untouched after this analysis.

## Task-level CSGD

| Task | Model | Source BA | Future BA | CSGD pp [95% CI] | Median CSGD | Subjects CSGD >1 pp |
|---|---|---:|---:|---:|---:|---:|
| OpenBMI_ERP | LiteBN | 84.84% | 84.80% | 0.05 [-4.46, 4.54] | -0.28 | 35.7% |
| OpenBMI_ERP | TFFormer | 85.07% | 84.95% | 0.12 [-4.47, 4.61] | -0.41 | 35.7% |
| OpenBMI_MI | LiteBN | 75.43% | 74.90% | 0.53 [-4.70, 5.30] | 1.93 | 64.3% |
| OpenBMI_MI | TFFormer | 75.62% | 74.82% | 0.80 [-4.31, 5.58] | 2.17 | 71.4% |
| OpenBMI_SSVEP | LiteBN | 95.18% | 91.02% | 4.16 [-0.60, 10.78] | 0.00 | 42.9% |
| OpenBMI_SSVEP | TFFormer | 95.30% | 91.29% | 4.01 [-0.80, 10.66] | 0.00 | 42.9% |
| WBCIC_MI | LiteBN | 80.07% | 80.47% | -0.41 [-7.92, 9.95] | -1.67 | 20.0% |
| WBCIC_MI | TFFormer | 80.45% | 80.79% | -0.34 [-7.93, 10.18] | -1.49 | 20.0% |

## Paired LiteBN vs TFFormer

| Task | LiteBN CSGD | TFFormer CSGD | TFFormer - LiteBN CSGD [95% CI] |
|---|---:|---:|---:|
| OpenBMI_MI | 0.53 | 0.80 | 0.27 [0.06, 0.50] |
| OpenBMI_ERP | 0.05 | 0.12 | 0.07 [-0.03, 0.18] |
| OpenBMI_SSVEP | 4.16 | 4.01 | -0.14 [-0.34, 0.02] |
| WBCIC_MI | -0.41 | -0.34 | 0.07 [-0.20, 0.35] |

## WBCIC session details

| Model | S0 BA | S1 BA | S2 BA | S0->S2 drop | S1->S2 drop |
|---|---:|---:|---:|---:|---:|
| LiteBN | 79.04% | 81.09% | 80.47% | -1.44 | 0.62 |
| TFFormer | 79.32% | 81.58% | 80.79% | -1.47 | 0.79 |

CSGD is computed after first averaging all 15 fold-seed evaluations within each biological subject. Confidence intervals use 20,000 bootstrap resamples of biological subjects, never folds or seeds.

Smaller CSGD means less future-session degradation. Negative paired delta means TFFormer degrades less. CSGD alone does not establish model superiority; it must be interpreted jointly with future-session BA and Macro-F1.
