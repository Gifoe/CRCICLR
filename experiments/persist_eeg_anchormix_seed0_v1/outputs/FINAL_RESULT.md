# AnchorMix-EEG seed-0 result

Final terminal: **FAIL**

| Dataset | EEGNet | LiteBN | REF50 | AnchorMix | DeltaBest | DeltaREF50 |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI | 75.77% | 79.15% | 80.00% | 78.17% | -0.98 pp | -1.83 pp |
| WBCIC | 78.27% | 78.70% | 79.40% | 78.20% | -0.50 pp | -1.19 pp |

Macro-F1 for every method is in `DATASET_SUMMARY.csv` and `FOLD_METRICS.csv`.

## Fold metrics

| Dataset | Fold | EEGNet | LiteBN | REF50 | AnchorMix | Delta vs best single | Delta vs REF50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI | 0 | 80.00% | 82.62% | 82.62% | 83.75% | +1.12 pp | +1.12 pp |
| OpenBMI | 1 | 81.25% | 83.25% | 84.25% | 84.00% | +0.75 pp | -0.25 pp |
| OpenBMI | 2 | 68.75% | 72.75% | 74.75% | 69.75% | -3.00 pp | -5.00 pp |
| OpenBMI | 3 | 76.00% | 78.75% | 80.25% | 77.00% | -1.75 pp | -3.25 pp |
| OpenBMI | 4 | 72.88% | 78.38% | 78.12% | 76.38% | -2.00 pp | -1.75 pp |
| WBCIC | 0 | 77.14% | 77.93% | 79.86% | 77.64% | -0.29 pp | -2.21 pp |
| WBCIC | 1 | 81.25% | 77.33% | 79.33% | 79.92% | -1.33 pp | +0.58 pp |
| WBCIC | 2 | 74.92% | 75.58% | 75.67% | 73.67% | -1.92 pp | -2.00 pp |
| WBCIC | 3 | 81.47% | 84.05% | 84.06% | 82.31% | -1.75 pp | -1.75 pp |
| WBCIC | 4 | 76.75% | 78.75% | 78.00% | 77.58% | -1.17 pp | -0.42 pp |

1. AnchorMix exceeds the strongest fixed single model on both datasets: NO
2. AnchorMix is within 0.5 pp of REF50 on both datasets: NO
3. OpenBMI/WBCIC direction is consistent: NO
4. Parameter count: see `MODEL_COST.json`.
5. Next step 3 seeds + ERP/SSVEP: NO; stop.

AnchorMix uses one latent representation and one classifier (`self.head`); it has no prediction-level fusion or auxiliary prediction head.
