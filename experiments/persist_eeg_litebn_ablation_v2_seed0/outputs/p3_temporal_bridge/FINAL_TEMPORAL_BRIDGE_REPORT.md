# Temporal-diversity P3 bridge

The Full model is the official frozen LiteBN, not a newly trained matched B0. All comparisons below are therefore descriptive.

| Task | Model | Coverage | Rank mean | PEEH pp [95% CI] | PSWA pp [95% CI] | Future Protected-Random pp | Full-model WS-BA |
|---|---|---:|---:|---:|---:|---:|---:|
| OpenBMI_ERP | B1_SAME_SCALE_63 | 5/15 | 4.80 | 9.72 [7.90, 11.54] | 9.48 [6.16, 13.34] | 11.86 | 0.8189 |
| OpenBMI_ERP | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/15 | 5.80 | 8.04 [6.96, 9.27] | 7.49 [4.98, 10.35] | 9.15 | 0.8150 |
| OpenBMI_MI | B1_SAME_SCALE_63 | 5/15 | 5.20 | 7.44 [5.73, 9.13] | 8.39 [6.87, 10.13] | 9.47 | 0.6767 |
| OpenBMI_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/15 | 7.20 | 10.38 [8.28, 12.77] | 8.86 [7.49, 10.22] | 8.83 | 0.7113 |
| OpenBMI_SSVEP | B1_SAME_SCALE_63 | 5/15 | 5.60 | 8.01 [5.43, 10.59] | 14.52 [12.74, 16.22] | 14.36 | 0.8950 |
| OpenBMI_SSVEP | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/15 | 5.60 | 7.83 [6.06, 9.71] | 14.06 [12.56, 15.49] | 13.47 | 0.8979 |
| WBCIC_MI | B1_SAME_SCALE_63 | 5/15 | 2.40 | 19.82 [13.59, 25.55] | 18.83 [10.75, 27.11] | 22.41 | 0.7191 |
| WBCIC_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/15 | 3.80 | 18.67 [12.90, 23.91] | 15.72 [9.13, 22.44] | 19.17 | 0.7352 |

## Full minus SameScale descriptive contrasts

| Task | PEEH pp | PSWA pp | Full-model WS-BA pp |
|---|---:|---:|---:|
| OpenBMI_MI | +2.93 | +0.47 | +3.46 |
| OpenBMI_ERP | -1.68 | -1.99 | -0.38 |
| OpenBMI_SSVEP | -0.18 | -0.46 | +0.29 |
| WBCIC_MI | -1.15 | -3.12 | +1.61 |
