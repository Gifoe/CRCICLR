# BNCI2015-001 seed-0 architecture ablation summary

This is descriptive external replication; it does not select an architecture.

| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA | Parameters |
|---|---:|---:|---:|---:|
| B0_FULL | 0.6642 | 0.6395 | 0.6333 | 45,626 |
| B1_SAME_SCALE_63 | 0.6325 | 0.6095 | 0.6096 | 45,498 |
| B2_SCALE_COLLAPSE | 0.6717 | 0.6463 | 0.6350 | 45,626 |
| B3_SINGLE_SPATIAL_BASIS | 0.6687 | 0.6415 | 0.6425 | 45,698 |
| B4_ONE_STAGE_BACKEND | 0.6429 | 0.6102 | 0.6204 | 39,418 |

| Variant - Full | Metric | Mean delta | 95% CI | Improved / tied / harmed subjects |
|---|---|---:|---:|---:|
| B1_SAME_SCALE_63 | future_S2_BA | -0.0317 | [-0.0687, +0.0054] | 3 / 1 / 8 |
| B1_SAME_SCALE_63 | WS_BA | -0.0238 | [-0.0571, +0.0104] | 5 / 0 / 7 |
| B2_SCALE_COLLAPSE | future_S2_BA | +0.0075 | [-0.0213, +0.0367] | 7 / 1 / 4 |
| B2_SCALE_COLLAPSE | WS_BA | +0.0017 | [-0.0142, +0.0175] | 8 / 0 / 4 |
| B3_SINGLE_SPATIAL_BASIS | future_S2_BA | +0.0046 | [-0.0221, +0.0354] | 6 / 0 / 6 |
| B3_SINGLE_SPATIAL_BASIS | WS_BA | +0.0092 | [-0.0100, +0.0350] | 7 / 0 / 5 |
| B4_ONE_STAGE_BACKEND | future_S2_BA | -0.0212 | [-0.0462, +0.0025] | 4 / 0 / 8 |
| B4_ONE_STAGE_BACKEND | WS_BA | -0.0129 | [-0.0438, +0.0142] | 6 / 1 / 5 |
