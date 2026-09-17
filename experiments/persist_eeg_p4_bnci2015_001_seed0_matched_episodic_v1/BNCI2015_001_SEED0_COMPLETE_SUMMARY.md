# BNCI2015-001 external replication — complete seed-0 synthesis

This report is descriptive. The original Full SIRE-EEG and EEGNet seed-0 results remain frozen.

## 1. Full-model external prediction

| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |
|---|---:|---:|---:|
| EEGNet | 0.6350 | 0.5909 | 0.6233 |
| SIRE-EEG | 0.6642 | 0.6395 | 0.6333 |

### Paired SIRE-EEG minus EEGNet

| Metric | Mean delta | 95% CI | Bootstrap unit |
|---|---:|---:|---|
| future S2 BA | +0.0292 | [+0.0033, +0.0575] | biological subject, 20,000 draws |
| WS-BA | +0.0100 | [-0.0187, +0.0408] | biological subject, 20,000 draws |

## 2. External PERSIST diagnostics

| Diagnostic | Model | Mean pp [95% CI] | Coverage |
|---|---|---:|---:|
| PEEH | EEGNet | 5.688 [2.139, 9.521] | 5/5 |
| PSWA | EEGNet | 7.233 [2.699, 12.702] | 5/5 |
| PEEH | SIRE-EEG | 4.476 [0.875, 8.535] | 5/5 |
| PSWA | SIRE-EEG | 7.555 [4.555, 10.373] | 5/5 |

PEEH/PSWA are diagnostic quantities and are not model-quality scores.

## 3. External architecture replication

| Model | Future S2 BA | WS-BA |
|---|---:|---:|
| B0_FULL | 0.6642 | 0.6333 |
| B1_SAME_SCALE_63 | 0.6325 | 0.6096 |
| B2_SCALE_COLLAPSE | 0.6717 | 0.6350 |
| B3_SINGLE_SPATIAL_BASIS | 0.6687 | 0.6425 |
| B4_ONE_STAGE_BACKEND | 0.6429 | 0.6204 |

### Paired variant minus Full

| Variant | Metric | Mean delta | 95% CI |
|---|---|---:|---:|
| B1_SAME_SCALE_63 | future_S2_BA | -0.0317 | [-0.0687, +0.0054] |
| B1_SAME_SCALE_63 | WS_BA | -0.0238 | [-0.0571, +0.0104] |
| B2_SCALE_COLLAPSE | future_S2_BA | +0.0075 | [-0.0213, +0.0367] |
| B2_SCALE_COLLAPSE | WS_BA | +0.0017 | [-0.0142, +0.0175] |
| B3_SINGLE_SPATIAL_BASIS | future_S2_BA | +0.0046 | [-0.0221, +0.0354] |
| B3_SINGLE_SPATIAL_BASIS | WS_BA | +0.0092 | [-0.0100, +0.0350] |
| B4_ONE_STAGE_BACKEND | future_S2_BA | -0.0212 | [-0.0462, +0.0025] |
| B4_ONE_STAGE_BACKEND | WS_BA | -0.0129 | [-0.0437, +0.0142] |

No architecture was selected from these outcomes.

## 4. Rank-matched ScaleCollapse control

| Metric | Scale harm pp | Random rank-16 harm pp | Excess pp [95% CI] |
|---|---:|---:|---:|
| future_S2_BA | +4.167 | +7.897 | -3.730 [-7.250, -0.454] |
| WS_BA | +2.708 | +6.977 | -4.269 [-7.562, -1.327] |
