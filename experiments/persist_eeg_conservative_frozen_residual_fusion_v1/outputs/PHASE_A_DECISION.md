# CFRF-v1 Phase-A Decision

| Metric | OpenBMI | WBCIC |
|---|---:|---:|
| EEGNet BA | 0.7577 | 0.7827 |
| LiteBN BA | 0.7915 | 0.7870 |
| Frozen LOGIT50 BA | 0.8000 | 0.7940 |
| GLOBAL-ALPHA BA | 0.7988 | 0.7932 |
| CFRF BA | 0.7990 | 0.7914 |
| CFRF gain vs EEGNet (pp) | 4.125 | 0.872 |
| CFRF gain vs LOGIT50 (pp) | -0.100 | -0.259 |
| Median subject gain vs EEGNet (pp) | 3.000 | 0.500 |
| Positive folds vs EEGNet | 5 | 4 |
| Harm <= -1 pp vs EEGNet | 0.125 | 0.194 |

1. Did CFRF preserve the existing fusion advantage? NO
2. Did CFRF improve WBCIC mean performance? NO
3. Did CFRF reduce the WBCIC harmful-subject tail? NO
4. Did CFRF outperform GLOBAL-ALPHA? NO
5. Is the sample-wise head justified? NO
6. Does CFRF proceed to multiseed? NO
7. If NO, is Frozen LOGIT50 selected as fallback final predictor? YES
8. Terminal: **CFRF_HURTS_VALIDATED_FUSION_USE_FROZEN_LOGIT50**

Next action: Freeze the selected final predictor and run the separate final held-out-subject confirmation on OpenBMI holdout subjects and WBCIC true outer subjects.
