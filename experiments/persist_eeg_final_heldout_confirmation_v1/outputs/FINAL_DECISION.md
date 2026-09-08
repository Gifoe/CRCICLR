# PERSIST-EEG Final Held-Out Subject Confirmation

Final predictor was frozen before held-out evaluation: `FROZEN_LOGIT50`.

| Dataset | EEGNet BA | LiteBN BA | Frozen LOGIT50 BA | Δ vs EEGNet | Median Δ | 95% CI | Positive subjects |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI | 0.7229 | 0.7490 | 0.7557 | +3.281 pp | +2.833 pp | [+1.519, +5.029] pp | 11/14 |
| WBCIC | 0.7936 | 0.7940 | 0.7998 | +0.620 pp | +0.167 pp | [-0.200, +1.523] pp | 6/10 |

## Development vs held-out

| Dataset | Development Δ vs EEGNet | Held-out Δ vs EEGNet |
|---|---:|---:|
| OpenBMI | +3.942 pp | +3.281 pp |
| WBCIC | +0.587 pp | +0.620 pp |

1. OpenBMI held-out confirmation: **STRONG**.
2. WBCIC held-out confirmation: **STRONG**.
3. Does the final fusion outperform EEGNet on both held-out datasets? **YES**.
4. Does the final fusion outperform LiteBN on both held-out datasets? **YES**.
5. Does carrier complementarity remain observable? **YES**.
6. Is there any catastrophic subject-level harm pattern? **NO**.
7. Overall terminal: `FINAL_DUALDATASET_STRONG_CONFIRMATION`.

8. Final scientific interpretation: The primary unit is the held-out subject after averaging its 15 fixed carrier-pair replicates. The predictor, checkpoints, source-only normalizers, and 50/50 logit rule were fixed before held-out labels were opened. The terminal follows the preregistered dataset gates without subject, seed, or fold exclusion. This result is final confirmation evidence and does not license model retuning.
