# Session-fair TSEG closure

## Scope

OpenBMI and WBCIC outer fold 0, EEGNet, seeds 0/1/2. Methods are subject-balanced ERM, two-trajectory mean (2TR), and TSEG. No sealed holdout or WBCIC outer-10 data were opened.

## Performance

| Dataset | Method | Mean BA | Median BA | SD | Range |
|---|---|---:|---:|---:|---:|
| OpenBMI | B0_SUBJECT_BALANCED_ERM | 78.0476 | 77.8571 | 0.5948 | 1.1429 |
| OpenBMI | B2_TWO_TRAJECTORY_MEAN | 77.9048 | 76.1429 | 3.0517 | 5.2857 |
| OpenBMI | B3_TSEG | 78.1905 | 78.4286 | 0.5408 | 1.0000 |
| WBCIC | B0_SUBJECT_BALANCED_ERM | 74.1333 | 74.7000 | 1.4364 | 2.7000 |
| WBCIC | B2_TWO_TRAJECTORY_MEAN | 66.2667 | 73.9000 | 14.0962 | 24.9000 |
| WBCIC | B3_TSEG | 68.4667 | 75.3000 | 12.2712 | 21.5000 |

## Paired deltas versus ERM (BA percentage points)

| Dataset | 2TR mean | 2TR median | 2TR nonnegative seeds | 2TR minimum | TSEG mean | TSEG minus 2TR |
|---|---:|---:|---:|---:|---:|---:|
| OpenBMI | -0.1429 | -1.4286 | 1/3 | -1.7143 | +0.1429 | +0.2857 |
| WBCIC | -7.8667 | -0.8000 | 1/3 | -25.2000 | -5.6667 | +2.2000 |

## Decision

Terminal: `TWO_TRAJECTORY_GAIN_EXPLAINED_OR_NOT_GENERAL`

Gates: A=False, B=False, C=False, D=False, session_fairness=True. The mechanism audit is positive for reduced trajectory disagreement (14,872 matched rows; mean delta_JS=-0.00485724; mean delta_risk-dispersion=-0.00763491), but this does not produce a session-fair generalization gain. TSEG-specific regularization is not supported as a general performance claim.

WBCIC session-2 labels were not used in training stages; methods share the same labeled-session universe. Held-out evaluation was post-training only.
