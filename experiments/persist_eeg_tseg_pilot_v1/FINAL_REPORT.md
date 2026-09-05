# TSEG pilot

## Performance

| Dataset | Method | Mean BA | Median BA | Seed SD | Seed Range |
|---|---|---:|---:|---:|---:|
| OpenBMI | B0_SUBJECT_BALANCED_ERM | 75.9524 | 77.5714 | 3.0551 | 5.4286 |
| OpenBMI | B1_PLAIN_MLDG | 72.7619 | 71.5714 | 3.5119 | 6.7143 |
| OpenBMI | B2_TWO_TRAJECTORY_MEAN | 77.6667 | 78.0000 | 0.5774 | 1.0000 |
| OpenBMI | B3_TSEG | 78.0952 | 77.5714 | 1.4310 | 2.7143 |
| WBCIC | B0_SUBJECT_BALANCED_ERM | 74.1333 | 74.7000 | 1.4364 | 2.7000 |
| WBCIC | B1_PLAIN_MLDG | 73.8667 | 74.1000 | 1.0693 | 2.1000 |
| WBCIC | B2_TWO_TRAJECTORY_MEAN | 75.0667 | 75.1000 | 0.9504 | 1.9000 |
| WBCIC | B3_TSEG | 74.2333 | 74.0000 | 0.7767 | 1.5000 |

## Relative

| Dataset | TSEG-ERM mean | positive/3 | min Δ | TSEG-MLDG | TSEG-2TR |
|---|---:|---:|---:|---:|---:|
| OpenBMI | +2.1429 | 2/3 | -0.0000 | +5.3333 | +0.4286 |
| WBCIC | +0.1000 | 2/3 | -1.2000 | +0.3667 | -0.8333 |

Terminal: `TSEG_STABILITY_ONLY_PARTIAL_SIGNAL`

Gates: G1=False, G2=False, G3=True, G4=True, G5=False, G6=True.

Held-out functional disagreement is post-training analysis only; it was not used for checkpoint selection.
TSEG targets functional trajectory stability, not parameter flatness.
Only OpenBMI/WBCIC fold0, EEGNet, seeds 0/1/2 were run; no sealed holdout or WBCIC outer-10 was opened.
