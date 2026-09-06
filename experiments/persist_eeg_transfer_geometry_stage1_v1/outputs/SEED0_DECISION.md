# Compact-TG Stage-1 seed-0 decision

| Dataset | EEGNet | Compact-ERM | Compact-TG | TG-vs-ERM | TG-vs-EEGNet |
|---|---:|---:|---:|---:|---:|
| OpenBMI | 0.742000 | 0.772750 | 0.773000 | +0.025 pp | +3.100 pp |
| WBCIC | 0.789743 | 0.773450 | 0.764421 | -0.903 pp | -2.532 pp |

1. Compact backbone over EEGNet: OpenBMI +3.075 pp; WBCIC -1.629 pp.
2. TG over exact Compact-ERM: OpenBMI +0.025 pp; WBCIC -0.903 pp.
3. TG gain >0.5 pp in both datasets: False.
4. TG gain >1.0 pp in both datasets: False.
5. Total gain >1.5 pp in both datasets: False.
6. TG-positive folds: OpenBMI 2/3; WBCIC 0/3.
7. Median subject TG gain: OpenBMI +0.000 pp; WBCIC -1.000 pp.
8. Gain concentration: improved/tie/harmed subjects OpenBMI 17/6/17; WBCIC 11/3/17.
9. Any protocol invalidity: false.

Terminal state: **TG_STAGE1_NEGATIVE_STOP**
