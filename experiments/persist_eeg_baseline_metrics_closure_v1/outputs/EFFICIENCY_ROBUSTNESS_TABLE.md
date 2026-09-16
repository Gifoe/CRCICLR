# Efficiency and robustness

Params are recorded trainable parameters. MACs use one verified Conv/Matmul convention;
CBraMod contains an FFT major operator not counted by that profiler, so its MACs are not estimated.

| Method | Task | Params | MACs (M) | WS-BA (%) | CSGD (pp) |
|---|---|---:|---:|---:|---:|
| EEGNet | OpenBMI_MI | 34162 | 32.90 | 67.16 | 0.99 |
| EEGNet | OpenBMI_ERP | 9586 | 8.22 | 81.14 | -0.13 |
| EEGNet | OpenBMI_SSVEP | 34292 | 32.90 | 87.29 | 4.11 |
| EEGNet | WBCIC_MI | 34098 | 30.78 | 70.65 | -1.11 |
| CBraMod | OpenBMI_MI | 4884002 | NE | 49.70 | -0.50 |
| CBraMod | OpenBMI_ERP | 4884002 | NE | 72.89 | 1.47 |
| CBraMod | OpenBMI_SSVEP | 4884404 | NE | 83.10 | 2.99 |
| CBraMod | WBCIC_MI | 4884002 | NE | 67.32 | -0.63 |
| TeCh | OpenBMI_MI | 793026 | 131.08 | 67.37 | -0.03 |
| TeCh | OpenBMI_ERP | 793026 | 32.97 | 76.87 | 0.38 |
| TeCh | OpenBMI_SSVEP | 6203140 | 3163.71 | 80.08 | 3.77 |
| TeCh | WBCIC_MI | 789954 | 130.56 | 66.49 | -0.15 |
| ModernTCN | OpenBMI_MI | 17391170 | 688.84 | 57.55 | -1.00 |
| ModernTCN | OpenBMI_ERP | 17026114 | 173.42 | 76.64 | 0.33 |
| ModernTCN | OpenBMI_SSVEP | 17883204 | 689.34 | 82.64 | 3.77 |
| ModernTCN | WBCIC_MI | 15914050 | 629.67 | 64.72 | 0.25 |
| Medformer | OpenBMI_MI | 3890178 | 587.91 | 63.26 | -0.69 |
| Medformer | OpenBMI_ERP | 3781890 | 133.79 | 76.62 | 0.66 |
| Medformer | OpenBMI_SSVEP | 4035332 | 588.05 | 87.07 | 3.60 |
| Medformer | WBCIC_MI | 3853314 | 584.81 | 68.92 | 0.53 |
| LiteBN | OpenBMI_MI | 16618 | 105.31 | 69.08 | 0.44 |
| LiteBN | OpenBMI_ERP | 16618 | 26.33 | 81.82 | 0.20 |
| LiteBN | OpenBMI_SSVEP | 16748 | 105.31 | INCOMPLETE | INCOMPLETE |
| LiteBN | WBCIC_MI | 16426 | 98.56 | INCOMPLETE | INCOMPLETE |
