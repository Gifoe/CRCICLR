# Main benchmark table: final true outer

All numeric cells are subject-equal percent. `INC` means incomplete 3-seed checkpoint matrix.
CBraMod MACs are not reported because FFT is unsupported by the unified counter.

| Method | MI BA | MI F1 | MI WS | ERP BA | ERP F1 | ERP WS | SSVEP BA | SSVEP F1 | SSVEP WS | WBCIC BA | WBCIC F1 | WBCIC WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 69.84 | 69.24 | 67.16 | 84.68 | 80.43 | 81.14 | 89.02 | 88.93 | 87.29 | 79.18 | 78.74 | 70.65 |
| CBraMod | 50.68 | 48.18 | 49.70 | 74.74 | 69.22 | 72.89 | 85.03 | 84.90 | 83.10 | 75.45 | 75.27 | 67.32 |
| TeCh | 71.18 | 69.89 | 67.37 | 80.46 | 75.07 | 76.87 | 82.35 | 82.20 | 80.08 | 74.37 | 74.10 | 66.49 |
| ModernTCN | 61.70 | 61.31 | 57.55 | 79.81 | 76.84 | 76.64 | 84.42 | 84.24 | 82.64 | 71.47 | 71.33 | 64.72 |
| Medformer | 67.16 | 66.64 | 63.26 | 80.07 | 78.90 | 76.62 | 88.61 | 88.47 | 87.07 | 75.81 | 75.53 | 68.92 |
| LiteBN | 72.13 | 70.93 | 69.08 | 85.04 | 80.71 | 81.82 | INCOMPLETE | INCOMPLETE | INCOMPLETE | INCOMPLETE | INCOMPLETE | INCOMPLETE |
