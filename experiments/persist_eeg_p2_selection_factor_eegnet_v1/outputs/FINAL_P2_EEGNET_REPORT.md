# P2 EEGNet selection-factor ablation

Scope: frozen EEGNet seed 0, four tasks, five folds. Neural training and fine-tuning count: **0**.
All selector construction used TRAIN information only. Joint is the exact corrected PEEH Protected union.

|Task|Random WS-BA|Persistence-only WS-BA|Utility-only WS-BA|Joint WS-BA|Joint-Utility [95% CI]|
|---|---:|---:|---:|---:|---:|
|OpenBMI_MI|53.77|59.73|59.98|59.98|0.000 [0.000, 0.000]|
|OpenBMI_ERP|61.46|61.99|72.37|72.37|0.000 [0.000, 0.000]|
|OpenBMI_SSVEP|60.97|45.89|78.63|75.37|-3.257 [-4.014, -2.500]|
|WBCIC_MI|48.78|58.28|68.91|68.91|0.000 [0.000, 0.000]|

|Task|Joint-Utility future BA [95% CI]|Joint-Persistence future BA [95% CI]|mean J/U overlap|mean J/P overlap|
|---|---:|---:|---:|---:|
|OpenBMI_MI|0.000 [0.000, 0.000]|0.424 [-0.179, 1.004]|1.000|0.960|
|OpenBMI_ERP|0.000 [0.000, 0.000]|10.201 [7.857, 12.656]|1.000|0.256|
|OpenBMI_SSVEP|-3.286 [-4.229, -2.414]|29.300 [25.800, 32.929]|0.800|0.000|
|WBCIC_MI|0.000 [0.000, 0.000]|12.563 [7.843, 16.781]|1.000|0.367|

All 20 task-fold cells were exact-rank feasible. Random draws were kept separate through the session-minimum operation before averaging.

Interpretation is limited to incremental selection value under the matched-rank retained-subspace protocol; these values are not full-model rankings.
