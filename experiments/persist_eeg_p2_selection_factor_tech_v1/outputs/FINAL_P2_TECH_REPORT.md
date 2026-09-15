# P2 TeCh selection-factor replication

Scope: frozen TeCh seed 0, four tasks, five folds. Neural training and fine-tuning count: **0**.
All selector construction used TRAIN information only. Joint is the exact corrected PEEH Protected union.

|Task|Random WS-BA|Persistence-only WS-BA|Utility-only WS-BA|Joint WS-BA|Joint-Utility [95% CI]|
|---|---:|---:|---:|---:|---:|
|OpenBMI_MI|56.11|54.17|64.11|64.11|0.000 [0.000, 0.000]|
|OpenBMI_ERP|61.82|63.79|69.53|69.53|0.000 [0.000, 0.000]|
|OpenBMI_SSVEP|58.36|59.34|69.57|68.23|-1.343 [-2.114, -0.586]|
|WBCIC_MI|56.59|55.28|63.06|63.06|0.000 [0.000, 0.000]|

|Task|Joint-Utility future BA [95% CI]|Joint-Persistence future BA [95% CI]|mean J/U overlap|mean J/P overlap|
|---|---:|---:|---:|---:|
|OpenBMI_MI|0.000 [0.000, 0.000]|12.165 [7.835, 17.277]|1.000|0.200|
|OpenBMI_ERP|0.000 [0.000, 0.000]|4.955 [3.482, 6.339]|1.000|0.486|
|OpenBMI_SSVEP|-1.186 [-1.886, -0.429]|8.114 [6.471, 9.900]|0.700|0.500|
|WBCIC_MI|0.000 [0.000, 0.000]|8.562 [5.281, 11.687]|1.000|0.500|

All 20 task-fold cells were exact-rank feasible. Random draws were kept separate through the session-minimum operation before averaging.

Interpretation is limited to incremental selection value under the matched-rank retained-subspace protocol; these values are not full-model rankings.
