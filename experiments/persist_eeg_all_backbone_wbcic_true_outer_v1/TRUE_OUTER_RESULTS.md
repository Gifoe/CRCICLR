# WBCIC True-Outer Backbone Re-evaluation

Cohort: `sub-4, sub-8, sub-10, sub-15, sub-20, sub-39, sub-40, sub-43, sub-46, sub-51`; session 2 only.

| Model | Evidence | BA | Macro-F1 |
|---|---:|---:|---:|
| EEGNet | 5 folds x 3 seeds | 79.18% | 78.74% |
| TCFormer | 5 folds x 3 seeds | 79.14% | 78.48% |
| CBraMod | 5 folds x 3 seeds | 75.45% | 75.27% |
| TeCh | 5 folds x 3 seeds | 74.37% | 74.10% |
| LiteBN | 5 folds x seed0 only | 73.15% | 71.65% |
| ModernTCN | 5 folds x seed0 only | 70.74% | 70.65% |
| Medformer | 5 folds x seed0 only | 75.43% | 75.02% |
| SGN | 5 folds x seed0 only | 66.24% | 66.05% |

Means over the five frozen fold checkpoints (and over three seeds where available). No retraining, tuning, calibration, or ensembling was performed on true outer.
