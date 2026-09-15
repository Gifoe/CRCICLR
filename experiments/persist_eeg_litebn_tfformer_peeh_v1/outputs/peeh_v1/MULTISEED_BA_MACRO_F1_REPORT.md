# Frozen LiteBN / TFFormer multi-seed BA and Macro-F1

No neural inference or retraining was performed. Predictions use cached 64-d h and the frozen final Linear readout.
Five folds and three seeds are averaged within biological subject before 20,000-draw subject bootstrap confidence intervals.

| Task | Model | BA mean [95% CI] | Macro-F1 mean [95% CI] | N subjects | Scope |
|---|---|---:|---:|---:|---|
| OpenBMI_MI | LiteBN | 74.905% [69.624, 80.695] | 74.042% [68.164, 80.128] | 14 | OpenBMI 14-subject internal-heldout diagnostic |
| OpenBMI_MI | TFFormer | 74.824% [69.410, 80.576] | 73.908% [67.851, 80.151] | 14 | OpenBMI 14-subject internal-heldout diagnostic |
| OpenBMI_ERP | LiteBN | 84.797% [80.393, 89.127] | 80.489% [75.948, 85.060] | 14 | OpenBMI 14-subject internal-heldout diagnostic |
| OpenBMI_ERP | TFFormer | 84.946% [80.482, 89.212] | 80.827% [76.237, 85.246] | 14 | OpenBMI 14-subject internal-heldout diagnostic |
| OpenBMI_SSVEP | LiteBN | 91.024% [84.557, 96.233] | 90.915% [84.269, 96.169] | 14 | OpenBMI 14-subject internal-heldout diagnostic |
| OpenBMI_SSVEP | TFFormer | 91.290% [84.766, 96.457] | 91.199% [84.679, 96.344] | 14 | OpenBMI 14-subject internal-heldout diagnostic |
| WBCIC_MI | LiteBN | 80.473% [71.573, 88.397] | 80.230% [71.046, 88.397] | 10 | true-outer S2 (already accessed) |
| WBCIC_MI | TFFormer | 80.787% [71.613, 88.903] | 80.682% [71.543, 88.996] | 10 | true-outer S2 (already accessed) |
