# LiteBN / TFFormer PEEH v1

No neural model was retrained. Protected selection and all ridge probes use inner-train biological subjects only.
OpenBMI rows use the 14-subject internal-heldout diagnostic cohort; WBCIC uses the 10-subject true-outer cohort.

| Task | Model | Intact BA | Protected-erased BA | Random-erased BA | Protected harm | Random harm | PEEH [95% CI] | Protected rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_ERP | LiteBN | 78.806% | 68.090% | 73.651% | 10.716 pp | 5.155 pp | 5.561 [4.895, 6.261] pp | 4.60 [0-8] |
| OpenBMI_ERP | TFFormer | 79.227% | 66.074% | 72.971% | 13.153 pp | 6.257 pp | 6.896 [6.067, 7.768] pp | 5.33 [0-9] |
| OpenBMI_MI | LiteBN | 74.919% | 61.652% | 70.422% | 13.267 pp | 4.498 pp | 8.769 [7.131, 10.464] pp | 6.13 [4-8] |
| OpenBMI_MI | TFFormer | 74.871% | 61.981% | 70.577% | 12.890 pp | 4.294 pp | 8.596 [7.244, 10.069] pp | 5.93 [4-8] |
| OpenBMI_SSVEP | LiteBN | 91.376% | 83.567% | 87.962% | 7.810 pp | 3.415 pp | 4.395 [2.869, 6.079] pp | 3.20 [0-8] |
| OpenBMI_SSVEP | TFFormer | 91.652% | 81.586% | 86.994% | 10.067 pp | 4.659 pp | 5.408 [3.498, 7.370] pp | 4.00 [0-8] |
| WBCIC_MI | LiteBN | 80.413% | 58.570% | 77.426% | 21.843 pp | 2.987 pp | 18.856 [13.173, 24.079] pp | 3.27 [1-4] |
| WBCIC_MI | TFFormer | 80.520% | 57.820% | 77.091% | 22.700 pp | 3.429 pp | 19.271 [13.438, 24.513] pp | 3.33 [1-4] |

PEEH > 0 means TRAIN-selected persistent Protected coordinates are more predictively consequential than equal-rank random coordinates.
Larger PEEH does not imply a better classifier; model superiority remains determined by BA and Macro-F1.
Folds and seeds were averaged within biological subject before the 20,000-draw subject bootstrap.
