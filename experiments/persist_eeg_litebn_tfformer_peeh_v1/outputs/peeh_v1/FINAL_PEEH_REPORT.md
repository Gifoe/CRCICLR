# LiteBN / TFFormer PEEH v1 - Appendix-L repaired primary diagnostic

No neural model was retrained. Protected selection and all ridge probes use inner-train biological subjects only.
OpenBMI rows use the 14-subject internal-heldout diagnostic cohort.
WBCIC is explicitly source-session persistence -> future-session consequence: persistence uses S0<->S1 and consequence uses true-outer S2.

| Task | Model | Intact probe BA | Protected-erased BA | Random-erased BA | Protected harm | Random harm | PEEH [95% CI] | Protected rank | Protected coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_ERP | LiteBN | 78.806% | 67.535% | 73.356% | 11.271 pp | 5.451 pp | 5.820 [4.964, 6.690] pp | 4.73 [0-12] | 14/15 |
| OpenBMI_ERP | TFFormer | 79.227% | 65.643% | 72.992% | 13.585 pp | 6.235 pp | 7.349 [6.594, 8.157] pp | 5.33 [0-9] | 14/15 |
| OpenBMI_MI | LiteBN | 74.919% | 61.652% | 70.422% | 13.267 pp | 4.498 pp | 8.769 [7.131, 10.464] pp | 6.13 [4-8] | 15/15 |
| OpenBMI_MI | TFFormer | 74.871% | 61.981% | 70.577% | 12.890 pp | 4.294 pp | 8.596 [7.244, 10.069] pp | 5.93 [4-8] | 15/15 |
| OpenBMI_SSVEP | LiteBN | 91.376% | 78.605% | 86.341% | 12.771 pp | 5.035 pp | 7.736 [5.313, 10.188] pp | 4.53 [0-8] | 14/15 |
| OpenBMI_SSVEP | TFFormer | 91.652% | 80.148% | 86.386% | 11.505 pp | 5.267 pp | 6.238 [4.045, 8.479] pp | 4.53 [0-8] | 13/15 |
| WBCIC_MI | LiteBN | 80.413% | 58.570% | 77.426% | 21.843 pp | 2.987 pp | 18.856 [13.173, 24.079] pp | 3.27 [1-4] | 15/15 |
| WBCIC_MI | TFFormer | 80.520% | 57.820% | 77.091% | 22.700 pp | 3.429 pp | 19.271 [13.438, 24.513] pp | 3.33 [1-4] | 15/15 |

PEEH > 0 means TRAIN-selected persistent Protected coordinates are more predictively consequential than equal-rank random coordinates.
Larger PEEH does not imply a better classifier; model superiority remains determined by BA and Macro-F1.
Folds and seeds were averaged within biological subject before the 20,000-draw subject bootstrap.
