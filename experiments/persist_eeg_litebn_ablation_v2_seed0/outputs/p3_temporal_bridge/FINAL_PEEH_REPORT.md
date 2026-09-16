# LiteBN / TFFormer PEEH v1 - Appendix-L repaired primary diagnostic

No neural model was retrained. Protected selection and all ridge probes use inner-train biological subjects only.
OpenBMI rows use the 14-subject internal-heldout diagnostic cohort.
WBCIC is explicitly source-session persistence -> future-session consequence: persistence uses S0<->S1 and consequence uses true-outer S2.

| Task | Model | Intact probe BA | Protected-erased BA | Random-erased BA | Protected harm | Random harm | PEEH [95% CI] | Protected rank | Protected coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBMI_ERP | B1_SAME_SCALE_63 | 79.250% | 64.132% | 73.855% | 15.118 pp | 5.395 pp | 9.723 [7.902, 11.542] pp | 4.80 [4-5] | 5/15 |
| OpenBMI_ERP | OFFICIAL_FINAL_LITEBN_REFERENCE | 78.835% | 63.446% | 71.488% | 15.389 pp | 7.346 pp | 8.042 [6.960, 9.272] pp | 5.80 [4-9] | 5/15 |
| OpenBMI_MI | B1_SAME_SCALE_63 | 71.871% | 61.000% | 68.445% | 10.871 pp | 3.427 pp | 7.445 [5.734, 9.130] pp | 5.20 [2-8] | 5/15 |
| OpenBMI_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | 74.143% | 58.957% | 69.336% | 15.186 pp | 4.807 pp | 10.378 [8.277, 12.769] pp | 7.20 [4-8] | 5/15 |
| OpenBMI_SSVEP | B1_SAME_SCALE_63 | 90.671% | 76.343% | 84.353% | 14.329 pp | 6.319 pp | 8.010 [5.432, 10.592] pp | 5.60 [4-8] | 5/15 |
| OpenBMI_SSVEP | OFFICIAL_FINAL_LITEBN_REFERENCE | 91.157% | 77.243% | 85.071% | 13.914 pp | 6.086 pp | 7.828 [6.056, 9.710] pp | 5.60 [4-8] | 5/15 |
| WBCIC_MI | B1_SAME_SCALE_63 | 80.210% | 57.650% | 77.475% | 22.560 pp | 2.735 pp | 19.825 [13.594, 25.552] pp | 2.40 [1-4] | 5/15 |
| WBCIC_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | 80.460% | 58.380% | 77.054% | 22.080 pp | 3.406 pp | 18.674 [12.904, 23.914] pp | 3.80 [3-4] | 5/15 |

PEEH > 0 means TRAIN-selected persistent Protected coordinates are more predictively consequential than equal-rank random coordinates.
Larger PEEH does not imply a better classifier; model superiority remains determined by BA and Macro-F1.
Folds and seeds were averaged within biological subject before the 20,000-draw subject bootstrap.
