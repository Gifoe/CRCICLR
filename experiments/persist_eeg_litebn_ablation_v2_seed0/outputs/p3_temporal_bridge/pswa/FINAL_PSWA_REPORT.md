# Frozen Full-reference / SameScale-63 Protected-Subspace Worst-Session Advantage

The Full model is the official frozen LiteBN, not a newly trained matched B0 control.

This post-hoc analysis performed no neural training, tuning, adaptation, checkpoint selection, Protected selection, or new random-null sampling. Missing evaluation-session embeddings were inferred once in eval mode; existing S2 embeddings were reused.

The corrected PEEH run did not serialize canonical basis matrices or random lists. We therefore reconstructed only the omitted deterministic basis arrays and the exact revised `final-random` lists from the frozen TRAIN embeddings and frozen stable-seed rule, validating rank, whitening floor, persistence block means, Protected unions, and hashes. No persistence permutations or Protected selection were rerun.

## Primary seed0 results

| Task | Model | Protected coverage | Protected-only WS-BA | Random-only WS-BA | PSWA pp [95% CI] | Future-session advantage pp [95% CI] | Recovery |
|---|---|---:|---:|---:|---:|---:|:---:|
| OpenBMI_ERP | B1_SAME_SCALE_63 | 5/5 | 62.23% | 52.75% | 9.48 [6.16, 13.34] | 11.86 [8.45, 15.54] | PASS |
| OpenBMI_ERP | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/5 | 61.95% | 54.46% | 7.49 [4.98, 10.35] | 9.15 [6.83, 11.71] | PASS |
| OpenBMI_MI | B1_SAME_SCALE_63 | 5/5 | 64.30% | 55.91% | 8.39 [6.87, 10.13] | 9.47 [7.39, 11.87] | PASS |
| OpenBMI_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/5 | 68.77% | 59.91% | 8.86 [7.49, 10.22] | 8.83 [7.36, 10.20] | PASS |
| OpenBMI_SSVEP | B1_SAME_SCALE_63 | 5/5 | 74.34% | 59.82% | 14.52 [12.74, 16.22] | 14.36 [12.66, 15.95] | PASS |
| OpenBMI_SSVEP | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/5 | 74.87% | 60.81% | 14.06 [12.56, 15.49] | 13.47 [12.07, 14.87] | PASS |
| WBCIC_MI | B1_SAME_SCALE_63 | 5/5 | 70.95% | 52.12% | 18.83 [10.75, 27.11] | 22.41 [14.91, 29.18] | PASS |
| WBCIC_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | 5/5 | 71.47% | 55.75% | 15.72 [9.13, 22.44] | 19.17 [13.15, 24.49] | PASS |

## Session-specific Protected-minus-Random advantage

| Task | Model | S0 | S1 | S2 / future |
|---|---|---:|---:|---:|
| OpenBMI_ERP | B1_SAME_SCALE_63 | n/a | 11.79 pp | 11.86 pp |
| OpenBMI_ERP | OFFICIAL_FINAL_LITEBN_REFERENCE | n/a | 9.47 pp | 9.15 pp |
| OpenBMI_MI | B1_SAME_SCALE_63 | n/a | 8.70 pp | 9.47 pp |
| OpenBMI_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | n/a | 8.97 pp | 8.83 pp |
| OpenBMI_SSVEP | B1_SAME_SCALE_63 | n/a | 15.27 pp | 14.36 pp |
| OpenBMI_SSVEP | OFFICIAL_FINAL_LITEBN_REFERENCE | n/a | 15.13 pp | 13.47 pp |
| WBCIC_MI | B1_SAME_SCALE_63 | 21.77 pp | 23.71 pp | 22.41 pp |
| WBCIC_MI | OFFICIAL_FINAL_LITEBN_REFERENCE | 18.02 pp | 19.75 pp | 19.17 pp |

## Paired Full reference vs SameScale

| Task | Full PSWA | SameScale PSWA | Full - SameScale [95% CI] |
|---|---:|---:|---:|
| OpenBMI_MI | 8.86 | 8.39 | 0.47 [-0.84, 1.77] |
| OpenBMI_ERP | 7.49 | 9.48 | -1.99 [-3.21, -0.78] |
| OpenBMI_SSVEP | 14.06 | 14.52 | -0.46 [-1.64, 0.70] |
| WBCIC_MI | 15.72 | 18.83 | -3.12 [-5.00, -1.12] |

PSWA uses the biological subject as the bootstrap unit after averaging all valid fold-seed runs. Positive PSWA means the TRAIN-selected Protected subspace alone supports better worst-session decoding than the exact equal-rank PEEH random controls. It is mechanistic and is not overall classifier performance.
