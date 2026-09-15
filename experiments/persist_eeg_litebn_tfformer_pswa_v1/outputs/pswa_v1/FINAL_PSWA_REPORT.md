# Frozen LiteBN / TFFormer Protected-Subspace Worst-Session Advantage

This post-hoc analysis performed no neural training, tuning, adaptation, checkpoint selection, Protected selection, or new random-null sampling. Missing evaluation-session embeddings were inferred once in eval mode; existing S2 embeddings were reused.

The corrected PEEH run did not serialize canonical basis matrices or random lists. We therefore reconstructed only the omitted deterministic basis arrays and the exact revised `final-random` lists from the frozen TRAIN embeddings and frozen stable-seed rule, validating rank, whitening floor, persistence block means, Protected unions, and hashes. No persistence permutations or Protected selection were rerun.

## Primary seed0 results

| Task | Model | Protected coverage | Protected-only WS-BA | Random-only WS-BA | PSWA pp [95% CI] | Future-session advantage pp [95% CI] | Recovery |
|---|---|---:|---:|---:|---:|---:|:---:|
| OpenBMI_ERP | LiteBN | 5/5 | 59.69% | 53.21% | 6.48 [4.46, 8.86] | 8.00 [5.71, 10.59] | PASS |
| OpenBMI_ERP | TFFormer | 4/5 | 60.61% | 53.17% | 7.44 [5.56, 9.54] | 9.09 [7.02, 11.31] | PASS |
| OpenBMI_MI | LiteBN | 5/5 | 68.77% | 59.87% | 8.90 [7.59, 10.29] | 8.78 [7.38, 10.04] | PASS |
| OpenBMI_MI | TFFormer | 5/5 | 68.87% | 59.86% | 9.01 [7.79, 10.23] | 8.36 [7.08, 9.56] | PASS |
| OpenBMI_SSVEP | LiteBN | 5/5 | 74.87% | 61.07% | 13.80 [12.25, 15.26] | 13.20 [11.80, 14.57] | PASS |
| OpenBMI_SSVEP | TFFormer | 4/5 | 75.55% | 67.46% | 8.09 [6.86, 9.25] | 7.54 [6.19, 8.78] | PASS |
| WBCIC_MI | LiteBN | 5/5 | 71.47% | 56.59% | 14.88 [8.77, 21.14] | 17.96 [12.22, 22.84] | PASS |
| WBCIC_MI | TFFormer | 5/5 | 71.65% | 55.67% | 15.98 [9.43, 22.66] | 19.36 [13.27, 24.51] | PASS |

## Session-specific Protected-minus-Random advantage

| Task | Model | S0 | S1 | S2 / future |
|---|---|---:|---:|---:|
| OpenBMI_ERP | LiteBN | n/a | 8.15 pp | 8.00 pp |
| OpenBMI_ERP | TFFormer | n/a | 9.16 pp | 9.09 pp |
| OpenBMI_MI | LiteBN | n/a | 9.07 pp | 8.78 pp |
| OpenBMI_MI | TFFormer | n/a | 9.42 pp | 8.36 pp |
| OpenBMI_SSVEP | LiteBN | n/a | 14.84 pp | 13.20 pp |
| OpenBMI_SSVEP | TFFormer | n/a | 9.03 pp | 7.54 pp |
| WBCIC_MI | LiteBN | 17.00 pp | 18.66 pp | 17.96 pp |
| WBCIC_MI | TFFormer | 18.34 pp | 19.47 pp | 19.36 pp |

## Paired LiteBN vs TFFormer

| Task | LiteBN PSWA | TFFormer PSWA | TFFormer - LiteBN [95% CI] |
|---|---:|---:|---:|
| OpenBMI_MI | 8.90 | 9.01 | 0.11 [-0.66, 0.88] |
| OpenBMI_ERP | 6.48 | 7.44 | 0.96 [0.44, 1.41] |
| OpenBMI_SSVEP | 13.80 | 8.09 | -5.71 [-6.84, -4.45] |
| WBCIC_MI | 14.88 | 15.98 | 1.09 [0.45, 1.79] |

PSWA uses the biological subject as the bootstrap unit after averaging all valid fold-seed runs. Positive PSWA means the TRAIN-selected Protected subspace alone supports better worst-session decoding than the exact equal-rank PEEH random controls. It is mechanistic and is not overall classifier performance.
