# TRAIN-only reliability gate review — negative checkpoint

This is not the final pooling comparison. Five frozen CBraMod/OpenBMI_MI/seed-0
TRAIN token extractions and their 500 subject-pairing, 500 Session-2-label,
and 200 subject split-half analyses per fold are complete. The
[`TRAIN_GATE_A_B_REVIEW.json`](outputs/TRAIN_GATE_A_B_REVIEW.json) artifact was
computed after verifying the source seals and every declared source-file SHA.
No OUTER_DEVELOPMENT or formal final-heldout EEG was read for this review.

| Fold | Real top-bottom quartile gap | Real R std | Pairing-null std p95 | Pairing-null spread p | Split-half Spearman | Split-half top-quartile Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.300 | 0.119 | 0.137 | 0.377 | 0.230 | 0.197 |
| 1 | 0.196 | 0.078 | 0.083 | 0.581 | -0.136 | 0.105 |
| 2 | 0.449 | 0.175 | 0.176 | 0.202 | 0.527 | 0.366 |
| 3 | 0.315 | 0.124 | 0.127 | 0.463 | 0.246 | 0.150 |
| 4 | 0.429 | 0.163 | 0.182 | 0.168 | 0.507 | 0.337 |

The quartile gap is visually substantial in four folds, but that alone is not
Gate A. In **zero of five folds** does real token-reliability spread exceed the
predeclared subject-pairing-null p95. Gate A therefore fails. The five-fold
average split-half Spearman is **0.275** (required 0.50) and top-quartile
Jaccard is **0.231** (required 0.40). Gate B also fails. Across-fold pairwise
Spearman averages 0.284, with a range from -0.147 to 0.544; the ranking is
not convincingly stable between folds.

Under the locked decision chain, later pooling gains cannot turn these failed
premises into a supported reliability-aware architecture. The current
source-only stopping interpretation is
`NO_MEANINGFUL_TOKEN_RELIABILITY_HETEROGENEITY`. This is a statement about the
**predeclared null/reproducibility gates**, not a claim that all token scores
are equal. TRAIN utility characterization is still running. Decoder selection,
OUTER evaluation, matched random controls, and bootstrap contrasts have not
been performed; no predictive claim is made here. The high-dimensional
all-token erasure utility fit emits convergence warnings and is provisional.
