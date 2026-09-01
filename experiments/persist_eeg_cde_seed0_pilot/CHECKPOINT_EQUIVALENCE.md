# CHECKPOINT EQUIVALENCE

The canonical seed-0 refit checkpoint was loaded without modification. Before final adapter training, outcome trial IDs, labels, predictions and probabilities were compared to the canonical seed-0 trial table; no outcome metric was computed or used for selection.

| Dataset | Fold | Trials | Max probability abs diff | IDs | Labels | Predictions |
|---|---:|---:|---:|---|---|---|
| OpenBMI | 0 | 1100 | 1.110e-16 | PASS | PASS | PASS |
| OpenBMI | 1 | 1100 | 1.110e-16 | PASS | PASS | PASS |
| OpenBMI | 2 | 1100 | 1.110e-16 | PASS | PASS | PASS |
| OpenBMI | 3 | 1100 | 1.110e-16 | PASS | PASS | PASS |
| OpenBMI | 4 | 1000 | 1.110e-16 | PASS | PASS | PASS |
| WBCIC | 0 | 1800 | 1.110e-16 | PASS | PASS | PASS |
| WBCIC | 1 | 1600 | 1.110e-16 | PASS | PASS | PASS |
| WBCIC | 2 | 1600 | 1.110e-16 | PASS | PASS | PASS |
| WBCIC | 3 | 1595 | 1.110e-16 | PASS | PASS | PASS |
| WBCIC | 4 | 1600 | 1.110e-16 | PASS | PASS | PASS |

checkpoint_equivalence = PASS
terminal on failure would have been `CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL`.
