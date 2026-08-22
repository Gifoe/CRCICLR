# Bug and repair record

Before the first server run, two implementation issues were repaired:

1. Anchor training performed a second forward pass only to count accuracy. That could update BatchNorm statistics and consume unnecessary compute. The repair stores the first forward logits and uses them for both loss and accuracy.
2. The initial outer-lock writer used the literal key `"**flags**"` instead of merging the flags dictionary. This was corrected before any outer command could run, and the outer status string was changed from a historical AGDI label to the Exp4-specific non-authorizing/authorizing states.

The V2/V3 response correction was a scientific iteration, not a bug fix; it is documented in `ITERATION_LEDGER.md`.
