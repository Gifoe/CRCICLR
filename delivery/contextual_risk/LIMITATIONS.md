# Limitations

1. Both branches failed preregistered development gates; there is no completed contextual method to evaluate on formal calibration, internal final, or CAP.
2. Historical V1--V3 work accessed all three datasets, so no historically untouched claim is available.
3. Frozen source-head seeds use seed-specific subject splits. Some evaluated subjects trained the corresponding source head; exact overlap is reported and prevents a uniformly unseen-subject interpretation.
4. The strong oracle headroom is not deployable evidence. A learned A predictor failed on HMC, and B selectors produced negative realized gain on both datasets.
5. Subject counts, not five seed rows, are the independent units; seed outputs are averaged within subject before gate statistics.

An implementation audit initially found that the shared-table builder derived unused formal-calibration/final surfaces before branch selection. Those artifacts and all downstream outputs were invalidated and quarantined. The final artifact was regenerated with role-gated cache-member access; the correction is documented in provenance and the isolation validation asserts zero non-development outcome rows.
