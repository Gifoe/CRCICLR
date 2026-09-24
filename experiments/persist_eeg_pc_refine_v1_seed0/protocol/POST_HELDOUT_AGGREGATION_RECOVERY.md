# Post-heldout aggregation recovery

All 20 final-heldout prediction cells completed under the hashed
`FINAL_EVAL_LOCK.json`. The locked `run.py aggregate` then failed before
writing primary metric tables. The failure occurred while calculating the
post-hoc top-three correction-energy share: it attempted to average
fold-specific energy vectors directly, although Protected rank differs across
folds.

`code/aggregate_recovery.py` copies the locked aggregation function and
changes only that post-hoc formula. It calculates the top-three energy share
within each fold's own Protected coordinates and then averages the five
shares. Subject-level BA, macro-F1, NLL, session summaries, five-fold
probability averaging, and 20,000-draw paired bootstrap formulas are identical
to the locked source. The script hashes itself, the original final lock, and
every prediction file in `AGGREGATION_RECOVERY_LOCK.json` before aggregation.

This code was added after heldout access. The recovered results are labeled
`EXPLORATORY_POST_HELDOUT` under the original protocol rule and cannot be
presented as an unmodified confirmatory V1 run. No model, basis, checkpoint,
prediction, or selected epoch was changed in this recovery.
