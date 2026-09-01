# CHECKPOINT EQUIVALENCE

Pending until the pilot run.  Before final adapter training, the runner must
load each canonical seed-0 refit checkpoint and compare outcome trial IDs,
labels, predictions and probabilities to the canonical seed-0 trial table.
The numerical tolerance is `1e-5`.  Failure terminal:
`CDE_PILOT_CHECKPOINT_EQUIVALENCE_FAIL`.  No outcome metric may enter this
equivalence check or fusion selection.
