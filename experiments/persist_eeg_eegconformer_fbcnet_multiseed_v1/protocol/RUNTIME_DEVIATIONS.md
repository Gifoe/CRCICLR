# Runtime deviations

None at protocol lock. The 128-sample batch is fixed. If a particular cell cannot fit, any batch/accumulation adjustment must be recorded here before that cell is retried; no architecture change is permitted.

Fixed FBCNet filter-bank outputs are stored as fold-local runtime memmaps to avoid repeatedly filtering the same training trials. This is numerically the same deterministic `float32` transform and is excluded from the repository. Resume snapshots capture the optimizer, RNG, selected state and epoch, so an interrupted cell resumes without redoing completed epochs.
