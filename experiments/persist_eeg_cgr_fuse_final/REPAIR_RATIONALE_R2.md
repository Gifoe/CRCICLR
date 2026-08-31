# Repair rationale R2

The first server smoke run failed before any model training because the
server pandas build raised `TypeError: 'RangeIndex' object does not support
the context manager protocol` on scalar `.loc` assignment while constructing
the historical leave-one-run-out I003 flag.  The implementation now computes
the same leave-one-run-out majority using grouped sums/counts and a vectorized
assignment.  It additionally guards against groups with fewer than two runs.

This is a compatibility/numerical-equivalence repair only.  The action menu,
six-run consensus definition, data access, architecture, recipe grid, and
selection gate are unchanged; no outcome was used to alter the method.

The second run also exposed a PyTorch warning caused by passing a read-only
NumPy view into `torch.as_tensor`.  The feature array is now copied before
Tensor construction, removing undefined-write behavior without changing any
values.

The same audit found that the implementation forced K2 as STRONGEST_KEEP even
when another legal KEEP-only candidate won the frozen development selection
pool.  Selection now uses the measured best candidate with deterministic
first-candidate tie breaking, as required by the protocol.

The required random control now receives the exact per-sample non-KEEP mass
emitted by CGR-Fuse; the earlier standalone instability proxy was not a
matched control.  The existing no-consensus/no-LCB controls and fixed
subject-level CVaR remain unchanged.

During the same validation pass, the WBCIC plumbing was checked against the
declared S0→S1 protocol.  The initial implementation was found to include S1
labels in source-model training and to pool S0 with S1 in the metric.  It now
trains on S0 labels only, retains both sessions in the bank audit, and reports
the primary WBCIC result on S1 only.  This is a protocol-compliance correction,
not an outcome-driven method change.
