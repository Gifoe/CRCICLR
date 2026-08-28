# Iteration ledger

## V0 — predeclared empirical subject-class residual transport

- Previous failure: not applicable; this is the first validity test.
- Diagnosis before run: residual arithmetic has never been shown to encode a
  class-preserving subject change.
- Change: none.  Use empirical class-balanced centroids, source-only diagonal
  metric scaling, alpha=1, and the two predeclared layers.
- Prediction before run: valid transport must beat no-transport and norm-matched
  random directions on target-subject affinity while preserving an independent
  class probe and avoiding excess off-manifold rate.
- Actual result: pending Stage-0 execution.
- Decision: pending.

### V0 engineering repair E1 (before any metric)

- Failure: the first launch stopped before representation extraction because an
  Arrow-backed Pandas boolean mask was read-only and `mask &= ...` raised
  `ValueError`.
- Diagnosis: writable-array assumption in `row_indices`; no scientific value,
  candidate layer, or gate had been observed.
- Proposed change before rerun: request an explicit writable NumPy copy for the
  subject mask.  No protocol, data role, estimator, control, or threshold changes.
- Actual result: pending rerun.
- Decision: retained as a pure execution repair; Stage-0 hashes must be refrozen.

### V0 engineering repair E2 (before any completed unit)

- Failure: the corrected run reached metric computation, but pairwise sklearn
  calls accumulated 335 CPU-seconds without completing a unit.
- Diagnosis: one-row kNN and class-probe calls caused Python/sklearn dispatch
  overhead; the frozen mathematical queries themselves are batchable.
- Proposed change before rerun: stack all target-subject queries within each
  source-subject/class cell and call the exact same frozen kNN/probe estimators in
  matrix batches.  Preserve every pair, seed, transport, control, and output row.
- Actual result: pending rerun.
- Decision: implementation-only acceleration; no partial unit or metric was
  inspected, and Stage-0 hashes must be refrozen again.
