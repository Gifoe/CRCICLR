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
- Actual result: validated `TRANSPORT_NOT_SUBJECT_FAITHFUL`.  Residual stability
  passed in every setting/layer, but alpha=1 target-affinity improvement was
  negative for both WBCIC backbones and the centroid-manifold distance ratio was
  1.76–2.68.  Several pre-embedding class gates also failed.
- Decision: do not train.  Authorize one magnitude-only repair because the
  directions were stable and beat norm-matched random controls, which diagnoses
  overshoot/session-scale mismatch rather than arbitrary directionality.

### V0 engineering repair E1 (before any metric)

- Failure: the first launch stopped before representation extraction because an
  Arrow-backed Pandas boolean mask was read-only and `mask &= ...` raised
  `ValueError`.
- Diagnosis: writable-array assumption in `row_indices`; no scientific value,
  candidate layer, or gate had been observed.
- Proposed change before rerun: request an explicit writable NumPy copy for the
  subject mask.  No protocol, data role, estimator, control, or threshold changes.
- Actual result: fixed; all 40 units subsequently completed.
- Decision: retained as a pure execution repair; Stage-0 hashes must be refrozen.

### V0 engineering repair E2 (before any completed unit)

- Failure: the corrected run reached metric computation, but pairwise sklearn
  calls accumulated 335 CPU-seconds without completing a unit.
- Diagnosis: one-row kNN and class-probe calls caused Python/sklearn dispatch
  overhead; the frozen mathematical queries themselves are batchable.
- Proposed change before rerun: stack all target-subject queries within each
  source-subject/class cell and call the exact same frozen kNN/probe estimators in
  matrix batches.  Preserve every pair, seed, transport, control, and output row.
- Actual result: fixed; 40 units completed in minutes with identical estimators.
- Decision: implementation-only acceleration; no partial unit or metric was
  inspected, and Stage-0 hashes must be refrozen again.

## V0.1 — magnitude-only transport repair (scientific repair 1/2)

- Previous failure: alpha=1 overshot target-subject structure on WBCIC and moved
  all candidates too far from the clean centroid manifold.
- Diagnosis: every residual-stability gate passed and SCST retained a large
  positive advantage over norm-matched random transport.  Exact quadratic
  interpolation of source-only distances predicts positive target affinity at
  alpha 0.25 and 0.5 for all four final embeddings.
- Proposed change before run: evaluate the global, prelocked alpha candidates
  `{0.25, 0.5}` with the same two layers, folds, seed, centroids, controls,
  independent class probe, manifold test, and gates.  A single alpha must support
  all four settings; prefer 0.5 if both pass.
- Prediction before run: smaller alpha should retain positive target affinity,
  repair the WBCIC sign, reduce off-manifold displacement, and weakly improve
  class preservation.  Failure ends transport development.
- Actual result: validated `TRANSPORT_OFF_MANIFOLD`; eligible global alphas `[]` and selected alpha `None`.
- Decision: reject the residual transport hypothesis, stop model training, and do not open outer resources.

## V0.2 — source-support-constrained transport (scientific repair 2/2)

- Previous failure: fixed alpha=0.25 retained subject/class fidelity but exceeded
  the unchanged 1.25 clean-manifold ratio on both WBCIC backbones.
- Diagnosis before run: a global step ignores local source support; the low
  WBCIC binary outlier rate but elevated mean 3NN distance is consistent with
  locally unsupported magnitudes rather than an arbitrary direction.
- Proposed change before run: retain the exact residual direction and
  `final_embedding`, cap every step at 0.25, and choose the largest value on the
  fixed 1/64 grid admitted by a Session-1-only same-class 3NN radius.  Session 2
  remains an independent validity test.  Random directions match the realized
  constrained transport norm exactly.
- Prediction before run: WBCIC manifold ratios should fall below 1.25 while
  target-subject affinity remains strictly positive and above matched random.
  Failure of any retained setting ends the transport line permanently.
- Actual result: validated `TRANSPORT_VALIDITY_NOT_SUPPORTED`; 20/20 units
  completed and 2/4 settings passed every gate.  OpenBMI 3NN ratios were
  1.16285 and 1.14937.  WBCIC ratios were 1.30796 and 1.34080, above the frozen
  1.25 maximum, although subject affinity, matched-random advantage, class
  fidelity, and binary off-manifold gates passed in all four settings.
- Decision: stop the constructive transport line permanently.  Do not create
  Repair-3, train SCST, inspect future performance, or open outer resources.
