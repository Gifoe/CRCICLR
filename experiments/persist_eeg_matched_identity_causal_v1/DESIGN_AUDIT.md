# Design audit

The upstream V3.1 spectrum is 128-dimensional and is produced by the frozen
EEGNet checkpoint.  The V1.1 representation cache is 64-dimensional and is a
different model/projection; using it with the V3.1 whitening matrices would be
invalid.  Experiment 3 therefore extracts the same V3.1 checkpoint
representation on the server and uses the persisted canonical spectrum,
rather than silently substituting the V1.1 cache.

Protected is the exact union of the MI block IDs in each V3.1
`SIGNED_ASSIGNMENTS_V3_1.json`.  No block is reselected here.  Controls are
formed only from non-overlapping, persistence-supported canonical coordinates.
The matching score was fixed before any validation outcome is read.  It is a
standardized train-only distance over structural and train-probe summaries;
validation BA is never available to this stage.  The initial implementation
also required every candidate feature to have absolute standardized deviation
at most 2.5.  A train-only diagnostic showed that this conjunction accepted no
legal candidate (the closest candidate had joint RMS distance 1.38 but one
correlated feature above that arbitrary cutoff).  That numerical rule was
repaired before freeze: all legal exact-rank candidates are deterministically
sorted by the joint distance and the first 50 are retained.  This preserves
exact rank, no overlap, the non-Protected persistence-supported source, and
the full structural diagnostics, while avoiding an empty set caused by a
redundant per-feature hard cutoff.  The repair used no validation outcomes,
does not select a Protected block, and is recorded in the matching payload.

The proposed experiment's simple full-erasure comparison was replaced with
continuous coordinate suppression and train-only response calibration.  This
removes the obvious confound that P and N may have different identity-removal
strength while preserving the same causal question.  The human-readable
identity accuracy is still reported at every dose.

The loader reads only the frozen train and development-validation subject
fields needed for this experiment.  It does not extract, enumerate, or
materialize outer subject membership, EEG, labels, or features.
