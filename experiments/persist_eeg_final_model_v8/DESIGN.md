# V8 design

The protocol fixes subject-only V8_SEARCH/internal-holdout partitions before
model search.  Source-fold non-outcome V8_SEARCH subjects provide legal
history-to-future meta-training episodes.  Source-fold outcome V8_SEARCH
subjects are exploratory Phase-A scoring subjects.  The internal holdout and
WBCIC outer cohort remain sealed.

The research order is deliberately gated: create action headroom, then recover
it with a generic prospective selector, then test PERSIST increment under a
capacity-matched comparison.  A selector is prohibited when the action bank's
subject oracle is below approximately +8 percentage points on either
benchmark.  Candidate competence, error correlation, rescue fractions, and
tail failures are audited to prevent weak lucky experts from manufacturing an
oracle.
