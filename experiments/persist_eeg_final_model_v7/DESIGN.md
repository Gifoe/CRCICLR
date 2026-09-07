# V7 development design

## Objective and falsification rule

V7 tested whether legal target-history context can predict the future-session
consequence of a candidate adaptation and whether PERSIST descriptors add
value beyond an information- and capacity-matched generic controller. The
primary performance threshold was approximately +5 balanced-accuracy
percentage points over the strongest fair matched baseline on both OpenBMI and
WBCIC development.

The PERSIST-specific hypothesis required at least one of the following against
META-GENERIC: a statistically credible BA gain, a consistent reduction in
harmful-subject adaptations, or improved worst-subject performance. A positive
utility-prediction metric without downstream benefit was defined as an
intermediate mechanistic signal, not method success.

## Locked information structure

| Benchmark | Subjects | Legal target history | Outcome session | History depth |
|---|---:|---|---|---:|
| OpenBMI | 54 | S1 labels | S2 labels, scoring-only | 1 |
| WBCIC authorized development | 41 | S1/S2 labels | S3 labels, scoring-only | 2 |

Five subject-disjoint folds were inherited from the frozen protocol. Each fold
contains non-overlapping model-fit, discovery, and outcome roles. For
non-outcome subjects, earlier-to-later session episodes provide meta-training
targets. For outcome subjects, future labels only compute final scores and
diagnostic oracles. They never enter within-run fitting or selection.

`outputs/protocol/OUTER_LOCK.json` records that the WBCIC outer split file,
subject identities, raw data, features, logits, and labels were not opened.

## Evaluation sequence

1. Reconstruct V6 and lock its strongest legal anchors: 83.204% OpenBMI and
   82.082% WBCIC development.
2. Construct fold-specific population representations without outcome
   subjects.
3. Build legal history-to-future episodes for non-outcome subjects and measure
   realized CE and BA utility for coarse adaptation components.
4. Cross-fit META-GENERIC and PERSIST controllers by subject. Give both the
   same component one-hot capacity and estimators; expose P/U/D/G/R only to the
   PERSIST controller.
5. Select controller family and risk policy from non-outcome meta-OOF scores,
   refit on all non-outcome episodes, and evaluate once on that fold's outcome
   subjects.
6. Test structurally distinct geometry, backbone, alignment, and hypernetwork
   families when the initial controller fails.
7. Compute paired subject bootstrap intervals and outcome-only oracle headroom
   as diagnostics. The oracle is never used to tune a deployable router.

## Matched controls

META-GENERIC and PERSIST share estimators (Ridge or ExtraTrees), component
identities, history labels, folds, seeds, utility targets, and policy search.
The generic design matrix retains zero-valued PERSIST slots so raw input width
does not create extra capacity for PERSIST. The low-rank hypernetwork similarly
retains zero-valued PERSIST channels for its generic control.

This is a reasonable capacity match, but it is not perfect causal isolation:
the descriptors can change regularization geometry after standardization, and
the one-seed development search is adaptive. Claims are therefore limited to
these implementations.

## Metrics and uncertainty

Primary performance is mean subject balanced accuracy. NLL, subject positivity,
nonnegative fraction, harmful fraction, worst-subject delta, and action fraction
are diagnostics. Final pairwise uncertainty uses 20,000 paired subject-level
bootstrap draws with seed `20260820`.

Utility prediction reports R2, Pearson correlation, Spearman correlation, and
sign accuracy. These rows are correlated across folds and controller families;
their mean is descriptive, not a sampling distribution. The controller's
`predicted_sigma` is an empirical component-wise standard deviation estimated
from meta-training subjects. It is not a calibrated posterior uncertainty.

## Adaptive-development limitation

OpenBMI and WBCIC development outcomes had already been used in V1-V6 and were
observed between V7 structural iterations. The clean discovery/confirmation
split requested in the research brief was therefore not achievable after the
fact. V7 used grouped folds and strict within-run sentinels, but this only
prevents direct leakage; it does not undo researcher adaptivity.

All V7 scores must be labeled exploratory. A valid confirmation would freeze
one algorithm and hyperparameter set before evaluating fresh subjects or the
still-sealed WBCIC outer cohort. V7 is not ready for that freeze because the
PERSIST increment is unresolved and below the strongest generic method.

## Stop decision

The search stopped after nine major structural families. The strongest generic
gain was under 0.6 pp on both benchmarks, PERSIST-specific paired intervals
included zero, several invariance/alignment methods caused large regressions,
and an outcome-only new-expert subject oracle had under 2.1 pp headroom.
Further threshold or mixture search on the same outcomes would be arbitrary
reuse rather than a new structural test. This justifies scientific exhaustion
only within the current one-seed data/cache/backbone scope; it is not evidence
that future-utility adaptation is impossible.
