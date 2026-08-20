# PERSIST-EEG V7 scientific report

## Decision

Terminal state: `V7_SCIENTIFIC_EXHAUSTION`. `READY_FOR_OUTER_FREEZE=false` and
`OUTER_TEST_USED=false`.

V7 finds a weak PERSIST signal in adaptive-development future-utility
prediction. It
does **not** establish a PERSIST performance advantage over the strongest
generic control and it misses the requested +5 pp-over-matched-baseline target.

## Strongest fair results

| Benchmark | Strongest method | BA | Delta vs V6 strong anchor |
|---|---|---:|---:|
| OpenBMI S1 -> S2 | `ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD` | 83.778% | +0.574 pp |
| WBCIC S1/S2 -> S3 dev | `ANCHOR_BLEND__CONFORMER_NORM_FIXED_HEAD` | 82.497% | +0.415 pp |

The paired subject-bootstrap intervals for those generic gains are
`[-0.630,+1.685]` pp on OpenBMI and
`[-0.195,+1.037]` pp on WBCIC. Neither gain
is statistically resolved.

Both strongest methods are generic fixed anchor/Conformer-history-head blends.
OpenBMI remains more than five points above its old EEGNet reference; WBCIC is
only 3.061 pp above its EEGNet reference.

## PERSIST-specific evidence

Within the capacity-matched initial controller family, PERSIST context changes
the point estimate by +0.093 pp on OpenBMI and
+0.233 pp on WBCIC. These are small and
their paired intervals, `[-0.130,+0.333]` and
`[-0.316,+0.843]` pp, include
zero. They are not sufficient to beat the strongest generic Conformer control.
P/U/D/G/R improve mean utility R2/correlation/sign accuracy in the correlated
fold/controller development summaries, particularly on WBCIC, so a
mechanistic signal exists in this analysis but does not translate into
target-level BA.

## Structural diagnosis

Nine genuinely distinct families were examined. History Euclidean alignment
and class-conditional session alignment hurt, consistent with useful persistent
spatial/session structure being erased. Filter-bank variance and the low-rank
hypernetwork fit training episodes but generalized poorly. Compact Conformer
predictions add modest diversity. Even the outcome-only new-expert subject
oracle reaches only 85.259% on OpenBMI and
83.461% on WBCIC, so another router
cannot reach the target.

## Limits

All development estimates are exploratory: OpenBMI and WBCIC development have
been heavily reused, and observed development outcomes guided later structural
iterations. One seed was used, as requested. No multi-seed robustness
claim is made. Within a run, outcome-subject future labels never fit or select a
model; across runs, observed development outcomes influenced structural
redesign. The WBCIC outer cohort was never opened or enumerated.
