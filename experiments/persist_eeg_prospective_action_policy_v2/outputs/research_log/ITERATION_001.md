# Iteration 001

## Failure diagnosis entering the iteration

V1 showed legal single-run effect regression did not convert association into positive policy gain.

## Hypothesis

Low single-run confidence may identify baseline errors with enough precision to route a disagreeing intervention.

## Implementation

- Policy: `I001_SINGLE_RUN_CONFIDENCE`
- Validation: Five outer subject folds; each uses disjoint inner train and calibration subject folds.
- Thresholds: `[1.1, 1.1, 0.47, 0.45, 0.49]`
- Default action: `KEEP`
- Outcome access: exploration subjects only

## Result

- Subject-balanced Delta BA: `-0.000260`
- Grouped bootstrap 95% CI: `[-0.002896, 0.002125]`
- Rescue AUPRC: `0.429778`
- Harm AUPRC: `0.882639`
- Rescue precision: `0.483622`
- Unsafe intervention rate: `0.516378`
- Action rate: `0.032848`
- Recovered oracle headroom: `-0.002088`
- Positive runs: `0.167`

## Decision

`ABANDON`

Confidence gating did not establish a robust positive lower bound, so test a learned but low-capacity error head.
