# Iteration 002

## Failure diagnosis entering the iteration

The scalar confidence gate could not separate sparse rescue from frequent harm.

## Hypothesis

A regularized baseline-error head can combine single-run confidence, counterfactual movement, and protected geometry.

## Implementation

- Policy: `I002_SINGLE_RUN_LOGISTIC_ERROR`
- Validation: Five outer subject folds; train/calibration/validation subjects are disjoint.
- Thresholds: `[0.55, 0.5, 0.65, 0.475, 0.475]`
- Default action: `KEEP`
- Outcome access: exploration subjects only

## Result

- Subject-balanced Delta BA: `-0.000198`
- Grouped bootstrap 95% CI: `[-0.002438, 0.002198]`
- Rescue AUPRC: `0.421301`
- Harm AUPRC: `0.880880`
- Rescue precision: `0.470882`
- Unsafe intervention rate: `0.529118`
- Action rate: `0.019019`
- Recovered oracle headroom: `-0.001587`
- Positive runs: `0.333`

## Decision

`ABANDON`

The learned head still fails the safety/gain gate; test a legal, label-free source of independent evidence: other frozen runs.
