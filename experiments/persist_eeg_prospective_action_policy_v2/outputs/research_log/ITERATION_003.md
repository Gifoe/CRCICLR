# Iteration 003

## Failure diagnosis entering the iteration

Single-run observables rank errors but cannot push intervention precision reliably above harm.

## Hypothesis

If the target run disagrees with a leave-target-run majority on the same trial, that disagreement is transferable evidence that a flipping intervention is beneficial.

## Implementation

- Policy: `I003_CROSS_RUN_FULL`
- Validation: Deterministic policy over all 40 exploration subjects; uncertainty is grouped bootstrap over subjects.
- Thresholds: `[0.5]`
- Default action: `KEEP`
- Outcome access: exploration subjects only

## Result

- Subject-balanced Delta BA: `0.013729`
- Grouped bootstrap 95% CI: `[0.010167, 0.017271]`
- Rescue AUPRC: `0.406406`
- Harm AUPRC: `0.816999`
- Rescue precision: `0.584211`
- Unsafe intervention rate: `0.415789`
- Action rate: `0.096203`
- Recovered oracle headroom: `0.110090`
- Positive runs: `1.000`

## Decision

`KEEP_AND_FREEZE`

Every strong-candidate stopping gate was reached. Stop exploration now; retain the protected-safe menu as the safety Pareto comparator.
