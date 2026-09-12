# LiteBN-HE96 seed0 final report

## Decision

`STOP_MODEL`. The first required gate, OpenBMI-MI outer development comparison,
failed: HE96 0.76775 BA vs LiteBN
0.79150 BA (-2.375 pp), with
0/5 positive folds. No later task or additional seed was started.

## Initialization audit

All five folds passed exact initial-function checks in eval/train and FP32/AMP:
zero logit and embedding differences, identical predictions, zero new point2
columns, and preserved RNG state.

## Internal heldout diagnostic

This is an internal heldout diagnostic, not an untouched final test: HE96
0.73200 BA vs LiteBN 0.74486 BA
(-1.286 pp) across 70 subject-fold rows.
