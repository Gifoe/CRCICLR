# MI-first Stage-A protocol amendment

This amendment is authorized after the currently running OpenBMI ERP fold-1
cell completes. It changes only the Stage-A execution order and screening gate;
it does not change the model definitions, preprocessing, frozen five-fold
splits, seed 0, optimizer, epoch budget, checkpoint rule, metrics, or any
held-out boundary.

1. Complete the already running OpenBMI ERP fold-1 cell at the current code
   revision, then stop the serial all-task screen at the fold boundary.
2. Run WBCIC MI for all five specified architectures and five frozen folds.
3. Combine the completed OpenBMI MI and WBCIC MI inner-validation results.
4. Always retain `LiteBN_BASELINE`. Retain a candidate for ERP/SSVEP only if
   its two-MI equal mean delta is strictly positive and neither MI-task delta
   is below -0.50 percentage points. If no candidate passes, retain the
   highest two-MI mean candidate diagnostically so the experiment remains
   informative rather than silently selecting a winner.
5. Run OpenBMI ERP and OpenBMI SSVEP only for the retained architectures.

All gates are computed from inner-validation only. No outer-dev, final
held-out, or test labels/predictions are used for this amendment.
