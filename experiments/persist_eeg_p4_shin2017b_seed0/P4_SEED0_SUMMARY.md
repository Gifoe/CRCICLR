# P4 seed-0 summary

- EEGNet (preserved, not retrained): future BA 0.721, future Macro-F1 0.712, WS-BA 0.621.
- Corrected SIRE-EEG: future BA 0.695, future Macro-F1 0.685, WS-BA 0.583.
- Paired SIRE-EEG minus EEGNet future BA: -0.026 [-0.076, 0.024].
- Paired SIRE-EEG minus EEGNet WS-BA: -0.038 [-0.083, 0.007].

## Frozen diagnostics

- EEGNet: PEEH 1.909 pp, nonempty-fold coverage 2/5; PSWA 8.286363636363637 pp, nonempty-fold coverage 2/5.
- Corrected SIRE-EEG: PEEH 1.205 pp, nonempty-fold coverage 3/5; PSWA -0.5941176470588205 pp, nonempty-fold coverage 3/5.
- Empty Protected contributes exactly zero to PEEH and remains undefined/omitted for PSWA. Coverage is nonempty fold/checkpoint coverage, not outer-subject count.

The corrected SIRE source/configuration/state/architecture audit passed before training; all five corrected folds completed. Diagnostic direction is descriptive and does not change selection, thresholds or coverage.
