# LiteBN Complex Harmonic Former v1 — Stage A Final Report

## Decision

`STOP_CHF_DIRECTION`

The seed-0 OpenBMI SSVEP Stage A screen completed all five frozen folds for F1 and F2. Neither candidate met the continuation rule. Internal-heldout data was not accessed.

## Frozen comparison

| Variant | Mean subject-equal inner-validation BA | Delta vs F0 | Non-negative folds vs F0 |
|---|---:|---:|---:|
| F0: current TFFormer | 0.902667 | 0.000 pp | 5/5 |
| F1: fine complex spectrum | 0.901333 | -0.133 pp | 2/5 |
| F2: F1 plus harmonic mixer | 0.901000 | -0.167 pp | 2/5 |

The required gate was at least +0.20 pp mean gain over F0 and non-negative performance in at least 3/5 folds. The better candidate, F1, failed both conditions.

## Per-fold BA

| Fold | Matched LiteBN | F0 | F1 | F2 |
|---:|---:|---:|---:|---:|
| 0 | 0.938333 | 0.946667 | 0.948333 | 0.948333 |
| 1 | 0.903333 | 0.908333 | 0.911667 | 0.911667 |
| 2 | 0.913333 | 0.933333 | 0.928333 | 0.928333 |
| 3 | 0.865000 | 0.880000 | 0.875000 | 0.873333 |
| 4 | 0.843333 | 0.845000 | 0.843333 | 0.843333 |

## Protocol and acceleration audit

- Effective sampling rate: 250 Hz, inherited as the frozen runtime convention from LiteBN-TFFormer v1.
- F0 checkpoints and inner-fold metrics were reused with checkpoint and normalizer provenance recorded.
- F1 and F2 each ran seed 0 for 60 epochs on five folds.
- Only deterministic, fold-normalized local STFT log-magnitudes and full-trial Hann rFFT values were cached. No learned output was cached.
- Historical BatchNorm affine parameters and running buffers remained frozen and eval-locked.
- Stage A used only canonical inner-train and inner-validation data.
- Internal-heldout was not accessed because the Stage A continuation gate failed.

## Interpretation

The fine-spectrum branch does not provide a robust improvement over TFFormer. Gains in folds 0 and 1 are offset by regressions in folds 2–4. The harmonic mixer adds no measurable benefit and slightly worsens the aggregate result. Continuing to the loss screen or internal-heldout evaluation would violate the frozen stopping rule.
