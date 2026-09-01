# CPV2 engineering repair rationale

## Single permitted repair pass

The first server launch stopped before producing a prediction because the
EEGNet constructor used `16 * width` for its embedding input, although
`width` was already measured after flattening all 16 feature channels.  The
resulting matrix mismatch was:

`mat1 and mat2 shapes cannot be multiplied (192x496 and 7936x64)`.

The repair changes only the input dimension to the probed flattened feature
dimension (`width`).  It does not change the architecture family, objective,
training budget, split salts, seed mapping, calibration, stack, λ grid,
inference unit, or stopping rules.  Local protocol tests remained green after
the repair.

No additional scientific or engineering search is authorized after this pass.
