# LiteBN-XNG architecture lock

The authoritative implementation is
`experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/litebn_x.py`.

LiteBN-XNG preserves the exact LiteBN-X wide stem (12 temporal channels and 24
spatial channels per branch), 96-feature temporal backend, residual temporal
mixers at dilations 1/2/4, and 576-to-128 readout with mean, standard deviation,
and four adaptive bins.

The only removed state is:

- `lambda_scale`
- `scale_mlp.0.weight`
- `scale_mlp.0.bias`
- `scale_mlp.2.weight`
- `scale_mlp.2.bias`

`scale_gate` is false, so branch outputs pass directly to
`stem.temporal_blocks`. No channel gate, auxiliary loss, ensemble,
distillation, calibration, BN recalibration, or test-time adaptation is used.

For each fold, an exact authoritative X model is constructed first. Every XNG
state tensor shared with X is copied bitwise, the post-X-construction RNG state
is restored, and the initialization and parameter audits must pass before
training begins.
