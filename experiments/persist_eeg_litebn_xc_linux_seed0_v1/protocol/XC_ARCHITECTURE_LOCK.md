# LiteBN-XC architecture lock

LiteBN-XC preserves every LiteBN-XNG component and adds only the exact original
LiteBN-XS input channel gate:

- per-electrode RMS and temporal-difference RMS descriptors;
- one MLP shared by every electrode: Linear 2-to-8, GELU, Linear 8-to-1;
- one scalar `lambda_channel`, initialized exactly to zero;
- input scale `1 + tanh(lambda_channel) * tanh(raw)`.

The wide 12/24 multi-scale stem, 96-feature backend, dilation 1/2/4 residual
mixers, and mean/std/four-bin 576-to-128 readout are unchanged. There is no
`scale_mlp`, `lambda_scale`, branch softmax, static branch weighting, channel
identity embedding, or electrode-specific learned parameter.

Expected WBCIC counts are 216,705 parameters for XNG and 216,739 for XC. The
only 34 added parameters belong to `channel_mlp` and `lambda_channel`.
