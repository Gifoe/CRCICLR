# LiteBN-X single-model architecture ladder

All candidates use one normalized EEG input, one neural network, one post-fusion
representation stream, and one classifier head/checkpoint. No teacher, logits,
distillation, ensemble, routing, mixture of experts, or test-time adaptation is used.

## LiteBN_BASELINE
Exact historical CompactLite(channels, bn): three temporal branches with kernels
15/63/127; 8 temporal then 16 spatial channels per branch; two historical
depthwise-separable temporal blocks 48-to-64-to-64; adaptive 8-bin pooling;
64-d embedding; and one classifier. Only classifier output dimension follows task classes.

## LiteBN-R
The historical stem and both historical temporal blocks are unchanged. The
64-channel post-block sequence receives residual temporal mixers with dilations
1/2/4. Each mixer has depthwise Conv1d(k=9), LayerNorm, F-to-2F-to-F GELU MLP,
dropout 0.10, and gamma initialized to 1e-3. Pooling is mean + standard deviation
+ four adaptive temporal bins (6F), then 96-d GELU/LayerNorm/dropout and one head.

## LiteBN-RG
LiteBN-R plus residual sample-dependent three-branch scale weighting after the
unchanged temporal+spatial branches. Descriptors are time means. The gate scale is
1 + tanh(lambda_scale) * (3 alpha_i - 1), lambda_scale initialized to zero.

## LiteBN-X
LiteBN-RG widened only as specified: temporal branch 8-to-12, spatial branch
16-to-24, concatenate 72, temporal blocks 72-to-96-to-96, mixer F=96, 6F-to-128 embedding.

## LiteBN-XS
LiteBN-X plus shared per-electrode residual gating before temporal stems. Each
electrode maps [RMS, temporal-difference RMS] through shared 2-to-8-to-1 MLP.
Input scaling is 1+tanh(lambda_channel)*tanh(g_c), with lambda_channel initialized
to zero. No channel, subject, session, or dataset identifiers are used.
