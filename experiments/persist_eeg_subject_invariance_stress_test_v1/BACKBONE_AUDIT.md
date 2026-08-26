# Backbone audit

EEGNet is the authoritative current OpenBMI baseline: F1=8, depth multiplier 2, F2=16, dropout 0.25, and a 64-dimensional linear/ELU/LayerNorm embedding.

EEGConformer is the already exercised OpenBMI CompactEEGConformer from PERSIST-EEG V7: 40 convolutional tokens, two 4-head Transformer layers, and a 64-dimensional linear/ELU/LayerNorm embedding. The V7 implementation is adapted only at its input boundary so it consumes the same externally train-normalized tensor as EEGNet. Its convolution, pooling, positional embedding, Transformer, channel-drop augmentation, representation, and task head definitions are unchanged.

Both expose `forward_features(x)` and a final linear `head`, which makes the identity and exact finite-decision intervention audit architecture-independent.
