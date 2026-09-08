# BNLOCK causal intervention

The frozen EEGNet anchor is always eval/no-grad. LiteBN receives gradients
through the existing fusion `0.5 logits_EEGNet + 0.5 logits_LiteBN`, but each
of its BatchNorm modules is forcibly returned to eval mode after every
`model.train()` call. Thus it always normalizes with the original carrier
running mean/variance rather than source-session mini-batch statistics.

The running mean, running variance, and batch-count buffers are checked for
bitwise equality to the source checkpoint after the first forward, first
update, every epoch, EMA loading, and final checkpoint writing. BN affine
parameters remain ordinary trainable parameters. No data split, sampler,
normalizer, alpha, loss coefficient, optimizer, epoch count, or checkpoint
selection rule changes relative to NRRF Phase A.
