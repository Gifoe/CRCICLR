# BNLOCK Stage-2 v1

BNLOCK is a preregistered continuation of the frozen carrier pair. It retains
the exact source EEGNet/LiteBN checkpoints and fixed 0.5/0.5 fusion rule, but
locks every LiteBN BatchNorm module in eval mode throughout source-session
training. Its running buffers therefore remain bitwise identical to the source
carrier reference while BatchNorm affine weight/bias remain trainable.

Phase A contains both datasets, five frozen folds, seed 0, and the two matched
objectives `BNLOCK-JOINTCE` and `BNLOCK-NRRF`. Outer development is evaluated
only after all twenty training cells complete. Runtime checkpoints remain
outside Git.
