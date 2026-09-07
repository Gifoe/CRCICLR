# Population-Consensus SDET

SDET uses the exact frozen canonical EEGNet backbone and a rank-4 low-rank latent intervention. Per-support-subject class-balanced gradients are unit-normalized and averaged without a second normalization; a global tanh gate constructs one frozen population adapter. Support/query episodes are subject-disjoint and test subjects are excluded. The internal V8 holdout is opened only after baseline and intervention freezing.
