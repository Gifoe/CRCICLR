# Theory note

Consider the linearized model `y = X beta + X b_s + eps`, with zero-mean
subject effects `E[b_s]=0` and independent sampling.  The population least
squares estimator is unbiased in expectation, but a finite set of observed
subjects contributes a nonzero empirical mean `mean_s b_s`; pooled ERM can
therefore absorb that realization into `beta`.  Centered, ridge-shrunk random
effects separate the empirical subject deviations from the shared component
under the stated zero-mean and correct-design assumptions.  This is a
proposition for the linearized model, not a theorem about the neural network.
The empirical experiment tests utility and separation, rather than claiming
that these assumptions hold exactly for EEG.

