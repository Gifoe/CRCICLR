# Frozen method

R2EEG consumes only `[channels, 1000]` cached epochs. A shared channel-wise
temporal encoder creates eight-bin channel representations, a shared
mean-centered channel mixer constructs a 96-dimensional temporal summary, and
separate 64-dimensional relation and residual heads form the classifier input.
There is no BatchNorm, transformer attention, graph/electrode prior, adaptation,
or reuse of legacy latent features/checkpoints.

R2EEG-ERM minimizes cross entropy. R2EEG-Rel uses the identical initialization,
episodes, optimizer, normalizer and epoch budget, with only `0.25 * PRD(z_rel)`
added. PRD compares each query subject's class-direction with the normalized
mean of support-subject class-directions. It has no absolute-centroid term.
