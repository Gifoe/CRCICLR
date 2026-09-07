# Minimal repair

R2EEG-v1 has a channel-shared temporal encoder followed by permutation-invariant
mean/std channel aggregation. SpatialRepair retains the temporal encoder,
temporal residual block, latent projections and classifier, but replaces only
the channel aggregation with a grouped full-montage convolution: each of the 48
temporal features learns two electrode-index-specific filters over the 62 fixed
OpenBMI channels. There are no coordinates, attention, graphs, PRD or auxiliary
losses. Training is cross entropy only for the frozen 60-epoch fold-0 schedule.
