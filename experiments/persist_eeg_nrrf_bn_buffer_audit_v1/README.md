# NRRF BatchNorm buffer audit v1

This is a frozen inference intervention on the completed NRRF-v1 Phase-A
checkpoints. It compares each exact epoch-20 checkpoint in two evaluation-only
states: `CURRENT` and `RESTORE_SOURCE_BN`. The latter keeps every stage-2
parameter bitwise identical and replaces only BatchNorm `running_mean`,
`running_var`, and `num_batches_tracked` with their original LiteBN carrier
values.

There is no training, optimizer, backward pass, EMA update, BatchNorm
adaptation, checkpoint selection, or sealed-holdout access. The primary scope
is seed 0, both datasets, five frozen folds, and the two matched Stage-2
methods. The raw epoch-20 variant is diagnostic only.
