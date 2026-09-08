# Frozen source-BN intervention

For each stage-2 expressive state `CURRENT` and its original carrier state
`SOURCE`, the audit discovers BatchNorm buffer names from concrete
`torch.nn.BatchNorm` modules. It forms `RESTORE_SOURCE_BN` by copying only
`running_mean`, `running_var`, and `num_batches_tracked` from `SOURCE`.

All parameters, including BatchNorm affine weight and bias, remain the
stage-2 values. EEGNet remains the original frozen anchor and prediction is
always `0.5 * logits_EEGNet + 0.5 * logits_LiteBN`. All models execute in eval
mode under `torch.no_grad`.

Evaluation reuses the carrier five-fold split, its source-session normalizer,
and the same SEARCH outer-development session. It does not access either
sealed holdout. Before interpretation, `CURRENT` must reproduce every Phase-A
dataset/fold/method mean subject BA within `1e-6`.
