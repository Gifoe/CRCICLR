# Frozen StatsResidual protocol

## Scope

- Task: `WBCIC_MI`
- Seed: `0`
- Folds: `0,1,2,3,4`
- Base: recovered historical 64-channel LiteBN selected checkpoint for each fold
- No OpenBMI task, additional seed, ablation, ensemble, adaptation, or sealed-test access

## Frozen base and feature tap

Every LiteBN parameter has `requires_grad=False`, and the complete base remains in evaluation mode. A forward pre-hook on the historical `AdaptiveAvgPool2d((1,8))` captures its input without changing the base forward. This is the 64-channel tensor after the second `k=31` backend block, ELU, average pooling by two, and evaluation-mode dropout.

For each sample, the residual feature is the concatenation of temporal mean and `std(unbiased=False)`, yielding exactly 128 dimensions. The sole new module is a zero-initialized `Linear(128,2)` with 258 trainable parameters. Final logits are `z0 + dz`; epoch 0 is therefore exactly the frozen B0 function.

## Data and selection

The frozen five-fold split, inner-training normalizer, recovered checkpoint, and original manifest provenance are replayed from the historical artifacts. Residual fitting uses inner-training subjects and WBCIC source sessions 0/1. Selection uses only future-session 2 from canonical inner-validation subjects. Outer-development and internal-heldout subjects are excluded from fitting and selection.

Frozen LiteBN logits and temporal statistics are precomputed for residual training and validation. Each fold audits a fixed 17-sample subset from both caches. A `1e-6` absolute tolerance is used for base-logit replay because changing CUDA GEMM batch shape produced at most `9.5367431640625e-7`; cached temporal statistics replayed exactly.

Optimization is fixed: cross entropy plus `1e-3 * (mean(dW^2) + mean(db^2))`, AdamW, learning rate `1e-4`, weight decay `5e-4`, batch size 512, gradient clipping 5, at most 30 epochs, and patience 5. Fold shuffle seeds are `10003 + fold`. Epoch 0 is eligible, strict BA improvement is required, and ties select the earliest epoch.

## Evaluation and decision

Outer results use subject-equal aggregation over the 31 development subjects. The internal-heldout analysis first averages the five fold replicates within each of the 10 real subjects and then averages subjects. Paired bootstraps use 10,000 subject resamples with seed 0.

All epoch-0, feature-cache, subject-manifest, B0 metric-replay, and frozen-state drift checks must pass. The terminal label follows the predeclared strong/weak/tradeoff/no-useful-signal rule in `aggregate_statsresidual.py`.
