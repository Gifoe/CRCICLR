# Protocol

## Fixed design

- Dataset: WBCIC/NEMAR `nm000348`, frozen 41-subject development pool only.
- S1/ses-0: supervised classifier-head adaptation; within-class chronological final 30% is validation.
- S2/ses-1: historical certificate and pre-S3 reliability features.
- S3/ses-2: outcome definition only, merged after feature/protocol lock.
- Backbones: EEGNet and EEGConformer; matched seeds 0, 1, 2.
- Anchors and adapter: unchanged from Stage-0. Encoder and normalization are frozen; head-only AdamW uses LR 0.001, weight decay 1e-4, maximum 50 epochs.
- Unit of inference: subject. Both backbone rows remain together in each fold and subject bootstrap.

## Analysis

The existing five outcome folds define grouped out-of-subject validation. M0-M8 are frozen in `RELIABILITY_FEATURE_PROTOCOL_LOCK.json`. Continuous predictors are standardized using training folds only. Logistic models use fixed L2 regularization (`C=1`). Identity M4 is explicitly unavailable because the prior frozen probe is a model-fit-domain aggregate, not a target-subject score.

The development policy always requires `Delta2 > 0`. Its M8 threshold is chosen using nested OOF predictions/outcomes inside each outer training split, subject to 20% training coverage. Held-out outcomes never select a threshold.

All Gate A-F criteria and terminal rules are numeric and frozen before S3 association.

