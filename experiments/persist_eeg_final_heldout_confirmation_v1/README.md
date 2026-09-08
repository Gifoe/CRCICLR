# PERSIST-EEG final held-out confirmation

This experiment performs inference only for the pre-selected `FROZEN_LOGIT50` predictor: `(EEGNet logits + LiteBN logits) / 2`.

The exact 5-fold x 3-seed carrier checkpoint pairs, source-only fold normalizers, and held-out memberships are locked in `protocol/`. Preflight inspects held-out signal metadata and label paths but never opens label values. One subsequent fixed command opens labels, evaluates all 15 replicate pairs for every held-out subject, and aggregates subjects as the primary statistical unit.

No optimizer, gradient, checkpoint selection, calibration, adaptation, or fusion fitting is present. Runtime/checkpoints/cache and raw signal data remain outside Git.
