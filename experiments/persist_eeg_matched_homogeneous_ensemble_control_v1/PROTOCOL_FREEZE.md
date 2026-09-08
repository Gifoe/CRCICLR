# Protocol freeze: matched homogeneous-ensemble control v1

This file is the protocol lock for the experiment directory
`persist_eeg_matched_homogeneous_ensemble_control_v1`.  Its SHA-256 is written
by the metadata-only preflight and must match before held-out labels are read.

## Immutable scope

The four required blocks are OpenBMI MI, OpenBMI ERP, OpenBMI SSVEP, and WBCIC
MI.  Each uses exactly the historical reserved subject-held-out future-session
cohort, source-only normalizer, five folds, checkpoint-selected frozen
EEGNet/LiteBN checkpoints and seeds `{0,1,2}` from the cited source
experiments.  No new split, trial definition, preprocessing, normalizer,
checkpoint, architecture, training, calibration, or adaptive fusion is
permitted.

For each fold and unordered seed pair `(0,1)`, `(0,2)`, `(1,2)`:

* `EE = 0.5 E_i + 0.5 E_j`
* `LL = 0.5 L_i + 0.5 L_j`
* `EL_A = 0.5 E_i + 0.5 L_j`
* `EL_B = 0.5 E_j + 0.5 L_i`
* `EL = mean(outcome(EL_A), outcome(EL_B))`

`EL` is an arithmetic average of the two two-model outcomes, not a four-model
prediction.  Both orientations and all three pairs are mandatory.  The
primary rule is `LOGIT50`; `PROB50` is a fixed 50/50 sensitivity analysis.

The primary pair-level quantities are `G_best = BA(fusion)-max(BA(A),BA(B))`,
`D_avg = G_EL - 0.5*(G_EE+G_LL)`, and
`D_best = G_EL - max(G_EE,G_LL)`.  The subject is the uncertainty unit:
within each block, each subject's 15 fixed fold/pair values are averaged before
paired bootstrap.  The global analysis bootstraps subjects independently in
each block and then averages the four block means equally.  Bootstrap seed is
`20260909` and repetitions are `10000`.

## Locked integrity and decision rules

Fusion is permitted only after an exact `(dataset, task, subject, session,
fold, seed, trial_index, label_hash, class_order)` ledger proves one-to-one
alignment.  Any missing/duplicate item or checkpoint-hash failure is fatal.

`HETEROGENEOUS_COMPLEMENTARITY_SUPPORTED` requires global equal-block
`D_avg > 0`, a 95% CI wholly above zero, directionally positive `D_avg` in at
least 3/4 blocks, no large reproducible disadvantage (a block `D_avg <= -1.0`
pp with CI wholly below zero), and both orientations retained.  Evidence below
0.25 pp is described as numerically small.  If the primary contrast is
non-positive or within ±0.25 pp with no decisive positive CI, the report must
not claim a heterogeneous-carrier-specific effect; it uses
`GENERIC_ENSEMBLING_NOT_RULED_OUT` unless homogeneous control clearly matches
or exceeds in at least 3/4 blocks, in which case it uses
`HOMOGENEOUS_ENSEMBLING_MATCHES_OR_EXCEEDS`.  Directional but incomplete
positive evidence uses `HETEROGENEOUS_ADVANTAGE_WEAK`.

No result may select an orientation, seed pair, fold, task, or rule.
