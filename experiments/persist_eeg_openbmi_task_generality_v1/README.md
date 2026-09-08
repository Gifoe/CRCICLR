# PERSIST-EEG OpenBMI task-generality test

This preregistered experiment evaluates the previously frozen `FROZEN_LOGIT50`
rule on OpenBMI ERP/P300 and SSVEP. It does not re-run or retune Motor Imagery.

For each task, fresh EEGNet and LiteBN carriers are trained from scratch on the
frozen 40-person SEARCH partition using the original five folds and seeds 0, 1,
and 2. The only fusion is `(EEGNet logits + LiteBN logits) / 2`. The 14-person
OpenBMI held-out set is evaluated only after the complete 60-run training grid
and SEARCH evaluation are complete.

Runtime checkpoints live outside this repository and are never committed.