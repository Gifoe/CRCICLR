# PERSIST-EEG Step 1: Cross-Batch Biological-Subject Harm Audit

This seed-0 audit asks whether a prospective certificate computed from four
trials blocks of a biological subject predicts the optimizer-step loss change
on a fifth, disjoint block of the same subject. It replays the committed PSG
V2 task-only AdamW trajectory from canonical EEGNet checkpoints. It is not a
new model, guard, hyperparameter search, or Step-2 experiment.

Datasets are OpenBMI and WBCIC development/source roles only, folds 0--4,
seed 0. WBCIC outer-10 and OpenBMI sealed/confirmation resources are never
opened. Runtime, cache, checkpoint, and raw EEG files remain untracked.

The final terminal is written to `FINAL_REPORT.md`, `FINAL_REPORT.json`, and
`results/VALIDATION.json`. `STEP2_AUTHORIZED = NO` for every terminal.
