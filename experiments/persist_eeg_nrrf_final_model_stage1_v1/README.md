# NRRF-v1 final-model Stage 1

This is a preregistered development-stage test of **No-Regret Residual Fusion
(NRRF-v1)**. It initializes the exact previously selected EEGNet and
CompactLite-BN carrier checkpoints, freezes EEGNet permanently, and trains only
LiteBN for a locked 20-epoch Stage 2.

Phase A is exactly OpenBMI and WBCIC, five frozen folds, seed 0, with two
matched trainable methods: `JOINT-CE` and `NRRF-v1` (20 new runs). The runner
does not evaluate outer-development data until every training cell is complete.
It never accesses V8 internal holdout or WBCIC true outer data.

Runtime checkpoints are stored outside the Git worktree. This experiment
commits code, compact protocol, diagnostics, and result tables only.

Run on the provisioned CUDA server:

```bash
export R2EEG_REPO=/root/rivermind-data/CRCICLR_NRRF_STAGE1_WORK
export CARRIER_5FOLD_RUNTIME=/root/rivermind-data/carrier_5fold_multiseed_stability_runtime
export NRRF_RUNTIME=/root/rivermind-data/nrrf_final_model_stage1_runtime
python experiments/persist_eeg_nrrf_final_model_stage1_v1/code/train_nrrf.py
python experiments/persist_eeg_nrrf_final_model_stage1_v1/code/evaluate_nrrf.py
python experiments/persist_eeg_nrrf_final_model_stage1_v1/code/aggregate_results.py
```

The primary new-model checkpoint is always the EMA parameter state after epoch
20; no inner- or outer-development metric selects an epoch.
