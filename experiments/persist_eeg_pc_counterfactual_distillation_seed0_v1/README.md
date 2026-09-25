# Native EEGNet counterfactual distillation, seed 0

This development experiment converts frozen P/C oracle outputs into training targets for five native EEGNet continuations. Evaluation uses a single native EEGNet forward on OpenBMI MI and SSVEP discovery subjects. No final-heldout arrays are read.

Run from the repository root after providing the canonical V1 refit checkpoints, V3 geometry files, source protocols, and OpenBMI cache:

```bash
python experiments/persist_eeg_pc_counterfactual_distillation_seed0_v1/code/run.py preflight
python experiments/persist_eeg_pc_counterfactual_distillation_seed0_v1/code/run.py cell --task OpenBMI_MI --fold 0
# Repeat cell for folds 0–4 of OpenBMI_MI and OpenBMI_SSVEP.
python experiments/persist_eeg_pc_counterfactual_distillation_seed0_v1/code/run.py aggregate
```

Set `PC_DISTILL_RUNTIME` to store trial teacher caches and epoch-10 checkpoints outside the repository. `outputs/` contains only the protocol, audits, subject results, paired biological-subject contrasts, and report. Source checkpoint and geometry hashes are checked against the frozen actionability protocol before any EEG is loaded. The source checkpoint was fit on non-final source-session subjects, including discovery subjects; this is a cross-session development proof of concept, not independent subject validation.
