# PERSIST-EEG Experiment 1 — Real Invariance Failure + PERSIST Rescue

This is a development-only mechanism experiment. It is not V9, a final model,
or an absolute-performance search. It asks whether independently trained
subject/domain-invariant MI encoders remove task-useful protected persistence,
and, only when that causal chain is observed, whether a rank-matched protected
residual restores more future-session performance than generic, PCA, or random
residuals.

The frozen roster is controlled EEGNet+GRL, a clean-room EEG-DG
reimplementation, and a clean-room SCLDGN reimplementation. The upstream
repositories contain no license file, so no upstream source is vendored.

OpenBMI MI development folds 0--2 and seeds 0--1 are used. For each fold, only
the frozen `train_subjects` and `validation_subjects` fields are indexed. Outer
membership is not extracted, enumerated, logged, featurized, or scored. The
complete `SPLIT_FREEZE.json` file is SHA-256 hashed as opaque bytes for
provenance, so its uninspected outer-field bytes contribute to that file digest;
no outer signal or label is accessed.

Run on the designated GPU server:

```powershell
$python = 'D:\nips-temp\TotalP\P2\.conda\gpu-baseline-v1\python.exe'
$root = 'D:\nips-temp\TotalP\P1\CRCICLR_INVARIANCE_RESCUE_V1'
& $python "$root\experiments\persist_eeg_invariance_rescue_v1\code\run.py" phase0
& $python "$root\experiments\persist_eeg_invariance_rescue_v1\code\run.py" smoke
& $python "$root\experiments\persist_eeg_invariance_rescue_v1\code\run.py" freeze
& $python "$root\experiments\persist_eeg_invariance_rescue_v1\code\run.py" full
& $python "$root\experiments\persist_eeg_invariance_rescue_v1\code\run.py" audit
& $python "$root\experiments\persist_eeg_invariance_rescue_v1\code\run.py" rescue
& $python "$root\experiments\persist_eeg_invariance_rescue_v1\code\run.py" finalize
```

Large representations and checkpoints remain in `outputs/cache` and
`outputs/checkpoints` on the execution server. Compact ledgers, statistics,
figures, and reports are versioned.

## Frozen execution and result

- Frozen scientific implementation: `873486f7f215ce3b5287d8220d660e06913ea10f`.
- First post-freeze reporting-only repair: `25e818d611c031ac7ed3465542c2c1494aade695`.
- Execution-server artifact root:
  `D:\nips-temp\TotalP\P1\CRCICLR_INVARIANCE_RESCUE_V1\experiments\persist_eeg_invariance_rescue_v1\outputs`.
- Server-only large artifacts: `outputs/cache`, `outputs/checkpoints`, and
  `outputs/smoke`; they can be regenerated with the commands above.
- Terminal state: `NO_ELIGIBLE_PROTECTED_LOSS_OBSERVED`.

No family passed the frozen I1+I2+I3 chain, so running rescue models would have
violated the protocol. Consequently Generic/PERSIST rescue BA, their paired CI,
and recovery ratios are not estimable rather than zero.
